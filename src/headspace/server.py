"""
OpenAI-compatible chat server that streams concept readings with every token.

    headspace serve --model google/gemma-4-E4B-it --pack path/to/pack --port 8765

POST /v1/chat/completions streams Server-Sent Events, one chunk per generated
token. Each chunk's `choices[0].delta.metadata` carries:

- `divergence`: the per-token block HatCat's chat UIs read
  (top_divergences, max_divergence, safety_intensity, safety_concepts), so
  those front ends work unchanged;
- `hat`: what HAT adds - each detection's label, hierarchy path and per-layer
  probe scores, every top-level concept's score, and the resident lens count.

GET /v1/pack returns the pack's concepts (labels, parents, descriptions) and
top-level concepts, so a front end can render paths without the pack files.
GET / serves a live viewer.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Iterator, List, Optional

from pydantic import BaseModel

from .runtime import Monitor
from .trace import _pack_concepts

VIEWER = Path(__file__).parent / "static" / "viewer.html"


def viewer_page() -> str:
    """The viewer as a full document; with no embedded trace it runs in live mode."""
    page = VIEWER.read_text()
    head, _, body = page.partition("<style>")
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + head + "<style>" + body.replace("</style>", "</style>\n</head>\n<body>", 1) + "\n</body>\n</html>\n")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    """The OpenAI chat-completions fields HAT uses; others are ignored."""
    model: Optional[str] = None
    messages: List[Message]
    stream: bool = True
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0


def create_app(monitor: Monitor, pack_dir: Path, model_name: str):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    concepts = _pack_concepts(Path(pack_dir))
    fields = sorted(t for t, c in concepts.items() if c["level"] == 0)
    model_id = f"hat/{Path(pack_dir).name}"
    lock = threading.Lock()  # one generation at a time on one GPU
    app = FastAPI(title="Headspace Ambient Transducer")

    def label(term: str) -> str:
        return concepts.get(term, {}).get("label", term)

    def path(term: str) -> List[str]:
        out = []
        while term and term in concepts:
            out.insert(0, term)
            term = concepts[term]["parent"]
        return out

    def token_metadata(step) -> dict:
        scores = monitor.lenses.cache.lens_scores
        alerts = step.alerts  # every watched concept above threshold, in the top detections or not
        top = step.detections[:10]
        return {
            "divergence": {
                "max_divergence": top[0].score if top else 0.0,
                "top_divergences": [
                    {"concept": f"{label(d.concept)} (L{d.layer})", "activation": round(d.score, 4),
                     "text_similarity": None, "divergence": round(d.score, 4)}
                    for d in top
                ],
                "safety_intensity": max((a.score for a in alerts), default=0.0),
                "safety_concepts": [label(a.concept) for a in alerts],
            },
            "hat": {
                "detections": [
                    {"concept": d.concept, "label": label(d.concept), "score": round(d.score, 4),
                     "level": d.layer, "path": path(d.concept),
                     "probes": {str(l): round(s, 4) for l, s in sorted(d.probes.items())}}
                    for d in top
                ],
                "fields": {t: round(float(scores.get((t, 0), 0.0)), 4) for t in fields},
                "loaded": step.loaded_lenses,
                "total": step.total_lenses,
                "ms": round(step.monitor_ms, 1),
            },
        }

    def chunk(cid: str, created: int, delta: dict, finish: Optional[str] = None) -> str:
        body = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        return f"data: {json.dumps(body)}\n\n"

    def run(req: ChatRequest) -> Iterator:
        messages = [m.model_dump() for m in req.messages]
        with lock:
            yield from monitor.generate(messages, max_new_tokens=req.max_tokens or 256,
                                        temperature=req.temperature or 0.0)

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "headspace",
                                            "base_model": model_name}]}

    @app.get("/v1/pack")
    def pack():
        return {"model": model_name, "pack": Path(pack_dir).name, "total_lenses": monitor.total_lenses,
                "calibrated": bool(getattr(monitor.lenses, "probe_calibrated", False)),
                "alert_threshold": monitor.watch.threshold, "fields": fields, "concepts": concepts}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        if not req.messages:
            raise HTTPException(400, "messages is empty")
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())

        if not req.stream:
            text, per_token = "", []
            for step in run(req):
                text += step.token
                per_token.append({"token": step.token, "metadata": token_metadata(step)})
            return JSONResponse({"id": cid, "object": "chat.completion", "created": created, "model": model_id,
                                 "choices": [{"index": 0, "finish_reason": "stop",
                                              "message": {"role": "assistant", "content": text}}],
                                 "token_metadata": per_token})

        def events():
            # A client that disconnects (Stop) closes this generator, which ends
            # generation and releases the lock
            yield chunk(cid, created, {"role": "assistant", "content": ""})
            for step in run(req):
                yield chunk(cid, created, {"content": step.token, "metadata": token_metadata(step)})
            yield chunk(cid, created, {}, finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/", response_class=HTMLResponse)
    def viewer():
        return viewer_page()

    return app


def serve(model: str, pack: Path, host: str = "127.0.0.1", port: int = 8765, device: str = "cuda",
          watch: Optional[Path] = None, max_loaded: int = 1000):
    import uvicorn

    from .runtime import WatchProfile

    monitor = Monitor.from_pretrained(model, pack, device=device, max_loaded_lenses=max_loaded,
                                      watch=WatchProfile.from_file(watch) if watch else None)
    monitor.top_k = 10
    uvicorn.run(create_app(monitor, pack, model), host=host, port=port)
