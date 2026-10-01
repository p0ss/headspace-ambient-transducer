"""
Record what a monitored model is thinking about, token by token, as JSON.

A trace holds, for each prompt, every generated token with the concepts that
fired on it (top detections with their hierarchy paths and per-layer probe
scores) and the score of every top-level concept, which stays resident for
the whole generation. It also carries the pack's labels and descriptions, so a
viewer needs nothing else.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .runtime import Monitor


def _pack_concepts(pack_dir: Path) -> Dict[str, dict]:
    """Labels, levels, parents and descriptions from the pack's bundled hierarchy."""
    concepts = {}
    for path in sorted((pack_dir / "hierarchy").glob("layer*.json")):
        data = json.loads(path.read_text())
        for c in data["concepts"]:
            if c.get("definition_only"):
                continue
            concepts[c["sumo_term"]] = {
                "label": c.get("label") or c["sumo_term"],
                "level": data["layer"],
                "parent": (c.get("parent_concepts") or [None])[0],
                "description": c.get("topic_description") or c.get("definition", ""),
            }
    return concepts


def record(
    monitor: Monitor,
    pack_dir: Path,
    prompts: Sequence[dict],
    max_new_tokens: int = 120,
    top: int = 8,
    chat: bool = True,
    model_name: Optional[str] = None,
) -> dict:
    """
    Generate from each prompt and record per-token readings.

    prompts: dicts with "prompt" and optional "title" / "note" (shown by viewers).
    """
    concepts = _pack_concepts(Path(pack_dir))
    top_level = sorted(t for t, c in concepts.items() if c["level"] == 0)
    runs = []
    for item in prompts:
        tokens = []
        start = time.time()
        for step in monitor.generate(item["prompt"], max_new_tokens=max_new_tokens, chat=chat):
            scores = monitor.lenses.cache.lens_scores
            tokens.append({
                "t": step.token,
                "ms": round(step.monitor_ms, 1),
                "loaded": step.loaded_lenses,
                "fields": [round(float(scores.get((t, 0), 0.0)), 3) for t in top_level],
                "top": [
                    [d.concept, round(d.score, 3), {str(l): round(s, 3) for l, s in sorted(d.probes.items())}]
                    for d in step.detections[:top]
                ],
            })
        runs.append({**item, "seconds": round(time.time() - start, 1), "tokens": tokens})
        print(f"  {item.get('title', item['prompt'][:40])}: {len(tokens)} tokens")
    return {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model_name,
        "pack": Path(pack_dir).name,
        "total_lenses": monitor.total_lenses,
        "calibrated": bool(getattr(monitor.lenses, "probe_calibrated", False)),
        "fields": top_level,
        "concepts": concepts,
        "runs": runs,
    }


def load_prompts(path: Path) -> List[dict]:
    """A JSON list of {"prompt", "title"?, "note"?} or a text file with one prompt per line."""
    text = Path(path).read_text()
    if Path(path).suffix == ".json":
        return json.loads(text)
    return [{"prompt": line.strip()} for line in text.splitlines() if line.strip()]
