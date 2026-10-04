"""
headspace run    --model google/gemma-3-4b-pt --pack path/to/pack "prompt"
headspace trace  --model google/gemma-4-E4B-it --pack path/to/pack --prompts prompts.json --output trace.json
headspace serve  --model google/gemma-4-E4B-it --pack path/to/pack --port 8765
headspace pack add-hierarchy path/to/pack path/to/concept_pack/hierarchy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _probes(detection) -> str:
    """Per-model-layer scores of a multi-layer lens, e.g. ' [L7 .12 L19 .91 L30 .64]'."""
    if not detection.probes:
        return ""
    return " [" + " ".join(f"L{layer} {score:.2f}" for layer, score in sorted(detection.probes.items())) + "]"


def _cmd_run(args) -> int:
    from .runtime import Monitor, WatchProfile

    watch = None
    if args.watch:
        watch = WatchProfile.from_file(args.watch, threshold=args.threshold)

    monitor = Monitor.from_pretrained(
        args.model,
        args.pack,
        hierarchy_dir=args.hierarchy,
        device=args.device,
        watch=watch,
        max_loaded_lenses=args.max_loaded,
        ram_mb=args.ram_mb,
    )
    monitor.top_k = args.top_k

    print(f"\n{args.prompt}", end="", flush=True)
    steps = []
    for step in monitor.generate(args.prompt, max_new_tokens=args.max_new_tokens, chat=args.chat):
        steps.append(step)
        print(step.token, end="", flush=True)
    print("\n")

    for step in steps:
        top = step.detections[0] if step.detections else None
        flag = "!" if step.alerts else " "
        line = f"{flag} {step.index:3d} {step.token!r:>14}  "
        line += f"[{step.loaded_lenses:4d}/{step.total_lenses} lenses {step.lens_memory_mb:6.1f}MB {step.monitor_ms:5.1f}ms]  "
        if top:
            line += " → ".join(top.path) + f" ({top.score:.2f}){_probes(top)}"
        print(line)
        for alert in step.alerts:
            print(f"      ALERT {' → '.join(alert.path)} ({alert.score:.2f}){_probes(alert)}")

    peak = max(steps, key=lambda s: s.loaded_lenses) if steps else None
    if peak:
        print(
            f"\nPeak resident: {peak.loaded_lenses} of {peak.total_lenses} lenses "
            f"({peak.loaded_lenses / peak.total_lenses:.1%}), {peak.lens_memory_mb:.1f}MB of lens weights"
        )
    return 0


def _cmd_trace(args) -> int:
    import json

    from .runtime import Monitor
    from .pack import resolve_pack
    from .trace import load_prompts, record

    monitor = Monitor.from_pretrained(args.model, args.pack, device=args.device, max_loaded_lenses=args.max_loaded,
                                      ram_mb=args.ram_mb)
    monitor.top_k = max(args.top, 10)
    trace = record(monitor, resolve_pack(args.pack), load_prompts(args.prompts), max_new_tokens=args.max_new_tokens,
                   top=args.top, chat=not args.raw, model_name=args.model)
    Path(args.output).write_text(json.dumps(trace, separators=(",", ":")))
    print(f"Wrote {sum(len(r['tokens']) for r in trace['runs'])} tokens across {len(trace['runs'])} prompts to {args.output}")
    return 0


def _cmd_serve(args) -> int:
    try:
        from .server import serve
    except ImportError as exc:
        raise SystemExit(f"headspace serve needs the serve extra: pip install 'headspace-ambient-transducer[serve]' ({exc})")
    serve(args.model, args.pack, host=args.host, port=args.port, device=args.device,
          watch=args.watch, max_loaded=args.max_loaded, ram_mb=args.ram_mb)
    return 0


def _cmd_add_hierarchy(args) -> int:
    from .pack import add_hierarchy

    counts = add_hierarchy(args.pack, args.source, overwrite=args.overwrite)
    total = sum(counts.values())
    print(f"Wrote {total} concepts across {len(counts)} layers to {Path(args.pack) / 'hierarchy'}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="headspace", description="Headspace Ambient Transducer")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Generate from a prompt and monitor concepts per token")
    run.add_argument("prompt")
    run.add_argument("--model", required=True)
    run.add_argument("--pack", required=True, type=Path, help="Lens pack directory, or Hugging Face repo id (org/name)")
    run.add_argument("--hierarchy", type=Path, help="Concept hierarchy dir, if the pack has none")
    run.add_argument("--watch", type=Path, help="File of concept names to alert on, one per line")
    run.add_argument("--threshold", type=float, default=None,
                     help="Alert threshold (default 0.99 for probe-calibrated packs, else 0.5)")
    run.add_argument("--device", default="cuda")
    run.add_argument("--max-new-tokens", type=int, default=48)
    run.add_argument("--max-loaded", type=int, default=None, help="Most lenses kept on the GPU, active plus warm (default 1000, or the pack's)")
    run.add_argument("--ram-mb", type=int, default=8192, help="Lens pack held in CPU RAM, MB (0: read from disk)")
    run.add_argument("--top-k", type=int, default=10)
    run.add_argument("--chat", action="store_true", help="Send the prompt through the chat template (instruct models)")
    run.set_defaults(func=_cmd_run)

    trace = sub.add_parser("trace", help="Record per-token concept readings for a set of prompts as JSON")
    trace.add_argument("--model", required=True)
    trace.add_argument("--pack", required=True, type=Path)
    trace.add_argument("--prompts", required=True, type=Path,
                       help='JSON list of {"prompt", "title"?, "note"?}, or a text file with one prompt per line')
    trace.add_argument("--output", required=True, type=Path)
    trace.add_argument("--max-new-tokens", type=int, default=120)
    trace.add_argument("--top", type=int, default=8, help="Detections recorded per token")
    trace.add_argument("--raw", action="store_true", help="Continue the prompt as raw text instead of a chat turn")
    trace.add_argument("--device", default="cuda")
    trace.add_argument("--max-loaded", type=int, default=None, help="Most lenses kept on the GPU, active plus warm (default 1000, or the pack's)")
    trace.add_argument("--ram-mb", type=int, default=8192, help="Lens pack held in CPU RAM, MB (0: read from disk)")
    trace.set_defaults(func=_cmd_trace)

    srv = sub.add_parser("serve", help="Serve an OpenAI-compatible chat API that streams concept readings, plus a live viewer")
    srv.add_argument("--model", required=True)
    srv.add_argument("--pack", required=True, type=Path)
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8765)
    srv.add_argument("--device", default="cuda")
    srv.add_argument("--watch", type=Path, help="Watch profile: concepts that set safety_intensity / alerts")
    srv.add_argument("--max-loaded", type=int, default=None, help="Most lenses kept on the GPU, active plus warm (default 1000, or the pack's)")
    srv.add_argument("--ram-mb", type=int, default=8192, help="Lens pack held in CPU RAM, MB (0: read from disk)")
    srv.set_defaults(func=_cmd_serve)

    pack = sub.add_parser("pack", help="Lens pack utilities")
    pack_sub = pack.add_subparsers(dest="pack_command", required=True)
    add_h = pack_sub.add_parser("add-hierarchy", help="Bundle a concept hierarchy into a lens pack")
    add_h.add_argument("pack", type=Path)
    add_h.add_argument("source", type=Path, help="Concept pack hierarchy dir (layer*.json)")
    add_h.add_argument("--overwrite", action="store_true")
    add_h.set_defaults(func=_cmd_add_hierarchy)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
