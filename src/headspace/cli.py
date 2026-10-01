"""
headspace run  --model google/gemma-3-4b-pt --pack path/to/pack "prompt"
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
    run.add_argument("--pack", required=True, type=Path)
    run.add_argument("--hierarchy", type=Path, help="Concept hierarchy dir, if the pack has none")
    run.add_argument("--watch", type=Path, help="File of concept names to alert on, one per line")
    run.add_argument("--threshold", type=float, default=None,
                     help="Alert threshold (default 0.99 for probe-calibrated packs, else 0.5)")
    run.add_argument("--device", default="cuda")
    run.add_argument("--max-new-tokens", type=int, default=48)
    run.add_argument("--max-loaded", type=int, default=1000)
    run.add_argument("--top-k", type=int, default=10)
    run.add_argument("--chat", action="store_true", help="Send the prompt through the chat template (instruct models)")
    run.set_defaults(func=_cmd_run)

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
