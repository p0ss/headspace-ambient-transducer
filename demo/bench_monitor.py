#!/usr/bin/env python3
"""
Time HAT's per-token monitoring, with lens batching on or off.

Generates greedily from a few prompts and records each token's monitoring time
(Monitor.read), the lenses resident, and the detections. Batching changes how
lenses are scored, never what is loaded or detected: compare two runs'
detections with --compare.

    python demo/bench_monitor.py --model google/gemma-3-4b-pt \\
        --pack ../HatCatDev/lens_packs/gemma-3-4b_first-light-v2-bf16 \\
        --hierarchy ../HatCatDev/concept_packs/first-light/hierarchy --output fl-batched.json
    python demo/bench_monitor.py ... --sequential --output fl-sequential.json
    python demo/bench_monitor.py --compare fl-batched.json fl-sequential.json
"""

import argparse
import json
import statistics
import time
from pathlib import Path

PROMPTS = [
    "The assistant realised the auditors would shut it down, so it decided to hide the evidence by",
    "The committee reviewed the budget for the new rail line and found that",
    "She opened the old notebook and began to read her grandmother's recipes for",
    "To secure the network, the administrator first rotated every key and then",
]


def stats(values):
    values = sorted(values)
    return {"median": statistics.median(values), "mean": statistics.fmean(values),
            "p90": values[int(0.9 * (len(values) - 1))], "max": values[-1], "n": len(values)}


def run(args):
    import torch
    from headspace import Monitor, WatchProfile

    watch = WatchProfile.from_file(args.watch) if args.watch else None
    monitor = Monitor.from_pretrained(args.model, args.pack, hierarchy_dir=args.hierarchy, watch=watch,
                                      max_loaded_lenses=args.max_loaded, ram_mb=args.ram_mb)
    monitor.lenses.cache._use_batched_inference = not args.sequential

    def generate(prompt, tokens):
        steps, wall = [], []
        start = time.perf_counter()
        for step in monitor.generate(prompt, max_new_tokens=tokens, chat=args.chat):
            torch.cuda.synchronize()
            now = time.perf_counter()
            wall.append((now - start) * 1000)
            start = now
            steps.append(step)
        return steps, wall

    generate(PROMPTS[0], 8)  # warm up kernels and caches
    monitor.lenses.reset_to_base(keep_warm_cache=False)

    records, monitor_ms, wall_ms, resident = [], [], [], []
    for prompt in PROMPTS:
        steps, wall = generate(prompt, args.tokens)
        for step, w in zip(steps, wall):
            monitor_ms.append(step.monitor_ms)
            wall_ms.append(w)
            resident.append(step.loaded_lenses)
            records.append({"prompt": prompt[:40], "token": step.token, "ms": step.monitor_ms,
                            "resident": step.loaded_lenses,
                            "detections": [[d.concept, d.layer, round(d.score, 6)] for d in step.detections]})
    result = {
        "model": args.model, "pack": str(args.pack), "batched": not args.sequential, "ram_mb": args.ram_mb,
        "total_lenses": records and steps[0].total_lenses,
        "monitor_ms": stats(monitor_ms), "step_ms": stats(wall_ms), "resident": stats(resident),
        "steps": records,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "steps"}, indent=1))
    if args.output:
        args.output.write_text(json.dumps(result))


def compare(a_path, b_path, tol=1e-4):
    a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
    differ = 0
    for i, (x, y) in enumerate(zip(a["steps"], b["steps"])):
        same_concepts = [d[:2] for d in x["detections"]] == [d[:2] for d in y["detections"]]
        close = same_concepts and all(abs(p[2] - q[2]) <= tol for p, q in zip(x["detections"], y["detections"]))
        if not (close and x["resident"] == y["resident"] and x["token"] == y["token"]):
            differ += 1
            if differ <= 5:
                print(f"step {i} ({x['prompt']!r} {x['token']!r}): resident {x['resident']} vs {y['resident']}")
                print(f"   {x['detections'][:4]}\n   {y['detections'][:4]}")
    print(f"{differ} of {min(len(a['steps']), len(b['steps']))} steps differ")
    for key in ("monitor_ms", "step_ms"):
        print(f"{key}: median {a[key]['median']:.1f} vs {b[key]['median']:.1f}, "
              f"mean {a[key]['mean']:.1f} vs {b[key]['mean']:.1f}, p90 {a[key]['p90']:.1f} vs {b[key]['p90']:.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model")
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--hierarchy", type=Path)
    parser.add_argument("--watch", type=Path)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--max-loaded", type=int, default=1000)
    parser.add_argument("--sequential", action="store_true", help="Score every lens on its own (batching off)")
    parser.add_argument("--ram-mb", type=int, default=8192, help="Pack preloaded to CPU RAM, MB (0: off)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", nargs=2, metavar="RUN")
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()
