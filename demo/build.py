"""Embed a trace (from `headspace trace`) in the viewer, giving one self-contained page.

    python demo/build.py demo/traces/gemma-4-e4b-it_university-v3-contrasts.json demo/index.html

--fragment writes the page without <html>/<head> wrapping, for hosts that add their own.
"""
import json
import sys
from pathlib import Path

args = [a for a in sys.argv[1:] if not a.startswith("--")]
fragment = "--fragment" in sys.argv
trace_path, out_path = Path(args[0]), Path(args[1])
viewer = (Path(__file__).parent / "viewer.html").read_text()
trace = json.loads(trace_path.read_text())
# Keep only concepts the page can show: Fields, and anything that fired
used = set(trace["fields"]) | {row[0] for run in trace["runs"] for tok in run["tokens"] for row in tok["top"]}
keep = set()
for term in used:
    while term and term in trace["concepts"] and term not in keep:
        keep.add(term)
        term = trace["concepts"][term]["parent"]
trace["concepts"] = {t: c for t, c in trace["concepts"].items() if t in keep}
page = viewer.replace("/*TRACE*/null", json.dumps(trace, separators=(",", ":")))
if not fragment:
    head, _, body = page.partition("<style>")
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + head + "<style>" + body.replace("</style>", "</style>\n</head>\n<body>", 1) + "\n</body>\n</html>\n")
out_path.write_text(page)
print(f"{out_path}: {out_path.stat().st_size // 1024} KB, {len(trace['runs'])} runs, {len(keep)} concepts")
