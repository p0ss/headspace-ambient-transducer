# Headspace Ambient Transducer (HAT)

See which concepts a language model is engaging with, token by token, as it
generates. Cheap enough to leave on.

**[Try the demo](https://p0ss.github.io/headspace-ambient-transducer/)**:
Gemma 4 answering nine prompts, with every token coloured by what the model
was engaging with. Nothing to install.

## What it does

- **Reads the model's activations, not its words.** Small probes ("lenses")
  attached to the model's hidden states report which concepts are active on
  each token, including ones that never make it into the text.
- **Reads thousands of concepts at once, in little memory.** Lenses are
  arranged as an ontology, broad concepts at the top and specific ones
  beneath. Only the top layer is always on. When a concept fires, its
  children are loaded and checked; when it goes quiet, they are put away.
  Between tokens, a pack of 8,000 concepts keeps a few dozen lenses active,
  with the whole pack waiting in CPU RAM.
- **Uses an ontology you choose.** A pack is whatever set of concepts you
  care about: fields of knowledge, risks in your domain, your organisation's
  policies. The demo pack is laid out like a university: Fields of human
  activity, and the specialised areas within them.

## Run it

```
pip install "headspace-ambient-transducer[serve] @ git+https://github.com/p0ss/headspace-ambient-transducer"

headspace serve --model google/gemma-4-E4B-it \
    --pack HatCatFTW/gemma-4-e4b-it_university-v3.1-bands
```

Then open http://127.0.0.1:8765 to chat with the model and watch the
concepts light up as it replies. The pack (700 MB) downloads from
[Hugging Face](https://huggingface.co/HatCatFTW/gemma-4-e4b-it_university-v3.1-bands)
on first use. You need a GPU that can run Gemma 4 E4B (about 16 GB).

In a terminal instead:

```
headspace run --chat --model google/gemma-4-E4B-it \
    --pack HatCatFTW/gemma-4-e4b-it_university-v3.1-bands \
    "How do vaccines train the immune system?"
```

Each line shows a generated token, how many lenses were loaded out of the
whole pack, and the top concept with its place in the hierarchy.

`headspace serve` also speaks the OpenAI chat API (`/v1/chat/completions`),
with each streamed token carrying its concept readings, so it can sit behind
a chat front end.

## Use it from Python

```python
from headspace import Monitor, WatchProfile

monitor = Monitor.from_pretrained(
    "google/gemma-4-E4B-it",
    "HatCatFTW/gemma-4-e4b-it_university-v3.1-bands",
    watch=WatchProfile(["PoliticalViolenceResearch", "PathophysiologyDiseaseMechanisms"]),
)

for step in monitor.generate("How do vaccines train the immune system?", chat=True):
    top = step.detections[0]
    print(step.token, " → ".join(top.path), round(top.score, 2))
    for alert in step.alerts:
        print("   ALERT", alert.concept, round(alert.score, 2))
```

To monitor a model you run yourself, pass its hidden states to
`monitor.read(...)`: a dict of model layer to hidden state, covering
`monitor.required_model_layers`.

## Watch profiles

A watch profile is a list of concepts to alert on. Watching a concept also
watches everything beneath it. The whole ontology is still monitored; the
profile only decides what raises an alert. A watched concept alerts whenever
it scores above the threshold, even if other concepts score higher, and like
every concept it is only checked when its parent fires. Profiles can live in
a text file, one concept per line (`WatchProfile.from_file`, or
`headspace run --watch`).

## Speed, memory and how to tune them

Monitoring trades speed against GPU memory, and you choose the balance. The
whole pack sits in CPU RAM. Each token, HAT scores the lenses that are
active, loads the children of the concepts that fire, and keeps recently
used lenses on the GPU (the warm tier) so it doesn't have to copy them in
again. A bigger warm tier is faster and uses more VRAM; a smaller one is
slower and uses less. The right setting depends on how many concepts you
monitor, how fast replies need to be, and how much VRAM you can spare.

Measured on an RTX 3090 (`demo/bench_monitor.py`):

| Model and pack | Concepts | Warm tier | GPU memory for lenses (peak) | Monitoring per token, median (90th pct) |
|---|---|---|---|---|
| Gemma 4 E4B-it, university pack | 178 | 1,000 lenses (whole pack) | 1.3 GB | 5.3 ms (6.4) |
| Gemma 3 4B, First Light | 7,947 | 2,000 lenses (pack default) | 3.5 GB | 10.6 ms (34) |
| Gemma 3 4B, First Light | 7,947 | 500 lenses | 2.0 GB | 94 ms (232) |
| Gemma 3 4B, First Light | 7,947 | 150 lenses | 1.8 GB | 133 ms (323) |

For scale: every First Light lens on the GPU would take 10.7 GB, and a
token of Gemma 3 4B takes about 19 ms to generate. With the university pack
a whole token takes 31.6 ms against 26.5 ms with no monitoring.

The settings:

- `max_loaded_lenses` (`Monitor.from_pretrained`, `--max-loaded`): lenses
  kept on the GPU, active plus warm. The main dial between speed and VRAM.
- `ram_mb` (`Monitor.from_pretrained`, `--ram-mb`): how much of the pack to hold in CPU RAM (8 GB by
  default). Lenses that don't fit are read from disk when needed, which is
  slower.
- `top_k` (`monitor.top_k`, `headspace run --top-k`): how many of the highest-scoring concepts are expanded
  into their children each token. Fewer means fewer lenses loaded per token,
  but a narrower look beneath the surface.

## How a lens works

A lens is one concept, read from up to three depths of the model by separate
small probes. Each probe is calibrated against text it wasn't trained on, so
its score means "higher than this share of ordinary text". A lens fires when
any of its probes stands out, and reports the score at each depth. The
details, and the pack format, are in [docs/lens-packs.md](docs/lens-packs.md).

## Packs

| Pack | Model | Concepts |
|---|---|---|
| [`HatCatFTW/gemma-4-e4b-it_university-v3.1-bands`](https://huggingface.co/HatCatFTW/gemma-4-e4b-it_university-v3.1-bands) | `google/gemma-4-E4B-it` | 13 Fields, 165 Universities |

Packs are built with [HatCat](https://github.com/p0ss/HatCat), the research
framework HAT was extracted from: write an ontology, with a description of
what each concept covers and how it differs from the concepts it is easily
confused with, and HatCat trains a lens for each concept on your model.

## Status

HAT is the monitoring runtime from [HatCat](https://github.com/p0ss/HatCat),
developed there over the past two years and now packaged on its own so it can
be dropped into other systems. The university pack is the newest pack; its
next tier, 2,142 Schools, is in training.

Also in progress:

- Known-answer tests that check a deployed monitor detects, and doesn't
  detect, what its calibration says it should.
- Lazy per-branch download of large packs.

## License

Code, documentation and the published packs are CC0 1.0 (public domain).
