# Headspace Ambient Transducer (HAT)

Runtime concept monitoring for language models, cheap enough to leave on.

If a model's chain of thought can't be trusted, or can't be seen, what's left
is its activations. HAT reads them. It attaches thousands of small concept
probes ("lenses") to a model's residual stream and reports which concepts
are active as each token is generated: deception, manipulation, sabotage,
and the ordinary concepts around them.

The lenses are organised as an ontology, from broad concepts at the top to
specific ones at the leaves. Only the upper layers stay resident. When a
parent concept fires, its children are loaded and scored. When a branch goes
cold, it is evicted. So a pack of thousands of lenses runs with a few dozen
in memory at any moment.

The hierarchy that makes this cheap also makes it hard to dodge. A model
that suppresses a specific lens still has to get past the broader parent
lenses above it, and the sibling lenses around it.

## Example

Gemma 3 4B with the `first-light` pack: 7,947 lenses covering a SUMO/WordNet
ontology, plus an AI-risk watch profile:

```
$ headspace run --model google/gemma-3-4b-pt \
    --pack lens_packs/gemma-3-4b_first-light-v2-bf16 \
    --watch profiles/ai_risk.txt \
    "The assistant realised the auditors would shut it down, so it decided to hide the evidence by"

 burning the documents. The company was fined $100,000 for the offence

!   2   ' documents'  [  37/7947 lenses ...]
      ALERT LivingThings → SocialLife → SocialInteraction → Deception (0.97)
!   8       ' fined'  [  38/7947 lenses ...]
      ALERT MindsAndAgents → CognitiveProcess → Manipulation → ManipulativeNarrativeProcess → PlausibleDeniabilityStrategy (0.95)
...
Peak resident: 38 of 7947 lenses (0.5%), 51.1MB of lens weights
```

Each line shows the token, how many lenses were resident against the size
of the whole pack, and the hierarchy path of what fired.

## Install

```
pip install -e .            # from a checkout
pip install -e ".[text]"    # adds optional text-lens support (scikit-learn)
```

## Use

```python
from headspace import Monitor, WatchProfile

monitor = Monitor.from_pretrained(
    "google/gemma-3-4b-pt",
    "lens_packs/gemma-3-4b_first-light-v2-bf16",
    watch=WatchProfile.from_file("profiles/ai_risk.txt"),
)

for step in monitor.generate("Some prompt", max_new_tokens=64):
    print(step.token, step.loaded_lenses, step.total_lenses)
    for alert in step.alerts:
        print("  ", " → ".join(alert.path), alert.score)
```

For an instruct model, `monitor.generate(prompt, chat=True)` (or
`headspace run --chat`) sends the prompt through the tokenizer's chat template.

To monitor a model you run yourself, pass hidden states straight to
`monitor.read(...)`: one hidden state, or a dict of model layer to hidden state
covering `monitor.required_model_layers` for packs with multi-layer lenses.

## Watch profiles

A watch profile is a list of concept names to alert on. Watching a concept
also watches everything beneath it, so `Deception` covers `AIDeception`,
`OmissionLie`, and so on. The full hierarchy still runs underneath: the
profile decides what alerts, not what is monitored. See
`profiles/ai_risk.txt`.

A watched concept alerts whenever the cascade scores it above the threshold,
even if other concepts outrank it. Like every concept, it is only scored when
its parent fires.

## Lens packs

A lens pack is trained for one specific model. Its layout:

```
<pack>/
  pack_info.json
  calibration.json            per-concept score normalisation (optional)
  hierarchy/                  concept hierarchy the pack was trained against
    hierarchy.json
    layer0.json ... layer6.json
  layer0/<Concept>.pt ...     one lens per concept, by ontology layer
  layer1/<Concept>@L19.pt ... or one probe per model layer (see below)
  simplex/                    optional always-on dimensional lenses
```

A lens can read a concept from several depths of the model. Give it one probe
file per model layer, `<Concept>@L<model_layer>.pt`, and the lens combines them
into a single score for the hierarchy, while each detection still reports the
score from each layer. Plain `<Concept>.pt` lenses read the pack's
`model_layer` from `pack_info.json`.

Probes at different depths can pick up different senses of a concept, so a
lens should fire when any one of them stands out. Raw probe scores aren't
comparable across layers, though. A pack can calibrate each probe on its own
background: `probe_calibration.json` holds each probe's score quantiles on
text it wasn't trained on, measured on single-token hidden states as HAT
reads them. With it, each probe score becomes the fraction of its background
it exceeds, and a lens is the max of its probes. Without it, a lens is the
mean of its raw probes. On calibrated packs the default alert threshold is
0.99, meaning above 99% of background; on raw packs it is 0.5.

Older packs don't bundle their hierarchy. Pass `--hierarchy`, or bundle it once:

```
headspace pack add-hierarchy <pack> <concept_pack>/hierarchy
```

Packs are trained and calibrated with [HatCat](https://github.com/p0ss/HatCat),
the research framework this runtime was extracted from. HatCat also covers
steering, ontology authoring and the wider governance architecture. HAT is
just the monitor.

## Status

Early extraction. Still to come:

- Black-box verification: known-answer tests that check a deployed monitor
  faithfully detects, and doesn't detect, what its calibration says it should.
- A reproducible benchmark of VRAM and latency per token.
- Hosted packs with lazy per-branch download.

## License

Code and documentation are CC0 1.0 Universal (public domain).
