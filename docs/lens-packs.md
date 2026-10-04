# Lens packs

A lens pack is trained for one specific model. Its layout:

```
<pack>/
  pack_info.json
  probe_calibration.json      per-probe background quantiles (optional)
  calibration.json            per-concept score normalisation (optional, older packs)
  hierarchy/                  concept hierarchy the pack was trained against
    hierarchy.json
    layer0.json ... layerN.json
  layer0/<Concept>@L19.pt ... one probe per model layer, by ontology layer
  layer1/<Concept>.pt ...     or one probe per concept
  simplex/                    optional always-on dimensional lenses
```

`--pack` and `Monitor.from_pretrained` take a pack directory or a Hugging Face
repo id (`org/name`), which is downloaded once and cached.

## Lenses and probes

A lens is one concept. It can read that concept from several depths of the
model: give it one probe file per model layer, `<Concept>@L<model_layer>.pt`,
and the lens combines them into a single score for the hierarchy, while each
detection still reports the score from each layer. Plain `<Concept>.pt` lenses
read the pack's `model_layer` from `pack_info.json`.

## Calibration

Probes at different depths can pick up different senses of a concept, so a
lens should fire when any one of them stands out. Raw probe scores aren't
comparable across layers, though. A pack can calibrate each probe on its own
background: `probe_calibration.json` holds each probe's score quantiles on
text it wasn't trained on, measured on single-token hidden states as HAT
reads them. With it, each probe score becomes the fraction of its background
it exceeds, and a lens is the max of its probes. Without it, a lens is the
mean of its raw probes. On calibrated packs the default alert threshold is
0.99, meaning above 99% of background; on raw packs it is 0.5.

## The hierarchy at runtime

The top layer of the hierarchy stays resident. On each token, HAT scores the
resident lenses; the children of the highest-scoring concepts are loaded and
scored in turn, and lenses that go quiet are moved out. A concept is only
scored when its parent fires, so a concept that wasn't scored was not active
as far as the hierarchy could tell.

Lenses move through three tiers: resident on the GPU (scored every token),
warm on the GPU (kept for quick reuse), and the pack's weights in CPU RAM
(`Monitor.from_pretrained(..., ram_mb=8192)` by default), so loading a lens is
a copy rather than a read from disk. Resident lenses are scored together in a
few batched operations.

## Older packs

Older packs don't bundle their hierarchy. Pass `--hierarchy`, or bundle it once:

```
headspace pack add-hierarchy <pack> <concept_pack>/hierarchy
```

## Building a pack

Packs are trained and calibrated with [HatCat](https://github.com/p0ss/HatCat):
an ontology of concepts (each with a description of what it covers, and
"differs because" contrasts against the concepts it is easily confused with)
becomes a concept pack, and HatCat trains a lens per concept on a given model.
