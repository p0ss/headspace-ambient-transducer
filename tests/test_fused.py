"""Batched (fused) lens scoring must equal scoring each lens on its own."""
import json
from pathlib import Path

import pytest
import torch

from headspace import DynamicLensManager, Monitor
from headspace.monitoring.lens_batched import FusedLensBank
from headspace.monitoring.lens_types import Lens, SimpleMLP

HIDDEN = 24
LAYERS = (2, 5, 9)


def _random_probe(seed: int, layer_norm: bool = False) -> SimpleMLP:
    torch.manual_seed(seed)
    probe = SimpleMLP(HIDDEN, layer_norm=layer_norm).eval()
    with torch.no_grad():
        for p in probe.parameters():
            p.normal_(0, 0.5)
    return probe


def _quantiles(seed: int, k: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.sort(torch.rand(k, generator=g)).values


def _mixed_lenses():
    """Single-layer, LayerNorm, raw multi-layer, calibrated multi-layer, and mixed quantile counts."""
    lenses = {}
    for i in range(4):
        lenses[(f"Single{i}", 0)] = _random_probe(i)
    lenses[("WithNorm", 0)] = _random_probe(10, layer_norm=True)
    for i in range(3):
        lenses[(f"Raw{i}", 1)] = Lens({l: _random_probe(100 + 10 * i + l) for l in LAYERS[: i + 1]}).eval()
    for i in range(3):
        probes = {l: _random_probe(200 + 10 * i + l) for l in LAYERS}
        k = 11 if i < 2 else 7
        lenses[(f"Calibrated{i}", 2)] = Lens(probes, {l: _quantiles(300 + 10 * i + l, k) for l in LAYERS}).eval()
    return lenses


def _states(seed: int, batch: int = 1):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(batch, HIDDEN, generator=g)
    return hidden, {l: torch.randn(batch, HIDDEN, generator=g) for l in LAYERS}


def test_fused_scores_equal_sequential():
    lenses = _mixed_lenses()
    bank, leftover = FusedLensBank.build(lenses)
    assert not leftover and len(bank) == len(lenses)

    for seed in range(5):
        hidden, layer_states = _states(seed)
        scores, logits = bank.score(hidden, layer_states)
        fused_probes = {k: dict(l.last_probe_scores) for k, l in lenses.items() if isinstance(l, Lens)}
        for key, lens in lenses.items():
            x = layer_states if isinstance(lens, Lens) else hidden
            prob, logit = lens(x, return_logits=True)
            assert scores[key] == pytest.approx(prob.item(), abs=1e-5), key
            assert logits[key] == pytest.approx(logit.item(), abs=1e-4), key
            if isinstance(lens, Lens):
                assert fused_probes[key] == pytest.approx(lens.last_probe_scores, abs=1e-5), key


def test_fused_scores_whole_batch():
    lenses = _mixed_lenses()
    bank, _ = FusedLensBank.build(lenses)
    hidden, layer_states = _states(7, batch=4)
    scores, _, _ = bank.score_tensors(hidden, layer_states)
    for i, (key, lens) in enumerate(lenses.items()):
        x = layer_states if isinstance(lens, Lens) else hidden
        assert torch.allclose(scores[i], lens(x).reshape(-1).float(), atol=1e-5), key


def test_unfusable_lenses_are_left_over():
    class Other(torch.nn.Module):
        def forward(self, x):
            return x.sum(-1)

    bank, leftover = FusedLensBank.build({("Odd", 0): Other(), ("Fine", 0): _random_probe(1)})
    assert bank.keys == [("Fine", 0)] and list(leftover) == [("Odd", 0)]
    assert FusedLensBank.build({}) == (None, {})


# --- The cascade: batched and sequential runs must load, prune and score alike ---

BRANCHING = (4, 3, 3)  # roots, children per root, grandchildren per child


def _tree():
    """name -> (hierarchy layer, children)."""
    tree = {}
    for r in range(BRANCHING[0]):
        root = f"R{r}"
        tree[root] = (0, [f"{root}C{c}" for c in range(BRANCHING[1])])
        for c in range(BRANCHING[1]):
            child = f"{root}C{c}"
            tree[child] = (1, [f"{child}G{g}" for g in range(BRANCHING[2])])
            for g in range(BRANCHING[2]):
                tree[f"{child}G{g}"] = (2, [])
    return tree


@pytest.fixture
def random_pack(tmp_path):
    """Roots are single-layer lenses; deeper concepts are calibrated three-probe lenses."""
    tree, pack = _tree(), tmp_path / "pack"
    calibration = {}
    for i, (name, (layer, _)) in enumerate(sorted(tree.items())):
        if layer == 0:
            path = pack / f"layer{layer}" / f"{name}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(_random_probe(i).state_dict(), path)
            continue
        for l in LAYERS:
            path = pack / f"layer{layer}" / f"{name}@L{l}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(_random_probe(1000 + 10 * i + l).state_dict(), path)
            calibration[f"layer{layer}/{name}@L{l}"] = {"model_layer": l,
                                                       "quantiles": _quantiles(i * 10 + l, 21).tolist()}
    (pack / "probe_calibration.json").write_text(json.dumps({"method": "percentile", "probes": calibration}))
    (pack / "pack_info.json").write_text(json.dumps({"source_pack": "synthetic"}))
    hierarchy = pack / "hierarchy"
    hierarchy.mkdir()
    for layer in range(3):
        records = [{"sumo_term": n, "layer": l, "category_children": kids,
                    "parent_concepts": [p for p, (_, k) in tree.items() if n in k]}
                   for n, (l, kids) in tree.items() if l == layer]
        (hierarchy / f"layer{layer}.json").write_text(json.dumps({"layer": layer, "concepts": records}))
    return pack


def _run(pack: Path, batched: bool, tokens: int = 12):
    manager = DynamicLensManager(lenses_dir=pack, device="cpu", base_layers=[0], keep_top_k=6)
    manager.cache._use_batched_inference = batched
    monitor = Monitor(model=None, tokenizer=None, lens_manager=manager, top_k=5, hidden_layer=-1)
    history = []
    for t in range(tokens):
        hidden, layer_states = _states(100 + t)
        detections, _ = monitor.read({**{l: s[0] for l, s in layer_states.items()}, -1: hidden[0]})
        history.append({
            "loaded": sorted(manager.cache.loaded_lenses),
            "warm": sorted(manager.cache.warm_cache),
            "detections": [(d.concept, d.layer, d.score, dict(d.probes)) for d in detections],
        })
    return history


def test_cascade_batched_equals_sequential(random_pack):
    batched, sequential = _run(random_pack, True), _run(random_pack, False)
    assert any(len(step["loaded"]) > BRANCHING[0] for step in batched)  # children were loaded
    for t, (b, s) in enumerate(zip(batched, sequential)):
        assert b["loaded"] == s["loaded"], t
        assert b["warm"] == s["warm"], t
        assert [d[:2] for d in b["detections"]] == [d[:2] for d in s["detections"]], t
        for db, ds in zip(b["detections"], s["detections"]):
            assert db[2] == pytest.approx(ds[2], abs=1e-5), (t, db[0])
            assert db[3] == pytest.approx(ds[3], abs=1e-5), (t, db[0])


# --- Alerts come from every lens scored, not only the top-k ----------------------

def test_watched_concept_alerts_outside_the_top_k(random_pack):
    from headspace import WatchProfile
    manager = DynamicLensManager(lenses_dir=random_pack, device="cpu", base_layers=[0], keep_top_k=6)
    monitor = Monitor(model=None, tokenizer=None, lens_manager=manager, top_k=1, hidden_layer=-1,
                      watch=WatchProfile([f"R{r}" for r in range(BRANCHING[0])], threshold=0.0))
    for t in range(4):
        hidden, layer_states = _states(200 + t)
        detections, _ = monitor.read({**{l: s[0] for l, s in layer_states.items()}, -1: hidden[0]})
        assert len(detections) == 1
        alerted = {a.concept for a in monitor.alerts}
        assert {f"R{r}" for r in range(BRANCHING[0])} <= alerted, t  # every root was scored, so all alert
        assert set(manager.last_scores) >= set(manager.cache.loaded_lenses), t
        assert manager.last_peak_loaded >= len(manager.cache.loaded_lenses)
