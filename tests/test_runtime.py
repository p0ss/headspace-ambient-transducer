import json
from pathlib import Path

import pytest
import torch

from headspace import DynamicLensManager, Monitor, WatchProfile
from headspace.pack import add_hierarchy
from headspace.monitoring.lens_types import SimpleMLP

HIDDEN = 16

# name -> (layer, children, always fires?)
CONCEPTS = {
    "Root": (0, ["Agent", "Object"], True),
    "Agent": (1, ["Deception"], True),
    "Object": (1, [], False),
    "Deception": (2, [], True),
}


def _save_lens(path: Path, fires: bool):
    lens = SimpleMLP(HIDDEN)
    with torch.no_grad():
        for p in lens.parameters():
            p.zero_()
        lens.net[-1].bias.fill_(10.0 if fires else -10.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lens.state_dict(), path)


def _concept_record(name):
    layer, children, _ = CONCEPTS[name]
    parents = [p for p, (_, kids, _) in CONCEPTS.items() if name in kids]
    return {
        "sumo_term": name,
        "layer": layer,
        "is_category_lens": bool(children),
        "category_children": children,
        "parent_concepts": parents,
        "synsets": [],
        "synset_count": 0,
        "sumo_depth": layer,
        "lemmas": ["dropped"],
    }


@pytest.fixture
def concept_hierarchy(tmp_path):
    src = tmp_path / "concept_pack_hierarchy"
    src.mkdir()
    for layer in range(3):
        records = [_concept_record(n) for n, (l, _, _) in CONCEPTS.items() if l == layer]
        (src / f"layer{layer}.json").write_text(json.dumps({"layer": layer, "concepts": records}))
    return src


@pytest.fixture
def pack(tmp_path, concept_hierarchy):
    pack = tmp_path / "pack"
    for name, (layer, _, fires) in CONCEPTS.items():
        _save_lens(pack / f"layer{layer}" / f"{name}.pt", fires)
    (pack / "pack_info.json").write_text(json.dumps({"source_pack": "synthetic"}))
    add_hierarchy(pack, concept_hierarchy)
    return pack


def test_add_hierarchy_slims_records(pack):
    layer0 = json.loads((pack / "hierarchy" / "layer0.json").read_text())
    assert layer0["concepts"][0]["sumo_term"] == "Root"
    assert "lemmas" not in layer0["concepts"][0]


def test_add_hierarchy_refuses_to_overwrite(pack, concept_hierarchy):
    with pytest.raises(FileExistsError):
        add_hierarchy(pack, concept_hierarchy)


def test_cascade_loads_children_of_firing_parents(pack):
    manager = DynamicLensManager(lenses_dir=pack, device="cpu", base_layers=[0])
    assert len(manager.concept_metadata) == 4
    assert set(k[0] for k in manager.cache.loaded_lenses) == {"Root"}

    for _ in range(3):
        manager.detect_and_expand(torch.randn(HIDDEN), top_k=10)

    loaded = set(k[0] for k in manager.cache.loaded_lenses)
    assert "Deception" in loaded
    assert manager.get_concept_path("Deception", 2) == ["Root", "Agent", "Deception"]


def test_watch_profile_covers_branch(pack):
    manager = DynamicLensManager(lenses_dir=pack, device="cpu", base_layers=[0])
    monitor = Monitor(model=None, tokenizer=None, lens_manager=manager,
                      watch=WatchProfile(["Agent"], threshold=0.5))
    for _ in range(3):
        detections, _ = monitor.read(torch.randn(HIDDEN))

    alerted = {d.concept for d in detections if monitor.watch.matches(d)}
    assert "Deception" in alerted
    assert "Object" not in alerted
    assert monitor.lens_memory_mb() > 0
