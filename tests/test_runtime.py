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


# --- Multi-layer lenses -----------------------------------------------------
#
# Deception gets one probe per model layer. Each probe fires on +e0 and stays
# quiet on -e0, so feeding different hidden states per layer shows which layer
# each probe actually read.

PROBE_LAYERS = (3, 7)


def _save_direction_probe(path: Path):
    lens = SimpleMLP(HIDDEN)
    with torch.no_grad():
        for p in lens.parameters():
            p.zero_()
        lens.net[0].weight[0, 0] = 1.0
        lens.net[3].weight[0, 0] = 1.0
        lens.net[6].weight[0, 0] = 20.0
        lens.net[6].bias.fill_(-10.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(lens.state_dict(), path)


def _state(sign: float) -> torch.Tensor:
    h = torch.zeros(HIDDEN)
    h[0] = sign
    return h


@pytest.fixture
def multi_layer_pack(tmp_path, concept_hierarchy):
    pack = tmp_path / "multi"
    for name, (layer, _, fires) in CONCEPTS.items():
        if name == "Deception":
            for model_layer in PROBE_LAYERS:
                _save_direction_probe(pack / f"layer{layer}" / f"{name}@L{model_layer}.pt")
        else:
            _save_lens(pack / f"layer{layer}" / f"{name}.pt", fires)
    (pack / "pack_info.json").write_text(json.dumps({"source_pack": "synthetic", "model_layer": 5}))
    add_hierarchy(pack, concept_hierarchy)
    return pack


def _deception(detections):
    return next(d for d in detections if d.concept == "Deception")


def test_parse_probe_filename():
    from headspace.monitoring.lens_types import parse_probe_filename
    assert parse_probe_filename("Economics@L19") == ("Economics", 19)
    assert parse_probe_filename("Economics") == ("Economics", None)
    assert parse_probe_filename("Economics_classifier") == ("Economics", None)
    assert parse_probe_filename("Odd@Lname") == ("Odd@Lname", None)


def test_multi_layer_lens_reads_each_probe_from_its_own_layer(multi_layer_pack):
    manager = DynamicLensManager(lenses_dir=multi_layer_pack, device="cpu", base_layers=[0])
    monitor = Monitor(model=None, tokenizer=None, lens_manager=manager)
    assert monitor.hidden_layer == 5
    assert monitor.required_model_layers == list(PROBE_LAYERS)

    def read(l3, l7):
        states = {5: torch.randn(HIDDEN), 3: _state(l3), 7: _state(l7)}
        for _ in range(3):
            detections, _ = monitor.read(states)
        return _deception(detections)

    early = read(+1, -1)
    assert early.probes[3] > 0.9 and early.probes[7] < 0.1
    # uncalibrated probes aren't comparable across layers, so the lens takes their mean
    assert early.score == pytest.approx(sum(early.probes.values()) / 2, abs=1e-3)

    late = read(-1, +1)
    assert late.probes[3] < 0.1 and late.probes[7] > 0.9

    manager.detect_and_expand(torch.randn(HIDDEN), layer_states={3: _state(-1), 7: _state(-1)})
    raw = manager.cache.lens_scores[("Deception", 2)]
    assert raw < 0.1


def test_multi_layer_lenses_are_fused_and_ram_cached(multi_layer_pack):
    from headspace.monitoring.lens_types import Lens
    states = {3: _state(1), 7: _state(-1)}

    def scores(preload):
        manager = DynamicLensManager(lenses_dir=multi_layer_pack, device="cpu", base_layers=[0])
        if preload:
            manager.preload_pack_to_ram()
            assert set(manager.cache.tepid_cache[("Deception", 2)]) == set(PROBE_LAYERS)
        for _ in range(3):
            manager.detect_and_expand(torch.randn(HIDDEN), layer_states=states)
        assert isinstance(manager.cache.loaded_lenses[("Deception", 2)], Lens)
        assert ("Deception", 2) in manager.cache.get_fused_bank().keys
        return manager.cache.lens_scores[("Deception", 2)], manager.cache.stats["tepid_hits"]

    (from_disk, _), (from_ram, tepid_hits) = scores(False), scores(True)
    assert tepid_hits > 0
    assert from_ram == pytest.approx(from_disk)


def test_generate_feeds_model_layers_to_lenses(multi_layer_pack):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(vocab_size=32, hidden_size=HIDDEN, intermediate_size=32,
                         num_hidden_layers=8, num_attention_heads=2, num_key_value_heads=2)
    model = LlamaForCausalLM(config).eval()
    model.generation_config.eos_token_id = None  # a random model mustn't stop early on a default EOS id

    class Tokenizer:
        eos_token_id = None

        def __call__(self, text, return_tensors=None):
            return type("Enc", (), {"input_ids": torch.tensor([[1, 2, 3]])})()

        def decode(self, ids):
            return f"<{int(ids[0])}>"

    manager = DynamicLensManager(lenses_dir=multi_layer_pack, device="cpu", base_layers=[0])
    monitor = Monitor(model, Tokenizer(), manager)

    seen = []
    original = manager.detect_and_expand

    def spy(hidden_state, **kwargs):
        seen.append(set(kwargs["layer_states"]))
        return original(hidden_state, **kwargs)

    manager.detect_and_expand = spy
    steps = list(monitor.generate("prompt", max_new_tokens=4))

    assert len(steps) == 4
    assert all(layers == {3, 5, 7} for layers in seen)
    assert set(steps[0].layer_states) == {3, 5, 7}
    assert all(s.peak_lenses >= s.loaded_lenses for s in steps)
    assert any(d.probes for s in steps for d in s.detections if d.concept == "Deception")


def test_calibrated_lens_is_max_of_probe_percentiles(multi_layer_pack):
    # Background quantiles: each probe's raw scores on unrelated text sit near 0,
    # so a probe firing at ~1 exceeds all of its background
    quantiles = [0.0, 0.001, 0.01, 0.05, 0.2]
    calibration = {"method": "percentile", "probes": {
        f"layer2/Deception@L{layer}": {"model_layer": layer, "quantiles": quantiles} for layer in PROBE_LAYERS}}
    (multi_layer_pack / "probe_calibration.json").write_text(json.dumps(calibration))

    manager = DynamicLensManager(lenses_dir=multi_layer_pack, device="cpu", base_layers=[0])
    monitor = Monitor(model=None, tokenizer=None, lens_manager=manager,
                      watch=WatchProfile(["Agent"]))
    assert manager.probe_calibrated
    assert monitor.watch.threshold == 0.99  # "above 99% of background" for calibrated packs

    states = {5: torch.randn(HIDDEN), 3: _state(+1), 7: _state(-1)}
    for _ in range(3):
        detections, _ = monitor.read(states)
    d = _deception(detections)
    assert d.probes[3] == pytest.approx(1.0)       # fires above all of its background
    assert d.probes[7] < 0.3                        # ~0 raw sits inside its background
    assert d.score == pytest.approx(1.0, abs=1e-3)  # max of calibrated probes
    assert monitor.watch.matches(d)


def test_trace_records_tokens_fields_and_concepts(multi_layer_pack):
    from transformers import LlamaConfig, LlamaForCausalLM
    from headspace.trace import record

    config = LlamaConfig(vocab_size=32, hidden_size=HIDDEN, intermediate_size=32,
                         num_hidden_layers=8, num_attention_heads=2, num_key_value_heads=2)
    model = LlamaForCausalLM(config).eval()
    model.generation_config.eos_token_id = None  # a random model mustn't stop early on a default EOS id

    class Tokenizer:
        eos_token_id = None

        def __call__(self, text, return_tensors=None):
            return type("Enc", (), {"input_ids": torch.tensor([[1, 2, 3]])})()

        def decode(self, ids):
            return f"<{int(ids[0])}>"

    manager = DynamicLensManager(lenses_dir=multi_layer_pack, device="cpu", base_layers=[0])
    trace = record(Monitor(model, Tokenizer(), manager), multi_layer_pack,
                   [{"prompt": "hello", "title": "Greeting"}], max_new_tokens=3, chat=False)

    assert trace["fields"] == ["Root"]
    assert trace["concepts"]["Deception"]["parent"] == "Agent"
    run = trace["runs"][0]
    assert run["title"] == "Greeting" and len(run["tokens"]) == 3
    token = run["tokens"][0]
    assert len(token["fields"]) == 1 and token["top"]
    assert all(set(row[2]) <= {"3", "7"} for row in token["top"])  # per-layer scores keyed by model layer


def test_server_streams_tokens_with_concept_metadata(multi_layer_pack):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from transformers import LlamaConfig, LlamaForCausalLM
    from headspace.server import create_app

    config = LlamaConfig(vocab_size=32, hidden_size=HIDDEN, intermediate_size=32,
                         num_hidden_layers=8, num_attention_heads=2, num_key_value_heads=2)
    model = LlamaForCausalLM(config).eval()
    model.generation_config.eos_token_id = None

    class Tokenizer:
        eos_token_id = None

        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def decode(self, ids):
            return f"<{int(ids[0])}>"

    manager = DynamicLensManager(lenses_dir=multi_layer_pack, device="cpu", base_layers=[0])
    client = TestClient(create_app(Monitor(model, Tokenizer(), manager), multi_layer_pack, "tiny-llama"))

    assert client.get("/v1/models").json()["data"][0]["id"] == "hat/multi"
    pack = client.get("/v1/pack").json()
    assert pack["fields"] == ["Root"] and pack["concepts"]["Deception"]["parent"] == "Agent"
    assert "What the model is thinking about" in client.get("/").text

    body = {"messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": 3}
    with client.stream("POST", "/v1/chat/completions", json=body) as res:
        lines = [l for l in res.iter_lines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[6:]) for l in lines[:-1]]
    tokens = [c for c in chunks if "metadata" in c["choices"][0]["delta"]]
    assert len(tokens) == 3
    meta = tokens[0]["choices"][0]["delta"]["metadata"]
    assert set(meta) == {"divergence", "hat"}
    assert meta["divergence"]["top_divergences"] and "safety_intensity" in meta["divergence"]
    assert set(meta["hat"]["fields"]) == {"Root"} and meta["hat"]["detections"][0]["path"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    full = client.post("/v1/chat/completions", json={**body, "stream": False}).json()
    assert len(full["token_metadata"]) == 3 and full["choices"][0]["message"]["content"]
