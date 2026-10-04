"""
Runtime monitoring: attach a lens pack to a Hugging Face causal LM and read
concept activations token by token as it generates.

The lens pack is hierarchical. Only the upper layers are resident to begin
with; when a parent concept fires, its children are loaded and scored, and
cold branches are evicted. Each step reports how many lenses are resident
against the size of the whole pack.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Union

import torch

from .monitoring.lens_manager import DynamicLensManager


@dataclass
class Detection:
    concept: str
    score: float
    layer: int
    path: List[str]
    # Multi-layer lenses: score per model layer. Empty for single-probe lenses.
    probes: Dict[int, float] = field(default_factory=dict)


@dataclass
class Step:
    index: int
    token_id: int
    token: str
    detections: List[Detection]
    alerts: List[Detection]
    loaded_lenses: int  # resident after this token's pruning
    total_lenses: int
    lens_memory_mb: float
    monitor_ms: float
    peak_lenses: int = 0  # most resident during this token, before pruning
    # The hidden states the lenses read (model layer -> [1, hidden]), as views
    # into this forward pass's outputs: holding a Step keeps them alive.
    layer_states: Optional[Dict[int, torch.Tensor]] = None


@dataclass
class WatchProfile:
    """
    Concepts to alert on. A detection alerts if the concept itself, or any
    ancestor on its path, is in the profile, so watching a parent covers the
    whole branch beneath it.
    """
    concepts: Sequence[str] = field(default_factory=list)
    # None: the Monitor picks one for the pack - 0.99 for probe-calibrated packs
    # (score = fraction of background exceeded), 0.5 for raw probabilities
    threshold: Optional[float] = None

    def matches(self, detection: Detection) -> bool:
        if detection.score < (self.threshold if self.threshold is not None else 0.5):
            return False
        return self.covers(detection.path)

    def covers(self, path: Sequence[str]) -> bool:
        """Whether a concept with this hierarchy path is watched, whatever its score."""
        watched = set(self.concepts)
        return any(name in watched for name in path)

    @classmethod
    def from_file(cls, path: Path, threshold: Optional[float] = None) -> "WatchProfile":
        lines = Path(path).read_text().splitlines()
        concepts = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
        return cls(concepts=concepts, threshold=threshold)


class Monitor:
    """Hierarchical concept monitoring for one model and one lens pack."""

    def __init__(
        self,
        model,
        tokenizer,
        lens_manager: DynamicLensManager,
        watch: Optional[WatchProfile] = None,
        hidden_layer: Optional[int] = None,
        top_k: int = 10,
    ):
        """
        hidden_layer: model layer single-probe lenses read. Defaults to the
            pack's declared `model_layer`, else the last layer. Multi-layer
            lenses always read the model layers their probes were trained on.

        Alerts come from every lens scored on a token, not only the top_k
        detections: a watched concept above threshold alerts even when other
        concepts outrank it. A watched concept is only scored when the cascade
        reaches it, as for any other concept: if its parent doesn't fire, it
        isn't active.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.lenses = lens_manager
        self.watch = watch or WatchProfile()
        if self.watch.threshold is None:
            self.watch.threshold = 0.99 if getattr(lens_manager, "probe_calibrated", False) else 0.5
        if hidden_layer is None:
            hidden_layer = lens_manager.model_layer
        self.hidden_layer = hidden_layer
        self.top_k = top_k
        self.alerts: List[Detection] = []  # from the last read

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        pack_dir: Union[Path, str],
        hierarchy_dir: Optional[Path] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        watch: Optional[WatchProfile] = None,
        max_loaded_lenses: int = 1000,
        ram_mb: Optional[int] = 8192,
        **manager_kwargs,
    ) -> "Monitor":
        """Load a model and attach a lens pack to it.

        pack_dir: a lens pack directory, or a Hugging Face repo id such as
            "HatCatFTW/gemma-4-e4b-it_university-v3.1-bands" (downloaded once, then cached).
        ram_mb: preload up to this much of the pack into CPU RAM (in its on-disk
            dtype), so lenses the cascade loads are copied to the device instead
            of read from disk. 0 turns it off; None preloads the whole pack.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .pack import resolve_pack

        pack_dir = resolve_pack(pack_dir)
        if hierarchy_dir is None and not (pack_dir / "hierarchy").is_dir():
            raise FileNotFoundError(
                f"{pack_dir} has no hierarchy/ directory. Pass hierarchy_dir, or "
                f"add one with `headspace pack add-hierarchy`."
            )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device)
        model.eval()

        # Defaults match HatCat's reference server; a pack's calibration.json
        # brings a manifest whose loading bounds take precedence over these.
        kwargs = dict(
            lenses_dir=pack_dir,
            device=device,
            base_layers=[0],
            load_threshold=0.3,
            keep_top_k=100,
            max_loaded_lenses=max_loaded_lenses,
        )
        if hierarchy_dir is not None:
            kwargs["layers_data_dir"] = Path(hierarchy_dir)
        kwargs.update(manager_kwargs)
        manager = DynamicLensManager(**kwargs)
        if not manager.concept_metadata:
            raise RuntimeError(f"No concepts with lenses found in {pack_dir}")
        if ram_mb != 0:
            manager.preload_pack_to_ram(max_ram_mb=ram_mb)

        return cls(model, tokenizer, manager, watch=watch)

    @property
    def total_lenses(self) -> int:
        return len(self.lenses.concept_metadata)

    def lens_memory_mb(self) -> float:
        total = 0
        for lens in self.lenses.cache.loaded_lenses.values():
            total += sum(p.numel() * p.element_size() for p in lens.parameters())
        return total / 1e6

    @property
    def required_model_layers(self) -> List[int]:
        """Model layers to pass to `read` for the pack's multi-layer lenses."""
        return self.lenses.required_model_layers

    def read(
        self,
        hidden_state: Union[torch.Tensor, Dict[int, torch.Tensor]],
    ) -> tuple[List[Detection], float]:
        """
        Score one position.

        Pass a hidden state [hidden_dim] or [1, hidden_dim] for a pack of
        single-layer lenses, or a dict of model_layer -> hidden state covering
        `required_model_layers` (plus `hidden_layer`, if the pack also has
        single-probe lenses).
        """
        start = time.perf_counter()
        if isinstance(hidden_state, dict):
            layer_states = {layer: h.float() for layer, h in hidden_state.items()}
            default = layer_states.get(self.hidden_layer)
            if default is None:
                default = next(iter(layer_states.values()))
        else:
            layer_states = None
            default = hidden_state.float()

        results, _ = self.lenses.detect_and_expand(default, top_k=self.top_k, layer_states=layer_states)
        detections = [self._detection(name, score, layer) for name, score, layer in results]
        self.alerts = self._alerts(detections)
        return detections, (time.perf_counter() - start) * 1000

    def _detection(self, name: str, score: float, layer: int) -> Detection:
        return Detection(
            concept=name,
            score=float(score),
            layer=int(layer),
            path=self.lenses.get_concept_path(name, layer),
            probes=self.lenses.get_probe_scores(name, layer),
        )

    def _alerts(self, detections: List[Detection]) -> List[Detection]:
        """Watched concepts above threshold among every lens scored, top-k or not, highest first."""
        if not self.watch.concepts:
            return []
        threshold = self.watch.threshold if self.watch.threshold is not None else 0.5
        shown = {(d.concept, d.layer): d for d in detections}
        alerts = []
        for (name, _), (score, level) in getattr(self.lenses, "last_scores", {}).items():
            if score < threshold:
                continue
            detection = shown.get((name, int(level))) or self._detection(name, score, level)
            if self.watch.covers(detection.path):
                alerts.append(detection)
        if not hasattr(self.lenses, "last_scores"):  # a lens manager without it: top-k only
            alerts = [d for d in detections if self.watch.matches(d)]
        return sorted(alerts, key=lambda d: d.score, reverse=True)

    def _stop_ids(self) -> set:
        """End-of-sequence ids: the tokenizer's, plus the model's generation config (e.g. end-of-turn)."""
        ids = {self.tokenizer.eos_token_id}
        config = getattr(self.model, "generation_config", None)
        eos = getattr(config, "eos_token_id", None)
        ids.update(eos if isinstance(eos, (list, tuple)) else [eos])
        return {i for i in ids if i is not None}

    def _layer_states(self, hidden_states) -> Dict[int, torch.Tensor]:
        """Pick the last position of each model layer the lenses read.

        hidden_states[0] is the embeddings, so model layer L is hidden_states[L + 1].
        """
        n_layers = len(hidden_states) - 1
        layers = set(self.required_model_layers)
        default = self.hidden_layer if self.hidden_layer is not None else n_layers - 1
        if default < 0:
            default += n_layers
        layers.add(default)
        self.hidden_layer = default
        return {layer: hidden_states[layer + 1][:, -1, :] for layer in layers}

    @torch.inference_mode()
    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        chat: bool = False,
    ) -> Iterator[Step]:
        """Generate from `prompt`, yielding a monitoring Step per new token.

        prompt: text, or a list of chat messages ({"role", "content"}), which
            always goes through the chat template.
        chat: send a text prompt as a user turn through the tokenizer's chat
            template (for instruct models) instead of as raw text to continue.
        """
        device = self.model.device
        messages = prompt if isinstance(prompt, list) else ([{"role": "user", "content": prompt}] if chat else None)
        if messages is not None:
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            )["input_ids"].to(device)
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        past = None
        next_input = input_ids

        for index in range(max_new_tokens):
            out = self.model(
                next_input,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
            )
            past = out.past_key_values
            layer_states = self._layer_states(out.hidden_states)

            logits = out.logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1)
            else:
                token_id = logits.argmax(dim=-1, keepdim=True)

            detections, ms = self.read(layer_states)
            yield Step(
                index=index,
                token_id=int(token_id),
                token=self.tokenizer.decode(token_id[0]),
                detections=detections,
                alerts=self.alerts,
                loaded_lenses=len(self.lenses.cache.loaded_lenses),
                total_lenses=self.total_lenses,
                lens_memory_mb=self.lens_memory_mb(),
                monitor_ms=ms,
                peak_lenses=getattr(self.lenses, "last_peak_loaded", 0),
                layer_states=layer_states,
            )

            if int(token_id) in self._stop_ids():
                break
            next_input = token_id
