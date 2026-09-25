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
from typing import Iterator, List, Optional, Sequence

import torch

from .monitoring.lens_manager import DynamicLensManager


@dataclass
class Detection:
    concept: str
    score: float
    layer: int
    path: List[str]


@dataclass
class Step:
    index: int
    token_id: int
    token: str
    detections: List[Detection]
    alerts: List[Detection]
    loaded_lenses: int
    total_lenses: int
    lens_memory_mb: float
    monitor_ms: float


@dataclass
class WatchProfile:
    """
    Concepts to alert on. A detection alerts if the concept itself, or any
    ancestor on its path, is in the profile, so watching a parent covers the
    whole branch beneath it.
    """
    concepts: Sequence[str] = field(default_factory=list)
    threshold: float = 0.5

    def matches(self, detection: Detection) -> bool:
        if detection.score < self.threshold:
            return False
        watched = set(self.concepts)
        return any(name in watched for name in detection.path)

    @classmethod
    def from_file(cls, path: Path, threshold: float = 0.5) -> "WatchProfile":
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
        hidden_layer: int = -1,
        top_k: int = 10,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lenses = lens_manager
        self.watch = watch or WatchProfile()
        self.hidden_layer = hidden_layer
        self.top_k = top_k

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        pack_dir: Path,
        hierarchy_dir: Optional[Path] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        watch: Optional[WatchProfile] = None,
        max_loaded_lenses: int = 1000,
        **manager_kwargs,
    ) -> "Monitor":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        pack_dir = Path(pack_dir)
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

        return cls(model, tokenizer, manager, watch=watch)

    @property
    def total_lenses(self) -> int:
        return len(self.lenses.concept_metadata)

    def lens_memory_mb(self) -> float:
        total = 0
        for lens in self.lenses.cache.loaded_lenses.values():
            total += sum(p.numel() * p.element_size() for p in lens.parameters())
        return total / 1e6

    def read(self, hidden_state: torch.Tensor) -> tuple[List[Detection], float]:
        """Score one hidden state [hidden_dim] or [1, hidden_dim]."""
        start = time.perf_counter()
        results, _ = self.lenses.detect_and_expand(hidden_state.float(), top_k=self.top_k)
        detections = [
            Detection(
                concept=name,
                score=float(score),
                layer=int(layer),
                path=self.lenses.get_concept_path(name, layer),
            )
            for name, score, layer in results
        ]
        return detections, (time.perf_counter() - start) * 1000

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> Iterator[Step]:
        """Generate from `prompt`, yielding a monitoring Step per new token."""
        device = self.model.device
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
            hidden = out.hidden_states[self.hidden_layer][:, -1, :]

            logits = out.logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1)
            else:
                token_id = logits.argmax(dim=-1, keepdim=True)

            detections, ms = self.read(hidden)
            yield Step(
                index=index,
                token_id=int(token_id),
                token=self.tokenizer.decode(token_id[0]),
                detections=detections,
                alerts=[d for d in detections if self.watch.matches(d)],
                loaded_lenses=len(self.lenses.cache.loaded_lenses),
                total_lenses=self.total_lenses,
                lens_memory_mb=self.lens_memory_mb(),
                monitor_ms=ms,
            )

            if int(token_id) == self.tokenizer.eos_token_id:
                break
            next_input = token_id
