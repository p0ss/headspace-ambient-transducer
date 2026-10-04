#!/usr/bin/env python3
"""
Lens Types and Basic Classes

Core types used throughout the lens management system:
- LensRole: Enum for different lens purposes
- SimpleMLP: The MLP classifier architecture
- SimplexBinding: Configuration for simplex-concept bindings
- ConceptMetadata: Metadata for a single concept
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn


class LensRole(Enum):
    """Roles for different lens types in the monitoring system."""
    CONCEPT = "concept"        # Hierarchical discrimination vs siblings
    SIMPLEX = "simplex"        # Intensity tracking relative to baseline (tripole)
    BEHAVIORAL = "behavioral"  # Pattern detection (e.g., deception markers)
    CATEGORY = "category"      # Domain/layer markers (layer 0 style)


class SimpleMLP(nn.Module):
    """Simple MLP classifier matching SUMO training architecture."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dtype: torch.dtype = None, layer_norm: bool = False):
        """
        Args:
            input_dim: Input feature dimension (model hidden_dim)
            hidden_dim: MLP hidden layer dimension
            dtype: Parameter dtype. If None, uses default (float32).
                   Use torch.bfloat16 for memory-efficient inference.
            layer_norm: If True, include LayerNorm at input (matches new training arch)
        """
        super().__init__()
        self.has_layer_norm = layer_norm

        # Keep 'net' name for backward compatibility with saved lenses
        layers = []
        if layer_norm:
            layers.append(nn.LayerNorm(input_dim, dtype=dtype))
        layers.extend([
            nn.Linear(input_dim, hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1, dtype=dtype),
        ])
        self.net = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_logits=False):
        """
        Forward pass.

        Args:
            x: Input tensor
            return_logits: If True, return (probability, logit) tuple

        Returns:
            If return_logits=False: probability [0,1]
            If return_logits=True: (probability, logit) tuple
        """
        logits = self.net(x).squeeze(-1)
        probs = self.sigmoid(logits)

        if return_logits:
            return probs, logits
        return probs


def detect_layer_norm(state_dict: dict) -> bool:
    """Detect if state_dict has LayerNorm at input (1D weight vs 2D)."""
    first_key = "net.0.weight" if "net.0.weight" in state_dict else "0.weight"
    if first_key in state_dict:
        return len(state_dict[first_key].shape) == 1
    return False


def empty_mlp(input_dim: int, device, layer_norm: bool = False) -> SimpleMLP:
    """A SimpleMLP allocated on `device` with uninitialised weights, to load a state dict into.

    Building one the ordinary way initialises every weight on the CPU and copies
    it to the device, only for load_state_dict to overwrite it.
    """
    with torch.device("meta"):
        lens = SimpleMLP(input_dim, layer_norm=layer_norm)
    return lens.to_empty(device=device).eval()


def create_lens_from_state_dict(state_dict: dict, hidden_dim: int, device: str) -> SimpleMLP:
    """Create SimpleMLP matching the state_dict architecture."""
    has_ln = detect_layer_norm(state_dict)
    lens = empty_mlp(hidden_dim, device, layer_norm=has_ln)

    # Handle missing net. prefix
    if "0.weight" in state_dict and "net.0.weight" not in state_dict:
        new_state_dict = {f"net.{k}": v for k, v in state_dict.items()}
        state_dict = new_state_dict

    lens.load_state_dict(state_dict)
    return lens


class Lens(nn.Module):
    """
    One concept read from several model layers.

    Each probe reads the hidden state of its own model layer; the lens combines
    them into the single score the hierarchy runs on. Per-probe scores from the
    last forward pass are kept in `last_probe_scores` so callers can see where
    in depth the concept showed up.

    With per-probe calibration (quantiles of each probe's scores on background
    text, from the lens pack's probe_calibration.json), every probe score becomes
    the fraction of background it exceeds, and the lens is the max of those: it
    fires when any layer's evidence stands out against that layer's own
    background. Without calibration, raw probe scores aren't comparable across
    layers, so the lens takes their mean.

    The input is a dict of model_layer -> hidden state [1, hidden_dim], already
    normalised the same way as single-layer lenses.
    """

    has_layer_norm = False

    def __init__(self, probes: Dict[int, nn.Module], calibration: Optional[Dict[int, torch.Tensor]] = None):
        super().__init__()
        self.model_layers = sorted(probes)
        self.probes = nn.ModuleDict({str(layer): probes[layer] for layer in self.model_layers})
        self.calibrated = bool(calibration) and all(layer in calibration for layer in self.model_layers)
        if self.calibrated:
            for layer in self.model_layers:
                self.register_buffer(f"quantiles_{layer}", calibration[layer].float())
        self.last_probe_scores: Dict[int, float] = {}

    def _percentile(self, prob: torch.Tensor, layer: int) -> torch.Tensor:
        """Fraction of this probe's background that scored below `prob` (linear between quantiles)."""
        q = getattr(self, f"quantiles_{layer}").to(prob.device)
        k = q.numel()
        idx = torch.searchsorted(q, prob.float().contiguous()).clamp(1, k - 1)
        lo, hi = q[idx - 1], q[idx]
        frac = torch.where(hi > lo, (prob.float() - lo) / (hi - lo), torch.zeros_like(lo)).clamp(0, 1)
        pct = (idx - 1 + frac) / (k - 1)
        return torch.where(prob.float() <= q[0], torch.zeros_like(pct), torch.where(prob.float() >= q[-1], torch.ones_like(pct), pct))

    def forward(self, layer_states: Dict[int, torch.Tensor], return_logits: bool = False):
        if not isinstance(layer_states, dict):
            raise TypeError(
                f"Lens reads model layers {self.model_layers}; pass layer_states "
                f"(model_layer -> hidden state) to the lens manager"
            )
        scores = []
        for layer in self.model_layers:
            prob = self.probes[str(layer)](layer_states[layer])
            scores.append(self._percentile(prob, layer) if self.calibrated else prob.float())
        scores = torch.stack(scores)
        self.last_probe_scores = {
            layer: float(s) for layer, s in zip(self.model_layers, scores.reshape(len(scores), -1)[:, 0])
        }

        prob = scores.max(dim=0).values if self.calibrated else scores.mean(dim=0)
        if return_logits:
            return prob, torch.logit(prob.clamp(1e-6, 1 - 1e-6))
        return prob


def parse_probe_filename(stem: str) -> Tuple[str, Optional[int]]:
    """
    Split a lens file stem into (concept, model_layer).

    `Economics@L19` reads model layer 19. `Economics` and
    `Economics_classifier` carry no model layer and read the pack default.
    """
    if "@L" in stem:
        term, _, layer = stem.rpartition("@L")
        if layer.isdigit():
            return term, int(layer)
    if stem.endswith("_classifier"):
        stem = stem[: -len("_classifier")]
    return stem, None


@dataclass
class SimplexBinding:
    """Configuration for a simplex bound to a concept."""
    simplex_term: str
    always_on: bool = False
    poles: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=lambda: {
        'baseline_window': 100,
        'alert_threshold': 2.0,
        'trend_window': 500
    })


class LensPolarity(Enum):
    """Polarity for polar lens probes."""
    DEFAULT = "default"    # Single probe (traditional)
    POSITIVE = "positive"  # Positive pole of bipolar concept
    NEGATIVE = "negative"  # Negative pole of bipolar concept


@dataclass
class ConceptMetadata:
    """
    Metadata for a single concept lens.

    Terminology:
    - layer: The transformer model layer where activations are extracted (e.g., 17)
    - level: The ontological/hierarchical abstraction level (e.g., L1=1, L2=2, L3=3 for polar;
             or SUMO layers 0-6 where 0=broad, 6=specific)

    For legacy SUMO lenses, layer == level (same value used for both).
    For polar lenses, layer is the model layer (17) and level is the ontological level (1-3).
    """
    sumo_term: str
    layer: int  # Model layer where probe extracts activations
    category_children: List[str] = field(default_factory=list)
    parent_concepts: List[str] = field(default_factory=list)
    synset_count: int = 0
    sumo_depth: int = 0

    # Ontological level - hierarchical abstraction level
    # For polar: 1=L1, 2=L2, 3=L3
    # For SUMO: same as layer (0-6)
    # None means not specified (defaults to layer for backward compat)
    level: Optional[int] = None

    # Role and simplex binding (new in MAP Meld Protocol)
    role: LensRole = LensRole.CONCEPT
    simplex_binding: Optional[SimplexBinding] = None
    domain: Optional[str] = None

    # Lens paths (set by manager)
    activation_lens_path: Optional[Path] = None
    text_lens_path: Optional[Path] = None
    simplex_lens_path: Optional[Path] = None
    has_text_lens: bool = False
    has_activation_lens: bool = False
    has_simplex_lens: bool = False

    # Polar lens support - multiple probes per concept with different polarities
    # Keys are LensPolarity values: "positive", "negative"
    # Falls back to activation_lens_path if polar_lenses is empty (backward compat)
    polar_lenses: Dict[str, Path] = field(default_factory=dict)

    # Multi-layer lens support - one probe per model layer, keyed by model layer.
    # Empty for single-probe lenses, which read the pack's default model layer.
    probe_paths: Dict[int, Path] = field(default_factory=dict)

    @property
    def has_polar_lenses(self) -> bool:
        """True if concept has multiple polarity probes."""
        return len(self.polar_lenses) > 1

    @property
    def is_polar(self) -> bool:
        """True if this is a polar concept (has positive and negative)."""
        return "positive" in self.polar_lenses and "negative" in self.polar_lenses

    @property
    def ontological_level(self) -> int:
        """Get ontological level, defaulting to layer for backward compat."""
        return self.level if self.level is not None else self.layer


__all__ = [
    "LensRole",
    "LensPolarity",
    "SimpleMLP",
    "Lens",
    "parse_probe_filename",
    "SimplexBinding",
    "ConceptMetadata",
    "detect_layer_norm",
    "create_lens_from_state_dict",
    "empty_mlp",
]
