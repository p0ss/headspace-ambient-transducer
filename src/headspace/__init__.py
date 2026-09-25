"""
Headspace Ambient Transducer (HAT)

Runtime concept monitoring for language models. A lens pack of thousands of
hierarchically organised concept probes is attached to a model's activations;
only the branches that are firing are kept resident, so broad ontological
coverage fits in a small VRAM budget.
"""

from .monitoring.lens_manager import DynamicLensManager
from .runtime import Detection, Monitor, Step, WatchProfile

__version__ = "0.1.0"

__all__ = ["DynamicLensManager", "Detection", "Monitor", "Step", "WatchProfile"]
