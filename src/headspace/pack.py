"""
Lens pack utilities.

A self-contained lens pack carries the concept hierarchy it was trained
against, so it can be run without the concept pack it came from:

    <pack>/
        pack_info.json
        calibration.json          (optional)
        deployment_manifest.json  (optional)
        hierarchy/
            hierarchy.json        child_to_parent / parent_to_children
            layer0.json ...       per-layer concept records
        layer0/*.pt ...           activation lenses
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict

# Fields the runtime reads from each concept record, plus definitions for display.
CONCEPT_FIELDS = (
    "sumo_term",
    "layer",
    "is_category_lens",
    "parent_concepts",
    "category_children",
    "synsets",
    "synset_count",
    "sumo_depth",
    "sumo_definition",
    "definition",
    "related_concepts",
    "equivalent_concepts",
    "label",
    "topic_description",
    "definition_only",
)


def add_hierarchy(pack_dir: Path, source_hierarchy_dir: Path, overwrite: bool = False) -> Dict[str, int]:
    """Copy a slimmed concept hierarchy into `pack_dir/hierarchy/`."""
    pack_dir = Path(pack_dir)
    source = Path(source_hierarchy_dir)
    dest = pack_dir / "hierarchy"

    layer_files = sorted(source.glob("layer*.json"))
    if not layer_files:
        raise FileNotFoundError(f"No layer*.json files in {source}")
    if dest.exists():
        if not overwrite:
            raise FileExistsError(f"{dest} already exists (use overwrite=True to replace)")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    counts = {}
    for layer_file in layer_files:
        with open(layer_file) as f:
            layer_data = json.load(f)
        concepts = [
            {k: c[k] for k in CONCEPT_FIELDS if k in c}
            for c in layer_data["concepts"]
        ]
        with open(dest / layer_file.name, "w") as f:
            json.dump({"layer": layer_data.get("layer"), "concepts": concepts}, f)
        counts[layer_file.stem] = len(concepts)

    if (source / "hierarchy.json").exists():
        shutil.copy2(source / "hierarchy.json", dest / "hierarchy.json")

    return counts
