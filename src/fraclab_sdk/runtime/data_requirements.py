"""Helpers for runtime dataset delivery requirements."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_raw_drs(input_dir: Path) -> dict[str, Any]:
    """Load the raw input drs.json.

    Returns an empty mapping when drs.json is missing or malformed so runtime reads
    can fall back to raw delivery semantics.
    """
    drs_path = input_dir / "drs.json"
    if not drs_path.exists():
        return {}

    try:
        loaded = json.loads(drs_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    return loaded if isinstance(loaded, dict) else {}


def resolve_dataset_delivery_hz(raw_drs: dict[str, Any], dataset_key: str) -> int | None:
    """Resolve an optional delivery frequency for a dataset from raw drs.json."""
    datasets = raw_drs.get("datasets")
    if not isinstance(datasets, list):
        return None

    for dataset in datasets:
        if not isinstance(dataset, dict) or dataset.get("key") != dataset_key:
            continue
        value = _extract_delivery_hz(dataset)
        if value is None:
            return None
        safe_value = int(value)
        if safe_value <= 0:
            raise ValueError(f"deliveryHz for dataset {dataset_key} must be positive")
        return safe_value

    return None


def _extract_delivery_hz(dataset: dict[str, Any]) -> Any:
    sampling = dataset.get("sampling")
    if isinstance(sampling, dict):
        for key in ("deliveryHz", "delivery_hz", "hz"):
            if sampling.get(key) is not None:
                return sampling.get(key)

    for key in ("deliveryHz", "delivery_hz", "frequencyHz", "usageFrequencyHz"):
        if dataset.get(key) is not None:
            return dataset.get(key)
    return None
