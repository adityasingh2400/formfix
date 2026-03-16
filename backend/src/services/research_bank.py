from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any

from .. import schemas

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_RESEARCH_BANK_PATH = _DATA_DIR / "research_bank.json"


@lru_cache(maxsize=1)
def _load_bank() -> dict:
    return json.loads(_RESEARCH_BANK_PATH.read_text())


def get_reference_values() -> Dict[str, Any]:
    """Get reference values for comparing user metrics."""
    return _load_bank().get("reference_values", {})


def get_proficient_value(metric_key: str, shot_type: str = "3pt") -> Optional[float]:
    """Get the proficient shooter mean for a given metric."""
    refs = get_reference_values()
    metric_data = refs.get(metric_key, {})
    key = f"proficient_{shot_type}"
    if key in metric_data:
        return metric_data[key].get("mean")
    return None


def get_optimal_range(metric_key: str) -> Optional[tuple]:
    """Get the optimal range for a given metric."""
    refs = get_reference_values()
    metric_data = refs.get(metric_key, {})
    opt_range = metric_data.get("optimal_range")
    if opt_range and len(opt_range) == 2:
        return tuple(opt_range)
    return None


def lookup_deep_dive(key: Optional[str]) -> Optional[schemas.PlaybackDeepDive]:
    if not key:
        return None

    payload = _load_bank()
    entry = payload.get("entries", {}).get(key)
    if not entry:
        return None

    return schemas.PlaybackDeepDive(
        summary=entry.get("summary", ""),
        stats=[],  # Stats are now generated dynamically with relative values
        sources=[
            schemas.ResearchSource(
                label=item.get("label", ""),
                detail=item.get("detail", ""),
                url=item.get("url"),
            )
            for item in entry.get("sources", [])
        ],
    )


def get_specific_findings(key: str) -> list:
    """Get specific research findings for a given key."""
    payload = _load_bank()
    entry = payload.get("entries", {}).get(key, {})
    return entry.get("specific_findings", [])


def get_target_advice(key: str) -> Dict[str, str]:
    """Get target values/advice for a given key."""
    payload = _load_bank()
    entry = payload.get("entries", {}).get(key, {})
    return entry.get("your_target", {})


def get_coaching_insights() -> Dict[str, Any]:
    """Get coaching insights from NBA trainers."""
    return _load_bank().get("coaching_insights", {})


def get_key_statistics() -> Dict[str, Any]:
    """Get key research statistics (effect sizes, correlations, etc.)."""
    return _load_bank().get("key_statistics", {})
