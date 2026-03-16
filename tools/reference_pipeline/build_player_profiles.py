#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.src.services import comparison  # noqa: E402

FEATURE_KEYS = list(comparison.REFERENCE_FEATURE_KEYS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build player-level reference profiles from exported FormFix feature rows."
    )
    parser.add_argument("--input", required=True, help="Input JSONL from export_features.py")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument(
        "--min-clips",
        type=int,
        default=6,
        help="Minimum clips required to create a player profile",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def grouped_rows(rows: List[Dict[str, object]]) -> Dict[tuple[str, str], List[Dict[str, object]]]:
    grouped: Dict[tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata", {}) or {}
        player_id = metadata.get("player_id") or metadata.get("player_name")
        shot_type = str(row.get("shot_type") or "unknown")
        if not player_id:
            continue
        grouped[(str(player_id), shot_type)].append(row)
    return grouped


def build_targets(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    targets: Dict[str, Dict[str, float]] = {}
    for feature_key in FEATURE_KEYS:
        values = [
            float(row["features"][feature_key])
            for row in rows
            if row["features"].get(feature_key) is not None
        ]
        if not values:
            continue
        tolerance = max(float(np.std(values)) * 1.2, 0.08 if "duration" in feature_key or "height" in feature_key else 10.0)
        targets[feature_key] = {
            "value": round(float(np.median(values)), 4),
            "tolerance": round(tolerance, 4),
            "weight": round(comparison._FEATURE_SPECS[feature_key].weight, 3),  # type: ignore[attr-defined]
        }
    return targets


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.input))
    grouped = grouped_rows(rows)
    profiles: List[Dict[str, object]] = []

    for (player_id, shot_type), player_rows in sorted(grouped.items()):
        if len(player_rows) < args.min_clips:
            continue

        metadata = player_rows[0].get("metadata", {}) or {}
        player_name = metadata.get("player_name") or player_id
        targets = build_targets(player_rows)
        if len(targets) < 4:
            continue

        profiles.append(
            {
                "id": f"player_{player_id}_{shot_type}",
                "label": str(player_name),
                "category": "player",
                "source": "trained",
                "shot_types": [shot_type],
                "summary": (
                    f"Player profile trained from {len(player_rows)} labeled {shot_type.replace('_', ' ')} clips."
                ),
                "description": (
                    f"Use this to compare a user's shot against {player_name}'s measured clip set instead of a generic archetype."
                ),
                "style_tags": [
                    "player profile",
                    shot_type.replace("_", " "),
                    f"{len(player_rows)} clips",
                ],
                "targets": targets,
            }
        )

    payload = {
        "version": 1,
        "source": "trained_player_profiles",
        "profiles": profiles,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"profiles": len(profiles), "output": str(output_path)}))
    return 0 if profiles else 1


if __name__ == "__main__":
    raise SystemExit(main())
