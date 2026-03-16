#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.src.services import comparison  # noqa: E402

MIN_TOLERANCES = {
    "load.knee_flexion": 10.0,
    "set.elbow_angle": 10.0,
    "rise.shoulder_angle": 10.0,
    "release.wrist_height": 0.12,
    "release.shoulder_angle": 10.0,
    "release.wrist_flexion_vel_peak": 20.0,
    "follow_through.duration": 0.08,
    "follow_through.index_pip_angle": 10.0,
}

FEATURE_KEYS = list(comparison.REFERENCE_FEATURE_KEYS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an archetype reference library from exported FormFix feature rows."
    )
    parser.add_argument("--input", required=True, help="Input JSONL from export_features.py")
    parser.add_argument("--output", required=True, help="Output JSON file matching FormFix reference schema")
    parser.add_argument(
        "--clusters",
        type=int,
        default=4,
        help="Maximum clusters per shot type group",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=8,
        help="Minimum clips per cluster before we keep it",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for deterministic centroid init",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def enough_features(row: Dict[str, object]) -> bool:
    features = row.get("features", {})
    present = sum(1 for key in FEATURE_KEYS if features.get(key) is not None)
    return present >= max(4, math.ceil(len(FEATURE_KEYS) * 0.5))


def row_vector(row: Dict[str, object]) -> np.ndarray:
    features = row["features"]
    return np.array([features.get(key, np.nan) for key in FEATURE_KEYS], dtype=float)


def fill_missing(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    col_means = np.nanmean(vectors, axis=0)
    inds = np.where(np.isnan(vectors))
    filled = vectors.copy()
    filled[inds] = np.take(col_means, inds[1])
    return filled, col_means


def standardize(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(vectors, axis=0)
    stds = np.std(vectors, axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    return (vectors - means) / stds, means, stds


def init_centroids(vectors: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    indices = list(range(len(vectors)))
    rng.shuffle(indices)
    return vectors[indices[:k]].copy()


def kmeans(vectors: np.ndarray, k: int, seed: int, iterations: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    centroids = init_centroids(vectors, k, seed)
    assignments = np.zeros(len(vectors), dtype=int)

    for _ in range(iterations):
        distances = np.linalg.norm(vectors[:, None, :] - centroids[None, :, :], axis=2)
        new_assignments = np.argmin(distances, axis=1)
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments
        for cluster_idx in range(k):
            members = vectors[assignments == cluster_idx]
            if len(members):
                centroids[cluster_idx] = np.mean(members, axis=0)

    return assignments, centroids


def tolerance_for(key: str, values: Sequence[float]) -> float:
    if not values:
        return MIN_TOLERANCES.get(key, 1.0)
    stdev = float(np.std(values))
    return max(stdev * 1.35, MIN_TOLERANCES.get(key, 1.0))


def descriptor_from_profile(targets: Dict[str, Dict[str, float]], global_means: Dict[str, float]) -> Tuple[str, str, List[str]]:
    tags: List[str] = []
    release_height = targets["release.wrist_height"]["value"] - global_means["release.wrist_height"]
    load_depth = targets["load.knee_flexion"]["value"] - global_means["load.knee_flexion"]
    wrist_speed = targets["release.wrist_flexion_vel_peak"]["value"] - global_means["release.wrist_flexion_vel_peak"]
    finish_hold = targets["follow_through.duration"]["value"] - global_means["follow_through.duration"]

    if release_height > 0.08:
        main = "High release"
        tags.append("tall finish")
    elif release_height < -0.08:
        main = "Compact release"
        tags.append("compact path")
    elif load_depth < -6.0:
        main = "Deep load"
        tags.append("strong leg load")
    else:
        main = "Balanced rhythm"
        tags.append("steady rhythm")

    if finish_hold > 0.05:
        secondary = "hold"
        tags.append("hold the finish")
    elif wrist_speed > 15.0:
        secondary = "snap"
        tags.append("quick wrist finish")
    elif load_depth < -6.0:
        secondary = "lift"
        tags.append("legs first")
    else:
        secondary = "flow"
        tags.append("smooth release")

    label = f"{main} {secondary}".title()
    summary = (
        f"{label} style with a release height of {targets['release.wrist_height']['value']:.2f} "
        f"and follow-through hold of {targets['follow_through.duration']['value']:.2f}s."
    )
    tags.extend(
        [
            "trained library",
            "data-driven",
        ]
    )
    return label, summary, tags[:4]


def main() -> int:
    args = parse_args()
    rows = [row for row in load_rows(Path(args.input)) if enough_features(row)]
    if not rows:
        raise SystemExit("No usable rows found in input.")

    by_shot_type: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_shot_type[str(row.get("shot_type") or "unknown")].append(row)

    profiles: List[Dict[str, object]] = []
    global_rows = np.array([row_vector(row) for row in rows], dtype=float)
    global_rows_filled, _ = fill_missing(global_rows)
    global_means = {
        key: float(value)
        for key, value in zip(FEATURE_KEYS, np.mean(global_rows_filled, axis=0))
    }

    for shot_type, shot_rows in sorted(by_shot_type.items()):
        raw_vectors = np.array([row_vector(row) for row in shot_rows], dtype=float)
        filled_vectors, _ = fill_missing(raw_vectors)
        norm_vectors, _, _ = standardize(filled_vectors)

        max_clusters = max(1, min(args.clusters, len(shot_rows) // max(args.min_cluster_size, 1) or 1))
        assignments, _ = kmeans(norm_vectors, max_clusters, args.seed)

        for cluster_idx in range(max_clusters):
            member_indices = np.where(assignments == cluster_idx)[0]
            if len(member_indices) < args.min_cluster_size:
                continue

            members = [shot_rows[index] for index in member_indices]
            targets: Dict[str, Dict[str, float]] = {}
            for feature_key in FEATURE_KEYS:
                values = [
                    float(member["features"][feature_key])
                    for member in members
                    if member["features"].get(feature_key) is not None
                ]
                if not values:
                    continue
                targets[feature_key] = {
                    "value": round(float(np.median(values)), 4),
                    "tolerance": round(tolerance_for(feature_key, values), 4),
                    "weight": round(comparison._FEATURE_SPECS[feature_key].weight, 3),  # type: ignore[attr-defined]
                }

            if len(targets) < 4:
                continue

            label, summary, tags = descriptor_from_profile(targets, global_means)
            profiles.append(
                {
                    "id": f"{shot_type}_cluster_{cluster_idx}",
                    "label": f"{label} {shot_type.replace('_', ' ').title()}",
                    "category": "archetype",
                    "source": "trained",
                    "shot_types": [shot_type],
                    "summary": summary,
                    "description": (
                        f"Cluster trained from {len(members)} clips in the {shot_type.replace('_', ' ')} group."
                    ),
                    "style_tags": tags,
                    "targets": targets,
                }
            )

    payload = {
        "version": 1,
        "source": "trained_reference_library",
        "profiles": profiles,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"profiles": len(profiles), "output": str(output_path)}))
    return 0 if profiles else 1


if __name__ == "__main__":
    raise SystemExit(main())
