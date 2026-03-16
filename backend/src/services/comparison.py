from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .. import schemas

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_REFERENCE_LIBRARY_PATH = _DATA_DIR / "reference_profiles.json"


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    phase: str
    field: str
    kind: str
    label: str
    unit: str
    weight: float
    direction: str
    phase_label: str
    aligned_detail: str
    lower_detail: str
    lower_cue: Optional[str]
    higher_detail: str
    higher_cue: Optional[str]


@dataclass(frozen=True)
class TargetValue:
    value: float
    tolerance: float
    weight: float


@dataclass(frozen=True)
class ReferenceProfile:
    profile_id: str
    label: str
    category: str
    source: str
    shot_types: tuple[str, ...]
    summary: str
    description: str
    style_tags: tuple[str, ...]
    targets: Dict[str, TargetValue]


_FEATURE_SPECS: Dict[str, FeatureSpec] = {
    "load.knee_flexion": FeatureSpec(
        key="load.knee_flexion",
        phase="load",
        field="knee_flexion",
        kind="angle",
        label="Load depth",
        unit="deg",
        weight=1.15,
        direction="lower",
        phase_label="load",
        aligned_detail="Your load depth already looks a lot like this style.",
        lower_detail="You sink a little deeper than this style during the load.",
        lower_cue="Stay smooth out of the dip so the extra load does not slow down the shot.",
        higher_detail="This style starts from a slightly lower base than you do.",
        higher_cue="Start a little lower before you rise so the shot can build from your legs.",
    ),
    "set.elbow_angle": FeatureSpec(
        key="set.elbow_angle",
        phase="set",
        field="elbow_angle",
        kind="angle",
        label="Set position",
        unit="deg",
        weight=0.95,
        direction="neutral",
        phase_label="set",
        aligned_detail="Your set position is already close to this style.",
        lower_detail="Your set looks a little more tucked than this style.",
        lower_cue="Let the elbow relax slightly so the ball can sit on a cleaner shooting line.",
        higher_detail="Your set looks a little more open than this style.",
        higher_cue="Bring the shooting elbow in slightly and keep the ball tighter to your line.",
    ),
    "rise.shoulder_angle": FeatureSpec(
        key="rise.shoulder_angle",
        phase="rise",
        field="shoulder_angle",
        kind="angle",
        label="Arm lift timing",
        unit="deg",
        weight=0.8,
        direction="neutral",
        phase_label="rise",
        aligned_detail="Your arm lift timing already feels similar to this style.",
        lower_detail="This style gets the shooting side rising a little earlier through the lift.",
        lower_cue="Let the ball and shooting arm rise a touch more with the body instead of waiting on the top.",
        higher_detail="Your shooting side rises a little earlier than this style.",
        higher_cue="Keep the arm patient for a split second and let the legs start the lift first.",
    ),
    "release.wrist_height": FeatureSpec(
        key="release.wrist_height",
        phase="release",
        field="wrist_height",
        kind="angle",
        label="Release height",
        unit="body",
        weight=1.2,
        direction="higher",
        phase_label="release",
        aligned_detail="Your release height already fits this style well.",
        lower_detail="This style finishes a little taller at release.",
        lower_cue="Finish taller and let the ball leave your hand a little higher.",
        higher_detail="You already release a bit higher than this style.",
        higher_cue=None,
    ),
    "release.shoulder_angle": FeatureSpec(
        key="release.shoulder_angle",
        phase="release",
        field="shoulder_angle",
        kind="angle",
        label="Release lift",
        unit="deg",
        weight=0.75,
        direction="neutral",
        phase_label="release",
        aligned_detail="Your upper-body lift at release is close to this style.",
        lower_detail="This style reaches a slightly taller shoulder lift at release.",
        lower_cue="Think about getting fully up through the release instead of stopping just short.",
        higher_detail="Your shoulder lift is a little higher than this style at release.",
        higher_cue="Keep the same tall finish, but avoid forcing extra lift that changes the rhythm.",
    ),
    "release.wrist_flexion_vel_peak": FeatureSpec(
        key="release.wrist_flexion_vel_peak",
        phase="release",
        field="wrist_flexion_vel_peak",
        kind="angle",
        label="Wrist finish",
        unit="deg/s",
        weight=0.85,
        direction="higher",
        phase_label="release",
        aligned_detail="Your wrist finish speed already matches this style pretty well.",
        lower_detail="This style has a slightly cleaner wrist finish through release.",
        lower_cue="Flick the wrist through the ball and let the fingers finish down.",
        higher_detail="You already snap the wrist a little harder than this style.",
        higher_cue=None,
    ),
    "follow_through.duration": FeatureSpec(
        key="follow_through.duration",
        phase="follow_through",
        field="duration",
        kind="timing",
        label="Finish hold",
        unit="s",
        weight=0.9,
        direction="higher",
        phase_label="follow-through",
        aligned_detail="Your follow-through hold already looks like this style.",
        lower_detail="This style holds the finish a little longer.",
        lower_cue="Hold the follow-through until the ball reaches the rim.",
        higher_detail="You already hold the finish longer than this style.",
        higher_cue=None,
    ),
    "follow_through.index_pip_angle": FeatureSpec(
        key="follow_through.index_pip_angle",
        phase="follow_through",
        field="index_pip_angle",
        kind="angle",
        label="Finger roll-through",
        unit="deg",
        weight=0.55,
        direction="higher",
        phase_label="follow-through",
        aligned_detail="Your finger finish already looks close to this style.",
        lower_detail="This style gets a slightly cleaner fingertip roll-through.",
        lower_cue="Let the ball roll off your fingers instead of pushing it from the palm.",
        higher_detail="You already get more fingertip finish than this style.",
        higher_cue=None,
    ),
}

REFERENCE_FEATURE_KEYS = tuple(_FEATURE_SPECS.keys())


def _phase_map(phases: List[schemas.PhaseMetrics]) -> Dict[str, schemas.PhaseMetrics]:
    return {phase.name: phase for phase in phases}


def extract_feature_map(phases: List[schemas.PhaseMetrics]) -> Dict[str, Optional[float]]:
    phase_lookup = _phase_map(phases)
    feature_map: Dict[str, Optional[float]] = {}

    for key, spec in _FEATURE_SPECS.items():
        phase = phase_lookup.get(spec.phase)
        if phase is None:
            feature_map[key] = None
            continue

        if spec.kind == "timing":
            feature_map[key] = phase.timings.get(spec.field)
        else:
            feature_map[key] = phase.angles.get(spec.field)

    return feature_map


def build_training_row(
    phases: List[schemas.PhaseMetrics],
    shot_type: Optional[str],
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    features = extract_feature_map(phases)
    row: Dict[str, object] = {
        "shot_type": shot_type or "unknown",
        "features": features,
    }
    if metadata:
        row["metadata"] = metadata
    return row


@lru_cache(maxsize=1)
def load_reference_profiles(path: Optional[str] = None) -> List[ReferenceProfile]:
    env_path = os.getenv("FORMFIX_REFERENCE_LIBRARY")
    library_path = Path(path or env_path) if (path or env_path) else _REFERENCE_LIBRARY_PATH
    payload = json.loads(library_path.read_text())
    profiles: List[ReferenceProfile] = []

    for item in payload.get("profiles", []):
        targets = {
            key: TargetValue(
                value=float(value["value"]),
                tolerance=max(float(value["tolerance"]), 1e-6),
                weight=float(value.get("weight", _FEATURE_SPECS[key].weight if key in _FEATURE_SPECS else 1.0)),
            )
            for key, value in item.get("targets", {}).items()
            if key in _FEATURE_SPECS
        }
        profiles.append(
            ReferenceProfile(
                profile_id=item["id"],
                label=item["label"],
                category=item.get("category", "archetype"),
                source=item.get("source", "seeded"),
                shot_types=tuple(item.get("shot_types", ["any"])),
                summary=item.get("summary", ""),
                description=item.get("description", ""),
                style_tags=tuple(item.get("style_tags", [])),
                targets=targets,
            )
        )

    return profiles


def _is_profile_applicable(profile: ReferenceProfile, shot_type: Optional[str]) -> bool:
    if not shot_type:
        return True
    return "any" in profile.shot_types or shot_type in profile.shot_types


def _candidate_profiles(shot_type: Optional[str]) -> List[ReferenceProfile]:
    profiles = load_reference_profiles()
    if not shot_type:
        return list(profiles)

    exact = [profile for profile in profiles if shot_type in profile.shot_types]
    if exact:
        return exact

    return [profile for profile in profiles if "any" in profile.shot_types]


def _normalized_delta(user_value: float, target: TargetValue) -> float:
    return abs(user_value - target.value) / target.tolerance


def _score_profile(
    profile: ReferenceProfile,
    feature_map: Dict[str, Optional[float]],
) -> Optional[tuple[float, float, List[schemas.ComparisonMetricDelta]]]:
    weighted_distance = 0.0
    total_weight = 0.0
    deltas: List[schemas.ComparisonMetricDelta] = []

    for key, target in profile.targets.items():
        user_value = feature_map.get(key)
        if user_value is None:
            continue

        spec = _FEATURE_SPECS[key]
        delta = user_value - target.value
        abs_norm = _normalized_delta(user_value, target)
        weighted_distance += target.weight * (abs_norm ** 2)
        total_weight += target.weight

        if abs_norm <= 0.5:
            direction = "aligned"
            summary = spec.aligned_detail
            cue = None
        elif delta < 0:
            direction = "lower"
            summary = spec.lower_detail
            cue = spec.lower_cue
        else:
            direction = "higher"
            summary = spec.higher_detail
            cue = spec.higher_cue

        deltas.append(
            schemas.ComparisonMetricDelta(
                key=key,
                label=spec.label,
                user_value=float(user_value),
                reference_value=float(target.value),
                unit=spec.unit,
                direction=direction,
                summary=summary,
                cue=cue,
            )
        )

    if total_weight == 0.0:
        return None

    distance = float(np.sqrt(weighted_distance / total_weight))
    coverage = min(len(deltas) / max(len(profile.targets), 1), 1.0)
    fit_score = float(np.clip(100.0 - distance * 22.0, 0.0, 100.0))
    confidence = float(np.clip((1.0 - min(distance / 4.0, 1.0)) * (0.45 + 0.55 * coverage), 0.0, 1.0))
    return fit_score, confidence, deltas


def _rank_matching_traits(deltas: List[schemas.ComparisonMetricDelta]) -> List[schemas.ComparisonTrait]:
    aligned = [delta for delta in deltas if delta.direction == "aligned"]
    traits: List[schemas.ComparisonTrait] = []

    for delta in aligned[:3]:
        spec = _FEATURE_SPECS[delta.key]
        traits.append(
            schemas.ComparisonTrait(
                title=delta.label,
                detail=delta.summary,
                phase=spec.phase_label,
            )
        )

    return traits


def _borrow_priority(delta: schemas.ComparisonMetricDelta) -> float:
    spec = _FEATURE_SPECS[delta.key]
    if delta.direction == "aligned":
        return -1.0
    if spec.direction == "higher" and delta.direction == "higher":
        return -1.0
    if spec.direction == "lower" and delta.direction == "lower":
        return -1.0
    score = 1.0
    if delta.cue:
        score += 0.5
    if spec.weight >= 1.0:
        score += 0.5
    return score


def _rank_borrow_traits(deltas: List[schemas.ComparisonMetricDelta]) -> List[schemas.ComparisonTrait]:
    ranked = sorted(
        [delta for delta in deltas if _borrow_priority(delta) > 0],
        key=lambda delta: (_borrow_priority(delta), _FEATURE_SPECS[delta.key].weight),
        reverse=True,
    )

    traits: List[schemas.ComparisonTrait] = []
    for delta in ranked[:3]:
        spec = _FEATURE_SPECS[delta.key]
        traits.append(
            schemas.ComparisonTrait(
                title=delta.label,
                detail=delta.summary,
                cue=delta.cue,
                phase=spec.phase_label,
            )
        )
    return traits


def _build_match(
    profile: ReferenceProfile,
    fit_score: float,
    confidence: float,
    deltas: List[schemas.ComparisonMetricDelta],
) -> schemas.ComparisonMatch:
    matched_traits = _rank_matching_traits(deltas)
    borrow_traits = _rank_borrow_traits(deltas)
    return schemas.ComparisonMatch(
        profile_id=profile.profile_id,
        label=profile.label,
        category=profile.category,
        source=profile.source,
        fit_score=fit_score,
        confidence=confidence,
        summary=profile.summary,
        description=profile.description,
        style_tags=list(profile.style_tags),
        matched_traits=matched_traits,
        borrow_traits=borrow_traits,
        metric_deltas=deltas[:5],
    )


def _comparison_notes(profiles: Iterable[ReferenceProfile], shot_type: Optional[str]) -> List[str]:
    notes = [
        "This comparison library is archetype-based right now, so it points to shot families rather than exact player matches.",
        "The strongest future upgrade is replacing these seeded profiles with profiles trained from labeled clip sets.",
    ]

    if shot_type:
        notes.append(f"Your clip was compared against profiles that fit {shot_type.replace('_', ' ')} shots first.")
    elif any("any" in profile.shot_types for profile in profiles):
        notes.append("No shot type was selected, so the comparison used the full mixed library.")

    return notes[:3]


def build_comparison(
    phases: List[schemas.PhaseMetrics],
    shot_type: Optional[str] = None,
) -> schemas.ComparisonResult:
    profiles = [profile for profile in _candidate_profiles(shot_type) if _is_profile_applicable(profile, shot_type)]
    if not profiles:
        return schemas.ComparisonResult(
            status="unavailable",
            overview="No comparison library is available for this shot type yet.",
            notes=["Add or train reference profiles before enabling style matching here."],
        )

    feature_map = extract_feature_map(phases)
    scored_matches: List[tuple[ReferenceProfile, float, float, List[schemas.ComparisonMetricDelta]]] = []
    for profile in profiles:
        scored = _score_profile(profile, feature_map)
        if scored is None:
            continue
        fit_score, confidence, deltas = scored
        scored_matches.append((profile, fit_score, confidence, deltas))

    if not scored_matches:
        return schemas.ComparisonResult(
            status="unavailable",
            overview="We could not extract enough features to compare this clip against the reference styles.",
            notes=["Try a cleaner angle with the full body and release visible."],
        )

    scored_matches.sort(key=lambda item: (item[1], item[2]), reverse=True)
    primary_profile, primary_score, primary_confidence, primary_deltas = scored_matches[0]
    primary = _build_match(primary_profile, primary_score, primary_confidence, primary_deltas)

    alternatives = [
        _build_match(profile, fit_score, confidence, deltas)
        for profile, fit_score, confidence, deltas in scored_matches[1:3]
    ]

    overview = (
        f"Your closest current style match is {primary.label}. "
        "Use it as a style family to borrow from, not as a perfect shot to copy frame for frame."
    )

    status = "trained_reference" if primary.source == "trained" else "seeded_reference"
    return schemas.ComparisonResult(
        status=status,
        overview=overview,
        primary=primary,
        alternatives=alternatives,
        notes=_comparison_notes(profiles, shot_type),
    )
