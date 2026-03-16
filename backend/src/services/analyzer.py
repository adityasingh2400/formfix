from __future__ import annotations

import base64
import hashlib
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .. import schemas
from . import comparison, research_bank
from .video_utils import decode_video

logger = logging.getLogger(__name__)

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

# Minimal hand skeleton (MediaPipe Hands topology) for CV2 overlays.
_HAND_CONNECTIONS: List[Tuple[str, str]] = [
    ("wrist", "thumb_cmc"),
    ("thumb_cmc", "thumb_mcp"),
    ("thumb_mcp", "thumb_ip"),
    ("thumb_ip", "thumb_tip"),
    ("wrist", "index_finger_mcp"),
    ("index_finger_mcp", "index_finger_pip"),
    ("index_finger_pip", "index_finger_dip"),
    ("index_finger_dip", "index_finger_tip"),
    ("wrist", "middle_finger_mcp"),
    ("middle_finger_mcp", "middle_finger_pip"),
    ("middle_finger_pip", "middle_finger_dip"),
    ("middle_finger_dip", "middle_finger_tip"),
    ("wrist", "ring_finger_mcp"),
    ("ring_finger_mcp", "ring_finger_pip"),
    ("ring_finger_pip", "ring_finger_dip"),
    ("ring_finger_dip", "ring_finger_tip"),
    ("wrist", "pinky_mcp"),
    ("pinky_mcp", "pinky_pip"),
    ("pinky_pip", "pinky_dip"),
    ("pinky_dip", "pinky_tip"),
    # Palm connections
    ("index_finger_mcp", "middle_finger_mcp"),
    ("middle_finger_mcp", "ring_finger_mcp"),
    ("ring_finger_mcp", "pinky_mcp"),
]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_YOLO_WEIGHTS = _REPO_ROOT / "yolov8n.pt"


@dataclass
class _ArmObs:
    elbow_angle: Optional[float]
    shoulder_angle: Optional[float]
    wrist_height: Optional[float]
    wrist_flexion: Optional[float]
    index_pip_angle: Optional[float]
    arm_visibility: float
    hand_present: bool
    wrist_px: Optional[Tuple[int, int]]


@dataclass
class FrameMetrics:
    frame_index: int
    timestamp_s: float
    knee_angle: float
    hip_angle: float
    elbow_angle: float
    shoulder_angle: float
    wrist_height: float
    offhand_gap: Optional[float]
    visibility: float
    keypoints_px: Dict[str, Tuple[int, int]]
    ball_pos: Optional[Tuple[int, int]] = None
    shooting_side: str = "right"
    wrist_flexion: Optional[float] = None
    wrist_flexion_vel: Optional[float] = None
    index_pip_angle: Optional[float] = None
    hand_confidence: float = 0.0


@dataclass
class PreparedAnalysis:
    frames: List[np.ndarray]
    fps: float
    stride: int
    metrics: List[FrameMetrics]
    phases: List[schemas.PhaseMetrics]
    boundaries: dict
    min_dt: float
    labels: List[str]
    issues: List[schemas.Issue]
    confidence_notes: List[str]


_PerFrameObservation = Tuple[
    int,
    float,
    float,
    Dict[str, Tuple[int, int]],
    Optional[Tuple[int, int]],
    Dict[str, _ArmObs],
    Optional[float],
    Optional[float],
    float,
    float,
]


_PREPARED_CACHE: "OrderedDict[str, Tuple[float, PreparedAnalysis]]" = OrderedDict()
_PREPARED_CACHE_MAX_ITEMS = 4
_PREPARED_CACHE_TTL_S = 20 * 60

_ISSUE_GUIDANCE: Dict[str, Dict[str, str]] = {
    "Knee bend shallow": {
        "title": "Bend your knees more",
        "cue": "Bend your knees deeper and lower your hips before you start the shot.",
        "why": "More knee bend gives you power from your legs instead of forcing the ball with your arms.",
        "drill": "Before each shot, drop your hips lower and feel your legs push you up.",
        "research_title": "Your legs power the shot",
        "research_body": "Deeper knee bend creates more lift and makes your shot easier to repeat.",
    },
    "Elbow angle at set is off-band": {
        "title": "Bring your elbow in",
        "cue": "Tuck your shooting elbow closer to your body and point it at the rim.",
        "why": "A tucked elbow keeps the ball on a straight line to the basket.",
        "drill": "Practice one-hand form shots and feel your elbow directly under the ball.",
        "research_title": "Elbow alignment matters",
        "research_body": "When your elbow points at the rim, the ball travels straighter.",
    },
    "Low release height": {
        "title": "Release the ball higher",
        "cue": "Extend your arm fully and release the ball at the top of your reach.",
        "why": "A higher release is harder to block and gives the ball a better arc.",
        "drill": "Focus on fully extending your arm before letting go of the ball.",
        "research_title": "Higher release = better arc",
        "research_body": "Releasing higher creates a steeper angle into the basket, which means more room for the ball to go in.",
    },
    "Rushed upper-body sequencing": {
        "title": "Let your legs start the shot",
        "cue": "Push up with your legs first, then bring your arm through after.",
        "why": "Starting with your legs makes the shot smoother and more powerful.",
        "drill": "Think 'legs first, then arm' on every shot.",
        "research_title": "Power flows from the ground up",
        "research_body": "The best shooters start their motion from their legs, not their arms.",
    },
    "Off-hand too close": {
        "title": "Get your guide hand off the ball",
        "cue": "Let your guide hand fall away as you release. Only your shooting hand should push the ball.",
        "why": "If your guide hand stays on too long, it can push the ball off target.",
        "drill": "Practice releasing with your guide hand coming off early and clean.",
        "research_title": "Guide hand should only guide",
        "research_body": "Your off-hand balances the ball but should not add any force to the shot.",
    },
    "Wrist snap looks weak": {
        "title": "Snap your wrist harder",
        "cue": "Flick your wrist down as you release and let your fingers point at the rim.",
        "why": "A strong wrist snap gives the ball backspin and a cleaner release.",
        "drill": "Exaggerate the wrist flick on close-range shots until it feels natural.",
        "research_title": "Wrist snap creates backspin",
        "research_body": "Backspin from your wrist makes the ball bounce softer if it hits the rim.",
    },
    "Short follow-through": {
        "title": "Hold your follow-through",
        "cue": "Keep your arm extended and wrist relaxed until the ball hits the rim.",
        "why": "Holding your finish helps you shoot the same way every time.",
        "drill": "Freeze your follow-through on every shot until the ball reaches the basket.",
        "research_title": "Follow-through builds consistency",
        "research_body": "Holding your finish reinforces good muscle memory.",
    },
    "Finger roll-through looks limited": {
        "title": "Release off your fingertips",
        "cue": "Let the ball roll off your index and middle fingers, not your palm.",
        "why": "A fingertip release gives you better control and spin.",
        "drill": "On form shots, focus on feeling the ball leave your fingertips last.",
        "research_title": "Fingertips control the release",
        "research_body": "The ball should roll off your fingers, not get pushed from your palm.",
    },
}

_GENERIC_RESEARCH_NOTES = [
    schemas.ResearchNote(
        title="Consistency beats perfection",
        body="The goal is to repeat your best shot every time, not to copy someone else's form.",
    ),
    schemas.ResearchNote(
        title="Power comes from your legs",
        body="Good shooters use their legs to generate power, which makes the arm motion easier and more accurate.",
    ),
]

_ISSUE_PLAYBACK_SPECS: Dict[str, Dict[str, object]] = {
    "Knee bend shallow": {
        "title": "Bend your knees more",
        "targets": ["left_knee", "right_knee"],
        "directions": ["down", "down"],
        "motion_types": ["linear", "linear"],
        "magnitudes": [1.2, 1.2],
        "bubble_offset": (0.0, -0.18),
        "research_key": "lower_body_load",
        "compare_label": "Load depth",
    },
    "Elbow angle at set is off-band": {
        "title": "Bring your elbow in",
        "targets": ["shoot_elbow"],
        "directions": ["toward_midline"],
        "motion_types": ["arc_down"],
        "magnitudes": [1.0],
        "pivot_targets": ["shoot_shoulder"],
        "arc_radii": [0.08],
        "bubble_offset": None,
        "research_key": "elbow_line",
        "compare_label": "Set position",
    },
    "Low release height": {
        "title": "Finish taller",
        "targets": ["shoot_wrist", "shoot_shoulder"],
        "directions": ["up", "up"],
        "motion_types": ["arc_up", "linear"],
        "magnitudes": [1.3, 0.8],
        "pivot_targets": ["shoot_elbow", None],
        "arc_radii": [0.06, None],
        "bubble_offset": (0.0, -0.18),
        "research_key": "release_height",
        "compare_label": "Release height",
    },
    "Rushed upper-body sequencing": {
        "title": "Let your legs start the shot",
        "targets": ["left_knee", "right_knee"],
        "directions": ["up", "up"],
        "motion_types": ["linear", "linear"],
        "magnitudes": [1.0, 1.0],
        "bubble_offset": (0.0, -0.18),
        "research_key": "sequencing_lift",
        "compare_label": "Arm lift timing",
    },
    "Off-hand too close": {
        "title": "Get the guide hand off sooner",
        "targets": ["guide_wrist"],
        "directions": ["away_from_midline"],
        "motion_types": ["arc_down"],
        "magnitudes": [1.2],
        "pivot_targets": ["guide_elbow"],
        "arc_radii": [0.07],
        "bubble_offset": None,
        "research_key": "release_control",
    },
    "Wrist snap looks weak": {
        "title": "Snap the wrist through",
        "targets": ["shoot_wrist"],
        "directions": ["down"],
        "motion_types": ["rotate_cw"],
        "magnitudes": [1.4],
        "pivot_targets": ["shoot_wrist"],
        "arc_radii": [0.05],
        "bubble_offset": (0.0, -0.16),
        "research_key": "release_control",
        "compare_label": "Wrist finish",
    },
    "Short follow-through": {
        "title": "Hold your finish",
        "targets": ["shoot_wrist"],
        "directions": ["up"],
        "motion_types": ["arc_up"],
        "magnitudes": [1.0],
        "pivot_targets": ["shoot_elbow"],
        "arc_radii": [0.06],
        "bubble_offset": (0.0, -0.16),
        "research_key": "release_control",
        "compare_label": "Finish hold",
    },
    "Finger roll-through looks limited": {
        "title": "Let the ball roll off your fingers",
        "targets": ["shoot_index_tip", "shoot_middle_tip"],
        "directions": ["down", "down"],
        "motion_types": ["rotate_cw", "rotate_cw"],
        "magnitudes": [1.2, 1.2],
        "pivot_targets": ["shoot_wrist", "shoot_wrist"],
        "arc_radii": [0.03, 0.03],
        "bubble_offset": (0.0, -0.16),
        "research_key": "release_control",
        "compare_label": "Finger roll-through",
    },
}

_COMPARISON_PLAYBACK_SPECS: Dict[str, Dict[str, object]] = {
    "Load depth": {
        "title": "Bend your knees deeper",
        "targets": ["left_knee", "right_knee"],
        "directions": ["down", "down"],
        "motion_types": ["linear", "linear"],
        "magnitudes": [1.2, 1.2],
        "phase": "load",
        "bubble_offset": (0.0, -0.18),
        "research_key": "lower_body_load",
        "compare_label": "Load depth",
    },
    "Set position": {
        "title": "Tuck your elbow in more",
        "targets": ["shoot_elbow"],
        "directions": ["toward_midline"],
        "motion_types": ["arc_down"],
        "magnitudes": [1.0],
        "pivot_targets": ["shoot_shoulder"],
        "arc_radii": [0.08],
        "phase": "set",
        "bubble_offset": None,
        "research_key": "elbow_line",
        "compare_label": "Set position",
    },
    "Arm lift timing": {
        "title": "Push up with your legs first",
        "targets": ["left_knee", "right_knee"],
        "directions": ["up", "up"],
        "motion_types": ["linear", "linear"],
        "magnitudes": [1.0, 1.0],
        "phase": "rise",
        "bubble_offset": (0.0, -0.18),
        "research_key": "sequencing_lift",
        "compare_label": "Arm lift timing",
    },
    "Release height": {
        "title": "Extend your arm fully at release",
        "targets": ["shoot_wrist", "shoot_shoulder"],
        "directions": ["up", "up"],
        "motion_types": ["arc_up", "linear"],
        "magnitudes": [1.3, 0.8],
        "pivot_targets": ["shoot_elbow", None],
        "arc_radii": [0.06, None],
        "phase": "release",
        "bubble_offset": (0.0, -0.18),
        "research_key": "release_height",
        "compare_label": "Release height",
    },
    "Release lift": {
        "title": "Release the ball higher",
        "targets": ["shoot_wrist", "shoot_shoulder"],
        "directions": ["up", "up"],
        "motion_types": ["arc_up", "linear"],
        "magnitudes": [1.3, 0.8],
        "pivot_targets": ["shoot_elbow", None],
        "arc_radii": [0.06, None],
        "phase": "release",
        "bubble_offset": (0.0, -0.18),
        "research_key": "release_height",
        "compare_label": "Release lift",
    },
    "Wrist finish": {
        "title": "Snap your wrist down harder",
        "targets": ["shoot_wrist"],
        "directions": ["down"],
        "motion_types": ["rotate_cw"],
        "magnitudes": [1.4],
        "pivot_targets": ["shoot_wrist"],
        "arc_radii": [0.05],
        "phase": "release",
        "bubble_offset": (0.0, -0.16),
        "research_key": "release_control",
        "compare_label": "Wrist finish",
    },
    "Finish hold": {
        "title": "Hold your follow-through longer",
        "targets": ["shoot_wrist"],
        "directions": ["up"],
        "motion_types": ["arc_up"],
        "magnitudes": [1.0],
        "pivot_targets": ["shoot_elbow"],
        "arc_radii": [0.06],
        "phase": "follow_through",
        "bubble_offset": (0.0, -0.16),
        "research_key": "release_control",
        "compare_label": "Finish hold",
    },
    "Finger roll-through": {
        "title": "Release off your fingertips",
        "targets": ["shoot_index_tip", "shoot_middle_tip"],
        "directions": ["down", "down"],
        "motion_types": ["rotate_cw", "rotate_cw"],
        "magnitudes": [1.2, 1.2],
        "pivot_targets": ["shoot_wrist", "shoot_wrist"],
        "arc_radii": [0.03, 0.03],
        "phase": "follow_through",
        "bubble_offset": (0.0, -0.16),
        "research_key": "release_control",
        "compare_label": "Finger roll-through",
    },
}

_RELEASE_CONTROL_ISSUE_PRIORITY: Dict[str, int] = {
    "Wrist snap looks weak": 0,
    "Finger roll-through looks limited": 1,
    "Short follow-through": 2,
    "Off-hand too close": 3,
}

_RELEASE_CONTROL_COMPARE_PRIORITY: Dict[str, int] = {
    "Wrist finish": 0,
    "Finger roll-through": 1,
    "Finish hold": 2,
}

_COMPARISON_RESEARCH_LINES: Dict[str, str] = {
    "Load depth": "Deeper knee bend gives you more power from your legs and makes your shot easier to repeat.",
    "Set position": "Keeping your elbow tucked and aligned with the rim helps the ball travel straight.",
    "Arm lift timing": "Starting the shot with your legs instead of your arms makes it smoother and more powerful.",
    "Release height": "A higher release gives the ball better arc and is harder to block.",
    "Release lift": "Fully extending your arm at release creates a cleaner shot.",
    "Wrist finish": "A strong wrist snap adds backspin and makes the ball bounce softer on the rim.",
    "Finish hold": "Holding your follow-through builds muscle memory for consistent shooting.",
    "Finger roll-through": "Releasing off your fingertips gives you better control and spin.",
}


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute angle ABC in degrees."""
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) or 1e-6
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _normalize_shooting_side(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"left", "right"}:
        return normalized
    return None


def _opposite_side(side: str) -> str:
    return "left" if side == "right" else "right"


def _release_window_positions(per_frame: List[_PerFrameObservation]) -> Tuple[List[int], float]:
    shoulder_refs = [
        float(shoulder_width)
        for _, _, _, _, _, _, shoulder_width, _, _, _ in per_frame
        if shoulder_width and shoulder_width > 1e-6
    ]
    shoulder_scale = float(np.median(shoulder_refs)) if shoulder_refs else 1.0

    ball_positions_with_idx: List[Tuple[int, float]] = [
        (pos, float(ball[1]))
        for pos, (_, _, _, _, ball, _, _, _, _, _) in enumerate(per_frame)
        if ball is not None
    ]

    release_positions: List[int] = []
    if ball_positions_with_idx:
        peak_ball_y = min(y for _, y in ball_positions_with_idx)
        ball_span = max(y for _, y in ball_positions_with_idx) - peak_ball_y
        top_band = max(18.0, ball_span * 0.12)
        release_positions = [
            pos for pos, y in ball_positions_with_idx if y <= peak_ball_y + top_band
        ]
        if len(release_positions) > 6:
            release_positions = [
                pos
                for pos, _ in sorted(ball_positions_with_idx, key=lambda item: item[1])[:6]
            ]

    if not release_positions:
        wrist_height_rank = []
        for pos, (_, _, _, _, _, arms, _, _, _, _) in enumerate(per_frame):
            peak_wrist_height = max(
                float(arms["left"].wrist_height or -1e9),
                float(arms["right"].wrist_height or -1e9),
            )
            if np.isfinite(peak_wrist_height):
                wrist_height_rank.append((pos, peak_wrist_height))
        wrist_height_rank.sort(key=lambda item: item[1], reverse=True)
        release_positions = [pos for pos, _ in wrist_height_rank[: min(6, len(wrist_height_rank))]]

    if not release_positions:
        release_positions = list(range(min(len(per_frame), 6)))

    release_window = sorted(
        {
            min(max(pos + offset, 0), len(per_frame) - 1)
            for pos in release_positions
            for offset in (-1, 0, 1)
        }
    )
    return release_window, shoulder_scale


def _score_shooting_sides(
    per_frame: List[_PerFrameObservation],
) -> Tuple[Dict[str, Dict[str, Optional[float]]], Dict[str, float], Dict[str, float]]:
    release_window, shoulder_scale = _release_window_positions(per_frame)
    release_window_set = set(release_window)

    def _side_stats(side: str) -> Dict[str, Optional[float]]:
        overall_hand_hits = 0
        overall_vis: List[float] = []
        release_heights: List[float] = []
        release_vis: List[float] = []
        release_hand_hits = 0
        release_dists: List[float] = []
        snap_peaks: List[float] = []

        for pos, (_, _, _, _, ball, arms, shoulder_width, _, _, _) in enumerate(per_frame):
            arm = arms[side]
            overall_vis.append(float(arm.arm_visibility))
            if arm.hand_present:
                overall_hand_hits += 1

            if pos not in release_window_set:
                continue

            release_vis.append(float(arm.arm_visibility))
            if arm.wrist_height is not None:
                release_heights.append(float(arm.wrist_height))
            if arm.hand_present:
                release_hand_hits += 1
            if ball is not None and arm.wrist_px is not None:
                scale = (
                    float(shoulder_width)
                    if shoulder_width and shoulder_width > 1e-6
                    else shoulder_scale
                )
                bx, by = ball
                wx, wy = arm.wrist_px
                release_dists.append(float(np.hypot(wx - bx, wy - by) / max(scale, 1e-6)))

        for pos in range(1, len(per_frame)):
            if pos not in release_window_set and (pos - 1) not in release_window_set:
                continue
            prev_t = float(per_frame[pos - 1][1])
            curr_t = float(per_frame[pos][1])
            dt_s = max(curr_t - prev_t, 1e-6)
            prev_flex = per_frame[pos - 1][5][side].wrist_flexion
            curr_flex = per_frame[pos][5][side].wrist_flexion
            if prev_flex is None or curr_flex is None:
                continue
            snap_peaks.append(float((curr_flex - prev_flex) / dt_s))

        return {
            "release_dist": float(np.median(release_dists)) if release_dists else None,
            "release_height": float(np.mean(release_heights)) if release_heights else None,
            "release_hand_rate": release_hand_hits / max(len(release_window), 1),
            "overall_hand_rate": overall_hand_hits / max(len(per_frame), 1),
            "visibility": float(np.mean(overall_vis)) if overall_vis else 0.0,
            "release_visibility": float(np.mean(release_vis)) if release_vis else None,
            "snap_peak": max(snap_peaks) if snap_peaks else None,
        }

    stats = {side: _side_stats(side) for side in ("left", "right")}
    release_scores = {"left": 0.0, "right": 0.0}

    def _award_vote(
        key: str,
        *,
        higher_is_better: bool,
        threshold: float,
        points: float,
    ) -> None:
        left_value = stats["left"].get(key)
        right_value = stats["right"].get(key)
        if left_value is None or right_value is None:
            if left_value is not None and right_value is None:
                release_scores["left"] += points * 0.5
            elif right_value is not None and left_value is None:
                release_scores["right"] += points * 0.5
            return

        diff = float(left_value) - float(right_value)
        if not higher_is_better:
            diff *= -1.0
        if abs(diff) < threshold:
            return
        release_scores["left" if diff > 0 else "right"] += points

    # Release-window ball proximity matters, but only near the top of the shot.
    _award_vote("release_dist", higher_is_better=False, threshold=0.10, points=3.0)
    # The release hand should show the clearer snap and follow-through markers.
    _award_vote("snap_peak", higher_is_better=True, threshold=18.0, points=2.25)
    _award_vote("release_height", higher_is_better=True, threshold=0.035, points=1.0)
    _award_vote("release_hand_rate", higher_is_better=True, threshold=0.18, points=1.0)
    _award_vote("release_visibility", higher_is_better=True, threshold=0.08, points=0.5)

    composite_scores = {"left": release_scores["left"], "right": release_scores["right"]}
    for side in ("left", "right"):
        side_stats = stats[side]
        composite_scores[side] += 0.9 * float(side_stats.get("overall_hand_rate") or 0.0)
        composite_scores[side] += 0.6 * float(side_stats.get("visibility") or 0.0)
        composite_scores[side] += 0.35 * float(side_stats.get("release_height") or 0.0)
        composite_scores[side] += 0.012 * max(float(side_stats.get("snap_peak") or 0.0), 0.0)
        if side_stats.get("release_dist") is not None:
            composite_scores[side] += 2.6 / (1.0 + float(side_stats["release_dist"]))

    return stats, release_scores, composite_scores


def _resolve_tracking_side(
    selected_side: str | None,
    release_scores: Dict[str, float],
    composite_scores: Dict[str, float],
    *,
    flip_margin: float = 1.0,
) -> Tuple[str, bool]:
    """Determine which MediaPipe landmark side corresponds to the shooting hand.
    
    Based on testing with mirrored front-camera video:
    - User clicks "left" → their actual RIGHT hand gets highlighted
    - User clicks "right" → their actual LEFT hand gets highlighted
    
    This means MediaPipe's labeling is inverted from the user's perspective.
    We must INVERT the user's selection to highlight the correct hand.
    
    Returns:
        Tuple of (resolved_side, was_flipped)
    """
    auto_side = "left" if composite_scores["left"] > composite_scores["right"] else "right"
    
    logger.info(
        "Shooting side resolution: user_selected=%s, auto_detected=%s, "
        "release_scores={left=%.2f, right=%.2f}, composite_scores={left=%.2f, right=%.2f}",
        selected_side, auto_side,
        release_scores.get("left", 0), release_scores.get("right", 0),
        composite_scores.get("left", 0), composite_scores.get("right", 0),
    )
    
    if selected_side is None:
        logger.info("No user selection, using auto-detected side: %s", auto_side)
        return auto_side, False

    # INVERT the user's selection to match MediaPipe's labeling
    # User says "right" (wants their right hand) → use MediaPipe "left"
    # User says "left" (wants their left hand) → use MediaPipe "right"
    mediapipe_side = _opposite_side(selected_side)
    logger.info(
        "User selected '%s' → using MediaPipe '%s' (inverted)",
        selected_side, mediapipe_side
    )
    return mediapipe_side, True


def _cache_key(file_bytes: bytes, shot_type: str | None, shooting_hand: str | None) -> str:
    digest = hashlib.sha1(file_bytes).hexdigest()
    normalized_hand = _normalize_shooting_side(shooting_hand) or "auto"
    return f"{digest}:{shot_type or 'any'}:{normalized_hand}"


def _get_cached_prepared(cache_key: str) -> Optional[PreparedAnalysis]:
    now = time.time()
    cached = _PREPARED_CACHE.get(cache_key)
    if not cached:
        logger.debug("Cache miss for key: %s", cache_key)
        return None

    created_at, prepared = cached
    if now - created_at > _PREPARED_CACHE_TTL_S:
        logger.debug("Cache expired for key: %s", cache_key)
        _PREPARED_CACHE.pop(cache_key, None)
        return None

    logger.info("Cache hit for key: %s (shooting_side in cached result: %s)", 
                cache_key, prepared.metrics[0].shooting_side if prepared.metrics else "N/A")
    _PREPARED_CACHE.move_to_end(cache_key)
    return prepared


def _store_cached_prepared(cache_key: str, prepared: PreparedAnalysis) -> PreparedAnalysis:
    _PREPARED_CACHE[cache_key] = (time.time(), prepared)
    _PREPARED_CACHE.move_to_end(cache_key)
    while len(_PREPARED_CACHE) > _PREPARED_CACHE_MAX_ITEMS:
        _PREPARED_CACHE.popitem(last=False)
    return prepared


def _detect_ball(frames: List[np.ndarray]) -> List[Optional[Tuple[int, int]]]:
    """Run YOLOv8 to detect sports ball."""
    try:
        # Keep ultralytics optional; we can still run form analysis without ball tracking.
        from ultralytics import YOLO  # type: ignore

        if not _YOLO_WEIGHTS.exists():
            return [None] * len(frames)

        model = YOLO(str(_YOLO_WEIGHTS))  # smallest model
        results = model(frames, verbose=False)
        ball_positions = []
        
        for res in results:
            # class 32 is sports ball in COCO
            boxes = res.boxes
            ball_box = None
            max_conf = 0.0
            
            for box in boxes:
                if int(box.cls[0]) == 32:
                    conf = float(box.conf[0])
                    if conf > max_conf:
                        max_conf = conf
                        ball_box = box.xyxy[0].cpu().numpy()
            
            if ball_box is not None:
                x1, y1, x2, y2 = ball_box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                ball_positions.append((cx, cy))
            else:
                ball_positions.append(None)
        return ball_positions
    except Exception:
        # Fallback or if weights download fails
        return [None] * len(frames)


def _extract_pose_metrics(
    frames: List[np.ndarray],
    fps: float,
    stride: int,
    shooting_hand: str | None = None,
) -> List[FrameMetrics]:
    """
    Extract pose + hand landmarks using MediaPipe Holistic.

    Why Holistic:
    - Pose gives us gross mechanics (hips/knees/shoulders/elbows/wrists).
    - Hands give us wrist flexion ("flick") and finger joint cues.
    """
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=False,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    ball_positions = _detect_ball(frames)
    dt = stride / fps if fps else 1 / 30.0

    # First pass: collect per-frame observations for both sides so we can pick
    # the right tracked arm, or correct for a mirrored/swapped landmark stream.
    per_frame: List[_PerFrameObservation] = []

    for idx, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)
        if not result.pose_landmarks:
            continue

        pose_lm = result.pose_landmarks.landmark
        pose_world = (
            result.pose_world_landmarks.landmark if result.pose_world_landmarks else None
        )
        h, w = frame.shape[:2]

        def _pose_img(name: mp_pose.PoseLandmark) -> np.ndarray:
            l = pose_lm[name]
            return np.array([l.x, l.y, l.z], dtype=float)

        def _pose_world_pt(name: mp_pose.PoseLandmark) -> np.ndarray:
            if pose_world is None:
                return _pose_img(name)
            l = pose_world[name]
            return np.array([l.x, l.y, l.z], dtype=float)

        def _pose_px(name: mp_pose.PoseLandmark) -> Tuple[int, int]:
            l = pose_lm[name]
            return int(l.x * w), int(l.y * h)

        # Torso basis for a more view-robust "height" signal.
        l_hip_w = _pose_world_pt(mp_pose.PoseLandmark.LEFT_HIP)
        r_hip_w = _pose_world_pt(mp_pose.PoseLandmark.RIGHT_HIP)
        l_sh_w = _pose_world_pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
        r_sh_w = _pose_world_pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
        hip_center = 0.5 * (l_hip_w + r_hip_w)
        sh_center = 0.5 * (l_sh_w + r_sh_w)
        torso_axis = sh_center - hip_center
        torso_len = float(np.linalg.norm(torso_axis) or 1e-6)
        torso_unit = torso_axis / torso_len

        shoulder_width = float(np.linalg.norm(r_sh_w - l_sh_w) or 1e-6)
        wrist_gap = None
        try:
            lw = _pose_world_pt(mp_pose.PoseLandmark.LEFT_WRIST)
            rw = _pose_world_pt(mp_pose.PoseLandmark.RIGHT_WRIST)
            wrist_gap = float(np.linalg.norm(rw - lw))
        except Exception:
            wrist_gap = None

        # Lower body (use the most-bent knee/hip we can see).
        knee_angles: List[float] = []
        hip_angles: List[float] = []
        try:
            knee_angles.append(
                _angle(
                    _pose_world_pt(mp_pose.PoseLandmark.LEFT_HIP),
                    _pose_world_pt(mp_pose.PoseLandmark.LEFT_KNEE),
                    _pose_world_pt(mp_pose.PoseLandmark.LEFT_ANKLE),
                )
            )
        except Exception:
            pass
        try:
            knee_angles.append(
                _angle(
                    _pose_world_pt(mp_pose.PoseLandmark.RIGHT_HIP),
                    _pose_world_pt(mp_pose.PoseLandmark.RIGHT_KNEE),
                    _pose_world_pt(mp_pose.PoseLandmark.RIGHT_ANKLE),
                )
            )
        except Exception:
            pass
        try:
            hip_angles.append(
                _angle(
                    _pose_world_pt(mp_pose.PoseLandmark.LEFT_SHOULDER),
                    _pose_world_pt(mp_pose.PoseLandmark.LEFT_HIP),
                    _pose_world_pt(mp_pose.PoseLandmark.LEFT_KNEE),
                )
            )
        except Exception:
            pass
        try:
            hip_angles.append(
                _angle(
                    _pose_world_pt(mp_pose.PoseLandmark.RIGHT_SHOULDER),
                    _pose_world_pt(mp_pose.PoseLandmark.RIGHT_HIP),
                    _pose_world_pt(mp_pose.PoseLandmark.RIGHT_KNEE),
                )
            )
        except Exception:
            pass

        if not knee_angles or not hip_angles:
            continue
        knee_angle = float(min(knee_angles))
        hip_angle = float(min(hip_angles))

        # Pose visibility across core joints.
        vis_vals = [
            pose_lm[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility,
            pose_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility,
            pose_lm[mp_pose.PoseLandmark.LEFT_ELBOW].visibility,
            pose_lm[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility,
            pose_lm[mp_pose.PoseLandmark.LEFT_WRIST].visibility,
            pose_lm[mp_pose.PoseLandmark.RIGHT_WRIST].visibility,
            pose_lm[mp_pose.PoseLandmark.LEFT_HIP].visibility,
            pose_lm[mp_pose.PoseLandmark.RIGHT_HIP].visibility,
            pose_lm[mp_pose.PoseLandmark.LEFT_KNEE].visibility,
            pose_lm[mp_pose.PoseLandmark.RIGHT_KNEE].visibility,
            pose_lm[mp_pose.PoseLandmark.LEFT_ANKLE].visibility,
            pose_lm[mp_pose.PoseLandmark.RIGHT_ANKLE].visibility,
        ]
        visibility = float(np.mean(vis_vals))

        keypoints_px: Dict[str, Tuple[int, int]] = {
            "r_shoulder": _pose_px(mp_pose.PoseLandmark.RIGHT_SHOULDER),
            "r_elbow": _pose_px(mp_pose.PoseLandmark.RIGHT_ELBOW),
            "r_wrist": _pose_px(mp_pose.PoseLandmark.RIGHT_WRIST),
            "r_hip": _pose_px(mp_pose.PoseLandmark.RIGHT_HIP),
            "r_knee": _pose_px(mp_pose.PoseLandmark.RIGHT_KNEE),
            "r_ankle": _pose_px(mp_pose.PoseLandmark.RIGHT_ANKLE),
            "l_shoulder": _pose_px(mp_pose.PoseLandmark.LEFT_SHOULDER),
            "l_elbow": _pose_px(mp_pose.PoseLandmark.LEFT_ELBOW),
            "l_wrist": _pose_px(mp_pose.PoseLandmark.LEFT_WRIST),
            "l_hip": _pose_px(mp_pose.PoseLandmark.LEFT_HIP),
            "l_knee": _pose_px(mp_pose.PoseLandmark.LEFT_KNEE),
            "l_ankle": _pose_px(mp_pose.PoseLandmark.LEFT_ANKLE),
            "nose": _pose_px(mp_pose.PoseLandmark.NOSE),
            "r_eye": _pose_px(mp_pose.PoseLandmark.RIGHT_EYE),
            "l_eye": _pose_px(mp_pose.PoseLandmark.LEFT_EYE),
            "r_ear": _pose_px(mp_pose.PoseLandmark.RIGHT_EAR),
            "l_ear": _pose_px(mp_pose.PoseLandmark.LEFT_EAR),
        }

        def _hand_img_pt(hand_lms, name: mp_hands.HandLandmark) -> np.ndarray:
            l = hand_lms.landmark[name]
            return np.array([l.x, l.y, l.z], dtype=float)

        def _hand_px(hand_lms, name: mp_hands.HandLandmark) -> Tuple[int, int]:
            l = hand_lms.landmark[name]
            return int(l.x * w), int(l.y * h)

        # Add hand keypoints for overlays if present.
        if result.right_hand_landmarks:
            for hlm in mp_hands.HandLandmark:
                keypoints_px[f"r_hand_{hlm.name.lower()}"] = _hand_px(
                    result.right_hand_landmarks, hlm
                )
        if result.left_hand_landmarks:
            for hlm in mp_hands.HandLandmark:
                keypoints_px[f"l_hand_{hlm.name.lower()}"] = _hand_px(
                    result.left_hand_landmarks, hlm
                )

        def _arm_obs(side: str) -> _ArmObs:
            if side == "right":
                sh = _pose_world_pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
                el = _pose_world_pt(mp_pose.PoseLandmark.RIGHT_ELBOW)
                wr = _pose_world_pt(mp_pose.PoseLandmark.RIGHT_WRIST)
                hip = _pose_world_pt(mp_pose.PoseLandmark.RIGHT_HIP)
                wrist_px = _pose_px(mp_pose.PoseLandmark.RIGHT_WRIST)
                arm_vis = float(
                    np.mean(
                        [
                            pose_lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility,
                            pose_lm[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility,
                            pose_lm[mp_pose.PoseLandmark.RIGHT_WRIST].visibility,
                        ]
                    )
                )
                hand_lms = result.right_hand_landmarks
                elbow_img = _pose_img(mp_pose.PoseLandmark.RIGHT_ELBOW)
            else:
                sh = _pose_world_pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
                el = _pose_world_pt(mp_pose.PoseLandmark.LEFT_ELBOW)
                wr = _pose_world_pt(mp_pose.PoseLandmark.LEFT_WRIST)
                hip = _pose_world_pt(mp_pose.PoseLandmark.LEFT_HIP)
                wrist_px = _pose_px(mp_pose.PoseLandmark.LEFT_WRIST)
                arm_vis = float(
                    np.mean(
                        [
                            pose_lm[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility,
                            pose_lm[mp_pose.PoseLandmark.LEFT_ELBOW].visibility,
                            pose_lm[mp_pose.PoseLandmark.LEFT_WRIST].visibility,
                        ]
                    )
                )
                hand_lms = result.left_hand_landmarks
                elbow_img = _pose_img(mp_pose.PoseLandmark.LEFT_ELBOW)

            elbow_angle = _angle(sh, el, wr)
            shoulder_angle = _angle(hip, sh, el)
            wrist_height = float(np.dot((wr - hip_center), torso_unit) / torso_len)

            wrist_flexion = None
            index_pip_angle = None
            if hand_lms:
                try:
                    wrist = _hand_img_pt(hand_lms, mp_hands.HandLandmark.WRIST)
                    mid_mcp = _hand_img_pt(hand_lms, mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
                    # Wrist flexion proxy: 0=straight, higher=more "snap" (flexed).
                    wrist_flexion = float(180.0 - _angle(elbow_img, wrist, mid_mcp))

                    idx_mcp = _hand_img_pt(hand_lms, mp_hands.HandLandmark.INDEX_FINGER_MCP)
                    idx_pip = _hand_img_pt(hand_lms, mp_hands.HandLandmark.INDEX_FINGER_PIP)
                    idx_dip = _hand_img_pt(hand_lms, mp_hands.HandLandmark.INDEX_FINGER_DIP)
                    index_pip_angle = float(_angle(idx_mcp, idx_pip, idx_dip))
                except Exception:
                    wrist_flexion = None
                    index_pip_angle = None

            return _ArmObs(
                elbow_angle=float(elbow_angle),
                shoulder_angle=float(shoulder_angle),
                wrist_height=float(wrist_height),
                wrist_flexion=wrist_flexion,
                index_pip_angle=index_pip_angle,
                arm_visibility=arm_vis,
                hand_present=bool(hand_lms),
                wrist_px=wrist_px,
            )

        arms = {"right": _arm_obs("right"), "left": _arm_obs("left")}
        per_frame.append(
            (
                idx,
                idx * dt,
                visibility,
                keypoints_px,
                ball_positions[idx],
                arms,
                shoulder_width,
                wrist_gap,
                knee_angle,
                hip_angle,
            )
        )

    holistic.close()

    if not per_frame:
        return []

    normalized_side = _normalize_shooting_side(shooting_hand)
    logger.info(
        "Extracting pose metrics: shooting_hand=%s, normalized_side=%s",
        shooting_hand, normalized_side
    )
    _, release_scores, composite_scores = _score_shooting_sides(per_frame)
    shooting_side, was_flipped = _resolve_tracking_side(
        normalized_side,
        release_scores,
        composite_scores,
    )
    logger.info(
        "Final shooting_side=%s (was_flipped=%s), guide_side=%s",
        shooting_side, was_flipped, _opposite_side(shooting_side)
    )
    guide_side = _opposite_side(shooting_side)

    metrics: List[FrameMetrics] = []
    for (
        frame_index,
        t_s,
        vis,
        keypoints_px,
        ball,
        arms,
        shoulder_width,
        wrist_gap,
        knee_angle,
        hip_angle,
    ) in per_frame:
        shoot = arms[shooting_side]
        guide = arms[guide_side]
        if shoot.elbow_angle is None or shoot.shoulder_angle is None or shoot.wrist_height is None:
            continue
        offhand_gap = None
        if shoulder_width and wrist_gap is not None:
            offhand_gap = float(wrist_gap / max(shoulder_width, 1e-6))

        metrics.append(
            FrameMetrics(
                frame_index=frame_index,
                timestamp_s=t_s,
                knee_angle=float(knee_angle),
                hip_angle=float(hip_angle),
                elbow_angle=float(shoot.elbow_angle),
                shoulder_angle=float(shoot.shoulder_angle),
                wrist_height=float(shoot.wrist_height),
                offhand_gap=offhand_gap,
                visibility=float(vis),
                keypoints_px=keypoints_px,
                ball_pos=ball,
                shooting_side=shooting_side,
                wrist_flexion=shoot.wrist_flexion,
                index_pip_angle=shoot.index_pip_angle,
                hand_confidence=1.0 if shoot.hand_present else 0.0,
            )
        )

    # Wrist/finger velocities (deg/s) using finite differences on available spans.
    for i in range(1, len(metrics)):
        dt_s = metrics[i].timestamp_s - metrics[i - 1].timestamp_s
        dt_s = max(float(dt_s), 1e-6)
        if metrics[i].wrist_flexion is not None and metrics[i - 1].wrist_flexion is not None:
            metrics[i].wrist_flexion_vel = float(
                (metrics[i].wrist_flexion - metrics[i - 1].wrist_flexion) / dt_s
            )

    return metrics


def _mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else None


def _phase_summary(subset: List[FrameMetrics], name: str, min_dt: float) -> schemas.PhaseMetrics:
    angles: Dict[str, Optional[float]] = {
            "knee_flexion": _mean([m.knee_angle for m in subset]),
            "hip_flexion": _mean([m.hip_angle for m in subset]),
            "elbow_angle": _mean([m.elbow_angle for m in subset]),
            "shoulder_angle": _mean([m.shoulder_angle for m in subset]),
            "wrist_height": _mean([m.wrist_height for m in subset]),
        # Hand/finger kinematics (may be missing if hands are occluded).
        "wrist_flexion": _mean([m.wrist_flexion for m in subset]),
        "index_pip_angle": _mean([m.index_pip_angle for m in subset]),
    }
    vels = [m.wrist_flexion_vel for m in subset if m.wrist_flexion_vel is not None]
    if vels:
        angles["wrist_flexion_vel_peak"] = float(np.max(vels))

    return schemas.PhaseMetrics(
        name=name,
        angles=angles,
        timings={
            "duration": max(
                subset[-1].timestamp_s - subset[0].timestamp_s if len(subset) > 1 else 0.0,
                min_dt,
            )
        },
        confidence=_mean([m.visibility for m in subset]) or 0.0,
    )


def _segment_phases(metrics: List[FrameMetrics]) -> Tuple[List[schemas.PhaseMetrics], dict, float]:
    # Use ball height if available, else wrist
    def get_height(m: FrameMetrics) -> float:
        if m.ball_pos:
            # ball_pos is (cx, cy) pixels from top-left
            # smaller y = higher
            return -m.ball_pos[1] 
        return m.wrist_height

    # Identify key frames
    load_idx = int(np.argmin([m.knee_angle for m in metrics]))
    
    # Peak height (ball or wrist) after load: a rough "end of rise" anchor.
    after_load = metrics[load_idx:] or [metrics[load_idx]]
    peak_rel_idx = int(np.argmax([get_height(m) for m in after_load]))
    peak_idx = load_idx + peak_rel_idx

    # Set: most-bent elbow between load and peak.
    if peak_idx > load_idx:
        segment = metrics[load_idx : peak_idx + 1]
        set_rel_idx = int(np.argmin([m.elbow_angle for m in segment]))
        set_idx = load_idx + set_rel_idx
    else:
        set_idx = load_idx

    # Release moment: prefer peak wrist-flexion velocity (proxy for "snap") near the top.
    search_start = min(set_idx, peak_idx)
    search_end = max(set_idx, peak_idx)
    release_moment_idx = peak_idx
    vel_window = [
        (i, metrics[i].wrist_flexion_vel)
        for i in range(search_start, search_end + 1)
        if metrics[i].wrist_flexion_vel is not None
    ]
    if vel_window:
        release_moment_idx = int(max(vel_window, key=lambda t: t[1] or -1e9)[0])

    # Release is short; treat 1 frame (or 2 if available) as "release" window.
    follow_idx = len(metrics) - 1
    release_end_idx = min(release_moment_idx + 1, follow_idx)

    def slice_metrics(start: int, end: int) -> List[FrameMetrics]:
        return metrics[start : end + 1] if end >= start else [metrics[start]]

    min_dt = metrics[1].timestamp_s - metrics[0].timestamp_s if len(metrics) > 1 else 1 / 30.0
    phases = [
        _phase_summary(slice_metrics(0, load_idx), "load", min_dt),
        _phase_summary(slice_metrics(load_idx, set_idx), "set", min_dt),
        _phase_summary(slice_metrics(set_idx, release_moment_idx), "rise", min_dt),
        _phase_summary(slice_metrics(release_moment_idx, release_end_idx), "release", min_dt),
        _phase_summary(slice_metrics(release_end_idx, follow_idx), "follow_through", min_dt),
    ]
    boundaries = {
        "load_idx": load_idx,
        "set_idx": set_idx,
        "rise_idx": release_moment_idx,
        "release_idx": release_end_idx,
        "follow_idx": follow_idx,
    }
    return phases, boundaries, min_dt


def _issues_from_metrics(
    metrics: List[FrameMetrics],
    phases: List[schemas.PhaseMetrics],
    boundaries: dict,
) -> List[schemas.Issue]:
    issues: List[schemas.Issue] = []
    load_phase = next((p for p in phases if p.name == "load"), None)
    set_phase = next((p for p in phases if p.name == "set"), None)
    release_phase = next((p for p in phases if p.name == "release"), None)
    follow_phase = next((p for p in phases if p.name == "follow_through"), None)

    avg_vis = _mean([m.visibility for m in metrics]) or 0.5
    hand_rate = _mean([m.hand_confidence for m in metrics]) or 0.0
    hand_conf = float(np.clip(avg_vis * max(hand_rate, 0.1), 0.0, 1.0))

    load_idx = int(boundaries.get("load_idx", 0))
    set_idx = int(boundaries.get("set_idx", load_idx))
    rise_idx = int(boundaries.get("rise_idx", set_idx))
    release_end_idx = int(boundaries.get("release_idx", rise_idx))

    # Broad baseline bands (MVP). These are intentionally wide; tune with real pro data later.
    load_knee = load_phase.angles.get("knee_flexion") if load_phase else None
    if load_knee is not None and load_knee > 150:
        issues.append(
            schemas.Issue(
                name="Knee bend shallow",
                severity="medium",
                delta="Start the shot a little lower, then rise through it smoothly.",
                confidence=avg_vis,
                phase="load",
            )
        )

    # Set-point elbow angle (too tucked or too open can reduce repeatability).
    set_elbow = set_phase.angles.get("elbow_angle") if set_phase else None
    if set_elbow is not None and (set_elbow < 55 or set_elbow > 125):
        issues.append(
            schemas.Issue(
                name="Elbow angle at set is off-band",
                severity="low",
                delta="Bring your shooting elbow in slightly and keep the ball on a clean line.",
                confidence=avg_vis,
                phase="set",
            )
        )

    # Release height (body-relative, more view-robust than pixels).
    if release_phase and load_phase:
        wrist_gain = (release_phase.angles.get("wrist_height") or 0) - (load_phase.angles.get("wrist_height") or 0)
        if wrist_gain < 0.25:
            issues.append(
                schemas.Issue(
                    name="Low release height",
                    severity="medium",
                    delta="Finish taller and let the ball leave your hand higher.",
                    confidence=avg_vis,
                    phase="release",
                )
            )

    # Sequencing: legs load → legs start to extend → arm rises. Flag if arm rises before leg drive.
    knee0 = metrics[load_idx].knee_angle if 0 <= load_idx < len(metrics) else metrics[0].knee_angle
    wrist0 = metrics[load_idx].wrist_height if 0 <= load_idx < len(metrics) else metrics[0].wrist_height
    leg_drive_start = None
    arm_lift_start = None
    for i in range(load_idx, min(load_idx + 12, len(metrics))):
        if leg_drive_start is None and metrics[i].knee_angle > knee0 + 5:
            leg_drive_start = i
        if arm_lift_start is None and metrics[i].wrist_height > wrist0 + 0.10:
            arm_lift_start = i
    if leg_drive_start is not None and arm_lift_start is not None and arm_lift_start + 1 < leg_drive_start:
        issues.append(
            schemas.Issue(
                name="Rushed upper-body sequencing",
                severity="medium",
                delta="Let your legs start the shot, then bring the ball and shooting arm through.",
                confidence=avg_vis,
                phase="rise",
            )
        )

    # Guide-hand spacing near release (if wrists are trackable).
    rel_offhand = _mean([m.offhand_gap for m in metrics[rise_idx : release_end_idx + 1]])
    if rel_offhand is not None and rel_offhand < 0.25:
        issues.append(
            schemas.Issue(
                name="Off-hand too close",
                severity="low",
                delta="Keep the guide hand soft and let it leave the ball cleanly.",
                confidence=avg_vis,
                phase="release",
            )
        )

    # Wrist snap (requires hands): look for a clear positive peak in wrist-flexion velocity near release.
    release_vels = [
        m.wrist_flexion_vel
        for m in metrics[max(set_idx, 0) : release_end_idx + 1]
        if m.wrist_flexion_vel is not None
    ]
    if release_vels and float(np.max(release_vels)) < 80.0:
        issues.append(
            schemas.Issue(
                name="Wrist snap looks weak",
                severity="low",
                delta="Flick your wrist through the shot and let your fingers finish down.",
                confidence=hand_conf,
                phase="release",
            )
        )

    # Follow-through: hold wrist flexion briefly after release (hands required).
    if follow_phase:
        follow_dur = float(follow_phase.timings.get("duration", 0) or 0)
        flex_thr = 15.0
        hold_t = 0.0
        if 0 <= release_end_idx < len(metrics):
            t0 = metrics[release_end_idx].timestamp_s
            last_t = t0
            for i in range(release_end_idx, len(metrics)):
                if metrics[i].wrist_flexion is None or metrics[i].wrist_flexion < flex_thr:
                    break
                last_t = metrics[i].timestamp_s
            hold_t = max(0.0, last_t - t0)

        if follow_dur < 0.15 or (hold_t and hold_t < 0.12):
            issues.append(
                schemas.Issue(
                    name="Short follow-through",
                    severity="low",
                    delta="Hold your finish until the ball reaches the rim.",
                    confidence=float(np.clip(avg_vis * (0.5 + 0.5 * hand_rate), 0.0, 1.0)),
                    phase="follow_through",
                )
            )

        # Finger roll-through proxy (very rough): index finger less curled during/after release.
        idx_pip = follow_phase.angles.get("index_pip_angle")
        if idx_pip is not None and idx_pip < 140 and hand_rate >= 0.5:
            issues.append(
                schemas.Issue(
                    name="Finger roll-through looks limited",
                    severity="low",
                    delta="Let the ball roll off your fingers instead of pushing it.",
                    confidence=hand_conf,
                    phase="follow_through",
                )
            )

    return issues


def _sort_issues(issues: List[schemas.Issue]) -> List[schemas.Issue]:
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        issues,
        key=lambda issue: (
            severity_rank.get(issue.severity, 3),
            -(issue.confidence or 0.0),
            issue.name,
        ),
    )


def _prioritize_playback_issues(issues: List[schemas.Issue]) -> List[schemas.Issue]:
    sorted_issues = _sort_issues(issues)
    release_issues = sorted(
        (issue for issue in sorted_issues if issue.name in _RELEASE_CONTROL_ISSUE_PRIORITY),
        key=lambda issue: (
            _RELEASE_CONTROL_ISSUE_PRIORITY[issue.name],
            -(issue.confidence or 0.0),
            issue.name,
        ),
    )

    ordered: List[schemas.Issue] = []
    seen_names: set[str] = set()

    def push(issue: schemas.Issue) -> None:
        if issue.name in seen_names:
            return
        ordered.append(issue)
        seen_names.add(issue.name)

    if sorted_issues and sorted_issues[0].name not in _RELEASE_CONTROL_ISSUE_PRIORITY:
        push(sorted_issues[0])

    for issue in release_issues[:2]:
        push(issue)

    for issue in sorted_issues:
        push(issue)

    return ordered


def _prioritize_playback_traits(
    traits: List[schemas.BorrowTrait],
) -> List[schemas.BorrowTrait]:
    release_traits = sorted(
        (trait for trait in traits if trait.title in _RELEASE_CONTROL_COMPARE_PRIORITY),
        key=lambda trait: (
            _RELEASE_CONTROL_COMPARE_PRIORITY[trait.title],
            trait.title,
        ),
    )

    ordered: List[schemas.BorrowTrait] = []
    seen_titles: set[str] = set()

    def push(trait: schemas.BorrowTrait) -> None:
        if trait.title in seen_titles:
            return
        ordered.append(trait)
        seen_titles.add(trait.title)

    if traits and traits[0].title not in _RELEASE_CONTROL_COMPARE_PRIORITY:
        push(traits[0])

    for trait in release_traits[:2]:
        push(trait)

    for trait in traits:
        push(trait)

    return ordered


def _issue_names(issues: List[schemas.Issue]) -> set[str]:
    return {issue.name for issue in issues}


def _phase_lookup(phases: List[schemas.PhaseMetrics]) -> Dict[str, schemas.PhaseMetrics]:
    return {phase.name: phase for phase in phases}


def _build_strengths(
    metrics: List[FrameMetrics],
    phases: List[schemas.PhaseMetrics],
    issues: List[schemas.Issue],
) -> List[schemas.Strength]:
    strengths: List[schemas.Strength] = []
    issue_names = _issue_names(issues)
    phase_map = _phase_lookup(phases)
    hand_rate = _mean([m.hand_confidence for m in metrics]) or 0.0

    load_knee = phase_map.get("load").angles.get("knee_flexion") if phase_map.get("load") else None
    if load_knee is not None and load_knee <= 150 and "Knee bend shallow" not in issue_names:
        strengths.append(
            schemas.Strength(
                title="Balanced base",
                detail="You are starting the shot from a base that already looks fairly stable.",
            )
        )

    if "Rushed upper-body sequencing" not in issue_names:
        strengths.append(
            schemas.Strength(
                title="Connected timing",
                detail="Your lower body and upper body look reasonably connected through the rise.",
            )
        )

    if "Short follow-through" not in issue_names and hand_rate >= 0.5:
        strengths.append(
            schemas.Strength(
                title="Clean finish shape",
                detail="Your finish looks close to something you can repeat without overthinking it.",
            )
        )

    if not strengths:
        strengths.append(
            schemas.Strength(
                title="Usable read",
                detail="This clip gives enough signal to coach from, so the next step is sharpening the biggest habit first.",
            )
        )

    return strengths[:3]


def _build_comparison_drill(cue: str) -> str:
    return (
        "Use this as a comparison rep: take 5 slow makes, focus on this single cue, "
        "and ignore every other adjustment for the set."
    )


def _build_coaching_plan(
    issues: List[schemas.Issue],
    comparison_result: Optional[schemas.ComparisonResult] = None,
) -> List[schemas.CoachingStep]:
    coaching_steps: List[schemas.CoachingStep] = []
    sorted_issues = _sort_issues(issues)

    for idx, issue in enumerate(sorted_issues[:3]):
        guide = _ISSUE_GUIDANCE.get(issue.name)
        if guide is None:
            cue = issue.delta or "Make this part of the motion a little cleaner and more repeatable."
            coaching_steps.append(
                schemas.CoachingStep(
                    title=issue.name,
                    cue=cue,
                    why="This is one of the clearest opportunities to make the shot feel simpler.",
                    drill="Focus on just this cue for your next few reps instead of trying to fix everything at once.",
                    phase=issue.phase,
                    priority="now" if idx == 0 else "next",
                    confidence=issue.confidence,
                )
            )
            continue

        coaching_steps.append(
            schemas.CoachingStep(
                title=guide["title"],
                cue=guide["cue"],
                why=guide["why"],
                drill=guide["drill"],
                phase=issue.phase,
                priority="now" if idx == 0 else "next",
                confidence=issue.confidence,
            )
        )

    if coaching_steps:
        return coaching_steps

    primary_match = comparison_result.primary if comparison_result else None
    if primary_match and primary_match.borrow_traits:
        comparison_step = primary_match.borrow_traits[0]
        return [
            schemas.CoachingStep(
                title=f"Borrow from {primary_match.label}",
                cue=comparison_step.cue or comparison_step.detail,
                why="This is the clearest difference between your current shot and the reference style.",
                drill=_build_comparison_drill(comparison_step.cue or comparison_step.detail),
                phase=comparison_step.phase,
                priority="now",
                confidence=primary_match.confidence,
            )
        ]

    return [
        schemas.CoachingStep(
            title="Keep the same rhythm",
            cue="Your shot does not need a rebuild right now. Stay smooth and repeat the same motion.",
            why="When the base looks clean, the biggest win is usually repetition and confidence.",
            drill="Take a small set of relaxed makes and freeze the same finish every time.",
            phase=None,
            priority="now",
            confidence=0.8,
        )
    ]


def _build_research_notes(issues: List[schemas.Issue]) -> List[schemas.ResearchNote]:
    notes: List[schemas.ResearchNote] = list(_GENERIC_RESEARCH_NOTES)

    for issue in _sort_issues(issues):
        guide = _ISSUE_GUIDANCE.get(issue.name)
        if guide is None:
            continue
        title = guide.get("research_title")
        body = guide.get("research_body")
        if not title or not body:
            continue
        if any(note.title == title for note in notes):
            continue
        notes.append(schemas.ResearchNote(title=title, body=body))
        if len(notes) >= 3:
            break

    return notes[:3]


def _canonical_phase_name(phase: Optional[str]) -> Optional[str]:
    if not phase:
        return None
    normalized = phase.strip().lower().replace("-", "_").replace(" ", "_")
    return "follow_through" if normalized == "followthrough" else normalized


def _phase_anchor_index(boundaries: dict, phase: Optional[str]) -> int:
    normalized = _canonical_phase_name(phase) or "release"
    if normalized == "load":
        return int(boundaries.get("load_idx", 0))
    if normalized == "set":
        return int(boundaries.get("set_idx", boundaries.get("load_idx", 0)))
    if normalized == "rise":
        return int(boundaries.get("rise_idx", boundaries.get("set_idx", 0)))
    if normalized == "follow_through":
        release_idx = int(boundaries.get("release_idx", boundaries.get("rise_idx", 0)))
        return min(int(boundaries.get("follow_idx", release_idx)), release_idx + 1)
    return int(boundaries.get("release_idx", boundaries.get("rise_idx", 0)))


def _direction_for_side(direction: str, shooting_side: str) -> str:
    if direction == "toward_midline":
        return "left" if shooting_side == "right" else "right"
    if direction == "away_from_midline":
        return "right" if shooting_side == "right" else "left"
    return direction


def _resolve_target_candidates(metric: FrameMetrics, target_name: str) -> List[str]:
    shoot_prefix = "r" if metric.shooting_side == "right" else "l"
    guide_prefix = "l" if shoot_prefix == "r" else "r"
    target_map = {
        "left_knee": ["l_knee"],
        "right_knee": ["r_knee"],
        "shoot_elbow": [f"{shoot_prefix}_elbow"],
        "shoot_wrist": [f"{shoot_prefix}_wrist"],
        "shoot_shoulder": [f"{shoot_prefix}_shoulder"],
        "guide_wrist": [f"{guide_prefix}_wrist"],
        "shoot_index_tip": [
            f"{shoot_prefix}_hand_index_finger_tip",
            f"{shoot_prefix}_wrist",
        ],
        "shoot_middle_tip": [
            f"{shoot_prefix}_hand_middle_finger_tip",
            f"{shoot_prefix}_wrist",
        ],
    }
    return target_map.get(target_name, [])


def _normalized_overlay_point(
    point: Tuple[int, int], frame_shape: Tuple[int, int, int], rotate_output: bool
) -> Tuple[float, float]:
    frame_h, frame_w = frame_shape[:2]
    px = float(point[0])
    py = float(point[1])
    if rotate_output:
        return (
            float(np.clip(py / max(frame_h, 1), 0.0, 1.0)),
            float(np.clip(1.0 - (px / max(frame_w, 1)), 0.0, 1.0)),
        )
    return (
        float(np.clip(px / max(frame_w, 1), 0.0, 1.0)),
        float(np.clip(py / max(frame_h, 1), 0.0, 1.0)),
    )


def _bubble_offset_for_direction(direction: str) -> Tuple[float, float]:
    offsets = {
        "down": (0.0, -0.16),
        "up": (0.0, -0.18),
        "left": (0.14, -0.08),
        "right": (-0.14, -0.08),
        "down_left": (0.14, -0.14),
        "down_right": (-0.14, -0.14),
    }
    return offsets.get(direction, (0.0, -0.16))


def _style_trait_for_phase(
    comparison_result: Optional[schemas.ComparisonResult], phase: Optional[str]
) -> Optional[schemas.ComparisonTrait]:
    primary = comparison_result.primary if comparison_result else None
    if primary is None:
        return None
    canonical_phase = _canonical_phase_name(phase)
    for trait in primary.borrow_traits:
        if _canonical_phase_name(trait.phase) == canonical_phase:
            return trait
    return None


def _issue_research_basis(
    issue: schemas.Issue,
    comparison_result: Optional[schemas.ComparisonResult] = None,
) -> str:
    guide = _ISSUE_GUIDANCE.get(issue.name)
    basis = (
        guide.get("research_body")
        if guide
        else "This is the most important fix we found in your shot."
    )
    return basis


def _comparison_delta_lookup(
    match: schemas.ComparisonMatch,
) -> Dict[str, schemas.ComparisonMetricDelta]:
    return {delta.label: delta for delta in match.metric_deltas}


def _comparison_research_basis(
    label: str,
    match: schemas.ComparisonMatch,
    delta: Optional[schemas.ComparisonMetricDelta],
) -> str:
    base = _COMPARISON_RESEARCH_LINES.get(
        label,
        "This is one of the biggest improvements you can make to your shot.",
    )
    return base


def _format_measurement(value: Optional[float], unit: str) -> Optional[str]:
    if value is None:
        return None
    if unit == "deg":
        return f"{value:.0f} deg"
    if unit == "deg/s":
        return f"{value:.0f} deg/s"
    if unit == "s":
        return f"{value:.2f}s"
    if unit == "body":
        return f"{value:.2f} body lengths"
    return f"{value:.2f} {unit}".strip()


def _metric_relative_sentence(
    research_key: Optional[str],
    metric: FrameMetrics,
) -> Optional[str]:
    # Reference values from Cabarkapa et al. 2021 research
    PROFICIENT_KNEE_ANGLE = 104.3
    PROFICIENT_ELBOW_ANGLE = 58.4
    PROFICIENT_RELEASE_HEIGHT = 1.47
    
    if research_key == "lower_body_load" and metric.knee_angle is not None:
        diff = metric.knee_angle - PROFICIENT_KNEE_ANGLE
        if diff > 3:
            return f"Your knee angle is {metric.knee_angle:.0f}°. Proficient shooters average {PROFICIENT_KNEE_ANGLE:.0f}°. Bend {diff:.0f}° deeper."
        return None
    
    if research_key == "sequencing_lift" and metric.shoulder_angle is not None:
        return f"Your shoulder is at {metric.shoulder_angle:.0f}°. Your arms are moving before your legs finish pushing."
    
    if research_key == "release_height" and metric.wrist_height is not None:
        diff_pct = ((PROFICIENT_RELEASE_HEIGHT - metric.wrist_height) / max(metric.wrist_height, 0.01)) * 100
        if diff_pct > 3:
            return f"Your release height is {metric.wrist_height:.2f}x your height. Proficient shooters release at {PROFICIENT_RELEASE_HEIGHT:.2f}x. Release {diff_pct:.0f}% higher."
        return None
    
    if research_key == "elbow_line" and metric.elbow_angle is not None:
        diff = metric.elbow_angle - PROFICIENT_ELBOW_ANGLE
        if diff > 3:
            return f"Your elbow angle is {metric.elbow_angle:.0f}°. Proficient shooters average {PROFICIENT_ELBOW_ANGLE:.0f}°. Tuck {diff:.0f}° tighter."
        return None
    
    if research_key == "release_control" and metric.wrist_flexion is not None:
        return f"Your wrist snap is {metric.wrist_flexion:.0f}°. A firm, consistent snap in the final 0.01s determines accuracy."
    
    return None


def _comparison_relative_sentence(
    compare_label: Optional[str],
    delta: Optional[schemas.ComparisonMetricDelta],
    match: Optional[schemas.ComparisonMatch],
) -> Optional[str]:
    if delta is None or match is None or compare_label is None:
        return None

    user_value = _format_measurement(delta.user_value, delta.unit)
    reference_value = _format_measurement(delta.reference_value, delta.unit)
    if user_value is None or reference_value is None:
        return None

    relation_map = {
        "Load depth": {
            "higher": "more upright and less loaded than",
            "lower": "deeper than",
            "aligned": "very close to",
        },
        "Set position": {
            "higher": "more open than",
            "lower": "more tucked than",
            "aligned": "very close to",
        },
        "Arm lift timing": {
            "higher": "rising earlier than",
            "lower": "lagging a bit behind",
            "aligned": "very close to",
        },
        "Release height": {
            "higher": "a little taller than",
            "lower": "a little lower than",
            "aligned": "very close to",
        },
        "Release lift": {
            "higher": "lifting higher than",
            "lower": "not getting as tall as",
            "aligned": "very close to",
        },
        "Wrist finish": {
            "higher": "snapping harder than",
            "lower": "finishing softer than",
            "aligned": "very close to",
        },
        "Finish hold": {
            "higher": "holding longer than",
            "lower": "holding shorter than",
            "aligned": "very close to",
        },
        "Finger roll-through": {
            "higher": "finishing cleaner than",
            "lower": "showing less fingertip finish than",
            "aligned": "very close to",
        },
    }
    relation = relation_map.get(compare_label, {}).get(delta.direction, "different from")
    measure_label = {
        "Load depth": "load angle",
        "Set position": "set angle",
        "Arm lift timing": "rise angle",
        "Release height": "release height",
        "Release lift": "release lift",
        "Wrist finish": "wrist finish speed",
        "Finish hold": "finish hold",
        "Finger roll-through": "finger finish",
    }.get(compare_label, compare_label.lower())

    return (
        f"In this rep your {measure_label} is {user_value}; "
        f"your closest successful reference, {match.label}, sits around {reference_value}, "
        f"so you are {relation} that reference."
    )


def _relative_deep_dive(
    *,
    research_key: Optional[str],
    metric: FrameMetrics,
    compare_label: Optional[str],
    comparison_delta: Optional[schemas.ComparisonMetricDelta],
    match: Optional[schemas.ComparisonMatch],
) -> Optional[schemas.PlaybackDeepDive]:
    base = research_bank.lookup_deep_dive(research_key)
    if base is None:
        return None

    # Build relative stats based on user's actual measurements vs research data
    stats: List[schemas.ResearchStat] = []
    
    # Reference values from research (Cabarkapa et al. 2021)
    PROFICIENT_KNEE_ANGLE = 104.3  # degrees (3pt shooters)
    PROFICIENT_HIP_ANGLE = 134.6   # degrees (3pt shooters)
    PROFICIENT_ELBOW_ANGLE = 58.4  # degrees (3pt shooters)
    PROFICIENT_RELEASE_HEIGHT = 1.47  # body ratio (3pt shooters)
    
    if research_key == "lower_body_load":
        if metric.knee_angle is not None:
            diff = metric.knee_angle - PROFICIENT_KNEE_ANGLE
            if diff > 3:  # Only show if meaningfully different
                stats.append(schemas.ResearchStat(
                    label=f"{metric.knee_angle:.0f}°",
                    value="Your knee angle",
                    detail=f"Proficient shooters bend {abs(diff):.0f}° deeper than you. Aim for {PROFICIENT_KNEE_ANGLE:.0f}° or lower.",
                ))
        if metric.hip_angle is not None:
            diff = PROFICIENT_HIP_ANGLE - metric.hip_angle
            if abs(diff) > 5:
                direction = "more" if diff > 0 else "less"
                stats.append(schemas.ResearchStat(
                    label=f"{metric.hip_angle:.0f}°",
                    value="Your hip angle",
                    detail=f"Proficient shooters have {abs(diff):.0f}° {direction} hip flexion. Target: {PROFICIENT_HIP_ANGLE:.0f}°.",
                ))
    
    elif research_key == "release_height":
        if metric.wrist_height is not None:
            diff_pct = ((PROFICIENT_RELEASE_HEIGHT - metric.wrist_height) / max(metric.wrist_height, 0.01)) * 100
            if diff_pct > 3:  # Only show if meaningfully lower
                stats.append(schemas.ResearchStat(
                    label=f"{diff_pct:.0f}%",
                    value="Lower than proficient",
                    detail=f"Your release: {metric.wrist_height:.2f}x height. Proficient shooters: {PROFICIENT_RELEASE_HEIGHT:.2f}x. Release {diff_pct:.0f}% higher.",
                ))
                stats.append(schemas.ResearchStat(
                    label="45°",
                    value="optimal entry angle",
                    detail="Research shows 45° arc makes 11-12% more shots than flat or high arcs (Noah/NASA).",
                ))
    
    elif research_key == "elbow_line":
        if metric.elbow_angle is not None:
            diff = metric.elbow_angle - PROFICIENT_ELBOW_ANGLE
            if diff > 3:  # Elbow too wide
                stats.append(schemas.ResearchStat(
                    label=f"{diff:.0f}°",
                    value="Wider than optimal",
                    detail=f"Your elbow: {metric.elbow_angle:.0f}°. Proficient shooters: {PROFICIENT_ELBOW_ANGLE:.0f}°. Tuck {diff:.0f}° tighter.",
                ))
                stats.append(schemas.ResearchStat(
                    label="ES=2.40",
                    value="training effect",
                    detail="Elbow training produced very large 3pt improvements (BMC Sports Science 2025).",
                ))
    
    elif research_key == "sequencing_lift":
        if metric.shoulder_angle is not None:
            stats.append(schemas.ResearchStat(
                label=f"{metric.shoulder_angle:.0f}°",
                value="Your shoulder lift",
                detail="Your arms are moving before your legs finish. Proficient shooters have higher hip velocity (g > 1.146).",
            ))
        stats.append(schemas.ResearchStat(
            label="24%",
            value="velocity increase FT→3pt",
            detail="This power comes from legs. Skilled shooters maintain consistent release despite 24% more velocity.",
        ))
    
    elif research_key == "release_control":
        stats.append(schemas.ResearchStat(
            label="r=-0.96",
            value="velocity SD vs accuracy",
            detail="Release velocity consistency is the #1 predictor of 3pt accuracy (Slegers et al. 2021).",
        ))
        if metric.wrist_flexion is not None:
            stats.append(schemas.ResearchStat(
                label=f"{metric.wrist_flexion:.0f}°",
                value="Your wrist snap",
                detail="The final 0.01s before release determines accuracy. Consistent wrist snap is critical.",
            ))

    return schemas.PlaybackDeepDive(
        summary=base.summary,
        stats=stats,
        sources=base.sources,
    )


def _build_playback_cue_from_spec(
    *,
    cue_id: str,
    title: str,
    cue: str,
    why: str,
    research_basis: str,
    deep_dive: Optional[schemas.PlaybackDeepDive],
    phase: Optional[str],
    confidence: float,
    metric: FrameMetrics,
    frame_shape: Tuple[int, int, int],
    rotate_output: bool,
    spec: Dict[str, object],
) -> Optional[schemas.PlaybackCue]:
    target_names = list(spec.get("targets", []))
    directions = list(spec.get("directions", []))
    motion_types = list(spec.get("motion_types", ["linear"] * len(target_names)))
    magnitudes = list(spec.get("magnitudes", [1.0] * len(target_names)))
    pivot_targets = list(spec.get("pivot_targets", [None] * len(target_names)))
    arc_radii = list(spec.get("arc_radii", [None] * len(target_names)))
    anchors: List[schemas.PlaybackAnchor] = []

    for i, (target_name, direction_name) in enumerate(zip(target_names, directions)):
        direction = _direction_for_side(str(direction_name), metric.shooting_side)
        motion_type = motion_types[i] if i < len(motion_types) else "linear"
        magnitude = magnitudes[i] if i < len(magnitudes) else 1.0
        pivot_target = pivot_targets[i] if i < len(pivot_targets) else None
        arc_radius = arc_radii[i] if i < len(arc_radii) else None
        
        candidates = _resolve_target_candidates(metric, str(target_name))
        actual_point: Optional[Tuple[int, int]] = None
        for candidate in candidates:
            actual_point = metric.keypoints_px.get(candidate)
            if actual_point is not None:
                break
        if actual_point is None:
            continue

        norm_x, norm_y = _normalized_overlay_point(actual_point, frame_shape, rotate_output)
        
        pivot_x: Optional[float] = None
        pivot_y: Optional[float] = None
        if pivot_target is not None:
            pivot_candidates = _resolve_target_candidates(metric, str(pivot_target))
            for candidate in pivot_candidates:
                pivot_point = metric.keypoints_px.get(candidate)
                if pivot_point is not None:
                    pivot_x, pivot_y = _normalized_overlay_point(pivot_point, frame_shape, rotate_output)
                    break
        
        anchors.append(
            schemas.PlaybackAnchor(
                x=norm_x,
                y=norm_y,
                direction=direction,
                motion_type=str(motion_type),
                pivot_x=pivot_x,
                pivot_y=pivot_y,
                arc_radius=float(arc_radius) if arc_radius is not None else None,
                magnitude=float(magnitude),
            )
        )

    if not anchors:
        return None

    bubble_x = float(np.mean([anchor.x for anchor in anchors]))
    bubble_y = float(np.mean([anchor.y for anchor in anchors]))
    override_offset = spec.get("bubble_offset")
    if isinstance(override_offset, tuple):
        offset_x, offset_y = override_offset
    else:
        offset_x, offset_y = _bubble_offset_for_direction(anchors[0].direction)

    bubble_x = float(np.clip(bubble_x + float(offset_x), 0.14, 0.86))
    bubble_y = float(np.clip(bubble_y + float(offset_y), 0.12, 0.88))

    return schemas.PlaybackCue(
        id=cue_id,
        title=title,
        cue=cue,
        why=why,
        research_basis=research_basis,
        deep_dive=deep_dive,
        phase=_canonical_phase_name(phase),
        timestamp=max(metric.timestamp_s, 0.0),
        bubble_x=bubble_x,
        bubble_y=bubble_y,
        anchors=anchors,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
    )


def _build_playback_cues(
    metrics: List[FrameMetrics],
    boundaries: dict,
    issues: List[schemas.Issue],
    frames: List[np.ndarray],
    comparison_result: Optional[schemas.ComparisonResult] = None,
) -> List[schemas.PlaybackCue]:
    if not metrics or not frames:
        return []

    rotate_output = frames[0].shape[1] > frames[0].shape[0]
    frame_shape = frames[0].shape
    cues: List[schemas.PlaybackCue] = []
    seen_titles: set[str] = set()
    primary = comparison_result.primary if comparison_result else None
    delta_lookup = _comparison_delta_lookup(primary) if primary is not None else {}

    for issue in _prioritize_playback_issues(issues):
        spec = _ISSUE_PLAYBACK_SPECS.get(issue.name)
        if spec is None:
            continue
        phase = _canonical_phase_name(issue.phase)
        metric_idx = _phase_anchor_index(boundaries, phase)
        metric_idx = max(0, min(metric_idx, len(metrics) - 1))
        metric = metrics[metric_idx]
        guide = _ISSUE_GUIDANCE.get(issue.name, {})
        title = str(spec.get("title") or guide.get("title") or issue.name)
        if title in seen_titles:
            continue
        cue = str(guide.get("cue") or issue.delta or issue.name)
        why = str(guide.get("why") or "This change should make the shot easier to repeat.")
        research_key = spec.get("research_key")
        compare_label = spec.get("compare_label")
        comparison_delta = (
            delta_lookup.get(str(compare_label)) if compare_label is not None else None
        )
        playback_cue = _build_playback_cue_from_spec(
            cue_id=f"issue-{len(cues) + 1}",
            title=title,
            cue=cue,
            why=why,
            research_basis=_issue_research_basis(issue, comparison_result=comparison_result),
            deep_dive=_relative_deep_dive(
                research_key=str(research_key) if research_key is not None else None,
                metric=metric,
                compare_label=str(compare_label) if compare_label is not None else None,
                comparison_delta=comparison_delta,
                match=primary,
            ),
            phase=phase,
            confidence=issue.confidence,
            metric=metric,
            frame_shape=frame_shape,
            rotate_output=rotate_output,
            spec=spec,
        )
        if playback_cue is None:
            continue
        cues.append(playback_cue)
        seen_titles.add(title)
        if len(cues) >= 3:
            return cues

    if primary is None:
        return cues

    for trait in _prioritize_playback_traits(primary.borrow_traits):
        spec = _COMPARISON_PLAYBACK_SPECS.get(trait.title)
        if spec is None:
            continue
        title = str(spec.get("title") or trait.title)
        if title in seen_titles:
            continue
        phase = _canonical_phase_name(spec.get("phase") or trait.phase)
        metric_idx = _phase_anchor_index(boundaries, phase)
        metric_idx = max(0, min(metric_idx, len(metrics) - 1))
        metric = metrics[metric_idx]
        delta = delta_lookup.get(trait.title)
        research_key = spec.get("research_key")
        compare_label = spec.get("compare_label") or trait.title
        playback_cue = _build_playback_cue_from_spec(
            cue_id=f"comparison-{len(cues) + 1}",
            title=title,
            cue=str(trait.cue or trait.detail),
            why=f"This is the clearest difference between your shot and the reference style.",
            research_basis=_comparison_research_basis(trait.title, primary, delta),
            deep_dive=_relative_deep_dive(
                research_key=str(research_key) if research_key is not None else None,
                metric=metric,
                compare_label=str(compare_label) if compare_label is not None else None,
                comparison_delta=delta,
                match=primary,
            ),
            phase=phase,
            confidence=primary.confidence,
            metric=metric,
            frame_shape=frame_shape,
            rotate_output=rotate_output,
            spec=spec,
        )
        if playback_cue is None:
            continue
        cues.append(playback_cue)
        seen_titles.add(title)
        if len(cues) >= 3:
            break

    return cues


def _tracking_quality_label(confidence_notes: List[str], metrics: List[FrameMetrics]) -> str:
    avg_vis = _mean([m.visibility for m in metrics]) or 0.0
    hand_rate = _mean([m.hand_confidence for m in metrics]) or 0.0
    if avg_vis >= 0.8 and hand_rate >= 0.5:
        return "High confidence read"
    if avg_vis >= 0.6:
        return "Good confidence read"
    if confidence_notes:
        return "Use this as a rough read"
    return "Limited confidence read"


def _summary_label(issues: List[schemas.Issue]) -> str:
    medium_or_high = sum(1 for issue in issues if issue.severity in {"medium", "high"})
    if not issues:
        return "Strong foundation"
    if medium_or_high == 0 and len(issues) <= 2:
        return "Small tune-up"
    if medium_or_high <= 1:
        return "Promising base"
    return "Start with the fundamentals"


def _build_summary(
    metrics: List[FrameMetrics],
    issues: List[schemas.Issue],
    coaching_plan: List[schemas.CoachingStep],
    confidence_notes: List[str],
    comparison_result: Optional[schemas.ComparisonResult] = None,
) -> schemas.AnalysisSummary:
    label = _summary_label(issues)
    tracking_quality = _tracking_quality_label(confidence_notes, metrics)

    if not issues:
        primary_match = comparison_result.primary if comparison_result else None
        if primary_match and primary_match.borrow_traits:
            headline = (
                f"Your shot looks solid overall. The best next step is borrowing one clean cue "
                f"from the {primary_match.label.lower()} style."
            )
            encouragement = (
                "You do not need a rebuild. Use the comparison as a tune-up so the shot gets "
                "cleaner without feeling different."
            )
        else:
            headline = "Your shot looks clean overall. The best next step is simply repeating the same rhythm."
            encouragement = "This looks closer to a refinement job than a full mechanics rebuild."
        start_here = coaching_plan[0].cue
        return schemas.AnalysisSummary(
            label=label,
            headline=headline,
            start_here=start_here,
            encouragement=encouragement,
            tracking_quality=tracking_quality,
        )

    top_step = coaching_plan[0]
    headline = f"Your biggest win right now is to {top_step.cue[:-1].lower() if top_step.cue.endswith('.') else top_step.cue.lower()}."
    encouragement = "Focus on one cue at a time. Small, repeatable changes usually beat trying to fix everything in one session."

    return schemas.AnalysisSummary(
        label=label,
        headline=headline[0].upper() + headline[1:],
        start_here=top_step.cue,
        encouragement=encouragement,
        tracking_quality=tracking_quality,
    )


def _label_frames(metrics: List[FrameMetrics], boundaries: dict) -> List[str]:
    labels: List[str] = []
    for idx, _ in enumerate(metrics):
        if idx <= boundaries["load_idx"]:
            labels.append("load")
        elif idx <= boundaries["set_idx"]:
            labels.append("set")
        elif idx <= boundaries.get("rise_idx", boundaries["set_idx"]):
            labels.append("rise")
        elif idx <= boundaries["release_idx"]:
            labels.append("release")
        else:
            labels.append("follow_through")
    return labels


_PHASE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "load": (255, 165, 0),
    "set": (0, 255, 255),
    "rise": (80, 160, 255),
    "release": (0, 200, 0),
    "follow_through": (200, 0, 200),
}


def _draw_skeleton_on_frame(
    frame: np.ndarray,
    m: FrameMetrics,
    phase: str,
    phase_issues: List[schemas.Issue] | None = None,
    show_phase_label: bool = True,
) -> np.ndarray:
    """Draw pose + hand skeleton overlay on a single frame. Returns the annotated frame."""
    base_color = _PHASE_COLORS.get(phase, (255, 255, 255))
    kp = m.keypoints_px
    skeleton_layer = frame.copy()

    def line(a: str, b: str):
        if a in kp and b in kp:
            cv2.line(skeleton_layer, kp[a], kp[b], base_color, 4)

    def dot(a: str):
        if a in kp:
            cv2.circle(skeleton_layer, kp[a], 8, base_color, -1)
            cv2.circle(skeleton_layer, kp[a], 10, (0, 0, 0), 2)

    def hand_line(prefix: str, a: str, b: str, color: Tuple[int, int, int], thickness: int = 2):
        ka = f"{prefix}_hand_{a}"
        kb = f"{prefix}_hand_{b}"
        if ka in kp and kb in kp:
            cv2.line(skeleton_layer, kp[ka], kp[kb], color, thickness)

    def hand_dot(prefix: str, a: str, color: Tuple[int, int, int], radius: int = 4):
        ka = f"{prefix}_hand_{a}"
        if ka in kp:
            cv2.circle(skeleton_layer, kp[ka], radius, color, -1)

    # Body skeleton
    line("r_shoulder", "r_elbow")
    line("r_elbow", "r_wrist")
    line("r_shoulder", "r_hip")
    line("r_hip", "r_knee")
    line("r_knee", "r_ankle")
    line("l_shoulder", "l_elbow")
    line("l_elbow", "l_wrist")
    line("l_shoulder", "l_hip")
    line("l_hip", "l_knee")
    line("l_knee", "l_ankle")
    line("r_shoulder", "l_shoulder")
    line("r_hip", "l_hip")
    line("nose", "r_eye")
    line("nose", "l_eye")

    for name in [
        "r_shoulder", "r_elbow", "r_wrist", "r_hip", "r_knee", "r_ankle",
        "l_shoulder", "l_elbow", "l_wrist", "l_hip", "l_knee", "l_ankle",
        "nose", "r_eye", "l_eye", "r_ear", "l_ear",
    ]:
        dot(name)

    # Hands
    shoot_prefix = "r" if m.shooting_side == "right" else "l"
    guide_prefix = "l" if shoot_prefix == "r" else "r"
    shoot_color = base_color
    guide_color = (150, 150, 150)
    for a, b in _HAND_CONNECTIONS:
        hand_line(shoot_prefix, a, b, shoot_color, thickness=3)
        hand_line(guide_prefix, a, b, guide_color, thickness=2)
    for finger_tip in ["wrist", "thumb_tip", "index_finger_tip", "middle_finger_tip", "ring_finger_tip", "pinky_tip"]:
        hand_dot(shoot_prefix, finger_tip, shoot_color, radius=5)
        hand_dot(guide_prefix, finger_tip, guide_color, radius=4)

    # Ball
    if m.ball_pos:
        cv2.circle(skeleton_layer, m.ball_pos, 15, (0, 0, 255), 3)
        cv2.circle(skeleton_layer, m.ball_pos, 5, (0, 0, 255), -1)

    cv2.addWeighted(skeleton_layer, 0.58, frame, 0.42, 0, frame)

    # Phase label overlay
    if show_phase_label:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, phase.upper(), (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, base_color, 2, cv2.LINE_AA)
        if phase_issues:
            issue_text = phase_issues[0].name
            if len(phase_issues) > 1:
                issue_text += f" (+{len(phase_issues) - 1})"
            cv2.putText(frame, f"! {issue_text}", (180, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1, cv2.LINE_AA)

    return frame


def _encode_overlay_video(
    frames: List[np.ndarray],
    metrics: List[FrameMetrics],
    labels: List[str],
    issues: List[schemas.Issue],
    fps: float,
) -> Optional[str]:
    """Encode an MP4 video with skeleton overlays on every frame. Returns base64 data URL or None."""
    import tempfile
    import os
    import subprocess
    import shutil

    if not frames:
        return None

    rotate_output = frames[0].shape[1] > frames[0].shape[0]
    sample_frame = frames[0]
    if rotate_output:
        sample_frame = cv2.rotate(sample_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = sample_frame.shape[:2]

    # Build issue lookup by phase
    issue_by_phase: Dict[str, List[schemas.Issue]] = {}
    for iss in issues:
        if iss.phase:
            issue_by_phase.setdefault(iss.phase, []).append(iss)

    # First write with OpenCV (mp4v), then re-encode with ffmpeg for browser compatibility
    tmp_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_raw.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_raw.name, fourcc, fps, (w, h))

    for i, frame in enumerate(frames):
        annotated = frame.copy()
        if i < len(metrics):
            m = metrics[i]
            phase = labels[i] if i < len(labels) else "unknown"
            phase_issues = issue_by_phase.get(phase, [])
            annotated = _draw_skeleton_on_frame(annotated, m, phase, phase_issues, show_phase_label=True)
        if rotate_output:
            annotated = cv2.rotate(annotated, cv2.ROTATE_90_COUNTERCLOCKWISE)
        out.write(annotated)

    out.release()

    # Re-encode to H.264 with ffmpeg for browser compatibility
    tmp_h264 = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_h264.close()

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            subprocess.run(
                [
                    ffmpeg_path,
                    "-y",
                    "-i", tmp_raw.name,
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "28",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    tmp_h264.name,
                ],
                check=True,
                capture_output=True,
            )
            output_path = tmp_h264.name
        except subprocess.CalledProcessError:
            # ffmpeg failed, fall back to raw mp4v
            output_path = tmp_raw.name
    else:
        # No ffmpeg, use raw (may not play in browser)
        output_path = tmp_raw.name

    # Read back and encode
    with open(output_path, "rb") as f:
        video_bytes = f.read()

    # Cleanup
    try:
        os.unlink(tmp_raw.name)
    except OSError:
        pass
    try:
        os.unlink(tmp_h264.name)
    except OSError:
        pass

    b64 = base64.b64encode(video_bytes).decode("ascii")
    return f"data:video/mp4;base64,{b64}"


def _keyframe_images(
    frames: List[np.ndarray],
    metrics: List[FrameMetrics],
    labels: List[str],
    issues: List[schemas.Issue],
) -> List[schemas.KeyframePreview]:

    rotate_output = frames[0].shape[1] > frames[0].shape[0]
    keyframes: List[schemas.KeyframePreview] = []
    phase_to_indices: Dict[str, List[int]] = {}
    for idx, lbl in enumerate(labels):
        phase_to_indices.setdefault(lbl, []).append(idx)

    phase_order = ["load", "set", "rise", "release", "follow_through"]
    for phase in phase_order:
        if phase not in phase_to_indices:
            continue
        idxs = phase_to_indices[phase]
        mid = idxs[len(idxs) // 2]
        frame = frames[mid].copy()
        skeleton_layer = frame.copy()
        m = metrics[mid]
        
        base_color = _PHASE_COLORS.get(phase, (255, 255, 255))

        phase_issues = [i for i in issues if i.phase == phase]
        kp = m.keypoints_px

        def line(a: str, b: str):
            if a in kp and b in kp:
                cv2.line(skeleton_layer, kp[a], kp[b], base_color, 4)

        def dot(a: str):
            if a in kp:
                cv2.circle(skeleton_layer, kp[a], 8, base_color, -1)
                cv2.circle(skeleton_layer, kp[a], 10, (0, 0, 0), 2)  # outline

        def hand_line(prefix: str, a: str, b: str, color: Tuple[int, int, int], thickness: int = 2):
            ka = f"{prefix}_hand_{a}"
            kb = f"{prefix}_hand_{b}"
            if ka in kp and kb in kp:
                cv2.line(skeleton_layer, kp[ka], kp[kb], color, thickness)

        def hand_dot(prefix: str, a: str, color: Tuple[int, int, int], radius: int = 4):
            ka = f"{prefix}_hand_{a}"
            if ka in kp:
                cv2.circle(skeleton_layer, kp[ka], radius, color, -1)

        # Full skeleton
        line("r_shoulder", "r_elbow")
        line("r_elbow", "r_wrist")
        line("r_shoulder", "r_hip")
        line("r_hip", "r_knee")
        line("r_knee", "r_ankle")
        
        line("l_shoulder", "l_elbow")
        line("l_elbow", "l_wrist")
        line("l_shoulder", "l_hip")
        line("l_hip", "l_knee")
        line("l_knee", "l_ankle")
        
        line("r_shoulder", "l_shoulder")
        line("r_hip", "l_hip")
        line("nose", "r_eye")
        line("nose", "l_eye")

        # Body keypoints (avoid dotting every hand landmark; it's too noisy visually)
        for name in [
            "r_shoulder",
            "r_elbow",
            "r_wrist",
            "r_hip",
            "r_knee",
            "r_ankle",
            "l_shoulder",
            "l_elbow",
            "l_wrist",
            "l_hip",
            "l_knee",
            "l_ankle",
            "nose",
            "r_eye",
            "l_eye",
            "r_ear",
            "l_ear",
        ]:
            dot(name)

        # Hands (shooting hand emphasized)
        shoot_prefix = "r" if m.shooting_side == "right" else "l"
        guide_prefix = "l" if shoot_prefix == "r" else "r"
        shoot_color = base_color
        guide_color = (150, 150, 150)
        for a, b in _HAND_CONNECTIONS:
            hand_line(shoot_prefix, a, b, shoot_color, thickness=3)
            hand_line(guide_prefix, a, b, guide_color, thickness=2)
        for finger_tip in ["wrist", "thumb_tip", "index_finger_tip", "middle_finger_tip", "ring_finger_tip", "pinky_tip"]:
            hand_dot(shoot_prefix, finger_tip, shoot_color, radius=5)
            hand_dot(guide_prefix, finger_tip, guide_color, radius=4)

        # Ball
        if m.ball_pos:
            cv2.circle(skeleton_layer, m.ball_pos, 15, (0, 0, 255), 3)
            cv2.circle(skeleton_layer, m.ball_pos, 5, (0, 0, 255), -1)

        cv2.addWeighted(skeleton_layer, 0.58, frame, 0.42, 0, frame)

        if rotate_output:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Draw text overlay
        h, w, _ = frame.shape
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 80), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(
            frame,
            f"{phase.upper()}",
            (20, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        
        if phase_issues:
            issue_text = phase_issues[0].name
            if len(phase_issues) > 1:
                issue_text += f" (+{len(phase_issues)-1})"
            cv2.putText(
                frame,
                f"! {issue_text}",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (100, 100, 255),  # reddish
                1,
                cv2.LINE_AA,
            )

        _, buf = cv2.imencode(".png", frame)
        keyframes.append(
            schemas.KeyframePreview(
                phase=phase,
                image_b64="data:image/png;base64," + base64.b64encode(buf).decode("ascii"),
            )
        )
    return keyframes


def _boundaries_from_indices(metrics: List[FrameMetrics], boundaries: dict, min_dt: float) -> List[schemas.PhaseBoundary]:
    indices = [
        ("load", 0, boundaries["load_idx"]),
        ("set", boundaries["load_idx"], boundaries["set_idx"]),
        ("rise", boundaries["set_idx"], boundaries.get("rise_idx", boundaries["set_idx"])),
        ("release", boundaries.get("rise_idx", boundaries["set_idx"]), boundaries["release_idx"]),
        ("follow_through", boundaries["release_idx"], boundaries["follow_idx"]),
    ]
    phase_bounds: List[schemas.PhaseBoundary] = []
    for name, start_i, end_i in indices:
        start_t = metrics[start_i].timestamp_s
        end_t = metrics[end_i].timestamp_s
        if end_t < start_t:
            end_t = start_t + min_dt
        phase_bounds.append(schemas.PhaseBoundary(phase=name, start=start_t, end=end_t))
    return phase_bounds


def _build_confidence_notes(
    metrics: List[FrameMetrics],
    frames: List[np.ndarray],
    shooting_hand: str | None = None,
) -> List[str]:
    confidence_notes: List[str] = []
    avg_vis = _mean([m.visibility for m in metrics]) or 0.0
    if avg_vis < 0.6:
        confidence_notes.append("Tracking was a little shaky, so treat this as a rough read instead of a perfect diagnosis.")
    if len(metrics) < max(5, len(frames) // 4):
        confidence_notes.append("A lot of frames were hard to track. A cleaner side or front angle will help.")
    hand_rate = _mean([m.hand_confidence for m in metrics]) or 0.0
    if hand_rate < 0.5:
        confidence_notes.append("Your hand was not visible the whole time, so wrist and follow-through notes are less certain.")
    if metrics:
        selected_side = _normalize_shooting_side(shooting_hand)
        if selected_side is not None:
            tracked_side = metrics[0].shooting_side
            if tracked_side == selected_side:
                confidence_notes.append(
                    f"You marked this as a {selected_side}-hand shot, so elbow, wrist, and follow-through cues stay locked to that side."
                )
            else:
                confidence_notes.append(
                    f"You marked this as a {selected_side}-hand shot, and the landmark stream looked mirrored, so we flipped the tracked side to stay on your shooting hand."
                )
        else:
            confidence_notes.append(f"We read this clip as a {metrics[0].shooting_side}-hand shot.")
    return confidence_notes


def _prepare_analysis(
    file_bytes: bytes,
    shot_type: str | None = None,
    shooting_hand: str | None = None,
) -> PreparedAnalysis:
    normalized_hand = _normalize_shooting_side(shooting_hand)
    cache_key = _cache_key(file_bytes, shot_type, normalized_hand)
    cached = _get_cached_prepared(cache_key)
    if cached is not None:
        return cached

    frames, fps, stride = decode_video(file_bytes, sample_every_n=2)
    if not frames:
        raise ValueError("Could not decode video or video empty.")

    metrics = _extract_pose_metrics(frames, fps, stride, shooting_hand=normalized_hand)
    if len(metrics) < 3:
        raise ValueError("Insufficient pose detections; recapture with clearer framing.")

    phases, boundaries, min_dt = _segment_phases(metrics)
    labels = _label_frames(metrics, boundaries)
    issues = _issues_from_metrics(metrics, phases, boundaries)
    confidence_notes = _build_confidence_notes(metrics, frames, shooting_hand=normalized_hand)

    prepared = PreparedAnalysis(
        frames=frames,
        fps=fps,
        stride=stride,
        metrics=metrics,
        phases=phases,
        boundaries=boundaries,
        min_dt=min_dt,
        labels=labels,
        issues=issues,
        confidence_notes=confidence_notes,
    )
    return _store_cached_prepared(cache_key, prepared)


def analyze_video(
    file_bytes: bytes,
    shot_type: str | None = None,
    shooting_hand: str | None = None,
    return_visuals: bool = True,
) -> schemas.AnalysisResult:
    prepared = _prepare_analysis(
        file_bytes,
        shot_type=shot_type,
        shooting_hand=shooting_hand,
    )
    frames = prepared.frames
    fps = prepared.fps
    stride = prepared.stride
    metrics = prepared.metrics
    phases = prepared.phases
    boundaries = prepared.boundaries
    min_dt = prepared.min_dt
    labels = prepared.labels
    issues = prepared.issues
    confidence_notes = prepared.confidence_notes

    comparison_result = comparison.build_comparison(phases, shot_type=shot_type)
    coaching_plan = _build_coaching_plan(issues, comparison_result=comparison_result)
    strengths = _build_strengths(metrics, phases, issues)
    research_notes = _build_research_notes(issues)
    summary = _build_summary(
        metrics,
        issues,
        coaching_plan,
        confidence_notes,
        comparison_result=comparison_result,
    )
    playback_cues = _build_playback_cues(
        metrics,
        boundaries,
        issues,
        frames,
        comparison_result=comparison_result,
    )
    annotated_video_b64 = None
    keyframes: List[schemas.KeyframePreview] = []
    phase_boundaries: List[schemas.PhaseBoundary] = _boundaries_from_indices(metrics, boundaries, min_dt)
    if return_visuals:
        # Use only frames that had pose detections to align with metrics/labels
        pose_frames = [frames[m.frame_index] for m in metrics if m.frame_index < len(frames)]
        # Generate annotated video with skeleton overlay on every frame
        annotated_video_b64 = _encode_overlay_video(pose_frames, metrics, labels, issues, fps / stride)
        keyframes = _keyframe_images(pose_frames, metrics, labels, issues)

    return schemas.AnalysisResult(
        phases=phases,
        issues=issues,
        summary=summary,
        coaching_plan=coaching_plan,
        strengths=strengths,
        research_notes=research_notes,
        comparison=comparison_result,
        playback_cues=playback_cues,
        confidence_notes=confidence_notes,
        annotated_video_b64=annotated_video_b64,
        keyframes=keyframes,
        phase_boundaries=phase_boundaries,
        processing_mode="deep" if return_visuals else "quick",
    )


def new_job_id() -> str:
    return str(uuid.uuid4())
