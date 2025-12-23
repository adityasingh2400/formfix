from __future__ import annotations

import base64
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from .. import schemas
from .video_utils import decode_video

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


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute angle ABC in degrees."""
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) or 1e-6
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


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


def _extract_pose_metrics(frames: List[np.ndarray], fps: float, stride: int) -> List[FrameMetrics]:
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

    # First pass: collect per-frame observations for both sides so we can pick the shooting hand.
    per_frame: List[
        Tuple[
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
    ] = []

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

    def _choose_shooting_side() -> str:
        # Prefer ball proximity when available (works across most angles).
        dists: Dict[str, List[float]] = {"left": [], "right": []}
        for _, _, _, _, ball, arms, _, _, _, _ in per_frame:
            if not ball:
                continue
            bx, by = ball
            for side in ("left", "right"):
                wp = arms[side].wrist_px
                if wp is None:
                    continue
                wx, wy = wp
                dists[side].append(float(np.hypot(wx - bx, wy - by)))
        if len(dists["left"]) >= 3 or len(dists["right"]) >= 3:
            left_med = float(np.median(dists["left"])) if dists["left"] else 1e9
            right_med = float(np.median(dists["right"])) if dists["right"] else 1e9
            return "left" if left_med < right_med else "right"

        # Fallback: choose the arm with better visibility + more frequent hand detections.
        scores = {"left": 0.0, "right": 0.0}
        counts = {"left": 0, "right": 0}
        for _, _, _, _, _, arms, _, _, _, _ in per_frame:
            for side in ("left", "right"):
                a = arms[side]
                # Weight hand detections heavily: wrist/finger kinematics are the main discriminator.
                scores[side] += (
                    1.0 * a.arm_visibility
                    + (2.0 if a.hand_present else 0.0)
                    + (0.25 if a.wrist_flexion is not None else 0.0)
                    + (0.10 * float(a.wrist_height or 0.0))
                )
                counts[side] += 1
        left = scores["left"] / max(counts["left"], 1)
        right = scores["right"] / max(counts["right"], 1)
        return "left" if left > right else "right"

    shooting_side = _choose_shooting_side()
    guide_side = "left" if shooting_side == "right" else "right"

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
                delta="Try a deeper load (more knee flexion) before rising",
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
                delta="Aim for a comfortable, repeatable elbow bend at set (often ~70–110°)",
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
                    delta="Get to a higher set/release point (more lift before the snap)",
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
                delta="Let the legs initiate the rise, then bring the arm up (avoid early arm lift)",
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
                delta="Keep guide hand relaxed with a small gap",
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
                delta="Finish with a quicker, cleaner wrist snap (\"snap down\" through the ball)",
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
                    delta="Hold your finish a beat longer (wrist stays flexed; arm extends through)",
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
                    delta="Let the fingers (esp. index/middle) roll through the release more naturally",
                    confidence=hand_conf,
                    phase="follow_through",
            )
        )

    return issues


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

    def line(a: str, b: str):
        if a in kp and b in kp:
            cv2.line(frame, kp[a], kp[b], base_color, 4)

    def dot(a: str):
        if a in kp:
            cv2.circle(frame, kp[a], 8, base_color, -1)
            cv2.circle(frame, kp[a], 10, (0, 0, 0), 2)

    def hand_line(prefix: str, a: str, b: str, color: Tuple[int, int, int], thickness: int = 2):
        ka = f"{prefix}_hand_{a}"
        kb = f"{prefix}_hand_{b}"
        if ka in kp and kb in kp:
            cv2.line(frame, kp[ka], kp[kb], color, thickness)

    def hand_dot(prefix: str, a: str, color: Tuple[int, int, int], radius: int = 4):
        ka = f"{prefix}_hand_{a}"
        if ka in kp:
            cv2.circle(frame, kp[ka], radius, color, -1)

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
        cv2.circle(frame, m.ball_pos, 15, (0, 0, 255), 3)
        cv2.circle(frame, m.ball_pos, 5, (0, 0, 255), -1)

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
        m = metrics[mid]
        
        base_color = _PHASE_COLORS.get(phase, (255, 255, 255))

        phase_issues = [i for i in issues if i.phase == phase]
        kp = m.keypoints_px

        def line(a: str, b: str):
            if a in kp and b in kp:
                cv2.line(frame, kp[a], kp[b], base_color, 4)

        def dot(a: str):
            if a in kp:
                cv2.circle(frame, kp[a], 8, base_color, -1)
                cv2.circle(frame, kp[a], 10, (0, 0, 0), 2)  # outline

        def hand_line(prefix: str, a: str, b: str, color: Tuple[int, int, int], thickness: int = 2):
            ka = f"{prefix}_hand_{a}"
            kb = f"{prefix}_hand_{b}"
            if ka in kp and kb in kp:
                cv2.line(frame, kp[ka], kp[kb], color, thickness)

        def hand_dot(prefix: str, a: str, color: Tuple[int, int, int], radius: int = 4):
            ka = f"{prefix}_hand_{a}"
            if ka in kp:
                cv2.circle(frame, kp[ka], radius, color, -1)

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
            cv2.circle(frame, m.ball_pos, 15, (0, 0, 255), 3)
            cv2.circle(frame, m.ball_pos, 5, (0, 0, 255), -1)

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


def analyze_video(
    file_bytes: bytes, shot_type: str | None = None, return_visuals: bool = True
) -> schemas.AnalysisResult:
    frames, fps, stride = decode_video(file_bytes, sample_every_n=2)
    if not frames:
        raise ValueError("Could not decode video or video empty.")

    metrics = _extract_pose_metrics(frames, fps, stride)
    if len(metrics) < 3:
        raise ValueError("Insufficient pose detections; recapture with clearer framing.")

    phases, boundaries, min_dt = _segment_phases(metrics)
    labels = _label_frames(metrics, boundaries)
    issues = _issues_from_metrics(metrics, phases, boundaries)

    confidence_notes: List[str] = []
    avg_vis = _mean([m.visibility for m in metrics]) or 0.0
    if avg_vis < 0.6:
        confidence_notes.append("Low pose confidence; camera angle/lighting may reduce accuracy.")
    if len(metrics) < max(5, len(frames) // 4):
        confidence_notes.append("Many frames missing pose; try clearer side/front angle.")
    hand_rate = _mean([m.hand_confidence for m in metrics]) or 0.0
    if hand_rate < 0.5:
        confidence_notes.append(
            "Hands/fingers often occluded; wrist/finger analysis (wrist flick, follow-through) may be limited."
        )
    # Expose our inferred shooting side for debugging/UX.
    if metrics:
        confidence_notes.append(f"Inferred shooting arm: {metrics[0].shooting_side}.")

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
        confidence_notes=confidence_notes,
        annotated_video_b64=annotated_video_b64,
        keyframes=keyframes,
        phase_boundaries=phase_boundaries,
    )


def new_job_id() -> str:
    return str(uuid.uuid4())

