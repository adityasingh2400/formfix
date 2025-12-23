# Shooting biomechanics (MVP heuristics + how we measure them)

This repo currently uses **single-view video** + **MediaPipe Holistic** (pose + hands) to estimate kinematics and flag likely form issues. The current logic is **heuristic** and intentionally uses **broad “pro-like” bands** so it works across different bodies and camera setups.

## Keypoint sources (what we track)

- **Pose**: MediaPipe Holistic pose landmarks (33 points; shoulders/elbows/wrists/hips/knees/ankles are core).
- **Hands**: MediaPipe Holistic hand landmarks (21 per hand; wrist + finger joints).

Why hands matter: a lot of “feel” issues show up in **wrist flexion velocity (“snap”)** and **follow‑through hold**, which you can’t infer reliably from pose-only wrists.

## Making the analysis robust across camera angles

Single-view 2D is inherently camera-dependent. To make features more stable across *different* viewpoints and zoom:

- **Angles, not pixels**: Most checks use **joint angles** (e.g., elbow angle) which are more viewpoint-stable than raw x/y positions.
- **3D where possible**: When Holistic provides `pose_world_landmarks`, we compute angles/distances in that space (less sensitive to foreshortening).
- **Body-relative “height”**: `wrist_height` is computed as a **projection along the torso axis** and normalized by torso length:
  - 0 ≈ hip level
  - 1 ≈ shoulder level
  - >1 ≈ above shoulders

Practical limitation: extreme occlusions (hands behind the ball/body), severe motion blur, or very oblique angles can still reduce reliability. The API returns confidence notes when hands/pose are missing frequently.

## Metrics we compute (current)

### Pose kinematics

- **`knee_flexion`**: actually the **knee joint angle** (degrees). Smaller angle = deeper bend.
- **`hip_flexion`**: hip joint angle proxy (degrees).
- **`elbow_angle`**: elbow joint angle (degrees).
- **`shoulder_angle`**: shoulder joint angle proxy (degrees).
- **`wrist_height`**: body-relative wrist height (unitless; normalized).

### Hand / finger kinematics

- **`wrist_flexion`**: proxy for wrist “snap” angle (degrees), computed as:
  - \(180 - \angle(\text{elbow}, \text{wrist}, \text{middle\_mcp})\)
  - 0 ≈ straight wrist; higher ≈ more flexed
- **`wrist_flexion_vel_peak`**: peak wrist-flexion angular velocity in the phase window (deg/s).
- **`index_pip_angle`**: index finger PIP joint angle (degrees). Near 180 ≈ more extended.

## Shot phases (current heuristic segmentation)

The backend segments into:

- **`load`**: start → deepest knee bend (min knee angle)
- **`set`**: load → most-bent elbow between load and peak height
- **`rise`**: set → inferred release moment
- **`release`**: a short window around the inferred release moment (1–2 frames)
- **`follow_through`**: post-release window → end of clip

Release moment is inferred using **peak wrist flexion velocity** when hands are visible; otherwise it falls back to “top of motion” (peak ball/wrist height).

## “Pro-like” heuristics (what we flag)

These are MVP checks designed to be:
- tolerant of different body types and styles
- robust to camera differences
- easy to tune later with real “elite shooter” datasets

Current flags include:

- **Knee bend shallow (load)**: knee angle too large at load (not enough bend).
- **Elbow angle at set off-band (set)**: elbow too tucked or too open.
- **Low release height (release)**: insufficient `wrist_height` gain from load → release.
- **Rushed sequencing (rise)**: arm lift starts before leg drive.
- **Off-hand too close (release)**: guide-hand spacing too small near release (when trackable).
- **Wrist snap looks weak (release)**: low peak `wrist_flexion_vel` when hands are visible.
- **Short follow-through (follow_through)**: follow-through duration or wrist-flex “hold time” too short.
- **Finger roll-through limited (follow_through)**: index finger stays too curled (very rough proxy).

## Research references (used for intuition, not strict ground truth)

We use these as directional guidance for what *tends* to matter biomechanically, not as universal “one true form.”

- **Arm joint contributions / release mechanics**: *Kinematics of Arm Joint Motions in Basketball Shooting* (Procedia Engineering). `https://www.sciencedirect.com/science/article/pii/S187770581501471X`
- **Example joint-angle values from motion capture** (free throw knee/elbow angles reported): `https://ir.kdu.ac.lk/handle/345/8863`
- **Upper-extremity coordination & accuracy** (3‑pt shooting): `https://pubmed.ncbi.nlm.nih.gov/40453894/`

Next step for “best shooter” tuning: build a labeled dataset of elite clips with camera-angle tags and learn **per-phase bands** (p10–p90) per shot type + camera bucket, then replace the broad MVP thresholds with data-driven ones.


