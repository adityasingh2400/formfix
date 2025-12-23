# Scope, Motions, Joints, Metrics

## Motions (initial coverage)
- Free throws, spot-up catch-and-shoot (corner/above-break), simple off-dribble pull-up.
- Angles/metrics conditioned on camera bucket (front/45/side) and shot distance.

## Joint set (basketball-focused)
- Lower body: ankles, knees, hips; feet orientation.
- Trunk: pelvis, spine base, neck/head tilt.
- Upper: shoulders, elbows, wrists, hands (shooting/off-hand).
- Optional fingers/ball proxy if model exposes them.

## Core metrics (per phase)
- Load: knee/hip flexion, ankle dorsiflexion, COM over base.
- Set: elbow angle, ball-to-head distance, shooting line alignment (shoulder–elbow–wrist–ball–rim).
- Rise: angular velocities of hip/shoulder; elbow lift timing vs lower-body extension.
- Release: wrist flex/extension velocity, elbow extension timing, release angle/height, ball path angle proxy.
- Off-hand: lateral distance, contact duration/torque risk.
- Follow-through: wrist flexion hold, elbow position, trunk sway.
- Balance: foot alignment, sway, landing symmetry.

## Success metrics
- Pose/3D lift accuracy: PCK/OKS on held-out clips.
- Issue detection: per-issue precision/recall/F1; timing error recall.
- Latency: <8s per 5s clip on GPU path; “quick check” target <3s.
- Robustness: fraction of frames with high pose confidence; graceful degradation with confidence flags.

