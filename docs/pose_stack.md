# Pose & Kinematics Stack

## 2D Pose (MVP)
- MoveNet/RTMPose/MediaPipe exported to ONNX for server inference.
- Shoot-time optimizations: half/float16, batching by clip, confidence thresholds.

## 3D Lift (next)
- Single-view lift: VideoPose3D or MHFormer; align joints to basketball set.
- Normalization: limb-length scaling; shooting-plane alignment to reduce camera variance.

## Smoothing & Gaps
- Temporal smoothing (Savitzky–Golay or 1D Kalman) on key joints/angles.
- Gap filling with linear/velocity-aware interpolation; mark low-confidence spans.

## Quality flags
- Per-frame pose confidence; fraction of usable frames.
- Reject/soft-fail clips with insufficient visibility; prompt recapture guidance.

