# Features, Phases, Thresholds (MVP-first)

## Phase segmentation
- Heuristic/threshold-based initially: catch → dip/load → set → rise → release → follow-through using joint velocities/angles.
- Upgrade path: small temporal classifier to harden boundaries.

## Per-phase features
- Load: knee/hip flexion, ankle dorsiflexion, COM over base, foot angle.
- Set: elbow angle, ball-to-head distance, shooting-line alignment.
- Rise: hip/shoulder angular velocities; elbow lift timing vs lower-body extension.
- Release: wrist flex/extension velocity, elbow extension timing, release angle/height proxy; ball path angle if detectable.
- Off-hand: lateral distance, contact duration; torque/push risk.
- Follow-through: wrist flexion hold duration, finger roll-through proxies (e.g., index PIP angle), elbow position, trunk sway.
- Balance: sway and landing symmetry.

## Baseline thresholds (MVP)
- Use broad pro bands (p10–p90) per shot type and camera bucket.
- Flag outliers via z-scores; attach confidence if pose quality low.
- Timing rules: legs load before arm lift; wrist flick after elbow/hip extension.

## Later (retain backlog)
- Archetype-specific bands (clustered pro styles).
- DTW/template-based sequencing scores.
- Outcome-conditioned weighting if makes/misses available.

