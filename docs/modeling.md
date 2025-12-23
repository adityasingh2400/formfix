# Modeling Plan

## Model A (ship first)
- Rule-based scoring on z-scores vs broad pro bands.
- Timing checks from heuristic phase segmentation.
- Outputs: issue list with severity + confidence flags.

## Model B (next)
- Sequence model (Temporal CNN/Transformer) for issue classification and suggested angle deltas.
- Input: pose/angle sequences + phase markers; masking for low-confidence frames.
- Calibration: per-class thresholds or temperature scaling to reduce false positives.

## Model C (later)
- Archetype-aware: cluster pros, pick nearest archetype per user; archetype-specific thresholds.
- Sequencing quality: DTW/template alignment scores for rhythm; timing delta suggestions.
- Outcome-aware: if makes/misses present, weight fixes that correlate with improved % within archetype.

## Inference modes
- Quick check: 2D + rules; fast path for feedback.
- Deep check: 3D + sequence model; richer suggestions.

