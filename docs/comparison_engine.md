# Comparison Engine

## What ships now

FormFix now supports a real comparison layer on top of the mechanics analyzer:

- **Seeded archetype matching** in `/Users/aditya/Desktop/formfix/backend/src/services/comparison.py`
- **Reference profile library** in `/Users/aditya/Desktop/formfix/backend/src/data/reference_profiles.json`
- **Frontend comparison UX** in `/Users/aditya/Desktop/formfix/frontend/index.html`

Each analyzed clip now returns:

- the closest current shot-style family
- traits the user already shares with that style
- traits worth borrowing from that style
- alternative nearby styles

## Why archetypes first

There is still no clean public dataset that gives all of these at once:

- player identity
- single-shot windows
- per-phase biomechanics labels
- enough clips per player and shot type
- licensing that is clean enough for commercial product work

That means the right progression is:

1. ship seeded archetypes
2. train data-driven archetypes from exported clip features
3. build player profiles once labeled clip volume exists
4. replace centroid matching with embedding retrieval

## Current matching logic

The current engine compares each clip against reference targets for:

- load knee flexion
- set elbow angle
- rise shoulder angle
- release wrist height
- release shoulder angle
- release wrist snap speed
- follow-through duration
- follow-through finger roll-through

The engine computes:

- weighted normalized distance to each reference profile
- fit score from 0 to 100
- comparison confidence based on distance and feature coverage
- aligned traits
- borrow-next traits

The current library is intentionally **archetype-based**, not player-based.

## Important product note

One clean rep is enough for a rough coaching pass, but it is **not** enough to make a confident long-term archetype call on a shooter.

The comparison result in the current product should be treated as:

- a one-shot style read
- a lightweight reference for cueing
- not a stable player archetype label

For a real archetype or player-style classification flow, the product should move to:

1. a multi-shot upload or session video
2. aggregation across repeated reps
3. stability checks before labeling the shooter

## Real training path

### Stage 1: Export features from clip libraries

Use:

- `/Users/aditya/Desktop/formfix/tools/reference_pipeline/export_features.py`

This produces JSONL rows with:

- clip metadata
- per-phase metrics
- flattened comparison features
- issue flags
- confidence notes

### Stage 2: Train archetype clusters

Use:

- `/Users/aditya/Desktop/formfix/tools/reference_pipeline/train_reference_library.py`

This script:

- groups clips by shot type
- clusters feature vectors with k-means
- estimates centroid targets and tolerances
- writes a trained reference library in the same JSON schema as the seeded library

### Stage 3: Build player profiles

Use:

- `/Users/aditya/Desktop/formfix/tools/reference_pipeline/build_player_profiles.py`

This script groups by:

- `player_id + shot_type`

and builds a player profile when enough clips exist.

## Recommended model roadmap

### Model A: Current nearest-profile baseline

- Input: flattened per-phase metrics
- Output: nearest archetype or player profile
- Strength: easy to debug, easy to explain
- Weakness: does not learn temporal style directly

### Model B: Sequence embedding model

- Input: angle sequences + phase boundaries + confidence masks
- Backbone: temporal CNN or Transformer
- Objective: metric learning
- Loss: contrastive or triplet loss
- Retrieval target: same player, same archetype, or coach-labeled "similar style"

This supports:

- better nearest-neighbor search
- player-to-player similarity
- clip-to-clip retrieval
- smoother comparison than centroid-only matching

### Model C: Retrieval plus outcome weighting

- Add make/miss and miss-direction labels
- Learn which style deltas matter most for a user's actual misses
- Surface comparison cues that are both style-relevant and outcome-relevant

## Evaluation plan

### Retrieval quality

- Top-1 and Top-3 style retrieval accuracy
- player retrieval accuracy when player labels are available
- nearest-neighbor agreement with human coach judgments

### Product usefulness

- percentage of users who apply the first comparison cue
- percentage who say the comparison section is clearer than raw metrics
- session-to-session movement toward the borrowed cue

### Robustness

- stability across camera buckets
- stability across repeated clips from the same shooter
- degradation under low hand visibility

## Product rule

Comparison should never sound like:

- "copy this pro exactly"
- "change three things at once"
- "your shot is wrong because it differs"

Comparison should sound like:

- "your shot already shares these traits"
- "if you want to borrow one thing from this style, start here"
- "use this as a guide, not a template"
