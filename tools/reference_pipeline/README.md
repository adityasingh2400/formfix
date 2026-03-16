# Reference Pipeline

These scripts turn raw shot clips into a comparison library that FormFix can use for:

- archetype matching
- player-level profile matching
- future retrieval or ranking models

## 1. Export clip features

```bash
cd /Users/aditya/Desktop/formfix
source .venv/bin/activate
python tools/reference_pipeline/export_features.py \
  --video-dir /path/to/shot_clips \
  --metadata-csv /path/to/metadata.csv \
  --output data/reference/features.jsonl \
  --recursive
```

Expected metadata columns:

- `filename`
- `shot_type`
- `player_id`
- `player_name`
- `outcome`
- `source`
- `split`

Only `filename` is required. Everything else is optional but strongly recommended.

## 2. Train archetype library

```bash
python tools/reference_pipeline/train_reference_library.py \
  --input data/reference/features.jsonl \
  --output data/reference/trained_archetypes.json \
  --clusters 4 \
  --min-cluster-size 8
```

This produces a data-driven archetype library in the same JSON format as the seeded library at:

- `/Users/aditya/Desktop/formfix/backend/src/data/reference_profiles.json`

To point FormFix at a trained library, set:

```bash
export FORMFIX_REFERENCE_LIBRARY=/Users/aditya/Desktop/formfix/data/reference/trained_archetypes.json
```

## 3. Build player profiles

```bash
python tools/reference_pipeline/build_player_profiles.py \
  --input data/reference/features.jsonl \
  --output data/reference/player_profiles.json \
  --min-clips 6
```

This creates one profile per `player_id + shot_type` group.

## Recommended real-world flow

1. Start with archetypes trained from a mixed clip pool.
2. Add player profiles after you have enough clips per player and shot type.
3. Merge the archetype and player libraries if you want both experiences in-product.
4. Replace nearest-centroid matching later with embedding retrieval once you have enough labeled data.

## What these scripts do not solve

- licensing footage
- trimming plays into clean shot windows
- human QA of mislabeled or low-confidence clips
- syncing NBA shot metadata to the right broadcast frames

Those should sit one layer upstream in the ingestion pipeline.
