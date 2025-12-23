# Data Intake & Licensing

## Sources
- Licensed NBA broadcast/practice footage with clear shooting views.
- Metadata: player, play type, camera angle, distance, outcome (if available), clip quality.

## Pipeline
- Ingest → store signed URLs/paths + metadata in DB.
- Preprocess: trim to shot window; downscale to 720p if needed; denoise/light stabilization.
- Auto-pose pass → flag low-confidence frames for QA.
- Human QA on key frames (load, set, release, follow-through) to validate joints and camera tags.

## Augmentation
- Camera jitter, slight occlusion masks, brightness/contrast shifts.
- Optional 3D mocap transfer for bootstrapping rare angles.

## Governance
- Track license terms and retention per source.
- Store consent for user uploads; delete-on-complete option.
- Face blur default on user media.

