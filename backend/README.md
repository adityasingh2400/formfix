# Backend (Functional MVP)

FastAPI service that ingests a video, inspects the capture profile, preserves a local master asset, runs **MediaPipe Holistic (pose + hands)** to extract joints + finger landmarks, segments basic phases, and produces heuristic issues (knee bend, release height, sequencing, off-hand spacing, **wrist snap / follow-through**).

The service now also supports:

- quick coaching vs deep visual passes
- async job processing with disk-backed manifests
- slow-motion-first media inspection and URL-based replay artifacts
- seeded archetype comparison
- trained reference-library overrides via `FORMFIX_REFERENCE_LIBRARY`

## Requirements

- **Python 3.10–3.12** (MediaPipe doesn't support 3.13 yet)
- **ffmpeg** and **ffprobe** (for media inspection and browser-compatible encoding)

## Setup
```bash
# From project root
python3.11 -m venv .venv        # or python3.10, python3.12
source .venv/bin/activate
pip install -r backend/requirements.txt
python -m uvicorn backend.src.main:app --reload --port 8000
```

## Endpoints
- `GET /health` — service health.
- `POST /analysis-jobs` — queue an async analysis job and return media-profile metadata.
- `GET /analysis-jobs/{job_id}` — fetch current stage, status, and final result payload.
- `POST /analyze` — blocking compatibility wrapper for quick local testing.

## Reference Libraries

Default library:

- `backend/src/data/reference_profiles.json`

Override it with a trained library:

```bash
export FORMFIX_REFERENCE_LIBRARY=/absolute/path/to/trained_reference_library.json
```

## Notes
- Dependencies include `mediapipe` and `opencv-python-headless`; wheels are provided for common platforms (tested on macOS).
- Use Python `3.11` or `3.12` for the backend runtime. The default system Python `3.14` is not supported by the MediaPipe stack in this repo.
- Optional: ball detection uses **YOLOv8** via `ultralytics` if installed. If not installed, analysis still runs (just without ball-based signals).
- Current phase segmentation is heuristic (load/set/rise/release/follow-through) using single-view pose+hands. The new pipeline adds a coarse pass plus dense slow-motion refinement windows, but it is still heuristic rather than learned.
