# Backend (Functional MVP)

FastAPI service that ingests a video, runs **MediaPipe Holistic (pose + hands)** to extract joints + finger landmarks, segments basic phases, and produces heuristic issues (knee bend, release height, sequencing, off-hand spacing, **wrist snap / follow-through**).

## Requirements

- **Python 3.10–3.12** (MediaPipe doesn't support 3.13 yet)
- **ffmpeg** (for browser-compatible video encoding)

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
- `POST /analyze` — upload a video file (`video/*`); returns phases, heuristic issues, confidence notes, and (if requested) annotated video/keyframe overlays.

## Notes
- Dependencies include `mediapipe` and `opencv-python-headless`; wheels are provided for common platforms (tested on macOS).
- Optional: ball detection uses **YOLOv8** via `ultralytics` if installed. If not installed, analysis still runs (just without ball-based signals).
- Current phase segmentation is heuristic (load/set/rise/release/follow-through) using single-view pose+hands. Replace `services/analyzer.py` with improved models as they land (better 3D, archetypes, learned phase boundaries, etc.).

