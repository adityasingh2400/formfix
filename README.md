# FormFix 🏀

AI-powered basketball shot form analyzer. Upload a video of your shot and get instant feedback on your mechanics.

![FormFix Screenshot](docs/screenshot.png)

## Features

- **Pose + Hand Tracking** — MediaPipe Holistic extracts 33 body landmarks + 21 hand landmarks per hand
- **Phase Segmentation** — Automatically detects Load → Set → Rise → Release → Follow-through
- **Wrist Snap Detection** — Tracks wrist flexion angle and velocity for flick analysis
- **Multi-Angle Support** — Works from front, side, or diagonal camera angles
- **Actionable Feedback** — Flags issues like shallow knee bend, weak wrist snap, short follow-through
- **Skeleton Overlay Video** — See the AI tracking frame-by-frame

## Quick Start

### Requirements

- **Python 3.10–3.12** (not 3.13 — MediaPipe doesn't support it yet)
- **ffmpeg** (for video encoding)
  ```bash
  # macOS
  brew install ffmpeg
  
  # Ubuntu/Debian
  sudo apt install ffmpeg
  ```

### Backend

```bash
cd formfix
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python -m uvicorn backend.src.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
python3 -m http.server 3000
```

Then open **http://localhost:3000** and upload a video!

## Project Structure

```
formfix/
├── backend/
│   ├── src/
│   │   ├── main.py          # FastAPI app
│   │   ├── schemas.py       # Pydantic models
│   │   └── services/
│   │       ├── analyzer.py  # Core pose extraction + analysis
│   │       └── video_utils.py
│   └── requirements.txt
├── frontend/
│   └── index.html           # Single-page app
├── docs/                     # Design docs & research
└── yolov8n.pt               # Ball detection weights (optional)
```

## How It Works

1. **Video Upload** → Frames extracted at 15fps
2. **Pose Estimation** → MediaPipe Holistic (body + hands)
3. **Phase Detection** → Heuristic segmentation based on knee angle, wrist height, wrist velocity
4. **Issue Detection** → Compare joint angles/timings against biomechanics research baselines
5. **Visualization** → Skeleton overlay rendered with OpenCV, encoded to H.264

## API

### `POST /analyze`

Upload a video file and get analysis results.

**Request:**
```
Content-Type: multipart/form-data
- file: video file (.mp4, .mov, etc.)
- shot_type: optional ("free_throw", "spot_up", "pull_up")
- return_visuals: "true" to include annotated video + keyframes
```

**Response:**
```json
{
  "job_id": "uuid",
  "status": "completed",
  "result": {
    "phases": [...],
    "issues": [...],
    "keyframes": [...],
    "annotated_video_b64": "data:video/mp4;base64,...",
    "confidence_notes": [...]
  }
}
```

## Research References

- Biomechanical analysis of basketball shooting (KDU study) — knee ~122°, elbow ~79° at free throw
- Kinematics of Arm Joint Motions (ScienceDirect) — shoulder rotation → vertical velocity, elbow/wrist → horizontal + backspin
- MediaPipe Holistic — 33 pose + 21×2 hand landmarks

## License

MIT

