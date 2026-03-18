# FormFix 🏀

**AI-powered basketball shot form analyzer.** Upload one clip, get real coaching cues in plain English — then dive into the tracked replay.

---

## Why FormFix?

Most shot analysis tools dump raw numbers. FormFix gives you **coaching**, not metrics.

| Traditional Tools | FormFix |
|-------------------|---------|
| "Elbow angle: 78°" | "Keep your elbow tighter at the set point" |
| "Knee flexion: 115°" | "Sink a bit deeper — you're losing power" |
| "Release height: 2.1m" | "Your release is clean, matches a textbook set shot" |

One clean rep is all you need.

---

## ✨ Features

### 🎯 Quick Coaching First
Plain-language cues appear in seconds — before any heavy visuals render. No waiting around.

### 🔬 Full Pose + Hand Tracking
MediaPipe Holistic extracts **33 body landmarks** plus **21 hand landmarks per hand**. We see your wrist snap, finger roll-through, and follow-through extension.

### 📊 Automatic Phase Detection
The shot is segmented into **Load → Set → Rise → Release → Follow-through** — each phase analyzed independently.

### 🎨 Style Comparison
Your shot is matched against an **archetype library**. See which style family you're closest to, what traits you already share, and what to borrow next.

### 🎬 Slow-Mo Evidence Replay
Native slow-motion uploads are preferred. FormFix preserves the master clip, renders a guided replay, and builds cue clips, proof frames, and frame strips so the feedback feels earned.

### 📱 Native Slow-Mo First
The ideal upload is the original `4K 240 fps` or `1080p 120/240 fps` clip from your phone. In-browser recording still works, but it is treated as a fallback path.

---

## 🚀 Quick Start

### One-Command Setup

**Terminal 1 — Backend:**
```bash
cd backend && ./back.sh
```

**Terminal 2 — Frontend:**
```bash
cd frontend && ./front.sh
```

Open **http://localhost:3000** → upload or record a shot → get feedback.

---

### Manual Setup

#### Requirements
- **Python 3.10–3.12** (MediaPipe doesn't support 3.13 yet)
- **ffmpeg** for video encoding

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

#### Backend
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python -m uvicorn backend.src.main:app --reload --port 8000
```

#### Frontend
```bash
cd frontend
python3 -m http.server 3000
```

---

## 🏗 Architecture

```
formfix/
├── backend/
│   ├── src/
│   │   ├── main.py              # FastAPI endpoints
│   │   ├── schemas.py           # Pydantic models
│   │   ├── data/
│   │   │   ├── reference_profiles.json   # Archetype library
│   │   │   └── research_bank.json        # Biomechanics research
│   │   └── services/
│   │       ├── analyzer.py      # Pose extraction + phase detection
│   │       ├── comparison.py    # Style matching engine
│   │       └── video_utils.py   # Frame extraction + encoding
│   └── requirements.txt
├── frontend/
│   └── index.html               # Single-page app (vanilla JS)
├── tools/
│   └── reference_pipeline/      # Train your own archetype library
│       ├── export_features.py
│       ├── train_reference_library.py
│       └── build_player_profiles.py
└── docs/
    ├── comparison_engine.md     # How style matching works
    └── datasets_references.md   # Data sources + research
```

---

## ⚙️ How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Upload    │ ──▶ │    Pose     │ ──▶ │   Phase     │
│   Video     │     │  Extraction │     │  Detection  │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
       ┌───────────────────────────────────────┘
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Quick     │ ──▶ │   Style     │ ──▶ │   Deep      │
│  Coaching   │     │   Match     │     │  Breakdown  │
└─────────────┘     └─────────────┘     └─────────────┘
```

1. **Video Upload** — Native slow-motion is preferred; `4K 240 fps` is the gold-standard capture
2. **Media Inspection** — `ffprobe` classifies frame density, resolution, and capture tier
3. **Coarse Scan** — A reduced rendition locates the shot rhythm and likely cue windows
4. **Dense Refinement** — High-detail windows sharpen load, release, and follow-through timing
5. **Style Match + Coaching** — Strongest findings turn into plain-language cues
6. **Evidence Pack** — Guided replay, cue clips, proof frames, and frame strips are rendered as URL-based artifacts

---

## 📡 API

### `POST /analysis-jobs`

Queue an async analysis job.

**Request:**
```
Content-Type: multipart/form-data

file: video file (.mp4, .mov, .webm, etc.)
shot_type: optional — "free_throw" | "spot_up" | "pull_up"
shooting_hand: optional — "left" | "right"
```

**Response:**
```json
{
  "job_id": "uuid",
  "status": "queued",
  "stage": "queued",
  "progress_message": "Clip uploaded. Checking whether this is a high-detail slow-motion read.",
  "media_profile": {
    "fps": 240.0,
    "detail_tier": "ultra_detail"
  }
}
```

### `GET /analysis-jobs/{job_id}`

Poll for status and fetch the final URL-based result payload.

### `POST /analyze`

Blocking compatibility wrapper for quick local testing.

**Request:**
```
Content-Type: multipart/form-data

file: video file (.mp4, .mov, etc.)
shot_type: optional — "free_throw" | "spot_up" | "pull_up"
return_visuals: "true" to include replay assets
```

**Response:**
```json
{
  "job_id": "uuid",
  "status": "completed",
  "result": {
    "phases": [...],
    "issues": [...],
    "coaching_cues": [...],
    "comparison": {
      "style_family": "textbook_set",
      "fit_score": 82,
      "aligned_traits": [...],
      "borrow_next": [...]
    },
    "media_profile": {...},
    "artifacts": {
      "annotated_replay_url": "/media/jobs/<id>/artifacts/annotated-replay.mp4",
      "cue_clip_urls": [...],
      "cue_still_urls": [...],
      "frame_strip_urls": [...]
    },
    "playback_script": [...]
  }
}
```

### `GET /health`

Service health check.

---

## 🔧 Reference Pipeline

Train your own archetype library from labeled clips:

```bash
# 1. Export features from your clip library
python tools/reference_pipeline/export_features.py \
  --input clips/ --output features.jsonl

# 2. Cluster into archetypes
python tools/reference_pipeline/train_reference_library.py \
  --input features.jsonl --output trained_library.json

# 3. (Optional) Build player profiles
python tools/reference_pipeline/build_player_profiles.py \
  --input features.jsonl --output player_profiles.json
```

Override the default library:
```bash
export FORMFIX_REFERENCE_LIBRARY=/path/to/trained_library.json
```

See `docs/comparison_engine.md` for the full training roadmap.

---

## 📚 Research References

- **Biomechanical analysis of basketball shooting** (KDU study) — optimal knee ~122°, elbow ~79° at free throw
- **Kinematics of Arm Joint Motions** (ScienceDirect) — shoulder rotation → vertical velocity; elbow/wrist → horizontal + backspin
- **MediaPipe Holistic** — 33 pose + 21×2 hand landmarks

---

## 📄 License

MIT
