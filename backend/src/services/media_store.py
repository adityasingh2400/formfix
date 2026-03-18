from __future__ import annotations

import os
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MEDIA_ROOT = _REPO_ROOT / ".formfix_media"


class LocalMediaStore:
    def __init__(self, root: str | Path | None = None) -> None:
        configured_root = root or os.getenv("FORMFIX_MEDIA_ROOT") or _DEFAULT_MEDIA_ROOT
        self.root = Path(configured_root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)

    def ensure_job_dirs(self, job_id: str) -> dict[str, Path]:
        base = self.root / "jobs" / job_id
        paths = {
            "base": base,
            "uploads": base / "uploads",
            "renditions": base / "renditions",
            "artifacts": base / "artifacts",
            "manifest": base / "manifest.json",
        }
        for key, path in paths.items():
            if key == "manifest":
                continue
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def job_manifest_path(self, job_id: str) -> Path:
        return self.root / "jobs" / job_id / "manifest.json"

    def media_url(self, path: str | Path) -> str:
        full_path = Path(path).resolve()
        relative = full_path.relative_to(self.root).as_posix()
        return f"/media/{relative}"

    @staticmethod
    def safe_filename(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
        cleaned = cleaned.strip(".-") or "upload.mp4"
        return cleaned[:120]
