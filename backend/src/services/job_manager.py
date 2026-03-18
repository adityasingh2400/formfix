from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

from .. import schemas
from . import analyzer
from .media_store import LocalMediaStore
from .video_utils import create_analysis_rendition, inspect_video_path, normalize_video


class AnalysisJobManager:
    def __init__(self, store: LocalMediaStore, max_workers: int = 2) -> None:
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="formfix-job")
        self._lock = threading.Lock()

    def create_job(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        shot_type: str | None,
        shooting_hand: str | None,
    ) -> schemas.AnalysisJobResponse:
        job_id = analyzer.new_job_id()
        dirs = self.store.ensure_job_dirs(job_id)
        upload_path = dirs["uploads"] / self.store.safe_filename(filename)
        upload_path.write_bytes(file_bytes)
        media_profile = inspect_video_path(upload_path)
        payload = {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress_message": "Clip uploaded. Checking whether this is a high-detail slow-motion read.",
            "media_profile": media_profile.model_dump(),
            "result": None,
            "error": None,
            "inputs": {
                "upload_path": str(upload_path),
                "shot_type": shot_type,
                "shooting_hand": shooting_hand,
            },
        }
        self._write_manifest(job_id, payload)
        self.executor.submit(self._run_job, job_id)
        return self.get_job(job_id)  # type: ignore[return-value]

    def run_inline_analysis(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        shot_type: str | None,
        shooting_hand: str | None,
    ) -> tuple[str, schemas.AnalysisResult]:
        job_id = analyzer.new_job_id()
        dirs = self.store.ensure_job_dirs(job_id)
        upload_path = dirs["uploads"] / self.store.safe_filename(filename)
        upload_path.write_bytes(file_bytes)
        media_profile = inspect_video_path(upload_path)
        result = self._process_job(
            job_id=job_id,
            upload_path=upload_path,
            shot_type=shot_type,
            shooting_hand=shooting_hand,
            media_profile=media_profile,
            progress_hook=None,
        )
        return job_id, result

    def get_job(self, job_id: str) -> Optional[schemas.AnalysisJobResponse]:
        payload = self._read_manifest(job_id)
        if payload is None:
            return None
        return schemas.AnalysisJobResponse.model_validate(
            {
                "job_id": payload["job_id"],
                "status": payload["status"],
                "stage": payload["stage"],
                "progress_message": payload["progress_message"],
                "media_profile": payload.get("media_profile"),
                "result": payload.get("result"),
                "error": payload.get("error"),
            }
        )

    def _run_job(self, job_id: str) -> None:
        payload = self._read_manifest(job_id)
        if payload is None:
            return
        inputs = payload.get("inputs") or {}
        upload_path = Path(inputs["upload_path"])
        shot_type = inputs.get("shot_type")
        shooting_hand = inputs.get("shooting_hand")
        media_profile = schemas.MediaProfile.model_validate(payload.get("media_profile") or {})

        try:
            result = self._process_job(
                job_id=job_id,
                upload_path=upload_path,
                shot_type=shot_type,
                shooting_hand=shooting_hand,
                media_profile=media_profile,
                progress_hook=self._make_progress_hook(job_id),
            )
        except Exception as exc:
            self._update_manifest(
                job_id,
                status="failed",
                stage="failed",
                progress_message="Analysis failed before the replay could be built.",
                error=str(exc),
            )
            return

        self._update_manifest(
            job_id,
            status="completed",
            stage="completed",
            progress_message="Analysis complete. The slow-motion evidence pack is ready.",
            result=result.model_dump(),
            error=None,
        )

    def _process_job(
        self,
        *,
        job_id: str,
        upload_path: Path,
        shot_type: str | None,
        shooting_hand: str | None,
        media_profile: schemas.MediaProfile,
        progress_hook: Optional[Callable[[str, str], None]],
    ) -> schemas.AnalysisResult:
        if media_profile.duration_s > 12.0:
            raise ValueError(
                "Clip is too long for a useful one-shot read. Upload one rep or trim the clip to about 3-6 seconds."
            )

        dirs = self.store.ensure_job_dirs(job_id)
        normalized_path = dirs["renditions"] / "normalized-source.mp4"
        coarse_path = dirs["renditions"] / "coarse-analysis.mp4"

        if progress_hook:
            progress_hook("normalizing", "Normalizing the source clip and preserving a browser-safe master.")
        normalize_video(upload_path, normalized_path)

        if progress_hook:
            progress_hook("coarse_scan", "Scanning the full clip to find the shot rhythm and likely cue windows.")
        create_analysis_rendition(normalized_path, coarse_path)

        result = analyzer.analyze_video_paths(
            coarse_video_path=coarse_path,
            dense_video_path=normalized_path,
            shot_type=shot_type,
            shooting_hand=shooting_hand,
            media_profile=media_profile,
            artifact_dir=dirs["artifacts"],
            url_builder=self.store.media_url,
            normalized_source_url=self.store.media_url(normalized_path),
            progress_hook=progress_hook,
        )
        return result

    def _make_progress_hook(self, job_id: str) -> Callable[[str, str], None]:
        def _hook(stage: str, message: str) -> None:
            self._update_manifest(
                job_id,
                status="running",
                stage=stage,
                progress_message=message,
            )

        return _hook

    def _read_manifest(self, job_id: str) -> Optional[dict[str, Any]]:
        manifest_path = self.store.job_manifest_path(job_id)
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text())

    def _write_manifest(self, job_id: str, payload: dict[str, Any]) -> None:
        manifest_path = self.store.job_manifest_path(job_id)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            manifest_path.write_text(json.dumps(payload, indent=2))

    def _update_manifest(self, job_id: str, **updates: Any) -> None:
        manifest = self._read_manifest(job_id)
        if manifest is None:
            return
        manifest.update(updates)
        self._write_manifest(job_id, manifest)
