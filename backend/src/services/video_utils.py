from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .. import schemas


DEFAULT_COARSE_TARGET_FPS = 24.0
DEFAULT_DENSE_TARGET_FPS = 60.0
DEFAULT_COARSE_MAX_DIMENSION = 1080


def _write_temp_video(file_bytes: bytes, suffix: str = ".mp4") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name


def _parse_frame_rate(raw_value: str | None) -> float:
    if not raw_value or raw_value in {"0/0", "0"}:
        return 0.0
    if "/" in raw_value:
        numerator, denominator = raw_value.split("/", 1)
        den = float(denominator or 0)
        if den == 0:
            return 0.0
        return float(numerator) / den
    return float(raw_value)


def classify_detail_tier(width: int, height: int, fps: float) -> str:
    max_side = max(width, height)
    if fps >= 200.0 and max_side >= 3000:
        return "ultra_detail"
    if fps >= 100.0:
        return "high_detail"
    return "standard_detail"


def capture_assessment_for_profile(width: int, height: int, fps: float) -> str:
    detail_tier = classify_detail_tier(width, height, fps)
    if detail_tier == "ultra_detail":
        return "Ideal capture. Native 4K 240 fps gives us the cleanest timing windows and strongest evidence clips."
    if detail_tier == "high_detail":
        return "Strong slow-motion capture. This clip has enough frame density for a high-detail read."
    if fps >= 60.0:
        return "Solid fallback capture. We can analyze it, but the proof moments will be less precise than true slow-motion."
    return "Supported fallback only. Uploading the original slow-motion clip from your phone will give you a much stronger read."


def inspect_video_path(video_path: str | Path) -> schemas.MediaProfile:
    path = Path(video_path).expanduser().resolve()
    ffprobe_path = shutil.which("ffprobe")
    width = 0
    height = 0
    fps = 0.0
    duration_s = 0.0
    frame_count = 0
    codec = None
    bitrate = None
    rotation = 0

    if ffprobe_path:
        command = [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,codec_name,bit_rate:stream_tags=rotate:side_data=rotation:format=duration,bit_rate",
            "-of",
            "json",
            str(path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode == 0 and completed.stdout:
            payload = json.loads(completed.stdout)
            streams = payload.get("streams") or []
            stream = streams[0] if streams else {}
            fmt = payload.get("format") or {}
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
            fps = _parse_frame_rate(stream.get("avg_frame_rate"))
            duration_s = float(fmt.get("duration") or 0.0)
            frame_count = int(float(stream.get("nb_frames") or 0) or 0)
            codec = stream.get("codec_name")
            stream_bitrate = stream.get("bit_rate")
            bitrate = int(stream_bitrate or fmt.get("bit_rate") or 0) or None
            side_data_list = stream.get("side_data_list") or stream.get("side_data") or []
            for item in side_data_list:
                try:
                    rotation = int(item.get("rotation") or 0)
                    break
                except (TypeError, ValueError):
                    continue
            if rotation == 0:
                rotate_tag = (stream.get("tags") or {}).get("rotate")
                if rotate_tag:
                    try:
                        rotation = int(rotate_tag)
                    except ValueError:
                        rotation = 0

    if width <= 0 or height <= 0 or fps <= 0.0 or duration_s <= 0.0:
        cap = cv2.VideoCapture(str(path))
        try:
            width = width or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = height or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = fps or float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            frame_count = frame_count or int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration_s = duration_s or (frame_count / fps if fps else 0.0)
        finally:
            cap.release()

    if frame_count <= 0 and fps > 0.0 and duration_s > 0.0:
        frame_count = int(round(duration_s * fps))

    detail_tier = classify_detail_tier(width, height, fps)
    return schemas.MediaProfile(
        width=max(width, 1),
        height=max(height, 1),
        fps=max(fps, 0.0),
        duration_s=max(duration_s, 0.0),
        frame_count=max(frame_count, 0),
        codec=codec,
        bitrate=bitrate,
        rotation=rotation,
        detail_tier=detail_tier,
        capture_assessment=capture_assessment_for_profile(width, height, fps),
    )


def derive_sample_stride(source_fps: float, target_fps: float | None) -> int:
    if not target_fps or target_fps <= 0 or source_fps <= 0:
        return 1
    return max(int(round(source_fps / target_fps)), 1)


def _resize_frame(frame: np.ndarray, max_dimension: int | None) -> np.ndarray:
    if not max_dimension:
        return frame
    height, width = frame.shape[:2]
    current_max = max(height, width)
    if current_max <= max_dimension:
        return frame
    scale = max_dimension / float(current_max)
    new_width = max(int(round(width * scale)), 1)
    new_height = max(int(round(height * scale)), 1)
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def decode_video_path(
    video_path: str | Path,
    *,
    target_fps: float | None = None,
    sample_every_n: int | None = None,
    start_s: float = 0.0,
    duration_s: float | None = None,
    max_dimension: int | None = None,
) -> Tuple[List[np.ndarray], float, int]:
    path = Path(video_path).expanduser().resolve()
    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        stride = sample_every_n or derive_sample_stride(fps, target_fps)
        if start_s > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
        frames: List[np.ndarray] = []
        idx = 0
        end_s = start_s + duration_s if duration_s is not None else None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            current_time_s = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            if end_s is not None and current_time_s > end_s:
                break
            if idx % stride == 0:
                frames.append(_resize_frame(frame, max_dimension))
            idx += 1
        return frames, fps, stride
    finally:
        cap.release()


def decode_video(
    file_bytes: bytes,
    sample_every_n: int = 2,
) -> Tuple[List[np.ndarray], float, int]:
    path = _write_temp_video(file_bytes)
    try:
        return decode_video_path(path, sample_every_n=sample_every_n)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _ffmpeg_scale_filter(max_dimension: int | None) -> Optional[str]:
    if not max_dimension:
        return None
    return (
        f"scale='if(gt(iw,ih),min({max_dimension},iw),-2)':'if(gt(ih,iw),min({max_dimension},ih),-2)'"
    )


def transcode_video(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_dimension: int | None = None,
    target_fps: float | None = None,
    crf: int = 22,
) -> Path:
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        shutil.copy2(source, destination)
        return destination

    filters: List[str] = []
    scale_filter = _ffmpeg_scale_filter(max_dimension)
    if scale_filter:
        filters.append(scale_filter)
    if target_fps and target_fps > 0:
        filters.append(f"fps={target_fps:g}")

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if filters:
        command.extend(["-vf", ",".join(filters)])
    command.append(str(destination))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg transcode failed for {source.name}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return destination


def normalize_video(input_path: str | Path, output_path: str | Path) -> Path:
    return transcode_video(input_path, output_path, crf=20)


def create_analysis_rendition(input_path: str | Path, output_path: str | Path) -> Path:
    return transcode_video(
        input_path,
        output_path,
        max_dimension=DEFAULT_COARSE_MAX_DIMENSION,
        target_fps=DEFAULT_COARSE_TARGET_FPS,
        crf=24,
    )


def extract_subclip(
    input_path: str | Path,
    output_path: str | Path,
    *,
    start_s: float,
    duration_s: float,
) -> Path:
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        shutil.copy2(source, destination)
        return destination

    command = [
        ffmpeg_path,
        "-y",
        "-ss",
        f"{max(start_s, 0.0):.3f}",
        "-i",
        str(source),
        "-t",
        f"{max(duration_s, 0.05):.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg subclip failed for {source.name}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return destination


def effective_sample_fps(source_fps: float, stride: int) -> float:
    if source_fps <= 0:
        return 0.0
    return source_fps / max(stride, 1)


def dense_target_fps_for_profile(profile: schemas.MediaProfile | None) -> float:
    if profile is None:
        return DEFAULT_DENSE_TARGET_FPS
    if profile.fps >= 200.0:
        return 60.0
    if profile.fps >= 120.0:
        return 48.0
    return min(max(profile.fps, 30.0), 60.0)


def coarse_target_fps_for_profile(profile: schemas.MediaProfile | None) -> float:
    if profile is None:
        return DEFAULT_COARSE_TARGET_FPS
    return min(DEFAULT_COARSE_TARGET_FPS, max(profile.fps, 12.0))
