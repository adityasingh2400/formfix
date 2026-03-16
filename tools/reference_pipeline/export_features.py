#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.src.services import analyzer, comparison  # noqa: E402

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export FormFix-ready feature rows from a directory of shot videos."
    )
    parser.add_argument("--video-dir", required=True, help="Directory containing shot clips.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument(
        "--metadata-csv",
        help="Optional CSV keyed by filename. Columns like shot_type, player_id, player_name are passed through.",
    )
    parser.add_argument(
        "--default-shot-type",
        default=None,
        help="Fallback shot type when metadata does not include one.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively.",
    )
    return parser.parse_args()


def load_metadata(path: str | None) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}

    metadata: Dict[str, Dict[str, str]] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filename = row.get("filename") or row.get("file") or row.get("clip")
            if not filename:
                continue
            metadata[filename] = {key: value for key, value in row.items() if value}
    return metadata


def iter_video_files(video_dir: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in sorted(video_dir.glob(pattern)):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            yield path


def main() -> int:
    args = parse_args()
    video_dir = Path(args.video_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.metadata_csv)

    processed = 0
    failures = 0
    with output_path.open("w") as handle:
        for clip_path in iter_video_files(video_dir, args.recursive):
            clip_meta = metadata.get(clip_path.name, {})
            shot_type = clip_meta.get("shot_type") or args.default_shot_type

            try:
                prepared = analyzer._prepare_analysis(clip_path.read_bytes(), shot_type=shot_type)
            except Exception as exc:  # pragma: no cover - utility script
                failures += 1
                print(f"skip {clip_path.name}: {exc}", file=sys.stderr)
                continue

            row = comparison.build_training_row(
                prepared.phases,
                shot_type,
                metadata={
                    **clip_meta,
                    "filename": clip_path.name,
                    "path": str(clip_path),
                },
            )
            row.update(
                {
                    "clip_id": clip_meta.get("clip_id") or clip_path.stem,
                    "issues": [issue.model_dump() for issue in prepared.issues],
                    "phases": [phase.model_dump() for phase in prepared.phases],
                    "confidence_notes": list(prepared.confidence_notes),
                }
            )
            handle.write(json.dumps(row) + "\n")
            processed += 1
            print(f"exported {clip_path.name}", file=sys.stderr)

    print(
        json.dumps(
            {
                "processed": processed,
                "failures": failures,
                "output": str(output_path),
            }
        )
    )
    return 0 if processed else 1


if __name__ == "__main__":
    raise SystemExit(main())
