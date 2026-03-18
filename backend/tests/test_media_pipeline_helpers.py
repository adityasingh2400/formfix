import unittest
from pathlib import Path

from backend.src import schemas
from backend.src.services import analyzer
from backend.src.services.media_store import LocalMediaStore
from backend.src.services.video_utils import (
    capture_assessment_for_profile,
    classify_detail_tier,
    derive_sample_stride,
)


class MediaPipelineHelperTests(unittest.TestCase):
    def test_classify_ultra_detail_profile(self) -> None:
        self.assertEqual(classify_detail_tier(3840, 2160, 240.0), "ultra_detail")

    def test_classify_standard_profile(self) -> None:
        self.assertEqual(classify_detail_tier(1280, 720, 30.0), "standard_detail")

    def test_capture_assessment_mentions_ideal_for_ultra_detail(self) -> None:
        assessment = capture_assessment_for_profile(3840, 2160, 240.0)
        self.assertIn("Ideal", assessment)

    def test_stride_targets_dense_sampling(self) -> None:
        self.assertEqual(derive_sample_stride(240.0, 60.0), 4)

    def test_local_media_store_creates_media_urls(self) -> None:
        store = LocalMediaStore(root=Path("/tmp/formfix-test-store"))
        job_paths = store.ensure_job_dirs("job-1")
        sample_path = job_paths["artifacts"] / "annotated-replay.mp4"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_bytes(b"demo")
        self.assertEqual(
            store.media_url(sample_path),
            "/media/jobs/job-1/artifacts/annotated-replay.mp4",
        )

    def test_playback_script_uses_detail_tier(self) -> None:
        script = analyzer._build_playback_script(
            [
                schemas.PlaybackCue(
                    id="cue-1",
                    title="Cue",
                    cue="Cue body",
                    why="Why",
                    research_basis="Reason",
                    phase="release",
                    timestamp=1.2,
                    bubble_x=0.5,
                    bubble_y=0.5,
                    anchors=[],
                    confidence=0.9,
                )
            ],
            [schemas.PhaseBoundary(phase="release", start=0.9, end=1.4)],
            schemas.MediaProfile(
                width=3840,
                height=2160,
                fps=240.0,
                duration_s=2.0,
                frame_count=480,
                detail_tier="ultra_detail",
                capture_assessment="Ideal",
            ),
        )
        self.assertEqual(len(script), 1)
        self.assertLess(script[0].focus_rate, 0.4)
        self.assertGreaterEqual(script[0].freeze_ms, 1000)


if __name__ == "__main__":
    unittest.main()
