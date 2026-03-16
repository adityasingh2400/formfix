import unittest

from backend.src.services import analyzer


class ResolveTrackingSideTests(unittest.TestCase):
    def test_auto_mode_uses_highest_composite_score(self) -> None:
        """When no user selection, use auto-detection based on scores."""
        side, flipped = analyzer._resolve_tracking_side(
            None,
            {"left": 2.0, "right": 5.0},
            {"left": 3.0, "right": 6.0},
        )

        self.assertEqual(side, "right")
        self.assertFalse(flipped)

    def test_user_right_inverts_to_mediapipe_left(self) -> None:
        """User's 'right' hand inverts to MediaPipe 'left'."""
        side, flipped = analyzer._resolve_tracking_side(
            "right",
            {"left": 0.0, "right": 0.0},
            {"left": 0.0, "right": 0.0},
        )

        self.assertEqual(side, "left")
        self.assertTrue(flipped)

    def test_user_left_inverts_to_mediapipe_right(self) -> None:
        """User's 'left' hand inverts to MediaPipe 'right'."""
        side, flipped = analyzer._resolve_tracking_side(
            "left",
            {"left": 0.0, "right": 0.0},
            {"left": 0.0, "right": 0.0},
        )

        self.assertEqual(side, "right")
        self.assertTrue(flipped)

    def test_auto_mode_left_wins(self) -> None:
        """Auto-detection picks left when left scores higher."""
        side, flipped = analyzer._resolve_tracking_side(
            None,
            {"left": 5.0, "right": 2.0},
            {"left": 6.0, "right": 3.0},
        )

        self.assertEqual(side, "left")
        self.assertFalse(flipped)


if __name__ == "__main__":
    unittest.main()
