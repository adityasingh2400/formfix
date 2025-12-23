import os
import tempfile
from typing import List, Tuple

import cv2
import numpy as np


def _write_temp_video(file_bytes: bytes, suffix: str = ".mp4") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name


def decode_video(
    file_bytes: bytes, sample_every_n: int = 2
) -> Tuple[List[np.ndarray], float, int]:
    """
    Decode video bytes and return sampled frames (BGR), FPS, and stride.
    """
    path = _write_temp_video(file_bytes)
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames: List[np.ndarray] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % sample_every_n == 0:
                frames.append(frame)
            idx += 1
        return frames, fps, sample_every_n
    finally:
        cap.release()
        try:
            os.unlink(path)
        except OSError:
            pass

