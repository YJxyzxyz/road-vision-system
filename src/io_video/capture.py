from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import cv2


@dataclass(slots=True)
class Frame:
    ok: bool
    image: Any
    ts: float  # 捕获时间戳（秒，float）


def _looks_like_gstreamer(source: Any) -> bool:
    if not isinstance(source, str):
        return False
    tokens = source.strip().split()
    if not tokens:
        return False
    head = tokens[0]
    return "!" in source or head.startswith("nvarguscamerasrc") or head.startswith("v4l2src")


class VideoSource:
    def __init__(self, source=0, width=1280, height=720, fps_request=30, backend="auto"):
        self.source = source
        self.backend = (backend or "auto").lower()
        self.cap = None
        self._backend_used = None
        self._open(width, height, fps_request)

    def _open(self, width: int, height: int, fps_request: int) -> None:
        prefer_gst = self.backend == "gstreamer" or (
            self.backend == "auto" and _looks_like_gstreamer(self.source)
        )

        cap = None
        backend_used = None

        if prefer_gst:
            cap = cv2.VideoCapture(self.source, cv2.CAP_GSTREAMER)
            backend_used = "gstreamer"
            if not cap.isOpened() and self.backend == "auto":
                cap.release()
                cap = None

        if cap is None:
            cap = cv2.VideoCapture(self.source)
            backend_used = "default"

        if not cap or not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self.source}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps_request)

        self.cap = cap
        self._backend_used = backend_used

    def read(self) -> Frame:
        if not self.cap:
            raise RuntimeError("VideoSource 尚未初始化")
        ok, img = self.cap.read()
        ts = time.time()
        return Frame(ok, img, ts)

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
