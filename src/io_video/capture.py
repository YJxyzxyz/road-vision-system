from __future__ import annotations

import time
from typing import Any, Dict

import cv2


class Frame:
    __slots__ = ("ok", "image", "ts", "index")

    def __init__(self, ok: bool, image, ts: float, index: int | None = None):
        self.ok = ok
        self.image = image
        self.ts = ts  # 捕获时间戳（秒，float）
        self.index = index


class VideoSource:
    def __init__(self, config: Dict[str, Any]):
        cfg = config or {}
        self.source = cfg.get("source", 0)
        self.width = int(cfg.get("width", 1280))
        self.height = int(cfg.get("height", 720))
        self.fps_request = cfg.get("fps_request", 30)
        self.backend = (cfg.get("backend") or "auto").lower()
        self._counter = 0

        if self.backend == "gstreamer":
            pipeline = self._build_gstreamer_pipeline(cfg.get("gstreamer", {}))
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps_request)
            buffersize = int(cfg.get("buffersize", 2))
            try:
                if buffersize > 0:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)
            except Exception:
                pass

        if not self.is_opened():
            print("⚠️ 视频源未成功打开，请检查 camera 配置")

    def _build_gstreamer_pipeline(self, gst_cfg: Dict[str, Any]) -> str:
        cfg = gst_cfg or {}
        pipeline = (cfg.get("pipeline") or "").strip()
        if pipeline:
            return pipeline

        sensor_id = int(cfg.get("sensor_id", 0))
        capture_width = int(cfg.get("capture_width", self.width))
        capture_height = int(cfg.get("capture_height", self.height))
        display_width = int(cfg.get("display_width", self.width))
        display_height = int(cfg.get("display_height", self.height))
        framerate = cfg.get("framerate", self.fps_request)
        if isinstance(framerate, (list, tuple)) and len(framerate) == 2:
            fr_num, fr_den = int(framerate[0]), int(framerate[1]) or 1
        else:
            fr_num, fr_den = int(framerate), 1
        flip_method = int(cfg.get("flip_method", 0))

        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
            f"format=(string)NV12, framerate=(fraction){fr_num}/{fr_den} ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
            "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1"
        )

    def read(self) -> Frame:
        if not self.cap:
            return Frame(False, None, time.time(), None)
        ok, img = self.cap.read()
        ts = time.time()
        idx = self._counter
        self._counter += 1
        return Frame(ok, img, ts, idx)

    def is_opened(self) -> bool:
        return bool(self.cap) and bool(self.cap.isOpened())

    def release(self):
        if self.cap:
            self.cap.release()
