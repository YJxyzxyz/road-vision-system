from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, TYPE_CHECKING

import numpy as np

from src.detect import Detection

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from src.io_video.capture import Frame, VideoSource


@dataclass(slots=True)
class ProcessResult:
    """处理结果结构体."""

    raw: np.ndarray
    proc: np.ndarray
    ts: float
    detections: List[Detection]


_SENTINEL = object()


class AsyncPipeline:
    """多线程视频采集 + 处理流水线."""

    def __init__(
        self,
        video_source: "VideoSource",
        processor: Callable[["Frame"], Optional[ProcessResult]],
        queue_size: int = 4,
        drop_policy: str = "oldest",
        max_latency: Optional[float] = None,
        result_timeout: float = 0.25,
    ) -> None:
        self.video_source = video_source
        self._processor = processor
        self._frame_queue: "queue.Queue[object]" = queue.Queue(max(1, int(queue_size)))
        self._result_queue: "queue.Queue[object]" = queue.Queue(max(1, int(queue_size)))
        self._drop_policy = (drop_policy or "oldest").lower()
        self._max_latency = max_latency if max_latency and max_latency > 0 else None
        self._result_timeout = max(0.01, float(result_timeout))
        self._stop_event = threading.Event()
        self._stopped = False
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._stats_lock = threading.Lock()
        self._dropped_frames = 0
        self._skipped_frames = 0

    def start(self) -> None:
        if not self._capture_thread.is_alive():
            self._capture_thread.start()
        if not self._worker_thread.is_alive():
            self._worker_thread.start()

    def close(self) -> None:
        self._stop_event.set()
        self._push_frame(_SENTINEL, force=True)
        self._push_result(_SENTINEL, force=True)
        self._capture_thread.join(timeout=1.0)
        self._worker_thread.join(timeout=1.0)
        self._stopped = True

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def dropped_frames(self) -> int:
        with self._stats_lock:
            return self._dropped_frames

    @property
    def skipped_frames(self) -> int:
        with self._stats_lock:
            return self._skipped_frames

    def get_result(self, timeout: Optional[float] = None) -> Optional[ProcessResult]:
        if self._stopped:
            return None
        try:
            item = self._result_queue.get(timeout=timeout if timeout is not None else self._result_timeout)
        except queue.Empty:
            return None
        if item is _SENTINEL:
            self._stopped = True
            return None
        return item  # type: ignore[return-value]

    # 内部实现 ---------------------------------------------------------------

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            frame = self.video_source.read()
            if not frame.ok:
                self._push_frame(_SENTINEL, force=True)
                break
            self._push_frame(frame)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            frame = self._get_frame()
            if frame is None:
                break
            try:
                result = self._processor(frame)
            except Exception as exc:  # pragma: no cover - 运行期防护
                print(f"⚠️ 异步处理异常: {exc}")
                continue
            if result is None:
                continue
            self._push_result(result)
        self._push_result(_SENTINEL, force=True)
        self._stopped = True

    def _push_frame(self, frame: object, force: bool = False) -> None:
        while not self._stop_event.is_set():
            try:
                self._frame_queue.put(frame, timeout=0.05)
                return
            except queue.Full:
                if frame is _SENTINEL:
                    force = True
                if not force and self._drop_policy == "newest":
                    with self._stats_lock:
                        self._dropped_frames += 1
                    return
                try:
                    self._frame_queue.get_nowait()
                    with self._stats_lock:
                        self._dropped_frames += 1
                except queue.Empty:
                    pass

    def _push_result(self, result: object, force: bool = False) -> None:
        while True:
            try:
                self._result_queue.put(result, timeout=0.05)
                return
            except queue.Full:
                if result is _SENTINEL:
                    force = True
                if not force:
                    try:
                        self._result_queue.get_nowait()
                    except queue.Empty:
                        pass
                else:
                    try:
                        self._result_queue.get_nowait()
                    except queue.Empty:
                        pass

    def _get_frame(self) -> Optional["Frame"]:
        while not self._stop_event.is_set():
            try:
                item = self._frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is _SENTINEL:
                return None
            frame = item  # type: ignore[assignment]
            if self._max_latency is not None:
                age = time.time() - frame.ts
                if age > self._max_latency:
                    with self._stats_lock:
                        self._skipped_frames += 1
                    continue
            return frame
        return None
