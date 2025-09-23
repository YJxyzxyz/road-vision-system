from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..detect import Detection, build_detector
from ..io_video.capture import VideoSource
from ..preprocess import PreprocessPipeline
from .config import RuntimeConfigStore


@dataclass(slots=True)
class FramePacket:
    seq_id: int
    image: np.ndarray
    timestamp: float
    capture_ms: float


@dataclass(slots=True)
class InferencePacket:
    seq_id: int
    timestamp: float
    raw: np.ndarray
    processed: np.ndarray
    detections: List[Detection]
    capture_ms: float
    profile: Dict[str, float]


class _CaptureThread(threading.Thread):
    def __init__(
        self,
        camera_cfg: dict,
        frame_queue: queue.Queue,
        stop_event: threading.Event,
        drop_oldest: bool,
        num_consumers: int,
    ) -> None:
        super().__init__(daemon=True)
        self._camera_cfg = camera_cfg
        self._queue = frame_queue
        self._stop_event = stop_event
        self._drop_oldest = drop_oldest
        self._num_consumers = num_consumers
        self.error: Optional[Exception] = None

    def request_stop(self) -> None:
        self._stop_event.set()

    def _open_source(self) -> VideoSource:
        cfg = self._camera_cfg or {}
        return VideoSource(
            source=cfg.get("source", 0),
            width=int(cfg.get("width", 1280)),
            height=int(cfg.get("height", 720)),
            fps_request=int(cfg.get("fps_request", 30)),
            backend=cfg.get("backend", "auto"),
        )

    def run(self) -> None:  # pragma: no cover - background thread
        seq_id = 0
        source: Optional[VideoSource] = None
        try:
            source = self._open_source()
            while not self._stop_event.is_set():
                t0 = time.perf_counter()
                frame = source.read()
                capture_ms = (time.perf_counter() - t0) * 1000.0
                if not frame.ok:
                    break
                packet = FramePacket(
                    seq_id=seq_id,
                    image=frame.image,
                    timestamp=frame.ts,
                    capture_ms=capture_ms,
                )
                seq_id += 1
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(packet, timeout=0.05)
                        break
                    except queue.Full:
                        if not self._drop_oldest:
                            continue
                        try:
                            dropped = self._queue.get_nowait()
                            if dropped is None:
                                # Sentinel from stop(), requeue it
                                self._queue.put_nowait(None)
                        except queue.Empty:
                            pass
        except Exception as exc:  # pragma: no cover - log and propagate later
            self.error = exc
        finally:
            if source is not None:
                source.release()
            for _ in range(self._num_consumers):
                try:
                    self._queue.put_nowait(None)
                except queue.Full:
                    while True:
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            break
                    self._queue.put(None)


class _InferWorker(threading.Thread):
    def __init__(
        self,
        worker_id: int,
        frame_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        config_store: RuntimeConfigStore,
    ) -> None:
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self._frame_queue = frame_queue
        self._output_queue = output_queue
        self._stop_event = stop_event
        self._config_store = config_store
        self._pipeline: Optional[PreprocessPipeline] = None
        self._detector = None
        self._local_version = -1

    def _close_modules(self) -> None:
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None
        self._pipeline = None

    def _rebuild(self, cfg: dict) -> None:
        self._close_modules()
        pp_cfg = cfg.get("preprocess", {}) or {}
        det_cfg = cfg.get("detect", {}) or {}
        self._pipeline = PreprocessPipeline(pp_cfg)
        if det_cfg.get("enabled", False):
            self._detector = build_detector(det_cfg)
        else:
            self._detector = None

    def run(self) -> None:  # pragma: no cover - background thread
        try:
            while not self._stop_event.is_set():
                if self._local_version != self._config_store.version():
                    version, cfg = self._config_store.snapshot()
                    try:
                        self._rebuild(cfg)
                    except Exception as exc:
                        print(f"⚠️ Worker{self.worker_id} 重建失败: {exc}")
                        time.sleep(0.5)
                        continue
                    self._local_version = version

                try:
                    packet = self._frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if packet is None:
                    self._output_queue.put(None)
                    break

                raw = packet.image
                proc = raw.copy()
                profile = {"preprocess_ms": 0.0, "infer_ms": 0.0}

                t0 = time.perf_counter()
                if self._pipeline is not None:
                    try:
                        proc = self._pipeline(proc, ts=packet.timestamp)
                    except Exception as exc:
                        print(f"⚠️ 预处理失败: {exc}")
                        proc = raw.copy()
                t1 = time.perf_counter()

                dets: List[Detection] = []
                if self._detector is not None:
                    try:
                        dets = self._detector.infer(proc)
                    except Exception as exc:
                        print(f"⚠️ 推理失败: {exc}")
                        dets = []
                t2 = time.perf_counter()

                profile["preprocess_ms"] = (t1 - t0) * 1000.0
                profile["infer_ms"] = (t2 - t1) * 1000.0

                self._output_queue.put(
                    InferencePacket(
                        seq_id=packet.seq_id,
                        timestamp=packet.timestamp,
                        raw=raw,
                        processed=proc,
                        detections=dets,
                        capture_ms=packet.capture_ms,
                        profile=profile,
                    )
                )
        finally:
            self._close_modules()


class PerfAggregator:
    def __init__(self, interval: int = 60) -> None:
        self.interval = max(1, int(interval))
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.capture_ms = 0.0
        self.preprocess_ms = 0.0
        self.detect_ms = 0.0
        self.track_ms = 0.0

    def update(
        self,
        capture: Optional[float] = None,
        preprocess: Optional[float] = None,
        detect: Optional[float] = None,
        track: Optional[float] = None,
    ) -> None:
        self.count += 1
        if capture is not None:
            self.capture_ms += capture
        if preprocess is not None:
            self.preprocess_ms += preprocess
        if detect is not None:
            self.detect_ms += detect
        if track is not None:
            self.track_ms += track

    def maybe_log(
        self,
        queue_in: Tuple[int, int],
        queue_out: Tuple[int, int],
    ) -> None:
        if self.count < self.interval:
            return
        denom = max(1, self.count)
        avg_capture = self.capture_ms / denom
        avg_pre = self.preprocess_ms / denom
        avg_det = self.detect_ms / denom
        avg_track = self.track_ms / denom
        qi, qim = queue_in
        qo, qom = queue_out
        print(
            f"[Perf] capture={avg_capture:.1f}ms preprocess={avg_pre:.1f}ms "
            f"detect={avg_det:.1f}ms track={avg_track:.1f}ms "
            f"queue_in={qi}/{qim} queue_out={qo}/{qom}"
        )
        self.reset()


class AsyncPipeline:
    def __init__(
        self,
        camera_cfg: dict,
        async_cfg: dict,
        config_store: RuntimeConfigStore,
    ) -> None:
        self._stop_event = threading.Event()
        capture_cfg = (async_cfg or {}).get("capture", {}) or {}
        infer_cfg = (async_cfg or {}).get("infer", {}) or {}
        output_cfg = (async_cfg or {}).get("output", {}) or {}

        capture_queue_size = max(1, int(capture_cfg.get("queue_size", 4)))
        output_queue_size = max(1, int(output_cfg.get("queue_size", 2)))
        self.frame_queue: queue.Queue = queue.Queue(maxsize=capture_queue_size)
        self.output_queue: queue.Queue = queue.Queue(maxsize=output_queue_size)

        self._workers_num = max(1, int(infer_cfg.get("workers", 1)))
        self.profile_interval = max(1, int(infer_cfg.get("profile_interval", 60)))

        drop_oldest = bool(capture_cfg.get("drop_oldest", True))
        self._capture_thread = _CaptureThread(
            camera_cfg=camera_cfg,
            frame_queue=self.frame_queue,
            stop_event=self._stop_event,
            drop_oldest=drop_oldest,
            num_consumers=self._workers_num,
        )

        self._workers = [
            _InferWorker(i, self.frame_queue, self.output_queue, self._stop_event, config_store)
            for i in range(self._workers_num)
        ]

        self._buffer: Dict[int, InferencePacket] = {}
        self._expected_seq = 0
        self._active_workers = self._workers_num
        self._finished = False

    def start(self) -> None:
        self._capture_thread.start()
        for worker in self._workers:
            worker.start()

    def stop(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self._capture_thread.request_stop()
        for _ in range(self._workers_num):
            try:
                self.frame_queue.put_nowait(None)
            except queue.Full:
                pass
        self._capture_thread.join(timeout=2.0)
        for worker in self._workers:
            worker.join(timeout=2.0)
        self._finished = True
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break
        self.output_queue.put(None)

    def is_finished(self) -> bool:
        return self._finished and not self._buffer

    def get_next(self, timeout: float = 0.2) -> Optional[InferencePacket]:
        if self.is_finished():
            return None

        while True:
            buffered = self._buffer.pop(self._expected_seq, None)
            if buffered is not None:
                self._expected_seq += 1
                return buffered

            if self._finished:
                return None

            try:
                item = self.output_queue.get(timeout=timeout)
            except queue.Empty:
                if self._stop_event.is_set() and self._active_workers == 0:
                    self._finished = True
                return None

            if item is None:
                self._active_workers -= 1
                if self._active_workers <= 0:
                    self._finished = True
                continue

            self._buffer[item.seq_id] = item

    def queue_status(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (
            (self.frame_queue.qsize(), self.frame_queue.maxsize),
            (self.output_queue.qsize(), self.output_queue.maxsize),
        )

    @property
    def capture_error(self) -> Optional[Exception]:
        return self._capture_thread.error


__all__ = ["AsyncPipeline", "InferencePacket", "PerfAggregator"]
