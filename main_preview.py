from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from src.config import load_config, resolve_config_path
from src.detect import build_detector
from src.geometry import build_projector, GroundProjector
from src.io_video.capture import VideoSource
from src.io_video.fps_meter import FPSMeter
from src.preprocess import PreprocessPipeline
from src.runtime import AsyncPipeline, ConfigWatcher, PerfAggregator, RuntimeConfigStore
from src.track import build_tracker
from src.vis import draw_detections


def make_canvas(
    raw_bgr: np.ndarray,
    proc_bgr: np.ndarray,
    layout: str = "h",
    divider_px: int = 4,
    label_raw: str = "RAW",
    label_proc: str = "PROC",
    fps: float | None = None,
    show_fps: bool = True,
) -> np.ndarray:
    h, w = raw_bgr.shape[:2]
    divider_px = max(0, int(divider_px))

    def put_label(img: np.ndarray, org: Tuple[int, int], text: str, color=(50, 220, 50)) -> None:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    if layout.lower() == "v":
        divider = (
            np.full((divider_px, w, 3), (40, 40, 40), dtype=np.uint8)
            if divider_px
            else None
        )
        canvas = (
            np.vstack([raw_bgr, divider, proc_bgr])
            if divider is not None
            else np.vstack([raw_bgr, proc_bgr])
        )
        put_label(canvas, (10, 30), label_raw)
        put_label(canvas, (10, h + divider_px + 30), label_proc, color=(0, 200, 255))
    else:
        divider = (
            np.full((h, divider_px, 3), (40, 40, 40), dtype=np.uint8)
            if divider_px
            else None
        )
        canvas = (
            np.hstack([raw_bgr, divider, proc_bgr])
            if divider is not None
            else np.hstack([raw_bgr, proc_bgr])
        )
        put_label(canvas, (10, 30), label_raw)
        put_label(canvas, (w + divider_px + 10, 30), label_proc, color=(0, 200, 255))

    if show_fps and fps is not None:
        put_label(canvas, (10, max(60, h - 10)), f"FPS: {fps:.1f}", color=(0, 255, 255))
    return canvas


def _build_tracker_from_cfg(cfg: dict):
    track_cfg = cfg.get("tracking", {}) or {}
    if not track_cfg.get("enabled", False):
        return None
    try:
        return build_tracker(track_cfg)
    except Exception as exc:
        print(f"⚠️ 跟踪器初始化失败: {exc}")
        return None


def _build_projector_from_cfg(cfg: dict):
    geom_cfg = cfg.get("geometry", {}) or {}
    if not geom_cfg.get("enabled", False):
        return None
    try:
        return build_projector(geom_cfg)
    except Exception as exc:
        print(f"⚠️ 几何投影初始化失败: {exc}")
        return None


def _apply_geometry_without_tracker(dets, projector: GroundProjector | None) -> None:
    if projector is None:
        return
    for d in dets:
        dist = projector.distance_for_bbox((d.x1, d.y1, d.x2, d.y2))
        if dist is not None:
            d.distance_m = dist


def _extract_preview_cfg(cfg: dict):
    preview_cfg = cfg.get("preview", {}) or {}
    compare_cfg = preview_cfg.get("compare", {}) or {}
    record_cfg = preview_cfg.get("record", {}) or {}
    vis_cfg = cfg.get("vis", {}) or {}
    draw_cfg = vis_cfg.get("draw", {}) or {}
    return preview_cfg, compare_cfg, record_cfg, draw_cfg


def run_sync(_cfg_file: Path, cfg: dict) -> None:
    cam_cfg = cfg.get("camera", {}) or {}
    preview_cfg, compare_cfg, record_cfg, draw_cfg = _extract_preview_cfg(cfg)
    pp_cfg = cfg.get("preprocess", {}) or {}
    det_cfg = cfg.get("detect", {}) or {}

    vs = VideoSource(
        source=cam_cfg.get("source", 0),
        width=int(cam_cfg.get("width", 1280)),
        height=int(cam_cfg.get("height", 720)),
        fps_request=int(cam_cfg.get("fps_request", 30)),
        backend=cam_cfg.get("backend", "auto"),
    )
    fpsm = FPSMeter(alpha=0.1)
    pipeline = PreprocessPipeline(pp_cfg)

    detector = build_detector(det_cfg) if det_cfg.get("enabled", False) else None
    tracker = _build_tracker_from_cfg(cfg)
    projector = _build_projector_from_cfg(cfg)

    writer = None  # TODO: 实现录像输出
    want_record = bool(record_cfg.get("enable", False))
    if want_record:
        print("⚠️ 录像输出尚未实现")
    want_compare = bool(compare_cfg.get("enable", True))
    layout = compare_cfg.get("layout", "h")
    divider_px = int(compare_cfg.get("divider_px", 4))

    try:
        while True:
            fr = vs.read()
            if not fr.ok:
                print("⚠️ 读取失败/视频结束")
                break
            raw = fr.image
            proc = pipeline(raw.copy(), ts=fr.ts)

            dets = detector.infer(proc) if detector is not None else []
            if tracker is not None:
                display_dets = tracker.update(dets, fr.ts, projector=projector)
            else:
                display_dets = dets
                _apply_geometry_without_tracker(display_dets, projector)

            if draw_cfg.get("det", True) and display_dets:
                draw_detections(
                    proc,
                    display_dets,
                    thickness=int(draw_cfg.get("thickness", 2)),
                    font_scale=float(draw_cfg.get("font_scale", 0.6)),
                )

            fps = fpsm.tick(fr.ts)

            if want_compare:
                canvas = make_canvas(
                    raw,
                    proc,
                    layout=layout,
                    divider_px=divider_px,
                    label_raw=compare_cfg.get("label_raw", "RAW"),
                    label_proc=compare_cfg.get("label_proc", "PROC"),
                    fps=fps,
                    show_fps=bool(preview_cfg.get("show_fps", True)),
                )
                if writer:
                    writer.write(canvas)
                cv2.imshow("Compare Preview", canvas)
            else:
                frame_disp = proc.copy()
                if preview_cfg.get("show_fps", True):
                    cv2.putText(
                        frame_disp,
                        f"FPS:{fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2,
                    )
                if writer:
                    writer.write(frame_disp)
                cv2.imshow("Preview", frame_disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        if writer:
            writer.release()
        if tracker:
            tracker.close()
        if detector:
            detector.close()
        vs.release()
        cv2.destroyAllWindows()


def run_async(cfg_file: Path, initial_cfg: dict, async_cfg: dict) -> None:
    store = RuntimeConfigStore(initial_cfg)
    pipeline = AsyncPipeline(initial_cfg.get("camera", {}) or {}, async_cfg, store)
    pipeline.start()

    hot_cfg = (async_cfg or {}).get("hot_reload", {}) or {}
    watcher: ConfigWatcher | None = None
    if hot_cfg.get("enabled", False):
        watcher = ConfigWatcher(cfg_file, store, interval=float(hot_cfg.get("watch_interval", 2.0)))
        watcher.start()

    fpsm = FPSMeter(alpha=0.1)
    perf = PerfAggregator(pipeline.profile_interval)

    current_version = store.version()
    cfg = store.get_config()
    tracker = _build_tracker_from_cfg(cfg)
    projector = _build_projector_from_cfg(cfg)
    preview_cfg, compare_cfg, record_cfg, draw_cfg = _extract_preview_cfg(cfg)
    want_compare = bool(compare_cfg.get("enable", True))
    layout = compare_cfg.get("layout", "h")
    divider_px = int(compare_cfg.get("divider_px", 4))
    want_record = bool(record_cfg.get("enable", False))
    if want_record:
        print("⚠️ 录像输出尚未实现")
    writer = None

    try:
        while True:
            packet = pipeline.get_next(timeout=0.2)
            if packet is None:
                if pipeline.is_finished():
                    break
                continue

            if store.version() != current_version:
                current_version, cfg = store.snapshot()
                print("ℹ️ 检测到配置更新，重新初始化组件")
                if tracker:
                    tracker.close()
                tracker = _build_tracker_from_cfg(cfg)
                projector = _build_projector_from_cfg(cfg)
                preview_cfg, compare_cfg, record_cfg, draw_cfg = _extract_preview_cfg(cfg)
                want_compare = bool(compare_cfg.get("enable", True))
                layout = compare_cfg.get("layout", "h")
                divider_px = int(compare_cfg.get("divider_px", 4))
                want_record = bool(record_cfg.get("enable", False))
                if want_record:
                    print("⚠️ 录像输出尚未实现")

            detections = packet.detections
            t0 = time.perf_counter()
            if tracker is not None:
                display_dets = tracker.update(detections, packet.timestamp, projector=projector)
            else:
                display_dets = detections
                _apply_geometry_without_tracker(display_dets, projector)
            track_ms = (time.perf_counter() - t0) * 1000.0

            if draw_cfg.get("det", True) and display_dets:
                draw_detections(
                    packet.processed,
                    display_dets,
                    thickness=int(draw_cfg.get("thickness", 2)),
                    font_scale=float(draw_cfg.get("font_scale", 0.6)),
                )

            fps = fpsm.tick(packet.timestamp)

            perf.update(
                capture=packet.capture_ms,
                preprocess=packet.profile.get("preprocess_ms"),
                detect=packet.profile.get("infer_ms"),
                track=track_ms,
            )
            perf.maybe_log(*pipeline.queue_status())

            if want_compare:
                canvas = make_canvas(
                    packet.raw,
                    packet.processed,
                    layout=layout,
                    divider_px=divider_px,
                    label_raw=compare_cfg.get("label_raw", "RAW"),
                    label_proc=compare_cfg.get("label_proc", "PROC"),
                    fps=fps,
                    show_fps=bool(preview_cfg.get("show_fps", True)),
                )
                if writer:
                    writer.write(canvas)
                cv2.imshow("Compare Preview", canvas)
            else:
                frame_disp = packet.processed.copy()
                if preview_cfg.get("show_fps", True):
                    cv2.putText(
                        frame_disp,
                        f"FPS:{fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2,
                    )
                if writer:
                    writer.write(frame_disp)
                cv2.imshow("Preview", frame_disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        pipeline.stop()
        if watcher:
            watcher.stop()
            watcher.join(timeout=1.0)
        if writer:
            writer.release()
        if tracker:
            tracker.close()
        cv2.destroyAllWindows()
        if pipeline.capture_error:
            print(f"⚠️ 采集线程异常: {pipeline.capture_error}")


def main(cfg_path: str | None = None) -> None:
    cfg_file = resolve_config_path(cfg_path)
    cfg = load_config(str(cfg_file))
    runtime_cfg = (cfg.get("runtime", {}) or {}).get("async", {}) or {}
    if runtime_cfg.get("enabled", False):
        run_async(cfg_file, cfg, runtime_cfg)
    else:
        run_sync(cfg_file, cfg)


if __name__ == "__main__":
    main()
