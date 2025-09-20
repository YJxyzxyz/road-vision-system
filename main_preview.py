from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import load_config, resolve_config_path
from src.io_video.capture import Frame, VideoSource
from src.io_video.fps_meter import FPSMeter
from src.preprocess import PreprocessPipeline
from src.detect import build_detector
from src.track import build_tracker
from src.geometry import build_projector
from src.vis import draw_detections
from src.runtime import AsyncPipeline, ConfigWatcher, ProcessResult


class ProcessorBundle:
    def __init__(
        self,
        pipeline: PreprocessPipeline,
        detector,
        tracker,
        projector,
    ) -> None:
        self.pipeline = pipeline
        self.detector = detector
        self.tracker = tracker
        self.projector = projector

    def process(self, frame: Frame) -> Optional[ProcessResult]:
        if not frame.ok or frame.image is None:
            return None
        raw = frame.image
        proc = self.pipeline(raw, ts=frame.ts)
        if proc is raw:
            proc = raw.copy()
        dets = []
        if self.detector is not None:
            dets = self.detector.infer(proc)
        if self.tracker is not None:
            dets = self.tracker.update(dets, frame.ts, projector=self.projector)
        elif self.projector is not None:
            for d in dets:
                dist = self.projector.distance_for_bbox((d.x1, d.y1, d.x2, d.y2))
                if dist is not None:
                    d.distance_m = dist
        return ProcessResult(raw=raw, proc=proc, ts=frame.ts, detections=dets)

    def close(self) -> None:
        if self.tracker is not None:
            self.tracker.close()
        if self.detector is not None:
            self.detector.close()


def make_canvas(raw_bgr, proc_bgr, layout="h", divider_px=4,
                label_raw="RAW", label_proc="PROC", fps=None, show_fps=True):
    h, w = raw_bgr.shape[:2]
    divider_px = max(0, int(divider_px))

    def put_label(img, org, text, color=(50, 220, 50)):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    if layout.lower() == "v":
        divider = np.full((divider_px, w, 3), (40, 40, 40), dtype=np.uint8) if divider_px else None
        canvas = np.vstack([raw_bgr, divider, proc_bgr]) if divider is not None else np.vstack([raw_bgr, proc_bgr])
        put_label(canvas, (10, 30), label_raw)
        put_label(canvas, (10, h + divider_px + 30), label_proc, color=(0, 200, 255))
    else:
        divider = np.full((h, divider_px, 3), (40, 40, 40), dtype=np.uint8) if divider_px else None
        canvas = np.hstack([raw_bgr, divider, proc_bgr]) if divider is not None else np.hstack([raw_bgr, proc_bgr])
        put_label(canvas, (10, 30), label_raw)
        put_label(canvas, (w + divider_px + 10, 30), label_proc, color=(0, 200, 255))

    if show_fps and fps is not None:
        put_label(canvas, (10, max(60, h - 10)), f"FPS: {fps:.1f}", color=(0, 255, 255))
    return canvas


def ensure_writer(writer, record_cfg, frame_shape):
    if writer is not None and writer.isOpened():
        return writer
    path = record_cfg.get("path", "out_compare.mp4")
    fps = float(record_cfg.get("fps", 30))
    h, w = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"⚠️ 无法打开录像输出: {path}")
        return None
    return writer


def build_processor(pp_cfg, det_cfg, track_cfg, geom_cfg) -> ProcessorBundle:
    pipeline = PreprocessPipeline(pp_cfg)

    detector = None
    if det_cfg.get("enabled", False):
        try:
            detector = build_detector(det_cfg)
        except Exception as exc:
            print(f"⚠️ 检测器初始化失败: {exc}")
            detector = None

    tracker = None
    if track_cfg.get("enabled", False):
        try:
            tracker = build_tracker(track_cfg)
        except Exception as exc:
            print(f"⚠️ 跟踪器初始化失败: {exc}")
            tracker = None

    projector = None
    if geom_cfg.get("enabled", False):
        try:
            projector = build_projector(geom_cfg)
        except Exception as exc:
            print(f"⚠️ 几何投影初始化失败: {exc}")
            projector = None

    return ProcessorBundle(pipeline, detector, tracker, projector)


def parse_args():
    parser = argparse.ArgumentParser(description="Road vision preview")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径，默认使用 configs/default.yaml")
    return parser.parse_args()


def run_preview(cfg: dict, cfg_path: Path) -> str:
    cam_cfg = cfg.get("camera", {}) or {}
    preview_cfg = cfg.get("preview", {}) or {}
    compare_cfg = preview_cfg.get("compare", {}) or {}
    record_cfg = preview_cfg.get("record", {}) or {}
    runtime_cfg = cfg.get("runtime", {}) or {}
    async_cfg = runtime_cfg.get("async", {}) or {}
    hot_cfg = runtime_cfg.get("hot_reload", {}) or {}
    pp_cfg = cfg.get("preprocess", {}) or {}
    det_cfg = cfg.get("detect", {}) or {}
    track_cfg = cfg.get("tracking", {}) or {}
    geom_cfg = cfg.get("geometry", {}) or {}
    vis_cfg = cfg.get("vis", {}) or {}
    draw_cfg = vis_cfg.get("draw", {}) or {}

    vs = VideoSource(cam_cfg)
    fpsm = FPSMeter(alpha=0.1)
    processor = build_processor(pp_cfg, det_cfg, track_cfg, geom_cfg)

    watcher = None
    if hot_cfg.get("enabled", False):
        watcher = ConfigWatcher(cfg_path, float(hot_cfg.get("interval_sec", 2.0)))

    want_record = bool(record_cfg.get("enable", False))
    want_compare = bool(compare_cfg.get("enable", True))
    layout = compare_cfg.get("layout", "h")
    divider_px = int(compare_cfg.get("divider_px", 4))
    writer = None

    use_async = bool(async_cfg.get("enabled", False))
    max_latency = float(async_cfg.get("max_latency_ms", 180.0)) / 1000.0
    result_timeout = float(async_cfg.get("result_timeout_ms", 250.0)) / 1000.0
    queue_size = int(async_cfg.get("queue_size", 4))
    drop_policy = (async_cfg.get("drop_policy") or "oldest").lower()
    runner: Optional[AsyncPipeline] = None
    if use_async:
        runner = AsyncPipeline(
            vs,
            processor.process,
            queue_size=queue_size,
            drop_policy=drop_policy,
            max_latency=max_latency if max_latency > 0 else None,
            result_timeout=result_timeout,
        )
        runner.start()

    reload_requested = False

    while True:
        if watcher and watcher.poll():
            print("🔁 检测到配置文件更新，准备重载...")
            reload_requested = True
            break

        if use_async and runner is not None:
            result = runner.get_result(timeout=result_timeout)
            if result is None:
                if runner.stopped:
                    break
                continue
        else:
            frame = vs.read()
            if not frame.ok:
                print("⚠️ 读取失败/视频结束")
                break
            result = processor.process(frame)
            if result is None:
                continue

        if result.proc is None or result.raw is None:
            continue

        if draw_cfg.get("det", True) and result.detections:
            draw_detections(
                result.proc,
                result.detections,
                thickness=int(draw_cfg.get("thickness", 2)),
                font_scale=float(draw_cfg.get("font_scale", 0.6)),
            )

        fps = fpsm.tick(result.ts)

        if want_compare:
            canvas = make_canvas(
                result.raw,
                result.proc,
                layout=layout,
                divider_px=divider_px,
                label_raw=compare_cfg.get("label_raw", "RAW"),
                label_proc=compare_cfg.get("label_proc", "PROC"),
                fps=fps,
                show_fps=bool(preview_cfg.get("show_fps", True)),
            )
            if want_record:
                writer = ensure_writer(writer, record_cfg, canvas.shape)
                if writer:
                    writer.write(canvas)
            cv2.imshow("Compare Preview", canvas)
        else:
            frame_disp = result.proc.copy()
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
            if want_record:
                writer = ensure_writer(writer, record_cfg, frame_disp.shape)
                if writer:
                    writer.write(frame_disp)
            cv2.imshow("Preview", frame_disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    if writer:
        writer.release()
    if runner:
        runner.close()
        dropped = runner.dropped_frames
        skipped = runner.skipped_frames
        if dropped or skipped:
            print(f"ℹ️ 异步采集丢弃帧 {dropped}，跳过旧帧 {skipped}")
    processor.close()
    vs.release()

    return "reload" if reload_requested else "exit"


def main():
    args = parse_args()
    cfg_path = resolve_config_path(args.config)
    while True:
        cfg = load_config(str(cfg_path))
        action = run_preview(cfg, cfg_path)
        if action == "reload":
            continue
        break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
