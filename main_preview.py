import cv2
import numpy as np
from src.config import load_config
from src.io_video.capture import VideoSource
from src.io_video.fps_meter import FPSMeter
from src.preprocess import PreprocessPipeline
from src.detect import build_detector
from src.vis import draw_detections

def make_canvas(raw_bgr, proc_bgr, layout="h", divider_px=4,
                label_raw="RAW", label_proc="PROC", fps=None, show_fps=True):
    h, w = raw_bgr.shape[:2]
    divider_px = max(0, int(divider_px))

    def put_label(img, org, text, color=(50, 220, 50)):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    if layout.lower() == "v":
        divider = np.full((divider_px, w, 3), (40,40,40), dtype=np.uint8) if divider_px else None
        canvas = np.vstack([raw_bgr, divider, proc_bgr]) if divider is not None else np.vstack([raw_bgr, proc_bgr])
        put_label(canvas, (10, 30), label_raw)
        put_label(canvas, (10, h + divider_px + 30), label_proc, color=(0,200,255))
    else:
        divider = np.full((h, divider_px, 3), (40,40,40), dtype=np.uint8) if divider_px else None
        canvas = np.hstack([raw_bgr, divider, proc_bgr]) if divider is not None else np.hstack([raw_bgr, proc_bgr])
        put_label(canvas, (10, 30), label_raw)
        put_label(canvas, (w + divider_px + 10, 30), label_proc, color=(0,200,255))

    if show_fps and fps is not None:
        put_label(canvas, (10, max(60, h - 10)), f"FPS: {fps:.1f}", color=(0,255,255))
    return canvas

def main():
    cfg = load_config()

    cam_cfg = cfg.get("camera", {})
    preview_cfg = cfg.get("preview", {})
    compare_cfg = preview_cfg.get("compare", {}) or {}
    record_cfg = preview_cfg.get("record", {}) or {}
    pp_cfg = cfg.get("preprocess", {}) or {}
    det_cfg = cfg.get("detect", {}) or {}
    vis_cfg = cfg.get("vis", {}) or {}
    draw_cfg = vis_cfg.get("draw", {}) or {}

    vs = VideoSource(
        source=cam_cfg.get("source", 0),
        width=cam_cfg.get("width", 1280),
        height=cam_cfg.get("height", 720),
        fps_request=cam_cfg.get("fps_request", 30),
        backend=cam_cfg.get("backend", "auto"),
    )
    fpsm = FPSMeter(alpha=0.1)
    pipeline = PreprocessPipeline(pp_cfg)

    detector = None
    if det_cfg.get("enabled", False):
        detector = build_detector(det_cfg)

    # 录像器
    writer = None
    want_record = bool(record_cfg.get("enable", False))
    want_compare = bool(compare_cfg.get("enable", True))
    layout = compare_cfg.get("layout", "h")
    divider_px = int(compare_cfg.get("divider_px", 4))

    # 主循环
    while True:
        fr = vs.read()
        if not fr.ok:
            print("⚠️ 读取失败/视频结束")
            break
        raw = fr.image
        proc = pipeline(raw, ts=fr.ts)

        # 检测（在处理后帧上）
        dets = []
        if detector is not None:
            dets = detector.infer(proc)

        # 叠加绘制
        if draw_cfg.get("det", True) and dets:
            draw_detections(proc, dets,
                            thickness=int(draw_cfg.get("thickness", 2)),
                            font_scale=float(draw_cfg.get("font_scale", 0.6)))

        fps = fpsm.tick(fr.ts)

        if want_compare:
            canvas = make_canvas(
                raw, proc,
                layout=layout,
                divider_px=divider_px,
                label_raw=compare_cfg.get("label_raw", "RAW"),
                label_proc=compare_cfg.get("label_proc", "PROC"),
                fps=fps,
                show_fps=bool(preview_cfg.get("show_fps", True)),
            )
            if writer: writer.write(canvas)
            cv2.imshow("Compare Preview", canvas)
        else:
            frame_disp = proc.copy()
            if preview_cfg.get("show_fps", True):
                cv2.putText(frame_disp, f"FPS:{fps:.1f}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            if writer: writer.write(frame_disp)
            cv2.imshow("Preview", frame_disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    if writer: writer.release()
    if detector: detector.close()
    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
