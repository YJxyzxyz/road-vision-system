import cv2
from src.config import load_config
from src.io_video.capture import VideoSource
from src.io_video.fps_meter import FPSMeter

def main():
    cfg = load_config()
    cam = cfg["camera"]
    preview = cfg["preview"]

    vs = VideoSource(
        source=cam["source"],
        width=cam["width"],
        height=cam["height"],
        fps_request=cam["fps_request"],
        backend=cam.get("backend","auto"),
    )
    fpsm = FPSMeter(alpha=0.1)

    while True:
        frame = vs.read()
        if not frame.ok:
            print("⚠️ 读取失败/视频结束")
            break

        fps = fpsm.tick(frame.ts)

        img = frame.image
        if preview["show_fps"]:
            cv2.putText(img, f"FPS:{fps:.1f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.imshow("Module 1 Preview", img)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
