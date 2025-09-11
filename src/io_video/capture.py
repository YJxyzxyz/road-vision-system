import cv2, time

class Frame:
    __slots__ = ("ok","image","ts")
    def __init__(self, ok, image, ts):
        self.ok = ok
        self.image = image
        self.ts = ts  # 捕获时间戳（秒，float）

class VideoSource:
    def __init__(self, source=0, width=1280, height=720, fps_request=30, backend="auto"):
        # 先用 OpenCV 标准接口；后续可在 backend=="gstreamer" 替换
        self.cap = cv2.VideoCapture(source)
        # 尽力设置（不保证成功）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,         fps_request)

    def read(self) -> Frame:
        ok, img = self.cap.read()
        ts = time.time()
        return Frame(ok, img, ts)

    def release(self):
        if self.cap: self.cap.release()
