import cv2
from ..base import PreprocessOp

class MedianDerain(PreprocessOp):
    """
    轻量“去雨”（占位）：中值滤波，去除细小雨丝与噪点。
    params:
      - ksize: 3/5/7（奇数）
    """
    def __call__(self, image):
        k = int(self.params.get("ksize", 3))
        if k % 2 == 0: k += 1
        k = max(3, min(k, 9))
        return cv2.medianBlur(image, k)
