import cv2
from ..base import PreprocessOp
from ..ops.median_derain import MedianDerain  # 作为软回退

def _cuda_available() -> bool:
    return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0

class CUDAMedianDerain(PreprocessOp):
    """
    CUDA 中值滤波：
    - OpenCV CUDA 的 median 支持 CV_8UC1/8UC4，所以把 BGR 转为 BGRA 处理
    - 若 CUDA 不可用 -> 回退 CPU 版 MedianDerain
    params:
      - ksize: 3/5/7/9（奇数）
    """
    def __init__(self, **params):
        super().__init__(**params)
        self._fallback = MedianDerain(**params)
        self._ok = _cuda_available()
        if self._ok:
            try:
                # 尝试创建滤波器（需要指定类型）
                k = int(self.params.get("ksize", 3))
                if k % 2 == 0: k += 1
                k = max(3, min(k, 9))
                # 对 8UC4 生效：后续传入 BGRA
                self._mf = cv2.cuda.createMedianFilter(cv2.CV_8UC4, k)
            except Exception:
                self._ok = False
                self._mf = None

    def __call__(self, image):
        if not self._ok:
            return self._fallback(image)

        # 上传 + 转 BGRA（CUDA）
        g_bgr = cv2.cuda_GpuMat(); g_bgr.upload(image)
        g_bgra = cv2.cuda.cvtColor(g_bgr, cv2.COLOR_BGR2BGRA)

        # 中值滤波（CUDA）
        g_out = self._mf.apply(g_bgra)

        # 回到 BGR
        g_bgr2 = cv2.cuda.cvtColor(g_out, cv2.COLOR_BGRA2BGR)
        return g_bgr2.download()
