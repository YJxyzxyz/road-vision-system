import cv2
import numpy as np
from ..base import PreprocessOp
from ..ops.clahe_dehaze import CLAHEDehaze  # 作为软回退

def _cuda_available() -> bool:
    return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0

class CUDACLAHEDehaze(PreprocessOp):
    """
    CUDA 加速的 CLAHE “去雾/增强”
    - 尽量全流程在 GPU；少量分离/合并在 CPU（简单稳妥）
    - 若 CUDA 不可用，则调用 CPU 版 CLAHEDehaze
    params:
      - space: "YCrCb" | "LAB"
      - clip_limit: float
      - tile_grid: int
    """

    def __init__(self, **params):
        super().__init__(**params)
        self._fallback = CLAHEDehaze(**params)

        self._ok = _cuda_available()
        if self._ok:
            # 预创建 CLAHE 对象
            clip = float(self.params.get("clip_limit", 2.0))
            grid = int(self.params.get("tile_grid", 8))
            grid = max(2, grid)
            try:
                self._clahe = cv2.cuda.createCLAHE(clip, (grid, grid))
            except Exception:
                # 某些 OpenCV 构建可能缺少该算子，软回退
                self._ok = False
                self._clahe = None

    def __call__(self, image):
        if not self._ok:
            return self._fallback(image)

        space = self.params.get("space", "YCrCb").upper()

        # upload 到 GPU
        g_bgr = cv2.cuda_GpuMat()
        g_bgr.upload(image)

        # 颜色空间转换（CUDA）
        if space == "LAB":
            g_lab = cv2.cuda.cvtColor(g_bgr, cv2.COLOR_BGR2LAB)
            # cv2.cuda 不直提供 split，这里下载到 CPU 处理 L 通道，再回传
            lab = g_lab.download()
            l, a, b = cv2.split(lab)
            g_l = cv2.cuda_GpuMat()
            g_l.upload(l)
            g_l2 = self._clahe.apply(g_l)
            l2 = g_l2.download()
            out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
            return out
        else:
            # 默认 YCrCb
            g_ycc = cv2.cuda.cvtColor(g_bgr, cv2.COLOR_BGR2YCrCb)
            ycc = g_ycc.download()
            y, cr, cb = cv2.split(ycc)
            g_y = cv2.cuda_GpuMat(); g_y.upload(y)
            g_y2 = self._clahe.apply(g_y)
            y2 = g_y2.download()
            out = cv2.cvtColor(cv2.merge([y2, cr, cb]), cv2.COLOR_YCrCb2BGR)
            return out
