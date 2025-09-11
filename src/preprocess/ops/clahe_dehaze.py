import cv2
import numpy as np
from ..base import PreprocessOp

class CLAHEDehaze(PreprocessOp):
    """
    轻量“去雾/增强”：对亮度通道做 CLAHE。
    params:
      - space: "YCrCb" | "LAB"
      - clip_limit: float
      - tile_grid: int  (GxG)
    """
    def __call__(self, image):
        space = self.params.get("space", "YCrCb").upper()
        clip_limit = float(self.params.get("clip_limit", 2.0))
        grid = int(self.params.get("tile_grid", 8))
        grid = max(2, grid)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))

        if space == "LAB":
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = clahe.apply(l)
            out = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
        else:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y2 = clahe.apply(y)
            out = cv2.cvtColor(cv2.merge([y2, cr, cb]), cv2.COLOR_YCrCb2BGR)

        return out
