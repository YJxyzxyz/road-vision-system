from typing import List, Dict, Any
import numpy as np
import cv2

from .registry import get_op_class

class PreprocessPipeline:
    """
    可插拔预处理流水线：
    - 根据 config 构建算子链
    - __call__(image, ts=None) -> image
    """
    def __init__(self, config: Dict[str, Any]):
        self.enabled = bool(config.get("enabled", True))
        self.chain_cfg = config.get("chain", []) or []
        self.auto_gate_cfg = config.get("auto_gate", {}) or {}
        self.ops = []
        for node in self.chain_cfg:
            name = node.get("name")
            params = node.get("params", {})
            cls = get_op_class(name)
            self.ops.append(cls(**params))

    def _low_contrast(self, image) -> bool:
        # 非严格：用灰度直方图跨度估个对比度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mn, mx = int(gray.min()), int(gray.max())
        span = mx - mn
        thresh = float(self.auto_gate_cfg.get("contrast_thresh", 20.0))
        return span < thresh

    def __call__(self, image: np.ndarray, ts: float = None) -> np.ndarray:
        if not self.enabled or not self.ops:
            return image

        # 自动门限（可选）
        if self.auto_gate_cfg.get("enable_low_contrast_gate", False):
            if not self._low_contrast(image):
                # 认为画面对比度足够 -> 跳过整个链
                return image

        out = image
        for op in self.ops:
            out = op(out)
        return out
