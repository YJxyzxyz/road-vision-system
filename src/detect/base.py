from abc import ABC, abstractmethod
from typing import List
import numpy as np
from .types import Detection

class Detector(ABC):
    @abstractmethod
    def infer(self, bgr: np.ndarray) -> List[Detection]:
        """输入 BGR uint8(H,W,3)，返回检测结果列表"""
        raise NotImplementedError()

    def close(self):  # 预留资源释放
        pass
