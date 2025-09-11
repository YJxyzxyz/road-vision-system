from abc import ABC, abstractmethod
from typing import Any, Dict

class PreprocessOp(ABC):
    """
    预处理算子基类：所有具体操作都继承它。
    约定：__call__(image) -> image
    - image: np.ndarray, BGR (H,W,3), uint8
    - 返回：处理后的 image（可原位或拷贝）
    """
    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def __call__(self, image):
        pass
