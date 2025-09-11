from typing import Dict, Type
from .base import PreprocessOp
from .ops import CLAHEDehaze, MedianDerain

# 新增：尝试导入 CUDA 版（若失败不报错，仍可用 CPU 版）
try:
    from .ops_cuda import CUDACLAHEDehaze, CUDAMedianDerain
    _HAS_CUDA_OPS = True
except Exception:
    CUDACLAHEDehaze = None
    CUDAMedianDerain = None
    _HAS_CUDA_OPS = False

REGISTRY: Dict[str, Type[PreprocessOp]] = {
    "CLAHEDehaze": CLAHEDehaze,
    "MedianDerain": MedianDerain,
}

if _HAS_CUDA_OPS:
    REGISTRY.update({
        "CUDACLAHEDehaze": CUDACLAHEDehaze,
        "CUDAMedianDerain": CUDAMedianDerain,
    })

def get_op_class(name: str) -> Type[PreprocessOp]:
    if name not in REGISTRY:
        raise KeyError(f"Preprocess op '{name}' not found. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name]
