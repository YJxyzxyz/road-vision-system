from typing import Dict, Any
from .base import Detector
from .yolo_ultralytics import YOLOUltralytics

def build_detector(cfg: Dict[str, Any]) -> Detector:
    backend = (cfg.get("backend") or "ultralytics").lower()
    if backend == "ultralytics":
        return YOLOUltralytics(cfg)
    raise ValueError(f"未知检测后端: {backend}")
