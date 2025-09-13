import numpy as np
from typing import List, Dict, Any
from .base import Detector
from .types import Detection

class YOLOUltralytics(Detector):
    def __init__(self, cfg: Dict[str, Any]):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise ImportError(
                "未安装 ultralytics，请先 pip install ultralytics"
            ) from e

        device = cfg.get("device", "auto")
        self.model = YOLO(cfg.get("model", "yolov8n.pt"))
        self.model.fuse()  # 更快一些
        self.device = None if device == "auto" else device
        self.conf = float(cfg.get("conf_thres", 0.25))
        self.iou  = float(cfg.get("iou_thres", 0.7))
        self.max_det = int(cfg.get("max_det", 100))
        self.keep = set(int(x) for x in cfg.get("classes_keep", []))
        # 类别名
        self.names = self.model.model.names if hasattr(self.model, "model") else self.model.names

    def infer(self, bgr: np.ndarray) -> List[Detection]:
        # Ultralytics 支持直接喂 np.ndarray（BGR）
        res = self.model.predict(
            source=bgr,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            verbose=False
        )
        out: List[Detection] = []
        if not res:
            return out
        r0 = res[0]
        boxes = r0.boxes
        if boxes is None or boxes.shape[0] == 0:
            return out

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls  = boxes.cls.cpu().numpy().astype(int)

        for (x1,y1,x2,y2), c, k in zip(xyxy, conf, cls):
            if self.keep and k not in self.keep:
                continue
            name = str(self.names[k]) if self.names is not None and k in self.names else str(k)
            out.append(Detection(float(x1), float(y1), float(x2), float(y2), float(c), k, name))
        return out

    def close(self):
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
