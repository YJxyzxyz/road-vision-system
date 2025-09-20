from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .base import Detector
from .types import Detection


class YOLOTensorRT(Detector):
    """基于 Ultralytics TensorRT engine 的检测封装."""

    def __init__(self, cfg: Dict[str, Any]):
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - 运行期提示
            raise ImportError(
                "未安装 ultralytics，请先 pip install ultralytics"
            ) from exc

        engine_path = cfg.get("engine") or cfg.get("model")
        if not engine_path:
            raise ValueError("TensorRT 后端需要提供 engine 或 model 路径")

        self.device = cfg.get("device", "cuda:0")
        self.imgsz = cfg.get("imgsz", 640)
        self.half = bool(cfg.get("half", True))
        self.max_det = int(cfg.get("max_det", 100))
        self.conf = float(cfg.get("conf_thres", 0.25))
        self.iou = float(cfg.get("iou_thres", 0.7))
        self.keep = set(int(x) for x in cfg.get("classes_keep", []))
        self.model = YOLO(engine_path)
        self.names = self.model.model.names if hasattr(self.model, "model") else self.model.names

        warmup = int(cfg.get("warmup_runs", 1))
        if warmup > 0:
            dummy = np.zeros((int(self.imgsz), int(self.imgsz), 3), dtype=np.uint8)
            for _ in range(warmup):
                self.model.predict(
                    source=dummy,
                    device=self.device,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    max_det=self.max_det,
                    half=self.half,
                    verbose=False,
                )

    def infer(self, bgr: np.ndarray) -> List[Detection]:
        res = self.model.predict(
            source=bgr,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            half=self.half,
            verbose=False,
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
        cls = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            if self.keep and k not in self.keep:
                continue
            name = str(self.names[k]) if self.names is not None and k in self.names else str(k)
            out.append(Detection(float(x1), float(y1), float(x2), float(y2), float(c), k, name))
        return out

    def close(self):
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - 软清理
            pass
