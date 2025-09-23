from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .base import Detector
from .types import Detection


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
    if boxes.size == 0:
        return []
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_i + area_rest - inter
        iou = np.where(union > 0.0, inter / union, 0.0)
        idx = np.where(iou <= threshold)[0]
        order = rest[idx]
    return keep


class YOLOTensorRT(Detector):
    """TensorRT runtime for YOLO models exported from Ultralytics."""

    def __init__(self, cfg: Dict[str, object]):
        try:
            import tensorrt as trt
        except Exception as exc:  # pragma: no cover - dependency missing
            raise ImportError(
                "未安装 TensorRT Python 包，请按照 README 中的 Jetson 指南安装"
            ) from exc

        try:
            import pycuda.driver as cuda  # type: ignore
            import pycuda.autoinit  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover - dependency missing
            raise ImportError("未安装 pycuda，请 pip install pycuda") from exc

        self.trt = trt
        self.cuda = cuda
        self.stream = cuda.Stream()
        self.logger = trt.Logger(trt.Logger.WARNING)

        engine_path = cfg.get("engine")
        if not engine_path:
            raise ValueError("detect.engine 未配置 TensorRT engine 路径")
        engine_file = Path(str(engine_path))
        if not engine_file.exists():
            raise FileNotFoundError(f"TensorRT engine 文件不存在: {engine_file}")

        with open(engine_file, "rb") as f:
            engine_bytes = f.read()

        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError("TensorRT engine 反序列化失败")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("TensorRT 执行上下文创建失败")

        self.engine = engine
        self.context = context
        self.bindings: List[int] = [0] * self.engine.num_bindings
        self._input_host: Optional[np.ndarray] = None
        self._input_device = None
        self._input_shape: Optional[Sequence[int]] = None
        self._output_specs: List[tuple[int, np.ndarray, object, Sequence[int]]] = []
        self.input_binding: Optional[int] = None
        self.input_hw: Optional[tuple[int, int]] = None

        self._setup_bindings(cfg)

        self.conf = float(cfg.get("conf_thres", 0.25))
        self.iou = float(cfg.get("iou_thres", 0.7))
        self.max_det = int(cfg.get("max_det", 100))
        keep = cfg.get("classes_keep", []) or []
        self.keep = {int(x) for x in keep}
        names_cfg = cfg.get("names")
        if isinstance(names_cfg, dict):
            self.names = {int(k): str(v) for k, v in names_cfg.items()}
        elif isinstance(names_cfg, list):
            self.names = {i: str(v) for i, v in enumerate(names_cfg)}
        else:
            self.names = {}
        self.num_classes = int(cfg.get("num_classes", len(self.names) or 80))
        if not self.names:
            self.names = {i: str(i) for i in range(self.num_classes)}
        self._has_obj_conf: Optional[bool] = (
            bool(cfg["has_obj_conf"]) if "has_obj_conf" in cfg else None
        )

    def _setup_bindings(self, cfg: Dict[str, object]) -> None:
        trt = self.trt
        input_hw_cfg = cfg.get("input_hw")
        if input_hw_cfg is not None:
            if not isinstance(input_hw_cfg, (list, tuple)) or len(input_hw_cfg) != 2:
                raise ValueError("input_hw 需要形如 [height, width]")
            input_hw = (int(input_hw_cfg[0]), int(input_hw_cfg[1]))
        else:
            input_hw = None

        # Configure input binding first
        for idx in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(idx):
                continue
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(idx)))
            shape = list(self.engine.get_binding_shape(idx))
            if any(dim == -1 for dim in shape):
                if input_hw is None:
                    raise ValueError(
                        "TensorRT engine 为动态 shape，请在配置中提供 input_hw: [height, width]"
                    )
                shape = [1, 3, int(input_hw[0]), int(input_hw[1])]
                self.context.set_binding_shape(idx, tuple(shape))
            self._input_shape = tuple(shape)
            size = int(math.prod(shape))
            host_mem = self.cuda.pagelocked_empty(size, dtype=dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)
            self._input_host = host_mem
            self._input_device = device_mem
            self.bindings[idx] = int(device_mem)
            self.input_binding = idx
            self.input_hw = (shape[2], shape[3])
            break

        if self.input_binding is None:
            raise RuntimeError("未找到 TensorRT 输入 binding")

        # Output bindings
        for idx in range(self.engine.num_bindings):
            if self.engine.binding_is_input(idx):
                continue
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(idx)))
            shape = tuple(self.context.get_binding_shape(idx))
            if any(dim == -1 for dim in shape):
                raise RuntimeError("TensorRT 输出 shape 未固定，请重新导出 engine")
            size = int(math.prod(shape))
            host_mem = self.cuda.pagelocked_empty(size, dtype=dtype)
            device_mem = self.cuda.mem_alloc(host_mem.nbytes)
            self.bindings[idx] = int(device_mem)
            self._output_specs.append((idx, host_mem, device_mem, shape))

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, Dict[str, float]]:
        if self.input_hw is None:
            raise RuntimeError("TensorRT 输入尺寸未初始化")
        input_h, input_w = self.input_hw
        h, w = image.shape[:2]
        scale = min(input_w / w, input_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        top = (input_h - new_h) // 2
        left = (input_w - new_w) // 2
        canvas[top : top + new_h, left : left + new_w, :] = resized
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = canvas.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, 0)
        return np.ascontiguousarray(tensor, dtype=np.float32), {
            "scale": scale,
            "pad_x": float(left),
            "pad_y": float(top),
            "orig_h": float(h),
            "orig_w": float(w),
        }

    def _flatten_predictions(self, pred: np.ndarray) -> np.ndarray:
        if pred.ndim == 3:
            if pred.shape[1] > pred.shape[2]:
                pred = np.transpose(pred, (0, 2, 1))
            pred = pred[0]
        elif pred.ndim == 2:
            pass
        else:
            raise ValueError("不支持的 TensorRT 输出维度")
        return np.ascontiguousarray(pred)

    def _postprocess(self, pred: np.ndarray, meta: Dict[str, float]) -> List[Detection]:
        pred = self._flatten_predictions(pred)
        if pred.size == 0:
            return []
        num_attrs = pred.shape[1]
        if num_attrs < 6:
            return []

        has_obj = self._has_obj_conf
        if has_obj is None:
            if num_attrs - 5 == self.num_classes:
                has_obj = True
            elif num_attrs - 4 == self.num_classes:
                has_obj = False
            elif num_attrs - 5 > 0:
                has_obj = True
            else:
                has_obj = False
            self._has_obj_conf = has_obj

        if has_obj:
            class_scores = pred[:, 5:]
            objectness = pred[:, 4:5]
            scores_matrix = class_scores * objectness
        else:
            class_scores = pred[:, 4:]
            scores_matrix = class_scores

        num_classes = scores_matrix.shape[1]
        if self.keep:
            valid_classes = [c for c in sorted(self.keep) if 0 <= c < num_classes]
        else:
            valid_classes = list(range(num_classes))
        if not valid_classes:
            return []

        selected_scores = scores_matrix[:, valid_classes]
        cls_indices = np.argmax(selected_scores, axis=1)
        confidences = selected_scores[np.arange(selected_scores.shape[0]), cls_indices]
        mask = confidences >= self.conf
        if not np.any(mask):
            return []

        selected_scores = confidences[mask]
        cls_indices = cls_indices[mask]
        pred = pred[mask]
        boxes = np.zeros((pred.shape[0], 4), dtype=np.float32)
        boxes[:, 0] = pred[:, 0] - 0.5 * pred[:, 2]
        boxes[:, 1] = pred[:, 1] - 0.5 * pred[:, 3]
        boxes[:, 2] = pred[:, 0] + 0.5 * pred[:, 2]
        boxes[:, 3] = pred[:, 1] + 0.5 * pred[:, 3]

        scale = meta["scale"]
        pad_x = meta["pad_x"]
        pad_y = meta["pad_y"]
        orig_w = meta["orig_w"]
        orig_h = meta["orig_h"]
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / max(1e-6, scale)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / max(1e-6, scale)
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h - 1)

        detections: List[Detection] = []
        for cls in set(int(valid_classes[idx]) for idx in cls_indices):
            mask_cls = [i for i, c in enumerate(cls_indices) if int(valid_classes[c]) == cls]
            if not mask_cls:
                continue
            cls_boxes = boxes[mask_cls]
            cls_scores = selected_scores[mask_cls]
            keep_idx = _nms(cls_boxes, cls_scores, self.iou)
            for idx in keep_idx:
                global_idx = mask_cls[idx]
                cls_id = int(valid_classes[cls_indices[global_idx]])
                name = self.names.get(cls_id, str(cls_id))
                box = boxes[global_idx]
                detections.append(
                    Detection(
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                        float(selected_scores[global_idx]),
                        cls_id,
                        name,
                    )
                )

        if len(detections) > self.max_det:
            detections.sort(key=lambda d: d.conf, reverse=True)
            detections = detections[: self.max_det]
        return detections

    def infer(self, bgr: np.ndarray) -> List[Detection]:
        if self._input_host is None or self._input_device is None:
            raise RuntimeError("TensorRT 输入缓冲区未初始化")
        tensor, meta = self._preprocess(bgr)
        np.copyto(self._input_host, tensor.reshape(-1))
        self.cuda.memcpy_htod_async(self._input_device, self._input_host, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        outputs: List[np.ndarray] = []
        for _, host_mem, device_mem, shape in self._output_specs:
            self.cuda.memcpy_dtoh_async(host_mem, device_mem, self.stream)
            outputs.append(host_mem.reshape(shape))
        self.stream.synchronize()
        if not outputs:
            return []
        return self._postprocess(outputs[0], meta)

    def close(self) -> None:
        try:
            for _, _, device_mem, _ in self._output_specs:
                device_mem.free()
        except Exception:
            pass
        if self._input_device is not None:
            try:
                self._input_device.free()
            except Exception:
                pass
        self._output_specs.clear()
        self._input_host = None
        self._input_device = None
        self.context = None
        self.engine = None


__all__ = ["YOLOTensorRT"]
