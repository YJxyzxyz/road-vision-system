"""绘制检测/跟踪结果的辅助函数。"""
from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

from src.detect.types import Detection

_COLOR_TABLE: Tuple[Tuple[int, int, int], ...] = (
    (255, 128, 64),
    (0, 255, 255),
    (80, 175, 76),
    (255, 0, 255),
    (0, 128, 255),
    (255, 64, 64),
    (64, 255, 64),
    (128, 128, 255),
    (255, 200, 0),
    (0, 255, 128),
)


def draw_detections(
    image: np.ndarray,
    detections: Iterable[Detection],
    thickness: int = 2,
    font_scale: float = 0.6,
) -> None:
    """在图像上绘制检测框、ID、测距与速度。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(thickness))
    for det in detections:
        if det is None:
            continue
        color = _COLOR_TABLE[det.cls_id % len(_COLOR_TABLE)] if len(_COLOR_TABLE) else (0, 255, 0)
        x1, y1, x2, y2 = map(int, [det.x1, det.y1, det.x2, det.y2])
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        cls_name = det.cls_name or str(det.cls_id)
        label_main = f"{cls_name} {det.conf:.2f}" if det.conf is not None else cls_name
        if det.track_id is not None:
            label_main = f"ID {det.track_id} | {label_main}"
        _draw_label_top(image, label_main, (x1, y1), color, font_scale, thickness, font)

        metrics = []
        if det.distance_m is not None:
            metrics.append(f"{det.distance_m:.1f} m")
        if det.speed_kmh is not None:
            metrics.append(f"{det.speed_kmh:.1f} km/h")
        if metrics:
            bottom_text = " / ".join(metrics)
            _draw_label_bottom(image, bottom_text, (x1, y2 + 4), color, font_scale, thickness, font)


def _draw_label_top(
    img: np.ndarray,
    text: str,
    topleft: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float,
    thickness: int,
    font: int,
) -> None:
    if not text:
        return
    x, y = topleft
    x = int(max(0, x))
    y = int(max(0, y))
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 2
    box_top = max(0, y - th - baseline - pad * 2)
    box_bottom = y
    cv2.rectangle(img, (x, box_top), (x + tw + pad * 2, box_bottom), color, -1)
    text_org = (x + pad, max(box_top + th, pad + th))
    cv2.putText(img, text, text_org, font, font_scale, (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)


def _draw_label_bottom(
    img: np.ndarray,
    text: str,
    bottomleft: Tuple[int, int],
    color: Tuple[int, int, int],
    font_scale: float,
    thickness: int,
    font: int,
) -> None:
    if not text:
        return
    x, y = bottomleft
    x = int(max(0, x))
    y = int(max(0, y))
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 2
    box_top = min(max(0, y), img.shape[0] - th - baseline - pad * 2)
    box_bottom = min(img.shape[0], box_top + th + baseline + pad * 2)
    cv2.rectangle(img, (x, box_top), (x + tw + pad * 2, box_bottom), color, -1)
    text_org = (x + pad, min(img.shape[0] - baseline - 1, box_top + th + baseline))
    cv2.putText(img, text, text_org, font, font_scale, (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)
