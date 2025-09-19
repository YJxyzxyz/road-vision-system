"""跟踪器基类。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

from src.detect.types import Detection
from src.geometry import GroundProjector


class Tracker(ABC):
    """跟踪器接口。"""

    @abstractmethod
    def update(
        self,
        detections: Iterable[Detection],
        timestamp: float,
        projector: Optional[GroundProjector] = None,
    ) -> List[Detection]:
        """更新跟踪器并返回带 ID/测距/速度信息的检测结果。"""

    def close(self) -> None:
        """预留释放资源。"""
