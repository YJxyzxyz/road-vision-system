from dataclasses import dataclass
from typing import Optional

@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int
    cls_name: str
    track_id: Optional[int] = None  # 先占位，M4 会用
    distance_m: Optional[float] = None
    speed_kmh: Optional[float] = None
