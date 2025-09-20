"""地面投影模块：将像素坐标映射到道路平面。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

Point2D = Tuple[float, float]


class GroundProjector(ABC):
    """抽象基类：提供图像坐标到地面坐标的映射。"""

    def __init__(self,
                 origin: Sequence[float] | None = None,
                 max_distance: float | None = None) -> None:
        if origin is None:
            origin = (0.0, 0.0)
        if len(origin) != 2:
            raise ValueError("origin 必须是长度为 2 的序列")
        self.origin = np.asarray(origin, dtype=np.float32)
        self.max_distance = float(max_distance) if max_distance is not None else None

    @abstractmethod
    def project_point(self, x: float, y: float) -> Optional[Point2D]:
        """将图像坐标 (x, y) 映射到地面坐标系。"""

    def project_bbox(self, bbox: Sequence[float]) -> Optional[Point2D]:
        """将检测框底边中心映射到地面。"""
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (float(x1) + float(x2))
        cy = float(y2)
        return self.project_point(cx, cy)

    def distance(self, point: Optional[Sequence[float]]) -> Optional[float]:
        """计算点到相机投影原点的距离。"""
        if point is None:
            return None
        vec = np.asarray(point, dtype=np.float32) - self.origin
        dist = float(np.linalg.norm(vec))
        if not np.isfinite(dist):
            return None
        if self.max_distance is not None:
            dist = min(dist, self.max_distance)
        return dist

    def distance_for_bbox(self, bbox: Sequence[float]) -> Optional[float]:
        """辅助：直接对检测框计算距离。"""
        return self.distance(self.project_bbox(bbox))


class HomographyProjector(GroundProjector):
    """基于平面单应性的地面投影。"""

    def __init__(self, cfg: dict) -> None:
        origin = cfg.get("origin", (0.0, 0.0))
        max_dist = cfg.get("max_distance")
        super().__init__(origin=origin, max_distance=max_dist)

        img_pts = np.asarray(cfg.get("image_points", []), dtype=np.float32)
        world_pts = np.asarray(cfg.get("world_points", []), dtype=np.float32)
        if img_pts.ndim != 2 or img_pts.shape[0] < 4 or img_pts.shape[1] != 2:
            raise ValueError("homography 至少需要 4 个像素坐标点 (x, y)")
        if world_pts.shape != img_pts.shape:
            raise ValueError("image_points 与 world_points 形状需一致")

        H, status = cv2.findHomography(img_pts, world_pts)
        if H is None or status is None:
            raise ValueError("cv2.findHomography 计算失败")
        self._H = H.astype(np.float64)

    def project_point(self, x: float, y: float) -> Optional[Point2D]:
        pt = np.array([float(x), float(y), 1.0], dtype=np.float64)
        mapped = self._H @ pt
        w = float(mapped[2])
        if abs(w) < 1e-6:
            return None
        X = mapped[0] / w
        Y = mapped[1] / w
        if not (np.isfinite(X) and np.isfinite(Y)):
            return None
        return float(X), float(Y)


def build_projector(cfg: dict) -> GroundProjector:
    """根据配置构建地面投影器。"""
    proj_cfg = cfg.get("projector") if isinstance(cfg, dict) else None
    if proj_cfg is None:
        proj_cfg = cfg
    proj_type = (proj_cfg.get("type") or "homography").lower()
    if proj_type == "homography":
        return HomographyProjector(proj_cfg)
    raise ValueError(f"未知的投影类型: {proj_type}")
