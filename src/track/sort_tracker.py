"""轻量级 SORT 跟踪实现，支持时间戳与测距/测速。"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from filterpy.kalman import KalmanFilter
except ImportError as exc:  # pragma: no cover - 依赖缺失时给出提示
    raise ImportError("未安装 filterpy，请先 pip install filterpy") from exc

from src.detect.types import Detection
from src.geometry import GroundProjector

from .base import Tracker

BBox = Tuple[float, float, float, float]


def _bbox_to_z(bbox: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w = max(1e-3, float(x2) - float(x1))
    h = max(1e-3, float(y2) - float(y1))
    cx = float(x1) + 0.5 * w
    cy = float(y1) + 0.5 * h
    s = w * h
    r = w / h
    return np.array([[cx], [cy], [s], [r]], dtype=np.float32)


def _x_to_bbox(state: np.ndarray) -> np.ndarray:
    cx, cy, s, r = state[:4].reshape(-1)
    w = math.sqrt(max(1e-6, s * r))
    h = s / max(1e-6, w)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _create_kf() -> KalmanFilter:
    kf = KalmanFilter(dim_x=7, dim_z=4)
    kf.F = np.eye(7, dtype=float)
    kf.H = np.zeros((4, 7), dtype=float)
    kf.H[:4, :4] = np.eye(4, dtype=float)
    kf.R[2:, 2:] *= 10.0
    kf.P[4:, 4:] *= 1000.0
    kf.P *= 10.0
    return kf


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return float(inter_area / denom)


def _iou_matrix(track_boxes: np.ndarray, det_boxes: np.ndarray) -> np.ndarray:
    if track_boxes.size == 0 or det_boxes.size == 0:
        return np.zeros((track_boxes.shape[0], det_boxes.shape[0]), dtype=np.float32)
    ious = np.zeros((track_boxes.shape[0], det_boxes.shape[0]), dtype=np.float32)
    for i, t in enumerate(track_boxes):
        for j, d in enumerate(det_boxes):
            ious[i, j] = _iou(t, d)
    return ious


class _Track:
    def __init__(self, track_id: int, bbox: BBox, timestamp: float, min_hits: int, speed_window: float) -> None:
        self.id = track_id
        self.kf = _create_kf()
        self._update_motion_matrix(1.0)
        self.kf.x[:4, 0] = _bbox_to_z(bbox).reshape(-1)
        self.last_predict_ts = float(timestamp)
        self.last_update_ts = float(timestamp)
        self.first_timestamp = float(timestamp)
        self.min_hits = max(1, int(min_hits))
        self.speed_window = max(0.05, float(speed_window))
        self.hits = 1
        self.hit_streak = 1
        self.history_positions: List[Tuple[float, float, float]] = []
        self.current_distance: Optional[float] = None
        self.current_speed: Optional[float] = None
        self.class_id: Optional[int] = None
        self.cls_name: Optional[str] = None
        self.confidence: Optional[float] = None

    def _update_motion_matrix(self, dt: float) -> None:
        dt = float(max(1e-3, dt))
        F = np.eye(7, dtype=float)
        F[0, 4] = dt
        F[1, 5] = dt
        F[2, 6] = dt
        self.kf.F = F
        q = np.zeros((7, 7), dtype=float)
        q[0, 0] = q[1, 1] = q[2, 2] = 0.04 * dt * dt
        q[4, 4] = q[5, 5] = q[6, 6] = 1.0 * dt
        self.kf.Q = q

    def predict(self, timestamp: float) -> np.ndarray:
        dt = float(timestamp) - self.last_predict_ts
        self._update_motion_matrix(dt)
        self.kf.predict()
        self.last_predict_ts = float(timestamp)
        return self.get_state()

    def update(self, bbox: BBox, timestamp: float, det: Detection) -> None:
        dt = float(timestamp) - self.last_predict_ts
        self._update_motion_matrix(dt)
        self.kf.update(_bbox_to_z(bbox))
        self.last_predict_ts = float(timestamp)
        self.last_update_ts = float(timestamp)
        self.hits += 1
        self.hit_streak += 1
        self.class_id = det.cls_id
        self.cls_name = det.cls_name
        self.confidence = det.conf

    def mark_missed(self) -> None:
        self.hit_streak = 0

    def time_since_update(self, timestamp: float) -> float:
        return float(timestamp) - self.last_update_ts

    def get_state(self) -> np.ndarray:
        return _x_to_bbox(self.kf.x)

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= self.min_hits

    def update_metrics(self, projector: GroundProjector, bbox: BBox, timestamp: float) -> None:
        ground = projector.project_bbox(bbox)
        if ground is None:
            self.current_distance = None
            self.current_speed = None
            return
        self.current_distance = projector.distance(ground)
        self.history_positions.append((float(timestamp), float(ground[0]), float(ground[1])))
        # 只保留时间窗口内的历史
        while self.history_positions and (float(timestamp) - self.history_positions[0][0]) > self.speed_window:
            self.history_positions.pop(0)
        if len(self.history_positions) > 32:
            self.history_positions = self.history_positions[-32:]

        if len(self.history_positions) >= 2:
            t0, x0, y0 = self.history_positions[0]
            t1, x1, y1 = self.history_positions[-1]
            dt = max(1e-3, t1 - t0)
            dist = math.hypot(x1 - x0, y1 - y0)
            self.current_speed = dist / dt
        else:
            self.current_speed = None


class SortTracker(Tracker):
    """基于 SORT 的轻量跟踪器，支持真实时间戳。"""

    def __init__(self, cfg: dict) -> None:
        self.max_staleness = float(cfg.get("max_staleness", 1.0))
        self.min_hits = int(cfg.get("min_hits", 3))
        self.iou_threshold = float(cfg.get("iou_threshold", 0.3))
        self.speed_window = float(cfg.get("speed_window", 0.75))
        self._tracks: List[_Track] = []
        self._next_id = 1

    def _associate(self, detections: List[Detection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not self._tracks or not detections:
            return [], list(range(len(self._tracks))), list(range(len(detections)))

        track_boxes = np.array([trk.get_state() for trk in self._tracks], dtype=np.float32)
        det_boxes = np.array([[det.x1, det.y1, det.x2, det.y2] for det in detections], dtype=np.float32)
        iou_matrix = _iou_matrix(track_boxes, det_boxes)

        matches: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(len(self._tracks)))
        unmatched_dets = set(range(len(detections)))

        if iou_matrix.size == 0:
            return matches, list(unmatched_tracks), list(unmatched_dets)

        while True:
            idx = int(np.argmax(iou_matrix))
            max_iou = float(iou_matrix.flat[idx])
            if max_iou < self.iou_threshold:
                break
            t_idx, d_idx = np.unravel_index(idx, iou_matrix.shape)
            if t_idx in unmatched_tracks and d_idx in unmatched_dets:
                matches.append((int(t_idx), int(d_idx)))
                unmatched_tracks.remove(int(t_idx))
                unmatched_dets.remove(int(d_idx))
            iou_matrix[t_idx, :] = -1.0
            iou_matrix[:, d_idx] = -1.0

        return matches, list(unmatched_tracks), list(unmatched_dets)

    def update(
        self,
        detections: Iterable[Detection],
        timestamp: float,
        projector: Optional[GroundProjector] = None,
    ) -> List[Detection]:
        det_list = list(detections)
        for det in det_list:
            det.track_id = None
            det.distance_m = None
            det.speed_kmh = None

        if not det_list and not self._tracks:
            return det_list

        # 预测所有轨迹状态
        for trk in self._tracks:
            trk.predict(timestamp)

        matches, unmatched_tracks, unmatched_dets = self._associate(det_list)

        # 更新匹配轨迹
        for t_idx, d_idx in matches:
            track = self._tracks[t_idx]
            det = det_list[d_idx]
            bbox = (det.x1, det.y1, det.x2, det.y2)
            track.update(bbox, timestamp, det)
            if projector is not None:
                track.update_metrics(projector, bbox, timestamp)
            det.track_id = track.id
            if track.current_distance is not None:
                det.distance_m = track.current_distance
            elif projector is not None:
                det.distance_m = projector.distance_for_bbox(bbox)
            if track.current_speed is not None:
                det.speed_kmh = track.current_speed * 3.6

        # 未匹配轨迹标记为 miss
        for idx in unmatched_tracks:
            self._tracks[idx].mark_missed()

        # 为新的检测创建轨迹
        for idx in unmatched_dets:
            det = det_list[idx]
            bbox = (det.x1, det.y1, det.x2, det.y2)
            track = _Track(self._next_id, bbox, timestamp, self.min_hits, self.speed_window)
            track.class_id = det.cls_id
            track.cls_name = det.cls_name
            track.confidence = det.conf
            if projector is not None:
                track.update_metrics(projector, bbox, timestamp)
                if track.current_distance is not None:
                    det.distance_m = track.current_distance
                if track.current_speed is not None:
                    det.speed_kmh = track.current_speed * 3.6
            det.track_id = track.id
            self._tracks.append(track)
            self._next_id += 1

        # 移除长时间未更新的轨迹
        alive_tracks: List[_Track] = []
        for trk in self._tracks:
            if trk.time_since_update(timestamp) <= self.max_staleness:
                alive_tracks.append(trk)
        self._tracks = alive_tracks

        return det_list

    def close(self) -> None:
        self._tracks.clear()
