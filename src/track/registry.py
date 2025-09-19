"""跟踪器注册表。"""
from __future__ import annotations

from typing import Any, Dict

from .base import Tracker
from .sort_tracker import SortTracker


def build_tracker(cfg: Dict[str, Any]) -> Tracker:
    backend = (cfg.get("backend") or "sort").lower()
    if backend == "sort":
        return SortTracker(cfg)
    raise ValueError(f"未知跟踪后端: {backend}")
