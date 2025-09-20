from __future__ import annotations

import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional

from ..config import load_config, resolve_config_path


class RuntimeConfigStore:
    """Thread-safe store that keeps the latest configuration snapshot."""

    def __init__(self, initial_cfg: dict) -> None:
        self._lock = threading.Lock()
        self._cfg = deepcopy(initial_cfg)
        self._version = 0

    def snapshot(self) -> tuple[int, dict]:
        """Return a deep-copied configuration with its version."""
        with self._lock:
            return self._version, deepcopy(self._cfg)

    def version(self) -> int:
        with self._lock:
            return self._version

    def get_config(self) -> dict:
        with self._lock:
            return deepcopy(self._cfg)

    def update(self, cfg: dict) -> int:
        with self._lock:
            self._cfg = deepcopy(cfg)
            self._version += 1
            return self._version


class ConfigWatcher(threading.Thread):
    """Watch a configuration file and push updates to :class:`RuntimeConfigStore`."""

    def __init__(
        self,
        path: str | Path | None,
        store: RuntimeConfigStore,
        interval: float = 2.0,
        loader: Optional[Callable[[str | Path], dict]] = None,
    ) -> None:
        super().__init__(daemon=True)
        self._store = store
        self._interval = max(0.2, float(interval))
        self._stop_event = threading.Event()
        self._loader = loader or (lambda p: load_config(str(p)))
        self._path = resolve_config_path(path)
        self._last_mtime: Optional[float] = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:  # pragma: no cover - background thread
        while not self._stop_event.is_set():
            try:
                stat = Path(self._path).stat()
                mtime = float(stat.st_mtime)
            except FileNotFoundError:
                mtime = None

            if mtime is not None and self._last_mtime is None:
                self._last_mtime = mtime
            elif (
                mtime is not None
                and self._last_mtime is not None
                and mtime > self._last_mtime + 1e-3
            ):
                try:
                    cfg = self._loader(self._path)
                except Exception as exc:  # pragma: no cover - best effort logging
                    print(f"⚠️ 配置重载失败: {exc}")
                else:
                    self._store.update(cfg)
                    self._last_mtime = mtime
            time.sleep(self._interval)


__all__ = ["RuntimeConfigStore", "ConfigWatcher"]
