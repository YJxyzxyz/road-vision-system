from __future__ import annotations

import time
from pathlib import Path


class ConfigWatcher:
    """简单的配置文件轮询器."""

    def __init__(self, path: str | Path, interval: float = 2.0) -> None:
        self.path = Path(path)
        self.interval = max(0.2, float(interval))
        self._last_check = 0.0
        self._last_mtime: float | None = None

    def poll(self) -> bool:
        now = time.time()
        if now - self._last_check < self.interval:
            return False
        self._last_check = now
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return False
        mtime = stat.st_mtime
        if self._last_mtime is None:
            self._last_mtime = mtime
            return False
        if mtime > self._last_mtime + 1e-6:
            self._last_mtime = mtime
            return True
        return False
