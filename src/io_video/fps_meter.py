import time

class FPSMeter:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self._prev = None
        self.fps = 0.0

    def tick(self, now=None):
        now = now or time.time()
        if self._prev is None:
            self._prev = now
            return self.fps
        dt = max(1e-6, now - self._prev)
        inst = 1.0 / dt
        self.fps = (1 - self.alpha) * self.fps + self.alpha * inst
        self._prev = now
        return self.fps
