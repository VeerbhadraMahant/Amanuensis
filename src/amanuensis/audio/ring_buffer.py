import threading

import numpy as np


class RingBuffer:
    """Thread-safe float32 ring buffer. The capture callback writes; the pipeline reads."""

    def __init__(self, capacity: int) -> None:
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._cap = capacity
        self._start = 0
        self._size = 0
        self._lock = threading.Lock()
        self.overruns = 0  # samples dropped because the reader fell behind

    def __len__(self) -> int:
        return self._size

    def write(self, samples: np.ndarray) -> None:
        with self._lock:
            n = len(samples)
            if n >= self._cap:
                self.overruns += self._size + n - self._cap
                samples, n = samples[-self._cap :], self._cap
                self._start, self._size = 0, 0
            drop = max(0, self._size + n - self._cap)
            if drop:
                self.overruns += drop
                self._start = (self._start + drop) % self._cap
                self._size -= drop
            end = (self._start + self._size) % self._cap
            first = min(n, self._cap - end)
            self._buf[end : end + first] = samples[:first]
            self._buf[: n - first] = samples[first:]
            self._size += n

    def read(self, n: int) -> np.ndarray | None:
        """Pop exactly n samples, or return None if fewer are available."""
        with self._lock:
            if self._size < n:
                return None
            idx = (self._start + np.arange(n)) % self._cap
            out = self._buf[idx].copy()
            self._start = (self._start + n) % self._cap
            self._size -= n
            return out
