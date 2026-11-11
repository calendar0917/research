"""E3 -- bounded async prefetch of the CPU batch assembly.

The producer is one daemon thread per epoch with a small bounded queue
(default depth 3, so at most ~3 x 1.3 MB of collated tensors are alive).

Why a thread and not a process pool: the collated batch is the *only* object
that crosses the boundary, the mmap cache arrays are read-only and shared by
the process anyway, and a thread cannot duplicate the 5.5 GB mapping.  The
numpy gathers release the GIL for their array work, so the producer overlaps
part of the collation with the consumer.  If the measured gain had required
real CPU parallelism, a process pool over the same read-only mmap would have
been the fallback; it was not needed (see REPORT, stage 2).

The queue is bounded in *both* directions: the producer blocks when the
consumer is behind and the consumer blocks when the producer is behind, so
memory stays constant and ordering is exactly the epoch order.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

import ew_common as C

Collator = Callable[[Mapping[str, Any], Sequence[int]], dict[str, Any]]


class Prefetcher:
    """Ordered, bounded producer of ``(rows, raw_batch, y_z)`` tuples."""

    def __init__(
        self,
        cache: Mapping[str, Any],
        y_z: np.ndarray,
        *,
        collator: Collator,
        batch_size: int = C.BATCH_SIZE,
        depth: int = 3,
    ):
        self.cache = cache
        self.y_z = y_z
        self.collator = collator
        self.batch_size = int(batch_size)
        self.depth = max(1, int(depth))
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self.counters = {"batches": 0, "wait_seconds": 0.0, "produce_seconds": 0.0}

    # -- producer -----------------------------------------------------------
    def _produce(self, order: np.ndarray) -> None:
        assert self._queue is not None
        try:
            for i in range(0, int(order.size), self.batch_size):
                if self._stop.is_set():
                    break
                rows = order[i : i + self.batch_size]
                t = time.perf_counter()  # type: ignore[attr-defined]
                raw = self.collator(self.cache, rows.tolist())
                y = np.ascontiguousarray(self.y_z[rows], dtype=np.float32)
                self.counters["produce_seconds"] += float(time.perf_counter() - t)  # type: ignore[attr-defined]
                item = (rows, raw, y)
                while not self._stop.is_set():
                    try:
                        self._queue.put(item, timeout=0.05)
                        self.counters["batches"] += 1
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:  # noqa: BLE001 - forwarded to the consumer
            self._error = exc
        finally:
            try:
                self._queue.put(None, timeout=1.0)
            except queue.Full:  # pragma: no cover - defensive
                pass

    # -- consumer -----------------------------------------------------------
    def epoch(self, order: np.ndarray) -> Iterator[tuple[np.ndarray, dict[str, Any], np.ndarray]]:
        self._queue = queue.Queue(maxsize=self.depth)
        self._stop = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._produce, args=(order,), daemon=True)
        self._thread.start()
        try:
            while True:
                t = time.perf_counter()  # type: ignore[attr-defined]
                item = self._queue.get()
                self.counters["wait_seconds"] += float(time.perf_counter() - t)  # type: ignore[attr-defined]
                if item is None:
                    break
                yield item
        finally:
            self.close()
        if self._error is not None:
            raise RuntimeError(f"prefetcher producer failed: {type(self._error).__name__}: {self._error}") from self._error

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=10)
        self._thread = None
        if self._queue is not None:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break


__all__ = ["Prefetcher"]
