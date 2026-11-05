"""Device-transfer strategies: the frozen one and the optimised one.

Measured bottleneck (``analysis/profile.json`` + ``analysis/h2d_probe.json``):
the host-to-device phase costs **29.1 ms of the 73.4 ms step (40%)**, of which
26.6 ms is host time inside 21 *pageable* ``cudaMemcpyAsync + stream sync``
round trips (~1.27 ms of fixed cost per tensor, ~1.75 MB of payload).  The
memcpy engine itself only needs ~0.16 ms; a single pageable copy of the same
1.75 MB takes 0.32 ms, and a pinned async copy 0.24 ms.  So the cost is the
*number of pageable copies*, not bandwidth.

``transfer_pageable`` is the frozen ``pc_model.to_device`` (unchanged).
``PinnedStager`` stages into pinned host memory and issues non-blocking copies,
keeping the pinned sources alive for ``depth`` steps (pinned blocks are cheap:
1.75 MB per step).  Same tensors, same dtypes, same shapes, same bytes.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


def transfer_pageable(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Verbatim copy of ``pc_model.to_device`` (the frozen behaviour)."""
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def labels_pageable(y_rows: np.ndarray, device: torch.device) -> torch.Tensor:
    """Verbatim copy of the label transfer inside ``pc_train.train``."""
    return torch.as_tensor(y_rows, dtype=torch.float32, device=device)


class PinnedStager:
    """Stage CPU tensors in pinned memory, then copy asynchronously.

    The pinned sources are retained for ``depth`` calls; with a stream that is
    drained at least once per step (the frozen loop does it several times) one
    step of retention is already sufficient, two is belt and braces.
    """

    def __init__(self, device: torch.device, depth: int = 2):
        self.device = device
        self.depth = int(depth)
        self._keep: list[list[torch.Tensor]] = []
        self.counters = {"batch_calls": 0, "label_calls": 0, "tensors_staged": 0}
        #: pinned host memory only exists when a CUDA device is present; the
        #: CPU-only path is a dry-run convenience and is recorded as such.
        self.pinned_available = bool(torch.cuda.is_available())

    def _pin(self, value: torch.Tensor) -> torch.Tensor:
        if not self.pinned_available:
            return value
        return value if value.is_pinned() else value.pin_memory()

    def _retain(self, refs: list[torch.Tensor]) -> None:
        self._keep.append(refs)
        while len(self._keep) > self.depth:
            self._keep.pop(0)

    def batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        refs: list[torch.Tensor] = []
        for key, value in batch.items():
            if torch.is_tensor(value):
                pinned = self._pin(value)
                refs.append(pinned)
                moved[key] = pinned.to(self.device, non_blocking=self.pinned_available)
            else:
                moved[key] = value
        self._retain(refs)
        self.counters["batch_calls"] += 1
        self.counters["tensors_staged"] += len(refs)
        return moved

    def labels(self, y_rows: np.ndarray) -> torch.Tensor:
        arr = np.ascontiguousarray(y_rows, dtype=np.float32)
        pinned = self._pin(torch.from_numpy(arr))
        self._retain([pinned])
        self.counters["label_calls"] += 1
        return pinned.to(self.device, non_blocking=self.pinned_available)

    def release(self) -> None:
        self._keep.clear()


class PageableTransfer:
    """The frozen transfer path behind the same interface, for the A/B."""

    def __init__(self, device: torch.device):
        self.device = device
        self.counters = {"batch_calls": 0, "label_calls": 0, "tensors_staged": 0}

    def batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        self.counters["batch_calls"] += 1
        self.counters["tensors_staged"] += sum(1 for v in batch.values() if torch.is_tensor(v))
        return transfer_pageable(batch, self.device)

    def labels(self, y_rows: np.ndarray) -> torch.Tensor:
        self.counters["label_calls"] += 1
        return labels_pageable(y_rows, self.device)

    def release(self) -> None:
        return None


def make_transfer(name: str, device: torch.device) -> PageableTransfer | PinnedStager:
    if name in ("pageable", "original", "baseline"):
        return PageableTransfer(device)
    if name in ("pinned", "fast", "optimized", "optimised"):
        return PinnedStager(device)
    raise KeyError(name)


__all__ = [
    "PageableTransfer",
    "PinnedStager",
    "labels_pageable",
    "make_transfer",
    "transfer_pageable",
]
