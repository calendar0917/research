from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import tr_common as C  # noqa: E402


@pytest.fixture(scope="session")
def task():
    try:
        return C.load_task_gated()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PCBA cache unavailable: {type(exc).__name__}: {exc}")


@pytest.fixture(scope="session")
def spec(task):
    return task.spec


@pytest.fixture(scope="session")
def norm(task):
    return task.norm


@pytest.fixture(scope="session")
def plan(task):
    return C.make_plan40(task)


@pytest.fixture(scope="session")
def model(task):
    m, audit = C.build_w2(task, device="cpu")
    return m, audit


@pytest.fixture(scope="session")
def first_batch(task, plan):
    rows = np.asarray(plan[0]["batches"][0], dtype=np.int64)
    batch = C.BC.PC.collate(task.cache, rows.tolist())
    y = torch.as_tensor(task.labels[rows], dtype=torch.float32)
    return batch, y


def toy_logits(seed: int = 11, batch: int = 23, tasks: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(int(seed))
    logits = torch.randn(batch, tasks, generator=gen)
    raw = torch.randn(batch, tasks, generator=gen)
    y = torch.where(raw > 0.35, 1.0, torch.where(raw < -0.15, 0.0, float("nan")))
    return logits, y
