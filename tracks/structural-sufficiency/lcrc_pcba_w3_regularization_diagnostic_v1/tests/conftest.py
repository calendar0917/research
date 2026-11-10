"""Shared fixtures: the frozen task, the two runs, a real batch and tiny splits."""

import copy
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import rd_common as C  # noqa: E402
import rd_data as D  # noqa: E402
import rd_model as M  # noqa: E402


@pytest.fixture(scope="session")
def task():
    return D.load_task()


@pytest.fixture(scope="session")
def test_labels(task):
    return D.read_official_test_labels(task)


@pytest.fixture(scope="session")
def splits(task, test_labels):
    return D.build_split_data(task, test_labels["labels"])


@pytest.fixture(scope="session")
def plan(task):
    return C.SD.make_plan40(task, seed=C.SEED)


@pytest.fixture(scope="session")
def batch(task, plan):
    import torch

    rows = [int(r) for r in plan[0]["batches"][0]]
    return C.PC.collate(task.cache, rows)


@pytest.fixture(scope="session")
def models(task):
    import torch

    out = {}
    for key in C.RUN_ORDER:
        model, audit = M.build_model(key, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
        out[key] = (model, audit)
    return out


def _tiny(task, n_train: int = 256, n_valid: int = 256, n_test: int = 256):
    tiny = copy.copy(task)
    tiny.train_rows = task.train_rows[:n_train]
    tiny.valid_rows = task.valid_rows[:n_valid]
    tiny.test_rows = task.test_rows[:n_test]
    tiny.split = {"train": tiny.train_rows, "valid": tiny.valid_rows, "test": tiny.test_rows}
    return tiny


@pytest.fixture(scope="session")
def tiny_task(task):
    return _tiny(task)


@pytest.fixture(scope="session")
def tiny_plan(tiny_task):
    return C.SD.make_plan40(tiny_task, seed=C.SEED)


@pytest.fixture(scope="session")
def tiny_splits(tiny_task):
    return D.build_split_data(tiny_task, _official_test_labels_for(tiny_task.test_rows))


def _official_test_labels_for(rows):
    import numpy as np

    raw, _ = D.read_raw_label_matrix()
    return np.ascontiguousarray(raw[np.asarray(rows, dtype=np.int64)], dtype=np.float32)


@pytest.fixture(scope="session")
def torch_batch(task, plan):
    import torch

    rows = [int(r) for r in plan[0]["batches"][0]]
    batch = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in C.PC.collate(task.cache, rows).items()}
    y = torch.as_tensor(task.labels[[int(r) for r in rows]], dtype=torch.float32)
    return batch, y
