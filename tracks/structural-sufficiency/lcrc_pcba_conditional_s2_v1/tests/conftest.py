import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import cs2_common as C  # noqa: E402
import cs2_model as M  # noqa: E402
import bs_model as BSM  # noqa: E402


@pytest.fixture(scope="session")
def task():
    return C.SD.load_task()


@pytest.fixture(scope="session")
def spec(task):
    return task.spec


@pytest.fixture(scope="session")
def norm(task):
    return task.norm


@pytest.fixture(scope="session")
def models(spec, norm):
    base, base_audit = BSM.build_model(C.BASE_MODEL_KEY, spec, norm, C.N_TASKS, seed=C.SEED, device="cpu")
    out = {"W2": (base, base_audit)}
    for key in C.MODEL_ORDER:
        out[key] = M.build_model(key, spec, norm, C.N_TASKS, seed=C.SEED, device="cpu", base_model=base)
    return out


@pytest.fixture(scope="session")
def batch(task):
    import torch

    rows = [int(x) for x in task.train_rows[:8]]
    return {
        k: (v.clone() if torch.is_tensor(v) else v)
        for k, v in C.PC.collate(task.cache, rows).items()
    }
