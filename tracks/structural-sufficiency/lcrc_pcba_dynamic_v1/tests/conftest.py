import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import dy_common as C  # noqa: E402
import dy_model as M  # noqa: E402


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
def built(spec, norm):
    return M.build_dynamic_model(spec, norm, C.N_TASKS, seed=C.SEED, device="cpu")


@pytest.fixture(scope="session")
def model(built):
    return built[0]


@pytest.fixture(scope="session")
def model_audit(built):
    return built[1]


@pytest.fixture(scope="session")
def nondegenerate(spec, norm):
    return M.build_dynamic_model(
        spec, norm, C.N_TASKS, seed=C.SEED, device="cpu", zero_residual=False
    )[0]


@pytest.fixture(scope="session")
def reference(spec, norm):
    return M.untrained_w2(spec, norm, C.N_TASKS, seed=C.SEED, device="cpu")[0]


@pytest.fixture(scope="session")
def batch(task):
    import torch

    rows = [int(x) for x in task.train_rows[:8]]
    return {
        k: (v.clone() if torch.is_tensor(v) else v)
        for k, v in C.PC.collate(task.cache, rows).items()
    }
