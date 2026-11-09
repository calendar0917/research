import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import cpc_common as C  # noqa: E402
import cpc_model as M  # noqa: E402
import cpc_paths as P  # noqa: E402


@pytest.fixture(scope="session")
def task():
    return C.SD.load_task()


@pytest.fixture(scope="session")
def cpc_cache(task, tmp_path_factory):
    rows = list(map(int, task.train_rows[:24])) + list(map(int, task.valid_rows[:8]))
    out = tmp_path_factory.mktemp("cpc_cache")
    P.build_cpc_cache(task, out, rows=rows, log_every=0)
    return P.load_cpc_cache(out)


@pytest.fixture(scope="session")
def task_with_cpc(task, cpc_cache):
    task.cpc = cpc_cache
    return task


@pytest.fixture(scope="session")
def batch(task_with_cpc):
    import torch

    rows = [int(x) for x in task_with_cpc.train_rows[:6]]
    return {
        k: (v.clone() if torch.is_tensor(v) else v)
        for k, v in P.collate_with_cpc(task_with_cpc, task_with_cpc.cpc, rows).items()
    }


@pytest.fixture(scope="session")
def built(task):
    return M.build_cpc_model(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")


@pytest.fixture(scope="session")
def model(built):
    return built[0]


@pytest.fixture(scope="session")
def model_audit(built):
    return built[1]


@pytest.fixture(scope="session")
def reference(task):
    return M.untrained_w2(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")[0]


@pytest.fixture(scope="session")
def nondegenerate(task):
    return M.build_cpc_model(
        task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu", zero_output=False
    )[0]
