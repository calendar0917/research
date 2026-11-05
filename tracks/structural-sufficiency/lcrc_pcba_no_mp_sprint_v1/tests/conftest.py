import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import nmp_common as C  # noqa: E402
import nmp_data as D  # noqa: E402
import nmp_model as M  # noqa: E402


@pytest.fixture(scope="session")
def task():
    return D.load_task()


@pytest.fixture(scope="session")
def reference0(task):
    return M.build_reference(task.spec, task.norm, C.N_TASKS, 0)


@pytest.fixture(scope="session")
def reference1(task):
    return M.build_reference(task.spec, task.norm, C.N_TASKS, 1)
