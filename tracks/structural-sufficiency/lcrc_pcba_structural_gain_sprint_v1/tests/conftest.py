import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import sg_common as C  # noqa: E402
import sg_checkpoint as CK  # noqa: E402
import sg_data as D  # noqa: E402
import sg_model as M  # noqa: E402


@pytest.fixture(scope="session")
def task():
    return D.load_task()


@pytest.fixture(scope="session")
def source():
    return CK.load_source_checkpoint()


@pytest.fixture(scope="session")
def reference(task):
    return M.build_reference(task.spec, task.norm, C.N_TASKS, C.SEED)
