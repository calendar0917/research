import sys
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1]
CODE = EXP / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))


@pytest.fixture(scope="session")
def cfg():
    import r42_common as C

    return C.load_config()


@pytest.fixture(scope="session")
def split():
    import r42_data as D

    return D.load_split()


@pytest.fixture(scope="session")
def labels():
    import r42_labels as L

    return L.load_train_valid_labels()


@pytest.fixture(scope="session")
def structures():
    import r42_data as D

    return D.load_structure_data()
