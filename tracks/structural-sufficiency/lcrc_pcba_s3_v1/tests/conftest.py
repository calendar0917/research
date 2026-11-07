from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import s3_common as C  # noqa: E402
from feature_spec import lipo_v1_spec  # noqa: E402


@pytest.fixture(scope="session")
def spec():
    return lipo_v1_spec()


@pytest.fixture(scope="session")
def norm(spec):
    """Stand-in normaliser: the trunk init does not depend on these values."""
    return {
        "global_mean": np.zeros(spec.global_dim, dtype=np.float32),
        "global_scale": np.ones(spec.global_dim, dtype=np.float32),
        "scalar_mean": np.zeros(spec.n_scalars, dtype=np.float32),
        "scalar_scale": np.ones(spec.n_scalars, dtype=np.float32),
    }


@pytest.fixture(scope="session")
def has_full_cache():
    return bool((C.S3_CACHE_DIR / "meta.json").is_file())


def pytest_collection_modifyitems(config, items):
    """`slow` tests need the on-disk PCBA cache and are skipped without it."""
    if not (C.S3_CACHE_DIR / "meta.json").is_file():
        skip = pytest.mark.skip(reason="S3 cache not built (run s3_audit.py --modes cache)")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)
