"""Contract tests for the route-resolution sprint.

The heavy checks live in ``code/sr_checks.py`` so they can be run as the formal
gate and write ``analysis/checks.json``.  These wrappers keep pytest discovery
honest without duplicating the implementations.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROUND = Path(__file__).resolve().parents[1]
CODE = ROUND / "code"


def test_catalogue_freeze_metadata() -> None:
    freeze = json.loads((ROUND / "analysis" / "catalogue" / "catalogue_freeze.json")
                        .read_text(encoding="utf-8"))
    assert freeze["labels_read"] is False
    assert freeze["n_pairs"] == 2133
    assert freeze["definition"]["cut"].startswith("bridge, OGB bond type SINGLE")
    assert freeze["pairs_sha256"]


@pytest.mark.slow
def test_pretraining_checks(tmp_path: Path) -> None:
    out = tmp_path / "checks.json"
    cp = subprocess.run([sys.executable, str(CODE / "sr_checks.py"),
                         "--out", str(out)], cwd=CODE,
                        capture_output=True, text=True, timeout=1800)
    assert cp.returncode == 0, cp.stdout[-4000:] + cp.stderr[-4000:]
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["all_pass"] is True
    assert payload["n_pass"] == payload["n_checks"]


@pytest.mark.slow
def test_wl_feature_invariance() -> None:
    sys.path.insert(0, str(CODE))
    import numpy as np
    import sr_catalogue as CAT
    import sr_ckernel as K
    import sr_data as D

    bundle = D.load_bundle("D0")
    arrays, _meta = CAT.load_frozen()
    cache = K.FeatureCache(bundle, arrays)
    groups: dict[tuple, np.ndarray] = {}
    worst = 0.0
    for i in range(int(arrays["row_a"].size)):
        for side in ("a", "b"):
            key = (arrays[f"s_key_{side}"][i], tuple(arrays[f"cut_{side}"][i]))
            ep = cache.get(int(arrays[f"row_{side}"][i]),
                           int(arrays[f"edge_{side}"][i]),
                           int(arrays[f"s_port_{side}"][i]),
                           int(arrays[f"c_port_{side}"][i]),
                           arrays[f"cut_{side}"][i],
                           int(arrays[f"s_size_{side}"][i]),
                           int(arrays["c_size"][i]))
            if key in groups:
                worst = max(worst, float(np.abs(groups[key] - ep.s_feat).max()))
            else:
                groups[key] = ep.s_feat
    assert worst <= 1e-12, worst