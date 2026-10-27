"""Shared plumbing for ``zinc_support_weighted_objective_v1``.

The round asks one question: with the *strongest* frozen body (DCC-L / B-R0,
forward and 494,193 parameters untouched), does a conservative, train-only,
structure-derived per-molecule weighting of the raw-y L1 objective improve the
official-valid result?

E = uniform raw-y L1 (the historical R0 objective, unchanged).
W = the same objective with a frozen per-training-molecule weight vector built
    only from radius-2 structural support measured on the training 10k.

Nothing here touches the encoder, the composer, the reader, the optimizer, the
schedule, the batch order or the checkpoint rule.  Official test is never read.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_AD = _SS / "zinc_arch_discovery_v1" / "code"
_DCC = _SS / "zinc_dual_channel_composer_v1" / "code"
_BJM = _SS / "zinc_bosi_joint_moment_v1" / "code"
_ZPS = _SS / "zinc_performance_search_v1" / "code"
for _p in (str(_HERE), str(_AD), str(_DCC), str(_BJM), str(_ZPS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ROUND_ID = "zinc-support-weighted-objective-v1"
ROUND = _HERE.parent
RUNS = ROUND / "runs"
ANALYSIS = ROUND / "analysis"
CONFIGS = ROUND / "configs"
CONFIG = CONFIGS / "swo_v1.json"

# The frozen DCC-L seed-0 run that *is* the historical E anchor
# (zinc_performance_search_v1 records B-R0 s0 as a reuse of this exact run).
E_ANCHOR = _SS / "zinc_dual_channel_composer_v1" / "runs" / "L_s0"

PARAM_CAP = 500_000
N_BOOT = 10_000


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def rel(p) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROUND.parents[1].resolve()))
    except Exception:
        return str(p)


def read_json(p) -> dict:
    return json.loads(Path(p).read_text())


def write_json(p, obj) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n")
    tmp.replace(p)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def boot_ci(delta: np.ndarray, rng, n_boot: int = N_BOOT) -> tuple[float, float]:
    n = len(delta)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = delta[idx].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# --------------------------------------------------------------------------
# recipe / model identity
# --------------------------------------------------------------------------
def r0_recipe() -> dict:
    """The frozen R0 recipe, read from its owning round (no transcription)."""
    import zps_runner  # noqa: E402  (import after sys.path is set)
    return dict(zps_runner.RECIPES["R0"])


def dcc_candidate():
    """(candidate name, model builder, collate) for the DCC-L body (= B-R0)."""
    import dcc_runner as DR  # noqa: E402
    return "DCC_L", DR.make_builder("L"), DR.collate_p3n


# --------------------------------------------------------------------------
# frozen valid groupings (identical definitions to the rest of the track)
# --------------------------------------------------------------------------
def valid_groups() -> dict:
    import ad_data as D  # noqa: E402
    import bjm_common as BC  # noqa: E402

    vcache = D.load_cache("valid")
    tcache = D.load_cache("train")
    y = np.asarray(vcache["labels"], dtype=np.float64)
    sup = BC.radius2_support(BC.molecules_from_cache(tcache),
                             BC.molecules_from_cache(vcache))
    tier = BC.support_tier(sup["unseen_frac"], sup["low4_frac"])
    rings = np.asarray([BC.cycle_rank(g) >= 4 for g in BC.molecules_from_cache(vcache)])
    masks = {
        "overall": np.ones(len(y), dtype=bool),
        "ordinary_y_gt_-3": y > -3.0,
        "tail_y_le_-3": y <= -3.0,
        "radius2_all_supported": tier == "all_supported",
        "radius2_low_support": tier == "low_support",
        "radius2_any_unseen": tier == "any_unseen",
        "rings_ge4": rings,
        "drop_row172": np.arange(len(y)) != 172,
    }
    return {"y": y, "masks": masks, "support": {
        "unseen_frac_mean": float(sup["unseen_frac"].mean()),
        "low4_frac_mean": float(sup["low4_frac"].mean()),
        "train_vocab_size": int(sup["train_vocab_size"]),
    }}


def load_pred(run_dir: Path, ckpt: str = "best") -> np.ndarray | None:
    """Valid predictions of a run, from either the dcc or the ad layout."""
    d = Path(run_dir)
    cand = [d / f"valid_{ckpt}.npz", d / "predictions" / f"valid_{ckpt}.npz",
            d / f"predictions_{ckpt}.pt", d / "predictions" / f"predictions_{ckpt}.pt",
            d / "predictions" / f"{ckpt}.pt"]
    for p in cand:
        if not p.exists():
            continue
        if p.suffix == ".npz":
            return np.asarray(np.load(p)["pred"], dtype=np.float64)
        import torch
        blob = torch.load(p, map_location="cpu")
        if isinstance(blob, dict):
            for key in ("pred", "predictions", "valid_pred"):
                if key in blob:
                    return np.asarray(blob[key], dtype=np.float64).reshape(-1)
        return np.asarray(blob, dtype=np.float64).reshape(-1)
    return None
