"""Shared plumbing for ``zinc_block_ownership_v2``.

Strategy: **reuse** the frozen v1 code instead of copying it.  Every model,
block-partition, collate and training-loop line comes from

    tracks/structural-sufficiency/zinc_block_ownership_v1/code/

and is byte-verified here before anything runs (config sha, gates sha, and the
per-file ``code_sha256`` block that v1 froze).  v2 owns only:

* a separate ``runs/`` and ``analysis/`` directory, so no v1 artifact is
  overwritten;
* the Phase A diagnostics, the S-arm authorization gates and the Phase B/C
  decision rules (``configs/zb2_v1.json``).

The official test split is never constructed, opened, predicted or scored.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
SUFF = ROUND.parent
REPO = SUFF.parents[1]

V1ROUND = SUFF / "zinc_block_ownership_v1"
V1CODE = V1ROUND / "code"
for _p in (str(V1CODE), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import zbo_analyze as ZA  # noqa: E402
import zbo_blocks as ZB  # noqa: E402
import zbo_common as BC  # noqa: E402
import zbo_models as ZM  # noqa: E402
import zbo_train as ZT  # noqa: E402

ROUND_ID = "zinc-block-ownership-v2"
DRIVER_ROUND_ID = BC.ROUND_ID
CONFIGS = ROUND / "configs"
CONFIG = CONFIGS / "zb2_v1.json"
RUNS = ROUND / "runs"
RUNS_SMOKE = ROUND / "runs_smoke"
ANALYSIS = Path(os.environ.get("ZB2_ANALYSIS_DIR", str(ROUND / "analysis")))

V1_CONFIG = V1ROUND / "configs" / "zbo_v1.json"
V1_CONFIG_SHA = "151deb021d12ad77d8997d5abf99c3a42012efd9bc73ca9e15f7fac35958c0bd"
V1_GATES = V1ROUND / "analysis" / "cpu_checks.json"
V1_GATES_SHA = "177271fb541ab5f732ab1f8303435ee016cfe4e05a147d2130c32595bc27a9fb"
V1_PREP = V1ROUND / "analysis" / "prep.json"

# v1's frozen caches are reused verbatim; the trainer resolves them through
# ``BC.CACHE``, so pinning it here is the whole "reuse" switch.
BC.CACHE = V1ROUND / "cache"
BC.RUNS = RUNS
BC.RUNS_SMOKE = RUNS_SMOKE
BC.ANALYSIS = ANALYSIS

# cache fingerprints recorded by v1's prep (file sha256)
V1_CACHE_SHA = {
    ("rel", "train"): "afa9ace02624db7e39bd730a77fb30f77f7831c59a00b600941b4e1776b38ba2",
    ("rel", "valid"): None,      # filled from prep.json at check time
    ("blocks", "train"): "c9f13d5447c4567ec9902320cd3318e723b395c5be97457d2430b56d6e1af788",
    ("blocks", "valid"): None,
}

BOOT = 2000
BOOT_SEED = 20261027


# ---------------------------------------------------------------------------
# integrity
# ---------------------------------------------------------------------------


def v1_integrity() -> dict:
    """Assert that the v1 round v2 depends on has not moved under us."""
    checks = {
        "v1_config_path": BC.rel_or_abs(V1_CONFIG),
        "v1_config_sha256": BC.sha256_file(V1_CONFIG),
        "v1_config_sha256_expected": V1_CONFIG_SHA,
        "v1_gates_sha256": BC.sha256_file(V1_GATES),
        "v1_gates_sha256_expected": V1_GATES_SHA,
    }
    checks["config_unchanged"] = checks["v1_config_sha256"] == V1_CONFIG_SHA
    checks["gates_unchanged"] = checks["v1_gates_sha256"] == V1_GATES_SHA
    cfg = read_json(V1_CONFIG)
    per_file = {}
    for name, want in cfg.get("code_sha256", {}).items():
        p = V1CODE / name
        got = BC.sha256_file(p) if p.exists() else None
        per_file[name] = {"sha256": got, "expected": want, "match": got == want}
    checks["code_sha256"] = per_file
    checks["code_unchanged"] = all(v["match"] for v in per_file.values())
    prep = read_json(V1_PREP) if V1_PREP.exists() else {}
    cache = {}
    for kind in ("relation", "block"):
        for split in ("train", "valid"):
            key = "relation_cache" if kind == "relation" else "block_cache"
            sha_key = ("relation_file_sha256" if kind == "relation"
                       else "block_file_sha256")
            want = prep.get(split, {}).get(sha_key)
            p = (BC.rel_path(split) if kind == "relation" else BC.block_path(split))
            got = BC.sha256_file(p) if p.exists() else None
            cache[f"{kind}_{split}"] = {"path": BC.rel_or_abs(p), "sha256": got,
                                        "expected": want, "match": got == want}
    checks["caches"] = cache
    checks["caches_unchanged"] = all(v["match"] for v in cache.values())
    checks["ok"] = bool(checks["config_unchanged"] and checks["gates_unchanged"]
                        and checks["code_unchanged"] and checks["caches_unchanged"])
    return checks


def require_v1_integrity() -> dict:
    chk = v1_integrity()
    if not chk["ok"]:
        bad = [k for k in ("config_unchanged", "gates_unchanged",
                           "code_unchanged", "caches_unchanged") if not chk[k]]
        raise RuntimeError(f"v1 frozen inputs changed: {bad}")
    return chk


# ---------------------------------------------------------------------------
# io / hashing (delegate to the frozen helpers so hashes stay comparable)
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def read_json(path) -> Any:
    return BC.read_json(path)


def write_json(path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    BC.write_json(path, payload)


def sha256_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def now() -> str:
    return BC.now()


def jsonable(obj):
    return ZA.jsonable(obj)


def bootstrap_ci(values: np.ndarray, n: int = BOOT) -> list[float]:
    """Deterministic percentile bootstrap, identical rule to v1's analyser."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(values), size=(int(n), len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


# ---------------------------------------------------------------------------
# valid context + groups (v1 definitions, plus v2 cross-coverage cells)
# ---------------------------------------------------------------------------


def valid_context() -> dict:
    """v1's group context, reused verbatim, plus v2-only cells."""
    cfg = read_json(V1_CONFIG)
    ctx = ZA.load_valid_context(cfg)
    y = ctx["y"]
    vblocks = ctx["vblocks"]
    vaf = BC.block_row_features(vblocks)
    n_multi = np.asarray(vaf["n_multi_blocks"], dtype=np.int64)
    groups = dict(ctx["groups"])
    extra = {}
    # n_multi = 1 is the R/S input-equivalence negative control: with exactly one
    # multi-edge block, S's multi-block mean equals that block's own mean.
    extra["n_multi_eq_1"] = n_multi == 1
    extra["n_multi_eq_2"] = n_multi == 2
    extra["n_multi_ge_3"] = n_multi >= 3
    # cross-coverage cells (label-free structure x label-side split)
    tier = np.asarray(ctx["tiers"], dtype=object)
    for t in ("all_supported", "low_support", "any_unseen"):
        m = tier == t
        extra[f"{t}__ordinary"] = m & (y > -3.0)
        extra[f"{t}__tail"] = m & (y <= -3.0)
    extra["ordinary_and_sig_seen"] = ~np.asarray(ctx["sig_novel"]) & (y > -3.0)
    extra["ordinary_and_sig_novel"] = np.asarray(ctx["sig_novel"]) & (y > -3.0)
    extra["size_novel__ordinary"] = np.asarray(ctx["size_novel"]) & (y > -3.0)
    extra["size_novel__tail"] = np.asarray(ctx["size_novel"]) & (y <= -3.0)
    extra["sig_novel__low_support"] = np.asarray(ctx["sig_novel"]) & (tier == "low_support")
    extra["sig_novel__any_unseen"] = np.asarray(ctx["sig_novel"]) & (tier == "any_unseen")
    for k, m in list(extra.items()):
        if m.dtype != bool:
            raise RuntimeError(f"group {k} is not boolean")
    groups.update({k: v for k, v in extra.items() if v.any()})
    ctx["group_extra_empty"] = [k for k, v in extra.items() if not v.any()]
    ctx["groups"] = groups
    ctx["n_multi"] = n_multi
    ctx["vaf"] = vaf
    return ctx


def group_stat(delta: np.ndarray, mask: np.ndarray, ae_a: np.ndarray,
               ae_b: np.ndarray) -> dict:
    """One group's paired statistics (positive delta = arm A better)."""
    return ZA.group_stat(delta, mask, ae_a, ae_b)


# ---------------------------------------------------------------------------
# run loading
# ---------------------------------------------------------------------------


def load_run(run_dir: Path, ctx: Mapping[str, Any]) -> dict:
    """Load a formal run: metadata + saved predictions + verified CPU recompute.

    The recompute is a *consistency* gate (GPU float32 vs CPU float32 reduction
    order), not a new measurement: the reported numbers come from the saved
    predictions written by the trainer.
    """
    run_dir = Path(run_dir)
    info = read_json(run_dir / "train.json")
    arm = str(info["arm"])
    seed = int(info["seed"])
    norm = BC.load_norm()
    model = ZM.BlockBOSI(norm, block_arm=arm)
    model.eval()
    preds = {}
    import torch

    for tag in ("best", "final"):
        ck = torch.load(run_dir / f"{tag}.pt", map_location="cpu",
                        weights_only=False)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        preds[tag] = BC.predict(model, ctx["vcache"], BC.load_rel("valid"),
                                ctx["vblocks"], np.arange(len(ctx["y"]),
                                                          dtype=np.int64),
                                device="cpu")
    saved = {}
    maxdiff = {}
    for tag in ("best", "final"):
        z = np.load(run_dir / f"valid_{tag}.npz")
        saved[tag] = np.asarray(z["pred"], dtype=np.float64)
        assert np.array_equal(np.asarray(z["row"], dtype=np.int64),
                              np.arange(len(ctx["y"]), dtype=np.int64)), \
            f"{tag}: saved rows are not the official valid order"
        maxdiff[tag] = float(np.max(np.abs(saved[tag] - preds[tag])))
    y = ctx["y"]
    out = {
        "dir": BC.rel_or_abs(run_dir),
        "arm": arm,
        "seed": seed,
        "parameters": int(info["parameters"]),
        "best_valid": float(info["best"]["valid_mae"]),
        "best_epoch": int(info["best"]["epoch"]),
        "best_step": int(info["best"]["step"]),
        "final_valid": float(info["reads"]["final"]["valid_mae"]),
        "W20": float(info["W20"]),
        "T100": float(info["T100"]),
        "train_at_best": float(info["reads"]["best"]["train_mae_eval_mode"]),
        "train_at_final": float(info["reads"]["final"]["train_mae_eval_mode"]),
        "mechanism": info.get("mechanism", {}),
        "mechanism_active": bool(info.get("mechanism_active", False)),
        "gpu_hours": float(info.get("gpu_hours", 0.0)),
        "initial_state_sha256": info.get("initial", {}).get("initial_state_sha256"),
        "first_batch_rows_sha256": info.get("first_batch_rows_sha256"),
        "config_sha256": info.get("config_sha256"),
        "git_head": info.get("git_head"),
        "git_dirty": info.get("git_dirty"),
        "checkpoint_sha256": info.get("checkpoint_sha256", {}),
        "total_updates": int(info.get("total_updates", 0)),
        "official_test_loaded": bool(
            (info.get("data") or {}).get("official_test_loaded", True)),
        "pred": {k: saved[k] for k in saved},
        "pred_cpu_recompute_max_abs_diff": maxdiff,
        "pred_sha256": {k: sha256_array(np.asarray(saved[k], np.float32))
                        for k in saved},
        "ae": {k: np.abs(saved[k] - y) for k in saved},
        "_model": model,
        "_info": info,
    }
    return out


def probe_contents(run: Mapping[str, Any], ctx: Mapping[str, Any],
                   arm_from: str, arm_to: str) -> dict:
    """Fixed-weight intervention: same checkpoint, different block content."""
    rows = np.arange(len(ctx["y"]), dtype=np.int64)
    vrel = BC.load_rel("valid")
    p_from = BC.predict(run["_model"], ctx["vcache"], vrel, ctx["vblocks"], rows,
                        device="cpu", arm=arm_from)
    p_to = BC.predict(run["_model"], ctx["vcache"], vrel, ctx["vblocks"], rows,
                      device="cpu", arm=arm_to)
    y = ctx["y"]
    ae_from = np.abs(p_from - y)
    ae_to = np.abs(p_to - y)
    delta = ae_to - ae_from
    groups = {}
    for k, m in ctx["groups"].items():
        groups[k] = {"n": int(m.sum()), "delta_mae": float(delta[m].mean()),
                     "contribution": float(delta[m].mean() * m.mean())}
    return {
        "run": run["dir"], "model_arm": run["arm"], "seed": run["seed"],
        "content_from": arm_from, "content_to": arm_to,
        "mae_from": float(ae_from.mean()), "mae_to": float(ae_to.mean()),
        "mean_abs_pred_delta": float(np.mean(np.abs(p_to - p_from))),
        "max_abs_pred_delta": float(np.max(np.abs(p_to - p_from))),
        "n_rows_pred_changed_gt_1e-6": int((np.abs(p_to - p_from) > 1e-6).sum()),
        "delta_mae": float(delta.mean()),
        "delta_ci95": bootstrap_ci(delta),
        "delta_by_group": groups,
    }
