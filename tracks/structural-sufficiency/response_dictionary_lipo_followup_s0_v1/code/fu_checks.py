"""Targeted pre-flight checks (no new gate bureaucracy).

Each check answers one question the follow-up unit depends on:

* ``parameters``          -- JOINT/ADD/RESP expose the same parameter set and count;
* ``init_state``          -- the canonical S0 seed-0 state loads strictly (hash ``ccc8171d...``);
* ``w0_predictions``      -- at ``w = 0`` JOINT, RESP and ADD produce the same ``F(G)``;
* ``views_and_order``     -- frozen fragment cache + frozen Train order are the S0 ones;
* ``single_vs_batch``     -- a molecule scored alone equals the same molecule in a batch;
* ``three_steps``         -- three real optimiser steps on JOINT are finite and move ``w``;
* ``reproduce_epoch1``    -- the copied loop, run as RESP, reproduces S0 epoch 1.

Nothing here selects a model, a hyper-parameter or a checkpoint.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import fu_common as FC
import fu_model as FM
import rd_common as RC
import rd_model as RM
import rd_train as RT


def parameters(dataset, config, state) -> dict[str, Any]:
    out: dict[str, Any] = {"modes": {}}
    reference_keys = None
    for mode in ("RESP", "ADD", "JOINT"):
        model = FM.build_model(dataset.dims, FM.model_config(config, mode), state)
        layout = {k: list(v.shape) for k, v in model.state_dict().items()}
        if reference_keys is None:
            reference_keys = layout
        elif layout != reference_keys:
            raise AssertionError(f"{mode} parameter layout differs from RESP")
        out["modes"][mode] = {
            "n_parameters": RM.n_parameters(model),
            "state_hash": RM.state_hash(model.state_dict()),
        }
    counts = {v["n_parameters"] for v in out["modes"].values()}
    hashes = {v["state_hash"] for v in out["modes"].values()}
    expected = int(FC.load_config()["joint"]["expected_parameters"])
    if counts != {expected}:
        raise AssertionError(f"parameter counts {counts} != {expected}")
    if hashes != {FC.EXPECTED_INIT_STATE_HASH}:
        raise AssertionError(f"init hashes {hashes} != {FC.EXPECTED_INIT_STATE_HASH}")
    out["identical_layout"] = True
    out["expected_parameters"] = expected
    out["ok"] = True
    return out


def w0_predictions(dataset, config, state, device: str) -> dict[str, Any]:
    models = {mode: FM.build_model(dataset.dims, FM.model_config(config, mode), state).to(device)
              for mode in ("RESP", "ADD", "JOINT")}
    valid = dataset.splits["valid"]
    full: dict[str, np.ndarray] = {}
    for mode, model in models.items():
        model.eval()
        pred = RT.evaluate_split(model, dataset, "valid", device, int(config["eval_batch_size"]))
        full[mode] = np.asarray(pred["pred_raw"], dtype=np.float64)
    subset = np.arange(32, dtype=np.int64)
    small: dict[str, np.ndarray] = {}
    for mode, model in models.items():
        model.eval()
        scores, extras = RT.molecule_scores(model, dataset, dataset.splits["train"], subset, device)
        small[mode] = np.asarray(scores, dtype=np.float64)
    out = {
        "valid_max_abs_diff_JOINT_vs_RESP": float(np.max(np.abs(full["JOINT"] - full["RESP"]))),
        "valid_max_abs_diff_JOINT_vs_ADD": float(np.max(np.abs(full["JOINT"] - full["ADD"]))),
        "train32_max_abs_diff_JOINT_vs_RESP": float(np.max(np.abs(small["JOINT"] - small["RESP"]))),
        "n_valid": int(full["RESP"].size),
        "n_train_subset": int(subset.size),
        "tolerance": 1e-6,
    }
    tol = out["tolerance"]
    if max(out["valid_max_abs_diff_JOINT_vs_RESP"], out["valid_max_abs_diff_JOINT_vs_ADD"],
           out["train32_max_abs_diff_JOINT_vs_RESP"]) > tol:
        raise AssertionError("w=0 predictions are not identical across arms")
    out["ok"] = True
    return out


def views_and_order(dataset, order: np.ndarray, order_sha256: str) -> dict[str, Any]:
    expected = FC.EXPECTED_TRAIN_ORDER_SHA256_PREFIX
    manifest = RC.read_json(FC.S0_DIR / "data" / "fragment_cache_manifest.json")["ogbg-mollipo"]
    frag_path = FC.S0_DIR / "data" / "ogbg_mollipo_rd_frag_v1.npz"
    out = {
        "dataset": dataset.name,
        "fragment_cache_file": str(frag_path.relative_to(FC.REPO_ROOT)),
        "fragment_cache_file_sha256": RC.sha256_file(frag_path),
        "expected_fragment_cache_file_sha256": FC.EXPECTED_FRAGMENT_CACHE["ogbg-mollipo"],
        "embedded_source_cache_sha256": dataset.cache_sha256,
        "expected_source_cache_sha256": manifest["source_cache_sha256"],
        "train_order_sha256": order_sha256,
        "expected_train_order_sha256_prefix": expected,
        "n_train": int(len(dataset.splits["train"].row_id)),
        "n_valid": int(len(dataset.splits["valid"].row_id)),
        "n_test": int(len(dataset.splits["test"].row_id)),
        "n_views_train": int(dataset.splits["train"].view_s.size),
        "max_views": int(np.max(np.diff(dataset.splits["train"].mol_ptr))),
        "scaler": dataset.scaler,
    }
    if out["fragment_cache_file_sha256"] != out["expected_fragment_cache_file_sha256"]:
        raise AssertionError("fragment cache file identity changed")
    if out["embedded_source_cache_sha256"] != out["expected_source_cache_sha256"]:
        raise AssertionError("embedded official-cache identity changed")
    if not order_sha256.startswith(expected):
        raise AssertionError(f"train order hash {order_sha256} does not start with {expected}")
    if (out["n_train"], out["n_valid"], out["n_test"]) != (3360, 420, 420):
        raise AssertionError("official split sizes changed")
    if out["max_views"] > int(FC.load_parent_config()["max_views_per_molecule"]):
        raise AssertionError("more views than the frozen cap")
    out["ok"] = True
    return out


def single_vs_batch(dataset, config, state, device: str, arm: str = "JOINT") -> dict[str, Any]:
    model = FM.build_model(dataset.dims, FM.model_config(config, arm), state).to(device)
    model.eval()
    split = dataset.splits["valid"]
    n_views = np.diff(split.mol_ptr)
    picks = np.flatnonzero(n_views > 0)[:8]
    batch_scores, _ = RT.molecule_scores(model, dataset, split, picks, device)
    singles = []
    for mol in picks.tolist():
        score, _ = RT.molecule_scores(model, dataset, split, np.asarray([mol]), device)
        singles.append(float(score[0]))
    diff = float(np.max(np.abs(np.asarray(singles) - np.asarray(batch_scores))))
    out = {"arm": arm, "n_molecules": int(picks.size), "max_abs_diff": diff, "tolerance": 1e-5}
    if diff > out["tolerance"]:
        raise AssertionError(f"single/batch mismatch {diff}")
    out["molecule_indices"] = picks.tolist()
    out["ok"] = True
    return out


def three_steps(dataset, config, state, order: np.ndarray, device: str, arm: str = "JOINT") -> dict[str, Any]:
    torch.manual_seed(int(config["seed"]))
    model = FM.build_model(dataset.dims, FM.model_config(config, arm), state).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["peak_lr"]),
        weight_decay=float(config["weight_decay"]),
        betas=tuple(float(b) for b in config["betas"]),
        eps=float(config["eps"]),
    )
    train = dataset.splits["train"]
    batch_size = int(config["batch_size"])
    steps_per_epoch = max(1, (len(train.row_id) + batch_size - 1) // batch_size)
    lr_at = RT.lr_schedule(config, steps_per_epoch, int(config["epochs"]))
    perm = order[0]
    losses = []
    w_before = model.w.detach().clone()
    model.train()
    for step in range(3):
        mol_idx = perm[step * batch_size:(step + 1) * batch_size]
        optimizer.param_groups[0]["lr"] = lr_at(step)
        optimizer.zero_grad(set_to_none=True)
        loss = RT.molecule_loss(model, dataset, train, mol_idx, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        optimizer.step()
        losses.append(float(loss.detach()))
    grad_w = float(torch.linalg.vector_norm((model.w.detach() - w_before)).item())
    out = {
        "arm": arm,
        "losses": losses,
        "all_finite": bool(np.all(np.isfinite(losses))),
        "w_delta_norm": grad_w,
        "w_norm": float(torch.linalg.vector_norm(model.w.detach()).item()),
        "device": device,
    }
    if not out["all_finite"]:
        raise AssertionError("non-finite smoke loss")
    if not grad_w > 0:
        raise AssertionError("w did not move on the first three steps")
    out["ok"] = True
    return out


def reproduce_epoch1(dataset, config, state, order: np.ndarray, device: str,
                     parent_curve: Path) -> dict[str, Any]:
    """Run the copied loop as RESP for one epoch and compare to committed S0 numbers."""
    import fu_train as FT

    messages: list[str] = []
    result = FT.train_arm(
        dataset=dataset,
        arm="RESP",
        config=config,
        device=device,
        init_state=state,
        order=order,
        epochs=1,
        log=messages.append,
        eval_batch_size=int(config["eval_batch_size"]),
    )
    row = result["curves"][0]
    reference = _curve_row(parent_curve, 1)
    keys = ("train_loss", "train_rmse_raw", "valid_rmse_raw", "test_rmse_raw",
            "train_mae_raw", "valid_mae_raw", "test_mae_raw", "lr")
    diffs = {key: float(abs(float(row[key]) - float(reference[key]))) for key in keys}
    out = {
        "reference": "parent results/lipo_RESP/curve.csv epoch 1 (commit 185db246)",
        "reproduced": {key: float(row[key]) for key in keys},
        "reference_values": {key: float(reference[key]) for key in keys},
        "abs_diff": diffs,
        "tolerance": 1e-5,
        "device": device,
    }
    if max(diffs.values()) > out["tolerance"]:
        raise AssertionError(f"loop reproduction outside tolerance: {diffs}")
    out["ok"] = True
    return out


def _curve_row(path: Path, epoch: int) -> dict[str, Any]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["epoch"]) == int(epoch):
                return row
    raise KeyError(f"epoch {epoch} not in {path}")


def run_preflight(dataset, config, state, order, order_sha256: str, device: str,
                  parent_curve: Path, log=print) -> dict[str, Any]:
    out: dict[str, Any] = {"device": device}
    steps = (
        ("parameters", lambda: parameters(dataset, config, state)),
        ("w0_predictions", lambda: w0_predictions(dataset, config, state, device)),
        ("views_and_order", lambda: views_and_order(dataset, order, order_sha256)),
        ("single_vs_batch", lambda: single_vs_batch(dataset, config, state, device)),
        ("three_steps", lambda: three_steps(dataset, config, state, order, device)),
        ("reproduce_epoch1", lambda: reproduce_epoch1(dataset, config, state, order, device, parent_curve)),
    )
    for name, fn in steps:
        log(f"[preflight] {name} ...")
        out[name] = fn()
        log(f"[preflight] {name} ok")
    out["ok"] = True
    return out
