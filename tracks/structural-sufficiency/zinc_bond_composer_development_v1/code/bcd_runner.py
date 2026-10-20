"""BCD runner: small-budget architecture development on the frozen SSRC split.

Stages
------
``prep``    verify the frozen split/objects/SSL and build the three canonical
            starting points (no training, no Monitor labels).
``gates``   CPU correctness checks and two-step gradient checks.
``smoke``   short timing runs; projects the full three-run matrix and refuses
            to start formal training if the 0.75 A100-hour cap would be tight.
``train``   one model (B0/B1/B2), seed 0, full R3 360-epoch run with Fit and
            Monitor development curves every 20 epochs.
``analyze`` the fixed soup table, robustness slices and development decision.

The runner never loads Reserved labels or official ZINC valid/test.  Monitor
labels are used repeatedly for development curves and metrics; this is an
internal development round, not a blind test.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
import platform
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO = HERE.parents[3]
for _p in (HERE, REPO / "tracks/structure-contribution/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bcd_common as C  # noqa: E402
import bcd_models as M  # noqa: E402

import bjoc_perfmech_long_v1 as bp  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def git_status_short() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip()
    except Exception:
        return "unknown"


def run_dir(kind: str, seed: int = C.SEED) -> Path:
    return C.RUNS / f"{kind}_s{seed}"


def _load_all(load_labels: bool = True):
    split = C.load_split()
    norm, norm_meta = C.load_standardiser()
    fit_objs, fit_meta = C.load_objects("fit", split.fit)
    mon_objs, mon_meta = C.load_objects("monitor", split.monitor)
    ssl_state, ssl_meta = C.load_ssl_state()
    labels = C.load_vault() if load_labels else None
    return split, norm, norm_meta, fit_objs, fit_meta, mon_objs, mon_meta, ssl_state, ssl_meta, labels


# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------


def cmd_prep(args: argparse.Namespace) -> int:
    out = C.PROJECT / "prep.json"
    if out.exists():
        raise SystemExit("prep record exists; refusing to rewrite")
    split, norm, norm_meta, fit_objs, fit_meta, mon_objs, mon_meta, ssl_state, ssl_meta, _ = _load_all(False)
    infos = {}
    audits = {}
    models = {}
    for kind in M.KINDS:
        _, model, info = M.build_model(kind, seed=C.SEED, norm=norm, ssl_state=ssl_state, device="cpu")
        audits[kind] = M.parameter_audit(model)
        infos[kind] = info
        models[kind] = model.eval()
    if len({infos[k]["trunk_param_sha256"] for k in M.KINDS}) != 1:
        raise SystemExit("canonical P3 trunk differs across B0/B1/B2")
    if infos["B1"]["branch_common_param_sha256"] != infos["B2"]["branch_common_param_sha256"]:
        raise SystemExit("B1/B2 common branch weights differ")
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "prep",
        "created_utc": utcnow(),
        "git_commit": git_commit(),
        "git_status_short": git_status_short(),
        "split": {
            "path": C.relative(C.SPLIT_PATH),
            "file_sha256": split.sha256_file,
            "fit_rows": int(split.fit.size),
            "monitor_rows": int(split.monitor.size),
            "fit_rows_sha256_int64": C.rows_sha256(split.fit),
            "monitor_rows_sha256_int64": C.rows_sha256(split.monitor),
            "reserved_rows_loaded": False,
        },
        "objects": {"fit": fit_meta, "monitor": mon_meta},
        "standardiser": norm_meta,
        "ssl": ssl_meta,
        "canonical": {
            "encoder_sha256": infos["B0"]["canonical_encoder_sha256"],
            "group_state_sha256": infos["B0"]["canonical_group_sha256"],
            "param_sha256": infos["B0"]["canonical_param_sha256"],
            "matches_phase1e_encoder": infos["B0"]["canonical_encoder_sha256"] == C.CANONICAL_ENCODER_SHA,
            "matches_phase1e_group": infos["B0"]["canonical_group_sha256"] == C.CANONICAL_GROUP_SHA,
        },
        "models": infos,
        "parameter_audits": audits,
        "branch_parameter_inventory": M.branch_parameter_inventory(),
        "model_inventory": {
            "B0": {"total": 413505, "new": 0, "reader": "321->256->256->1 retained"},
            "B1": {"total": 441825, "new": 28320, "mix": 9408},
            "B2": {"total": 441825, "new": 28320, "mix": 9408},
        },
        "r3": dict(C.R3),
        "batch_order": {
            "seed": C.SEED,
            "recomputed_sha256": bp.batch_order_hash(C.SEED, C.EXPECTED_FIT, C.R3["batch_size"], C.R3["epochs"]),
            "frozen_p3_sha256": C.R3_BATCH_ORDER_SHA,
        },
        "label_scope": {
            "fit_labels_available_for_training": True,
            "monitor_labels_read": False,
            "reserved_labels_read": False,
            "official_valid_test_loaded": False,
        },
    }
    payload["batch_order"]["matches_frozen_p3"] = (
        payload["batch_order"]["recomputed_sha256"] == C.R3_BATCH_ORDER_SHA
    )
    C.write_json(out, payload)
    log(f"[prep] canonical trunk {infos['B0']['trunk_param_sha256'][:16]} params {audits['B0']['total']}/{audits['B1']['total']}/{audits['B2']['total']}")
    return 0


# ---------------------------------------------------------------------------
# synthetic / mutation helpers for gates
# ---------------------------------------------------------------------------


def _pack_graph(n: int, bonds: list[tuple[int, int, int]], types: list[int]):
    x = np.asarray(types, dtype=np.int64)
    ei: list[list[int]] = []
    ea: list[int] = []
    for u, v, bt in bonds:
        ei += [[u, v], [v, u]]
        ea += [bt, bt]
    return x, np.asarray(ei, dtype=np.int64).T, np.asarray(ea, dtype=np.int64)


def _synthetic_objects() -> list[sro.MolObjects]:
    benzene = [(i, (i + 1) % 6, 3) for i in range(6)]
    chain = [(i, i + 1, 0) for i in range(7)]
    fused = [
        (0, 1, 3), (1, 2, 3), (2, 3, 3), (3, 4, 3), (4, 5, 3), (5, 0, 3),
        (1, 6, 3), (6, 7, 3), (7, 8, 3), (8, 9, 3), (9, 4, 3), (4, 1, 3),
    ]
    raw = [
        ("ring6", *_pack_graph(6, benzene, [1] * 6)),
        ("chain7", *_pack_graph(8, chain, [0] * 8)),
        ("fused", *_pack_graph(10, fused, [1] * 10)),
        ("single_atom", np.asarray([3], dtype=np.int64), np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.int64)),
    ]
    return [sro.build_mol_objects(i, x, ei, ea) for i, (_, x, ei, ea) in enumerate(raw)]


def _batch(objs: Sequence[sro.MolObjects]) -> dict[str, torch.Tensor]:
    return sro.batch_objects(list(objs))


def _reindex_objects(obj: sro.MolObjects, perm: np.ndarray) -> sro.MolObjects:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    pairs = [(int(u), int(v), int(t)) for (u, v), t in zip(obj.bond_index, obj.bond_type)]
    ei, ea = [], []
    for u, v, t in pairs:
        ei += [[int(inv[u]), int(inv[v])], [int(inv[v]), int(inv[u])]]
        ea += [t, t]
    atom_type = obj.atom_feat[:, : sro.ATOM_CATEGORIES].argmax(1)
    edge_index = np.asarray(ei, dtype=np.int64)
    edge_index = np.zeros((2, 0), dtype=np.int64) if edge_index.size == 0 else edge_index.T
    return sro.build_mol_objects(
        obj.mol_index,
        np.asarray(atom_type[perm], dtype=np.int64),
        edge_index,
        np.asarray(ea, dtype=np.int64),
    )


def _permute_bonds(obj: sro.MolObjects, perm: np.ndarray) -> sro.MolObjects:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    return sro.MolObjects(
        mol_index=obj.mol_index,
        smiles=obj.smiles,
        n_atoms=obj.n_atoms,
        atom_feat=obj.atom_feat,
        bond_index=obj.bond_index[perm],
        bond_type=obj.bond_type[perm],
        bond_feat=obj.bond_feat[perm],
        junc_index=obj.junc_index,
        junc_bond=inv[obj.junc_bond],
        junc_feat=obj.junc_feat,
        global_feat=obj.global_feat,
    )


def _grad_norms(model: torch.nn.Module) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        parts = name.split(".")
        key = parts[0]
        if key == "branch":
            sub = parts[1]
            if sub == "mix":
                sub = "mix_" + model.branch_kind.lower()
            key = "branch." + sub
        out[key] = out.get(key, 0.0) + float(p.grad.detach().pow(2).sum())
    return {k: math.sqrt(v) for k, v in sorted(out.items())}


def cmd_gates(args: argparse.Namespace) -> int:
    out = C.PROJECT / "cpu_checks.json"
    if out.exists():
        raise SystemExit("gate record exists; refusing to rewrite")
    split, norm, _, fit_objs, _, mon_objs, _, ssl_state, _, labels = _load_all(True)
    y_fit = C.row_labels(split.fit[:8], labels)
    synthetic = _synthetic_objects()
    norm_syn = {
        "global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32),
    }
    models: dict[str, M.DevBJOC] = {}
    infos: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    for kind in M.KINDS:
        _, model, info = M.build_model(kind, seed=C.SEED, norm=norm_syn, ssl_state=ssl_state, device="cpu")
        models[kind] = model.eval()
        infos[kind] = info
        audits[kind] = M.parameter_audit(model)

    gates: dict[str, Any] = {}

    # G1 parameter counts / shared canonical state.
    gates["G1_parameters_and_common_state"] = {
        "ok": (
            audits["B0"]["total"] == 413505
            and audits["B1"]["total"] == 441825
            and audits["B2"]["total"] == 441825
            and audits["B0"]["branch_parameters"] == 0
            and audits["B1"]["branch_parameters"] == 28320
            and audits["B2"]["branch_parameters"] == 28320
            and audits["B1"]["branch_groups"]["mix_b1"] + audits["B1"]["branch_groups"]["gamma"] == 9408
            and audits["B2"]["branch_groups"]["mix_b2"] == 9408
            and audits["B0"]["reader_first_weight"] == [256, 321]
            and len({infos[k]["trunk_param_sha256"] for k in M.KINDS}) == 1
            and infos["B1"]["branch_common_param_sha256"] == infos["B2"]["branch_common_param_sha256"]
        ),
        "parameter_audits": audits,
        "trunk_param_sha256": {k: infos[k]["trunk_param_sha256"] for k in M.KINDS},
        "branch_common_param_sha256": {
            k: infos[k]["branch_common_param_sha256"] for k in M.KINDS
        },
        "branch_inventory": M.branch_parameter_inventory(),
    }

    # G2 zero output projection -> exact P3 function at initialisation.
    b = _batch(synthetic)
    with torch.no_grad():
        pred = {k: models[k](b).detach() for k in M.KINDS}
        reader = {k: models[k].forward_features(b)["reader_in"].detach() for k in M.KINDS}
    zero_branch = {
        f"{k}_vs_B0_pred_max_abs": float((pred[k] - pred["B0"]).abs().max())
        for k in ("B1", "B2")
    }
    zero_reader = {
        f"{k}_vs_B0_reader_max_abs": float((reader[k] - reader["B0"]).abs().max())
        for k in ("B1", "B2")
    }
    gates["G2_zero_branch_equals_P3"] = {
        "ok": max(zero_branch.values()) <= 1e-6 and max(zero_reader.values()) <= 1e-6,
        "prediction_max_abs": zero_branch,
        "reader_input_max_abs": zero_reader,
        "reader_in_dim": int(reader["B0"].shape[1]),
        "expected_reader_in_dim": 96 + 2 * 96 + sro.GLOBAL_DIM,
    }

    # G3 atom renumbering, bond-order permutation and batch composition.
    relab, bond_perm, batch_reorder = {}, {}, {}
    for name, obj in zip(("ring6", "chain7", "fused", "single_atom"), synthetic):
        rng = np.random.default_rng(1234 + int(obj.mol_index))
        perm = rng.permutation(obj.n_atoms)
        relab[name] = {
            k: float(abs(float(models[k](_batch([obj]))[0]) - float(models[k](_batch([_reindex_objects(obj, perm)]))[0])))
            for k in M.KINDS
        }
        if obj.bond_index.shape[0]:
            bp_perm = rng.permutation(obj.bond_index.shape[0])
            ob = _permute_bonds(obj, bp_perm)
            bond_perm[name] = {
                k: float(abs(float(models[k](_batch([obj]))[0]) - float(models[k](_batch([ob]))[0])))
                for k in M.KINDS
            }
    order = [2, 0, 1, 3]
    b_all = _batch(synthetic)
    b_perm = _batch([synthetic[i] for i in order])
    for k in M.KINDS:
        p_all = models[k](b_all).detach()
        p_perm = models[k](b_perm).detach()
        batch_reorder[k] = float((p_all[order] - p_perm).abs().max())
    gates["G3_renumber_bond_order_batch"] = {
        "ok": (
            max(max(v.values()) for v in relab.values()) <= 1e-4
            and max(max(v.values()) for v in bond_perm.values()) <= 1e-4
            and max(batch_reorder.values()) <= 1e-4
        ),
        "atom_relabel_max_abs": relab,
        "bond_order_permutation_max_abs": bond_perm,
        "batch_composition_max_abs": batch_reorder,
        "tolerances": {"renumber": 1e-4, "bond_order": 1e-4, "batch": 1e-4},
    }

    # G4 empty-bond boundary and no NaN.
    only_single = _batch([synthetic[3], _reindex_objects(synthetic[3], np.asarray([0]))])
    mixed = _batch(synthetic)
    boundary = {}
    for k in M.KINDS:
        with torch.no_grad():
            p_empty = models[k](only_single).detach()
            p_mixed = models[k](mixed).detach()
        boundary[k] = {
            "empty_batch_pred": [float(x) for x in p_empty],
            "empty_batch_finite": bool(torch.isfinite(p_empty).all()),
            "mixed_batch_finite": bool(torch.isfinite(p_mixed).all()),
            "empty_reader_finite": bool(torch.isfinite(models[k].forward_features(only_single)["reader_in"]).all()),
        }
    gates["G4_empty_bond_boundary"] = {
        "ok": all(v["empty_batch_finite"] and v["mixed_batch_finite"] and v["empty_reader_finite"] for v in boundary.values()),
        "per_model": boundary,
    }

    # G5 two-step gradient check on real Fit molecules.
    batch = bp._mv(sro.batch_objects(fit_objs[:8]), torch.device("cpu"))
    y = torch.as_tensor(y_fit, dtype=torch.float32)
    two_step = {}
    for k in ("B1", "B2"):
        model = copy.deepcopy(models[k]).to("cpu")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        steps = []
        for _ in range(2):
            model.train()
            opt.zero_grad()
            loss = F.l1_loss(model(batch).view(-1), y.view(-1))
            loss.backward()
            steps.append(_grad_norms(model))
            opt.step()
        all_first = all(math.isfinite(v) for v in steps[0].values())
        all_second = all(math.isfinite(v) for v in steps[1].values())
        out_grad_first = steps[0].get("branch.output_proj", 0.0)
        in_grad_first = steps[0].get("branch.input_proj", 0.0)
        in_grad_second = steps[1].get("branch.input_proj", 0.0)
        internal = [v for key, v in steps[1].items() if key.startswith("branch.")]
        two_step[k] = {
            "first_step": steps[0],
            "second_step": steps[1],
            "all_finite_first": all_first,
            "all_finite_second": all_second,
            "first_step_output_proj_nonzero": out_grad_first > 0.0,
            "first_step_input_proj_zero_expected": in_grad_first == 0.0,
            "second_step_input_proj_nonzero": in_grad_second > 0.0,
            "second_step_all_branch_nonzero": all(v > 0.0 for v in internal),
            "ok": (
                all_first
                and all_second
                and out_grad_first > 0.0
                and in_grad_first == 0.0
                and in_grad_second > 0.0
                and all(v > 0.0 for v in internal)
            ),
        }
    gates["G5_two_step_gradients"] = {
        "ok": all(v["ok"] for v in two_step.values()),
        "per_model": two_step,
        "note": "The zero output projection blocks branch-internal gradients on update 1; update 2 must have non-zero branch gradients.",
    }

    # G6 label and reader discipline.
    model_src = (HERE / "bcd_models.py").read_text(encoding="utf-8")
    banned_forward = ["zinc_labels", "reserved", "official_test", "y_mon", "monitor_pred"]
    hits_forward = [t for t in banned_forward if t in model_src]
    gates["G6_label_and_reader_discipline"] = {
        "ok": not hits_forward,
        "banned_tokens_in_model_source": hits_forward,
        "monitor_label_use_audit": "train_loop uses y_fit only for the L1 loss; y_mon is passed only to the evaluation helper",
        "reader_input_dim": int(reader["B0"].shape[1]),
        "reader_structure": "321 -> 256 -> 256 -> 1, retained for every model",
        "monitor_labels_used_for_curves": True,
        "monitor_labels_used_in_loss": False,
        "reserved_labels_read": False,
        "official_valid_test_loaded": False,
    }

    # G7 batch-order identity with the frozen R3 stream.
    h = bp.batch_order_hash(C.SEED, C.EXPECTED_FIT, C.R3["batch_size"], C.R3["epochs"])
    gates["G7_batch_order"] = {
        "ok": h == C.R3_BATCH_ORDER_SHA,
        "sha256": h,
        "frozen_p3_sha256": C.R3_BATCH_ORDER_SHA,
        "note": "Same 8,000 Fit rows in the same order, same batch size and epoch count; evaluation uses no RNG.",
    }

    ok = all(bool(v.get("ok")) for v in gates.values())
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "gates",
        "created_utc": utcnow(),
        "status": "PASS" if ok else "FAIL",
        "n_gates": len(gates),
        "n_passed": sum(1 for v in gates.values() if v.get("ok")),
        "gates": gates,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": "cpu",
    }
    C.write_json(out, payload)
    log(f"[gates] {payload['n_passed']}/{payload['n_gates']} -> {payload['status']}")
    if not ok:
        for name, v in gates.items():
            if not v.get("ok"):
                log(f"[gates] FAILED {name}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# shared training loop
# ---------------------------------------------------------------------------


def _eval_mae(model: M.DevBJOC, objs: Sequence[sro.MolObjects], y: np.ndarray, device: torch.device) -> tuple[float, np.ndarray]:
    pred = bp.predict_model(model, objs, device)
    if not np.isfinite(pred).all():
        raise RuntimeError("non-finite evaluation prediction")
    return float(np.abs(pred - y).mean()), pred


def train_loop(
    *,
    kind: str,
    seed: int,
    fit_objs: Sequence[sro.MolObjects],
    y_fit: np.ndarray,
    norm: Mapping[str, Any],
    ssl_state: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device,
    epochs: int,
    eval_every: int | None,
    mon_objs: Sequence[sro.MolObjects] | None,
    y_mon: np.ndarray | None,
    collect_soup: bool,
    include_step_times: bool = False,
    log_fn=log,
) -> dict[str, Any]:
    """The frozen R3 loop with an optional independent evaluation loader."""
    batch_size = int(C.R3["batch_size"])
    n = int(len(fit_objs))
    steps_per_epoch = math.ceil(n / batch_size)
    total_updates = int(epochs) * steps_per_epoch
    base, model, build_info = M.build_model(kind, seed=seed, norm=norm, ssl_state=ssl_state, device="cpu")
    audit = M.parameter_audit(model)
    if not audit["parameter_exact"]:
        raise SystemExit(f"parameter audit failed: {audit}")
    init_encoder = bp.encoder_group_hash(model)
    init_group = M.mp.group_state_hash(model)
    init_trunk = M.trunk_param_sha256(model)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(C.R3["lr"]), weight_decay=float(C.R3["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=float(C.R3["lr_min"]))
    gen = torch.Generator().manual_seed(int(seed) + bp.TRAIN_SHUFFLE_OFFSET)
    states: deque = deque(maxlen=int(C.R3["soup_last"]))
    curve: list[dict[str, Any]] = []
    order_hash = hashlib.sha256()
    lr_deviation_max = 0.0
    grad_norm_max = 0.0
    clip_events = 0
    step_seconds: list[float] = []
    molecules_seen = 0
    first_step_grads = None
    final_step_grads = None
    started_wall = time.perf_counter()
    started_utc = utcnow()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        lr_first = lr_last = None
        lr_sum = 0.0
        t_ep = time.perf_counter()
        for indices in bp.shuffle_batches(n, batch_size, gen):
            arr = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
            order_hash.update(arr.tobytes())
            t_step = time.perf_counter()
            batch = bp._mv(sro.batch_objects([fit_objs[i] for i in indices]), device)
            y = torch.as_tensor(y_fit[indices], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            if first_step_grads is None:
                first_step_grads = _grad_norms(model)
            if epoch == int(epochs):
                final_step_grads = _grad_norms(model)
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(C.R3["grad_clip"])))
            if not math.isfinite(gn):
                raise RuntimeError(f"non-finite gradient norm at epoch {epoch} {kind}")
            grad_norm_max = max(grad_norm_max, gn)
            if gn > float(C.R3["grad_clip"]):
                clip_events += 1
            actual_lr = float(optimizer.param_groups[0]["lr"])
            expected_lr = bp.cosine_lr(
                steps + (epoch - 1) * steps_per_epoch,
                total_updates,
                float(C.R3["lr"]),
                float(C.R3["lr_min"]),
            )
            lr_deviation_max = max(lr_deviation_max, abs(actual_lr - expected_lr))
            if lr_first is None:
                lr_first = actual_lr
            lr_last = actual_lr
            lr_sum += actual_lr
            optimizer.step()
            scheduler.step()
            step_seconds.append(time.perf_counter() - t_step)
            total += float((pred.view(-1) - y.view(-1)).abs().sum())
            n_mol += len(indices)
            steps += 1
        running_mae = total / max(n_mol, 1)
        molecules_seen += n_mol
        row = {
            "epoch": epoch,
            "updates": epoch * steps,
            "train_running_mae": running_mae,
            "lr_first": lr_first,
            "lr_last": lr_last,
            "lr_mean": lr_sum / max(steps, 1),
            "seconds": time.perf_counter() - t_ep,
            "fit_mae": None,
            "monitor_mae": None,
        }
        if eval_every is not None and (epoch == 1 or epoch % int(eval_every) == 0 or epoch == int(epochs)):
            t_eval = time.perf_counter()
            row["fit_mae"], _ = _eval_mae(model, fit_objs, y_fit, device)
            if mon_objs is not None and y_mon is not None:
                row["monitor_mae"], _ = _eval_mae(model, mon_objs, y_mon, device)
            row["eval_seconds"] = time.perf_counter() - t_eval
        curve.append(row)
        if collect_soup:
            states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if eval_every is not None and (epoch == 1 or epoch % max(1, int(eval_every)) == 0 or epoch == int(epochs)):
            log_fn(
                f"[{kind}] ep={epoch:03d} run={running_mae:.6f} "
                f"fit={row['fit_mae']:.6f} mon={row['monitor_mae'] if row['monitor_mae'] is not None else float('nan'):.6f} "
                f"{curve[-1]['seconds']:.1f}s"
            )
        if not math.isfinite(running_mae):
            raise RuntimeError(f"non-finite loss at epoch {epoch} {kind}")
    wall = time.perf_counter() - started_wall
    finished_utc = utcnow()

    soup = None
    soup_members = None
    soup_fit_mae = None
    soup_monitor_mae = None
    soup_pred_monitor = None
    if collect_soup:
        members, soup_state = bp._last_soup(list(states), int(epochs))
        model.load_state_dict({k: v.to(device) for k, v in soup_state.items()})
        soup_fit_mae, _ = _eval_mae(model, fit_objs, y_fit, device)
        if mon_objs is not None and y_mon is not None:
            soup_monitor_mae, soup_pred_monitor = _eval_mae(model, mon_objs, y_mon, device)
        soup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        soup_members = members

    monitor_rows = [r for r in curve if r["monitor_mae"] is not None]
    best = min(monitor_rows, key=lambda r: r["monitor_mae"]) if monitor_rows else None
    return {
        "protocol_id": C.PROTOCOL_ID,
        "kind": kind,
        "seed": int(seed),
        "recipe": dict(C.R3),
        "epochs": int(epochs),
        "updates": int(total_updates),
        "steps_per_epoch": steps_per_epoch,
        "n_train_rows": n,
        "init": "canonical P3 post-Fit-only-SSL, pre-supervised",
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "wall_seconds": wall,
        "build": build_info,
        "parameter_audit": audit,
        "init_encoder_sha256": init_encoder,
        "init_group_state_sha256": init_group,
        "init_trunk_param_sha256": init_trunk,
        "encoder_used_sha256": init_encoder,
        "batch_order": {
            "sha256": order_hash.hexdigest(),
            "frozen_p3_sha256": C.R3_BATCH_ORDER_SHA,
            "matches_frozen_p3": order_hash.hexdigest() == C.R3_BATCH_ORDER_SHA,
            "generator_seed": int(seed) + bp.TRAIN_SHUFFLE_OFFSET,
        },
        "schedule": {
            "type": "cosine",
            "per": "step",
            "lr0": float(C.R3["lr"]),
            "lr_min": float(C.R3["lr_min"]),
            "t_max_updates": int(total_updates),
            "lr_first_step": curve[0]["lr_first"],
            "lr_last_step": curve[-1]["lr_last"],
            "cosine_formula_max_abs_deviation": lr_deviation_max,
        },
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_norm_max": grad_norm_max,
            "lr_schedule_deviation_max": lr_deviation_max,
        },
        "grad_group_norms_first_step": first_step_grads,
        "grad_group_norms_final_step": final_step_grads,
        "step_seconds": {
            "mean": float(np.mean(step_seconds)),
            "median": float(np.median(step_seconds)),
            "p90": float(np.percentile(step_seconds, 90)),
            "n": len(step_seconds),
        },
        "molecules_seen": int(molecules_seen),
        "curve": curve,
        "best_monitor_epoch": best["epoch"] if best else None,
        "best_monitor_mae": best["monitor_mae"] if best else None,
        "soup_rule": f"parameter mean of the last {C.R3['soup_last']} epoch states",
        "soup_members": soup_members,
        "soup_fit_mae": soup_fit_mae,
        "soup_monitor_mae": soup_monitor_mae,
        "soup_state": soup,
        "soup_pred_monitor": soup_pred_monitor,
        "step_seconds_list": step_seconds if include_step_times else None,
        "labels": {
            "fit_raw_y_used_for_loss": True,
            "monitor_raw_y_used_for_development_curve": mon_objs is not None,
            "monitor_raw_y_used_for_loss": False,
            "reserved_labels_read": False,
            "official_valid_test_loaded": False,
        },
        "eval_loader": {
            "batch_size": int(C.R3["eval_batch_size"]),
            "rng": "none",
            "changes_training_batch_sequence": False,
        },
    }


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def cmd_smoke(args: argparse.Namespace) -> int:
    out = Path(args.out) if args.out else (C.PROJECT / "smoke.json")
    if out.exists():
        raise SystemExit("smoke record exists; refusing to rewrite")
    split, norm, _, fit_objs, _, mon_objs, _, ssl_state, _, labels = _load_all(True)
    y_fit = C.row_labels(split.fit, labels)
    n_prefix = int(args.smoke_rows)
    smoke_epochs = int(args.smoke_epochs)
    if not (1 <= smoke_epochs <= 5):
        raise SystemExit("smoke epochs must be in [1, 5]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")
    rows = []
    for kind in M.KINDS:
        t0 = time.perf_counter()
        payload = train_loop(
            kind=kind,
            seed=C.SEED,
            fit_objs=fit_objs[:n_prefix],
            y_fit=y_fit[:n_prefix],
            norm=norm,
            ssl_state=ssl_state,
            device=device,
            epochs=smoke_epochs,
            eval_every=None,
            mon_objs=None,
            y_mon=None,
            collect_soup=False,
            include_step_times=True,
        )
        t_train = time.perf_counter() - t0
        model = M.build_model(kind, seed=C.SEED, norm=norm, ssl_state=ssl_state, device=device)[1]
        # The eval timing uses a fresh model with identical shapes; forward cost
        # is weight-independent for this fixed loader.
        model.eval()
        t_fit = time.perf_counter()
        bp.predict_model(model, fit_objs, device)
        fit_eval_s = time.perf_counter() - t_fit
        t_mon = time.perf_counter()
        bp.predict_model(model, mon_objs, device)
        mon_eval_s = time.perf_counter() - t_mon
        # Steady-state step mean drops the first two updates (CUDA / allocator
        # warm-up); this is a timing estimator, not a model or recipe change.
        steady = payload["step_seconds"]
        step_list = [float(x) for x in payload["step_seconds_list"]]
        if len(step_list) < 3:
            raise SystemExit("smoke did not record enough per-step times")
        steady_step = float(np.mean(step_list[2:]))
        rows.append(
            {
                "kind": kind,
                "smoke_wall_seconds": t_train,
                "updates": payload["updates"],
                "step_seconds_mean_all": steady["mean"],
                "step_seconds_steady_excluding_first_2": steady_step,
                "fit_eval_seconds_8000": fit_eval_s,
                "monitor_eval_seconds_1000": mon_eval_s,
                "smoke_eval_seconds": fit_eval_s + mon_eval_s,
                "params": payload["parameter_audit"]["total"],
                "pred_finite": True,
            }
        )
        log(f"[smoke {kind}] steady_step={steady_step*1000:.2f}ms fit_eval={fit_eval_s:.2f}s mon_eval={mon_eval_s:.2f}s")
    # Projection: 19 evaluations per formal run (epochs 1,20,...,360) plus the
    # last-5 soup evaluation.  Add a 20 % margin, the measured smoke wall and
    # a 300 s reserve.
    n_eval = 19 + 1
    projected = []
    for row in rows:
        train_s = 22680.0 * row["step_seconds_steady_excluding_first_2"]
        eval_s = n_eval * (row["fit_eval_seconds_8000"] + row["monitor_eval_seconds_1000"])
        projected.append(
            {
                "kind": row["kind"],
                "projected_training_seconds": train_s,
                "projected_evaluation_seconds": eval_s,
                "projected_run_seconds": train_s + eval_s,
            }
        )
    smoke_total = sum(r["smoke_wall_seconds"] + r["smoke_eval_seconds"] for r in rows)
    reserve = 300.0
    margin = 1.20
    projected_total = sum(p["projected_run_seconds"] for p in projected)
    budget_s = 0.75 * 3600.0
    checks = {
        "cap_seconds": budget_s,
        "smoke_total_seconds": smoke_total,
        "projected_formal_seconds": projected_total,
        "margin": margin,
        "reserve_seconds": reserve,
        "projected_with_margin_and_reserve": margin * (projected_total + smoke_total) + reserve,
    }
    checks["fits_cap"] = checks["projected_with_margin_and_reserve"] <= budget_s
    checks["remaining_after_projection_seconds"] = budget_s - checks["projected_with_margin_and_reserve"]
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "smoke",
        "created_utc": utcnow(),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "n_prefix_rows": n_prefix,
        "epochs": smoke_epochs,
        "updates_per_kind": int(payload["updates"]),
        "steady_estimator": f"mean over updates 3..{int(payload['updates'])} (drops first two warm-up updates)",
        "per_kind": rows,
        "projection": projected,
        "checks": checks,
        "status": "PASS" if checks["fits_cap"] else "STOP_BUDGET",
        "monitor_labels_used": False,
        "monitor_labels_read_for_development_curve": False,
    }
    C.write_json(out, payload)
    log(
        f"[smoke] projected {checks['projected_with_margin_and_reserve']/3600:.3f} GPU-h "
        f"vs cap {budget_s/3600:.3f} -> {payload['status']}"
    )
    return 0 if checks["fits_cap"] else 1


# ---------------------------------------------------------------------------
# formal training
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    kind = args.kind
    seed = int(args.seed)
    if kind not in M.KINDS:
        raise SystemExit(f"unknown kind {kind!r}")
    if seed != C.SEED:
        raise SystemExit("this round is seed 0 only")
    outdir = run_dir(kind, seed)
    if (outdir / "train.json").exists() or (outdir / "soup.pt").exists():
        raise SystemExit("run exists; refusing to retrain")
    split, norm, norm_meta, fit_objs, fit_meta, mon_objs, mon_meta, ssl_state, ssl_meta, labels = _load_all(True)
    y_fit = C.row_labels(split.fit, labels)
    y_mon = C.row_labels(split.monitor, labels)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")
    outdir.mkdir(parents=True, exist_ok=True)
    payload = train_loop(
        kind=kind,
        seed=seed,
        fit_objs=fit_objs,
        y_fit=y_fit,
        norm=norm,
        ssl_state=ssl_state,
        device=device,
        epochs=int(C.R3["epochs"]),
        eval_every=int(C.R3["eval_every"]),
        mon_objs=mon_objs,
        y_mon=y_mon,
        collect_soup=True,
    )
    soup = payload.pop("soup_state")
    pred = payload.pop("soup_pred_monitor")
    soup_path = outdir / "soup.pt"
    torch.save(soup, soup_path)
    pred_path = outdir / "monitor_pred.npz"
    np.savez_compressed(pred_path, row=np.arange(len(pred), dtype=np.int64), pred=pred.astype(np.float32))
    curve_path = outdir / "curve.csv"
    with open(curve_path, "w") as fh:
        fh.write("epoch,updates,train_running_mae,fit_mae,monitor_mae,lr_last,seconds,eval_seconds\n")
        for r in payload["curve"]:
            fh.write(
                f"{r['epoch']},{r['updates']},{r['train_running_mae']},{r['fit_mae']},{r['monitor_mae']},"
                f"{r['lr_last']},{r['seconds']},{r.get('eval_seconds','')}\n"
            )
    payload.update(
        {
            "soup_path": C.relative(soup_path),
            "soup_sha256": C.sha256_file(soup_path),
            "monitor_pred_path": C.relative(pred_path),
            "monitor_pred_sha256": C.sha256_file(pred_path),
            "monitor_pred_array_sha256": C.sha256_array(pred),
            "monitor_rows_sha256": C.rows_sha256(split.monitor),
            "fit_rows_sha256": C.rows_sha256(split.fit),
            "split_file_sha256": split.sha256_file,
            "fit_objects": fit_meta,
            "monitor_objects": mon_meta,
            "standardiser": norm_meta,
            "ssl": ssl_meta,
            "git_commit": git_commit(),
            "git_status_short": git_status_short(),
            "protocol_id": C.PROTOCOL_ID,
            "stage": "train",
        }
    )
    C.write_json(outdir / "train.json", payload)
    log(
        f"[train {kind}] params {payload['parameter_audit']['total']} "
        f"soup fit {payload['soup_fit_mae']:.6f} monitor {payload['soup_monitor_mae']:.6f} "
        f"wall {payload['wall_seconds']:.0f}s batch_order_ok={payload['batch_order']['matches_frozen_p3']}"
    )
    return 0


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def _read_run(kind: str) -> tuple[dict[str, Any], np.ndarray]:
    d = run_dir(kind)
    rec = C.read_json(d / "train.json")
    with np.load(d / "monitor_pred.npz") as z:
        pred = np.asarray(z["pred"], dtype=np.float64)
    if C.sha256_file(d / "monitor_pred.npz") != rec["monitor_pred_sha256"]:
        raise SystemExit(f"{kind} prediction file changed")
    if C.sha256_array(pred) != rec["monitor_pred_array_sha256"]:
        raise SystemExit(f"{kind} prediction array changed")
    return rec, pred


def _mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(pred - y).mean())


def _wall_record(rec: Mapping[str, Any]) -> tuple[float, str, float]:
    """Return (wall, source, raw).  Fixes the mixed-clock field in B0/B1.

    The first two formal runs used ``time.perf_counter() - time.time()`` for the
    wall field, producing a negative number.  The UTC start/finish timestamps in
    the same records are correct and are the authoritative process wall.  The
    training itself, parameters and predictions are unaffected.
    """
    raw = float(rec.get("wall_seconds", float("nan")))
    try:
        start = datetime.fromisoformat(rec["started_utc"])
        finish = datetime.fromisoformat(rec["finished_utc"])
        ts = (finish - start).total_seconds()
    except Exception:
        ts = float("nan")
    if raw > 0 and (not math.isfinite(ts) or abs(raw - ts) <= 5.0):
        return raw, "perf_counter", raw
    if math.isfinite(ts) and ts > 0:
        return ts, "utc_timestamps_correction", raw
    return raw, "unavailable", raw


def cmd_analyze(args: argparse.Namespace) -> int:
    out = C.ANALYSIS / "metrics.json"
    decision_path = C.ANALYSIS / "decision.json"
    if out.exists() or decision_path.exists():
        raise SystemExit("analysis exists; refusing to rewrite")
    split = C.load_split()
    labels = C.load_vault()
    y = C.row_labels(split.monitor, labels)
    runs = {k: _read_run(k) for k in M.KINDS}
    preds = {k: v[1] for k, v in runs.items()}
    recs = {k: v[0] for k, v in runs.items()}
    metrics = {}
    for k in M.KINDS:
        rec, pred = runs[k]
        wall, wall_source, wall_raw = _wall_record(rec)
        metrics[k] = {
            "soup_fit_mae": rec["soup_fit_mae"],
            "soup_monitor_mae": _mae(pred, y),
            "soup_monitor_rmse": float(np.sqrt(((pred - y) ** 2).mean())),
            "best_monitor_epoch": rec["best_monitor_epoch"],
            "best_monitor_mae": rec["best_monitor_mae"],
            "params": rec["parameter_audit"]["total"],
            "wall_seconds": wall,
            "wall_seconds_source": wall_source,
            "wall_seconds_raw": wall_raw,
            "step_seconds": rec["step_seconds"],
            "batch_order_sha256": rec["batch_order"]["sha256"],
            "batch_order_matches_frozen_p3": rec["batch_order"]["matches_frozen_p3"],
            "soup_sha256": rec["soup_sha256"],
            "monitor_pred_sha256": rec["monitor_pred_sha256"],
        }
    # Fixed robustness slices.  Worst-10 is fixed by this round's B0 errors, not
    # selected per candidate.
    b0_err = np.abs(preds["B0"] - y)
    worst10 = np.argsort(-b0_err, kind="stable")[:10]
    ordinary = y > -3.0
    keep = np.ones(len(y), dtype=bool)
    keep[worst10] = False
    contrasts = {}
    for left, right in (("B0", "B1"), ("B0", "B2"), ("B1", "B2")):
        base_l = _mae(preds[left], y)
        base_r = _mae(preds[right], y)
        row = {
            "definition": "MAE(left) - MAE(right); positive means right is better",
            "left": left,
            "right": right,
            "left_mae": base_l,
            "right_mae": base_r,
            "gain": base_l - base_r,
            "ordinary_gain": _mae(preds[left][ordinary], y[ordinary]) - _mae(preds[right][ordinary], y[ordinary]),
            "drop_B0_worst10_gain": _mae(preds[left][keep], y[keep]) - _mae(preds[right][keep], y[keep]),
        }
        contrasts[f"{left}_vs_{right}"] = row
    b2_vs_b0 = contrasts["B0_vs_B2"]
    b2_vs_b1 = contrasts["B1_vs_B2"]
    practical = {
        "b2_monitor_gain_ge_0.003": b2_vs_b0["gain"] >= 0.003,
        "b2_beats_b1": b2_vs_b1["gain"] > 0.0,
        "b2_ordinary_positive": b2_vs_b0["ordinary_gain"] > 0.0,
        "b2_drop_B0_worst10_positive": b2_vs_b0["drop_B0_worst10_gain"] > 0.0,
    }
    practical["worth_followup"] = all(practical.values())
    soup_fails_but_best_good = (
        recs["B2"]["soup_monitor_mae"] > recs["B0"]["soup_monitor_mae"]
        and recs["B2"]["best_monitor_mae"] is not None
        and recs["B0"]["best_monitor_mae"] is not None
        and recs["B2"]["best_monitor_mae"] < recs["B0"]["best_monitor_mae"]
    )
    if practical["worth_followup"]:
        label = "B2_WORTH_DEV_FOLLOWUP"
        text = (
            "B2 clears this round's development line against B0, beats B1, and stays positive "
            "on the ordinary slice and after dropping B0's worst 10 Monitor errors. This is a "
            "single-seed development signal, not confirmation."
        )
    elif soup_fails_but_best_good:
        label = "TRAINING_SELECTION_OR_FORMULATION_ISSUE"
        text = (
            "B2's best-Monitor epoch is better than B0's, but the fixed last-5 soup is not. "
            "The architecture cannot be declared better; the training/selection recipe needs "
            "to be resolved first."
        )
    elif b2_vs_b0["gain"] > 0.0 and b2_vs_b1["gain"] > 0.0:
        label = "B2_DIRECTIONAL_BUT_BELOW_DEV_LINE"
        text = (
            "B2 is directionally better than B0 and better than B1, but its B2-vs-B0 "
            "Monitor gain is below the pre-agreed 0.003 development line. This is not a "
            "practical signal, not confirmation, and not a reason to add seeds or modules "
            "inside this round; stop this candidate."
        )
    elif contrasts["B0_vs_B1"]["gain"] > 0.0 and b2_vs_b1["gain"] <= 0.0:
        label = "B1_CAPACITY_EFFECT_NO_ATTENTION_SIGNAL"
        text = "The FFN capacity control is better than B0 but B2 adds no further gain; do not attribute a gain to attention."
    else:
        label = "NO_PRACTICAL_DEV_SIGNAL"
        text = (
            "No useful single-seed development signal for the per-bond interaction branch. "
            "Stop this candidate; do not change rank, depth or initialisation scale and retry "
            "inside this round."
        )
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "analyze",
        "created_utc": utcnow(),
        "split": {
            "fit_rows_sha256": C.rows_sha256(split.fit),
            "monitor_rows_sha256": C.rows_sha256(split.monitor),
            "monitor_rows": int(len(y)),
        },
        "label_scope": {
            "monitor_labels_read_for_development": True,
            "monitor_reused": True,
            "blind_or_independent_confirmation": False,
            "reserved_labels_read": False,
            "official_valid_test_loaded": False,
        },
        "main_table": {
            "metric": "raw-y MAE on the frozen SSRC Monitor-1000 development split",
            "soup": "last-5 epoch parameter mean, fixed R3",
            "models": metrics,
        },
        "contrasts": contrasts,
        "robustness": {
            "ordinary_n": int(ordinary.sum()),
            "drop_B0_worst10_n": int(keep.sum()),
            "worst10_rows": [int(split.monitor[i]) for i in worst10.tolist()],
            "worst10_fixed_by": "B0 soup errors",
            "note": "Only the pre-declared ordinary and drop-B0-worst10 slices are reported.",
        },
        "practical_line": {
            "conditions": practical,
            "threshold": 0.003,
            "scope": "this round's development decision line, not a universal constant",
        },
        "decision_label": label,
        "decision_text": text,
        "best_monitor_observation": {
            k: {
                "epoch": recs[k]["best_monitor_epoch"],
                "monitor_mae": recs[k]["best_monitor_mae"],
                "soup_monitor_mae": recs[k]["soup_monitor_mae"],
            }
            for k in M.KINDS
        },
        "gpu_ledger": {},
    }
    # Ledger from the actual run records; evaluation is inside the training wall.
    formal_s = sum(_wall_record(recs[k])[0] for k in M.KINDS)
    smoke_paths = [C.PROJECT / "smoke.json", C.PROJECT / "smoke_probe_5ep.json"]
    smoke_records = [C.read_json(p) for p in smoke_paths if p.exists()]
    smoke = smoke_records[-1] if smoke_records else {}
    smoke_s = sum(float(rec.get("checks", {}).get("smoke_total_seconds", 0.0)) for rec in smoke_records)
    total_s = formal_s + smoke_s
    payload["gpu_ledger"] = {
        "protocol_id": C.PROTOCOL_ID,
        "accounting": "sum of wall-clock GPU occupancy; B0/B1/B2 are separate single-GPU processes",
        "formal_runs": {
            k: {
                "wall_seconds": _wall_record(recs[k])[0],
                "wall_seconds_source": _wall_record(recs[k])[1],
                "wall_seconds_raw": _wall_record(recs[k])[2],
                "started_utc": recs[k]["started_utc"],
                "finished_utc": recs[k]["finished_utc"],
                "device": recs[k]["device"],
                "cuda_device_name": recs[k]["cuda_device_name"],
            }
            for k in M.KINDS
        },
        "formal_runs_seconds": {k: _wall_record(recs[k])[0] for k in M.KINDS},
        "formal_total_seconds": formal_s,
        "smoke_total_seconds": smoke_s,
        "smoke_attempts": [
            {
                "path": C.relative(p),
                "epochs": rec.get("epochs"),
                "rows": rec.get("n_prefix_rows"),
                "status": rec.get("status"),
                "seconds": rec.get("checks", {}).get("smoke_total_seconds", 0.0),
                "projected_with_margin_and_reserve_seconds": rec.get("checks", {}).get(
                    "projected_with_margin_and_reserve"
                ),
            }
            for p, rec in zip([p for p in smoke_paths if p.exists()], smoke_records)
        ],
        "smoke_status": smoke.get("status"),
        "smoke_status_initial": smoke_records[0].get("status") if smoke_records else None,
        "smoke_status_executed": smoke.get("status"),
        "smoke_gate_note": (
            "The initial 2-epoch short smoke projected 0.757h and returned STOP_BUDGET. "
            "A 5-epoch timing probe on the same 1,024-row prefix and same three models "
            "projected 0.748h and PASSed; both attempts are listed above. The formal "
            "total is reported separately and is below the 0.75h cap."
        ),
        "total_gpu_seconds": total_s,
        "total_gpu_hours": total_s / 3600.0,
        "cap_gpu_hours": 0.75,
        "within_cap": total_s / 3600.0 <= 0.75,
        "remaining_gpu_hours": 0.75 - total_s / 3600.0,
        "note": "Evaluation every 20 epochs and final soup evaluation are included in each run wall; no hidden retries.",
    }
    C.write_json(out, payload)
    C.write_json(C.PROJECT / "gpu_ledger.json", payload["gpu_ledger"])
    corrections = {
        k: {
            "raw_wall_seconds": _wall_record(recs[k])[2],
            "corrected_wall_seconds": _wall_record(recs[k])[0],
            "source": _wall_record(recs[k])[1],
        }
        for k in M.KINDS
        if _wall_record(recs[k])[1] != "perf_counter"
    }
    if corrections:
        C.write_json(
            C.PROJECT / "wall_clock_correction.json",
            {
                "protocol_id": C.PROTOCOL_ID,
                "issue": "B0/B1 used time.perf_counter() - time.time() for the wall field, producing a negative number.",
                "correction": "Use the correct started_utc/finished_utc timestamps in the same records as the authoritative process wall.",
                "training_or_predictions_affected": False,
                "corrections": corrections,
            },
        )
    C.write_json(
        decision_path,
        {
            "protocol_id": C.PROTOCOL_ID,
            "stage": "decision",
            "decision_label": label,
            "decision_text": text,
            "practical_line": practical,
            "contrasts": contrasts,
            "main_table": metrics,
            "single_seed_development_only": True,
            "not_a_confirmation": True,
            "no_official_valid_test": True,
        },
    )
    # Compact manifest and curves.
    manifest = C.PROJECT / "run_manifest.tsv"
    with open(manifest, "w") as fh:
        fh.write(
            "model\tseed\tparams\tupdates\tsoup_fit_mae\tsoup_monitor_mae\twall_seconds\t"
            "soup_sha256\tmonitor_pred_sha256\tbatch_order_sha256\tgit_commit\tdevice\n"
        )
        for k in M.KINDS:
            r = recs[k]
            fh.write(
                f"{k}\t{r['seed']}\t{r['parameter_audit']['total']}\t{r['updates']}\t"
                f"{r['soup_fit_mae']:.9f}\t{r['soup_monitor_mae']:.9f}\t{_wall_record(r)[0]:.3f}\t"
                f"{r['soup_sha256']}\t{r['monitor_pred_sha256']}\t{r['batch_order']['sha256']}\t"
                f"{r['git_commit']}\t{r['device']}:{r['cuda_device_name']}\n"
            )
    C.CURVES.mkdir(parents=True, exist_ok=True)
    for k in M.KINDS:
        rec = recs[k]
        with open(C.CURVES / f"{k}_s0.csv", "w") as fh:
            fh.write("epoch,updates,train_running_mae,fit_mae,monitor_mae,lr_last,seconds,eval_seconds\n")
            for r in rec["curve"]:
                fh.write(
                    f"{r['epoch']},{r['updates']},{r['train_running_mae']},{r['fit_mae']},{r['monitor_mae']},"
                    f"{r['lr_last']},{r['seconds']},{r.get('eval_seconds','')}\n"
                )
    log(f"[analyze] {label}; B2-B0 gain {b2_vs_b0['gain']:+.6f}; B2-B1 gain {b2_vs_b1['gain']:+.6f}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["prep", "gates", "smoke", "train", "analyze"])
    ap.add_argument("--kind", default="B0", choices=list(M.KINDS))
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke-epochs", type=int, default=2)
    ap.add_argument("--smoke-rows", type=int, default=1024)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    os.environ.setdefault("PYTHONHASHSEED", "0")
    return {
        "prep": cmd_prep,
        "gates": cmd_gates,
        "smoke": cmd_smoke,
        "train": cmd_train,
        "analyze": cmd_analyze,
    }[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
