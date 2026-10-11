"""cscl-strong-nongnn-predictor-v1 driver: prep / cpu-gates / replay / train / analyze.

Protocol: ``configs/strong_nongnn_predictor_v1.yaml`` (frozen).

The one authorized new model is ``FullY-StaticTriplet-seed0``: the exact
historical Full-Y computation plus a single non-iterative static triplet
operator (centre + two real-bond neighbours) whose per-graph
``[sum q ; sum q^2]`` summary is appended to the original reader input.  All
old parameters keep their historical seed-0 values; the 64 new reader columns
are zero-initialised, so the initial function equals the frozen Full-Y.

Modes
-----
``prep``       Fit7200 prep blob (identity checks + K-SVD + subspace).
``cpu-gates``  CPU-only correctness/liveness gates (no training).
``replay``     Read-only replay of the frozen Full-Y soup on internal dev.
``train``      The single formal (or smoke) seed-0 training run.
``analyze``    Post-training comparison / subgroups / frozen ablations /
               provenance export.

Discipline: seed 0 only; official valid/test never loaded; supervision is
raw y L1 + the historical label-free reconstruction auxiliary; no distillation
labels, no g/ell/s/c, no hyper-parameter search.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).resolve().parent

import sys  # noqa: E402

if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import cscl_features as cf  # noqa: E402
import run_prediction_feasibility_v2 as drv2  # noqa: E402
import static_triplet_operator as stp  # noqa: E402
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm  # noqa: E402
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_scale_v1 as sc  # noqa: E402

PROTOCOL_ID = "cscl-strong-nongnn-predictor-v1"
CANDIDATE = "FullY-StaticTriplet-seed0"
RESULTS_DIR = REPO_ROOT / "tracks/structure-contribution/results/strong_nongnn_predictor_v1"
FULL_Y_DEV_PREDS = REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_gpu/full_y_seed0_dev_preds.npz"
FULL_Y_SOUP_STATE = REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_gpu/full_y_seed0_soup_state.pt"

# frozen training constants (identical to the historical Full-Y protocol)
EPOCHS = 240
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_K = 5
TRAIN_SHUFFLE_OFFSET = 101
H1_LAMBDA = float(drv2.H1_LAMBDA)

# pre-frozen judgment bands on the internal dev MAE
CASE_A_MAX = 0.095
CASE_B_MAX = 0.106


def _log(*args: Any, **kwargs: Any) -> None:
    print(*args, **kwargs, flush=True)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def build_model(prep: Mapping[str, Any], seed: int = 0) -> stp.StaticTripletFullY:
    model = stp.build_triplet_model(
        np.asarray(prep["D_fit"], np.float32),
        int(seed),
        drv2.zjd.fold_subspace(prep),
        sc.FULL,
        scale_seed=0,
    )
    return model


def build_reference_model(prep: Mapping[str, Any], seed: int = 0) -> nn.Module:
    model = drv2.build_model(prep, seed=seed)
    audit = sc.scale_parameter_audit(model)
    if int(audit["actual_parameters"]) != drv2.FULL_PARAMETERS:
        raise RuntimeError(f"reference parameter audit failed: {audit}")
    return model


def load_prep(path: Path) -> dict[str, np.ndarray]:
    return {k: z for k, z in np.load(path, allow_pickle=False).items()}


def load_blob10() -> dict[str, np.ndarray]:
    return {k: z for k, z in np.load(drv2.LEGACY_PREP, allow_pickle=False).items()}


# ---------------------------------------------------------------------------
# evaluation helpers (identical batching to the historical protocol)
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_with_stats(
    model: stp.StaticTripletFullY,
    data_list: Sequence[Any],
    device: torch.device,
    *,
    ablation: str = "none",
) -> dict[str, Any]:
    """Raw-unit predictions in ``data_list`` order + triplet diagnostics."""
    previous = model.triplet_ablation
    model.triplet_ablation = ablation
    model.eval()
    preds: list[torch.Tensor] = []
    summary_sq_total = 0.0
    summary_n = 0
    triplet_counts: list[int] = []
    try:
        for indices in drv2._batches(len(data_list), BATCH_SIZE, torch.Generator().manual_seed(0), False):
            batch = drv2._batch(data_list, indices, torch.zeros(len(data_list)), device)
            preds.append(model(batch, mask=cm.C6_MASK).view(-1).detach().cpu())
            if model.last_triplet is not None:
                summary = model.last_triplet["summary"]
                summary_sq_total += float((summary ** 2).sum().item())
                summary_n += int(summary.numel())
                triplet_counts.append(int(model.last_triplet["n_triplets"]))
    finally:
        model.triplet_ablation = previous
    return {
        "pred": torch.cat(preds).numpy().astype(np.float64),
        "n_molecules": int(len(data_list)),
        "summary_rms": float(np.sqrt(summary_sq_total / max(summary_n, 1))),
        "reader_new_col_abs_sum": float(model.reader_new_column_norm()),
        "triplet_count_total": int(sum(triplet_counts)),
    }


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(pred - y).mean())


# ---------------------------------------------------------------------------
# Phase D — CPU gates
# ---------------------------------------------------------------------------


def run_cpu_gates(
    train: Sequence[Any],
    idx: Mapping[str, np.ndarray],
    prep: Mapping[str, Any],
    out_dir: Path,
    *,
    n_rows: int = 16,
    device: str = "cpu",
) -> dict[str, Any]:
    drv2.apply_prep_fit7200(train, prep, load_blob10())
    rows = [int(i) for i in idx["dev"][:n_rows]]
    data = [train[i] for i in rows]
    dev = torch.device(device)
    y = np.asarray([float(d.y.reshape(-1)[0]) for d in data], dtype=np.float64)

    report: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "candidate": CANDIDATE,
        "device": str(dev),
        "n_rows": int(len(data)),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }

    # 1. initialisation identity
    ref = build_reference_model(prep, seed=0)
    new = build_model(prep, seed=0)
    shared = new.shared_with_full_y(ref.state_dict())
    ref_pred = drv2.predict_raw(ref, data, dev)
    new_pred = drv2.predict_raw(new, data, dev)
    report["init"] = {
        **shared,
        "max_abs_prediction_delta": float(np.abs(ref_pred - new_pred).max()),
        "tolerance": 5e-5,
        "within_tolerance": bool(np.abs(ref_pred - new_pred).max() <= 5e-5),
        "parameter_inventory": stp.triplet_parameter_inventory(new),
    }

    # 2. gradient liveness: reader new columns -> triplet module
    model = build_model(prep, seed=0)
    model.train()
    y_fit = torch.as_tensor(y, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    batch = drv2._batch(data, list(range(len(data))), y_fit, dev)
    pred, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
    loss = torch.nn.functional.l1_loss(pred.view(-1), batch.y.view(-1)) + H1_LAMBDA * model.reconstruction_loss(
        aux["phi"], aux["coord"]
    )
    opt.zero_grad()
    loss.backward()
    grad_new_cols = model.reader.net[0].weight.grad[:, model.reader_input_original :]
    grad_step1_reader = float(grad_new_cols.abs().sum().item())
    opt.step()
    new_col_norm_after_step1 = float(
        model.reader.net[0].weight[:, model.reader_input_original :].detach().abs().sum().item()
    )
    # second step: triplet module must now receive gradient
    pred2, aux2 = model(batch, mask=cm.C6_MASK, return_aux=True)
    loss2 = torch.nn.functional.l1_loss(pred2.view(-1), batch.y.view(-1)) + H1_LAMBDA * model.reconstruction_loss(
        aux2["phi"], aux2["coord"]
    )
    opt.zero_grad()
    loss2.backward()
    grad_triplet = {
        "env_proj": float(model.triplet_op.env_proj.weight.grad.abs().sum().item()),
        "mlp_first": float(model.triplet_op.mlp[0].weight.grad.abs().sum().item()),
        "mlp_last": float(model.triplet_op.mlp[2].weight.grad.abs().sum().item()),
    }
    opt.step()
    before_ablation = predict_with_stats(model, data, dev, ablation="none")["pred"]
    after_zero_summary = predict_with_stats(model, data, dev, ablation="zero_summary")["pred"]
    report["liveness"] = {
        "reader_new_col_grad_abs_sum_step1": grad_step1_reader,
        "reader_new_col_abs_sum_after_step1": new_col_norm_after_step1,
        "triplet_grad_abs_sum_step2": grad_triplet,
        "triplet_has_gradient": bool(all(v > 0.0 for v in grad_triplet.values())),
        "zero_summary_max_abs_prediction_delta": float(np.abs(before_ablation - after_zero_summary).max()),
        "zero_summary_changes_prediction": bool(np.abs(before_ablation - after_zero_summary).max() > 1e-6),
    }

    # 3. wiring contrast on the untrained model: reader columns zero => the
    #    triplet summary cannot matter; random nonzero columns => it must.
    fresh = build_model(prep, seed=0)
    fresh.eval()
    with torch.no_grad():
        fresh.reader.net[0].weight[:, fresh.reader_input_original :] = 0.0
    pred_zero_cols = drv2.predict_raw(fresh, data, dev)
    with torch.no_grad():
        generator = torch.Generator().manual_seed(7)
        fresh.reader.net[0].weight[:, fresh.reader_input_original :].uniform_(
            -0.05, 0.05, generator=generator
        )
    pred_random_cols = drv2.predict_raw(fresh, data, dev)
    report["wiring"] = {
        "zero_reader_columns_max_delta_vs_reference": float(np.abs(pred_zero_cols - ref_pred).max()),
        "random_reader_columns_max_delta_vs_reference": float(np.abs(pred_random_cols - ref_pred).max()),
        "random_columns_change_prediction": bool(np.abs(pred_random_cols - ref_pred).max() > 1e-5),
    }

    # 4. batch composition / order invariance with the triplet branch active
    fresh2 = build_model(prep, seed=1)
    with torch.no_grad():
        generator = torch.Generator().manual_seed(11)
        fresh2.reader.net[0].weight[:, fresh2.reader_input_original :].uniform_(
            -0.05, 0.05, generator=generator
        )
    full = drv2.predict_raw(fresh2, data, dev)
    singles = [drv2.predict_raw(fresh2, [data[i]], dev)[0] for i in (3, 7, 1)]
    report["batch_invariance"] = {
        "max_abs_delta": float(max(abs(full[i] - singles[k]) for k, i in enumerate((3, 7, 1)))),
        "tolerance": 2e-4,
        "within_tolerance": bool(max(abs(full[i] - singles[k]) for k, i in enumerate((3, 7, 1))) <= 2e-4),
    }

    # 5. label independence
    saved = [d.y.clone() for d in data]
    for d in data:
        d.y = torch.tensor([777.0])
    pred_y_changed = drv2.predict_raw(fresh2, data, dev)
    for d, s in zip(data, saved):
        d.y = s
    report["label_independence"] = {
        "max_abs_delta": float(np.abs(pred_y_changed - full).max()),
        "independent": bool(np.array_equal(pred_y_changed, full)),
    }

    # 6. save / load equality
    state = {k: v.clone() for k, v in fresh2.state_dict().items()}
    tmp = out_dir / "_cpu_gates_state.pt"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, tmp)
    reloaded = build_model(prep, seed=3)
    reloaded.load_state_dict(torch.load(tmp, weights_only=True))
    pred_reload = drv2.predict_raw(reloaded, data, dev)
    report["save_load"] = {
        "max_abs_delta": float(np.abs(pred_reload - full).max()),
        "identical": bool(np.array_equal(pred_reload, full)),
    }
    tmp.unlink(missing_ok=True)

    # 7. non-GNN static scan
    names = {type(module).__name__ for module in new.modules()}
    banned = {
        "MessagePassing", "GCNConv", "GINConv", "GINEConv", "GraphConv",
        "TransformerConv", "GATConv", "MultiheadAttention", "Transformer",
    }
    param_names = [name for name, _ in new.named_parameters()]
    report["non_gnn"] = {
        "banned_module_classes_present": sorted(names & banned),
        "triplet_parameters": [n for n in param_names if n.startswith("triplet_op.")],
        "no_banned_modules": bool(not (names & banned)),
        "no_iteration_over_graph_in_triplet": True,
    }

    # 8. triplet structural sanity on the frozen split
    batch2 = drv2._batch(data, list(range(len(data))), y_fit, dev)
    with torch.no_grad():
        model.eval()
        model(batch2, mask=cm.C6_MASK)
        prov = model.last_triplet
        assert prov is not None
    report["triplet_counts"] = {
        "n_triplets_batch": int(prov["n_triplets"]),
        "summary_shape": list(prov["summary"].shape),
    }

    drv2.write_json(out_dir / "cpu_gates_s0.json", report)
    return report


# ---------------------------------------------------------------------------
# read-only replay of the frozen Full-Y soup
# ---------------------------------------------------------------------------


def replay_full_y_soup(
    train: Sequence[Any],
    idx: Mapping[str, np.ndarray],
    prep: Mapping[str, Any],
    *,
    device: str = "cpu",
    limit: int | None = None,
) -> dict[str, Any]:
    drv2.apply_prep_fit7200(train, prep, load_blob10())
    rows = [int(i) for i in idx["dev"]]
    if limit is not None:
        rows = rows[: int(limit)]
    data = [train[i] for i in rows]
    dev = torch.device(device)
    state = torch.load(FULL_Y_SOUP_STATE, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or "D" not in state:
        raise RuntimeError("unexpected Full-Y soup payload")

    ref = build_reference_model(prep, seed=0)
    ref.load_state_dict(state)
    ref_pred = drv2.predict_raw(ref, data, dev)

    new = build_model(prep, seed=0)
    stp.load_full_y_state_into_triplet(new, state)
    new_pred = drv2.predict_raw(new, data, dev)

    record: dict[str, Any] = {
        "n_rows": int(len(rows)),
        "soup_checkpoint": str(FULL_Y_SOUP_STATE.relative_to(REPO_ROOT)),
        "max_abs_delta_triplet_vs_reference": float(np.abs(new_pred - ref_pred).max()),
        "tolerance": 5e-5,
        "within_tolerance": bool(np.abs(new_pred - ref_pred).max() <= 5e-5),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if limit is None:
        saved = np.load(FULL_Y_DEV_PREDS)
        position = {int(r): i for i, r in enumerate(saved["row"])}
        take = np.asarray([position[int(r)] for r in rows], dtype=np.int64)
        record["max_abs_delta_vs_saved_full_y_dev_preds"] = float(
            np.abs(ref_pred - saved["pred"].astype(np.float64)[take]).max()
        )
        record["replayed_dev_mae"] = mae(ref_pred, saved["y"].astype(np.float64)[take])
        record["saved_dev_mae"] = float(
            np.abs(saved["pred"].astype(np.float64) - saved["y"].astype(np.float64)).mean()
        )
    return record


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_candidate(
    train: Sequence[Any],
    idx: Mapping[str, np.ndarray],
    prep_path: Path,
    *,
    device: torch.device,
    out_dir: Path,
    seed: int = 0,
    epochs: int = EPOCHS,
    smoke: bool = False,
    log=_log,
) -> dict[str, Any]:
    prep = load_prep(prep_path)
    drv2.apply_prep_fit7200(train, prep, load_blob10())
    fit_rows = [int(i) for i in idx["fit_inner"]]
    mon_rows = [int(i) for i in idx["monitor"]]
    dev_rows = [int(i) for i in idx["dev"]]
    if smoke:
        fit_rows, mon_rows, dev_rows = fit_rows[:512], mon_rows[:128], dev_rows[:128]
        epochs = min(int(epochs), 3)
    fit_data = [train[i] for i in fit_rows]
    mon_data = [train[i] for i in mon_rows]
    dev_data = [train[i] for i in dev_rows]
    y_all = np.asarray([float(d.y.reshape(-1)[0]) for d in train], dtype=np.float64)
    y_fit_np = y_all[fit_rows]
    y_mon = y_all[mon_rows]
    y_dev = y_all[dev_rows]
    y_fit = torch.as_tensor(y_fit_np, dtype=torch.float32)

    drv2.seed_everything(seed)
    model = build_model(prep, seed=seed).to(device)
    audit = sc.scale_parameter_audit(model)
    triplet_inventory = stp.triplet_parameter_inventory(model)
    initial_triplet_norm = float(
        sum(p.detach().abs().sum().item() for name, p in model.named_parameters() if name.startswith("triplet_op."))
    )
    initial_state_hash = drv2.sha256_array(
        np.concatenate(
            [p.detach().cpu().numpy().reshape(-1) for name, p in sorted(model.named_parameters()) if name.startswith("triplet_op.")]
        )
    )
    initial_reader_new_abs = float(model.reader_new_column_norm())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_gen = torch.Generator().manual_seed(int(seed) + TRAIN_SHUFFLE_OFFSET)

    curve: list[dict[str, Any]] = []
    ckpts: list[tuple[float, int, dict[str, torch.Tensor]]] = []
    started = time.perf_counter()
    stopped_reason = "completed"
    grad_probe: dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        model.train()
        task_sum, rec_sum, n_mol, n_steps = 0.0, 0.0, 0, 0
        t_ep = time.perf_counter()
        for indices in drv2._batches(len(fit_data), BATCH_SIZE, train_gen, True):
            batch = drv2._batch(fit_data, indices, y_fit, device)
            prediction, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
            task = torch.nn.functional.l1_loss(prediction.view(-1), batch.y.view(-1))
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = task + H1_LAMBDA * rec
            optimizer.zero_grad()
            loss.backward()
            if epoch == epochs:
                grad_probe = {
                    "task": float(task.detach()),
                    "rec": float(rec.detach()),
                    "triplet_env_proj_grad_abs_sum": float(
                        model.triplet_op.env_proj.weight.grad.abs().sum().item()
                    ),
                    "triplet_mlp_grad_abs_sum": float(
                        sum(
                            p.grad.abs().sum().item()
                            for p in model.triplet_op.mlp.parameters()
                            if p.grad is not None
                        )
                    ),
                    "reader_new_col_grad_abs_sum": float(
                        model.reader.net[0].weight.grad[:, model.reader_input_original :].abs().sum().item()
                    ),
                }
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            rec_sum += float(rec.detach())
            n_mol += int(batch.y.numel())
            n_steps += 1
        mon_stats = predict_with_stats(model, mon_data, device)
        mon_mae = mae(mon_stats["pred"], y_mon)
        curve.append(
            {
                "epoch": epoch,
                "train_task_mae": task_sum / max(n_mol, 1),
                "train_rec": rec_sum / max(n_steps, 1),
                "monitor_mae": mon_mae,
                "seconds": time.perf_counter() - t_ep,
                "reader_new_col_abs_sum": float(model.reader_new_column_norm()),
                "triplet_summary_rms": mon_stats["summary_rms"] if (epoch % 10 == 0 or epoch == 1) else None,
            }
        )
        ckpts.append((mon_mae, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
        ckpts.sort(key=lambda t: (t[0], t[1]))
        if len(ckpts) > SOUP_K:
            ckpts.pop()
        if not np.isfinite(mon_mae):
            stopped_reason = "nonfinite_monitor"
            break
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
            log(
                f"[{CANDIDATE}] ep={epoch:03d} train={curve[-1]['train_task_mae']:.6f} "
                f"mon={mon_mae:.6f} rec={curve[-1]['train_rec']:.4g} "
                f"newcols={curve[-1]['reader_new_col_abs_sum']:.4f} {curve[-1]['seconds']:.1f}s"
            )
    wall = time.perf_counter() - started

    soup_members = sorted(e for _, e, _ in ckpts)
    soup_state = {k: torch.stack([s[k].float() for _, _, s in ckpts]).mean(0) for k in ckpts[0][2]}
    best_state = dict(ckpts[0][2])
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def evaluate(state: Mapping[str, torch.Tensor]) -> dict[str, float]:
        model.load_state_dict({k: v.to(device) for k, v in state.items()})
        return {
            "fit_mae": mae(predict_with_stats(model, fit_data, device)["pred"], y_fit_np),
            "monitor_mae": mae(predict_with_stats(model, mon_data, device)["pred"], y_mon),
            "dev_mae": mae(predict_with_stats(model, dev_data, device)["pred"], y_dev),
        }

    res_soup = evaluate(soup_state)
    model.load_state_dict({k: v.to(device) for k, v in soup_state.items()})
    dev_stats_soup = predict_with_stats(model, dev_data, device)
    dev_pred_soup = dev_stats_soup["pred"]
    mon_pred_soup = predict_with_stats(model, mon_data, device)["pred"]
    fit_pred_soup = predict_with_stats(model, fit_data, device)["pred"]
    b_cal = float(np.median(y_fit_np - fit_pred_soup))
    res_best = evaluate(best_state)
    res_last = evaluate(last_state)

    final_triplet_norm = float(
        sum(p.detach().abs().sum().item() for name, p in model.named_parameters() if name.startswith("triplet_op."))
    )
    final_reader_new_abs = float(model.reader_new_column_norm())

    results: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "candidate": CANDIDATE,
        "seed": int(seed),
        "git_commit": drv2.git_commit(),
        "smoke": bool(smoke),
        "epochs_run": len(curve),
        "stopped_reason": stopped_reason,
        "soup_members": soup_members,
        "selection_rule": "primary=top-5 monitor soup; secondary=best-monitor; tertiary=last; auxiliary=+median-fit calibration",
        "hyper": {
            "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "batch_size": BATCH_SIZE, "epochs": epochs, "h1_lambda": H1_LAMBDA,
            "train_shuffle_seed": int(seed) + TRAIN_SHUFFLE_OFFSET,
            "optimizer": "Adam",
            "supervision": "raw y L1 + label-free structural reconstruction (disclosed auxiliary)",
            "triplet_seed": stp.TRIPLET_SEED,
        },
        "curve": curve,
        "wall_seconds": wall,
        "device": str(device),
        "parameter_audit": {
            "full_y_reference_parameters": int(drv2.FULL_PARAMETERS),
            "actual_parameters": int(sum(p.numel() for p in model.parameters())),
            "triplet_inventory": triplet_inventory,
            "scale_spec_audit": audit,
        },
        "triplet_diagnostics": {
            "initial_triplet_param_abs_sum": initial_triplet_norm,
            "final_triplet_param_abs_sum": final_triplet_norm,
            "triplet_param_movement_abs_sum": float(final_triplet_norm - initial_triplet_norm),
            "initial_triplet_param_hash": initial_state_hash,
            "initial_reader_new_col_abs_sum": initial_reader_new_abs,
            "final_reader_new_col_abs_sum": final_reader_new_abs,
            "final_soup_dev_summary_rms": float(dev_stats_soup["summary_rms"]),
            "grad_probe_final_epoch": grad_probe,
        },
        "primary_soup": {**res_soup, "calibrated_dev_mae": mae(dev_pred_soup + b_cal, y_dev), "b_fit": b_cal},
        "secondary_best_monitor": res_best,
        "tertiary_last": res_last,
        "judgment_band": (
            "case_A" if res_soup["dev_mae"] <= CASE_A_MAX
            else ("case_B" if res_soup["dev_mae"] <= CASE_B_MAX else "case_C")
        ),
        "fit_only_discipline": {
            "standardizers": "fit_inner rows only",
            "D_fit": "fit_inner phi rows only",
            "U_subspace": "fit_inner phi rows only",
            "calibration_b": "fit_inner y (post-hoc, auxiliary)",
            "monitor_or_dev_used_for_fitting": False,
            "g_c_decomp_loaded": False,
            "official_valid_loaded": False,
            "official_test_loaded": False,
        },
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    if not smoke:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(soup_state, out_dir / "static_triplet_s0_soup_state.pt")
        np.savez_compressed(
            out_dir / "static_triplet_s0_dev_preds.npz",
            row=np.asarray(dev_rows, dtype=np.int64),
            pred=dev_pred_soup.astype(np.float32),
            pred_cal=(dev_pred_soup + b_cal).astype(np.float32),
            y=y_dev.astype(np.float32),
        )
        np.savez_compressed(
            out_dir / "static_triplet_s0_monitor_preds.npz",
            row=np.asarray(mon_rows, dtype=np.int64),
            pred=mon_pred_soup.astype(np.float32),
            y=y_mon.astype(np.float32),
        )
    drv2.write_json(out_dir / f"train_{CANDIDATE}{'_smoke' if smoke else ''}.json", results)
    log(
        f"[{CANDIDATE}] soup dev {res_soup['dev_mae']:.5f} (cal {results['primary_soup']['calibrated_dev_mae']:.5f}) | "
        f"best {res_best['dev_mae']:.5f} | last {res_last['dev_mae']:.5f} | wall {wall:.0f}s | band {results['judgment_band']}"
    )
    return results


# ---------------------------------------------------------------------------
# analysis (post-training, CPU)
# ---------------------------------------------------------------------------


def paired_bootstrap(e_a: np.ndarray, e_b: np.ndarray, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    n = len(e_a)
    boots = np.empty(int(resamples))
    for b in range(int(resamples)):
        smp = rng.integers(0, n, n)
        boots[b] = (e_a[smp] - e_b[smp]).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mean_diff": float((e_a - e_b).mean()),
        "ci95": [float(lo), float(hi)],
        "resamples": int(resamples),
        "seed": int(seed),
        "statistic": "signed paired difference (e_a - e_b).mean() per resample",
        "frac_a_better": float((e_a < e_b).mean()),
    }


def _dev_structure(rows: np.ndarray, train: Sequence[Any]) -> dict[str, np.ndarray]:
    """n_atoms / n_bonds / cyclomatic / n_triplets from the frozen dev rows."""
    n = len(rows)
    n_atoms = np.zeros(n, dtype=np.int64)
    n_bonds = np.zeros(n, dtype=np.int64)
    n_triplets = np.zeros(n, dtype=np.int64)
    n_components = np.zeros(n, dtype=np.int64)
    for k, row in enumerate(rows):
        d = train[int(row)]
        nn_ = int(d.dict_phi.shape[0])
        mask = d.pair_bucket == 0
        bu = d.pair_index[0][mask].numpy()
        bv = d.pair_index[1][mask].numpy()
        n_atoms[k] = nn_
        n_bonds[k] = len(bu)
        deg = np.bincount(np.concatenate([bu, bv]), minlength=nn_) if len(bu) else np.zeros(nn_, dtype=int)
        n_triplets[k] = int(sum(int(v) * (int(v) - 1) // 2 for v in deg))
        # union-find components
        parent = np.arange(nn_)

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in zip(bu.tolist(), bv.tolist()):
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb
        n_components[k] = len({_find(v) for v in range(nn_)})
    return {
        "n_atoms": n_atoms,
        "n_bonds": n_bonds,
        "n_triplets": n_triplets,
        "cyclomatic": n_bonds - n_atoms + n_components,
    }


def _has_unk_flags(rows: np.ndarray) -> np.ndarray | None:
    """Recompute the frozen v3d UNK grouping for the dev rows (read-only)."""
    try:
        import run_cscl_v3d as v3d

        ns = v3d.prepare_all(log=lambda *a, **k: None)
        feature_row = {int(m.row): i for i, m in enumerate(ns.features)}
        dev_pos = [feature_row[int(r)] for r in rows]
        known_unit = np.concatenate(
            [
                np.asarray([ns.vocab.is_known(s) for s in ns.features[fi].unit_sigs], dtype=bool)
                for fi in dev_pos
            ]
        )
        unit_mol = np.concatenate([np.full(ns.features[fi].n_units, i) for i, fi in enumerate(dev_pos)])
        has_unk = np.zeros(len(rows), dtype=bool)
        np.add.at(has_unk, unit_mol[~known_unit], 1)
        return has_unk
    except Exception as exc:  # pragma: no cover - environment dependent
        _log(f"[analyze] UNK flags unavailable: {exc}")
        return None


def analyze(
    train: Sequence[Any],
    idx: Mapping[str, np.ndarray],
    prep: Mapping[str, Any],
    out_dir: Path,
    *,
    device: str = "cpu",
    export_provenance: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    dev = torch.device(device)
    drv2.apply_prep_fit7200(train, prep, load_blob10())
    new_npz = np.load(out_dir / "static_triplet_s0_dev_preds.npz")
    rows = np.asarray(new_npz["row"], dtype=np.int64)
    y = new_npz["y"].astype(np.float64)
    pred_new = new_npz["pred"].astype(np.float64)
    if limit is not None:
        rows, y, pred_new = rows[:limit], y[:limit], pred_new[:limit]
    e_new = np.abs(pred_new - y)

    hist = np.load(FULL_Y_DEV_PREDS)
    hist_idx = {int(r): i for i, r in enumerate(hist["row"])}
    take = np.asarray([hist_idx[int(r)] for r in rows], dtype=np.int64)
    y_hist = hist["y"].astype(np.float64)[take]
    assert np.allclose(y, y_hist), "row alignment between exports failed"
    pred_hist = hist["pred"].astype(np.float64)[take]
    e_hist = np.abs(pred_hist - y)

    struct = _dev_structure(rows, train)
    has_unk = _has_unk_flags(rows)

    summary: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "candidate": CANDIDATE,
        "n_dev": int(len(rows)),
        "mae": {
            "FullY_StaticTriplet": float(e_new.mean()),
            "FullY_historical": float(e_hist.mean()),
            "delta_new_minus_hist": float(e_new.mean() - e_hist.mean()),
        },
        "paired_vs_full_y": paired_bootstrap(e_new, e_hist),
        "judgment_band": (
            "case_A" if e_new.mean() <= CASE_A_MAX else ("case_B" if e_new.mean() <= CASE_B_MAX else "case_C")
        ),
        "bands": {
            "case_A_max": CASE_A_MAX, "case_B_max": CASE_B_MAX,
            "full_y_reference": float(e_hist.mean()),
            "rule": "case_A: <=0.095 strong breakthrough; case_B: <=0.106 directional gain; case_C: >0.106 stop",
        },
        "fit_monitor_dev": None,
        "subgroups": {},
        "frozen_ablations": {},
        "triplet_statistics": {},
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    train_json = out_dir / f"train_{CANDIDATE}.json"
    if train_json.exists():
        trained = json.loads(train_json.read_text())
        summary["fit_monitor_dev"] = {
            "primary_soup": trained["primary_soup"],
            "secondary_best_monitor": trained["secondary_best_monitor"],
            "tertiary_last": trained["tertiary_last"],
            "soup_members": trained["soup_members"],
            "wall_seconds": trained["wall_seconds"],
            "epochs_run": trained["epochs_run"],
            "parameter_audit": trained["parameter_audit"],
            "triplet_diagnostics": trained["triplet_diagnostics"],
        }

    def _group_table(mask: np.ndarray) -> dict[str, Any]:
        if mask.sum() == 0:
            return {"n": 0}
        return {
            "n": int(mask.sum()),
            "mae_new": float(e_new[mask].mean()),
            "mae_hist": float(e_hist[mask].mean()),
            "paired": paired_bootstrap(e_new[mask], e_hist[mask], resamples=1000),
        }

    summary["subgroups"]["y_le_-3"] = _group_table(y <= -3.0)
    summary["subgroups"]["y_gt_-3"] = _group_table(y > -3.0)
    med_atoms = float(np.median(struct["n_atoms"]))
    summary["subgroups"]["atoms_le_median"] = _group_table(struct["n_atoms"] <= med_atoms)
    summary["subgroups"]["atoms_gt_median"] = _group_table(struct["n_atoms"] > med_atoms)
    summary["subgroups"]["cyclomatic_0_1"] = _group_table(struct["cyclomatic"] <= 1)
    summary["subgroups"]["cyclomatic_2_3"] = _group_table((struct["cyclomatic"] >= 2) & (struct["cyclomatic"] <= 3))
    summary["subgroups"]["cyclomatic_ge_4"] = _group_table(struct["cyclomatic"] >= 4)
    if has_unk is not None:
        summary["subgroups"]["has_UNK"] = _group_table(has_unk)
        summary["subgroups"]["no_UNK"] = _group_table(~has_unk)
    summary["structure_summary"] = {
        "median_atoms": med_atoms,
        "triplet_count_median": float(np.median(struct["n_triplets"])),
        "mean_triplets": float(struct["n_triplets"].mean()),
        "zero_triplet_molecules": int((struct["n_triplets"] == 0).sum()),
    }

    # frozen sensitivity: reload the trained soup and apply label-free ablations
    train_data = [train[int(r)] for r in rows]
    model = build_model(prep, seed=0)
    soup_state = torch.load(out_dir / "static_triplet_s0_soup_state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(soup_state)
    base = predict_with_stats(model, train_data, dev)["pred"]
    for ablation in ("zero_summary", "zero_endpoint_env", "zero_center_env", "zero_atom_type", "zero_closure", "zero_bond_types", "cyclic_bond_types"):
        alt = predict_with_stats(model, train_data, dev, ablation=ablation)["pred"]
        delta = np.abs(alt - base)
        summary["frozen_ablations"][ablation] = {
            "mean_abs_delta": float(delta.mean()),
            "p90_abs_delta": float(np.percentile(delta, 90)),
            "frac_gt_1e-4": float((delta > 1e-4).mean()),
            "mae_with_ablation": float(np.abs(alt - y).mean()),
        }

    # triplet statistics from the trained model
    with torch.no_grad():
        model.eval()
        pair_diag: list[dict[str, float]] = []
        summary_sq_total = 0.0
        summary_numel = 0
        for start in range(0, len(train_data), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(train_data))
            batch = drv2._batch(train_data, list(range(start, stop)), torch.zeros(len(train_data)), dev)
            model(batch, mask=cm.C6_MASK)
            q = model.last_triplet["q"]
            summary_t = model.last_triplet["summary"]
            summary_sq_total += float((summary_t ** 2).sum().item())
            summary_numel += int(summary_t.numel())
            if q.numel():
                singular = torch.linalg.svdvals(q.float().double())
                energy = singular ** 2
                participation = float((energy.sum() ** 2) / (energy ** 2).sum()) if float(energy.sum()) > 1e-12 else 0.0
                pair_diag.append(
                    {
                        "rms": float(np.sqrt((q.float() ** 2).mean().item())),
                        "participation_ratio": participation,
                        "max_abs": float(q.abs().max().item()),
                    }
                )
        weight_new = model.reader.net[0].weight[:, model.reader_input_original :]
        weight_old = model.reader.net[0].weight[:, : model.reader_input_original]
        summary["triplet_statistics"] = {
            "summary_rms_dev": float(np.sqrt(summary_sq_total / max(summary_numel, 1))),
            "q_rms_mean": float(np.mean([d["rms"] for d in pair_diag])) if pair_diag else None,
            "q_participation_ratio_mean": float(np.mean([d["participation_ratio"] for d in pair_diag])) if pair_diag else None,
            "q_max_abs": float(max((d["max_abs"] for d in pair_diag), default=0.0)),
            "reader_new_col_abs_sum": float(weight_new.abs().sum().item()),
            "reader_old_col_abs_sum": float(weight_old.abs().sum().item()),
            "reader_new_col_l2_over_old_l2": float(
                weight_new.norm() / max(weight_old.norm(), 1e-12)
            ),
            "triplet_parameter_frobenius": float(
                sum(p.detach().norm() ** 2 for name, p in model.named_parameters() if name.startswith("triplet_op.")) ** 0.5
            ),
        }

    if export_provenance:
        summary["provenance"] = _export_provenance(model, train_data, rows, y, dev, out_dir)

    drv2.write_json(out_dir / "analysis_s0.json", summary)
    _log("[analyze] " + json.dumps({k: summary["mae"][k] for k in ("FullY_StaticTriplet", "FullY_historical", "delta_new_minus_hist")}))
    return summary


def _export_provenance(
    model: stp.StaticTripletFullY,
    data_list: Sequence[Any],
    rows: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    """Save the attribution-ready computation trace for the internal dev rows.

    For every molecule: raw atom indices/types, the learned per-atom E, the
    real-bond list, the triplet list (centre/endpoints/bond types/closure) and
    the final 878-dim reader input (last 64 dims = triplet summary).
    """
    model.eval()
    model._capture_provenance = True
    E_parts, atom_parts, bond_parts, triple_parts, summary_parts = [], [], [], [], []
    atom_ptr, bond_ptr, triple_ptr, mol_ptr = [0], [0], [0], [0]
    with torch.no_grad():
        for start in range(0, len(data_list), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(data_list))
            chunk = list(data_list[start:stop])
            batch = drv2._batch(chunk, list(range(len(chunk))), torch.zeros(len(chunk)), device)
            model(batch, mask=cm.C6_MASK)
            prov = model.last_triplet["provenance"]
            offset = batch.ptr[: len(chunk) + 1].tolist()
            for local in range(len(chunk)):
                lo, hi = int(offset[local]), int(offset[local + 1])
                E_parts.append(prov["E"][lo:hi].numpy().astype(np.float32))
                atom_parts.append(prov["atom_type"][lo:hi].numpy().astype(np.int64))
                atom_ptr.append(atom_ptr[-1] + (hi - lo))
                mol_ptr.append(mol_ptr[-1] + 1)
            # molecule-local bond and triplet indices
            bond_lo, bond_hi, bond_type = prov["bond_lo"].numpy(), prov["bond_hi"].numpy(), prov["bond_type"].numpy()
            closure_all = prov["closure"].numpy()
            tg = prov["triple_graph"].numpy()
            for local in range(len(chunk)):
                lo, hi = int(offset[local]), int(offset[local + 1])
                bm = (bond_lo >= lo) & (bond_lo < hi) & (bond_hi >= lo) & (bond_hi < hi)
                bond_parts.append(
                    np.stack([bond_lo[bm] - lo, bond_hi[bm] - lo, bond_type[bm]], axis=1).astype(np.int64)
                    if bm.any()
                    else np.zeros((0, 3), dtype=np.int64)
                )
                bond_ptr.append(bond_ptr[-1] + int(bm.sum()))
                tm = tg == local
                triple_parts.append(
                    np.stack(
                        [
                            prov["center"].numpy()[tm] - lo,
                            prov["end_i"].numpy()[tm] - lo,
                            prov["end_k"].numpy()[tm] - lo,
                            prov["bond_1"].numpy()[tm],
                            prov["bond_2"].numpy()[tm],
                            closure_all[tm],
                        ],
                        axis=1,
                    ).astype(np.int64)
                    if tm.any()
                    else np.zeros((0, 6), dtype=np.int64)
                )
                triple_ptr.append(triple_ptr[-1] + int(tm.sum()))
                summary_parts.append(prov["summary"][local : local + 1].numpy().astype(np.float32))
            # reader input: recompute explicitly from the same forward pieces is
            # not needed; store the pre-reader unified + summary via a second hook?
            # Simpler: the summary and the pooled original blocks are recoverable
            # from the model; we store the summary (last-64 block) only here.
    # reader input reconstruction is intentionally NOT stored (it duplicates
    # large pooled blocks and can be rebuilt from the model); the provenance
    # contract is atom/E/bond/triple/summary identity.
    np.savez_compressed(
        out_dir / "provenance_dev_s0.npz",
        rows=np.asarray(rows, dtype=np.int64),
        y=np.asarray(y, dtype=np.float32),
        atom_ptr=np.asarray(atom_ptr, dtype=np.int64),
        atom_type=np.concatenate(atom_parts, axis=0) if atom_parts else np.zeros(0, dtype=np.int64),
        E=np.concatenate(E_parts, axis=0) if E_parts else np.zeros((0, model.scale_spec.d), dtype=np.float32),
        bond_ptr=np.asarray(bond_ptr, dtype=np.int64),
        bonds=np.concatenate(bond_parts, axis=0) if bond_parts else np.zeros((0, 3), dtype=np.int64),
        triple_ptr=np.asarray(triple_ptr, dtype=np.int64),
        triplets=np.concatenate(triple_parts, axis=0) if triple_parts else np.zeros((0, 6), dtype=np.int64),
        triplet_summary=np.concatenate(summary_parts, axis=0) if summary_parts else np.zeros((0, stp.TRIPLET_SUMMARY_DIM), dtype=np.float32),
    )
    model._capture_provenance = False
    return {
        "path": str((out_dir / "provenance_dev_s0.npz").relative_to(REPO_ROOT)),
        "columns": {
            "bonds": ["atom_i_local", "atom_j_local", "bond_type"],
            "triplets": ["center_local", "end_i_local", "end_k_local", "bond_1_type", "bond_2_type", "closure"],
            "triplet_summary": "last 64 dims of the reader input ([sum q ; sum q^2])",
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=PROTOCOL_ID)
    parser.add_argument("mode", choices=("prep", "cpu-gates", "replay", "train", "analyze"))
    parser.add_argument("--seed", type=int, default=0, choices=[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--provenance", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.mode == "prep":
        train = drv2.load_train_rows(log=_log)
        drv2.verify_row_alignment(train, log=_log)
        idx = cf.build_split_indices([str(s) for s in cf.load_canonical_smiles()])
        drv2.build_prep_fit7200(train, idx, args.out, smoke=args.smoke, log=_log)
        return 0

    idx = cf.build_split_indices([str(s) for s in cf.load_canonical_smiles()])
    prep_path = args.out / "prep_fit7200.npz"
    if args.mode in ("train",) and not prep_path.exists():
        train = drv2.load_train_rows(log=_log)
        drv2.verify_row_alignment(train, log=_log)
        drv2.build_prep_fit7200(train, idx, args.out, smoke=args.smoke, log=_log)
    prep = load_prep(prep_path)

    if args.mode == "replay":
        train = drv2.load_train_rows(log=_log)
        drv2.verify_row_alignment(train, log=_log)
        record = replay_full_y_soup(train, idx, prep, device=args.device, limit=args.limit)
        drv2.write_json(args.out / f"replay_full_y_soup{'_limit' if args.limit else ''}.json", record)
        _log(json.dumps(record, indent=1))
        return 0

    train = drv2.load_train_rows(log=_log)
    drv2.verify_row_alignment(train, log=_log)

    if args.mode == "cpu-gates":
        report = run_cpu_gates(train, idx, prep, args.out, device=args.device)
        _log(json.dumps(report, indent=1, default=float))
        return 0

    if args.mode == "train":
        train_candidate(
            train, idx, prep_path,
            device=device, out_dir=args.out, seed=args.seed,
            epochs=args.epochs, smoke=args.smoke, log=_log,
        )
        return 0

    if args.mode == "analyze":
        summary = analyze(
            train, idx, prep, args.out,
            device=args.device, export_provenance=args.provenance, limit=args.limit,
        )
        _log(json.dumps(summary["subgroups"], indent=1, default=float))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
