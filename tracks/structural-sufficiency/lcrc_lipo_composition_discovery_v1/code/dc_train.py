"""One shared training loop for all discovery arms.

The recipe is the frozen LCRC recipe: AdamW, batch 128, 360 epochs
(27 steps/epoch, 9,720 updates), peak LR 3e-4 -> 1e-5 per-step cosine,
weight decay 1e-3, gradient clip 5.0, MSE on the Train3360-standardised
target, dropout 0.10 on the final atom-level local representation, and a
last-five-epoch state-dict soup.  No early stopping, no Validation-based
checkpoint selection and no Test access.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import dc_common as C
import dc_data as D
import dc_models as M

import lcrc_data  # noqa: E402
import r42_data as R42D  # noqa: E402
import r42_metrics as MET  # noqa: E402
import r42_native_train as NT  # noqa: E402

PROCESS_T0 = time.perf_counter()


def _average_states(states: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("empty soup")
    keys = list(states[0].keys())
    return {key: torch.stack([s[key].detach().float().cpu() for s in states], dim=0).mean(dim=0) for key in keys}


def _correction_output(module: torch.nn.Module, family: str):
    if family == "g2l_scalar":
        return module.context.w_out
    if family in ("g2l_all", "g2l_loo", "g2l_local_cap"):
        return module.context.W_out
    if family in ("dist_true", "dist_blind"):
        return module.dist.W_out
    if family in ("path_bind", "path_unbound"):
        return module.pathbind.W_out
    raise KeyError(family)


def predict_with_correction_disabled(model, cache, family: str, device="cpu", batch_size: int = 256) -> np.ndarray:
    out = _correction_output(model, family)
    w = out.weight.detach().clone()
    b = out.bias.detach().clone()
    try:
        with torch.no_grad():
            out.weight.zero_()
            out.bias.zero_()
        return M.predict(model, cache, device=device, batch_size=batch_size)
    finally:
        with torch.no_grad():
            out.weight.copy_(w)
            out.bias.copy_(b)


def _grad_norms(model: torch.nn.Module, names: Sequence[str]) -> dict[str, float]:
    params = dict(model.named_parameters())
    return {name: float(params[name].grad.detach().norm()) for name in names if params[name].grad is not None}


def gradient_probe(model, cache, names, device="cpu", steps: int = 2, lr: float = C.TRAIN_RECIPE["peak_lr"]):
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(lr), weight_decay=0.0)
    params = dict(model.named_parameters())
    before = {name: params[name].detach().clone() for name in names}
    history = []
    rows = list(range(min(16, int(cache["n_molecules"]))))
    for step in range(int(steps)):
        batch = M.to_device(D.collate(cache, rows), device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = F.mse_loss(pred, torch.zeros_like(pred))
        loss.backward()
        grads = {}
        for name in names:
            g = params[name].grad
            grads[name] = {
                "finite": bool(g is not None and torch.isfinite(g).all()),
                "norm": float(g.detach().norm()) if g is not None else 0.0,
            }
        history.append({"step": step + 1, "loss": float(loss.detach()), "grads": grads})
        optimizer.step()
    delta = {name: float((params[name].detach() - before[name]).abs().max()) for name in names}
    return {
        "steps": int(steps),
        "history": history,
        "all_finite": bool(all(g["finite"] for row in history for g in row["grads"].values())),
        "all_nonzero_at_final_step": bool(all(history[-1]["grads"][name]["norm"] > 0.0 for name in names)),
        "parameter_max_abs_update": delta,
        "all_updated": bool(all(v > 0.0 for v in delta.values())),
        "passed": bool(
            all(g["finite"] for row in history for g in row["grads"].values())
            and all(history[-1]["grads"][name]["norm"] > 0.0 for name in names)
            and all(v > 0.0 for v in delta.values())
        ),
    }


def correction_diagnostics(model, family: str, cache, device="cpu", batch_size: int = 64) -> dict[str, Any]:
    """Magnitude/usage statistics of the learned correction."""
    model.eval()
    n = int(cache["n_molecules"])
    change_l2: list[float] = []
    delta_abs: list[float] = []
    graph_change: list[float] = []
    graph_delta: list[float] = []
    n_atoms = 0
    n_hetero = 0
    n_hetero_atoms = 0
    n_path_atoms = 0
    n_multi_type_atoms = 0
    path_type_counts: list[int] = []
    bind_gap_norm: list[float] = []
    d34_nonempty = 0
    d5_nonempty = 0
    both_empty = 0
    n_distance_atoms = 0
    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            rows = list(range(start, min(start + int(batch_size), n)))
            batch = M.to_device(D.collate(cache, rows), device)
            _y, aux = model(batch, return_aux=True)
            atom_batch = batch["atom_batch"].detach().cpu()
            n_atoms += int(atom_batch.numel())
            if "het_mask" in aux:
                het = aux["het_mask"].detach().cpu()
                n_hetero += int(het.sum())
            if family.startswith("g2l"):
                h = aux["h_original"]
                h_new = aux["h_new"]
                change = (h_new - h).detach().float().cpu().norm(dim=1).numpy()
                change_l2.extend(change.tolist())
                for g in range(len(rows)):
                    mask = (atom_batch == g).numpy()
                    graph_change.append(float(change[mask].max()) if mask.any() else 0.0)
                if family == "g2l_scalar":
                    d = aux["delta_atom"].detach().float().cpu().numpy()
                    delta_abs.extend(np.abs(d).tolist())
                    for g in range(len(rows)):
                        mask = (atom_batch == g).numpy()
                        graph_delta.append(float(np.abs(d[mask]).max()) if mask.any() else 0.0)
            elif family.startswith("dist"):
                counts34 = aux["d34_counts"].detach().cpu().numpy()
                counts5 = aux["d5_counts"].detach().cpu().numpy()
                d34_nonempty += int((counts34 > 0).sum())
                d5_nonempty += int((counts5 > 0).sum())
                both_empty += int(((counts34 == 0) & (counts5 == 0)).sum())
                n_distance_atoms += int(counts34.size)
                change = aux["change"].detach().float().cpu().norm(dim=1).numpy()
                change_l2.extend(change.tolist())
                for g in range(len(rows)):
                    mask = (atom_batch == g).numpy()
                    graph_change.append(float(change[mask].max()) if mask.any() else 0.0)
            elif family.startswith("path"):
                counts = aux["path_counts"].detach().cpu().numpy()
                n_path_atoms += int((counts > 0).sum())
                path_type_counts.extend(counts.tolist())
                change = aux["change"].detach().float().cpu().norm(dim=1).numpy()
                change_l2.extend(change.tolist())
                # bind-gap only for roots with paths
                bind = aux["bind"].detach().float().cpu()
                if aux.get("mean_v") is not None and aux.get("mean_r") is not None:
                    gap = (bind - aux["mean_v"].detach().float().cpu() * aux["mean_r"].detach().float().cpu()).norm(dim=1).numpy()
                    bind_gap_norm.extend(gap[counts > 0].tolist())
                for g in range(len(rows)):
                    mask = (atom_batch == g).numpy()
                    graph_change.append(float(change[mask].max()) if mask.any() else 0.0)
                # multi path-type roots within this batch
                path_atom = batch["path_atom"].detach().cpu()
                path_cat = batch["path_cat"].detach().cpu()
                for root in torch.unique(path_atom).tolist():
                    cats = path_cat[path_atom == int(root)].unique()
                    if cats.numel() > 1:
                        n_multi_type_atoms += 1
    change_arr = np.asarray(change_l2, dtype=np.float64)
    out: dict[str, Any] = {
        "n_molecules": n,
        "n_atoms": n_atoms,
        "n_hetero_atoms": n_hetero,
        "n_atoms_with_correction_path": int(np.sum(change_arr > 0)),
        "change_l2_mean": float(change_arr.mean()) if change_arr.size else 0.0,
        "change_l2_p95": float(np.quantile(change_arr, 0.95)) if change_arr.size else 0.0,
        "change_l2_max": float(change_arr.max()) if change_arr.size else 0.0,
        "graph_max_change_mean": float(np.mean(graph_change)) if graph_change else 0.0,
        "graph_max_change_max": float(np.max(graph_change)) if graph_change else 0.0,
    }
    if family == "g2l_scalar":
        da = np.abs(np.asarray(delta_abs, dtype=np.float64))
        out.update(
            {
                "delta_abs_mean": float(da.mean()) if da.size else 0.0,
                "delta_abs_p95": float(np.quantile(da, 0.95)) if da.size else 0.0,
                "delta_abs_max": float(da.max()) if da.size else 0.0,
                "graph_max_delta_mean": float(np.mean(graph_delta)) if graph_delta else 0.0,
            }
        )
    if family.startswith("dist"):
        out.update(
            {
                "n_distance_atoms": n_distance_atoms,
                "fraction_d34_nonempty": float(d34_nonempty / max(n_distance_atoms, 1)),
                "fraction_d5_nonempty": float(d5_nonempty / max(n_distance_atoms, 1)),
                "fraction_both_empty": float(both_empty / max(n_distance_atoms, 1)),
            }
        )
    if family.startswith("path"):
        out.update(
            {
                "n_atoms_with_paths": n_path_atoms,
                "fraction_atoms_with_paths": float(n_path_atoms / max(n_atoms, 1)),
                "n_roots_with_multiple_path_types": n_multi_type_atoms,
                "bind_gap_norm_mean": float(np.mean(bind_gap_norm)) if bind_gap_norm else 0.0,
                "bind_gap_norm_p95": float(np.quantile(bind_gap_norm, 0.95)) if bind_gap_norm else 0.0,
                "bind_gap_norm_max": float(np.max(bind_gap_norm)) if bind_gap_norm else 0.0,
                "bind_gap_note": "norm(mean(v*r) - mean(v)*mean(r)) per root with paths",
            }
        )
    return out



def rho_input_audit(model, cache, rows, atol: float = 1.0e-5) -> dict[str, Any]:
    """Check that rho receives statistics of the original h, not h_new."""
    model.eval()
    device = M.model_device(model)
    batch = M.to_device(D.collate(cache, [int(r) for r in rows]), device)
    captured_h, captured_rho = [], []
    h_local = model.local.register_forward_hook(lambda module, inputs, output: captured_h.append(output.detach().clone()))
    rho_pre = model.rho.register_forward_pre_hook(lambda module, inputs: captured_rho.append(inputs[0].detach().clone()))
    try:
        with torch.no_grad():
            _y, aux = model(batch, return_aux=True)
    finally:
        h_local.remove()
        rho_pre.remove()
    h = captured_h[0]
    actual = captured_rho[0]
    n_graphs = int(batch["n_graphs"])
    atom_batch = batch["atom_batch"]
    width = h.shape[1]
    count = h.new_zeros(n_graphs).index_add_(0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype))
    mean_h = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h) / count.clamp(min=1.0).unsqueeze(1)
    mean_h2 = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h * h) / count.clamp(min=1.0).unsqueeze(1)
    g = (batch["global"] - model.global_mean) / model.global_scale
    expected = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
    h_new = aux["h_new"]
    mean_h_new = h_new.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h_new) / count.clamp(min=1.0).unsqueeze(1)
    mean_h2_new = h_new.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h_new * h_new) / count.clamp(min=1.0).unsqueeze(1)
    expected_new = torch.cat([g, mean_h_new, mean_h2_new, torch.log1p(count).unsqueeze(1)], dim=1)
    diff = float((actual - expected).abs().max())
    change_scale = float((h_new - h).abs().max())
    diff_new = float((actual - expected_new).abs().max())
    return {
        "n_graphs": len(rows),
        "atol": float(atol),
        "max_abs_diff_vs_h_original_reconstruction": diff,
        "max_abs_diff_vs_h_new_reconstruction": diff_new,
        "max_abs_h_new_minus_h_original": change_scale,
        "rho_input_matches_h_original": bool(diff <= atol),
        "rho_input_vs_h_new_reconstruction_differs": bool(diff_new > atol),
        "discriminating_or_scalar_unchanged": bool(change_scale == 0.0 or diff_new > atol),
        "passed": bool(diff <= atol and (change_scale == 0.0 or diff_new > atol)),
    }


def dropout_activity_audit(model, cache, rows, generator_seed: int = 0) -> dict[str, Any]:
    """DropoutAtLocal must mask in train mode and be the identity in eval mode."""
    device = M.model_device(model)
    batch = M.to_device(D.collate(cache, [int(r) for r in rows]), device)
    head = getattr(model, "_v2_dropout_head", None)
    if head is None:
        return {"hook_module_present": False, "passed": False}

    def _run(mode: str):
        pairs = []
        handle = head.register_forward_hook(lambda module, inputs, output: pairs.append((inputs[0].detach(), output.detach())))
        try:
            torch.manual_seed(int(generator_seed) + (12_345 if mode == "train" else 54_321))
            with torch.no_grad():
                y1 = model(batch).detach().float().cpu().numpy().reshape(-1)
            torch.manual_seed(int(generator_seed) + (99_999 if mode == "train" else 87_654))
            with torch.no_grad():
                y2 = model(batch).detach().float().cpu().numpy().reshape(-1)
        finally:
            handle.remove()
        return pairs, y1, y2

    model.train()
    head_training_train = bool(head.training)
    train_pairs, y_train_1, y_train_2 = _run("train")
    model.eval()
    head_training_eval = bool(head.training)
    eval_pairs, y_eval_1, y_eval_2 = _run("eval")

    def _max_diff(pairs):
        return max((float((a - b).abs().max()) for a, b in pairs), default=0.0)

    train_identity = [bool(torch.equal(a, b)) for a, b in train_pairs] if train_pairs else [True]
    eval_identity = [bool(torch.equal(a, b)) for a, b in eval_pairs] if eval_pairs else [True]
    train_forward_diff = float(np.abs(y_train_1 - y_train_2).max())
    eval_forward_diff = float(np.abs(y_eval_1 - y_eval_2).max())
    passed = bool(
        float(head.p) == C.TRAIN_RECIPE["dropout"]
        and head_training_train
        and not head_training_eval
        and train_pairs
        and not all(train_identity)
        and train_forward_diff > 0.0
        and eval_pairs
        and all(eval_identity)
    )
    return {
        "dropout_p": float(head.p),
        "hook_module_present": True,
        "head_training_in_train_mode": head_training_train,
        "head_training_in_eval_mode": head_training_eval,
        "head_calls_train": len(train_pairs),
        "head_calls_eval": len(eval_pairs),
        "train_mode_masked_atoms": bool(not all(train_identity)),
        "train_mode_max_abs_dropout_delta": _max_diff(train_pairs),
        "train_mode_forward_max_abs_diff_different_masks": train_forward_diff,
        "eval_mode_is_identity": bool(all(eval_identity)),
        "eval_mode_max_abs_dropout_delta": _max_diff(eval_pairs),
        "eval_mode_forward_max_abs_diff": eval_forward_diff,
        "passed": passed,
    }

def train_arm(
    *,
    arm_key: str,
    cfg: Mapping[str, Any],
    views: Mapping[str, Any],
    device: torch.device,
    run_dir: Path,
    limit_train: int | None = None,
    limit_valid: int | None = None,
    max_epochs: int | None = None,
    soup_last: int | None = None,
    pool: str = "local",
    smoke: bool = False,
) -> dict[str, Any]:
    if arm_key not in C.ARM_SPECS:
        raise ValueError(f"unknown arm {arm_key!r}")
    arm_spec = C.ARM_SPECS[arm_key]
    family = str(arm_spec["family"])
    seed = int(arm_spec["seed"])
    recipe = dict(cfg["training"])
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    spec = views["spec"]
    train_cache = views["train_cache"]
    valid_cache = views["valid_cache"]
    labels = views["labels"]
    split = views["split"]
    if limit_train is not None:
        train_cache = D.build_distance_buckets(
            R42D.select_cache(train_cache, list(range(min(int(limit_train), int(train_cache["n_molecules"])))), split_label="smoke_train")
        )
        views = {**views, "train_cache": train_cache}
    if limit_valid is not None:
        valid_cache = D.build_distance_buckets(
            R42D.select_cache(valid_cache, list(range(min(int(limit_valid), int(valid_cache["n_molecules"])))), split_label="smoke_valid")
        )
        views = {**views, "valid_cache": valid_cache}
    n_train = int(train_cache["n_molecules"])
    n_valid = int(valid_cache["n_molecules"])
    if limit_train is not None:
        train_norm = lcrc_data.fit_norm(train_cache, spec, source="smoke_train_structures")
        target_norm = R42D.fit_target_scaler(labels["y_train"][:n_train])
    else:
        train_norm = views["train_norm"]
        target_norm = views["target_norm"]
    epochs = int(max_epochs if max_epochs is not None else recipe["epochs"])
    soup_k = int(soup_last if soup_last is not None else C.SOUP_LAST)
    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(n_train / batch_size)

    model, meta = M.build_family(family, spec, train_norm, seed, device)
    init_hash = C.state_dict_fingerprint(model.state_dict())
    new_names = M.new_parameter_names(model, family)
    prefix = M.new_parameter_prefix(family)
    init_new_params = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if name.startswith(prefix)}
    params_total = M.parameter_count(model)
    params_base = int(meta.get("parameters_base", params_total))

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(recipe["peak_lr"]),
        betas=tuple(float(x) for x in recipe["adam_betas"]),
        eps=float(recipe["adam_eps"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps_per_epoch * epochs, eta_min=float(recipe["min_lr"])
    )
    generator = torch.Generator().manual_seed(seed + C.SHUFFLE_OFFSET)
    y_train_all = np.asarray(labels["y_train"][:n_train], dtype=np.float64)
    y_train_std = NT.target_standardise(y_train_all, target_norm)
    y_valid = np.asarray(labels["y_valid"][:n_valid], dtype=np.float64)

    snapshots: list[dict[str, torch.Tensor]] = []
    curve: list[dict[str, Any]] = []
    batch_order: list[list[int]] = []
    clip_events = 0
    grad_norm_max = 0.0
    probe_steps: list[dict[str, Any]] = []
    probe_agg = {name: {"min": math.inf, "max": -math.inf, "sum": 0.0, "n": 0} for name in new_names}
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    gpu_peak = 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(n_train, generator=generator).tolist()
        batches = [perm[i : i + batch_size] for i in range(0, len(perm), batch_size)]
        batch_order.extend(batches)
        abs_sum = 0.0
        sq_sum = 0.0
        seen = 0
        epoch_clips = 0
        epoch_gnorm_max = 0.0
        for step_index, rows in enumerate(batches):
            batch = M.to_device(D.collate(train_cache, rows), device)
            y = y_train_std[rows].to(device)
            pred = model(batch)
            loss = F.mse_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip"])))
            if not math.isfinite(float(loss)) or not math.isfinite(gnorm):
                raise RuntimeError(f"non-finite loss/gradient at epoch {epoch} step {step_index + 1}")
            if gnorm > float(recipe["gradient_clip"]):
                clip_events += 1
                epoch_clips += 1
            grad_norm_max = max(grad_norm_max, gnorm)
            epoch_gnorm_max = max(epoch_gnorm_max, gnorm)
            if new_names:
                norms = _grad_norms(model, new_names)
                global_step = (epoch - 1) * steps_per_epoch + step_index + 1
                if global_step <= 5:
                    probe_steps.append({"global_step": global_step, "grad_norm": norms})
                for name, value in norms.items():
                    probe_agg[name]["min"] = min(probe_agg[name]["min"], value)
                    probe_agg[name]["max"] = max(probe_agg[name]["max"], value)
                    probe_agg[name]["sum"] += value
                    probe_agg[name]["n"] += 1
            optimizer.step()
            scheduler.step()
            err = (pred.view(-1) - y.view(-1)).detach()
            abs_sum += float(err.abs().sum())
            sq_sum += float((err * err).sum())
            seen += int(err.numel())
        snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if len(snapshots) > soup_k:
            snapshots.pop(0)

        valid_pred = NT.denormalise(M.predict(model, valid_cache, device), target_norm)
        valid_metrics = MET.all_metrics(valid_pred, y_valid)
        curve.append(
            {
                "epoch": epoch,
                "steps": steps_per_epoch,
                "train_loss_standardized": float(sq_sum / max(seen, 1)),
                "train_mae_standardized": float(abs_sum / max(seen, 1)),
                "train_rmse_standardized": math.sqrt(float(sq_sum / max(seen, 1))),
                "valid_rmse_raw": valid_metrics["rmse"],
                "valid_mae_raw": valid_metrics["mae"],
                "valid_r2_raw": valid_metrics["r2"],
                "lr": float(optimizer.param_groups[0]["lr"]),
                "grad_norm_max": float(epoch_gnorm_max),
                "grad_clip_events": int(epoch_clips),
                "seconds": time.perf_counter() - t0,
            }
        )
        if device.type == "cuda":
            gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))

    wall = time.perf_counter() - started
    soup_members = [int(e) for e in range(epochs - len(snapshots) + 1, epochs + 1)]
    soup = _average_states(snapshots)
    soup_hash = C.state_dict_fingerprint(soup)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    model.eval()

    train_pred = NT.denormalise(M.predict(model, train_cache, device), target_norm)
    train_metrics = MET.all_metrics(train_pred, y_train_all)
    valid_pred = NT.denormalise(M.predict(model, valid_cache, device), target_norm)
    valid_metrics = MET.all_metrics(valid_pred, y_valid)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_inf = time.perf_counter()
    for _ in range(3):
        M.predict(model, valid_cache, device, batch_size=256)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = (time.perf_counter() - t_inf) / 3.0
    if device.type == "cuda":
        gpu_peak = max(gpu_peak, int(torch.cuda.max_memory_allocated(device)))

    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "state_dict": soup,
            "model": C.ARM_MODEL_NAME[family],
            "arm": arm_key,
            "family": family,
            "seed": seed,
            "run_key": arm_key,
            "protocol_id": C.PROTOCOL_ID,
            "soup_members": soup_members,
            "soup_state_hash": soup_hash,
            "init_state_hash": init_hash,
        },
        checkpoint_path,
    )
    np.savez(
        run_dir / "train_predictions.npz",
        rows=np.asarray(split["rows"]["train"][:n_train], dtype=np.int64),
        pred=train_pred,
        y_true=np.asarray(y_train_all, dtype=np.float64),
    )
    np.savez(
        run_dir / "valid_predictions.npz",
        rows=np.asarray(split["rows"]["valid"][:n_valid], dtype=np.int64),
        pred=valid_pred,
        y_true=np.asarray(y_valid, dtype=np.float64),
    )
    with open(run_dir / "curve.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)

    diagnostics: dict[str, Any] = {}
    diagnostics["rho_after_soup"] = rho_input_audit(model, valid_cache, list(range(min(8, n_valid))))
    diagnostics["dropout"] = dropout_activity_audit(model, valid_cache, list(range(min(8, n_valid))))
    diagnostics["correction_usage_valid"] = correction_diagnostics(model, family, valid_cache, device)
    diagnostics["correction_usage_train"] = correction_diagnostics(
        model, family, train_cache, device, batch_size=128
    )
    disabled_valid = NT.denormalise(predict_with_correction_disabled(model, valid_cache, family, device), target_norm)
    disabled_metrics = MET.all_metrics(disabled_valid, y_valid)
    diagnostics["w_out_zero_ablation_valid"] = {
        "rmse": disabled_metrics["rmse"],
        "mae": disabled_metrics["mae"],
        "r2": disabled_metrics["r2"],
        "mean_abs_prediction_change_vs_full": float(np.abs(valid_pred - disabled_valid).mean()),
        "max_abs_prediction_change_vs_full": float(np.abs(valid_pred - disabled_valid).max()),
        "rmse_change_vs_full": float(disabled_metrics["rmse"] - valid_metrics["rmse"]),
        "note": "same trained model with the correction output projection zeroed in eval (its own base path)",
    }
    diagnostics["trained_new_parameter_delta_from_init"] = {
        name: float((p.detach().cpu() - init_new_params[name]).abs().max())
        for name, p in model.named_parameters()
        if name.startswith(prefix)
    }
    if new_names:
        diagnostics["gradient_probe"] = gradient_probe(model, valid_cache, new_names, device)
        diagnostics["gradient_probe"]["training_probe"] = {
            "first_five_steps": probe_steps,
            "aggregate": {
                name: {
                    "min": vals["min"],
                    "max": vals["max"],
                    "mean": vals["sum"] / max(vals["n"], 1),
                    "n_steps": vals["n"],
                    "all_nonzero": bool(vals["min"] > 0.0),
                }
                for name, vals in probe_agg.items()
            },
        }
    if family.startswith("dist"):
        diagnostics["distance_coverage_valid"] = D.distance_bucket_coverage(valid_cache)
        diagnostics["distance_coverage_train"] = D.distance_bucket_coverage(train_cache)
    if family.startswith("path"):
        diagnostics["path_coverage_valid"] = {
            "fraction_atoms_with_paths": diagnostics["correction_usage_valid"]["fraction_atoms_with_paths"],
            "n_roots_with_multiple_path_types": diagnostics["correction_usage_valid"]["n_roots_with_multiple_path_types"],
            "bind_gap_norm_mean": diagnostics["correction_usage_valid"]["bind_gap_norm_mean"],
            "bind_gap_norm_p95": diagnostics["correction_usage_valid"]["bind_gap_norm_p95"],
        }

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "arm": arm_key,
        "run_key": arm_key,
        "family": family,
        "model": C.ARM_MODEL_NAME[family],
        "seed": seed,
        "split_seed": C.SPLIT_SEED,
        "smoke": bool(smoke),
        "pool": str(pool),
        "device": str(device),
        "device_note": "rr sets CUDA_VISIBLE_DEVICES per pool; the pool GPU is cuda:0 in-process",
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "precision": "fp32",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": __import__("platform").python_version(),
        "hostname": __import__("platform").node(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "config_hash": C.sha256_json(dict(cfg)),
        "data": {
            "split_file": C.track_rel(C.V1_SPLIT_FILE),
            "split_file_sha256": C.V1_SPLIT_SHA256,
            "labels_file": C.track_rel(C.V1_LABELS_FILE),
            "labels_file_sha256": C.V1_LABELS_SHA256,
            "train_rows_sha256": C.rows_sha256_int64(split["rows"]["train"][:n_train]),
            "valid_rows_sha256": C.rows_sha256_int64(split["rows"]["valid"][:n_valid]),
            "n_train": n_train,
            "n_valid": n_valid,
            "lcrc_cache_path": C.track_rel(views["cache_built"]["path"]),
            "lcrc_cache_sha256": views["cache_built"]["file_sha256"],
            "lcrc_cache_content_fingerprint": views["cache_built"]["content_fingerprint"],
            "lcrc_cache_labels_present": False,
            "feature_spec_hash": spec.hash(),
            "train_norm_fingerprint": lcrc_data.norm_fingerprint(train_norm),
            "target_norm": dict(target_norm),
            "label_audit": dict(labels["audit"]),
            "distance_bucket_meta": dict(train_cache.get("distance_bucket_meta", {})),
            "test_row_values_used": False,
            "test_labels_read": False,
            "test_predictions_read": False,
            "test_metrics_read": False,
            "label_scope": "train3360_and_valid420_only",
        },
        "model_audit": {
            "parameters": int(params_total),
            "parameters_base": int(params_base),
            "parameters_new": int(params_total - params_base),
            "parameter_breakdown": M.parameter_breakdown(model),
            "init_state_hash": init_hash,
            "soup_state_hash": soup_hash,
            "shared_init": meta.get("shared_init"),
            "init_parity_with_context": meta.get("init_parity_with_context"),
            "dropout_hook_calls": int(meta["dropout_hook"].calls),
            "definition": meta.get("definition"),
        },
        "recipe": {
            "optimizer": "AdamW",
            "adam_betas": [float(x) for x in recipe["adam_betas"]],
            "adam_eps": float(recipe["adam_eps"]),
            "peak_lr": float(recipe["peak_lr"]),
            "min_lr": float(recipe["min_lr"]),
            "scheduler": "per_step_cosine",
            "weight_decay": float(recipe["weight_decay"]),
            "batch_size": batch_size,
            "epochs_planned": epochs,
            "epochs_run": len(curve),
            "steps_per_epoch": steps_per_epoch,
            "optimizer_steps": int(len(curve) * steps_per_epoch),
            "gradient_clip": float(recipe["gradient_clip"]),
            "dropout": float(recipe["dropout"]),
            "dropout_position": "final atom-level local h (v2 DropoutAtLocal on model.local)",
            "dropout_rng_reseeded_after_init": True,
            "loss": "mse_on_train3360_standardised_target",
            "checkpoint_rule": str(recipe["checkpoint"]),
            "soup_last": soup_k,
            "soup_members": soup_members,
            "early_stopping": False,
            "validation_used_for_checkpoint_selection": False,
            "batch_order_sha256": NT.batch_order_hash(batch_order),
            "seed": seed,
            "split_seed": C.SPLIT_SEED,
            "shuffle_generator_seed": seed + C.SHUFFLE_OFFSET,
        },
        "metrics": {
            "train": {**train_metrics, "units": "raw_logD", "checkpoint": "soup_last_5"},
            "valid": {**valid_metrics, "units": "raw_logD", "checkpoint": "soup_last_5"},
            "train_loss_standardized_last_epoch": float(curve[-1]["train_loss_standardized"]),
            "train_mae_standardized_last_epoch": float(curve[-1]["train_mae_standardized"]),
        },
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_norm_max": float(grad_norm_max),
            "nonfinite_events": 0,
        },
        "timing": {
            "train_wall_seconds": float(wall),
            "process_wall_seconds": float(time.perf_counter() - PROCESS_T0),
            "inference_seconds_per_valid420": float(inference_seconds),
            "inference_batch_size": 256,
            "inference_device": str(device),
        },
        "gpu": {
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "peak_memory_allocated_bytes": int(gpu_peak) if device.type == "cuda" else 0,
        },
        "diagnostics": diagnostics,
        "curve": curve,
        "checkpoint_path": C.track_rel(checkpoint_path),
        "checkpoint_sha256": C.sha256_file(checkpoint_path),
        "predictions": {
            "valid_path": C.track_rel(run_dir / "valid_predictions.npz"),
            "valid_sha256": C.sha256_file(run_dir / "valid_predictions.npz"),
            "train_path": C.track_rel(run_dir / "train_predictions.npz"),
            "train_sha256": C.sha256_file(run_dir / "train_predictions.npz"),
            "curve_path": C.track_rel(run_dir / "curve.csv"),
            "curve_sha256": C.sha256_file(run_dir / "curve.csv"),
        },
        "label_scope": "train3360_and_valid420_only",
        "test_labels_read": False,
        "test_predictions_read": False,
        "test_metrics_read": False,
    }
    C.write_json(run_dir / "train.json", payload)
    C.write_json(run_dir / "diagnostics.json", diagnostics)
    C.log(
        f"[{arm_key}] family={family} seed={seed} params={params_total} epochs={len(curve)} "
        f"train={train_metrics['rmse']:.5f} valid={valid_metrics['rmse']:.5f} wall={wall:.1f}s"
    )
    return payload


__all__ = ["train_arm", "PROCESS_T0", "predict_with_correction_disabled", "gradient_probe", "correction_diagnostics"]
