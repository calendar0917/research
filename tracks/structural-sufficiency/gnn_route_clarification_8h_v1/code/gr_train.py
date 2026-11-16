"""Frozen training / every-epoch evaluation loop for every unit of this round.

One loop, three readout families, two recipe packs.  The loop is copied
semantically from the previous round's ``cr_train`` R3 loop (same seed, same
``np.random.default_rng(0).permutation`` batch order, same every-epoch full
Train/Val/Test eval, same selection rule) and extended only where the task brief
requires it (``Rf`` warmup, the additive head, the frozen head D, oracle
targets).

Nothing here selects on Test: epoch and recipe selection use Val only.
"""

from __future__ import annotations

import shutil
import time
from typing import Any, Mapping, Sequence

import numpy as np

import gr_common as C
import gr_data as D
import gr_models as M
import gr_select as S


# ---------------------------------------------------------------------------
# target attachment
# ---------------------------------------------------------------------------


def _attach_target(batch: dict[str, Any], chunk: np.ndarray, y_std: np.ndarray, device=None) -> None:
    import torch

    batch["y"] = torch.as_tensor(np.asarray(y_std, dtype=np.float32)[chunk], dtype=torch.float32, device=device)


def _to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {key: (value.to(device) if hasattr(value, "to") else value) for key, value in batch.items()}


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def evaluate(model, b, split: str, device, *, with_parts: bool = False) -> dict[str, Any]:
    import torch

    order, row_ids, targets_raw = D.eval_order(b, split)
    y_std = (targets_raw - b.scaler["mean"]) / b.scaler["std"]
    model.eval()
    preds: list[np.ndarray] = []
    parts_store: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, order.size, C.EVAL_BATCH_SIZE):
            chunk = order[start : start + C.EVAL_BATCH_SIZE]
            batch = _to_device(D.collate(b, split, chunk), device)
            _attach_target(batch, chunk, y_std, device)
            if with_parts:
                parts = model.parts(batch)
                pred = parts["pred_std"]
                block = {"chunk": np.asarray(chunk, dtype=np.int64)}
                for key in ("h", "s_v", "m_v", "a_v", "u_terms", "sum_s", "mean_m", "u_sum", "intercept", "contrib_atom", "rho", "pred_std"):
                    if key in parts and torch.is_tensor(parts[key]):
                        block[key] = parts[key].detach().cpu().numpy().astype(np.float64)
                dec = batch.get("atom_counts")
                if dec is not None:
                    block["atom_counts"] = dec.detach().cpu().numpy()
                parts_store.append(block)
            else:
                pred = model(batch)
            preds.append(pred.detach().cpu().numpy().astype(np.float64).reshape(-1))
    pred_std = np.concatenate(preds) if preds else np.zeros(0)
    mean, std = float(b.scaler["mean"]), float(b.scaler["std"])
    pred_raw = pred_std * std + mean
    out = {
        "split": split,
        "row_ids": row_ids,
        "y_raw": targets_raw,
        "pred_std": pred_std,
        "pred_raw": pred_raw,
        "rmse_raw": C.rmse(pred_raw, targets_raw),
        "mae_raw": C.mae(pred_raw, targets_raw),
        "seconds": time.perf_counter() - t0,
        "n_rows": int(row_ids.size),
    }
    if with_parts:
        out["parts"] = parts_store
    return out


# ---------------------------------------------------------------------------
# task identity
# ---------------------------------------------------------------------------


def task_signature(task: Mapping[str, Any], b, *, init: Mapping[str, Any] | None = None) -> dict[str, Any]:
    gine = D.gine_cache(b.dataset) if task["dataset"] in C.DATASETS else None
    payload: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "task": task["key"],
        "stage": task["stage"],
        "dataset": task["dataset"],
        "model": task["model"],
        "recipe": task["recipe"],
        "epochs": int(task["epochs"]),
        "seed": C.TRAIN_SEED,
        "batch_size": C.BATCH_SIZE,
        "eval_batch_size": C.EVAL_BATCH_SIZE,
        "dropout": C.DROPOUT,
        "grad_clip": C.GRAD_CLIP,
        "lr_recipe": C.RECIPES.get(task["recipe"], task["recipe"]),
        "min_lr": C.MIN_LR,
        "adam_betas": list(C.ADAM_BETAS),
        "adam_eps": C.ADAM_EPS,
        "n_train": int(b.caches["train"]["n_molecules"]) if hasattr(b, "caches") else None,
        "graph_cache_sha256": b.payload.get("cache_sha256"),
        "lipo_raw_sha256": b.lipo_raw.get("cache_sha256"),
        "feature_spec_hash": b.extra.get("spec_hash"),
        "norm_fingerprint": b.extra.get("norm_fingerprint"),
        "target_scaler": {k: b.scaler[k] for k in ("mean", "std")},
        "active_global": {
            "n_active": int(b.active_global["n_active"]),
            "indices_sha256": C.sha256_array(np.asarray(b.active_global["active_indices"], dtype=np.int64)),
        },
        "row_id_sha256": {split: C.sha256_array(b.global_row_ids[split]) for split in C.SPLITS},
        "code_sha256": C.code_hashes(),
        "init": init or {},
    }
    if gine is not None:
        payload["category_counts"] = gine.category_counts()
    payload["signature"] = C.sha256_json({k: v for k, v in payload.items() if k != "signature"})
    return payload


def _status_path(out_dir) -> Any:
    return out_dir / "status.json"


def _completed(out_dir, signature: str) -> dict[str, Any] | None:
    path = _status_path(out_dir)
    if not path.is_file():
        return None
    try:
        status = C.read_json(path)
    except Exception:
        return None
    if status.get("state") == "completed" and status.get("signature") == signature:
        return C.read_json(out_dir / "trajectory.json")
    return None


# ---------------------------------------------------------------------------
# generic trajectory
# ---------------------------------------------------------------------------


def _save_checkpoint(path, model, payload: Mapping[str, Any]) -> None:
    import torch

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model_state": model.state_dict(), **payload}, tmp)
    tmp.replace(path)


def run_trajectory(
    *,
    task: Mapping[str, Any],
    b,
    model,
    init_audit: Mapping[str, Any],
    out_dir,
    device,
    epochs: int,
    recipe: str | None,
    resume: bool = True,
    evaluate_real: bool = True,
) -> dict[str, Any]:
    """Train ``model`` for a fixed number of epochs and evaluate three splits."""
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    signature = task_signature(task, b, init=init_audit)
    sig = signature["signature"]
    done = _completed(out_dir, sig) if resume else None
    if done is not None:
        done["resumed_skip"] = True
        return done

    model = model.to(device)
    params = M.parameter_report(model)
    n_train = int(b.caches["train"]["n_molecules"])
    train_rows = np.arange(n_train, dtype=np.int64)
    spe = C.steps_per_epoch(n_train)
    total = int(epochs) * spe
    cfg = C.RECIPES[recipe] if recipe else None
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["peak_lr"]) if cfg else C.MIN_LR,
        betas=tuple(C.ADAM_BETAS),
        eps=float(C.ADAM_EPS),
        weight_decay=float(cfg["weight_decay"]) if cfg else 0.0,
    )

    state: dict[str, Any] = {
        "epoch": 0,
        "global_step": 0,
        "curves": [],
        "best": {"valid_rmse_raw": float("inf"), "epoch": None},
        "final": None,
        "batch_order_hashes": [],
        "first_batch_rows": [],
        "train_seconds": 0.0,
        "eval_seconds": 0.0,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    best_state = None
    resume_path = out_dir / "resume_state.pt"
    rng = np.random.default_rng(C.TRAIN_SEED)
    optimizer_state = None
    if resume and resume_path.is_file():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("signature") == sig:
            model.load_state_dict(saved["model_state"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state"])
            state = saved["state"]
            best_state = saved.get("best_state")
            rng.bit_generator.state = saved["np_rng_state"]
            C.restore_rng_payload(saved["rng_payload"])
            optimizer_state = True
        else:
            raise RuntimeError(f"{task['key']}: resume signature mismatch; refusing to mix trajectories")
    if optimizer_state is None:
        C.set_global_seeds(C.TRAIN_SEED)  # reset the training RNG after construction

    state.setdefault("environment", C.environment())
    device_label = str(device)
    t0 = time.perf_counter()
    for epoch in range(state["epoch"] + 1, int(epochs) + 1):
        t_epoch = time.perf_counter()
        model.train()
        order = rng.permutation(train_rows)
        state["batch_order_hashes"].append(C.sha256_array(order))
        if not state["first_batch_rows"]:
            state["first_batch_rows"] = [int(v) for v in order[: C.BATCH_SIZE].tolist()]
        t_train = time.perf_counter()
        y_std = b.y_std["train"]
        for start in range(0, order.size, C.BATCH_SIZE):
            chunk = order[start : start + C.BATCH_SIZE]
            batch = _to_device(D.collate(b, "train", chunk), device)
            _attach_target(batch, chunk, y_std, device)
            lr = C.lr_at(recipe, state["global_step"], n_train, int(epochs)) if recipe else C.MIN_LR
            optimizer.param_groups[0]["lr"] = float(lr)
            state["global_step"] += 1
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(batch), batch["y"].view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(C.GRAD_CLIP))
            optimizer.step()
        state["train_seconds"] += time.perf_counter() - t_train
        metrics: dict[str, Any] = {"epoch": epoch}
        preds_best_epoch: dict[str, Any] = {}
        preds_epoch: dict[str, Any] = {}
        t_eval = time.perf_counter()
        for split in C.SPLITS:
            out = evaluate(model, b, split, device)
            metrics[f"{split}_rmse_raw"] = out["rmse_raw"]
            metrics[f"{split}_mae_raw"] = out["mae_raw"]
            preds_epoch[split] = out
        state["eval_seconds"] += time.perf_counter() - t_eval
        metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        metrics["seconds"] = time.perf_counter() - t_epoch
        metrics["train_seconds"] = state["train_seconds"]
        state["curves"].append(metrics)
        if metrics["valid_rmse_raw"] < state["best"]["valid_rmse_raw"]:
            state["best"] = {"valid_rmse_raw": metrics["valid_rmse_raw"], "epoch": epoch}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            preds_best_epoch = {split: _pred_payload(out) for split, out in preds_epoch.items()}
            _save_checkpoint(
                out_dir / "checkpoint_best_val.pt",
                model,
                {"epoch": epoch, "signature": sig, "valid_rmse_raw": metrics["valid_rmse_raw"]},
            )
            C.atomic_write_npz(out_dir / "best_predictions.npz", {
                f"{split}_{key}": value
                for split, payload in preds_best_epoch.items()
                for key, value in payload.items()
            })
        state["final"] = {split: _pred_payload(out) for split, out in preds_epoch.items()}
        C.atomic_write_npz(out_dir / "final_predictions.npz", {
            f"{split}_{key}": value for split, payload in state["final"].items() for key, value in payload.items()
        })
        state["epoch"] = epoch
        if epoch % 5 == 0 or epoch == int(epochs):
            _save_resume(
                resume_path, model, optimizer, state, best_state, sig,
                np_rng_state=rng.bit_generator.state, rng_payload=C.training_rng_payload(),
            )
            _write_curve_csv(out_dir / "curve.csv", state["curves"])
            _save_checkpoint(out_dir / "checkpoint_final.pt", model, {"epoch": epoch, "signature": sig})

    best_epoch = int(state["best"]["epoch"])
    best_row = state["curves"][best_epoch - 1]
    last20 = {
        f"{split}_rmse_raw": float(np.mean([row[f"{split}_rmse_raw"] for row in state["curves"][-20:]]))
        for split in C.SPLITS
    }
    last20.update(
        {
            f"{split}_mae_raw": float(np.mean([row[f"{split}_mae_raw"] for row in state["curves"][-20:]]))
            for split in C.SPLITS
        }
    )
    elapsed = time.perf_counter() - t0
    summary = {
        "task": dict(task),
        "dataset": task["dataset"],
        "model": task["model"],
        "recipe": recipe,
        "signature": sig,
        "epochs": int(epochs),
        "steps_per_epoch": spe,
        "total_steps": state["global_step"],
        "first_batch_rows": state["first_batch_rows"],
        "batch_order_sha256": C.sha256_json(state["batch_order_hashes"]),
        "batch_order_hashes_first3": state["batch_order_hashes"][:3],
        "curves": state["curves"],
        "selected": {
            "epoch": best_epoch,
            **{k: best_row[k] for k in best_row if k.startswith(("train_", "valid_", "test_"))},
        },
        "final": {**{k: v for k, v in state["curves"][-1].items() if k.startswith(("train_", "valid_", "test_"))}, "epoch": int(epochs)},
        "last20": last20,
        "parameters": params,
        "init": dict(init_audit),
        "n_parameters": params["n_parameters"],
        "train_seconds": state["train_seconds"],
        "eval_seconds": state["eval_seconds"],
        "seconds_total": elapsed,
        "device": device_label,
        "environment": state["environment"],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resumed_from": None if optimizer_state is None else int(state.get("resumed_epoch", 0)),
    }
    C.write_json(out_dir / "trajectory.json", summary)
    C.write_json(
        out_dir / "status.json",
        {
            "state": "completed",
            "signature": sig,
            "task": task["key"],
            "selected_epoch": best_epoch,
            "selected_valid_rmse_raw": state["best"]["valid_rmse_raw"],
            "finished_at": summary["finished_at"],
        },
    )
    return summary


def _pred_payload(out: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "row_id": np.asarray(out["row_ids"], dtype=np.int64),
        "y_raw": np.asarray(out["y_raw"], dtype=np.float64),
        "pred_raw": np.asarray(out["pred_raw"], dtype=np.float64),
        "pred_std": np.asarray(out["pred_std"], dtype=np.float64),
    }


def _save_resume(path, model, optimizer, state, best_state, sig, **extra) -> None:
    import torch

    payload = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "state": {k: (v if k not in ("final",) else v) for k, v in state.items()},
        "best_state": best_state,
        "signature": sig,
        **extra,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _write_curve_csv(path, curves: Sequence[Mapping[str, Any]]) -> None:
    if not curves:
        return
    keys = ["epoch", "train_rmse_raw", "valid_rmse_raw", "test_rmse_raw", "train_mae_raw", "valid_mae_raw", "test_mae_raw", "lr", "seconds"]
    lines = [",".join(keys)]
    for row in curves:
        lines.append(",".join(str(row.get(key, "")) for key in keys))
    C.atomic_write_text(path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def build_model(task: Mapping[str, Any], b, ms: M.ModelSet, *, trunk_state: Mapping[str, Any] | None = None):
    stage = task["stage"]
    model_name = task["model"]
    if stage == C.STAGE1:
        if model_name == "L":
            return ms.lcrc(), {"kind": "canonical_c1"}
        return ms.dense(model_name), {"kind": f"canonical_{model_name.lower()}"}
    if stage in (C.STAGE2, C.STAGE_ORACLE) and model_name == "A":
        kind = S.selected_trunk_model()
        model = ms.additive(
            kind,
            active_indices=[] if kind == "G0" else b.active_global["active_indices"],
            size_scaler=D.size_scaler(b),
        )
        return model, {
            "kind": f"canonical_{kind.lower()}_trunk_plus_isolated_head",
            "trunk": kind,
            "head_budget": model.head.budget,
        }
    if stage == C.STAGE_ORACLE and model_name in ("L", "T"):
        if model_name == "L":
            return ms.lcrc(), {"kind": "canonical_c1"}
        kind = S.selected_trunk_model()
        return ms.dense(kind), {"kind": f"canonical_{kind.lower()}"}
    raise KeyError(f"no builder for {stage}/{model_name}")


def run_standard_task(task: Mapping[str, Any], *, device: str, b=None) -> dict[str, Any]:
    """stage 1 / stage 2 / oracle: full structural target."""
    if task["stage"] == C.STAGE_ORACLE:
        b = D.oracle_bundle("ogbg-mollipo", task["property"])
    else:
        b = b or D.bundle(task["dataset"])
    ms = M.ModelSet(b)
    model, init_note = build_model(task, b, ms)
    init_audit = {**init_note, "state_hash_before_training": M.model_state_hash(model)}
    if getattr(model, "head", None) is not None and hasattr(model.head, "budget"):
        init_audit["budget"] = model.head.budget
    recipe = task["recipe"]
    if recipe == "selected":
        recipe = S.lipo_recipe_for_model(task["model"])
        init_audit["selected_recipe_from_real_lipo"] = recipe
    out_dir = C.task_dir(task["stage"], task["dataset"], task["model"], task["recipe"])
    summary = run_trajectory(
        task=task,
        b=b,
        model=model,
        init_audit=init_audit,
        out_dir=out_dir,
        device=device,
        epochs=int(task["epochs"]),
        recipe=recipe,
    )
    return summary


# ---------------------------------------------------------------------------
# stage 3: frozen representation head D
# ---------------------------------------------------------------------------


def _t_checkpoint(dataset: str):
    import torch

    t = S.selected_trunk_model()
    recipe = S.selected_recipe(dataset, t)
    path = C.task_dir(C.STAGE1, dataset, t, recipe) / "checkpoint_best_val.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing stage-1 best checkpoint for T={t}: {path}")
    return t, recipe, torch.load(path, map_location="cpu", weights_only=False)


def cache_representation(b, kind: str, state: Mapping[str, Any], ms: M.ModelSet, device: str) -> dict[str, Any]:
    import torch

    trunk = M.build_trunk(kind, b)
    model = M.DenseGraphModel(trunk, M.DenseHead(kind, b.spec.global_dim))
    trunk_state = {
        k[len("trunk.") :]: v
        for k, v in state["model_state"].items()
        if k.startswith("trunk.")
    }
    missing = set(trunk.state_dict()) - set(trunk_state)
    if missing:
        raise RuntimeError(f"T checkpoint trunk missing {sorted(missing)}")
    trunk.load_state_dict(trunk_state, strict=True)
    model = model.to(device).eval()
    out: dict[str, Any] = {}
    for split in C.SPLITS:
        order, row_ids, targets_raw = D.eval_order(b, split)
        hs: list[np.ndarray] = []
        zs: list[np.ndarray] = []
        ns: list[np.ndarray] = []
        teacher: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, order.size, C.EVAL_BATCH_SIZE):
                chunk = order[start : start + C.EVAL_BATCH_SIZE]
                batch = _to_device(D.collate(b, split, chunk), device)
                parts = model.parts(batch)
                hs.append(parts["h"].detach().cpu().numpy().astype(np.float32))
                if kind == "G0":
                    mean, scale = D.size_scaler(b)
                    npy = batch["gine_n_atoms"].detach().cpu().numpy().astype(np.float64)
                    zs.append((((np.log1p(npy) - mean) / scale)).astype(np.float32))
                else:
                    zs.append(trunk.z_global(batch).detach().cpu().numpy().astype(np.float32))
                ns.append(batch["gine_n_atoms"].detach().cpu().numpy().astype(np.int64))
                teacher.append(parts["pred_std"].detach().cpu().numpy().astype(np.float64))
        out[split] = {
            "row_ids": row_ids,
            "y_raw": targets_raw,
            "y_std": (targets_raw - b.scaler["mean"]) / b.scaler["std"],
            "h": np.concatenate(hs, axis=0),
            "z_full": np.concatenate(zs, axis=0),
            "n_atoms": np.concatenate(ns),
            "teacher_std": np.concatenate(teacher),
            "order": order,
        }
    out["kind"] = kind
    return out


def run_stage3(task: Mapping[str, Any], *, device: str) -> dict[str, Any]:
    """Fit the A head on the frozen final representation (teacher = T output)."""
    import torch

    b = D.bundle(task["dataset"])
    t, t_recipe, ckpt = _t_checkpoint(task["dataset"])
    ms = M.ModelSet(b)
    size_scaler = D.size_scaler(b)
    active = [] if t == "G0" else b.active_global["active_indices"]
    # identical head construction to A (same isolated seed, same H)
    head = ms._new_head(t, n_active=(1 if t == "G0" else len(active)), active_indices=active, size_scaler=size_scaler)
    out_dir = C.task_dir(task["stage"], task["dataset"], task["model"], task["recipe"])
    out_dir.mkdir(parents=True, exist_ok=True)
    signature = task_signature(
        task,
        b,
        init={
            "kind": "frozen_T_representation",
            "T": t,
            "T_recipe": t_recipe,
            "T_checkpoint_sha256": C.sha256_file(
                C.task_dir(C.STAGE1, task["dataset"], t, t_recipe) / "checkpoint_best_val.pt"
            ),
            "head_budget": head.budget,
            "active_global": b.active_global["active_indices"] if t == "Gx" else "size_only",
        },
    )
    sig = signature["signature"]
    done = _completed(out_dir, sig)
    if done is not None:
        done["resumed_skip"] = True
        return done
    cache = cache_representation(b, t, ckpt, ms, device)
    z_index = [] if t == "G0" else list(b.active_global["active_indices"])
    epochs = int(task["epochs"])
    n_train = int(cache["train"]["h"].shape[0])
    spe = C.steps_per_epoch(n_train)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, betas=tuple(C.ADAM_BETAS), eps=float(C.ADAM_EPS), weight_decay=0.0)
    head = head.to(device)
    rng = np.random.default_rng(C.TRAIN_SEED)
    C.set_global_seeds(C.TRAIN_SEED)
    curves: list[dict[str, Any]] = []
    best = {"fidelity": float("inf"), "epoch": None}
    best_payload: dict[str, Any] = {}
    state = {"epoch": 0}
    resume_path = out_dir / "resume_state.pt"
    if resume_path.is_file():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved.get("signature") == sig:
            head.load_state_dict(saved["head_state"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state"])
            curves = saved["curves"]
            best = saved["best"]
            best_payload = saved["best_payload"]
            state = saved["state"]
            rng.bit_generator.state = saved["np_rng_state"]
            C.restore_rng_payload(saved["rng_payload"])
    if state["epoch"] == 0:
        C.set_global_seeds(C.TRAIN_SEED)

    def batch_forward(split: str, idx: np.ndarray):
        block = cache[split]
        h = torch.as_tensor(block["h"][idx], dtype=torch.float32, device=device)
        if t == "G0":
            n = torch.as_tensor(block["n_atoms"][idx], dtype=torch.long, device=device)
            z = head.size_z(n)
            atom_batch = torch.repeat_interleave(torch.arange(idx.size, device=device), n)
        else:
            n = torch.as_tensor(block["n_atoms"][idx], dtype=torch.long, device=device)
            z = torch.as_tensor(block["z_full"][idx][:, z_index], dtype=torch.float32, device=device)
            atom_batch = torch.repeat_interleave(torch.arange(idx.size, device=device), n)
        return h, z, n, atom_batch

    for epoch in range(state["epoch"] + 1, epochs + 1):
        t0 = time.perf_counter()
        head.train()
        order = rng.permutation(np.arange(n_train, dtype=np.int64))
        total = epochs * spe
        step = (epoch - 1) * spe
        for start in range(0, order.size, C.BATCH_SIZE):
            idx = order[start : start + C.BATCH_SIZE]
            h, z, n, atom_batch = batch_forward("train", idx)
            progress = min(1.0, max(0.0, step / max(1, total)))
            lr = C.MIN_LR + 0.5 * (1e-3 - C.MIN_LR) * (1.0 + np.cos(np.pi * progress))
            step += 1
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            parts = head.forward_parts(h, atom_batch, n, z)
            teacher = torch.as_tensor(cache["train"]["teacher_std"][idx], dtype=torch.float32, device=device)
            loss = torch.nn.functional.mse_loss(parts["pred_std"], teacher)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), float(C.GRAD_CLIP))
            optimizer.step()
        head.eval()
        row: dict[str, Any] = {"epoch": epoch, "lr": lr}
        payloads: dict[str, Any] = {}
        with torch.no_grad():
            for split in C.SPLITS:
                block = cache[split]
                preds_std = []
                n_rows = int(block["h"].shape[0])
                for start in range(0, n_rows, C.EVAL_BATCH_SIZE):
                    idx = np.arange(start, min(start + C.EVAL_BATCH_SIZE, n_rows), dtype=np.int64)
                    h, z, n, atom_batch = batch_forward(split, idx)
                    preds_std.append(head.forward_parts(h, atom_batch, n, z)["pred_std"].detach().cpu().numpy())
                pred_std = np.concatenate(preds_std).astype(np.float64)
                fid = C.rmse(pred_std, block["teacher_std"])
                pred_raw = pred_std * b.scaler["std"] + b.scaler["mean"]
                row[f"{split}_fidelity_rmse_std"] = fid
                row[f"{split}_rmse_raw"] = C.rmse(pred_raw, block["y_raw"])
                row[f"{split}_mae_raw"] = C.mae(pred_raw, block["y_raw"])
                payloads[split] = {
                    "row_id": block["row_ids"],
                    "y_raw": block["y_raw"],
                    "teacher_std": block["teacher_std"],
                    "pred_std": pred_std,
                    "pred_raw": pred_raw,
                }
        row["seconds"] = time.perf_counter() - t0
        curves.append(row)
        if row["valid_fidelity_rmse_std"] < best["fidelity"]:
            best = {"fidelity": row["valid_fidelity_rmse_std"], "epoch": epoch, "rmse_raw": row["valid_rmse_raw"]}
            best_payload = payloads
            _save_head(out_dir / "checkpoint_best_val.pt", head, {"epoch": epoch, "signature": sig, "valid_fidelity": best["fidelity"]})
            C.atomic_write_npz(
                out_dir / "best_predictions.npz",
                {f"{split}_{key}": value for split, payload in payloads.items() for key, value in payload.items()},
            )
        state["epoch"] = epoch
        if epoch % 5 == 0 or epoch == epochs:
            _save_head_resume(resume_path, head, optimizer, curves, best, best_payload, state, sig, rng=rng)
            _write_curve_csv(out_dir / "curve.csv", curves)

    selected = curves[best["epoch"] - 1] if best["epoch"] else curves[-1]
    summary = {
        "task": dict(task),
        "dataset": task["dataset"],
        "model": "D",
        "recipe": task["recipe"],
        "T": t,
        "T_recipe": t_recipe,
        "signature": sig,
        "epochs": epochs,
        "curves": curves,
        "selected": {
            "epoch": best["epoch"],
            **{k: selected[k] for k in selected if k.startswith(("train_", "valid_", "test_"))},
        },
        "final": {**{k: v for k, v in curves[-1].items() if k.startswith(("train_", "valid_", "test_"))}, "epoch": epochs},
        "last20": {
            key: float(np.mean([row[key] for row in curves[-20:]]))
            for key in curves[-1]
            if key.startswith(("train_", "valid_", "test_"))
        },
        "parameters": M.parameter_report(head),
        "budget": head.budget,
        "init": {"kind": "frozen_T_representation", "T": t, "T_recipe": t_recipe},
        "n_parameters": int(sum(p.numel() for p in head.parameters())),
        "teacher": "T final representation; gradients only through the head",
        "selection": "minimum Val teacher-fidelity RMSE; ties keep the earlier epoch",
        "seconds_total": float(sum(row["seconds"] for row in curves)),
        "environment": C.environment(),
        "device": str(device),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    C.write_json(out_dir / "trajectory.json", summary)
    C.write_json(
        out_dir / "status.json",
        {"state": "completed", "signature": sig, "task": task["key"], "selected_epoch": best["epoch"], "finished_at": summary["finished_at"]},
    )
    return summary


def _save_head(path, head, payload) -> None:
    import torch

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model_state": {k: v.detach().cpu() for k, v in head.state_dict().items()}, **payload}, tmp)
    tmp.replace(path)


def _save_head_resume(path, head, optimizer, curves, best, best_payload, state, sig, rng) -> None:
    import torch

    payload = {
        "head_state": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "curves": curves,
        "best": best,
        "best_payload": best_payload,
        "state": state,
        "signature": sig,
        "np_rng_state": rng.bit_generator.state,
        "rng_payload": C.training_rng_payload(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
