"""The two training institutions and the one formal unit entry point.

Institution **F (FROZEN)**: the selected, already trained C1 checkpoint for the
task is loaded, every one of its parameters and buffers is frozen and the module
is kept in ``eval()``.  Only the composition branch is trained, with the full
standardised-prediction MSE plus the pre-registered output penalty
``0.02 * mean(delta_comp^2)``.  Epoch 0 is a legal candidate and equals the frozen
C1 prediction exactly, so *best-Valid is better than C1 by construction* in this
institution; only JOINT vs MARGINAL is informative here.

Institution **E (END_TO_END)**: a fresh model is rebuilt from the canonical C1
seed-0 initial state (never from the trained teacher) and the C1 parameters and
the branch are trained together under the frozen Rf recipe.  The branch is
zero-initialised, so epoch 0 is the canonical C1 initial state.

Randomness discipline
---------------------
* batch order: ``np.random.default_rng(0)`` permutations over Train split-local
  rows, one per epoch — identical for every unit of this round;
* C1 dropout draws from the global torch stream, exactly as in the historical Rf
  runs, so the shared C1 part sees the historical stream;
* the branch has its **own** ``torch.Generator`` (``BRANCH_DROPOUT_SEED``) so it
  can never shift the C1 stream, and JOINT / MARGINAL draw the same masks because
  their tensors have identical shapes and the generator is re-seeded per epoch.
"""

from __future__ import annotations

import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import bc_common as C
import bc_data as D
import bc_models as M
import bc_path3 as P3


# ---------------------------------------------------------------------------
# teacher loading / identity
# ---------------------------------------------------------------------------


def teacher_identity(name: str) -> dict[str, Any]:
    XR = C.modules()["xr_common"]
    path = XR.CHECKPOINTS[name]
    expected = XR.EXPECTED_CHECKPOINT_SHA256[name]
    actual = C.sha256_file(path) if path.is_file() else None
    payload = torch.load(path, map_location="cpu", weights_only=False) if path.is_file() else {}
    return {
        "name": name,
        "path": C.relpath(path),
        "sha256": actual,
        "sha256_expected": expected,
        "sha256_match": bool(actual == expected),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_valid_rmse_raw": payload.get("valid_rmse_raw"),
        "state_keys": len(payload.get("model_state", {})),
        "source_path": XR.SOURCE_CHECKPOINT_PATHS[name],
    }


def load_teacher_model(task: str, b) -> tuple[torch.nn.Module, dict[str, Any]]:
    """The frozen C1 teacher for one task, loaded and identity-verified."""
    name = C.TASK_TEACHER[task]
    ident = teacher_identity(name)
    if not ident["sha256_match"]:
        raise RuntimeError(f"TEACHER_UNAVAILABLE {name}: sha256 {ident['sha256']} != {ident['sha256_expected']}")
    payload = torch.load(C.modules()["xr_common"].CHECKPOINTS[name], map_location="cpu", weights_only=False)
    state = payload["model_state"]
    arm_state = {k[len("arm.") :]: v for k, v in state.items() if k.startswith("arm.")}
    if not arm_state:
        raise RuntimeError(f"{name}: checkpoint has no 'arm.' parameters")
    CM = C.modules()["cr_models"]
    arm = CM.ArmA(b.spec, b.norm)
    arm.load_state_dict({k: v.clone() for k, v in arm_state.items()}, strict=True)
    arm.eval()
    for param in arm.parameters():
        param.requires_grad_(False)
    ident["n_parameters"] = int(sum(p.numel() for p in arm.parameters()))
    ident["state_hash"] = C.sha256_state(arm.state_dict())
    return arm, ident


# ---------------------------------------------------------------------------
# canonical branch initialisation (one shared tensor-level state)
# ---------------------------------------------------------------------------


def _init_linear_like(module: torch.nn.Linear, generator: torch.Generator) -> None:
    torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=generator)
    if module.bias is not None:
        bound = 1.0 / math.sqrt(module.in_features) if module.in_features > 0 else 0.0
        torch.nn.init.uniform_(module.bias, -bound, bound, generator=generator)


def canonical_branch_state() -> dict[str, torch.Tensor]:
    """One isolated, version-frozen initial state shared by JOINT and MARGINAL."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(C.BRANCH_INIT_SEED)
    composer = M.Composer()
    with torch.no_grad():
        for name in ("r_proj", "b_proj", "psi_in", "psi_out"):
            _init_linear_like(getattr(composer, name), generator)
        torch.nn.init.zeros_(composer.readout.weight)
    return {k: v.detach().clone() for k, v in composer.state_dict().items()}


def _strip_arm_prefix(state: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical namespace: ``ArmA`` nests the C1 trunk under ``trunk.``."""
    return {
        (k[len("trunk.") :] if k.startswith("trunk.") else k): v for k, v in state.items()
    }


def build_model(task: str, b, variant: str, institution: str, *, seed: int) -> tuple[M.BAJC, dict[str, Any]]:
    """Build one research model, with the frozen initialisation identity recorded."""
    if institution not in C.INSTITUTIONS:
        raise KeyError(institution)
    if variant not in C.VARIANTS:
        raise KeyError(variant)
    CM = C.modules()["cr_models"]
    arm = CM.ArmA(b.spec, b.norm)
    canonical = M.c1_canonical_state(b.spec, b.norm)
    arm.load_canonical(canonical)
    model = M.BAJC(arm, variant)
    audit: dict[str, Any] = {
        "institution": institution,
        "variant": variant,
        "task": task,
        "seed": int(seed),
        "canonical_c1_state_hash": C.sha256_state(canonical),
        "variant_note": C.VARIANT_NOTE[variant],
    }
    branch_state = canonical_branch_state()
    model.composer.load_state_dict({k: v.clone() for k, v in branch_state.items()}, strict=True)
    model.composer.set_dropout_seed(C.BRANCH_DROPOUT_SEED)
    audit["branch_init_hash"] = C.sha256_state(branch_state)
    audit["branch_init_seed"] = C.BRANCH_INIT_SEED
    if institution == "FROZEN":
        teacher, ident = load_teacher_model(task, b)
        model.c1.load_state_dict({k: v.clone() for k, v in teacher.state_dict().items()}, strict=True)
        audit["teacher"] = ident
        audit["freeze"] = model.freeze_c1()
        audit["c1_state_hash_after_load"] = C.sha256_state(model.c1.state_dict())
        audit["c1_state_hash_equals_teacher"] = bool(
            audit["c1_state_hash_after_load"] == ident["state_hash"]
        )
    else:
        built = _strip_arm_prefix(model.c1.state_dict())
        audit["c1_state_hash_after_build"] = C.sha256_state(built)
        audit["c1_matches_canonical"] = bool(
            audit["c1_state_hash_after_build"] == audit["canonical_c1_state_hash"]
        )
    audit["parameter_report"] = model.parameter_report()
    return model, audit


# ---------------------------------------------------------------------------
# schedules / batch order
# ---------------------------------------------------------------------------


def batch_perms(n_rows: int, epochs: int, *, seed: int = C.TRAIN_SEED_DEFAULT) -> list[np.ndarray]:
    generator = np.random.default_rng(int(seed))
    return [generator.permutation(int(n_rows)) for _ in range(int(epochs))]


def lr_at(step: int, total_steps: int, steps_per_epoch: int) -> float:
    warmup_steps = int(C.RECIPE_E2E["warmup_epochs"]) * int(steps_per_epoch)
    if step < warmup_steps:
        return float(C.RECIPE_E2E["peak_lr"]) * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    lo = float(C.RECIPE_E2E["min_lr"])
    hi = float(C.RECIPE_E2E["peak_lr"])
    return lo + 0.5 * (hi - lo) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------


def evaluate(
    model: M.BAJC,
    b,
    p3: P3.Path3Split,
    split: str,
    *,
    device,
    scaler: Mapping[str, Any],
    batch_size: int = 256,
    branch: bool = True,
) -> dict[str, Any]:
    if isinstance(p3, Mapping):
        p3 = p3[split]
    out = D.predict_std(model, b, p3, split, device=device, batch_size=batch_size, branch=branch)
    pred_raw = D.to_raw(out["pred_std"], scaler)
    base_raw = D.to_raw(out["base_std"], scaler)
    delta_raw = pred_raw - base_raw
    return {
        "row_id": out["row_id"],
        "y_raw": out["y_raw"],
        "pred_std": out["pred_std"],
        "base_std": out["base_std"],
        "delta_std": out["delta_std"],
        "pred_raw": pred_raw,
        "base_raw": base_raw,
        "delta_raw": delta_raw,
        "n_paths3": out["n_paths3"],
        "rmse_raw": C.rmse(pred_raw, out["y_raw"]),
        "mae_raw": C.mae(pred_raw, out["y_raw"]),
        "rmse_base_raw": C.rmse(base_raw, out["y_raw"]),
    }


def _event_summary(c_P: np.ndarray, graph: np.ndarray, n_graphs: int) -> dict[str, np.ndarray]:
    """Per-molecule aggregation of the native event contributions."""
    abs_c = np.abs(np.asarray(c_P, dtype=np.float64))
    total = np.zeros(n_graphs)
    maxabs = np.zeros(n_graphs)
    arg = np.full(n_graphs, -1, dtype=np.int64)
    n_ev = np.zeros(n_graphs, dtype=np.int64)
    if c_P.size:
        total = total + np.bincount(graph, weights=abs_c, minlength=n_graphs)
        n_ev = np.bincount(graph, minlength=n_graphs).astype(np.int64)
        order = np.lexsort((np.arange(c_P.size), -abs_c, graph))
        first = np.ones(c_P.size, dtype=bool)
        first[1:] = graph[order][1:] != graph[order][:-1]
        rows = graph[order][first]
        maxabs[rows] = abs_c[order][first]
        arg[rows] = order[first]
    return {"abs_c_sum": total, "abs_c_max": maxabs, "argmax_event": arg, "n_events": n_ev}


# ---------------------------------------------------------------------------
# the unit
# ---------------------------------------------------------------------------


def run_unit(
    *,
    institution: str,
    task: str,
    variant: str,
    seed: int,
    device: str = "cuda",
    rebuild_paths: bool = False,
    max_epochs: int | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """One formal training unit.  Never retrained, never resumed silently."""
    torch_device = torch.device(device)
    C.set_global_seeds(C.TRAIN_SEED_DEFAULT)
    started = time.time()
    paths = C.unit_paths(institution, task, variant, seed)
    C.ensure_dir(paths["dir"])
    b = D.bundle(task)
    p3 = D.build_paths3(b, rebuild=rebuild_paths)
    scaler = b.scaler
    y_std = {split: np.asarray(b.y_std[split], dtype=np.float64) for split in C.SPLITS}

    model, audit = build_model(task, b, variant, institution, seed=seed)
    model.to(torch_device)
    report = dict(audit["parameter_report"])
    if not report["c1_matches_frozen"]:
        raise RuntimeError(f"{variant}: C1 parameter count {report['c1']} != frozen {C.C1_PARAMETERS}")
    if not report["branch_within_budget"]:
        raise RuntimeError(f"{variant}: branch {report['branch']} exceeds the 15% budget")

    n_train = int(np.asarray(b.caches["train"]["n_molecules"]))
    bs = int(C.RECIPE_E2E["batch_size"])
    steps_per_epoch = int(math.ceil(n_train / bs))
    if institution == "FROZEN":
        epochs = int(C.RECIPE_FROZEN["epochs"])
        optimizer = torch.optim.AdamW(
            model.branch_parameters(),
            lr=float(C.RECIPE_FROZEN["lr"]),
            betas=tuple(C.RECIPE_FROZEN["adam_betas"]),
            eps=float(C.RECIPE_FROZEN["adam_eps"]),
            weight_decay=float(C.RECIPE_FROZEN["weight_decay"]),
        )
        total_steps = steps_per_epoch * epochs
    else:
        epochs = int(C.RECIPE_E2E["epochs"])
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(C.RECIPE_E2E["peak_lr"]),
            betas=tuple(C.RECIPE_E2E["adam_betas"]),
            eps=float(C.RECIPE_E2E["adam_eps"]),
            weight_decay=float(C.RECIPE_E2E["weight_decay"]),
        )
        total_steps = steps_per_epoch * epochs
    if max_epochs is not None:
        epochs = int(max_epochs)
        total_steps = steps_per_epoch * epochs

    perms = batch_perms(n_train, epochs, seed=C.TRAIN_SEED_DEFAULT)
    curve: list[dict[str, Any]] = []
    best = {"epoch": -1, "valid_rmse_raw": float("inf"), "state": None}
    update = 0

    # epoch 0: pre-training state (== frozen teacher in F, == canonical C1 in E)
    model.eval()
    ev0 = {split: evaluate(model, b, p3, split, device=torch_device, scaler=scaler) for split in ("train", "valid")}
    curve.append(
        {
            "epoch": 0,
            "update": 0,
            "lr": 0.0,
            "train_loss_std_mean": float("nan"),
            "train_rmse_raw": ev0["train"]["rmse_raw"],
            "valid_rmse_raw": ev0["valid"]["rmse_raw"],
            "train_rmse_base_raw": ev0["train"]["rmse_base_raw"],
            "valid_rmse_base_raw": ev0["valid"]["rmse_base_raw"],
            "train_delta_rms_std": float(np.sqrt(np.mean(ev0["train"]["delta_std"] ** 2))),
            "valid_delta_rms_std": float(np.sqrt(np.mean(ev0["valid"]["delta_std"] ** 2))),
            "grad_norm_max": 0.0,
            "train_seconds": 0.0,
            "eval_seconds": 0.0,
            "elapsed_s": time.time() - started,
        }
    )
    best = {"epoch": 0, "valid_rmse_raw": ev0["valid"]["rmse_raw"], "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    C.atomic_write_npz(
        paths["epoch0"],
        {
            "valid_row_id": ep0row(ev0, "valid", "row_id"),
            "valid_y_raw": ep0row(ev0, "valid", "y_raw"),
            "valid_pred_raw": ep0row(ev0, "valid", "pred_raw"),
            "train_row_id": ep0row(ev0, "train", "row_id"),
            "train_y_raw": ep0row(ev0, "train", "y_raw"),
            "train_pred_raw": ep0row(ev0, "train", "pred_raw"),
        },
    )

    for epoch in range(1, epochs + 1):
        model.train()
        if institution == "FROZEN":
            model.c1.eval()
        model.composer.set_dropout_seed(C.BRANCH_DROPOUT_SEED + epoch)
        t0 = time.time()
        losses: list[float] = []
        grad_norm_max = 0.0
        order = perms[epoch - 1]
        for start in range(0, n_train, bs):
            rows = order[start : start + bs]
            bt = D.make_batch(b, "train", rows, p3).to(torch_device)
            if institution == "E2E":
                lr = lr_at(update, total_steps, steps_per_epoch)
                for group in optimizer.param_groups:
                    group["lr"] = lr
            parts = model(bt.batch, bt.p3, return_parts=True)
            pred = parts["pred_std"]
            target = torch.as_tensor(
                y_std["train"][rows], device=pred.device, dtype=pred.dtype
            )
            loss = torch.mean((pred - target) ** 2)
            if institution == "FROZEN":
                loss = loss + float(C.RECIPE_FROZEN["branch_penalty"]) * torch.mean(parts["delta_comp"] ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(C.RECIPE_E2E["gradient_clip"]),
            ))
            grad_norm_max = max(grad_norm_max, gnorm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            update += 1
        train_seconds = time.time() - t0
        model.eval()
        t1 = time.time()
        tr = evaluate(model, b, p3, "train", device=torch_device, scaler=scaler)
        va = evaluate(model, b, p3, "valid", device=torch_device, scaler=scaler)
        eval_seconds = time.time() - t1
        row = {
            "epoch": epoch,
            "update": update,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss_std_mean": float(np.mean(losses)) if losses else float("nan"),
            "train_rmse_raw": tr["rmse_raw"],
            "valid_rmse_raw": va["rmse_raw"],
            "train_rmse_base_raw": tr["rmse_base_raw"],
            "valid_rmse_base_raw": va["rmse_base_raw"],
            "train_delta_rms_std": float(np.sqrt(np.mean(tr["delta_std"] ** 2))),
            "valid_delta_rms_std": float(np.sqrt(np.mean(va["delta_std"] ** 2))),
            "grad_norm_max": grad_norm_max,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            "elapsed_s": time.time() - started,
        }
        curve.append(row)
        if row["valid_rmse_raw"] < best["valid_rmse_raw"]:
            best = {
                "epoch": epoch,
                "valid_rmse_raw": row["valid_rmse_raw"],
                "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
            }
        if not quiet and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
            print(
                f"[{institution}/{task}/{variant}/s{seed}] epoch {epoch}/{epochs} "
                f"train {row['train_rmse_raw']:.4f} valid {row['valid_rmse_raw']:.4f} "
                f"delta_rms {row['train_delta_rms_std']:.4f}/{row['valid_delta_rms_std']:.4f} "
                f"lr {row['lr']:.2e}",
                flush=True,
            )

    final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict({k: v.clone() for k, v in best["state"].items()}, strict=True)
    model.eval()
    evals: dict[str, Any] = {}
    for split in C.SPLITS:
        evals[split] = evaluate(model, b, p3, split, device=torch_device, scaler=scaler)
    with_branch_off = evaluate(model, b, p3, "valid", device=torch_device, scaler=scaler, branch=False)

    # per-molecule native event summary for the best state (valid + test)
    summaries: dict[str, np.ndarray] = {}
    collected: dict[str, dict[str, np.ndarray]] = {}
    for split in ("valid", "test"):
        c_list, g_list, row_list = [], [], []
        model.eval()
        order, row_ids, _y = D.eval_order(b, split)
        with torch.no_grad():
            for start in range(0, order.size, 256):
                chunk = order[start : start + 256]
                bt = D.make_batch(b, split, chunk, p3).to(torch_device)
                parts = model(bt.batch, bt.p3, return_parts=True)
                c_list.append(parts["c_P"].detach().float().cpu().numpy().astype(np.float64))
                g_list.append(bt.p3["p3_graph"].detach().cpu().numpy().astype(np.int64))
                row_list.append(bt.p3["p3_index"].detach().cpu().numpy().astype(np.int64))
        c_P = np.concatenate(c_list) if c_list else np.zeros(0)
        graph = np.concatenate(g_list) if g_list else np.zeros(0, dtype=np.int64)
        index = np.concatenate(row_list) if row_list else np.zeros(0, dtype=np.int64)
        n_graphs = int(order.size)
        sumry = _event_summary(c_P, graph, n_graphs)
        collected[split] = {
            "c_P": c_P.astype(np.float32),
            "graph": graph,
            "event_index": index,
            "row_id": np.asarray(row_ids, dtype=np.int64),
        }
        summaries.update({f"{split}_{k}": v for k, v in sumry.items()})

    C.atomic_torch_save(
        paths["checkpoint_best"],
        {
            "model_state": best["state"],
            "epoch": int(best["epoch"]),
            "valid_rmse_raw": float(best["valid_rmse_raw"]),
            "institution": institution,
            "task": task,
            "variant": variant,
            "seed": int(seed),
            "audit": audit,
            "parameter_report": report,
        },
    )
    C.atomic_torch_save(
        paths["checkpoint_final"],
        {
            "model_state": final_state,
            "epoch": int(epochs),
            "institution": institution,
            "task": task,
            "variant": variant,
            "seed": int(seed),
            "audit": audit,
        },
    )
    C.write_csv(
        paths["curve"],
        curve,
        columns=sorted({k for row in curve for k in row}),
    )

    # decision-independent prediction table for this unit
    npz: dict[str, np.ndarray] = {}
    for split in C.SPLITS:
        e = evals[split]
        npz[f"{split}_row_id"] = e["row_id"]
        npz[f"{split}_y_raw"] = e["y_raw"]
        npz[f"{split}_pred_raw"] = e["pred_raw"]
        npz[f"{split}_base_raw"] = e["base_raw"]
        npz[f"{split}_delta_raw"] = e["delta_raw"]
        npz[f"{split}_pred_std"] = e["pred_std"]
        npz[f"{split}_n_paths3"] = e["n_paths3"]
    for split in ("valid", "test"):
        for key, value in collected[split].items():
            npz[f"{split}_event_{key}"] = value
    npz["valid_branch_off_pred_raw"] = with_branch_off["pred_raw"]
    npz["valid_branch_off_row_id"] = with_branch_off["row_id"]
    C.atomic_write_npz(paths["predictions"], npz)

    elapsed = time.time() - started
    status = {
        "unit": f"{institution}/{task}/{variant}/seed{seed}",
        "institution": institution,
        "task": task,
        "variant": variant,
        "seed": int(seed),
        "epochs_run": int(epochs),
        "total_updates": int(update),
        "steps_per_epoch": int(steps_per_epoch),
        "selected_epoch": int(best["epoch"]),
        "selected_valid_rmse_raw": float(best["valid_rmse_raw"]),
        "epoch0_valid_rmse_raw": float(ev0["valid"]["rmse_raw"]),
        "epoch0_train_rmse_raw": float(ev0["train"]["rmse_raw"]),
        "final_train_rmse_raw": float(evals["train"]["rmse_raw"]),
        "final_valid_rmse_raw": float(evals["valid"]["rmse_raw"]),
        "development_test_rmse_raw": float(evals["test"]["rmse_raw"]),
        "development_test_note": (
            "DEVELOPMENT test: evaluated exactly once, from the checkpoint selected on Valid, "
            "after the model and the selection rule were frozen"
        ),
        "valid_rmse_raw_branch_off": float(with_branch_off["rmse_raw"]),
        "delta_rms_std_valid": float(np.sqrt(np.mean(evals["valid"]["delta_std"] ** 2))),
        "delta_rms_std_train": float(np.sqrt(np.mean(evals["train"]["delta_std"] ** 2))),
        "teacher_valid_std": float(np.std(evals["valid"]["base_std"], ddof=0)),
        "delta_over_teacher_std": float(
            np.sqrt(np.mean(evals["valid"]["delta_std"] ** 2)) / max(1e-12, float(np.std(evals["valid"]["base_std"], ddof=0)))
        ),
        "scaler": dict(scaler),
        "parameter_report": report,
        "audit": audit,
        "paths3_sha256": {split: p3[split].sha256 for split in C.SPLITS},
        "device": str(torch_device),
        "environment": C.environment_snapshot(),
        "git_commit": C.git_commit(),
        "git_branch": C.git_branch(),
        "git_dirty_tracked": C.git_dirty_tracked(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": float(elapsed),
        "gpu_hours": float(elapsed / 3600.0),
    }
    C.atomic_write_json(paths["status"], status)
    C.atomic_write_json(
        paths["ledger"],
        {
            "unit": status["unit"],
            "elapsed_seconds": status["elapsed_seconds"],
            "gpu_hours": status["gpu_hours"],
            "device": status["device"],
            "kind": "FROZEN branch training" if institution == "FROZEN" else "END_TO_END Rf training",
            "epochs": int(epochs),
            "selected_epoch": int(best["epoch"]),
        },
    )
    if not quiet:
        print(
            f"[{institution}/{task}/{variant}/s{seed}] DONE best_epoch={best['epoch']} "
            f"valid={best['valid_rmse_raw']:.4f} test={status['development_test_rmse_raw']:.4f} "
            f"epoch0_valid={status['epoch0_valid_rmse_raw']:.4f} ({elapsed/60:.1f} min)",
            flush=True,
        )
    return status


def ep0row(ev0: Mapping[str, Any], split: str, key: str) -> np.ndarray:
    return np.asarray(ev0[split][key])


__all__ = [
    "run_unit",
    "build_model",
    "canonical_branch_state",
    "teacher_identity",
    "load_teacher_model",
    "batch_perms",
    "evaluate",
    "lr_at",
]
