"""Training / valid evaluation for the second-hop composition pilot.

Recipe is the frozen LCRC-v1 recipe, verbatim:

    AdamW, peak LR 1e-3, min LR 1e-5, per-step cosine after a 500-update
    warm-up, weight decay 1e-5, batch 128, 600 epochs (79 steps/epoch,
    47,400 updates), raw-y L1, gradient clip 5, no dropout, no EMA, no soup.
    Full official valid every epoch; strict-argmin checkpoint selection.

Only the model input changes (the 128-wide second-hop block).  Official test is
never constructed.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import sh_common as S
import sh_data as D
import sh_models as M

RECIPE = {
    "optimizer": "adamw",
    "lr": 1.0e-3,
    "min_lr": 1.0e-5,
    "weight_decay": 1.0e-5,
    "warmup_steps": 500,
    "epochs": 600,
    "batch_size": 128,
    "valid_batch_size": 256,
    "dropout": 0.0,
    "grad_clip": 5.0,
    "loss": "l1",
}


def run_dir(arm: str, seed: int, tag: str | None = None) -> Path:
    prefix = f"{tag}_" if tag else ""
    return S.RUNS / f"sh{arm}_{prefix}s{int(seed)}"


def _to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    cache: Mapping[str, Any],
    sup: Mapping[str, Any],
    rows: Sequence[int],
    *,
    device,
    batch_size: int = 256,
    return_aux: bool = False,
    pair_mode: str | None = None,
):
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    preds, aux_parts = [], {}
    with S.L.C.isolated_rng():
        for start in range(0, len(rows), int(batch_size)):
            chunk = rows[start:start + int(batch_size)]
            batch = _to_device(D.collate(cache, sup, chunk), device)
            if return_aux:
                pred, aux = model(batch, return_aux=True, pair_mode=pair_mode)
                for k, v in aux.items():
                    if torch.is_tensor(v):
                        aux_parts.setdefault(k, []).append(v.detach().float().cpu())
            else:
                pred = model(batch, pair_mode=pair_mode)
            preds.append(pred.detach().float().cpu().numpy().reshape(-1))
    pred = np.concatenate(preds, 0) if preds else np.zeros(0, dtype=np.float32)
    y = np.asarray(cache["labels"], dtype=np.float64)[rows]
    mae = float(np.mean(np.abs(pred.astype(np.float64) - y))) if len(y) else float("nan")
    if not return_aux:
        return pred, mae
    aux = {k: (torch.cat(v, 0).numpy() if v else np.zeros(0)) for k, v in aux_parts.items()}
    return pred, mae, aux


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def weight_diagnostics(model: M.SecondHopLCRC) -> dict:
    W = model.local.mlp.lins[0].weight.detach()
    new = W[:, S.BASE_X_DIM:]
    old = W[:, :S.BASE_X_DIM]
    col_norms = torch.linalg.vector_norm(new, dim=0)
    return {
        "W_old_norm": float(torch.linalg.vector_norm(old).item()),
        "W_new_norm": float(torch.linalg.vector_norm(new).item()),
        "W_new_over_old": float(torch.linalg.vector_norm(new).item()
                               / max(torch.linalg.vector_norm(old).item(), 1e-12)),
        "W_new_col_norm_median": float(col_norms.median().item()),
        "W_new_col_norm_max": float(col_norms.max().item()),
        "W_new_nonzero_columns": int((col_norms > 0).sum().item()),
        "W_new_overlap_bond_block_norm": float(torch.linalg.vector_norm(new[:, :S.SECOND_HOP_DIM]).item()),
        "W_new_pair_block_norm": float(torch.linalg.vector_norm(new[:, S.SECOND_HOP_DIM:]).item()),
    }


def grad_diagnostics(model: M.SecondHopLCRC) -> dict:
    W = model.local.mlp.lins[0].weight.grad
    if W is None:
        return {"available": False}
    new = W[:, S.BASE_X_DIM:]
    old = W[:, :S.BASE_X_DIM]
    col_norms = torch.linalg.vector_norm(new, dim=0)
    return {
        "available": True,
        "all_finite": bool(torch.isfinite(W).all().item()),
        "grad_old_norm": float(torch.linalg.vector_norm(old).item()),
        "grad_new_norm": float(torch.linalg.vector_norm(new).item()),
        "grad_new_over_old": float(torch.linalg.vector_norm(new).item()
                                   / max(torch.linalg.vector_norm(old).item(), 1e-12)),
        "grad_new_nonzero_columns": int((col_norms > 0).sum().item()),
        "grad_new_col_norm_max": float(col_norms.max().item()),
    }


@torch.no_grad()
def input_diagnostics(model: M.SecondHopLCRC, cache: Mapping[str, Any], sup: Mapping[str, Any],
                      rows: Sequence[int], *, device, n_molecules: int = 512) -> dict:
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)[: int(n_molecules)]
    batch = _to_device(D.collate(cache, sup, rows), device)
    a, b, _h, aux = model._encode(batch)
    scal = (batch["atom_scalar"] - model.scalar_mean) / model.scalar_scale
    x = model.local.build_input(batch, a, b, scal, aux["second_hop_extra"])
    W = model.local.mlp.lins[0].weight
    z_old = x[:, :S.BASE_X_DIM] @ W[:, :S.BASE_X_DIM].T
    z_new = x[:, S.BASE_X_DIM:] @ W[:, S.BASE_X_DIM:].T

    def rms(t):
        return float(t.pow(2).mean().sqrt().item())

    n = batch["path_count"].to(torch.float32)
    c_real = aux["C_real"]
    c_null = aux["C_null"]
    diff = torch.linalg.vector_norm(c_real - c_null, dim=1)
    return {
        "n_molecules": int(len(rows)),
        "n_atoms": int(x.shape[0]),
        "x_old_rms": rms(x[:, :S.BASE_X_DIM]),
        "B_rms": rms(aux["B"]),
        "C_real_rms": rms(c_real),
        "C_null_rms": rms(c_null),
        "second_hop_extra_rms": rms(aux["second_hop_extra"]),
        "z_old_rms": rms(z_old),
        "z_new_rms": rms(z_new),
        "z_new_over_old": rms(z_new) / max(rms(z_old), 1e-12),
        "path_count_zero_atoms": int((n == 0).sum().item()),
        "path_count_one_atoms": int((n == 1).sum().item()),
        "path_count_ge2_atoms": int((n >= 2).sum().item()),
        "atoms_C_real_neq_C_null": int((diff > 0).sum().item()),
        "atoms_C_real_neq_C_null_frac": float((diff > 0).float().mean().item()),
        "mean_abs_C_real_minus_C_null": float(diff.mean().item()),
    }


@torch.no_grad()
def c_switch_probe(model: M.SecondHopLCRC, cache: Mapping[str, Any], sup: Mapping[str, Any],
                   rows: Sequence[int], *, device, batch_size: int = 256) -> dict:
    base = "real" if model.arm == "Real" else "null"
    other = "null" if base == "real" else "real"
    p0, m0 = evaluate(model, cache, sup, rows, device=device, batch_size=batch_size, pair_mode=base)
    p1, m1 = evaluate(model, cache, sup, rows, device=device, batch_size=batch_size, pair_mode=other)
    y = np.asarray(cache["labels"], dtype=np.float64)[np.asarray(rows, dtype=np.int64)]
    ae0 = np.abs(p0.astype(np.float64) - y)
    ae1 = np.abs(p1.astype(np.float64) - y)
    d = np.abs(p1.astype(np.float64) - p0.astype(np.float64))
    return {
        "arm": model.arm,
        "from": base,
        "to": other,
        "mae_from": float(m0),
        "mae_to": float(m1),
        "delta_mae": float(m1 - m0),
        "mean_abs_pred_change": float(d.mean()),
        "max_abs_pred_change": float(d.max()),
        "frac_molecules_changed": float((d > 1e-6).mean()),
        "frac_molecules_improved": float((ae1 < ae0).mean()),
    }


def _h_diagnostics(mean_h: np.ndarray) -> dict:
    H = np.asarray(mean_h, dtype=np.float64)
    Hc = H - H.mean(0, keepdims=True)
    cov = Hc.T @ Hc / max(len(Hc) - 1, 1)
    ev = np.clip(np.linalg.eigvalsh(cov), 0, None)
    return {
        "h_participation_ratio": float((ev.sum() ** 2) / max((ev ** 2).sum(), 1e-12)),
        "h_top1_energy_fraction": float(ev[-1] / max(ev.sum(), 1e-12)),
    }


def _param_audit(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    total = S.L.C.parameter_count(model)
    opt = S.L.C.optimizer_parameter_count(optimizer)
    requires = sum(1 for p in model.parameters() if p.requires_grad)
    nparams = sum(1 for p in model.parameters())
    return {
        "parameters": total,
        "optimizer_parameters": opt,
        "parameters_in_optimizer": bool(total == opt),
        "parameter_tensors": nparams,
        "requires_grad_tensors": requires,
        "all_parameters_require_grad": bool(requires == nparams),
        "param_cap_ok": bool(total <= S.PARAM_CAP),
    }


def _window_diagnostics(curve: Sequence[Mapping[str, Any]]) -> dict:
    vals = [float(c["valid_mae"]) for c in curve]
    if not vals:
        return {"W20": None, "T100": None}
    w20 = None
    if len(vals) >= 20:
        w20 = min(float(np.median(vals[i:i + 20])) for i in range(len(vals) - 19))
    return {"W20": w20, "T100": float(np.median(vals[-min(100, len(vals)):]))}


def _save_checkpoint(path: Path, model, *, arm, seed, epoch, valid_mae) -> str:
    payload = {
        "state_dict": S.L.C.cpu_state(model),
        "arm": arm,
        "seed": int(seed),
        "epoch": int(epoch),
        "valid_mae": float(valid_mae),
        "config": model.cfg,
    }
    S.L.atomic_save(payload, path)
    return S.L.C.sha256_file(path)


def train(arm: str, seed: int, device: str, *, epochs: int | None = None,
          tag: str | None = None, force: bool = False, out_dir: Path | None = None) -> dict:
    S.L.require_cuda(device)
    out = Path(out_dir) if out_dir is not None else run_dir(arm, seed, tag)
    S.L.ensure_dir(out)
    train_json = out / "train.json"
    if train_json.exists() and not force:
        S.L.log(f"[{arm} s{seed}] already complete: {train_json}")
        return S.L.read_json(train_json)

    cfg_file = S.load_config()
    frozen_sup = S.read_json_or_none(S.ANALYSIS / "supplement.json") or {}
    norm = S.D.load_norm()
    if S.D.norm_fingerprint(norm) != cfg_file["reuse"]["norm"]["fingerprint"]:
        raise RuntimeError("norm fingerprint drifted from the frozen value")
    train_cache = S.D.load_cache("train")
    valid_cache = S.D.load_cache("valid")
    for split, cache in (("train", train_cache), ("valid", valid_cache)):
        if cache["meta"]["feature_version"] != S.L.FEATURE_VERSION:
            raise RuntimeError(f"{split} cache feature version mismatch")
    sup_train = D.load_supplement("train", verify_sha=_frozen_sha(frozen_sup, "train"))
    sup_valid = D.load_supplement("valid", verify_sha=_frozen_sha(frozen_sup, "valid"))
    y_train = np.asarray(train_cache["labels"], dtype=np.float64)
    rows_train = np.arange(len(y_train), dtype=np.int64)
    rows_valid = np.arange(len(valid_cache["labels"]), dtype=np.int64)

    cfg = dict(M.DEFAULT_CONFIG)
    model, model_info = M.build_model(arm, norm, seed, device, cfg)
    recipe = dict(RECIPE)
    n_epochs = int(recipe["epochs"] if epochs is None else epochs)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"]
    )
    audit = _param_audit(model, optimizer)
    if not audit["parameters_in_optimizer"] or not audit["param_cap_ok"]:
        raise RuntimeError(f"parameter audit failed: {audit}")

    diag_init = {"weights": weight_diagnostics(model), "inputs": None}
    diag_init["inputs"] = input_diagnostics(model, valid_cache, sup_valid, rows_valid,
                                           device=device, n_molecules=256)

    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(len(rows_train) / batch_size)
    total_updates = steps_per_epoch * n_epochs
    warmup = min(int(recipe["warmup_steps"]), max(1, total_updates // 20))
    gen = torch.Generator().manual_seed(int(seed) + 101)
    curve: list[dict] = []
    best = {"valid_mae": float("inf"), "epoch": -1}
    best_state = None
    best_sha = None
    started = time.perf_counter()
    global_step = 0
    last_lr = float(recipe["lr"])
    first_batch_rows = None
    first_batch_grad = None
    step1_weights = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(rows_train), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), batch_size):
            rows = rows_train[order[start:start + batch_size]]
            if first_batch_rows is None:
                first_batch_rows = rows.copy()
            batch = _to_device(D.collate(train_cache, sup_train, rows), device)
            target = torch.as_tensor(y_train[rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if first_batch_grad is None:
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                first_batch_grad = {
                    "loss": float(loss.detach().item()),
                    "loss_finite": bool(torch.isfinite(loss).item()),
                    "grad_tensors": len(grads),
                    "nonzero_grad_tensors": int(sum(1 for g in grads if torch.count_nonzero(g).item() > 0)),
                    "all_grad_finite": bool(all(torch.isfinite(g).all().item() for g in grads)),
                    "first_layer": grad_diagnostics(model),
                }
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["grad_clip"]))
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at epoch {epoch}")
            global_step += 1
            last_lr = S.L.C.cosine_with_warmup_lr(
                global_step, total_updates, warmup, float(recipe["lr"]), float(recipe["min_lr"])
            )
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            if step1_weights is None:
                step1_weights = weight_diagnostics(model)
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(rows)
        online = total_err / max(n_mol, 1)
        _, valid_mae = evaluate(model, valid_cache, sup_valid, rows_valid, device=device,
                                batch_size=int(recipe["valid_batch_size"]))
        sec = time.perf_counter() - t_ep
        entry = {
            "epoch": int(epoch),
            "online_train_mae": float(online),
            "valid_mae": float(valid_mae),
            "lr": float(last_lr),
            "seconds": float(sec),
            "steps": int(steps_per_epoch),
            "W_new_norm": float(torch.linalg.vector_norm(
                model.local.mlp.lins[0].weight.detach()[:, S.BASE_X_DIM:]).item()),
        }
        curve.append(entry)
        if valid_mae < best["valid_mae"]:
            best = {"valid_mae": float(valid_mae), "epoch": int(epoch)}
            best_state = S.L.C.cpu_state(model)
            best_sha = _save_checkpoint(out / "best.pt", model, arm=arm, seed=seed,
                                        epoch=epoch, valid_mae=valid_mae)
        if not math.isfinite(valid_mae):
            raise RuntimeError(f"non-finite valid MAE at epoch {epoch}")
        S.L.log(f"[{arm} s{seed}] ep={epoch:03d} valid={valid_mae:.6f} online={online:.6f} "
                f"lr={last_lr:.3e} best={best['valid_mae']:.6f}@{best['epoch']} "
                f"Wnew={entry['W_new_norm']:.4f} {sec:.2f}s")

    wall = time.perf_counter() - started
    final_sha = _save_checkpoint(out / "final.pt", model, arm=arm, seed=seed,
                                 epoch=n_epochs, valid_mae=curve[-1]["valid_mae"])
    S.L.C.atomic_torch_save(
        {
            "state_dict": S.L.C.cpu_state(model),
            "optimizer": optimizer.state_dict(),
            "gen_state": gen.get_state(),
            "epoch": n_epochs,
            "global_step": global_step,
            "best": best,
            "curve": curve,
            "arm": arm,
            "seed": int(seed),
        },
        out / "final_state.pt",
    )

    best_model = M.load_model_from_state(arm, norm, best_state, device, cfg)
    best_pred_valid, best_valid, best_aux = evaluate(
        best_model, valid_cache, sup_valid, rows_valid, device=device,
        batch_size=int(recipe["valid_batch_size"]), return_aux=True)
    _, best_train = evaluate(best_model, train_cache, sup_train, rows_train, device=device,
                             batch_size=int(recipe["valid_batch_size"]))
    final_pred_valid, final_valid = evaluate(model, valid_cache, sup_valid, rows_valid,
                                             device=device, batch_size=int(recipe["valid_batch_size"]))
    _, final_train = evaluate(model, train_cache, sup_train, rows_train, device=device,
                              batch_size=int(recipe["valid_batch_size"]))
    if abs(best_valid - best["valid_mae"]) > 1e-6:
        raise RuntimeError(f"best checkpoint replay mismatch: {best_valid} vs {best['valid_mae']}")
    if abs(final_valid - curve[-1]["valid_mae"]) > 1e-6:
        raise RuntimeError(f"final checkpoint replay mismatch: {final_valid} vs {curve[-1]['valid_mae']}")

    switch = c_switch_probe(best_model, valid_cache, sup_valid, rows_valid, device=device,
                            batch_size=int(recipe["valid_batch_size"]))
    diag_best = {
        "weights": weight_diagnostics(best_model),
        "inputs": input_diagnostics(best_model, valid_cache, sup_valid, rows_valid,
                                    device=device, n_molecules=512),
        "h": _h_diagnostics(np.asarray(best_aux["mean_h"])),
        "c_switch": switch,
        "contrib_mean_abs": float(np.mean(np.abs(np.asarray(best_aux["contrib"])))),
        "rho_mean_abs": float(np.mean(np.abs(np.asarray(best_aux["rho"])))),
    }

    pred_dir = out / "predictions"
    S.L.ensure_dir(pred_dir)
    np.savez(pred_dir / "valid_best.npz", row=rows_valid, pred=best_pred_valid)
    np.savez(pred_dir / "valid_final.npz", row=rows_valid, pred=final_pred_valid)
    np.savez(pred_dir / "valid_best_switch.npz", row=rows_valid,
             pred_real=np.asarray(
                 evaluate(best_model, valid_cache, sup_valid, rows_valid, device=device,
                          batch_size=int(recipe["valid_batch_size"]), pair_mode="real")[0]),
             pred_null=np.asarray(
                 evaluate(best_model, valid_cache, sup_valid, rows_valid, device=device,
                          batch_size=int(recipe["valid_batch_size"]), pair_mode="null")[0]))
    S.L.C.write_curve_csv(out / "curve.csv", curve)
    diag = _window_diagnostics(curve)
    payload = {
        "round": S.ROUND_ID,
        "candidate": f"SHC-{arm}",
        "arm": arm,
        "seed": int(seed),
        "status": "completed",
        "git_commit": S.L.C.git_commit(),
        "environment": S.L.env_summary(),
        "device_name": (torch.cuda.get_device_name(0) if str(device).startswith("cuda") else "cpu"),
        "data": {
            "source": "official PyG ZINC subset=True train/valid",
            "n_train": int(len(y_train)),
            "n_valid": int(len(rows_valid)),
            "official_test_loaded": False,
            "train_cache_file_sha256": S.L.C.sha256_file(S.D.cache_path("train")),
            "valid_cache_file_sha256": S.L.C.sha256_file(S.D.cache_path("valid")),
            "train_cache_content_sha256": S.D.content_fingerprint(train_cache),
            "valid_cache_content_sha256": S.D.content_fingerprint(valid_cache),
            "supplement_content_sha256": {
                "train": S.supplement_content_hash(sup_train),
                "valid": S.supplement_content_hash(sup_valid),
            },
            "supplement_file_sha256": {
                "train": S.L.C.sha256_file(S.supplement_path("train")),
                "valid": S.L.C.sha256_file(S.supplement_path("valid")),
            },
            "supplement_crosscheck": {
                "train": sup_train["crosscheck"]["passed"],
                "valid": sup_valid["crosscheck"]["passed"],
            },
            "train_label_sha256": S.L.C.sha256_array(y_train),
            "valid_label_sha256": S.L.C.sha256_array(valid_cache["labels"]),
            "feature_version": S.FEATURE_VERSION,
            "supplement_version": S.SUPPLEMENT_VERSION,
        },
        "model_info": model_info,
        "recipe": recipe,
        "warmup_steps": int(warmup),
        "steps_per_epoch": int(steps_per_epoch),
        "total_updates": int(total_updates),
        "epochs": int(n_epochs),
        "param_audit": audit,
        "init_diagnostics": diag_init,
        "step1_weights": step1_weights,
        "gradient_audit_first_batch": first_batch_grad,
        "first_batch_rows": None if first_batch_rows is None else first_batch_rows.tolist(),
        "data_order_seed": int(seed) + 101,
        "norm": {"source": norm["source"], "fingerprint": S.D.norm_fingerprint(norm)},
        "best": best,
        "final_valid_mae": float(final_valid),
        "best_valid_mae_eval": float(best_valid),
        "best_train_mae_eval": float(best_train),
        "final_train_mae_eval": float(final_train),
        "W20": diag["W20"],
        "T100": diag["T100"],
        "wall_seconds": float(wall),
        "gpu_hours_estimate": float(wall / 3600.0),
        "checkpoint_sha256": {"best": best_sha, "final": final_sha},
        "best_diagnostics": diag_best,
        "curve": curve,
    }
    S.L.write_json(train_json, payload)
    S.L.log(f"[{arm} s{seed}] done best={best['valid_mae']:.6f}@{best['epoch']} "
            f"final={final_valid:.6f} wall={wall:.1f}s")
    return payload


def _frozen_sha(frozen: Mapping[str, Any], split: str):
    block = frozen.get(split) if isinstance(frozen, Mapping) else None
    if not block:
        return None
    return block.get("content_sha256")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="second-hop composition pilot")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prep")
    p_prep.add_argument("--splits", default="train,valid")

    p_train = sub.add_parser("train")
    p_train.add_argument("--arm", required=True, choices=list(M.ARMS))
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--device", default="cuda:0")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--tag", default=None)
    p_train.add_argument("--out", default=None)
    p_train.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "prep":
        res = {s: D.build_supplement(s) for s in args.splits.split(",")}
        S.L.log(str({k: v["content_sha256"] for k, v in res.items()}))
        return 0
    if args.cmd == "train":
        res = train(args.arm, args.seed, args.device, epochs=args.epochs, tag=args.tag,
                    force=bool(args.force),
                    out_dir=None if args.out is None else Path(args.out))
        S.L.log(f"best valid {res['best']['valid_mae']:.6f} @ {res['best']['epoch']}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
