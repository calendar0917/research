"""Formal single-arm training for SES-v0 (DENSE / RANDOM / SELECT).

All three arms share the exact same frozen ``GwicModel(use_words=False)``
predictor, the same initial state, the same fit-only normalisers and the same
optimisation recipe.  Only the input graphs / loss differ:

* DENSE  : complete original graph, standardized-label MSE;
* RANDOM : every candidate encoded independently, mean_j (y_j-y)^2;
* SELECT : same candidates, per-molecule softmax over a detached pre-readout
           representation, p_train = 0.9 p + 0.1/m, expected squared error.

Evaluation is the full official Train/Valid/Test split every epoch.  Test is
``DEVELOPMENT_TEST``; it never selects a checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import math
import pickle
import platform
import socket
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import ses_common as C
import ses_data as D
import ses_structure as S
import ses_tensor as T

ARMS = ("DENSE", "RANDOM", "SELECT")
TRAIN_FILES = ("ses_common.py", "ses_data.py", "ses_structure.py", "ses_tensor.py",
               "ses_train.py")


def build_selector(dim_z: int, seed: int) -> nn.Sequential:
    """MLP(dim_z, 64, 1) with SiLU and a private deterministic initialisation."""
    selector = nn.Sequential(nn.Linear(int(dim_z), 64), nn.SiLU(), nn.Linear(64, 1))
    gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1000003 + 71)
    with torch.no_grad():
        for layer in selector.modules():
            if isinstance(layer, nn.Linear):
                bound = 1.0 / math.sqrt(layer.in_features)
                layer.weight.uniform_(-bound, bound, generator=gen)
                if layer.bias is not None:
                    layer.bias.uniform_(-bound, bound, generator=gen)
    return selector


def selector_parameters(selector: nn.Module | None) -> int:
    if selector is None:
        return 0
    return int(sum(p.numel() for p in selector.parameters()))


def load_prepared(path: Path | str, bundle: D.DataBundle) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        prepared = pickle.load(handle)
    if prepared.get("format") != "ses-v0-prepared-v1":
        raise ValueError("prepared artifact format mismatch")
    if prepared.get("raw_cache_sha256") != C.RAW_CACHE_SHA256:
        raise ValueError("prepared artifact raw-cache mismatch")
    if prepared.get("split_identity_sha256") != bundle.hashes["split_identity"]:
        raise ValueError("prepared artifact split mismatch")
    if len(prepared["full_structs"]) != int(bundle.labels.size):
        raise ValueError("prepared artifact row count mismatch")
    return prepared


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def _chunk_structs(rows: Sequence[int], prepared: Mapping[str, Any]
                   ) -> tuple[list[S.Struct], np.ndarray, np.ndarray]:
    structs: list[S.Struct] = []
    mol_index: list[int] = []
    ptr = [0]
    for i, row in enumerate(rows):
        cs = prepared["candidates"][int(row)]
        structs.extend(cs)
        mol_index.extend([i] * len(cs))
        ptr.append(len(structs))
    return structs, np.asarray(mol_index, np.int64), np.asarray(ptr, np.int64)


@torch.no_grad()
def eval_arm(model: Any, selector: nn.Module | None, arm: str,
             rows: np.ndarray, prepared: Mapping[str, Any], bundle: D.DataBundle,
             device: Any, batch_mol: int = 64, *,
             return_candidates: bool = False) -> dict[str, Any]:
    rows = np.asarray(rows, np.int64)
    model.eval()
    if selector is not None:
        selector.eval()
    y_raw = bundle.y_raw(rows)
    y_std = bundle.y_std(rows)
    mean, std = float(bundle.scaler["mean"]), float(bundle.scaler["std"])
    if arm == "DENSE":
        pred = np.zeros(rows.size, np.float64)
        for start in range(0, rows.size, int(batch_mol)):
            part = rows[start:start + int(batch_mol)]
            structs = [prepared["full_structs"][int(r)] for r in part.tolist()]
            batch = T.collate(structs, device)
            out = model(batch, arm="SENTENCE_BASE")
            pred[start:start + part.size] = (
                out["yhat_std"].detach().cpu().numpy().astype(np.float64)) * std + mean
        return {"rows": rows, "y_raw": y_raw, "pred_raw": pred}

    n = int(rows.size)
    pred_fixed = np.zeros(n, np.float64)
    pred_select = np.zeros(n, np.float64)
    pred_mean = np.zeros(n, np.float64)
    mse_expected = np.zeros(n, np.float64)
    mae_expected = np.zeros(n, np.float64)
    chosen_index = np.zeros(n, np.int64)
    fixed = np.asarray(prepared["fixed_index"], np.int64)
    all_preds: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    cand_ptr_all = [0]
    for start in range(0, n, int(batch_mol)):
        part = rows[start:start + int(batch_mol)]
        structs, mol_index, ptr = _chunk_structs(part, prepared)
        batch = T.collate(structs, device)
        out = model(batch, arm="SENTENCE_BASE",
                    return_aux=(selector is not None))
        pred_std = out["yhat_std"].detach().cpu().numpy().astype(np.float64)
        preds = pred_std * std + mean
        scores = None
        if selector is not None:
            scores = selector(out["readout_in"].detach()).view(-1).detach().cpu().numpy()
            scores = scores.astype(np.float64)
        for i, row in enumerate(part.tolist()):
            lo, hi = int(ptr[i]), int(ptr[i + 1])
            pm = preds[lo:hi]
            y = y_raw[start + i]
            se = (pm - y) ** 2
            mse_expected[start + i] = float(se.mean())
            mae_expected[start + i] = float(np.abs(pm - y).mean())
            pred_mean[start + i] = float(pm.mean())
            fi = int(fixed[row])
            if fi >= pm.size:
                raise RuntimeError("fixed candidate index out of range")
            pred_fixed[start + i] = pm[fi]
            if scores is not None:
                j = int(np.argmax(scores[lo:hi]))
                chosen_index[start + i] = j
                pred_select[start + i] = pm[j]
            all_preds.append(pm.astype(np.float64))
        if scores is not None and return_candidates:
            all_scores.append(scores)
        if return_candidates:
            cand_ptr_all.extend((np.asarray([x.size for x in all_preds[-len(part):]])
                                 .cumsum() + cand_ptr_all[-1]).tolist())
    out: dict[str, Any] = {
        "rows": rows, "y_raw": y_raw, "pred_fixed": pred_fixed,
        "pred_select": pred_select if selector is not None else None,
        "pred_mean": pred_mean, "mse_expected": mse_expected,
        "mae_expected": mae_expected, "chosen_index": chosen_index,
    }
    if return_candidates:
        out["cand_ptr"] = np.asarray(cand_ptr_all, np.int64)
        out["cand_pred"] = (np.concatenate(all_preds) if all_preds
                            else np.zeros(0, np.float64))
        out["cand_scores"] = (np.concatenate(all_scores) if all_scores else None)
        out["fixed_index"] = fixed[rows]
    return out


def metrics_dense(result: Mapping[str, Any]) -> dict[str, float]:
    y = result["y_raw"]
    p = result["pred_raw"]
    return {"rmse": C.rmse(y, p), "mae": C.mae(y, p)}


def metrics_candidates(result: Mapping[str, Any], arm: str) -> dict[str, float]:
    y = result["y_raw"]
    out = {
        "rmse": float(np.sqrt(result["mse_expected"].mean())),
        "mae": float(result["mae_expected"].mean()),
        "rmse_expected": float(np.sqrt(result["mse_expected"].mean())),
        "mae_expected": float(result["mae_expected"].mean()),
        "mse_expected": float(result["mse_expected"].mean()),
        "rmse_fixed": C.rmse(y, result["pred_fixed"]),
        "mae_fixed": C.mae(y, result["pred_fixed"]),
    }
    if arm == "SELECT":
        out.update({
            "rmse": C.rmse(y, result["pred_select"]),
            "mae": C.mae(y, result["pred_select"]),
            "rmse_select": C.rmse(y, result["pred_select"]),
            "mae_select": C.mae(y, result["pred_select"]),
        })
    return out


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _loss_dense(model: Any, batch: Mapping[str, Any], y_std: np.ndarray,
                device: Any) -> torch.Tensor:
    y = torch.as_tensor(y_std, dtype=torch.float32, device=device)
    out = model(batch, arm="SENTENCE_BASE")
    return torch.mean((out["yhat_std"] - y) ** 2)


def _candidate_loss(model: Any, selector: nn.Module | None, arm: str,
                    structs: Sequence[S.Struct], mol_index: np.ndarray,
                    y_std: np.ndarray, n_mol: int, device: Any,
                    mixture: float = 0.9) -> tuple[torch.Tensor, dict[str, Any]]:
    batch = T.collate(structs, device)
    out = model(batch, arm="SENTENCE_BASE", return_aux=(selector is not None))
    y = torch.as_tensor(y_std, dtype=torch.float32, device=device)
    mol = torch.as_tensor(mol_index, dtype=torch.long, device=device)
    se = (out["yhat_std"] - y[mol]) ** 2
    count = torch.zeros(int(n_mol), dtype=se.dtype, device=device)
    count.index_add_(0, mol, torch.ones_like(se))
    total = torch.zeros(int(n_mol), dtype=se.dtype, device=device)
    total.index_add_(0, mol, se)
    if arm == "RANDOM":
        loss = torch.mean(total / count.clamp(min=1.0))
        return loss, {"m": count.detach().cpu().numpy()}
    if selector is None:
        raise ValueError("SELECT needs a selector")
    scores = selector(out["readout_in"].detach()).view(-1)
    mx = torch.full((int(n_mol),), -1e30, dtype=scores.dtype, device=device)
    mx.scatter_reduce_(0, mol, scores, reduce="amax", include_self=True)
    e = torch.exp(scores - mx[mol])
    denom = torch.zeros(int(n_mol), dtype=e.dtype, device=device)
    denom.index_add_(0, mol, e)
    p = e / denom[mol].clamp(min=1e-30)
    p_train = float(mixture) * p + (1.0 - float(mixture)) / count[mol]
    expected = torch.zeros(int(n_mol), dtype=se.dtype, device=device)
    expected.index_add_(0, mol, p_train * se)
    loss = torch.mean(expected)
    return loss, {"m": count.detach().cpu().numpy(),
                  "p_mean": p.detach().mean().cpu().item()}


def _save_best(path: Path, model: Any, selector: nn.Module | None,
               payload: Mapping[str, Any]) -> None:
    state = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
             "payload": dict(payload)}
    if selector is not None:
        state["selector"] = {k: v.detach().cpu() for k, v in selector.state_dict().items()}
    torch.save(state, path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    t_process = time.time()
    if args.arm not in ARMS:
        raise KeyError(args.arm)
    if args.arm == "DENSE" and args.selector:
        raise ValueError("DENSE has no selector")
    C.seed_everything(int(args.seed))
    cfg = C.load_config()
    recipe = dict(cfg["recipe"])
    topology = C.assert_frozen_sources()
    bundle = D.load_bundle()
    atom_dims, bond_dims = D.atom_bond_dims()
    prepared = load_prepared(Path(args.prepared) if args.prepared
                             else C.PREPARED_DIR / "ses_prepared.pkl", bundle)
    out_dir = Path(args.out_dir) if args.out_dir else (C.RESULTS_DIR / args.arm.lower())
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    log_handle = log_path.open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model, info = C.build_predictor(atom_dims, bond_dims, bundle, cfg, int(args.seed))
    zeroed = model.zero_unseen(bundle.atom_ok, bundle.bond_ok, atom_dims, bond_dims)
    init_hashes = dict(info["init_hashes"])
    init_hashes["_model_state"] = C.array_sha256(*[
        p.detach().cpu().numpy() for _, p in sorted(model.state_dict().items())])
    model = model.to(device)
    selector = None
    if args.arm == "SELECT":
        selector = build_selector(model.readout_in_dim, int(args.seed)).to(device)
    predictor_params = int(sum(p.numel() for p in model.parameters()))
    selector_params = selector_parameters(selector)
    if predictor_params != C.PARAMS_TOTAL:
        raise RuntimeError(f"unexpected predictor parameters {predictor_params}")

    epochs = int(args.epochs if args.epochs is not None
                 else cfg["horizon"]["target_epochs"])
    if epochs not in [int(x) for x in cfg["horizon"]["allowed"]]:
        raise ValueError(f"horizon {epochs} outside the frozen allowed set")
    batch_mol = int(recipe["batch_molecules"])
    n_train = int(bundle.train_rows.size)
    steps_per_epoch = int(math.ceil(n_train / batch_mol))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(recipe["warmup_epochs"]) * steps_per_epoch
    params = list(model.parameters()) + (list(selector.parameters()) if selector is not None else [])
    opt = torch.optim.AdamW(params, lr=float(recipe["peak_lr"]),
                            weight_decay=float(recipe["weight_decay"]))
    log(f"run arm={args.arm} seed={args.seed} device={device} host={socket.gethostname()} "
        f"commit={C.git_commit()[:12]} epochs={epochs}")
    log(f"split train/valid/test={n_train}/{bundle.valid_rows.size}/{bundle.test_rows.size} "
        f"labels={bundle.hashes['labels']}")
    log(f"params predictor={predictor_params} selector={selector_params} "
        f"init={init_hashes['_model_state']}")
    log(f"prepared candidates={prepared['stats']['n_candidates']} "
        f"build_seconds={prepared['stats']['build_seconds']:.2f} "
        f"fixed_index={prepared['stats']['fixed_index']['sha256'][:12]}")
    log(f"recipe peak_lr={recipe['peak_lr']} wd={recipe['weight_decay']} "
        f"warmup={warmup_steps} total_steps={total_steps}")
    log(f"zeroed_unseen={zeroed} target_scaler={bundle.scaler}")

    curve: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = -1
    best_metrics: dict[str, Any] = {}
    best_state: Any = None
    epoch_times: list[float] = []
    global_step = 0
    t_train_start = time.time()
    stop_after = int(args.max_steps) if args.max_steps else None
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        model.train()
        if selector is not None:
            selector.train()
        perm = np.random.default_rng(int(args.seed) * 1_000_003 + epoch).permutation(
            np.asarray(bundle.train_rows, np.int64))
        loss_sum = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_mol):
            part = perm[start:start + batch_mol]
            lr = C.lr_at(global_step, warmup_steps, total_steps,
                         float(recipe["peak_lr"]), float(recipe["min_lr"]))
            for group in opt.param_groups:
                group["lr"] = lr
            y_std = bundle.y_std(part)
            if args.arm == "DENSE":
                structs = [prepared["full_structs"][int(r)] for r in part.tolist()]
                batch = T.collate(structs, device)
                loss = _loss_dense(model, batch, y_std, device)
            else:
                structs, mol_index, _ = _chunk_structs(part, prepared)
                loss, _ = _candidate_loss(model, selector, args.arm, structs,
                                          mol_index, y_std, part.size, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(recipe["gradient_clip"]))
            opt.step()
            loss_sum += float(loss.detach().cpu())
            n_batches += 1
            global_step += 1
            if stop_after is not None and global_step >= stop_after:
                break
        # full eval-mode evaluation of every split
        train_res = eval_arm(model, selector, args.arm, bundle.train_rows, prepared,
                             bundle, device, int(recipe["eval_batch_molecules"]))
        valid_res = eval_arm(model, selector, args.arm, bundle.valid_rows, prepared,
                             bundle, device, int(recipe["eval_batch_molecules"]))
        test_res = eval_arm(model, selector, args.arm, bundle.test_rows, prepared,
                            bundle, device, int(recipe["eval_batch_molecules"]))
        if args.arm == "DENSE":
            tr, va, te = (metrics_dense(train_res), metrics_dense(valid_res),
                          metrics_dense(test_res))
        else:
            tr = metrics_candidates(train_res, args.arm)
            va = metrics_candidates(valid_res, args.arm)
            te = metrics_candidates(test_res, args.arm)
        score = va["rmse"] if args.arm != "RANDOM" else va["rmse_expected"]
        row: dict[str, Any] = {"epoch": epoch, "lr": lr,
                               "train_loss": loss_sum / max(1, n_batches)}
        for prefix, vals in (("train", tr), ("valid", va), ("test", te)):
            for key, value in vals.items():
                row[f"{prefix}_{key}"] = value
        curve.append(row)
        epoch_times.append(time.time() - t_epoch)
        if score < best_score - 1e-12:
            best_score = score
            best_epoch = epoch
            best_metrics = {"train": tr, "valid": va, "test": te}
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "selector": (copy.deepcopy(selector.state_dict())
                             if selector is not None else None),
            }
        log(f"epoch {epoch}/{epochs} lr={lr:.3e} loss={row['train_loss']:.5f} "
            f"train_rmse={tr['rmse']:.5f} valid_rmse={va['rmse']:.5f} "
            f"test_rmse={te['rmse']:.5f} "
            f"valid_expected={va.get('rmse_expected', float('nan')):.5f} "
            f"epoch_seconds={epoch_times[-1]:.2f}")
        if stop_after is not None and global_step >= stop_after:
            break
    train_seconds = time.time() - t_train_start
    if best_state is None:
        raise RuntimeError("no checkpoint was selected")

    model.load_state_dict(best_state["model"])
    if selector is not None:
        selector.load_state_dict(best_state["selector"])
    model.eval()
    if selector is not None:
        selector.eval()

    # full per-molecule / per-candidate records for the best checkpoint
    pred_path = out_dir / "predictions.npz"
    records: dict[str, np.ndarray] = {}
    for split_name, split_rows in (("train", bundle.train_rows),
                                   ("valid", bundle.valid_rows),
                                   ("test", bundle.test_rows)):
        res = eval_arm(model, selector, args.arm, split_rows, prepared, bundle,
                       device, int(recipe["eval_batch_molecules"]),
                       return_candidates=True)
        rows = res["rows"]
        records[f"{split_name}_rows"] = rows
        records[f"{split_name}_y_raw"] = res["y_raw"]
        if args.arm == "DENSE":
            records[f"{split_name}_pred_raw"] = res["pred_raw"]
            continue
        records[f"{split_name}_pred_fixed"] = res["pred_fixed"]
        records[f"{split_name}_pred_mean"] = res["pred_mean"]
        records[f"{split_name}_mse_expected"] = res["mse_expected"]
        records[f"{split_name}_mae_expected"] = res["mae_expected"]
        records[f"{split_name}_fixed_index"] = res["fixed_index"]
        records[f"{split_name}_cand_ptr"] = res["cand_ptr"]
        records[f"{split_name}_cand_pred"] = res["cand_pred"]
        if res["pred_select"] is not None:
            records[f"{split_name}_pred_select"] = res["pred_select"]
            records[f"{split_name}_chosen_index"] = res["chosen_index"]
            records[f"{split_name}_cand_scores"] = res["cand_scores"]
        # candidate supports (molecule-local atoms; global bond indices)
        node_parts: list[np.ndarray] = []
        bond_parts: list[np.ndarray] = []
        node_ptr = [0]
        bond_ptr = [0]
        for row in rows.tolist():
            for st in prepared["candidates"][int(row)]:
                node_parts.append(st.support_atoms.astype(np.int64))
                bond_parts.append(st.support_bonds.astype(np.int64))
                node_ptr.append(node_ptr[-1] + int(st.support_atoms.size))
                bond_ptr.append(bond_ptr[-1] + int(st.support_bonds.size))
            records.setdefault(f"{split_name}_orig_n_atoms", []).append(
                int(prepared["full_structs"][int(row)].n_atoms))
            records.setdefault(f"{split_name}_orig_n_bonds", []).append(
                int(prepared["full_structs"][int(row)].n_bonds))
        records[f"{split_name}_cand_node_ptr"] = np.asarray(node_ptr, np.int64)
        records[f"{split_name}_cand_nodes"] = (
            np.concatenate(node_parts) if node_parts else np.zeros(0, np.int64))
        records[f"{split_name}_cand_bond_ptr"] = np.asarray(bond_ptr, np.int64)
        records[f"{split_name}_cand_bonds"] = (
            np.concatenate(bond_parts) if bond_parts else np.zeros(0, np.int64))
    for key in list(records):
        if isinstance(records[key], list):
            records[key] = np.asarray(records[key], np.int64)
    np.savez_compressed(pred_path, **records)

    wall_seconds = time.time() - t_process
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    payload = {
        "arm": args.arm, "seed": int(args.seed), "epochs": epochs,
        "best_epoch": int(best_epoch), "best_metrics": best_metrics,
        "best_score": float(best_score),
        "predictor_parameters": predictor_params,
        "selector_parameters": selector_params,
        "init_hashes": init_hashes,
        "topology": topology,
        "commit": C.git_commit(), "host": socket.gethostname(), "device": str(device),
        "gpu_name": gpu_name, "python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "prepared": {"path": str(args.prepared) if args.prepared
                     else str(C.PREPARED_DIR / "ses_prepared.pkl"),
                     "raw_cache_sha256": prepared["raw_cache_sha256"],
                     "split_identity_sha256": prepared["split_identity_sha256"],
                     "n_candidates": prepared["stats"]["n_candidates"],
                     "build_seconds": prepared["stats"]["build_seconds"]},
        "data_hashes": bundle.hashes,
        "target_scaler": bundle.scaler,
        "config": cfg,
        "wall_seconds": float(wall_seconds),
        "train_seconds": float(train_seconds),
        "epoch_seconds": epoch_times,
        "gpu_hours_estimate": float(wall_seconds / 3600.0)
        if torch.cuda.is_available() else 0.0,
        "curve_path": str((out_dir / "curve.csv").resolve()),
        "predictions_path": str(pred_path.resolve()),
        "code_sha256": C.code_sha(TRAIN_FILES),
        "max_steps_override": int(args.max_steps) if args.max_steps else None,
    }
    _save_best(out_dir / "best.pt", model, selector, payload)
    C.write_csv(out_dir / "curve.csv", curve)
    C.write_json(out_dir / "manifest.json", payload)
    log(f"best_epoch={best_epoch} best_valid={best_score:.6f} "
        f"wall={wall_seconds:.1f}s saved={out_dir}")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=list(ARMS))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--prepared", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="preflight only; formal runs leave this unset")
    parser.add_argument("--selector", action="store_true",
                        help="unused compatibility flag; SELECT implies a selector")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(args)
