"""Training / validation loop for the six ZINC architecture-discovery candidates.

One process trains one candidate/seed.  Valid is evaluated every epoch and the
single best-valid checkpoint is selected (ties -> earlier epoch).  Test is never
constructed.  Online train MAE is the within-epoch pre-update running mean;
the reported ``train_mae_eval_mode`` is a separate fixed-model measurement.
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

import ad_common as C
import ad_data as D
import ad_models as M

# Pre-declared recipes.  Epoch counts are the planned initial values; the
# timing job may lower them before the formal run.  The formal run records the
# actual horizon and never resets the cosine during training.
RECIPES: dict[str, dict[str, Any]] = {
    "R0": dict(optimizer="adam", lr=1.0e-3, min_lr=1.0e-5, weight_decay=1.0e-5,
               warmup_steps=500, epochs=360, dropout=0.0, grad_clip=5.0,
               batch_size=128, valid_batch_size=256),
    "R1": dict(optimizer="adamw", lr=8.0e-4, min_lr=1.0e-5, weight_decay=1.0e-4,
               warmup_steps=500, epochs=300, dropout=0.0, grad_clip=5.0,
               batch_size=128, valid_batch_size=256),
    "T": dict(optimizer="adamw", lr=1.0e-3, min_lr=1.0e-5, weight_decay=1.0e-5,
              warmup_steps=500, epochs=360, dropout=0.0, grad_clip=5.0,
              batch_size=128, valid_batch_size=256),
    "P0": dict(optimizer="adamw", lr=1.0e-3, min_lr=1.0e-5, weight_decay=1.0e-5,
               warmup_steps=500, epochs=300, dropout=0.0, grad_clip=5.0,
               batch_size=128, valid_batch_size=16),
    "P1": dict(optimizer="adamw", lr=8.0e-4, min_lr=1.0e-5, weight_decay=1.0e-4,
               warmup_steps=500, epochs=240, dropout=0.0, grad_clip=5.0,
               batch_size=128, valid_batch_size=8),
    "B": dict(optimizer="adamw", lr=1.0e-3, min_lr=1.0e-5, weight_decay=1.0e-4,
              warmup_steps=500, epochs=300, dropout=0.0, grad_clip=5.0,
              batch_size=128, valid_batch_size=256),
}


def run_dir(candidate: str, seed: int, tag: str | None = None) -> Path:
    prefix = f"{tag}_" if tag else ""
    return C.RUNS / f"{prefix}{candidate}_s{int(seed)}"


def _to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, tuple):
            out[k] = tuple(x.to(device, non_blocking=True) if torch.is_tensor(x) else x for x in v)
        elif isinstance(v, list):
            out[k] = [x.to(device, non_blocking=True) if torch.is_tensor(x) else x for x in v]
        else:
            out[k] = v
    return out


def _ranges(ptr: np.ndarray, rows: np.ndarray):
    rows = np.asarray(rows, dtype=np.int64)
    starts = ptr[rows].astype(np.int64)
    ends = ptr[rows + 1].astype(np.int64)
    counts = ends - starts
    idx = np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in zip(starts, ends)])
    batch = np.repeat(np.arange(len(rows), dtype=np.int64), counts)
    local = np.concatenate([np.arange(c, dtype=np.int64) for c in counts])
    return idx, batch, local, counts


def collate(cache: Mapping[str, Any], rows: Sequence[int], candidate: str, *,
            valid: bool = False, walk_cols: np.ndarray | None = None) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=np.int64)
    B = len(rows)
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    atom_idx, atom_batch, atom_local, atom_counts = _ranges(atom_ptr, rows)
    bond_idx, bond_batch, _, bond_counts = _ranges(bond_ptr, rows)
    global_feat = torch.from_numpy(np.asarray(cache["global_feat"])[rows].astype(np.float32))
    atom_type = torch.from_numpy(np.asarray(cache["atom_type"])[atom_idx].astype(np.int64))
    bond_type = torch.from_numpy(np.asarray(cache["bond_type"])[bond_idx].astype(np.int64))

    if candidate in ("R0", "R1"):
        batch: dict[str, Any] = {
            "n_graphs": B,
            "atom_feat": torch.from_numpy(np.asarray(cache["atom_feat"])[atom_idx].astype(np.float32)),
            "atom_type": atom_type,
            "atom_batch": torch.from_numpy(atom_batch),
            "atom_local": torch.from_numpy(atom_local),
            "joint": torch.from_numpy(np.asarray(cache["root_joint"])[atom_idx]),
            "shell": torch.from_numpy(np.asarray(cache["root_shell"])[atom_idx]),
            "root_scalar": torch.from_numpy(np.asarray(cache["root_scalar"])[atom_idx].astype(np.float32)),
            "global": global_feat,
            "bond_type": bond_type,
            "bond_batch": torch.from_numpy(bond_batch),
        }
        if candidate == "R1":
            Nmax = int(atom_counts.max())
            mask = torch.arange(Nmax).view(1, Nmax) < torch.from_numpy(atom_counts).view(B, 1)
            dist_pad = torch.zeros((B, Nmax, Nmax), dtype=torch.int64)
            dist_ptr = np.asarray(cache["dist_ptr"])
            dist_flat = np.asarray(cache["dist_flat"])
            for i, row in enumerate(rows):
                n = int(atom_counts[i])
                dist_pad[i, :n, :n] = torch.from_numpy(
                    dist_flat[dist_ptr[row]:dist_ptr[row + 1]].reshape(n, n).astype(np.int64)
                )
            th_ptr = np.asarray(cache["twohop_ptr"])
            us, vs, cats, cnts, bs = [], [], [], [], []
            for i, row in enumerate(rows):
                s, e = int(th_ptr[row]), int(th_ptr[row + 1])
                if e > s:
                    us.append(np.asarray(cache["twohop_u"])[s:e])
                    vs.append(np.asarray(cache["twohop_v"])[s:e])
                    cats.append(np.asarray(cache["twohop_cat"])[s:e])
                    cnts.append(np.asarray(cache["twohop_count"])[s:e])
                    bs.append(np.full(e - s, i, dtype=np.int64))
            if us:
                twohop = (
                    torch.from_numpy(np.concatenate(bs).astype(np.int64)),
                    torch.from_numpy(np.concatenate(us).astype(np.int64)),
                    torch.from_numpy(np.concatenate(vs).astype(np.int64)),
                    torch.from_numpy(np.concatenate(cats).astype(np.int64)),
                    torch.from_numpy(np.concatenate(cnts).astype(np.int64)),
                )
            else:
                twohop = tuple(torch.zeros(0, dtype=torch.int64) for _ in range(5))
            batch.update({"Nmax": Nmax, "mask": mask, "dist_pad": dist_pad, "twohop": twohop})
        return batch

    if candidate == "T":
        junc_ptr = np.asarray(cache["junc_ptr"])
        junc_idx, junc_batch, _, junc_counts = _ranges(junc_ptr, rows)
        # Batch-local offsets, not dataset-global offsets: the encoders index
        # the concatenated batch tensors.
        atom_off = np.concatenate([[0], np.cumsum(atom_counts)[:-1]]).astype(np.int64)
        bond_off = np.concatenate([[0], np.cumsum(bond_counts)[:-1]]).astype(np.int64)
        bi = np.asarray(cache["bond_index"])[bond_idx].astype(np.int64)
        bi = bi + atom_off[bond_batch].reshape(-1, 1)
        ji = np.asarray(cache["junc_index"])[junc_idx].astype(np.int64)
        ji = ji + atom_off[junc_batch].reshape(-1, 1)
        jb = np.asarray(cache["junc_bond"])[junc_idx].astype(np.int64)
        jb = jb + bond_off[junc_batch].reshape(-1, 1)
        return {
            "n_graphs": B,
            "atom_feat": torch.from_numpy(np.asarray(cache["atom_feat"])[atom_idx].astype(np.float32)),
            "atom_batch": torch.from_numpy(atom_batch),
            "atom_type": atom_type,
            "bond_feat": torch.from_numpy(np.asarray(cache["bond_feat"])[bond_idx].astype(np.float32)),
            "bond_type": bond_type,
            "bond_batch": torch.from_numpy(bond_batch),
            "bond_index": torch.from_numpy(bi),
            "junc_feat": torch.from_numpy(np.asarray(cache["junc_feat"])[junc_idx].astype(np.float32)),
            "junc_batch": torch.from_numpy(junc_batch),
            "junc_index": torch.from_numpy(ji),
            "junc_bond": torch.from_numpy(jb),
            "global": global_feat,
        }

    if candidate in ("P0", "P1"):
        if valid:
            S = D.WALK_BANK
            cols = np.arange(D.WALK_BANK, dtype=np.int64)
        else:
            if walk_cols is None:
                raise ValueError("training collate for P0/P1 needs walk_cols")
            S = len(walk_cols)
            cols = np.asarray(walk_cols, dtype=np.int64)
        groups = (cols // D.WALK_PER_LENGTH).astype(np.int64)
        lengths = np.asarray(D.WALK_LENGTHS)[groups].astype(np.int64)
        order = np.asarray(cache["walk_order"])[rows]
        gather = order[:, cols]  # [B,S]
        wa = np.asarray(cache["walk_atom"])[rows[:, None], gather]  # [B,S,9]
        wb = np.asarray(cache["walk_bond"])[rows[:, None], gather]  # [B,S,8]
        wn = np.asarray(cache["walk_anon"])[rows[:, None], gather]
        wb_pad = np.concatenate(
            [wb, np.zeros((B, S, 1), dtype=np.uint8)], axis=2
        )
        return {
            "n_graphs": B,
            "walk_atom": torch.from_numpy(wa.reshape(B * S, -1).astype(np.int64)),
            "walk_bond": torch.from_numpy(wb_pad.reshape(B * S, -1).astype(np.int64)),
            "walk_anon": torch.from_numpy(wn.reshape(B * S, -1).astype(np.int64)),
            "walk_len": torch.from_numpy(np.tile(lengths, B).astype(np.int64)),
            "walk_group": torch.from_numpy(np.tile(groups, B).astype(np.int64)),
            "walk_graph": torch.from_numpy(np.repeat(np.arange(B), S).astype(np.int64)),
            "walk_local": torch.from_numpy(np.tile(np.arange(S), B).astype(np.int64)),
            "S": int(S),
            "atom_count": torch.from_numpy(np.asarray(cache["atom_count_graph"])[rows].astype(np.float32)),
            "bond_count": torch.from_numpy(np.asarray(cache["bond_count_graph"])[rows].astype(np.float32)),
            "n_atoms": torch.from_numpy(np.asarray(cache["n_atoms_graph"])[rows].astype(np.float32)),
            "n_bonds": torch.from_numpy(np.asarray(cache["n_bonds_graph"])[rows].astype(np.float32)),
            "global": global_feat,
        }

    if candidate == "B":
        block_ptr = np.asarray(cache["block_ptr"])
        block_idx, block_batch, block_local, block_counts = _ranges(block_ptr, rows)
        feat = np.concatenate(
            [
                np.asarray(cache["block_joint"])[block_idx],
                np.asarray(cache["block_atom_type"])[block_idx],
                np.asarray(cache["block_bond_type"])[block_idx],
                np.asarray(cache["block_artic_type"])[block_idx],
                np.asarray(cache["block_scalar"])[block_idx].astype(np.float32),
            ],
            axis=1,
        )
        Bmax = int(block_counts.max())
        pad = torch.zeros((B, Bmax, feat.shape[1]), dtype=torch.float32)
        pad[torch.from_numpy(block_batch), torch.from_numpy(block_local)] = torch.from_numpy(feat)
        mask = torch.arange(Bmax).view(1, Bmax) < torch.from_numpy(block_counts).view(B, 1)
        dist_pad = torch.zeros((B, Bmax, Bmax), dtype=torch.int64)
        overlap_pad = torch.zeros((B, Bmax, Bmax), dtype=torch.int64)
        pair_ptr = np.asarray(cache["block_pair_ptr"])
        for i, row in enumerate(rows):
            nb = int(block_counts[i])
            s, e = int(pair_ptr[row]), int(pair_ptr[row + 1])
            dist_pad[i, :nb, :nb] = torch.from_numpy(
                np.asarray(cache["block_dist"])[s:e].reshape(nb, nb).astype(np.int64)
            )
            overlap_pad[i, :nb, :nb] = torch.from_numpy(
                np.asarray(cache["block_overlap"])[s:e].reshape(nb, nb).astype(np.int64)
            )
        return {
            "n_graphs": B,
            "block_feat": pad,
            "block_mask": mask,
            "block_dist": dist_pad,
            "block_overlap": overlap_pad,
            "atom_count": torch.from_numpy(np.asarray(cache["atom_count_graph"])[rows].astype(np.float32)),
            "bond_count": torch.from_numpy(np.asarray(cache["bond_count_graph"])[rows].astype(np.float32)),
            "n_atoms": torch.from_numpy(np.asarray(cache["n_atoms_graph"])[rows].astype(np.float32)),
            "n_bonds": torch.from_numpy(np.asarray(cache["n_bonds_graph"])[rows].astype(np.float32)),
            "global": global_feat,
        }

    raise ValueError(f"unknown candidate {candidate!r}")


@torch.no_grad()
def evaluate(model: torch.nn.Module, cache: Mapping[str, Any], rows: np.ndarray, *, candidate: str,
             device, batch_size: int) -> tuple[np.ndarray, float]:
    model.eval()
    out = []
    with C.isolated_rng():
        for start in range(0, len(rows), int(batch_size)):
            chunk = rows[start:start + int(batch_size)]
            batch = _to_device(collate(cache, chunk, candidate, valid=True), device)
            pred = model(batch)
            out.append(pred.detach().float().cpu().numpy().reshape(-1))
    pred = np.concatenate(out, 0) if out else np.zeros(0, dtype=np.float32)
    y = np.asarray(cache["labels"], dtype=np.float64)[rows]
    return pred, C.mae(pred, y)


def _walk_cols(gen: torch.Generator) -> np.ndarray:
    cols = []
    for g in range(len(D.WALK_LENGTHS)):
        p = torch.randperm(D.WALK_PER_LENGTH, generator=gen)[:D.TRAIN_WALK_SAMPLE]
        cols.append((p + g * D.WALK_PER_LENGTH).numpy())
    return np.concatenate(cols).astype(np.int64)


def _param_audit(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    total = C.parameter_count(model)
    opt = C.optimizer_parameter_count(optimizer)
    requires = sum(1 for p in model.parameters() if p.requires_grad)
    nparams = sum(1 for p in model.parameters())
    return {
        "parameters": total,
        "optimizer_parameters": opt,
        "parameters_in_optimizer": bool(total == opt),
        "parameter_tensors": nparams,
        "requires_grad_tensors": requires,
        "all_parameters_require_grad": bool(requires == nparams),
    }


def train(candidate: str, seed: int, device, *, epochs: int | None = None,
          tag: str | None = None, force: bool = False) -> dict:
    recipe = dict(RECIPES[candidate])
    n_epochs = int(recipe["epochs"] if epochs is None else epochs)
    out = run_dir(candidate, seed, tag)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.json").exists() and not force:
        C.log(f"[{candidate} s{seed}] already complete; use --force to rerun")
        return C.read_json(out / "train.json")

    norm = D.load_norm()
    train_cache = D.load_cache("train")
    valid_cache = D.load_cache("valid")
    y_train = np.asarray(train_cache["labels"], dtype=np.float64)
    y_valid = np.asarray(valid_cache["labels"], dtype=np.float64)
    rows_train = np.arange(len(y_train), dtype=np.int64)
    rows_valid = np.arange(len(y_valid), dtype=np.int64)

    model, model_info = M.build_model(candidate, norm, seed, device)
    params = C.parameter_count(model)
    if params > 500_000:
        raise RuntimeError(f"{candidate}: {params} > 500k")
    if recipe["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    elif recipe["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    else:
        raise ValueError(recipe["optimizer"])
    audit = _param_audit(model, optimizer)
    if not audit["parameters_in_optimizer"]:
        raise RuntimeError(f"optimizer does not cover all parameters: {audit}")

    batch_size = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(len(rows_train) / batch_size)
    total_updates = steps_per_epoch * n_epochs
    warmup = min(int(recipe["warmup_steps"]), max(1, total_updates // 20))
    gen = torch.Generator().manual_seed(int(seed) + 101)
    curve: list[dict] = []
    best = {"valid_mae": float("inf"), "epoch": -1}
    best_state = None
    gradient_audit: dict[str, Any] = {}
    started = time.perf_counter()
    last_lr = float(recipe["lr"])
    global_step = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(rows_train), generator=gen).numpy()
        walk_cols = _walk_cols(gen) if candidate in ("P0", "P1") else None
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            batch = _to_device(collate(train_cache, rows, candidate, valid=False, walk_cols=walk_cols), device)
            target = torch.as_tensor(y_train[rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if epoch == 1 and start == 0:
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                nonzero = sum(1 for g in grads if torch.count_nonzero(g).item() > 0)
                gradient_audit = {
                    "grad_tensors": len(grads),
                    "nonzero_grad_tensors": int(nonzero),
                    "all_grad_finite": bool(all(torch.isfinite(g).all().item() for g in grads)),
                    "loss_finite": bool(torch.isfinite(loss).item()),
                    "first_batch_rows": int(len(rows)),
                }
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["grad_clip"]))
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at epoch {epoch}")
            global_step += 1
            last_lr = C.cosine_with_warmup_lr(
                global_step, total_updates, warmup, float(recipe["lr"]), float(recipe["min_lr"])
            )
            for group in optimizer.param_groups:
                group["lr"] = last_lr
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(rows)
        online = total_err / max(n_mol, 1)
        _, valid_mae = evaluate(
            model, valid_cache, rows_valid, candidate=candidate, device=device,
            batch_size=int(recipe["valid_batch_size"]),
        )
        sec = time.perf_counter() - t_ep
        curve.append({
            "epoch": int(epoch),
            "online_train_mae": float(online),
            "valid_mae": float(valid_mae),
            "lr": float(last_lr),
            "seconds": float(sec),
            "steps": int(steps_per_epoch),
        })
        if valid_mae < best["valid_mae"]:
            best = {"valid_mae": float(valid_mae), "epoch": int(epoch)}
            best_state = C.cpu_state(model)
            C.atomic_torch_save(
                {"state_dict": best_state, "epoch": int(epoch), "valid_mae": float(valid_mae),
                 "candidate": candidate, "seed": int(seed)},
                out / "best.pt",
            )
        C.log(
            f"[{candidate} s{seed}] ep={epoch:03d} valid={valid_mae:.6f} online={online:.6f} "
            f"lr={last_lr:.3e} best={best['valid_mae']:.6f}@{best['epoch']} {sec:.2f}s"
        )

    wall = time.perf_counter() - started
    C.atomic_torch_save(
        {
            "state_dict": C.cpu_state(model),
            "optimizer": optimizer.state_dict(),
            "gen_state": gen.get_state(),
            "epoch": n_epochs,
            "global_step": global_step,
            "best": best,
            "curve": curve,
            "candidate": candidate,
            "seed": int(seed),
        },
        out / "final.pt",
    )

    # Fixed-object evaluation of the selected best checkpoint.
    eval_batch = 64 if candidate in ("P0", "P1") else 256
    best_model, _ = M.build_model(candidate, norm, seed, device)
    C.load_state(best_model, best_state, device)
    best_pred_valid, best_valid = evaluate(
        best_model, valid_cache, rows_valid, candidate=candidate, device=device,
        batch_size=int(recipe["valid_batch_size"]),
    )
    _, best_train = evaluate(
        best_model, train_cache, rows_train, candidate=candidate, device=device,
        batch_size=eval_batch,
    )
    _, final_valid = evaluate(
        model, valid_cache, rows_valid, candidate=candidate, device=device,
        batch_size=int(recipe["valid_batch_size"]),
    )
    _, final_train = evaluate(
        model, train_cache, rows_train, candidate=candidate, device=device,
        batch_size=eval_batch,
    )
    pred_dir = out / "predictions"
    pred_dir.mkdir(exist_ok=True)
    np.savez(pred_dir / "valid_best.npz", row=rows_valid, pred=best_pred_valid)

    info = {
        "round": C.ROUND_ID,
        "candidate": candidate,
        "seed": int(seed),
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "data": {
            "source": "official PyG ZINC subset=True train/valid",
            "n_train": int(len(y_train)),
            "n_valid": int(len(y_valid)),
            "official_test_loaded": False,
            "train_label_sha256": C.sha256_array(y_train),
            "valid_label_sha256": C.sha256_array(y_valid),
            "train_cache_meta": train_cache["meta"],
            "valid_cache_meta": valid_cache["meta"],
        },
        "model_info": model_info,
        "parameter_count": int(params),
        "parameter_audit": audit,
        "recipe": {
            **recipe,
            "epochs": int(n_epochs),
            "steps_per_epoch": int(steps_per_epoch),
            "updates": int(total_updates),
            "warmup_steps": int(warmup),
            "actual_global_steps": int(global_step),
            "loss": "raw-y L1",
            "valid_every_epoch": True,
        },
        "gradient_audit": gradient_audit,
        "best_valid": best,
        "fixed_objects": {
            "best": {"valid_mae": float(best_valid), "train_mae_eval_mode": float(best_train),
                     "epoch": int(best["epoch"])},
            "final": {"valid_mae": float(final_valid), "train_mae_eval_mode": float(final_train),
                      "epoch": int(n_epochs)},
        },
        "curve": curve,
        "wall_seconds": float(wall),
        "gpu_hours": None if str(device).startswith("cpu") else float(wall) / 3600.0,
        "smoke_override": bool(epochs is not None),
        "device": str(device),
        "notes": [
            "online_train_mae is a within-epoch pre-update running mean",
            "train_mae_eval_mode is a full-10k eval-mode measurement of the fixed checkpoint",
            "selection is argmin(valid_mae); ties keep the earlier epoch",
        ],
    }
    C.write_json(out / "train.json", info)
    C.write_curve_csv(out / "curve.csv", curve)
    C.log(f"[{candidate} s{seed}] done best={best_valid:.6f}@{best['epoch']} "
          f"final={final_valid:.6f} wall={wall:.1f}s")
    return info


def timing(candidates: Sequence[str], seed: int, device, *, epochs: int = 2) -> dict:
    out = {}
    for cand in candidates:
        t0 = time.perf_counter()
        info = train(cand, seed, device, epochs=epochs, tag="timing", force=True)
        sec = time.perf_counter() - t0
        per_epoch = float(np.mean([e["seconds"] for e in info["curve"]]))
        out[cand] = {
            "timed_seconds": sec,
            "mean_epoch_seconds_including_valid": per_epoch,
            "projected_gpu_hours_360": per_epoch * 360 / 3600.0,
            "projected_gpu_hours_at_recipe_epochs": per_epoch * RECIPES[cand]["epochs"] / 3600.0,
            "parameters": info["parameter_count"],
            "gradient_audit": info["gradient_audit"],
            "best_valid_two_epoch": info["best_valid"]["valid_mae"],
        }
    C.write_json(C.ANALYSIS / "timing.json", out)
    C.log(out)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "timing"])
    p.add_argument("--candidate", default="R0")
    p.add_argument("--candidates", default="R0,R1,T,P0,P1,B")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if args.command == "timing":
        timing([x for x in args.candidates.split(",") if x], args.seed, device, epochs=args.epochs or 2)
    else:
        train(args.candidate, args.seed, device, epochs=args.epochs, tag=args.tag, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
