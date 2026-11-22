"""Seed-0 training for the three graph-level-main-loss arms.

Only the main task loss differs from the corresponding parent / CSOI trainer:

    loss_main = mean((f_bar[:n_main] - y[:n_main]) ** 2)
    loss      = loss_main + 0.25 * loss_cutresp_aux

The auxiliary CUTRESP pair loss, optimiser, schedule, batch composition,
initialisation, checkpoint rule and data view are inherited unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cpo_common as C


class RunData:
    def __init__(self, cfg: Mapping[str, Any], dataset: str, log=print,
                 train_subset: int = 0, valid_subset: int = 0) -> None:
        self.cfg = dict(cfg)
        self.dataset = dataset
        self.pack = C.load_pack(cfg, dataset)
        self.labels = np.asarray(self.pack["labels"], np.float64)
        self.row_ids = np.asarray(self.pack["row_ids"], np.int64)
        train = np.asarray(self.pack["train_idx"], np.int64)
        valid = np.asarray(self.pack["valid_idx"], np.int64)
        self.full_train = train.copy()
        if train_subset:
            train = train[:int(train_subset)]
        if valid_subset:
            valid = valid[:int(valid_subset)]
        self.splits = {"train": train, "valid": valid}
        self.scaler = C.target_scaler(self.labels, self.splits["train"])
        self.y_std = (self.labels - self.scaler["mean"]) / self.scaler["std"]
        self.atom_dims, self.bond_dims = C.official_feature_dims()
        log(f"  scaler mean={self.scaler['mean']:.6f} std={self.scaler['std']:.6f} "
            f"n_train={train.size} n_valid={valid.size} test_loaded=False")


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def raw_metrics(pred_std: np.ndarray, labels: np.ndarray,
                scaler: Mapping[str, float]) -> dict[str, float]:
    pred = np.asarray(pred_std, np.float64) * float(scaler["std"]) + float(scaler["mean"])
    return {"rmse": C.rmse(pred, labels), "mae": C.mae(pred, labels)}


def evaluate_epoch(model, family: str, data: RunData, cfg: Mapping[str, Any]) -> dict[str, Any]:
    recipe = cfg["recipe"]
    batch = int(recipe["eval_batch_parent"] if family == "parent" else recipe["eval_batch_csoi"])
    adapter = C.CutAdapter(model, family, data.pack, device=str(model.device))
    adapter.prepare()
    out: dict[str, Any] = {}
    preds: dict[str, np.ndarray] = {}
    for split, rows in data.splits.items():
        res = C.evaluate_cut_rows(adapter, data.pack, rows, data.y_std, data.scaler,
                                  batch=batch, collect_parses=False)
        preds[split] = res["per_molecule"]["f_bar_std"]
        out[split] = res
    return {"stats": out, "preds": preds, "adapter": adapter}


def save_predictions(path: Path, data: RunData, preds: Mapping[str, np.ndarray]) -> str:
    payload: dict[str, np.ndarray] = {}
    for split, rows in data.splits.items():
        p = np.asarray(preds[split], np.float64)
        payload[f"{split}_row_id"] = data.row_ids[rows]
        payload[f"{split}_label"] = data.labels[rows]
        payload[f"{split}_pred_std"] = p
        payload[f"{split}_pred"] = p * float(data.scaler["std"]) + float(data.scaler["mean"])
    np.savez(path, **payload)
    return C.sha256_file(path)


# ---------------------------------------------------------------------------
# gradient diagnostics
# ---------------------------------------------------------------------------


def group_grad_norms(model, family: str) -> dict[str, float]:
    named = dict(model.named_parameters())
    if family == "csoi":
        groups = model.module_groups()
    else:
        groups = {
            "frame_encoder": ["phi"],
            "fragment_encoder": ["enc"],
            "cut_head": ["a", "q", "r", "b", "beta", "null_cut", "h_empty"],
            "global_head": ["global_head"],
        }
    out: dict[str, float] = {}
    for group, prefixes in groups.items():
        total = 0.0
        for name, p in named.items():
            if not any(name == pref or name.startswith(pref + ".") for pref in prefixes):
                continue
            if p.grad is None:
                continue
            total += float(p.grad.detach().norm()) ** 2
        out[group] = math.sqrt(total)
    return out


def total_grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        total += float(p.grad.detach().norm()) ** 2
    return math.sqrt(total)


# ---------------------------------------------------------------------------
# post-training mechanism diagnostics
# ---------------------------------------------------------------------------


class _ZeroOut:
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)

    def __call__(self, x):
        import torch

        return torch.zeros((x.shape[0], self.dim), device=x.device, dtype=x.dtype)


def interaction_diagnostics(model, family: str, pack: Mapping[str, Any],
                            rows: np.ndarray, device: str) -> dict[str, Any]:
    import torch

    rows = np.asarray(rows, np.int64)[:16]
    idx = torch.as_tensor(rows, device=device)
    out: dict[str, Any] = {"n_molecules": int(rows.size), "family": family}
    try:
        with torch.no_grad():
            if family == "csoi":
                frames = model.frame_embeddings()
                pidx, _counts = model._parse_index(idx)
                s_ids = model.parse_S[pidx]
                c_ids = model.parse_C[pidx]
                bond = model.parse_cut_bond_row[pidx]
                full = model.cut_scores(s_ids, c_ids, bond, frames)
                additive = model._additive_only(s_ids, c_ids, bond, frames)
                diff = (full - additive).detach().double().cpu().numpy()
                out["interaction_contribution"] = {
                    "mean_abs": float(np.mean(np.abs(diff))),
                    "mean": float(np.mean(diff)),
                    "std": float(np.std(diff)),
                    "min": float(np.min(diff)),
                    "max": float(np.max(diff)),
                }
                if getattr(model, "interaction", False):
                    diag: dict[str, Any] = {}
                    model.molecule_cuts(idx, frames, diag=diag)
                    ent = diag.get("attention_entropy", [])
                    eff = diag.get("effective_tokens", [])
                    out["attention"] = {
                        "mean_entropy_nats": float(np.mean(ent)) if ent else float("nan"),
                        "mean_effective_tokens": float(np.mean(eff)) if eff else float("nan"),
                    }
            elif family == "parent" and model.arm == "FACTOR":
                phi = model.frame_phi()
                pidx, _counts = model._parse_index(idx)
                frag_s = model.parse_S[pidx]
                frag_c = model.parse_C[pidx]
                has_c = frag_c >= 0
                need = torch.cat([frag_s, frag_c[has_c]])
                uniq, inv = torch.unique(need, sorted=True, return_inverse=True)
                h = model.encode_fragments(uniq, phi)
                n_parse = int(frag_s.numel())
                h_s = h[inv[:n_parse]]
                h_c = model.h_empty[None, :].repeat(n_parse, 1)
                if bool(has_c.any()):
                    h_c = h_c.index_copy(0, torch.nonzero(has_c).squeeze(-1), h[inv[n_parse:]])
                e = model._cut_embedding(model.parse_cut_bond_row[pidx])
                full = model.cut_scores(h_c, h_s, e, has_c)
                q_orig, r_orig = model.q, model.r
                try:
                    model.q = _ZeroOut(model.rank)
                    model.r = _ZeroOut(model.rank)
                    additive = model.cut_scores(h_c, h_s, e, has_c)
                finally:
                    model.q, model.r = q_orig, r_orig
                diff = (full - additive).detach().double().cpu().numpy()
                out["interaction_contribution"] = {
                    "mean_abs": float(np.mean(np.abs(diff))),
                    "mean": float(np.mean(diff)),
                    "std": float(np.std(diff)),
                    "min": float(np.min(diff)),
                    "max": float(np.max(diff)),
                }
    except Exception as exc:  # diagnostics must never invalidate a completed run
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _patch(base: dict, patch: Mapping[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            _patch(base[key], value)
        else:
            base[key] = value


def train(args: argparse.Namespace) -> int:
    import torch

    C.set_deterministic(int(args.seed))
    cfg = dict(C.load_config())
    if args.overrides:
        _patch(cfg, json.loads(args.overrides))
    run_id = args.run_id
    rdir = C.run_dir(run_id)
    log_handle = (rdir / "train.log").open("a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    family = args.family
    arm = args.arm
    device = args.device
    log(f"run={run_id} family={family} arm={arm} device={device} epochs={args.epochs} seed={args.seed}")
    data = RunData(cfg, args.dataset, log=log,
                   train_subset=args.train_subset, valid_subset=args.valid_subset)
    model, zeroed = C.build_model(family, cfg, data.pack, arm, apply_support=True)
    model = model.to(device)
    ledger = model.parameter_ledger()
    hashes = {
        "init_state": None,
    }
    # capture the shared-module init hashes exactly as the parent/CSOI rounds did
    prefixes = (["phi", "token_proj", "bucket_emb", "mult_proj", "token_ln",
                 "cut_proj", "null_cut", "null_ctx", "add_s", "add_c", "bond_bias", "beta"]
                if family == "csoi"
                else ["phi", "enc", "a", "q", "r", "b", "head", "a_add",
                      "global_head", "null_cut", "beta", "h_empty"])
    hashes = PM_module_hashes(model, prefixes)
    log(f"  params={ledger['total']} ledger={ledger}")
    log(f"  init_hashes={hashes}")
    log(f"  zeroed_category_rows={zeroed}")

    # The pair directory is always built from the full official Train split.
    # For formal runs this is identical to data.splits["train"]; for smoke runs
    # it only keeps the auxiliary sampler non-empty.  It never touches Valid.
    store = C.PPAIR.PairStore(cfg, data.pack, data.labels, None,
                              data.full_train, data.row_ids, log=log)
    log(f"  pair store: {store.summary()}")
    sampler = store.sampler()

    recipe = cfg["recipe"]
    n_train = int(data.splits["train"].size)
    steps_per_epoch = int(math.ceil(n_train / int(recipe["batch_molecules"])))
    warmup_steps = int(recipe["warmup_epochs"]) * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=float(recipe["peak_lr"]),
                            weight_decay=float(recipe["weight_decay"]))
    epochs_target = int(args.epochs)
    if epochs_target <= 0:
        epochs_target = int(recipe["epochs_frozen"])
    total_steps = epochs_target * steps_per_epoch
    log(f"  horizon E={epochs_target} steps_per_epoch={steps_per_epoch}")

    epoch_rng = np.random.default_rng(int(args.seed) + 17)
    orders = [epoch_rng.permutation(n_train) for _ in range(400)]

    curves: list[dict[str, Any]] = []
    best = {"valid": float("inf"), "epoch": -1}
    step = 0
    t_resume = time.time()
    for epoch in range(1, epochs_target + 1):
        t_epoch = time.time()
        model.train()
        order = orders[(epoch - 1) % len(orders)]
        main_losses: list[float] = []
        aux_losses: list[float] = []
        batch_graph_losses: list[float] = []
        batch_cut_losses: list[float] = []
        batch_var_losses: list[float] = []
        grad_snapshot: dict[str, float] = {}
        total_grad_snapshot = 0.0
        for start in range(0, n_train, int(recipe["batch_molecules"])):
            batch_rows = data.splits["train"][order[start:start + int(recipe["batch_molecules"])]]
            lr = C.lr_at(step, warmup_steps, total_steps, float(recipe["peak_lr"]),
                         float(recipe["min_lr"]))
            for group in opt.param_groups:
                group["lr"] = lr
            pair_idx, aux_mols = sampler.sample(int(recipe["batch_pairs"]))
            all_rows = np.concatenate([batch_rows, aux_mols[0], aux_mols[1]])
            rows_t = torch.as_tensor(all_rows.astype(np.int64), device=device)
            res = model.molecule_forward(rows_t)
            n_main = len(batch_rows)
            y = torch.as_tensor(data.y_std[all_rows], device=device, dtype=torch.float32)
            f_bar = res["f_bar"]
            # ---- the only changed line: graph-level main loss ----
            loss_main = torch.mean((f_bar[:n_main] - y[:n_main]) ** 2)
            pairs_t = store.tensor_pairs(pair_idx, device)
            delta = model.pair_cut_delta(pairs_t)
            pair_label = torch.as_tensor(store.delta_std_of(pair_idx), device=device,
                                         dtype=torch.float32)
            loss_aux = torch.mean((delta - pair_label) ** 2) / float(store.tau2)
            loss = loss_main + float(recipe["aux_weight"]) * loss_aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if start == 0:
                total_grad_snapshot = total_grad_norm(model)
                grad_snapshot = group_grad_norms(model, family)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["clip_norm"]))
            opt.step()
            step += 1
            with torch.no_grad():
                f_cut_b = res["f_cut"]
                owner_b = res["owner"]
                cnt_b = res["counts"].float().clamp(min=1.0)
                per_b = torch.zeros(len(all_rows), device=device).index_add_(
                    0, owner_b, (f_cut_b - y[owner_b]) ** 2) / cnt_b
                var_b = torch.zeros(len(all_rows), device=device).index_add_(
                    0, owner_b, (f_cut_b - f_bar[owner_b]) ** 2) / cnt_b
                batch_graph_losses.append(float(loss_main.detach()))
                batch_cut_losses.append(float(per_b[:n_main].mean()))
                batch_var_losses.append(float(var_b[:n_main].mean()))
            main_losses.append(float(loss_main.detach()))
            aux_losses.append(float(loss_aux.detach()))
            if int(getattr(args, "max_steps", 0)) and step >= int(args.max_steps):
                break
        if int(getattr(args, "max_steps", 0)) and step >= int(args.max_steps):
            log(f"  smoke limit reached: {step} steps")

        ev = evaluate_epoch(model, family, data, cfg)
        stats = ev["stats"]
        preds = ev["preds"]
        metrics = {s: raw_metrics(preds[s], data.labels[data.splits[s]], data.scaler)
                   for s in ("train", "valid")}
        row: dict[str, Any] = {
            "run_id": run_id, "epoch": epoch,
            "lr": C.lr_at(step - 1, warmup_steps, total_steps, float(recipe["peak_lr"]),
                          float(recipe["min_lr"])),
            "train_loss": float(np.mean(main_losses)),
            "train_graph_mse_batch": float(np.mean(batch_graph_losses)),
            "train_cut_mse_batch": float(np.mean(batch_cut_losses)),
            "train_cut_var_batch": float(np.mean(batch_var_losses)),
            "aux_loss": float(np.mean(aux_losses)),
            "grad_norm_total": float(total_grad_snapshot),
            "epoch_seconds": time.time() - t_epoch,
        }
        for gname, gval in grad_snapshot.items():
            row[f"grad_{gname}"] = gval
        for s in ("train", "valid"):
            row[f"{s}_rmse"] = metrics[s]["rmse"]
            row[f"{s}_mae"] = metrics[s]["mae"]
            st = stats[s]
            for key in ("cut_mse_std", "graph_mse_std", "cut_var_std", "cut_mse",
                        "graph_mse", "cut_var", "cancel_share", "cut_span_std",
                        "cut_span", "mixed_sign_fraction", "n_parses_mean",
                        "var_share_of_lcut", "graph_share_of_lcut",
                        "identity_max_abs", "corr_var_vs_abs_graph_err"):
                row[f"{s}_{key}"] = st.get(key, float("nan"))
            row[f"{s}_cut_n"] = st.get("n_cuts")
            row[f"{s}_n_molecules"] = st.get("n_molecules")
        curves.append(row)
        C.write_csv(rdir / "curves.csv", curves)
        if row["valid_rmse"] < best["valid"] - 1e-12:
            best = {"valid": row["valid_rmse"], "epoch": epoch}
            torch.save({"state": model.state_dict(), "epoch": epoch, "row": row,
                        "scaler": data.scaler, "ledger": ledger, "hashes": hashes,
                        "arm": arm, "family": family, "loss": "graph_main"},
                       rdir / "state_best.pt")
            save_predictions(rdir / "predictions_best.npz", data, preds)
        log(f"epoch {epoch:3d}/{epochs_target} lr={row['lr']:.2e} "
            f"train={row['train_rmse']:.4f} valid={row['valid_rmse']:.4f} "
            f"g_mse={row['train_graph_mse_std']:.4f} c_mse={row['train_cut_mse_std']:.4f} "
            f"var={row['train_cut_var_std']:.4f} aux={row['aux_loss']:.4f} "
            f"{row['epoch_seconds']:.1f}s")

    # final state + predictions
    torch.save({"state": model.state_dict(), "epoch": int(curves[-1]["epoch"]),
                "row": curves[-1], "scaler": data.scaler, "ledger": ledger,
                "hashes": hashes, "arm": arm, "family": family,
                "loss": "graph_main"}, rdir / "state_final.pt")
    ev_final = evaluate_epoch(model, family, data, cfg)
    save_predictions(rdir / "predictions_final.npz", data, ev_final["preds"])
    wall = time.time() - t_resume

    # mechanism diagnostics on the final (not selected) model, for evidence only
    diag = interaction_diagnostics(model, family, data.pack, data.splits["valid"], device)
    C.write_json(rdir / "mechanism_diagnostics.json", diag)

    manifest = {
        "run_id": run_id, "family": family, "arm": arm, "dataset": args.dataset,
        "epochs": int(curves[-1]["epoch"]), "epochs_planned": epochs_target,
        "steps_per_epoch": steps_per_epoch,
        "batch_molecules": int(recipe["batch_molecules"]),
        "batch_pairs": int(recipe["batch_pairs"]),
        "supervision": "CUTRESP", "aux_weight": float(recipe["aux_weight"]),
        "main_loss": "graph_level_mean_squared_error",
        "main_loss_expression": "mean((f_bar[:n_main] - y[:n_main])**2)",
        "seed": int(args.seed), "signature": C.check_signature(),
        "cache_sha256": data.pack["cache_sha256"],
        "cache_content_sha256": data.pack.get("content_sha256"),
        "cache_path_used": data.pack.get("cache_path_used"),
        "raw_cache_sha256": data.pack.get("cache_sha256"),
        "scaler": data.scaler, "n_train": n_train,
        "n_valid": int(data.splits["valid"].size),
        "params": ledger, "init_hashes": hashes, "zeroed_categories": zeroed,
        "best": best, "final": curves[-1],
        "wall_seconds": wall, "gpu_hours": wall / 3600.0,
        "max_vram_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "environment": C.environment(),
        "pair_store": store.summary(),
        "overrides": json.loads(args.overrides) if args.overrides else None,
        "argv": list(getattr(args, "argv", [])),
        "train_subset": int(args.train_subset), "valid_subset": int(args.valid_subset),
        "max_steps": int(getattr(args, "max_steps", 0)),
        "test_policy": "no test labels read in this round",
        "diagnostics": diag,
    }
    C.write_json(rdir / "manifest.json", manifest)
    C.write_json(rdir / "config.json", {"cfg": cfg, "family": family, "arm": arm})
    log(f"done wall={wall / 60:.2f}min best_valid={best['valid']:.6f}@{best['epoch']}")
    log_handle.close()
    return 0


# import late to keep the module import graph identical to the parent rounds
def PM_module_hashes(model, prefixes):
    import trc_model as PM

    return PM.module_hashes(model, prefixes)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--family", required=True, choices=["parent", "csoi"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--dataset", default="lipo")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--overrides", default="")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="smoke only: stop after N optimisation steps")
    ap.add_argument("--train-subset", type=int, default=0,
                    help="smoke only: first N official Train molecules")
    ap.add_argument("--valid-subset", type=int, default=0,
                    help="smoke only: first N official Valid molecules")
    args = ap.parse_args(argv)
    args.argv = list(argv) if argv is not None else []
    arm_choices = {"parent": ["FACTOR", "ADD", "GLOBAL", "DIRECT"],
                   "csoi": list(C.CM.CSOI_ARMS)}
    if args.arm not in arm_choices[args.family]:
        raise SystemExit(f"arm {args.arm} is not valid for family {args.family}")
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())