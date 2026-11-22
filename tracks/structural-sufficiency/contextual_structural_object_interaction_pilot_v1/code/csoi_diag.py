"""Interaction-mechanism diagnostics for CSOI-v1 (post-training, Valid only).

For every trained arm this records

* attention entropy and effective token count (per head, per query token),
* relation-bias statistics (bucket-pair / total-distance / cut-bond paths),
* gradient-norm trajectories per module group (from ``curves.csv``),
* the interaction contribution ``f(S|C) - f_additive(S|C)`` and how it relates to
  the pair-delta error,
* a **remove-bias-under-frozen-weights** probe: CSOI_REL with its relation branch
  zeroed is evaluated without retraining (this is `CSOI_GLOBAL` up to the trained
  relation parameters, not a new arm).

Only Train/Valid predictions are used; official Test labels are never read.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import csoi_common as C
import csoi_data as D
import csoi_model as M
import csoi_train as T

RUNS = {"B_CSOI_ADD": "CSOI_ADD_LIPO_R0", "C_CSOI_GLOBAL": "CSOI_GLOBAL_LIPO_R0",
        "D_CSOI_REL": "CSOI_REL_LIPO_R0"}


def _load(arm: str, run_id: str, cfg: dict, pack: dict, device: str):
    import torch

    rdir = C.run_dir(run_id)
    ckpt_path = rdir / "state_best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    model = M.CSOIModel(cfg, *C.official_feature_dims(), arm)
    model.load_pack(pack)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state"])
    model = model.to(device).float().eval()
    return model, ckpt, rdir


def _attention_stats(model, pack, rows: np.ndarray, device: str, max_mol: int = 48) -> dict:
    import torch

    rows = rows[:max_mol]
    cap: dict[str, Any] = {}
    model._capture = cap
    try:
        with torch.no_grad():
            model.molecule_cuts(torch.as_tensor(rows, device=device))
    finally:
        model._capture = None
    alpha = torch.cat(cap["alpha"], 0).double()          # [n_entries, heads]
    qs = torch.cat(cap["qs"], 0)
    if alpha.numel() == 0:
        return {"note": "no attention entries captured"}
    # per query token (across its keys) and per head
    # entropy per (query token, head)
    nq = int(qs.max().item()) + 1
    per_q = torch.zeros(nq, alpha.shape[1], dtype=alpha.dtype)
    per_q.index_add_(0, qs, -(alpha.clamp_min(1e-12).log() * alpha))
    n_keys = torch.zeros(nq, dtype=alpha.dtype).index_add_(
        0, qs, torch.ones_like(qs, dtype=alpha.dtype))
    ent_q = per_q / n_keys[:, None].clamp_min(1.0)
    eff = ent_q.exp()
    out = {
        "n_query_tokens": int(nq),
        "n_entries": int(alpha.shape[0]),
        "heads": int(alpha.shape[1]),
        "alpha_max": float(alpha.max()),
        "alpha_mean": float(alpha.mean()),
        "attention_entropy_nats": {f"head_{h}": C.quantiles(ent_q[:, h].numpy())
                                   for h in range(alpha.shape[1])},
        "effective_tokens": {f"head_{h}": C.quantiles(eff[:, h].numpy())
                             for h in range(alpha.shape[1])},
        "keys_per_query_tokens": C.quantiles(n_keys.numpy()),
        "effective_token_ratio": float((eff.mean(1) / n_keys).mean()),
    }
    out["attention_entropy_nats"]["all_heads"] = C.quantiles(ent_q.mean(1).numpy())
    out["effective_tokens"]["all_heads"] = C.quantiles(eff.mean(1).numpy())
    return out


def _bias_stats(model, pack, rows: np.ndarray, device: str, max_mol: int = 48) -> dict:
    import torch

    if not getattr(model, "use_relation", False):
        return {"note": "arm has no relation bias"}
    frames = model.frame_embeddings()
    view = D.TokenView(pack)
    with torch.no_grad():
        values = []
        for r in rows[:max_mol].tolist():
            idx = int(np.asarray(pack["mol_parse_ptr"])[r])
            end = int(np.asarray(pack["mol_parse_ptr"])[r + 1])
            for p in range(idx, end):
                s = int(np.asarray(pack["parse_S"])[p])
                cc = int(np.asarray(pack["parse_C"])[p])
                bond = int(np.asarray(pack["parse_cut_bond_row"])[p])
                st, ct, bt = (torch.as_tensor([s], device=device),
                              torch.as_tensor([cc], device=device),
                              torch.as_tensor([bond], device=device))
                xs, _, _, sb = model.token_embeddings(st, frames)
                if cc < 0:
                    continue
                xc, _, _, cb = model.token_embeddings(ct, frames)
                z = (model.rel_ds(sb)[:, None, :] + model.rel_dc(cb)[None, :, :]
                     + model.rel_dt(model.dtot[sb[:, None] * C.N_BUCKETS + cb[None, :]])
                     + model.rel_cut(model._cut_tokens(bt, model.cut_embeddings()))[:, None, :])
                b = model.rel_mlp(z)
                values.append(b.reshape(-1).detach())
    if not values:
        return {"note": "no real-cut parses sampled"}
    v = torch.cat(values).double().numpy()
    return {"n_values": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
            "abs_mean": float(np.abs(v).mean()), "min": float(v.min()), "max": float(v.max()),
            "quantiles": C.quantiles(v)}


def _interaction_contribution(model, pack, rows: np.ndarray, device: str) -> dict:
    import torch

    frames = model.frame_embeddings()
    out = {"per_cut": {}, "per_molecule": {}}
    with torch.no_grad():
        full, add = [], []
        for lo in range(0, rows.size, 32):
            chunk = torch.as_tensor(rows[lo:lo + 32], device=device)
            res = model.molecule_cuts(chunk, frames)
            full.append(res["f_bar"].double().cpu())
            pidx, counts = model._parse_index(chunk)
            f_add = model._additive_only(model.parse_S[pidx], model.parse_C[pidx],
                                         model.parse_cut_bond_row[pidx], frames)
            owner = torch.repeat_interleave(torch.arange(chunk.numel(), device=device), counts)
            bar = f_add.new_zeros(chunk.numel()).index_add_(0, owner, f_add)
            add.append((bar / counts.to(f_add.dtype).clamp(min=1.0)).double().cpu())
        f = torch.cat(full).numpy()
        a = torch.cat(add).numpy()
    d = f - a
    out["per_molecule"] = {"mean_abs": float(np.mean(np.abs(d))), "std": float(np.std(d)),
                           "mean": float(np.mean(d)), "quantiles": C.quantiles(d),
                           "corr_with_prediction": C.pearson(f, a),
                           "variance_reduction": float(1.0 - np.var(d) / max(1e-12, np.var(f)))}
    if model.interaction:
        pidx, _counts = model._parse_index(torch.as_tensor(rows, device=device))
        inv = model.interaction_contribution(
            model.parse_S[pidx], model.parse_C[pidx],
            model.parse_cut_bond_row[pidx], frames)
        out["per_cut"] = {"mean_abs": float(inv.abs().mean()), "std": float(inv.std())}
    return out


def _remove_bias(model, pack, cfg, device: str, rows: np.ndarray,
                 scaler: Mapping[str, float]) -> dict:
    """Zero the relation branch output under frozen weights (no retraining)."""
    import torch

    if not getattr(model, "use_relation", False):
        return {"note": "arm has no relation bias"}
    before = _predict(model, pack, device, rows)
    saved = {n: p.detach().clone() for n, p in model.named_parameters()
             if n.startswith("rel_")}
    with torch.no_grad():
        model.rel_mlp[-1].weight.zero_()
        model.rel_mlp[-1].bias.zero_()
    after = _predict(model, pack, device, rows)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in saved:
                p.copy_(saved[n])
    diff = after - before
    scale, shift = float(scaler["std"]), float(scaler["mean"])
    y = np.asarray(pack["labels"])[rows]
    return {"rmse_with_bias": C.rmse(before * scale + shift, y),
            "rmse_without_bias": C.rmse(after * scale + shift, y),
            "max_abs_change": float(np.max(np.abs(diff))),
            "mean_abs_change": float(np.mean(np.abs(diff))),
            "note": "same weights, relation-bias output zeroed: isolates the bias path"}


def _predict(model, pack, device: str, rows: np.ndarray) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for lo in range(0, rows.size, 64):
            chunk = torch.as_tensor(rows[lo:lo + 64], device=device)
            out.append(model.molecule_cuts(chunk)["f_bar"].double().cpu().numpy())
    return np.concatenate(out)


def main(argv: Sequence[str] | None = None) -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lipo")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    cfg = C.load_config()
    pack = D.load_pack(cfg, args.dataset)
    valid = np.asarray(pack["valid_idx"], np.int64)
    labels = np.asarray(pack["labels"], np.float64)
    device = args.device
    out: dict[str, Any] = {
        "protocol": cfg["protocol_id"], "cache_sha256": pack["cache_sha256"],
        "device": device, "note": "Valid molecules only; Test labels never read",
        "token_guard": (C.read_json(C.analysis_dir() / "token_stats.json")["guard"]
                        if (C.analysis_dir() / "token_stats.json").is_file() else None),
        "arms": {}}
    for name, run_id in RUNS.items():
        try:
            model, ckpt, rdir = _load(name.split("_", 1)[1], run_id, cfg, pack, device)
        except FileNotFoundError as exc:
            print(f"[skip] {name}: {exc}")
            continue
        scaler = ckpt["scaler"]
        rows_small = valid[:48]
        arm: dict[str, Any] = {
            "run_id": run_id, "best_epoch": ckpt.get("epoch"),
            "parameters": model.parameter_ledger()["total"],
        }
        if model.interaction:
            arm["attention"] = _attention_stats(model, pack, rows_small, device)
        arm["relation_bias"] = _bias_stats(model, pack, rows_small, device)
        arm["interaction_contribution"] = _interaction_contribution(model, pack, rows_small,
                                                                   device)
        curves = C.read_csv(rdir / "curves.csv") if (rdir / "curves.csv").is_file() else []
        grad_cols = [c for c in (curves[0].keys() if curves else []) if c.startswith("grad_")]
        if grad_cols:
            arm["gradient_norms"] = {
                c: {"first": float(curves[0][c]), "last": float(curves[-1][c]),
                    "min": float(min(float(r[c]) for r in curves)),
                    "max": float(max(float(r[c]) for r in curves))}
                for c in grad_cols}
        if model.use_relation:
            arm["bias_removal_frozen_weights"] = _remove_bias(model, pack, cfg, device,
                                                             valid, scaler)
        # Valid interaction magnitude in target units
        f_std = _predict(model, pack, device, valid)
        arm["valid_pred_std_mean"] = float(np.mean(f_std))
        arm["valid_rmse_std_units"] = C.rmse(f_std, (labels[valid] - scaler["mean"]) / scaler["std"])
        out["arms"][name] = arm
        print(f"  {name}: {json_keys(arm)}")
    C.write_json(C.results_root() / "interaction_diagnostics.json", out)
    print(f"-> {C.results_root() / 'interaction_diagnostics.json'}")
    return 0


def json_keys(d: Mapping[str, Any]) -> str:
    return ",".join(d.keys())


if __name__ == "__main__":
    raise SystemExit(main())
