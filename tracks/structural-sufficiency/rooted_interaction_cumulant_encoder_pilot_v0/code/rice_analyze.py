"""Post-training analysis for RICE-v0.

Everything here is evaluated on a **frozen** best-Dev checkpoint.  No retraining,
no capacity-matched claims, and no Dev-based word-type selection.

Sections
  1  main table and paired Dev bootstrap       -> analysis/{arms.csv,summary.json}
  2  per-order kappa statistics (Dev)          -> analysis/kappa_stats.json
  3  frozen-checkpoint interventions (Dev)     -> analysis/interventions.json
  4  root-conditioned word types (Fit screen)  -> analysis/word_types.{json,csv}
  5  verdict                                   -> analysis/verdict.json

The word-type ablation is exact and cheap because ``u_i`` depends only on the
atom fields and ``J_e`` only on the bond fields, so every instance of one
root-marked word type carries the **same** ``kappa`` vector.  Zeroing a type
therefore only changes the pooled statistics of the roots that contain it.
The recomputed prediction path is validated against a full model forward pass
before any ablation is reported.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import rice_common as C
import rice_data as D
import rice_model as M
import rice_structure as S
import rice_tensor as T

SIZES = (2, 3, 4)
RNG_SEED = 20261124


# ---------------------------------------------------------------------------
# cached evaluation of one frozen checkpoint on one row set
# ---------------------------------------------------------------------------

@dataclass
class SplitEval:
    rows: np.ndarray
    atoms_per_size: dict[int, np.ndarray]
    support_counts: dict[int, np.ndarray]
    type_ids: dict[int, np.ndarray]
    signatures: dict[int, np.ndarray]
    values: dict[int, torch.Tensor]
    stats: dict[int, dict[str, torch.Tensor]]
    sentence: torch.Tensor
    h: torch.Tensor
    mol_of_root: np.ndarray
    mol_atom_count: np.ndarray
    mol_h_sum: torch.Tensor
    mol_h_sqsum: torch.Tensor
    graph: torch.Tensor
    yhat_std: np.ndarray
    n_atoms: int
    n_molecules: int


def _segment(values: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    out = values.new_zeros((int(n), int(values.shape[1])))
    out.index_add_(0, seg, values)
    return out


def _signatures(structs: Mapping[int, S.MoleculeStruct], rows: Sequence[int], size: int,
                atom_dims: Sequence[int], bond_dims: Sequence[int]) -> np.ndarray:
    parts = [S.canonical_signature(structs[int(r)], size, atom_dims, bond_dims) for r in rows]
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 10), np.int64)


def _void(sig: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(sig).view(np.dtype((np.void, sig.shape[1] * 8))).ravel()


def evaluate_split(model: M.RiceModel, structs: Mapping[int, S.MoleculeStruct],
                   rows: Sequence[int], bundle: D.DataBundle, arm: str,
                   atom_dims: Sequence[int], bond_dims: Sequence[int],
                   map_types: Mapping[int, np.ndarray] | None = None,
                   device: Any = "cpu", batch: int = 512) -> SplitEval:
    rows = np.asarray(rows, np.int64)
    model.eval()
    stats: dict[int, dict[str, list[torch.Tensor]]] = {
        order: {"sum": [], "sumsq": [], "count": []} for order in (1, 2, 3, 4)}
    values: dict[int, list[torch.Tensor]] = {order: [] for order in (1, 2, 3, 4)}
    roots_g: dict[int, list[np.ndarray]] = {size: [] for size in SIZES}
    sigs: dict[int, list[np.ndarray]] = {size: [] for size in SIZES}
    counts: dict[int, list[np.ndarray]] = {size: [] for size in SIZES}
    sentence_parts: list[torch.Tensor] = []
    h_parts: list[torch.Tensor] = []
    mol_of_root: list[np.ndarray] = []
    mol_count: list[np.ndarray] = []
    mol_sum: list[torch.Tensor] = []
    mol_sqsum: list[torch.Tensor] = []
    graph_parts: list[torch.Tensor] = []
    preds: list[np.ndarray] = []
    n_roots_total = 0
    n_mol_total = 0
    for start in range(0, rows.size, int(batch)):
        part = rows[start:start + int(batch)]
        sub = {int(r): structs[int(r)] for r in part}
        coll = T.collate(part.tolist(), sub, bundle.raw, device)
        with torch.no_grad():
            out = model(coll, arm=arm, return_aux=True)
        for size in SIZES:
            roots_g[size].append(coll[f"atoms_{size}"][:, 0].cpu().numpy() + n_roots_total)
            sigs[size].append(_signatures(structs, part, size, atom_dims, bond_dims))
            counts[size].append(np.asarray(
                [int(structs[int(r)].atoms[size].shape[0]) for r in part], np.int64))
        for order in (1, 2, 3, 4):
            w = out["words"][order]
            stats[order]["sum"].append(w["sum"].cpu())
            stats[order]["sumsq"].append(w["sumsq"].cpu())
            stats[order]["count"].append(w["count"].cpu())
            values[order].append(w["values"].cpu())
        sentence_parts.append(out["sentence"].cpu())
        h_parts.append(out["h"].cpu())
        ptr = coll["mol_atom_ptr"].cpu().numpy()
        mol_of_root.append(np.repeat(np.arange(n_mol_total, n_mol_total + part.size),
                                     np.diff(ptr)))
        mol_count.append(np.diff(ptr))
        mol_sum.append(_segment(out["h"].cpu(), coll["atom_mol"].cpu(), part.size))
        mol_sqsum.append(_segment(out["h"].cpu() ** 2, coll["atom_mol"].cpu(), part.size))
        graph_parts.append(out["graph"].cpu())
        preds.append(out["yhat_std"].cpu().numpy())
        n_roots_total += int(coll["n_atoms"])
        n_mol_total += int(part.size)
    stats_out = {order: {k: torch.cat(v, dim=0) for k, v in d.items()}
                 for order, d in stats.items()}
    signatures = {size: (np.concatenate(sigs[size]) if sigs[size]
                         else np.zeros((0, 10), np.int64)) for size in SIZES}
    if map_types is None:
        type_ids = {size: np.full(signatures[size].shape[0], -1, np.int64) for size in SIZES}
    else:
        type_ids = {size: _map_types(signatures[size], map_types[size]) for size in SIZES}
    return SplitEval(
        rows=rows,
        atoms_per_size={size: (np.concatenate(roots_g[size]) if roots_g[size]
                               else np.zeros(0, np.int64)) for size in SIZES},
        support_counts={size: np.concatenate(counts[size]) for size in SIZES},
        type_ids=type_ids, signatures=signatures,
        values={order: torch.cat(values[order], dim=0) for order in (1, 2, 3, 4)},
        stats=stats_out, sentence=torch.cat(sentence_parts, dim=0),
        h=torch.cat(h_parts, dim=0), mol_of_root=np.concatenate(mol_of_root),
        mol_atom_count=np.concatenate(mol_count),
        mol_h_sum=torch.cat(mol_sum, dim=0), mol_h_sqsum=torch.cat(mol_sqsum, dim=0),
        graph=torch.cat(graph_parts, dim=0), yhat_std=np.concatenate(preds),
        n_atoms=int(n_roots_total), n_molecules=int(rows.size))


# ---------------------------------------------------------------------------
# word-type vocabulary (Fit only)
# ---------------------------------------------------------------------------

def build_type_vocabulary(fit: SplitEval) -> dict[int, np.ndarray]:
    vocab = {}
    for size in SIZES:
        sig = fit.signatures[size]
        if sig.shape[0] == 0:
            vocab[size] = np.zeros((0, 10), np.int64)
            continue
        uniq, _inv = np.unique(_void(sig), return_inverse=True)
        vocab[size] = np.asarray(uniq).view(np.int64).reshape(-1, sig.shape[1])
    return vocab


def _map_types(sig: np.ndarray, vocab: np.ndarray) -> np.ndarray:
    if sig.shape[0] == 0:
        return np.zeros(0, np.int64)
    if vocab.shape[0] == 0:
        return np.full(sig.shape[0], -1, np.int64)
    order = np.argsort(_void(vocab))
    keys = _void(vocab)[order]
    pos = np.clip(np.searchsorted(keys, _void(sig)), 0, keys.size - 1)
    hit = keys[pos] == _void(sig)
    out = np.full(sig.shape[0], -1, np.int64)
    out[hit] = order[pos[hit]]
    return out


# ---------------------------------------------------------------------------
# exact frozen-checkpoint intervention
# ---------------------------------------------------------------------------

def _block(model: M.RiceModel, total: torch.Tensor, sq: torch.Tensor,
           count: torch.Tensor, order: int) -> torch.Tensor:
    denom = count.clamp(min=1.0)
    mean = total / denom[:, None]
    var = (sq / denom[:, None] - mean * mean).clamp(min=0.0)
    std = torch.sqrt(var + 1e-12)
    std = torch.where((count > 1)[:, None], std, torch.zeros_like(std))
    normed = model.order_norm[order - 1](torch.cat([total, mean, std], dim=1))
    return torch.cat([normed, torch.log1p(count).unsqueeze(1)], dim=1)


@dataclass
class Intervention:
    predictions: np.ndarray
    affected_molecules: np.ndarray
    removed_supports: int


def intervene(model: M.RiceModel, split: SplitEval, keep: Mapping[int, np.ndarray],
              *, zero_counts: bool = False) -> Intervention:
    """Zero the kappa contribution of every support with ``keep[size] == False``."""
    removed = 0
    changed_roots: list[torch.Tensor] = []
    per_size: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for size in SIZES:
        mask = np.asarray(keep[size], bool)
        if mask.size == 0 or bool(mask.all()):
            continue
        drop = np.nonzero(~mask)[0]
        removed += int(drop.size)
        roots = torch.as_tensor(split.atoms_per_size[size][drop], dtype=torch.long)
        vals = split.values[size][torch.as_tensor(drop, dtype=torch.long)]
        n_atoms = split.n_atoms
        d_sum = vals.new_zeros((n_atoms, vals.shape[1])).index_add(0, roots, vals)
        d_sq = vals.new_zeros((n_atoms, vals.shape[1])).index_add(0, roots, vals * vals)
        d_count = vals.new_zeros(n_atoms).index_add(
            0, roots, torch.ones(drop.size, dtype=vals.dtype))
        per_size[size] = (d_sum, d_sq, d_count)
        changed_roots.append(roots)
    if not changed_roots:
        return Intervention(predictions=split.yhat_std.copy(),
                            affected_molecules=np.zeros(0, np.int64), removed_supports=0)

    idx = torch.unique(torch.cat(changed_roots))
    blocks = []
    for order in (1, 2, 3, 4):
        total = split.stats[order]["sum"]
        sq = split.stats[order]["sumsq"]
        count = split.stats[order]["count"]
        if order in per_size:
            d_sum, d_sq, d_count = per_size[order]
            total = total - d_sum
            sq = sq - d_sq
            if zero_counts:
                count = count - d_count
        blocks.append(_block(model, total[idx], sq[idx], count[idx], order))
    block = torch.cat(blocks, dim=1)
    with torch.no_grad():
        new_h = model.sentence_mlp(block)
    old_h = split.h[idx]
    mol_idx = torch.as_tensor(split.mol_of_root[idx.numpy()], dtype=torch.long)
    h_sum = split.mol_h_sum.clone().index_add(0, mol_idx, new_h - old_h)
    h_sqsum = split.mol_h_sqsum.clone().index_add(0, mol_idx, new_h * new_h - old_h * old_h)
    affected = np.unique(mol_idx.numpy())
    sel = torch.as_tensor(affected, dtype=torch.long)
    count = torch.as_tensor(split.mol_atom_count, dtype=torch.float32)[sel]
    mean = h_sum[sel] / count[:, None]
    std = torch.sqrt((h_sqsum[sel] / count[:, None] - mean * mean).clamp(min=0.0) + 1e-12)
    readout_in = torch.cat([h_sum[sel], mean, std, torch.log1p(count).unsqueeze(1),
                            split.graph[sel]], dim=1)
    preds = split.yhat_std.copy()
    with torch.no_grad():
        preds[affected] = (model.readout_mlp(readout_in).view(-1) + model.beta).numpy()
    return Intervention(predictions=preds, affected_molecules=affected,
                        removed_supports=removed)


def cross_check_intervention(model: M.RiceModel, structs: Mapping[int, S.MoleculeStruct],
                             rows: Sequence[int], bundle: D.DataBundle,
                             size: int, type_id: int, vocab: Mapping[int, np.ndarray],
                             scale: float, shift: float) -> dict[str, Any]:
    """Compare ``intervene`` with a direct masked model forward on the same rows."""
    rows = np.asarray(rows, np.int64)
    sub = {int(r): structs[int(r)] for r in rows}
    coll = T.collate(rows.tolist(), sub, bundle.raw, "cpu")
    sig = _signatures(structs, rows, size, bundle.atom_dims, bundle.bond_dims)
    tids = _map_types(sig, vocab[size])
    mask = torch.as_tensor((tids != int(type_id)), dtype=torch.float32)
    with torch.no_grad():
        direct = model(coll, arm="RICE_CUMULANT", word_mask={size: mask})["yhat_std"].numpy()
    split = evaluate_split(model, structs, rows, bundle, "RICE_CUMULANT",
                          bundle.atom_dims, bundle.bond_dims, map_types=vocab)
    keep = {s: np.ones(len(split.atoms_per_size[s]), bool) for s in SIZES}
    keep[size] = split.type_ids[size] != int(type_id)
    res = intervene(model, split, keep)
    return {"max_abs_diff": float(np.abs(direct - res.predictions).max()),
            "removed_supports": int(res.removed_supports),
            "ok": bool(np.abs(direct - res.predictions).max() < 1e-5)}


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------

def participation_ratio(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    x = np.asarray(matrix, np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    e = s ** 2
    denom = float((e ** 2).sum())
    return float(e.sum() ** 2 / denom) if denom > 0 else 0.0


def bootstrap_delta(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray,
                    n_boot: int = 10000, seed: int = RNG_SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = y.size
    idx = rng.integers(0, n, size=(int(n_boot), n))
    sq_a = (pred_a[idx] - y[idx]) ** 2
    sq_b = (pred_b[idx] - y[idx]) ** 2
    delta = np.sqrt(sq_a.mean(axis=1)) - np.sqrt(sq_b.mean(axis=1))
    point = float(math.sqrt(np.mean((pred_a - y) ** 2))
                  - math.sqrt(np.mean((pred_b - y) ** 2)))
    return {"point": point,
            "ci_low": float(np.percentile(delta, 2.5)),
            "ci_high": float(np.percentile(delta, 97.5)),
            "fraction_positive": float((delta > 0).mean()),
            "n_boot": int(n_boot), "seed": int(seed),
            "sign": "RMSE(a) - RMSE(b); positive = b better"}


def _rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(math.sqrt(np.mean((np.asarray(pred, np.float64) - y) ** 2)))


def _load_run(arm: str, results_dir: Any) -> dict[str, Any]:
    run_dir = results_dir / f"{arm}_S0"
    manifest = C.read_json(run_dir / "manifest.json")
    dev = np.load(run_dir / "predictions_best_dev.npz")
    fit = np.load(run_dir / "predictions_best_fit.npz")
    state = torch.load(run_dir / "state_best.pt", map_location="cpu", weights_only=False)
    return {"arm": arm, "dir": run_dir, "manifest": manifest, "dev": dev, "fit": fit,
            "state": state}


# ---------------------------------------------------------------------------
# word-type screen
# ---------------------------------------------------------------------------

def analyse_word_types(model: M.RiceModel, fit_eval: SplitEval, dev_eval: SplitEval,
                       bundle: D.DataBundle, vocab: Mapping[int, np.ndarray],
                       y_fit: np.ndarray, y_dev: np.ndarray, cfg: Mapping[str, Any],
                       top_n: int) -> dict[str, Any]:
    min_mol = int(cfg["mechanism_analysis"]["word_type_min_fit_molecules"])
    min_scaf = int(cfg["mechanism_analysis"]["word_type_min_fit_scaffolds"])
    scaffold = {int(k): v for k, v in bundle.scaffold_by_row.items()}
    scale = float(bundle.scaler["std"])
    shift = float(bundle.scaler["mean"])
    base_fit = fit_eval.yhat_std * scale + shift
    base_dev = dev_eval.yhat_std * scale + shift
    base_fit_mse = float(np.mean((base_fit - y_fit) ** 2))
    base_dev_rmse = _rmse(base_dev, y_dev)

    mol_of_support: dict[int, np.ndarray] = {}
    for size in SIZES:
        mol_of_support[size] = np.repeat(np.arange(fit_eval.n_molecules),
                                         fit_eval.support_counts[size])

    kept: list[dict[str, Any]] = []
    n_types = {size: int(vocab[size].shape[0]) for size in SIZES}
    for size in SIZES:
        ids = fit_eval.type_ids[size]
        if ids.size == 0:
            continue
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        bounds = np.searchsorted(sorted_ids, np.arange(n_types[size] + 1))
        for tid in range(n_types[size]):
            lo, hi = int(bounds[tid]), int(bounds[tid + 1])
            if hi <= lo:
                continue
            sample = order[lo:hi]
            mols = np.unique(mol_of_support[size][sample])
            n_scaf = len({scaffold[int(fit_eval.rows[m])] for m in mols})
            if mols.size >= min_mol and n_scaf >= min_scaf:
                kept.append({"size": int(size), "type_id": int(tid),
                             "fit_molecules": int(mols.size),
                             "fit_scaffolds": int(n_scaf),
                             "fit_supports": int(hi - lo),
                             "fit_supports_per_molecule":
                                 float((hi - lo) / max(1, int(mols.size)))})

    ablated: list[dict[str, Any]] = []
    for c in kept:
        size = c["size"]
        keep = {s: np.ones(len(fit_eval.atoms_per_size[s]), bool) for s in SIZES}
        keep[size] = fit_eval.type_ids[size] != c["type_id"]
        res = intervene(model, fit_eval, keep)
        pred = res.predictions * scale + shift
        mse = float(np.mean((pred - y_fit) ** 2))
        delta_pred = pred - base_fit
        affected = res.affected_molecules
        c.update({"fit_delta_mse": mse - base_fit_mse,
                  "fit_delta_rmse": math.sqrt(mse) - math.sqrt(base_fit_mse),
                  "fit_mean_abs_change_affected":
                      float(np.abs(delta_pred[affected]).mean()) if affected.size else 0.0,
                  "fit_mean_abs_change_all": float(np.abs(delta_pred).mean()),
                  "fit_abs_change_max": float(np.abs(delta_pred).max()),
                  "affected_molecules": int(affected.size)})
        ablated.append(c)
    ablated.sort(key=lambda c: -c["fit_delta_mse"])

    top = ablated[:int(top_n)]
    dev_rows: list[dict[str, Any]] = []
    for c in top:
        size = c["size"]
        keep = {s: np.ones(len(dev_eval.atoms_per_size[s]), bool) for s in SIZES}
        keep[size] = dev_eval.type_ids[size] != c["type_id"]
        res = intervene(model, dev_eval, keep)
        pred = res.predictions * scale + shift
        n_dev = int((dev_eval.type_ids[size] == c["type_id"]).sum())
        dev_mols = np.unique(np.repeat(np.arange(dev_eval.n_molecules),
                                       dev_eval.support_counts[size])[
            dev_eval.type_ids[size] == c["type_id"]]) if n_dev else np.zeros(0, np.int64)
        entry = dict(c)
        entry.update({
            "dev_supports": n_dev,
            "dev_molecules": int(dev_mols.size),
            "dev_delta_rmse": _rmse(pred, y_dev) - base_dev_rmse,
            "dev_mean_abs_change": float(np.abs(pred - base_dev).mean()),
            "certificate": np.asarray(vocab[size][c["type_id"]]).tolist(),
            "shape": S.decode_signature(vocab[size][c["type_id"]], size,
                                        bundle.atom_dims, bundle.bond_dims),
        })
        dev_rows.append(entry)

    summary = {
        "n_types_total": {str(size): n_types[size] for size in SIZES},
        "n_types_min_support": int(len(kept)),
        "min_fit_molecules": min_mol, "min_fit_scaffolds": min_scaf,
        "n_ablated": int(len(ablated)),
        "n_positive_fit_contributors": int(sum(1 for c in ablated if c["fit_delta_mse"] > 0)),
        "n_negative_fit_contributors": int(sum(1 for c in ablated if c["fit_delta_mse"] < 0)),
        "n_reusable_high_impact": int(sum(1 for c in top if c["fit_delta_mse"] > 0)),
        "top_share_of_fit_mse": float(sum(c["fit_delta_mse"] for c in top)),
        "fit_mse": base_fit_mse,
        "dev_top": dev_rows,
        "note": ("Word types are screened on Fit only.  The Dev column is descriptive for "
                 "the types already chosen on Fit and is never used for selection."),
    }
    csv_rows = [{"size": c["size"], "type_id": c["type_id"],
                 "fit_molecules": c["fit_molecules"], "fit_scaffolds": c["fit_scaffolds"],
                 "fit_supports": c["fit_supports"],
                 "fit_delta_mse": c["fit_delta_mse"],
                 "fit_delta_rmse": c["fit_delta_rmse"],
                 "fit_mean_abs_change_affected": c["fit_mean_abs_change_affected"],
                 "affected_molecules": c["affected_molecules"]}
                for c in ablated[:200]]
    return {"summary": summary, "rows": csv_rows}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-types", type=int, default=12)
    args = parser.parse_args()
    cfg = C.load_config()
    bundle = D.load_bundle()
    y_dev = bundle.labels[bundle.eval_rows]
    y_fit = bundle.labels[bundle.fit_rows]
    historical = cfg["historical_d0_rmse"]

    # ---- 1. main table + paired bootstrap ---------------------------------
    runs = {arm: _load_run(arm, C.RESULTS_DIR) for arm in M.ARMS}
    arms = []
    for arm, run in runs.items():
        m = run["manifest"]
        arms.append({"arm": arm, "params": m["params"]["total"],
                     "train_rmse": m["train_rmse_at_best"], "dev_rmse": m["dev_rmse"],
                     "dev_mae": m["dev_mae"], "best_epoch": m["best_epoch"],
                     "epochs_run": m["epochs_run"], "train_seconds": m["train_seconds"],
                     "wall_seconds": m["wall_seconds"], "gpu_hours": m["gpu_hours"],
                     "prep_seconds": m["prep_seconds"],
                     "words_per_molecule_fit":
                         m["structure_stats"]["mean_words_per_molecule"]})
    C.write_csv(C.ANALYSIS_DIR / "arms.csv", arms)
    cum, raw = runs["RICE_CUMULANT"], runs["RICE_RAW"]
    if not np.array_equal(cum["dev"]["row_id"], raw["dev"]["row_id"]):
        raise RuntimeError("Dev row order differs between the two arms")
    delta = bootstrap_delta(raw["dev"]["pred"], cum["dev"]["pred"], y_dev)
    cum_dev = float(cum["manifest"]["dev_rmse"])
    summary = {
        "arms": arms,
        "delta_raw_minus_cumulant": delta,
        "historical": {"r2_sentence_s0": historical["R2_SENTENCE_S0"],
                       "r2_sentence_s1": historical["R2_SENTENCE_S1"],
                       "gine": historical["GINE_REF"],
                       "cumulant_minus_r2_s0": cum_dev - historical["R2_SENTENCE_S0"],
                       "cumulant_minus_r2_s1": cum_dev - historical["R2_SENTENCE_S1"],
                       "cumulant_minus_gine": cum_dev - historical["GINE_REF"]},
        "gpu_hours_both_arms": float(cum["manifest"]["gpu_hours"]
                                     + raw["manifest"]["gpu_hours"]),
        "prep_seconds_total": float(cum["manifest"]["prep_seconds"]
                                    + raw["manifest"]["prep_seconds"]),
    }

    # ---- 2/3. mechanism on the frozen CUMULANT checkpoint ------------------
    model = M.load_model_from_state(cum["state"]["state"], bundle.atom_dims,
                                    bundle.bond_dims, D.normalisers(bundle), config=cfg,
                                    device=args.device)
    model.eval()
    structs = S.build_all(bundle, np.union1d(bundle.fit_rows, bundle.eval_rows))
    fit_eval = evaluate_split(model, structs, bundle.fit_rows, bundle, "RICE_CUMULANT",
                              bundle.atom_dims, bundle.bond_dims, device=args.device)
    vocab = build_type_vocabulary(fit_eval)
    fit_eval.type_ids = {size: _map_types(fit_eval.signatures[size], vocab[size])
                         for size in SIZES}
    dev_eval = evaluate_split(model, structs, bundle.eval_rows, bundle, "RICE_CUMULANT",
                              bundle.atom_dims, bundle.bond_dims, map_types=vocab,
                              device=args.device)
    scale, shift = float(bundle.scaler["std"]), float(bundle.scaler["mean"])
    replay_fit = _rmse(fit_eval.yhat_std * scale + shift, y_fit)
    replay_dev = _rmse(dev_eval.yhat_std * scale + shift, y_dev)
    replay = {"fit_rmse": replay_fit, "dev_rmse": replay_dev,
              "manifest_fit_rmse": cum["manifest"]["train_rmse_at_best"],
              "manifest_dev_rmse": cum["manifest"]["dev_rmse"],
              "fit_abs_diff": abs(replay_fit - cum["manifest"]["train_rmse_at_best"]),
              "dev_abs_diff": abs(replay_dev - cum["manifest"]["dev_rmse"]),
              "ok": abs(replay_dev - cum["manifest"]["dev_rmse"]) < 1e-5}

    # intervention path self-test: a no-op ablation must reproduce the predictions
    noop = intervene(model, dev_eval, {s: np.ones(len(dev_eval.atoms_per_size[s]), bool)
                                       for s in SIZES})
    replay["noop_intervention_max_abs_diff"] = float(
        np.abs(noop.predictions - dev_eval.yhat_std).max())
    replay["noop_ok"] = bool(replay["noop_intervention_max_abs_diff"] < 1e-6)
    summary["replay"] = replay
    C.write_json(C.ANALYSIS_DIR / "summary.json", summary)

    atom_rows = np.concatenate([structs[int(r)].atom_feat for r in bundle.fit_rows], axis=0)
    n_fit_atom_types = int(np.unique(_void(atom_rows)).size)
    order_stats: dict[str, Any] = {}
    for order in (1, 2, 3, 4):
        k = dev_eval.values[order].numpy()
        norms = np.linalg.norm(k, axis=1) if k.size else np.zeros(0)
        order_stats[str(order)] = {
            "count": int(k.shape[0]),
            "mean": float(k.mean()) if k.size else 0.0,
            "std": float(k.std()) if k.size else 0.0,
            "abs_mean": float(np.abs(k).mean()) if k.size else 0.0,
            "norm_mean": float(norms.mean()) if norms.size else 0.0,
            "norm_std": float(norms.std()) if norms.size else 0.0,
            "norm_max": float(norms.max()) if norms.size else 0.0,
            "norm_min": float(norms.min()) if norms.size else 0.0,
            "participation_ratio": participation_ratio(k),
            "n_types_seen": int(np.unique(dev_eval.type_ids[order]).size)
            if order in SIZES else n_fit_atom_types,
        }
    C.write_json(C.ANALYSIS_DIR / "kappa_stats.json", order_stats)

    interventions: dict[str, Any] = {}
    base_dev_raw = dev_eval.yhat_std * scale + shift
    for spec in cfg["mechanism_analysis"]["interventions"]:
        sizes = set(int(x) for x in spec["sizes"])
        keep = {size: np.ones(len(dev_eval.atoms_per_size[size]), bool) for size in SIZES}
        for size in sizes:
            keep[size] = np.zeros(len(dev_eval.atoms_per_size[size]), bool)
        res = intervene(model, dev_eval, keep, zero_counts=bool(spec["zero_counts"]))
        pred = res.predictions * scale + shift
        paired = bootstrap_delta(np.asarray(base_dev_raw, np.float64),
                                 np.asarray(pred, np.float64), y_dev)
        interventions[spec["id"]] = {
            "sizes": sorted(sizes), "zero_counts": bool(spec["zero_counts"]),
            "removed_supports": int(res.removed_supports),
            "dev_rmse": _rmse(pred, y_dev), "dev_mae": C.mae(pred, y_dev),
            "delta_rmse_vs_full": float(_rmse(pred, y_dev) - _rmse(base_dev_raw, y_dev)),
            "mean_abs_pred_change": float(np.abs(pred - base_dev_raw).mean()),
            "max_abs_pred_change": float(np.abs(pred - base_dev_raw).max()),
            "paired_rmse_increase": paired,
        }
    C.write_json(C.ANALYSIS_DIR / "interventions.json", interventions)

    # ---- 3b. ablation cross-check against a direct masked forward ----------
    probe = np.asarray(bundle.fit_rows[:512], np.int64)
    probe_size = 2
    probe_ids = _map_types(_signatures(structs, probe, probe_size, bundle.atom_dims,
                                       bundle.bond_dims), vocab[probe_size])
    if probe_ids.size:
        counts = np.bincount(probe_ids[probe_ids >= 0])
        probe_type = int(np.argmax(counts)) if counts.size else -1
    else:
        probe_type = -1
    cross = cross_check_intervention(model, structs, probe, bundle, probe_size,
                                     probe_type, vocab, scale, shift) if probe_type >= 0 else {}
    summary["ablation_cross_check"] = cross
    C.write_json(C.ANALYSIS_DIR / "summary.json", summary)

    # ---- 4. word types -----------------------------------------------------
    words = analyse_word_types(model, fit_eval, dev_eval, bundle, vocab, y_fit, y_dev,
                               cfg, int(args.top_types))
    C.write_json(C.ANALYSIS_DIR / "word_types.json", words["summary"])
    C.write_csv(C.ANALYSIS_DIR / "word_types_top.csv", words["rows"])

    # ---- 5. verdict --------------------------------------------------------
    raw_dev = float(raw["manifest"]["dev_rmse"])
    d_raw = raw_dev - cum_dev
    zero_ge3 = float(interventions["zero_ge3"]["delta_rmse_vs_full"])
    order3_active = bool(order_stats["3"]["count"] > 0
                         and order_stats["3"]["norm_mean"] > 1e-3)
    order4_active = bool(order_stats["4"]["count"] > 0
                         and order_stats["4"]["norm_mean"] > 1e-3)
    reusable = int(words["summary"]["n_reusable_high_impact"])
    promising = (cum_dev <= cfg["verdict"]["promising_max_dev_rmse"]
                 and d_raw >= cfg["verdict"]["promising_min_delta_vs_raw"]
                 and zero_ge3 >= cfg["verdict"]["promising_min_zero_ge3_degradation"]
                 and order3_active and order4_active and reusable > 0)
    mechanism_only = (d_raw > 0.0 and zero_ge3 > 0.0 and order3_active)
    label = ("RICE_PROMISING" if promising
             else "RICE_MECHANISM_ONLY" if mechanism_only
             else "RICE_NO_SUPPORTED_GAIN")
    verdict = {
        "label": label,
        "cumulant_dev_rmse": cum_dev, "raw_dev_rmse": raw_dev,
        "delta_raw_minus_cumulant": d_raw,
        "delta_raw_minus_cumulant_bootstrap": delta,
        "cumulant_beats_raw": bool(d_raw > 0),
        "zero_ge3_degradation": zero_ge3,
        "order3_active": order3_active, "order4_active": order4_active,
        "reusable_words": reusable,
        "delta_vs_r2_s0": cum_dev - historical["R2_SENTENCE_S0"],
        "delta_vs_r2_s1": cum_dev - historical["R2_SENTENCE_S1"],
        "delta_vs_gine": cum_dev - historical["GINE_REF"],
        "replay": replay,
        "criteria": dict(cfg["verdict"]),
    }
    C.write_json(C.ANALYSIS_DIR / "verdict.json", verdict)
    print(f"verdict: {label}")
    print(f"cumulant dev {cum_dev:.4f}  raw dev {raw_dev:.4f}  delta {d_raw:+.4f}")
    print(f"zero>=3 degradation {zero_ge3:+.4f}  replay ok={replay['ok']} "
          f"noop ok={replay['noop_ok']}")


if __name__ == "__main__":
    main()