"""Post-training analysis for WOBR-v0.

One forward diagnostic of the best-Valid checkpoint over all 4,200 official
rows.  No retraining, no checkpoint switching, no Valid/Test-label ranking of
words.  Word types are exact root-fixed attributed-graph canonical
certificates; the top-word ranking uses Train mean absolute contribution only.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import wobr_canonical as CAN
import wobr_common as C
import wobr_data as D
import wobr_model as M
import wobr_structure as S
import wobr_tensor as T

RUN_DIR = C.RESULTS_DIR / "seed0"
MAX_ATOMS_KEY = 128


def _norm(bundle: D.DataBundle) -> dict[str, np.ndarray]:
    return D.normalisers(bundle)


def _load_model(bundle: D.DataBundle, run: Mapping[str, Any], device: Any):
    ckpt = torch.load(RUN_DIR / "best.pt", map_location=device, weights_only=False)
    model, info = M.build_model(bundle.atom_dims, bundle.bond_dims,
                                int(run["seed"]), _norm(bundle),
                                config=ckpt.get("config"))
    model.load_state_dict(ckpt["state"])
    model.to(device)
    model.eval()
    return model, ckpt, info


def _effective_rank(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        return {"participation_ratio": 0.0, "entropy_rank": 0.0,
                "top1_share": 0.0, "mean_dim_std": 0.0, "min_dim_std": 0.0,
                "near_constant_fraction": 1.0, "total_variance": 0.0}
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 0.0, None)
    total = float(ev.sum())
    if total <= 1e-20:
        return {"participation_ratio": 0.0, "entropy_rank": 0.0,
                "top1_share": 1.0, "mean_dim_std": 0.0, "min_dim_std": 0.0,
                "near_constant_fraction": 1.0, "total_variance": total}
    p = ev / total
    pr = float(total * total / np.sum(ev * ev))
    ent = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))
    std = x.std(axis=0)
    return {"participation_ratio": pr, "entropy_rank": ent,
            "top1_share": float(ev.max() / total),
            "mean_dim_std": float(std.mean()), "min_dim_std": float(std.min()),
            "near_constant_fraction": float((std < 1e-4).mean()),
            "total_variance": total}


def _stats(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, np.float64)
    if x.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0,
                "max": 0.0, "min": 0.0, "std": 0.0}
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "p99": float(np.percentile(x, 99)),
            "max": float(x.max()), "min": float(x.min()), "std": float(x.std())}


def _hist(values: np.ndarray, lo: int, hi: int) -> dict[str, int]:
    vals, counts = np.unique(np.asarray(values, np.int64), return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals.tolist(), counts.tolist())}


def _gather(bundle: D.DataBundle, structs: Mapping[int, S.MoleculeStruct],
            model: M.WobrModel, device: Any, batch_size: int = 128) -> dict[str, Any]:
    rows_all = np.concatenate([bundle.train_idx, bundle.valid_idx, bundle.test_idx])
    splits = np.concatenate([np.zeros(bundle.train_idx.size, np.int8),
                             np.ones(bundle.valid_idx.size, np.int8),
                             np.full(bundle.test_idx.size, 2, np.int8)])
    row_to_idx = np.full(4200, -1, np.int64)
    row_to_idx[rows_all] = np.arange(rows_all.size)
    atom_row: list[np.ndarray] = []
    atom_local: list[np.ndarray] = []
    atom_split: list[np.ndarray] = []
    A_parts: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []
    bond_row: list[np.ndarray] = []
    bond_local: list[np.ndarray] = []
    bond_u: list[np.ndarray] = []
    bond_v: list[np.ndarray] = []
    bond_split: list[np.ndarray] = []
    Q_parts: list[np.ndarray] = []
    lam_parts: list[np.ndarray] = []
    word_row: list[np.ndarray] = []
    word_split: list[np.ndarray] = []
    word_root: list[np.ndarray] = []
    word_support: list[np.ndarray] = []
    word_internal: list[np.ndarray] = []
    word_size: list[np.ndarray] = []
    alpha_parts: list[np.ndarray] = []
    c_parts: list[np.ndarray] = []
    c_self_parts: list[np.ndarray] = []
    c_bond_parts: list[np.ndarray] = []
    cert_parts: list[np.ndarray] = []
    mol_row: list[np.ndarray] = []
    mol_split: list[np.ndarray] = []
    yhat_parts: list[np.ndarray] = []
    self_parts: list[np.ndarray] = []
    bond_sum_parts: list[np.ndarray] = []
    atom_count_parts: list[np.ndarray] = []
    bond_count_parts: list[np.ndarray] = []
    beta = c_n = c_m = None

    with torch.no_grad():
        for split_code, (name, rows) in enumerate((("train", bundle.train_idx),
                                                    ("valid", bundle.valid_idx),
                                                    ("test", bundle.test_idx))):
            rows = np.asarray(rows, np.int64)
            for start in range(0, rows.size, batch_size):
                part = rows[start:start + batch_size]
                batch = T.collate(part.tolist(), structs, bundle.raw, device)
                out = model(batch, return_aux=True)
                contrib = model.word_contributions(batch, out)
                rows_np = np.asarray(part, np.int64)
                atom_start = batch["mol_atom_ptr"].cpu().numpy()
                bond_start = batch["mol_bond_ptr"].cpu().numpy()
                atom_mol = batch["atom_mol"].cpu().numpy()
                bond_mol = batch["bond_mol"].cpu().numpy()
                word_mol = batch["word_mol"].cpu().numpy()
                w_root = batch["word_root"].cpu().numpy()
                w_sup = batch["word_support"].cpu().numpy()
                w_int = batch["word_internal"].cpu().numpy()
                root_local = w_root - atom_start[word_mol]
                sup_local = np.where(w_sup >= 0, w_sup - atom_start[word_mol, None], -1)
                int_local = np.where(w_int >= 0, w_int - bond_start[word_mol, None], -1)
                atom_row.append(rows_np[atom_mol])
                atom_local.append(np.arange(atom_mol.size, dtype=np.int64) - atom_start[atom_mol])
                atom_split.append(np.full(atom_mol.size, split_code, np.int8))
                A_parts.append(out["A"].cpu().numpy())
                z_parts.append(out["z"].cpu().numpy())
                bond_row.append(rows_np[bond_mol])
                bond_local.append(np.arange(bond_mol.size, dtype=np.int64) - bond_start[bond_mol])
                bi = batch["bond_index"].cpu().numpy()
                bond_u.append(np.where(bi.size, bi[:, 0] - atom_start[bond_mol], -1))
                bond_v.append(np.where(bi.size, bi[:, 1] - atom_start[bond_mol], -1))
                bond_split.append(np.full(bond_mol.size, split_code, np.int8))
                Q_parts.append(out["Q"].cpu().numpy())
                lam_parts.append(out["lam"].cpu().numpy())
                word_row.append(rows_np[word_mol])
                word_split.append(np.full(word_mol.size, split_code, np.int8))
                word_root.append(root_local)
                word_support.append(sup_local)
                word_internal.append(int_local)
                word_size.append(batch["word_size"].cpu().numpy())
                alpha_parts.append(out["alpha"].cpu().numpy())
                c_parts.append(contrib["c"].cpu().numpy())
                c_self_parts.append((out["alpha"] * contrib["dot_a_w"]).cpu().numpy())
                c_bond_parts.append((out["alpha"] * 0.5 * contrib["qsum"]).cpu().numpy())
                cert_parts.append(np.concatenate([
                    CAN.canonical_certificates(structs[int(r)], bundle.atom_dims,
                                               bundle.bond_dims)
                    for r in part.tolist()], axis=0))
                mol_row.append(rows_np)
                mol_split.append(np.full(rows_np.size, split_code, np.int8))
                yhat_parts.append(out["yhat_std"].cpu().numpy())
                self_parts.append(out["self_sum"].cpu().numpy())
                bond_sum_parts.append(out["bond_sum"].cpu().numpy())
                atom_count_parts.append(out["atom_count"].cpu().numpy())
                bond_count_parts.append(out["bond_count"].cpu().numpy())
                if beta is None:
                    beta = float(model.beta.detach().cpu())
                    c_n = float(model.c_n.detach().cpu())
                    c_m = float(model.c_m.detach().cpu())

    out: dict[str, Any] = {
        "atom_row": np.concatenate(atom_row), "atom_local": np.concatenate(atom_local),
        "atom_split": np.concatenate(atom_split), "A": np.concatenate(A_parts),
        "z": np.concatenate(z_parts),
        "bond_row": np.concatenate(bond_row), "bond_local": np.concatenate(bond_local),
        "bond_u": np.concatenate(bond_u), "bond_v": np.concatenate(bond_v),
        "bond_split": np.concatenate(bond_split), "Q": np.concatenate(Q_parts),
        "lam": np.concatenate(lam_parts),
        "word_row": np.concatenate(word_row), "word_split": np.concatenate(word_split),
        "word_root": np.concatenate(word_root), "word_support": np.concatenate(word_support),
        "word_internal": np.concatenate(word_internal),
        "word_size": np.concatenate(word_size),
        "alpha": np.concatenate(alpha_parts), "c": np.concatenate(c_parts),
        "c_self": np.concatenate(c_self_parts), "c_bond": np.concatenate(c_bond_parts),
        "cert": np.concatenate(cert_parts, axis=0),
        "mol_row": np.concatenate(mol_row), "mol_split": np.concatenate(mol_split),
        "yhat": np.concatenate(yhat_parts), "self_sum": np.concatenate(self_parts),
        "bond_sum": np.concatenate(bond_sum_parts),
        "atom_count": np.concatenate(atom_count_parts),
        "bond_count": np.concatenate(bond_count_parts),
        "rows_all": rows_all, "splits": splits, "row_to_idx": row_to_idx,
        "beta": beta, "c_n": c_n, "c_m": c_m,
    }
    return out


def _prediction_check(bundle: D.DataBundle, gathered: Mapping[str, Any]) -> dict[str, Any]:
    z = np.load(RUN_DIR / "predictions_best.npz", allow_pickle=False)
    pred = np.asarray(z["pred_raw"], np.float64)
    rows = np.asarray(z["row_id"], np.int64)
    split = np.asarray(z["split"])
    order = np.argsort(rows)
    pred_by_row = np.full(4200, np.nan, np.float64)
    pred_by_row[rows] = pred
    raw = gathered["yhat"] * bundle.scaler["std"] + bundle.scaler["mean"]
    diff = np.abs(raw - pred_by_row[gathered["mol_row"]])
    return {"max_abs_reload_diff": float(diff.max()) if diff.size else 0.0,
            "n_rows": int(rows.size),
            "splits_in_file": {str(s): int(np.sum(split == s)) for s in np.unique(split)},
            "prediction_reload_ok": bool(diff.size == 4200 and diff.max() < 1e-5)}


def _selection_stats(gathered: Mapping[str, Any]) -> dict[str, Any]:
    rows_idx = gathered["row_to_idx"]
    atom_root = rows_idx[gathered["atom_row"]] * MAX_ATOMS_KEY + gathered["atom_local"]
    word_root = rows_idx[gathered["word_row"]] * MAX_ATOMS_KEY + gathered["word_root"]
    n_root_slots = int(4200 * MAX_ATOMS_KEY)
    alpha = gathered["alpha"]
    size = gathered["word_size"]
    a_split = gathered["atom_split"]
    w_split = gathered["word_split"]
    atom_split = gathered["atom_split"]
    out: dict[str, Any] = {}
    for code, name in ((0, "train"), (1, "valid"), (2, "test")):
        atom_mask = atom_split == code
        root_ids = atom_root[atom_mask]
        root_count = int(root_ids.size)
        nonzero = np.bincount(word_root[w_split == code],
                              weights=(alpha[w_split == code] > 0).astype(np.float64),
                              minlength=n_root_slots)[root_ids]
        top1 = np.zeros(n_root_slots, np.float64)
        np.maximum.at(top1, word_root[w_split == code], alpha[w_split == code])
        top1 = top1[root_ids]
        mass = {}
        counts = {}
        for s in (1, 2, 3, 4):
            m = (w_split == code) & (size == s)
            mass[str(s)] = float(alpha[m].sum()) / max(1, root_count)
            counts[str(s)] = int(m.sum())
        z = gathered["z"][atom_mask]
        A = gathered["A"][atom_mask]
        Q = gathered["Q"][gathered["bond_split"] == code]
        mol_mask = gathered["mol_split"] == code
        atom_abs = np.zeros(int(mol_mask.sum()))
        np.add.at(atom_abs, rows_idx[gathered["atom_row"][atom_mask]] -
                  int(np.flatnonzero(mol_mask)[0]), np.abs(A))
        bond_abs = np.zeros(int(mol_mask.sum()))
        bond_rows = gathered["bond_row"][gathered["bond_split"] == code]
        np.add.at(bond_abs, rows_idx[bond_rows] - int(np.flatnonzero(mol_mask)[0]),
                  np.abs(Q))
        self_sum = gathered["self_sum"][mol_mask]
        bond_sum = gathered["bond_sum"][mol_mask]
        eps = 1e-12
        out[name] = {
            "roots": root_count,
            "nonzero_candidates_per_root": _stats(nonzero),
            "nonzero_candidates_histogram": _hist(nonzero, 0, 100),
            "top1_weight": _stats(top1),
            "top1_weight_histogram": _hist(np.round(top1 * 20).astype(np.int64), 0, 20),
            "alpha_mass_by_size": mass,
            "candidate_count_by_size": counts,
            "z": _effective_rank(z),
            "A_rms": float(np.sqrt(np.mean(A * A))) if A.size else 0.0,
            "A_mean_abs": float(np.mean(np.abs(A))) if A.size else 0.0,
            "Q_rms": float(np.sqrt(np.mean(Q * Q))) if Q.size else 0.0,
            "Q_mean_abs": float(np.mean(np.abs(Q))) if Q.size else 0.0,
            "Q_nonzero_fraction_1e_8": float(np.mean(np.abs(Q) > 1e-8)) if Q.size else 0.0,
            "Q_nonzero_fraction_1e_6": float(np.mean(np.abs(Q) > 1e-6)) if Q.size else 0.0,
            "lambda_mean_abs": float(np.mean(np.abs(gathered["lam"]))) if gathered["lam"].size else 0.0,
            "lambda_nonzero_fraction_1e_6": float(np.mean(np.abs(gathered["lam"]) > 1e-6)) if gathered["lam"].size else 0.0,
            "self_cancellation": float(np.mean(np.abs(self_sum) / (atom_abs + eps))) if atom_abs.size else 0.0,
            "bond_cancellation": float(np.mean(np.abs(bond_sum) / (bond_abs + eps))) if bond_abs.size else 0.0,
            "total_cancellation": float(np.mean(np.abs(self_sum + bond_sum) /
                                                (atom_abs + bond_abs + eps))) if atom_abs.size else 0.0,
            "self_sum_rms": float(np.sqrt(np.mean(self_sum ** 2))) if self_sum.size else 0.0,
            "bond_sum_rms": float(np.sqrt(np.mean(bond_sum ** 2))) if bond_sum.size else 0.0,
        }
    return out


def _contribution_stats(gathered: Mapping[str, Any]) -> dict[str, Any]:
    w_split = gathered["word_split"]
    size = gathered["word_size"]
    c = gathered["c"]
    out: dict[str, Any] = {"by_split": {}, "by_size": {}, "overall": {}}
    for code, name in ((0, "train"), (1, "valid"), (2, "test")):
        m = w_split == code
        entry = {}
        for s in (1, 2, 3, 4):
            mm = m & (size == s)
            vals = c[mm]
            entry[str(s)] = _contribution_block(vals)
        entry["all"] = _contribution_block(c[m])
        out["by_split"][name] = entry
    for s in (1, 2, 3, 4):
        out["by_size"][str(s)] = _contribution_block(c[size == s])
    out["overall"] = _contribution_block(c)
    # singleton vs multi-atom, in absolute and net terms
    single = size == 1
    multi = ~single
    for name, mask in (("singleton", single), ("multi_atom", multi)):
        vals = c[mask]
        out[f"{name}_share"] = {
            "abs_share": float(np.sum(np.abs(vals)) / max(1e-12, np.sum(np.abs(c)))),
            "net_share": float(np.sum(vals) / max(1e-12, np.sum(c))),
            "count": int(vals.size), "mean": float(vals.mean()) if vals.size else 0.0,
            "mean_abs": float(np.mean(np.abs(vals))) if vals.size else 0.0,
            "positive_fraction": float(np.mean(vals > 0)) if vals.size else 0.0,
        }
    return out


def _contribution_block(vals: np.ndarray) -> dict[str, float]:
    vals = np.asarray(vals, np.float64)
    if vals.size == 0:
        return {"count": 0, "sum": 0.0, "sum_abs": 0.0, "mean": 0.0,
                "mean_abs": 0.0, "std": 0.0, "positive_fraction": 0.0,
                "cancellation": 0.0}
    net = float(vals.sum())
    ab = float(np.abs(vals).sum())
    return {"count": int(vals.size), "sum": net, "sum_abs": ab,
            "mean": float(vals.mean()), "mean_abs": float(np.abs(vals).mean()),
            "std": float(vals.std()), "positive_fraction": float(np.mean(vals > 0)),
            "cancellation": float(1.0 - abs(net) / ab) if ab > 0 else 0.0}


def _word_reuse(bundle: D.DataBundle, gathered: Mapping[str, Any],
                atom_dims: Sequence[int], bond_dims: Sequence[int]) -> dict[str, Any]:
    row_to_idx = gathered["row_to_idx"]
    cert = np.asarray(gathered["cert"], np.int64)
    keys = CAN.certificate_key(cert)
    uniq_keys, inv = np.unique(keys, return_inverse=True)
    n_classes = int(uniq_keys.size)
    rep_cert = np.zeros((n_classes, 10), np.int64)
    rep_cert[inv] = cert
    rep_size = np.zeros(n_classes, np.int64)
    rep_size[inv] = gathered["word_size"]
    w_row_idx = row_to_idx[gathered["word_row"]]
    w_split = gathered["word_split"]
    c = gathered["c"].astype(np.float64)

    # Train scaffold ids for the official Train rows.
    row_scaf = np.full(4200, -1, np.int64)
    scaf_ids = {str(s): i for i, s in enumerate(sorted(set(
        bundle.scaffold_by_row.values())))}
    for row, scaf in bundle.scaffold_by_row.items():
        row_scaf[int(row)] = scaf_ids[scaf]
    if np.any(row_scaf[bundle.train_idx] < 0):
        raise ValueError("missing scaffold for a Train row")

    def distinct_counts(class_ids: np.ndarray, aux: np.ndarray) -> np.ndarray:
        pairs = np.stack([class_ids, aux], axis=1).astype(np.int64)
        upairs = np.unique(pairs, axis=0)
        return np.bincount(upairs[:, 0], minlength=n_classes)

    train_m = w_split == 0
    train_mol = distinct_counts(inv[train_m], w_row_idx[train_m])
    train_scaf = distinct_counts(inv[train_m], row_scaf[gathered["word_row"][train_m]])
    valid_mol = distinct_counts(inv[w_split == 1], w_row_idx[w_split == 1])
    test_mol = distinct_counts(inv[w_split == 2], w_row_idx[w_split == 2])

    per_split: dict[str, dict[str, np.ndarray]] = {}
    for code, name in ((0, "train"), (1, "valid"), (2, "test")):
        m = w_split == code
        idx = inv[m]
        vals = c[m]
        count = np.bincount(idx, minlength=n_classes)
        sum_c = np.bincount(idx, weights=vals, minlength=n_classes)
        sum_abs = np.bincount(idx, weights=np.abs(vals), minlength=n_classes)
        sumsq = np.bincount(idx, weights=vals * vals, minlength=n_classes)
        pos = np.bincount(idx, weights=(vals > 0).astype(np.float64), minlength=n_classes)
        per_split[name] = {"count": count, "sum": sum_c, "sum_abs": sum_abs,
                           "sumsq": sumsq, "pos": pos}
    mean_abs_train = per_split["train"]["sum_abs"] / np.maximum(1, per_split["train"]["count"])
    qualify = (train_mol >= int(C.load_config()["word_reuse"]["min_train_molecules"])) & \
              (train_scaf >= int(C.load_config()["word_reuse"]["min_train_scaffolds"]))
    qidx = np.flatnonzero(qualify)
    order = qidx[np.argsort(-mean_abs_train[qidx])]
    top_n = int(C.load_config()["word_reuse"]["top_words"])
    top = order[:top_n]

    by_size = {}
    for s in (1, 2, 3, 4):
        by_size[str(s)] = {
            "classes": int(np.sum(rep_size == s)),
            "qualifying_classes": int(np.sum((rep_size == s) & qualify)),
        }

    def _entry(i: int) -> dict[str, Any]:
        row = {
            "size": int(rep_size[i]),
            "train_molecules": int(train_mol[i]),
            "train_scaffolds": int(train_scaf[i]),
            "train_count": int(per_split["train"]["count"][i]),
            "train_mean_abs": float(mean_abs_train[i]),
            "valid_count": int(per_split["valid"]["count"][i]),
            "valid_molecules": int(valid_mol[i]),
            "test_count": int(per_split["test"]["count"][i]),
            "test_molecules": int(test_mol[i]),
        }
        for name in ("train", "valid", "test"):
            cnt = per_split[name]["count"][i]
            mean = per_split[name]["sum"][i] / cnt if cnt else 0.0
            var = per_split[name]["sumsq"][i] / cnt - mean * mean if cnt else 0.0
            row[f"{name}_mean"] = float(mean)
            row[f"{name}_std"] = float(math.sqrt(max(0.0, var)))
            row[f"{name}_mean_raw"] = float(mean * bundle.scaler["std"])
            row[f"{name}_std_raw"] = float(math.sqrt(max(0.0, var)) * bundle.scaler["std"])
            row[f"{name}_positive_fraction"] = float(per_split[name]["pos"][i] / cnt) if cnt else 0.0
        row["train_mean_abs_raw"] = float(mean_abs_train[i] * bundle.scaler["std"])
        decoded = CAN.certificate_row(rep_cert[i], int(rep_size[i]), atom_dims, bond_dims)
        row["certificate"] = decoded["certificate"]
        row["node_fields"] = decoded["node_fields"]
        row["internal_bonds"] = decoded["internal_bonds"]
        row["root_slot"] = 0
        return row

    top_entries = [_entry(int(i)) for i in top.tolist()]
    all_qualifying = []
    for i in order.tolist():
        all_qualifying.append(_entry(int(i)))
    summary = {
        "n_exact_rooted_word_classes": n_classes,
        "classes_by_size": by_size,
        "n_qualifying_train_molecules_ge_20_and_scaffolds_ge_3": int(qidx.size),
        "train_word_instances": int(per_split["train"]["count"].sum()),
        "train_qualifying_instances": int(per_split["train"]["count"][qualify].sum()),
        "train_qualifying_molecules_covered": int(len(set(
            w_row_idx[train_m & qualify[inv]].tolist()))),
        "valid_word_instances": int(per_split["valid"]["count"].sum()),
        "test_word_instances": int(per_split["test"]["count"].sum()),
        "min_train_molecules": int(C.load_config()["word_reuse"]["min_train_molecules"]),
        "min_train_scaffolds": int(C.load_config()["word_reuse"]["min_train_scaffolds"]),
        "ranking": "Train mean absolute contribution; Valid/Test labels never used",
    }
    return {"summary": summary, "top": top_entries,
            "all_qualifying": all_qualifying}


def _high_contribution_words(bundle: D.DataBundle, gathered: Mapping[str, Any],
                             top_k: int = 3) -> list[dict[str, Any]]:
    raw_atom = np.asarray(bundle.raw["atom_feat"], np.int64)
    raw_bond = np.asarray(bundle.raw["bond_feat"], np.int64)
    node_ptr = np.asarray(bundle.raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(bundle.raw["bond_ptr"], np.int64)
    smiles = np.asarray(bundle.raw["smiles"])
    rows = gathered["mol_row"]
    records: list[dict[str, Any]] = []
    for i, row in enumerate(rows.tolist()):
        sel = np.flatnonzero(gathered["word_row"] == row)
        if sel.size == 0:
            continue
        order = sel[np.argsort(-np.abs(gathered["c"][sel]))[:top_k]]
        a0 = int(node_ptr[row])
        b0 = int(bond_ptr[row])
        for rank, wi in enumerate(order.tolist(), start=1):
            support = gathered["word_support"][wi]
            support = support[support >= 0]
            internal = gathered["word_internal"][wi]
            internal = internal[internal >= 0]
            atom_fields = raw_atom[a0 + support]
            bond_fields = raw_bond[b0 + internal] if internal.size else np.zeros((0, 3), np.int64)
            records.append({
                "row_id": int(row),
                "split": ["train", "valid", "test"][int(gathered["mol_split"][i])],
                "rank": int(rank),
                "contribution_std": float(gathered["c"][wi]),
                "contribution_raw": float(gathered["c"][wi] * bundle.scaler["std"]),
                "alpha": float(gathered["alpha"][wi]),
                "word_size": int(gathered["word_size"][wi]),
                "root_local": int(gathered["word_root"][wi]),
                "support_local": [int(x) for x in support.tolist()],
                "support_atom_fields": [list(map(int, x)) for x in atom_fields],
                "internal_bond_fields": [list(map(int, x)) for x in bond_fields],
                "smiles": str(smiles[row]),
            })
    return records


def _verdict(run: Mapping[str, Any], selection: Mapping[str, Any],
             contributions: Mapping[str, Any]) -> dict[str, Any]:
    valid_rmse = float(run["valid_rmse"])
    if valid_rmse <= 0.65:
        predictive = "PREDICTIVE_CANDIDATE"
    elif valid_rmse <= 0.70:
        predictive = "WEAK_PREDICTIVE_SIGNAL"
    else:
        predictive = "NO_COMPETITIVE_SIGNAL"
    train_mass = selection["train"]["alpha_mass_by_size"]
    multi_mass = float(sum(train_mass[str(s)] for s in (2, 3, 4)))
    train_blocks = contributions["by_split"]["train"]
    total_abs = float(train_blocks["all"]["sum_abs"])
    multi_abs = float(sum(train_blocks[str(s)]["sum_abs"] for s in (2, 3, 4)))
    multi_share = multi_abs / total_abs if total_abs > 0 else 0.0
    if predictive == "NO_COMPETITIVE_SIGNAL":
        word = "STOP_THIS_IMPLEMENTATION"
    elif multi_mass < 0.10 or multi_share < 0.10:
        word = "PREDICTIVE_BUT_WORD_EVIDENCE_WEAK"
    else:
        word = "CONTINUE_DISCUSSION"
    return {
        "valid_rmse": valid_rmse,
        "predictive_verdict": predictive,
        "word_evidence_verdict": word,
        "train_multi_atom_alpha_mass": multi_mass,
        "train_multi_atom_abs_contribution_share": multi_share,
        "word_evidence_rule": {
            "no_competitive_max_valid_rmse": 0.70,
            "weak_multi_atom_alpha_mass_below": 0.10,
            "weak_multi_atom_abs_contribution_share_below": 0.10,
        },
        "gine_reference": "0.61-0.64 historical, different recipe, descriptive only",
        "test_metric_label": "DEVELOPMENT_TEST",
        "causal_claim": False,
    }


def main() -> dict[str, Any]:
    cfg = C.load_config()
    bundle = D.load_bundle()
    run = C.read_json(RUN_DIR / "run.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt, info = _load_model(bundle, run, device)
    all_rows = np.union1d(np.union1d(bundle.train_idx, bundle.valid_idx), bundle.test_idx)
    structs = S.build_all(bundle.raw, all_rows)
    gathered = _gather(bundle, structs, model, device)
    pred_check = _prediction_check(bundle, gathered)
    selection = _selection_stats(gathered)
    contributions = _contribution_stats(gathered)
    verdict = _verdict(run, selection, contributions)
    reuse = _word_reuse(bundle, gathered, bundle.atom_dims, bundle.bond_dims)
    high = _high_contribution_words(bundle, gathered, top_k=3)

    # per-molecule exact identity
    rows_idx = gathered["row_to_idx"]
    word_mol_idx = rows_idx[gathered["word_row"]]
    n_mol = int(gathered["mol_row"].size)
    sum_c = np.zeros(n_mol, np.float64)
    np.add.at(sum_c, word_mol_idx, gathered["c"].astype(np.float64))
    target = (gathered["yhat"] - gathered["beta"]
              - gathered["c_n"] * np.log1p(gathered["atom_count"])
              - gathered["c_m"] * np.log1p(gathered["bond_count"]))
    ident_err = np.abs(sum_c - target)
    identity = {
        "max_abs_error": float(ident_err.max()) if ident_err.size else 0.0,
        "mean_abs_error": float(ident_err.mean()) if ident_err.size else 0.0,
        "p99_abs_error": float(np.percentile(ident_err, 99)) if ident_err.size else 0.0,
        "tolerance_float32": 1e-4,
        "pass": bool(ident_err.size == 4200 and ident_err.max() < 1e-4),
    }

    # save reproducible per-word contributions and diagnostics
    np.savez_compressed(
        RUN_DIR / "contributions_best.npz",
        row=gathered["word_row"].astype(np.int16),
        split=gathered["word_split"].astype(np.int8),
        root_local=gathered["word_root"].astype(np.int16),
        support_local=gathered["word_support"].astype(np.int16),
        internal_local=gathered["word_internal"].astype(np.int16),
        word_size=gathered["word_size"].astype(np.int8),
        alpha=gathered["alpha"].astype(np.float32),
        contribution_std=gathered["c"].astype(np.float32),
        contribution_raw=(gathered["c"] * bundle.scaler["std"]).astype(np.float32),
        contribution_self_std=gathered["c_self"].astype(np.float32),
        contribution_bond_std=gathered["c_bond"].astype(np.float32),
        atom_row=gathered["atom_row"].astype(np.int16),
        atom_local=gathered["atom_local"].astype(np.int16),
        atom_split=gathered["atom_split"].astype(np.int8),
        A_std=gathered["A"].astype(np.float32),
        bond_row=gathered["bond_row"].astype(np.int16),
        bond_local=gathered["bond_local"].astype(np.int16),
        bond_split=gathered["bond_split"].astype(np.int8),
        bond_u_local=gathered["bond_u"].astype(np.int16),
        bond_v_local=gathered["bond_v"].astype(np.int16),
        Q_std=gathered["Q"].astype(np.float32),
        mol_row=gathered["mol_row"].astype(np.int16),
        mol_split=gathered["mol_split"].astype(np.int8),
        yhat_std=gathered["yhat"].astype(np.float32),
        self_sum_std=gathered["self_sum"].astype(np.float32),
        bond_sum_std=gathered["bond_sum"].astype(np.float32),
        identity_error=ident_err.astype(np.float32),
    )
    mol_rows = []
    for i, row in enumerate(gathered["mol_row"].tolist()):
        mol_rows.append({
            "row_id": int(row),
            "split": ["train", "valid", "test"][int(gathered["mol_split"][i])],
            "yhat_std": float(gathered["yhat"][i]),
            "self_sum_std": float(gathered["self_sum"][i]),
            "bond_sum_std": float(gathered["bond_sum"][i]),
            "atom_count": int(gathered["atom_count"][i]),
            "bond_count": int(gathered["bond_count"][i]),
            "identity_error": float(ident_err[i]),
            "n_words": int(np.sum(word_mol_idx == i)),
        })
    C.write_csv(RUN_DIR / "contributions_best.csv", mol_rows)
    C.write_csv(C.ANALYSIS_DIR / "high_contribution_words.csv", high)

    C.write_json(C.ANALYSIS_DIR / "word_selection.json", {
        "checkpoint": str(RUN_DIR / "best.pt"),
        "best_epoch": int(run["best_epoch"]),
        "selection": selection,
        "identity": identity,
        "prediction_reload": pred_check,
    })
    C.write_json(C.ANALYSIS_DIR / "word_contributions.json", {
        "checkpoint": str(RUN_DIR / "best.pt"),
        "best_epoch": int(run["best_epoch"]),
        "stats": contributions,
        "contribution_units": {
            "standardized": "Train-label standard deviations",
            "raw": "standardized * Train label population std",
            "train_label_std": float(bundle.scaler["std"]),
        },
        "identity": identity,
        "high_contribution_words_top_k": 3,
        "high_contribution_words_records": len(high),
    })
    C.write_json(C.ANALYSIS_DIR / "verdict.json", verdict)
    C.write_json(C.ANALYSIS_DIR / "word_reuse.json", {
        "checkpoint": str(RUN_DIR / "best.pt"),
        "best_epoch": int(run["best_epoch"]),
        "summary": reuse["summary"],
        "top": reuse["top"],
    })
    # all qualifying classes are kept in a CSV (top 2000 by Train mean |c|)
    rows = []
    for entry in reuse["all_qualifying"][:2000]:
        rows.append({
            "size": entry["size"],
            "certificate": " ".join(str(x) for x in entry["certificate"]),
            "node_fields": json.dumps(entry["node_fields"]),
            "internal_bonds": json.dumps(entry["internal_bonds"]),
            "train_molecules": entry["train_molecules"],
            "train_scaffolds": entry["train_scaffolds"],
            "train_count": entry["train_count"],
            "train_mean_abs": entry["train_mean_abs"],
            "train_mean_abs_raw": entry["train_mean_abs_raw"],
            "train_mean": entry["train_mean"],
            "train_std": entry["train_std"],
            "train_mean_raw": entry["train_mean_raw"],
            "train_std_raw": entry["train_std_raw"],
            "train_positive_fraction": entry["train_positive_fraction"],
            "valid_count": entry["valid_count"],
            "valid_molecules": entry["valid_molecules"],
            "valid_mean": entry["valid_mean"],
            "valid_std": entry["valid_std"],
            "valid_mean_raw": entry["valid_mean_raw"],
            "valid_std_raw": entry["valid_std_raw"],
            "valid_positive_fraction": entry["valid_positive_fraction"],
            "test_count": entry["test_count"],
            "test_molecules": entry["test_molecules"],
            "test_mean": entry["test_mean"],
            "test_std": entry["test_std"],
            "test_mean_raw": entry["test_mean_raw"],
            "test_std_raw": entry["test_std_raw"],
            "test_positive_fraction": entry["test_positive_fraction"],
        })
    C.write_csv(C.ANALYSIS_DIR / "word_types.csv", rows)
    # per-epoch curve copy as JSON
    with (RUN_DIR / "curves.csv").open(newline="", encoding="utf-8") as handle:
        curve_rows = list(csv.DictReader(handle))
    C.write_json(C.ANALYSIS_DIR / "curves.json", curve_rows)

    summary = {
        "best_epoch": int(run["best_epoch"]),
        "valid_rmse": float(run["valid_rmse"]),
        "valid_mae": float(run["valid_mae"]),
        "test_rmse": float(run["test_rmse"]),
        "test_mae": float(run["test_mae"]),
        "test_metric_label": "DEVELOPMENT_TEST",
        "prediction_reload": pred_check,
        "identity": identity,
        "selection": selection,
        "contribution_stats": contributions,
        "word_reuse_summary": reuse["summary"],
        "verdict": verdict,
    }
    C.write_json(C.ANALYSIS_DIR / "analysis_summary.json", summary)
    print(json.dumps(C.jsonable(summary), indent=1, sort_keys=True)[:6000])
    return summary


if __name__ == "__main__":
    main()