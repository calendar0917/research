"""Trained-function dependence diagnostics and per-example prediction traceback.

Two frozen analyses from the protocol:

A. **Fixed-weight interventions** (122 Valid, first 64 rows, FULL seeds 0/1/2):
   zero standardised ``g``, zero the atom-sum contribution, zero ``mean_h2`` at
   the rho input, zero the typed-S2 block, and switch the FULL joint/content
   aggregation to the B5 (root, bt2) marginal definition.  These are dependence
   diagnostics on one trained function; they do not identify mechanisms.

B. **Leave-one-occurrence-out over typed-S2 paths** (FULL seed0, the 8 frozen
   Valid rows): remove one S2 record from its root atom's input, keeping the
   graph, S1 terms, raw scalars and ``g`` fixed.  This is an input-channel
   intervention on the model, *not* a chemical edit of the molecule.

Every number is in raw logD units and is labelled model-internal.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import dissect_common as C
import dissect_data as D
import dissect_model as M
import dissect_source as S

ATTR = C.ANALYSIS / "attribution"
FIG = ATTR / "figures"

INTERVENTIONS = [
    ("g_zero", {"zero_g": True}, "standardised global g set to 0"),
    ("contrib_zero", {"zero_contrib": True}, "atom-sum contribution L set to 0"),
    ("mean_h2_zero", {"zero_mean_h2": True}, "mean_v h_v^2 zeroed at the rho input"),
    ("typed_s2_zero", {"zero_s2": True}, "typed-S2 block (200-d) zeroed; raw scalars and g fixed"),
    ("agg_b5", {"s2_agg_override": "marginal_bt2"}, "FULL weights, joint/content aggregation = B5"),
]


def _view(repeat: int):
    return D.build_view(repeat)


def _model(seed: int, variant: str = "A0", repeat: int = C.MAIN_REPEAT):
    view = _view(repeat)
    ckpt_path = C.run_dir(repeat, variant, seed) / "checkpoint.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"missing checkpoint {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = M.load_checkpoint_model(
        view.spec, view.train_norm, variant, ckpt["state_dict"], torch.device("cpu")
    )
    model.eval()
    return view, model, ckpt


def _metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = pred - y
    return {
        "n": int(y.size),
        "rmse": float(np.sqrt((err * err).mean())),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
    }


# ---------------------------------------------------------------------------
# A. fixed-weight interventions
# ---------------------------------------------------------------------------


def part_a(rows: Sequence[int] | None = None) -> dict[str, Any]:
    view = _view(C.MAIN_REPEAT)
    rows = list(range(64)) if rows is None else list(rows)
    batch = D.to_device(D.PREP_DATA.collate(view.valid_cache, rows), torch.device("cpu"))
    y = view.y_valid[rows]
    out: dict[str, Any] = {
        "repeat": C.MAIN_REPEAT,
        "valid_row_positions": rows,
        "n_rows": len(rows),
        "checkpoints": {},
        "seeds": {},
    }
    for seed in C.SEEDS:
        _, model, ckpt = _model(seed)
        out["checkpoints"][str(seed)] = C.sha256_file(C.run_dir(C.MAIN_REPEAT, "A0", seed) / "checkpoint.pt")
        tgt = ckpt["target_norm"]
        scale, mean = float(tgt["scale"]), float(tgt["mean"])
        with torch.no_grad():
            y0, a0 = model(batch, return_aux=True)
        pred0 = y0.numpy() * scale + mean
        contrib0 = a0["contrib"].numpy() * scale
        rho0 = a0["rho"].numpy() * scale
        entry: dict[str, Any] = {
            "baseline": {
                "valid_rmse": _metrics(pred0, y)["rmse"],
                "valid_mae": _metrics(pred0, y)["mae"],
                "contrib_mean": float(contrib0.mean()),
                "contrib_sd": float(contrib0.std(ddof=1)),
                "rho_mean": float(rho0.mean()),
                "rho_sd": float(rho0.std(ddof=1)),
                "corr_contrib_rho": float(np.corrcoef(contrib0, rho0)[0, 1]),
                "norm_g": float(np.linalg.norm(a0["rho_input"][:, : C.GLOBAL_DIM].numpy(), axis=1).mean()),
                "norm_contrib": float(np.abs(contrib0).mean()),
                "norm_mean_h2": float(a0["mean_h2"].norm(dim=1).mean()),
                "norm_path_agg": float(a0["path_agg"].norm(dim=1).mean()),
                "norm_content_agg": float(a0["content_agg"].norm(dim=1).mean()),
            },
            "interventions": {},
        }
        for name, kwargs, note in INTERVENTIONS:
            with torch.no_grad():
                y1, a1 = model(batch, return_aux=True, **kwargs)
            pred1 = y1.numpy() * scale + mean
            delta = pred1 - pred0
            norms = {
                "norm_g": float(np.linalg.norm(a1["rho_input"][:, : C.GLOBAL_DIM].numpy(), axis=1).mean()),
                "norm_contrib": float(np.abs(a1["contrib_used"].numpy() * scale).mean()),
                "norm_mean_h2": float(
                    np.linalg.norm(
                        a1["rho_input"][
                            :, C.GLOBAL_DIM + int(model.cfg["width"]) : C.GLOBAL_DIM + 2 * int(model.cfg["width"])
                        ].numpy(),
                        axis=1,
                    ).mean()
                ),
                "norm_path_agg": float(a1["path_agg"].norm(dim=1).mean()),
                "norm_content_agg": float(a1["content_agg"].norm(dim=1).mean()),
            }
            entry["interventions"][name] = {
                "note": note,
                "mean_abs_delta": float(np.abs(delta).mean()),
                "max_abs_delta": float(np.abs(delta).max()),
                "rmse_after": _metrics(pred1, y)["rmse"],
                "delta_rmse": _metrics(pred1, y)["rmse"] - _metrics(pred0, y)["rmse"],
                "coverage_rows": float((np.abs(delta) > 1e-6).mean()),
                "norms_after": norms,
            }
        out["seeds"][str(seed)] = entry
    # aggregate across seeds
    agg = {}
    for name, _, note in INTERVENTIONS:
        vals = [out["seeds"][str(s)]["interventions"][name] for s in C.SEEDS]
        agg[name] = {
            "note": note,
            "mean_abs_delta_mean": float(np.mean([v["mean_abs_delta"] for v in vals])),
            "max_abs_delta_max": float(np.max([v["max_abs_delta"] for v in vals])),
            "delta_rmse_mean": float(np.mean([v["delta_rmse"] for v in vals])),
            "delta_rmse_per_seed": [round(v["delta_rmse"], 6) for v in vals],
            "coverage_rows_mean": float(np.mean([v["coverage_rows"] for v in vals])),
        }
    out["aggregate"] = agg
    return out


# ---------------------------------------------------------------------------
# B. leave-one-occurrence-out typed-S2 paths
# ---------------------------------------------------------------------------


def single_molecule_cache(cache: Mapping[str, Any], pos: int, drop_path: int | None = None) -> dict[str, Any]:
    """Slice molecule ``pos`` out of a split cache, optionally dropping one path record."""
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    a0, a1 = int(atom_ptr[pos]), int(atom_ptr[pos + 1])
    b0, b1 = int(bond_ptr[pos]), int(bond_ptr[pos + 1])
    p0, p1 = int(path_ptr[a0]), int(path_ptr[a1])
    keep = np.asarray([j for j in range(p0, p1) if j != drop_path], dtype=np.int64)
    n_atoms = a1 - a0
    # per-atom path counts after the drop
    per_atom = []
    for g in range(a0, a1):
        lo, hi = int(path_ptr[g]), int(path_ptr[g + 1])
        per_atom.append(sum(1 for j in range(lo, hi) if j != drop_path))
    new_path_ptr = np.concatenate([[0], np.cumsum(per_atom)]).astype(np.int64)
    nbr_ptr_all = np.asarray(cache["nbr_ptr"])
    per_atom_nbr = [int(nbr_ptr_all[g + 1]) - int(nbr_ptr_all[g]) for g in range(a0, a1)]
    new_nbr_ptr = np.concatenate([[0], np.cumsum(per_atom_nbr)]).astype(np.int64)
    sub: dict[str, Any] = {
        "atom_ptr": np.asarray([0, n_atoms], dtype=np.int64),
        "bond_ptr": np.asarray([0, b1 - b0], dtype=np.int64),
        "nbr_ptr": new_nbr_ptr,
        "path_ptr": new_path_ptr,
        "atom_feat": np.asarray(cache["atom_feat"][a0:a1]),
        "atom_type": np.asarray(cache["atom_type"][a0:a1]),
        "atom_scalar": np.asarray(cache["atom_scalar"][a0:a1]),
        "bond_feat": np.asarray(cache["bond_feat"][b0:b1]),
        "bond_type": np.asarray(cache["bond_type"][b0:b1]),
        "bond_index": np.asarray(cache["bond_index"][b0:b1]),
        "nbr_code": np.asarray(cache["nbr_code"][int(cache["nbr_ptr"][a0]) : int(cache["nbr_ptr"][a1])]),
        "path_cat": np.asarray(cache["path_cat"])[keep],
        "path_end63": np.asarray(cache["path_end63"])[keep],
        "path_end": np.asarray(cache["path_end"])[keep],
        "path_closure": np.asarray(cache["path_closure"])[keep],
        "global": np.asarray(cache["global"][pos : pos + 1]),
        "n_components": np.asarray(cache["n_components"][pos : pos + 1]),
        "n_junctions": np.asarray(cache["n_junctions"][pos : pos + 1]),
        "n_ring_systems": np.asarray(cache["n_ring_systems"][pos : pos + 1]),
        "molecule_ids": [str(cache["molecule_ids"][pos])],
        "smiles": [str(cache["smiles"][pos])],
        "n_molecules": 1,
        "split": cache["split"],
        "meta": dict(cache["meta"]),
        "rows": np.asarray([int(np.asarray(cache["rows"])[pos])], dtype=np.int64),
        "dataset": cache.get("dataset"),
        "split_name": cache.get("split_name"),
        "repeat": cache.get("repeat"),
        "cache_version": cache.get("cache_version"),
        "source_commit": cache.get("source_commit"),
    }
    sub["meta"]["n_atoms_total"] = int(n_atoms)
    sub["meta"]["n_bonds_total"] = int(b1 - b0)
    sub["meta"]["n_s1_codes"] = int(sub["nbr_code"].size)
    sub["meta"]["n_s2_paths"] = int(keep.size)
    return sub


def _atom_symbols(smiles: str) -> list[str]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(smiles)
    return [atom.GetSymbol() for atom in mol.GetAtoms()]


def _path_labels(cache_sub: Mapping[str, Any], ptr: np.ndarray) -> list[dict[str, Any]]:
    """Identity of every S2 record: root / candidate middles / end + two bond types.

    The frozen S2 record stores ``(path_cat, path_end, closure)``; the middle
    atom index is *not* encoded (only its type, inside ``path_cat``), so the
    middle is reported as the set of graph atoms consistent with
    ``(root, bt1, end, bt2)``.
    """
    from rdkit import Chem

    smiles = str(cache_sub["smiles"][0])
    mol = Chem.MolFromSmiles(smiles)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(mol.GetNumAtoms())}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt_raw = bond.GetBondTypeAsDouble()
        bt = 4 if bt_raw == 1.5 else int(bt_raw)
        adj[i].append((j, bt))
        adj[j].append((i, bt))
    out = []
    for j, (cat, end) in enumerate(zip(cache_sub["path_cat"], cache_sub["path_end"])):
        cat, end = int(cat), int(end)
        bt1 = (cat // C.N_BOND_TYPES) % C.N_BOND_TYPES + 1
        bt2 = cat % C.N_BOND_TYPES + 1
        root = int(np.searchsorted(ptr, j, side="right") - 1)
        middles = sorted(
            w for w, b1 in adj.get(root, []) if b1 == bt1 and any(u == end and b2 == bt2 for u, b2 in adj[w])
        )
        out.append(
            {
                "path_index": j,
                "root": root,
                "root_symbol": symbols[root] if root < len(symbols) else "?",
                "middle_candidates": middles,
                "middle_symbol": symbols[middles[0]] if len(middles) == 1 else "?",
                "end": end,
                "end_symbol": symbols[end] if end < len(symbols) else "?",
                "bt1": bt1,
                "bt2": bt2,
                "closure": int(cache_sub["path_closure"][j]),
            }
        )
    return out


def part_b(variant: str = "A0", seed: int = 0) -> dict[str, Any]:
    view = _view(C.MAIN_REPEAT)
    src = S.load_lipop(C.MAIN_REPEAT, labels="train_valid")
    rows_all = list(view.rows["valid"])
    y = view.y_valid
    _, model, ckpt = _model(seed, variant)
    tgt = ckpt["target_norm"]
    scale, mean = float(tgt["scale"]), float(tgt["mean"])
    with torch.no_grad():
        pred_all = D.predict_scores(model, view.valid_cache, torch.device("cpu")) * scale + mean
    err = np.abs(pred_all - y)
    order = sorted(range(len(err)), key=lambda i: (float(err[i]), int(rows_all[i])), reverse=True)
    ranks = [r for r in C.ATTRIB_RANKS if r < len(order)]
    chosen = [order[r] for r in ranks]
    out: dict[str, Any] = {
        "variant": variant,
        "seed": seed,
        "rank_rule": "122 Valid |error| descending, ties by CSV row id ascending",
        "ranks": ranks,
        "target_norm": dict(tgt),
        "checkpoint_sha256": C.sha256_file(C.run_dir(C.MAIN_REPEAT, variant, seed) / "checkpoint.pt"),
        "examples": [],
    }
    for rank, pos in zip(ranks, chosen):
        row_id = int(rows_all[pos])
        smiles = src["smiles"][row_id]
        base_cache = single_molecule_cache(view.valid_cache, pos)
        batch = D.to_device(D.PREP_DATA.collate(base_cache, [0]), torch.device("cpu"))
        with torch.no_grad():
            y0, a0 = model(batch, return_aux=True)
        pred0 = float(y0.item() * scale + mean)
        contrib_atom0 = a0["contrib_atom"].numpy() * scale
        contrib0 = float(a0["contrib"].item() * scale)
        rho0 = float(a0["rho"].item() * scale)
        symbols = _atom_symbols(smiles)
        if len(symbols) != contrib_atom0.size:
            raise SystemExit(f"row {row_id}: RDKit atoms {len(symbols)} != cache atoms {contrib_atom0.size}")
        labels = _path_labels(base_cache, np.asarray(base_cache["path_ptr"]))
        example: dict[str, Any] = {
            "rank": rank,
            "valid_position": pos,
            "row_id": row_id,
            "smiles": smiles,
            "n_atoms": len(symbols),
            "y_true": float(y[pos]),
            "prediction": pred0,
            "abs_error": float(abs(pred0 - y[pos])),
            "target_mean": mean,
            "contrib_raw": contrib0,
            "rho_raw": rho0,
            "identity_residual": float(mean + contrib0 + rho0 - pred0),
            "atoms": [
                {"atom_index": i, "element": symbols[i], "f_h": float(contrib_atom0[i] / scale), "c_raw": float(contrib_atom0[i])}
                for i in range(len(symbols))
            ],
            "paths": [],
        }
        n_paths = int(np.asarray(base_cache["path_cat"]).size)
        p0 = int(np.asarray(view.valid_cache["path_ptr"])[int(np.asarray(view.valid_cache["atom_ptr"])[pos])])
        for j in range(n_paths):
            sub = single_molecule_cache(view.valid_cache, pos, drop_path=p0 + j)
            b1 = D.to_device(D.PREP_DATA.collate(sub, [0]), torch.device("cpu"))
            if int(b1["path_cat"].numel()) != n_paths - 1:
                raise SystemExit("path drop did not remove exactly one record")
            with torch.no_grad():
                y1, a1 = model(b1, return_aux=True)
            pred1 = float(y1.item() * scale + mean)
            d_contrib_atom = (a1["contrib_atom"].numpy() - a0["contrib_atom"].numpy()) * scale
            d_contrib = float(a1["contrib"].item() * scale) - contrib0
            d_rho = float(a1["rho"].item() * scale) - rho0
            dh = (a1["h"] - a0["h"]).abs()
            rec = labels[j]
            root = int(rec["root"])
            d_h_root = float(dh[root].max().item())
            d_h_other = float(np.delete(dh.numpy(), root, axis=0).max()) if dh.shape[0] > 1 else 0.0
            other = np.delete(d_contrib_atom, root)
            example["paths"].append(
                {
                    **{
                        k: rec[k]
                        for k in (
                            "path_index",
                            "root",
                            "root_symbol",
                            "middle_candidates",
                            "middle_symbol",
                            "end",
                            "end_symbol",
                            "bt1",
                            "bt2",
                            "closure",
                        )
                    },
                    "pred_before": pred0,
                    "pred_after": pred1,
                    "rho_before": rho0,
                    "rho_after": float(a1["rho"].item() * scale),
                    "f_root_before": float(contrib_atom0[root]),
                    "f_root_after": float(contrib_atom0[root] + d_contrib_atom[root]),
                    "h_root_l2_before": float(a0["h"][root].norm().item()),
                    "h_root_l2_after": float(a1["h"][root].norm().item()),
                    "delta_pred": pred1 - pred0,
                    "delta_contrib_root": float(d_contrib_atom[root]),
                    "delta_contrib_sum": d_contrib,
                    "delta_rho": d_rho,
                    "delta_root_h_max": d_h_root,
                    "other_atoms_h_max_abs_delta": d_h_other,
                    "other_atoms_f_max_abs_delta": float(np.abs(other).max()) if other.size else 0.0,
                    "identity_gap": float((pred1 - pred0) - (d_contrib + d_rho)),
                }
            )
        example["paths"].sort(key=lambda r: -abs(r["delta_pred"]))
        out["examples"].append(example)
    return out


# ---------------------------------------------------------------------------
# C. per-atom term tables for the fixed display rows (all seeds + COMPACT)
# ---------------------------------------------------------------------------


def display_rows() -> tuple[list[int], list[int], list[int]]:
    """(valid positions, ranks, csv row ids) for the frozen 8 display cases."""
    view = _view(C.MAIN_REPEAT)
    _, model, ckpt = _model(0, "A0")
    tgt = ckpt["target_norm"]
    pred = D.predict_scores(model, view.valid_cache, torch.device("cpu")) * float(tgt["scale"]) + float(tgt["mean"])
    err = np.abs(pred - view.y_valid)
    rows = list(view.rows["valid"])
    order = sorted(range(len(err)), key=lambda i: (float(err[i]), int(rows[i])), reverse=True)
    ranks = [r for r in C.ATTRIB_RANKS if r < len(order)]
    pos = [order[r] for r in ranks]
    return pos, ranks, [int(rows[p]) for p in pos]


def part_c(variants: Sequence[str] = ("A0",), seeds: Sequence[int] = C.SEEDS) -> list[dict[str, Any]]:
    view = _view(C.MAIN_REPEAT)
    src = S.load_lipop(C.MAIN_REPEAT, labels="train_valid")
    pos, ranks, row_ids = display_rows()
    out: list[dict[str, Any]] = []
    for variant in variants:
        has_local = bool(C.VARIANT_DEFS[variant]["use_local"])
        has_f = bool(C.VARIANT_DEFS[variant]["use_f"])
        if not has_local:
            for seed in seeds:
                for rank, p, row_id in zip(ranks, pos, row_ids):
                    out.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "rank": rank,
                            "row_id": row_id,
                            "smiles": src["smiles"][row_id],
                            "y_true": float(view.y_valid[p]),
                            "prediction": None,
                            "target_mean": None,
                            "L_raw": 0.0 if not has_f else None,
                            "rho_raw": None,
                            "atom_index": None,
                            "element": None,
                            "f_h": None,
                            "c_raw": 0.0,
                            "note": "no local encoder: no atom contributions exist",
                        }
                    )
            continue
        for seed in seeds:
            _, model, ckpt = _model(seed, variant)
            tgt = ckpt["target_norm"]
            scale, mean = float(tgt["scale"]), float(tgt["mean"])
            cache = single_molecule_cache(view.valid_cache, 0)
            del cache
            for rank, p, row_id in zip(ranks, pos, row_ids):
                sub = single_molecule_cache(view.valid_cache, p)
                batch = D.to_device(D.PREP_DATA.collate(sub, [0]), torch.device("cpu"))
                with torch.no_grad():
                    y, aux = model(batch, return_aux=True)
                pred = float(y.item() * scale + mean)
                symbols = _atom_symbols(src["smiles"][row_id])
                contrib = aux["contrib_atom"]
                c_raw = (contrib.numpy() * scale) if contrib is not None else np.zeros(len(symbols))
                out.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "rank": rank,
                        "row_id": row_id,
                        "smiles": src["smiles"][row_id],
                        "y_true": float(view.y_valid[p]),
                        "prediction": pred,
                        "target_mean": mean,
                        "L_raw": float(aux["contrib"].item() * scale),
                        "rho_raw": float(aux["rho"].item() * scale),
                        "atom_index": None,
                        "element": None,
                        "f_h": None,
                        "c_raw": None,
                        "note": "summary row",
                    }
                )
                for i, sym in enumerate(symbols):
                    out.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "rank": rank,
                            "row_id": row_id,
                            "smiles": src["smiles"][row_id],
                            "y_true": float(view.y_valid[p]),
                            "prediction": pred,
                            "target_mean": mean,
                            "L_raw": float(aux["contrib"].item() * scale),
                            "rho_raw": float(aux["rho"].item() * scale),
                            "atom_index": i,
                            "element": sym,
                            "f_h": float(contrib[i].item()) if contrib is not None else 0.0,
                            "c_raw": float(c_raw[i]),
                            "note": "atom row",
                        }
                    )
    return out


# ---------------------------------------------------------------------------
# figures + markdown
# ---------------------------------------------------------------------------


def figures(paths: Mapping[str, Any]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    written = []
    for ex in paths["examples"]:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1.0]})
        atoms = ex["atoms"]
        xs = np.arange(len(atoms))
        vals = np.asarray([a["c_raw"] for a in atoms])
        colors = ["#3b6ea5" if a["element"] == "C" else "#b5523b" if a["element"] == "O" else "#4b8b5a" if a["element"] == "N" else "#8a6fb0" for a in atoms]
        axes[0].bar(xs, vals, color=colors, width=0.8)
        axes[0].axhline(0, color="black", lw=0.6)
        axes[0].set_xticks(xs[:: max(1, len(xs) // 18)])
        axes[0].set_xticklabels([f"{i}{atoms[i]['element']}" for i in xs[:: max(1, len(xs) // 18)]], rotation=90, fontsize=7)
        axes[0].set_title(
            f"row {ex['row_id']} rank {ex['rank']}: atom terms c_v (raw logD)\n"
            f"pred={ex['prediction']:.3f}  y={ex['y_true']:.3f}  rho={ex['rho_raw']:.3f}  target_mean={ex['target_mean']:.3f}"
        )
        axes[0].set_ylabel("c_v (raw)")
        top = ex["paths"][:15]
        labels = [f"r{p['root']}{p['root_symbol']} bt{p['bt1']}={p['bt2']} end{p['end']}{p['end_symbol']}" for p in top]
        dv = [p["delta_pred"] for p in top]
        axes[1].barh(np.arange(len(top)), dv, color="#5b5b5b")
        axes[1].set_yticks(np.arange(len(top)))
        axes[1].set_yticklabels(labels, fontsize=7)
        axes[1].invert_yaxis()
        axes[1].axvline(0, color="black", lw=0.6)
        axes[1].set_xlabel("leave-one-S2-occurrence-out delta prediction (raw logD)")
        axes[1].set_title("top path interventions (model input channel)")
        fig.tight_layout()
        p = FIG / f"example_rank{ex['rank']:03d}_row{ex['row_id']}.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        written.append(str(p.relative_to(ATTR)))
    return written


def markdown(paths: Mapping[str, Any], interventions: Mapping[str, Any], fig_paths: Sequence[str]) -> str:
    lines = ["# 回溯表（模型内部预测回溯，非化学因果）", ""]
    lines += [
        "## A. 固定权重干预（122 Valid 前 64 行）",
        "",
        "| 干预 | mean abs Δpred | max abs Δpred | ΔRMSE (per seed) | 覆盖行比例 |",
        "|---|---|---|---|---|",
    ]
    for name, item in interventions["aggregate"].items():
        lines.append(
            f"| `{name}` | {item['mean_abs_delta_mean']:.4f} | {item['max_abs_delta_max']:.4f} | "
            f"{item['delta_rmse_mean']:+.4f} {item['delta_rmse_per_seed']} | {item['coverage_rows_mean']:.3f} |"
        )
    lines += [
        "",
        "每个 seed 的 baseline：contrib 与 rho 的均值/标准差/相关系数（原始单位，前 64 行）。",
        "",
        "| seed | contrib mean | contrib sd | rho mean | rho sd | corr(contrib, rho) |",
        "|---|---|---|---|---|---|",
    ]
    for seed in sorted(interventions["seeds"]):
        b = interventions["seeds"][seed]["baseline"]
        lines.append(
            f"| {seed} | {b['contrib_mean']:+.4f} | {b['contrib_sd']:.4f} | {b['rho_mean']:+.4f} | "
            f"{b['rho_sd']:.4f} | {b['corr_contrib_rho']:+.4f} |"
        )
    lines += [
        "",
        "干预前后的输入/激活范数（seed 平均；L2 per row）：",
        "",
        "| 干预 | ‖g‖ before→after | ‖L‖ | ‖mean_h2‖ | ‖path_agg‖ | ‖content_agg‖ |",
        "|---|---|---|---|---|---|",
    ]
    for name, _, _ in INTERVENTIONS:
        def pair(key):
            before = np.mean([interventions["seeds"][str(s)]["baseline"][key] for s in C.SEEDS])
            after = np.mean([interventions["seeds"][str(s)]["interventions"][name]["norms_after"][key] for s in C.SEEDS])
            return f"{before:.3f}→{after:.3f}"
        lines.append(
            f"| `{name}` | {pair('norm_g')} | {pair('norm_contrib')} | {pair('norm_mean_h2')} | "
            f"{pair('norm_path_agg')} | {pair('norm_content_agg')} |"
        )
    lines += ["", "## B. Leave-one-S2-occurrence-out（FULL seed0）", ""]
    for ex, fig in zip(paths["examples"], fig_paths):
        lines += [
            f"### rank {ex['rank']} · row {ex['row_id']} · y={ex['y_true']:.3f} pred={ex['prediction']:.3f}",
            "",
            f"`{ex['smiles']}`",
            "",
            f"target_mean={ex['target_mean']:.4f}, L={ex['contrib_raw']:+.4f}, rho={ex['rho_raw']:+.4f}, "
            f"identity residual {ex['identity_residual']:+.2e}",
            "",
            f"![rank {ex['rank']}]({fig})",
            "",
            "| path | root→mid→end | bt1→bt2 | closure | pred before→after | f_root before→after | ΔL | Δrho | Δpred | max Δ other f | identity gap |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for p in ex["paths"][:12]:
            mid = ",".join(str(m) for m in p["middle_candidates"]) or "?"
            lines.append(
                f"| {p['path_index']} | {p['root']}{p['root_symbol']}→{mid}→{p['end']}{p['end_symbol']} | "
                f"{p['bt1']}→{p['bt2']} | {p['closure']} | {p['pred_before']:+.3f}→{p['pred_after']:+.3f} | "
                f"{p['f_root_before']:+.3f}→{p['f_root_after']:+.3f} | "
                f"{p['delta_contrib_sum']:+.4f} | {p['delta_rho']:+.4f} | {p['delta_pred']:+.4f} | "
                f"{p['other_atoms_f_max_abs_delta']:.2e} | {p['identity_gap']:+.2e} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--variant", default="A0")
    ap.add_argument("--compact", default=None, help="optional COMPACT variant for the atom tables")
    ap.add_argument(
        "--out-tag",
        default=None,
        help="write path LOO to path_loo_<tag>.json only (no figures, no TABLES.md)",
    )
    args = ap.parse_args(argv)
    ATTR.mkdir(parents=True, exist_ok=True)
    variants = ["A0"] + ([args.compact] if args.compact and args.compact != "A0" else [])
    c = part_c(variants)
    import csv as _csv

    (ATTR / "atom_terms.csv").parent.mkdir(parents=True, exist_ok=True)
    with (ATTR / "atom_terms.csv").open("w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=list(c[0].keys()))
        writer.writeheader()
        for row in c:
            writer.writerow(row)
    C.log(f"[attrib] atom_terms.csv written ({len(c)} rows)")

    ia = None
    if not args.skip_a:
        ia = part_a()
        C.write_json(ATTR / "interventions.json", ia)
        C.log("[attrib] interventions.json written")
    paths = None
    if not args.skip_b and args.out_tag:
        paths = part_b(variant=args.variant)
        C.write_json(ATTR / f"path_loo_{args.out_tag}.json", paths)
        C.log(f"[attrib] path_loo_{args.out_tag}.json written ({args.variant})")
    elif not args.skip_b:
        paths = part_b(variant=args.variant)
        C.write_json(ATTR / "path_loo.json", paths)
        figs = figures(paths)
        ia = ia or C.read_json(ATTR / "interventions.json")
        (ATTR / "TABLES.md").write_text(markdown(paths, ia, figs))
        C.log(f"[attrib] path_loo.json + {len(figs)} figures + TABLES.md written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
