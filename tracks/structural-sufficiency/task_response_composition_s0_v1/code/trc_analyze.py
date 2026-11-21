"""Core analysis: whole-molecule table, factorial, response metrics, consistency,
cross-combinations and route judgements.

Only observed molecules are used; no pseudo-labels are fabricated for unobserved
`(context, word)` combinations.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

import trc_common as C
import trc_pairs as PR
import trc_prepare as P

CELL_NAMES = {0: "train", 1: "valid", 2: "test", 3: "cross"}


# ---------------------------------------------------------------------------
# run loading
# ---------------------------------------------------------------------------


def run_ids() -> list[str]:
    root = C.results_root()
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or "invalid" in path.name:
            continue
        if (path / "manifest.json").is_file() or (path / "curves.csv").is_file():
            out.append(path.name)
    return out


def load_run(run_id: str) -> dict[str, Any] | None:
    root = C.results_root() / run_id
    if not (root / "curves.csv").is_file():
        return None
    import csv

    with (root / "curves.csv").open(encoding="utf-8") as handle:
        curves = list(csv.DictReader(handle))
    for row in curves:
        for key, value in list(row.items()):
            if key in ("run_id",):
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    manifest = C.read_json(root / "manifest.json") if (root / "manifest.json").is_file() else {}
    return {"run_id": run_id, "dir": str(root), "curves": curves, "manifest": manifest}


def best_row(run: Mapping[str, Any]) -> dict[str, Any]:
    rows = run["curves"]
    return min(rows, key=lambda r: r["valid_rmse"])


def final_row(run: Mapping[str, Any]) -> dict[str, Any]:
    return run["curves"][-1]


# ---------------------------------------------------------------------------
# paired bootstrap
# ---------------------------------------------------------------------------


def paired_bootstrap(pred_a: np.ndarray, pred_b: np.ndarray, labels: np.ndarray,
                     n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """RMSE(control) - RMSE(candidate) on the same resampled molecules."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    if n == 0:
        return {"delta": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    base = C.rmse(pred_a, labels) - C.rmse(pred_b, labels)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = C.rmse(pred_a[idx], labels[idx]) - C.rmse(pred_b[idx], labels[idx])
    return {"delta": float(base), "lo": float(np.percentile(stats, 2.5)),
            "hi": float(np.percentile(stats, 97.5)), "n": int(n)}


def predictions(run_id: str, kind: str = "best") -> dict[str, np.ndarray] | None:
    path = C.results_root() / run_id / f"predictions_{kind}.npz"
    if not path.is_file():
        return None
    with np.load(path) as handle:
        return {k: handle[k] for k in handle.files}


# ---------------------------------------------------------------------------
# main table
# ---------------------------------------------------------------------------


def main_table(runs: Mapping[str, dict]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id, run in runs.items():
        man = run["manifest"]
        b = best_row(run)
        f = final_row(run)
        rows.append({
            "run_id": run_id,
            "arm": man.get("arm"), "supervision": man.get("supervision"),
            "dataset": man.get("dataset"), "train_fraction": man.get("train_fraction"),
            "epochs": man.get("epochs"), "epochs_planned": man.get("epochs_planned"),
            "active_params": (man.get("params") or {}).get("trainable"),
            "best_epoch": int(b["epoch"]),
            "best_train_rmse": b["train_rmse"], "best_valid_rmse": b["valid_rmse"],
            "best_test_rmse": b["test_rmse"],
            "best_train_mae": b["train_mae"], "best_valid_mae": b["valid_mae"],
            "best_test_mae": b["test_mae"],
            "final_train_rmse": f["train_rmse"], "final_valid_rmse": f["valid_rmse"],
            "final_test_rmse": f["test_rmse"],
            "wall_minutes": (man.get("wall_seconds") or 0.0) / 60.0,
            "gpu_hours": man.get("gpu_hours"),
            "peak_vram_gb": (man.get("max_vram_bytes") or 0) / 1e9,
            "erase_distance": man.get("erase_distance"), "lambda_var": man.get("lambda_var"),
            "completed": bool(man.get("stopped_reason") != "cancelled"),
        })
    return rows


# ---------------------------------------------------------------------------
# structure freeze / support
# ---------------------------------------------------------------------------


def structure_freeze(pack, cfg) -> dict[str, Any]:
    n_mol = int(pack["n_molecules"])
    parse_mol = np.asarray(pack["parse_mol"])
    fallback = set(np.asarray(pack["fallback_rows"]).tolist())
    out: dict[str, Any] = {
        "cache_path": pack.get("cache_path"), "cache_sha256": pack.get("cache_sha256"),
        "commit": C.git_commit(),
        "n_molecules": n_mol, "n_parses": int(pack["n_parses"]),
        "n_fragments": int(pack["n_fragments"]), "n_frames": int(pack["n_frames"]),
        "n_atom_attribute_rows": int(np.asarray(pack["atom_table"]).shape[0]),
        "n_bond_attribute_rows": int(np.asarray(pack["bond_table"]).shape[0]),
        "fallback_molecules": sorted(fallback),
        "config": {k: cfg[k] for k in ("cuts", "dims", "heads", "recipe")},
        "timing": json.loads(str(pack["timing_json"])),
        "splits": {},
    }
    for split in C.SPLITS:
        rows = np.asarray(pack[f"{split}_idx"], np.int64)
        sel = np.isin(parse_mol, rows)
        fb = sorted(set(rows.tolist()) & fallback)
        out["splits"][split] = {
            "n_molecules": int(rows.size),
            "n_parses": int(sel.sum()),
            "parses_per_molecule_mean": float(sel.sum() / max(1, rows.size)),
            "fallback_molecules": fb,
            "n_parses_at_fallback_molecules": int(np.isin(parse_mol, fb).sum()) if fb else 0,
        }
    counts = np.asarray(pack["frag_counts"])
    out["distance_bucket_totals"] = counts.sum(0).tolist()
    out["fragment_atoms_hist"] = np.bincount(np.asarray(pack["frag_natoms"]), minlength=12)[:12].tolist()
    out["fragment_components_hist"] = np.bincount(np.asarray(pack["frag_ncomp"]), minlength=4)[:4].tolist()
    return out


def support_report(pack, cfg, labels) -> dict[str, Any]:
    cell = np.asarray(pack["pair_cell"])
    comp = np.asarray(pack["pair_component"])
    transform = np.asarray(pack["pair_transform"])
    pa = np.asarray(pack["pair_mol_a"])
    pb = np.asarray(pack["pair_mol_b"])
    out: dict[str, Any] = {"cells": {}, "n_pairs_total": int(cell.size)}
    train = np.asarray(pack["train_idx"], np.int64)
    store = PR.PairStore(cfg, pack, labels, None, train, np.asarray(pack["row_ids"]),
                         log=lambda *_: None)
    out["train_directory"] = store.support()
    train_t = set(np.asarray(store.transform).tolist())
    for c, name in CELL_NAMES.items():
        sel = np.nonzero(cell == c)[0]
        tt = set(transform[sel].tolist())
        mols = set(pa[sel].tolist()) | set(pb[sel].tolist())
        comps = set(comp[sel].tolist())
        out["cells"][name] = {
            "n_pairs": int(sel.size),
            "n_transforms": len(tt),
            "n_transforms_supported_by_train": len(tt & train_t),
            "n_molecules": len(mols),
            "n_components_in_train_graph": len([x for x in comps if x >= 0]),
            "n_pairs_with_train_support": int(np.sum([t in train_t for t in transform[sel].tolist()])),
        }
    if cell.size:
        dy = np.asarray(labels, float)[pb] - np.asarray(labels, float)[pa]
        out["label_delta_abs_mean"] = float(np.mean(np.abs(dy)))
        dup = defaultdict(list)
        for i, (c, t) in enumerate(zip(cell.tolist(), transform.tolist())):
            dup[(c, t)].append(i)
        multi = [v for v in dup.values() if len(v) > 1]
        out["cell_transform_units"] = len(dup)
        out["cell_transform_units_with_multiple_pairs"] = len(multi)
        out["max_pairs_per_cell_transform"] = int(max((len(v) for v in dup.values()), default=0))
        out["repeat_unit_label_delta_abs_mean"] = float(np.mean(
            [np.mean([abs(dy[i]) for i in v]) for v in multi])) if multi else 0.0
    return out


# ---------------------------------------------------------------------------
# response metrics on the real directory
# ---------------------------------------------------------------------------


def _train_transform_table(store: PR.PairStore):
    """transform -> component -> mean standardized delta on the train pairs."""
    table: dict[int, dict[int, float]] = defaultdict(dict)
    for t, groups in store.transform_groups.items():
        for c, idxs in groups.items():
            table[int(t)][int(c)] = float(np.mean(store.delta_std[idxs]))
    return table


def _model_deltas(model, items: Mapping[str, np.ndarray], kind: str,
                  chunk: int = 512) -> np.ndarray:
    import torch

    out = np.zeros(items["C"].size, np.float64)
    model.eval()
    with torch.no_grad():
        phi = model.frame_phi() if model.arm != "GLOBAL" else None
        for start in range(0, items["C"].size, chunk):
            sl = slice(start, start + chunk)
            if kind == "cut":
                pairs = {"C": torch.as_tensor(items["C"][sl], device=model.device),
                         "Sa": torch.as_tensor(items["Sa"][sl], device=model.device),
                         "Sb": torch.as_tensor(items["Sb"][sl], device=model.device),
                         "bond": torch.as_tensor(items["bond"][sl], device=model.device)}
                out[sl] = model.pair_cut_delta(pairs, phi).double().cpu().numpy()
            else:
                n = int(items["a"][sl].size)
                rows = np.concatenate([items["a"][sl], items["b"][sl]])
                idx = torch.as_tensor(rows.astype(np.int64), device=model.device)
                fb = model.molecule_forward(idx)["f_bar"]
                out[sl] = (fb[n:] - fb[:n]).double().cpu().numpy()
    return out


def response_metrics(pack, cfg, labels, models: Mapping[str, Any]):
    """Returns (report, per-model delta arrays)."""
    cell = np.asarray(pack["pair_cell"])
    transform = np.asarray(pack["pair_transform"])
    comp = np.asarray(pack["pair_component"])
    pa = np.asarray(pack["pair_mol_a"], np.int64)
    pb = np.asarray(pack["pair_mol_b"], np.int64)
    pc = np.asarray(pack["pair_C"], np.int64)
    sa = np.asarray(pack["pair_Sa"], np.int64)
    sb = np.asarray(pack["pair_Sb"], np.int64)
    bond = np.asarray(pack["pair_cut_bond_row"], np.int64)
    train = np.asarray(pack["train_idx"], np.int64)
    store = PR.PairStore(cfg, pack, labels, None, train, np.asarray(pack["row_ids"]),
                         log=lambda *_: None)
    table = _train_transform_table(store)
    mol_comp = np.asarray(pack["molecule_component"], np.int64)
    y = np.asarray(labels, float)

    dy = (y[pb] - y[pa]) / store.sigma
    items = {"a": pa, "b": pb, "C": pc, "Sa": sa, "Sb": sb, "bond": bond}
    out: dict[str, Any] = {
        "sigma_y": store.sigma, "tau2": store.tau2,
        "n_pairs_total": int(cell.size),
    }
    out["models"] = {}
    for run_id, model in models.items():
        entry = {"arm": model.arm}
        entry["delta_mol"] = _model_deltas(model, items, "mol")
        if model.arm != "GLOBAL":
            entry["delta_cut"] = _model_deltas(model, items, "cut")
        out["models"][run_id] = entry

    # TABLE reference and zero-delta baseline
    table_hat = np.full(cell.size, np.nan)
    table_strict = np.full(cell.size, np.nan)
    supported = np.zeros(cell.size, bool)
    strict_supported = np.zeros(cell.size, bool)
    for i in range(cell.size):
        comps = table.get(int(transform[i]))
        if not comps:
            continue
        supported[i] = True
        table_hat[i] = float(np.mean(list(comps.values())))
        drop = {int(mol_comp[pa[i]]), int(mol_comp[pb[i]])}
        kept = [v for c, v in comps.items() if int(c) not in drop]
        if kept:
            strict_supported[i] = True
            table_strict[i] = float(np.mean(kept))
    out["support"] = {
        "n_supported": int(supported.sum()),
        "n_strict_supported": int(strict_supported.sum()),
        "coverage_total": float(supported.mean()) if cell.size else 0.0,
        "coverage_strict": float(strict_supported.mean()) if cell.size else 0.0,
    }
    # repeat-transform subdirectory (transform seen in >= 2 train components)
    repeat_t = set()
    for t, groups in store.transform_groups.items():
        if len(groups) >= 2:
            repeat_t.add(int(t))
    repeat = np.isin(transform, list(repeat_t)) if repeat_t else np.zeros(cell.size, bool)
    out["repeat_subset"] = {"n_pairs": int(repeat.sum())}

    def score(pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        if mask.sum() == 0:
            return {"n": 0}
        p = pred[mask]
        t = dy[mask]
        ok = np.isfinite(p)
        if ok.sum() == 0:
            return {"n": 0}
        p, t = p[ok], t[ok]
        return {
            "n": int(p.size),
            "mae": float(np.mean(np.abs(p - t))),
            "rmse": float(math.sqrt(np.mean((p - t) ** 2))),
            "sign_accuracy": float(np.mean(np.sign(p) == np.sign(t))),
            "corr": float(np.corrcoef(p, t)[0, 1]) if p.size > 2 and np.std(p) > 0 else 0.0,
        }

    masks = {
        "all": np.ones(cell.size, bool),
        "unsupported": ~supported,
        "train_supported": supported,
        "train_supported_strict": strict_supported,
        "repeat_transform": repeat,
    }
    for name, mask in masks.items():
        out.setdefault("table", {})[name] = score(table_hat, mask)
        out.setdefault("zero_delta", {})[name] = score(np.zeros(cell.size), mask)
        if name.endswith("strict"):
            out.setdefault("table_strict", {})[name] = score(
                np.where(np.isfinite(table_strict), table_strict, np.nan), mask)
        for run_id, entry in out["models"].items():
            entry.setdefault("metrics", {})[name] = {
                "delta_mol": score(entry["delta_mol"], mask),
            }
            if "delta_cut" in entry:
                entry["metrics"][name]["delta_cut"] = score(entry["delta_cut"], mask)
    # per-cell metrics
    out["cells"] = {}
    for c, name in CELL_NAMES.items():
        sel = cell == c
        if sel.sum() == 0:
            continue
        block = {
            "n_pairs": int(sel.sum()),
            "table": score(table_hat, sel),
            "zero_delta": score(np.zeros(cell.size), sel),
            "models": {rid: {"delta_mol": score(e["delta_mol"], sel),
                             **({"delta_cut": score(e["delta_cut"], sel)}
                                if "delta_cut" in e else {})}
                       for rid, e in out["models"].items()},
        }
        out["cells"][name] = block

    # component-grouped bootstrap of the cut/molecule delta MAE for the best pair
    out["bootstrap"] = {}
    for run_id, entry in out["models"].items():
        for kind in ("delta_mol", "delta_cut"):
            if kind not in entry:
                continue
            groups = defaultdict(list)
            for i in np.nonzero(supported)[0].tolist():
                groups[int(comp[i])].append(i)
            gkeys = [g for g in groups if g >= 0]
            if len(gkeys) < 10:
                out["bootstrap"][f"{run_id}:{kind}"] = {"n_groups": len(gkeys), "note": "too few groups"}
                continue
            rng = np.random.default_rng(0)
            vals = []
            for _ in range(500):
                pick = rng.choice(len(gkeys), len(gkeys), replace=True)
                idx = np.concatenate([groups[gkeys[k]] for k in pick])
                p = entry[kind][idx]
                t = dy[idx]
                ok = np.isfinite(p)
                vals.append(float(np.mean(np.abs(p[ok] - t[ok]))))
            out["bootstrap"][f"{run_id}:{kind}"] = {
                "n_groups": len(gkeys), "mae": float(np.mean(vals)),
                "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
            }
    deltas = {}
    for rid, entry in out["models"].items():
        deltas[rid] = {"arm": entry["arm"], "delta_mol": entry.pop("delta_mol")}
        if "delta_cut" in entry:
            deltas[rid]["delta_cut"] = entry.pop("delta_cut")
    return out, deltas


# ---------------------------------------------------------------------------
# cross-combination double differences
# ---------------------------------------------------------------------------


def cross_combinations(pack, cfg, labels, models: Mapping[str, Any],
                       max_groups: int = 2000) -> dict[str, Any]:
    cell = np.asarray(pack["pair_cell"])
    pa = np.asarray(pack["pair_mol_a"], np.int64)
    pb = np.asarray(pack["pair_mol_b"], np.int64)
    pc = np.asarray(pack["pair_C"], np.int64)
    sa = np.asarray(pack["pair_Sa"], np.int64)
    sb = np.asarray(pack["pair_Sb"], np.int64)
    bond = np.asarray(pack["pair_cut_bond_row"], np.int64)
    y = np.asarray(labels, float)
    by_words: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, (a_, b_, bd) in enumerate(zip(sa.tolist(), sb.tolist(), bond.tolist())):
        by_words[(a_, b_, bd)].append(i)
    quads: list[tuple[int, int]] = []
    for key, idxs in by_words.items():
        by_ctx: dict[int, list[int]] = defaultdict(list)
        for i in idxs:
            by_ctx[int(pc[i])].append(i)
        ctxs = sorted(by_ctx)
        for c1, c2 in itertools.combinations(ctxs, 2):
            for i in by_ctx[c1]:
                for j in by_ctx[c2]:
                    if len({int(pa[i]), int(pb[i]), int(pa[j]), int(pb[j])}) < 4:
                        continue
                    quads.append((i, j))
    total_quads = len(quads)
    if len(quads) > max_groups:
        order = np.argsort([hashlib.sha256(f"trc-quad:{i}:{j}".encode()).hexdigest()
                            for i, j in quads])[:max_groups]
        quads = [quads[k] for k in order]
    out: dict[str, Any] = {"n_quads_available": total_quads, "n_quads_used": len(quads)}
    if not quads:
        out["ok"] = False
        return out
    i1 = np.asarray([q[0] for q in quads])
    i2 = np.asarray([q[1] for q in quads])
    D_y = ((y[pb[i1]] - y[pa[i1]]) - (y[pb[i2]] - y[pa[i2]]))
    c1 = cell[i1]
    c2 = cell[i2]
    eval_sets = {
        "train_valid_any_valid": (np.isin(c1, (0, 1)) & np.isin(c2, (0, 1))
                                  & ((c1 == 1) | (c2 == 1))),
        "train_test_any_test": (np.isin(c1, (0, 2)) & np.isin(c2, (0, 2))
                                & ((c1 == 2) | (c2 == 2))),
        "all_held_out": (np.isin(c1, (1, 2)) & np.isin(c2, (1, 2))),
    }
    out["sets"] = {k: int(v.sum()) for k, v in eval_sets.items()}
    out["n_unique_molecules"] = int(len({int(x) for x in np.concatenate(
        [pa[i1], pb[i1], pa[i2], pb[i2]]).tolist()}))
    out["models"] = {}
    for run_id, entry in models.items():
        res = {}
        dm = entry["delta_mol"]
        dc = entry.get("delta_cut")
        for name, mask in eval_sets.items():
            if mask.sum() == 0:
                res[name] = {"n": 0}
                continue
            block = {"n": int(mask.sum())}
            pred_mol = dm[i1] - dm[i2]
            block["D_mol"] = _dd_score(pred_mol, D_y, mask)
            if dc is not None:
                pred_cut = dc[i1] - dc[i2]
                block["D_cut"] = _dd_score(pred_cut, D_y, mask)
            res[name] = block
        out["models"][run_id] = res
    return out


def _dd_score(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    p = pred[mask]
    t = target[mask]
    ok = np.isfinite(p) & np.isfinite(t)
    if ok.sum() == 0:
        return {"n": 0}
    p, t = p[ok], t[ok]
    return {"n": int(p.size), "mae": float(np.mean(np.abs(p - t))),
            "rmse": float(math.sqrt(np.mean((p - t) ** 2))),
            "sign_accuracy": float(np.mean(np.sign(p) == np.sign(t)))}


# ---------------------------------------------------------------------------
# factorial
# ---------------------------------------------------------------------------


def factorial(runs: Mapping[str, dict]) -> dict[str, Any]:
    by_key: dict[tuple[str, str], str] = {}
    for run_id, run in runs.items():
        man = run["manifest"]
        by_key[(man.get("arm"), man.get("supervision"))] = run_id
    out: dict[str, Any] = {"available": sorted(f"{a}/{s}" for a, s in by_key)}

    def val(run_id: str, split: str, kind: str = "best") -> float | None:
        if run_id is None or run_id not in runs:
            return None
        row = best_row(runs[run_id]) if kind == "best" else final_row(runs[run_id])
        return row.get(f"{split}_rmse")

    for split in ("valid", "test"):
        d_none = val(by_key.get(("DIRECT", "NONE")), split)
        d_resp = val(by_key.get(("DIRECT", "CUTRESP")), split)
        f_none = val(by_key.get(("FACTOR", "NONE")), split)
        f_resp = val(by_key.get(("FACTOR", "CUTRESP")), split)
        f_mol = val(by_key.get(("FACTOR", "MOLPAIR")), split)
        a_resp = val(by_key.get(("ADD", "CUTRESP")), split)
        g_none = val(by_key.get(("GLOBAL", "NONE")), split)
        block = {
            "DIRECT_NONE": d_none, "DIRECT_CUTRESP": d_resp,
            "FACTOR_NONE": f_none, "FACTOR_CUTRESP": f_resp,
            "FACTOR_MOLPAIR": f_mol, "ADD_CUTRESP": a_resp, "GLOBAL_NONE": g_none,
        }
        if None not in (d_none, f_none):
            block["gain_share_none"] = d_none - f_none
        if None not in (d_resp, f_resp):
            block["gain_share_resp"] = d_resp - f_resp
        if None not in (f_none, f_resp):
            block["gain_resp_factor"] = f_none - f_resp
        if None not in (d_none, d_resp):
            block["gain_resp_direct"] = d_none - d_resp
        if "gain_resp_factor" in block and "gain_resp_direct" in block:
            block["interaction"] = block["gain_resp_factor"] - block["gain_resp_direct"]
        if None not in (f_resp, f_mol):
            block["cutresp_minus_molpair"] = f_mol - f_resp
        if None not in (f_resp, a_resp):
            block["factor_minus_add"] = a_resp - f_resp
        out[split] = block
    return out


def consistency(runs: Mapping[str, dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for run_id, run in runs.items():
        b = best_row(run)
        out[run_id] = {
            "epoch": int(b["epoch"]),
            "cut_var_mean": {s: b.get(f"{s}_cut_var_mean") for s in C.SPLITS},
            "cut_disp": {s: b.get(f"{s}_cut_disp") for s in C.SPLITS},
            "cut_rmse": {s: b.get(f"{s}_cut_rmse") for s in C.SPLITS},
            "cut_rmse_std": {s: b.get(f"{s}_cut_rmse_std") for s in C.SPLITS},
            "mol_rmse": {s: b.get(f"{s}_rmse") for s in C.SPLITS},
            "train_cut_minus_mol": (b.get("train_cut_rmse") - b.get("train_rmse"))
            if b.get("train_cut_rmse") is not None else None,
            "parses_mean": {s: b.get(f"{s}_parses_mean") for s in C.SPLITS},
        }
    return out


def split_gap(runs: Mapping[str, dict]) -> dict[str, Any]:
    """Diagnostic only: how much the reported best-Valid checkpoint costs on Test.

    Reported for every run so a Valid/Test reversal can be read directly; the
    protocol checkpoint stays the best-Valid one.
    """
    out: dict[str, Any] = {}
    for run_id, run in runs.items():
        rows = run["curves"]
        bv = min(rows, key=lambda r: r["valid_rmse"])
        bt = min(rows, key=lambda r: r["test_rmse"])
        stab = [r for r in reversed(rows) if r["epoch"] >= 0.75 * rows[-1]["epoch"]]
        out[run_id] = {
            "best_valid": {"epoch": int(bv["epoch"]), "valid": bv["valid_rmse"],
                           "test": bv["test_rmse"]},
            "best_test": {"epoch": int(bt["epoch"]), "valid": bt["valid_rmse"],
                          "test": bt["test_rmse"]},
            "valid_test_gap_at_best_valid": bv["test_rmse"] - bv["valid_rmse"],
            "cost_of_valid_selection": bt["test_rmse"] - bv["test_rmse"],
            "tail_median": {
                "valid": float(np.median([r["valid_rmse"] for r in stab])) if stab else None,
                "test": float(np.median([r["test_rmse"] for r in stab])) if stab else None,
            },
        }
    return out


def judgement(table: Mapping[str, Any], fact: Mapping[str, Any],
              resp: Mapping[str, Any], cross: Mapping[str, Any],
              esol: Mapping[str, Any]) -> dict[str, Any]:
    ref = {"GINE_RAW_REF": 0.6101478703, "WORD_PREFIX_valid": 0.6990918679,
           "BAG_CONT_valid": 0.6923730579}
    out: dict[str, Any] = {"reference_values": ref, "notes": []}
    lipo = {r["run_id"]: r for r in table.get("lipo", [])}
    if not lipo:
        out["status"] = "no lipo runs"
        return out
    best_run = min((r for r in lipo.values() if r.get("arm") in C.ARMS),
                   key=lambda r: r["best_valid_rmse"])
    out["best_lipo"] = {"run_id": best_run["run_id"],
                        "valid": best_run["best_valid_rmse"],
                        "test": best_run["best_test_rmse"]}
    refv = C.load_config()["reference_values"]
    gine = [r for r in table.get("reference", []) if r.get("arm") == "GINE_RAW"]
    if gine:
        lowest = min(gine, key=lambda r: r["best_valid_rmse"])
        out["gine_same_protocol"] = {"run_id": lowest["run_id"],
                                     "valid": lowest["best_valid_rmse"],
                                     "test": lowest["best_test_rmse"]}
        out["gap_to_gine_valid"] = best_run["best_valid_rmse"] - lowest["best_valid_rmse"]
        out["gap_to_gine_test"] = best_run["best_test_rmse"] - lowest["best_test_rmse"]
    out["gap_to_historic_gine_valid"] = (best_run["best_valid_rmse"]
                                         - refv["GINE_RAW_REF"]["valid_rmse"])
    out["gap_to_word_prefix_valid"] = best_run["best_valid_rmse"] - refv["WORD_PREFIX"]["valid_rmse"]
    out["gap_to_word_prefix_test"] = best_run["best_test_rmse"] - refv["WORD_PREFIX"]["test_rmse"]
    out["shared_value"] = {s: fact.get(s, {}).get("gain_share_none") for s in ("valid", "test")}
    out["response_specificity"] = {s: fact.get(s, {}).get("gain_resp_factor") for s in ("valid", "test")}
    out["interaction"] = {s: fact.get(s, {}).get("interaction") for s in ("valid", "test")}
    # resource stop condition
    factor_runs = [r for r in lipo.values() if r["arm"] == "FACTOR"]
    glob = [r for r in lipo.values() if r["arm"] == "GLOBAL"]
    direct = {(r["supervision"]): r for r in lipo.values() if r["arm"] == "DIRECT"}
    stop = False
    if factor_runs and glob:
        g_valid = min(r["best_valid_rmse"] for r in glob)
        stop = all(r["best_valid_rmse"] > 0.76 for r in factor_runs) and \
            all(r["best_valid_rmse"] - g_valid > 0.04 for r in factor_runs)
        for r in factor_runs:
            d = direct.get(r["supervision"])
            if d is not None and d["best_valid_rmse"] - r["best_valid_rmse"] >= 0.01:
                stop = False
    out["resource_stop_triggered"] = bool(stop)
    return out


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-models", action="store_true")
    args = ap.parse_args(argv)
    cfg = dict(C.load_config())
    cfg["cache_stem"] = "trc_cache"
    out_dir = C.analysis_dir()
    runs = {rid: load_run(rid) for rid in run_ids()}
    runs = {k: v for k, v in runs.items() if v is not None}
    missing = [rid for rid, r in runs.items() if not r["manifest"]]
    for rid in missing:
        runs.pop(rid, None)
    print(f"runs found: {sorted(runs)}")

    lipo_runs = {rid: r for rid, r in runs.items() if r["manifest"].get("dataset") == "lipo"
                 and r["manifest"].get("arm") in C.ARMS}
    esol_runs = {rid: r for rid, r in runs.items() if r["manifest"].get("dataset") == "esol"
                 and r["manifest"].get("arm") in C.ARMS}

    table_rows: dict[str, list] = {"lipo": [], "esol": [], "reference": []}
    for rid, run in runs.items():
        if rid in lipo_runs:
            key = "lipo"
        elif rid in esol_runs:
            key = "esol"
        else:
            key = "reference"
        table_rows[key].extend(main_table({rid: run}))
    all_rows = table_rows["lipo"] + table_rows["esol"] + table_rows["reference"]
    C.write_csv(out_dir / "main_table.csv", all_rows)
    C.write_json(out_dir / "main_table.json", table_rows)

    pack = P.load_pack(cfg, "lipo")
    labels = np.asarray(pack["labels"], float)
    C.write_json(out_dir / "structure_freeze.json", structure_freeze(pack, cfg))
    C.write_json(out_dir / "support.json", support_report(pack, cfg, labels))

    # bootstrap comparisons against the paired controls
    cmp_out: dict[str, Any] = {}
    preds = {rid: predictions(rid, "best") for rid in lipo_runs}
    pairs_to_compare = [
        ("LIPO_DIRECT_NONE_R0", "LIPO_FACTOR_NONE_R0"),
        ("LIPO_DIRECT_CUTRESP_R0", "LIPO_FACTOR_CUTRESP_R0"),
        ("LIPO_FACTOR_NONE_R0", "LIPO_FACTOR_CUTRESP_R0"),
        ("LIPO_DIRECT_NONE_R0", "LIPO_DIRECT_CUTRESP_R0"),
        ("LIPO_FACTOR_MOLPAIR_R0", "LIPO_FACTOR_CUTRESP_R0"),
        ("LIPO_ADD_CUTRESP_R0", "LIPO_FACTOR_CUTRESP_R0"),
        ("LIPO_GLOBAL_NONE_R0", "LIPO_FACTOR_NONE_R0"),
    ]
    for control, cand in pairs_to_compare:
        if control in preds and cand in preds and preds[control] is not None and preds[cand] is not None:
            block = {}
            for split in C.SPLITS:
                block[split] = paired_bootstrap(
                    preds[control][f"{split}_pred"], preds[cand][f"{split}_pred"],
                    preds[control][f"{split}_label"])
            cmp_out[f"{control} - {cand}"] = block
    C.write_json(out_dir / "bootstrap.json", cmp_out)

    fact = factorial(lipo_runs)
    C.write_json(out_dir / "factorial.json", fact)
    cons = consistency(lipo_runs)
    C.write_json(out_dir / "consistency.json", cons)
    C.write_json(out_dir / "split_gap.json", split_gap(runs))

    resp: dict[str, Any] = {}
    cross: dict[str, Any] = {}
    if not args.skip_models:
        models = load_models(lipo_runs, cfg, pack, args.device)
        resp, deltas = response_metrics(pack, cfg, labels, models)
        C.write_json(out_dir / "response_metrics.json", resp)
        cross = cross_combinations(pack, cfg, labels, deltas)
        C.write_json(out_dir / "cross_combinations.json", cross)
        del models

    esol_summary = {"runs": main_table(esol_runs)} if esol_runs else {}
    if esol_summary:
        C.write_json(out_dir / "esol_table.json", esol_summary)
    jud = judgement(table_rows, fact, resp, cross, esol_summary)
    C.write_json(out_dir / "judgement.json", jud)
    C.write_json(out_dir / "run_manifest.json", {
        rid: {"manifest": r["manifest"], "best": best_row(r), "final": final_row(r),
              "dir": r["dir"]}
        for rid, r in runs.items()})
    print(f"wrote analysis to {out_dir}")
    return 0


def load_models(runs: Mapping[str, dict], cfg, pack, device: str) -> dict[str, Any]:
    import torch

    import trc_model as M

    out: dict[str, Any] = {}
    atom_dims, bond_dims = C.official_feature_dims()
    for rid, run in sorted(runs.items()):
        path = C.results_root() / rid / "state_best.pt"
        if not path.is_file():
            continue
        arm = run["manifest"].get("arm")
        blob = torch.load(path, map_location=device, weights_only=False)
        model = M.ResponseModel(cfg, atom_dims, bond_dims, arm)
        model.load_pack(pack)
        model.load_state_dict(blob["state"])
        model.to(device)
        model.eval()
        out[rid] = model
    return out


if __name__ == "__main__":
    raise SystemExit(main())
