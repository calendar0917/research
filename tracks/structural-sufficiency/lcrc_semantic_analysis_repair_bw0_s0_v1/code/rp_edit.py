"""Corrected real edit response: coverage first, then metrics.

Reuses the EDIT round's existing pairs (never re-mines, never relaxes legality or
label filters).  Steps, in the required order:

1. map every pair to official OGB rows by *unique canonical SMILES*, verify the
   real labels and molecule version, deduplicate by unordered row pair, fix the
   direction ``row_a < row_b``, ``Δy = y_b - y_a``, ``Δpred = pred_b - pred_a``;
2. build the undirected pair graph over all mapped pairs **once** and compute
   connected components; ``component_id`` is the smallest global row id in the
   component and is reused by every model / split cell;
3. report coverage per fixed cell (TT, Train-Valid, Train-Test, HH, E_T, ALL);
4. per model, per cell: CE-MAE (component mean of ``e=|Δpred-Δy|``, then
   equal-weight over components with a pair), pair-MAE, sign agreement
   (``|Δy|>=0.1``, component- then equal-weight), and the zero-change baseline
   ``|Δy|`` under the same weights;
5. component-level Exp(1) common-weight bootstrap (500 reps, seed 20261115) on
   CE-MAE differences for supported cells.

``LOW_SUPPORT`` is a **component count** rule (<20 components in the cell);
component-equal weighting is not interchangeable with pair weighting.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import rp_common as RP

SIGN_MIN = 0.1
CELLS = ("TT", "Train-Valid", "Train-Test", "HH", "E_T", "ALL")


# ---------------------------------------------------------------------------
# pairs + identity
# ---------------------------------------------------------------------------


def load_pairs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import lp_source as LPS

    src = RP.EDIT / "analysis/pairs.csv"
    rows = list(csv.DictReader(open(src, newline="", encoding="utf-8")))
    raw = LPS.load_raw_artifact()
    from rdkit import Chem

    def _canon(smi: str) -> str | None:
        m = Chem.MolFromSmiles(str(smi))
        return None if m is None else Chem.MolToSmiles(m)

    canon = [_canon(r["canonical_smiles"]) for r in raw]
    if any(c is None for c in canon) or len(set(canon)) != len(canon):
        raise SystemExit("canonical SMILES not unique across the 4200 official rows")
    canon_to_row = {c: i for i, c in enumerate(canon)}
    labels = LPS.load_labels()
    y = np.asarray(labels["y"], dtype=np.float64)
    split_payload = LPS.load_split()
    row2split: dict[int, str] = {}
    for _s in RP.SPLITS:
        for _r in split_payload["rows"][_s]:
            row2split[int(_r)] = _s

    out: list[dict[str, Any]] = []
    problems: list[str] = []
    raise_wrong_rows: list[str] = []
    for r in rows:
        ca = _canon(r["smiles_a"])
        cb = _canon(r["smiles_b"])
        a = canon_to_row.get(ca) if ca is not None else None
        b = canon_to_row.get(cb) if cb is not None else None
        if a is None or b is None:
            problems.append(f"{r['pair_id']}: canonical SMILES not in the official 4200")
            continue
        if int(r["row_a"]) != a or int(r["row_b"]) != b:
            problems.append(
                f"{r['pair_id']}: canonical row {a}/{b} != stored row {r['row_a']}/{r['row_b']} "
                f"(stored smiles are not RDKit-canonical; mapped via RDKit canonical SMILES)"
            )
            raise_wrong_rows.append(r["pair_id"])
            continue
        out.append(
            {
                "pair_id": r["pair_id"],
                "row_a": a,
                "row_b": b,
                "split_a": row2split[a],
                "split_b": row2split[b],
                "stored_split_a": r["split_a"],
                "stored_split_b": r["split_b"],
                "class_id": r.get("class_id", ""),
                "core_smiles": r.get("core_smiles", ""),
                "core_class": r.get("core_class", ""),
                "delta_y": float(y[b] - y[a]),
            }
        )
    n_relabelled = sum(1 for p in out if (p["split_a"], p["split_b"]) != (p["stored_split_a"], p["stored_split_b"]))
    return out, {
        "n_source_rows": len(rows),
        "problems": problems,
        "canonical_unique": True,
        "split_source": "official ogbg-mollipo scaffold split, re-derived per global row_id",
        "n_split_labels_relabelled_from_pairs_csv": n_relabelled,
        "note": (
            "the EDIT pairs.csv split_a/split_b columns do not agree with the official scaffold "
            "split for most rows; this round takes the split from the official row membership"
        ),
    }


def canonicalise(pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Deduplicate by unordered row pair, direction row_a < row_b."""
    seen: dict[tuple[int, int], dict[str, Any]] = {}
    n_dup = 0
    for p in pairs:
        a, b = p["row_a"], p["row_b"]
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            n_dup += 1
            continue
        q = dict(p)
        if a > b:
            q["row_a"], q["row_b"] = b, a
            q["split_a"], q["split_b"] = p["split_b"], p["split_a"]
            q["delta_y"] = -p["delta_y"]
        seen[key] = q
    return list(seen.values()), {"n_duplicate_unordered": n_dup}


def components(pairs: list[dict[str, Any]]) -> tuple[dict[int, int], dict[str, Any]]:
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for p in pairs:
        union(p["row_a"], p["row_b"])
    comp = {n: find(n) for n in parent}
    n_comp = len(set(comp.values()))
    return comp, {"n_endpoints": len(comp), "n_components": n_comp}


def cell_of(p: Mapping[str, Any]) -> set[str]:
    sa, sb = p["split_a"], p["split_b"]
    both = {sa, sb}
    cells: set[str] = set()
    if sa == "train" and sb == "train":
        cells.add("TT")
    if both == {"train", "valid"}:
        cells.add("Train-Valid")
    if both == {"train", "test"}:
        cells.add("Train-Test")
    if sa != "train" and sb != "train":
        cells.add("HH")
    if "test" in both:
        cells.add("E_T")
    cells.add("ALL")
    return cells


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _weighted_component(values: dict[int, list[float]], weights: Mapping[int, float] | None = None) -> dict[str, Any]:
    comps = [c for c, v in values.items() if v]
    if not comps:
        return {"n_components": 0, "ce_mean": None, "pair_mean": None, "n_pairs": 0}
    means = np.array([float(np.mean(values[c])) for c in comps], dtype=np.float64)
    if weights is None:
        w = np.ones_like(means)
    else:
        w = np.array([float(weights.get(c, 0.0)) for c in comps], dtype=np.float64)
    ce = float(np.sum(w * means) / np.sum(w)) if np.sum(w) > 0 else None
    allvals = [x for c in comps for x in values[c]]
    return {
        "n_components": len(comps),
        "ce_mean": ce,
        "pair_mean": float(np.mean(allvals)) if allvals else None,
        "n_pairs": len(allvals),
    }


def _metrics_for_cell(
    pairs: list[dict[str, Any]],
    comp: Mapping[int, int],
    d_pred: Mapping[tuple[int, int], float],
    weights: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    e_by_c: dict[int, list[float]] = {}
    sign_by_c: dict[int, list[float]] = {}
    zero_by_c: dict[int, list[float]] = {}
    n_cell_pairs = 0
    for p in pairs:
        key = (p["row_a"], p["row_b"])
        if key not in d_pred:
            continue
        c = comp[p["row_a"]]
        e_by_c.setdefault(c, []).append(abs(d_pred[key] - p["delta_y"]))
        zero_by_c.setdefault(c, []).append(abs(p["delta_y"]))
        if abs(p["delta_y"]) >= SIGN_MIN:
            same = 1.0 if np.sign(d_pred[key]) == np.sign(p["delta_y"]) else 0.0
            sign_by_c.setdefault(c, []).append(same)
        n_cell_pairs += 1
    main = _weighted_component(e_by_c, weights)
    zero = _weighted_component(zero_by_c, weights)
    sign = _weighted_component(sign_by_c, weights)
    return {
        "n_pairs": n_cell_pairs,
        "n_components": main["n_components"],
        "ce_mae": main["ce_mean"],
        "pair_mae": main["pair_mean"],
        "zero_ce_mae": zero["ce_mean"],
        "zero_pair_mae": zero["pair_mean"],
        "sign_agreement_component": sign["ce_mean"],
        "sign_agreement_pair": sign["pair_mean"],
        "sign_n_pairs": sign["n_pairs"],
        "low_support": bool(main["n_components"] < RP.LOW_SUPPORT_MIN_COMPONENTS),
    }


def arm_predictions(arm_dir: Path) -> dict[int, float] | None:
    """global row_id -> best raw prediction (all splits)."""
    out: dict[int, float] = {}
    for split in RP.SPLITS:
        p = arm_dir / f"pred_{split}_best.npz"
        if not p.is_file():
            return None
        d = np.load(p, allow_pickle=False)
        rows = np.asarray(d["rows"]).tolist()
        raw = np.asarray(d["raw"], dtype=np.float64)
        for r, v in zip(rows, raw):
            out[int(r)] = float(v)
    return out


def molecule_rmse(arm_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in RP.SPLITS:
        p = arm_dir / f"pred_{split}_best.npz"
        if not p.is_file():
            out[split] = None
            continue
        d = np.load(p, allow_pickle=False)
        raw = np.asarray(d["raw"], float)
        y = np.asarray(d["y_true"], float)
        out[split] = {"rmse": float(np.sqrt(np.mean((raw - y) ** 2))), "n": int(y.size)}
    return out


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _arms() -> dict[str, Path]:
    arms = {k: RP.LIPO_ARMS[k] for k in ("B_R", "LS_R", "FS_R", "LP_R", "FP_R", "LS_W0", "FS_W0")}
    if (RP.BW0_DIR / "pred_valid_best.npz").is_file():
        arms["B_W0"] = RP.BW0_DIR
    return arms


def run() -> dict[str, Any]:
    raw_pairs, identity = load_pairs()
    pairs, dedup = canonicalise(raw_pairs)
    comp, comp_stats = components(pairs)
    for p in pairs:
        p["component_id"] = int(comp[p["row_a"]])
        p["cells"] = sorted(cell_of(p))

    # ---- coverage ------------------------------------------------------
    coverage: dict[str, Any] = {}
    for cell in CELLS:
        sel = [p for p in pairs if cell in p["cells"]]
        comps = {comp[p["row_a"]] for p in sel}
        endpoints = {p["row_a"] for p in sel} | {p["row_b"] for p in sel}
        cov = {
            "n_pairs": len(sel),
            "n_endpoints": len(endpoints),
            "n_components": len(comps),
            "split_endpoint_counts": {
                s: len({r for p in sel for r, sp in ((p["row_a"], p["split_a"]), (p["row_b"], p["split_b"])) if sp == s})
                for s in RP.SPLITS
            },
            "low_support": bool(len(comps) < RP.LOW_SUPPORT_MIN_COMPONENTS),
        }
        if cell == "TT":
            cov["train_test_side"] = {
                "n_test_molecules": len({r for p in sel for r, sp in ((p["row_a"], p["split_a"]), (p["row_b"], p["split_b"])) if sp == "test"}),
                "n_full_graph_components": len({comp[r] for p in sel for r, sp in ((p["row_a"], p["split_a"]), (p["row_b"], p["split_b"])) if sp == "test"}),
            }
        coverage[cell] = cov

    # split-combo table (for the reference numbers)
    combos: dict[str, int] = {}
    for p in pairs:
        key = "-".join(sorted([p["split_a"], p["split_b"]]))
        combos[key] = combos.get(key, 0) + 1

    # ---- per-model metrics --------------------------------------------
    arms = _arms()
    metrics: dict[str, Any] = {}
    missing: list[str] = []
    preds: dict[str, dict[int, float]] = {}
    for name, d in arms.items():
        pr = arm_predictions(d)
        if pr is None:
            missing.append(name)
            continue
        preds[name] = pr
        nd: dict[tuple[int, int], float] = {}
        for p in pairs:
            a, b = p["row_a"], p["row_b"]
            if a in pr and b in pr:
                nd[(a, b)] = pr[b] - pr[a]
        cell_metrics = {cell: _metrics_for_cell([p for p in pairs if cell in p["cells"]], comp, nd) for cell in CELLS}
        metrics[name] = {
            "arm_dir": RP.track_rel(d),
            "n_pairs_with_prediction": len(nd),
            "cells": cell_metrics,
            "molecule_rmse": molecule_rmse(d),
        }

    # ---- bootstrap on supported cells ---------------------------------
    boot_cells = [c for c in ("TT", "ALL") if not coverage[c]["low_support"]]
    rng = np.random.default_rng(RP.BOOTSTRAP_SEED)
    boot: dict[str, Any] = {"cells": {}, "replicates": RP.BOOTSTRAP_REPLICATES, "seed": RP.BOOTSTRAP_SEED}
    for cell in boot_cells:
        sel = [p for p in pairs if cell in p["cells"]]
        comps = sorted({comp[p["row_a"]] for p in sel})
        w = rng.exponential(1.0, size=(RP.BOOTSTRAP_REPLICATES, len(comps)))
        cell_out: dict[str, Any] = {"n_components": len(comps), "models": {}, "differences": {}}
        per_model_ce: dict[str, np.ndarray] = {}
        for name, pr in preds.items():
            nd = {(p["row_a"], p["row_b"]): pr[p["row_b"]] - pr[p["row_a"]] for p in sel if p["row_a"] in pr and p["row_b"] in pr}
            vals: dict[int, list[float]] = {}
            for p in sel:
                k = (p["row_a"], p["row_b"])
                if k in nd:
                    vals.setdefault(comp[p["row_a"]], []).append(abs(nd[k] - p["delta_y"]))
            means = np.array([float(np.mean(vals.get(c, [0.0]))) for c in comps])
            per_model_ce[name] = means
            ce_reps = (w * means[None, :]).sum(1) / w.sum(1)
            cell_out["models"][name] = {
                "ce_mae": float(means.mean()),
                "ce_mae_boot_mean": float(ce_reps.mean()),
                "ci95": [float(np.percentile(ce_reps, 2.5)), float(np.percentile(ce_reps, 97.5))],
            }
        for a in per_model_ce:
            for b in per_model_ce:
                if a >= b:
                    continue
                diff = (w * (per_model_ce[a] - per_model_ce[b])[None, :]).sum(1) / w.sum(1)
                cell_out["differences"][f"{a}-{b}"] = {
                    "delta_ce_mae": float(per_model_ce[a].mean() - per_model_ce[b].mean()),
                    "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                    "crosses_zero": bool(np.percentile(diff, 2.5) <= 0.0 <= np.percentile(diff, 97.5)),
                }
        boot["cells"][cell] = cell_out

    # ---- artifacts -----------------------------------------------------
    map_csv = RP.ANALYSIS / "pair_mapping_checked.csv"
    with open(map_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair_id", "row_a", "row_b", "split_a", "split_b",
                                           "component_id", "delta_y", "class_id", "cells"])
        w.writeheader()
        for p in pairs:
            w.writerow({**{k: p[k] for k in ("pair_id", "row_a", "row_b", "split_a", "split_b",
                                             "component_id", "delta_y", "class_id")},
                        "cells": "|".join(p["cells"])})
    comp_csv = RP.ANALYSIS / "pair_components.csv"
    with open(comp_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["component_id", "n_pairs", "rows"])
        w.writeheader()
        for cid in sorted(set(comp.values())):
            sel = [p for p in pairs if comp[p["row_a"]] == cid]
            rows_ = sorted({p["row_a"] for p in sel} | {p["row_b"] for p in sel})
            w.writerow({"component_id": cid, "n_pairs": len(sel), "rows": "|".join(map(str, rows_))})

    cov_payload = {
        "schema": "lcrc-semantic-analysis-repair-edit-coverage-v1",
        "protocol_id": RP.PROTOCOL_ID,
        "created_utc": RP.utcnow(),
        "identity": identity,
        "dedup": dedup,
        "graph": comp_stats,
        "split_combos": combos,
        "cells": coverage,
        "reference_expected": {
            "n_edges": 823, "n_endpoints": 704, "n_components": 231,
            "Train-Train": 791, "Train-Valid": 5, "Train-Test": 27, "both_non_train": 0,
        },
    }
    RP.write_json(RP.ANALYSIS / "edit_coverage.json", cov_payload)

    met_payload = {
        "schema": "lcrc-semantic-analysis-repair-edit-metrics-v1",
        "protocol_id": RP.PROTOCOL_ID,
        "created_utc": RP.utcnow(),
        "definitions": {
            "ce_mae": "per-component mean of e=|Δpred-Δy|, then equal weight over components with a pair",
            "pair_mae": "plain mean over pairs (NOT component equal weight)",
            "sign": "|Δy|>=0.1, per-component fraction then equal weight over qualifying components",
            "zero_baseline": "|Δy| with the same cell/weights; tests whether predictions move at all",
            "low_support": f"component count < {RP.LOW_SUPPORT_MIN_COMPONENTS}",
        },
        "low_support_components": RP.LOW_SUPPORT_MIN_COMPONENTS,
        "missing_models": missing,
        "models": metrics,
        "bootstrap": boot,
    }
    RP.write_json(RP.ANALYSIS / "edit_metrics_corrected.json", met_payload)

    # ---- before / after ------------------------------------------------
    old = RP.read_json(RP.EDIT / "analysis/comparisons.json") if (RP.EDIT / "analysis/comparisons.json").is_file() else {}
    ba_rows = []
    for name in metrics:
        for cell in CELLS:
            ba_rows.append({
                "model": name,
                "cell": cell,
                "ce_mae_corrected": metrics[name]["cells"][cell]["ce_mae"],
                "pair_mae_corrected": metrics[name]["cells"][cell]["pair_mae"],
                "n_components": metrics[name]["cells"][cell]["n_components"],
                "n_pairs": metrics[name]["cells"][cell]["n_pairs"],
                "low_support": metrics[name]["cells"][cell]["low_support"],
            })
    ba_csv = RP.ANALYSIS / "edit_before_after.csv"
    with open(ba_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ba_rows[0].keys()))
        w.writeheader()
        w.writerows(ba_rows)

    return {
        "graph": comp_stats,
        "split_combos": combos,
        "cells": {c: {"n_pairs": coverage[c]["n_pairs"], "n_components": coverage[c]["n_components"],
                      "low_support": coverage[c]["low_support"]} for c in CELLS},
        "missing_models": missing,
        "old_comparisons_present": bool(old),
        "artifacts": [RP.track_rel(p) for p in (map_csv, comp_csv,
                                                RP.ANALYSIS / "edit_coverage.json",
                                                RP.ANALYSIS / "edit_metrics_corrected.json", ba_csv)],
    }


__all__ = [name for name in dir() if not name.startswith("_")]
