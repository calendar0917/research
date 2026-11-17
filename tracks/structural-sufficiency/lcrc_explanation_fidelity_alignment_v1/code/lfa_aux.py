"""Auxiliary 823-pair Lipophilicity diagnostic (CPU, description only).

The pairs are a fixed, pre-existing 823-pair catalogue from the sprint round.
Nothing here selects pairs by label, expands labels, or adds a pair-aware
explainer: the eight (method, model) explainers are the *same* frozen ones
scored in the main experiment, merely re-used as descriptions on the pair
endpoints.  The edit region is a structural difference, not causal property
truth, so absence of Top-1 overlap is reported as a description, never as
proof of unfaithfulness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import lfa_common as C
import lfa_data as D
import lfa_main as MAIN
import lfa_search as S

SELECTORS_ORDER = ("A_LCRC_GRAD", "B_LCRC_LOCAL", "C_LCRC_MASK50", "D_TOPOLOGY",
                   "E_GINE_GRAD", "F_GINE_MASK50")

METRIC_KEYS = (
    "atom_precision", "atom_recall", "atom_jaccard",
    "bond_precision", "bond_recall", "bond_jaccard",
    "hit_attachment_atom", "hit_attachment_bond", "hit_attachment_boundary",
)
SLICES = ("all", "train_train", "non_train", "both_non_train", "pair_correct", "pair_wrong")
METHOD_LABELS = {
    "A_LCRC_GRAD": "A_LCRC_GRAD", "B_LCRC_LOCAL": "B_LCRC_LOCAL",
    "C_LCRC_MASK50": "C_LCRC_MASK50", "D_TOPOLOGY": "D_TOPOLOGY",
    "E_GINE_GRAD": "E_GINE_GRAD", "F_GINE_MASK50": "F_GINE_MASK50",
}


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def load_endpoints() -> tuple[dict[tuple[str, str], dict], dict[str, dict]]:
    endpoints: dict[tuple[str, str], dict] = {}
    for rec in C.read_jsonl_gz(C.XR_DIR / "analysis" / "pair_endpoints.jsonl.gz"):
        endpoints[(str(rec["pair_id"]), str(rec["side"]))] = rec
    catalog = {str(rec["pair_id"]): rec for rec in C.read_jsonl_gz(C.XR_DIR / "analysis" / "pair_catalog.jsonl.gz")}
    return endpoints, catalog


def region_sets(ep: Mapping[str, Any], mol) -> dict[str, set]:
    var_atoms = {int(v) for v in ep["var_atoms"]}
    boundary_atoms = {int(v) for v in ep["boundary_atoms"]}
    edit_atoms = {int(v) for v in ep["edit_atoms"]}
    core_atoms = set(range(int(mol.n_atoms))) - edit_atoms
    edit_bonds = {tuple(sorted(int(x) for x in b)) for b in ep["edit_bonds"]}
    attach_bonds = {tuple(sorted(int(x) for x in b)) for b in ep["attach_bonds"]}
    changed_core = {tuple(sorted(int(x) for x in b)) for b in ep["changed_core_bonds"]}
    var_bonds = {tuple(sorted(int(x) for x in b)) for b in ep["var_bonds"]} - attach_bonds
    all_bonds = {(min(int(mol.bond_u[b]), int(mol.bond_v[b])), max(int(mol.bond_u[b]), int(mol.bond_v[b])))
                 for b in range(int(mol.n_bonds))}
    other = all_bonds - edit_bonds - attach_bonds - changed_core
    return {
        "var_atoms": var_atoms, "boundary_atoms": boundary_atoms, "edit_atoms": edit_atoms,
        "core_atoms": core_atoms, "var_bonds": var_bonds, "attach_bonds": attach_bonds,
        "changed_core_bonds": changed_core, "other_bonds": other, "all_bonds": all_bonds,
        "edit_bonds": edit_bonds,
    }


def _bond_key(mol, b: int) -> tuple[int, int]:
    return (min(int(mol.bond_u[b]), int(mol.bond_v[b])), max(int(mol.bond_u[b]), int(mol.bond_v[b])))


def composition(mol, bonds: Sequence[int], regions: Mapping[str, set]) -> dict[str, Any]:
    cand_bonds = [_bond_key(mol, int(b)) for b in bonds]
    cand_atoms = {int(v) for v in mol.bond_atoms([int(b) for b in bonds])}
    out: dict[str, Any] = {"n_bonds": len(cand_bonds), "n_atoms": len(cand_atoms)}
    for name in ("var_atoms", "boundary_atoms", "core_atoms"):
        out[f"atom_fraction_{name}"] = len(cand_atoms & regions[name]) / len(cand_atoms) if cand_atoms else None
    for name in ("var_bonds", "attach_bonds", "changed_core_bonds", "other_bonds"):
        out[f"bond_fraction_{name}"] = len(set(cand_bonds) & regions[name]) / len(cand_bonds) if cand_bonds else None
    return out


def score_mass_shares(fields_lcrc: Mapping[str, Any], fields_gine: Mapping[str, Any],
                      mol, regions: Mapping[str, set]) -> dict[str, Any]:
    """Share of the raw per-element score mass landing in each region.

    A/E use |g_e| per physical bond, B uses |c_v| per atom.  D is a set-level
    S2-count and therefore has no per-element mass.
    """
    out: dict[str, Any] = {}
    for name, rec in (("LCRC_GRAD", fields_lcrc), ("GINE_GRAD", fields_gine)):
        g = np.abs(np.asarray(rec["grad"], dtype=np.float64))
        total = float(g.sum())
        if total <= 0.0:
            out[f"{name}_share_var"], out[f"{name}_share_attach"], out[f"{name}_share_core"] = None, None, None
            continue
        per = {"var": 0.0, "attach": 0.0, "core": 0.0}
        for b in range(int(mol.n_bonds)):
            key = _bond_key(mol, b)
            if key in regions["var_bonds"]:
                per["var"] += float(g[b])
            elif key in regions["attach_bonds"] or key in regions["changed_core_bonds"]:
                per["attach"] += float(g[b])
            else:
                per["core"] += float(g[b])
        out[f"{name}_share_var"] = per["var"] / total
        out[f"{name}_share_attach"] = per["attach"] / total
        out[f"{name}_share_core"] = per["core"] / total
    local = np.abs(np.asarray(fields_lcrc["contrib"], dtype=np.float64))
    total = float(local.sum())
    if total > 0.0:
        out["LCRC_LOCAL_share_var"] = float(local[sorted(regions["var_atoms"])].sum() / total)
        out["LCRC_LOCAL_share_boundary"] = float(local[sorted(regions["boundary_atoms"])].sum() / total)
        out["LCRC_LOCAL_share_core"] = float(local[sorted(regions["core_atoms"])].sum() / total)
    return out


# ---------------------------------------------------------------------------
# endpoint rows
# ---------------------------------------------------------------------------


def endpoint_rows(*, pool_path: Path | str, select_lcrc: Path | str, select_gine: Path | str,
                  split_filter: Sequence[str] | None = None) -> list[dict]:
    import xr_discover

    pool = MAIN.load_jsonl_map(pool_path)
    lcrc = MAIN.load_jsonl_map(select_lcrc)
    gine = MAIN.load_jsonl_map(select_gine)
    endpoints, catalog = load_endpoints()
    b = D.bundle()
    rows: list[dict] = []
    missing = 0
    for (pair_id, side), ep in sorted(endpoints.items()):
        split = str(ep["split"])
        if split_filter is not None and split not in split_filter:
            continue
        key = (split, int(ep["cache_row"]))
        if key not in pool:
            missing += 1
            continue
        prec, lrec, grec = pool[key], lcrc[key], gine[key]
        mol = C.source_modules()["as_gate"].SingleMolecule(b.caches[split], int(ep["cache_row"]), b.spec)
        regions = region_sets(ep, mol)
        edit = {"edit_atoms": ep["edit_atoms"], "edit_bonds": ep["edit_bonds"],
                "attach_atom": int(ep["attach_atom"]), "attach_bonds": ep["attach_bonds"]}
        contrib = np.asarray(ep["oracle_delta"], dtype=np.float64)
        denom = float(np.abs(contrib).sum())
        meta = {"pair_id": pair_id, "side": side, "component": int(ep["component"]),
                "split": split, "split_pair": str(ep["split_pair"]),
                "train_train": bool(ep["train_train"]), "cache_row": int(ep["cache_row"]),
                "row_id": int(ep["row_id"]), "n_atoms": int(mol.n_atoms), "n_bonds": int(mol.n_bonds),
                "evaluable": bool(ep["evaluable"])}
        for method in C.SELECTORS:
            if method in ("C_LCRC_MASK50", "F_GINE_MASK50"):
                sel_rec = lrec if method.startswith("C_") else grec
                top = sel_rec.get("mask50_selection")
                controls = sel_rec.get("mask50_control_bonds", [])
            else:
                top = prec["selections"][method]
                controls = [rec["bonds"] for rec in prec["controls"][method].get("controls", [])]
            if top is None:
                continue
            bonds = [int(v) for v in top["bonds"]]
            loc = xr_discover.localisation_for_candidate(mol, bonds, edit, [{"bonds": list(c)} for c in controls])
            atoms = [int(v) for v in mol.bond_atoms(bonds)]
            capture = None
            if denom > 0:
                capture = float(np.abs(contrib[atoms]).sum() / denom) if atoms else 0.0
            row = dict(meta)
            row.update({
                "method": method, "k": len(bonds),
                "bonds": json.dumps(bonds),
                "atom_capture": capture,
                "atom_capture_random_median": None,
                "oracle_defined": bool(denom > 0),
                "n_controls": int(len(controls)),
            })
            cap_ctrl = []
            if denom > 0:
                for c in controls:
                    ca = [int(v) for v in mol.bond_atoms([int(x) for x in c])]
                    cap_ctrl.append(float(np.abs(contrib[ca]).sum() / denom) if ca else 0.0)
            if cap_ctrl:
                row["atom_capture_random_median"] = float(np.median(cap_ctrl))
                row["atom_capture_enrichment"] = float((capture if capture is not None else 0.0)
                                                       - float(np.median(cap_ctrl)))
            else:
                row["atom_capture_enrichment"] = None
            for metric in METRIC_KEYS:
                row[f"top_{metric}"] = float(loc["top1"][metric])
                row[f"random_{metric}"] = loc["random"].get(f"mean_{metric}")
                row[f"adv_{metric}"] = loc["advantage"].get(metric)
            row.update(composition(mol, bonds, regions))
            fl, fg = _fields_for(split, int(ep["cache_row"]))
            if fl is not None and fg is not None:
                row.update(score_mass_shares(fl, fg, mol, regions))
            rows.append(row)
    if missing:
        print(f"[aux] warning: {missing} endpoints had no pool record", flush=True)
    return rows


_FIELD_CACHE: dict[str, dict[tuple[str, int], dict]] = {}


def _fields_for(split: str, cache_row: int) -> tuple[dict | None, dict | None]:
    out = []
    for family in ("lcrc", "gine"):
        path = C.FIELDS_DIR / f"{family}_lipo_{C.SCOPE_AUX}.jsonl"
        if not path.is_file():
            out.append(None)
            continue
        if family not in _FIELD_CACHE:
            _FIELD_CACHE[family] = MAIN.load_jsonl_map(path)
        out.append(_FIELD_CACHE[family].get((split, int(cache_row))))
    return out[0], out[1]


# ---------------------------------------------------------------------------
# prediction diagnostics
# ---------------------------------------------------------------------------


def prediction_diagnostics(*, pred_path: Path | str) -> dict[str, Any]:
    npz = C.read_npz(pred_path)
    catalog = {str(rec["pair_id"]): rec for rec in C.read_jsonl_gz(C.XR_DIR / "analysis" / "pair_catalog.jsonl.gz")}
    b = D.bundle()
    labels = np.asarray(b.payload["labels"], dtype=np.float64)
    out: dict[str, Any] = {"splits": {}, "pairs": {}}
    for split in ("valid", "test", "train"):
        for family in ("lcrc", "gine"):
            key = f"{split}_{family}"
            if f"{key}_pred" not in npz:
                continue
            pred = np.asarray(npz[f"{key}_pred"], dtype=np.float64)
            y = np.asarray(npz[f"{key}_y"], dtype=np.float64)
            out["splits"][key] = {
                "n": int(pred.size), "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
                "mae": float(np.mean(np.abs(pred - y))),
            }
    row_index: dict[tuple[str, str], dict[int, int]] = {}
    for f in ("lcrc", "gine"):
        for s in ("train", "valid", "test"):
            key = f"{s}_{f}_row"
            if key in npz:
                row_index[(s, f)] = {int(r): i for i, r in enumerate(np.asarray(npz[key], dtype=np.int64))}
    for family in ("lcrc", "gine"):
        deltas, labels_d, rows = [], [], []
        for pair_id, rec in sorted(catalog.items()):
            ra, rb = int(rec["row_a"]), int(rec["row_b"])
            sa, sb = str(rec["split_a"]), str(rec["split_b"])
            if (sa, family) not in row_index or (sb, family) not in row_index:
                continue
            ia, ib = row_index[(sa, family)].get(ra), row_index[(sb, family)].get(rb)
            if ia is None or ib is None:
                continue
            d_pred = float(np.asarray(npz[f"{sa}_{family}_pred"], dtype=np.float64)[ia]
                           - np.asarray(npz[f"{sb}_{family}_pred"], dtype=np.float64)[ib])
            d_true = float(labels[ra] - labels[rb])
            deltas.append(d_pred)
            labels_d.append(d_true)
            rows.append({"pair_id": pair_id, "delta_pred": d_pred, "delta_true": d_true,
                         "abs_error": abs(d_pred - d_true), "train_train": bool(rec["train_train"]),
                         "component": int(rec["component"]), "split_pair": str(rec["split_pair"])})
        if not deltas:
            continue
        d_pred = np.asarray(deltas, dtype=np.float64)
        d_true = np.asarray(labels_d, dtype=np.float64)
        err = np.abs(d_pred - d_true)
        strong = np.abs(d_true) >= 0.1
        entry = {
            "n_pairs": int(err.size),
            "mean_abs_delta_error": float(np.nanmean(err)),
            "median_abs_delta_error": float(np.nanmedian(err)),
            "rmse_delta": float(np.sqrt(np.nanmean((d_pred - d_true) ** 2))),
            "direction_accuracy_all": float(np.mean(np.sign(d_pred) == np.sign(d_true))),
            "direction_accuracy_strong": float(np.mean(np.sign(d_pred[strong]) == np.sign(d_true[strong]))) if strong.any() else None,
            "n_strong": int(strong.sum()),
            "mean_abs_true_delta": float(np.mean(np.abs(d_true))),
        }
        out["pairs"][family] = {"summary": entry, "rows": rows}
    return out


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------


def _slice(rows: Sequence[Mapping[str, Any]], name: str) -> list[Mapping[str, Any]]:
    if name == "all":
        return list(rows)
    if name == "train_train":
        return [r for r in rows if r["train_train"]]
    if name == "non_train":
        return [r for r in rows if not r["train_train"]]
    if name == "both_non_train":
        return [r for r in rows if "train" not in str(r.get("split_pair"))]
    if name in ("pair_correct", "pair_wrong"):
        family = {"A_LCRC_GRAD": "lcrc", "B_LCRC_LOCAL": "lcrc", "C_LCRC_MASK50": "lcrc",
                  "D_TOPOLOGY": "lcrc", "E_GINE_GRAD": "gine", "F_GINE_MASK50": "gine"}
        want = name == "pair_correct"
        return [r for r in rows if r.get(f"pair_correct_{family[r['method']]}") is want]
    return list(rows)


def delta_localization() -> dict[str, Any]:
    """How localized the frozen pair Δ contribution is (description only).

    If the Δ contribution is non-zero only on the edit atoms, a "capture" of
    that Δ over a candidate set carries no information beyond "does S touch the
    edit atoms"; the report must say so instead of calling it independent.
    """
    endpoints, _ = load_endpoints()
    fracs, outside, subset = [], [], 0
    for ep in endpoints.values():
        delta = np.abs(np.asarray(ep["oracle_delta"], dtype=np.float64))
        nonzero = set(int(i) for i in np.flatnonzero(delta > 1e-12))
        edit = set(int(i) for i in ep["edit_atoms"])
        fracs.append(len(nonzero) / max(1, int(delta.size)))
        outside.append(len(nonzero - edit))
        subset += int(nonzero <= edit)
    return {
        "n_endpoints": len(fracs),
        "mean_nonzero_atom_fraction": float(np.mean(fracs)),
        "median_nonzero_atom_fraction": float(np.median(fracs)),
        "endpoints_support_subset_of_edit_atoms": int(subset),
        "mean_atoms_outside_edit": float(np.mean(outside)),
        "max_atoms_outside_edit": int(np.max(outside)),
    }


def summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method in SELECTORS_ORDER:
        sub = [r for r in rows if r["method"] == method]
        if not sub:
            continue
        for slice_name in SLICES:
            part = _slice(sub, slice_name)
            if not part:
                out[f"{method}|{slice_name}"] = {"method": method, "slice": slice_name, "n": 0,
                                                 "support": "NO_SUPPORT"}
                continue
            units = np.asarray([int(r["component"]) for r in part], dtype=np.int64)
            n_components = int(np.unique(units).size)
            entry: dict[str, Any] = {
                "method": method, "slice": slice_name, "n": len(part),
                "n_components": n_components,
                "n_molecules": int(len({(r["split"], r["cache_row"]) for r in part})),
                "n_train_train": int(sum(bool(r["train_train"]) for r in part)),
                "n_non_train": int(sum(not bool(r["train_train"]) for r in part)),
                "support": C.support_label(len(part), n_components),
                "k_mean": float(np.mean([r["k"] for r in part])),
            }
            for metric in METRIC_KEYS:
                top = np.asarray([float(r[f"top_{metric}"]) for r in part], dtype=np.float64)
                rand = np.asarray([np.nan if r[f"random_{metric}"] is None else float(r[f"random_{metric}"])
                                   for r in part], dtype=np.float64)
                adv = np.asarray([np.nan if r[f"adv_{metric}"] is None else float(r[f"adv_{metric}"])
                                  for r in part], dtype=np.float64)
                mask = np.isfinite(adv)
                boot = C.bootstrap_columns({"top": top, "random": rand, "adv": adv}, mask, unit_ids=units)
                entry[metric] = {
                    "top_mean": boot["point"]["top"]["mean"],
                    "top_median": boot["point"]["top"]["median"],
                    "random_mean": boot["point"]["random"]["mean"],
                    "random_median": boot["point"]["random"]["median"],
                    "adv_mean": boot["point"]["adv"]["mean"],
                    "adv_median": boot["point"]["adv"]["median"],
                    "adv_ci95_mean": boot["ci95"]["adv"]["mean"],
                    "adv_ci95_median": boot["ci95"]["adv"]["median"],
                    "n_evaluable": int(mask.sum()),
                    "adv_gt0_fraction": float(np.mean(adv[mask] > 0)) if mask.any() else None,
                }
            cap = np.asarray([np.nan if r["atom_capture"] is None else float(r["atom_capture"]) for r in part])
            cap_adv = np.asarray([np.nan if r.get("atom_capture_enrichment") is None
                                  else float(r["atom_capture_enrichment"]) for r in part])
            mask = np.isfinite(cap_adv)
            boot = C.bootstrap_columns({"capture": cap, "enrichment": cap_adv}, mask, unit_ids=units)
            entry["atom_capture"] = {
                "mean": boot["point"]["capture"]["mean"],
                "median": boot["point"]["capture"]["median"],
                "enrichment_mean": boot["point"]["enrichment"]["mean"],
                "enrichment_ci95_mean": boot["ci95"]["enrichment"]["mean"],
                "n_evaluable": int(mask.sum()),
            }
            for metric in ("atom_fraction_var_atoms", "atom_fraction_boundary_atoms", "atom_fraction_core_atoms",
                           "bond_fraction_var_bonds", "bond_fraction_attach_bonds",
                           "bond_fraction_changed_core_bonds", "bond_fraction_other_bonds",
                           "LCRC_GRAD_share_var", "LCRC_GRAD_share_attach", "LCRC_GRAD_share_core",
                           "GINE_GRAD_share_var", "GINE_GRAD_share_attach", "GINE_GRAD_share_core",
                           "LCRC_LOCAL_share_var", "LCRC_LOCAL_share_boundary", "LCRC_LOCAL_share_core"):
                vals = np.asarray([np.nan if r.get(metric) is None else float(r[metric]) for r in part])
                entry[metric] = {
                    "mean": float(np.nanmean(vals)) if np.isfinite(vals).any() else None,
                    "median": float(np.nanmedian(vals)) if np.isfinite(vals).any() else None,
                    "n": int(np.isfinite(vals).sum()),
                }
            out[f"{method}|{slice_name}"] = entry
    return out


def paired_vs_single_model(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Paired endpoint-level differences between methods (same component units)."""
    import itertools
    out = []
    for slice_name in SLICES:
        for a, b in itertools.combinations(SELECTORS_ORDER, 2):
            sa = {int(r["endpoint_index"]): r for r in _slice(rows, slice_name) if r["method"] == a}
            sb = {int(r["endpoint_index"]): r for r in _slice(rows, slice_name) if r["method"] == b}
            common = sorted(set(sa) & set(sb))
            if not common:
                continue
            for metric in ("adv_atom_jaccard", "adv_atom_recall", "adv_bond_recall", "adv_hit_attachment_boundary",
                           "atom_capture_enrichment"):
                va = np.asarray([np.nan if sa[i].get(metric) is None else float(sa[i][metric]) for i in common])
                vb = np.asarray([np.nan if sb[i].get(metric) is None else float(sb[i][metric]) for i in common])
                diff = va - vb
                mask = np.isfinite(diff)
                units = np.asarray([int(sa[i]["component"]) for i in common], dtype=np.int64)
                boot = C.bootstrap_columns({"diff": diff}, mask, unit_ids=units)
                out.append({
                    "slice": slice_name, "metric": metric, "method_a": a, "method_b": b,
                    "n": int(mask.sum()), "n_components": int(np.unique(units[mask]).size),
                    "mean_diff": boot["point"]["diff"]["mean"],
                    "ci95_mean": boot["ci95"]["diff"]["mean"],
                    "gt0_fraction": float(np.mean(diff[mask] > 0)) if mask.any() else None,
                })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(*, pool_dir: Path | str = C.POOL_DIR, select_dir: Path | str = C.SELECT_DIR,
        out_dir: Path | str = C.ANALYSIS_DIR, pred_path: Path | str | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    C.ensure_dir(out_dir)
    pool_path = Path(pool_dir) / "pool_aux.jsonl"
    rows = endpoint_rows(pool_path=pool_path,
                         select_lcrc=Path(select_dir) / "select_lcrc_aux.jsonl",
                         select_gine=Path(select_dir) / "select_gine_aux.jsonl")
    pred: dict[str, Any] | None = None
    if pred_path is not None and Path(pred_path).is_file():
        pred = prediction_diagnostics(pred_path=pred_path)
        # fixed thresholds, frozen before the run: a pair prediction is
        # "correct" when |delta_pred - delta_true| <= 0.25
        for family, block in pred["pairs"].items():
            error = {row["pair_id"]: float(row["abs_error"]) for row in block["rows"]}
            for row in rows:
                if row["pair_id"] in error:
                    row[f"pair_correct_{family}"] = bool(error[row["pair_id"]] <= 0.25)
                    row[f"pair_delta_error_{family}"] = error[row["pair_id"]]
    # stable endpoint index for paired diffs
    index: dict[tuple[str, str], int] = {}
    for rec in rows:
        key = (str(rec["pair_id"]), str(rec["side"]))
        if key not in index:
            index[key] = len(index)
    for rec in rows:
        rec["endpoint_index"] = index[(str(rec["pair_id"]), str(rec["side"]))]
    columns = ["pair_id", "side", "endpoint_index", "component", "split", "split_pair", "train_train",
               "cache_row", "row_id", "n_atoms", "n_bonds", "evaluable", "method", "k", "bonds",
               "atom_capture", "atom_capture_random_median", "atom_capture_enrichment", "oracle_defined",
               "n_controls"]
    columns += [f"{prefix}_{metric}" for metric in METRIC_KEYS for prefix in ("top", "random", "adv")]
    columns += ["pair_correct_lcrc", "pair_correct_gine", "pair_delta_error_lcrc", "pair_delta_error_gine"]
    columns += ["atom_fraction_var_atoms", "atom_fraction_boundary_atoms",
                "atom_fraction_core_atoms", "bond_fraction_var_bonds", "bond_fraction_attach_bonds",
                "bond_fraction_changed_core_bonds", "bond_fraction_other_bonds",
                "LCRC_GRAD_share_var", "LCRC_GRAD_share_attach", "LCRC_GRAD_share_core",
                "GINE_GRAD_share_var", "GINE_GRAD_share_attach", "GINE_GRAD_share_core",
                "LCRC_LOCAL_share_var", "LCRC_LOCAL_share_boundary", "LCRC_LOCAL_share_core"]
    columns = list(dict.fromkeys(columns))
    C.write_csv(out_dir / "aux_endpoint_results.csv", rows, columns)
    summary = summarise(rows)
    paired = paired_vs_single_model(rows)
    payload: dict[str, Any] = {
        "round_id": C.ROUND_ID, "created_utc": C.utc_now(),
        "n_endpoint_rows": len(rows),
        "n_endpoints": len(index),
        "n_pairs": len({str(r["pair_id"]) for r in rows}),
        "n_components": len({int(r["component"]) for r in rows}),
        "pool_sha256": C.sha256_file(pool_path),
        "summary": summary,
        "paired": paired,
        "pair_delta_localization": delta_localization(),
        "notes": [
            "edit region = structural difference between the two molecules, not causal property truth",
            "absence of Top-1 overlap alone is not evidence of unfaithfulness",
            "bootstrap unit = shared-endpoint component",
        ],
    }
    if pred is not None:
        payload["predictions"] = {k: v for k, v in pred["splits"].items()}
        payload["pair_predictions"] = {k: v["summary"] for k, v in pred["pairs"].items()}
        C.write_csv(out_dir / "aux_pair_predictions.csv",
                    [dict(row, family=family) for family, block in pred["pairs"].items() for row in block["rows"]],
                    ["family", "pair_id", "split_pair", "train_train", "component", "delta_pred", "delta_true",
                     "abs_error"])
        payload["pair_prediction_rows"] = {k: v["rows"] for k, v in pred["pairs"].items()}
    C.write_json(out_dir / "aux_summary.json", payload)
    C.write_csv(out_dir / "aux_paired_differences.csv", paired)
    print(f"[aux] endpoints={len(index)} rows={len(rows)}", flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", default=str(C.POOL_DIR))
    parser.add_argument("--select-dir", default=str(C.SELECT_DIR))
    parser.add_argument("--out-dir", default=str(C.ANALYSIS_DIR))
    parser.add_argument("--pred", default=str(C.REPLAY_DIR / "lipo_predictions.npz"))
    args = parser.parse_args()
    out = run(pool_dir=args.pool_dir, select_dir=args.select_dir, out_dir=args.out_dir, pred_path=args.pred)
    print(json.dumps({k: out[k] for k in ("n_endpoints", "n_pairs", "n_components")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
