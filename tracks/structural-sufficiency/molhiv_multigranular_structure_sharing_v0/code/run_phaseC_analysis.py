"""Phase C: structure ↔ task analyses, no new models (PROTOCOL.md §5).

Reads the frozen vocabulary, the Phase B fits and the Phase D/E predictions, and
writes:

* ``analysis/support_bins.csv``            the four Train-scaffold-frequency bins
* ``analysis/coefficient_bins.csv``        B2 vs B3 effective FULL coefficients per bin
* ``analysis/case_families.json``          12 Train-selected COARSE families + examples
* ``analysis/case_families.csv``           the same, flat
* ``analysis/subgroup_generalisation.csv`` per-molecule low-support/unseen subgroups
* ``analysis/bootstrap_contrasts.json``    paired scaffold bootstrap for main contrasts
* ``analysis/predictions/*.npz``           per-molecule logits used by the bootstrap
* ``analysis/figures/*.png``               real molecules with the matched substructure highlighted
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy.sparse as sp

import msds_common as C
import msds_data as D
import msds_enum as E
import msds_stats as ST


def load_all() -> dict[str, Any]:
    with np.load(C.CACHE / "design_CF.npz") as z:
        n_rows, n_coarse, n_full = int(z["n_rows"]), int(z["n_coarse"]), int(z["n_full"])
        Cm = sp.csr_matrix((z["C_data"], z["C_indices"], z["C_indptr"]), shape=(n_rows, n_coarse))
        Fm = sp.csr_matrix((z["F_data"], z["F_indices"], z["F_indptr"]), shape=(n_rows, n_full))
    with np.load(C.CACHE / "vocab.npz") as z:
        vocab = {k: np.asarray(z[k]) for k in z.files}
    with np.load(C.CACHE / "phaseA_instances.npz", allow_pickle=True) as z:
        discovered = {
            "full_indptr": z["full_indptr"], "full_indices": z["full_indices"], "full_data": z["full_data"],
            "full_keys": [bytes(k) for k in z["full_keys"]],
            "coarse_keys": [bytes(k) for k in z["coarse_keys"]],
            "full_family": np.asarray(z["full_family"]),
            "coarse_family": np.asarray(z["coarse_family"]),
            "full_parent": np.asarray(z["full_parent"]),
            "examples": [list(x) for x in z["examples"]],
            "n_instances_family": np.asarray(z["n_instances_family"]),
        }
    phase_b = C.read_json(C.ANALYSIS / "phaseB_linear.json")
    with np.load(C.CACHE / "phaseB_coefficients.npz") as z:
        coefs = {k: np.asarray(z[k], dtype=np.float64) for k in z.files}
    with np.load(C.CACHE / "phaseB_predictions.npz") as z:
        preds = {k: np.asarray(z[k]) for k in z.files}
    return {"Cm": Cm, "Fm": Fm, "vocab": vocab, "discovered": discovered,
            "phase_b": phase_b, "coefs": coefs, "preds": preds}


def support_bins(m_f: np.ndarray) -> tuple[np.ndarray, list[float]]:
    q = np.percentile(m_f, [25, 50, 75])
    edges = [float(m_f.min()) - 0.5, float(q[0]), float(q[1]), float(q[2]), float(m_f.max()) + 0.5]
    bins = np.digitize(m_f, q, right=False)
    return bins, edges


def main() -> None:
    C.ensure_dirs()
    t0 = time.perf_counter()
    created = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    molecules = D.load_molecules()
    splits = {k: np.asarray(v, dtype=np.int64) for k, v in molecules.split.items()}
    y = molecules.label.astype(np.int64)
    scaffolds, scaffold_meta = D.load_scaffolds(molecules)
    uniq_scaf = {s: i for i, s in enumerate(sorted(set(scaffolds)))}
    group = np.asarray([uniq_scaf[s] for s in scaffolds], dtype=np.int64)
    data = load_all()
    Cm, Fm, vocab, disc = data["Cm"], data["Fm"], data["vocab"], data["discovered"]
    full_ids = vocab["full_ids"].astype(np.int64)
    coarse_ids = vocab["coarse_ids"].astype(np.int64)
    a = vocab["support_weight"].astype(np.float64)
    a_null = vocab["support_weight_null"].astype(np.float64)
    parent = vocab["full_parent_in_vocab"].astype(np.int64)
    full_family = vocab["full_family"].astype(np.int64)
    phase_b = data["phase_b"]

    # ---- per-type train statistics come from the frozen vocabulary tables
    full_df = np.asarray([int(r["document_frequency_train"]) for r in _registry("full")], dtype=np.int64)
    full_scaf = np.asarray([int(r["scaffold_frequency_train"]) for r in _registry("full")], dtype=np.int64)
    full_pos = np.asarray([int(r["positive_molecules_train"]) for r in _registry("full")], dtype=np.int64)
    full_neg = np.asarray([int(r["negative_molecules_train"]) for r in _registry("full")], dtype=np.int64)
    coarse_df = np.asarray([int(r["document_frequency_train"]) for r in _registry("coarse")], dtype=np.int64)
    coarse_scaf = np.asarray([int(r["scaffold_frequency_train"]) for r in _registry("coarse")], dtype=np.int64)
    coarse_pos = np.asarray([int(r["positive_molecules_train"]) for r in _registry("coarse")], dtype=np.int64)
    coarse_neg = np.asarray([int(r["negative_molecules_train"]) for r in _registry("coarse")], dtype=np.int64)

    # ================= §6.1 support bins =================
    bins, edges = support_bins(full_scaf.astype(np.float64))
    sel_b2 = phase_b["arms"]["B2"]["selected"]["candidate_C"]
    sel_b3 = phase_b["arms"]["B3"]["selected"]["candidate_C"]
    w_b2 = data["coefs"][f"B2|{sel_b2}"]
    w_b3 = data["coefs"][f"B3|{sel_b3}"].copy()
    n_morgan = data["phase_b"]["M"]["n_columns"]
    n_coarse = Cm.shape[1]
    v_b2 = w_b2[n_morgan + n_coarse:]
    v_b3 = w_b3[n_morgan + n_coarse:] / np.sqrt(a)  # effective coefficient on the original F columns
    v_b3_null = None
    if f"B3-NULL|{phase_b['arms']['B3-NULL']['selected']['candidate_C']}" in data["coefs"]:
        sel_b3n = phase_b["arms"]["B3-NULL"]["selected"]["candidate_C"]
        v_b3_null = data["coefs"][f"B3-NULL|{sel_b3n}"][n_morgan + n_coarse:] / np.sqrt(a_null)

    # unscaled instance counts per type (counts, not the scaled design values)
    with np.load(C.CACHE / "phaseA_instances.npz", allow_pickle=True) as z:
        full_all = sp.csr_matrix((z["full_data"], z["full_indices"], z["full_indptr"]),
                                 shape=(int(z["n_molecules"]), int(z["full_indices"].max()) + 1))
    counts_full_selected = np.asarray(full_all[:, full_ids].sum(axis=0)).reshape(-1).astype(np.int64)

    support_rows = []
    coef_rows = []
    for b in range(4):
        mask = bins == b
        row: dict[str, Any] = {
            "bin": b,
            "scaffold_frequency_min": (edges[b] if b else 0),
            "scaffold_frequency_max": edges[b + 1] - 0.5,
            "n_full_types": int(mask.sum()),
            "n_instances_all_molecules": int(counts_full_selected[mask].sum()),
            "mean_document_frequency": float(full_df[mask].mean()) if mask.any() else None,
            "mean_scaffold_frequency": float(full_scaf[mask].mean()) if mask.any() else None,
            "mean_support_weight_a_f": float(a[mask].mean()) if mask.any() else None,
        }
        # instance coverage on each split (unscaled instance counts of the retained types)
        for split_name, rows in splits.items():
            per_type = np.asarray(full_all[rows][:, full_ids].sum(axis=0)).reshape(-1)
            row[f"instances_{split_name}"] = int(per_type[mask].sum())
        row["share_of_retained_instances_train"] = row["instances_train"] / max(float(counts_full_selected.sum()), 1.0)
        # what fraction of *that split's* enumerated instances land in this bin
        row["share_of_split_instances_train"] = row["instances_train"] / max(int(np.asarray(full_all[splits["train"]].sum())), 1)
        row["share_of_split_instances_valid"] = row["instances_valid"] / max(int(np.asarray(full_all[splits["valid"]].sum())), 1)
        row["share_of_split_instances_test"] = row["instances_test"] / max(int(np.asarray(full_all[splits["test"]].sum())), 1)
        support_rows.append(row)

        coef_rows.append({
            "bin": b,
            "n_full_types": int(mask.sum()),
            "mean_abs_v_B2": float(np.abs(v_b2[mask]).mean()) if mask.any() else None,
            "mean_abs_v_B3": float(np.abs(v_b3[mask]).mean()) if mask.any() else None,
            "mean_abs_v_B3_NULL": float(np.abs(v_b3_null[mask]).mean()) if (v_b3_null is not None and mask.any()) else None,
            "ratio_mean_abs_B3_over_B2": float(np.abs(v_b3[mask]).mean() / max(np.abs(v_b2[mask]).mean(), 1e-12)) if mask.any() else None,
            "median_abs_v_B2": float(np.median(np.abs(v_b2[mask]))) if mask.any() else None,
            "median_abs_v_B3": float(np.median(np.abs(v_b3[mask]))) if mask.any() else None,
            "l2_mass_B2": float((v_b2[mask] ** 2).sum()),
            "l2_mass_B3": float((v_b3[mask] ** 2).sum()),
            "mean_abs_change_B3_minus_B2": float((np.abs(v_b3[mask]) - np.abs(v_b2[mask])).mean()) if mask.any() else None,
            "share_of_types_shrunk_by_B3": float((np.abs(v_b3[mask]) < np.abs(v_b2[mask])).mean()) if mask.any() else None,
        })
    _write_csv(C.ANALYSIS / "support_bins.csv", support_rows)
    _write_csv(C.ANALYSIS / "coefficient_bins.csv", coef_rows)

    # ================= §6.2 case families =================
    children_of = {p: np.flatnonzero(parent == p) for p in range(n_coarse)}
    multi = {p: ch for p, ch in children_of.items() if ch.size >= 2}
    correction = np.zeros((molecules.n_molecules,), dtype=np.float64)
    per_family_std: dict[int, float] = {}
    train_rows = splits["train"]
    Fcsc = Fm.tocsc()
    for p, ch in children_of.items():
        if ch.size == 0:
            continue
        contrib = np.asarray(Fcsc[:, ch] @ v_b2[ch]).reshape(-1)
        per_family_std[p] = float(contrib[train_rows].std())
    support_spread = {p: float(full_scaf[ch].max() / max(full_scaf[ch].min(), 1)) for p, ch in multi.items()}

    chosen: list[tuple[str, int]] = []
    cat_a = sorted(multi, key=lambda p: (-int(coarse_df[p]), -int(ch.size)))[:4]
    chosen += [("A_high_frequency_multi_child", p) for p in cat_a]
    used = set(cat_a)
    cat_b = [p for p in sorted(per_family_std, key=lambda p: -per_family_std[p])
             if p not in used and children_of[p].size >= 2][:4]
    chosen += [("B_large_train_correction_variation", p) for p in cat_b]
    used |= set(cat_b)
    cat_c = [p for p in sorted(support_spread, key=lambda p: -support_spread[p])
             if p not in used and children_of[p].size >= 3][:4]
    chosen += [("C_child_support_spread", p) for p in cat_c]
    used |= set(cat_c)
    # fill up to 12 from the remaining multi-child families if any category was short
    for p in sorted(multi, key=lambda p: (-int(coarse_df[p]))):
        if len(chosen) >= 12:
            break
        if p not in used:
            chosen.append(("D_fill", p))
            used.add(p)

    case_payload = {
        "created_utc": created,
        "selection_uses_only": "Train features + Train labels + fitted B2 parameters; no Test information",
        "categories": {
            "A_high_frequency_multi_child": "highest Train document frequency among COARSE parents with >=2 retained FULL children",
            "B_large_train_correction_variation": "highest std over Train molecules of the family's FULL correction sum_f F_f v_f (B2 coefficients)",
            "C_child_support_spread": "largest max/min Train scaffold frequency ratio among children (>=3 children)",
        },
        "families": [],
        "figures": [],
    }
    fig_dir = C.FIGURES
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows_flat = []
    order_checks: list[bool] = []
    for fam_idx, (category, p) in enumerate(chosen):
        ch = children_of[p]
        examples = []
        mol_examples: list[dict[str, Any]] = []
        for f in ch[:4]:
            gid_full = int(full_ids[f])
            ex = disc["examples"][gid_full][:3]
            examples.append({
                "full_column": int(f),
                "family": C.FAMILIES[int(full_family[f])],
                "signature": E.describe_key(_key(disc["full_keys"][gid_full]), "FULL"),
                "document_frequency_train": int(full_df[f]),
                "scaffold_frequency_train": int(full_scaf[f]),
                "positive_molecules_train": int(full_pos[f]),
                "negative_molecules_train": int(full_neg[f]),
                "support_weight_a_f": float(a[f]),
                "v_B2": float(v_b2[f]),
                "v_B3": float(v_b3[f]),
                "instances": [{"graph_index": int(g), "nodes": [int(n) for n in nodes]} for g, nodes in ex],
            })
            for gid, nodes in ex[:1]:
                mol_examples.append({"graph_index": int(gid), "nodes": [int(n) for n in nodes],
                                     "smiles": molecules.smiles[gid], "label": int(y[gid]),
                                     "full_column": int(f), "scaffold": scaffolds[gid]})
        # make sure at least two *different* molecules are shown for the family
        for f in ch:
            gid_full = int(full_ids[f])
            for gid, nodes in disc["examples"][gid_full][:4]:
                if len({m["graph_index"] for m in mol_examples}) >= 2:
                    break
                if all(m["graph_index"] != int(gid) for m in mol_examples):
                    mol_examples.append({"graph_index": int(gid), "nodes": [int(n) for n in nodes],
                                         "smiles": molecules.smiles[gid], "label": int(y[gid]),
                                         "full_column": int(f), "scaffold": scaffolds[gid]})
        entry = {
            "case_index": fam_idx,
            "category": category,
            "coarse_column": int(p),
            "family": C.FAMILIES[int(vocab["coarse_family"][p])],
            "signature": E.describe_key(_key(disc["coarse_keys"][int(coarse_ids[p])]), "COARSE"),
            "document_frequency_train": int(coarse_df[p]),
            "scaffold_frequency_train": int(coarse_scaf[p]),
            "positive_molecules_train": int(coarse_pos[p]),
            "negative_molecules_train": int(coarse_neg[p]),
            "u_B2": float(w_b2[n_morgan + p]),
            "u_B3": float(w_b3[n_morgan + p]),
            "children": examples,
            "molecules": mol_examples,
            "train_correction_std": float(per_family_std.get(int(p), float("nan"))),
            "child_support_spread": float(support_spread.get(int(p), float("nan"))),
        }
        case_payload["families"].append(entry)
        rows_flat.append({
            "case_index": fam_idx, "category": category, "coarse_column": int(p),
            "family": entry["family"], "node_signature": entry["signature"]["node_signature"],
            "bond_signature": entry["signature"]["bond_signature"],
            "document_frequency_train": entry["document_frequency_train"],
            "scaffold_frequency_train": entry["scaffold_frequency_train"],
            "n_children": int(ch.size), "u_B2": entry["u_B2"],
            "train_correction_std": entry["train_correction_std"],
            "child_support_spread": entry["child_support_spread"],
            "canonical_key": entry["signature"]["canonical_key"],
        })
        # figure: draw every distinct example molecule with the match highlighted
        for m_i, m in enumerate(mol_examples[:3]):
            path = fig_dir / f"case{fam_idx:02d}_mol{m_i}_{m['graph_index']}.png"
            info = _draw(C, molecules, m["graph_index"], m["nodes"], path,
                         title=f"case {fam_idx} ({category.split('_')[0]}) mol {m['graph_index']}"
                               f" | {entry['signature']['node_signature']}"
                               f" | match {tuple(m['nodes'])}")
            order_checks.append(bool(info.get("rdkit_atom_order_matches_raw_features")))
            case_payload["figures"].append({**info, "path": C.rel(path), "nodes": m["nodes"],
                                            "case_index": fam_idx, "smiles": m["smiles"]})
    case_payload["rdkit_atom_order_matches_raw_features"] = bool(order_checks) and all(order_checks)
    case_payload["n_figures"] = len(case_payload["figures"])
    _write_csv(C.ANALYSIS / "case_families.csv", rows_flat)
    C.write_json(C.ANALYSIS / "case_families.json", case_payload)

    # ================= §6.3 subgroup generalisation =================
    selected_full_mask = np.zeros(full_all.shape[1], dtype=bool)
    selected_full_mask[full_ids] = True
    bin_of_full = np.full(full_all.shape[1], -1, dtype=np.int64)
    bin_of_full[full_ids] = bins
    low_support_cols = full_ids[bins == 0]
    n_inst_total = np.asarray(full_all.sum(axis=1)).reshape(-1).astype(np.float64)
    n_low = np.asarray(full_all[:, low_support_cols].sum(axis=1)).reshape(-1).astype(np.float64)
    n_unseen = n_inst_total - np.asarray(full_all[:, full_ids].sum(axis=1)).reshape(-1).astype(np.float64)
    ratio = (n_low + n_unseen) / np.maximum(n_inst_total, 1.0)
    unseen_any = n_unseen > 0
    edges_ratio = [0.0, 0.25, 0.5, 0.75, 1.0000001]
    ratio_bin = np.clip(np.digitize(ratio, edges_ratio[1:-1], right=False), 0, 3)

    # every model's per-molecule logits on the same rows
    logits: dict[str, dict[str, np.ndarray]] = {}
    for arm, res in phase_b["arms"].items():
        c_val = res["selected"]["candidate_C"]
        logits[arm] = {s: data["preds"][f"{arm}|{c_val}|{s}"].astype(np.float64) for s in ("valid", "test")}
    xgb_path = C.CACHE / "phaseD_xgb_struct_predictions.npz"
    if xgb_path.exists():
        with np.load(xgb_path) as z:
            logits["XGB_STRUCT"] = {"valid": z["valid"].astype(np.float64), "test": z["test"].astype(np.float64)}
    hist_path = C.HIST_BASELINE_DIR
    logits["historical_XGB"] = {}
    for split_name in ("valid", "test"):
        with np.load(hist_path / f"ogbg-molhiv_{split_name}_predictions.npz") as z:
            logits["historical_XGB"][split_name] = np.asarray(z["pred"]).reshape(-1).astype(np.float64)
    for arm_dir in sorted((C.ANALYSIS / "phaseE").glob("*_seed0.json")) if (C.ANALYSIS / "phaseE").exists() else []:
        payload = C.read_json(arm_dir)
        pred_path = arm_dir.with_suffix(".predictions.npz")
        if pred_path.exists():
            with np.load(pred_path) as z:
                logits[payload["arm"]] = {"valid": z["valid"].astype(np.float64), "test": z["test"].astype(np.float64)}

    subgroup_rows = []
    for split_name in ("valid", "test"):
        rows = splits[split_name]
        groups = {
            "ratio_0.00-0.25": ratio_bin[rows] == 0,
            "ratio_0.25-0.50": ratio_bin[rows] == 1,
            "ratio_0.50-0.75": ratio_bin[rows] == 2,
            "ratio_0.75-1.00": ratio_bin[rows] == 3,
            "any_unseen_full": unseen_any[rows],
            "no_unseen_full": ~unseen_any[rows],
            "all": np.ones(rows.size, dtype=bool),
        }
        for arm, per_split in logits.items():
            for gname, mask in groups.items():
                m = ST.auc_ap(y[rows][mask], per_split[split_name][mask])
                subgroup_rows.append({"split": split_name, "arm": arm, "group": gname,
                                      **{k: v for k, v in m.items() if k != "status"},
                                      "status": m.get("status", "OK"),
                                      "evidence_label": "EXPOSED_DEVELOPMENT_TEST" if split_name == "test" else "DEVELOPMENT_VALID"})
    _write_csv(C.ANALYSIS / "subgroup_generalisation.csv", subgroup_rows)

    # ================= bootstrap contrasts =================
    contrasts = [
        ("B1", "B0"), ("B2", "B1"), ("B2", "BF"), ("B2", "B2-PNULL"),
        ("B3", "B2"), ("B3", "B2-U"), ("B3", "B3-NULL"),
        ("XGB_STRUCT", "historical_XGB"),
    ]
    if "G-M" in logits and "G-STRUCT-U" in logits:
        contrasts.append(("G-STRUCT-U", "G-M"))
        contrasts.append(("G-M", "historical_XGB"))
    boot_payload: dict[str, Any] = {"created_utc": created, "n_bootstrap": 1000, "seed": C.SEED,
                                    "grouping": "Bemis-Murcko scaffold group resampling (whole groups)",
                                    "contrasts": []}
    pred_dir = C.ANALYSIS / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for arm, per_split in logits.items():
        np.savez_compressed(pred_dir / f"{arm}.npz",
                            valid=per_split["valid"].astype(np.float32),
                            test=per_split["test"].astype(np.float32),
                            valid_rows=splits["valid"], test_rows=splits["test"], y=y)
    for arm_a, arm_b in contrasts:
        if arm_a not in logits or arm_b not in logits:
            continue
        for split_name in ("valid", "test"):
            rows = splits[split_name]
            boot = ST.paired_scaffold_bootstrap(
                y[rows], logits[arm_a][split_name], logits[arm_b][split_name], group[rows],
                label_a=arm_a, label_b=arm_b,
            )
            boot["split"] = split_name
            boot["evidence_label"] = "EXPOSED_DEVELOPMENT_TEST" if split_name == "test" else "DEVELOPMENT_VALID"
            boot_payload["contrasts"].append(boot)
    C.write_json(C.ANALYSIS / "bootstrap_contrasts.json", boot_payload)

    _plot_support_bins(support_rows, coef_rows)
    _plot_subgroups(subgroup_rows)

    # ---- small bookkeeping payload
    meta = {
        "round_id": C.ROUND_ID,
        "phase": "C: structure ↔ task analyses",
        "created_utc": created,
        "support_bin_edges_train_scaffold_frequency": edges,
        "support_bin_counts": {str(b): int((bins == b).sum()) for b in range(4)},
        "low_support_definition": "bottom Train-scaffold-frequency quartile of retained FULL types",
        "unseen_definition": "FULL type not in the frozen vocabulary (filtered out or absent from Train)",
        "subgroup_ratio": "fraction of a molecule's enumerated instances whose FULL type is low-support or unseen",
        "subgroup_ratio_edges": edges_ratio,
        "models_with_predictions": sorted(logits),
        "scaffold_groups": {"n_groups": len(uniq_scaf), "acyclic_group": scaffold_meta["n_empty_acyclic_scaffold"]},
        "seconds": float(time.perf_counter() - t0),
        "git": C.git_state(),
    }
    C.write_json(C.ANALYSIS / "phaseC_meta.json", meta)
    print(f"[phaseC] bins {meta['support_bin_counts']} | models {meta['models_with_predictions']}", flush=True)
    print(f"[phaseC] wrote support/coefficient/case/subgroup/bootstrap tables in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)


def _plot_support_bins(support_rows: list[dict[str, Any]], coef_rows: list[dict[str, Any]]) -> None:
    """Two panels: where the instances live, and how B3 reweights the coefficients."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=200)
    xs = np.arange(4)
    width = 0.26
    for offset, split_name, colour in ((-width, "train", "#4c72b0"), (0.0, "valid", "#dd8452"),
                                       (width, "test", "#55a868")):
        vals = [float(row[f"share_of_split_instances_{split_name}"]) for row in support_rows]
        axes[0].bar(xs + offset, vals, width=width, label=split_name, color=colour)
    axes[0].set_xticks(xs, [f"bin {r['bin']}\n({r['n_full_types']} types)" for r in support_rows])
    axes[0].set_ylabel("share of that split's induced instances")
    axes[0].set_title("FULL types by Train scaffold-frequency quartile")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].plot(xs, [r["mean_abs_v_B2"] for r in coef_rows], "o-", label="B2 (plain L2)", color="#4c72b0")
    axes[1].plot(xs, [r["mean_abs_v_B3"] for r in coef_rows], "s-", label="B3 (support-weighted)", color="#c44e52")
    for i, r in enumerate(coef_rows):
        axes[1].annotate(f"×{r['ratio_mean_abs_B3_over_B2']:.2f}", (i, max(r["mean_abs_v_B2"], r["mean_abs_v_B3"])),
                         textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    axes[1].set_xticks(xs, [f"bin {r['bin']}" for r in coef_rows])
    axes[1].set_ylabel("mean |effective FULL coefficient|")
    axes[1].set_title("support weighting shrinks the tail, feeds the head")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "support_bins_and_coefficients.png", bbox_inches="tight")
    plt.close(fig)


def _plot_subgroups(rows: list[dict[str, Any]]) -> None:
    """Test-split AUC per low-support/unseen-ratio bin for the main arms."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = ["B0", "B2", "B3", "XGB_STRUCT", "G-STRUCT-U"]
    groups = [f"ratio_{lo}" for lo in ("0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00")]
    lookup = {(r["arm"], r["group"]): r for r in rows if r["split"] == "test"}
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    xs = np.arange(len(groups))
    for arm in arms:
        ys, ns = [], []
        for g in groups:
            row = lookup.get((arm, g))
            ys.append(float(row["roc_auc"]) if row and row["roc_auc"] else np.nan)
            ns.append(row["n"] if row else 0)
        ax.plot(xs, ys, "o-", label=arm, lw=1.4, ms=4)
    for i, g in enumerate(groups):
        row = lookup.get(("B0", g))
        if row is None or not row["roc_auc"]:
            ax.annotate("INSUFFICIENT\nSUPPORT", (i, 0.52), ha="center", fontsize=6.5, color="#888888")
        else:
            ax.annotate(f"n={row['n']}\npos={row['n_positive']}", (i, 0.505), ha="center", fontsize=6.5,
                        color="#888888")
    ax.set_xticks(xs, ["0–0.25", "0.25–0.50", "0.50–0.75", "0.75–1.00"])
    ax.set_xlabel("share of low-support or unseen FULL instances (Train-defined bins)")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("EXPOSED_DEVELOPMENT_TEST — directional only", fontsize=8)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "subgroup_generalisation_test.png", bbox_inches="tight")
    plt.close(fig)


def _registry(granularity: str) -> list[dict[str, str]]:
    """Registry rows, ordered by their column index (must match the design matrix)."""
    path = C.ANALYSIS / f"structure_registry_{granularity}.csv"
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: int(r["column_index"]))
    assert [int(r["column_index"]) for r in rows] == list(range(len(rows))), "registry column order"
    return rows


def _key(raw: bytes) -> tuple:
    return eval(raw.decode())  # noqa: S307 - our own repr tuple


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


#: element colours for the matplotlib depiction (no RDKit drawing backend needed:
#: the air-gapped runner image lacks libXrender, so we lay the molecule out with
#: ``rdDepictor`` and paint it ourselves)
_ELEMENT_COLOURS = {
    "C": "#3f3f3f", "N": "#2b5fd9", "O": "#d93025", "S": "#c9a227", "F": "#2e9e4f",
    "Cl": "#2e9e4f", "Br": "#8a4b1e", "I": "#6a1b9a", "P": "#e07b00", "B": "#e07b00",
    "Si": "#7a7a7a", "H": "#9e9e9e",
}


def _draw(C: Any, molecules: Any, graph_index: int, nodes: list[int], path: Path, *, title: str) -> dict[str, Any]:
    """Draw one real molecule and highlight the matched atoms (matplotlib, Agg)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDepictor

    RDLogger.DisableLog("rdApp.*")
    smiles = molecules.smiles[graph_index]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"graph_index": int(graph_index), "drawn": False, "reason": "unparsable smiles"}
    # the enumeration indexes nodes with the OGB/raw atom order; verify that the
    # RDKit re-parse yields the same atomic numbers before trusting the indexing
    from ogb.utils.features import allowable_features

    num_list = allowable_features["possible_atomic_num_list"]
    raw_z = [int(num_list[int(v)]) if isinstance(num_list[int(v)], int) else 0
             for v in molecules.atoms(graph_index)[:, 0]]
    par_z = [int(a.GetAtomicNum()) for a in mol.GetAtoms()]
    order_ok = raw_z == par_z
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    xy = np.asarray([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])
    hl = [int(n) for n in nodes if int(n) < mol.GetNumAtoms()] if order_ok else []
    fig, ax = plt.subplots(figsize=(4.6, 3.6), dpi=220)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], "-", color="#9a9a9a", lw=1.6, zorder=1)
    sym = [a.GetSymbol() for a in mol.GetAtoms()]
    cols = [_ELEMENT_COLOURS.get(s, "#d16ba5") for s in sym]
    sizes = [95 if i in hl else 62 for i in range(mol.GetNumAtoms())]
    ax.scatter(xy[:, 0], xy[:, 1], s=[15] * mol.GetNumAtoms(), c=cols, zorder=2)
    ax.scatter(xy[:, 0], xy[:, 1], s=sizes, facecolors=cols, edgecolors="white", linewidths=0.6, zorder=3)
    if hl:
        ax.scatter(xy[hl, 0], xy[hl, 1], s=np.asarray(sizes)[hl] + 260, facecolors="none",
                   edgecolors="#e4572e", linewidths=1.7, zorder=4)
    ax.set_title(title, fontsize=6.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"graph_index": int(graph_index), "drawn": True, "highlighted": hl,
            "rdkit_atom_order_matches_raw_features": bool(order_ok),
            "n_atoms": int(mol.GetNumAtoms()), "n_highlighted": len(hl)}


if __name__ == "__main__":
    main()
