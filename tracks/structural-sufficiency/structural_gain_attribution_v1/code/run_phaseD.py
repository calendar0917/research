"""Phase D — understanding the signal from frozen predictions and fitted trees.

No new model.  Everything here reads

* the fitted boosters of ``X-C`` / ``X-F`` / ``X-PRES`` (``runs/phaseA/*_model.json``),
* the frozen per-molecule predictions (``analysis/predictions/*.npz``),
* the frozen vocabulary/registry and the Phase-0.2 per-molecule OOV statistics,
* the real molecules of ogbg-molhiv (attributes, SMILES, labels).

Outputs
  analysis/phaseD_structure_classes.csv    gain per block × family × support bin
  analysis/phaseD_molecules.csv            per-molecule deltas and coverage features
  analysis/phaseD_subgroups.json           binned improvement analysis with bootstrap CIs
  analysis/case_structures.json            3 supported types + 3 coarse/fine families
  analysis/case_molecules.json             3 improved + 3 worsened molecules
  analysis/figures/*.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import sga_common as K

K.add_prev_code_path()

import msds_data as D  # noqa: E402
import sga_chem as CHEM  # noqa: E402

FAMILIES = ("size-2", "size-3", "size-4", "cycle-5", "cycle-6")


def _p(logit: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logit, dtype=np.float64), -60, 60)))


# ---------------------------------------------------------------------------
# D1 structures
# ---------------------------------------------------------------------------


def load_vocab() -> dict[str, np.ndarray]:
    with np.load(K.MSDS_DATA / "vocab.npz") as z:
        out = {k: np.asarray(z[k]) for k in z.files}
    return out


def booster_gains(path: Path) -> dict[int, float]:
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(path)
    gain = booster.get_score(importance_type="total_gain")
    return {int(k[1:]): float(v) for k, v in gain.items()}


def iter_trees(path: Path):
    """Yield ``(tree_index, [(node_id, split_feature, split_condition)])`` for leaf-reaching splits."""
    payload = json.loads(path.read_text())
    for t_i, tree in enumerate(payload["trees"]):
        yield t_i, tree


def structure_classes() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vocab = load_vocab()
    coarse_family = vocab["coarse_family"]
    full_family = vocab["full_family"]
    full_scaf = None
    # Train scaffold frequency of each retained FULL type, from the MSDS-v0 registry
    reg = list(csv.DictReader((K.MSDS_ANALYSIS / "structure_registry_full.csv").open()))
    full_scaf = np.asarray([int(r["scaffold_frequency_train"]) for r in reg], dtype=np.int64)
    full_df = np.asarray([int(r["document_frequency_train"]) for r in reg], dtype=np.int64)

    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for arm, block, offset, size in (("X-C", "C", 2258, 754), ("X-F", "F", 2258, 3652),
                                     ("X-PRES", "PRES", 2258, 4406)):
        gains = booster_gains(K.RUNS / "phaseA" / f"{arm}_model.json")
        struct = {c - offset: g for c, g in gains.items() if c >= offset}
        total_struct = sum(struct.values())
        per_family: dict[str, dict[str, float]] = {f: {"n_used": 0, "n_columns": 0, "gain": 0.0}
                                                   for f in FAMILIES}
        for col, g in struct.items():
            if block == "C":
                fam = FAMILIES[int(coarse_family[col])]
                per_family[fam]["n_columns"] += 1
            elif block == "F":
                fam = FAMILIES[int(full_family[col])]
                per_family[fam]["n_columns"] += 1
            else:  # PRES = C(754) then F(3652)
                if col < 754:
                    fam = FAMILIES[int(coarse_family[col])]
                else:
                    fam = FAMILIES[int(full_family[col - 754])]
            per_family[fam]["n_used"] += 1
            per_family[fam]["gain"] += g
        for fam, d in per_family.items():
            rows.append({"arm": arm, "block": block, "family": fam,
                         "n_columns_used": d["n_used"],
                         "gain": d["gain"],
                         "gain_share_of_structure": (d["gain"] / total_struct) if total_struct else None})
        # support bins of the FULL types used by X-F
        if block == "F":
            quart = np.quantile(full_scaf, [0.25, 0.5, 0.75])
            bins = np.digitize(full_scaf, quart)
            for b in range(4):
                cols = [c for c in struct if bins[c] == b]
                rows.append({"arm": arm, "block": "F", "family": f"support_q{b + 1}",
                             "n_columns_used": len(cols),
                             "gain": float(sum(struct[c] for c in cols)),
                             "gain_share_of_structure": (float(sum(struct[c] for c in cols)) / total_struct)
                             if total_struct else None,
                             "support_bin_edges_train_scaffold_frequency": [float(v) for v in quart]})
        detail[arm] = {
            "total_structural_gain": float(total_struct),
            "n_structural_columns": int(size),
            "n_columns_used": int(len(struct)),
            "top": sorted(({"block_column": int(c), "gain": float(g),
                            "family": FAMILIES[int((coarse_family if block == "C" else full_family)[
                                c if block != "PRES" or c < 754 else c - 754])]}
                           for c, g in struct.items()), key=lambda r: -r["gain"])[:20],
        }
    return rows, detail


def case_structures() -> dict[str, Any]:
    """Three well-supported specific types + three real coarse/fine families."""
    vocab = load_vocab()
    full_ids = vocab["full_ids"]
    coarse_ids = vocab["coarse_ids"]
    full_parent = vocab["full_parent_in_vocab"]
    reg_full = list(csv.DictReader((K.MSDS_ANALYSIS / "structure_registry_full.csv").open()))
    reg_coarse = list(csv.DictReader((K.MSDS_ANALYSIS / "structure_registry_coarse.csv").open()))
    regv = K.ANALYSIS / "registry_v1"
    corrected_full = list(csv.DictReader((regv / "structure_registry_full_v1.csv").open()))
    corrected_coarse = list(csv.DictReader((regv / "structure_registry_coarse_v1.csv").open()))

    gains_f = booster_gains(K.RUNS / "phaseA" / "X-F_model.json")
    gains_c = booster_gains(K.RUNS / "phaseA" / "X-C_model.json")
    struct_f = {c - 2258: g for c, g in gains_f.items() if c >= 2258}
    struct_c = {c - 2258: g for c, g in gains_c.items() if c >= 2258}

    df = np.asarray([int(r["document_frequency_train"]) for r in reg_full], dtype=np.int64)
    scaf = np.asarray([int(r["scaffold_frequency_train"]) for r in reg_full], dtype=np.int64)
    # well supported = high Train document frequency AND high Train scaffold frequency
    supported = np.argsort(-(df * 1.0) - 0.001 * scaf)[:400]
    supported = sorted(supported.tolist(), key=lambda c: -struct_f.get(c, 0.0))
    picks: list[dict[str, Any]] = []
    seen_family: dict[int, int] = {}
    for c in supported:
        fam = int(vocab["full_family"][c])
        if seen_family.get(fam, 0) >= 1:
            continue
        seen_family[fam] = seen_family.get(fam, 0) + 1
        picks.append(_type_card(c, "FULL", reg_full, corrected_full, full_ids, struct_f.get(c, 0.0),
                                int(full_parent[c]), reg_coarse, corrected_coarse, coarse_ids, struct_c))
        if len(picks) == 3:
            break

    families: list[dict[str, Any]] = []
    child_count = np.bincount(full_parent, minlength=coarse_ids.size)
    coarse_gain = np.asarray([struct_c.get(int(p), 0.0) for p in range(coarse_ids.size)])
    ordered = np.argsort(-coarse_gain)[:400]
    used_families: set[str] = set()
    for p in ordered:
        fam_label = FAMILIES[int(vocab["coarse_family"][p])]
        if fam_label in used_families:
            continue
        if child_count[p] < 2:
            continue
        used_families.add(fam_label)
        if len(families) == 3:
            break
        children = np.flatnonzero(full_parent == p)
        children = children[np.argsort(-np.asarray([struct_f.get(int(c), 0.0) for c in children]))][:4]
        families.append({
            "coarse_column": int(p),
            "coarse_total_gain_in_fitted_tree": float(struct_c.get(int(p), 0.0)),
            "coarse_card": _type_card(int(p), "COARSE", reg_coarse, corrected_coarse, coarse_ids,
                                      struct_c.get(int(p), 0.0), None, reg_coarse, corrected_coarse,
                                      coarse_ids, struct_c),
            "n_children_in_vocabulary": int(child_count[p]),
            "children": [
                _type_card(int(c), "FULL", reg_full, corrected_full, full_ids,
                           struct_f.get(int(c), 0.0), int(full_parent[c]), reg_coarse,
                           corrected_coarse, coarse_ids, struct_c)
                for c in children
            ],
        })
    return {"supported_types": picks, "families": families}


def _type_card(col: int, granularity: str, reg_old, reg_new, ids, gain, parent_col,
               reg_coarse, reg_coarse_new, coarse_ids, gains_c) -> dict[str, Any]:
    old = reg_old[col]
    new = reg_new[col]
    return {
        "granularity": granularity,
        "column": int(col),
        "family": old["family"],
        "document_frequency_train": int(old["document_frequency_train"]),
        "scaffold_frequency_train": int(old["scaffold_frequency_train"]),
        "positive_molecules_train": int(old["positive_molecules_train"]),
        "negative_molecules_train": int(old["negative_molecules_train"]),
        "total_gain_in_fitted_tree": float(gain),
        "corrected_node_signature": new["node_signature"],
        "corrected_bond_signature": new["bond_signature"],
        "corrected_element_sequence": new["element_sequence"],
        "corrected_formal_charge_sequence": new["formal_charge_sequence"],
        "corrected_hybridization_sequence": new["hybridization_sequence"],
        "predecessor_node_signature": old["node_signature"],
        "predecessor_bond_signature": old["bond_signature"],
        "canonical_key": new["canonical_key"],
        "parent_coarse_column": parent_col,
        "parent_coarse_node_signature": (reg_coarse_new[int(parent_col)]["node_signature"]
                                         if parent_col is not None else None),
    }


# ---------------------------------------------------------------------------
# D2 molecules
# ---------------------------------------------------------------------------


def molecule_table() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    molecules = D.load_molecules()
    splits = {k: np.asarray(v, dtype=np.int64) for k, v in molecules.split.items()}
    y = molecules.label.astype(np.float64)
    with np.load(K.CACHE / "oov_per_molecule.npz") as z:
        oov = {k: np.asarray(z[k]) for k in z.files}
    preds = {a: {s: np.load(K.ANALYSIS / "predictions" / f"{a}.npz")[s] for s in ("valid", "test")}
             for a in ("X0", "X-CF", "N0", "N-CF", "X-WL", "X-C")}
    total = np.maximum(oov["total"], 1.0)
    share_in = oov["in_vocab"] / total
    share_seen_not = oov["seen_not_frozen"] / total
    share_never = oov["never_seen"] / total

    # Train-defined bin edges for n_atoms
    n_atoms = molecules.n_atoms.astype(np.float64)
    edges = np.quantile(n_atoms[splits["train"]], [0.2, 0.4, 0.6, 0.8])
    size_bin = np.digitize(n_atoms, edges)
    cov_edges = [0.0, 0.25, 0.5, 0.75, 1.0000001]
    cov_bin = np.clip(np.digitize(share_in, cov_edges[1:-1], right=False), 0, 3)
    never_edges = [0.0, 0.02, 0.1, 0.3, 1.0000001]
    never_bin = np.clip(np.digitize(share_never, never_edges[1:-1], right=False), 0, 3)

    rows: list[dict[str, Any]] = []
    for split in ("valid", "test"):
        for i, r in enumerate(splits[split]):
            rows.append({
                "split": split, "row": int(r), "y": float(y[r]),
                "n_atoms": float(n_atoms[r]),
                "size_bin": int(size_bin[r]),
                "in_vocab_share": float(share_in[r]),
                "seen_not_frozen_share": float(share_seen_not[r]),
                "never_seen_share": float(share_never[r]),
                "coverage_bin": int(cov_bin[r]),
                "never_seen_bin": int(never_bin[r]),
                "p_X0": float(_p(preds["X0"][split][i])),
                "p_XCF": float(_p(preds["X-CF"][split][i])),
                "p_XWL": float(_p(preds["X-WL"][split][i])),
                "p_XC": float(_p(preds["X-C"][split][i])),
                "p_N0": float(_p(preds["N0"][split][i])),
                "p_NCF": float(_p(preds["N-CF"][split][i])),
                "delta_XCF": float(_p(preds["X-CF"][split][i]) - _p(preds["X0"][split][i])),
                "delta_XWL": float(_p(preds["X-WL"][split][i]) - _p(preds["X0"][split][i])),
                "delta_NCF": float(_p(preds["N-CF"][split][i]) - _p(preds["N0"][split][i])),
            })
    meta = {
        "size_bin_edges_train_n_atoms": [float(v) for v in edges],
        "coverage_bin_edges": cov_edges,
        "never_seen_bin_edges": never_edges,
        "size_bin_counts_train": np.bincount(size_bin[splits["train"]], minlength=5).tolist(),
    }
    return rows, meta


def subgroup_analysis(rows: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    by_split: dict[str, Any] = {}
    for split in ("valid", "test"):
        sub = [r for r in rows if r["split"] == split]
        out: dict[str, Any] = {}
        for key, n_bins in (("size_bin", 5), ("coverage_bin", 4), ("never_seen_bin", 4)):
            out[key] = []
            for b in range(n_bins):
                g = [r for r in sub if r[key] == b]
                if not g:
                    continue
                yy = np.asarray([r["y"] for r in g])
                entry = {"bin": b, "n": len(g), "n_positive": int(yy.sum()),
                         "mean_delta_XCF": float(np.mean([r["delta_XCF"] for r in g])),
                         "mean_delta_NCF": float(np.mean([r["delta_NCF"] for r in g])),
                         "mean_in_vocab_share": float(np.mean([r["in_vocab_share"] for r in g])),
                         "mean_n_atoms": float(np.mean([r["n_atoms"] for r in g]))}
                if yy.sum() >= 5 and (yy.size - yy.sum()) >= 5:
                    a = np.asarray([r["p_X0"] for r in g])
                    b2 = np.asarray([r["p_XCF"] for r in g])
                    entry["auc_X0"] = float(roc_auc_score(yy, a))
                    entry["auc_XCF"] = float(roc_auc_score(yy, b2))
                    entry["auc_delta_XCF"] = float(roc_auc_score(yy, b2) - roc_auc_score(yy, a))
                    entry["n_improved_predictions"] = int(sum(1 for r in g if r["delta_XCF"] > 0))
                    entry["n_worsened_predictions"] = int(sum(1 for r in g if r["delta_XCF"] < 0))
                else:
                    entry["status"] = "INSUFFICIENT_SUPPORT"
                out[key].append(entry)
        # within-size comparison of coverage bins (avoids reading size as an OOV effect)
        within = []
        for sb in range(5):
            for cb in range(4):
                g = [r for r in sub if r["size_bin"] == sb and r["coverage_bin"] == cb]
                if len(g) < 30:
                    continue
                yy = np.asarray([r["y"] for r in g])
                within.append({"size_bin": sb, "coverage_bin": cb, "n": len(g),
                               "n_positive": int(yy.sum()),
                               "mean_delta_XCF": float(np.mean([r["delta_XCF"] for r in g])),
                               "mean_delta_NCF": float(np.mean([r["delta_NCF"] for r in g]))})
        out["within_size_coverage"] = within
        by_split[split] = out
    return {"meta": meta, "by_split": by_split}


# ---------------------------------------------------------------------------
# D3 cases
# ---------------------------------------------------------------------------


def case_molecules(n_per_kind: int = 3) -> dict[str, Any]:
    molecules = D.load_molecules()
    splits = {k: np.asarray(v, dtype=np.int64) for k, v in molecules.split.items()}
    vocab = load_vocab()
    corrected_full = list(csv.DictReader(
        (K.ANALYSIS / "registry_v1" / "structure_registry_full_v1.csv").open()))
    corrected_coarse = list(csv.DictReader(
        (K.ANALYSIS / "registry_v1" / "structure_registry_coarse_v1.csv").open()))
    discovered_examples = _discovered_examples()
    pred_x0 = np.load(K.ANALYSIS / "predictions" / "X0.npz")
    pred_cf = np.load(K.ANALYSIS / "predictions" / "X-CF.npz")

    out: dict[str, Any] = {"improved": [], "worsened": []}
    for split in ("test",):
        rows = splits[split]
        y = molecules.label[rows]
        p0 = _p(pred_x0[split])
        p1 = _p(pred_cf[split])
        delta = p1 - p0
        # rank by change in the right direction for the molecule's true class
        gain = np.where(y > 0.5, delta, -delta)
        for kind, order in (("improved", np.argsort(-gain)), ("worsened", np.argsort(gain))):
            picked = 0
            for idx in order.tolist():
                if picked >= n_per_kind:
                    break
                r = int(rows[idx])
                matches = _best_matches(r, molecules, vocab, discovered_examples)
                if not matches:
                    continue
                out[kind].append({
                    "split": split, "row": r, "label": int(y[idx]),
                    "smiles": molecules.smiles[r],
                    "p_X0": float(p0[idx]), "p_XCF": float(p1[idx]),
                    "delta_p": float(delta[idx]),
                    "n_atoms": int(molecules.n_atoms[r]),
                    "full_matches": matches,
                    "note": "post-hoc diagnostic; not an independent validation",
                })
                picked += 1
    out["definition"] = {
        "improved": "Test molecules whose predicted probability moved furthest toward their "
                    "true class after adding C+F (X-CF - X0)",
        "worsened": "the same quantity in the opposite direction",
        "decode": "all chemical names use analysis/registry_v1 (corrected OGB decoding)",
    }
    return out


def _discovered_examples() -> dict[int, list]:
    with np.load(K.MSDS_DATA / "phaseA_instances.npz", allow_pickle=True) as z:
        return {i: list(x) for i, x in enumerate(z["examples"])}


def _best_matches(r: int, molecules, vocab, examples, k: int = 3) -> list[dict[str, Any]]:
    """FULL types of this molecule with the highest Train support and largest count."""
    full_ids = vocab["full_ids"]
    with np.load(K.MSDS_DATA / "vocab.npz") as z:
        pass
    import scipy.sparse as sp

    with np.load(K.MSDS_DATA / "design_CF.npz") as z:
        n_rows, n_full = int(z["n_rows"]), int(z["n_full"])
        indptr = z["F_indptr"]
        indices = z["F_indices"]
        data = z["F_data"]
    lo, hi = int(indptr[r]), int(indptr[r + 1])
    cols = indices[lo:hi]
    vals = data[lo:hi]
    if cols.size == 0:
        return []
    reg = list(csv.DictReader((K.MSDS_ANALYSIS / "structure_registry_full.csv").open()))
    corrected = list(csv.DictReader(
        (K.ANALYSIS / "registry_v1" / "structure_registry_full_v1.csv").open()))
    fam_priority = {"cycle-6": 5, "cycle-5": 4, "size-4": 3, "size-3": 2, "size-2": 1}
    order = np.argsort([-(fam_priority.get(reg[int(c)]["family"], 0) * 1000 + float(v))
                        for c, v in zip(cols, vals)])
    sites = _match_sites(r, molecules, vocab)
    out = []
    for j in order[:k]:
        col = int(cols[j])
        out.append({
            "full_column": col,
            "family": reg[col]["family"],
            "scaled_count": float(vals[j]),
            "document_frequency_train": int(reg[col]["document_frequency_train"]),
            "scaffold_frequency_train": int(reg[col]["scaffold_frequency_train"]),
            "corrected_node_signature": corrected[col]["node_signature"],
            "corrected_bond_signature": corrected[col]["bond_signature"],
            "corrected_element_sequence": corrected[col]["element_sequence"],
            "predecessor_node_signature": reg[col]["node_signature"],
            "matched_nodes": sites.get(col),
        })
    return out


def _match_sites(r: int, molecules, vocab, *_: Any) -> dict[int, list[int]]:
    """Exact node lists of this molecule's instances for the frozen FULL columns."""
    import msds_enum as ENUM

    full_ids = vocab["full_ids"]
    col_of_key = {int(t): c for c, t in enumerate(full_ids.tolist())}
    bu, bv, bf = molecules.bonds(r)
    n = int(molecules.n_atoms[r])
    lo, hi = int(molecules.atom_offsets[r]), int(molecules.atom_offsets[r + 1])
    atom_feat = molecules.atom_feat[lo:hi]
    bond_attr = {(int(u), int(v)): tuple(int(x) for x in bf[i])
                 for i, (u, v) in enumerate(zip(bu.tolist(), bv.tolist()))}
    out: dict[int, list[int]] = {}
    for _fam, nodes in ENUM.instances_for_molecule(n, bu, bv):
        internal = {}
        for a in nodes:
            for b in nodes:
                if (a, b) in bond_attr:
                    internal[(a, b)] = bond_attr[(a, b)]
        key = ENUM.canonical_key(nodes, atom_feat, internal, tuple(range(atom_feat.shape[1])), (0, 1, 2))
        col = col_of_key.get(_type_digest_of_key(key))
        if col is not None and col not in out:
            out[col] = [int(x) for x in nodes]
    return out


def _type_digest_of_key(key: tuple) -> int:
    """Inverse lookup key: the frozen vocabulary is indexed by (family, key) via ids."""
    return _KEY_INDEX.get(key, -1)


_KEY_INDEX: dict[tuple, int] = {}


def build_key_index(vocab) -> None:
    """Map canonical key -> discovered FULL type id (the frozen type ids)."""
    import msds_vocab as VOCAB

    discovered = VOCAB.load_discovered(str(K.MSDS_DATA / "phaseA_instances.npz"))
    _KEY_INDEX.clear()
    for tid in range(discovered.full.shape[1]):
        _KEY_INDEX[discovered.key_tuple(tid, "FULL")] = tid


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures(structure_rows, mol_rows, cases) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    K.FIGURES.mkdir(parents=True, exist_ok=True)
    fam_rows = [r for r in structure_rows if r["family"] in FAMILIES]
    arms = ["X-C", "X-F", "X-PRES"]
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
    width = 0.26
    xs = np.arange(len(FAMILIES))
    for i, arm in enumerate(arms):
        vals = [next((r["gain_share_of_structure"] or 0.0 for r in fam_rows
                      if r["arm"] == arm and r["family"] == f), 0.0) for f in FAMILIES]
        ax.bar(xs + (i - 1) * width, vals, width, label=arm)
    ax.set_xticks(xs)
    ax.set_xticklabels(FAMILIES, fontsize=7)
    ax.set_ylabel("share of the arm's structural total gain", fontsize=7)
    ax.set_title("Phase D1: which structure families the fitted trees actually use", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = K.FIGURES / "d1_family_gain.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(K.rel(p))

    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=200)
    for split, style in (("valid", "-"), ("test", "--")):
        sub = [r for r in mol_rows if r["split"] == split]
        bins = sorted({r["coverage_bin"] for r in sub})
        means = [float(np.mean([r["delta_XCF"] for r in sub if r["coverage_bin"] == b])) for b in bins]
        ax.plot(bins, means, style, marker="o", label=f"X-CF − X0 ({split})")
    edges = ["0–.25", ".25–.5", ".5–.75", ".75–1"]
    ax.set_xticks(range(4))
    ax.set_xticklabels(edges, fontsize=7)
    ax.set_xlabel("share of FULL instances covered by the frozen vocabulary", fontsize=7)
    ax.set_ylabel("mean Δ probability", fontsize=7)
    ax.set_title("Phase D2: structure gain vs in-vocabulary coverage", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = K.FIGURES / "d2_coverage_gain.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(K.rel(p))
    return paths


def main() -> None:
    K.ensure_dirs()
    structure_rows, structure_detail = structure_classes()
    _write_csv(K.ANALYSIS / "phaseD_structure_classes.csv", structure_rows)
    K.write_json(K.ANALYSIS / "phaseD_structure_detail.json", structure_detail)
    cases_struct = case_structures()
    K.write_json(K.ANALYSIS / "case_structures.json", cases_struct)

    mol_rows, meta = molecule_table()
    _write_csv(K.ANALYSIS / "phaseD_molecules.csv", mol_rows)
    subgroups = subgroup_analysis(mol_rows, meta)
    K.write_json(K.ANALYSIS / "phaseD_subgroups.json", subgroups)

    build_key_index(load_vocab())
    cases = case_molecules()
    K.write_json(K.ANALYSIS / "case_molecules.json", cases)
    figs = make_figures(structure_rows, mol_rows, cases)
    K.write_json(K.ANALYSIS / "phaseD_figures.json", {"figures": figs})
    print(f"[D] structure rows {len(structure_rows)}, molecules {len(mol_rows)}, "
          f"cases {len(cases['improved'])}+{len(cases['worsened'])}, figures {figs}", flush=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
