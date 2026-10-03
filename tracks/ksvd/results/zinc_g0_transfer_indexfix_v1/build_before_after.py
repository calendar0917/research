"""Build the old->new comparison table and verify node-key invariance.

Reads the old audit JSONs (read-only) and the fixed recompute outputs.
Recomputes the OLD (broken) unique edge key counts to complete the table.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = Path("tracks/ksvd/results/zinc_g0_transfer_audit_v1").resolve()
sys.path.insert(0, str(HERE))
import part_c_indexfix as fx  # noqa: E402

VIEWS = ("edge", "node")
VARIANTS = ("residual_support", "full_descriptor")
SPLITS = ("meta_g0", "dev_g0")


def load_old():
    comp = json.loads((OLD / "part_c_class_composition.json").read_text())["class_composition"]
    matching = json.loads((OLD / "part_c_matching.json").read_text())["matching"]
    return comp, matching


def main() -> int:
    comp_old, matching_old = load_old()
    cov_new = json.loads((HERE / "coverage_fixed.json").read_text())
    comp_new = json.loads((HERE / "class_composition_fixed.json").read_text())
    matching_new = json.loads((HERE / "matching_fixed.json").read_text())

    # OLD broken unique edge key counts (recompute with broken=True)
    old_unique = {}
    train_data = fx.load_train_only()
    g = fx.build_index(train_data)
    for fold in fx.FOLDS:
        model = fx.load_model(fold)
        kb = fx.compute_model_keys_fixed(model, g, broken=True)
        old_unique[fold] = {
            "unique_edge_support_keys": len(set(kb["edge_support"])),
            "unique_edge_desc_keys": len(set(kb["edge_desc"])),
            "unique_edge_joint_keys": len(set((kb["edge_support"][i], kb["edge_sem"][i])
                                              for i in range(len(kb["edge_support"])))),
        }

    rows = []
    for fold in fx.FOLDS:
        fn = cov_new["folds"][fold]
        for view in VIEWS:
            for var in VARIANTS:
                for split in SPLITS:
                    key = f"{fold}|{view}|{var}|{split}"
                    old_c = comp_old[key]
                    new_c = comp_new[key]
                    new_m = matching_new[fold][view][var][split]
                    old_m = matching_old[fold][view][var][split]
                    row = {
                        "fold": fold, "view": view, "key": var, "split": split,
                        "joint_seen_old": old_c["joint_seen"]["mean"],
                        "joint_seen_new": new_c["joint_seen"]["mean"],
                        "joint_rare_marginals_seen_old": old_c["joint_rare_marginals_seen"]["mean"],
                        "joint_rare_marginals_seen_new": new_c["joint_rare_marginals_seen"]["mean"],
                        "structure_rare_old": old_c["structure_rare"]["mean"],
                        "structure_rare_new": new_c["structure_rare"]["mean"],
                        "semantic_rare_old": old_c["semantic_rare"]["mean"],
                        "semantic_rare_new": new_c["semantic_rare"]["mean"],
                        "both_rare_old": old_c["both_rare"]["mean"],
                        "both_rare_new": new_c["both_rare"]["mean"],
                        "unseen_joint_old": old_c["_unseen_joint"]["mean"],
                        "unseen_joint_new": new_c["_unseen_joint"]["mean"],
                        "high_n_old": old_m["n_high"], "high_n_new": new_m["n_high"],
                        "low_n_old": old_m["n_low"], "low_n_new": new_m["n_low"],
                        "n_pairs_old": old_m["n_pairs"], "n_pairs_new": new_m["n_pairs"],
                        "matched_mae_high_minus_low_old": old_m["paired_bootstrap"].get("mae_high_minus_low"),
                        "matched_mae_high_minus_low_new": new_m["paired_bootstrap"].get("mae_high_minus_low"),
                    }
                    rows.append(row)

    # structural-key cardinalities (edge keys only, per fold)
    for fold in fx.FOLDS:
        fn = cov_new["folds"][fold]
        rows.append({
            "fold": fold, "view": "edge", "key": "cardinality", "split": "train_fold",
            "joint_seen_old": "", "joint_seen_new": "",
            "joint_rare_marginals_seen_old": "", "joint_rare_marginals_seen_new": "",
            "structure_rare_old": old_unique[fold]["unique_edge_support_keys"],
            "structure_rare_new": fn["unique_edge_support_keys"],
            "semantic_rare_old": old_unique[fold]["unique_edge_desc_keys"],
            "semantic_rare_new": fn["unique_edge_desc_keys"],
            "both_rare_old": old_unique[fold]["unique_edge_joint_keys"],
            "both_rare_new": fn["unique_edge_joint_keys"],
            "unseen_joint_old": "", "unseen_joint_new": "",
            "high_n_old": "", "high_n_new": "", "low_n_old": "", "low_n_new": "",
            "n_pairs_old": "", "n_pairs_new": "",
            "matched_mae_high_minus_low_old": "", "matched_mae_high_minus_low_new": "",
            "note": "structure_rare_* column = unique support keys; semantic_rare_* = unique descriptor keys; "
                    "both_rare_* = unique joint keys",
        })

    for r in rows:
        r.setdefault("note", "")
    fields = list(rows[0].keys())
    with open(HERE / "before_after.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # node-key invariance: old vs new per-molecule node rows must be identical
    old_z = np.load(OLD / "part_c_per_molecule.npz", allow_pickle=False)
    new_z = np.load(HERE / "per_molecule_fixed.npz", allow_pickle=False)
    inv = {}
    for fold in fx.FOLDS:
        for view in VIEWS:
            for var in VARIANTS:
                for split in SPLITS:
                    k = f"{fold}|{view}|{var}|{split}"
                    same = list(old_z[k].tolist()) == list(new_z[k].tolist())
                    inv[k] = same
    node_inv = {k: v for k, v in inv.items() if "|node|" in k}
    edge_changed = {k: v for k, v in inv.items() if "|edge|" in k}
    out = {"node_rows_identical_to_old": node_inv,
           "all_node_identical": bool(all(node_inv.values())),
           "edge_rows_identical_to_old": edge_changed,
           "all_edge_changed": bool(not any(edge_changed.values())),
           "old_unique_edge_keys": old_unique}
    (HERE / "node_key_invariance.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"node_all_identical": out["all_node_identical"],
                      "edge_all_changed": out["all_edge_changed"],
                      "old_unique": old_unique}, indent=2))
    print("wrote before_after.csv with", len(rows), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())