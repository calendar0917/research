"""Real matched molecular pairs (11.4).

The pair directory is the parent round's real replacement directory: two
observed molecules under an identical exact context, i.e. a real pair, not a
synthetic counterfactual.  Only the whole-molecule model delta is evaluated:

    predicted_delta = yhat(G_b) - yhat(G_a)

Scopes: Train×Train (in-sample for the model, still real pairs),
Train×Valid (cross-split, the useful generalisation probe), Valid×Valid
(15 pairs, descriptive only).  Test-touching pairs are dropped *before* their
labels are read.  Bootstrap clusters: pair component for Train×Train, the Valid
molecule for Train×Valid; Valid×Valid has no meaningful cluster structure.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

import acra_common as C
import acra_data as D
import acra_cache as K

PARENT_ROUND = C.PARENT_ROUND
PARENT_PACK = PARENT_ROUND / "data" / "trc_cache_lipo.npz"
RUNS = {"ACRA_ADD": "ACRA_ADD_S0", "ACRA_COND": "ACRA_COND_S0", "ACRA_REL": "ACRA_REL_S0"}


def load_pairs() -> dict[str, Any]:
    with np.load(PARENT_PACK, allow_pickle=False) as handle:
        pack = {k: handle[k] for k in handle.files}
    return pack


def molecule_predictions(run_id: str, expected_rows: Mapping[int, int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-molecule composed prediction and term split from the best checkpoint."""
    import torch

    rdir = C.run_dir(run_id)
    tr = np.load(rdir / "predictions_best_train.npz")
    va = np.load(rdir / "predictions_best.npz")
    out = {}
    manifest = C.read_json(rdir / "manifest.json")
    out["label_std"] = np.asarray(float(manifest["scaler"]["std"]))
    out["label_mean"] = np.asarray(float(manifest["scaler"]["mean"]))
    for name, payload in (("train", tr), ("valid", va)):
        for key in ("pred", "pred_std", "e_atom_std", "e_bond_std", "e_rel_std",
                    "beta_std", "row_id", "label"):
            out[f"{name}_{key}"] = np.asarray(payload[key], np.float64)
    return out


def _scope_masks(pack: Mapping[str, Any]) -> dict[str, np.ndarray]:
    split = np.asarray(pack["molecule_split"], np.uint8)
    a = np.asarray(pack["pair_mol_a"], np.int64)
    b = np.asarray(pack["pair_mol_b"], np.int64)
    sa, sb = split[a], split[b]
    allowed = (sa != 2) & (sb != 2)
    return {
        "train_train": allowed & (sa == 0) & (sb == 0),
        "train_valid": allowed & (((sa == 0) & (sb == 1)) | ((sa == 1) & (sb == 0))),
        "valid_valid": allowed & (sa == 1) & (sb == 1),
        "_test_touching": ~allowed,
    }


def _compare_delta_mae(pred: Mapping[str, np.ndarray], truth: np.ndarray,
                       clusters: np.ndarray, pairs: Sequence[tuple[str, str]],
                       n_boot: int, seed: int) -> dict[str, Any]:
    """Paired cluster bootstrap of delta-MAE differences between arms."""
    out = {}
    uniq = np.unique(clusters)
    rng = np.random.default_rng(int(seed))
    for candidate, control in pairs:
        p_c = pred[candidate]
        p_o = pred[control]
        observed = float(np.mean(np.abs(p_o - truth)) - np.mean(np.abs(p_c - truth)))
        diffs = []
        for _ in range(int(n_boot)):
            pick = rng.integers(0, uniq.size, uniq.size)
            idx = np.concatenate([np.nonzero(clusters == uniq[j])[0] for j in pick])
            diffs.append(float(np.mean(np.abs(p_o[idx] - truth[idx]))
                               - np.mean(np.abs(p_c[idx] - truth[idx]))))
        out[f"{candidate}_vs_{control}"] = {
            "delta_mae_difference": observed,
            "delta_mae_difference_lo": float(np.percentile(diffs, 2.5)),
            "delta_mae_difference_hi": float(np.percentile(diffs, 97.5)),
            "positive_means_candidate_better": observed > 0,
            "definition": "mean|pred(control)-y| - mean|pred(candidate)-y| on the same pairs",
            "n_pairs": int(truth.size),
            "n_clusters": int(uniq.size),
            "bootstrap_reps": int(n_boot),
        }
    return out


def run(cfg: Mapping[str, Any], log=print) -> dict[str, Any]:
    pairs = load_pairs()
    raw = D.load_raw(cfg)
    labels = np.asarray(raw["labels"], np.float64)
    smiles = np.asarray(raw["smiles"])
    split = np.asarray(pairs["molecule_split"], np.uint8)
    # sanity: the parent split array must agree with the official split rows
    check = np.full(split.size, 2, np.uint8)
    for name, value in (("train", 0), ("valid", 1)):
        check[D.official_splits(raw)[name]] = value
    if not np.array_equal(check, split):
        raise ValueError("parent pair pack split does not match the official split")
    masks = _scope_masks(pairs)
    a = np.asarray(pairs["pair_mol_a"], np.int64)
    b = np.asarray(pairs["pair_mol_b"], np.int64)
    component = np.asarray(pairs["pair_component"], np.int64)
    transform = np.asarray(pairs["pair_transform"], np.int64)

    out: dict[str, Any] = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "pair_source": str(PARENT_PACK.relative_to(C.REPO_ROOT)),
        "definition": "predicted_delta = yhat(G_b) - yhat(G_a) of the full composed model; "
                      "local response differences are never called a replacement effect",
        "test_policy": "pairs touching the official Test split are dropped before any "
                       "label is read",
        "pair_counts": {k: int(v.sum()) for k, v in masks.items()},
        "scopes": {},
    }
    arm_deltas: dict[str, np.ndarray] = {}
    arm_truth: dict[str, np.ndarray] = {}
    arm_clusters: dict[str, np.ndarray] = {}
    for run_id in RUNS.values():
        rdir = C.run_dir(run_id)
        if not (rdir / "predictions_best_train.npz").is_file():
            log(f"  {run_id}: predictions missing; skipped")
            continue
        pred = molecule_predictions(run_id, {})
        mol_pred = np.full(labels.size, np.nan)
        mol_terms = {k: np.full(labels.size, np.nan) for k in ("e_atom_std", "e_bond_std",
                                                               "e_rel_std", "pred_std")}
        for scope in ("train", "valid"):
            rows = np.asarray(pred[f"{scope}_row_id"], np.int64)
            mol_pred[rows] = pred[f"{scope}_pred"]
            for k in mol_terms:
                mol_terms[k][rows] = pred[f"{scope}_{k}"]
        rec: dict[str, Any] = {"run_id": run_id}
        for scope in ("train_train", "train_valid", "valid_valid"):
            mask = masks[scope]
            if not mask.any():
                rec[scope] = {"n": 0}
                continue
            idx = np.nonzero(mask)[0]
            true_delta = labels[b[idx]] - labels[a[idx]]
            pred_delta = mol_pred[b[idx]] - mol_pred[a[idx]]
            if scope == "train_train":
                comp = component[idx]
                clusters = np.where(comp >= 0, comp, -(np.arange(comp.size) + 1))
                unit = "pair component; pairs without a shared component are their own cluster"
            elif scope == "train_valid":
                sv = split[a[idx]] + split[b[idx]]
                clusters = np.where(split[a[idx]] == 1, a[idx], b[idx])
                unit = "Valid molecule of the crossing pair"
            else:
                clusters = np.arange(idx.size)
                unit = "none (15 pairs, descriptive only)"
            stats = C.paired_delta_stats(true_delta, pred_delta, clusters,
                                         n_boot=int(cfg["pairs"]["bootstrap_reps"]),
                                         seed=int(cfg["pairs"]["seed"]))
            stats["cluster_unit"] = unit
            stats["pred_delta_quantiles"] = C.quantiles(pred_delta, (5, 50, 95))
            stats["true_delta_quantiles"] = C.quantiles(true_delta, (5, 50, 95))
            rec[scope] = stats
            if scope == "train_valid":
                arm_deltas[run_id] = pred_delta
                arm_truth[run_id] = true_delta
                arm_clusters[run_id] = clusters
        # fixed representative rule on the cross-split scope
        idx = np.nonzero(masks["train_valid"])[0]
        representatives = []
        if idx.size:
            by_transform: dict[int, list[int]] = defaultdict(list)
            for i in idx.tolist():
                by_transform[int(transform[i])].append(i)
            candidates = [(t, v) for t, v in by_transform.items() if len(v) >= 3]
            if not candidates:
                candidates = [(t, v) for t, v in by_transform.items() if len(v) >= 1]
            t, members = max(candidates, key=lambda kv: (len(kv[1]), -kv[0]))
            truth = {i: labels[b[i]] - labels[a[i]] for i in members}
            order = sorted(members, key=lambda i: (abs(truth[i]), i))
            picks = [order[-1], order[len(order) // 2], order[0]]
            seen: set[int] = set()
            for i in picks:
                if i in seen:
                    continue
                seen.add(i)
                ga, gb = int(a[i]), int(b[i])
                representatives.append({
                    "pair_index": int(i),
                    "transform": int(t),
                    "component": int(component[i]),
                    "mol_a": {"row_id": int(raw["row_ids"][ga]), "smiles": str(smiles[ga]),
                              "label": float(labels[ga])},
                    "mol_b": {"row_id": int(raw["row_ids"][gb]), "smiles": str(smiles[gb]),
                              "label": float(labels[gb])},
                    "true_delta": float(truth[i]),
                    "predicted_delta": float(mol_pred[gb] - mol_pred[ga]),
                    "decomposition_std": {
                        "delta_e_atom": float(mol_terms["e_atom_std"][gb]
                                              - mol_terms["e_atom_std"][ga]),
                        "delta_e_bond": float(mol_terms["e_bond_std"][gb]
                                              - mol_terms["e_bond_std"][ga]),
                        "delta_e_rel": float(mol_terms["e_rel_std"][gb]
                                             - mol_terms["e_rel_std"][ga]),
                        "sum": float(mol_terms["pred_std"][gb] - mol_terms["pred_std"][ga]),
                    },
                    "decomposition_label_units": {
                        "delta_e_atom": float((mol_terms["e_atom_std"][gb]
                                               - mol_terms["e_atom_std"][ga])
                                              * float(pred["label_std"])),
                        "delta_e_bond": float((mol_terms["e_bond_std"][gb]
                                               - mol_terms["e_bond_std"][ga])
                                              * float(pred["label_std"])),
                        "delta_e_rel": float((mol_terms["e_rel_std"][gb]
                                              - mol_terms["e_rel_std"][ga])
                                             * float(pred["label_std"])),
                        "sum": float((mol_terms["pred_std"][gb] - mol_terms["pred_std"][ga])
                                     * float(pred["label_std"])),
                    },
                })
            rec["representatives_train_valid"] = representatives
            rec["representative_rule"] = (
                "inside the largest Train×Valid pair-transform group with >=3 pairs, take "
                "the largest, median and smallest |true Δy| (ties by pair index); "
                "decomposition is exact model algebra in standardised units")
        out["scopes"][run_id] = rec
    if all(RUNS[a] in arm_deltas for a in RUNS):
        base = RUNS["ACRA_ADD"]
        out["arm_comparisons_train_valid"] = _compare_delta_mae(
            {arm: arm_deltas[RUNS[arm]] for arm in RUNS}, arm_truth[base],
            arm_clusters[base],
            (("ACRA_COND", "ACRA_ADD"), ("ACRA_REL", "ACRA_COND"),
             ("ACRA_REL", "ACRA_ADD")),
            int(cfg["pairs"]["bootstrap_reps"]), int(cfg["pairs"]["seed"]))
    out["test_labels_read"] = False
    C.write_json(C.analysis_dir() / "pair_delta.json", out)
    log("wrote pair_delta.json")
    for run_id, rec in out["scopes"].items():
        tv = rec.get("train_valid", {})
        log(f"  {run_id}: train_valid delta_mae={tv.get('delta_mae', float('nan')):.3f} "
            f"dir_acc={tv.get('direction_accuracy', float('nan')):.3f}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    _args = ap.parse_args(argv)
    run(C.load_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())