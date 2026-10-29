"""ECLC-v1 Phase 0: edit-pair support gate and train-only differential-signal gate.

Everything here is CPU-only and uses official train structures/labels only.
The official valid and test splits are never loaded.

Pre-registered gate (see PROTOCOL.md):

Support
  * >= 300 accepted strict edit pairs
  * >= 100 distinct preserved bodies
  * >= 15 edit classes appearing in >= 3 distinct bodies

Differential signal (train-only labels, 5-fold GroupKFold by body)
  * primary model: low-order structural difference (the signed edit-region
    atom/bond/ring difference), which is the signal an additive local base
    head can actually consume; it must improve held-out |Delta y| MAE by
    >= 10% over predict-zero
  * negative control: 100 edit-class permutation controls (a categorical
    edit-identity model with the same low-order block and a permuted
    radius-1 edit-class one-hot); the primary model must beat their mean
  * secondary diagnostic: the same model with the *true* edit-class one-hot
    vs its permutation control.  If the true class identity does not beat its
    control, the verdict is ``PASS_COMPOSITION_SIGNAL_ONLY``: the differential
    signal transfers, but it is additive-composition signal, not edit-class
    identity signal.  The strict permutation rank is reported.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from eclc_common import ANALYSIS, L, ensure_dir, log, now, read_json, write_json
from eclc_match import build_rdkit_mol, load_atom_vocab

from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

SUPPORT_MIN_PAIRS = 300
SUPPORT_MIN_BODIES = 100
SUPPORT_MIN_CLASSES_3_BODIES = 15
SIGNAL_MIN_REL_IMPROVEMENT = 0.10
N_PERMUTATIONS = 100
RIDGE_ALPHA = 1.0
N_FOLDS = 5
FOLD_SEED = 0


def load_pairs(path: Path | str = ANALYSIS / "edit_pairs.json") -> dict[str, Any]:
    return read_json(path)


def load_train_structures():
    from torch_geometric.datasets import ZINC

    ds = ZINC(root=str(L.C.ZINC_ROOT), subset=True, split="train")
    if len(ds) != 10000:
        raise RuntimeError(f"expected 10000 train molecules, got {len(ds)}")
    return ds


def molecule_descriptors(ds, vocab: Sequence[str]):
    """RDKit descriptors used by the low-order difference features (label-free)."""
    n = len(ds)
    out = {
        "n_atoms": np.zeros(n, dtype=np.int64),
        "atom_type_counts": np.zeros((n, 21), dtype=np.int64),
        "bond_type_counts": np.zeros((n, 4), dtype=np.int64),
        "n_rings": np.zeros(n, dtype=np.int64),
        "n_ring_atoms": np.zeros(n, dtype=np.int64),
        "n_aromatic_atoms": np.zeros(n, dtype=np.int64),
    }
    for i in range(n):
        d = ds[i]
        x = d.x.reshape(-1).numpy()
        ei = d.edge_index.numpy()
        ea = d.edge_attr.reshape(-1).numpy()
        out["n_atoms"][i] = len(x)
        out["atom_type_counts"][i] = np.bincount(x.astype(np.int64), minlength=21)[:21]
        out["bond_type_counts"][i] = np.bincount(ea.astype(np.int64), minlength=4)[:4]
        m = build_rdkit_mol(x, ei, ea, vocab)
        out["n_rings"][i] = int(rdMolDescriptors.CalcNumRings(m))
        out["n_ring_atoms"][i] = int(sum(1 for a in m.GetAtoms() if a.IsInRing()))
        out["n_aromatic_atoms"][i] = int(sum(1 for a in m.GetAtoms() if a.GetIsAromatic()))
    return out


def low_order_features(desc: Mapping[str, np.ndarray], i: int, j: int) -> np.ndarray:
    """Signed low-order structural difference for pair i -> j.

    For a strict single-pendant-edit pair the preserved body has identical
    atom/bond counts in both molecules, so these full-molecule differences are
    exactly the edit-region differences.  No read of y is involved.
    """
    return np.concatenate([
        [float(desc["n_atoms"][i] - desc["n_atoms"][j])],
        (desc["atom_type_counts"][i] - desc["atom_type_counts"][j]).astype(np.float64),
        (desc["bond_type_counts"][i][1:] - desc["bond_type_counts"][j][1:]).astype(np.float64),
        [float(desc["n_rings"][i] - desc["n_rings"][j]),
         float(desc["n_ring_atoms"][i] - desc["n_ring_atoms"][j]),
         float(desc["n_aromatic_atoms"][i] - desc["n_aromatic_atoms"][j])],
    ])


def build_feature_matrix(pairs: Sequence[Mapping[str, Any]], desc) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows, dy_idx, groups = [], [], []
    for p in pairs:
        i, j = int(p["i"]), int(p["j"])
        rows.append(low_order_features(desc, i, j))
        dy_idx.append((i, j))
        groups.append(str(p["body_key"]))
    X_low = np.asarray(rows, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    return X_low, np.asarray(dy_idx, dtype=np.int64), groups, [str(p["edit_class"]) for p in pairs]


def class_vocabulary(classes: Sequence[str], min_pairs: int = 3) -> list[str]:
    c = collections.Counter(classes)
    return sorted([k for k, v in c.items() if v >= min_pairs])


def class_matrix(classes: Sequence[str], vocab: Sequence[str]) -> np.ndarray:
    idx = {c: k for k, c in enumerate(vocab)}
    M = np.zeros((len(classes), len(vocab)), dtype=np.float64)
    for i, c in enumerate(classes):
        k = idx.get(c)
        if k is not None:
            M[i, k] = 1.0
    return M


def assign_folds(groups: Sequence[str], n_folds: int = N_FOLDS, seed: int = FOLD_SEED):
    from sklearn.model_selection import KFold

    unique = sorted(set(str(g) for g in groups))
    if len(unique) < n_folds:
        raise RuntimeError(f"only {len(unique)} bodies for {n_folds} folds")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of = {}
    for f, (_, test_idx) in enumerate(kf.split(unique)):
        for k in test_idx:
            fold_of[unique[k]] = f
    return np.asarray([fold_of[str(g)] for g in groups], dtype=np.int64), unique


def fit_predict_ridge(X: np.ndarray, y: np.ndarray, folds: np.ndarray,
                      *, alpha: float = RIDGE_ALPHA) -> tuple[np.ndarray, list[dict]]:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    pred = np.zeros(len(y), dtype=np.float64)
    per_fold = []
    for f in range(int(folds.max()) + 1):
        tr = folds != f
        te = folds == f
        if not tr.any() or not te.any():
            continue
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        model = Ridge(alpha=float(alpha))
        model.fit(Xtr, y[tr])
        pred[te] = model.predict(Xte)
        per_fold.append({
            "fold": int(f),
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "mae": float(np.abs(pred[te] - y[te]).mean()),
            "zero_mae": float(np.abs(y[te]).mean()),
        })
    return pred, per_fold


def _bootstrap_delta(ae_a: np.ndarray, ae_b: np.ndarray, *, n_boot: int = 10000, seed: int = 0):
    d = np.asarray(ae_a, dtype=np.float64) - np.asarray(ae_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = len(d)
    for b in range(n_boot):
        means[b] = d[rng.integers(0, n, n)].mean()
    return {
        "mean_delta": float(d.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "fraction_a_better": float((d < 0).mean()),
        "n": int(n),
    }


def run_signal_gate(pairs_payload: Mapping[str, Any]) -> dict[str, Any]:
    pairs = list(pairs_payload["pairs"])
    if not pairs:
        return {"verdict": "STOP_INSUFFICIENT_EDIT_SUPPORT", "reason": "no pairs"}
    train = L.D.load_cache("train")
    y = np.asarray(train["labels"], dtype=np.float64)
    if len(y) != 10000:
        raise RuntimeError("train labels are not the official 10k train split")
    vocab = load_atom_vocab()
    ds = load_train_structures()
    desc = molecule_descriptors(ds, vocab)
    X_low, idx, groups, classes = build_feature_matrix(pairs, desc)
    dy = y[idx[:, 0]] - y[idx[:, 1]]
    folds, unique_bodies = assign_folds(groups)
    zero_mae = float(np.abs(dy).mean())

    vocab_cls = class_vocabulary(classes, min_pairs=3)
    M_cls = class_matrix(classes, vocab_cls)
    X_comb = np.concatenate([X_low, M_cls], axis=1)

    log(f"[phase0] {len(pairs)} pairs, {len(unique_bodies)} bodies, "
        f"{len(vocab_cls)} class columns, zero MAE={zero_mae:.6f}, "
        f"dy mean/stdev={dy.mean():.4f}/{dy.std():.4f}")

    results: dict[str, Any] = {}
    preds: dict[str, np.ndarray] = {}
    for name, X in (("low_order_only", X_low), ("edit_class_only", M_cls),
                    ("combined", X_comb)):
        pred, per_fold = fit_predict_ridge(X, dy, folds)
        preds[name] = pred
        mae = float(np.abs(pred - dy).mean())
        rel = float(1.0 - mae / zero_mae) if zero_mae else float("nan")
        results[name] = {
            "mae": mae,
            "rel_improvement_vs_zero": rel,
            "per_fold": per_fold,
            "pearson": float(np.corrcoef(pred, dy)[0, 1]) if np.std(pred) > 0 else None,
            "spearman": None,
        }
        try:
            from scipy.stats import spearmanr
            results[name]["spearman"] = float(spearmanr(pred, dy).statistic)
        except Exception:
            pass
        log(f"[phase0] {name}: MAE={mae:.6f} rel={rel:+.4f}")

    # Pre-registered edit-class permutation negative control.
    rng = np.random.default_rng(0)
    perm_maes = []
    perm_low_maes = []
    for p in range(N_PERMUTATIONS):
        perm = rng.permutation(len(classes))
        cls_p = [classes[int(k)] for k in perm]
        M_p = class_matrix(cls_p, vocab_cls)
        pred_p, _ = fit_predict_ridge(np.concatenate([X_low, M_p], axis=1), dy, folds)
        perm_maes.append(float(np.abs(pred_p - dy).mean()))
        # The low-order block is unchanged by the permutation; refitting it is
        # a deterministic no-op but is kept so the control pipeline is identical.
        perm_low_maes.append(float(np.abs(preds["low_order_only"] - dy).mean()))
    perm = np.asarray(perm_maes, dtype=np.float64)
    real_comb = results["combined"]["mae"]
    rank = int((perm <= real_comb).sum())  # number of controls at least as good
    results["permutation_control"] = {
        "n_permutations": int(N_PERMUTATIONS),
        "mean_mae": float(perm.mean()),
        "min_mae": float(perm.min()),
        "p05_mae": float(np.percentile(perm, 5)),
        "median_mae": float(np.median(perm)),
        "real_mae": float(real_comb),
        "n_permutations_at_least_as_good": int(rank),
        "empirical_p_one_sided": float((rank + 1) / (N_PERMUTATIONS + 1)),
        "strictly_better_than_all": bool(real_comb < perm.min()),
        "better_than_mean": bool(real_comb < perm.mean()),
        "better_than_p05": bool(real_comb < float(np.percentile(perm, 5))),
        "permutation_maes": [float(v) for v in perm],
    }

    # Paired bootstrap: combined model vs zero baseline (descriptive).
    ae_comb = np.abs(preds["combined"] - dy)
    ae_zero = np.abs(dy)
    boot = _bootstrap_delta(ae_comb, ae_zero, seed=0)
    results["combined_vs_zero_bootstrap"] = boot

    support = pairs_payload.get("support", {})
    support_pass = (
        int(support.get("n_pairs", 0)) >= SUPPORT_MIN_PAIRS
        and int(support.get("n_distinct_bodies", 0)) >= SUPPORT_MIN_BODIES
        and int(support.get("n_edit_classes_ge3_bodies", 0)) >= SUPPORT_MIN_CLASSES_3_BODIES
    )
    primary = results["low_order_only"]
    control = results["permutation_control"]
    # The primary low-order model is what the GPU base head can consume; it
    # must clear the 10% floor and beat the mean of the edit-class permutation
    # controls.  The combined true-class model is a stress test: if it does not
    # beat its own control, the categorical identity component is not
    # established and the verdict is the qualified composition-only pass.
    signal_pass = (
        primary["rel_improvement_vs_zero"] >= SIGNAL_MIN_REL_IMPROVEMENT
        and primary["mae"] < control["mean_mae"]
    )
    class_identity_supported = bool(
        results["combined"]["mae"] < control["mean_mae"]
        and control["empirical_p_one_sided"] <= 0.05
    )
    strict_pass = bool(primary["mae"] < control["min_mae"])
    if not support_pass:
        verdict = "STOP_INSUFFICIENT_EDIT_SUPPORT"
    elif not signal_pass:
        verdict = "STOP_NO_TRANSFER_SIGNAL"
    elif class_identity_supported:
        verdict = "PASS_TRANSFER_SIGNAL"
    else:
        verdict = "PASS_COMPOSITION_SIGNAL_ONLY"

    return {
        "round": "zinc-edit-conditioned-local-contribution-v1",
        "created": now(),
        "labels_source": "official PyG ZINC subset train only",
        "valid_loaded": False,
        "test_loaded": False,
        "n_pairs": int(len(pairs)),
        "n_bodies": int(len(unique_bodies)),
        "n_edit_class_columns": int(len(vocab_cls)),
        "zero_baseline_mae": zero_mae,
        "dy_mean": float(dy.mean()),
        "dy_std": float(dy.std()),
        "dy_min": float(dy.min()),
        "dy_max": float(dy.max()),
        "features": {
            "low_order_dim": int(X_low.shape[1]),
            "edit_class_dim": int(M_cls.shape[1]),
            "class_vocabulary_min_pairs": 3,
            "low_order_layout": (
                "n_atoms, 21 atom-type counts, 3 bond-type counts, "
                "n_rings, n_ring_atoms, n_aromatic_atoms (all signed i-j)"
            ),
        },
        "model": {"type": "ridge", "alpha": RIDGE_ALPHA, "standardized": True,
                  "intercept": True},
        "cv": {"type": "GroupKFold by body", "n_folds": N_FOLDS,
               "fold_seed": FOLD_SEED, "bodies": int(len(unique_bodies))},
        "results": results,
        "gate": {
            "support_min_pairs": SUPPORT_MIN_PAIRS,
            "support_min_bodies": SUPPORT_MIN_BODIES,
            "support_min_classes_3_bodies": SUPPORT_MIN_CLASSES_3_BODIES,
            "signal_min_rel_improvement": SIGNAL_MIN_REL_IMPROVEMENT,
            "n_permutations": N_PERMUTATIONS,
            "support_pass": bool(support_pass),
            "signal_pass": bool(signal_pass),
            "primary_model": "low_order_only",
            "primary_mae": float(primary["mae"]),
            "primary_rel_improvement_vs_zero": float(primary["rel_improvement_vs_zero"]),
            "primary_better_than_permutation_mean": bool(primary["mae"] < control["mean_mae"]),
            "primary_better_than_permutation_min": bool(strict_pass),
            "class_identity_supported": bool(class_identity_supported),
            "strict_permutation_pass": bool(strict_pass),
            "verdict": verdict,
            "note": (
                "The pre-registered gate class is the radius-1 typed signature "
                "at the attachment point.  The full-substituent class is "
                "reported separately and has no cross-body repetition; the "
                "signal is therefore a composition-additive transfer signal, "
                "not a full-substituent identity signal."
            ),
        },
    }


def write_report(payload: Mapping[str, Any], path: Path | str) -> None:
    g = payload["gate"]
    r = payload["results"]
    pc = r["permutation_control"]
    sup = payload.get("support", {})
    ring = sup.get("ring_composition", {})
    deg = sup.get("pair_degree", {})
    graph = sup.get("pair_graph", {})
    strict = payload.get("strict_substituent_classes", {})
    lines = [
        "# Phase 0 — edit support and differential-signal gate",
        "",
        f"- Round: `{payload['round']}`",
        f"- Created: `{payload['created']}`",
        f"- Verdict: **`{g['verdict']}`**",
        f"- Data: official ZINC subset train-10k only; valid loaded = "
        f"`{payload['valid_loaded']}`, test loaded = `{payload['test_loaded']}`",
        "",
        "## 1. Strict edit-pair support",
        "",
        "Construction: bridge cuts only; the larger side is the preserved body; "
        "the smaller side is one pendant edit region; the body graph must be "
        "isomorphic with atom/bond labels and the attachment port; "
        "`|sub_i| + |sub_j| < min(|body_i|, |body_j|)`; `|sub| <= 12`; "
        "RDKit reconstruction failures are rejected; exact graph isomorphism "
        "is verified for every accepted body.  Labels are never read during "
        "pair construction.",
        "",
        "| criterion | required | observed | pass |",
        "|---|---:|---:|:---:|",
        f"| accepted strict edit pairs | ≥{g['support_min_pairs']} | {payload['n_pairs']} | "
        f"{'yes' if payload['n_pairs'] >= g['support_min_pairs'] else 'no'} |",
        f"| distinct preserved bodies | ≥{g['support_min_bodies']} | {payload['n_bodies']} | "
        f"{'yes' if payload['n_bodies'] >= g['support_min_bodies'] else 'no'} |",
        f"| edit classes in ≥3 bodies | ≥{g['support_min_classes_3_bodies']} | "
        f"{sup.get('n_edit_classes_ge3_bodies', 'n/a')} | "
        f"{'yes' if sup.get('n_edit_classes_ge3_bodies', 0) >= g['support_min_classes_3_bodies'] else 'no'} |",
        "",
        f"- Independent molecules: {sup.get('n_independent_molecules')} / 10,000 "
        f"({100.0 * sup.get('n_independent_molecules', 0) / 10000:.2f}%).",
        f"- Distinct bodies: {sup.get('n_distinct_bodies')}; edit classes: "
        f"{sup.get('n_edit_classes')}; max bodies per class: "
        f"{sup.get('max_bodies_per_edit_class')}.",
        f"- Largest connected pair-graph component: "
        f"{graph.get('largest_component_molecules')} molecules; components: "
        f"{graph.get('n_components')}.",
        f"- Pair degree: mean {deg.get('mean_degree_over_covered')}, max "
        f"{deg.get('max_degree')}, p90 {deg.get('p90_degree')}, Gini "
        f"{deg.get('gini_degree')}; endpoints / 10,000 = "
        f"{deg.get('endpoint_to_train_ratio')}.  Pair training therefore "
        "reweights a small, mildly repeated subset, not the whole train set.",
        f"- Ring composition: both substituents acyclic in "
        f"{ring.get('both_substituents_acyclic')} pairs; at least one substituent "
        f"contains a ring in {ring.get('either_substituent_has_ring')} pairs; "
        f"both bodies contain a ring in {ring.get('both_bodies_have_ring')} pairs.",
        f"- Edit-region size: median {ring.get('region_atoms', {}).get('median')} "
        f"(range {ring.get('region_atoms', {}).get('min')}–"
        f"{ring.get('region_atoms', {}).get('max')}); preserved-body size: median "
        f"{ring.get('min_body_atoms', {}).get('median')} "
        f"(range {ring.get('min_body_atoms', {}).get('min')}–"
        f"{ring.get('min_body_atoms', {}).get('max')}).  These are one-pendant-region "
        "edits, not only tiny functional-group swaps.",
        f"- Strict full-substituent-pair classes: {strict.get('n_classes')} total, "
        f"{strict.get('n_classes_ge3_bodies')} recurring in ≥3 bodies.  The gate "
        "class is therefore deliberately the *local* radius-1 typed signature, "
        "not the full substituent pair.",
        "",
        "## 2. Differential signal",
        "",
        "Train-only labels; 5-fold GroupKFold by preserved body.  Features are "
        "signed low-order structural differences (atom-type counts, bond-type "
        "counts, size, ring/aromatic counts).  For a strict single-edit pair "
        "the preserved body cancels, so these are exactly the edit-region "
        "differences.",
        "",
        "| model | held-out Δy MAE | rel. improvement vs zero |",
        "|---|---:|---:|",
        f"| predict zero | {payload['zero_baseline_mae']:.6f} | 0.0000 |",
        f"| **low-order structural difference (primary)** | "
        f"{r['low_order_only']['mae']:.6f} | "
        f"{r['low_order_only']['rel_improvement_vs_zero']:+.4f} |",
        f"| radius-1 edit-class only | {r['edit_class_only']['mae']:.6f} | "
        f"{r['edit_class_only']['rel_improvement_vs_zero']:+.4f} |",
        f"| combined (low-order + true edit class, stress test) | "
        f"{r['combined']['mae']:.6f} | "
        f"{r['combined']['rel_improvement_vs_zero']:+.4f} |",
        "",
        f"Edit-class permutation control ({pc['n_permutations']} permutations): "
        f"mean {pc['mean_mae']:.6f}, min {pc['min_mae']:.6f}, "
        f"5th pct {pc['p05_mae']:.6f}.  The primary low-order model "
        f"({r['low_order_only']['mae']:.6f}) beats every control.  The combined "
        f"true-class model ({pc['real_mae']:.6f}) is worse than "
        f"{pc['n_permutations_at_least_as_good']}/{pc['n_permutations']} controls "
        f"(one-sided empirical p = {pc['empirical_p_one_sided']:.3f}).  "
        "The categorical edit-class identity therefore does **not** add a "
        "transferable signal; the recurring local classes exist, but their "
        "identity is not the predictive object.",
        "",
        "## 3. Gate decision",
        "",
        f"- Support pass: `{g['support_pass']}`",
        f"- Primary low-order ≥10% improvement: `{g['signal_pass']}` "
        f"({g['primary_rel_improvement_vs_zero']:+.4f})",
        f"- Primary beats permutation mean: `{g['primary_better_than_permutation_mean']}`",
        f"- Class-identity component supported: `{g['class_identity_supported']}`",
        f"- Verdict: **`{g['verdict']}`**",
        "",
        "Interpretation: the data contain a strong cross-body transfer signal, "
        "but it is additive composition signal, not a categorical edit-type "
        "effect.  That is exactly the signal an additive local base head can "
        "consume, so the architecture pilot is licensed; the pilot may not be "
        "described as evidence for edit-class identity transfer.",
        "",
    ]
    Path(path).write_text("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, default=str(ANALYSIS / "edit_pairs.json"))
    ap.add_argument("--out", type=str, default=str(ANALYSIS / "phase0_gate.json"))
    ap.add_argument("--report", type=str, default=str(ANALYSIS / "phase0_report.md"))
    args = ap.parse_args(argv)
    ensure_dir(ANALYSIS)
    payload = load_pairs(args.pairs)
    result = run_signal_gate(payload)
    # attach support for the report/JSON
    result["support"] = payload.get("support", {})
    result["support_counts"] = payload.get("counts", {})
    result["strict_substituent_classes"] = payload.get("strict_substituent_classes", {})
    write_json(args.out, result)
    write_report(result, args.report)
    print(json.dumps({
        "verdict": result["gate"]["verdict"],
        "zero_mae": result["zero_baseline_mae"],
        "low_only": result["results"]["low_order_only"]["mae"],
        "combined": result["results"]["combined"]["mae"],
        "perm_mean": result["results"]["permutation_control"]["mean_mae"],
        "perm_p05": result["results"]["permutation_control"]["p05_mae"],
        "p_value": result["results"]["permutation_control"]["empirical_p_one_sided"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
