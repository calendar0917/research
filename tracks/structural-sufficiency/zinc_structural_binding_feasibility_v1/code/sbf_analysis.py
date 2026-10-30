"""Aggregate the structural-binding feasibility run and apply the strict gate.

Reads analysis/results.json + analysis/raw_arrays.npz (written by
sbf_run.py --stage all).  No model is trained and no property label is read.
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

TRACK = Path(__file__).resolve().parents[1]
ANALYSIS = TRACK / "analysis"
sys.path.insert(0, str(TRACK / "code"))
import sbf_data as D  # noqa: E402
import sbf_run as RUN  # noqa: E402

SALT = "sbf-analysis-v2"
KNOWN = ["binding_s0", "binding_s1", "marginal_s0", "marginal_s1",
         "linear_s0", "linear_s1", "oracle_s0", "oracle_s1",
         "binding_compat_s0", "linear_pair_only_s0", "frequency"]


def _rng(tag: str) -> np.random.Generator:
    h = hashlib.sha256((SALT + "|" + tag).encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(h, "little"))


def _table(summary: dict[str, Any], group: str) -> dict[str, Any]:
    out = {}
    for prefix in KNOWN:
        if prefix not in summary[group]:
            continue
        s = summary[group][prefix]
        folds = np.asarray(s["fold_acc"], dtype=np.float64)
        out[prefix] = {
            "n_roots": int(s["n_roots"]),
            "n_molecules": int(s["n_molecules"]),
            "pair_acc": float(s["pair_acc"]),
            "ci_low": float(s["ci_low"]),
            "ci_high": float(s["ci_high"]),
            "fold_acc": [float(x) for x in folds],
            "fold_min": float(folds.min()),
            "fold_max": float(folds.max()),
            "fold_std": float(folds.std(ddof=1)) if len(folds) > 1 else 0.0,
        }
    return out


def _paired_diff(ca: np.ndarray, cb: np.ndarray, mol: np.ndarray, tag: str,
                 n_boot: int = 4000) -> dict[str, Any]:
    ca = np.asarray(ca, dtype=np.float64)
    cb = np.asarray(cb, dtype=np.float64)
    mol = np.asarray(mol, dtype=np.int64)
    if len(ca) != len(cb) or len(ca) != len(mol):
        raise ValueError(f"unaligned arrays for {tag}: {len(ca)} {len(cb)} {len(mol)}")
    diff = ca - cb
    uniq, inv = np.unique(mol, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    sums = np.bincount(inv, weights=diff, minlength=len(uniq))
    rng = _rng(tag + "|diff")
    vals = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        idx = rng.integers(0, len(uniq), size=len(uniq))
        vals[k] = sums[idx].sum() / counts[idx].sum()
    se = float(np.sqrt(np.var(diff, ddof=1) / len(diff))) if len(diff) > 1 else float("nan")
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
        "se_root": se,
        "z_root": float(diff.mean() / se) if se > 0 else float("nan"),
        "n_roots": int(len(diff)),
        "n_molecules": int(len(uniq)),
    }


def _acc_ci(correct: np.ndarray, mol: np.ndarray, tag: str,
            n_boot: int = 4000) -> dict[str, Any]:
    correct = np.asarray(correct, dtype=np.float64)
    mol = np.asarray(mol, dtype=np.int64)
    uniq, inv = np.unique(mol, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    sums = np.bincount(inv, weights=correct, minlength=len(uniq))
    rng = _rng(tag + "|acc")
    vals = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        idx = rng.integers(0, len(uniq), size=len(uniq))
        vals[k] = sums[idx].sum() / counts[idx].sum()
    return {
        "n_roots": int(len(correct)),
        "n_molecules": int(len(uniq)),
        "pair_acc": float(correct.mean()),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
    }


def _normalized_coverage(cov: dict[str, Any], cache: dict[str, Any],
                         fakes: dict[str, Any]) -> dict[str, Any]:
    root_mol = np.asarray(cache["root_mol"], dtype=np.int64)
    root_ptr = np.asarray(cache["root_ptr"], dtype=np.int64)
    npaths = np.diff(root_ptr)
    out: dict[str, Any] = {}
    for name, key in [("standard", "standard"), ("hard", "hard")]:
        mask = np.asarray([f is not None for f in fakes[key]])
        vals = np.asarray(sorted(collections.Counter(root_mol[mask].tolist()).values(),
                                 reverse=True), dtype=np.int64)
        n_mol = len(vals)
        entry: dict[str, Any] = {
            "valid_roots": int(mask.sum()),
            "valid_molecules": int(n_mol),
            "max_roots_per_molecule": int(vals.max()) if n_mol else 0,
            "singleton_molecule_fraction": float((vals == 1).mean()) if n_mol else 0.0,
        }
        for p in (1, 5, 10, 20):
            k = max(1, int(np.ceil(n_mol * p / 100.0))) if n_mol else 0
            entry[f"top{p}pct_bearing_molecule_root_share"] = (
                float(vals[:k].sum() / vals.sum()) if k and vals.sum() else 0.0)
        types = collections.Counter(int(x) for x in np.asarray(cache["root_type"])[mask])
        rings = collections.Counter(int(x) for x in np.asarray(cache["root_ring"])[mask])
        npath_counts = collections.Counter(int(x) for x in npaths[mask])
        total = max(1, int(mask.sum()))
        entry["root_type_counts"] = {str(k): int(v) for k, v in types.most_common()}
        entry["root_type_fractions"] = {str(k): float(v / total) for k, v in types.most_common()}
        entry["ring_counts"] = {str(k): int(v) for k, v in rings.most_common()}
        entry["ring_fraction"] = float(rings.get(1, 0) / total)
        entry["npaths_counts"] = {str(k): int(v) for k, v in sorted(npath_counts.items())}
        out[name] = entry
    return out


def main() -> None:
    res = json.loads((ANALYSIS / "results.json").read_text())
    raw = np.load(ANALYSIS / "raw_arrays.npz")
    summary = res["summary"]

    def g(group: str, prefix: str, key: str = "correct") -> np.ndarray:
        if prefix == "frequency":
            group = "hard_frequency" if group == "hard" else "frequency"
            prefix = "frequency"
        return np.asarray(raw[f"{group}__{prefix}_{key}"])

    standard = _table(summary, "standard")
    hard = _table(summary, "hard")

    diffs_std = {}
    for a, b in [("binding_s0", "marginal_s0"),
                 ("binding_s0", "linear_s0"),
                 ("binding_s0", "frequency"),
                 ("binding_s0", "linear_pair_only_s0"),
                 ("binding_s0", "oracle_s0"),
                 ("binding_s0", "binding_compat_s0"),
                 ("linear_s0", "frequency")]:
        diffs_std[f"{a}_minus_{b}"] = _paired_diff(
            g("standard", a), g("standard", b), g("standard", a, "mol"), "std")

    diffs_hard = {}
    for a, b in [("binding_s0", "marginal_s0"),
                 ("binding_s0", "frequency"),
                 ("binding_s0", "linear_s0"),
                 ("binding_s0", "linear_pair_only_s0"),
                 ("binding_s0", "oracle_s0"),
                 ("binding_s0", "binding_compat_s0"),
                 ("binding_s1", "marginal_s1"),
                 ("binding_s1", "frequency"),
                 ("frequency", "marginal_s0"),
                 ("linear_s0", "marginal_s0"),
                 ("linear_pair_only_s0", "marginal_s0")]:
        diffs_hard[f"{a}_minus_{b}"] = _paired_diff(
            g("hard", a), g("hard", b), g("hard", a, "mol"), "hard")

    # hard frequency-prior diagnostic
    b = g("hard", "binding_s0")
    f = g("hard", "frequency")
    fm = np.asarray(raw["hard_frequency__frequency_margins"], dtype=np.float64)
    bm = np.asarray(raw["hard__binding_s0_margins"], dtype=np.float64)
    lp = g("hard", "linear_pair_only_s0")
    if not (len(b) == len(f) == len(fm) == len(bm) == len(lp)):
        raise ValueError("hard frequency diagnostic arrays not aligned")
    pos, neg = fm > 0, fm < 0
    qs = np.quantile(fm, [0.25, 0.50, 0.75])
    bins = np.digitize(fm, qs)
    quartiles = []
    for q in range(4):
        m = bins == q
        quartiles.append({
            "bin": q,
            "n_roots": int(m.sum()),
            "mean_frequency_margin": float(fm[m].mean()) if m.any() else None,
            "binding_acc": float(b[m].mean()) if m.any() else None,
            "frequency_table_acc": float(f[m].mean()) if m.any() else None,
            "linear_pair_only_acc": float(lp[m].mean()) if m.any() else None,
        })
    context_freq = None
    ctx_path = ANALYSIS / "context_freq.json"
    if ctx_path.exists():
        context_freq = json.loads(ctx_path.read_text())

    freq_diag = {
        "n_roots": int(len(fm)),
        "pearson_margin_corr": float(np.corrcoef(fm, bm)[0, 1]),
        "binding_acc": float(b.mean()),
        "frequency_acc": float(f.mean()),
        "binding_acc_freq_positive": float(b[pos].mean()) if pos.any() else None,
        "binding_acc_freq_negative": float(b[neg].mean()) if neg.any() else None,
        "frequency_positive_fraction": float(pos.mean()),
        "frequency_negative_fraction": float(neg.mean()),
        "binding_minus_frequency": float(b.mean() - f.mean()),
        "frequency_quartiles": quartiles,
    }

    # novel-context transfer from the raw per-root arrays
    bc = np.asarray(raw["novel_raw__binding_s0_correct"], dtype=np.float64)
    lc = np.asarray(raw["novel_raw__linear_s0_correct"], dtype=np.float64)
    nmol = np.asarray(raw["novel_raw__mol"], dtype=np.int64)
    seen = np.asarray(raw["novel_raw__seen"], dtype=bool)
    unseen = np.asarray(raw["novel_raw__unseen"], dtype=bool)
    upair = np.asarray(raw["novel_raw__unseen_pair"], dtype=bool)
    if not (len(bc) == len(lc) == len(nmol) == len(seen) == len(unseen) == len(upair)):
        raise ValueError("novel raw arrays not aligned")
    novel = {
        "seen": {"binding": _acc_ci(bc[seen], nmol[seen], "novel-seen-b"),
                 "linear": _acc_ci(lc[seen], nmol[seen], "novel-seen-l")},
        "unseen": {"binding": _acc_ci(bc[unseen], nmol[unseen], "novel-unseen-b"),
                   "linear": _acc_ci(lc[unseen], nmol[unseen], "novel-unseen-l")},
        "unseen_pair": {"binding": _acc_ci(bc[upair], nmol[upair], "novel-pair-b"),
                        "linear": _acc_ci(lc[upair], nmol[upair], "novel-pair-l")},
        "unseen_fraction": float(unseen.mean()),
        "unseen_pair_fraction": float(upair.mean()),
        "binding_minus_linear_seen": _paired_diff(bc[seen], lc[seen], nmol[seen], "novel-seen"),
        "binding_minus_linear_unseen": _paired_diff(bc[unseen], lc[unseen], nmol[unseen], "novel-unseen"),
        "binding_minus_linear_unseen_pair": _paired_diff(bc[upair], lc[upair], nmol[upair], "novel-pair"),
    }
    novel["binding_unseen_gap"] = (novel["seen"]["binding"]["pair_acc"]
                                   - novel["unseen"]["binding"]["pair_acc"])
    novel["linear_unseen_gap"] = (novel["seen"]["linear"]["pair_acc"]
                                  - novel["unseen"]["linear"]["pair_acc"])

    # normalized coverage from the frozen cache; no labels are touched
    cache = D.load_cache()
    fakes = RUN.prepare_fakes(cache)
    coverage_normalized = _normalized_coverage(res["coverage"], cache, fakes)

    controls = {
        "marginal_exact": res["marginal_exact"],
        "marginal_all_exact": all(x["max_abs_real_fake_diff"] == 0.0
                                  and x["tie_rate"] == 1.0 for x in res["marginal_exact"]),
        "path_reorder": res["path_reorder"],
        "lc_array_rebuild_all_equal": res["gates"]["lc_array_rebuild"]["all_equal"],
        "renumber_all_equal": res["gates"]["renumber"]["all_equal"],
        "molecule_isolation": {
            "folds": res["gates"]["molecule_split"]["fold_sizes"],
            "duplicate_groups_crossing_folds":
                res["gates"]["molecule_split"]["duplicate_groups_crossing_folds"],
        },
        "random_label_clean": {p: summary["random_label_clean"][p]["pair_acc"]
                               for p in summary["random_label_clean"]},
        "random_label_permuted": {p: summary["random_label_permuted"][p]["pair_acc"]
                                  for p in summary["random_label_permuted"]},
        "untrained_clean": {p: summary["untrained_clean"][p]["pair_acc"]
                            for p in summary["untrained_clean"]},
        "untrained_permuted": {p: summary["untrained_permuted"][p]["pair_acc"]
                               for p in summary["untrained_permuted"]},
        "labels_accessed": res["labels_accessed"],
        "official_test_loaded": res["official_test_loaded"],
        "cuda_used": res["cuda_used"],
        "raw_arrays": res.get("raw_arrays"),
    }

    # Strict operationalization of the requested gate.  The hard control is the
    # decisive one because it preserves local compatibility and end marginals.
    h0, h1 = hard["binding_s0"], hard["binding_s1"]
    d_freq = diffs_hard["binding_s0_minus_frequency"]
    d_lp = diffs_hard["binding_s0_minus_linear_pair_only_s0"]
    criteria = {
        "hard_above_chance": h0["ci_low"] > 0.5,
        "hard_above_frequency_ci": d_freq["ci_low"] > 0.0,
        "hard_above_linear_pair_only_ci": d_lp["ci_low"] > 0.0,
        "both_seeds_above_chance": h0["ci_low"] > 0.5 and h1["ci_low"] > 0.5,
        "seed_gap_lt_0.05": abs(h0["pair_acc"] - h1["pair_acc"]) < 0.05,
        "fold_min_above_0.5": h0["fold_min"] > 0.5,
        "enough_hard_roots": h0["n_roots"] >= 300,
    }
    nontrivial = all(criteria.values())
    verdict = "BINDING_SIGNAL_PRESENT" if nontrivial else "NO_NONTRIVIAL_BINDING_SIGNAL"

    report = {
        "round": res["round"],
        "created": res["created"],
        "env": res["env"],
        "git": res["git"],
        "args": res["args"],
        "seconds": res["seconds"],
        "coverage": res["coverage"],
        "coverage_normalized": coverage_normalized,
        "model_params": res["model_params"],
        "standard": standard,
        "hard": hard,
        "paired_diffs_standard": diffs_std,
        "paired_diffs_hard": diffs_hard,
        "novel": novel,
        "frequency_diagnostic": freq_diag,
        "context_frequency": context_freq,
        "controls": controls,
        "decision_rule": criteria,
        "verdict": verdict,
        "binding_signal_present": bool(nontrivial),
        "recommend_gpu_pretraining": False,
        "interpretation": (
            "The standard real-vs-random-permutation signal is real but trivial: "
            "the deterministic local type-compatibility rule reaches 0.9989, and "
            "the binding head with the compatibility bit matches it.  The hard "
            "control preserves that compatibility and the end-type marginal, so "
            "the oracle rule and the marginal head are exactly 0.5; the remaining "
            "hard signal is a simple pair/context frequency statistic.  A "
            "no-training context-conditioned table using root/mid degree, ring "
            "flags, min-ring, closure, atom type, and bond type reaches 0.6514 "
            "on hard, above the binding head's 0.5919.  The binding head is not "
            "reliably above the simple pair-frequency table or the pair-only "
            "linear head, and it transfers worse than the linear model to unseen "
            "local combinations.  Under the requested strict criterion this is "
            "not a nontrivial binding signal; do not start a GPU pretraining "
            "experiment on this basis."
        ),
        "limitations": [
            "The hard control covers only 370 roots in 341 molecules (0.16% of "
            "roots, 3.41% of molecules), all with 4-7 paths, 95.7% ring atoms, "
            "and only four root atom-type indices.",
            "The hard control preserves per-path compatibility but not the joint "
            "pair-type marginal; a global pair-frequency prior remains and is "
            "detected by the simple frequency table.",
            "The hard negative is a compatibility-preserving 2-swap, not a full "
            "conditional null over all compatible pairings; the pool is small.",
            "The random-label clean-test score is slightly above 0.5 because of "
            "input/initialization asymmetry; the proper permuted-label test is "
            "at chance (0.5005 / 0.5008).",
            "The context-frequency table is a post-hoc simple-statistic check; "
            "it is not part of the original 7-point gate, but it reinforces the "
            "NO verdict.",
        ],
    }
    (ANALYSIS / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in
                      ["verdict", "binding_signal_present", "decision_rule",
                       "frequency_diagnostic", "paired_diffs_hard"]},
                     indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
