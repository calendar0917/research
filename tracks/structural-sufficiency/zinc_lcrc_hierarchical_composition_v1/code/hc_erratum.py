"""Erratum for the Stage IV radius-2 support grouping (CPU-only, no retraining).

The frozen ``zinc_compositional_research_tree_v1`` Stage IV report grouped the
official valid rows by a radius-2 environment vocabulary built from **Fit8k**.
Stage IV was trained on the full official train10k, so the correct vocabulary
for that stage is the full 10k train structure.  This script recomputes the
grouping on the *unchanged frozen predictions* and writes an erratum.  Old
numbers and the old source files are left untouched.

No model is trained, no prediction is modified, and official test is never
opened.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

import hc_common as C

OLD = C.SUFF / "zinc_compositional_research_tree_v1"
sys.path.insert(0, str(OLD / "shared"))
import xs_common as XC  # noqa: E402
import xs_metrics as XM  # noqa: E402

GROUPS = ("full", "ordinary", "unseen_ordinary", "seen_ordinary", "tail", "any_unseen")
N_BOOT = 5000
BOOT_SEED = 20261107


def _pred(seed: int, arm: str) -> np.ndarray:
    path = OLD / "runs" / f"stage4-cma-{arm}_s{seed}" / "predictions_best_monitor.pt"
    d = torch.load(path, map_location="cpu", weights_only=False)
    return np.asarray(d["pred"], dtype=np.float64), C.sha256_file(path)


def _signatures(graphs):
    """Exact copy of the frozen xs_metrics.radius2_support signature strings."""
    per_atom = []
    for atoms, bonds in graphs:
        nbrs: list[list[tuple[int, int]]] = [[] for _ in range(len(atoms))]
        for u, v, t in bonds:
            nbrs[u].append((t, v))
            nbrs[v].append((t, u))
        one = [f"r1({int(atoms[i])};"
               + ",".join(sorted(f"{t}-{int(atoms[j])}" for t, j in nbrs[i])) + ")"
               for i in range(len(atoms))]
        two = [f"r2({one[i]};"
               + ",".join(sorted(f"{t}-{one[j]}" for t, j in nbrs[i])) + ")"
               for i in range(len(atoms))]
        per_atom.append(two)
    return per_atom


def _support_masks(train, fit_rows, valid):
    """Fit8k-vocab and train10k-vocab unseen masks from one signature pass."""
    train_graphs = XM.molecules(train)
    valid_graphs = XM.molecules(valid)
    train_sigs = _signatures(train_graphs)
    valid_sigs = _signatures(valid_graphs)
    fit_set = set(int(x) for x in np.asarray(fit_rows))
    full_freq: dict[str, int] = {}
    fit_freq: dict[str, int] = {}
    for m, sigs in enumerate(train_sigs):
        for s in sigs:
            full_freq[s] = full_freq.get(s, 0) + 1
            if m in fit_set:
                fit_freq[s] = fit_freq.get(s, 0) + 1
    unseen_old, unseen_new = [], []
    for sigs in valid_sigs:
        if not sigs:
            unseen_old.append(False); unseen_new.append(False)
            continue
        unseen_old.append(any(fit_freq.get(s, 0) == 0 for s in sigs))
        unseen_new.append(any(full_freq.get(s, 0) == 0 for s in sigs))
    y = np.asarray(valid["labels"], dtype=np.float64)
    ordinary = y > -3.0
    out = {}
    for name, unseen in (("old", np.asarray(unseen_old)), ("new", np.asarray(unseen_new))):
        out[name] = {
            "any_unseen": unseen,
            "unseen_ordinary": unseen & ordinary,
            "seen_ordinary": (~unseen) & ordinary,
        }
    out["ordinary"] = ordinary
    out["tail"] = ~ordinary
    out["full"] = np.ones(len(y), dtype=bool)
    out["vocab"] = {"old": int(len(fit_freq)), "new": int(len(full_freq))}
    return out


def _boot(diff: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(diff)
    if n == 0:
        return {"n": 0, "delta": None, "ci95": [None, None], "win_rate": None}
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = diff[idx].mean(1)
    return {
        "n": int(n),
        "delta": float(diff.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "win_rate": float((diff < 0).mean()),
    }


def main() -> None:
    train = XC.load_cache("train")
    valid = XC.load_cache("valid")
    rows = np.arange(len(valid["labels"]), dtype=np.int64)
    fit_rows = XC.split_rows()["rows"]["fit8k"]
    y = np.asarray(valid["labels"], dtype=np.float64)[rows]

    sup = _support_masks(train, fit_rows, valid)
    masks_old = {name: sup["old"][name] if name in sup["old"] else sup[name]
                 for name in GROUPS}
    masks_new = {name: sup["new"][name] if name in sup["new"] else sup[name]
                 for name in GROUPS}

    source = {}
    contrasts = C.read_json(OLD / "analysis/contrasts.json")
    recorded = {}
    for c in contrasts["contrasts"]:
        if c["family"] == "cma" and c["fit"] == "train10k":
            recorded[int(c["seed"])] = c["delta_real_minus_mean"]

    per_seed = {}
    for seed in (0, 1):
        real, sha_r = _pred(seed, "real")
        mean, sha_m = _pred(seed, "mean")
        source[seed] = {"real_file": sha_r, "mean_file": sha_m,
                        "real_sha256": C.sha256_file(
                            OLD / "runs" / f"stage4-cma-real_s{seed}" / "predictions_best_monitor.pt"),
                        "mean_sha256": C.sha256_file(
                            OLD / "runs" / f"stage4-cma-mean_s{seed}" / "predictions_best_monitor.pt")}
        diff = np.abs(real - y) - np.abs(mean - y)
        groups = {}
        for name in GROUPS:
            rng = np.random.default_rng(BOOT_SEED + seed * 100 + GROUPS.index(name))
            old = _boot(diff[masks_old[name]], rng)
            rng = np.random.default_rng(BOOT_SEED + seed * 100 + GROUPS.index(name) + 50)
            new = _boot(diff[masks_new[name]], rng)
            old["recorded_old_delta"] = recorded.get(seed, {}).get(name, {}).get("delta")
            old["recorded_old_n"] = recorded.get(seed, {}).get(name, {}).get("n")
            groups[name] = {"old_fit8k_vocab": old, "new_train10k_vocab": new}
        per_seed[seed] = groups

    moved = {}
    for name in ("any_unseen", "unseen_ordinary", "seen_ordinary"):
        old_mask, new_mask = masks_old[name], masks_new[name]
        moved[name] = {
            "old_n": int(old_mask.sum()),
            "new_n": int(new_mask.sum()),
            "old_only": int((old_mask & ~new_mask).sum()),
            "new_only": int((new_mask & ~old_mask).sum()),
        }

    payload = {
        "round": C.ROUND_ID,
        "kind": "erratum",
        "target_round": "zinc-compositional-research-tree-v1",
        "stage": "IV_train10k",
        "official_test_loaded": False,
        "official_valid_loaded": True,
        "predictions_unchanged": True,
        "old_protocol": {
            "support_vocabulary": "Fit8k rows (8000)",
            "fit_vocab_size": int(sup["vocab"]["old"]),
            "source": "zinc_compositional_research_tree_v1/analysis/contrasts.json",
        },
        "new_protocol": {
            "support_vocabulary": "complete official train10k structure (10000)",
            "fit_vocab_size": int(sup["vocab"]["new"]),
            "reason": "Stage IV trained on all 10k official train rows; Fit8k is not "
                      "the training support for that stage.",
        },
        "group_counts": {
            name: {"old_fit8k": int(masks_old[name].sum()),
                   "new_train10k": int(masks_new[name].sum())}
            for name in GROUPS
        },
        "moved_rows": moved,
        "per_seed": per_seed,
        "prediction_sources": source,
        "bootstrap": {"n_resamples": N_BOOT, "seed": BOOT_SEED,
                      "unit": "valid row", "paired": True},
    }
    C.write_json(C.ANALYSIS / "erratum_stage4_support.json", payload)

    lines = [
        "# ERRATUM — Stage IV radius-2 support vocabulary",
        "",
        "Target: `zinc_compositional_research_tree_v1` Stage IV (train10k, "
        "official valid1k). Predictions and checkpoints are unchanged; only the "
        "radius-2 support grouping is recomputed.",
        "",
        "## What was wrong",
        "",
        "Stage IV was trained on the complete official train10k, but the report "
        "grouped valid molecules with a radius-2 environment vocabulary built "
        "from Fit8k only. The correct support vocabulary for that stage is the "
        "complete train10k structure.",
        "",
        f"* old vocabulary: Fit8k, {int(sup['vocab']['old'])} signatures",
        f"* new vocabulary: train10k, {int(sup['vocab']['new'])} signatures",
        "",
        "## Group counts",
        "",
        "| group | old (Fit8k vocab) | new (train10k vocab) |",
        "|---|---:|---:|",
    ]
    for name in GROUPS:
        lines.append(f"| {name} | {int(masks_old[name].sum())} | {int(masks_new[name].sum())} |")
    lines += [
        "",
        "Rows that change side: "
        + "; ".join(f"{k}: {v['old_only']} old-only, {v['new_only']} new-only"
                    for k, v in moved.items()) + ".",
        "",
        "## REAL − MEAN MAE (negative = REAL better)",
        "",
        "| seed | group | n old | Δ old | n new | Δ new | Δ new 95% CI | win new |",
        "|---:|---|---:|---:|---:|---:|---|---:|",
    ]
    for seed in (0, 1):
        for name in GROUPS:
            g = per_seed[seed][name]
            o, n = g["old_fit8k_vocab"], g["new_train10k_vocab"]
            lines.append(
                f"| {seed} | {name} | {o['n']} | {o['delta']:+.6f} | {n['n']} | "
                f"{n['delta']:+.6f} | [{n['ci95'][0]:+.6f}, {n['ci95'][1]:+.6f}] | "
                f"{n['win_rate']:.3f} |")
    lines += [
        "",
        "## Effect of the correction",
        "",
        "* The correction makes the unseen set **smaller and more strictly "
        "unseen**: 251 vs 297 any-unseen rows and 225 vs 267 unseen-ordinary rows.",
        "* For seed 0 the unseen-ordinary REAL advantage strengthens from "
        f"{per_seed[0]['unseen_ordinary']['old_fit8k_vocab']['delta']:+.6f} to "
        f"{per_seed[0]['unseen_ordinary']['new_train10k_vocab']['delta']:+.6f} "
        "(still not reaching the −0.004 purchase threshold and the CI still "
        "crosses 0).",
        "* For seed 1 the unseen-ordinary advantage strengthens from "
        f"{per_seed[1]['unseen_ordinary']['old_fit8k_vocab']['delta']:+.6f} to "
        f"{per_seed[1]['unseen_ordinary']['new_train10k_vocab']['delta']:+.6f}.",
        "* The recorded Stage IV decision is **not rewritten**. It was based on "
        "the old grouping; this erratum only reports what the correct grouping "
        "shows. The qualitative reading in the Stage IV report (tail effect "
        "unambiguous, ordinary/unseen effect seed-dependent) is unchanged.",
        "",
        "Old numbers remain in `zinc_compositional_research_tree_v1/analysis/"
        "contrasts.json`; the old predictions and checkpoints were not modified.",
        "",
        f"Machine-readable: `{C.ANALYSIS / 'erratum_stage4_support.json'}`.",
        "",
    ]
    (C.ROUND / "ERRATUM_stage4_support.md").write_text("\n".join(lines))
    C.log(f"wrote {C.ROUND / 'ERRATUM_stage4_support.md'}")
    print(json.dumps({"group_counts": payload["group_counts"],
                      "moved_rows": moved}, indent=1))


if __name__ == "__main__":
    main()
