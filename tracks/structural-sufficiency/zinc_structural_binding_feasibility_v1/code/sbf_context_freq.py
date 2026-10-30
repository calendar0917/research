"""Simple degree/ring/type context-conditioned pair-frequency baseline.

This is a no-training, no-label statistic.  It answers the question: can the
hard-control signal be explained by simple atom type + bond type + degree +
ring flag + closure statistics?  The context is

    (root_deg, root_ring, root_mring, mid_deg, mid_ring, closure)

and each candidate path is scored with a Dirichlet-backoff pair frequency:

    p(pair | context) = (count(context, pair) + alpha * p_global(pair))
                        / (count(context) + alpha)

with alpha = 20.  No model is trained; only training-molecule counts are used.

Outputs analysis/context_freq.json and analysis/context_freq_raw.npz.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

TRACK = Path(__file__).resolve().parents[1]
ANALYSIS = TRACK / "analysis"
sys.path.insert(0, str(TRACK / "code"))
import sbf_common as C  # noqa: E402
import sbf_data as D  # noqa: E402
import sbf_run as R  # noqa: E402
import sbf_analysis as A  # noqa: E402

ALPHA = 20.0
V_PAIR = C.N_MID_CODES * C.N_END_TYPES


def _context(r: dict[str, Any], i: int) -> tuple[int, ...]:
    return (int(r["root_deg"]), int(r["root_ring"]), int(r["root_mring"]),
            int(r["mid_deg"][i]), int(r["mid_ring"][i]), int(r["closure"][i]))


def _build_counts(cache: dict[str, Any], train_root: np.ndarray):
    global_counts: collections.Counter = collections.Counter()
    ctx_counts: dict[tuple[int, ...], collections.Counter] = collections.defaultdict(collections.Counter)
    ctx_total: collections.Counter = collections.Counter()
    for ri in train_root:
        r = D.root_slice(cache, int(ri))
        for i in range(r["n"]):
            pair = (int(r["mid"][i]), int(r["end"][i]))
            ctx = _context(r, i)
            global_counts[pair] += 1
            ctx_counts[ctx][pair] += 1
            ctx_total[ctx] += 1
    total = sum(global_counts.values())
    p_global = {pair: (global_counts.get(pair, 0) + 1.0) / (total + V_PAIR)
                for pair in ((m, e) for m in range(C.N_MID_CODES) for e in range(C.N_END_TYPES))}
    return ctx_counts, ctx_total, p_global


def _score(r: dict[str, Any], ends: np.ndarray, ctx_counts, ctx_total, p_global) -> float:
    s = 0.0
    for i in range(r["n"]):
        pair = (int(r["mid"][i]), int(ends[i]))
        ctx = _context(r, i)
        pg = p_global[pair]
        p = (ctx_counts[ctx].get(pair, 0) + ALPHA * pg) / (ctx_total[ctx] + ALPHA)
        s += float(np.log(p))
    return s


def _correct(cache: dict[str, Any], root_ids: np.ndarray, fakes: list[np.ndarray | None],
             ctx_counts, ctx_total, p_global) -> np.ndarray:
    out = np.empty(len(root_ids), dtype=np.float64)
    for k, ri in enumerate(root_ids):
        r = D.root_slice(cache, int(ri))
        real = _score(r, np.asarray(r["end"], dtype=np.int64), ctx_counts, ctx_total, p_global)
        fake = _score(r, np.asarray(fakes[k], dtype=np.int64), ctx_counts, ctx_total, p_global)
        out[k] = 1.0 if real > fake else (0.5 if real == fake else 0.0)
    return out


def main() -> None:
    cache = D.load_cache()
    fakes = R.prepare_fakes(cache)
    splits = D.assign_folds(cache)
    root_mol = np.asarray(cache["root_mol"], dtype=np.int64)
    std = np.asarray([f is not None for f in fakes["standard"]])
    hard = np.asarray([f is not None for f in fakes["hard"]])

    std_correct, std_mol, hard_correct, hard_mol = [], [], [], []
    std_fold, hard_fold = [], []
    for f in range(C.N_FOLDS):
        train_mols, _, test_mols = D.split_indices(splits["fold"], f)
        train_root = np.where(std & np.isin(root_mol, train_mols))[0]
        test_root = np.where(std & np.isin(root_mol, test_mols))[0]
        hard_root = np.where(hard & np.isin(root_mol, test_mols))[0]
        ctx_counts, ctx_total, p_global = _build_counts(cache, train_root)
        cs = _correct(cache, test_root, [fakes["standard"][ri] for ri in test_root],
                      ctx_counts, ctx_total, p_global)
        ch = _correct(cache, hard_root, [fakes["hard"][ri] for ri in hard_root],
                      ctx_counts, ctx_total, p_global)
        std_correct.append(cs); std_mol.append(root_mol[test_root])
        hard_correct.append(ch); hard_mol.append(root_mol[hard_root])
        std_fold.append(float(cs.mean())); hard_fold.append(float(ch.mean()))
        print(f"[context-freq fold {f}] std {cs.mean():.4f} ({len(cs)}) "
              f"hard {ch.mean():.4f} ({len(ch)})")

    std_correct_c = np.concatenate(std_correct)
    std_mol_c = np.concatenate(std_mol)
    hard_correct_c = np.concatenate(hard_correct)
    hard_mol_c = np.concatenate(hard_mol)
    np.savez_compressed(ANALYSIS / "context_freq_raw.npz",
                        std_correct=std_correct_c, std_mol=std_mol_c,
                        hard_correct=hard_correct_c, hard_mol=hard_mol_c)

    raw = np.load(ANALYSIS / "raw_arrays.npz")
    b_std = np.asarray(raw["standard__binding_s0_correct"], dtype=np.float64)
    f_std = np.asarray(raw["frequency__frequency_correct"], dtype=np.float64)
    b_hard = np.asarray(raw["hard__binding_s0_correct"], dtype=np.float64)
    f_hard = np.asarray(raw["hard_frequency__frequency_correct"], dtype=np.float64)
    b_std_mol = np.asarray(raw["standard__binding_s0_mol"], dtype=np.int64)
    b_hard_mol = np.asarray(raw["hard__binding_s0_mol"], dtype=np.int64)
    assert len(b_std) == len(std_correct_c) and len(b_hard) == len(hard_correct_c)
    assert np.array_equal(b_std_mol, std_mol_c) and np.array_equal(b_hard_mol, hard_mol_c)

    out = {
        "alpha": ALPHA,
        "context": ["root_deg", "root_ring", "root_mring", "mid_deg", "mid_ring", "closure"],
        "standard": {
            "pair_acc": float(std_correct_c.mean()),
            "fold_acc": [float(x) for x in std_fold],
            **A._acc_ci(std_correct_c, std_mol_c, "ctx-std"),
            "minus_binding_s0": A._paired_diff(std_correct_c, b_std, std_mol_c, "ctx-std-b"),
            "minus_frequency": A._paired_diff(std_correct_c, f_std, std_mol_c, "ctx-std-f"),
        },
        "hard": {
            "pair_acc": float(hard_correct_c.mean()),
            "fold_acc": [float(x) for x in hard_fold],
            **A._acc_ci(hard_correct_c, hard_mol_c, "ctx-hard"),
            "minus_binding_s0": A._paired_diff(hard_correct_c, b_hard, hard_mol_c, "ctx-hard-b"),
            "minus_frequency": A._paired_diff(hard_correct_c, f_hard, hard_mol_c, "ctx-hard-f"),
        },
        "n_standard_roots": int(len(std_correct_c)),
        "n_hard_roots": int(len(hard_correct_c)),
    }
    (ANALYSIS / "context_freq.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
