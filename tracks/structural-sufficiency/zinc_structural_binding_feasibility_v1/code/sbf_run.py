"""CPU feasibility gate: real LCRC mid--end binding vs marginal controls.

Run:  uv run python tracks/structural-sufficiency/zinc_structural_binding_feasibility_v1/code/sbf_run.py --stage all
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import sbf_common as C
import sbf_data as D
import sbf_models as M

torch.set_num_threads(8)
LABELS = None  # populated as a tensor of [1, 0, 1, 0, ...] for a pair bundle


# ---------------------------------------------------------------------------
# tensor bundles
# ---------------------------------------------------------------------------

def build_bundle(cache: Mapping[str, Any], root_ids: Sequence[int],
                 fakes: Sequence[np.ndarray | None]) -> dict[str, Any]:
    """Build a real/fake pair bundle.  Row 2k is real, row 2k+1 is fake."""
    root_ids = [int(x) for x in root_ids]
    n_roots = len(root_ids)
    n_rows = 2 * n_roots
    maxp = C.MAX_PATHS
    out = {
        "mid": np.zeros((n_rows, maxp), dtype=np.int64),
        "end": np.zeros((n_rows, maxp), dtype=np.int64),
        "mask": np.zeros((n_rows, maxp), dtype=np.float32),
        "mid_type": np.zeros((n_rows, maxp), dtype=np.int64),
        "mid_deg": np.zeros((n_rows, maxp), dtype=np.int64),
        "mid_ring": np.zeros((n_rows, maxp), dtype=np.int64),
        "closure": np.zeros((n_rows, maxp), dtype=np.int64),
        "compat": np.zeros((n_rows, maxp), dtype=np.float32),
        "root_type": np.zeros(n_rows, dtype=np.int64),
        "root_deg": np.zeros(n_rows, dtype=np.int64),
        "root_ring": np.zeros(n_rows, dtype=np.int64),
        "root_mring": np.zeros(n_rows, dtype=np.int64),
    }
    root_mol = np.zeros(n_roots, dtype=np.int64)
    root_idx = np.zeros(n_roots, dtype=np.int64)
    npaths = np.zeros(n_roots, dtype=np.int64)
    for k, ri in enumerate(root_ids):
        r = D.root_slice(cache, ri)
        root_mol[k] = int(cache["root_mol"][ri])
        root_idx[k] = ri
        npaths[k] = int(r["n"])
        fake = fakes[k]
        for row, ends in ((2 * k, r["end"]), (2 * k + 1, np.asarray(fake, dtype=np.int64))):
            n = int(r["n"])
            out["mid"][row, :n] = r["mid"]
            out["end"][row, :n] = ends
            out["mask"][row, :n] = 1.0
            out["mid_type"][row, :n] = r["mid_type"]
            out["mid_deg"][row, :n] = r["mid_deg"]
            out["mid_ring"][row, :n] = r["mid_ring"]
            out["closure"][row, :n] = r["closure"]
            allowed = r["allowed"]
            out["compat"][row, :n] = np.asarray([(int(allowed[i]) >> int(ends[i])) & 1
                                                 for i in range(n)], dtype=np.float32)
            out["root_type"][row] = r["root_type"]
            out["root_deg"][row] = r["root_deg"]
            out["root_ring"][row] = r["root_ring"]
            out["root_mring"][row] = r["root_mring"]
    tensors = {k: torch.from_numpy(v) for k, v in out.items()}
    labels = torch.zeros(n_rows, dtype=torch.float32)
    labels[0::2] = 1.0
    tensors["_labels"] = labels
    return {
        "tensors": tensors,
        "root_ids": np.asarray(root_ids, dtype=np.int64),
        "root_mol": root_mol,
        "root_idx": root_idx,
        "npaths": npaths,
        "n_roots": n_roots,
    }


def subset_tensors(t: Mapping[str, torch.Tensor], rows: torch.Tensor) -> dict[str, torch.Tensor]:
    return {k: v[rows] for k, v in t.items()}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def pair_scores(model: torch.nn.Module, tensors: Mapping[str, torch.Tensor]) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(tensors).detach().cpu().numpy().reshape(-1, 2)


def pair_metrics(scores: np.ndarray) -> dict[str, Any]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1, 2)
    real, fake = s[:, 0], s[:, 1]
    win = real > fake
    tie = np.isclose(real, fake, rtol=0.0, atol=1e-12)
    correct = win.astype(np.float64) + 0.5 * tie.astype(np.float64)
    return {
        "n_roots": int(len(s)),
        "pair_acc": float(correct.mean()) if len(s) else float("nan"),
        "win_rate": float(win.mean()) if len(s) else float("nan"),
        "tie_rate": float(tie.mean()) if len(s) else float("nan"),
        "margin_mean": float((real - fake).mean()) if len(s) else float("nan"),
        "real_mean": float(real.mean()) if len(s) else float("nan"),
        "fake_mean": float(fake.mean()) if len(s) else float("nan"),
        "_correct": correct,
    }


def frequency_table_scores(train_root: Sequence[int], cache: Mapping[str, Any],
                           eval_root: Sequence[int], fakes: Sequence[np.ndarray | None]) -> np.ndarray:
    """Simple atom/bond-type pair-frequency table (no training).

    Counts real (mid_code, end_type) pairs in the training molecules, adds-one
    smoothing, and scores a candidate pairing by the summed log frequency.
    This is the "simple atom-type + bond-type statistics" baseline.
    """
    counts = collections.Counter()
    total = 0
    for ri in train_root:
        r = D.root_slice(cache, int(ri))
        for i in range(r["n"]):
            counts[(int(r["mid"][i]), int(r["end"][i]))] += 1
            total += 1
    denom = float(total + C.N_MID_CODES * C.N_END_TYPES)

    def score(r, ends):
        return float(sum(math.log((counts.get((int(r["mid"][i]), int(ends[i])), 0) + 1.0) / denom)
                         for i in range(r["n"])))

    out = np.zeros((len(eval_root), 2), dtype=np.float64)
    for k, ri in enumerate(eval_root):
        r = D.root_slice(cache, int(ri))
        out[k, 0] = score(r, r["end"])
        out[k, 1] = score(r, np.asarray(fakes[k], dtype=np.int64))
    return out


def bootstrap_ci(correct: np.ndarray, mol_ids: np.ndarray, *, n_boot: int = 2000,
                 seed: int = 0) -> dict[str, Any]:
    correct = np.asarray(correct, dtype=np.float64)
    mol_ids = np.asarray(mol_ids, dtype=np.int64)
    uniq, inv = np.unique(mol_ids, return_inverse=True)
    m = len(uniq)
    if m == 0:
        return {"n_molecules": 0, "ci_low": float("nan"), "ci_high": float("nan"),
                "boot_mean": float("nan"), "boot_std": float("nan")}
    sums = np.bincount(inv, weights=correct, minlength=m)
    counts = np.bincount(inv, minlength=m).astype(np.float64)
    rng = np.random.default_rng(int(seed))
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        samp = rng.integers(0, m, size=m)
        denom = counts[samp].sum()
        stats[b] = sums[samp].sum() / denom if denom > 0 else np.nan
    stats = stats[np.isfinite(stats)]
    return {
        "n_molecules": int(m),
        "ci_low": float(np.percentile(stats, 2.5)),
        "ci_high": float(np.percentile(stats, 97.5)),
        "boot_mean": float(stats.mean()),
        "boot_std": float(stats.std()),
    }


def add_ci(metrics: dict[str, Any], mol_ids: np.ndarray, seed: int) -> dict[str, Any]:
    out = dict(metrics)
    del out["_correct"]
    ci = bootstrap_ci(metrics["_correct"], mol_ids, seed=seed)
    out.update(ci)
    return out


def group_metrics(scores: np.ndarray, mask: np.ndarray, mol_ids: np.ndarray,
                  *, seed: int) -> dict[str, Any]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return {"n_roots": 0, "pair_acc": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_molecules": 0}
    m = pair_metrics(scores[idx])
    return add_ci(m, mol_ids[idx], seed)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train_model(model: torch.nn.Module, train: Mapping[str, torch.Tensor],
                *, seed: int, epochs: int = 10, batch_size: int = 8192,
                labels: torch.Tensor | None = None,
                orders: list[torch.Tensor] | None = None,
                lr: float = 2e-3) -> torch.nn.Module:
    y = labels if labels is not None else train["_labels"]
    n = y.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if orders is None:
        orders = [torch.randperm(n, generator=torch.Generator().manual_seed(
            int(seed) * 100003 + ep)) for ep in range(epochs)]
    for ep in range(epochs):
        model.train()
        order = orders[ep]
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            b = {k: v[idx] for k, v in train.items() if k != "_labels"}
            loss = F.binary_cross_entropy_with_logits(model(b), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


def make_orders(n: int, seed: int, epochs: int) -> list[torch.Tensor]:
    return [torch.randperm(n, generator=torch.Generator().manual_seed(
        int(seed) * 100003 + ep)) for ep in range(epochs)]


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def lc_array_gate(cache: Mapping[str, Any]) -> dict[str, Any]:
    frozen = C.load_frozen_lc_train()
    keys = [
        "atom_ptr", "bond_ptr", "nbr_ptr", "path_ptr", "pair_ptr",
        "atom_feat", "atom_type", "atom_scalar", "bond_feat", "bond_type",
        "bond_index", "nbr_code", "path_cat", "path_end63", "path_end",
        "path_closure", "pair_i", "pair_j", "pair_cat", "global",
    ]
    out = {"keys": {}, "all_equal": True}
    ours = cache["lc_arrays"]
    for k in keys:
        a = np.asarray(ours[k])
        b = np.asarray(frozen[k])
        if a.shape != b.shape:
            out["all_equal"] = False
            out["keys"][k] = {"shape": [list(a.shape), list(b.shape)], "equal": False}
            continue
        if a.dtype.kind in "fc":
            diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            eq = bool(np.array_equal(a, b))
        else:
            diff = 0.0 if np.array_equal(a, b) else float(np.max(np.abs(a.astype(np.int64) - b.astype(np.int64))))
            eq = bool(np.array_equal(a, b))
        out["keys"][k] = {"equal": eq, "max_abs_diff": diff}
        out["all_equal"] = out["all_equal"] and eq
    out["frozen_cache_sha256"] = frozen["_sha256"]
    out["frozen_cache_labels_poisoned"] = isinstance(frozen.get("labels"), C._Poison)
    out["frozen_cache_label_accessed"] = False
    out["expected_paths"] = 692716
    out["path_count"] = int(cache["meta"]["n_paths"])
    out["path_count_ok"] = int(cache["meta"]["n_paths"]) == 692716
    return out


def renumber_gate(cache: Mapping[str, Any], n_molecules: int = 500) -> dict[str, Any]:
    """Random atom renumbering: canonical root-signature multiset must not move."""
    import structural_relation_objects_v1 as sro

    raws = D.load_raw_train()
    rng = np.random.default_rng(int.from_bytes(
        hashlib.sha256(C.SALT_RENUMBER.encode()).digest()[:8], "little"))
    mismatches = 0
    roots_compared = 0
    mols = 0
    for mi in range(min(n_molecules, len(raws))):
        raw = raws[mi]
        n = len(raw["x"])
        perm = rng.permutation(n).astype(np.int64)
        # new = perm[old]
        x_new = np.empty_like(raw["x"])
        x_new[perm] = raw["x"]
        ei_new = perm[raw["edge_index"]]
        obj = sro.build_mol_objects(mi, x_new, ei_new, raw["edge_attr"])
        lc_new = C.LD.build_molecule_from_object(obj)
        b_new = D.build_binding_molecule(lc_new)
        # original binding records from the cache
        root_ids = np.where(np.asarray(cache["root_mol"]) == mi)[0]
        sig_old = collections.Counter()
        for ri in root_ids:
            sig_old[D.canonical_signature(D.root_slice(cache, ri))] += 1
        sig_new = collections.Counter()
        for r in b_new["roots"]:
            rr = {
                "mid": np.asarray([p["mid"] for p in r["paths"]], dtype=np.int64),
                "end": np.asarray([p["end"] for p in r["paths"]], dtype=np.int64),
                "mid_type": np.asarray([p["mid_type"] for p in r["paths"]], dtype=np.int64),
                "mid_deg": np.asarray([p["mid_deg"] for p in r["paths"]], dtype=np.int64),
                "mid_ring": np.asarray([p["mid_ring"] for p in r["paths"]], dtype=np.int64),
                "closure": np.asarray([p["closure"] for p in r["paths"]], dtype=np.int64),
                "allowed": np.asarray([p["allowed"] for p in r["paths"]], dtype=np.int64),
                "bt2": np.asarray([p["bt2"] for p in r["paths"]], dtype=np.int64),
                "root_type": int(r["type"]), "root_deg": int(r["deg"]),
                "root_ring": int(r["ring"]), "root_mring": int(r["mring"]),
                "n": len(r["paths"]),
            }
            sig_new[D.canonical_signature(rr)] += 1
        if sig_old != sig_new:
            mismatches += 1
        roots_compared += len(root_ids)
        mols += 1
    return {
        "molecules": mols,
        "roots_compared": roots_compared,
        "signature_multiset_mismatches": mismatches,
        "all_equal": mismatches == 0,
    }


def path_reorder_gate(model: torch.nn.Module, tensors: Mapping[str, torch.Tensor],
                      n_rows: int = 512) -> dict[str, Any]:
    model.eval()
    idx = torch.arange(min(n_rows, tensors["mid"].shape[0]))
    b = {k: v[idx] for k, v in tensors.items() if k != "_labels"}
    with torch.no_grad():
        s0 = model(b).cpu().numpy()
    rng = np.random.default_rng(20261110)
    bp = {k: v.clone() for k, v in b.items()}
    for i in range(bp["mid"].shape[0]):
        n = int(bp["mask"][i].sum().item())
        p = torch.from_numpy(rng.permutation(n))
        for key in ("mid", "end", "mask", "mid_type", "mid_deg", "mid_ring",
                    "closure", "compat"):
            bp[key][i, :n] = bp[key][i, :n][p]
    with torch.no_grad():
        s1 = model(bp).cpu().numpy()
    return {
        "rows": int(len(s0)),
        "max_abs_diff": float(np.max(np.abs(s0 - s1))),
        "all_equal": bool(np.array_equal(s0, s1)),
    }


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def _top_share(counts: np.ndarray, frac: float) -> float:
    counts = np.sort(np.asarray(counts, dtype=np.float64))[::-1]
    k = max(1, int(math.ceil(frac * len(counts))))
    return float(counts[:k].sum() / max(counts.sum(), 1.0))


def coverage(cache: Mapping[str, Any], standard_fake: Sequence[np.ndarray | None],
             hard_fake: Sequence[np.ndarray | None],
             standard_both_compat: np.ndarray) -> dict[str, Any]:
    root_mol = np.asarray(cache["root_mol"])
    n_roots = len(root_mol)
    n_mol = int(cache["n_molecules"])
    std = np.asarray([f is not None for f in standard_fake])
    hard = np.asarray([f is not None for f in hard_fake])
    both = np.asarray(standard_both_compat, dtype=bool)
    std_mols = np.unique(root_mol[std])
    hard_mols = np.unique(root_mol[hard])
    per_mol_std = np.bincount(root_mol[std], minlength=n_mol)
    per_mol_hard = np.bincount(root_mol[hard], minlength=n_mol)
    npaths = np.diff(np.asarray(cache["root_ptr"], dtype=np.int64))

    def dist(mask, arr):
        c = collections.Counter(int(x) for x in arr[mask])
        return {str(k): int(v) for k, v in sorted(c.items())}

    pair_counts_std = collections.Counter()
    pair_counts_hard = collections.Counter()
    for ri in np.where(std)[0]:
        r = D.root_slice(cache, ri)
        for i in range(r["n"]):
            pair_counts_std[(int(r["mid"][i]), int(r["end"][i]))] += 1
    for ri in np.where(hard)[0]:
        r = D.root_slice(cache, ri)
        for i in range(r["n"]):
            pair_counts_hard[(int(r["mid"][i]), int(r["end"][i]))] += 1
    hard_changed = []
    for ri in np.where(hard)[0]:
        r = D.root_slice(cache, ri)
        hard_changed.append(int(np.sum(r["end"] != np.asarray(hard_fake[ri]))))
    return {
        "n_molecules": n_mol,
        "n_roots_total": int(n_roots),
        "n_paths_total": int(cache["meta"]["n_paths"]),
        "n_atoms_total": int(cache["meta"]["n_atoms"]),
        "n_bonds_total": int(cache["meta"]["n_bonds"]),
        "standard_valid_roots": int(std.sum()),
        "standard_valid_root_fraction": float(std.mean()),
        "standard_valid_molecules": int(len(std_mols)),
        "standard_valid_molecule_fraction": float(len(std_mols) / n_mol),
        "standard_both_compatible_roots": int(both.sum()),
        "standard_both_compatible_fraction_of_valid": float(both.sum() / max(std.sum(), 1)),
        "hard_valid_roots": int(hard.sum()),
        "hard_valid_root_fraction": float(hard.mean()),
        "hard_valid_molecules": int(len(hard_mols)),
        "hard_valid_molecule_fraction": float(len(hard_mols) / n_mol),
        "hard_valid_fraction_of_standard_valid": float(hard.sum() / max(std.sum(), 1)),
        "hard_changed_paths": {
            "min": int(min(hard_changed)) if hard_changed else None,
            "max": int(max(hard_changed)) if hard_changed else None,
            "mean": float(np.mean(hard_changed)) if hard_changed else None,
        },
        "standard_root_type": dist(std, np.asarray(cache["root_type"])),
        "hard_root_type": dist(hard, np.asarray(cache["root_type"])),
        "standard_root_ring": dist(std, np.asarray(cache["root_ring"])),
        "hard_root_ring": dist(hard, np.asarray(cache["root_ring"])),
        "standard_root_mring": dist(std, np.asarray(cache["root_mring"])),
        "hard_root_mring": dist(hard, np.asarray(cache["root_mring"])),
        "standard_npaths": dist(std, npaths),
        "hard_npaths": dist(hard, npaths),
        "standard_top1pct_molecule_share": _top_share(per_mol_std, 0.01),
        "standard_top5pct_molecule_share": _top_share(per_mol_std, 0.05),
        "hard_top1pct_molecule_share": _top_share(per_mol_hard, 0.01),
        "hard_top5pct_molecule_share": _top_share(per_mol_hard, 0.05),
        "hard_max_roots_per_molecule": int(per_mol_hard.max()),
        "standard_max_roots_per_molecule": int(per_mol_std.max()),
        "standard_pair_types": int(len(pair_counts_std)),
        "hard_pair_types": int(len(pair_counts_hard)),
        "standard_top_pairs": [
            {"mid": int(k[0]), "end": int(k[1]), "count": int(v)}
            for k, v in pair_counts_std.most_common(10)
        ],
        "hard_top_pairs": [
            {"mid": int(k[0]), "end": int(k[1]), "count": int(v)}
            for k, v in pair_counts_hard.most_common(10)
        ],
    }


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------

def build_or_load(*, rebuild: bool = False) -> dict[str, Any]:
    if not rebuild:
        try:
            cache = D.load_cache()
            C.log(f"[sbf] loaded binding cache {cache['meta']['content_sha256'][:12]} "
                  f"({cache['meta']['n_roots']} roots / {cache['meta']['n_paths']} paths)")
            return cache
        except FileNotFoundError:
            pass
    C.log("[sbf] building binding cache from raw PyG train10k")
    cache = D.build_binding_cache()
    sha = D.save_cache(cache)
    C.log(f"[sbf] cache sha {sha[:12]} content {cache['meta']['content_sha256'][:12]}")
    return cache


def prepare_fakes(cache: Mapping[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    n_roots = int(cache["meta"]["n_roots"])
    standard = [None] * n_roots
    hard = [None] * n_roots
    both_compat = np.zeros(n_roots, dtype=bool)
    std_stats = {}
    for ri in range(n_roots):
        r = D.root_slice(cache, ri)
        f = D.make_standard_fake(r)
        if f is not None:
            standard[ri] = f
            st = D.fake_stats(r, f)
            both_compat[ri] = bool(st["fully_compatible_fake"])
            std_stats[ri] = st
        h = D.make_hard_fake(r)
        if h is not None:
            hard[ri] = h
        if (ri + 1) % 50000 == 0:
            C.log(f"[sbf fakes] {ri + 1}/{n_roots} {time.perf_counter() - t0:.1f}s")
    C.log(f"[sbf fakes] standard {sum(f is not None for f in standard)} "
          f"hard {sum(f is not None for f in hard)} "
          f"both-compat {int(both_compat.sum())} {time.perf_counter() - t0:.1f}s")
    return {"standard": standard, "hard": hard, "both_compat": both_compat,
            "standard_stats": std_stats}


def save_raw(results: Mapping[str, Any], path: Path) -> Path:
    """Save per-root arrays needed for paired bootstrap analyses."""
    flat: dict[str, np.ndarray] = {}
    for group, d in results.items():
        for key, val in d.items():
            if isinstance(val, list) and val and all(isinstance(x, np.ndarray) for x in val):
                flat[f"{group}__{key}"] = np.concatenate(val, axis=0)
    np.savez_compressed(path, **flat)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--quick", action="store_true", help="2 folds / 1 seed / 3 epochs")
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    C.log(f"[sbf] round {C.ROUND_ID}")
    env = C.env_state()
    C.log(f"[sbf] env {env}")
    t_start = time.perf_counter()
    cache = build_or_load(rebuild=args.rebuild)
    fakes = prepare_fakes(cache)
    splits = D.assign_folds(cache)
    C.log(f"[sbf] folds {splits['fold_sizes']} duplicate-crossing {splits['duplicate_groups_crossing_folds']}")

    # gates that do not depend on a trained model
    gates = {
        "lc_array_rebuild": lc_array_gate(cache),
        "renumber": renumber_gate(cache, n_molecules=200 if args.quick else 500),
        "binding_cache_content_sha256": cache["meta"]["content_sha256"],
        "binding_cache_file_sha256": C.sha256_file(C.CACHE / "binding_train10k.pt"),
        "labels_accessed": False,
        "official_test_loaded": False,
        "cuda_used": False,
        "molecule_split": splits,
    }
    C.log(f"[sbf] gates lc_equal={gates['lc_array_rebuild']['all_equal']} "
          f"renumber={gates['renumber']['all_equal']}")

    cov = coverage(cache, fakes["standard"], fakes["hard"], fakes["both_compat"])
    C.write_json(C.ANALYSIS / "coverage.json", cov)
    C.log(f"[sbf] coverage std {cov['standard_valid_roots']}/{cov['n_roots_total']} "
          f"hard {cov['hard_valid_roots']}/{cov['n_roots_total']}")

    # standard-valid roots and fold assignment
    root_mol = np.asarray(cache["root_mol"], dtype=np.int64)
    root_fold = splits["fold"][root_mol]
    std_valid = np.asarray([f is not None for f in fakes["standard"]])
    hard_valid = np.asarray([f is not None for f in fakes["hard"]])
    n_roots = len(std_valid)
    C.log(f"[sbf] std_valid {int(std_valid.sum())} hard_valid {int(hard_valid.sum())}")

    folds = list(range(C.N_FOLDS))
    if args.quick:
        folds = [0, 1]
    seeds = [0] if args.quick else [0, 1]
    epochs = 3 if args.quick else int(args.epochs)

    # accumulators
    results = {
        "standard": collections.defaultdict(list),
        "hard": collections.defaultdict(list),
        "novel": collections.defaultdict(list),
        "novel_raw": collections.defaultdict(list),
        "frequency": collections.defaultdict(list),
        "hard_frequency": collections.defaultdict(list),
        "random_label_clean": collections.defaultdict(list),
        "random_label_permuted": collections.defaultdict(list),
        "untrained_clean": collections.defaultdict(list),
        "untrained_permuted": collections.defaultdict(list),
    }
    per_fold = []
    model_param_counts = {}
    marginal_exact = []
    reorder_checks = []

    for f in folds:
        t_fold = time.perf_counter()
        train_mols, dev_mols, test_mols = D.split_indices(splits["fold"], f)
        train_root = np.where(std_valid & np.isin(root_mol, train_mols))[0]
        dev_root = np.where(std_valid & np.isin(root_mol, dev_mols))[0]
        test_root = np.where(std_valid & np.isin(root_mol, test_mols))[0]
        hard_test_root = np.where(hard_valid & np.isin(root_mol, test_mols))[0]
        C.log(f"[sbf fold {f}] train {len(train_root)} dev {len(dev_root)} "
              f"test {len(test_root)} hard {len(hard_test_root)} "
              f"mols {len(train_mols)}/{len(dev_mols)}/{len(test_mols)}")

        tr = build_bundle(cache, train_root, [fakes["standard"][ri] for ri in train_root])
        dv = build_bundle(cache, dev_root, [fakes["standard"][ri] for ri in dev_root])
        te = build_bundle(cache, test_root, [fakes["standard"][ri] for ri in test_root])
        hte = build_bundle(cache, hard_test_root, [fakes["hard"][ri] for ri in hard_test_root])

        # simple atom/bond-type pair-frequency table (training counts only)
        s_freq = frequency_table_scores(train_root, cache, test_root,
                                        [fakes["standard"][ri] for ri in test_root])
        mf = pair_metrics(s_freq)
        results["frequency"]["frequency_correct"].append(mf["_correct"])
        results["frequency"]["frequency_mol"].append(te["root_mol"])
        results["frequency"]["frequency_fold"].append(mf["pair_acc"])
        s_freq_h = frequency_table_scores(train_root, cache, hard_test_root,
                                          [fakes["hard"][ri] for ri in hard_test_root])
        mfh = pair_metrics(s_freq_h)
        results["hard_frequency"]["frequency_correct"].append(mfh["_correct"])
        results["hard_frequency"]["frequency_mol"].append(hte["root_mol"])
        results["hard_frequency"]["frequency_fold"].append(mfh["pair_acc"])
        results["hard_frequency"]["frequency_margins"].append(s_freq_h[:, 0] - s_freq_h[:, 1])

        # novelty flags for test roots: keys seen in the training molecules only
        train_keys = set()
        train_pairs = set()
        for ri in train_root:
            r = D.root_slice(cache, ri)
            train_keys |= D.root_keys(r)
            for i in range(r["n"]):
                train_pairs.add((int(r["mid"][i]), int(r["end"][i])))
        unseen_key = np.zeros(len(test_root), dtype=bool)
        unseen_pair = np.zeros(len(test_root), dtype=bool)
        for k, ri in enumerate(test_root):
            r = D.root_slice(cache, ri)
            unseen_key[k] = any(key not in train_keys for key in D.root_keys(r))
            unseen_pair[k] = any((int(r["mid"][i]), int(r["end"][i])) not in train_pairs
                                 for i in range(r["n"]))
        # hard roots' novelty (relative to the same train keys)
        hard_unseen_key = np.zeros(len(hard_test_root), dtype=bool)
        for k, ri in enumerate(hard_test_root):
            r = D.root_slice(cache, ri)
            hard_unseen_key[k] = any(key not in train_keys for key in D.root_keys(r))

        fold_row = {
            "fold": f,
            "n_train_roots": len(train_root), "n_dev_roots": len(dev_root),
            "n_test_roots": len(test_root), "n_hard_test_roots": len(hard_test_root),
            "n_unseen_key_test_roots": int(unseen_key.sum()),
            "n_unseen_pair_test_roots": int(unseen_pair.sum()),
            "n_unseen_key_hard_roots": int(hard_unseen_key.sum()),
            "standard_both_compat_test_roots": int(fakes["both_compat"][test_root].sum()),
            "frequency_test_acc": mf["pair_acc"],
            "frequency_hard_acc": mfh["pair_acc"],
        }

        # per-seed models
        for seed in seeds:
            orders = make_orders(len(tr["tensors"]["_labels"]), seed + 100 * f, epochs)
            # binding
            torch.manual_seed(C.MODEL_SEED_BASE + seed * 1009 + f)
            binding = M.PairHead(marginal=False, use_compat=False)
            binding = train_model(binding, tr["tensors"], seed=seed, epochs=epochs,
                                  labels=tr["tensors"]["_labels"], orders=orders)
            model_param_counts.setdefault("binding", M.n_params(binding))
            s_bind = pair_scores(binding, te["tensors"])
            m_bind = pair_metrics(s_bind)
            results["standard"][f"binding_s{seed}_correct"].append(m_bind["_correct"])
            results["standard"][f"binding_s{seed}_mol"].append(te["root_mol"])
            results["standard"][f"binding_s{seed}_fold"].append(m_bind["pair_acc"])
            results["standard"][f"binding_s{seed}_margin"].append(m_bind["margin_mean"])
            # groups
            gc = fakes["both_compat"][test_root]
            results["standard"][f"binding_s{seed}_decisive"].append(
                group_metrics(s_bind, ~gc, te["root_mol"], seed=seed))
            results["standard"][f"binding_s{seed}_ambiguous"].append(
                group_metrics(s_bind, gc, te["root_mol"], seed=seed))
            results["novel"][f"binding_s{seed}_seen"].append(
                group_metrics(s_bind, ~unseen_key, te["root_mol"], seed=seed))
            results["novel"][f"binding_s{seed}_unseen"].append(
                group_metrics(s_bind, unseen_key, te["root_mol"], seed=seed))
            results["novel"][f"binding_s{seed}_unseen_pair"].append(
                group_metrics(s_bind, unseen_pair, te["root_mol"], seed=seed))
            # hard zero-shot
            s_hard = pair_scores(binding, hte["tensors"])
            mh = pair_metrics(s_hard)
            results["hard"][f"binding_s{seed}_correct"].append(mh["_correct"])
            results["hard"][f"binding_s{seed}_mol"].append(hte["root_mol"])
            results["hard"][f"binding_s{seed}_fold"].append(mh["pair_acc"])
            results["hard"][f"binding_s{seed}_margin"].append(mh["margin_mean"])
            if seed == 0:
                results["hard"]["binding_s0_margins"].append(s_hard[:, 0] - s_hard[:, 1])
            results["hard"][f"binding_s{seed}_seen"].append(
                group_metrics(s_hard, ~hard_unseen_key, hte["root_mol"], seed=seed))
            results["hard"][f"binding_s{seed}_unseen"].append(
                group_metrics(s_hard, hard_unseen_key, hte["root_mol"], seed=seed))

            # marginal
            torch.manual_seed(C.MODEL_SEED_BASE + seed * 1009 + f)
            marginal = M.PairHead(marginal=True, use_compat=False)
            marginal = train_model(marginal, tr["tensors"], seed=seed, epochs=epochs,
                                   labels=tr["tensors"]["_labels"], orders=orders)
            model_param_counts.setdefault("marginal", M.n_params(marginal))
            s_marg = pair_scores(marginal, te["tensors"])
            mm = pair_metrics(s_marg)
            results["standard"][f"marginal_s{seed}_correct"].append(mm["_correct"])
            results["standard"][f"marginal_s{seed}_mol"].append(te["root_mol"])
            results["standard"][f"marginal_s{seed}_fold"].append(mm["pair_acc"])
            results["standard"][f"marginal_s{seed}_margin"].append(mm["margin_mean"])
            marginal_exact.append({
                "fold": f, "seed": seed,
                "max_abs_real_fake_diff": float(np.max(np.abs(s_marg[:, 0] - s_marg[:, 1]))),
                "tie_rate": mm["tie_rate"],
            })
            s_marg_hard = pair_scores(marginal, hte["tensors"])
            mhm = pair_metrics(s_marg_hard)
            results["hard"][f"marginal_s{seed}_correct"].append(mhm["_correct"])
            results["hard"][f"marginal_s{seed}_mol"].append(hte["root_mol"])
            results["hard"][f"marginal_s{seed}_fold"].append(mhm["pair_acc"])
            results["hard"][f"marginal_s{seed}_margin"].append(mhm["margin_mean"])

            # simple log-linear with compat
            torch.manual_seed(C.MODEL_SEED_BASE + seed * 1009 + f)
            linear = M.LinearPair(use_compat=True)
            linear = train_model(linear, tr["tensors"], seed=seed, epochs=epochs,
                                 labels=tr["tensors"]["_labels"], orders=orders)
            model_param_counts.setdefault("linear_compat", M.n_params(linear))
            s_lin = pair_scores(linear, te["tensors"])
            ml = pair_metrics(s_lin)
            results["standard"][f"linear_s{seed}_correct"].append(ml["_correct"])
            results["standard"][f"linear_s{seed}_mol"].append(te["root_mol"])
            results["standard"][f"linear_s{seed}_fold"].append(ml["pair_acc"])
            results["standard"][f"linear_s{seed}_margin"].append(ml["margin_mean"])
            results["novel"][f"linear_s{seed}_seen"].append(
                group_metrics(s_lin, ~unseen_key, te["root_mol"], seed=seed))
            results["novel"][f"linear_s{seed}_unseen"].append(
                group_metrics(s_lin, unseen_key, te["root_mol"], seed=seed))
            if seed == 0:
                results["novel_raw"]["binding_s0_correct"].append(m_bind["_correct"])
                results["novel_raw"]["linear_s0_correct"].append(ml["_correct"])
                results["novel_raw"]["mol"].append(te["root_mol"])
                results["novel_raw"]["seen"].append(~unseen_key)
                results["novel_raw"]["unseen"].append(unseen_key)
                results["novel_raw"]["unseen_pair"].append(unseen_pair)
            s_lin_hard = pair_scores(linear, hte["tensors"])
            mlh = pair_metrics(s_lin_hard)
            results["hard"][f"linear_s{seed}_correct"].append(mlh["_correct"])
            results["hard"][f"linear_s{seed}_mol"].append(hte["root_mol"])
            results["hard"][f"linear_s{seed}_fold"].append(mlh["pair_acc"])
            results["hard"][f"linear_s{seed}_margin"].append(mlh["margin_mean"])

            # binding head with the explicit compatibility bit (diagnostic only)
            if seed == 0:
                torch.manual_seed(C.MODEL_SEED_BASE + 5009 + f)
                bcomp = M.PairHead(marginal=False, use_compat=True)
                bcomp = train_model(bcomp, tr["tensors"], seed=seed, epochs=epochs,
                                    labels=tr["tensors"]["_labels"], orders=orders)
                model_param_counts.setdefault("binding_compat", M.n_params(bcomp))
                s_bc = pair_scores(bcomp, te["tensors"])
                mbc = pair_metrics(s_bc)
                results["standard"]["binding_compat_s0_correct"].append(mbc["_correct"])
                results["standard"]["binding_compat_s0_mol"].append(te["root_mol"])
                results["standard"]["binding_compat_s0_fold"].append(mbc["pair_acc"])
                s_bc_hard = pair_scores(bcomp, hte["tensors"])
                mbch = pair_metrics(s_bc_hard)
                results["hard"]["binding_compat_s0_correct"].append(mbch["_correct"])
                results["hard"]["binding_compat_s0_mol"].append(hte["root_mol"])
                results["hard"]["binding_compat_s0_fold"].append(mbch["pair_acc"])
                fold_row["binding_compat_s0_test_acc"] = mbc["pair_acc"]
                fold_row["binding_compat_s0_hard_acc"] = mbch["pair_acc"]

            # pair-frequency-only linear baseline (no compat feature)
            if seed == 0:
                torch.manual_seed(C.MODEL_SEED_BASE + 3009 + f)
                lpair = M.LinearPair(use_compat=False)
                lpair = train_model(lpair, tr["tensors"], seed=seed, epochs=epochs,
                                    labels=tr["tensors"]["_labels"], orders=orders)
                model_param_counts.setdefault("linear_pair_only", M.n_params(lpair))
                s_lp = pair_scores(lpair, te["tensors"])
                mlp = pair_metrics(s_lp)
                results["standard"]["linear_pair_only_s0_correct"].append(mlp["_correct"])
                results["standard"]["linear_pair_only_s0_mol"].append(te["root_mol"])
                results["standard"]["linear_pair_only_s0_fold"].append(mlp["pair_acc"])
                s_lp_h = pair_scores(lpair, hte["tensors"])
                mlph = pair_metrics(s_lp_h)
                results["hard"]["linear_pair_only_s0_correct"].append(mlph["_correct"])
                results["hard"]["linear_pair_only_s0_mol"].append(hte["root_mol"])
                results["hard"]["linear_pair_only_s0_fold"].append(mlph["pair_acc"])
                fold_row["linear_pair_only_s0_test_acc"] = mlp["pair_acc"]
                fold_row["linear_pair_only_s0_hard_acc"] = mlph["pair_acc"]

            # oracle compatibility rule
            s_or = M.oracle_compat_score(te["tensors"]).cpu().numpy().reshape(-1, 2)
            mo = pair_metrics(s_or)
            results["standard"][f"oracle_s{seed}_correct"].append(mo["_correct"])
            results["standard"][f"oracle_s{seed}_mol"].append(te["root_mol"])
            results["standard"][f"oracle_s{seed}_fold"].append(mo["pair_acc"])
            s_or_hard = M.oracle_compat_score(hte["tensors"]).cpu().numpy().reshape(-1, 2)
            moh = pair_metrics(s_or_hard)
            results["hard"][f"oracle_s{seed}_correct"].append(moh["_correct"])
            results["hard"][f"oracle_s{seed}_mol"].append(hte["root_mol"])
            results["hard"][f"oracle_s{seed}_fold"].append(moh["pair_acc"])

            # random-label control (within-root swap), binding head only
            rng = np.random.default_rng(int.from_bytes(
                hashlib.sha256((C.SALT_MODEL + f"|swap|{f}|{seed}").encode()).digest()[:8], "little"))
            n_tr = len(tr["root_mol"])
            swap = rng.integers(0, 2, size=n_tr).astype(np.float32)
            y_rand = tr["tensors"]["_labels"].clone()
            y_rand[0::2] = torch.from_numpy(1.0 - swap)
            y_rand[1::2] = torch.from_numpy(swap)
            torch.manual_seed(C.MODEL_SEED_BASE + seed * 1009 + f + 777)
            rand_model = M.PairHead(marginal=False, use_compat=False)
            rand_model = train_model(rand_model, tr["tensors"], seed=seed, epochs=epochs,
                                     labels=y_rand, orders=orders)
            s_rand = pair_scores(rand_model, te["tensors"])
            mr = pair_metrics(s_rand)
            results["random_label_clean"][f"binding_s{seed}_correct"].append(mr["_correct"])
            results["random_label_clean"][f"binding_s{seed}_mol"].append(te["root_mol"])
            results["random_label_clean"][f"binding_s{seed}_fold"].append(mr["pair_acc"])
            # proper null: evaluate the same random-label process on the test
            # pairs.  A model that learned nothing must be at 0.5 here.
            rng_t = np.random.default_rng(int.from_bytes(
                hashlib.sha256((C.SALT_MODEL + f"|swap_test|{f}|{seed}").encode()).digest()[:8],
                "little"))
            n_te = len(te["root_mol"])
            swap_t = rng_t.integers(0, 2, size=n_te).astype(np.float32)
            tie_t = np.isclose(s_rand[:, 0], s_rand[:, 1], rtol=0.0, atol=1e-12).astype(np.float64)
            correct_perm = np.where(swap_t == 0, s_rand[:, 0] > s_rand[:, 1],
                                    s_rand[:, 0] < s_rand[:, 1]).astype(np.float64)
            correct_perm = correct_perm + 0.5 * tie_t
            results["random_label_permuted"][f"binding_s{seed}_correct"].append(correct_perm)
            results["random_label_permuted"][f"binding_s{seed}_mol"].append(te["root_mol"])
            results["random_label_permuted"][f"binding_s{seed}_fold"].append(float(correct_perm.mean()))

            # untrained model: the initialization alone already has a small
            # real/fake bias because the input distribution is asymmetric.
            if seed == 0:
                torch.manual_seed(C.MODEL_SEED_BASE + 9009 + f)
                untrained = M.PairHead(marginal=False, use_compat=False)
                s_un = pair_scores(untrained, te["tensors"])
                mu = pair_metrics(s_un)
                results["untrained_clean"]["binding_s0_correct"].append(mu["_correct"])
                results["untrained_clean"]["binding_s0_mol"].append(te["root_mol"])
                results["untrained_clean"]["binding_s0_fold"].append(mu["pair_acc"])
                tie_u = np.isclose(s_un[:, 0], s_un[:, 1], rtol=0.0, atol=1e-12).astype(np.float64)
                correct_u = np.where(swap_t == 0, s_un[:, 0] > s_un[:, 1],
                                     s_un[:, 0] < s_un[:, 1]).astype(np.float64) + 0.5 * tie_u
                results["untrained_permuted"]["binding_s0_correct"].append(correct_u)
                results["untrained_permuted"]["binding_s0_mol"].append(te["root_mol"])
                results["untrained_permuted"]["binding_s0_fold"].append(float(correct_u.mean()))
                fold_row["untrained_s0_test_acc"] = mu["pair_acc"]
                fold_row["untrained_s0_permuted_acc"] = float(correct_u.mean())

            # path reorder gate on the trained binding model
            if f == folds[0] and seed == seeds[0]:
                reorder_checks.append(path_reorder_gate(binding, te["tensors"], n_rows=512))

            fold_row[f"binding_s{seed}_test_acc"] = m_bind["pair_acc"]
            fold_row[f"marginal_s{seed}_test_acc"] = mm["pair_acc"]
            fold_row[f"linear_s{seed}_test_acc"] = ml["pair_acc"]
            fold_row[f"oracle_s{seed}_test_acc"] = mo["pair_acc"]
            fold_row[f"binding_s{seed}_hard_acc"] = mh["pair_acc"]
            fold_row[f"linear_s{seed}_hard_acc"] = mlh["pair_acc"]
            fold_row[f"marginal_s{seed}_hard_acc"] = mhm["pair_acc"]
            fold_row[f"oracle_s{seed}_hard_acc"] = moh["pair_acc"]
            fold_row[f"random_s{seed}_test_acc"] = mr["pair_acc"]

        per_fold.append(fold_row)
        C.log(f"[sbf fold {f}] {fold_row} ({time.perf_counter() - t_fold:.1f}s)")

    # aggregate
    def aggregate(prefix: str, group: str) -> dict[str, Any]:
        corr = np.concatenate(results[group][f"{prefix}_correct"])
        mol = np.concatenate(results[group][f"{prefix}_mol"])
        # pair_metrics expects scores; build directly
        acc = float(corr.mean())
        ci = bootstrap_ci(corr, mol, seed=int.from_bytes(
            hashlib.sha256((C.SALT_BOOTSTRAP + "|" + group + "|" + prefix).encode()).digest()[:4],
            "little"))
        folds = results[group].get(f"{prefix}_fold", [])
        return {
            "n_roots": int(len(corr)),
            "pair_acc": acc,
            "fold_acc": [float(x) for x in folds],
            "fold_mean": float(np.mean(folds)) if folds else float("nan"),
            "fold_std": float(np.std(folds)) if folds else float("nan"),
            **ci,
        }

    summary = {
        "standard": {},
        "hard": {},
        "novel": {},
        "random_label_clean": {},
        "random_label_permuted": {},
        "untrained_clean": {},
        "untrained_permuted": {},
    }
    for prefix in ["binding_s0", "binding_s1", "marginal_s0", "marginal_s1",
                   "linear_s0", "linear_s1", "oracle_s0", "oracle_s1",
                   "binding_compat_s0", "linear_pair_only_s0"]:
        if f"{prefix}_correct" in results["standard"]:
            summary["standard"][prefix] = aggregate(prefix, "standard")
    for prefix in ["binding_s0", "binding_s1", "marginal_s0", "marginal_s1",
                   "linear_s0", "linear_s1", "oracle_s0", "oracle_s1",
                   "binding_compat_s0", "linear_pair_only_s0"]:
        if f"{prefix}_correct" in results["hard"]:
            summary["hard"][prefix] = aggregate(prefix, "hard")
    if "frequency_correct" in results["frequency"]:
        summary["standard"]["frequency"] = aggregate("frequency", "frequency")
    if "frequency_correct" in results["hard_frequency"]:
        summary["hard"]["frequency"] = aggregate("frequency", "hard_frequency")

    for prefix in ["binding_s0", "linear_s0"]:
        for sub in ["seen", "unseen", "unseen_pair"]:
            key = f"{prefix}_{sub}"
            if key in results["novel"]:
                summary["novel"][key] = {
                    "folds": results["novel"][key],
                    "n_roots": int(sum(r["n_roots"] for r in results["novel"][key])),
                }
    for sub in ["seen", "unseen"]:
        key = f"binding_s0_{sub}"
        if key in results["hard"]:
            summary["hard"][key] = {
                "folds": results["hard"][key],
                "n_roots": int(sum(r["n_roots"] for r in results["hard"][key])),
            }
    for group, out_key in [("random_label_clean", "random_label_clean"),
                           ("random_label_permuted", "random_label_permuted"),
                           ("untrained_clean", "untrained_clean"),
                           ("untrained_permuted", "untrained_permuted")]:
        for prefix in ["binding_s0", "binding_s1"]:
            if f"{prefix}_correct" in results[group]:
                summary[out_key][prefix] = aggregate(prefix, group)

    out = {
        "round": C.ROUND_ID,
        "created": C.now(),
        "env": env,
        "git": C.git_state(),
        "args": vars(args),
        "gates": gates,
        "coverage": cov,
        "model_params": model_param_counts,
        "per_fold": per_fold,
        "summary": summary,
        "marginal_exact": marginal_exact,
        "path_reorder": reorder_checks,
        "seconds": float(time.perf_counter() - t_start),
        "official_test_loaded": False,
        "labels_accessed": False,
        "cuda_used": False,
    }
    raw_path = save_raw(results, C.ANALYSIS / "raw_arrays.npz")
    out["raw_arrays"] = {
        "path": raw_path.name,
        "sha256": C.sha256_file(raw_path),
        "bytes": int(raw_path.stat().st_size),
    }
    C.write_json(C.ANALYSIS / "results.json", out)
    C.log(f"[sbf] wrote {C.ANALYSIS / 'results.json'} ({out['seconds']:.1f}s)")
    print(C.read_json(C.ANALYSIS / "results.json")["summary"])


if __name__ == "__main__":
    main()
