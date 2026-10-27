"""Shared, label-free support/alpha machinery for SCSB v1.

The only inputs to the support table are (i) the frozen official-train cache and
(ii) the model's Fit row list.  No label, no prediction, no Monitor/Locked/valid
structure enters the table.  Official test is never referenced.

The radius-2 signature function is imported from the frozen DCC-L bottleneck
audit (``dca_common.radius2_signature_sets``), which is the same construction
used by ``zinc_error_mechanism_audit_v1``.  The support count is a document
frequency over Fit molecules, matching the audit's ``signature_coverage``.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
SUFF = ROUND.parent
REPO = SUFF.parents[1]
CFG_PATH = ROUND / "configs" / "scb_v1.json"
ANALYSIS = ROUND / "analysis"
CACHE_DIR = ROUND / "cache"
RUNS = ROUND / "runs"

_V1 = SUFF / "zinc_arch_discovery_v1" / "code"
_DCB = SUFF / "zinc_dccl_bottleneck_v1" / "code"
_DCC = SUFF / "zinc_dual_channel_composer_v1" / "code"
for _p in (str(_V1), str(_DCB), str(_DCC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import dca_common as DC  # noqa: E402
import dcb_train as DB  # noqa: E402
import dcc_models as M  # noqa: E402
import dcc_runner as DR  # noqa: E402

BJOC = REPO / "tracks/structure-contribution/results/bjoc_predictive_bottleneck_v1"
CYCLE_TRAIN_LABEL = REPO / "tracks/ksvd/results/zinc_long_cycle_audit/train_cycle_audit_label.csv"
CYCLE_VALID_LABEL = REPO / "tracks/ksvd/results/zinc_long_cycle_audit/valid_cycle_audit_label.csv"
EXACT_LONGEST = REPO / "tracks/ksvd/results/zinc_long_cycle_audit/exact_longest_all.csv"

DCB_RUNS = SUFF / "zinc_dccl_bottleneck_v1" / "runs"
DCC_RUNS = SUFF / "zinc_dual_channel_composer_v1" / "runs"
PERF_RUNS = SUFF / "zinc_performance_search_v1" / "runs"

# Frozen group constants (Phase 0 recomputes and asserts these).
LOCKED_TIER_COUNTS = {"unseen": 287, "low4": 401, "full": 312}
MONITOR_TIER_COUNTS = {"unseen": 317, "low4": 359, "full": 324}
HELDOUT_TIER_COUNTS = {"unseen": 604, "low4": 760, "full": 636}
NATURAL_WEIGHTS = {
    "unseen": 604.0 / 2000.0,
    "low4": 760.0 / 2000.0,
    "full": 636.0 / 2000.0,
}
PHASE1_PRIMARY_N = 278
PHASE2_PRIMARY_N = 241


# --------------------------------------------------------------------------- io
def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_cfg() -> dict:
    return read_json(CFG_PATH)


def load_train_cache() -> dict:
    return D.load_cache("train")


def load_valid_cache() -> dict:
    return D.load_cache("valid")


def eval_with_alpha(
    model: torch.nn.Module,
    cache: Mapping[str, Any],
    rows: np.ndarray,
    alpha: np.ndarray | None,
    device,
    *,
    batch_size: int = 256,
) -> tuple[np.ndarray, float]:
    """Evaluate with an optional per-bond alpha array.

    The collate preserves row order and concatenates each molecule's bonds in
    cache order, so ``alpha[global_bond_idx]`` is exactly the batch-local alpha
    vector in bond order.  This is asserted before every forward.
    """
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    out = []
    with C.isolated_rng():
        for start in range(0, len(rows), int(batch_size)):
            chunk = rows[start:start + int(batch_size)]
            batch = T1._to_device(DR.collate_p3n(cache, chunk, "P3N", valid=True), device)
            if alpha is None:
                pred = model(batch)
            else:
                gidx = np.concatenate([np.arange(bp[r], bp[r + 1]) for r in chunk])
                if len(gidx) != int(batch["bond_feat"].shape[0]):
                    raise RuntimeError(
                        f"bond order mismatch: {len(gidx)} vs {int(batch['bond_feat'].shape[0])}"
                    )
                a = torch.as_tensor(alpha[gidx], dtype=torch.float32, device=device)
                pred = model(batch, a)
            out.append(pred.detach().float().cpu().numpy().reshape(-1))
    pred = np.concatenate(out, 0) if out else np.zeros(0, dtype=np.float32)
    y = np.asarray(cache["labels"], dtype=np.float64)[rows]
    return pred, mae(pred, y)


# ------------------------------------------------------------------------ split
def split_rows() -> dict:
    split = read_json(BJOC / "split_a.json")
    fit = np.asarray(split["fit_rows"], dtype=np.int64)
    monitor = np.asarray(split["monitor_rows"], dtype=np.int64)
    locked = np.asarray(split["locked_rows"], dtype=np.int64)
    payload = {
        "fit": fit,
        "monitor": monitor,
        "locked": locked,
        "split_sha256": sha256_file(BJOC / "split_a.json"),
        "hashes": {
            "fit": sha256_array(fit),
            "monitor": sha256_array(monitor),
            "locked": sha256_array(locked),
        },
    }
    cfg = load_cfg()["data"]
    assert len(fit) == cfg["fit_rows"] == 8000, (len(fit), cfg["fit_rows"])
    assert len(monitor) == cfg["monitor_rows"] == 1000
    assert len(locked) == cfg["locked_rows"] == 1000
    assert payload["hashes"]["fit"] == cfg["fit_rows_sha256"]
    assert payload["hashes"]["monitor"] == cfg["monitor_rows_sha256"]
    assert payload["hashes"]["locked"] == cfg["locked_rows_sha256"]
    assert payload["split_sha256"] == cfg["split_sha256"]
    assert not (set(fit.tolist()) & set(monitor.tolist()))
    assert not (set(fit.tolist()) & set(locked.tolist()))
    assert not (set(monitor.tolist()) & set(locked.tolist()))
    assert len(set(fit.tolist()) | set(monitor.tolist()) | set(locked.tolist())) == 10000
    return payload


# -------------------------------------------------------------------- signatures
def build_signatures(cache: Mapping[str, Any]) -> tuple[list[set[str]], list[list[str]]]:
    """Exact audit k2 signatures for every molecule in the cache.

    ``dca_common.radius2_signature_sets`` is the frozen bottleneck-audit
    implementation.  We record its sha256 in the support manifest so a future
    change of that file cannot silently change the support definition.
    """
    graphs = DC.molecule_graphs(cache)
    sets, per_atom = DC.radius2_signature_sets(graphs)
    return sets, per_atom


def build_support(
    cache: Mapping[str, Any],
    fit_rows: np.ndarray,
    per_atom: Sequence[Sequence[str]],
) -> dict:
    """Fit-only document-frequency support table for all cache atoms."""
    df: Counter[str] = Counter()
    for r in np.asarray(fit_rows, dtype=np.int64).tolist():
        df.update(set(per_atom[int(r)]))
    vocab = sorted(df)
    index = {s: i for i, s in enumerate(vocab)}
    df_arr = np.asarray([df[s] for s in vocab], dtype=np.int64)

    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    atom_sig_id = np.full(int(ap[-1]), -1, dtype=np.int32)
    for r in range(int(cache["n_molecules"])):
        s0, s1 = int(ap[r]), int(ap[r + 1])
        atom_sig_id[s0:s1] = np.asarray(
            [index.get(s, -1) for s in per_atom[r]], dtype=np.int32
        )

    atom_support_df = np.zeros_like(atom_sig_id, dtype=np.int64)
    known = atom_sig_id >= 0
    atom_support_df[known] = df_arr[atom_sig_id[known]]
    atom_support_loo = atom_support_df.copy()
    for r in np.asarray(fit_rows, dtype=np.int64).tolist():
        s0, s1 = int(ap[r]), int(ap[r + 1])
        atom_support_loo[s0:s1] = np.maximum(atom_support_df[s0:s1] - 1, 0)

    return {
        "vocab": vocab,
        "df": df_arr,
        "atom_sig_id": atom_sig_id,
        "atom_support_df": atom_support_df,
        "atom_support_loo": atom_support_loo,
        "fit_rows": np.asarray(fit_rows, dtype=np.int64),
        "meta": {
            "vocab_size": int(len(vocab)),
            "fit_rows": int(len(fit_rows)),
            "df_sha256": sha256_array(df_arr.astype(np.int64)),
            "atom_sig_id_sha256": sha256_array(atom_sig_id.astype(np.int32)),
            "atom_support_df_sha256": sha256_array(atom_support_df.astype(np.int64)),
            "atom_support_loo_sha256": sha256_array(atom_support_loo.astype(np.int64)),
            "signature_function_sha256": sha256_file(_DCB / "dca_common.py"),
            "count_unit": "document frequency over Fit molecules",
            "labels_used": False,
            "heldout_in_table": False,
        },
    }


# ---------------------------------------------------------------------- alpha
def alpha_from_support_values(s: np.ndarray, denominator: float = 4.0) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    return (s / (s + float(denominator))).astype(np.float32)


def alpha_for_rows(
    cache: Mapping[str, Any],
    support: Mapping[str, Any],
    rows: Iterable[int],
    *,
    leave_one_out_for: Iterable[int] | None = None,
) -> np.ndarray:
    """Bond alpha array indexed by the cache's global bond order.

    ``leave_one_out_for`` is the Fit row set.  A Fit row uses ``df - 1``
    (self-excluded); every other row uses ``df``.  This is the single estimand
    "number of Fit molecules other than the query molecule".
    """
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    n_bonds = int(bp[-1])
    out = np.zeros(n_bonds, dtype=np.float32)
    loo_set = set(int(r) for r in (leave_one_out_for if leave_one_out_for is not None else []))
    df_sup = np.asarray(support["atom_support_df"], dtype=np.int64)
    loo_sup = np.asarray(support["atom_support_loo"], dtype=np.int64)
    for r in np.asarray(list(rows), dtype=np.int64).tolist():
        b0, b1 = int(bp[r]), int(bp[r + 1])
        if b0 == b1:
            continue
        b = bi[b0:b1]
        sup = loo_sup if r in loo_set else df_sup
        s = np.minimum(sup[int(ap[r]) + b[:, 0]], sup[int(ap[r]) + b[:, 1]])
        out[b0:b1] = alpha_from_support_values(s)
    return out


def bond_support_for_rows(
    cache: Mapping[str, Any],
    support: Mapping[str, Any],
    rows: Iterable[int],
    *,
    leave_one_out_for: Iterable[int] | None = None,
) -> np.ndarray:
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    out = np.zeros(int(bp[-1]), dtype=np.int64)
    loo_set = set(int(r) for r in (leave_one_out_for if leave_one_out_for is not None else []))
    df_sup = np.asarray(support["atom_support_df"], dtype=np.int64)
    loo_sup = np.asarray(support["atom_support_loo"], dtype=np.int64)
    for r in np.asarray(list(rows), dtype=np.int64).tolist():
        b0, b1 = int(bp[r]), int(bp[r + 1])
        if b0 == b1:
            continue
        b = bi[b0:b1]
        sup = loo_sup if r in loo_set else df_sup
        out[b0:b1] = np.minimum(sup[int(ap[r]) + b[:, 0]], sup[int(ap[r]) + b[:, 1]])
    return out


def permute_alpha_within_molecules(
    cache: Mapping[str, Any],
    alpha: np.ndarray,
    rows: Iterable[int],
    salt: int,
) -> np.ndarray:
    """Permute each molecule's bond-alpha vector with SeedSequence([salt, row])."""
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    out = np.asarray(alpha, dtype=np.float32).copy()
    for r in np.asarray(list(rows), dtype=np.int64).tolist():
        b0, b1 = int(bp[r]), int(bp[r + 1])
        n = b1 - b0
        if n <= 1:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([int(salt), int(r)]))
        perm = rng.permutation(n)
        out[b0:b1] = out[b0:b1][perm]
    return out


# ---------------------------------------------------------------------- groups
def load_penalty(split: str) -> dict[int, bool]:
    path = CYCLE_TRAIN_LABEL if split == "train" else CYCLE_VALID_LABEL
    out: dict[int, bool] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] != split:
                continue
            out[int(row["subset_index"])] = float(row["label_excess"]) > 0.0
    return out


def tier_per_row(cache: Mapping[str, Any], support: Mapping[str, Any], rows: Iterable[int]) -> np.ndarray:
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    sup = np.asarray(support["atom_support_df"], dtype=np.int64)
    out = []
    for r in np.asarray(list(rows), dtype=np.int64).tolist():
        s = sup[ap[r]:ap[r + 1]]
        if np.any(s == 0):
            out.append("unseen")
        elif np.any(s < 5):
            out.append("low4")
        else:
            out.append("full")
    return np.asarray(out, dtype=object)


def group_payload(
    cache: Mapping[str, Any],
    support: Mapping[str, Any],
    rows: np.ndarray,
    split: str,
) -> dict:
    """All evaluation masks for a row list (Locked, Monitor or official valid)."""
    rows = np.asarray(rows, dtype=np.int64)
    tier = tier_per_row(cache, support, rows)
    penalty_map = load_penalty(split)
    penalty = np.asarray([bool(penalty_map[int(r)]) for r in rows], dtype=bool)
    y = np.asarray(cache["labels"], dtype=np.float64)[rows]
    masks = {
        "all": np.ones(len(rows), dtype=bool),
        "unseen_all": tier == "unseen",
        "low4_all": tier == "low4",
        "full_all": tier == "full",
        "penalty_all": penalty,
        "primary": (tier == "unseen") & (~penalty),
        "low4_nopen": (tier == "low4") & (~penalty),
        "full_nopen": (tier == "full") & (~penalty),
        "ordinary_seen": (tier != "unseen") & (~penalty),
        "ordinary_label": y > -3.0,
        "tail": y <= -3.0,
        "nopen": ~penalty,
    }
    return {
        "rows": rows,
        "y": y,
        "tier": tier,
        "penalty": penalty,
        "masks": masks,
        "counts": {k: int(v.sum()) for k, v in masks.items()},
    }


def check_frozen_group_counts(locked: Mapping[str, Any], monitor: Mapping[str, Any]) -> dict:
    locked_tiers = Counter(locked["tier"].tolist())
    monitor_tiers = Counter(monitor["tier"].tolist())
    assert dict(locked_tiers) == LOCKED_TIER_COUNTS, (dict(locked_tiers), LOCKED_TIER_COUNTS)
    assert dict(monitor_tiers) == MONITOR_TIER_COUNTS, (dict(monitor_tiers), MONITOR_TIER_COUNTS)
    assert int(locked["counts"]["primary"]) == PHASE1_PRIMARY_N
    return {
        "locked_tiers": dict(locked_tiers),
        "monitor_tiers": dict(monitor_tiers),
        "heldout_tiers": {
            k: int(locked_tiers[k] + monitor_tiers[k])
            for k in ("unseen", "low4", "full")
        },
        "natural_weights": NATURAL_WEIGHTS,
        "locked_counts": dict(locked["counts"]),
        "monitor_counts": dict(monitor["counts"]),
    }


# --------------------------------------------------------------------- metrics
def mae(pred: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> float:
    d = np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64))
    if mask is not None:
        d = d[np.asarray(mask, dtype=bool)]
    return float(d.mean()) if d.size else float("nan")


def natural_overall(pred: np.ndarray, group: Mapping[str, Any]) -> float:
    """Tier-weighted MAE with the frozen held-out tier proportions."""
    total = 0.0
    for k, w in NATURAL_WEIGHTS.items():
        total += w * mae(pred, group["y"], group["tier"] == k)
    return float(total)


def bootstrap_mean(x: np.ndarray, *, n_boot: int = 10000, seed: int = 20261028) -> dict:
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(int(n_boot), len(x)))
    vals = x[idx].mean(axis=1)
    return {
        "mean": float(x.mean()),
        "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
        "n": int(len(x)),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def cluster_bootstrap_mean(
    x: np.ndarray,
    clusters: np.ndarray,
    *,
    n_boot: int = 10000,
    seed: int = 20261028,
) -> dict:
    """Cluster bootstrap of a mean: clusters are the sampling units."""
    x = np.asarray(x, dtype=np.float64)
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    counts = np.bincount(inv).astype(np.float64)
    sums = np.bincount(inv, weights=x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(int(n_boot), len(uniq)))
    num = sums[idx].sum(axis=1)
    den = counts[idx].sum(axis=1)
    vals = num / den
    return {
        "mean": float(x.mean()),
        "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
        "n_molecules": int(len(x)),
        "n_clusters": int(len(uniq)),
        "n_singletons": int(np.sum(counts == 1)),
        "largest_cluster": int(counts.max()),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def key_clusters(
    support: Mapping[str, Any],
    per_atom: Sequence[Sequence[str]],
    rows: np.ndarray,
) -> np.ndarray:
    """Exact set-of-Fit-unseen-signatures cluster id for each row."""
    df = {s: int(d) for s, d in zip(support["vocab"], support["df"])}
    keys = []
    for r in np.asarray(rows, dtype=np.int64).tolist():
        key = tuple(sorted(s for s in per_atom[int(r)] if df.get(s, 0) == 0))
        keys.append(key)
    _, inv = np.unique(np.asarray(keys, dtype=object), return_inverse=True)
    return inv.astype(np.int32)


def w20_t100(curve: Sequence[Mapping[str, Any]], key: str = "monitor_mae") -> dict:
    vals = np.asarray([float(e[key]) for e in curve], dtype=np.float64)
    if vals.size == 0:
        return {"w20": None, "t100": None, "n_epochs": 0}
    if vals.size >= 20:
        med = np.asarray([np.median(vals[i:i + 20]) for i in range(vals.size - 19)])
        w20 = float(med.min())
    else:
        w20 = float(np.median(vals))
    t100 = float(np.median(vals[-min(100, vals.size):]))
    return {
        "w20": w20,
        "t100": t100,
        "n_epochs": int(vals.size),
        "best_epoch": int(np.argmin(vals) + 1),
        "best": float(vals.min()),
        "final": float(vals[-1]),
    }


def drop_top_k(delta: np.ndarray, k: int) -> dict:
    d = np.asarray(delta, dtype=np.float64)
    if k <= 0 or k >= len(d):
        return {"k": int(k), "mean": float(d.mean()), "n": int(len(d))}
    order = np.argsort(-np.abs(d))
    keep = np.ones(len(d), dtype=bool)
    keep[order[:k]] = False
    return {
        "k": int(k),
        "mean": float(d[keep].mean()),
        "n": int(keep.sum()),
        "removed_indices_in_group": order[:k].tolist(),
        "removed_abs_sum": float(np.abs(d[order[:k]]).sum()),
    }


def load_exact_longest() -> dict[int, float]:
    out: dict[int, float] = {}
    with open(EXACT_LONGEST, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == "train":
                out[int(row["idx"])] = float(row["longest_simple_cycle_length"])
    return out


def size_cycle_stats(cache: Mapping[str, Any], rows: np.ndarray, mask: np.ndarray) -> dict:
    rows = np.asarray(rows, dtype=np.int64)[np.asarray(mask, dtype=bool)]
    n_atoms = np.asarray(cache["n_atoms_graph"], dtype=np.float64)[rows]
    longest = load_exact_longest()
    lc = np.asarray([longest[int(r)] for r in rows], dtype=np.float64)
    return {
        "n": int(len(rows)),
        "n_atoms_mean": float(n_atoms.mean()) if len(rows) else None,
        "n_atoms_median": float(np.median(n_atoms)) if len(rows) else None,
        "n_atoms_min": int(n_atoms.min()) if len(rows) else None,
        "n_atoms_max": int(n_atoms.max()) if len(rows) else None,
        "longest_ge7_frac": float(np.mean(lc >= 7)) if len(rows) else None,
        "longest_mean": float(lc.mean()) if len(rows) else None,
    }


def assert_no_test_access(code_dir: Path = HERE) -> dict:
    """Static scan: this round's code must never name the official test cache."""
    bad = []
    own = inspect.getsource(assert_no_test_access)
    for path in sorted(code_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8").replace(own, "")
        for token in ('load_cache("test")', "load_cache('test')", "ad_test.pt"):
            if token in text:
                bad.append({"file": path.name, "token": token})
    return {"scanned": [p.name for p in sorted(code_dir.glob("*.py"))],
            "forbidden_hits": bad, "official_test_loaded": False}


def manual_sidecar_info() -> dict:
    """Provenance for the frozen label-derived evaluation masks."""
    return {
        "train_cycle_label_sha256": sha256_file(CYCLE_TRAIN_LABEL),
        "valid_cycle_label_sha256": sha256_file(CYCLE_VALID_LABEL),
        "exact_longest_sha256": sha256_file(EXACT_LONGEST),
        "penalty_definition": "label_excess > 0 (zema_audit penalty mask)",
        "used_for": "evaluation grouping only; never training, support, alpha or selection",
    }
