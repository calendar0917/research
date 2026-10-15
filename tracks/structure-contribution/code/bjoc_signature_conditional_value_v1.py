"""cscl-bjoc-signature-conditional-value-v1 — CPU, label-free conditional-value analysis.

Frozen-model, label-free primitives for the preregistered round:

* within-molecule DEV / |IG| percentiles and the residual ``r = q - d``
* molecule-capped signature/context residual tables (Arm B / Arm C)
* stratified conditional pseudo-control (DEV-quartile residual permutation)
* fixed Arm A/B/C scores, stable top-k selection, response/oracle-overlap metrics
* the frozen four-label decision rule

No model inference, no training, no labels, no official-test / Locked access.
Every per-Junction quantity is reused from the frozen
``bjoc_signature_transfer_v1`` raw artifacts; the runner re-verifies the reuse
by exact reproduction of four published metrics.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_signature_transfer_v1 as bst  # noqa: E402

PROTOCOL_ID = "cscl-bjoc-signature-conditional-value-v1"
NAMESPACE = "bjoc_signature_conditional_value_v1"
TRACK = REPO_ROOT / "tracks/structure-contribution"
OUT = TRACK / "results" / NAMESPACE
CONFIG_PATH = TRACK / "configs" / f"{NAMESPACE}.yaml"
PRIOR = TRACK / "results" / "bjoc_signature_transfer_v1"

# ---- frozen protocol constants -------------------------------------------
SIGNATURE_FLOOR_MOLECULES = 20
SIGNATURE_SHRINK = 20.0
CONTEXT_FLOOR_MOLECULES = 10
CONTEXT_SHRINK = 20.0
TOP_K_PRIMARY = 4
TOP_K_SENSITIVITY = (1, 2, 4, 8)
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 0
PSEUDO_REPLICATES = 16
PSEUDO_DEV_BINS = 4
PSEUDO_RNG_BASE = 810_000_000
RECOVERY_FRACTION_MIN = 0.05
CONCENTRATION_TOP1_SHARE_MAX = 0.5
ARMS = ("A", "B", "C")
SIG_LABELS = (
    "SIGNATURE_HAS_CONDITIONAL_VALUE",
    "CONTEXT_ADDS_VALUE_BUT_SIGNATURE_WEAK",
    "INCONCLUSIVE",
    "NO_CONFIRMED_CONDITIONAL_VALUE",
)

DISCIPLINE = {
    "labels_read": False,
    "official_valid_structures_only": True,
    "official_valid_exposed_dev_set": True,
    "official_test_loaded": False,
    "locked_split_read": False,
    "new_training": False,
    "model_inference": False,
    "gpu_used": False,
}


# ---------------------------------------------------------------------------
# io (thin wrappers so callers do not need bst directly)
# ---------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    return bst.read_json(path)


def write_json(path: Path, payload: Any) -> None:
    bst.write_json(path, payload)


def file_sha256(path: Path) -> str:
    return bst.file_sha256(path)


# ---------------------------------------------------------------------------
# signatures / context
# ---------------------------------------------------------------------------


def signature_key(sig_json: Sequence[Any]) -> tuple:
    return bst.sig_key(sig_json)


def signature_join(sig_json: Sequence[Any]) -> str:
    return bst.sig_join(bst.sig_key(sig_json))


def context_key(sig_join: str, degree: int, endpoint_ring_count: int) -> str:
    return f"{sig_join}|{int(degree)}|{int(endpoint_ring_count)}"


def shrunk_residual(value: float, n_molecules: int, shrink: float, floor: int) -> float:
    """Pre-registered shrinkage: 0 below the support floor, else n/(n+shrink)*value."""
    n = int(n_molecules)
    if n < int(floor):
        return 0.0
    return float(n) / (float(n) + float(shrink)) * float(value)


def percentile(values: Sequence[float]) -> np.ndarray:
    """Within-molecule average-rank percentile (identical to the previous round)."""
    return bst.average_rank_percentile(values)


# ---------------------------------------------------------------------------
# junction sets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JunctionSet:
    """One (seed, molecule) view; r is the residual the fit is keyed on."""

    seed: int
    row: int
    keys: tuple[str, ...]
    degree: np.ndarray
    endpoint_ring_count: np.ndarray
    d: np.ndarray
    q: np.ndarray
    r: np.ndarray

    @property
    def n_j(self) -> int:
        return int(self.d.shape[0])


def junction_set_from_arrays(
    seed: int,
    row: int,
    keys: Sequence[str],
    degree: np.ndarray,
    endpoint_ring_count: np.ndarray,
    dev: np.ndarray,
    ig_abs: np.ndarray,
    residual_override: np.ndarray | None = None,
) -> JunctionSet:
    d = percentile(dev)
    q = bst.average_rank_percentile(ig_abs)
    r = q - d if residual_override is None else np.asarray(residual_override, dtype=np.float64)
    return JunctionSet(
        seed=int(seed),
        row=int(row),
        keys=tuple(str(k) for k in keys),
        degree=np.asarray(degree, dtype=np.int64),
        endpoint_ring_count=np.asarray(endpoint_ring_count, dtype=np.int64),
        d=np.asarray(d, dtype=np.float64),
        q=np.asarray(q, dtype=np.float64),
        r=np.asarray(r, dtype=np.float64),
    )


def junction_sets_from_diag(
    mols_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
) -> list[JunctionSet]:
    """Discovery junction sets (stored signatures); audit-friendly, no inference."""
    out: list[JunctionSet] = []
    for seed in sorted(int(s) for s in mols_by_seed):
        for mol in mols_by_seed[seed]:
            keys = [signature_join(s) for s in mol["signatures"]]
            out.append(
                junction_set_from_arrays(
                    seed,
                    int(mol["row"]),
                    keys,
                    mol["degree"],
                    mol["endpoint_ring_count"],
                    np.asarray(mol["dev"], dtype=np.float64),
                    np.asarray(mol["ig_abs"], dtype=np.float64),
                )
            )
    return out


# ---------------------------------------------------------------------------
# residual tables (discovery-only by construction)
# ---------------------------------------------------------------------------


def _molecule_capped_mean(
    values_by_ms: Mapping[tuple[int, int], Sequence[float]],
) -> tuple[float, int, int]:
    """Mean over molecules of the per-molecule seed-mean; returns (R, n_molecules, n_seed_units)."""
    per_row: dict[int, list[float]] = defaultdict(list)
    n_seed_units = 0
    for (seed, row), vals in values_by_ms.items():
        vals = list(vals)
        if not vals:
            continue
        per_row[int(row)].append(float(np.mean(vals)))
        n_seed_units += 1
    if not per_row:
        return float("nan"), 0, 0
    mol_means = [float(np.mean(v)) for v in per_row.values()]
    return float(np.mean(mol_means)), int(len(mol_means)), int(n_seed_units)


def build_residual_tables(
    junction_sets: Sequence[JunctionSet],
    signature_floor: int = SIGNATURE_FLOOR_MOLECULES,
    signature_shrink: float = SIGNATURE_SHRINK,
    context_floor: int = CONTEXT_FLOOR_MOLECULES,
    context_shrink: float = CONTEXT_SHRINK,
) -> dict[str, Any]:
    """Molecule-capped Arm B / Arm C residual tables from discovery junctions only."""
    sig_vals: dict[str, dict[tuple[int, int], list[float]]] = defaultdict(lambda: defaultdict(list))
    for js in junction_sets:
        for j in range(js.n_j):
            sig_vals[js.keys[j]][(js.seed, js.row)].append(float(js.r[j]))

    sig_table: dict[str, dict[str, Any]] = {}
    for key, by_ms in sig_vals.items():
        R, n_mol, n_seed_units = _molecule_capped_mean(by_ms)
        sig_table[key] = {
            "n_molecules": int(n_mol),
            "n_seed_units": int(n_seed_units),
            "R": float(R),
            "R_shrunk": shrunk_residual(R, n_mol, signature_shrink, signature_floor),
        }

    ctx_vals: dict[str, dict[tuple[int, int], list[float]]] = defaultdict(lambda: defaultdict(list))
    for js in junction_sets:
        for j in range(js.n_j):
            key = js.keys[j]
            bonus = sig_table.get(key, {}).get("R_shrunk", 0.0)
            ck = context_key(key, int(js.degree[j]), int(js.endpoint_ring_count[j]))
            ctx_vals[ck][(js.seed, js.row)].append(float(js.r[j]) - float(bonus))

    ctx_table: dict[str, dict[str, Any]] = {}
    for key, by_ms in ctx_vals.items():
        R_ctx, n_mol, n_seed_units = _molecule_capped_mean(by_ms)
        sig_key, degree, erc = key.rsplit("|", 2)
        ctx_table[key] = {
            "signature": sig_key,
            "center_degree": int(degree),
            "endpoint_ring_count": int(erc),
            "n_molecules": int(n_mol),
            "n_seed_units": int(n_seed_units),
            "R_raw": float(R_ctx),
            "R_shrunk": shrunk_residual(R_ctx, n_mol, context_shrink, context_floor),
        }

    return {
        "protocol": PROTOCOL_ID,
        "signature_floor_molecules": int(signature_floor),
        "signature_shrink": float(signature_shrink),
        "context_floor_molecules": int(context_floor),
        "context_shrink": float(context_shrink),
        "n_signature_units": int(len(sig_table)),
        "n_context_units": int(len(ctx_table)),
        "sig": sig_table,
        "ctx": ctx_table,
    }


def sig_bonus_values(tables: Mapping[str, Any], keys: Sequence[str]) -> np.ndarray:
    table = tables["sig"]
    return np.asarray(
        [float(table[k]["R_shrunk"]) if k in table else 0.0 for k in keys],
        dtype=np.float64,
    )


def ctx_bonus_values(tables: Mapping[str, Any], keys: Sequence[str], degree: np.ndarray, erc: np.ndarray) -> np.ndarray:
    table = tables["ctx"]
    out = np.zeros(len(keys), dtype=np.float64)
    for j, k in enumerate(keys):
        ck = context_key(k, int(degree[j]), int(erc[j]))
        if ck in table:
            out[j] = float(table[ck]["R_shrunk"])
    return out


def bonus_arrays(
    tables: Mapping[str, Any], keys: Sequence[str], degree: np.ndarray, erc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return sig_bonus_values(tables, keys), ctx_bonus_values(tables, keys, degree, erc)


def arm_score(
    d: np.ndarray,
    sig_bonus: np.ndarray,
    ctx_bonus: np.ndarray,
    arm: str,
) -> np.ndarray:
    """Frozen Arm A/B/C score; A is the raw DEV percentile."""
    if arm == "A":
        return np.asarray(d, dtype=np.float64).copy()
    if arm == "B":
        return np.asarray(d, dtype=np.float64) + np.asarray(sig_bonus, dtype=np.float64)
    if arm == "C":
        return (
            np.asarray(d, dtype=np.float64)
            + np.asarray(sig_bonus, dtype=np.float64)
            + np.asarray(ctx_bonus, dtype=np.float64)
        )
    raise ValueError(f"unknown arm {arm!r}")


# ---------------------------------------------------------------------------
# conditional pseudo-control
# ---------------------------------------------------------------------------


def pseudo_junction_sets(
    junction_sets: Sequence[JunctionSet],
    replicate: int,
    dev_bins: int = PSEUDO_DEV_BINS,
) -> list[JunctionSet]:
    """Permute r inside each (seed, molecule, DEV-quartile) stratum."""
    out: list[JunctionSet] = []
    for js in junction_sets:
        rng = np.random.default_rng([PSEUDO_RNG_BASE, int(replicate), int(js.seed), int(js.row)])
        bins = np.minimum((js.d * float(dev_bins)).astype(np.int64), int(dev_bins) - 1)
        r2 = js.r.copy()
        for b in range(int(dev_bins)):
            idx = np.where(bins == b)[0]
            if idx.size > 1:
                r2[idx] = js.r[idx][rng.permutation(idx.size)]
        out.append(replace(js, r=r2))
    return out


def pseudo_tables(
    junction_sets: Sequence[JunctionSet],
    replicates: int = PSEUDO_REPLICATES,
    dev_bins: int = PSEUDO_DEV_BINS,
    **build_kwargs: Any,
) -> list[dict[str, Any]]:
    return [
        build_residual_tables(pseudo_junction_sets(junction_sets, p, dev_bins), **build_kwargs)
        for p in range(int(replicates))
    ]


def compact_tables(tables: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fixed-order arrays for the pseudo tables (JSON/NPZ friendly)."""
    sig_keys = sorted({k for t in tables for k in t["sig"]})
    ctx_keys = sorted({k for t in tables for k in t["ctx"]})
    sig = np.asarray([[float(t["sig"].get(k, {}).get("R_shrunk", 0.0)) for k in sig_keys] for t in tables])
    ctx = np.asarray([[float(t["ctx"].get(k, {}).get("R_shrunk", 0.0)) for k in ctx_keys] for t in tables])
    return {"sig_keys": sig_keys, "sig_shrunk": sig, "ctx_keys": ctx_keys, "ctx_shrunk": ctx}


def tables_from_compact(compact: Mapping[str, Any], i: int) -> dict[str, Any]:
    """Rebuild one table view (lookup values only) from the compact arrays."""
    sig = {k: {"R_shrunk": float(compact["sig_shrunk"][i, j])} for j, k in enumerate(compact["sig_keys"])}
    ctx = {k: {"R_shrunk": float(compact["ctx_shrunk"][i, j])} for j, k in enumerate(compact["ctx_keys"])}
    return {"sig": sig, "ctx": ctx}


# ---------------------------------------------------------------------------
# selection / response metrics
# ---------------------------------------------------------------------------


def top_k_indices(scores: Sequence[float], k: int) -> np.ndarray:
    return bst.top_k_indices(np.asarray(scores, dtype=np.float64), k)


def topk_mean_abs(delta: Sequence[float], score: Sequence[float], k: int) -> float:
    idx = top_k_indices(score, k)
    if idx.size == 0:
        return float("nan")
    return float(np.abs(np.asarray(delta, dtype=np.float64)[idx]).mean())


def topk_oracle_overlap(score: Sequence[float], oracle: Sequence[float], k: int) -> float:
    a = set(int(x) for x in top_k_indices(score, k))
    b = set(int(x) for x in top_k_indices(oracle, k))
    if not b:
        return float("nan")
    return float(len(a & b) / len(b))


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def contrast(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    boot = bst.bootstrap_mean(d, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
    sign = bst.sign_test_one_sided(d)
    return {
        "mean": float(boot["mean"]),
        "ci95_lower": float(boot["ci95_lower"]),
        "ci95_upper": float(boot["ci95_upper"]),
        "n_molecules": int(boot["n"]),
        "fraction_positive": float(np.mean(d > 0)) if d.size else float("nan"),
        "sign_test": sign,
    }


def merged_seed_arrays(per_seed: Sequence[np.ndarray]) -> np.ndarray:
    """Average the two seeds per molecule (cross-seed correlation preserved)."""
    return np.mean(np.stack([np.asarray(x, dtype=np.float64) for x in per_seed]), axis=0)


def support_histogram(supports: Iterable[int]) -> dict[str, int]:
    buckets = {"<10": 0, "10-19": 0, "20-49": 0, "50-99": 0, "100-199": 0, ">=200": 0}
    for n in supports:
        n = int(n)
        if n < 10:
            buckets["<10"] += 1
        elif n < 20:
            buckets["10-19"] += 1
        elif n < 50:
            buckets["20-49"] += 1
        elif n < 100:
            buckets["50-99"] += 1
        elif n < 200:
            buckets["100-199"] += 1
        else:
            buckets[">=200"] += 1
    return buckets


# ---------------------------------------------------------------------------
# decision rule
# ---------------------------------------------------------------------------


def decide_label(gates: Mapping[str, bool]) -> str:
    """Frozen four-label decision tree (see configs/...v1.yaml)."""
    a_keys = [f"A{i}" for i in range(1, 9)]
    b_keys = [f"B{i}" for i in range(1, 6)]
    robust = ["A4", "A5", "A6", "A7"]
    a_stats = all(bool(gates.get(k, False)) for k in ("A1", "A2", "A3"))
    a_all = all(bool(gates.get(k, False)) for k in a_keys)
    b_all = all(bool(gates.get(k, False)) for k in b_keys)
    if a_all:
        return "SIGNATURE_HAS_CONDITIONAL_VALUE"
    if b_all and not a_stats:
        return "CONTEXT_ADDS_VALUE_BUT_SIGNATURE_WEAK"
    # positive but statistically uncertain / seed-inconsistent
    if (bool(gates.get("A1", False)) and not bool(gates.get("A2", False))) or (
        bool(gates.get("B1", False)) and not bool(gates.get("B2", False))
    ):
        return "INCONCLUSIVE"
    # statistically reliable B-A that is robust but practically marginal
    if a_stats and all(bool(gates.get(k, False)) for k in robust) and not bool(gates.get("A8", False)):
        return "INCONCLUSIVE"
    # statistically reliable B-A that fails a robustness gate (concentration,
    # pseudo-control, anti-memorisation or cross-scaffold direction) cannot be
    # attributed to stable shared type information
    return "NO_CONFIRMED_CONDITIONAL_VALUE"
