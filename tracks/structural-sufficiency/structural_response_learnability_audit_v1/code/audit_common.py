"""Shared data plumbing for the structural-response learnability audit.

Read-only inputs
----------------
* ``task_response_composition_s0_v1/data/trc_cache_lipo.npz``: real matched-pair
  directory on the official Train rows (pair rows, parses, labels).
* ``acra_relation_generalization_pilot_v1/data/acra_cache_lipo_v2.npz``: the
  ACRA local identities (radius-1 S identity, radius-2 C identity, directed
  events) and the Train-sub / Dev scaffold split.
* ``acra_relation_generalization_pilot_v1/data/split_train_sub_dev.json``:
  frozen Train-sub / Dev assignment (scaffold-disjoint).

The audit never reads official Valid or Test labels.  ``labels`` is masked to
NaN outside the official Train rows immediately after loading, and every
label-dependent function asserts that its rows are inside the mask.
"""

from __future__ import annotations

import hashlib
import json
import resource
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
TRC_DIR = REPO_ROOT / "tracks" / "structural-sufficiency" / "task_response_composition_s0_v1"
ACRA_DIR = REPO_ROOT / "tracks" / "structural-sufficiency" / "acra_relation_generalization_pilot_v1"
ANALYSIS_DIR = ROUND_DIR / "analysis"

TRC_CACHE = TRC_DIR / "data" / "trc_cache_lipo.npz"
ACRA_CACHE = ACRA_DIR / "data" / "acra_cache_lipo_v2.npz"
SPLIT_JSON = ACRA_DIR / "data" / "split_train_sub_dev.json"

# sha256 of the two prepared caches as recorded by the parent rounds.
TRC_CACHE_SHA256 = "c53fee5d16869b7b7ff721d404a84289d0a315d392440ebfb436cbd307e3abc3"
ACRA_CACHE_SHA256 = "1f6d7a3f870513a83b50f6aa6edbe0143f37c2a883308627f47145e2078fdc3b"

# frozen audit settings (no search is performed)
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 0
SHRINK_K = 1.0
RIDGE_LAMBDA = 10.0
FINGERPRINT_BITS = 256
FINGERPRINT_ITERS = 2
SCOPE_NAMES = ("train_sub", "cross", "dev_dev")


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if np.isnan(value):
            return None
        if np.isinf(value):
            return None
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path | str, payload: Mapping[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(payload), indent=1, sort_keys=False, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit() -> str:
    import subprocess

    try:
        out = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                      text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return "unknown"


def peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def relative(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------


def components(n_nodes: int, aa: np.ndarray, bb: np.ndarray) -> np.ndarray:
    """Union-find component labels for an undirected edge list."""
    parent = list(range(int(n_nodes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for x, y in zip(np.asarray(aa, np.int64).tolist(), np.asarray(bb, np.int64).tolist()):
        rx, ry = find(int(x)), find(int(y))
        if rx != ry:
            parent[rx] = ry
    return np.asarray([find(i) for i in range(int(n_nodes))], np.int32)


def component_sizes(comp: np.ndarray, nodes: np.ndarray) -> dict[int, int]:
    """Sizes (over the given node subset) of components touched by ``nodes``."""
    counts: dict[int, int] = defaultdict(int)
    for x in np.asarray(nodes, np.int64).tolist():
        counts[int(comp[int(x)])] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# input loading
# ---------------------------------------------------------------------------


@dataclass
class Inputs:
    trc: Mapping[str, np.ndarray]
    acra: Mapping[str, np.ndarray]
    split: Mapping[str, Any]
    n_molecules: int
    official_train: np.ndarray
    train_sub: np.ndarray
    dev: np.ndarray
    labels: np.ndarray          # NaN outside official Train
    scaffold_of_mol: np.ndarray  # object array of scaffold strings (Train rows)
    hashes: dict[str, Any]
    checks: dict[str, Any]


def load_inputs() -> Inputs:
    trc_sha = sha256_file(TRC_CACHE)
    acra_sha = sha256_file(ACRA_CACHE)
    split_sha = sha256_file(SPLIT_JSON)
    if trc_sha != TRC_CACHE_SHA256:
        raise ValueError(f"TRC cache sha256 mismatch: {trc_sha} != {TRC_CACHE_SHA256}")
    if acra_sha != ACRA_CACHE_SHA256:
        raise ValueError(f"ACRA cache sha256 mismatch: {acra_sha} != {ACRA_CACHE_SHA256}")

    with np.load(TRC_CACHE, allow_pickle=False) as z:
        trc = {k: np.asarray(z[k]) for k in z.files}
    with np.load(ACRA_CACHE, allow_pickle=False) as z:
        acra = {k: np.asarray(z[k]) for k in z.files}
    split = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))

    n_mol = int(trc["n_molecules"])
    if int(acra["n_molecules"]) != n_mol:
        raise ValueError("cache molecule counts differ")
    if not np.array_equal(np.asarray(trc["row_ids"], np.int64),
                          np.asarray(acra["row_ids"], np.int64)):
        raise ValueError("cache row_ids differ")

    official_train = np.zeros(n_mol, bool)
    official_train[np.asarray(trc["train_idx"], np.int64)] = True
    train_sub = np.zeros(n_mol, bool)
    train_sub[np.asarray(split["train_sub_rows"], np.int64)] = True
    dev = np.zeros(n_mol, bool)
    dev[np.asarray(split["dev_rows"], np.int64)] = True

    if int(official_train.sum()) != 3360:
        raise ValueError("unexpected official Train size")
    if int(train_sub.sum()) != 2856 or int(dev.sum()) != 504:
        raise ValueError("unexpected Train-sub / Dev size")
    if np.any(train_sub & dev):
        raise ValueError("Train-sub and Dev overlap")
    if np.any((train_sub | dev) & ~official_train):
        raise ValueError("split contains non-Train rows")

    labels = np.asarray(trc["labels"], np.float64).copy()
    labels[~official_train] = np.nan

    scaffold_of_mol = np.empty(n_mol, dtype=object)
    for key, value in dict(split["scaffold_by_row"]).items():
        scaffold_of_mol[int(key)] = str(value)
    missing = [int(r) for r in np.nonzero(official_train)[0] if scaffold_of_mol[int(r)] is None]
    if missing:
        raise ValueError(f"missing scaffold for {len(missing)} Train rows")

    hashes = {
        "trc_cache": {"path": relative(TRC_CACHE), "sha256": trc_sha},
        "acra_relation_cache": {"path": relative(ACRA_CACHE), "sha256": acra_sha},
        "split_json": {
            "path": relative(SPLIT_JSON),
            "sha256": split_sha,
            "internal_split_sha256": str(split.get("split_sha256", "")),
        },
        "raw_cache_sha256": str(trc.get("cache_sha256", "")),
    }
    checks = {
        "row_ids_equal": True,
        "official_train_n": int(official_train.sum()),
        "train_sub_n": int(train_sub.sum()),
        "dev_n": int(dev.sum()),
        "train_sub_dev_overlap": int((train_sub & dev).sum()),
        "split_subset_of_official_train": bool(not np.any((train_sub | dev) & ~official_train)),
        "labels_masked_outside_official_train": int(np.isnan(labels).sum()),
        "labels_finite_on_official_train": bool(np.isfinite(labels[official_train]).all()),
    }
    return Inputs(trc=trc, acra=acra, split=split, n_molecules=n_mol,
                  official_train=official_train, train_sub=train_sub, dev=dev,
                  labels=labels, scaffold_of_mol=scaffold_of_mol,
                  hashes=hashes, checks=checks)


# ---------------------------------------------------------------------------
# pair table and local-identity mapping
# ---------------------------------------------------------------------------


@dataclass
class PairTable:
    idx: np.ndarray             # index into the original TRC pair arrays
    a: np.ndarray
    b: np.ndarray
    C: np.ndarray
    Sa: np.ndarray
    Sb: np.ndarray
    bond_row: np.ndarray
    T: np.ndarray
    coarse: np.ndarray          # (n, 3): radius-1 word identities + cut bond row
    r1w_a: np.ndarray
    r2w_a: np.ndarray
    r1w_b: np.ndarray
    r2w_b: np.ndarray
    r1e: np.ndarray
    r2e: np.ndarray
    scope: np.ndarray           # 0 train_sub, 1 cross, 2 dev_dev
    comp_train: np.ndarray      # component id of pair.a in the Train-sub pair graph
    comp_all: np.ndarray        # component id of pair.a in the full Train pair graph
    comp_train_nodes: np.ndarray  # node-level component ids, Train-sub pair graph
    comp_all_nodes: np.ndarray    # node-level component ids, full Train pair graph
    d: np.ndarray               # y(b) - y(a), original units
    checks: dict[str, Any]


def _parse_lookup(trc: Mapping[str, np.ndarray]) -> dict[tuple[int, int, int, int], list[int]]:
    parse_mol = np.asarray(trc["parse_mol"], np.int64)
    parse_S = np.asarray(trc["parse_S"], np.int64)
    parse_C = np.asarray(trc["parse_C"], np.int64)
    parse_bondrow = np.asarray(trc["parse_cut_bond_row"], np.int64)
    parse_kind = np.asarray(trc["parse_kind"], np.uint8)
    out: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for p in np.nonzero(parse_kind == 0)[0].tolist():
        out[(int(parse_mol[p]), int(parse_S[p]), int(parse_C[p]), int(parse_bondrow[p]))].append(int(p))
    return out


def _endpoint_keys(trc: Mapping[str, np.ndarray], acra: Mapping[str, np.ndarray],
                   pmap: Mapping[tuple[int, int, int, int], Sequence[int]],
                   mol: int, word: int, context: int, bond_row: int) -> tuple[int, int, int, int]:
    """Map one pair endpoint to (r1_word, r2_word, r1_env, r2_env).

    The forward ACRA event points from the substituent (word) attachment to the
    environment attachment.  Its radius-1 S identity is the word-side local
    identity and its radius-2 C identity is the environment-side local identity.
    The reverse event supplies the environment radius-1 identity and the word
    radius-2 identity.
    """
    parses = pmap.get((int(mol), int(word), int(context), int(bond_row)), [])
    if not parses:
        raise KeyError(f"no parse for endpoint mol={mol} word={word} ctx={context} bond={bond_row}")
    ev_ptr = np.asarray(acra["ev_ptr"], np.int64)
    ev_dir = np.asarray(acra["ev_dir"], np.uint8)
    ev_bond = np.asarray(acra["ev_bond"], np.int32)
    ev_ks = np.asarray(acra["ev_key_s"], np.int32)
    ev_kc = np.asarray(acra["ev_key_c"], np.int32)
    parse_bondidx = np.asarray(trc["parse_bond_index"], np.int64)
    parse_u = np.asarray(trc["parse_cut_u"], np.int64)
    parse_v = np.asarray(trc["parse_cut_v"], np.int64)
    parse_portS = np.asarray(trc["parse_port_S"], np.int64)
    found: set[tuple[int, int, int, int]] = set()
    for p in parses:
        bi = int(parse_bondidx[p])
        src = int(parse_portS[p])
        u, v = int(parse_u[p]), int(parse_v[p])
        e0 = int(ev_ptr[int(mol)]) + 2 * bi
        e1 = e0 + 1
        if int(ev_bond[e0]) != bi or int(ev_bond[e1]) != bi:
            raise ValueError("ACRA event index does not match parse bond index")
        fwd = e0 if ((int(ev_dir[e0]) == 0 and u == src) or (int(ev_dir[e0]) == 1 and v == src)) else e1
        rev = e1 if fwd == e0 else e0
        found.add((int(ev_ks[fwd]), int(ev_kc[rev]), int(ev_ks[rev]), int(ev_kc[fwd])))
    if len(found) != 1:
        raise ValueError(f"ambiguous endpoint keys for mol={mol} word={word} ctx={context}")
    return found.pop()


def build_pair_table(inp: Inputs) -> PairTable:
    trc, acra = inp.trc, inp.acra
    a_all = np.asarray(trc["pair_mol_a"], np.int64)
    b_all = np.asarray(trc["pair_mol_b"], np.int64)
    keep = np.nonzero(inp.official_train[a_all] & inp.official_train[b_all])[0]
    if keep.size != 4646:
        raise ValueError(f"expected 4646 official-Train pairs, got {keep.size}")

    a = a_all[keep]
    b = b_all[keep]
    C = np.asarray(trc["pair_C"], np.int64)[keep]
    Sa = np.asarray(trc["pair_Sa"], np.int64)[keep]
    Sb = np.asarray(trc["pair_Sb"], np.int64)[keep]
    bond_row = np.asarray(trc["pair_cut_bond_row"], np.int64)[keep]
    T = np.asarray(trc["pair_transform"], np.int64)[keep]

    # transform arrays must agree with the per-pair keys
    twa = np.asarray(trc["transform_word_a"], np.int64)[T]
    twb = np.asarray(trc["transform_word_b"], np.int64)[T]
    tbr = np.asarray(trc["transform_bond_row"], np.int64)[T]
    transform_consistent = bool(np.array_equal(twa, Sa) and np.array_equal(twb, Sb)
                                and np.array_equal(tbr, bond_row))

    pmap = _parse_lookup(trc)
    n = keep.size
    r1w_a = np.zeros(n, np.int64)
    r2w_a = np.zeros(n, np.int64)
    r1w_b = np.zeros(n, np.int64)
    r2w_b = np.zeros(n, np.int64)
    r1e = np.zeros(n, np.int64)
    r2e = np.zeros(n, np.int64)
    word_map: dict[int, tuple[int, int]] = {}
    ctx_map: dict[int, tuple[int, int]] = {}
    multi_parse = 0
    env_mismatch = 0
    for j, i in enumerate(keep.tolist()):
        ps_a = pmap.get((int(a[j]), int(Sa[j]), int(C[j]), int(bond_row[j])), [])
        ps_b = pmap.get((int(b[j]), int(Sb[j]), int(C[j]), int(bond_row[j])), [])
        multi_parse += int(len(ps_a) > 1 or len(ps_b) > 1)
        xa = _endpoint_keys(trc, acra, pmap, int(a[j]), int(Sa[j]), int(C[j]), int(bond_row[j]))
        xb = _endpoint_keys(trc, acra, pmap, int(b[j]), int(Sb[j]), int(C[j]), int(bond_row[j]))
        if (xa[2], xa[3]) != (xb[2], xb[3]):
            env_mismatch += 1
        r1w_a[j], r2w_a[j] = xa[0], xa[1]
        r1w_b[j], r2w_b[j] = xb[0], xb[1]
        r1e[j], r2e[j] = xa[2], xa[3]
        for word, pair_keys in ((int(Sa[j]), (xa[0], xa[1])),
                                (int(Sb[j]), (xb[0], xb[1]))):
            old = word_map.setdefault(word, pair_keys)
            if old != pair_keys:
                raise ValueError(f"word {word} maps to inconsistent local keys")
        old = ctx_map.setdefault(int(C[j]), (xa[2], xa[3]))
        if old != (xa[2], xa[3]):
            raise ValueError(f"context {C[j]} maps to inconsistent local keys")
    if env_mismatch:
        raise ValueError(f"environment keys differ across {env_mismatch} pair endpoints")

    scope = np.zeros(n, np.int8)
    ta, tb = inp.train_sub[a], inp.train_sub[b]
    da, db = inp.dev[a], inp.dev[b]
    scope[ta & tb] = 0
    scope[(ta & db) | (da & tb)] = 1
    scope[da & db] = 2
    if np.any(~((ta | da) & (tb | db))):
        raise ValueError("pair endpoint outside Train-sub / Dev")

    # components: training graph (Train-sub--Train-sub) and full Train graph
    train_edges = np.nonzero(scope == 0)[0]
    comp_train_nodes = components(inp.n_molecules, a[train_edges], b[train_edges])
    comp_all_nodes = components(inp.n_molecules, a, b)
    comp_train = comp_train_nodes[a]
    comp_all = comp_all_nodes[a]

    d = inp.labels[b] - inp.labels[a]
    if not np.isfinite(d).all():
        raise ValueError("non-finite label difference in the audit pair table")

    # duplicate pair keys (the directory builder already deduplicates)
    keys = np.stack([np.minimum(a, b), np.maximum(a, b), C, Sa, Sb], 1)
    n_duplicate_keys = int(keys.shape[0] - np.unique(keys, axis=0).shape[0])

    checks = {
        "n_pairs_analyzed": int(n),
        "n_pairs_source": int(a_all.size),
        "n_pairs_excluded_non_train": int(a_all.size - n),
        "scope_counts": {SCOPE_NAMES[s]: int((scope == s).sum()) for s in range(3)},
        "transform_consistent": transform_consistent,
        "n_multi_parse_pairs": int(multi_parse),
        "max_parses_per_endpoint": int(max((len(v) for v in pmap.values()), default=0)),
        "environment_key_mismatch": int(env_mismatch),
        "n_word_keys": int(len(word_map)),
        "n_context_keys": int(len(ctx_map)),
        "n_duplicate_pair_keys": n_duplicate_keys,
        "n_components_train": int(len(set(comp_train_nodes[np.unique(np.concatenate([a[train_edges], b[train_edges]]))].tolist()))),
        "n_components_all": int(len(set(comp_all_nodes[np.unique(np.concatenate([a, b]))].tolist()))),
    }
    return PairTable(idx=keep, a=a, b=b, C=C, Sa=Sa, Sb=Sb, bond_row=bond_row,
                     T=T, coarse=np.stack([r1w_a, r1w_b, bond_row], 1),
                     r1w_a=r1w_a, r2w_a=r2w_a, r1w_b=r1w_b, r2w_b=r2w_b,
                     r1e=r1e, r2e=r2e, scope=scope,
                     comp_train=comp_train, comp_all=comp_all,
                     comp_train_nodes=comp_train_nodes, comp_all_nodes=comp_all_nodes,
                     d=d, checks=checks)


# ---------------------------------------------------------------------------
# grouping helpers
# ---------------------------------------------------------------------------


def hashable_key(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return tuple(int(x) for x in value.tolist())
    if isinstance(value, np.generic):
        return int(value)
    if isinstance(value, (list, tuple)):
        return tuple(hashable_key(x) for x in value)
    return value


def group_positions(keys: Sequence[Any], positions: Sequence[int]) -> dict[Any, np.ndarray]:
    groups: dict[Any, list[int]] = defaultdict(list)
    for pos in positions:
        groups[hashable_key(keys[pos])].append(int(pos))
    return {k: np.asarray(v, np.int64) for k, v in groups.items()}


def union_molecules(a: np.ndarray, b: np.ndarray, positions: np.ndarray) -> set[int]:
    return set(int(x) for x in a[positions].tolist()) | set(int(x) for x in b[positions].tolist())


def union_components(comp: np.ndarray, a: np.ndarray, b: np.ndarray,
                     positions: np.ndarray) -> set[int]:
    return set(int(x) for x in comp[a[positions]].tolist()) | set(int(x) for x in comp[b[positions]].tolist())


# ---------------------------------------------------------------------------
# metrics and bootstrap
# ---------------------------------------------------------------------------


def delta_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, np.float64)
    pred = np.asarray(pred, np.float64)
    n = int(y.size)
    if n == 0:
        return {"n": 0}
    err = pred - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    bias = float(np.mean(err))
    if float(np.std(pred)) > 0 and float(np.std(y)) > 0:
        corr = float(np.corrcoef(y, pred)[0, 1])
        slope = float(np.cov(pred, y, ddof=0)[0, 1] / np.var(pred))
    else:
        corr = float("nan")
        slope = float("nan")
    sign = float(np.mean(np.sign(pred) == np.sign(y)))
    return {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "corr": corr,
        "sign": sign,
        "slope_true_on_pred": slope,
        "mean_true": float(np.mean(y)),
        "mean_pred": float(np.mean(pred)),
        "sd_true": float(np.std(y)),
        "sd_pred": float(np.std(pred)),
    }


def percentile_ci(values: np.ndarray, level: float = 95.0) -> list[float]:
    vals = np.asarray(values, np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return [float("nan"), float("nan")]
    lo = (100.0 - level) / 2.0
    hi = 100.0 - lo
    return [float(np.percentile(vals, lo)), float(np.percentile(vals, hi))]


def cluster_bootstrap(y: np.ndarray, predictors: Mapping[str, np.ndarray],
                      clusters: np.ndarray, reps: int = BOOTSTRAP_REPS,
                      seed: int = BOOTSTRAP_SEED,
                      mask: np.ndarray | None = None) -> dict[str, Any]:
    """Percentile cluster bootstrap for delta metrics.

    ``predictors`` maps a name to per-pair predictions.  The same resampled
    clusters are used for every predictor, so paired comparisons are valid.
    """
    y = np.asarray(y, np.float64)
    clusters = np.asarray(clusters, np.int64)
    if mask is not None:
        mask = np.asarray(mask, bool)
        y = y[mask]
        clusters = clusters[mask]
        predictors = {k: np.asarray(v, np.float64)[mask] for k, v in predictors.items()}
    out: dict[str, Any] = {
        "n_pairs": int(y.size),
        "n_clusters": int(np.unique(clusters).size),
        "reps": int(reps),
        "seed": int(seed),
        "ci": "percentile 95%",
        "point": {k: delta_metrics(y, v) for k, v in predictors.items()},
        "bootstrap": {},
    }
    if y.size == 0:
        return out
    uniq = np.unique(clusters)
    members = {int(c): np.nonzero(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(int(seed))
    draws: dict[str, list[np.ndarray]] = {k: [] for k in predictors}
    for _ in range(int(reps)):
        picks = uniq[rng.integers(0, uniq.size, size=uniq.size)]
        idx = np.concatenate([members[int(c)] for c in picks])
        yy = y[idx]
        for name, pred in predictors.items():
            pp = pred[idx]
            draws[name].append(np.asarray([
                np.mean(np.abs(pp - yy)),
                np.sqrt(np.mean((pp - yy) ** 2)),
                np.mean(pp - yy),
                np.mean(np.sign(pp) == np.sign(yy)),
            ], np.float64))
    for name in predictors:
        arr = np.stack(draws[name], 0)
        out["bootstrap"][name] = {
            "mae_ci": percentile_ci(arr[:, 0]),
            "rmse_ci": percentile_ci(arr[:, 1]),
            "bias_ci": percentile_ci(arr[:, 2]),
            "sign_ci": percentile_ci(arr[:, 3]),
        }
    return out


def paired_error_bootstrap(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                           clusters: np.ndarray, reps: int = BOOTSTRAP_REPS,
                           seed: int = BOOTSTRAP_SEED,
                           mask: np.ndarray | None = None) -> dict[str, Any]:
    """Cluster bootstrap of MAE(B)-MAE(A) and RMSE(B)-RMSE(A)."""
    y = np.asarray(y, np.float64)
    pred_a = np.asarray(pred_a, np.float64)
    pred_b = np.asarray(pred_b, np.float64)
    clusters = np.asarray(clusters, np.int64)
    if mask is not None:
        mask = np.asarray(mask, bool)
        y, pred_a, pred_b, clusters = y[mask], pred_a[mask], pred_b[mask], clusters[mask]
    if y.size == 0:
        return {"n_pairs": 0}
    uniq = np.unique(clusters)
    members = {int(c): np.nonzero(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(int(reps)):
        picks = uniq[rng.integers(0, uniq.size, size=uniq.size)]
        idx = np.concatenate([members[int(c)] for c in picks])
        ea = np.abs(pred_a[idx] - y[idx])
        eb = np.abs(pred_b[idx] - y[idx])
        draws.append([float(np.mean(eb) - np.mean(ea)),
                      float(np.sqrt(np.mean((pred_b[idx] - y[idx]) ** 2))
                            - np.sqrt(np.mean((pred_a[idx] - y[idx]) ** 2)))])
    arr = np.asarray(draws, np.float64)
    ea = np.abs(pred_a - y)
    eb = np.abs(pred_b - y)
    return {
        "n_pairs": int(y.size),
        "n_clusters": int(uniq.size),
        "reps": int(reps),
        "seed": int(seed),
        "delta_mae_point": float(np.mean(eb) - np.mean(ea)),
        "delta_mae_ci": percentile_ci(arr[:, 0]),
        "delta_rmse_point": float(np.sqrt(np.mean((pred_b - y) ** 2))
                                  - np.sqrt(np.mean((pred_a - y) ** 2))),
        "delta_rmse_ci": percentile_ci(arr[:, 1]),
        "definition": "B - A; negative means B is better",
    }


# ---------------------------------------------------------------------------
# metadata block
# ---------------------------------------------------------------------------


def meta_block(inp: Inputs, protocol_id: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    meta = {
        "protocol_id": protocol_id,
        "generated_utc": now_utc(),
        "git_commit": git_commit(),
        "scope": "official Train rows only (3360); Train-sub 2856 / Dev 504",
        "test_labels_used": False,
        "valid_labels_used": False,
        "valid_used_for_model_selection": False,
        "label_policy": "non-Train label entries are masked to NaN immediately after loading and are never used",
        "labels_used": "official Train labels only",
        "hashes": inp.hashes,
        "split": {
            "n_official_train": int(inp.official_train.sum()),
            "n_train_sub": int(inp.train_sub.sum()),
            "n_dev": int(inp.dev.sum()),
            "n_scaffolds_official_train": int(inp.split.get("n_scaffolds_official_train", -1)),
            "n_scaffolds_train_sub": int(inp.split.get("n_scaffolds_train_sub", -1)),
            "n_scaffolds_dev": int(inp.split.get("n_scaffolds_dev", -1)),
            "seed": int(inp.split.get("seed", -1)),
            "assignment_rule": str(inp.split.get("assignment_rule", "")),
            "scaffold_definition": str(inp.split.get("scaffold_definition", "")),
        },
        "settings": {
            "bootstrap_reps": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "shrink_k": SHRINK_K,
            "ridge_lambda": RIDGE_LAMBDA,
            "fingerprint_bits": FINGERPRINT_BITS,
            "fingerprint_iters": FINGERPRINT_ITERS,
        },
    }
    if extra:
        meta.update(dict(extra))
    return meta