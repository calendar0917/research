"""Supervised query discovery (method A) and the complexity-matched random
library (control R).

Discovery uses **Train labels only**.  Val/Test are evaluated every round but
never enter candidate scoring, residuals, thresholds, parent ranking or the
stopping rule; they exist so the round-by-round curve is on record.

Per outer round
---------------
1. score the query pool (B + selected + last round's unused candidates) with the
   fixed residual-stump score, take the 24 best plus 8 uniformly random ones as
   parents (top-up from node primitives in canonical order if short);
2. draw 256 valid, AST-distinct candidates (96 AND = 48 parent+primitive and
   48 parent+parent, 96 NBR, 64 BRANCH) with one shared ``Generator(1701)``,
   at most 10,240 attempts, never relaxing the grammar;
3. select up to 4 positive-score candidates (ties by canonical string);
4. refit the 128-tree proxy from scratch on the enlarged bank, refresh the
   residuals and record the three-split metrics.

The proxy is a *discovery device only*: the final head is refitted on the final
bank, and no proxy state is distilled into it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

import sq_ast as A
import sq_bank as B
import sq_common as C
from sq_ast import PackedGraph

PROXY_MAX_DEPTH = 3
PROXY_ETA = 0.10
PROXY_ROUNDS = 128
PROXY_MIN_CHILD_WEIGHT = 5
PROXY_REG_LAMBDA = 10.0
PROXY_MAX_BIN = 64
PROXY_NTHREAD = 4

MAX_OUTER_ROUNDS = 16
MAX_NEW_PER_ROUND = 4
MAX_DISCOVERED = 64
N_PARENTS = 32
N_PARENTS_SCORED = 24
N_PARENTS_RANDOM = 8
N_CANDIDATES = 256
N_AND_PRIMITIVE = 48
N_AND_PAIR = 48
N_NBR = 96
N_BRANCH = 64
MAX_GENERATION_ATTEMPTS = 10_240
DISCOVERY_TIME_LIMIT_S = 45 * 60

QUANTILES = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
HALF_INTEGER_THRESHOLDS = (0.5, 1.5, 2.5, 3.5)
STUMP_MIN_MOLECULES = 8
STUMP_RIDGE = 10.0
COMPLEXITY_PENALTY = 0.05


# ---------------------------------------------------------------------------
# xgboost helpers
# ---------------------------------------------------------------------------


def xgboost_backend(requested: str = "cuda:0") -> tuple[str, dict[str, Any]]:
    """Resolve the XGBoost device honestly (CPU hist fallback is recorded)."""
    import xgboost

    info = xgboost.build_info()
    has_cuda = bool(info.get("USE_CUDA"))
    status: dict[str, Any] = {
        "requested": requested,
        "compiled_with_cuda": has_cuda,
        "fallback_reason": None,
    }
    if requested.startswith("cuda") and not has_cuda:
        status["fallback_reason"] = "xgboost built without CUDA support"
        return "cpu", status
    if requested.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                status["fallback_reason"] = "torch reports no visible CUDA device"
                return "cpu", status
        except Exception:  # noqa: BLE001 - torch absent is a CPU fallback, not a crash
            status["fallback_reason"] = "torch unavailable for a device probe"
            return "cpu", status
    return requested, status


def proxy_params(device: str) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": device,
        "max_depth": PROXY_MAX_DEPTH,
        "eta": PROXY_ETA,
        "min_child_weight": PROXY_MIN_CHILD_WEIGHT,
        "reg_lambda": PROXY_REG_LAMBDA,
        "reg_alpha": 0.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "max_bin": PROXY_MAX_BIN,
        "base_score": 0.0,
        "seed": C.TRAIN_SEED,
        "nthread": PROXY_NTHREAD,
    }


def fit_proxy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    device: str,
    *,
    names: Sequence[str] | None = None,
    eval_sets: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Fit the frozen 128-tree proxy from scratch and return train predictions."""
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(names) if names else None)
    evals = [(dtrain, "train")]
    dmats: dict[str, Any] = {"train": dtrain}
    if eval_sets:
        for split, (X, y) in eval_sets.items():
            dmats[split] = xgb.DMatrix(X, label=y, feature_names=list(names) if names else None)
            evals.append((dmats[split], split))
    t0 = time.perf_counter()
    booster = xgb.train(
        proxy_params(device),
        dtrain,
        num_boost_round=PROXY_ROUNDS,
        evals=evals,
        verbose_eval=False,
        early_stopping_rounds=None,
    )
    seconds = time.perf_counter() - t0
    preds = {
        split: booster.predict(dmat, iteration_range=(0, PROXY_ROUNDS)).astype(np.float64)
        for split, dmat in dmats.items()
    }
    return {
        "booster": booster,
        "preds": preds,
        "seconds": seconds,
        "n_rounds": PROXY_ROUNDS,
        "n_features": int(X_train.shape[1]),
    }


# ---------------------------------------------------------------------------
# residual stump scoring
# ---------------------------------------------------------------------------


def stump_thresholds(values: np.ndarray, *, counts_column: bool) -> list[float]:
    candidates: list[float] = [0.0]
    finite = values[np.isfinite(values)]
    if finite.size:
        for q in QUANTILES:
            candidates.append(float(np.quantile(finite, q)))
    if counts_column:
        candidates.extend(HALF_INTEGER_THRESHOLDS)
    unique: list[float] = []
    for value in sorted(candidates):
        if not unique or value != unique[-1]:
            unique.append(value)
    return unique


def best_stump(
    values: np.ndarray,
    residual: np.ndarray,
    *,
    counts_column: bool,
    min_molecules: int = STUMP_MIN_MOLECULES,
) -> dict[str, Any] | None:
    """Best residual stump over the frozen threshold grid (float64 arithmetic)."""
    x = np.asarray(values, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64)
    n = int(r.size)
    if n == 0 or np.all(x == x[0]):
        return None
    total = float(r.sum())
    best: dict[str, Any] | None = None
    for threshold in stump_thresholds(x, counts_column=counts_column):
        left = x <= threshold
        n_left = int(left.sum())
        n_right = n - n_left
        if n_left < min_molecules or n_right < min_molecules:
            continue
        s_left = float(r[left].sum())
        s_right = total - s_left
        gain = (
            s_left * s_left / (n_left + STUMP_RIDGE)
            + s_right * s_right / (n_right + STUMP_RIDGE)
            - total * total / (n + STUMP_RIDGE)
        ) / n
        if best is None or gain > best["gain"]:
            best = {
                "gain": float(gain),
                "threshold": float(threshold),
                "n_left": n_left,
                "n_right": n_right,
            }
    return best


def score_query(
    train_packed: PackedGraph,
    ast: tuple,
    residual: np.ndarray,
    *,
    cache: B.MaskCache | None = None,
    train_gain: float | None = None,
) -> dict[str, Any]:
    """Frozen score: best gain over the c and p columns, discounted by AST cost."""
    mask = B.root_mask(train_packed, ast, cache)
    features = A.query_features(train_packed, mask)
    counts = features[:, 0].astype(np.float64)
    density = features[:, 1].astype(np.float64)
    support = int((features[:, 2] > 0).sum())
    result: dict[str, Any] = {
        "ast": ast,
        "canon": A.canon(ast),
        "nodes": A.ast_nodes(ast),
        "hops": A.ast_hops(ast),
        "train_support": support,
        "score": 0.0,
        "gain": 0.0,
        "reject_reason": None,
        "features": features,
    }
    if support < STUMP_MIN_MOLECULES:
        result["reject_reason"] = "support"
        return result
    if np.all(features == features[0:1, :]):
        result["reject_reason"] = "constant"
        return result
    stump_c = best_stump(counts, residual, counts_column=True)
    stump_p = best_stump(density, residual, counts_column=False)
    stumps = [stump for stump in (stump_c, stump_p) if stump is not None]
    if not stumps:
        result["reject_reason"] = "no_split"
        return result
    best = max(stumps, key=lambda item: item["gain"])
    gain = float(best["gain"])
    penalty = 1.0 + COMPLEXITY_PENALTY * (A.ast_nodes(ast) - 1)
    result["gain"] = gain
    result["score"] = gain / penalty
    result["stump"] = best
    if train_gain is not None:
        result["train_gain"] = float(train_gain)
    if gain <= 0.0:
        result["reject_reason"] = "non_positive"
    return result


# ---------------------------------------------------------------------------
# candidate generation
# ---------------------------------------------------------------------------


@dataclass
class GenerationStats:
    attempts: int = 0
    accepted: int = 0
    and_primitive: int = 0
    and_pair: int = 0
    nbr: int = 0
    branch: int = 0
    duplicate_ast: int = 0
    invalid_ast: int = 0
    exhausted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "accepted": self.accepted,
            "and_primitive": self.and_primitive,
            "and_pair": self.and_pair,
            "nbr": self.nbr,
            "branch": self.branch,
            "duplicate_ast": self.duplicate_ast,
            "invalid_ast": self.invalid_ast,
            "exhausted": self.exhausted,
        }


def generate_candidates(
    parents: Sequence[tuple],
    node_prim_list: Sequence[tuple],
    bond_list: Sequence[tuple],
    *,
    rng: np.random.Generator,
    existing: set[str],
) -> tuple[list[tuple], GenerationStats]:
    stats = GenerationStats()
    quotas = {
        "and_primitive": N_AND_PRIMITIVE,
        "and_pair": N_AND_PAIR,
        "nbr": N_NBR,
        "branch": N_BRANCH,
    }
    seen = set(existing)
    out: list[tuple] = []
    kinds = list(quotas)

    def pick_parent() -> tuple:
        return parents[int(rng.integers(0, len(parents)))]

    while any(quota > 0 for quota in quotas.values()) and stats.attempts < MAX_GENERATION_ATTEMPTS:
        live = [kind for kind in kinds if quotas[kind] > 0]
        kind = live[int(rng.integers(0, len(live)))]
        quotas[kind] -= 1
        stats.attempts += 1
        if kind == "and_primitive":
            parent = pick_parent()
            prim = node_prim_list[int(rng.integers(0, len(node_prim_list)))]
            candidate = A.AND(parent, prim)
        elif kind == "and_pair":
            candidate = A.AND(pick_parent(), pick_parent())
        elif kind == "nbr":
            parent = pick_parent()
            bond = bond_list[int(rng.integers(0, len(bond_list)))]
            m = int(rng.integers(1, 4))
            candidate = A.NBR(parent, bond, m)
        else:
            bond1 = bond_list[int(rng.integers(0, len(bond_list)))]
            bond2 = bond_list[int(rng.integers(0, len(bond_list)))]
            candidate = A.BRANCH(pick_parent(), pick_parent(), bond1, bond2)
        if not A.validates(candidate):
            stats.invalid_ast += 1
            continue
        key = A.canon(candidate)
        if key in seen:
            stats.duplicate_ast += 1
            continue
        seen.add(key)
        out.append(candidate)
        stats.accepted += 1
        stats.__dict__[kind] += 1
    stats.exhausted = any(quota > 0 for quota in quotas.values())
    return out, stats


# ---------------------------------------------------------------------------
# discovery driver
# ---------------------------------------------------------------------------


def discover_queries(
    *,
    train_packed: PackedGraph,
    valid_packed: PackedGraph,
    test_packed: PackedGraph,
    base_records: Sequence[B.QueryRecord],
    vocabulary: Mapping[str, Any],
    y_train_std: np.ndarray,
    y_valid_std: np.ndarray,
    y_test_std: np.ndarray,
    scaler: Mapping[str, float],
    device: str,
    cache: B.MaskCache | None = None,
    time_limit_s: float = DISCOVERY_TIME_LIMIT_S,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the frozen discovery loop and return the final bank plus the log."""
    rng = np.random.Generator(np.random.PCG64(C.SEARCH_RNG_SEED))
    node_prim_list = B.node_primitives(vocabulary)
    bond_list = B.bond_predicates(vocabulary)

    bank: list[B.QueryRecord] = list(base_records)
    selected: list[B.QueryRecord] = []
    log: list[dict[str, Any]] = []
    truncated = False
    stop_reason = "max_rounds"
    t_start = time.perf_counter()
    proxy_seconds_total = 0.0

    def build_matrix(packed: PackedGraph) -> np.ndarray:
        matrix, _names = B.design_matrix(packed, bank, cache=cache, include_base=True)
        return matrix

    X_train = build_matrix(train_packed)
    X_valid = build_matrix(valid_packed)
    X_test = build_matrix(test_packed)
    fit0 = fit_proxy(
        X_train,
        y_train_std,
        device,
        eval_sets={"valid": (X_valid, y_valid_std), "test": (X_test, y_test_std)},
    )
    proxy_seconds_total += fit0["seconds"]
    residual = np.asarray(y_train_std, dtype=np.float64) - fit0["preds"]["train"]
    round_metrics = _proxy_round_metrics(fit0["preds"], scaler, y_train_std, y_valid_std, y_test_std)
    log.append(
        {
            "outer_round": 0,
            "n_queries": len(bank),
            "n_discovered": 0,
            "proxy_seconds": fit0["seconds"],
            "proxy_rounds": PROXY_ROUNDS,
            "elapsed_seconds": time.perf_counter() - t_start,
            "new_queries": [],
            **round_metrics,
        }
    )
    if progress is not None:
        progress(log[-1])

    pool: list[tuple] = [record.ast for record in bank]
    frontier: list[tuple] = []
    feature_keys = {A.query_features(train_packed, B.root_mask(train_packed, rec.ast, cache)).tobytes() for rec in bank}

    for outer_round in range(1, MAX_OUTER_ROUNDS + 1):
        if timeout_hit(t_start, time_limit_s):
            truncated = True
            stop_reason = "RESOURCE_TRUNCATED"
            break
        if len(selected) >= MAX_DISCOVERED:
            stop_reason = "max_discovered"
            break

        # ---- 1. pool scores and parents -----------------------------------
        pool = _unique_asts(list(pool) + frontier + [rec.ast for rec in bank])
        t_score = time.perf_counter()
        scored = [
            score_query(train_packed, ast, residual, cache=cache)
            for ast in pool
        ]
        scored.sort(key=lambda item: (-item["score"], item["canon"]))
        top_scored = [item for item in scored if item["score"] > 0][:N_PARENTS_SCORED]
        parents: list[tuple] = [item["ast"] for item in top_scored]
        if len(parents) < N_PARENTS_SCORED:
            parents.extend(_top_up_parents(parents, node_prim_list, N_PARENTS_SCORED))
        candidates_pool = [item["ast"] for item in scored]
        if len(candidates_pool) > len(parents):
            randoms = rng.choice(
                len(candidates_pool), size=min(N_PARENTS_RANDOM, len(candidates_pool)), replace=False
            )
            for idx in sorted(int(i) for i in randoms):
                ast = candidates_pool[idx]
                if ast not in parents:
                    parents.append(ast)
        parents = parents[:N_PARENTS]
        if len(parents) < N_PARENTS:
            parents = _top_up_parents(parents, node_prim_list, N_PARENTS)
        score_seconds = time.perf_counter() - t_score
        # frontier = the highest-scoring *unselected* candidates for the next round
        frontier = [item["ast"] for item in scored if item["ast"] not in {r.ast for r in bank}][:N_PARENTS]

        # ---- 2. candidate generation (Train structures only) ---------------
        existing_keys = {A.canon(rec.ast) for rec in bank}
        candidates, gen_stats = generate_candidates(
            parents, node_prim_list, bond_list, rng=rng, existing=existing_keys
        )

        # ---- 3. score, dedupe, select up to 4 positive ---------------------
        cand_scored = [score_query(train_packed, ast, residual, cache=cache) for ast in candidates]
        cand_scored.sort(key=lambda item: (-item["score"], item["canon"]))
        chosen: list[dict[str, Any]] = []
        for item in cand_scored:
            if len(chosen) >= MAX_NEW_PER_ROUND or len(selected) + len(chosen) >= MAX_DISCOVERED:
                break
            if item["score"] <= 0.0:
                break
            key = item["features"].tobytes()
            if key in feature_keys:
                continue
            feature_keys.add(key)
            chosen.append(item)
        if not chosen:
            stop_reason = "no_positive_candidate"
            log.append(
                {
                    "outer_round": outer_round,
                    "n_queries": len(bank),
                    "n_discovered": len(selected),
                    "proxy_seconds": 0.0,
                    "proxy_rounds": 0,
                    "elapsed_seconds": time.perf_counter() - t_start,
                    "new_queries": [],
                    "n_candidates": len(candidates),
                    "score_seconds": score_seconds,
                    "generation": gen_stats.to_dict(),
                    "stop_reason": stop_reason,
                }
            )
            if progress is not None:
                progress(log[-1])
            break

        for item in chosen:
            record = B.QueryRecord(
                ast=item["ast"],
                label=f"A{len(selected):03d}",
                source="discovered",
                outer_round=outer_round,
                train_support=int(item["train_support"]),
                train_gain=float(item["gain"]),
            )
            selected.append(record)
            bank.append(record)

        # ---- 4. refit proxy on the enlarged bank ---------------------------
        X_train = build_matrix(train_packed)
        X_valid = build_matrix(valid_packed)
        X_test = build_matrix(test_packed)
        fit = fit_proxy(
            X_train,
            y_train_std,
            device,
            eval_sets={"valid": (X_valid, y_valid_std), "test": (X_test, y_test_std)},
        )
        proxy_seconds_total += fit["seconds"]
        residual = np.asarray(y_train_std, dtype=np.float64) - fit["preds"]["train"]
        round_metrics = _proxy_round_metrics(fit["preds"], scaler, y_train_std, y_valid_std, y_test_std)
        entry = {
            "outer_round": outer_round,
            "n_queries": len(bank),
            "n_discovered": len(selected),
            "proxy_seconds": fit["seconds"],
            "proxy_rounds": PROXY_ROUNDS,
            "elapsed_seconds": time.perf_counter() - t_start,
            "n_candidates": len(candidates),
            "score_seconds": score_seconds,
            "generation": gen_stats.to_dict(),
            "new_queries": [
                {
                    "label": record.label,
                    "ast": A.canon(record.ast),
                    "train_gain": record.train_gain,
                    "train_support": record.train_support,
                    "nodes": A.ast_nodes(record.ast),
                    "hops": A.ast_hops(record.ast),
                }
                for record in selected[-len(chosen):]
            ],
            **round_metrics,
        }
        log.append(entry)
        if progress is not None:
            progress(entry)

    elapsed = time.perf_counter() - t_start
    if timeout_hit(t_start, time_limit_s) and not truncated:
        truncated = True
        stop_reason = "RESOURCE_TRUNCATED"

    return {
        "bank": bank,
        "base_records": list(base_records),
        "discovered": selected,
        "log": log,
        "stop_reason": stop_reason,
        "resource_truncated": truncated,
        "elapsed_seconds": elapsed,
        "proxy_seconds_total": proxy_seconds_total,
        "n_outer_rounds": len(log) - 1,
        "discovery_time_limit_s": float(time_limit_s),
    }


def timeout_hit(start: float, limit_s: float) -> bool:
    return (time.perf_counter() - start) > limit_s


def _unique_asts(asts: Iterable[tuple]) -> list[tuple]:
    seen: dict[str, tuple] = {}
    for ast in asts:
        seen.setdefault(A.canon(ast), ast)
    return [seen[key] for key in sorted(seen)]


def _top_up_parents(parents: Sequence[tuple], node_prim_list: Sequence[tuple], target: int) -> list[tuple]:
    out = list(parents)
    for ast in node_prim_list:
        if len(out) >= target:
            break
        if ast not in out:
            out.append(ast)
    return out


def _proxy_round_metrics(
    preds: Mapping[str, np.ndarray],
    scaler: Mapping[str, float],
    y_train_std: np.ndarray,
    y_valid_std: np.ndarray,
    y_test_std: np.ndarray,
) -> dict[str, float]:
    """Proxy metrics in raw units (recorded only; never used for selection)."""
    mean = float(scaler["mean"])
    std = float(scaler["std"])
    out: dict[str, float] = {}
    for split, y_std in (("train", y_train_std), ("valid", y_valid_std), ("test", y_test_std)):
        pred = preds[split] * std + mean
        target = np.asarray(y_std, dtype=np.float64) * std + mean
        out[f"proxy_{split}_rmse_raw"] = C.rmse(pred, target)
        out[f"proxy_{split}_mae_raw"] = C.mae(pred, target)
    return out


# ---------------------------------------------------------------------------
# control R: complexity-matched random library
# ---------------------------------------------------------------------------

MAX_R_ATTEMPTS_PER_SLOT = 512
R_TIME_LIMIT_S = 45 * 60


def ast_shape(ast: tuple) -> dict[str, Any]:
    return A.operator_shape(ast)


def _parameter_maps(
    ast: tuple, vocabulary: Mapping[str, Any], rng: np.random.Generator
) -> dict[tuple[str, ...], dict[int, int]]:
    """One random bijection per (query, column) for each parameter family."""
    maps: dict[tuple[str, ...], dict[int, int]] = {}
    for kind, leaf in A.leaf_params(ast):
        if kind == "node" and leaf[0] == "ATOM_EQ":
            key = ("atom", str(leaf[1]))
            if key not in maps:
                values = list(vocabulary["atom_values"][leaf[1]])
                maps[key] = _random_bijection(values, rng)
        elif kind == "bond":
            key = ("bond", str(leaf[1]))
            if key not in maps:
                values = list(vocabulary["bond_values"][leaf[1]])
                maps[key] = _random_bijection(values, rng)
        elif kind == "node" and leaf[0] == "DEG_EQ":
            key = ("deg",)
            if key not in maps:
                maps[key] = _random_bijection(list(range(0, 6)), rng)
        elif kind == "node" and leaf[0] == "MIN_RING_EQ":
            key = ("ring",)
            if key not in maps:
                maps[key] = _random_bijection(list(range(3, 9)), rng)
    return maps


def _random_bijection(values: Sequence[int], rng: np.random.Generator) -> dict[int, int]:
    unique = sorted(set(int(v) for v in values))
    if len(unique) <= 1:
        return {value: value for value in unique}
    permuted = list(unique)
    rng.shuffle(permuted)
    return {value: permuted[i] for i, value in enumerate(unique)}


def remap_ast(ast: tuple, maps: Mapping[tuple[str, ...], Mapping[int, int]]) -> tuple:
    kind = ast[0]
    if kind == "ATOM_EQ":
        return A.ATOM_EQ(ast[1], maps[("atom", str(ast[1]))][ast[2]])
    if kind == "DEG_EQ":
        return A.DEG_EQ(maps[("deg",)][ast[1]])
    if kind == "MIN_RING_EQ":
        return A.MIN_RING_EQ(maps[("ring",)][ast[1]])
    if kind in A.NODE_KINDS:
        return ast
    if kind == "AND":
        return A.AND(*[remap_ast(child, maps) for child in ast[1:]])
    if kind == "NBR":
        bond = ast[2]
        if bond[0] == "BOND_EQ":
            bond = A.BOND_EQ(bond[1], maps[("bond", str(bond[1]))][bond[2]])
        return A.NBR(remap_ast(ast[1], maps), bond, ast[3])
    if kind == "BRANCH":
        b1, b2 = ast[3], ast[4]
        if b1[0] == "BOND_EQ":
            b1 = A.BOND_EQ(b1[1], maps[("bond", str(b1[1]))][b1[2]])
        if b2[0] == "BOND_EQ":
            b2 = A.BOND_EQ(b2[1], maps[("bond", str(b2[1]))][b2[2]])
        return A.BRANCH(
            remap_ast(ast[1], maps), remap_ast(ast[2], maps), b1, b2
        )
    raise ValueError(f"cannot remap AST kind {kind!r}")


def build_random_library(
    *,
    discovered: Sequence[B.QueryRecord],
    train_packed: PackedGraph,
    vocabulary: Mapping[str, Any],
    existing_records: Sequence[B.QueryRecord],
    cache: B.MaskCache | None = None,
    rng_seed: int = C.RANDOM_LIBRARY_SEED,
    max_attempts_per_slot: int = MAX_R_ATTEMPTS_PER_SLOT,
    time_limit_s: float = R_TIME_LIMIT_S,
) -> dict[str, Any]:
    """Shape-matched random queries; parameters remapped, structure preserved."""
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    t_start = time.perf_counter()
    existing_ast = {A.canon(rec.ast) for rec in existing_records}
    feature_keys = {
        A.query_features(train_packed, B.root_mask(train_packed, rec.ast, cache)).tobytes()
        for rec in existing_records
    }
    records: list[B.QueryRecord] = []
    slots: list[dict[str, Any]] = []
    truncated = False
    counter = 0
    for source in discovered:
        if (time.perf_counter() - t_start) > time_limit_s:
            truncated = True
        filled: B.QueryRecord | None = None
        attempts = 0
        reasons: dict[str, int] = {}
        while filled is None and attempts < max_attempts_per_slot and not truncated:
            attempts += 1
            maps = _parameter_maps(source.ast, vocabulary, rng)
            candidate = remap_ast(source.ast, maps)
            if not A.validates(candidate):
                reasons["invalid_ast"] = reasons.get("invalid_ast", 0) + 1
                continue
            key = A.canon(candidate)
            if key in existing_ast:
                reasons["duplicate_ast"] = reasons.get("duplicate_ast", 0) + 1
                continue
            ok, reason, mask = B.validate_query_on_train(
                train_packed, candidate, cache=cache
            )
            if not ok:
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            fkey = A.query_features(train_packed, mask).tobytes()
            if fkey in feature_keys:
                reasons["duplicate_features"] = reasons.get("duplicate_features", 0) + 1
                continue
            feature_keys.add(fkey)
            existing_ast.add(key)
            record = B.QueryRecord(
                ast=candidate,
                label=f"R{counter:03d}",
                source="random",
                outer_round=int(source.outer_round),
                train_support=int((train_packed.graph_counts(mask) > 0).sum()),
                train_gain=None,
            )
            counter += 1
            records.append(record)
            filled = record
        slots.append(
            {
                "source_label": source.label,
                "source_ast": A.canon(source.ast),
                "filled": filled is not None,
                "label": None if filled is None else filled.label,
                "ast": None if filled is None else A.canon(filled.ast),
                "attempts": attempts,
                "reject_reasons": reasons,
                "shape_match": None
                if filled is None
                else ast_shape(filled.ast) == ast_shape(source.ast),
            }
        )
    n_missing = sum(1 for slot in slots if not slot["filled"])
    return {
        "records": records,
        "slots": slots,
        "n_requested": len(discovered),
        "n_filled": len(records),
        "n_missing": int(n_missing),
        "status": "MATCH_COMPLETE" if n_missing == 0 else "MATCH_INCOMPLETE",
        "resource_truncated": bool(truncated),
        "elapsed_seconds": time.perf_counter() - t_start,
        "rng_seed": int(rng_seed),
        "all_shapes_match": all(slot["shape_match"] for slot in slots if slot["filled"]),
    }
