"""Phase B + Phase C: sparse rule models, B2 augmentation, joint-rule contrasts.

Phase B (PROTOCOL §5)
    z(G) = b + a1*z(log1p n_atoms) + a2*z(log1p n_bonds) + sum_k w_k z(r_k(G))
    R0 / R16 / R64 / R128 by Train-only forward selection (gradient score, C=0.03).

Phase C (§6)
    z(G) = z_B2(G) + b_delta + sum_k w_k z(r_k(G))         (B2 frozen)
    B2-L64 (counts) vs B2-U64 (thresholds) on identical structures,
    then B2-U+S / B2-U+P / B2-U+N (K joint rules) and R64+P.

All fitting uses Train only; Valid chooses C ∈ {0.03, 0.3}; Test is
``DEVELOPMENT_TEST`` (evaluated, never used for selection).
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

import isrd_common as K
import isrd_rules as R

C_SELECT = 0.03
C_GRID = (0.03, 0.3)
BATCH = 16
CAPACITIES = (16, 64, 128)


# ---------------------------------------------------------------------------
# design
# ---------------------------------------------------------------------------


def size_matrix(data: K.Data) -> np.ndarray:
    return np.column_stack([np.log1p(data.n_atoms.astype(np.float64)),
                            np.log1p(data.n_bonds.astype(np.float64))])


def standardise(train_vals: np.ndarray, other_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    mu = train_vals.mean(axis=0)
    sd = train_vals.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (train_vals - mu) / sd, (other_vals - mu) / sd, {"mean": mu, "std": sd}


def rule_columns(rules: list[dict], data: K.Data) -> np.ndarray:
    return np.column_stack([R.rule_activation(data, r).astype(np.float64) for r in rules])


def pair_columns(pairs: list[dict], by_id: dict[str, dict], data: K.Data) -> np.ndarray:
    return np.column_stack([R.pair_activation(data, p, by_id).astype(np.float64) for p in pairs])


def count_columns(singles_idx: list[dict], data: K.Data) -> np.ndarray:
    """Frozen scaled counts ``x_s(G)`` for the 64 R64 structure identities."""
    return np.column_stack([data.col(c["granularity"], c["column"]).astype(np.float64)
                            for c in singles_idx])


# ---------------------------------------------------------------------------
# forward selection (Phase B)
# ---------------------------------------------------------------------------


def forward_select(data: K.Data, singles: list[dict], cap: int) -> list[dict]:
    train = data.rows["train"]
    y = data.y[train].astype(np.float64)
    sw = K.class_balanced_weights(y)
    size_tr = size_matrix(data)[train]
    size_z, _, size_stats = standardise(size_tr, size_tr)
    acts = np.column_stack([R.rule_activation(data, r).astype(np.float64) for r in singles]) \
        if False else None
    activations = R.rule_matrix(data, singles)[train]
    mu = activations.mean(axis=0)
    sd = activations.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    acts_z = (activations - mu) / sd

    chosen: list[int] = []
    log: list[dict] = []
    size_coef: np.ndarray | None = None
    X_all = size_z
    fit = K.fit_logistic(X_all, y, sw, C_SELECT)
    p = K.sigmoid(fit["logit"])
    while len(chosen) < cap:
        used_cols = {(singles[i]["granularity"], singles[i]["column"]) for i in chosen}
        resid = sw * (y - p)
        numer = acts_z.T @ resid
        curv = ((acts_z ** 2).T @ (sw * p * (1.0 - p))) + 1.0 / C_SELECT
        score = numer ** 2 / curv
        allowed = np.array([(c["granularity"], c["column"]) not in used_cols
                            and i not in chosen for i, c in enumerate(singles)])
        score = np.where(allowed, score, -np.inf)
        order = [int(i) for i in np.argsort(-score, kind="stable") if np.isfinite(score[i])]
        take: list[int] = []
        for i in order:
            if len(take) >= BATCH:
                break
            key = (singles[i]["granularity"], singles[i]["column"])
            if key in used_cols or any((singles[j]["granularity"], singles[j]["column"]) == key
                                       for j in take):
                continue
            take.append(i)
        if not take:
            break
        # PROTOCOL §5.2: select the batch, then jointly refit the selected rules once
        chosen.extend(take)
        fit = K.fit_logistic(np.column_stack([size_z, acts_z[:, chosen]]), y, sw, C_SELECT)
        p = K.sigmoid(fit["logit"])
        log.append({"step": len(chosen), "batch": [singles[i]["rule_id"] for i in take],
                    "scores": [float(score[i]) for i in take],
                    "objective": float(fit["objective"])})
        print(f"[B] selected {len(chosen)} rules (objective {fit['objective']:.3f})", flush=True)
    return [singles[i] for i in chosen], log


# ---------------------------------------------------------------------------
# model fitting / evaluation helpers
# ---------------------------------------------------------------------------


def fit_and_eval(name: str, X_all: np.ndarray, y: np.ndarray, sw: np.ndarray,
                 offset_all: np.ndarray | None, data: K.Data,
                 penalize_intercept: bool = True) -> dict[str, Any]:
    """Fit on Train for every C, evaluate all splits, select by Valid ROC-AUC."""
    train = data.rows["train"]
    fits = []
    for C in C_GRID:
        fit = K.fit_logistic(X_all[train], y[train], sw[train], C,
                             offset=None if offset_all is None else offset_all[train],
                             penalize_intercept=penalize_intercept)
        logit = np.full(data.y.size, np.nan)
        if offset_all is None:
            Xa = np.hstack([np.ones((data.y.size, 1)), X_all])
            logit = Xa @ np.concatenate([[fit["intercept"]], fit["coef"]])
        else:
            logit = offset_all + fit["intercept"] + X_all @ fit["coef"]
        m = K.all_metrics(logit, data.y, data.rows)
        fits.append({"C": C, "fit": fit, "logit": logit, "metrics": m})
    best = max(fits, key=lambda f: (f["metrics"]["valid"]["roc_auc"], -f["C"]))
    return {"name": name, "fits": fits, "selected": best["C"]}


def summarize(fitted: dict[str, Any]) -> dict[str, Any]:
    best = next(f for f in fitted["fits"] if f["C"] == fitted["selected"])
    return {"arm": fitted["name"], "selected_C": best["C"],
            "logit": best["logit"], "metrics": best["metrics"],
            "intercept": best["fit"]["intercept"], "coef": best["fit"]["coef"],
            "converged": best["fit"]["converged"], "grad_inf": best["fit"]["grad_inf"],
            "objective": best["fit"]["objective"]}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-K", type=int, default=16)
    args = ap.parse_args()
    t0 = time.perf_counter()
    data = K.Data()
    phaseA = K.read_json(K.ANALYSIS / "phaseA_rules.json")
    singles = phaseA["singles"]
    pairs = phaseA["pairs"]
    randoms = phaseA["randoms"]
    train = data.rows["train"]
    y = data.y.astype(np.float64)
    sw = K.class_balanced_weights(y[train])   # per-Train-row weights
    sw_all = np.zeros(data.y.size)
    sw_all[train] = sw
    sindex = {c["rule_id"]: c for c in singles}

    # ---------------- B2 replay (identity check, no refit) -----------------
    b2_coef = np.load(K.B2_COEF)["B2|0.03"].astype(np.float64)
    with np.load(K.B2_PRED) as z:
        b2_saved = {k: z[f"B2|0.03|{k}"].astype(np.float64) for k in K.SPLITS}
    z_b2_full = np.zeros(data.y.size)
    for k in K.SPLITS:
        z_b2_full[data.rows[k]] = b2_saved[k]
    b2_check = b2_replay(data, b2_coef, b2_saved)
    print(f"[C] B2 replay max|dz|={b2_check['max_abs_diff']:.3e}", flush=True)

    # ---------------- Phase B: R0 / R16 / R64 / R128 ----------------------
    size_all = size_matrix(data)
    _, size_z, size_stats = standardise(size_all[train], size_all)
    results: dict[str, dict] = {}
    r0 = fit_and_eval("R0", size_z, y, sw_all, None, data)
    results["R0"] = summarize(r0)

    selected, sel_log = forward_select(data, singles, cap=max(CAPACITIES))
    K.write_json(K.ANALYSIS / "phaseB_selection.json",
                 {"selection_log": sel_log, "selected_rule_ids": [c["rule_id"] for c in selected],
                  "C_select": C_SELECT, "batch": BATCH})
    rule_all = R.rule_matrix(data, selected)
    _, _rule_z, _rstat = standardise(rule_all[train], rule_all)
    mu_r, sd_r = _rstat["mean"], _rstat["std"]
    rule_z = _rule_z

    arms_X: dict[str, np.ndarray] = {"R0": size_z}
    arms_off: dict[str, np.ndarray | None] = {"R0": None}
    for cap in CAPACITIES:
        n = min(cap, len(selected))
        X = np.column_stack([size_z, rule_z[:, :n]])
        arms_X[f"R{cap}"] = X
        arms_off[f"R{cap}"] = None
        results[f"R{cap}"] = summarize(fit_and_eval(f"R{cap}", X, y, sw_all, None, data))
        print(f"[B] R{cap} valid={results[f'R{cap}']['metrics']['valid']['roc_auc']:.4f} "
              f"test={results[f'R{cap}']['metrics']['test']['roc_auc']:.4f}", flush=True)

    r64_rules = selected[:min(64, len(selected))]
    r64_idx = list(range(len(r64_rules)))

    # ---------------- Phase C: B2 increments ------------------------------
    u64_act = R.rule_matrix(data, r64_rules)
    u64_z = (u64_act - u64_act[train].mean(axis=0)) / np.where(
        u64_act[train].std(axis=0) > 0, u64_act[train].std(axis=0), 1.0)
    l64_vals = count_columns(r64_rules, data)
    l64_z = (l64_vals - l64_vals[train].mean(axis=0)) / np.where(
        l64_vals[train].std(axis=0) > 0, l64_vals[train].std(axis=0), 1.0)

    z_b2 = z_b2_full
    c_arms = {}
    c_arms["B2"] = {"arm": "B2", "logit": z_b2, "metrics": K.all_metrics(z_b2, data.y, data.rows),
                    "intercept": 0.0, "coef": np.zeros(0)}
    for name, X in (("B2-L64", l64_z), ("B2-U64", u64_z)):
        arms_X[name] = X
        arms_off[name] = z_b2
        res = summarize(fit_and_eval(name, X, y, sw_all, z_b2, data, penalize_intercept=False))
        c_arms[name] = res
        print(f"[C] {name} valid={res['metrics']['valid']['roc_auc']:.4f} "
              f"test={res['metrics']['test']['roc_auc']:.4f}", flush=True)

    # identity check: same structures, counts vs threshold predicates
    struct_ids = [(c["granularity"], c["column"]) for c in r64_rules]
    assert len(set(struct_ids)) == len(struct_ids), "R64 uses a structure twice"

    # ---------------- joint-rule contrasts (base B2-U64) ------------------
    base = c_arms["B2-U64"]
    base_train_logit = base["logit"][train]
    p_base = K.sigmoid(base_train_logit)
    resid = sw * (y[train] - p_base)
    curv0 = sw * p_base * (1.0 - p_base)

    def score_columns(cols_train: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
        zc = (cols_train - mu) / sd
        numer = zc.T @ resid
        curv = ((zc ** 2).T @ curv0) + 1.0 / C_SELECT
        return numer ** 2 / curv

    represented = set()
    for c in r64_rules:
        represented.add((c["granularity"], c["column"], float(np.float32(c["threshold"]))))
        represented.add((c["granularity"], c["column"], "COMPLEMENT"))
        represented.add((c["granularity"], c["column"]))   # structure identity is in U64

    eligible = [p for p in pairs if pair_eligible(p, sindex, r64_rules)]

    def score_pairs(pool: list[dict]) -> list[tuple[float, dict]]:
        if not pool:
            return []
        M = pair_columns(pool, sindex, data)
        mu = M[train].mean(axis=0)
        sd = np.where(M[train].std(axis=0) > 0, M[train].std(axis=0), 1.0)
        sc = score_columns(M[train], mu, sd)
        return sorted([(float(s), p) for s, p in zip(sc, pool)], key=lambda t: -t[0])

    real_scored = score_pairs(eligible)
    random_scored = score_pairs(randoms)

    used_single = {c["rule_id"] for c in r64_rules}
    fresh = [c for c in singles if c["rule_id"] not in used_single]
    fresh_M = rule_columns(fresh, data) if fresh else np.zeros((data.y.size, 0))
    if fresh:
        mu = fresh_M[train].mean(axis=0)
        sd = np.where(fresh_M[train].std(axis=0) > 0, fresh_M[train].std(axis=0), 1.0)
        fresh_sc = score_columns(fresh_M[train], mu, sd)
    else:
        fresh_sc = np.zeros(0)
    single_scored = sorted([(float(s), c) for s, c in zip(fresh_sc, fresh)], key=lambda t: -t[0])

    K_avail = min(len(real_scored), len(random_scored), len(single_scored), args.joint_K)
    joint = {"n_eligible_real": len(eligible), "n_random": len(randoms),
             "n_fresh_single": len(single_scored), "K": K_avail,
             "min_budget": 8}
    print(f"[C] joint budget {joint}", flush=True)

    joint_arms: dict[str, dict] = {}
    if K_avail >= 8:
        real_sel = [p for _s, p in real_scored[:K_avail]]
        rand_sel = [p for _s, p in random_scored[:K_avail]]
        fresh_sel = [c for _s, c in single_scored[:K_avail]]

        def with_extra(extra_cols: np.ndarray) -> np.ndarray:
            if extra_cols.size == 0:
                return u64_z
            mu = extra_cols[train].mean(axis=0)
            sd = np.where(extra_cols[train].std(axis=0) > 0, extra_cols[train].std(axis=0), 1.0)
            return np.column_stack([u64_z, (extra_cols - mu) / sd])

        arm_specs = [
            ("B2-U+S", rule_columns(fresh_sel, data), [c["rule_id"] for c in fresh_sel]),
            ("B2-U+P", pair_columns(real_sel, sindex, data), [p["pair_id"] for p in real_sel]),
            ("B2-U+N", pair_columns(rand_sel, sindex, data), [p["pair_id"] for p in rand_sel]),
        ]
        for name, extra, members in arm_specs:
            X = with_extra(extra)
            res = summarize(fit_and_eval(name, X, y, sw_all, z_b2, data, penalize_intercept=False))
            joint_arms[name] = res
            joint_arms[name]["members"] = members
            arms_X[name], arms_off[name] = X, z_b2
            print(f"[C] {name} valid={res['metrics']['valid']['roc_auc']:.4f} "
                  f"test={res['metrics']['test']['roc_auc']:.4f}", flush=True)

        # R64+P: size + U64 + same K joint rules, fitted directly on y
        X_rp = np.column_stack([size_z, with_extra(pair_columns(real_sel, sindex, data))])
        rp = summarize(fit_and_eval("R64+P", X_rp, y, sw_all, None, data))
        joint_arms["R64+P"] = rp
        joint_arms["R64+P"]["members"] = [p["pair_id"] for p in real_sel]
        arms_X["R64+P"], arms_off["R64+P"] = X_rp, None
        print(f"[B] R64+P valid={rp['metrics']['valid']['roc_auc']:.4f} "
              f"test={rp['metrics']['test']['roc_auc']:.4f}", flush=True)
    else:
        joint["stop_rule"] = "INSUFFICIENT_PAIR_SUPPORT"

    # ---------------- save predictions ------------------------------------
    K.PREDICTIONS.mkdir(parents=True, exist_ok=True)
    all_best = {**results, **c_arms, **joint_arms}
    for name, res in all_best.items():
        safe = name.replace("+", "_")
        np.savez_compressed(K.PREDICTIONS / f"{safe}.npz",
                            logit=res["logit"].astype(np.float64), y=data.y,
                            split_code=split_code(data), graph_id=np.arange(data.y.size),
                            scaffold=data.scaffold)

    # ---------------- contributions & closure -----------------------------
    contributions = {"B2": {"abs_diff": 0.0, "closed": True, "n_terms": 0}}
    for name, X in arms_X.items():
        res = {**results, **c_arms, **joint_arms}[name]
        contributions[name] = closure(X, arms_off[name], res)
    for name in list(joint_arms):
        if name not in contributions:
            contributions[name] = {"evaluated": False}

    payload = {
        "round_id": K.ROUND_ID,
        "phase": "B+C",
        "seconds": float(time.perf_counter() - t0),
        "C_select": C_SELECT, "C_grid": list(C_GRID), "batch": BATCH,
        "size_standardisation": {"mean": size_stats["mean"], "std": size_stats["std"]},
        "b2_replay": b2_check,
        "phaseB": {"selected_rule_ids": [c["rule_id"] for c in selected],
                   "rule_standardisation": {"mean": mu_r, "std": sd_r},
                   "arms": {k: no_logit(v) for k, v in results.items()}},
        "phaseC": {"r64_structures": struct_ids,
                   "arms": {k: no_logit(v) for k, v in c_arms.items()},
                   "joint": joint,
                   "joint_arms": {k: no_logit(v) for k, v in joint_arms.items()}},
        "contributions": contributions,
        "git": K.git_state(),
    }
    K.write_json(K.ANALYSIS / "phaseBC.json", payload)
    np.savez_compressed(K.ANALYSIS / "phaseBC_arrays.npz",
                        size_z=size_z, rule_z=rule_z, u64_act=u64_act, l64_vals=l64_vals,
                        sindex_ids=np.array([c["rule_id"] for c in selected]),
                        r64_ids=np.array([c["rule_id"] for c in r64_rules]))
    print(f"[BC] done in {time.perf_counter() - t0:.1f}s", flush=True)


def no_logit(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("logit", "coef")} | {
        "coef": (d["coef"].tolist() if isinstance(d.get("coef"), np.ndarray) else d.get("coef"))}


def split_code(data: K.Data) -> np.ndarray:
    code = np.zeros(data.y.size, dtype=np.int8)
    for i, k in enumerate(K.SPLITS):
        code[data.rows[k]] = i
    return code


def pair_eligible(pair: dict, sindex: dict, r64: list[dict]) -> bool:
    """Both single conditions are represented in the shared base model U64."""
    rep = {(c["granularity"], c["column"]) for c in r64}
    a, b = sindex[pair["a"]], sindex[pair["b"]]
    return (a["granularity"], a["column"]) in rep and (b["granularity"], b["column"]) in rep


def closure(X: np.ndarray, offset: np.ndarray | None, res: dict) -> dict[str, Any]:
    coef = res["coef"]
    z = np.full(X.shape[0], res["intercept"])
    if offset is not None:
        z = z + offset
    z = z + X @ coef
    d = float(np.max(np.abs(z - res["logit"])))
    return {"abs_diff": d, "closed": bool(d < 1e-6), "n_terms": int(X.shape[1])}


def b2_intercept() -> float:
    """Frozen B2 intercept at C=0.03 (read from the MSDS-v0 Phase-B record)."""
    rec = K.read_json(K.PHASEB_JSON)
    for f in rec["arms"]["B2"]["fits"]:
        if abs(float(f["candidate_C"]) - 0.03) < 1e-12:
            return float(f["coefficients"]["intercept"])
    raise KeyError("B2 C=0.03 fit not found")


def b2_replay(data: K.Data, coef: np.ndarray, saved: dict[str, np.ndarray]) -> dict[str, Any]:
    """Recompute B2's raw logit ``b + [M|C|F] w`` from the stored coefficients.

    B2 is *not* refitted: the coefficient vector and intercept are read from the
    frozen MSDS-v0 store, the design is rebuilt from the frozen artifacts, and the
    result is compared with the stored per-molecule logits.
    """
    import scipy.sparse as sp

    with np.load(K.MORGAN) as zm:
        bits = np.unpackbits(zm["packed"], axis=1)[:, : int(zm["n_bits"])].astype(np.float64)
    a = np.log1p(data.n_atoms.astype(np.float64))
    b = np.log1p(data.n_bonds.astype(np.float64))
    tr = data.rows["train"]
    za = (a - a[tr].mean()) / a[tr].std()
    zb = (b - b[tr].mean()) / b[tr].std()
    M = sp.hstack([sp.csr_matrix(bits), sp.csr_matrix(np.stack([za, zb], axis=1))], format="csr")
    X = sp.hstack([M, data.C.astype(np.float64), data.F.astype(np.float64)], format="csr")
    logit = b2_intercept() + np.asarray(X @ coef).ravel()
    diffs = {k: float(np.max(np.abs(logit[data.rows[k]] - saved[k]))) for k in K.SPLITS}
    return {"max_abs_diff": max(diffs.values()) if np.isfinite(max(diffs.values())) else None, "per_split": diffs,
            "n_coefficients": int(coef.size), "recomputed": True}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(K.MSDS / "code"))
    main()
