"""Phase B runner (ISRD-v1): repaired single-rule models, B2 increments, joint contrasts.

Re-runs only the arms affected by the v0 rule-mapping error:

    R0 (verified replay) / R16 / R64 / R128
    B2-L64 (continuous counts) / B2-U64 (threshold predicates) on the R64 identities
    B2-U+S / B2-U+P / B2-U+N and R64+P when the strict matched-pair support exists

All fitting uses Train only; Valid chooses C in {0.03, 0.3}; Test is DEVELOPMENT_TEST.
"""

from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np

import isrd1_common as K
import isrd1_rules as R
from run_phaseA import key_from_json, key_to_json

C_SELECT = 0.03
C_GRID = (0.03, 0.3)
BATCH = 16
CAPACITIES = (16, 64, 128)
JOINT_MAX = 16
JOINT_MIN = 8
DATA: K.Data | None = None


# ---------------------------------------------------------------------------
# design helpers (identical recipe to v0)
# ---------------------------------------------------------------------------


def standardise(train_vals: np.ndarray, other_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    mu = train_vals.mean(axis=0)
    sd = train_vals.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (train_vals - mu) / sd, (other_vals - mu) / sd, {"mean": mu, "std": sd}


def zscore_apply(vals: np.ndarray, stats: dict) -> np.ndarray:
    return (vals - stats["mean"]) / stats["std"]


def forward_select(data: K.Data, singles: list[dict], cap: int) -> tuple[list[dict], list[dict]]:
    train = data.rows["train"]
    y = data.y[train].astype(np.float64)
    sw = K.class_balanced_weights(y)
    size_tr = data.size_matrix()[train]
    size_z, _, _ = standardise(size_tr, size_tr)
    activations = R.rule_matrix(data, singles, train)
    mu = activations.mean(axis=0)
    sd = activations.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    acts_z = (activations - mu) / sd

    chosen: list[int] = []
    log: list[dict] = []
    fit = K.fit_logistic(size_z, y, sw, C_SELECT)
    p = K.sigmoid(fit["logit"])
    while len(chosen) < cap:
        used = {(singles[i]["granularity"], singles[i]["column"]) for i in chosen}
        resid = sw * (y - p)
        numer = acts_z.T @ resid
        curv = ((acts_z ** 2).T @ (sw * p * (1.0 - p))) + 1.0 / C_SELECT
        score = numer ** 2 / curv
        allowed = np.array([(c["granularity"], c["column"]) not in used and i not in chosen
                            for i, c in enumerate(singles)])
        score = np.where(allowed, score, -np.inf)
        order = [int(i) for i in np.argsort(-score, kind="stable") if np.isfinite(score[i])]
        take: list[int] = []
        for i in order:
            if len(take) >= BATCH:
                break
            key = (singles[i]["granularity"], singles[i]["column"])
            if key in used or any((singles[j]["granularity"], singles[j]["column"]) == key
                                  for j in take):
                continue
            take.append(i)
        if not take:
            break
        chosen.extend(take)
        fit = K.fit_logistic(np.column_stack([size_z, acts_z[:, chosen]]), y, sw, C_SELECT)
        p = K.sigmoid(fit["logit"])
        log.append({"step": len(chosen), "batch": [singles[i]["rule_id"] for i in take],
                    "scores": [float(score[i]) for i in take],
                    "objective": float(fit["objective"])})
        print(f"[B] selected {len(chosen)} rules (objective {fit['objective']:.3f})", flush=True)
    return [singles[i] for i in chosen], log


def fit_and_eval(name: str, X_all: np.ndarray, y: np.ndarray, sw: np.ndarray,
                 offset_all: np.ndarray | None, data: K.Data,
                 penalize_intercept: bool = True) -> dict[str, Any]:
    train = data.rows["train"]
    fits = []
    for C in C_GRID:
        fit = K.fit_logistic(X_all[train], y[train], sw[train], C,
                             offset=None if offset_all is None else offset_all[train],
                             penalize_intercept=penalize_intercept)
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
    return {"arm": fitted["name"], "selected_C": best["C"], "logit": best["logit"],
            "metrics": best["metrics"], "intercept": best["fit"]["intercept"],
            "coef": best["fit"]["coef"], "converged": best["fit"]["converged"],
            "grad_inf": best["fit"]["grad_inf"], "objective": best["fit"]["objective"],
            "C_grid": {str(f["C"]): f["metrics"]["valid"]["roc_auc"] for f in fitted["fits"]}}


def no_logit(d: dict) -> dict:
    out = {k: v for k, v in d.items() if k not in ("logit", "coef")}
    out["coef"] = (d["coef"].tolist() if isinstance(d.get("coef"), np.ndarray) else d.get("coef"))
    return out


def b2_intercept() -> float:
    rec = K.read_json(K.MSDS / "analysis" / "phaseB_linear.json")
    for f in rec["arms"]["B2"]["fits"]:
        if abs(float(f["candidate_C"]) - 0.03) < 1e-12:
            return float(f["coefficients"]["intercept"])
    raise KeyError("B2 C=0.03 fit not found")


def load_b2(data: K.Data) -> tuple[np.ndarray, np.ndarray, dict]:
    coef = np.load(K.B2_COEF)["B2|0.03"].astype(np.float64)
    with np.load(K.B2_PRED) as z:
        saved = {k: z[f"B2|0.03|{k}"].astype(np.float64) for k in K.SPLITS}
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
    z_b2 = np.zeros(data.y.size)
    for k in K.SPLITS:
        z_b2[data.rows[k]] = saved[k]
    check = {"max_abs_diff": max(diffs.values()), "per_split": diffs,
             "n_coefficients": int(coef.size), "design_width": int(X.shape[1]),
             "recomputed": True}
    del X, M, bits
    return z_b2, coef, check


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.perf_counter()
    data = K.Data()
    pa = K.read_json(K.ANALYSIS / "phaseA_rules.json")
    singles = pa["singles"]
    pairs = pa["pairs"]
    all_real_pairs = {(key_from_json(a), key_from_json(b)) for a, b in pa["all_real_path_pairs"]}
    for c in singles:
        c["key"] = (c["granularity"], int(c["column"]), R.f32(c["threshold"]), c["direction"])
    for p in pairs:
        p["key_a"], p["key_b"] = key_from_json(p["key_a"]), key_from_json(p["key_b"])
    train = data.rows["train"]
    y = data.y.astype(np.float64)
    sw = K.class_balanced_weights(y[train])
    sw_all = np.zeros(data.y.size)
    sw_all[train] = sw
    sindex = {c["rule_id"]: c for c in singles}

    # ---------------- B2 frozen replay ------------------------------------
    z_b2, b2_coef, b2_check = load_b2(data)
    print(f"[B] B2 replay max|dz|={b2_check['max_abs_diff']:.3e}", flush=True)
    r0_reuse = {"status": "NOT_CHECKED"}

    # ---------------- Phase B: R0 / R16 / R64 / R128 ----------------------
    size_all = data.size_matrix()
    _, size_z, size_stats = standardise(size_all[train], size_all)
    results: dict[str, dict] = {}
    r0 = summarize(fit_and_eval("R0", size_z, y, sw_all, None, data))
    with np.load(K.V0 / "analysis" / "predictions" / "R0.npz") as zv:
        v0_r0 = zv["logit"].astype(np.float64)
    r0_reuse = {"status": "EXACT_MATCH" if np.max(np.abs(r0["logit"] - v0_r0)) < 1e-9
                else "RECOMPUTED_DIFFERS",
                "max_abs_diff_vs_v0": float(np.max(np.abs(r0["logit"] - v0_r0))),
                "v0_source": K.rel(K.V0 / "analysis" / "predictions" / "R0.npz")}
    results["R0"] = r0
    print(f"[B] R0 valid={r0['metrics']['valid']['roc_auc']:.4f} "
          f"test={r0['metrics']['test']['roc_auc']:.4f} reuse={r0_reuse['status']}", flush=True)

    selected, sel_log = forward_select(data, singles, cap=max(CAPACITIES))
    K.write_json(K.ANALYSIS / "phaseB_selection.json",
                 {"selection_log": sel_log,
                  "selected_rule_ids": [c["rule_id"] for c in selected],
                  "selected_conditions": [R.condition_key_str(c["key"]) for c in selected],
                  "C_select": C_SELECT, "batch": BATCH})
    rule_all = R.rule_matrix(data, selected)
    _, rule_z, rstat = standardise(rule_all[train], rule_all)

    for cap in CAPACITIES:
        n = min(cap, len(selected))
        X = np.column_stack([size_z, rule_z[:, :n]])
        results[f"R{cap}"] = summarize(fit_and_eval(f"R{cap}", X, y, sw_all, None, data))
        m = results[f"R{cap}"]["metrics"]
        print(f"[B] R{cap} valid={m['valid']['roc_auc']:.4f} test={m['test']['roc_auc']:.4f}",
              flush=True)

    r64_rules = selected[:min(64, len(selected))]
    u64_act = R.rule_matrix(data, r64_rules)
    u64_stats = standardise(u64_act[train], u64_act)
    u64_z = u64_stats[1]
    l64_vals = np.column_stack([data.col(c["granularity"], c["column"]).astype(np.float64)
                                for c in r64_rules])
    l64_stats = standardise(l64_vals[train], l64_vals)
    l64_z = l64_stats[1]

    # ---------------- Phase C: B2 increments ------------------------------
    c_arms: dict[str, dict] = {
        "B2": {"arm": "B2", "logit": z_b2, "metrics": K.all_metrics(z_b2, data.y, data.rows),
               "intercept": 0.0, "coef": np.zeros(0), "selected_C": None}}
    for name, X in (("B2-L64", l64_z), ("B2-U64", u64_z)):
        res = summarize(fit_and_eval(name, X, y, sw_all, z_b2, data, penalize_intercept=False))
        c_arms[name] = res
        print(f"[C] {name} valid={res['metrics']['valid']['roc_auc']:.4f} "
              f"test={res['metrics']['test']['roc_auc']:.4f}", flush=True)

    struct_ids = [(c["granularity"], c["column"]) for c in r64_rules]
    assert len(set(struct_ids)) == len(struct_ids), "R64 uses one structure twice"

    # ---------------- A4: exact eligibility + matched random control ------
    pool = R.u64_condition_closure(r64_rules)
    eligible = R.with_resolved_pairs(pairs, pool)
    n_complement = int(sum(1 for p in eligible for side in ("a", "b")
                           if p[f"resolved_{side}"]["origin"] == "U64_COMPLEMENT"))
    print(f"[C] eligible real pairs={len(eligible)} ({n_complement} condition slots are "
          f"exact complements)", flush=True)
    matched, match_info = R.match_random_pairs(data, eligible, pool, all_real_pairs)
    match_stats = match_info["stats"]
    print(f"[C] matched random pairs={match_stats['n_matched_pairs']} "
          f"coverage={match_stats['coverage']:.3f}", flush=True)

    joint: dict[str, Any] = {"n_eligible_real": len(eligible),
                             "n_condition_slots_complement": n_complement,
                             "match": match_info,
                             "n_fresh_single_pool": len(singles) - len(r64_rules),
                             "min_budget": JOINT_MIN, "max_budget": JOINT_MAX}

    def activation_of_key(key: tuple, rows: np.ndarray | None = None) -> np.ndarray:
        return R.rule_activation(data, {"granularity": key[0], "column": key[1],
                                        "threshold": key[2], "direction": key[3]}, rows)

    def cols_from_keys(keys: list[tuple], rows: np.ndarray | None = None) -> np.ndarray:
        if not keys:
            n = data.y.size if rows is None else rows.size
            return np.zeros((n, 0))
        return np.column_stack([activation_of_key(k, rows) for k in keys])

    # common U64 Train residual (frozen B2 + fitted U64 correction)
    base = c_arms["B2-U64"]
    p_base = K.sigmoid(base["logit"][train])
    resid = sw * (y[train] - p_base)
    curv = sw * p_base * (1.0 - p_base)

    real_scored: list[tuple] = []
    if matched:
        by_id = {p["pair_id"]: p for p in eligible}
        keys_real = [(by_id[m["matched_real_pair"]]["key_a"],
                      by_id[m["matched_real_pair"]]["key_b"]) for m in matched]
        keys_rand = [(m["key_a"], m["key_b"]) for m in matched]
        M_real = cols_from_keys([k[0] for k in keys_real], train) * \
            cols_from_keys([k[1] for k in keys_real], train)
        M_rand = cols_from_keys([k[0] for k in keys_rand], train) * \
            cols_from_keys([k[1] for k in keys_rand], train)
        s_real = R.score_rule_columns(M_real, resid, curv, C_SELECT)
        s_rand = R.score_rule_columns(M_rand, resid, curv, C_SELECT)
        real_scored = sorted(zip(0.5 * (s_real + s_rand), range(len(matched)),
                                 s_real, s_rand), key=lambda t: -t[0])
        del M_real, M_rand
    fresh = [c for c in singles if c["rule_id"] not in {r["rule_id"] for r in r64_rules}]
    fresh_M = R.rule_matrix(data, fresh, train) if fresh else np.zeros((train.size, 0))
    fresh_sc = R.score_rule_columns(fresh_M, resid, curv, C_SELECT) if fresh else np.zeros(0)
    single_scored = sorted(zip(fresh_sc, range(len(fresh))), key=lambda t: -t[0])
    del fresh_M

    n_pairs_sel = min(JOINT_MAX, len(real_scored))
    k_avail = min(n_pairs_sel, len(single_scored))
    joint["n_matched_pairs_scored"] = len(real_scored)
    joint["K_available"] = int(k_avail)
    print(f"[C] joint budget K={k_avail} (matched={len(real_scored)} singles={len(fresh)})",
          flush=True)

    joint_arms: dict[str, dict] = {}
    if k_avail >= JOINT_MIN:
        chosen = real_scored[:k_avail]
        by_id = {p["pair_id"]: p for p in eligible}
        real_keys = [(by_id[matched[i]["matched_real_pair"]]["key_a"],
                      by_id[matched[i]["matched_real_pair"]]["key_b"]) for _m, i, _s, _r in chosen]
        rand_keys = [(matched[i]["key_a"], matched[i]["key_b"]) for _m, i, _s, _r in chosen]
        fresh_sel = [fresh[i] for _m, i in single_scored[:k_avail]]
        joint["selected"] = {
            "n_pairs": int(k_avail),
            "real_pairs": [{"pair_id": by_id[matched[i]["matched_real_pair"]]["pair_id"],
                            "condition_a": R.condition_key_str(a),
                            "condition_b": R.condition_key_str(b),
                            "matched_random_pairs": [],
                            "mean_score": float(m)}
                           for (a, b), (m, i, _s, _r) in zip(real_keys, chosen)],
            "random_pairs": [{"pair_id": matched[i]["pair_id"],
                              "condition_a": R.condition_key_str(a),
                              "condition_b": R.condition_key_str(b),
                              "matched_real_pair": matched[i]["matched_real_pair"]}
                             for (a, b), (m, i, _s, _r) in zip(rand_keys, chosen)],
            "fresh_singles": [{"rule_id": c["rule_id"],
                               "condition": R.condition_key_str(c["key"]),
                               "score": float(_s)}
                              for c, (_s, _i) in zip(fresh_sel, single_scored[:k_avail])],
        }
        for rec_all, (a, b), (m, i, _s, _r) in zip(joint["selected"]["real_pairs"], real_keys, chosen):
            rec_all["matched_random_pairs"] = [matched[i]["pair_id"]]

        def pair_block(keys: list[tuple]) -> np.ndarray:
            return cols_from_keys([k[0] for k in keys]) * cols_from_keys([k[1] for k in keys])

        specs = {
            "B2-U+S": R.rule_matrix(data, fresh_sel),
            "B2-U+P": pair_block(real_keys),
            "B2-U+N": pair_block(rand_keys),
        }

        def zblock(extra: np.ndarray) -> np.ndarray:
            mu = extra[train].mean(axis=0)
            sd = np.where(extra[train].std(axis=0) > 0, extra[train].std(axis=0), 1.0)
            return (extra - mu) / sd

        members = {"B2-U+S": [c["rule_id"] for c in fresh_sel],
                   "B2-U+P": [r["pair_id"] for r in joint["selected"]["real_pairs"]],
                   "B2-U+N": [r["pair_id"] for r in joint["selected"]["random_pairs"]]}
        extras = {}
        for name, extra in specs.items():
            extras[name] = zblock(extra)
            X = np.column_stack([u64_z, extras[name]])
            res = summarize(fit_and_eval(name, X, y, sw_all, z_b2, data, penalize_intercept=False))
            res["members"] = members[name]
            joint_arms[name] = res
            print(f"[C] {name} valid={res['metrics']['valid']['roc_auc']:.4f} "
                  f"test={res['metrics']['test']['roc_auc']:.4f}", flush=True)
        X_rp = np.column_stack([size_z, u64_z, extras["B2-U+P"]])
        rp = summarize(fit_and_eval("R64+P", X_rp, y, sw_all, None, data))
        rp["members"] = members["B2-U+P"]
        joint_arms["R64+P"] = rp
        print(f"[B] R64+P valid={rp['metrics']['valid']['roc_auc']:.4f} "
              f"test={rp['metrics']['test']['roc_auc']:.4f}", flush=True)
        joint["status"] = "EXECUTED"
    else:
        joint["status"] = "INSUFFICIENT_MATCHED_PAIR_SUPPORT"
        print(f"[C] JOINT SKIPPED: {joint['status']}", flush=True)

    # ---------------- save predictions ------------------------------------
    K.PREDICTIONS.mkdir(parents=True, exist_ok=True)
    all_arms = {**results, **c_arms, **joint_arms}
    for name, res in all_arms.items():
        safe = name.replace("+", "_")
        np.savez_compressed(K.PREDICTIONS / f"{safe}.npz",
                            logit=np.asarray(res["logit"], dtype=np.float64), y=data.y,
                            split_code=data.split_code(), graph_id=np.arange(data.y.size),
                            scaffold=data.scaffold)

    # ---------------- closure ---------------------------------------------
    designs = {"R0": size_z}
    for cap in CAPACITIES:
        designs[f"R{cap}"] = np.column_stack([size_z, rule_z[:, :min(cap, len(selected))]])
    designs["B2-L64"] = l64_z
    designs["B2-U64"] = u64_z
    contributions = {}
    for name, X in designs.items():
        res = all_arms[name]
        off = z_b2 if name.startswith("B2") else None
        z = np.full(data.y.size, res["intercept"]) + (0 if off is None else off) + X @ res["coef"]
        d = float(np.max(np.abs(z - res["logit"])))
        contributions[name] = {"abs_diff": d, "closed": bool(d < 1e-6), "n_terms": int(X.shape[1])}

    payload = {
        "round_id": K.ROUND_ID, "phase": "B+C (repaired rules)",
        "seconds": float(time.perf_counter() - t0),
        "C_select": C_SELECT, "C_grid": list(C_GRID), "batch": BATCH,
        "size_standardisation": {"mean": size_stats["mean"].tolist(),
                                 "std": size_stats["std"].tolist()},
        "r0_reuse_check": r0_reuse,
        "b2_replay": b2_check,
        "phaseB": {"selected_rule_ids": [c["rule_id"] for c in selected],
                   "selected_conditions": [R.condition_key_str(c["key"]) for c in selected],
                   "rule_standardisation": {"mean": rstat["mean"].tolist(),
                                            "std": rstat["std"].tolist()},
                   "arms": {k: no_logit(v) for k, v in results.items()}},
        "phaseC": {"r64_structures": struct_ids,
                   "r64_rule_ids": [c["rule_id"] for c in r64_rules],
                   "r64_thresholds": [c["threshold"] for c in r64_rules],
                   "r64_directions": [c["direction"] for c in r64_rules],
                   "u64_standardisation": {"mean": u64_stats[2]["mean"].tolist(),
                                           "std": u64_stats[2]["std"].tolist()},
                   "l64_standardisation": {"mean": l64_stats[2]["mean"].tolist(),
                                           "std": l64_stats[2]["std"].tolist()},
                   "arms": {k: no_logit(v) for k, v in c_arms.items()},
                   "joint": joint,
                   "joint_arms": {k: no_logit(v) for k, v in joint_arms.items()}},
        "contributions": contributions,
        "git": K.git_state(),
    }
    K.write_json(K.ANALYSIS / "phaseBC.json", payload)
    np.savez_compressed(K.ANALYSIS / "phaseBC_arrays.npz",
                        size_z=size_z, rule_z=rule_z, u64_act=u64_act, l64_vals=l64_vals,
                        all_rule_ids=np.array([c["rule_id"] for c in selected]),
                        r64_ids=np.array([c["rule_id"] for c in r64_rules]),
                        u64_rule_ids=np.array([c["rule_id"] for c in r64_rules]))
    print(f"[BC] done in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
