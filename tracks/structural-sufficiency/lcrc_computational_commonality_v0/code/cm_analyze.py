"""Part A analysis + verdict/report tables (CPU, official macro AP only).

``uv run python code/cm_analyze.py all``
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cm_common as C

AN = C.ANALYSIS
BOOT = 2000
MIN_SUPPORT = 20
PRACTICAL = 0.005
N_FOLDS = 3
WGRID = (0.0, 0.25, 0.5, 0.75, 1.0)

SAME_ARCH = (("W2_s0", "W2_s1"), ("FIXED1_s0", "FIXED1_s1"), ("DYNAMIC_s0", "DYNAMIC_s1"))
CROSS_PRIMARY = (("W2_s0", "FIXED1_s0"), ("W2_s0", "DYNAMIC_s0"))
CROSS_AUX = (("FIXED1_s0", "DYNAMIC_s0"), ("W2_s0", "S3_s0"), ("W2_s0", "RC_REAL_s0"))
CROSS_SEED1 = (("W2_s1", "FIXED1_s1"), ("W2_s1", "DYNAMIC_s1"), ("FIXED1_s1", "DYNAMIC_s1"))


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------


class Scores:
    def __init__(self) -> None:
        self.rank: dict[str, dict[str, np.ndarray]] = {"valid": {}, "test": {}}
        self.raw: dict[str, dict[str, np.ndarray]] = {"valid": {}, "test": {}}
        self.y = {"valid": C.y_valid(), "test": C.y_test()}
        self.base_ap: dict[str, dict[str, Any]] = {"valid": {}, "test": {}}
        for split in ("valid", "test"):
            for label in C.MODELS:
                try:
                    logits = C.load_logits(label, split)
                except RuntimeError:
                    continue
                self.raw[split][label] = logits
                self.rank[split][label] = C.rank_score(logits)
                self.base_ap[split][label] = C.macro_ap(self.y[split], C.score_logits(logits))

    def labels(self, split: str) -> list[str]:
        return sorted(self.rank[split])


def combo(parts: Sequence[np.ndarray], weights: Sequence[float] | None = None) -> np.ndarray:
    w = np.asarray(weights if weights is not None else [1.0] * len(parts), dtype=np.float64)
    w = w / w.sum()
    out = np.zeros_like(parts[0], dtype=np.float64)
    for wi, p in zip(w, parts):
        out += wi * np.asarray(p, dtype=np.float64)
    return out.astype(np.float32)


def ap(y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    return C.macro_ap(y, score)


# ---------------------------------------------------------------------------
# scaffold folds
# ---------------------------------------------------------------------------


def scaffold_folds(rows: np.ndarray, task: Any, *, n_folds: int = N_FOLDS) -> dict[str, Any]:
    """Bemis-Murcko scaffold folds balanced by scaffold size, fixed hash order."""
    import hashlib

    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    smiles = {}
    path = C.REPO / "tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1/cache/pcba_v1/smiles.jsonl"
    with open(path) as fh:
        for i, line in enumerate(fh):
            smiles[i] = json.loads(line)
    groups: dict[str, list[int]] = {}
    for r in rows.tolist():
        smi = smiles.get(int(r))
        if smi is None:
            scaf = f"__missing__{r}"
        else:
            mol = Chem.MolFromSmiles(smi)
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol is not None else f"__bad__{r}"
        groups.setdefault(scaf, []).append(int(r))
    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), hashlib.sha256(kv[0].encode()).hexdigest()))
    sizes = [0] * n_folds
    fold_of = np.zeros(rows.size, dtype=np.int64)
    index = {int(r): i for i, r in enumerate(rows.tolist())}
    for _scaf, members in order:
        target = int(np.argmin(sizes))
        sizes[target] += len(members)
        for r in members:
            fold_of[index[r]] = target
    return {"fold_of": fold_of, "n_scaffolds": len(groups), "fold_sizes": sizes}


# ---------------------------------------------------------------------------
# task weights
# ---------------------------------------------------------------------------


def task_ap_matrix(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.metrics import average_precision_score

    n_tasks = y.shape[1]
    apv = np.full(n_tasks, np.nan)
    pos = np.zeros(n_tasks, dtype=np.int64)
    obs = np.zeros(n_tasks, dtype=np.int64)
    for t in range(n_tasks):
        yy = y[:, t]
        m = np.isfinite(yy)
        obs[t] = int(m.sum())
        pos[t] = int((yy[m] == 1).sum())
        if pos[t] and (obs[t] - pos[t]):
            apv[t] = float(average_precision_score(yy[m], score[m, t]))
    return apv, pos, obs


def macro_of(per_task: np.ndarray) -> float:
    ok = np.isfinite(per_task)
    return float(per_task[ok].mean())


def pair_weights(r1: np.ndarray, r2: np.ndarray, w: float) -> np.ndarray:
    return ((1.0 - w) * r1.astype(np.float64) + w * r2.astype(np.float64)).astype(np.float32)


def fit_task_weights(y: np.ndarray, r1: np.ndarray, r2: np.ndarray, w_global: float,
                     *, min_support: int = MIN_SUPPORT, alpha_scale: float = 100.0) -> dict[str, Any]:
    """Per-task weights with tie-break, low-support fallback and support shrinkage."""
    grid_ap = {}
    for w in WGRID:
        grid_ap[w], pos, obs = task_ap_matrix(y, pair_weights(r1, r2, w))
    n_tasks = y.shape[1]
    w_task = np.full(n_tasks, np.nan)
    w_final = np.full(n_tasks, w_global)
    fallback = []
    for t in range(n_tasks):
        pos_t = int(pos[t])
        neg_t = int(obs[t] - pos[t])
        if pos_t < min_support or neg_t < min_support:
            fallback.append(t)
            continue
        vals = np.asarray([grid_ap[w][t] for w in WGRID])
        if not np.isfinite(vals).any():
            fallback.append(t)
            continue
        best = np.nanmax(vals)
        cand = [w for w, v in zip(WGRID, vals) if np.isfinite(v) and abs(v - best) <= 1e-12]
        cand.sort(key=lambda w: (abs(w - w_global), w))
        w_task[t] = cand[0]
        alpha = pos_t / (pos_t + alpha_scale)
        w_final[t] = alpha * cand[0] + (1 - alpha) * w_global
    return {"w_task": w_task, "w_final": w_final, "n_fallback": len(fallback),
            "fallback_tasks": fallback, "grid_ap": {str(k): v for k, v in grid_ap.items()},
            "pos": pos, "obs": obs}


def apply_task_weights(r1: np.ndarray, r2: np.ndarray, w_final: np.ndarray) -> np.ndarray:
    w = np.asarray(w_final, dtype=np.float64)[None, :]
    return ((1.0 - w) * r1.astype(np.float64) + w * r2.astype(np.float64)).astype(np.float32)


def global_weight(y: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> dict[str, Any]:
    rows = []
    for w in WGRID:
        rep = ap(y, pair_weights(r1, r2, w))
        rows.append({"w": float(w), "macro_ap": rep["macro_ap"]})
    best = max(rows, key=lambda r: (r["macro_ap"], -abs(r["w"] - 0.5)))
    return {"best_w": float(best["w"]), "grid": rows}


# ---------------------------------------------------------------------------
# main analysis
# ---------------------------------------------------------------------------


def run_part_a() -> dict[str, Any]:
    t0 = time.perf_counter()
    S = Scores()
    labels = S.labels("valid")
    y = S.y
    out: dict[str, Any] = {"created_utc": C.utcnow(), "models": {}, "ensembles": {}}

    # ---- single models ---------------------------------------------------
    for label in labels:
        rec = {
            "arch": C.MODELS[label]["arch"],
            "seed": int(C.MODELS[label]["seed"]),
            "parameters": int(C.MODELS[label]["parameters"]),
            "horizon_epochs": int(C.MODELS[label]["horizon_epochs"]),
            "best_epoch": int(C.MODELS[label]["best_epoch"]),
            "valid_macro_ap": S.base_ap["valid"][label]["macro_ap"],
            "test_macro_ap": S.base_ap["test"][label]["macro_ap"],
            "valid_recorded": C.MODELS[label].get("valid_ap_recorded"),
            "test_recorded": C.MODELS[label].get("test_ap_recorded"),
            "n_tasks_scored_valid": S.base_ap["valid"][label]["n_tasks_scored"],
            "n_tasks_scored_test": S.base_ap["test"][label]["n_tasks_scored"],
            "recipe_note": C.MODELS[label].get("recipe_note"),
            "valid_per_task_ap": S.base_ap["valid"][label]["per_task_ap"],
            "test_per_task_ap": S.base_ap["test"][label]["per_task_ap"],
            "train_positive_support": None,
        }
        out["models"][label] = rec
    best_single = max(labels, key=lambda l: out["models"][l]["valid_macro_ap"])
    out["best_single_by_valid"] = best_single

    # ---- ensembles -------------------------------------------------------
    def add(name: str, members: Sequence[str], kind: str) -> None:
        if any(m not in labels for m in members):
            return
        entry = {"kind": kind, "members": list(members),
                 "parameters_total": int(sum(C.MODELS[m]["parameters"] for m in members))}
        for split in ("valid", "test"):
            parts = [S.rank[split][m] for m in members]
            score = combo(parts)
            rep = ap(y[split], score)
            raw = combo([S.raw[split][m] for m in members])
            rep_raw = ap(y[split], C.score_logits(raw))
            entry[split] = {
                "macro_ap": rep["macro_ap"], "macro_ap_raw_logit_mean": rep_raw["macro_ap"],
                "per_task_ap": rep["per_task_ap"],
                "best_member_ap": max(S.base_ap[split][m]["macro_ap"] for m in members),
            }
            entry[split]["delta_vs_best_member"] = entry[split]["macro_ap"] - entry[split]["best_member_ap"]
        out["ensembles"][name] = entry

    for a, b in SAME_ARCH:
        add(f"same:{a}+{b}", (a, b), "same_arch_seed")
    for a, b in CROSS_PRIMARY:
        add(f"cross:{a}+{b}", (a, b), "cross_arch_primary")
    for a, b in CROSS_AUX:
        add(f"cross:{a}+{b}", (a, b), "cross_arch_auxiliary")
    for a, b in CROSS_SEED1:
        add(f"cross:{a}+{b}", (a, b), "cross_arch_seed1")
    add("CORE", ("W2_s0", "FIXED1_s0", "DYNAMIC_s0"), "core_multimodel")
    extended = tuple(m for m in C.EXTENDED_ARMS if m in labels)
    add("EXTENDED", extended, "extended_multimodel")

    # homologous reference for each cross pair: mean gain of the same-arch combos
    same_gains = [v["valid"]["delta_vs_best_member"] for k, v in out["ensembles"].items() if v["kind"] == "same_arch_seed"]
    out["homologous_reference"] = {
        "mean_same_arch_gain_valid": float(np.mean(same_gains)),
        "mean_same_arch_gain_test": float(np.mean([v["test"]["delta_vs_best_member"] for k, v in out["ensembles"].items() if v["kind"] == "same_arch_seed"])),
        "same_arch_gains": {k: {"valid": v["valid"]["delta_vs_best_member"], "test": v["test"]["delta_vs_best_member"]}
                            for k, v in out["ensembles"].items() if v["kind"] == "same_arch_seed"},
    }

    # ---- task weights ----------------------------------------------------
    task = C.DC.SD.load_task()
    rows_valid = np.asarray(task.valid_rows, dtype=np.int64)
    folds = scaffold_folds(rows_valid, task)
    fold_of = folds["fold_of"]
    weight_out: dict[str, Any] = {"created_utc": C.utcnow(), "folds": {k: v for k, v in folds.items() if k != "fold_of"},
                                  "pairs": {}, "w_grid": list(WGRID), "min_support": MIN_SUPPORT,
                                  "shrinkage": "alpha = n_pos / (n_pos + 100)"}
    for a, b in CROSS_PRIMARY + CROSS_SEED1:
        if a not in labels or b not in labels:
            continue
        key = f"{a}+{b}"
        r1v, r2v = S.rank["valid"][a], S.rank["valid"][b]
        r1t, r2t = S.rank["test"][a], S.rank["test"][b]
        gw = global_weight(y["valid"], r1v, r2v)
        w_global = gw["best_w"]
        equal = ap(y["valid"], pair_weights(r1v, r2v, 0.5))
        glob = ap(y["valid"], pair_weights(r1v, r2v, w_global))
        tk = fit_task_weights(y["valid"], r1v, r2v, w_global)
        task_full = ap(y["valid"], apply_task_weights(r1v, r2v, tk["w_final"]))
        # ---- cross-fitted OOF (weights fitted on the other two folds) ----
        oof = np.zeros_like(r1v, dtype=np.float64)
        fold_records = []
        for f in range(N_FOLDS):
            tr = fold_of != f
            te = ~tr
            gw_f = global_weight(y["valid"][tr], r1v[tr], r2v[tr])
            tk_f = fit_task_weights(y["valid"][tr], r1v[tr], r2v[tr], gw_f["best_w"])
            oof[te] = (1.0 - tk_f["w_final"][None, :]) * r1v[te] + tk_f["w_final"][None, :] * r2v[te]
            fold_records.append({"fold": f, "w_global": gw_f["best_w"], "n_fallback": tk_f["n_fallback"],
                                 "n_rows": int(te.sum())})
        oof_ap = ap(y["valid"], oof.astype(np.float32))
        # ---- Test with full-Valid weights -------------------------------
        test_glob = ap(y["test"], pair_weights(r1t, r2t, w_global))
        test_equal = ap(y["test"], pair_weights(r1t, r2t, 0.5))
        test_task = ap(y["test"], apply_task_weights(r1t, r2t, tk["w_final"]))
        # ---- oracles -----------------------------------------------------
        pa_v = np.stack([np.asarray(S.base_ap["valid"][a]["per_task_ap"], dtype=float),
                         np.asarray(S.base_ap["valid"][b]["per_task_ap"], dtype=float)], axis=1)
        pa_t = np.stack([np.asarray(S.base_ap["test"][a]["per_task_ap"], dtype=float),
                         np.asarray(S.base_ap["test"][b]["per_task_ap"], dtype=float)], axis=1)
        best_v = np.where(np.nan_to_num(pa_v[:, 1], nan=-1) > np.nan_to_num(pa_v[:, 0], nan=-1), 1, 0)
        picked = pa_t[np.arange(len(best_v)), best_v]
        oracle_valid_pick = float(np.nanmean(np.where(np.isfinite(picked), picked, np.nan)))
        best_t = np.where(np.nan_to_num(pa_t[:, 1], nan=-1) > np.nan_to_num(pa_t[:, 0], nan=-1), 1, 0)
        picked_t = pa_t[np.arange(len(best_t)), best_t]
        oracle_test_pick = float(np.nanmean(np.where(np.isfinite(picked_t), picked_t, np.nan)))
        grid_test = {str(w): ap(y["test"], pair_weights(r1t, r2t, w))["macro_ap"] for w in WGRID}
        oracle_w_test = max(grid_test.items(), key=lambda kv: kv[1])
        grid_test_per_task = {w: task_ap_matrix(y["test"], pair_weights(r1t, r2t, w))[0] for w in WGRID}
        Gt = np.stack([grid_test_per_task[w] for w in WGRID], axis=1)
        oracle_per_task_w = float(np.nanmean(Gt.max(axis=1)))
        # ---- complementarity per task -----------------------------------
        per_task_combo = np.asarray(task_full["per_task_ap"], dtype=float)
        glob_pt = np.asarray(glob["per_task_ap"], dtype=float)
        equal_pt = np.asarray(equal["per_task_ap"], dtype=float)
        test_task_pt = np.asarray(test_task["per_task_ap"], dtype=float)
        test_glob_pt = np.asarray(test_glob["per_task_ap"], dtype=float)
        test_equal_pt = np.asarray(test_equal["per_task_ap"], dtype=float)
        pos_v = tk["pos"]
        obs_v = tk["obs"]
        pos_t = np.asarray([int(np.sum(y["test"][np.isfinite(y["test"][:, t]), t] == 1)) for t in range(C.N_TASKS)])
        obs_t = np.asarray([int(np.isfinite(y["test"][:, t]).sum()) for t in range(C.N_TASKS)])
        rows = []
        for t in range(C.N_TASKS):
            rows.append({
                "task": t,
                "valid_positive": int(pos_v[t]), "valid_observed": int(obs_v[t]),
                "test_positive": int(pos_t[t]), "test_observed": int(obs_t[t]),
                f"valid_ap_{a}": pa_v[t, 0], f"valid_ap_{b}": pa_v[t, 1],
                f"test_ap_{a}": pa_t[t, 0], f"test_ap_{b}": pa_t[t, 1],
                "valid_ap_combo_task_weight": per_task_combo[t] if np.isfinite(per_task_combo[t]) else None,
                "valid_ap_combo_global": glob_pt[t],
                "valid_ap_combo_equal": equal_pt[t],
                "test_ap_combo_task_weight": test_task_pt[t],
                "test_ap_combo_global": test_glob_pt[t],
                "test_ap_combo_equal": test_equal_pt[t],
                "valid_delta_vs_best_member": float(glob_pt[t] - np.nanmax(pa_v[t]))
                if np.isfinite(glob_pt[t]) else None,
                "test_delta_vs_best_member": float(test_glob_pt[t] - np.nanmax(pa_t[t]))
                if np.isfinite(test_glob_pt[t]) else None,
                "w_global": w_global, "w_task": float(tk["w_task"][t]) if np.isfinite(tk["w_task"][t]) else None,
                "w_final": float(tk["w_final"][t]),
                "fallback": bool(t in tk["fallback_tasks"]),
            })
        C.save_csv(AN / f"task_complementarity_{a}__{b}.csv", rows)
        # ---- statistics -------------------------------------------------
        common = np.array([(r["valid_positive"] >= MIN_SUPPORT and r["test_positive"] >= MIN_SUPPORT) for r in rows])
        dv = np.array([r["valid_delta_vs_best_member"] if r["valid_delta_vs_best_member"] is not None else np.nan for r in rows])
        dt = np.array([r["test_delta_vs_best_member"] if r["test_delta_vs_best_member"] is not None else np.nan for r in rows])
        both = np.isfinite(dv) & np.isfinite(dt)
        rng = np.random.default_rng(0)
        boots = []
        idx_ok = np.flatnonzero(both)
        for _ in range(BOOT):
            pick = rng.choice(idx_ok, size=idx_ok.size, replace=True)
            boots.append(float(dt[pick].mean()))
        sens = {}
        for k in (1, 3, 5):
            order = np.argsort(-np.abs(np.nan_to_num(dt, nan=0.0)))[:k]
            keep = np.ones(len(rows), dtype=bool)
            keep[order] = False
            sens[f"drop_top{k}_test_delta"] = float(np.nanmean(dt[keep]))
            sens[f"drop_top{k}_tasks"] = order.tolist()
        weight_out["pairs"][key] = {
            "global": {"valid": glob["macro_ap"], "test": test_glob["macro_ap"], "w": w_global, "grid_valid": gw["grid"]},
            "equal": {"valid": equal["macro_ap"], "test": test_equal["macro_ap"]},
            "task": {"valid_in_sample": task_full["macro_ap"], "valid_oof": oof_ap["macro_ap"],
                     "test": test_task["macro_ap"], "n_fallback": tk["n_fallback"],
                     "folds": fold_records},
            "members": {"valid": {a: S.base_ap["valid"][a]["macro_ap"], b: S.base_ap["valid"][b]["macro_ap"]},
                        "test": {a: S.base_ap["test"][a]["macro_ap"], b: S.base_ap["test"][b]["macro_ap"]}},
            "oracle": {
                "valid_best_single_model_to_test": oracle_valid_pick,
                "test_best_single_model_POST_HOC": oracle_test_pick,
                "test_best_discrete_weight_POST_HOC": {"w": float(oracle_w_test[0]), "macro_ap": float(oracle_w_test[1])},
                "test_best_weight_per_task_POST_HOC": oracle_per_task_w,
            },
            "direction_agreement": {
                "share_same_sign": float(np.mean(np.sign(dv[both]) == np.sign(dt[both]))),
                "n_tasks": int(both.sum()),
                "spearman_valid_test_delta": float(np.corrcoef(np.argsort(np.argsort(dv[both])), np.argsort(np.argsort(dt[both])))[0, 1]),
            },
            "high_support_subset": {
                "n_tasks": int(common.sum()),
                "valid_delta_mean": float(np.nanmean(dv[common])) if common.any() else None,
                "test_delta_mean": float(np.nanmean(dt[common])) if common.any() else None,
            },
            "bootstrap_task": {"n": BOOT, "mean": float(np.mean(boots)),
                               "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                               "share_positive": float(np.mean(np.asarray(boots) > 0))},
            "drop_topk_sensitivity": sens,
            "test_ap_distances": {"to_0.28": float(0.28 - test_task["macro_ap"]), "to_0.29": float(0.29 - test_task["macro_ap"]),
                                  "global_to_0.28": float(0.28 - test_glob["macro_ap"]), "global_to_0.29": float(0.29 - test_glob["macro_ap"])},
        }
    out["task_weights"] = weight_out
    out["seconds"] = float(time.perf_counter() - t0)
    C.write_json(AN / "part_a_raw.json", out)
    return out


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("stage", choices=["parta"])
    args = ap_.parse_args()
    if args.stage == "parta":
        rep = run_part_a()
        print(json.dumps({"best_single": rep["best_single_by_valid"],
                          "ensembles": {k: {s: round(v[s]["macro_ap"], 5) for s in ("valid", "test")}
                                        for k, v in rep["ensembles"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
