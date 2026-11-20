"""Projection certificate and high-precision projection for the frozen K=16 dictionaries.

For every non-prototype fragment of the three compressed ``tau`` classes this
script

1. certifies the frozen 64-step Frank-Wolfe solution with the convex duality gap
   ``g(alpha).alpha - min(g(alpha))`` (a valid upper bound on ``f(alpha)-f*``);
2. solves the unchanged simplex projection in Float64 with SciPy SLSQP from the
   old FW64 alpha and from the uniform alpha (exact analytic gradient, one
   ``sum=1`` equality constraint, non-negative bounds, ``ftol=1e-12``,
   ``maxiter=2000``);
3. applies the frozen pass rule and the frozen main-solution rule, keeps FW64 for
   ``NUMERICALLY_UNRESOLVED`` rows in the primary version, and reports an
   explicitly uncertified sensitivity version that uses the best feasible SLSQP
   start for those rows;
4. recomputes full ``F`` for all three dictionaries under FW64,
   ``HP_CERTIFIED`` and ``HP_SLSQP_ALL`` (Train/Val/Test RMSE, MAE, fidelity);
5. measures whether two qualified starts with a panel-tied objective produce
   different actual Val/Test readouts (``PANEL_UNDERDETERMINES_PREDICTION``).

No training, no optimiser update, no change of objective, panel, prototypes, D0
or split.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import ec_common as EC  # noqa: E402
import rd_common as RC  # noqa: E402
import rd_dictionary as RD  # noqa: E402
import rd_dict_run as RDR  # noqa: E402

TOL_REL = 1e-8
SIMPLEX_ABS = 1e-10
SUM_ABS = 1e-9
MATERIAL_PRED_DIFF = 1e-6


def _problem(P: np.ndarray, y: np.ndarray):
    """``f(alpha)=mean((P.T a-y)^2)`` and its exact gradient."""

    def fun(a: np.ndarray) -> float:
        r = P.T @ a - y
        return float(np.mean(r * r))

    def jac(a: np.ndarray) -> np.ndarray:
        r = P.T @ a - y
        return 2.0 * P @ r / P.shape[1]

    return fun, jac


def certify(P: np.ndarray, y: np.ndarray, alpha: np.ndarray) -> dict[str, Any]:
    """f, duality gap and simplex diagnostics of one candidate alpha."""
    fun, jac = _problem(P, y)
    a = np.asarray(alpha, dtype=np.float64)
    g = jac(a)
    gap = float(g @ a - np.min(g))
    return {
        "f": float(fun(a)),
        "gap": gap,
        "min_alpha": float(a.min()),
        "sum_err": float(abs(a.sum() - 1.0)),
        "tol": float(TOL_REL * max(1.0, float(np.mean(y * y)))),
    }


def qualifies(cert: Mapping[str, Any]) -> bool:
    return bool(cert["gap"] <= cert["tol"]
                and cert["min_alpha"] >= -SIMPLEX_ABS
                and cert["sum_err"] <= SUM_ABS)


def solve_row(P: np.ndarray, y: np.ndarray, a0: np.ndarray,
              settings: Mapping[str, Any]) -> dict[str, Any]:
    """One SLSQP solve; returns the certificate and solver metadata."""
    from scipy.optimize import minimize

    fun, jac = _problem(P, y)
    K = P.shape[0]
    cons = [{
        "type": "eq",
        "fun": lambda a: float(np.sum(a) - 1.0),
        "jac": lambda a: np.ones_like(a),
    }]
    x0 = np.asarray(a0, dtype=np.float64).copy()
    result = minimize(
        fun, x0, jac=jac, method=str(settings["method"]),
        bounds=[(0.0, 1.0)] * K, constraints=cons,
        options={"ftol": float(settings["ftol"]), "maxiter": int(settings["maxiter"])},
    )
    a = np.asarray(result.x, dtype=np.float64)
    cert = certify(P, y, a)
    cert.update({
        "success": bool(result.success),
        "status": int(result.status),
        "nit": int(getattr(result, "nit", 0)),
        "qualified": qualifies(cert),
        "alpha": a,
        "message": str(result.message),
    })
    return cert


def _pick(sols: Sequence[Mapping[str, Any]], tol: float) -> Mapping[str, Any]:
    """Smaller f; if the f difference is within the row tolerance, prefer start 0 (FW)."""
    best = min(sols, key=lambda s: s["f"])
    if len(sols) > 1 and sols[0]["f"] - best["f"] <= tol:
        return sols[0]
    return best


def _feasible(sol: Mapping[str, Any]) -> bool:
    return bool(sol["min_alpha"] >= -SIMPLEX_ABS and sol["sum_err"] <= SUM_ABS)


def solve_class(frozen, inventory, spec: Mapping[str, Any], tau, frags: Sequence[int],
                alpha_fw: Mapping[int, np.ndarray], targets: np.ndarray,
                panels: Sequence[int], settings: Mapping[str, Any],
                dict_name: str) -> tuple[list[dict[str, Any]], dict[int, np.ndarray], dict[int, np.ndarray]]:
    P = RD.response_matrix(frozen, spec["prototypes"][tau], spec["d0"][tau], panels)
    proto_set = {int(p) for p in spec["prototypes"][tau]}
    rows: list[dict[str, Any]] = []
    alpha_cert: dict[int, np.ndarray] = {}
    alpha_all: dict[int, np.ndarray] = {}
    for j, frag in enumerate(frags):
        frag = int(frag)
        if frag in proto_set:
            alpha_cert[frag] = np.asarray(alpha_fw[frag], dtype=np.float64)
            alpha_all[frag] = np.asarray(alpha_fw[frag], dtype=np.float64)
            continue
        y = np.asarray(targets[j], dtype=np.float64)
        a_fw = np.asarray(alpha_fw[frag], dtype=np.float64)
        fw = certify(P, y, a_fw)
        fw_ok = qualifies(fw)
        s_fw = solve_row(P, y, a_fw, settings)
        s_uni = solve_row(P, y, np.full(P.shape[0], 1.0 / P.shape[0]), settings)
        sols = [s_fw, s_uni]
        qualified = [s for s in sols if s["qualified"]]
        feasible = [s for s in sols if _feasible(s)]
        if qualified:
            main = _pick(qualified, fw["tol"])
            main_start = "fw" if main is s_fw else "uniform"
            alpha_cert[frag] = np.asarray(main["alpha"], dtype=np.float64)
            status = ("CERTIFIED_TIE_FW" if len(qualified) == 2
                      and abs(s_fw["f"] - s_uni["f"]) <= fw["tol"]
                      else f"CERTIFIED_{main_start.upper()}")
        else:
            alpha_cert[frag] = a_fw.copy()
            main = None
            main_start = "fw64_fallback"
            status = "NUMERICALLY_UNRESOLVED"
        if feasible:
            best_all = _pick(feasible, fw["tol"])
            alpha_all[frag] = np.asarray(best_all["alpha"], dtype=np.float64)
            all_start = "fw" if best_all is s_fw else "uniform"
        else:
            alpha_all[frag] = a_fw.copy()
            all_start = "fw64_fallback"
        both_qualified = bool(s_fw["qualified"] and s_uni["qualified"])
        alpha_diff = float(np.max(np.abs(np.asarray(s_fw["alpha"]) - np.asarray(s_uni["alpha"]))))
        f_diff = float(abs(s_fw["f"] - s_uni["f"]))
        ambiguous = bool(both_qualified and f_diff <= fw["tol"]
                         and alpha_diff > 1e-8 and _feasible(s_fw) and _feasible(s_uni))
        rows.append({
            "dictionary": dict_name,
            "tau": RD.tau_label(tau),
            "fragment_id": frag,
            "fragment_key_prefix": fragment_key(frozen, frag)[:16],
            "n_atoms": int(frozen.n_atoms[frag]),
            "is_train_word": bool(frag in inventory.word_mol_freq),
            "n_panel": int(P.shape[1]),
            "n_prototypes": int(P.shape[0]),
            "y2_mean": float(np.mean(y * y)),
            "tol": float(fw["tol"]),
            "f_fw64": float(fw["f"]),
            "gap_fw64": float(fw["gap"]),
            "min_alpha_fw64": float(fw["min_alpha"]),
            "sum_err_fw64": float(fw["sum_err"]),
            "pass_fw64": bool(fw_ok),
            "f_slsqp_fw": float(s_fw["f"]),
            "gap_slsqp_fw": float(s_fw["gap"]),
            "min_alpha_slsqp_fw": float(s_fw["min_alpha"]),
            "sum_err_slsqp_fw": float(s_fw["sum_err"]),
            "pass_slsqp_fw": bool(s_fw["qualified"]),
            "slsqp_fw_status": int(s_fw["status"]),
            "slsqp_fw_nit": int(s_fw["nit"]),
            "f_slsqp_uniform": float(s_uni["f"]),
            "gap_slsqp_uniform": float(s_uni["gap"]),
            "min_alpha_slsqp_uniform": float(s_uni["min_alpha"]),
            "sum_err_slsqp_uniform": float(s_uni["sum_err"]),
            "pass_slsqp_uniform": bool(s_uni["qualified"]),
            "slsqp_uniform_status": int(s_uni["status"]),
            "slsqp_uniform_nit": int(s_uni["nit"]),
            "f_diff_starts": f_diff,
            "alpha_inf_diff_starts": alpha_diff,
            "both_qualified": both_qualified,
            "ambiguous_panel_solution": ambiguous,
            "main_start": main_start,
            "f_main": float(main["f"]) if main is not None else float(fw["f"]),
            "f_improvement_fw_to_main": float(fw["f"] - (main["f"] if main is not None else fw["f"])),
            "all_version_start": all_start,
            "status": status,
            "certificate_label": ("PROJECTION_CERTIFIED" if status.startswith("CERTIFIED")
                                  else "NUMERICALLY_UNRESOLVED"),
        })
    return rows, alpha_cert, alpha_all


# ---------------------------------------------------------------------------
# small helper for fragment metadata stored outside ``Frozen``
# ---------------------------------------------------------------------------


def _attach_inventory(frozen, inventory, fragment_payload):
    frozen.n_atoms = np.asarray(fragment_payload["frag_n_atoms"], dtype=np.int64)
    frozen.keys = list(inventory.keys)


def fragment_key(frozen, frag: int) -> str:
    return str(frozen.keys[int(frag)])


# ---------------------------------------------------------------------------
# panel-tied ambiguity: actual readout differences
# ---------------------------------------------------------------------------


def _readout_map(frozen, inventory, payload) -> dict[int, list[tuple[str, int, int, int]]]:
    """fragment -> list of (split, view index, s, c) for every actual view."""
    out: dict[int, list[tuple[str, int, int, int]]] = {}
    for split in EC.SPLITS:
        p = frozen.per_split[split]
        s = np.asarray(p["view_s_frag"], dtype=np.int64)
        c = np.asarray(p["view_c_frag"], dtype=np.int64)
        for v in range(s.size):
            out.setdefault(int(s[v]), []).append((split, int(v), int(s[v]), int(c[v])))
    return out


def _readout_delta(ctx, a_main, q_main, a_alt, q_alt, frag, readout_map,
                   split_names=("valid", "test")) -> dict[str, Any]:
    frozen = ctx["frozen"]
    std = ctx["std"]
    scale = frozen.q.shape[1] ** 0.5
    rows = readout_map.get(int(frag), [])
    out: dict[str, Any] = {"n_views": 0, "n_molecules": 0,
                           "max_abs_ds_raw": 0.0, "median_abs_ds_raw": 0.0,
                           "max_abs_df_raw": 0.0, "n_views_gt_1e-6": 0,
                           "n_molecules_gt_1e-6": 0}
    for split in split_names:
        sub = [(v, s, c) for (sp, v, s, c) in rows if sp == split]
        if not sub:
            out[f"n_{split}_views"] = 0
            out[f"n_{split}_molecules"] = 0
            out[f"max_abs_ds_{split}_raw"] = 0.0
            out[f"max_abs_df_{split}_raw"] = 0.0
            continue
        ds = []
        mols = []
        for (v, s, c) in sub:
            da = float(a_alt[s] - a_main[s])
            dq = q_alt[s] - q_main[s]
            ds.append((da + float((dq * frozen.k[c] * frozen.w).sum() / scale)) * std)
            mols.append(int(np.asarray(frozen.per_split[split]["view_mol_local"])[v]))
        ds = np.asarray(ds, dtype=np.float64)
        # full-F change if only this fragment's dictionary encoding changes
        mol_ids = np.asarray(mols, dtype=np.int64)
        n_mol = int(np.max(np.asarray(frozen.per_split[split]["view_mol_local"]).max()) + 1) \
            if np.asarray(frozen.per_split[split]["view_mol_local"]).size else 0
        per_mol = np.bincount(mol_ids, weights=ds, minlength=n_mol)
        counts = np.bincount(mol_ids, minlength=n_mol).astype(np.float64)
        active = counts > 0
        df = per_mol[active] / counts[active]
        out[f"n_{split}_views"] = int(ds.size)
        out[f"n_{split}_molecules"] = int(active.sum())
        out[f"max_abs_ds_{split}_raw"] = float(np.max(np.abs(ds)))
        out[f"max_abs_df_{split}_raw"] = float(np.max(np.abs(df))) if df.size else 0.0
        out["n_views"] += int(ds.size)
        out["n_molecules"] += int(active.sum())
        out["max_abs_ds_raw"] = max(out["max_abs_ds_raw"], float(np.max(np.abs(ds))))
        out["max_abs_df_raw"] = max(out["max_abs_df_raw"], float(np.max(np.abs(df))) if df.size else 0.0)
        out["n_views_gt_1e-6"] += int(np.sum(np.abs(ds) > MATERIAL_PRED_DIFF))
        out["n_molecules_gt_1e-6"] += int(np.sum(np.abs(df) > MATERIAL_PRED_DIFF))
    return out


# ---------------------------------------------------------------------------
# main stage
# ---------------------------------------------------------------------------


def run(ctx: Mapping[str, Any], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    config = ctx["config"]
    settings = config["projection"]
    frozen, inventory = ctx["frozen"], ctx["inventory"]
    # attach fragment atom counts and keys for reporting
    _attach_inventory(frozen, inventory, RC.fragment_cache(config["dataset"]))

    replay = EC.replay_checks(ctx)
    cert_rows: list[dict[str, Any]] = []
    alpha_versions: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    fmaps: dict[str, dict[int, np.ndarray]] = {}
    supports: dict[str, dict[int, bool]] = {}
    per_class_summary: list[dict[str, Any]] = []

    for name in EC.DICT_NAMES:
        spec = ctx["dicts16"][name]
        alpha, support = RDR._alpha_all(frozen, inventory, {name: spec},
                                        ctx["recipe"]["dictionary"])
        assert set(support[name].values()) == {True}
        alpha_fw = alpha[name]
        supports[name] = support[name]
        fmap = dict(alpha_fw)
        cert_alpha: dict[int, np.ndarray] = dict(alpha_fw)
        all_alpha: dict[int, np.ndarray] = dict(alpha_fw)
        for tau in ctx["compressed"]:
            frags = sorted(int(f) for f in inventory.word_fragments
                           if inventory.tau.get(int(f)) == tau)
            panels = spec["panels"][tau]
            targets = RD.response_matrix(frozen, frags, spec["d0"][tau], panels)
            rows, ac, aa = solve_class(frozen, inventory, spec, tau, frags, alpha_fw,
                                       targets, panels, settings, name)
            cert_rows.extend(rows)
            cert_alpha.update(ac)
            all_alpha.update(aa)
            # per-class aggregate
            sub = [r for r in rows if r["tau"] == RD.tau_label(tau)]
            if sub:
                g = np.asarray([r["gap_fw64"] for r in sub])
                gf = np.asarray([r["gap_slsqp_fw"] for r in sub])
                gu = np.asarray([r["gap_slsqp_uniform"] for r in sub])
                ff = np.asarray([r["f_fw64"] for r in sub])
                ffw = np.asarray([r["f_slsqp_fw"] for r in sub])
                funi = np.asarray([r["f_slsqp_uniform"] for r in sub])
                fmain = np.asarray([r["f_main"] for r in sub])
                imp = np.asarray([r["f_improvement_fw_to_main"] for r in sub])
                cert = [r for r in sub if str(r["status"]).startswith("CERTIFIED")]
                imp_cert = np.asarray([r["f_improvement_fw_to_main"] for r in cert])
                per_class_summary.append({
                    "dictionary": name, "tau": RD.tau_label(tau),
                    "n_candidates": int(len(inventory.class_words[tau])),
                    "n_prototypes": int(len(spec["prototypes"][tau])),
                    "n_rows": int(len(sub)),
                    "fw_f_median": float(np.median(ff)),
                    "fw_f_p95": float(np.percentile(ff, 95)),
                    "fw_f_max": float(ff.max()),
                    "slsqp_fw_f_median": float(np.median(ffw)),
                    "slsqp_fw_f_max": float(ffw.max()),
                    "slsqp_uniform_f_median": float(np.median(funi)),
                    "slsqp_uniform_f_max": float(funi.max()),
                    "main_f_median": float(np.median(fmain)),
                    "main_f_p95": float(np.percentile(fmain, 95)),
                    "main_f_max": float(fmain.max()),
                    "fw_gap_median": float(np.median(g)),
                    "fw_gap_p95": float(np.percentile(g, 95)),
                    "fw_gap_max": float(g.max()),
                    "fw_pass_fraction": float(np.mean([r["pass_fw64"] for r in sub])),
                    "slsqp_fw_gap_median": float(np.median(gf)),
                    "slsqp_fw_gap_p95": float(np.percentile(gf, 95)),
                    "slsqp_fw_gap_max": float(gf.max()),
                    "slsqp_fw_pass_fraction": float(np.mean([r["pass_slsqp_fw"] for r in sub])),
                    "slsqp_uniform_gap_median": float(np.median(gu)),
                    "slsqp_uniform_gap_p95": float(np.percentile(gu, 95)),
                    "slsqp_uniform_gap_max": float(gu.max()),
                    "slsqp_uniform_pass_fraction": float(np.mean([r["pass_slsqp_uniform"] for r in sub])),
                    "n_certified": int(sum(1 for r in sub if r["main_start"] != "fw64_fallback")),
                    "n_unresolved": int(sum(1 for r in sub if r["main_start"] == "fw64_fallback")),
                    "n_both_qualified": int(sum(1 for r in sub if r["both_qualified"])),
                    "n_ambiguous": int(sum(1 for r in sub if r["ambiguous_panel_solution"])),
                    "n_main_fw": int(sum(1 for r in sub if r["main_start"] == "fw")),
                    "n_main_uniform": int(sum(1 for r in sub if r["main_start"] == "uniform")),
                    "n_improved_certified": int(sum(1 for x in imp_cert if x > 0)),
                    "fw_to_main_f_improvement_median_all": float(np.median(imp)),
                    "fw_to_main_f_improvement_median_certified": float(np.median(imp_cert))
                    if len(imp_cert) else float("nan"),
                    "fw_to_main_f_improvement_p90_certified": float(
                        np.percentile(imp_cert, 90)) if len(imp_cert) else float("nan"),
                    "fw_to_main_f_improvement_max": float(imp.max()),
                })
        alpha_versions[name] = {"fw64": alpha_fw, "cert": cert_alpha, "all": all_alpha}
        fmaps[name] = fmap

    # ---- predictions --------------------------------------------------------
    resp = EC.resp_predictions(ctx)
    predictions: dict[str, dict[str, Any]] = {}
    for name in EC.DICT_NAMES:
        for version, amap in alpha_versions[name].items():
            a_words, q_words = RDR._word_interfaces(frozen, inventory,
                                                    ctx["dicts16"][name], amap,
                                                    supports[name])
            predictions[f"{name}|{version}"] = EC.predict_all(ctx, a_words, q_words,
                                                              include_views=False)

    fw_predictions = {}
    for name in EC.DICT_NAMES:
        for split in EC.SPLITS:
            fw_predictions[f"{name}_{split}_pred_raw"] = predictions[f"{name}|fw64"]["pred_raw"][split]
    replay_fw = EC.check_fw_predictions(ctx, fw_predictions)

    # ---- metrics ------------------------------------------------------------
    metric_rows: list[dict[str, Any]] = []
    saved_table = _load_saved_stress_table(ctx)
    for name in EC.DICT_NAMES:
        for version in ("fw64", "cert", "all"):
            pred = predictions[f"{name}|{version}"]
            for row in EC.metric_row(pred, resp):
                key = (name, row["split"])
                saved = saved_table.get((name, version if version != "fw64" else "fw64", row["split"]))
                metric_rows.append({
                    "dictionary": name,
                    "version": {"fw64": "FW64", "cert": "HP_CERTIFIED",
                                "all": "HP_SLSQP_ALL"}[version],
                    **row,
                    "saved_fw64_rmse_raw": saved["rmse_raw"] if saved else float("nan"),
                    "saved_fw64_mae_raw": saved["mae_raw"] if saved else float("nan"),
                    "saved_fw64_fidelity_raw": saved["fidelity_rmse_raw"] if saved else float("nan"),
                    "rmse_minus_saved_fw64": (row["rmse_raw"] - saved["rmse_raw"]) if saved else float("nan"),
                })

    # ---- panel-tied starts: actual readout differences ----------------------
    # Every row where both deterministic starts meet the pass rule has a panel
    # objective tied within solver precision.  Re-solve those rows once to keep
    # both starts and measure the real Val/Test readout difference.
    readout_map = _readout_map(frozen, inventory, None)
    pua_rows: list[dict[str, Any]] = []
    for r in [x for x in cert_rows if x["both_qualified"]]:
        name = r["dictionary"]
        frag = int(r["fragment_id"])
        spec = ctx["dicts16"][name]
        tau = inventory.tau[frag]
        P = RD.response_matrix(frozen, spec["prototypes"][tau], spec["d0"][tau],
                               spec["panels"][tau])
        frags = sorted(int(f) for f in inventory.word_fragments
                       if inventory.tau.get(int(f)) == tau)
        j = frags.index(frag)
        y = RD.response_matrix(frozen, frags, spec["d0"][tau], spec["panels"][tau])[j]
        a_fw = np.asarray(alpha_versions[name]["fw64"][frag], dtype=np.float64)
        s_fw = solve_row(P, y, a_fw, settings)
        s_uni = solve_row(P, y, np.full(P.shape[0], 1.0 / P.shape[0]), settings)
        a_main, q_main = RDR._word_interfaces(frozen, inventory, spec,
                                              {frag: s_fw["alpha"]}, {frag: True})
        a_alt, q_alt = RDR._word_interfaces(frozen, inventory, spec,
                                            {frag: s_uni["alpha"]}, {frag: True})
        read = _readout_delta(ctx, a_main, q_main, a_alt, q_alt, frag, readout_map)
        alpha_diff = float(np.max(np.abs(np.asarray(s_fw["alpha"])
                                         - np.asarray(s_uni["alpha"]))))
        pua_rows.append({
            "dictionary": name,
            "tau": RD.tau_label(tau),
            "fragment_id": frag,
            "fragment_key_prefix": fragment_key(frozen, frag)[:16],
            "n_atoms": int(frozen.n_atoms[frag]),
            "f_fw_start": float(s_fw["f"]),
            "f_uniform_start": float(s_uni["f"]),
            "f_diff": float(abs(s_fw["f"] - s_uni["f"])),
            "tol": float(r["tol"]),
            "alpha_inf_diff": alpha_diff,
            "alpha_diff_gt_1e-8": bool(alpha_diff > 1e-8),
            **read,
        })
    pua_rows.sort(key=lambda x: (-x["max_abs_df_raw"], x["dictionary"], x["fragment_id"]))
    ambiguous_summary = {
        "n_panel_tied_rows": int(len(pua_rows)),
        "n_alpha_diff_gt_1e-8": int(sum(1 for r in pua_rows if r["alpha_diff_gt_1e-8"])),
        "n_with_val_or_test_views": int(sum(1 for r in pua_rows if r["n_views"] > 0)),
        "n_with_readout_gt_1e-7": int(sum(
            1 for r in pua_rows if r["max_abs_df_raw"] > 1e-7)),
        "n_with_readout_gt_1e-6": int(sum(
            1 for r in pua_rows if r["max_abs_df_raw"] > MATERIAL_PRED_DIFF)),
        "n_views_gt_1e-7": int(sum(1 for r in pua_rows if r["max_abs_ds_raw"] > 1e-7)),
        "n_views_gt_1e-6": int(sum(r["n_views_gt_1e-6"] for r in pua_rows)),
        "n_molecules_gt_1e-6": int(sum(r["n_molecules_gt_1e-6"] for r in pua_rows)),
        "max_abs_ds_raw": float(max([r["max_abs_ds_raw"] for r in pua_rows], default=0.0)),
        "max_abs_df_raw": float(max([r["max_abs_df_raw"] for r in pua_rows], default=0.0)),
        "material_threshold_raw": MATERIAL_PRED_DIFF,
        "label": ("PANEL_UNDERDETERMINES_PREDICTION"
                  if any(r["max_abs_df_raw"] > MATERIAL_PRED_DIFF for r in pua_rows)
                  else "PANEL_TIES_DO_NOT_MATERIALLY_CHANGE_READOUT"),
        "note": ("Both SLSQP starts meet the frozen pass rule, so their panel objectives are "
                 "tied within solver precision. The table lists the actual Val/Test readout "
                 "difference of the two alphas, holding every other fragment fixed. A material "
                 "difference (>1e-6 raw) means the panel does not identify a unique prediction; "
                 "no new panel or tie-breaker is added. Nonzero but sub-1e-6 differences are "
                 "reported, not hidden."),
    }

    # ---- decisions ----------------------------------------------------------
    decision = _projection_decision(metric_rows, config)

    # ---- write tables -------------------------------------------------------
    RC.write_csv(out / "projection_certificate.csv", cert_rows)
    RC.write_csv(out / "projection_summary.csv", per_class_summary)
    RC.write_csv(out / "projection_full_f_metrics.csv", metric_rows)
    RC.write_csv(out / "panel_underdetermination.csv", pua_rows)
    _write_alpha_npz(out / "projection_alpha.npz", alpha_versions, ctx["compressed"], inventory)
    _write_predictions(out / "projection_predictions.npz", predictions)
    RC.write_json(out / "panel_underdetermination.json", ambiguous_summary)
    RC.write_json(out / "projection_checks.json", {
        "replay": replay,
        "fw64_vs_saved_stress_predictions": replay_fw,
        "solver": {k: settings[k] for k in
                   ("method", "ftol", "maxiter", "certificate_rel_tol", "simplex_abs_tol",
                    "sum_abs_tol")},
        "n_certificate_rows": int(len(cert_rows)),
        "n_fw64_pass": int(sum(1 for r in cert_rows if r["pass_fw64"])),
        "n_slsqp_fw_pass": int(sum(1 for r in cert_rows if r["pass_slsqp_fw"])),
        "n_slsqp_uniform_pass": int(sum(1 for r in cert_rows if r["pass_slsqp_uniform"])),
        "n_both_qualified": int(sum(1 for r in cert_rows if r["both_qualified"])),
        "n_unresolved": int(sum(1 for r in cert_rows if r["main_start"] == "fw64_fallback")),
        "n_main_fw": int(sum(1 for r in cert_rows if r["main_start"] == "fw")),
        "n_main_uniform": int(sum(1 for r in cert_rows if r["main_start"] == "uniform")),
        "version_prediction_max_abs_diff_raw": _version_diff(predictions),
        "decision": decision,
        "seconds": time.perf_counter() - t0,
    })
    summary = {
        "stage": "projection",
        "n_certificate_rows": int(len(cert_rows)),
        "n_fw64_pass": int(sum(1 for r in cert_rows if r["pass_fw64"])),
        "n_unresolved": int(sum(1 for r in cert_rows if r["main_start"] == "fw64_fallback")),
        "panel_underdetermination": ambiguous_summary,
        "decision": decision,
        "seconds": time.perf_counter() - t0,
    }
    RC.write_json(out / "projection_summary.json", summary)
    return {
        "cert_rows": cert_rows,
        "alpha_versions": alpha_versions,
        "supports": supports,
        "predictions": predictions,
        "resp": resp,
        "metric_rows": metric_rows,
        "pua_rows": pua_rows,
        "pua_summary": ambiguous_summary,
        "decision": decision,
        "replay": replay,
        "seconds": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _load_saved_stress_table(ctx) -> dict[tuple[str, str, str], dict[str, float]]:
    """The follow-up FW64 table, used only as a replay reference."""
    import csv
    path = (EC.FU_RESULTS / f"{ctx['config']['task']}_RESP" / "dictionary_stress_k16"
            / "stress_table.csv")
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    if not path.is_file():
        return out
    for row in csv.DictReader(path.open(encoding="utf-8")):
        out[(row["dictionary"], "fw64", row["split"])] = {
            "rmse_raw": float(row["rmse_raw"]),
            "mae_raw": float(row["mae_raw"]),
            "fidelity_rmse_raw": float(row["fidelity_rmse_raw"]),
        }
    return out


def _write_alpha_npz(path: Path, alpha_versions, compressed, inventory) -> None:
    payload: dict[str, np.ndarray] = {}
    for name, versions in alpha_versions.items():
        for version, amap in versions.items():
            frags = sorted(int(f) for f in inventory.word_fragments
                           if inventory.tau.get(int(f)) in set(compressed))
            payload[f"{name}_{version}_frag"] = np.asarray(frags, dtype=np.int64)
            payload[f"{name}_{version}_alpha"] = np.stack(
                [np.asarray(amap[f], dtype=np.float64) for f in frags])
    np.savez_compressed(path, **payload)


def _write_predictions(path: Path, predictions) -> None:
    payload: dict[str, np.ndarray] = {}
    for key, pred in predictions.items():
        name, version = key.split("|")
        for split in EC.SPLITS:
            payload[f"{name}_{version}_{split}_pred_raw"] = np.asarray(pred["pred_raw"][split])
            payload[f"{name}_{version}_{split}_row_ids"] = np.asarray(pred["row_ids"][split])
            payload[f"{name}_{version}_{split}_target_raw"] = np.asarray(pred["target"][split])
    np.savez_compressed(path, **payload)


def _version_diff(predictions) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in EC.DICT_NAMES:
        for a, b in (("fw64", "cert"), ("cert", "all"), ("fw64", "all")):
            key = f"{name}|{a}"; other = f"{name}|{b}"
            diff = max(float(np.max(np.abs(
                predictions[key]["pred_raw"][split] - predictions[other]["pred_raw"][split])))
                for split in EC.SPLITS)
            out[f"{name}|{a}_vs_{b}"] = diff
    return out


def _projection_decision(metric_rows, config) -> dict[str, Any]:
    table = {(r["dictionary"], r["version"], r["split"]): r for r in metric_rows}
    versions = ("FW64", "HP_CERTIFIED", "HP_SLSQP_ALL")
    out: dict[str, Any] = {"per_version": {}}
    for version in versions:
        fid = {split: {name: table[(name, version, split)]["fidelity_rmse_raw"]
                       for name in EC.DICT_NAMES}
               for split in ("valid", "test")}
        better = {split: min(fid[split]["D_LATENT"], fid[split]["D_FREQ"])
                  for split in ("valid", "test")}
        reduction = {split: 1.0 - fid[split]["D_RESPONSE"] / better[split]
                     for split in ("valid", "test")}
        best = {split: min(EC.DICT_NAMES, key=lambda n: fid[split][n])
                for split in ("valid", "test")}
        out["per_version"][version] = {
            "fidelity_rmse": fid,
            "better_baseline": better,
            "relative_reduction_vs_better_baseline": reduction,
            "best_dictionary": best,
            "response_is_best_both": bool(best["valid"] == "D_RESPONSE"
                                          and best["test"] == "D_RESPONSE"),
        }
    hp = [out["per_version"][v] for v in ("HP_CERTIFIED", "HP_SLSQP_ALL")]
    reversed_ = any(not x["response_is_best_both"] for x in hp)
    shrunk = any(min(x["relative_reduction_vs_better_baseline"].values()) < 0.20 for x in hp)
    if reversed_:
        label = "RESPONSE_ADVANTAGE_REVERSED"
    elif shrunk:
        label = "RESPONSE_ADVANTAGE_STEP_LIMITED"
    else:
        label = "RESPONSE_ADVANTAGE_SURVIVES_HIGH_PRECISION"
    out["label"] = label
    out["note"] = ("The 20% reduction is the frozen follow-up investment reference, not a general "
                   "effect threshold. HP_CERTIFIED is the primary version; HP_SLSQP_ALL is an "
                   "uncertified sensitivity on rows where the frozen SLSQP setting missed the 1e-8 "
                   "gap certificate.")
    return out
