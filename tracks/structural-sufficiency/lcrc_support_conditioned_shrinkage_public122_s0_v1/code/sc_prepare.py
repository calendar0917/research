"""Structural preparation + Train-only feasibility preflight (zero training).

    uv run --group lipo python code/sc_prepare.py all          # everything
    uv run --group lipo python code/sc_prepare.py keys         # per-root inventory
    uv run --group lipo python code/sc_prepare.py groups       # canonical groups
    uv run --group lipo python code/sc_prepare.py gates        # E1-E5 for both keys
    uv run --group lipo python code/sc_prepare.py select       # frozen key choice

No label, prediction or error of any split is read here.  The gate report and the
key choice depend only on Train structures.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import sc_common as C
import sc_support as S

T0 = time.perf_counter()


def stage_keys(datasets: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in datasets:
        t = time.perf_counter()
        rk = S.load_or_build_rootkeys(dataset)
        active = {s: int(rk[s]["active"].sum()) for s in C.SPLITS}
        roots = {s: int(rk[s]["active"].size) for s in C.SPLITS}
        out[dataset] = {
            "sha256": rk["meta"]["sha256"], "n_key_ids_fine": rk["meta"]["n_key_ids_fine"],
            "n_key_ids_joint": rk["meta"]["n_key_ids_joint"],
            "collision_check_K_FINE": rk["meta"][f"collision_check_{C.KEY_FINE}"],
            "collision_check_K_JOINT": rk["meta"][f"collision_check_{C.KEY_JOINT}"],
            "n_roots": roots, "n_active_roots": active,
            "frac_active": {s: (active[s] / roots[s] if roots[s] else 0.0) for s in C.SPLITS},
            "r_stats": {s: {"max": float(rk[s]["r"].max()) if roots[s] else 0.0,
                            "n_positive": int((rk[s]["r"] > 0).sum())} for s in C.SPLITS},
            "build_seconds": time.perf_counter() - t,
        }
        C.log(f"[keys] {dataset}: {out[dataset]['n_key_ids_fine']} K_FINE ids, "
              f"{out[dataset]['n_key_ids_joint']} K_JOINT ids, "
              f"active roots {active} / {roots}")
    return out


def stage_groups(datasets: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in datasets:
        t = time.perf_counter()
        g = S.build_groups(dataset)
        path = S.save_groups(dataset, g)
        rk = S.load_or_build_rootkeys(dataset)
        ident = S.check_group_identity(dataset, rk, g)
        out[dataset] = {
            "rdkit": g["rdkit"], "definition": g["definition"],
            "n_structures_total": g["n_structures_total"],
            "atommapped_input": g["atommapped_input"],
            "artifact_canonical_match_fraction": g["artifact_canonical_matches"],
            "fold_rule": g["fold_rule"],
            "splits": {s: {"n_records": g[s]["n_records"], "n_groups": g[s]["n_groups"],
                           "n_duplicate_records": g[s]["n_records"] - g[s]["n_groups"],
                           "fold_counts": np.bincount(g[s]["fold"], minlength=C.N_FOLDS).tolist()}
                       for s in C.SPLITS},
            "path": C.track_rel(path), "sha256": C.sha256_file(path),
            "identity": ident, "build_seconds": time.perf_counter() - t,
        }
        C.log(f"[groups] {dataset}: {g['n_structures_total']} structures, "
              f"duplicate groups checked {ident['duplicate_groups_checked']}, "
              f"identity {ident['status']}")
    return out


def stage_gates(candidates: Sequence[tuple[str, str, str]]) -> dict[str, Any]:
    """``(dataset, fit, key)`` -> E1-E5 gate report (Train structures only)."""
    out: dict[str, Any] = {}
    for dataset, fit, key in candidates:
        t = time.perf_counter()
        rk = S.load_or_build_rootkeys(dataset)
        groups = S.load_groups(dataset)
        full = S.support_alpha_tables(dataset, fit, key, rk, groups)
        oof = S.support_alpha_tables(dataset, fit, key, rk, groups, oof=True)
        rep = S.gate_report(dataset, fit, key, full, oof, groups)
        frozen = S.freeze_support(dataset, fit, key, rk, groups, a_oof=rep["a_oof"])
        rep["support_file"] = frozen
        rep["mean_a_select_fit"] = frozen["fit_mean_a_select"]
        rep["uniform_mean_check"] = {
            "select": abs(frozen["fit_mean_a_select"] - C.UNIFORM_A),
            "tolerance": C.GATE_TOL,
            "pass": abs(frozen["fit_mean_a_select"] - C.UNIFORM_A) <= C.GATE_TOL,
        }
        rep["seconds"] = time.perf_counter() - t
        out[f"{dataset}|F{fit}|{key}"] = rep
        C.log(f"[gates] {dataset} F{fit} {key}: pass={rep['pass']} "
              f"spread_full={rep['checks']['E2_spread_fullfit']['value']:.3f} "
              f"spread_oof={rep['checks']['E3_spread_oof']['value']:.3f} "
              f"a_oof={rep['a_oof']:.4f}")
    return out


def stage_select(gates: dict[str, Any], *, dataset: str = C.PRIMARY_DATASET) -> dict[str, Any]:
    """The frozen key-selection rule (Train structures + gates only)."""
    per_fit: dict[str, Any] = {}
    for fit in C.LIPO_FITS:
        ok = [k for k in C.KEY_PRIORITY if gates[f"{dataset}|F{fit}|{k}"]["pass"]]
        per_fit[fit] = {"passing_keys": ok, "chosen": ok[0] if ok else None}

    chosen_both = [k for k in C.KEY_PRIORITY
                   if all(per_fit[f]["chosen"] == k for f in C.LIPO_FITS)]
    fit100_ok = per_fit["100"]["chosen"]
    if chosen_both:
        status = "KEYS_PASS_BOTH_N"
        chosen = chosen_both[0]
        purchase = ["100", "25"]
    elif fit100_ok is not None:
        status = "KEY_PASSES_FIT100_ONLY"
        chosen = fit100_ok
        purchase = ["100"]
    else:
        status = "NO_TARGETABLE_SUPPORT_UNDER_REGISTERED_KEYS"
        chosen = None
        purchase = []
    return {
        "dataset": dataset, "status": status, "chosen_key": chosen,
        "purchase_fits": purchase,
        "per_fit": per_fit,
        "not_run_fits": {f: ("NOT_RUN_NO_TARGETABLE_SUPPORT" if f not in purchase else None)
                         for f in C.LIPO_FITS},
        "rule": ("K_FINE before K_JOINT; buy both N if one key passes both, else only "
                 "fit100 if a key passes fit100; stop when no key passes fit100"),
        "a_oof": {f: gates[f"{dataset}|F{f}|{chosen}"]["a_oof"]
                  for f in C.LIPO_FITS if chosen is not None},
        "source": "Train structures only; no Valid/Test distribution, label or error read",
    }


def stage_freeze(*, datasets: Sequence[str] = ("Lipophilicity", "BACE")) -> dict[str, Any]:
    """Everything a formal run needs, hashed: key, support tables, norms, fit rows."""
    sel = C.read_json(C.ANALYSIS / "key_selection.json")
    out: dict[str, Any] = {
        "round_id": C.ROUND_ID, "created_utc": C.utcnow(),
        "chosen_key": sel["chosen_key"], "purchase_fits": sel["purchase_fits"],
        "status": sel["status"], "arms": list(C.ARMS),
        "gate_constants": {
            "UNIFORM_A": C.UNIFORM_A, "GATE_SLOPE": C.GATE_SLOPE,
            "GATE_RANGE": list(C.GATE_RANGE), "GATE_TOL": C.GATE_TOL,
            "E1_MIN_ACTIVE_GROUPS": C.E1_MIN_ACTIVE_GROUPS,
            "E1_MIN_ACTIVE_KEYS": C.E1_MIN_ACTIVE_KEYS,
            "E2_MIN_SPREAD_FULLFIT": C.E2_MIN_SPREAD_FULLFIT,
            "E3_MIN_SPREAD_OOF": C.E3_MIN_SPREAD_OOF,
            "E4_MAX_UNKNOWN_SHARE": C.E4_MAX_UNKNOWN_SHARE,
            "E5_MIN_LOW_SHARE": C.E5_MIN_LOW_SHARE,
            "E5_MIN_HIGH_SHARE": C.E5_MIN_HIGH_SHARE,
            "LOW_C_MAX": C.LOW_C_MAX, "HIGH_C_MIN": C.HIGH_C_MIN,
        },
        "datasets": {}, "recipes": {"Lipophilicity": C.C1_R3, "BACE": C.C1_R1},
        "budget": {"U0": {d: C.u0(d) for d in datasets},
                   "grid_points": C.GRID_POINTS, "window": C.WINDOW},
        "provenance": {"git_commit": C.git_commit(), "code_hashes": C.code_hashes()},
    }
    for d in datasets:
        rk = S.load_or_build_rootkeys(d)
        g = S.load_groups(d)
        entry: dict[str, Any] = {
            "rootkeys_sha256": rk["meta"]["sha256"],
            "groups_sha256": g["sha256"],
            "n_key_ids": {"K_FINE": rk["meta"]["n_key_ids_fine"],
                          "K_JOINT": rk["meta"]["n_key_ids_joint"]},
            "structure_norm_sha256": C.sha256_file(C.STRUCT_NORM_REF[d]),
            "target_norm_sha256": (C.sha256_file(C.TARGET_NORM_REF[d])
                                   if d in C.TARGET_NORM_REF else None),
            "splits": {s: {"rows_sha256": C.row_sha256(g[s]["rows"]),
                           "n_records": g[s]["n_records"]} for s in C.SPLITS},
            "support": {},
        }
        for fit in (C.LIPO_FITS if d == C.PRIMARY_DATASET else ("100",)):
            sup = S.load_support(d, fit, sel["chosen_key"])
            subset = C.LD.nested_subset_positions(d, fit)
            entry["support"][fit] = {
                "sha256": sup["sha256"], "a_oof": float(sup["a_oof"]),
                "fit_mean_a_select": float(sup["fit_mean_a_select"]),
                "n_support_structures": int(sup["n_support"]),
                "fit_positions_sha256": subset["positions_sha256"],
                "fit_global_rows_sha256": subset["global_rows_sha256"],
                "n_fit": subset["n"],
            }
        out["datasets"][d] = entry
    C.write_json(C.ANALYSIS / "freeze.json", out)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    stage = (argv or sys.argv[1:] or ["all"])[0]
    datasets = ("Lipophilicity", "BACE")
    preflight_path = C.ANALYSIS / "preflight.json"
    out = C.read_json(preflight_path) if preflight_path.exists() else {}

    if stage in ("all", "keys"):
        out.setdefault("stage_keys", {}).update(stage_keys(datasets))
        C.write_json(preflight_path, out)
    if stage in ("all", "groups"):
        g = stage_groups(datasets)
        out.setdefault("stage_groups", {}).update(g)
        C.write_json(preflight_path, out)
        bad = [d for d, v in g.items() if v["identity"]["status"] != "OK"]
        if bad:
            C.log(f"[groups] IDENTITY PROBLEM in {bad}: stopping preparation")
            return 2
    if stage in ("all", "gates", "select"):
        cands: list[tuple[str, str, str]] = []
        for fit in C.LIPO_FITS:
            for key in C.KEY_PRIORITY:
                cands.append((C.PRIMARY_DATASET, fit, key))
        for key in C.KEY_PRIORITY:
            cands.append(("BACE", "100", key))
        gates = out.get("stage_gates", {})
        gates.update(stage_gates(cands))
        out["stage_gates"] = gates
        sel = stage_select(gates)
        out["key_selection"] = sel
        C.write_json(preflight_path, out)
        C.write_json(C.ANALYSIS / "key_selection.json", sel)
        C.log(f"[select] status={sel['status']} key={sel['chosen_key']} "
              f"purchase={sel['purchase_fits']}")
    if stage in ("all", "freeze"):
        fr = stage_freeze()
        C.log(f"[freeze] key={fr['chosen_key']} fits={fr['purchase_fits']}")
    out["updated_utc"] = C.utcnow()
    out["seconds"] = time.perf_counter() - T0
    C.write_json(preflight_path, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
