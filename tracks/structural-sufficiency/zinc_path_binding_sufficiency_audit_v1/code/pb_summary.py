"""Assemble the machine-readable digest used by REPORT.md."""

from __future__ import annotations

import json

import numpy as np

import pb_common as P


def _load(name, default=None):
    p = P.ANALYSIS / name
    if not p.exists():
        return default if default is not None else {}
    return P.read_json(p)


def main() -> int:
    paths = _load("paths.json")
    math = _load("math.json")
    binding = _load("binding.json")
    recon = _load("reconstruction.json")
    graph = _load("graphlevel.json")
    support = _load("support.json")
    batch = _load("batch_invariance.json")
    decision = _load("decision.json")
    probe = _load("probe_control.json")
    prov = _load("provenance.json")

    out = {
        "round": P.ROUND_ID,
        "provenance": {
            "git_commit": prov.get("git", {}).get("commit"),
            "git_short": prov.get("git", {}).get("short"),
            "git_dirty": prov.get("git", {}).get("dirty"),
            "protocol_sha256": prov.get("protocol_sha256"),
            "checkpoints": {k: {"sha256": v.get("sha256"), "valid_mae_frozen": v.get("valid_mae_frozen")}
                            for k, v in prov.get("checkpoints", {}).items()},
            "data": {k: {"sha256": v.get("sha256"), "content_sha256": v.get("content_sha256"),
                         "matches_frozen_config": v.get("matches_frozen_config")}
                     for k, v in prov.get("data", {}).items()},
            "env": prov.get("env"),
            "gpu_hours": 0.0,
            "official_test_loaded": False,
            "labels_used": False,
        },
        "L1_paths": {
            "train_crosscheck": paths.get("elementwise_vs_frozen_cache", {}).get("train", {}).get("passed"),
            "valid_crosscheck": paths.get("elementwise_vs_frozen_cache", {}).get("valid", {}).get("passed"),
            "n_paths_train": paths.get("elementwise_vs_frozen_cache", {}).get("train", {}).get("n_paths"),
            "n_paths_valid": paths.get("elementwise_vs_frozen_cache", {}).get("valid", {}).get("n_paths"),
            "invariance": paths.get("invariance"),
            "boundary": {k: v for k, v in paths.get("boundary", {}).items()
                         if k in ("zero_path_atoms", "single_path_atoms", "all_finite",
                                  "zero_path_T_is_zero", "single_path_D_is_zero")},
        },
        "L1_math": {
            "flat_vs_factorized_max_abs": math.get("flat_factorized", {}).get("flat_vs_factorized_max_abs"),
            "stats_vs_flat_max_abs": math.get("flat_factorized", {}).get("stats_vs_flat_max_abs"),
            "mc_null_bias_max_z": max((v["mean_D_over_draws_max_abs_z"]
                                       for v in math.get("monte_carlo_independent_paths", {})
                                       .get("per_n", {}).values()), default=None),
            "kernel_demo": math.get("kernel_demo_within_cell_reassignment"),
            "path_order_permutation_max_abs_diff": math.get("path_order_permutation_max_abs_diff"),
        },
        "L1_note": "mathematical statements only; no empirical claim",
    }

    # binding headline
    b = {}
    for key, entry in (binding.get("per_checkpoint_seed") or {}).items():
        va, tr = entry["valid"], entry["train"]
        b[key] = {
            "rms_T_real": va["rms_T_real"], "rms_T_null": va["rms_T_null"],
            "rms_D": va["rms_D"], "rms_D_perm": va["rms_D_perm"],
            "rms_T_type": va["rms_T_type"], "rms_D_type": va["rms_D_type"],
            "rms_U": va["rms_U_within_cell"], "rms_U_over_rms_T": va["rms_U_over_rms_T"],
            "rho_median": va["rho"]["rho_median"], "rho_p10": va["rho"]["rho_p10"],
            "frac_atoms_rho_ge_0.05": va["rho"]["frac_atoms_rho_ge_0.05"],
            "frac_atoms_rho_eq_0": va["rho"]["frac_atoms_rho_eq_0"],
            "top1pct_energy_share": va["rho"]["top1pct_energy_share"],
            "mol_median_rho": va["molecules"]["median_mol_rho_median"],
            "frac_mol_median_rho_ge_0.10": va["molecules"]["frac_molecules_median_rho_ge_0.10"],
            "eff_rank_D": va["blocks"]["D"]["entropy_rank"],
            "pr_D": va["blocks"]["D"]["participation_ratio"],
            "eff_rank_T_real": va["blocks"]["T_real"]["entropy_rank"],
            "rho_type_median": va["rho_type"]["rho_median"],
            "rho_type_own_median": va["rho_type_own"]["rho_median"],
            "rms_D_type_over_rms_T_type": va["rms_D_type"] / max(va["rms_T_type"], 1e-12),
            "frac_atoms_multi_cell": va["swap_cells"]["frac_atoms_with_multi_cell"],
            "frac_atoms_U_nonzero": va["swap_cells"]["frac_atoms_U_nonzero"],
            "n_paths_frac_one": va["n_paths"]["frac_one"],
            "rho_by_path_count": {k: v.get("rho_median") for k, v in va["strata"]["by_path_count"].items()},
            "energy_share_by_degree": {k: v.get("energy_share") for k, v in va["strata"]["by_degree"].items()},
            "rho_by_degree": {k: v.get("rho_median") for k, v in va["strata"]["by_degree"].items()},
        }
    out["L2_binding"] = b

    # reconstruction headline
    r = {}
    for key, entry in (recon.get("runs") or {}).items():
        r[key] = {}
        for iname, kinds in entry.items():
            if iname in ("n_fit_atoms", "n_monitor_atoms", "n_valid_atoms",
                         "degenerate_channels", "near_constant_channels", "valid_target_std"):
                continue
            r[key][iname] = {kind: {blk: (v.get("r2") if isinstance(v, dict) else None)
                                    for blk, v in res.get("blocks", {}).items()}
                             for kind, res in kinds.items()}
    out["L2_reconstruction"] = r
    perm = {}
    for key, entry in (recon.get("permutation_controls") or {}).items():
        perm[key] = {iname: {kind: res.get("blocks", {}).get("D", {}).get("r2")
                             for kind, res in kinds.items()}
                     for iname, kinds in entry.items()}
    out["L2_reconstruction_permutation_D"] = perm
    out["L2_degenerate_channels"] = {
        key: {"degenerate": entry.get("degenerate_channels"),
              "near_constant": entry.get("near_constant_channels")}
        for key, entry in (recon.get("runs") or {}).items()}

    # graph
    g = {}
    for key, entry in (graph.get("runs") or {}).items():
        g[key] = {"E_to_G": {kind: {blk: v.get("r2") for blk, v in entry["E_to_G"][kind]["blocks"].items()}
                             for kind in ("ridge", "mlp")},
                  "G_to_E": {kind: entry["G_to_E"][kind]["blocks"]["E"].get("r2")
                             for kind in ("ridge", "mlp")}}
    out["L2_graph"] = g

    out["L2_support"] = {
        "vocabulary": support.get("vocabulary"),
        "counts": support.get("valid_atom_counts"),
        "fractions": support.get("valid_atom_fractions"),
        "molecule_has_unseen": support.get("molecule_has_unseen"),
        "groups": support.get("groups"),
    }
    out["L2_probe_control"] = (probe.get("runs") or {}) and {
        ck: {iname: {kind: {blk: v.get("r2") for blk, v in res["blocks"].items()}
                     for kind, res in kinds.items() if isinstance(res, dict)}
             for iname, kinds in entry.items() if isinstance(kinds, dict)
             and not iname.startswith("type_vs_learned")}
        for ck, entry in probe["runs"].items()}
    out["L2_probe_type_vs_learned"] = {
        ck: entry.get("type_vs_learned") for ck, entry in (probe.get("runs") or {}).items()}
    out["L3_batch_invariance"] = {
        "historical_N_batch_dependence": (batch.get("historical_N") or {}).get("max_abs_range"),
        "R_batch_dependence": (batch.get("R_reference") or {}).get("max_abs_range"),
        "corrected_N_batch_dependence": (batch.get("corrected_N") or {}).get("max_abs_range"),
        "trained_N_s0": {k: v for k, v in (batch.get("trained_N_s0") or {}).items()
                         if k in ("checkpoint_sha256", "head_norm", "batch_sizes",
                                  "max_abs_range", "mean_abs_range")},
    }
    out["decision"] = {k: v for k, v in decision.items()
                       if k in ("criteria", "no_go_triggers", "graph_level", "verdict",
                                "borderline_hits", "levels", "gpu_recommendation")}
    P.write_json(P.ANALYSIS / "summary.json", out)
    print(json.dumps({k: out[k] for k in ("decision",)}, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
