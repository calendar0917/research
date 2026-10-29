"""Machine-check every number quoted in ``REPORT.md`` against the frozen JSON/CSV artifacts.

    python zpr_verify_report.py

Fails loudly on a transcription error.  Read-only: it recomputes nothing from
the models and touches no test data beyond the already-pulled ``test_read.json``.
"""

from __future__ import annotations

import csv
import sys

import numpy as np

import zpr_common as Z

TOL = 5e-6


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    A = Z.ANALYSIS
    decision = Z.read_json(A / "decision.json")
    main_table = Z.read_json(A / "main_table.json")
    phase_a = Z.read_json(A / "phase_a_summary.json")
    prov = Z.read_json(A / "phase_a_label_provenance.json")
    basis = Z.read_json(A / "phase_a_basis_artifact.json")
    div = Z.read_json(A / "divergence.json")
    c4 = Z.read_json(A / "c4_resolution.json")
    test = Z.read_json(A / "test_read.json")

    g = main_table["groups"]
    ema = main_table["ema_within_arm"]
    pb = main_table["paired_valid"]["online_best"]
    mech = phase_a["mechanism_checks"]
    res = test["results"]

    checks: list[tuple[str, bool, str]] = []

    def ck(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    # ---- §2 decision ------------------------------------------------------
    ck("verdict", decision["verdict"] == "NO_COMPELLING_RECIPE_GAIN", decision["verdict"])
    ck("delta_full", _close(g["full"]["delta_online_G1_minus_G0"], +0.00298),
       f"{g['full']['delta_online_G1_minus_G0']:+.5f}")
    ck("delta_ordinary", _close(g["ordinary"]["delta_online_G1_minus_G0"], +0.00407),
       f"{g['ordinary']['delta_online_G1_minus_G0']:+.5f}")
    ck("delta_minus_row172", _close(g["minus_row172"]["delta_online_G1_minus_G0"], +0.00537),
       f"{g['minus_row172']['delta_online_G1_minus_G0']:+.5f}")
    for key, quoted in (("r2_unseen_ordinary", +0.00233), ("seen_ordinary", +0.00462),
                        ("minus_fixed_worst10", +0.00798), ("W20", +0.00470),
                        ("T100", +0.00474), ("tail", -0.01265)):
        ck(f"delta_{key}", _close(g[key]["delta_online_G1_minus_G0"], quoted),
           f"{g[key]['delta_online_G1_minus_G0']:+.5f}")
    ck("tail_values", _close(g["tail"]["G0_online"], 0.28948) and _close(g["tail"]["G1_online"], 0.27683),
       f"{g['tail']['G0_online']:.5f}/{g['tail']['G1_online']:.5f}")
    ck("best_online_G0", _close(main_table["arms"]["G0"]["best_online"]["valid_mae"], 0.0792349),
       f"{main_table['arms']['G0']['best_online']['valid_mae']:.7f}")
    ck("best_online_G1", _close(main_table["arms"]["G1"]["best_online"]["valid_mae"], 0.0822147),
       f"{main_table['arms']['G1']['best_online']['valid_mae']:.7f}")
    ck("train_at_best", _close(main_table["arms"]["G0"]["train_at_best"], 0.00341)
       and _close(main_table["arms"]["G1"]["train_at_best"], 0.01385),
       f"{main_table['arms']['G0']['train_at_best']:.5f}/{main_table['arms']['G1']['train_at_best']:.5f}")
    ck("bootstrap_ci", _close(pb["delta_mae_bootstrap_ci95"][0], -0.0057, 5e-4)
       and _close(pb["delta_mae_bootstrap_ci95"][1], +0.0098, 5e-4),
       f"[{pb['delta_mae_bootstrap_ci95'][0]:+.4f}, {pb['delta_mae_bootstrap_ci95'][1]:+.4f}]")

    # ---- §3 EMA -----------------------------------------------------------
    ck("ema_valid_G0", _close(ema["G0"]["delta_ema_minus_online"], +0.00043),
       f"{ema['G0']['delta_ema_minus_online']:+.5f}")
    ck("ema_valid_G1", _close(ema["G1"]["delta_ema_minus_online"], +0.00025),
       f"{ema['G1']['delta_ema_minus_online']:+.5f}")
    ck("ema_direction_flips", np.sign(ema["G0"]["delta_ema_minus_online"])
       != np.sign(test["contrasts"]["ema_vs_online_G0"]["delta_b_minus_a"]),
       "valid worse, test better")

    # ---- §1 phase A -------------------------------------------------------
    ck("failure_rows_union", phase_a["failure_rows_union_size"] == 26,
       str(phase_a["failure_rows_union_size"]))
    ck("explanations", phase_a["explanations"] == {"SHARED_FAILURE": 14, "MODEL_COMPLEMENTARY": 12},
       str(phase_a["explanations"]))
    sf = mech["shared_failure_supported_by_structure"]
    ck("shared_unseen", sf["n_shared_with_unseen_r2"] == 10, str(sf["n_shared_with_unseen_r2"]))
    ck("shared_penalty", sf["n_shared_with_cycle_penalty"] == 4, str(sf["n_shared_with_cycle_penalty"]))
    ck("shared_neither", sf["shared_rows_with_neither"] == [336], str(sf["shared_rows_with_neither"]))
    ck("ae_share_unseen", _close(sf["share_of_total_ae_unseen_r2"]["lcrc"], 0.4459, 1e-4)
       and _close(sf["share_of_total_ae_unseen_r2"]["dcc_s0"], 0.4758, 1e-4),
       f"{sf['share_of_total_ae_unseen_r2']['lcrc']:.4f}/{sf['share_of_total_ae_unseen_r2']['dcc_s0']:.4f}")
    ck("ae_share_penalty", _close(sf["share_of_total_ae_cycle_penalty"]["lcrc"], 0.1480, 1e-4)
       and _close(sf["share_of_total_ae_cycle_penalty"]["dcc_s0"], 0.1440, 1e-4),
       f"{sf['share_of_total_ae_cycle_penalty']['lcrc']:.4f}")
    ck("blend_gain_asymmetry",
       _close(mech["model_complementarity_supported_by"]["mean_blend_gain_on_lcrc_only"], 0.1631, 1e-3)
       and _close(mech["model_complementarity_supported_by"]["mean_blend_gain_on_dcc_only"], -0.1012, 1e-3),
       "lcrc-only +0.163 / dcc-only -0.101")
    lf = mech["label_and_data_findings"]
    ck("identity_unverified", lf["identity_unverified_rows"] == [208, 680, 770],
       str(lf["identity_unverified_rows"]))
    ck("reconstruction_outliers", lf["reconstruction_outlier_rows"] == [924, 976],
       str(lf["reconstruction_outlier_rows"]))
    ck("penalty_rows", lf["n_cycle_penalty_rows"] == 52, str(lf["n_cycle_penalty_rows"]))
    ck("penalty_basis_only", basis["penalty_basis_only_rows"]
       and [r["row"] for r in basis["penalty_basis_only_rows"]] == [230, 402, 547, 670, 687, 728],
       str([r["row"] for r in basis["penalty_basis_only_rows"]]))
    ck("basis_rows_have_no_ring_ge7",
       all(max(r["mcb_sizes"]) <= 6 for r in basis["penalty_basis_only_rows"])
       and all(r["rdkit_max_ring_size"] <= 6 for r in basis["penalty_basis_only_rows"])
       and all(max(r["basis_sizes"]) > 6 for r in basis["penalty_basis_only_rows"])
       and all(r["longest_simple_cycle"] > 6 for r in basis["penalty_basis_only_rows"]),
       "basis > 6, MCB <= 6, rdkit ring <= 6, long simple cycle > 6")
    ck("outlier_rows_are_low_error",
       all(r["lcrc_ae"] < 0.11 and r["dcc_ae"] < 0.12 for r in prov["outlier_rows"]),
       "no failure row among the 2 outliers")
    # the 5 caveat rows' AE, quoted as 0.0004-0.11 in the report
    with open(A / "phase_a_rows.csv", newline="") as fh:
        rows = {int(r["row"]): r for r in csv.DictReader(fh)}
    caveat = [208, 680, 770, 924, 976]
    aes = [float(rows[r][c]) for r in caveat for c in ("lcrc_ae", "dcc_ae")]
    ck("caveat_rows_small_ae", max(aes) < 0.12, f"max AE {max(aes):.3f}")

    # ---- §4 divergence / C4 ----------------------------------------------
    cmpc = div["comparisons"]
    ck("replay_vs_replay", _close(cmpc["frozen_a_vs_frozen_b"]["max_valid_diff"], 0.427, 1e-3),
       f"{cmpc['frozen_a_vs_frozen_b']['max_valid_diff']:.4f}")
    ck("replay_vs_history_a", _close(cmpc["frozen_a_vs_history"]["max_valid_diff"], 0.397, 1e-3),
       f"{cmpc['frozen_a_vs_history']['max_valid_diff']:.4f}")
    ck("replay_vs_history_b", _close(cmpc["frozen_b_vs_history"]["max_valid_diff"], 0.177, 1e-3),
       f"{cmpc['frozen_b_vs_history']['max_valid_diff']:.4f}")
    ck("G0_vs_history", _close(cmpc["G0_vs_history"]["max_valid_diff"], 0.359, 1e-3),
       f"{cmpc['G0_vs_history']['max_valid_diff']:.4f}")
    ck("c4_corrected_pass", c4["corrected_c4_pass"] is True, str(c4["corrected_c4_pass"]))
    ck("replay_exceeds_control", cmpc["frozen_a_vs_frozen_b"]["max_valid_diff"]
       > cmpc["G0_vs_history"]["max_valid_diff"], "frozen-vs-frozen > control-vs-history")

    # ---- §2/§5 test read --------------------------------------------------
    ck("test_read_count", test["read_count"] == 1, str(test["read_count"]))
    ck("test_label", test["label"] == "TEST_AS_DEV_DESCRIPTIVE", test["label"])
    ck("test_loaded_after_freeze", test["official_test_loaded_before_freeze"] is False,
       str(test["official_test_loaded_before_freeze"]))
    ck("test_G0_online", _close(res["G0_online"]["test_mae"], 0.067030), f"{res['G0_online']['test_mae']:.6f}")
    ck("test_G1_online", _close(res["G1_online"]["test_mae"], 0.073041), f"{res['G1_online']['test_mae']:.6f}")
    ck("test_p90", _close(res["G0_online"]["test_p90_ae"], 0.1391, 1e-4)
       and _close(res["G1_online"]["test_p90_ae"], 0.1509, 1e-4),
       f"{res['G0_online']['test_p90_ae']:.4f}->{res['G1_online']['test_p90_ae']:.4f}")
    ck("test_max_ae", _close(res["G0_online"]["test_max_ae"], 1.530, 1e-3)
       and _close(res["G1_online"]["test_max_ae"], 2.508, 1e-3),
       f"{res['G0_online']['test_max_ae']:.3f}->{res['G1_online']['test_max_ae']:.3f}")
    hr = test["historical_reproduction"]
    ck("historical_reproduction", all(v["matches_published"] for v in hr.values()),
       ",".join(f"{k}={v['test_mae']:.10f}" for k, v in hr.items()))
    ck("anchor_selfcheck", test["grouping_rule_selfcheck"]["exact"] is True,
       f"{test['grouping_rule_selfcheck']['n_unseen_rule']}/"
       f"{test['grouping_rule_selfcheck']['n_unseen_anchor']}")
    ck("test_group_sizes", test["group_info"]["n"] == 1000 and test["group_info"]["n_r2_unseen"] == 244,
       f"n={test['group_info']['n']} unseen={test['group_info']['n_r2_unseen']}")

    bad = [(n, d) for n, ok, d in checks if not ok]
    Z.log(f"[verify-report] {len(checks) - len(bad)}/{len(checks)} checks pass")
    for n, d in bad:
        Z.log(f"[verify-report] FAIL {n}: {d}")
    Z.write_json(A / "report_consistency.json", {
        "round": Z.ROUND_ID,
        "n_checks": len(checks),
        "n_failed": len(bad),
        "checks": [{"name": n, "pass": ok, "observed": d} for n, ok, d in checks],
        "source": "recomputed from analysis/*.json + phase_a_rows.csv; REPORT.md is the prose",
        "created": Z.now(),
    })
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
