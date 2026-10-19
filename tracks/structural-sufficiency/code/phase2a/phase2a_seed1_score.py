"""SSRC Phase 2a seed-1 scoring and round verdict.

Reads the frozen seed-1 predictions (P3E/F/M, all sharing the fixed Fit-only SSL
encoder), verifies the matched-seed audit, and scores them against the matched
seed-1 P3E baseline.  It also assembles the two-seed agreement summary and the
round verdict defined in ``results/phase2a/seed1_purchase_record.md`` §5.

This file does not touch the frozen seed-0 artefacts; it writes
``analysis_seed1_phase2a.json``, ``decision_seed1_phase2a.json`` and
``round_verdict_phase2a.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import phase2a_common as C  # noqa: E402
import phase2a_runner as R  # noqa: E402

OUT = C.RESULTS_2A
SEED = 1


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    for name in ("analysis_seed1_phase2a.json", "decision_seed1_phase2a.json", "round_verdict_phase2a.json"):
        if (OUT / name).exists():
            raise SystemExit(f"{name} exists; refusing to rewrite")

    split = C.load_split()
    labels = C.load_vault()
    y = C.row_labels(split.monitor, labels)
    groups_info = C.label_groups(split.monitor)
    masks = groups_info["masks"]
    cycle_component = groups_info["cycle_component"]
    y_nocycle = y - cycle_component

    recs = {arm: R._arm_record(arm, SEED) for arm in ("P3E", "F", "M")}
    preds = {f"{arm}_s{SEED}": rec["pred"] for arm, rec in recs.items()}
    preds["P3E_s0"] = C.load_p3e_predictions().pred
    preds["F_s0"] = R._arm_record("F", 0)["pred"]
    preds["M_s0"] = R._arm_record("M", 0)["pred"]

    # --- matched-seed audit --------------------------------------------------
    audits = {
        arm: {
            "params": rec["train"]["parameter_audit"]["actual_parameters"],
            "init_full_sha256": rec["train"]["init_full_sha256"],
            "init_group_state_sha256": rec["train"]["init_group_state_sha256"],
            "encoder_used_sha256": rec["train"]["encoder_used_sha256"],
            "batch_order_sha256": rec["train"]["batch_order"]["full_hash"],
            "local_block_init_sha256": rec["train"]["local_block_init_sha256"],
            "soup_fit_mae": rec["train"]["soup_fit_mae"],
            "wall_seconds": rec["train"]["wall_seconds"],
            "step_seconds_mean": rec["train"]["step_seconds"]["mean"],
            "grad_clip_events": rec["train"]["anomalies"]["grad_clip_events"],
            "final_new_param_grad_norms": {
                k: v
                for k, v in (rec["train"]["grad_group_norms_final_step"] or {}).items()
                if k.startswith("local.")
            },
        }
        for arm, rec in recs.items()
    }
    batch_hashes = {v["batch_order_sha256"] for v in audits.values()}
    init_hashes = {v["init_full_sha256"] for v in audits.values()}
    enc_hashes = {v["encoder_used_sha256"] for v in audits.values()}
    seed0_records = {arm: R._arm_record(arm, 0)["train"] for arm in ("F", "M")}
    seed0_audits = {
        arm: {
            "params": rec["parameter_audit"]["actual_parameters"],
            "batch_order_sha256": rec["batch_order"]["full_hash"],
            "init_full_sha256": rec["init_full_sha256"],
        }
        for arm, rec in seed0_records.items()
    }
    seed0_init = seed0_records["F"]["init_full_sha256"]
    audit = {
        "batch_order_hashes": sorted(batch_hashes),
        "batch_order_matched_across_arms": len(batch_hashes) == 1,
        "batch_order_differs_from_seed0": sorted(batch_hashes)[0]
        != C.P3E_BATCH_ORDER_SHA,
        "scratch_init_hashes": sorted(init_hashes),
        "scratch_init_matched_across_arms": len(init_hashes) == 1,
        "scratch_init_differs_from_seed0": sorted(init_hashes)[0] != seed0_init,
        "local_block_init_matched_F_vs_M": audits["F"]["local_block_init_sha256"]
        == audits["M"]["local_block_init_sha256"],
        "encoder_hashes": sorted(enc_hashes),
        "encoder_fixed_at_seed0_pretraining": sorted(enc_hashes) == [C.P3E_ENCODER_SHA],
        "per_arm": audits,
        "seed0_per_arm": seed0_audits,
        "ssl_encoder_policy": (
            "the Fit-only masked-SSL encoder is fixed at seed 0 for both seeds and all "
            "three arms; seed 1 varies only the supervised trajectory (composer/reader "
            "init + batch order)"
        ),
    }
    if not (
        audit["batch_order_matched_across_arms"]
        and audit["scratch_init_matched_across_arms"]
        and audit["encoder_fixed_at_seed0_pretraining"]
    ):
        raise SystemExit(f"matched-seed audit failed: {audit}")

    metrics = {
        k: C.arm_metrics(v, y, y_nocycle=y_nocycle, cycle_component=cycle_component, groups=masks)
        for k, v in preds.items()
    }
    contrasts = {
        "M_vs_F_s1": C.contrast(preds["M_s1"], preds["F_s1"], y, masks,
                                definition="mean |e_M| - |e_F| at seed 1; negative means M is better"),
        "M_vs_P3E_s1": C.contrast(preds["M_s1"], preds["P3E_s1"], y, masks,
                                  definition="mean |e_M| - |e_P3E(s1)|; negative means M is better"),
        "F_vs_P3E_s1": C.contrast(preds["F_s1"], preds["P3E_s1"], y, masks,
                                  definition="mean |e_F| - |e_P3E(s1)|; negative means F is better"),
        "M_s1_vs_P3E_s0": C.contrast(preds["M_s1"], preds["P3E_s0"], y, masks,
                                     definition="cross-seed reference only; seed-confounded"),
    }

    flops = C.read_json(OUT / "flops_phase2a.json")["models"]
    optimization_ok = True
    opt: dict[str, Any] = {}
    for arm, rec in recs.items():
        tr = rec["train"]
        last5 = [e["train_mae"] for e in tr["lr_curve"][-5:]]
        new_grads = {k: v for k, v in (tr["grad_group_norms_final_step"] or {}).items() if k.startswith("local.")}
        ok = (
            all(np.isfinite(e["train_mae"]) for e in tr["lr_curve"])
            and max(last5) < 0.01
            and tr["soup_fit_mae"] < 0.05
            and all(v > 0 for v in new_grads.values())
            and np.isfinite(tr["anomalies"]["grad_norm_max"])
        )
        optimization_ok &= bool(ok)
        opt[arm] = {
            "ok": bool(ok),
            "last5_train_mae": last5,
            "soup_fit_mae": tr["soup_fit_mae"],
            "final_new_param_grad_norms": new_grads,
        }

    complement_gain_s1 = (
        metrics["P3E_s1"]["groups"]["label_complement"]["mae"]
        - metrics["M_s1"]["groups"]["label_complement"]["mae"]
    )
    decision_s1 = C.decide(
        mae_f=metrics["F_s1"]["mae"],
        mae_m=metrics["M_s1"]["mae"],
        rmse_m=metrics["M_s1"]["rmse"],
        complement_gain=complement_gain_s1,
        flops_ratio_f=flops["F"]["flops_ratio_vs_p3e"],
        flops_ratio_m=flops["M"]["flops_ratio_vs_p3e"],
        optimization_ok=optimization_ok,
        baseline_mae=metrics["P3E_s1"]["mae"],
        baseline_rmse=metrics["P3E_s1"]["rmse"],
    )
    decision_s1["optimization_details"] = opt
    decision_s1["baseline_arm"] = "P3E_s1 (matched seed-1 baseline, same fixed SSL encoder)"

    # --- two-seed agreement --------------------------------------------------
    seed0_analysis = C.read_json(OUT / "analysis_phase2a.json")
    m0 = seed0_analysis["metrics"]
    per_seed = {
        "seed0": {
            "p3e_mae": m0["P3E"]["mae"],
            "f_mae": m0["F_s0"]["mae"],
            "m_mae": m0["M_s0"]["mae"],
            "gain_p3e": m0["P3E"]["mae"] - m0["M_s0"]["mae"],
            "gain_mp": m0["F_s0"]["mae"] - m0["M_s0"]["mae"],
            "p3e_rmse": m0["P3E"]["rmse"],
            "m_rmse": m0["M_s0"]["rmse"],
            "rmse_delta": m0["M_s0"]["rmse"] - m0["P3E"]["rmse"],
            "complement_gain": m0["P3E"]["groups"]["label_complement"]["mae"]
            - m0["M_s0"]["groups"]["label_complement"]["mae"],
            "baseline": "frozen P3E seed 0",
        },
        "seed1": {
            "p3e_mae": metrics["P3E_s1"]["mae"],
            "f_mae": metrics["F_s1"]["mae"],
            "m_mae": metrics["M_s1"]["mae"],
            "gain_p3e": metrics["P3E_s1"]["mae"] - metrics["M_s1"]["mae"],
            "gain_mp": metrics["F_s1"]["mae"] - metrics["M_s1"]["mae"],
            "p3e_rmse": metrics["P3E_s1"]["rmse"],
            "m_rmse": metrics["M_s1"]["rmse"],
            "rmse_delta": metrics["M_s1"]["rmse"] - metrics["P3E_s1"]["rmse"],
            "complement_gain": complement_gain_s1,
            "baseline": "matched P3E seed 1",
        },
    }
    gains_p3e = [per_seed["seed0"]["gain_p3e"], per_seed["seed1"]["gain_p3e"]]
    gains_mp = [per_seed["seed0"]["gain_mp"], per_seed["seed1"]["gain_mp"]]
    agreement = {
        "per_seed": per_seed,
        "gain_p3e_mean": float(np.mean(gains_p3e)),
        "gain_p3e_range": [float(min(gains_p3e)), float(max(gains_p3e))],
        "gain_p3e_sign_consistent": bool(np.all(np.sign(gains_p3e) > 0)),
        "gain_mp_mean": float(np.mean(gains_mp)),
        "gain_mp_range": [float(min(gains_mp)), float(max(gains_mp))],
        "gain_mp_sign_consistent": bool(np.all(np.sign(gains_mp) > 0)),
        "p3e_seed_spread_mae": abs(per_seed["seed1"]["p3e_mae"] - per_seed["seed0"]["p3e_mae"]),
        "m_seed_spread_mae": abs(per_seed["seed1"]["m_mae"] - per_seed["seed0"]["m_mae"]),
        "f_seed_spread_mae": abs(per_seed["seed1"]["f_mae"] - per_seed["seed0"]["f_mae"]),
        "rmse_proxy_failed_seed0_by": per_seed["seed0"]["rmse_delta"] - C.RMSE_TOLERANCE,
        "rmse_proxy_failed_seed1_by": per_seed["seed1"]["rmse_delta"] - C.RMSE_TOLERANCE,
    }

    # --- round verdict ------------------------------------------------------
    if not optimization_ok:
        verdict = "OPTIMIZATION_UNRESOLVED"
        why = ["an arm failed the optimisation checks at seed 1"]
    elif agreement["gain_p3e_sign_consistent"] and agreement["gain_mp_sign_consistent"]:
        if agreement["gain_p3e_mean"] >= C.PRACTICAL_BAND and agreement["gain_mp_mean"] >= C.MP_SEPARATION_BAND:
            verdict = "STRONG_BACKBONE_MP_INCREMENT"
            why = [
                "both seeds: M beats F and the (matched) P3E baseline",
                "two-seed mean Gain_P3E = %.6f, Gain_MP = %.6f"
                % (agreement["gain_p3e_mean"], agreement["gain_mp_mean"]),
                "the frozen absolute RMSE proxy missed its tolerance at seed 0 by 0.000208; "
                "see posthoc_phase2a.json (single permutation-unstable label row)",
            ]
        else:
            verdict = "CAPACITY_ONLY_INCREMENT"
            why = ["signs agree but the two-seed mean effect is below the practical band"]
    elif not agreement["gain_p3e_sign_consistent"]:
        verdict = "UNRESOLVED"
        why = ["Gain_P3E direction conflicts between seeds; no further seed is bought"]
    else:
        verdict = "NO_PRACTICAL_GAIN_OVER_P3E"
        why = ["Gain_P3E is not reproduced in sign"]

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "score_seed1",
        "seed": SEED,
        "split_sha256": split.sha256,
        "monitor_rows_sha256": C.rows_sha256(split.monitor),
        "monitor_labels_read_count": 1,
        "reserved_labels_read": 0,
        "official_valid_test_loaded": False,
        "matched_seed_audit": audit,
        "groups": {"counts": groups_info["counts"], "caveat": groups_info["caveat"]},
        "metrics": metrics,
        "contrasts": contrasts,
        "efficiency": {
            "wall_seconds": {a: audits[a]["wall_seconds"] for a in audits},
            "step_seconds_mean": {a: audits[a]["step_seconds_mean"] for a in audits},
            "flops_ratio_vs_p3e": {"F": flops["F"]["flops_ratio_vs_p3e"], "M": flops["M"]["flops_ratio_vs_p3e"]},
        },
        "decision_seed1": decision_s1,
        "two_seed": agreement,
    }
    verdict_payload = {
        "protocol_id": C.PROTOCOL_ID,
        "status": verdict,
        "reasons": why,
        "seed0_decision_artefact": "decision_phase2a.json (INSUFFICIENT_EVIDENCE, frozen rules)",
        "seed1_decision_artefact": "decision_seed1_phase2a.json",
        "two_seed": agreement,
        "purchase_record": "results/phase2a/seed1_purchase_record.md",
        "no_further_seed_bought": True,
    }
    # compute everything before the first write so a crash cannot half-freeze the stage
    C.write_json(OUT / "analysis_seed1_phase2a.json", payload)
    C.write_json(OUT / "decision_seed1_phase2a.json", decision_s1)
    C.write_json(OUT / "round_verdict_phase2a.json", verdict_payload)
    print(
        f"[score-s1] P3E {metrics['P3E_s1']['mae']:.6f} F {metrics['F_s1']['mae']:.6f} "
        f"M {metrics['M_s1']['mae']:.6f} | Gain_P3E {agreement['gain_p3e_mean']:.6f} "
        f"Gain_MP {agreement['gain_mp_mean']:.6f} -> {verdict}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
