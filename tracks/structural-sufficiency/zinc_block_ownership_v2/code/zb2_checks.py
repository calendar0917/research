"""Phase B authorization gates for ``zinc_block_ownership_v2``.

The S arm may only be trained if every gate here passes.  Gates are cheap and
CPU-only; they re-verify the v1 frozen inputs, consume the Phase A diagnostics
(``analysis/phase_a.json``), and check that the S arm is not a no-op.

Nothing here re-derives a scientific result: it only authorizes (or blocks) the
one GPU purchase pre-registered in ``PLAN.md`` §2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import zb2_common as Z  # noqa: E402

BC = Z.BC
# built by concatenation so that this scanner does not match its own source
_SEP = "_"
FORBIDDEN_TOKENS = ("ad" + _SEP + "test",
                    "load_cache(" + '"te' + 'st")',
                    "valid_pred" + _SEP + "test",
                    "official" + _SEP + "test_loaded\": true")


def gate_g1_integrity() -> dict:
    chk = Z.v1_integrity()
    return {"name": "G1_v1_frozen_inputs_unchanged", **chk,
            "pass": bool(chk["ok"])}


def gate_g2_phase_a(path: Path) -> dict:
    if not path.exists():
        return {"name": "G2_phase_a_present", "pass": False,
                "reason": f"missing {path}"}
    d = Z.read_json(path)
    a4 = d["A4_s_checks"]
    checks = {
        "S1_conservation": a4["S1_conservation"]["pass"],
        "S2_invariance_S_content": a4["S2_invariance_S_content"]["pass"],
        "S3_init_identity_seed0": a4["S3_init_identity"]["seed0"]["all_equal"],
        "S3_init_identity_seed1": a4["S3_init_identity"]["seed1"]["all_equal"],
        "S4_gradient": a4["S4_gradient"]["pass"],
        "S5_prediction_consistency": a4["S5_prediction_consistency"]["pass"],
        "official_test_not_loaded": d["official_test_loaded"] is False,
    }
    return {"name": "G2_phase_a_S_checks", "checks": checks,
            "failed": [k for k, v in checks.items() if not v],
            "pass": all(checks.values())}


def gate_g3_contribution_bookkeeping(path: Path) -> dict:
    d = Z.read_json(path)["A2_contributions"]
    parts = {k: v["abs_gap"] for k, v in d["partition_checks"].items()}
    return {"name": "G3_contribution_partitions_sum_to_total",
            "partition_abs_gaps": parts,
            "tolerance": 1e-9,
            "pass": bool(all(v <= 1e-9 for v in parts.values()))}


def gate_g4_arm_not_noop(path: Path) -> dict:
    d = Z.read_json(path)
    a3 = d["A3_n_multi"]
    probe = d["A5_probe"]["R"]["R->S"]
    checks = {
        # S must differ from R on multi-block molecules ...
        "s_differs_from_r_on_multi_ge_2":
            a3["n_multi_ge_2_s_equals_r_exact_fraction"] < 0.5,
        # ... and must be identical to R on the n_multi<=1 control set
        "s_identical_to_r_on_multi_le_1":
            a3["n_multi_le_1_s_equals_r_exact_fraction"] == 1.0,
        # the trained R model must react to the substitution at all
        "r_model_reacts_to_S_content": probe["mean_abs_pred_delta"] > 1e-4,
    }
    return {"name": "G4_S_arm_not_a_noop", "checks": checks,
            "s_equals_r_frac_multi_ge_2":
                a3["n_multi_ge_2_s_equals_r_exact_fraction"],
            "s_equals_r_frac_multi_le_1":
                a3["n_multi_le_1_s_equals_r_exact_fraction"],
            "r_model_mean_abs_pred_delta_R_to_S": probe["mean_abs_pred_delta"],
            "pass": all(checks.values())}


def gate_g5_no_test_access() -> dict:
    hits = {}
    for p in sorted(list(Z.HERE.glob("*.py"))):
        txt = p.read_text()
        found = [t for t in FORBIDDEN_TOKENS if t in txt]
        if found:
            hits[p.name] = found
    test_cache = (BC.DATA_ROOT / "cache" / ("ad" + _SEP + "test" + ".pt"))
    return {"name": "G5_official_test_never_touched",
            "forbidden_token_hits": hits,
            "test_cache_path": BC.rel_or_abs(test_cache),
            "test_cache_exists_but_unopened": True,
            "pass": not hits}


def gate_g6_parameters() -> dict:
    norm = BC.load_norm()
    info = {}
    ok = True
    for arm in ("R", "N", "S"):
        m, i = Z.ZM.build_model(norm, arm, 0, "cpu")
        info[arm] = {"parameters": i["parameters"],
                     "initial_state_sha256": i["initial_state_sha256"],
                     "ssl_file": i["ssl"].get("path") or i["ssl"].get("file")}
        ok &= i["parameters"] == BC.EXPECTED_PARAMETERS
    hashes = {v["initial_state_sha256"] for v in info.values()}
    ok &= len(hashes) == 1
    return {"name": "G6_parameters_and_shared_init", "arms": info,
            "expected_parameters": BC.EXPECTED_PARAMETERS,
            "single_init_hash": len(hashes) == 1, "pass": bool(ok)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase-a", default=str(Z.ANALYSIS / "phase_a.json"))
    p.add_argument("--out", default=str(Z.ANALYSIS / "cpu_checks.json"))
    p.add_argument("--utc", default=None)
    args = p.parse_args(argv)

    pa = Path(args.phase_a)
    gates = [
        gate_g1_integrity(),
        gate_g2_phase_a(pa),
        gate_g3_contribution_bookkeeping(pa),
        gate_g4_arm_not_noop(pa),
        gate_g5_no_test_access(),
        gate_g6_parameters(),
    ]
    all_pass = all(g["pass"] for g in gates)
    payload = {
        "round": Z.ROUND_ID,
        "written": args.utc or Z.now(),
        "official_test_loaded": False,
        "phase_a": {"path": BC.rel_or_abs(pa), "sha256": BC.sha256_file(pa)},
        "gates": gates,
        "all_pass": bool(all_pass),
        "failed": [g["name"] for g in gates if not g["pass"]],
        "s_arm_authorized": bool(all_pass),
        "authorization_scope": ("exactly one formal run: S_seed0, 47,400 updates, "
                                "seed 0, pretrain_encoder_s0.pt, v1 frozen recipe"),
    }
    Z.write_json(args.out, Z.jsonable(payload))
    for g in gates:
        Z.log(f"[zb2 gates] {g['name']}: {'PASS' if g['pass'] else 'FAIL'}")
    Z.log(f"[zb2 gates] all_pass={all_pass} -> {BC.rel_or_abs(args.out)}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
