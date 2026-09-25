"""Independent local analysis of the E2E-DictEnv-Purify-v0 artifacts.

Recomputes, from the *pulled* files only, what the frozen preregistration
declares: the parameter ledger, the Stage-0 equivalence evidence, both arm
soups, ``Delta_purification`` per seed, the seed-0 case, the seed-1 purchase
authorization, the two-seed verdict, the mechanism gates, the constant-channel
ablation and the purity counts.  Nothing is re-trained and no CUDA is touched.

Exit code 1 if any recomputed value disagrees with the recorded artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
TRACK = REPO_ROOT / "tracks/ksvd"
RESULTS = TRACK / "results/e2e_dictenv_purify_v0"

# frozen pins, repeated literally so this script cannot follow a moved goalpost
REFERENCE_PARAMS = 97487
PURIFIED_PARAMS = 95711
H1_HISTORICAL_SOUP = 0.12354862861608853
REPRODUCTION_TOLERANCE = 0.005
SEED0_PASS = 0.002
SEED0_AMBIGUOUS_MAX = 0.004
MEAN_PASS = 0.002
WORST_SEED_TOLERANCE = 0.004
GATE_ZERO = 0.030
GATE_NODE = 0.010
GATE_ALL = 0.015
HORIZON = 320
EQUIVALENCE_TOLERANCE = 1.0e-6


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def seed0_case(delta: float) -> str:
    if delta <= SEED0_PASS:
        return "non_inferior"
    if delta <= SEED0_AMBIGUOUS_MAX:
        return "ambiguous"
    return "performance_failure"


def two_seed_verdict(mean_delta: float, worst_delta: float) -> str:
    if mean_delta <= MEAN_PASS and worst_delta <= WORST_SEED_TOLERANCE:
        return "PURIFIED_ARCHITECTURE_ACCEPTED"
    if mean_delta <= SEED0_AMBIGUOUS_MAX:
        return "PURIFICATION_INCONCLUSIVE"
    return "PURIFICATION_REJECTED"


def analyze() -> dict[str, Any]:
    errors: list[str] = []
    checks = 0

    def check(name: str, condition: bool, detail: Any) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(f"{name}: {detail}")

    ledger = read_json(RESULTS / "parameter_ledger.json")
    check("ledger.reference", int(ledger["reference"]["whole_model"]) == REFERENCE_PARAMS, ledger["reference"]["whole_model"])
    check("ledger.purified", int(ledger["purified"]["whole_model"]) == PURIFIED_PARAMS, ledger["purified"]["whole_model"])
    check("ledger.rule", ledger["purified"]["whole_model"] <= ledger["reference"]["whole_model"], ledger["purified"]["whole_model"])
    check(
        "ledger.actual",
        ledger.get("actual_parameters", {}).get("reference") == REFERENCE_PARAMS
        and ledger.get("actual_parameters", {}).get("purified") == PURIFIED_PARAMS,
        ledger.get("actual_parameters"),
    )
    check("ledger.delta", int(ledger["delta"]["absolute"]) == PURIFIED_PARAMS - REFERENCE_PARAMS, ledger["delta"])

    equivalence = read_json(RESULTS / "semantic_refactor_equivalence.json")
    cpu_block = equivalence["blocks"].get("cpu", {})
    check("equiv.cpu_bit_identical", bool(cpu_block.get("bit_identical")), cpu_block.get("comparisons_max_abs"))
    check("equiv.cpu_max_abs", float(cpu_block.get("max_abs", 1.0)) <= EQUIVALENCE_TOLERANCE, cpu_block.get("max_abs"))
    check("equiv.prediction", float(cpu_block.get("comparisons_max_abs", {}).get("prediction", 1.0)) <= EQUIVALENCE_TOLERANCE, cpu_block)
    check("equiv.all_blocks_passed", bool(equivalence.get("passed")), equivalence.get("blocks", {}).keys())

    purity = read_json(RESULTS / "purity_audit.json")
    check("purity.reference_bypasses", int(purity["reference"]["local_raw_chemistry_bypass_count"]) == 3, purity["reference"])
    check("purity.purified_bypasses", int(purity["purified"]["local_raw_chemistry_bypass_count"]) == 0, purity["purified"])
    check("purity.entry_points", int(purity["reference"]["local_chemistry_entry_points"]) == 5 and int(purity["purified"]["local_chemistry_entry_points"]) == 2, purity)

    arms: dict[str, dict[str, Any]] = {}
    for arm in ("reference", "purified"):
        for seed in (0, 1):
            path = RESULTS / f"{arm}_seed{seed}.json"
            if path.exists():
                arms[f"{arm}_seed{seed}"] = read_json(path)
    for key, payload in arms.items():
        check(f"{key}.horizon", int(payload["horizon"]) == HORIZON, payload["horizon"])
        check(f"{key}.test_blocked", payload["official_test_loaded"] is False, payload["official_test_loaded"])
        check(f"{key}.lambda", abs(float(payload["lambda_rec"]) - 33.95873017865987) < 1e-9, payload["lambda_rec"])
        expected_params = REFERENCE_PARAMS if key.startswith("reference") else PURIFIED_PARAMS
        check(f"{key}.params", int(payload["actual_params"]) == expected_params, payload["actual_params"])
        check(f"{key}.soup_len", len(payload["soup"]["members"]) == 5, payload["soup"]["members"])

    seeds = sorted({int(key.rsplit("seed", 1)[1]) for key in arms})
    deltas = {}
    for seed in seeds:
        if f"reference_seed{seed}" in arms and f"purified_seed{seed}" in arms:
            deltas[seed] = float(
                arms[f"purified_seed{seed}"]["soup"]["soup_valid_mae"]
                - arms[f"reference_seed{seed}"]["soup"]["soup_valid_mae"]
            )
    check("seeds.seed0_present", 0 in deltas, seeds)

    recorded = read_json(RESULTS / "paired_performance.json")
    for seed, delta in deltas.items():
        check(f"paired.delta_seed{seed}", abs(float(recorded["delta"][str(seed)]) - delta) < 1e-12, (recorded["delta"], delta))
    reference_drift = float(recorded["reference_soup"]["0"] - H1_HISTORICAL_SOUP)
    check("paired.reproduction_value", abs(float(recorded["reference_reproduction"]["drift"]) - reference_drift) < 1e-12, (recorded["reference_reproduction"]["drift"], reference_drift))
    check(
        "paired.reproduction_flag",
        bool(recorded["reference_reproduction"]["passed"]) == (abs(reference_drift) <= REPRODUCTION_TOLERANCE),
        (recorded["reference_reproduction"]["passed"], reference_drift),
    )
    if deltas:
        check("paired.seed0_case", recorded["seed0_case"] == seed0_case(deltas[0]), (recorded["seed0_case"], deltas[0]))
        authorized = bool(recorded["reference_reproduction"]["passed"] and seed0_case(deltas[0]) != "performance_failure")
        check("paired.seed1_authorized", bool(recorded["seed1_authorized"]) == authorized, recorded["seed1_authorized"])
        seeds_run = sorted(int(key.rsplit("seed", 1)[1]) for key in arms if key.startswith("reference"))
        if not authorized:
            check("stop.no_seed1_artifacts", seeds_run == [0], seeds_run)
            check("stop.no_mechanism", not (RESULTS / "mechanism_interventions.json").exists(), "mechanism artifact present after a stop")
            check("stop.no_ablation", not (RESULTS / "constant_channel_ablation.json").exists(), "ablation artifact present after a stop")
            check("stop.no_health", not (RESULTS / "dictionary_health.json").exists(), "health artifact present after a stop")
    if len(deltas) > 1:
        mean_delta = sum(deltas.values()) / len(deltas)
        worst = max(deltas.values())
        check("paired.mean", abs(float(recorded["mean_delta"]) - mean_delta) < 1e-12, (recorded["mean_delta"], mean_delta))
        check("paired.worst", abs(float(recorded["worst_seed_delta"]) - worst) < 1e-12, (recorded["worst_seed_delta"], worst))
        check("paired.verdict", recorded["verdict"] == two_seed_verdict(mean_delta, worst), (recorded["verdict"], mean_delta, worst))

    mechanism = read_json(RESULTS / "mechanism_interventions.json") if (RESULTS / "mechanism_interventions.json").exists() else None
    if mechanism is not None:
        degradation = mechanism["degradation"]
        check("mechanism.zero", float(degradation["zero"]) >= GATE_ZERO, degradation["zero"])
        check("mechanism.node", float(degradation["node"]) >= GATE_NODE, degradation["node"])
        check("mechanism.all", float(degradation["all"]) >= GATE_ALL, degradation["all"])
        check("mechanism.clean", abs(float(mechanism["clean_soup_valid_mae"]) - float(arms["purified_seed0"]["soup"]["soup_valid_mae"])) < 1e-9, mechanism["clean_soup_valid_mae"])
        check(
            "mechanism.preserved_flag",
            bool(mechanism["mechanism_preserved"])
            == (float(degradation["zero"]) >= GATE_ZERO and float(degradation["node"]) >= GATE_NODE and float(degradation["all"]) >= GATE_ALL),
            mechanism["mechanism_preserved"],
        )

    ablation = read_json(RESULTS / "constant_channel_ablation.json") if (RESULTS / "constant_channel_ablation.json").exists() else None
    if ablation is not None:
        check("ablation.clean", abs(float(ablation["clean_soup_valid_mae"]) - float(arms["purified_seed0"]["soup"]["soup_valid_mae"])) < 1e-9, ablation["clean_soup_valid_mae"])
        for tag in ("node_off", "edge_off", "both_off"):
            check(f"ablation.{tag}", tag in ablation["ablations"], sorted(ablation["ablations"]))

    decision = read_json(RESULTS / "decision.json")
    check("decision.test_blocked", decision["official_test_loaded"] is False, decision["official_test_loaded"])
    check("decision.verdict_present", isinstance(decision["verdict"], str) and decision["verdict"], decision["verdict"])
    if not recorded["reference_reproduction"]["passed"]:
        check("decision.stop_verdict", decision["verdict"] == "REFERENCE_REPRODUCTION_FAILURE", decision["verdict"])
        check("decision.candidate_not_interpreted", decision["mechanism"] is None and decision["dictionary_health"] is None, decision["verdict"])
        not_run = {record["artifact"] for record in decision["not_run"]}
        check(
            "decision.not_run_complete",
            {"reference_seed1.json", "purified_seed1.json", "mechanism_interventions.json", "constant_channel_ablation.json", "dictionary_health.json"} <= not_run,
            sorted(not_run),
        )

    state_protocol = None
    h1_state = TRACK / "results/e2e_dictenv_p2_abs/states/H1_soup_state.pt"
    if h1_state.exists() and (results_ok := not errors):
        import torch
        from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
        from tracks.ksvd.experiments.luyin16 import e2e_dictenv_purify_v0 as pur
        from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_purify_v0 as run

        D, _sha = run.dictionary()
        loader = p1.make_env_loader(run.load_split("valid"), 128, False, run.EVAL_SHUFFLE_OFFSET)
        model = pur.build_model(pur.reference_config(), D, seed=0).eval()
        pur.load_p2_state(model, torch.load(h1_state, map_location="cpu", weights_only=False))
        value = float(run.evaluate(model, loader, torch.device("cpu"))["mae"])
        check("diagnosis.evaluator_identical", abs(value - H1_HISTORICAL_SOUP) <= 1e-8, (value, H1_HISTORICAL_SOUP))
        state_protocol = value

    seed1_status = "RUN" if 1 in deltas else "NOT RUN"
    return {
        "checks": checks,
        "errors": errors,
        "ok": not errors,
        "outcome": "consistent" if not errors else "inconsistent",
        "seeds_available": seeds,
        "delta": deltas,
        "recorded_verdict": decision["verdict"],
        "seed1_status": seed1_status,
        "seed1_authorized": recorded.get("seed1_authorized"),
        "reference_reproduction_drift": reference_drift,
        "mechanism_degradation": None if mechanism is None else mechanism["degradation"],
        "ablation": None if ablation is None else {tag: ablation["ablations"][tag]["mae_delta"] for tag in ablation["ablations"]},
        "h1_state_under_round_evaluator": state_protocol,
        "candidate_not_interpreted": bool(decision["verdict"] == "REFERENCE_REPRODUCTION_FAILURE"),
        "purity": {
            "reference_bypasses": purity["reference"]["local_raw_chemistry_bypass_count"],
            "purified_bypasses": purity["purified"]["local_raw_chemistry_bypass_count"],
            "reference_params": purity["reference"]["trainable_parameters"],
            "purified_params": purity["purified"]["trainable_parameters"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write results/e2e_dictenv_purify_v0/local_analysis.json")
    args = parser.parse_args(argv)
    payload = analyze()
    if args.write:
        (RESULTS / "local_analysis.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
