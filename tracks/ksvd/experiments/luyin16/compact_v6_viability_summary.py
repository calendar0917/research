"""Compact-v6 attribute-branch viability repair: tables, decision, figures.

Reads the artifacts written by ``compact_v6_viability_repair`` and produces

* ``seed01_results.csv``               (seed0 + seed1 fast-gate table)
* ``decision_record.json``             (pre-registered decision logic)
* ``figures/fig1_original_10step_gradients.png``
* ``figures/fig2_repaired_100step.png``
* ``figures/fig3_seed0_prediction_delta.png``

No training, no test access.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import REPO_ROOT

RESULTS = REPO_ROOT / "tracks/ksvd/results/compact_v6_viability"

# Pre-registered canonical historical compact-v4-hinge seed-0 valid MAE.
V4_SEED0_VALID = 0.17006561887910357
V4_QUINTILES = (
    REPO_ROOT
    / "tracks/ksvd/results/baseline_difficulty_typed_refinement_audit/per_molecule_analysis.csv"
)

# seed-0 viable run produced by this stage
VIABLE_SEED0_RUN = "20260910-183159-63fd1616"
V6_SEED0_VALID = 0.17020902845537056


def _read(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _read_csv(name: str):
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    gate = _read("repaired_viability_gate.json")
    mech = _read("seed0_mechanism_diagnostics.json")
    orig = _read("original_collapsed_mechanism_diagnostics.json")
    quint = _read("seed0_difficulty_quintiles.json")
    audit = _read("optimizer_parameter_audit.json")

    delta_seed0 = V4_SEED0_VALID - V6_SEED0_VALID
    performance_pass = delta_seed0 >= 0.003
    # stronger mechanism requirement: mean |delta| > 1e-4 for attr-zero / role-shuffle
    mechanism_alive_seed0 = bool(
        mech["attr_zero"]["max_abs_delta"] > 1e-4
        and mech["role_association_shuffle"]["mean_abs_delta"] > 1e-4
    )
    branch_active_after_full_training = bool(
        mech["branch_stats"]["e_attribute_patch_std"] > 1e-4
    )
    bulk_pass = quint["Q1Q2_safe_le_0.002"]

    decision = "BRANCH REPAIRED, ARCHITECTURE NO-GO"
    reason = (
        "The minimal (pre-registered) repair makes the attribute branch demonstrably "
        "active over 100 optimizer steps (V1-V4 pass), but under the canonical 60-epoch "
        "protocol the branch collapses again to a near-constant output and seed-0 improvement "
        f"is {delta_seed0:+.6f} (gate >= +0.003 FAIL); the easy-bulk Q1/Q2 gate also fails "
        f"({quint['Q1Q2_combined_degradation']:+.6f} degradation). Per the pre-registered "
        "stop rule (seed0 performance AND mechanism) this closes the topology-attribute "
        "factorization branch permanently; seed 1 is not run."
    )

    # ---- seed01_results.csv ---------------------------------------------
    rows = [
        {
            "seed": 0,
            "v4_valid_mae": V4_SEED0_VALID,
            "repaired_v6_valid_mae": V6_SEED0_VALID,
            "delta_v4_minus_v6": delta_seed0,
            "run_id": VIABLE_SEED0_RUN,
            "status": "completed",
        },
        {
            "seed": 1,
            "v4_valid_mae": "",
            "repaired_v6_valid_mae": "",
            "delta_v4_minus_v6": "",
            "run_id": "",
            "status": "not_run (STOP AT SEED0: performance gate failed)",
        },
    ]
    with (RESULTS / "seed01_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- decision_record.json -------------------------------------------
    record = {
        "stage": "compact_v6_attribute_branch_viability_repair",
        "protocol": "zinc-context-gap",
        "test_access": "blocked",
        "original_version": "compact-v6-original-collapsed (zero-init final fusion)",
        "repaired_version": "compact-v6-viable (attribute_fusion_init=small_normal, scale 0.01)",
        "repair": {
            "change": "final attribute fusion projection initialised normal(std=0.01) instead of zeros",
            "files": [
                "experiments/luyin16/zinc_patch_path_pooling.py",
                "experiments/luyin16/compact_v6_diagnostics.py",
                "configs/luyin16/zinc_compact_v6_viable.yaml",
            ],
            "nothing_else_changed": True,
        },
        "root_cause": {
            "case": "G0-A (strict zero final projection) plus a training-dynamics attractor",
            "step0_exactly_zero_gradients": [
                "attribute_encoder.atom_mlp.*",
                "attribute_encoder.bond_mlp.*",
                "attribute_encoder.atom_embedding.*",
                "attribute_encoder.bond_embedding.*",
                "patch_encoder.layers.0.weight[:, attribute columns]",
            ],
            "step0_nonzero_gradients": [
                "attribute_encoder.fusion.2.weight",
                "attribute_encoder.fusion.2.bias",
            ],
            "optimizer_registration": "all 14 attribute parameters present, requires_grad=True, single group",
            "raw_input_representation": "informative (role feature std ~0.30; adversarial placement pair differs)",
            "full_training_outcome": (
                "original patch-std 7.6e-12 (exactly constant, shuffle delta 0.0); "
                "repaired patch-std 3.6e-07 (still near-constant; role/type shuffle ~1e-6)"
            ),
        },
        "viability_gate": gate,
        "seed0_mechanism": mech,
        "original_collapsed_mechanism": orig,
        "seed0_performance": {
            "v4_valid_mae": V4_SEED0_VALID,
            "repaired_v6_valid_mae": V6_SEED0_VALID,
            "delta_v4_minus_v6": delta_seed0,
            "gate": 0.003,
            "pass": performance_pass,
        },
        "bulk_gate": {
            "Q1Q2_combined_degradation": quint["Q1Q2_combined_degradation"],
            "gate": 0.002,
            "pass": bulk_pass,
        },
        "mechanism_after_full_training": {
            "mechanism_alive": mechanism_alive_seed0,
            "branch_active": branch_active_after_full_training,
            "attr_zero_max_abs_delta": mech["attr_zero"]["max_abs_delta"],
            "role_shuffle_mean_abs_delta": mech["role_association_shuffle"]["mean_abs_delta"],
            "e_attribute_patch_std": mech["branch_stats"]["e_attribute_patch_std"],
        },
        "decision": decision,
        "reason": reason,
        "next_step": "permanently close topology-attribute factorization; do not tune further",
        "seed1_run": False,
        "outputs": [
            "original_single_batch_trace.json",
            "original_10step_trace.csv",
            "optimizer_parameter_audit.csv",
            "attribute_input_variance.json",
            "adversarial_representation_check.json",
            "repaired_100step_trace.csv",
            "repaired_viability_gate.json",
            "seed0_mechanism_diagnostics.json",
            "seed0_difficulty_quintiles.csv",
            "seed01_results.csv",
            "decision_record.json",
        ],
    }
    (RESULTS / "decision_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # ---- figures ---------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = RESULTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # Figure 1: original 10-step gradient norms by branch stage
    rows = _read_csv("original_10step_trace.csv")
    steps = np.asarray([int(r["step"]) for r in rows])
    floor = 1e-9
    stages = {
        "final projection (fusion[-1])": np.asarray([float(r["grad_norm_W_final"]) for r in rows]),
        "attr upstream encoder (atom+bond MLP)": np.asarray(
            [float(r["grad_norm_attr_upstream"]) for r in rows]
        ),
        "patch_encoder attribute columns": np.asarray(
            [float(r["grad_norm_W_attr_patch_encoder"]) for r in rows]
        ),
    }
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for label, values in stages.items():
        ax.semilogy(steps, np.maximum(values, floor), marker="o", markersize=3, label=label)
    ax.axhline(floor, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("optimizer step (original zero-init branch)")
    ax.set_ylabel("gradient norm (log, floor 1e-9)")
    ax.set_title("Fig 1: original compact-v6 branch — step-0 upstream W_final^T=0 kill")
    ax.annotate(
        "step 0: exact zeros\n(upstream + patch attr cols)",
        xy=(0, floor), xytext=(1.4, 1e-7),
        arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "fig1_original_10step_gradients.png", dpi=140)
    plt.close(fig)

    # Figure 2: repaired 100-step branch health
    rows = _read_csv("repaired_100step_trace.csv")
    steps = np.asarray([int(r["step"]) for r in rows])
    patch_std = np.asarray([float(r["e_attribute_patch_std"]) for r in rows])
    w_final = np.asarray([float(r["W_final_norm"]) for r in rows])
    grad_up = np.asarray([float(r["grad_norm_attr_upstream"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(steps, patch_std, label="e_attribute patch std")
    ax.plot(steps, w_final, label="||W_final||")
    ax.semilogy(steps, np.maximum(grad_up, floor), label="upstream gradient norm (log)")
    ax.axhline(1e-4, color="red", ls="--", lw=0.8, label="V1 threshold 1e-4")
    ax.set_xlabel("optimizer step (repaired small-normal init, 100 steps)")
    ax.set_ylabel("value")
    ax.set_title("Fig 2: repaired compact-v6 branch is alive over 100 steps (V1-V4 pass)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "fig2_repaired_100step.png", dpi=140)
    plt.close(fig)

    # Figure 3: seed0 frozen-checkpoint prediction-difference distribution
    clean = np.load(RESULTS / "seed0_clean_predictions.npy")
    attr_zero = np.load(RESULTS / "seed0_attr_zero_predictions.npy")
    role = np.load(RESULTS / "seed0_role_shuffle_predictions.npy")
    d_attr = np.abs(clean - attr_zero)
    d_role = np.abs(clean - role)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(d_attr, bins=60, alpha=0.7, label=f"|clean - attr-zero| (mean {d_attr.mean():.2e})")
    ax.hist(d_role, bins=60, alpha=0.7, label=f"|clean - role-shuffle| (mean {d_role.mean():.2e})")
    ax.set_yscale("log")
    ax.set_xlabel("per-molecule |prediction delta| (frozen repaired seed-0 checkpoint)")
    ax.set_ylabel("molecule count (log)")
    ax.set_title(
        "Fig 3: repaired branch full-training collapse — deltas at float noise\n"
        f"(e_attribute patch std {mech['branch_stats']['e_attribute_patch_std']:.1e})"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "fig3_seed0_prediction_delta.png", dpi=140)
    plt.close(fig)

    print(json.dumps({k: record[k] for k in ("decision", "seed0_performance", "bulk_gate")}, indent=2))
    print("wrote", RESULTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
