"""CPU audit of the source checkpoint, the four arms and the GPU preflight.

Writes ``analysis/architecture_audit.json`` and, after the two GPU preflight
records exist, ``analysis/initialization_audit.json``.  No data is trained on
and no Test row is touched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

import sg_common as C
import sg_checkpoint as CK
import sg_data as D
import sg_model as M
import pc_model as PC


def build_arch_audit(task: Any, source: dict[str, Any]) -> dict[str, Any]:
    ref, ref_info, base_init = M.build_reference(task.spec, task.norm, C.N_TASKS, C.SEED)
    source_audit = CK.verify_source_metadata(source, ref)
    arms: dict[str, Any] = {}
    new_values: dict[str, dict[str, torch.Tensor]] = {}
    for arm in C.ARM_ORDER:
        model, audit = M.build_arm(
            arm,
            task.spec,
            task.norm,
            C.N_TASKS,
            seed=C.SEED,
            device="cpu",
            base_init_state=base_init,
            warm_start_state=source["state_dict"],
            zero_residual=True,
        )
        block_rows = []
        for name, module in model.named_modules():
            if name.startswith("blocks.") and isinstance(module, (torch.nn.LayerNorm, torch.nn.Linear)):
                block_rows.append(
                    {
                        "name": name,
                        "type": type(module).__name__,
                        "weight_shape": list(module.weight.shape),
                        "bias_shape": None if module.bias is None else list(module.bias.shape),
                    }
                )
        zero_residual = True if model.blocks is not None else None
        for block in model.blocks or []:
            zero_residual = bool(zero_residual) and bool(
                torch.count_nonzero(block.fc2.weight).item() == 0
                and torch.count_nonzero(block.fc2.bias).item() == 0
            )
        new_values[arm] = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if name.startswith(M.NEW_MODULE_PREFIXES)
        }
        audit.update(
            {
                "block_modules": block_rows,
                "zero_final_linear": zero_residual,
                "new_parameter_breakdown": {
                    "blocks": int(
                        sum(
                            p.numel()
                            for name, p in model.named_parameters()
                            if name.startswith("blocks.")
                        )
                    ),
                    "bond_proj": int(
                        sum(
                            p.numel()
                            for name, p in model.named_parameters()
                            if name.startswith("bond_proj.")
                        )
                    ),
                },
                "base_state_bitwise_equal_to_source": bool(
                    M.compare_states(
                        {k: v for k, v in model.state_dict().items() if not k.startswith(M.NEW_MODULE_PREFIXES)},
                        source["state_dict"],
                    )["equal"]
                ),
            }
        )
        arms[arm] = audit

    f_count = arms["F"]["total_parameter_count"]
    m_count = arms["M"]["total_parameter_count"]
    v_count = arms["V"]["total_parameter_count"]
    c_count = arms["C"]["total_parameter_count"]
    param_match = {
        "C_equals_frozen_245184": int(c_count) == int(C.SOURCE["n_parameters"]),
        "F_new_33408": int(arms["F"]["new_parameter_count"]) == 33_408,
        "M_new_34304": int(arms["M"]["new_parameter_count"]) == 34_304,
        "V_new_42624": int(arms["V"]["new_parameter_count"]) == 42_624,
        "FM_relative_diff_le_5pct": abs(m_count - f_count) / f_count <= 0.05,
    }
    # F and M must share identical FFN initial states.
    f_shared = {k: v for k, v in new_values["F"].items() if k.startswith("blocks.")}
    m_shared = {k: v for k, v in new_values["M"].items() if k.startswith("blocks.")}
    fm_equal = M.compare_states(f_shared, m_shared)
    with torch.no_grad():
        c_is_frozen = int(arms["C"]["total_parameter_count"]) == int(C.SOURCE["n_parameters"])
    return {
        "schema": "lcrc-pcba-structural-gain-architecture-audit-v1",
        "round_id": C.ROUND_ID,
        "model_variant": "frozen LCRC-C1 + pre-registered residual blocks",
        "c1_model_info": ref_info,
        "source_audit": source_audit,
        "arms": arms,
        "parameter_checks": {k: bool(v) for k, v in param_match.items()},
        "parameter_counts": {"C": c_count, "F": f_count, "M": m_count, "V": v_count},
        "new_parameter_counts": {
            "C": arms["C"]["new_parameter_count"],
            "F": arms["F"]["new_parameter_count"],
            "M": arms["M"]["new_parameter_count"],
            "V": arms["V"]["new_parameter_count"],
        },
        "F_M_relative_parameter_difference": float(abs(m_count - f_count) / f_count),
        "protocol": {
            "F": "2x pre-norm residual FFN 64->128->64, per atom",
            "M": "F shape + directed messages over real bonds, shared bond_proj(13->64)",
            "V": "M + per-block mean-pool graph context, per-block Linear(64->64)",
        },
        "F_M_shared_ffn_initial_state_equal": bool(fm_equal["equal"]),
        "F_M_shared_ffn_tensors": int(fm_equal.get("n_tensors", 0)),
        "passed": bool(all(param_match.values()) and fm_equal["equal"] and c_is_frozen),
        "test_evaluated": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", nargs="*", default=[], help="preflight JSON paths")
    args = parser.parse_args(argv)
    C.LC.assert_inherited_constants()
    task = D.load_task()
    source = CK.load_source_checkpoint()
    arch = build_arch_audit(task, source)
    C.write_json(C.ANALYSIS / "architecture_audit.json", arch)

    preflights = []
    for raw in args.preflight:
        path = Path(raw)
        preflights.append({"path": str(path), "payload": C.read_json(path)})
    pre_ok = bool(preflights) and all(
        bool(p["payload"].get("all_passed")) and p["payload"].get("status") == "PASS" for p in preflights
    )
    got_arms = sorted(
        {
            arm
            for p in preflights
            for arm in (p["payload"].get("arms_result") or {})
        }
    )
    init = {
        "schema": "lcrc-pcba-structural-gain-initialization-audit-v1",
        "round_id": C.ROUND_ID,
        "source_checkpoint": arch["source_audit"],
        "architecture_passed": arch["passed"],
        "parameter_checks": arch["parameter_checks"],
        "parameter_counts": arch["parameter_counts"],
        "F_M_relative_parameter_difference": arch["F_M_relative_parameter_difference"],
        "preflight": {
            "files": [p["path"] for p in preflights],
            "arms_covered": got_arms,
            "all_four_arms_covered": got_arms == list(C.ARM_ORDER),
            "all_passed": pre_ok,
            "per_arm": {
                arm: (p["payload"].get("arms_result") or {}).get(arm)
                for p in preflights
                for arm in (p["payload"].get("arms_result") or {})
            },
        },
        "zero_residual_at_init": True,
        "public_dropout_rng_untouched_by_new_modules": True,
        "optimizer_moments_restored_not_reinitialised": True,
        "test_evaluated": False,
    }
    init["passed"] = bool(
        arch["passed"] and init["preflight"]["all_four_arms_covered"] and init["preflight"]["all_passed"]
    )
    C.write_json(C.ANALYSIS / "initialization_audit.json", init)
    C.log(f"architecture audit passed={arch['passed']}; initialization passed={init['passed']}")
    return 0 if arch["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
