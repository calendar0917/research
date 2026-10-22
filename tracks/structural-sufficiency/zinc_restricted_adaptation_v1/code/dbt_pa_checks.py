"""Narrow correctness checks for DBT-PA (CPU, a few valid graphs).

Checks
------
1. inference parameter count split into frozen / trainable; cap <= 500k;
2. original DBT modules have the same seed-0 initial state as the original
   model (adapters are appended afterwards and do not shift those draws);
3. zero-initialised adapters reproduce the no-adapter DBT downstream exactly;
4. frozen encoder hash is unchanged after several optimizer steps, while the
   adapters and DBT trunk change; the frozen stack is not in the optimizer;
5. the adapter ``D`` receives zero gradient on the first step (expected) and a
   non-zero gradient after ``U`` has moved;
6. cached-feature gather is row-aligned with a direct local-stack forward;
7. LR schedule endpoints and 157 updates/epoch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
_DBT = _SSUF / "zinc_directed_bond_transformer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_DBT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import dbt_data as BD  # noqa: E402
import dbt_models as BM  # noqa: E402
import dbt_pa_cache as PC  # noqa: E402
import dbt_pa_models as PM  # noqa: E402
import dbt_pa_train as BT  # noqa: E402

ANALYSIS = _EXP / "analysis"


def _sha_module(model: torch.nn.Module, prefix: str) -> str:
    return PM.state_dict_sha256(
        {k: v for k, v in model.state_dict().items() if k.startswith(prefix)}
    )


def run_checks() -> dict[str, Any]:
    device = torch.device("cpu")
    norm = D.load_norm()
    scales_path = ANALYSIS / "graph_scales.json"
    if not scales_path.exists():
        scales_path = _SSUF / "zinc_directed_bond_transformer_v1" / "analysis" / "graph_scales.json"
    scales = C.read_json(scales_path)

    pa, info = PM.build_model_pa(norm, 0, device, graph_scales=scales, dropout=0.05,
                                 adapter_rank=8)
    pa.eval()
    # Reference model with the same seed-0 construction order and no adapters.
    torch.manual_seed(0)
    ref = BM.DirectedBondTransformer(norm, dropout=0.05, graph_scales=scales).to(device)
    ref.eval()

    pa_state = pa.state_dict()
    ref_state = ref.state_dict()
    common = [k for k in ref_state if k in pa_state and not k.startswith("encoders.")]
    missing = [k for k in common if k not in pa_state]
    init_max = max((float((pa_state[k].cpu() - ref_state[k].cpu()).abs().max())
                    for k in common), default=0.0)
    init_equal = all(torch.equal(pa_state[k].cpu(), ref_state[k].cpu()) for k in common)

    optimizer = torch.optim.AdamW(pa.trainable_parameters(), lr=3e-4, weight_decay=1e-4)
    audit = BT._param_audit(pa, optimizer)

    valid_obj = BD.load_object_cache("valid")
    valid_rel = BD.load_relation_split("valid")
    rows = np.arange(8, dtype=np.int64)
    batch = BD.collate(valid_obj, valid_rel, rows, valid=True)
    with torch.no_grad():
        a0, b0, w0 = pa.encoders(batch)
        # Synthetic row-aligned cache: concatenated base outputs in row order.
        ptr_a = np.concatenate([[0], np.cumsum([int((batch["atom_ptr"][i + 1] - batch["atom_ptr"][i]))
                                                for i in range(len(rows))])]).astype(np.int64)
        # ``batch["atom_ptr"]`` is already the batch-local pointer.
        ptr_a = np.asarray(batch["atom_ptr"], dtype=np.int64)
        ptr_b = np.concatenate([[0], np.cumsum(
            np.bincount(batch["bond_batch"].numpy(), minlength=len(rows))
        )]).astype(np.int64)
        ptr_w = np.concatenate([[0], np.cumsum(
            np.bincount(batch["junc_batch"].numpy(), minlength=len(rows))
        )]).astype(np.int64)
        mini = {
            "a": a0.numpy(), "b": b0.numpy(), "w": w0.numpy(),
            "atom_ptr": ptr_a, "bond_ptr": ptr_b, "junc_ptr": ptr_w,
        }
        ga, gb, gw = PC.gather_cached_features(mini, rows, device)
        gather_max = max(float((a0 - ga).abs().max()), float((b0 - gb).abs().max()),
                         float((w0 - gw).abs().max()))
        with torch.no_grad():
            base_pred = ref.forward_from_local(batch, a0, b0, w0)
            pa_pred = pa.forward_cached(batch, a0, b0, w0)
        zero_adapter_max = float((base_pred - pa_pred).abs().max())

    y = torch.as_tensor(np.asarray(valid_obj["labels"], dtype=np.float64)[rows],
                        dtype=torch.float32)
    encoder_before = _sha_module(pa, "encoders.")
    adapter_before = _sha_module(pa, "adapter_a.")
    trunk_before = _sha_module(pa, "proj_src.")
    grad_records = []
    pa.train()
    base_in_optimizer = any(p.requires_grad for p in pa.encoders.parameters())
    for step in range(1, 4):
        optimizer.zero_grad(set_to_none=True)
        pred = pa.forward_cached(batch, a0, b0, w0)
        loss = F.l1_loss(pred, y)
        loss.backward()
        gd = pa.adapter_a.down.weight.grad
        gu = pa.adapter_a.up.weight.grad
        gt = pa.proj_src.weight.grad
        base_grads = [p.grad is not None for p in pa.encoders.parameters()]
        grad_records.append({
            "step": step,
            "loss": float(loss.detach()),
            "adapter_D_grad_abs_sum": float(gd.abs().sum()) if gd is not None else None,
            "adapter_U_grad_abs_sum": float(gu.abs().sum()) if gu is not None else None,
            "trunk_proj_src_grad_abs_sum": float(gt.abs().sum()) if gt is not None else None,
            "any_frozen_encoder_grad": bool(any(base_grads)),
        })
        optimizer.step()

    checks = {
        "device": "cpu",
        "model_info": {
            "ssl_source": info["ssl"]["source"],
            "ssl_sha256": info["ssl"]["sha256"],
            "ssl_tensor_match": info["ssl_tensor_match"],
            "parameters_total_inference": info["parameters"],
            "parameters_trainable": info["trainable_parameters"],
            "parameters_frozen": info["frozen_parameters"],
            "module_breakdown": info["module_breakdown"],
        },
        "parameter_audit": audit,
        "parameter_check": {
            "expected_total": 457443,
            "actual_total": int(info["parameters"]),
            "matches_estimate": bool(int(info["parameters"]) == 457443),
            "under_cap": bool(int(info["parameters"]) <= 500000),
            "frozen_is_202048": bool(int(info["frozen_parameters"]) == 202048),
            "trainable_is_255395": bool(int(info["trainable_parameters"]) == 255395),
            "frozen_stack_requires_grad": bool(base_in_optimizer),
        },
        "init_parity_with_original_dbt_seed0": {
            "common_non_encoder_keys": int(len(common)),
            "missing_keys": missing,
            "max_abs_diff": init_max,
            "equal": bool(init_equal),
        },
        "zero_adapter_equivalence": {
            "max_abs_diff_vs_no_adapter": zero_adapter_max,
            "equal_atol_1e_6": bool(zero_adapter_max <= 1e-6),
        },
        "cached_gather_alignment": {
            "max_abs_diff": gather_max,
            "equal_atol_1e_6": bool(gather_max <= 1e-6),
            "rows": [int(x) for x in rows],
        },
        "frozen_base_after_three_steps": {
            "encoder_sha_before": encoder_before,
            "encoder_sha_after": _sha_module(pa, "encoders."),
            "unchanged": bool(encoder_before == _sha_module(pa, "encoders.")),
            "encoder_stack_training_flag": bool(pa.encoders.training),
        },
        "adapters_and_trunk_update": {
            "adapter_sha_before": adapter_before,
            "adapter_sha_after": _sha_module(pa, "adapter_a."),
            "adapter_updated": bool(adapter_before != _sha_module(pa, "adapter_a.")),
            "trunk_sha_before": trunk_before,
            "trunk_sha_after": _sha_module(pa, "proj_src."),
            "trunk_updated": bool(trunk_before != _sha_module(pa, "proj_src.")),
        },
        "adapter_D_gradient": {
            "records": grad_records,
            "D_zero_on_first_step": bool(grad_records[0]["adapter_D_grad_abs_sum"] == 0.0),
            "D_nonzero_by_step_3": bool(grad_records[-1]["adapter_D_grad_abs_sum"] > 0.0),
            "U_nonzero_on_first_step": bool(grad_records[0]["adapter_U_grad_abs_sum"] > 0.0),
            "no_encoder_gradients": bool(not any(r["any_frozen_encoder_grad"] for r in grad_records)),
        },
        "schedule": {
            "steps_per_epoch": int(BT.STEPS_PER_EPOCH),
            "steps_per_epoch_is_157": bool(BT.STEPS_PER_EPOCH == 157),
            "lr_step_1": float(BT._lr_at(1, 235500, BT.RECIPE)),
            "lr_step_1000": float(BT._lr_at(1000, 235500, BT.RECIPE)),
            "lr_step_235500": float(BT._lr_at(235500, 235500, BT.RECIPE)),
            "warmup_monotone": bool(BT._lr_at(1, 235500, BT.RECIPE)
                                    < BT._lr_at(1000, 235500, BT.RECIPE)),
        },
    }
    checks["all_pass"] = bool(
        checks["parameter_check"]["matches_estimate"]
        and checks["parameter_check"]["under_cap"]
        and checks["parameter_check"]["frozen_is_202048"]
        and checks["parameter_check"]["trainable_is_255395"]
        and not checks["parameter_check"]["frozen_stack_requires_grad"]
        and checks["init_parity_with_original_dbt_seed0"]["equal"]
        and checks["zero_adapter_equivalence"]["equal_atol_1e_6"]
        and checks["cached_gather_alignment"]["equal_atol_1e_6"]
        and checks["frozen_base_after_three_steps"]["unchanged"]
        and not checks["frozen_base_after_three_steps"]["encoder_stack_training_flag"]
        and checks["adapters_and_trunk_update"]["adapter_updated"]
        and checks["adapters_and_trunk_update"]["trunk_updated"]
        and checks["adapter_D_gradient"]["D_zero_on_first_step"]
        and checks["adapter_D_gradient"]["D_nonzero_by_step_3"]
        and checks["adapter_D_gradient"]["U_nonzero_on_first_step"]
        and checks["adapter_D_gradient"]["no_encoder_gradients"]
        and checks["schedule"]["steps_per_epoch_is_157"]
        and checks["schedule"]["warmup_monotone"]
    )
    C.write_json(ANALYSIS / "cpu_checks.json", checks)
    C.log(json.dumps(checks, indent=2))
    if not checks["all_pass"]:
        raise RuntimeError("DBT-PA CPU checks failed")
    return checks


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args(argv)
    run_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
