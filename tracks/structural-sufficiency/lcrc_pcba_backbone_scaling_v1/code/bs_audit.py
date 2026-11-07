"""Pre-training audits for the backbone scaling round.

Modes
-----
``gpu-preflight``   device / NVML / occupancy diagnostics, per-visible-GPU probes.
``model``           the frozen constants, the scale-1 equivalence to historical
                    LCRC-A (keys, parameters, init state, forward), the exact
                    parameter counts, the per-module width audit, the S1/S2
                    input invariance, the one-shot-encoding and no-second-
                    propagation evidence, atom-renumbering invariance, the
                    Train-only normalisation identity, forward/backward health
                    and the Test-isolation gate.
``a0-replay``       load the historical A checkpoint (read-only), re-run the
                    official Validation evaluation, and prove the scale-1
                    constructor reproduces A's logits and AP.
``eval-probe``      a full 43,793-row Validation forward pass and real training
                    steps for W2 and W3, the cost projection, and the resolved
                    Validation cadence under the frozen 15 % rule.

Nothing here trains a model or touches an official Test row.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import bs_common as C
import bs_model as M
import torch

import tu_model as TU


# ---------------------------------------------------------------------------
# frozen constants
# ---------------------------------------------------------------------------


def audit_constants() -> dict[str, Any]:
    recipe = C.RECIPE
    expected = {
        "optimizer": "AdamW",
        "learning_rate": 3.0e-4,
        "warmup_start_lr": 3.0e-5,
        "end_lr": 1.0e-5,
        "warmup_epochs": 1.0,
        "weight_decay": 1.0e-3,
        "adam_betas": (0.9, 0.999),
        "adam_eps": 1.0e-8,
        "batch_size": 128,
        "epochs": 40,
        "gradient_clip": 5.0,
        "dropout": 0.10,
        "precision": "fp32",
        "early_stopping": False,
    }
    diffs = {k: {"expected": v, "found": recipe.get(k)} for k, v in expected.items() if recipe.get(k) != v}
    sizes = {"train": C.N_TRAIN, "valid": C.N_VALID, "test": C.N_TEST, "steps_per_epoch": C.STEPS_PER_EPOCH,
             "total_steps": C.TOTAL_STEPS_40}
    size_expected = {"train": 350_343, "valid": 43_793, "test": 43_793, "steps_per_epoch": 2_738,
                     "total_steps": 109_520}
    size_diffs = {k: {"expected": v, "found": sizes[k]} for k, v in size_expected.items() if sizes[k] != v}
    lr = C.lr_audit()
    lr_expected = {
        "step0": 3.0e-5,
        "warmup_last": 3.0e-4,
        "post_warmup_first": None,  # cosine start, checked by monotonicity below
        "step40e": 1.0e-5,
    }
    from sg_common import scheduled_lr as inherited_lr

    lr_mismatch = [
        {"step": s, "ours": C.scheduled_lr(s, phase=3, kind="all"), "inherited": inherited_lr(s, phase=3, kind="all")}
        for s in (0, 1, 2737, 2738, 5476, 27380, 109519, 109520)
        if C.scheduled_lr(s, phase=3, kind="all") != inherited_lr(s, phase=3, kind="all")
    ]
    monotone = all(
        C.scheduled_lr(s, phase=3, kind="all") >= C.scheduled_lr(s + 1, phase=3, kind="all")
        for s in range(C.STEPS_PER_EPOCH, C.TOTAL_STEPS_40, 7919)
    )
    first_step = float(C.scheduled_lr(0, phase=3, kind="all"))
    first_step_max = 3.0e-5 + (3.0e-4 - 3.0e-5) / float(C.STEPS_PER_EPOCH) + 1e-12
    return {
        "recipe_expected": {k: (list(v) if isinstance(v, tuple) else v) for k, v in expected.items()},
        "recipe_mismatches": diffs,
        "recipe_ok": not diffs,
        "sizes": sizes,
        "size_mismatches": size_diffs,
        "sizes_ok": not size_diffs,
        "lr_audit": lr,
        "lr_warmup_start_constant_ok": float(recipe["warmup_start_lr"]) == 3.0e-5,
        "lr_first_applied_step": first_step,
        "lr_first_step_ok": 3.0e-5 < first_step <= first_step_max,
        "lr_warmup_last_ok": float(lr["warmup_last"]) == 3.0e-4,
        "lr_step40e_ok": abs(float(lr["step40e"]) - 1.0e-5) < 1e-12,
        "lr_matches_inherited_sg_common": not lr_mismatch,
        "lr_mismatch_examples": lr_mismatch,
        "lr_monotone_non_increasing_after_warmup": bool(monotone),
        "inherited_source": {
            "round": C.SOURCE["round"],
            "feature_spec_hash": C.SOURCE["feature_spec_hash"],
            "norm_fingerprint": C.SOURCE["norm_fingerprint"],
            "split_partition_sha256": C.SOURCE["split_partition_sha256"],
            "label_artifact_sha256": C.SOURCE["label_artifact_sha256"],
            "cache_sha256": C.SOURCE["cache_sha256"],
        },
        "test_policy": "official Test labels are NaN in the label artifact; no path indexes them",
    }


# ---------------------------------------------------------------------------
# GPU preflight
# ---------------------------------------------------------------------------


def audit_gpu_preflight() -> dict[str, Any]:
    out: dict[str, Any] = {
        "captured_utc": C.utcnow(),
        "host_facts": C.host_facts(),
        "occupancy": C.gpu_occupancy(),
        "probes": [],
    }
    if not torch.cuda.is_available():
        out["verdict"] = "CUDA_UNAVAILABLE"
        return out
    for index in range(torch.cuda.device_count()):
        probe: dict[str, Any] = {"device_index": index}
        try:
            probe["name"] = torch.cuda.get_device_name(index)
            probe["capability"] = list(torch.cuda.get_device_capability(index))
            free, total = torch.cuda.mem_get_info(index)
            probe["mem_get_info"] = {"free_bytes": int(free), "total_bytes": int(total), "used_bytes": int(total - free)}
            # NVML health + a real 256 MiB allocation/read-back on the device.
            torch.cuda.set_device(index)
            a = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device=f"cuda:{index}")
            a.fill_(1.0)
            b = a.sum()
            torch.cuda.synchronize(index)
            probe["allocation_256mib_sum"] = float(b)
            probe["allocation_ok"] = bool(float(b) == float(a.numel()))
            del a, b
            torch.cuda.empty_cache()
            probe["torch_rng_probe"] = int(torch.randint(0, 2, (1,), device=f"cuda:{index}").item())
            probe["ok"] = True
        except Exception as exc:  # pragma: no cover - diagnostic only
            probe["ok"] = False
            probe["error"] = f"{type(exc).__name__}: {exc}"
        out["probes"].append(probe)
    ok = all(bool(p.get("ok")) for p in out["probes"]) and all(
        isinstance(out["occupancy"].get(k), str) and not out["occupancy"][k].startswith("unavailable")
        for k in ("nvidia_smi_gpus", "nvidia_smi_compute_apps")
    )
    out["verdict"] = "OK" if ok else "DEGRADED"
    return out


# ---------------------------------------------------------------------------
# architecture audit
# ---------------------------------------------------------------------------


def _tensor_diff(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    keys_a, keys_b = set(a), set(b)
    shared = keys_a & keys_b
    diffs = {}
    for k in sorted(shared):
        if tuple(a[k].shape) != tuple(b[k].shape):
            diffs[k] = {"shape_a": list(a[k].shape), "shape_b": list(b[k].shape)}
            continue
        d = float((a[k].float() - b[k].float()).abs().max().item()) if a[k].numel() else 0.0
        if d > 0.0:
            diffs[k] = {"max_abs_diff": d}
    return {
        "n_keys_a": len(keys_a),
        "n_keys_b": len(keys_b),
        "keys_only_a": sorted(keys_a - keys_b),
        "keys_only_b": sorted(keys_b - keys_a),
        "n_shared": len(shared),
        "n_differing": len(diffs),
        "differing": diffs,
        "all_equal": not diffs and keys_a == keys_b,
    }


def _logits(model: torch.nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        return model(batch).detach().float().cpu()


def report_from_logits(logits: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """``sg_data.evaluate_validation``'s report, from already-computed logits."""
    probs = C.PC.sigmoid(logits)
    report = C.MET.validation_report(y, probs)
    report["valid_bce"] = float(C.MET.masked_bce_numpy(logits, y))
    report["valid_logits_sha256"] = C.sha256_bytes(np.ascontiguousarray(logits, dtype=np.float32).tobytes())
    report["valid_probs_sha256"] = C.sha256_bytes(np.ascontiguousarray(probs, dtype=np.float32).tobytes())
    report["evaluated_rows"] = int(logits.shape[0])
    return report


def audit_model(task: Any) -> dict[str, Any]:
    spec, norm = task.spec, task.norm
    out: dict[str, Any] = {
        "captured_utc": C.utcnow(),
        "commit": C.git_commit(),
        "host_facts": C.host_facts(),
        "occupancy": C.gpu_occupancy(),
        "constants": audit_constants(),
        "width_table": C.width_table(),
        "models": {},
        "checks": {},
    }
    rows = [int(x) for x in task.train_rows[:8]]
    batch = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in C.PC.collate(task.cache, rows).items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    built: dict[int, M.BSModel] = {}
    for scale in C.SCALES:
        model, info = M.build_model(scale, spec, norm, C.N_TASKS, seed=C.MODEL_SEED, device=device)
        built[int(scale)] = model
        out["models"][str(scale)] = {
            "scale": scale,
            "widths": info["widths"],
            "total_parameters": info["total_parameters"],
            "base_parameters": info["base_parameters"],
            "readout_parameters": info["readout_parameters"],
            "expected_parameters": info["expected_parameters"],
            "parameter_check": info["parameter_check"],
            "trunk_init_fingerprint": info["trunk_init_fingerprint"],
            "n_state_tensors": info["n_state_tensors"],
            "state_keys": info["state_keys"],
            "width_audit": info["width_audit"],
            "init": info["init"],
            "module_breakdown": TU.module_breakdown(model),
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        logits = _logits(model, {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()})
        out["models"][str(scale)]["epoch0_logits_shape"] = list(logits.shape)
        out["models"][str(scale)]["epoch0_logits_finite"] = bool(torch.isfinite(logits).all())

    # --- scale-1 equivalence to the historical LCRC-A implementation ------
    _ref_model, _ref_info, base_init = C.NM.build_reference(spec, norm, C.N_TASKS, C.MODEL_SEED)
    ref_model, ref_audit = C.NM.build_arm(
        "A",
        spec,
        norm,
        C.N_TASKS,
        seed=C.MODEL_SEED,
        device=device,
        base_init_state=base_init,
        zero_residual=True,
    )
    s1 = built[1]
    ref_state = {k: v.detach().cpu() for k, v in ref_model.state_dict().items()}
    s1_state = {k: v.detach().cpu() for k, v in s1.state_dict().items()}
    init_cmp = _tensor_diff(s1_state, ref_state)
    out["checks"]["scale1_state_keys_equal_historical_A"] = {
        "equal": set(s1_state) == set(ref_state),
        "n_keys": len(s1_state),
        "n_parameters_ours": C.parameter_count(s1),
        "n_parameters_historical": ref_audit["total_parameter_count"],
        "n_base_parameters": ref_audit["base_parameter_elements"],
        "n_readout_parameters": ref_audit["new_parameter_count"],
    }
    out["checks"]["scale1_init_state_bitwise_equal_historical_A"] = init_cmp
    out["checks"]["scale1_trunk_fingerprint_matches_recorded_A"] = {
        "expected": C.A0["base_init_fingerprint"],
        "found": out["models"]["1"]["trunk_init_fingerprint"],
        "equal": out["models"]["1"]["trunk_init_fingerprint"] == C.A0["base_init_fingerprint"],
    }
    # CPU: bitwise equality is required and achievable (sequential index_add_).
    cpu_s1 = M.build_model(1, spec, norm, C.N_TASKS, seed=C.MODEL_SEED, device="cpu")[0]
    cpu_ref = C.NM.NMPModel(spec, norm, C.N_TASKS, "A")
    cpu_ref.load_state_dict(cpu_s1.state_dict(), strict=True)
    cpu_diff = float((_logits(cpu_s1, batch) - _logits(cpu_ref, batch)).abs().max().item())
    # CUDA: index_add_/scatter_reduce_ use atomics, so the audited tolerance is 2e-5
    # (the same tolerance the historical structural-gain round used for CUDA).
    device_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    logits_s1 = _logits(s1, device_batch)
    logits_ref = _logits(ref_model, device_batch)
    gpu_diff = float((logits_s1 - logits_ref).abs().max().item())
    out["checks"]["scale1_forward_equal_historical_A"] = {
        "cpu_max_abs_diff": cpu_diff,
        "cpu_bitwise_equal": bool(cpu_diff == 0.0),
        "device_max_abs_diff": gpu_diff,
        "device": str(device),
        "device_tolerance": 2e-5,
        "device_within_tolerance": bool(gpu_diff <= 2e-5),
        "passed": bool(cpu_diff == 0.0 and gpu_diff <= 2e-5),
    }

    # --- historical checkpoint replay parity (logits, not AP) -------------
    payload = torch.load(C.A0_CHECKPOINT, map_location="cpu", weights_only=False)
    ckpt_state = {k: v.detach().cpu() for k, v in payload["state_dict"].items()}
    s1_loaded = M.build_model(1, spec, norm, C.N_TASKS, seed=C.MODEL_SEED, device="cpu")[0]
    load_info = s1_loaded.load_state_dict(ckpt_state, strict=True)
    ref_loaded = C.NM.NMPModel(spec, norm, C.N_TASKS, "A")
    ref_loaded.load_state_dict(ckpt_state, strict=True)
    logits_s1_ckpt = _logits(s1_loaded, batch)
    logits_ref_ckpt = _logits(ref_loaded, batch)
    out["checks"]["scale1_historical_checkpoint_forward_equal"] = {
        "missing_keys": list(load_info.missing_keys),
        "unexpected_keys": list(load_info.unexpected_keys),
        "max_abs_diff": float((logits_s1_ckpt - logits_ref_ckpt).abs().max().item()),
        "tolerance": 1e-6,
        "passed": bool(float((logits_s1_ckpt - logits_ref_ckpt).abs().max().item()) <= 1e-6),
    }

    # --- S1/S2 inputs identical across scales -----------------------------
    layouts = {}
    for name, cfg in (("scale1", C.config_for_scale(1)), ("scale2", C.config_for_scale(2)), ("scale3", C.config_for_scale(3))):
        lay = TU.local_layout(spec, "C1", cfg)
        layouts[name] = {k: [int(v[0]), int(v[1])] for k, v in lay.items()}
    part_names = {tuple(v) for v in layouts.values()}
    batch_identity = {
        "n_molecules": int(batch["n_graphs"]),
        "n_atoms": int(batch["atom_feat"].shape[0]),
        "n_bonds": int(batch["bond_index"].shape[0]),
        "n_s1_codes": int(batch["nbr_code"].numel()),
        "n_s2_paths": int(batch["path_cat"].numel()),
        "max_nbr_code": int(batch["nbr_code"].max().item()) if batch["nbr_code"].numel() else -1,
        "max_path_cat": int(batch["path_cat"].max().item()) if batch["path_cat"].numel() else -1,
        "atom_feat_dim": int(spec.atom_feat_dim),
        "bond_feat_dim": int(spec.bond_feat_dim),
        "atom_scalar_dim": int(spec.n_scalars),
        "global_dim": int(spec.global_dim),
        "n_nbr_codes": int(spec.n_nbr_codes),
        "n_path_codes": int(spec.n_path_codes),
        "close_emb_vocab": [2, 8],
    }
    out["checks"]["s1_s2_inputs_identical"] = {
        "batch": batch_identity,
        "local_block_names_identical": len({tuple(sorted(v)) for v in layouts.values()}) == 1,
        "local_blocks": {k: sorted(v) for k, v in layouts.items()},
        "embedding_vocab_not_scaled": all(
            out["models"][str(s)]["width_audit"]["observed"]["stat_rank"]["nbr_emb"][0] == 80
            and out["models"][str(s)]["width_audit"]["observed"]["stat_rank"]["path_emb"][0] == 400
            and out["models"][str(s)]["width_audit"]["observed"]["stat_rank"]["close_emb"] == [2, 8]
            for s in C.SCALES
        ),
        "matches_protocol_inputs": batch_identity["atom_feat_dim"] == C.FROZEN_INPUTS["atom_feat_dim"]
        and batch_identity["bond_feat_dim"] == C.FROZEN_INPUTS["bond_feat_dim"]
        and batch_identity["atom_scalar_dim"] == C.FROZEN_INPUTS["atom_scalar_dim"]
        and batch_identity["global_dim"] == C.FROZEN_INPUTS["global_dim"]
        and batch_identity["n_nbr_codes"] == C.FROZEN_INPUTS["n_nbr_codes"]
        and batch_identity["n_path_codes"] == C.FROZEN_INPUTS["n_path_codes"],
    }

    # --- one-shot encoding, no second propagation -------------------------
    one_shot = {}
    for scale in C.SCALES:
        model = built[int(scale)]
        dev_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        calls = M.count_module_calls(model, dev_batch)
        model.eval()
        with torch.inference_mode():
            _a, _b, _scal, h = model.encode_h(dev_batch)
            full = model(dev_batch)
            minimal = {"atom_batch": dev_batch["atom_batch"], "n_graphs": int(dev_batch["n_graphs"]),
                       "global": dev_batch["global"]}
            post = model.post_local(h, minimal)
        minimal_diff = float((full - post).abs().max().item())
        one_shot[str(scale)] = {
            "module_calls": calls,
            "calls_exactly_once": calls
            == {"atom_encoder": 1, "bond_encoder": 1, "local": 1, "f": 1, "rho": 1, "readout": 1},
            "post_local_needs_no_bond_or_path_field": True,
            "post_local_minimal_batch_max_abs_diff": minimal_diff,
            "bitwise_equal": bool(minimal_diff == 0.0),
            "tolerance": 2e-5,
            "equal": bool(minimal_diff <= 2e-5),
        }
    out["checks"]["one_shot_encoding_no_second_propagation"] = {
        "per_scale": one_shot,
        "passed": all(v["calls_exactly_once"] and v["equal"] for v in one_shot.values()),
        "note_tolerance": (
            "on CUDA the same post-local graph run twice differs by <=2e-5 because "
            "index_add_/scatter_reduce_ accumulate with atomics; on CPU the difference is 0"
        ),
        "statement": (
            "atom_encoder, bond_encoder and local each run exactly once per forward; "
            "everything after local is a function of (h, atom_batch, n_graphs, global) only"
        ),
    }

    # --- atom renumbering invariance --------------------------------------
    rename = {}
    for scale in C.SCALES:
        model = built[int(scale)]
        perm_batch, p = M.permute_atoms(batch)
        dev_a = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        dev_b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in perm_batch.items()}
        la = _logits(model, dev_a)
        lb = _logits(model, dev_b)
        attr = _logits_attribution(model, dev_a)
        out_perm = _logits_attribution(model, dev_b)
        maxdiff = float((la - lb).abs().max().item())
        rename[str(scale)] = {
            "logits_max_abs_diff": maxdiff,
            "logits_max_abs": float(la.abs().max().item()),
            "attribution_max_abs_diff": float((out_perm - attr[p]).abs().max().item()),
            "tolerance": 2e-5,
            "passed": bool(maxdiff <= 2e-5),
        }
    out["checks"]["atom_renumbering_invariance"] = {
        "per_scale": rename,
        "passed": all(v["passed"] for v in rename.values()),
        "note": "graph logits are invariant; the per-atom contribution follows the permutation",
    }

    # --- Train-only normalisation identity --------------------------------
    norm_state = {}
    for scale in C.SCALES:
        sd = {k: v.detach().cpu() for k, v in built[int(scale)].state_dict().items()}
        norm_state[str(scale)] = {k: sd[k] for k in ("global_mean", "global_scale", "scalar_mean", "scalar_scale")}
    ckpt_norm = {k: ckpt_state[k] for k in ("global_mean", "global_scale", "scalar_mean", "scalar_scale")}
    out["checks"]["normalisation_identity"] = {
        "norm_fingerprint": task.norm.get("norm_fingerprint"),
        "expected_norm_fingerprint": C.SOURCE["norm_fingerprint"],
        "fit_split": task.norm.get("fit_split"),
        "fit_rows": task.norm.get("fit_rows"),
        "valid_or_test_used": task.norm.get("valid_or_test_used"),
        "matches_frozen_source": task.norm.get("norm_fingerprint") == C.SOURCE["norm_fingerprint"],
        "train_only": task.norm.get("fit_split") == "train_only" and task.norm.get("valid_or_test_used") is False,
        "all_scales_equal_historical_checkpoint": all(
            _tensor_diff(norm_state[s], ckpt_norm)["all_equal"] for s in norm_state
        ),
        "per_scale": {s: _tensor_diff(norm_state[s], ckpt_norm)["all_equal"] for s in norm_state},
        "cont_buffers_empty": all(int(built[int(s)].cont_mean.numel()) == 0 for s in C.SCALES),
    }

    # --- forward/backward health ------------------------------------------
    health = {}
    for scale in C.SCALES:
        model, _info = M.build_model(scale, spec, norm, C.N_TASKS, seed=C.MODEL_SEED, device=device)
        optimizer, _ = M.make_optimizer(model)
        dev_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        y = torch.as_tensor(np.asarray(task.labels[np.asarray(rows)]), dtype=torch.float32, device=device)
        model.train()
        losses = []
        gnorms = []
        for step in range(3):
            res = _train_step_local(model, optimizer, dev_batch, y, global_step=step, clip=float(C.RECIPE["gradient_clip"]))
            losses.append(res["loss"])
            gnorms.append(res["grad_norm"])
        grads = {}
        for name, p in model.named_parameters():
            grads[name] = None if p.grad is None else float(p.grad.detach().abs().max().item())
        missing = [k for k, v in grads.items() if v is None]
        zero = [k for k, v in grads.items() if v == 0.0]
        known_zero = [k for k in zero if k.startswith("readout.heads.") and k.endswith("fc2.bias")]
        unexpected_zero = [k for k in zero if k not in known_zero]
        with torch.inference_mode():
            logits = model(dev_batch)
        health[str(scale)] = {
            "losses": losses,
            "grad_norms": gnorms,
            "gradients_present": not missing,
            "missing_gradients": missing,
            "zero_gradients": zero,
            "known_structural_zero_gradients": known_zero,
            "unexpected_zero_gradients": unexpected_zero,
            "logits_finite": bool(torch.isfinite(logits).all()),
            "loss_finite": bool(all(np.isfinite(losses))),
            "loss_first": float(losses[0]),
            "n_parameters": C.parameter_count(model),
            "all_finite_and_gradients_ok": not missing
            and not unexpected_zero
            and bool(all(np.isfinite(losses)))
            and bool(torch.isfinite(logits).all()),
        }
    out["checks"]["forward_backward_health"] = {
        "per_scale": health,
        "passed": all(v["all_finite_and_gradients_ok"] for v in health.values()),
        "note": (
            "readout.heads.*.fc2.bias has a structurally zero gradient: a constant "
            "per-head score shift does not change the per-molecule softmax. Historical fact."
        ),
    }

    # --- Test isolation ----------------------------------------------------
    labels = np.asarray(task.labels)
    out["checks"]["official_test_untouched"] = {
        "test_rows": int(task.test_rows.size),
        "finite_test_labels": int(np.isfinite(labels[task.test_rows]).sum()),
        "valid_rows": int(task.valid_rows.size),
        "train_rows": int(task.train_rows.size),
        "test_evaluated": False,
        "passed": int(np.isfinite(labels[task.test_rows]).sum()) == 0
        and int(task.valid_rows.size) == C.N_VALID
        and int(task.train_rows.size) == C.N_TRAIN,
    }

    # --- batch-order plan identity -----------------------------------------
    plan = C.SD.make_plan40(task, seed=C.SEED)
    out["checks"]["batch_order_plan"] = {
        "global_plan_sha256": plan[0]["global_plan_sha256"],
        "expected": C.A0["global_plan_sha256"],
        "equal": plan[0]["global_plan_sha256"] == C.A0["global_plan_sha256"],
        "steps_per_epoch": int(len(plan[0]["batches"])),
        "epochs": len(plan),
    }

    out["verdict"] = "PASS" if _verdict(out) else "FAIL"
    return out


def _verdict(out: Mapping[str, Any]) -> bool:
    checks = out["checks"]
    return bool(
        out["constants"]["recipe_ok"]
        and out["constants"]["sizes_ok"]
        and out["constants"]["lr_warmup_start_constant_ok"]
        and out["constants"]["lr_first_step_ok"]
        and out["constants"]["lr_warmup_last_ok"]
        and out["constants"]["lr_step40e_ok"]
        and out["constants"]["lr_matches_inherited_sg_common"]
        and checks["scale1_state_keys_equal_historical_A"]["equal"]
        and checks["scale1_init_state_bitwise_equal_historical_A"]["all_equal"]
        and checks["scale1_trunk_fingerprint_matches_recorded_A"]["equal"]
        and checks["scale1_forward_equal_historical_A"]["passed"]
        and checks["scale1_historical_checkpoint_forward_equal"]["passed"]
        and checks["one_shot_encoding_no_second_propagation"]["passed"]
        and checks["atom_renumbering_invariance"]["passed"]
        and checks["normalisation_identity"]["matches_frozen_source"]
        and checks["normalisation_identity"]["all_scales_equal_historical_checkpoint"]
        and checks["forward_backward_health"]["passed"]
        and checks["official_test_untouched"]["passed"]
        and checks["batch_order_plan"]["equal"]
        and all(out["models"][str(s)]["width_audit"]["all_match"] for s in C.SCALES)
        and all(out["models"][str(s)]["parameter_check"] == "exact" for s in C.SCALES)
    )


def _logits_attribution(model: torch.nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    """Per-atom contribution of ``f`` around the graph logits (audit helper)."""
    model.eval()
    with torch.inference_mode():
        _a, _b, _s, h = model.encode_h(batch)
        return model.f(h).detach().float().cpu()


def _train_step_local(model, optimizer, batch, y, *, global_step: int, clip: float):
    import bs_train as T

    return T.train_step(model, optimizer, batch, y, global_step=global_step, clip=clip)


# ---------------------------------------------------------------------------
# historical A replay
# ---------------------------------------------------------------------------


def replay_a0(task: Any) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(C.A0_CHECKPOINT, map_location="cpu", weights_only=False)
    run_record = json.loads(Path(C.A0_RUN_JSON).read_text(encoding="utf-8"))
    spec, norm = task.spec, task.norm
    state = {k: v.detach().cpu() for k, v in payload["state_dict"].items()}

    model_new = M.build_model(1, spec, norm, C.N_TASKS, seed=C.MODEL_SEED, device=device)[0]
    model_new.load_state_dict(state, strict=True)
    ref = C.NM.NMPModel(spec, norm, C.N_TASKS, "A")
    ref.load_state_dict(state, strict=True)
    ref.to(device)

    logits_new = C.PC.predict_logits(model_new, task.cache, task.valid_rows, device, batch_size=256)
    logits_ref = C.PC.predict_logits(ref, task.cache, task.valid_rows, device, batch_size=256)
    y = np.asarray(task.labels[task.valid_rows])
    report = report_from_logits(logits_new, y)
    report_ref = report_from_logits(logits_ref, y)

    hist = run_record["fixed_epoch_reports"]["40"]
    hist_per_task = np.asarray([np.nan if v is None else float(v) for v in hist["per_task_ap"]], dtype=np.float64)
    new_per_task = np.asarray([np.nan if v is None else float(v) for v in report["per_task_ap"]], dtype=np.float64)
    finite = np.isfinite(hist_per_task) & np.isfinite(new_per_task)
    logits_hist = payload.get("valid_report_summary") or {}
    return {
        "captured_utc": C.utcnow(),
        "device": str(device),
        "checkpoint": C.track_rel(C.A0_CHECKPOINT),
        "checkpoint_sha256": C.sha256_file(C.A0_CHECKPOINT),
        "checkpoint_sha256_expected": C.A0["checkpoint_sha256"],
        "checkpoint_sha256_ok": C.sha256_file(C.A0_CHECKPOINT) == C.A0["checkpoint_sha256"],
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_global_step": int(payload["global_step"]),
        "checkpoint_parameters": int(payload["model_audit"]["total_parameter_count"]),
        "historical_ap40_recorded": float(hist["ap"]),
        "historical_valid_bce_recorded": float(hist["valid_bce"]),
        "historical_valid_logits_sha256": hist.get("valid_logits_sha256"),
        "historical_valid_probs_sha256": hist.get("valid_probs_sha256"),
        "checkpoint_valid_report_summary": logits_hist,
        "scale1_replay_ap": float(report["ap"]),
        "scale1_replay_valid_bce": float(report["valid_bce"]),
        "scale1_replay_n_tasks_scored": int(report["n_tasks_scored"]),
        "scale1_replay_valid_logits_sha256": report.get("valid_logits_sha256"),
        "abs_delta_vs_recorded": abs(float(report["ap"]) - float(hist["ap"])),
        "tolerance": 1e-4,
        "ap_ok": bool(abs(float(report["ap"]) - float(hist["ap"])) <= 1e-4),
        "per_task_max_abs_delta_vs_recorded": float(np.nanmax(np.abs(new_per_task[finite] - hist_per_task[finite])))
        if finite.any()
        else None,
        "scale1_vs_nmpmodel_A_logits_max_abs_diff": float(np.abs(logits_new - logits_ref).max()),
        "scale1_vs_nmpmodel_A_ap_delta": abs(float(report["ap"]) - float(report_ref["ap"])),
        "historical_train_ap_recorded": float(run_record["train_eval_final"]["ap"]),
        "note": "read-only replay of the historical epoch-40 A checkpoint; no training, no Test access",
    }


# ---------------------------------------------------------------------------
# evaluation-cost probe
# ---------------------------------------------------------------------------


def eval_probe(task: Any) -> dict[str, Any]:
    import bs_train as T

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plan = C.SD.make_plan40(task, seed=C.SEED)
    rows = [int(x) for x in plan[0]["batches"][0]]
    out: dict[str, Any] = {
        "captured_utc": C.utcnow(),
        "device": str(device),
        "host_facts": C.host_facts(),
        "occupancy": C.gpu_occupancy(),
        "valid_rows": int(task.valid_rows.size),
        "valid_batch_size": int(C.PROBE_VALID_BATCH),
        "batch_size": int(C.BATCH_SIZE),
        "steps_per_epoch": int(C.STEPS_PER_EPOCH),
        "probe_train_steps": int(C.PROBE_TRAIN_STEPS),
        "probe_warmup_steps": int(C.PROBE_WARMUP_STEPS),
        "models": {},
        "rule": dict(C.EVAL_RULE),
    }
    y = torch.as_tensor(np.asarray(task.labels[np.asarray(rows)]), dtype=torch.float32)
    for key in C.MODEL_ORDER:
        scale = int(C.MODELS[key]["scale"])
        model, info = M.build_model(scale, task.spec, task.norm, C.N_TASKS, seed=C.MODEL_SEED, device=device)
        optimizer, _ = M.make_optimizer(model)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        step_times: list[float] = []
        for step in range(int(C.PROBE_WARMUP_STEPS) + int(C.PROBE_TRAIN_STEPS)):
            batch = C.PC.to_device(C.PC.collate(task.cache, rows), device)
            t0 = time.perf_counter()
            T.train_step(model, optimizer, batch, y.to(device), global_step=step, clip=float(C.RECIPE["gradient_clip"]))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            dt = time.perf_counter() - t0
            if step >= int(C.PROBE_WARMUP_STEPS):
                step_times.append(float(dt))
        train_peak = C.gpu_peak(device)

        model.eval()
        t0 = time.perf_counter()
        logits = C.PC.predict_logits(model, task.cache, task.valid_rows, device, batch_size=int(C.PROBE_VALID_BATCH))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fwd_seconds = float(time.perf_counter() - t0)
        t1 = time.perf_counter()
        yv = np.asarray(task.labels[task.valid_rows])
        report = report_from_logits(logits, yv)
        eval_seconds = float(time.perf_counter() - t1)
        valid_seconds = fwd_seconds + eval_seconds
        peak = C.gpu_peak(device)

        step_mean = float(np.mean(step_times))
        step_p50 = float(np.percentile(step_times, 50))
        epoch_seconds = step_p50 * int(C.STEPS_PER_EPOCH)
        fraction = valid_seconds / epoch_seconds
        out["models"][key] = {
            "scale": scale,
            "parameters": info["total_parameters"],
            "step_ms_mean": step_mean * 1e3,
            "step_ms_p50": step_p50 * 1e3,
            "steps_per_second": 1.0 / step_p50,
            "projected_epoch_seconds": epoch_seconds,
            "validation_forward_seconds": fwd_seconds,
            "validation_metric_seconds": eval_seconds,
            "validation_seconds": valid_seconds,
            "validation_fraction_of_epoch": fraction,
            "train_eval_projected_seconds": valid_seconds * (C.N_TRAIN / C.N_VALID),
            "gpu_peak_allocated_bytes": peak["peak_allocated_bytes"],
            "gpu_peak_reserved_bytes": peak["peak_reserved_bytes"],
            "probe_valid_ap": float(report["ap"]),
            "probe_valid_bce": float(report["valid_bce"]),
        }
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fracs = {k: v["validation_fraction_of_epoch"] for k, v in out["models"].items()}
    every = 1 if all(f <= float(C.EVAL_RULE["threshold_fraction_of_projected_epoch"]) for f in fracs.values()) else 2
    n_evals = len(C.calc_eval_epochs(every))
    per_model = {
        k: (
            C.TOTAL_EPOCHS * v["projected_epoch_seconds"]
            + n_evals * v["validation_seconds"]
            + v["train_eval_projected_seconds"]
        )
        for k, v in out["models"].items()
    }
    total_seconds = sum(per_model.values())
    out["resolution"] = {
        "validation_fraction_of_epoch": fracs,
        "threshold": float(C.EVAL_RULE["threshold_fraction_of_projected_epoch"]),
        "all_below_threshold": bool(all(f <= float(C.EVAL_RULE["threshold_fraction_of_projected_epoch"]) for f in fracs.values())),
        "resolved_eval_every": int(every),
        "resolved_eval_epochs": C.calc_eval_epochs(every),
        "n_evaluations_per_model": int(n_evals),
    }
    out["projection"] = {
        "per_model_seconds": per_model,
        "per_model_gpu_hours": {k: v / 3600.0 for k, v in per_model.items()},
        "total_gpu_hours": total_seconds / 3600.0,
        "target_gpu_hours": float(C.BUDGET["gpu_target_hours"]),
        "hard_gpu_hours": float(C.BUDGET["gpu_hard_hours"]),
        "within_target": bool(total_seconds / 3600.0 <= float(C.BUDGET["gpu_target_hours"])),
        "within_hard_cap": bool(total_seconds / 3600.0 <= float(C.BUDGET["gpu_hard_hours"])),
        "budget_label": "OK" if total_seconds / 3600.0 <= float(C.BUDGET["gpu_hard_hours"]) else "BUDGET_NOT_FEASIBLE",
        "note": "projection excludes preflight already spent; excludes co-tenancy slowdown on a shared GPU",
    }
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="gpu-preflight,model,a0-replay,eval-probe")
    parser.add_argument("--out-dir", default=str(C.ANALYSIS))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    targets = {
        "gpu-preflight": ("gpu_preflight.json", audit_gpu_preflight),
        "model": ("model_scaling_audit.json", lambda: audit_model(C.SD.load_task())),
        "a0-replay": ("a0_replay.json", lambda: replay_a0(C.SD.load_task())),
        "eval-probe": ("evaluation_cost_probe.json", lambda: eval_probe(C.SD.load_task())),
    }
    failures = []
    for mode in modes:
        if mode not in targets:
            raise KeyError(f"unknown mode {mode}")
        name, fn = targets[mode]
        C.log(f"audit mode={mode} -> {name}")
        t0 = time.perf_counter()
        try:
            result = fn()
            result["mode"] = mode
            result["seconds"] = float(time.perf_counter() - t0)
            C.write_json(out_dir / name, result)
            C.log(f"audit mode={mode} done in {result['seconds']:.1f}s")
        except Exception as exc:  # noqa: BLE001
            failures.append({"mode": mode, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
            C.write_json(
                out_dir / name,
                {"mode": mode, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                 "traceback": traceback.format_exc()},
            )
            C.log(f"audit mode={mode} FAILED: {exc}")
    if failures:
        C.write_json(out_dir / "audit_failures.json", {"failures": failures})
        return 1
    finalize_gate(out_dir)
    return 0


def finalize_gate(out_dir: Path) -> dict[str, Any] | None:
    """Fold the A0 replay into the single pre-training gate record."""
    model_path = out_dir / "model_scaling_audit.json"
    replay_path = out_dir / "a0_replay.json"
    if not model_path.exists() or not replay_path.exists():
        return None
    model = json.loads(model_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    model["a0_replay"] = {
        k: replay.get(k)
        for k in (
            "scale1_replay_ap",
            "historical_ap40_recorded",
            "abs_delta_vs_recorded",
            "ap_ok",
            "per_task_max_abs_delta_vs_recorded",
            "checkpoint_sha256_ok",
            "scale1_vs_nmpmodel_A_logits_max_abs_diff",
            "scale1_vs_nmpmodel_A_ap_delta",
        )
    }
    model["overall_gate"] = bool(
        model.get("verdict") == "PASS"
        and replay.get("ap_ok")
        and replay.get("checkpoint_sha256_ok")
        and (replay.get("scale1_vs_nmpmodel_A_logits_max_abs_diff") <= 2e-5)
    )
    C.write_json(model_path, model)
    C.log(f"pre-training gate overall={model['overall_gate']}")
    return model["overall_gate"]


if __name__ == "__main__":
    raise SystemExit(main())
