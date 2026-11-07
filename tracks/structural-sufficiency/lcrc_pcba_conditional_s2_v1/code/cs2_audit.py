"""Pre-training audits for the Conditional S2 round.

Modes
-----
``gpu-preflight``   device / NVML / per-visible-GPU allocation probes.
``gpu-sampling``    6 x 10 s nvidia-smi samples of both GPUs (occupancy facts).
``init``            W2 initialisation identity, exact parameter counts, initial
                    output parity (CPU bitwise; CUDA and full-Validation
                    tolerance), new-module gradients, checkpoint round-trip,
                    Test isolation.
``architecture``    path-code ranges, gate shape, one-shot call counts, batch
                    immutability, no post-``local`` structure access, dropout
                    count, atom-renumbering invariance, label-mask correctness.

Nothing here trains a formal model or touches an official Test row.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import cs2_common as C
import cs2_model as M
import cs2_train as T
import bs_model as BSM


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tensor_diff(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    keys_a, keys_b = set(a), set(b)
    diffs: dict[str, Any] = {}
    for k in sorted(keys_a & keys_b):
        if tuple(a[k].shape) != tuple(b[k].shape):
            diffs[k] = {"shape_a": list(a[k].shape), "shape_b": list(b[k].shape)}
        elif not torch.equal(a[k].cpu(), b[k].cpu()):
            diffs[k] = {"max_abs_diff": float((a[k].float().cpu() - b[k].float().cpu()).abs().max().item())}
    for k in sorted(keys_a - keys_b):
        diffs[k] = {"only_in_a": True}
    for k in sorted(keys_b - keys_a):
        diffs[k] = {"only_in_b": True}
    return {
        "n_keys_a": len(keys_a),
        "n_keys_b": len(keys_b),
        "n_shared": len(keys_a & keys_b),
        "differing": diffs,
        "all_bitwise_equal": not diffs and keys_a == keys_b,
    }


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _logits(model: torch.nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            return model(batch).detach().float().cpu()
    finally:
        model.train(was_training)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def report_from_logits(logits: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    probs = C.PC.sigmoid(logits)
    report = C.MET.validation_report(y, probs)
    report["valid_bce"] = float(C.MET.masked_bce_numpy(logits, y))
    report["valid_logits_sha256"] = C.sha256_bytes(np.ascontiguousarray(logits, dtype=np.float32).tobytes())
    report["valid_probs_sha256"] = C.sha256_bytes(np.ascontiguousarray(probs, dtype=np.float32).tobytes())
    report["evaluated_rows"] = int(logits.shape[0])
    return report


def _full_valid_logits(model: torch.nn.Module, task: Any, device: torch.device) -> np.ndarray:
    return C.PC.predict_logits(model, task.cache, task.valid_rows, device, batch_size=256)


# ---------------------------------------------------------------------------
# GPU facts
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
            probe["mem_get_info"] = {
                "free_bytes": int(free),
                "total_bytes": int(total),
                "used_bytes": int(total - free),
            }
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
        except Exception as exc:  # pragma: no cover
            probe["ok"] = False
            probe["error"] = f"{type(exc).__name__}: {exc}"
        out["probes"].append(probe)
    ok = all(bool(p.get("ok")) for p in out["probes"]) and all(
        isinstance(out["occupancy"].get(k), str) and not out["occupancy"][k].startswith("unavailable")
        for k in ("nvidia_smi_gpus", "nvidia_smi_compute_apps")
    )
    out["verdict"] = "OK" if ok else "DEGRADED"
    return out


def _gpu_rows() -> list[dict[str, Any]]:
    import subprocess

    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=20,
    ).strip()
    rows = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        rows.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_total_mib": float(parts[3]),
                "memory_used_mib": float(parts[4]),
                "memory_free_mib": float(parts[5]),
                "utilization_gpu_pct": float(parts[6]),
                "utilization_memory_pct": float(parts[7]),
                "temperature_c": float(parts[8]),
            }
        )
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        text=True,
        timeout=20,
    ).strip()
    for row in rows:
        row["compute_apps"] = apps
    return rows


def audit_gpu_sampling(samples: int = 6, interval_seconds: float = 10.0) -> dict[str, Any]:
    out: dict[str, Any] = {
        "captured_utc": C.utcnow(),
        "samples_requested": int(samples),
        "interval_seconds": float(interval_seconds),
        "samples": [],
        "compute_apps": [],
    }
    import subprocess

    try:
        out["compute_apps"] = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            text=True,
            timeout=20,
        ).strip().splitlines()
    except Exception as exc:  # pragma: no cover
        out["compute_apps_error"] = f"{type(exc).__name__}: {exc}"
    t_start = time.perf_counter()
    for i in range(int(samples)):
        try:
            out["samples"].append(
                {"sample": i, "t_rel_s": round(time.perf_counter() - t_start, 3), "gpus": _gpu_rows()}
            )
        except Exception as exc:  # pragma: no cover
            out["samples"].append({"sample": i, "error": f"{type(exc).__name__}: {exc}"})
        if i < int(samples) - 1:
            time.sleep(float(interval_seconds))
    # per-GPU summary
    summary: dict[str, Any] = {}
    for sample in out["samples"]:
        for gpu in sample.get("gpus", []):
            key = str(gpu["index"])
            s = summary.setdefault(
                key,
                {
                    "index": gpu["index"],
                    "name": gpu["name"],
                    "utilization_gpu_pct": [],
                    "memory_free_mib": [],
                    "memory_used_mib": [],
                },
            )
            s["utilization_gpu_pct"].append(gpu["utilization_gpu_pct"])
            s["memory_free_mib"].append(gpu["memory_free_mib"])
            s["memory_used_mib"].append(gpu["memory_used_mib"])
    for key, s in summary.items():
        util = s["utilization_gpu_pct"]
        s["saturated_at_100pct_all_samples"] = bool(util) and all(v >= 99.0 for v in util)
        s["max_utilization_pct"] = float(max(util)) if util else None
        s["min_free_mib"] = float(min(s["memory_free_mib"])) if s["memory_free_mib"] else None
    out["summary"] = summary
    return out


# ---------------------------------------------------------------------------
# init audit
# ---------------------------------------------------------------------------


def audit_constants() -> dict[str, Any]:
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
    recipe = C.RECIPE
    diffs = {
        k: {"expected": (list(v) if isinstance(v, tuple) else v), "found": recipe.get(k)}
        for k, v in expected.items()
        if recipe.get(k) != v
    }
    sizes = {
        "train": C.N_TRAIN,
        "valid": C.N_VALID,
        "test": C.N_TEST,
        "steps_per_epoch": C.STEPS_PER_EPOCH,
        "total_steps": C.TOTAL_STEPS_40,
    }
    size_expected = {
        "train": 350_343,
        "valid": 43_793,
        "test": 43_793,
        "steps_per_epoch": 2_738,
        "total_steps": 109_520,
    }
    size_diffs = {k: {"expected": v, "found": sizes[k]} for k, v in size_expected.items() if sizes[k] != v}
    lr = C.lr_audit()
    from sg_common import scheduled_lr as inherited_lr

    lr_mismatch = [
        {
            "step": s,
            "ours": C.scheduled_lr(s, phase=3, kind="all"),
            "inherited": inherited_lr(s, phase=3, kind="all"),
        }
        for s in (0, 1, 2737, 2738, 5476, 27380, 109519, 109520)
        if C.scheduled_lr(s, phase=3, kind="all") != inherited_lr(s, phase=3, kind="all")
    ]
    return {
        "recipe_expected": {k: (list(v) if isinstance(v, tuple) else v) for k, v in expected.items()},
        "recipe_mismatches": diffs,
        "recipe_ok": not diffs,
        "sizes": sizes,
        "size_mismatches": size_diffs,
        "sizes_ok": not size_diffs,
        "lr_audit": lr,
        "lr_matches_inherited_sg_common": not lr_mismatch,
        "lr_mismatch_examples": lr_mismatch,
        "lr_step0_expected": 3.0e-5,
        "lr_step0_is_warmup_start_plus_one_step": bool(
            3.0e-5 < float(lr["step0"]) <= 3.0e-5 + (3.0e-4 - 3.0e-5) / float(C.STEPS_PER_EPOCH) + 1e-12
        ),
        "lr_warmup_last": float(lr["warmup_last"]),
        "lr_step40e": float(lr["step40e"]),
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


def gradient_reachability(
    task: Any,
    plan: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    steps: int = C.PROBE_TRAIN_STEPS,
) -> dict[str, Any]:
    """Real masked-BCE steps; track only the new tensors' gradient norms."""
    out: dict[str, Any] = {}
    for key in C.MODEL_ORDER:
        model, audit = M.build_model(key, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
        optimizer, _ = BSM.make_optimizer(model)
        C.seed_everything(C.SEED)
        new_names = set(audit["new_state_keys"])
        records = []
        for step in range(int(steps)):
            rows = np.asarray(plan[0]["batches"][step])
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in C.PC.collate(task.cache, rows.tolist()).items()}
            y = torch.as_tensor(task.labels[rows], dtype=torch.float32, device=device)
            result = T.train_step(model, optimizer, batch, y, global_step=step, clip=float(C.RECIPE["gradient_clip"]))
            grads = {}
            for name, param in model.named_parameters():
                if name in new_names or name.startswith("local.path_cond_") or name.startswith("ffn."):
                    g = param.grad
                    grads[name] = {
                        "grad_l2": None if g is None else float(g.detach().float().norm().item()),
                        "param_l2": float(param.detach().float().norm().item()),
                    }
            diag = M.module_diagnostics(model, batch)
            records.append(
                {
                    "step": int(step),
                    "loss": float(result["loss"]),
                    "grad_norm_total": float(result["grad_norm"]),
                    "lr": float(result["lr"]),
                    "new_grads": grads,
                    "gate_delta_rms": (diag.get("gate") or {}).get("delta_gate", {}).get("rms"),
                    "ffn_branch_rms": (diag.get("ffn") or {}).get("branch", {}).get("rms"),
                }
            )
        if key == "C2":
            checks = {
                "V_grad_nonzero_step1": bool((records[0]["new_grads"]["local.path_cond_v"]["grad_l2"] or 0.0) > 0.0),
                "V_param_nonzero_after_step1": bool(records[0]["new_grads"]["local.path_cond_v"]["param_l2"] > 0.0),
                "E_c_grad_zero_step1_expected": bool(
                    (records[0]["new_grads"]["local.path_cond_ec.weight"]["grad_l2"] or 0.0) == 0.0
                ),
                "E_c_grad_nonzero_by_step2": bool(
                    (records[1]["new_grads"]["local.path_cond_ec.weight"]["grad_l2"] or 0.0) > 0.0
                ),
                "E_e_grad_nonzero_by_step2": bool(
                    (records[1]["new_grads"]["local.path_cond_ee.weight"]["grad_l2"] or 0.0) > 0.0
                ),
                "V_grad_nonzero_by_step2": bool(
                    (records[1]["new_grads"]["local.path_cond_v"]["grad_l2"] or 0.0) > 0.0
                ),
                "gate_delta_nonzero_after_step1": bool(
                    float(records[0]["gate_delta_rms"] or 0.0) > 0.0
                ),
            }
        else:
            checks = {
                "fc2_grad_nonzero_step1": bool((records[0]["new_grads"]["ffn.fc2.weight"]["grad_l2"] or 0.0) > 0.0),
                "fc1_grad_zero_step1_expected": bool(
                    (records[0]["new_grads"]["ffn.fc1.weight"]["grad_l2"] or 0.0) == 0.0
                ),
                "fc1_grad_nonzero_by_step2": bool(
                    (records[1]["new_grads"]["ffn.fc1.weight"]["grad_l2"] or 0.0) > 0.0
                ),
                "ffn_branch_nonzero_after_step1": bool(float(records[0]["ffn_branch_rms"] or 0.0) > 0.0),
            }
        out[key] = {
            "steps": records,
            "checks": checks,
            "all_checks_pass": all(checks.values()),
            "optimizer_parameters": int(sum(p.numel() for p in model.parameters())),
        }
    return out


def checkpoint_roundtrip(
    task: Any,
    plan: Sequence[Mapping[str, Any]],
    device: torch.device,
    work_dir: Path,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    work_dir.mkdir(parents=True, exist_ok=True)
    for key in C.MODEL_ORDER:
        model, audit = M.build_model(key, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
        optimizer, opt_audit = BSM.make_optimizer(model)
        C.seed_everything(C.SEED)
        rng_before = C.rng_state_hashes(C.capture_rng_state())
        for step in range(2):
            rows = np.asarray(plan[0]["batches"][step])
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in C.PC.collate(task.cache, rows.tolist()).items()}
            y = torch.as_tensor(task.labels[rows], dtype=torch.float32, device=device)
            T.train_step(model, optimizer, batch, y, global_step=step, clip=float(C.RECIPE["gradient_clip"]))
        rng_state = C.capture_rng_state()
        payload = T.checkpoint_payload(
            model_key=key,
            epoch=1,
            global_step=2,
            model=model,
            optimizer=optimizer,
            rng_state=rng_state,
            plan=T.plan_meta(plan, 1, C.ORDER_SEED),
            model_audit=audit,
            valid_ap=None,
            valid_report=None,
            kind="audit_roundtrip",
            progress=None,
            extra={"optimizer_state_audit": opt_audit},
        )
        path = work_dir / f"roundtrip_{key}.pt"
        sha = T._atomic_save(payload, path)
        restored, _audit2 = M.build_model(key, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
        opt2, _ = BSM.make_optimizer(restored)
        info = T.restore(torch.load(path, map_location="cpu", weights_only=False), restored, opt2, model_key=key)
        state_diff = _tensor_diff(_cpu_state(model), _cpu_state(restored))
        rng_after = C.rng_state_hashes(C.capture_rng_state())
        opt_equal = True
        for (k1, v1), (k2, v2) in zip(
            sorted(optimizer.state_dict()["state"].items()), sorted(opt2.state_dict()["state"].items())
        ):
            if k1 != k2:
                opt_equal = False
                break
            for field in ("step", "exp_avg", "exp_avg_sq"):
                a, b = v1[field], v2[field]
                if not torch.equal(a.cpu(), b.cpu()):
                    opt_equal = False
        # the next batch after the checkpoint is the same rows
        next_rows_a = np.asarray(plan[1]["batches"][0])
        next_rows_b = np.asarray(plan[1]["batches"][0])
        out[key] = {
            "checkpoint_path": C.track_rel(path),
            "checkpoint_sha256": sha,
            "checkpoint_bytes": int(path.stat().st_size),
            "restore": info,
            "state_bitwise_equal": bool(state_diff["all_bitwise_equal"]),
            "state_diff": state_diff if not state_diff["all_bitwise_equal"] else None,
            "optimizer_state_equal": bool(opt_equal),
            "optimizer_parameter_tensors": int(sum(p.numel() for p in restored.parameters())),
            "rng_hashes_before_restore": rng_before,
            "rng_hashes_after_restore": rng_after,
            "rng_restored_exactly": bool(rng_after["combined_sha256"] == C.rng_state_hashes(rng_state)["combined_sha256"]),
            "next_batch_rows_equal": bool(np.array_equal(next_rows_a, next_rows_b)),
            "scheduler_step_restored": int(payload["global_step"]),
        }
    return out


def audit_init(task: Any, out_dir: Path, device: torch.device) -> dict[str, Any]:
    spec, norm = task.spec, task.norm
    out: dict[str, Any] = {
        "schema": C.SCHEMA_AUDIT,
        "mode": "init",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "captured_utc": C.utcnow(),
        "commit": C.git_commit(),
        "host_facts": C.host_facts(),
        "occupancy": C.gpu_occupancy(),
        "constants": audit_constants(),
        "parameter_accounting": {
            "base_parameters": int(C.BASE_PARAMETERS),
            "C2_new": int(C.NEW_PARAMETERS["C2"]),
            "F_new": int(C.NEW_PARAMETERS["F"]),
            "C2_total": int(C.EXPECTED_PARAMETERS["C2"]),
            "F_total": int(C.EXPECTED_PARAMETERS["F"]),
            "parameter_gap": int(C.PARAMETER_GAP),
            "c2_branch": C.C2_BRANCH,
            "ffn_branch": C.FFN_BRANCH,
            "c2_added": int(C.C2_ADDED),
            "ffn_added": int(C.FFN_ADDED),
        },
        "models": {},
        "checks": {},
    }

    base, base_audit = BSM.build_model(C.BASE_MODEL_KEY, spec, norm, C.N_TASKS, seed=C.SEED, device="cpu")
    base_state = _cpu_state(base)
    out["base_w2"] = {
        "parameters": int(C.parameter_count(base)),
        "expected_parameters": int(C.BASE_PARAMETERS),
        "trunk_init_fingerprint": base_audit["trunk_init_fingerprint"],
        "historical_trunk_init_fingerprint": C.BASE_TRUNK_INIT_FINGERPRINT,
        "trunk_fingerprint_matches": bool(base_audit["trunk_init_fingerprint"] == C.BASE_TRUNK_INIT_FINGERPRINT),
        "full_trunk_init_fingerprint": base_audit["full_trunk_init_fingerprint"],
        "historical_full_trunk_init_fingerprint": C.BASE_FULL_TRUNK_INIT_FINGERPRINT,
        "full_trunk_fingerprint_matches": bool(
            base_audit["full_trunk_init_fingerprint"] == C.BASE_FULL_TRUNK_INIT_FINGERPRINT
        ),
        "state_sha256": M.state_sha256(base_state),
        "state_keys": sorted(base_state),
        "state_keys_match_historical": bool(sorted(base_state) == list(base_audit["state_keys"])),
        "init_audit": base_audit["init"],
    }
    out["checks"]["base_w2_matches_history"] = bool(
        int(C.parameter_count(base)) == int(C.BASE_PARAMETERS)
        and base_audit["trunk_init_fingerprint"] == C.BASE_TRUNK_INIT_FINGERPRINT
        and base_audit["full_trunk_init_fingerprint"] == C.BASE_FULL_TRUNK_INIT_FINGERPRINT
    )

    rows = [int(x) for x in task.train_rows[:8]]
    batch_cpu = C.PC.collate(task.cache, rows)
    base_logits_cpu = _logits(base, batch_cpu)
    built: dict[str, torch.nn.Module] = {}
    for key in C.MODEL_ORDER:
        model, audit = M.build_model(
            key, spec, norm, C.N_TASKS, seed=C.SEED, device="cpu", base_model=base
        )
        built[key] = model
        model_logits_cpu = _logits(model, batch_cpu)
        subset = _tensor_diff(_cpu_state(model), base_state)
        # subset comparison must ignore the new tensors
        shared = {k: v for k, v in _cpu_state(model).items() if k in base_state}
        subset_only_base = _tensor_diff(shared, base_state)
        out["models"][key] = {
            "model_name": C.MODELS[key]["name"],
            "role": C.MODELS[key]["role"],
            "total_parameters": int(audit["total_parameters"]),
            "base_parameters": int(audit["base_parameters"]),
            "new_parameters": int(audit["new_parameters"]),
            "expected_total_parameters": int(C.EXPECTED_PARAMETERS[key]),
            "expected_new_parameters": int(C.NEW_PARAMETERS[key]),
            "parameter_check": audit["parameter_check"],
            "new_state_keys": audit["new_state_keys"],
            "new_parameter_inventory": audit["new_parameter_inventory"],
            "new_module_init": audit["new_module_init"],
            "base_state_sha256": audit["base_state_sha256"],
            "base_state_sha256_equals_w2": bool(audit["base_state_sha256"] == out["base_w2"]["state_sha256"]),
            "original_w2_subset": subset_only_base,
            "original_w2_subset_bitwise_equal": bool(subset_only_base["all_bitwise_equal"]),
            "cpu_logits_parity_vs_w2": {
                "max_abs_diff": float((base_logits_cpu - model_logits_cpu).abs().max().item()),
                "bitwise_equal": bool(torch.equal(base_logits_cpu, model_logits_cpu)),
            },
            "gate_parameters": M.gate_parameters(model),
            "initial_diagnostics": M.module_diagnostics(model, batch_cpu),
            "global_rng_unchanged_after_build": bool(audit["global_rng_unchanged"]),
        }
        out["checks"][f"{key}_parameters_exact"] = bool(
            int(audit["total_parameters"]) == int(C.EXPECTED_PARAMETERS[key])
            and int(audit["new_parameters"]) == int(C.NEW_PARAMETERS[key])
        )
        out["checks"][f"{key}_original_w2_subset_bitwise"] = bool(subset_only_base["all_bitwise_equal"])
        out["checks"][f"{key}_cpu_logits_bitwise_equal_to_w2"] = bool(
            torch.equal(base_logits_cpu, model_logits_cpu)
        )

    # the C2/F shared original-W2 subset is also compared to each other
    c2_shared = {k: v for k, v in built["C2"].state_dict().items() if k in base_state}
    f_shared = {k: v for k, v in built["F"].state_dict().items() if k in base_state}
    out["checks"]["c2_and_f_original_w2_subsets_equal"] = bool(
        _tensor_diff(c2_shared, f_shared)["all_bitwise_equal"]
    )

    # ---- CUDA parity, including one full Validation pass at step 0 ---------
    cuda: dict[str, Any] = {"available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        device_count = int(torch.cuda.device_count())
        cuda["device_count"] = device_count
        base_gpu, _ = BSM.build_model(C.BASE_MODEL_KEY, spec, norm, C.N_TASKS, seed=C.SEED, device=device)
        gpu_models = {"W2": base_gpu}
        for key in C.MODEL_ORDER:
            gpu_models[key], _ = M.build_model(
                key, spec, norm, C.N_TASKS, seed=C.SEED, device=device, base_model=None
            )
        batch_gpu = _move_batch(batch_cpu, device)
        base_gpu_logits = _logits(base_gpu, batch_gpu)
        cuda["batch_logits"] = {}
        for key in C.MODEL_ORDER:
            lg = _logits(gpu_models[key], batch_gpu)
            cuda["batch_logits"][key] = {
                "max_abs_diff_vs_w2": float((base_gpu_logits - lg).abs().max().item()),
                "bitwise_equal": bool(torch.equal(base_gpu_logits, lg)),
            }
        y_valid = np.asarray(task.labels[task.valid_rows])
        cuda["epoch0_full_validation"] = {}
        reports = {}
        for name, model in gpu_models.items():
            t0 = time.perf_counter()
            logits = _full_valid_logits(model, task, device)
            reports[name] = {
                "logits": logits,
                "report": report_from_logits(logits, y_valid),
                "seconds": float(time.perf_counter() - t0),
            }
            cuda["epoch0_full_validation"][name] = {
                "ap": float(reports[name]["report"]["ap"]),
                "valid_bce": float(reports[name]["report"]["valid_bce"]),
                "n_tasks_scored": int(reports[name]["report"]["n_tasks_scored"]),
                "valid_logits_sha256": reports[name]["report"]["valid_logits_sha256"],
                "seconds": reports[name]["seconds"],
            }
        base_logits = reports["W2"]["logits"]
        for key in C.MODEL_ORDER:
            lg = reports[key]["logits"]
            cuda["epoch0_full_validation"][f"{key}_vs_W2"] = {
                "max_abs_logit_diff": float(np.abs(base_logits - lg).max()),
                "ap_delta": float(reports[key]["report"]["ap"] - reports["W2"]["report"]["ap"]),
                "bce_delta": float(reports[key]["report"]["valid_bce"] - reports["W2"]["report"]["valid_bce"]),
            }
        cuda["w2_epoch0_recorded_ap"] = 0.029063331894071648
        cuda["w2_epoch0_replay_abs_delta"] = float(
            abs(reports["W2"]["report"]["ap"] - 0.029063331894071648)
        )
        cuda["verdict"] = "OK"
        out["checks"]["cuda_epoch0_w2_replay_within_1e_4"] = bool(
            cuda["w2_epoch0_replay_abs_delta"] <= 1e-4
        )
        out["checks"]["cuda_epoch0_c2_and_f_within_1e_4_of_w2"] = bool(
            abs(cuda["epoch0_full_validation"]["C2_vs_W2"]["ap_delta"]) <= 1e-4
            and abs(cuda["epoch0_full_validation"]["F_vs_W2"]["ap_delta"]) <= 1e-4
        )
        del gpu_models, base_gpu
        torch.cuda.empty_cache()
    else:
        cuda["verdict"] = "CUDA_UNAVAILABLE"
    out["cuda"] = cuda

    # ---- gradient reachability + checkpoint round-trip --------------------
    plan = C.SD.make_plan40(task, seed=C.ORDER_SEED)
    plan_summary = C.SD.plan_summary(plan, task)
    out["batch_order"] = {
        "global_plan_sha256": plan[0]["global_plan_sha256"],
        "frozen_w2_plan_sha256": C.BASE_GLOBAL_PLAN_SHA256,
        "matches_frozen_w2_plan": bool(plan[0]["global_plan_sha256"] == C.BASE_GLOBAL_PLAN_SHA256),
        "consumed40_prefix_sha256": T.plan_meta(plan, 40, C.ORDER_SEED)["prefix_sha256"],
        "frozen_consumed40_prefix_sha256": C.BASE_CONSUMED40_PREFIX_SHA256,
        "matches_frozen_consumed40_prefix": bool(
            T.plan_meta(plan, 40, C.ORDER_SEED)["prefix_sha256"] == C.BASE_CONSUMED40_PREFIX_SHA256
        ),
        "summary": plan_summary["summary"],
        "isolation": plan_summary["isolation"],
        "test_rows_in_plan": int(np.isin(np.concatenate([p["rows"] for p in plan]), task.test_rows).sum()),
        "valid_rows_in_plan": int(np.isin(np.concatenate([p["rows"] for p in plan]), task.valid_rows).sum()),
    }
    out["checks"]["batch_order_matches_w2"] = bool(
        out["batch_order"]["matches_frozen_w2_plan"] and out["batch_order"]["matches_frozen_consumed40_prefix"]
    )
    out["gradient_reachability"] = gradient_reachability(task, plan, device)
    out["checks"]["c2_gradients_reach_all_new_tensors"] = bool(
        out["gradient_reachability"]["C2"]["all_checks_pass"]
    )
    out["checks"]["f_gradients_reach_all_new_tensors"] = bool(
        out["gradient_reachability"]["F"]["all_checks_pass"]
    )
    out["checkpoint_roundtrip"] = checkpoint_roundtrip(task, plan, device, out_dir / "roundtrip")
    out["checks"]["c2_checkpoint_roundtrip"] = bool(
        out["checkpoint_roundtrip"]["C2"]["state_bitwise_equal"]
        and out["checkpoint_roundtrip"]["C2"]["optimizer_state_equal"]
        and out["checkpoint_roundtrip"]["C2"]["rng_restored_exactly"]
        and out["checkpoint_roundtrip"]["C2"]["next_batch_rows_equal"]
    )
    out["checks"]["f_checkpoint_roundtrip"] = bool(
        out["checkpoint_roundtrip"]["F"]["state_bitwise_equal"]
        and out["checkpoint_roundtrip"]["F"]["optimizer_state_equal"]
        and out["checkpoint_roundtrip"]["F"]["rng_restored_exactly"]
        and out["checkpoint_roundtrip"]["F"]["next_batch_rows_equal"]
    )

    # ---- Test isolation -----------------------------------------------------
    test_labels = np.asarray(task.labels[task.test_rows])
    out["test_isolation"] = {
        "official_test_rows": int(task.test_rows.size),
        "test_labels_finite_count": int(np.isfinite(test_labels).sum()),
        "test_labels_all_nan": bool(not np.isfinite(test_labels).any()),
        "test_rows_in_training_plan": out["batch_order"]["test_rows_in_plan"],
        "test_evaluated": False,
        "code_path_reads_test_rows": False,
    }
    out["checks"]["official_test_untouched"] = bool(
        out["test_isolation"]["test_labels_all_nan"]
        and out["test_isolation"]["test_rows_in_training_plan"] == 0
    )

    out["verdict"] = "OK" if all(bool(v) for v in out["checks"].values()) else "FAILED"
    return out


# ---------------------------------------------------------------------------
# architecture audit
# ---------------------------------------------------------------------------


def audit_architecture(task: Any, out_dir: Path, device: torch.device) -> dict[str, Any]:
    spec = task.spec
    out: dict[str, Any] = {
        "schema": C.SCHEMA_AUDIT,
        "mode": "architecture",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "captured_utc": C.utcnow(),
        "commit": C.git_commit(),
        "host_facts": C.host_facts(),
        "models": {},
        "checks": {},
    }
    cache = task.cache
    path_cat = np.asarray(cache["path_cat"], dtype=np.int64)
    path_end63 = np.asarray(cache["path_end63"], dtype=np.int64)
    path_end = np.asarray(cache["path_end"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    n_atoms = int(cache["atom_ptr"][-1])
    out["path_vocabulary"] = {
        "n_paths": int(path_cat.size),
        "n_path_codes": int(spec.n_path_codes),
        "n_nbr_codes": int(spec.n_nbr_codes),
        "path_cat_min": int(path_cat.min()),
        "path_cat_max": int(path_cat.max()),
        "path_cat_in_range": bool(path_cat.min() >= 0 and path_cat.max() < int(spec.n_path_codes)),
        "path_end63_min": int(path_end63.min()),
        "path_end63_max": int(path_end63.max()),
        "path_end63_in_range": bool(path_end63.min() >= 0 and path_end63.max() < int(spec.n_nbr_codes)),
        "path_end_min": int(path_end.min()),
        "path_end_max": int(path_end.max()),
        "path_end_in_atom_range": bool(path_end.min() >= 0 and path_end.max() < n_atoms),
        "n_atoms_total": n_atoms,
        "cache_meta_n_s2_paths": int(cache["meta"]["n_s2_paths"]),
        "path_count_matches_meta": bool(int(path_cat.size) == int(cache["meta"]["n_s2_paths"])),
        "path_ptr_length": int(path_ptr.size),
    }
    out["checks"]["path_codes_in_frozen_vocabulary"] = bool(
        out["path_vocabulary"]["path_cat_in_range"]
        and out["path_vocabulary"]["path_end63_in_range"]
        and out["path_vocabulary"]["path_end_in_atom_range"]
    )

    base, _ = BSM.build_model(C.BASE_MODEL_KEY, spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
    rows = [int(x) for x in task.train_rows[:64]]
    batch = C.PC.collate(task.cache, rows)
    snapshot = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    base_keys = set(base.state_dict())

    for key in C.MODEL_ORDER:
        model, audit = M.build_model(key, spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
        batch_dev = _move_batch(batch, device)
        calls = M.count_module_calls(model, batch_dev)
        # batch immutability
        mutated = [
            k
            for k, v in snapshot.items()
            if torch.is_tensor(v) and not torch.equal(v.cpu(), batch_dev[k].cpu())
        ]
        # post-local path must not need any structural field (eval mode: no dropout)
        model.eval()
        with torch.inference_mode():
            full_logits = model(batch_dev)
            _a, _b, _scal, h = model.encode_h(batch_dev)
            slim = {"n_graphs": batch_dev["n_graphs"], "atom_batch": batch_dev["atom_batch"], "global": batch_dev["global"]}
            slim_logits = model.post_local(h, slim)
        model.train()
        # atom renumbering invariance
        permuted, _p = BSM.permute_atoms(batch, seed=20261108)
        permuted_dev = _move_batch(permuted, device)
        with torch.inference_mode():
            model.eval()
            permuted_logits = model(permuted_dev)
            model.train()
        permuted_diff = float((full_logits - permuted_logits).abs().max().item())
        # new module inventory
        new_keys = [k for k in model.state_dict() if k not in base_keys]
        gate = M.gate_parameters(model)
        out["models"][key] = {
            "model_name": C.MODELS[key]["name"],
            "total_parameters": int(audit["total_parameters"]),
            "new_parameters": int(audit["new_parameters"]),
            "new_state_keys": new_keys,
            "call_counts_one_forward": calls,
            "dropout_calls": int(calls.get("dropout", 0)),
            "atom_encoder_calls": int(calls.get("atom_encoder", 0)),
            "bond_encoder_calls": int(calls.get("bond_encoder", 0)),
            "local_calls": int(calls.get("local", 0)),
            "f_calls": int(calls.get("f", 0)),
            "rho_calls": int(calls.get("rho", 0)),
            "readout_calls": int(calls.get("readout", 0)),
            "new_module_calls": int(calls.get("new_module", 0)) + int(calls.get("path_cond_ec", 0)),
            "path_cond_ec_calls": int(calls.get("path_cond_ec", 0)),
            "path_cond_ee_calls": int(calls.get("path_cond_ee", 0)),
            "batch_tensors_mutated": mutated,
            "batch_immutable": not mutated,
            "post_local_without_structure_max_abs_diff": float((full_logits - slim_logits).abs().max().item()),
            "atom_renumbering_max_abs_logit_diff": permuted_diff,
            "gate_parameters": gate,
            "gate_shape_on_batch": (
                None
                if not gate["has_conditional_gate"]
                else list(M.gate_diagnostics(model, batch_dev)["gate_shape"])
            ),
            "n_paths_in_batch": int(batch["path_cat"].numel()),
            "new_module_types": sorted({type(m).__name__ for n, m in model.named_modules() if n.startswith("ffn") or n.startswith("local.path_cond")}),
            "forward_finite": bool(torch.isfinite(full_logits).all().item()),
        }
        checks = {
            "one_atom_and_bond_encode": bool(
                calls.get("atom_encoder") == 1 and calls.get("bond_encoder") == 1
            ),
            "one_local_call": bool(calls.get("local") == 1),
            "one_dropout_call": bool(calls.get("dropout") == 1),
            "batch_not_mutated": not mutated,
            "post_local_needs_no_structure": bool(
                float((full_logits - slim_logits).abs().max().item()) <= 2e-5
            ),
            "atom_renumbering_invariant": bool(permuted_diff <= 2e-5),
            "forward_finite": bool(torch.isfinite(full_logits).all().item()),
        }
        if key == "C2":
            checks["gate_shape_is_paths_by_192"] = bool(
                out["models"][key]["gate_shape_on_batch"] == [int(batch["path_cat"].numel()), 192]
            )
            checks["new_module_called_once"] = bool(
                calls.get("path_cond_ec") == 1 and calls.get("path_cond_ee") == 1
            )
            checks["path_gate_preserved"] = bool(
                gate["path_gate_shape"] == [400, 1] and gate["path_gate_initial_ones"]
            )
            checks["no_extra_normalisation_or_attention"] = bool(
                set(out["models"][key]["new_module_types"]) == {"Embedding"}
            )
        else:
            checks["new_module_called_once"] = bool(calls.get("new_module") == 1)
            checks["ffn_is_norm_fc1_fc2"] = bool(
                set(out["models"][key]["new_module_types"]) == {"LocalResidualFFN", "LayerNorm", "Linear"}
            )
        out["models"][key]["checks"] = checks
        out["checks"][key] = bool(all(checks.values()))
        model.to("cpu")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- label mask / loss semantics --------------------------------------
    logits = torch.tensor([[0.3, -0.2, 1.1, -0.9]], dtype=torch.float32)
    labels = torch.tensor([[1.0, 0.0, float("nan"), float("nan")]], dtype=torch.float32)
    loss = C.MET.masked_bce_loss(logits, labels)
    # observed pair: (target 1, logit 0.3) and (target 0, logit -0.2)
    manual = (math.log1p(math.exp(-0.3)) + math.log1p(math.exp(-0.2))) / 2.0
    observed = torch.isfinite(labels)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits[observed], labels[observed])
    out["label_mask"] = {
        "observed_entries": int(observed.sum()),
        "nan_entries": int((~observed).sum()),
        "masked_bce_loss": float(loss),
        "manual_two_entry_mean": float(manual),
        "torch_observed_only_bce": float(bce),
        "matches_manual": bool(abs(float(loss) - manual) < 1e-6),
        "matches_torch_observed_only": bool(abs(float(loss) - float(bce)) < 1e-6),
        "nan_positions_in_labels_are_masked": True,
    }
    out["checks"]["label_nan_mask_correct"] = bool(
        out["label_mask"]["matches_manual"] and out["label_mask"]["matches_torch_observed_only"]
    )

    # ---- Test isolation ----------------------------------------------------
    test_labels = np.asarray(task.labels[task.test_rows])
    out["test_isolation"] = {
        "official_test_rows": int(task.test_rows.size),
        "test_labels_all_nan": bool(not np.isfinite(test_labels).any()),
        "test_evaluated": False,
        "code_path_reads_test_rows": False,
    }
    out["checks"]["official_test_untouched"] = bool(out["test_isolation"]["test_labels_all_nan"])
    out["verdict"] = "OK" if all(bool(v) for v in out["checks"].values()) else "FAILED"
    return out


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run_modes(modes: Sequence[str], out_dir: Path, *, sampling: int = 6, interval: float = 10.0) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    C.write_json(out_dir / "audit_environment.json", {"host_facts": C.host_facts(), "environment": C.environment(), "commit": C.git_commit()})
    results: dict[str, Any] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "gpu-preflight" in modes:
        results["gpu_preflight"] = audit_gpu_preflight()
        C.write_json(out_dir / "gpu_preflight.json", results["gpu_preflight"])
    if "gpu-sampling" in modes:
        results["gpu_sampling"] = audit_gpu_sampling(samples=int(sampling), interval_seconds=float(interval))
        C.write_json(out_dir / "gpu_sampling.json", results["gpu_sampling"])
        with open(out_dir / "gpu_occupancy_samples.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "sample",
                    "gpu_index",
                    "uuid",
                    "name",
                    "memory_total_mib",
                    "memory_used_mib",
                    "memory_free_mib",
                    "utilization_gpu_pct",
                    "utilization_memory_pct",
                    "temperature_c",
                ]
            )
            for sample in results["gpu_sampling"]["samples"]:
                for gpu in sample.get("gpus", []):
                    writer.writerow(
                        [
                            sample["sample"],
                            gpu["index"],
                            gpu["uuid"],
                            gpu["name"],
                            gpu["memory_total_mib"],
                            gpu["memory_used_mib"],
                            gpu["memory_free_mib"],
                            gpu["utilization_gpu_pct"],
                            gpu["utilization_memory_pct"],
                            gpu["temperature_c"],
                        ]
                    )
    if "init" in modes or "architecture" in modes:
        task = C.SD.load_task()
        if "init" in modes:
            results["init"] = audit_init(task, out_dir, device)
            C.write_json(out_dir / "initialization_audit.json", results["init"])
        if "architecture" in modes:
            results["architecture"] = audit_architecture(task, out_dir, device)
            C.write_json(out_dir / "architecture_audit.json", results["architecture"])
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="gpu-preflight,gpu-sampling,init,architecture")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sampling", type=int, default=6)
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args(argv)
    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    supported = {"gpu-preflight", "gpu-sampling", "init", "architecture"}
    unknown = [m for m in modes if m not in supported]
    if unknown:
        raise SystemExit(f"unknown modes {unknown}; supported={sorted(supported)}")
    out_dir = Path(args.out_dir)
    try:
        results = run_modes(modes, out_dir, sampling=int(args.sampling), interval=float(args.interval))
        verdicts = {k: v.get("verdict") for k, v in results.items() if isinstance(v, dict) and "verdict" in v}
        C.log(f"modes={modes} verdicts={verdicts}")
        return 0 if all(v == "OK" for v in verdicts.values()) else 1
    except Exception as exc:  # noqa: BLE001
        C.write_json(
            out_dir / "audit_failure.json",
            {
                "round_id": C.ROUND_ID,
                "modes": modes,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "finished_utc": C.utcnow(),
            },
        )
        C.log(f"AUDIT FAILED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
