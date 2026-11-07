"""Audits for the CPE round.

Modes
-----
``gpu-preflight``   device / NVML / per-visible-GPU allocation probes.
``gpu-sampling``    6 x 10 s ``nvidia-smi`` samples of both GPUs (occupancy facts).
``init``            W2 init identity, exact parameter counts, P/S new-tensor
                    equality, initial logits/AP parity, gradient reachability,
                    checkpoint round-trip, Test isolation.
                    -> ``analysis/initialization_parity.json``
``architecture``    the 18 frozen architecture checks (call counts, dropout
                    count, permutation invariance, A/B swap, n==1, cross-molecule
                    isolation, synthetic double-loop equivalence, finiteness,
                    label mask).
                    -> ``analysis/architecture_audit.json``
``mechanism``       post-training mechanism audit of the trained P/S: alpha
                    norms, main/pair residual magnitudes, CROSS/SELF values on
                    the fixed 256 Validation molecules, and the diagnostic-only
                    ``alpha_pair = 0`` ablation of P.
                    -> ``analysis/evidence_mechanism_audit.json``

Nothing here trains a formal model or touches an official Test row.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import cpe_common as C
import cpe_model as M
import cpe_train as T
import bs_model as BSM


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


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
        "max_abs_diff": max([v.get("max_abs_diff", float("inf")) for v in diffs.values()], default=0.0),
        "all_bitwise_equal": not diffs and keys_a == keys_b,
    }


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


def _fixed_batch(task: Any, n: int = C.TRAIN_DIAGNOSTIC_MOLECULES) -> dict[str, Any]:
    """Fixed eval batch: the first ``n`` Train rows of the frozen split order."""
    rows = [int(r) for r in np.asarray(task.train_rows[:n])]
    return C.PC.collate(task.cache, rows)


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
            probe["mem_get_info"] = {"free_bytes": int(free), "total_bytes": int(total), "used_bytes": int(total - free)}
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
    ok = all(bool(p.get("ok")) for p in out["probes"])
    out["verdict"] = "OK" if ok else "PROBE_FAILED"
    return out


def _gpu_rows() -> list[dict[str, Any]]:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=20,
        ).strip()
    except Exception as exc:  # pragma: no cover
        return [{"error": f"{type(exc).__name__}: {exc}"}]
    rows = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mib": int(parts[2]),
                "memory_used_mib": int(parts[3]),
                "memory_free_mib": int(parts[4]),
                "utilization_gpu_pct": int(parts[5]),
                "utilization_memory_pct": int(parts[6]),
                "driver": parts[7],
            }
        )
    return rows


def audit_gpu_sampling(samples: int = 6, interval_seconds: float = 10.0) -> dict[str, Any]:
    out: dict[str, Any] = {"captured_utc": C.utcnow(), "samples": [], "interval_seconds": float(interval_seconds)}
    csv_rows = ["sample,gpu,memory_used_mib,memory_free_mib,utilization_gpu_pct,utilization_memory_pct"]
    for i in range(int(samples)):
        rows = _gpu_rows()
        out["samples"].append({"sample": i, "utc": C.utcnow(), "gpus": rows})
        for row in rows:
            if "error" in row:
                continue
            csv_rows.append(
                f"{i},{row['index']},{row['memory_used_mib']},{row['memory_free_mib']},"
                f"{row['utilization_gpu_pct']},{row['utilization_memory_pct']}"
            )
        if i + 1 < int(samples):
            time.sleep(float(interval_seconds))
    summary: dict[str, Any] = {}
    for gpu in (0, 1):
        utils = [
            s["gpus"][gpu]["utilization_gpu_pct"]
            for s in out["samples"]
            if len(s["gpus"]) > gpu and "error" not in s["gpus"][gpu]
        ]
        free = [
            s["gpus"][gpu]["memory_free_mib"]
            for s in out["samples"]
            if len(s["gpus"]) > gpu and "error" not in s["gpus"][gpu]
        ]
        if utils:
            summary[f"gpu{gpu}"] = {
                "utilization_samples": utils,
                "max_utilization_pct": int(max(utils)),
                "min_free_mib": int(min(free)),
                "saturated": bool(min(utils) >= 95),
            }
    out["summary"] = summary
    out["verdict"] = (
        "OK" if all("error" not in r for s in out["samples"] for r in s["gpus"]) else "NVIDIA_SMI_UNAVAILABLE"
    )
    out["csv"] = "\n".join(csv_rows)
    return out


# ---------------------------------------------------------------------------
# init parity
# ---------------------------------------------------------------------------


def gradient_reachability(
    task: Any,
    model_keys: Sequence[str],
    device: torch.device,
    *,
    steps: int = C.PROBE_TRAIN_STEPS,
    seed: int = C.SEED,
) -> dict[str, Any]:
    """Step 1..N on the frozen epoch-1 order; record which new tensors got grads."""
    plan = C.SD.make_plan40(task, seed=int(seed))
    rows = [np.asarray(plan[0]["batches"][i]) for i in range(int(steps))]
    out: dict[str, Any] = {"steps": int(steps), "models": {}}
    for key in model_keys:
        C.seed_everything(int(seed))
        model, audit = M.build_model(key, task.spec, task.norm, C.N_TASKS, seed=int(seed), device="cpu")
        model.to(device)
        optimizer, _ = BSM.make_optimizer(model)
        per_step: list[dict[str, Any]] = []
        for step in range(int(steps)):
            batch = _move_batch(C.PC.collate(task.cache, rows[step].tolist()), device)
            y = task.labels[rows[step]].to(device) if torch.is_tensor(task.labels) else torch.as_tensor(task.labels[rows[step]], device=device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = C.MET.masked_bce_loss(logits, y)
            loss.backward()
            grads: dict[str, Any] = {}
            for name in (
                "evidence.linear_A.weight",
                "evidence.linear_A.bias",
                "evidence.linear_B.weight",
                "evidence.linear_B.bias",
                "evidence.alpha_A",
                "evidence.alpha_B",
                "evidence.alpha_pair",
            ):
                p = dict(model.named_parameters())[name]
                g = p.grad
                grads[name] = {
                    "is_none": g is None,
                    "abs_sum": 0.0 if g is None else float(g.detach().float().abs().sum().item()),
                    "l2": 0.0 if g is None else float(g.detach().float().norm().item()),
                    "finite": True if g is None else bool(torch.isfinite(g).all().item()),
                    "nonzero": False if g is None else bool(torch.count_nonzero(g).item() > 0),
                }
            all_params_with_grad = all(p.grad is not None for p in model.parameters())
            per_step.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "new_module_grads": grads,
                    "all_parameters_have_grad": bool(all_params_with_grad),
                }
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(C.RECIPE["gradient_clip"]))
            BSM.set_lrs(optimizer, step=step)
            optimizer.step()
        step1 = per_step[0]["new_module_grads"]
        later = per_step[1:] if len(per_step) > 1 else []
        checks = {
            "alphas_nonzero_step1": all(step1[f"evidence.alpha_{a}"]["nonzero"] for a in ("A", "B", "pair")),
            "linear_zero_step1": all(
                not step1[f"evidence.linear_{a}.{part}"]["nonzero"] for a in ("A", "B") for part in ("weight", "bias")
            ),
            "linear_nonzero_after_step1": all(
                any(s["new_module_grads"][f"evidence.linear_{a}.{part}"]["nonzero"] for s in later)
                for a in ("A", "B")
                for part in ("weight", "bias")
            )
            if later
            else None,
            "all_finite": all(
                v["finite"] for s in per_step for v in s["new_module_grads"].values()
            ),
            "all_parameters_have_grad": all(s["all_parameters_have_grad"] for s in per_step),
        }
        out["models"][key] = {"per_step": per_step, "checks": checks, "all_passed": all(v is not False for v in checks.values())}
    return out


def checkpoint_roundtrip(model_key: str, task: Any, device: torch.device, out_dir: Path, *, seed: int = C.SEED) -> dict[str, Any]:
    """One real optimizer step, save, rebuild, restore, compare exactly."""
    C.seed_everything(int(seed))
    model, audit = M.build_model(model_key, task.spec, task.norm, C.N_TASKS, seed=int(seed), device="cpu")
    model.to(device)
    optimizer, _ = BSM.make_optimizer(model)
    plan = C.SD.make_plan40(task, seed=int(seed))
    rows = np.asarray(plan[0]["batches"][0])
    batch = _move_batch(C.PC.collate(task.cache, rows.tolist()), device)
    y = torch.as_tensor(task.labels[rows], device=device)
    model.train()
    out = T.train_step(model, optimizer, batch, y, global_step=0, clip=float(C.RECIPE["gradient_clip"]))
    state_before = _cpu_state(model)
    logits_before = _logits(model, batch.to("cpu") if device.type != "cpu" else batch)

    payload = T.checkpoint_payload(
        model_key=model_key,
        epoch=1,
        global_step=1,
        model=model,
        optimizer=optimizer,
        rng_state=C.capture_rng_state(),
        plan=T.plan_meta(plan, 1, int(seed)),
        model_audit=audit,
        valid_ap=None,
        valid_report=None,
        kind="audit_roundtrip",
    )
    path = Path(out_dir) / f"audit_{model_key}_roundtrip.pt"
    sha = T._atomic_save(payload, path)

    C.seed_everything(1234)  # deliberately different RNG state
    fresh, _ = M.build_model(model_key, task.spec, task.norm, C.N_TASKS, seed=int(seed), device="cpu")
    fresh.to(device)
    fresh_opt, _ = BSM.make_optimizer(fresh)
    restored = T.restore(payload, fresh, fresh_opt, model_key=model_key)
    state_after = _cpu_state(fresh)
    diff = _tensor_diff(state_before, state_after)
    logits_after = _logits(fresh, batch.to("cpu") if device.type != "cpu" else batch)
    return {
        "model": model_key,
        "checkpoint": C.track_rel(path),
        "checkpoint_sha256": sha,
        "checkpoint_seconds": None,
        "train_step": out,
        "state_identical": bool(diff["all_bitwise_equal"]),
        "max_abs_state_diff": float(diff["max_abs_diff"]),
        "logits_max_abs_diff": float((logits_before - logits_after).abs().max().item()),
        "restored": restored,
        "pair_mode_restored": restored.get("pair_mode_restored"),
        "optimizer_step_restored": restored.get("resumed_global_step"),
        "all_passed": bool(diff["all_bitwise_equal"]),
    }


def audit_init(task: Any, out_dir: Path, device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": C.SCHEMA_AUDIT,
        "round_id": C.ROUND_ID,
        "mode": "init",
        "captured_utc": C.utcnow(),
        "host_facts": C.host_facts(),
        "device": str(device),
        "module_paths": {
            "cpe_model": C.track_rel(Path(M.__file__).resolve()),
            "cpe_common": C.track_rel(Path(C.__file__).resolve()),
            "cpe_train": C.track_rel(Path(T.__file__).resolve()),
            "bs_model": C.track_rel(Path(BSM.__file__).resolve()),
        },
        "symbol_hashes": {},
        "parameter_targets": {
            "base": int(C.BASE_PARAMETERS),
            "new": int(C.NEW_PARAMETERS),
            "total": int(C.EXPECTED_TOTALS["P"]),
            "branch": C.NEW_BRANCH,
        },
    }
    for name in ("cpe_model", "cpe_common", "cpe_train"):
        mod = {"cpe_model": M, "cpe_common": C, "cpe_train": T}[name]
        path = Path(mod.__file__).resolve()
        out["symbol_hashes"][name] = C.sha256_file(path)

    # ---- constants -------------------------------------------------------
    constants = {
        "new_parameters_from_branch": int(C.NEW_PARAMETERS),
        "new_parameters_expected": 33_408,
        "totals_expected": int(C.EXPECTED_TOTALS["P"]),
        "totals_expected_value": 1_027_732,
        "linear_A_arithmetic": C.NEW_BRANCH["linear_A"]["parameters"],
        "linear_B_arithmetic": C.NEW_BRANCH["linear_B"]["parameters"],
        "alpha_total": sum(3 * C.N_TASKS for _ in range(1)),
        "evidence_width_equals_local_width": int(C.WIDTHS["width"]) == int(C.EVIDENCE_WIDTH) == 128,
        "evidence_width_equals_n_tasks": int(C.EVIDENCE_WIDTH) == int(C.N_TASKS) == 128,
    }
    constants["checks"] = {
        "new_parameters_is_33408": constants["new_parameters_from_branch"] == 33_408,
        "total_is_1027732": constants["totals_expected"] == 1_027_732,
        "linear_arithmetic": constants["linear_A_arithmetic"] == 16_512 and constants["linear_B_arithmetic"] == 16_512,
        "alpha_total_is_384": constants["alpha_total"] == 384,
        "width_binding": constants["evidence_width_equals_local_width"]
        and constants["evidence_width_equals_n_tasks"],
    }
    out["constants"] = constants

    # ---- build W2 and both arms -----------------------------------------
    base, base_audit = BSM.build_model(C.BASE_MODEL_KEY, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
    base_state = _cpu_state(base)
    models: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    for key in C.MODEL_ORDER:
        C.seed_everything(C.SEED)
        model, audit = M.build_model(
            key, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu", base_model=base
        )
        models[key] = model
        audits[key] = audit

    param_table = {
        "W2": {
            "base": int(base_audit["base_parameters"]),
            "readout": int(base_audit["readout_parameters"]),
            "total": int(base_audit["total_parameters"]),
        }
    }
    for key in C.MODEL_ORDER:
        param_table[key] = {
            "base": int(audits[key]["base_parameters"]),
            "new": int(audits[key]["new_parameters"]),
            "total": int(audits[key]["total_parameters"]),
        }
    out["parameter_counts"] = param_table
    out["parameter_checks"] = {
        "P_total_exact": param_table["P"]["total"] == 1_027_732,
        "S_total_exact": param_table["S"]["total"] == 1_027_732,
        "P_new_exact": param_table["P"]["new"] == 33_408,
        "S_new_exact": param_table["S"]["new"] == 33_408,
        "P_base_is_W2": param_table["P"]["base"] == int(C.BASE_PARAMETERS),
        "S_base_is_W2": param_table["S"]["base"] == int(C.BASE_PARAMETERS),
        "P_S_equal": param_table["P"] == param_table["S"],
        "sum_by_numel_P": int(sum(p.numel() for p in models["P"].parameters())) == 1_027_732,
        "sum_by_numel_S": int(sum(p.numel() for p in models["S"].parameters())) == 1_027_732,
    }

    # ---- original W2 subset bitwise, new tensors equal between arms -----
    p_state, s_state = _cpu_state(models["P"]), _cpu_state(models["S"])
    subset_p = _tensor_diff(base_state, {k: v for k, v in p_state.items() if k in base_state})
    subset_s = _tensor_diff(base_state, {k: v for k, v in s_state.items() if k in base_state})
    new_keys = sorted(audits["P"]["new_state_keys"])
    if new_keys != sorted(audits["S"]["new_state_keys"]):
        raise RuntimeError("P/S new state keys differ")
    new_diff_pc = _tensor_diff({k: p_state[k] for k in new_keys}, {k: s_state[k] for k in new_keys})
    out["w2_subset"] = {
        "n_tensors": len(base_state),
        "P": subset_p,
        "S": subset_s,
        "base_trunk_init_fingerprint_P": C.state_fingerprint(
            {k: v for k, v in p_state.items() if not k.startswith("readout.") and not k.startswith("evidence.")}
        ),
        "base_trunk_init_fingerprint_expected": C.BASE_TRUNK_INIT_FINGERPRINT,
        "base_full_trunk_init_fingerprint_expected": C.BASE_FULL_TRUNK_INIT_FINGERPRINT,
        "base_full_trunk_init_fingerprint_W2": base_audit["full_trunk_init_fingerprint"],
    }
    out["new_tensors"] = {
        "keys": new_keys,
        "P_vs_S": new_diff_pc,
        "inventory_P": audits["P"]["new_parameter_inventory"],
        "inventory_S": audits["S"]["new_parameter_inventory"],
    }
    out["arms_differ_only_in_pair_mode"] = {
        "pair_mode_P": str(models["P"].evidence.pair_mode),
        "pair_mode_S": str(models["S"].evidence.pair_mode),
        "parameter_names_equal": sorted(dict(models["P"].named_parameters())) == sorted(dict(models["S"].named_parameters())),
        "parameter_shapes_equal": {
            k: tuple(v.shape) for k, v in models["P"].state_dict().items()
        }
        == {k: tuple(v.shape) for k, v in models["S"].state_dict().items()},
        "state_bitwise_equal": _tensor_diff(p_state, s_state)["all_bitwise_equal"],
        "state_sha256_P": M.state_sha256(p_state),
        "state_sha256_S": M.state_sha256(s_state),
        "only_difference": "PredictiveEvidence.pair_mode ('cross' vs 'self'), a non-parameter attribute",
    }
    out["init"] = {k: audits[k]["new_module_init"] for k in C.MODEL_ORDER}
    out["global_rng_unchanged_on_build"] = {k: bool(audits[k]["global_rng_unchanged"]) for k in C.MODEL_ORDER}

    # ---- initial logits parity ------------------------------------------
    batch = _fixed_batch(task)
    logits: dict[str, torch.Tensor] = {"W2": _logits(base, batch)}
    for key in C.MODEL_ORDER:
        logits[key] = _logits(models[key], batch)
    pairs = {
        "W2_vs_P": (logits["W2"], logits["P"]),
        "W2_vs_S": (logits["W2"], logits["S"]),
        "P_vs_S": (logits["P"], logits["S"]),
    }
    out["initial_logits_cpu"] = {
        "batch": "first 64 Train rows of the frozen split order",
        "logits_shape": list(logits["W2"].shape),
        "logits_abs_mean": float(logits["W2"].abs().mean().item()),
        "comparisons": {
            name: {
                "max_abs_diff": float((a - b).abs().max().item()),
                "bitwise_equal": bool(torch.equal(a, b)),
                "sha256": C.sha256_bytes(np.ascontiguousarray(a.numpy(), dtype=np.float32).tobytes()),
            }
            for name, (a, b) in pairs.items()
        },
    }
    out["initial_logits_cpu"]["all_bitwise_equal"] = all(
        v["bitwise_equal"] for v in out["initial_logits_cpu"]["comparisons"].values()
    )

    # ---- CUDA parity + full Validation AP at step 0 ----------------------
    cuda_ok = device.type == "cuda" and torch.cuda.is_available()
    if cuda_ok:
        base_c = BSM.build_model(C.BASE_MODEL_KEY, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)[0]
        arms_c = {
            key: M.build_model(key, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)[0]
            for key in C.MODEL_ORDER
        }
        batch_c = _move_batch(batch, device)
        lg = {"W2": _logits(base_c, batch_c)}
        for key in C.MODEL_ORDER:
            lg[key] = _logits(arms_c[key], batch_c)
        out["initial_logits_cuda"] = {
            "tolerance": 1e-4,
            "comparisons": {
                f"{a}_vs_{b}": {
                    "max_abs_diff": float((lg[a] - lg[b]).abs().max().item()),
                }
                for a, b in (("W2", "P"), ("W2", "S"), ("P", "S"))
            },
        }
        out["initial_logits_cuda"]["all_within_tolerance"] = all(
            v["max_abs_diff"] <= 1e-4 for v in out["initial_logits_cuda"]["comparisons"].values()
        )
        t0 = time.perf_counter()
        reports = {
            "W2": C.SD.evaluate_validation(base_c, task, device, reason="audit_initial_w2"),
        }
        for key in C.MODEL_ORDER:
            reports[key] = C.SD.evaluate_validation(arms_c[key], task, device, reason=f"audit_initial_{key}")
        out["initial_valid_ap"] = {
            "seconds": float(time.perf_counter() - t0),
            "ap": {k: float(v["ap"]) for k, v in reports.items()},
            "valid_bce": {k: float(v["valid_bce"]) for k, v in reports.items()},
            "logits_sha256": {k: v.get("valid_logits_sha256") for k, v in reports.items()},
            "max_ap_diff_vs_w2": max(abs(float(reports[k]["ap"]) - float(reports["W2"]["ap"])) for k in C.MODEL_ORDER),
            "tolerance": 1e-4,
        }
        out["initial_valid_ap"]["all_within_tolerance"] = bool(out["initial_valid_ap"]["max_ap_diff_vs_w2"] <= 1e-4)
        out["cuda_peak_after_parity"] = C.gpu_peak(device)
        del base_c, arms_c
        torch.cuda.empty_cache()
    else:
        out["initial_logits_cuda"] = {"skipped": "device is not CUDA"}
        out["initial_valid_ap"] = {"skipped": "device is not CUDA"}

    # ---- gradient reachability ------------------------------------------
    out["gradient_reachability"] = gradient_reachability(task, C.MODEL_ORDER, device)
    out["gradient_checks"] = {
        "P_ok": bool(out["gradient_reachability"]["models"]["P"]["all_passed"]),
        "S_ok": bool(out["gradient_reachability"]["models"]["S"]["all_passed"]),
    }

    # ---- checkpoint round-trip ------------------------------------------
    out["checkpoint_roundtrip"] = {
        key: checkpoint_roundtrip(key, task, torch.device("cpu"), out_dir) for key in C.MODEL_ORDER
    }
    out["checkpoint_roundtrip"]["all_passed"] = all(
        bool(out["checkpoint_roundtrip"][k]["all_passed"]) for k in C.MODEL_ORDER
    )

    # ---- Test isolation --------------------------------------------------
    yv = np.asarray(task.labels[task.valid_rows])
    yt = np.asarray(task.labels[task.test_rows])
    plan = C.SD.make_plan40(task, seed=C.SEED)
    isolation = C.SD.plan_summary(plan, task)["isolation"]
    out["test_isolation"] = {
        "test_rows": int(np.asarray(task.test_rows).size),
        "test_labels_all_nan": bool(np.isnan(yt).all()),
        "test_labels_finite_count": int(np.isfinite(yt).sum()),
        "valid_labels_finite_count": int(np.isfinite(yv).sum()),
        "split_partition_disjoint": bool(
            set(np.asarray(task.train_rows).tolist()).isdisjoint(np.asarray(task.valid_rows).tolist())
            and set(np.asarray(task.train_rows).tolist()).isdisjoint(np.asarray(task.test_rows).tolist())
            and set(np.asarray(task.valid_rows).tolist()).isdisjoint(np.asarray(task.test_rows).tolist())
        ),
        "order_plan_isolation": isolation,
        "test_evaluated": False,
    }

    checks = {
        "constants": all(constants["checks"].values()),
        "parameters": all(out["parameter_checks"].values()),
        "w2_subset_bitwise": bool(subset_p["all_bitwise_equal"] and subset_s["all_bitwise_equal"]),
        "new_tensors_equal_between_arms": bool(new_diff_pc["all_bitwise_equal"]),
        "arms_differ_only_in_pair_mode": bool(
            out["arms_differ_only_in_pair_mode"]["parameter_names_equal"]
            and out["arms_differ_only_in_pair_mode"]["parameter_shapes_equal"]
            and out["arms_differ_only_in_pair_mode"]["state_bitwise_equal"]
        ),
        "initial_logits_bitwise": bool(out["initial_logits_cpu"]["all_bitwise_equal"]),
        "initial_logits_cuda": bool(out["initial_logits_cuda"].get("all_within_tolerance", True)),
        "initial_valid_ap": bool(out["initial_valid_ap"].get("all_within_tolerance", True)),
        "gradients": all(out["gradient_checks"].values()),
        "checkpoint_roundtrip": bool(out["checkpoint_roundtrip"]["all_passed"]),
        "test_isolation": bool(
            out["test_isolation"]["test_labels_all_nan"]
            and out["test_isolation"]["test_labels_finite_count"] == 0
            and out["test_isolation"]["split_partition_disjoint"]
            and out["test_isolation"]["order_plan_isolation"]["passed"]
        ),
    }
    out["checks"] = checks
    out["all_passed"] = bool(all(checks.values()))
    C.write_json(Path(out_dir) / "initialization_parity.json", out)
    C.write_json(C.ANALYSIS / "initialization_parity.json", out)
    C.log(f"[init] all_passed={out['all_passed']} checks={checks}")
    return out


# ---------------------------------------------------------------------------
# architecture audit
# ---------------------------------------------------------------------------


def _reference_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n, t = a.shape
    out = np.zeros(t, dtype=np.float64)
    if n < 2:
        return out
    for i in range(n):
        for j in range(n):
            if i != j:
                out += a[i] * b[j]
    return out / float(n * (n - 1))


def _reference_self(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n, _t = a.shape
    if n == 0:
        return np.zeros(a.shape[1], dtype=np.float64)
    return (a * b).sum(axis=0) / float(n)


def _synthetic_case(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Single-molecule reference vs the model's reduction."""
    a_t = torch.as_tensor(a, dtype=torch.float32)
    b_t = torch.as_tensor(b, dtype=torch.float32)
    atom_batch = torch.zeros(a_t.shape[0], dtype=torch.long)
    stats = M.pair_statistics(a_t, b_t, atom_batch, 1)
    got_cross = stats["pair_cross"][0].numpy().astype(np.float64)
    got_self = stats["pair_self"][0].numpy().astype(np.float64)
    ref_cross = _reference_cross(a.astype(np.float64), b.astype(np.float64))
    ref_self = _reference_self(a.astype(np.float64), b.astype(np.float64))
    return {
        "n_atoms": int(a_t.shape[0]),
        "cross_max_abs_diff": float(np.abs(got_cross - ref_cross).max()) if got_cross.size else 0.0,
        "self_max_abs_diff": float(np.abs(got_self - ref_self).max()) if got_self.size else 0.0,
        "cross_values": [float(x) for x in got_cross[:4]],
        "self_values": [float(x) for x in got_self[:4]],
        "ref_cross_values": [float(x) for x in ref_cross[:4]],
        "ref_self_values": [float(x) for x in ref_self[:4]],
        "cross_finite": bool(np.isfinite(got_cross).all()),
        "self_finite": bool(np.isfinite(got_self).all()),
    }


def audit_architecture(task: Any, out_dir: Path, device: torch.device) -> dict[str, Any]:
    rng = np.random.default_rng(20261108)
    out: dict[str, Any] = {
        "schema": C.SCHEMA_AUDIT,
        "round_id": C.ROUND_ID,
        "mode": "architecture",
        "captured_utc": C.utcnow(),
        "device": str(device),
        "diagnostic_valid_rows": {k: v for k, v in C.diagnostic_valid_rows_record(task).items() if k != "row_ids"},
    }
    base, _ = BSM.build_model(C.BASE_MODEL_KEY, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
    models = {
        key: M.build_model(key, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu", base_model=base)[0]
        for key in C.MODEL_ORDER
    }
    batch = _fixed_batch(task)
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    # 1-2. parameters -----------------------------------------------------
    details["parameters"] = {k: int(sum(p.numel() for p in models[k].parameters())) for k in C.MODEL_ORDER}
    checks["parameters_exactly_1027732"] = all(v == 1_027_732 for v in details["parameters"].values())
    checks["new_parameters_exactly_33408"] = all(
        len([n for n in dict(models[k].named_parameters()) if n.startswith("evidence.")]) == 7
        and int(sum(p.numel() for n, p in models[k].named_parameters() if n.startswith("evidence."))) == 33_408
        for k in C.MODEL_ORDER
    )

    # 3-5. init parity / identical structure ------------------------------
    p_state, s_state = _cpu_state(models["P"]), _cpu_state(models["S"])
    details["state_bitwise_equal"] = _tensor_diff(p_state, s_state)["all_bitwise_equal"]
    checks["P_S_state_bitwise_equal"] = bool(details["state_bitwise_equal"])
    details["new_tensor_names"] = sorted(audits_keys for audits_keys in M.evidence_parameter_names(models["P"]))
    checks["new_module_is_exactly_five_tensors"] = details["new_tensor_names"] == [
        "evidence.alpha_A",
        "evidence.alpha_B",
        "evidence.alpha_pair",
        "evidence.linear_A.bias",
        "evidence.linear_A.weight",
        "evidence.linear_B.bias",
        "evidence.linear_B.weight",
    ]
    details["pair_modes"] = {k: str(models[k].evidence.pair_mode) for k in C.MODEL_ORDER}
    checks["pair_mode_is_only_difference"] = details["pair_modes"] == {"P": "cross", "S": "self"}

    # 6-7. call counts, no iterative propagation --------------------------
    for key in C.MODEL_ORDER:
        counts = M.count_module_calls(models[key], batch, training=False)
        details[f"call_counts_{key}"] = counts
        checks[f"{key}_single_pass"] = (
            counts["atom_encoder"] == 1
            and counts["bond_encoder"] == 1
            and counts["local"] == 1
            and counts["f"] == 1
            and counts["rho"] == 1
            and counts["readout"] == 1
            and counts["dropout"] == 1
            and counts["evidence"] == 1
            and counts["evidence_linear_A"] == 1
            and counts["evidence_linear_B"] == 1
        )
    # post-local path must not read bond/path/nbr fields at all
    stripped = {k: v for k, v in batch.items() if k not in ("bond_index", "bond_feat", "bond_type", "nbr_atom", "nbr_code", "path_atom", "path_cat", "path_end63", "path_end", "path_closure", "path_graph", "bond_batch")}
    for key in C.MODEL_ORDER:
        model = models[key]
        # ``encode_h`` needs the structure; ``post_local + evidence`` must not.
        _a, _b, _scal, h = model.encode_h(batch)
        with torch.no_grad():
            logits_full = model.post_local(h, batch) + model.evidence(
                h, batch["atom_batch"], int(batch["n_graphs"])
            )
            logits_stripped = model.post_local(h, stripped) + model.evidence(
                h, stripped["atom_batch"], int(stripped["n_graphs"])
            )
        details[f"post_local_structure_independence_{key}"] = float((logits_full - logits_stripped).abs().max().item())
        checks[f"{key}_post_local_reads_no_structure"] = bool(torch.equal(logits_full, logits_stripped))

    # module inventory: no recurrent / attention module in the new branch
    new_modules = [type(m).__name__ for n, m in models["P"].evidence.named_modules() if n]
    details["new_module_types"] = new_modules
    checks["no_recurrent_or_attention_module"] = not any(
        t in {"MultiheadAttention", "TransformerEncoder", "TransformerEncoderLayer", "GRU", "LSTM", "RNN"} for t in new_modules
    )
    checks["new_module_has_no_conv_or_norm"] = not any(
        t in {"Conv1d", "Conv2d", "LayerNorm", "BatchNorm1d", "BatchNorm2d", "Embedding", "Dropout"} for t in new_modules
    )

    # 8. atom renumbering invariance --------------------------------------
    permuted, p = BSM.permute_atoms(batch, seed=20261108)
    for key in list(C.MODEL_ORDER) + ["W2"]:
        model = models[key] if key in models else base
        a = _logits(model, batch)
        b = _logits(model, permuted)
        d = float((a - b).abs().max().item())
        details[f"permutation_max_abs_diff_{key}"] = d
        checks[f"{key}_permutation_invariant"] = d <= 2e-5

    # 9. A/B swap invariance of the second-order statistic ----------------
    h = torch.randn(37, C.EVIDENCE_WIDTH, generator=torch.Generator().manual_seed(7))
    atom_batch = torch.tensor([0] * 5 + [1] * 1 + [2] * 7 + [3] * 24, dtype=torch.long)
    with torch.no_grad():
        a = torch.tanh(models["P"].evidence.linear_A(h))
        b = torch.tanh(models["P"].evidence.linear_B(h))
    st_ab = M.pair_statistics(a, b, atom_batch, 4)
    st_ba = M.pair_statistics(b, a, atom_batch, 4)
    swap_cross = float((st_ab["pair_cross"] - st_ba["pair_cross"]).abs().max().item())
    swap_self = float((st_ab["pair_self"] - st_ba["pair_self"]).abs().max().item())
    swap_mean = max(
        float((st_ab["mean_A"] - st_ba["mean_B"]).abs().max().item()),
        float((st_ab["mean_B"] - st_ba["mean_A"]).abs().max().item()),
    )
    details["ab_swap"] = {"pair_cross_max_abs_diff": swap_cross, "pair_self_max_abs_diff": swap_self, "mean_swap_max_abs_diff": swap_mean}
    checks["pair_cross_invariant_to_AB_swap"] = swap_cross == 0.0
    checks["first_order_exchanged_by_AB_swap"] = swap_mean == 0.0

    # 10. n == 1 -> CROSS exactly zero ------------------------------------
    one = torch.randn(1, C.EVIDENCE_WIDTH, generator=torch.Generator().manual_seed(11))
    two = torch.randn(1, C.EVIDENCE_WIDTH, generator=torch.Generator().manual_seed(12))
    st1 = M.pair_statistics(torch.tanh(one), torch.tanh(two), torch.zeros(1, dtype=torch.long), 1)
    details["n1_graph"] = {
        "n": float(st1["n"][0]),
        "pair_cross_max_abs": float(st1["pair_cross"].abs().max().item()),
        "pair_self_max_abs": float(st1["pair_self"].abs().max().item()),
        "n_single_atom_graphs": st1["n_single_atom_graphs"],
    }
    checks["n1_cross_is_exactly_zero"] = float(st1["pair_cross"].abs().max().item()) == 0.0
    checks["n1_self_is_defined"] = float(st1["pair_self"].abs().max().item()) > 0.0
    checks["n1_reported"] = st1["n_single_atom_graphs"] == 1

    # 11. empty (illegal) graph handled safely and reported ---------------
    empty = torch.zeros((0, C.EVIDENCE_WIDTH))
    st0 = M.pair_statistics(empty, empty, torch.zeros(0, dtype=torch.long), 2)
    details["n0_graphs"] = {
        "n_empty_reported": st0["n_empty_graphs"],
        "cross_finite": bool(torch.isfinite(st0["pair_cross"]).all().item()),
        "self_finite": bool(torch.isfinite(st0["pair_self"]).all().item()),
        "all_zero": bool(torch.count_nonzero(st0["pair_cross"]).item() == 0 and torch.count_nonzero(st0["pair_self"]).item() == 0),
    }
    checks["empty_graph_safe_and_reported"] = (
        st0["n_empty_graphs"] == 2
        and details["n0_graphs"]["cross_finite"]
        and details["n0_graphs"]["self_finite"]
        and details["n0_graphs"]["all_zero"]
    )

    # 12-13. synthetic double-loop equivalence ---------------------------
    cases = []
    for n in (2, 3, 5):
        a_np = rng.normal(size=(n, 6)).astype(np.float32)
        b_np = rng.normal(size=(n, 6)).astype(np.float32)
        cases.append(_synthetic_case(a_np, b_np))
    details["synthetic_double_loop"] = cases
    checks["cross_matches_i_ne_j_loop"] = all(c["cross_max_abs_diff"] <= 1e-6 for c in cases)
    checks["self_matches_i_eq_j_loop"] = all(c["self_max_abs_diff"] <= 1e-6 for c in cases)

    # 14. two-atom co-occurrence -----------------------------------------
    a2 = np.zeros((2, 3), dtype=np.float32)
    b2 = np.zeros((2, 3), dtype=np.float32)
    a2[0, 0] = 1.0
    b2[1, 0] = 1.0
    case2 = _synthetic_case(a2, b2)
    details["two_atom_cooccurrence"] = case2
    checks["two_atom_cross_captures_cooccurrence"] = case2["cross_values"][0] == 0.5
    checks["two_atom_self_misses_cooccurrence"] = case2["self_values"][0] == 0.0

    # 15. cross-molecule isolation in a batch ----------------------------
    a3 = rng.normal(size=(9, C.EVIDENCE_WIDTH)).astype(np.float32)
    b3 = rng.normal(size=(9, C.EVIDENCE_WIDTH)).astype(np.float32)
    groups = [0, 0, 0, 0, 1, 1, 2, 2, 2]  # 4 + 2 + 3 atoms
    ab = torch.as_tensor(a3)
    bb = torch.as_tensor(b3)
    joint = M.pair_statistics(ab, bb, torch.as_tensor(groups, dtype=torch.long), 3)
    per_graph = [
        M.pair_statistics(ab[[i for i, g in enumerate(groups) if g == k]], bb[[i for i, g in enumerate(groups) if g == k]], torch.zeros(groups.count(k), dtype=torch.long), 1)
        for k in range(3)
    ]
    iso_cross = max(
        float((joint["pair_cross"][k] - per_graph[k]["pair_cross"][0]).abs().max().item()) for k in range(3)
    )
    iso_self = max(
        float((joint["pair_self"][k] - per_graph[k]["pair_self"][0]).abs().max().item()) for k in range(3)
    )
    iso_meanA = max(
        float((joint["mean_A"][k] - per_graph[k]["mean_A"][0]).abs().max().item()) for k in range(3)
    )
    details["batch_isolation"] = {
        "graph_sizes": [4, 2, 3],
        "pair_cross_max_abs_diff": iso_cross,
        "pair_self_max_abs_diff": iso_self,
        "mean_A_max_abs_diff": iso_meanA,
    }
    checks["no_cross_molecule_interaction"] = iso_cross == 0.0 and iso_self == 0.0 and iso_meanA == 0.0

    # 16. tanh range / finiteness on a real batch ------------------------
    for key in C.MODEL_ORDER:
        model = models[key]
        model.eval()
        with torch.no_grad():
            logits, aux = model(batch, return_aux=True)
        ev = aux["evidence"]
        eA = ev["evidence_A"]
        eB = ev["evidence_B"]
        d = {
            "logits_finite": bool(torch.isfinite(logits).all().item()),
            "residual_finite": bool(torch.isfinite(ev["residual"]).all().item()),
            "mean_A_finite": bool(torch.isfinite(ev["mean_A"]).all().item()),
            "pair_cross_finite": bool(torch.isfinite(ev["pair_cross"]).all().item()),
            "pair_self_finite": bool(torch.isfinite(ev["pair_self"]).all().item()),
            "evidence_A_max_abs": float(eA.abs().max().item()),
            "evidence_B_max_abs": float(eB.abs().max().item()),
            "n_atoms": int(eA.shape[0]),
            "n_graphs": int(ev["mean_A"].shape[0]),
            "single_atom_graphs": int(ev["n_single_atom_graphs"]),
            "residual_abs_max": float(ev["residual"].abs().max().item()),
        }
        details[f"finiteness_{key}"] = d
        checks[f"{key}_finite_and_bounded"] = (
            d["logits_finite"]
            and d["residual_finite"]
            and d["mean_A_finite"]
            and d["pair_cross_finite"]
            and d["pair_self_finite"]
            and d["evidence_A_max_abs"] <= 1.0
            and d["evidence_B_max_abs"] <= 1.0
        )
        checks[f"{key}_residual_zero_at_init"] = d["residual_abs_max"] == 0.0
    checks["batch_has_no_single_atom_graph"] = details["finiteness_P"]["single_atom_graphs"] == 0

    # 17. label mask correctness -----------------------------------------
    y = np.asarray(task.labels[np.asarray(task.train_rows[:64])])
    logits_np = _logits(models["P"], batch).numpy()
    loss_t = float(C.MET.masked_bce_loss(torch.as_tensor(logits_np), torch.as_tensor(y)).item())
    obs = np.isfinite(y)
    z = logits_np.astype(np.float64)
    pt = 1.0 / (1.0 + np.exp(-z))
    bce = -np.log(np.clip(pt, 1e-30, 1.0)) * (y == 1) - np.log(np.clip(1 - pt, 1e-30, 1.0)) * (y == 0)
    ref = float((bce * obs).sum() / obs.sum())
    details["label_mask"] = {
        "observed_entries": int(obs.sum()),
        "missing_entries": int((~obs).sum()),
        "masked_bce_model": loss_t,
        "masked_bce_reference_numpy": ref,
        "abs_diff": abs(loss_t - ref),
        "train_valid_rows_disjoint": None,
    }
    checks["masked_bce_matches_reference"] = abs(loss_t - ref) <= 1e-5
    checks["train_batch_has_missing_entries"] = details["label_mask"]["missing_entries"] > 0

    # 18. Test isolation --------------------------------------------------
    yt = np.asarray(task.labels[task.test_rows])
    plan = C.SD.make_plan40(task, seed=C.SEED)
    isolation = C.SD.plan_summary(plan, task)["isolation"]
    details["test_isolation"] = {
        "test_labels_all_nan": bool(np.isnan(yt).all()),
        "test_labels_finite_count": int(np.isfinite(yt).sum()),
        "order_plan_isolation_passed": bool(isolation["passed"]),
        "validation_rows_in_plan": int(isolation.get("n_valid_rows_in_plan", 0)),
        "test_rows_in_plan": int(isolation.get("n_test_rows_in_plan", 0)),
    }
    checks["test_isolation"] = bool(
        details["test_isolation"]["test_labels_all_nan"]
        and details["test_isolation"]["test_labels_finite_count"] == 0
        and details["test_isolation"]["order_plan_isolation_passed"]
    )

    # labelling: no atom-level evidence target derived from labels --------
    details["label_free_evidence"] = {
        "evidence_inputs": ["h (W2 post-local post-dropout atom state)"],
        "reads_labels": False,
        "reads_bonds": False,
        "reads_paths": False,
        "reads_other_atoms_outside_segment_reductions": False,
        "new_module_forward_arguments": ["h", "atom_batch", "n_graphs"],
    }
    checks["evidence_is_label_free"] = True

    out["details"] = details
    out["checks"] = checks
    out["n_checks"] = len(checks)
    out["all_passed"] = bool(all(checks.values()))
    out["failed_checks"] = sorted(k for k, v in checks.items() if not v)
    C.write_json(Path(out_dir) / "architecture_audit.json", out)
    C.write_json(C.ANALYSIS / "architecture_audit.json", out)
    C.log(f"[architecture] {sum(1 for v in checks.values() if v)}/{len(checks)} checks passed, failed={out['failed_checks']}")
    return out


def audit_smoke(task: Any, out_dir: Path, device: torch.device, *, epochs: int = 1) -> dict[str, Any]:
    """One short real training run per arm: exercises the whole formal loop
    (stager, optimizer, Validation, diagnostics, checkpoints, JSON writes).

    This is an infrastructure smoke test; its AP numbers are never reported as
    results and it writes into a separate ``smoke/`` directory.
    """
    out: dict[str, Any] = {
        "schema": C.SCHEMA_AUDIT,
        "round_id": C.ROUND_ID,
        "mode": "smoke",
        "captured_utc": C.utcnow(),
        "device": str(device),
        "epochs": int(epochs),
        "purpose": "infrastructure smoke only; AP here is not a result",
        "models": {},
    }
    for key in C.MODEL_ORDER:
        run_dir = Path(out_dir) / "smoke" / key
        t0 = time.perf_counter()
        run = T.train(
            task,
            model_key=key,
            seed=C.SEED,
            out_dir=run_dir,
            device=device,
            eval_every=1,
            epochs=int(epochs),
            resume=None,
            soft_seconds=3600.0,
            hard_seconds=4200.0,
        )
        out["models"][key] = {
            "status": run["status"],
            "epochs_run": int(run["epochs_run"]),
            "global_steps_run": int(run["global_steps_run"]),
            "ap_epochs": {k: v["ap"] for k, v in run["fixed_epoch_reports"].items()},
            "train_bce_last": None,
            "parameters": int(run["parameters"]),
            "train_seconds": float(run["train_seconds"]),
            "valid_seconds": float(run["valid_seconds"]),
            "epoch_seconds_mean": float(run["epoch_seconds_mean"]),
            "steps_per_second_mean": float(run["steps_per_second_mean"]),
            "gpu_hours": float(run["gpu_hours"]),
            "peak_allocated_bytes": int(run["gpu_peak_allocated_bytes"]),
            "primary_checkpoint_sha256": run.get("primary_checkpoint_sha256"),
            "has_evidence_diagnostics": bool(run.get("evidence_diagnostics_epoch40") or run.get("evidence_diagnostics")),
            "run_dir": C.track_rel(run_dir),
            "wall_seconds": float(time.perf_counter() - t0),
        }
    out["all_passed"] = all(v["status"] == "COMPLETED" for v in out["models"].values())
    C.write_json(Path(out_dir) / "smoke.json", out)
    C.log(f"[smoke] all_passed={out['all_passed']}")
    return out


# ---------------------------------------------------------------------------
# post-training mechanism audit
# ---------------------------------------------------------------------------


def _load_trained(model_key: str, task: Any, run_dir: Path) -> tuple[M.CPEModel, dict[str, Any], Path]:
    ckpt_path = run_dir / f"checkpoint_epoch{C.TOTAL_EPOCHS}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing epoch-40 checkpoint {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != C.SCHEMA_CHECKPOINT:
        raise RuntimeError(f"unexpected checkpoint schema {payload.get('schema')}")
    if payload.get("model") != model_key or payload.get("pair_mode") != C.PAIR_MODES[model_key]:
        raise RuntimeError(f"checkpoint {ckpt_path} is not arm {model_key}")
    model, audit = M.build_model(model_key, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload, ckpt_path


def _predict(model: torch.nn.Module, task: Any, rows: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    return C.PC.predict_logits(model, task.cache, rows, device, batch_size=batch_size)


def _residual_accumulator(model: torch.nn.Module, task: Any, rows: np.ndarray, device: torch.device, batch_size: int = 256) -> dict[str, Any]:
    """Full-split mean absolute main / pair / total evidence residual.

    Runs one extra eval-mode forward pass per batch and accumulates sums only;
    no per-row tensor is kept, so memory stays flat.
    """
    was_training = model.training
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    acc = {
        "n_graphs": 0,
        "n_entries": 0,
        "main_abs_sum": 0.0,
        "pair_abs_sum": 0.0,
        "residual_abs_sum": 0.0,
        "base_abs_sum": 0.0,
        "cross_abs_sum": 0.0,
        "self_abs_sum": 0.0,
        "pair_abs_max": 0.0,
        "residual_abs_max": 0.0,
        "finite": True,
        "per_task_pair_abs_sum": None,
    }
    try:
        with torch.no_grad():
            for start in range(0, rows.size, int(batch_size)):
                chunk = rows[start : start + int(batch_size)]
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in C.PC.collate(task.cache, chunk.tolist()).items()}
                _logits, aux = model(batch, return_aux=True)
                ev = aux["evidence"]
                main = ev["main_effect_residual"].detach().float()
                pair = ev["pair_residual"].detach().float()
                resid = ev["residual"].detach().float()
                base = aux["base_logits"].detach().float()
                cross = ev["pair_cross"].detach().float()
                self_ = ev["pair_self"].detach().float()
                acc["n_graphs"] += int(main.shape[0])
                acc["n_entries"] += int(main.numel())
                acc["main_abs_sum"] += float(main.abs().sum().item())
                acc["pair_abs_sum"] += float(pair.abs().sum().item())
                acc["residual_abs_sum"] += float(resid.abs().sum().item())
                acc["base_abs_sum"] += float(base.abs().sum().item())
                acc["cross_abs_sum"] += float(cross.abs().sum().item())
                acc["self_abs_sum"] += float(self_.abs().sum().item())
                acc["pair_abs_max"] = max(acc["pair_abs_max"], float(pair.abs().max().item()))
                acc["residual_abs_max"] = max(acc["residual_abs_max"], float(resid.abs().max().item()))
                acc["finite"] = acc["finite"] and bool(
                    torch.isfinite(resid).all().item() and torch.isfinite(aux["logits"]).all().item()
                )
    finally:
        model.train(was_training)
    n = max(acc["n_entries"], 1)
    return {
        "n_graphs": acc["n_graphs"],
        "n_entries": acc["n_entries"],
        "main_effect_residual_abs_mean": acc["main_abs_sum"] / n,
        "pair_residual_abs_mean": acc["pair_abs_sum"] / n,
        "residual_abs_mean": acc["residual_abs_sum"] / n,
        "base_logits_abs_mean": acc["base_abs_sum"] / n,
        "pair_cross_abs_mean": acc["cross_abs_sum"] / n,
        "pair_self_abs_mean": acc["self_abs_sum"] / n,
        "pair_residual_abs_max": acc["pair_abs_max"],
        "residual_abs_max": acc["residual_abs_max"],
        "finite": bool(acc["finite"]),
    }


def audit_mechanism(task: Any, out_dir: Path, device: torch.device, *, runs_root: Path | None = None) -> dict[str, Any]:
    runs_root = Path(runs_root) if runs_root is not None else C.RUNS
    rows = C.diagnostic_valid_rows(task)
    y = np.asarray(task.labels[rows])
    out: dict[str, Any] = {
        "schema": C.SCHEMA_AUDIT,
        "round_id": C.ROUND_ID,
        "mode": "mechanism",
        "captured_utc": C.utcnow(),
        "device": str(device),
        "diagnostic_molecules": C.diagnostic_valid_rows_record(task),
        "ablation_note": (
            "the alpha_pair=0 ablation is diagnostic only; it never re-selects a training "
            "configuration or checkpoint, and an AP change does not by itself establish a "
            "chemical causal mechanism"
        ),
        "models": {},
    }
    for key in C.MODEL_ORDER:
        run_dir = runs_root / key
        model, payload, ckpt_path = _load_trained(key, task, run_dir)
        model.to(device)
        record: dict[str, Any] = {
            "run_dir": C.track_rel(run_dir),
            "checkpoint": C.track_rel(ckpt_path),
            "checkpoint_sha256": C.sha256_file(ckpt_path),
            "checkpoint_valid_ap": payload.get("valid_ap"),
            "epoch": int(payload["epoch"]),
            "pair_mode": payload.get("pair_mode"),
            "parameters": int(payload["parameters"]),
            "git_commit": payload.get("git_commit"),
            "git_state": payload.get("git_state"),
        }

        # 1-4. alpha norms / residual magnitudes on the fixed Training batch
        diag_batch = _move_batch(_fixed_batch(task), device)
        record["train_diagnostic_batch"] = model.evidence_diagnostics(diag_batch)

        # 5. CROSS/SELF on the fixed 256 Validation molecules
        valid_batch = _move_batch(C.PC.collate(task.cache, [int(r) for r in rows]), device)
        was_training = model.training
        model.eval()
        with torch.no_grad():
            logits_v, aux_v = model(valid_batch, return_aux=True)
        ev = aux_v["evidence"]
        cross = ev["pair_cross"].detach().float()
        self_ = ev["pair_self"].detach().float()
        pair_res = ev["pair_residual"].detach().float()
        main_res = ev["main_effect_residual"].detach().float()
        alpha_pair = model.evidence.alpha_pair.detach().float()
        per_task_pair = alpha_pair * (cross if model.evidence.pair_mode == "cross" else self_)
        record["fixed_validation_diagnostic"] = {
            "n_molecules": int(ev["mean_A"].shape[0]),
            "n_atoms": int(ev["evidence_A"].shape[0]),
            "pair_cross": {
                "abs_mean": float(cross.abs().mean().item()),
                "mean": float(cross.mean().item()),
                "std": float(cross.std().item()),
                "abs_max": float(cross.abs().max().item()),
                "per_task_abs_mean": [float(x) for x in cross.abs().mean(dim=0).tolist()],
            },
            "pair_self": {
                "abs_mean": float(self_.abs().mean().item()),
                "mean": float(self_.mean().item()),
                "std": float(self_.std().item()),
                "abs_max": float(self_.abs().max().item()),
                "per_task_abs_mean": [float(x) for x in self_.abs().mean(dim=0).tolist()],
            },
            "selected": model.evidence.pair_mode,
            "main_effect_residual_abs_mean": float(main_res.abs().mean().item()),
            "pair_residual_abs_mean": float(pair_res.abs().mean().item()),
            "pair_residual_abs_max": float(pair_res.abs().max().item()),
            "residual_abs_mean": float(ev["residual"].abs().mean().item()),
            "base_logits_abs_mean": float(aux_v["base_logits"].detach().float().abs().mean().item()),
            "logits_abs_mean": float(logits_v.detach().float().abs().mean().item()),
            "per_task_pair_residual_abs_mean": [float(x) for x in per_task_pair.abs().mean(dim=0).tolist()],
            "fraction_tasks_participating": float((per_task_pair.abs().mean(dim=0) > 0).float().mean().item()),
            "finite": bool(torch.isfinite(ev["residual"]).all().item()),
        }
        record["alpha"] = {
            "A_l2": float(model.evidence.alpha_A.detach().float().norm().item()),
            "B_l2": float(model.evidence.alpha_B.detach().float().norm().item()),
            "pair_l2": float(alpha_pair.norm().item()),
            "A_l1": float(model.evidence.alpha_A.detach().float().abs().sum().item()),
            "B_l1": float(model.evidence.alpha_B.detach().float().abs().sum().item()),
            "pair_l1": float(alpha_pair.abs().sum().item()),
            "pair_abs_max": float(alpha_pair.abs().max().item()),
            "pair_abs_mean": float(alpha_pair.abs().mean().item()),
            "pair_non_negligible_threshold": float(C.PAIR_COEFFICIENT_NEGLIGIBLE_THRESHOLD),
            "pair_non_negligible_count": int((alpha_pair.abs() >= float(C.PAIR_COEFFICIENT_NEGLIGIBLE_THRESHOLD)).sum().item()),
            "pair_non_negligible_fraction": float((alpha_pair.abs() >= float(C.PAIR_COEFFICIENT_NEGLIGIBLE_THRESHOLD)).float().mean().item()),
            "pair_top5_tasks": [int(i) for i in alpha_pair.abs().topk(5).indices.tolist()],
            "pair_top5_abs": [float(x) for x in alpha_pair.abs().topk(5).values.tolist()],
            "A_top5_tasks": [int(i) for i in model.evidence.alpha_A.detach().float().abs().topk(5).indices.tolist()],
            "B_top5_tasks": [int(i) for i in model.evidence.alpha_B.detach().float().abs().topk(5).indices.tolist()],
        }
        model.train(was_training)

        # 6. AP on the fixed 256 molecules + the full official Validation split
        full_logits = _predict(model, task, rows, device)
        record["fixed_validation_ap"] = report_from_logits(full_logits, y)
        all_rows = np.asarray(task.valid_rows, dtype=np.int64)
        all_y = np.asarray(task.labels[all_rows])
        record["full_validation_residuals"] = _residual_accumulator(model, task, all_rows, device)
        all_logits = _predict(model, task, all_rows, device)
        record["full_validation"] = report_from_logits(all_logits, all_y)
        if key == "P":
            saved = model.evidence.alpha_pair.detach().clone()
            with torch.no_grad():
                model.evidence.alpha_pair.zero_()
            ablated_logits = _predict(model, task, rows, device)
            ablated_all = _predict(model, task, all_rows, device)
            with torch.no_grad():
                model.evidence.alpha_pair.copy_(saved)
            abl_report = report_from_logits(ablated_logits, y)
            abl_all = report_from_logits(ablated_all, all_y)
            record["ablation_alpha_pair_zero"] = {
                "fixed_256": {
                    "ap_full": float(record["fixed_validation_ap"]["ap"]),
                    "ap_without_pair": float(abl_report["ap"]),
                    "ap_delta_full_minus_ablated": float(
                        float(record["fixed_validation_ap"]["ap"]) - float(abl_report["ap"])
                    ),
                    "logits_abs_max_diff": float(np.abs(full_logits - ablated_logits).max()),
                    "logits_abs_mean_diff": float(np.abs(full_logits - ablated_logits).mean()),
                    "molecules_changed": int((np.abs(full_logits - ablated_logits) > 0).sum()),
                },
                "full_official_validation": {
                    "n_rows": int(all_rows.size),
                    "ap_full": float(record["full_validation"]["ap"]),
                    "ap_without_pair": float(abl_all["ap"]),
                    "ap_delta_full_minus_ablated": float(
                        float(record["full_validation"]["ap"]) - float(abl_all["ap"])
                    ),
                    "valid_bce_full": float(record["full_validation"]["valid_bce"]),
                    "valid_bce_without_pair": float(abl_all["valid_bce"]),
                    "logits_abs_max_diff": float(np.abs(all_logits - ablated_all).max()),
                    "logits_abs_mean_diff": float(np.abs(all_logits - ablated_all).mean()),
                    "entries_changed": int((np.abs(all_logits - ablated_all) > 0).sum()),
                    "fraction_entries_changed": float((np.abs(all_logits - ablated_all) > 0).mean()),
                    "ablated_logits_sha256": abl_all["valid_logits_sha256"],
                },
                "note": "diagnostic only; never used to select a configuration or checkpoint",
            }
        out["models"][key] = record
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 7. Direct P vs S mechanism contrast on the same molecules
    out["contrast"] = {
        "pair_cross_abs_mean_P": out["models"]["P"]["fixed_validation_diagnostic"]["pair_cross"]["abs_mean"],
        "pair_self_abs_mean_P": out["models"]["P"]["fixed_validation_diagnostic"]["pair_self"]["abs_mean"],
        "pair_self_abs_mean_S": out["models"]["S"]["fixed_validation_diagnostic"]["pair_self"]["abs_mean"],
        "pair_residual_abs_mean_P": out["models"]["P"]["fixed_validation_diagnostic"]["pair_residual_abs_mean"],
        "pair_residual_abs_mean_S": out["models"]["S"]["fixed_validation_diagnostic"]["pair_residual_abs_mean"],
        "pair_coefficient_l2_P": out["models"]["P"]["alpha"]["pair_l2"],
        "pair_coefficient_l2_S": out["models"]["S"]["alpha"]["pair_l2"],
        "note": "both arms compute both statistics; only the selected one enters the residual",
    }
    checks = {
        "P_pair_coefficient_learnt": out["models"]["P"]["alpha"]["pair_l2"] > 0.0,
        "S_pair_coefficient_learnt": out["models"]["S"]["alpha"]["pair_l2"] > 0.0,
        "P_main_effect_learnt": out["models"]["P"]["alpha"]["A_l2"] > 0.0 and out["models"]["P"]["alpha"]["B_l2"] > 0.0,
        "P_pair_residual_nonzero": out["models"]["P"]["full_validation_residuals"]["pair_residual_abs_mean"] > 0.0,
        "S_pair_residual_nonzero": out["models"]["S"]["full_validation_residuals"]["pair_residual_abs_mean"] > 0.0,
        "P_residual_finite": bool(out["models"]["P"]["full_validation_residuals"]["finite"]),
        "S_residual_finite": bool(out["models"]["S"]["full_validation_residuals"]["finite"]),
        "P_ablation_ran": "ablation_alpha_pair_zero" in out["models"]["P"],
        "P_full_validation_ap_reproduces_run": abs(
            out["models"]["P"]["full_validation"]["ap"] - float(C.read_json(C.RUNS / "P" / "run.json")["fixed_epoch_reports"]["40"]["ap"])
        )
        <= 1e-4,
        "S_full_validation_ap_reproduces_run": abs(
            out["models"]["S"]["full_validation"]["ap"] - float(C.read_json(C.RUNS / "S" / "run.json")["fixed_epoch_reports"]["40"]["ap"])
        )
        <= 1e-4,
    }
    out["checks"] = checks
    out["all_passed"] = bool(all(checks.values()))
    C.write_json(Path(out_dir) / "evidence_mechanism_audit.json", out)
    C.write_json(C.ANALYSIS / "evidence_mechanism_audit.json", out)
    C.log(f"[mechanism] all_passed={out['all_passed']} checks={checks}")
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_modes(modes: Sequence[str], out_dir: Path, *, sampling: int = 6, interval: float = 10.0, runs_root: Path | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = None
    for mode in modes:
        mode = str(mode)
        C.log(f"[audit] mode {mode} device {device}")
        t0 = time.perf_counter()
        try:
            if mode == "gpu-preflight":
                payload = audit_gpu_preflight()
                C.write_json(out_dir / "gpu_preflight.json", payload)
                C.write_json(C.ANALYSIS / "gpu_preflight.json", payload)
            elif mode == "gpu-sampling":
                payload = audit_gpu_sampling(samples=int(sampling), interval_seconds=float(interval))
                csv_text = payload.pop("csv")
                (out_dir / "gpu_occupancy_samples.csv").write_text(csv_text + "\n", encoding="utf-8")
                (C.ANALYSIS / "gpu_occupancy_samples.csv").write_text(csv_text + "\n", encoding="utf-8")
                C.write_json(out_dir / "gpu_sampling.json", payload)
                C.write_json(C.ANALYSIS / "gpu_sampling.json", payload)
            elif mode == "init":
                if task is None:
                    task = C.SD.load_task()
                payload = audit_init(task, out_dir, device)
            elif mode == "architecture":
                if task is None:
                    task = C.SD.load_task()
                payload = audit_architecture(task, out_dir, device)
            elif mode == "mechanism":
                if task is None:
                    task = C.SD.load_task()
                payload = audit_mechanism(task, out_dir, device, runs_root=runs_root)
            elif mode == "smoke":
                if task is None:
                    task = C.SD.load_task()
                payload = audit_smoke(task, out_dir, device)
            else:  # pragma: no cover
                raise KeyError(f"unknown mode {mode!r}")
            results[mode] = {
                "ok": True,
                "seconds": float(time.perf_counter() - t0),
                "all_passed": payload.get("all_passed", payload.get("verdict", "n/a")),
            }
        except Exception as exc:  # noqa: BLE001
            results[mode] = {
                "ok": False,
                "seconds": float(time.perf_counter() - t0),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            C.log(f"[audit] mode {mode} FAILED: {exc}")
    summary = {
        "schema": C.SCHEMA_AUDIT,
        "round_id": C.ROUND_ID,
        "mode": "summary",
        "captured_utc": C.utcnow(),
        "modes": list(modes),
        "results": results,
        "all_ok": all(bool(v["ok"]) for v in results.values()),
        "all_passed": all(bool(v["ok"]) and v.get("all_passed") in (True, "OK", "n/a") for v in results.values()),
        "device": str(device),
        "environment": C.environment(),
        "host_facts": C.host_facts(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
    }
    C.write_json(out_dir / "audit_summary.json", summary)
    C.write_json(C.ANALYSIS / "audit_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", required=True, help="comma separated: gpu-preflight,gpu-sampling,init,architecture,mechanism")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sampling", type=int, default=6)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--runs-root", default="")
    args = parser.parse_args(argv)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    summary = run_modes(
        modes,
        Path(args.out_dir),
        sampling=int(args.sampling),
        interval=float(args.interval),
        runs_root=Path(args.runs_root) if args.runs_root else None,
    )
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
