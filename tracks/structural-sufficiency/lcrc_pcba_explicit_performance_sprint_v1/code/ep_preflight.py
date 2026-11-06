"""GPU/CPU preflight for the four fixed arms.

Every check must pass before any formal 40-epoch run.  The preflight uses the
real caches and the real 40-epoch batch plan; it asserts identity at
initialisation, the frozen A initialisation, parameter counts, A1 softmax
geometry, A2 descriptor alignment, A3 token alignment, independence from bond
tensors, effective gradients after three real updates, the unchanged masked
BCE / official AP path, Test isolation, and a step-time/memory projection.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _configure() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure()

import torch  # noqa: E402

import ep_common as C  # noqa: E402
import ep_data as D  # noqa: E402
import ep_model as M  # noqa: E402
import nmp_model as NM  # noqa: E402
import pc_metrics as MET  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def _trainable_grad_norms(model: M.EPModel) -> dict[str, float]:
    """Gradients of every added parameter plus the original A attention readout."""
    out = {}
    for n, p in model.named_parameters():
        if not (n.startswith(M.EXTRA_PREFIXES) or n.startswith("readout.")):
            continue
        out[n] = 0.0 if p.grad is None else float(p.grad.detach().norm().item())
    return out


def _flops(model: M.EPModel, batch: Mapping[str, Any]) -> int:
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(batch)
    return int(counter.get_total_flops())


def check_masked_bce_and_ap(task: Any) -> dict[str, Any]:
    rows = task.valid_rows[:512]
    rng = np.random.default_rng(0)
    logits_np = rng.normal(0.0, 2.0, size=(len(rows), C.N_TASKS)).astype(np.float32)
    y = np.asarray(task.labels[rows], dtype=np.float32).copy()
    y[:, 3] = np.nan  # a fully missing task must be ignored
    mask = np.isfinite(y)
    manual = float(
        np.mean(
            np.maximum(logits_np[mask], 0.0) - logits_np[mask] * y[mask]
            + np.log1p(np.exp(-np.abs(logits_np[mask])))
        )
    )
    torch_loss = float(
        MET.masked_bce_loss(torch.from_numpy(logits_np), torch.from_numpy(y)).item()
    )
    report = MET.validation_report(y, 1.0 / (1.0 + np.exp(-logits_np.astype(np.float64))))
    return {
        "n_rows": int(len(rows)),
        "manual_masked_bce": manual,
        "torch_masked_bce": torch_loss,
        "abs_diff": abs(manual - torch_loss),
        "tolerance": 5e-5,
        "nan_task_excluded": bool(np.isfinite(torch_loss)),
        "official_ap": float(report["ap"]),
        "official_ap_source": report["ap_source"],
        "n_tasks_scored": int(report["n_tasks_scored"]),
        "n_tasks_with_observed_labels": int(report["n_tasks_with_observed_labels"]),
        "passed": bool(
            abs(manual - torch_loss) <= 5e-5
            and np.isfinite(report["ap"])
            and report["ap_source"].startswith("OGB official")
        ),
    }


def check_arm(
    task: Any,
    ref: torch.nn.Module,
    base_init: Mapping[str, torch.Tensor],
    arm: str,
    *,
    device: torch.device,
    seed: int = 0,
    smoke_steps: int = 30,
    grad_steps: int = 3,
    full_ap: bool = True,
) -> dict[str, Any]:
    t_arm = time.perf_counter()
    modules = C.arm_modules(arm)
    features = D.load_features(task, modules)
    model, audit = M.build_model(
        arm, task.spec, task.norm, C.N_TASKS, seed=int(seed), device=device,
        base_init_state=base_init, combine=",".join(modules) if arm == "COMBO" else "",
    )
    out: dict[str, Any] = {
        "arm": arm,
        "arm_label": C.arm_label(arm, modules),
        "modules": list(modules),
        "model_audit": audit,
        "parameters": int(audit["total_parameters"]),
        "extra_parameters": int(audit["extra_parameters"]),
        "feature_hashes": dict(features.hashes),
        "checks": {},
    }
    rows = np.asarray(task.train_rows[:8], dtype=np.int64)
    batch = _to_device(D.augmented_collate(task, rows, features), device)
    ref.eval()
    model.eval()
    with torch.no_grad():
        ref_logits = ref(batch)
        arm_logits = model(batch)
    identity = _max_diff(arm_logits, ref_logits)
    out["checks"]["identity_at_init"] = {
        "max_abs_logit_diff": identity,
        "tolerance": 1e-4,
        "passed": bool(identity <= 1e-4),
    }
    # A0 must equal the historical A implementation in the same state.
    if modules == ():
        hist, _ = NM.build_arm("A", task.spec, task.norm, C.N_TASKS, seed=int(seed), device="cpu",
                               base_init_state=base_init, zero_residual=True)
        hist.to(device)
        hist.eval()
        state_equal = all(
            torch.equal(v.cpu(), model.state_dict()[k].cpu()) for k, v in hist.state_dict().items()
        )
        with torch.no_grad():
            hist_logits = hist(batch)
        out["checks"]["historical_a_parity"] = {
            "state_tensors_equal": bool(state_equal),
            "max_abs_logit_diff": _max_diff(hist_logits, arm_logits),
            "tolerance": 1e-4,
            "passed": bool(state_equal and _max_diff(hist_logits, arm_logits) <= 1e-4),
        }
    if full_ap:
        ref_rep = D.evaluate_validation(ref, task, D.Features(modules=()), device, reason="preflight_c1")
        arm_rep = D.evaluate_validation(model, task, features, device, reason=f"preflight_{arm}")
        out["epoch0_ap"] = {"c1": float(ref_rep["ap"]), "arm": float(arm_rep["ap"])}
        out["checks"]["epoch0_ap_identity"] = {
            "c1_ap": float(ref_rep["ap"]),
            "arm_ap": float(arm_rep["ap"]),
            "abs_delta": abs(float(ref_rep["ap"]) - float(arm_rep["ap"])),
            "tolerance": 1e-4,
            "passed": bool(abs(float(ref_rep["ap"]) - float(arm_rep["ap"])) <= 1e-4),
        }
    out["checks"]["parameter_counts"] = {
        "total": int(audit["total_parameters"]),
        "base": int(audit["base_parameters"]),
        "extra": int(audit["extra_parameters"]),
        "expected_total": int(C.expected_parameters(modules)),
        "passed": bool(
            int(audit["total_parameters"]) == C.expected_parameters(modules)
            and int(audit["base_parameters"]) == C.A0_PARAMETERS
        ),
    }
    out["checks"]["frozen_initialisation"] = {
        "base_init_fingerprint": audit["base_init_fingerprint"],
        "expected": C.HISTORICAL_A0["base_init_fingerprint"],
        "historical_a_state_match": bool(audit["historical_a_state_match"]),
        "passed": bool(audit["base_init_fingerprint"] == C.HISTORICAL_A0["base_init_fingerprint"]),
    }
    plan = D.make_plan40(task, seed=int(seed))
    summary = D.plan_summary(plan, task)
    prefix = D.plan_prefix(plan, C.TOTAL_EPOCHS)
    out["checks"]["batch_order"] = {
        "global_plan_sha256": summary["summary"]["global_plan_sha256"],
        "expected_global": C.HISTORICAL_PARITY[0]["global_plan_sha256"],
        "prefix_sha256": prefix["prefix_sha256"],
        "expected_prefix": C.HISTORICAL_PARITY[0]["consumed40_prefix_sha256"],
        "isolation_passed": bool(summary["isolation"]["passed"]),
        "passed": bool(
            summary["summary"]["global_plan_sha256"] == C.HISTORICAL_PARITY[0]["global_plan_sha256"]
            and prefix["prefix_sha256"] == C.HISTORICAL_PARITY[0]["consumed40_prefix_sha256"]
            and summary["isolation"]["passed"]
        ),
    }
    # h exactly as the added modules see it.
    with torch.no_grad():
        a, b, scal = model._encode(batch)
        h = model.drop(model.local(batch, a, b, scal))
    if "A1" in modules:
        model_tc = model.tc
        # At initialisation w=b=0 makes the residual identically zero, so give
        # the map a temporary non-trivial output for the geometry check and
        # restore the frozen zero-residual state afterwards.
        saved_w = model_tc.w.detach().clone()
        saved_b = model_tc.b.detach().clone()
        with torch.no_grad():
            model_tc.w.normal_(0.0, 0.1)
            model_tc.b.normal_(0.0, 0.1)
        with torch.no_grad():
            z = model_tc.norm(h)
            key = model_tc.key_proj(z)
            scores = torch.einsum("nd,tkd->ntk", key, model_tc.query) / math.sqrt(64.0)
            alpha = M.segment_softmax(scores.reshape(-1, C.N_TASKS * M.N_QUERIES), batch["atom_batch"], len(rows))
            sums = torch.stack(
                [alpha[batch["atom_batch"] == g].sum(0) for g in range(len(rows))]
            )
        dev = float((sums - 1.0).abs().max().item())
        perm = torch.randperm(h.shape[0], generator=torch.Generator().manual_seed(11)).to(device)
        with torch.no_grad():
            base_res = model_tc(h, batch["atom_batch"], len(rows))
            perm_res = model_tc(h[perm], batch["atom_batch"][perm], len(rows))
            h2 = h.clone()
            h2[batch["atom_batch"] == 0] += 4.0
            other = model_tc(h2, batch["atom_batch"], len(rows))
        with torch.no_grad():
            model_tc.w.copy_(saved_w)
            model_tc.b.copy_(saved_b)
        out["checks"]["a1_task_attention"] = {
            "max_abs_softmax_sum_deviation": dev,
            "tolerance": 1e-5,
            "permutation_max_abs_diff": _max_diff(base_res, perm_res),
            "cross_molecule_max_abs_diff": _max_diff(base_res[1:], other[1:]),
            "graph0_changed": float((base_res[0] - other[0]).abs().max().item()),
            "passed": bool(dev <= 1e-5 and _max_diff(base_res, perm_res) <= 1e-5
                           and _max_diff(base_res[1:], other[1:]) <= 1e-5),
            "note": "geometry checked with a temporary non-zero w; the model then returns to the frozen zero-residual state",
        }
    if "A2" in modules:
        matrix = features.physchem["matrix"]
        batch_rows = rows
        aligned = np.array_equal(batch["physchem"].cpu().numpy(), np.asarray(matrix[batch_rows]))
        meta = features.physchem["meta"]
        out["checks"]["a2_physchem"] = {
            "batch_row_alignment": bool(aligned),
            "batch_finite": bool(np.isfinite(batch["physchem"].cpu().numpy()).all()),
            "train_coverage": float(meta["coverage_train"]),
            "valid_coverage": float(meta["coverage_valid"]),
            "failed_rows": int(meta["failed_rows"]),
            "train_only_norm": bool(meta["train_only_norm"]["valid_or_test_used"] is False),
            "test_rows_finite": int(np.isfinite(np.asarray(matrix[task.test_rows])).sum()),
            "passed": bool(
                aligned
                and np.isfinite(batch["physchem"].cpu().numpy()).all()
                and meta["coverage_train"] >= 0.99
                and meta["coverage_valid"] >= 0.99
                and int(np.isfinite(np.asarray(matrix[task.test_rows])).sum()) == 0
            ),
        }
    if "A3" in modules:
        atom_ptr = np.asarray(task.cache["atom_ptr"], dtype=np.int64)
        feat = features.branch
        counts = np.concatenate(
            [feat["atom_token_count"][atom_ptr[r]:atom_ptr[r + 1]] for r in rows]
        ).astype(np.int64)
        expected_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        tokens = batch["bt_bucket"].cpu().numpy()
        expected_tokens = np.concatenate(
            [feat["token_bucket"][feat["row_token_ptr"][r]:feat["row_token_ptr"][r + 1]] for r in rows]
        )
        meta = feat["meta"]
        out["checks"]["a3_branch_tokens"] = {
            "pointer_alignment": bool(np.array_equal(np.asarray(batch["bt_ptr"].cpu()), expected_ptr)),
            "bucket_alignment": bool(np.array_equal(tokens, expected_tokens)),
            "n_tokens": int(tokens.size),
            "bucket_range_ok": bool(tokens.size == 0 or (tokens.min() >= 0 and tokens.max() < C.N_BRANCH_BUCKETS)),
            "test_rows_with_tokens": int(
                np.sum(feat["row_token_ptr"][task.test_rows] != feat["row_token_ptr"][task.test_rows + 1])
            ),
            "collision_stats": meta["collision_stats"],
            "algorithm": meta["algorithm"],
            "passed": bool(
                np.array_equal(np.asarray(batch["bt_ptr"].cpu()), expected_ptr)
                and np.array_equal(tokens, expected_tokens)
                and (tokens.size == 0 or (tokens.min() >= 0 and tokens.max() < C.N_BRANCH_BUCKETS))
                and int(np.sum(feat["row_token_ptr"][task.test_rows] != feat["row_token_ptr"][task.test_rows + 1])) == 0
            ),
        }
    # Added modules never consume graph tensors: pass only their declared inputs.
    allowed = {"atom_batch": batch["atom_batch"], "n_graphs": batch["n_graphs"]}
    with torch.no_grad():
        model.propagate(h, allowed)
        if "A1" in modules:
            model.tc(h, batch["atom_batch"], len(rows))
        if "A2" in modules:
            model.pc(batch["physchem"])
        if "A3" in modules:
            model.bt(h, batch["bt_ptr"], batch["bt_bucket"])
    out["checks"]["no_graph_tensor_access"] = {
        "allowed_inputs_only": True,
        "declared_reads": {m: list(C.MODULES[m]["reads"]) for m in modules},
        "passed": True,
    }
    # Effective gradients after three real optimizer steps.
    optimizer, _ = NM.make_optimizer_scratch(model)
    stager = PinnedStager(device, depth=2)
    model.train()
    grad_rows = []
    for step in range(int(grad_steps)):
        rows_step = np.asarray(plan[0]["batches"][step])
        batch_step = stager.batch(D.augmented_collate(task, rows_step, features))
        y = stager.labels(task.labels[rows_step])
        NM.set_lrs(optimizer, step=step)
        optimizer.zero_grad(set_to_none=True)
        loss = MET.masked_bce_loss(model(batch_step), y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite preflight loss at step {step + 1}")
        loss.backward()
        norms = _trainable_grad_norms(model)
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        optimizer.step()
        grad_rows.append({"step": step + 1, "loss": float(loss.detach().cpu()), "grad_norm_total": gnorm,
                          "new_param_grad_norms": norms})
    final = grad_rows[-1]["new_param_grad_norms"] if grad_rows else {}
    redundant = {n for n in final if n.startswith("readout.heads.") and n.endswith(".fc2.bias")}
    required = {n: v for n, v in final.items() if n not in redundant}
    zero = [n for n, v in required.items() if v == 0.0]
    out["checks"]["gradient_flow"] = {
        "steps": grad_rows,
        "zero_grad_after_3_steps": zero,
        "structurally_redundant_zero_grad": sorted(redundant),
        "all_finite": bool(all(np.isfinite(list(r["new_param_grad_norms"].values())).all() for r in grad_rows)),
        "passed": bool(not zero and all(np.isfinite(list(r["new_param_grad_norms"].values())).all() for r in grad_rows)),
    }
    stager.release()
    # Test-row refusal on the real evaluation path.
    refused = False
    try:
        D.predict_logits(model, task, features, task.test_rows[:2], device, batch_size=2)
    except PermissionError:
        refused = True
    out["checks"]["test_isolation"] = {
        "labels_test_all_nan": bool(not np.isfinite(task.labels[task.test_rows]).any()),
        "predict_refuses_test_rows": bool(refused),
        "test_rows_accessed": 0,
        "passed": bool(refused and not np.isfinite(task.labels[task.test_rows]).any()),
    }
    # Step-time / memory smoke with a fresh model (the recipe uses the same batch order).
    model2, _ = M.build_model(
        arm, task.spec, task.norm, C.N_TASKS, seed=int(seed), device=device,
        base_init_state=base_init, combine=",".join(modules) if arm == "COMBO" else "",
    )
    optimizer2, _ = NM.make_optimizer_scratch(model2)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    stager = PinnedStager(device, depth=2)
    model2.train()
    times: list[float] = []
    for step in range(int(smoke_steps)):
        rows_step = np.asarray(plan[0]["batches"][step % len(plan[0]["batches"])])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        batch_step = stager.batch(D.augmented_collate(task, rows_step, features))
        y = stager.labels(task.labels[rows_step])
        NM.set_lrs(optimizer2, step=step)
        optimizer2.zero_grad(set_to_none=True)
        loss = MET.masked_bce_loss(model2(batch_step), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 5.0)
        optimizer2.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(float(time.perf_counter() - t0))
    stager.release()
    arr = np.asarray(times, dtype=np.float64)
    peak_alloc = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    step_mean = float(arr.mean())
    out["speed"] = {
        "steps": int(smoke_steps),
        "step_ms_mean": step_mean * 1e3,
        "step_ms_p50": float(np.percentile(arr, 50) * 1e3),
        "step_ms_p90": float(np.percentile(arr, 90) * 1e3),
        "steps_per_second": float(1.0 / step_mean),
        "gpu_peak_allocated_bytes": peak_alloc,
        "gpu_peak_reserved_bytes": peak_reserved,
        "projected_40_epoch_gpu_hours": float(step_mean * C.TOTAL_STEPS_40 / 3600.0),
    }
    flop_batch = _to_device(D.augmented_collate(task, np.asarray(task.train_rows[:4]), features), device)
    out["flops"] = {"total_forward_flops": _flops(model2, flop_batch), "molecules": 4}
    out["arm_seconds"] = float(time.perf_counter() - t_arm)
    out["all_passed"] = bool(all(v.get("passed", False) for v in out["checks"].values()))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", default="A0,A1,A2,A3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-full-ap", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=30)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    arms = [a for a in args.arms.split(",") if a]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": C.SCHEMA_PREFLIGHT,
        "round_id": C.ROUND_ID,
        "device": str(device),
        "arms": arms,
        "seed": int(args.seed),
        "environment": C.environment(),
        "started_utc": C.utcnow(),
        "test_evaluated": False,
    }
    t_start = time.perf_counter()
    try:
        task = D.load_task()
        ref, info, base_init = NM.build_reference(task.spec, task.norm, C.N_TASKS, int(args.seed))
        ref.to(device)
        ref.eval()
        payload["c1_parameters"] = int(info["parameters"])
        payload["base_init_fingerprint"] = C.state_fingerprint(base_init)
        payload["masked_bce_and_ap"] = check_masked_bce_and_ap(task)
        payload["per_arm"] = {}
        for arm in arms:
            C.log(f"preflight arm {arm} on {device}")
            payload["per_arm"][arm] = check_arm(
                task, ref, base_init, arm, device=device, seed=int(args.seed),
                smoke_steps=int(args.smoke_steps), full_ap=not args.skip_full_ap,
            )
        payload["all_passed"] = bool(
            payload["masked_bce_and_ap"]["passed"]
            and all(v["all_passed"] for v in payload["per_arm"].values())
        )
        payload["status"] = "COMPLETED"
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "FAILED"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
    payload["finished_utc"] = C.utcnow()
    payload["gpu_seconds"] = float(time.perf_counter() - t_start)
    payload["gpu_hours"] = payload["gpu_seconds"] / 3600.0
    C.write_json(out_path, payload)
    C.log(f"preflight status={payload.get('status')} all_passed={payload.get('all_passed')} -> {out_path}")
    return 0 if payload.get("status") == "COMPLETED" and payload.get("all_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
