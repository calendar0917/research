"""GPU preflight for the no-MP arms: identity, access, gradients, speed, FLOPs.

Run one shard per GPU; the two JSON files are merged by ``nmp_audit.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import torch  # noqa: E402

import nmp_common as C  # noqa: E402
import nmp_data as D  # noqa: E402
import nmp_model as M  # noqa: E402
import pc_metrics as MET  # noqa: E402
import pc_model as BM  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def _new_grad_norms(model: M.NMPModel) -> dict[str, float]:
    out = {}
    for name, p in model.named_parameters():
        if name.startswith(("blocks.", "readout.")):
            out[name] = 0.0 if p.grad is None else float(p.grad.detach().norm().item())
    return out


def _flops(model: M.NMPModel, batch: Mapping[str, Any], h: torch.Tensor) -> dict[str, Any]:
    from torch.utils.flop_counter import FlopCounterMode

    model.eval()
    with torch.no_grad():
        with FlopCounterMode(display=False) as total_counter:
            model(batch)
        total = int(total_counter.get_total_flops())
        with FlopCounterMode(display=False) as new_counter:
            model.propagate(h, batch, return_attention=True)
        added = int(new_counter.get_total_flops())
    return {"total_forward_flops": total, "new_module_forward_flops": added}


def check_arm(
    task: Any,
    ref: torch.nn.Module,
    base_init: Mapping[str, torch.Tensor],
    arm: str,
    *,
    device: torch.device,
    seed: int = 0,
    full_ap: bool = True,
    smoke_steps: int = C.PREFLIGHT_SMOKE_STEPS,
    grad_steps: int = C.PREFLIGHT_GRAD_STEPS,
) -> dict[str, Any]:
    out: dict[str, Any] = {"arm": arm, "checks": {}, "seed": int(seed)}
    model, audit = M.build_arm(
        arm, task.spec, task.norm, C.N_TASKS, seed=int(seed), device=device,
        base_init_state=base_init, zero_residual=True,
    )
    out["model_audit"] = audit
    out["parameters"] = int(audit["total_parameter_count"])
    out["new_parameters"] = int(audit["new_parameter_count"])

    rows = np.arange(C.PREFLIGHT_IDENTITY_MOLECULES, dtype=np.int64)
    batch = _to_device(BM.collate(task.cache, rows.tolist()), device)

    ref.eval()
    model.eval()
    with torch.no_grad():
        ref_logits = ref(batch)
        arm_logits = model(batch)
    identity = _max_diff(arm_logits, ref_logits)
    out["checks"]["identity_eval"] = {
        "max_abs_logit_diff": identity,
        "tolerance": 1e-4,
        "passed": bool(identity <= 1e-4),
    }

    if full_ap:
        ref_rep = D.evaluate_validation(ref, task, device, reason="preflight_reference")
        arm_rep = D.evaluate_validation(model, task, device, reason="preflight_arm")
        out["reference_valid_ap"] = float(ref_rep["ap"])
        out["arm_valid_ap"] = float(arm_rep["ap"])
        out["checks"]["epoch0_ap_identity"] = {
            "reference_ap": float(ref_rep["ap"]),
            "arm_ap": float(arm_rep["ap"]),
            "abs_delta": abs(float(arm_rep["ap"]) - float(ref_rep["ap"])),
            "tolerance": 1e-4,
            "passed": bool(abs(float(arm_rep["ap"]) - float(ref_rep["ap"])) <= 1e-4),
        }

    # h after the frozen local encode, exactly as the arm sees it.
    with torch.no_grad():
        a, b, scal = model._encode(batch)
        h = model.drop(model.local(batch, a, b, scal))

    # Perturb real graph tensors: the added module must be invariant when h is fixed.
    perturbed = dict(batch)
    if batch["bond_index"].numel():
        idx = batch["bond_index"].flip(0).flip(1)
        perturbed["bond_index"] = idx
        perturbed["bond_feat"] = batch["bond_feat"].flip(0)
    else:  # pragma: no cover - PCBA batches always have bonds
        perturbed["bond_index"] = torch.tensor([[0, 0]], dtype=torch.long, device=device)
        perturbed["bond_feat"] = torch.zeros(1, task.spec.bond_feat_dim, device=device)
    with torch.no_grad():
        o1, a1, p1, r1 = model.propagate(h, batch, return_attention=True)
        o2, a2, p2, r2 = model.propagate(h, perturbed, return_attention=True)
    if arm == "A":
        bond_diff = max(
            _max_diff(o1, o2),
            _max_diff(a1, a2),
            _max_diff(p1, p2),
            _max_diff(r1, r2),
        )
        bond_ok = bool(bond_diff <= 1e-5)
    else:
        bond_diff = _max_diff(o1, o2)
        bond_ok = bool(torch.equal(o1, o2))
    out["checks"]["bond_perturbation_invariance"] = {
        "max_abs_diff": float(bond_diff),
        "bitwise_equal_for_added_state": bool(bond_ok and arm != "A"),
        "tolerance_for_A_cuda_index_add": 1e-5 if arm == "A" else 0.0,
        "note": "A's pooled sums use CUDA index_add_, whose reduction order is not bitwise stable",
        "passed": bool(bond_ok),
    }

    # Per-molecule isolated changes: graph 0 atoms only.
    atom_batch = batch["atom_batch"]
    h2 = h.clone()
    h2[atom_batch == 0] += 3.0
    with torch.no_grad():
        o1, a1, p1, r1 = model.propagate(h, batch, return_attention=True)
        o2, a2, p2, r2 = model.propagate(h2, batch, return_attention=True)
    if arm == "A":
        graph_diff = max(
            _max_diff(r1[1:], r2[1:]),
            _max_diff(p1[1:], p2[1:]),
            _max_diff(a1[atom_batch != 0], a2[atom_batch != 0]),
        )
        graph_ok = bool(graph_diff <= 1e-5)
        sums = []
        for g in range(int(batch["n_graphs"])):
            sums.append(a1[atom_batch == g].sum(dim=0).detach().cpu().tolist())
        out["checks"]["attention_softmax"] = {
            "per_graph_head_sums": sums,
            "max_abs_deviation": float(
                max(abs(float(x) - 1.0) for row in sums for x in row)
            ),
            "tolerance": 1e-5,
            "passed": bool(max(abs(float(x) - 1.0) for row in sums for x in row) <= 1e-5),
        }
    else:
        mask = atom_batch != 0
        graph_diff = _max_diff(o1[mask], o2[mask])
        graph_ok = bool(torch.equal(o1[mask], o2[mask]))
    out["checks"]["graph_isolation"] = {
        "max_abs_diff": float(graph_diff),
        "tolerance": 1e-5,
        "passed": bool(graph_ok),
        "cross_graph_changed": bool(not graph_ok),
    }

    # Atom-order permutation of the new module.
    one = _to_device(BM.collate(task.cache, [0]), device)
    with torch.no_grad():
        aa, bb, ss = model._encode(one)
        h1 = model.drop(model.local(one, aa, bb, ss))
    n = h1.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(2026)).to(device)
    b1 = {"atom_batch": torch.zeros(n, dtype=torch.long, device=device), "n_graphs": 1}
    with torch.no_grad():
        o1, a1, p1, r1 = model.propagate(h1, b1, return_attention=True)
        o2, a2, p2, r2 = model.propagate(h1[perm], b1, return_attention=True)
    if arm == "A":
        perm_diff = max(_max_diff(o2, o1[perm]), _max_diff(p2, p1), _max_diff(r2, r1), _max_diff(a2, a1[perm]))
        perm_tol = 1e-5
    else:
        perm_diff = _max_diff(o2, o1[perm])
        perm_tol = 1e-5
    out["checks"]["permutation_invariance"] = {
        "max_abs_diff": float(perm_diff),
        "tolerance": float(perm_tol),
        "passed": bool(perm_diff <= perm_tol),
    }

    # Effective gradients after a few real updates.
    optimizer, _ = M.make_optimizer_scratch(model)
    plan = D.make_plan40(task, seed=0)
    stager = PinnedStager(device, depth=2)
    model.train()
    grad_rows = []
    for step in range(int(grad_steps)):
        rows_step = plan[0]["batches"][step]
        batch_step = stager.batch(BM.collate(task.cache, rows_step.tolist()))
        y = stager.labels(task.labels[rows_step])
        M.set_lrs(optimizer, step=step)
        optimizer.zero_grad(set_to_none=True)
        loss = MET.masked_bce_loss(model(batch_step), y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite preflight loss at step {step + 1}")
        loss.backward()
        norms = _new_grad_norms(model)
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        optimizer.step()
        grad_rows.append(
            {
                "step": step + 1,
                "loss": float(loss.detach().cpu().item()),
                "grad_norm_total": gnorm,
                "new_param_grad_norms": norms,
                "all_finite": bool(torch.isfinite(loss).item()),
            }
        )
    final_norms = grad_rows[-1]["new_param_grad_norms"]
    redundant = {
        name for name in final_norms
        if name.startswith("readout.heads.") and name.endswith(".fc2.bias")
    }
    required = {name: value for name, value in final_norms.items() if name not in redundant}
    grad_ok = bool(
        all(np.isfinite(v) for v in final_norms.values())
        and all(v > 0.0 for v in required.values())
    )
    out["checks"]["gradient_flow"] = {
        "steps": grad_rows,
        "final_new_param_grad_norms": final_norms,
        "all_required_new_params_nonzero": bool(all(v > 0.0 for v in required.values())),
        "structurally_redundant_zero_grad_params": {
            name: "per-graph softmax is invariant to a constant shift; the head output bias adds a graph-constant score"
            for name in sorted(redundant)
        },
        "passed": grad_ok,
    }

    # Speed and memory (fresh model, same recipe, 30 real batches).
    model, audit_speed = M.build_arm(
        arm, task.spec, task.norm, C.N_TASKS, seed=int(seed), device=device,
        base_init_state=base_init, zero_residual=True,
    )
    optimizer, _ = M.make_optimizer_scratch(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    model.train()
    times: list[float] = []
    step_global = 0
    for step in range(int(smoke_steps)):
        rows_step = plan[0]["batches"][step % len(plan[0]["batches"])]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        batch_step = stager.batch(BM.collate(task.cache, rows_step.tolist()))
        y = stager.labels(task.labels[rows_step])
        M.set_lrs(optimizer, step=step_global)
        optimizer.zero_grad(set_to_none=True)
        loss = MET.masked_bce_loss(model(batch_step), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(float(time.perf_counter() - t0))
        step_global += 1
    stager.release()
    arr = np.asarray(times, dtype=np.float64)
    if device.type == "cuda":
        peak_alloc = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_alloc = peak_reserved = 0
    out["speed"] = {
        "steps": int(smoke_steps),
        "step_ms_mean": float(arr.mean() * 1e3),
        "step_ms_p50": float(np.percentile(arr, 50) * 1e3),
        "step_ms_p90": float(np.percentile(arr, 90) * 1e3),
        "steps_per_second": float(1.0 / arr.mean()),
        "gpu_peak_allocated_bytes": peak_alloc,
        "gpu_peak_reserved_bytes": peak_reserved,
        "projected_40_epoch_gpu_hours": float(arr.mean() * C.TOTAL_STEPS_40 / 3600.0),
    }

    # FLOPs on a fixed four-molecule batch.
    flop_batch = _to_device(BM.collate(task.cache, [0, 1, 2, 3]), device)
    with torch.no_grad():
        aa, bb, ss = model._encode(flop_batch)
        h_flop = model.drop(model.local(flop_batch, aa, bb, ss))
    out["flops"] = _flops(model, flop_batch, h_flop)
    out["flops"]["molecules"] = 4
    out["flops"]["atoms"] = int(flop_batch["atom_feat"].shape[0])

    out["checks"]["parameters"] = {
        "total": int(audit["total_parameter_count"]),
        "new": int(audit["new_parameter_count"]),
        "expected_total": int(C.EXPECTED_PARAMETERS[arm]),
        "expected_new": int(C.ARMS[arm]["new_parameters"]),
        "passed": bool(
            int(audit["total_parameter_count"]) == int(C.EXPECTED_PARAMETERS[arm])
            and int(audit["new_parameter_count"]) == int(C.ARMS[arm]["new_parameters"])
        ),
    }
    out["checks"]["base_parity"] = {
        "base_init_fingerprint": audit["base_init_fingerprint"],
        "expected_seed0": C.HISTORICAL_PARITY[0]["base_init_fingerprint"],
        "passed": bool(audit["base_init_fingerprint"] == C.HISTORICAL_PARITY[0]["base_init_fingerprint"]),
    }
    out["all_passed"] = bool(all(v.get("passed", False) for v in out["checks"].values()))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-full-ap", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=C.PREFLIGHT_SMOKE_STEPS)
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    arms = [a for a in (args.arms.split(",") if args.arms else C.ARM_ORDER) if a]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "lcrc-pcba-no-mp-preflight-v1",
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
        ref, info, base_init = M.build_reference(task.spec, task.norm, C.N_TASKS, int(args.seed))
        ref.to(device)
        ref.eval()
        payload["c1_parameters"] = int(info["parameters"])
        payload["base_init_fingerprint"] = C.state_fingerprint(base_init)
        payload["per_arm"] = {}
        for arm in arms:
            C.log(f"preflight arm {arm} on {device}")
            payload["per_arm"][arm] = check_arm(
                task, ref, base_init, arm, device=device, seed=int(args.seed),
                full_ap=not args.skip_full_ap, smoke_steps=int(args.smoke_steps),
            )
        payload["all_passed"] = bool(all(v["all_passed"] for v in payload["per_arm"].values()))
        payload["status"] = "COMPLETED"
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "FAILED"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
    payload["finished_utc"] = C.utcnow()
    payload["gpu_seconds"] = float(time.perf_counter() - t_start)
    payload["gpu_hours"] = payload["gpu_seconds"] / 3600.0
    C.write_json(out_path, payload)
    C.log(f"preflight status={payload.get('status')} -> {out_path}")
    return 0 if payload.get("status") == "COMPLETED" and payload.get("all_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
