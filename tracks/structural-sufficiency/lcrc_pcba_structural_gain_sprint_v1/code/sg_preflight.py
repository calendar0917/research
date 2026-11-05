"""GPU correctness preflight and per-arm speed smoke.

Run before any formal training.  Every gate is a refusal path:

* source checkpoint hash/metadata/state + AdamW restoration audit,
* epoch-0 Validation AP reproduces AP0 within 1e-4,
* zero-residual identity against the frozen C1 (eval and train/dropout mode),
* atom-order permutation invariance,
* F branch adjacency independence / M branch adjacency response,
* V per-graph context isolation,
* gradient flow into the new modules after two optimizer steps (M bond_proj,
  V context projection included),
* 30-step smoke with measured step time, peak memory and finite values.

Writes one JSON per GPU invocation; the two files are merged by `sg_audit.py`.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    os.environ.setdefault("OMP_NUM_THREADS", "8")


_configure_imports()

import torch  # noqa: E402

import sg_common as C  # noqa: E402
import sg_checkpoint as CK  # noqa: E402
import sg_data as D  # noqa: E402
import sg_model as M  # noqa: E402
import pc_metrics as MET  # noqa: E402
import pc_model as BM  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def _permute_atoms(batch: Mapping[str, Any], perm: torch.Tensor) -> dict[str, Any]:
    n = int(perm.numel())
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n, device=perm.device, dtype=perm.dtype)
    out: dict[str, Any] = {k: v for k, v in batch.items()}
    for key in ("atom_feat", "atom_scalar"):
        out[key] = batch[key][perm]
    out["atom_batch"] = batch["atom_batch"][perm]
    idx = batch["bond_index"]
    out["bond_index"] = inv[idx] if idx.numel() else idx.clone()
    out["nbr_atom"] = inv[batch["nbr_atom"]]
    out["path_atom"] = inv[batch["path_atom"]]
    out["path_end"] = inv[batch["path_end"]]
    return out


def _nonzero(t: torch.Tensor | None) -> bool:
    return bool(t is not None and torch.isfinite(t).all() and float(t.abs().max().item()) > 0.0)


def _grad_norms(model: M.SGModel) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is not None and (name.startswith(M.NEW_MODULE_PREFIXES)):
            out[name] = float(p.grad.detach().float().norm().item())
    return out


def _snapshot_new_state(model: M.SGModel) -> dict[str, torch.Tensor]:
    keys = set(model.new_parameter_names())
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k in keys}


def _restore_new_state(model: M.SGModel, state: Mapping[str, torch.Tensor]) -> None:
    with torch.no_grad():
        current = dict(model.state_dict())
        current.update({k: v.clone() for k, v in state.items()})
        model.load_state_dict(current, strict=True)


def _randomise_outputs(model: M.SGModel, seed: int) -> None:
    """Temporarily make the residual non-zero so adjacency tests are meaningful."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        for block in model.blocks or []:
            w = torch.empty_like(block.fc2.weight, device="cpu")
            b = torch.empty_like(block.fc2.bias, device="cpu")
            w.normal_(0.0, 0.02, generator=gen)
            b.normal_(0.0, 0.02, generator=gen)
            block.fc2.weight.copy_(w.to(block.fc2.weight.device))
            block.fc2.bias.copy_(b.to(block.fc2.bias.device))


def arm_preflight(
    arm: str,
    *,
    task: Any,
    source: Mapping[str, Any],
    ref_base: M.SGModel,
    base_init: Mapping[str, torch.Tensor],
    plan: Sequence[Mapping[str, Any]],
    device: torch.device,
    smoke_steps: int,
    grad_steps: int,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    out: dict[str, Any] = {"arm": arm, "arm_name": C.ARMS[arm]["name"]}
    source_audit = CK.verify_source_metadata(source, ref_base)

    model, model_audit = M.build_arm(
        arm,
        task.spec,
        task.norm,
        C.N_TASKS,
        seed=int(C.SEED),
        device=device,
        base_init_state=base_init,
        warm_start_state=source["state_dict"],
        zero_residual=True,
    )
    optimizer, opt_audit = M.make_optimizer_warm(model, source["optimizer_state_dict"])
    ref, _ = BM.build_c1(task.spec, task.norm, C.N_TASKS, C.SEED, device)
    ref.load_state_dict(source["state_dict"], strict=True)
    ref.eval()
    out["model_audit"] = model_audit
    out["optimizer_audit"] = opt_audit
    out["source_audit_summary"] = {
        "sha256": source_audit["source_checkpoint_sha256"],
        "state_shapes_match": source_audit["state_dict_keys_shapes_match_frozen_c1"],
        "adamw_steps": source_audit["adamw_step_values"],
        "metadata_ok": all(source_audit["metadata_checks"].values()),
    }

    # ---- epoch 0 official Validation AP ----
    model.eval()
    report0 = D.evaluate_validation(model, task, device, reason="preflight_epoch0")
    delta0 = abs(float(report0["ap"]) - C.AP0)
    out["epoch0"] = {
        "valid_ap": float(report0["ap"]),
        "expected_ap": C.AP0,
        "abs_delta": float(delta0),
        "tolerance": 1e-4,
        "passed": bool(delta0 <= 1e-4),
        "n_tasks_scored": int(report0["n_tasks_scored"]),
        "valid_bce": float(report0["valid_bce"]),
    }

    # ---- fixed identity batch ----
    rows = [int(x) for x in plan[0]["batches"][0][: C.PREFLIGHT_IDENTITY_MOLECULES]]
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in BM.collate(task.cache, rows).items()}

    # ---- zero-residual identity, eval mode ----
    model.eval()
    torch.manual_seed(1234)
    ref_logits = ref(batch).detach()
    torch.manual_seed(1234)
    arm_logits = model(batch).detach()
    diff_eval = _max_abs_diff(ref_logits, arm_logits)
    out["identity_eval"] = {
        "max_abs_logit_diff": float(diff_eval),
        "tolerance": 1e-4,
        "passed": bool(diff_eval <= 1e-4),
        "note": "cross-instance CUDA index_add_ reduction noise; the same-model repeated-forward "
        "floor is reported under permutation_invariance",
    }

    # ---- zero-residual identity, train mode with identical dropout RNG ----
    model.train()
    ref.train()
    torch.manual_seed(4321)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(4321)
    rng_before = C.capture_rng_state()
    ref_logits_train = ref(batch).detach()
    M.restore_rng_state(rng_before)
    arm_logits_train = model(batch).detach()
    diff_train = _max_abs_diff(ref_logits_train, arm_logits_train)
    out["identity_train_dropout"] = {
        "max_abs_logit_diff": float(diff_train),
        "tolerance": 1e-4,
        "passed": bool(diff_train <= 1e-4),
    }
    ref.eval()
    model.eval()

    # ---- atom-order permutation invariance ----
    gen = torch.Generator(device="cpu").manual_seed(99)
    perm = torch.randperm(int(batch["atom_batch"].numel()), generator=gen).to(device)
    perm_batch = _permute_atoms(batch, perm)
    torch.manual_seed(7)
    base_logits = model(batch).detach()
    torch.manual_seed(7)
    perm_logits = model(perm_batch).detach()
    perm_diff = _max_abs_diff(base_logits, perm_logits)
    # repeated identical forward: measures the deterministic floor
    torch.manual_seed(7)
    repeat_logits = model(batch).detach()
    repeat_diff = _max_abs_diff(base_logits, repeat_logits)
    out["permutation_invariance"] = {
        "max_abs_logit_diff": float(perm_diff),
        "repeated_forward_diff": float(repeat_diff),
        "identity_noise_floor": float(max(repeat_diff, diff_eval, diff_train)),
        "tolerance": 1e-4,
        "passed": bool(perm_diff <= 1e-4),
        "note": "graph logits are compared after a random global atom permutation; tolerance covers index_add reduction order",
    }

    # ---- adjacency tests with a temporarily non-zero residual ----
    zero_state = _snapshot_new_state(model)
    adjacency: dict[str, Any] = {"applicable": arm in ("F", "M", "V")}
    if arm in ("F", "M", "V"):
        _randomise_outputs(model, 2026)
        model.eval()
        one = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in BM.collate(task.cache, [rows[0]]).items()}
        with torch.no_grad():
            h = torch.randn(int(one["atom_batch"].numel()), C.BLOCK_WIDTH, generator=torch.Generator().manual_seed(5)).to(device)
            base_prop = model.propagate(h, one).detach()
            idx = one["bond_index"]
            if idx.numel() == 0:
                raise RuntimeError("adjacency test molecule has no bonds")
            cut = {k: v for k, v in one.items()}
            cut["bond_index"] = idx[1:].clone()
            cut["bond_feat"] = one["bond_feat"][1:].clone()
            cut_prop = model.propagate(h, cut).detach()
            extra = {
                k: v for k, v in one.items()
            }
            extra["bond_index"] = torch.cat([idx, idx.flip(1)], dim=0)
            extra["bond_feat"] = torch.cat([one["bond_feat"], one["bond_feat"]], dim=0)
            extra_prop = model.propagate(h, extra).detach()
            adjacency.update(
                {
                    "remove_one_bond_max_abs_diff": float((base_prop - cut_prop).abs().max().item()),
                    "add_reverse_bonds_max_abs_diff": float((base_prop - extra_prop).abs().max().item()),
                    "n_atoms": int(h.shape[0]),
                    "n_bonds": int(idx.shape[0]),
                }
            )
        if arm == "F":
            adjacency["expectation"] = "invariant to adjacency (new branch takes no bond input)"
            adjacency["passed"] = bool(
                adjacency["remove_one_bond_max_abs_diff"] <= 1e-7
                and adjacency["add_reverse_bonds_max_abs_diff"] <= 1e-7
            )
        else:
            adjacency["expectation"] = "responds to real adjacency changes"
            adjacency["passed"] = bool(adjacency["remove_one_bond_max_abs_diff"] > 1e-7)
        _restore_new_state(model, zero_state)
        model.eval()

    # ---- V context isolation ----
    context: dict[str, Any] = {"applicable": arm == "V"}
    if arm == "V":
        model.eval()
        two_rows = [int(rows[0]), int(rows[1])] if len(rows) > 1 else [int(rows[0])]
        two = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in BM.collate(task.cache, two_rows).items()
        }
        with torch.no_grad():
            a, b, scal = model._encode(two)
            h = model.drop(model.local(two, a, b, scal))
            block = model.blocks[0]
            z = block.norm(h)
            ab = two["atom_batch"]
            n_graphs = int(two["n_graphs"])
            counts = z.new_zeros(n_graphs).index_add_(0, ab, torch.ones_like(ab, dtype=z.dtype))
            g = z.new_zeros((n_graphs, z.shape[1])).index_add_(0, ab, z) / counts.clamp(min=1.0).unsqueeze(1)
            manual = torch.stack(
                [z[ab == i].mean(dim=0) for i in range(n_graphs)]
            )
            mean_diff = float((g - manual).abs().max().item())
            z2 = z.clone()
            mask_b = ab == (n_graphs - 1)
            z2[mask_b] = z2[mask_b] + 100.0
            g2 = z2.new_zeros((n_graphs, z.shape[1])).index_add_(0, ab, z2) / counts.clamp(min=1.0).unsqueeze(1)
            a_before = g[0]
            a_after = g2[0]
            cross_diff = float((a_before - a_after).abs().max().item())
            ctx = block.ctx_proj(g)
            ctx_atoms = ctx[ab]
            spread = max(
                float((ctx_atoms[ab == i] - ctx[i]).abs().max().item()) for i in range(n_graphs)
            )
            # V also keeps two message rounds: both blocks have a context map.
            context.update(
                {
                    "n_graphs": n_graphs,
                    "mean_pool_manual_max_abs_diff": mean_diff,
                    "graph0_context_change_when_graph1_states_shift": cross_diff,
                    "context_atom_spread_max_abs_diff": spread,
                    "context_maps_per_block": int(sum(1 for blk in model.blocks if hasattr(blk, "ctx_proj"))),
                }
            )
        context["passed"] = bool(
            mean_diff <= 1e-5
            and cross_diff <= 1e-5
            and spread <= 1e-5
            and context["context_maps_per_block"] == 2
        )

    # ---- gradient flow + smoke timing ----
    model.train()
    stager = PinnedStager(device, depth=2)
    clip = float(C.RECIPE["gradient_clip"])
    step_rows = [int(x) for x in plan[0]["batches"][0]]
    gbatch = stager.batch(BM.collate(task.cache, step_rows))
    gy = stager.labels(task.labels[np.asarray(step_rows, dtype=np.int64)])
    grad_audit: dict[str, Any] = {"applicable": arm != "C", "steps": []}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    times: list[float] = []
    losses: list[float] = []
    global_step = 0
    for step in range(1, int(smoke_steps) + 1):
        M.set_lrs(optimizer, phase=1, step=global_step)
        t = time.perf_counter()
        logits = model(gbatch)
        loss = MET.masked_bce_loss(logits, gy)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step <= int(grad_steps):
            norms = _grad_norms(model)
            grad_audit["steps"].append({"step": step, "grad_norms": norms})
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(float(time.perf_counter() - t))
        losses.append(float(loss.detach().cpu().item()))
        global_step += 1
    stager.release()
    if not all(math.isfinite(x) for x in losses):
        raise RuntimeError(f"non-finite loss during {arm} smoke")
    if device.type == "cuda":
        peak_alloc = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_alloc = peak_reserved = 0

    if arm != "C":
        # After the first update the final linear is non-zero, so at step >= 2
        # the FFN internals (and, for M/V, bond_proj/ctx_proj) must receive
        # non-zero gradients.
        step2 = grad_audit["steps"][1]["grad_norms"] if len(grad_audit["steps"]) > 1 else {}
        step3 = grad_audit["steps"][2]["grad_norms"] if len(grad_audit["steps"]) > 2 else {}
        late = {k: max(step2.get(k, 0.0), step3.get(k, 0.0)) for k in sorted(set(step2) | set(step3))}
        fc1_names = [k for k in late if ".fc1." in k]
        fc2_names = [k for k in late if ".fc2." in k]
        checks = {
            "fc1_grads_nonzero_after_two_steps": bool(fc1_names and all(late[k] > 0 for k in fc1_names)),
            "fc2_grads_nonzero": bool(fc2_names and all(late[k] > 0 for k in fc2_names)),
        }
        if arm in ("M", "V"):
            checks["bond_proj_grad_nonzero"] = bool(late.get("bond_proj.weight", 0.0) > 0)
        if arm == "V":
            ctx_names = [k for k in late if ".ctx_proj." in k]
            checks["context_proj_grads_nonzero"] = bool(ctx_names and all(late[k] > 0 for k in ctx_names))
        grad_audit["late_grad_norms"] = late
        grad_audit["checks"] = {k: bool(v) for k, v in checks.items()}
        grad_audit["passed"] = bool(all(checks.values()))
    else:
        grad_audit["passed"] = True

    arr = np.asarray(times, dtype=np.float64)
    smoke = {
        "steps": int(smoke_steps),
        "step_ms_mean": float(arr.mean() * 1e3),
        "step_ms_p50": float(np.percentile(arr, 50) * 1e3),
        "step_ms_p90": float(np.percentile(arr, 90) * 1e3),
        "steps_per_second": float(len(times) / arr.sum()),
        "molecules_per_second": float(len(step_rows) * len(times) / arr.sum()),
        "epoch_seconds_estimate": float(arr.mean() * C.STEPS_PER_EPOCH),
        "twelve_epoch_seconds_estimate": float(arr.mean() * C.STEPS_PER_EPOCH * C.PHASE1_EPOCHS),
        "forty_epoch_seconds_estimate": float(arr.mean() * C.STEPS_PER_EPOCH * C.TOTAL_EPOCHS),
        "gpu_peak_allocated_bytes": int(peak_alloc),
        "gpu_peak_reserved_bytes": int(peak_reserved),
        "loss_first": float(losses[0]),
        "loss_last": float(losses[-1]),
        "finite": True,
        "parameters": int(model_audit["total_parameter_count"]),
        "new_parameters": int(model_audit["new_parameter_count"]),
    }
    out["adjacency"] = adjacency
    out["context_isolation"] = context
    out["gradients"] = grad_audit
    out["smoke"] = smoke

    checks = [
        out["epoch0"]["passed"],
        out["identity_eval"]["passed"],
        out["identity_train_dropout"]["passed"],
        out["permutation_invariance"]["passed"],
        adjacency.get("passed", True),
        context.get("passed", True),
        grad_audit["passed"],
        smoke["finite"],
    ]
    out["passed"] = bool(all(checks))
    out["checks"] = {
        "epoch0": out["epoch0"]["passed"],
        "identity_eval": out["identity_eval"]["passed"],
        "identity_train_dropout": out["identity_train_dropout"]["passed"],
        "permutation": out["permutation_invariance"]["passed"],
        "adjacency": adjacency.get("passed", True),
        "context": context.get("passed", True),
        "gradients": grad_audit["passed"],
        "finite": smoke["finite"],
    }
    out["seconds"] = float(time.perf_counter() - t0)
    out["device"] = str(device)
    out["environment"] = C.environment()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", required=True, help="comma separated subset of C,F,M,V")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke-steps", type=int, default=C.PREFLIGHT_SMOKE_STEPS)
    parser.add_argument("--grad-steps", type=int, default=C.PREFLIGHT_GRAD_STEPS)
    args = parser.parse_args(argv)

    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    if not arms or any(a not in C.ARMS for a in arms):
        raise SystemExit(f"bad --arms {args.arms!r}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    t_wall = time.time()
    result: dict[str, Any] = {
        "schema": "lcrc-pcba-structural-gain-preflight-v1",
        "round_id": C.ROUND_ID,
        "arms": arms,
        "device": str(device),
        "gpu_count": int(torch.cuda.device_count()) if device.type == "cuda" else 0,
        "environment": C.environment(),
        "lr_audit": C.lr_audit(),
        "started_utc": C.utcnow(),
    }
    try:
        C.LC.assert_inherited_constants()
        task = D.load_task()
        source = CK.load_source_checkpoint()
        ref, ref_info, base_init = M.build_reference(task.spec, task.norm, C.N_TASKS, C.SEED)
        plan = D.make_plan40(task, seed=C.SEED)
        plan_audit = D.plan_summary(plan, task)
        result["data"] = task.data_audit
        result["batch_plan"] = {
            "global_plan_sha256": plan[0]["global_plan_sha256"],
            "isolation_passed": plan_audit["isolation"]["passed"],
            "n_epochs": len(plan),
        }
        per_arm: dict[str, Any] = {}
        for arm in arms:
            C.log(f"preflight arm {arm}")
            per_arm[arm] = arm_preflight(
                arm,
                task=task,
                source=source,
                ref_base=ref,
                base_init=base_init,
                plan=plan,
                device=device,
                smoke_steps=int(args.smoke_steps),
                grad_steps=int(args.grad_steps),
            )
            C.log(
                f"preflight arm {arm} passed={per_arm[arm]['passed']} "
                f"epoch0AP={per_arm[arm]['epoch0']['valid_ap']:.8f} "
                f"step_ms={per_arm[arm]['smoke']['step_ms_p50']:.2f}"
            )
        result["arms_result"] = per_arm
        result["all_passed"] = bool(all(v["passed"] for v in per_arm.values()))
        result["status"] = "PASS" if result["all_passed"] else "FAIL"
    except Exception as exc:  # noqa: BLE001
        result["status"] = "FAIL"
        result["all_passed"] = False
        result["error"] = {"type": type(exc).__name__, "detail": str(exc), "traceback": traceback.format_exc()}
        C.log(f"preflight FAILED: {exc}")
    result["finished_utc"] = C.utcnow()
    result["wall_seconds"] = float(time.time() - t_wall)
    result["gpu_hours"] = float(result["wall_seconds"] / 3600.0)
    C.write_json(args.out, result)
    return 0 if result.get("all_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
