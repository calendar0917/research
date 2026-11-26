"""FSR v0 light mechanism diagnostics (no SREN / explanation pipeline).

Recorded for both blocks of both arms, on the best checkpoint:

* ``D = A.weight @ B.weight`` and ``R = ||D||_F / ||W||_F`` with ``W = fc1.weight``;
* RMS of the original ``fc1(z + m)`` term, of the new low-rank correction, of
  ``z`` and ``m`` on a real Validation batch;
* the gradient norms of ``A`` and ``B`` from one real-batch backward pass;
* a frozen intervention that zeroes the whole new branch (``A = 0``, so
  ``D = 0``) and re-scores Validation;
* for at most three real Validation molecules the centre/neighbour
  pre-nonlinearity decomposition and its reconstruction error.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

import frs_common as FC  # sets sys.path to the inherited code
import ds_common as C
import ds_model as DSM
import pc_metrics as MET


def _rms(x: torch.Tensor) -> float:
    return float(x.detach().float().pow(2).mean().sqrt().item()) if x.numel() else 0.0


def _d_stat(branch: Any, fc1: torch.nn.Linear) -> dict[str, Any]:
    D = branch.effective_weight().detach()
    W = fc1.weight.detach()
    nw = float(W.norm().item())
    nd = float(D.norm().item())
    return {
        "D_shape": list(D.shape),
        "D_frobenius": nd,
        "W_frobenius": nw,
        "R_ratio": (nd / nw) if nw else None,
        "D_abs_max": float(D.abs().max().item()),
        "W_abs_max": float(W.abs().max().item()),
        "rank_bound": int(FC.RANK),
        "D_cosine_to_W": float(
            torch.nn.functional.cosine_similarity(D.flatten(), W.flatten(), dim=0).item()
        ),
    }


def _block_scales(model: Any, batch: Mapping[str, Any]) -> dict[str, Any]:
    """RMS of z, m, the fc1 pre-activation and the low-rank correction."""
    model.eval()
    with torch.inference_mode():
        _a, _b, _scal, h0 = model.encode_h0(batch)
        src, dst, edge_msg = DSM.directed_bonds(model, batch)
        h1, m1 = model.stage1(h0, src, dst, edge_msg)
        blk2 = model.blocks[1]
        z_target = blk2.norm(h1)
        z_source = blk2.norm(h0)
        m2 = DSM._agg_source(z_source, src, dst, edge_msg)
        z1 = model.blocks[0].norm(h0)
        out: dict[str, Any] = {"n_atoms": int(h0.shape[0]), "n_directed_edges": int(src.numel())}
        for i, (z, m) in enumerate(((z1, m1), (z_target, m2))):
            primary = model.blocks[i].fc1(z + m)
            correction = model.lowrank[i](model.branch_input(z, m))
            rp, rc = _rms(primary), _rms(correction)
            out[str(i)] = {
                "rms_z": _rms(z),
                "rms_m": _rms(m),
                "rms_z_plus_m": _rms(z + m),
                "rms_fc1_z_plus_m": rp,
                "rms_lowrank_correction": rc,
                "correction_to_fc1_ratio": (rc / rp) if rp else None,
                "rms_branch_input": _rms(model.branch_input(z, m)),
                "lowrank_weight_norm": float(model.lowrank[i].effective_weight().norm().item()),
                "fc1_weight_norm": float(model.blocks[i].fc1.weight.norm().item()),
            }
        return out


def _real_batch(task: Any, device: torch.device, n_rows: int) -> dict[str, Any]:
    rows = [int(x) for x in task.valid_rows[: int(n_rows)]]
    batch = C.PC.collate(task.cache, rows)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _lowrank_gradients(model: Any, task: Any, device: torch.device, n_rows: int) -> dict[str, Any]:
    """One real Validation-batch backward pass; no optimiser step."""
    rows = [int(x) for x in task.valid_rows[: int(n_rows)]]
    batch = _real_batch(task, device, n_rows)
    y = torch.as_tensor(task.labels[rows], dtype=torch.float32, device=device)
    model.train()
    model.zero_grad(set_to_none=True)
    logits = model(batch)
    loss = MET.masked_bce_loss(logits, y)
    loss.backward()
    grads = {
        name: (None if p.grad is None else float(p.grad.detach().norm().item()))
        for name, p in model.named_parameters()
        if FC.is_lowrank(name)
    }
    model.zero_grad(set_to_none=True)
    model.eval()
    return {
        "n_rows": int(len(rows)),
        "loss": float(loss.detach()),
        "per_tensor_norm": grads,
        "any_nonzero": bool(any(v for v in grads.values() if v)),
        "note": "diagnostic backward on real Valid rows; no optimiser step",
    }


def _molecule_decomposition(model: Any, task: Any, device: torch.device, n_rows: int) -> list[dict[str, Any]]:
    """Centre / neighbour pre-nonlinearity decomposition for a few real molecules."""
    out: list[dict[str, Any]] = []
    rs = str(getattr(model, "frs_arm", "ARM"))
    for row in [int(x) for x in task.valid_rows[: int(n_rows)]]:
        batch = C.PC.collate(task.cache, [row])
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.inference_mode():
            _a, _b, _scal, h0 = model.encode_h0(batch)
            src, dst, edge_msg = DSM.directed_bonds(model, batch)
            h1, m1 = model.stage1(h0, src, dst, edge_msg)
            z1 = model.blocks[0].norm(h0)
            z_target = model.blocks[1].norm(h1)
            z_source = model.blocks[1].norm(h0)
            m2 = DSM._agg_source(z_source, src, dst, edge_msg)
            rec: dict[str, Any] = {
                "valid_row": int(row),
                "n_atoms": int(h0.shape[0]),
                "n_directed_edges": int(src.numel()),
                "blocks": {},
            }
            for i, (z, m) in enumerate(((z1, m1), (z_target, m2))):
                W = model.blocks[i].fc1.weight.detach()
                D = model.lowrank[i].effective_weight().detach()
                if rs == "RSSC":
                    direct = z @ W.t() + m @ W.t() + (z - m) @ D.t()
                    centre = z @ (W + D).t()
                    neighbour = m @ (W - D).t()
                else:  # TIED
                    direct = (z + m) @ W.t() + (z + m) @ D.t()
                    centre = z @ (W + D).t()
                    neighbour = m @ (W + D).t()
                rec_term = centre + neighbour
                diff = (direct - rec_term).abs()
                scale = direct.abs().max().clamp(min=1e-12)
                rec["blocks"][str(i)] = {
                    "rms_centre_term": _rms(centre),
                    "rms_neighbour_term": _rms(neighbour),
                    "rms_direct_preactivation": _rms(direct),
                    "reconstruction_max_abs_error": float(diff.max().item()),
                    "reconstruction_rel_error": float((diff.max() / scale).item()),
                    "centre_share": (
                        _rms(centre) / (_rms(centre) + _rms(neighbour))
                        if (_rms(centre) + _rms(neighbour)) else None
                    ),
                }
        out.append(rec)
    return out


def _frozen_intervention(
    model: Any, task: Any, device: torch.device, arm: str
) -> dict[str, Any]:
    """Zero the added branch (A = 0 -> D = 0) and re-score full Validation."""
    if str(arm) != "RSSC":
        return {
            "applied": False,
            "reason": "protocol asks for the frozen branch-off intervention on the RSSC best checkpoint only",
        }
    model.eval()
    with torch.inference_mode():
        before = C.evaluate_split(model, task, device, "valid", per_task=True, keep_logits=True,
                                  reason="intervention_before")
    saved = {name: p.detach().clone() for name, p in model.named_parameters() if FC.is_lowrank(name)}
    with torch.no_grad():
        for i in (0, 1):
            model.lowrank[i].A.weight.zero_()
    with torch.inference_mode():
        after = C.evaluate_split(model, task, device, "valid", per_task=True, keep_logits=True,
                                 reason="intervention_branch_off")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in saved:
                p.copy_(saved[name])
    import numpy as np

    dl = np.asarray(after["logits"], dtype=np.float64) - np.asarray(before["logits"], dtype=np.float64)
    delta_abs = np.abs(dl)
    per_task_delta = [
        (None if (a is None or b is None) else float(b - a))
        for a, b in zip(before["per_task_ap"], after["per_task_ap"])
    ]
    scored = [d for d in per_task_delta if d is not None]
    return {
        "applied": True,
        "intervention": "A.weight := 0 in both blocks (equivalently D = 0); fc1, LN, residual, readout untouched",
        "n_tasks_scored": int(len(scored)),
        "valid_ap_with_branch": float(before["ap"]),
        "valid_ap_without_branch": float(after["ap"]),
        "valid_ap_delta": float(after["ap"] - before["ap"]),
        "valid_bce_with_branch": float(before["bce"]),
        "valid_bce_without_branch": float(after["bce"]),
        "valid_bce_delta": float(after["bce"] - before["bce"]),
        "logits_max_abs_change": float(delta_abs.max()),
        "logits_mean_abs_change": float(delta_abs.mean()),
        "logits_rms_change": float(np.sqrt((dl ** 2).mean())),
        "logits_changed_fraction_gt_1e-6": float((delta_abs > 1e-6).mean()),
        "tasks_improved_without_branch": int(sum(1 for d in scored if d > 1e-6)),
        "tasks_degraded_without_branch": int(sum(1 for d in scored if d < -1e-6)),
        "note": "frozen intervention only shows whether the trained model uses the branch; it is not an independent training result",
    }


def run(
    *,
    model: Any,
    task: Any,
    device: torch.device,
    arm: str,
    best_checkpoint: Any,
    model_audit: Mapping[str, Any],
    n_rows: int = 64,
) -> dict[str, Any]:
    batch = _real_batch(task, device, max(8, int(n_rows)))
    weights_source = "best_checkpoint" if best_checkpoint is not None else "last_epoch"
    record: dict[str, Any] = {
        "schema": FC.SCHEMA_MECHANISM,
        "round_id": FC.ROUND_ID,
        "arm": str(arm),
        "base_computation": "FIXED1",
        "branch_input": "z + m" if arm == "TIED" else "z - m",
        "rank": int(FC.RANK),
        "weights_source": weights_source,
        "parameters": int(model_audit["parameters"]["total"]),
        "blocks": {
            str(i): {
                "rank_branch": _d_stat(model.lowrank[i], model.blocks[i].fc1),
            }
            for i in (0, 1)
        },
        "scales_on_real_valid_batch": _block_scales(model, batch),
        "lowrank_gradients": _lowrank_gradients(model, task, device, max(8, int(n_rows))),
        "molecules": _molecule_decomposition(model, task, device, 3),
        "frozen_branch_zero": _frozen_intervention(model, task, device, str(arm)),
        "created_utc": C.utcnow(),
    }
    return record


__all__ = ["run"]
