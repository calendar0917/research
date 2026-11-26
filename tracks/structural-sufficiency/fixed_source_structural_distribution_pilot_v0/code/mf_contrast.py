"""Real-Valid contrast measurement for the MEANFIELD intervention.

Runs on a **fixed** official Valid sample (the first ``CONTRAST_ROWS`` rows of the
frozen Valid split, no shuffling) and measures how much the mean-field
replacement actually changes the computation:

* group-size distribution over directed edges and the singleton ratio;
* fraction of directed messages whose content is changed;
* RMS of the change of the aggregated message, per block, relative to the
  FIXED1 aggregation it replaces;
* per-group message-sum conservation error.

The gate is frozen in ``PROTOCOL.md``: if the actual intervention barely changes
the aggregation the round is labelled ``INSUFFICIENT_CONTRAST`` and no formal
training is started.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

import mf_common as R
import mf_model as MF
import ds_common as C
import ds_model as DM

CONTRAST_ROWS = 1024
CONTRAST_BATCH = 256
#: frozen gate (see PROTOCOL.md section 5) -- practical short-run scale only
MIN_RELATIVE_RMS_CHANGE = 1e-2
MIN_CHANGED_MESSAGE_FRACTION = 0.5


def _rms(x: torch.Tensor) -> float:
    return float(x.detach().double().pow(2).mean().sqrt().item()) if x.numel() else 0.0


def measure(task: Any, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    rows = np.asarray(task.valid_rows[:CONTRAST_ROWS], dtype=np.int64)
    sample_sha = C.sha256_bytes(np.ascontiguousarray(rows, dtype=np.int64).tobytes())
    was_training = model.training
    model.eval()

    sq_change = {1: 0.0, 2: 0.0}
    sq_base = {1: 0.0, 2: 0.0}
    n_values = {1: 0, 2: 0}
    sq_change_e2e = 0.0
    sq_base_e2e = 0.0
    n_edges = 0
    n_changed = 0
    group_sizes: list[torch.Tensor] = []
    max_abs_conservation_error = 0.0
    max_rel_conservation_error = 0.0
    n_groups_total = 0
    n_signatures: set[int] = set()
    degenerate_singleton_groups = 0
    odd_group_sizes = 0

    with torch.inference_mode():
        for start in range(0, int(rows.size), CONTRAST_BATCH):
            chunk = rows[start : start + CONTRAST_BATCH]
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in C.PC.collate(task.cache, chunk.tolist()).items()
            }
            _a, _b, _scal, h0 = model.encode_h0(batch)
            src, dst, edge_msg = DM.directed_bonds(model, batch)
            bf_dir, gb_dir = MF.directed_bond_features(batch)
            gid, n_groups, audit = MF.group_index(bf_dir, gb_dir)
            if int(src.numel()) == 0:
                continue
            sizes = MF.group_sizes(gid, n_groups)
            group_sizes.append(sizes.cpu())
            n_groups_total += int(n_groups)
            n_signatures.add(int(audit["n_signatures"]))
            degenerate_singleton_groups += int((sizes == 1).sum().item())
            odd_group_sizes += int((sizes % 2 == 1).sum().item())

            blk1, blk2 = model.blocks[0], model.blocks[1]
            z1 = blk1.norm(h0)
            m1_f1, p1, _ = MF.meanfield_message(z1, src, dst, edge_msg, gid, n_groups)
            # FIXED1 aggregation of the untouched content (same z_source)
            m1_ref = torch.zeros_like(z1).index_add_(0, dst, p1)
            _gm1, p1_mf = MF.meanfield_content(p1, gid, n_groups)
            m1_mf = torch.zeros_like(z1).index_add_(0, dst, p1_mf)
            h1 = h0 + DM._apply_ffn(blk1, z1, m1_mf)

            z2_source = blk2.norm(h0)
            p2 = torch.nn.functional.silu(z2_source.index_select(0, src) + edge_msg)
            _gm2, p2_mf = MF.meanfield_content(p2, gid, n_groups)
            m2_ref = torch.zeros_like(z2_source).index_add_(0, dst, p2)
            m2_mf = torch.zeros_like(z2_source).index_add_(0, dst, p2_mf)

            # end-to-end block-2 aggregation (each arm on its own h1 state)
            z1_f1 = blk1.norm(h0)
            h1_ref = h0 + DM._apply_ffn(blk1, z1_f1, m1_ref)
            m2_ref_e2e = torch.zeros_like(z2_source).index_add_(
                0, dst, torch.nn.functional.silu(blk2.norm(h1_ref).index_select(0, src) + edge_msg)
            )
            m2_mf_e2e = torch.zeros_like(z2_source).index_add_(
                0, dst, torch.nn.functional.silu(blk2.norm(h1).index_select(0, src) + edge_msg)
            )

            sq_change[1] += float((m1_mf - m1_ref).double().pow(2).sum().item())
            sq_base[1] += float(m1_ref.double().pow(2).sum().item())
            n_values[1] += int(m1_mf.numel())
            sq_change[2] += float((m2_mf - m2_ref).double().pow(2).sum().item())
            sq_base[2] += float(m2_ref.double().pow(2).sum().item())
            n_values[2] += int(m2_mf.numel())
            sq_change_e2e += float((m2_mf_e2e - m2_ref_e2e).double().pow(2).sum().item())
            sq_base_e2e += float(m2_ref_e2e.double().pow(2).sum().item())

            n_edges += int(p1.shape[0])
            n_changed += int((p1_mf != p1).any(dim=1).sum().item())

            before = torch.zeros(
                n_groups, p1.shape[1], dtype=torch.float64, device=p1.device
            )
            before.index_add_(0, gid, p1.double())
            after = torch.zeros_like(before)
            after.index_add_(0, gid, p1_mf.double())
            err = (after - before).abs()
            max_abs_conservation_error = max(max_abs_conservation_error, float(err.max().item()))
            rel = (err / before.abs().clamp(min=1e-30)).max()
            max_rel_conservation_error = max(max_rel_conservation_error, float(rel.item()))
    if was_training:
        model.train()

    sizes_all = torch.cat(group_sizes) if group_sizes else torch.zeros(0, dtype=torch.long)
    total_edges = int(sizes_all.sum().item()) if sizes_all.numel() else 0
    weighted = sizes_all.to(torch.float64) * sizes_all.to(torch.float64)  # edges per group size class
    hist = {
        str(int(s)): int((sizes_all == s).sum().item())
        for s in sorted({int(x) for x in sizes_all.tolist()})
    }
    rms = {
        l: float(np.sqrt(sq_change[l] / max(n_values[l], 1))) for l in (1, 2)
    }
    base = {l: float(np.sqrt(sq_base[l] / max(n_values[l], 1))) for l in (1, 2)}
    rel_rms = {l: (rms[l] / base[l]) if base[l] else 0.0 for l in (1, 2)}
    out = {
        "schema": R.SCHEMA_CHECKS + "-contrast",
        "round_id": R.ROUND_ID,
        "arm": R.ARM,
        "sample": {
            "split": "valid",
            "rows": "first %d rows of the frozen official Valid split (no shuffle)" % CONTRAST_ROWS,
            "n_rows": int(rows.size),
            "row_sha256": sample_sha,
            "first_row": int(rows[0]),
            "last_row": int(rows[-1]),
            "model_state": "MEANFIELD at its official initialisation (before any training step)",
        },
        "groups": {
            "n_directed_edges": int(n_edges),
            "n_groups": int(n_groups_total),
            "n_distinct_signatures_per_batch": sorted(n_signatures),
            "mean_group_size": (float(n_edges / n_groups_total) if n_groups_total else 0.0),
            "group_size_histogram": hist,
            "group_size_p50": int(np.percentile(sizes_all, 50)) if sizes_all.numel() else 0,
            "group_size_p90": int(np.percentile(sizes_all, 90)) if sizes_all.numel() else 0,
            "group_size_max": int(sizes_all.max().item()) if sizes_all.numel() else 0,
            "singleton_group_count": int(degenerate_singleton_groups),
            "singleton_fraction_of_groups": (
                float(degenerate_singleton_groups / n_groups_total) if n_groups_total else 0.0
            ),
            "singleton_fraction_of_edges": 0.0 if total_edges == 0 else float((sizes_all == 1).sum().item() * 1.0 / total_edges),
            "odd_size_group_count": int(odd_group_sizes),
            "note": (
                "a physical bond contributes two directed edges with the same raw bond feature, so both "
                "fall in one group: directed-edge group sizes are even and no group is a singleton"
            ),
            "_weighted_group_size": float(weighted.sum().item()) if weighted.numel() else 0.0,
        },
        "intervention": {
            "changed_message_fraction": (float(n_changed / n_edges) if n_edges else 0.0),
            "n_messages_content_changed": int(n_changed),
            "block1_rms_fixed1_aggregation": base[1],
            "block1_rms_change": rms[1],
            "block1_relative_rms_change": rel_rms[1],
            "block2_rms_fixed1_aggregation_same_state": base[2],
            "block2_rms_change_same_state": rms[2],
            "block2_relative_rms_change_same_state": rel_rms[2],
            "block2_relative_rms_change_end_to_end": (
                float(np.sqrt(sq_change_e2e / max(sq_base_e2e, 1e-30)))
            ),
            "max_abs_group_sum_conservation_error": max_abs_conservation_error,
            "max_relative_group_sum_conservation_error": max_rel_conservation_error,
        },
        "gate": {
            "min_relative_rms_change": MIN_RELATIVE_RMS_CHANGE,
            "min_changed_message_fraction": MIN_CHANGED_MESSAGE_FRACTION,
        },
    }
    out["groups"].pop("_weighted_group_size")
    enough = (
        rel_rms[1] >= MIN_RELATIVE_RMS_CHANGE
        and out["intervention"]["changed_message_fraction"] >= MIN_CHANGED_MESSAGE_FRACTION
    )
    out["status"] = "SUFFICIENT_CONTRAST" if enough else "INSUFFICIENT_CONTRAST"
    out["passed"] = bool(enough)
    return out


__all__ = ["CONTRAST_ROWS", "MIN_CHANGED_MESSAGE_FRACTION", "MIN_RELATIVE_RMS_CHANGE", "measure"]
