"""Phase 0 CPU research gates for ``zinc_block_ownership_v1``.

Run:
    python .../zbo_checks.py            # full gate suite (CPU only)

Writes ``analysis/cpu_checks.json`` and ``configs/zbo_v1.json``.  The config is
frozen only after every gate has run; it records the observed stop-gate values,
the block-cache hashes, the fixed worst-10 anchor rows and the group bins.

No GPU, no official test, no model selection.  Labels are read only in the
small gradient test (train rows) and in the fixed-worst-10 anchor (official
valid, from a frozen historical model); they never enter any structure/feature
construction.
"""

from __future__ import annotations

import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import zbo_common as BC  # noqa: E402  (also puts bosi_data on the path)
import bosi_data as BD  # noqa: E402
import zbo_blocks as ZB  # noqa: E402
import zbo_models as ZM  # noqa: E402

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(jsonable(v) for v in obj)
    return obj


def max_state_diff(a: Mapping[str, torch.Tensor],
                   b: Mapping[str, torch.Tensor]) -> float:
    m = 0.0
    for key in a:
        if key not in b:
            raise KeyError(key)
        m = max(m, float((a[key].float() - b[key].float()).abs().max()))
    return m


def max_tensor_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def make_mini_cache(specs: Sequence[Mapping[str, Any]]) -> dict:
    """A tiny cache dict with the exact structural keys used by the pipeline."""
    atom_ptr = [0]
    bond_ptr = [0]
    atom_feats, bond_feats, bond_indices, globs = [], [], [], []
    for s in specs:
        af = np.asarray(s["atom_feat"], dtype=np.float32)
        bi = np.asarray(s["bond_index"], dtype=np.int64).reshape(-1, 2)
        bf = np.asarray(s["bond_feat"], dtype=np.float32)
        atom_feats.append(af)
        bond_feats.append(bf)
        bond_indices.append(bi)
        globs.append(np.asarray(s["global"], dtype=np.float32).reshape(-1))
        atom_ptr.append(atom_ptr[-1] + len(af))
        bond_ptr.append(bond_ptr[-1] + len(bi))

    def cat(xs, dtype):
        return np.concatenate(xs, 0).astype(dtype) if xs else np.zeros((0,), dtype)

    n = len(specs)
    return {
        "split": "synthetic",
        "n_molecules": n,
        "atom_ptr": np.asarray(atom_ptr, dtype=np.int64),
        "bond_ptr": np.asarray(bond_ptr, dtype=np.int64),
        "atom_feat": cat(atom_feats, np.float32),
        "bond_feat": cat(bond_feats, np.float32),
        "bond_index": cat(bond_indices, np.int64),
        "bond_type": np.zeros(int(bond_ptr[-1]), dtype=np.int64),
        "global_feat": np.stack(globs, 0) if globs else np.zeros((0, BC.GLOBAL_DIM), np.float32),
        "labels": np.zeros(n, dtype=np.float32),
    }


def specs_from_rows(cache: Mapping[str, Any], rows: Sequence[int]) -> list[dict]:
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    af = np.asarray(cache["atom_feat"], dtype=np.float32)
    bf = np.asarray(cache["bond_feat"], dtype=np.float32)
    gf = np.asarray(cache["global_feat"], dtype=np.float32)
    out = []
    for row in rows:
        a0, a1 = int(ap[row]), int(ap[row + 1])
        b0, b1 = int(bp[row]), int(bp[row + 1])
        out.append({
            "atom_feat": af[a0:a1].copy(),
            "bond_index": bi[b0:b1].copy(),
            "bond_feat": bf[b0:b1].copy(),
            "global": gf[row].copy(),
        })
    return out


def build_mini(specs: Sequence[Mapping[str, Any]]):
    cache = make_mini_cache(specs)
    rel = BD.build_relation_cache(cache)
    blocks = ZB.build_split_blocks(cache)
    batch = ZB.collate(cache, rel, blocks, np.arange(len(specs)))
    return cache, rel, blocks, batch


def model_h(model: ZM.BlockBOSI, c: torch.Tensor, batch: Mapping[str, Any],
            arm: str) -> torch.Tensor:
    cf, cs = ZB.block_content(c, batch, arm)
    u = ZB.block_input(cf, cs, batch)
    phi = model.block_mlp(model.block_norm(u))
    return ZB.pooled_block_channel(phi, batch, BC.BLOCK_OUT)


# ---------------------------------------------------------------------------
# C1: block partition correctness
# ---------------------------------------------------------------------------


def gate_c1(cache: Mapping[str, Any], split: str) -> dict:
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    old_bptr = np.asarray(cache["block_ptr"], dtype=np.int64)
    old_bbp = np.asarray(cache["block_bond_ptr"], dtype=np.int64)
    old_bb = np.asarray(cache["block_bond_idx"], dtype=np.int64)
    old_bap = np.asarray(cache["block_atom_ptr"], dtype=np.int64)
    old_scalar = np.asarray(cache["block_scalar"], dtype=np.float32)

    n_mol = len(ap) - 1
    mismatch_nx = 0
    mismatch_old = 0
    unassigned = 0
    duplicate = 0
    bad_rank = 0
    bad_artic = 0
    old_zero_edge = 0
    old_blocks = 0
    ours_blocks = 0
    examples: list[dict] = []
    t0 = time.perf_counter()
    for i in range(n_mol):
        a0, a1 = int(ap[i]), int(ap[i + 1])
        b0, b1 = int(bp[i]), int(bp[i + 1])
        r = None
        try:
            r = ZB.build_molecule_blocks(a1 - a0, bi[b0:b1])
        except Exception as exc:  # Tarjan/NetworkX disagreement is a failed gate
            mismatch_nx += 1
            if len(examples) < 3:
                examples.append({"molecule": i, "error": repr(exc)})
            continue
        ours = [frozenset(int(e) for e in comp) for comp in r["block_bonds"]]
        ours_blocks += len(ours)
        if len(set(ours)) != len(ours):
            duplicate += 1
        if b1 > b0 and (np.asarray(r["bond_block"]) < 0).any():
            unassigned += 1
        flat = sorted(int(e) for comp in r["block_bonds"] for e in comp)
        if flat != list(range(b1 - b0)):
            unassigned += 1
        for bid in range(len(r["block_bonds"])):
            if int(r["rank"][bid]) != int(r["n_edges"][bid] - r["n_atoms"][bid] + 1):
                bad_rank += 1

        g0, g1 = int(old_bptr[i]), int(old_bptr[i + 1])
        old_sets = []
        old_by_set = {}
        for ob in range(g0, g1):
            s, e = int(old_bbp[ob]), int(old_bbp[ob + 1])
            if e == s:
                old_zero_edge += 1
                continue
            key = frozenset(int(x) for x in old_bb[s:e])
            old_sets.append(key)
            old_by_set[key] = ob
        old_blocks += len(old_sets)
        if set(ours) != set(old_sets):
            mismatch_old += 1
            if len(examples) < 3:
                examples.append({"molecule": i, "ours": [sorted(x) for x in ours],
                                 "old": [sorted(x) for x in old_sets]})
        for bid, comp in enumerate(r["block_bonds"]):
            key = frozenset(int(e) for e in comp)
            ob = old_by_set.get(key)
            if ob is None:
                continue
            old_ne = int(old_bbp[ob + 1] - old_bbp[ob])
            old_na = int(old_bap[ob + 1] - old_bap[ob])
            if old_ne != int(r["n_edges"][bid]) or old_na != int(r["n_atoms"][bid]):
                bad_rank += 1
            if int(round(float(old_scalar[ob, 9]))) != int(r["n_artic"][bid]):
                bad_artic += 1

    elapsed = time.perf_counter() - t0
    passed = (mismatch_nx == 0 and mismatch_old == 0 and unassigned == 0
              and duplicate == 0 and bad_rank == 0 and bad_artic == 0)
    return {
        "split": split,
        "n_molecules": n_mol,
        "n_blocks_ours": ours_blocks,
        "n_blocks_old_cache": old_blocks,
        "old_zero_edge_blocks": old_zero_edge,
        "mismatch_networkx": mismatch_nx,
        "mismatch_old_cache": mismatch_old,
        "unassigned_or_missing_bonds": unassigned,
        "duplicate_block_sets": duplicate,
        "bad_rank": bad_rank,
        "bad_articulation_count": bad_artic,
        "examples": examples,
        "seconds": elapsed,
        "pass": passed,
    }


# ---------------------------------------------------------------------------
# C2: invariance
# ---------------------------------------------------------------------------


def gate_c2(cache: Mapping[str, Any], blocks: Mapping[str, Any]) -> dict:
    n_multi = BC.block_row_features(blocks)["n_multi_blocks"]
    candidate_rows = np.where(n_multi >= 2)[0][:8]
    specs = specs_from_rows(cache, candidate_rows)
    _, _, _, base_batch = build_mini(specs)
    norm = BC.load_norm()
    model, info = ZM.build_model(norm, "R", 0, "cpu")
    # Make the block channel actually affect the output for this gate.
    gen = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        model.reader_ext.weight.normal_(0.0, 0.05, generator=gen)
    model.eval()
    with torch.no_grad():
        base_pred, base_st = model(base_batch, return_stats=True)
        base_z = base_st["z_new"].clone()

    def run_variant(var_specs):
        _, _, _, batch = build_mini(var_specs)
        with torch.no_grad():
            pred, st = model(batch, return_stats=True)
        return pred, st["z_new"]

    # Atom relabelling, independently per molecule.
    rng = np.random.default_rng(7)
    relabelled = []
    for s in specs:
        n = len(s["atom_feat"])
        p = rng.permutation(n)
        inv = np.empty(n, dtype=np.int64)
        inv[p] = np.arange(n)
        item = dict(s)
        item["atom_feat"] = s["atom_feat"][p]
        item["bond_index"] = inv[s["bond_index"]]
        relabelled.append(item)
    pred_r, z_r = run_variant(relabelled)

    # Edge reversal.
    reversed_specs = []
    for s in specs:
        item = dict(s)
        item["bond_index"] = s["bond_index"][:, ::-1].copy()
        reversed_specs.append(item)
    pred_rev, z_rev = run_variant(reversed_specs)

    # Edge order.
    rng2 = np.random.default_rng(11)
    ordered_specs = []
    for s in specs:
        perm = rng2.permutation(len(s["bond_index"]))
        item = dict(s)
        item["bond_index"] = s["bond_index"][perm].copy()
        item["bond_feat"] = s["bond_feat"][perm].copy()
        ordered_specs.append(item)
    pred_ord, z_ord = run_variant(ordered_specs)

    # Batch reordering.
    perm = list(range(len(specs)))[::-1]
    _, _, _, batch_perm = build_mini([specs[i] for i in perm])
    with torch.no_grad():
        pred_perm, _ = model(batch_perm, return_stats=True)
    back = np.argsort(np.asarray(perm))
    pred_perm_back = pred_perm[torch.as_tensor(back, dtype=torch.long)]

    out = {
        "n_molecules": len(specs),
        "reader_ext_randomised": True,
        "atom_relabel_max_pred_diff": max_tensor_diff(base_pred, pred_r),
        "atom_relabel_max_znew_diff": max_tensor_diff(base_z, z_r),
        "edge_reverse_max_pred_diff": max_tensor_diff(base_pred, pred_rev),
        "edge_reverse_max_znew_diff": max_tensor_diff(base_z, z_rev),
        "edge_order_max_pred_diff": max_tensor_diff(base_pred, pred_ord),
        "edge_order_max_znew_diff": max_tensor_diff(base_z, z_ord),
        "batch_reorder_max_pred_diff": max_tensor_diff(base_pred, pred_perm_back),
        "parameters": info["parameters"],
    }
    vals = [v for k, v in out.items() if k.endswith("_diff") and isinstance(v, float)]
    out["pass"] = all(v <= 1e-5 for v in vals)
    return out


# ---------------------------------------------------------------------------
# C3: conservation
# ---------------------------------------------------------------------------


def gate_c3(cache: Mapping[str, Any], rel: Mapping[str, Any],
            blocks: Mapping[str, Any]) -> dict:
    norm = BC.load_norm()
    model, _ = ZM.build_model(norm, "R", 0, "cpu")
    model.eval()
    rows = np.arange(128)
    batch = ZB.collate(cache, rel, blocks, rows)
    with torch.no_grad():
        _, st = model(batch, return_stats=True)
    c64 = st["block_c"].double()
    batch64 = dict(batch)
    batch64["block_n_edges"] = batch["block_n_edges"].double()
    B = int(batch["n_graphs"])
    W = c64.shape[1]
    true_f = torch.zeros(B, W, dtype=torch.float64).index_add_(
        0, batch["bond_batch"], c64)
    true_s = torch.zeros(B, W, dtype=torch.float64).index_add_(
        0, batch["bond_batch"], c64 * c64)

    arms = {}
    for arm in ("R", "N", "S"):
        cf, cs = ZB.block_content(c64, batch64, arm)
        tf = torch.zeros(B, W, dtype=torch.float64).index_add_(
            0, batch["block_batch"], cf)
        ts = torch.zeros(B, W, dtype=torch.float64).index_add_(
            0, batch["block_batch"], cs)
        arms[arm] = {
            "first_max_abs_diff_vs_true": float((tf - true_f).abs().max()),
            "second_max_abs_diff_vs_true": float((ts - true_s).abs().max()),
        }

    cfR, csR = ZB.block_content(c64, batch64, "R")
    cfN, csN = ZB.block_content(c64, batch64, "N")
    cfS, csS = ZB.block_content(c64, batch64, "S")
    bn = batch64["block_n_edges"]
    bb = batch["block_batch"]
    e = torch.zeros(B, dtype=torch.float64).index_add_(
        0, batch["bond_batch"],
        torch.ones(batch["bond_batch"].numel(), dtype=torch.float64))
    frac_n = bn / e[bb].clamp(min=1.0)
    frac_sum = torch.zeros(B, dtype=torch.float64).index_add_(0, bb, frac_n)
    multi = bn > 1.0
    em = torch.zeros(B, dtype=torch.float64).index_add_(
        0, bb, torch.where(multi, bn, torch.zeros_like(bn)))
    frac_s = bn / em[bb].clamp(min=1.0)
    frac_s_sum = torch.zeros(B, dtype=torch.float64).index_add_(
        0, bb, torch.where(multi, frac_s, torch.zeros_like(frac_s)))
    has_multi = em > 0
    s_dev = float((frac_s_sum[has_multi] - 1.0).abs().max()) if bool(has_multi.any()) else 0.0
    s_no_multi_max = float(frac_s_sum[~has_multi].abs().max()) if bool((~has_multi).any()) else 0.0
    single = bn == 1.0
    out = {
        "n_rows": int(len(rows)),
        "n_blocks": int(batch["n_blocks"]),
        "arms": arms,
        "R_N_first_max_abs_diff": float((cfR - cfN).abs().max()),
        "R_N_second_max_abs_diff": float((csR - csN).abs().max()),
        "R_S_first_max_abs_diff": float((cfR - cfS).abs().max()),
        "R_S_second_max_abs_diff": float((csR - csS).abs().max()),
        "S_singleton_first_max_abs_diff": float((cfS[single] - cfR[single]).abs().max())
        if bool(single.any()) else 0.0,
        "S_singleton_second_max_abs_diff": float((csS[single] - csR[single]).abs().max())
        if bool(single.any()) else 0.0,
        "N_allocation_sum_max_dev": float((frac_sum - 1.0).abs().max()),
        "S_multi_allocation_sum_max_dev": s_dev,
        "S_no_multi_graph_sum_max_abs": s_no_multi_max,
        "n_graphs_with_multi": int(has_multi.sum()),
        "n_graphs_without_multi": int((~has_multi).sum()),
    }
    tol = 1e-9
    out["pass"] = all([
        arms["R"]["first_max_abs_diff_vs_true"] <= tol,
        arms["R"]["second_max_abs_diff_vs_true"] <= tol,
        arms["N"]["first_max_abs_diff_vs_true"] <= tol,
        arms["N"]["second_max_abs_diff_vs_true"] <= tol,
        arms["S"]["first_max_abs_diff_vs_true"] <= tol,
        arms["S"]["second_max_abs_diff_vs_true"] <= tol,
        out["N_allocation_sum_max_dev"] <= tol,
        out["S_multi_allocation_sum_max_dev"] <= tol,
        out["S_no_multi_graph_sum_max_abs"] <= tol,
        out["S_singleton_first_max_abs_diff"] <= tol,
        out["S_singleton_second_max_abs_diff"] <= tol,
    ])
    return out


# ---------------------------------------------------------------------------
# C4: synthetic counterexample
# ---------------------------------------------------------------------------


def gate_c4() -> dict:
    # Two triangles joined by a bridge: blocks [3, 1, 3].
    atom_feat = np.zeros((6, BC.N_ATOM_FEAT), dtype=np.float32)
    bond_feat = np.zeros((7, BC.BOND_FEAT), dtype=np.float32)
    bi = np.asarray([[0, 1], [1, 2], [2, 0],
                     [3, 4], [4, 5], [5, 3],
                     [2, 3]], dtype=np.int64)
    spec = {"atom_feat": atom_feat, "bond_index": bi, "bond_feat": bond_feat,
            "global": np.zeros(BC.GLOBAL_DIM, dtype=np.float32)}
    _, _, blocks, batch = build_mini([spec])
    ne = np.asarray(blocks["block_n_edges"], dtype=np.int64)
    if sorted(ne.tolist()) != [1, 3, 3]:
        raise RuntimeError(f"unexpected synthetic block sizes {ne}")

    gen = torch.Generator().manual_seed(20261027)
    vals = torch.randn(7, BC.WIDTH, generator=gen)
    # Assignment A: triangle A gets 0,1,2; triangle B gets 3,4,5; bridge 6.
    # Assignment B: same multiset, different ownership.
    assign_a = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
    assign_b = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 5, 6: 6}
    c_a = torch.stack([vals[assign_a[i]] for i in range(7)], 0)
    c_b = torch.stack([vals[assign_b[i]] for i in range(7)], 0)
    if not torch.equal(torch.sort(c_a, 0).values, torch.sort(c_b, 0).values):
        raise RuntimeError("counterexample bond-state multisets differ")

    norm = BC.load_norm()
    model, _ = ZM.build_model(norm, "R", 0, "cpu")
    model.eval()
    out = {"block_sizes": sorted(int(x) for x in ne), "bond_multiset_equal": True}
    hs = {}
    for arm in ("R", "N", "S"):
        with torch.no_grad():
            hs[arm] = (model_h(model, c_a, batch, arm),
                       model_h(model, c_b, batch, arm))
        out[f"{arm}_max_abs_diff"] = max_tensor_diff(hs[arm][0], hs[arm][1])
        out[f"{arm}_h_norm"] = float(hs[arm][0].norm())
    # A purely linear pooling of the per-block mean content is invariant.
    with torch.no_grad():
        linear = []
        for c in (c_a, c_b):
            cf, _ = ZB.block_content(c, batch, "R")
            u = ZB.block_input(cf, torch.zeros_like(cf), batch)
            linear.append(u[:, :BC.WIDTH].sum(0))
        out["linear_pool_max_abs_diff"] = max_tensor_diff(linear[0], linear[1])
    out["pass"] = (
        out["R_max_abs_diff"] > 1e-6
        and out["N_max_abs_diff"] <= 1e-6
        and out["S_max_abs_diff"] <= 1e-6
        and out["linear_pool_max_abs_diff"] <= 1e-6
    )
    out["interpretation"] = (
        "Operator-level witness only: the same bond-state multiset and the same "
        "block sizes can carry different true ownership. A nonlinear shared phi "
        "separates R; N and S are numerically blind to the permutation "
        "(float32 reduction order only); a purely linear pooling is also blind. "
        "This is not a chemistry-label proof."
    )
    return out


# ---------------------------------------------------------------------------
# C5: real ZINC statistics + R/S coverage
# ---------------------------------------------------------------------------


def gate_c5(cache: Mapping[str, Any], rel: Mapping[str, Any],
            blocks: Mapping[str, Any]) -> dict:
    stats = ZB.block_stats(blocks)
    n_multi = BC.block_row_features(blocks)["n_multi_blocks"]
    structural = float((n_multi >= 2).mean())

    norm = BC.load_norm()
    model, _ = ZM.build_model(norm, "R", 0, "cpu")
    model.eval()
    rel_first, rel_second = [], []
    n_affected_blocks = 0
    n_affected_molecules = 0
    rows = np.arange(int(cache["n_molecules"]))
    bsz = 256
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(rows), bsz):
            chunk = rows[start:start + bsz]
            batch = ZB.collate(cache, rel, blocks, chunk)
            _, st = model(batch, return_stats=True)
            c64 = st["block_c"].double()
            batch64 = dict(batch)
            batch64["block_n_edges"] = batch["block_n_edges"].double()
            cfR, csR = ZB.block_content(c64, batch64, "R")
            cfS, csS = ZB.block_content(c64, batch64, "S")
            bn = batch64["block_n_edges"]
            bb = batch["block_batch"]
            B = int(batch["n_graphs"])
            n_multi_g = torch.zeros(B, dtype=torch.float64).index_add_(
                0, bb, (bn > 1.0).double())
            affected = (n_multi_g[bb] >= 2.0) & (bn > 1.0)
            n_affected_blocks += int(affected.sum())
            n_affected_molecules += int((n_multi_g >= 2.0).sum())
            if bool(affected.any()):
                denom = bn.clamp(min=1.0).unsqueeze(1)
                mR = cfR / denom
                mS = cfS / denom
                d1 = (mR - mS).norm(dim=1) / (mR.norm(dim=1) + 1e-12)
                m2R = csR / denom
                m2S = csS / denom
                d2 = (m2R - m2S).norm(dim=1) / (m2R.norm(dim=1) + 1e-12)
                rel_first.append(d1[affected].cpu().numpy())
                rel_second.append(d2[affected].cpu().numpy())
    elapsed = time.perf_counter() - t0
    r1 = np.concatenate(rel_first) if rel_first else np.zeros(0)
    r2 = np.concatenate(rel_second) if rel_second else np.zeros(0)
    frac_1pct = float((r1 > 0.01).mean()) if len(r1) else 0.0
    mean_rel = float(r1.mean()) if len(r1) else 0.0
    out = {
        "block_stats": stats,
        "fraction_molecules_ge2_multi": structural,
        "n_affected_blocks": int(n_affected_blocks),
        "n_affected_molecules_batches": int(n_affected_molecules),
        "rs_first_relative_diff": {
            "n": int(len(r1)),
            "mean": mean_rel,
            "median": float(np.median(r1)) if len(r1) else 0.0,
            "p90": float(np.percentile(r1, 90)) if len(r1) else 0.0,
            "fraction_gt_1pct": frac_1pct,
        },
        "rs_second_relative_diff": {
            "n": int(len(r2)),
            "mean": float(r2.mean()) if len(r2) else 0.0,
            "median": float(np.median(r2)) if len(r2) else 0.0,
            "p90": float(np.percentile(r2, 90)) if len(r2) else 0.0,
            "fraction_gt_1pct": float((r2 > 0.01).mean()) if len(r2) else 0.0,
        },
        "seconds": elapsed,
    }
    out["stop_gate_structural_pass"] = structural >= 0.20
    out["stop_gate_numeric_pass"] = (frac_1pct > 0.50 and mean_rel > 0.01)
    out["pass"] = bool(out["stop_gate_structural_pass"] and out["stop_gate_numeric_pass"])
    return out


# ---------------------------------------------------------------------------
# C6: historical BlockModel contrast
# ---------------------------------------------------------------------------


def gate_c6() -> dict:
    import ast
    import re

    core = [HERE / "zbo_blocks.py", HERE / "zbo_models.py", HERE / "zbo_prep.py"]
    forbidden_names = {"block_joint", "block_dist", "block_overlap",
                       "GlobalAttention", "BlockModel"}
    hits = {}
    for path in core:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        names = set()
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Import):
                modules.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.add(node.module or "")
        quoted = set(re.findall(r"[\"'](block_joint|block_dist|block_overlap)[\"']", text))
        hit = sorted((names | quoted) & forbidden_names)
        if any("ad_models" in m for m in modules):
            hit.append("ad_models")
        hits[path.name] = sorted(set(hit))
    return {
        "core_files": [p.name for p in core],
        "forbidden_token_hits": hits,
        "old_block_feature_dims": {
            "joint_histogram": 693,
            "atom_type_histogram": 21,
            "bond_type_histogram": 3,
            "block_scalars": 11,
            "attention": "1-layer GlobalAttention(block-cut-tree distance/overlap)",
        },
        "new_block_feature_dims": {
            "learned_content_first": BC.WIDTH,
            "learned_content_second": BC.WIDTH,
            "block_scalars": ZB.N_BLOCK_SCALARS,
            "attention": None,
        },
        "old_block_model_parameters": 338_805,
        "new_model_parameters": BC.EXPECTED_PARAMETERS,
        "old_block_model_result": "best-valid 0.165418 (zinc_arch_discovery_v1)",
        "pass": all(len(v) == 0 for v in hits.values()),
        "interpretation": (
            "Only the deterministic biconnected partition is shared with the "
            "historical BlockModel; its type histograms, block-cut-tree "
            "distance/overlap and attention are not imported or called."
        ),
    }


# ---------------------------------------------------------------------------
# C7: parameters / initialisation / forward equivalence
# ---------------------------------------------------------------------------


def gate_c7(cache: Mapping[str, Any], rel: Mapping[str, Any],
            blocks: Mapping[str, Any]) -> dict:
    norm = BC.load_norm()
    models = {}
    infos = {}
    for arm in ("R", "N", "S"):
        models[arm], infos[arm] = ZM.build_model(norm, arm, 0, "cpu")
    states = {arm: models[arm].state_dict() for arm in models}
    pairwise = {}
    for a, b in (("R", "N"), ("R", "S"), ("N", "S")):
        pairwise[f"{a}_{b}"] = max_state_diff(states[a], states[b])

    ref = ZM.reference_bosi(norm, 0, "cpu")
    ref_state = ref.state_dict()
    common = max_state_diff({k: states["R"][k] for k in ref_state}, ref_state)

    rows = np.arange(64)
    batch = ZB.collate(cache, rel, blocks, rows)
    models["R"].eval()
    ref.eval()
    with torch.no_grad():
        y_new = models["R"](batch)
        y_ref = ref(batch)
    forward_diff = max_tensor_diff(y_new, y_ref)
    forward_equal = bool(torch.equal(y_new, y_ref))

    again, _ = ZM.build_model(norm, "R", 0, "cpu")
    repeat = max_state_diff(states["R"], again.state_dict())

    opt = torch.optim.AdamW(models["R"].parameters(), lr=1e-3)
    opt_params = int(sum(p.numel() for group in opt.param_groups for p in group["params"]))

    infos_out = {arm: {
        "parameters": infos[arm]["parameters"],
        "parameters_breakdown": infos[arm]["parameters_breakdown"],
        "initial_state_sha256": infos[arm]["initial_state_sha256"],
        "ssl": infos[arm]["ssl"],
    } for arm in infos}
    return {
        "parameters": infos["R"]["parameters"],
        "parameters_breakdown": infos["R"]["parameters_breakdown"],
        "parameter_cap": BC.PARAM_CAP,
        "pairwise_full_state_max_abs_diff": pairwise,
        "common_vs_reference_bosi_b_state_max_abs_diff": common,
        "forward_vs_reference_bosi_b_max_abs_diff": forward_diff,
        "forward_bitwise_equal": forward_equal,
        "repeat_build_state_max_abs_diff": repeat,
        "reader_ext_weight_zero_at_init": bool(
            torch.count_nonzero(models["R"].reader_ext.weight) == 0),
        "optimizer_parameter_count": opt_params,
        "optimizer_covers_all": opt_params == infos["R"]["parameters"],
        "arms": infos_out,
        "pass": (
            infos["R"]["parameters"] == BC.EXPECTED_PARAMETERS
            and infos["R"]["parameters"] <= BC.PARAM_CAP
            and all(v == 0.0 for v in pairwise.values())
            and common == 0.0
            and forward_diff == 0.0
            and forward_equal
            and repeat == 0.0
            and bool(torch.count_nonzero(models["R"].reader_ext.weight) == 0)
            and opt_params == infos["R"]["parameters"]
        ),
    }


# ---------------------------------------------------------------------------
# C8: gradients / activation / numerical stability
# ---------------------------------------------------------------------------


def gate_c8(cache: Mapping[str, Any], rel: Mapping[str, Any],
            blocks: Mapping[str, Any]) -> dict:
    norm = BC.load_norm()
    model, _ = ZM.build_model(norm, "R", 0, "cpu")
    rows = np.arange(64)
    batch = ZB.collate(cache, rel, blocks, rows)
    y = np.asarray(cache["labels"], dtype=np.float64)[rows]
    target = torch.as_tensor(y, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                            weight_decay=BC.RECIPE["weight_decay"])

    def grad_norm(module):
        return float(sum((p.grad.detach() ** 2).sum()
                         for p in module.parameters() if p.grad is not None) ** 0.5)

    log = []
    for step in range(4):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = F.l1_loss(pred.view(-1), target.view(-1))
        loss.backward()
        log.append({
            "step": step,
            "loss": float(loss.detach()),
            "block_norm_grad": grad_norm(model.block_norm),
            "block_mlp_grad": grad_norm(model.block_mlp),
            "reader_ext_grad": grad_norm(model.reader_ext),
        })
        opt.step()

    model.eval()
    with torch.no_grad():
        _, st = model(batch, return_stats=True)
        znew_std = float(st["z_new"].std())
        znew_absmax = float(st["z_new"].abs().max())
        phi_std = float(st["block_phi"].std())
        finite = all(bool(torch.isfinite(t).all()) for t in
                     (st["z_new"], st["block_u"], st["block_phi"], st["block_c"]))
    weights_finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    re_norm = float(model.reader_ext.weight.norm())
    out = {
        "rows": int(len(rows)),
        "grad_log": log,
        "block_mlp_grad_step1": log[1]["block_mlp_grad"],
        "reader_ext_grad_step0": log[0]["reader_ext_grad"],
        "znew_std_after_4_steps": znew_std,
        "znew_absmax_after_4_steps": znew_absmax,
        "phi_std_after_4_steps": phi_std,
        "reader_ext_weight_norm_after_4_steps": re_norm,
        "finite_activations": finite,
        "finite_parameters": weights_finite,
    }
    out["pass"] = (
        out["block_mlp_grad_step1"] > 1e-8
        and out["reader_ext_grad_step0"] > 1e-8
        and out["znew_std_after_4_steps"] > 1e-6
        and out["reader_ext_weight_norm_after_4_steps"] > 1e-6
        and out["finite_activations"]
        and out["finite_parameters"]
        and all(math.isfinite(x["loss"]) for x in log)
    )
    return out


# ---------------------------------------------------------------------------
# C9: label isolation
# ---------------------------------------------------------------------------


def gate_c9(cache_train: Mapping[str, Any], cache_valid: Mapping[str, Any],
            rel_valid: Mapping[str, Any], blocks_valid: Mapping[str, Any]) -> dict:
    # Rebuilding the block cache with labels set to NaN must not change it.
    def nan_copy(cache):
        c = dict(cache)
        c["labels"] = np.full_like(np.asarray(cache["labels"]), np.nan)
        return c

    hash_train_a = ZB.build_split_blocks(cache_train)["content_sha256"]
    hash_train_b = ZB.build_split_blocks(nan_copy(cache_train))["content_sha256"]
    hash_valid_a = ZB.build_split_blocks(cache_valid)["content_sha256"]
    hash_valid_b = ZB.build_split_blocks(nan_copy(cache_valid))["content_sha256"]

    norm = BC.load_norm()
    model, _ = ZM.build_model(norm, "R", 0, "cpu")
    rows = np.arange(64)
    pred_a = BC.predict(model, cache_valid, rel_valid, blocks_valid, rows, device="cpu")
    # The prediction path never reads labels; this is a runtime witness.
    pred_b = BC.predict(model, nan_copy(cache_valid), rel_valid, blocks_valid,
                        rows, device="cpu")
    pred_diff = float(np.max(np.abs(pred_a - pred_b)))

    core_text = "\n".join((HERE / name).read_text(encoding="utf-8")
                          for name in ("zbo_blocks.py", "zbo_models.py", "zbo_prep.py"))
    test_refs = [s for s in ("ad_test.pt", "DATA_TEST", "official_test_cache")
                 if s in core_text]
    label_access = {name: any(tok in (HERE / name).read_text(encoding="utf-8")
                              for tok in ('["labels"]', "['labels']"))
                    for name in ("zbo_blocks.py", "zbo_models.py", "zbo_prep.py")}
    out = {
        "block_hash_train": hash_train_a,
        "block_hash_train_with_nan_labels": hash_train_b,
        "block_hash_valid": hash_valid_a,
        "block_hash_valid_with_nan_labels": hash_valid_b,
        "prediction_max_abs_diff_with_nan_valid_labels": pred_diff,
        "test_cache_references_in_core": test_refs,
        "label_access_in_core": label_access,
        "pass": (
            hash_train_a == hash_train_b
            and hash_valid_a == hash_valid_b
            and pred_diff == 0.0
            and not test_refs
            and not any(label_access.values())
        ),
    }
    return out


# ---------------------------------------------------------------------------
# C10: cache reproducibility
# ---------------------------------------------------------------------------


def gate_c10(cache_train: Mapping[str, Any], cache_valid: Mapping[str, Any],
             blocks_train: Mapping[str, Any], blocks_valid: Mapping[str, Any],
             rel_train: Mapping[str, Any], rel_valid: Mapping[str, Any]) -> dict:
    t0 = time.perf_counter()
    rel_train2 = BD.build_relation_cache(cache_train)
    rel_valid2 = BD.build_relation_cache(cache_valid)
    blk_train2 = ZB.build_split_blocks(cache_train)
    blk_valid2 = ZB.build_split_blocks(cache_valid)
    out = {
        "relation_train_content_hash": BD.relation_content_hash(rel_train),
        "relation_train_rebuild_hash": BD.relation_content_hash(rel_train2),
        "relation_valid_content_hash": BD.relation_content_hash(rel_valid),
        "relation_valid_rebuild_hash": BD.relation_content_hash(rel_valid2),
        "block_train_content_hash": blocks_train["content_sha256"],
        "block_train_rebuild_hash": blk_train2["content_sha256"],
        "block_valid_content_hash": blocks_valid["content_sha256"],
        "block_valid_rebuild_hash": blk_valid2["content_sha256"],
        "known_relation_hashes": {
            "train": "c698a2f30e0fe9f2fc51e79f37156e81da56f44b6a47843293dddb308ceda2b7",
            "valid": "6505e23bdcbbaf42b64aa38f11af06474f76081d0d363cddd25ea6427c7cebbb",
        },
        "seconds": time.perf_counter() - t0,
    }
    out["pass"] = (
        out["relation_train_content_hash"] == out["relation_train_rebuild_hash"]
        == out["known_relation_hashes"]["train"]
        and out["relation_valid_content_hash"] == out["relation_valid_rebuild_hash"]
        == out["known_relation_hashes"]["valid"]
        and out["block_train_content_hash"] == out["block_train_rebuild_hash"]
        and out["block_valid_content_hash"] == out["block_valid_rebuild_hash"]
    )
    return out


# ---------------------------------------------------------------------------
# C11: synthetic small graph exactness
# ---------------------------------------------------------------------------


def gate_c11() -> dict:
    atom_feat = np.zeros((6, BC.N_ATOM_FEAT), dtype=np.float32)
    bond_feat = np.zeros((6, BC.BOND_FEAT), dtype=np.float32)

    def blocks_for(edges, n_atoms):
        bi = np.asarray(edges, dtype=np.int64)
        return ZB.build_molecule_blocks(n_atoms, bi)

    cases = {
        "chain4": {
            "edges": [[0, 1], [1, 2], [2, 3]], "n_atoms": 4,
            "sizes": [1, 1, 1], "ranks": [0, 0, 0], "artic": [1, 2, 1],
        },
        "triangle": {
            "edges": [[0, 1], [1, 2], [2, 0]], "n_atoms": 3,
            "sizes": [3], "ranks": [1], "artic": [0],
        },
        "fused_triangles": {
            "edges": [[0, 1], [1, 2], [2, 0], [1, 3], [3, 2]], "n_atoms": 4,
            "sizes": [5], "ranks": [2], "artic": [0],
        },
        "two_triangles_bridge": {
            "edges": [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [2, 3]],
            "n_atoms": 6, "sizes": [3, 1, 3], "ranks": [1, 0, 1],
            "artic": [1, 2, 1],
        },
        "square_chord": {
            "edges": [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], "n_atoms": 4,
            "sizes": [5], "ranks": [2], "artic": [0],
        },
    }
    out = {}
    ok = True
    for name, c in cases.items():
        r = blocks_for(c["edges"], c["n_atoms"])
        got = {
            "sizes": sorted(int(x) for x in r["n_edges"]),
            "ranks": sorted(int(x) for x in r["rank"]),
            "artic": sorted(int(x) for x in r["n_artic"]),
        }
        want = {
            "sizes": sorted(c["sizes"]),
            "ranks": sorted(c["ranks"]),
            "artic": sorted(c["artic"]),
        }
        case_ok = got == want
        ok = ok and case_ok
        out[name] = {"got": got, "want": want, "pass": case_ok}
    return {"cases": out, "pass": ok}


# ---------------------------------------------------------------------------
# C12: checkpoint roundtrip
# ---------------------------------------------------------------------------


def gate_c12(cache: Mapping[str, Any], rel: Mapping[str, Any],
             blocks: Mapping[str, Any]) -> dict:
    norm = BC.load_norm()
    model, info = ZM.build_model(norm, "R", 0, "cpu")
    gen = torch.Generator().manual_seed(99)
    with torch.no_grad():
        model.reader_ext.weight.normal_(0.0, 0.02, generator=gen)
    rows = np.arange(32)
    batch = ZB.collate(cache, rel, blocks, rows)
    model.eval()
    with torch.no_grad():
        pred_a = model(batch).detach().clone()
    saved_hash = BC.state_hash(model.state_dict())
    tmp = Path(tempfile.gettempdir()) / "zbo_roundtrip.pt"
    torch.save({"state_dict": model.state_dict(), "info": info}, tmp)
    model2, _ = ZM.build_model(norm, "R", 0, "cpu")
    payload = torch.load(tmp, map_location="cpu", weights_only=False)
    model2.load_state_dict(payload["state_dict"])
    model2.eval()
    with torch.no_grad():
        pred_b = model2(batch).detach().clone()
    diff = max_tensor_diff(pred_a, pred_b)
    tmp.unlink(missing_ok=True)
    return {
        "prediction_max_abs_diff": diff,
        "state_hash_equal": saved_hash == BC.state_hash(payload["state_dict"]),
        "pass": diff == 0.0,
    }


# ---------------------------------------------------------------------------
# C13: provenance
# ---------------------------------------------------------------------------


def gate_c13(cache_train: Mapping[str, Any], cache_valid: Mapping[str, Any],
             norm: Mapping[str, Any]) -> dict:
    train_g = np.asarray(cache_train["global_feat"], dtype=np.float64)
    mean = train_g.mean(0)
    scale = train_g.std(0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    norm_mean = np.asarray(norm["global_mean"], dtype=np.float64)
    norm_scale = np.asarray(norm["global_scale"], dtype=np.float64)
    ssl_info = {}
    for seed in (0, 1):
        path = BC.ssl_path(seed)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        keys = sorted(payload.keys())
        ssl_info[str(seed)] = {
            "path": BC.rel_or_abs(path),
            "sha256": BC.sha256_file(path),
            "keys": keys,
            "has_side_or_compose_or_reader": any(
                k in keys for k in ("side", "compose", "reader", "reader_ext",
                                    "block_mlp", "block_norm")),
        }
    out = {
        "data_train_sha256": BC.sha256_file(BC.DATA_TRAIN),
        "data_valid_sha256": BC.sha256_file(BC.DATA_VALID),
        "norm_json_sha256": BC.sha256_file(BC.NORM_JSON),
        "norm_mean_max_abs_diff": float(np.abs(mean - norm_mean).max()),
        "norm_scale_max_abs_diff": float(np.abs(scale - norm_scale).max()),
        "norm_valid_or_test_used": bool(norm["valid_or_test_used"]),
        "split_train": str(cache_train.get("split")),
        "split_valid": str(cache_valid.get("split")),
        "n_train": int(cache_train["n_molecules"]),
        "n_valid": int(cache_valid["n_molecules"]),
        "ssl": ssl_info,
        "official_test_loaded": False,
    }
    out["pass"] = (
        out["norm_mean_max_abs_diff"] < 1e-5
        and out["norm_scale_max_abs_diff"] < 1e-5
        and not out["norm_valid_or_test_used"]
        and out["n_train"] == 10000
        and out["n_valid"] == 1000
        and all(not v["has_side_or_compose_or_reader"] for v in ssl_info.values())
    )
    return out


# ---------------------------------------------------------------------------
# C14: fixed worst-10 anchor
# ---------------------------------------------------------------------------


def gate_c14() -> dict:
    rows = BC.fixed_worst10_rows()
    labels = BC.official_valid_labels()
    path = BC.BJM_RUNS / "B_s0" / "valid_best.npz"
    if not path.exists():
        path = BC.DCC_VALID_PRED
    z = np.load(path)
    pred = np.asarray(z["pred"], dtype=np.float64)
    anchor_rows = np.asarray(z["row"], dtype=np.int64)
    ae = np.abs(pred - labels[anchor_rows])
    selected = [int(r) for r in rows]
    return {
        "anchor": BC.rel_or_abs(path),
        "rows": selected,
        "n_unique": len(set(selected)),
        "all_in_range": all(0 <= r < 1000 for r in selected),
        "anchor_mae": float(ae.mean()),
        "anchor_top10_ae": [float(ae[r]) for r in selected],
        "pass": len(selected) == 10 and len(set(selected)) == 10
        and all(0 <= r < 1000 for r in selected),
    }


# ---------------------------------------------------------------------------
# C15: group definitions (structure-only)
# ---------------------------------------------------------------------------


def gate_c15(blocks_train: Mapping[str, Any], blocks_valid: Mapping[str, Any]) -> dict:
    tr = BC.block_row_features(blocks_train)
    va = BC.block_row_features(blocks_valid)
    q = np.quantile(tr["n_blocks"], [0.25, 0.5, 0.75])
    bins = [int(x) for x in q]

    def nblock_bin(n):
        if n <= bins[0]:
            return f"le_{bins[0]}"
        if n <= bins[1]:
            return f"{bins[0] + 1}_{bins[1]}"
        if n <= bins[2]:
            return f"{bins[1] + 1}_{bins[2]}"
        return f"ge_{bins[2] + 1}"

    train_size_vocab = set(tr["block_size_combo"])
    train_sig_vocab = set(tr["block_sig_combo"])
    valid_size_novel = np.asarray([c not in train_size_vocab for c in va["block_size_combo"]])
    valid_sig_novel = np.asarray([c not in train_sig_vocab for c in va["block_sig_combo"]])

    def counts(values):
        out = {}
        for v in values:
            out[str(v)] = out.get(str(v), 0) + 1
        return out

    return {
        "n_blocks_quantiles_train": bins,
        "n_multi_bins": "0,1,2,3+",
        "valid_n_multi_counts": counts(va["n_multi_blocks"]),
        "valid_n_blocks_bin_counts": counts([nblock_bin(int(x)) for x in va["n_blocks"]]),
        "valid_block_size_combo_novel_counts": counts(valid_size_novel.astype(int)),
        "valid_block_sig_combo_novel_counts": counts(valid_sig_novel.astype(int)),
        "train_size_vocab_size": len(train_size_vocab),
        "train_sig_vocab_size": len(train_sig_vocab),
        "fraction_valid_size_combo_novel": float(valid_size_novel.mean()),
        "fraction_valid_sig_combo_novel": float(valid_sig_novel.mean()),
        "pass": True,
        "note": ("Group definitions are structure-only. They are descriptive; "
                 "no decision gate uses the novelty subgroup."),
    }


# ---------------------------------------------------------------------------
# config + main
# ---------------------------------------------------------------------------


def write_config(checks: Mapping[str, Any]) -> dict:
    import subprocess

    prep = BC.read_json(BC.ANALYSIS / "prep.json")
    gate_path = BC.ANALYSIS / "cpu_checks.json"
    gate_sha = BC.sha256_file(gate_path)
    code_hashes = {p.name: BC.sha256_file(p) for p in sorted((HERE).glob("*.py"))}
    try:
        git_head = subprocess.check_output(
            ["git", "-C", str(BC.REPO), "rev-parse", "HEAD"], text=True).strip()
        git_dirty = bool(subprocess.check_output(
            ["git", "-C", str(BC.REPO), "status", "--porcelain"], text=True).strip())
    except Exception:
        git_head, git_dirty = None, None

    failed = [k for k, v in checks.items()
              if isinstance(v, dict) and v.get("pass") is False]
    config = {
        "protocol_id": "zinc-block-ownership-v1",
        "frozen_utc": BC.now(),
        "frozen_before_commit": True,
        "git_head_at_freeze": git_head,
        "git_dirty_at_freeze": git_dirty,
        "design": "tracks/structural-sufficiency/zinc_block_ownership_v1/DESIGN.md",
        "protocol": "tracks/structural-sufficiency/zinc_block_ownership_v1/PROTOCOL.md",
        "code_sha256": code_hashes,
        "model": {
            "base": "BOSI-B (arm B relation statistics, unchanged)",
            "block_input_dim": BC.BLOCK_IN,
            "block_hidden": BC.BLOCK_HIDDEN,
            "block_out": BC.BLOCK_OUT,
            "reader_old_in": BC.READER_OLD_IN,
            "reader_new_in": BC.READER_NEW_IN,
            "expected_parameters": BC.EXPECTED_PARAMETERS,
            "parameter_cap": BC.PARAM_CAP,
            "reader_ext_zero_init": True,
            "implementation": ("old reader first layer + zero-init Linear(32,256,bias=False); "
                               "algebraically a 449-column first layer with zero new columns"),
            "block_content": {
                "R": "true biconnected-block sums; bridge is a one-edge block",
                "N": "global graph mean-field, allocated by n_edges(B)/E[G]",
                "S": "true content on singleton blocks; multi blocks share the MULTI mean by edge count",
            },
        },
        "recipe": BC.RECIPE,
        "data": {
            "train": prep["train"],
            "valid": prep["valid"],
            "norm": {
                "path": BC.rel_or_abs(BC.NORM_JSON),
                "sha256": BC.sha256_file(BC.NORM_JSON),
                "valid_or_test_used": False,
            },
            "ssl": {
                str(seed): {
                    "path": BC.rel_or_abs(BC.ssl_path(seed)),
                    "sha256": BC.sha256_file(BC.ssl_path(seed)),
                } for seed in (0, 1)
            },
            "official_test_loaded": False,
        },
        "cpu_gates": {
            "path": "tracks/structural-sufficiency/zinc_block_ownership_v1/analysis/cpu_checks.json",
            "sha256": gate_sha,
            "all_pass": bool(checks["all_pass"]),
            "failed_checks": failed,
        },
        "phase0_observed": {
            "fraction_molecules_ge2_multi": checks["C5"]["fraction_molecules_ge2_multi"],
            "rs_first_relative_diff": checks["C5"]["rs_first_relative_diff"],
            "rs_second_relative_diff": checks["C5"]["rs_second_relative_diff"],
            "n_blocks_train": checks["C1_train"]["n_blocks_ours"],
            "n_blocks_valid": checks["C1_valid"]["n_blocks_ours"],
            "n_affected_blocks": checks["C5"]["n_affected_blocks"],
        },
        "stop_gates": {
            "min_fraction_molecules_ge2_multi": 0.20,
            "min_rs_fraction_gt_1pct": 0.50,
            "min_rs_mean_relative_diff": 0.01,
            "gpu_authorized": bool(checks["all_pass"]),
            "reason_if_stopped": checks.get("stop_reason"),
        },
        "fixed_worst10_rows": checks["C14"]["rows"],
        "fixed_worst10_rows_hash": BC.rows_hash(checks["C14"]["rows"]),
        "fixed_worst10_anchor": checks["C14"]["anchor"],
        "fixed_worst10_anchor_mae": checks["C14"]["anchor_mae"],
        "group_definitions": {
            "n_blocks_quantiles_train": checks["C15"]["n_blocks_quantiles_train"],
            "n_multi_bins": [0, 1, 2, 3],
            "block_size_combo_novel": "valid molecule's sorted n_edges tuple absent from train",
            "block_sig_combo_novel": ("valid molecule's sorted "
                                      "(n_edges,n_atoms,rank,n_artic) tuple absent from train"),
            "support_tier": "radius-2 typed signature support: all_supported / low_support / any_unseen",
            "ordinary_tail": "ordinary y > -3; tail y <= -3",
            "rings_ge4": "molecule cycle rank >= 4",
            "penalized_proxy": "max_ring_size >= 7",
        },
        "run_plan": {
            "batch1": ["R_s0", "N_s0"],
            "batch2_iff_seed0_pass": ["R_s1", "N_s1"],
            "batch3_iff_both_seeds_positive": ["S_s0", "S_s1"],
            "rule": ("No width/depth/lr/wd/normalisation/block-type change after any result. "
                     "Each seed uses its own SSL checkpoint; no supervised warm start."),
        },
        "seed0_gate": {
            "d_best_min": 0.005,
            "R_best_max": 0.082,
            "d_W20_min": -0.002,
            "d_T100_min": -0.002,
            "at_least_one_of_W20_T100_positive": True,
            "ordinary_paired_dMAE_min": 0.0,
            "fixed_worst10_removed_dMAE_strictly_positive": True,
            "single_row_contribution_max_fraction_of_total": 0.50,
            "mechanism_active": True,
        },
        "two_seed_gate": {
            "both_d_best_min": 0.005,
            "same_direction": True,
        },
        "outcome_labels": [
            "OWNERSHIP_NOT_USEFUL",
            "SINGLETON_NONLINEARITY_ONLY",
            "STRUCTURAL_GAIN_BUT_NOT_COMPETITIVE",
            "PROMISING_COMPOSITION_MECHANISM",
            "UNRESOLVED_DUE_TO_VARIANCE",
            "MECHANISM_INACTIVE",
        ],
        "decision_thresholds": {
            "R_S_two_seed_mean_min": 0.003,
            "R_seed_mean_competitive_max": 0.077,
            "DCCL_seed_mean_valid": 0.075990,
            "DCCL_one_shot_test": 0.067319,
            "official_goal_test": 0.060,
        },
        "claim_limits": [
            "N loses ownership information: R-N is not a strict same-information contrast.",
            "R better than N but not better than S does not support block-level compositional generalisation.",
            "No recursive MP, learned junction token, attention, distillation, or test tuning.",
            "Two seeds share the same two SSL checkpoints; they are not an independent SSL family.",
            "PROMISING is a follow-up signal, not a confirmation or a competitive claim.",
            "The official test split is never read in this round.",
        ],
        "gpu_budget_hours": {
            "soft": BC.GPU_SOFT_BUDGET_H,
            "hard": BC.GPU_HARD_BUDGET_H,
            "estimated_formal": "3.0-3.5 for six trajectories plus smoke",
        },
        "analysis_products": [
            "best-valid / W20 / T100 / final-valid / train@best / train@final",
            "paired dMAE with bootstrap CI",
            "ordinary / tail / radius-2 support groups",
            "rings>=4, median/P90 AE",
            "fixed worst-10 and drop-top-10",
            "drop-row172",
            "block-count / novelty groups",
            "parameters, GPU time, initial-state hash, commit, checkpoint hash",
        ],
        "forbidden": [
            "test split access", "test-tuned hyperparameters", "recursive message passing",
            "learned junction token", "attention", "distillation", "ensembling",
            "multi-seed averaging at inference", "block-type change after results",
        ],
    }
    BC.CONFIGS.mkdir(parents=True, exist_ok=True)
    BC.write_json(BC.CONFIG, jsonable(config))
    return config


def main() -> int:
    t0 = time.perf_counter()
    BC.log(f"[checks] round={BC.ROUND_ID} cpu-only")
    train = BC.load_cache("train")
    valid = BC.load_cache("valid")
    rel_train = BC.load_rel("train")
    rel_valid = BC.load_rel("valid")
    blocks_train = BC.load_blocks("train")
    blocks_valid = BC.load_blocks("valid")
    norm = BC.load_norm()

    checks: dict[str, Any] = {}
    checks["C1_train"] = gate_c1(train, "train")
    BC.log(f"[C1] train {checks['C1_train']['pass']}")
    checks["C1_valid"] = gate_c1(valid, "valid")
    BC.log(f"[C1] valid {checks['C1_valid']['pass']}")
    checks["C2"] = gate_c2(train, blocks_train)
    BC.log(f"[C2] invariance {checks['C2']['pass']}")
    checks["C3"] = gate_c3(train, rel_train, blocks_train)
    BC.log(f"[C3] conservation {checks['C3']['pass']}")
    checks["C4"] = gate_c4()
    BC.log(f"[C4] counterexample {checks['C4']['pass']}")
    checks["C5"] = gate_c5(train, rel_train, blocks_train)
    BC.log(f"[C5] coverage {checks['C5']['pass']}")
    checks["C6"] = gate_c6()
    BC.log(f"[C6] historical contrast {checks['C6']['pass']}")
    checks["C7"] = gate_c7(train, rel_train, blocks_train)
    BC.log(f"[C7] parameters/init/forward {checks['C7']['pass']}")
    checks["C8"] = gate_c8(train, rel_train, blocks_train)
    BC.log(f"[C8] gradients {checks['C8']['pass']}")
    checks["C9"] = gate_c9(train, valid, rel_valid, blocks_valid)
    BC.log(f"[C9] label isolation {checks['C9']['pass']}")
    checks["C10"] = gate_c10(train, valid, blocks_train, blocks_valid,
                             rel_train, rel_valid)
    BC.log(f"[C10] cache reproducibility {checks['C10']['pass']}")
    checks["C11"] = gate_c11()
    BC.log(f"[C11] synthetic small graphs {checks['C11']['pass']}")
    checks["C12"] = gate_c12(valid, rel_valid, blocks_valid)
    BC.log(f"[C12] checkpoint roundtrip {checks['C12']['pass']}")
    checks["C13"] = gate_c13(train, valid, norm)
    BC.log(f"[C13] provenance {checks['C13']['pass']}")
    checks["C14"] = gate_c14()
    BC.log(f"[C14] fixed worst-10 {checks['C14']['pass']}")
    checks["C15"] = gate_c15(blocks_train, blocks_valid)
    BC.log(f"[C15] groups {checks['C15']['pass']}")

    gate_keys = [k for k in checks if k.startswith("C")]
    checks["all_pass"] = all(bool(checks[k].get("pass")) for k in gate_keys)
    if not checks["C5"]["stop_gate_structural_pass"]:
        checks["stop_reason"] = "structural_coverage_below_20pct"
    elif not checks["C5"]["stop_gate_numeric_pass"]:
        checks["stop_reason"] = "rs_input_difference_too_weak"
    elif not checks["all_pass"]:
        checks["stop_reason"] = "cpu_gate_failed:" + ",".join(
            k for k in gate_keys if not bool(checks[k].get("pass")))
    else:
        checks["stop_reason"] = None
    checks["seconds_total"] = time.perf_counter() - t0

    BC.ANALYSIS.mkdir(parents=True, exist_ok=True)
    BC.write_json(BC.ANALYSIS / "cpu_checks.json", jsonable(checks))
    config = write_config(checks)
    BC.log(f"[checks] all_pass={checks['all_pass']} "
           f"gpu_authorized={config['stop_gates']['gpu_authorized']} "
           f"reason={checks['stop_reason']} total={checks['seconds_total']:.1f}s")
    if not checks["all_pass"]:
        BC.log("[checks] GPU is NOT authorized; see analysis/cpu_checks.json")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
