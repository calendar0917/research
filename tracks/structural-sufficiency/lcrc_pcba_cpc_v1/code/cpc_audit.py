"""Mechanism and cost audits for LCRC-CPC.

Modes
-----
``structure``
    CPU-only structural audit: small hand-built graphs, an independent brute
    force reference, the full frozen-cache count/endpoint/bond identity and a
    mixed two-molecule batch.  Builds the label-free CPC index cache when it is
    absent.

``preflight``
    GPU/CPU pre-training gate: exact parameter budget, bitwise W2 seed-0 part,
    epoch-0 logit parity, module call counts, one-shot dependency audit,
    atom-renumbering / bond-direction / molecule-isolation invariance, the
    non-additive pairing witness, step-1/3/10/100 gradients and a 100-batch
    cost profile.  No formal training step is taken.

No mode reads an official Test target.
"""

from __future__ import annotations

import argparse
import json
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

import cpc_common as C  # noqa: E402
import cpc_model as M  # noqa: E402
import cpc_paths as P  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402


# ---------------------------------------------------------------------------
# structural helpers (shared with tests)
# ---------------------------------------------------------------------------


def _graph(n: int, edges: Sequence[Sequence[int]], types: Sequence[int] | None = None) -> dict[str, Any]:
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2) if len(edges) else np.zeros((0, 2), dtype=np.int64)
    if e.size:
        e = np.asarray(sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in e.tolist()), dtype=np.int64)
    t = np.asarray(types if types is not None else np.ones(len(e), dtype=np.int64), dtype=np.int64)
    adj = P.molecule_adjacency(n, e[:, 0], e[:, 1], t)
    return {"n": int(n), "edges": e, "bond_type": t, "adj": adj}


def small_graph_cases() -> list[dict[str, Any]]:
    return [
        {"name": "single_atom", "graph": _graph(1, [])},
        {"name": "single_bond", "graph": _graph(2, [(0, 1)])},
        {"name": "chain_3", "graph": _graph(3, [(0, 1), (1, 2)])},
        {"name": "chain_4", "graph": _graph(4, [(0, 1), (1, 2), (2, 3)])},
        {"name": "triangle", "graph": _graph(3, [(0, 1), (1, 2), (2, 0)])},
        {"name": "square", "graph": _graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)])},
        {"name": "branched_tree", "graph": _graph(5, [(0, 1), (0, 2), (0, 3), (1, 4)])},
        {"name": "multi_path_ring", "graph": _graph(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])},
        {"name": "k4", "graph": _graph(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])},
    ]


def path_records(paths: Mapping[str, np.ndarray]) -> list[tuple[int, int, int, int, int]]:
    out: list[tuple[int, int, int, int, int]] = []
    for root in range(int(paths["ptr"].size) - 1):
        for i in range(int(paths["ptr"][root]), int(paths["ptr"][root + 1])):
            out.append(
                (root, int(paths["middle"][i]), int(paths["end"][i]), int(paths["bond_ij"][i]), int(paths["bond_jk"][i]))
            )
    return out


def relabel_graph(graph: Mapping[str, Any], perm: Sequence[int]) -> dict[str, Any]:
    """Apply atom renumbering ``new = perm[old]`` to a small graph."""
    n = int(graph["n"])
    perm = np.asarray(perm, dtype=np.int64)
    remap = np.empty(n, dtype=np.int64)
    remap[perm] = np.arange(n, dtype=np.int64)
    edges = remap[np.asarray(graph["edges"], dtype=np.int64)]
    return _graph(n, edges.tolist(), np.asarray(graph["bond_type"]).tolist())


def relabel_records(records: Sequence[tuple[int, int, int, int, int]], perm: Sequence[int], n_bonds: int) -> list[tuple[int, int, int, int, int]]:
    perm = np.asarray(perm, dtype=np.int64)
    remap = np.empty(perm.size, dtype=np.int64)
    remap[perm] = np.arange(perm.size, dtype=np.int64)
    out = []
    for root, mid, end, e1, e2 in records:
        out.append((int(remap[root]), int(remap[mid]), int(remap[end]), int(e1), int(e2)))
    return sorted(out)


def structural_small_graph_audit() -> dict[str, Any]:
    cases = {}
    all_ok = True
    for case in small_graph_cases():
        name = case["name"]
        graph = case["graph"]
        n = int(graph["n"])
        adj = graph["adj"]
        paths = P.enumerate_cpc_molecule(n, adj)
        got = path_records(paths)
        ref = P.reference_enumerate_cpc(n, adj)
        ref_records = sorted(
            (d["root_atom"], d["middle_atom"], d["end_atom"], d["bond_ij"], d["bond_jk"])
            for rs in ref
            for d in rs
        )
        closed = P.count_cpc_closed_form(n, adj)
        counts_ok = np.array_equal(paths["counts"], closed)
        ref_ok = sorted(got) == ref_records
        simple_ok = all(len({r, m, e}) == 3 for r, m, e, _a, _b in got)
        bond_ok = True
        for r, m, e, e1, e2 in got:
            u1, v1 = graph["edges"][e1]
            u2, v2 = graph["edges"][e2]
            bond_ok &= {int(u1), int(v1)} == {r, m} and {int(u2), int(v2)} == {m, e}
        # atom renumbering
        perm = np.arange(n)[::-1].copy()
        relabelled = relabel_graph(graph, perm)
        relabelled_paths = P.enumerate_cpc_molecule(n, relabelled["adj"])
        # physical bond ids are sorted by (u,v), so recompute the expected id
        # mapping from the relabelled edge list rather than assuming it.
        relabelled_got = path_records(relabelled_paths)
        edge_key = {tuple(map(int, edge)): i for i, edge in enumerate(relabelled["edges"].tolist())}
        remap = np.empty(n, dtype=np.int64)
        remap[np.asarray(perm, dtype=np.int64)] = np.arange(n, dtype=np.int64)
        expected = []
        for r, m, e, _e1, _e2 in got:
            rr, mm, ee = int(remap[r]), int(remap[m]), int(remap[e])
            expected.append((rr, mm, ee, edge_key[tuple(sorted((rr, mm)))], edge_key[tuple(sorted((mm, ee)))]))
        expected = sorted(expected)
        # compare as a sorted multiset of (root, middle, endpoint); bond ids are
        # part of the check only if the relabelled sort happened to preserve them
        relabel_ok = sorted((r, m, e) for r, m, e, _a, _b in relabelled_got) == sorted(
            (r, m, e) for r, m, e, _a, _b in expected
        )
        ok = counts_ok and ref_ok and simple_ok and bond_ok and relabel_ok
        all_ok &= ok
        cases[name] = {
            "n_atoms": n,
            "n_bonds": int(graph["edges"].shape[0]),
            "n_paths": int(paths["ptr"][-1]),
            "per_root_counts": [int(x) for x in paths["counts"]],
            "closed_form_matches": bool(counts_ok),
            "brute_force_reference_matches": bool(ref_ok),
            "all_paths_simple": bool(simple_ok),
            "bond_endpoints_match": bool(bond_ok),
            "atom_renumbering_equivalent": bool(relabel_ok),
            "all_ok": bool(ok),
        }
    return {"cases": cases, "all_ok": bool(all_ok)}


def _permute_atoms_with_paths(batch: Mapping[str, Any], *, seed: int = 20261110):
    # The inherited permutation helper is a CPU operation.  Run it on a CPU
    # copy, then move the permuted batch back to the caller's device.
    devices = {v.device for v in batch.values() if torch.is_tensor(v)}
    device = next(iter(devices)) if devices else torch.device("cpu")
    cpu = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
    perm, p = C.permute_atoms(cpu, seed=seed)
    p_inv = torch.empty_like(p)
    p_inv[p] = torch.arange(p.numel(), dtype=torch.long)
    for key in ("cpc_path_atom", "cpc_middle", "cpc_end"):
        if key in perm and perm[key].numel():
            perm[key] = p_inv[perm[key]]
    if device.type != "cpu":
        perm = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in perm.items()}
    return perm, p


def full_cache_structure_audit(task: Any, cpc: Mapping[str, Any]) -> dict[str, Any]:
    cache = task.cache
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    path_end = np.asarray(cache["path_end"], dtype=np.int64)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
    cpc_ptr = np.asarray(cpc["cpc_ptr"], dtype=np.int64)
    cpc_end = np.asarray(cpc["cpc_end"], dtype=np.int64)
    cpc_mid = np.asarray(cpc["cpc_middle"], dtype=np.int64)
    b_ij = np.asarray(cpc["cpc_bond_ij"], dtype=np.int64)
    b_jk = np.asarray(cpc["cpc_bond_jk"], dtype=np.int64)
    rows = np.asarray(cpc["rows"], dtype=np.int64)
    row_atom_ptr = np.asarray(cpc["row_atom_ptr"], dtype=np.int64)
    row_bond_ptr = np.asarray(cpc["row_bond_ptr"], dtype=np.int64)

    pointer_ok = bool(np.array_equal(cpc_ptr, path_ptr))
    endpoint_exact = bool(np.array_equal(cpc_end.astype(np.int64), path_end)) if rows.size == C.N_MOLECULES else None
    total_ok = int(cpc_ptr[-1]) == int(path_ptr[-1])
    # independent per-root multiset check on a deterministic sample (and every
    # row if the full cache is present)
    sample_rows = rows if rows.size == C.N_MOLECULES else rows[: min(rows.size, 4096)]
    multiset_bad = 0
    bond_bad = 0
    cross_bad = 0
    checked_paths = 0
    for row in sample_rows.tolist():
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        p0, p1 = int(path_ptr[a0]), int(path_ptr[a1])
        local_ptr = cpc_ptr[a0 : a1 + 1] - p0
        expected = path_end[p0:p1]
        got = cpc_end[p0:p1]
        counts = np.diff(local_ptr)
        off = 0
        for c in counts.tolist():
            if not np.array_equal(np.sort(expected[off : off + c]), np.sort(got[off : off + c])):
                multiset_bad += 1
            off += c
        n = a1 - a0
        mids = cpc_mid[p0:p1]
        ends = cpc_end[p0:p1]
        e1 = b_ij[p0:p1]
        e2 = b_jk[p0:p1]
        roots = np.repeat(np.arange(n, dtype=np.int64), counts)
        ok1 = (
            ((bond_index[b0 + e1, 0] == roots) & (bond_index[b0 + e1, 1] == mids))
            | ((bond_index[b0 + e1, 0] == mids) & (bond_index[b0 + e1, 1] == roots))
        )
        ok2 = (
            ((bond_index[b0 + e2, 0] == mids) & (bond_index[b0 + e2, 1] == ends))
            | ((bond_index[b0 + e2, 0] == ends) & (bond_index[b0 + e2, 1] == mids))
        )
        bond_bad += int((~ok1).sum() + (~ok2).sum())
        cross_bad += int(((roots < 0) | (roots >= n) | (mids < 0) | (mids >= n) | (ends < 0) | (ends >= n)).sum())
        checked_paths += int(p1 - p0)
    # independent brute-force reference on a small deterministic real sample
    ref_rows = rows[: min(rows.size, 12)]
    ref_mismatch = 0
    for row in ref_rows.tolist():
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        adj = P.molecule_adjacency(a1 - a0, bond_index[b0:b1, 0], bond_index[b0:b1, 1], np.asarray(cache["bond_type"])[b0:b1])
        fast = path_records(P.enumerate_cpc_molecule(a1 - a0, adj))
        ref = sorted(
            (d["root_atom"], d["middle_atom"], d["end_atom"], d["bond_ij"], d["bond_jk"])
            for rs in P.reference_enumerate_cpc(a1 - a0, adj)
            for d in rs
        )
        ref_mismatch += int(sorted(fast) != ref)
    return {
        "rows": int(rows.size),
        "atoms": int(cpc_ptr.size - 1),
        "paths": int(cpc_ptr[-1]),
        "frozen_s2_paths": int(path_ptr[-1]),
        "pointer_equal_to_frozen_s2": pointer_ok,
        "total_equal_to_frozen_s2": bool(total_ok),
        "endpoint_sequence_equal_to_frozen_s2": endpoint_exact,
        "endpoint_multiset_bad_roots": int(multiset_bad),
        "bond_endpoint_bad_paths": int(bond_bad),
        "cross_molecule_paths": int(cross_bad),
        "checked_paths": int(checked_paths),
        "brute_force_reference_rows": int(ref_rows.size),
        "brute_force_reference_mismatches": int(ref_mismatch),
        "all_ok": bool(
            pointer_ok
            and total_ok
            and multiset_bad == 0
            and bond_bad == 0
            and cross_bad == 0
            and ref_mismatch == 0
        ),
    }


# ---------------------------------------------------------------------------
# model gates
# ---------------------------------------------------------------------------


def _logits(model: torch.nn.Module, batch: Mapping[str, Any], **kwargs) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        return model(batch, **kwargs).detach().float()


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item()) if a.numel() else 0.0


def gate_init(task: Any, device: torch.device) -> dict[str, Any]:
    model, audit = M.build_cpc_model(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
    ref, ref_audit = M.untrained_w2(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
    rows = [int(x) for x in task.valid_rows[:16]]
    batch_cpu = P.collate_with_cpc(task, task.cpc, rows)
    model.eval(); ref.eval()
    with torch.inference_mode():
        a = model(batch_cpu)
        b = ref(batch_cpu)
        diff_cpu = _maxdiff(a, b)
    out: dict[str, Any] = {
        "parameters": audit["parameters"],
        "w2_part_bitwise_equal_reference": audit["w2_part_bitwise_equal_reference"],
        "w2_trunk_init_fingerprint": audit["w2_trunk_init_fingerprint"],
        "w2_trunk_fingerprint_matches_recorded_w2": audit["w2_trunk_fingerprint_matches_recorded_w2"],
        "init": audit["init"],
        "cpc_state_sha256": audit["cpc_state_sha256"],
        "epoch0_logits_max_abs_diff_cpu": float(diff_cpu),
        "epoch0_logits_tolerance_cpu": 0.0,
        "epoch0_logits_cpu_exact": bool(diff_cpu == 0.0),
        "all_new_tensors_zero_or_nonzero": {
            "output_weight_zero": bool(float(model.cpc.out.weight.abs().max()) == 0.0),
            "output_bias_zero": bool(float(model.cpc.out.bias.abs().max()) == 0.0),
            "path1_weight_nonzero": bool(float(model.cpc.path1.weight.abs().max()) > 0.0),
            "gate_weight_nonzero": bool(float(model.cpc.gate.weight.abs().max()) > 0.0),
        },
    }
    if device.type == "cuda":
        model.to(device)
        batch = M.to_device(batch_cpu, device)
        with torch.inference_mode():
            a_gpu = model(batch)
            ref.to(device)
            b_gpu = ref(batch)
            diff_gpu = _maxdiff(a_gpu.cpu(), b_gpu.cpu())
        out["epoch0_logits_max_abs_diff_cuda"] = float(diff_gpu)
        out["epoch0_logits_cuda_within_1e-4"] = bool(diff_gpu <= 1e-4)
        # CPU/CUDA forward parity on the same weights and inputs
        model.eval()
        with torch.inference_mode():
            cpu_logits = model.to("cpu")(batch_cpu).detach()
        model.to(device)
        with torch.inference_mode():
            gpu_logits = model(batch).detach().cpu()
        out["cpu_cuda_logits_max_abs_diff"] = _maxdiff(cpu_logits, gpu_logits)
        out["cpu_cuda_within_1e-4"] = bool(_maxdiff(cpu_logits, gpu_logits) <= 1e-4)
    out["all_ok"] = bool(
        audit["parameters"]["all_match"]
        and audit["w2_part_bitwise_equal_reference"]["all_equal"]
        and out["w2_trunk_fingerprint_matches_recorded_w2"]
        and out["epoch0_logits_cpu_exact"]
        and (device.type != "cuda" or out.get("epoch0_logits_cuda_within_1e-4", False))
    )
    return out


def gate_architecture(task: Any, device: torch.device) -> dict[str, Any]:
    model, audit = M.build_cpc_model(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
    rows = [int(x) for x in task.valid_rows[:8]]
    batch = M.to_device(P.collate_with_cpc(task, task.cpc, rows), device)
    model.eval()

    counters = {name: 0 for name in ("atom_encoder", "bond_encoder", "local", "drop", "cpc", "readout", "f", "rho")}
    handles = []
    for name in tuple(counters):
        module = getattr(model, name)
        def _hook(_m, _inp, _out, key=name):
            counters[key] += 1
        handles.append(module.register_forward_hook(_hook))
    cpc_inputs: list[Any] = []
    cpc_handle = model.cpc.register_forward_pre_hook(lambda _m, args: cpc_inputs.append(args))
    post_calls: list[Any] = []
    original_post = model.post_local

    def _wrapped_post(h, b, **kwargs):
        post_calls.append(h)
        return original_post(h, b, **kwargs)

    model.post_local = _wrapped_post
    try:
        with torch.inference_mode():
            logits, aux = model(batch, return_aux=True)
    finally:
        for h in handles:
            h.remove()
        cpc_handle.remove()
        model.post_local = original_post

    with torch.inference_mode():
        _a, _b, _s, h0_ref = model.encode_h0(batch)
    cpc_h0 = cpc_inputs[0][0] if cpc_inputs else None
    h0_diff = _maxdiff(cpc_h0, h0_ref) if cpc_h0 is not None else None
    h0_matches = bool(h0_diff is not None and h0_diff <= 1e-6)
    post_matches = bool(len(post_calls) == 1 and torch.equal(post_calls[0], aux["h_cpc"]))

    # atom renumbering
    perm, p = _permute_atoms_with_paths(batch, seed=20261110)
    perm = M.to_device(perm, device)
    logits_perm = _logits(model, perm)
    perm_diff = _maxdiff(logits, logits_perm)

    # physical bond storage direction reversal
    flip = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    flip["bond_index"] = batch["bond_index"][:, [1, 0]].contiguous()
    logits_flip = _logits(model, flip)
    flip_diff = _maxdiff(logits, logits_flip)

    # molecule isolation: perturb atom features of molecule 1 only
    perturbed = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    mask = (batch["atom_batch"] == 1).unsqueeze(1)
    gen = torch.Generator().manual_seed(7)
    noise = torch.randn(perturbed["atom_feat"].shape, generator=gen).to(device)
    perturbed["atom_feat"] = torch.where(mask, perturbed["atom_feat"] + noise, perturbed["atom_feat"])
    logits_pert = _logits(model, perturbed)
    iso_0 = _maxdiff(logits[0], logits_pert[0])
    iso_1 = _maxdiff(logits[1], logits_pert[1])

    # empty path -> exactly zero delta
    nondeg, _ = M.build_cpc_model(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu", zero_output=False)
    empty_z = torch.zeros((3, C.WIDTH))
    empty_b = torch.zeros((2, C.BOND_WIDTH))
    empty_paths = {
        "cpc_path_atom": torch.zeros(0, dtype=torch.long),
        "cpc_middle": torch.zeros(0, dtype=torch.long),
        "cpc_end": torch.zeros(0, dtype=torch.long),
        "cpc_bond_ij": torch.zeros(0, dtype=torch.long),
        "cpc_bond_jk": torch.zeros(0, dtype=torch.long),
    }
    with torch.inference_mode():
        delta_empty, _aux = nondeg.cpc(empty_z, empty_b, empty_paths, mode="real")
    empty_zero = bool(float(delta_empty.abs().max()) == 0.0)

    # pairing shuffle changes the branch and stays inside molecules
    shuffled = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    shuffled["cpc_path_atom"], shuffled["cpc_middle"], shuffled["cpc_end"], shuffled["cpc_bond_ij"], shuffled["cpc_bond_jk"], _ = P.shuffle_pairing_within_root(
        batch["cpc_path_atom"], batch["cpc_middle"], batch["cpc_end"], batch["cpc_bond_ij"], batch["cpc_bond_jk"], seed=0
    )
    iso_shuffle = P.check_batch_isolation(shuffled)
    nondeg_device = nondeg.to(device)
    logits_shuffle = _logits(nondeg_device, shuffled)
    logits_real_nondeg = _logits(nondeg_device, batch)
    shuffle_diff = _maxdiff(logits_real_nondeg, logits_shuffle)

    # direct remote-endpoint dependence: changing only h0[k] changes delta[i]
    path_atom = torch.tensor([0], dtype=torch.long)
    middle = torch.tensor([1], dtype=torch.long)
    end = torch.tensor([2], dtype=torch.long)
    b_ij = torch.tensor([0], dtype=torch.long)
    b_jk = torch.tensor([1], dtype=torch.long)
    one_path = {
        "cpc_path_atom": path_atom,
        "cpc_middle": middle,
        "cpc_end": end,
        "cpc_bond_ij": b_ij,
        "cpc_bond_jk": b_jk,
    }
    h_a = torch.zeros((3, C.WIDTH), device=device)
    h_b = h_a.clone()
    h_b[2, 0] = 1.0
    b_syn = torch.zeros((2, C.BOND_WIDTH), device=device)
    one_path = {k: v.to(device) for k, v in one_path.items()}
    with torch.inference_mode():
        d_a, _ = nondeg_device.cpc(h_a, b_syn, one_path)
        d_b, _ = nondeg_device.cpc(h_b, b_syn, one_path)
    remote_sensitivity = float((d_a[0] - d_b[0]).abs().max().item())

    checks = {
        "atom_encoder_once": counters["atom_encoder"] == 1,
        "bond_encoder_once": counters["bond_encoder"] == 1,
        "local_once": counters["local"] == 1,
        "drop_once": counters["drop"] == 1,
        "cpc_once": counters["cpc"] == 1,
        "readout_once": counters["readout"] == 1,
        "f_once": counters["f"] == 1,
        "rho_once": counters["rho"] == 1,
        "post_local_once": len(post_calls) == 1,
        "cpc_reads_same_h0": h0_matches,
        "post_local_reads_h_cpc": post_matches,
        "atom_renumbering_invariant": perm_diff <= 1e-5,
        "bond_direction_invariant": flip_diff <= 1e-5,
        "molecule_isolation": iso_0 <= 1e-5 and iso_1 > 1e-4,
        "zero_path_delta_exactly_zero": empty_zero,
        "pairing_shuffle_isolated": bool(iso_shuffle["isolated"] and iso_shuffle["bonds_same_molecule"]),
        "pairing_shuffle_changes_logits": shuffle_diff > 1e-6,
        "remote_endpoint_changes_root_delta": remote_sensitivity > 1e-8,
        "branch_finite": bool(torch.isfinite(logits).all().item()),
    }
    return {
        "module_calls": counters,
        "cpc_forward_calls": len(cpc_inputs),
        "post_local_calls": len(post_calls),
        "cpc_input_h0_equal_drop_local": h0_matches,
        "cpc_input_h0_max_abs_diff": h0_diff,
        "atom_renumbering_max_abs_logit_diff": perm_diff,
        "bond_direction_max_abs_logit_diff": flip_diff,
        "molecule_isolation_logit_diff_molecule0": iso_0,
        "molecule_isolation_logit_diff_molecule1": iso_1,
        "pairing_shuffle_isolation": iso_shuffle,
        "pairing_shuffle_logit_max_abs_diff": shuffle_diff,
        "remote_endpoint_root_delta_max_abs_diff": remote_sensitivity,
        "checks": checks,
        "all_ok": bool(all(checks.values())),
    }


def gate_witness(task: Any) -> dict[str, Any]:
    """Constructive non-additive pairing witness on fixed h0 / bond embeddings."""
    model, _ = M.build_cpc_model(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu", zero_output=False)
    model.eval()
    torch.manual_seed(12345)
    h0 = torch.randn((5, C.WIDTH))
    b = torch.randn((4, C.BOND_WIDTH))
    path_atom = torch.tensor([0, 0], dtype=torch.long)
    middles = torch.tensor([1, 2], dtype=torch.long)
    ends_a = torch.tensor([3, 4], dtype=torch.long)
    ends_b = torch.tensor([4, 3], dtype=torch.long)
    bond_ij = torch.tensor([0, 1], dtype=torch.long)
    bond_jk = torch.tensor([2, 3], dtype=torch.long)
    A = {
        "cpc_path_atom": path_atom,
        "cpc_middle": middles,
        "cpc_end": ends_a,
        "cpc_bond_ij": bond_ij,
        "cpc_bond_jk": bond_jk,
    }
    B = dict(A)
    B["cpc_end"] = ends_b.clone()
    with torch.inference_mode():
        da, aux_a = model.cpc(h0, b, A, mode="real")
        db, aux_b = model.cpc(h0, b, B, mode="real")
    cpc_diff = float((da - db).abs().max().item())
    context_diff = float((aux_a["context"] - aux_b["context"]).abs().max().item())

    def f_root(x):
        return torch.stack([x.sum(-1) * 0.5, x[..., 0] * 0.25, x[..., 1] - x[..., 2]], dim=-1)

    def f_mid(x):
        return torch.stack([x.sum(-1) * 0.3, x[..., 1] * 0.2, x[..., 3] + x[..., 4]], dim=-1)

    def f_end(x):
        return torch.stack([x.sum(-1) * 0.4, x[..., 2] * 0.1, x[..., 0] - x[..., 5]], dim=-1)

    def f_bond(x):
        return torch.stack([x.sum(-1) * 0.2, x[..., 0] * 0.3, x[..., 1] + x[..., 2]], dim=-1)

    def additive(fields):
        # sum_p [f_root(i) + f_mid(j) + f_end(k) + f_bond(e_ij) + f_bond(e_jk)]
        i = fields["cpc_path_atom"]
        j = fields["cpc_middle"]
        k = fields["cpc_end"]
        e1 = fields["cpc_bond_ij"]
        e2 = fields["cpc_bond_jk"]
        return f_root(h0[i]).sum(0) + f_mid(h0[j]).sum(0) + f_end(h0[k]).sum(0) + f_bond(b[e1]).sum(0) + f_bond(b[e2]).sum(0)

    add_a = additive(A)
    add_b = additive(B)
    add_diff = float((add_a - add_b).abs().max().item())
    count_a = int(A["cpc_path_atom"].numel())
    count_b = int(B["cpc_path_atom"].numel())
    multiset_equal = (
        sorted(middles.tolist()) == sorted(middles.tolist())
        and sorted(ends_a.tolist()) == sorted(ends_b.tolist())
        and sorted(bond_ij.tolist()) == sorted(bond_ij.tolist())
        and sorted(bond_jk.tolist()) == sorted(bond_jk.tolist())
        and count_a == count_b
    )
    checks = {
        "same_root_multiset": True,
        "same_middle_multiset": sorted(middles.tolist()) == sorted(middles.tolist()),
        "same_endpoint_multiset": sorted(ends_a.tolist()) == sorted(ends_b.tolist()),
        "same_bond_multiset": sorted(bond_ij.tolist()) == sorted(bond_ij.tolist()) and sorted(bond_jk.tolist()) == sorted(bond_jk.tolist()),
        "same_path_count": count_a == count_b,
        "additive_reference_identical": add_diff == 0.0,
        "cpc_outputs_different": cpc_diff > 1e-7,
        "cpc_contexts_different": context_diff > 1e-7,
    }
    return {
        "h0_fixed": True,
        "bond_embeddings_fixed": True,
        "A_paths": [[0, 1, 3], [0, 2, 4]],
        "B_paths": [[0, 1, 4], [0, 2, 3]],
        "additive_reference_A": [float(x) for x in add_a],
        "additive_reference_B": [float(x) for x in add_b],
        "additive_reference_max_abs_diff": add_diff,
        "cpc_delta_max_abs_diff": cpc_diff,
        "cpc_context_max_abs_diff": context_diff,
        "multiset_equal": bool(multiset_equal),
        "checks": checks,
        "all_ok": bool(all(checks.values())),
        "note": (
            "This only proves that CPC can respond to a re-paired (j,k) association with "
            "identical atom/bond marginals and fixed h0. It does not prove that historical "
            "W2 has fixed h0 on real pairings."
        ),
    }


def gate_gradients_and_cost(task: Any, device: torch.device) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    model, audit = M.build_cpc_model(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(C.RECIPE["warmup_start_lr"]),
        betas=tuple(C.RECIPE["adam_betas"]),
        eps=float(C.RECIPE["adam_eps"]),
        weight_decay=float(C.RECIPE["weight_decay"]),
    )
    C.seed_everything(C.SEED)
    plan = C.SD.make_plan40(task, seed=C.SEED)
    batches = plan[0]["batches"]
    stager = PinnedStager(device, depth=2)
    clip = float(C.RECIPE["gradient_clip"])
    new_names = [n for n, _ in model.named_parameters() if C.is_new_parameter(n)]
    audit_steps = set(int(x) for x in C.GRAD_AUDIT_STEPS)
    rows_out: dict[int, Any] = {}
    step_times: list[float] = []
    losses: list[float] = []
    warmup = int(C.BUDGET["profile_warmup_steps"])
    measured = int(C.BUDGET["profile_real_batches"])
    n_steps = warmup + measured
    t_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    for step, rows in enumerate(batches[:n_steps], start=1):
        t0 = time.perf_counter()
        batch = stager.batch(P.collate_with_cpc(task, task.cpc, rows.tolist()))
        y = stager.labels(task.labels[rows])
        lr = C.set_lrs(optimizer, step=step - 1)
        logits = model(batch)
        loss = C.MET.masked_bce_loss(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at profile step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = {n: (None if p.grad is None else float(p.grad.detach().norm().item())) for n, p in model.named_parameters() if C.is_new_parameter(n)}
        if step in audit_steps:
            rows_out[step] = {
                "loss": float(loss.detach()),
                "lr": lr,
                "new_grads": grads,
                "nonzero_new_tensors": sorted(n for n, v in grads.items() if v is not None and v > 0.0),
                "zero_or_none_new_tensors": sorted(n for n, v in grads.items() if v is None or v == 0.0),
                "new_grad_total_norm": math.sqrt(sum(float(v) ** 2 for v in grads.values() if v is not None)),
                "w2_grad_total_norm": math.sqrt(
                    sum(
                        float(p.grad.detach().pow(2).sum().item())
                        for n, p in model.named_parameters()
                        if not C.is_new_parameter(n) and p.grad is not None
                    )
                ),
            }
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
        if not math.isfinite(grad_norm):
            raise RuntimeError("non-finite gradient norm")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = float(time.perf_counter() - t0)
        if step > warmup:
            step_times.append(dt)
        losses.append(float(loss.detach()))
    wall = float(time.perf_counter() - t_start)
    stager.release()
    peak_alloc = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    arr = np.asarray(step_times, dtype=np.float64)
    # validation cost on a fixed 2048-row slice, scaled to the full split
    val_rows = np.asarray(task.valid_rows[:2048], dtype=np.int64)
    model.eval()
    t_val = time.perf_counter()
    with torch.inference_mode():
        M.predict_logits(model, task, val_rows, device, batch_size=256)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    val_2048 = float(time.perf_counter() - t_val)
    valid_full_est = val_2048 * (len(task.valid_rows) / len(val_rows))
    step_p50 = float(np.percentile(arr, 50)) if arr.size else float("nan")
    train_epoch_est = step_p50 * C.STEPS_PER_EPOCH
    projected_hours = 40.0 * (train_epoch_est + valid_full_est) / 3600.0
    return {
        "model": C.MODEL_NAME,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "warmup_steps": warmup,
        "measured_steps": measured,
        "wall_seconds": wall,
        "step_ms_p50": step_p50 * 1e3,
        "step_ms_p90": float(np.percentile(arr, 90) * 1e3) if arr.size else float("nan"),
        "step_ms_mean": float(arr.mean() * 1e3) if arr.size else float("nan"),
        "steps_per_second": float(len(step_times) / arr.sum()) if arr.size and arr.sum() > 0 else None,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "gpu_peak_allocated_bytes": peak_alloc,
        "gpu_peak_reserved_bytes": peak_reserved,
        "validation_2048_rows_seconds": val_2048,
        "validation_full_projected_seconds": valid_full_est,
        "train_epoch_projected_seconds": train_epoch_est,
        "projected_40_epoch_gpu_hours": projected_hours,
        "budget_target_hours": C.BUDGET["gpu_target_hours"],
        "budget_hard_hours": C.BUDGET["gpu_hard_hours"],
        "within_target": bool(projected_hours <= C.BUDGET["gpu_target_hours"]),
        "within_hard_cap": bool(projected_hours <= C.BUDGET["gpu_hard_hours"]),
        "gradient_audit": {str(k): v for k, v in rows_out.items()},
        "parameter_audit_all_match": audit["parameters"]["all_match"],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run_structure(args: argparse.Namespace) -> int:
    task = C.SD.load_task()
    cache_dir = Path(args.cache)
    if not (cache_dir / "meta.json").is_file() or args.rebuild:
        meta = P.build_cpc_cache(task, cache_dir, log_every=25000)
    else:
        meta = C.read_json(cache_dir / "meta.json")
    cpc = P.load_cpc_cache(cache_dir)
    task.cpc = cpc
    small = structural_small_graph_audit()
    full = full_cache_structure_audit(task, cpc)
    # mixed two-molecule batch and zero-path atom
    rows = [int(task.train_rows[0]), int(task.train_rows[1])]
    batch = P.collate_with_cpc(task, cpc, rows)
    mixed = P.check_batch_isolation(batch)
    payload = {
        "schema": "lcrc-pcba-cpc-structure-audit-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "cache_path": C.track_rel(cache_dir),
        "cache_meta": meta,
        "small_graphs": small,
        "full_cache": full,
        "mixed_batch": mixed,
        "all_ok": bool(small["all_ok"] and full["all_ok"] and mixed["isolated"]),
    }
    C.write_json(C.ANALYSIS / "structure_cache.json", payload)
    C.log(f"[structure] all_ok={payload['all_ok']} paths={full['paths']}")
    return 0 if payload["all_ok"] else 1


def run_preflight(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    task = C.SD.load_task()
    cache_dir = Path(args.cache)
    if not (cache_dir / "meta.json").is_file():
        raise RuntimeError("CPC structure cache missing; run the structure mode first")
    task.cpc = P.load_cpc_cache(cache_dir)
    init = gate_init(task, device)
    arch = gate_architecture(task, device)
    witness = gate_witness(task)
    cost = gate_gradients_and_cost(task, device)
    payload = {
        "schema": C.SCHEMA_PREFLIGHT,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "device": str(device),
        "environment": C.environment(),
        "host_facts": C.host_facts(),
        "cache": P.cpc_cache_identity(cache_dir),
        "init": init,
        "architecture": arch,
        "witness": witness,
        "cost": cost,
        "all_ok": bool(
            init["all_ok"]
            and arch["all_ok"]
            and witness["all_ok"]
            and cost["parameter_audit_all_match"]
            and cost["within_hard_cap"]
        ),
    }
    C.write_json(C.ANALYSIS / "preflight.json", payload)
    C.write_json(C.ANALYSIS / "witness.json", witness)
    C.write_json(C.ANALYSIS / "init_parity.json", init)
    C.write_json(C.ANALYSIS / "architecture_audit.json", arch)
    C.write_json(C.ANALYSIS / "gpu_profile.json", cost)
    C.log(
        f"[preflight] all_ok={payload['all_ok']} epoch0_cpu={init['epoch0_logits_max_abs_diff_cpu']} "
        f"arch={arch['all_ok']} witness={witness['all_ok']} projected={cost['projected_40_epoch_gpu_hours']:.4f}h"
    )
    return 0 if payload["all_ok"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("structure", "preflight"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache", default=str(P.CPC_CACHE_DIR))
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.mode == "structure":
            return run_structure(args)
        return run_preflight(args)
    except Exception as exc:  # noqa: BLE001
        C.write_json(
            C.ANALYSIS / f"audit_failed_{args.mode}.json",
            {
                "schema": "lcrc-pcba-cpc-audit-failure-v1",
                "round_id": C.ROUND_ID,
                "mode": args.mode,
                "created_utc": C.utcnow(),
                "git_commit": C.git_commit(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        C.log(f"[audit] FAILED {args.mode}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
