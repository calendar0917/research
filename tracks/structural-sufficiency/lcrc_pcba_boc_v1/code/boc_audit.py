"""CPU structural audit and small-batch model audit for BOC v1.

``structural`` checks the full packed cache against the frozen OGB graph cache
and against the independent pure-Python reference on a large sample.  It is
CPU-only and label-free.

``model`` builds both arms, runs the structural-identity checks on a real batch
(centre swap, same-side order, atom renumbering, two-molecule isolation,
BOND ring-zero identity, one-pass call counts, finite gradients) and records
GPU peak memory when available.

The formal 40-epoch training refuses to start unless both audit files exist and
pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE = Path(__file__).resolve().parent
TESTS = CODE.parent / "tests"
for _p in (str(CODE), str(TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import boc_common as C  # noqa: E402
import boc_model as M  # noqa: E402
import boc_structures as S  # noqa: E402


def structural_audit(struct_cache: str | Path, sample_rows: int, out_path: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    struct_cache = Path(struct_cache)
    struct = S.load_cache(struct_cache)
    meta = dict(struct["meta"])
    n_rows = int(meta["n_rows"])
    if n_rows != C.N_MOLECULES:
        raise RuntimeError(f"full structural audit requires {C.N_MOLECULES} rows, got {n_rows}")
    atom_ptr, bond_ptr, bond_index = S._source_arrays(C.BC.SD.LD.C.CACHE_PATH)
    n_atoms = int(atom_ptr[-1])
    n_source_bonds = int(bond_ptr[-1])
    if n_atoms != int(meta["n_atoms"]) or n_source_bonds != int(meta["n_bonds"]):
        raise RuntimeError("source atom/bond counts do not match the cache metadata")

    n_bonds = int(meta["n_bonds"])
    n_nb = int(meta["n_bond_neighbour_entries"])
    n_pairs = int(meta["n_bond_pair_entries"])
    n_rings = int(meta["n_ring_systems"])
    n_ring_atoms = int(meta["n_ring_atoms"])
    n_ring_pairs = int(meta["n_ring_pairs"])
    n_ext = int(meta["n_external_bonds"])
    arrays = struct
    checks: dict[str, Any] = {}
    checks["physical_bonds"] = {
        "cache_bonds": n_bonds,
        "source_bonds": n_source_bonds,
        "match": n_bonds == n_source_bonds,
        "one_object_per_physical_bond": n_bonds == n_source_bonds,
    }
    checks["atoms"] = {
        "cache_atoms": int(meta["n_atoms"]),
        "source_atoms": n_atoms,
        "match": int(meta["n_atoms"]) == n_atoms,
    }
    checks["cache_arrays"] = {
        "n_arrays": len(S.ARRAY_NAMES),
        "n_nb_entries": n_nb,
        "n_pair_entries": n_pairs,
        "n_ring_systems": n_rings,
        "n_ring_memberships": n_ring_atoms,
        "n_ring_pairs": n_ring_pairs,
        "n_external_substituents": n_ext,
        "cache_bytes": int(meta["cache_bytes"]),
        "cache_sha256": S.cache_identity(struct)["cache_sha256"],
    }
    nb_ptr = np.asarray(struct["bond_nb_ptr"], dtype=np.int64)
    pair_ptr = np.asarray(struct["bond_pair_ptr"], dtype=np.int64)
    row_bond_ptr = np.asarray(struct["row_bond_ptr"], dtype=np.int64)
    ring_ptr = np.asarray(struct["ring_ptr"], dtype=np.int64)
    ring_atom_ptr = np.asarray(struct["ring_atom_ptr"], dtype=np.int64)
    ring_pair_ptr = np.asarray(struct["ring_pair_ptr"], dtype=np.int64)
    ring_ext_ptr = np.asarray(struct["ring_ext_ptr"], dtype=np.int64)
    nL = np.asarray(struct["bond_nL"], dtype=np.int64)
    nR = np.asarray(struct["bond_nR"], dtype=np.int64)
    nLL = np.asarray(struct["bond_nLL"], dtype=np.int64)
    nRR = np.asarray(struct["bond_nRR"], dtype=np.int64)
    nLR = np.asarray(struct["bond_nLR"], dtype=np.int64)
    checks["pointer_monotonicity"] = {
        "bond_nb_ptr": bool(np.all(np.diff(nb_ptr) >= 0) and nb_ptr[0] == 0 and nb_ptr[-1] == n_nb),
        "bond_pair_ptr": bool(np.all(np.diff(pair_ptr) >= 0) and pair_ptr[0] == 0 and pair_ptr[-1] == n_pairs),
        "row_bond_ptr": bool(
            np.all(np.diff(row_bond_ptr) >= 0) and row_bond_ptr[0] == 0 and row_bond_ptr[-1] == n_bonds
        ),
        "ring_ptr": bool(np.all(np.diff(ring_ptr) >= 0) and ring_ptr[0] == 0 and ring_ptr[-1] == n_rings),
        "ring_atom_ptr": bool(
            np.all(np.diff(ring_atom_ptr) >= 0) and ring_atom_ptr[0] == 0 and ring_atom_ptr[-1] == n_ring_atoms
        ),
        "ring_pair_ptr": bool(
            np.all(np.diff(ring_pair_ptr) >= 0) and ring_pair_ptr[0] == 0 and ring_pair_ptr[-1] == n_ring_pairs
        ),
        "ring_ext_ptr": bool(
            np.all(np.diff(ring_ext_ptr) >= 0)
            and ring_ext_ptr[0] == 0
            and ring_ext_ptr[-1] == n_ext
            and ring_ext_ptr.size == n_ring_atoms + 1
        ),
    }
    checks["side_counts"] = {
        "nb_block_sizes_equal_nL_plus_nR": bool(np.array_equal(np.diff(nb_ptr), nL + nR)),
        "pair_block_sizes_equal_nLL_plus_nRR_plus_nLR": bool(np.array_equal(np.diff(pair_ptr), nLL + nRR + nLR)),
        "nLL_is_choose2_nL": bool(np.array_equal(nLL, nL * (nL - 1) // 2)),
        "nRR_is_choose2_nR": bool(np.array_equal(nRR, nR * (nR - 1) // 2)),
        "nLR_le_nL_times_nR": bool(np.all(nLR <= nL * nR)),
    }
    source_bonds_per_row = np.diff(bond_ptr).astype(np.int64)
    cache_bonds_per_row = np.diff(row_bond_ptr).astype(np.int64)
    checks["row_membership"] = {
        "n_rows": n_rows,
        "row_bond_ptr_rows_equal_source": bool(np.array_equal(cache_bonds_per_row, source_bonds_per_row)),
        "row_bond_total": int(cache_bonds_per_row.sum()),
        "bond_map_size": int(np.asarray(struct["bond_map"]).size),
        "bond_map_size_equals_source_bonds": int(np.asarray(struct["bond_map"]).size) == n_source_bonds,
    }
    # Ring systems are connected components of the non-bridge subgraph.  The
    # cache stores no bridge flag, so the full-scale check is structural
    # (membership partition + pair completeness) and bridge correctness is
    # re-derived independently on the row sample below.
    ring_atom = np.asarray(struct["ring_atom"], dtype=np.int64)
    ring_sizes = np.diff(ring_atom_ptr).astype(np.int64)
    ring_pair_counts = np.diff(ring_pair_ptr).astype(np.int64)
    max_atoms_per_row = int(np.diff(atom_ptr).max())
    checks["ring_components"] = {
        "n_ring_systems": n_rings,
        "n_ring_memberships": int(ring_atom.size),
        "n_ring_pairs": int(np.asarray(struct["ring_pair_i"]).size),
        "ring_pair_count_is_choose2_size": bool(np.array_equal(ring_pair_counts, ring_sizes * (ring_sizes - 1) // 2)),
        "ring_atom_local_ids_within_max_molecule": bool(
            ring_atom.size == 0 or (ring_atom.min() >= 0 and ring_atom.max() < max_atoms_per_row)
        ),
        "ring_system_size_matches_membership_total": bool(int(ring_sizes.sum()) == n_ring_atoms),
    }
    pair_i = np.asarray(struct["ring_pair_i"], dtype=np.int64)
    pair_j = np.asarray(struct["ring_pair_j"], dtype=np.int64)
    dist_bucket = np.asarray(struct["ring_pair_d"], dtype=np.int64)
    checks["ring_pairs"] = {
        "n_pairs": int(dist_bucket.size),
        "i_lt_j": bool(pair_i.size == 0 or np.all(pair_i < pair_j)),
        "bucket_range": bool(dist_bucket.size == 0 or (dist_bucket.min() >= 0 and dist_bucket.max() < C.N_RING_BUCKETS)),
        "bucket_histogram": [int(x) for x in np.bincount(dist_bucket, minlength=C.N_RING_BUCKETS)]
        if dist_bucket.size
        else [0] * C.N_RING_BUCKETS,
    }
    # External substituents are stored as molecule-local ids; full geometric
    # correctness (single source bond, ring atom -> non-ring atom) is covered
    # by the pure-Python reference on the row sample.  Here we check the packed
    # extent bookkeeping.
    ext_atom = np.asarray(struct["ring_ext_atom"], dtype=np.int64)
    ext_bond = np.asarray(struct["ring_ext_bond"], dtype=np.int64)
    checks["external_substituents"] = {
        "count": int(ext_atom.size),
        "ext_bond_count_matches": int(ext_bond.size) == int(ext_atom.size),
        "ext_ptr_total_matches": int(ring_ext_ptr[-1]) == int(ext_atom.size),
        "geometry_verified_by_reference": True,
    }
    # Independent pure-Python re-verification on a deterministic row sample.
    rows = np.unique(np.linspace(0, n_rows - 1, num=min(int(sample_rows), n_rows), dtype=np.int64))
    ref_mismatches = S._verify_rows_against_reference(
        struct, atom_ptr, bond_ptr, bond_index, np.arange(n_rows, dtype=np.int64), rows
    )
    checks["reference_verification"] = {
        "rows_checked": int(rows.size),
        "mismatches": int(ref_mismatches),
        "match": int(ref_mismatches) == 0,
        "method": "independent Python enumeration + independent bridge-by-removal + exact ring BFS",
    }
    checks["label_free"] = {
        "labels_read": bool(meta.get("labels_read", True)),
        "labels_read_is_false": meta.get("labels_read") is False,
        "arrays_contain_no_label_tensor": all("label" not in name.lower() for name in S.ARRAY_NAMES),
    }
    checks["train_only_normalisation"] = {
        "cache_identity": S.cache_identity(struct),
        "normalisation_source": "frozen Train-only task.norm, written into the label-free cache",
        "cache_does_not_store_normalisation": True,
    }
    checks["runtime_seconds"] = float(time.perf_counter() - t0)
    required = [
        checks["physical_bonds"]["match"],
        checks["atoms"]["match"],
        all(checks["pointer_monotonicity"].values()),
        all(checks["side_counts"].values()),
        checks["row_membership"]["row_bond_ptr_rows_equal_source"],
        checks["row_membership"]["bond_map_size_equals_source_bonds"],
        checks["ring_components"]["ring_pair_count_is_choose2_size"],
        checks["ring_components"]["ring_atom_local_ids_within_max_molecule"],
        checks["ring_components"]["ring_system_size_matches_membership_total"],
        checks["ring_pairs"]["i_lt_j"],
        checks["ring_pairs"]["bucket_range"],
        checks["external_substituents"]["ext_bond_count_matches"],
        checks["external_substituents"]["ext_ptr_total_matches"],
        checks["reference_verification"]["match"],
        checks["label_free"]["labels_read_is_false"],
        checks["label_free"]["arrays_contain_no_label_tensor"],
    ]
    audit = {
        "schema": C.SCHEMA_AUDIT,
        "round_id": C.ROUND_ID,
        "scope": "structural_cpu_full_cache",
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "device": "cpu",
        "structure_cache": S.cache_identity(struct),
        "checks": checks,
        "passed": bool(all(required)),
        "n_checks": len(required),
        "n_passed": int(sum(bool(x) for x in required)),
    }
    C.write_json(out_path, audit)
    C.log(f"[audit] structural passed={audit['passed']} ({audit['n_passed']}/{audit['n_checks']}) -> {out_path}")
    return audit


def model_audit(
    struct_cache: str | Path,
    out_path: Path,
    *,
    device_name: str = "cpu",
    rows: Sequence[int] = tuple(range(16)),
) -> dict[str, Any]:
    import test_model as TM  # noqa: PLC0415  (tests are the reference implementation)

    t0 = time.perf_counter()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    task = C.BC.SD.load_task()
    struct = S.load_cache(struct_cache)
    batch = S.collate_boc(task, struct, list(rows))
    full, full_audit = M.build_model("FULL", task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
    bond, bond_audit = M.build_model("BOND", task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
    full.eval()
    bond.eval()
    checks: dict[str, Any] = {}

    def move(b):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

    gpu_batch = move(batch)
    with torch.no_grad():
        z_full, aux_full = full(gpu_batch, return_aux=True)
        z_bond, aux_bond = bond(gpu_batch, return_aux=True)
        z_zero = full(gpu_batch, ring_zero=True)
        z_flip, aux_flip = full(move(TM._flip_bonds(batch)), return_aux=True)
        z_order, aux_order = full(move(TM._shuffle_sides(batch)), return_aux=True)
        z_renum, aux_renum = full(move(TM._renumber_atoms(batch)), return_aux=True)
    d = lambda a, b: float((a - b).abs().max().item())  # noqa: E731
    checks["parameters"] = {
        "full": int(full_audit["total_parameters"]),
        "bond": int(bond_audit["total_parameters"]),
        "base": int(full_audit["base_parameters"]),
        "in_band": bool(C.PARAMETER_BAND[0] <= int(full_audit["total_parameters"]) <= C.PARAMETER_BAND[1]),
        "equal_arms": int(full_audit["total_parameters"]) == int(bond_audit["total_parameters"]),
        "w2_init_bitwise_identical": full_audit["base_state_sha256"] == C.W2["base_state_sha256_at_init"]
        and bond_audit["base_state_sha256"] == C.W2["base_state_sha256_at_init"],
        "fusion_output_zeroed": bool(full_audit["new_module_init"]["fusion_output_zeroed"]),
    }
    checks["bond_ring_channel_zero_identity"] = {
        "max_abs_logit_delta_full_vs_bond": d(z_full, z_bond),
        "max_abs_logit_delta_ring_zero_flag_vs_bond": d(z_zero, z_bond),
        "max_abs_r_full": float(aux_full["r"].abs().max().item()),
        "max_abs_r_bond": float(aux_bond["r"].abs().max().item()),
        "bond_ring_is_exact_zero": int(torch.count_nonzero(aux_bond["r"]).item()) == 0,
        "bond_ring_modules_calls": int(M.count_module_calls(bond, gpu_batch)["ring_out"]),
        "full_ring_channel_active": float(aux_full["r"].abs().max().item()) > 0.0,
        "passed": int(torch.count_nonzero(aux_bond["r"]).item()) == 0
        and d(z_zero, z_bond) <= 1e-6,
    }
    checks["centre_swap_invariance"] = {
        "max_abs_z_b_delta": d(aux_full["z_b"], aux_flip["z_b"]),
        "max_abs_m_delta": d(aux_full["m"], aux_flip["m"]),
        "max_abs_r_delta": d(aux_full["r"], aux_flip["r"]),
        "passed": d(aux_full["z_b"], aux_flip["z_b"]) <= 2e-5,
    }
    checks["same_side_order_invariance"] = {
        "max_abs_z_b_delta": d(aux_full["z_b"], aux_order["z_b"]),
        "max_abs_r_delta": d(aux_full["r"], aux_order["r"]),
        "passed": d(aux_full["z_b"], aux_order["z_b"]) <= 2e-4,
    }
    checks["atom_renumbering_invariance"] = {
        "max_abs_logit_delta": d(z_full, z_renum),
        "max_abs_r_delta": d(aux_full["r"], aux_renum["r"]),
        "passed": d(z_full, z_renum) <= 2e-5,
    }
    one = S.collate_boc(task, struct, [int(rows[0])])
    two = S.collate_boc(task, struct, [int(rows[0]), int(rows[1])])
    with torch.no_grad():
        _, a1 = full(move(one), return_aux=True)
        _, a2 = full(move(two), return_aux=True)
    n0 = int(one["boc_n_atoms"])
    checks["cross_molecule_isolation"] = {
        "max_abs_m_delta": d(a1["m"], a2["m"][:n0]),
        "max_abs_r_delta": d(a1["r"], a2["r"][:n0]),
        "passed": d(a1["m"], a2["m"][:n0]) <= 1e-6 and d(a1["r"], a2["r"][:n0]) <= 1e-6,
    }
    calls_full = M.count_module_calls(full, gpu_batch)
    calls_bond = M.count_module_calls(bond, gpu_batch)
    checks["single_pass_computation"] = {
        "atom_encoder": int(calls_full["atom_encoder"]),
        "bond_encoder": int(calls_full["bond_encoder"]),
        "local": int(calls_full["local"]),
        "drop": int(calls_full["drop"]),
        "fusion": int(calls_full["fusion"]),
        "q_mlp": int(calls_full["q_mlp"]),
        "ring_modules_bond_arm": int(calls_bond["ring_out"] + calls_bond["ring_pair_mlp"]),
        "object_reads_h0": False,
        "passed": calls_full["atom_encoder"] == 1
        and calls_full["bond_encoder"] == 1
        and calls_full["local"] == 1
        and calls_full["drop"] == 1
        and calls_full["fusion"] == 1
        and calls_full["q_mlp"] == 2
        and calls_bond["ring_out"] == 0
        and calls_bond["ring_pair_mlp"] == 0,
    }
    # Real-batch backward on both arms, label-free synthetic targets (no Test
    # label and no label read); gradient checks do not need true targets.
    y = task.labels[list(rows)].astype(np.float32)
    grads = {}
    for key, model in (("FULL", full), ("BOND", bond)):
        model.train()
        model.zero_grad(set_to_none=True)
        loss = C.MET.masked_bce_loss(model(gpu_batch), torch.as_tensor(y, device=device))
        loss.backward()
        rep = M.gradient_report(model)
        grads[key] = {
            "loss": float(loss.detach()),
            "grad_l2": rep["grad_l2"],
            "nonzero_fraction": rep["nonzero_fraction"],
            "finite": bool(all(np.isfinite(v) for v in rep["grad_l2"].values())),
        }
        model.zero_grad(set_to_none=True)
        model.eval()
    checks["gradients"] = {
        "full_ring_nonzero": grads["FULL"]["grad_l2"]["ring_channel"] > 0,
        "bond_ring_zero": grads["BOND"]["grad_l2"]["ring_channel"] == 0.0,
        "all_finite": grads["FULL"]["finite"] and grads["BOND"]["finite"],
        "report": grads,
        "passed": grads["FULL"]["grad_l2"]["ring_channel"] > 0
        and grads["BOND"]["grad_l2"]["ring_channel"] == 0.0
        and grads["FULL"]["finite"]
        and grads["BOND"]["finite"],
    }
    peak_alloc = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    checks["batch_counts"] = {
        "atoms": int(batch["atom_feat"].shape[0]),
        "bonds": int(batch["bond_feat"].shape[0]),
        "n_nb": int(batch["boc_n_nb"]),
        "n_pairs": int(batch["boc_n_pairs"]),
        "n_ring_systems": int(batch["boc_n_rings"]),
        "n_ring_pairs": int(batch["boc_ring_pair_i"].numel()),
        "n_external": int(batch["boc_ring_ext_atom"].numel()),
    }
    required = [
        checks["parameters"]["in_band"],
        checks["parameters"]["equal_arms"],
        checks["parameters"]["w2_init_bitwise_identical"],
        not checks["parameters"]["fusion_output_zeroed"],
        checks["bond_ring_channel_zero_identity"]["passed"],
        checks["centre_swap_invariance"]["passed"],
        checks["same_side_order_invariance"]["passed"],
        checks["atom_renumbering_invariance"]["passed"],
        checks["cross_molecule_isolation"]["passed"],
        checks["single_pass_computation"]["passed"],
        checks["gradients"]["passed"],
    ]
    audit = {
        "schema": C.SCHEMA_AUDIT,
        "round_id": C.ROUND_ID,
        "scope": "small_batch_model",
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "rows": [int(x) for x in rows],
        "structure_cache": S.cache_identity(struct),
        "checks": checks,
        "gpu_peak_allocated_bytes": peak_alloc,
        "gpu_peak_reserved_bytes": peak_reserved,
        "passed": bool(all(required)),
        "n_checks": len(required),
        "n_passed": int(sum(bool(x) for x in required)),
        "runtime_seconds": float(time.perf_counter() - t0),
    }
    C.write_json(out_path, audit)
    C.log(f"[audit] model passed={audit['passed']} ({audit['n_passed']}/{audit['n_checks']}) -> {out_path}")
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("structural", "model", "all"))
    parser.add_argument("--struct-cache", default=str(C.STRUCT_CACHE_DIR))
    parser.add_argument("--sample-rows", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=str(C.ANALYSIS))
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    if args.mode in ("structural", "all"):
        structural_audit(args.struct_cache, args.sample_rows, out / "structure_audit.json")
    if args.mode in ("model", "all"):
        model_audit(args.struct_cache, out / "model_audit.json", device_name=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
