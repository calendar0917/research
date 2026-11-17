"""Prepare stage: freeze the topology partition, the routing cache and the budget.

Everything here is CPU-only and runs before any GPU training:

1. build (or re-verify) the three per-split routing caches and print their
   SHA-256; the two GPU workers read exactly these bytes;
2. structural support statistics over *all* official rows (label free);
3. the frozen parameter budget / initial-state identity of R1 and R2;
4. the protocol freeze record (constants, code hashes, file hashes).

The support gate is declared in advance: if fewer than 50% of the Valid
molecules have at least two blocks **and** at least one cross-boundary S2
occurrence, the round stops with ``UNINFORMATIVE_TOPOLOGY_PARTITION`` and no
architecture training happens.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

import rr_common as C
import rr_data as D
import rr_models as M
import rr_partition as P
import rr_routing as R
import rr_train as T


# ---------------------------------------------------------------------------
# routing cache
# ---------------------------------------------------------------------------


def build_routing_caches(bundle, *, force: bool = False) -> dict[str, Any]:
    C.ensure_dir(C.CACHE_DIR)
    manifest_path = C.CACHE_DIR / "manifest.json"
    previous = C.read_json(manifest_path) if manifest_path.is_file() else {}
    out: dict[str, Any] = {"splits": {}, "manifest_path": str(manifest_path)}
    for split in C.SPLITS:
        path = C.routing_cache_path(split)
        cache = bundle.caches[split]
        n_bond_types = int(bundle.extra["n_bond_types"])
        rebuilt = R.build_split_routing(cache, split, n_bond_types=n_bond_types)
        digest = C.sha256_file(path) if path.is_file() else None
        fingerprint = rebuilt.fingerprint
        reuse = (
            not force
            and digest is not None
            and previous.get("splits", {}).get(split, {}).get("fingerprint") == fingerprint
        )
        if reuse:
            arrays = D.load_routing(split, verify=False)[0]
            stored = previous["splits"][split]
        else:
            payload = {key: np.ascontiguousarray(value) for key, value in rebuilt.arrays.items()}
            C.atomic_write_npz(path, payload)
            digest = C.sha256_file(path)
            with np.load(path, allow_pickle=False) as mesh:
                arrays = {key: np.asarray(mesh[key]) for key in mesh.files}
            stored = {
                "sha256": digest,
                "fingerprint": fingerprint,
                "stats": rebuilt.stats,
            }
            out["rebuilt"] = True
        stored = dict(stored)
        stored["sha256"] = digest
        stored["stats"] = rebuilt.stats
        # the rebuilt and re-read arrays must be identical (no silent drift)
        if C.sha256_json({k: C.sha256_array(v) for k, v in sorted(arrays.items())}) != fingerprint:
            raise RuntimeError(f"{split}: routing cache on disk does not match the rebuilt fingerprint")
        out["splits"][split] = stored
    manifest = {
        "round_id": C.ROUND_ID,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generator": "rr_routing.build_split_routing",
        "partition_rule": "bridges + non-bridge components + maximal bridge chains + zero-bond atoms",
        "splits": {split: dict(out["splits"][split]) for split in C.SPLITS},
    }
    C.atomic_write_json(manifest_path, manifest)
    out["manifest"] = manifest
    return out


# ---------------------------------------------------------------------------
# support statistics (all official rows, label free)
# ---------------------------------------------------------------------------


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def partition_support(bundle, manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "dataset": bundle.dataset,
        "gate": {
            "criterion": ">=2 blocks AND >=1 cross-boundary S2 occurrence on the molecule",
            "split": "valid",
            "required_fraction": C.PARTITION_GATE_FRACTION,
        },
        "splits": {},
    }
    if manifest is None:
        manifest = C.read_json(C.CACHE_DIR / "manifest.json")
    for split in C.SPLITS:
        arrays = bundle.routing[split]
        stats = manifest["splits"][split]["stats"]
        cache = bundle.caches[split]
        n_mol = int(cache["n_molecules"])
        n_blocks = bundle.routing_stats[split]["n_blocks"]
        n_cross = bundle.routing_stats[split]["n_cross_s2"]
        n_iface = bundle.routing_stats[split]["n_interfaces"]
        usable = bundle.routing_stats[split]["usable"]
        path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
        atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
        path_counts = np.diff(path_ptr)
        s2_per_mol = np.asarray(
            [int(path_ptr[int(atom_ptr[i + 1])] - path_ptr[int(atom_ptr[i])]) for i in range(n_mol)],
            dtype=np.int64,
        )
        n_components = np.asarray(cache["n_components"], dtype=np.int64)
        blk_n_bonds = np.asarray(arrays["blk_n_bonds"], dtype=np.int64)
        blk_n_atoms = np.asarray(arrays["blk_n_atoms"], dtype=np.int64)
        blk_kind = np.asarray(arrays["blk_kind"], dtype=np.int64)
        cyclic = n_components == 1  # filled properly below with cyclomatic number
        bond_counts = np.asarray(
            [int(np.asarray(cache["bond_ptr"])[i + 1] - np.asarray(cache["bond_ptr"])[i]) for i in range(n_mol)],
            dtype=np.int64,
        )
        cyclomatic = bond_counts - np.asarray(
            [int(atom_ptr[i + 1] - atom_ptr[i]) for i in range(n_mol)], dtype=np.int64
        ) + n_components
        ring_molecule = cyclomatic > 0
        entry = {
            "n_molecules": n_mol,
            "n_blocks": int(n_blocks.sum()),
            "mean_blocks_per_molecule": float(n_blocks.mean()),
            "block_count_distribution": _value_counts(n_blocks),
            "block_bond_size_distribution": _value_counts(blk_n_bonds),
            "block_atom_size_distribution": _value_counts(blk_n_atoms),
            "block_kind_counts": {
                "cycle": int((blk_kind == 0).sum()),
                "chain": int((blk_kind == 1).sum()),
                "isolated_atom": int((blk_kind == 2).sum()),
            },
            "max_block_bonds": int(blk_n_bonds.max()) if blk_n_bonds.size else 0,
            "max_block_atoms": int(blk_n_atoms.max()) if blk_n_atoms.size else 0,
            "zero_bond_blocks": int((blk_n_bonds == 0).sum()),
            "n_shared_atoms": int(stats["n_atoms_in_multiple_blocks"]),
            "shared_atom_fraction": float(stats["n_atoms_in_multiple_blocks"]) / max(1, int(stats["n_s1_occurrences"]) // 2),
            "n_interfaces": int(n_iface.sum()),
            "mean_interfaces_per_molecule": float(n_iface.mean()),
            "interface_count_distribution": _value_counts(n_iface),
            "n_s1_occurrences": int(stats["n_s1_occurrences"]),
            "n_s2_total": int(stats["n_s2_total"]),
            "n_s2_internal": int(stats["n_s2_internal"]),
            "n_s2_cross": int(stats["n_s2_cross"]),
            "s2_per_molecule": _quantiles(s2_per_mol),
            "s2_internal_fraction": float(stats["n_s2_internal"]) / max(1, int(stats["n_s2_total"])),
            "molecules_with_ge2_blocks": int((n_blocks >= 2).sum()),
            "molecules_with_cross_s2": int((n_cross > 0).sum()),
            "molecules_usable": int(usable.sum()),
            "usable_fraction": float(usable.mean()),
            "cyclic_molecules": int(ring_molecule.sum()),
            "acyclic_molecules": int((~ring_molecule).sum()),
            "cyclic_usable_fraction": float(usable[ring_molecule].mean()) if ring_molecule.any() else 0.0,
            "acyclic_usable_fraction": float(usable[~ring_molecule].mean()) if (~ring_molecule).any() else 0.0,
            "cyclic_mean_blocks": float(n_blocks[ring_molecule].mean()) if ring_molecule.any() else 0.0,
            "acyclic_mean_blocks": float(n_blocks[~ring_molecule].mean()) if (~ring_molecule).any() else 0.0,
            "single_block_molecules": int((n_blocks < 2).sum()),
            "molecules_without_s2": int((s2_per_mol == 0).sum()),
        }
        out["splits"][split] = entry
    valid_fraction = out["splits"]["valid"]["usable_fraction"]
    out["gate"]["observed_fraction"] = float(valid_fraction)
    out["gate"]["status"] = (
        "PARTITION_USABLE" if valid_fraction >= C.PARTITION_GATE_FRACTION else "UNINFORMATIVE_TOPOLOGY_PARTITION"
    )
    out["gate"]["pass"] = bool(valid_fraction >= C.PARTITION_GATE_FRACTION)
    out["routing_cache_sha256"] = {
        split: C.sha256_file(C.routing_cache_path(split)) for split in C.SPLITS
    }
    out["routing_array_fingerprint"] = {
        split: bundle.routing_fingerprints[split] for split in C.SPLITS
    }
    out["routing_cache_file_sha256"] = dict(out["routing_cache_sha256"])
    return out


def _value_counts(values: Sequence[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in np.asarray(values, dtype=np.int64).tolist():
        out[str(int(v))] = out.get(str(int(v)), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


# ---------------------------------------------------------------------------
# parameter budget / init identity
# ---------------------------------------------------------------------------


def parameter_budget(bundle) -> dict[str, Any]:
    spec, norm = bundle.spec, bundle.norm
    canonical = M.canonical_trunk_state(spec, norm)
    expected = M.expected_parameter_counts(int(spec.global_dim))
    out: dict[str, Any] = {
        "analytic": expected,
        "c1_reference_total": int(M.CM.FULL_C1_PARAMS),
        "budget_rule": "total parameters within +/-25% of the frozen C1 total",
        "capacity_rule": f"R2/R1 parameter ratio <= {C.CAPACITY_CONFOUNDED_RATIO} else CAPACITY_CONFOUNDED",
        "models": {},
    }
    states = {}
    for variant in C.MODELS:
        model = M.build_model(variant, spec, norm, canonical, device="cpu")
        report = model.parameter_report()
        states[variant] = model.state_dict()
        out["models"][variant] = {
            "total": report["total"],
            "trunk": report["trunk"],
            "readout": report["readout"],
            "breakdown": report["breakdown"],
            "matches_analytic": int(report["total"]) == int(expected[f"{variant}_total"]),
            "readout_state_hash": C.sha256_state(model.readout_state()),
            "full_state_hash": C.sha256_state(model.state_dict()),
            "trunk_matches_canonical": C.sha256_state(
                {k[len(M.ARM_PREFIX):]: v for k, v in model.state_dict().items() if k.startswith(M.ARM_PREFIX)}
            )
            == C.sha256_state(canonical, keys=M.CM.trunk_keys(canonical)),
        }
    shared = [k for k in states["R1"] if not k.startswith("context.")]
    same = all(
        bool(np.array_equal(states["R1"][k].numpy(), states["R2"][k].numpy())) for k in shared
    )
    out["shared_initial_state_identical"] = bool(same)
    out["shared_parameter_tensors"] = len(shared)
    r1, r2 = out["models"]["R1"]["total"], out["models"]["R2"]["total"]
    out["ratio"] = {
        "r1_over_c1": float(r1) / float(M.CM.FULL_C1_PARAMS),
        "r2_over_c1": float(r2) / float(M.CM.FULL_C1_PARAMS),
        "r2_over_r1": float(r2) / float(r1),
        "within_25pct_of_c1": bool(
            abs(out["models"]["R1"]["total"] / M.CM.FULL_C1_PARAMS - 1.0) <= 0.25
            and abs(out["models"]["R2"]["total"] / M.CM.FULL_C1_PARAMS - 1.0) <= 0.25
        ),
        "capacity_confounded": bool(float(r2) / float(r1) > C.CAPACITY_CONFOUNDED_RATIO),
    }
    # the schedule is the frozen Rf schedule of the route-clarification round
    import importlib

    gr_common = None
    if _has_module("gr_common"):
        gr_common = importlib.import_module("gr_common")
    n_train = int(bundle.split_sizes["train"])
    schedule_check = None
    if gr_common is not None:
        steps = T.total_steps(n_train)
        diffs = [
            abs(T.lr_at(step, n_train) - float(gr_common.lr_at("Rf", step, n_train)))
            for step in range(0, steps + 1)
        ]
        schedule_check = {
            "reference": "gnn_route_clarification_8h_v1.gr_common.lr_at('Rf', ...)",
            "n_steps_checked": int(steps + 1),
            "max_abs_diff": float(max(diffs)),
            "match": bool(max(diffs) <= 1e-12),
        }
    out["lr_schedule_check"] = schedule_check
    return out


def _has_module(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # pragma: no cover - defensive
        return False


# ---------------------------------------------------------------------------
# protocol freeze
# ---------------------------------------------------------------------------


def protocol_freeze(bundle, routing: Mapping[str, Any], support: Mapping[str, Any], budget: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git": {
            "commit": C.git_commit(),
            "branch": C.git_branch(),
            "dirty_tracked": C.git_dirty_tracked(),
            "lockfile_sha256": C.lockfile_sha256(),
        },
        "dataset": {
            "name": C.DATASET,
            "frozen_files": bundle.extra["frozen_files"],
            "split_sizes": bundle.split_sizes,
            "targets": {
                task: {
                    "description": C.TASK_TARGET_NAME[task],
                    "scaler": bundle.scaler(task),
                }
                for task in C.TASKS
            },
            "wc_logp_label_sha256": bundle.extra["wc_logp_label_sha256"],
            "oracle_contrib_sha256": bundle.extra["oracle_contrib_sha256"],
        },
        "partition_rule": {
            "statement": (
                "bridges of the physical bond graph; non-bridge connected components = cycle blocks; "
                "maximal bridge paths with degree-2 non-ring interiors = chain blocks; isolated atoms = "
                "zero-bond blocks; every bond owned exactly once"
            ),
            "uses_atom_input_order": False,
            "uses_bond_ids": False,
            "uses_labels": False,
            "uses_smarts": False,
            "weights": "w(v,i) = #incident bonds owned by block i / degree(v)",
            "reference_impl": "rr_partition.partition_molecule",
        },
        "routing_rule": {
            "s1": "each S1 occurrence routes to owner(bond)",
            "s2_internal": "owner(e1)==owner(e2) -> internal S2 of that block (whole occurrence)",
            "s2_cross": "owner(e1)!=owner(e2) -> interface {owner(e1),owner(e2)} (whole occurrence, once)",
            "reference_impl": "rr_routing.build_split_routing",
        },
        "architecture": {
            "block_token": "z_i = sum_v w(v,i) h_(v,i); Z_i = [z_i, log1p(n_bonds_i), log1p(n_atoms_i)]",
            "local_input_width": M.LOCAL_INPUT_DIM,
            "interface_evidence_dim": C.IFACE_EVIDENCE_DIM,
            "interface_readout": {
                "R1": "c_ij = f_interface(t_ij)",
                "R2": "c_ij = f_interface(t_ij + MLP_context([Z_i+Z_j, |Z_i-Z_j|, Z_i*Z_j]))",
            },
            "full_graph_term": "rho_g(g, log1p(n_atoms)) only; no mean(h), no mean(h^2)",
            "dropout": "one shared dropout(0.10) on the block-conditioned local representation",
            "bias": "one learnable scalar added to every molecule",
        },
        "budget": budget,
        "recipe": T.recipe_record(),
        "partition_support_gate": support["gate"],
        "decision_scales": {
            "retention_factor_valid": C.RETENTION_FACTOR,
            "capacity_confounded_ratio": C.CAPACITY_CONFOUNDED_RATIO,
            "bootstrap_draws": C.BOOTSTRAP_DRAWS,
            "bootstrap_seed": C.BOOTSTRAP_SEED,
            "oracle_control_seed": C.ORACLE_CONTROL_SEED,
            "oracle_controls": [C.MIN_ORACLE_CONTROLS, C.MAX_ORACLE_CONTROLS],
            "selection_score": "argmax_i |delta_i| / sqrt(max(1, n_bonds_i)); ties -> smallest block index",
        },
        "routing_cache": {
            "file_sha256": support["routing_cache_file_sha256"],
            "array_fingerprints": support["routing_array_fingerprint"],
        },
        "code_sha256": T.code_hashes(),
    }
    payload["freeze_hash"] = C.sha256_json({k: v for k, v in payload.items() if k != "freeze_hash"})
    return payload


def prepare_all(*, force: bool = False) -> dict[str, Any]:
    base_bundle = D.build_bundle(verify=True, require_routing=False, verify_routing_sha=False)
    routing = build_routing_caches(base_bundle, force=force)
    D._BUNDLE_CACHE.clear()          # re-read the freshly written routing caches
    bundle = D.build_bundle(verify=True, verify_routing_sha=False)
    for split in C.SPLITS:
        if bundle.routing_fingerprints[split] != routing["manifest"]["splits"][split]["fingerprint"]:
            raise RuntimeError(f"{split}: re-read routing cache does not match the rebuilt fingerprint")
    support = partition_support(bundle, routing["manifest"])
    budget = parameter_budget(bundle)
    freeze = protocol_freeze(bundle, routing, support, budget)
    C.ensure_dir(C.ANALYSIS_DIR)
    C.atomic_write_json(C.ANALYSIS_DIR / "partition_support.json", support)
    C.atomic_write_json(C.ANALYSIS_DIR / "parameter_budget.json", budget)
    C.atomic_write_json(C.ANALYSIS_DIR / "dataset_audit.json", D.target_audit(bundle))
    C.atomic_write_json(C.ANALYSIS_DIR / "protocol_freeze.json", freeze)
    return {
        "stage": "prepare",
        "partition_gate": support["gate"],
        "routing_cache_file_sha256": support["routing_cache_file_sha256"],
        "parameter_ratio": budget["ratio"],
        "freeze_hash": freeze["freeze_hash"],
    }


__all__ = [
    "build_routing_caches",
    "partition_support",
    "parameter_budget",
    "protocol_freeze",
    "prepare_all",
]
