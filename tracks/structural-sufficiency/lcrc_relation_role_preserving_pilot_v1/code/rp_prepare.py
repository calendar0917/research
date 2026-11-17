"""Prepare stage: re-verify the frozen reuse, freeze the roles, the budget and the protocol.

Everything here is CPU-only and runs before any GPU training:

1. byte-level SHA-256 of the three frozen routing caches reused from
   ``lcrc_relation_interface_routing_pilot_v1`` **and** a full rebuild of every
   routing array from the frozen partition implementation; the two GPU workers
   read exactly the same bytes;
2. the frozen CONTROL S2 type -> role table (label-free hash), its manifest and
   the per-split group sizes of both REAL and CONTROL;
3. the parameter budget: the two models must be identical in count and in
   initial state, and differ from C1 by exactly the two 200-d gamma vectors;
4. the protocol freeze record (constants, code hashes, file hashes, recipe).
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

import rp_common as C
import rp_data as D
import rp_models as M
import rp_train as T


# ---------------------------------------------------------------------------
# frozen-reuse verification
# ---------------------------------------------------------------------------


def verify_reuse(bundle) -> dict[str, Any]:
    """File SHA-256 + full partition/routing rebuild of every split."""
    PM = C.pred_modules()
    n_bond_types = int(bundle.extra["n_bond_types"])
    files = D.verify_reuse_files()
    out: dict[str, Any] = {
        "source_round": C.PRED_DIR.name,
        "manifest_sha256": C.sha256_file(C.PRED_MANIFEST) if C.PRED_MANIFEST.is_file() else None,
        "files": files["files"],
        "splits": {},
        "all_pass": bool(files["all_match"]),
    }
    for split in C.SPLITS:
        t0 = time.perf_counter()
        rebuilt = PM["rr_routing"].build_split_routing(
            bundle.caches[split], split, n_bond_types=n_bond_types
        )
        frozen = bundle.routing[split]
        array_match = {}
        for key in sorted(rebuilt.arrays):
            if key not in frozen:
                array_match[key] = False
                continue
            array_match[key] = bool(
                np.array_equal(np.asarray(rebuilt.arrays[key]), np.asarray(frozen[key]))
            )
        fingerprint_match = rebuilt.fingerprint == C.EXPECTED_ROUTING_FINGERPRINT[split]
        ok = bool(all(array_match.values())) and fingerprint_match
        out["splits"][split] = {
            "fingerprint": rebuilt.fingerprint,
            "expected_fingerprint": C.EXPECTED_ROUTING_FINGERPRINT[split],
            "fingerprint_match": bool(fingerprint_match),
            "arrays_match": array_match,
            "n_arrays": len(array_match),
            "rebuild_seconds": time.perf_counter() - t0,
            "pass": ok,
        }
        out["all_pass"] = out["all_pass"] and ok
    if not out["all_pass"]:
        raise RuntimeError(f"frozen routing cache could not be reproduced: {out['splits']}")
    return out


# ---------------------------------------------------------------------------
# parameter budget / initial-state identity
# ---------------------------------------------------------------------------


def parameter_budget(bundle) -> dict[str, Any]:
    spec, norm = bundle.spec, bundle.norm
    canonical = M.canonical_state(spec, norm)
    expected = M.expected_parameter_counts(int(spec.global_dim))
    out: dict[str, Any] = {
        "analytic": expected,
        "c1_reference_total": int(M.CM.FULL_C1_PARAMS),
        "role_dim": int(C.S2_ROLE_DIM),
        "budget_rule": "identical total parameters for RR_REAL and RR_CONTROL; delta over C1 is exactly 2*200",
        "models": {},
    }
    states = {}
    for variant in C.MODELS:
        model = M.build_model(variant, spec, norm, canonical, device="cpu")
        report = model.parameter_report()
        states[variant] = {k: v.clone() for k, v in model.state_dict().items()}
        canonical_part = model.canonical_parameters()
        out["models"][variant] = {
            "total": report["total"],
            "trunk": report["trunk"],
            "readout": report["readout"],
            "role_gamma": report["role_gamma"],
            "breakdown": report["breakdown"],
            "matches_analytic": int(report["total"]) == int(expected["rr_total"]),
            "c1_part_parameters": int(sum(v.numel() for v in canonical_part.values())),
            "c1_part_matches_c1_total": int(sum(v.numel() for v in canonical_part.values()))
            == int(M.CM.FULL_C1_PARAMS),
            "c1_part_state_hash": C.sha256_state(canonical_part),
            "full_state_hash": C.sha256_state(model.state_dict()),
            "gamma_all_ones": bool(
                (model.gamma_internal == 1).all() and (model.gamma_cross == 1).all()
            ),
        }
    same = set(states["RR_REAL"]) == set(states["RR_CONTROL"]) and all(
        bool(np.array_equal(states["RR_REAL"][k].numpy(), states["RR_CONTROL"][k].numpy()))
        for k in states["RR_REAL"]
    )
    out["initial_state_identical"] = bool(same)
    out["full_initial_state_hash"] = C.sha256_state(states["RR_REAL"])
    out["c1_canonical_state_hash"] = C.sha256_state(canonical)
    real = out["models"]["RR_REAL"]["total"]
    control = out["models"]["RR_CONTROL"]["total"]
    out["ratio"] = {
        "real_over_c1": float(real) / float(M.CM.FULL_C1_PARAMS),
        "control_over_c1": float(control) / float(M.CM.FULL_C1_PARAMS),
        "real_over_control": float(real) / float(control),
        "identical_counts": bool(real == control),
    }
    # the schedule is the frozen Rf schedule of the route-clarification round
    out["lr_schedule_check"] = T.schedule_audit(int(bundle.split_sizes["train"]))
    return out


# ---------------------------------------------------------------------------
# protocol freeze
# ---------------------------------------------------------------------------


def protocol_freeze(
    bundle,
    reuse: Mapping[str, Any],
    groups: Mapping[str, Any],
    budget: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, Any]:
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
            "split_sizes": dict(bundle.split_sizes),
            "targets": {
                task: {"description": C.TASK_TARGET_NAME[task], "scaler": bundle.scaler(task)}
                for task in C.TASKS
            },
            "frozen_reuse": {
                "graph_cache_sha256": bundle.extra["graph_cache_sha256"],
                "lipo_raw_sha256": bundle.extra["lipo_raw_sha256"],
                "feature_spec_hash": bundle.extra["spec_hash"],
                "norm_fingerprint": bundle.extra["norm_fingerprint"],
                "wc_logp_label_sha256": bundle.extra["wc_logp_label_sha256"],
                "oracle_contrib_sha256": bundle.extra["oracle_contrib_sha256"],
            },
        },
        "frozen_reuse_check": {
            "routing_cache_sha256": dict(C.EXPECTED_ROUTING_CACHE_SHA256),
            "routing_fingerprint": dict(C.EXPECTED_ROUTING_FINGERPRINT),
            "all_pass": bool(reuse["all_pass"]),
            "rebuild_arrays_checked_per_split": True,
            "source_round": C.PRED_DIR.name,
        },
        "architecture": {
            "statement": (
                "the frozen C1 function is unchanged; only the 200 S2 columns of the local "
                "input are split into two role groups and rescaled by two trainable per-channel vectors"
            ),
            "c1_local_input_width": int(C.LOCAL_INPUT_DIM),
            "s2_role_dim": int(C.S2_ROLE_DIM),
            "s2_typed_dim": int(C.S2_TYPED_DIM),
            "s2_content_dim": int(C.S2_CONTENT_DIM),
            "gamma_init": "exactly one for every channel of both groups",
            "gamma_scope": "shared across all molecules and all occurrences of a group",
            "readout": "sum_v f(h_v) + rho([g, mean_v h_v, mean_v h^2_v, log1p n_v]) (unchanged)",
            "dropout": "one shared dropout(0.10) on the local representation (unchanged)",
            "role_grouping": {
                "RR_REAL": C.MODEL_ROLE_SOURCE["RR_REAL"],
                "RR_CONTROL": C.MODEL_ROLE_SOURCE["RR_CONTROL"],
                "control_salt": C.CONTROL_SALT,
                "control_hash": control["manifest"]["hash"],
                "control_table_sha256": control["manifest"]["sha256"],
            },
        },
        "role_groups": {
            split: {
                model: {
                    "n_group0": groups["splits"][split][model]["n_group0"],
                    "n_group1": groups["splits"][split][model]["n_group1"],
                    "fraction_group0": groups["splits"][split][model]["fraction_group0"],
                    "fraction_group1": groups["splits"][split][model]["fraction_group1"],
                }
                for model in C.MODELS
            }
            for split in C.SPLITS
        },
        "budget": budget,
        "recipe": T.recipe_record(),
        "decision_scales": {
            "retention_factor_lipo_valid": C.RETENTION_FACTOR,
            "bootstrap_draws": C.BOOTSTRAP_DRAWS,
            "bootstrap_seed": C.BOOTSTRAP_SEED,
            "increment_probability": C.INCREMENT_PROB,
            "increment_alpha": C.INCREMENT_ALPHA,
            "oracle_control_seed": C.ORACLE_CONTROL_SEED,
            "oracle_controls": [C.MIN_ORACLE_CONTROLS, C.MAX_ORACLE_CONTROLS],
            "oracle_positive_fraction": C.ORACLE_POSITIVE_FRACTION,
            "native_block_mask": (
                "an S2 occurrence survives a block mask iff neither of its two owning blocks "
                "is the masked block; score_i = |delta_i| / sqrt(max(1, n_bonds_i)); "
                "ties -> smallest block index"
            ),
        },
        "code_sha256": T.code_hashes(),
    }
    payload["freeze_hash"] = C.sha256_json({k: v for k, v in payload.items() if k != "freeze_hash"})
    return payload


def prepare_all(*, force: bool = False) -> dict[str, Any]:
    D._BUNDLE_CACHE.clear()
    # the CONTROL table is derived from the frozen split caches, so the base
    # bundle is built first and the table is frozen before the round bundle
    raw = C.modules()["rr_data"].build_bundle(verify=True)
    control = D.build_control_table(raw.caches, force=force)
    D._BUNDLE_CACHE.clear()
    bundle = D.build_bundle(verify=True)
    reuse = verify_reuse(bundle)
    groups = D.role_group_audit(bundle)
    budget = parameter_budget(bundle)
    freeze = protocol_freeze(bundle, reuse, groups, budget, control)
    C.ensure_dir(C.ANALYSIS_DIR)
    C.atomic_write_json(C.ANALYSIS_DIR / "frozen_reuse.json", reuse)
    C.atomic_write_json(C.ANALYSIS_DIR / "role_groups.json", groups)
    C.atomic_write_json(C.ANALYSIS_DIR / "parameter_budget.json", budget)
    C.atomic_write_json(C.ANALYSIS_DIR / "dataset_audit.json", D.target_audit(bundle))
    C.atomic_write_json(C.ANALYSIS_DIR / "protocol_freeze.json", freeze)
    return {
        "stage": "prepare",
        "reuse_pass": bool(reuse["all_pass"]),
        "groups_pass": bool(groups["all_pass"]),
        "parameter_ratio": budget["ratio"],
        "initial_state_identical": bool(budget["initial_state_identical"]),
        "freeze_hash": freeze["freeze_hash"],
    }


__all__ = ["verify_reuse", "parameter_budget", "protocol_freeze", "prepare_all"]
