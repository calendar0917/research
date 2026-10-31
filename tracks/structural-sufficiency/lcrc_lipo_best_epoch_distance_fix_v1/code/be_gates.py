"""CPU correctness gates (must pass before any formal GPU run).

The centrepiece is the distance-aggregation check: the corrected per-root
bucket means are compared against a hand-computed 6-atom chain and against an
independent loop-based reference implementation on real molecules.  The
historical aggregation is reconstructed as well, to document exactly what the
bug did (it wrote every bucket sum to the *target* atom, which collapses the
"context" to a scaled copy of the atom's own representation).

Everything else is the standard inventory: data/label alignment, Train3360-only
normalisation, shared initialisation, no cross-molecule mixing, atom-renumbering
and molecule-order invariance, real gradients, ``rho`` reading the original
``h``, dropout activity, best-epoch tracker semantics and the absence of any
Test access in the training path.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

import be_common as C
import be_data as D
import be_models as M
import be_train as T

import r42_metrics as MET  # noqa: E402
import r42_native_train as NT  # noqa: E402
import dc_data as DD  # noqa: E402
import x1_models as X1M  # noqa: E402
import dc_train as DCT  # noqa: E402

CHECK_TOL = 1.0e-6
CHAIN_H = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _chain_buckets() -> dict[str, Any]:
    """Hand-written D34/D5 layout for the chain 0-1-2-3-4-5."""
    d34 = {0: [3, 4], 1: [4, 5], 2: [5], 3: [0], 4: [0, 1], 5: [1, 2]}
    d5 = {0: [5], 1: [], 2: [], 3: [], 4: [], 5: [0]}
    counts34 = [len(d34[i]) for i in range(6)]
    counts5 = [len(d5[i]) for i in range(6)]
    return {
        "d34_idx": torch.as_tensor([j for i in range(6) for j in d34[i]], dtype=torch.long),
        "d5_idx": torch.as_tensor([j for i in range(6) for j in d5[i]], dtype=torch.long),
        "d34_counts": torch.as_tensor(counts34, dtype=torch.long),
        "d5_counts": torch.as_tensor(counts5, dtype=torch.long),
        "d34": d34,
        "d5": d5,
    }


def _chain_batch(atom_batch: torch.Tensor | None = None) -> dict[str, Any]:
    buckets = _chain_buckets()
    batch = {
        "d34_idx": buckets["d34_idx"],
        "d5_idx": buckets["d5_idx"],
        "d34_counts": buckets["d34_counts"],
        "d5_counts": buckets["d5_counts"],
        "atom_batch": torch.zeros(6, dtype=torch.long) if atom_batch is None else atom_batch,
    }
    return batch


def _buckets_match_reference(batch: Mapping[str, Any], ref: Mapping[str, Any]) -> bool:
    """Per-root set equality between batch bucket arrays and the BFS reference."""
    for prefix, key in (("d34", "d34"), ("d5", "d5")):
        counts = batch[f"{prefix}_counts"]
        if not torch.equal(counts, ref[f"{prefix}_counts"]):
            return False
        ptr = batch[f"{prefix}_ptr"]
        idx = batch[f"{prefix}_idx"]
        for root in range(int(counts.numel())):
            lo, hi = int(ptr[root]), int(ptr[root + 1])
            if sorted(idx[lo:hi].tolist()) != sorted(ref[key][root]):
                return False
    return True


def _records(checks: list[dict[str, Any]], gate_id: str, passed: bool, detail: Any) -> bool:
    checks.append({"id": gate_id, "passed": bool(passed), "detail": detail})
    return bool(passed)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def gate_data_identity(views) -> dict[str, Any]:
    split = views["split"]
    detail = {
        "split_file_sha256": C.sha256_file(C.V1_SPLIT_FILE),
        "labels_file_sha256": C.sha256_file(C.V1_LABELS_FILE),
        "split_sizes": {k: len(split["rows"][k]) for k in ("train", "valid", "test")},
        "row_hashes": {k: C.rows_sha256_int64(split["rows"][k]) for k in ("train", "valid", "test")},
        "cache_content_fingerprint": views["cache_built"]["content_fingerprint"],
        "train_norm_fingerprint": views["train_norm_fingerprint"],
        "test_label_vector_sha256": C.vector_sha256_float64(views["test"]["y_test"]),
    }
    passed = bool(
        detail["split_file_sha256"] == C.V1_SPLIT_SHA256
        and detail["labels_file_sha256"] == C.V1_LABELS_SHA256
        and detail["split_sizes"] == C.SPLIT_SIZES
        and detail["row_hashes"] == C.SPLIT_SHA256
        and detail["cache_content_fingerprint"] == C.LCRC_CACHE_CONTENT_FINGERPRINT
        and detail["train_norm_fingerprint"] == C.LCRC_TRAIN_NORM_FINGERPRINT
        and detail["test_label_vector_sha256"] == C.TEST_VECTOR_SHA256
    )
    return {"id": "data_identity", "passed": passed, "detail": detail}


def gate_row_alignment(views) -> dict[str, Any]:
    split = views["split"]
    rows = {k: [int(x) for x in split["rows"][k]] for k in ("train", "valid", "test")}
    cache_ids = {}
    smiles_ok = True
    for name in ("train", "valid", "test"):
        cache = views[f"{name}_cache"]
        cache_ids[name] = [int(x) for x in cache["molecule_ids"]]
        smiles_ok &= [str(x) for x in cache["smiles"]] == [str(split["smiles"][r]) for r in rows[name]]
    y_train_rows = [int(r) for r in split["rows"]["train"]]
    detail = {
        "cache_row_identity_matches_split": {k: cache_ids[k] == rows[k] for k in rows},
        "smiles_match": bool(smiles_ok),
        "cache_molecule_ids": cache_ids,
        "disjoint_splits": bool(
            set(rows["train"]).isdisjoint(rows["valid"])
            and set(rows["train"]).isdisjoint(rows["test"])
            and set(rows["valid"]).isdisjoint(rows["test"])
        ),
        "union_is_all_molecules": len(set(rows["train"]) | set(rows["valid"]) | set(rows["test"]))
        == C.N_MOLECULES,
        "label_lengths": {
            "train": int(views["labels"]["y_train"].size),
            "valid": int(views["labels"]["y_valid"].size),
            "test": int(views["test"]["y_test"].size),
        },
        "train_label_vector_sha256": C.vector_sha256_float64(views["labels"]["y_train"]),
        "valid_label_vector_sha256": C.vector_sha256_float64(views["labels"]["y_valid"]),
        "labels_ok": y_train_rows == rows["train"],
    }
    passed = bool(
        all(detail["cache_row_identity_matches_split"].values())
        and detail["smiles_match"]
        and detail["disjoint_splits"]
        and detail["union_is_all_molecules"]
        and detail["label_lengths"] == C.SPLIT_SIZES
        and detail["train_label_vector_sha256"] == C.TARGET_VECTOR_SHA256
        and detail["valid_label_vector_sha256"] == C.VALID_VECTOR_SHA256
        and detail["labels_ok"]
    )
    return {"id": "row_and_label_alignment", "passed": passed, "detail": detail}


def gate_target_norm_train_only(views) -> dict[str, Any]:
    y_train = np.asarray(views["labels"]["y_train"], dtype=np.float64)
    y_valid = np.asarray(views["labels"]["y_valid"], dtype=np.float64)
    y_test = np.asarray(views["test"]["y_test"], dtype=np.float64)
    mean = float(y_train.mean())
    scale = float(y_train.std(ddof=0))
    mean_all = float(np.concatenate([y_train, y_valid, y_test]).mean())
    detail = {
        "recomputed_train_mean": mean,
        "recomputed_train_scale": scale,
        "frozen_mean": C.TARGET_MEAN,
        "frozen_scale": C.TARGET_SCALE,
        "mean_if_all_splits_used": mean_all,
        "discriminating": bool(abs(mean_all - C.TARGET_MEAN) > 1e-6),
        "target_norm_recorded_source": views["target_norm"].get("fit_on"),
    }
    passed = bool(
        abs(mean - C.TARGET_MEAN) < 1e-12
        and abs(scale - C.TARGET_SCALE) < 1e-12
        and detail["discriminating"]
        and str(views["target_norm"].get("fit_on")) == "Train3360 only"
    )
    return {"id": "target_norm_train_only", "passed": passed, "detail": detail}


def gate_chain_hand_computed() -> dict[str, Any]:
    """The mandated 6-atom chain check of the aggregation content."""
    batch = _chain_batch()
    h = torch.as_tensor(CHAIN_H, dtype=torch.float64).unsqueeze(1)
    c34, c5 = D.bucket_means(h, batch["d34_idx"], batch["d5_idx"], batch["d34_counts"], batch["d5_counts"])
    expected34 = [12.5, 20.5, 25.0, 0.0, 0.5, 2.5]
    expected5 = [25.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    got34 = [float(x) for x in c34.squeeze(1)]
    got5 = [float(x) for x in c5.squeeze(1)]
    detail = {
        "chain": "0-1-2-3-4-5",
        "h_scalar_vector": CHAIN_H,
        "expected_c34": expected34,
        "expected_c5": expected5,
        "got_c34": got34,
        "got_c5": got5,
        "root0_D34": [3, 4],
        "root0_D5": [5],
        "root0_c34": got34[0],
        "root0_c5": got5[0],
        "empty_bucket_is_exact_zero": {
            "root1_D5": got5[1] == 0.0,
            "root2_D5": got5[2] == 0.0,
            "root3_D34": got34[3] == 0.0,
        },
    }
    passed = bool(
        got34 == expected34
        and got5 == expected5
        and got34[0] == 12.5
        and got5[0] == 25.0
        and all(detail["empty_bucket_is_exact_zero"].values())
    )
    return {"id": "distance_chain_hand_computed", "passed": passed, "detail": detail}


def gate_aggregation_matches_reference(views) -> dict[str, Any]:
    """Corrected aggregation vs an independent loop implementation."""
    batch = D.collate(views["valid_cache"], list(range(16)))
    torch.manual_seed(0)
    h = torch.randn(int(batch["atom_batch"].numel()), C.WIDTH, dtype=torch.float64)
    c34, c5 = D.bucket_means(h, batch["d34_idx"], batch["d5_idx"], batch["d34_counts"], batch["d5_counts"])
    ref34, ref5 = D.reference_bucket_means(batch, h.numpy())
    ref = D.reference_buckets(batch)
    idx_ok = bool(
        torch.equal(batch["d34_idx"], ref["d34_targets"])
        and torch.equal(batch["d5_idx"], ref["d5_targets"])
        and torch.equal(batch["d34_counts"], ref["d34_counts"])
        and torch.equal(batch["d5_counts"], ref["d5_counts"])
    )
    d34 = float(np.abs(c34.numpy() - ref34).max())
    d5 = float(np.abs(c5.numpy() - ref5).max())
    detail = {
        "n_atoms": int(batch["atom_batch"].numel()),
        "n_d34_entries": int(batch["d34_idx"].numel()),
        "n_d5_entries": int(batch["d5_idx"].numel()),
        "bucket_index_arrays_match_reference": idx_ok,
        "max_abs_diff_c34": d34,
        "max_abs_diff_c5": d5,
        "tolerance": CHECK_TOL,
    }
    passed = bool(idx_ok and d34 <= CHECK_TOL and d5 <= CHECK_TOL)
    return {"id": "distance_aggregation_matches_reference", "passed": passed, "detail": detail}


def gate_historical_bug_is_self_representation(views) -> dict[str, Any]:
    """Document the historical bug numerically (root vs target aggregation)."""
    batch = D.collate(views["valid_cache"], list(range(8)))
    torch.manual_seed(1)
    h = torch.randn(int(batch["atom_batch"].numel()), C.WIDTH, dtype=torch.float64)
    n_atoms = int(h.shape[0])
    # Historical (buggy) aggregation: sums written to the *target* rows.
    buggy34 = h.new_zeros((n_atoms, C.WIDTH)).index_add_(0, batch["d34_idx"], h.index_select(0, batch["d34_idx"]))
    buggy34 = buggy34 / batch["d34_counts"].clamp(min=1).unsqueeze(1)
    in_deg34 = h.new_zeros(n_atoms).index_add_(
        0, batch["d34_idx"], torch.ones_like(batch["d34_idx"], dtype=h.dtype)
    )
    ratio = in_deg34 / batch["d34_counts"].clamp(min=1).to(h.dtype)
    predicted_buggy = h * ratio.unsqueeze(1)
    degenerate = float((buggy34 - predicted_buggy).abs().max())
    c34, _c5 = D.bucket_means(h, batch["d34_idx"], batch["d5_idx"], batch["d34_counts"], batch["d5_counts"])
    fixed_vs_buggy = float((c34 - buggy34).abs().max())
    # Correlation of the fixed context with the root atom's own representation.
    mask = batch["d34_counts"] > 0
    hh = h[mask].numpy()
    cc = c34[mask].numpy()
    corr = float(np.corrcoef(hh.ravel(), cc.ravel())[0, 1])
    bb = buggy34[mask].numpy()
    corr_buggy = float(np.corrcoef(hh.ravel(), bb.ravel())[0, 1])
    detail = {
        "n_atoms": n_atoms,
        "max_abs_buggy_minus_scaled_self": degenerate,
        "max_abs_fixed_minus_buggy": fixed_vs_buggy,
        "correlation_fixed_context_with_own_h": corr,
        "correlation_historical_context_with_own_h": corr_buggy,
        "note": (
            "historical c34 row i equals h_i * (#roots whose bucket contains i)/|D34_i|, "
            "i.e. a rescaled copy of the atom's own representation"
        ),
    }
    passed = bool(degenerate <= 1e-12 and fixed_vs_buggy > 1e-2 and corr_buggy > 0.999 and abs(corr) < corr_buggy)
    return {"id": "historical_bug_degenerates_to_self", "passed": passed, "detail": detail}


def gate_module_empty_bucket(views) -> dict[str, Any]:
    """No bucket content -> exactly zero correction, even with a non-zero bias."""
    spec, norm = views["spec"], views["train_norm"]
    model, _meta = M.build_family("dist_true_fixed", spec, norm, 0, "cpu")
    with torch.no_grad():
        model.dist.W_out.bias.fill_(0.5)
    batch = _chain_batch()
    batch = dict(batch)
    batch["d34_idx"] = torch.zeros(0, dtype=torch.long)
    batch["d5_idx"] = torch.zeros(0, dtype=torch.long)
    batch["d34_counts"] = torch.zeros(6, dtype=torch.long)
    batch["d5_counts"] = torch.zeros(6, dtype=torch.long)
    h = torch.randn(6, C.WIDTH)
    change, ctx = model.dist(h, batch, 1, "true")
    zeros_ok = bool((change == 0).all() and (ctx["c34"] == 0).all() and (ctx["c5"] == 0).all())
    # Chain case: existing buckets but empty D5 for roots 1..4.
    batch2 = _chain_batch()
    h2 = torch.randn(6, C.WIDTH)
    _change2, ctx2 = model.dist(h2, batch2, 1, "true")
    empty5_ok = bool((ctx2["c5"][1:5] == 0).all() and (ctx2["p5"][1:5] == 0).all())
    detail = {
        "all_buckets_empty_change_exactly_zero": zeros_ok,
        "empty_d5_roots_have_zero_c5_and_p5": empty5_ok,
        "c34_chain": ctx2["c34"].sum().item(),
    }
    return {"id": "distance_empty_bucket_zero", "passed": bool(zeros_ok and empty5_ok), "detail": detail}


def gate_no_cross_molecule(views) -> dict[str, Any]:
    spec, norm = views["spec"], views["train_norm"]
    model, _meta = M.build_family("dist_true_fixed", spec, norm, 0, "cpu")
    model.eval()
    rows = [3, 11, 42, 77]
    batch = D.collate(views["valid_cache"], rows)
    with torch.no_grad():
        y_batch, aux_batch = model(batch, return_aux=True)
        per_graph = []
        max_pred = 0.0
        max_ctx34 = 0.0
        max_ctx5 = 0.0
        atom_batch = batch["atom_batch"]
        for g, row in enumerate(rows):
            single = D.collate(views["valid_cache"], [row])
            y_single, aux_single = model(single, return_aux=True)
            mask = atom_batch == g
            d_pred = float((y_batch[g] - y_single[0]).abs().max())
            d34 = float((aux_batch["c34"][mask] - aux_single["c34"]).abs().max())
            d5 = float((aux_batch["c5"][mask] - aux_single["c5"]).abs().max())
            max_pred = max(max_pred, d_pred)
            max_ctx34 = max(max_ctx34, d34)
            max_ctx5 = max(max_ctx5, d5)
            per_graph.append({"row": row, "d_pred": d_pred, "d_c34": d34, "d_c5": d5})
    # Perturb molecule 0's atom features; molecule 1 must not move at all.
    feat = batch["atom_feat"].clone()
    feat[atom_batch == 0] = 0.0
    perturbed = dict(batch)
    perturbed["atom_feat"] = feat
    with torch.no_grad():
        y_pert, aux_pert = model(perturbed, return_aux=True)
    mask1 = atom_batch == 1
    d_pred_other = float((y_batch[1] - y_pert[1]).abs().max())
    d_ctx_other = float((aux_batch["c34"][mask1] - aux_pert["c34"][mask1]).abs().max())
    detail = {
        "rows": rows,
        "max_prediction_diff_single_vs_batch": max_pred,
        "max_c34_diff_single_vs_batch": max_ctx34,
        "max_c5_diff_single_vs_batch": max_ctx5,
        "other_molecule_prediction_diff_after_perturbation": d_pred_other,
        "other_molecule_c34_diff_after_perturbation": d_ctx_other,
        "tolerance": CHECK_TOL,
        "per_graph": per_graph,
    }
    passed = bool(
        max(max_pred, max_ctx34, max_ctx5, d_pred_other, d_ctx_other) <= CHECK_TOL
    )
    return {"id": "distance_no_cross_molecule_mixing", "passed": passed, "detail": detail}


def gate_permutation_and_order(views) -> dict[str, Any]:
    spec, norm = views["spec"], views["train_norm"]
    model, _meta = M.build_family("dist_true_fixed", spec, norm, 0, "cpu")
    model.eval()
    rows = [5, 19, 33, 64, 81]
    batch = D.collate(views["valid_cache"], rows)
    generator = torch.Generator().manual_seed(1234)
    permuted = D.permute_batch_atoms(batch, generator)
    with torch.no_grad():
        y = model(batch).detach().numpy().reshape(-1)
        y_perm = model(permuted).detach().numpy().reshape(-1)
    perm_diff = float(np.abs(y - y_perm).max())
    reordered = D.collate(views["valid_cache"], list(reversed(rows)))
    with torch.no_grad():
        y_rev = model(reordered).detach().numpy().reshape(-1)
    order_diff = float(np.abs(y - y_rev[::-1]).max())
    detail = {
        "rows": rows,
        "atom_permutation_max_abs_diff": perm_diff,
        "molecule_order_max_abs_diff": order_diff,
        "tolerance_atoms": 1.0e-5,
        "tolerance_order": 1.0e-6,
    }
    passed = bool(perm_diff <= 1.0e-5 and order_diff <= 1.0e-6)
    return {"id": "distance_atom_permutation_and_order_invariance", "passed": passed, "detail": detail}


def gate_permutation_utility_correctness(views) -> dict[str, Any]:
    """The frozen permutation helper is inconsistent; the local one is not."""
    rows = [6, 14, 22]
    batch = D.collate(views["valid_cache"], rows)
    gen_frozen = torch.Generator().manual_seed(2026)
    gen_local = torch.Generator().manual_seed(2026)
    frozen_permuted = DD.permute_batch_atoms(batch, gen_frozen)
    local_permuted = D.permute_batch_atoms(batch, gen_local)
    ref_frozen = D.reference_buckets(frozen_permuted)
    ref_local = D.reference_buckets(local_permuted)
    frozen_consistent = _buckets_match_reference(frozen_permuted, ref_frozen)
    local_consistent = _buckets_match_reference(local_permuted, ref_local)
    detail = {
        "rows": rows,
        "historical_dc_data_permutation_consistent_with_atom_order": frozen_consistent,
        "local_permutation_consistent_with_atom_order": local_consistent,
        "note": (
            "dc_data.permute_batch_atoms re-draws the permutation for the bucket index rebuild "
            "(the frozen helper's generator advances between the two draws); this round uses the "
            "single-draw version in be_data.permute_batch_atoms. Training is unaffected: the helper "
            "is only used by invariance audits."
        ),
    }
    return {"id": "permutation_utility_is_consistent", "passed": bool(local_consistent), "detail": detail}


def gate_gradient_to_true_targets(views) -> dict[str, Any]:
    """Gradient must reach the real targets, and only the containing roots."""
    batch = D.collate(views["valid_cache"], list(range(4)))
    n_atoms = int(batch["atom_batch"].numel())
    torch.manual_seed(7)
    h = torch.randn(n_atoms, C.WIDTH, dtype=torch.float64, requires_grad=True)
    c34, c5 = D.bucket_means(h, batch["d34_idx"], batch["d5_idx"], batch["d34_counts"], batch["d5_counts"])
    (c34.sum() + c5.sum()).backward()
    grad = h.grad.detach()
    ref = D.reference_buckets(batch)
    # atom j receives gradient iff some root bucket contains j
    in_any34 = torch.zeros(n_atoms, dtype=torch.bool)
    in_any5 = torch.zeros(n_atoms, dtype=torch.bool)
    for i, bucket in enumerate(ref["d34"]):
        if bucket:
            in_any34[torch.as_tensor(bucket, dtype=torch.long)] = True
    for i, bucket in enumerate(ref["d5"]):
        if bucket:
            in_any5[torch.as_tensor(bucket, dtype=torch.long)] = True
    expected_nonzero = in_any34 | in_any5
    got_nonzero = grad.abs().sum(dim=1) > 0
    pattern_ok = bool(torch.equal(expected_nonzero, got_nonzero))
    # Module level: gradients must reach the distance parameters.
    spec, norm = views["spec"], views["train_norm"]
    model, _meta = M.build_family("dist_true_fixed", spec, norm, 0, "cpu")
    model.train()
    b = M.to_device(D.collate(views["valid_cache"], list(range(8))), "cpu")
    out, aux = model(b, return_aux=True)
    (out.sum() + aux["change"].pow(2).sum()).backward()
    param_grads = {
        name: float(p.grad.detach().norm()) if p.grad is not None else 0.0
        for name, p in model.dist.named_parameters()
    }
    detail = {
        "n_atoms": n_atoms,
        "n_atoms_expected_nonzero_grad": int(expected_nonzero.sum()),
        "n_atoms_with_nonzero_grad": int(got_nonzero.sum()),
        "grad_support_matches_bucket_membership": pattern_ok,
        "max_abs_grad": float(grad.abs().max()),
        "module_parameter_gradient_norms": param_grads,
    }
    passed = bool(pattern_ok and float(grad.abs().max()) > 0.0 and all(v > 0 for v in param_grads.values()))
    return {"id": "distance_gradient_reaches_true_target_atoms", "passed": passed, "detail": detail}


def gate_target_perturbation(views) -> dict[str, Any]:
    """Changing h_j changes exactly the roots whose bucket contains j."""
    batch = D.collate(views["valid_cache"], list(range(3)))
    n_atoms = int(batch["atom_batch"].numel())
    torch.manual_seed(11)
    h = torch.randn(n_atoms, C.WIDTH, dtype=torch.float64)
    c34, c5 = D.bucket_means(h, batch["d34_idx"], batch["d5_idx"], batch["d34_counts"], batch["d5_counts"])
    ref = D.reference_buckets(batch)
    tested = 0
    mismatches = []
    for j in range(min(n_atoms, 24)):
        h2 = h.clone()
        h2[j] += 1.0
        c34b, c5b = D.bucket_means(h2, batch["d34_idx"], batch["d5_idx"], batch["d34_counts"], batch["d5_counts"])
        changed34 = ((c34b - c34).abs().sum(dim=1) > 0).tolist()
        expected34 = [j in bucket for bucket in ref["d34"]]
        changed5 = ((c5b - c5).abs().sum(dim=1) > 0).tolist()
        expected5 = [j in bucket for bucket in ref["d5"]]
        tested += 1
        if changed34 != expected34 or changed5 != expected5:
            mismatches.append({"atom": j, "changed34": changed34, "expected34": expected34, "changed5": changed5, "expected5": expected5})
    detail = {
        "n_atoms": n_atoms,
        "n_atoms_tested": tested,
        "mismatches": mismatches,
    }
    passed = bool(tested > 0 and not mismatches)
    return {"id": "distance_target_perturbation_locality", "passed": passed, "detail": detail}


def gate_shared_init(views) -> dict[str, Any]:
    spec, norm = views["spec"], views["train_norm"]
    init_hashes: dict[str, dict[str, str]] = {}
    twin_ok = {}
    for seed in C.SEEDS:
        base, base_meta = M.build_family("base", spec, norm, seed, "cpu")
        per_seed = {"base": base_meta["init_state_hash"]}
        for family in C.FAMILIES:
            if family == "base":
                continue
            model, meta = M.build_family(family, spec, norm, seed, "cpu")
            audit = X1M.shared_init_audit(base, model)
            if not audit["all_shared_bitwise_identical"]:
                raise RuntimeError(f"{family}/{seed}: shared init mismatch {audit['mismatched']}")
            per_seed[family] = meta["init_state_hash"]
        true_model, _true_meta = M.build_family("dist_true_fixed", spec, norm, seed, "cpu")
        blind_model, _blind_meta = M.build_family("dist_blind", spec, norm, seed, "cpu")
        true_state = true_model.state_dict()
        blind_state = blind_model.state_dict()
        identical = set(true_state) == set(blind_state) and all(
            torch.equal(true_state[k], blind_state[k]) for k in true_state
        )
        twin_ok[str(seed)] = bool(identical)
        init_hashes[str(seed)] = per_seed
    # Different seeds must produce different initialisations.
    distinct_seeds = len({init_hashes[str(s)]["base"] for s in C.SEEDS}) == len(C.SEEDS)
    detail = {
        "shared_init_bitwise_identical_for_every_family_and_seed": True,
        "init_state_hashes": init_hashes,
        "dist_true_fixed_equals_dist_blind_state_dict": twin_ok,
        "distinct_seeds_have_distinct_init": bool(distinct_seeds),
    }
    passed = bool(all(twin_ok.values()) and distinct_seeds)
    return {"id": "shared_init_and_distance_twin", "passed": passed, "detail": detail}


def gate_zero_correction_recovers_base(views) -> dict[str, Any]:
    spec, norm = views["spec"], views["train_norm"]
    rows = list(range(12))
    batch = M.to_device(D.collate(views["valid_cache"], rows), "cpu")
    base, _ = M.build_family("base", spec, norm, 0, "cpu")
    base.eval()
    with torch.no_grad():
        y_base = base(batch).detach().numpy().reshape(-1)
    per_family = {}
    for family in C.FAMILIES:
        if family == "base":
            continue
        model, _meta = M.build_family(family, spec, norm, 0, "cpu")
        out = M.correction_output(model, family)
        with torch.no_grad():
            out.weight.zero_()
            out.bias.zero_()
        model.eval()
        with torch.no_grad():
            y = model(batch).detach().numpy().reshape(-1)
        per_family[family] = float(np.abs(y - y_base).max())
    detail = {"rows": rows, "max_abs_diff_vs_base": per_family, "tolerance": 1.0e-6}
    passed = bool(all(v <= 1e-6 for v in per_family.values()))
    return {"id": "zero_correction_recovers_base", "passed": passed, "detail": detail}


def gate_rho_and_dropout(views) -> dict[str, Any]:
    spec, norm = views["spec"], views["train_norm"]
    rows = list(range(8))
    per_family = {}
    for family in ("roleblind", "dist_true_fixed"):
        model, _meta = M.build_family(family, spec, norm, 0, "cpu")
        per_family[family] = {
            "rho": DCT.rho_input_audit(model, views["valid_cache"], rows),
            "dropout": DCT.dropout_activity_audit(model, views["valid_cache"], rows),
        }
    passed = bool(
        all(v["rho"]["passed"] and v["dropout"]["passed"] for v in per_family.values())
    )
    return {"id": "rho_reads_original_h_and_dropout_active", "passed": passed, "detail": per_family}


def gate_no_test_access() -> dict[str, Any]:
    views = D.load_views(include_test=False)
    sources = {
        name: (C.CODE_DIR / name).read_text(encoding="utf-8")
        for name in ("be_train.py", "be_models.py", "be_data.py")
    }
    banned = ("y_test", "test_cache", "rows_test", 'views["test"]', "include_test=True")
    hits = {
        name: [token for token in banned if token in text]
        for name, text in sources.items()
        if name in ("be_train.py", "be_models.py")
    }
    detail = {
        "views_test_key_is_none": views.get("test") is None,
        "labels_have_no_test_key": "test" not in views["labels"] and "y_test" not in views["labels"],
        "forbidden_tokens_in_training_modules": hits,
        "target_norm_valid_or_test_used": views["target_norm"].get("valid_or_test_used"),
        "cache_labels_present": views["cache_built"].get("labels_present"),
    }
    passed = bool(
        detail["views_test_key_is_none"]
        and detail["labels_have_no_test_key"]
        and not any(hits.values())
        and detail["target_norm_valid_or_test_used"] is False
        and detail["cache_labels_present"] is False
    )
    return {"id": "no_test_access_in_training_path", "passed": passed, "detail": detail}


def gate_metric_arithmetic() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    y = rng.normal(size=97)
    pred = y + rng.normal(scale=0.3, size=97)
    m = MET.all_metrics(pred, y)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mae = float(np.mean(np.abs(pred - y)))
    r2 = float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    detail = {
        "module": {"rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"]},
        "independent": {"rmse": rmse, "mae": mae, "r2": r2},
        "max_abs_diff": max(abs(m["rmse"] - rmse), abs(m["mae"] - mae), abs(m["r2"] - r2)),
    }
    passed = bool(detail["max_abs_diff"] <= 1e-12)
    return {"id": "metric_arithmetic", "passed": passed, "detail": detail}


def gate_best_epoch_tracker() -> dict[str, Any]:
    """Tie keeps the earlier epoch; only strict improvements replace the state."""
    model = torch.nn.Linear(1, 1, bias=False)
    tracker = T.BestEpochTracker()
    sequence = [(1, 0.60, 0.50), (2, 0.55, 0.45), (3, 0.55, 0.40), (4, 0.58, 0.42)]
    recorded = []
    for epoch, rmse, mae in sequence:
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        improved = tracker.update(epoch, rmse, mae, model)
        recorded.append({"epoch": epoch, "rmse": rmse, "improved": bool(improved)})
    tie_ok = tracker.best_epoch == 2
    state_ok = bool(float(tracker.state["weight"]) == 2.0)
    mae_at_best = tracker.best_mae == 0.45
    detail = {
        "sequence": sequence,
        "improvements": recorded,
        "best_epoch": tracker.best_epoch,
        "best_rmse": tracker.best_rmse,
        "best_mae": tracker.best_mae,
        "tie_keeps_earlier_epoch": bool(tie_ok),
        "state_is_best_epoch_state": state_ok,
        "mae_is_the_recorded_epoch_mae": bool(mae_at_best),
    }
    passed = bool(tie_ok and state_ok and mae_at_best)
    return {"id": "best_epoch_tracker_semantics", "passed": passed, "detail": detail}


def gate_best_epoch_end_to_end(views) -> dict[str, Any]:
    """A tiny real training run must save the argmin epoch, not the last epoch."""
    cfg = C.load_config()
    out_root = C.SMOKE_CPU / "best_epoch_audit"
    out_root.mkdir(parents=True, exist_ok=True)
    payload = T.train_arm(
        arm="dist_true_fixed_s0",
        cfg=cfg,
        views=views,
        device=torch.device("cpu"),
        run_dir=out_root,
        limit_train=32,
        limit_valid=32,
        max_epochs=4,
        pool="local",
        smoke=True,
    )
    ckpt = torch.load(out_root / "checkpoint.pt", map_location="cpu", weights_only=False)
    curve = payload["curve"]
    valid = np.asarray([row["valid_rmse_raw"] for row in curve])
    argmin = int(np.argmin(valid)) + 1
    detail = {
        "epochs_run": len(curve),
        "valid_curve": valid.tolist(),
        "argmin_epoch": argmin,
        "checkpoint_best_epoch": int(ckpt["best_epoch"]),
        "checkpoint_best_valid_rmse": float(ckpt["best_valid_rmse"]),
        "recheck_abs_diff": float(payload["checkpoint"]["valid_rmse_recheck_abs_diff"]),
        "checkpoint_state_hash": ckpt["best_state_hash"],
        "final_epoch_state_hash": ckpt["final_epoch_state_hash"],
        "checkpoint_is_final_epoch": bool(ckpt["checkpoint_is_final_epoch"]),
        "soup_members": ckpt["soup_members"],
        "n_improvements": int(payload["checkpoint"]["n_improvements"]),
        "improvement_epochs": payload["checkpoint"]["improvement_epochs"],
    }
    passed = bool(
        int(ckpt["best_epoch"]) == argmin
        and abs(float(ckpt["best_valid_rmse"]) - float(valid[argmin - 1])) == 0.0
        and float(payload["checkpoint"]["valid_rmse_recheck_abs_diff"]) == 0.0
        and ckpt["soup_members"] is None
    )
    return {"id": "best_epoch_end_to_end", "passed": passed, "detail": detail}


def gate_distance_coverage(views) -> dict[str, Any]:
    train = D.distance_bucket_coverage(views["train_cache"])
    valid = D.distance_bucket_coverage(views["valid_cache"])
    detail = {"train": train, "valid": valid}
    passed = bool(
        train["n_atoms"] > 0
        and valid["n_atoms"] > 0
        and train["fraction_d34_nonempty"] > 0.5
        and train["fraction_d5_nonempty"] > 0.5
        and train["n_d5_entries"] > 0
    )
    return {"id": "distance_bucket_coverage", "passed": passed, "detail": detail}


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def run_gates() -> dict[str, Any]:
    views = D.load_views(include_test=True)
    checks: list[dict[str, Any]] = []
    checks.append(gate_data_identity(views))
    checks.append(gate_row_alignment(views))
    checks.append(gate_target_norm_train_only(views))
    checks.append(gate_chain_hand_computed())
    checks.append(gate_aggregation_matches_reference(views))
    checks.append(gate_historical_bug_is_self_representation(views))
    checks.append(gate_module_empty_bucket(views))
    checks.append(gate_no_cross_molecule(views))
    checks.append(gate_permutation_and_order(views))
    checks.append(gate_permutation_utility_correctness(views))
    checks.append(gate_gradient_to_true_targets(views))
    checks.append(gate_target_perturbation(views))
    checks.append(gate_shared_init(views))
    checks.append(gate_zero_correction_recovers_base(views))
    checks.append(gate_rho_and_dropout(views))
    checks.append(gate_no_test_access())
    checks.append(gate_metric_arithmetic())
    checks.append(gate_best_epoch_tracker())
    checks.append(gate_distance_coverage(views))
    # Last: it trains a few epochs on a tiny subset.
    checks.append(gate_best_epoch_end_to_end(views))
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "config_hash": C.sha256_json(C.load_config()),
        "code_hashes": C.code_file_hashes(),
        "n_gates": len(checks),
        "n_passed": int(sum(1 for c in checks if c["passed"])),
        "ok": bool(all(c["passed"] for c in checks)),
        "checks": checks,
        "environment": C.env_info(),
    }
    C.write_json(C.CPU_GATES, payload)
    return payload


__all__ = ["run_gates"]
