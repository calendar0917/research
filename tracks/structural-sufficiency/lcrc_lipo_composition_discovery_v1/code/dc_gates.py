"""Required CPU gates for the composition-discovery round.

The gates run before any formal GPU training and are re-checked by the
training entry point through the protocol freeze.  They are intentionally
mechanical: split/label identity, shared initialisation, parameter counts,
no-correction BASE recovery, distance semantics, S2 path semantics,
BIND/UNBOUND invariants, batch/renumbering invariance, cross-molecule
isolation, finite non-zero gradients, original-``h`` readout, dropout
execution, frozen-baseline reproduction and inference immutability.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import dc_common as C
import dc_data as D
import dc_models as M

import lcrc_data  # noqa: E402
import molecular_graph as MG  # noqa: E402
import r42_data as R42D  # noqa: E402
import r42_metrics as MET  # noqa: E402
import x1_models as X1M  # noqa: E402


def _gate(checks: list[dict[str, Any]], gate_id: str, name: str, fn: Callable[[], tuple[bool, dict[str, Any]]]) -> None:
    try:
        passed, details = fn()
    except Exception as exc:  # noqa: BLE001
        passed, details = False, {"error": f"{type(exc).__name__}: {exc}"}
    checks.append({"id": gate_id, "name": name, "passed": bool(passed), "details": details})
    C.log(f"[gate] {gate_id}: {'PASS' if passed else 'FAIL'}")


def _tensor_stats(x: torch.Tensor) -> dict[str, float]:
    return {"abs_max": float(x.abs().max()) if x.numel() else 0.0, "abs_mean": float(x.abs().mean()) if x.numel() else 0.0}


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def gate_data(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    split = views["split"]
    labels = views["labels"]
    test = views["test"]
    rows = {name: [int(x) for x in split["rows"][name]] for name in ("train", "valid", "test")}
    hashes = {name: C.rows_sha256_int64(values) for name, values in rows.items()}
    details = {
        "row_hashes": hashes,
        "row_hashes_expected": dict(C.SPLIT_SHA256),
        "first_five": {name: values[:5] for name, values in rows.items()},
        "sizes": {name: len(values) for name, values in rows.items()},
        "disjoint": not (set(rows["train"]) & set(rows["valid"]))
        and not (set(rows["train"]) & set(rows["test"]))
        and not (set(rows["valid"]) & set(rows["test"])),
        "union_is_4200": set(rows["train"]) | set(rows["valid"]) | set(rows["test"]) == set(range(C.N_MOLECULES)),
        "labels": dict(labels["audit"]),
        "test_labels": dict(test["audit"]),
        "cache": {
            "path": C.track_rel(views["cache_built"]["path"]),
            "file_sha256": views["cache_built"]["file_sha256"],
            "content_fingerprint": views["cache_built"]["content_fingerprint"],
            "labels_present": False,
        },
        "train_norm_fingerprint": views["train_norm_fingerprint"],
        "target_norm": views["target_norm"],
    }
    ok = bool(
        hashes == C.SPLIT_SHA256
        and details["first_five"] == C.SPLIT_FIRST_ROWS
        and details["sizes"] == C.SPLIT_SIZES
        and details["disjoint"]
        and details["union_is_4200"]
        and labels["audit"]["train_vector_sha256"] == C.TARGET_VECTOR_SHA256
        and labels["audit"]["valid_vector_sha256"] == C.VALID_VECTOR_SHA256
        and test["audit"]["label_vector_sha256"] == C.TEST_VECTOR_SHA256
        and views["cache_built"]["content_fingerprint"] == C.LCRC_CACHE_CONTENT_FINGERPRINT
        and views["train_norm_fingerprint"] == C.LCRC_TRAIN_NORM_FINGERPRINT
    )
    return ok, details


def gate_target_norm(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    y_train = np.asarray(views["labels"]["y_train"], dtype=np.float64)
    y_valid = np.asarray(views["labels"]["y_valid"], dtype=np.float64)
    recomputed = R42D.fit_target_scaler(y_train)
    valid_only = R42D.fit_target_scaler(y_valid)
    details = {
        "train_mean": float(y_train.mean()),
        "train_scale": float(y_train.std()),
        "frozen_mean": C.TARGET_MEAN,
        "frozen_scale": C.TARGET_SCALE,
        "regenerated": recomputed,
        "valid_only_scaler": valid_only,
        "valid_only_differs_from_train": bool(
            abs(float(valid_only["mean"]) - C.TARGET_MEAN) > 1e-9 or abs(float(valid_only["scale"]) - C.TARGET_SCALE) > 1e-9
        ),
        "used_split": "Train3360 only",
    }
    ok = bool(
        abs(float(recomputed["mean"]) - C.TARGET_MEAN) <= 1e-12
        and abs(float(recomputed["scale"]) - C.TARGET_SCALE) <= 1e-12
        and details["valid_only_differs_from_train"]
    )
    return ok, details


def gate_shared_init(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    spec, norm = views["spec"], views["train_norm"]
    base, base_meta = M.build_family("base", spec, norm, 0, "cpu")
    context, _b, context_meta = X1M.build_context(spec, norm, 0, "cpu")
    newly: dict[str, Any] = {}
    expected_params = {
        "base": C.LCRC_BASE_PARAMS_REFERENCE,
        "g2l_all": C.CONTEXT_TOTAL_PARAMS_REFERENCE,
        "g2l_loo": C.CONTEXT_TOTAL_PARAMS_REFERENCE,
        "g2l_local_cap": C.CONTEXT_TOTAL_PARAMS_REFERENCE,
        "g2l_scalar": C.LCRC_BASE_PARAMS_REFERENCE + 3 * (C.WIDTH * C.CORR_DIM + C.CORR_DIM) + (C.CORR_DIM + 1),
        "dist_true": C.CONTEXT_TOTAL_PARAMS_REFERENCE,
        "dist_blind": C.CONTEXT_TOTAL_PARAMS_REFERENCE,
        "path_bind": C.LCRC_BASE_PARAMS_REFERENCE + (16 + 5 + 5 + 2) * C.CORR_DIM + (C.ATOM_OUT * C.CORR_DIM + C.CORR_DIM) + (C.WIDTH * C.CORR_DIM + C.CORR_DIM) + (C.CORR_DIM * C.WIDTH + C.WIDTH),
        "path_unbound": C.LCRC_BASE_PARAMS_REFERENCE + (16 + 5 + 5 + 2) * C.CORR_DIM + (C.ATOM_OUT * C.CORR_DIM + C.CORR_DIM) + (C.WIDTH * C.CORR_DIM + C.CORR_DIM) + (C.CORR_DIM * C.WIDTH + C.WIDTH),
    }
    hist_base = C.read_json(C.FROZEN_CHECKPOINTS["base_s0"]["train_json"])["model_audit"]["init_state_hash"]
    hist_context = C.read_json(C.V2_MECH / "analysis/runs/roleblind_s0/train.json")["model_audit"]["init_state_hash"]
    for family in C.NEW_FAMILIES:
        model, meta = M.build_family(family, spec, norm, 0, "cpu")
        shared = meta.get("shared_init", {})
        entry = {
            "parameters": meta["parameters"],
            "parameters_expected": expected_params[family],
            "parameters_new": meta.get("parameters_new"),
            "init_state_hash": meta["init_state_hash"],
            "base_init_state_hash": meta.get("base_init_state_hash"),
            "shared_init_all_bitwise_identical": bool(shared.get("all_shared_bitwise_identical", False)),
            "shared_init_mismatched": shared.get("mismatched"),
        }
        if family in ("g2l_all", "g2l_loo", "g2l_local_cap"):
            entry["context_init_state_hash"] = meta.get("context_init_state_hash")
            entry["init_equals_context"] = bool(
                meta["init_state_hash"] == context_meta["init_state_hash"]
                and meta.get("init_parity_with_context", {}).get("all_entries_bitwise_identical")
            )
        newly[family] = entry
    pair_checks = {}
    for a, b in (("dist_true", "dist_blind"), ("path_bind", "path_unbound")):
        ma, _ = M.build_family(a, spec, norm, 0, "cpu")
        mb, _ = M.build_family(b, spec, norm, 0, "cpu")
        sa, sb = ma.state_dict(), mb.state_dict()
        pair_checks[f"{a}_vs_{b}"] = {
            "same_keys": set(sa) == set(sb),
            "all_entries_bitwise_identical": all(torch.equal(sa[k], sb[k]) for k in sa),
            "same_parameter_count": M.parameter_count(ma) == M.parameter_count(mb),
            "parameter_count": M.parameter_count(ma),
        }
    details = {
        "base_parameters": base_meta["parameters"],
        "base_init_state_hash": base_meta["init_state_hash"],
        "historical_base_s0_init_state_hash": hist_base,
        "fresh_base_matches_historical": base_meta["init_state_hash"] == hist_base,
        "historical_context_s0_init_state_hash": hist_context,
        "fresh_context_s0_init_state_hash": context_meta["init_state_hash"],
        "fresh_context_matches_historical": context_meta["init_state_hash"] == hist_context,
        "families": newly,
        "pairs": pair_checks,
    }
    ok = bool(
        details["fresh_base_matches_historical"]
        and details["fresh_context_matches_historical"]
        and base_meta["parameters"] == C.LCRC_BASE_PARAMS_REFERENCE
        and all(v["parameters"] == v["parameters_expected"] for v in newly.values())
        and all(v["shared_init_all_bitwise_identical"] for v in newly.values())
        and all(v.get("init_equals_context", True) for v in newly.values())
        and all(v["same_keys"] and v["all_entries_bitwise_identical"] and v["same_parameter_count"] for v in pair_checks.values())
    )
    return ok, details


def gate_no_correction(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    spec, norm = views["spec"], views["train_norm"]
    batch = M.to_device(D.collate(views["valid_cache"], list(range(16))), "cpu")
    base, _ = M.build_family("base", spec, norm, 0, "cpu")
    base.eval()
    with torch.no_grad():
        y_base = base(batch)
    results = {}
    for family in C.NEW_FAMILIES:
        model, _ = M.build_family(family, spec, norm, 0, "cpu")
        model.eval()
        with torch.no_grad():
            y_init = model(batch)
        out = model.context.w_out if family == "g2l_scalar" else (
            model.context.W_out if family.startswith("g2l") else (model.dist.W_out if family.startswith("dist") else model.pathbind.W_out)
        )
        w, b = out.weight.detach().clone(), out.bias.detach().clone()
        with torch.no_grad():
            out.weight.zero_()
            out.bias.zero_()
            y_zero = model(batch)
            out.weight.copy_(w)
            out.bias.copy_(b)
        results[family] = {
            "max_abs_diff_zeroed_vs_base": float((y_zero - y_base).abs().max()),
            "max_abs_diff_init_vs_base": float((y_init - y_base).abs().max()),
        }
    ok = all(v["max_abs_diff_zeroed_vs_base"] <= 1e-5 for v in results.values()) and all(
        v["max_abs_diff_init_vs_base"] > 0.0 for v in results.values()
    )
    return ok, {"families": results, "tolerance": 1e-5}


def gate_distance_semantics(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    cache = views["valid_cache"]
    rows = list(range(16))
    batch = D.collate(cache, rows)
    atom_batch = batch["atom_batch"]
    bond_index = batch["bond_index"]
    n_atoms = int(atom_batch.numel())
    adj: list[list[int]] = [[] for _ in range(n_atoms)]
    for u, v in bond_index.tolist():
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))

    def bfs(source: int) -> list[int]:
        dist = [-1] * n_atoms
        dist[source] = 0
        queue = [source]
        for u in queue:
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    queue.append(int(v))
        return dist

    mismatches = []
    for i in range(n_atoms):
        dist = bfs(i)
        exp34 = sorted(j for j, d in enumerate(dist) if 3 <= d <= 4)
        exp5 = sorted(j for j, d in enumerate(dist) if d >= 5)
        got34 = sorted(batch["d34_idx"][batch["d34_ptr"][i] : batch["d34_ptr"][i + 1]].tolist())
        got5 = sorted(batch["d5_idx"][batch["d5_ptr"][i] : batch["d5_ptr"][i + 1]].tolist())
        if exp34 != got34 or exp5 != got5:
            mismatches.append({"atom": i, "exp34": exp34, "got34": got34, "exp5": exp5, "got5": got5})
            break
    d34 = batch["d34_idx"]
    d5 = batch["d5_idx"]
    details = {
        "n_atoms_checked": n_atoms,
        "mismatches": mismatches,
        "d34_d5_disjoint": bool(all(
            not (
                set(batch["d34_idx"][batch["d34_ptr"][i] : batch["d34_ptr"][i + 1]].tolist())
                & set(batch["d5_idx"][batch["d5_ptr"][i] : batch["d5_ptr"][i + 1]].tolist())
            )
            for i in range(n_atoms)
        )),
        "no_self_targets": bool(not any(int(x) == i for i in range(n_atoms) for x in batch["d34_idx"][batch["d34_ptr"][i] : batch["d34_ptr"][i + 1]])),
        "entry_counts_match_ptrs": bool(
            int(batch["d34_ptr"][-1]) == int(batch["d34_idx"].numel()) and int(batch["d5_ptr"][-1]) == int(batch["d5_idx"].numel())
        ),
        "coverage_valid": D.distance_bucket_coverage(cache),
    }
    ok = bool(not mismatches and details["d34_d5_disjoint"] and details["no_self_targets"] and details["entry_counts_match_ptrs"])
    return ok, details


def gate_path_semantics(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    cache = views["valid_cache"]
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    atom_type = np.asarray(cache["atom_type"], dtype=np.int64)
    bond_type = np.asarray(cache["bond_type"], dtype=np.int64)
    bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
    path_cat = np.asarray(cache["path_cat"], dtype=np.int64)
    path_end = np.asarray(cache["path_end"], dtype=np.int64)
    path_end63 = np.asarray(cache["path_end63"], dtype=np.int64)
    path_closure = np.asarray(cache["path_closure"], dtype=np.int64)
    mismatches = []
    max_mid = max_b1 = max_b2 = -1
    for g in range(8):
        a0, a1 = int(atom_ptr[g]), int(atom_ptr[g + 1])
        b0, b1 = int(bond_ptr[g]), int(bond_ptr[g + 1])
        n = a1 - a0
        u = bond_index[b0:b1, 0]
        v = bond_index[b0:b1, 1]
        adj = MG.adjacency(n, u, v, bond_type[b0:b1])
        dist = MG.all_pairs_shortest_paths(n, adj)
        records = MG.enumerate_s2(atom_type[a0:a1], adj, 5, dist)
        for i in range(n):
            cached = sorted(
                (
                    int(path_cat[p]),
                    int(path_end[p]),
                    int(path_closure[p]),
                    int(path_end63[p]),
                )
                for p in range(int(path_ptr[a0 + i]), int(path_ptr[a0 + i + 1]))
            )
            expected = sorted(
                (int(r["path_cat"]), int(r["end"]), int(r["closure"]), int(r["end63"]))
                for r in records[i]
            )
            if cached != expected:
                mismatches.append({"mol": g, "root": i, "cached": cached[:5], "expected": expected[:5]})
                break
            for c, e, cl, e63 in cached:
                mid = c // 25
                b1t = (c % 25) // 5
                b2t = c % 5
                max_mid = max(max_mid, mid)
                max_b1 = max(max_b1, b1t)
                max_b2 = max(max_b2, b2t)
        if mismatches:
            break
    details = {
        "n_molecules_checked": 8,
        "mismatches": mismatches,
        "decoded_ranges": {"mid_type": [0, max_mid], "bond_type_1": [0, max_b1], "bond_type_2": [0, max_b2]},
        "decoded_ranges_valid": bool(0 <= max_mid <= 15 and 0 <= max_b1 <= 4 and 0 <= max_b2 <= 4),
        "path_end63_formula_checked": True,
    }
    return bool(not mismatches and details["decoded_ranges_valid"]), details


def gate_path_binding_invariants(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    spec, norm = views["spec"], views["train_norm"]
    model, _ = M.build_family("path_bind", spec, norm, 0, "cpu")
    model.eval()
    batch = M.to_device(D.collate(views["valid_cache"], list(range(32))), "cpu")
    a, _b, h = model._encode(batch)
    with torch.no_grad():
        _cb, ab = model.pathbind(h, a, batch, "bind")
        _cu, au = model.pathbind(h, a, batch, "unbound")
    counts = ab["path_counts"]
    one = counts == 1
    multi = counts >= 2
    gap = (ab["bind"] - au["bind"]).abs().max(dim=1).values
    # Constant relation vectors across all paths.
    b_const_r = dict(batch)
    b_const_r["path_cat"] = torch.zeros_like(batch["path_cat"])
    b_const_r["path_closure"] = torch.zeros_like(batch["path_closure"])
    with torch.no_grad():
        _c, ab_r = model.pathbind(h, a, b_const_r, "bind")
        _c, au_r = model.pathbind(h, a, b_const_r, "unbound")
    # Constant endpoint content across all paths.
    a_const = torch.zeros_like(a)
    with torch.no_grad():
        _c, ab_v = model.pathbind(h, a_const, batch, "bind")
        _c, au_v = model.pathbind(h, a_const, batch, "unbound")
    no_cross = bool(torch.equal(batch["path_graph"], batch["atom_batch"][batch["path_atom"]])) and bool(
        torch.equal(batch["atom_batch"][batch["path_end"]], batch["path_graph"])
    ) if batch["path_cat"].numel() else True
    details = {
        "n_paths": int(batch["path_cat"].numel()),
        "n_roots_with_paths": int((counts > 0).sum()),
        "n_single_path_roots": int(one.sum()),
        "n_multi_path_roots": int(multi.sum()),
        "single_path_bind_difference_max": float(gap[one].max()) if one.any() else 0.0,
        "multi_path_bind_difference_max": float(gap[multi].max()) if multi.any() else 0.0,
        "multi_path_nonzero_difference_roots": int((gap[multi] > 1e-7).sum()) if multi.any() else 0,
        "constant_relation_bind_difference_max": float((ab_r["bind"] - au_r["bind"]).abs().max()),
        "constant_endpoint_bind_difference_max": float((ab_v["bind"] - au_v["bind"]).abs().max()),
        "no_cross_molecule_path_indices": no_cross,
        "tolerance": 1e-6,
    }
    ok = bool(
        details["single_path_bind_difference_max"] <= 1e-6
        and details["constant_relation_bind_difference_max"] <= 1e-6
        and details["constant_endpoint_bind_difference_max"] <= 1e-6
        and details["multi_path_bind_difference_max"] > 1e-6
        and details["multi_path_nonzero_difference_roots"] > 0
        and no_cross
    )
    return ok, details


def gate_batch_invariance(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    spec, norm = views["spec"], views["train_norm"]
    cache = views["valid_cache"]
    rows = list(range(16))
    results = {}
    for family in C.NEW_FAMILIES:
        model, _ = M.build_family(family, spec, norm, 0, "cpu")
        model.eval()
        with torch.no_grad():
            single = np.asarray([float(model(M.to_device(D.collate(cache, [r]), "cpu")).detach().cpu()[0]) for r in rows])
            full = model(M.to_device(D.collate(cache, rows), "cpu")).detach().cpu().numpy()
            rev = model(M.to_device(D.collate(cache, list(reversed(rows))), "cpu")).detach().cpu().numpy()[::-1]
            even = model(M.to_device(D.collate(cache, rows[::2]), "cpu")).detach().cpu().numpy()
            mixed_rows = rows[8:] + list(range(16, 32)) + rows[:8]
            mixed = model(M.to_device(D.collate(cache, mixed_rows), "cpu")).detach().cpu().numpy()
            mixed_aligned = mixed[24:32]
            # Cross-molecule isolation: perturb graph 0 only.
            batch = M.to_device(D.collate(cache, rows[:8]), "cpu")
            y0 = model(batch).detach().cpu()
            perturbed = dict(batch)
            feat = batch["atom_feat"].clone()
            mask = batch["atom_batch"] == 0
            feat[mask] = feat[mask] + 0.5
            perturbed["atom_feat"] = feat
            y1 = model(perturbed).detach().cpu()
        diffs = {
            "single_vs_full": float(np.max(np.abs(single - full))),
            "single_vs_reversed": float(np.max(np.abs(single - rev))),
            "single_vs_even": float(np.max(np.abs(single[::2] - even))),
            "single_vs_mixed": float(np.max(np.abs(single[:8] - mixed_aligned))),
            "other_graphs_after_perturbation": float((y1[1:] - y0[1:]).abs().max()),
            "perturbed_graph_changed": float((y1[0] - y0[0]).abs().max()),
        }
        results[family] = diffs
    tolerance = 1e-5
    ok = all(
        max(v["single_vs_full"], v["single_vs_reversed"], v["single_vs_even"], v["single_vs_mixed"], v["other_graphs_after_perturbation"])
        <= tolerance
        and v["perturbed_graph_changed"] > 0.0
        for v in results.values()
    )
    return ok, {"families": results, "tolerance": tolerance}


def gate_renumbering(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    spec, norm = views["spec"], views["train_norm"]
    cache = views["valid_cache"]
    results = {}
    for family in C.NEW_FAMILIES:
        model, _ = M.build_family(family, spec, norm, 0, "cpu")
        model.eval()
        batch = M.to_device(D.collate(cache, list(range(16))), "cpu")
        g = torch.Generator().manual_seed(1234)
        permuted = M.to_device(D.permute_batch_atoms(batch, g), "cpu")
        with torch.no_grad():
            y = model(batch).detach().cpu()
            yp = model(permuted).detach().cpu()
        entry = {"atom_renumbering_max_abs_diff": float((y - yp).abs().max())}
        if family.startswith("path"):
            g2 = torch.Generator().manual_seed(5678)
            path_perm = M.to_device(D.permute_paths_within_roots(batch, g2), "cpu")
            with torch.no_grad():
                ypp = model(path_perm).detach().cpu()
            entry["path_order_permutation_max_abs_diff"] = float((y - ypp).abs().max())
        results[family] = entry
    tolerance = 1e-5
    ok = all(
        v["atom_renumbering_max_abs_diff"] <= tolerance
        and v.get("path_order_permutation_max_abs_diff", 0.0) <= tolerance
        for v in results.values()
    )
    return ok, {"families": results, "tolerance": tolerance}


def gate_gradients(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    spec, norm = views["spec"], views["train_norm"]
    cache = views["valid_cache"]
    results = {}
    for family in C.NEW_FAMILIES:
        model, _ = M.build_family(family, spec, norm, 0, "cpu")
        names = M.new_parameter_names(model, family)
        probe = __import__("dc_train", fromlist=["gradient_probe"]).gradient_probe(model, cache, names, "cpu")
        results[family] = {
            "new_parameters": names,
            "gradient_probe": probe,
        }
    ok = all(v["gradient_probe"]["passed"] for v in results.values())
    return ok, results


def gate_rho_original(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    spec, norm = views["spec"], views["train_norm"]
    cache = views["valid_cache"]
    results = {}
    for family in C.NEW_FAMILIES:
        model, _ = M.build_family(family, spec, norm, 0, "cpu")
        model.eval()
        batch = M.to_device(D.collate(cache, list(range(8))), "cpu")
        captured_h, captured_rho, captured_f_input = [], [], []
        h_local = model.local.register_forward_hook(lambda module, inputs, output: captured_h.append(output.detach().clone()))
        rho_pre = model.rho.register_forward_pre_hook(lambda module, inputs: captured_rho.append(inputs[0].detach().clone()))
        f_pre = model.f.register_forward_pre_hook(lambda module, inputs: captured_f_input.append(inputs[0].detach().clone()))
        try:
            with torch.no_grad():
                _y, aux = model(batch, return_aux=True)
        finally:
            h_local.remove()
            rho_pre.remove()
            f_pre.remove()
        h = captured_h[0]
        n_graphs = int(batch["n_graphs"])
        atom_batch = batch["atom_batch"]
        width = h.shape[1]
        count = h.new_zeros(n_graphs).index_add_(0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype))
        mean_h = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h) / count.clamp(min=1.0).unsqueeze(1)
        mean_h2 = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h * h) / count.clamp(min=1.0).unsqueeze(1)
        g = (batch["global"] - model.global_mean) / model.global_scale
        expected = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
        diff_orig = float((captured_rho[0] - expected).abs().max())
        h_new = aux["h_new"]
        change_scale = float((h_new - h).abs().max())
        mean_h_new = h_new.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h_new) / count.clamp(min=1.0).unsqueeze(1)
        mean_h2_new = h_new.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h_new * h_new) / count.clamp(min=1.0).unsqueeze(1)
        expected_new = torch.cat([g, mean_h_new, mean_h2_new, torch.log1p(count).unsqueeze(1)], dim=1)
        diff_new = float((captured_rho[0] - expected_new).abs().max())
        f_input_diff = float((captured_f_input[0] - h_new).abs().max()) if captured_f_input else 0.0
        entry = {
            "rho_input_matches_h_original_max_abs_diff": diff_orig,
            "rho_input_vs_h_new_reconstruction_max_abs_diff": diff_new,
            "h_new_minus_h_max_abs": change_scale,
            "f_input_matches_h_new_max_abs_diff": f_input_diff,
            "discriminating": bool(family == "g2l_scalar" or (change_scale > 0.0 and diff_new > 0.0)),
        }
        entry["passed"] = bool(diff_orig <= 1e-5 and f_input_diff <= 1e-5 and entry["discriminating"])
        results[family] = entry
    ok = all(v["passed"] for v in results.values())
    return ok, {"families": results}


def gate_dropout(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    spec, norm = views["spec"], views["train_norm"]
    cache = views["valid_cache"]
    results = {}
    import dc_train

    for family in C.NEW_FAMILIES:
        model, _ = M.build_family(family, spec, norm, 0, "cpu")
        audit = dc_train.dropout_activity_audit(model, cache, list(range(8)))
        results[family] = audit
    ok = all(v["passed"] for v in results.values())
    return ok, results


def gate_frozen_baselines(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    spec, norm = views["spec"], views["train_norm"]
    target_norm = views["target_norm"]
    y_valid = np.asarray(views["labels"]["y_valid"], dtype=np.float64)
    results = {}
    for key in ("base_s0", "roleblind_s0"):
        entry = C.FROZEN_CHECKPOINTS[key]
        C.sha256_file(entry["path"]) == entry["sha256"] or (_ for _ in ()).throw(PermissionError(f"{key} sha mismatch"))
        model, _ = M.build_family(entry["family"], spec, norm, int(entry["seed"]), "cpu")
        payload = M.load_checkpoint(model, entry["path"], expected_sha256=entry["sha256"], expected_family=entry["family"], expected_seed=int(entry["seed"]))
        pred = M.predict(model, views["valid_cache"], "cpu")
        pred = pred * float(target_norm["scale"]) + float(target_norm["mean"])
        metrics = MET.all_metrics(pred, y_valid)
        results[key] = {
            "valid_rmse": metrics["rmse"],
            "reference_valid_rmse": entry["valid_rmse_reference"],
            "abs_diff": abs(metrics["rmse"] - entry["valid_rmse_reference"]),
            "checkpoint_sha256": C.sha256_file(entry["path"]),
            "soup_state_hash": payload.get("soup_state_hash"),
        }
    ok = all(v["abs_diff"] <= 1e-5 for v in results.values())
    return ok, results


def gate_inference_immutable(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    spec, norm = views["spec"], views["train_norm"]
    cache = views["valid_cache"]
    results = {}
    for key, family in (("base_s0", "base"), ("roleblind_s0", "roleblind")):
        entry = C.FROZEN_CHECKPOINTS[key]
        model, _ = M.build_family(entry["family"], spec, norm, int(entry["seed"]), "cpu")
        M.load_checkpoint(model, entry["path"], expected_sha256=entry["sha256"], expected_family=entry["family"], expected_seed=int(entry["seed"]))
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.eval()
        _ = M.predict(model, cache, "cpu", batch_size=256)
        after = model.state_dict()
        changed = [k for k in before if not torch.equal(before[k], after[k].detach())]
        grads = [n for n, p in model.named_parameters() if p.grad is not None]
        results[key] = {"n_changed_state_entries": len(changed), "parameters_with_grad": grads, "model_training": bool(model.training)}
    ok = all(v["n_changed_state_entries"] == 0 and not v["parameters_with_grad"] and not v["model_training"] for v in results.values())
    return ok, results


def gate_metrics_arithmetic(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    rng = np.random.default_rng(123)
    pred = rng.normal(size=64)
    y = rng.normal(size=64)
    metrics = MET.all_metrics(pred, y)
    manual = {
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "mae": float(np.mean(np.abs(pred - y))),
        "r2": float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)),
    }
    scale, mean = 1.7, -0.3
    denorm = (pred * scale + mean - mean) / scale
    details = {
        "metrics": metrics,
        "manual": manual,
        "max_abs_diff": {k: abs(metrics[k] - manual[k]) for k in manual},
        "denormalisation_roundtrip_max_abs_diff": float(np.max(np.abs(denorm - pred))),
    }
    ok = all(v <= 1e-12 for v in details["max_abs_diff"].values()) and details["denormalisation_roundtrip_max_abs_diff"] <= 1e-12
    return ok, details


def gate_test_scope(views: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Static scan: training/model modules must not mention Test loading."""
    import dc_train

    hits = []
    for path in (__import__("pathlib").Path(dc_train.__file__), __import__("pathlib").Path(M.__file__)):
        text = path.read_text(encoding="utf-8")
        for token in ("test_cache", "y_test", "rows_test"):
            if token in text:
                hits.append({"file": path.name, "token": token})
    return not hits, {"hits": hits, "checked_files": ["dc_train.py", "dc_models.py"]}


GATE_SPECS = (
    ("data_split_hashes", "frozen split rows, labels, cache fingerprint and Train-only structure normalisation", gate_data),
    ("target_norm_train_only", "target scaler reproduces the frozen Train3360 mean/scale", gate_target_norm),
    ("shared_init_and_params", "new arms share bit-identical BASE parameters; mechanism pairs are init/parameter identical", gate_shared_init),
    ("no_correction_recovers_base", "zeroing each correction output projection exactly recovers BASE with the same weights", gate_no_correction),
    ("distance_bucket_semantics", "D34/D5 are independently recomputed from shortest paths and exclude self/dist<=2/unreachable", gate_distance_semantics),
    ("path_category_and_endpoint_semantics", "S2 path categories decode to valid 0..15/0..4/0..4 and match independent enumeration", gate_path_semantics),
    ("path_binding_invariants", "BIND/UNBOUND agree for one path and constant roles/content and differ for mixed multi-path roots", gate_path_binding_invariants),
    ("batch_and_cross_molecule_invariance", "predictions are independent of batch composition/order and do not mix molecules", gate_batch_invariance),
    ("atom_and_path_order_invariance", "atom renumbering and S2 path instance order do not change predictions", gate_renumbering),
    ("new_parameter_gradients", "every new tensor gets finite non-zero gradients and changes after two optimizer steps", gate_gradients),
    ("rho_reads_original_h", "rho receives statistics of the original h, not h_new; f receives h_new", gate_rho_original),
    ("dropout_executes", "DropoutAtLocal masks in train mode and is the identity in eval mode", gate_dropout),
    ("frozen_baseline_reproduction", "BASE_s0 and ROLEBLIND_s0 reproduce their frozen Validation420 RMSE", gate_frozen_baselines),
    ("frozen_inference_immutable", "frozen-checkpoint inference changes no weight and creates no gradient", gate_inference_immutable),
    ("metric_and_denormalisation_arithmetic", "RMSE/MAE/R2 and raw-logD denormalisation arithmetic are correct", gate_metrics_arithmetic),
    ("training_code_test_scope", "training/model modules contain no Test cache or Test label access", gate_test_scope),
)


def run_gates() -> dict[str, Any]:
    views = D.load_views(include_test=True)
    checks: list[dict[str, Any]] = []
    for gate_id, name, fn in GATE_SPECS:
        _gate(checks, gate_id, name, lambda fn=fn: fn(views))
    n_passed = sum(1 for c in checks if c["passed"])
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "config_hash": C.sha256_json(C.load_config()),
        "code_hashes": C.code_file_hashes(),
        "n_gates": len(checks),
        "n_passed": n_passed,
        "ok": n_passed == len(checks),
        "checks": checks,
        "environment": C.env_info(),
        "test_labels_read": True,
        "test_predictions_read": False,
        "test_metrics_read": False,
    }
    C.write_json(C.CPU_GATES, payload)
    return payload


__all__ = ["run_gates", "GATE_SPECS"]
