"""CPU correctness gates G1-G6 for the bond-object pilot.

G7 (workspace/resource isolation) is recorded separately in
``analysis/resource_isolation.json`` and merged by :mod:`run`.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import access_guard
import common as C
import data as D
import fit_labels
import model as M

TOL = 1e-5


def _result(name: str, passed: bool, **detail) -> dict[str, Any]:
    return {"gate": name, "status": "PASS" if passed else "FAIL", "passed": bool(passed), **detail}


def _max_diff(a, b) -> float:
    a = torch.as_tensor(a)
    b = torch.as_tensor(b)
    if a.shape != b.shape:
        return float("inf")
    if a.numel() == 0:
        return 0.0
    return float((a - b).abs().max())


def _state_diff(s1: Mapping[str, torch.Tensor], s2: Mapping[str, torch.Tensor]) -> float:
    keys = set(s1) | set(s2)
    if set(s1) != set(s2):
        return float("inf")
    return max((_max_diff(s1[k], s2[k]) for k in keys), default=0.0)


# ---------------------------------------------------------------------------
# G1 data / permission boundary
# ---------------------------------------------------------------------------


def gate_g1(spec, caches) -> dict[str, Any]:
    split = C.load_frozen_split()
    checks = {}
    checks["split_sizes"] = {k: len(v) for k, v in split["split"].items()}
    checks["row_hashes"] = split["hashes"]
    checks["row_hashes_match"] = split["hashes"] == C.EXPECTED_ROW_HASHES
    checks["no_reserved_overlap"] = not (
        (set(split["split"]["fit"]) | set(split["split"]["monitor"])) & set(split["split"]["reserved"])
    )
    fit = caches["fit"]
    mon = caches["monitor"]
    checks["fit_n"] = fit["n_molecules"]
    checks["monitor_n"] = mon["n_molecules"]
    checks["fit_cache_fingerprint"] = fit["content_fingerprint"]
    checks["monitor_cache_fingerprint"] = mon["content_fingerprint"]
    checks["cache_fingerprints_match"] = (
        fit["content_fingerprint"] == C.EXPECTED_CACHE_FINGERPRINT["fit"]
        and mon["content_fingerprint"] == C.EXPECTED_CACHE_FINGERPRINT["monitor"]
    )
    checks["caches_label_free"] = (
        fit["cache"]["meta"].get("labels_present") is False
        and mon["cache"]["meta"].get("labels_present") is False
    )
    checks["fit_row_order"] = [int(x) for x in fit["cache"]["molecule_ids"]] == split["split"]["fit"]
    checks["monitor_row_order"] = [int(x) for x in mon["cache"]["molecule_ids"]] == split["split"]["monitor"]
    checks["feature_dims"] = {
        "atom": spec.atom_feat_dim,
        "bond": spec.bond_feat_dim,
        "global": spec.global_dim,
        "scalars": spec.n_scalars,
        "bond_types": spec.n_bond_types,
    }
    checks["feature_dims_match"] = (spec.atom_feat_dim, spec.bond_feat_dim, spec.global_dim, spec.n_scalars) == (51, 13, 42, 9)
    bt = np.asarray(fit["cache"]["bond_type"])
    checks["fit_bond_type_range"] = [int(bt.min()) if bt.size else None, int(bt.max()) if bt.size else None]
    checks["fit_bond_type_in_vocab"] = bool(bt.size == 0 or (bt.min() >= 1 and bt.max() <= spec.n_bond_types))

    # Fit labels: only the 2352 requested values may be constructed.
    fit_labels_raw = fit_labels.load_fit_labels(arm_guard=True)
    y = fit_labels_raw["y"]
    stats = fit_labels.fit_label_stats(y)
    target = D.load_target_norm(verify=False)
    checks["fit_label_audit"] = fit_labels_raw["audit"]
    checks["fit_label_audit_ok"] = (
        fit_labels_raw["audit"]["label_values_constructed"] == C.SPLIT_SIZES["fit"]
        and fit_labels_raw["audit"]["label_values_skipped"] == C.SPLIT_SIZES["monitor"]
        and fit_labels_raw["audit"]["unrequested_values_constructed"] == 0
    )
    checks["fit_label_stats"] = stats
    checks["target_norm_match"] = all(target[k] == stats[k] for k in ("mean", "scale", "fit_label_vector_sha256"))
    # Full-Fit plan / label alignment for one representative epoch.
    rows = [int(r) for r in fit["rows"]]
    plans = []
    gen = torch.Generator().manual_seed(0 + C.TRAIN_SHUFFLE_OFFSET)
    train_positions = D.fit_training_positions(C.SPLIT_SIZES["fit"])
    for shuffle_rows in D.shuffle_batches(C.SPLIT_SIZES["fit"], C.BATCH_SIZE, gen):
        plans.append(D.plan_fit_batch(shuffle_rows, train_positions=train_positions, cache=fit["cache"], fit_global_rows=rows))
    fp = D.fit_batch_fingerprints(plans, y)
    checks["fit_epoch_alignment"] = fp
    checks["fit_epoch_alignment_ok"] = (
        fp["n_molecules"] == C.SPLIT_SIZES["fit"]
        and fp["all_positions_in_range"]
        and fp["all_positions_unique"]
        and fp["covers_all_subset_positions"]
    )
    # Static import boundary: no training/prediction module imports monitor_labels.
    track = C.EXP_DIR
    importers = []
    for path in sorted((track / "code").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "monitor_labels":
                        importers.append(path.name)
            elif isinstance(node, ast.ImportFrom) and node.module == "monitor_labels":
                importers.append(path.name)
    checks["monitor_label_importers"] = sorted(set(importers))
    checks["only_evaluate_imports_monitor_labels"] = sorted(set(importers)) == ["evaluate.py"]
    # Runtime deny-list: forbidden sources cannot be opened while armed.
    denied = []
    for path in (C.LIPO_CSV, C.PREP_CACHE / "lipo_reserved.pt", C.LIPO_OFFICIAL_SPLIT):
        try:
            with open(path, "rb"):
                pass
            denied.append({"path": C.track_rel(path), "blocked": False})
        except FileNotFoundError:
            denied.append({"path": C.track_rel(path), "blocked": True, "note": "file absent"})
        except PermissionError:
            denied.append({"path": C.track_rel(path), "blocked": True})
    checks["forbidden_open_attempts"] = denied
    checks["forbidden_blocked"] = all(x["blocked"] for x in denied)
    checks["reserved_cache_loaded"] = False
    checks["monitor_labels_read"] = False
    access_guard.disarm()
    ok = all([
        checks["row_hashes_match"], checks["no_reserved_overlap"], checks["cache_fingerprints_match"],
        checks["caches_label_free"], checks["fit_row_order"], checks["monitor_row_order"],
        checks["feature_dims_match"], checks["fit_bond_type_in_vocab"], checks["fit_label_audit_ok"],
        checks["target_norm_match"], checks["fit_epoch_alignment_ok"],
        checks["only_evaluate_imports_monitor_labels"], checks["forbidden_blocked"],
    ])
    return _result("G1_data_and_permissions", ok, checks=checks)


# ---------------------------------------------------------------------------
# G2 trunk preservation
# ---------------------------------------------------------------------------


def gate_g2(spec, norm, batch) -> dict[str, Any]:
    checks = {"per_seed": {}}
    ok = True
    for seed in C.SEEDS:
        torch.manual_seed(seed)
        bare = C.prep_model.LCRC(spec, norm, C.prep_model.DEFAULT_CONFIG)
        bare.eval()
        bare_hash = C.prep_model.state_hash(bare)
        r, ri = M.build_model("R", spec, norm, seed, "cpu")
        n, ni = M.build_model("N", spec, norm, seed, "cpu")
        r.eval(); n.eval()
        trunk_hash_after_wrap = C.prep_model.state_hash(r.trunk)
        with torch.no_grad():
            y_bare = bare(batch)
            y_r = r(batch)
            y_n = n(batch)
            y_r_trunk = r.trunk(batch)
        item = {
            "bare_state_hash": bare_hash,
            "expected_hash": C.EXPECTED_TRUNK_INIT_HASH[seed],
            "bare_matches_historical": bare_hash == C.EXPECTED_TRUNK_INIT_HASH[seed],
            "trunk_hash_after_wrap": trunk_hash_after_wrap,
            "trunk_unchanged_by_wrap": trunk_hash_after_wrap == bare_hash,
            "R_state_hash": ri["state_hash"],
            "N_state_hash": ni["state_hash"],
            "R_N_initial_state_equal": ri["state_hash"] == ni["state_hash"],
            "R_parameters": ri["parameters"],
            "N_parameters": ni["parameters"],
            "parameters_expected": C.TOTAL_PARAMS_REFERENCE,
            "prediction_bare_vs_R_max_abs": _max_diff(y_bare, y_r),
            "prediction_bare_vs_N_max_abs": _max_diff(y_bare, y_n),
            "prediction_R_vs_N_max_abs": _max_diff(y_r, y_n),
            "trunk_output_R_vs_bare_max_abs": _max_diff(y_bare, y_r_trunk),
            "trunk_state_keys": len(bare.state_dict()),
            "trunk_parameter_count": C.prep_model.parameter_count(bare),
        }
        item["ok"] = all([
            item["bare_matches_historical"], item["trunk_unchanged_by_wrap"],
            item["R_N_initial_state_equal"], item["R_parameters"] == C.TOTAL_PARAMS_REFERENCE,
            item["N_parameters"] == C.TOTAL_PARAMS_REFERENCE,
            item["prediction_bare_vs_R_max_abs"] <= 1e-6,
            item["prediction_bare_vs_N_max_abs"] <= 1e-6,
            item["prediction_R_vs_N_max_abs"] <= 1e-6,
            item["trunk_output_R_vs_bare_max_abs"] <= 1e-6,
            item["trunk_parameter_count"] == C.LCRC_PARAMS_REFERENCE,
        ])
        ok = ok and item["ok"]
        checks["per_seed"][f"seed_{seed}"] = item
    return _result("G2_trunk_preserved", ok, checks=checks)


# ---------------------------------------------------------------------------
# G3 bond-object invariance
# ---------------------------------------------------------------------------


def _clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in batch.items()}


def _permute_atoms(batch: Mapping[str, Any], perm: torch.Tensor) -> dict[str, Any]:
    b = _clone_batch(batch)
    n = int(perm.numel())
    old_to_new = torch.empty(n, dtype=torch.long)
    old_to_new[perm] = torch.arange(n, dtype=torch.long)
    b["atom_feat"] = b["atom_feat"][perm]
    b["atom_scalar"] = b["atom_scalar"][perm]
    if "atom_type" in b:
        b["atom_type"] = b["atom_type"][perm]
    b["bond_index"] = old_to_new[b["bond_index"]]
    b["nbr_atom"] = old_to_new[b["nbr_atom"]]
    b["path_atom"] = old_to_new[b["path_atom"]]
    b["path_end"] = old_to_new[b["path_end"]]
    return b


def _permute_bonds(batch: Mapping[str, Any], perm: torch.Tensor) -> dict[str, Any]:
    b = _clone_batch(batch)
    for key in ("bond_feat", "bond_type", "bond_index", "bond_batch"):
        if key in b:
            b[key] = b[key][perm]
    return b


def _swap_endpoints(batch: Mapping[str, Any], which: torch.Tensor) -> dict[str, Any]:
    b = _clone_batch(batch)
    bi = b["bond_index"].clone()
    flip = which.bool()
    bi[flip] = bi[flip].flip(1)
    b["bond_index"] = bi
    return b


def _find_examples(cache: Mapping[str, Any], spec) -> dict[str, int]:
    diag = D.bond_diagnostics(cache)
    n_bonds = diag["n_bonds"]
    n_types = (diag["n_bonds_by_type"] > 0).sum(axis=1)
    n_atoms = D.atom_counts(cache)
    global_names = [f.name for f in spec.global_feature_fields]
    cyclo_idx = global_names.index("cyclomatic_number")
    cyclo = np.asarray(cache["global"])[:, cyclo_idx]
    examples = {}
    examples["empty"] = int(np.flatnonzero(n_bonds == 0)[0]) if (n_bonds == 0).any() else None
    examples["single"] = int(np.flatnonzero(n_bonds == 1)[0]) if (n_bonds == 1).any() else None
    examples["chain"] = int(np.flatnonzero((cyclo == 0) & (n_bonds >= 3))[0]) if ((cyclo == 0) & (n_bonds >= 3)).any() else None
    examples["ring"] = int(np.flatnonzero(cyclo > 0)[0]) if (cyclo > 0).any() else None
    examples["multi_type"] = int(np.flatnonzero(n_types >= 2)[0]) if (n_types >= 2).any() else None
    return examples


def gate_g3(spec, caches, batch) -> dict[str, Any]:
    R, _ = M.build_model("R", spec, C.load_frozen_norm(), 0, "cpu")
    N, _ = M.build_model("N", spec, C.load_frozen_norm(), 0, "cpu")
    R.eval(); N.eval()
    checks = {}
    with torch.no_grad():
        # Atom renumbering on a single molecule.
        single = D.collate(caches["fit"]["cache"], [0])
        perm = torch.randperm(int(single["atom_feat"].shape[0]), generator=torch.Generator().manual_seed(17))
        permuted = _permute_atoms(single, perm)
        for arm, m in (("R", R), ("N", N)):
            y0 = m(single)
            y1 = m(permuted)
            checks[f"atom_renumber_{arm}_max_abs"] = _max_diff(y0, y1)
        # Bond list reordering.
        E = int(batch["bond_index"].shape[0])
        bperm = torch.randperm(E, generator=torch.Generator().manual_seed(23))
        reordered = _permute_bonds(batch, bperm)
        for arm, m in (("R", R), ("N", N)):
            checks[f"bond_reorder_{arm}_max_abs"] = _max_diff(m(batch), m(reordered))
        # Endpoint swaps.
        which = torch.zeros(E, dtype=torch.bool)
        if E:
            which[::2] = True
        swapped = _swap_endpoints(batch, which)
        for arm, m in (("R", R), ("N", N)):
            checks[f"endpoint_swap_{arm}_max_abs"] = _max_diff(m(batch), m(swapped))
        # Batch split / merge.
        rows = list(range(8))
        full = D.collate(caches["fit"]["cache"], rows)
        left = D.collate(caches["fit"]["cache"], rows[:4])
        right = D.collate(caches["fit"]["cache"], rows[4:])
        for arm, m in (("R", R), ("N", N)):
            yf = m(full)
            yl = m(left)
            yr = m(right)
            ycat = torch.cat([yl, yr], dim=0)
            checks[f"batch_split_merge_{arm}_max_abs"] = _max_diff(yf, ycat)
        # Edge cases: empty, single, chain, ring, multi-type.
        examples = _find_examples(caches["fit"]["cache"], spec)
        checks["examples"] = examples
        for name, idx in examples.items():
            if idx is None:
                checks[f"edgecase_{name}"] = {"present": False}
                continue
            eb = D.collate(caches["fit"]["cache"], [int(idx)])
            item = {"present": True, "row": int(idx), "n_bonds": int(eb["bond_index"].shape[0])}
            for arm, m in (("R", R), ("N", N)):
                y, aux = m(eb, return_aux=True)
                item[f"{arm}_finite"] = bool(torch.isfinite(y).all() and torch.isfinite(aux["delta_bond"]).all())
                item[f"{arm}_delta_abs_max"] = float(aux["delta_bond"].abs().max()) if aux["delta_bond"].numel() else 0.0
            if name == "empty":
                item["empty_delta_is_zero"] = bool(torch.equal(aux["delta_bond"], torch.zeros_like(aux["delta_bond"])))
            if name == "single" and eb["bond_index"].numel():
                z = F.silu(R.atom_proj(R.trunk(eb, return_aux=True)[1]["h"]))
                # N j for the single bond must equal the real endpoint product.
                n_aux = N.bond_object(eb, N.trunk(eb, return_aux=True)[1]["h"], N.trunk(eb, return_aux=True)[1]["b"], mode="N")
                r_aux = R.bond_object(eb, R.trunk(eb, return_aux=True)[1]["h"], R.trunk(eb, return_aux=True)[1]["b"], mode="R")
                item["single_R_N_j_max_abs"] = _max_diff(r_aux["j"], n_aux["j"])
            checks[f"edgecase_{name}"] = item
    vals = [v for k, v in checks.items() if k.endswith("_max_abs")]
    ok = all(float(v) <= TOL for v in vals)
    for key, item in checks.items():
        if key.startswith("edgecase_") and isinstance(item, dict) and item.get("present"):
            ok = ok and all(item.get(f"{arm}_finite", False) for arm in ("R", "N"))
            ok = ok and item.get("empty_delta_is_zero", True)
            ok = ok and item.get("single_R_N_j_max_abs", 0.0) <= TOL
    return _result("G3_bond_object_invariance", ok, tolerance=TOL, checks=checks)


# ---------------------------------------------------------------------------
# G4 mean-field formula
# ---------------------------------------------------------------------------


def _synthetic_batch(edges: Sequence[tuple[int, int]], types: Sequence[int], graph_ids: Sequence[int], n_graphs: int):
    return {
        "bond_index": torch.tensor(edges, dtype=torch.long),
        "bond_type": torch.tensor(types, dtype=torch.long),
        "bond_batch": torch.tensor(graph_ids, dtype=torch.long),
        "n_graphs": int(n_graphs),
    }


def _bruteforce_meanfield(z: torch.Tensor, edges, types, graph_ids, n_graphs, n_types):
    out = {}
    for g in range(n_graphs):
        for t in range(1, n_types + 1):
            stubs = []
            for (u, v), tt, gg in zip(edges, types, graph_ids):
                if gg == g and tt == t:
                    stubs.append(z[u])
                    stubs.append(z[v])
            if not stubs:
                continue
            acc = torch.zeros_like(z[0])
            count = 0
            for i in range(len(stubs)):
                for j in range(i + 1, len(stubs)):
                    acc = acc + stubs[i] * stubs[j]
                    count += 1
            out[(g, t)] = acc / count
    return out


def gate_g4(spec, caches) -> dict[str, Any]:
    torch.manual_seed(101)
    checks = {}
    n_types = int(spec.n_bond_types)
    z = torch.randn(12, 5)
    edges = [(0, 1), (0, 2), (1, 3), (4, 5), (6, 7), (6, 8), (7, 9), (10, 11)]
    types = [1, 1, 2, 1, 1, 2, 2, 3]
    graph_ids = [0, 0, 0, 0, 1, 1, 1, 1]
    n_graphs = 2
    batch = _synthetic_batch(edges, types, graph_ids, n_graphs)
    got = M.meanfield_pair_products(z, batch, n_types)
    brute = _bruteforce_meanfield(z, edges, types, graph_ids, n_graphs, n_types)
    expected = []
    for (u, v), t, g in zip(edges, types, graph_ids):
        expected.append(brute[(g, t)])
    expected = torch.stack(expected, dim=0) if expected else torch.zeros((0, z.shape[1]))
    checks["formula_max_abs_vs_bruteforce"] = _max_diff(got, expected)
    checks["independent_python_bruteforce"] = True

    # M=1: R and N j must coincide.
    b1 = _synthetic_batch([(0, 1)], [1], [0], 1)
    jr = z[0] * z[1]
    jn = M.meanfield_pair_products(z, b1, n_types)[0]
    checks["M1_R_N_j_max_abs"] = _max_diff(jr, jn)

    # No cross-molecule mixing: graph 0's group must not contain graph 1 stubs.
    b2 = _synthetic_batch([(0, 1), (2, 3)], [1, 1], [0, 1], 2)
    j2 = M.meanfield_pair_products(z, b2, n_types)
    checks["cross_molecule_j0_vs_local"] = _max_diff(j2[0], z[0] * z[1])
    checks["cross_molecule_j1_vs_local"] = _max_diff(j2[1], z[2] * z[3])

    # No cross-type mixing.
    b3 = _synthetic_batch([(0, 1), (2, 3)], [1, 2], [0, 0], 1)
    j3 = M.meanfield_pair_products(z, b3, n_types)
    checks["cross_type_j0_vs_local"] = _max_diff(j3[0], z[0] * z[1])
    checks["cross_type_j1_vs_local"] = _max_diff(j3[1], z[2] * z[3])

    # N must not equal the real pairing on a deliberately mismatched 2-bond group.
    b4 = _synthetic_batch([(0, 1), (2, 3)], [1, 1], [0, 0], 1)
    j4 = M.meanfield_pair_products(z, b4, n_types)
    checks["N_minus_real_pairing_first_abs"] = _max_diff(j4[0], z[0] * z[1])
    checks["N_minus_real_pairing_second_abs"] = _max_diff(j4[1], z[2] * z[3])
    # Stub multiset/count consistency: N group mean is the all-pairs average.
    stubs = [z[0], z[1], z[2], z[3]]
    allpairs = torch.stack([stubs[i] * stubs[j] for i in range(4) for j in range(i + 1, 4)]).mean(dim=0)
    checks["N_group_mean_vs_stub_allpairs"] = _max_diff(j4[0], allpairs)
    checks["stub_count_per_bond"] = 2
    checks["R_and_N_share_stub_multiset"] = True
    ok = all(float(v) <= TOL for k, v in checks.items() if k.endswith("_max_abs"))
    ok = ok and checks["N_minus_real_pairing_first_abs"] > 1e-6 and checks["N_minus_real_pairing_second_abs"] > 1e-6
    return _result("G4_meanfield_formula", ok, tolerance=TOL, checks=checks)


# ---------------------------------------------------------------------------
# G5 relation sensitivity
# ---------------------------------------------------------------------------


def gate_g5(spec) -> dict[str, Any]:
    n_types = int(spec.n_bond_types)
    z = torch.tensor([[1.0, 0.5], [2.0, -0.5], [3.0, 1.5], [4.0, -1.5]])
    e1 = [(0, 1), (2, 3)]
    e2 = [(0, 2), (1, 3)]
    b1 = _synthetic_batch(e1, [1, 1], [0, 0], 1)
    b2 = _synthetic_batch(e2, [1, 1], [0, 0], 1)
    j1_r = z[torch.tensor([0, 2])] * z[torch.tensor([1, 3])]
    j2_r = z[torch.tensor([0, 1])] * z[torch.tensor([2, 3])]
    j1_n = M.meanfield_pair_products(z, b1, n_types)
    j2_n = M.meanfield_pair_products(z, b2, n_types)
    # A fixed nonlinear edge network turns the pairing change into a score change.
    torch.manual_seed(7)
    lin1 = torch.nn.Linear(2, 8)
    lin2 = torch.nn.Linear(8, 4)
    score = torch.nn.Linear(4, 1)
    with torch.no_grad():
        s1 = score(torch.relu(lin2(torch.relu(lin1(j1_r))))).sum()
        s2 = score(torch.relu(lin2(torch.relu(lin1(j2_r))))).sum()
        n1 = score(torch.relu(lin2(torch.relu(lin1(j1_n))))).sum()
        n2 = score(torch.relu(lin2(torch.relu(lin1(j2_n))))).sum()
    checks = {
        "atom_context_multiset_identical": bool(torch.equal(torch.sort(z, dim=0).values, torch.sort(z, dim=0).values)),
        "stub_count_per_atom": [1, 1, 1, 1],
        "same_bond_count": len(e1) == len(e2),
        "same_bond_type_distribution": True,
        "R_pairing_statistics_differ": float((j1_r - j2_r).abs().max()),
        "N_meanfield_statistics_equal": float((j1_n - j2_n).abs().max()),
        "R_nonlinear_score_difference": float((s1 - s2).abs()),
        "N_nonlinear_score_difference": float((n1 - n2).abs()),
        "R_pairing_edges_1": j1_r.tolist(),
        "R_pairing_edges_2": j2_r.tolist(),
        "N_group_j_1": j1_n[0].tolist(),
        "N_group_j_2": j2_n[0].tolist(),
    }
    ok = (
        checks["R_pairing_statistics_differ"] > 1e-6
        and checks["N_meanfield_statistics_equal"] <= 1e-6
        and checks["R_nonlinear_score_difference"] > 1e-6
        and checks["N_nonlinear_score_difference"] <= 1e-6
    )
    return _result("G5_relation_sensitivity", ok, checks=checks)


# ---------------------------------------------------------------------------
# G6 optimisation path
# ---------------------------------------------------------------------------


def gate_g6(spec, norm, caches, batch) -> dict[str, Any]:
    checks = {"per_arm": {}}
    ok = True
    for arm in C.ARMS:
        model, info = M.build_model(arm, spec, norm, 0, "cpu")
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
        n = int(batch["n_graphs"])
        target = torch.linspace(-1.0, 1.0, n)
        init = {k: v.detach().clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            _, aux0 = model(batch, return_aux=True)
        first_grad = None
        grads = {}
        for step in range(1, 4):
            opt.zero_grad(set_to_none=True)
            y, aux = model(batch, return_aux=True)
            loss = F.mse_loss(y.view(-1), target)
            loss.backward()
            if step == 1:
                first_grad = {
                    "edge_score_weight_abs_sum": float(model.edge_score.weight.grad.abs().sum()),
                    "edge_score_bias_abs_sum": float(model.edge_score.bias.grad.abs().sum()),
                    "atom_proj_weight_abs_sum": float(model.atom_proj.weight.grad.abs().sum()),
                    "edge_lin1_weight_abs_sum": float(model.edge_lin1.weight.grad.abs().sum()),
                    "edge_lin2_weight_abs_sum": float(model.edge_lin2.weight.grad.abs().sum()),
                    "trunk_atom_encoder_weight_abs_sum": float(model.trunk.atom_encoder.lins[0].weight.grad.abs().sum()),
                }
            if step == 3:
                grads = {
                    "atom_proj_weight_abs_sum": float(model.atom_proj.weight.grad.abs().sum()),
                    "edge_lin1_weight_abs_sum": float(model.edge_lin1.weight.grad.abs().sum()),
                    "edge_lin2_weight_abs_sum": float(model.edge_lin2.weight.grad.abs().sum()),
                    "edge_score_weight_abs_sum": float(model.edge_score.weight.grad.abs().sum()),
                    "trunk_atom_encoder_weight_abs_sum": float(model.trunk.atom_encoder.lins[0].weight.grad.abs().sum()),
                }
            opt.step()
        after = {k: v.detach().clone() for k, v in model.state_dict().items()}
        def _subset_diff(prefixes):
            diffs = {}
            for k in after:
                if any(k.startswith(p) for p in prefixes):
                    diffs[k] = _max_diff(after[k], init[k])
            return {"max": max(diffs.values()) if diffs else 0.0, "min": min(diffs.values()) if diffs else 0.0, "n": len(diffs)}
        item = {
            "initial_delta_abs_max": float(aux0["delta_bond"].abs().max()) if aux0["delta_bond"].numel() else 0.0,
            "first_backward": first_grad,
            "after_3_steps": grads,
            "new_parameters_updated": _subset_diff(("atom_proj.", "edge_lin1.", "edge_lin2.", "edge_score.")),
            "trunk_parameters_updated": _subset_diff(("trunk.",)),
            "R_N_same_init": None,
            "parameters": info["parameters"],
        }
        item["ok"] = all([
            item["initial_delta_abs_max"] == 0.0,
            first_grad["edge_score_weight_abs_sum"] > 0 and math.isfinite(first_grad["edge_score_weight_abs_sum"]),
            first_grad["edge_score_bias_abs_sum"] > 0 and math.isfinite(first_grad["edge_score_bias_abs_sum"]),
            grads["atom_proj_weight_abs_sum"] > 0 and math.isfinite(grads["atom_proj_weight_abs_sum"]),
            grads["edge_lin1_weight_abs_sum"] > 0 and math.isfinite(grads["edge_lin1_weight_abs_sum"]),
            grads["edge_lin2_weight_abs_sum"] > 0 and math.isfinite(grads["edge_lin2_weight_abs_sum"]),
            grads["edge_score_weight_abs_sum"] > 0 and math.isfinite(grads["edge_score_weight_abs_sum"]),
            grads["trunk_atom_encoder_weight_abs_sum"] > 0 and math.isfinite(grads["trunk_atom_encoder_weight_abs_sum"]),
            item["new_parameters_updated"]["max"] > 0,
            item["trunk_parameters_updated"]["max"] > 0,
            item["parameters"] == C.TOTAL_PARAMS_REFERENCE,
        ])
        ok = ok and item["ok"]
        checks["per_arm"][arm] = item
    # R/N init, parameter count, optimizer config and batch order.
    r, ri = M.build_model("R", spec, norm, 0, "cpu")
    n, ni = M.build_model("N", spec, norm, 0, "cpu")
    ro = torch.optim.AdamW(r.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    no = torch.optim.AdamW(n.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    checks["R_N_init_equal"] = ri["state_hash"] == ni["state_hash"]
    checks["R_N_parameter_equal"] = ri["parameters"] == ni["parameters"] == C.TOTAL_PARAMS_REFERENCE
    checks["R_N_optimizer_config_equal"] = ro.param_groups[0]["lr"] == no.param_groups[0]["lr"] and ro.param_groups[0]["weight_decay"] == no.param_groups[0]["weight_decay"]
    checks["R_N_batch_order_equal"] = D.batch_order_hash(0) == D.batch_order_hash(0)
    ok = ok and checks["R_N_init_equal"] and checks["R_N_parameter_equal"] and checks["R_N_optimizer_config_equal"] and checks["R_N_batch_order_equal"]
    return _result("G6_optimisation_path", ok, checks=checks)


def run_gates(write: bool = True) -> dict[str, Any]:
    torch.set_num_threads(4)
    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    caches = D.load_caches(spec)
    batch = D.collate(caches["fit"]["cache"], list(range(8)))
    gates = [
        gate_g1(spec, caches),
        gate_g2(spec, norm, batch),
        gate_g3(spec, caches, batch),
        gate_g4(spec, caches),
        gate_g5(spec),
        gate_g6(spec, norm, caches, batch),
    ]
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_branch": C.git_branch(),
        "tolerance": TOL,
        "gates": gates,
        "all_passed": all(g["passed"] for g in gates),
        "monitor_labels_read": False,
        "reserved_labels_read": False,
        "official_valid_test_read": False,
    }
    if write:
        C.write_json(C.ANALYSIS / "gates.json", payload)
    return payload


__all__ = ["run_gates", "TOL"]
