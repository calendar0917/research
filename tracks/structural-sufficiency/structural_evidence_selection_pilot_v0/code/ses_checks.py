"""Pre-training correctness checks for SES-v0.

Every check is computed from the frozen cache, the prepared candidate artifact
and freshly built seed-0 models.  No formal-training result is read.  The output
is ``analysis/checks.json``; formal training must not start unless all checks
pass.
"""

from __future__ import annotations

import copy
import pickle
import socket
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import ses_common as C
import ses_data as D
import ses_structure as S
import ses_tensor as T
import ses_train as TR

import gwic_data as GD
import gwic_structure as GS
import gwic_tensor as GT

CHECKS: list[dict[str, Any]] = []


def add(name: str, passed: bool, detail: Any) -> None:
    CHECKS.append({"name": name, "status": "PASS" if passed else "FAIL",
                   "detail": C.jsonable(detail)})


def _pred(model: Any, structs: Sequence[S.Struct], device: str = "cpu",
          selector: Any = None, need_scores: bool = False) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        out = model(T.collate(structs, device), arm="SENTENCE_BASE",
                    return_aux=need_scores)
        y = out["yhat_std"].cpu().numpy().astype(np.float64)
        scores = None
        if need_scores:
            scores = selector(out["readout_in"].detach()).view(-1).cpu().numpy()
    return y, scores


def _permute_molecule(raw: Mapping[str, Any], row: int, rng: np.random.Generator
                      ) -> dict[str, Any]:
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
    b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
    n = a1 - a0
    m = b1 - b0
    new = {k: (np.asarray(v).copy() if isinstance(v, np.ndarray) else v)
           for k, v in raw.items()}
    perm = rng.permutation(n)
    old2new = np.empty(n, np.int64)
    old2new[perm] = np.arange(n, dtype=np.int64)
    new["atom_feat"][a0:a1] = np.asarray(raw["atom_feat"])[a0:a1][perm]
    for key in ("degree", "min_ring", "in_cycle"):
        new[key][a0:a1] = np.asarray(raw[key])[a0:a1][perm]
    border = rng.permutation(m)
    old_u = np.asarray(raw["bond_u"])[b0:b1][border]
    old_v = np.asarray(raw["bond_v"])[b0:b1][border]
    new["bond_u"][b0:b1] = old2new[old_u]
    new["bond_v"][b0:b1] = old2new[old_v]
    new["bond_feat"][b0:b1] = np.asarray(raw["bond_feat"])[b0:b1][border]
    return new


def _candidate_certs(structs: Sequence[S.Struct]) -> list[str]:
    return sorted(S.certificate(st.atom_feat, st.bond_u, st.bond_v, st.bond_feat)
                  for st in structs)


def check_data_identity(bundle: D.DataBundle, prepared: Mapping[str, Any]) -> None:
    detail = {
        "cache_sha256": bundle.hashes["raw_cache"]["sha256"],
        "split_identity_sha256": bundle.hashes["split_identity"],
        "rows": bundle.hashes["official_rows"],
        "labels": bundle.hashes["labels"],
        "vault": bundle.hashes["label_vault"],
        "target_scaler": bundle.scaler,
        "unseen_positions": bundle.stats["unseen_category_positions"],
    }
    vault = bundle.hashes["label_vault"]
    vault_ok = (True if not vault["available"]
                else vault["max_abs_diff"] is not None and vault["max_abs_diff"] <= 1e-6)
    ok = (bundle.hashes["raw_cache"]["sha256"] == C.RAW_CACHE_SHA256
          and bundle.train_rows.size == 3360 and bundle.valid_rows.size == 420
          and bundle.test_rows.size == 420
          and bundle.scaler["n"] == 3360 and vault_ok)
    add("C1_official_identity_labels_splits", bool(ok), detail)

    # normalisers recomputed from full Train only
    y = bundle.y_raw(bundle.train_rows)
    mean = float(y.mean())
    std = float(y.std(ddof=0))
    norm_ok = (abs(mean - bundle.scaler["mean"]) < 1e-12
               and abs(std - bundle.scaler["std"]) < 1e-12
               and bundle.scaler["n"] == int(bundle.train_rows.size))
    graph_re = bundle.graph_all[bundle.train_rows].mean(axis=0)
    graph_sd = bundle.graph_all[bundle.train_rows].std(axis=0)
    graph_sd = np.where(graph_sd > 1e-12, graph_sd, 1.0)
    atom_rows = [GD.atom_scalars(GD.molecule_view(bundle.raw, int(r)))
                 for r in bundle.train_rows.tolist()]
    atom_all = np.concatenate(atom_rows, axis=0)
    atom_mu = atom_all.mean(axis=0)
    atom_sd = atom_all.std(axis=0)
    atom_sd = np.where(atom_sd > 1e-12, atom_sd, 1.0)
    norm_ok = norm_ok and np.allclose(graph_re, bundle.graph_mean, atol=1e-12) \
        and np.allclose(graph_sd, bundle.graph_scale, atol=1e-12) \
        and np.allclose(atom_mu, bundle.atom_scalar_mean, atol=1e-12) \
        and np.allclose(atom_sd, bundle.atom_scalar_scale, atol=1e-12)
    add("C2_train_only_normalisers", bool(norm_ok), {
        "target": bundle.scaler,
        "graph_mean": bundle.graph_mean, "graph_scale": bundle.graph_scale,
        "atom_mean": bundle.atom_scalar_mean, "atom_scale": bundle.atom_scalar_scale,
        "n_train_atoms": int(atom_all.shape[0]),
    })
    add("C3_prepared_artifact_identity", bool(
        prepared["raw_cache_sha256"] == C.RAW_CACHE_SHA256
        and prepared["split_identity_sha256"] == bundle.hashes["split_identity"]
        and len(prepared["full_structs"]) == bundle.labels.size),
        {"n_candidates": prepared["stats"]["n_candidates"],
         "candidate_count_histogram": prepared["stats"]["candidate_count_histogram"]})


def check_candidates(bundle: D.DataBundle, prepared: Mapping[str, Any]) -> None:
    raw = bundle.raw
    bad_connect = 0
    bad_edges = 0
    bad_size = 0
    bad_dup = 0
    below_k = 0
    n_candidates = 0
    n_atoms_one = 0
    max_cands = 0
    for row in range(int(bundle.labels.size)):
        sts = prepared["candidates"][row]
        n_candidates += len(sts)
        max_cands = max(max_cands, len(sts))
        full = prepared["full_structs"][row]
        k = int(sts[0].target_k)
        if full.n_atoms == 1:
            n_atoms_one += 1
        seen = set()
        for st in sts:
            support = tuple(int(x) for x in st.support_atoms.tolist())
            if support in seen:
                bad_dup += 1
            seen.add(support)
            # connectivity
            adj, _ = S._adjacency(st.n_atoms, st.bond_u, st.bond_v)
            seen_nodes = {0}
            stack = [0]
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if y not in seen_nodes:
                        seen_nodes.add(y)
                        stack.append(y)
            if len(seen_nodes) != st.n_atoms:
                bad_connect += 1
            if st.n_atoms > k:
                bad_size += 1
            if st.n_atoms < k:
                below_k += 1
            # exact induced edges from the raw physical graph
            atom_set = set(support)
            expected = set()
            for e in range(int(raw["bond_ptr"][row]), int(raw["bond_ptr"][row + 1])):
                u = int(raw["bond_u"][e])
                v = int(raw["bond_v"][e])
                if u in atom_set and v in atom_set:
                    expected.add(tuple(sorted((u, v))))
            got = {tuple(sorted((int(st.support_atoms[u]), int(st.support_atoms[v]))))
                   for u, v in zip(st.bond_u, st.bond_v)}
            if got != expected:
                bad_edges += 1
            if len(st.support_bonds) != st.n_bonds:
                bad_edges += 1
    ok = (bad_connect == 0 and bad_edges == 0 and bad_size == 0 and bad_dup == 0
          and max_cands <= 4 and below_k == prepared["stats"]["candidates_below_k"])
    add("C4_candidates_connected_induced_dedup", bool(ok), {
        "n_candidates": n_candidates, "bad_connectivity": bad_connect,
        "bad_induced_edges": bad_edges, "bad_size_above_k": bad_size,
        "duplicate_supports": bad_dup, "candidates_below_k": below_k,
        "rows_with_component_below_k": prepared["stats"]["rows_with_component_below_k"],
        "max_candidates_per_molecule": max_cands, "n_atoms_one_rows": n_atoms_one,
    })
    add("C5_candidate_budget_and_coverage", bool(
        n_candidates == prepared["stats"]["n_candidates"]
        and prepared["stats"]["splits"]["valid"]["molecules_with_candidate_below_k"] == 1),
        prepared["stats"]["splits"])


def check_internal_fields(bundle: D.DataBundle, prepared: Mapping[str, Any],
                          rng: np.random.Generator) -> None:
    bad = []
    sample = rng.choice(int(bundle.labels.size), size=300, replace=False)
    for row in sample.tolist():
        for st in prepared["candidates"][int(row)]:
            if st.n_atoms > 0:
                if st.dist1_atoms.size and (st.dist1_atoms.min() < 0
                                            or st.dist1_atoms.max() >= st.n_atoms):
                    bad.append((row, "dist1"))
                if st.dist2_atoms.size and (st.dist2_atoms.min() < 0
                                            or st.dist2_atoms.max() >= st.n_atoms):
                    bad.append((row, "dist2"))
                if st.path_end_atom.size and (st.path_end_atom.min() < 0
                                              or st.path_end_atom.max() >= st.n_atoms):
                    bad.append((row, "path_atom"))
                if st.r2_bond.size and (st.r2_bond.min() < 0
                                        or st.r2_bond.max() >= st.n_bonds):
                    bad.append((row, "r2_bond"))
            if st.graph_scalars[0] != np.log1p(st.n_atoms) or \
                    st.graph_scalars[1] != np.log1p(st.n_bonds):
                bad.append((row, "graph_scalars_not_local"))
    add("C6_s1_s2_r2_only_selected_structure", len(bad) == 0, {
        "sampled_molecules": 300, "violations": bad[:10]})


def _model_bundle(bundle: D.DataBundle, cfg: Mapping[str, Any]):
    atom_dims, bond_dims = D.atom_bond_dims()
    model, info = C.build_predictor(atom_dims, bond_dims, bundle, cfg, int(cfg["seed"]))
    model.zero_unseen(bundle.atom_ok, bundle.bond_ok, atom_dims, bond_dims)
    return model, info, atom_dims, bond_dims


def check_single_vs_batch(model: Any, bundle: D.DataBundle,
                          prepared: Mapping[str, Any], rng: np.random.Generator) -> None:
    rows = rng.choice(int(bundle.labels.size), size=30, replace=False)
    single, batched = [], []
    all_structs: list[S.Struct] = []
    for row in rows.tolist():
        sts = prepared["candidates"][int(row)]
        all_structs.extend(sts)
        for st in sts[:1]:
            y, _ = _pred(model, [st])
            single.append(float(y[0]))
    y_all, _ = _pred(model, all_structs)
    idx = 0
    for row in rows.tolist():
        sts = prepared["candidates"][int(row)]
        batched.append(float(y_all[idx]))
        idx += len(sts)
    diff = float(np.max(np.abs(np.asarray(single) - np.asarray(batched))))
    add("C7_single_vs_batch_candidate_prediction", diff <= 1e-6, {
        "max_abs_diff": diff, "n_candidates": len(single)})


def check_outside_modification(bundle: D.DataBundle, prepared: Mapping[str, Any],
                               model: Any, rng: np.random.Generator) -> None:
    raw = bundle.raw
    rows = rng.choice(int(bundle.labels.size), size=8, replace=False)
    max_diff = 0.0
    n = 0
    for row in rows.tolist():
        row = int(row)
        sts = prepared["candidates"][row]
        st = sts[0]
        mod = {k: (np.asarray(v).copy() if isinstance(v, np.ndarray) else v)
               for k, v in raw.items()}
        a0 = int(np.asarray(raw["node_ptr"])[row])
        a1 = int(np.asarray(raw["node_ptr"])[row + 1])
        support = set(int(x) for x in st.support_atoms.tolist())
        outside = [i for i in range(int(a1 - a0)) if i not in support]
        for i in outside:
            mod["atom_feat"][a0 + i, 0] = (int(mod["atom_feat"][a0 + i, 0]) + 1) % 119
            mod["degree"][a0 + i] = int(mod["degree"][a0 + i]) + 7
            mod["min_ring"][a0 + i] = int(mod["min_ring"][a0 + i]) + 9
            mod["in_cycle"][a0 + i] = 1 - int(mod["in_cycle"][a0 + i])
        b0 = int(np.asarray(raw["bond_ptr"])[row])
        b1 = int(np.asarray(raw["bond_ptr"])[row + 1])
        for e in range(b0, b1):
            u = int(np.asarray(raw["bond_u"])[e])
            v = int(np.asarray(raw["bond_v"])[e])
            if u not in support and v not in support:
                mod["bond_feat"][e, 0] = (int(mod["bond_feat"][e, 0]) + 1) % 5
        spec = S.CandidateSpec(
            source_rank=st.source_rank,
            ranks=tuple(int(x) for x in st.support_atoms.tolist()),
            original_atoms=st.support_atoms.copy(),
            original_bonds=st.support_bonds.copy(),
            target_k=st.target_k, component_size=st.component_size)
        st2 = S.build_candidate_struct(mod, row, spec)
        for key in ("atom_feat", "bond_feat", "bond_u", "bond_v", "degree",
                    "min_ring", "in_cycle", "atom_scalars", "graph_scalars",
                    "dist1_atoms", "dist2_atoms", "path_cat", "path_end_type",
                    "path_closure", "path_end_atom", "r2_bond", "r2_bond_bucket"):
            if not np.array_equal(np.asarray(getattr(st, key)), np.asarray(getattr(st2, key))):
                raise AssertionError(f"outside modification changed {key}")
        y1, _ = _pred(model, [st])
        y2, _ = _pred(model, [st2])
        max_diff = max(max_diff, float(abs(y1[0] - y2[0])))
        n += 1
    add("C8_outside_structure_cannot_change_candidate", max_diff <= 1e-8,
        {"n_candidates": n, "max_abs_prediction_diff": max_diff})


def check_other_candidate_independence(bundle: D.DataBundle, prepared: Mapping[str, Any],
                                       model: Any, rng: np.random.Generator) -> None:
    rows = rng.choice(int(bundle.labels.size), size=20, replace=False)
    max_y = 0.0
    max_s = 0.0
    for row in rows.tolist():
        sts = prepared["candidates"][int(row)]
        if len(sts) < 2:
            continue
        base_y, _ = _pred(model, [sts[0]])
        shuffled = list(sts[1:]) + [sts[0]]
        y_all, _ = _pred(model, shuffled)
        max_y = max(max_y, float(abs(y_all[-1] - base_y[0])))
        removed_y, _ = _pred(model, [sts[0]])
        max_y = max(max_y, float(abs(removed_y[0] - base_y[0])))
    add("C9_other_candidates_do_not_change_prediction", max_y <= 1e-6,
        {"max_abs_prediction_diff": max_y})


def check_relabelling(bundle: D.DataBundle, prepared: Mapping[str, Any],
                      model: Any, rng: np.random.Generator) -> dict[str, Any]:
    raw = bundle.raw
    rows = rng.choice(int(bundle.labels.size), size=12, replace=False)
    cert_ok = True
    pred_diff = 0.0
    dense_diff = 0.0
    for row in rows.tolist():
        row = int(row)
        old_structs = prepared["candidates"][row]
        old_certs = _candidate_certs(old_structs)
        old_y, _ = _pred(model, old_structs)
        new_raw = _permute_molecule(raw, row, rng)
        specs, _ = S.generate_candidates(new_raw, row)
        new_structs = [S.build_candidate_struct(new_raw, row, sp) for sp in specs]
        new_certs = _candidate_certs(new_structs)
        if old_certs != new_certs:
            cert_ok = False
        new_y, _ = _pred(model, new_structs)
        # compare prediction multiset, which is the relabelling-invariant statement
        pred_diff = max(pred_diff, float(np.max(np.abs(np.sort(old_y) - np.sort(new_y)))))
        dense_old, _ = _pred(model, [prepared["full_structs"][row]])
        full_new = S.canonical_full_struct(new_raw, row)
        dense_new, _ = _pred(model, [full_new])
        dense_diff = max(dense_diff, float(abs(dense_old[0] - dense_new[0])))
    add("C10_candidate_generation_and_encoder_relabelling", bool(cert_ok and pred_diff <= 1e-6),
        {"n_molecules": int(rows.size), "canonical_certificate_sets_equal": cert_ok,
         "max_sorted_prediction_diff": pred_diff,
         "max_dense_prediction_diff": dense_diff})

    # candidate-local renumbering: permute local atoms and bond order
    local_diff = 0.0
    for row in rows[:6].tolist():
        st = prepared["candidates"][int(row)][0]
        n = st.n_atoms
        perm = rng.permutation(n)
        old2new = np.empty(n, np.int64)
        old2new[perm] = np.arange(n, dtype=np.int64)
        atom_feat = st.atom_feat[perm]
        degree = st.degree[perm]
        min_ring = st.min_ring[perm]
        in_cycle = st.in_cycle[perm]
        bu = old2new[st.bond_u]
        bv = old2new[st.bond_v]
        pair = np.stack([np.minimum(bu, bv), np.maximum(bu, bv)], axis=1)
        border = np.lexsort((pair[:, 1], pair[:, 0]))
        st2 = S.build_struct(
            atom_feat=atom_feat, bond_feat=st.bond_feat[border],
            bond_u=pair[border, 0], bond_v=pair[border, 1],
            degree=degree, min_ring=min_ring, in_cycle=in_cycle,
            graph_scalars=st.graph_scalars, row=st.row)
        y1, _ = _pred(model, [st])
        y2, _ = _pred(model, [st2])
        local_diff = max(local_diff, float(abs(y1[0] - y2[0])))
    add("C11_candidate_local_renumbering", local_diff <= 1e-6,
        {"max_abs_prediction_diff": local_diff})
    return {"cert_ok": cert_ok, "pred_diff": pred_diff, "dense_diff": dense_diff}


def check_select_argmax(bundle: D.DataBundle, prepared: Mapping[str, Any],
                        model: Any, selector: Any, rng: np.random.Generator) -> None:
    rows = rng.choice(bundle.valid_rows, size=40, replace=False)
    res = TR.eval_arm(model, selector, "SELECT", rows, prepared, bundle, "cpu",
                      64, return_candidates=True)
    ok = True
    max_diff = 0.0
    for i, row in enumerate(rows.tolist()):
        lo, hi = int(res["cand_ptr"][i]), int(res["cand_ptr"][i + 1])
        sc = res["cand_scores"][lo:hi]
        j = int(np.argmax(sc))
        if j != int(res["chosen_index"][i]):
            ok = False
        if abs(res["pred_select"][i] - res["cand_pred"][lo + j]) > 0:
            ok = False
        # independent single encoding of the chosen candidate
        y_one, _ = _pred(model, [prepared["candidates"][int(row)][j]])
        p_one = float(y_one[0]) * bundle.scaler["std"] + bundle.scaler["mean"]
        max_diff = max(max_diff, abs(p_one - res["pred_select"][i]))
    add("C12_select_eval_is_argmax_single_candidate", bool(ok and max_diff <= 1e-6), {
        "n_molecules": int(rows.size), "max_abs_diff_vs_independent": max_diff})


def check_gradients(bundle: D.DataBundle, prepared: Mapping[str, Any],
                    model: Any, selector: Any, cfg: Mapping[str, Any]) -> None:
    for module in (model, selector):
        module.zero_grad(set_to_none=True)
    model.train()
    selector.train()
    rows = np.asarray(bundle.train_rows[:64], np.int64)
    structs, mol_index, _ = TR._chunk_structs(rows, prepared)
    loss, _ = TR._candidate_loss(model, selector, "SELECT", structs, mol_index,
                                 bundle.y_std(rows), rows.size, "cpu")
    loss.backward()
    pred_norm = float(sum(float(p.grad.abs().sum()) for p in model.parameters()
                          if p.grad is not None))
    sel_norm = float(sum(float(p.grad.abs().sum()) for p in selector.parameters()
                         if p.grad is not None))
    finite = all(torch.isfinite(p.grad).all().item()
                 for p in list(model.parameters()) + list(selector.parameters())
                 if p.grad is not None)
    n_pred = sum(1 for p in model.parameters() if p.grad is not None
                 and float(p.grad.abs().sum()) > 0)
    n_sel = sum(1 for p in selector.parameters() if p.grad is not None
                and float(p.grad.abs().sum()) > 0)
    add("C13_select_predictor_and_head_gradients", bool(
        finite and pred_norm > 0 and sel_norm > 0 and n_pred > 0 and n_sel == 4),
        {"predictor_abs_grad_sum": pred_norm, "selector_abs_grad_sum": sel_norm,
         "predictor_params_with_grad": n_pred, "selector_params_with_grad": n_sel,
         "loss": float(loss.detach())})
    model.zero_grad(set_to_none=True)
    selector.zero_grad(set_to_none=True)


def check_shared_init(bundle: D.DataBundle, cfg: Mapping[str, Any]) -> None:
    models = {}
    infos = {}
    for arm in TR.ARMS:
        model, info, _, _ = _model_bundle(bundle, cfg)
        models[arm] = model
        infos[arm] = info
    ref = C.array_sha256(*[p.detach().cpu().numpy()
                           for _, p in sorted(models["DENSE"].state_dict().items())])
    equal = True
    hashes = {}
    for arm, model in models.items():
        h = C.array_sha256(*[p.detach().cpu().numpy()
                             for _, p in sorted(model.state_dict().items())])
        hashes[arm] = h
        equal = equal and h == ref
    counts = {arm: int(info["parameters"]["total"]) for arm, info in infos.items()}
    add("C14_shared_predictor_initial_state_bitwise", bool(
        equal and all(v == C.PARAMS_TOTAL for v in counts.values())),
        {"state_sha256": hashes, "parameter_counts": counts})


def check_parameter_ledgers(bundle: D.DataBundle, cfg: Mapping[str, Any]) -> None:
    model, info, atom_dims, bond_dims = _model_bundle(bundle, cfg)
    selector = TR.build_selector(model.readout_in_dim, int(cfg["seed"]))
    pred = int(sum(p.numel() for p in model.parameters()))
    sel = int(sum(p.numel() for p in selector.parameters()))
    # RANDOM has no selector; predictor count is identical by construction
    ok = (pred == C.PARAMS_TOTAL and sel == 28289
          and int(info["readout_in_dim"]) == 440)
    add("C15_parameter_ledgers", bool(ok), {
        "predictor_parameters": pred, "selector_parameters": sel,
        "readout_in_dim": int(info["readout_in_dim"]),
        "selector_only_in_select": True})


def check_dense_gwic_equivalence(bundle: D.DataBundle, cfg: Mapping[str, Any],
                                 rng: np.random.Generator) -> None:
    model, _, _, _ = _model_bundle(bundle, cfg)
    model.eval()
    rows = rng.choice(int(bundle.labels.size), size=12, replace=False).tolist()
    gs_structs = {}
    for row in rows:
        view = GD.molecule_view(bundle.raw, int(row))
        view["atom_scalars"] = GD.atom_scalars(view)
        gs_structs[int(row)] = GS.build_molecule_struct(view)
    batch = GT.collate(rows, gs_structs, bundle.raw, "cpu")
    with torch.no_grad():
        y0 = model(batch, arm="SENTENCE_BASE")["yhat_std"].cpu().numpy()
    ses_structs = [S.canonical_full_struct(bundle.raw, int(r)) for r in rows]
    y1, _ = _pred(model, ses_structs)
    diff = float(np.max(np.abs(y0 - y1)))
    add("C16_dense_matches_frozen_gwic_full_graph", diff <= 1e-6, {
        "n_molecules": len(rows), "max_abs_diff": diff})


def check_forbidden_inputs(bundle: D.DataBundle, prepared: Mapping[str, Any],
                           model: Any) -> None:
    st = prepared["candidates"][int(bundle.valid_rows[0])][0]
    batch = T.collate([st], "cpu")
    allowed = {"atom_feat", "bond_feat", "bond_index", "atom_scalars", "atom_mol",
               "bond_mol", "graph_scalars", "dist1_atoms", "dist2_atoms",
               "dist1_seg", "dist2_seg", "path_cat", "path_end_type",
               "path_closure", "path_end_atom", "path_seg", "r2_bond",
               "r2_bucket", "r2_seg", "r2_bucket_seg", "n_molecules", "n_atoms",
               "n_bonds", "mol_atom_ptr"}
    forbidden_present = sorted(set(batch) - allowed)
    # candidate graph scalars must be candidate-local, never full-graph values
    local_scalars_ok = (abs(float(st.graph_scalars[0]) - float(np.log1p(st.n_atoms))) < 1e-12
                        and abs(float(st.graph_scalars[1]) - float(np.log1p(st.n_bonds))) < 1e-12)
    # all support atoms still carry their original raw categorical columns
    row = int(bundle.valid_rows[0])
    a0 = int(bundle.raw["node_ptr"][row])
    raw_atoms = np.asarray(bundle.raw["atom_feat"])[a0 + st.support_atoms]
    atom_fields_ok = np.array_equal(np.asarray(raw_atoms), np.asarray(st.atom_feat))
    raw_bonds = np.asarray(bundle.raw["bond_feat"])[st.support_bonds]
    bond_fields_ok = np.array_equal(np.asarray(raw_bonds), np.asarray(st.bond_feat))
    add("C17_no_forbidden_predictor_inputs", bool(
        not forbidden_present and local_scalars_ok and atom_fields_ok and bond_fields_ok),
        {"batch_keys": sorted(batch), "forbidden_present": forbidden_present,
         "candidate_local_graph_scalars": local_scalars_ok,
         "raw_atom_columns_preserved": atom_fields_ok,
         "raw_bond_columns_preserved": bond_fields_ok})


def main(prepared_path: str | None = None) -> dict[str, Any]:
    CHECKS.clear()
    C.assert_frozen_sources()
    bundle = D.load_bundle()
    cfg = C.load_config()
    path = prepared_path or str(C.PREPARED_DIR / "ses_prepared.pkl")
    prepared = TR.load_prepared(path, bundle)
    rng = np.random.default_rng(20261124)
    check_data_identity(bundle, prepared)
    check_candidates(bundle, prepared)
    check_internal_fields(bundle, prepared, rng)
    model, info, atom_dims, bond_dims = _model_bundle(bundle, cfg)
    selector = TR.build_selector(model.readout_in_dim, int(cfg["seed"]))
    check_single_vs_batch(model, bundle, prepared, rng)
    check_outside_modification(bundle, prepared, model, rng)
    check_other_candidate_independence(bundle, prepared, model, rng)
    check_relabelling(bundle, prepared, model, rng)
    check_select_argmax(bundle, prepared, model, selector, rng)
    check_gradients(bundle, prepared, model, selector, cfg)
    check_shared_init(bundle, cfg)
    check_parameter_ledgers(bundle, cfg)
    check_dense_gwic_equivalence(bundle, cfg, rng)
    check_forbidden_inputs(bundle, prepared, model)
    passed = all(c["status"] == "PASS" for c in CHECKS)
    payload = {
        "host": socket.gethostname(),
        "commit": C.git_commit(),
        "prepared_path": path,
        "prepared_fixed_index_sha256": prepared["stats"]["fixed_index"]["sha256"],
        "prepared_n_candidates": prepared["stats"]["n_candidates"],
        "checks": CHECKS,
        "n_checks": len(CHECKS),
        "n_pass": int(sum(c["status"] == "PASS" for c in CHECKS)),
        "all_pass": bool(passed),
        "code_sha256": C.code_sha(("ses_common.py", "ses_data.py", "ses_structure.py",
                                   "ses_tensor.py", "ses_train.py", "ses_checks.py")),
    }
    C.write_json(C.ANALYSIS_DIR / "checks.json", payload)
    print(f"[checks] {payload['n_pass']}/{payload['n_checks']} PASS "
          f"all_pass={passed}", flush=True)
    for c in CHECKS:
        if c["status"] != "PASS":
            print(f"  FAIL {c['name']}: {c['detail']}", flush=True)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", default=None)
    args = parser.parse_args()
    main(args.prepared)
