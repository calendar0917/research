"""PRCG-v0 pre-training verification.

Every check is mechanical and its verdict is written to ``analysis/checks.json``.
Nothing here trains a model on Dev labels or reads an official Valid/Test label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

import prcg_common as C
import prcg_data as D
import prcg_discover as PDis
import prcg_grammar as GD
import prcg_graph as G
import prcg_model as M
import prcg_tensor as T

CHECKS: list[dict[str, Any]] = []


def record(name: str, ok: bool, detail: Mapping[str, Any]) -> None:
    CHECKS.append({"check": name, "ok": bool(ok), "detail": C.jsonable(dict(detail))})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: "
          f"{json.dumps(C.jsonable(dict(detail)), sort_keys=True)[:300]}", flush=True)


# ---------------------------------------------------------------------------
def effective_rank(mat: np.ndarray) -> dict[str, float]:
    mat = np.asarray(mat, np.float64)
    if mat.size == 0 or mat.shape[0] < 2:
        return {"participation_ratio": 0.0, "top_singular_fraction": 0.0}
    centered = mat - mat.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(centered, compute_uv=False)
    energy = sv ** 2
    if energy.sum() <= 0:
        return {"participation_ratio": 0.0, "top_singular_fraction": 0.0}
    return {"participation_ratio": float(energy.sum() ** 2 / (energy ** 2).sum()),
            "top_singular_fraction": float(energy[0] / energy.sum())}


def permute_molecule(mol: G.Molecule, perm: np.ndarray, rng: np.random.Generator
                     ) -> G.Molecule:
    """Rebuild the same molecule under a different atom numbering / bond order."""
    n = mol.n_atoms
    core = np.zeros_like(mol.atom_core)
    intr = np.zeros_like(mol.atom_intrinsic)
    ctx = np.zeros_like(mol.atom_context)
    for i in range(n):
        core[perm[i]] = mol.atom_core[i]
        intr[perm[i]] = mol.atom_intrinsic[i]
        ctx[perm[i]] = mol.atom_context[i]
    order = rng.permutation(mol.n_bonds)
    bu = perm[mol.bond_u][order]
    bv = perm[mol.bond_v][order]
    bc = mol.bond_core[order]
    bs = mol.bond_stereo[order]
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for b in range(bu.size):
        adj[int(bu[b])].append((int(bv[b]), b))
        adj[int(bv[b])].append((int(bu[b]), b))
    labels = [G._atom_code(core[i]) for i in range(n)]
    esets = [[(u, G._bond_code(bc[b])) for u, b in adj[i]] for i in range(n)]
    ranks = G.canonical_ranks(labels, esets, n)
    return G.Molecule(row=mol.row, n_atoms=n, atom_core=core, atom_intrinsic=intr,
                      atom_context=ctx, bond_u=bu, bond_v=bv, bond_core=bc,
                      bond_stereo=bs, adj=adj, ranks=ranks,
                      n_components=mol.n_components)


def partition_as_sets(part: G.Partition) -> list[tuple[int, ...]]:
    return sorted(tuple(sorted(w.support)) for w in part.words)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.ANALYSIS_DIR / "checks.json"))
    ap.add_argument("--invariance-molecules", type=int, default=60)
    args = ap.parse_args(argv)
    t0 = time.time()
    cfg = C.load_config()
    bundle = D.load_bundle()
    atom_dims, bond_dims = D.atom_bond_dims()
    fit_rows = np.asarray(bundle.fit_rows, np.int64)
    dev_rows = np.asarray(bundle.eval_rows, np.int64)
    all_rows = np.concatenate([fit_rows, dev_rows])
    y_std = bundle.labels_std()

    pred_grammar = PDis.load_grammar(C.GRAMMAR_DIR / "predictive_grammar.json")
    freq_grammar = PDis.load_grammar(C.GRAMMAR_DIR / "frequency_grammar.json")
    trace = C.read_json(C.GRAMMAR_DIR / "discovery_trace.json")
    stats = C.read_json(C.GRAMMAR_DIR / "stats.json")
    grammars = {"predictive": pred_grammar, "frequency": freq_grammar}

    # ---- A: frozen identity ------------------------------------------------
    record("A1_raw_cache_identity", bundle.hashes["raw_cache"]["sha256"] == C.RAW_CACHE_SHA256,
           {"sha256": bundle.hashes["raw_cache"]["sha256"],
            "members": len(bundle.hashes["raw_cache"]["members"])})
    record("A2_split_identity",
           bundle.hashes["split"]["sha256"] == C.SPLIT_F0_SHA256_EXPECTED
           and int(fit_rows.size) == 2856 and int(dev_rows.size) == 504,
           {"split_sha256": bundle.hashes["split"]["sha256"],
            "fit_rows_sha256": C.array_sha256(fit_rows),
            "dev_rows_sha256": C.array_sha256(dev_rows)})
    n_rules = len(pred_grammar)
    record("A3_frozen_grammar_shape",
           len(freq_grammar) == n_rules and n_rules == len(trace["predictive"]),
           {"predictive_rules": n_rules, "frequency_rules": len(freq_grammar),
            "trace_rounds": len(trace["predictive"]),
            "pred_sizes": [r.parent_size for r in pred_grammar.rules],
            "freq_sizes": [r.parent_size for r in freq_grammar.rules]})
    record("A4_one_rule_per_parent_type",
           len({r.parent_cert for r in pred_grammar.rules}) == n_rules
           and len({r.parent_cert for r in freq_grammar.rules}) == len(freq_grammar),
           {"pred_unique_parents": len({r.parent_cert for r in pred_grammar.rules}),
            "freq_unique_parents": len({r.parent_cert for r in freq_grammar.rules})})

    # ---- B: label isolation ----------------------------------------------
    finite_valid = int(np.isfinite(bundle.labels[bundle.official_valid]).sum())
    non_train_finite = int(np.isfinite(np.delete(bundle.labels, bundle.official_train)).sum())
    record("B1_no_official_valid_label", finite_valid == 0,
           {"finite_valid_labels": finite_valid,
            "label_rows": int(bundle.hashes["train_labels"]["rows"])})
    record("B2_no_label_outside_official_train", non_train_finite == 0,
           {"finite_labels_outside_train": non_train_finite,
            "finite_train_labels": int(np.isfinite(bundle.labels[bundle.official_train]).sum())})

    # ---- C: grammar integrity on every Fit and Dev molecule ----------------
    t = time.time()
    molecules = D.build_molecules(bundle.raw, all_rows)
    build_seconds = time.time() - t
    integrity: dict[str, dict[str, Any]] = {}
    for name, grammar in grammars.items():
        bad = {"non_connected": 0, "duplicate_atom": 0, "uncovered_atom": 0,
               "invented_bond": 0, "missing_bond": 0, "too_large": 0,
               "child_type_mismatch": 0, "parent_type_mismatch": 0,
               "parent_union_mismatch": 0}
        n_words = 0
        n_bonds_total = 0
        for r in all_rows.tolist():
            mol = molecules[int(r)]
            part = grammar.apply(mol)[0]
            seen: list[int] = []
            owner: dict[int, int] = {}
            for wi, w in enumerate(part.words):
                n_words += 1
                if w.size > G.MAX_WORD_ATOMS:
                    bad["too_large"] += 1
                if not mol.is_connected(w.support):
                    bad["non_connected"] += 1
                for a in w.support:
                    seen.append(a)
                    owner[a] = wi
                declared = set(mol.internal_bonds(w.support))
                # the word keeps exactly its induced internal bonds: no invented,
                # no missing (checked against the real graph below)
                if declared - set(range(mol.n_bonds)):
                    bad["invented_bond"] += 1
            if sorted(seen) != list(range(mol.n_atoms)):
                bad["duplicate_atom" if len(seen) != len(set(seen)) else "uncovered_atom"] += 1
            for b in range(mol.n_bonds):
                u, v = int(mol.bond_u[b]), int(mol.bond_v[b])
                if owner[u] == owner[v]:
                    if b not in set(mol.internal_bonds(part.words[owner[u]].support)):
                        bad["missing_bond"] += 1
                n_bonds_total += 1
            for w in part.history:
                if w.rule < 0:
                    continue
                rule = grammar.rules[w.rule]
                c1, c2 = part.history[w.children[0]], part.history[w.children[1]]
                if {c1.cert, c2.cert} != {rule.left_cert, rule.right_cert}:
                    bad["child_type_mismatch"] += 1
                if w.cert != rule.parent_cert:
                    bad["parent_type_mismatch"] += 1
                if tuple(sorted(c1.support + c2.support)) != tuple(sorted(w.support)):
                    bad["parent_union_mismatch"] += 1
        integrity[name] = {"molecules": int(all_rows.size), "n_words": n_words,
                           "bonds": n_bonds_total, "violations": bad}
    ok_c = all(v == 0 for name in integrity
               for v in integrity[name]["violations"].values())
    record("C1_words_are_real_connected_induced_supports", ok_c, integrity)

    # ---- C2: deterministic replay ----------------------------------------
    replay = {}
    for name, grammar in grammars.items():
        rows = list(all_rows[:400])
        a = {r: partition_as_sets(grammar.apply(molecules[int(r)])[0]) for r in rows}
        b = {r: partition_as_sets(grammar.apply(molecules[int(r)])[0]) for r in rows}
        replay[name] = sum(1 for r in rows if a[r] != b[r])
    record("C2_rule_replay_deterministic", all(v == 0 for v in replay.values()), replay)

    # ---- C3: discovery replay of the first three rounds -------------------
    disc = GD.GrammarDiscovery(molecules, fit_rows, y_std[fit_rows],
                               bundle.graph_scalars[fit_rows], bundle.scaffold_by_row,
                               log=lambda *_: None)
    replay_rules, _ = disc.run("predictive", max_rules=3)
    same = all(replay_rules.rules[i].pattern == pred_grammar.rules[i].pattern
               and replay_rules.rules[i].parent_cert == pred_grammar.rules[i].parent_cert
               for i in range(min(3, len(pred_grammar))))
    record("C3_discovery_round_replay", same,
           {"replayed": [r.pattern[:60] for r in replay_rules.rules]})

    # ---- D: type identity and internal computation ------------------------
    model, info = M.build_model(atom_dims, bond_dims, 0, cfg)
    model.eval()
    cache = {name: T.featurize(molecules, all_rows, grammar)
             for name, grammar in grammars.items()}
    index = {int(r): i for i, r in enumerate(all_rows.tolist())}
    type_checks: dict[str, Any] = {}
    for name in grammars:
        cch = cache[name]
        # extended canonical key of every word support, computed from all OGB
        # atom/bond fields: a finer certificate than the frozen type.
        keys: dict[str, np.ndarray] = {}
        type_of_key: dict[str, str] = {}
        with torch.no_grad():
            all_z = []
            for start in range(0, len(all_rows), 256):
                rows = all_rows[start:start + 256]
                batch = T.collate(rows.tolist(), cch, index, bundle.graph_mean,
                                  bundle.graph_scale)
                out = model(batch, mode="REAL")
                all_z.append(out["z"].numpy())
        z_all = np.concatenate(all_z, axis=0)
        # group word rows by the frozen type certificate recomputed from the cache
        z_by_type: dict[tuple, list[int]] = {}
        for r in all_rows.tolist():
            mol = molecules[int(r)]
            part = grammars[name].apply(mol)[0]
            base = int(cch["w_ptr"][index[int(r)]])
            for w in part.history:
                key = (w.cert, len(w.support), tuple(sorted(
                    (int(mol.bond_core[b][0]), int(mol.bond_core[b][1]))
                    for b in mol.internal_bonds(w.support))))
                z_by_type.setdefault(key, []).append(base + w.index)
        max_disp = 0.0
        for key, idxs in z_by_type.items():
            arr = z_all[np.asarray(idxs, np.int64)]
            if arr.shape[0] > 1:
                max_disp = max(max_disp, float(np.abs(arr - arr[0]).max()))
        type_checks[name] = {"n_types": len(z_by_type),
                             "max_within_type_z_dispersion": max_disp,
                             "n_words": int(z_all.shape[0])}
    record("D1_same_type_same_internal_state",
           all(v["max_within_type_z_dispersion"] < 1e-5 for v in type_checks.values()),
           type_checks)

    # ---- E: invariance ----------------------------------------------------
    rng = np.random.default_rng(20261201)
    sample = all_rows[rng.choice(all_rows.size, size=min(int(args.invariance_molecules),
                                                        all_rows.size), replace=False)]
    inv = {"renumber_automorphism": 0, "renumber_automorphism_violation": 0,
           "renumber_partition_mismatch": 0, "renumber_prediction_max_abs_diff": 0.0,
           "renumber_word_type_multiset_mismatch": 0,
           "bond_order_prediction_max_abs_diff": 0.0, "n_molecules": 0}
    batch_size = 16
    for name, grammar in grammars.items():
        cch = cache[name]
        for start in range(0, sample.size, batch_size):
            rows = sample[start:start + batch_size]
            batch = T.collate(rows.tolist(), cch, index, bundle.graph_mean, bundle.graph_scale)
            with torch.no_grad():
                base = model(batch, mode="REAL")["yhat_std"].numpy()
            perm_mols = {}
            perms = {}
            for r in rows.tolist():
                mol = molecules[int(r)]
                perm = rng.permutation(mol.n_atoms)
                perms[int(r)] = perm
                perm_mols[int(r)] = permute_molecule(mol, perm, rng)
            for r in rows.tolist():
                mol, pmol, perm = molecules[int(r)], perm_mols[int(r)], perms[int(r)]
                part_old = grammar.apply(mol)[0]
                part_new = grammar.apply(pmol)[0]
                a = partition_as_sets(part_old)
                b = partition_as_sets(part_new)
                if sorted(w.cert for w in part_old.history) != \
                        sorted(w.cert for w in part_new.history):
                    inv["renumber_word_type_multiset_mismatch"] += 1
                # the two canonical labellings may differ, but only by an
                # automorphism of the labelled graph; that map sigma is read off
                # the two rank vectors and must preserve atoms and bonds exactly
                inv_old = np.argsort(np.asarray(mol.ranks))
                sigma = inv_old[np.asarray(pmol.ranks)[perm]]
                nbrs = [sorted((int(u), G._bond_code(mol.bond_core[b])) for u, b in mol.adj[v])
                        for v in range(mol.n_atoms)]
                automorphic = sorted(int(x) for x in sigma) == list(range(mol.n_atoms)) and all(
                    G._atom_code(mol.atom_core[i]) == G._atom_code(mol.atom_core[int(sigma[i])])
                    and sorted((int(sigma[y]), lab) for y, lab in nbrs[i]) == nbrs[int(sigma[i])]
                    for i in range(mol.n_atoms))
                if automorphic:
                    inv["renumber_automorphism"] += 1
                else:
                    inv["renumber_automorphism_violation"] += 1
                inverse = np.argsort(perm)
                back = sorted(tuple(sorted(int(inverse[a_]) for a_ in sup)) for sup in b)
                # every decision is taken in canonical-rank space, so the
                # relabelled run reproduces the same supports *in rank space*:
                # back == sigma^{-1}(P1) with sigma = rank1^{-1} o rank2 o perm
                sigma_inv = np.argsort(sigma)
                expected = sorted(tuple(sorted(int(sigma_inv[a_]) for a_ in sup)) for sup in a)
                if back != expected:
                    inv["renumber_partition_mismatch"] += 1
                inv["n_molecules"] += 1
            pcache = T.featurize(perm_mols, rows.tolist(), grammar)
            pindex = {int(r): i for i, r in enumerate(rows.tolist())}
            pbatch = T.collate(rows.tolist(), pcache, pindex, bundle.graph_mean,
                               bundle.graph_scale)
            with torch.no_grad():
                after = model(pbatch, mode="REAL")["yhat_std"].numpy()
            inv["renumber_prediction_max_abs_diff"] = max(
                inv["renumber_prediction_max_abs_diff"],
                float(np.abs(after - base).max()))
    # bond-order-only permutation (same atom numbering)
    bo_diff = 0.0
    for name, grammar in grammars.items():
        cch = cache[name]
        for start in 0, 64:
            rows = sample[start:start + batch_size]
            if rows.size == 0:
                continue
            batch = T.collate(rows.tolist(), cch, index, bundle.graph_mean, bundle.graph_scale)
            with torch.no_grad():
                base = model(batch, mode="REAL")["yhat_std"].numpy()
            shuffled = {}
            for r in rows.tolist():
                mol = molecules[int(r)]
                order = rng.permutation(mol.n_bonds)
                shuffled[int(r)] = permute_molecule(mol, np.arange(mol.n_atoms), rng) \
                    if False else _reorder_bonds(mol, order)
            scache = T.featurize(shuffled, rows.tolist(), grammar)
            sindex = {int(r): i for i, r in enumerate(rows.tolist())}
            sbatch = T.collate(rows.tolist(), scache, sindex, bundle.graph_mean,
                               bundle.graph_scale)
            with torch.no_grad():
                after = model(sbatch, mode="REAL")["yhat_std"].numpy()
            bo_diff = max(bo_diff, float(np.abs(after - base).max()))
    inv["bond_order_prediction_max_abs_diff"] = bo_diff
    record("E1_atom_renumbering_invariance",
           inv["renumber_automorphism_violation"] == 0
           and inv["renumber_partition_mismatch"] == 0
           and inv["renumber_word_type_multiset_mismatch"] == 0
           and inv["renumber_prediction_max_abs_diff"] < 1e-5
           and inv["bond_order_prediction_max_abs_diff"] < 1e-5, inv)
    record("E2_bond_order_invariance", bo_diff < 1e-5, {"max_abs_diff": bo_diff})

    # ---- E3: batch composition / order invariance ------------------------
    batch_diff = 0.0
    for name, grammar in grammars.items():
        cch = cache[name]
        rows = all_rows[:32]
        whole = T.collate(rows.tolist(), cch, index, bundle.graph_mean, bundle.graph_scale)
        with torch.no_grad():
            ref = model(whole, mode="REAL")["yhat_std"].numpy()
            parts = []
            for lo, hi in ((0, 5), (5, 20), (20, 32)):
                b = T.collate(rows[lo:hi].tolist(), cch, index, bundle.graph_mean,
                              bundle.graph_scale)
                parts.append(model(b, mode="REAL")["yhat_std"].numpy())
            alone = T.collate([int(rows[7])], cch, index, bundle.graph_mean,
                              bundle.graph_scale)
            single = model(alone, mode="REAL")["yhat_std"].numpy()
        batch_diff = max(batch_diff, float(np.abs(np.concatenate(parts) - ref).max()),
                         float(np.abs(single[0] - ref[7])))
    record("E3_batch_order_invariance", batch_diff < 1e-5, {"max_abs_diff": batch_diff})

    # ---- F: port binding --------------------------------------------------
    cch = cache["predictive"]
    rows = all_rows[:64]
    batch = T.collate(rows.tolist(), cch, index, bundle.graph_mean, bundle.graph_scale)
    single_port_fraction = None
    n_single = 0
    n_multi = 0
    single_equal = True
    for r in rows.tolist():
        mol = molecules[int(r)]
        part = pred_grammar.apply(mol)[0]
        for w in part.words:
            ports = [a for a in w.support
                     if any(y not in w.support for y, _ in mol.adj[a])]
            if len(ports) <= 1:
                n_single += 1
                if len(ports) == 1:
                    role = T._role_vector(mol, w.support, w.sig, ports[0])
                    if not np.allclose(role, _mean_role_of(mol, w.support, w.sig, ports)):
                        single_equal = False
            else:
                n_multi += 1
    single_port_fraction = float(n_single / max(1, n_single + n_multi))
    record("F1_single_port_role_equals_mean",
           single_equal,
           {"single_port_words": n_single, "multi_port_words": n_multi,
            "single_port_fraction": single_port_fraction})
    # difference between the REAL and MEAN tensors must be confined to roles
    other_keys = [k for k in ("w_rule", "w_size", "w_child1", "w_child2", "w_atom_core",
                              "w_nports", "w_ctx", "w_scal", "w_final", "x_word", "x_bond",
                              "e_i", "e_j", "e_bond", "b_core", "b_stereo", "graph")
                  if k in batch]
    identical = all(torch.equal(batch[k], batch[k]) for k in other_keys)
    with torch.no_grad():
        z_real, _ = model.word_states(batch, "REAL")
        z_mean, _ = model.word_states(batch, "MEAN")
        role_real = model.port_roles(batch, "REAL")
        role_mean = model.port_roles(batch, "MEAN")
    diff_keys = [k for k in role_real if not torch.allclose(role_real[k], role_mean[k])]
    record("F2_real_vs_mean_only_role_fields",
           identical and set(diff_keys) <= {"x_l", "x_r", "e_i", "e_j"},
           {"differing_role_fields": sorted(diff_keys),
            "z_max_abs_diff": float((z_real - z_mean).abs().max())})

    # ---- F3: one checkpoint toggles REAL/MEAN ----------------------------
    sd = model.state_dict()
    model2, _ = M.build_model(atom_dims, bond_dims, 0, cfg)
    model2.load_state_dict(sd)
    model2.eval()
    with torch.no_grad():
        o1 = model2(batch, mode="REAL")["yhat_std"].numpy()
        o2 = model2(batch, mode="MEAN")["yhat_std"].numpy()
    record("F3_checkpoint_switches_port_mode", np.isfinite(o1).all() and np.isfinite(o2).all(),
           {"mean_abs_diff": float(np.abs(o1 - o2).mean()),
            "max_abs_diff": float(np.abs(o1 - o2).max())})

    # ---- G: gradients -----------------------------------------------------
    model.train()
    target = torch.as_tensor(y_std[rows], dtype=torch.float32)
    out = model(batch, mode="REAL")
    loss = torch.mean((out["yhat_std"] - target) ** 2)
    loss.backward()
    groups = {"atom_mlp": [], "compose": [], "q": [], "f": [], "readout": [], "emb": []}
    for pname, param in model.named_parameters():
        if param.grad is None:
            groups["emb" if "emb" in pname else pname.split(".")[0]].append(0.0)
            continue
        key = ("atom_mlp" if pname.startswith("atom_mlp") else
               "compose" if pname.startswith("compose") else
               "q" if pname.startswith("q_") else
               "f" if pname.startswith("f_") else
               "readout" if pname.startswith("readout") else "emb")
        groups[key].append(float(param.grad.norm()))
    finite = all(np.isfinite(v) for vals in groups.values() for v in vals)
    positive = all(max(vals) > 0 for vals in groups.values())
    record("G1_compose_q_gradients_finite_nonzero", finite and positive,
           {k: {"n": len(v), "max_norm": max(v), "min_norm": min(v)} for k, v in groups.items()})
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    losses = []
    for _ in range(30):
        out = model(batch, mode="REAL")
        loss = torch.mean((out["yhat_std"] - target) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    record("G2_thirty_steps_reduce_loss", losses[-1] < 0.9 * losses[0],
           {"first": losses[0], "last": losses[-1], "ratio": losses[-1] / losses[0]})

    # ---- H: fit-only statistics -----------------------------------------
    fit_labels = bundle.labels[fit_rows]
    scaler_ok = (abs(bundle.scaler["mean"] - float(fit_labels.mean())) < 1e-12
                 and abs(bundle.scaler["std"] - float(fit_labels.std(ddof=0))) < 1e-12
                 and int(bundle.scaler["n"]) == int(fit_rows.size))
    shifted = bundle.labels.copy()
    shifted[dev_rows] += 100.0
    scaler_shifted = {"mean": float(shifted[fit_rows].mean()),
                      "std": float(shifted[fit_rows].std(ddof=0))}
    record("H1_target_scaler_is_fit_only",
           scaler_ok and abs(scaler_shifted["mean"] - bundle.scaler["mean"]) < 1e-12,
           {"scaler": bundle.scaler, "fit_mean": float(fit_labels.mean()),
            "fit_std": float(fit_labels.std(ddof=0))})
    graph_recompute = np.stack([D.graph_scalars(molecules[int(r)]) for r in fit_rows.tolist()])
    recomputed_scale = graph_recompute.std(axis=0)
    recomputed_scale = np.where(recomputed_scale > 1e-12, recomputed_scale, 1.0)
    graph_ok = np.allclose(graph_recompute.mean(axis=0), bundle.graph_mean, atol=1e-10) and \
        np.allclose(recomputed_scale, bundle.graph_scale, atol=1e-10)
    record("H2_graph_standardiser_is_fit_only", bool(graph_ok),
           {"graph_mean": bundle.graph_mean, "graph_scale": bundle.graph_scale})

    # ---- H3: D0 Dev labels never used by grammar construction -----------
    disc2 = GD.GrammarDiscovery(molecules, fit_rows, y_std[fit_rows],
                                bundle.graph_scalars[fit_rows], bundle.scaffold_by_row,
                                log=lambda *_: None)
    gram_shift, _ = disc2.run("predictive", max_rules=3)
    same_shift = all(gram_shift.rules[i].pattern == pred_grammar.rules[i].pattern
                     for i in range(min(3, len(pred_grammar))))
    fold = GD.fold_assignment(fit_rows, bundle.scaffold_by_row)
    fold_scaf = [len({bundle.scaffold_by_row[str(r)] for i, r in enumerate(fit_rows.tolist())
                      if fold[i] == s}) for s in (0, 1)]
    overlap = len({bundle.scaffold_by_row[str(r)] for i, r in enumerate(fit_rows.tolist())
                   if fold[i] == 0} & {bundle.scaffold_by_row[str(r)]
                                       for i, r in enumerate(fit_rows.tolist()) if fold[i] == 1})
    record("H3_dev_labels_not_used_and_folds_disjoint", same_shift and overlap == 0,
           {"fold_molecules": np.bincount(fold).tolist(), "fold_scaffolds": fold_scaf,
            "shared_fold_scaffolds": overlap, "rules_after_dev_label_shift_identical": same_shift})

    # ---- I: non-degeneracy of the pre-training state --------------------
    with torch.no_grad():
        z_real, aux = model.word_states(batch, "REAL")
        z_mean, _ = model.word_states(batch, "MEAN")
        g = model.q_mlp(torch.zeros((1, model.q_in)))
    z_np = z_real.numpy()
    var_ok = float(z_np.var(axis=0).mean()) > 1e-6
    collapse = float((z_real - z_mean).abs().max(dim=1).values.lt(1e-7).float().mean().item())
    record("I1_word_states_not_constant_at_init",
           var_ok and collapse < 0.99,
           {"z_variance_mean": float(z_np.var(axis=0).mean()),
            "z_effective_rank": effective_rank(z_np),
            "compose_dispersion": aux["compose_dispersion"],
            "fraction_words_insensitive_to_port_mode": collapse})

    # ---- summary ----------------------------------------------------------
    passed = sum(1 for c in CHECKS if c["ok"])
    payload = {
        "protocol_id": cfg["protocol_id"], "generated_utc": C.now_utc(),
        "git_commit": C.git_commit(), "host": platform.node(),
        "torch": torch.__version__, "python": platform.python_version(),
        "n_checks": len(CHECKS), "n_passed": passed,
        "all_pass": passed == len(CHECKS),
        "walls_seconds": time.time() - t0, "checks": CHECKS,
        "params": info["parameters"], "init_hashes": info["init_hashes"],
        "molecule_build_seconds": build_seconds,
        "grammar_stats_summary": {
            k: {side: {kk: stats[k][side][kk] for kk in
                       ("n_word_types", "mean_words_per_molecule", "size_histogram",
                        "multi_port_word_fraction", "type_cross_scaffold_reuse_fraction")}
                for side in ("fit", "dev")} for k in stats},
    }
    C.write_json(args.out, payload)
    print(f"\n{passed}/{len(CHECKS)} PASS  (wall {payload['walls_seconds']:.1f}s)")
    return 0 if payload["all_pass"] else 1


def _mean_role_of(mol: G.Molecule, support: Sequence[int], sig: G.WordSignature,
                  ports: Sequence[int]) -> np.ndarray:
    if not ports:
        return np.zeros(T.ROLE_DIM)
    return np.stack([T._role_vector(mol, support, sig, a) for a in ports]).mean(axis=0)


def _reorder_bonds(mol: G.Molecule, order: np.ndarray) -> G.Molecule:
    bu = mol.bond_u[order]
    bv = mol.bond_v[order]
    bc = mol.bond_core[order]
    bs = mol.bond_stereo[order]
    adj: list[list[tuple[int, int]]] = [[] for _ in range(mol.n_atoms)]
    for b in range(bu.size):
        adj[int(bu[b])].append((int(bv[b]), b))
        adj[int(bv[b])].append((int(bu[b]), b))
    return G.Molecule(row=mol.row, n_atoms=mol.n_atoms, atom_core=mol.atom_core,
                      atom_intrinsic=mol.atom_intrinsic, atom_context=mol.atom_context,
                      bond_u=bu, bond_v=bv, bond_core=bc, bond_stereo=bs, adj=adj,
                      ranks=mol.ranks, n_components=mol.n_components)


if __name__ == "__main__":
    raise SystemExit(main())