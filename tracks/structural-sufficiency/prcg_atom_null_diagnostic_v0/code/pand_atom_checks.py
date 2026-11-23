"""PRCG-ATOM-NULL part B checks: the empty grammar and the ATOM_ONLY arm.

Checks are mechanical and written to ``analysis/atom_checks.json``.  They cover
the five properties named in the protocol (atom coverage, bond coverage,
renumbering invariance, batch invariance, label isolation), plus the explicit
construction claims: no Compose execution, no atom-only-specific network, and
bit-identical width/initialisation to PRCG-v0.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import pand_common as P
import prcg_checks as CK
import prcg_model as M

CHECKS: list[dict[str, Any]] = []


def record(name: str, ok: bool, detail: Mapping[str, Any]) -> None:
    CHECKS.append({"check": name, "ok": bool(ok), "detail": P.C.jsonable(dict(detail))})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: "
          f"{json.dumps(P.C.jsonable(dict(detail)), sort_keys=True)[:280]}", flush=True)


def _partition_checks(molecules: Mapping[int, P.G.Molecule], rows: np.ndarray,
                      grammar: P.G.Grammar) -> dict[str, Any]:
    bad = {"word_count": 0, "not_singleton": 0, "not_atom_rule": 0,
           "coverage": 0, "history_not_atom": 0, "applied_nonempty": 0,
           "intra_word_bond": 0, "same_word_bond": 0}
    n_atoms = 0
    n_words = 0
    n_bonds = 0
    for r in rows.tolist():
        mol = molecules[int(r)]
        part, log = grammar.apply(mol, record_log=True)
        n_atoms += mol.n_atoms
        n_words += len(part.words)
        n_bonds += mol.n_bonds
        if len(part.words) != mol.n_atoms:
            bad["word_count"] += 1
        if any(w.size != 1 for w in part.words):
            bad["not_singleton"] += 1
        if any(w.rule != -1 for w in part.words):
            bad["not_atom_rule"] += 1
        supports = sorted(tuple(w.support) for w in part.words)
        if supports != [(a,) for a in range(mol.n_atoms)]:
            bad["coverage"] += 1
        if any(w.rule != -1 or w.children != (-1, -1) for w in part.history):
            bad["history_not_atom"] += 1
        if list(part.applied):
            bad["applied_nonempty"] += 1
        owner = {int(a): i for i, w in enumerate(part.words) for a in w.support}
        for w in part.words:
            if mol.internal_bonds(w.support):
                bad["intra_word_bond"] += 1
        for b in range(mol.n_bonds):
            u, v = int(mol.bond_u[b]), int(mol.bond_v[b])
            if owner[u] == owner[v]:
                bad["same_word_bond"] += 1
    return {"rows": int(rows.size), "atoms": n_atoms, "words": n_words,
            "bonds": n_bonds, "violations": bad}


def _cache_checks(molecules: Mapping[int, P.G.Molecule], rows: np.ndarray,
                  cache: Mapping[str, np.ndarray]) -> dict[str, Any]:
    bad = {"w_ptr": 0, "e_ptr": 0, "x_ptr": 0, "b_ptr": 0,
           "w_rule": 0, "w_size": 0, "w_child": 0, "w_final": 0,
           "w_atom_core": 0, "w_atom_intr": 0, "w_nports": 0,
           "e_bond_sequence": 0, "e_endpoints": 0, "b_core": 0, "b_stereo": 0,
           "x_words": 0}
    for i, r in enumerate(rows.tolist()):
        mol = molecules[int(r)]
        wlo, whi = int(cache["w_ptr"][i]), int(cache["w_ptr"][i + 1])
        elo, ehi = int(cache["e_ptr"][i]), int(cache["e_ptr"][i + 1])
        xlo, xhi = int(cache["x_ptr"][i]), int(cache["x_ptr"][i + 1])
        blo, bhi = int(cache["b_ptr"][i]), int(cache["b_ptr"][i + 1])
        if whi - wlo != mol.n_atoms:
            bad["w_ptr"] += 1
        if ehi - elo != mol.n_bonds:
            bad["e_ptr"] += 1
        if xhi != xlo:
            bad["x_ptr"] += 1
        if bhi - blo != mol.n_bonds:
            bad["b_ptr"] += 1
        if not np.all(cache["w_rule"][wlo:whi] < 0):
            bad["w_rule"] += 1
        if not np.all(cache["w_size"][wlo:whi] == 1):
            bad["w_size"] += 1
        if not (np.all(cache["w_child1"][wlo:whi] == -1)
                and np.all(cache["w_child2"][wlo:whi] == -1)):
            bad["w_child"] += 1
        if not np.all(cache["w_final"][wlo:whi] == 1):
            bad["w_final"] += 1
        if not np.array_equal(cache["w_atom_core"][wlo:whi], mol.atom_core):
            bad["w_atom_core"] += 1
        if not np.array_equal(cache["w_atom_intr"][wlo:whi], mol.atom_intrinsic):
            bad["w_atom_intr"] += 1
        # for a singleton word, every neighbour is outside the word, so the
        # word has one port iff the atom is not isolated (not one per bond)
        nports = np.asarray([1.0 if mol.adj[a] else 0.0 for a in range(mol.n_atoms)],
                            np.float32)
        if not np.allclose(cache["w_nports"][wlo:whi], nports, atol=0.0):
            bad["w_nports"] += 1
        if not np.array_equal(cache["e_bond"][elo:ehi], np.arange(mol.n_bonds)):
            bad["e_bond_sequence"] += 1
        if not (np.array_equal(cache["e_i"][elo:ehi], mol.bond_u)
                and np.array_equal(cache["e_j"][elo:ehi], mol.bond_v)):
            bad["e_endpoints"] += 1
        if not np.array_equal(cache["b_core"][blo:bhi], mol.bond_core):
            bad["b_core"] += 1
        if not np.array_equal(cache["b_stereo"][blo:bhi], mol.bond_stereo):
            bad["b_stereo"] += 1
        if xhi > xlo:
            bad["x_words"] += 1
    return {"violations": bad,
            "totals": {"words": int(cache["w_rule"].size),
                       "inter_bonds": int(cache["e_i"].size),
                       "crossing_bonds": int(cache["x_word"].size),
                       "bonds": int(cache["b_core"].shape[0])}}


def _predict(model: M.PrcgModel, rows: np.ndarray, cache: Mapping[str, np.ndarray],
             index: Mapping[int, int], bundle: P.D.DataBundle,
             batch_size: int = 256) -> np.ndarray:
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, rows.size, batch_size):
            chunk = rows[start:start + batch_size]
            bt = P.T.collate(chunk.tolist(), cache, index, bundle.graph_mean,
                             bundle.graph_scale)
            out.append(model(bt, mode="REAL")["yhat_std"].cpu().numpy())
    return np.concatenate(out, axis=0)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(P.ANALYSIS_DIR / "atom_checks.json"))
    ap.add_argument("--renumber-molecules", type=int, default=60)
    args = ap.parse_args(argv)
    t0 = time.time()
    bundle = P.D.load_bundle()
    parent = P.parent_config()
    cfg = P.load_config()
    atom_dims, bond_dims = P.D.atom_bond_dims()
    fit_rows = np.asarray(bundle.fit_rows, np.int64)
    dev_rows = np.asarray(bundle.eval_rows, np.int64)
    all_rows = np.concatenate([fit_rows, dev_rows])
    molecules = P.D.build_molecules(bundle.raw, all_rows)
    grammar = P.empty_grammar()
    cache = P.T.featurize(molecules, all_rows.tolist(), grammar)
    index = {int(r): i for i, r in enumerate(all_rows.tolist())}

    # ---- A/B: coverage and real bonds ------------------------------------
    part = _partition_checks(molecules, all_rows, grammar)
    record("A1_atom_coverage_and_singleton_partition",
           all(v == 0 for v in part["violations"].values()), part)
    cache_info = _cache_checks(molecules, all_rows, cache)
    record("B1_bond_and_tensor_coverage",
           all(v == 0 for v in cache_info["violations"].values()), cache_info)
    record("B2_no_crossing_bonds",
           int(cache["x_word"].size) == 0 and int(cache["x_bond"].size) == 0,
           {"x_word": int(cache["x_word"].size), "x_bond": int(cache["x_bond"].size)})

    # ---- C: model construction and no Compose execution ------------------
    pr = P.parent_results("PRED_GRAMMAR")
    model, ledger = M.build_model(atom_dims, bond_dims, int(cfg["new_arm"]["init_seed"]),
                                  parent)
    model_identical = (ledger["parameters"] == pr["parameters"]
                       and ledger["init_hashes"] == pr["init_hashes"])
    record("C1_width_and_init_identical_to_pred_grammar", model_identical,
           {"parameters_equal": ledger["parameters"] == pr["parameters"],
            "init_hashes_equal": ledger["init_hashes"] == pr["init_hashes"],
            "total": ledger["parameters"]["total"]})
    record("C2_compose_is_instantiated",
           int(ledger["parameters"]["compose"]) > 0
           and int(ledger["parameters"]["compose_norm"]) > 0,
           {"compose": ledger["parameters"]["compose"],
            "compose_norm": ledger["parameters"]["compose_norm"]})

    ckpt_path = P.RESULTS_DIR / "ATOM_ONLY_seed0.pt"
    trained = False
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        trained = True
    model.eval()
    rows_eval = dev_rows[:64]
    batch = P.T.collate(rows_eval.tolist(), cache, index, bundle.graph_mean,
                        bundle.graph_scale)
    calls = {"compose": 0, "compose_norm": 0}
    h1 = model.compose.register_forward_hook(
        lambda *a: calls.__setitem__("compose", calls["compose"] + 1))
    h2 = model.compose_norm.register_forward_hook(
        lambda *a: calls.__setitem__("compose_norm", calls["compose_norm"] + 1))
    with torch.no_grad():
        out = model(batch, mode="REAL")
        atom_z = model.atom_states(batch)
    h1.remove()
    h2.remove()
    record("C3_compose_never_executed", calls == {"compose": 0, "compose_norm": 0},
           {"calls": calls, "loaded_trained_checkpoint": trained})
    z_diff = float((out["z"] - atom_z).abs().max())
    record("C4_word_state_is_atom_state",
           z_diff < 1e-7 and out["aux"]["compose_dispersion"] == [],
           {"max_abs_z_diff": z_diff,
            "compose_dispersion": out["aux"]["compose_dispersion"],
            "words": int(out["z"].shape[0])})
    model.zero_grad(set_to_none=True)
    out_grad = model(batch, mode="REAL")
    out_grad["yhat_std"].pow(2).mean().backward()
    compose_grads = [p.grad for p in model.compose.parameters()]
    norm_grads = [p.grad for p in model.compose_norm.parameters()]
    absent = all(g is None for g in compose_grads + norm_grads)
    record("C5_no_compose_gradient_path", absent,
           {"compose_grads_all_none": all(g is None for g in compose_grads),
            "compose_norm_grads_all_none": all(g is None for g in norm_grads)})

    # ---- D: invariances on the trained (or initial) model ----------------
    rng = np.random.default_rng(20261201)
    sample = all_rows[rng.choice(all_rows.size,
                                 size=min(int(args.renumber_molecules), all_rows.size),
                                 replace=False)]
    base = _predict(model, sample, cache, index, bundle)
    perm_mols = {}
    perms = {}
    for r in sample.tolist():
        mol = molecules[int(r)]
        perm = rng.permutation(mol.n_atoms)
        perms[int(r)] = perm
        perm_mols[int(r)] = CK.permute_molecule(mol, perm, rng)
    pcache = P.T.featurize(perm_mols, sample.tolist(), grammar)
    pindex = {int(r): i for i, r in enumerate(sample.tolist())}
    after = _predict(model, sample, pcache, pindex, bundle)
    renum_diff = float(np.abs(after - base).max())
    # the partition is a singleton partition in both numberings; compare the
    # back-mapped supports and the atom-type multiset
    partition_ok = True
    for r in sample.tolist():
        back = sorted(tuple(sorted(int(np.argsort(perms[int(r)])[a]) for a in sup))
                      for sup in [w.support for w in grammar.apply(perm_mols[int(r)])[0].words])
        expected = [(a,) for a in range(molecules[int(r)].n_atoms)]
        if back != expected:
            partition_ok = False
    record("D1_renumbering_invariance",
           renum_diff < 1e-5 and partition_ok,
           {"max_abs_prediction_diff": renum_diff, "partition_backmap_ok": partition_ok,
            "n_molecules": int(sample.size), "model_trained": trained})

    # bond-order-only permutation (same atom numbering)
    bo_mols = {}
    for r in sample.tolist():
        mol = molecules[int(r)]
        order = rng.permutation(mol.n_bonds)
        bo_mols[int(r)] = CK._reorder_bonds(mol, order)
    bocache = P.T.featurize(bo_mols, sample.tolist(), grammar)
    bo_after = _predict(model, sample, bocache, pindex, bundle)
    bo_diff = float(np.abs(bo_after - base).max())
    record("D2_bond_order_invariance", bo_diff < 1e-5,
           {"max_abs_prediction_diff": bo_diff, "n_molecules": int(sample.size)})

    # batch composition and order
    rows = all_rows[:32]
    whole = _predict(model, rows, cache, index, bundle)
    pieces = []
    for lo, hi in ((0, 5), (5, 20), (20, 32)):
        pieces.append(_predict(model, rows[lo:hi], cache, index, bundle))
    single = _predict(model, rows[7:8], cache, index, bundle)
    batch_diff = max(float(np.abs(np.concatenate(pieces) - whole).max()),
                     float(np.abs(single[0] - whole[7])))
    order = rng.permutation(rows.size)
    perm_rows = rows[order]
    shuffled = _predict(model, perm_rows, cache, index, bundle)
    order_diff = float(np.abs(shuffled - whole[order]).max())
    record("D3_batch_composition_invariance", batch_diff < 1e-5,
           {"max_abs_prediction_diff": batch_diff})
    record("D4_batch_order_invariance", order_diff < 1e-5,
           {"max_abs_prediction_diff": order_diff})

    # ---- E: label isolation ----------------------------------------------
    finite = np.flatnonzero(np.isfinite(bundle.labels))
    record("E1_finite_labels_only_on_official_train",
           bool(np.array_equal(np.sort(finite), bundle.official_train)),
           {"finite": int(finite.size), "official_train": int(bundle.official_train.size)})
    record("E2_no_official_valid_label",
           not bool(np.isfinite(bundle.labels[bundle.official_valid]).any()),
           {"finite_valid": int(np.isfinite(bundle.labels[bundle.official_valid]).sum())})
    record("E3_fit_dev_disjoint_and_cover_train",
           np.intersect1d(fit_rows, dev_rows).size == 0
           and np.union1d(fit_rows, dev_rows).size == bundle.official_train.size
           and np.isin(np.union1d(fit_rows, dev_rows), bundle.official_train).all(),
           {"fit": int(fit_rows.size), "dev": int(dev_rows.size),
            "official_train": int(bundle.official_train.size)})
    record("E4_split_hash_is_frozen", bundle.hashes["split"]["sha256"]
           == P.C.SPLIT_F0_SHA256_EXPECTED,
           {"sha256": bundle.hashes["split"]["sha256"]})
    y_fit = bundle.labels[fit_rows]
    scaler_ok = (abs(bundle.scaler["mean"] - float(y_fit.mean())) < 1e-12
                 and abs(bundle.scaler["std"] - float(y_fit.std(ddof=0))) < 1e-12)
    record("E5_target_scaler_is_fit_only", scaler_ok, bundle.scaler)
    g = np.stack([P.D.graph_scalars(molecules[int(r)]) for r in fit_rows.tolist()])
    g_scale = g.std(axis=0)
    g_scale = np.where(g_scale > 1e-12, g_scale, 1.0)
    graph_ok = (np.allclose(g.mean(axis=0), bundle.graph_mean, atol=1e-10)
                and np.allclose(g_scale, bundle.graph_scale, atol=1e-10))
    record("E6_graph_scaler_is_fit_only", bool(graph_ok),
           {"graph_mean": bundle.graph_mean.tolist()})
    record("E7_grammar_is_label_free", len(grammar) == 0,
           {"n_rules": len(grammar), "construction": "empty Grammar()"})

    passed = sum(1 for c in CHECKS if c["ok"])
    payload = {
        "protocol_id": P.PROTOCOL_ID,
        "part": "B_atom_only_checks",
        "generated_utc": P.C.now_utc(),
        "git_commit": P.git_commit(),
        "host": platform.node(), "torch": torch.__version__,
        "python": platform.python_version(),
        "n_checks": len(CHECKS), "n_passed": passed,
        "all_pass": passed == len(CHECKS),
        "wall_seconds": time.time() - t0,
        "trained_checkpoint_loaded": trained,
        "checks": CHECKS,
        "cache_facts": {
            "n_words": int(cache["w_rule"].size),
            "n_inter_word_bonds": int(cache["e_i"].size),
            "n_crossing_bonds": int(cache["x_word"].size),
            "n_bonds": int(cache["b_core"].shape[0]),
            "all_words_are_atoms": bool(np.all(cache["w_rule"] < 0)),
            "all_word_sizes_one": bool(np.all(cache["w_size"] == 1)),
        },
    }
    P.write_json(args.out, payload)
    print(f"\n{passed}/{len(CHECKS)} PASS (wall {payload['wall_seconds']:.1f}s)")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())