"""Cheap exactness tests for PRCG-ATOM-NULL.

Run from the repo root:

    uv run python tracks/structural-sufficiency/prcg_atom_null_diagnostic_v0/tests/test_pand.py

The full data/grammar audit lives in the ``pand_*`` scripts; these tests only
pin the algebra and the empty-grammar construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import pand_common as P          # noqa: E402
import pand_ridge_audit as A    # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, info: object = "") -> None:
    print(("[PASS] " if ok else "[FAIL] ") + name + (f": {info}" if info != "" else ""))
    if not ok:
        FAILURES.append(name)


def test_strict_augmented_ridge_matches_schur_formula() -> None:
    rng = np.random.default_rng(20261201)
    n, p, m = 240, 11, 9
    x = np.concatenate([np.ones((n, 1)), rng.normal(size=(n, p - 1))], axis=1)
    y = rng.normal(size=n)
    fold = (rng.random(n) < 0.5).astype(np.int8)
    c = rng.poisson(0.35, size=(n, m)).astype(np.float64)
    # a few deliberate extreme count rows, the regime where the denominator gap matters
    c[rng.choice(n, size=3, replace=False), 0] = 8.0
    direct, formula, detail = A.strict_scores(x, y, fold, c, lam=5.0)
    check("R1_direct_equals_augmented_schur_formula",
          float(np.abs(direct - formula).max()) < 1e-10,
          float(np.abs(direct - formula).max()))
    check("R2_gap_is_lambda_norm_S_squared",
          float(np.abs(detail["gap"] - detail["gap_identity"]).max()) < 1e-10,
          float(np.abs(detail["gap"] - detail["gap_identity"]).max()))
    # the strict denominator is exactly old denominator + gap
    check("R3_strict_denominator_is_old_plus_gap",
          np.allclose(detail["den_strict"], detail["den_old"] + detail["gap"], atol=1e-10),
          float(np.abs(detail["den_strict"] - detail["den_old"]
                       - detail["gap"]).max()))
    # the parent RoundScreener is the old-denominator formula
    old = P.GD.RoundScreener(x[:, 1:], y, fold, c).score()
    # reconstruct the old formula from the returned diagnostics and compare
    # pointwise using the same folds/design
    old_formula = np.zeros_like(old)
    for side in (0, 1):
        tr = fold != side
        te = fold == side
        xd = x
        gram = xd[tr].T @ xd[tr] + np.diag(A._penalty(x.shape[1], 5.0))
        base = xd[te] @ np.linalg.solve(gram, xd[tr].T @ y[tr])
        base_mse = float(np.mean((y[te] - base) ** 2))
        s = np.linalg.solve(gram, xd[tr].T @ c[tr])
        rtr = c[tr] - xd[tr] @ s
        rte = c[te] - xd[te] @ s
        beta = (rtr.T @ y[tr]) / (np.einsum("ij,ij->j", rtr, rtr) + 5.0)
        old_formula[side] = base_mse - np.mean(
            (y[te][:, None] - (base[:, None] + rte * beta[None, :])) ** 2, axis=0)
    check("R4_parent_screener_is_old_denominator",
          float(np.abs(old - old_formula).max()) < 1e-10,
          float(np.abs(old - old_formula).max()))
    check("R5_gap_nonnegative_up_to_roundoff",
          float(detail["gap"].min()) > -1e-9, float(detail["gap"].min()))


def test_empty_grammar_on_real_molecules() -> None:
    b = P.D.load_bundle()
    rows = np.concatenate([b.fit_rows[:8], b.eval_rows[:4]])
    mols = P.D.build_molecules(b.raw, rows)
    grammar = P.empty_grammar()
    cache = P.T.featurize(mols, rows.tolist(), grammar)
    check("E1_empty_grammar_has_no_rules", len(grammar) == 0, len(grammar))
    check("E2_every_word_is_an_atom",
          bool(np.all(cache["w_rule"] < 0) and np.all(cache["w_size"] == 1)
               and np.all(cache["w_final"] == 1)),
          {"n_words": int(cache["w_rule"].size)})
    check("E3_no_crossing_bonds_and_all_bonds_inter",
          int(cache["x_word"].size) == 0
          and int(cache["e_i"].size) == int(cache["b_core"].shape[0]),
          {"x": int(cache["x_word"].size), "e": int(cache["e_i"].size),
           "b": int(cache["b_core"].shape[0])})
    ok = True
    for r in rows.tolist():
        mol = mols[int(r)]
        part = grammar.apply(mol)[0]
        if sorted(tuple(w.support) for w in part.words) != [(a,) for a in range(mol.n_atoms)]:
            ok = False
        if any(mol.internal_bonds(w.support) for w in part.words):
            ok = False
    check("E4_exact_singleton_coverage_and_no_internal_bonds", ok)


def test_compose_is_never_called_for_atom_only() -> None:
    b = P.D.load_bundle()
    rows = b.fit_rows[:32]
    mols = P.D.build_molecules(b.raw, rows)
    grammar = P.empty_grammar()
    cache = P.T.featurize(mols, rows.tolist(), grammar)
    index = {int(r): i for i, r in enumerate(rows.tolist())}
    cfg = P.parent_config()
    atom_dims, bond_dims = P.D.atom_bond_dims()
    model, _ = P.M.build_model(atom_dims, bond_dims, 0, cfg)
    model.eval()
    batch = P.T.collate(rows.tolist(), cache, index, b.graph_mean, b.graph_scale)
    calls = {"n": 0}
    handle = model.compose.register_forward_hook(lambda *a: calls.__setitem__("n", calls["n"] + 1))
    with torch.no_grad():
        out = model(batch, mode="REAL")
        z_atom = model.atom_states(batch)
    handle.remove()
    check("C1_compose_not_called", calls["n"] == 0, calls)
    check("C2_word_state_equals_atom_state",
          float((out["z"] - z_atom).abs().max()) == 0.0,
          float((out["z"] - z_atom).abs().max()))


def test_empty_grammar_renumbering_invariance() -> None:
    b = P.D.load_bundle()
    rows = b.fit_rows[:6]
    mols = P.D.build_molecules(b.raw, rows)
    grammar = P.empty_grammar()
    cache = P.T.featurize(mols, rows.tolist(), grammar)
    index = {int(r): i for i, r in enumerate(rows.tolist())}
    cfg = P.parent_config()
    atom_dims, bond_dims = P.D.atom_bond_dims()
    model, _ = P.M.build_model(atom_dims, bond_dims, 0, cfg)
    model.eval()
    rng = np.random.default_rng(1)
    with torch.no_grad():
        base = model(P.T.collate(rows.tolist(), cache, index, b.graph_mean,
                                 b.graph_scale), mode="REAL")["yhat_std"].numpy()
    perm_mols = {}
    for r in rows.tolist():
        mol = mols[int(r)]
        perm = rng.permutation(mol.n_atoms)
        perm_mols[int(r)] = P.CK.permute_molecule(mol, perm, rng)
    pcache = P.T.featurize(perm_mols, rows.tolist(), grammar)
    pindex = {int(r): i for i, r in enumerate(rows.tolist())}
    with torch.no_grad():
        after = model(P.T.collate(rows.tolist(), pcache, pindex, b.graph_mean,
                                  b.graph_scale), mode="REAL")["yhat_std"].numpy()
    diff = float(np.abs(after - base).max())
    check("I1_renumbering_invariance", diff < 1e-5, diff)


def test_bootstrap_indices_are_paired_and_deterministic() -> None:
    a = P.bootstrap_indices(504)
    b = P.bootstrap_indices(504)
    check("B1_bootstrap_shape", a.shape == (10000, 504), a.shape)
    check("B2_bootstrap_deterministic", np.array_equal(a, b))


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print()
    print(f"{len(FAILURES)} failure(s)")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())