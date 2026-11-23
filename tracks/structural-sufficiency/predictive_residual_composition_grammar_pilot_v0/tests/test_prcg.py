"""Exactness / invariance / leakage tests for PRCG-v0.

Run from the repo root:

    uv run python tracks/structural-sufficiency/predictive_residual_composition_grammar_pilot_v0/tests/test_prcg.py

Every test is cheap (seconds) and checks a property that the protocol claims.
The heavier end-to-end verification lives in ``code/prcg_checks.py``.
"""

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import prcg_checks as CK       # noqa: E402
import prcg_common as C        # noqa: E402
import prcg_data as D          # noqa: E402
import prcg_discover as PD     # noqa: E402
import prcg_grammar as PG      # noqa: E402
import prcg_graph as G         # noqa: E402
import prcg_tensor as T        # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, info: object = "") -> None:
    print(("[PASS] " if ok else "[FAIL] ") + name + (f": {info}" if info != "" else ""))
    if not ok:
        FAILURES.append(name)


def random_graph(rng: random.Random, n: int) -> tuple[list[int], list[list[tuple[int, int]]]]:
    labels = [rng.randint(0, 2) for _ in range(n)]
    esets: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < 0.55:
                lab = rng.randint(0, 2)
                esets[u].append((v, lab))
                esets[v].append((u, lab))
    return labels, esets


def relabel(labels, esets, perm):
    n = len(labels)
    l2 = [0] * n
    e2: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for i in range(n):
        l2[perm[i]] = labels[i]
    for u in range(n):
        for v, lab in esets[u]:
            e2[perm[u]].append((perm[v], lab))
    return l2, e2


def is_automorphism(labels, esets, sigma) -> bool:
    n = len(labels)
    if sorted(sigma) != list(range(n)):
        return False
    nbrs = [sorted((u, lab) for u, lab in esets[v]) for v in range(n)]
    for i in range(n):
        if labels[i] != labels[sigma[i]]:
            return False
        if sorted((sigma[y], lab) for y, lab in nbrs[i]) != nbrs[sigma[i]]:
            return False
    return True


# ---------------------------------------------------------------------------
def test_canonical_order_is_a_labelling_and_automorphism_equivariant() -> None:
    rng = random.Random(20261201)
    equivariant = 0
    automorphic = 0
    trials = 400
    for _ in range(trials):
        n = rng.randint(3, 8)
        labels, esets = random_graph(rng, n)
        ranks = G.canonical_ranks(labels, esets, n)
        if sorted(ranks) != list(range(n)):
            check("B1_canonical_ranks_are_a_labelling", False, (n, ranks))
            return
        perm = list(range(n))
        rng.shuffle(perm)
        l2, e2 = relabel(labels, esets, perm)
        ranks2 = G.canonical_ranks(l2, e2, n)
        if [ranks2[perm[i]] for i in range(n)] == ranks:
            equivariant += 1
            automorphic += 1
            continue
        inv = np.argsort(np.asarray(ranks))
        sigma = [int(inv[ranks2[perm[i]]]) for i in range(n)]
        if is_automorphism(labels, esets, sigma):
            automorphic += 1
    check("B1_canonical_ranks_are_a_labelling", True, {"trials": trials})
    check("B2_canonical_order_automorphism_equivariant", automorphic == trials,
          {"exact": equivariant, "up_to_automorphism": automorphic, "trials": trials})


def test_canonical_string_is_a_graph_invariant() -> None:
    """The canonical string must be identical for every relabelling of a graph.

    This is the property the grammar relies on: the string is the minimum over
    the reachable individualisation leaves, and that reachable set is itself a
    graph invariant, so relabelling may only permute the atoms *inside* the
    canonical order (an automorphism).
    """

    rng = random.Random(7)
    bad = 0
    trials = 150
    for _ in range(trials):
        n = rng.randint(3, 7)
        labels, esets = random_graph(rng, n)
        nbrs = [sorted((u, lab) for u, lab in esets[v]) for v in range(n)]
        ranks = G.canonical_ranks(labels, esets, n)
        seq = [0] * n
        for v, r in enumerate(ranks):
            seq[r] = v
        base = G._canonical_string(seq, labels, nbrs, n)
        for _ in range(3):
            perm = list(range(n))
            rng.shuffle(perm)
            l2, e2 = relabel(labels, esets, perm)
            nbrs2 = [sorted((u, lab) for u, lab in e2[v]) for v in range(n)]
            ranks2 = G.canonical_ranks(l2, e2, n)
            seq2 = [0] * n
            for v, r in enumerate(ranks2):
                seq2[r] = v
            if G._canonical_string(seq2, l2, nbrs2, n) != base:
                bad += 1
    check("B3_canonical_string_is_a_graph_invariant", bad == 0,
          {"bad": bad, "trials": trials})


def test_word_signature_is_isomorphism_invariant() -> None:
    rng = random.Random(11)
    bad = 0
    trials = 200
    for _ in range(trials):
        # connected graphs only: words are connected induced supports
        while True:
            n = rng.randint(3, 5)
            labels, esets = random_graph(rng, n)
            seen = {0}
            stack = [0]
            while stack:
                x = stack.pop()
                for y, _ in esets[x]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            if len(seen) == n:
                break
        support = list(range(n))
        perm = list(range(n))
        rng.shuffle(perm)
        l2, e2 = relabel(labels, esets, perm)
        a = G.word_signature(_fake_molecule(labels, esets), support)
        b = G.word_signature(_fake_molecule(l2, e2), [perm[i] for i in support])
        if (a.cert, a.orbit_rep, a.orbit_size, a.atom_codes, a.bond_codes) != \
                (b.cert, b.orbit_rep, b.orbit_size, b.atom_codes, b.bond_codes):
            bad += 1
    check("B4_word_signature_and_orbits_invariant", bad == 0, {"bad": bad, "trials": trials})


def _fake_molecule(labels, esets) -> G.Molecule:
    n = len(labels)
    pairs = []
    for u in range(n):
        for v, lab in esets[u]:
            if u < v:
                pairs.append((u, v, lab))
    bond_u = np.asarray([p[0] for p in pairs], np.int64)
    bond_v = np.asarray([p[1] for p in pairs], np.int64)
    bond_core = np.asarray([[p[2], 0] for p in pairs], np.int64).reshape(-1, 2)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for b in range(bond_u.size):
        adj[int(bond_u[b])].append((int(bond_v[b]), b))
        adj[int(bond_v[b])].append((int(bond_u[b]), b))
    atom_core = np.asarray([[labels[i], 0, 0] for i in range(n)], np.int64)
    return G.Molecule(row=-1, n_atoms=n, atom_core=atom_core,
                      atom_intrinsic=np.zeros((n, 4), np.int64),
                      atom_context=np.zeros((n, 5), np.float64),
                      bond_u=bond_u, bond_v=bond_v, bond_core=bond_core,
                      bond_stereo=np.zeros(bond_u.size, np.int64), adj=adj,
                      ranks=G.canonical_ranks(labels, esets, n))


def test_merge_occurrence_order_is_structure_derived() -> None:
    b = D.load_bundle()
    mols = D.build_molecules(b.raw, b.fit_rows[:6])
    grammars = _grammars()
    rng = np.random.default_rng(3)
    bad = 0
    for row, mol in mols.items():
        perm = rng.permutation(mol.n_atoms)
        pmol = CK.permute_molecule(mol, perm, rng)
        for name, grammar in grammars.items():
            p1 = G.initial_partition(mol)
            p2 = G.initial_partition(pmol)
            for rule in grammar.rules:
                o1 = [o.key for o in G.merge_occurrences(mol, p1.words, rule.left_cert,
                                                         rule.right_cert)
                      if o.pattern == rule.pattern]
                o2 = [o.key for o in G.merge_occurrences(pmol, p2.words, rule.left_cert,
                                                         rule.right_cert)
                      if o.pattern == rule.pattern]
                if o1 != o2:
                    bad += 1
                p1, _, _ = G.apply_rule(mol, p1, rule)
                p2, _, _ = G.apply_rule(pmol, p2, rule)
    check("B5_merge_occurrence_keys_and_patterns_invariant", bad == 0, {"bad": bad})


def test_folds_are_scaffold_disjoint_and_fit_only() -> None:
    b = D.load_bundle()
    folds = PG.fold_assignment(b.fit_rows, b.scaffold_by_row, 20261201)
    fold_of = {int(r): int(f) for r, f in zip(b.fit_rows.tolist(), folds.tolist())}
    fit_scaffolds = {b.scaffold_by_row[str(int(r))] for r in b.fit_rows}
    fold_scaffolds = [set() for _ in range(2)]
    fold_sizes = [0, 0]
    for r in b.fit_rows:
        f = fold_of[int(r)]
        fold_sizes[f] += 1
        fold_scaffolds[f].add(b.scaffold_by_row[str(int(r))])
    check("C1_folds_cover_fit_rows_exactly", sum(fold_sizes) == b.fit_rows.size, fold_sizes)
    check("C2_folds_are_scaffold_disjoint", not (fold_scaffolds[0] & fold_scaffolds[1]),
          {"shared": len(fold_scaffolds[0] & fold_scaffolds[1])})
    check("C3_fold_minimum_sizes",
          min(fold_sizes) >= 8 and min(len(fs) for fs in fold_scaffolds) >= 4,
          {"molecules": fold_sizes, "scaffolds": [len(fs) for fs in fold_scaffolds]})
    check("C4_fold_scaffolds_are_fit_only", fold_scaffolds[0] | fold_scaffolds[1] <= fit_scaffolds,
          {"unused_fit_scaffolds": len(fit_scaffolds)})


def test_labels_only_exist_on_official_train() -> None:
    b = D.load_bundle()
    finite = np.flatnonzero(np.isfinite(b.labels))
    check("D1_finite_labels_only_on_official_train",
          bool(np.array_equal(np.sort(finite), b.official_train)),
          {"finite": int(finite.size)})
    check("D2_no_official_valid_label", not bool(np.isfinite(b.labels[b.official_valid]).any()))
    check("D3_fit_and_eval_rows_are_disjoint",
          np.intersect1d(b.fit_rows, b.eval_rows).size == 0,
          {"fit": int(b.fit_rows.size), "eval": int(b.eval_rows.size)})
    # ``sha256`` is the canonical (sorted-key) JSON hash recorded by the frozen
    # split; ``file_sha256`` is the on-disk byte hash of the same file
    check("D4_split_hash_is_the_frozen_one",
          b.hashes["split"]["sha256"] == C.SPLIT_F0_SHA256_EXPECTED and
          len(b.hashes["split"]["file_sha256"]) == 64,
          {"canonical": b.hashes["split"]["sha256"],
           "file": b.hashes["split"]["file_sha256"]})
    check("D5_raw_cache_hash_is_the_frozen_one",
          b.hashes["raw_cache"]["sha256"] == C.RAW_CACHE_SHA256,
          b.hashes["raw_cache"]["sha256"])


def test_graph_and_target_standardisers_are_fit_only() -> None:
    b = D.load_bundle()
    y = b.labels[b.fit_rows]
    check("E1_target_scaler_from_fit_rows",
          abs(b.scaler["mean"] - float(y.mean())) < 1e-9 and
          abs(b.scaler["std"] - float(y.std(ddof=0))) < 1e-9, b.scaler)
    g = np.asarray([D.graph_scalars(D.build_molecules(b.raw, [int(r)])[int(r)])
                    for r in b.fit_rows], np.float64)
    scale = g.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    check("E2_graph_scaler_from_fit_rows",
          np.allclose(g.mean(axis=0), b.graph_mean, atol=1e-10) and
          np.allclose(scale, b.graph_scale, atol=1e-10),
          {"mean": b.graph_mean.tolist()})


def test_grammar_artifacts() -> None:
    grammars = _grammars()
    pred, freq = grammars["predictive"], grammars["frequency"]
    check("F1_equal_final_rule_count", len(pred) == len(freq) == 32, [len(pred), len(freq)])
    check("F2_one_rule_per_parent_type",
          len({r.parent_cert for r in pred.rules}) == len(pred.rules) and
          len({r.parent_cert for r in freq.rules}) == len(freq.rules))
    check("F3_rules_respect_word_bound",
          all(2 <= r.parent_size <= 5 for r in list(pred.rules) + list(freq.rules)))
    check("F4_predictive_rules_were_screened_on_fit_only",
          all(r.origin == "predictive" for r in pred.rules))
    stats = C.read_json(C.GRAMMAR_DIR / "stats.json")
    rule_counts = {k: len(v["rules"] if isinstance(v["rules"], list) else [])
                   for k, v in stats.items() if k in ("predictive", "frequency")}
    check("F5_grammar_stats_present", rule_counts == {"predictive": 32, "frequency": 32},
          rule_counts)
    fit = {k: v["fit"] for k, v in stats.items() if k in ("predictive", "frequency")}
    check("F6_word_statistics_recorded",
          all("mean_words_per_molecule" in v and "n_word_types" in v for v in fit.values()),
          {k: {"words_per_molecule": round(v["mean_words_per_molecule"], 2),
               "types": v["n_word_types"]} for k, v in fit.items()})


def test_real_vs_mean_differs_only_in_port_fields_and_model_is_invariant() -> None:
    import torch

    b = D.load_bundle()
    rows = np.asarray(sorted(b.fit_rows)[:8], np.int64)
    mols = D.build_molecules(b.raw, rows)
    grammar = _grammars()["predictive"]
    cache = T.featurize(mols, rows, grammar)
    index = {int(r): i for i, r in enumerate(rows.tolist())}
    atom_dims, bond_dims = D.atom_bond_dims()
    model, _ = _model(atom_dims, bond_dims)
    batch = T.collate(rows.tolist(), cache, index, b.graph_mean, b.graph_scale)
    with torch.no_grad():
        real = model(batch, mode="REAL")
        mean = model(batch, mode="MEAN")
    diff = (real["z"] - mean["z"]).abs()
    check("G1_real_vs_mean_word_state_differs", float(diff.max()) > 1e-5, float(diff.max()))
    with torch.no_grad():
        y1 = model(batch, mode="REAL")["yhat_std"]
        y2 = model(batch, mode="MEAN")["yhat_std"]
    check("G2_real_vs_mean_prediction_differs", float((y1 - y2).abs().max()) > 1e-6,
          float((y1 - y2).abs().max()))
    # renumbering invariance of the whole pipeline
    rng = np.random.default_rng(0)
    worst = 0.0
    for r in rows.tolist():
        mol = mols[r]
        perm = rng.permutation(mol.n_atoms)
        pmol = CK.permute_molecule(mol, perm, rng)
        single = T.collate([r], cache, index, b.graph_mean, b.graph_scale)
        pcache = T.featurize({r: pmol}, [r], grammar)
        pbatch = T.collate([r], pcache, {r: 0}, b.graph_mean, b.graph_scale)
        with torch.no_grad():
            a = float(model(single, mode="REAL")["yhat_std"][0])
            c = float(model(pbatch, mode="REAL")["yhat_std"][0])
        worst = max(worst, abs(a - c))
    check("G3_renumbering_invariance", worst < 1e-5, {"max_abs_diff": worst})


def test_no_iterative_passing_or_type_lookup() -> None:
    atom_dims, bond_dims = D.atom_bond_dims()
    model, ledger = _model(atom_dims, bond_dims)
    check("H1_parameter_budget", 100_000 <= ledger["parameters"]["total"] <= 300_000,
          ledger["parameters"]["total"])
    check("H2_no_per_type_compose_lookup",
          not any("compose" in n and "rule" in n for n in ledger["parameters"]),
          sorted(ledger["parameters"]))
    # one shared Compose MLP, no per-rule tables, no recurrent modules
    check("H3_single_shared_compose",
          sum(1 for n in ledger["parameters"] if n.startswith("compose")) == 2,
          sorted(ledger["parameters"]))
    check("H4_no_recurrent_modules",
          not any(word in type(m).__name__.lower()
                  for m in model.modules() for word in ("rnn", "gru", "lstm")))


def _grammars() -> dict[str, G.Grammar]:
    return {name: PD.load_grammar(C.GRAMMAR_DIR / f"{name}_grammar.json")
            for name in ("predictive", "frequency")}


def _model(atom_dims, bond_dims):
    import prcg_model as M

    seed = int(C.load_config().get("seed", 0))
    return M.build_model(atom_dims, bond_dims, seed, C.load_config())


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print()
    print(f"{len(FAILURES)} failure(s)")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())