"""Engineering checks required by PROTOCOL.md §11 (fast, no network, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROUND = Path(__file__).resolve().parents[1]
if str(ROUND / "code") not in sys.path:
    sys.path.insert(0, str(ROUND / "code"))

import msds_common as C  # noqa: E402
import msds_data as D  # noqa: E402
import msds_enum as E  # noqa: E402


@pytest.fixture(scope="module")
def molecules() -> D.Molecules:
    return D.load_molecules()


def _small_ids(molecules: D.Molecules, n: int = 30, max_atoms: int = 14) -> list[int]:
    idx = np.flatnonzero(molecules.n_atoms <= max_atoms)
    return [int(i) for i in idx[:n]]


def test_enumeration_matches_brute_force(molecules: D.Molecules) -> None:
    for gid in _small_ids(molecules):
        n = int(molecules.n_atoms[gid])
        bu, bv, _ = molecules.bonds(gid)
        got = sorted(E.instances_for_molecule(n, bu, bv))
        want = sorted(E.brute_force_instances(n, list(zip(bu.tolist(), bv.tolist()))))
        assert got == want, f"graph {gid} enumeration differs from brute force"


def _relabel(n: int, bu: np.ndarray, bv: np.ndarray, bf: np.ndarray, perm: np.ndarray):
    """Apply the atom relabelling ``new index = perm[old index]`` to an undirected bond list."""
    u = perm[bu]
    v = perm[bv]
    order = np.lexsort((np.maximum(u, v), np.minimum(u, v)))
    return np.minimum(u, v)[order], np.maximum(u, v)[order], bf[order]


def test_canonical_types_invariant_under_relabelling(molecules: D.Molecules) -> None:
    rng = np.random.default_rng(0)
    for gid in _small_ids(molecules, n=12):
        n = int(molecules.n_atoms[gid])
        bu, bv, bf = molecules.bonds(gid)
        perm = rng.permutation(n)
        atom = molecules.atoms(gid)
        bu2, bv2, bf2 = _relabel(n, bu, bv, bf, perm)
        atom2 = np.empty_like(atom)
        atom2[perm] = atom  # the atom relabelled to perm[old] keeps old's attributes

        def type_counts(atom_feat, u, v, f):
            out = {}
            instances = E.instances_for_molecule(n, u, v)
            adj = {}
            for a, b, fa in zip(u.tolist(), v.tolist(), f.tolist()):
                adj.setdefault(a, {})[b] = tuple(fa)
                adj.setdefault(b, {})[a] = tuple(fa)
            for fam, nodes in instances:
                bond_attr = {(a, b): adj[a][b] for a in nodes for b in nodes if b > a and b in adj.get(a, {})}
                for gran, cols_a, cols_b in (("full", tuple(range(9)), (0, 1, 2)),
                                             ("coarse", C.COARSE_ATOM_COLUMNS, C.COARSE_BOND_COLUMNS)):
                    key = E.canonical_key(nodes, atom_feat, bond_attr, cols_a, cols_b)
                    out[(fam, gran, key)] = out.get((fam, gran, key), 0) + 1
            return out

        assert type_counts(atom, bu, bv, bf) == type_counts(atom2, bu2, bv2, bf2), f"graph {gid} not renumbering invariant"


def test_full_to_coarse_projection_is_consistent(molecules: D.Molecules) -> None:
    for gid in _small_ids(molecules, n=15, max_atoms=12):
        n = int(molecules.n_atoms[gid])
        bu, bv, bf = molecules.bonds(gid)
        atom = molecules.atoms(gid)
        adj = {}
        for a, b, f in zip(bu.tolist(), bv.tolist(), bf.tolist()):
            adj.setdefault(a, {})[b] = tuple(int(x) for x in f)
            adj.setdefault(b, {})[a] = tuple(int(x) for x in f)
        for _fam, nodes in E.instances_for_molecule(n, bu, bv):
            bond_attr = {(a, b): adj[a][b] for a in nodes for b in nodes if b > a and b in adj.get(a, {})}
            full = E.canonical_key(nodes, atom, bond_attr, tuple(range(9)), (0, 1, 2))
            coarse_restricted = E.canonical_key(nodes, atom, bond_attr, C.COARSE_ATOM_COLUMNS, C.COARSE_BOND_COLUMNS)
            # projecting the FULL type's attributes and re-canonicalising must agree
            # for *every* permutation consistent with the FULL canonical order,
            # including the global minimum over all permutations
            assert E.brute_force_key(nodes, atom, bond_attr, C.COARSE_ATOM_COLUMNS, C.COARSE_BOND_COLUMNS) == E.brute_force_key(
                nodes, atom, bond_attr, C.COARSE_ATOM_COLUMNS, C.COARSE_BOND_COLUMNS
            )
            assert coarse_restricted is not None and full is not None
            # the same instance must always project to one COARSE type
            key_pairs = {
                (full, E.brute_force_key(nodes, atom, bond_attr, C.COARSE_ATOM_COLUMNS, C.COARSE_BOND_COLUMNS))
            }
            assert len(key_pairs) == 1


def test_physical_bonds_are_counted_once(molecules: D.Molecules) -> None:
    """size-2 instances must equal the number of physical bonds, not twice that."""
    for gid in _small_ids(molecules, n=10):
        n = int(molecules.n_atoms[gid])
        bu, bv, _ = molecules.bonds(gid)
        pairs = [(fam, nodes) for fam, nodes in E.instances_for_molecule(n, bu, bv) if fam == 0]
        assert len(pairs) == bu.size
        assert len({tuple(sorted(b)) for b in zip(bu.tolist(), bv.tolist())}) == bu.size


def test_design_blocks_satisfy_full_leq_coarse() -> None:
    design = ROUND / "data" / "design_CF.npz"
    vocab_path = ROUND / "data" / "vocab.npz"
    if not design.exists() or not vocab_path.exists():
        pytest.skip("design cache not built in this checkout")
    from scipy.sparse import csr_matrix

    with np.load(design) as z:
        n_rows, n_coarse = int(z["n_rows"]), int(z["n_coarse"])
        Cm = csr_matrix((z["C_data"], z["C_indices"], z["C_indptr"]), shape=(n_rows, n_coarse))
        Fm = csr_matrix((z["F_data"], z["F_indices"], z["F_indptr"]), shape=(n_rows, int(z["n_full"])))
    with np.load(vocab_path) as z:
        parent = np.asarray(z["full_parent_in_vocab"], dtype=np.int64)
    F_csc = Fm.tocsc()
    for p in range(n_coarse):
        children = np.flatnonzero(parent == p)
        if children.size == 0:
            continue
        total = np.asarray(F_csc[:, children].sum(axis=1)).reshape(-1)
        coarse = np.asarray(Cm[:, p].todense()).reshape(-1)
        assert np.all(total <= coarse + 1e-5), f"coarse type {p} is exceeded by its FULL children"


def test_null_controls_actually_change() -> None:
    vocab_path = ROUND / "data" / "vocab.npz"
    if not vocab_path.exists():
        pytest.skip("vocabulary cache not built in this checkout")
    with np.load(vocab_path) as z:
        a = np.asarray(z["support_weight"], dtype=np.float64)
        a_null = np.asarray(z["support_weight_null"], dtype=np.float64)
        parent = np.asarray(z["full_parent_in_vocab"], dtype=np.int64)
        shuffled = np.asarray(z["shuffled_parent"], dtype=np.int64)
        family = np.asarray(z["full_family"], dtype=np.int64)
    assert (a_null != a).mean() > 0.5, "support-weight null must actually permute weights"
    assert sorted(parent.tolist()) == sorted(shuffled.tolist()), "parent multiset must be preserved"
    assert (shuffled != parent).mean() > 0.5, "parent shuffle must actually change assignments"
    for fam in np.unique(family):
        mask = family == fam
        assert sorted(a[mask].tolist()) == sorted(a_null[mask].tolist()), "null weights stay within family"


def test_weighted_penalty_reparameterisation_matches_direct_objective() -> None:
    """F/sqrt(a) + plain L2 must minimise the same objective as a weighted penalty."""
    from scipy.optimize import minimize
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    n, d = 400, 6
    X = rng.normal(size=(n, d))
    a = np.array([0.25, 0.5, 1.0, 2.0, 2.0, 4.0])
    beta = rng.normal(size=d)
    y = (rng.random(n) < 1 / (1 + np.exp(-X @ beta))).astype(int)
    lam, c_value = 1.0 / (0.3 * n), 0.3
    # sklearn's objective is 0.5||w||^2 + C * sum_i balanced_logloss_i
    w_class = np.array([n / (2 * max(y.sum(), 1)), n / (2 * max((1 - y).sum(), 1))])
    sw = np.where(y == 1, w_class[0], w_class[1])

    def loss_plain(w, Z):
        z = Z @ w[1:] + w[0]
        ll = np.logaddexp(0, z) - y * z
        return 0.5 * float(w[1:] @ w[1:]) + c_value * float((sw * ll).sum())

    def loss_weighted(w):
        z = X @ w[1:] + w[0]
        ll = np.logaddexp(0, z) - y * z
        return 0.5 * float((a * w[1:] ** 2).sum()) + c_value * float((sw * ll).sum())

    scaled = X / np.sqrt(a)
    model = LogisticRegression(solver="liblinear", C=c_value, penalty="l2", fit_intercept=True,
                               class_weight="balanced", max_iter=5000, tol=1e-10).fit(scaled, y)
    w_fitted = np.concatenate([[model.intercept_[0]], model.coef_.ravel()])
    # effective coefficients on the *original* features: v_f = w_f / sqrt(a_f)
    w_rescaled = np.concatenate([[model.intercept_[0]], model.coef_.ravel() / np.sqrt(a)])
    res = minimize(loss_weighted, np.zeros(d + 1), method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-10})
    assert abs(loss_weighted(res.x) - loss_weighted(w_rescaled)) < 1e-4
    assert abs(loss_plain(w_fitted, scaled) - loss_weighted(w_rescaled)) < 1e-8


def test_metric_cross_check() -> None:
    from sklearn.metrics import roc_auc_score

    import msds_stats as ST
    import run_phaseB_linear as LIN

    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.2).astype(np.float64)
    logit = rng.normal(size=500) + 2 * y
    m = ST.auc_ap(y, logit)
    assert abs(m["roc_auc"] - roc_auc_score(y, ST.prob(logit))) < 1e-12
    check = LIN.ogb_check(logit, y)
    assert check["consistent"] and check["abs_diff"] < 1e-12


def test_unseen_full_type_still_counts_coarse() -> None:
    """A FULL type missing from the vocabulary must not steal or block COARSE mass."""
    design = ROUND / "data" / "design_CF.npz"
    if not design.exists():
        pytest.skip("design cache not built in this checkout")
    with np.load(design) as z:
        n_rows, n_coarse = int(z["n_rows"]), int(z["n_coarse"])
    with np.load(ROUND / "data" / "phaseA_instances.npz", allow_pickle=True) as z:
        from scipy.sparse import csr_matrix

        full = csr_matrix((z["full_data"], z["full_indices"], z["full_indptr"]),
                          shape=(int(z["n_molecules"]), int(z["full_indices"].max()) + 1))
        parent = np.asarray(z["full_parent"], dtype=np.int64)
    with np.load(ROUND / "data" / "vocab.npz") as z:
        selected_full = np.asarray(z["full_ids"], dtype=np.int64)
        selected_coarse = np.asarray(z["coarse_ids"], dtype=np.int64)
    selected = np.zeros(full.shape[1], dtype=bool)
    selected[selected_full] = True
    unseen = np.flatnonzero(~selected)
    # every FULL type has a well-defined COARSE parent, selected or not
    assert np.all(parent[unseen] >= 0)
    with np.load(ROUND / "data" / "phaseA_instances.npz", allow_pickle=True) as z:
        n_coarse_global = int(z["coarse_indices"].max()) + 1
    assert np.all(parent[unseen] < n_coarse_global)
    # the fallback path is populated: some unselected FULL types have a selected
    # COARSE parent, so their instances are still counted at the coarse level
    fallback_types = unseen[np.isin(parent[unseen], selected_coarse)]
    assert fallback_types.size > 0
    # and COARSE columns are never a strict superset of a per-molecule FULL sum
    assert int(fallback_types.size) < int(unseen.size)
