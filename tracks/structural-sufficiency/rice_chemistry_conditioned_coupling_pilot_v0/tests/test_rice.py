"""Fast CPU unit tests for RICE-CC-v0 (no training, no GPU, D0 Fit/Dev only)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

ATOM_DIMS = [119, 5, 12, 12, 10, 6, 6, 2, 2]
BOND_DIMS = [5, 6, 2]


def make_mol(n_atoms, edges, atom_fields=None, bond_fields=None, row=0):
    if atom_fields is None:
        atom_fields = [[6, 0, 0, 0, 0, 0, 0, 0, 0]] * n_atoms
    if bond_fields is None:
        bond_fields = [[0, 0, 0]] * len(edges)
    return {
        "row": int(row), "n_atoms": int(n_atoms), "n_bonds": len(edges),
        "atom_feat": np.asarray(atom_fields, np.int64).reshape(n_atoms, 9),
        "bond_feat": np.asarray(bond_fields, np.int64).reshape(len(edges), 3),
        "bond_u": np.asarray([e[0] for e in edges], np.int64),
        "bond_v": np.asarray([e[1] for e in edges], np.int64),
        "graph_scalars": np.zeros(8),
    }


def all_graphs(maxn=4):
    out = []
    for n in range(1, maxn + 1):
        pairs = list(itertools.combinations(range(n), 2))
        for mask in range(1 << len(pairs)):
            out.append((n, [p for i, p in enumerate(pairs) if (mask >> i) & 1]))
    return out


def run_engine(mol, u, j, dtype=torch.float64, arm="CC_NULL"):
    import rice_checks as CK
    return CK.vectorised_phi_kappa(mol, u, j, arm=arm, dtype=dtype)


def test_inversion_identity_all_small_graphs():
    import rice_energy as E

    rng = np.random.default_rng(12)
    checked = 0
    for n, edges in all_graphs(4):
        if n == 1:
            continue
        mol = make_mol(n, edges)
        u = rng.normal(size=(n, 3, 2))
        j = rng.normal(size=(len(edges), 3, 3))
        bond_list = [(int(a), int(b), i) for i, (a, b) in enumerate(edges)]
        for arm in ("CC_NULL", "CC_REAL"):
            entry = run_engine(mol, u, j, arm=arm)
            j_used = entry["j_used"].numpy()
            for root in range(n):
                radius = _radius(n, edges, root)
                ref = E.brute_force_kappa(u, j_used, n, bond_list, root, radius)
                for size in (2, 3, 4):
                    atoms = entry["batch"][f"atoms_{size}"]
                    for i in range(int(atoms.shape[0])):
                        slots = tuple(int(x) for x in atoms[i] if int(x) >= 0)
                        if slots[0] != root:
                            continue
                        key = frozenset(slots)
                        total = np.zeros_like(ref["phi"][key])
                        for sub in range(1, len(key) + 1):
                            for combo in itertools.combinations(sorted(key - {root}),
                                                                 sub - 1):
                                total = total + ref["kappa"][frozenset((root,) + combo)]
                        assert np.abs(ref["phi"][key] - total).max() < 1e-10
                        assert np.abs(entry["kappa"][size][i].numpy().reshape(3, 2)
                                      - ref["kappa"][key]).max() < 1e-10
                        checked += 1
    assert checked > 1000


def _radius(n, edges, root):
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    dist = {root: 0}
    frontier = [root]
    while frontier:
        nxt = []
        for x in frontier:
            for y in adj[x]:
                if y not in dist:
                    dist[y] = dist[x] + 1
                    nxt.append(y)
        frontier = nxt
    return sorted(k for k, v in dist.items() if v <= 2)


def test_kappa_zero_for_disconnected_and_zero_pairwise():
    import rice_energy as E

    rng = np.random.default_rng(3)
    for n, edges in all_graphs(4):
        if n < 2:
            continue
        mol = make_mol(n, edges)
        u = rng.normal(size=(n, 3, 2))
        bond_list = [(int(a), int(b), i) for i, (a, b) in enumerate(edges)]
        j = rng.normal(size=(len(edges), 3, 3))
        for arm in ("CC_NULL", "CC_REAL"):
            entry = run_engine(mol, u, j, arm=arm)
            for root in range(n):
                ref = E.brute_force_kappa(u, entry["j_used"].numpy(), n, bond_list,
                                          root, _radius(n, edges, root))
                for key, kappa in ref["kappa"].items():
                    if len(key) > 1 and not E.reference_is_connected(sorted(key),
                                                                     bond_list):
                        assert np.abs(kappa).max() < 1e-11
            zero = run_engine(mol, u, np.zeros_like(j), arm=arm)
            for size in (2, 3, 4):
                if int(zero["kappa"][size].numel()):
                    assert float(zero["kappa"][size].abs().max()) < 1e-11
            if arm == "CC_REAL" and zero["delta"].numel():
                assert float(zero["delta"].abs().max()) == 0.0


def test_triangle_reads_closing_bond_and_clique_all_slots():
    import rice_structure as S

    tri = S.build_molecule(make_mol(3, [(0, 1), (1, 2), (0, 2)]))
    assert int((tri.bonds[3][0] >= 0).sum()) == 3
    clique = S.build_molecule(make_mol(4, list(itertools.combinations(range(4), 2))))
    assert int((clique.bonds[4][0] >= 0).sum()) == 6
    chain = S.build_molecule(make_mol(3, [(0, 1), (1, 2)]))
    assert int((chain.bonds[3][0] >= 0).sum()) == 2


def test_support_enumeration_radius_and_root_slot():
    import rice_structure as S

    path = S.build_molecule(make_mol(4, [(0, 1), (1, 2), (2, 3)]))
    roots4 = path.atoms[4][:, 0].tolist()
    assert roots4 == [1, 2], roots4
    for size in (2, 3, 4):
        assert int(path.atoms[size].shape[0]) == path.atoms[size].shape[0]
        assert (path.atoms[size][:, 0] >= 0).all()


def test_canonical_certificate_is_relabelling_invariant():
    import rice_structure as S

    rng = np.random.default_rng(5)
    mol = make_mol(6, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5)],
                   atom_fields=[[6, 0, 0, 0, 0, 0, 0, 0, 0], [8, 0, 0, 0, 0, 0, 0, 0, 0],
                                [7, 0, 0, 0, 0, 0, 0, 0, 0], [6, 0, 0, 0, 0, 0, 0, 1, 1],
                                [16, 0, 0, 0, 0, 0, 0, 0, 5], [6, 0, 0, 0, 0, 0, 0, 0, 0]],
                   bond_fields=[[0, 1, 1], [1, 1, 1], [0, 1, 0], [0, 1, 0], [0, 0, 0],
                                [0, 0, 0]])
    perm = rng.permutation(6)
    inverse = np.argsort(perm)
    relabelled = dict(mol)
    relabelled["atom_feat"] = mol["atom_feat"][inverse]
    relabelled["bond_u"] = perm[mol["bond_u"]]
    relabelled["bond_v"] = perm[mol["bond_v"]]

    def cert(m):
        st = S.build_molecule(m)
        return {size: np.array(sorted(map(tuple, S.canonical_signature(
            st, size, ATOM_DIMS, BOND_DIMS).tolist()))) for size in (2, 3, 4)}

    a, b = cert(mol), cert(relabelled)
    for size in (2, 3, 4):
        assert a[size].shape == b[size].shape
        assert np.array_equal(a[size], b[size])


def test_engine_invariances_atom_renumbering_and_bond_order():
    import rice_checks as CK
    import rice_structure as S

    mol = make_mol(
        5, [(0, 1), (1, 2), (2, 3), (3, 0), (2, 4)],
        atom_fields=[[6, 0, 0, 0, 0, 0, 0, 0, 0], [8, 0, 0, 0, 0, 0, 0, 0, 0],
                     [7, 0, 0, 0, 0, 0, 0, 0, 0], [6, 0, 0, 0, 0, 0, 0, 1, 1],
                     [6, 0, 0, 0, 0, 0, 0, 0, 0]],
        bond_fields=[[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1], [0, 1, 1]])
    u, j = CK.field_potentials(mol)
    for arm in ("CC_NULL", "CC_REAL"):
        base = {k: v.numpy() for k, v in run_engine(mol, u, j, arm=arm)["kappa"].items()}

        order = np.array([3, 0, 4, 2, 1])
        shuffled = dict(mol)
        shuffled["bond_u"] = mol["bond_u"][order]
        shuffled["bond_v"] = mol["bond_v"][order]
        shuffled["bond_feat"] = mol["bond_feat"][order]
        out = run_engine(shuffled, u, j[order], arm=arm)["kappa"]
        for k in (2, 3, 4):
            assert np.abs(out[k].numpy() - base[k]).max() < 1e-12

        rng = np.random.default_rng(9)
        perm = rng.permutation(5)
        inverse = np.argsort(perm)
        rel = dict(mol)
        rel["atom_feat"] = mol["atom_feat"][inverse]
        rel["bond_u"] = perm[mol["bond_u"]]
        rel["bond_v"] = perm[mol["bond_v"]]
        st_a = S.build_molecule(mol)
        st_b = S.build_molecule(rel)
        entry_a = run_engine(mol, u, j, arm=arm)
        entry_b = run_engine(rel, u[inverse], j, arm=arm)
        for size in (2, 3, 4):
            sig_a = S.canonical_signature(st_a, size, ATOM_DIMS, BOND_DIMS)
            sig_b = S.canonical_signature(st_b, size, ATOM_DIMS, BOND_DIMS)
            if not int(sig_a.shape[0]):
                continue
            oa = np.lexsort(sig_a.T[::-1])
            ob = np.lexsort(sig_b.T[::-1])
            assert np.array_equal(sig_a[oa], sig_b[ob])
            assert np.abs(entry_a["kappa"][size].numpy()[oa]
                          - entry_b["kappa"][size].numpy()[ob]).max() < 1e-10


def test_potentials_bounded_and_conditioned_bound():
    import rice_energy as E
    import rice_model as M

    cfg = {"dims": {"channels": 4}}
    model, _ = M.build_model(ATOM_DIMS, BOND_DIMS, 0,
                             {"graph_mean": np.zeros(8), "graph_scale": np.ones(8)},
                             config=cfg)
    atoms = torch.zeros(5, 9, dtype=torch.long)
    bonds = torch.zeros(4, 3, dtype=torch.long)
    bond_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=torch.long)
    with torch.no_grad():
        torch.nn.init.normal_(model.atom_mlp[-2].weight, 0.0, 50.0)
        torch.nn.init.normal_(model.bond_mlp[-2].weight, 0.0, 50.0)
        u, j = model.potentials({"atom_feat": atoms, "bond_feat": bonds})
        jr, delta = E.couple_pairs(u, j, bond_index)
    assert float(u.abs().max()) <= model.potential_bound + 1e-6
    assert float(j.abs().max()) <= model.potential_bound + 1e-6
    assert float(jr.abs().max()) <= 4.0 + 1e-6
    assert float(delta.abs().max()) <= 2.0 + 1e-6
    # endpoint symmetry
    jr2, _ = E.couple_pairs(u, j, bond_index[:, [1, 0]])
    assert float((jr - jr2).abs().max()) < 1e-12


def test_coupling_formula_and_state_swap_covariance():
    import rice_energy as E

    rng = np.random.default_rng(23)
    u = rng.normal(size=(6, 3, 2))
    j = rng.normal(size=(5, 3, 3))
    idx = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], np.int64)
    h = np.tanh((u[..., 1] - u[..., 0]) / 2.0)
    k = (j[..., 0] + j[..., 2] - 2.0 * j[..., 1]) / 4.0
    d = k * h[idx[:, 0]] * h[idx[:, 1]]
    jr_ref = np.stack([j[..., 0] + d, j[..., 1] - d, j[..., 2] + d], axis=-1)
    jr, delta = E.couple_pairs(torch.as_tensor(u), torch.as_tensor(j),
                               torch.as_tensor(idx))
    assert np.abs(jr.numpy() - jr_ref).max() < 1e-14
    assert np.abs(delta.numpy() - d).max() < 1e-14

    u_swap = torch.as_tensor(u).flip(-1)
    _, delta_swap = E.couple_pairs(u_swap, torch.as_tensor(j), torch.as_tensor(idx))
    assert float((delta_swap - delta).abs().max()) < 1e-12
    j_swap = torch.as_tensor(np.stack([j[..., 2], j[..., 1], j[..., 0]], axis=-1))
    jr_swap, _ = E.couple_pairs(u_swap, j_swap, torch.as_tensor(idx))
    jr_relabel = torch.stack([jr[..., 2], jr[..., 1], jr[..., 0]], dim=-1)
    assert float((jr_swap - jr_relabel).abs().max()) < 1e-12


def test_cc_real_delta_path_has_gradients():
    import rice_model as M

    cfg = {"dims": {"channels": 4, "atom_emb": 8, "bond_emb": 6,
                    "atom_hidden": 16, "bond_hidden": 16, "h_dim": 8,
                    "sentence_hidden": 8, "readout_hidden": 8, "readout_mid": 8}}
    model, _ = M.build_model(ATOM_DIMS, BOND_DIMS, 0,
                             {"graph_mean": np.zeros(8), "graph_scale": np.ones(8)},
                             config=cfg)
    atoms = torch.tensor([[6, 0, 0, 0, 0, 0, 0, 0, 0],
                          [8, 0, 0, 0, 0, 0, 0, 0, 0],
                          [6, 0, 0, 0, 0, 0, 0, 0, 0],
                          [7, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
    bonds = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.long)
    u, j = model.potentials({"atom_feat": atoms, "bond_feat": bonds})
    jr, delta = model.pair_potential(j, u, torch.tensor([[0, 1], [1, 2], [2, 3]]),
                                     "CC_REAL")
    gu, gj = torch.autograd.grad(delta.sum(), [u, j], allow_unused=False)
    assert torch.isfinite(gu).all() and torch.isfinite(gj).all()
    assert float(gu.norm()) > 0 and float(gj.norm()) > 0
    assert torch.isfinite(jr).all()


def test_conditioning_uses_endpoint_atom_types():
    import rice_checks as CK
    import rice_energy as E
    import rice_structure as S
    import rice_tensor as T

    c = [6, 0, 0, 0, 0, 0, 0, 0, 0]
    n_atom = [7, 0, 0, 0, 1, 0, 0, 0, 0]
    o_atom = [8, 0, 0, 0, 0, 0, 0, 0, 0]
    mol = make_mol(4, [(0, 1), (0, 2), (0, 3)], [c, n_atom, o_atom, c],
                   [[0, 0, 0]] * 3)
    u, j = CK.field_potentials(mol)
    st = S.build_molecule(mol)
    batch = T.collate([0], {0: st}, None, "cpu")
    jr, delta = E.couple_pairs(torch.as_tensor(u), torch.as_tensor(j),
                               batch["bond_index"])
    assert float((torch.as_tensor(j)[0] - torch.as_tensor(j)[1]).abs().max()) < 1e-12
    assert float((delta[0] - delta[1]).abs().max()) > 1e-3
    assert float((jr[0] - jr[1]).abs().max()) > 1e-3


def test_data_layer_is_d0_and_label_safe():
    import rice_data as D

    bundle = D.load_bundle()
    assert bundle.fit_rows.size == 2856
    assert bundle.eval_rows.size == 504
    assert np.intersect1d(bundle.fit_rows, bundle.eval_rows).size == 0
    assert np.union1d(bundle.fit_rows, bundle.eval_rows).size == 3360
    assert np.isnan(bundle.labels[bundle.official_valid]).all()
    assert np.isfinite(bundle.labels[bundle.official_train]).all()
    assert abs(bundle.scaler["mean"] - 2.185815826311436) < 1e-12
    assert abs(bundle.scaler["std"] - 1.2139664740467941) < 1e-12
    mutated = bundle.labels.copy()
    mutated[bundle.eval_rows] += 100.0
    y = mutated[bundle.fit_rows]
    assert abs(y.mean() - bundle.scaler["mean"]) < 1e-12


def test_two_arms_share_every_parameter():
    import rice_model as M

    norm = {"graph_mean": np.zeros(8), "graph_scale": np.ones(8)}
    a, info_a = M.build_model(ATOM_DIMS, BOND_DIMS, 0, norm)
    b, info_b = M.build_model(ATOM_DIMS, BOND_DIMS, 0, norm)
    assert info_a["parameters"] == info_b["parameters"]
    assert info_a["init_hashes"] == info_b["init_hashes"]
    assert set(a.state_dict()) == set(b.state_dict())
    for key in a.state_dict():
        assert torch.equal(a.state_dict()[key], b.state_dict()[key])


@pytest.mark.parametrize("size", [2, 3, 4])
def test_subset_indices_are_connected_root_containing(size):
    import rice_energy as E
    import rice_structure as S

    mol = make_mol(6, [(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (4, 5), (5, 2)])
    st = S.build_molecule(mol)
    bond_list = [(int(a), int(b), i) for i, (a, b) in enumerate(
        zip(mol["bond_u"], mol["bond_v"]))]
    atoms = st.atoms[size]
    for i in range(int(atoms.shape[0])):
        root = int(atoms[i, 0])
        support = [int(x) for x in atoms[i] if int(x) >= 0]
        assert root in support
        assert E.reference_is_connected(support, bond_list)
        if size == 3:
            for j in range(st.sub2[3].shape[1]):
                idx = int(st.sub2[3][i, j])
                if idx < 0:
                    continue
                sub = [int(x) for x in st.atoms[2][idx] if int(x) >= 0]
                assert int(st.atoms[2][idx, 0]) == root
                assert set(sub) <= set(support)
                assert E.reference_is_connected(sub, bond_list)
        if size == 4:
            for j in range(st.sub3[4].shape[1]):
                idx = int(st.sub3[4][i, j])
                if idx < 0:
                    continue
                sub = [int(x) for x in st.atoms[3][idx] if int(x) >= 0]
                assert int(st.atoms[3][idx, 0]) == root
                assert set(sub) <= set(support)
                assert E.reference_is_connected(sub, bond_list)