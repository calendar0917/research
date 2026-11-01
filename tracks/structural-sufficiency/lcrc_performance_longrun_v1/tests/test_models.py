"""Model-arm invariants: zero-init BASE, gradients, symmetry, shapes, reload."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import cl_common as C
import cl_data as CD
import cl_model as CM
import lcrc_data as PREP_DATA
import mb2_data as D


def _batch(cache, rows):
    return D.to_device(PREP_DATA.collate(cache, rows), torch.device("cpu"))


def _predict(model, batch):
    model.eval()
    with torch.no_grad():
        return model(batch).detach().cpu().numpy()


def permute_single_molecule(cache, g: int, perm: np.ndarray):
    """Return a one-molecule cache with atoms relabelled by ``perm``.

    ``perm[i]`` is the old atom index that becomes new atom ``i``.  Every
    structural tensor (atom features, bonds, S1 blocks, S2 blocks) is permuted
    consistently.
    """
    perm = np.asarray(perm, dtype=np.int64)
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    nbr_ptr = np.asarray(cache["nbr_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    a0, a1 = int(atom_ptr[g]), int(atom_ptr[g + 1])
    b0, b1 = int(bond_ptr[g]), int(bond_ptr[g + 1])
    n = a1 - a0
    if sorted(perm.tolist()) != list(range(n)):
        raise ValueError("perm must be a permutation of the atom indices")
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n, dtype=np.int64)

    atom_feat = np.asarray(cache["atom_feat"])[a0:a1][perm]
    atom_type = np.asarray(cache["atom_type"])[a0:a1][perm]
    atom_scalar = np.asarray(cache["atom_scalar"])[a0:a1][perm]

    old_bonds = np.asarray(cache["bond_index"])[b0:b1].astype(np.int64)
    new_bonds = np.stack([inv[old_bonds[:, 0]], inv[old_bonds[:, 1]]], axis=1) if len(old_bonds) else old_bonds
    if len(new_bonds):
        new_bonds.sort(axis=1)
        order = np.lexsort((new_bonds[:, 1], new_bonds[:, 0]))
        new_bonds = new_bonds[order]
    else:
        order = np.zeros(0, dtype=np.int64)
    bond_feat = np.asarray(cache["bond_feat"])[b0:b1][order]
    bond_type = np.asarray(cache["bond_type"])[b0:b1][order]

    # S1: one block of neighbour codes per atom.
    nbr_counts = nbr_ptr[a0 + 1 : a1 + 1] - nbr_ptr[a0:a1]
    nbr_codes = np.asarray(cache["nbr_code"])
    old_blocks = [
        nbr_codes[int(nbr_ptr[a0 + i]) : int(nbr_ptr[a0 + i]) + int(nbr_counts[i])] for i in range(n)
    ]
    new_blocks = [old_blocks[int(perm[i])] for i in range(n)]
    new_nbr_counts = nbr_counts[perm]
    new_nbr_ptr = np.concatenate([[0], np.cumsum(new_nbr_counts)]).astype(np.int64)
    new_nbr_code = np.concatenate(new_blocks) if new_blocks else np.zeros(0, dtype=np.int32)

    # S2: one block of rooted typed paths per atom.
    path_counts = path_ptr[a0 + 1 : a1 + 1] - path_ptr[a0:a1]
    path_cat = np.asarray(cache["path_cat"])
    path_end63 = np.asarray(cache["path_end63"])
    path_end = np.asarray(cache["path_end"])
    path_closure = np.asarray(cache["path_closure"])
    old_path_blocks = []
    for i in range(n):
        start = int(path_ptr[a0 + i])
        stop = start + int(path_counts[i])
        old_path_blocks.append(
            (
                path_cat[start:stop],
                path_end63[start:stop],
                path_end[start:stop],
                path_closure[start:stop],
            )
        )
    new_path_blocks = [old_path_blocks[int(perm[i])] for i in range(n)]
    new_path_counts = path_counts[perm]
    new_path_ptr = np.concatenate([[0], np.cumsum(new_path_counts)]).astype(np.int64)
    if new_path_blocks:
        new_path_cat = np.concatenate([x[0] for x in new_path_blocks])
        new_path_end63 = np.concatenate([x[1] for x in new_path_blocks])
        new_path_end = np.concatenate([inv[x[2]] for x in new_path_blocks]).astype(np.int64)
        new_path_closure = np.concatenate([x[3] for x in new_path_blocks])
    else:
        new_path_cat = np.zeros(0, dtype=np.int32)
        new_path_end63 = np.zeros(0, dtype=np.int32)
        new_path_end = np.zeros(0, dtype=np.int64)
        new_path_closure = np.zeros(0, dtype=np.int8)

    return {
        "n_molecules": 1,
        "rows": np.asarray([int(np.asarray(cache["rows"])[g])], dtype=np.int64),
        "molecule_ids": [str(cache["molecule_ids"][g])],
        "smiles": [str(cache["smiles"][g])],
        "atom_ptr": np.asarray([0, n], dtype=np.int64),
        "bond_ptr": np.asarray([0, len(new_bonds)], dtype=np.int64),
        "nbr_ptr": new_nbr_ptr,
        "path_ptr": new_path_ptr,
        "atom_feat": np.asarray(atom_feat, dtype=np.float32),
        "atom_type": np.asarray(atom_type, dtype=np.int64),
        "atom_scalar": np.asarray(atom_scalar, dtype=np.float32),
        "bond_feat": np.asarray(bond_feat, dtype=np.float32),
        "bond_type": np.asarray(bond_type, dtype=np.int64),
        "bond_index": np.asarray(new_bonds, dtype=np.int64),
        "nbr_code": np.asarray(new_nbr_code, dtype=np.int32),
        "path_cat": np.asarray(new_path_cat, dtype=np.int32),
        "path_end63": np.asarray(new_path_end63, dtype=np.int32),
        "path_end": np.asarray(new_path_end, dtype=np.int64),
        "path_closure": np.asarray(new_path_closure, dtype=np.int8),
        "global": np.asarray(cache["global"])[g : g + 1],
        "n_components": np.asarray(cache["n_components"])[g : g + 1],
        "n_junctions": np.asarray(cache["n_junctions"])[g : g + 1],
        "n_ring_systems": np.asarray(cache["n_ring_systems"])[g : g + 1],
        "split": "permuted_single",
        "meta": {"labels_present": False, "split": "permuted_single"},
    }


def test_step0_all_arms_matches_base():
    base = CD.base_view("ESOL", "published")
    ext = CD.extend_view(base, "ESOL")
    rows = [0, 1, 2, 3]
    b_batch = _batch(base.valid_cache, rows)
    e_batch = _batch(ext.valid_cache, rows)
    base_model, _ = CM.build_base_model(base.spec, base.train_norm, 0, torch.device("cpu"))
    y_base = _predict(base_model, b_batch)
    assert y_base.shape == (len(rows),)

    for arm in C.ARMS[1:]:
        spec = ext.spec if arm in CM.CHEM8_ARMS else base.spec
        model, meta = CM.build_variant_model(
            spec, base.spec, base.train_norm, 0, torch.device("cpu"), arm=arm
        )
        y = _predict(model, e_batch if arm in CM.CHEM8_ARMS else b_batch)
        assert np.max(np.abs(y - y_base)) <= 1e-6, (arm, np.max(np.abs(y - y_base)))
        assert meta["extension_zero_init"]["passed"] is True


def test_shared_tensors_are_copied_exactly():
    base = CD.base_view("ESOL", "published")
    ext = CD.extend_view(base, "ESOL")
    base_model, _ = CM.build_base_model(base.spec, base.train_norm, 0, torch.device("cpu"))
    model, _ = CM.build_variant_model(
        ext.spec, base.spec, base.train_norm, 0, torch.device("cpu"), arm="chem8"
    )
    b = base_model.state_dict()
    v = model.state_dict()
    for key, value in b.items():
        if key == "atom_encoder.lins.0.weight":
            assert np.array_equal(value.numpy(), v[key].numpy()[:, :51])
            assert float(v[key].numpy()[:, 51:].max()) == 0.0
        else:
            assert np.array_equal(value.numpy(), v[key].numpy()), key
    assert model.atom_encoder.lins[0].in_features == 59


def test_rhoshape_moments_match_naive_and_batch_isolation():
    view = CD.base_view("ESOL", "published")
    model, _ = CM.build_variant_model(
        view.spec, view.spec, view.train_norm, 0, torch.device("cpu"), arm="rhoshape"
    )
    rows = [0, 1, 2, 3]
    batch = _batch(view.valid_cache, rows)
    model.eval()
    with torch.no_grad():
        _y, aux = model(batch, return_aux=True)
    h = aux["h"].numpy()
    atom_batch = batch["atom_batch"].numpy()
    ref = CM.per_graph_stats_naive(h, atom_batch, len(rows))
    assert np.allclose(aux["max_h"].numpy(), ref["max"], atol=1e-6)
    assert np.allclose(aux["mad_h"].numpy(), ref["mad"], atol=1e-6)
    assert np.allclose(aux["mean_h"].numpy(), ref["mean"], atol=1e-6)
    # A per-graph maximum must not leak across graph boundaries.
    for g in range(len(rows)):
        assert np.allclose(aux["max_h"].numpy()[g], h[atom_batch == g].max(0), atol=1e-6)


def test_atom_permutation_invariance():
    base = CD.base_view("ESOL", "published")
    ext = CD.extend_view(base, "ESOL")
    g = 0
    n_atoms = int(np.asarray(base.valid_cache["atom_ptr"])[1] - np.asarray(base.valid_cache["atom_ptr"])[0])
    perm = np.arange(n_atoms, dtype=np.int64)[::-1].copy()
    permuted_base = permute_single_molecule(base.valid_cache, g, perm)
    permuted_ext = permute_single_molecule(ext.valid_cache, g, perm)
    original_batch = _batch(base.valid_cache, [g])
    permuted_batch = _batch(permuted_base, [0])
    original_ext = _batch(ext.valid_cache, [g])
    permuted_ext_batch = _batch(permuted_ext, [0])
    for arm in ("base", "rhoshape", "multicontrib", "chem8", "combo_chem_contrib"):
        spec = ext.spec if arm in CM.CHEM8_ARMS else base.spec
        if arm == "base":
            model, _ = CM.build_base_model(spec, base.train_norm, 0, torch.device("cpu"))
        else:
            model, _ = CM.build_variant_model(
                spec, base.spec, base.train_norm, 0, torch.device("cpu"), arm=arm
            )
        y1 = _predict(model, original_ext if arm in CM.CHEM8_ARMS else original_batch)
        y2 = _predict(model, permuted_ext_batch if arm in CM.CHEM8_ARMS else permuted_batch)
        assert np.max(np.abs(y1 - y2)) <= 1e-6, (arm, y1, y2)


def test_multitask_shapes():
    view = CD.base_view("ESOL", "published")
    rows = [0, 1, 2]
    batch = _batch(view.valid_cache, rows)
    for arm in ("base", "rhoshape", "multicontrib", "combo_rho_contrib"):
        if arm == "base":
            model, _ = CM.build_base_model(view.spec, view.train_norm, 0, torch.device("cpu"), n_tasks=12)
        else:
            model, _ = CM.build_variant_model(
                view.spec, view.spec, view.train_norm, 0, torch.device("cpu"), arm=arm, n_tasks=12
            )
        model.eval()
        with torch.no_grad():
            y, aux = model(batch, return_aux=True)
        assert y.shape == (len(rows), 12)
        if arm in CM.CONTRIB_ARMS:
            assert aux["delta"].shape == (len(rows), 12)
            assert aux["q_graph"].shape == (len(rows), 8)


def test_new_parameters_receive_gradients_and_leave_zero():
    view = CD.base_view("ESOL", "published")
    ext = CD.extend_view(view, "ESOL")
    rows = list(range(64))
    b_batch = _batch(view.train_cache, rows)
    e_batch = _batch(ext.train_cache, rows)
    y = torch.randn(len(rows))
    for arm in ("chem8", "rhoshape", "multicontrib"):
        spec = ext.spec if arm in CM.CHEM8_ARMS else view.spec
        model, _ = CM.build_variant_model(
            spec, view.spec, view.train_norm, 0, torch.device("cpu"), arm=arm
        )
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        batch = e_batch if arm in CM.CHEM8_ARMS else b_batch
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(batch), y)
            loss.backward()
            opt.step()
        if arm in CM.CHEM8_ARMS:
            w = model.atom_encoder.lins[0].weight.detach()
            assert float(w[:, 51:].abs().max()) > 0.0
            assert model.atom_encoder.lins[0].weight.grad is not None
        if arm == "rhoshape":
            w = model.rho.lins[0].weight.detach()
            assert float(w[:, -128:].abs().max()) > 0.0
            assert model.rho.lins[0].weight.grad is not None
        if arm == "multicontrib":
            model.eval()
            with torch.no_grad():
                _y, aux = model(batch, return_aux=True)
            assert float(aux["delta"].abs().max()) > 0.0
            assert float(aux["q_graph"].abs().max()) > 0.0
            grad_q = max(float(p.grad.abs().max()) for n, p in model.named_parameters() if n.startswith("q.") and p.grad is not None)
            assert grad_q > 0.0
            assert model.delta.lins[-1].weight.grad is not None


def test_checkpoint_reload_is_identical():
    view = CD.base_view("ESOL", "published")
    ext = CD.extend_view(view, "ESOL")
    rows = list(range(16))
    batch = _batch(ext.train_cache, rows)
    y = torch.randn(len(rows))
    model, _ = CM.build_variant_model(
        ext.spec, view.spec, view.train_norm, 0, torch.device("cpu"), arm="combo_chem_contrib"
    )
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(batch), y)
        loss.backward()
        opt.step()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    y1 = _predict(model, batch)
    reloaded = CM.load_variant_model(
        ext.spec,
        view.train_norm,
        state,
        torch.device("cpu"),
        arm="combo_chem_contrib",
    )
    y2 = _predict(reloaded, batch)
    assert np.max(np.abs(y1 - y2)) <= 1e-7


def test_scale_m_parameters_are_in_range():
    view = CD.base_view("ESOL", "published")
    ext = CD.extend_view(view, "ESOL")
    base_model, base_meta = CM.build_base_model(
        view.spec, view.train_norm, 0, torch.device("cpu"), cfg=C.SCALE_M_CONFIG
    )
    assert 320_000 <= base_meta["parameters"] <= 450_000
    for arm in ("chem8", "rhoshape", "multicontrib"):
        spec = ext.spec if arm in CM.CHEM8_ARMS else view.spec
        model, meta = CM.build_variant_model(
            spec,
            view.spec,
            view.train_norm,
            0,
            torch.device("cpu"),
            arm=arm,
            cfg=C.SCALE_M_CONFIG,
        )
        assert 320_000 <= meta["parameters"] <= 450_000, (arm, meta["parameters"])
