"""Narrow CPU checks for the two arms (A_joint_BF, B_object_tokens).

Not an audit project: the checks are limited to the items that would silently
invalidate a run --

A: initial state == SSL-start P3 (not supervised P3B); zero-gated epoch-0
   forward; all parameters trainable; phi gradient opens after one step.
B: one token per atom; object/member/multiplicity correctness against an
   independent subset enumeration; chordless-cycle definition and typed
   rotation/reversal alignment; atom-renumbering invariance of content, object
   counts, relations and prediction; batch-regrouping / cross-graph isolation;
   empty branch and empty cycle groups; a single shared raw-embedding table;
   parameter count; gradients through every live module.

Everything is label-free.  Synthetic graphs are used for the hand proofs; the
real cached objects are only read for structure.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
_BF = _SS / "zinc_branching_fragment_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_BF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T  # noqa: E402
import bf_data as BF  # noqa: E402
import ot_data as O  # noqa: E402
import ot_joint as J  # noqa: E402
import ot_models as M  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"
RESULTS: dict[str, dict] = {}


def run(name):
    def deco(fn):
        fn.check_name = name

        def wrapped():
            try:
                RESULTS[name] = {"pass": True, "detail": fn()}
            except Exception as exc:  # noqa: BLE001
                RESULTS[name] = {
                    "pass": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                }
        wrapped.check_name = name
        return wrapped
    return deco


# ---------------------------------------------------------------------------
# synthetic graphs / caches
# ---------------------------------------------------------------------------


def fake_obj(categories, edges, bond_types=None):
    cats = np.asarray(categories, dtype=np.int64)
    n = len(cats)
    feat = np.zeros((n, 108), dtype=np.float32)
    feat[np.arange(n), cats] = 1.0
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if bond_types is None:
        bond_types = np.ones((len(edges),), dtype=np.int64)
    bond_types = np.asarray(bond_types, dtype=np.int64)
    bfeat = np.zeros((len(edges), 7), dtype=np.float32)
    bfeat[:, 4] = (bond_types == 4).astype(np.float32)
    bfeat[:, :4] = np.eye(4, dtype=np.float32)[np.clip(bond_types, 0, 3)]
    return SimpleNamespace(
        n_atoms=n,
        atom_feat=feat,
        bond_index=edges,
        bond_type=bond_types,
        bond_feat=bfeat,
        junc_feat=np.zeros((0, 32), dtype=np.float32),
        junc_index=np.zeros((0, 3), dtype=np.int64),
        junc_bond=np.zeros((0, 2), dtype=np.int64),
        global_feat=np.zeros((33,), dtype=np.float32),
    )


def ad_cache_from_objs(objs, split):
    """Minimal ad-style cache (the fields the B collate/model read)."""
    atom_ptr = [0]
    atom_feat, global_feat = [], []
    for obj in objs:
        atom_feat.append(np.asarray(obj.atom_feat, dtype=np.float32))
        global_feat.append(np.asarray(obj.global_feat, dtype=np.float32).reshape(1, 33))
        atom_ptr.append(atom_ptr[-1] + int(obj.n_atoms))
    return {
        "split": split,
        "n_molecules": len(objs),
        "atom_ptr": np.asarray(atom_ptr, dtype=np.int32),
        "atom_feat": np.concatenate(atom_feat, axis=0),
        "global_feat": np.concatenate(global_feat, axis=0),
        "labels": np.zeros((len(objs),), dtype=np.float32),
    }


def register_objects(objs, split, source=None):
    ot = O.build_from_objects(objs, split, source or {"source": "synthetic"}, verbose=False)
    O.register_cache(split, ot)
    return ad_cache_from_objs(objs, split), ot


def norm_stub():
    return {"global_mean": np.zeros((33,), np.float32),
            "global_scale": np.ones((33,), np.float32)}


def build_b_model(ad, ot, device="cpu", seed=0):
    mean = {
        "atom": ot["meta"]["mean_atoms_per_molecule"],
        "branch": ot["meta"]["mean_branch_tokens_per_molecule"],
        "cycle": ot["meta"]["mean_cycle_tokens_per_molecule"],
    }
    return M.build_model(norm_stub(), group_mean=mean, device=device, seed=seed)


def predict_b(model, ad, split, rows, device="cpu"):
    model.eval()
    with torch.no_grad():
        batch = O.collate_ot(ad, np.asarray(rows), valid=True)
        return model(T._to_device(batch, device)).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# independent enumerations
# ---------------------------------------------------------------------------


def brute_cycle_sets(obj, max_len=8):
    """Independent definition: a chordless cycle = induced subgraph that is a
    single cycle (all induced degrees 2, connected).  Returns {L: {nodes}}."""
    n = int(obj.n_atoms)
    bmap = BF.bond_map(obj)
    out = {L: set() for L in range(3, max_len + 1)}
    for L in range(3, max_len + 1):
        for subset in itertools.combinations(range(n), L):
            ss = set(subset)
            deg = {v: 0 for v in subset}
            for v in subset:
                for w in ss:
                    if v < w and (v, w) in bmap:
                        deg[v] += 1
                        deg[w] += 1
            if any(d != 2 for d in deg.values()):
                continue
            # connectivity of the induced subgraph
            seen = {subset[0]}
            frontier = [subset[0]]
            while frontier:
                u = frontier.pop()
                for w in ss:
                    if w not in seen and (u, w) in bmap:
                        seen.add(w)
                        frontier.append(w)
            if len(seen) == L:
                out[L].add(tuple(sorted(subset)))
    return out


def brute_canonical(order, cats, bmap):
    """Independent canonical content: all Hamiltonian cycles on the node set.

    Enumerates every permutation (fixing the smallest node first to remove
    rotations) and both directions, builds the typed atom/bond sequences with
    ``bond[i] = bond(seq[i], seq[(i+1) % L])`` and takes the lexicographic
    minimum.  This is deliberately a different algorithm from
    ``ot_data.canonical_cycle`` (subset permutations instead of rotations of a
    found order).
    """
    nodes = sorted(order)
    L = len(nodes)
    best = None
    first = nodes[0]
    for perm in itertools.permutations(nodes[1:]):
        cycle = (first,) + perm
        for direction in (0, 1):
            seq = list(cycle) if direction == 0 else [cycle[0]] + list(reversed(cycle[1:]))
            bonds = tuple(int(bmap.get((seq[i], seq[(i + 1) % L]), 0)) for i in range(L))
            if any(b == 0 for b in bonds):
                continue  # not a real cycle order
            atoms = tuple(int(cats[v]) for v in seq)
            key = atoms + bonds
            if best is None or key < best:
                best = key
    return best[:L], best[L:]


def canonical_realised_in_graph(atoms, bonds, cats, bmap, nodes):
    """Is the canonical (atoms, bonds) pair realised with aligned bonds?"""
    L = len(atoms)
    assert len(nodes) == L
    for seq in itertools.permutations(nodes, L):
        if tuple(int(cats[v]) for v in seq) != tuple(atoms):
            continue
        got = tuple(int(bmap.get((seq[i], seq[(i + 1) % L]), 0)) for i in range(L))
        if got == tuple(bonds):
            return True
    return False


# ---------------------------------------------------------------------------
# A checks
# ---------------------------------------------------------------------------


@run("a_init_parity_ssl_not_supervised")
def check_a_init():
    norm = D.load_norm()
    model, info = J.build_model_a(norm, seed=0, device="cpu")
    ref = J.historical_p3_ssl_init()
    mine = J.p3_core_state(model)
    diffs = {}
    for key, value in ref.items():
        if key not in mine:
            diffs[key] = "missing"
            continue
        d = float((mine[key].float() - value.float()).abs().max())
        if d > 0.0:
            diffs[key] = d
    assert not diffs, diffs
    # the supervised P3B checkpoint must not be the initial state
    p3b_path = _SS / "zinc_recipe_trunk_v2" / "runs" / "xBssl_P3N_s0" / "best.pt"
    p3b = torch.load(p3b_path, map_location="cpu", weights_only=False)["state_dict"]
    sup_diff = {k: float((mine[k].float() - v.float()).abs().max())
                for k, v in p3b.items() if k in mine}
    assert sup_diff["reader.lins.0.weight"] > 0.1, sup_diff
    assert sup_diff["group_composer.lins.0.weight"] > 0.1, sup_diff
    assert model.w_s.weight.abs().sum().item() == 0.0
    assert info["parameters"] == 447_505
    assert all(p.requires_grad for p in model.parameters())
    assert model.branch.phi.lins[0].weight.requires_grad
    return {
        "compared_p3_tensors": int(len(ref)),
        "n_differing": 0,
        "bitwise_identical_to_ssl_start_p3": True,
        "supervised_p3b_max_abs_diff": float(max(sup_diff.values())),
        "w_s_zero": True,
        "parameters": info["parameters"],
        "ssl_source": info["ssl_encoders"]["source"],
        "ssl_sha256": info["ssl_encoders"]["sha256"],
        "p3b_used": False,
        "old_bf_used": False,
    }


@run("a_zero_gate_and_gradient_opening")
def check_a_gradients():
    norm = D.load_norm()
    valid = D.load_cache("valid")
    rows = np.arange(8, dtype=np.int64)
    model, _ = J.build_model_a(norm, seed=0, device="cpu")
    import bf_data as BFD
    batch = BFD.collate_bf(valid, rows)
    y = torch.as_tensor(np.asarray(valid["labels"])[rows], dtype=torch.float32)
    # epoch-0 forward equals the same-init SSL-start P3 forward
    import ad2_models as A2
    p3 = A2.P3NullModel(norm)
    p3.load_bjoc_state(J.historical_p3_ssl_init())
    model.eval(); p3.eval()
    with torch.no_grad():
        y_a = model(batch)
        y_p3 = p3(batch)
    assert float((y_a - y_p3).abs().max()) == 0.0
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    assert len(opt.param_groups) == 1
    assert C.optimizer_parameter_count(opt) == 447_505
    watch = {"w_s": model.w_s.weight, "phi": model.branch.phi.lins[0].weight,
             "atom": model.encoders.atom_encoder.lins[0].weight,
             "composer": model.group_composer.lins[0].weight}
    records = []
    model.train()
    for step in range(1, 4):
        opt.zero_grad(set_to_none=True)
        loss = F.l1_loss(model(batch), y)
        loss.backward()
        records.append({k: float(p.grad.abs().max()) for k, p in watch.items()})
        opt.step()
    assert records[0]["phi"] == 0.0
    assert records[1]["phi"] > 0.0
    return {
        "epoch0_forward_equals_ssl_start_p3": True,
        "max_abs_diff": 0.0,
        "optimizer_parameters": C.optimizer_parameter_count(opt),
        "param_groups": 1,
        "grad_max_by_step": records,
        "phi_first_nonzero_step": 2,
    }


# ---------------------------------------------------------------------------
# B checks
# ---------------------------------------------------------------------------


@run("b_token_coverage_vs_bf")
def check_b_coverage():
    out = {}
    for split in ("train", "valid", "test"):
        path = O.cache_path(split)
        if not path.exists():
            continue
        pc = O.load_cache(split)
        meta = pc["meta"]
        n_atoms = int(np.asarray(pc["n_atoms_per_mol"]).sum())
        tok_type = np.asarray(pc["tok_type"])
        assert meta["n_atoms"] == n_atoms
        assert int((tok_type == O.TYPE_A).sum()) == n_atoms, "atom tokens != atoms"
        assert int((tok_type == O.TYPE_C).sum()) == int(meta["n_cycle_tokens"])
        assert int(((tok_type >= 1) & (tok_type <= 3)).sum()) == int(meta["n_branch_tokens"])
        assert n_atoms + int(meta["n_branch_tokens"]) + int(meta["n_cycle_tokens"]) == int(
            meta["n_tokens"])
        bf_path = _BF.parent / "cache" / f"bf_{split}.pt"
        bf_occ = None
        obj_level_match = None
        if bf_path.exists():
            bfc = torch.load(bf_path, map_location="cpu", weights_only=False)
            bf_occ = bfc["meta"]["occurrences_by_family"]
            assert bf_occ == meta["branch_by_family"], (bf_occ, meta["branch_by_family"])
            # exact per-(molecule, family, root) token count == BF occurrence count
            n_mol = int(meta["n_molecules"])
            max_atoms = int(meta["max_atoms_per_molecule"]) + 1
            nf = len(O.FAMILIES)
            ent_ptr = np.asarray(bfc["ent_ptr"])
            ent_pat = None
            mol_bf = np.repeat(np.arange(n_mol, dtype=np.int64), np.diff(ent_ptr))
            flat_bf = (mol_bf * nf + np.asarray(bfc["ent_family"])) * max_atoms \
                + np.asarray(bfc["ent_root"])
            want = np.bincount(flat_bf, weights=np.asarray(bfc["ent_count"], dtype=np.float64),
                               minlength=n_mol * nf * max_atoms)
            tptr = np.asarray(pc["tok_ptr"])
            tik = np.concatenate([np.arange(int(a), int(b)) for a, b in zip(tptr[:-1], tptr[1:])])
            tmol = np.repeat(np.arange(n_mol, dtype=np.int64), np.diff(tptr))
            ttype = np.asarray(pc["tok_type"])[tik]
            br = (ttype >= 1) & (ttype <= 3)
            flat_ot = (tmol[br] * nf + np.asarray(pc["tok_family"])[tik][br]) * max_atoms \
                + np.asarray(pc["tok_root"])[tik][br]
            got = np.bincount(flat_ot, minlength=n_mol * nf * max_atoms)
            obj_level_match = bool(np.array_equal(got.astype(np.float64), want))
            assert obj_level_match, "per-(molecule, family, root) counts differ from BF"
        # (family, root, node set) exists exactly once, and identical content on
        # different node sets survives as separate tokens (true multiplicity)
        members = np.asarray(pc["tok_members"])
        content = np.asarray(pc["tok_content"])
        nmem = np.asarray(pc["tok_nmembers"])
        family = np.asarray(pc["tok_family"])
        root_arr = np.asarray(pc["tok_root"])
        ptr = np.asarray(pc["tok_ptr"])
        multi = 0
        for m in range(int(meta["n_molecules"])):
            s, e = int(ptr[m]), int(ptr[m + 1])
            seen = set()
            by_content = defaultdict(set)
            for t in range(s, e):
                if int(tok_type[t]) == O.TYPE_A:
                    continue
                mem = tuple(int(x) for x in members[t][: int(nmem[t])])
                key = (int(family[t]), int(root_arr[t]), mem)
                assert key not in seen, "duplicate (family, root, node set) token"
                seen.add(key)
                ckey = (int(tok_type[t]), int(content[t]))
                by_content[ckey].add(mem)
            multi += sum(len(v) - 1 for v in by_content.values() if len(v) > 1)
        # every member set is distinct per content, so multiplicity is real
        assert multi > 0
        out[split] = {
            "n_atoms": n_atoms,
            "n_tokens": int(meta["n_tokens"]),
            "n_branch_tokens": int(meta["n_branch_tokens"]),
            "n_cycle_tokens": int(meta["n_cycle_tokens"]),
            "branch_by_family": meta["branch_by_family"],
            "bf_occurrences_match": bf_occ == meta["branch_by_family"] if bf_occ else None,
            "multiplicity_tokens": int(multi),
            "per_molecule_family_root_counts_match_bf": obj_level_match,
        }
    return out


@run("b_enumeration_hand_graphs")
def check_b_hand():
    graphs = {
        "star3": fake_obj([0, 1, 1, 2], [(0, 1), (0, 2), (0, 3)]),
        "star4": fake_obj([0, 1, 1, 1, 1], [(0, 1), (0, 2), (0, 3), (0, 4)]),
        "fork": fake_obj([0, 1, 2, 3, 3], [(0, 1), (0, 2), (1, 3), (1, 4)]),
        "path": fake_obj([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)]),
        "ring6": fake_obj([0] * 6, [(i, (i + 1) % 6) for i in range(6)]),
        "ring6_chord_0_3": fake_obj([0] * 6, [(i, (i + 1) % 6) for i in range(6)] + [(0, 3)]),
        "k4": fake_obj([0] * 4, [(i, j) for i in range(4) for j in range(i + 1, 4)]),
        "ring5": fake_obj([0] * 5, [(i, (i + 1) % 5) for i in range(5)]),
    }
    out = {}
    for name, obj in graphs.items():
        branches = O.enumerate_branching(obj)
        cycles = O.enumerate_cycles(obj)
        brute = brute_cycle_sets(obj)
        ours = {L: {tuple(nodes) for (LL, nodes, _, _) in cycles if LL == L}
                for L in range(3, 9)}
        assert ours == brute, (name, ours, brute)
        bmap = BF.bond_map(obj)
        cats = BF.atom_categories(obj)
        for L, nodes, atoms, bonds in cycles:
            assert len(nodes) == L == len(atoms) == len(bonds)
            assert canonical_realised_in_graph(atoms, bonds, cats, bmap, nodes), (name, atoms, bonds)
            a2, b2 = brute_canonical(nodes, cats, bmap)
            assert (a2, b2) == (tuple(atoms), tuple(bonds)), (name, a2, b2, atoms, bonds)
        # branch counts on the hand graphs (root -> family counts)
        per_root = defaultdict(lambda: {"S3": 0, "S4": 0, "F4": 0})
        for fi, root, members, key in branches:
            per_root[root][O.FAMILIES[fi]] += 1
        out[name] = {
            "n_atoms": int(obj.n_atoms),
            "branch_total": len(branches),
            "cycles": {str(L): len(brute[L]) for L in range(3, 9)},
            "cycle_nodes": sorted(nodes for (_, nodes, _, _) in cycles),
            "per_root": {str(r): dict(v) for r, v in sorted(per_root.items())},
        }
        if name == "star3":
            assert len(branches) == 4  # S3=1 at root 0, F4=3
        if name == "star4":
            assert sum(1 for b in branches if b[0] == 1) == 1
        if name in ("path", "ring6", "ring5"):
            assert not branches
            assert len(cycles) == (0 if name == "path" else 1)
        if name == "ring6_chord_0_3":
            assert len(cycles) == 2 and all(L == 4 for L, _, _, _ in cycles)
        if name == "k4":
            assert len(cycles) == 4 and all(L == 3 for L, _, _, _ in cycles)
    return out


@run("b_ring_canonical_alignment")
def check_b_ring_alignment():
    # asymmetric ring: distinct atom categories and all three bond types
    n = 6
    cats = [0, 3, 1, 5, 2, 4]
    edges = [(i, (i + 1) % n) for i in range(n)]
    bt = [1, 2, 3, 1, 2, 3]
    obj = fake_obj(cats, edges, bt)
    (L, nodes, atoms, bonds), = O.enumerate_cycles(obj)
    assert L == 6 and list(atoms) == cats and list(bonds) == bt, (atoms, bonds)
    bmap = BF.bond_map(obj)
    cats_arr = BF.atom_categories(obj)
    assert canonical_realised_in_graph(atoms, bonds, cats_arr, bmap, nodes)
    # same physical ring with rotated labels and with reversed labels
    variants = {}
    for shift in (1, 3):
        edges_r = [((i + shift) % n, (i + 1 + shift) % n) for i in range(n)]
        # shift the atom categories with the labels
        cats_r = [0] * n
        for i in range(n):
            cats_r[(i + shift) % n] = cats[i]
        obj_r = fake_obj(cats_r, edges_r, bt)
        (Lr, nodes_r, atoms_r, bonds_r), = O.enumerate_cycles(obj_r)
        variants[f"rot{shift}"] = {"atoms": list(atoms_r), "bonds": list(bonds_r)}
        assert (atoms_r, bonds_r) == (atoms, bonds), (shift, atoms_r, bonds_r)
    for direction in (0, 1):
        rev = list(range(n)) if direction == 0 else [0] + list(range(n - 1, 0, -1))
        cats_v = [0] * n
        for i in range(n):
            cats_v[rev[i]] = cats[i]
        edges_v = [(rev[i], rev[(i + 1) % n]) for i in range(n)]
        bt_v = [bt[i] for i in range(n)]
        obj_v = fake_obj(cats_v, edges_v, bt_v)
        (Lv, nodes_v, atoms_v, bonds_v), = O.enumerate_cycles(obj_v)
        variants[f"rev{direction}"] = {"atoms": list(atoms_v), "bonds": list(bonds_v)}
        assert (atoms_v, bonds_v) == (atoms, bonds), (direction, atoms_v, bonds_v)
    # negative control: pairing a reversed atom sequence with the forward bond
    # array is only "realised" if the reverse walk really has those bonds
    misaligned = (tuple(reversed(atoms)), tuple(bonds))
    aligned_realised = canonical_realised_in_graph(
        misaligned[0], misaligned[1], cats_arr, bmap, nodes)
    return {
        "cycle_length": L,
        "canonical_atoms": list(atoms),
        "canonical_bonds": list(bonds),
        "rotated_and_reversed_graphs_give_identical_content": variants,
        "misaligned_reverse_atoms_with_forward_bonds_realised": bool(aligned_realised),
        "note": ("the canonical key is realised with bond[i] between seq[i] and "
                 "seq[i+1]; reversing the atom array without reversing the bond "
                 "array is never produced as the canonical content"),
    }


@run("b_permutation_invariance")
def check_b_permutation():
    import ad_common as Cc
    Cc.ensure_p3_modules()
    import structural_relation_objects_v1 as sro

    out = {}
    # real molecule
    ds = D.pyg_zinc("valid")
    d = ds[0]
    x = np.asarray(d.x).reshape(-1).astype(np.int64)
    ei = np.asarray(d.edge_index)
    ea = np.asarray(d.edge_attr).reshape(-1).astype(np.int64)
    obj = sro.build_mol_objects(0, x, ei, ea)
    rng = np.random.default_rng(11)
    perm = rng.permutation(len(x))
    old_to_new = np.empty((len(x),), dtype=np.int64)
    old_to_new[perm] = np.arange(len(x))
    obj_p = sro.build_mol_objects(0, x[perm], old_to_new[ei], ea.copy())
    ad_a, ot_a = register_objects([obj], "__perm_a__")
    ad_b, ot_b = register_objects([obj_p], "__perm_b__")
    model, info = build_b_model(ad_a, ot_a)
    ya = predict_b(model, ad_a, "__perm_a__", [0])
    yb = predict_b(model, ad_b, "__perm_b__", [0])
    same = {
        "atoms": int(ot_a["meta"]["n_atoms"]) == int(ot_b["meta"]["n_atoms"]),
        "tokens": int(ot_a["meta"]["n_tokens"]) == int(ot_b["meta"]["n_tokens"]),
        "branch": ot_a["meta"]["branch_by_family"] == ot_b["meta"]["branch_by_family"],
        "cycles": ot_a["meta"]["cycle_by_length"] == ot_b["meta"]["cycle_by_length"],
    }
    assert all(same.values()), (same, ot_a["meta"], ot_b["meta"])
    d_pred = float(abs(ya[0] - yb[0]))
    assert d_pred < 1e-5, d_pred
    out["real_molecule"] = {"n_atoms": int(len(x)), "pred_abs_diff": d_pred,
                            "counts_equal": same}
    # synthetic asymmetric ring (different atom + bond types), relabelled
    n = 6
    cats0 = [0, 3, 1, 5, 2, 4]
    bonds0 = [1, 2, 3, 1, 2, 3]
    shift = 2
    cats_r = [0] * n
    for i in range(n):
        cats_r[(i + shift) % n] = cats0[i]
    obj2 = fake_obj(cats0, [(i, (i + 1) % n) for i in range(n)], bonds0)
    # the same physical bonds, relabelled: endpoint shift and bond types follow
    obj2p = fake_obj(cats_r,
                     [((i + shift) % n, (i + 1 + shift) % n) for i in range(n)],
                     bonds0)
    ad_a, ot_a = register_objects([obj2], "__perm_c__")
    ad_b, ot_b = register_objects([obj2p], "__perm_d__")
    model2, _ = build_b_model(ad_a, ot_a)
    ya = predict_b(model2, ad_a, "__perm_c__", [0])
    yb = predict_b(model2, ad_b, "__perm_d__", [0])
    d2 = float(abs(ya[0] - yb[0]))
    assert d2 < 1e-5, d2
    out["asymmetric_ring"] = {"pred_abs_diff": d2,
                              "cycle_tokens": int(ot_b["meta"]["n_cycle_tokens"])}
    return out


@run("b_batch_regrouping_and_isolation")
def check_b_batch_isolation():
    valid = D.load_cache("valid")
    pc = O.load_cache("valid")
    det = np.asarray(pc["n_cycle_per_mol"]) > 0
    rows = np.nonzero(det)[0][:4]
    assert len(rows) == 4
    mean = {
        "atom": pc["meta"]["mean_atoms_per_molecule"],
        "branch": pc["meta"]["mean_branch_tokens_per_molecule"],
        "cycle": pc["meta"]["mean_cycle_tokens_per_molecule"],
    }
    model, _ = M.build_model(D.load_norm(), group_mean=mean, device="cpu", seed=0)
    rows = np.asarray(rows, dtype=np.int64)
    single = [predict_b(model, valid, "valid", [int(r)]) for r in rows]
    pair = predict_b(model, valid, "valid", rows)
    diff_batch = float(np.abs(pair - np.concatenate(single)).max())
    assert diff_batch < 1e-5, diff_batch
    # an independent token content/relations change must not move other graphs
    other = np.array([int(r) for r in rows], dtype=np.int64)
    batch = O.collate_ot(valid, other)
    batch2 = O.collate_ot(valid, np.append(other, 7))
    model.eval()
    with torch.no_grad():
        y1 = model(batch)
        y2 = model(batch2)
    d_iso = float((y1 - y2[: len(other)]).abs().max())
    assert d_iso < 1e-5, d_iso
    # the valid-key mask is block diagonal in (graph, local token) layout, so
    # no cross-graph token pair can enter any attention row
    v = batch["tok_valid"].view(int(batch["n_graphs"]), int(batch["Tmax"]))
    tb = batch["token_batch"].numpy()
    assert set(np.unique(tb).tolist()) == set(range(int(batch["n_graphs"])))
    for b in range(int(batch["n_graphs"])):
        assert int((tb == b).sum()) == int(v[b].sum())
    return {
        "pred_diff_single_vs_batch": diff_batch,
        "pred_diff_after_adding_extra_graph": d_iso,
        "rows": rows.tolist(),
        "padding_masked": True,
        "valid_keys_block_diagonal_by_construction": True,
    }


@run("b_empty_groups")
def check_b_empty():
    path = fake_obj([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    ring = fake_obj([0] * 6, [(i, (i + 1) % 6) for i in range(6)])
    ad, ot = register_objects([path, ring], "__empty__")
    model, _ = build_b_model(ad, ot)
    y = predict_b(model, ad, "__empty__", [0, 1])
    assert np.isfinite(y).all()
    assert int(ot["meta"]["branch_by_family"]["S3"]) == 0
    assert int(ot["meta"]["n_cycle_tokens"]) == 1
    b_path = O.collate_ot(ad, np.array([0]))
    assert int(b_path["branch_tok_pos"].numel()) == 0
    assert int(b_path["ring_tok_pos"].numel()) == 0
    b_ring = O.collate_ot(ad, np.array([1]))
    assert int(b_ring["branch_tok_pos"].numel()) == 0
    assert int(b_ring["ring_tok_pos"].numel()) == 1
    assert int(b_ring["ring_len"][0]) == 6
    return {
        "path_prediction_finite": bool(np.isfinite(y[0])),
        "ring_prediction_finite": bool(np.isfinite(y[1])),
        "path_branch_tokens": 0,
        "path_cycle_tokens": 0,
        "ring_branch_tokens": 0,
        "ring_cycle_tokens": 1,
        "path_has_no_branch_or_cycle_group": True,
    }


@run("b_params_shared_embeddings_gradients")
def check_b_params():
    valid = D.load_cache("valid")
    pc = O.load_cache("valid")
    mean = {
        "atom": pc["meta"]["mean_atoms_per_molecule"],
        "branch": pc["meta"]["mean_branch_tokens_per_molecule"],
        "cycle": pc["meta"]["mean_cycle_tokens_per_molecule"],
    }
    model, info = M.build_model(D.load_norm(), group_mean=mean, device="cpu", seed=0)
    assert info["parameters"] == 470_469, info["parameters"]
    # one shared raw-embedding table each
    emb = [name for name, mod in model.named_modules() if isinstance(mod, torch.nn.Embedding)]
    assert sorted(emb) == ["atom_emb", "pair_emb", "type_emb"], emb
    # no unused P3 bond / junction / composer stack is created
    names = [n for n, _ in model.named_parameters()]
    assert not any("bond_encoder" in n for n in names)
    assert not any("junction" in n for n in names)
    assert not any(n.startswith("group_composer") for n in names)
    bd = info["parameters_breakdown"]
    assert sum(v for k, v in bd.items() if k != "total") == bd["total"] == info["parameters"]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    assert C.optimizer_parameter_count(opt) == 470_469
    assert all(p.requires_grad for p in model.parameters())
    # a batch with real branches and cycles
    det = (np.asarray(pc["n_cycle_per_mol"]) > 0) & (np.asarray(pc["n_branch_per_mol"]) > 0)
    rows = np.nonzero(det)[0][:8]
    batch = O.collate_ot(valid, rows)
    assert int(batch["branch_tok_pos"].numel()) > 0
    assert int(batch["ring_tok_pos"].numel()) > 0
    assert all(not (torch.is_tensor(v) and v.requires_grad) for v in batch.values())
    assert all(not (torch.is_tensor(v) and v.grad_fn is not None) for v in batch.values())
    y = torch.as_tensor(np.asarray(valid["labels"])[rows], dtype=torch.float32)
    watch = {
        "atom_encoder": model.atom_encoder.lins[0].weight,
        "branch_phi": model.branch_phi.lins[0].weight,
        "ring_phi": model.ring_phi.lins[0].weight,
        "proj_atom": model.proj_atom.weight,
        "proj_branch": model.proj_branch.lins[0].weight,
        "proj_cycle": model.proj_cycle.lins[0].weight,
        "type_emb": model.type_emb.weight,
        "rel_type": model.rel_type,
        "rel_dist": model.rel_dist,
        "block0_qkv": model.blocks[0].qkv.weight,
        "block3_ff": model.blocks[3].ff[0].weight,
        "reader": model.reader[0].weight,
    }
    model.train()
    loss = F.l1_loss(model(batch), y)
    loss.backward()
    grads = {k: (float(p.grad.abs().max()) if p.grad is not None else None)
             for k, p in watch.items()}
    zero = [k for k, v in grads.items() if not v]
    assert not zero, zero
    # relation tables start at zero but receive gradients through the bias
    assert float(model.rel_type.grad.abs().max()) > 0.0
    return {
        "parameters": info["parameters"],
        "breakdown": info["parameters_breakdown"],
        "embeddings": emb,
        "optimizer_parameters": C.optimizer_parameter_count(opt),
        "batch_tokens": int(batch["n_tokens_total"]),
        "grad_max": grads,
        "all_live_modules_have_gradient": True,
        "cache_tensors_are_static": True,
    }


@run("b_relation_semantics")
def check_b_relations():
    # star3 root 0 with three single bonds (types 1,2,3) plus a disconnected
    # 3-ring; check the relation categories directly against a numpy version.
    obj = fake_obj([0, 1, 2, 3, 4, 5, 6, 7],
                   [(0, 1), (0, 2), (0, 3), (4, 5), (5, 6), (6, 4)],
                   [1, 2, 3, 1, 1, 1])
    ad, ot = register_objects([obj], "__rel__")
    model, _ = build_b_model(ad, ot)
    batch = O.collate_ot(ad, np.array([0]))
    B, T = 1, int(batch["Tmax"])
    cat = model.relation_categories(batch, B, T)
    tt = batch["tok_type_pad"].view(B, T).numpy()[0]
    root = batch["tok_root_pad"].view(B, T).numpy()[0]
    mem = batch["tok_members_pad"].view(B, T, O.MAX_MEMBERS).numpy()[0]
    nm = batch["tok_member_mask"].view(B, T, O.MAX_MEMBERS).numpy()[0]
    valid = batch["tok_valid"].view(B, T).numpy()[0]
    members = [set(int(x) for x in mem[i][nm[i]]) for i in range(T)]
    # independent distance matrix
    n = int(obj.n_atoms)
    bmap = BF.bond_map(obj)
    adj = BF.adjacency(n, bmap)
    INF = 10 ** 9
    dist = np.full((n, n), INF)
    for s in range(n):
        dist[s, s] = 0
        frontier = [s]
        d = 0
        while frontier:
            d += 1
            nxt = []
            for u in frontier:
                for w in adj[u]:
                    if dist[s, w] == INF:
                        dist[s, w] = d
                        nxt.append(w)
            frontier = nxt
    errs = {}
    for i in range(T):
        for j in range(T):
            if not (valid[i] and valid[j]):
                continue
            ov = len(members[i] & members[j])
            got_ov = int(cat["overlap"][0, i, j])
            assert got_ov == ov, (i, j, got_ov, ov)
            d = min(dist[u, v] for u in members[i] for v in members[j])
            want = 10 if d >= INF else (9 if d >= 9 else d)
            got = int(cat["dist"][0, i, j])
            assert got == want, (i, j, got, want, d)
            want_t = int(tt[i]) * 5 + int(tt[j])
            assert int(cat["type"][0, i, j]) == want_t
            if tt[i] == 0 and tt[j] == 0:
                u, v = mem[i][0], mem[j][0]
                b = int(bmap.get((u, v), 0))
                want_b = b + 1 if b > 0 else 1
            else:
                want_b = 0
            assert int(cat["bond"][0, i, j]) == want_b, (i, j, cat["bond"][0, i, j], want_b)
            if root[i] >= 0 and root[j] >= 0:
                want_r = 2 if root[i] == root[j] else 1
            else:
                want_r = 0
            assert int(cat["root"][0, i, j]) == want_r, (i, j, cat["root"][0, i, j], want_r)
    # the disconnected 3-ring group is at distance 10 from the star
    types = np.asarray(ot["tok_type"])
    cyc = np.nonzero(types == O.TYPE_C)[0]
    atom = np.nonzero(types == O.TYPE_A)[0]
    d_cross = int(cat["dist"][0, int(atom[0]), int(cyc[0])])
    assert d_cross == 10, d_cross
    # the additive bias must equal explicit table lookups (the shipped model
    # uses a chunked one-hot matmul instead of advanced indexing) and must still
    # receive gradients on every relation table
    bias = model.relation_bias(batch, B, T)
    want_bias = (model.rel_type[cat["type"]] + model.rel_overlap[cat["overlap"]]
                 + model.rel_dist[cat["dist"]] + model.rel_bond[cat["bond"]]
                 + model.rel_root[cat["root"]]).permute(0, 3, 1, 2)
    keep = torch.as_tensor(valid.astype(bool))[None, None, None, :]
    want_bias = want_bias.masked_fill(~keep, -1e9)
    assert torch.allclose(bias, want_bias, atol=1e-6), (bias - want_bias).abs().max()
    pad_cols = ~keep[0, 0, 0]
    if int(pad_cols.sum()):
        assert float(bias[..., pad_cols].max()) <= -1e8
    model.zero_grad(set_to_none=True)
    loss = sum(bias[:, h] * float(h + 1) for h in range(4)).sum()
    loss.backward()
    grads = {k: float(getattr(model, k).grad.abs().sum()) for k in
             ("rel_type", "rel_overlap", "rel_dist", "rel_bond", "rel_root")}
    assert all(v > 0 for v in grads.values()), grads
    model.zero_grad(set_to_none=True)
    errs = {"checked_pairs": int(valid.sum() ** 2)}
    return {
        "bias_equals_table_lookup": True,
        "bias_grads": grads,
        "checked_pairs": errs["checked_pairs"],
        "tokens": int(valid.sum()),
        "bond_relations_exact": True,
        "root_relations_exact": True,
        "distance_categories_exact": True,
        "disconnected_category": d_cross,
        "overlap_exact": True,
    }


@run("b_schedule_and_checkpoint_semantics")
def check_b_schedule():
    steps = 79
    total = steps * 600
    lrs = [C.cosine_with_warmup_lr(k, total, 500, 1e-3, 1e-5) for k in (1, 500, 501, total)]
    assert abs(lrs[0] - 1e-3 / 500) < 1e-15
    assert 0.999e-3 <= lrs[2] <= 1e-3
    assert abs(lrs[3] - 1e-5) < 1e-12
    assert math.ceil(10000 / 128) == steps
    return {
        "steps_per_epoch": steps,
        "total_updates_600": total,
        "lr_examples": {"1": lrs[0], "500": lrs[1], "501": lrs[2], str(total): lrs[3]},
        "warmup_updates": 500,
    }


def write_report() -> dict:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    payload = {"checks": RESULTS}
    n_fail = sum(1 for r in RESULTS.values() if not r["pass"])
    payload["n_pass"] = len(RESULTS) - n_fail
    payload["n_fail"] = n_fail
    C.write_json(ANALYSIS / "cpu_checks.json", payload)
    return payload


def main() -> int:
    for split in ("valid",):
        if not O.cache_path(split).exists():
            O.save_split(split)
    order = (
        check_a_init, check_a_gradients,
        check_b_coverage, check_b_hand, check_b_ring_alignment,
        check_b_permutation, check_b_batch_isolation, check_b_empty,
        check_b_params, check_b_relations, check_b_schedule,
    )
    for fn in order:
        fn()
        name = getattr(fn, "check_name", fn.__name__)
        C.log(f"[ot-check] {name}: {'pass' if RESULTS[name]['pass'] else 'FAIL'}")
        if not RESULTS[name]["pass"]:
            C.log(RESULTS[name].get("traceback", RESULTS[name].get("error", "")))
    payload = write_report()
    C.log(f"[ot-check] {payload['n_pass']} passed / {payload['n_fail']} failed")
    return 1 if payload["n_fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
