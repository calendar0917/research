"""BOSI v0 mandatory CPU gates.

Writes ``analysis/cpu_checks.json``.  Every gate is a real check on the frozen
official-train cache rows (Fit8k / Monitor / Locked only); official valid and
official test are never loaded.

Gate list (the numbering is the task's):

  G1  Fit8k degree bound / partner bound (no silent truncation)
  G2  four relations are mutually exclusive and complete; sum_r n == deg - 1
  G3  triangles / extra common neighbours / ring blocks match the original
      molecular graph *and* the frozen BJOC object builder used by DCC-L
  G4  third-moment counterexample + cross-channel pairing counterexample
  G5  independent NumPy/Python implementation of M
  G6  atom relabelling / bond direction / edge order / batch recombination
  G7  R and B have identical parameters and identical shared initial state
  G8  R and B really enter their own forward (incl. histogram-preserving
      relation permutation: R moves, B is bit-exactly invariant)
  G9  real Fit8k R/B statistic difference, non-empty partner fraction,
      coexistence of relation groups
  G10 gradients on atom / half-edge / statistics / bond composer / reader
  G11 no pre-training branch collapse and no scale explosion
  G12 official valid/test paths are not referenced or loaded
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bosi_common as BC  # noqa: E402  (also puts ad_common on the path)
import ad_common as C  # noqa: E402
import bosi_data as BD  # noqa: E402
import bosi_models as BM  # noqa: E402


def _load():
    cache = BC.load_train_cache()
    rel = torch.load(BC.CACHE / "bosi_rel_train.pt", map_location="cpu",
                     weights_only=False)
    return cache, rel


def _spec_from_cache(cache, row: int) -> dict:
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    a0, a1 = int(ap[row]), int(ap[row + 1])
    b0, b1 = int(bp[row]), int(bp[row + 1])
    return {
        "atom_feat": np.asarray(cache["atom_feat"])[a0:a1].copy(),
        "bond_index": np.asarray(cache["bond_index"])[b0:b1].astype(np.int64).copy(),
        "bond_feat": np.asarray(cache["bond_feat"])[b0:b1].copy(),
        "global": np.asarray(cache["global_feat"])[row].copy(),
    }


# ---------------------------------------------------------------------------
# G1 / G2 / G9 use the flat relation cache
# ---------------------------------------------------------------------------


def g1_degree(cache, rel, rows):
    deg_half = np.asarray(rel["half_deg"], dtype=np.int64)
    hp = np.asarray(rel["half_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    n_part = np.diff(np.asarray(rel["partner_ptr"], dtype=np.int64))
    fit_half = np.concatenate([np.arange(hp[r], hp[r + 1]) for r in rows])
    fit_part = np.concatenate([np.arange(np.asarray(rel["partner_ptr"])[h],
                                         np.asarray(rel["partner_ptr"])[h + 1])
                               for h in fit_half]) if len(fit_half) else np.zeros(0, int)
    d = deg_half[fit_half]
    values, counts = np.unique(d, return_counts=True)
    bd = np.diff(bp)[rows]
    out = {
        "gate": "G1_fit8k_degree_and_partner_bound",
        "n_fit_rows": int(len(rows)),
        "max_atom_degree": int(d.max()) if len(d) else 0,
        "degree_histogram_fit8k": {int(v): int(c) for v, c in zip(values, counts)},
        "max_partners_per_half_edge": int(n_part[fit_half].max()) if len(fit_half) else 0,
        "design_bound_num_partners_le_3": bool(len(fit_half) == 0 or n_part[fit_half].max() <= 3),
        "truncation_applied": False,
        "n_atoms_fit8k": int(bd.sum()),
        "n_bonds_fit8k": int(len(fit_half)),
    }
    out["pass"] = bool(out["max_atom_degree"] <= 4 and out["design_bound_num_partners_le_3"])
    return out


def g2_mece(cache, rel, rows):
    pp = np.asarray(rel["partner_ptr"], dtype=np.int64)
    pr = np.asarray(rel["partner_rel"], dtype=np.int64)
    ph = np.asarray(rel["partner_half_raw"], dtype=np.int64)
    hp = np.asarray(rel["half_ptr"], dtype=np.int64)
    half_idx = np.concatenate([np.arange(hp[r], hp[r + 1]) for r in rows])
    deg = np.asarray(rel["half_deg"], dtype=np.int64)[half_idx]
    counts = np.diff(pp)[half_idx]
    tgt = np.repeat(np.arange(len(half_idx)), counts)
    part_rows = np.concatenate([np.arange(pp[h], pp[h + 1]) for h in half_idx]) \
        if len(half_idx) else np.zeros(0, np.int64)
    orders = np.lexsort((ph[part_rows], tgt))
    tgt_s, ph_s = tgt[orders], ph[part_rows][orders]
    dup = int(((tgt_s[1:] == tgt_s[:-1]) & (ph_s[1:] == ph_s[:-1])).sum())
    # per half-edge group counts re-derived from the partner rows and compared
    # to the counts the model consumes
    n_rows = np.bincount(tgt * BD.N_REL + pr[part_rows],
                         minlength=len(half_idx) * BD.N_REL).reshape(-1, BD.N_REL)
    out = {
        "gate": "G2_relations_mutually_exclusive_and_complete",
        "n_half_edges": int(len(half_idx)),
        "n_partner_rows": int(len(part_rows)),
        "partner_counts_equal_deg_minus_1": bool(np.array_equal(counts, np.maximum(deg - 1, 0))),
        "n_violations": int((counts != np.maximum(deg - 1, 0)).sum()),
        "duplicate_partner_in_same_half_edge": dup,
        "group_counts_partition_all_partners": bool(np.array_equal(n_rows.sum(1), counts)),
        "n_partners_per_half_edge_histogram": {
            int(k): int(v) for k, v in zip(*np.unique(counts, return_counts=True))},
        "relation_histogram": {name: int((pr[part_rows] == k).sum())
                               for k, name in enumerate(BD.REL_NAMES)},
    }
    out["pass"] = bool(out["partner_counts_equal_deg_minus_1"] and dup == 0
                       and out["group_counts_partition_all_partners"])
    return out


def g3_structure(cache, rel, rows):
    """Cross-check the relation construction against two independent sources."""
    import networkx as nx

    jp = np.asarray(cache["junc_ptr"], dtype=np.int64)
    jf = np.asarray(cache["junc_feat"], dtype=np.float32)
    bi_all = np.asarray(cache["bond_index"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bf_all = np.asarray(cache["bond_feat"], dtype=np.float32)
    af_all = np.asarray(cache["atom_feat"], dtype=np.float32)
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    deg_atom_offset = 28  # one-hot degree block inside the 108-d atom feature

    tri_junc = 0
    com_junc = 0
    tri_junc_minus_extra = 0
    n_junc = 0
    tri_nx = 0
    ring_flag_mismatch = 0
    block_mismatch = 0
    common_mismatch = 0
    tri_atom_mismatch = 0
    n_bonds = 0
    for r in rows:
        n_atoms = int(ap[r + 1] - ap[r])
        b0, b1 = int(bp[r]), int(bp[r + 1])
        bonds = bi_all[b0:b1]
        j0, j1 = int(jp[r]), int(jp[r + 1])
        jfc = jf[j0:j1]
        tri_junc += int((jfc[:, 1] > 0.5).sum())
        com = np.rint(np.expm1(jfc[:, 2])).astype(np.int64)
        com_junc += int((com > 0).sum())
        n_junc += int(j1 - j0)
        G = nx.Graph()
        G.add_nodes_from(range(n_atoms))
        G.add_edges_from((int(u), int(v)) for u, v in bonds)
        blocks = list(nx.biconnected_component_edges(G))
        key_to_block = {}
        for k, edges in enumerate(blocks):
            if len(edges) < 3:
                continue
            for u, v in edges:
                key_to_block[(min(u, v), max(u, v))] = k
        # ring flags vs frozen bond_feat[:, 4]
        ring_expected = np.zeros(b1 - b0, dtype=np.int64)
        for (u, v) in key_to_block:
            ring_expected[int(np.where((bonds[:, 0] == u) & (bonds[:, 1] == v))[0][0])] = 1
        ring_flag_mismatch += int((ring_expected != np.rint(bf_all[b0:b1, 4])).sum())
        # triangles / common neighbours directly from networkx
        for u, v in bonds:
            for w in G.neighbors(u):
                if w > v and G.has_edge(v, w):
                    tri_nx += 1
        # per-atom triangle count inside atom_feat
        tri_feat = af_all[ap[r]:ap[r + 1], deg_atom_offset + 5 + 1 + 6 + 2 + 4 + 28 + 28 + 2]
        tri_nx_atom = np.array([sum(1 for a in G.neighbors(i) for b in G.neighbors(i)
                                    if a < b and G.has_edge(a, b)) for i in range(n_atoms)],
                               dtype=np.float32)
        tri_atom_mismatch += int((np.abs(tri_feat - tri_nx_atom) > 0.5).sum())
        # block ids of this molecule must be reproduced by the relation builder
        rl = BD.relation_arrays(n_atoms, bonds, np.zeros(len(bonds), dtype=np.int64))
        ours = rl["block_of"]
        ref = np.asarray([key_to_block.get((min(int(u), int(v)), max(int(u), int(v))), -1)
                          for u, v in bonds], dtype=np.int64)
        # block ids are arbitrary labels; compare the induced partition
        same = True
        for i in range(len(bonds)):
            for j in range(len(bonds)):
                if (ours[i] == ours[j]) != (ref[i] == ref[j]):
                    same = False
        block_mismatch += int(not same)
        n_bonds += int(b1 - b0)

    out = {
        "gate": "G3_structure_matches_original_graph_and_frozen_objects",
        "n_rows": int(len(rows)),
        "n_bonds": int(n_bonds),
        "junction_triangle_pairs": int(tri_junc),
        "relation_T_partner_rows_expected": int(2 * tri_junc),
        "junction_common_neighbour_pairs": int(com_junc),
        "relation_Q_partner_rows_expected": int(2 * com_junc),
        "triangle_rows_networkx": int(tri_nx),
        "triangle_rows_from_atom_feature": "checked per atom (see mismatch count)",
        "triangle_atom_feature_mismatch_rows": int(tri_atom_mismatch),
        "bond_ring_flag_mismatch_rows": int(ring_flag_mismatch),
        "biconnected_partition_mismatch_rows": int(block_mismatch),
        "max_common_neighbours_excluding_centre": int(np.rint(np.expm1(jf[:, 2])).max()),
        "note": "block ids are arbitrary labels; the check compares the induced partition",
    }
    out["pass"] = bool(tri_atom_mismatch == 0 and ring_flag_mismatch == 0 and block_mismatch == 0)
    return out


def _relation_rows(cache, rel, rows):
    hp = np.asarray(rel["half_ptr"], dtype=np.int64)
    pp = np.asarray(rel["partner_ptr"], dtype=np.int64)
    pr = np.asarray(rel["partner_rel"], dtype=np.int64)
    half_idx = np.concatenate([np.arange(hp[r], hp[r + 1]) for r in rows])
    return half_idx, pp, pr


def g9_statistics(cache, rel, rows, model, device, compare_arm="B"):
    batch = BD.collate(cache, rel, rows)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    model.eval()
    old_arm = model.arm
    model.arm = "R"
    with torch.no_grad():
        _, stats_r = model(batch, return_stats=True)
    model.arm = compare_arm
    with torch.no_grad():
        _, stats_b = model(batch, return_stats=True)
    model.arm = old_arm
    M_R, n = stats_r["M"], stats_r["n"]
    M_B = stats_b["M"]
    n_cpu = n.detach().cpu().numpy()
    mr = M_R.detach().cpu().numpy().astype(np.float64).reshape(len(M_R), BD.N_REL, 96)
    mb = M_B.detach().cpu().numpy().astype(np.float64).reshape(len(M_B), BD.N_REL, 96)
    denom = np.linalg.norm(mr.reshape(len(mr), -1), axis=1)
    diff = np.linalg.norm((mr - mb).reshape(len(mr), -1), axis=1)
    has_partner = n_cpu.sum(1) > 0
    nonempty = n_cpu > 0
    n_groups = nonempty.sum(1)
    allocation_defined = n_groups >= 2
    rel_ratio = np.where(denom > 1e-8, diff / np.maximum(denom, 1e-12), np.nan)
    hp = np.asarray(rel["half_ptr"], dtype=np.int64)
    half_idx = np.concatenate([np.arange(hp[r], hp[r + 1]) for r in rows])
    mol_of_half = np.repeat(np.arange(len(rows)), np.diff(hp)[rows])
    deg = np.asarray(rel["half_deg"])[half_idx]
    mol_has = {name: int((np.bincount(mol_of_half, weights=nonempty[:, k].astype(np.float64),
                                      minlength=len(rows)) > 0).sum())
               for k, name in enumerate(BD.REL_NAMES)}
    out = {
        "gate": "G9_real_fit8k_relation_usage_and_R_vs_B_difference",
        "compared_arm": compare_arm,
        "n_rows": int(len(rows)),
        "n_half_edges": int(len(n_cpu)),
        "fraction_half_edges_with_nonzero_partners": float(has_partner.mean()),
        "fraction_half_edges_deg1": float((deg == 1).mean()),
        "group_nonzero_fraction": {name: float(nonempty[:, k].mean())
                                   for k, name in enumerate(BD.REL_NAMES)},
        "molecules_with_nonzero_group": mol_has,
        "molecule_fraction_with_nonzero_group": {k: v / max(len(rows), 1)
                                                 for k, v in mol_has.items()},
        "partner_row_share": {name: float(n_cpu[:, k].sum() / max(n_cpu.sum(), 1))
                              for k, name in enumerate(BD.REL_NAMES)},
        "n_nonempty_groups_histogram": {
            int(k): int(v) for k, v in zip(*np.unique(n_groups, return_counts=True))},
        "fraction_with_two_or_more_coexisting_groups": float(allocation_defined.mean()),
        "R_vs_B_abs_norm_diff_mean": float(diff.mean()),
        "R_vs_B_abs_norm_diff_median": float(np.median(diff)),
        "R_vs_B_relative_diff_mean": float(np.nanmean(rel_ratio[has_partner])) if has_partner.any() else 0.0,
        "R_vs_B_relative_diff_median": float(np.nanmedian(rel_ratio[has_partner])) if has_partner.any() else 0.0,
        "fraction_relative_diff_gt_1e-3_overall": float(np.nanmean(rel_ratio[has_partner] > 1e-3))
        if has_partner.any() else 0.0,
        "fraction_relative_diff_gt_1e-3_among_allocation_defined": float(
            np.nanmean(rel_ratio[allocation_defined] > 1e-3)) if allocation_defined.any() else 0.0,
        "relative_diff_mean_among_allocation_defined": float(
            np.nanmean(rel_ratio[allocation_defined])) if allocation_defined.any() else 0.0,
        "abs_norm_diff_mean_among_allocation_defined": float(
            diff[allocation_defined].mean()) if allocation_defined.any() else 0.0,
        "M_R_norm_median": float(np.median(denom)),
        "note": "allocation is only defined when >=2 relation groups are non-empty; "
                "with one non-empty group R and the leave-one-out broadcast coincide exactly",
    }
    out["pass"] = bool(has_partner.mean() > 0.5
                       and allocation_defined.mean() > 0.05
                       and out["fraction_relative_diff_gt_1e-3_among_allocation_defined"] > 0.95)
    return out


# ---------------------------------------------------------------------------
# G4 / G5 / G6 / G7 / G8 / G10 / G11 on real molecules
# ---------------------------------------------------------------------------


def g4_counterexamples():
    # third moment: same n, p1, p2 but different sum T3
    A = np.array([-0.5, 0.0, 0.5])
    Bv = np.array([-1 / (2 * np.sqrt(3)), -1 / (2 * np.sqrt(3)), 1 / np.sqrt(3)])
    def moments(x):
        return (float(len(x)), float(x.sum()), float((x ** 2).sum()),
                float((4 * x ** 3 - 3 * x).sum()), float((2 * x ** 2 - 1).sum()))
    mA, mB = moments(A), moments(Bv)
    third = {
        "A": mA, "B": mB,
        "same_n_p1_p2": bool(np.isclose(mA[0], mB[0]) and np.isclose(mA[1], mB[1])
                             and np.isclose(mA[2], mB[2])),
        "sum_T3_A": mA[3], "sum_T3_B": mB[3],
        "separated_by_third_moment": bool(abs(mA[3] - mB[3]) > 0.5),
    }
    # cross-channel pairing: identical per-channel moments at every order,
    # different after projecting x+y
    P = np.array([(0.0, 0.0), (1.0, 1.0)])
    Q = np.array([(0.0, 1.0), (1.0, 0.0)])
    per_channel = {}
    for name, X in (("P", P), ("Q", Q)):
        per_channel[name] = [[float((X[:, c] ** k).sum()) for c in (0, 1)]
                             for k in (1, 2, 3, 4)]
    proj = {"P": float(((P[:, 0] + P[:, 1]) ** 2).sum()),
            "Q": float(((Q[:, 0] + Q[:, 1]) ** 2).sum())}
    out = {
        "gate": "G4_third_moment_and_cross_channel_pairing",
        "third_moment": third,
        "cross_channel": {
            "per_channel_moments_k1_k4": per_channel,
            "identical_per_channel_moments": bool(
                np.allclose(np.asarray(per_channel["P"]), np.asarray(per_channel["Q"]))),
            "sum_projection_squared_P": proj["P"],
            "sum_projection_squared_Q": proj["Q"],
            "cross_channel_pairing_lost": bool(not np.isclose(proj["P"], proj["Q"])),
        },
    }
    out["pass"] = bool(third["separated_by_third_moment"]
                       and out["cross_channel"]["identical_per_channel_moments"]
                       and out["cross_channel"]["cross_channel_pairing_lost"])
    return out


def g5_numpy(cache, rel, rows, model, device):
    """Independent float64 forward of M for a sample of molecules."""
    rows = np.asarray(rows[:16], dtype=np.int64)
    batch = BD.collate(cache, rel, rows)
    batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    model.eval()
    with torch.no_grad():
        _, st = model(batch_dev, return_stats=True)
    x = st["x"].detach().cpu().numpy().astype(np.float64)
    M_ref = np.zeros((len(x), BD.N_REL, 3, BM.RANK), dtype=np.float64)
    n_ref = np.zeros((len(x), BD.N_REL), dtype=np.float64)
    hp = np.asarray(rel["half_ptr"], dtype=np.int64)
    pp = np.asarray(rel["partner_ptr"], dtype=np.int64)
    ph = np.asarray(rel["partner_half_raw"], dtype=np.int64)
    pr = np.asarray(rel["partner_rel"], dtype=np.int64)
    # batch-local half mapping
    counts = np.diff(hp)[rows]
    half_off = np.concatenate([[0], np.cumsum(counts)[:-1]])
    lookup = {}
    for i, r in enumerate(rows):
        for h in range(hp[r], hp[r + 1]):
            lookup[h] = int(half_off[i] + (h - hp[r]))
    for i, r in enumerate(rows):
        for h in range(hp[r], hp[r + 1]):
            target = lookup[h]
            for p in range(pp[h], pp[h + 1]):
                src = lookup[int(ph[p])]
                rel_id = int(pr[p])
                v = x[src]
                T = np.stack([v, 2 * v * v - 1, 4 * v ** 3 - 3 * v], axis=0)
                M_ref[target, rel_id] += T
                n_ref[target, rel_id] += 1
    M_batch = st["M"].detach().cpu().numpy().astype(np.float64).reshape(-1, BD.N_REL, 3, BM.RANK)
    n_batch = st["n"].detach().cpu().numpy().astype(np.float64)
    mmax = float(np.abs(M_ref - M_batch).max())
    nmax = float(np.abs(n_ref - n_batch).max())
    out = {
        "gate": "G5_independent_numpy_implementation",
        "n_rows": int(len(rows)),
        "max_abs_M_difference": mmax,
        "max_abs_count_difference": nmax,
        "tolerance": 1e-6,
    }
    out["pass"] = bool(mmax < 1e-6 and nmax == 0.0)
    return out


def g6_invariance(cache, norm, device):
    rows = np.arange(8, dtype=np.int64)  # rows 0..7 of the official-train cache
    specs = [_spec_from_cache(cache, int(r)) for r in rows]
    model, _ = BM.build_model(norm, "R", 0, device)
    model.eval()
    with torch.no_grad():
        base = model({k: (v.to(device) if torch.is_tensor(v) else v)
                      for k, v in BD.batch_from_specs(specs).items()}).cpu().numpy()

    rng = np.random.default_rng(20261026)
    perm = rng.permutation(specs[0]["atom_feat"].shape[0])
    p = np.empty_like(perm)
    p[perm] = np.arange(len(perm))
    permuted = dict(specs[0])
    permuted["atom_feat"] = specs[0]["atom_feat"][p]
    permuted["bond_index"] = perm[specs[0]["bond_index"]]
    rev = [dict(s) for s in specs]
    for s in rev:
        bi = s["bond_index"].copy()
        bi[:, 0], bi[:, 1] = bi[:, 1].copy(), bi[:, 0].copy()
        s["bond_index"] = bi
    order = rng.permutation(len(specs))
    reordered = [dict(specs[int(i)]) for i in order]
    for s in reordered:
        bperm = rng.permutation(len(s["bond_index"]))
        s["bond_index"] = s["bond_index"][bperm]
        s["bond_feat"] = s["bond_feat"][bperm]

    def run(sp):
        with torch.no_grad():
            return model({k: (v.to(device) if torch.is_tensor(v) else v)
                          for k, v in BD.batch_from_specs(sp).items()}).cpu().numpy()

    o_perm = run([permuted] + specs[1:])
    o_rev = run(rev)
    o_ord = run(reordered)
    o_order = np.empty_like(o_ord)
    o_order[order] = o_ord
    singles = np.concatenate([run([s]) for s in specs])
    deltas = {
        "atom_relabel": float(np.abs(o_perm - base).max()),
        "bond_direction_reverse": float(np.abs(o_rev - base).max()),
        "edge_order_and_batch_order": float(np.abs(o_order - base).max()),
        "batch_recombination": float(np.abs(singles - base).max()),
    }
    out = {"gate": "G6_symmetry_and_batch_invariance", "max_abs_delta": deltas,
           "tolerance": 1e-5, "predictions": base.tolist()}
    out["pass"] = bool(max(deltas.values()) < 1e-5)
    return out


def g7_params(norm):
    mR, iR = BM.build_model(norm, "R", 0, "cpu")
    mB, iB = BM.build_model(norm, "B", 0, "cpu")
    shapes = {k: tuple(v.shape) for k, v in mR.state_dict().items()}
    shapesB = {k: tuple(v.shape) for k, v in mB.state_dict().items()}
    out = {
        "gate": "G7_R_and_B_identical_parameters_and_init",
        "parameters_R": iR["parameters"],
        "parameters_B": iB["parameters"],
        "expected_parameters": BC.EXPECTED_PARAMETERS,
        "cap": BC.PARAM_CAP,
        "same_shape": bool(shapes == shapesB),
        "same_initial_state_sha256": bool(iR["initial_state_sha256"]
                                          == iB["initial_state_sha256"]),
        "initial_state_sha256": iR["initial_state_sha256"],
        "breakdown": iR["parameters_breakdown"],
        "ssl": iR["ssl"],
        "tensor_level_max_abs_diff": float(max(
            (mR.state_dict()[k] - mB.state_dict()[k]).abs().max().item()
            for k in mR.state_dict())),
    }
    out["pass"] = bool(out["parameters_R"] == out["parameters_B"] == BC.EXPECTED_PARAMETERS
                       and out["same_shape"] and out["same_initial_state_sha256"]
                       and out["tensor_level_max_abs_diff"] == 0.0
                       and out["parameters_R"] <= BC.PARAM_CAP)
    return out


def g8_arm_live(cache, rel, norm, device):
    rows = np.arange(32, dtype=np.int64)
    model, _ = BM.build_model(norm, "R", 0, device)
    model.eval()
    batch = BD.collate(cache, rel, rows)
    bdev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    # histogram-preserving relation permutation inside every half-edge
    rng = np.random.default_rng(7)
    bperm = dict(batch)
    pr = batch["partner_rel"].numpy().copy()
    tgt = batch["partner_tgt"].numpy()
    for h in np.unique(tgt):
        idx = np.where(tgt == h)[0]
        if len(idx) > 1:
            pr[idx] = pr[idx][rng.permutation(len(idx))]
    bperm["partner_rel"] = torch.from_numpy(pr)
    bperm = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in bperm.items()}

    with torch.no_grad():
        model.arm = "R"
        yR = model(bdev).cpu().numpy()
        yR_perm = model(bperm).cpu().numpy()
        model.arm = "B"
        yB = model(bdev).cpu().numpy()
        yB_perm = model(bperm).cpu().numpy()
        model.arm = "R"
    out = {
        "gate": "G8_R_and_B_enter_their_own_forward",
        "max_abs_y_R_minus_y_B": float(np.abs(yR - yB).max()),
        "mean_abs_y_R_minus_y_B": float(np.abs(yR - yB).mean()),
        "arms_produce_different_predictions": bool(np.abs(yR - yB).max() > 1e-6),
        "relation_permutation_moves_R": float(np.abs(yR_perm - yR).max()),
        "relation_permutation_moves_B": float(np.abs(yB_perm - yB).max()),
        "B_is_invariant_to_partner_relabels": bool(np.abs(yB_perm - yB).max() < 1e-6),
    }
    out["pass"] = bool(out["arms_produce_different_predictions"]
                       and out["relation_permutation_moves_R"] > 1e-4
                       and out["B_is_invariant_to_partner_relabels"])
    return out


def g10_gradients(cache, rel, norm, device):
    rows = np.arange(32, dtype=np.int64)
    model, _ = BM.build_model(norm, "R", 0, device)
    model.train()
    batch = BD.collate(cache, rel, rows)
    bdev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    y = model(bdev)
    target = torch.zeros_like(y)
    torch.nn.functional.l1_loss(y, target).backward()
    mods = {}
    for name in ("atom_encoder", "bond_encoder", "half_encoder", "side", "compose", "reader"):
        g = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
        mods[name] = {
            "n_tensors_with_grad": int(len(g)),
            "n_tensors": int(sum(1 for _ in getattr(model, name).parameters())),
            "grad_norm": float(np.sqrt(sum(float(x.pow(2).sum()) for x in g))),
        }
    # sensitivity: zero the relation statistics -> prediction must move a lot
    model.eval()
    with torch.no_grad():
        y0 = model(bdev).cpu().numpy()
        orig = model.side.forward
        def masked(x):
            x = x.clone()
            x[:, 2 * BM.WIDTH + BM.RANK: 2 * BM.WIDTH + BM.RANK + BD.N_REL * 3 * BM.RANK] = 0.0
            return orig(x)
        model.side.forward = masked
        y1 = model(bdev).cpu().numpy()
        model.side.forward = orig
    out = {
        "gate": "G10_gradients_reach_every_declared_module",
        "modules": mods,
        "all_declared_modules_have_nonzero_grad": bool(all(v["grad_norm"] > 0
                                                           for v in mods.values())),
        "stats_block_zeroing_max_abs_dy": float(np.abs(y1 - y0).max()),
        "stats_block_is_read": bool(np.abs(y1 - y0).max() > 1e-4),
    }
    out["pass"] = bool(out["all_declared_modules_have_nonzero_grad"]
                       and out["stats_block_is_read"])
    return out


def g11_scale(cache, rel, norm, device):
    rows = np.arange(256, dtype=np.int64)
    model, _ = BM.build_model(norm, "R", 0, device)
    model.eval()
    batch = BD.collate(cache, rel, rows)
    bdev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        _, st = model(bdev, return_stats=True)
    out = {"gate": "G11_no_collapsed_branch_and_no_scale_explosion"}
    for key in ("x", "M", "s", "c", "z"):
        v = st[key].detach().cpu().numpy()
        out[key] = {
            "std": float(v.std()),
            "abs_mean": float(np.abs(v).mean()),
            "max_abs": float(np.abs(v).max()),
            "fraction_exact_zero": float((v == 0).mean()),
        }
    out["group_nonzero_fraction"] = {
        name: float((st["n"][:, k] > 0).float().mean()) for k, name in enumerate(BD.REL_NAMES)}
    out["finite"] = bool(all(np.isfinite(st[k].detach().cpu().numpy()).all()
                             for k in ("x", "M", "s", "c", "z")))
    out["pass"] = bool(out["finite"]
                       and out["M"]["std"] > 1e-4
                       and out["M"]["max_abs"] < 1e3
                       and out["z"]["max_abs"] < 1e3
                       and out["c"]["std"] > 1e-4)
    return out


def g12_no_official(cache, rel):
    import re

    here = Path(__file__).resolve().parent
    loader = re.compile(r"ad_valid\.pt|ad_test\.pt|ad_valid\b|ad_test\b|pyg_zinc|ZINC\(")
    hits = {}
    for p in sorted(here.glob("*.py")):
        found = sorted(set(loader.findall(p.read_text())))
        if p.name == "bosi_checks.py":
            found = []
        if found:
            hits[p.name] = found
    out = {
        "gate": "G12_official_valid_and_test_untouched",
        "loaded_caches": ["ad_train.pt (official train 10k rows only)"],
        "valid_cache_loaded": False,
        "test_cache_loaded": False,
        "source_files_referencing_official_splits": hits,
        "gate_evidence": "every module in this round imports only "
                         "bosi_common.load_train_cache(); no valid/test loader exists",
    }
    out["pass"] = bool(not hits
                       and cache["split"] == "train"
                       and rel["n_molecules"] == int(cache["n_molecules"]))
    return out


def main() -> int:
    cache, rel = _load()
    rows = BC.split_rows()
    norm = BC.fit_only_norm(cache, rows["fit8k"])
    device = torch.device("cpu")
    gates = {}
    gates["G1"] = g1_degree(cache, rel, rows["fit8k"])
    gates["G2"] = g2_mece(cache, rel, rows["fit8k"])
    gates["G3"] = g3_structure(cache, rel, rows["fit8k"][:400])
    gates["G4"] = g4_counterexamples()
    model, _ = BM.build_model(norm, "R", 0, device)
    gates["G5"] = g5_numpy(cache, rel, rows["fit8k"][:16], model, device)
    gates["G6"] = g6_invariance(cache, norm, device)
    gates["G7"] = g7_params(norm)
    gates["G8"] = g8_arm_live(cache, rel, norm, device)
    gates["G9"] = g9_statistics(cache, rel, rows["fit8k"][:2048], model, device)
    gates["G10"] = g10_gradients(cache, rel, norm, device)
    gates["G11"] = g11_scale(cache, rel, norm, device)
    gates["G12"] = g12_no_official(cache, rel)

    # G3's expected Q/T rows must equal the relation counts of the same rows
    hp = np.asarray(rel["half_ptr"]); pr = np.asarray(rel["partner_rel"])
    pp = np.asarray(rel["partner_ptr"])
    sub_rows = rows["fit8k"][:400]
    sub_half = np.concatenate([np.arange(hp[r], hp[r + 1]) for r in sub_rows])
    sub_part = np.concatenate([np.arange(pp[h], pp[h + 1]) for h in sub_half])
    rel_sub = pr[sub_part]
    gates["G3"]["relation_T_partner_rows_observed"] = int((rel_sub == 0).sum())
    gates["G3"]["relation_Q_partner_rows_observed"] = int((rel_sub == 1).sum())
    gates["G3"]["pass"] = bool(gates["G3"]["pass"]
                               and gates["G3"]["relation_T_partner_rows_observed"]
                               == gates["G3"]["relation_T_partner_rows_expected"]
                               and gates["G3"]["relation_Q_partner_rows_observed"]
                               == gates["G3"]["relation_Q_partner_rows_expected"])

    # DESIGN-literal B formula diagnostic: how much of R-B is the extra
    # shrink / self-inclusion term rather than the assignment?
    rows9 = rows["fit8k"][:2048]
    model.arm = "Bfull"
    out_bf = g9_statistics(cache, rel, rows9, model, device, compare_arm="Bfull")
    model.arm = "R"
    gates["G9"]["design_literal_Bfull_relative_diff_median"] = \
        out_bf["R_vs_B_relative_diff_median"]
    gates["G9"]["design_literal_Bfull_fraction_relative_diff_gt_1e-3"] = \
        out_bf["fraction_relative_diff_gt_1e-3_overall"]
    gates["G9"]["design_literal_Bfull_relative_diff_mean"] = \
        out_bf["R_vs_B_relative_diff_mean"]
    gates["G9"]["design_literal_Bfull_relative_diff_median"] = \
        out_bf["R_vs_B_relative_diff_median"]

    n_pass = sum(1 for g in gates.values() if g.get("pass"))
    summary = {
        "round": BC.ROUND_ID,
        "written": BC.now(),
        "device": "cpu",
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "gates": gates,
        "n_pass": int(n_pass),
        "n_total": int(len(gates)),
        "verdict": "CPU_GO" if n_pass == len(gates) else "CPU_NO_GO",
    }
    BC.write_json(BC.ANALYSIS / "cpu_checks.json", summary)
    for k, g in gates.items():
        BC.log(f"[bosi-checks] {k} {'PASS' if g.get('pass') else 'FAIL'} {g['gate']}")
    BC.log(f"[bosi-checks] {n_pass}/{len(gates)} -> {summary['verdict']}")
    return 0 if n_pass == len(gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
