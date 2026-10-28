"""Raw-graph, path-sampling, masking and masked-LCRC feature construction.

Frozen protocol (CJP-SSL v1), see ``DESIGN.md`` §3/§4:

* Only the official PyG ZINC ``subset=True`` **train** split raw ``x`` /
  ``edge_index`` / ``edge_attr`` are read.  ``y`` is never touched here.
* Structures are grouped by a canonical raw-graph fingerprint (typed
  Weisfeiler-Lehman hash) and split by a fixed SHA ordering into ~9,500
  fit / ~500 diagnostic structures.  A duplicated structure never crosses the
  split.
* A *visit* to a molecule selects at most three vertex-disjoint directed
  two-hop paths ``(v, w, u)`` from an independent ``SeedSequence`` keyed by
  ``(seed, molecule_id, visit_index, 20261029)``.
* Masking modifies the raw graph first (``x[u] := 27``, both directed records
  of the physical bond ``(w,u)`` := 0), then the frozen P3
  ``build_mol_objects`` rebuilds atom108 / bond7 / global33, then the masked
  LCRC arrays are composed from that rebuilt object.

The compose step is the only place where the mask booleans are consulted:
a categorical S1 record needs a visible neighbour **and** a visible bond; an
S2 typed-path record needs all four quantities visible.  Pure topology
(degrees, distances, path counts, closure ratio, relation pairs) always uses
the true unmodified topology.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
import torch

import cjp_common as C

L = C.L
D = C.D

import cscl_units as cu  # noqa: E402  (P3 canonicalisation helper)
import structural_relation_objects_v1 as sro  # noqa: E402  (frozen P3 builder)


# ---------------------------------------------------------------------------
# raw data (train only; labels never read)
# ---------------------------------------------------------------------------


def load_raw_train() -> list[dict[str, np.ndarray]]:
    """Official PyG ZINC subset train: raw ``x`` / ``edge_index`` / ``edge_attr``.

    The PyG property-label tensor is never accessed.  The returned order is the
    official split order, which is also the order of the frozen P3 object cache
    and of the frozen LCRC train cache.
    """
    from torch_geometric.datasets import ZINC

    ds = ZINC(root=str(L.C.ZINC_ROOT), subset=True, split="train")
    out: list[dict[str, np.ndarray]] = []
    for i in range(len(ds)):
        d = ds[i]
        out.append(
            {
                "mol_index": int(i),
                "x": np.ascontiguousarray(d.x.reshape(-1).long().numpy().astype(np.int64)),
                "edge_index": np.ascontiguousarray(d.edge_index.numpy().astype(np.int64)),
                "edge_attr": np.ascontiguousarray(
                    d.edge_attr.reshape(-1).long().numpy().astype(np.int64)
                ),
            }
        )
    return out


def graph_fingerprint(x: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray) -> str:
    """Canonical raw-graph fingerprint: typed Weisfeiler-Lehman hash (5 rounds)."""
    g = nx.Graph()
    g.add_nodes_from((int(i), {"t": int(t)}) for i, t in enumerate(np.asarray(x).reshape(-1)))
    seen: set[tuple[int, int]] = set()
    ei = np.asarray(edge_index)
    ea = np.asarray(edge_attr).reshape(-1)
    for k in range(ei.shape[1]):
        a, b = int(ei[0, k]), int(ei[1, k])
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        g.add_edge(a, b, bt=int(ea[k]))
    return nx.weisfeiler_lehman_graph_hash(
        g, node_attr="t", edge_attr="bt", iterations=5, digest_size=16
    )


def structure_split(
    raws: Sequence[Mapping[str, Any]],
    *,
    n_fit: int = C.PRETRAIN["n_fit"],
) -> dict[str, Any]:
    """Group by fingerprint, order groups by fixed SHA, fill fit then diag.

    Whole fingerprint groups are assigned to one side, so an exact duplicated
    structure can never appear in both the SSL fit and diagnostic sets.
    """
    t0 = time.perf_counter()
    fps = [graph_fingerprint(r["x"], r["edge_index"], r["edge_attr"]) for r in raws]
    groups: dict[str, list[int]] = {}
    for i, f in enumerate(fps):
        groups.setdefault(f, []).append(int(i))
    order = sorted(groups, key=lambda f: C.sha256_bytes(f.encode()))
    fit: list[int] = []
    diag: list[int] = []
    for f in order:
        members = groups[f]
        if len(fit) < int(n_fit):
            fit.extend(members)
        else:
            diag.extend(members)
    return {
        "n_molecules": len(raws),
        "n_unique_fingerprints": len(groups),
        "n_duplicate_groups": int(sum(1 for v in groups.values() if len(v) > 1)),
        "n_molecules_in_duplicate_groups": int(
            sum(len(v) for v in groups.values() if len(v) > 1)
        ),
        "fit_index": np.asarray(sorted(fit), dtype=np.int64),
        "diag_index": np.asarray(sorted(diag), dtype=np.int64),
        "fingerprints": fps,
        "fingerprint_order_sha": C.sha256_bytes("".join(order).encode()),
        "fingerprint_sha": C.sha256_bytes("".join(fps).encode()),
        "seconds": float(time.perf_counter() - t0),
    }


def verify_split_isomorphism(raws, split) -> dict[str, Any]:
    """Confirm that every multi-member fingerprint group is truly isomorphic."""

    def as_graph(r):
        g = nx.Graph()
        g.add_nodes_from((int(i), {"t": int(t)}) for i, t in enumerate(r["x"]))
        seen = set()
        ei, ea = r["edge_index"], r["edge_attr"]
        for k in range(ei.shape[1]):
            a, b = int(ei[0, k]), int(ei[1, k])
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            g.add_edge(a, b, bt=int(ea[k]))
        return g

    groups: dict[str, list[int]] = {}
    for i, f in enumerate(split["fingerprints"]):
        groups.setdefault(f, []).append(i)
    checked, failed = 0, []
    for f, members in groups.items():
        if len(members) < 2:
            continue
        ref = as_graph(raws[members[0]])
        for j in members[1:]:
            checked += 1
            if not nx.is_isomorphic(
                ref,
                as_graph(raws[j]),
                node_match=lambda a, b: a["t"] == b["t"],
                edge_match=lambda a, b: a["bt"] == b["bt"],
            ):
                failed.append((f, members))
    return {"pairs_checked": checked, "non_isomorphic_groups": failed,
            "pass": not failed}


# ---------------------------------------------------------------------------
# per-molecule topology + frozen P3 object (mask invariant parts)
# ---------------------------------------------------------------------------


def build_topology(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Build the mask-invariant topology cache and the unmasked LCRC arrays.

    The unmasked arrays are produced by the frozen ``lc_data.build_from_arrays``
    so that (a) the no-mask composition can be compared byte-for-byte with
    that reference and (b) the diagnostic path/cluster bookkeeping is identical.
    """
    x = np.asarray(raw["x"], dtype=np.int64).reshape(-1)
    ei = np.asarray(raw["edge_index"], dtype=np.int64)
    ea = np.asarray(raw["edge_attr"], dtype=np.int64).reshape(-1)
    n = int(len(x))
    mol_index = int(raw.get("mol_index", -1))

    directed = [(int(ei[0, k]), int(ei[1, k])) for k in range(ei.shape[1])]
    directed_types = [int(t) for t in ea]
    bonds = cu.unique_undirected_bonds(directed, directed_types)
    bu = np.asarray([b[0] for b in bonds], dtype=np.int64)
    bv = np.asarray([b[1] for b in bonds], dtype=np.int64)
    bt = np.asarray([b[2] for b in bonds], dtype=np.int64)
    m = int(len(bonds))

    pos: dict[tuple[int, int], int] = {}
    for k in range(ei.shape[1]):
        key = (int(ei[0, k]), int(ei[1, k]))
        if key in pos:
            raise RuntimeError(f"duplicate directed edge {key}")
        pos[key] = k
    dpos = np.asarray([[pos[(int(u), int(v))], pos[(int(v), int(u))]] for u, v in zip(bu, bv)],
                      dtype=np.int64).reshape(m, 2)

    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for bid in range(m):
        adj[int(bu[bid])].append((int(bv[bid]), bid))
        adj[int(bv[bid])].append((int(bu[bid]), bid))
    for i in range(n):
        adj[i].sort()

    # ---- frozen P3 object on the unmasked raw graph -----------------------
    obj = sro.build_mol_objects(mol_index, x, ei, ea)
    if not np.array_equal(obj.bond_index, np.stack([bu, bv], axis=1).astype(np.int64)):
        raise RuntimeError("P3 bond_index disagrees with PyG double-edge canonicalisation")
    if not np.array_equal(obj.bond_type, bt):
        raise RuntimeError("P3 bond_type disagrees with canonicalised bond types")
    if not np.array_equal(np.argmax(np.asarray(obj.atom_feat)[:, : C.N_ATOM_TYPES], axis=1), x):
        raise RuntimeError("P3 atom one-hot disagrees with raw PyG x")

    base = D.build_from_arrays(
        obj.atom_feat, x, obj.bond_index, obj.bond_type, obj.bond_feat, obj.global_feat
    )

    # ---- adjacency records (S1) in frozen enumeration order ---------------
    nbr_root, nbr_nbr, nbr_bond = [], [], []
    for v in range(n):
        for w, bid in adj[v]:
            nbr_root.append(v)
            nbr_nbr.append(w)
            nbr_bond.append(bid)
    nbr_root = np.asarray(nbr_root, dtype=np.int64)
    nbr_nbr = np.asarray(nbr_nbr, dtype=np.int64)
    nbr_bond = np.asarray(nbr_bond, dtype=np.int64)

    # ---- directed two-hop paths (S2) in frozen enumeration order ----------
    path_root, path_w, path_u, path_e1, path_e2 = [], [], [], [], []
    for v in range(n):
        for w, b1 in adj[v]:
            for u, b2 in adj[w]:
                if u == v:
                    continue
                path_root.append(v)
                path_w.append(w)
                path_u.append(u)
                path_e1.append(b1)
                path_e2.append(b2)
    path_root = np.asarray(path_root, dtype=np.int64)
    path_w = np.asarray(path_w, dtype=np.int64)
    path_u = np.asarray(path_u, dtype=np.int64)
    path_e1 = np.asarray(path_e1, dtype=np.int64)
    path_e2 = np.asarray(path_e2, dtype=np.int64)
    path_closure = np.asarray(base["path_closure"], dtype=np.int64) if len(path_root) \
        else np.zeros(0, dtype=np.int64)

    # ---- consistency with the frozen reference enumeration ----------------
    ref_nbr = np.asarray([int(x[u]) * C.N_BOND_TYPES + (int(bt[b]) - 1)
                          for u, b in zip(nbr_nbr, nbr_bond)], dtype=np.int64)
    if not np.array_equal(ref_nbr, np.asarray(base["nbr_code"], dtype=np.int64)):
        raise RuntimeError("S1 record order disagrees with frozen build_from_arrays")
    ref_cat = np.asarray(
        [int(x[w]) * 9 + (int(bt[b1]) - 1) * 3 + (int(bt[b2]) - 1)
         for w, b1, b2 in zip(path_w, path_e1, path_e2)], dtype=np.int64
    ) if len(path_root) else np.zeros(0, dtype=np.int64)
    ref_end63 = np.asarray(
        [int(x[u]) * 3 + (int(bt[b2]) - 1) for u, b2 in zip(path_u, path_e2)], dtype=np.int64
    ) if len(path_root) else np.zeros(0, dtype=np.int64)
    if not np.array_equal(ref_cat, np.asarray(base["path_cat"], dtype=np.int64)):
        raise RuntimeError("S2 typed-path order disagrees with frozen build_from_arrays")
    if not np.array_equal(ref_end63, np.asarray(base["path_end63"], dtype=np.int64)):
        raise RuntimeError("S2 end-code order disagrees with frozen build_from_arrays")
    if not np.array_equal(path_u.astype(np.int64), np.asarray(base["path_end"], dtype=np.int64)):
        raise RuntimeError("S2 endpoint order disagrees with frozen build_from_arrays")
    if not np.array_equal(path_closure.astype(np.int64),
                          np.asarray(base["path_closure"], dtype=np.int64)):
        raise RuntimeError("S2 closure order disagrees with frozen build_from_arrays")

    return {
        "n_atoms": n,
        "n_bonds": m,
        "mol_index": mol_index,
        "x_raw": x,
        "edge_index_raw": ei,
        "edge_attr_raw": ea,
        "bond_u": bu,
        "bond_v": bv,
        "bond_type_raw": bt,
        "directed_pos": dpos,
        "adj": adj,
        "base": base,
        "nbr_root": nbr_root,
        "nbr_nbr": nbr_nbr,
        "nbr_bond": nbr_bond,
        "path_root": path_root,
        "path_w": path_w,
        "path_u": path_u,
        "path_e1": path_e1,
        "path_e2": path_e2,
        "path_closure": path_closure,
    }


# ---------------------------------------------------------------------------
# frozen path sampler
# ---------------------------------------------------------------------------


def sample_paths(
    topo: Mapping[str, Any],
    *,
    seed: int,
    molecule_id: int,
    visit_index: int,
    max_paths: int = C.MAX_PATHS_PER_MOLECULE,
) -> np.ndarray:
    """All legal directed 2-paths, shuffled by an independent SeedSequence.

    ``SeedSequence([seed, molecule_id, visit_index, 20261029])`` -> shuffle all
    legal paths -> greedily take the first path whose three vertices are
    unseen -> at most ``max_paths``.  Returns path record ids.
    """
    n_paths = int(len(topo["path_root"]))
    if n_paths == 0:
        return np.zeros(0, dtype=np.int64)
    ss = np.random.SeedSequence([int(seed), int(molecule_id), int(visit_index),
                                C.MASK_SEED_TAG])
    rng = np.random.Generator(np.random.PCG64(ss))
    order = rng.permutation(n_paths)
    used: set[int] = set()
    selected: list[int] = []
    for pid in order:
        pid = int(pid)
        v = int(topo["path_root"][pid])
        w = int(topo["path_w"][pid])
        u = int(topo["path_u"][pid])
        if v in used or w in used or u in used:
            continue
        used.update((v, w, u))
        selected.append(pid)
        if len(selected) >= int(max_paths):
            break
    return np.asarray(selected, dtype=np.int64)


# ---------------------------------------------------------------------------
# masked graph -> P3 rebuild -> masked LCRC arrays
# ---------------------------------------------------------------------------


def apply_mask(
    topo: Mapping[str, Any],
    path_ids: Sequence[int],
) -> dict[str, Any]:
    """Mask the raw graph: x[u] := 27 and both directed records of (w,u) := 0."""
    n = int(topo["n_atoms"])
    m = int(topo["n_bonds"])
    x_masked = np.array(topo["x_raw"], dtype=np.int64, copy=True)
    ea_masked = np.array(topo["edge_attr_raw"], dtype=np.int64, copy=True)
    atom_mask = np.zeros(n, dtype=bool)
    bond_mask = np.zeros(m, dtype=bool)
    for pid in np.asarray(path_ids, dtype=np.int64):
        u = int(topo["path_u"][pid])
        bid = int(topo["path_e2"][pid])
        if atom_mask[u] or bond_mask[bid]:
            raise RuntimeError("selected paths are not vertex-disjoint")
        atom_mask[u] = True
        bond_mask[bid] = True
        x_masked[u] = C.ATOM_MASK_TOKEN
        ea_masked[topo["directed_pos"][bid]] = C.BOND_MASK_TOKEN
    bond_type_masked = np.array(topo["bond_type_raw"], dtype=np.int64, copy=True)
    bond_type_masked[bond_mask] = C.BOND_MASK_TOKEN
    return {
        "x": x_masked,
        "edge_attr": ea_masked,
        "atom_mask": atom_mask,
        "bond_mask": bond_mask,
        "bond_type": bond_type_masked,
    }


def compose_masked(
    topo: Mapping[str, Any],
    obj_masked,
    atom_type_masked: np.ndarray,
    bond_type_masked: np.ndarray,
    atom_mask: np.ndarray,
    bond_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Masked LCRC inputs (the only consumer of the mask booleans)."""
    n = int(topo["n_atoms"])
    base = topo["base"]

    # ---- S1: visible neighbour type AND visible bond type ----------------
    nr = topo["nbr_root"]
    nn = topo["nbr_nbr"]
    nb = topo["nbr_bond"]
    keep_s1 = (~atom_mask[nn]) & (~bond_mask[nb])
    nbr_root = nr[keep_s1]
    nbr_code = (
        atom_type_masked[nn[keep_s1]] * C.N_BOND_TYPES + (bond_type_masked[nb[keep_s1]] - 1)
    ).astype(np.uint8)
    nbr_counts = np.bincount(nbr_root, minlength=n).astype(np.int64)
    nbr_ptr = np.concatenate([[0], np.cumsum(nbr_counts)]).astype(np.int32)

    # ---- S2: all four typed quantities visible ---------------------------
    pr = topo["path_root"]
    pw = topo["path_w"]
    pu = topo["path_u"]
    pe1 = topo["path_e1"]
    pe2 = topo["path_e2"]
    keep_s2 = (~atom_mask[pw]) & (~atom_mask[pu]) & (~bond_mask[pe1]) & (~bond_mask[pe2])
    root2 = pr[keep_s2]
    w2 = pw[keep_s2]
    u2 = pu[keep_s2]
    bt1 = bond_type_masked[pe1[keep_s2]]
    bt2 = bond_type_masked[pe2[keep_s2]]
    path_cat = (
        atom_type_masked[w2] * (C.N_BOND_TYPES * C.N_BOND_TYPES)
        + (bt1 - 1) * C.N_BOND_TYPES
        + (bt2 - 1)
    ).astype(np.uint8)
    path_end63 = (atom_type_masked[u2] * C.N_BOND_TYPES + (bt2 - 1)).astype(np.uint8)
    path_counts = np.bincount(root2, minlength=n).astype(np.int64)
    path_ptr = np.concatenate([[0], np.cumsum(path_counts)]).astype(np.int32)

    return {
        "n_atoms": np.int64(n),
        "n_bonds": np.int64(int(topo["n_bonds"])),
        "atom_feat": np.asarray(obj_masked.atom_feat, dtype=np.float32),
        "atom_type": atom_type_masked.astype(np.int64),
        "atom_scalar": np.asarray(base["atom_scalar"], dtype=np.float32),
        "bond_feat": np.asarray(obj_masked.bond_feat, dtype=np.float32),
        "bond_type": bond_type_masked.astype(np.int64),
        "bond_index": np.asarray(base["bond_index"], dtype=np.int16),
        "nbr_ptr": nbr_ptr,
        "nbr_code": nbr_code,
        "path_ptr": path_ptr,
        "path_cat": path_cat,
        "path_end63": path_end63,
        "path_end": u2.astype(np.int16),
        "path_closure": topo["path_closure"][keep_s2].astype(np.uint8),
        "pair_i": np.asarray(base["pair_i"], dtype=np.int16),
        "pair_j": np.asarray(base["pair_j"], dtype=np.int16),
        "pair_cat": np.asarray(base["pair_cat"], dtype=np.uint8),
        "global": np.asarray(obj_masked.global_feat, dtype=np.float32).reshape(-1),
    }


def reference_compose_masked(
    topo: Mapping[str, Any],
    atom_feat: np.ndarray,
    bond_feat: np.ndarray,
    global_feat: np.ndarray,
    atom_type_masked: np.ndarray,
    bond_type_masked: np.ndarray,
    atom_mask: np.ndarray,
    bond_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Independent, deliberately naive re-implementation used by CPU gate G3.

    It re-derives S1/S2 with explicit Python loops over the *rebuilt* atom108
    adjacency/visibility instead of reusing the cached records, so a bug in the
    cached-record composition cannot hide.
    """
    n = int(topo["n_atoms"])
    bond_index = np.asarray(topo["base"]["bond_index"], dtype=np.int64)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for bid in range(bond_index.shape[0]):
        a, b = int(bond_index[bid, 0]), int(bond_index[bid, 1])
        adj[a].append((b, bid))
        adj[b].append((a, bid))
    for i in range(n):
        adj[i].sort()

    nbr_codes: list[int] = []
    nbr_atoms: list[int] = []
    for v in range(n):
        for w, b1 in adj[v]:
            if (not atom_mask[w]) and (not bond_mask[b1]):
                nbr_atoms.append(v)
                nbr_codes.append(int(atom_type_masked[w]) * 3
                                 + int(bond_type_masked[b1]) - 1)
    # closure flags come from the true topology (mask invariant)
    path_root, path_cat, path_end63, path_end, path_closure = [], [], [], [], []
    for v in range(n):
        for w, b1 in adj[v]:
            for u, b2 in adj[w]:
                if u == v:
                    continue
                visible = (
                    (not atom_mask[w]) and (not atom_mask[u])
                    and (not bond_mask[b1]) and (not bond_mask[b2])
                )
                if not visible:
                    continue
                path_root.append(v)
                path_cat.append(
                    int(atom_type_masked[w]) * 9 + (int(bond_type_masked[b1]) - 1) * 3
                    + (int(bond_type_masked[b2]) - 1)
                )
                path_end63.append(int(atom_type_masked[u]) * 3 + int(bond_type_masked[b2]) - 1)
                path_end.append(u)
    adj_sets = [set(nb for nb, _ in adj[i]) for i in range(n)]
    for k, v in enumerate(path_root):
        path_closure.append(1 if path_end[k] in adj_sets[v] else 0)

    def _ptr(counts):
        return np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)

    nbr_counts = np.bincount(np.asarray(nbr_atoms, dtype=np.int64), minlength=n)
    path_counts = np.bincount(np.asarray(path_root, dtype=np.int64), minlength=n)
    return {
        "n_atoms": np.int64(n),
        "n_bonds": np.int64(int(topo["n_bonds"])),
        "atom_feat": np.asarray(atom_feat, dtype=np.float32),
        "atom_type": np.asarray(atom_type_masked, dtype=np.int64),
        "atom_scalar": np.asarray(topo["base"]["atom_scalar"], dtype=np.float32),
        "bond_feat": np.asarray(bond_feat, dtype=np.float32),
        "bond_type": np.asarray(bond_type_masked, dtype=np.int64),
        "bond_index": np.asarray(topo["base"]["bond_index"], dtype=np.int16),
        "nbr_ptr": _ptr(nbr_counts),
        "nbr_code": np.asarray(nbr_codes, dtype=np.uint8),
        "path_ptr": _ptr(path_counts),
        "path_cat": np.asarray(path_cat, dtype=np.uint8),
        "path_end63": np.asarray(path_end63, dtype=np.uint8),
        "path_end": np.asarray(path_end, dtype=np.int16),
        "path_closure": np.asarray(path_closure, dtype=np.uint8),
        "pair_i": np.asarray(topo["base"]["pair_i"], dtype=np.int16),
        "pair_j": np.asarray(topo["base"]["pair_j"], dtype=np.int16),
        "pair_cat": np.asarray(topo["base"]["pair_cat"], dtype=np.uint8),
        "global": np.asarray(global_feat, dtype=np.float32).reshape(-1),
    }


def prepare_visit(
    topo: Mapping[str, Any],
    *,
    seed: int,
    molecule_id: int,
    visit_index: int,
    path_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """One masked visit: sample paths, mask raw graph, rebuild P3, compose LCRC."""
    if path_ids is None:
        path_ids = sample_paths(topo, seed=seed, molecule_id=molecule_id,
                                visit_index=visit_index)
    mask = apply_mask(topo, path_ids)
    obj = sro.build_mol_objects(
        int(topo["mol_index"]), mask["x"], topo["edge_index_raw"], mask["edge_attr"]
    )
    mol = compose_masked(
        topo,
        obj,
        mask["x"],
        mask["bond_type"],
        mask["atom_mask"],
        mask["bond_mask"],
    )
    queries = np.zeros((len(path_ids), 3), dtype=np.int64)
    for k, pid in enumerate(np.asarray(path_ids, dtype=np.int64)):
        u = int(topo["path_u"][pid])
        bid = int(topo["path_e2"][pid])
        w = int(topo["path_w"][pid])
        target = (
            C.N_BOND_TYPES * int(topo["x_raw"][u])
            + (int(topo["bond_type_raw"][bid]) - 1)
        )
        queries[k] = (int(topo["path_root"][pid]), w, target)
    masked_sha = C.tensor_sha(
        torch.from_numpy(mask["x"]),
        torch.from_numpy(topo["edge_index_raw"]),
        torch.from_numpy(mask["edge_attr"]),
        torch.from_numpy(queries),
    )
    stats = {
        "n_selected_paths": int(len(path_ids)),
        "n_queries": int(len(queries)),
        "atoms_masked": int(mask["atom_mask"].sum()),
        "bonds_masked": int(mask["bond_mask"].sum()),
        "n_atoms": int(topo["n_atoms"]),
        "n_bonds": int(topo["n_bonds"]),
    }
    return {
        "mol": mol,
        "queries": queries,
        "stats": stats,
        "masked_sha": masked_sha,
        "mask": mask,
        "path_ids": np.asarray(path_ids, dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# batching (reuses the frozen lc_data collate)
# ---------------------------------------------------------------------------


def cache_from_molecules(mols: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    return D._concat_molecules(list(mols))


def batch_from_molecules(
    mols: Sequence[Mapping[str, Any]],
    queries: Sequence[np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collate a list of masked molecules plus their query tables.

    Returns ``(batch, query)`` where ``query`` holds global atom indices for
    the visible root ``v`` and middle ``w`` plus the 63-class target.
    """
    mini = cache_from_molecules(mols)
    batch = D.collate(mini, np.arange(len(mols), dtype=np.int64))
    atom_off = batch["atom_off"].numpy()
    graph_idx, qv, qw, qt = [], [], [], []
    per_mol = np.zeros(len(mols), dtype=np.int64)
    for i, q in enumerate(queries):
        q = np.asarray(q, dtype=np.int64).reshape(-1, 3)
        per_mol[i] = len(q)
        if len(q) == 0:
            continue
        graph_idx.append(np.full(len(q), i, dtype=np.int64))
        qv.append(q[:, 0] + atom_off[i])
        qw.append(q[:, 1] + atom_off[i])
        qt.append(q[:, 2])
    if graph_idx:
        query = {
            "graph": torch.from_numpy(np.concatenate(graph_idx)),
            "v": torch.from_numpy(np.concatenate(qv)),
            "w": torch.from_numpy(np.concatenate(qw)),
            "target": torch.from_numpy(np.concatenate(qt)),
            "per_molecule": torch.from_numpy(per_mol),
        }
    else:
        query = {
            "graph": torch.zeros(0, dtype=torch.int64),
            "v": torch.zeros(0, dtype=torch.int64),
            "w": torch.zeros(0, dtype=torch.int64),
            "target": torch.zeros(0, dtype=torch.int64),
            "per_molecule": torch.from_numpy(per_mol),
        }
    return batch, query


def mask_coverage(stats: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    total_atoms = sum(s["n_atoms"] for s in stats)
    total_bonds = sum(s["n_bonds"] for s in stats)
    total_paths = sum(s["n_selected_paths"] for s in stats)
    return {
        "n_molecules": int(len(stats)),
        "n_selected_paths": int(total_paths),
        "paths_per_molecule": float(total_paths) / max(len(stats), 1),
        "atom_mask_fraction": float(sum(s["atoms_masked"] for s in stats))
        / max(total_atoms, 1),
        "bond_mask_fraction": float(sum(s["bonds_masked"] for s in stats))
        / max(total_bonds, 1),
        "molecules_with_no_path": int(sum(1 for s in stats if s["n_selected_paths"] == 0)),
    }


__all__ = [
    "load_raw_train", "graph_fingerprint", "structure_split",
    "verify_split_isomorphism", "build_topology", "sample_paths", "apply_mask",
    "compose_masked", "reference_compose_masked", "prepare_visit",
    "cache_from_molecules", "batch_from_molecules", "mask_coverage",
]
