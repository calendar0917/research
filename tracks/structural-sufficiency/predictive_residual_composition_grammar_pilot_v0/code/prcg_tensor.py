"""PRCG-v0 featurisation: word trees, real port roles and inter-word bonds.

Every quantity here is a pure function of one molecule plus the frozen grammar.
All records are keyed by molecule-local word indices; the collator only adds
integer offsets, so no atom id ever becomes a model feature.

Port role vector (12 deterministic floats, invariant under automorphism): the
canonical role of one port atom inside the word's internal graph.  ``PRED_PORT_
MEAN`` replaces it by the mean over the word's port set; nothing else changes.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import prcg_graph as G

ROLE_DIM = 12
CTX_DIM = 9   # environment-facing per-word atom context summary
WORD_SCALARS = 7
N_BOND_TYPES = 4


def _role_vector(mol: G.Molecule, support: Sequence[int], sig: G.WordSignature,
                 atom: int) -> np.ndarray:
    k = len(support)
    slot = sig.order.index(int(atom))
    orbit_size = float(sig.orbit_size[slot])
    orbit_rep = float(sig.orbit_rep[slot])
    inside = set(int(x) for x in support)
    at_atom: list[int] = []
    for y, b in mol.adj[int(atom)]:
        if y in inside:
            at_atom.append(int(b))
    deg = len(at_atom)
    type_counts = np.zeros(3, np.float64)
    conj_sum = 0.0
    for b in at_atom:
        t = int(mol.bond_core[b][0])
        if t == 0:
            type_counts[0] += 1
        elif t == 1 or t == 2:
            type_counts[1] += 1
        else:
            type_counts[2] += 1
        conj_sum += float(mol.bond_core[b][1])
    # internal distance profile from the port atom
    dist = {int(atom): 0}
    frontier = [int(atom)]
    d = 0
    while frontier:
        d += 1
        nxt = []
        for x in frontier:
            for y, _ in mol.adj[x]:
                if y in inside and y not in dist:
                    dist[y] = d
                    nxt.append(y)
        frontier = nxt
    prof = np.zeros(3, np.float64)
    for v, dd in dist.items():
        if dd == 0:
            continue
        prof[0 if dd == 1 else (1 if dd == 2 else 2)] += 1
    denom = max(1.0, k - 1.0)
    # is the port atom inside a cycle of the induced subgraph?
    in_ring = 0.0
    nbrs = [y for y, _ in mol.adj[int(atom)] if y in inside]
    for i in range(len(nbrs)):
        for j in range(i + 1, len(nbrs)):
            seen = {nbrs[i]}
            stack = [nbrs[i]]
            found = False
            while stack:
                x = stack.pop()
                if x == nbrs[j]:
                    found = True
                    break
                for y, _ in mol.adj[x]:
                    if y in inside and y != int(atom) and y not in seen:
                        seen.add(y)
                        stack.append(y)
            if found:
                in_ring = 1.0
                break
        if in_ring:
            break
    return np.asarray([
        orbit_size / max(1.0, float(k)),
        deg / max(1.0, k - 1.0),
        type_counts[0] / 4.0,
        type_counts[1] / 4.0,
        type_counts[2] / 4.0,
        prof[0] / denom,
        prof[1] / denom,
        prof[2] / denom,
        in_ring,
        (orbit_rep + 1.0) / max(1.0, float(k)),
        k / 5.0,
        conj_sum / max(1.0, float(deg)),
    ], np.float64)


def _external_bonds(mol: G.Molecule, support: Sequence[int]) -> list[int]:
    inside = set(int(x) for x in support)
    out = []
    for b in range(mol.n_bonds):
        u, v = int(mol.bond_u[b]), int(mol.bond_v[b])
        if (u in inside) != (v in inside):
            out.append(b)
    return out


def atom_context_rows(mol: G.Molecule) -> np.ndarray:
    """The nine context fields kept for the model (never part of a word type).

    degree, min_ring, in_cycle, log1p(degree), isolated, numH, hybridisation,
    chirality, radical electrons.  These are molecule-level properties of an
    atom, so they may only enter through the environment-facing channels
    (per-word context summary and the graph scalars), never through the word
    state itself.
    """
    deg = mol.atom_context[:, 0]
    return np.stack([
        deg / 4.0, mol.atom_context[:, 1] / 8.0, mol.atom_context[:, 2],
        np.log1p(deg) / 2.0, (deg == 0).astype(np.float64),
        mol.atom_intrinsic[:, 1] / 4.0, mol.atom_intrinsic[:, 3] / 5.0,
        mol.atom_intrinsic[:, 0] / 3.0, mol.atom_intrinsic[:, 2] / 4.0,
    ], axis=1)


def word_scalars(mol: G.Molecule, support: Sequence[int]) -> np.ndarray:
    k = len(support)
    internal = mol.internal_bonds(support)
    counts = np.zeros(N_BOND_TYPES, np.float64)
    for b in internal:
        counts[int(mol.bond_core[b][0])] += 1
    cycle_rank = max(0, len(internal) - k + 1)
    return np.asarray([k / 5.0, len(internal) / 10.0] + list(counts / 10.0)
                      + [cycle_rank / 3.0], np.float64)


def molecule_features(mol: G.Molecule, part: G.Partition) -> dict[str, Any]:
    """All per-molecule model inputs of one tokenisation."""
    ctx_rows = atom_context_rows(mol)
    hist = part.history
    final = {w.index: i for i, w in enumerate(part.words)}
    n_words = len(hist)
    role = np.zeros((n_words, ROLE_DIM), np.float64)
    mean_role = np.zeros((n_words, ROLE_DIM), np.float64)
    n_ports = np.zeros(n_words, np.int64)
    ctx = np.zeros((n_words, CTX_DIM), np.float64)
    scal = np.zeros((n_words, WORD_SCALARS), np.float64)
    rule = np.zeros(n_words, np.int64)
    size = np.zeros(n_words, np.int64)
    child1 = np.full(n_words, -1, np.int64)
    child2 = np.full(n_words, -1, np.int64)
    atom_core = np.zeros((n_words, 3), np.int64)
    atom_intr = np.zeros((n_words, 4), np.int64)
    for w in hist:
        i = w.index
        rule[i] = w.rule
        size[i] = w.size
        scal[i] = word_scalars(mol, w.support)
        ctx[i] = ctx_rows[list(w.support)].mean(axis=0)
        if w.rule < 0:
            a = int(w.support[0])
            atom_core[i] = mol.atom_core[a]
            atom_intr[i] = mol.atom_intrinsic[a]
        else:
            child1[i], child2[i] = w.children
        ports = [a for a in w.support
                 if any(y not in w.support for y, _ in mol.adj[a])]
        n_ports[i] = len(ports)
        if ports:
            roles = np.stack([_role_vector(mol, w.support, w.sig, a) for a in ports])
            mean_role[i] = roles.mean(axis=0)
    # crossing bonds of every merged word
    cross: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for w in hist:
        if w.rule < 0:
            continue
        a = hist[w.children[0]]
        b = hist[w.children[1]]
        for bidx in mol.bonds_between(a.support, b.support):
            u, v = int(mol.bond_u[bidx]), int(mol.bond_v[bidx])
            if u in a.support:
                pa, pb = u, v
            else:
                pa, pb = v, u
            cross.append((w.index, bidx,
                          _role_vector(mol, a.support, a.sig, pa),
                          _role_vector(mol, b.support, b.sig, pb)))
    # inter-word bonds of the final partition
    # every edge endpoint is a *history* word index: rows of all per-word
    # tensors are history rows, not positions inside the final partition
    owner = {}
    by_index = {}
    for w in part.words:
        by_index[w.index] = w
        for a in w.support:
            owner[a] = w.index
    inter: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []
    for b in range(mol.n_bonds):
        u, v = int(mol.bond_u[b]), int(mol.bond_v[b])
        wi, wj = owner[u], owner[v]
        if wi == wj:
            continue
        wa, wb = by_index[wi], by_index[wj]
        inter.append((wi, wj, b,
                      _role_vector(mol, wa.support, wa.sig, u),
                      _role_vector(mol, wb.support, wb.sig, v)))
    return {"n_words": n_words, "rule": rule, "size": size, "child1": child1,
            "child2": child2, "atom_core": atom_core, "atom_intr": atom_intr,
            "mean_role": mean_role, "n_ports": n_ports, "ctx": ctx, "scal": scal,
            "final": np.asarray([1 if w.index in final else 0 for w in hist], np.int64),
            "cross": cross, "inter": inter,
            "n_final": len(part.words)}


CACHE_KEYS = ("w_ptr", "w_rule", "w_size", "w_child1", "w_child2", "w_atom_core",
              "w_atom_intr", "w_mean_role", "w_nports", "w_ctx", "w_scal", "w_final",
              "x_ptr", "x_word", "x_bond", "x_role_l", "x_role_r",
              "e_ptr", "e_i", "e_j", "e_bond", "e_role_i", "e_role_j",
              "b_ptr", "b_core", "b_stereo", "mol_graph", "mol_atoms", "mol_bonds",
              "row_id")


def featurize(molecules: Mapping[int, G.Molecule], rows: Sequence[int],
              grammar: G.Grammar) -> dict[str, np.ndarray]:
    rows = [int(r) for r in np.asarray(rows, np.int64).tolist()]
    out: dict[str, list[Any]] = {k: [] for k in CACHE_KEYS[:-3]}
    w_ptr = [0]
    x_ptr = [0]
    e_ptr = [0]
    b_ptr = [0]
    graph = []
    n_atoms = []
    n_bonds = []
    for r in rows:
        mol = molecules[r]
        part, _ = grammar.apply(mol)
        f = molecule_features(mol, part)
        out["w_rule"].append(f["rule"])
        out["w_size"].append(f["size"])
        out["w_child1"].append(f["child1"])
        out["w_child2"].append(f["child2"])
        out["w_atom_core"].append(f["atom_core"])
        out["w_atom_intr"].append(f["atom_intr"])
        out["w_mean_role"].append(f["mean_role"])
        out["w_nports"].append(f["n_ports"])
        out["w_ctx"].append(f["ctx"])
        out["w_scal"].append(f["scal"])
        out["w_final"].append(f["final"])
        w_ptr.append(w_ptr[-1] + f["n_words"])
        if f["cross"]:
            out["x_word"].append(np.asarray([c[0] for c in f["cross"]], np.int64))
            out["x_bond"].append(np.asarray([c[1] for c in f["cross"]], np.int64))
            out["x_role_l"].append(np.stack([c[2] for c in f["cross"]]))
            out["x_role_r"].append(np.stack([c[3] for c in f["cross"]]))
        else:
            out["x_word"].append(np.zeros(0, np.int64))
            out["x_bond"].append(np.zeros(0, np.int64))
            out["x_role_l"].append(np.zeros((0, ROLE_DIM)))
            out["x_role_r"].append(np.zeros((0, ROLE_DIM)))
        x_ptr.append(x_ptr[-1] + len(f["cross"]))
        if f["inter"]:
            out["e_i"].append(np.asarray([c[0] for c in f["inter"]], np.int64))
            out["e_j"].append(np.asarray([c[1] for c in f["inter"]], np.int64))
            out["e_bond"].append(np.asarray([c[2] for c in f["inter"]], np.int64))
            out["e_role_i"].append(np.stack([c[3] for c in f["inter"]]))
            out["e_role_j"].append(np.stack([c[4] for c in f["inter"]]))
        else:
            out["e_i"].append(np.zeros(0, np.int64))
            out["e_j"].append(np.zeros(0, np.int64))
            out["e_bond"].append(np.zeros(0, np.int64))
            out["e_role_i"].append(np.zeros((0, ROLE_DIM)))
            out["e_role_j"].append(np.zeros((0, ROLE_DIM)))
        e_ptr.append(e_ptr[-1] + len(f["inter"]))
        out["b_core"].append(mol.bond_core)
        out["b_stereo"].append(mol.bond_stereo)
        b_ptr.append(b_ptr[-1] + mol.n_bonds)
        graph.append(prcg_graph_scalars(mol))
        n_atoms.append(mol.n_atoms)
        n_bonds.append(mol.n_bonds)

    def cat(name: str, dtype=np.float64, width: int | None = None) -> np.ndarray:
        items = out[name]
        if not items:
            return np.zeros((0, width or 0), dtype)
        return np.concatenate(items, axis=0).astype(dtype, copy=False)

    cache = {
        "w_ptr": np.asarray(w_ptr, np.int64),
        "w_rule": cat("w_rule", np.int64), "w_size": cat("w_size", np.int64),
        "w_child1": cat("w_child1", np.int64), "w_child2": cat("w_child2", np.int64),
        "w_atom_core": cat("w_atom_core", np.int64), "w_atom_intr": cat("w_atom_intr", np.int64),
        "w_mean_role": cat("w_mean_role").astype(np.float32),
        "w_nports": cat("w_nports", np.int64), "w_ctx": cat("w_ctx").astype(np.float32),
        "w_scal": cat("w_scal").astype(np.float32), "w_final": cat("w_final", np.int64),
        "x_ptr": np.asarray(x_ptr, np.int64),
        "x_word": cat("x_word", np.int64), "x_bond": cat("x_bond", np.int64),
        "x_role_l": cat("x_role_l").astype(np.float32),
        "x_role_r": cat("x_role_r").astype(np.float32),
        "e_ptr": np.asarray(e_ptr, np.int64),
        "e_i": cat("e_i", np.int64), "e_j": cat("e_j", np.int64),
        "e_bond": cat("e_bond", np.int64),
        "e_role_i": cat("e_role_i").astype(np.float32),
        "e_role_j": cat("e_role_j").astype(np.float32),
        "b_ptr": np.asarray(b_ptr, np.int64),
        "b_core": cat("b_core", np.int64), "b_stereo": cat("b_stereo", np.int64),
        "mol_graph": np.stack(graph, axis=0).astype(np.float32),
        "mol_atoms": np.asarray(n_atoms, np.int64),
        "mol_bonds": np.asarray(n_bonds, np.int64),
        "row_id": np.asarray(rows, np.int64),
    }
    return cache


def prcg_graph_scalars(mol: G.Molecule) -> np.ndarray:
    import prcg_data as D
    return D.graph_scalars(mol)


def collate(rows: Sequence[int], cache: Mapping[str, np.ndarray], index: Mapping[int, int],
            graph_mean: np.ndarray, graph_scale: np.ndarray, device: Any = "cpu"
            ) -> dict[str, Any]:
    """Concatenate the cached per-molecule records of one batch.

    Only integer offsets are added: a molecule-local word/bond index becomes a
    batch-local one, exactly as in the frozen GWIC-v0 collator.
    """
    import torch

    pos = [int(index[int(r)]) for r in rows]
    w_parts = [(int(cache["w_ptr"][p]), int(cache["w_ptr"][p + 1])) for p in pos]
    x_parts = [(int(cache["x_ptr"][p]), int(cache["x_ptr"][p + 1])) for p in pos]
    e_parts = [(int(cache["e_ptr"][p]), int(cache["e_ptr"][p + 1])) for p in pos]
    b_parts = [(int(cache["b_ptr"][p]), int(cache["b_ptr"][p + 1])) for p in pos]
    counts_w = np.asarray([hi - lo for lo, hi in w_parts], np.int64)
    counts_x = np.asarray([hi - lo for lo, hi in x_parts], np.int64)
    counts_e = np.asarray([hi - lo for lo, hi in e_parts], np.int64)
    counts_b = np.asarray([hi - lo for lo, hi in b_parts], np.int64)
    w_off = np.cumsum([0] + counts_w.tolist()[:-1]).astype(np.int64)
    b_off = np.cumsum([0] + counts_b.tolist()[:-1]).astype(np.int64)
    w_off_x = np.repeat(w_off, counts_x)
    w_off_e = np.repeat(w_off, counts_e)
    b_off_x = np.repeat(b_off, counts_x)
    b_off_e = np.repeat(b_off, counts_e)

    def take(key: str, parts, dtype=None) -> np.ndarray:
        arr = np.concatenate([cache[key][lo:hi] for lo, hi in parts], axis=0)
        return arr.astype(dtype, copy=False) if dtype is not None else arr

    n_words = int(counts_w.sum())
    out = {
        "w_rule": torch.as_tensor(take("w_rule", w_parts, np.int64), dtype=torch.long),
        "w_size": torch.as_tensor(take("w_size", w_parts, np.int64), dtype=torch.long),
        "w_child1": torch.as_tensor(take("w_child1", w_parts, np.int64) + np.repeat(w_off, counts_w),
                                    dtype=torch.long),
        "w_child2": torch.as_tensor(take("w_child2", w_parts, np.int64) + np.repeat(w_off, counts_w),
                                    dtype=torch.long),
        "w_atom_core": torch.as_tensor(take("w_atom_core", w_parts, np.int64), dtype=torch.long),
        "w_atom_intr": torch.as_tensor(take("w_atom_intr", w_parts, np.int64), dtype=torch.long),
        "w_mean_role": torch.as_tensor(take("w_mean_role", w_parts, np.float32), dtype=torch.float32),
        "w_nports": torch.as_tensor(take("w_nports", w_parts, np.float32), dtype=torch.float32),
        "w_ctx": torch.as_tensor(take("w_ctx", w_parts, np.float32), dtype=torch.float32),
        "w_scal": torch.as_tensor(take("w_scal", w_parts, np.float32), dtype=torch.float32),
        "w_final": torch.as_tensor(take("w_final", w_parts, np.float32), dtype=torch.float32),
        "x_word": torch.as_tensor(take("x_word", x_parts, np.int64) + w_off_x, dtype=torch.long),
        "x_bond": torch.as_tensor(take("x_bond", x_parts, np.int64) + b_off_x, dtype=torch.long),
        "x_role_l": torch.as_tensor(take("x_role_l", x_parts, np.float32), dtype=torch.float32),
        "x_role_r": torch.as_tensor(take("x_role_r", x_parts, np.float32), dtype=torch.float32),
        "e_i": torch.as_tensor(take("e_i", e_parts, np.int64) + w_off_e, dtype=torch.long),
        "e_j": torch.as_tensor(take("e_j", e_parts, np.int64) + w_off_e, dtype=torch.long),
        "e_bond": torch.as_tensor(take("e_bond", e_parts, np.int64) + b_off_e, dtype=torch.long),
        "e_role_i": torch.as_tensor(take("e_role_i", e_parts, np.float32), dtype=torch.float32),
        "e_role_j": torch.as_tensor(take("e_role_j", e_parts, np.float32), dtype=torch.float32),
        "b_core": torch.as_tensor(take("b_core", b_parts, np.int64), dtype=torch.long),
        "b_stereo": torch.as_tensor(take("b_stereo", b_parts, np.int64), dtype=torch.long),
        "graph": torch.as_tensor((np.stack([cache["mol_graph"][p] for p in pos]) - graph_mean)
                                 / graph_scale, dtype=torch.float32),
        "mol_word_ptr": torch.as_tensor(np.concatenate([[0], np.cumsum(counts_w)]), dtype=torch.long),
        "mol_inter_ptr": torch.as_tensor(np.concatenate([[0], np.cumsum(counts_e)]), dtype=torch.long),
        "n_words": n_words,
        "n_molecules": len(rows),
        "n_cross": int(counts_x.sum()),
        "n_inter": int(counts_e.sum()),
        "rows": np.asarray(rows, np.int64),
    }
    if device != "cpu":
        out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out.items()}
    return out
