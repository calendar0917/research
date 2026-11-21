"""Build the explicit structural objects, exact cuts, fragment CSR and replacement
directory for one dataset.

Two phases so that identical fragment content is computed once:

1. walk the molecules, enumerate rooted 1-4 objects with all root-fixed frames,
   dedup frames by their *complete* attribute record, and register every cut
   side ``(S, p_S)`` / ``(C, p_C)`` under its exact coloured-graph key;
2. for every unique fragment key, use the representative molecule's object
   universe, keep the objects contained in the fragment, bucket them by the
   **within-fragment** BFS distance root -> port, and emit the compressed
   ``fragment x distance-bucket x frame`` weighted counts.

Nothing here reads labels for structure, keys, buckets or the directory.
"""

from __future__ import annotations

import itertools
import json
import time
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import trc_common as C
import trc_data as D

CELL_TRAIN = 0
CELL_VALID = 1
CELL_TEST = 2
CELL_CROSS = 3
CELL_NAMES = {CELL_TRAIN: "train", CELL_VALID: "valid", CELL_TEST: "test", CELL_CROSS: "cross"}


# ---------------------------------------------------------------------------
# frame dedup table
# ---------------------------------------------------------------------------


class FrameTable:
    """Exact dedup of complete attribute frames (k, 4 atom rows, 6 bond rows)."""

    def __init__(self) -> None:
        self.index: dict[bytes, int] = {}
        self.atom: list[np.ndarray] = []
        self.bond: list[np.ndarray] = []
        self.k: list[int] = []

    def add_many(self, atom_rows: np.ndarray, bond_rows: np.ndarray, frame_k: np.ndarray) -> np.ndarray:
        out = np.empty(atom_rows.shape[0], dtype=np.int32)
        for i in range(atom_rows.shape[0]):
            key = (int(frame_k[i]).to_bytes(1, "little")
                   + atom_rows[i].astype(np.int32).tobytes()
                   + (bond_rows[i] + 1).astype(np.int32).tobytes())
            got = self.index.get(key)
            if got is None:
                got = len(self.atom)
                self.index[key] = got
                self.atom.append(np.asarray(atom_rows[i], np.int32))
                self.bond.append(np.asarray(bond_rows[i], np.int32))
                self.k.append(int(frame_k[i]))
            out[i] = got
        return out

    def finish(self) -> dict[str, np.ndarray]:
        return {
            "frame_atom": np.asarray(self.atom, np.int32).reshape(-1, C.MAX_OBJECT_ATOMS),
            "frame_bond": np.asarray(self.bond, np.int32).reshape(-1, len(C.FRAME_PAIRS)),
            "frame_k": np.asarray(self.k, np.uint8),
        }


# ---------------------------------------------------------------------------
# molecule-level object universe
# ---------------------------------------------------------------------------


class MoleculeUniverse:
    """Rooted 1-4 objects of one molecule plus their frame ids."""

    __slots__ = ("row", "n_atoms", "obj_atoms", "obj_root", "obj_frame_ptr",
                 "obj_frame_id", "obj_n_frames", "atom_row", "bond_row")

    def __init__(self, **kw: Any) -> None:
        for name in self.__slots__:
            setattr(self, name, kw[name])


def molecule_universe(mol: D.Molecule, atom_row: np.ndarray, bond_row: np.ndarray,
                      luts: Mapping[int, Mapping[str, np.ndarray]],
                      frames: FrameTable) -> MoleculeUniverse:
    objs = D.rooted_subsets(mol.adjacency_lists(), mol.n_atoms, C.MAX_OBJECT_ATOMS)
    fs = D.frames_for_objects(objs, atom_row, bond_row, luts)
    frame_ids = frames.add_many(fs.atom_rows, fs.bond_rows, fs.frame_k)
    n_obj = len(objs)
    obj_atoms = np.full((n_obj, C.MAX_OBJECT_ATOMS), -1, np.int32)
    obj_root = np.empty(n_obj, np.int32)
    for i, (root, atoms) in enumerate(objs):
        obj_root[i] = int(root)
        obj_atoms[i, :len(atoms)] = atoms
    n_frames = np.diff(fs.obj_frame_ptr)
    return MoleculeUniverse(
        row=int(mol.row_id), n_atoms=int(mol.n_atoms), obj_atoms=obj_atoms,
        obj_root=obj_root, obj_frame_ptr=fs.obj_frame_ptr, obj_frame_id=frame_ids,
        obj_n_frames=n_frames, atom_row=np.asarray(atom_row), bond_row=np.asarray(bond_row),
    )


# ---------------------------------------------------------------------------
# fragment key registration
# ---------------------------------------------------------------------------


def _local_fragment(mol: D.Molecule, atoms: Sequence[int], port: int,
                    bond_index: Mapping[tuple[int, int], int],
                    atom_rows: np.ndarray, bond_rows: np.ndarray) -> tuple[
                        list[np.ndarray], list[tuple[int, int, np.ndarray]], int]:
    order = sorted(int(a) for a in atoms)
    local = {a: i for i, a in enumerate(order)}
    atom_feats = [mol.atom_feat[a] for a in order]
    local_bonds: list[tuple[int, int, np.ndarray]] = []
    for a, b in itertools.combinations(order, 2):
        e = bond_index.get((a, b))
        if e is None:
            continue
        local_bonds.append((local[a], local[b], mol.bond_feat[e]))
    local_bonds.sort(key=lambda t: (t[0], t[1]))
    return atom_feats, local_bonds, local[port]


class FragmentRegistry:
    def __init__(self, colors: D.ColorSpace) -> None:
        self.colors = colors
        self.key_id: dict[str, int] = {}
        self.keys: list[str] = []
        self.rep_row: list[int] = []
        self.rep_atoms: list[tuple[int, ...]] = []
        self.rep_port: list[int] = []
        self.rep_natoms: list[int] = []
        self.rep_nbonds: list[int] = []
        self.rep_ncomp: list[int] = []

    def add(self, key: str, row: int, atoms: Sequence[int], port: int,
            n_bonds: int, n_comp: int) -> int:
        got = self.key_id.get(key)
        if got is not None:
            return got
        got = len(self.keys)
        self.key_id[key] = got
        self.keys.append(key)
        self.rep_row.append(int(row))
        self.rep_atoms.append(tuple(sorted(int(a) for a in atoms)))
        self.rep_port.append(int(port))
        self.rep_natoms.append(len(atoms))
        self.rep_nbonds.append(int(n_bonds))
        self.rep_ncomp.append(int(n_comp))
        return got


def _component_count(n_atoms: int, local_bonds: Sequence[tuple[int, int, Any]]) -> int:
    parent = list(range(n_atoms))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j, _b) in local_bonds:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
    return len({find(i) for i in range(n_atoms)})


# ---------------------------------------------------------------------------
# main build
# ---------------------------------------------------------------------------


def build_dataset(cfg: Mapping[str, Any], dataset: str, log=print) -> dict[str, Any]:
    t_start = time.perf_counter()
    raw = D.load_raw_cache(cfg, dataset)
    n_mol = int(raw["row_ids"].size)
    max_side = int(cfg["cuts"]["max_side_atoms"])
    max_k = int(cfg["cuts"]["max_object_atoms"])

    colors = D.ColorSpace()
    atoms_table = _AttrTable(C.ATOM_COLUMNS)
    bonds_table = _AttrTable(C.BOND_COLUMNS)
    luts = D.build_luts(max_k)
    frames = FrameTable()
    registry = FragmentRegistry(colors)

    # per molecule bookkeeping
    universes: list[MoleculeUniverse | None] = [None] * n_mol
    parse_rows: list[np.ndarray] = []
    parse_S: list[np.ndarray] = []
    parse_C: list[np.ndarray] = []
    parse_bond: list[np.ndarray] = []
    parse_kind: list[np.ndarray] = []
    parse_bidx: list[np.ndarray] = []
    parse_portS: list[np.ndarray] = []
    parse_portC: list[np.ndarray] = []
    parse_S_atoms: list[np.ndarray] = []
    parse_C_atoms: list[np.ndarray] = []
    parse_cut_u: list[np.ndarray] = []
    parse_cut_v: list[np.ndarray] = []
    fallback_mols: list[int] = []

    t0 = time.perf_counter()
    for row in range(n_mol):
        mol = D.raw_molecule(raw, row)
        adj = mol.adjacency_lists()
        bindex = mol.bond_index()
        atom_row = np.asarray([atoms_table.index_of(x) for x in mol.atom_feat], np.int32)
        bond_row = np.full((mol.n_atoms, mol.n_atoms), -1, np.int32)
        for e, (a, b) in enumerate(zip(mol.bond_u.tolist(), mol.bond_v.tolist())):
            bond_row[a, b] = bond_row[b, a] = bonds_table.index_of(mol.bond_feat[e])
        universes[row] = molecule_universe(mol, atom_row, bond_row, luts, frames)

        parses, n_bridge = D.cut_parses(mol, adj, max_side=max_side)
        if not parses:
            fallback_mols.append(row)
            parses = D.fallback_parses(mol)
        n = len(parses)
        s_ids = np.empty(n, np.int32)
        c_ids = np.empty(n, np.int32)
        b_rows = np.full(n, -1, np.int32)
        kinds = np.zeros(n, np.uint8)
        bidx = np.full(n, -1, np.int32)
        p_s = np.empty(n, np.int32)
        p_c = np.full(n, -1, np.int32)
        for i, p in enumerate(parses):
            af, lb, lp = _local_fragment(mol, p.atoms_s, p.port_s, bindex, atom_row, bond_row)
            key = D.canonical_key(af, lb, lp, colors)
            s_ids[i] = registry.add(key, row, p.atoms_s, p.port_s, len(lb),
                                    _component_count(len(af), lb))
            p_s[i] = p.port_s
            if p.bond_index is None:
                kinds[i] = 1
                c_ids[i] = -1
            else:
                af_c, lb_c, lp_c = _local_fragment(mol, p.atoms_c, p.port_c, bindex,
                                                   atom_row, bond_row)
                key_c = D.canonical_key(af_c, lb_c, lp_c, colors)
                c_ids[i] = registry.add(key_c, row, p.atoms_c, p.port_c, len(lb_c),
                                        _component_count(len(af_c), lb_c))
                b_rows[i] = bond_row[int(mol.bond_u[p.bond_index]),
                                     int(mol.bond_v[p.bond_index])]
                bidx[i] = int(p.bond_index)
                p_c[i] = int(p.port_c)
        parse_rows.append(np.full(n, row, np.int32))
        parse_S.append(s_ids)
        parse_C.append(c_ids)
        parse_bond.append(b_rows)
        parse_kind.append(kinds)
        parse_bidx.append(bidx)
        parse_portS.append(p_s)
        parse_portC.append(p_c)
        parse_S_atoms.append(np.concatenate([np.asarray(p.atoms_s, np.int32) for p in parses]))
        parse_C_atoms.append(np.concatenate([np.asarray(p.atoms_c, np.int32) for p in parses]))
        cu = np.asarray([-1 if p.bond_index is None else int(mol.bond_u[p.bond_index]) for p in parses], np.int32)
        cv = np.asarray([-1 if p.bond_index is None else int(mol.bond_v[p.bond_index]) for p in parses], np.int32)
        parse_cut_u.append(cu)
        parse_cut_v.append(cv)
        if row % 500 == 0:
            log(f"  molecule {row}/{n_mol} frags={len(registry.keys)} frames={len(frames.k)} "
                f"parses={sum(x.size for x in parse_S)}")
    t_walk = time.perf_counter() - t0
    log(f"  walk {t_walk:.1f}s: {sum(x.size for x in parse_S)} parses, "
        f"{len(registry.keys)} unique fragments, {len(frames.k)} unique frames")

    # ---- phase 1b: molecule-level size-bucket CSR for GLOBAL --------------
    t0 = time.perf_counter()
    mol_ptr = np.zeros(n_mol + 1, np.int64)
    mol_bucket_parts: list[np.ndarray] = []
    mol_frame_parts: list[np.ndarray] = []
    mol_weight_parts: list[np.ndarray] = []
    mol_counts = np.zeros((n_mol, C.MAX_OBJECT_ATOMS), np.int32)
    mol_meta = np.zeros((n_mol, 3), np.int32)
    for row in range(n_mol):
        uni = universes[row]
        sizes = np.zeros(uni.obj_atoms.shape[0], np.int32)
        for k in range(1, C.MAX_OBJECT_ATOMS + 1):
            sizes[(uni.obj_atoms >= 0).sum(1) == k] = k
        counts = np.bincount(sizes - 1, minlength=C.MAX_OBJECT_ATOMS)[:C.MAX_OBJECT_ATOMS]
        mol_counts[row] = counts
        fptr = uni.obj_frame_ptr
        obj_counts = fptr[1:] - fptr[:-1]
        total = int(obj_counts.sum())
        b_exp = np.repeat(sizes - 1, obj_counts)
        starts = np.cumsum(obj_counts) - obj_counts
        local = np.arange(total, dtype=np.int64) - np.repeat(starts, obj_counts)
        fids = uni.obj_frame_id[np.repeat(np.arange(uni.obj_atoms.shape[0]), obj_counts) + local]
        w = 1.0 / np.repeat(obj_counts, obj_counts).astype(np.float64)
        if total:
            code = b_exp.astype(np.int64) * (len(frames.k) + 1) + fids
            uniq, inv = np.unique(code, return_inverse=True)
            wsum = np.bincount(inv, weights=w, minlength=uniq.size)
            b_out = (uniq // (len(frames.k) + 1)).astype(np.uint8)
            f_out = (uniq % (len(frames.k) + 1)).astype(np.int32)
        else:
            b_out = np.zeros(0, np.uint8); f_out = np.zeros(0, np.int32); wsum = np.zeros(0)
        mol_bucket_parts.append(b_out)
        mol_frame_parts.append(f_out)
        mol_weight_parts.append(wsum.astype(np.float32))
        mol_ptr[row + 1] = b_out.size
        mol_meta[row, 0] = uni.n_atoms
        mol_meta[row, 1] = int((uni.bond_row >= 0).sum() // 2)
        mol_meta[row, 2] = int(np.asarray(raw["graph_n_components"])[row])
    mol_ptr = np.concatenate([[0], np.cumsum(mol_ptr[1:])]).astype(np.int64)
    log(f"  molecule csr {time.perf_counter() - t0:.1f}s nnz={mol_ptr[-1]}")

    # ---- phase 2: fragment CSR -------------------------------------------
    t0 = time.perf_counter()
    n_frag = len(registry.keys)
    frag_ptr = np.zeros(n_frag + 1, np.int64)
    bucket_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    frag_nbonds = np.asarray(registry.rep_nbonds, np.int32)
    frag_ncomp = np.asarray(registry.rep_ncomp, np.int32)
    frag_natoms = np.asarray(registry.rep_natoms, np.int32)
    frag_counts = np.zeros((n_frag, C.N_BUCKETS), np.int32)
    frag_port = np.asarray(registry.rep_port, np.int32)
    # group fragments by representative molecule to reuse the local trees
    order = np.argsort(np.asarray(registry.rep_row), kind="stable")
    pos = 0
    filled = np.zeros(n_frag, bool)
    while pos < n_frag:
        row = int(registry.rep_row[order[pos]])
        stop = pos
        while stop < n_frag and int(registry.rep_row[order[stop]]) == row:
            stop += 1
        fid_list = order[pos:stop]
        uni = universes[row]
        mol_atoms = uni.n_atoms
        for fid in fid_list:
            atoms = registry.rep_atoms[fid]
            port = registry.rep_port[fid]
            in_f = np.zeros(mol_atoms, bool)
            in_f[list(atoms)] = True
            sub = uni.obj_atoms[:, :C.MAX_OBJECT_ATOMS]
            keep = np.all((sub < 0) | in_f[np.maximum(sub, 0)], axis=1)
            roots = uni.obj_root[keep]
            # within-fragment BFS from the port
            dist = _bfs_within(mol_atoms, atoms, uni.bond_row, port)
            buckets = np.full(roots.size, 5, np.uint8)
            filled_d = dist[roots] >= 0
            dvals = dist[roots[filled_d]]
            buckets[filled_d] = np.where(dvals <= 3, dvals, 4).astype(np.uint8)
            # expand frames of the kept objects
            fptr = uni.obj_frame_ptr
            counts = fptr[np.nonzero(keep)[0] + 1] - fptr[np.nonzero(keep)[0]]
            obj_keep = np.nonzero(keep)[0]
            total = int(counts.sum())
            b_exp = np.repeat(buckets, counts) if total else np.zeros(0, np.uint8)
            starts = np.cumsum(counts) - counts
            local = np.arange(total, dtype=np.int64) - np.repeat(starts, counts) if total else np.zeros(0, np.int64)
            fids = uni.obj_frame_id[np.repeat(obj_keep, counts) + local] if total else np.zeros(0, np.int32)
            nfr = uni.obj_n_frames[obj_keep]
            w = 1.0 / np.repeat(nfr, counts).astype(np.float64) if total else np.zeros(0)
            if total:
                code = b_exp.astype(np.int64) * (len(frames.k) + 1) + fids
                uniq, inv = np.unique(code, return_inverse=True)
                wsum = np.bincount(inv, weights=w, minlength=uniq.size)
                b_out = (uniq // (len(frames.k) + 1)).astype(np.uint8)
                f_out = (uniq % (len(frames.k) + 1)).astype(np.int32)
            else:
                b_out = np.zeros(0, np.uint8)
                f_out = np.zeros(0, np.int32)
                wsum = np.zeros(0)
            frag_counts[fid] = np.bincount(buckets, minlength=C.N_BUCKETS)[:C.N_BUCKETS]
            bucket_parts.append(b_out)
            frame_parts.append(f_out)
            weight_parts.append(wsum.astype(np.float32))
            frag_ptr[fid + 1] = b_out.size
            filled[fid] = True
        pos = stop
    if not filled.all():
        raise AssertionError("fragment CSR left unbuilt rows")
    frag_ptr = np.concatenate([[0], np.cumsum(frag_ptr[1:])]).astype(np.int64)
    frag_bucket = np.concatenate(bucket_parts) if bucket_parts else np.zeros(0, np.uint8)
    frag_frame = np.concatenate(frame_parts) if frame_parts else np.zeros(0, np.int32)
    frag_weight = np.concatenate(weight_parts) if weight_parts else np.zeros(0, np.float32)
    t_csr = time.perf_counter() - t0
    log(f"  csr {t_csr:.1f}s: nnz={frag_frame.size} "
        f"({frag_frame.size / max(1, n_mol):.1f} per molecule)")

    frame_pack = frames.finish()
    parse_S_all = np.concatenate(parse_S)
    parse_C_all = np.concatenate(parse_C)
    parse_mol_all = np.concatenate(parse_rows)
    parse_bond_all = np.concatenate(parse_bond)
    parse_kind_all = np.concatenate(parse_kind)
    parse_bidx_all = np.concatenate(parse_bidx)
    parse_portS_all = np.concatenate(parse_portS)
    parse_portC_all = np.concatenate(parse_portC)
    mol_parse_ptr = np.concatenate(
        [[0], np.cumsum([x.size for x in parse_S])]).astype(np.int64)
    s_atoms_flat = np.concatenate(parse_S_atoms) if parse_S_atoms else np.zeros(0, np.int32)
    c_atoms_flat = np.concatenate(parse_C_atoms) if parse_C_atoms else np.zeros(0, np.int32)
    parse_S_atoms_ptr = np.concatenate(
        [[0], np.cumsum([x.size for x in parse_S_atoms])]).astype(np.int64)
    parse_C_atoms_ptr = np.concatenate(
        [[0], np.cumsum([x.size for x in parse_C_atoms])]).astype(np.int64)

    out = {
        "dataset": np.asarray(dataset),
        "n_molecules": np.asarray(n_mol, np.int32),
        "n_fragments": np.asarray(n_frag, np.int32),
        "n_frames": np.asarray(len(frames.k), np.int32),
        "n_parses": np.asarray(parse_S_all.size, np.int32),
        "mol_parse_ptr": mol_parse_ptr.astype(np.int32),
        "parse_mol": parse_mol_all.astype(np.int32),
        "parse_S": parse_S_all,
        "parse_C": parse_C_all,
        "parse_cut_bond_row": parse_bond_all,
        "parse_kind": parse_kind_all,
        "parse_bond_index": parse_bidx_all,
        "parse_port_S": parse_portS_all,
        "parse_port_C": parse_portC_all,
        "parse_cut_u": np.concatenate(parse_cut_u) if parse_cut_u else np.zeros(0, np.int32),
        "parse_cut_v": np.concatenate(parse_cut_v) if parse_cut_v else np.zeros(0, np.int32),
        "parse_S_atoms": s_atoms_flat,
        "parse_S_atoms_ptr": parse_S_atoms_ptr.astype(np.int64),
        "parse_C_atoms": c_atoms_flat,
        "parse_C_atoms_ptr": parse_C_atoms_ptr.astype(np.int64),
        "frag_ptr": frag_ptr.astype(np.int64),
        "frag_bucket": frag_bucket,
        "frag_frame": frag_frame,
        "frag_weight": frag_weight,
        "frag_natoms": frag_natoms,
        "frag_counts": frag_counts,
        "frag_nbonds": frag_nbonds,
        "frag_ncomp": frag_ncomp,
        "frag_port": frag_port,
        "atom_table": atoms_table.matrix(),
        "bond_table": bonds_table.matrix(),
        "row_ids": np.asarray(raw["row_ids"], np.int64),
        "labels": np.asarray(raw["labels"], np.float64),
        "train_idx": np.asarray(raw["train_idx"], np.int64),
        "valid_idx": np.asarray(raw["valid_idx"], np.int64),
        "test_idx": np.asarray(raw["test_idx"], np.int64),
        "graph_n_components": np.asarray(raw["graph_n_components"], np.int32),
        "mol_ptr": mol_ptr.astype(np.int64),
        "mol_bucket": np.concatenate(mol_bucket_parts) if mol_bucket_parts else np.zeros(0, np.uint8),
        "mol_frame": np.concatenate(mol_frame_parts) if mol_frame_parts else np.zeros(0, np.int32),
        "mol_weight": np.concatenate(mol_weight_parts) if mol_weight_parts else np.zeros(0, np.float32),
        "mol_counts": mol_counts,
        "mol_meta": mol_meta,
    }
    out.update(frame_pack)

    # ---- replacement directory ------------------------------------------
    t0 = time.perf_counter()
    pairs, pair_stats = build_pairs(cfg, out)
    out.update(pairs)
    t_pairs = time.perf_counter() - t0
    log(f"  pairs {t_pairs:.1f}s: {pair_stats['n_pairs']} pairs, "
        f"{pair_stats['n_transforms']} transforms")

    out["fragment_keys"] = np.asarray(registry.keys)
    out["timing_json"] = np.asarray(json.dumps({
        "walk_seconds": t_walk, "csr_seconds": t_csr, "pairs_seconds": t_pairs,
        "total_seconds": time.perf_counter() - t_start,
    }, sort_keys=True))
    out["fallback_rows"] = np.asarray(fallback_mols, np.int32)
    out["pair_stats_json"] = np.asarray(json.dumps(pair_stats, sort_keys=True))
    return out


def _bfs_within(n_atoms: int, atoms: Sequence[int], bond_row: np.ndarray, port: int) -> np.ndarray:
    dist = np.full(n_atoms, -1, np.int32)
    allowed = np.zeros(n_atoms, bool)
    allowed[list(atoms)] = True
    dist[port] = 0
    frontier = np.asarray([port], np.int64)
    step = 0
    while frontier.size:
        step += 1
        nb = np.nonzero((bond_row[frontier] >= 0) & (dist == -1)[None, :])[1]
        # keep only allowed
        nb = nb[allowed[nb]]
        nb = np.unique(nb)
        if nb.size == 0:
            break
        dist[nb] = step
        frontier = nb
    return dist


class _AttrTable:
    """Exact dedup of raw attribute rows (gather indices only)."""

    def __init__(self, width: int) -> None:
        self.width = int(width)
        self.index: dict[bytes, int] = {}
        self.rows: list[bytes] = []

    def index_of(self, row: np.ndarray) -> int:
        key = np.asarray(row, dtype=np.uint8).tobytes()
        got = self.index.get(key)
        if got is None:
            got = len(self.rows)
            self.index[key] = got
            self.rows.append(key)
        return got

    def matrix(self) -> np.ndarray:
        if not self.rows:
            return np.zeros((0, self.width), np.uint8)
        nbytes = self.width
        return np.frombuffer(b"".join(self.rows), np.uint8).reshape(-1, nbytes).copy()


# ---------------------------------------------------------------------------
# replacement directory
# ---------------------------------------------------------------------------


def build_pairs(cfg: Mapping[str, Any], pack: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Group real parses by exact context key; pair distinct observed words.

    A pair is a *real* pair of molecules observed under the identical context
    key ``(canon(C, p_C), cut bond fields)`` with two different word keys.  The
    canonical direction fixes ``Sa = min(word keys)``, so molecule ``a`` is the
    molecule that carries ``Sa``.
    """
    parse_mol = np.asarray(pack["parse_mol"], np.int64)
    parse_S = np.asarray(pack["parse_S"], np.int64)
    parse_C = np.asarray(pack["parse_C"], np.int64)
    parse_bond = np.asarray(pack["parse_cut_bond_row"], np.int64)
    parse_kind = np.asarray(pack["parse_kind"], np.uint8)
    train_idx = np.asarray(pack["train_idx"], np.int64)
    valid_idx = np.asarray(pack["valid_idx"], np.int64)
    test_idx = np.asarray(pack["test_idx"], np.int64)
    n_mol = int(pack["n_molecules"])
    split_of = np.full(n_mol, CELL_CROSS, np.uint8)
    split_of[train_idx] = CELL_TRAIN
    split_of[valid_idx] = CELL_VALID
    split_of[test_idx] = CELL_TEST

    real = np.nonzero(parse_kind == 0)[0]
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for p in real.tolist():
        groups[(int(parse_C[p]), int(parse_bond[p]))].append(p)

    seen: dict[tuple[int, int, int, int, int], int] = {}
    rec = {k: [] for k in ("a", "b", "C", "Sa", "Sb", "bond")}
    n_group_multi = 0
    n_duplicate_word_rows = 0
    for (ctx_c, ctx_b), plist in groups.items():
        words: dict[int, list[int]] = defaultdict(list)
        for p in plist:
            words[int(parse_S[p])].append(p)
        if len(words) < 2:
            continue
        n_group_multi += 1
        n_duplicate_word_rows += sum(len(v) - 1 for v in words.values() if len(v) > 1)
        word_keys = sorted(words)
        for wa, wb in itertools.combinations(word_keys, 2):
            for pa in words[wa]:
                for pb in words[wb]:
                    ma, mb = int(parse_mol[pa]), int(parse_mol[pb])
                    if ma == mb:
                        continue
                    key = (min(ma, mb), max(ma, mb), int(ctx_c), int(wa), int(wb))
                    if key in seen:
                        continue
                    seen[key] = 1
                    rec["a"].append(ma)
                    rec["b"].append(mb)
                    rec["C"].append(int(ctx_c))
                    rec["Sa"].append(int(wa))
                    rec["Sb"].append(int(wb))
                    rec["bond"].append(int(ctx_b))

    a = np.asarray(rec["a"], np.int64)
    b = np.asarray(rec["b"], np.int64)
    cells = np.asarray([CELL_TRAIN if split_of[x] == CELL_TRAIN and split_of[y] == CELL_TRAIN
                        else CELL_VALID if split_of[x] == CELL_VALID and split_of[y] == CELL_VALID
                        else CELL_TEST if split_of[x] == CELL_TEST and split_of[y] == CELL_TEST
                        else CELL_CROSS
                        for x, y in zip(a.tolist(), b.tolist())], np.uint8)
    tkeys = (np.stack([np.asarray(rec["Sa"], np.int64), np.asarray(rec["Sb"], np.int64),
                       np.asarray(rec["bond"], np.int64)], 1)
             if a.size else np.zeros((0, 3), np.int64))
    if a.size:
        tuniq, tinv = np.unique(tkeys, axis=0, return_inverse=True)
        tinv = np.asarray(tinv).reshape(-1)
    else:
        tuniq, tinv = np.zeros((0, 3), np.int64), np.zeros(0, np.int64)

    pack_out: dict[str, np.ndarray] = {
        "pair_mol_a": a.astype(np.int32),
        "pair_mol_b": b.astype(np.int32),
        "pair_C": np.asarray(rec["C"], np.int64).astype(np.int32),
        "pair_Sa": np.asarray(rec["Sa"], np.int64).astype(np.int32),
        "pair_Sb": np.asarray(rec["Sb"], np.int64).astype(np.int32),
        "pair_cut_bond_row": np.asarray(rec["bond"], np.int64).astype(np.int32),
        "pair_transform": tinv.astype(np.int32),
        "pair_cell": cells,
        "transform_word_a": tuniq[:, 0].astype(np.int32) if tuniq.size else np.zeros(0, np.int32),
        "transform_word_b": tuniq[:, 1].astype(np.int32) if tuniq.size else np.zeros(0, np.int32),
        "transform_bond_row": tuniq[:, 2].astype(np.int32) if tuniq.size else np.zeros(0, np.int32),
        "molecule_split": split_of,
    }

    train_sel = np.nonzero(cells == CELL_TRAIN)[0]
    comp = _components(n_mol, a[train_sel], b[train_sel])
    pack_out["molecule_component"] = comp.astype(np.int32)
    pair_comp = np.full(a.size, -1, np.int32)
    pair_comp[train_sel] = comp[a[train_sel]]
    pack_out["pair_component"] = pair_comp

    train_mols = set(a[train_sel].tolist()) | set(b[train_sel].tolist())
    stats = {
        "n_pairs": int(a.size),
        "n_transforms": int(tuniq.shape[0]),
        "cells": {CELL_NAMES[c]: int((cells == c).sum()) for c in (0, 1, 2, 3)},
        "n_context_groups_with_two_words": int(n_group_multi),
        "n_duplicate_word_rows": int(n_duplicate_word_rows),
        "n_train_pairs": int(train_sel.size),
        "n_train_components": int(len(set(comp[list(train_mols)].tolist()))) if train_mols else 0,
        "n_molecules_in_train_pairs": int(len(train_mols)),
    }
    return pack_out, stats


def hash_transform(wa: int, wb: int, bond: int) -> int:
    return wa * 2_000_003 + wb * 1009 + bond


def _components(n_mol: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    parent = list(range(n_mol))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for x, y in zip(a.tolist(), b.tolist()):
        rx, ry = find(int(x)), find(int(y))
        if rx != ry:
            parent[rx] = ry
    return np.asarray([find(i) for i in range(n_mol)], np.int32)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def cache_path(cfg: Mapping[str, Any], dataset: str):
    stem = str(cfg.get("cache_stem", "trc_cache"))
    suffix = "lipo" if dataset == "lipo" else dataset
    return C.data_dir() / f"{stem}_{suffix}.npz"


def save_pack(path, pack: Mapping[str, Any]) -> str:
    arrays = {k: v for k, v in pack.items() if isinstance(v, np.ndarray)}
    np.savez(path, **arrays)
    return C.sha256_file(path)


def load_pack(cfg: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    path = cache_path(cfg, dataset)
    if not path.is_file():
        raise FileNotFoundError(f"prepared cache missing: {path}")
    with np.load(path, allow_pickle=False) as handle:
        pack = {k: handle[k] for k in handle.files}
    pack["cache_path"] = str(path)
    pack["cache_sha256"] = C.sha256_file(path)
    return pack


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lipo", choices=["lipo", "esol"])
    ap.add_argument("--stem", default="trc_cache")
    args = ap.parse_args(argv)
    cfg = dict(C.load_config())
    cfg["cache_stem"] = args.stem
    pack = build_dataset(cfg, args.dataset)
    path = cache_path(cfg, args.dataset)
    digest = save_pack(path, pack)
    print(f"wrote {path} sha256={digest} size={path.stat().st_size / 1e6:.1f}MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
