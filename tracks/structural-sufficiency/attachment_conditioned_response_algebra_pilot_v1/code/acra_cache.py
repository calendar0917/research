"""CPU preprocessing: directed events, local balls, rooted frames, relation data.

The cache is **label free**.  It stores
  * raw atom attributes per molecule,
  * every directed connection event with its S/C local scalars,
  * the deduplicated rooted frame multiset of S and of C (unique pattern +
    multiplicity), with the global pattern table,
  * exact coloured-graph identities for support statistics,
  * the bond-pair distance / shared-ring relation table.

Labels, splits and the target scaler are never written here.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import acra_common as C
import acra_data as D

PD = C.PD


def cache_path(cfg: Mapping[str, Any]) -> Path:
    return C.data_dir() / str(cfg["cache"]["name"])


class PatternTable:
    """Global table of encoder inputs (raw attribute pattern + size + edges)."""

    def __init__(self) -> None:
        self.index: dict[tuple, int] = {}
        self.k: list[int] = []
        self.bits: list[int] = []
        self.atom: list[bytes] = []
        self.bond: list[bytes] = []

    def add(self, key: tuple) -> int:
        got = self.index.get(key)
        if got is not None:
            return got
        k, bits, atom_bytes, bond_bytes = key
        pid = len(self.k)
        self.index[key] = pid
        self.k.append(int(k))
        self.bits.append(int(bits))
        self.atom.append(atom_bytes)
        self.bond.append(bond_bytes)
        return pid

    def arrays(self) -> dict[str, np.ndarray]:
        n = len(self.k)
        atom = (np.frombuffer(b"".join(self.atom), np.uint8).reshape(n, C.MAX_OBJECT_ATOMS,
                                                                   C.ATOM_COLUMNS).copy()
                if n else np.zeros((0, C.MAX_OBJECT_ATOMS, C.ATOM_COLUMNS), np.uint8))
        bond = (np.frombuffer(b"".join(self.bond), np.uint8).reshape(n, len(C.FRAME_PAIRS),
                                                                   C.BOND_COLUMNS).copy()
                if n else np.zeros((0, len(C.FRAME_PAIRS), C.BOND_COLUMNS), np.uint8))
        return {
            "pat_k": np.asarray(self.k, np.uint8),
            "pat_bits": np.asarray(self.bits, np.uint16),
            "pat_atom": atom,
            "pat_bond": bond,
        }


def build_pack(cfg: Mapping[str, Any], raw: Mapping[str, Any],
               rows: Sequence[int] | None = None,
               log: Callable[[str], None] = print) -> dict[str, Any]:
    """Cache for the given raw rows (default: every row).  Label free."""
    n_total = int(np.asarray(raw["node_ptr"]).shape[0]) - 1
    rows = list(range(n_total)) if rows is None else [int(r) for r in rows]
    mols = [D.raw_molecule(raw, r) for r in rows]
    smiles = [str(x) for x in np.asarray(raw["smiles"])[rows]] if "smiles" in raw else [""] * len(rows)
    return pack_from_molecules(cfg, mols, smiles, PD.ColorSpace.from_raw(raw),
                               str(raw.get("cache_sha256", "")), rows, log)


def pack_from_molecules(cfg: Mapping[str, Any], mols: Sequence[PD.Molecule],
                        smiles: Sequence[str], colors: PD.ColorSpace,
                        raw_sha256: str, row_ids: Sequence[int] | None = None,
                        log: Callable[[str], None] = print) -> dict[str, Any]:
    t_start = time.time()
    rows = [int(r) for r in (row_ids if row_ids is not None else range(len(mols)))]
    luts = PD.build_luts(C.MAX_OBJECT_ATOMS)

    patterns = PatternTable()
    key_index: dict[str, int] = {}
    key_strings: list[str] = []
    combo_index: dict[tuple[int, int, int, int, int], int] = {}
    combo_rows: list[tuple[int, int, int, int, int]] = []

    atom_ptr = [0]
    atom_feat_parts: list[np.ndarray] = []
    mol_n_atoms: list[int] = []
    mol_n_bonds: list[int] = []
    mol_n_components: list[int] = []
    mol_cycle_rank: list[int] = []
    mol_smiles: list[str] = []

    ev_ptr = [0]
    ev_bond_attr: list[np.ndarray] = []
    ev_bond: list[int] = []
    ev_dir: list[int] = []
    ev_bridge: list[int] = []
    ev_in_ring: list[int] = []
    ev_key_s: list[int] = []
    ev_key_c: list[int] = []
    ev_combo: list[int] = []
    ev_scalars_s: list[np.ndarray] = []
    ev_scalars_c: list[np.ndarray] = []

    blk_count: list[int] = []
    blk_key: list[int] = []
    blk_ptr = [0]
    ent_pat: list[int] = []
    ent_mult: list[int] = []

    pair_ptr = [0]
    pair_e: list[int] = []
    pair_f: list[int] = []
    pair_bucket: list[int] = []
    pair_shared: list[int] = []

    memo: dict[str, dict[str, Any]] = {}
    t_ident = 0.0
    t_frames = 0.0
    n_new = 0
    for idx, (row, mol) in enumerate(zip(rows, mols)):
        adj = D.adjacency(mol)
        geo = D.bond_geometry(mol, adj)
        atom_ptr.append(atom_ptr[-1] + mol.n_atoms)
        atom_feat_parts.append(mol.atom_feat.astype(np.uint8, copy=True))
        mol_n_atoms.append(int(mol.n_atoms))
        mol_n_bonds.append(int(mol.bond_u.size))
        mol_n_components.append(int(len(set(PD.connected_components(adj, mol.n_atoms)))))
        mol_cycle_rank.append(int(mol.bond_u.size - mol.n_atoms + mol_n_components[-1]))
        mol_smiles.append(str(smiles[idx]))

        for e in range(int(mol.bond_u.size)):
            for direction in (0, 1):
                (root_s, atoms_s), (root_c, atoms_c) = D.event_balls(mol, adj, e, direction)
                ev_bond_attr.append(mol.bond_feat[e])
                ev_bond.append(int(e))
                ev_dir.append(int(direction))
                ev_bridge.append(int(geo["is_bridge"][e]))
                ev_in_ring.append(int(geo["in_ring"][e]))
                side_ids: list[int] = []
                side_scalars: list[np.ndarray] = []
                side_locs: list[dict[str, Any]] = []
                for side, atoms, root in ((0, atoms_s, root_s), (1, atoms_c, root_c)):
                    t0 = time.time()
                    # identity first (cheap early-out for the expensive part)
                    pairs = D.induced_bonds(mol, atoms)
                    identity = D.local_identity(mol, atoms, root, pairs, colors)
                    t_ident += time.time() - t0
                    got = memo.get(identity)
                    if got is None:
                        t1 = time.time()
                        frames = D.frame_patterns(mol, atoms, root, luts, adj=adj)
                        stats = _stats_from_pairs(mol, atoms, pairs, root)
                        blocks: dict[int, list[tuple[int, int]]] = {}
                        for k in range(1, C.MAX_OBJECT_ATOMS + 1):
                            lst = []
                            for key, mult in sorted(frames[k].items()):
                                lst.append((patterns.add(key), int(mult)))
                            blocks[k] = lst
                        got = {"blocks": blocks, "scalars": stats}
                        memo[identity] = got
                        n_new += 1
                        t_frames += time.time() - t1
                    key_id = key_index.get(identity)
                    if key_id is None:
                        key_id = len(key_strings)
                        key_index[identity] = key_id
                        key_strings.append(identity)
                    side_ids.append(key_id)
                    side_scalars.append(got["scalars"])
                    side_locs.append(got)
                    if side == 0:
                        ev_key_s.append(key_id)
                        ev_scalars_s.append(got["scalars"])
                    else:
                        ev_key_c.append(key_id)
                        ev_scalars_c.append(got["scalars"])
                bond_key = tuple(int(x) for x in mol.bond_feat[e])
                combo_key = (int(side_ids[0]), int(side_ids[1]), *bond_key)
                cid = combo_index.get(combo_key)
                if cid is None:
                    cid = len(combo_rows)
                    combo_index[combo_key] = cid
                    combo_rows.append(combo_key)
                ev_combo.append(cid)

                for side in (0, 1):
                    loc = side_locs[side]
                    blk_key.append(int(side_ids[side]))
                    for k in range(1, C.MAX_OBJECT_ATOMS + 1):
                        lst = loc["blocks"][k]
                        cnt = sum(m for _p, m in lst)
                        blk_count.append(int(cnt))
                        for pid, mult in lst:
                            ent_pat.append(int(pid))
                            ent_mult.append(int(mult))
                        blk_ptr.append(len(ent_pat))
        ev_ptr.append(len(ev_combo))

        pair_ptr.append(pair_ptr[-1] + int(geo["pair_e"].size))
        pair_e.extend(int(x) for x in geo["pair_e"].tolist())
        pair_f.extend(int(x) for x in geo["pair_f"].tolist())
        pair_bucket.extend(int(x) for x in geo["pair_bucket"].tolist())
        pair_shared.extend(int(x) for x in geo["pair_shared_ring"].tolist())

        if (idx + 1) % 500 == 0 or idx + 1 == len(rows):
            log(f"  cache: {idx + 1}/{len(rows)} molecules, events={len(ev_combo)}, "
                f"patterns={len(patterns.k)}, local_ids={len(key_strings)}, "
                f"entries={len(ent_pat)}")

    pack: dict[str, Any] = {
        "n_molecules": np.asarray(len(rows), np.int32),
        "row_ids": np.asarray(rows, np.int64),
        "smiles": np.asarray(mol_smiles),
        "atom_ptr": np.asarray(atom_ptr, np.int64),
        "atom_feat": np.concatenate(atom_feat_parts, axis=0) if atom_feat_parts
        else np.zeros((0, C.ATOM_COLUMNS), np.uint8),
        "mol_n_atoms": np.asarray(mol_n_atoms, np.int32),
        "mol_n_bonds": np.asarray(mol_n_bonds, np.int32),
        "mol_n_components": np.asarray(mol_n_components, np.int32),
        "mol_cycle_rank": np.asarray(mol_cycle_rank, np.int32),
        "ev_ptr": np.asarray(ev_ptr, np.int64),
        "ev_bond_attr": np.asarray(ev_bond_attr, np.uint8).reshape(-1, C.BOND_COLUMNS),
        "ev_bond": np.asarray(ev_bond, np.int32),
        "ev_dir": np.asarray(ev_dir, np.uint8),
        "ev_bridge": np.asarray(ev_bridge, np.uint8),
        "ev_in_ring": np.asarray(ev_in_ring, np.uint8),
        "ev_key_s": np.asarray(ev_key_s, np.int32),
        "ev_key_c": np.asarray(ev_key_c, np.int32),
        "ev_combo": np.asarray(ev_combo, np.int32),
        "ev_scalars_s": np.asarray(ev_scalars_s, np.float32).reshape(-1, D.N_LOCAL_SCALARS),
        "ev_scalars_c": np.asarray(ev_scalars_c, np.float32).reshape(-1, D.N_LOCAL_SCALARS),
        "key_strings": np.asarray(key_strings, dtype=str) if key_strings else np.asarray([""]),
        "combo_s": np.asarray([c[0] for c in combo_rows], np.int32),
        "combo_c": np.asarray([c[1] for c in combo_rows], np.int32),
        "combo_bond": np.asarray([c[2:] for c in combo_rows], np.uint8).reshape(-1, C.BOND_COLUMNS),
        "blk_ptr": np.asarray(blk_ptr, np.int64),
        "blk_count": np.asarray(blk_count, np.int32),
        "blk_key": np.asarray(blk_key, np.int32),
        "ent_pat": np.asarray(ent_pat, np.int32),
        "ent_mult": np.asarray(ent_mult, np.uint8),
        "pair_ptr": np.asarray(pair_ptr, np.int64),
        "pair_e": np.asarray(pair_e, np.int32),
        "pair_f": np.asarray(pair_f, np.int32),
        "pair_bucket": np.asarray(pair_bucket, np.uint8),
        "pair_shared_ring": np.asarray(pair_shared, np.uint8),
    }
    pack.update(patterns.arrays())
    missing = [name for name, arr in pack.items()
               if isinstance(arr, np.ndarray) and arr.dtype == object]
    if missing:
        raise AssertionError(f"object dtype arrays not allowed: {missing}")
    if int(pack["blk_ptr"][-1]) != int(pack["ent_pat"].size):
        raise AssertionError("block pointers do not cover all entries")
    if int(pack["ent_mult"].max(initial=0)) > 255:
        raise AssertionError("multiplicity exceeds uint8")
    if np.any(pack["blk_count"] < 0):
        raise AssertionError("negative block count")
    pack["meta_json"] = np.asarray(_meta_json(cfg, raw_sha256, pack, rows, t_start,
                                              t_ident, t_frames, n_new))
    return pack


def _stats_from_pairs(mol: PD.Molecule, atoms: Sequence[int],
                      pairs: Sequence[tuple[int, int, np.ndarray]], root: int) -> np.ndarray:
    pos = {int(a): i for i, a in enumerate(atoms)}
    local_adj: list[list[int]] = [[] for _ in atoms]
    for i, j, _b in pairs:
        local_adj[i].append(j)
        local_adj[j].append(i)
    return D.local_graph_stats(len(atoms), len(pairs), local_adj, pos[int(root)])


def _meta_json(cfg: Mapping[str, Any], raw_sha256: str, pack: Mapping[str, Any],
               rows: Sequence[int], t_start: float, t_ident: float, t_frames: float,
               n_new: int) -> str:
    import json

    meta = {
        "protocol_id": cfg["protocol_id"],
        "cache_version": cfg["cache"]["version"],
        "built_utc": C.now_utc(),
        "raw_cache_sha256": str(raw_sha256),
        "n_molecules": int(pack["n_molecules"]),
        "n_events": int(pack["ev_key_s"].size),
        "n_bonds": int(pack["mol_n_bonds"].sum()),
        "n_local_blocks": int(pack["blk_count"].size),
        "n_frame_entries": int(pack["ent_pat"].size),
        "n_frames_with_multiplicity": int(pack["blk_count"].sum()),
        "n_frame_patterns": int(pack["pat_k"].size),
        "n_local_identities": int(pack["key_strings"].size),
        "n_combo_identities": int(pack["combo_s"].size),
        "n_bond_pairs": int(pack["pair_e"].size),
        "unique_frame_compression": float(pack["ent_pat"].size)
        / max(1.0, float(pack["blk_count"].sum())),
        "distinct_local_identities": int(n_new),
        "timing_seconds": {
            "total": float(time.time() - t_start),
            "identity": float(t_ident),
            "frame_expansion": float(t_frames),
        },
        "row_range": [int(rows[0]), int(rows[-1])] if rows else [],
        "label_free": True,
    }
    return json.dumps(meta, sort_keys=True)


def save_pack(pack: Mapping[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in pack.items()})
    return C.sha256_file(path)


def load_pack(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as handle:
        pack = {k: handle[k] for k in handle.files}
    pack["cache_sha256"] = C.sha256_file(path)
    pack["cache_path"] = str(path)
    return pack


def ensure_pack(cfg: Mapping[str, Any], log: Callable[[str], None] = print) -> dict[str, Any]:
    path = cache_path(cfg)
    if not path.is_file():
        raw = D.load_raw(cfg)
        log(f"  cache missing; building {path}")
        pack = build_pack(cfg, raw, log=log)
        digest = save_pack(pack, path)
        log(f"  wrote {path} sha256={digest}")
        return load_pack(path)
    return load_pack(path)