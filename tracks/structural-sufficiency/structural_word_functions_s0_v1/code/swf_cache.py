"""Rebuildable structure cache: full enumeration, dedup tables and batch packing.

Two artefacts are produced from the committed raw label-free npz:

``data/swf_base.npz``
    raw arrays + words + word frames + the *computational* dedup tables for
    ordered word-frame attribute inputs.
``data/swf_sent.npz``
    sentences, covers, frame-pair rows and the computational dedup table for
    relation inputs.

The dedup ids are gather indices only; **no trainable embedding is indexed by
an ordered full-attribute record or a relation record**.  Only the atom / bond
categories and the 16 structural templates have embeddings.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import swf_common as C
import swf_data as D

BASE_FILE = "swf_base.npz"
SENT_FILE = "swf_sent.npz"
STATS_FILE = "swf_stats.json"

WIN_U16 = 11   # template + 4 atom codes + 6 bond codes
REL_U16 = 20   # 4 header bytes + E + C + 16 bond codes


def base_path() -> Path:
    return C.data_dir() / BASE_FILE


def sent_path() -> Path:
    return C.data_dir() / SENT_FILE


def stats_path() -> Path:
    return C.data_dir() / STATS_FILE


def _ranges(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Concatenation of ``arange(starts[i], ends[i])`` (int64)."""
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    counts = ends - starts
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    out = np.repeat(starts - np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
    return out + np.arange(total, dtype=np.int64)


def _local_pos(batch_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Position inside the packed batch of each global id in ``values``.

    A batch is an arbitrary set of molecules, so the packed arrays are a
    concatenation of ascending per-molecule blocks in *batch order*; the global
    id minus a fixed offset is therefore **not** a valid local index.
    """
    batch_ids = np.asarray(batch_ids, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64)
    if values.size == 0:
        return np.zeros(0, np.int64)
    order = np.argsort(batch_ids, kind="stable")
    sorted_ids = batch_ids[order]
    pos = np.searchsorted(sorted_ids, values)
    return order[pos].astype(np.int64)


def _bits_to_matrix(bits: np.ndarray) -> np.ndarray:
    """``uint16`` cell bitmask -> ``[n, 16]`` float32 matrix (bit ``i*4+j``)."""
    bits = np.asarray(bits, dtype=np.int64).reshape(-1)
    return np.stack([((bits >> s) & 1) for s in range(16)], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def build_structures(
    mols: Sequence[D.Molecule],
    codes: D.CodeTables,
    *,
    log_every: int = 0,
) -> dict[str, Any]:
    """Exhaustive word / sentence / cover / frame-pair enumeration for ``mols``."""
    n_mol = len(mols)
    split_of = None
    win_id: dict[bytes, int] = {}
    rel_id: dict[bytes, int] = {}
    win_payload: list[bytes] = []
    rel_payload: list[bytes] = []

    frame_win_parts: list[np.ndarray] = []
    frame_slots_parts: list[np.ndarray] = []
    word_root_parts: list[np.ndarray] = []
    word_n_parts: list[np.ndarray] = []
    word_tmpl_parts: list[np.ndarray] = []
    word_atoms_parts: list[np.ndarray] = []
    word_frame_ptr_parts: list[np.ndarray] = []
    mol_word_ptr = np.zeros(n_mol + 1, np.int64)
    sent_root_parts: list[np.ndarray] = []
    sent_n_parts: list[np.ndarray] = []
    sent_atoms_parts: list[np.ndarray] = []
    sent_ncovers_parts: list[np.ndarray] = []
    cover_sent_parts: list[np.ndarray] = []
    cover_u_parts: list[np.ndarray] = []
    cover_v_parts: list[np.ndarray] = []
    mol_sent_ptr = np.zeros(n_mol + 1, np.int64)
    mol_cover_ptr = np.zeros(n_mol + 1, np.int64)
    row_uv_parts: list[np.ndarray] = []
    row_vu_parts: list[np.ndarray] = []

    t0 = time.perf_counter()
    n_raw_frames = 0
    for row, mol in enumerate(mols):
        st = D.enumerate_molecule(mol, codes)
        nw = int(st.word_root.size)
        nf = int(st.frame_slots.shape[0])
        ns = int(st.sent_root.size)
        nc = int(st.cover_sent.size)
        nr = st.n_rows

        fids = np.empty(nf, np.int32)
        for i, payload in enumerate(st.frame_win_payload):
            got = win_id.get(payload)
            if got is None:
                got = len(win_payload)
                win_id[payload] = got
                win_payload.append(payload)
            fids[i] = got
        rids = np.empty(2 * nr, np.int32)
        for i, payload in enumerate(st.rel_payload):
            got = rel_id.get(payload)
            if got is None:
                got = len(rel_payload)
                rel_id[payload] = got
                rel_payload.append(payload)
            rids[i] = got

        frame_win_parts.append(fids)
        frame_slots_parts.append(st.frame_slots)
        word_root_parts.append(st.word_root)
        word_n_parts.append(st.word_n)
        word_tmpl_parts.append(st.word_tmpl)
        word_atoms_parts.append(st.word_atoms)
        word_frame_ptr_parts.append(st.word_frame_ptr[:-1])
        mol_word_ptr[row + 1] = mol_word_ptr[row] + nw
        sent_root_parts.append(st.sent_root)
        sent_n_parts.append(st.sent_n)
        sent_atoms_parts.append(st.sent_atoms)
        sent_ncovers_parts.append(st.sent_ncovers)
        # cover ids are local to the molecule -> shift to global word / sentence ids
        cover_sent_parts.append(st.cover_sent + mol_sent_ptr[row])
        cover_u_parts.append(st.cover_u + mol_word_ptr[row])
        cover_v_parts.append(st.cover_v + mol_word_ptr[row])
        mol_sent_ptr[row + 1] = mol_sent_ptr[row] + ns
        mol_cover_ptr[row + 1] = mol_cover_ptr[row] + nc
        row_uv_parts.append(rids[0::2])
        row_vu_parts.append(rids[1::2])
        n_raw_frames += nf
        if log_every and (row + 1) % log_every == 0:
            print(
                f"  enumerated {row + 1}/{n_mol} molecules ({time.perf_counter() - t0:.1f}s, "
                f"words={mol_word_ptr[row + 1]}, frames={n_raw_frames} raw / "
                f"{len(win_payload)} unique, rows={sum(p.size for p in row_uv_parts)} raw / "
                f"{len(rel_payload)} unique rels)",
                flush=True,
            )
    elapsed = time.perf_counter() - t0

    frame_counts = np.asarray([p.shape[0] for p in frame_slots_parts], np.int64)
    frame_offsets = np.concatenate(([0], np.cumsum(frame_counts)))[:-1]
    word_frame_ptr = np.concatenate(
        [p + off for p, off in zip(word_frame_ptr_parts, frame_offsets)]
        + [np.asarray([int(frame_counts.sum())], np.int64)]
    )

    cat = lambda parts, shape, dtype: (  # noqa: E731
        np.concatenate(parts, axis=0).astype(dtype) if parts else np.zeros(shape, dtype)
    )
    frame_win = cat(frame_win_parts, (0,), np.int32)
    frame_slots = cat(frame_slots_parts, (0, 4), np.int32)
    word_root = cat(word_root_parts, (0,), np.int32)
    word_n = cat(word_n_parts, (0,), np.uint8)
    word_tmpl = cat(word_tmpl_parts, (0,), np.uint8)
    word_atoms = cat(word_atoms_parts, (0, 4), np.int32)
    sent_root = cat(sent_root_parts, (0,), np.int32)
    sent_n = cat(sent_n_parts, (0,), np.uint8)
    sent_atoms = cat(sent_atoms_parts, (0, 6), np.int32)
    sent_ncovers = cat(sent_ncovers_parts, (0,), np.uint16)
    cover_sent = cat(cover_sent_parts, (0,), np.int32)
    cover_u = cat(cover_u_parts, (0,), np.int32)
    cover_v = cat(cover_v_parts, (0,), np.int32)
    row_rel_uv = cat(row_uv_parts, (0,), np.int32)
    row_rel_vu = cat(row_vu_parts, (0,), np.int32)

    win_mat = np.frombuffer(b"".join(win_payload), np.uint16).reshape(-1, WIN_U16).copy()
    rel_mat = np.frombuffer(b"".join(rel_payload), np.uint16).reshape(-1, REL_U16).copy()
    win_tmpl = win_mat[:, 0].astype(np.uint8)
    win_atom_idx = win_mat[:, 1:5].astype(np.int32)
    win_bond_raw = win_mat[:, 5:11].astype(np.int32)
    win_bond_idx = np.where(win_bond_raw == 0xFFFF, np.int32(-1), win_bond_raw).astype(np.int32)
    kk = D.TEMPLATE_K[win_tmpl]
    win_node_mask = (np.arange(4)[None, :] < kk[:, None]).astype(np.uint8)
    win_edge_mask = np.zeros((win_tmpl.size, 6), np.uint8)
    win_adj = np.zeros((win_tmpl.size, 6), np.uint8)
    for t in range(D.N_TEMPLATES):
        sel = win_tmpl == t
        if sel.any():
            win_edge_mask[sel] = D.TEMPLATE_EDGE_MASK[t]
            win_adj[sel] = D.TEMPLATE_ADJ[t]

    hdr = rel_mat[:, 0]
    base = {
        "mol_word_ptr": mol_word_ptr,
        "word_root": word_root,
        "word_n": word_n,
        "word_tmpl": word_tmpl,
        "word_atoms": word_atoms,
        "word_frame_ptr": word_frame_ptr,
        "frame_slots": frame_slots,
        "frame_win": frame_win,
        "atom_code": codes.atom_matrix(),
        "bond_code": codes.bond_matrix(),
        "win_tmpl": win_tmpl,
        "win_atom_idx": win_atom_idx,
        "win_bond_idx": win_bond_idx,
        "win_node_mask": win_node_mask,
        "win_edge_mask": win_edge_mask,
        "win_adj": win_adj,
    }
    sent = {
        "mol_sent_ptr": mol_sent_ptr,
        "mol_cover_ptr": mol_cover_ptr,
        "sent_root": sent_root,
        "sent_n": sent_n,
        "sent_atoms": sent_atoms,
        "sent_ncovers": sent_ncovers,
        "cover_sent": cover_sent,
        "cover_u": cover_u,
        "cover_v": cover_v,
        "row_rel_uv": row_rel_uv,
        "row_rel_vu": row_rel_vu,
        "rel_tmpl_u": (hdr & 0xFF).astype(np.uint8),
        "rel_tmpl_v": ((hdr >> 8) & 0xFF).astype(np.uint8),
        "rel_mask_u": (rel_mat[:, 1] & 0xFF).astype(np.uint8),
        "rel_mask_v": ((rel_mat[:, 1] >> 8) & 0xFF).astype(np.uint8),
        "rel_e": rel_mat[:, 2],
        "rel_c": rel_mat[:, 3],
        "rel_bond_idx": rel_mat[:, 4:20].astype(np.int32),
    }
    del split_of
    return {
        "base": base,
        "sent": sent,
        "build_seconds": elapsed,
        "n_unique_win": int(win_tmpl.size),
        "n_unique_rel": int(int(rel_mat.shape[0])),
        "n_raw_frames": int(n_raw_frames),
        "n_raw_rows": int(row_rel_uv.size),
    }


def build_cache(raw: Mapping[str, Any], *, log_every: int = 400) -> dict[str, Any]:
    n_mol = int(np.asarray(raw["node_ptr"]).size - 1)
    split_of = np.full(n_mol, -1, np.int8)
    for name, key in (("train", "train_idx"), ("valid", "valid_idx"), ("test", "test_idx")):
        split_of[np.asarray(raw[key], np.int64)] = C.SPLIT_ID[name]
    if (split_of < 0).any():
        raise ValueError("some official row is in no split")
    mols = [D.raw_molecule(raw, row) for row in range(n_mol)]
    built = build_structures(mols, D.CodeTables(), log_every=log_every)
    built["base"].update({
        "row_ids": np.asarray(raw["row_ids"], np.int64).astype(np.int32),
        "split_id": split_of,
        "labels": np.asarray(raw["labels"], np.float32),
        "node_ptr": np.asarray(raw["node_ptr"], np.int64),
        "bond_ptr": np.asarray(raw["bond_ptr"], np.int64),
        "atom_feat": np.asarray(raw["atom_feat"], np.uint8),
        "bond_u": np.asarray(raw["bond_u"], np.int32),
        "bond_v": np.asarray(raw["bond_v"], np.int32),
        "bond_feat": np.asarray(raw["bond_feat"], np.uint8),
    })
    return built


def _save(path: Path, payload: Mapping[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.ascontiguousarray(v) for k, v in payload.items()})
    return {"path": str(path), "sha256": C.sha256_file(path), "bytes": int(path.stat().st_size)}


def stats_from_cache(base: Mapping[str, Any], sent: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    split_id = np.asarray(base["split_id"])
    train_mask = split_id == C.SPLIT_ID["train"]
    n_train = int(train_mask.sum())
    mol_word_ptr = np.asarray(base["mol_word_ptr"])
    word_tmpl = np.asarray(base["word_tmpl"])
    mol_sent_ptr = np.asarray(sent["mol_sent_ptr"])
    sent_n = np.asarray(sent["sent_n"])
    word_frame_ptr = np.asarray(base["word_frame_ptr"])
    frames_per_word = np.diff(word_frame_ptr)
    labels = np.asarray(base["labels"], np.float64)
    train_idx = np.where(train_mask)[0]
    # per-molecule template counts
    counts = np.zeros((n_train, D.N_TEMPLATES), np.int64)
    for pos, row in enumerate(train_idx.tolist()):
        w0, w1 = int(mol_word_ptr[row]), int(mol_word_ptr[row + 1])
        counts[pos] = np.bincount(word_tmpl[w0:w1].astype(np.int64), minlength=D.N_TEMPLATES)
    tmpl_mean = counts.mean(axis=0)
    tmpl_df = (counts > 0).sum(axis=0)

    # per-molecule sentence counts by size
    sent_counts = np.zeros((n_train, 2), np.int64)
    for pos, row in enumerate(train_idx.tolist()):
        s0, s1 = int(mol_sent_ptr[row]), int(mol_sent_ptr[row + 1])
        sizes = sent_n[s0:s1]
        sent_counts[pos, 0] = int((sizes == 5).sum())
        sent_counts[pos, 1] = int((sizes == 6).sum())
    nu = sent_counts.mean(axis=0)

    atom_feat = np.asarray(base["atom_feat"])
    bond_feat = np.asarray(base["bond_feat"])
    node_ptr = np.asarray(base["node_ptr"])
    bond_ptr = np.asarray(base["bond_ptr"])
    atom_support = [set() for _ in range(C.ATOM_COLUMNS)]
    bond_support = [set() for _ in range(C.BOND_COLUMNS)]
    for row in train_idx.tolist():
        a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        block = atom_feat[a0:a1]
        for j in range(C.ATOM_COLUMNS):
            atom_support[j].update(int(v) for v in np.unique(block[:, j]).tolist())
        bonds = bond_feat[b0:b1]
        for j in range(C.BOND_COLUMNS):
            bond_support[j].update(int(v) for v in np.unique(bonds[:, j]).tolist())

    scaler = C.train_target_scaler(labels[train_idx])
    return {
        "n_molecules": int(split_id.size),
        "split_sizes": {name: int((split_id == C.SPLIT_ID[name]).sum()) for name in C.SPLITS},
        "n_words": int(word_tmpl.size),
        "n_frames": int(frames_per_word.sum()),
        "n_sentences": int(sent_n.size),
        "n_covers": int(np.asarray(sent["cover_sent"]).size),
        "n_rows": int(np.asarray(sent["row_rel_uv"]).size),
        "n_unique_win": int(np.asarray(base["win_tmpl"]).size),
        "n_unique_rel": int(np.asarray(sent["rel_tmpl_u"]).size),
        "templates": [D.TEMPLATES[t].as_dict() for t in range(D.N_TEMPLATES)],
        "template_train_mean_per_molecule": tmpl_mean.tolist(),
        "template_train_df_molecules": tmpl_df.tolist(),
        "template_active": (tmpl_df > 0).astype(int).tolist(),
        "mu_T": np.maximum(1.0, tmpl_mean).tolist(),
        "nu_k": np.maximum(1.0, nu).tolist(),
        "sentence_size_mean_per_molecule": nu.tolist(),
        "atom_support": [sorted(s) for s in atom_support],
        "bond_support": [sorted(s) for s in bond_support],
        "atom_code_values": sorted({int(v) for v in atom_feat.reshape(-1)}),
        "bond_code_values": sorted({int(v) for v in bond_feat.reshape(-1)}),
        "target_scaler": scaler,
        "raw_cache_sha256": raw.get("cache_sha256"),
    }



def prepare(*, verbose: bool = True) -> dict[str, Any]:
    cfg = C.load_config()
    raw = D.load_raw_cache(cfg)
    built = build_cache(raw, log_every=500 if verbose else 0)
    base_info = _save(base_path(), built["base"])
    sent_info = _save(sent_path(), built["sent"])
    stats = stats_from_cache(built["base"], built["sent"], raw)
    stats["files"] = {"base": base_info, "sent": sent_info}
    stats["build_seconds"] = built["build_seconds"]
    stats["templates_source"] = "swf_data.enumerate_templates (exhaustive, root fixed at slot 0)"
    C.write_json(stats_path(), stats)
    return stats


def cache_ready() -> bool:
    return base_path().is_file() and sent_path().is_file() and stats_path().is_file()


# ---------------------------------------------------------------------------
# loaded cache
# ---------------------------------------------------------------------------


class Cache:
    def __init__(
        self,
        *,
        need_sentences: bool = True,
        base: Mapping[str, np.ndarray] | None = None,
        sent: Mapping[str, np.ndarray] | None = None,
        stats: Mapping[str, Any] | None = None,
    ) -> None:
        if base is None:
            if not base_path().is_file():
                raise FileNotFoundError(f"structure cache missing: {base_path()} (run `prepare`)")
            with np.load(base_path(), allow_pickle=False) as handle:
                base = {key: handle[key] for key in handle.files}
        self.base = {k: np.asarray(v) for k, v in base.items()}
        self.sent: dict[str, np.ndarray] = {}
        if need_sentences:
            if sent is None:
                if not sent_path().is_file():
                    raise FileNotFoundError(f"sentence cache missing: {sent_path()} (run `prepare`)")
                with np.load(sent_path(), allow_pickle=False) as handle:
                    sent = {key: handle[key] for key in handle.files}
            self.sent = {k: np.asarray(v) for k, v in sent.items()}
        self.stats = dict(C.read_json(stats_path()) if stats is None else stats)
        b = self.base
        self.split_id = np.asarray(b["split_id"])
        self.split_rows = {
            name: np.where(self.split_id == C.SPLIT_ID[name])[0].astype(np.int64) for name in C.SPLITS
        }
        self.row_ids = np.asarray(b["row_ids"])
        self.labels = np.asarray(b["labels"], np.float64)
        self.mol_word_ptr = np.asarray(b["mol_word_ptr"], np.int64)
        self.word_frame_ptr = np.asarray(b["word_frame_ptr"], np.int64)
        self.word_tmpl = np.asarray(b["word_tmpl"])
        self.word_n = np.asarray(b["word_n"])
        self.frame_win = np.asarray(b["frame_win"], np.int32)
        self.atom_code = np.asarray(b["atom_code"], np.int64)
        self.bond_code = np.asarray(b["bond_code"], np.int64)
        self.n_atoms = np.diff(np.asarray(b["node_ptr"], np.int64))
        self.n_bonds = np.diff(np.asarray(b["bond_ptr"], np.int64))
        self.aut = D.TEMPLATE_NFRAMES[self.word_tmpl].astype(np.int64)
        self.n_mol = int(self.split_id.size)
        self.scaler = self.stats["target_scaler"]
        self.mu_T = np.asarray(self.stats["mu_T"], np.float64)
        self.nu_k = np.asarray(self.stats["nu_k"], np.float64)
        self.template_active = np.asarray(self.stats["template_active"], np.float64)
        if need_sentences:
            s = self.sent
            self.mol_sent_ptr = np.asarray(s["mol_sent_ptr"], np.int64)
            self.mol_cover_ptr = np.asarray(s["mol_cover_ptr"], np.int64)
            self.sent_n = np.asarray(s["sent_n"])
            self.sent_ncovers = np.asarray(s["sent_ncovers"], np.int64)
            self.cover_sent = np.asarray(s["cover_sent"], np.int32)
            self.cover_u = np.asarray(s["cover_u"], np.int32)
            self.cover_v = np.asarray(s["cover_v"], np.int32)
            self.row_rel_uv = np.asarray(s["row_rel_uv"], np.int32)
            self.row_rel_vu = np.asarray(s["row_rel_vu"], np.int32)
            au = self.aut[self.cover_u]
            av = self.aut[self.cover_v]
            self.cover_nrows = (au * av).astype(np.int64)
            self.cover_row_ptr = np.concatenate(
                [np.zeros(1, np.int64), np.cumsum(self.cover_nrows)]
            )
            self.rel_tmpl_u = np.asarray(s["rel_tmpl_u"])
            self.rel_tmpl_v = np.asarray(s["rel_tmpl_v"])
            self.rel_mask_u = np.asarray(s["rel_mask_u"])
            self.rel_mask_v = np.asarray(s["rel_mask_v"])
            self.rel_e = np.asarray(s["rel_e"])
            self.rel_c = np.asarray(s["rel_c"])
            self.rel_bond_idx = np.asarray(s["rel_bond_idx"], np.int32)

    # ---------------------------------------------------------------- packing
    def sample_sizes(self) -> dict[str, int]:
        return {
            "molecules": self.n_mol,
            "words": int(self.word_tmpl.size),
            "frames": int(self.frame_win.size),
            "sentences": int(self.sent_n.size) if self.sent else 0,
            "covers": int(self.cover_sent.size) if self.sent else 0,
            "rows": int(self.row_rel_uv.size) if self.sent else 0,
            "unique_win": int(np.asarray(self.base["win_tmpl"]).size),
            "unique_rel": int(np.asarray(self.sent["rel_tmpl_u"]).size) if self.sent else 0,
        }

    def pack(self, mols: Sequence[int], *, sentences: bool) -> dict[str, Any]:
        """Pack a set of molecule rows into flat index tensors (numpy, int64)."""
        mols = np.asarray(mols, dtype=np.int64)
        B = int(mols.size)
        mw0 = self.mol_word_ptr[mols]
        mw1 = self.mol_word_ptr[mols + 1]
        w_counts = mw1 - mw0
        nwb = int(w_counts.sum())
        w_idx = _ranges(mw0, mw1)
        w_mol = np.repeat(np.arange(B, dtype=np.int64), w_counts)

        fw_ptr = self.word_frame_ptr
        f0 = fw_ptr[mw0]
        f1 = fw_ptr[mw1]
        f_counts = f1 - f0
        nfb = int(f_counts.sum())
        f_idx = _ranges(f0, f1)
        f_mol = np.repeat(np.arange(B, dtype=np.int64), f_counts)
        f_win_global = self.frame_win[f_idx]
        uniq_win, f_win = np.unique(f_win_global, return_inverse=True)
        w_counts_seq = np.diff(fw_ptr)[w_idx]
        frame_word = np.repeat(np.arange(nwb, dtype=np.int64), w_counts_seq)

        out: dict[str, Any] = {
            "mols": mols,
            "n_molecules": B,
            "atom_code": self.atom_code,
            "bond_code": self.bond_code,
            "win_tmpl": np.asarray(self.base["win_tmpl"])[uniq_win].astype(np.int64),
            "win_atom_idx": np.asarray(self.base["win_atom_idx"])[uniq_win].astype(np.int64),
            "win_bond_idx": np.clip(
                np.asarray(self.base["win_bond_idx"])[uniq_win], 0, None
            ).astype(np.int64),
            "win_node_mask": np.asarray(self.base["win_node_mask"])[uniq_win].astype(np.float32),
            "win_edge_mask": np.asarray(self.base["win_edge_mask"])[uniq_win].astype(np.float32),
            "win_adj": np.asarray(self.base["win_adj"])[uniq_win].astype(np.float32),
            "frame_win": f_win.astype(np.int64),
            "frame_word": frame_word,
            "word_tmpl": self.word_tmpl[w_idx].astype(np.int64),
            "word_nframes": np.diff(fw_ptr)[w_idx].astype(np.float32),
            "word_mol": w_mol,
            "frame_mol": f_mol,
            "n_atoms": self.n_atoms[mols].astype(np.float32),
            "n_bonds": self.n_bonds[mols].astype(np.float32),
            "n_words": nwb,
            "n_frames": nfb,
            "n_unique_win": int(uniq_win.size),
        }
        if not sentences:
            return out

        ms0 = self.mol_sent_ptr[mols]
        ms1 = self.mol_sent_ptr[mols + 1]
        s_counts = ms1 - ms0
        nsb = int(s_counts.sum())
        s_idx = _ranges(ms0, ms1)
        s_mol = np.repeat(np.arange(B, dtype=np.int64), s_counts)

        mc0 = self.mol_cover_ptr[mols]
        mc1 = self.mol_cover_ptr[mols + 1]
        c_counts = mc1 - mc0
        ncb = int(c_counts.sum())
        c_idx = _ranges(mc0, mc1)
        c_mol = np.repeat(np.arange(B, dtype=np.int64), c_counts)

        r0 = self.cover_row_ptr[mc0]
        r1 = self.cover_row_ptr[mc1]
        rr_counts = r1 - r0
        nrow = int(rr_counts.sum())
        row_idx = _ranges(r0, r1)
        row_mol = np.repeat(np.arange(B, dtype=np.int64), rr_counts)

        cover_of_row = np.repeat(np.arange(ncb, dtype=np.int64), self.cover_nrows[c_idx])
        au_c = self.aut[self.cover_u[c_idx]]
        av_c = self.aut[self.cover_v[c_idx]]
        au_r = np.repeat(au_c, self.cover_nrows[c_idx])
        av_r = np.repeat(av_c, self.cover_nrows[c_idx])
        # position of the row inside its cover
        cover_starts = np.repeat(
            np.concatenate(([0], np.cumsum(self.cover_nrows[c_idx])[:-1])), self.cover_nrows[c_idx]
        )
        t = np.arange(nrow, dtype=np.int64) - cover_starts

        wu_g = self.cover_u[c_idx][cover_of_row].astype(np.int64)
        wv_g = self.cover_v[c_idx][cover_of_row].astype(np.int64)
        left_g = fw_ptr[wu_g] + t // av_r
        right_g = fw_ptr[wv_g] + t % av_r
        left = _local_pos(f_idx, left_g)
        right = _local_pos(f_idx, right_g)
        sent_g = self.cover_sent[c_idx][cover_of_row].astype(np.int64)
        sent_local = _local_pos(s_idx, sent_g)
        wu_local = _local_pos(w_idx, wu_g)
        wv_local = _local_pos(w_idx, wv_g)
        del row_mol
        row_of_mol = np.repeat(np.arange(B, dtype=np.int64), rr_counts)
        weight = 1.0 / (
            self.sent_ncovers[sent_g].astype(np.float64) * au_r.astype(np.float64) * av_r.astype(np.float64)
        )

        rel_uv_g = self.row_rel_uv[row_idx]
        rel_vu_g = self.row_rel_vu[row_idx]
        uniq_rel, rel_inv = np.unique(
            np.concatenate([rel_uv_g, rel_vu_g]).astype(np.int64), return_inverse=True
        )
        nrel = int(uniq_rel.size)
        rel_uv = rel_inv[:nrow]
        rel_vu = rel_inv[nrow:]

        win_left = f_win[left]
        win_right = f_win[right]

        out.update({
            "sent_n": self.sent_n[s_idx].astype(np.int64),
            "sent_mol": s_mol,
            "sent_ncovers": self.sent_ncovers[s_idx].astype(np.float32),
            "row_left": left,
            "row_right": right,
            "row_word_u": wu_local,
            "row_word_v": wv_local,
            "row_sent": sent_local,
            "row_weight": weight.astype(np.float32),
            "row_rel_uv": rel_uv.astype(np.int64),
            "row_rel_vu": rel_vu.astype(np.int64),
            "row_win_left": win_left.astype(np.int64),
            "row_win_right": win_right.astype(np.int64),
            "row_mol": row_of_mol,
            # relation table slice
            "rel_tmpl_u": self.rel_tmpl_u[uniq_rel].astype(np.int64),
            "rel_tmpl_v": self.rel_tmpl_v[uniq_rel].astype(np.int64),
            "rel_mask_u": (
                np.arange(4)[None, :] < self.rel_mask_u[uniq_rel].astype(np.int64)[:, None]
            ).astype(np.float32),
            "rel_mask_v": (
                np.arange(4)[None, :] < self.rel_mask_v[uniq_rel].astype(np.int64)[:, None]
            ).astype(np.float32),
            "rel_e": _bits_to_matrix(self.rel_e[uniq_rel].astype(np.int64)),
            "rel_c": _bits_to_matrix(self.rel_c[uniq_rel].astype(np.int64)),
            "rel_bond_idx": np.clip(
                self.rel_bond_idx[uniq_rel], 0, None
            ).astype(np.int64),
            "rel_c_mask": (self.rel_c[uniq_rel] != 0).astype(np.float32),
            "n_sentences": nsb,
            "n_covers": ncb,
            "n_rows": nrow,
            "n_unique_rel": nrel,
        })
        return out
