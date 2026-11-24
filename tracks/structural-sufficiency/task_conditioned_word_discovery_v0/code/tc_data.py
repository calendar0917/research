"""Neural data assembly for the frozen selected vocabularies.

Per-atom word counts are built directly from the exact rooted supports and the
frozen canonical certificates.  A selected vocabulary is a list of
(size, granularity, signature) entries; a signature that is not in the list is
ignored, so no unselected word type can reach the neural path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

import tc_common as C
import tc_vocab as V


@dataclass
class SplitData:
    split: str
    rows: np.ndarray
    fit_rows: np.ndarray
    eval_rows: np.ndarray
    fit_index: np.ndarray
    eval_index: np.ndarray
    atom_feat: list[np.ndarray]
    bond_feat: list[np.ndarray]
    bond_u: list[np.ndarray]
    bond_v: list[np.ndarray]
    counts: list[np.ndarray]          # [n_atoms, n_union]
    n_atoms: np.ndarray
    n_bonds: np.ndarray
    graph_scalars: np.ndarray         # raw, rows x 8
    graph_mean: np.ndarray
    graph_scale: np.ndarray
    target_raw: np.ndarray
    target_std: np.ndarray
    labels_fit: np.ndarray
    labels_eval: np.ndarray
    scaler: dict[str, float]
    atom_dims: list[int]
    bond_dims: list[int]
    atom_ok: list[np.ndarray]
    bond_ok: list[np.ndarray]
    union_vocab: list[dict[str, Any]]
    union_of_key: dict[str, int]
    arm_columns: dict[str, np.ndarray]
    n_task: int
    n_freq: int
    hashes: dict[str, Any]


def _load_vocab(split: str, arm: str) -> list[dict[str, Any]]:
    payload = C.read_json(C.ANALYSIS_DIR / f"vocab_{split}_{arm}.json")
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"empty vocabulary: {split}/{arm}")
    return payload


def build_union_vocab(task: Sequence[Mapping[str, Any]],
                      freq: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]],
                                                                    dict[str, int]]:
    union: list[dict[str, Any]] = []
    of_key: dict[str, int] = {}
    for entry in list(task) + list(freq):
        key = str(entry["key"])
        if key in of_key:
            continue
        of_key[key] = len(union)
        union.append({"key": key, "size": int(entry["size"]),
                      "gran": str(entry["gran"]),
                      "signature": [int(x) for x in entry["signature"]]})
    return union, of_key


def build_split_data(split: str, atom_dims: Sequence[int], bond_dims: Sequence[int]
                     ) -> SplitData:
    bundle = C.split_bundle(split)
    fit_rows = np.asarray(bundle.fit_rows, np.int64)
    eval_rows = np.asarray(bundle.eval_rows, np.int64)
    rows = np.concatenate([fit_rows, eval_rows]).astype(np.int64)
    if np.unique(rows).size != rows.size:
        raise ValueError("duplicate rows in fit+eval")
    fit_index = np.arange(fit_rows.size, dtype=np.int64)
    eval_index = np.arange(fit_rows.size, rows.size, dtype=np.int64)

    task = _load_vocab(split, "task")
    freq = _load_vocab(split, "freq")
    if len(task) != len(freq):
        raise ValueError(f"TASK/FREQ sizes differ: {len(task)} vs {len(freq)}")
    union, of_key = build_union_vocab(task, freq)
    arm_columns = {
        "D_TASK": np.asarray([of_key[str(e["key"])] for e in task], np.int64),
        "D_FREQ": np.asarray([of_key[str(e["key"])] for e in freq], np.int64),
    }
    n_union = len(union)
    key_of_family: dict[str, dict[bytes, tuple[int, int]]] = {}
    for entry in union:
        fam = V._family(str(entry["gran"]), int(entry["size"]))
        sig = C.signature_bytes(entry["signature"])
        key_of_family.setdefault(fam, {})[sig] = (of_key[str(entry["key"])],
                                                 int(entry["size"]))

    atom_feat: list[np.ndarray] = []
    bond_feat: list[np.ndarray] = []
    bond_u: list[np.ndarray] = []
    bond_v: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    n_atoms = np.zeros(rows.size, np.int64)
    n_bonds = np.zeros(rows.size, np.int64)
    graph = np.zeros((rows.size, 8), np.float64)
    for i, row in enumerate(rows.tolist()):
        st = C.build_struct(bundle, int(row))
        n_atoms[i] = st.n_atoms
        n_bonds[i] = st.n_bonds
        graph[i] = st.graph_scalars
        atom_feat.append(np.asarray(st.atom_feat, np.int64))
        bond_feat.append(np.asarray(st.bond_feat, np.int64))
        bond_u.append(np.asarray(st.bond_u, np.int64))
        bond_v.append(np.asarray(st.bond_v, np.int64))
        cnt = np.zeros((st.n_atoms, n_union), np.float32)
        for fam, mapping in key_of_family.items():
            gran, size_s = fam.split("_")
            size = int(size_s)
            sig = V._signatures_for(st, size, gran, atom_dims, bond_dims)
            if sig.shape[0] == 0:
                continue
            roots = np.arange(st.n_atoms) if size == 1 else \
                np.asarray(st.atoms[size][:, 0], np.int64)
            for k, s in enumerate(sig):
                hit = mapping.get(C.signature_bytes(s))
                if hit is not None:
                    cnt[int(roots[k]), int(hit[0])] += 1.0
        counts.append(cnt)

    labels = np.asarray(bundle.labels, np.float64)
    target_raw = labels[rows]
    if not np.isfinite(target_raw).all():
        raise ValueError("non-finite label on a training/eval row")
    target_std = (target_raw - bundle.scaler["mean"]) / bundle.scaler["std"]
    atom_ok, bond_ok = C.GD.category_support(bundle.raw, fit_rows, atom_dims, bond_dims)
    hashes = {
        "split_meta": C.split_meta(bundle),
        "rows_sha256": C.array_sha256(rows),
        "fit_rows_sha256": C.array_sha256(fit_rows),
        "eval_rows_sha256": C.array_sha256(eval_rows),
        "counts_sha256": hashlib.sha256(b"".join(np.ascontiguousarray(c).tobytes()
                                                 for c in counts)).hexdigest(),
        "task_vocab_sha256": C.sha256_json(task),
        "freq_vocab_sha256": C.sha256_json(freq),
        "union_vocab_sha256": C.sha256_json(union),
        "n_union_words": int(n_union),
        "n_task_words": int(len(task)),
        "n_freq_words": int(len(freq)),
        "n_atoms_total": int(n_atoms.sum()),
        "n_bonds_total": int(n_bonds.sum()),
    }
    return SplitData(
        split=split, rows=rows, fit_rows=fit_rows, eval_rows=eval_rows,
        fit_index=fit_index, eval_index=eval_index,
        atom_feat=atom_feat, bond_feat=bond_feat, bond_u=bond_u, bond_v=bond_v,
        counts=counts, n_atoms=n_atoms, n_bonds=n_bonds,
        graph_scalars=graph, graph_mean=np.asarray(bundle.graph_mean, np.float64),
        graph_scale=np.asarray(bundle.graph_scale, np.float64),
        target_raw=target_raw, target_std=target_std,
        labels_fit=target_raw[fit_index], labels_eval=target_raw[eval_index],
        scaler=dict(bundle.scaler), atom_dims=[int(x) for x in atom_dims],
        bond_dims=[int(x) for x in bond_dims], atom_ok=atom_ok, bond_ok=bond_ok,
        union_vocab=union, union_of_key=of_key, arm_columns=arm_columns,
        n_task=len(task), n_freq=len(freq), hashes=hashes)


def collate(data: SplitData, indices: Sequence[int], arm: str,
            device: Any = "cpu") -> dict[str, Any]:
    import torch

    idx = np.asarray(indices, np.int64)
    atom_parts = [data.atom_feat[int(i)] for i in idx]
    bond_parts = [data.bond_feat[int(i)] for i in idx]
    u_parts = [data.bond_u[int(i)] for i in idx]
    v_parts = [data.bond_v[int(i)] for i in idx]
    atom_off = 0
    bond_off = 0
    atom_feat = []
    bond_feat = []
    edge_src = []
    edge_dst = []
    edge_bond = []
    atom_mol = []
    word_parts = []
    cols = data.arm_columns.get(arm, None)
    for m, i in enumerate(idx.tolist()):
        a = atom_parts[m]
        b = bond_parts[m]
        u = u_parts[m]
        v = v_parts[m]
        n = int(a.shape[0])
        atom_feat.append(a)
        bond_feat.append(b)
        atom_mol.append(np.full(n, m, np.int64))
        if cols is not None:
            word_parts.append(np.asarray(data.counts[i][:, cols], np.float32))
        else:
            word_parts.append(np.zeros((n, data.n_task), np.float32))
        gu = u + atom_off
        gv = v + atom_off
        gb = np.arange(bond_off, bond_off + b.shape[0], dtype=np.int64)
        edge_src.extend([gu, gv])
        edge_dst.extend([gv, gu])
        edge_bond.extend([gb, gb])
        atom_off += n
        bond_off += b.shape[0]
    atom_feat_t = torch.as_tensor(np.concatenate(atom_feat), dtype=torch.long, device=device)
    bond_feat_t = torch.as_tensor(np.concatenate(bond_feat) if bond_feat else
                                  np.zeros((0, 3), np.int64), dtype=torch.long, device=device)
    e_src = torch.as_tensor(np.concatenate(edge_src) if edge_src else np.zeros(0, np.int64),
                            dtype=torch.long, device=device)
    e_dst = torch.as_tensor(np.concatenate(edge_dst) if edge_dst else np.zeros(0, np.int64),
                            dtype=torch.long, device=device)
    e_bond = torch.as_tensor(np.concatenate(edge_bond) if edge_bond else np.zeros(0, np.int64),
                             dtype=torch.long, device=device)
    atom_mol_t = torch.as_tensor(np.concatenate(atom_mol), dtype=torch.long, device=device)
    word = torch.as_tensor(np.concatenate(word_parts), dtype=torch.float32, device=device)
    g = (data.graph_scalars[idx] - data.graph_mean) / data.graph_scale
    return {
        "atom_feat": atom_feat_t,
        "bond_feat": bond_feat_t,
        "edge_src": e_src,
        "edge_dst": e_dst,
        "edge_bond": e_bond,
        "atom_mol": atom_mol_t,
        "word_counts": word,
        "n_atoms": torch.as_tensor(data.n_atoms[idx], dtype=torch.float32, device=device),
        "graph_scalars": torch.as_tensor(g, dtype=torch.float32, device=device),
        "rows": torch.as_tensor(data.rows[idx], dtype=torch.long, device=device),
        "n_molecules": int(idx.size),
        "n_atom_total": int(atom_feat_t.shape[0]),
        "n_bond_total": int(bond_feat_t.shape[0]),
    }