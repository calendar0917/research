"""Train/Validation-only Peptides-func loading, row identity and C1 task setup.

The official LRGB cache built by the historical Fast v1 round is treated as
read-only.  Nothing in this module can name the Test split: ``load_cache_split``
and ``iter_pyg_split`` accept only ``train`` / ``valid``.
"""

from __future__ import annotations

import numpy as np
import torch

import pc_common as C

# Imported read-only from the historical Fast v1 implementation.
import bp_common as FC  # noqa: E402
import bp_data as D  # noqa: E402
from feature_spec import load_spec  # noqa: E402


class DataGateError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Cache decoding
# ---------------------------------------------------------------------------


def _decode_one_hot(block: np.ndarray, vocab: tuple[int, ...], label: str) -> np.ndarray:
    block = np.asarray(block, dtype=np.float32)
    if block.ndim != 2:
        raise DataGateError(f"{label}: one-hot block is not 2-D")
    if block.shape[1] != int(sum(vocab)):
        raise DataGateError(f"{label}: one-hot width {block.shape[1]} != {sum(vocab)}")
    cols: list[np.ndarray] = []
    off = 0
    for size in vocab:
        part = block[:, off : off + size]
        if not np.allclose(part.sum(axis=1), 1.0, atol=1e-5):
            raise DataGateError(f"{label}: one-hot block does not sum to 1")
        if not np.all(np.isfinite(part)):
            raise DataGateError(f"{label}: non-finite one-hot value")
        cols.append(part.argmax(axis=1).astype(np.int64))
        off += int(size)
    return np.stack(cols, axis=1)


def decode_atom_raw(atom_feat: np.ndarray) -> np.ndarray:
    block = np.asarray(atom_feat, dtype=np.float32)
    width = int(sum(C.ATOM_VOCABS))
    if block.ndim != 2 or block.shape[1] < width:
        raise DataGateError(f"atom_feat: width {block.shape[1] if block.ndim == 2 else block.shape} < {width}")
    return _decode_one_hot(block[:, :width], C.ATOM_VOCABS, "atom_feat")


def decode_bond_raw(bond_feat: np.ndarray) -> np.ndarray:
    block = np.asarray(bond_feat, dtype=np.float32)
    expected = int(sum(C.BOND_VOCABS)) + 1
    if block.shape[1] != expected:
        raise DataGateError(f"bond_feat: width {block.shape[1]} != {expected}")
    return _decode_one_hot(block[:, : sum(C.BOND_VOCABS)], C.BOND_VOCABS, "bond_feat")


def load_cache_split(split: str) -> dict:
    if split not in C.ALLOWED_SPLITS:
        raise DataGateError(f"split {split!r} is not allowed in this round")
    path = C.PREV_CACHE / f"peptides_func_{split}.pt"
    if not path.is_file():
        raise DataGateError(f"missing cache {path}")
    return D.load_cache(path)


def cache_graphs(cache: dict, *, decode: bool = True) -> list[dict]:
    """Decode every graph from the frozen cache into label-free raw arrays."""
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    n_graphs = int(cache["n_molecules"])
    if atom_ptr.shape != (n_graphs + 1,) or bond_ptr.shape != (n_graphs + 1,):
        raise DataGateError("cache pointer length mismatch")
    atom_raw_all = decode_atom_raw(cache["atom_feat"]) if decode else None
    bond_raw_all = decode_bond_raw(cache["bond_feat"]) if decode else None
    atom_type = np.asarray(cache["atom_type"], dtype=np.int64)
    bond_type = np.asarray(cache["bond_type"], dtype=np.int64)
    bond_index_all = np.asarray(cache["bond_index"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    graphs: list[dict] = []
    for i in range(n_graphs):
        a0, a1 = int(atom_ptr[i]), int(atom_ptr[i + 1])
        b0, b1 = int(bond_ptr[i]), int(bond_ptr[i + 1])
        atom_raw = atom_raw_all[a0:a1] if atom_raw_all is not None else None
        bond_raw = bond_raw_all[b0:b1] if bond_raw_all is not None else None
        pairs = bond_index_all[b0:b1]
        if atom_raw is not None:
            if not np.array_equal(atom_raw[:, 0], atom_type[a0:a1]):
                raise DataGateError(f"graph {i}: atom_type != atom_raw[:,0]")
        if bond_raw is not None:
            if not np.array_equal(bond_raw[:, 0] + 1, bond_type[b0:b1]):
                raise DataGateError(f"graph {i}: bond_type != bond_raw[:,0]+1")
        if pairs.size:
            if np.any(pairs[:, 0] >= pairs[:, 1]):
                raise DataGateError(f"graph {i}: bond_index is not canonical (u<v)")
            if pairs.shape[0] > 1:
                strictly_ordered = (pairs[1:, 0] > pairs[:-1, 0]) | (
                    (pairs[1:, 0] == pairs[:-1, 0]) & (pairs[1:, 1] > pairs[:-1, 1])
                )
                if not bool(np.all(strictly_ordered)):
                    raise DataGateError(f"graph {i}: bond_index is not strictly sorted")
        graphs.append(
            {
                "gid": int(np.asarray(cache["gids"])[i]) if "gids" in cache else i,
                "n_atoms": a1 - a0,
                "n_bonds": b1 - b0,
                "atom_raw": atom_raw,
                "bond_raw": bond_raw,
                "bond_index": pairs,
                "n_paths_cache": int(path_ptr[a1] - path_ptr[a0]),
            }
        )
    return graphs


# ---------------------------------------------------------------------------
# Official PyG Train / Validation split (never Test)
# ---------------------------------------------------------------------------


def iter_pyg_split(split: str):
    from torch_geometric.datasets import LRGBDataset

    if split not in C.ALLOWED_SPLITS:
        raise DataGateError(f"split {split!r} is not allowed in this round")
    loader_split = "val" if split == "valid" else "train"
    ds = LRGBDataset(root=str(C.LRGB_ROOT), name="Peptides-func", split=loader_split)
    for pos in range(len(ds)):
        data = ds[pos]
        x = np.asarray(data.x.detach().cpu().numpy(), dtype=np.int64)
        ei = np.asarray(data.edge_index.detach().cpu().numpy(), dtype=np.int64)
        ea = np.asarray(data.edge_attr.detach().cpu().numpy(), dtype=np.int64)
        y = np.asarray(data.y.detach().cpu().numpy(), dtype=np.int64).reshape(-1)
        seen: dict[tuple[int, int], np.ndarray] = {}
        for k in range(ei.shape[1]):
            u, v = int(ei[0, k]), int(ei[1, k])
            key = (min(u, v), max(u, v))
            if key in seen:
                if not np.array_equal(seen[key], ea[k]):
                    raise DataGateError(f"{split}[{pos}]: reverse edge attributes disagree")
            else:
                seen[key] = ea[k]
        keys = sorted(seen)
        pairs = np.asarray(keys, dtype=np.int64).reshape(-1, 2)
        bond_raw = (
            np.asarray([seen[k] for k in keys], dtype=np.int64).reshape(-1, len(C.BOND_VOCABS))
            if keys
            else np.zeros((0, len(C.BOND_VOCABS)), dtype=np.int64)
        )
        yield pos, {
            "gid": pos,
            "atom_raw": x,
            "bond_index": pairs,
            "bond_raw": bond_raw,
            "y": y,
        }


# ---------------------------------------------------------------------------
# Row identity: official PyG rows <-> frozen cache rows
# ---------------------------------------------------------------------------


def row_identity(split: str, *, max_rows: int | None = None) -> dict:
    cache = load_cache_split(split)
    graphs = cache_graphs(cache)
    expected_n = C.SPLIT_SIZES[split]
    if int(cache["n_molecules"]) != expected_n:
        raise DataGateError(f"{split}: cache rows {cache['n_molecules']} != {expected_n}")
    if len(graphs) != expected_n:
        raise DataGateError(f"{split}: decoded rows {len(graphs)} != {expected_n}")
    y_path = C.PREV_ANALYSIS / f"peptides_func_y_{split}.npy"
    y_frozen = np.load(y_path)
    if y_frozen.shape != (expected_n, C.N_LABELS):
        raise DataGateError(f"{split}: frozen labels shape {y_frozen.shape}")

    n_atoms_total = 0
    n_bonds_total = 0
    n_atom_mismatch = 0
    n_bond_index_mismatch = 0
    n_bond_feature_mismatch = 0
    n_label_mismatch = 0
    n_path_count_mismatch = 0
    row_digests: list[str] = []
    from pc_s2 import enumerate_paths

    for pos, pyg in iter_pyg_split(split):
        if max_rows is not None and pos >= max_rows:
            break
        g = graphs[pos]
        n_atoms_total += int(pyg["atom_raw"].shape[0])
        n_bonds_total += int(pyg["bond_index"].shape[0])
        atom_ok = g["atom_raw"].shape == pyg["atom_raw"].shape and np.array_equal(
            g["atom_raw"], pyg["atom_raw"]
        )
        pairs_ok = g["bond_index"].shape == pyg["bond_index"].shape and np.array_equal(
            g["bond_index"], pyg["bond_index"]
        )
        bond_ok = g["bond_raw"].shape == pyg["bond_raw"].shape and np.array_equal(
            g["bond_raw"], pyg["bond_raw"]
        )
        label_ok = y_frozen[pos].shape == pyg["y"].shape and np.array_equal(
            y_frozen[pos].astype(np.int64), pyg["y"]
        )
        if not atom_ok:
            n_atom_mismatch += 1
        if not pairs_ok:
            n_bond_index_mismatch += 1
        if not bond_ok:
            n_bond_feature_mismatch += 1
        if not label_ok:
            n_label_mismatch += 1
        rec = enumerate_paths(g["atom_raw"], g["bond_raw"], g["bond_index"])
        if int(rec["n_paths"]) != int(g["n_paths_cache"]):
            n_path_count_mismatch += 1
        row_digests.append(
            C.sha256_text(
                C.sha256_array(g["atom_raw"])
                + C.sha256_array(g["bond_index"].astype(np.int64))
                + C.sha256_array(g["bond_raw"])
                + C.sha256_array(y_frozen[pos].astype(np.int64))
            )
        )

    checked = len(row_digests)
    if max_rows is not None and checked != min(max_rows, expected_n):
        raise DataGateError(f"{split}: checked {checked} rows")
    ok = (
        checked == expected_n
        and n_atom_mismatch == 0
        and n_bond_index_mismatch == 0
        and n_bond_feature_mismatch == 0
        and n_label_mismatch == 0
        and n_path_count_mismatch == 0
    )
    return {
        "split": split,
        "n_rows_expected": expected_n,
        "n_rows_checked": checked,
        "n_atoms_checked": n_atoms_total,
        "n_bonds_checked": n_bonds_total,
        "n_atom_feature_mismatch_rows": n_atom_mismatch,
        "n_bond_index_mismatch_rows": n_bond_index_mismatch,
        "n_bond_feature_mismatch_rows": n_bond_feature_mismatch,
        "n_label_mismatch_rows": n_label_mismatch,
        "n_s2_path_count_mismatch_rows": n_path_count_mismatch,
        "row_digest_sha256": C.sha256_text("".join(row_digests)),
        "row_digest_first": row_digests[0] if row_digests else None,
        "row_digest_last": row_digests[-1] if row_digests else None,
        "cache": C.track_rel(C.PREV_CACHE / f"peptides_func_{split}.pt"),
        "cache_sha256": C.sha256_file(C.PREV_CACHE / f"peptides_func_{split}.pt"),
        "labels": C.track_rel(y_path),
        "labels_sha256": C.sha256_file(y_path),
        "status": "PASS" if ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# Raw-array export for the isolated CPU fingerprint environment
# ---------------------------------------------------------------------------


def export_raw_arrays(split: str, out_path) -> dict:
    """Write PyG Train/Validation raw tensors (no labels) for ECFP rebuilding."""
    import os

    x_parts: list[np.ndarray] = []
    ei_parts: list[np.ndarray] = []
    ea_parts: list[np.ndarray] = []
    atom_ptr = [0]
    edge_ptr = [0]
    gids: list[int] = []
    for pos, g in iter_pyg_split(split):
        x_parts.append(np.asarray(g["atom_raw"], dtype=np.int64))
        ei_parts.append(np.asarray(g["bond_index"], dtype=np.int64))
        ea_parts.append(np.asarray(g["bond_raw"], dtype=np.int64))
        atom_ptr.append(atom_ptr[-1] + int(g["atom_raw"].shape[0]))
        edge_ptr.append(edge_ptr[-1] + int(g["bond_index"].shape[0]))
        gids.append(pos)
    x = np.concatenate(x_parts, axis=0) if x_parts else np.zeros((0, 9), np.int64)
    ei = np.concatenate(ei_parts, axis=0) if ei_parts else np.zeros((0, 2), np.int64)
    ea = np.concatenate(ea_parts, axis=0) if ea_parts else np.zeros((0, 3), np.int64)
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        atom_raw=x.astype(np.int16),
        bond_index=ei.astype(np.int32),
        bond_raw=ea.astype(np.int8),
        atom_ptr=np.asarray(atom_ptr, dtype=np.int64),
        edge_ptr=np.asarray(edge_ptr, dtype=np.int64),
        gids=np.asarray(gids, dtype=np.int64),
    )
    return {
        "split": split,
        "path": C.track_rel(out_path),
        "sha256": C.sha256_file(out_path),
        "n_graphs": len(gids),
        "n_atoms": int(x.shape[0]),
        "n_bonds": int(ei.shape[0]),
        "atom_ptr_sha256": C.sha256_array(np.asarray(atom_ptr, dtype=np.int64)),
        "edge_ptr_sha256": C.sha256_array(np.asarray(edge_ptr, dtype=np.int64)),
        "labels_included": False,
    }


# ---------------------------------------------------------------------------
# C1 task (same frozen spec / norm / caches as the historical implementation)
# ---------------------------------------------------------------------------


def load_c1_task() -> dict:
    spec_path = C.PREV_ANALYSIS / "peptides_func_spec.json"
    spec = load_spec(spec_path)
    train_cache = load_cache_split("train")
    valid_cache = load_cache_split("valid")
    train_rows = np.arange(int(train_cache["n_molecules"]), dtype=np.int64)
    valid_rows = np.arange(int(valid_cache["n_molecules"]), dtype=np.int64)
    if train_rows.size != C.SPLIT_SIZES["train"] or valid_rows.size != C.SPLIT_SIZES["valid"]:
        raise DataGateError("cache split size mismatch")
    y_train = np.load(C.PREV_ANALYSIS / "peptides_func_y_train.npy").astype(np.float64)
    y_valid = np.load(C.PREV_ANALYSIS / "peptides_func_y_valid.npy").astype(np.float64)
    if y_train.shape != (train_rows.size, C.N_LABELS) or y_valid.shape != (valid_rows.size, C.N_LABELS):
        raise DataGateError("label shape mismatch")
    norm = D.fit_norm_rows(train_cache, spec, train_rows, source="peptides_func_official_train_only")
    recipe = dict(FC.PEPTIDE_RECIPE)
    cache_identity = {
        "cache_train": C.track_rel(C.PREV_CACHE / "peptides_func_train.pt"),
        "cache_train_sha256": C.sha256_file(C.PREV_CACHE / "peptides_func_train.pt"),
        "cache_valid": C.track_rel(C.PREV_CACHE / "peptides_func_valid.pt"),
        "cache_valid_sha256": C.sha256_file(C.PREV_CACHE / "peptides_func_valid.pt"),
        "feature_spec_hash": spec.hash(),
        "norm_fingerprint": norm["norm_fingerprint"],
        "train_labels_sha256": C.sha256_file(C.PREV_ANALYSIS / "peptides_func_y_train.npy"),
        "valid_labels_sha256": C.sha256_file(C.PREV_ANALYSIS / "peptides_func_y_valid.npy"),
    }
    return {
        "kind": "peptide",
        "dataset": C.DATASET,
        "spec": spec,
        "norm": norm,
        "n_out": C.N_LABELS,
        "train_cache": train_cache,
        "valid_cache": valid_cache,
        "train_rows": train_rows,
        "valid_rows": valid_rows,
        "y_train": y_train,
        "y_valid": y_valid,
        "cont_slice": (0, 0),
        "cont_norm": {"dim": 0, "mean": [], "scale": [], "constant_indices": [], "fit_rows": int(train_rows.size)},
        "recipe": recipe,
        "cache_identity": cache_identity,
    }


__all__ = [
    "DataGateError",
    "cache_graphs",
    "decode_atom_raw",
    "decode_bond_raw",
    "export_raw_arrays",
    "iter_pyg_split",
    "load_c1_task",
    "load_cache_split",
    "row_identity",
]
