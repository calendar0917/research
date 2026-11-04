"""Feature assembly and caching for the five pre-registered arms.

Blocks
------
``G0``            :class:`tsb_g0` global composition/topology counts
``WL{r}``         :class:`tsb_wl` typed-WL counts for round ``r`` (one block per
                  published split because each split owns its Train vocabulary)
``MORGAN_COUNT``  :class:`tsb_morgan` radius-2 2048-bucket count fingerprint

Arms (frozen, see ``PROTOCOL.md`` §3)
-------------------------------------
``G0``            = G0
``WL2_BIN``       = G0 + WL0/1/2 (0/1)
``WL2_COUNT``     = G0 + WL0/1/2 (counts)
``WL3_COUNT``     = G0 + WL0/1/2/3 (counts)
``MORGAN_COUNT``  = G0 + Morgan counts

Nothing in the build path receives a target vector.  The only thing the build
needs from the split is the **row list** of the official Train, used to fit the
typed-WL vocabularies.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Sequence

import numpy as np
import scipy.sparse as sp

import tsb_common as C
import tsb_g0
import tsb_mol
import tsb_morgan
import tsb_wl


G0_PATH = C.CACHE / "g0.npz"
MORGAN_PATH = C.CACHE / "morgan_counts.npz"
MANIFEST_PATH = C.CACHE / "features_manifest.json"
WL_DIR = C.CACHE / "wl"

SCHEMA = "tsb-features-v1"


def wl_paths(suffix: int, round_index: int) -> dict[str, Any]:
    base = WL_DIR / f"suffix{int(suffix)}_r{int(round_index)}"
    return {
        "vocab": base.with_suffix(".vocab.json"),
        "counts": base.with_suffix(".counts.npz"),
        "binary": base.with_suffix(".binary.npz"),
    }


# ---------------------------------------------------------------------------
# block assembly
# ---------------------------------------------------------------------------


def arm_blocks(arm: str) -> list[tuple[str, str]]:
    """``[(block_kind, block_name)]`` in frozen column order."""
    if arm == "G0":
        return [("g0", "g0")]
    if arm == "MORGAN_COUNT":
        return [("g0", "g0"), ("morgan", "morgan")]
    if arm == "WL2_BIN":
        return [("g0", "g0")] + [("wl", f"r{r}:bin") for r in C.WL_ROUNDS["WL2"]]
    if arm == "WL2_COUNT":
        return [("g0", "g0")] + [("wl", f"r{r}:count") for r in C.WL_ROUNDS["WL2"]]
    if arm == "WL3_COUNT":
        return [("g0", "g0")] + [("wl", f"r{r}:count") for r in C.WL_ROUNDS["WL3"]]
    raise KeyError(f"unknown arm {arm!r}")


def _load_block(block_kind: str, block_name: str, suffix: int):
    if block_kind == "g0":
        with np.load(G0_PATH, allow_pickle=False) as handle:
            dense = np.asarray(handle["features"], dtype=np.float32)
        return sp.csr_matrix(dense), list(tsb_g0.COLUMN_NAMES)
    if block_kind == "morgan":
        return tsb_wl.load_csr(MORGAN_PATH), tsb_morgan.column_names()
    if block_kind == "wl":
        round_part, mode = block_name.split(":")
        round_index = int(round_part.lstrip("r"))
        paths = wl_paths(suffix, round_index)
        path = paths["binary"] if mode == "bin" else paths["counts"]
        vocab = C.read_json(paths["vocab"])
        matrix = tsb_wl.load_csr(path)
        return matrix, list(vocab["columns"])
    raise KeyError(block_kind)


def arm_matrix(suffix: int, arm: str) -> tuple[sp.csr_matrix, list[str]]:
    """Assemble one arm for one published split (columns in frozen order)."""
    blocks = arm_blocks(arm)
    matrices = []
    names: list[str] = []
    for kind, name in blocks:
        matrix, columns = _load_block(kind, name, suffix)
        matrices.append(matrix)
        prefix = {"g0": "g0_", "morgan": ""}.get(kind, "")
        names.extend(prefix + column for column in columns)
    matrix = sp.hstack(matrices, format="csr", dtype=np.float32)
    if len(names) != matrix.shape[1]:
        raise AssertionError(f"{arm}: {len(names)} column names for {matrix.shape[1]} columns")
    return matrix, names


def arm_specs() -> dict[str, Any]:
    specs: dict[str, Any] = {}
    for arm in C.ARMS:
        blocks = arm_blocks(arm)
        specs[arm] = {
            "id": arm,
            "blocks": [{"kind": kind, "name": name} for kind, name in blocks],
            "uses_full_train_vocabulary": any(kind == "wl" for kind, _ in blocks),
            "label_used": False,
        }
    return specs


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def build(rebuild: bool = False) -> dict[str, Any]:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    if MANIFEST_PATH.exists() and not rebuild:
        manifest = C.read_json(MANIFEST_PATH)
        _verify_cache(manifest)
        return manifest

    C.ensure_dir(C.CACHE)
    C.ensure_dir(WL_DIR)
    started = time.perf_counter()
    smiles = tsb_data_smiles()
    labels_touched = False  # documented invariant; no label is loaded below

    t0 = time.perf_counter()
    mols, graphs, round_counts = [], [], [list() for _ in range(max(C.WL_ROUNDS["WL3"]) + 1)]
    parse_failed = 0
    for smi in smiles:
        mol, graph, per_round = tsb_mol.molecule_tokens(smi, rounds=max(C.WL_ROUNDS["WL3"]))
        mols.append(mol)
        graphs.append(graph)
        for r, tokens in enumerate(per_round):
            round_counts[r].append(dict(tsb_mol.count_tokens(tokens)))
    tokenization_seconds = time.perf_counter() - t0

    # --- G0 ---------------------------------------------------------------
    t0 = time.perf_counter()
    g0 = tsb_g0.matrix(graphs)
    g0_seconds = time.perf_counter() - t0
    if g0.shape != (len(graphs), tsb_g0.DIM):
        raise AssertionError(f"G0 shape {g0.shape} != {(len(graphs), tsb_g0.DIM)}")
    if not np.all(np.isfinite(g0)):
        raise AssertionError("G0 contains NaN/Inf")
    np.savez_compressed(G0_PATH, features=g0.astype(np.float32))

    # --- Morgan -----------------------------------------------------------
    morgan, morgan_meta = tsb_morgan.build(mols)
    tsb_wl.save_csr(MORGAN_PATH, morgan)

    # --- typed-WL per split ----------------------------------------------
    splits = tsb_data_splits()
    wl_manifest: dict[str, Any] = {}
    train_labels_used = False
    for suffix, split in splits.items():
        train_rows = split["rows"]["train"]
        train_rows_sha = split["rows_sha256"]["train"]
        per_split: dict[str, Any] = {}
        for round_index in range(max(C.WL_ROUNDS["WL3"]) + 1):
            per_molecule = round_counts[round_index]
            aggregate = tsb_wl.aggregate_token_counts(
                per_molecule[row] for row in train_rows
            )
            vocab_tokens = tsb_wl.select_vocabulary(aggregate)
            payload = tsb_wl.vocabulary_payload(
                vocab_tokens,
                aggregate,
                round_index=round_index,
                train_rows_sha256=train_rows_sha,
                n_train_rows=int(train_rows.size),
            )
            paths = wl_paths(suffix, round_index)
            C.write_json(paths["vocab"], payload)
            counts = tsb_wl.build_matrix(per_molecule, vocab_tokens, binary=False)
            binary = tsb_wl.build_matrix(per_molecule, vocab_tokens, binary=True)
            tsb_wl.save_csr(paths["counts"], counts)
            tsb_wl.save_csr(paths["binary"], binary)
            coverage = {}
            for role in ("train", "valid", "test"):
                coverage[role] = tsb_wl.coverage_stats(
                    per_molecule, vocab_tokens, train_rows=split["rows"][role]
                )
            per_split[f"r{round_index}"] = {
                "round": int(round_index),
                "dim": int(counts.shape[1]),
                "vocab_path": C.track_rel(paths["vocab"]),
                "vocab_sha256": C.sha256_file(paths["vocab"]),
                "counts_path": C.track_rel(paths["counts"]),
                "counts_sha256": C.sha256_file(paths["counts"]),
                "binary_path": C.track_rel(paths["binary"]),
                "binary_sha256": C.sha256_file(paths["binary"]),
                "nonzero_column_fraction_train": tsb_wl.nonzero_column_fraction(
                    counts[np.asarray(split["rows"]["train"], dtype=np.int64)]
                ),
                "nonzero_column_fraction_valid": tsb_wl.nonzero_column_fraction(
                    counts[np.asarray(split["rows"]["valid"], dtype=np.int64)]
                ),
                "density_train": tsb_wl.sparse_density(
                    counts[np.asarray(split["rows"]["train"], dtype=np.int64)]
                ),
                "coverage": coverage,
            }
        wl_manifest[str(suffix)] = per_split

    manifest = {
        "schema": SCHEMA,
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty_tracked(),
        "n_molecules": len(smiles),
        "n_parse_failed": int(parse_failed),
        "parse_policy": "strict RDKit parse; failure is a hard error, never a zero vector",
        "label_used": bool(labels_touched),
        "graph_contract": {
            "atom_attributes": list(C.WL_ATOM_ATTRIBUTES),
            "bond_attributes": list(C.WL_BOND_ATTRIBUTES),
            "shared_by": ["G0", "WL2_BIN", "WL2_COUNT", "WL3_COUNT", "MORGAN_COUNT"],
            "chirality_representation": "RDKit CIP code R/S/none (stable under atom renumbering)",
        },
        "g0": {
            "path": C.track_rel(G0_PATH),
            "sha256": C.sha256_file(G0_PATH),
            "dim": int(g0.shape[1]),
            "column_names": list(tsb_g0.COLUMN_NAMES),
            "seconds": float(g0_seconds),
            "spec": tsb_g0.spec(),
        },
        "morgan": {
            "path": C.track_rel(MORGAN_PATH),
            "sha256": C.sha256_file(MORGAN_PATH),
            **morgan_meta,
            "spec": tsb_morgan.spec(),
        },
        "wl": wl_manifest,
        "arms": arm_specs(),
        "timing": {
            "tokenization_seconds": float(tokenization_seconds),
            "total_feature_stage_seconds": float(time.perf_counter() - started),
            "note": (
                "tokenization is a one-time shared cost for all 3 splits and all WL arms; "
                "per-arm assembly time is measured in the train stage"
            ),
        },
        "peak_rss_mb": C.peak_rss_mb(),
    }
    C.write_json(MANIFEST_PATH, manifest)
    return manifest


def tsb_data_smiles() -> list[str]:
    import tsb_data

    return tsb_data.load_smiles()


def tsb_data_splits() -> dict[int, dict[str, Any]]:
    import tsb_data

    return tsb_data.load_splits()


def _verify_cache(manifest: dict[str, Any]) -> None:
    entries: list[tuple[str, str]] = [(manifest["g0"]["path"], manifest["g0"]["sha256"]),
                                      (manifest["morgan"]["path"], manifest["morgan"]["sha256"])]
    for suffix_payload in manifest["wl"].values():
        for entry in suffix_payload.values():
            entries.append((entry["vocab_path"], entry["vocab_sha256"]))
            entries.append((entry["counts_path"], entry["counts_sha256"]))
            entries.append((entry["binary_path"], entry["binary_sha256"]))
    for rel, expected in entries:
        actual = C.sha256_file(C.REPO / rel)
        if actual != expected:
            raise ValueError(f"feature cache drift: {rel} {actual} != {expected}")


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("feature cache missing; run the runner's `features` stage first")
    manifest = C.read_json(MANIFEST_PATH)
    _verify_cache(manifest)
    return manifest


def load_g0() -> np.ndarray:
    with np.load(G0_PATH, allow_pickle=False) as handle:
        return np.asarray(handle["features"], dtype=np.float32)


def load_block_counts(suffix: int, round_index: int, *, binary: bool) -> sp.csr_matrix:
    paths = wl_paths(suffix, round_index)
    return tsb_wl.load_csr(paths["binary"] if binary else paths["counts"])


def load_vocab(suffix: int, round_index: int) -> dict[str, Any]:
    return C.read_json(wl_paths(suffix, round_index)["vocab"])
