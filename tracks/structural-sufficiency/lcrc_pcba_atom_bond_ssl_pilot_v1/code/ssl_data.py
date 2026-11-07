"""Structure-only PCBA access and the two masked pretext views.

This module is the *only* data entry point of the SSL pretraining stage.  It
deliberately does **not** import ``lr_data`` / ``sg_data`` / ``lr_common`` and
it never opens the frozen assay-label artifact:

* the cache is memory-mapped read-only and only structure arrays are touched
  (``atom_feat``, ``atom_type``, ``atom_ptr``, ``bond_feat``, ``bond_type``,
  ``bond_index``, ``bond_ptr``, ``gids``);
* the official split file is read for the Train/Valid/Test *row ids* only;
* :class:`PathAccessAudit` instruments ``builtins.open`` and ``numpy.load`` so
  the audit can prove which files were opened during pretraining;
* every SSL row id is checked to be an official Train row and the 4096 Probe
  molecules (sorted by ``SHA256("pcba-ssl-probe-v1|" + row id)``) are removed
  from the SSL-fit set.

Masked views (frozen in ``PROTOCOL.md``):

* atom view: zero the full 51-d ``atom_feat`` of ~15% of atoms (>=1 per
  non-empty molecule); the hidden ``atom_type`` is kept in a separate target
  tensor;
* bond view: an *independent* batch with ~15% of physical bonds masked; the
  full 13-d ``bond_feat`` of the bond **and** the full 51-d ``atom_feat`` of
  both endpoints are zeroed (conservative anti-leakage); the hidden
  ``bond_type - 1`` is the target.

The model input dictionary contains no target, no code and no scalar field.
"""

from __future__ import annotations

import builtins
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

import ssl_common as C
from feature_spec import FeatureSpec, load_spec

#: Only these cache arrays are ever opened by the pretraining path.
STRUCTURE_ARRAYS: tuple[str, ...] = (
    "atom_feat",
    "atom_ptr",
    "atom_type",
    "bond_feat",
    "bond_index",
    "bond_ptr",
    "bond_type",
    "gids",
)
#: Arrays that exist in the cache but are *forbidden* for the pretext net.
FORBIDDEN_ARRAYS: tuple[str, ...] = (
    "atom_scalar",
    "global",
    "nbr_code",
    "nbr_ptr",
    "path_cat",
    "path_end",
    "path_end63",
    "path_ptr",
    "path_closure",
    "n_components",
    "n_junctions",
    "n_ring_systems",
)


# ---------------------------------------------------------------------------
# opened-path audit
# ---------------------------------------------------------------------------


class PathAccessAudit:
    """Record every path opened through ``builtins.open`` or ``numpy.load``."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self._open = None
        self._np_load = None

    def __enter__(self) -> "PathAccessAudit":
        original_open = builtins.open
        original_load = np.load
        audit = self

        def traced_open(file, *args, **kwargs):  # noqa: ANN001,ANN002,ANN003
            try:
                audit.paths.append(os.path.abspath(os.fspath(file)))
            except TypeError:
                pass
            return original_open(file, *args, **kwargs)

        def traced_load(file, *args, **kwargs):  # noqa: ANN001,ANN002,ANN003
            try:
                audit.paths.append(os.path.abspath(os.fspath(file)))
            except TypeError:
                pass
            return original_load(file, *args, **kwargs)

        builtins.open = traced_open
        np.load = traced_load
        self._open, self._np_load = original_open, original_load
        return self

    def __exit__(self, *exc) -> bool:  # noqa: ANN002
        builtins.open = self._open
        np.load = self._np_load
        return False

    def unique(self) -> list[str]:
        return sorted(set(self.paths))

    def report(self) -> dict[str, Any]:
        unique = self.unique()
        label_open = [p for p in unique if Path(p).name == C.LABEL_ARTIFACT_NAME]
        return {
            "instrumented": "builtins.open + numpy.load",
            "n_open_calls": len(self.paths),
            "n_unique_paths": len(unique),
            "unique_paths": unique,
            "label_artifact_name": C.LABEL_ARTIFACT_NAME,
            "label_artifact_opened": bool(label_open),
            "label_artifact_open_calls": label_open,
            "forbidden_array_files_opened": [
                p for p in unique if Path(p).stem in FORBIDDEN_ARRAYS
            ],
        }


def assert_no_label_loaders(paths: Iterable[Path]) -> dict[str, Any]:
    """Static check: no pretraining source file *opens* the label artifact.

    Naming the artifact is allowed (the audit has to compare against it); a
    load/open call on it is not.  A line is an offender only when it both
    mentions a banned token and contains a loading primitive.
    """
    banned = (C.LABEL_ARTIFACT_NAME, "pcba_labels", "labels_trainvalid")
    loaders = ("np.load(", "torch.load(", "open(", "load_task(", "Path(")
    offenders: dict[str, list[str]] = {}
    mentions: dict[str, list[str]] = {}
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            hits = [token for token in banned if token in line]
            if not hits:
                continue
            record = f"{path}:{lineno}: {line.strip()}"
            mentions.setdefault(str(path), []).append(record)
            if any(prim in line for prim in loaders) and "LABEL_ARTIFACT_NAME" not in line:
                offenders.setdefault(str(path), []).append(record)
    return {
        "files_checked": [str(p) for p in paths],
        "banned_tokens": list(banned),
        "loader_primitives": list(loaders),
        "mentions": mentions,
        "offenders": offenders,
        "passed": not offenders,
        "note": (
            "ssl_common.py names the artifact path only so that the runtime path audit can "
            "prove it was never opened; no pretraining file loads it."
        ),
    }


# ---------------------------------------------------------------------------
# dataset container
# ---------------------------------------------------------------------------


@dataclass
class Structure:
    """Read-only structure access with the frozen split and Probe partition."""

    spec: FeatureSpec
    arrays: Mapping[str, np.ndarray]
    meta: Mapping[str, Any]
    train_rows: np.ndarray
    valid_rows: np.ndarray
    test_rows: np.ndarray
    probe_rows: np.ndarray
    fit_rows: np.ndarray
    split_partition_sha256: str
    cache_path: Path
    cache_identity: dict[str, Any]
    audit: dict[str, Any] = field(default_factory=dict)

    # -- derived -----------------------------------------------------------
    @property
    def n_molecules(self) -> int:
        return int(self.arrays["gids"].shape[0])

    @property
    def atom_ptr(self) -> np.ndarray:
        return self.arrays["atom_ptr"]

    @property
    def bond_ptr(self) -> np.ndarray:
        return self.arrays["bond_ptr"]

    def n_atoms(self, row: int) -> int:
        p = self.atom_ptr
        return int(p[int(row) + 1] - p[int(row)])

    def n_bonds(self, row: int) -> int:
        p = self.bond_ptr
        return int(p[int(row) + 1] - p[int(row)])

    def subset_counts(self, rows: Sequence[int]) -> dict[str, int]:
        rows = np.asarray(rows, dtype=np.int64)
        ap, bp = self.atom_ptr, self.bond_ptr
        return {
            "rows": int(rows.size),
            "atoms": int((ap[rows + 1] - ap[rows]).sum()),
            "bonds": int((bp[rows + 1] - bp[rows]).sum()),
        }


def probe_rows_from_train(train_rows: Sequence[int]) -> np.ndarray:
    """Deterministic 4096-molecule pretext Probe inside official Train."""
    rows = np.asarray(train_rows, dtype=np.int64)
    keys = [
        hashlib.sha256(f"{C.PROBE_ENTROPY}{int(r)}".encode("utf-8")).hexdigest()
        for r in rows
    ]
    order = sorted(range(rows.size), key=lambda i: (keys[i], int(rows[i])))
    chosen = rows[np.asarray(order[: C.N_PROBE], dtype=np.int64)]
    return np.sort(chosen)


def _cache_stats(path: Path) -> dict[str, Any]:
    files = [p for p in sorted(path.rglob("*")) if p.is_file()]
    return {
        "file_count": len(files),
        "bytes": int(sum(p.stat().st_size for p in files)),
        "names": [p.name for p in files],
    }


def load_structure(*, cache_path: Path | str = C.CACHE_PATH) -> Structure:
    """Open the frozen structure cache read-only and verify its identities."""
    cache_path = Path(cache_path)
    with PathAccessAudit() as loader_audit:
        spec = load_spec(C.SPEC_PATH)
        split_payload = C.read_json(C.SPLIT_PATH)
        split = {
            k: np.asarray(split_payload["rows"][k], dtype=np.int64)
            for k in ("train", "valid", "test")
        }
        arrays = {
            name: np.load(cache_path / f"{name}.npy", mmap_mode="r")
            for name in STRUCTURE_ARRAYS
        }
        meta = C.read_json(cache_path / "meta.json")
        manifest = C.read_json(C.CACHE_MANIFEST)

    files = [p for p in sorted(cache_path.iterdir()) if p.is_file()]
    cache_identity = {
        "path": C.track_rel(cache_path),
        "format": "npy-directory-v1-readonly-mmap",
        "file_count": len(files),
        "bytes": int(sum(p.stat().st_size for p in files)),
        "manifest_file_count": manifest.get("merged", {}).get("cache_files"),
        "manifest_bytes": manifest.get("merged", {}).get("cache_bytes"),
        "manifest_cache_sha256": manifest.get("merged", {}).get("cache_sha256"),
        "feature_spec_hash": spec.hash(),
        "meta": dict(meta),
    }

    # -- gates ------------------------------------------------------------
    if spec.hash() != C.SOURCE["feature_spec_hash"]:
        raise RuntimeError(f"feature spec hash {spec.hash()} != frozen {C.SOURCE['feature_spec_hash']}")
    if spec.name != "lipo_v1":
        raise RuntimeError(f"unexpected feature spec {spec.name}")
    if (spec.atom_feat_dim, spec.bond_feat_dim) != (C.ATOM_FEAT_DIM, C.BOND_FEAT_DIM):
        raise RuntimeError("unexpected atom/bond feature widths")
    if (spec.n_atom_types, spec.n_bond_types) != (C.N_ATOM_CLASSES, C.N_BOND_CLASSES):
        raise RuntimeError("unexpected atom/bond vocabularies")
    if str(split_payload.get("partition_sha256")) != C.SOURCE["split_partition_sha256"]:
        raise RuntimeError("split partition hash drifted")
    sizes = {k: int(v.size) for k, v in split.items()}
    if sizes != {"train": C.N_TRAIN, "valid": C.N_VALID, "test": C.N_TEST}:
        raise RuntimeError(f"split sizes {sizes} differ from the official counts")
    if str(meta.get("feature_spec_hash")) != spec.hash():
        raise RuntimeError("cache feature-spec hash does not match the frozen spec")
    if int(meta.get("n_molecules", -1)) != C.N_MOLECULES:
        raise RuntimeError(f"cache n_molecules {meta.get('n_molecules')} != {C.N_MOLECULES}")
    if "labels_present" in meta and bool(meta["labels_present"]):
        raise RuntimeError("structure cache unexpectedly reports labels")
    if not np.array_equal(np.asarray(arrays["gids"]), np.arange(C.N_MOLECULES)):
        raise RuntimeError("cache rows are not the official row order")
    for key, expected in (("n_atoms_total", 11_373_137), ("n_bonds_total", 12_309_186)):
        if int(meta.get(key, -1)) != int(expected):
            raise RuntimeError(f"cache meta {key}={meta.get(key)} != frozen {expected}")
    if int(arrays["atom_feat"].shape[1]) != C.ATOM_FEAT_DIM:
        raise RuntimeError("atom_feat width is not frozen lipo_v1")
    if int(arrays["bond_feat"].shape[1]) != C.BOND_FEAT_DIM:
        raise RuntimeError("bond_feat width is not frozen lipo_v1")
    for a, b in (
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ):
        if np.intersect1d(split[a], split[b]).size:
            raise RuntimeError(f"official split rows overlap: {a}/{b}")

    train_rows, valid_rows, test_rows = split["train"], split["valid"], split["test"]
    probe = probe_rows_from_train(train_rows)
    fit = np.setdiff1d(train_rows, probe, assume_unique=True)
    if probe.size != C.N_PROBE:
        raise RuntimeError(f"probe size {probe.size} != {C.N_PROBE}")
    if np.intersect1d(probe, fit).size:
        raise RuntimeError("probe and SSL-fit rows overlap")
    if not np.array_equal(np.sort(np.concatenate([probe, fit])), np.sort(train_rows)):
        raise RuntimeError("probe + SSL-fit is not exactly the official Train split")

    audit = {
        "schema": "lcrc-pcba-atom-bond-ssl-data-audit-v1",
        "round_id": C.ROUND_ID,
        "dataset": C.DATASET,
        "cache": cache_identity,
        "feature_spec": {
            "name": spec.name,
            "hash": spec.hash(),
            "atom_feat_dim": int(spec.atom_feat_dim),
            "bond_feat_dim": int(spec.bond_feat_dim),
            "n_atom_types": int(spec.n_atom_types),
            "n_bond_types": int(spec.n_bond_types),
            "atom_type_names": list(spec.atom_type_names),
            "bond_type_names": list(spec.bond_type_names),
        },
        "split": {
            "partition_sha256": str(split_payload.get("partition_sha256")),
            "sizes": sizes,
            "train_rows_sha256": C.sha256_int64(train_rows),
            "valid_rows_sha256": C.sha256_int64(valid_rows),
            "test_rows_sha256": C.sha256_int64(test_rows),
            "train_row_min": int(train_rows.min()),
            "train_row_max": int(train_rows.max()),
        },
        "probe": {
            "rule": 'sorted by SHA256("pcba-ssl-probe-v1|" + decimal molecule row id)',
            "n": int(probe.size),
            "rows_sha256": C.sha256_int64(probe),
            "first_rows": [int(x) for x in probe[:8]],
            "visit_index": C.PROBE_VISIT_INDEX,
            "used_for_parameter_updates": False,
        },
        "ssl_fit": {
            "n": int(fit.size),
            "rows_sha256": C.sha256_int64(fit),
            "all_rows_are_official_train": bool(np.isin(fit, train_rows).all()),
            "intersection_with_valid": int(np.intersect1d(fit, valid_rows).size),
            "intersection_with_test": int(np.intersect1d(fit, test_rows).size),
            "subset_counts": {
                "atoms": int((arrays["atom_ptr"][fit + 1] - arrays["atom_ptr"][fit]).sum()),
                "bonds": int((arrays["bond_ptr"][fit + 1] - arrays["bond_ptr"][fit]).sum()),
            },
        },
        "loader_file_access": loader_audit.report(),
        "forbidden_cache_arrays_never_opened": [
            name for name in FORBIDDEN_ARRAYS
            if name not in {Path(p).stem for p in loader_audit.unique()}
        ],
        "label_artifact_path_recorded_for_audit_only": C.track_rel(C.LABEL_ARTIFACT_PATH),
        "label_artifact_opened": bool(loader_audit.report()["label_artifact_opened"]),
        "labels_read": False,
        "test_structure_used_in_pretraining": False,
        "valid_structure_used_in_pretraining": False,
    }
    return Structure(
        spec=spec,
        arrays=arrays,
        meta=meta,
        train_rows=train_rows,
        valid_rows=valid_rows,
        test_rows=test_rows,
        probe_rows=probe,
        fit_rows=fit,
        split_partition_sha256=str(split_payload.get("partition_sha256")),
        cache_path=cache_path,
        cache_identity=cache_identity,
        audit=audit,
    )


# ---------------------------------------------------------------------------
# minimal structure-only collate (no S1/S2, no scalar, no global)
# ---------------------------------------------------------------------------


def _ranges(ptr: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = (ptr[rows + 1] - ptr[rows]).astype(np.int64)
    idx = np.concatenate(
        [np.arange(ptr[r], ptr[r + 1], dtype=np.int64) for r in rows]
    ) if rows.size else np.zeros(0, dtype=np.int64)
    batch = np.repeat(np.arange(rows.size, dtype=np.int64), counts)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64) if rows.size else np.zeros(0, np.int64)
    return idx, batch, offsets


def collate_structure(st: Structure, rows: Sequence[int]) -> dict[str, Any]:
    """Batch only the tensors the pretext encoders are allowed to see."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or rows.size == 0:
        raise ValueError("collate_structure needs a non-empty 1-D row set")
    if np.any(rows < 0) or np.any(rows >= st.n_molecules):
        raise ValueError("row outside cache")
    if len(set(int(x) for x in rows.tolist())) != rows.size:
        raise ValueError("duplicate row in batch")
    B = int(rows.size)
    arr = st.arrays
    atom_idx, atom_batch, atom_off = _ranges(st.atom_ptr, rows)
    bond_idx, bond_batch, bond_off = _ranges(st.bond_ptr, rows)
    bi = np.asarray(arr["bond_index"])[bond_idx].astype(np.int64) if bond_idx.size else np.zeros((0, 2), np.int64)
    if bi.size:
        bi = bi + atom_off[bond_batch].reshape(-1, 1)

    def _t(value, dtype):
        return torch.from_numpy(np.ascontiguousarray(np.asarray(value, dtype=dtype)))

    batch = {
        "n_graphs": B,
        "atom_feat": _t(np.asarray(arr["atom_feat"])[atom_idx], np.float32),
        "atom_batch": _t(atom_batch, np.int64),
        "atom_off": _t(atom_off, np.int64),
        "atom_counts": _t((st.atom_ptr[rows + 1] - st.atom_ptr[rows]), np.int64),
        "bond_feat": _t(np.asarray(arr["bond_feat"])[bond_idx], np.float32),
        "bond_index": _t(bi, np.int64),
        "bond_batch": _t(bond_batch, np.int64),
        "bond_off": _t(bond_off, np.int64),
        "bond_counts": _t((st.bond_ptr[rows + 1] - st.bond_ptr[rows]), np.int64),
        "atom_type": _t(np.asarray(arr["atom_type"])[atom_idx], np.int64),
        "bond_type": _t(np.asarray(arr["bond_type"])[bond_idx], np.int64),
    }
    return batch


#: Keys the pretext network is allowed to read.  Everything else in the batch
#: dictionary is either bookkeeping or a target and is never passed to a module.
MODEL_INPUT_KEYS: tuple[str, ...] = ("atom_feat", "bond_feat", "bond_index")


def model_input(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """The exact dictionary handed to the pretext encoder."""
    return {k: batch[k] for k in MODEL_INPUT_KEYS}


# ---------------------------------------------------------------------------
# mask sampling (stateless, hash-derived)
# ---------------------------------------------------------------------------


def mask_rng(view: str, row_id: int, visit_index: int, seed: int = C.SSL_SEED) -> np.random.Generator:
    """Hash-derived generator: ``SHA256(tag|seed|row|visit)`` -> 128-bit seed."""
    digest = C.mask_seed_bytes(view, int(seed), int(row_id), int(visit_index))
    return np.random.default_rng(int.from_bytes(digest[:16], "big"))


def sample_molecule_mask(
    *,
    row_id: int,
    visit_index: int,
    n_atoms: int,
    n_bonds: int,
    seed: int = C.SSL_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen per-molecule mask: ``max(1, round_half_up(0.15 n))`` objects.

    The two views draw from **independent** hash-derived generators, so an atom
    mask and a bond mask are statistically independent for the same molecule
    and visit.
    """
    n_atoms, n_bonds = int(n_atoms), int(n_bonds)
    if n_atoms > 0:
        rng_a = mask_rng(C.MASK_ATOM_TAG, row_id, visit_index, seed)
        k_a = C.n_masked(n_atoms)
        atoms = np.sort(rng_a.choice(n_atoms, size=k_a, replace=False)).astype(np.int64)
    else:  # pragma: no cover - PCBA has no atom-free molecule
        atoms = np.zeros(0, dtype=np.int64)
    if n_bonds > 0:
        rng_b = mask_rng(C.MASK_BOND_TAG, row_id, visit_index, seed)
        k_b = C.n_masked(n_bonds)
        bonds = np.sort(rng_b.choice(n_bonds, size=k_b, replace=False)).astype(np.int64)
    else:
        bonds = np.zeros(0, dtype=np.int64)
    return atoms, bonds


@dataclass
class PretextBatch:
    """Two independent masked views plus targets kept outside the inputs."""

    atom_view: dict[str, Any]
    bond_view: dict[str, Any]
    atom_ids: torch.Tensor          # global atom offsets inside the atom view
    atom_targets: torch.Tensor      # hidden atom_type (0..15)
    bond_ids: torch.Tensor          # global physical-bond offsets inside the bond view
    bond_targets: torch.Tensor      # hidden bond_type - 1 (0..4)
    atom_mask_by_molecule: list[np.ndarray]
    bond_mask_by_molecule: list[np.ndarray]
    rows: np.ndarray
    visit_index: int

    def visible_atom_input(self) -> torch.Tensor:
        return self.atom_view["atom_feat"]

    def visible_bond_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.bond_view["atom_feat"], self.bond_view["bond_feat"]

    def counts(self) -> dict[str, int]:
        return {
            "molecules": int(self.rows.size),
            "atoms": int(self.atom_view["atom_feat"].shape[0]),
            "bonds": int(self.bond_view["bond_feat"].shape[0]),
            "masked_atoms": int(self.atom_ids.numel()),
            "masked_bonds": int(self.bond_ids.numel()),
        }


def build_pretext_batch(
    st: Structure,
    rows: Sequence[int],
    visit_index: int,
    *,
    seed: int = C.SSL_SEED,
) -> PretextBatch:
    """Apply the two frozen masked views to one structure-only batch."""
    rows = np.asarray(rows, dtype=np.int64)
    base = collate_structure(st, rows)
    atom_off = base["atom_off"].numpy()
    bond_off = base["bond_off"].numpy()
    atom_counts = base["atom_counts"].numpy()
    bond_counts = base["bond_counts"].numpy()

    atom_ids: list[np.ndarray] = []
    bond_ids: list[np.ndarray] = []
    atom_targets: list[np.ndarray] = []
    bond_targets: list[np.ndarray] = []
    atom_masks: list[np.ndarray] = []
    bond_masks: list[np.ndarray] = []
    atom_type = base["atom_type"].numpy()
    bond_type = base["bond_type"].numpy()

    for i, row in enumerate(rows):
        a_local, b_local = sample_molecule_mask(
            row_id=int(row),
            visit_index=int(visit_index),
            n_atoms=int(atom_counts[i]),
            n_bonds=int(bond_counts[i]),
            seed=int(seed),
        )
        a_global = a_local + int(atom_off[i])
        b_global = b_local + int(bond_off[i])
        atom_masks.append(a_local)
        bond_masks.append(b_local)
        atom_ids.append(a_global)
        bond_ids.append(b_global)
        atom_targets.append(atom_type[a_global])
        bond_targets.append(bond_type[b_global] - 1)

    atom_ids_t = torch.from_numpy(np.concatenate(atom_ids) if atom_ids else np.zeros(0, np.int64))
    bond_ids_t = torch.from_numpy(np.concatenate(bond_ids) if bond_ids else np.zeros(0, np.int64))
    atom_targets_t = torch.from_numpy(
        np.concatenate(atom_targets).astype(np.int64) if atom_targets else np.zeros(0, np.int64)
    )
    bond_targets_t = torch.from_numpy(
        np.concatenate(bond_targets).astype(np.int64) if bond_targets else np.zeros(0, np.int64)
    )
    if bond_targets_t.numel():
        if int(bond_targets_t.min()) < 0 or int(bond_targets_t.max()) >= C.N_BOND_CLASSES:
            raise RuntimeError("bond target outside 0..4")

    atom_view = dict(base)
    atom_view_feat = base["atom_feat"].clone()
    atom_view_feat[atom_ids_t] = 0.0
    atom_view["atom_feat"] = atom_view_feat

    bond_view = dict(base)
    bond_view_feat = base["atom_feat"].clone()
    masked_bond_feat = base["bond_feat"].clone()
    if bond_ids_t.numel():
        endpoints = base["bond_index"][bond_ids_t].reshape(-1)
        bond_view_feat[endpoints] = 0.0
        masked_bond_feat[bond_ids_t] = 0.0
    bond_view["atom_feat"] = bond_view_feat
    bond_view["bond_feat"] = masked_bond_feat

    return PretextBatch(
        atom_view=atom_view,
        bond_view=bond_view,
        atom_ids=atom_ids_t,
        atom_targets=atom_targets_t.long(),
        bond_ids=bond_ids_t,
        bond_targets=bond_targets_t.long(),
        atom_mask_by_molecule=atom_masks,
        bond_mask_by_molecule=bond_masks,
        rows=rows,
        visit_index=int(visit_index),
    )


# ---------------------------------------------------------------------------
# class priors (SSL-fit only, labels never involved)
# ---------------------------------------------------------------------------


def class_priors(st: Structure, *, fit_rows: Sequence[int] | None = None) -> dict[str, Any]:
    """Atom/bond class frequencies of the SSL-fit molecules.

    Computed as ``all rows - probe rows`` so the 11.4M-atom cache is scanned
    once without materialising a second index array.
    """
    rows = st.fit_rows if fit_rows is None else np.asarray(fit_rows, dtype=np.int64)
    probe = st.probe_rows
    atom_type = np.asarray(st.arrays["atom_type"])
    bond_type = np.asarray(st.arrays["bond_type"])

    def _counts(ptr_name: str, values: np.ndarray, n_classes: int, offset: int) -> np.ndarray:
        ptr = np.asarray(st.arrays[ptr_name])
        total = np.bincount(values.astype(np.int64), minlength=n_classes + offset)[offset : offset + n_classes]
        probe_counts = np.zeros(n_classes, dtype=np.int64)
        for r in probe:
            lo, hi = int(ptr[r]), int(ptr[r + 1])
            if hi > lo:
                probe_counts += np.bincount(
                    values[lo:hi].astype(np.int64), minlength=n_classes + offset
                )[offset : offset + n_classes]
        if int(rows.size) == int(st.train_rows.size):
            fit_counts = total - probe_counts
        else:  # pragma: no cover - diagnostic path only
            fit_counts = np.zeros(n_classes, dtype=np.int64)
            for r in rows:
                lo, hi = int(ptr[r]), int(ptr[r + 1])
                if hi > lo:
                    fit_counts += np.bincount(
                        values[lo:hi].astype(np.int64), minlength=n_classes + offset
                    )[offset : offset + n_classes]
        return fit_counts.astype(np.int64)

    atom_counts = _counts("atom_ptr", atom_type, C.N_ATOM_CLASSES, 0)
    bond_counts = _counts("bond_ptr", bond_type, C.N_BOND_CLASSES, 1)

    out: dict[str, Any] = {"fit_rows": int(rows.size), "source": "SSL-fit molecules only (Train minus Probe)"}
    for name, counts, n_classes, names in (
        ("atom", atom_counts, C.N_ATOM_CLASSES, st.spec.atom_type_names),
        ("bond", bond_counts, C.N_BOND_CLASSES, st.spec.bond_type_names),
    ):
        total = int(counts.sum())
        p = counts.astype(np.float64) / max(total, 1)
        nz = p[p > 0]
        entropy = float(-(nz * np.log(nz)).sum())
        out[name] = {
            "n": total,
            "counts": [int(x) for x in counts],
            "class_names": list(names),
            "frequencies": [float(x) for x in p],
            "entropy_nats": entropy,
            "prior_ce_nats": entropy,
            "majority_class": int(np.argmax(p)),
            "majority_fraction": float(p.max()),
        }
    return out


def load_train_norm(st: Structure) -> dict[str, Any]:
    """Refit the frozen Train-only normalisation from the cache (label-free).

    The seed-0 LCRC builder needs a ``norm`` mapping to fill its buffers.  The
    refit uses ``atom_scalar`` / ``global`` only as *builder inputs*; they are
    never handed to the pretext network.  The refit is accepted only if its
    fingerprint equals the frozen value, so the model initialisation is the
    historical one.
    """
    import bp_data as BD

    with PathAccessAudit() as norm_audit:
        cache = {
            "n_molecules": int(st.n_molecules),
            "atom_ptr": st.arrays["atom_ptr"],
            "atom_scalar": np.load(st.cache_path / "atom_scalar.npy", mmap_mode="r"),
            "global": np.load(st.cache_path / "global.npy", mmap_mode="r"),
        }
        norm = BD.fit_norm_rows(cache, st.spec, st.train_rows, source="pcba_official_train_only")
    if norm["norm_fingerprint"] != C.SOURCE["norm_fingerprint"]:
        raise RuntimeError(
            f"Train-only norm fingerprint {norm['norm_fingerprint']} != frozen {C.SOURCE['norm_fingerprint']}"
        )
    audit = norm_audit.report()
    if audit["label_artifact_opened"]:
        raise RuntimeError("the label artifact was opened while fitting the frozen norm")
    norm_audit_meta = {
        "purpose": "seed-0 LCRC model builder only (buffers); never a pretext network input",
        "norm_fingerprint": norm["norm_fingerprint"],
        "fit_rows": int(norm["fit_rows"]),
        "fit_rows_sha256": norm["fit_rows_sha256"],
        "file_access": audit,
    }
    norm = dict(norm)
    norm["ssl_norm_audit"] = norm_audit_meta
    return norm


def prior_ce(targets: np.ndarray, frequencies: Sequence[float]) -> float:
    """Cross-entropy of targets under a fixed marginal prior."""
    t = np.asarray(targets, dtype=np.int64)
    f = np.asarray(frequencies, dtype=np.float64)
    if t.size == 0:
        return float("nan")
    p = np.clip(f[t], 1e-12, 1.0)
    return float(-np.log(p).mean())


__all__ = [
    "FORBIDDEN_ARRAYS",
    "MODEL_INPUT_KEYS",
    "PathAccessAudit",
    "PretextBatch",
    "STRUCTURE_ARRAYS",
    "Structure",
    "assert_no_label_loaders",
    "build_pretext_batch",
    "class_priors",
    "collate_structure",
    "load_structure",
    "load_train_norm",
    "mask_rng",
    "model_input",
    "prior_ce",
    "probe_rows_from_train",
    "sample_molecule_mask",
]
