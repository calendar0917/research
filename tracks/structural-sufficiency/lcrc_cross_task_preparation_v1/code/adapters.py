"""Dataset adapters: raw label-free structures -> canonical molecular graphs.

Two adapters are deliberately independent:

``lipo``
    Reads the committed label-free raw-graph artifact and the frozen Phase 1a
    split.  It builds ``lipo_v1`` atom/bond feature matrices from raw atom and
    bond attributes plus the canonical physical-bond graph.  It never opens a
    label file and never parses the ``exp`` CSV column.

``legacy_zinc``
    Reads the frozen label-free P3 ``MolObjects`` cache and passes through its
    108-d atom, 7-d bond and 33-d global tensors.  The S1/S2 codes are then
    assembled by the new generic pipeline so they can be compared with the old
    implementation.

No adapter performs dataset-level fitting.  Standardisation is a separate
Fit-only operation in ``lcrc_data``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import lcrc_common as C
import molecular_graph as MG
from feature_spec import FeatureSpec

# Raw Lipophilicity attribute conventions (from the committed artifact vocab).
RAW_ATOM_UNK = 15
RAW_ATOM_MASK = 16
CHARGE_VALUES = (-2, -1, 0, 1, 2)
CHARGE_UNK = -99
H_VALUES = (0, 1, 2, 3, 4)
H_UNK = -99
RAW_BOND_CLASSES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
RAW_BOND_UNK = 4
STEREO_BUCKETS = ("NONE", "E", "Z", "OTHER")
STEREO_UNK = 4


# ---------------------------------------------------------------------------
# source loading and verification
# ---------------------------------------------------------------------------


def load_lipo_raw_artifact(path: Path | str = C.LIPO_RAW) -> dict[str, Any]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("schema") != "bjoc-lipo-rawgraph-v1":
        raise ValueError(f"unexpected raw graph schema {payload.get('schema')!r}")
    meta = payload.get("meta", {})
    if meta.get("label_free") is not True:
        raise PermissionError("raw artifact does not certify label_free=true")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("raw artifact rows are missing")
    for i, row in enumerate(rows):
        if "exp" in row:
            raise PermissionError(f"raw row {i} unexpectedly contains a label field")
        if int(row.get("mol_index", -1)) != i:
            raise ValueError(f"raw row {i} has mol_index {row.get('mol_index')!r}")
    return payload


def load_lipo_frozen_split(path: Path | str = C.LIPO_SPLIT) -> dict[str, Any]:
    payload = C.read_json(path)
    if payload.get("protocol_id") != "ssrc-phase1a-lipo-v1":
        raise ValueError("unexpected Lipophilicity split protocol")
    rows = payload["rows"]
    split = {
        "fit": [int(x) for x in rows["fit"]],
        "monitor": [int(x) for x in rows["monitor"]],
        "reserved": [int(x) for x in rows["reserved"]],
    }
    expected = {"fit": 2352, "monitor": 504, "reserved": 504}
    for k, n in expected.items():
        if len(split[k]) != n:
            raise ValueError(f"{k}: {len(split[k])} rows != {n}")
        if len(set(split[k])) != n:
            raise ValueError(f"{k}: duplicate rows")
        actual_hash = C.row_sha256(split[k])
        if actual_hash != C.EXPECTED_ROW_HASHES[k]:
            raise ValueError(f"{k}: row hash {actual_hash} != frozen {C.EXPECTED_ROW_HASHES[k]}")
    for a in ("fit", "monitor", "reserved"):
        for b in ("fit", "monitor", "reserved"):
            if a < b and set(split[a]) & set(split[b]):
                raise ValueError(f"row overlap between {a} and {b}")
    union = set(split["fit"]) | set(split["monitor"]) | set(split["reserved"])
    if len(union) != int(payload.get("n_official_train", -1)):
        raise ValueError(f"union size {len(union)} != n_official_train {payload.get('n_official_train')}")
    scope = payload.get("scope", {})
    if scope.get("labels_read") is not False or scope.get("structure_only") is not True:
        raise PermissionError("split artifact does not certify structure-only scope")
    if scope.get("official_valid_test_labels_read") is not False:
        raise PermissionError("split artifact does not certify official valid/test label exclusion")
    return {
        "split": split,
        "union": sorted(union),
        "payload": payload,
        "hashes": {
            "split_file": C.sha256_file(path),
            "fit": C.row_sha256(split["fit"]),
            "monitor": C.row_sha256(split["monitor"]),
            "reserved": C.row_sha256(split["reserved"]),
        },
    }


def load_lipo_row_manifest(path: Path | str = C.LIPO_ROW_MANIFEST) -> dict[str, Any]:
    payload = C.read_json(path)
    if payload.get("n") != 4200:
        raise ValueError(f"row manifest n={payload.get('n')} != 4200")
    if payload.get("label_free") is not True:
        raise PermissionError("row manifest does not certify label_free=true")
    for key in ("compound_ids", "canonical_smiles_sha256", "scaffold_sha256", "n_atoms", "n_components"):
        if len(payload[key]) != 4200:
            raise ValueError(f"row manifest {key} length mismatch")
    return payload


def load_lipo_official_row_indices(path: Path | str = C.LIPO_OFFICIAL_SPLIT) -> dict[str, Any]:
    """Read only the official row-index arrays; never parse the label tail."""
    fields = C.stream_json_array_fields(path, ("train_rows", "valid_rows", "test_rows"), chunk_size=128)
    for key in ("train_rows", "valid_rows", "test_rows"):
        if key not in fields:
            raise ValueError(f"could not locate {key} in official split artifact")
    return fields


def canonical_smiles_sha256(smiles: str) -> str | None:
    try:
        from rdkit import Chem
    except Exception:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol, canonical=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def source_manifest() -> dict[str, Any]:
    """Hashes and row-level provenance for the label-free Lipophilicity sources.

    This function reads only hashes, row indices and label-free row metadata.
    It does not open ``labels_fit_monitor.json`` or parse the CSV.
    """
    raw_info = C.file_info(C.LIPO_RAW)
    split_info = C.file_info(C.LIPO_SPLIT)
    manifest_info = C.file_info(C.LIPO_ROW_MANIFEST)
    csv_info = C.file_info(C.LIPO_CSV)
    official_info = C.file_info(C.LIPO_OFFICIAL_SPLIT)
    split = load_lipo_frozen_split()
    row_manifest = load_lipo_row_manifest()
    official = load_lipo_official_row_indices()

    union = set(split["union"])
    official_train = set(int(x) for x in official["train_rows"])
    official_valid = set(int(x) for x in official["valid_rows"])
    official_test = set(int(x) for x in official["test_rows"])
    # Scaffold groups from label-free hashes.
    scaffold = row_manifest["scaffold_sha256"]
    groups = {k: {scaffold[i] for i in split["split"][k]} for k in ("fit", "monitor", "reserved")}
    group_overlap = {
        "fit_monitor": len(groups["fit"] & groups["monitor"]),
        "fit_reserved": len(groups["fit"] & groups["reserved"]),
        "monitor_reserved": len(groups["monitor"] & groups["reserved"]),
    }
    version = C.env_info()
    return {
        "generated_utc": C.utcnow(),
        "round_id": C.ROUND_ID,
        "environment": version,
        "sources": {
            "raw_graph": {**raw_info, "schema": "bjoc-lipo-rawgraph-v1", "label_free": True},
            "frozen_split": {**split_info, "protocol_id": "ssrc-phase1a-lipo-v1", "status": split["payload"].get("status")},
            "row_manifest": {**manifest_info, "label_free": True, "n": row_manifest.get("n")},
            "official_split_row_indices_only": {
                **official_info,
                "reader_audit": official["_audit"],
                "train_rows": len(official_train),
                "valid_rows": len(official_valid),
                "test_rows": len(official_test),
            },
            "csv_hash_only": {**csv_info, "parsed": False, "exp_column_read": False},
        },
        "hashes": {
            "raw_graph": raw_info.get("sha256"),
            "frozen_split": split_info.get("sha256"),
            "row_manifest": manifest_info.get("sha256"),
            "csv": csv_info.get("sha256"),
            "official_split": official_info.get("sha256"),
            "row_hashes": split["hashes"],
            "expected": dict(C.EXPECTED_HASHES),
        },
        "split": {
            "counts": {k: len(split["split"][k]) for k in ("fit", "monitor", "reserved")},
            "union_size": len(union),
            "n_official_train_declared": int(split["payload"]["n_official_train"]),
            "overlap": {
                "fit_monitor": len(set(split["split"]["fit"]) & set(split["split"]["monitor"])),
                "fit_reserved": len(set(split["split"]["fit"]) & set(split["split"]["reserved"])),
                "monitor_reserved": len(set(split["split"]["monitor"]) & set(split["split"]["reserved"])),
            },
            "union_equals_official_train_rows": union == official_train,
            "union_minus_official_train": len(union - official_train),
            "official_train_minus_union": len(official_train - union),
            "scaffold_group_counts": {k: len(groups[k]) for k in ("fit", "monitor", "reserved")},
            "scaffold_group_overlap": group_overlap,
            "scaffold_group_counts_declared": split["payload"].get("n_scaffold_groups"),
            "row_1646_in_frozen_union": 1646 in union,
            "row_1646_official_membership": (
                "train" if 1646 in official_train else "valid" if 1646 in official_valid else "test" if 1646 in official_test else "absent"
            ),
        },
        "label_isolation": {
            "labels_fit_monitor_opened": False,
            "reserved_labels_opened": False,
            "official_valid_test_labels_opened": False,
            "csv_exp_column_parsed": False,
            "label_values_parsed": False,
            "forbidden_files": [C.track_rel(p) for p in C.FORBIDDEN_LABEL_FILES],
        },
    }


# ---------------------------------------------------------------------------
# raw Lipophilicity row -> canonical graph
# ---------------------------------------------------------------------------


def _raw_charge_codes(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.int64).reshape(-1)
    if arr.size and arr.max() > 5:
        raise ValueError(f"raw atom_charge bucket {arr.max()} outside 0..5")
    out = np.full(arr.shape, CHARGE_UNK, dtype=np.int64)
    known = arr <= 4
    out[known] = np.asarray(CHARGE_VALUES, dtype=np.int64)[arr[known]]
    return out


def _raw_h_codes(raw: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.int64).reshape(-1)
    if arr.size and arr.max() > 5:
        raise ValueError(f"raw atom_h bucket {arr.max()} outside 0..5")
    out = np.full(arr.shape, H_UNK, dtype=np.int64)
    known = arr <= 4
    out[known] = arr[known]
    return out


def _charge_onehot(charges: np.ndarray, dim: int = 6) -> np.ndarray:
    out = np.zeros((len(charges), dim), dtype=np.float32)
    for i, c in enumerate(charges.tolist()):
        if c == CHARGE_UNK:
            out[i, 5] = 1.0
        else:
            try:
                slot = CHARGE_VALUES.index(int(c))
            except ValueError:
                raise ValueError(f"formal charge {c} outside supported values")
            out[i, slot] = 1.0
    return out


def _h_onehot(hs: np.ndarray, dim: int = 6) -> np.ndarray:
    out = np.zeros((len(hs), dim), dtype=np.float32)
    for i, h in enumerate(hs.tolist()):
        if h == H_UNK:
            out[i, 5] = 1.0
        else:
            if not (0 <= int(h) <= 4):
                raise ValueError(f"hydrogen count {h} outside 0..4")
            out[i, int(h)] = 1.0
    return out


def _degree_onehot(degree: np.ndarray, dim: int = 6) -> np.ndarray:
    out = np.zeros((len(degree), dim), dtype=np.float32)
    for i, d in enumerate(degree.tolist()):
        out[i, min(int(d), 5)] = 1.0
    return out


def _min_ring_onehot(min_ring: np.ndarray, dim: int = 7) -> np.ndarray:
    out = np.zeros((len(min_ring), dim), dtype=np.float32)
    for i, r in enumerate(min_ring.tolist()):
        r = int(r)
        if r <= 0:
            out[i, 0] = 1.0
        elif r == 3:
            out[i, 1] = 1.0
        elif r == 4:
            out[i, 2] = 1.0
        elif r == 5:
            out[i, 3] = 1.0
        elif r == 6:
            out[i, 4] = 1.0
        elif r == 7:
            out[i, 5] = 1.0
        elif r >= 8:
            out[i, 6] = 1.0
        else:
            raise ValueError(f"minimum ring size {r} outside supported range")
    return out


def lipo_atom_features(
    atom_type: np.ndarray,
    charges: np.ndarray,
    hydrogens: np.ndarray,
    analysis: MG.GraphAnalysis,
) -> np.ndarray:
    n = int(analysis.n_atoms)
    if atom_type.shape != (n,) or charges.shape != (n,) or hydrogens.shape != (n,):
        raise ValueError("lipo atom attribute shape mismatch")
    if np.any(atom_type < 0) or np.any(atom_type >= 16):
        raise ValueError("atom type outside lipo_v1 vocabulary")
    degree = analysis.degree
    incident = np.zeros((n, 5), dtype=np.float32)
    for v in range(n):
        for _u, bt, _e in analysis.adj[v]:
            if not (1 <= int(bt) <= 5):
                raise ValueError("bond type outside lipo_v1 vocabulary")
            incident[v, int(bt) - 1] += 1.0
    incident_frac = incident / np.maximum(degree, 1).reshape(-1, 1)
    ring_bond_frac = np.divide(
        analysis.ring_bond_count.astype(np.float32),
        np.maximum(degree.astype(np.float32), 1.0),
    )
    parts = [
        np.eye(16, dtype=np.float32)[atom_type],
        _charge_onehot(charges),
        _h_onehot(hydrogens),
        _degree_onehot(degree),
        analysis.ring_atom.astype(np.float32).reshape(-1, 1),
        _min_ring_onehot(analysis.min_ring_size),
        (analysis.ring_bond_count.astype(np.float32) / 4.0).reshape(-1, 1),
        ring_bond_frac.reshape(-1, 1),
        incident_frac,
        (degree == 1).astype(np.float32).reshape(-1, 1),
        (analysis.ring_bond_count >= 3).astype(np.float32).reshape(-1, 1),
    ]
    out = np.concatenate(parts, axis=1).astype(np.float32)
    assert out.shape == (n, 51), out.shape
    return out


def lipo_bond_features(
    bond_type: np.ndarray,
    stereo: np.ndarray,
    analysis: MG.GraphAnalysis,
) -> np.ndarray:
    m = int(analysis.n_bonds)
    bt = np.asarray(bond_type, dtype=np.int64).reshape(-1)
    st = np.asarray(stereo, dtype=np.int64).reshape(-1)
    if bt.shape != (m,) or st.shape != (m,):
        raise ValueError("lipo bond attribute shape mismatch")
    if np.any(bt < 1) or np.any(bt > 5):
        raise ValueError("bond type outside lipo_v1 vocabulary")
    if np.any(st < 0) or np.any(st > 4):
        raise ValueError("stereo bucket outside 0..4")
    ring = analysis.ring_atom
    # Endpoint ring flags are taken from the canonical bond list.
    exact_one = np.zeros(m, dtype=np.float32)
    both = np.zeros(m, dtype=np.float32)
    for k, (a, b) in enumerate(zip(analysis.bond_u.tolist(), analysis.bond_v.tolist())):
        ra, rb = bool(ring[a]), bool(ring[b])
        exact_one[k] = 1.0 if ra != rb else 0.0
        both[k] = 1.0 if (ra and rb) else 0.0
    parts = [
        np.eye(5, dtype=np.float32)[bt - 1],
        analysis.is_ring_bond.astype(np.float32).reshape(-1, 1),
        exact_one.reshape(-1, 1),
        both.reshape(-1, 1),
        np.eye(5, dtype=np.float32)[st],
    ]
    out = np.concatenate(parts, axis=1).astype(np.float32)
    assert out.shape == (m, 13), out.shape
    return out


def lipo_row_to_graph(
    row: Mapping[str, Any],
    row_index: int,
    spec: FeatureSpec,
) -> MG.MolecularGraph:
    """Build one canonical ``lipo_v1`` graph from a raw label-free row."""
    if "exp" in row:
        raise PermissionError("refusing to convert a row containing a label field")
    atom_class = np.asarray(row["atom_class"], dtype=np.int64).reshape(-1)
    if atom_class.size == 0:
        raise ValueError(f"row {row_index}: empty atom list")
    if np.any(atom_class < 0) or np.any(atom_class > RAW_ATOM_MASK):
        raise ValueError(f"row {row_index}: atom_class outside 0..{RAW_ATOM_MASK}")
    if np.any(atom_class == RAW_ATOM_MASK):
        raise ValueError(f"row {row_index}: atom_class 16 is the SSL MASK sentinel, not a chemical type")
    atom_type = atom_class.copy()
    charges = _raw_charge_codes(row["atom_charge"])
    hydrogens = _raw_h_codes(row["atom_h"])
    if charges.shape != atom_type.shape or hydrogens.shape != atom_type.shape:
        raise ValueError(f"row {row_index}: atom attribute length mismatch")

    edge_index = np.asarray(row["edge_index"], dtype=np.int64)
    if edge_index.shape[0] != 2:
        raise ValueError(f"row {row_index}: edge_index must be [2, 2M]")
    edge_class = np.asarray(row["edge_class"], dtype=np.int64).reshape(-1)
    edge_stereo = np.asarray(row["edge_stereo"], dtype=np.int64).reshape(-1)
    if edge_index.shape[1] != edge_class.shape[0] or edge_class.shape[0] != edge_stereo.shape[0]:
        raise ValueError(f"row {row_index}: directed edge attribute length mismatch")
    if np.any(edge_class < 0) or np.any(edge_class > RAW_BOND_UNK):
        raise ValueError(f"row {row_index}: edge_class outside 0..{RAW_BOND_UNK}")
    if np.any(edge_stereo < 0) or np.any(edge_stereo > STEREO_UNK):
        raise ValueError(f"row {row_index}: edge_stereo outside 0..{STEREO_UNK}")

    # Deduplicate directed records and keep the first occurrence's stereo.
    seen: dict[tuple[int, int], tuple[int, int]] = {}
    for k in range(edge_index.shape[1]):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        if u == v:
            raise ValueError(f"row {row_index}: self-loop")
        if not (0 <= u < atom_class.size and 0 <= v < atom_class.size):
            raise ValueError(f"row {row_index}: edge endpoint outside atom range")
        key = (min(u, v), max(u, v))
        val = (int(edge_class[k]), int(edge_stereo[k]))
        if key in seen and seen[key] != val:
            raise ValueError(f"row {row_index}: inconsistent directed records for {key}: {seen[key]} vs {val}")
        seen[key] = val
    keys = sorted(seen)
    bu = np.asarray([k[0] for k in keys], dtype=np.int64)
    bv = np.asarray([k[1] for k in keys], dtype=np.int64)
    # raw edge_class 0..3 -> 1..4; raw UNK 4 -> 5.
    btype = np.asarray([seen[k][0] + 1 if seen[k][0] <= 3 else 5 for k in keys], dtype=np.int64)
    bstereo = np.asarray([seen[k][1] for k in keys], dtype=np.int64)

    # Temporary graph to obtain ring/degree/path analysis before feature matrices.
    zeros_a = np.zeros((atom_class.size, spec.atom_feat_dim), dtype=np.float32)
    zeros_b = np.zeros((len(bu), spec.bond_feat_dim), dtype=np.float32)
    temp = MG.build_molecular_graph(
        atom_type=atom_type,
        bond_index=np.stack([bu, bv], axis=1),
        bond_type=btype,
        atom_feat=zeros_a,
        bond_feat=zeros_b,
        molecule_id=str(row_index),
        smiles=str(row.get("smiles", "")),
        raw_atom_charge=charges,
        raw_atom_h=hydrogens,
        bond_stereo=bstereo,
    )
    analysis = MG.analyse_graph(temp)
    atom_feat = lipo_atom_features(atom_type, charges, hydrogens, analysis)
    bond_feat = lipo_bond_features(btype, bstereo, analysis)
    graph = MG.build_molecular_graph(
        atom_type=atom_type,
        bond_index=np.stack([bu, bv], axis=1),
        bond_type=btype,
        atom_feat=atom_feat,
        bond_feat=bond_feat,
        molecule_id=str(row_index),
        smiles=str(row.get("smiles", "")),
        raw_atom_charge=charges,
        raw_atom_h=hydrogens,
        bond_stereo=bstereo,
        raw_metadata={"raw_n_components": int(row.get("n_components", -1))},
    )
    if int(row.get("n_components", graph.n_components)) != graph.n_components:
        raise ValueError(
            f"row {row_index}: raw n_components {row.get('n_components')} != graph {graph.n_components}"
        )
    return graph


# ---------------------------------------------------------------------------
# legacy ZINC adapter
# ---------------------------------------------------------------------------


def _ensure_p3_code_on_path() -> None:
    p = str(C.P3_CODE)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_legacy_p3_objects(split: str = "train") -> tuple[list[Any], dict[str, Any]]:
    import torch

    _ensure_p3_code_on_path()
    path = C.P3_TRAIN if split == "train" else C.P3_VALID if split == "valid" else None
    if path is None:
        raise ValueError("legacy objects are available only for train/valid")
    if not path.exists():
        raise FileNotFoundError(path)
    objs = torch.load(path, map_location="cpu", weights_only=False)
    expected_n = 10000 if split == "train" else 1000
    if len(objs) != expected_n:
        raise RuntimeError(f"{path}: expected {expected_n} objects, got {len(objs)}")
    return objs, {
        "path": C.track_rel(path),
        "sha256": C.sha256_file(path),
        "n": len(objs),
        "expected_sha256": C.EXPECTED_ZINC_OBJECT_HASHES.get(split),
        "label_free": True,
    }


def legacy_object_to_graph(obj: Any, spec: FeatureSpec) -> MG.MolecularGraph:
    atom_feat = np.asarray(obj.atom_feat, dtype=np.float32)
    bond_feat = np.asarray(obj.bond_feat, dtype=np.float32)
    bond_index = np.asarray(obj.bond_index, dtype=np.int64)
    bond_type = np.asarray(obj.bond_type, dtype=np.int64)
    global_feat = np.asarray(obj.global_feat, dtype=np.float32)
    if atom_feat.shape[1] != spec.atom_feat_dim:
        raise ValueError(f"legacy atom_feat dim {atom_feat.shape[1]} != {spec.atom_feat_dim}")
    if bond_feat.shape[1] != spec.bond_feat_dim:
        raise ValueError(f"legacy bond_feat dim {bond_feat.shape[1]} != {spec.bond_feat_dim}")
    if global_feat.shape != (spec.global_dim,):
        raise ValueError(f"legacy global_feat shape {global_feat.shape} != ({spec.global_dim},)")
    type_block = atom_feat[:, : spec.n_atom_types]
    if np.any(type_block.sum(axis=1) <= 0):
        raise ValueError("legacy atom type block is not one-hot")
    atom_type = np.argmax(type_block, axis=1).astype(np.int64)
    if atom_type.max(initial=0) >= spec.n_atom_types:
        raise ValueError("legacy atom type outside vocabulary")
    if bond_type.size and (bond_type.min() < 1 or bond_type.max() > spec.n_bond_types):
        raise ValueError(f"legacy bond type outside 1..{spec.n_bond_types}")
    graph = MG.build_molecular_graph(
        atom_type=atom_type,
        bond_index=bond_index,
        bond_type=bond_type,
        atom_feat=atom_feat,
        bond_feat=bond_feat,
        global_feat=global_feat,
        molecule_id=str(getattr(obj, "mol_index", "")),
        smiles=str(getattr(obj, "smiles", "")),
    )
    if graph.n_components != 1:
        raise ValueError(f"legacy molecule {graph.molecule_id}: {graph.n_components} components; schema is connected-only")
    return graph


def load_legacy_norm(path: Path | str = C.LEGACY_NORM) -> dict[str, np.ndarray | str]:
    d = C.read_json(path)
    out: dict[str, Any] = {
        k: np.asarray(d[k], dtype=np.float32)
        for k in ("global_mean", "global_scale", "scalar_mean", "scalar_scale")
    }
    out["source"] = str(d["source"])
    return out


def load_legacy_checkpoint(path: Path | str = C.LEGACY_CKPT) -> dict[str, Any]:
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in ck:
        raise ValueError("legacy checkpoint lacks state_dict")
    return ck


__all__ = [
    "RAW_ATOM_UNK",
    "RAW_ATOM_MASK",
    "CHARGE_VALUES",
    "CHARGE_UNK",
    "H_VALUES",
    "H_UNK",
    "RAW_BOND_CLASSES",
    "RAW_BOND_UNK",
    "STEREO_BUCKETS",
    "STEREO_UNK",
    "load_lipo_raw_artifact",
    "load_lipo_frozen_split",
    "load_lipo_row_manifest",
    "load_lipo_official_row_indices",
    "canonical_smiles_sha256",
    "source_manifest",
    "lipo_atom_features",
    "lipo_bond_features",
    "lipo_row_to_graph",
    "load_legacy_p3_objects",
    "legacy_object_to_graph",
    "load_legacy_norm",
    "load_legacy_checkpoint",
]
