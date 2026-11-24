"""Shared plumbing for task-conditioned structural vocabulary discovery v0.

Read-only reuse:

* ``r2_sentence_replication_v1`` supplies the label-safe D0/D1 loader, frozen
  split identities and the frozen R2 training recipe.
* ``rooted_interaction_cumulant_encoder_pilot_v0/rice_structure.py`` supplies the
  exact rooted radius-2 support enumeration and the root-marked attributed-graph
  canonical certificate.

No target-round source or historical result is written by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
STRUCT = REPO_ROOT / "tracks" / "structural-sufficiency"

R2_DIR = STRUCT / "r2_sentence_replication_v1"
R2_CODE = R2_DIR / "code"
RICE_DIR = STRUCT / "rooted_interaction_cumulant_encoder_pilot_v0"
RICE_CODE = RICE_DIR / "code"

for _p in (R2_CODE, RICE_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import r2_common as RC  # noqa: E402  (pinned frozen plumbing)
import r2_data as RD  # noqa: E402,F401  (pinned split-parameterised loader)
import gwic_data as GD  # noqa: E402  (pinned label-safe raw layer)
import rice_structure as RS  # noqa: E402  (pinned exact enumeration/canonicalisation)

CONFIG_PATH = ROUND_DIR / "configs" / "tc.json"
RESULTS_DIR = ROUND_DIR / "results"
ANALYSIS_DIR = ROUND_DIR / "analysis"

# sha256 of every external source this round imports (checked before use)
SOURCE_PINS = {
    "r2_common.py": (R2_CODE / "r2_common.py",
                     "6d0e0a5f7fc3d901ff89e2e24d04656e6fdf3cf9ae5a25b5ad1928561b9100c5"),
    "r2_data.py": (R2_CODE / "r2_data.py",
                   "a45806a2dcb76a0081a4198ccfcab5b1c6c82b0277c22e581523b9d5e2bfbb7a"),
    "gwic_common.py": (R2_CODE.parent.parent / "graph_word_incidence_composition_pilot_v0"
                       / "code" / "gwic_common.py",
                       "6d081769940ed00682c95c8a7c8be382f1592763db269c3067fc14bc3f1a97cb"),
    "gwic_data.py": (R2_CODE.parent.parent / "graph_word_incidence_composition_pilot_v0"
                     / "code" / "gwic_data.py",
                     "f02863be6e9f37a593e234c4cb7c5e1d9a760088be0c903aad36619a46aa0cef"),
    "gwic_structure.py": (R2_CODE.parent.parent / "graph_word_incidence_composition_pilot_v0"
                          / "code" / "gwic_structure.py",
                          "97e2e35b272e36043e03a476f58fb6a2edeb3beefc60b15e967c1096c65ecb56"),
    "gwic_tensor.py": (R2_CODE.parent.parent / "graph_word_incidence_composition_pilot_v0"
                       / "code" / "gwic_tensor.py",
                       "f969c1fd47255b3b4b897f540f501671def42ec45ec9515751ca7288204bedd0"),
    "rice_structure.py": (RICE_CODE / "rice_structure.py",
                          "cb044e37e3c2bfd90881fca196b3e40fa7e05429bc389747da3957a56269629b"),
}

sha256_file = RC.sha256_file
sha256_json = RC.sha256_json
array_sha256 = RC.array_sha256
jsonable = RC.jsonable
write_json = RC.write_json
read_json = RC.read_json
write_csv = RC.write_csv
rmse = RC.rmse
mae = RC.mae
now_utc = RC.now_utc
seed_everything = RC.seed_everything
module_hashes = RC.module_hashes


class PinMismatch(RuntimeError):
    """An imported read-only source no longer matches its pin."""


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:  # pragma: no cover - only outside a git checkout
        return "unknown"


def code_sha(paths: Sequence[str | Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        if not path.is_absolute():
            path = CODE_DIR / path
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def load_config() -> dict[str, Any]:
    return read_json(CONFIG_PATH)


def assert_source_pins() -> dict[str, str]:
    found: dict[str, str] = {}
    for name, (path, expected) in SOURCE_PINS.items():
        actual = sha256_file(path)
        found[name] = actual
        if actual != expected:
            raise PinMismatch(f"{name}: {actual} != {expected}")
    RC.assert_frozen_sources()
    return found


def split_bundle(split: str):
    if split not in ("D0", "D1"):
        raise KeyError(split)
    bundle = RD.load_bundle(split, "fit")
    return bundle


def split_meta(bundle: Any) -> dict[str, Any]:
    spec = dict(bundle.split)
    return {
        "split": str(split_name(bundle)),
        "identity_sha256": str(spec["identity_sha256"]),
        "file_sha256": str(spec["file_sha256"]),
        "fit_rows_sha256": str(spec["fit_rows_sha256"]),
        "eval_rows_sha256": str(spec["eval_rows_sha256"]),
        "n_fit": int(spec["n_fit"]),
        "n_eval": int(spec["n_eval"]),
        "scaler": dict(bundle.scaler),
        "raw_cache_sha256": str(bundle.hashes["raw_cache"]["sha256"]),
        "labels_sha256": str(bundle.hashes["train_labels"]["sha256"]),
    }


def split_name(bundle: Any) -> str:
    return str(bundle.split["name"])


def scaffold_map(split: str) -> dict[str, str]:
    """The frozen Bemis-Murcko scaffold map stored with the split file."""
    path = RC.D0_SPLIT if split == "D0" else RC.D1_SPLIT
    payload = RC.read_json(path)
    scaf = payload.get("scaffold_by_row")
    if not isinstance(scaf, dict) or len(scaf) != 3360:
        raise ValueError(f"{split}: missing or wrong scaffold map")
    return {str(k): str(v) for k, v in scaf.items()}


def molecule_record(bundle: Any, row: int) -> dict[str, Any]:
    view = GD.molecule_view(bundle.raw, int(row))
    return {
        "row": int(row),
        "n_atoms": int(view["n_atoms"]),
        "n_bonds": int(view["n_bonds"]),
        "atom_feat": np.asarray(view["atom_feat"], np.int64),
        "bond_feat": np.asarray(view["bond_feat"], np.int64).reshape(view["n_bonds"], 3),
        "bond_u": np.asarray(view["bond_u"], np.int64),
        "bond_v": np.asarray(view["bond_v"], np.int64),
        "smiles": str(view["smiles"]),
        "graph_scalars": np.asarray(bundle.graph_scalars[int(row)], np.float64),
    }


def build_struct(bundle: Any, row: int) -> RS.MoleculeStruct:
    return RS.build_molecule(molecule_record(bundle, int(row)))


def signature_key(gran: str, size: int, signature: Sequence[int]) -> str:
    import struct

    return f"{gran}:{int(size)}:" + struct.pack(">10q", *[int(x) for x in signature]).hex()


def signature_bytes(signature: Sequence[int]) -> bytes:
    import struct

    return struct.pack(">10q", *[int(x) for x in signature])


def canonical_key(gran: str, size: int, signature: Sequence[int]) -> str:
    return signature_key(gran, size, signature)


def atom_bond_dims() -> tuple[list[int], list[int]]:
    atom, bond = GD.atom_bond_dims()
    return [int(x) for x in atom], [int(x) for x in bond]


def selected_structure(atom_dims: Sequence[int], bond_dims: Sequence[int],
                       gran: str) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return (atom columns, atom dims, bond columns, bond dims) for a granularity."""
    if gran == "FULL":
        return list(range(9)), list(atom_dims), list(range(3)), list(bond_dims)
    if gran == "COARSE":
        ac = [0, 3, 7]   # atomic number, formal charge, aromaticity
        bc = [0]         # bond type
        return ac, [int(atom_dims[i]) for i in ac], bc, [int(bond_dims[i]) for i in bc]
    raise KeyError(gran)


def as_struct_view(st: RS.MoleculeStruct, atom_cols: Sequence[int],
                   bond_cols: Sequence[int]):
    import copy

    out = copy.copy(st)
    out.atom_feat = np.asarray(st.atom_feat)[:, list(atom_cols)]
    out.bond_feat = np.asarray(st.bond_feat)[:, list(bond_cols)]
    return out


def canonical_signature(st: RS.MoleculeStruct, size: int, gran: str,
                        atom_dims: Sequence[int], bond_dims: Sequence[int]
                        ) -> np.ndarray:
    ac, ad, bc, bd = selected_structure(atom_dims, bond_dims, gran)
    view = st if (list(ac) == list(range(9)) and list(bc) == list(range(3))) else \
        as_struct_view(st, ac, bc)
    return RS.canonical_signature(view, int(size), ad, bd)


def decode_word(entry: Mapping[str, Any], atom_dims: Sequence[int],
                bond_dims: Sequence[int]) -> dict[str, Any]:
    ac, ad, bc, bd = selected_structure(atom_dims, bond_dims, str(entry["gran"]))
    size = int(entry["size"])
    sig = [int(x) for x in entry["signature"]]
    decoded = RS.decode_signature(sig, size, ad, bd)
    # fields are in the selected COARSE columns; annotate their official names
    for i, fields in enumerate(decoded["node_fields"]):
        if str(entry["gran"]) == "COARSE":
            decoded["node_fields"][i] = {"atomic_num": fields[0],
                                         "formal_charge": fields[1],
                                         "is_aromatic": fields[2],
                                         "columns": ac}
        else:
            decoded["node_fields"][i] = {"fields9": fields, "columns": ac}
    for edge in decoded["internal_bonds"]:
        if str(entry["gran"]) == "COARSE":
            edge["bond_type"] = edge.pop("fields")[0]
            edge["columns"] = bc
    decoded["gran"] = str(entry["gran"])
    decoded["canonical_key"] = str(entry["key"])
    return decoded


def safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))