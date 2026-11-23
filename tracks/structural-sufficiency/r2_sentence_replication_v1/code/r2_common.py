"""Shared plumbing for R2-SENTENCE-REPLICATION-v1.

This module is the *only* place that reaches into the frozen GWIC-v0 round.  It

* puts the frozen ``graph_word_incidence_composition_pilot_v0/code`` directory on
  ``sys.path`` so ``gwic_model`` / ``gwic_structure`` / ``gwic_tensor`` /
  ``gwic_data`` / ``gwic_common`` are imported byte-for-byte (no copy, no fork),
* pins every frozen artefact by sha256 and refuses to run on a mismatch,
* re-exports the frozen IO/hash/metric helpers so this round cannot drift in
  numeric conventions.

Nothing in the frozen GWIC-v0 directory is written by this round.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
STRUCT = REPO_ROOT / "tracks" / "structural-sufficiency"

GWIC_ROUND = STRUCT / "graph_word_incidence_composition_pilot_v0"
GWIC_CODE = GWIC_ROUND / "code"
GWIC_CONFIG = GWIC_ROUND / "configs" / "gwic.json"
GWIC_RESULTS = GWIC_ROUND / "results"
if str(GWIC_CODE) not in sys.path:
    sys.path.insert(0, str(GWIC_CODE))

import gwic_common as GC  # noqa: E402  (frozen)
import gwic_data as GD  # noqa: E402,F401
import gwic_model as GM  # noqa: E402,F401
import gwic_structure as GS  # noqa: E402,F401
import gwic_tensor as GT  # noqa: E402,F401

CONFIG_PATH = ROUND_DIR / "configs" / "r2.json"
RESULTS_DIR = ROUND_DIR / "results"
ANALYSIS_DIR = ROUND_DIR / "analysis"

# --------------------------------------------------------------------------
# frozen identities (pinned; every one is re-verified at runtime)
# --------------------------------------------------------------------------
RAW_CACHE = GC.RAW_CACHE
RAW_CACHE_SHA256 = "fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c"
TRAIN_LABELS = GC.TRAIN_LABELS
TRAIN_LABELS_SHA256 = "a44d29deba1fd36e885feee27e788d1fda0247c01f69565c2d30e9c7251114c2"
D0_SPLIT = GC.SPLIT_F0
D0_FILE_SHA256 = "675289f6f997008d44fdabb4ab65350459fe5bccc5e1e5703e31e31fd794e845"
D0_IDENTITY = "60ac13c04fb0dab484fc8e7cc28f2a85450794ed3ebeeb0a87d1b41c3532559e"
D0_FIT_ROWS_SHA256 = "7edd3c68d946e9f3f7c5613993530dd44c2371956ee031582e123ef79c8bc6c0"
D0_EVAL_ROWS_SHA256 = "eacce8acde66acb6f0045813f8d6103381012ebab8181b7d314a2b66aa387193"
D1_SPLIT = STRUCT / "structural_path_composition_route_v1" / "data" / "split_train_sub_dev_d1.json"
D1_FILE_SHA256 = "bdcd3f5884c66015aa099bd9c32b45cd6717b0d08b35a6a6c7ef44635fe61fcc"
D1_IDENTITY = "c94e2d4bd92e51ab91119923d61e955f5a2f686fc1751883e858fce88dab6bee"
D1_FIT_ROWS_SHA256 = "b9f74030ee0decc67005e10963e64d7da16fd70d9a6bf9790f4a1c4afcebebbc"
D1_EVAL_ROWS_SHA256 = "342c1de0ece164e9acba17ba17a48151c90235242e77481e41a1a9820b0380a6"
D1_SPLIT_SEED = 20261122
D1_TARGET_HELDOUT = 504

GWIC_FILE_SHA256 = {
    "gwic_common.py": "6d081769940ed00682c95c8a7c8be382f1592763db269c3067fc14bc3f1a97cb",
    "gwic_data.py": "f02863be6e9f37a593e234c4cb7c5e1d9a760088be0c903aad36619a46aa0cef",
    "gwic_structure.py": "97e2e35b272e36043e03a476f58fb6a2edeb3beefc60b15e967c1096c65ecb56",
    "gwic_tensor.py": "f969c1fd47255b3b4b897f540f501671def42ec45ec9515751ca7288204bedd0",
    "gwic_model.py": "66926facafc97935905ba32ae4771f2a15585a34e0749f26cdb2fb8dff187e73",
    "gwic_train.py": "fa1031cee6457864c9b99cac387d97dc97004cf4b79f8e329946a5cb089c855d",
}
GWIC_TRAIN_MODULES = ("gwic_common.py", "gwic_data.py", "gwic_structure.py",
                      "gwic_tensor.py", "gwic_model.py", "gwic_train.py")
GWIC_CODE_SHA256 = "a58f942ebf4050114987c58c2cfeaec75f2cf42c6fc945c21bb45030813294a1"
GWIC_CONFIG_SHA256 = "9ed13ab795e7ae8c9e2c447d74a452fa9b6a8e0c3c6c6990b578e1f963a7e090"

RAW_STRUCT_KEYS = tuple(GC.RAW_STRUCT_KEYS)
ATOM_COLUMNS = int(GC.ATOM_COLUMNS)
BOND_COLUMNS = int(GC.BOND_COLUMNS)

PARAMS_TOTAL = 416177
BATCH_KEYS = (
    "atom_feat", "bond_feat", "bond_index", "atom_scalars", "atom_mol", "bond_mol",
    "graph_scalars", "dist1_atoms", "dist2_atoms", "dist1_seg", "dist2_seg",
    "path_cat", "path_end_type", "path_closure", "path_end_atom", "path_seg",
    "r2_bond", "r2_bucket", "r2_seg", "r2_bucket_seg", "word_root",
    "word_support", "word_internal", "word_scalars", "root_word_ptr",
    "n_molecules", "n_atoms", "n_bonds", "n_words", "rows", "mol_atom_ptr",
)
SENTENCE_BASE_STATE_KEYS = 44

# re-exported frozen helpers (identical numeric/IO conventions)
sha256_file = GC.sha256_file
sha256_json = GC.sha256_json
array_sha256 = GC.array_sha256
jsonable = GC.jsonable
write_json = GC.write_json
read_json = GC.read_json
write_csv = GC.write_csv
rmse = GC.rmse
mae = GC.mae
now_utc = GC.now_utc
seed_everything = GC.seed_everything
module_hashes = GC.module_hashes


class FrozenSourceMismatch(RuntimeError):
    """A pinned GWIC-v0 artefact no longer matches its frozen sha256."""


def git_commit() -> str:
    return GC.git_commit()


def code_sha(paths: Sequence[str | Path]) -> str:
    """sha256 over this round's own source files (sorted by file name)."""
    h = hashlib.sha256()
    for name in sorted(str(p) for p in paths):
        path = Path(name)
        if not path.is_absolute():
            path = CODE_DIR / path
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def frozen_gwic_code_sha() -> str:
    return GC.code_sha(list(GWIC_TRAIN_MODULES))


def load_config() -> dict[str, Any]:
    return read_json(CONFIG_PATH)


def assert_frozen_sources() -> dict[str, Any]:
    """Verify every frozen GWIC-v0 artefact against its pin.  Raises on drift."""
    details: dict[str, Any] = {"sources": {}, "artefacts": {}}
    for name, expected in GWIC_FILE_SHA256.items():
        path = GWIC_CODE / name
        if not path.is_file():
            raise FrozenSourceMismatch(f"frozen GWIC source missing: {path}")
        actual = sha256_file(path)
        details["sources"][name] = {"sha256": actual, "expected": expected,
                                    "ok": actual == expected}
        if actual != expected:
            raise FrozenSourceMismatch(f"{name} sha mismatch: {actual} != {expected}")
    aggregate = frozen_gwic_code_sha()
    details["code_sha256"] = aggregate
    details["code_sha256_expected"] = GWIC_CODE_SHA256
    if aggregate != GWIC_CODE_SHA256:
        raise FrozenSourceMismatch(f"aggregate GWIC code_sha mismatch: {aggregate}")
    cfg_sha = sha256_file(GWIC_CONFIG)
    details["config_sha256"] = cfg_sha
    if cfg_sha != GWIC_CONFIG_SHA256:
        raise FrozenSourceMismatch(f"frozen gwic.json sha mismatch: {cfg_sha}")
    for tag, path, expected in (
            ("raw_cache", RAW_CACHE, RAW_CACHE_SHA256),
            ("train_labels", TRAIN_LABELS, TRAIN_LABELS_SHA256),
            ("split_d0", D0_SPLIT, D0_FILE_SHA256),
            ("split_d1", D1_SPLIT, D1_FILE_SHA256)):
        actual = sha256_file(path)
        details["artefacts"][tag] = {"path": str(Path(path).relative_to(REPO_ROOT)),
                                     "sha256": actual, "expected": expected,
                                     "ok": actual == expected}
        if actual != expected:
            raise FrozenSourceMismatch(f"{tag} sha mismatch: {actual} != {expected}")
    return details


def frozen_config() -> dict[str, Any]:
    """The GWIC-v0 config, hash-checked.  It *is* this round's model config."""
    if sha256_file(GWIC_CONFIG) != GWIC_CONFIG_SHA256:
        raise FrozenSourceMismatch("frozen gwic.json changed")
    cfg = read_json(GWIC_CONFIG)
    if cfg.get("protocol_id") != "gwic-v0-graph-word-incidence-composition-pilot":
        raise FrozenSourceMismatch("unexpected frozen config protocol id")
    spec = cfg["arms"]["SENTENCE_BASE"]
    if bool(spec["use_words"]) or bool(spec["use_geometry"]):
        raise FrozenSourceMismatch("SENTENCE_BASE spec changed in the frozen config")
    return cfg


def split_pin(name: str) -> Mapping[str, Any]:
    if name not in ("D0", "D1"):
        raise KeyError(f"unknown split {name!r}")
    return load_config()["splits"][name]