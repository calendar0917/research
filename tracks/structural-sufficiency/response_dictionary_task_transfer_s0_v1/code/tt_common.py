"""Shared paths, frozen identities and small helpers for the task-transfer round.

This round is *read-only* with respect to the three source rounds:
``response_dictionary_small_s0_v1``, ``response_dictionary_lipo_followup_s0_v1``
and ``response_dictionary_explanation_check_s0_v1``.  The parent code is put on
``sys.path`` only to reuse the exact fragment identity / cut definitions and the
frozen file loaders.  No parent file is written or patched.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND_DIR = CODE_DIR.parent
REPO_ROOT = ROUND_DIR.parents[2]
SS_DIR = ROUND_DIR.parent

S0_DIR = SS_DIR / "response_dictionary_small_s0_v1"
S0_CODE = S0_DIR / "code"
S0_RESULTS = S0_DIR / "results"
FU_DIR = SS_DIR / "response_dictionary_lipo_followup_s0_v1"
FU_RESULTS = FU_DIR / "results"
EC_DIR = SS_DIR / "response_dictionary_explanation_check_s0_v1"
EC_RESULTS = EC_DIR / "results"
OLD_DIR = SS_DIR / "lcrc_lipo_edit_supervision_public122_s0_v1"

if str(S0_CODE) not in sys.path:
    sys.path.insert(0, str(S0_CODE))

import rd_common as RC  # noqa: E402  (parent round, read-only)
import rd_fragment as RF  # noqa: E402

CONFIG_PATH = ROUND_DIR / "configs" / "task_transfer_s0.json"
PROTOCOL_PATH = ROUND_DIR / "PROTOCOL.md"
ANALYSIS_DIR = ROUND_DIR / "analysis"
STRUCTURE_DIR = ANALYSIS_DIR / "structure"

DATASET = "ogbg-mollipo"
TASK = "lipo"
SPLITS = ("train", "valid", "test")
MODELS = ("RESP", "JOINT", "ADD", "D_RESPONSE", "D_LATENT", "D_FREQ")
DICTIONARIES = ("D_RESPONSE", "D_LATENT", "D_FREQ")
ZERO = "ZERO"
TABLE = "TRAIN_TRANSFORM_MEAN"

#: Expected prediction sources and their selected epochs (frozen objects).
NATIVE_PREDICTION = {
    "RESP": S0_RESULTS / "lipo_RESP" / "predictions_best.npz",
    "ADD": S0_RESULTS / "lipo_ADD" / "predictions_best.npz",
    "JOINT": FU_RESULTS / "lipo_JOINT" / "predictions_best.npz",
}
NATIVE_EPOCH = {"RESP": 63, "ADD": 91, "JOINT": 85}
DICT_PREDICTION = EC_RESULTS / "projection_predictions.npz"
DICT_VERSION = "HP_CERTIFIED"

OLD_PAIRS_CSV = OLD_DIR / "analysis" / "pairs.csv"
OLD_MAPPINGS_JSON = OLD_DIR / "analysis" / "pair_mappings.json"
OLD_FOLLOWUP_PAIRS_CSV = FU_DIR / "analysis" / "edit_pairs.csv"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path | str) -> str:
    return RC.sha256_file(Path(path))


def sha256_json(payload: Any) -> str:
    return RC.SQ.sha256_json(payload)


def write_json(path: Path | str, payload: Any) -> Path:
    return RC.write_json(Path(path), payload)


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> Path:
    return RC.write_csv(Path(path), rows, columns)


def protocol_hash() -> str:
    payload = {
        "config": load_config(),
        "protocol_md_sha256": sha256_file(PROTOCOL_PATH) if PROTOCOL_PATH.is_file() else None,
    }
    return sha256_json(payload)


def source_hashes() -> dict[str, str]:
    files = [
        PROTOCOL_PATH,
        CONFIG_PATH,
        CODE_DIR / "tt_common.py",
        CODE_DIR / "tt_catalogue.py",
        CODE_DIR / "tt_evaluate.py",
        CODE_DIR / "tt_run.py",
    ]
    return {str(p.relative_to(ROUND_DIR)): sha256_file(p) for p in files if p.is_file()}


def load_payload() -> dict[str, Any]:
    return RC.dataset_cache(DATASET)


def load_fragment_payload() -> dict[str, Any]:
    return RC.fragment_cache(DATASET)


def official_split(payload: Mapping[str, Any]) -> np.ndarray:
    out = np.empty(len(payload["smiles"]), dtype=object)
    for split in SPLITS:
        out[np.asarray(payload[f"{split}_idx"], dtype=np.int64)] = split
    return out


def canonical_isomeric_smiles(payload: Mapping[str, Any]) -> list[str]:
    from rdkit import Chem

    out: list[str] = []
    for text in payload["smiles"]:
        mol = Chem.MolFromSmiles(str(text))
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {text!r}")
        out.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    return out


def stereo_tokens(smiles: str) -> bool:
    """The old catalogue's stereo eligibility test (no @, / or \\ tokens)."""
    return ("@" in smiles) or ("/" in smiles) or ("\\" in smiles)


def row_graph(payload: Mapping[str, Any], row: int) -> dict[str, np.ndarray]:
    """The physical OGB graph of one row, with local bond indices."""
    a0 = int(payload["node_ptr"][row])
    a1 = int(payload["node_ptr"][row + 1])
    b0 = int(payload["bond_ptr"][row])
    b1 = int(payload["bond_ptr"][row + 1])
    return {
        "atom_feat": np.asarray(payload["atom_feat"][a0:a1], dtype=np.int16),
        "bond_u": np.asarray(payload["bond_u"][b0:b1], dtype=np.int32),
        "bond_v": np.asarray(payload["bond_v"][b0:b1], dtype=np.int32),
        "bond_feat": np.asarray(payload["bond_feat"][b0:b1], dtype=np.int16),
    }


def exact_fragment(
    atom_feat: np.ndarray,
    bond_u: np.ndarray,
    bond_v: np.ndarray,
    bond_feat: np.ndarray,
    atoms: Iterable[int],
    port_global: int,
) -> RF.Fragment:
    atoms_arr = np.asarray(sorted(int(a) for a in atoms), dtype=np.int64)
    return RF._induce(
        atoms_arr, atom_feat, bond_u, bond_v, bond_feat, port_global=int(port_global)
    )


def cut_feat(bond_feat: np.ndarray, edge: int) -> tuple[int, int, int]:
    return tuple(int(x) for x in np.asarray(bond_feat)[int(edge)])


def component_id_of_rows(rows: Iterable[int]) -> int:
    return int(min(int(r) for r in rows))


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def old_pair_catalogue() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The 823-row catalogue and its stored full atom mappings (read-only)."""
    import csv

    rows = list(csv.DictReader(Path(OLD_PAIRS_CSV).open(newline="", encoding="utf-8")))
    mappings = _read_json(OLD_MAPPINGS_JSON)["pairs"]
    if len(rows) != len(mappings):
        raise ValueError("old pairs.csv and pair_mappings.json disagree on length")
    by_id = {str(p["pair_id"]): p for p in mappings}
    merged = []
    for row in rows:
        mapping = by_id[str(row["pair_id"])]
        merged.append({**row, **mapping})
    return merged, mappings


def old_split_mismatch_count(payload: Mapping[str, Any]) -> int:
    """Reproduce the documented 331 stored-split mismatches (read-only)."""
    import csv

    official = official_split(payload)
    rows = list(csv.DictReader(Path(OLD_PAIRS_CSV).open(newline="", encoding="utf-8")))
    bad = 0
    for row in rows:
        a = int(row["row_a"])
        b = int(row["row_b"])
        if (row["split_a"], row["split_b"]) != (str(official[a]), str(official[b])):
            bad += 1
    return bad


def selected_views(payload: Mapping[str, Any], frag_payload: Mapping[str, Any]) -> dict[int, set[tuple[str, tuple[int, int, int]]]]:
    """The fixed <=4 selected views of every official row, as (C key, cut bond)."""
    keys = frag_payload["frag_key"]
    out: dict[int, set[tuple[str, tuple[int, int, int]]]] = {}
    for split in SPLITS:
        rows = np.asarray(frag_payload[f"{split}_mol_row_id"], dtype=np.int64)
        ptr = np.asarray(frag_payload[f"{split}_mol_view_ptr"], dtype=np.int64)
        c_ids = np.asarray(frag_payload[f"{split}_view_c"], dtype=np.int64)
        bond = np.asarray(frag_payload[f"{split}_view_bond_feat"], dtype=np.int64).reshape(-1, 3)
        for i, row in enumerate(rows.tolist()):
            bag = out.setdefault(int(row), set())
            for v in range(int(ptr[i]), int(ptr[i + 1])):
                bag.add((str(keys[int(c_ids[v])]), tuple(int(x) for x in bond[v])))
    return out


def train_word_inventory(frag_payload: Mapping[str, Any]) -> set[str]:
    """The existing dictionary's Train S-word candidate inventory (702 keys)."""
    keys = frag_payload["frag_key"]
    s_ids = np.asarray(frag_payload["train_view_s"], dtype=np.int64)
    return {str(keys[int(i)]) for i in s_ids.tolist()}


def prediction_arrays() -> dict[str, dict[str, Any]]:
    """Load every frozen per-molecule full-F prediction and verify identities."""
    payload = load_payload()
    labels = np.asarray(payload["labels"], dtype=np.float64)
    official = official_split(payload)
    n = len(labels)
    out: dict[str, dict[str, Any]] = {}

    for model, path in NATIVE_PREDICTION.items():
        with np.load(path) as data:
            epoch = int(np.asarray(data["selected_epoch"]).reshape(-1)[0])
            if epoch != NATIVE_EPOCH[model]:
                raise ValueError(f"{model}: selected epoch {epoch} != {NATIVE_EPOCH[model]}")
            pred = np.full(n, np.nan, dtype=np.float64)
            for split in SPLITS:
                rows = np.asarray(data[f"{split}_row_ids"], dtype=np.int64)
                values = np.asarray(data[f"{split}_pred_raw"], dtype=np.float64)
                targets = np.asarray(data[f"{split}_target_raw"], dtype=np.float64)
                expected = np.asarray(payload[f"{split}_idx"], dtype=np.int64)
                if not np.array_equal(np.sort(rows), np.sort(expected)):
                    raise ValueError(f"{model}: {split} row ids differ from the official split")
                if not np.allclose(targets, labels[rows], atol=1e-9):
                    raise ValueError(f"{model}: {split} targets differ from the official labels")
                if any(str(official[int(r)]) != split for r in rows.tolist()):
                    raise ValueError(f"{model}: {split} split membership mismatch")
                pred[rows] = values
        if not np.all(np.isfinite(pred)):
            raise ValueError(f"{model}: predictions do not cover all 4200 rows")
        out[model] = {
            "pred": pred,
            "path": str(path),
            "sha256": sha256_file(path),
            "epoch": epoch,
            "version": "native_full_F",
        }

    with np.load(DICT_PREDICTION) as data:
        for model in DICTIONARIES:
            pred = np.full(n, np.nan, dtype=np.float64)
            for split in SPLITS:
                rows = np.asarray(data[f"{model}_cert_{split}_row_ids"], dtype=np.int64)
                values = np.asarray(data[f"{model}_cert_{split}_pred_raw"], dtype=np.float64)
                targets = np.asarray(data[f"{model}_cert_{split}_target_raw"], dtype=np.float64)
                expected = np.asarray(payload[f"{split}_idx"], dtype=np.int64)
                if not np.array_equal(np.sort(rows), np.sort(expected)):
                    raise ValueError(f"{model}: {split} row ids differ from the official split")
                if not np.allclose(targets, labels[rows], atol=1e-9):
                    raise ValueError(f"{model}: {split} targets differ from the official labels")
                if any(str(official[int(r)]) != split for r in rows.tolist()):
                    raise ValueError(f"{model}: {split} split membership mismatch")
                pred[rows] = values
            if not np.all(np.isfinite(pred)):
                raise ValueError(f"{model}: predictions do not cover all 4200 rows")
            out[model] = {
                "pred": pred,
                "path": str(DICT_PREDICTION),
                "sha256": sha256_file(DICT_PREDICTION),
                "epoch": None,
                "version": DICT_VERSION,
                "keys": f"{model}_cert_<split>_pred_raw",
            }
    return out


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.mean(np.abs(pred - target)))


def component_equal(values: np.ndarray, components: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    components = np.asarray(components)
    uniq = sorted(set(components.tolist()))
    if not uniq:
        return float("nan")
    return float(np.mean([np.mean(values[components == c]) for c in uniq]))


def component_equal_rmse(values: np.ndarray, components: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    components = np.asarray(components)
    uniq = sorted(set(components.tolist()))
    if not uniq:
        return float("nan")
    return float(np.sqrt(np.mean([np.mean(values[components == c] ** 2) for c in uniq])))


def jsonable(value: Any) -> Any:
    return RC.jsonable(value)
