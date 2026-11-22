"""Shared plumbing for the structural path composition route (PSC).

Nothing scientific lives here: paths, IO, hashing, seeding, metrics and the
label-safe data boundaries.  Every scientific choice is frozen in
``configs/psc.json`` and ``PROTOCOL.md`` before formal training starts.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import socket
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
ACRA1_DIR = STRUCT / "attachment_conditioned_response_algebra_pilot_v1"
ACRA2_DIR = STRUCT / "acra_relation_generalization_pilot_v1"
AUDIT_DIR = STRUCT / "structural_response_learnability_audit_v1"
TRC_DIR = STRUCT / "task_response_composition_s0_v1"
WP_DIR = STRUCT / "word_sentence_port_binding_context_v2"
SQ_DIR = REPO_ROOT / "tracks" / "structural-query-learning"

RAW_CACHE = SQ_DIR / "data" / "ogbg_mollipo_sq-pilot-cache-v1.npz"
RAW_CACHE_SHA256 = "fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c"
TRC_CACHE = TRC_DIR / "data" / "trc_cache_lipo.npz"
TRC_CACHE_SHA256 = "c53fee5d16869b7b7ff721d404a84289d0a315d392440ebfb436cbd307e3abc3"
ACRA_CACHE = ACRA2_DIR / "data" / "acra_cache_lipo_v2.npz"
ACRA_CACHE_SHA256 = "1f6d7a3f870513a83b50f6aa6edbe0143f37c2a883308627f47145e2078fdc3b"
SPLIT_F0 = ACRA2_DIR / "data" / "split_train_sub_dev.json"
SPLIT_F0_SHA256 = "675289f6f997008d44fdabb4ab65350459fe5bccc5e1e5703e31e31fd794e845"
#: official Train labels, extracted once from the frozen ACRA-v1 run file
TRAIN_LABEL_SOURCE = ACRA1_DIR / "results" / "ACRA_REL_S0" / "predictions_best_train.npz"
#: official Valid labels -- opened only by the frozen final evaluator
VALID_LABEL_SOURCE = ACRA1_DIR / "results" / "ACRA_REL_S0" / "predictions_best.npz"

CONFIG_PATH = ROUND_DIR / "configs" / "psc.json"
SPLIT_F1 = ROUND_DIR / "data" / "split_train_sub_dev_d1.json"
TRAIN_LABELS = ROUND_DIR / "data" / "labels_official_train.npz"
PATH_CACHE = ROUND_DIR / "data" / "psc_path_cache_lipo.npz"

#: atom / bond raw columns of the official OGB raw schema
ATOM_COLUMNS = 9
BOND_COLUMNS = 3

#: label-free raw members that may be read from the structure cache
RAW_STRUCT_KEYS = (
    "dataset", "cache_version", "node_ptr", "bond_ptr", "atom_feat", "bond_feat",
    "bond_u", "bond_v", "degree", "min_ring", "in_cycle",
    "graph_n_components", "graph_cycle_rank", "graph_n_bonds", "graph_n_atoms",
    "smiles", "row_ids", "train_idx", "valid_idx",
)


# ---------------------------------------------------------------------------
# IO and identity
# ---------------------------------------------------------------------------


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_bytes(json.dumps(jsonable(payload), sort_keys=True).encode("utf-8"))


def array_sha256(*arrays: np.ndarray) -> str:
    """Order-sensitive content hash of one or more arrays."""
    h = hashlib.sha256()
    for arr in arrays:
        arr = np.ascontiguousarray(arr)
        h.update(str(arr.shape).encode())
        h.update(arr.dtype.str.encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def code_sha(paths: Sequence[str | Path]) -> str:
    """Hash of the code files that define a cache/model/protocol."""
    h = hashlib.sha256()
    for name in sorted(str(p) for p in paths):
        path = Path(name)
        if not path.is_absolute():
            path = CODE_DIR / path
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return None if (np.isnan(value) or np.isinf(value)) else value
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return None if (np.isnan(value) or np.isinf(value)) else value
    return value


def write_json(path: Path | str, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(jsonable(payload), indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return target


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]],
              columns: Sequence[str] | None = None) -> Path:
    import csv

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return target
    if columns is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        columns = ordered
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
    return target


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def code_provenance() -> dict[str, Any]:
    return {"git_commit": git_commit(), "code_dir": str(CODE_DIR.relative_to(REPO_ROOT)),
            "config": str(CONFIG_PATH.relative_to(REPO_ROOT))}


def environment() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_count": int(os.cpu_count() or 0),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = int(torch.cuda.device_count())
        try:
            info["nvidia_smi"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
                text=True, timeout=10).strip().splitlines()
        except Exception as exc:  # noqa: BLE001
            info["nvidia_smi_error"] = repr(exc)
    return info


def set_deterministic(seed: int) -> dict[str, Any]:
    import torch

    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {"seed": int(seed)}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    diff = np.asarray(pred, np.float64) - np.asarray(target, np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, np.float64) - np.asarray(target, np.float64))))


def percentile_ci(values: np.ndarray, level: float = 95.0) -> list[float]:
    vals = np.asarray(values, np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return [float("nan"), float("nan")]
    lo = (100.0 - level) / 2.0
    return [float(np.percentile(vals, lo)), float(np.percentile(vals, 100.0 - lo))]


def target_scaler(labels: np.ndarray, fit_rows: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, np.float64)[np.asarray(fit_rows, np.int64)]
    return {"mean": float(np.mean(y)), "std": float(np.std(y)), "n": int(y.size)}


# ---------------------------------------------------------------------------
# label-safe loading
# ---------------------------------------------------------------------------


def load_raw_structure() -> dict[str, Any]:
    """Label-free raw molecule structure, whitelist members only.

    The archive is opened lazily and only the explicit whitelist is touched;
    ``labels`` is never accessed here.
    """
    digest = sha256_file(RAW_CACHE)
    if digest != RAW_CACHE_SHA256:
        raise ValueError(f"raw cache sha256 mismatch: {digest} != {RAW_CACHE_SHA256}")
    out: dict[str, Any] = {}
    with np.load(RAW_CACHE, allow_pickle=False) as z:
        available = set(z.files)
        for key in RAW_STRUCT_KEYS:
            if key not in available:
                raise KeyError(f"raw structure member missing: {key}")
            out[key] = np.asarray(z[key])
    out["cache_sha256"] = digest
    out["cache_path"] = str(RAW_CACHE)
    out["keys_read"] = list(RAW_STRUCT_KEYS)
    return out


def _official_split(raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {"train": np.asarray(raw["train_idx"], np.int64),
            "valid": np.asarray(raw["valid_idx"], np.int64)}


def is_official_train_row(raw: Mapping[str, Any], rows: Sequence[int]) -> bool:
    train = set(_official_split(raw)["train"].tolist())
    return all(int(r) in train for r in np.asarray(rows, np.int64).tolist())


def build_train_labels(raw: Mapping[str, Any], path: Path | str = TRAIN_LABELS) -> dict[str, Any]:
    """Extract exactly two members (row_id, label) from the frozen Train file.

    Writes a round-local ``row_id,label`` archive so later stages never reopen
    the ACRA prediction file.  Valid and Test rows are untouched.
    """
    target = Path(path)
    with np.load(TRAIN_LABEL_SOURCE, allow_pickle=False) as z:
        row_id = np.asarray(z["row_id"], np.int64)
        label = np.asarray(z["label"], np.float64)
    official = _official_split(raw)
    train = np.sort(official["train"])
    if row_id.size != train.size:
        raise ValueError(f"train label rows {row_id.size} != official Train {train.size}")
    if np.unique(row_id).size != row_id.size:
        raise ValueError("duplicate row_id in train label source")
    if not np.array_equal(np.sort(row_id), train):
        raise ValueError("train label row_id set differs from the official Train split")
    if not np.isfinite(label).all():
        raise ValueError("non-finite train label")
    order = np.argsort(row_id)
    row_id, label = row_id[order], label[order]
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, row_id=row_id, label=label)
    return {
        "path": str(target.relative_to(REPO_ROOT)),
        "sha256": sha256_file(target),
        "n": int(row_id.size),
        "source": str(TRAIN_LABEL_SOURCE.relative_to(REPO_ROOT)),
        "source_sha256": sha256_file(TRAIN_LABEL_SOURCE),
        "members_read": ["row_id", "label"],
        "row_id_sha256": array_sha256(row_id),
        "label_sha256": array_sha256(label),
        "labels_read": "official Train rows only (3360)",
    }


def load_train_labels() -> dict[str, Any]:
    if not TRAIN_LABELS.is_file():
        raise FileNotFoundError(f"train label file missing: {TRAIN_LABELS} -- run psc_prepare")
    with np.load(TRAIN_LABELS, allow_pickle=False) as z:
        row_id = np.asarray(z["row_id"], np.int64)
        label = np.asarray(z["label"], np.float64)
    if np.unique(row_id).size != row_id.size or not np.isfinite(label).all():
        raise ValueError("corrupted train label file")
    return {"row_id": row_id, "label": label,
            "sha256": sha256_file(TRAIN_LABELS),
            "row_id_sha256": array_sha256(row_id),
            "label_sha256": array_sha256(label)}


def load_valid_labels(raw: Mapping[str, Any]) -> dict[str, Any]:
    """FINAL-EVAL ONLY.  Reads exactly two members of the Valid file.

    Documented and gated: this function is called once by
    ``psc_final_eval.py`` after ``final_eval_lock.json`` exists.
    """
    with np.load(VALID_LABEL_SOURCE, allow_pickle=False) as z:
        row_id = np.asarray(z["row_id"], np.int64)
        label = np.asarray(z["label"], np.float64)
    official = np.sort(_official_split(raw)["valid"])
    if row_id.size != official.size or not np.array_equal(np.sort(row_id), official):
        raise ValueError("valid label row_id set differs from the official Valid split")
    if not np.isfinite(label).all():
        raise ValueError("non-finite valid label")
    order = np.argsort(row_id)
    return {"row_id": row_id[order], "label": label[order],
            "source": str(VALID_LABEL_SOURCE.relative_to(REPO_ROOT)),
            "source_sha256": sha256_file(VALID_LABEL_SOURCE),
            "members_read": ["row_id", "label"],
            "row_id_sha256": array_sha256(row_id[order]),
            "label_sha256": array_sha256(label[order])}


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------


def load_split_f0() -> dict[str, Any]:
    path = SPLIT_F0
    digest = sha256_file(path)
    if digest != SPLIT_F0_SHA256:
        raise ValueError(f"F0 split sha256 mismatch: {digest} != {SPLIT_F0_SHA256}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["file_sha256"] = digest
    return payload


def assignment_rule() -> str:
    return ("sorted scaffold keys -> numpy default_rng(seed).permutation -> "
            "greedy add whole groups while not exceeding the target")


def build_split_d1(raw: Mapping[str, Any], seed: int, target_n: int,
                   log=print) -> dict[str, Any]:
    """Second scaffold-disjoint Train-sub / Dev split, frozen before results.

    Same algorithm as the existing F0 split: RDKit Bemis-Murcko scaffolds
    (``includeChirality=False``), acyclic molecules keyed by their own SMILES,
    sorted keys, ``default_rng(seed).permutation``, greedy whole-group
    assignment to the held-out side until the target count is hit.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    train = np.sort(np.asarray(raw["train_idx"], np.int64))
    smiles = np.asarray(raw["smiles"])

    def scaffold_key(sm: str) -> str:
        mol = Chem.MolFromSmiles(str(sm))
        if mol is None:
            return f"__UNPARSED__{sm}"
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaf if scaf else f"__ACYCLIC__{sm}"

    scaffold_of = {int(r): scaffold_key(str(smiles[int(r)])) for r in train.tolist()}
    groups: dict[str, list[int]] = {}
    for row, key in scaffold_of.items():
        groups.setdefault(key, []).append(row)
    keys = sorted(groups)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(keys))
    dev: list[int] = []
    dev_scaffolds: list[str] = []
    for i in order.tolist():
        key = keys[i]
        if len(dev) + len(groups[key]) <= int(target_n):
            dev.extend(groups[key])
            dev_scaffolds.append(key)
        if len(dev) == int(target_n):
            break
    dev = sorted(int(r) for r in dev)
    dev_set = set(dev)
    fit = [int(r) for r in train.tolist() if int(r) not in dev_set]
    if set(fit) & dev_set:
        raise AssertionError("D1 fit / heldout overlap")
    if sorted(fit + dev) != train.tolist():
        raise AssertionError("D1 fit and heldout do not cover official Train")
    fit_scaffolds = {scaffold_of[r] for r in fit}
    if fit_scaffolds & set(dev_scaffolds):
        raise AssertionError("D1 scaffold leakage between fit and heldout")
    payload = {
        "protocol_id": "structural-path-composition-route-v1",
        "generated_utc": now_utc(),
        "seed": int(seed),
        "fraction_dev_target": float(target_n) / float(train.size),
        "target_heldout": int(target_n),
        "n_official_train": int(train.size),
        "n_fit": int(len(fit)),
        "n_heldout": int(len(dev)),
        "n_scaffolds_official_train": int(len(keys)),
        "n_scaffolds_fit": int(len(fit_scaffolds)),
        "n_scaffolds_heldout": int(len(dev_scaffolds)),
        "fit_rows": fit,
        "heldout_rows": dev,
        "heldout_scaffolds": sorted(dev_scaffolds),
        "scaffold_by_row": {str(k): v for k, v in scaffold_of.items()},
        "label_free": True,
        "labels_read": False,
        "scaffold_definition": "rdkit Bemis-Murcko, acyclic molecules keyed by their own SMILES",
        "assignment_rule": assignment_rule(),
        "role": "split-robustness replication (NOT a fresh blind test)",
    }
    payload["split_sha256"] = sha256_json({k: payload[k] for k in ("fit_rows", "heldout_rows", "seed")})
    return payload


# ---------------------------------------------------------------------------
# split descriptions shared by every arm
# ---------------------------------------------------------------------------


def split_spec(name: str, raw: Mapping[str, Any],
               f0: Mapping[str, Any] | None = None,
               f1: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return ``{name, fit_rows, eval_rows, sha256, source}`` for a split role."""
    official = _official_split(raw)
    if name == "D0":
        payload = dict(f0 or load_split_f0())
        return {"name": "D0", "fit_rows": np.asarray(payload["train_sub_rows"], np.int64),
                "eval_rows": np.asarray(payload["dev_rows"], np.int64),
                "sha256": str(payload["split_sha256"]),
                "file_sha256": str(payload.get("file_sha256", "")),
                "source": str(SPLIT_F0.relative_to(REPO_ROOT)),
                "role": "development split (F0/D0)"}
    if name == "D1":
        if f1 is None:
            if not SPLIT_F1.is_file():
                raise FileNotFoundError(f"D1 split missing: {SPLIT_F1}")
            f1 = json.loads(SPLIT_F1.read_text(encoding="utf-8"))
            f1["file_sha256"] = sha256_file(SPLIT_F1)
        return {"name": "D1", "fit_rows": np.asarray(f1["fit_rows"], np.int64),
                "eval_rows": np.asarray(f1["heldout_rows"], np.int64),
                "sha256": str(f1["split_sha256"]),
                "file_sha256": str(f1.get("file_sha256", "")),
                "source": str(SPLIT_F1.relative_to(REPO_ROOT)),
                "role": "split-robustness replication (not a fresh blind test)"}
    if name == "FULL":
        return {"name": "FULL", "fit_rows": np.sort(official["train"]),
                "eval_rows": np.sort(official["valid"]),
                "sha256": array_sha256(np.sort(official["train"]))[:16],
                "file_sha256": "",
                "source": "official Train rows",
                "role": "full-Train refit; official Valid scored once after freeze"}
    raise ValueError(f"unknown split {name}")


def assert_fit_rows(official_train: np.ndarray, rows: Sequence[int]) -> None:
    rows = np.asarray(rows, np.int64)
    if not np.isin(rows, np.asarray(official_train, np.int64)).all():
        raise ValueError("fit rows outside official Train")