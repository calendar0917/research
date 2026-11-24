"""Shared plumbing for HSRF v0.

Read-only reuse (sha256 pinned at import time):

* ``sq_data.py`` / ``sq_common.py``: official OGB cache loader, split indices,
  hashing / IO helpers.
* ``rice_structure.py``: rooted connected induced-subgraph enumeration and the
  exact root-marked attributed-graph canonical certificate.

No parent model, embedding, checkpoint or statistic is imported.
"""

from __future__ import annotations

import hashlib
import json
import os
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
SQL = REPO_ROOT / "tracks" / "structural-query-learning"
RICE_CODE = STRUCT / "rooted_interaction_cumulant_encoder_pilot_v0" / "code"
SQL_CODE = SQL / "code"

for _p in (RICE_CODE, SQL_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sq_common as SC  # noqa: E402  (path setup first)
import sq_data as SD  # noqa: E402
import rice_structure as RS  # noqa: E402

CONFIG_PATH = ROUND_DIR / "configs" / "hsrf.json"
DATA_DIR = ROUND_DIR / "data"
ANALYSIS_DIR = ROUND_DIR / "analysis"
RESULTS_DIR = ROUND_DIR / "results"
RUNS_DIR = ROUND_DIR / "runs"

SOURCE_PINS = {
    "sq_data.py": (SQL_CODE / "sq_data.py",
                   "89ad5436be3ddb8825fe877535e7fbb13679f4cb012a3a3073d63deaf9a83647"),
    "sq_common.py": (SQL_CODE / "sq_common.py",
                     "fe7333978fd748cf2a61fcdb4d3fa73bf855ebbdc6b05c439477b2c921b9d8cb"),
    "rice_structure.py": (RICE_CODE / "rice_structure.py",
                          "cb044e37e3c2bfd90881fca196b3e40fa7e05429bc389747da3957a56269629b"),
}

CACHE = SQL / "data" / "ogbg_mollipo_sq-pilot-cache-v1.npz"
CACHE_SHA = "fe002287fa9c581278bc5430f5b4a5b26b35d9d4d5207d138f15956d93123c7c"

# Historical prediction artefacts (read only; never retrained here).
HIST_PATHS = {
    "GINE_sq_pilot": SQL / "results" / "sq-pilot-small-v1" / "ogbg-mollipo" / "method_G_pred.npz",
    "LCRC_sq_pilot": SQL / "results" / "sq-pilot-small-v1" / "ogbg-mollipo" / "method_L_pred.npz",
    "PAIR_RULE_REAL_FULL": (REPO_ROOT / ".rr" / "pulled"
                            / "sq-pair-rules-v2-20261118-102436-fde5d663" / "result"
                            / "tracks" / "structural-query-learning" / "pair_rule_pilot_v1"
                            / "runs" / "rules" / "REAL" / "best_FULL_predictions.npz"),
    "PAIR_RULE_MLP": (REPO_ROOT / ".rr" / "pulled"
                      / "sq-pair-rules-v2-20261118-102436-fde5d663" / "result"
                      / "tracks" / "structural-query-learning" / "pair_rule_pilot_v1"
                      / "runs" / "rules" / "MLP" / "best_predictions.npz"),
    "SWF_WORD_PREFIX": (STRUCT / "structural_word_functions_s0_v1" / "results"
                        / "lipo_WORD_PREFIX_s0" / "predictions_best.npz"),
}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def array_sha256(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        arr = np.ascontiguousarray(arr)
        h.update(str(arr.shape).encode())
        h.update(arr.dtype.str.encode())
        h.update(arr.tobytes())
    return h.hexdigest()


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


def sha256_json(payload: Any) -> str:
    return sha256_bytes(json.dumps(jsonable(payload), sort_keys=True).encode("utf-8"))


def write_json(path: Path | str, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(jsonable(payload), indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    )
    return target


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> Path:
    import csv

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        target.write_text("")
        return target
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: jsonable(row.get(k)) for k in fields})
    return target


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def load_config() -> dict[str, Any]:
    return read_json(CONFIG_PATH)


def assert_source_pins() -> dict[str, str]:
    found: dict[str, str] = {}
    for name, (path, expected) in SOURCE_PINS.items():
        actual = sha256_file(path)
        found[name] = actual
        if actual != expected:
            raise RuntimeError(f"pinned source changed: {name} {actual} != {expected}")
    return found


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    diff = np.asarray(pred, np.float64) - np.asarray(y, np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    diff = np.asarray(pred, np.float64) - np.asarray(y, np.float64)
    return float(np.mean(np.abs(diff)))


# ---------------------------------------------------------------------------
# official data
# ---------------------------------------------------------------------------

def load_payload(verify: bool = True) -> dict[str, Any]:
    payload = SD.load_cache("ogbg-mollipo", verify_sha=CACHE_SHA if verify else None)
    rows = np.asarray(payload["row_ids"], np.int64)
    labels = np.asarray(payload["labels"], np.float64)
    splits = {name: SD.split_indices(payload, name) for name in ("train", "valid", "test")}
    sizes = {name: int(arr.size) for name, arr in splits.items()}
    if sizes != {"train": 3360, "valid": 420, "test": 420}:
        raise ValueError(f"official split sizes changed: {sizes}")
    if not np.array_equal(rows, np.arange(rows.size)):
        raise ValueError("official row ids are not 0..N-1")
    if not np.isfinite(labels).all():
        raise ValueError("label vector contains non-finite entries")
    union = np.union1d(np.union1d(splits["train"], splits["valid"]), splits["test"])
    if union.size != labels.size or not np.array_equal(union, np.arange(labels.size)):
        raise ValueError("official split partition does not cover the 4200 rows")
    if np.intersect1d(splits["train"], splits["valid"]).size or \
       np.intersect1d(splits["train"], splits["test"]).size or \
       np.intersect1d(splits["valid"], splits["test"]).size:
        raise ValueError("official splits overlap")
    payload["split_rows"] = splits
    payload["split_sizes"] = sizes
    payload["label_sha256"] = array_sha256(labels.astype(np.float64))
    return payload


def target_scaler(labels: np.ndarray, train_rows: np.ndarray) -> dict[str, Any]:
    y = np.asarray(labels, np.float64)[np.asarray(train_rows, np.int64)]
    mean = float(y.mean())
    std = float(y.std(ddof=0))
    if not (std > 0):
        raise ValueError("zero Train target scale")
    z = (y - mean) / std
    back = z * std + mean
    return {"mean": mean, "std": std, "n": int(y.size), "ddof": 0,
            "roundtrip_max_abs_error": float(np.abs(back - y).max())}


def molecule_record(payload: Mapping[str, Any], row: int) -> dict[str, Any]:
    """One molecule's label-free local view (same slicing as gwic_data.molecule_view)."""
    a0 = int(payload["node_ptr"][row]); a1 = int(payload["node_ptr"][row + 1])
    b0 = int(payload["bond_ptr"][row]); b1 = int(payload["bond_ptr"][row + 1])
    return {
        "row": int(row),
        "n_atoms": int(a1 - a0),
        "n_bonds": int(b1 - b0),
        "atom_feat": np.asarray(payload["atom_feat"][a0:a1], np.int64),
        "bond_feat": np.asarray(payload["bond_feat"][b0:b1], np.int64).reshape(b1 - b0, 3),
        "bond_u": np.asarray(payload["bond_u"][b0:b1], np.int64),
        "bond_v": np.asarray(payload["bond_v"][b0:b1], np.int64),
        "smiles": str(np.asarray(payload["smiles"])[row]),
        "graph_n_atoms": int(payload["graph_n_atoms"][row]),
        "graph_n_bonds": int(payload["graph_n_bonds"][row]),
        "graph_n_components": int(payload["graph_n_components"][row]),
        "graph_cycle_rank": int(payload["graph_cycle_rank"][row]),
        "graph_scalars": np.zeros(8, np.float64),
    }


def build_struct(payload: Mapping[str, Any], row: int) -> RS.MoleculeStruct:
    return RS.build_molecule(molecule_record(payload, int(row)))


def atom_bond_dims() -> tuple[list[int], list[int]]:
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

    atom = [int(x) for x in get_atom_feature_dims()]
    bond = [int(x) for x in get_bond_feature_dims()]
    if len(atom) != 9 or len(bond) != 3:
        raise ValueError("unexpected official OGB feature widths")
    return atom, bond


def sigma_bytes(signature: Sequence[int]) -> bytes:
    import struct

    return struct.pack(">10q", *[int(x) for x in signature])


def signature_key(gran: str, size: int, signature: Sequence[int]) -> str:
    import struct

    return f"{gran}:{int(size)}:" + struct.pack(">10q", *[int(x) for x in signature]).hex()


def gran_view(st: RS.MoleculeStruct, gran: str):
    """Return a shallow view of ``st`` with the granularity's attribute columns."""
    import copy

    if gran == "FULL":
        return st
    if gran == "TOPO":
        out = copy.copy(st)
        out.atom_feat = np.zeros_like(st.atom_feat)
        out.bond_feat = np.zeros_like(st.bond_feat)
        return out
    if gran == "ELEMENT":
        out = copy.copy(st)
        out.atom_feat = np.asarray(st.atom_feat)[:, [0]]
        out.bond_feat = np.asarray(st.bond_feat)[:, [0]]
        return out
    if gran == "COARSE":
        out = copy.copy(st)
        out.atom_feat = np.asarray(st.atom_feat)[:, [0, 3, 7]]
        out.bond_feat = np.asarray(st.bond_feat)[:, [0]]
        return out
    raise KeyError(gran)


def gran_dims(atom_dims: Sequence[int], bond_dims: Sequence[int],
              gran: str) -> tuple[list[int], list[int]]:
    if gran == "FULL":
        return list(atom_dims), list(bond_dims)
    if gran == "TOPO":
        return [1], [1]
    if gran == "ELEMENT":
        return [int(atom_dims[0])], [int(bond_dims[0])]
    if gran == "COARSE":
        return [int(atom_dims[0]), int(atom_dims[3]), int(atom_dims[7])], [int(bond_dims[0])]
    raise KeyError(gran)


def signatures_for(st: RS.MoleculeStruct, size: int, gran: str,
                   atom_dims: Sequence[int], bond_dims: Sequence[int]) -> np.ndarray:
    ad, bd = gran_dims(atom_dims, bond_dims, gran)
    if size == 1:
        label = RS._mixed_radix(st.atom_feat, ad)  # noqa: SLF001 exact reuse
        sig = np.full((st.n_atoms, 10), -1, np.int64)
        sig[:, 0] = label
        return sig
    return RS.canonical_signature(gran_view(st, gran), size, ad, bd)


# ---------------------------------------------------------------------------
# scaffolds / bootstrap
# ---------------------------------------------------------------------------

def scaffold_groups(payload: Mapping[str, Any], split: str) -> dict[str, Any]:
    from collections import defaultdict

    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    rows = np.asarray(payload["split_rows"][split], np.int64)
    groups: dict[str, list[int]] = defaultdict(list)
    kinds = {"scaffold": 0, "acyclic_canonical_smiles": 0, "parse_failed": 0}
    labels: list[str] = []
    for pos, row in enumerate(rows.tolist()):
        mol = Chem.MolFromSmiles(str(payload["smiles"][row]))
        if mol is None:
            key, kind = f"ROW:{row}", "parse_failed"
        else:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if scaf:
                key, kind = f"SCAFFOLD:{scaf}", "scaffold"
            else:
                key, kind = ("ACYCLIC:" + Chem.MolToSmiles(mol, isomericSmiles=False),
                             "acyclic_canonical_smiles")
        groups[key].append(pos)
        labels.append(key)
        kinds[kind] += 1
    return {"split": split, "rows": rows, "cluster_rows": list(groups.values()),
            "cluster_label": labels, "n_rows": int(rows.size),
            "n_clusters": len(groups), "kind_counts": kinds}


def paired_bootstrap(y: np.ndarray, predictions: Mapping[str, np.ndarray],
                     cluster_rows: Sequence[np.ndarray],
                     comparisons: Sequence[tuple[str, str, str]],
                     resamples: int, seed: int) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Paired cluster bootstrap.

    ``comparisons`` entries are ``(name, reference, candidate)``; delta is
    RMSE(reference) - RMSE(candidate), positive = candidate better.
    Returns the row table and an array ``[n_comparisons, resamples]``.
    """
    y = np.asarray(y, np.float64)
    preds = {k: np.asarray(v, np.float64) for k, v in predictions.items()}
    clusters = [np.asarray(c, np.int64) for c in cluster_rows]
    rng = np.random.default_rng(int(seed))
    draws = np.zeros((len(comparisons), int(resamples)), np.float64)
    for it in range(int(resamples)):
        chosen = rng.choice(len(clusters), size=len(clusters), replace=True)
        idx = np.concatenate([clusters[int(c)] for c in chosen])
        for ci, (_name, ref, cand) in enumerate(comparisons):
            draws[ci, it] = rmse(y[idx], preds[ref][idx]) - rmse(y[idx], preds[cand][idx])
    full = np.arange(y.size)
    rows = []
    for ci, (name, ref, cand) in enumerate(comparisons):
        r_ref = rmse(y[full], preds[ref])
        r_cand = rmse(y[full], preds[cand])
        diff = np.abs(preds[ref] - y) - np.abs(preds[cand] - y)
        drop = int(np.argmax(np.abs(diff)))
        keep = np.ones(y.size, bool)
        keep[drop] = False
        rows.append({
            "comparison": name,
            "reference": ref,
            "candidate": cand,
            "rmse_reference": r_ref,
            "rmse_candidate": r_cand,
            "delta": float(r_ref - r_cand),
            "ci_low": float(np.quantile(draws[ci], 0.025)),
            "ci_high": float(np.quantile(draws[ci], 0.975)),
            "mae_delta": float(mae(y, preds[ref]) - mae(y, preds[cand])),
            "drop_one_delta": float(rmse(y[keep], preds[ref][keep]) - rmse(y[keep], preds[cand][keep])),
            "n_clusters": int(len(clusters)),
            "n_rows": int(y.size),
            "resamples": int(resamples),
            "seed": int(seed),
        })
    return rows, draws


# ---------------------------------------------------------------------------
# historical coordinates
# ---------------------------------------------------------------------------

def load_historical(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Load frozen historical predictions, verify identity, recompute RMSE/MAE."""
    out: dict[str, Any] = {"arms": {}, "paths": {}, "frozen": {}}
    labels = np.asarray(payload["labels"], np.float64)
    for name, path in HIST_PATHS.items():
        if not Path(path).is_file():
            out["arms"][name] = {"available": False, "path": str(path)}
            continue
        z = np.load(path, allow_pickle=False)
        rec: dict[str, Any] = {"available": True, "path": str(path.relative_to(REPO_ROOT)),
                               "sha256": sha256_file(path), "splits": {}}
        for split, rk, pk, yk in (
            ("train", "train_row_id", "train_pred_raw", "train_y_raw"),
            ("valid", "valid_row_id", "valid_pred_raw", "valid_y_raw"),
            ("test", "test_row_id", "test_pred_raw", "test_y_raw"),
        ):
            if any(k not in z.files for k in (rk, pk, yk)):
                if split == "train":
                    rk, pk, yk = "train_row_id", "train_pred", "train_label"
                elif split == "valid":
                    rk, pk, yk = "valid_row_id", "valid_pred", "valid_label"
                else:
                    rk, pk, yk = "test_row_id", "test_pred", "test_label"
            rid = np.asarray(z[rk], np.int64)
            y = np.asarray(z[yk], np.float64)
            p = np.asarray(z[pk], np.float64)
            expected_rows = np.asarray(payload["split_rows"][split], np.int64)
            identity_ok = bool(rid.size == expected_rows.size and
                               np.array_equal(np.sort(rid), np.sort(expected_rows)))
            # align to the official split order
            pos = {int(r): i for i, r in enumerate(rid.tolist())}
            order = np.asarray([pos[int(r)] for r in expected_rows.tolist()], np.int64)
            y = y[order]
            p = p[order]
            label_ok = bool(np.allclose(y, labels[expected_rows], atol=1e-6, rtol=0))
            rec["splits"][split] = {
                "row_id": expected_rows.copy(),
                "y": y,
                "pred": p,
                "identity_ok": identity_ok,
                "label_ok": label_ok,
                "rmse": rmse(y, p),
                "mae": mae(y, p),
                "n": int(y.size),
            }
        out["arms"][name] = rec
    return out