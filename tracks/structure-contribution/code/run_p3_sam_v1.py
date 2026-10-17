#!/usr/bin/env python
"""Driver for protocol ``cscl-p3-sam-v1`` (P3 vs P3-SAM).

Stages
------
``split``               local CPU, RDKit, label-blind: rebuild the four
                        SMILES/PyG mismatch rows, derive identity from the PyG
                        graph, build the new identity-grouped split T with an
                        **explicit** salt, assert the saved salt/hash, and
                        compare against the historical E/F/S splits.
``verify-inputs``       frozen upstream hashes, split sanity, parameter and
                        recipe constants, no official valid/test loader.
``objects``             LabelGuard build of Fit/Monitor/Locked objects and the
                        Fit-only global standardiser.
``cpu-gates``           synthetic SAM two-step / restore / step-count / clip /
                        last-5 soup gates plus the frozen P3-equivalence gate.
``freeze``              ``protocol_freeze.json`` (code + config + inputs +
                        budget) before any GPU use.
``stage-inputs``        rebuild the machine-local object artifacts on a compute
                        checkout and verify them semantically.
``smoke``               post-freeze 20-update SSL + 1-epoch mini run per arm,
                        with measured throughput for the budget projection.
``ssl``                 Fit-only masked atom/bond SSL for one seed (no labels).
``train``               P3 and P3-SAM for one seed (Fit labels only).
``predict``             own-arm Monitor + Locked predictions (no labels).
``freeze-predictions``  hash-freeze all predictions and checkpoints.
``score-monitor``       single read of Monitor labels only; if the frozen gate
                        fails, stop without reading Locked labels.
``score-locked``        only after a Monitor pass: single read of Locked labels.
``finalize``            evidence manifest + budget summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_arch_breakthrough_v1 as ab  # noqa: E402
import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import bond_relation_attribution_v1 as bra  # noqa: E402
import p3_sam_v1 as ps  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

REPO_ROOT = _CODE_DIR.parents[2]
TRACK = _CODE_DIR.parent
OUT = Path(os.environ.get("P3SAM_OUT_DIR", str(TRACK / "results" / "p3_sam_v1")))
CONFIG_PATH = TRACK / "configs" / "p3_sam_v1.yaml"
PROTOCOL_DOC = OUT / "PROTOCOL.md"
ZINC_ROOT = REPO_ROOT / "data/ZINC"
DATA_TRAIN = ZINC_ROOT / "subset/processed/train.pt"
CANON = REPO_ROOT / "tracks/ksvd/results/zinc_cssd_basis_reuse_v1/train_canonical_smiles.npz"
ENCODED_TRAIN = REPO_ROOT / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt"
STRUCT_CONTEXT_CODE = TRACK / "results/structure_context_support_v1/code"

P3_RESULTS = TRACK / "results" / "p3_bond_relation_bridge_v1"
SAC_RESULTS = TRACK / "results" / "sac_v1"
BRA_RESULTS = TRACK / "results" / "bond_relation_attribution_v1"

PROTOCOL_ID = ps.PROTOCOL_ID
SPLIT_TAG = ps.SPLIT_TAG
SEEDS = ps.SEEDS
ARMS = ps.ARMS
ZONES = ps.ZONES

FROZEN_INPUT_HASHES = {
    "train_pt": "e0f1837b4a19714fcf3a5c6727a30c23f57c3398c8987cab2983214a0e76955b",
    "canonical_smiles": "04a590355fb7bba71e35e52bed60c974999b92af9e96e4ee99bed4afe42dcaff",
    "encoded_train": "10d764955ef1b040a317105be1dd3fbb2e81292b03b79e7d078cce08e18e5f1d",
}
HISTORICAL_SPLITS = {
    "E": {"salt": "cscl-bond-relation-attribution-v1", "path": BRA_RESULTS / "split_E.json"},
    "F": {"salt": "cscl-p3-bond-relation-bridge-v1", "path": P3_RESULTS / "split_F.json"},
    "S": {"salt": "cscl-bjoc-sac-v1", "path": SAC_RESULTS / "split_S.json"},
}
GPU_BUDGET_CAP_HOURS = ps.GPU_BUDGET_CAP_HOURS
SMOKE_N = 512
FIT_TARGET = 7_996
MONITOR_TARGET = 1_000
LOCKED_TARGET = 1_000


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, default=_json_default)
        fh.write("\n")
    tmp.replace(path)


def read_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_sha256(rows: Sequence[int]) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()


def array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes()).hexdigest()


def git_commit() -> str:
    try:
        return (
            __import__("subprocess")
            .check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True)
            .strip()
        )
    except Exception:  # pragma: no cover
        return "unknown"


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"[{PROTOCOL_ID}] {msg}")


def now_utc() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def config() -> Mapping[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def config_sha256() -> str:
    return file_sha256(CONFIG_PATH)


def code_hashes() -> dict[str, str]:
    return {name: file_sha256(_CODE_DIR / name) for name in ("p3_sam_v1.py", "run_p3_sam_v1.py")}


def split_sha256(payload: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(str(payload["salt"]).encode())
    for zone in ("fit", "monitor", "locked"):
        h.update(zone.encode())
        h.update(np.ascontiguousarray(np.asarray(payload[f"{zone}_rows"], dtype=np.int64)).tobytes())
    return h.hexdigest()


def object_zone_sha256(objs: Sequence[sro.MolObjects]) -> str:
    h = hashlib.sha256()
    for obj in objs:
        h.update(np.asarray([int(obj.mol_index)], dtype=np.int64).tobytes())
        for name in ("atom_feat", "bond_feat", "junc_feat", "global_feat", "bond_index", "junc_index", "junc_bond"):
            arr = np.ascontiguousarray(np.asarray(getattr(obj, name)))
            h.update(name.encode())
            h.update(str(arr.dtype).encode())
            h.update(arr.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def split_path() -> Path:
    return OUT / f"split_{SPLIT_TAG}.json"


def objects_path(zone: str) -> Path:
    return OUT / f"objects_{zone}_{SPLIT_TAG}.pt"


def prep_path() -> Path:
    return OUT / f"prep_fit_{SPLIT_TAG}.json"


def ssl_path(seed: int) -> Path:
    return OUT / f"ssl_{SPLIT_TAG}_s{seed}.pt"


def ssl_json_path(seed: int) -> Path:
    return OUT / f"ssl_{SPLIT_TAG}_s{seed}.json"


def soup_path(arm: str, seed: int) -> Path:
    return OUT / f"soup_{arm}_{SPLIT_TAG}_s{seed}.pt"


def train_json_path(arm: str, seed: int) -> Path:
    return OUT / f"train_{arm}_{SPLIT_TAG}_s{seed}.json"


def pred_path(zone: str, arm: str, seed: int) -> Path:
    return OUT / "predictions" / f"{zone}_{arm}_{SPLIT_TAG}_s{seed}.npz"


def smoke_dir() -> Path:
    return OUT / "smoke"


def ledger_path() -> Path:
    return OUT / "gpu_budget_ledger.json"


def freeze_path() -> Path:
    return OUT / "protocol_freeze.json"


def monitor_lock_path() -> Path:
    return OUT / "holdout_read_monitor.lock"


def locked_lock_path() -> Path:
    return OUT / "holdout_read_locked.lock"


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def load_split() -> Mapping[str, Any]:
    require(split_path().exists(), "split_T.json missing: run the `split` stage first")
    split = read_json(split_path())
    require(split["protocol"] == PROTOCOL_ID, "split belongs to another protocol")
    require(split["salt"] == ps.SPLIT_SALT, f"split salt changed: {split['salt']!r}")
    actual = split_sha256(split)
    require(actual == split["split_sha256"], "saved split hash does not match its rows/salt")
    require(bool(split.get("salt_reproduced_from_saved_metadata")), "split salt was not reproduced")
    return split


def load_zone_objects(zone: str) -> list[sro.MolObjects]:
    return torch.load(objects_path(zone), map_location="cpu", weights_only=False)


def load_fit_y() -> np.ndarray:
    split = load_split()
    return bp.load_y_rows(ENCODED_TRAIN, [int(i) for i in split["fit_rows"]])


def load_norm() -> Mapping[str, Any]:
    return read_json(prep_path())


def load_ssl(seed: int) -> Mapping[str, Any]:
    return torch.load(ssl_path(seed), map_location="cpu", weights_only=False)


def load_official_train():
    import cscl_features as cf

    return cf.load_official_train()


def build_guard():
    split = load_split()
    ds = load_official_train()
    guard = ab.LabelGuard(ds, list(split["monitor_rows"]) + list(split["locked_rows"]))
    return split, ds, guard


def _append_ledger(entry: Mapping[str, Any]) -> None:
    with _file_lock(ledger_path()):
        payload = read_json(ledger_path()) if ledger_path().exists() else {"protocol": PROTOCOL_ID, "entries": []}
        payload["entries"].append({**entry, "recorded_utc": now_utc()})
        total = float(sum(float(e.get("wall_seconds", 0.0)) for e in payload["entries"]))
        payload["total_wall_hours"] = total / 3600.0
        payload["cap_hours"] = GPU_BUDGET_CAP_HOURS
        payload["within_cap"] = bool(total / 3600.0 <= GPU_BUDGET_CAP_HOURS)
        write_json(ledger_path(), payload)
    log(f"[ledger] total {payload['total_wall_hours']:.4f} h / cap {GPU_BUDGET_CAP_HOURS} h")
    if not payload["within_cap"]:
        raise SystemExit("GPU budget cap exceeded; stop per protocol")


def _file_lock(path: Path | str):
    import contextlib
    import fcntl

    @contextlib.contextmanager
    def _ctx():
        lock_path = Path(str(path) + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    return _ctx()


# ---------------------------------------------------------------------------
# freeze verification
# ---------------------------------------------------------------------------


def verify_freeze(require_exact_code: bool = True) -> Mapping[str, Any]:
    require(freeze_path().exists(), "protocol_freeze.json missing: run the `freeze` stage first")
    fr = read_json(freeze_path())
    require(fr["protocol"] == PROTOCOL_ID, "freeze belongs to a different protocol")
    if require_exact_code:
        for name, rec in fr["code"].items():
            require(file_sha256(_CODE_DIR / name) == rec["sha256"], f"code changed after freeze: {name}")
        require(file_sha256(CONFIG_PATH) == fr["config"]["sha256"], "config changed after freeze")
    for name, rec in fr["inputs"].items():
        path = OUT / name
        require(path.exists(), f"frozen input missing: {name}")
        require(file_sha256(path) == rec["sha256"], f"frozen input changed: {name}")
    return fr


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def _structure_context_module():
    if str(STRUCT_CONTEXT_CODE) not in sys.path:
        sys.path.insert(0, str(STRUCT_CONTEXT_CODE))
    import structure_context_support_v1 as scs  # noqa: E402

    return scs


def identity_partition(rows: Sequence[int], keys: Mapping[int, str]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for r in rows:
        out.setdefault(str(keys[int(r)]), set()).add(int(r))
    return out


def _zone_overlap(a: Mapping[str, Sequence[int]], b: Mapping[str, Sequence[int]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for za in ("fit", "monitor", "locked"):
        sa = set(int(i) for i in a[za])
        out[za] = {}
        for zb in ("fit", "monitor", "locked"):
            sb = set(int(i) for i in b[zb])
            inter = len(sa & sb)
            out[za][zb] = {
                "n_overlap": int(inter),
                "frac_of_this_zone": float(inter / max(len(sa), 1)),
                "frac_of_other_zone": float(inter / max(len(sb), 1)),
            }
    return out


def cmd_split(_args: argparse.Namespace) -> int:
    from rdkit import Chem

    require(not freeze_path().exists(), "split is frozen already; refusing to rebuild")
    scs = _structure_context_module()
    pairing = scs.pairing_verification()
    write_json(OUT / "pairing_verification.json", pairing)
    require(pairing["n_mismatch"] == 4, f"expected 4 SMILES/PyG mismatches, got {pairing['n_mismatch']}")
    require(tuple(pairing["mismatch_rows"]) == tuple(bra.EXCLUDED_ROWS), f"mismatch rows changed: {pairing['mismatch_rows']}")
    require(pairing["reproduces_frozen_exclusion_list"], "exclusion list does not reproduce")
    rows = [int(r) for r in range(10_000) if int(r) not in set(bra.EXCLUDED_ROWS)]
    require(len(rows) == bra.N_USABLE, "usable row count changed")

    structs = scs.load_pyg_structure_only(rows)  # label-blind: y discarded
    require([int(s["row"]) for s in structs] == rows, "structure rows out of order")
    graph_identity = {int(s["row"]): str(scs._pyg_rebuilt_smiles(s)) for s in structs}
    identity_keys = [graph_identity[r] for r in rows]

    with np.load(CANON, allow_pickle=True) as blob:
        table = [str(s) for s in blob["smiles"]]
    table_identity = {r: Chem.CanonSmiles(table[r], useChiral=False) for r in rows}
    require(
        identity_partition(rows, graph_identity) == identity_partition(rows, table_identity),
        "graph identity partition != committed-SMILES partition",
    )

    # ---- the salt actually used is explicit and asserted -------------------
    used_salt = ps.SPLIT_SALT
    split = bra.build_identity_split(
        rows,
        identity_keys,
        salt=used_salt,
        fit_target=FIT_TARGET,
        monitor_target=MONITOR_TARGET,
    )
    require(split["salt"] == used_salt, f"split builder used salt {split['salt']!r}, expected {used_salt!r}")
    require(split["duplicate_groups_cross_sides"] == 0, "identity group crosses split zones")
    sizes = (len(split["fit_rows"]), len(split["monitor_rows"]), len(split["locked_rows"]))
    require(abs(sizes[0] - FIT_TARGET) <= 10, f"fit size off target: {sizes}")
    require(abs(sizes[1] - MONITOR_TARGET) <= 10, f"monitor size off target: {sizes}")
    require(abs(sizes[2] - LOCKED_TARGET) <= 10, f"locked size off target: {sizes}")
    all_rows = sorted(split["fit_rows"] + split["monitor_rows"] + split["locked_rows"])
    require(all_rows == rows, "split union does not equal the usable rows")

    # ---- independent recomputation from the saved identity metadata -------
    saved_keys = [str(split["identity_of_row"][int(r)]) for r in rows]
    recomputed = bra.build_identity_split(
        rows,
        saved_keys,
        salt=used_salt,
        fit_target=FIT_TARGET,
        monitor_target=MONITOR_TARGET,
    )
    salt_reproduced = all(
        sorted(recomputed[f"{z}_rows"]) == sorted(int(i) for i in split[f"{z}_rows"])
        for z in ("fit", "monitor", "locked")
    )
    require(salt_reproduced, "recomputed split from the saved salt/identity does not match the saved rows")

    own_hashes = {z: rows_sha256(split[f"{z}_rows"]) for z in ("fit", "monitor", "locked")}
    payload_for_hash = {
        "salt": used_salt,
        "fit_rows": split["fit_rows"],
        "monitor_rows": split["monitor_rows"],
        "locked_rows": split["locked_rows"],
    }
    actual_hash = split_sha256(payload_for_hash)
    require(actual_hash == split_sha256({**payload_for_hash}), "split hash is not deterministic")

    # ---- salt is load-bearing ---------------------------------------------
    probe = bra.build_identity_split(
        rows,
        identity_keys,
        salt=used_salt + "-probe",
        fit_target=FIT_TARGET,
        monitor_target=MONITOR_TARGET,
    )
    require(rows_sha256(probe["fit_rows"]) != own_hashes["fit"], "salt probe did not change the split")

    # ---- historical splits: no row-identical zone, no full-split identity ---
    new_sets = {z: [int(i) for i in split[f"{z}_rows"]] for z in ("fit", "monitor", "locked")}
    historical: dict[str, Any] = {}
    old_row_sets: dict[str, dict[str, Sequence[int]]] = {}
    for tag, spec in HISTORICAL_SPLITS.items():
        old_path = spec["path"]
        require(old_path.exists(), f"historical split {tag} missing at {old_path}")
        old = read_json(old_path)
        old_rows = {z: [int(i) for i in old[f"{z}_rows"]] for z in ("fit", "monitor", "locked")}
        old_hashes = {z: rows_sha256(old_rows[z]) for z in ("fit", "monitor", "locked")}
        collisions = {z: bool(own_hashes[z] == old_hashes[z]) for z in own_hashes}
        identical = all(sorted(new_sets[z]) == sorted(old_rows[z]) for z in ("fit", "monitor", "locked"))
        require(not any(collisions.values()), f"new split collides with historical {tag}: {collisions}")
        require(not identical, f"new split is row-identical to historical {tag}")
        historical[tag] = {
            "path": str(old_path.relative_to(REPO_ROOT)),
            "recorded_salt": old.get("salt"),
            "expected_salt": spec["salt"],
            "fit_sha256_rows": old_hashes["fit"],
            "monitor_sha256_rows": old_hashes["monitor"],
            "locked_sha256_rows": old_hashes["locked"],
            "zone_hash_collisions": collisions,
            "row_identical": bool(identical),
            "overlap": _zone_overlap(new_sets, old_rows),
        }
        old_row_sets[tag] = old_rows

    seen = sorted({r for tag in old_row_sets for z in old_row_sets[tag] for r in old_row_sets[tag][z]})
    seen_set = set(seen)
    seen_fraction = float(sum(1 for r in rows if r in seen_set) / len(rows))

    identity_digest = hashlib.sha256(
        "\n".join(f"{r}\t{graph_identity[r]}" for r in rows).encode()
    ).hexdigest()
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": SPLIT_TAG,
        "salt": used_salt,
        "salt_source": "explicit argument p3_sam_v1.SPLIT_SALT (never a builder default)",
        "split_sha256": actual_hash,
        "identity_digest": identity_digest,
        "fit_rows": new_sets["fit"],
        "monitor_rows": new_sets["monitor"],
        "locked_rows": new_sets["locked"],
        "fit_sha256_rows": own_hashes["fit"],
        "monitor_sha256_rows": own_hashes["monitor"],
        "locked_sha256_rows": own_hashes["locked"],
        "sizes": {"fit": len(new_sets["fit"]), "monitor": len(new_sets["monitor"]), "locked": len(new_sets["locked"])},
        "duplicate_groups": split["duplicate_groups"],
        "duplicate_groups_cross_sides": int(split["duplicate_groups_cross_sides"]),
        "identity_of_row": split["identity_of_row"],
        "group_id": split["group_id"],
        "salt_reproduced_from_saved_metadata": bool(salt_reproduced),
        "salt_is_load_bearing": True,
        "historical_splits": historical,
        "historical_rows_seen_fraction": seen_fraction,
        "historical_rows_seen_note": (
            "The same official-train 10k has been used by many prior rounds; split T is a new "
            "resampling of an exposed population, not an independent blind test."
        ),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "labels_read": False,
    }
    write_json(split_path(), payload)
    write_json(
        OUT / f"split_compare_{SPLIT_TAG}.json",
        {
            "protocol": PROTOCOL_ID,
            "git_commit": git_commit(),
            "new_split": SPLIT_TAG,
            "new_split_sha256": actual_hash,
            "new_salt": used_salt,
            "historical": historical,
            "historical_rows_seen_fraction": seen_fraction,
            "note": payload["historical_rows_seen_note"],
        },
    )
    log(
        f"[split] salt={used_salt!r} sizes={sizes} hash={actual_hash[:16]} "
        f"seen_frac={seen_fraction:.3f}"
    )
    return 0


# ---------------------------------------------------------------------------
# verify inputs
# ---------------------------------------------------------------------------


def cmd_verify_inputs(_args: argparse.Namespace) -> int:
    actual = {
        "train_pt": file_sha256(DATA_TRAIN),
        "canonical_smiles": file_sha256(CANON),
        "encoded_train": file_sha256(ENCODED_TRAIN),
    }
    require(actual == FROZEN_INPUT_HASHES, f"frozen input hashes changed: {actual}")

    cfg = config()
    require(cfg["protocol"] == PROTOCOL_ID, "config protocol mismatch")
    require(cfg["split_tag"] == SPLIT_TAG, "config split tag mismatch")
    require(cfg["salt"] == ps.SPLIT_SALT, "config salt mismatch")
    require(tuple(cfg["seeds"]) == SEEDS, "config seeds mismatch")
    require(tuple(cfg["arms"]) == ARMS, "config arms mismatch")
    require(float(cfg["sam"]["rho"]) == ps.SAM_RHO, "config SAM rho mismatch")
    require(int(cfg["sam"]["steps"]) == ps.SAM_STEPS, "config SAM steps mismatch")
    require(bool(cfg["sam"]["first_gradient_clipped"]) is False, "config first-gradient clip mismatch")
    require(bool(cfg["sam"]["clip_after_second_backward"]) is True, "config clip-order mismatch")
    require(int(cfg["base_parameters"]) == ps.BASE_PARAMETERS, "config P3 parameter count mismatch")
    require(int(cfg["param_cap"]) == ps.PARAM_CAP, "config parameter cap mismatch")
    require(float(cfg["gate"]["meaningful_delta"]) == ps.MEANINGFUL_DELTA, "config meaningful delta mismatch")
    require(float(cfg["gate"]["rmse_tol"]) == ps.RMSE_TOL, "config RMSE tolerance mismatch")
    require(float(cfg["gate"]["tail_tol"]) == ps.TAIL_TOL, "config tail tolerance mismatch")
    require(int(cfg["gate"]["bootstrap_n"]) == ps.BOOTSTRAP_N, "config bootstrap n mismatch")
    require(int(cfg["gate"]["bootstrap_seed"]) == ps.BOOTSTRAP_SEED, "config bootstrap seed mismatch")
    require(float(cfg["budget"]["gpu_hours"]) == GPU_BUDGET_CAP_HOURS, "config budget mismatch")
    recipe = bp.RECIPES[cfg["recipe"]["key"]]
    require(int(cfg["recipe"]["epochs"]) == int(recipe["epochs"]), "config epochs != frozen R3 recipe")
    require(float(cfg["recipe"]["lr"]) == float(recipe["lr"]), "config lr != frozen R3 recipe")
    require(float(cfg["recipe"]["lr_min"]) == float(recipe["lr_min"]), "config lr_min != frozen R3 recipe")
    require(float(cfg["recipe"]["weight_decay"]) == float(recipe["weight_decay"]), "config wd != frozen R3 recipe")
    require(int(cfg["recipe"]["batch_size"]) == int(bp.BATCH_SIZE), "config batch size != frozen recipe")
    require(int(cfg["recipe"]["soup_last"]) == int(bp.SOUP_LAST), "config soup != frozen recipe")
    require(int(cfg["recipe"]["shuffle_offset"]) == int(bp.TRAIN_SHUFFLE_OFFSET), "config shuffle offset != frozen recipe")
    require(float(cfg["recipe"]["grad_clip"]) == float(bp.GRAD_CLIP), "config grad clip != frozen recipe")
    require(int(cfg["ssl"]["updates"]) == int(bp.PRETRAIN_UPDATES), "config SSL updates != frozen SSL budget")
    require(int(cfg["split"]["fit_target"]) == FIT_TARGET, "config fit target mismatch")
    require(int(cfg["split"]["monitor_target"]) == MONITOR_TARGET, "config monitor target mismatch")
    require(int(cfg["split"]["locked_target"]) == LOCKED_TARGET, "config locked target mismatch")
    require(tuple(cfg["split"]["excluded_rows"]) == tuple(bra.EXCLUDED_ROWS), "config excluded rows mismatch")

    split = load_split()
    require(split["split_sha256"] == split_sha256(split), "split hash mismatch")

    upstream = {
        "structural_relation_composer_v1.py": _CODE_DIR / "structural_relation_composer_v1.py",
        "structural_relation_objects_v1.py": _CODE_DIR / "structural_relation_objects_v1.py",
        "bjoc_perfmech_long_v1.py": _CODE_DIR / "bjoc_perfmech_long_v1.py",
        "bjoc_masked_pretraining_v1.py": _CODE_DIR / "bjoc_masked_pretraining_v1.py",
        "bjoc_arch_breakthrough_v1.py": _CODE_DIR / "bjoc_arch_breakthrough_v1.py",
        "bond_relation_attribution_v1.py": _CODE_DIR / "bond_relation_attribution_v1.py",
    }
    reuse = {name: file_sha256(path) for name, path in upstream.items()}
    p3_audit = read_json(P3_RESULTS / "input_audit.json")
    expected = dict(p3_audit["reused_modules"])
    for name, sha in expected.items():
        require(name in reuse, f"P3 upstream module not checked: {name}")
        require(reuse[name] == sha, f"upstream module changed since the P3 round: {name}")

    norm = {"global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32), "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32)}
    p3 = ps.build_p3(0, norm, "cpu")
    audit = {
        "p3_parameters": ps.model_parameters(p3),
        "reader_parameters": int(sum(p.numel() for p in p3.reader.parameters())),
        "reader_in": int(p3.config.reader_in),
    }
    require(audit["p3_parameters"] == ps.BASE_PARAMETERS, "P3 parameter audit failed")
    require(audit["p3_parameters"] <= ps.PARAM_CAP, "P3 exceeds the parameter cap")

    split_tokens = ["split=" + q + name + q for q in ('"', "'") for name in ("valid", "test")]
    path_tokens = ["processed/" + name for name in ("valid", "test")]
    scan_hits = []
    for path in (_CODE_DIR / "p3_sam_v1.py", _CODE_DIR / "run_p3_sam_v1.py"):
        text = path.read_text(encoding="utf-8")
        for token in split_tokens + path_tokens:
            if token in text:
                scan_hits.append(f"{path.name}:{token}")
    require(not scan_hits, f"official valid/test loader token found: {scan_hits}")

    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "input_hashes": actual,
        "input_hashes_match_frozen": True,
        "upstream_code_hashes": reuse,
        "upstream_matches_p3_round": True,
        "parameter_audit": audit,
        "recipe": dict(recipe),
        "split_sizes": split["sizes"],
        "split_sha256": split["split_sha256"],
        "official_loader_scan_hits": scan_hits,
        "label_guard_selfcheck": ab.label_guard_selfcheck(),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "input_audit_T.json", payload)
    log(f"[verify-inputs] P3={audit['p3_parameters']} split={split['sizes']} hash={split['split_sha256'][:16]}")
    return 0


# ---------------------------------------------------------------------------
# objects
# ---------------------------------------------------------------------------


def cmd_objects(_args: argparse.Namespace) -> int:
    require(not freeze_path().exists(), "objects are frozen already; refusing to rebuild")
    split, ds, guard = build_guard()
    started = time.perf_counter()
    objs: dict[str, list[sro.MolObjects]] = {}
    for zone in ("fit", "monitor", "locked"):
        rows = [int(i) for i in split[f"{zone}_rows"]]
        built = []
        for i in rows:
            d = guard[int(i)]
            built.append(sro.build_mol_objects(int(i), d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy()))
        require([o.mol_index for o in built] == rows, f"object row order mismatch in {zone}")
        torch.save(built, objects_path(zone))
        objs[zone] = built
    prep = bp.fit_standardiser(objs["fit"])
    write_json(prep_path(), prep)
    require(guard.access_attempts == [], "guarded label access during object build")
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "build_seconds": time.perf_counter() - started,
        "zones": {
            zone: {
                "n": len(objs[zone]),
                "rows_sha256": rows_sha256([o.mol_index for o in objs[zone]]),
                "object_state_sha256": object_zone_sha256(objs[zone]),
            }
            for zone in objs
        },
        "prep_fit_sha256": file_sha256(prep_path()),
        "guarded_rows": 2000,
        "forbidden_label_access_attempts": len(guard.access_attempts),
        "monitor_labels_read": False,
        "locked_labels_read": False,
    }
    write_json(OUT / "objects_manifest_T.json", payload)
    log(f"[objects] built in {payload['build_seconds']:.1f}s; standardiser fit-only")
    return 0


# ---------------------------------------------------------------------------
# CPU gates
# ---------------------------------------------------------------------------


def cmd_cpu_gates(_args: argparse.Namespace) -> int:
    require(not freeze_path().exists(), "CPU gates are frozen already; refusing to rebuild")
    split = load_split()
    n_fit = len(split["fit_rows"])
    result = ps.run_sam_unit_gates(n_fit=n_fit)
    extra = {
        "split_sizes": split["sizes"],
        "split_sha256": split["split_sha256"],
        "fit_updates_expected": int(bp.RECIPES[ps.RECIPE_KEY]["epochs"]) * math.ceil(n_fit / bp.BATCH_SIZE),
        "aggregate_ok": bool(result["all_passed"] and split["split_sha256"] == split_sha256(split)),
    }
    result.update(extra)
    result["all_passed"] = bool(result["all_passed"] and extra["aggregate_ok"])
    write_json(OUT / "cpu_gates_T.json", result)
    for gate in result["gates"]:
        log(f"[cpu-gates {gate['name']}] {'PASS' if gate['ok'] else 'FAIL'}")
    require(result["all_passed"], "CPU gates did not all pass")
    log(f"[cpu-gates] all {len(result['gates'])} gates passed; no ZINC/GPU/labels")
    return 0


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------


def cmd_freeze(args: argparse.Namespace) -> int:
    amendment = int(getattr(args, "amendment", 0) or 0)
    if amendment == 0:
        require(not freeze_path().exists(), "already frozen")
    else:
        require(freeze_path().exists(), "an amendment requires an existing freeze")
        old = read_json(freeze_path())
        require(int(old.get("amendment", 0)) == amendment - 1, "amendment numbers must be consecutive")
        require(bool(getattr(args, "amendment_reason", None)), "an amendment requires --amendment-reason")
    gates = read_json(OUT / "cpu_gates_T.json")
    require(gates["all_passed"], "CPU gates did not all pass")
    require(bool(args.attest_no_gpu_before_freeze), "--attest-no-gpu-before-freeze is required")
    inputs = {
        split_path().name: split_path(),
        prep_path().name: prep_path(),
        "objects_manifest_T.json": OUT / "objects_manifest_T.json",
        "cpu_gates_T.json": OUT / "cpu_gates_T.json",
        "input_audit_T.json": OUT / "input_audit_T.json",
        "split_compare_T.json": OUT / "split_compare_T.json",
        "pairing_verification.json": OUT / "pairing_verification.json",
        "PROTOCOL.md": PROTOCOL_DOC,
    }
    payload = {
        "protocol": PROTOCOL_ID,
        "split_tag": SPLIT_TAG,
        "amendment": amendment,
        "amendment_reason": getattr(args, "amendment_reason", None),
        "git_commit": git_commit(),
        "frozen_utc": now_utc(),
        "code": {name: {"sha256": sha} for name, sha in code_hashes().items()},
        "config": {"path": str(CONFIG_PATH.relative_to(REPO_ROOT)), "sha256": config_sha256()},
        "inputs": {name: {"sha256": file_sha256(path)} for name, path in inputs.items()},
        "upstream_input_hashes": FROZEN_INPUT_HASHES,
        "parameters": {"p3": ps.BASE_PARAMETERS, "cap": ps.PARAM_CAP},
        "sam": {
            "variant": "standard_two_step_global_l2",
            "rho": ps.SAM_RHO,
            "steps": ps.SAM_STEPS,
            "first_gradient_clipped": False,
            "second_gradient_clipped": True,
            "clip_after_second_backward": True,
        },
        "budget": {
            "cap_gpu_hours": GPU_BUDGET_CAP_HOURS,
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "gpu_assignment": {"0": 0, "1": 1},
            "no_seed_extension": True,
            "no_rho_search": True,
        },
        "labels_before_freeze": "Fit y only (CPU gate P3/SAM equivalence)",
        "monitor_labels_read": False,
        "locked_labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "gpu_started_before_freeze": False,
    }
    write_json(freeze_path(), payload)
    log(
        f"[freeze] amendment={amendment} code={payload['code']['p3_sam_v1.py']['sha256'][:12]} "
        f"config={payload['config']['sha256'][:12]}"
    )
    return 0


# ---------------------------------------------------------------------------
# stage inputs on a compute checkout
# ---------------------------------------------------------------------------


def cmd_stage_inputs(_args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    require(not monitor_lock_path().exists() and not locked_lock_path().exists(), "refusing to restage after labels were read")
    split, ds, guard = build_guard()
    frozen_manifest = read_json(OUT / "objects_manifest_T.json")
    frozen_prep = read_json(prep_path())
    started = time.perf_counter()
    objs: dict[str, list[sro.MolObjects]] = {}
    for zone in ("fit", "monitor", "locked"):
        rows = [int(i) for i in split[f"{zone}_rows"]]
        built = []
        for i in rows:
            d = guard[int(i)]
            built.append(sro.build_mol_objects(int(i), d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy()))
        require([o.mol_index for o in built] == rows, f"object row order mismatch in {zone}")
        torch.save(built, objects_path(zone))
        objs[zone] = built
    prep = bp.fit_standardiser(objs["fit"])
    mean_diff = float(np.abs(np.asarray(prep["global_mean"]) - np.asarray(frozen_prep["global_mean"])).max())
    scale_diff = float(np.abs(np.asarray(prep["global_scale"]) - np.asarray(frozen_prep["global_scale"])).max())
    require(mean_diff <= 1e-12 and scale_diff <= 1e-12, "rebuild changed the Fit-only standardiser")

    payload: dict[str, Any] = {"protocol": PROTOCOL_ID, "git_commit": git_commit(), "zones": {}}
    for zone in ("fit", "monitor", "locked"):
        sha = object_zone_sha256(objs[zone])
        expected = frozen_manifest["zones"][zone]["object_state_sha256"]
        require(sha == expected, f"{zone}: rebuilt object tensors changed")
        require(rows_sha256([o.mol_index for o in objs[zone]]) == frozen_manifest["zones"][zone]["rows_sha256"], f"{zone}: row hash changed")
        payload["zones"][zone] = {
            "n_molecules": len(objs[zone]),
            "rows_sha256": frozen_manifest["zones"][zone]["rows_sha256"],
            "object_state_sha256": sha,
            "objects_file_sha256": file_sha256(objects_path(zone)),
            "matches_frozen_semantics": True,
        }
        log(f"[stage-inputs {zone}] n={len(objs[zone])} object_sha={sha[:12]} (match frozen)")
    require(guard.access_attempts == [], "guarded label access during staging")
    payload.update(
        {
            "freeze_sha256": file_sha256(freeze_path()),
            "standardiser_max_abs_diff": {"mean": mean_diff, "scale": scale_diff},
            "build_seconds": time.perf_counter() - started,
            "forbidden_label_access_attempts": len(guard.access_attempts),
            "note": "torch .pt bytes are machine-local; equality is verified semantically",
            "monitor_labels_read": False,
            "locked_labels_read": False,
        }
    )
    write_json(OUT / "staged_inputs_T.json", payload)
    log(f"[stage-inputs] ok in {payload['build_seconds']:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def cmd_smoke(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    device = torch.device(args.device)
    split, ds, guard = build_guard()
    fit_objs = load_zone_objects("fit")
    require([o.mol_index for o in fit_objs] == [int(i) for i in split["fit_rows"]], "fit objects do not match split")
    mini = list(fit_objs[:SMOKE_N])
    norm = load_norm()
    y_fit = load_fit_y()
    y_mini = y_fit[: len(mini)]
    smoke_dir().mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"protocol": PROTOCOL_ID, "device": str(device), "n_mini": len(mini), "arms": {}}

    ssl = bp.pretrain_masked(
        seed=1,
        fit_objs=fit_objs,
        fit_rows=[int(i) for i in split["fit_rows"]],
        raw_dataset=guard,
        device=device,
        mode="atom_bond",
        smoke_updates=20,
        log_fn=log,
    )
    require(guard.access_attempts == [], "guarded label access during smoke SSL")
    enc = ssl["encoder_state"]
    torch.save({g: {k: v.cpu() for k, v in s.items()} for g, s in enc.items()}, smoke_dir() / "ssl_s1_smoke.pt")
    ssl_projected = float(ssl["wall_seconds"]) / 20.0 * int(bp.PRETRAIN_UPDATES)
    out["ssl"] = {
        "updates": 20,
        "wall_seconds": float(ssl["wall_seconds"]),
        "updates_per_second": 20.0 / max(float(ssl["wall_seconds"]), 1e-9),
        "projected_full_seconds": ssl_projected,
    }

    p3_info = bp.train_supervised(
        arm="P3",
        seed=1,
        recipe_key=ps.RECIPE_KEY,
        init="ssl",
        train_objs=mini,
        rows=[o.mol_index for o in mini],
        y_fit=y_mini,
        norm=norm,
        device=device,
        ssl_state=enc,
        model_kind="base",
        smoke_epochs=1,
        log_fn=log,
    )
    p3_epoch = float(p3_info["lr_curve"][-1]["seconds"])
    p3_projected = p3_epoch * (len(fit_objs) / len(mini)) * int(bp.RECIPES[ps.RECIPE_KEY]["epochs"])

    model_sam = ps.build_p3(1, norm, device)
    mp.load_encoder_state(model_sam, enc, device)
    sam_info = ps.train_supervised_sam(
        arm="P3SAM",
        seed=1,
        model=model_sam,
        train_objs=mini,
        y_fit=y_mini,
        device=device,
        epochs=1,
        log_fn=log,
    )
    sam_epoch = float(sam_info["curve"][-1]["seconds"])
    sam_projected = sam_epoch * (len(fit_objs) / len(mini)) * int(bp.RECIPES[ps.RECIPE_KEY]["epochs"])

    predict_projected = 60.0  # conservative per-seed estimate for 2 zones x 2 arms
    per_seed = {
        "SSL": ssl_projected,
        "P3": p3_projected,
        "P3SAM": sam_projected,
        "predict_estimate": predict_projected,
    }
    two_seed = 2.0 * sum(per_seed.values())
    out["arms"]["P3"] = {
        "parameters": int(p3_info["parameter_audit"]["actual_parameters"]),
        "epoch_seconds_on_mini": p3_epoch,
        "projected_360_epoch_seconds": p3_projected,
        "molecules_per_second": float(len(mini) / max(p3_epoch, 1e-9)),
        "wall_seconds": float(p3_info["wall_seconds"]),
    }
    out["arms"]["P3SAM"] = {
        "parameters": int(sam_info["parameter_audit"]["actual_parameters"]),
        "epoch_seconds_on_mini": sam_epoch,
        "projected_360_epoch_seconds": sam_projected,
        "molecules_per_second": float(len(mini) / max(sam_epoch, 1e-9)),
        "wall_seconds": float(sam_info["wall_seconds"]),
        "forward_backward_multiplier": 2.0,
    }
    out["projection"] = {
        "n_fit": len(fit_objs),
        "per_seed_seconds": per_seed,
        "two_seed_total_seconds": float(two_seed),
        "two_seed_total_gpu_hours": float(two_seed / 3600.0),
        "cap_gpu_hours": GPU_BUDGET_CAP_HOURS,
        "within_cap": bool(two_seed / 3600.0 <= GPU_BUDGET_CAP_HOURS * 0.9),
    }
    out["device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    require(guard.access_attempts == [], "guarded label access during smoke training")
    write_json(smoke_dir() / "smoke_T.json", out)
    _append_ledger({"phase": "smoke_ssl", "arm": "SSL", "seed": 1, "wall_seconds": float(ssl["wall_seconds"]), "device": str(device)})
    _append_ledger({"phase": "smoke_train", "arm": "P3", "seed": 1, "wall_seconds": float(p3_info["wall_seconds"]), "device": str(device)})
    _append_ledger({"phase": "smoke_train", "arm": "P3SAM", "seed": 1, "wall_seconds": float(sam_info["wall_seconds"]), "device": str(device)})
    log(
        f"[smoke] ok device={out['device_name']} projected total={out['projection']['two_seed_total_gpu_hours']:.2f} GPU-h "
        f"(cap {GPU_BUDGET_CAP_HOURS})"
    )
    require(out["projection"]["within_cap"], "measured throughput projects beyond the GPU budget cap")
    return 0


# ---------------------------------------------------------------------------
# ssl
# ---------------------------------------------------------------------------


def _ssl_encoder_hashes(enc: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, str]:
    combined = (
        {f"atom:{k}": v for k, v in enc["atom_encoder"].items()}
        | {f"bond:{k}": v for k, v in enc["bond_encoder"].items()}
        | {f"junc:{k}": v for k, v in enc["junction_encoder"].items()}
    )
    return {
        "full_encoder_group_sha256": bp.state_group_sha256(combined),
        "atom_encoder_sha256": bp.state_group_sha256({k: v for k, v in combined.items() if k.startswith("atom:")}),
    }


def cmd_ssl(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    seed = int(args.seed)
    smoke_updates = int(args.smoke_updates) if args.smoke_updates else None
    if smoke_updates is None and (ssl_path(seed).exists() or ssl_json_path(seed).exists()):
        raise SystemExit(f"SSL for seed {seed} already exists; retraining is not allowed")
    split, ds, guard = build_guard()
    objs = load_zone_objects("fit")
    require([o.mol_index for o in objs] == [int(i) for i in split["fit_rows"]], "fit objects do not match split")
    log(f"[ssl s{seed}] device={args.device} n={len(objs)} updates={smoke_updates or bp.PRETRAIN_UPDATES}")
    res = bp.pretrain_masked(
        seed=seed,
        fit_objs=objs,
        fit_rows=[int(i) for i in split["fit_rows"]],
        raw_dataset=guard,
        device=torch.device(args.device),
        mode="atom_bond",
        smoke_updates=smoke_updates,
        log_fn=log,
    )
    require(guard.access_attempts == [], f"guarded label access during SSL: {guard.access_attempts[:3]}")
    require(res.get("labels_read") is False, "SSL unexpectedly read labels")
    enc = res["encoder_state"]
    out_pt = ssl_path(seed) if smoke_updates is None else smoke_dir() / f"ssl_T_s{seed}_smoke.pt"
    torch.save({g: {k: v.cpu() for k, v in s.items()} for g, s in enc.items()}, out_pt)
    hashes = _ssl_encoder_hashes(enc)
    res.pop("encoder_state", None)
    res.update(
        {
            "protocol": PROTOCOL_ID,
            "split_tag": SPLIT_TAG,
            "seed": seed,
            "smoke_updates": smoke_updates,
            "encoder_path": str(out_pt.relative_to(REPO_ROOT)),
            "encoder_file_sha256": file_sha256(out_pt),
            **hashes,
            "forbidden_label_access_attempts": len(guard.access_attempts),
            "official_valid_loaded": False,
            "official_test_loaded": False,
        }
    )
    require(hashes["full_encoder_group_sha256"] == res["encoder_final_sha256"], "SSL full-encoder hash convention changed")
    if smoke_updates is None:
        write_json(ssl_json_path(seed), res)
        _append_ledger({"phase": "ssl", "arm": "SSL", "seed": seed, "wall_seconds": res["wall_seconds"], "device": res["device"]})
    else:
        write_json(smoke_dir() / f"ssl_T_s{seed}_smoke.json", res)
    log(f"[ssl s{seed}] wall={res['wall_seconds']:.0f}s atom_acc={res.get('final_acc_atom'):.4f}")
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    seed = int(args.seed)
    device = torch.device(args.device)
    norm = load_norm()
    ssl = load_ssl(seed)
    split = load_split()
    fit_objs = load_zone_objects("fit")
    require([o.mol_index for o in fit_objs] == [int(i) for i in split["fit_rows"]], "fit objects do not match split")
    y_fit = load_fit_y()
    _, ds, guard = build_guard()
    out: dict[str, Any] = {"protocol": PROTOCOL_ID, "seed": seed, "device": str(device), "arms": {}}

    for arm in ARMS:
        if train_json_path(arm, seed).exists() and not args.retrain:
            raise SystemExit(f"{arm} seed {seed} already trained; refusing to retrain")
    # ---- A: frozen P3 trainer ---------------------------------------------
    info_p3 = bp.train_supervised(
        arm="P3",
        seed=seed,
        recipe_key=ps.RECIPE_KEY,
        init="ssl",
        train_objs=fit_objs,
        rows=[int(i) for i in split["fit_rows"]],
        y_fit=y_fit,
        norm=norm,
        device=device,
        ssl_state=ssl,
        model_kind="base",
        log_fn=log,
    )
    require(guard.access_attempts == [], "guarded label access during P3 training")
    p3_soup = info_p3.pop("soup_state")
    torch.save(p3_soup, soup_path("P3", seed))
    info_p3.update(
        {
            "protocol": PROTOCOL_ID,
            "split_tag": SPLIT_TAG,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "soup_file": str(soup_path("P3", seed).relative_to(REPO_ROOT)),
            "soup_sha256": file_sha256(soup_path("P3", seed)),
            "forbidden_label_access_attempts": len(guard.access_attempts),
            "official_valid_loaded": False,
            "official_test_loaded": False,
            "cost": {
                "optimizer_updates": int(info_p3["updates"]),
                "forward_calls": int(info_p3["updates"]),
                "backward_calls": int(info_p3["updates"]),
                "relative_forward_backward_multiplier_vs_p3": 1.0,
            },
        }
    )
    write_json(train_json_path("P3", seed), info_p3)
    _append_ledger({"phase": "train", "arm": "P3", "seed": seed, "wall_seconds": info_p3["wall_seconds"], "device": str(device)})
    out["arms"]["P3"] = {
        "soup_fit_mae": info_p3["soup_fit_mae"],
        "wall_seconds": info_p3["wall_seconds"],
        "parameters": info_p3["parameter_audit"]["actual_parameters"],
        "updates": info_p3["updates"],
        "batch_order_sha256": info_p3["batch_order"]["full_hash"],
        "soup_sha256": info_p3["soup_sha256"],
    }

    # ---- B: same init / SSL / order / budget, two-step SAM ----------------
    model_sam = ps.build_p3(seed, norm, device)
    init_pre_ssl = ps.full_state_hash(model_sam)
    init_group_pre_ssl = ps.shared_group_hash(model_sam)
    init_encoder_pre_ssl = ps.encoder_hash(model_sam)
    mp.load_encoder_state(model_sam, ssl, device)
    init_post_ssl = ps.full_state_hash(model_sam)
    encoder_post_ssl = ps.encoder_hash(model_sam)
    require(init_pre_ssl == info_p3["init_full_sha256"], "P3/SAM pre-SSL init full hash mismatch")
    require(init_group_pre_ssl == info_p3["init_group_state_sha256"], "P3/SAM pre-SSL group hash mismatch")
    require(init_encoder_pre_ssl == info_p3["init_encoder_sha256"], "P3/SAM pre-SSL encoder hash mismatch")
    require(encoder_post_ssl == info_p3["encoder_used_sha256"], "P3/SAM SSL encoder hash mismatch")

    info_sam = ps.train_supervised_sam(
        arm="P3SAM",
        seed=seed,
        model=model_sam,
        train_objs=fit_objs,
        y_fit=y_fit,
        device=device,
        log_fn=log,
    )
    require(guard.access_attempts == [], "guarded label access during SAM training")
    require(info_sam["updates"] == info_p3["updates"], "P3/SAM update budget mismatch")
    require(info_sam["batch_order"]["full_hash"] == info_p3["batch_order"]["full_hash"], "P3/SAM batch order mismatch")
    require(info_sam["batch_order"]["full_matches"] is True, "SAM batch order hash is not reproducible")
    require(info_sam["parameter_audit"]["actual_parameters"] == ps.BASE_PARAMETERS, "SAM model parameter audit failed")
    require(info_sam["anomalies"]["restore_max_abs_error"] == 0.0, "SAM parameter restore was not exact")
    require(
        abs(info_sam["anomalies"]["perturbation_norm_min"] - ps.SAM_RHO) <= 1e-6
        and abs(info_sam["anomalies"]["perturbation_norm_max"] - ps.SAM_RHO) <= 1e-6,
        "SAM perturbation norm is not rho",
    )
    sam_soup = info_sam.pop("soup_state")
    torch.save(sam_soup, soup_path("P3SAM", seed))
    info_sam.update(
        {
            "protocol": PROTOCOL_ID,
            "split_tag": SPLIT_TAG,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "init": {
                "pre_ssl_full_sha256": init_pre_ssl,
                "pre_ssl_group_sha256": init_group_pre_ssl,
                "pre_ssl_encoder_sha256": init_encoder_pre_ssl,
                "post_ssl_full_sha256": init_post_ssl,
                "post_ssl_encoder_sha256": encoder_post_ssl,
            },
            "init_matches_p3": True,
            "soup_file": str(soup_path("P3SAM", seed).relative_to(REPO_ROOT)),
            "soup_sha256": file_sha256(soup_path("P3SAM", seed)),
            "forbidden_label_access_attempts": len(guard.access_attempts),
            "official_valid_loaded": False,
            "official_test_loaded": False,
        }
    )
    write_json(train_json_path("P3SAM", seed), info_sam)
    _append_ledger({"phase": "train", "arm": "P3SAM", "seed": seed, "wall_seconds": info_sam["wall_seconds"], "device": str(device)})
    out["arms"]["P3SAM"] = {
        "soup_fit_mae": info_sam["soup_fit_mae"],
        "wall_seconds": info_sam["wall_seconds"],
        "parameters": info_sam["parameter_audit"]["actual_parameters"],
        "updates": info_sam["updates"],
        "batch_order_sha256": info_sam["batch_order"]["full_hash"],
        "soup_sha256": info_sam["soup_sha256"],
        "forward_backward_multiplier": 2.0,
    }
    out["equality"] = {
        "same_pre_ssl_init": True,
        "same_ssl_encoder": True,
        "same_batch_order": True,
        "same_update_budget": True,
        "p3_updates": int(info_p3["updates"]),
        "sam_updates": int(info_sam["updates"]),
        "p3_batch_order_sha256": info_p3["batch_order"]["full_hash"],
        "sam_batch_order_sha256": info_sam["batch_order"]["full_hash"],
        "p3_soup_sha256": info_p3["soup_sha256"],
        "sam_soup_sha256": info_sam["soup_sha256"],
    }
    write_json(OUT / f"train_round_{SPLIT_TAG}_s{seed}.json", out)
    log(
        f"[train s{seed}] P3 fit={info_p3['soup_fit_mae']:.6f} wall={info_p3['wall_seconds']:.0f}s | "
        f"P3SAM fit={info_sam['soup_fit_mae']:.6f} wall={info_sam['wall_seconds']:.0f}s"
    )
    return 0


# ---------------------------------------------------------------------------
# predict / freeze predictions
# ---------------------------------------------------------------------------


def _load_soup_model(arm: str, seed: int, norm, device) -> torch.nn.Module:
    model = ps.build_p3(seed, norm, device)
    state = torch.load(soup_path(arm, seed), map_location="cpu", weights_only=False)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.eval()
    return model


def _record_prediction(path: Path, zone: str, arm: str, seed: int, pred: np.ndarray, rows: Sequence[int]) -> dict[str, Any]:
    require(pred.shape == (len(rows),), f"{zone}/{arm}: prediction shape {pred.shape}")
    require(bool(np.all(np.isfinite(pred))), f"{zone}/{arm}: non-finite prediction")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, pred=pred.astype(np.float64), rows=np.asarray(rows, dtype=np.int64))
    record = {
        "file": path.name,
        "file_sha256": file_sha256(path),
        "zone": zone,
        "arm": arm,
        "kind": "own",
        "seed": int(seed),
        "n": int(len(pred)),
        "pred_sha256": array_sha256(pred),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "rows_sha256": rows_sha256(list(rows)),
        "frozen_utc": now_utc(),
    }
    with _file_lock(OUT / "prediction_manifest_T.json"):
        manifest = (
            read_json(OUT / "prediction_manifest_T.json")
            if (OUT / "prediction_manifest_T.json").exists()
            else {"protocol": PROTOCOL_ID, "records": {}, "complete": False}
        )
        manifest["records"][path.name] = record
        write_json(OUT / "prediction_manifest_T.json", manifest)
    return record


def cmd_predict(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    seed = int(args.seed)
    device = torch.device(args.device)
    norm = load_norm()
    models = {arm: _load_soup_model(arm, seed, norm, device) for arm in ARMS}
    for zone in ZONES:
        objs = load_zone_objects(zone)
        rows = [int(o.mol_index) for o in objs]
        for arm in ARMS:
            pred = ps.predict_model(models[arm], objs, device)
            rec = _record_prediction(pred_path(zone, arm, seed), zone, arm, seed, pred, rows)
            log(f"[predict {zone} {arm} s{seed}] sha={rec['pred_sha256'][:12]} mean={rec['pred_mean']:.4f}")
    return 0


def cmd_freeze_predictions(_args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    manifest = read_json(OUT / "prediction_manifest_T.json")
    expected = {pred_path(z, a, s).name for z in ZONES for a in ARMS for s in SEEDS}
    missing = sorted(expected - set(manifest["records"]))
    require(not missing, f"prediction set incomplete: {missing}")
    hashes = []
    for name in sorted(manifest["records"]):
        path = OUT / "predictions" / name
        require(file_sha256(path) == manifest["records"][name]["file_sha256"], f"prediction drifted: {name}")
        hashes.append(f"{name}:{manifest['records'][name]['file_sha256']}")
    combined = hashlib.sha256("\n".join(hashes).encode()).hexdigest()

    checkpoints: dict[str, Any] = {}
    for arm in ARMS:
        for seed in SEEDS:
            info = read_json(train_json_path(arm, seed))
            path = soup_path(arm, seed)
            require(path.exists(), f"checkpoint missing: {path}")
            sha = file_sha256(path)
            require(sha == info["soup_sha256"], f"checkpoint drifted: {path.name}")
            checkpoints[f"{arm}_s{seed}"] = {
                "file": str(path.relative_to(REPO_ROOT)),
                "sha256": sha,
                "train_json_sha256": file_sha256(train_json_path(arm, seed)),
            }
    ssl_checkpoints = {}
    for seed in SEEDS:
        info = read_json(ssl_json_path(seed))
        path = ssl_path(seed)
        sha = file_sha256(path)
        require(sha == info["encoder_file_sha256"], f"SSL checkpoint drifted: {path.name}")
        ssl_checkpoints[f"s{seed}"] = {
            "file": str(path.relative_to(REPO_ROOT)),
            "sha256": sha,
            "ssl_json_sha256": file_sha256(ssl_json_path(seed)),
        }

    out = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "n_files": len(hashes),
        "combined_pred_sha256": combined,
        "records": manifest["records"],
        "checkpoints": checkpoints,
        "ssl_checkpoints": ssl_checkpoints,
        "frozen_utc": now_utc(),
        "labels_read_after_this_point": True,
        "monitor_labels_read": False,
        "locked_labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "prediction_manifest_T.json", {**out, "complete": True})
    log(f"[freeze-predictions] {len(hashes)} files combined={combined[:16]}")
    return 0


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def _load_preds(zone: str, arm: str) -> dict[int, np.ndarray]:
    return {seed: np.asarray(np.load(pred_path(zone, arm, seed))["pred"], dtype=np.float64) for seed in SEEDS}


def _paired_bootstrap(delta: np.ndarray, *, n: int = ps.BOOTSTRAP_N, seed: int = ps.BOOTSTRAP_SEED) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    k = len(delta)
    means = np.empty(int(n), dtype=np.float64)
    for i in range(int(n)):
        idx = rng.integers(0, k, k)
        means[i] = delta[idx].mean()
    return {
        "mean": float(delta.mean()),
        "ci_lo": float(np.quantile(means, 0.025)),
        "ci_hi": float(np.quantile(means, 0.975)),
        "frac_better": float((delta < 0).mean()),
        "n_resamples": int(n),
        "seed": int(seed),
        "unit": "molecule",
    }


def _arm_stats(y: np.ndarray, preds: Mapping[int, np.ndarray]) -> dict[str, Any]:
    err = {s: np.abs(preds[s] - y) for s in SEEDS}
    seed_mean = np.mean([err[s] for s in SEEDS], axis=0)
    per_seed_mae = {str(s): float(err[s].mean()) for s in SEEDS}
    vals = np.asarray(list(per_seed_mae.values()), dtype=np.float64)
    ordinal = y > ps.TAIL_THRESHOLD
    return {
        "mae_seed_mean": float(np.mean(vals)),
        "mae_per_seed": per_seed_mae,
        "seed_spread_max_minus_min": float(vals.max() - vals.min()),
        "seed_spread_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "rmse_seed_mean": float(np.sqrt(np.mean(np.mean([(preds[s] - y) ** 2 for s in SEEDS], axis=0)))),
        "median_ae": float(np.median(seed_mean)),
        "p90_ae": float(np.quantile(seed_mean, 0.90)),
        "p95_ae": float(np.quantile(seed_mean, 0.95)),
        "p99_ae": float(np.quantile(seed_mean, 0.99)),
        "max_ae": float(seed_mean.max()),
        "ordinary_mae": float(seed_mean[ordinal].mean()),
        "tail_mae": float(seed_mean[~ordinal].mean()),
        "tail_n": int((~ordinal).sum()),
    }


def _contrast(y: np.ndarray, new: Mapping[int, np.ndarray], ref: Mapping[int, np.ndarray]) -> dict[str, Any]:
    e_new = {s: np.abs(new[s] - y) for s in SEEDS}
    e_ref = {s: np.abs(ref[s] - y) for s in SEEDS}
    delta = np.mean([e_new[s] - e_ref[s] for s in SEEDS], axis=0)
    per_seed = {str(s): float((e_new[s] - e_ref[s]).mean()) for s in SEEDS}
    ordinal = y > ps.TAIL_THRESHOLD
    ordinary_delta = float(delta[ordinal].mean()) if ordinal.any() else 0.0
    tail_delta = float(delta[~ordinal].mean()) if (~ordinal).any() else 0.0
    # Frozen top-10 definition: the ten molecules with the largest baseline-A
    # seed-mean absolute error are removed; the same rows are removed for both
    # arms before recomputing the paired delta.
    ref_seed_mean = np.mean([e_ref[s] for s in SEEDS], axis=0)
    worst = np.argsort(-ref_seed_mean)[: ps.TOP_K_REMOVED]
    keep = np.ones(len(delta), dtype=bool)
    keep[worst] = False
    boot = _paired_bootstrap(delta)
    return {
        "new_arm": "P3SAM",
        "ref_arm": "P3",
        "mae_new_seed_mean": float(np.mean([e_new[s].mean() for s in SEEDS])),
        "mae_ref_seed_mean": float(np.mean([e_ref[s].mean() for s in SEEDS])),
        "delta_seed_mean": float(np.mean(list(per_seed.values()))),
        "per_seed_delta": per_seed,
        "bootstrap": boot,
        "ordinary_delta": ordinary_delta,
        "tail_delta": tail_delta,
        "tail_n": int((~ordinal).sum()),
        "top10_removed_delta": float(delta[keep].mean()),
        "top10_removed_rows": [int(i) for i in worst],
        "frac_molecules_better": float((delta < 0).mean()),
        "rmse_new": float(np.sqrt(np.mean(np.mean([(new[s] - y) ** 2 for s in SEEDS], axis=0)))),
        "rmse_ref": float(np.sqrt(np.mean(np.mean([(ref[s] - y) ** 2 for s in SEEDS], axis=0)))),
    }


def _gate(contrast: Mapping[str, Any]) -> dict[str, Any]:
    conditions = {
        "seed_mean_delta_le_minus_0.004": bool(contrast["delta_seed_mean"] <= -ps.MEANINGFUL_DELTA),
        "both_seeds_negative": bool(all(float(v) < 0.0 for v in contrast["per_seed_delta"].values())),
        "paired_ci_upper_negative": bool(float(contrast["bootstrap"]["ci_hi"]) < 0.0),
        "ordinary_delta_negative": bool(float(contrast["ordinary_delta"]) < 0.0),
        "top10_removed_delta_negative": bool(float(contrast["top10_removed_delta"]) < 0.0),
        "rmse_delta_within_tol": bool(float(contrast["rmse_new"]) - float(contrast["rmse_ref"]) <= ps.RMSE_TOL),
        "tail_delta_within_tol": bool(float(contrast["tail_delta"]) <= ps.TAIL_TOL),
    }
    return {
        "conditions": conditions,
        "passed": bool(all(conditions.values())),
        "thresholds": {
            "meaningful_delta": ps.MEANINGFUL_DELTA,
            "rmse_tol": ps.RMSE_TOL,
            "tail_tol": ps.TAIL_TOL,
            "tail_threshold": ps.TAIL_THRESHOLD,
            "top_k_removed": ps.TOP_K_REMOVED,
        },
    }


def _score_zone(zone: str) -> dict[str, Any]:
    split = load_split()
    y = bp.load_y_rows(ENCODED_TRAIN, [int(i) for i in split[f"{zone}_rows"]])
    preds = {arm: _load_preds(zone, arm) for arm in ARMS}
    stats = {arm: _arm_stats(y, preds[arm]) for arm in ARMS}
    contrast = _contrast(y, preds["P3SAM"], preds["P3"])
    gate = _gate(contrast)
    return {
        "zone": zone,
        "n": int(len(y)),
        "y_sha256": array_sha256(y),
        "rows_sha256": split[f"{zone}_sha256_rows"],
        "arm_stats": stats,
        "contrast": contrast,
        "gate": gate,
        "seed_spread_delta": {
            "P3": stats["P3"]["seed_spread_max_minus_min"],
            "P3SAM": stats["P3SAM"]["seed_spread_max_minus_min"],
            "delta": float(stats["P3SAM"]["seed_spread_max_minus_min"] - stats["P3"]["seed_spread_max_minus_min"]),
        },
    }


def cmd_score_monitor(_args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    require(not monitor_lock_path().exists(), "Monitor labels were already read in this round")
    require(not locked_lock_path().exists(), "Locked labels must not be read before a Monitor pass")
    manifest = read_json(OUT / "prediction_manifest_T.json")
    require(bool(manifest.get("complete")), "prediction manifest is not complete")
    for name, rec in manifest["records"].items():
        require(file_sha256(OUT / "predictions" / name) == rec["file_sha256"], f"prediction drifted: {name}")

    result = _score_zone("monitor")
    write_json(OUT / "monitor_score_T.json", result)
    write_json(OUT / "monitor_gate_T.json", result["gate"])
    split = load_split()
    write_json(
        monitor_lock_path(),
        {
            "protocol": PROTOCOL_ID,
            "read_utc": now_utc(),
            "zone": "monitor",
            "n": int(result["n"]),
            "y_sha256": result["y_sha256"],
            "rows_sha256": split["monitor_sha256_rows"],
            "prediction_manifest_sha256": manifest["combined_pred_sha256"],
            "locked_labels_read": False,
            "single_read_monitor_only": True,
        },
    )
    c = result["contrast"]
    g = result["gate"]
    log(
        f"[score-monitor] P3={result['arm_stats']['P3']['mae_seed_mean']:.6f} "
        f"P3SAM={result['arm_stats']['P3SAM']['mae_seed_mean']:.6f} "
        f"delta={c['delta_seed_mean']:+.6f} ci=[{c['bootstrap']['ci_lo']:+.6f},{c['bootstrap']['ci_hi']:+.6f}] "
        f"gate={'PASS' if g['passed'] else 'FAIL'}"
    )
    if not g["passed"]:
        write_json(
            OUT / "decision.json",
            {
                "protocol": PROTOCOL_ID,
                "git_commit": git_commit(),
                "status": "SAM_NO_USEFUL_GAIN",
                "stage": "monitor",
                "monitor_gate_passed": False,
                "locked_evaluated": False,
                "locked_labels_read": False,
                "monitor_delta": c["delta_seed_mean"],
                "monitor_ci": [c["bootstrap"]["ci_lo"], c["bootstrap"]["ci_hi"]],
                "monitor_conditions": g["conditions"],
                "full10k_authorized": False,
                "official_test_authorized": False,
                "frozen_utc": now_utc(),
            },
        )
        log("[score-monitor] FAIL -> stop; Locked labels remain unread")
    else:
        write_json(
            OUT / "decision.json",
            {
                "protocol": PROTOCOL_ID,
                "git_commit": git_commit(),
                "status": "MONITOR_PASSED_LOCKED_PENDING",
                "stage": "monitor",
                "monitor_gate_passed": True,
                "locked_evaluated": False,
                "locked_labels_read": False,
                "monitor_delta": c["delta_seed_mean"],
                "monitor_ci": [c["bootstrap"]["ci_lo"], c["bootstrap"]["ci_hi"]],
                "monitor_conditions": g["conditions"],
                "full10k_authorized": False,
                "official_test_authorized": False,
                "frozen_utc": now_utc(),
            },
        )
        log("[score-monitor] PASS -> Locked evaluation is now authorized")
    return 0


def cmd_score_locked(_args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    require(monitor_lock_path().exists(), "Monitor must be scored first")
    require(not locked_lock_path().exists(), "Locked labels were already read in this round")
    monitor_gate = read_json(OUT / "monitor_gate_T.json")
    require(bool(monitor_gate["passed"]), "Monitor gate failed; Locked labels must remain unread")
    manifest = read_json(OUT / "prediction_manifest_T.json")
    require(bool(manifest.get("complete")), "prediction manifest is not complete")
    for name, rec in manifest["records"].items():
        require(file_sha256(OUT / "predictions" / name) == rec["file_sha256"], f"prediction drifted: {name}")

    result = _score_zone("locked")
    write_json(OUT / "locked_score_T.json", result)
    write_json(OUT / "locked_gate_T.json", result["gate"])
    split = load_split()
    write_json(
        locked_lock_path(),
        {
            "protocol": PROTOCOL_ID,
            "read_utc": now_utc(),
            "zone": "locked",
            "n": int(result["n"]),
            "y_sha256": result["y_sha256"],
            "rows_sha256": split["locked_sha256_rows"],
            "prediction_manifest_sha256": manifest["combined_pred_sha256"],
            "monitor_gate_passed": True,
            "single_read_after_monitor_pass": True,
        },
    )
    c = result["contrast"]
    g = result["gate"]
    log(
        f"[score-locked] P3={result['arm_stats']['P3']['mae_seed_mean']:.6f} "
        f"P3SAM={result['arm_stats']['P3SAM']['mae_seed_mean']:.6f} "
        f"delta={c['delta_seed_mean']:+.6f} ci=[{c['bootstrap']['ci_lo']:+.6f},{c['bootstrap']['ci_hi']:+.6f}] "
        f"gate={'PASS' if g['passed'] else 'FAIL'}"
    )
    status = "GENERALIZATION_OPTIMIZATION_PROMISING" if g["passed"] else "SAM_NO_USEFUL_GAIN"
    write_json(
        OUT / "decision.json",
        {
            "protocol": PROTOCOL_ID,
            "git_commit": git_commit(),
            "status": status,
            "stage": "locked",
            "monitor_gate_passed": True,
            "locked_gate_passed": bool(g["passed"]),
            "locked_evaluated": True,
            "monitor_delta": read_json(OUT / "monitor_score_T.json")["contrast"]["delta_seed_mean"],
            "monitor_ci": [
                read_json(OUT / "monitor_score_T.json")["contrast"]["bootstrap"]["ci_lo"],
                read_json(OUT / "monitor_score_T.json")["contrast"]["bootstrap"]["ci_hi"],
            ],
            "locked_delta": c["delta_seed_mean"],
            "locked_ci": [c["bootstrap"]["ci_lo"], c["bootstrap"]["ci_hi"]],
            "locked_conditions": g["conditions"],
            "full10k_proposal_allowed": bool(g["passed"]),
            "full10k_authorized": False,
            "official_test_authorized": False,
            "note": (
                "A single full10k confirmation may be proposed, not automatically executed."
                if g["passed"]
                else "Locked did not preserve the Monitor improvement; stop the SAM route."
            ),
            "frozen_utc": now_utc(),
        },
    )
    log(f"[score-locked] {status}")
    return 0


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------


def cmd_finalize(_args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    require((OUT / "monitor_score_T.json").exists(), "score Monitor first")
    tracked = []
    for path in sorted(OUT.rglob("*")):
        if path.is_dir() or path.suffix in (".pt", ".lock", ".tmp"):
            continue
        tracked.append({"path": str(path.relative_to(REPO_ROOT)), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    for path in sorted((OUT / "predictions").glob("*.npz")):
        tracked.append({"path": str(path.relative_to(REPO_ROOT)), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    ledger = read_json(ledger_path()) if ledger_path().exists() else {"entries": [], "total_wall_hours": 0.0}
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "frozen_utc": now_utc(),
        "n_artifacts": len(tracked),
        "artifacts": tracked,
        "gpu_budget": {
            "total_wall_hours": float(ledger.get("total_wall_hours", 0.0)),
            "cap_hours": GPU_BUDGET_CAP_HOURS,
            "within_cap": bool(float(ledger.get("total_wall_hours", 0.0)) <= GPU_BUDGET_CAP_HOURS),
            "entries": ledger.get("entries", []),
        },
        "checkpoints": {
            f"{arm}_s{seed}": {
                "file": str(soup_path(arm, seed).relative_to(REPO_ROOT)),
                "sha256": file_sha256(soup_path(arm, seed)),
            }
            for arm in ARMS
            for seed in SEEDS
        },
        "ssl_checkpoints": {
            f"s{seed}": {"file": str(ssl_path(seed).relative_to(REPO_ROOT)), "sha256": file_sha256(ssl_path(seed))}
            for seed in SEEDS
        },
        "prediction_manifest_sha256": read_json(OUT / "prediction_manifest_T.json")["combined_pred_sha256"],
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "evidence_manifest.json", payload)
    log(f"[finalize] {len(tracked)} artifacts hashed; GPU {payload['gpu_budget']['total_wall_hours']:.3f} h")
    return 0


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split")
    sub.add_parser("verify-inputs")
    sub.add_parser("objects")
    sub.add_parser("cpu-gates")
    p_freeze = sub.add_parser("freeze")
    p_freeze.add_argument("--attest-no-gpu-before-freeze", action="store_true", default=False)
    p_freeze.add_argument("--amendment", type=int, default=0)
    p_freeze.add_argument("--amendment-reason", default=None)
    sub.add_parser("stage-inputs")
    p_smoke = sub.add_parser("smoke")
    p_smoke.add_argument("--device", default="cuda:0")
    p_ssl = sub.add_parser("ssl")
    p_ssl.add_argument("--seed", type=int, required=True)
    p_ssl.add_argument("--device", default="cuda:0")
    p_ssl.add_argument("--smoke-updates", type=int, default=None)
    p_train = sub.add_parser("train")
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--device", default="cuda:0")
    p_train.add_argument("--retrain", action="store_true")
    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--seed", type=int, required=True)
    p_pred.add_argument("--device", default="cuda:0")
    sub.add_parser("freeze-predictions")
    sub.add_parser("score-monitor")
    sub.add_parser("score-locked")
    sub.add_parser("finalize")
    args = parser.parse_args(argv)

    table = {
        "split": cmd_split,
        "verify-inputs": cmd_verify_inputs,
        "objects": cmd_objects,
        "cpu-gates": cmd_cpu_gates,
        "freeze": cmd_freeze,
        "stage-inputs": cmd_stage_inputs,
        "smoke": cmd_smoke,
        "ssl": cmd_ssl,
        "train": cmd_train,
        "predict": cmd_predict,
        "freeze-predictions": cmd_freeze_predictions,
        "score-monitor": cmd_score_monitor,
        "score-locked": cmd_score_locked,
        "finalize": cmd_finalize,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    return int(table[args.cmd](args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
