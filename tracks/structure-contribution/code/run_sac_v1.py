#!/usr/bin/env python
"""Driver for protocol ``cscl-bjoc-sac-v1`` (BJOC-SAC).

Stages
------
``split``               (local CPU, RDKit; label-blind) rebuild the four
                        SMILES/PyG mismatch rows, derive identity from the PyG
                        graph, build the new identity-grouped split S with an
                        **explicit** salt; assert the salt that produced the
                        saved rows, reproduce the historical E/F splits from
                        their salts, and prove the new split differs.
``verify-inputs``       reuse hashes of the frozen upstream modules, split
                        sanity, parameter/recipe constants, no official
                        valid/test loader (label-blind).
``objects``             LabelGuard build of Fit/Monitor/Locked BJOC objects and
                        the Fit-only global standardiser.
``relations``           R1/R2/R3 + dense ``e<f`` pair lists for every zone,
                        plus a label-blind relation audit.
``cpu-gates``           the twelve pre-registered gates of PROTOCOL.md §6.
``freeze``              ``protocol_freeze.json`` (code + config + inputs +
                        budget).
``stage-inputs``        rebuild the machine-local, gitignored object/relation
                        artifacts on a compute checkout and verify them against
                        the frozen split / standardiser / pair totals.
``smoke``               post-freeze 20-update SSL + 1-epoch mini run per arm,
                        with measured throughput for the budget projection.
``ssl``                 Fit-only masked atom/bond SSL for one seed (no labels).
``train``               P3 / SACR / SACU for one seed (Fit labels only).
``predict``             own-arm predictions (Monitor + Locked) and the frozen
                        label-free diagnostics: content-null swap and T-zeroing.
``freeze-predictions``  hash-freeze every prediction file.
``score-holdout``       single read of Monitor + Locked labels, paired scoring,
                        frozen A/B/C/E/D decision.
``post-hoc``            frozen mechanism diagnostics (no new label read).
``finalize``            evidence manifest + budget summary.
"""

from __future__ import annotations

import argparse
import csv
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
import sac_v1 as sac  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

REPO_ROOT = _CODE_DIR.parents[2]
TRACK = _CODE_DIR.parent
OUT = Path(os.environ.get("SAC_OUT_DIR", str(TRACK / "results" / "sac_v1")))
CONFIG_PATH = TRACK / "configs" / "sac_v1.yaml"
PROTOCOL_DOC = OUT / "PROTOCOL.md"
ZINC_ROOT = REPO_ROOT / "data/ZINC"
DATA_TRAIN = ZINC_ROOT / "subset/processed/train.pt"
CANON = REPO_ROOT / "tracks/ksvd/results/zinc_cssd_basis_reuse_v1/train_canonical_smiles.npz"
ENCODED_TRAIN = REPO_ROOT / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt"
STRUCT_CONTEXT_CODE = TRACK / "results/structure_context_support_v1/code"

P3_RESULTS = TRACK / "results" / "p3_bond_relation_bridge_v1"
BRA_RESULTS = TRACK / "results" / "bond_relation_attribution_v1"

PROTOCOL_ID = sac.PROTOCOL_ID
SPLIT_TAG = sac.SPLIT_TAG
SEEDS = sac.SEEDS
ARMS = sac.ARMS
ZONES = sac.ZONES

FROZEN_INPUT_HASHES = {
    "train_pt": "e0f1837b4a19714fcf3a5c6727a30c23f57c3398c8987cab2983214a0e76955b",
    "canonical_smiles": "04a590355fb7bba71e35e52bed60c974999b92af9e96e4ee99bed4afe42dcaff",
    "encoded_train": "10d764955ef1b040a317105be1dd3fbb2e81292b03b79e7d078cce08e18e5f1d",
}
#: historical splits reproduced by the frozen split builder (D-1 regression test).
#: ``effective_salt`` is the salt that actually produced the rows; ``recorded_salt``
#: is what the historical metadata claims.  For split F the two differ (the P3
#: round's driver called the builder without a salt, so the builder default was
#: used while the metadata recorded the round's salt -- DEVIATIONS D-1 there).
HISTORICAL_SPLITS = {
    "E": {
        "recorded_salt": "cscl-bond-relation-attribution-v1",
        "effective_salt": "cscl-bond-relation-attribution-v1",
        "path": BRA_RESULTS / "split_E.json",
    },
    "F": {
        "recorded_salt": "cscl-p3-bond-relation-bridge-v1",
        "effective_salt": "cscl-bond-relation-attribution-v1",
        "path": P3_RESULTS / "split_F.json",
    },
}
GPU_BUDGET_CAP_HOURS = 4.0
RELATION_AUDIT_SAMPLE = 300
FORBIDDEN_OFFICIAL = ("valid", "test")

GATE_BATCH = 16
EQUIV_EPOCHS = 2
GRAD_STEPS = 1


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
    return {
        name: file_sha256(_CODE_DIR / name)
        for name in ("sac_v1.py", "run_sac_v1.py")
    }


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def split_path() -> Path:
    return OUT / f"split_{SPLIT_TAG}.json"


def objects_path(zone: str) -> Path:
    return OUT / f"objects_{zone}_{SPLIT_TAG}.pt"


def relations_path(zone: str) -> Path:
    return OUT / f"relations_{zone}_{SPLIT_TAG}.pt"


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


def swap_pred_path(zone: str, model_arm: str, mode: str, seed: int) -> Path:
    return OUT / "predictions" / f"swap_{zone}_m{model_arm}_k{mode}_{SPLIT_TAG}_s{seed}.npz"


def zero_pred_path(zone: str, arm: str, seed: int) -> Path:
    return OUT / "predictions" / f"zeroT_{zone}_{arm}_{SPLIT_TAG}_s{seed}.npz"


def smoke_dir() -> Path:
    return OUT / "smoke"


def ledger_path() -> Path:
    return OUT / "gpu_budget_ledger.json"


def freeze_path() -> Path:
    return OUT / "protocol_freeze.json"


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def load_split() -> Mapping[str, Any]:
    return read_json(split_path())


def load_zone_objects(zone: str) -> list[sro.MolObjects]:
    return torch.load(objects_path(zone), map_location="cpu", weights_only=False)


def load_zone_relations(zone: str) -> sac.ZoneRelations:
    return sac.ZoneRelations.from_state(torch.load(relations_path(zone), map_location="cpu", weights_only=False))


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
# freeze
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
# split (local, RDKit)
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

    structs = scs.load_pyg_structure_only(rows)  # label-blind: y is discarded
    require([int(s["row"]) for s in structs] == rows, "structure rows out of order")
    graph_identity = {int(s["row"]): str(scs._pyg_rebuilt_smiles(s)) for s in structs}
    identity_keys = [graph_identity[r] for r in rows]

    with np.load(CANON, allow_pickle=True) as blob:
        table = [str(s) for s in blob["smiles"]]
    table_identity = {r: Chem.CanonSmiles(table[r], useChiral=False) for r in rows}
    partition_equal = identity_partition(rows, graph_identity) == identity_partition(rows, table_identity)
    require(partition_equal, "graph identity partition != committed-SMILES partition")

    # ---- the salt actually used is explicit and asserted -------------------
    used_salt = sac.SPLIT_SALT
    split = bra.build_identity_split(rows, identity_keys, salt=used_salt)
    require(split["salt"] == used_salt, f"split builder used salt {split['salt']!r}, expected {used_salt!r}")
    require(split["duplicate_groups_cross_sides"] == 0, "identity group crosses split zones")
    sizes = (len(split["fit_rows"]), len(split["monitor_rows"]), len(split["locked_rows"]))
    require(abs(sizes[0] - 7996) <= 10, f"fit size off target: {sizes}")
    require(abs(sizes[1] - 1000) <= 10, f"monitor size off target: {sizes}")
    require(abs(sizes[2] - 1000) <= 10, f"locked size off target: {sizes}")
    all_rows = sorted(split["fit_rows"] + split["monitor_rows"] + split["locked_rows"])
    require(all_rows == rows, "split union does not equal the usable rows")

    # independent recomputation of the greedy order from the metadata salt
    order = sorted(
        {k: None for k in identity_keys},
        key=lambda k: hashlib.sha256(f"{split['salt']}|{k}".encode()).hexdigest(),
    )
    groups: dict[str, list[int]] = {}
    for r, k in zip(rows, identity_keys):
        groups.setdefault(k, []).append(int(r))
    refit: list[int] = []
    remon: list[int] = []
    relock: list[int] = []
    for k in order:
        members = sorted(groups[k])
        if len(refit) < 7996:
            refit.extend(members)
        elif len(remon) < 1000:
            remon.extend(members)
        else:
            relock.extend(members)
    recomputed = {"fit_rows": sorted(refit), "monitor_rows": sorted(remon), "locked_rows": sorted(relock)}
    salt_reproduced = all(recomputed[f"{z}_rows"] == sorted(split[f"{z}_rows"]) for z in ("fit", "monitor", "locked"))
    require(salt_reproduced, "recomputed split from the saved salt does not match the saved rows")

    # ---- D-1 regression: reproduce the historical splits from their salts --
    historical: dict[str, Any] = {}
    old_row_sets: dict[str, dict[str, Sequence[int]]] = {}
    for tag, spec in HISTORICAL_SPLITS.items():
        old_path = spec["path"]
        recorded_salt = spec["recorded_salt"]
        effective_salt = spec["effective_salt"]
        require(old_path.exists(), f"historical split {tag} missing at {old_path}")
        old = read_json(old_path)
        require(old["salt"] == recorded_salt, f"historical split {tag} records a different salt: {old['salt']}")
        rebuilt = bra.build_identity_split(rows, identity_keys, salt=effective_salt)
        match = all(
            sorted(rebuilt[f"{z}_rows"]) == sorted(int(i) for i in old[f"{z}_rows"])
            for z in ("fit", "monitor", "locked")
        )
        require(match, f"historical split {tag} is NOT reproduced by effective salt {effective_salt!r}")
        metadata_salt_reproduces = all(
            sorted(bra.build_identity_split(rows, identity_keys, salt=recorded_salt)[f"{z}_rows"])
            == sorted(int(i) for i in old[f"{z}_rows"])
            for z in ("fit", "monitor", "locked")
        )
        historical[tag] = {
            "path": str(old_path.relative_to(REPO_ROOT)),
            "recorded_salt": recorded_salt,
            "effective_salt": effective_salt,
            "metadata_salt_reproduces_rows": bool(metadata_salt_reproduces),
            "metadata_salt_bug_confirmed": bool(not metadata_salt_reproduces and effective_salt != recorded_salt),
            "reproduced_row_for_row": True,
            "fit_sha256_rows": rows_sha256([int(i) for i in old["fit_rows"]]),
            "monitor_sha256_rows": rows_sha256([int(i) for i in old["monitor_rows"]]),
            "locked_sha256_rows": rows_sha256([int(i) for i in old["locked_rows"]]),
        }
        old_row_sets[tag] = {z: [int(i) for i in old[f"{z}_rows"]] for z in ("fit", "monitor", "locked")}

    own_hashes = {z: rows_sha256(split[f"{z}_rows"]) for z in ("fit", "monitor", "locked")}
    collisions = {
        tag: sum(1 for z in ("fit", "monitor", "locked") if own_hashes[z] == historical[tag][f"{z}_sha256_rows"])
        for tag in historical
    }
    require(all(v == 0 for v in collisions.values()), f"new split collides with a historical zone hash: {collisions}")
    identical = {tag: all(sorted(split[f"{z}_rows"]) == sorted(old_row_sets[tag][z]) for z in ("fit", "monitor", "locked")) for tag in old_row_sets}
    require(not any(identical.values()), f"new split is row-identical to a historical split: {identical}")

    # ---- salt is load-bearing -------------------------------------------
    probe = bra.build_identity_split(rows, identity_keys, salt=used_salt + "-probe")
    require(rows_sha256(probe["fit_rows"]) != own_hashes["fit"], "salt probe did not change the split")

    # ---- overlap disclosure ----------------------------------------------
    new_sets = {z: [int(i) for i in split[f"{z}_rows"]] for z in ("fit", "monitor", "locked")}
    overlap = {tag: _zone_overlap(new_sets, old_row_sets[tag]) for tag in old_row_sets}
    any_fit_overlap = {
        tag: sum(1 for r in new_sets["fit"] if r in set(old_row_sets[tag]["fit"])) for tag in old_row_sets
    }
    any_seen = sorted({r for tag in old_row_sets for z in ("fit", "monitor", "locked") for r in old_row_sets[tag][z]})
    seen_set = set(any_seen)

    identity_digest = hashlib.sha256("\n".join(f"{r}\t{graph_identity[r]}" for r in rows).encode()).hexdigest()
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": SPLIT_TAG,
        "salt": split["salt"],
        "salt_source": "explicit argument sac_v1.SPLIT_SALT (never a builder default)",
        "salt_reproduced_from_saved_metadata": bool(salt_reproduced),
        "salt_is_load_bearing": True,
        "n_rows": len(rows),
        "n_groups": int(split["n_groups"]),
        "n_fit": len(split["fit_rows"]),
        "n_monitor": len(split["monitor_rows"]),
        "n_locked": len(split["locked_rows"]),
        "fit_rows": split["fit_rows"],
        "monitor_rows": split["monitor_rows"],
        "locked_rows": split["locked_rows"],
        "rows_sorted": rows,
        "group_ids_sorted": [int(split["group_id"][str(r)]) for r in rows],
        "fit_sha256_rows": own_hashes["fit"],
        "monitor_sha256_rows": own_hashes["monitor"],
        "locked_sha256_rows": own_hashes["locked"],
        "excluded_rows": list(bra.EXCLUDED_ROWS),
        "identity_source": "canonical constitution SMILES rebuilt from the official PyG graph",
        "identity_digest_sha256": identity_digest,
        "identity_partition_equal_to_committed_smiles": bool(partition_equal),
        "duplicate_groups": split["duplicate_groups"],
        "duplicate_groups_cross_sides": int(split["duplicate_groups_cross_sides"]),
        "historical_splits": historical,
        "historical_zone_hash_collisions": collisions,
        "row_identical_to_historical": identical,
        "overlap_with_historical": overlap,
        "new_fit_rows_seen_in_historical_fit": any_fit_overlap,
        "n_rows_seen_in_any_historical_zone": int(sum(1 for r in rows if r in seen_set)),
        "frac_rows_seen_in_any_historical_zone": float(sum(1 for r in rows if r in seen_set) / len(rows)),
        "label_information_used": False,
        "monitor_labels_read": False,
        "locked_labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(split_path(), payload)
    write_json(
        OUT / "split_compare_S.json",
        {
            "protocol": PROTOCOL_ID,
            "git_commit": git_commit(),
            "pairing_verification": pairing,
            "n_rows_official_train": 10_000,
            "n_rows_excluded": len(bra.EXCLUDED_ROWS),
            "n_rows_usable": len(rows),
            "sizes": dict(zip(("fit", "monitor", "locked"), sizes)),
            "new_zone_sha256_rows": own_hashes,
            "historical_splits": historical,
            "identical_to_historical": identical,
            "overlap": overlap,
            "disclosure": "the same official-train 10k pool has been used by several prior rounds; "
            "split S is a new partition, NOT a new blind test population",
            "labels_read": False,
        },
    )
    log(f"[split] n={len(rows)} sizes={sizes} salt={used_salt!r} historical-E/F reproduced, no hash collision")
    log(f"[split] overlap new-monitor vs old-F fit/locked: "
        f"{overlap['F']['monitor']['fit']['n_overlap']}/{overlap['F']['monitor']['locked']['n_overlap']}")
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
    require(tuple(cfg["seeds"]) == SEEDS and tuple(cfg["arms"]) == ARMS, "config arms/seeds mismatch")
    require(int(cfg["project_dim"]) == sac.PROJECT_DIM, "config project_dim mismatch")
    require(int(cfg["base_parameters"]) == sac.BASE_PARAMETERS, "config base parameter count mismatch")
    require(int(cfg["sac_parameters"]) == sac.EXPECTED_SAC_PARAMETERS, "config SAC parameter count mismatch")
    require(int(cfg["param_cap"]) == sac.PARAM_CAP, "config parameter cap mismatch")
    require(float(cfg["meaningful_delta_r_vs_p3"]) == sac.MEANINGFUL_DELTA_R_VS_P3, "config R-P3 threshold mismatch")
    require(float(cfg["meaningful_delta_r_vs_u"]) == sac.MEANINGFUL_DELTA_R_VS_U, "config R-U threshold mismatch")
    require(float(cfg["budget_gpu_hours"]) == GPU_BUDGET_CAP_HOURS, "config budget mismatch")
    recipe = bp.RECIPES[cfg["recipe"]]
    require(int(cfg["epochs"]) == int(recipe["epochs"]), "config epochs != frozen R3 recipe")
    require(float(cfg["lr"]) == float(recipe["lr"]), "config lr != frozen R3 recipe")
    require(float(cfg["lr_min"]) == float(recipe["lr_min"]), "config lr_min != frozen R3 recipe")
    require(float(cfg["weight_decay"]) == float(recipe["weight_decay"]), "config wd != frozen R3 recipe")
    require(int(cfg["batch_size"]) == int(bp.BATCH_SIZE), "config batch size != frozen recipe")
    require(int(cfg["soup_last"]) == int(bp.SOUP_LAST), "config soup != frozen recipe")
    require(int(cfg["shuffle_offset"]) == int(bp.TRAIN_SHUFFLE_OFFSET), "config shuffle offset != frozen recipe")
    require(int(cfg["ssl_updates"]) == int(mp.PRETRAIN_UPDATES), "config SSL updates != frozen SSL budget")

    split = load_split()
    require(split["protocol"] == PROTOCOL_ID, "split belongs to another protocol")
    require(split["salt"] == sac.SPLIT_SALT, "split salt mismatch")
    require(split["salt_reproduced_from_saved_metadata"] is True, "split salt was not reproduced")
    require(split["salt_is_load_bearing"] is True, "salt probe failed")
    require(all(int(v) == 0 for v in split["historical_zone_hash_collisions"].values()), "split collides with history")
    require(not any(split["row_identical_to_historical"].values()), "split is row-identical to a historical split")

    upstream = {
        "structural_relation_composer_v1.py": _CODE_DIR / "structural_relation_composer_v1.py",
        "structural_relation_objects_v1.py": _CODE_DIR / "structural_relation_objects_v1.py",
        "bjoc_perfmech_long_v1.py": _CODE_DIR / "bjoc_perfmech_long_v1.py",
        "bjoc_masked_pretraining_v1.py": _CODE_DIR / "bjoc_masked_pretraining_v1.py",
        "bjoc_arch_breakthrough_v1.py": _CODE_DIR / "bjoc_arch_breakthrough_v1.py",
        "bond_relation_attribution_v1.py": _CODE_DIR / "bond_relation_attribution_v1.py",
        "p3_bond_relation_bridge_v1.py": _CODE_DIR / "p3_bond_relation_bridge_v1.py",
    }
    reuse = {name: file_sha256(path) for name, path in upstream.items()}
    p3_freeze = read_json(P3_RESULTS / "protocol_freeze.json")
    expected = {name: rec["sha256"] for name, rec in p3_freeze["code"].items() if name in upstream}
    require(expected, "P3 freeze does not record upstream code hashes")
    for name, sha in expected.items():
        require(reuse[name] == sha, f"upstream module changed since the P3 round: {name}")

    # parameter audit on both arm families (deterministic CPU construction)
    norm = {"global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32), "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32)}
    p3 = sac.build_p3(0, norm, "cpu")
    sac_model = sac.build_sac(0, norm, "cpu")
    audit = {
        "p3_parameters": int(sum(p.numel() for p in p3.parameters())),
        "sac_parameters": int(sum(p.numel() for p in sac_model.parameters())),
        "sac_new_parameters": sac_model.new_parameters,
        "p3_reader_parameters": int(sum(p.numel() for p in p3.reader.parameters())),
        "sac_reader_parameters": int(sum(p.numel() for p in sac_model.reader.parameters())),
        "project_parameters": int(sum(p.numel() for p in sac_model.project.parameters())),
        "p3_reader_in": int(p3.config.reader_in),
        "sac_reader_in": int(sac_model.config.reader_in) + sac.PROJECT_OUT + sac.N_RELATIONS,
    }
    require(audit["p3_parameters"] == sac.BASE_PARAMETERS, "P3 parameter audit failed")
    require(audit["sac_parameters"] == sac.EXPECTED_SAC_PARAMETERS, "SAC parameter audit failed")
    require(audit["sac_new_parameters"] == 26_184, "SAC new-parameter count mismatch")
    require(audit["sac_parameters"] <= sac.PARAM_CAP, "SAC exceeds the parameter cap")

    # no official valid/test loader anywhere in this round's code
    # (tokens assembled at runtime so this scanner does not match its own literals)
    split_tokens = ["split=" + q + name + q for q in ('"', "'") for name in ("valid", "test")]
    path_tokens = ["processed/" + name for name in ("valid", "test")]
    scan_hits = []
    for path in (_CODE_DIR / "sac_v1.py", _CODE_DIR / "run_sac_v1.py"):
        text = path.read_text(encoding="utf-8")
        for token in split_tokens + path_tokens:
            if token in text:
                scan_hits.append(f"{path.name}:{token}")
    require(not scan_hits, f"official valid/test loader found: {scan_hits}")

    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "input_hashes": actual,
        "input_hashes_match_frozen": True,
        "upstream_code_hashes": reuse,
        "upstream_matches_p3_round": True,
        "parameter_audit": audit,
        "recipe": dict(recipe),
        "official_loader_scan_hits": scan_hits,
        "label_guard_selfcheck": ab.label_guard_selfcheck(),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "input_audit.json", payload)
    log(f"[verify-inputs] P3={audit['p3_parameters']} SAC={audit['sac_parameters']} (+{audit['sac_new_parameters']})")
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
        "zones": {z: {"n": len(objs[z]), "rows_sha256": rows_sha256([o.mol_index for o in objs[z]])} for z in objs},
        "prep_fit_sha256": file_sha256(prep_path()),
        "guarded_rows": 2000,
        "forbidden_label_access_attempts": len(guard.access_attempts),
        "monitor_labels_read": False,
        "locked_labels_read": False,
    }
    write_json(OUT / "objects_manifest_S.json", payload)
    log(f"[objects] built in {payload['build_seconds']:.1f}s; standardiser fit-only")
    return 0


# ---------------------------------------------------------------------------
# relations
# ---------------------------------------------------------------------------


def cmd_relations(_args: argparse.Namespace) -> int:
    require(not freeze_path().exists(), "relations are frozen already; refusing to rebuild")
    split = load_split()
    started = time.perf_counter()
    payload: dict[str, Any] = {"protocol": PROTOCOL_ID, "git_commit": git_commit(), "zones": {}}
    for zone in ("fit", "monitor", "locked"):
        objs = load_zone_objects(zone)
        require([o.mol_index for o in objs] == [int(i) for i in split[f"{zone}_rows"]], f"{zone}: object rows changed")
        mols = [sac.build_mol_relations(o) for o in objs]
        zr = sac.build_zone_relations(mols)
        torch.save(zr.to_state(), relations_path(zone))
        m = zr.m
        totals = zr.totals()
        n_pairs = int(sum(sac.choose2(int(n)) for n in zr.n_bonds))
        payload["zones"][zone] = {
            "n_molecules": len(zr),
            "n_bonds_total": int(zr.n_bonds.sum()),
            "n_complete_pairs": zr.n_complete(),
            "n_pairs_total": n_pairs,
            "n_pairs": totals,
            "per_relation": {
                name: {
                    "total": totals[name],
                    "mean_per_molecule": float(m[:, s].mean()),
                    "max_per_molecule": int(m[:, s].max()),
                    "frac_molecules_zero": float((m[:, s] == 0).mean()),
                    "frac_of_complete_pairs": float(totals[name] / max(n_pairs, 1)),
                }
                for s, name in enumerate(sac.REL_NAMES)
            },
            "coverage_all_relations": float(sum(totals.values()) / max(n_pairs, 1)),
            "relations_file_sha256": file_sha256(relations_path(zone)),
        }
        log(
            f"[relations {zone}] m1={m[:,0].mean():.1f} m2={m[:,1].mean():.1f} m3={m[:,2].mean():.1f} "
            f"complete={zr.n_complete()} coverage={payload['zones'][zone]['coverage_all_relations']:.3f}"
        )

    # label-blind audit: all molecules (summary) + a deterministic sample (detail)
    audit_path = OUT / "relation_audit.json"
    zone_records: dict[str, Any] = {}
    for zone in ("fit", "monitor", "locked"):
        objs = load_zone_objects(zone)
        records = [sac.audit_molecule(o) for o in objs]
        summary = sac.summarize_audit(records)
        step = max(len(records) // RELATION_AUDIT_SAMPLE, 1)
        sample = [records[i] for i in range(0, len(records), step)][:RELATION_AUDIT_SAMPLE]
        zone_records[zone] = {"summary": summary, "sample_records": sample}
        require(summary["all_disjoint"], f"{zone}: relation sets overlap")
    write_json(
        audit_path,
        {
            "protocol": PROTOCOL_ID,
            "git_commit": git_commit(),
            "definition": {
                "R1": "dist_line == 1",
                "R2": "dist_line == 2",
                "R3": "same ring-bearing biconnected block and dist_line >= 3",
            },
            "zones": zone_records,
            "build_seconds": time.perf_counter() - started,
            "labels_read": False,
        },
    )
    payload["relation_audit_sha256"] = file_sha256(audit_path)
    payload["build_seconds"] = time.perf_counter() - started
    payload["label_information_used"] = False
    write_json(OUT / "relations_manifest_S.json", payload)
    log(f"[relations] audit written in {payload['build_seconds']:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# CPU gates
# ---------------------------------------------------------------------------


def _mol(idx: int, atom_types: Sequence[int], bonds: Sequence[tuple[int, int, int]]) -> sro.MolObjects:
    n = len(atom_types)
    x = np.asarray(atom_types, dtype=np.int64).reshape(n, 1)
    edge_index, edge_attr = [], []
    for u, v, bt in bonds:
        edge_index.extend([(u, v), (v, u)])
        edge_attr.extend([bt, bt])
    return sro.build_mol_objects(
        idx, x, np.asarray(edge_index, dtype=np.int64).T.reshape(2, -1), np.asarray(edge_attr, dtype=np.int64)
    )


def chain(n_atoms: int, bt: int = 1) -> sro.MolObjects:
    return _mol(0, range(n_atoms), [(i, i + 1, bt) for i in range(n_atoms - 1)])


def ring(n_atoms: int, alternating: bool = False) -> sro.MolObjects:
    bonds = [(i, (i + 1) % n_atoms, (2 if (alternating and i % 2 == 0) else 1)) for i in range(n_atoms)]
    return _mol(0, range(n_atoms), bonds)


def spiro() -> sro.MolObjects:
    # two 4-rings sharing atom 0
    return _mol(0, range(7), [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1), (0, 4, 1), (4, 5, 1), (5, 6, 1), (6, 0, 1)])


def fused() -> sro.MolObjects:
    # naphthalene skeleton: two 6-rings sharing the bond (4,5)
    bonds = [(0, 1, 2), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 5, 2), (5, 0, 1), (4, 9, 1), (9, 8, 2), (8, 7, 1), (7, 6, 2), (6, 5, 1)]
    return _mol(0, range(10), bonds)


def disconnected() -> sro.MolObjects:
    return _mol(0, range(8), [(0, 1, 1), (1, 2, 1), (2, 3, 1), (4, 5, 1), (5, 6, 1), (6, 7, 1)])


def _batch(objs: Sequence[sro.MolObjects], device: torch.device) -> dict[str, Any]:
    return sac.move_batch(sro.batch_objects(objs), device)


def _gate(name: str, ok: Any, **detail: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), **detail}


def _zone_rel(objs: Sequence[sro.MolObjects]) -> sac.ZoneRelations:
    return sac.build_zone_relations([sac.build_mol_relations(o) for o in objs])


def _relabel(o: sro.MolObjects, perm: np.ndarray) -> sro.MolObjects:
    """Rebuild a molecule under an atom relabelling (y is not represented here)."""
    bi = np.asarray(o.bond_index)
    x = np.zeros((o.n_atoms,), dtype=np.int64)
    # recover the atom one-hot from the feature block
    block = sro.atom_blocks()["atom_type"]
    x[:] = np.argmax(np.asarray(o.atom_feat)[:, block], axis=1)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    new_x = x[perm].reshape(-1, 1)
    bonds = [(int(inv[u]), int(inv[v]), int(t)) for (u, v), t in zip(bi, np.asarray(o.bond_type))]
    return _mol(o.mol_index, new_x.reshape(-1).tolist(), bonds)


def _relation_automorphism(perm: np.ndarray, zr: sac.ZoneRelations, mol: int = 0) -> bool:
    """True iff ``perm`` preserves every relation pair set (i.e. all three)."""
    pairs = []
    for r in range(sac.N_RELATIONS):
        a = zr.src[r][zr.starts[mol, r] : zr.ends[mol, r]]
        b = zr.dst[r][zr.starts[mol, r] : zr.ends[mol, r]]
        pairs.append(set(zip(a.tolist(), b.tolist())))
    for r in range(sac.N_RELATIONS):
        mapped = {tuple(sorted((int(perm[i]), int(perm[j])))) for i, j in pairs[r]}
        if mapped != pairs[r]:
            return False
    return True


def cmd_cpu_gates(_args: argparse.Namespace) -> int:
    gates: list[dict[str, Any]] = []
    device = torch.device("cpu")
    norm = {"global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32), "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32)}
    torch.manual_seed(1234)

    # ---- G1 physical bond dedup on real ZINC rows ------------------------
    split = load_split()
    ds = load_official_train()
    sample_rows = [int(i) for i in split["fit_rows"][:400]] + [int(i) for i in split["locked_rows"][:100]]
    dedup_ok = True
    dup_offenders = []
    for r in sample_rows:
        d = ds[r]
        ei = np.asarray(d.edge_index)
        pairs = set()
        for k in range(ei.shape[1]):
            u, v = int(ei[0, k]), int(ei[1, k])
            pairs.add((min(u, v), max(u, v)))
        if ei.shape[1] != 2 * len(pairs):
            dedup_ok = False
            dup_offenders.append(r)
        o = sro.build_mol_objects(r, d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy())
        bi = np.asarray(o.bond_index)
        keys = {(int(u), int(v)) for u, v in bi}
        if len(keys) != bi.shape[0] or bi.shape[0] != len(pairs):
            dedup_ok = False
            dup_offenders.append(r)
    gates.append(
        _gate(
            "G1_physical_bonds_deduplicated",
            dedup_ok,
            n_rows=len(sample_rows),
            offenders=sorted(set(dup_offenders))[:10],
            note="PyG stores every bond twice; one physical bond object per undirected pair",
        )
    )

    # ---- G2 relation sets pairwise disjoint (real + synthetic) -----------
    real_objs = [load_zone_objects("fit")[i] for i in range(0, 400, 7)]
    disjoint_ok = True
    worst = 0
    cover_records = []
    for o in real_objs + [chain(9), ring(6), ring(8, True), spiro(), fused(), disconnected()]:
        rec = sac.audit_molecule(o)
        disjoint_ok = disjoint_ok and rec["disjoint"]
        worst = max(worst, rec["max_pair_multiplicity"])
        cover_records.append(rec)
    gates.append(
        _gate(
            "G2_relations_pairwise_disjoint",
            disjoint_ok and worst <= 1,
            n_molecules=len(cover_records),
            max_pair_multiplicity=int(worst),
            mean_coverage=float(np.mean([r["relation_coverage"] for r in cover_records])),
        )
    )

    # ---- G3 ring-block edge cases ---------------------------------------
    cases: dict[str, Any] = {}
    ok3 = True

    c = chain(7)
    m = sac.relation_counts(c)
    n_c = int(c.bond_index.shape[0])
    cases["chain"] = {"n_bonds": n_c, "m": m.tolist()}
    ok3 &= (int(m[0]) == n_c - 1) and int(m[1]) == max(n_c - 2, 0) and int(m[2]) == 0

    r6 = ring(6)
    m6 = sac.relation_counts(r6)
    block6, nb6, nbr6 = sac.bond_ring_blocks(r6)
    cases["ring6"] = {"n_bonds": 6, "m": m6.tolist(), "n_ring_blocks": nb6, "n_bridge_blocks": nbr6,
                      "all_bonds_ring": bool((block6 >= 0).all())}
    ok3 &= (m6.tolist() == [6, 6, 3]) and nb6 == 1 and nbr6 == 0

    sp = spiro()
    block_sp, nb_sp, nbr_sp = sac.bond_ring_blocks(sp)
    pairs_sp = sac.relation_pair_lists(sp)
    cross = [
        (int(a), int(b))
        for a, b in zip(pairs_sp["R3"]["src"], pairs_sp["R3"]["dst"])
        if block_sp[int(a)] != block_sp[int(b)]
    ]
    # cross-ring pairs at line distance >= 3 exist, and none of them is in R3
    all_pairs = sac.complete_pair_list(int(sp.bond_index.shape[0]))
    dist_sp = sac.line_distances(sac.pb.line_adjacency(sp))
    cross_far = sum(
        1
        for a, b in zip(all_pairs["src"], all_pairs["dst"])
        if block_sp[int(a)] != block_sp[int(b)] and dist_sp[int(a), int(b)] >= 3
    )
    cases["spiro"] = {"n_ring_blocks": nb_sp, "n_bridge_blocks": nbr_sp,
                      "cross_block_far_pairs": int(cross_far), "cross_block_R3_pairs": len(cross)}
    ok3 &= nb_sp == 2 and len(cross) == 0 and cross_far > 0

    fu = fused()
    block_fu, nb_fu, nbr_fu = sac.bond_ring_blocks(fu)
    m_fu = sac.relation_counts(fu)
    cases["fused"] = {"n_ring_blocks": nb_fu, "n_bridge_blocks": nbr_fu, "m": m_fu.tolist(),
                      "all_ring_in_one_block": bool(len(set(block_fu[block_fu >= 0].tolist())) == 1)}
    ok3 &= nb_fu == 1 and int(m_fu[2]) > 0 and int(nbr_fu) == 0

    di = disconnected()
    pairs_di = sac.relation_pair_lists(di)
    dist_di = sac.line_distances(sac.pb.line_adjacency(di))
    unreachable = int((dist_di >= sac.INF_DIST).sum() // 2)
    r3_pair_count = len(pairs_di["R3"]["src"])
    cases["disconnected"] = {"unreachable_pairs": unreachable, "R3_pairs": r3_pair_count}
    ok3 &= unreachable > 0 and r3_pair_count == 0

    one = _mol(0, [0, 1], [(0, 1, 1)])
    two = chain(3)
    cases["single_bond"] = {"m": sac.relation_counts(one).tolist()}
    cases["two_bonds"] = {"m": sac.relation_counts(two).tolist()}
    ok3 &= sac.relation_counts(one).tolist() == [0, 0, 0] and sac.relation_counts(two).tolist() == [1, 0, 0]
    gates.append(_gate("G3_ring_block_semantics", ok3, cases=cases))

    # ---- G4 permutation / relabelling / batch invariance -----------------
    p3 = sac.build_p3(0, norm, device)
    smodel = sac.build_sac(0, norm, device)
    mixed = ring(8, alternating=True)
    n_at = mixed.n_atoms
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_at)
    mixed2 = _relabel(mixed, perm)
    objs_a = [chain(5), mixed, fused()]
    objs_b = [fused(), chain(5), mixed]
    rel_a = _zone_rel(objs_a)
    pa = sac.predict_model(smodel, objs_a, rel_a, "R", device)
    pb_ = sac.predict_model(smodel, objs_b, _zone_rel(objs_b), "R", device)
    pc = sac.predict_model(smodel, [mixed2], _zone_rel([mixed2]), "R", device)
    pmixed = sac.predict_model(smodel, [mixed], _zone_rel([mixed]), "R", device)
    batch_perm_err = float(np.abs(pa[[2, 0, 1]] - pb_).max())
    relabel_err = float(abs(pc[0] - pmixed[0]))
    p3_relabel_err = float(
        abs(sac.predict_model(p3, [mixed2], None, None, device)[0] - sac.predict_model(p3, [mixed], None, None, device)[0])
    )
    gates.append(
        _gate(
            "G4_permutation_invariance",
            max(batch_perm_err, relabel_err, p3_relabel_err) <= 1e-5,
            batch_permutation_max_abs=batch_perm_err,
            atom_relabel_SAC_max_abs=relabel_err,
            atom_relabel_P3_max_abs=p3_relabel_err,
            bond_endpoint_swap="canonical u<v enforced by unique_undirected_bonds",
        )
    )

    # ---- G5 matched initialisation --------------------------------------
    p3_a = sac.build_p3(0, norm, device)
    p3_b = sac.build_p3(0, norm, device)
    s_a = sac.build_sac(0, norm, device)
    s_b = sac.build_sac(0, norm, device)
    reader_content_equal = torch.equal(
        s_a.reader.lins[0].weight[:, : int(p3_a.config.reader_in)], p3_a.reader.lins[0].weight
    ) and torch.equal(s_a.reader.lins[0].bias, p3_a.reader.lins[0].bias)
    later_layers_equal = all(
        torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias)
        for a, b in zip(s_a.reader.lins[1:], p3_a.reader.lins[1:])
    )
    g5 = {
        "sac_r_u_full_state_identical": sac.full_state_hash(s_a) == sac.full_state_hash(s_b),
        "p3_repeatable": sac.full_state_hash(p3_a) == sac.full_state_hash(p3_b),
        "shared_trunk_matches_p3": sac.shared_state_hash(s_a) == sac.shared_state_hash(p3_a),
        "reader_content_columns_match_p3": bool(reader_content_equal),
        "reader_later_layers_match_p3": bool(later_layers_equal),
        "sac_parameters": s_a.n_parameters,
        "p3_parameters": int(sum(p.numel() for p in p3_a.parameters())),
        "sac_new_parameters": s_a.new_parameters,
        "param_cap_ok": s_a.n_parameters <= sac.PARAM_CAP,
        "sac_full_state_sha256": sac.full_state_hash(s_a),
        "p3_full_state_sha256": sac.full_state_hash(p3_a),
        "shared_trunk_sha256": sac.shared_state_hash(s_a),
    }
    gates.append(_gate("G5_matched_initialisation", all(v for k, v in g5.items() if isinstance(v, bool)) and g5["param_cap_ok"], **g5))

    # ---- G6 gradient liveness -------------------------------------------
    y = torch.tensor([-1.5, 0.5, 2.0], dtype=torch.float32, device=device)
    probe_R = sac.gradient_probe(smodel, _batch(objs_a, device), rel_a.batch([0, 1, 2], "R", device), y)
    probe_U = sac.gradient_probe(smodel, _batch(objs_a, device), rel_a.batch([0, 1, 2], "U", device), y)
    grad_keys = [k for k in probe_R if k.startswith("grad_norm_")]
    live = all(probe_R[k] > 0 for k in grad_keys) and all(probe_U[k] > 0 for k in grad_keys)
    gates.append(
        _gate(
            "G6_gradient_liveness",
            live and probe_R["grad_norm_T_R1"] > 0 and probe_R["grad_norm_T_R3"] > 0,
            probe_R=probe_R,
            probe_U=probe_U,
        )
    )

    # ---- G7 relation-assignment witness ---------------------------------
    witness_mol = fused()
    wr = _zone_rel([witness_mol])
    feats = smodel.base.forward_features(_batch([witness_mol], device))
    q = [torch.tanh(lin(feats["c"])).detach() for lin in smodel.project]
    n_b = int(witness_mol.bond_index.shape[0])
    rel_R = wr.batch([0], "R", device)
    rel_U = wr.batch([0], "U", device)
    T_R = sac.graph_tensors(q, rel_R)[0]
    T_U = sac.graph_tensors(q, rel_U)[0]
    best = None
    n_non_auto = 0
    u_change_max = 0.0
    for i in range(n_b):
        for j in range(i + 1, n_b):
            cand = np.arange(n_b)
            cand[i], cand[j] = cand[j], cand[i]
            if _relation_automorphism(cand, wr, 0):
                continue
            n_non_auto += 1
            perm_t = torch.as_tensor(cand, dtype=torch.long)
            q_perm = [x[perm_t] for x in q]
            T_R_perm = sac.graph_tensors(q_perm, rel_R)[0]
            T_U_perm = sac.graph_tensors(q_perm, rel_U)[0]
            d_r = max(float((a - b).abs().max()) for a, b in zip(T_R, T_R_perm))
            d_u = max(float((a - b).abs().max()) for a, b in zip(T_U, T_U_perm))
            u_change_max = max(u_change_max, d_u)
            if best is None or d_r > best["delta_R"]:
                best = {"transposition": [int(i), int(j)], "delta_R": d_r, "delta_U": d_u}
    require(best is not None, "no non-automorphism transposition found for the witness")
    delta_R = float(best["delta_R"])
    delta_U = float(u_change_max)
    c0 = int(best["transposition"][0])
    c1 = int(best["transposition"][1])
    cand = np.arange(n_b)
    cand[c0], cand[c1] = cand[c1], cand[c0]
    q_perm = [x[torch.as_tensor(cand, dtype=torch.long)] for x in q]
    multiset_equal = bool(torch.equal(torch.sort(q[0], dim=0).values, torch.sort(q_perm[0], dim=0).values))
    gates.append(
        _gate(
            "G7_relation_assignment_witness",
            bool(multiset_equal and delta_R >= 1e-4 and delta_R >= 100.0 * max(delta_U, 1e-9)),
            transposition=best["transposition"],
            n_non_automorphism_transpositions=int(n_non_auto),
            bond_content_multiset_preserved=multiset_equal,
            T_R_max_abs_change=delta_R,
            T_U_max_abs_change=delta_U,
            ratio=float(delta_R / max(delta_U, 1e-12)),
            interpretation="same bond-content multiset, different structural assignment: real tensors move, the content null is invariant",
        )
    )

    # ---- G8 no message passing / no per-bond aggregation ------------------
    base_feats = smodel.base.forward_features(_batch(objs_a, device))
    rel_R3 = rel_a.batch([0, 1, 2], "R", device)
    sac_feats = smodel.forward_features(_batch(objs_a, device), rel_R3)
    c_unchanged = torch.equal(base_feats["c"], sac_feats["c"])
    z_unchanged = torch.equal(base_feats["z"], sac_feats["z"])
    q_pure = all(torch.equal(sac_feats["q"][s], torch.tanh(lin(sac_feats["c"]))) for s, lin in enumerate(smodel.project))
    # explicit double-loop reference for T
    ref_ok = True
    for s in range(sac.N_RELATIONS):
        ref = np.zeros((1, sac.PROJECT_DIM), dtype=np.float64)
        q_np = q[s].detach().numpy().astype(np.float64)
        a = wr.src[s][wr.starts[0, s] : wr.ends[0, s]]
        b = wr.dst[s][wr.starts[0, s] : wr.ends[0, s]]
        for x, z in zip(a.tolist(), b.tolist()):
            ref[0] += q_np[x] * q_np[z]
        ref[0] *= 1.0 / math.sqrt(max(1, int(wr.m[0, s])))
        ref_ok = ref_ok and float(np.abs(ref - T_R[s].detach().numpy()).max()) <= 2e-5
    forbidden = src.forbidden_module_scan(smodel)
    src_text = (_CODE_DIR / "sac_v1.py").read_text(encoding="utf-8")
    driver_text = (_CODE_DIR / "run_sac_v1.py").read_text(encoding="utf-8")
    scan_tokens = ["Message" + "Passing", "Multihead" + "Attention", "Transformer" + "Encoder", "nn." + "GRU", "nn." + "LSTM", "scatter_" + "softmax"]
    token_hits = [t for t in scan_tokens if t in src_text or t in driver_text]
    index_add_lines = [ln.strip() for ln in src_text.splitlines() if "index_add" in ln and not ln.strip().startswith("#")]
    aggregation_only_graph_level = all("accum.index_add" in ln for ln in index_add_lines) and len(index_add_lines) == 1
    gates.append(
        _gate(
            "G8_no_message_passing",
            c_unchanged
            and z_unchanged
            and q_pure
            and ref_ok
            and not forbidden
            and not token_hits
            and aggregation_only_graph_level,
            composed_bond_vector_unchanged=c_unchanged,
            z_unchanged=z_unchanged,
            q_depends_only_on_own_bond=bool(q_pure),
            T_matches_explicit_double_loop=bool(ref_ok),
            forbidden_modules=forbidden,
            forbidden_tokens=token_hits,
            index_add_sites=index_add_lines,
            note="the only index_add is the graph-level pair reduction; c_e is never refreshed",
        )
    )

    # ---- G9 degenerate / extreme numeric stability ----------------------
    small_objs = [one, two, ring(3), chain(3)]
    small_rel = _zone_rel(small_objs)
    stable = True
    detail9: dict[str, Any] = {}
    for mode in ("R", "U"):
        relb = small_rel.batch(list(range(len(small_objs))), mode, device)
        with torch.no_grad():
            feats = smodel.forward_features(_batch(small_objs, device), relb)
            pred = smodel(_batch(small_objs, device), relb)
            no_nan = bool(torch.isfinite(pred).all() and all(torch.isfinite(t).all() for t in feats["T"]))
            zero_rel_m = [float(feats["T"][s][0].abs().max()) for s in range(sac.N_RELATIONS)]
        stable = stable and no_nan and zero_rel_m[0] == 0.0 and zero_rel_m[1] == 0.0 and zero_rel_m[2] == 0.0
        detail9[mode] = {
            "finite": no_nan,
            "single_bond_T_abs_max": zero_rel_m,
            "n_pairs_per_relation": [int(relb["src"][s].numel()) for s in range(sac.N_RELATIONS)],
        }
    # U with a single bond must be strictly zero even though m_s / C(n,2) is undefined
    solo = small_rel.batch([0], "U", device)
    with torch.no_grad():
        feats_solo = smodel.forward_features(_batch([one], device), solo)
    single_zero = all(float(t.abs().max()) == 0.0 for t in feats_solo["T"])
    gates.append(_gate("G9_degenerate_stability", stable and single_zero, **detail9, U_single_bond_strict_zero=bool(single_zero)))

    # ---- G10 label discipline + no official valid/test -------------------
    guard = ab.LabelGuard([type("M", (), {"x": np.zeros((2, 1)), "edge_index": np.zeros((2, 0), dtype=np.int64), "edge_attr": np.zeros((0,), dtype=np.int64), "y": np.asarray([1.0])})() for _ in range(2)], [1])
    blocked = False
    try:
        guard[1].y  # type: ignore[attr-defined]
    except Exception:
        blocked = True
    structure_ok = hasattr(guard[1], "x")
    driver_scan = []
    for path in (_CODE_DIR / "run_sac_v1.py", _CODE_DIR / "sac_v1.py"):
        text = path.read_text(encoding="utf-8")
        for token in ["split=" + q + z + q for q in ('"', "'") for z in ("valid", "test")]:
            if token in text:
                driver_scan.append(f"{path.name}:{token}")
    require(not driver_scan, f"official split token in sac_v1.py: {driver_scan}")
    gates.append(
        _gate(
            "G10_label_discipline",
            blocked and structure_ok and not driver_scan,
            label_guard_selfcheck=ab.label_guard_selfcheck(),
            guarded_y_blocked=blocked,
            guarded_structure_readable=structure_ok,
            official_loader_tokens=driver_scan,
        )
    )

    # ---- G11 frozen-trainer equivalence (P3) ----------------------------
    eq_objs = [load_zone_objects("fit")[i] for i in range(0, 64)]
    n_eq = len(eq_objs)
    rows_eq = [int(o.mol_index) for o in eq_objs]
    y_eq = bp.load_y_rows(ENCODED_TRAIN, rows_eq)
    ref = bp.train_supervised(
        arm="P3",
        seed=0,
        recipe_key="R3",
        init="scratch",
        train_objs=eq_objs,
        rows=rows_eq,
        y_fit=y_eq,
        norm=norm,
        device=device,
        model_kind="base",
        smoke_epochs=EQUIV_EPOCHS,
        log_fn=lambda *_: None,
    )
    mine = sac.train_supervised_sac(
        arm="P3",
        seed=0,
        model=sac.build_p3(0, norm, device),
        relations=None,
        train_objs=eq_objs,
        y_fit=y_eq,
        device=device,
        recipe_key="R3",
        epochs=EQUIV_EPOCHS,
        log_fn=lambda *_: None,
    )
    ref_soup = ref.get("soup_state")
    mine_soup = mine["soup_state"]
    ref_hash = bp.state_group_sha256(ref_soup) if ref_soup is not None else ""
    mine_hash = bp.state_group_sha256(mine_soup)
    gates.append(
        _gate(
            "G11_trainer_equivalence",
            ref_hash == mine_hash and ref["batch_order"]["full_hash"] == mine["batch_order"]["full_hash"],
            reference_soup_sha256=ref_hash,
            sac_loop_soup_sha256=mine_hash,
            batch_order_equal=ref["batch_order"]["full_hash"] == mine["batch_order"]["full_hash"],
            epochs=EQUIV_EPOCHS,
            n_molecules=n_eq,
            note="bitwise soup equality with the frozen bjoc_perfmech_long_v1 trainer for P3",
        )
    )

    # ---- G12 content-null identity / scaling ----------------------------
    # U equals the specified formula exactly; both arms keep the same total weight m_s.
    u_ok = True
    u_detail = {}
    for mol_i, obj in enumerate(objs_a):
        zz = _zone_rel([obj])
        relu = zz.batch([0], "U", device)
        feats = smodel.base.forward_features(_batch([obj], device))
        qq = [torch.tanh(lin(feats["c"])).detach() for lin in smodel.project]
        T_u = sac.graph_tensors(qq, relu)[0]
        n = int(zz.n_bonds[0])
        for s in range(sac.N_RELATIONS):
            m_s = int(zz.m[0, s])
            q_np = qq[s].numpy().astype(np.float64)
            total = np.zeros(sac.PROJECT_DIM, dtype=np.float64)
            for a in range(n):
                for b in range(a + 1, n):
                    total += q_np[a] * q_np[b]
            if n < 2 or m_s == 0:
                expected = np.zeros(sac.PROJECT_DIM)
            else:
                expected = (m_s / (n * (n - 1) / 2)) * total / math.sqrt(m_s)
            err = float(np.abs(expected - T_u[s].numpy()).max())
            u_ok = u_ok and err <= 2e-5
            u_detail[f"mol{mol_i}_s{s}"] = {"m": m_s, "n": n, "max_abs_err": err}
    gates.append(_gate("G12_content_null_identity", u_ok, detail=u_detail))

    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "n_gates": len(gates),
        "n_passed": int(sum(1 for g in gates if g["ok"])),
        "all_passed": bool(all(g["ok"] for g in gates)),
        "gates": gates,
        "torch": torch.__version__,
        "device": str(device),
        "labels_read": "Fit y only (gate 11 trainer equivalence)",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "cpu_gates.json", payload)
    for g in gates:
        log(f"[cpu-gates] {'PASS' if g['ok'] else 'FAIL'} {g['name']}")
    require(payload["all_passed"], "CPU gates failed; refusing to freeze or run GPU work")
    return 0


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------


def cmd_freeze(args: argparse.Namespace) -> int:
    require(not freeze_path().exists(), "already frozen")
    gates = read_json(OUT / "cpu_gates.json")
    require(gates["all_passed"], "CPU gates did not all pass")
    require(bool(args.attest_no_gpu_before_freeze), "--attest-no-gpu-before-freeze is required")
    inputs = {
        split_path().name: split_path(),
        prep_path().name: prep_path(),
        "objects_manifest_S.json": OUT / "objects_manifest_S.json",
        "relations_manifest_S.json": OUT / "relations_manifest_S.json",
        "relation_audit.json": OUT / "relation_audit.json",
        "cpu_gates.json": OUT / "cpu_gates.json",
        "input_audit.json": OUT / "input_audit.json",
        "split_compare_S.json": OUT / "split_compare_S.json",
        "pairing_verification.json": OUT / "pairing_verification.json",
        "PROTOCOL.md": PROTOCOL_DOC,
    }
    payload = {
        "protocol": PROTOCOL_ID,
        "split_tag": SPLIT_TAG,
        "git_commit": git_commit(),
        "frozen_utc": now_utc(),
        "code": {name: {"sha256": sha} for name, sha in code_hashes().items()},
        "config": {"path": str(CONFIG_PATH.relative_to(REPO_ROOT)), "sha256": config_sha256()},
        "inputs": {name: {"sha256": file_sha256(path)} for name, path in inputs.items()},
        "upstream_input_hashes": FROZEN_INPUT_HASHES,
        "parameters": {
            "p3": sac.BASE_PARAMETERS,
            "sac": sac.EXPECTED_SAC_PARAMETERS,
            "sac_new": 26_184,
            "cap": sac.PARAM_CAP,
        },
        "budget": {
            "cap_gpu_hours": GPU_BUDGET_CAP_HOURS,
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "gpu_assignment": {"0": 0, "1": 1},
            "no_seed_extension": True,
            "no_hyperparameter_search": True,
        },
        "labels_before_freeze": "Fit y only (gate 11 trainer equivalence)",
        "monitor_labels_read": False,
        "locked_labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "gpu_started_before_freeze": False,
    }
    write_json(freeze_path(), payload)
    log(f"[freeze] code={payload['code']['sac_v1.py']['sha256'][:12]} config={payload['config']['sha256'][:12]}")
    return 0


# ---------------------------------------------------------------------------
# stage inputs (compute checkout)
# ---------------------------------------------------------------------------


def cmd_stage_inputs(_args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    require(not (OUT / "holdout_read.lock").exists(), "refusing to restage inputs after labels were read")
    split, ds, guard = build_guard()
    frozen_rel = read_json(OUT / "relations_manifest_S.json")
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
        mols = [sac.build_mol_relations(o) for o in objs[zone]]
        zr = sac.build_zone_relations(mols)
        torch.save(zr.to_state(), relations_path(zone))
        totals = zr.totals()
        expected = {k: int(v) for k, v in frozen_rel["zones"][zone]["n_pairs"].items()}
        require(totals == expected, f"{zone}: relation totals changed {totals} vs {expected}")
        require(zr.n_complete() == int(frozen_rel["zones"][zone]["n_complete_pairs"]), f"{zone}: complete pair count changed")
        require(int(zr.m.sum()) == sum(expected.values()), f"{zone}: relation count sum changed")
        payload["zones"][zone] = {
            "n_molecules": len(zr),
            "n_pairs": totals,
            "n_complete_pairs": zr.n_complete(),
            "n_pairs_match_frozen": True,
            "objects_file_sha256": file_sha256(objects_path(zone)),
            "relations_file_sha256": file_sha256(relations_path(zone)),
        }
        log(f"[stage-inputs {zone}] n={len(zr)} pairs={totals} complete={zr.n_complete()} (match frozen)")
    require(guard.access_attempts == [], "guarded label access during staging")
    payload.update(
        {
            "freeze_sha256": file_sha256(freeze_path()),
            "standardiser_max_abs_diff": {"mean": mean_diff, "scale": scale_diff},
            "build_seconds": time.perf_counter() - started,
            "forbidden_label_access_attempts": len(guard.access_attempts),
            "note": "torch .pt bytes are machine-local; equality is verified semantically "
            "(row order, standardiser, per-relation pair totals, complete-pair totals)",
            "monitor_labels_read": False,
            "locked_labels_read": False,
        }
    )
    write_json(OUT / "staged_inputs_S.json", payload)
    log(f"[stage-inputs] ok in {payload['build_seconds']:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# smoke (post-freeze, Fit labels only)
# ---------------------------------------------------------------------------


def cmd_smoke(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    device = torch.device(args.device)
    split, ds, guard = build_guard()
    fit_objs = load_zone_objects("fit")
    fit_rel = load_zone_relations("fit")
    mini = fit_objs[:512]
    norm = load_norm()
    y_mini = bp.load_y_rows(ENCODED_TRAIN, [int(i) for i in split["fit_rows"][:512]])
    smoke_dir().mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"protocol": PROTOCOL_ID, "device": str(device), "n_mini": len(mini), "arms": {}}

    res = bp.pretrain_masked(
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
    enc = res["encoder_state"]
    out["ssl"] = {
        "updates": 20,
        "wall_seconds": res["wall_seconds"],
        "updates_per_second": 20.0 / max(res["wall_seconds"], 1e-9),
        "projected_full_seconds": res["wall_seconds"] / 20.0 * mp.PRETRAIN_UPDATES,
    }
    torch.save({g: {k: v.cpu() for k, v in s.items()} for g, s in enc.items()}, smoke_dir() / "ssl_s1_smoke.pt")

    projected = dict(out["ssl"]["projected_full_seconds"])
    for arm in ARMS:
        model = sac.build_arm(arm, 1, norm, device)
        mp.load_encoder_state(model.base if isinstance(model, sac.SAC) else model, enc, device)
        rel = fit_rel if arm != "P3" else None
        info = sac.train_supervised_sac(
            arm=arm,
            seed=1,
            model=model,
            relations=rel,
            train_objs=mini,
            y_fit=y_mini,
            device=device,
            recipe_key="R3",
            epochs=1,
            log_fn=log,
        )
        epoch_seconds = float(info["curve"][-1]["seconds"])
        projected[arm] = epoch_seconds * int(bp.RECIPES["R3"]["epochs"])
        pred = sac.predict_model(model, mini, rel, "R" if arm != "P3" else None, device)
        out["arms"][arm] = {
            "parameters": int(info["parameter_audit"]["actual_parameters"]),
            "fit_mae_after_1_epoch": float(np.abs(pred - y_mini).mean()),
            "epoch_seconds_on_512": epoch_seconds,
            "projected_360_epoch_seconds": epoch_seconds * int(bp.RECIPES["R3"]["epochs"]),
            "molecules_per_second": float(len(mini) / max(epoch_seconds, 1e-9)),
            "wall_seconds": info["wall_seconds"],
        }
        log(
            f"[smoke {arm}] params={out['arms'][arm]['parameters']} mae={out['arms'][arm]['fit_mae_after_1_epoch']:.4f} "
            f"512-mol epoch={epoch_seconds:.1f}s -> projected {projected[arm]/60:.0f} min"
        )
    total_projected = 0.0
    for seed in SEEDS:
        total_projected += projected["SSL"]
        for arm in ARMS:
            total_projected += projected[arm]
    out["projection"] = {
        "per_seed_seconds": {k: float(v) for k, v in projected.items()},
        "two_seed_total_seconds": float(total_projected),
        "two_seed_total_gpu_hours": float(total_projected / 3600.0),
        "cap_gpu_hours": GPU_BUDGET_CAP_HOURS,
        "within_cap": bool(total_projected / 3600.0 <= GPU_BUDGET_CAP_HOURS * 0.9),
    }
    out["device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    require(guard.access_attempts == [], "guarded label access during smoke training")
    write_json(smoke_dir() / "smoke_S.json", out)
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
    log(f"[ssl s{seed}] device={args.device} n={len(objs)} updates={smoke_updates or mp.PRETRAIN_UPDATES}")
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
    out_pt = ssl_path(seed) if smoke_updates is None else smoke_dir() / f"ssl_S_s{seed}_smoke.pt"
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
        write_json(smoke_dir() / f"ssl_S_s{seed}_smoke.json", res)
    log(f"[ssl s{seed}] wall={res['wall_seconds']:.0f}s atom_acc={res.get('final_acc_atom'):.4f}")
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    seed = int(args.seed)
    device = torch.device(args.device)
    arms = [args.arm] if args.arm else list(ARMS)
    norm = load_norm()
    ssl = load_ssl(seed)
    split = load_split()
    fit_objs = load_zone_objects("fit")
    fit_rel = load_zone_relations("fit")
    require([o.mol_index for o in fit_objs] == [int(i) for i in split["fit_rows"]], "fit objects do not match split")
    y_fit = load_fit_y()
    _, ds, guard = build_guard()
    out: dict[str, Any] = {"protocol": PROTOCOL_ID, "seed": seed, "device": str(device), "arms": {}}

    probe_n = min(128, len(fit_objs))
    probe_objs = list(fit_objs[:probe_n])
    probe_batch = {k: v.to(device) for k, v in sro.batch_objects(probe_objs).items()}
    probe_y = torch.as_tensor(y_fit[:probe_n], dtype=torch.float32, device=device)

    for arm in arms:
        if train_json_path(arm, seed).exists() and not args.retrain:
            raise SystemExit(f"{arm} seed {seed} already trained; refusing to retrain")
        model = sac.build_arm(arm, seed, norm, device)
        base = model.base if isinstance(model, sac.SAC) else model
        mp.load_encoder_state(base, ssl, device)
        rel = fit_rel if arm != "P3" else None
        mode = sac.RELATION_MODE_OF_ARM.get(arm)
        probe_rel = fit_rel.batch(list(range(probe_n)), mode, device) if rel is not None else None  # type: ignore[arg-type]
        init_info = {
            "full_state_sha256": sac.full_state_hash(model),
            "shared_trunk_sha256": sac.shared_state_hash(model),
            "encoder_sha256": bp.encoder_group_hash(base),
            "ssl_full_encoder_group_sha256": ssl.get("full_encoder_group_sha256"),
            "parameters": sac.model_parameters(model),
        }
        grad_init = sac.gradient_probe(model, probe_batch, probe_rel, probe_y)
        info = sac.train_supervised_sac(
            arm=arm,
            seed=seed,
            model=model,
            relations=rel,
            train_objs=fit_objs,
            y_fit=y_fit,
            device=device,
            recipe_key="R3",
            log_fn=log,
        )
        require(guard.access_attempts == [], "guarded label access during training")
        grad_final = sac.gradient_probe(model, probe_batch, probe_rel, probe_y)
        soup = info.pop("soup_state")
        torch.save(soup, soup_path(arm, seed))
        info.update(
            {
                "protocol": PROTOCOL_ID,
                "split_tag": SPLIT_TAG,
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "init": init_info,
                "gradient_probe": {"at_init": grad_init, "after_soup": grad_final},
                "soup_file": str(soup_path(arm, seed).relative_to(REPO_ROOT)),
                "soup_sha256": file_sha256(soup_path(arm, seed)),
                "n_relation_pairs_per_molecule": (
                    float(sum(fit_rel.totals().values()) / max(len(fit_objs), 1)) if rel is not None else 0.0
                ),
                "n_complete_pairs_per_molecule": (
                    float(fit_rel.n_complete() / max(len(fit_objs), 1)) if rel is not None else 0.0
                ),
                "forbidden_label_access_attempts": len(guard.access_attempts),
                "official_valid_loaded": False,
                "official_test_loaded": False,
            }
        )
        write_json(train_json_path(arm, seed), info)
        out["arms"][arm] = {
            "soup_fit_mae": info["soup_fit_mae"],
            "wall_seconds": info["wall_seconds"],
            "parameters": info["parameter_audit"]["actual_parameters"],
            "shared_trunk_sha256": init_info["shared_trunk_sha256"],
            "soup_sha256": info["soup_sha256"],
            "grad_T_R1_after_soup": grad_final.get("grad_norm_T_R1"),
            "grad_T_R3_after_soup": grad_final.get("grad_norm_T_R3"),
        }
        _append_ledger({"phase": "train", "arm": arm, "seed": seed, "wall_seconds": info["wall_seconds"], "device": str(device)})
        log(
            f"[train {arm} s{seed}] fit_mae={info['soup_fit_mae']:.6f} wall={info['wall_seconds']:.0f}s "
            f"params={info['parameter_audit']['actual_parameters']}"
        )
    write_json(OUT / f"train_round_S_s{seed}.json", out)
    return 0


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def _load_soup_model(arm: str, seed: int, norm, device) -> torch.nn.Module:
    model = sac.build_arm(arm, seed, norm, device)
    state = torch.load(soup_path(arm, seed), map_location="cpu", weights_only=False)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.eval()
    return model


def _record_prediction(
    path: Path,
    zone: str,
    arm: str,
    seed: int,
    pred: np.ndarray,
    rows: Sequence[int],
    kind: str,
    kernel_mode: str | None = None,
    zero_tensors: bool = False,
) -> dict[str, Any]:
    require(pred.shape == (len(rows),), f"{zone}/{arm}: prediction shape {pred.shape}")
    require(bool(np.all(np.isfinite(pred))), f"{zone}/{arm}: non-finite prediction")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, pred=pred.astype(np.float64), rows=np.asarray(rows, dtype=np.int64))
    record = {
        "file": path.name,
        "file_sha256": file_sha256(path),
        "zone": zone,
        "arm": arm,
        "kind": kind,
        "kernel_mode": kernel_mode,
        "zero_tensors": bool(zero_tensors),
        "seed": int(seed),
        "n": int(len(pred)),
        "pred_sha256": array_sha256(pred),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "rows_sha256": rows_sha256(list(rows)),
        "frozen_utc": now_utc(),
    }
    with _file_lock(OUT / "prediction_manifest_S.json"):
        manifest = (
            read_json(OUT / "prediction_manifest_S.json")
            if (OUT / "prediction_manifest_S.json").exists()
            else {"protocol": PROTOCOL_ID, "records": {}, "complete": False}
        )
        manifest["records"][path.name] = record
        write_json(OUT / "prediction_manifest_S.json", manifest)
    return record


def cmd_predict(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    seed = int(args.seed)
    device = torch.device(args.device)
    norm = load_norm()
    models = {arm: _load_soup_model(arm, seed, norm, device) for arm in ARMS}
    for zone in ZONES:
        objs = load_zone_objects(zone)
        rel = load_zone_relations(zone)
        rows = [int(o.mol_index) for o in objs]
        for arm in ARMS:
            mode = sac.RELATION_MODE_OF_ARM.get(arm)
            pred = sac.predict_model(models[arm], objs, rel if arm != "P3" else None, mode, device)
            rec = _record_prediction(pred_path(zone, arm, seed), zone, arm, seed, pred, rows, "own", mode)
            log(f"[predict {zone} {arm} s{seed}] sha={rec['pred_sha256'][:12]} mean={rec['pred_mean']:.4f}")
        # frozen-model x swapped-relation diagnostic (label-free)
        for arm, other in (("SACR", "U"), ("SACU", "R")):
            pred = sac.predict_model(models[arm], objs, rel, other, device)
            rec = _record_prediction(swap_pred_path(zone, arm, other, seed), zone, arm, seed, pred, rows, "swap", other)
            log(f"[predict-swap {zone} m{arm} k{other} s{seed}] sha={rec['pred_sha256'][:12]}")
        # tensor-zeroing diagnostic (T_1..T_3 -> 0, counts kept)
        for arm in sac.RELATION_ARMS:
            mode = sac.RELATION_MODE_OF_ARM[arm]
            pred = sac.predict_model(models[arm], objs, rel, mode, device, zero_tensors=True)
            rec = _record_prediction(zero_pred_path(zone, arm, seed), zone, arm, seed, pred, rows, "zero_tensors", mode, True)
            log(f"[predict-zeroT {zone} {arm} s{seed}] sha={rec['pred_sha256'][:12]}")
    return 0


def cmd_freeze_predictions(_args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    manifest = read_json(OUT / "prediction_manifest_S.json")
    expected = {pred_path(z, a, s).name for z in ZONES for a in ARMS for s in SEEDS}
    expected |= {swap_pred_path(z, a, m, s).name for z in ZONES for a, m in (("SACR", "U"), ("SACU", "R")) for s in SEEDS}
    expected |= {zero_pred_path(z, a, s).name for z in ZONES for a in sac.RELATION_ARMS for s in SEEDS}
    missing = sorted(expected - set(manifest["records"]))
    require(not missing, f"prediction set incomplete: {missing}")
    hashes = []
    for name in sorted(manifest["records"]):
        path = OUT / "predictions" / name
        require(file_sha256(path) == manifest["records"][name]["file_sha256"], f"prediction drifted: {name}")
        hashes.append(f"{name}:{manifest['records'][name]['file_sha256']}")
    combined = hashlib.sha256("\n".join(hashes).encode()).hexdigest()

    # label-free participation diagnostic, frozen together with the predictions
    participation = {}
    for zone in ZONES:
        rows = {name: np.load(OUT / "predictions" / name)["pred"] for name in manifest["records"] if manifest["records"][name]["zone"] == zone}
        for arm, other in (("SACR", "U"), ("SACU", "R")):
            for seed in SEEDS:
                own = rows[pred_path(zone, arm, seed).name]
                swapped = rows[swap_pred_path(zone, arm, other, seed).name]
                d = np.abs(swapped - own)
                participation[f"{zone}|{arm}|seed{seed}|swap_to_{other}"] = {
                    "mean_abs_delta": float(d.mean()),
                    "max_abs_delta": float(d.max()),
                    "p99_abs_delta": float(np.quantile(d, 0.99)),
                }
            mean_d = float(
                np.mean([participation[f"{zone}|{arm}|seed{s}|swap_to_{other}"]["mean_abs_delta"] for s in SEEDS])
            )
            max_d = float(
                np.max([participation[f"{zone}|{arm}|seed{s}|swap_to_{other}"]["max_abs_delta"] for s in SEEDS])
            )
            participation[f"{zone}|{arm}|swap_to_{other}|seed_mean"] = {
                "mean_abs_delta": mean_d,
                "max_abs_delta": max_d,
                "criterion_mean_ok": bool(mean_d >= sac.PARTICIPATION_MEAN_ABS),
                "criterion_max_ok": bool(max_d >= sac.PARTICIPATION_MAX_ABS),
                "passes": bool(mean_d >= sac.PARTICIPATION_MEAN_ABS and max_d >= sac.PARTICIPATION_MAX_ABS),
            }
        for arm in sac.RELATION_ARMS:
            for seed in SEEDS:
                own = rows[pred_path(zone, arm, seed).name]
                zero = rows[zero_pred_path(zone, arm, seed).name]
                d = np.abs(zero - own)
                participation[f"{zone}|{arm}|seed{seed}|zero_T"] = {
                    "mean_abs_delta": float(d.mean()),
                    "max_abs_delta": float(d.max()),
                }
    out = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "n_files": len(hashes),
        "combined_pred_sha256": combined,
        "records": manifest["records"],
        "participation_label_free": participation,
        "participation_criterion": {
            "zone": sac.PRIMARY_ZONE,
            "mean_abs_delta_min": sac.PARTICIPATION_MEAN_ABS,
            "max_abs_delta_min": sac.PARTICIPATION_MAX_ABS,
            "arm": "SACR",
        },
        "frozen_utc": now_utc(),
        "labels_read_after_this_point": True,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "prediction_manifest_S.json", {**out, "complete": True})
    log(f"[freeze-predictions] {len(hashes)} files combined={combined[:16]}")
    return 0


# ---------------------------------------------------------------------------
# score holdout
# ---------------------------------------------------------------------------


def _load_preds(zone: str, arm: str) -> dict[int, np.ndarray]:
    return {seed: np.asarray(np.load(pred_path(zone, arm, seed))["pred"], dtype=np.float64) for seed in SEEDS}


def _load_swap(zone: str, model_arm: str, mode: str) -> dict[int, np.ndarray]:
    return {seed: np.asarray(np.load(swap_pred_path(zone, model_arm, mode, seed))["pred"], dtype=np.float64) for seed in SEEDS}


def _load_zero(zone: str, arm: str) -> dict[int, np.ndarray]:
    return {seed: np.asarray(np.load(zero_pred_path(zone, arm, seed))["pred"], dtype=np.float64) for seed in SEEDS}


def _arm_stats(y: np.ndarray, preds: Mapping[int, np.ndarray]) -> dict[str, Any]:
    err = {s: np.abs(preds[s] - y) for s in SEEDS}
    seed_mean = np.mean([err[s] for s in SEEDS], axis=0)
    sse = seed_mean**2
    order = np.argsort(-seed_mean)
    ordinal = y > sac.TAIL_THRESHOLD
    return {
        "mae_seed_mean": float(np.mean([err[s].mean() for s in SEEDS])),
        "mae_per_seed": {str(s): float(err[s].mean()) for s in SEEDS},
        "rmse_seed_mean": float(np.sqrt(np.mean(np.mean([(preds[s] - y) ** 2 for s in SEEDS], axis=0)))),
        "median_ae": float(np.median(seed_mean)),
        "p90_ae": float(np.quantile(seed_mean, 0.90)),
        "p95_ae": float(np.quantile(seed_mean, 0.95)),
        "p99_ae": float(np.quantile(seed_mean, 0.99)),
        "max_ae": float(seed_mean.max()),
        "ordinary_mae": float(seed_mean[ordinal].mean()),
        "tail_mae": float(seed_mean[~ordinal].mean()),
        "tail_n": int((~ordinal).sum()),
        "top10_sse_share": float(sse[order[:10]].sum() / max(sse.sum(), 1e-30)),
        "worst_rows": [{"row": int(i), "y": float(y[i]), "ae": float(seed_mean[i])} for i in order[:5]],
    }


def _contrast(y: np.ndarray, a: Mapping[int, np.ndarray], b: Mapping[int, np.ndarray], meaningful: float) -> dict[str, Any]:
    base = ab.compute_contrast(a, b, y, bootstrap_n=sac.BOOTSTRAP_N, bootstrap_seed=sac.BOOTSTRAP_SEED)
    label = sac.contrast_label(
        base["delta_seed_mean"],
        base["per_seed_delta"],
        base["bootstrap"]["ci_lo"],
        base["bootstrap"]["ci_hi"],
        base["ordinary_delta"],
        meaningful,
    )
    per_seed = {}
    for seed in SEEDS:
        one = ab.compute_contrast({seed: a[seed]}, {seed: b[seed]}, y, bootstrap_n=sac.BOOTSTRAP_N, bootstrap_seed=sac.BOOTSTRAP_SEED)
        per_seed[str(seed)] = {
            "delta": one["delta_seed_mean"],
            "ci_lo": one["bootstrap"]["ci_lo"],
            "ci_hi": one["bootstrap"]["ci_hi"],
            "mae_a": one["mae_new_seed_mean"],
            "mae_b": one["mae_ref_seed_mean"],
        }
    return {**base, "label": label, "meaningful_delta": float(meaningful), "per_seed_bootstrap": per_seed}


def cmd_score_holdout(_args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    lock = OUT / "holdout_read.lock"
    require(not lock.exists(), "held-out labels were already read in this round")
    manifest = read_json(OUT / "prediction_manifest_S.json")
    require(bool(manifest.get("complete")), "prediction manifest is not complete")
    for name, rec in manifest["records"].items():
        require(file_sha256(OUT / "predictions" / name) == rec["file_sha256"], f"prediction drifted: {name}")

    split = load_split()
    y = {zone: bp.load_y_rows(ENCODED_TRAIN, [int(i) for i in split[f"{zone}_rows"]]) for zone in ZONES}
    np.savez(OUT / "holdout_labels_S.npz", monitor=y["monitor"], locked=y["locked"])
    write_json(
        lock,
        {
            "protocol": PROTOCOL_ID,
            "read_utc": now_utc(),
            "n_monitor": int(len(y["monitor"])),
            "n_locked": int(len(y["locked"])),
            "monitor_y_sha256": array_sha256(y["monitor"]),
            "locked_y_sha256": array_sha256(y["locked"]),
            "monitor_rows_sha256": split["monitor_sha256_rows"],
            "locked_rows_sha256": split["locked_sha256_rows"],
            "prediction_manifest_sha256": manifest["combined_pred_sha256"],
            "single_read_both_zones": True,
        },
    )

    result: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": SPLIT_TAG,
        "delta_definition": "mean over molecules of |err_A| - |err_B|; negative = A better",
        "primary_zone": sac.PRIMARY_ZONE,
        "replication_zone": sac.REPLICATION_ZONE,
        "zones": {},
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    for zone in ZONES:
        preds = {arm: _load_preds(zone, arm) for arm in ARMS}
        stats = {arm: _arm_stats(y[zone], preds[arm]) for arm in ARMS}
        contrasts = {
            "SACR_minus_P3": _contrast(y[zone], preds["SACR"], preds["P3"], sac.MEANINGFUL_DELTA_R_VS_P3),
            "SACR_minus_SACU": _contrast(y[zone], preds["SACR"], preds["SACU"], sac.MEANINGFUL_DELTA_R_VS_U),
            "SACU_minus_P3": _contrast(y[zone], preds["SACU"], preds["P3"], sac.MEANINGFUL_DELTA_R_VS_P3),
        }
        # swapped / zeroed mechanism contrasts (no extra label read)
        mech = {}
        for arm, other in (("SACR", "U"), ("SACU", "R")):
            cross = _load_swap(zone, arm, other)
            d = np.mean([np.abs(cross[s] - preds[arm][s]) for s in SEEDS], axis=0)
            mech[f"{arm}_own_vs_{other}_relations"] = {
                "mean_abs_pred_delta": float(d.mean()),
                "max_abs_pred_delta": float(d.max()),
                "mae_own": stats[arm]["mae_seed_mean"],
                "mae_swapped": float(np.mean([np.abs(cross[s] - y[zone]).mean() for s in SEEDS])),
            }
        for arm in sac.RELATION_ARMS:
            zero = _load_zero(zone, arm)
            d = np.mean([np.abs(zero[s] - preds[arm][s]) for s in SEEDS], axis=0)
            mech[f"{arm}_own_vs_zero_T"] = {
                "mean_abs_pred_delta": float(d.mean()),
                "max_abs_pred_delta": float(d.max()),
                "mae_zero_T": float(np.mean([np.abs(zero[s] - y[zone]).mean() for s in SEEDS])),
            }
        result["zones"][zone] = {"arm_stats": stats, "contrasts": contrasts, "mechanism": mech}
        with (OUT / f"per_molecule_errors_{zone}.csv").open("w", newline="") as fh:
            rows = []
            for i in range(len(y[zone])):
                rec = {"row": i, "y": float(y[zone][i])}
                for arm in ARMS:
                    for seed in SEEDS:
                        rec[f"{arm}_s{seed}_pred"] = float(preds[arm][seed][i])
                        rec[f"{arm}_s{seed}_ae"] = float(abs(preds[arm][seed][i] - y[zone][i]))
                rows.append(rec)
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    primary = result["zones"][sac.PRIMARY_ZONE]["contrasts"]
    replication = result["zones"][sac.REPLICATION_ZONE]["contrasts"]
    rep = {
        name: {
            "seed_mean_delta": float(np.mean(list(replication[name]["per_seed_delta"].values()))),
            "per_seed_delta": dict(replication[name]["per_seed_delta"]),
            "replicated": bool(
                np.mean(list(replication[name]["per_seed_delta"].values())) < 0.0
                and all(v < 0.0 for v in replication[name]["per_seed_delta"].values())
            ),
        }
        for name in primary
    }
    part = manifest["participation_label_free"][
        f"{sac.PRIMARY_ZONE}|SACR|swap_to_U|seed_mean"
    ]
    participation_ok = bool(part["passes"])
    case = sac.sac_case(
        label_r_p3=primary["SACR_minus_P3"]["label"],
        label_r_u=primary["SACR_minus_SACU"]["label"],
        label_u_p3=primary["SACU_minus_P3"]["label"],
        replicated_r_p3=rep["SACR_minus_P3"]["replicated"],
        replicated_r_u=rep["SACR_minus_SACU"]["replicated"],
        replicated_u_p3=rep["SACU_minus_P3"]["replicated"],
        top10_r_p3=primary["SACR_minus_P3"]["top10_removed_delta"],
        participation=participation_ok,
    )
    strong_positive = {
        "SACR_minus_P3_le_0.004": bool(primary["SACR_minus_P3"]["delta_seed_mean"] <= -sac.MEANINGFUL_DELTA_R_VS_P3),
        "SACR_minus_SACU_le_0.003": bool(primary["SACR_minus_SACU"]["delta_seed_mean"] <= -sac.MEANINGFUL_DELTA_R_VS_U),
        "both_seeds_same_direction_R_minus_P3": bool(all(v < 0 for v in primary["SACR_minus_P3"]["per_seed_delta"].values())),
        "both_seeds_same_direction_R_minus_U": bool(all(v < 0 for v in primary["SACR_minus_SACU"]["per_seed_delta"].values())),
        "paired_ci_R_minus_P3_excludes_0": bool(primary["SACR_minus_P3"]["bootstrap"]["ci_hi"] < 0),
        "paired_ci_R_minus_U_excludes_0": bool(primary["SACR_minus_SACU"]["bootstrap"]["ci_hi"] < 0),
        "ordinary_not_worse": bool(primary["SACR_minus_P3"]["ordinary_delta"] <= 0.0),
        "top10_removed_not_worse": bool(primary["SACR_minus_P3"]["top10_removed_delta"] <= 0.0),
        "relation_tensors_participate": participation_ok,
    }
    result["replication"] = rep
    result["participation"] = {"locked_SACR_swap_to_U": part, "passes": participation_ok}
    result["strong_positive"] = strong_positive
    result["strong_positive_all"] = bool(all(strong_positive.values()))
    result["decision"] = case
    write_json(OUT / "holdout_score_S.json", result)
    write_json(
        OUT / "decision.json",
        {
            "protocol": PROTOCOL_ID,
            "git_commit": git_commit(),
            "split_tag": SPLIT_TAG,
            **case,
            "strong_positive": strong_positive,
            "strong_positive_all": result["strong_positive_all"],
            "primary_zone": sac.PRIMARY_ZONE,
            "primary_mae": {arm: result["zones"][sac.PRIMARY_ZONE]["arm_stats"][arm]["mae_seed_mean"] for arm in ARMS},
            "primary_labels": {k: v["label"] for k, v in primary.items()},
            "primary_deltas": {k: v["delta_seed_mean"] for k, v in primary.items()},
            "primary_ci": {k: [v["bootstrap"]["ci_lo"], v["bootstrap"]["ci_hi"]] for k, v in primary.items()},
            "replication_deltas": {k: v["seed_mean_delta"] for k, v in rep.items()},
            "replication_flags": {k: v["replicated"] for k, v in rep.items()},
            "full10k_authorized": False,
            "official_test_authorized": False,
            "frozen_utc": now_utc(),
        },
    )
    log(
        "[score-holdout] locked "
        + " ".join(f"{arm}={result['zones'][sac.PRIMARY_ZONE]['arm_stats'][arm]['mae_seed_mean']:.6f}" for arm in ARMS)
        + f" | R-P3={primary['SACR_minus_P3']['delta_seed_mean']:+.6f}"
        + f" R-U={primary['SACR_minus_SACU']['delta_seed_mean']:+.6f}"
        + f" U-P3={primary['SACU_minus_P3']['delta_seed_mean']:+.6f} -> {case['case']}"
    )
    return 0


# ---------------------------------------------------------------------------
# post hoc / finalize
# ---------------------------------------------------------------------------


def cmd_post_hoc(_args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    require((OUT / "holdout_read.lock").exists(), "score the frozen predictions first")
    score = read_json(OUT / "holdout_score_S.json")
    labels = np.load(OUT / "holdout_labels_S.npz")
    manifest = read_json(OUT / "prediction_manifest_S.json")
    out: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "note": "diagnostics on frozen models/predictions; labels from holdout_labels_S.npz (no new read)",
        "participation": manifest["participation_label_free"],
        "gradients": {},
        "regions": {},
        "delta_quantiles": {},
    }
    for arm in ARMS:
        for seed in SEEDS:
            info = read_json(train_json_path(arm, seed))
            out["gradients"][f"{arm}_s{seed}"] = info["gradient_probe"]
    for zone in ZONES:
        y = labels[zone]
        region = {}
        for arm in ARMS:
            preds = _load_preds(zone, arm)
            err = np.mean([np.abs(preds[s] - y) for s in SEEDS], axis=0)
            for name, mask in (
                ("ordinary", y > sac.TAIL_THRESHOLD),
                ("tail", y <= sac.TAIL_THRESHOLD),
                ("top_decile_error", err >= np.quantile(err, 0.9)),
            ):
                region[f"{arm}_{name}"] = {"n": int(mask.sum()), "mae": float(err[mask].mean())}
        out["regions"][zone] = region
        quant = {}
        for name, a, b in sac.CONTRASTS:
            preds_a, preds_b = _load_preds(zone, a), _load_preds(zone, b)
            delta = np.mean([np.abs(preds_a[s] - y) - np.abs(preds_b[s] - y) for s in SEEDS], axis=0)
            quant[name] = {
                "q10": float(np.quantile(delta, 0.10)),
                "median": float(np.median(delta)),
                "q90": float(np.quantile(delta, 0.90)),
                "frac_better": float((delta < 0).mean()),
            }
        out["delta_quantiles"][zone] = quant
        for arm in sac.RELATION_ARMS:
            preds = _load_preds(zone, arm)
            zero = _load_zero(zone, arm)
            other = "U" if arm == "SACR" else "R"
            swap = _load_swap(zone, arm, other)
            out[f"mechanism_{zone}_{arm}"] = {
                "mae_own": score["zones"][zone]["arm_stats"][arm]["mae_seed_mean"],
                "mae_zero_T": float(np.mean([np.abs(zero[s] - y).mean() for s in SEEDS])),
                "mae_swapped": float(np.mean([np.abs(swap[s] - y).mean() for s in SEEDS])),
                "mean_abs_delta_zero_T": float(np.mean([np.abs(zero[s] - preds[s]).mean() for s in SEEDS])),
                "mean_abs_delta_swapped": float(np.mean([np.abs(swap[s] - preds[s]).mean() for s in SEEDS])),
            }
    write_json(OUT / "post_hoc_S.json", out)
    log("[post-hoc] gradients/participation/regions/quantiles written")
    return 0


def cmd_finalize(_args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    require((OUT / "holdout_read.lock").exists(), "score the frozen predictions first")
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
        "prediction_manifest_sha256": read_json(OUT / "prediction_manifest_S.json")["combined_pred_sha256"],
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
    sub.add_parser("relations")
    sub.add_parser("cpu-gates")
    p_freeze = sub.add_parser("freeze")
    p_freeze.add_argument("--attest-no-gpu-before-freeze", action="store_true", default=False)
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
    p_train.add_argument("--arm", default=None, choices=list(ARMS))
    p_train.add_argument("--retrain", action="store_true")
    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--seed", type=int, required=True)
    p_pred.add_argument("--device", default="cuda:0")
    sub.add_parser("freeze-predictions")
    sub.add_parser("score-holdout")
    sub.add_parser("post-hoc")
    sub.add_parser("finalize")
    args = parser.parse_args(argv)

    table = {
        "split": cmd_split,
        "verify-inputs": cmd_verify_inputs,
        "objects": cmd_objects,
        "relations": cmd_relations,
        "cpu-gates": cmd_cpu_gates,
        "freeze": cmd_freeze,
        "stage-inputs": cmd_stage_inputs,
        "smoke": cmd_smoke,
        "ssl": cmd_ssl,
        "train": cmd_train,
        "predict": cmd_predict,
        "freeze-predictions": cmd_freeze_predictions,
        "score-holdout": cmd_score_holdout,
        "post-hoc": cmd_post_hoc,
        "finalize": cmd_finalize,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    return int(table[args.cmd](args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
