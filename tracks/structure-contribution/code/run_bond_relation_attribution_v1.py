#!/usr/bin/env python
"""Driver for protocol ``cscl-bond-relation-attribution-v1``.

Stages
------
``split``              (local CPU, needs RDKit; label-blind) reproduce the four
                       SMILES/PyG mismatches, define identity from the *PyG graph*,
                       build the new 7,996/1,000/1,000 identity-grouped split.
``verify-inputs``      hashes of every frozen input (label-blind).
``objects``            LabelGuard build of Fit/Monitor/Locked objects + Fit-only
                       standardiser (Monitor/Locked ``.y`` unreachable).
``kernels``            R/D/U triplet caches for all zones + label-blind kernel audit.
``cpu-gates``          parameter/init/forward/gradient/invariance/reuse gates.
``freeze``             protocol_freeze.json (code + config + inputs + budgets).
``smoke``              post-freeze 20-update SSL + 1-epoch mini run (Fit labels only).
``ssl``                Fit-only masked atom/bond SSL for one seed (no labels).
``train``              R / D / U (BondRWComposer) or P3 (BJOC-Null), one seed.
``predict``            own-kernel predictions on Monitor and Locked.
``predict-cross``      frozen-model x frozen-kernel cross predictions (diagnostic).
``freeze-predictions`` hash-freeze every prediction file.
``score-holdout``      single read of Monitor + Locked labels, paired scoring.
``post-hoc``           frozen-model mechanism diagnostics (no label re-read).
``finalize``           evidence manifest + budget summary.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import math
import os
import subprocess
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
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import bond_random_walk_organization_v1 as bwo  # noqa: E402
import bond_relation_attribution_v1 as bra  # noqa: E402
import bond_rw_local_vs_multiscale_v1 as lm  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

REPO_ROOT = _CODE_DIR.parents[2]
TRACK = _CODE_DIR.parent
OUT = Path(
    os.environ.get(
        "BRA_OUT_DIR",
        str(TRACK / "results" / "bond_relation_attribution_v1"),
    )
)
CONFIG_PATH = TRACK / "configs" / "bond_relation_attribution_v1.yaml"
PROTOCOL_DOC = OUT / "PROTOCOL.md"
ZINC_ROOT = REPO_ROOT / "data/ZINC"
DATA_TRAIN = ZINC_ROOT / "subset/processed/train.pt"
CANON = REPO_ROOT / "tracks/ksvd/results/zinc_cssd_basis_reuse_v1/train_canonical_smiles.npz"
ENCODED_TRAIN = REPO_ROOT / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt"
STRUCT_CONTEXT_CODE = TRACK / "results/structure_context_support_v1/code"

PROTOCOL_ID = bra.PROTOCOL_ID
SPLIT_TAG = bra.SPLIT_TAG
SEEDS = bra.SEEDS
ARMS = bra.ARMS
RDU = bra.RDU_ARMS
ZONES = bra.ZONES

FROZEN_INPUT_HASHES = {
    "train_pt": "e0f1837b4a19714fcf3a5c6727a30c23f57c3398c8987cab2983214a0e76955b",
    "canonical_smiles": "04a590355fb7bba71e35e52bed60c974999b92af9e96e4ee99bed4afe42dcaff",
    "encoded_train": "10d764955ef1b040a317105be1dd3fbb2e81292b03b79e7d078cce08e18e5f1d",
}
GPU_BUDGET_CAP_HOURS = 4.0
KERNEL_AUDIT_N = 400
FORBIDDEN_OFFICIAL = ("valid", "test")

ARMS_JSON = {
    "R": {"kernel": "sym(P) lazy one-hop line graph", "parameters": 349985, "family": "bondrw"},
    "D": {"kernel": "K_D_ij = s_i s_j / sum(s)", "parameters": 349985, "family": "bondrw"},
    "U": {"kernel": "1/N_bond", "parameters": 349985, "family": "bondrw"},
    "P3": {"kernel": "none (BJOC-Null junction objects)", "parameters": 413505, "family": "bjoc"},
}


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=False, default=_json_default)
        fh.write("\n")


@contextlib.contextmanager
def _file_lock(path: Path | str):
    """Exclusive cross-process lock for read-modify-write JSON updates.

    Two seed jobs run concurrently on two GPUs and share the prediction
    manifest and the budget ledger; without the lock one job can silently
    drop the other's records/entries.
    """
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def read_json(path: Path | str) -> Any:
    with Path(path).open() as fh:
        return json.load(fh)


def file_sha256(path: Path | str) -> str:
    return bp.file_sha256(Path(path))


def rows_sha256(rows: Sequence[int]) -> str:
    return bp.rows_sha256(rows)


def array_sha256(a: np.ndarray) -> str:
    return bp.array_sha256(np.asarray(a))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def config() -> Mapping[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def config_sha256() -> str:
    return file_sha256(CONFIG_PATH)


ROUND_CODE = ("bond_relation_attribution_v1.py", "run_bond_relation_attribution_v1.py")
SCORING_CODE = ("bond_relation_attribution_v1.py", "run_bond_relation_attribution_v1.py")


def code_hashes() -> dict[str, str]:
    return {name: file_sha256(_CODE_DIR / name) for name in ROUND_CODE}


def scoring_code_hashes() -> dict[str, str]:
    return {name: file_sha256(_CODE_DIR / name) for name in SCORING_CODE}


# ---------------------------------------------------------------------------
# artifact paths
# ---------------------------------------------------------------------------


def split_path() -> Path:
    return OUT / "split_E.json"


def objects_path(zone: str) -> Path:
    return OUT / f"objects_{zone}_E.pt"


def prep_path() -> Path:
    return OUT / "prep_fit_E.json"


def kernels_path(zone: str) -> Path:
    return OUT / f"kernels_{zone}_E.pt"


def ssl_path(seed: int) -> Path:
    return OUT / f"ssl_E_s{int(seed)}.pt"


def ssl_json_path(seed: int) -> Path:
    return OUT / f"ssl_E_s{int(seed)}.json"


def soup_path(arm: str, seed: int) -> Path:
    return OUT / f"soup_{arm}_E_s{int(seed)}.pt"


def train_json_path(arm: str, seed: int) -> Path:
    return OUT / f"train_{arm}_E_s{int(seed)}.json"


def pred_path(zone: str, arm: str, seed: int) -> Path:
    return OUT / "predictions" / f"{zone}_{arm}_E_s{int(seed)}.npz"


def cross_pred_path(zone: str, model_arm: str, kernel: str, seed: int) -> Path:
    return OUT / "predictions" / f"cross_{zone}_m{model_arm}_k{kernel}_E_s{int(seed)}.npz"


def smoke_dir() -> Path:
    return OUT / "smoke"


def ledger_path() -> Path:
    return OUT / "gpu_budget_ledger.json"


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def load_split() -> Mapping[str, Any]:
    return read_json(split_path())


def load_zone_objects(zone: str) -> list[sro.MolObjects]:
    return list(torch.load(objects_path(zone), map_location="cpu", weights_only=False))


def load_zone_kernels(zone: str) -> Mapping[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    payload = torch.load(kernels_path(zone), map_location="cpu", weights_only=False)
    return {
        arm: [(np.asarray(s), np.asarray(d), np.asarray(w)) for s, d, w in payload[arm]]
        for arm in RDU
    }


def load_fit_y() -> np.ndarray:
    rows = [int(i) for i in load_split()["fit_rows"]]
    return bp.load_y_rows(ENCODED_TRAIN, rows)


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


def cmd_split(_args: argparse.Namespace) -> int:
    from rdkit import Chem

    scs = _structure_context_module()
    pairing = scs.pairing_verification()
    write_json(OUT / "pairing_verification.json", pairing)
    require(pairing["n_mismatch"] == 4, f"expected 4 SMILES/PyG mismatches, got {pairing['n_mismatch']}")
    require(
        tuple(pairing["mismatch_rows"]) == bra.EXCLUDED_ROWS,
        f"mismatch rows changed: {pairing['mismatch_rows']}",
    )
    require(pairing["reproduces_frozen_exclusion_list"], "exclusion list does not reproduce")
    rows = [int(r) for r in range(10_000) if int(r) not in set(bra.EXCLUDED_ROWS)]
    require(len(rows) == bra.N_USABLE, "usable row count changed")

    structs = scs.load_pyg_structure_only(rows)  # label-blind: y is discarded
    require([int(s["row"]) for s in structs] == rows, "structure rows out of order")
    graph_identity = {int(s["row"]): str(scs._pyg_rebuilt_smiles(s)) for s in structs}

    with np.load(CANON, allow_pickle=True) as blob:
        table = [str(s) for s in blob["smiles"]]
    table_identity = {r: Chem.CanonSmiles(table[r], useChiral=False) for r in rows}

    graph_part = identity_partition(rows, graph_identity)
    table_part = identity_partition(rows, table_identity)
    partition_equal = graph_part == table_part
    n_graph_groups = len(graph_part)
    n_table_groups = len(table_part)
    require(
        partition_equal,
        f"graph identity partition != committed-SMILES partition ({n_graph_groups} vs {n_table_groups})",
    )

    split = bra.build_identity_split(rows, [graph_identity[r] for r in rows])
    require(split["duplicate_groups_cross_sides"] == 0, "identity group crosses split zones")
    require(
        (len(split["fit_rows"]), len(split["monitor_rows"]), len(split["locked_rows"]))
        == (bra.FIT_TARGET, bra.MONITOR_TARGET, bra.LOCKED_TARGET),
        "split sizes do not match the pre-registered targets",
    )

    all_rows = split["fit_rows"] + split["monitor_rows"] + split["locked_rows"]
    require(sorted(all_rows) == rows, "split union does not equal the usable rows")
    gid = split["group_id"]
    group_ids = [int(gid[str(r)]) for r in sorted(all_rows)]
    sorted_rows = sorted(all_rows)
    identity_digest = hashlib.sha256(
        "\n".join(f"{r}\t{graph_identity[r]}" for r in sorted_rows).encode()
    ).hexdigest()
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": SPLIT_TAG,
        "salt": bra.SPLIT_SALT,
        "n_rows": len(sorted_rows),
        "n_groups": int(split["n_groups"]),
        "n_fit": len(split["fit_rows"]),
        "n_monitor": len(split["monitor_rows"]),
        "n_locked": len(split["locked_rows"]),
        "fit_rows": split["fit_rows"],
        "monitor_rows": split["monitor_rows"],
        "locked_rows": split["locked_rows"],
        "rows_sorted": sorted_rows,
        "group_ids_sorted": group_ids,
        "fit_sha256_rows": rows_sha256(split["fit_rows"]),
        "monitor_sha256_rows": rows_sha256(split["monitor_rows"]),
        "locked_sha256_rows": rows_sha256(split["locked_rows"]),
        "excluded_rows": list(bra.EXCLUDED_ROWS),
        "identity_source": "canonical constitution SMILES rebuilt from the official PyG graph",
        "identity_digest_sha256": identity_digest,
        "identity_partition_equal_to_committed_smiles": bool(partition_equal),
        "duplicate_groups": split["duplicate_groups"],
        "duplicate_groups_cross_sides": int(split["duplicate_groups_cross_sides"]),
        "label_information_used": False,
        "monitor_labels_read": False,
        "locked_labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(split_path(), payload)
    audit = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "pairing_verification": pairing,
        "n_rows_official_train": 10_000,
        "n_rows_excluded": len(bra.EXCLUDED_ROWS),
        "n_rows_usable": len(rows),
        "n_graph_groups": n_graph_groups,
        "n_committed_smiles_groups": n_table_groups,
        "identity_partition_equal": bool(partition_equal),
        "n_duplicate_identity_groups": len(split["duplicate_groups"]),
        "duplicate_identities_sha256": hashlib.sha256(
            json.dumps(split["duplicate_groups"], sort_keys=True).encode()
        ).hexdigest(),
        "split_sizes": {
            "fit": len(split["fit_rows"]),
            "monitor": len(split["monitor_rows"]),
            "locked": len(split["locked_rows"]),
        },
        "split_fingerprints": {
            "fit_sha256_rows": rows_sha256(split["fit_rows"]),
            "monitor_sha256_rows": rows_sha256(split["monitor_rows"]),
            "locked_sha256_rows": rows_sha256(split["locked_rows"]),
        },
        "identity_digest_sha256": identity_digest,
        "why_excluded": (
            "the committed canonical-SMILES identity contradicts the official PyG graph "
            "constitution for exactly these four rows; using the SMILES string as the "
            "identity would silently group/separate the wrong molecules.  All arms use "
            "the same 9,996-row usable set.  Historical 10k splits kept those rows, so "
            "absolute MAE is not comparable to historical split numbers."
        ),
        "labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "split_audit.json", audit)
    log(
        f"[split] usable={len(rows)} groups={n_graph_groups} "
        f"sizes={len(split['fit_rows'])}/{len(split['monitor_rows'])}/{len(split['locked_rows'])} "
        f"cross={split['duplicate_groups_cross_sides']} identity_partition_equal={partition_equal}"
    )
    return 0


# ---------------------------------------------------------------------------
# verify-inputs
# ---------------------------------------------------------------------------


def cmd_verify_inputs(_args: argparse.Namespace) -> int:
    cfg = config()
    require(cfg["protocol"] == PROTOCOL_ID, "config protocol id mismatch")
    require(cfg["status"] == "frozen", "config is not frozen")
    hashes = {
        "train_pt": file_sha256(DATA_TRAIN),
        "canonical_smiles": file_sha256(CANON),
        "encoded_train": file_sha256(ENCODED_TRAIN),
        "config": file_sha256(CONFIG_PATH),
        "protocol_doc": file_sha256(PROTOCOL_DOC),
    }
    for key, expected in FROZEN_INPUT_HASHES.items():
        require(hashes[key] == expected, f"frozen input drifted: {key} {hashes[key]} != {expected}")

    split = load_split()
    require(split["protocol"] == PROTOCOL_ID, "split protocol id mismatch")
    require(int(split["n_fit"]) == bra.FIT_TARGET, "fit size changed")
    require(int(split["n_monitor"]) == bra.MONITOR_TARGET, "monitor size changed")
    require(int(split["n_locked"]) == bra.LOCKED_TARGET, "locked size changed")
    require(split["rows_sorted"] == sorted(split["fit_rows"] + split["monitor_rows"] + split["locked_rows"]),
            "split row lists and sorted union disagree")
    require(len(set(split["rows_sorted"])) == len(split["rows_sorted"]), "duplicate rows in split")
    require(not set(bra.EXCLUDED_ROWS) & set(split["rows_sorted"]), "excluded row present in split")
    # group ids must not cross zones
    side = {int(r): "fit" for r in split["fit_rows"]}
    side.update({int(r): "monitor" for r in split["monitor_rows"]})
    side.update({int(r): "locked" for r in split["locked_rows"]})
    by_group: dict[int, set[str]] = {}
    for r, g in zip(split["rows_sorted"], split["group_ids_sorted"]):
        by_group.setdefault(int(g), set()).add(side[int(r)])
    require(all(len(v) == 1 for v in by_group.values()), "identity group crosses zones")
    require(
        split["fit_sha256_rows"] == rows_sha256(split["fit_rows"])
        and split["monitor_sha256_rows"] == rows_sha256(split["monitor_rows"])
        and split["locked_sha256_rows"] == rows_sha256(split["locked_rows"]),
        "split row fingerprints do not match the stored row lists",
    )
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "inputs": hashes,
        "split": {
            "n_rows": int(split["n_rows"]),
            "n_groups": int(split["n_groups"]),
            "n_fit": int(split["n_fit"]),
            "n_monitor": int(split["n_monitor"]),
            "n_locked": int(split["n_locked"]),
            "fit_sha256_rows": split["fit_sha256_rows"],
            "monitor_sha256_rows": split["monitor_sha256_rows"],
            "locked_sha256_rows": split["locked_sha256_rows"],
            "identity_digest_sha256": split["identity_digest_sha256"],
            "excluded_rows": split["excluded_rows"],
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "input_audit.json", payload)
    log(f"[verify-inputs] inputs ok; split rows={split['n_rows']} groups={split['n_groups']}")
    return 0


# ---------------------------------------------------------------------------
# objects + Fit-only standardiser
# ---------------------------------------------------------------------------


def cmd_objects(_args: argparse.Namespace) -> int:
    split, ds, guard = build_guard()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    objs = {}
    for zone in ZONES + ("fit",):
        rows = [int(i) for i in split[f"{zone}_rows"]]
        built = []
        for i in rows:
            d = guard[int(i)]
            built.append(sro.build_mol_objects(int(i), d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy()))
        require([o.mol_index for o in built] == rows, f"object row order mismatch in {zone}")
        torch.save(built, objects_path(zone))
        objs[zone] = built
    prep = bp.fit_standardiser(objs["fit"])
    prep["protocol"] = PROTOCOL_ID
    prep["split_tag"] = SPLIT_TAG
    write_json(prep_path(), prep)
    enc = torch.load(ENCODED_TRAIN, map_location="cpu", weights_only=False)
    check_rows = [int(i) for i in split["fit_rows"][:200]]
    max_diff = 0.0
    for i in check_rows:
        max_diff = max(max_diff, abs(float(ds[i].y.reshape(-1)[0]) - float(enc[int(i)].y.reshape(-1)[0])))
    del enc
    require(guard.access_attempts == [], f"guarded label access during object build: {guard.access_attempts[:3]}")
    require(max_diff <= 1e-9, f"Fit label alignment failed: {max_diff}")
    manifest = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "build_seconds": time.perf_counter() - started,
        "objects_fit_sha256": file_sha256(objects_path("fit")),
        "objects_monitor_sha256": file_sha256(objects_path("monitor")),
        "objects_locked_sha256": file_sha256(objects_path("locked")),
        "prep_fit_sha256": file_sha256(prep_path()),
        "alignment_rows_checked": len(check_rows),
        "alignment_rows_scope": "Fit rows only",
        "alignment_max_abs_diff": float(max_diff),
        "guarded_rows": len(guard.forbidden_rows),
        "forbidden_label_access_attempts": len(guard.access_attempts),
        "monitor_labels_read": False,
        "locked_labels_read": False,
    }
    write_json(OUT / "objects_manifest_E.json", manifest)
    log(
        f"[objects] {manifest['build_seconds']:.1f}s fit={len(objs['fit'])} monitor={len(objs['monitor'])} "
        f"locked={len(objs['locked'])} alignment={max_diff:.2e} guarded_attempts={len(guard.access_attempts)}"
    )
    return 0


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


def cmd_kernels(_args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"protocol": PROTOCOL_ID, "git_commit": git_commit(), "zones": {}}
    for zone in ("fit", "monitor", "locked"):
        objs = load_zone_objects(zone)
        caches = {arm: [] for arm in RDU}
        start = time.perf_counter()
        for o in objs:
            triplets = {arm: bra.triplets_for(arm, o) for arm in RDU}
            n_b = int(np.asarray(o.bond_index).reshape(-1, 2).shape[0])
            for arm in RDU:
                s_a, d_a, w_a = triplets[arm]
                require(s_a.shape == d_a.shape == w_a.shape, f"{zone}/{arm}: triplet shape mismatch")
                require(int(s_a.shape[0]) == n_b * (n_b + 1) // 2, f"{zone}/{arm}: unexpected triplet count")
            for arm in ("D", "U"):
                require(np.array_equal(triplets["R"][0], triplets[arm][0]) and np.array_equal(triplets["R"][1], triplets[arm][1]),
                        f"{zone}/{arm}: pair layout differs from R")
            for arm in RDU:
                caches[arm].append(triplets[arm])
        torch.save(caches, kernels_path(zone))
        counts = {arm: int(sum(int(s.shape[0]) for s, _, _ in caches[arm])) for arm in RDU}
        manifest["zones"][zone] = {
            "n_molecules": len(objs),
            "n_triplets_total": counts,
            "build_seconds": time.perf_counter() - start,
            "file_sha256": file_sha256(kernels_path(zone)),
        }
        log(f"[kernels {zone}] n={len(objs)} triplets={counts['R']} {manifest['zones'][zone]['build_seconds']:.1f}s")

    # label-blind audit on a fixed subset of each zone
    per_mol = []
    for zone in ("fit", "monitor", "locked"):
        objs = load_zone_objects(zone)
        take = objs[: min(KERNEL_AUDIT_N, len(objs))]
        for o in take:
            rec = bra.relation_audit_molecule(o)
            rec["zone"] = zone
            per_mol.append(rec)
    summary = bra.summarize_relation_audit(per_mol)
    per_zone = {}
    for zone in ("fit", "monitor", "locked"):
        sub = [r for r in per_mol if r["zone"] == zone]
        per_zone[zone] = bra.summarize_relation_audit(sub)
    kernel_payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "audit_molecules_per_zone": KERNEL_AUDIT_N,
        "definitions": {
            "R": "K^R = sym(P), P = (I + D^-1 A_L)/2, zero-degree rows one-hot",
            "D": "s_i = sum_j K^R_ij ; K^D_ij = s_i s_j / sum_k s_k",
            "U": "K^U_ef = 1/N_bond",
            "triplet_convention": "unordered e<=f of the symmetric matrix; diagonal weight halved in pair_aggregate",
        },
        "checks_note": (
            "R/D/U enumerate exactly the same unordered pair list (n(n+1)/2 per molecule), "
            "so pair-MLP FLOPs and batch composition are identical across arms; only the "
            "weights differ.  Zero-weight terms still instantiate the same phi computation."
        ),
        "aggregate": summary["aggregate"],
        "conclusions": summary["conclusions"],
        "per_zone": {z: {"aggregate": per_zone[z]["aggregate"], "conclusions": per_zone[z]["conclusions"]} for z in per_zone},
        "per_molecule": per_mol,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "kernel_audit.json", kernel_payload)
    write_json(OUT / "kernels_manifest_E.json", manifest)
    require(summary["conclusions"]["row_sums_match"], "D row sums do not match R")
    require(summary["conclusions"]["total_weight_matches"], "D total weight does not match R")
    require(summary["conclusions"]["D_is_rank_one"], "D is not rank one")
    require(summary["conclusions"]["D_differs_from_R"], "D degenerates to R")
    require(summary["conclusions"]["D_differs_from_U"], "D degenerates to U")
    require(summary["conclusions"]["bond_permutation_invariant"], "D is not bond-permutation invariant")
    log(f"[kernels] audit ok: {json.dumps(summary['conclusions'])}")
    return 0


# ---------------------------------------------------------------------------
# CPU gates
# ---------------------------------------------------------------------------


def _synthetic_objects() -> list[sro.MolObjects]:
    def mol(idx: int, atom_types, bonds):
        n = len(atom_types)
        x = np.asarray(atom_types, dtype=np.int64).reshape(n, 1)
        ei, ea = [], []
        for u, v, bt in bonds:
            ei.extend([(u, v), (v, u)])
            ea.extend([bt, bt])
        return sro.build_mol_objects(idx, x, np.asarray(ei, dtype=np.int64).T, np.asarray(ea, dtype=np.int64))

    def chain(n, idx=0):
        return mol(idx, range(n), [(i, i + 1, 1) for i in range(n - 1)])

    def ring(n, idx=0):
        return mol(idx, range(n), [(i, (i + 1) % n, 1) for i in range(n)])

    def branched(idx=0):
        return mol(idx, range(6), [(0, 1, 1), (1, 2, 1), (1, 3, 1), (3, 4, 1), (1, 5, 1)])

    return [mol(0, [0, 1], [(0, 1, 1)]), chain(4, 1), chain(7, 2), ring(5, 3), ring(8, 4), branched(5)]


def _batch(obj_list, triplets, device="cpu"):
    return bwo.move_batch(bwo.batch_rw_objects(obj_list, triplets), torch.device(device))


def _forward(model, obj, triplet) -> float:
    model.eval()
    with torch.no_grad():
        return float(model(_batch([obj], [triplet])))


def _relabel(o: sro.MolObjects, perm: np.ndarray) -> sro.MolObjects:
    inv = np.argsort(perm)
    return sro.MolObjects(
        mol_index=o.mol_index,
        smiles=o.smiles,
        n_atoms=o.n_atoms,
        atom_feat=o.atom_feat[perm],
        bond_index=inv[o.bond_index],
        bond_type=o.bond_type,
        bond_feat=o.bond_feat,
        junc_index=o.junc_index,
        junc_bond=o.junc_bond,
        junc_feat=o.junc_feat,
        global_feat=o.global_feat,
    )


def _bond_permute(o: sro.MolObjects, perm: np.ndarray) -> sro.MolObjects:
    """Permute bond slot order (endpoint ids untouched)."""
    new_bond_index = o.bond_index[perm]
    o2 = sro.MolObjects(
        mol_index=o.mol_index,
        smiles=o.smiles,
        n_atoms=o.n_atoms,
        atom_feat=o.atom_feat,
        bond_index=np.asarray(new_bond_index, dtype=np.int64),
        bond_type=o.bond_type[perm],
        bond_feat=o.bond_feat[perm],
        junc_index=o.junc_index,
        junc_bond=o.junc_bond,
        junc_feat=o.junc_feat,
        global_feat=o.global_feat,
    )
    # slot i of the new object corresponds to old slot perm[i]; K_new = K_old[perm][:, perm]
    return o2


def cmd_cpu_gates(_args: argparse.Namespace) -> int:
    split = load_split()
    norm = load_norm()
    out: dict[str, Any] = {"protocol": PROTOCOL_ID, "git_commit": git_commit(), "checks": {}}

    # --- reuse hashes -------------------------------------------------------
    reuse = {}
    for name, expected in bra.REUSE_HASHES.items():
        got = file_sha256(_CODE_DIR / name)
        reuse[name] = {"expected": expected, "actual": got, "match": got == expected}
    out["reuse_audit"] = reuse
    require(all(v["match"] for v in reuse.values()), "historical module hash drifted; refusing to continue")

    # --- kernel math on synthetic + real molecules --------------------------
    synth = _synthetic_objects()
    synth_audit = bra.summarize_relation_audit([bra.relation_audit_molecule(o) for o in synth])
    out["synthetic_kernel_audit"] = synth_audit
    for key, value in synth_audit["conclusions"].items():
        require(bool(value), f"synthetic kernel conclusion failed: {key}")

    real_objs = load_zone_objects("fit")[:64]
    real_audit = bra.summarize_relation_audit([bra.relation_audit_molecule(o) for o in real_objs])
    out["real_kernel_audit_64"] = real_audit
    for key, value in real_audit["conclusions"].items():
        require(bool(value), f"real kernel conclusion failed: {key}")

    # --- historical R1 triplet builder is the same object ------------------
    for o in real_objs[:16]:
        a = bra.r_triplets_from_objects(o)
        b = lm.k1_triplets_from_objects(o)
        require(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), "R triplet layout differs from historical R1")
        require(np.allclose(a[2], b[2], atol=0.0, rtol=0.0), "R triplet weights differ from historical R1")
    out["checks"]["r_triplets_identical_to_historical_r1"] = True

    # --- parameter / init parity -------------------------------------------
    init = {}
    for seed in SEEDS:
        m_r = bwo.build_model(seed, norm, "cpu")
        m_d = bwo.build_model(seed, norm, "cpu")
        m_u = bwo.build_model(seed, norm, "cpu")
        p3 = bp.build_bjoc_null(seed, norm, "cpu")
        hashes = [bwo.state_dict_hash(m) for m in (m_r, m_d, m_u)]
        init[str(seed)] = {
            "n_params_bondrw": bwo.parameter_count(m_r),
            "n_params_p3": int(sum(p.numel() for p in p3.parameters())),
            "bondrw_init_sha256": hashes,
            "bondrw_init_all_equal": len(set(hashes)) == 1,
            "p3_init_sha256": src.state_dict_hash(p3),
            "forbidden_modules_bondrw": bwo.forbidden_module_scan(m_r),
            "forbidden_modules_p3": src.forbidden_module_scan(p3),
        }
        require(init[str(seed)]["n_params_bondrw"] == bra.EXPECTED_PARAMETERS, "BondRW parameter count changed")
        require(init[str(seed)]["n_params_p3"] == bra.P3_EXPECTED_PARAMETERS, "P3 parameter count changed")
        require(init[str(seed)]["bondrw_init_all_equal"], "R/D/U init hashes differ")
        require(init[str(seed)]["forbidden_modules_bondrw"] == [], "forbidden module in BondRW model")
        require(init[str(seed)]["forbidden_modules_p3"] == [], "forbidden module in P3 model")
    out["init_parity"] = init

    # --- batch order / updates ---------------------------------------------
    n_fit = int(split["n_fit"])
    batch_info = {}
    for seed in SEEDS:
        h = bp.batch_order_hash(seed, n_fit, bp.BATCH_SIZE, int(bp.RECIPES["R3"]["epochs"]))
        batch_info[str(seed)] = {
            "n_fit": n_fit,
            "steps_per_epoch": math.ceil(n_fit / bp.BATCH_SIZE),
            "updates": math.ceil(n_fit / bp.BATCH_SIZE) * int(bp.RECIPES["R3"]["epochs"]),
            "batch_order_hash": h,
            "same_for_all_four_arms": True,
        }
        require(batch_info[str(seed)]["updates"] == 22680, "update count changed")
    out["batch"] = batch_info

    # --- forward differences at init (kernel consumed) ----------------------
    # The reader averages the pair aggregate, so the naive prediction difference
    # at random init is small; the decisive check is that the kernel reaches the
    # pair aggregate ``r`` itself (recorded below) and that predictions are not
    # bit-identical.
    objs = load_zone_objects("fit")[:8]
    kernels = load_zone_kernels("fit")
    m = bwo.build_model(0, norm, "cpu")
    preds, rblocks, rscale = {}, {}, {}
    for arm in RDU:
        trip = kernels[arm][:8]
        batch = _batch(objs, trip)
        m.eval()
        with torch.no_grad():
            feats = m.forward_features(batch)
            preds[arm] = m(batch).detach().numpy()
            rblocks[arm] = feats["r"].detach().numpy()
        rscale[arm] = float(np.abs(rblocks[arm]).mean())
    out["init_forward"] = {
        "max_abs_pred_R_minus_D": float(np.abs(preds["R"] - preds["D"]).max()),
        "max_abs_pred_R_minus_U": float(np.abs(preds["R"] - preds["U"]).max()),
        "max_abs_pred_D_minus_U": float(np.abs(preds["D"] - preds["U"]).max()),
        "max_abs_r_R_minus_D": float(np.abs(rblocks["R"] - rblocks["D"]).max()),
        "max_abs_r_D_minus_U": float(np.abs(rblocks["D"] - rblocks["U"]).max()),
        "mean_abs_r_by_arm": rscale,
    }
    require(out["init_forward"]["max_abs_r_R_minus_D"] > 1e-5, "R and D give identical pair aggregates")
    require(out["init_forward"]["max_abs_r_D_minus_U"] > 1e-5, "D and U give identical pair aggregates")
    require(out["init_forward"]["max_abs_pred_R_minus_D"] > 1e-7, "R and D give identical initial predictions")
    require(out["init_forward"]["max_abs_pred_D_minus_U"] > 1e-7, "D and U give identical initial predictions")

    # --- invariances / batch composition / gradients -----------------------
    inv = {}
    rng = np.random.default_rng(7)
    for arm in RDU:
        vals = {"relabel": 0.0, "bond_perm": 0.0, "batch_composition": 0.0}
        for o, trip in zip(objs, kernels[arm][:8]):
            p0 = _forward(m, o, trip)
            perm_atom = rng.permutation(o.n_atoms)
            p1 = _forward(m, _relabel(o, perm_atom), trip)
            vals["relabel"] = max(vals["relabel"], abs(p0 - p1))
            # bond permutation: rebuild object and conjugate kernel
            perm_bond = rng.permutation(int(o.bond_index.shape[0]))
            o2 = _bond_permute(o, perm_bond)
            K = bra.r_matrix(o) if arm == "R" else (bra.d_matrix(o)[0] if arm == "D" else bra.u_matrix(int(o.bond_index.shape[0])))
            K2 = K[np.ix_(perm_bond, perm_bond)]
            trip2 = bra.kernel_triplets(K2)
            p2 = _forward(m, o2, trip2)
            vals["bond_perm"] = max(vals["bond_perm"], abs(p0 - p2))
        batch = _batch(objs, kernels[arm][:8])
        with torch.no_grad():
            bp_all = m(batch).detach().numpy()
        singles = np.asarray([_forward(m, o, t) for o, t in zip(objs, kernels[arm][:8])])
        vals["batch_composition"] = float(np.abs(bp_all - singles).max())
        for key, value in vals.items():
            require(value <= 1e-6, f"{arm} invariance {key} failed: {value}")
        inv[arm] = vals
    out["invariances"] = inv

    # --- gradient liveness --------------------------------------------------
    grads = {}
    for arm in RDU:
        mm = bwo.build_model(0, norm, "cpu")
        mm.train()
        batch = _batch(objs, kernels[arm][:8])
        yb = torch.zeros(8)
        loss = torch.nn.functional.l1_loss(mm(batch).view(-1), yb)
        loss.backward()
        grads[arm] = {}
        for name, module in (
            ("atom_encoder", mm.atom_encoder),
            ("bond_encoder", mm.bond_encoder),
            ("pair_mlp", mm.pair_mlp),
            ("group_composer", mm.group_composer),
            ("reader", mm.reader),
        ):
            norms = [float(p.grad.norm()) for p in module.parameters() if p.grad is not None]
            grads[arm][name] = float(np.sqrt(np.sum(np.square(norms)))) if norms else 0.0
        require(all(v > 0 for v in grads[arm].values()), f"{arm}: dead gradient branch {grads[arm]}")
    out["gradient_liveness"] = grads

    # --- save/load roundtrip ------------------------------------------------
    state = {k: v.detach().clone() for k, v in m.state_dict().items()}
    m2 = bwo.build_model(0, norm, "cpu")
    m2.load_state_dict(state)
    with torch.no_grad():
        a = m(_batch(objs, kernels["R"][:8]))
        b = m2(_batch(objs, kernels["R"][:8]))
    out["save_load_max_abs"] = float((a - b).abs().max())
    require(out["save_load_max_abs"] == 0.0, "save/load changed the forward pass")

    # --- micro-smoke determinism on real Fit rows --------------------------
    y = load_fit_y()[:32]
    small = load_zone_objects("fit")[:32]
    smoke = {}
    for arm in RDU:
        trips = kernels[arm][:32]
        ma, ia = bwo.train_supervised_rw(
            arm=arm, seed=0, train_objs=small, train_triplets=trips, y_fit=y,
            norm=norm, device=torch.device("cpu"), ssl_state=None, smoke_epochs=1,
            log_fn=lambda _m: None,
        )
        mb, ib = bwo.train_supervised_rw(
            arm=arm, seed=0, train_objs=small, train_triplets=trips, y_fit=y,
            norm=norm, device=torch.device("cpu"), ssl_state=None, smoke_epochs=1,
            log_fn=lambda _m: None,
        )
        sa = {k: v.detach().cpu().float() for k, v in ma.state_dict().items()}
        sb = {k: v.detach().cpu().float() for k, v in mb.state_dict().items()}
        max_soup_diff = float(max(float((sa[k] - sb[k]).abs().max()) for k in sa))
        smoke[arm] = {
            "soup_identical_bitwise": bwo.state_dict_hash(ma) == bwo.state_dict_hash(mb),
            "soup_max_abs_diff": max_soup_diff,
            "soup_within_float32_noise_floor": bool(max_soup_diff <= 1e-6),
            "batch_order_matches": ia["batch_order"]["full_matches"] and ib["batch_order"]["full_matches"],
            "init_equal_to_reference": ia["init_full_sha256"] == init["0"]["bondrw_init_sha256"][0],
            "finite": bool(np.isfinite(ia["soup_fit_mae"])),
        }
        # Amendment-1 (pre-GPU, no labels): remote CPU float32 reductions are
        # run-to-run non-deterministic at ~1e-7 (documented in earlier rounds),
        # so the gate is a noise floor, not bitwise equality.
        require(
            smoke[arm]["soup_within_float32_noise_floor"]
            and smoke[arm]["batch_order_matches"]
            and smoke[arm]["init_equal_to_reference"]
            and smoke[arm]["finite"],
            f"{arm}: micro-smoke gate failed {smoke[arm]}",
        )
    out["micro_smoke"] = smoke

    # --- LabelGuard positive control ---------------------------------------
    out["label_guard_selfcheck"] = ab.label_guard_selfcheck()
    require(all(out["label_guard_selfcheck"].values()), "LabelGuard self-check failed")

    # --- no official valid/test path ---------------------------------------
    import re as _re

    official_pattern = _re.compile(r"split\s*=\s*[\"'](?:valid|test)[\"']")
    scan = []
    for name in ROUND_CODE:
        text = (_CODE_DIR / name).read_text(encoding="utf-8")
        for match in official_pattern.finditer(text):
            scan.append({"file": name, "match": match.group(0)})
    out["official_path_scan"] = {"hits": scan, "clean": scan == []}
    require(scan == [], f"official valid/test load token found: {scan}")

    write_json(OUT / "reuse_audit.json", {"protocol": PROTOCOL_ID, "hashes": reuse})
    write_json(OUT / "cpu_gates.json", out)
    log(
        f"[cpu-gates] reuse=OK synth={synth_audit['conclusions']} real={real_audit['conclusions']} "
        f"init_forward={out['init_forward']} grads=OK"
    )
    return 0


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------


def cmd_freeze(_args: argparse.Namespace) -> int:
    for path in (
        OUT / "input_audit.json",
        OUT / "split_audit.json",
        OUT / "objects_manifest_E.json",
        OUT / "kernels_manifest_E.json",
        OUT / "kernel_audit.json",
        OUT / "cpu_gates.json",
        OUT / "reuse_audit.json",
    ):
        require(path.exists(), f"freeze dependency missing: {path.name}")
    split = load_split()
    norm = load_norm()
    init_per_seed = {}
    for seed in SEEDS:
        m = bwo.build_model(seed, norm, "cpu")
        p3 = bp.build_bjoc_null(seed, norm, "cpu")
        init_per_seed[str(seed)] = {
            "bondrw_init_sha256": bwo.state_dict_hash(m),
            "p3_init_sha256": src.state_dict_hash(p3),
        }
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "frozen_utc": now_utc(),
        "config_sha256": config_sha256(),
        "protocol_doc_sha256": file_sha256(PROTOCOL_DOC),
        "code_hashes": code_hashes(),
        "scoring_code_sha256": scoring_code_hashes(),
        "inputs": read_json(OUT / "input_audit.json")["inputs"],
        "split": {
            key: split[key]
            for key in (
                "n_rows",
                "n_groups",
                "n_fit",
                "n_monitor",
                "n_locked",
                "fit_sha256_rows",
                "monitor_sha256_rows",
                "locked_sha256_rows",
                "identity_digest_sha256",
            )
        },
        "kernel_files": {
            zone: file_sha256(kernels_path(zone)) for zone in ("fit", "monitor", "locked")
        },
        "objects": {
            "fit": file_sha256(objects_path("fit")),
            "monitor": file_sha256(objects_path("monitor")),
            "locked": file_sha256(objects_path("locked")),
            "prep_fit": file_sha256(prep_path()),
        },
        "expected_init_per_seed": init_per_seed,
        "arms": ARMS_JSON,
        "recipe": {
            "name": "R3",
            "epochs": int(bp.RECIPES["R3"]["epochs"]),
            "updates": math.ceil(int(split["n_fit"]) / bp.BATCH_SIZE) * int(bp.RECIPES["R3"]["epochs"]),
            "loss": "raw-y L1",
            "soup": "last-5 epoch-state parameter mean",
            "shuffle_offset": bp.TRAIN_SHUFFLE_OFFSET,
        },
        "ssl": {"updates": int(bp.PRETRAIN_UPDATES), "mode": "atom_bond", "labels_read": False},
        "gpu_budget_cap_hours": GPU_BUDGET_CAP_HOURS,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "protocol_freeze.json", payload)
    log(f"[freeze] code={payload['code_hashes']} config={payload['config_sha256'][:12]}")
    return 0


def verify_freeze(require_exact_code: bool = True) -> Mapping[str, Any]:
    path = OUT / "protocol_freeze.json"
    require(path.exists(), "protocol_freeze.json missing; run freeze first")
    payload = read_json(path)
    require(payload["protocol"] == PROTOCOL_ID, "freeze protocol id mismatch")
    if require_exact_code:
        for name, sha in payload["code_hashes"].items():
            require(file_sha256(_CODE_DIR / name) == sha, f"code drift detected in {name}")
        require(config_sha256() == payload["config_sha256"], "config drifted from freeze")
        require(file_sha256(PROTOCOL_DOC) == payload["protocol_doc_sha256"], "protocol document drifted from freeze")
    for zone, sha in payload["kernel_files"].items():
        require(file_sha256(kernels_path(zone)) == sha, f"kernel cache drifted: {zone}")
    return payload


# ---------------------------------------------------------------------------
# SSL
# ---------------------------------------------------------------------------


def cmd_ssl(args: argparse.Namespace) -> int:
    payload = verify_freeze(require_exact_code=True)
    seed = int(args.seed)
    smoke_updates = int(args.smoke_updates) if args.smoke_updates else None
    if smoke_updates is None and (ssl_path(seed).exists() or ssl_json_path(seed).exists()):
        raise SystemExit(f"SSL for seed {seed} already exists; retraining is not allowed")
    split, ds, guard = build_guard()
    objs = load_zone_objects("fit")
    require([o.mol_index for o in objs] == [int(i) for i in split["fit_rows"]], "fit objects do not match split")
    log(f"[ssl s{seed}] device={args.device} n={len(objs)} updates={smoke_updates or payload['ssl']['updates']}")
    res = bp.pretrain_masked(
        seed=seed,
        fit_objs=objs,
        fit_rows=[int(i) for i in split["fit_rows"]],
        raw_dataset=guard,
        device=torch.device(args.device),
        mode=payload["ssl"]["mode"],
        smoke_updates=smoke_updates,
        log_fn=log,
    )
    require(guard.access_attempts == [], f"guarded label access during SSL: {guard.access_attempts[:3]}")
    require(res.get("labels_read") is False, "SSL unexpectedly read labels")
    enc = res["encoder_state"]
    out_pt = ssl_path(seed)
    if smoke_updates is not None:
        out_pt = smoke_dir() / f"ssl_E_s{seed}_smoke.pt"
    torch.save({group: {k: v.cpu() for k, v in state.items()} for group, state in enc.items()}, out_pt)
    atom_bond = {
        **{f"atom:{k}": v for k, v in enc["atom_encoder"].items()},
        **{f"bond:{k}": v for k, v in enc["bond_encoder"].items()},
    }
    res.pop("encoder_state", None)
    res.pop("atom_head_state", None)
    res.update(
        {
            "protocol": PROTOCOL_ID,
            "split_tag": SPLIT_TAG,
            "seed": seed,
            "smoke_updates": smoke_updates,
            "encoder_path": str(out_pt.relative_to(REPO_ROOT)),
            "encoder_file_sha256": file_sha256(out_pt),
            "atom_bond_group_sha256": bp.state_group_sha256(atom_bond),
            "full_encoder_group_sha256": bp.state_group_sha256(
                {
                    **{f"atom:{k}": v for k, v in enc["atom_encoder"].items()},
                    **{f"bond:{k}": v for k, v in enc["bond_encoder"].items()},
                    **{f"junc:{k}": v for k, v in enc["junction_encoder"].items()},
                }
            ),
            "forbidden_label_access_attempts": len(guard.access_attempts),
            "official_valid_loaded": False,
            "official_test_loaded": False,
        }
    )
    # internal consistency: the recomputed group hash must equal the trainer's
    # canonical encoder hash (established convention in earlier rounds).
    require(
        res["full_encoder_group_sha256"] == res["encoder_final_sha256"],
        "SSL full-encoder hash convention changed",
    )
    if smoke_updates is None:
        write_json(ssl_json_path(seed), res)
        _append_ledger({"phase": "ssl", "arm": "SSL", "seed": seed, "wall_seconds": res["wall_seconds"], "device": res["device"]})
    else:
        write_json(smoke_dir() / f"ssl_E_s{seed}_smoke.json", res)
    log(f"[ssl s{seed}] wall={res['wall_seconds']:.0f}s atom_acc={res.get('final_acc_atom'):.4f}")
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    arm, seed = args.arm, int(args.seed)
    require(arm in ARMS, f"unknown arm {arm!r}")
    smoke = bool(args.smoke)
    if smoke:
        require(1 <= int(args.epochs or 0) <= 5, "--smoke requires --epochs in [1,5]")
    split = load_split()
    norm = load_norm()
    objs = load_zone_objects("fit")
    y = load_fit_y()
    require(len(objs) == len(y) == int(split["n_fit"]), "fit alignment changed")
    device = torch.device(args.device)
    ssl_state = None
    if smoke:
        sp = smoke_dir() / f"ssl_E_s{seed}_smoke.pt"
        require(sp.exists(), "run the smoke SSL first")
        ssl_state = torch.load(sp, map_location="cpu", weights_only=False)
    else:
        require(ssl_path(seed).exists(), f"missing SSL for seed {seed}")
        ssl_state = load_ssl(seed)

    if arm in RDU:
        kernels = load_zone_kernels("fit")
        log(f"[train {arm} s{seed}] family=bondrw n={len(objs)} updates={math.ceil(len(objs)/128)*int(args.epochs or 360)}")
        model, info = bwo.train_supervised_rw(
            arm=arm,
            seed=seed,
            train_objs=objs,
            train_triplets=kernels[arm],
            y_fit=y,
            norm=norm,
            device=device,
            ssl_state=ssl_state,
            smoke_epochs=int(args.epochs) if smoke else None,
            log_fn=log,
        )
        info["family"] = "bondrw"
        info["kernel"] = ARMS_JSON[arm]["kernel"]
        info["trainer"] = "bond_random_walk_organization_v1.train_supervised_rw (unchanged)"
        soup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        log(f"[train P3 s{seed}] family=bjoc n={len(objs)}")
        info = bp.train_supervised(
            arm="P3",
            seed=seed,
            recipe_key="R3",
            init="ssl",
            train_objs=objs,
            rows=[int(i) for i in split["fit_rows"]],
            y_fit=y,
            norm=norm,
            device=device,
            ssl_state=ssl_state,
            smoke_epochs=int(args.epochs) if smoke else None,
            log_fn=log,
        )
        info["family"] = "bjoc"
        info["kernel"] = ARMS_JSON["P3"]["kernel"]
        info["trainer"] = "bjoc_perfmech_long_v1.train_supervised (unchanged)"
        soup = info.pop("soup_state")

    info["protocol"] = PROTOCOL_ID
    info["split_tag"] = SPLIT_TAG
    info["smoke"] = smoke
    info["fit_sha256_rows"] = split["fit_sha256_rows"]
    info["ssl_source"] = "smoke" if smoke else ssl_json_path(seed).name
    info["official_valid_loaded"] = False
    info["official_test_loaded"] = False
    require(bool(info["batch_order"]["full_matches"]), "batch order hash self-check failed")
    if arm in RDU:
        require(info["parameter_audit"]["actual_parameters"] == bra.EXPECTED_PARAMETERS, "BondRW parameter audit failed")
        if not smoke:
            require(
                info["init_full_sha256"]
                == read_json(OUT / "protocol_freeze.json")["expected_init_per_seed"][str(seed)]["bondrw_init_sha256"],
                "BondRW init hash does not match the frozen expectation",
            )
    else:
        require(info["parameter_audit"]["actual_parameters"] == bra.P3_EXPECTED_PARAMETERS, "P3 parameter audit failed")

    if smoke:
        torch.save(soup, smoke_dir() / f"soup_{arm}_E_s{seed}_smoke.pt")
        write_json(smoke_dir() / f"train_{arm}_E_s{seed}_smoke.json", info)
        log(f"[train {arm} s{seed}] SMOKE only epochs={info['epochs']}")
        return 0
    torch.save(soup, soup_path(arm, seed))
    write_json(train_json_path(arm, seed), info)
    _append_ledger(
        {
            "phase": "train",
            "arm": arm,
            "seed": seed,
            "wall_seconds": info["wall_seconds"],
            "device": info["device"],
            "epochs": info["epochs"],
        }
    )
    log(f"[train {arm} s{seed}] soup_fit_mae={info['soup_fit_mae']:.6f} wall={info['wall_seconds']:.0f}s")
    return 0


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def _load_model_for(arm: str, seed: int, smoke: bool, device: torch.device):
    norm = load_norm()
    if arm in RDU:
        model = bwo.build_model(seed, norm, device)
        path = smoke_dir() / f"soup_{arm}_E_s{seed}_smoke.pt" if smoke else soup_path(arm, seed)
        require(path.exists(), f"missing soup {path}")
        model.load_state_dict({k: v.to(device) for k, v in torch.load(path, map_location="cpu").items()})
        return model, bwo.predict_model
    model = bp.build_bjoc_null(seed, norm, device)
    path = smoke_dir() / f"soup_P3_E_s{seed}_smoke.pt" if smoke else soup_path("P3", seed)
    require(path.exists(), f"missing soup {path}")
    model.load_state_dict({k: v.to(device) for k, v in torch.load(path, map_location="cpu").items()})
    return model, bp.predict_model


def _save_prediction(path: Path, pred: np.ndarray, meta: Mapping[str, Any], manifest_name: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, pred=pred.astype(np.float32), **{k: v for k, v in meta.items() if isinstance(v, (str, int, float, np.ndarray))})
    record = {
        **{k: v for k, v in meta.items()},
        "n_molecules": int(len(pred)),
        "pred_sha256": array_sha256(pred),
        "file_sha256": file_sha256(path),
        "labels_used": "none",
    }
    manifest_path = OUT / manifest_name
    with _file_lock(manifest_path):
        manifest = read_json(manifest_path) if manifest_path.exists() else {"protocol": PROTOCOL_ID, "records": {}}
        manifest["records"][path.name] = record
        write_json(manifest_path, manifest)
    return record


def cmd_predict(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    arm, seed = args.arm, int(args.seed)
    require(arm in ARMS, f"unknown arm {arm!r}")
    smoke = bool(args.smoke)
    device = torch.device(args.device)
    model, predictor = _load_model_for(arm, seed, smoke, device)
    if arm in RDU:
        zone_trip = {zone: load_zone_kernels(zone)[arm] for zone in ZONES}
    else:
        zone_trip = None
    manifest_name = "smoke/prediction_manifest_smoke.json" if smoke else "prediction_manifest_E.json"
    if smoke:
        (OUT / "smoke").mkdir(parents=True, exist_ok=True)
    for zone in ZONES:
        objs = load_zone_objects(zone)
        if arm in RDU:
            pred = bwo.predict_model(model, objs, zone_trip[zone], device)
        else:
            pred = predictor(model, objs, device)
        path = (smoke_dir() / f"{zone}_{arm}_E_s{seed}_smoke.npz") if smoke else pred_path(zone, arm, seed)
        rec = _save_prediction(
            path,
            pred,
            {
                "protocol": PROTOCOL_ID,
                "zone": zone,
                "arm": arm,
                "seed": seed,
                "kernel": ARMS_JSON[arm]["kernel"],
                "smoke": bool(smoke),
                "device": str(device),
            },
            manifest_name,
        )
        log(f"[predict {arm} s{seed} {zone}] n={len(pred)} sha={rec['pred_sha256'][:12]}")
    return 0


def cmd_predict_cross(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    model_arm, kernel, seed = args.model, args.kernel, int(args.seed)
    require(model_arm in RDU and kernel in RDU, "cross predictions only exist for R/D/U")
    require(model_arm != kernel, "diagonal cross predictions duplicate the own-kernel files")
    device = torch.device(args.device)
    model, _ = _load_model_for(model_arm, seed, False, device)
    for zone in ZONES:
        objs = load_zone_objects(zone)
        kernels = load_zone_kernels(zone)[kernel]
        pred = bwo.predict_model(model, objs, kernels, device)
        rec = _save_prediction(
            cross_pred_path(zone, model_arm, kernel, seed),
            pred,
            {
                "protocol": PROTOCOL_ID,
                "zone": zone,
                "model_arm": model_arm,
                "kernel": kernel,
                "seed": seed,
                "diagnostic": "frozen-model x frozen-kernel forward",
                "device": str(device),
            },
            "prediction_manifest_E.json",
        )
        log(f"[predict-cross {zone} m{model_arm}/k{kernel} s{seed}] n={len(pred)} sha={rec['pred_sha256'][:12]}")
    return 0


# ---------------------------------------------------------------------------
# freeze predictions
# ---------------------------------------------------------------------------


def _expected_prediction_keys() -> list[str]:
    keys = []
    for zone in ZONES:
        for arm in ARMS:
            for seed in SEEDS:
                keys.append(pred_path(zone, arm, seed).name)
        for model_arm in RDU:
            for kernel in RDU:
                if model_arm == kernel:
                    continue
                for seed in SEEDS:
                    keys.append(cross_pred_path(zone, model_arm, kernel, seed).name)
    return keys


def cmd_freeze_predictions(_args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    manifest_path = OUT / "prediction_manifest_E.json"
    require(manifest_path.exists(), "prediction manifest missing")
    manifest = read_json(manifest_path)
    expected = _expected_prediction_keys()
    missing = [k for k in expected if k not in manifest["records"]]
    require(not missing, f"missing predictions: {missing[:5]}")
    extra = [k for k in manifest["records"] if k not in expected]
    require(not extra, f"unexpected prediction records: {extra[:5]}")
    for name, rec in manifest["records"].items():
        path = OUT / "predictions" / name
        require(path.exists(), f"prediction file missing: {name}")
        require(file_sha256(path) == rec["file_sha256"], f"prediction drifted: {name}")

    # cross-check that CPU/GPU own-kernel forwards are the same computation
    fp_max = {}
    for zone in ZONES:
        for arm in RDU:
            for seed in SEEDS:
                own = np.asarray(np.load(pred_path(zone, arm, seed))["pred"], dtype=np.float64)
                fp_max[f"{zone}_{arm}_s{seed}"] = {"n": int(own.size), "finite": bool(np.isfinite(own).all())}
                own_manifest = manifest["records"][pred_path(zone, arm, seed).name]
                require(own_manifest["pred_sha256"] == array_sha256(own), f"own prediction hash mismatch {zone}/{arm}/{seed}")

    # matching audits
    matching = {}
    for seed in SEEDS:
        ssl = read_json(ssl_json_path(seed))
        rec = {"ssl_atom_bond_sha256": ssl["atom_bond_group_sha256"], "ssl_full_sha256": ssl["full_encoder_group_sha256"]}
        for arm in RDU:
            info = read_json(train_json_path(arm, seed))
            rec[f"{arm}_encoder_used_sha256"] = info["encoder_used_sha256"]
            rec[f"{arm}_batch_order"] = info["batch_order"]["full_hash"]
            rec[f"{arm}_init"] = info["init_full_sha256"]
            rec[f"{arm}_updates"] = info["updates"]
        p3 = read_json(train_json_path("P3", seed))
        rec["P3_encoder_used_sha256"] = p3["encoder_used_sha256"]
        rec["P3_batch_order"] = p3["batch_order"]["full_hash"]
        rec["P3_init"] = p3["init_full_sha256"]
        rec["P3_updates"] = p3["updates"]
        rec["rdu_encoder_equal_ssl_atom_bond"] = all(
            rec[f"{a}_encoder_used_sha256"] == ssl["atom_bond_group_sha256"] for a in RDU
        )
        rec["p3_encoder_equal_ssl_full"] = (
            p3["encoder_used_sha256"] == ssl["encoder_final_sha256"]
            and ssl["encoder_final_sha256"] == ssl["full_encoder_group_sha256"]
        )
        rec["rdu_init_all_equal"] = len({rec[f"{a}_init"] for a in RDU}) == 1
        rec["batch_order_all_equal"] = len({rec[f"{a}_batch_order"] for a in RDU} | {rec["P3_batch_order"]}) == 1
        rec["updates_all_equal"] = len({rec[f"{a}_updates"] for a in RDU} | {rec["P3_updates"]}) == 1
        require(rec["rdu_encoder_equal_ssl_atom_bond"], f"seed {seed}: R/D/U encoder != SSL atom+bond")
        require(rec["p3_encoder_equal_ssl_full"], f"seed {seed}: P3 encoder != SSL full")
        require(rec["rdu_init_all_equal"], f"seed {seed}: R/D/U init differ")
        require(rec["batch_order_all_equal"], f"seed {seed}: batch order differs across arms")
        require(rec["updates_all_equal"], f"seed {seed}: update count differs across arms")
        matching[str(seed)] = rec

    manifest = {
        **manifest,
        "complete": True,
        "n_predictions": len(manifest["records"]),
        "expected_keys": expected,
        "frozen_utc": now_utc(),
        "matching_audit": matching,
        "own_kernel_prediction_checks": fp_max,
        "combined_pred_sha256": hashlib.sha256(
            json.dumps({k: manifest["records"][k]["pred_sha256"] for k in sorted(expected)}, sort_keys=True).encode()
        ).hexdigest(),
        "labels_used": "none",
    }
    write_json(manifest_path, manifest)
    log(f"[freeze-predictions] {manifest['n_predictions']} files frozen combined={manifest['combined_pred_sha256'][:12]}")
    return 0


# ---------------------------------------------------------------------------
# scoring (single read of the two held-out zones)
# ---------------------------------------------------------------------------


def _load_preds(zone: str, arm: str) -> dict[int, np.ndarray]:
    return {seed: np.asarray(np.load(pred_path(zone, arm, seed))["pred"], dtype=np.float64) for seed in SEEDS}


def _contrast(y: np.ndarray, a: dict[int, np.ndarray], b: dict[int, np.ndarray]) -> dict[str, Any]:
    base = ab.compute_contrast(a, b, y, bootstrap_n=bra.BOOTSTRAP_N, bootstrap_seed=bra.BOOTSTRAP_SEED)
    label = bra.contrast_label(base["delta_seed_mean"], base["per_seed_delta"], base["bootstrap"]["ci_lo"], base["bootstrap"]["ci_hi"])
    per_seed = {}
    for seed in SEEDS:
        one = ab.compute_contrast({seed: a[seed]}, {seed: b[seed]}, y, bootstrap_n=bra.BOOTSTRAP_N, bootstrap_seed=bra.BOOTSTRAP_SEED)
        per_seed[str(seed)] = {
            "delta": one["delta_seed_mean"],
            "ci_lo": one["bootstrap"]["ci_lo"],
            "ci_hi": one["bootstrap"]["ci_hi"],
            "mae_a": one["mae_new_seed_mean"],
            "mae_b": one["mae_ref_seed_mean"],
        }
    return {**base, "label": label, "per_seed_bootstrap": per_seed}


def _arm_stats(y: np.ndarray, preds: Mapping[int, np.ndarray]) -> dict[str, Any]:
    err = {s: np.abs(preds[s] - y) for s in SEEDS}
    seed_mean = np.mean([err[s] for s in SEEDS], axis=0)
    sse = seed_mean**2
    order = np.argsort(-seed_mean)
    return {
        "mae_seed_mean": float(np.mean([err[s].mean() for s in SEEDS])),
        "mae_per_seed": {str(s): float(err[s].mean()) for s in SEEDS},
        "rmse_seed_mean": float(np.sqrt(np.mean(np.mean([(preds[s] - y) ** 2 for s in SEEDS], axis=0)))),
        "median_ae": float(np.median(seed_mean)),
        "p90_ae": float(np.quantile(seed_mean, 0.90)),
        "p95_ae": float(np.quantile(seed_mean, 0.95)),
        "p99_ae": float(np.quantile(seed_mean, 0.99)),
        "max_ae": float(seed_mean.max()),
        "top10_sse_share": float(sse[order[:10]].sum() / max(sse.sum(), 1e-30)),
        "worst_rows": [
            {"row": int(i), "y": float(y[i]), "ae": float(seed_mean[i])} for i in order[:5]
        ],
    }


CONTRASTS = (
    ("R_minus_D", "R", "D"),
    ("R_minus_U", "R", "U"),
    ("D_minus_U", "D", "U"),
    ("R_minus_P3", "R", "P3"),
    ("D_minus_P3", "D", "P3"),
    ("U_minus_P3", "U", "P3"),
)


def cmd_score_holdout(_args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    lock = OUT / "holdout_read.lock"
    require(not lock.exists(), "held-out labels were already read in this round")
    manifest = read_json(OUT / "prediction_manifest_E.json")
    require(bool(manifest.get("complete")), "prediction manifest is not complete")
    for name, rec in manifest["records"].items():
        require(file_sha256(OUT / "predictions" / name) == rec["file_sha256"], f"prediction drifted: {name}")

    split = load_split()
    y = {zone: bp.load_y_rows(ENCODED_TRAIN, [int(i) for i in split[f"{zone}_rows"]]) for zone in ZONES}
    np.savez(OUT / "holdout_labels_E.npz", monitor=y["monitor"], locked=y["locked"])
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
        "primary_zone": bra.PRIMARY_ZONE,
        "replication_zone": bra.REPLICATION_ZONE,
        "zones": {},
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    for zone in ZONES:
        preds = {arm: _load_preds(zone, arm) for arm in ARMS}
        stats = {arm: _arm_stats(y[zone], preds[arm]) for arm in ARMS}
        contrasts = {name: _contrast(y[zone], preds[a], preds[b]) for name, a, b in CONTRASTS}
        result["zones"][zone] = {"arm_stats": stats, "contrasts": contrasts}
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

    primary = result["zones"][bra.PRIMARY_ZONE]["contrasts"]
    replication = result["zones"][bra.REPLICATION_ZONE]["contrasts"]
    case = bra.overall_case(
        p1_locked_label=primary["R_minus_D"]["label"],
        p1_monitor=replication["R_minus_D"]["per_seed_delta"],
        p2_locked_label=primary["R_minus_U"]["label"],
        p2_monitor=replication["R_minus_U"]["per_seed_delta"],
        p1_locked_delta=primary["R_minus_D"]["delta_seed_mean"],
        p1_monitor_delta=replication["R_minus_D"]["delta_seed_mean"],
        p2_locked_delta=primary["R_minus_U"]["delta_seed_mean"],
        r_minus_p3_locked=primary["R_minus_P3"]["delta_seed_mean"],
    )
    result["decision"] = case
    write_json(OUT / "holdout_score_E.json", result)
    write_json(
        OUT / "decision.json",
        {
            "protocol": PROTOCOL_ID,
            "git_commit": git_commit(),
            "split_tag": SPLIT_TAG,
            **case,
            "primary_zone": bra.PRIMARY_ZONE,
            "primary_labels": {k: v["label"] for k, v in primary.items()},
            "primary_deltas": {k: v["delta_seed_mean"] for k, v in primary.items()},
            "replication_deltas": {k: v["delta_seed_mean"] for k, v in replication.items()},
            "frozen_utc": now_utc(),
        },
    )
    log(
        f"[score-holdout] locked R={result['zones']['locked']['arm_stats']['R']['mae_seed_mean']:.6f} "
        f"D={result['zones']['locked']['arm_stats']['D']['mae_seed_mean']:.6f} "
        f"U={result['zones']['locked']['arm_stats']['U']['mae_seed_mean']:.6f} "
        f"P3={result['zones']['locked']['arm_stats']['P3']['mae_seed_mean']:.6f}"
    )
    log(
        f"  R-D {primary['R_minus_D']['delta_seed_mean']:+.6f} [{primary['R_minus_D']['bootstrap']['ci_lo']:+.6f},"
        f"{primary['R_minus_D']['bootstrap']['ci_hi']:+.6f}] {primary['R_minus_D']['label']}; "
        f"R-U {primary['R_minus_U']['delta_seed_mean']:+.6f} {primary['R_minus_U']['label']}; "
        f"R-P3 {primary['R_minus_P3']['delta_seed_mean']:+.6f} -> {case['case']}"
    )
    return 0


# ---------------------------------------------------------------------------
# post-hoc diagnostics
# ---------------------------------------------------------------------------


def _module_grad_norms(model, batch, yb) -> dict[str, float]:
    model.train()
    model.zero_grad(set_to_none=True)
    loss = torch.nn.functional.l1_loss(model(batch).view(-1), yb)
    loss.backward()
    out = {}
    for name in ("atom_encoder", "bond_encoder", "pair_mlp", "group_composer", "reader"):
        module = getattr(model, name)
        norms = [float(p.grad.norm()) for p in module.parameters() if p.grad is not None]
        out[name] = float(np.sqrt(np.sum(np.square(norms)))) if norms else 0.0
    model.zero_grad(set_to_none=True)
    return out


def cmd_post_hoc(args: argparse.Namespace) -> int:
    require((OUT / "holdout_read.lock").exists(), "post-hoc requires the single holdout read to have happened")
    labels = np.load(OUT / "holdout_labels_E.npz")
    y = {"monitor": labels["monitor"], "locked": labels["locked"]}
    norm = load_norm()
    device = torch.device(args.device)
    score = read_json(OUT / "holdout_score_E.json")

    out: dict[str, Any] = {"protocol": PROTOCOL_ID, "git_commit": git_commit()}

    # 1. cross-kernel forward diagnostics (predictions already frozen)
    cross = {}
    for zone in ZONES:
        own = {arm: _load_preds(zone, arm) for arm in RDU}
        for model_arm in RDU:
            for kernel in RDU:
                if model_arm == kernel:
                    continue
                p = {
                    seed: np.asarray(np.load(cross_pred_path(zone, model_arm, kernel, seed))["pred"], dtype=np.float64)
                    for seed in SEEDS
                }
                diff = np.mean([np.abs(p[s] - own[model_arm][s]).mean() for s in SEEDS])
                diff_kernel = np.mean([np.abs(p[s] - own[kernel][s]).mean() for s in SEEDS])
                mae = float(np.mean([np.abs(p[s] - y[zone]).mean() for s in SEEDS]))
                cross[f"{zone}_m{model_arm}_k{kernel}"] = {
                    "mae_seed_mean": mae,
                    "mean_abs_diff_to_same_model_own_kernel": float(diff),
                    "mean_abs_diff_to_kernel_arm_own_model": float(diff_kernel),
                }
    out["cross_kernel"] = cross

    # 2. gradient liveness on one Fit batch, frozen soups
    fit_objs = load_zone_objects("fit")
    kernels = load_zone_kernels("fit")
    y_fit = load_fit_y()
    grads = {}
    for seed in SEEDS:
        batch_objs = fit_objs[:16]
        yb = torch.as_tensor(y_fit[:16], dtype=torch.float32, device=device)
        grads[str(seed)] = {}
        for arm in RDU:
            model = bwo.build_model(seed, norm, device)
            model.load_state_dict({k: v.to(device) for k, v in torch.load(soup_path(arm, seed), map_location="cpu").items()})
            batch = bwo.move_batch(bwo.batch_rw_objects(batch_objs, kernels[arm][:16]), device)
            grads[str(seed)][arm] = _module_grad_norms(model, batch, yb)
    out["post_training_grad_norms"] = grads

    # 3. weight-distribution diagnostics on the audited molecules
    audit = read_json(OUT / "kernel_audit.json")
    out["kernel_stats"] = audit["aggregate"]
    out["kernel_per_zone"] = {z: audit["per_zone"][z]["aggregate"] for z in ZONES + ("fit",)}

    # 4. per-molecule delta correlation and error-region table
    per_zone = {}
    for zone in ZONES:
        own = {arm: _load_preds(zone, arm) for arm in RDU + ("P3",)}
        err = {arm: np.mean([np.abs(own[arm][s] - y[zone]) for s in SEEDS], axis=0) for arm in own}
        d_rd = err["R"] - err["D"]
        d_ru = err["R"] - err["U"]
        d_du = err["D"] - err["U"]
        ordinal = y[zone] > bra.TAIL_THRESHOLD
        per_zone[zone] = {
            "corr_delta_RD_delta_RU": float(np.corrcoef(d_rd, d_ru)[0, 1]),
            "corr_delta_RD_delta_DU": float(np.corrcoef(d_rd, d_du)[0, 1]),
            "frac_rows_R_better_than_D": float((d_rd < 0).mean()),
            "frac_rows_R_better_than_U": float((d_ru < 0).mean()),
            "ordinary": {
                arm: float(err[arm][ordinal].mean()) for arm in own
            },
            "tail": {
                arm: float(err[arm][~ordinal].mean()) for arm in own
            },
            "tail_n": int((~ordinal).sum()),
        }
    out["per_zone"] = per_zone

    # 5. summary of the frozen scoring decision (no new decision)
    out["decision"] = score["decision"]
    write_json(OUT / "post_hoc_mechanism.json", out)
    log(f"[post-hoc] cross={len(cross)} keys; per-zone deltas corr={per_zone['locked']['corr_delta_RD_delta_RU']:.3f}")
    return 0


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------


def cmd_finalize(_args: argparse.Namespace) -> int:
    freeze = verify_freeze(require_exact_code=True)
    ledger = read_json(ledger_path())
    files = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file():
            continue
        if path.name in {"REPORT.md", "evidence_manifest.json"}:
            continue
        if path.suffix in {".pt", ".npz", ".csv"}:
            files.append({"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path), "bytes": path.stat().st_size})
        elif path.suffix in {".json", ".lock", ".md", ".yaml"}:
            files.append({"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path), "bytes": path.stat().st_size})
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "freeze_sha256": file_sha256(OUT / "protocol_freeze.json"),
        "freeze_code_hashes": freeze["code_hashes"],
        "gpu": {
            "total_wall_hours": ledger["total_wall_hours"],
            "cap_hours": ledger["cap_hours"],
            "within_cap": ledger["within_cap"],
            "entries": ledger["entries"],
        },
        "artifacts": files,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "written_utc": now_utc(),
    }
    write_json(OUT / "evidence_manifest.json", payload)
    log(f"[finalize] {len(files)} artifacts hashed; gpu={ledger['total_wall_hours']:.4f}h/{ledger['cap_hours']}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in (
        "split",
        "verify-inputs",
        "objects",
        "kernels",
        "cpu-gates",
        "freeze",
        "freeze-predictions",
        "score-holdout",
        "post-hoc",
        "finalize",
    ):
        sp = sub.add_parser(name)
        sp.add_argument("--device", default="cpu")
        sp.set_defaults(func=globals()[f"cmd_{name.replace('-', '_')}"])
    for name in ("ssl", "train"):
        sp = sub.add_parser(name)
        sp.add_argument("--seed", type=int, required=True)
        sp.add_argument("--device", default="cuda")
        if name == "ssl":
            sp.add_argument("--smoke-updates", type=int, default=0)
            sp.set_defaults(func=cmd_ssl)
        else:
            sp.add_argument("--arm", required=False)
            sp.add_argument("--smoke", action="store_true")
            sp.add_argument("--epochs", type=int, default=None)
            sp.set_defaults(func=cmd_train)
    for name in ("predict",):
        sp = sub.add_parser(name)
        sp.add_argument("--arm", required=True)
        sp.add_argument("--seed", type=int, required=True)
        sp.add_argument("--device", default="cuda")
        sp.add_argument("--smoke", action="store_true")
        sp.set_defaults(func=cmd_predict)
    sp = sub.add_parser("predict-cross")
    sp.add_argument("--model", required=True)
    sp.add_argument("--kernel", required=True)
    sp.add_argument("--seed", type=int, required=True)
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=cmd_predict_cross)
    sp = sub.add_parser("smoke")
    sp.add_argument("--seed", type=int, required=True)
    sp.add_argument("--device", default="cuda")
    sp.set_defaults(func=cmd_smoke)
    return p


def cmd_smoke(args: argparse.Namespace) -> int:
    """Post-freeze end-to-end mini run: 20 SSL updates + 1 epoch per arm."""
    verify_freeze(require_exact_code=True)
    seed = int(args.seed)
    smoke_dir().mkdir(parents=True, exist_ok=True)
    ns = argparse.Namespace(seed=seed, device=args.device, smoke_updates=20)
    rc = cmd_ssl(ns)
    if rc != 0:
        return rc
    for arm in ARMS:
        rc = cmd_train(
            argparse.Namespace(arm=arm, seed=seed, device=args.device, smoke=True, epochs=1)
        )
        if rc != 0:
            return rc
        rc = cmd_predict(
            argparse.Namespace(arm=arm, seed=seed, device=args.device, smoke=True)
        )
        if rc != 0:
            return rc
    log(f"[smoke s{seed}] complete")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
