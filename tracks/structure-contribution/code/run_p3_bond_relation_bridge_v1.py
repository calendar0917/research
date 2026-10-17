#!/usr/bin/env python
"""Driver for protocol ``cscl-p3-bond-relation-bridge-v1``.

Stages
------
``split``               (local CPU, RDKit; label-blind) reproduce the four
                        SMILES/PyG mismatches, derive identity from the PyG
                        graph, build the new identity-grouped split F.
``verify-inputs``       reuse hashes of the frozen upstream modules + split
                        sanity + P3 parameter/recipe constants (label-blind).
``objects``             LabelGuard build of Fit/Monitor/Locked objects and the
                        Fit-only global standardiser (Monitor/Locked ``.y``
                        unreachable).
``stage-inputs``        rebuild the machine-local, gitignored object/pair
                        artifacts on a compute checkout and verify them against
                        the frozen split/standardiser/pair totals (amendment 1).
``kernels``             R/U pair lists for every zone + label-blind kernel audit.
``cpu-gates``           the eight pre-registered gates of PROTOCOL.md §6.
``freeze``              protocol_freeze.json (code + config + inputs + budgets).
``smoke``               post-freeze 20-update SSL + 1-epoch mini run per arm.
``ssl``                 Fit-only masked atom/bond SSL for one seed (no labels).
``train``               P3 / P3R / P3U for one seed (Fit labels only).
``predict``             own-kernel predictions on Monitor and Locked.
``predict-cross``       frozen-model x swapped-kernel diagnostic predictions.
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
import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import bond_relation_attribution_v1 as bra  # noqa: E402
import p3_bond_relation_bridge_v1 as pb  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

REPO_ROOT = _CODE_DIR.parents[2]
TRACK = _CODE_DIR.parent
OUT = Path(os.environ.get("PBR_OUT_DIR", str(TRACK / "results" / "p3_bond_relation_bridge_v1")))
CONFIG_PATH = TRACK / "configs" / "p3_bond_relation_bridge_v1.yaml"
PROTOCOL_DOC = OUT / "PROTOCOL.md"
ZINC_ROOT = REPO_ROOT / "data/ZINC"
DATA_TRAIN = ZINC_ROOT / "subset/processed/train.pt"
CANON = REPO_ROOT / "tracks/ksvd/results/zinc_cssd_basis_reuse_v1/train_canonical_smiles.npz"
ENCODED_TRAIN = REPO_ROOT / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt"
STRUCT_CONTEXT_CODE = TRACK / "results/structure_context_support_v1/code"

PROTOCOL_ID = pb.PROTOCOL_ID
SPLIT_TAG = pb.SPLIT_TAG
SEEDS = pb.SEEDS
ARMS = pb.ARMS
ZONES = pb.ZONES

FROZEN_INPUT_HASHES = {
    "train_pt": "e0f1837b4a19714fcf3a5c6727a30c23f57c3398c8987cab2983214a0e76955b",
    "canonical_smiles": "04a590355fb7bba71e35e52bed60c974999b92af9e96e4ee99bed4afe42dcaff",
    "encoded_train": "10d764955ef1b040a317105be1dd3fbb2e81292b03b79e7d078cce08e18e5f1d",
}
GPU_BUDGET_CAP_HOURS = 4.0
KERNEL_AUDIT_N = 400
FORBIDDEN_OFFICIAL = ("valid", "test")

GATE_BATCH = 32
EQUIV_EPOCHS = 5
GRAD_STEPS = 3


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
        json_dump(payload, fh)
        fh.write("\n")


def json_dump(payload: Mapping[str, Any], fh) -> None:
    json.dump(payload, fh, indent=1, sort_keys=False, default=_json_default)


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


ROUND_CODE = ("p3_bond_relation_bridge_v1.py", "run_p3_bond_relation_bridge_v1.py")
SCORING_CODE = ROUND_CODE


def code_hashes() -> dict[str, str]:
    return {name: file_sha256(_CODE_DIR / name) for name in ROUND_CODE}


# ---------------------------------------------------------------------------
# artifact paths
# ---------------------------------------------------------------------------


def split_path() -> Path:
    return OUT / "split_F.json"


def objects_path(zone: str) -> Path:
    return OUT / f"objects_{zone}_F.pt"


def prep_path() -> Path:
    return OUT / "prep_fit_F.json"


def pairs_path(zone: str) -> Path:
    return OUT / f"pairs_{zone}_F.pt"


def ssl_path(seed: int) -> Path:
    return OUT / f"ssl_F_s{int(seed)}.pt"


def ssl_json_path(seed: int) -> Path:
    return OUT / f"ssl_F_s{int(seed)}.json"


def soup_path(arm: str, seed: int) -> Path:
    return OUT / f"soup_{arm}_F_s{int(seed)}.pt"


def train_json_path(arm: str, seed: int) -> Path:
    return OUT / f"train_{arm}_F_s{int(seed)}.json"


def pred_path(zone: str, arm: str, seed: int) -> Path:
    return OUT / "predictions" / f"{zone}_{arm}_F_s{int(seed)}.npz"


def cross_pred_path(zone: str, model_arm: str, kernel: str, seed: int) -> Path:
    return OUT / "predictions" / f"cross_{zone}_m{model_arm}_k{kernel}_F_s{int(seed)}.npz"


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
    return list(torch.load(objects_path(zone), map_location="cpu", weights_only=False))


def load_zone_pairs(zone: str, arm: str) -> pb.PairSet:
    payload = torch.load(pairs_path(zone), map_location="cpu", weights_only=False)
    return pb.PairSet.from_state(payload[pb.KERNEL_OF_ARM[arm]])


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


def cmd_split(_args: argparse.Namespace) -> int:
    from rdkit import Chem

    require(not freeze_path().exists(), "split is frozen already; refusing to rebuild")
    scs = _structure_context_module()
    pairing = scs.pairing_verification()
    write_json(OUT / "pairing_verification.json", pairing)
    require(pairing["n_mismatch"] == 4, f"expected 4 SMILES/PyG mismatches, got {pairing['n_mismatch']}")
    require(tuple(pairing["mismatch_rows"]) == pb.EXCLUDED_ROWS, f"mismatch rows changed: {pairing['mismatch_rows']}")
    require(pairing["reproduces_frozen_exclusion_list"], "exclusion list does not reproduce")
    rows = [int(r) for r in range(10_000) if int(r) not in set(pb.EXCLUDED_ROWS)]
    require(len(rows) == pb.N_USABLE, "usable row count changed")

    structs = scs.load_pyg_structure_only(rows)  # label-blind: y is discarded
    require([int(s["row"]) for s in structs] == rows, "structure rows out of order")
    graph_identity = {int(s["row"]): str(scs._pyg_rebuilt_smiles(s)) for s in structs}

    with np.load(CANON, allow_pickle=True) as blob:
        table = [str(s) for s in blob["smiles"]]
    table_identity = {r: Chem.CanonSmiles(table[r], useChiral=False) for r in rows}

    graph_part = identity_partition(rows, graph_identity)
    table_part = identity_partition(rows, table_identity)
    partition_equal = graph_part == table_part
    require(partition_equal, "graph identity partition != committed-SMILES partition")

    split = bra.build_identity_split(rows, [graph_identity[r] for r in rows])
    require(split["duplicate_groups_cross_sides"] == 0, "identity group crosses split zones")
    sizes = (len(split["fit_rows"]), len(split["monitor_rows"]), len(split["locked_rows"]))
    require(abs(sizes[0] - pb.FIT_TARGET) <= pb.SIZE_TOLERANCE, f"fit size off target: {sizes}")
    require(abs(sizes[1] - pb.MONITOR_TARGET) <= pb.SIZE_TOLERANCE, f"monitor size off target: {sizes}")
    require(abs(sizes[2] - pb.LOCKED_TARGET) <= pb.SIZE_TOLERANCE, f"locked size off target: {sizes}")

    all_rows = split["fit_rows"] + split["monitor_rows"] + split["locked_rows"]
    require(sorted(all_rows) == rows, "split union does not equal the usable rows")
    gid = split["group_id"]
    sorted_rows = sorted(all_rows)
    identity_digest = hashlib.sha256(
        "\n".join(f"{r}\t{graph_identity[r]}" for r in sorted_rows).encode()
    ).hexdigest()
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": SPLIT_TAG,
        "salt": pb.SPLIT_SALT,
        "n_rows": len(sorted_rows),
        "n_groups": int(split["n_groups"]),
        "n_fit": len(split["fit_rows"]),
        "n_monitor": len(split["monitor_rows"]),
        "n_locked": len(split["locked_rows"]),
        "fit_rows": split["fit_rows"],
        "monitor_rows": split["monitor_rows"],
        "locked_rows": split["locked_rows"],
        "rows_sorted": sorted_rows,
        "group_ids_sorted": [int(gid[str(r)]) for r in sorted_rows],
        "fit_sha256_rows": rows_sha256(split["fit_rows"]),
        "monitor_sha256_rows": rows_sha256(split["monitor_rows"]),
        "locked_sha256_rows": rows_sha256(split["locked_rows"]),
        "excluded_rows": list(pb.EXCLUDED_ROWS),
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
    write_json(
        OUT / "split_audit_F.json",
        {
            "protocol": PROTOCOL_ID,
            "git_commit": git_commit(),
            "pairing_verification": pairing,
            "n_rows_official_train": 10_000,
            "n_rows_excluded": len(pb.EXCLUDED_ROWS),
            "n_rows_usable": len(rows),
            "n_groups_identity": len(graph_part),
            "sizes": {"fit": sizes[0], "monitor": sizes[1], "locked": sizes[2]},
            "targets": {"fit": pb.FIT_TARGET, "monitor": pb.MONITOR_TARGET, "locked": pb.LOCKED_TARGET},
            "labels_read": False,
        },
    )
    log(f"[split] n={len(sorted_rows)} groups={split['n_groups']} sizes={sizes} cross=0")
    return 0


# ---------------------------------------------------------------------------
# verify inputs
# ---------------------------------------------------------------------------


def cmd_verify_inputs(_args: argparse.Namespace) -> int:
    require(split_path().exists(), "run `split` first")
    observed = {}
    for name, path in (
        ("train_pt", DATA_TRAIN),
        ("canonical_smiles", CANON),
        ("encoded_train", ENCODED_TRAIN),
    ):
        got = file_sha256(path)
        require(got == FROZEN_INPUT_HASHES[name], f"frozen input drifted: {name} {got}")
        observed[name] = got

    reuse = {}
    for name, expected in pb.REUSE_HASHES.items():
        got = file_sha256(_CODE_DIR / name)
        require(got == expected, f"reused module drifted: {name} {got} != {expected}")
        reuse[name] = got

    cfg = src.BJOCConfig()
    base_model = bp.build_bjoc_null(0, {"global_mean": np.zeros(sro.GLOBAL_DIM), "global_scale": np.ones(sro.GLOBAL_DIM)}, "cpu")
    p3_params = pb.model_parameters(base_model)
    require(p3_params == pb.BASE_PARAMETERS == bp.EXPECTED_PARAMETERS, f"P3 parameter count changed: {p3_params}")
    require(pb.BRIDGE_PARAMETERS == pb.BASE_PARAMETERS + 9344 + 96, "bridge parameter arithmetic is wrong")
    recipe = bp.RECIPES["R3"]
    require(int(recipe["epochs"]) == 360, "R3 epochs changed")
    require(abs(float(recipe["lr"]) - 1e-3) < 1e-12 and abs(float(recipe["lr_min"]) - 1e-5) < 1e-12, "R3 lr changed")
    require(abs(float(recipe["weight_decay"]) - 1e-5) < 1e-12, "R3 weight decay changed")
    require(abs(float(bp.GRAD_CLIP) - 5.0) < 1e-12, "grad clip changed")
    require(int(bp.BATCH_SIZE) == 128, "BATCH_SIZE changed")

    split = load_split()
    sizes = {z: len(split[f"{z}_rows"]) for z in ("fit", "monitor", "locked")}
    all_rows = sorted(split["fit_rows"] + split["monitor_rows"] + split["locked_rows"])
    require(len(set(all_rows)) == len(all_rows), "split rows are not unique")
    require(all_rows == [r for r in range(10_000) if r not in set(pb.EXCLUDED_ROWS)], "split union changed")

    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "frozen_inputs": observed,
        "reused_modules": reuse,
        "p3_parameters": int(p3_params),
        "bridge_parameters": int(pb.BRIDGE_PARAMETERS),
        "new_parameters_per_arm": int(pb.BRIDGE_PARAMETERS - pb.BASE_PARAMETERS),
        "recipe_R3": {k: (list(v) if isinstance(v, tuple) else v) for k, v in dict(recipe).items()},
        "batch_size": int(bp.BATCH_SIZE),
        "grad_clip": float(bp.GRAD_CLIP),
        "soup_last": int(bp.SOUP_LAST),
        "shuffle_offset": int(bp.TRAIN_SHUFFLE_OFFSET),
        "bjoc_config": {
            "atom_hidden": cfg.atom_hidden,
            "bond_hidden": cfg.bond_hidden,
            "junc_hidden": cfg.junc_hidden,
            "compose_out": cfg.compose_out,
            "reader_hidden": cfg.reader_hidden,
        },
        "split_sizes": sizes,
        "split_monitor_sha256_rows": split["monitor_sha256_rows"],
        "split_locked_sha256_rows": split["locked_sha256_rows"],
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "labels_read": False,
    }
    write_json(OUT / "input_audit.json", payload)
    write_json(OUT / "reuse_audit.json", {**payload, "reuse_hashes_ok": True})
    log(f"[verify-inputs] p3={p3_params} bridge={pb.BRIDGE_PARAMETERS} split={sizes}")
    return 0


# ---------------------------------------------------------------------------
# objects
# ---------------------------------------------------------------------------


def cmd_objects(_args: argparse.Namespace) -> int:
    require(not freeze_path().exists(), "objects are frozen already; refusing to rebuild")
    split, ds, guard = build_guard()
    OUT.mkdir(parents=True, exist_ok=True)
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
    write_json(OUT / "objects_manifest_F.json", manifest)
    log(
        f"[objects] {manifest['build_seconds']:.1f}s fit={len(objs['fit'])} monitor={len(objs['monitor'])} "
        f"locked={len(objs['locked'])} alignment={max_diff:.2e}"
    )
    return 0



# ---------------------------------------------------------------------------
# stage-inputs (amendment 1): machine-local artifacts on a compute checkout
# ---------------------------------------------------------------------------


def cmd_stage_inputs(_args: argparse.Namespace) -> int:
    """Rebuild the gitignored ``.pt`` artifacts and verify them semantically.

    ``objects_*.pt`` / ``pairs_*.pt`` are deterministic, label-free functions of
    the official PyG graphs, but they are machine-local (torch serialisation is
    not byte-stable across checkouts), so a compute checkout rebuilds them and
    checks: identical row order, identical Fit-only standardiser, identical
    bond/pair totals per zone, and zero guarded label access.
    """
    fr = verify_freeze(require_exact_code=True)
    require(not (OUT / "holdout_read.lock").exists(), "refusing to restage inputs after labels were read")
    split, ds, guard = build_guard()
    frozen_pairs = read_json(OUT / "pairs_manifest_F.json")
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
        zone_payload = {}
        for arm in pb.BRIDGE_ARMS:
            pset = pb.build_pair_set(objs[zone], pb.KERNEL_OF_ARM[arm])
            zone_payload[pb.KERNEL_OF_ARM[arm]] = pset.to_state()
        torch.save(zone_payload, pairs_path(zone))
        totals = {k: int(v["src"].shape[0]) for k, v in zone_payload.items()}
        expected = {k: int(frozen_pairs["zones"][zone]["n_pairs"][k]) for k in totals}
        require(totals == expected, f"{zone}: pair totals changed {totals} vs {expected}")
        payload["zones"][zone] = {
            "n_molecules": len(objs[zone]),
            "n_pairs": totals,
            "n_pairs_match_frozen": True,
            "objects_file_sha256": file_sha256(objects_path(zone)),
            "pairs_file_sha256": file_sha256(pairs_path(zone)),
        }
        log(f"[stage-inputs {zone}] n={len(objs[zone])} pairs={totals} (match frozen)")
    require(guard.access_attempts == [], "guarded label access during staging")
    payload.update(
        {
            "freeze_sha256": file_sha256(freeze_path()),
            "freeze_amendment": fr.get("amendment", 0),
            "standardiser_max_abs_diff": {"mean": mean_diff, "scale": scale_diff},
            "build_seconds": time.perf_counter() - started,
            "forbidden_label_access_attempts": len(guard.access_attempts),
            "note": "torch .pt bytes are machine-local; equality is verified semantically",
            "monitor_labels_read": False,
            "locked_labels_read": False,
        }
    )
    write_json(OUT / "staged_inputs_F.json", payload)
    log(f"[stage-inputs] ok in {payload['build_seconds']:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# kernels / pair lists
# ---------------------------------------------------------------------------


def cmd_kernels(_args: argparse.Namespace) -> int:
    require(not freeze_path().exists(), "kernels are frozen already; refusing to rebuild")
    manifest: dict[str, Any] = {"protocol": PROTOCOL_ID, "git_commit": git_commit(), "zones": {}}
    for zone in ("fit", "monitor", "locked"):
        objs = load_zone_objects(zone)
        started = time.perf_counter()
        payload = {}
        for arm in pb.BRIDGE_ARMS:
            pset = pb.build_pair_set(objs, pb.KERNEL_OF_ARM[arm])
            require(len(pset) == len(objs), f"{zone}/{arm}: pair set length mismatch")
            payload[pb.KERNEL_OF_ARM[arm]] = pset.to_state()
        torch.save(payload, pairs_path(zone))
        counts = {k: int(v["src"].shape[0]) for k, v in payload.items()}
        manifest["zones"][zone] = {
            "n_molecules": len(objs),
            "n_pairs": counts,
            "pairs_per_molecule": {k: counts[k] / max(len(objs), 1) for k in counts},
            "build_seconds": time.perf_counter() - started,
            "file_sha256": file_sha256(pairs_path(zone)),
        }
        log(f"[kernels {zone}] n={len(objs)} pairs={counts}")

    per_mol = []
    for zone in ("fit", "monitor", "locked"):
        objs = load_zone_objects(zone)
        for o in objs[: min(KERNEL_AUDIT_N, len(objs))]:
            rec = pb.audit_molecule(o)
            rec["zone"] = zone
            per_mol.append(rec)
    overall = pb.summarize_audit(per_mol)
    per_zone = {z: pb.summarize_audit([r for r in per_mol if r["zone"] == z]) for z in ("fit", "monitor", "locked")}
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "audit_molecules_per_zone": KERNEL_AUDIT_N,
        "definitions": {
            "R": "K_R[e,f] = A_line[e,f] / deg_line(e), zero diagonal, row sum 1, no self loops",
            "U": "K_U[e,f] = 1/(n_bonds-1) for e != f, n_bonds=1 -> zero, row sum 1, no self loops",
            "pair_convention": "unordered e<f with both row weights stored (w_src for row e, w_dst for row f)",
        },
        "checks_note": (
            "R and U store different pair lists (R: line-graph edges only, U: all e<f), so the "
            "pair-MLP FLOPs differ; row sums and total mass match exactly on every molecule with "
            "n_bonds >= 2.  R==U only when the line graph is complete (always at n_bonds=2)."
        ),
        "aggregate": overall,
        "per_zone": per_zone,
        "per_molecule": per_mol,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "relation_audit.json", payload)
    write_json(OUT / "pairs_manifest_F.json", manifest)

    require(overall["all_checks_true"], f"kernel audit failed: {overall['n_checks_true']}")
    require(overall["frac_R_equals_U"] <= 0.10, f"R degenerates to U on too many molecules: {overall['frac_R_equals_U']:.3f}")
    for zone in ("fit", "monitor", "locked"):
        require(per_zone[zone]["all_checks_true"], f"{zone}: kernel audit failed")
    log(
        f"[kernels] audit ok molecules={overall['n_molecules']} R==U {overall['frac_R_equals_U']:.4f} "
        f"single={overall['n_single_bond']} two={overall['n_two_bonds']}"
    )
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


def _batch(objs: Sequence[sro.MolObjects], device: torch.device) -> dict[str, torch.Tensor]:
    return pb.move_batch(sro.batch_objects(list(objs)), device)


def _gate_result(name: str, ok: Any, **detail: Any) -> dict[str, Any]:
    return {"gate": name, "passed": bool(ok), **detail}


def cmd_cpu_gates(_args: argparse.Namespace) -> int:
    require(not freeze_path().exists(), "gates are frozen already; refusing to rebuild")
    device = torch.device("cpu")
    gates: list[dict[str, Any]] = []
    cfg = config()

    # ---- G1: kernel properties on real molecules + edge cases ---------------
    objs_fit = load_zone_objects("fit")
    sample = objs_fit[: min(300, len(objs_fit))]
    audit = pb.summarize_audit([pb.audit_molecule(o) for o in sample])
    synth = _synthetic_objects()
    synth_audit = [pb.audit_molecule(o) for o in synth]
    renumber_max = 0.0
    for o in sample[:100]:
        perm = np.random.default_rng(int(o.mol_index) + 17).permutation(int(o.n_atoms))
        o2 = _relabel(o, perm)
        for arm in ("R", "U"):
            a1 = pb.kernel_matrices(o)[arm]
            a2 = pb.kernel_matrices(o2)[arm]
            renumber_max = max(renumber_max, float(np.abs(a1 - a2).max()) if a1.size else 0.0)
    gates.append(
        _gate_result(
            "G1_kernel_properties",
            audit["all_checks_true"] and all(r["checks"]["R_rowsum_ok"] and r["checks"]["U_rowsum_ok"] for r in synth_audit)
            and renumber_max == 0.0,
            n_molecules=audit["n_molecules"],
            checks=audit["n_checks_true"],
            renumbering_max_abs_diff=renumber_max,
            single_bond_ok=all(r["checks"]["U_rowsum_ok"] for r in synth_audit[:1]),
            R_equals_U_fraction=audit["frac_R_equals_U"],
        )
    )

    # ---- G2: matched parameters / init / batch order ------------------------
    norm = load_norm()
    p3 = bp.build_bjoc_null(0, norm, device)
    r = pb.build_bridge(0, norm, device)
    u = pb.build_bridge(0, norm, device)
    p3_params = pb.model_parameters(p3)
    r_params = pb.model_parameters(r)
    u_params = pb.model_parameters(u)
    r_hash, u_hash = pb.bridge_state_hash(r), pb.bridge_state_hash(u)
    base_hashes = [src.state_dict_hash(m) for m in (p3, r.base, u.base)]
    r_pair = {k: v for k, v in r.state_dict().items() if k.startswith("pair_mlp") or k.startswith("rel_scale")}
    u_pair = {k: v for k, v in u.state_dict().items() if k.startswith("pair_mlp") or k.startswith("rel_scale")}
    pair_equal = all(torch.equal(r_pair[k], u_pair[k]) for k in r_pair)
    scale_zero = bool(torch.count_nonzero(r.rel_scale).item() == 0)
    genomes = [bp.shuffle_batches(200, 128, torch.Generator().manual_seed(s + 101)) for s in (0, 1)]
    order_hashes = [hashlib.sha256(np.concatenate([np.asarray(b, dtype=np.int64) for b in g]).tobytes()).hexdigest() for g in genomes]
    gates.append(
        _gate_result(
            "G2_matched_parameters_and_init",
            p3_params == pb.BASE_PARAMETERS
            and r_params == u_params == pb.BRIDGE_PARAMETERS
            and r_hash == u_hash
            and len(set(base_hashes)) == 1
            and pair_equal
            and scale_zero
            and order_hashes[0] != order_hashes[1],
            parameters={"P3": p3_params, "P3R": r_params, "P3U": u_params},
            state_hash_P3R=r_hash,
            state_hash_P3U=u_hash,
            base_init_hash=base_hashes[0],
            new_parameter_tensors_identical=bool(pair_equal),
            rel_scale_all_zero=scale_zero,
            batch_order_sha256=order_hashes,
            batch_orders_differ_by_seed=bool(order_hashes[0] != order_hashes[1]),
        )
    )

    # ---- G3: gamma = 0 -> bit-identical to P3 ------------------------------
    batch_objs = objs_fit[:GATE_BATCH]
    base_pred = pb.predict_model(p3, batch_objs, None, device)
    r_pred = pb.predict_model(r, batch_objs, load_zone_pairs("fit", "P3R"), device, chunk=GATE_BATCH)
    u_pred = pb.predict_model(u, batch_objs, load_zone_pairs("fit", "P3U"), device, chunk=GATE_BATCH)
    batch = _batch(batch_objs, device)
    feats_base = p3.forward_features(batch)
    feats_r = r.forward_features(batch, load_zone_pairs("fit", "P3R").batch(list(range(GATE_BATCH)), device))
    z_diff = float((feats_r["bridge_z"] - feats_base["z"]).abs().max())
    out_max = max(float(np.abs(r_pred - base_pred).max()), float(np.abs(u_pred - base_pred).max()))
    gates.append(
        _gate_result(
            "G3_zero_init_equals_P3",
            out_max == 0.0 and z_diff == 0.0,
            max_abs_pred_diff_vs_P3=out_max,
            max_abs_z_block_diff=z_diff,
            bitwise_identical=bool(out_max == 0.0 and z_diff == 0.0),
        )
    )

    # ---- G4: the relation branch really changes the forward ----------------
    pairs_r = load_zone_pairs("fit", "P3R")
    pairs_u = load_zone_pairs("fit", "P3U")
    b_feat = p3.forward_features(batch)["b"]
    with torch.no_grad():
        r.rel_scale.fill_(0.05)
        pred_scale_r = pb.predict_model(r, batch_objs, pairs_r, device, chunk=GATE_BATCH)
        pred_scale_u = pb.predict_model(r, batch_objs, pairs_u, device, chunk=GATE_BATCH)
        r.rel_scale.fill_(1.0)
        agg_r = r.relation_aggregate(b_feat, pairs_r.batch(list(range(GATE_BATCH)), device))
        agg_u = r.relation_aggregate(b_feat, pairs_u.batch(list(range(GATE_BATCH)), device))
        r.rel_scale.zero_()
    rel_effect = float(np.abs(pred_scale_r - base_pred).max())
    kernel_effect = float(np.abs(pred_scale_r - pred_scale_u).max())
    agg_kernel_effect = float((agg_r - agg_u).abs().max())
    gates.append(
        _gate_result(
            "G4_relation_affects_forward",
            rel_effect > 1e-6 and kernel_effect > 1e-6 and agg_kernel_effect > 1e-6,
            max_abs_delta_with_nonzero_gamma=rel_effect,
            max_abs_delta_R_kernel_vs_U_kernel=kernel_effect,
            max_abs_aggregate_diff_R_vs_U=agg_kernel_effect,
            kernel_is_consumed=bool(kernel_effect > 1e-6 and agg_kernel_effect > 1e-6),
            note="gamma=0.05 probe at initialisation; the trained residual is measured in post-hoc",
        )
    )

    # ---- G5: gradients reach the new parameters ---------------------------
    y_fit = load_fit_y()
    grad_objs = objs_fit[:GATE_BATCH]
    grad_model = pb.build_bridge(0, norm, device)
    opt = torch.optim.Adam(grad_model.parameters(), lr=1e-3)
    probe = {}
    for step in range(GRAD_STEPS):
        gb = _batch(grad_objs, device)
        gpred = grad_model(gb, pairs_r.batch(list(range(GATE_BATCH)), device))
        gt = torch.as_tensor(y_fit[:GATE_BATCH], dtype=torch.float32, device=device)
        loss = torch.nn.functional.l1_loss(gpred.view(-1), gt.view(-1))
        opt.zero_grad()
        loss.backward()
        probe[str(step + 1)] = {
            "loss": float(loss),
            "pair_mlp_grad_norm": pb._module_grad_norm(grad_model.pair_mlp),
            "rel_scale_grad_norm": float(grad_model.rel_scale.grad.norm()),
        }
        opt.step()
    gates.append(
        _gate_result(
            "G5_gradient_flow",
            probe["1"]["rel_scale_grad_norm"] > 0.0
            and probe["2"]["pair_mlp_grad_norm"] > 0.0
            and probe["3"]["pair_mlp_grad_norm"] > 0.0,
            probe=probe,
            first_step_pair_mlp_grad_zero_by_construction=bool(probe["1"]["pair_mlp_grad_norm"] == 0.0),
        )
    )

    # ---- G6: no label path / junction path untouched ----------------------
    keys = sorted(batch.keys())
    require("y" not in keys, "batch unexpectedly carries labels")
    base_before = src.state_dict_hash(r.base)
    with torch.no_grad():
        _ = r.forward(batch, pairs_r.batch(list(range(GATE_BATCH)), device))
    base_after = src.state_dict_hash(r.base)
    junction_equal = bool(torch.equal(feats_r["bridge_reader_in"][:, 96:], feats_base["reader_in"][:, 96:]))
    gates.append(
        _gate_result(
            "G6_no_label_leak_and_base_untouched",
            base_before == base_after and junction_equal and "y" not in keys,
            batch_keys=keys,
            base_hash_before=base_before,
            base_hash_after=base_after,
            base_unchanged=bool(base_before == base_after),
            junction_and_atom_blocks_identical=bool(junction_equal),
            relation_inputs="bond representation b + pair indices + kernel weights only",
            labels_in_model_path=False,
            differences_from_history={
                "organized_null_bjoc": "re-broadcast junction content into a bond-gated context; no explicit bond-bond pair MLP",
                "glrc": "jointly re-weighted existing per-bond composer inputs; no new pair interactions",
                "this_round": "explicit shared symmetric pair MLP over real one-hop bond pairs, added as a zero-init residual on c",
            },
            new_information_independence_guaranteed=False,
        )
    )

    # ---- G7: edge-case graphs -------------------------------------------
    edge = []
    for o in synth:
        b = _batch([o], device)
        for arm, pset_arm in (("R", "P3R"), ("U", "P3U")):
            ps = pb.PairSet([pb.pair_lists(o, arm)], [int(np.asarray(o.bond_index).reshape(-1, 2).shape[0])])
            with torch.no_grad():
                pred = float(r(b, ps.batch([0], device))[0]) if int(o.n_atoms) > 0 else 0.0
                agg = r.relation_aggregate(r.base.forward_features(b)["b"], ps.batch([0], device))
            edge.append(
                {
                    "mol_index": int(o.mol_index),
                    "n_bonds": int(np.asarray(o.bond_index).reshape(-1, 2).shape[0]),
                    "kernel": arm,
                    "n_pairs": int(ps.n_pairs),
                    "finite": bool(np.isfinite(pred)),
                    "agg_abs_max": float(agg.abs().max()),
                }
            )
    single = [e for e in edge if e["n_bonds"] == 1]
    gates.append(
        _gate_result(
            "G7_edge_cases",
            all(e["finite"] for e in edge) and all(e["n_pairs"] == 0 for e in single),
            cases=edge,
            single_bond_zero_pairs=all(e["n_pairs"] == 0 for e in single),
        )
    )

    # ---- G8: frozen-loop equivalence for the P3 arm ----------------------
    rows = [int(i) for i in load_split()["fit_rows"][:GATE_BATCH]]
    y_eq = y_fit[:GATE_BATCH]
    mine = pb.train_supervised_bridge(
        arm="P3",
        seed=0,
        model=bp.build_bjoc_null(0, norm, device),
        pairs=None,
        train_objs=load_zone_objects("fit")[:GATE_BATCH],
        y_fit=y_eq,
        device=device,
        recipe_key="R3",
        epochs=EQUIV_EPOCHS,
        log_fn=lambda *_: None,
    )
    theirs = bp.train_supervised(
        arm="P3",
        seed=0,
        recipe_key="R3",
        init="scratch",
        train_objs=load_zone_objects("fit")[:GATE_BATCH],
        rows=rows,
        y_fit=y_eq,
        norm=norm,
        device=device,
        smoke_epochs=EQUIV_EPOCHS,
        log_fn=lambda *_: None,
    )
    diffs = {
        k: float((mine["soup_state"][k].float() - theirs["soup_state"][k].float()).abs().max())
        for k in mine["soup_state"]
    }
    max_diff = max(diffs.values()) if diffs else 0.0
    order_equal = mine["batch_order"]["full_hash"] == theirs["batch_order"]["full_hash"]
    mae_equal = abs(float(mine["soup_fit_mae"]) - float(theirs["soup_fit_mae"])) <= 1e-9
    gates.append(
        _gate_result(
            "G8_frozen_loop_equivalence",
            max_diff <= 1e-6 and order_equal and mae_equal,
            max_abs_soup_diff=max_diff,
            batch_order_equal=bool(order_equal),
            soup_fit_mae_mine=float(mine["soup_fit_mae"]),
            soup_fit_mae_frozen_trainer=float(theirs["soup_fit_mae"]),
            tolerance="1e-6 (CPU float32 reduction noise floor)",
        )
    )

    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "device": str(device),
        "gates": gates,
        "all_passed": all(g["passed"] for g in gates),
        "frozen_constants": {
            "pair_hidden": cfg["pair_hidden"],
            "pair_init_offset": cfg["pair_init_offset"],
            "meaningful_delta": cfg["meaningful_delta"],
            "tail_threshold": cfg["tail_threshold"],
            "bootstrap_n": cfg["bootstrap_n"],
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "monitor_labels_read": False,
        "locked_labels_read": False,
    }
    write_json(OUT / "cpu_gates.json", payload)
    for g in gates:
        log(f"[gate] {g['gate']:<38} {'PASS' if g['passed'] else 'FAIL'}")
    require(payload["all_passed"], "CPU gates failed; formal training must not start")
    return 0


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------


def cmd_freeze(args: argparse.Namespace) -> int:
    ctx = config()
    require(OUT.joinpath("split_F.json").exists(), "missing split")
    require(OUT.joinpath("cpu_gates.json").exists(), "missing cpu gates")
    require(read_json(OUT / "cpu_gates.json")["all_passed"], "cannot freeze failing gates")
    if freeze_path().exists() and not args.amendment:
        raise SystemExit("protocol already frozen; pass --amendment to re-freeze after a documented fix")
    if freeze_path().exists():
        old = read_json(freeze_path())
        target = OUT / f"protocol_freeze_amendment{int(old.get('amendment', 0)) + 1}.json"
        target.write_text(freeze_path().read_text())
        log(f"[freeze] archived previous freeze to {target.name}")
        amendment = int(old.get("amendment", 0)) + 1
    else:
        amendment = 0
    inputs = {}
    for name in (
        "PROTOCOL.md",
        "split_F.json",
        "split_audit_F.json",
        "pairing_verification.json",
        "objects_manifest_F.json",
        "prep_fit_F.json",
        "pairs_manifest_F.json",
        "relation_audit.json",
        "cpu_gates.json",
        "input_audit.json",
        "reuse_audit.json",
    ):
        path = OUT / name
        require(path.exists(), f"missing frozen input {name}")
        inputs[name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    payload = {
        "protocol": PROTOCOL_ID,
        "split_tag": SPLIT_TAG,
        "salt": pb.SPLIT_SALT,
        "git_commit": git_commit(),
        "frozen_utc": now_utc(),
        "amendment": amendment,
        "code": {name: {"sha256": file_sha256(_CODE_DIR / name)} for name in ROUND_CODE},
        "config": {"path": str(CONFIG_PATH.relative_to(REPO_ROOT)), "sha256": config_sha256()},
        "inputs": inputs,
        "scoring": {
            "meaningful_delta": ctx["meaningful_delta"],
            "tail_threshold": ctx["tail_threshold"],
            "top_k_removed": ctx["top_k_removed"],
            "bootstrap_n": ctx["bootstrap_n"],
            "bootstrap_seed": ctx["bootstrap_seed"],
            "primary_zone": ctx["primary_zone"],
            "replication_zone": ctx["replication_zone"],
            "label_rule": "GAIN: delta <= -0.004, both seeds < 0, CI upper < 0, ordinary delta <= 0; LOSS: mirror; else NEUTRAL",
            "case_rule": "A > B > C > E > D as implemented in p3_bond_relation_bridge_v1.bridge_case",
        },
        "budget": {"gpu_hours_cap": GPU_BUDGET_CAP_HOURS, "ssl_updates": ctx["ssl_updates"], "updates": ctx["updates"]},
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "parameters": {"P3": pb.BASE_PARAMETERS, "P3R": pb.BRIDGE_PARAMETERS, "P3U": pb.BRIDGE_PARAMETERS},
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(freeze_path(), payload)
    log(f"[freeze] amendment={amendment} code={payload['code']['run_p3_bond_relation_bridge_v1.py']['sha256'][:12]}")
    return 0


# ---------------------------------------------------------------------------
# model construction for the GPU stages
# ---------------------------------------------------------------------------


def build_arm_model(arm: str, seed: int, norm: Mapping[str, Any], ssl_state: Mapping[str, Any] | None, device):
    if arm == "P3":
        model = bp.build_bjoc_null(seed, norm, device)
        if ssl_state is not None:
            mp.load_encoder_state(model, ssl_state, device)
        return model
    model = pb.build_bridge(seed, norm, device)
    if ssl_state is not None:
        mp.load_encoder_state(model.base, ssl_state, device)
    return model


def _ssl_encoder_hashes(enc: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, str]:
    atom_bond = {**{f"atom:{k}": v for k, v in enc["atom_encoder"].items()}, **{f"bond:{k}": v for k, v in enc["bond_encoder"].items()}}
    full = {
        **{f"atom:{k}": v for k, v in enc["atom_encoder"].items()},
        **{f"bond:{k}": v for k, v in enc["bond_encoder"].items()},
        **{f"junc:{k}": v for k, v in enc["junction_encoder"].items()},
    }
    return {"atom_bond_group_sha256": bp.state_group_sha256(atom_bond), "full_encoder_group_sha256": bp.state_group_sha256(full)}


# ---------------------------------------------------------------------------
# smoke (post-freeze, Fit labels only)
# ---------------------------------------------------------------------------


def cmd_smoke(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    device = torch.device(args.device)
    split, ds, guard = build_guard()
    fit_objs = load_zone_objects("fit")
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
    torch.save({g: {k: v.cpu() for k, v in s.items()} for g, s in enc.items()}, smoke_dir() / "ssl_s1_smoke.pt")
    out["ssl"] = {"updates": 20, "wall_seconds": res["wall_seconds"], **_ssl_encoder_hashes(enc)}

    for arm in ARMS:
        pairs_fit = load_zone_pairs("fit", arm) if arm != "P3" else None
        if pairs_fit is not None:
            pairs_fit = _slice_pair_set(pairs_fit, 0, len(mini))
        model = build_arm_model(arm, 1, norm, enc, device)
        info = pb.train_supervised_bridge(
            arm=arm,
            seed=1,
            model=model,
            pairs=pairs_fit,
            train_objs=mini,
            y_fit=y_mini,
            device=device,
            recipe_key="R3",
            epochs=1,
            log_fn=log,
        )
        pred = pb.predict_model(model, mini, pairs_fit, device)
        out["arms"][arm] = {
            "parameters": int(info["parameter_audit"]["actual_parameters"]),
            "fit_mae_after_1_epoch": float(np.abs(pred - y_mini).mean()),
            "wall_seconds": info["wall_seconds"],
            "rel_scale_abs_mean": info["bridge"].get("rel_scale_abs_mean"),
            "bridge": info["bridge"],
        }
        log(f"[smoke {arm}] params={out['arms'][arm]['parameters']} mae={out['arms'][arm]['fit_mae_after_1_epoch']:.4f}")
    out["device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    require(guard.access_attempts == [], "guarded label access during smoke training")
    write_json(smoke_dir() / "smoke_F.json", out)
    log(f"[smoke] ok device={out['device_name']}")
    return 0


def _slice_pair_set(pset: pb.PairSet, start: int, stop: int) -> pb.PairSet:
    state = pset.to_state()
    pairs = []
    n_bonds = []
    for i in range(start, stop):
        sl = slice(int(state["starts"][i]), int(state["ends"][i]))
        pairs.append({k: state[k][sl] for k in ("src", "dst", "w_src", "w_dst")})
        n_bonds.append(int(state["n_bonds"][i]))
    return pb.PairSet(pairs, n_bonds)


# ---------------------------------------------------------------------------
# ssl
# ---------------------------------------------------------------------------


def cmd_ssl(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    seed = int(args.seed)
    smoke_updates = int(args.smoke_updates) if args.smoke_updates else None
    if smoke_updates is None and (ssl_path(seed).exists() or ssl_json_path(seed).exists()):
        raise SystemExit(f"SSL for seed {seed} already exists; retraining is not allowed")
    split, ds, guard = build_guard()
    objs = load_zone_objects("fit")
    require([o.mol_index for o in objs] == [int(i) for i in split["fit_rows"]], "fit objects do not match split")
    log(f"[ssl s{seed}] device={args.device} n={len(objs)} updates={smoke_updates or 7560}")
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
    out_pt = ssl_path(seed) if smoke_updates is None else smoke_dir() / f"ssl_F_s{seed}_smoke.pt"
    torch.save({g: {k: v.cpu() for k, v in s.items()} for g, s in enc.items()}, out_pt)
    hashes = _ssl_encoder_hashes(enc)
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
        write_json(smoke_dir() / f"ssl_F_s{seed}_smoke.json", res)
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
    require([o.mol_index for o in fit_objs] == [int(i) for i in split["fit_rows"]], "fit objects do not match split")
    y_fit = load_fit_y()
    _, ds, guard = build_guard()
    out: dict[str, Any] = {"protocol": PROTOCOL_ID, "seed": seed, "device": str(device), "arms": {}}
    for arm in arms:
        if train_json_path(arm, seed).exists() and not args.retrain:
            raise SystemExit(f"{arm} seed {seed} already trained; refusing to retrain")
        pairs = load_zone_pairs("fit", arm) if arm != "P3" else None
        model = build_arm_model(arm, seed, norm, ssl, device)
        init_base = src.state_dict_hash(model) if arm == "P3" else src.state_dict_hash(model.base)
        info = pb.train_supervised_bridge(
            arm=arm,
            seed=seed,
            model=model,
            pairs=pairs,
            train_objs=fit_objs,
            y_fit=y_fit,
            device=device,
            recipe_key="R3",
            log_fn=log,
        )
        require(guard.access_attempts == [], "guarded label access during training")
        soup = info.pop("soup_state")
        torch.save(soup, soup_path(arm, seed))
        info.update(
            {
                "protocol": PROTOCOL_ID,
                "split_tag": SPLIT_TAG,
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "init_encoder_sha256": bp.encoder_group_hash(model if arm == "P3" else model.base),
                "init_base_state_sha256": init_base,
                "ssl_full_encoder_group_sha256": ssl.get("full_encoder_group_sha256"),
                "soup_file": str(soup_path(arm, seed).relative_to(REPO_ROOT)),
                "soup_sha256": file_sha256(soup_path(arm, seed)),
                "n_pairs_per_molecule": float(pairs.n_pairs / max(len(fit_objs), 1)) if pairs is not None else 0.0,
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
            "rel_scale_abs_mean": info["bridge"].get("rel_scale_abs_mean"),
        }
        _append_ledger(
            {"phase": "train", "arm": arm, "seed": seed, "wall_seconds": info["wall_seconds"], "device": str(device)}
        )
        log(
            f"[train {arm} s{seed}] fit_mae={info['soup_fit_mae']:.6f} wall={info['wall_seconds']:.0f}s "
            f"params={info['parameter_audit']['actual_parameters']}"
        )
    write_json(OUT / f"train_round_F_s{seed}.json", out)
    return 0


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def _load_soup_model(arm: str, seed: int, norm, device):
    model = build_arm_model(arm, seed, norm, None, device)
    state = torch.load(soup_path(arm, seed), map_location="cpu", weights_only=False)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.eval()
    return model


def _predict_zone(model, zone: str, arm: str, seed: int, device, kernel_arm: str | None = None) -> dict[str, Any]:
    objs = load_zone_objects(zone)
    pairs = None if arm == "P3" else load_zone_pairs(zone, kernel_arm or arm)
    pred = pb.predict_model(model, objs, pairs, device)
    require(pred.shape == (len(objs),), f"{zone}/{arm}: prediction shape {pred.shape}")
    require(bool(np.all(np.isfinite(pred))), f"{zone}/{arm}: non-finite prediction")
    target = pred_path(zone, arm, seed)
    if kernel_arm is not None and kernel_arm != arm:
        target = cross_pred_path(zone, arm, pb.KERNEL_OF_ARM[kernel_arm], seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, pred=pred.astype(np.float64), rows=np.asarray([o.mol_index for o in objs], dtype=np.int64))
    record = {
        "file": target.name,
        "file_sha256": file_sha256(target),
        "zone": zone,
        "arm": arm,
        "kernel_arm": kernel_arm or arm,
        "seed": int(seed),
        "n": int(len(pred)),
        "pred_sha256": array_sha256(pred),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "rows_sha256": rows_sha256([int(o.mol_index) for o in objs]),
        "frozen_utc": now_utc(),
    }
    with _file_lock(OUT / "prediction_manifest_F.json"):
        manifest = (
            read_json(OUT / "prediction_manifest_F.json")
            if (OUT / "prediction_manifest_F.json").exists()
            else {"protocol": PROTOCOL_ID, "records": {}, "complete": False}
        )
        manifest["records"][target.name] = record
        write_json(OUT / "prediction_manifest_F.json", manifest)
    return record


def cmd_predict(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    seed = int(args.seed)
    device = torch.device(args.device)
    norm = load_norm()
    models = {arm: _load_soup_model(arm, seed, norm, device) for arm in ARMS}
    for zone in ZONES:
        for arm in ARMS:
            rec = _predict_zone(models[arm], zone, arm, seed, device)
            log(f"[predict {zone} {arm} s{seed}] sha={rec['pred_sha256'][:12]} mean={rec['pred_mean']:.4f}")
    return 0


def cmd_predict_cross(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    seed = int(args.seed)
    device = torch.device(args.device)
    norm = load_norm()
    for model_arm, kernel_arm in (("P3R", "P3U"), ("P3U", "P3R")):
        model = _load_soup_model(model_arm, seed, norm, device)
        for zone in ZONES:
            rec = _predict_zone(model, zone, model_arm, seed, device, kernel_arm=kernel_arm)
            log(f"[predict-cross {zone} m{model_arm} k{kernel_arm} s{seed}] sha={rec['pred_sha256'][:12]}")
    return 0


def cmd_freeze_predictions(_args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    manifest = read_json(OUT / "prediction_manifest_F.json")
    expected = {pred_path(z, a, s).name for z in ZONES for a in ARMS for s in SEEDS}
    expected |= {
        cross_pred_path(z, m, pb.KERNEL_OF_ARM[k], s).name
        for z in ZONES
        for m, k in (("P3R", "P3U"), ("P3U", "P3R"))
        for s in SEEDS
    }
    require(set(manifest["records"]) == expected, f"prediction set incomplete: {sorted(expected - set(manifest['records']))}")
    hashes = []
    for name in sorted(manifest["records"]):
        path = OUT / "predictions" / name
        require(file_sha256(path) == manifest["records"][name]["file_sha256"], f"prediction drifted: {name}")
        hashes.append(f"{name}:{manifest['records'][name]['file_sha256']}")
    combined = hashlib.sha256("\n".join(hashes).encode()).hexdigest()
    out = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "n_files": len(hashes),
        "combined_pred_sha256": combined,
        "records": manifest["records"],
        "frozen_utc": now_utc(),
        "labels_read_after_this_point": True,
    }
    write_json(OUT / "prediction_manifest_F.json", {**out, "complete": True})
    log(f"[freeze-predictions] {len(hashes)} files combined={combined[:16]}")
    return 0


# ---------------------------------------------------------------------------
# score holdout
# ---------------------------------------------------------------------------


def _load_preds(zone: str, arm: str) -> dict[int, np.ndarray]:
    return {seed: np.asarray(np.load(pred_path(zone, arm, seed))["pred"], dtype=np.float64) for seed in SEEDS}


def _load_cross(zone: str, model_arm: str, kernel_arm: str) -> dict[int, np.ndarray]:
    return {
        seed: np.asarray(np.load(cross_pred_path(zone, model_arm, pb.KERNEL_OF_ARM[kernel_arm], seed))["pred"], dtype=np.float64)
        for seed in SEEDS
    }


def _arm_stats(y: np.ndarray, preds: Mapping[int, np.ndarray]) -> dict[str, Any]:
    err = {s: np.abs(preds[s] - y) for s in SEEDS}
    seed_mean = np.mean([err[s] for s in SEEDS], axis=0)
    sse = seed_mean**2
    order = np.argsort(-seed_mean)
    ordinal = y > pb.TAIL_THRESHOLD
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


def _contrast(y: np.ndarray, a: Mapping[int, np.ndarray], b: Mapping[int, np.ndarray]) -> dict[str, Any]:
    base = ab.compute_contrast(a, b, y, bootstrap_n=pb.BOOTSTRAP_N, bootstrap_seed=pb.BOOTSTRAP_SEED)
    label = pb.contrast_label(
        base["delta_seed_mean"],
        base["per_seed_delta"],
        base["bootstrap"]["ci_lo"],
        base["bootstrap"]["ci_hi"],
        base["ordinary_delta"],
    )
    per_seed = {}
    for seed in SEEDS:
        one = ab.compute_contrast({seed: a[seed]}, {seed: b[seed]}, y, bootstrap_n=pb.BOOTSTRAP_N, bootstrap_seed=pb.BOOTSTRAP_SEED)
        per_seed[str(seed)] = {
            "delta": one["delta_seed_mean"],
            "ci_lo": one["bootstrap"]["ci_lo"],
            "ci_hi": one["bootstrap"]["ci_hi"],
            "mae_a": one["mae_new_seed_mean"],
            "mae_b": one["mae_ref_seed_mean"],
        }
    return {**base, "label": label, "per_seed_bootstrap": per_seed}


CONTRASTS = pb.CONTRASTS


def cmd_score_holdout(_args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    lock = OUT / "holdout_read.lock"
    require(not lock.exists(), "held-out labels were already read in this round")
    manifest = read_json(OUT / "prediction_manifest_F.json")
    require(bool(manifest.get("complete")), "prediction manifest is not complete")
    for name, rec in manifest["records"].items():
        require(file_sha256(OUT / "predictions" / name) == rec["file_sha256"], f"prediction drifted: {name}")

    split = load_split()
    y = {zone: bp.load_y_rows(ENCODED_TRAIN, [int(i) for i in split[f"{zone}_rows"]]) for zone in ZONES}
    np.savez(OUT / "holdout_labels_F.npz", monitor=y["monitor"], locked=y["locked"])
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
        "primary_zone": pb.PRIMARY_ZONE,
        "replication_zone": pb.REPLICATION_ZONE,
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

    primary = result["zones"][pb.PRIMARY_ZONE]["contrasts"]
    replication = result["zones"][pb.REPLICATION_ZONE]["contrasts"]
    rep = {
        name: pb.replication_status(replication[name]["per_seed_delta"]) for name, _, _ in CONTRASTS
    }
    case = pb.bridge_case(
        label_r_p3=primary["P3R_minus_P3"]["label"],
        label_r_u=primary["P3R_minus_P3U"]["label"],
        label_u_p3=primary["P3U_minus_P3"]["label"],
        replicated_r_p3=rep["P3R_minus_P3"]["replicated"],
        replicated_r_u=rep["P3R_minus_P3U"]["replicated"],
        replicated_u_p3=rep["P3U_minus_P3"]["replicated"],
    )
    result["replication"] = rep
    result["decision"] = case
    write_json(OUT / "holdout_score_F.json", result)
    write_json(
        OUT / "decision.json",
        {
            "protocol": PROTOCOL_ID,
            "git_commit": git_commit(),
            "split_tag": SPLIT_TAG,
            **case,
            "primary_zone": pb.PRIMARY_ZONE,
            "primary_labels": {k: v["label"] for k, v in primary.items()},
            "primary_deltas": {k: v["delta_seed_mean"] for k, v in primary.items()},
            "primary_ci": {k: [v["bootstrap"]["ci_lo"], v["bootstrap"]["ci_hi"]] for k, v in primary.items()},
            "replication_deltas": {k: v["delta_seed_mean"] for k, v in replication.items()},
            "replication_flags": {k: v["replicated"] for k, v in rep.items()},
            "frozen_utc": now_utc(),
        },
    )
    log(
        "[score-holdout] locked "
        + " ".join(f"{arm}={result['zones']['locked']['arm_stats'][arm]['mae_seed_mean']:.6f}" for arm in ARMS)
        + f" | R-P3={primary['P3R_minus_P3']['delta_seed_mean']:+.6f} R-U={primary['P3R_minus_P3U']['delta_seed_mean']:+.6f}"
        + f" U-P3={primary['P3U_minus_P3']['delta_seed_mean']:+.6f} -> {case['case']}"
    )
    return 0


# ---------------------------------------------------------------------------
# post hoc (frozen models, no new label read)
# ---------------------------------------------------------------------------


def _effect_probe(arm: str, seed: int, norm, device, n: int = 128) -> dict[str, Any]:
    model = _load_soup_model(arm, seed, norm, device)
    objs = load_zone_objects("fit")[:n]
    pairs = load_zone_pairs("fit", arm)
    pairs = _slice_pair_set(pairs, 0, n)
    batch = _batch(objs, device)
    rel = pairs.batch(list(range(n)), device)
    with torch.no_grad():
        feats = model.forward_features(batch, rel)
        c, agg, scale = feats["c"], feats["bridge_agg"], model.rel_scale.unsqueeze(0)
        delta = scale * agg
        c_norm = c.norm(dim=1)
        d_norm = delta.norm(dim=1)
        cos = torch.nn.functional.cosine_similarity(delta, c, dim=1)
    return {
        "arm": arm,
        "seed": int(seed),
        "mean_abs_c": float(c.abs().mean()),
        "mean_abs_residual": float(delta.abs().mean()),
        "relative_residual": float(delta.abs().mean() / c.abs().mean()),
        "mean_cosine": float(cos.mean()),
        "frac_bonds_residual_above_1pct_of_c": float((d_norm > 0.01 * c_norm).float().mean()),
        "rel_scale_abs_mean": float(model.rel_scale.abs().mean()),
    }


def cmd_post_hoc(_args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    require((OUT / "holdout_read.lock").exists(), "score the frozen predictions first")
    device = torch.device("cpu")
    norm = load_norm()
    score = read_json(OUT / "holdout_score_F.json")
    labels = np.load(OUT / "holdout_labels_F.npz")
    out: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "note": "diagnostics on frozen models/predictions; no label was re-read (labels from holdout_labels_F.npz)",
        "effects": [_effect_probe(arm, seed, norm, device) for arm in pb.BRIDGE_ARMS for seed in SEEDS],
        "kernel_swap": {},
        "regions": {},
        "delta_quantiles": {},
    }
    for zone in ZONES:
        y = labels[zone]
        swap = {}
        for model_arm, other in (("P3R", "P3U"), ("P3U", "P3R")):
            own = _load_preds(zone, model_arm)
            cross = _load_cross(zone, model_arm, other)
            d = np.mean([np.abs(cross[s] - own[s]) for s in SEEDS], axis=0)
            swap[f"{model_arm}_own_vs_{other}_kernel"] = {
                "mean_abs_pred_delta": float(d.mean()),
                "max_abs_pred_delta": float(d.max()),
                "mae_own": score["zones"][zone]["arm_stats"][model_arm]["mae_seed_mean"],
                "mae_swapped": float(np.mean([np.abs(cross[s] - y).mean() for s in SEEDS])),
            }
        out["kernel_swap"][zone] = swap
        region = {}
        for arm in ARMS:
            preds = _load_preds(zone, arm)
            err = np.mean([np.abs(preds[s] - y) for s in SEEDS], axis=0)
            for name, mask in (
                ("ordinary", y > pb.TAIL_THRESHOLD),
                ("tail", y <= pb.TAIL_THRESHOLD),
                ("top_decile_error", err >= np.quantile(err, 0.9)),
            ):
                region[f"{arm}_{name}"] = {"n": int(mask.sum()), "mae": float(err[mask].mean())}
        out["regions"][zone] = region
        quant = {}
        for name, a, b in CONTRASTS:
            preds_a, preds_b = _load_preds(zone, a), _load_preds(zone, b)
            delta = np.mean([np.abs(preds_a[s] - y) - np.abs(preds_b[s] - y) for s in SEEDS], axis=0)
            quant[name] = {
                "q10": float(np.quantile(delta, 0.10)),
                "q25": float(np.quantile(delta, 0.25)),
                "median": float(np.median(delta)),
                "q75": float(np.quantile(delta, 0.75)),
                "q90": float(np.quantile(delta, 0.90)),
                "frac_better": float((delta < 0).mean()),
                "frac_worse": float((delta > 0).mean()),
            }
        out["delta_quantiles"][zone] = quant
    write_json(OUT / "post_hoc_bridge.json", out)
    log("[post-hoc] effects/regions/quantiles written")
    return 0


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------


def cmd_finalize(_args: argparse.Namespace) -> int:
    fr = verify_freeze(require_exact_code=False)
    artifacts = []
    total_bytes = 0
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name == "evidence_manifest.json":
            continue
        if path.suffix in (".pt", ".npz", ".csv") or path.name.endswith(".json") or path.suffix == ".md":
            rec = {"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            artifacts.append(rec)
            total_bytes += rec["bytes"]
    ledger = read_json(ledger_path()) if ledger_path().exists() else {"total_wall_hours": 0.0, "entries": []}
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "freeze_sha256": file_sha256(freeze_path()),
        "freeze_amendment": fr.get("amendment", 0),
        "n_artifacts": len(artifacts),
        "total_bytes": total_bytes,
        "artifacts": artifacts,
        "gpu_budget": {
            "total_wall_hours": ledger.get("total_wall_hours", 0.0),
            "cap_hours": GPU_BUDGET_CAP_HOURS,
            "within_cap": bool(ledger.get("total_wall_hours", 0.0) <= GPU_BUDGET_CAP_HOURS),
            "entries": ledger.get("entries", []),
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "finalized_utc": now_utc(),
    }
    write_json(OUT / "evidence_manifest.json", payload)
    log(f"[finalize] {len(artifacts)} artifacts {total_bytes/1e6:.1f} MB, gpu {payload['gpu_budget']['total_wall_hours']:.4f} h")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in (
        "split",
        "verify-inputs",
        "objects",
        "kernels",
        "stage-inputs",
        "cpu-gates",
        "freeze",
        "smoke",
        "ssl",
        "train",
        "predict",
        "predict-cross",
        "freeze-predictions",
        "score-holdout",
        "post-hoc",
        "finalize",
    ):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--arm", type=str, default=None, choices=list(ARMS))
        p.add_argument("--device", type=str, default="cpu")
        p.add_argument("--smoke-updates", type=int, default=None)
        p.add_argument("--amendment", action="store_true")
        p.add_argument("--retrain", action="store_true")
    args = parser.parse_args(argv)
    fn = {
        "split": cmd_split,
        "verify-inputs": cmd_verify_inputs,
        "objects": cmd_objects,
        "kernels": cmd_kernels,
        "stage-inputs": cmd_stage_inputs,
        "cpu-gates": cmd_cpu_gates,
        "freeze": cmd_freeze,
        "smoke": cmd_smoke,
        "ssl": cmd_ssl,
        "train": cmd_train,
        "predict": cmd_predict,
        "predict-cross": cmd_predict_cross,
        "freeze-predictions": cmd_freeze_predictions,
        "score-holdout": cmd_score_holdout,
        "post-hoc": cmd_post_hoc,
        "finalize": cmd_finalize,
    }[args.cmd]
    return int(fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
