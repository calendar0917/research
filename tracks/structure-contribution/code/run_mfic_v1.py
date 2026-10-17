#!/usr/bin/env python
"""Driver for protocol ``cscl-bjoc-mfic-v1`` (MFIC: Matrix-Function Incidence Composition).

CPU stages
----------
``split``          identity-grouped split V with explicit salt, hash replay,
                   salt probe, and overlap matrix vs E/F/S/T/U.
``verify-inputs``  frozen upstream hashes, split sanity, parameter audit,
                   official-loader scan.
``objects``        label-blind Fit/Monitor/Locked objects, Fit-only global
                   standardiser, cached K_R / K_U kernels.
``cpu-signal``     four-fold grouped Ridge probe on Fit only (CPU-Base vs
                   CPU-Matrix); no model predictions, no Monitor/Locked labels.
``math-gates``     K/R/K_U analytic and numerical gates.
``model-audit``    parameter, gradient, consumption, invariance audit.
``cpu-gate``       frozen purchase-gate evaluation.
``freeze-pre-gpu`` only after a CPU pass: pre-GPU protocol freeze.
``finalize-cpu``   evidence manifest and CPU status.

GPU stages are intentionally not implemented until the CPU gate passes; the
protocol text freezes their design, and this module will be extended only
after a CPU pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
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
import bond_relation_attribution_v1 as bra  # noqa: E402
import mfic_v1 as mf  # noqa: E402
import p3_slim_v1 as sl  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

REPO_ROOT = _CODE_DIR.parents[2]
TRACK = _CODE_DIR.parent
OUT = Path(os.environ.get("MFIC_OUT_DIR", str(TRACK / "results" / "mfic_v1")))
CONFIG_PATH = TRACK / "configs" / "mfic_v1.yaml"
PROTOCOL_DOC = OUT / "PROTOCOL.md"
ZINC_ROOT = REPO_ROOT / "data/ZINC"
DATA_TRAIN = ZINC_ROOT / "subset/processed/train.pt"
CANON = REPO_ROOT / "tracks/ksvd/results/zinc_cssd_basis_reuse_v1/train_canonical_smiles.npz"
ENCODED_TRAIN = REPO_ROOT / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt"
STRUCT_CONTEXT_CODE = TRACK / "results/structure_context_support_v1/code"

P3_RESULTS = TRACK / "results" / "p3_bond_relation_bridge_v1"
SAC_RESULTS = TRACK / "results" / "sac_v1"
BRA_RESULTS = TRACK / "results" / "bond_relation_attribution_v1"
SAM_RESULTS = TRACK / "results" / "p3_sam_v1"
SLIM_RESULTS = TRACK / "results" / "p3_slim_v1"

PROTOCOL_ID = mf.PROTOCOL_ID
SPLIT_TAG = mf.SPLIT_TAG
SPLIT_SALT = mf.SPLIT_SALT
SEEDS = mf.SEEDS
ARMS = mf.ARMS
ZONES = ("fit", "monitor", "locked")

FROZEN_INPUT_HASHES = {
    "train_pt": "e0f1837b4a19714fcf3a5c6727a30c23f57c3398c8987cab2983214a0e76955b",
    "canonical_smiles": "04a590355fb7bba71e35e52bed60c974999b92af9e96e4ee99bed4afe42dcaff",
    "encoded_train": "10d764955ef1b040a317105be1dd3fbb2e81292b03b79e7d078cce08e18e5f1d",
}
HISTORICAL_SPLITS = {
    "E": {"salt": "cscl-bond-relation-attribution-v1", "path": BRA_RESULTS / "split_E.json"},
    "F": {"salt": "cscl-p3-bond-relation-bridge-v1", "path": P3_RESULTS / "split_F.json"},
    "S": {"salt": "cscl-bjoc-sac-v1", "path": SAC_RESULTS / "split_S.json"},
    "T": {"salt": "cscl-p3-sam-v1", "path": SAM_RESULTS / "split_T.json"},
    "U": {"salt": "cscl-p3-slim-v1", "path": SLIM_RESULTS / "split_U.json"},
}
FIT_TARGET = 7_996
MONITOR_TARGET = 1_000
LOCKED_TARGET = 1_000


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rows_sha256(rows: Sequence[int]) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()


def array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def now_utc() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def config() -> Mapping[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def config_sha256() -> str:
    return file_sha256(CONFIG_PATH)


def code_hashes() -> dict[str, str]:
    return {name: file_sha256(_CODE_DIR / name) for name in ("mfic_v1.py", "run_mfic_v1.py")}


def split_sha256(payload: Mapping[str, Any]) -> str:
    blob = {
        "salt": payload["salt"],
        "fit_rows": [int(r) for r in payload["fit_rows"]],
        "monitor_rows": [int(r) for r in payload["monitor_rows"]],
        "locked_rows": [int(r) for r in payload["locked_rows"]],
    }
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()


def object_zone_sha256(objs: Sequence[sro.MolObjects]) -> str:
    h = hashlib.sha256()
    for o in objs:
        h.update(int(o.mol_index).to_bytes(8, "little", signed=False))
        for name in ("atom_feat", "bond_index", "bond_type", "bond_feat", "junc_index", "junc_feat", "global_feat"):
            a = np.ascontiguousarray(getattr(o, name))
            h.update(name.encode())
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
    return h.hexdigest()


def kernel_digest(kernels: Mapping[str, Sequence[torch.Tensor]]) -> dict[str, str]:
    out = {}
    for mode in ("R", "U"):
        h = hashlib.sha256()
        for t in kernels[mode]:
            a = np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32)
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        out[mode] = h.hexdigest()
    return out


def split_path() -> Path:
    return OUT / "split_V.json"


def objects_path(zone: str) -> Path:
    return OUT / f"objects_{zone}_V.pt"


def kernels_path(zone: str) -> Path:
    return OUT / f"kernels_{zone}_V.pt"


def prep_path() -> Path:
    return OUT / "prep_fit_V.json"


def cpu_signal_path() -> Path:
    return OUT / "cpu_signal.json"


def cpu_gate_path() -> Path:
    return OUT / "cpu_gate.json"


def decision_path() -> Path:
    return OUT / "decision_cpu.json"


def math_gates_path() -> Path:
    return OUT / "math_gates.json"


def model_audit_path() -> Path:
    return OUT / "model_audit.json"


def input_audit_path() -> Path:
    return OUT / "input_audit_V.json"


def freeze_path() -> Path:
    return OUT / "protocol_freeze.json"


def load_split() -> Mapping[str, Any]:
    require(split_path().exists(), "split_V.json missing: run `split` first")
    return read_json(split_path())


def load_zone_objects(zone: str) -> list[sro.MolObjects]:
    require(objects_path(zone).exists(), f"objects_{zone}_V.pt missing")
    return torch.load(objects_path(zone), map_location="cpu", weights_only=False)


def load_kernels(zone: str) -> dict[str, list[torch.Tensor]]:
    require(kernels_path(zone).exists(), f"kernels_{zone}_V.pt missing")
    return torch.load(kernels_path(zone), map_location="cpu", weights_only=False)


def load_fit_y() -> np.ndarray:
    split = load_split()
    rows = [int(r) for r in split["fit_rows"]]
    return bp.load_y_rows(ENCODED_TRAIN, rows)


def load_official_train():
    import cscl_features as cf

    return cf.load_official_train()


def build_guard():
    split = load_split()
    ds = load_official_train()
    guard = ab.LabelGuard(ds, list(split["monitor_rows"]) + list(split["locked_rows"]))
    return split, ds, guard


def require(cond: Any, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


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
    for za in ZONES:
        sa = set(int(i) for i in a[za])
        out[za] = {}
        for zb in ZONES:
            sb = set(int(i) for i in b[zb])
            inter = len(sa & sb)
            out[za][zb] = {
                "n_overlap": int(inter),
                "frac_of_this_zone": float(inter / max(len(sa), 1)),
                "frac_of_other_zone": float(inter / max(len(sb), 1)),
            }
    return out


def cmd_split(args: argparse.Namespace) -> int:
    if split_path().exists() and not args.force:
        raise SystemExit("split_V.json already exists; refusing to rebuild without --force")
    from rdkit import Chem

    scs = _structure_context_module()
    pairing = scs.pairing_verification()
    write_json(OUT / "pairing_verification.json", pairing)
    require(pairing["n_mismatch"] == 4, f"expected 4 SMILES/PyG mismatches, got {pairing['n_mismatch']}")
    require(tuple(pairing["mismatch_rows"]) == tuple(bra.EXCLUDED_ROWS), "mismatch rows changed")
    require(pairing["reproduces_frozen_exclusion_list"], "exclusion list does not reproduce")
    rows = [int(r) for r in range(10_000) if int(r) not in set(bra.EXCLUDED_ROWS)]
    require(len(rows) == bra.N_USABLE, "usable row count changed")

    structs = scs.load_pyg_structure_only(rows)
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

    used_salt = SPLIT_SALT
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
        for z in ZONES
    )
    require(salt_reproduced, "recomputed split from the saved salt/identity does not match the saved rows")

    own_hashes = {z: rows_sha256(split[f"{z}_rows"]) for z in ZONES}
    payload_for_hash = {
        "salt": used_salt,
        "fit_rows": split["fit_rows"],
        "monitor_rows": split["monitor_rows"],
        "locked_rows": split["locked_rows"],
    }
    actual_hash = split_sha256(payload_for_hash)
    require(actual_hash == split_sha256({**payload_for_hash}), "split hash is not deterministic")

    probe = bra.build_identity_split(
        rows,
        identity_keys,
        salt=used_salt + "-probe",
        fit_target=FIT_TARGET,
        monitor_target=MONITOR_TARGET,
    )
    require(rows_sha256(probe["fit_rows"]) != own_hashes["fit"], "salt probe did not change the split")

    new_sets = {z: [int(i) for i in split[f"{z}_rows"]] for z in ZONES}
    historical: dict[str, Any] = {}
    old_row_sets: dict[str, dict[str, Sequence[int]]] = {}
    for tag, spec in HISTORICAL_SPLITS.items():
        old_path = spec["path"]
        require(old_path.exists(), f"historical split {tag} missing at {old_path}")
        old = read_json(old_path)
        old_rows = {z: [int(i) for i in old[f"{z}_rows"]] for z in ZONES}
        old_hashes = {z: rows_sha256(old_rows[z]) for z in ZONES}
        collisions = {z: bool(own_hashes[z] == old_hashes[z]) for z in own_hashes}
        identical = all(sorted(new_sets[z]) == sorted(old_rows[z]) for z in ZONES)
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
        "salt_source": "explicit argument mfic_v1.SPLIT_SALT (never a builder default)",
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
            "The same official-train 10k has been used by many prior rounds; split V is a new "
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
    require(cfg["salt"] == SPLIT_SALT, "config salt mismatch")
    require(tuple(cfg["seeds"]) == SEEDS, "config seeds mismatch")
    require(tuple(cfg["arms"]) == ARMS, "config arms mismatch")
    require(int(cfg["alpha"] * 1000) == int(mf.ALPHA * 1000), "config alpha mismatch")
    require(int(cfg["q_dim"]) == mf.Q_DIM, "config q_dim mismatch")
    require(int(cfg["reader_hidden"]) == mf.READER_HIDDEN, "config reader hidden mismatch")
    require(int(cfg["reader_layers"]) == mf.READER_LAYERS, "config reader layers mismatch")
    require(int(cfg["reader_in"]) == mf.READER_IN, "config reader input mismatch")
    require(int(cfg["model"]["p3_parameters"]) == mf.P3_PARAMETERS, "config P3 params mismatch")
    require(int(cfg["model"]["mfic_parameters"]) == mf.MFIC_PARAMETERS, "config MFIC params mismatch")
    require(int(cfg["model"]["mfic_reader_parameters"]) == mf.MFIC_READER_PARAMETERS, "config MFIC reader params mismatch")
    require(int(cfg["model"]["q_projection_parameters"]) == mf.Q_PROJECTION_PARAMETERS, "config q projection params mismatch")
    require(int(cfg["param_cap"]) == mf.PARAM_CAP, "config parameter cap mismatch")
    require(int(cfg["cpu"]["folds"]) == mf.CPU_FOLDS, "config CPU folds mismatch")
    require(int(cfg["cpu"]["pca_components"]) == mf.CPU_PCA_COMPONENTS, "config PCA components mismatch")
    require(float(cfg["cpu"]["ridge_alpha"]) == mf.CPU_RIDGE_ALPHA, "config Ridge alpha mismatch")
    require(int(cfg["cpu"]["trim_k"]) == mf.CPU_TRIM_K, "config trim k mismatch")
    require(float(cfg["cpu"]["mean_improvement_min"]) == mf.CPU_MIN_MEAN_IMPROVEMENT, "config CPU threshold mismatch")
    require(int(cfg["cpu"]["folds_improved_min"]) == mf.CPU_MIN_FOLDS_IMPROVED, "config CPU fold threshold mismatch")
    require(float(cfg["gate"]["meaningful_delta"]) == mf.MEANINGFUL_DELTA, "config R-P3 threshold mismatch")
    require(float(cfg["gate"]["null_delta"]) == mf.NULL_DELTA, "config R-U threshold mismatch")
    require(float(cfg["budget"]["gpu_hours"]) == mf.GPU_BUDGET_CAP_HOURS, "config GPU budget mismatch")
    require(float(cfg["budget"]["smoke_stop_hours"]) == mf.SMOKE_STOP_HOURS, "config smoke stop mismatch")

    split = load_split()
    require(split["split_sha256"] == split_sha256(split), "split hash mismatch")
    require(split["salt"] == SPLIT_SALT, "split salt mismatch")
    sizes = split["sizes"]
    require(abs(int(sizes["fit"]) - FIT_TARGET) <= 10, f"split fit size off target: {sizes}")
    require(abs(int(sizes["monitor"]) - MONITOR_TARGET) <= 10, f"split monitor size off target: {sizes}")
    require(abs(int(sizes["locked"]) - LOCKED_TARGET) <= 10, f"split locked size off target: {sizes}")

    inv = mf.parameter_inventory()
    require(inv["all_expected"], f"parameter inventory failed: {inv}")

    scan_tokens = [
        "split=" + q + name + q
        for q in ('"', "'")
        for name in ("valid", "test")
    ] + ["processed/" + name for name in ("valid", "test")]
    scan_hits = []
    for path in (_CODE_DIR / "mfic_v1.py", _CODE_DIR / "run_mfic_v1.py"):
        text = path.read_text(encoding="utf-8")
        for token in scan_tokens:
            if token in text:
                scan_hits.append(f"{path.name}:{token}")
    require(not scan_hits, f"official valid/test loader token found: {scan_hits}")

    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "input_hashes": actual,
        "input_hashes_match_frozen": True,
        "split_sha256": split["split_sha256"],
        "split_sizes": split["sizes"],
        "parameter_inventory": inv,
        "official_loader_scan_hits": scan_hits,
        "label_guard_selfcheck": ab.label_guard_selfcheck(),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(input_audit_path(), payload)
    log(
        f"[verify-inputs] P3={mf.P3_PARAMETERS} MFIC={mf.MFIC_PARAMETERS} "
        f"split={split['sizes']} hash={split['split_sha256'][:16]}"
    )
    return 0


# ---------------------------------------------------------------------------
# objects + K cache
# ---------------------------------------------------------------------------


def cmd_objects(_args: argparse.Namespace) -> int:
    require(not freeze_path().exists(), "protocol is frozen already; refusing to rebuild objects")
    split, ds, guard = build_guard()
    started = time.perf_counter()
    objs: dict[str, list[sro.MolObjects]] = {}
    kernels: dict[str, dict[str, list[torch.Tensor]]] = {}
    for zone in ZONES:
        rows = [int(i) for i in split[f"{zone}_rows"]]
        built = []
        for i in rows:
            d = guard[int(i)]
            built.append(sro.build_mol_objects(int(i), d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy()))
        require([o.mol_index for o in built] == rows, f"object row order mismatch in {zone}")
        torch.save(built, objects_path(zone))
        kr = []
        ku = []
        for o in built:
            K = mf.kernel_from_adjacency(mf.physical_adjacency(o))["K"]
            kr.append(torch.as_tensor(K, dtype=torch.float32))
            ku.append(torch.as_tensor(mf.content_null_kernel(K), dtype=torch.float32))
        torch.save({"R": kr, "U": ku, "alpha": mf.ALPHA}, kernels_path(zone))
        objs[zone] = built
        kernels[zone] = {"R": kr, "U": ku}
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
                "kernels_sha256": file_sha256(kernels_path(zone)),
                "kernel_digest": kernel_digest(kernels[zone]),
                "kernel_stats": {
                    "R_finite": bool(all(torch.isfinite(t).all().item() for t in kernels[zone]["R"])),
                    "U_finite": bool(all(torch.isfinite(t).all().item() for t in kernels[zone]["U"])),
                    "R_symmetric": bool(all(torch.allclose(t, t.T, atol=0.0) for t in kernels[zone]["R"])),
                    "U_symmetric": bool(all(torch.allclose(t, t.T, atol=0.0) for t in kernels[zone]["U"])),
                },
            }
            for zone in ZONES
        },
        "prep_fit_sha256": file_sha256(prep_path()),
        "guarded_rows": len(split["monitor_rows"]) + len(split["locked_rows"]),
        "forbidden_label_access_attempts": len(guard.access_attempts),
        "monitor_labels_read": False,
        "locked_labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "objects_manifest_V.json", payload)
    log(f"[objects] built in {payload['build_seconds']:.1f}s; standardiser fit-only; K cached")
    return 0


# ---------------------------------------------------------------------------
# CPU signal probe
# ---------------------------------------------------------------------------


def cmd_cpu_signal(_args: argparse.Namespace) -> int:
    require(not freeze_path().exists(), "protocol is frozen already; refusing to recompute CPU signal")
    split = load_split()
    fit_rows = [int(r) for r in split["fit_rows"]]
    objs = load_zone_objects("fit")
    require([o.mol_index for o in objs] == fit_rows, "Fit object rows mismatch")
    y = load_fit_y()
    require(y.shape == (len(fit_rows),), "Fit label length mismatch")
    require(np.all(np.isfinite(y)), "non-finite Fit labels")
    identity_of_row = split["identity_of_row"]
    folds = mf.grouped_folds(fit_rows, identity_of_row, n_folds=mf.CPU_FOLDS, salt=SPLIT_SALT)
    fold_payload = {
        "protocol": PROTOCOL_ID,
        "salt": f"{SPLIT_SALT}|fold",
        "n_folds": len(folds),
        "rule": "identity-grouped, sha256(salt|fold|identity), greedy smallest fold",
        "fold_sizes": [len(f) for f in folds],
        "fold_rows_sha256": [rows_sha256(f) for f in folds],
        "fold_groups_sha256": [
            hashlib.sha256(
                "\n".join(
                    sorted(str(identity_of_row.get(int(r), identity_of_row.get(str(r)))) for r in f)
                ).encode()
            ).hexdigest()
            for f in folds
        ],
        "folds": folds,
        "labels_read": False,
    }
    write_json(OUT / "cpu_folds_V.json", fold_payload)

    started = time.perf_counter()
    kernels = [mf.kernel_from_adjacency(mf.physical_adjacency(o))["K"] for o in objs]
    kernel_seconds = time.perf_counter() - started
    result = mf.cpu_ridge_signal(
        rows=fit_rows,
        objects=objs,
        y=y,
        identity_of_row=identity_of_row,
        kernels=kernels,
        folds=folds,
        n_folds=mf.CPU_FOLDS,
        alpha=mf.CPU_RIDGE_ALPHA,
        n_components=mf.CPU_PCA_COMPONENTS,
    )
    summary = result["summary"]
    summary["kernel_build_seconds"] = float(kernel_seconds)
    summary["cpu_signal_wall_seconds"] = float(time.perf_counter() - started)
    summary["fit_y_sha256"] = array_sha256(y)
    summary["fit_rows_sha256"] = rows_sha256(fit_rows)
    summary["encoded_train_sha256"] = file_sha256(ENCODED_TRAIN)
    summary["monitor_labels_read"] = False
    summary["locked_labels_read"] = False
    summary["official_valid_loaded"] = False
    summary["official_test_loaded"] = False
    write_json(cpu_signal_path(), {"summary": summary, "folds": result["folds"]})

    arrays = result["arrays"]
    np.savez_compressed(
        OUT / "cpu_signal_predictions.npz",
        **{k: np.concatenate(v) for k, v in arrays.items()},
    )
    # Label-access log: only Fit rows are passed to the label loader.
    write_json(
        OUT / "cpu_label_access.json",
        {
            "protocol": PROTOCOL_ID,
            "label_source": str(ENCODED_TRAIN.relative_to(REPO_ROOT)),
            "label_source_sha256": file_sha256(ENCODED_TRAIN),
            "fit_rows": len(fit_rows),
            "fit_rows_sha256": rows_sha256(fit_rows),
            "fit_y_sha256": array_sha256(y),
            "monitor_rows_passed_to_loader": 0,
            "locked_rows_passed_to_loader": 0,
            "monitor_labels_read": False,
            "locked_labels_read": False,
            "official_valid_loaded": False,
            "official_test_loaded": False,
            "model_predictions_used": False,
        },
    )
    log(
        f"[cpu-signal] base={summary['mean_mae_base']:.6f} matrix={summary['mean_mae_matrix']:.6f} "
        f"delta={summary['mean_delta_matrix_minus_base']:+.6f} folds={summary['fold_directions']} "
        f"trim={summary['mean_trimmed10_delta']:+.6f} ordinary={summary['mean_ordinary_delta']:+.6f}"
    )
    return 0


# ---------------------------------------------------------------------------
# CPU math and model gates
# ---------------------------------------------------------------------------


def _real_sample(max_n: int = 200) -> list[sro.MolObjects]:
    objs = load_zone_objects("fit")
    return list(objs[: int(max_n)])


def cmd_math_gates(_args: argparse.Namespace) -> int:
    require(not freeze_path().exists(), "protocol is frozen already; refusing to recompute math gates")
    result = mf.run_math_gates(_real_sample(200), max_real=200)
    write_json(math_gates_path(), result)
    for name, ok in result["checks"].items():
        if name == "sum_R_used_as_feature":
            log(f"[math-gate {name}] INFO (must be False: sum(R) is not used as a feature)")
        else:
            log(f"[math-gate {name}] {'PASS' if ok else 'FAIL'}")
    require(result["all_passed"], "math gates did not all pass")
    return 0


def cmd_model_audit(_args: argparse.Namespace) -> int:
    require(not freeze_path().exists(), "protocol is frozen already; refusing to recompute model audit")
    result = mf.run_model_audit(_real_sample(64), seed=0, n_objects=32)
    write_json(model_audit_path(), result)
    for name, ok in result["checks"].items():
        log(f"[model-audit {name}] {'PASS' if ok else 'FAIL'}")
    require(result["all_passed"], "model audit did not all pass")
    return 0


# ---------------------------------------------------------------------------
# CPU purchase gate
# ---------------------------------------------------------------------------


def cmd_cpu_gate(_args: argparse.Namespace) -> int:
    require(cpu_signal_path().exists(), "cpu_signal.json missing")
    require(math_gates_path().exists(), "math_gates.json missing")
    require(model_audit_path().exists(), "model_audit.json missing")
    require(input_audit_path().exists(), "input_audit_V.json missing")
    signal = read_json(cpu_signal_path())
    summary = signal["summary"]
    math_gates = read_json(math_gates_path())
    model_audit = read_json(model_audit_path())
    input_audit = read_json(input_audit_path())
    label_access = read_json(OUT / "cpu_label_access.json")

    conditions = {
        "C1_mean_mae_improvement_at_least_0.020": bool(
            summary["mean_mae_improvement"] >= mf.CPU_MIN_MEAN_IMPROVEMENT
        ),
        "C2_at_least_3_of_4_folds_negative": bool(
            summary["folds_negative"] >= mf.CPU_MIN_FOLDS_IMPROVED
        ),
        "C3_trimmed10_mean_direction_negative": bool(summary["mean_trimmed10_delta"] < 0.0),
        "C4_ordinary_mean_direction_negative": bool(summary["mean_ordinary_delta"] < 0.0),
        "C5_math_model_and_invariance_gates": bool(math_gates["all_passed"] and model_audit["all_passed"]),
        "C6_no_official_or_model_prediction_dependence": bool(
            not summary["official_valid_loaded"]
            and not summary["official_test_loaded"]
            and not summary["monitor_labels_read"]
            and not summary["locked_labels_read"]
            and not summary["model_predictions_used"]
            and not input_audit["official_valid_loaded"]
            and not input_audit["official_test_loaded"]
            and not input_audit["official_loader_scan_hits"]
            and label_access["monitor_rows_passed_to_loader"] == 0
            and label_access["locked_rows_passed_to_loader"] == 0
        ),
    }
    passed = bool(all(conditions.values()))
    decision = "MATRIX_FEATURE_CHEAP_SIGNAL" if passed else "MATRIX_FEATURE_NO_CHEAP_SIGNAL"
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "decision": decision,
        "passed": passed,
        "conditions": conditions,
        "summary": summary,
        "math_gates_all_passed": bool(math_gates["all_passed"]),
        "model_audit_all_passed": bool(model_audit["all_passed"]),
        "cpu_gate_thresholds": {
            "mean_improvement_min": mf.CPU_MIN_MEAN_IMPROVEMENT,
            "folds_improved_min": mf.CPU_MIN_FOLDS_IMPROVED,
            "trim_k": mf.CPU_TRIM_K,
            "tail_threshold": mf.TAIL_THRESHOLD,
        },
        "labels_used": "Fit rows raw y only",
        "monitor_labels_read": False,
        "locked_labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(cpu_gate_path(), payload)
    write_json(
        decision_path(),
        {
            "protocol": PROTOCOL_ID,
            "decision": decision,
            "passed": passed,
            "conditions": conditions,
            "stop_rule": (
                "CPU gate failed: stop, report MATRIX_FEATURE_NO_CHEAP_SIGNAL, do not start GPU, "
                "do not change alpha / PCA dimension / kernel / Ridge."
                if not passed
                else "CPU gate passed: proceed to pre-GPU freeze; GPU is not automatically started by this command."
            ),
            "git_commit": git_commit(),
        },
    )
    for name, ok in conditions.items():
        log(f"[cpu-gate {name}] {'PASS' if ok else 'FAIL'}")
    log(f"[cpu-gate] decision={decision}")
    return 0 if passed else 1


# ---------------------------------------------------------------------------
# pre-GPU freeze / finalize
# ---------------------------------------------------------------------------


def _freeze_inputs() -> dict[str, str]:
    names = [
        "split_V.json",
        "split_compare_V.json",
        "pairing_verification.json",
        "input_audit_V.json",
        "objects_manifest_V.json",
        "prep_fit_V.json",
        "cpu_folds_V.json",
        "cpu_signal.json",
        "cpu_signal_predictions.npz",
        "cpu_label_access.json",
        "math_gates.json",
        "model_audit.json",
        "cpu_gate.json",
        "decision_cpu.json",
        "objects_fit_V.pt",
        "objects_monitor_V.pt",
        "objects_locked_V.pt",
        "kernels_fit_V.pt",
        "kernels_monitor_V.pt",
        "kernels_locked_V.pt",
    ]
    out = {}
    for name in names:
        p = OUT / name
        require(p.exists(), f"missing freeze input {name}")
        out[name] = file_sha256(p)
    return out


def cmd_freeze_pre_gpu(_args: argparse.Namespace) -> int:
    require(decision_path().exists(), "decision_cpu.json missing")
    decision = read_json(decision_path())
    require(decision["decision"] == "MATRIX_FEATURE_CHEAP_SIGNAL" and decision["passed"], "CPU gate did not pass; refusing GPU freeze")
    payload = {
        "protocol": PROTOCOL_ID,
        "stage": "pre_gpu",
        "git_commit": git_commit(),
        "created_utc": now_utc(),
        "protocol_doc": {"path": str(PROTOCOL_DOC.relative_to(REPO_ROOT)), "sha256": file_sha256(PROTOCOL_DOC)},
        "config": {"path": str(CONFIG_PATH.relative_to(REPO_ROOT)), "sha256": config_sha256()},
        "code": {name: {"path": str((_CODE_DIR / name).relative_to(REPO_ROOT)), "sha256": sha} for name, sha in code_hashes().items()},
        "inputs": _freeze_inputs(),
        "cpu_gate_passed": True,
        "cpu_gate_decision": decision["decision"],
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(freeze_path(), payload)
    log(f"[freeze-pre-gpu] wrote {freeze_path().name} with {len(payload['inputs'])} input hashes")
    return 0


def cmd_finalize_cpu(_args: argparse.Namespace) -> int:
    require(decision_path().exists(), "decision_cpu.json missing")
    decision = read_json(decision_path())
    files = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in ("evidence_manifest.json",):
            files.append({"path": p.name, "sha256": file_sha256(p), "bytes": int(p.stat().st_size)})
    payload = {
        "protocol": PROTOCOL_ID,
        "stage": "cpu",
        "git_commit": git_commit(),
        "created_utc": now_utc(),
        "decision": decision["decision"],
        "passed": bool(decision["passed"]),
        "code": code_hashes(),
        "config_sha256": config_sha256(),
        "protocol_doc_sha256": file_sha256(PROTOCOL_DOC),
        "artifacts": files,
        "n_artifacts": len(files),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "monitor_labels_read": False,
        "locked_labels_read": False,
    }
    write_json(OUT / "evidence_manifest.json", payload)
    log(f"[finalize-cpu] {len(files)} artifacts hashed; decision={decision['decision']}")
    return 0




# ---------------------------------------------------------------------------
# Stage B — GPU smoke / SSL / train / predict / freeze / score
# ---------------------------------------------------------------------------

import bjoc_masked_pretraining_v1 as mp  # noqa: E402

GPU_CAP_HOURS = 3.0
GPU_SMOKE_STOP_HOURS = 2.7
MONITOR_DELTA = 0.004
NULL_DELTA = 0.003
PART_MEAN = 0.0025
PART_MAX = 0.02
RMSE_TOL = 0.005
TAIL_TOL = 0.005
TAIL_THRESHOLD = -3.0
TOP_K_REMOVED = 10
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 0


def ssl_path(seed: int) -> Path:
    return OUT / f"ssl_V_s{int(seed)}.pt"


def ssl_info_path(seed: int) -> Path:
    return OUT / f"ssl_V_s{int(seed)}.json"


def soup_path(arm: str, seed: int) -> Path:
    return OUT / f"soup_{arm}_V_s{int(seed)}.pt"


def train_info_path(arm: str, seed: int) -> Path:
    return OUT / f"train_{arm}_V_s{int(seed)}.json"


def pred_path(zone: str, arm: str, seed: int) -> Path:
    return OUT / f"pred_{zone}_{arm}_V_s{int(seed)}.npz"


def part_path(zone: str, seed: int) -> Path:
    return OUT / f"participation_{zone}_V_s{int(seed)}.npz"


def part_info_path(zone: str, seed: int) -> Path:
    return OUT / f"participation_{zone}_V_s{int(seed)}.json"


def smoke_path() -> Path:
    return OUT / "smoke_V.json"


def prediction_manifest_path() -> Path:
    return OUT / "prediction_manifest_V.json"


def monitor_lock_path() -> Path:
    return OUT / "monitor_read.lock"


def locked_lock_path() -> Path:
    return OUT / "locked_read.lock"


def monitor_gate_path() -> Path:
    return OUT / "monitor_gate_V.json"


def monitor_score_path() -> Path:
    return OUT / "monitor_score_V.json"


def locked_gate_path() -> Path:
    return OUT / "locked_gate_V.json"


def locked_score_path() -> Path:
    return OUT / "locked_score_V.json"


def gpu_decision_path() -> Path:
    return OUT / "decision.json"


def ledger_path() -> Path:
    return OUT / "gpu_budget_ledger.json"


def _gpu_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    require(torch.cuda.is_available(), "CUDA is not available")
    return torch.device("cuda:0")


def require_freeze() -> Mapping[str, Any]:
    require(freeze_path().exists(), "protocol_freeze.json missing: run freeze-pre-gpu first")
    payload = read_json(freeze_path())
    require(payload.get("cpu_gate_passed") is True, "freeze does not record a CPU pass")
    frozen_code = {name: rec["sha256"] for name, rec in payload["code"].items()}
    require(frozen_code == code_hashes(), "code changed after freeze; refusing GPU run")
    require(payload["config"]["sha256"] == config_sha256(), "config changed after freeze")
    require(payload["protocol_doc"]["sha256"] == file_sha256(PROTOCOL_DOC), "protocol doc changed after freeze")
    return payload


def append_ledger(label: str, seconds: float, meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = read_json(ledger_path()) if ledger_path().exists() else {"protocol": PROTOCOL_ID, "entries": []}
    entry = {
        "label": str(label),
        "wall_seconds": float(seconds),
        "utc": now_utc(),
        "device": (meta or {}).get("device"),
        "note": (meta or {}).get("note"),
    }
    payload["entries"].append(entry)
    payload["total_seconds"] = float(sum(float(e["wall_seconds"]) for e in payload["entries"]))
    payload["total_gpu_hours"] = payload["total_seconds"] / 3600.0
    payload["cap_gpu_hours"] = GPU_CAP_HOURS
    payload["within_cap"] = bool(payload["total_gpu_hours"] <= GPU_CAP_HOURS)
    write_json(ledger_path(), payload)
    log(f"[ledger] +{seconds/60:.2f} min ({label}); total={payload['total_gpu_hours']:.3f} GPU-h")
    return payload


def _fit_inputs() -> tuple[list[sro.MolObjects], list[int], np.ndarray, Mapping[str, Any]]:
    split = load_split()
    fit_objs = load_zone_objects("fit")
    fit_rows = [int(r) for r in split["fit_rows"]]
    require([int(o.mol_index) for o in fit_objs] == fit_rows, "fit object row order mismatch")
    y_fit = load_fit_y()
    require(len(y_fit) == len(fit_objs), "fit y length mismatch")
    norm = read_json(prep_path())
    return fit_objs, fit_rows, y_fit, norm


def _kernels_np(zone: str, mode: str) -> list[np.ndarray]:
    k = load_kernels(zone)
    return [np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32) for t in k[mode]]


def _load_soup(model: torch.nn.Module, arm: str, seed: int) -> dict[str, Any]:
    payload = torch.load(soup_path(arm, seed), map_location="cpu", weights_only=False)
    model.load_state_dict({k: v for k, v in payload["state_dict"].items()})
    return payload


def _rows_sha256(rows: Sequence[int]) -> str:
    return hashlib.sha256(b"".join(int(r).to_bytes(8, "little", signed=False) for r in rows)).hexdigest()


def cmd_stage_inputs(_args: argparse.Namespace) -> int:
    """Rebuild the label-blind objects/K caches on the compute host and verify hashes."""
    require_freeze() if freeze_path().exists() else None
    expected = read_json(OUT / "objects_manifest_V.json")
    started = time.perf_counter()
    split, ds, guard = build_guard()
    objs: dict[str, list[sro.MolObjects]] = {}
    kernels: dict[str, dict[str, list[torch.Tensor]]] = {}
    for zone in ZONES:
        rows = [int(i) for i in split[f"{zone}_rows"]]
        built = []
        for i in rows:
            d = guard[int(i)]
            built.append(sro.build_mol_objects(int(i), d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy()))
        require([o.mol_index for o in built] == rows, f"object row order mismatch in {zone}")
        torch.save(built, objects_path(zone))
        kr, ku = [], []
        for o in built:
            K = mf.kernel_from_adjacency(mf.physical_adjacency(o))["K"]
            kr.append(torch.as_tensor(K, dtype=torch.float32))
            ku.append(torch.as_tensor(mf.content_null_kernel(K), dtype=torch.float32))
        torch.save({"R": kr, "U": ku, "alpha": mf.ALPHA}, kernels_path(zone))
        objs[zone] = built
        kernels[zone] = {"R": kr, "U": ku}
    prep = bp.fit_standardiser(objs["fit"])
    write_json(prep_path(), prep)
    require(guard.access_attempts == [], "guarded label access during stage-inputs")
    checks = {}
    for zone in ZONES:
        actual = {
            "n": len(objs[zone]),
            "rows_sha256": rows_sha256([o.mol_index for o in objs[zone]]),
            "object_state_sha256": object_zone_sha256(objs[zone]),
            "kernel_digest": kernel_digest(kernels[zone]),
        }
        exp = expected["zones"][zone]
        checks[zone] = {
            "actual": actual,
            "expected": {k: exp[k] for k in ("n", "rows_sha256", "object_state_sha256", "kernel_digest")},
            "rows_match": actual["rows_sha256"] == exp["rows_sha256"],
            "objects_match": actual["object_state_sha256"] == exp["object_state_sha256"],
            "kernel_digest_match": actual["kernel_digest"] == exp["kernel_digest"],
        }
        require(checks[zone]["rows_match"], f"{zone} rows hash mismatch on compute host")
        require(checks[zone]["objects_match"], f"{zone} object hash mismatch on compute host")
        require(checks[zone]["kernel_digest_match"], f"{zone} kernel digest mismatch on compute host")
    exp_prep = read_json(OUT / "prep_fit_V.json")
    prep_match = bool(
        np.array_equal(np.asarray(prep["global_mean"]), np.asarray(exp_prep["global_mean"]))
        and np.array_equal(np.asarray(prep["global_scale"]), np.asarray(exp_prep["global_scale"]))
        and int(prep.get("n_fit", -1)) == int(exp_prep.get("n_fit", -2))
    )
    require(prep_match, "Fit-only standardiser mismatch on compute host")
    payload = {
        "protocol": PROTOCOL_ID,
        "stage": "stage_inputs",
        "git_commit": git_commit(),
        "utc": now_utc(),
        "wall_seconds": time.perf_counter() - started,
        "checks": checks,
        "prep_match": prep_match,
        "forbidden_label_access_attempts": len(guard.access_attempts),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "staged_inputs_V.json", payload)
    append_ledger("stage-inputs", payload["wall_seconds"], {"device": "cpu"})
    log("[stage-inputs] all object/kernel/standardiser hashes match the frozen CPU stage")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Short GPU smoke and budget projection.  Writes no formal checkpoint."""
    require_freeze()
    device = _gpu_device(args.device)
    started = time.perf_counter()
    fit_objs, fit_rows, y_fit, norm = _fit_inputs()
    split, ds, guard = build_guard()
    # 20 SSL updates, all Fit rows available but only 20 batches consumed.
    t0 = time.perf_counter()
    ssl = bp.pretrain_masked(
        seed=0,
        fit_objs=fit_objs,
        fit_rows=fit_rows,
        raw_dataset=guard,
        device=device,
        mode="atom_bond",
        smoke_updates=20,
        log_fn=log,
    )
    ssl_wall = time.perf_counter() - t0
    require(guard.access_attempts == [], "guarded label access during SSL smoke")
    trio = mf.build_three_arms(0, norm, device)
    for m in trio.values():
        mp.load_encoder_state(m, ssl["encoder_state"], device=device)
    n_smoke = 512
    smoke_objs = fit_objs[:n_smoke]
    smoke_y = y_fit[:n_smoke]
    kr = _kernels_np("fit", "R")[:n_smoke]
    t0 = time.perf_counter()
    p3_info = sl.train_supervised_model(
        arm="P3", seed=0, model=trio["P3"], train_objs=smoke_objs, y_fit=smoke_y,
        device=device, epochs=1, batch_size=bp.BATCH_SIZE, log_fn=log,
    )
    p3_wall = time.perf_counter() - t0
    t0 = time.perf_counter()
    mfic_info = mf.train_supervised_mfic(
        arm="MFIC-R", seed=0, model=trio["MFIC-R"], train_objs=smoke_objs, kernels=kr,
        y_fit=smoke_y, device=device, epochs=1, batch_size=bp.BATCH_SIZE, log_fn=log,
    )
    mfic_wall = time.perf_counter() - t0
    mon_objs = load_zone_objects("monitor")[:256]
    mon_kr = _kernels_np("monitor", "R")[:256]
    t0 = time.perf_counter()
    mf.predict_mfic(trio["MFIC-R"], mon_objs, mon_kr, device)
    pred_wall = time.perf_counter() - t0
    wall = time.perf_counter() - started

    steps_full = math.ceil(len(fit_objs) / bp.BATCH_SIZE) * 360
    steps_smoke = math.ceil(n_smoke / bp.BATCH_SIZE) * 1
    ssl_full = ssl_wall / 20.0 * bp.PRETRAIN_UPDATES
    train_p3_full = p3_wall * steps_full / max(steps_smoke, 1)
    train_mfic_full = mfic_wall * steps_full / max(steps_smoke, 1)
    train_all = 2.0 * (train_p3_full + 2.0 * train_mfic_full)
    pred_full = pred_wall * (1000.0 / 256.0) * 12.0 + pred_wall * (1000.0 / 256.0) * 4.0
    projected = 2.0 * ssl_full + train_all + pred_full + wall
    projected_hours = projected / 3600.0
    enforce = bool(device.type == "cuda" and not args.no_budget_stop)
    stop = bool(enforce and projected_hours > GPU_SMOKE_STOP_HOURS)
    payload = {
        "protocol": PROTOCOL_ID,
        "stage": "gpu_smoke",
        "git_commit": git_commit(),
        "utc": now_utc(),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "wall_seconds": wall,
        "ssl_smoke_seconds": ssl_wall,
        "p3_one_epoch_512_seconds": p3_wall,
        "mfic_one_epoch_512_seconds": mfic_wall,
        "prediction_256_seconds": pred_wall,
        "estimated_full": {
            "ssl_per_seed_seconds": ssl_full,
            "p3_train_per_seed_seconds": train_p3_full,
            "mfic_train_per_arm_per_seed_seconds": train_mfic_full,
            "three_arms_two_seeds_train_seconds": train_all,
            "predictions_seconds": pred_full,
            "total_seconds": projected,
            "total_gpu_hours": projected_hours,
        },
        "budget_stop_hours": GPU_SMOKE_STOP_HOURS,
        "hard_cap_hours": GPU_CAP_HOURS,
        "enforce_budget_on_this_device": enforce,
        "stop": stop,
        "parameter_audit": {
            "p3": p3_info["parameter_audit"],
            "mfic": mfic_info["parameter_audit"],
        },
        "forbidden_label_access_attempts": len(guard.access_attempts),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(smoke_path(), payload)
    append_ledger("smoke", wall, {"device": str(device), "note": "20 SSL updates + 1 epoch on 512 + 256 predictions"})
    log(f"[smoke] projected total {projected_hours:.3f} GPU-h (stop>{GPU_SMOKE_STOP_HOURS}); stop={stop}")
    if stop:
        log("[smoke] BUDGET STOP: projected formal run exceeds 2.7 GPU-h; do not start SSL/train")
        return 3
    return 0


def cmd_ssl(args: argparse.Namespace) -> int:
    require_freeze()
    seed = int(args.seed)
    require(seed in SEEDS, "unknown seed")
    device = _gpu_device(args.device)
    fit_objs, fit_rows, _y_fit, _norm = _fit_inputs()
    split, ds, guard = build_guard()
    log(f"[ssl s{seed}] starting {bp.PRETRAIN_UPDATES} masked updates on {len(fit_objs)} Fit rows")
    info = bp.pretrain_masked(
        seed=seed,
        fit_objs=fit_objs,
        fit_rows=fit_rows,
        raw_dataset=guard,
        device=device,
        mode="atom_bond",
        smoke_updates=None,
        log_fn=log,
    )
    require(guard.access_attempts == [], "guarded label access during SSL")
    require(info["fit_rows_only"] is True and info["labels_read"] is False, "SSL protocol violation")
    state = info.pop("encoder_state")
    payload = {
        "state": state,
        "encoder_final_sha256": info["encoder_final_sha256"],
        "encoder_init_sha256": info["encoder_init_sha256"],
        "seed": seed,
        "protocol": PROTOCOL_ID,
        "fit_rows_sha256": _rows_sha256(fit_rows),
        "info": info,
    }
    torch.save(payload, ssl_path(seed))
    write_json(ssl_info_path(seed), {
        **info,
        "checkpoint_file": ssl_path(seed).name,
        "checkpoint_sha256": file_sha256(ssl_path(seed)),
        "fit_rows_sha256": _rows_sha256(fit_rows),
        "forbidden_label_access_attempts": len(guard.access_attempts),
    })
    append_ledger(f"ssl-s{seed}", info["wall_seconds"], {"device": str(device)})
    log(f"[ssl s{seed}] done in {info['wall_seconds']/60:.2f} min; encoder={info['encoder_final_sha256'][:16]}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    require_freeze()
    seed = int(args.seed)
    require(seed in SEEDS, "unknown seed")
    device = _gpu_device(args.device)
    fit_objs, fit_rows, y_fit, norm = _fit_inputs()
    ssl_payload = torch.load(ssl_path(seed), map_location="cpu", weights_only=False)
    enc_state = ssl_payload["state"]
    require(ssl_payload["fit_rows_sha256"] == _rows_sha256(fit_rows), "SSL checkpoint was fit on different rows")
    trio = mf.build_three_arms(seed, norm, device)
    for arm, model in trio.items():
        mp.load_encoder_state(model, enc_state, device=device)
    shared = {arm: mf.shared_hash(m) for arm, m in trio.items()}
    full_before = {arm: mf.full_hash(m) for arm, m in trio.items()}
    enc = {arm: mf.encoder_hash(m) for arm, m in trio.items()}
    require(len(set(shared.values())) == 1, "common-module initialisation differs across arms")
    require(full_before["MFIC-R"] == full_before["MFIC-U"], "MFIC-R / MFIC-U full init differs")
    require(all(v == ssl_payload["encoder_final_sha256"] for v in enc.values()), "SSL encoder hash mismatch after load")
    kr = _kernels_np("fit", "R")
    ku = _kernels_np("fit", "U")
    require(len(kr) == len(fit_objs) == len(ku), "kernel list length mismatch")
    infos: dict[str, Any] = {}
    t_all = time.perf_counter()
    for arm in ("P3", "MFIC-R", "MFIC-U"):
        t0 = time.perf_counter()
        if arm == "P3":
            info = sl.train_supervised_model(
                arm="P3", seed=seed, model=trio[arm], train_objs=fit_objs, y_fit=y_fit,
                device=device, epochs=360, batch_size=bp.BATCH_SIZE, log_fn=log,
            )
        else:
            info = mf.train_supervised_mfic(
                arm=arm, seed=seed, model=trio[arm], train_objs=fit_objs,
                kernels=kr if arm == "MFIC-R" else ku, y_fit=y_fit,
                device=device, epochs=360, batch_size=bp.BATCH_SIZE, log_fn=log,
            )
        info["wall_seconds_total"] = time.perf_counter() - t0
        info["init_audit"] = {
            "shared_hash": shared[arm],
            "encoder_hash_after_ssl_load": enc[arm],
            "full_hash_before_training": full_before[arm],
            "all_three_shared_hashes_equal": len(set(shared.values())) == 1,
            "R_U_full_init_equal": full_before["MFIC-R"] == full_before["MFIC-U"],
            "encoder_matches_ssl": enc[arm] == ssl_payload["encoder_final_sha256"],
        }
        state = info.pop("soup_state")
        torch.save({
            "state_dict": state,
            "arm": arm,
            "seed": seed,
            "protocol": PROTOCOL_ID,
            "soup_fit_mae": info["soup_fit_mae"],
            "batch_order": info["batch_order"],
            "init_audit": info["init_audit"],
        }, soup_path(arm, seed))
        write_json(train_info_path(arm, seed), {
            **info,
            "checkpoint_file": soup_path(arm, seed).name,
            "checkpoint_sha256": file_sha256(soup_path(arm, seed)),
            "fit_rows_sha256": _rows_sha256(fit_rows),
        })
        infos[arm] = info
        append_ledger(f"train-{arm}-s{seed}", info["wall_seconds_total"], {"device": str(device)})
    orders = {arm: infos[arm]["batch_order"]["full_hash"] for arm in infos}
    expected_order = bp.batch_order_hash(seed, len(fit_objs), bp.BATCH_SIZE, 360)
    require(len(set(orders.values())) == 1, "batch order differs across arms")
    require(all(v == expected_order for v in orders.values()), "batch order does not match independent recomputation")
    require(all(int(infos[a]["updates"]) == 22680 for a in infos), "update count is not 22,680")
    summary = {
        "protocol": PROTOCOL_ID,
        "stage": "train",
        "seed": seed,
        "device": str(device),
        "fit_rows_sha256": _rows_sha256(fit_rows),
        "n_fit": len(fit_objs),
        "batch_order": orders,
        "batch_order_matches_independent": True,
        "updates_per_arm": {a: infos[a]["updates"] for a in infos},
        "soup_fit_mae": {a: infos[a]["soup_fit_mae"] for a in infos},
        "wall_seconds": {a: infos[a]["wall_seconds_total"] for a in infos},
        "total_wall_seconds": time.perf_counter() - t_all,
        "init_audit": {a: infos[a]["init_audit"] for a in infos},
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / f"train_summary_V_s{seed}.json", summary)
    log(f"[train s{seed}] done; P3={summary['soup_fit_mae']['P3']:.6f} R={summary['soup_fit_mae']['MFIC-R']:.6f} U={summary['soup_fit_mae']['MFIC-U']:.6f}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    require_freeze()
    seed = int(args.seed)
    require(seed in SEEDS, "unknown seed")
    device = _gpu_device(args.device)
    split = load_split()
    norm = read_json(prep_path())
    started = time.perf_counter()
    models = {}
    for arm in ARMS:
        if arm == "P3":
            m = mf.build_p3(seed, norm, device)
        elif arm == "MFIC-R":
            m = mf.build_mfic(seed, norm, device, kernel_mode="R")
        else:
            m = mf.build_mfic(seed, norm, device, kernel_mode="U")
        _load_soup(m, arm, seed)
        m.eval()
        models[arm] = m
    for zone in ("monitor", "locked"):
        rows = [int(r) for r in split[f"{zone}_rows"]]
        objs = load_zone_objects(zone)
        require([int(o.mol_index) for o in objs] == rows, f"{zone} object row order mismatch")
        k = load_kernels(zone)
        kr_np = [np.ascontiguousarray(t.numpy(), dtype=np.float32) for t in k["R"]]
        ku_np = [np.ascontiguousarray(t.numpy(), dtype=np.float32) for t in k["U"]]
        for arm in ARMS:
            if arm == "P3":
                pred = bp.predict_model(models[arm], objs, device)
                kernel_mode = "none"
            elif arm == "MFIC-R":
                pred = mf.predict_mfic(models[arm], objs, kr_np, device)
                kernel_mode = "R"
            else:
                pred = mf.predict_mfic(models[arm], objs, ku_np, device)
                kernel_mode = "U"
            require(pred.shape == (len(rows),) and np.isfinite(pred).all(), f"bad predictions {zone} {arm}")
            np.savez_compressed(
                pred_path(zone, arm, seed),
                rows=np.asarray(rows, dtype=np.int64),
                pred=np.asarray(pred, dtype=np.float64),
                arm=arm,
                seed=seed,
                zone=zone,
                kernel_mode=kernel_mode,
            )
            if arm == "MFIC-R":
                pred_swap = mf.predict_mfic(models[arm], objs, ku_np, device)
                zero_np = [np.zeros_like(x, dtype=np.float32) for x in kr_np]
                pred_zero = mf.predict_mfic(models[arm], objs, zero_np, device)
                diff = np.abs(pred - pred_swap)
                diff_zero = np.abs(pred - pred_zero)
                np.savez_compressed(
                    part_path(zone, seed),
                    rows=np.asarray(rows, dtype=np.int64),
                    pred_real=pred,
                    pred_swap=pred_swap,
                    pred_zero=pred_zero,
                )
                part = {
                    "protocol": PROTOCOL_ID,
                    "zone": zone,
                    "seed": seed,
                    "n": len(rows),
                    "mean_abs_diff": float(diff.mean()),
                    "max_abs_diff": float(diff.max()),
                    "median_abs_diff": float(np.median(diff)),
                    "p95_abs_diff": float(np.quantile(diff, 0.95)),
                    "mean_abs_diff_zero": float(diff_zero.mean()),
                    "max_abs_diff_zero": float(diff_zero.max()),
                    "threshold_mean": PART_MEAN,
                    "threshold_max": PART_MAX,
                    "passed": bool(diff.mean() >= PART_MEAN and diff.max() >= PART_MAX),
                    "prediction_file": pred_path(zone, arm, seed).name,
                    "real_prediction_sha256": hashlib.sha256(np.asarray(pred, dtype=np.float64).tobytes()).hexdigest(),
                    "swap_prediction_sha256": hashlib.sha256(np.asarray(pred_swap, dtype=np.float64).tobytes()).hexdigest(),
                }
                write_json(part_info_path(zone, seed), part)
    wall = time.perf_counter() - started
    append_ledger(f"predict-s{seed}", wall, {"device": str(device)})
    log(f"[predict s{seed}] 6 arm/zone prediction files + participation diagnostics written; {wall:.1f}s")
    return 0


def cmd_freeze_predictions(_args: argparse.Namespace) -> int:
    require_freeze()
    split = load_split()
    pred_files, soup_files, ssl_files, part_files = [], [], [], []
    for zone in ("monitor", "locked"):
        rows = [int(r) for r in split[f"{zone}_rows"]]
        for arm in ARMS:
            for seed in SEEDS:
                p = pred_path(zone, arm, seed)
                require(p.exists(), f"missing prediction {p.name}")
                z = np.load(p)
                require(np.array_equal(np.asarray(z["rows"], dtype=np.int64), rows), f"row mismatch in {p.name}")
                require(np.isfinite(z["pred"]).all(), f"non-finite prediction in {p.name}")
                pred_files.append({"path": p.name, "sha256": file_sha256(p), "bytes": int(p.stat().st_size)})
                soup = soup_path(arm, seed)
                require(soup.exists(), f"missing soup {soup.name}")
                soup_files.append({"path": soup.name, "sha256": file_sha256(soup), "bytes": int(soup.stat().st_size)})
        for seed in SEEDS:
            p = part_path(zone, seed)
            require(p.exists(), f"missing participation file {p.name}")
            part_files.append({"path": p.name, "sha256": file_sha256(p), "bytes": int(p.stat().st_size)})
    for seed in SEEDS:
        p = ssl_path(seed)
        require(p.exists(), f"missing SSL checkpoint {p.name}")
        ssl_files.append({"path": p.name, "sha256": file_sha256(p), "bytes": int(p.stat().st_size)})
    combined = hashlib.sha256()
    for rec in sorted(pred_files, key=lambda r: r["path"]):
        combined.update(rec["path"].encode())
        combined.update(rec["sha256"].encode())
    participation = {}
    for zone in ("monitor", "locked"):
        participation[zone] = {}
        for seed in SEEDS:
            info = read_json(part_info_path(zone, seed))
            participation[zone][str(seed)] = info
    payload = {
        "protocol": PROTOCOL_ID,
        "stage": "prediction_freeze",
        "git_commit": git_commit(),
        "utc": now_utc(),
        "complete": True,
        "prediction_set_sha256": combined.hexdigest(),
        "n_predictions": len(pred_files),
        "predictions": pred_files,
        "soups": soup_files,
        "ssl_checkpoints": ssl_files,
        "participation_files": part_files,
        "participation": participation,
        "participation_passed": bool(all(v["passed"] for z in participation.values() for v in z.values())),
        "monitor_labels_read": False,
        "locked_labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(prediction_manifest_path(), payload)
    log(f"[freeze-predictions] {len(pred_files)} predictions frozen; set={combined.hexdigest()[:16]}; participation={payload['participation_passed']}")
    return 0


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def _load_preds(zone: str, arm: str) -> dict[int, np.ndarray]:
    return {int(s): np.asarray(np.load(pred_path(zone, arm, s))["pred"], dtype=np.float64) for s in SEEDS}


def _paired_bootstrap(delta: np.ndarray, *, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
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
    ordinal = y > TAIL_THRESHOLD
    return {
        "mae_seed_mean": float(vals.mean()),
        "mae_per_seed": per_seed_mae,
        "seed_spread_max_minus_min": float(vals.max() - vals.min()),
        "seed_spread_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "rmse_seed_mean": float(np.sqrt(np.mean(np.mean([(preds[s] - y) ** 2 for s in SEEDS], axis=0)))),
        "median_ae": float(np.median(seed_mean)),
        "p90_ae": float(np.quantile(seed_mean, 0.90)),
        "p95_ae": float(np.quantile(seed_mean, 0.95)),
        "p99_ae": float(np.quantile(seed_mean, 0.99)),
        "max_ae": float(seed_mean.max()),
        "ordinary_mae": float(seed_mean[ordinal].mean()) if ordinal.any() else 0.0,
        "tail_mae": float(seed_mean[~ordinal].mean()) if (~ordinal).any() else 0.0,
        "tail_n": int((~ordinal).sum()),
    }


def _contrast(y: np.ndarray, new: Mapping[int, np.ndarray], ref: Mapping[int, np.ndarray], new_name: str, ref_name: str) -> dict[str, Any]:
    e_new = {s: np.abs(new[s] - y) for s in SEEDS}
    e_ref = {s: np.abs(ref[s] - y) for s in SEEDS}
    delta = np.mean([e_new[s] - e_ref[s] for s in SEEDS], axis=0)
    per_seed = {str(s): float((e_new[s] - e_ref[s]).mean()) for s in SEEDS}
    ordinal = y > TAIL_THRESHOLD
    ordinary_delta = float(delta[ordinal].mean()) if ordinal.any() else 0.0
    tail_delta = float(delta[~ordinal].mean()) if (~ordinal).any() else 0.0
    ref_seed_mean = np.mean([e_ref[s] for s in SEEDS], axis=0)
    worst = np.argsort(-ref_seed_mean)[:TOP_K_REMOVED]
    keep = np.ones(len(delta), dtype=bool)
    keep[worst] = False
    boot = _paired_bootstrap(delta)
    return {
        "new_arm": new_name,
        "ref_arm": ref_name,
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


def _gate(score: Mapping[str, Any], zone: str) -> dict[str, Any]:
    rp3 = score["contrasts"]["R-P3"]
    ru = score["contrasts"]["R-U"]
    part = score["participation"][zone]
    conditions = {
        "1_R_minus_P3_mean_le_minus_0.004": bool(rp3["delta_seed_mean"] <= -MONITOR_DELTA),
        "2_both_seeds_R_minus_P3_negative": bool(all(float(v) < 0.0 for v in rp3["per_seed_delta"].values())),
        "3_R_minus_P3_paired_ci_upper_negative": bool(float(rp3["bootstrap"]["ci_hi"]) < 0.0),
        "4a_ordinary_improved": bool(float(rp3["ordinary_delta"]) < 0.0),
        "4b_top10_removed_improved": bool(float(rp3["top10_removed_delta"]) < 0.0),
        "5a_rmse_degradation_le_0.005": bool(float(rp3["rmse_new"]) - float(rp3["rmse_ref"]) <= RMSE_TOL),
        "5b_tail_degradation_le_0.005": bool(float(rp3["tail_delta"]) <= TAIL_TOL),
        "6a_R_minus_U_mean_le_minus_0.003": bool(ru["delta_seed_mean"] <= -NULL_DELTA),
        "6b_both_seeds_R_minus_U_negative": bool(all(float(v) < 0.0 for v in ru["per_seed_delta"].values())),
        "7_participation_mean_ge_0.0025_and_max_ge_0.02": bool(
            all(float(part[str(s)]["mean_abs_diff"]) >= PART_MEAN and float(part[str(s)]["max_abs_diff"]) >= PART_MAX for s in SEEDS)
        ),
    }
    return {
        "conditions": conditions,
        "passed": bool(all(conditions.values())),
        "thresholds": {
            "R_minus_P3_mean": -MONITOR_DELTA,
            "R_minus_U_mean": -NULL_DELTA,
            "participation_mean": PART_MEAN,
            "participation_max": PART_MAX,
            "rmse_tol": RMSE_TOL,
            "tail_tol": TAIL_TOL,
            "tail_threshold": TAIL_THRESHOLD,
            "top_k_removed": TOP_K_REMOVED,
        },
    }


def _score_zone(zone: str, y: np.ndarray | None = None) -> dict[str, Any]:
    split = load_split()
    rows = [int(i) for i in split[f"{zone}_rows"]]
    if y is None:
        y = bp.load_y_rows(ENCODED_TRAIN, rows)
    require(len(y) == len(rows), f"{zone} y length mismatch")
    preds = {arm: _load_preds(zone, arm) for arm in ARMS}
    stats = {arm: _arm_stats(y, preds[arm]) for arm in ARMS}
    contrasts = {
        "R-P3": _contrast(y, preds["MFIC-R"], preds["P3"], "MFIC-R", "P3"),
        "R-U": _contrast(y, preds["MFIC-R"], preds["MFIC-U"], "MFIC-R", "MFIC-U"),
        "U-P3": _contrast(y, preds["MFIC-U"], preds["P3"], "MFIC-U", "P3"),
    }
    participation = {zone: {str(s): read_json(part_info_path(zone, s)) for s in SEEDS}}
    fit_mae = {}
    cost = {}
    for arm in ARMS:
        fit_mae[arm] = {str(s): float(read_json(train_info_path(arm, s))["soup_fit_mae"]) for s in SEEDS}
        cost[arm] = {str(s): float(read_json(train_info_path(arm, s))["wall_seconds_total"]) for s in SEEDS}
    gate = _gate({"contrasts": contrasts, "participation": participation}, zone)
    return {
        "protocol": PROTOCOL_ID,
        "zone": zone,
        "n": len(rows),
        "rows_sha256": _rows_sha256(rows),
        "y_sha256": hashlib.sha256(np.asarray(y, dtype=np.float64).tobytes()).hexdigest(),
        "arm_stats": stats,
        "contrasts": contrasts,
        "participation": participation,
        "fit_mae": fit_mae,
        "train_wall_seconds": cost,
        "gate": gate,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }


def _read_labels_once(zone: str) -> tuple[np.ndarray, dict[str, Any]]:
    split = load_split()
    rows = [int(i) for i in split[f"{zone}_rows"]]
    lock = monitor_lock_path() if zone == "monitor" else locked_lock_path()
    require(not lock.exists(), f"{zone} labels were already read; refusing a second read")
    y = bp.load_y_rows(ENCODED_TRAIN, rows)
    payload = {
        "protocol": PROTOCOL_ID,
        "zone": zone,
        "utc": now_utc(),
        "n": len(rows),
        "rows_sha256": _rows_sha256(rows),
        "y_sha256": hashlib.sha256(np.asarray(y, dtype=np.float64).tobytes()).hexdigest(),
        "encoded_train_sha256": file_sha256(ENCODED_TRAIN),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(lock, payload)
    return y, payload


def cmd_score_monitor(_args: argparse.Namespace) -> int:
    require_freeze()
    require(prediction_manifest_path().exists(), "prediction manifest missing; freeze predictions first")
    manifest = read_json(prediction_manifest_path())
    require(manifest.get("complete") is True, "prediction manifest is not complete")
    require(not monitor_lock_path().exists(), "Monitor labels were already read")
    y, lock = _read_labels_once("monitor")
    score = _score_zone("monitor", y)
    write_json(monitor_score_path(), score)
    write_json(monitor_gate_path(), {"protocol": PROTOCOL_ID, "zone": "monitor", **score["gate"], "utc": now_utc()})
    if score["gate"]["passed"]:
        log("[score-monitor] Monitor gate PASSED; Locked labels remain unread")
        return 0
    decision = {
        "protocol": PROTOCOL_ID,
        "stage": "monitor",
        "decision": "MFIC_NO_USEFUL_GAIN",
        "passed": False,
        "monitor_gate": score["gate"],
        "locked_gate": None,
        "monitor_labels_read": True,
        "locked_labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "test_0.06_claim": False,
        "utc": now_utc(),
        "reason": "Monitor purchase gate failed; stopping without reading Locked labels.",
    }
    write_json(gpu_decision_path(), decision)
    log("[score-monitor] Monitor gate FAILED -> MFIC_NO_USEFUL_GAIN; Locked unread")
    return 0


def cmd_score_locked(_args: argparse.Namespace) -> int:
    require_freeze()
    require(prediction_manifest_path().exists(), "prediction manifest missing")
    require(monitor_gate_path().exists(), "monitor gate missing; run score-monitor first")
    monitor = read_json(monitor_gate_path())
    require(monitor.get("passed") is True, "Monitor gate did not pass; refusing to read Locked labels")
    require(not locked_lock_path().exists(), "Locked labels were already read")
    y, lock = _read_labels_once("locked")
    score = _score_zone("locked", y)
    write_json(locked_score_path(), score)
    write_json(locked_gate_path(), {"protocol": PROTOCOL_ID, "zone": "locked", **score["gate"], "utc": now_utc()})
    passed = bool(score["gate"]["passed"])
    decision = {
        "protocol": PROTOCOL_ID,
        "stage": "locked",
        "decision": "MFIC_GENERALIZATION_CANDIDATE" if passed else "MFIC_NO_USEFUL_GAIN",
        "passed": passed,
        "monitor_gate": monitor,
        "locked_gate": score["gate"],
        "monitor_labels_read": True,
        "locked_labels_read": True,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "test_0.06_claim": False,
        "utc": now_utc(),
        "reason": (
            "Both Monitor and Locked purchase gates passed; this only proposes a further independent full10k confirmation."
            if passed
            else "Locked gate failed after a Monitor pass; no candidate claim."
        ),
    }
    write_json(gpu_decision_path(), decision)
    log(f"[score-locked] Locked gate {'PASSED' if passed else 'FAILED'} -> {decision['decision']}")
    return 0


def cmd_finalize(_args: argparse.Namespace) -> int:
    require_freeze()
    decision = read_json(gpu_decision_path()) if gpu_decision_path().exists() else None
    ledger = read_json(ledger_path()) if ledger_path().exists() else {"entries": [], "total_seconds": 0.0}
    files = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in ("evidence_manifest_gpu.json",):
            files.append({"path": p.name, "sha256": file_sha256(p), "bytes": int(p.stat().st_size)})
    payload = {
        "protocol": PROTOCOL_ID,
        "stage": "gpu_finalize",
        "git_commit": git_commit(),
        "utc": now_utc(),
        "decision": decision,
        "budget": {
            "cap_gpu_hours": GPU_CAP_HOURS,
            "total_gpu_hours": ledger.get("total_gpu_hours", 0.0),
            "within_cap": bool(ledger.get("total_gpu_hours", 0.0) <= GPU_CAP_HOURS),
            "entries": ledger.get("entries", []),
        },
        "artifacts": files,
        "n_artifacts": len(files),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "test_0.06_claim": False,
    }
    write_json(OUT / "evidence_manifest_gpu.json", payload)
    log(f"[finalize] {len(files)} artifacts; total={payload['budget']['total_gpu_hours']:.3f} GPU-h; decision={(decision or {}).get('decision')}")
    return 0


def cmd_ledger_note(args: argparse.Namespace) -> int:
    append_ledger(args.label, float(args.seconds), {"device": args.device, "note": args.note})
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("split")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_split)
    for name, func in (
        ("verify-inputs", cmd_verify_inputs),
        ("objects", cmd_objects),
        ("cpu-signal", cmd_cpu_signal),
        ("math-gates", cmd_math_gates),
        ("model-audit", cmd_model_audit),
        ("cpu-gate", cmd_cpu_gate),
        ("freeze-pre-gpu", cmd_freeze_pre_gpu),
        ("finalize-cpu", cmd_finalize_cpu),
        ("stage-inputs", cmd_stage_inputs),
        ("freeze-predictions", cmd_freeze_predictions),
        ("score-monitor", cmd_score_monitor),
        ("score-locked", cmd_score_locked),
        ("finalize", cmd_finalize),
    ):
        sp = sub.add_parser(name)
        sp.set_defaults(func=func)
    for name, func in (("ssl", cmd_ssl), ("train", cmd_train), ("predict", cmd_predict)):
        sp = sub.add_parser(name)
        sp.add_argument("--seed", type=int, required=True)
        sp.add_argument("--device", default=None)
        sp.set_defaults(func=func)
    sp = sub.add_parser("smoke")
    sp.add_argument("--device", default=None)
    sp.add_argument("--no-budget-stop", action="store_true")
    sp.set_defaults(func=cmd_smoke)
    sp = sub.add_parser("ledger-note")
    sp.add_argument("--label", required=True)
    sp.add_argument("--seconds", type=float, required=True)
    sp.add_argument("--device", default=None)
    sp.add_argument("--note", default=None)
    sp.set_defaults(func=cmd_ledger_note)
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
