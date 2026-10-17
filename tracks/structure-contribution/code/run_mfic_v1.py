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
            hashlib.sha256("\n".join(sorted(str(identity_of_row[int(r)]) for r in f)).encode()).hexdigest()
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
    ):
        sp = sub.add_parser(name)
        sp.set_defaults(func=func)
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
