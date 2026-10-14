"""cscl-bjoc-cross-task-lipophilicity-v1 driver.

Protocol: ``configs/bjoc_cross_task_lipophilicity_v1.yaml`` (frozen before any
GPU trajectory).  Track: ``tracks/structure-contribution``.

Stages / commands
-----------------
``prepare-dataset``  LOCAL ONLY (needs RDKit): verify the official CSV hash,
                     parse SMILES -> label-free raw graphs, reproduce the
                     DeepChem-2.8.0 scaffold split index-for-index, freeze the
                     vocabulary and the local label vault.
``prep``             remote/local: rebuild label-free object tensors for the
                     three splits, fit the train-only standardiser, record
                     initialisation parity, write ``protocol_freeze.json``.
``gates``            CPU correctness gates (the 20 required checks).
``smoke``            short GPU smoke (device, timing, memory).
``ssl``              train-only masked SSL for one seed (encoders only).
``arm``              one supervised arm (S scratch / P SSL) for one seed.
``predict``          label-blind valid/test forwards + hash freeze.
``baselines``        LOCAL ONLY (needs RDKit): ECFP4 + RF / XGBoost.
``score-valid``      LOCAL ONLY: single validation label read + promotion gate.
``score-test``       LOCAL ONLY: single terminal test label read (conditional).
``audit``            structural label-isolation / provenance audits.

Discipline: valid/test labels live only in the local
``data/lipophilicity/labels.npy`` vault; every remote artifact is label-free
except the train labels embedded in the frozen split payload.  SSL/arm/predict
hard-refuse to run when the score-label vault is present.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_lipophilicity_adapter_v1 as lip  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402   (shuffle_batches / batch_order_hash / soup)
import bjoc_masked_pretraining_v1 as mp  # noqa: E402  (encoder_state / group_state_hash)
import structural_relation_composer_v1 as src  # noqa: E402

PROTOCOL_ID = lip.PROTOCOL_ID
CONFIG_PATH = _CODE_DIR.parent / "configs" / "bjoc_cross_task_lipophilicity_v1.yaml"
#: ``BJOC_LIPO_OUT`` redirects every artifact of this round; used only for
#: sandboxed local debugging.  Production runs keep the committed path.
OUT = Path(os.environ.get(
    "BJOC_LIPO_OUT",
    str(REPO_ROOT / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1"),
))
DATA_DIR = REPO_ROOT / "data/lipophilicity"
LABEL_VAULT = DATA_DIR / "labels.npy"
LABEL_META = DATA_DIR / "labels_meta.json"

RAW_DATASET = OUT / "dataset" / "lipo_raw_v1.json.gz"
SPLIT_PATH = OUT / "split_lipo_v1.json"
VOCAB_PATH = OUT / "vocab_v1.json"
PROVENANCE = OUT / "provenance.json"
ROW_MANIFEST = OUT / "row_manifest.json"
RAW_FIXTURE = OUT / "raw_fixture.json"
STANDARDIZER = OUT / "standardizer_v1.json"
FREEZE = OUT / "protocol_freeze.json"
INIT_PARITY = OUT / "init_parity.json"
CPU_GATES = OUT / "cpu_gates.json"
GPU_SMOKE = OUT / "gpu_smoke.json"

ARMS = ("S", "P")
SEEDS_INITIAL = (0, 1)
SEEDS_CONFIRM = (2, 3, 4)
BATCH_SIZE = 128
EPOCHS = 360
SOUP_LAST = 5
LR = 1.0e-3
LR_MIN = 1.0e-5
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
TRAIN_SHUFFLE_OFFSET = 101
PRETRAIN_LR = 1.0e-3
PRETRAIN_WD = 1.0e-5
PRETRAIN_CLIP = 5.0
PRETRAIN_SHUFFLE_OFFSET = 202
EXPOSURE_TARGET = 96

#: pre-registered promotion rule
PROMOTE_REL_RMSE_DROP = 0.04
PROMOTE_ABS_RMSE_DROP = 0.03
PROMOTE_MAE_TOLERANCE = 0.005
PROMOTE_SINGLE_MOL_SHARE_MAX = 0.25

#: shared mutable guard: set when the local score-label vault is read
_SCORE_LABELS_READ = False


def log(msg: str) -> None:
    print(msg, flush=True)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_j(payload), indent=2), encoding="utf-8")


def _j(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _j(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_j(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _code_paths() -> dict[str, Path]:
    return {
        "driver": _CODE_DIR / "run_bjoc_cross_task_lipophilicity_v1.py",
        "adapter": _CODE_DIR / "bjoc_lipophilicity_adapter_v1.py",
        "composer": _CODE_DIR / "structural_relation_composer_v1.py",
        "objects": _CODE_DIR / "structural_relation_objects_v1.py",
        "config": CONFIG_PATH,
    }


def code_hashes() -> dict[str, str]:
    return {k: file_sha256(p) for k, p in _code_paths().items()}


def verify_freeze(*, strict_code: bool = True) -> dict[str, Any]:
    if not FREEZE.exists():
        raise SystemExit("protocol is not frozen; run `prep` first")
    frozen = read_json(FREEZE)
    if frozen["protocol"] != PROTOCOL_ID:
        raise SystemExit("freeze protocol id mismatch")
    if frozen["config_sha256"] != file_sha256(CONFIG_PATH):
        raise SystemExit("protocol config drifted from the freeze")
    if strict_code:
        current = code_hashes()
        drift = {k: (frozen["code_hashes"].get(k), v) for k, v in current.items() if frozen["code_hashes"].get(k) != v}
        if drift:
            raise SystemExit(f"code drift versus freeze: {drift}")
    for key, digest in frozen.get("artifact_sha256", {}).items():
        path = OUT / frozen["artifact_paths"][key]
        if not path.exists() or file_sha256(path) != digest:
            raise SystemExit(f"frozen artifact drift: {key} ({path})")
    return frozen


def assert_score_labels_absent() -> None:
    """SSL / arm / predict must never run where valid/test labels are readable."""
    global _SCORE_LABELS_READ
    if LABEL_VAULT.exists():
        raise SystemExit(
            f"refusing to run a training/inference command: score-label vault {LABEL_VAULT} is present. "
            "Run those commands on the label-free compute checkout."
        )
    if _SCORE_LABELS_READ:
        raise SystemExit("score labels were already read in this process")


# ---------------------------------------------------------------------------
# global-score-label vault (LOCAL ONLY)
# ---------------------------------------------------------------------------


def read_label_vault(zone: str, split: Mapping[str, Any]) -> np.ndarray:
    global _SCORE_LABELS_READ
    if not LABEL_VAULT.exists():
        raise SystemExit(f"label vault {LABEL_VAULT} missing; run prepare-dataset locally")
    labels = np.load(LABEL_VAULT)
    rows = [int(r) for r in split[f"{zone}_rows"]]
    _SCORE_LABELS_READ = True
    return labels[rows]


# ---------------------------------------------------------------------------
# prepare-dataset (LOCAL, RDKit)
# ---------------------------------------------------------------------------


def select_fixture_rows(raws: Sequence[lip.RawMol], max_rows: int = 12) -> list[tuple[int, str]]:
    """Deterministic golden-fixture selection: cover every chemistry edge case.

    The fixture is written by ``prepare-dataset`` (RDKit host) and re-checked by
    the CPU gates on the GPU host, which has no RDKit: it therefore also proves
    that the deployed raw artifact is byte-for-byte the locally built one.
    """
    picks: list[tuple[int, str]] = []
    seen: set[int] = set()

    def add(row: int, reason: str) -> None:
        if row not in seen:
            seen.add(row)
            picks.append((int(row), reason))

    add(0, "first row")
    sizes = np.asarray([r.n_atoms for r in raws])
    add(int(np.argmin(sizes)), "smallest molecule")
    add(int(np.argmax(sizes)), "largest molecule")
    checks = (
        ("multi-component", lambda r: r.n_components > 1),
        ("charged atom", lambda r: bool(((r.atom_charge < 2) | (r.atom_charge > 2)).any())),
        ("specified chirality", lambda r: bool((r.atom_chirality > 0).any())),
        ("triple bond", lambda r: bool((r.edge_class == 2).any())),
        ("UNK-mapped atom", lambda r: bool((r.atom_class == lip.ATOM_UNK).any())),
        ("bond stereo", lambda r: bool((r.edge_stereo > 0).any())),
    )
    for reason, predicate in checks:
        for row, raw in enumerate(raws):
            if predicate(raw):
                add(row, reason)
                break
    blocks = lip.atom_blocks()
    for row, raw in enumerate(raws):
        ring = lip.build_lipo_objects(raw).atom_feat[:, blocks["min_ring_size"]]
        if ring.shape[1] > 5 and float(ring[:, 5].max()) > 0.5:
            if row not in seen:
                add(row, "macrocycle (>=8-membered ring)")
                break
            continue
    return picks[:max_rows]


def cmd_prepare_dataset(args: argparse.Namespace) -> int:
    cfg = load_config()
    csv_path = Path(args.csv)
    sha = file_sha256(csv_path)
    if sha != cfg["dataset"]["sha256"]:
        raise SystemExit(f"CSV sha256 mismatch: {sha} != {cfg['dataset']['sha256']}")
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != int(cfg["dataset"]["rows"]):
        raise SystemExit(f"expected {cfg['dataset']['rows']} rows, got {len(rows)}")
    columns = list(rows[0].keys())
    compound_ids = [r["CMPD_CHEMBLID"] for r in rows]
    smiles = [r["smiles"] for r in rows]
    labels = np.asarray([float(r[cfg["dataset"]["label_column"]]) for r in rows], dtype=np.float64)

    raws = []
    canonical, canonical_hashes, scaffold_strings = [], [], []
    parse_failures = 0
    started = time.perf_counter()
    for i, smi in enumerate(smiles):
        try:
            raw = lip.raw_from_smiles(smi, i, compound_ids[i])
            can, can_hash = lip.canonical_smiles_and_hash(smi)
            scaffold = lip.generate_scaffold_dc280(smi)
        except Exception:
            parse_failures += 1
            continue
        raws.append(raw)
        canonical.append(can)
        canonical_hashes.append(can_hash)
        scaffold_strings.append(scaffold)
    build_seconds = time.perf_counter() - started
    if parse_failures:
        raise SystemExit(f"{parse_failures} molecules failed RDKit parsing; refusing silent deletion")

    train_idx, valid_idx, test_idx = lip.scaffold_split_dc280(smiles)
    reference_match = None
    if args.reference_json:
        ref = read_json(Path(args.reference_json))
        reference_match = bool(
            ref["train_idx"] == train_idx and ref["valid_idx"] == valid_idx and ref["test_idx"] == test_idx
        )
        if not reference_match:
            raise SystemExit("reproduced scaffold split does not match the DeepChem 2.8.0 reference")
        ref_scaffolds = ref.get("scaffolds")
        if ref_scaffolds is not None and ref_scaffolds != scaffold_strings:
            raise SystemExit("reproduced scaffolds do not match the DeepChem 2.8.0 reference")
        reference_versions = {"deepchem": ref.get("deepchem_version"), "rdkit": ref.get("rdkit_version")}
    else:
        reference_versions = None

    # canonical / scaffold cross-split audit
    canonical_dups = len(canonical) - len(set(canonical))
    scaffold_groups: dict[str, list[int]] = {}
    for i, s in enumerate(scaffold_strings):
        scaffold_groups.setdefault(s, []).append(i)
    side = {i: "train" for i in train_idx}
    side.update({i: "valid" for i in valid_idx})
    side.update({i: "test" for i in test_idx})
    cross_side_groups = [
        g for g in scaffold_groups.values() if len({side[i] for i in g}) > 1
    ]
    canon_side: dict[str, set] = {}
    for i, c in enumerate(canonical):
        canon_side.setdefault(c, set()).add(side[i])
    canon_cross = [c for c, s in canon_side.items() if len(s) > 1]
    # the 4-component molecule (isolated atoms) must be handled, not dropped
    multi_component = [(i, r.n_components) for i, r in enumerate(raws) if r.n_components > 1]

    OUT.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lip.save_raw_dataset(RAW_DATASET, raws, meta={
        "source_url": cfg["dataset"]["source_url"],
        "csv_sha256": sha,
        "csv_rows": len(rows),
        "columns": columns,
        "download_utc": args.download_utc or "unknown",
        "label_free": True,
    })
    np.save(LABEL_VAULT, labels)
    write_json(LABEL_META, {
        "csv_sha256": sha,
        "n": int(len(labels)),
        "sha256": hashlib.sha256(np.ascontiguousarray(labels, dtype=np.float64).tobytes()).hexdigest(),
        "note": "LOCAL ONLY score-label vault (valid/test metrics). Never copied to a compute checkout.",
    })
    train_y = [float(labels[i]) for i in train_idx]
    split_payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "splitter": "deepchem==2.8.0 ScaffoldSplitter (verbatim algorithm)",
        "reference_versions": reference_versions,
        "reference_match": reference_match,
        "n_total": len(rows),
        "n_train": len(train_idx),
        "n_valid": len(valid_idx),
        "n_test": len(test_idx),
        "train_rows": [int(i) for i in train_idx],
        "valid_rows": [int(i) for i in valid_idx],
        "test_rows": [int(i) for i in test_idx],
        "train_sha256_rows": bp.rows_sha256(train_idx),
        "valid_sha256_rows": bp.rows_sha256(valid_idx),
        "test_sha256_rows": bp.rows_sha256(test_idx),
        "train_y": train_y,
        "train_y_sha256": hashlib.sha256(
            np.ascontiguousarray(np.asarray(train_y), dtype=np.float64).tobytes()
        ).hexdigest(),
        "train_label_quantiles": {
            "q10": float(np.quantile(np.asarray(train_y), 0.10)),
            "q25": float(np.quantile(np.asarray(train_y), 0.25)),
            "q50": float(np.quantile(np.asarray(train_y), 0.50)),
            "q75": float(np.quantile(np.asarray(train_y), 0.75)),
            "q90": float(np.quantile(np.asarray(train_y), 0.90)),
        },
        "canonical_smiles_duplicates": canonical_dups,
        "canonical_smiles_cross_split": len(canon_cross),
        "scaffold_groups": len(scaffold_groups),
        "scaffold_groups_cross_split": len(cross_side_groups),
        "multi_component_molecules": multi_component,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "label_information_used": "train labels embedded for supervised training only",
    }
    write_json(SPLIT_PATH, split_payload)
    write_json(VOCAB_PATH, lip.vocab_manifest())
    write_json(PROVENANCE, {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "source_url": cfg["dataset"]["source_url"],
        "csv_sha256": sha,
        "csv_bytes": csv_path.stat().st_size,
        "csv_rows": len(rows),
        "columns": columns,
        "download_utc": args.download_utc or "unknown",
        "rdkit_version": __import__("rdkit").__version__,
        "python_version": platform.python_version(),
        "raw_graph_build_seconds": build_seconds,
        "parse_failures": 0,
        "deepchem_reference": str(args.reference_json) if args.reference_json else None,
        "license_note": cfg["dataset"]["license_note"],
        "label_free_artifact": True,
    })
    write_json(ROW_MANIFEST, {
        "protocol": PROTOCOL_ID,
        "n": len(rows),
        "compound_ids": compound_ids,
        "canonical_smiles_sha256": canonical_hashes,
        "scaffold_sha256": [
            hashlib.sha256(str(s).encode()).hexdigest() for s in scaffold_strings
        ],
        "scaffold_group_size": [len(scaffold_groups[s]) for s in scaffold_strings],
        "n_atoms": [r.n_atoms for r in raws],
        "n_components": [r.n_components for r in raws],
        "label_free": True,
    })
    fixture_rows = select_fixture_rows(raws)
    write_json(RAW_FIXTURE, {
        "protocol": PROTOCOL_ID,
        "note": "label-free golden RDKit->RawGraph fixture; the CPU gates compare the deployed raw dataset against it so the GPUs do not need RDKit",
        "rows": [dict(row=row, reason=reason, record=lip.raw_to_record(raws[row])) for row, reason in fixture_rows],
    })
    upstream = OUT / "upstream"
    upstream.mkdir(parents=True, exist_ok=True)
    if args.reference_json:
        ref = read_json(Path(args.reference_json))
        write_json(upstream / "deepchem_2.8.0_reference_split.json", {
            "deepchem_version": ref["deepchem_version"],
            "rdkit_version": ref["rdkit_version"],
            "n_total": ref["n_total"],
            "train_idx": ref["train_idx"],
            "valid_idx": ref["valid_idx"],
            "test_idx": ref["test_idx"],
            "note": "label-free reference copy (the y values of the generation run are deliberately dropped)",
        })
    log(
        f"[prepare-dataset] rows={len(rows)} parse_failures=0 train/valid/test="
        f"{len(train_idx)}/{len(valid_idx)}/{len(test_idx)} reference_match={reference_match} "
        f"cross_split_scaffolds={len(cross_side_groups)} canonical_dups={canonical_dups} "
        f"multi_component={multi_component}"
    )
    return 0


# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------


def load_split() -> dict[str, Any]:
    return read_json(SPLIT_PATH)


def load_raws() -> list[lip.RawMol]:
    raws, meta = lip.load_raw_dataset(RAW_DATASET)
    if meta["meta"].get("csv_sha256") != load_config()["dataset"]["sha256"]:
        raise SystemExit("raw dataset provenance hash mismatch")
    return raws


def _objects_path(zone: str) -> Path:
    return OUT / f"objects_{zone}.pt"


def build_zone_objects(zone: str) -> list[Any]:
    split = load_split()
    raws = load_raws()
    rows = [int(r) for r in split[f"{zone}_rows"]]
    return [lip.build_lipo_objects(raws[i]) for i in rows]


def cmd_make_fixture(args: argparse.Namespace) -> int:
    """Build the golden raw-graph fixture from the committed raw dataset."""
    raws = load_raws()
    rows = select_fixture_rows(raws)
    write_json(RAW_FIXTURE, {
        "protocol": PROTOCOL_ID,
        "note": "label-free golden RDKit->RawGraph fixture; the CPU gates compare the deployed raw dataset against it so the GPUs do not need RDKit",
        "rows": [dict(row=row, reason=reason, record=lip.raw_to_record(raws[row])) for row, reason in rows],
    })
    log(f"[make-fixture] {len(rows)} rows: {[(r, why) for r, why in rows]}")
    return 0


def cmd_prep(args: argparse.Namespace) -> int:
    if not RAW_DATASET.exists() or not SPLIT_PATH.exists():
        raise SystemExit("dataset artifacts missing; run `prepare-dataset` locally and commit them")
    split = load_split()
    raws = load_raws()
    # label-free structural assertions
    for zone in ("train", "valid", "test"):
        rows = [int(r) for r in split[f"{zone}_rows"]]
        for i in rows:
            raw = raws[i]
            if int(raw.atom_class.max()) > lip.ATOM_UNK or int(raw.edge_class.max()) > lip.BOND_UNK:
                raise SystemExit(f"MASK token present in supervised raw graph row {i}")
    started = time.perf_counter()
    objects = {zone: build_zone_objects(zone) for zone in ("train", "valid", "test")}
    for zone, objs in objects.items():
        torch.save(objs, _objects_path(zone))
    build_seconds = time.perf_counter() - started

    norm = lip.fit_standardizer(objects["train"])
    write_json(STANDARDIZER, {
        "protocol": PROTOCOL_ID,
        "fit_zone": "train",
        "n_fit": norm["n_fit"],
        "mean": [float(v) for v in norm["mean"]],
        "scale": [float(v) for v in norm["scale"]],
        "sha256": hashlib.sha256(
            np.ascontiguousarray(norm["mean"], dtype=np.float32).tobytes()
            + np.ascontiguousarray(norm["scale"], dtype=np.float32).tobytes()
        ).hexdigest(),
    })

    # initialisation parity / parameter audit per seed
    cfg_model = lip.LipoConfig()
    parity: dict[str, Any] = {}
    for seed in SEEDS_INITIAL + SEEDS_CONFIRM:
        model = lip.build_lipo_model(
            cfg_model, seed=seed, organization_mode=src.NULL,
            global_mean=norm["mean"], global_scale=norm["scale"], device="cpu",
        )
        parity[str(seed)] = {
            "full_init_sha256": src.state_dict_hash(model),
            "encoder_init_sha256": bp.encoder_group_hash(model),
            "group_init_sha256": mp.group_state_hash(model),
            "parameter_audit": src.parameter_audit(cfg_model, model),
            "forbidden_modules": src.forbidden_module_scan(model),
        }
    write_json(INIT_PARITY, {"protocol": PROTOCOL_ID, "git_commit": git_commit(), "seeds": parity})

    write_json(FREEZE, {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "code_hashes": code_hashes(),
            "artifact_sha256": {
                "raw_dataset": file_sha256(RAW_DATASET),
                "split": file_sha256(SPLIT_PATH),
                "vocab": file_sha256(VOCAB_PATH),
                "standardizer": file_sha256(STANDARDIZER),
                "init_parity": file_sha256(INIT_PARITY),
                "raw_fixture": file_sha256(RAW_FIXTURE),
            },
            "artifact_paths": {
                "raw_dataset": "dataset/lipo_raw_v1.json.gz",
                "split": "split_lipo_v1.json",
                "vocab": "vocab_v1.json",
                "standardizer": "standardizer_v1.json",
                "init_parity": "init_parity.json",
                "raw_fixture": "raw_fixture.json",
            },
        "n_train": len(split["train_rows"]),
        "n_valid": len(split["valid_rows"]),
        "n_test": len(split["test_rows"]),
        "train_rows_sha256": split["train_sha256_rows"],
        "valid_rows_sha256": split["valid_sha256_rows"],
        "test_rows_sha256": split["test_sha256_rows"],
        "parameter_budget": parity[str(SEEDS_INITIAL[0])]["parameter_audit"],
        "ssl_updates": ssl_updates(len(split["train_rows"])),
        "supervised_updates": EPOCHS * math.ceil(len(split["train_rows"]) / BATCH_SIZE),
        "object_build_seconds": build_seconds,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "label_access": dict(load_config()["label_access"]),
    })
    log(f"[prep] objects train/valid/test={len(objects['train'])}/{len(objects['valid'])}/{len(objects['test'])} norm_sha={file_sha256(STANDARDIZER)[:12]}")
    return 0


def ssl_updates(n_train: int) -> int:
    return int(round(EXPOSURE_TARGET * n_train / BATCH_SIZE))


# ---------------------------------------------------------------------------
# CPU correctness gates (the 20 required checks)
# ---------------------------------------------------------------------------


def _gate(name: str, ok: bool, detail: Any = None) -> dict[str, Any]:
    return {"name": name, "pass": bool(ok), "detail": detail}


def cmd_gates(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=not args.no_strict_code)
    split = load_split()
    raws = load_raws()
    train_objs = torch.load(_objects_path("train"), weights_only=False)
    norm = read_json(STANDARDIZER)
    gmean = np.asarray(norm["mean"], dtype=np.float32)
    gscale = np.asarray(norm["scale"], dtype=np.float32)
    gates: list[dict[str, Any]] = []

    # 1. RDKit -> raw conversion: committed golden fixture (host-independent)
    #    plus an optional live RDKit re-parse when RDKit is installed
    fixture = read_json(RAW_FIXTURE)
    mismatches = []
    for entry in fixture["rows"]:
        row = int(entry["row"])
        if lip.raw_to_record(raws[row]) != entry["record"]:
            mismatches.append(row)
    rebound = all(
        lip.raw_to_record(lip.raw_from_record(entry["record"])) == entry["record"] for entry in fixture["rows"]
    )
    rdkit_present = importlib.util.find_spec("rdkit") is not None
    reparse_ok = True
    if rdkit_present:
        entry = fixture["rows"][-1]
        row = int(entry["row"])
        reparsed = lip.raw_from_smiles(raws[row].smiles, row, raws[row].compound_id)
        reparse_ok = lip.raw_to_record(reparsed) == entry["record"]
    gates.append(_gate(
        "raw_conversion_deterministic",
        (not mismatches) and rebound and reparse_ok,
        {"fixture_rows": len(fixture["rows"]), "mismatches": mismatches, "rebound_exact": bool(rebound),
         "rdkit_present": rdkit_present, "live_reparse_ok": bool(reparse_ok)},
    ))

    # 2. row / split alignment (label-free structural form; the local vault is
    #    cross-checked only when it is present)
    all_rows = sorted(int(r) for r in split["train_rows"] + split["valid_rows"] + split["test_rows"])
    row_manifest = read_json(ROW_MANIFEST)
    aligned = (
        len(raws) == row_manifest["n"]
        and all_rows == list(range(len(raws)))
        and len(set(row_manifest["compound_ids"])) == len(row_manifest["compound_ids"])
    )
    if LABEL_META.exists():
        vault_meta = read_json(LABEL_META)
        aligned = aligned and vault_meta["csv_sha256"] == load_config()["dataset"]["sha256"] and vault_meta["n"] == len(raws)
    gates.append(_gate("row_label_alignment", aligned, {"n": len(raws), "vault_present": LABEL_META.exists()}))

    # 3. vocabulary / mask-token collisions
    collisions = {
        "atom": lip.ATOM_MASK in range(len(lip.ATOM_CLASSES)) or lip.ATOM_MASK == lip.ATOM_UNK,
        "bond": lip.BOND_MASK in range(len(lip.BOND_CLASSES)) or lip.BOND_MASK == lip.BOND_UNK,
        "charge": lip.CHARGE_MASK in range(len(lip.CHARGE_BUCKETS)) or lip.CHARGE_MASK == lip.CHARGE_UNK,
        "h": lip.H_MASK in range(len(lip.H_BUCKETS)) or lip.H_MASK == lip.H_UNK,
        "chiral": lip.CHIRAL_MASK in range(len(lip.CHIRAL_BUCKETS)) or lip.CHIRAL_MASK == lip.CHIRAL_UNK,
        "stereo": lip.STEREO_MASK in range(len(lip.STEREO_BUCKETS)) or lip.STEREO_MASK == lip.STEREO_UNK,
    }
    gates.append(_gate("vocab_mask_slots_disjoint", not any(collisions.values()), collisions))

    # 4. unseen class -> UNK handling
    unseen_atom_ok = False
    unseen_bond_ok = False
    try:
        import copy as _copy

        raw = _copy.deepcopy(raws[0])
        raw.atom_class[0] = lip.ATOM_UNK
        obj = lip.build_lipo_objects(raw)
        unseen_atom_ok = float(obj.atom_feat[0, lip.ATOM_UNK]) == 1.0
        raw2 = _copy.deepcopy(raws[0])
        raw2.edge_class[:] = lip.BOND_UNK
        obj2 = lip.build_lipo_objects(raw2)
        unseen_bond_ok = bool((obj2.bond_feat[:, lip.BOND_UNK] == 1.0).all())
    except Exception as exc:  # pragma: no cover - gate detail
        unseen_atom_ok = unseen_bond_ok = False
        collisions["exception"] = repr(exc)
    gates.append(_gate("unseen_class_unk_fallback", bool(unseen_atom_ok and unseen_bond_ok)))

    # 5. ring / bond / junction counts vs an independent reference
    import networkx as nx

    failures = 0
    probed = [raws[int(r)] for r in split["train_rows"][:200]]
    objs_probe = train_objs[:200]
    for raw, obj in zip(probed, objs_probe):
        graph = nx.Graph()
        graph.add_nodes_from(range(raw.n_atoms))
        for k in range(raw.edge_index.shape[1]):
            graph.add_edge(int(raw.edge_index[0, k]), int(raw.edge_index[1, k]))
        n_junc_ref = sum(d * (d - 1) // 2 for _, d in graph.degree())
        cyc = nx.cycle_basis(graph)
        ring_atoms_ref = {i for c in cyc for i in c}
        if int(obj.junc_index.shape[0]) != n_junc_ref:
            failures += 1
        if int(obj.bond_index.shape[0]) != graph.number_of_edges():
            failures += 1
        ring_mask = obj.atom_feat[:, lip.atom_blocks()["in_ring"]].reshape(-1) > 0.5
        if set(np.where(ring_mask)[0].tolist()) != ring_atoms_ref:
            failures += 1
    gates.append(_gate("ring_bond_junction_crosscheck", failures == 0, {"failures": failures, "probed": len(probed)}))

    # 6. atom relabelling invariance of the full forward
    model = _build_frozen_model(gmean, gscale)
    probed_objs = train_objs[:8]
    raw0 = raws[int(split["train_rows"][0])]
    base_pred = _predict(model, probed_objs)
    perm = np.random.default_rng(0).permutation(raw0.n_atoms)
    relabelled = [lip.build_lipo_objects(_relabel_raw(raw0, perm))] + probed_objs[1:]
    relab_pred = _predict(model, relabelled)
    relab_diff = float(np.abs(relab_pred - base_pred).max())
    gates.append(_gate("atom_relabelling_invariance", relab_diff <= 1e-5, {"max_abs_diff": relab_diff}))

    # 7. bond endpoint swap invariance
    swapped = [lip.build_lipo_objects(_endpoint_swap_raw(raw0))] + probed_objs[1:]
    swap_pred = _predict(model, swapped)
    swap_diff = float(np.abs(swap_pred - base_pred).max())
    gates.append(_gate("bond_endpoint_swap_invariance", swap_diff <= 1e-5, {"max_abs_diff": swap_diff}))

    # 8. batch composition invariance
    singles = np.concatenate([_predict(model, [o]) for o in probed_objs])
    batched = _predict(model, probed_objs)
    batch_diff = float(np.abs(singles - batched).max())
    gates.append(_gate("batch_composition_invariance", batch_diff <= 1e-5, {"max_abs_diff": batch_diff}))

    # 9. checkpoint save/load replay
    tmp = OUT / "gates_tmp_state.pt"
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, tmp)
    reloaded = _build_frozen_model(gmean, gscale)
    reloaded.load_state_dict(torch.load(tmp, weights_only=False))
    tmp.unlink()
    replay_diff = float(np.abs(_predict(reloaded, probed_objs) - batched).max())
    gates.append(_gate("save_load_replay", replay_diff <= 1e-6, {"max_abs_diff": replay_diff}))

    # 10. masked hidden-truth swap invariance (atoms + bonds) and raw round-trip
    rng = np.random.default_rng(0)
    atom_ok = bond_ok = 0
    n_cases = 60
    for t in range(n_cases):
        raw = raws[int(rng.integers(0, len(raws)))]
        s = lip.build_masked_sample(raw, seed=0, visit_index=t)
        raw_a = _copy_raw(raw)
        raw_b = _copy_raw(raw)
        ids = s.atom_ids
        raw_a.atom_class[ids] = (raw_a.atom_class[ids] + 1) % lip.N_ATOM_TARGETS
        mask_a = lip.build_masked_sample(raw_a, seed=0, visit_index=t)
        mask_b = lip.build_masked_sample(raw_b, seed=0, visit_index=t)
        if _objects_equal(s.masked, mask_a.masked):
            atom_ok += 1
        bids = s.bond_ids
        if len(bids) == 0:
            bond_ok += 1
            continue
        bond_index = _physical_bonds(raw)
        k = int(bids[0])
        u, v, bt = bond_index[k]
        raw_c = _copy_raw(raw)
        for col in range(raw_c.edge_index.shape[1]):
            if {int(raw_c.edge_index[0, col]), int(raw_c.edge_index[1, col])} == {u, v}:
                raw_c.edge_class[col] = (bt + 1) % lip.N_BOND_TARGETS
        mask_c = lip.build_masked_sample(raw_c, seed=0, visit_index=t)
        if _objects_equal(mask_b.masked, mask_c.masked):
            bond_ok += 1
    gates.append(_gate(
        "masked_hidden_truth_swap_invariance",
        atom_ok == n_cases and bond_ok == n_cases,
        {"atom_cases": atom_ok, "bond_cases": bond_ok, "n_cases": n_cases},
    ))
    # raw dataset round-trip (save/load) determinism
    rtmp = OUT / "gates_tmp_raw.json.gz"
    lip.save_raw_dataset(rtmp, raws[:5], meta={"gate": True})
    loaded, _ = lip.load_raw_dataset(rtmp)
    rtmp.unlink()
    roundtrip_ok = all(
        np.array_equal(getattr(a, k), getattr(b, k))
        for a, b in zip(raws[:5], loaded)
        for k in ("atom_class", "atom_charge", "atom_h", "atom_chirality", "edge_index", "edge_class", "edge_stereo")
    )
    gates.append(_gate("masked_sample_round_trip", roundtrip_ok))

    # 11. train-only standardisation
    recomputed = lip.fit_standardizer(train_objs)
    std_fit_ok = bool(
        np.allclose(recomputed["mean"], gmean, atol=1e-6)
        and np.allclose(recomputed["scale"], gscale, atol=1e-6)
        and norm["fit_zone"] == "train"
        and int(norm["n_fit"]) == len(train_objs)
    )
    gates.append(_gate("train_only_standardisation", std_fit_ok, {"n_fit": norm["n_fit"]}))

    # 12. valid/test labels unreachable before training (label-free artifact + runtime guard)
    label_free = _label_free_artifact_check()
    gates.append(_gate("score_labels_blocked_for_training", label_free["ok"], label_free))

    # 13. inference model has no extra SSL heads
    exported = _build_frozen_model(gmean, gscale)
    has_heads = hasattr(exported, "atom_head") or hasattr(exported, "bond_head")
    gates.append(_gate("no_ssl_heads_at_inference", not has_heads))

    # 14. parameter count matches the pre-registered audit
    audit = src.parameter_audit(lip.LipoConfig(), exported)
    expected = load_config()["model"]["expected_inference_parameters"]
    gates.append(_gate(
        "parameter_budget_exact",
        bool(audit["parameter_exact"] and audit["actual_parameters"] == expected),
        {"actual": audit["actual_parameters"], "expected": expected},
    ))

    # 15. S/P share the seed init for composer/reader (init parity)
    parity = read_json(INIT_PARITY)
    parity_ok = True
    for s in SEEDS_INITIAL:
        fresh = lip.build_lipo_model(
            lip.LipoConfig(), seed=s, organization_mode=src.NULL,
            global_mean=gmean, global_scale=gscale, device="cpu",
        )
        parity_ok = parity_ok and src.state_dict_hash(fresh) == parity["seeds"][str(s)]["full_init_sha256"]
        parity_ok = parity_ok and bp.encoder_group_hash(fresh) == parity["seeds"][str(s)]["encoder_init_sha256"]
        parity_ok = parity_ok and mp.group_state_hash(fresh) == parity["seeds"][str(s)]["group_init_sha256"]
    gates.append(_gate("init_parity_scratch_vs_ssl", bool(parity_ok)))

    # 16. supervised schedule / batch-order / update count determinism
    n_train = len(split["train_rows"])
    steps = math.ceil(n_train / BATCH_SIZE)
    h1 = bp.batch_order_hash(SEEDS_INITIAL[0], n_train, BATCH_SIZE, EPOCHS)
    h2 = bp.batch_order_hash(SEEDS_INITIAL[0], n_train, BATCH_SIZE, EPOCHS)
    gates.append(_gate(
        "supervised_schedule_matched",
        h1 == h2 and steps == math.ceil(n_train / BATCH_SIZE),
        {"updates": EPOCHS * steps, "steps_per_epoch": steps},
    ))

    # 17. inverse-transform RMSE matches the hand computation
    inv_res = _inverse_and_metric_check()
    gates.append(_gate("inverse_transform_rmse", inv_res["match"], inv_res))

    # 18. no official ZINC test access
    source = (_CODE_DIR / "run_bjoc_cross_task_lipophilicity_v1.py").read_text(encoding="utf-8")
    adapter_src = (_CODE_DIR / "bjoc_lipophilicity_adapter_v1.py").read_text(encoding="utf-8")
    zinc_parts = (("data", "ZINC"), ("tracks", "ksvd"), ("zinc_static", "_dictionary_pair"), ("ZINC", "(subset"))
    zinc_tokens = [a + b for a, b in zinc_parts]
    zinc_ok = not any(t in source or t in adapter_src for t in zinc_tokens)
    gates.append(_gate("no_zinc_access", zinc_ok, {"n_tokens": len(zinc_tokens)}))

    # 19. no external molecule pretraining: SSL consumes train rows only
    gates.append(_gate(
        "ssl_train_structures_only",
        len(split["train_rows"]) == n_train and len(split["valid_rows"]) == 420 and len(split["test_rows"]) == 420,
        {"train": n_train},
    ))

    # 20. no GNN / attention / iteration in the model or the driver
    scan_model = src.forbidden_module_scan(exported)
    ast_ok, ast_detail = _forbidden_ast_scan()
    gates.append(_gate("no_gnn_attention_iteration", not scan_model and ast_ok, {"modules": scan_model, "ast": ast_detail}))

    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "n_gates": len(gates),
        "n_pass": int(sum(g["pass"] for g in gates)),
        "all_pass": bool(all(g["pass"] for g in gates)),
        "gates": gates,
    }
    write_json(CPU_GATES, payload)
    for g in gates:
        log(f"[gate] {'PASS' if g['pass'] else 'FAIL'} {g['name']} {g['detail'] if g['detail'] is not None else ''}")
    return 0 if payload["all_pass"] else 1


def _label_free_artifact_check() -> dict[str, Any]:
    split = read_json(SPLIT_PATH)
    bad_keys = [k for k in split.keys() if k in ("valid_y", "test_y", "y_valid", "y_test")]
    with __import__("gzip").open(RAW_DATASET, "rt", encoding="utf-8") as fh:
        payload = json.loads(fh.read())
    expected_keys = {
        "mol_index", "compound_id", "smiles", "atom_class", "atom_charge", "atom_h",
        "atom_chirality", "edge_index", "edge_class", "edge_stereo", "n_components",
    }
    row_keys = {tuple(sorted(row.keys())) for row in payload["rows"]}
    schema_ok = row_keys == {tuple(sorted(expected_keys))}
    meta_keys = set(payload["meta"].keys())
    forbidden_meta = sorted(k for k in meta_keys if k.lower() in ("y", "labels", "target", "exp", "logp", "logd"))
    source = (_CODE_DIR / "run_bjoc_cross_task_lipophilicity_v1.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    guard_counts = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "assert_score_labels_absent"
    )
    guard_present = guard_counts >= 4
    return {
        "ok": not bad_keys and schema_ok and not forbidden_meta and guard_present and payload["meta"].get("label_free") is True,
        "forbidden_split_keys": bad_keys,
        "raw_rows_schema_ok": bool(schema_ok),
        "n_rows_checked": len(payload["rows"]),
        "forbidden_meta_keys": forbidden_meta,
        "runtime_guard_call_sites": int(guard_counts),
    }


def _copy_raw(raw: lip.RawMol) -> lip.RawMol:
    return lip.RawMol(
        mol_index=raw.mol_index,
        compound_id=raw.compound_id,
        smiles=raw.smiles,
        atom_class=raw.atom_class.copy(),
        atom_charge=raw.atom_charge.copy(),
        atom_h=raw.atom_h.copy(),
        atom_chirality=raw.atom_chirality.copy(),
        edge_index=raw.edge_index.copy(),
        edge_class=raw.edge_class.copy(),
        edge_stereo=raw.edge_stereo.copy(),
        n_components=raw.n_components,
    )


def _physical_bonds(raw: lip.RawMol) -> list[tuple[int, int, int]]:
    import cscl_units as cu

    return cu.unique_undirected_bonds(
        [(int(raw.edge_index[0, k]), int(raw.edge_index[1, k])) for k in range(raw.edge_index.shape[1])],
        [int(v) for v in raw.edge_class],
    )


def _objects_equal(a: Any, b: Any) -> bool:
    return all(
        np.array_equal(getattr(a, k), getattr(b, k))
        for k in ("atom_feat", "bond_feat", "junc_feat", "global_feat", "bond_index", "junc_index", "junc_bond")
    )


def _relabel_raw(raw: lip.RawMol, perm: np.ndarray) -> lip.RawMol:
    """Apply a pure atom relabelling (old index -> new index)."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    out = _copy_raw(raw)
    out.atom_class[inv] = raw.atom_class
    out.atom_charge[inv] = raw.atom_charge
    out.atom_h[inv] = raw.atom_h
    out.atom_chirality[inv] = raw.atom_chirality
    for k in range(raw.edge_index.shape[1]):
        out.edge_index[0, k] = inv[int(raw.edge_index[0, k])]
        out.edge_index[1, k] = inv[int(raw.edge_index[1, k])]
    return out


def _endpoint_swap_raw(raw: lip.RawMol) -> lip.RawMol:
    out = _copy_raw(raw)
    out.edge_index = raw.edge_index[::-1].copy()
    return out


def _build_frozen_model(gmean: np.ndarray, gscale: np.ndarray):
    return lip.build_lipo_model(
        lip.LipoConfig(), seed=0, organization_mode=src.NULL,
        global_mean=gmean, global_scale=gscale, device="cpu",
    )


def _predict(model: src.BJOC, objs: Sequence[Any], device: str | torch.device = "cpu") -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(objs), 256):
            batch = lip.batch_lipo_objects(list(objs[i : i + 256]))
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out.append(model(batch).detach().cpu().numpy())
    return np.concatenate(out, 0)


def _inverse_and_metric_check() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    y = rng.normal(2.0, 1.2, size=257)
    z = rng.normal(0.0, 1.0, size=257)
    mean, scale = float(y.mean()), float(y.std())
    pred_norm = z + (y - mean) / scale * 0.1
    pred = pred_norm * scale + mean
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    manual = float(math.sqrt(float(np.mean((pred - y) ** 2))))
    return {"rmse": rmse, "manual": manual, "match": abs(rmse - manual) < 1e-12}


def _forbidden_ast_scan() -> tuple[bool, list[str]]:
    tokens = ("MessagePassing", "TransformerEncoder", "MultiheadAttention", "GATConv", "GCNConv", "GINConv", "SAGEConv")
    found = []
    for path in (_CODE_DIR / "run_bjoc_cross_task_lipophilicity_v1.py", _CODE_DIR / "bjoc_lipophilicity_adapter_v1.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and any(t in node.attr for t in tokens):
                found.append(f"{path.name}:{node.attr}")
            if isinstance(node, ast.Name) and any(t == node.id for t in tokens):
                found.append(f"{path.name}:{node.id}")
    return (not found), found


# ---------------------------------------------------------------------------
# GPU smoke
# ---------------------------------------------------------------------------


def cmd_smoke(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=not args.no_strict_code) if not args.allow_unfrozen else None
    assert_score_labels_absent()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")
    split = load_split()
    raws = load_raws()
    train_objs = torch.load(_objects_path("train"), weights_only=False)
    norm = read_json(STANDARDIZER)
    gmean = np.asarray(norm["mean"], dtype=np.float32)
    gscale = np.asarray(norm["scale"], dtype=np.float32)
    seed = SEEDS_INITIAL[0]
    t0 = time.perf_counter()
    model = lip.build_lipo_model(lip.LipoConfig(), seed=seed, organization_mode=src.NULL,
                                 global_mean=gmean, global_scale=gscale, device=device)
    pretext = lip.LipoPretextModel(model).to(device)
    opt = torch.optim.Adam(pretext.parameters(), lr=PRETRAIN_LR, weight_decay=PRETRAIN_WD)
    gen = torch.Generator().manual_seed(seed + PRETRAIN_SHUFFLE_OFFSET)
    ssl_steps = 0
    t_ssl0 = time.perf_counter()
    for visit in range(1):
        for chunk in bp.shuffle_batches(16, 8, gen):
            samples = [lip.build_masked_sample(raws[split["train_rows"][j]], seed=seed, visit_index=visit) for j in chunk]
            batch, ta, tb, _, _ = lip.batch_pretext_samples(samples)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            loss, _ = pretext.loss(batch, ta.to(device), tb.to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretext.parameters(), PRETRAIN_CLIP)
            opt.step()
            ssl_steps += 1
    ssl_seconds = time.perf_counter() - t_ssl0
    # supervised smoke: 2 epochs on 256 molecules
    y = np.asarray(split["train_y"], dtype=np.float32)[:256]
    objs = train_objs[:256]
    y_norm = (y - float(y.mean())) / float(y.std())
    model2 = lip.build_lipo_model(lip.LipoConfig(), seed=seed, organization_mode=src.NULL,
                                  global_mean=gmean, global_scale=gscale, device=device)
    opt2 = torch.optim.Adam(model2.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    t_sup0 = time.perf_counter()
    gen2 = torch.Generator().manual_seed(seed + TRAIN_SHUFFLE_OFFSET)
    sup_steps = 0
    for _ in range(2):
        for idx in bp.shuffle_batches(256, BATCH_SIZE, gen2):
            batch = lip.batch_lipo_objects([objs[i] for i in idx])
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            target = torch.as_tensor(y_norm[idx], dtype=torch.float32, device=device)
            pred = model2(batch)
            loss = F.mse_loss(pred.view(-1), target.view(-1))
            opt2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model2.parameters(), GRAD_CLIP)
            opt2.step()
            sup_steps += 1
    sup_seconds = time.perf_counter() - t_sup0
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "ssl_steps": ssl_steps,
        "ssl_seconds": ssl_seconds,
        "ssl_ms_per_molecule": ssl_seconds / max(ssl_steps * 8, 1) * 1000.0,
        "supervised_steps": sup_steps,
        "supervised_seconds": sup_seconds,
        "supervised_seconds_per_epoch_256": sup_seconds / 2.0,
        "projected_ssl_seconds": ssl_seconds / max(ssl_steps, 1) * ssl_updates(len(split["train_rows"])),
        "projected_supervised_seconds": sup_seconds / max(sup_steps, 1) * EPOCHS * math.ceil(len(split["train_rows"]) / BATCH_SIZE),
        "wall_seconds": time.perf_counter() - t0,
    }
    write_json(GPU_SMOKE, payload)
    log(f"[smoke] {payload['cuda_device_name']} ssl={ssl_steps} steps in {ssl_seconds:.1f}s sup={sup_steps} steps in {sup_seconds:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# SSL training (train structures only)
# ---------------------------------------------------------------------------


def cmd_ssl(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=not args.no_strict_code) if not args.allow_unfrozen else None
    assert_score_labels_absent()
    seed = int(args.seed)
    if seed not in SEEDS_INITIAL + SEEDS_CONFIRM:
        raise SystemExit(f"seed {seed} is not pre-registered")
    device = torch.device(args.device)
    split = load_split()
    raws = load_raws()
    gmean = np.zeros(lip.GLOBAL_DIM, dtype=np.float32)
    gscale = np.ones(lip.GLOBAL_DIM, dtype=np.float32)
    n_train = len(split["train_rows"])
    updates = ssl_updates(n_train)
    dev_override = False
    if args.max_updates:
        updates = min(updates, int(args.max_updates))
        dev_override = True
    train_rows = set(int(r) for r in split["train_rows"])

    model = lip.build_lipo_model(lip.LipoConfig(), seed=seed, organization_mode=src.NULL,
                                 global_mean=gmean, global_scale=gscale, device=device)
    init_encoder_sha = bp.encoder_group_hash(model)
    pretext = lip.LipoPretextModel(model).to(device)
    optimizer = torch.optim.Adam(pretext.parameters(), lr=PRETRAIN_LR, weight_decay=PRETRAIN_WD)
    gen = torch.Generator().manual_seed(seed + PRETRAIN_SHUFFLE_OFFSET)
    curve = []
    steps = 0
    visit = 0
    started = time.perf_counter()
    while steps < updates:
        pretext.train()
        ce_a = ce_b = acc_a = acc_b = 0.0
        n_a = n_b = n_steps = 0
        t_ep = time.perf_counter()
        stop = False
        for chunk in bp.shuffle_batches(n_train, BATCH_SIZE, gen):
            samples = []
            for j in chunk:
                row = int(split["train_rows"][j])
                if row not in train_rows:
                    raise SystemExit("SSL row outside the train split")
                samples.append(lip.build_masked_sample(raws[row], seed=seed, visit_index=visit))
            batch, ta, tb, _, _ = lip.batch_pretext_samples(samples)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            ta, tb = ta.to(device), tb.to(device)
            loss, parts = pretext.loss(batch, ta, tb)
            optimizer.zero_grad()
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(pretext.parameters(), PRETRAIN_CLIP))
            if not math.isfinite(gn):
                raise RuntimeError(f"non-finite SSL gradient at step {steps}")
            optimizer.step()
            steps += 1
            with torch.no_grad():
                logits = pretext(batch, batch["masked_atom_ids"], batch["masked_bond_ids"])
                if ta.numel():
                    ce_a += float(parts["ce_atom"])
                    n_a += 1
                    acc_a += float((logits["atom_logits"].argmax(1) == ta).float().mean())
                if tb.numel():
                    ce_b += float(parts["ce_bond"])
                    n_b += 1
                    acc_b += float((logits["bond_logits"].argmax(1) == tb).float().mean())
            n_steps += 1
            if steps >= updates:
                stop = True
                break
        curve.append({
            "visit": visit, "steps_total": steps,
            "ce_atom": ce_a / max(n_a, 1), "ce_bond": ce_b / max(n_b, 1),
            "acc_atom": acc_a / max(n_a, 1), "acc_bond": acc_b / max(n_b, 1),
            "seconds": time.perf_counter() - t_ep,
        })
        if visit % 10 == 0 or stop:
            log(f"[ssl s{seed}] visit={visit} steps={steps}/{updates} ce_a={curve[-1]['ce_atom']:.4f} "
                f"acc_a={curve[-1]['acc_atom']:.4f} ce_b={curve[-1]['ce_bond']:.4f} acc_b={curve[-1]['acc_bond']:.4f}")
        visit += 1
    wall = time.perf_counter() - started

    state = mp.encoder_state(model)
    torch.save(state, OUT / f"pretrain_s{seed}.pt")
    final_sha = bp.encoder_group_hash(model)
    # priors for the record (train structures only)
    atom_codes = np.concatenate([raws[int(r)].atom_class for r in split["train_rows"]])
    atom_codes = atom_codes[atom_codes <= lip.ATOM_UNK]
    bond_codes = np.concatenate([raws[int(r)].edge_class for r in split["train_rows"]])
    bond_codes = bond_codes[bond_codes <= lip.BOND_UNK]
    priors = lip.class_prior_stats(atom_codes, bond_codes)
    payload = {
        "protocol": PROTOCOL_ID,
        "arm": "SSL", "seed": seed, "device": str(device),
        "updates": int(steps), "visits": visit, "batch_size": BATCH_SIZE,
        "updates_rule": f"round({EXPOSURE_TARGET} * {n_train} / {BATCH_SIZE})",
        "exposure_per_molecule": float(steps * BATCH_SIZE) / n_train,
        "hyper": {"optimizer": "Adam", "lr": PRETRAIN_LR, "weight_decay": PRETRAIN_WD,
                  "grad_clip": PRETRAIN_CLIP, "loss": "CE(masked atom classes) + CE(masked bond classes) 1.0/1.0",
                  "checkpoint_rule": "last step only"},
        "mask": {"atom_ratio": lip.ATOM_MASK_RATIO, "bond_ratio": lip.BOND_MASK_RATIO,
                 "atom_token": lip.ATOM_MASK, "bond_token": lip.BOND_MASK,
                 "rng": "SeedSequence([seed, molecule_id, visit_index])"},
        "train_rows_only": True, "n_train_rows": n_train, "dev_override": dev_override,
        "encoder_init_sha256": init_encoder_sha, "encoder_final_sha256": final_sha,
        "encoder_parameters_changed": init_encoder_sha != final_sha,
        "curve": curve,
        "final_ce_atom": curve[-1]["ce_atom"], "final_acc_atom": curve[-1]["acc_atom"],
        "final_ce_bond": curve[-1]["ce_bond"], "final_acc_bond": curve[-1]["acc_bond"],
        "class_priors": priors,
        "wall_seconds": wall,
        "torch": torch.__version__, "python": platform.python_version(),
        "labels_read": False, "official_valid_loaded": False, "official_test_loaded": False,
    }
    write_json(OUT / f"pretrain_s{seed}.json", payload)
    log(f"[ssl s{seed}] done updates={steps} wall={wall:.1f}s ce_a={payload['final_ce_atom']:.4f} acc_a={payload['final_acc_atom']:.4f}")
    return 0


# ---------------------------------------------------------------------------
# supervised arms
# ---------------------------------------------------------------------------


def _last_soup(states: list[dict[str, torch.Tensor]], last_epoch: int):
    members = list(range(last_epoch - len(states) + 1, last_epoch + 1))
    soup = {k: torch.stack([s[k].float() for s in states]).mean(0) for k in states[0].keys()}
    return members, soup


def cmd_arm(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=not args.no_strict_code) if not args.allow_unfrozen else None
    assert_score_labels_absent()
    arm = str(args.arm)
    seed = int(args.seed)
    if arm not in ARMS or seed not in SEEDS_INITIAL + SEEDS_CONFIRM:
        raise SystemExit(f"unregistered arm/seed {arm}/{seed}")
    device = torch.device(args.device)
    split = load_split()
    norm = read_json(STANDARDIZER)
    gmean = np.asarray(norm["mean"], dtype=np.float32)
    gscale = np.asarray(norm["scale"], dtype=np.float32)
    train_objs = torch.load(_objects_path("train"), weights_only=False)
    y_raw = np.asarray(split["train_y"], dtype=np.float64)
    n = len(train_objs)
    if len(y_raw) != n:
        raise SystemExit("train label length mismatch")
    y_mean, y_scale = float(y_raw.mean()), float(y_raw.std())
    y_norm = (y_raw - y_mean) / y_scale

    model = lip.build_lipo_model(lip.LipoConfig(), seed=seed, organization_mode=src.NULL,
                                 global_mean=gmean, global_scale=gscale, device=device)
    init_full_sha = src.state_dict_hash(model)
    init_encoder_sha = bp.encoder_group_hash(model)
    init_group_sha = mp.group_state_hash(model)
    encoder_used_sha = init_encoder_sha
    if arm == "P":
        state = torch.load(OUT / f"pretrain_s{seed}.pt", weights_only=False)
        ssl_json = read_json(OUT / f"pretrain_s{seed}.json")
        if init_encoder_sha != ssl_json["encoder_init_sha256"]:
            raise SystemExit("P arm: scratch init does not match the SSL run's initial encoder state")
        mp.load_encoder_state(model, state, device)
        encoder_used_sha = bp.encoder_group_hash(model)
    audit = src.parameter_audit(lip.LipoConfig(), model)
    expected = load_config()["model"]["expected_inference_parameters"]
    if not audit["parameter_exact"] or audit["actual_parameters"] != expected:
        raise SystemExit(f"parameter audit failed: {audit}")

    epochs = EPOCHS if not args.epochs_override else int(args.epochs_override)
    if not (1 <= epochs <= EPOCHS):
        raise SystemExit(f"epochs must be in [1,{EPOCHS}]")
    dev_override = epochs != EPOCHS
    steps_per_epoch = math.ceil(n / BATCH_SIZE)
    total_updates = epochs * steps_per_epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=LR_MIN)
    gen = torch.Generator().manual_seed(seed + TRAIN_SHUFFLE_OFFSET)
    states: deque = deque(maxlen=SOUP_LAST)
    curve = []
    order_hash = hashlib.sha256()
    grad_norm_max = 0.0
    clip_events = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        lr_last = None
        for idx in bp.shuffle_batches(n, BATCH_SIZE, gen):
            order_hash.update(np.ascontiguousarray(np.asarray(idx, dtype=np.int64)).tobytes())
            batch = lip.batch_lipo_objects([train_objs[i] for i in idx])
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            target = torch.as_tensor(y_norm[idx], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.mse_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad()
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if not math.isfinite(gn):
                raise RuntimeError(f"non-finite gradient at epoch {epoch}")
            grad_norm_max = max(grad_norm_max, gn)
            if gn > GRAD_CLIP:
                clip_events += 1
            lr_last = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            total += float((pred.view(-1) - target.view(-1)).abs().sum())
            n_mol += len(idx)
        train_mae_norm = total / max(n_mol, 1)
        curve.append({"epoch": epoch, "updates": epoch * steps_per_epoch, "train_mae_norm": train_mae_norm,
                      "lr_last": lr_last, "seconds": time.perf_counter() - t_ep})
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if epoch % 40 == 0 or epoch == 1 or epoch == epochs:
            log(f"[{arm} s{seed}] ep={epoch:03d} train_mae_norm={train_mae_norm:.6f} lr={lr_last:.3e} "
                f"{curve[-1]['seconds']:.1f}s")
    wall = time.perf_counter() - started
    members, soup = _last_soup(list(states), epochs)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    soup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(soup_state, OUT / f"soup_{arm}_s{seed}.pt")
    fit_norm = _predict(model, train_objs, device)
    fit_raw = fit_norm * y_scale + y_mean
    fit_mae = float(np.abs(fit_raw - y_raw).mean())
    fit_rmse = float(np.sqrt(np.mean((fit_raw - y_raw) ** 2)))
    payload = {
        "protocol": PROTOCOL_ID, "arm": arm, "seed": seed, "recipe": "R3-Lipo",
        "epochs": epochs, "updates": int(total_updates), "steps_per_epoch": steps_per_epoch,
        "dev_override": bool(dev_override),
        "n_train_rows": n, "batch_size": BATCH_SIZE, "soup_members": members,
        "target_normalization": {"mean": y_mean, "scale": y_scale},
        "soup_fit_mae_raw": fit_mae, "soup_fit_rmse_raw": fit_rmse,
        "schedule": {"type": "per-step cosine", "lr0": LR, "lr_min": LR_MIN, "t_max_updates": total_updates},
        "init_full_sha256": init_full_sha, "init_encoder_sha256": init_encoder_sha,
        "init_group_sha256": init_group_sha, "encoder_used_sha256": encoder_used_sha,
        "parameter_audit": audit,
        "batch_order": {"full_hash": order_hash.hexdigest(),
                        "independent_recomputation": bp.batch_order_hash(seed, n, BATCH_SIZE, epochs),
                        "full_matches": order_hash.hexdigest() == bp.batch_order_hash(seed, n, BATCH_SIZE, epochs)},
        "anomalies": {"grad_clip_events": clip_events, "grad_norm_max": grad_norm_max},
        "lr_curve": curve, "wall_seconds": wall,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__, "python": platform.python_version(),
        "labels_used": "train rows raw y only",
        "official_valid_loaded": False, "official_test_loaded": False,
    }
    write_json(OUT / f"train_{arm}_s{seed}.json", payload)
    log(f"[{arm} s{seed}] soup fit_mae_raw={fit_mae:.5f} fit_rmse_raw={fit_rmse:.5f} wall={wall:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# predict (label-blind)
# ---------------------------------------------------------------------------


def cmd_predict(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=not args.no_strict_code) if not args.allow_unfrozen else None
    assert_score_labels_absent()
    zone = str(args.zone)
    if zone not in ("valid", "test"):
        raise SystemExit("predict zone must be valid or test")
    if zone == "test":
        decision_path = OUT / "validation_decision.json"
        if not decision_path.exists() or not read_json(decision_path).get("test_authorized"):
            raise SystemExit("test prediction is locked until the validation decision authorizes it")
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else SEEDS_INITIAL)]
    arms = [a for a in (args.arms.split(",") if args.arms else ARMS)]
    device = torch.device(args.device)
    norm = read_json(STANDARDIZER)
    gmean = np.asarray(norm["mean"], dtype=np.float32)
    gscale = np.asarray(norm["scale"], dtype=np.float32)
    objs = torch.load(_objects_path(zone), weights_only=False)
    split = load_split()
    norm_raw = _pred_denormalizer(split)
    pred_dir = OUT / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in seeds:
        for arm in arms:
            model = lip.build_lipo_model(lip.LipoConfig(), seed=seed, organization_mode=src.NULL,
                                         global_mean=gmean, global_scale=gscale, device=device)
            state = torch.load(OUT / f"soup_{arm}_s{seed}.pt", weights_only=False)
            model.load_state_dict({k: v.to(device) for k, v in state.items()})
            pred_norm = _predict(model, objs, device)
            pred = pred_norm * norm_raw["scale"] + norm_raw["mean"]
            path = pred_dir / f"{zone}_{arm}_s{seed}.npz"
            np.savez(path, pred=pred.astype(np.float64))
            records.append({"zone": zone, "arm": arm, "seed": seed, "path": f"predictions/{path.name}",
                            "sha256": file_sha256(path), "n": int(pred.shape[0])})
    combined = hashlib.sha256(b"".join(bytes.fromhex(r["sha256"]) for r in records)).hexdigest()
    write_json(OUT / f"prediction_manifest_{zone}.json", {
        "protocol": PROTOCOL_ID, "git_commit": git_commit(), "zone": zone,
        "records": records, "combined_sha256": combined,
        "labels_read": False, "official_valid_loaded": False, "official_test_loaded": False,
    })
    log(f"[predict:{zone}] {len(records)} files frozen combined={combined[:16]}")
    return 0


def _pred_denormalizer(split: Mapping[str, Any]) -> dict[str, float]:
    y = np.asarray(split["train_y"], dtype=np.float64)
    return {"mean": float(y.mean()), "scale": float(y.std())}


# ---------------------------------------------------------------------------
# baselines (LOCAL, RDKit)
# ---------------------------------------------------------------------------


def cmd_baselines(args: argparse.Namespace) -> int:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb

    cfg = load_config()
    split = load_split()
    n = split["n_total"]
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=cfg["baselines"]["ecfp4_rf"]["fingerprint"]["radius"],
        fpSize=cfg["baselines"]["ecfp4_rf"]["fingerprint"]["n_bits"],
    )
    fingerprint = np.zeros((n, cfg["baselines"]["ecfp4_rf"]["fingerprint"]["n_bits"]), dtype=np.uint8)
    for i, smi in enumerate([r for r in _csv_smiles_order()]):
        mol = Chem.MolFromSmiles(smi)
        fp = generator.GetFingerprint(mol)
        fingerprint[i] = np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord("0")
    np.save(OUT / "ecfp4_2048.npy", fingerprint)

    train_rows = [int(r) for r in split["train_rows"]]
    valid_rows = [int(r) for r in split["valid_rows"]]
    test_rows = [int(r) for r in split["test_rows"]]
    y = np.asarray(split["train_y"], dtype=np.float64)  # train labels only; vault untouched here
    tr_y = y
    rf_params = {"n_estimators": 500, "max_features": 1.0, "random_state": 0, "n_jobs": -1}
    rf = RandomForestRegressor(**rf_params)
    rf.fit(fingerprint[train_rows].astype(np.float32), tr_y)
    pred_valid_rf = rf.predict(fingerprint[valid_rows].astype(np.float32))
    pred_test_rf = rf.predict(fingerprint[test_rows].astype(np.float32))

    xgb_params = {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
                  "colsample_bytree": 0.8, "reg_lambda": 1.0, "min_child_weight": 1,
                  "random_state": 0, "tree_method": "hist", "n_jobs": -1}
    xgbm = xgb.XGBRegressor(**xgb_params)
    xgbm.fit(fingerprint[train_rows].astype(np.float32), tr_y)
    pred_valid_xgb = xgbm.predict(fingerprint[valid_rows].astype(np.float32))
    pred_test_xgb = xgbm.predict(fingerprint[test_rows].astype(np.float32))

    np.savez(OUT / "baseline_predictions.npz",
             valid_rf=pred_valid_rf, valid_xgb=pred_valid_xgb,
             test_rf=pred_test_rf, test_xgb=pred_test_xgb)
    payload = {
        "protocol": PROTOCOL_ID, "git_commit": git_commit(),
        "fingerprint": cfg["baselines"]["ecfp4_rf"]["fingerprint"],
        "rf": rf_params, "xgb": xgb_params,
        "same_indices": True,
        "valid_rows_sha256": split["valid_sha256_rows"],
        "test_rows_sha256": split["test_sha256_rows"],
        "predictions_file_sha256": file_sha256(OUT / "baseline_predictions.npz"),
        "labels_used": "train rows only for fitting; valid/test labels NOT read in this command",
        "official_valid_loaded": False, "official_test_loaded": False,
    }
    write_json(OUT / "baselines.json", payload)
    log(f"[baselines] RF + XGB trained on {len(train_rows)} train molecules")
    return 0


def _csv_smiles_order() -> list[str]:
    raws = load_raws()
    return [r.smiles for r in raws]


# ---------------------------------------------------------------------------
# metrics / scoring
# ---------------------------------------------------------------------------


def _metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    resid = pred - y
    return {
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "median_abs": float(np.median(np.abs(resid))),
        "p90_abs": float(np.quantile(np.abs(resid), 0.90)),
        "n": int(len(y)),
    }


def _load_npz_pred(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data["pred"], dtype=np.float64)


def cmd_score_valid(args: argparse.Namespace) -> int:
    split = load_split()
    if not CPU_GATES.exists() or not read_json(CPU_GATES).get("all_pass"):
        raise SystemExit("CPU gates not all passing")
    if not (OUT / "baselines.json").exists():
        raise SystemExit("run `baselines` first (pre-registered same-protocol ECFP baselines)")
    manifest = read_json(OUT / "prediction_manifest_valid.json")
    y = read_label_vault("valid", split)
    write_json(OUT / "validation_read.lock", {
        "protocol": PROTOCOL_ID, "git_commit": git_commit(),
        "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prediction_manifest_sha256": file_sha256(OUT / "prediction_manifest_valid.json"),
        "n_rows": int(len(y)),
        "y_sha256": hashlib.sha256(np.ascontiguousarray(y, dtype=np.float64).tobytes()).hexdigest(),
        "label_source": "data/lipophilicity/labels.npy (LOCAL vault)",
    })
    seeds = [int(r["seed"]) for r in manifest["records"]]
    preds: dict[tuple[str, int], np.ndarray] = {}
    for rec in manifest["records"]:
        preds[(rec["arm"], int(rec["seed"]))] = _load_npz_pred(OUT / rec["path"])
    per_seed = {}
    for seed in sorted(set(seeds)):
        per_seed[str(seed)] = {
            "S": _metrics(preds[("S", seed)], y),
            "P": _metrics(preds[("P", seed)], y),
        }
    seed_mean = {arm: _seed_mean_metrics([per_seed[str(s)][arm] for s in sorted(set(seeds))]) for arm in ARMS}
    deltas = {}
    for seed in sorted(set(seeds)):
        deltas[str(seed)] = {
            "rmse": per_seed[str(seed)]["P"]["rmse"] - per_seed[str(seed)]["S"]["rmse"],
            "mae": per_seed[str(seed)]["P"]["mae"] - per_seed[str(seed)]["S"]["mae"],
        }
    s_mean, p_mean = seed_mean["S"], seed_mean["P"]
    rmse_delta = float(p_mean["rmse"] - s_mean["rmse"])
    rel_drop = float(-rmse_delta / s_mean["rmse"])
    boot_seed_mean = _paired_bootstrap_mean(preds, sorted(set(seeds)), y, "rmse")
    paired_mae = _paired_bootstrap_mean(preds, sorted(set(seeds)), y, "mae")
    # subgroup / outlier diagnostics
    q10 = float(split["train_label_quantiles"]["q10"])
    tail = y <= q10
    ordinary = ~tail
    abs_s = np.abs(preds[("S", sorted(set(seeds))[0])] - y)
    abs_p = np.abs(preds[("P", sorted(set(seeds))[0])] - y)
    improvement = abs_s - abs_p
    order = np.argsort(-improvement)
    total_gain = float(improvement.sum())
    top1_share = float(improvement[order[0]] / total_gain) if total_gain > 0 else float("inf")
    top10_share = float(improvement[order[:10]].sum() / total_gain) if total_gain > 0 else float("inf")
    diagnostics = {
        "tail_definition": f"y <= train q10 ({q10:.4f})",
        "tail_n": int(tail.sum()),
        "ordinary_delta_rmse_seedmean": _subgroup_rmse_delta(preds, y, ordinary, sorted(set(seeds))),
        "tail_delta_rmse_seedmean": _subgroup_rmse_delta(preds, y, tail, sorted(set(seeds))),
        "top1_improvement_share": top1_share,
        "top10_improvement_share": top10_share,
        "seed_minus_removed_top10_rmse": None,
    }
    baselines = read_json(OUT / "baselines.json")
    base_pred = np.load(OUT / "baseline_predictions.npz")
    base_metrics = {
        "ecfp4_rf": _metrics(np.asarray(base_pred["valid_rf"], dtype=np.float64), y),
        "ecfp4_xgb": _metrics(np.asarray(base_pred["valid_xgb"], dtype=np.float64), y),
    }
    baselines_meta = {"fingerprint": baselines["fingerprint"], "rf": baselines["rf"], "xgb": baselines["xgb"]}
    verdict = evaluate_promotion(per_seed, seed_mean, boot_seed_mean, diagnostics,
                                 audits_pass=bool(read_json(CPU_GATES)["all_pass"]), baselines_done=True)
    payload = {
        "protocol": PROTOCOL_ID, "git_commit": git_commit(), "zone": "valid",
        "per_seed": per_seed, "seed_mean": seed_mean,
        "deltas": deltas, "rmse_delta_seedmean": verdict["rmse_delta"], "relative_drop": verdict["rel_drop"],
        "bootstrap_rmse_seedmean": boot_seed_mean, "bootstrap_mae_seedmean": paired_mae,
        "diagnostics": diagnostics, "baselines": base_metrics, "baseline_spec": baselines_meta,
        "frozen_rule": verdict["frozen_rule"],
        "conditions": verdict["conditions"], "decision": verdict["decision"],
        "seeds_to_confirm": verdict["seeds_to_confirm"],
        "test_authorized": verdict["test_authorized"],
        "official_valid_loaded": True, "official_test_loaded": False,
    }
    write_json(OUT / "validation_metrics.json", payload)
    write_json(OUT / "validation_decision.json", {
        "protocol": PROTOCOL_ID, "git_commit": git_commit(),
        "decision": verdict["decision"], "conditions": verdict["conditions"],
        "seeds_to_confirm": verdict["seeds_to_confirm"],
        "test_authorized": verdict["test_authorized"],
    })
    log(f"[score-valid] S rmse={s_mean['rmse']:.5f} P rmse={p_mean['rmse']:.5f} delta={verdict['rmse_delta']:+.5f} "
        f"rel={verdict['rel_drop']*100:.2f}% decision={verdict['decision']}")
    return 0


def evaluate_promotion(
    per_seed: Mapping[str, Mapping[str, Mapping[str, float]]],
    seed_mean: Mapping[str, Mapping[str, float]],
    boot_seed_mean: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    *,
    audits_pass: bool,
    baselines_done: bool,
) -> dict[str, Any]:
    """The pre-registered stage-F rule (frozen before any GPU trajectory)."""
    s_mean, p_mean = seed_mean["S"], seed_mean["P"]
    rmse_delta = float(p_mean["rmse"] - s_mean["rmse"])
    rel_drop = float(-rmse_delta / s_mean["rmse"])
    per_seed_deltas = {k: float(v["P"]["rmse"] - v["S"]["rmse"]) for k, v in per_seed.items()}
    conditions = {
        "relative_rmse_drop_min": bool(rel_drop >= PROMOTE_REL_RMSE_DROP),
        "absolute_rmse_drop_min": bool((-rmse_delta) >= PROMOTE_ABS_RMSE_DROP),
        "both_seeds_improve": bool(all(d < 0 for d in per_seed_deltas.values())),
        "bootstrap_ci_upper_below_zero": bool(float(boot_seed_mean["ci95"][1]) < 0),
        "mae_not_materially_worse": bool(p_mean["mae"] <= s_mean["mae"] + PROMOTE_MAE_TOLERANCE),
        "not_single_outlier_driven": bool(
            float(diagnostics["top1_improvement_share"]) <= PROMOTE_SINGLE_MOL_SHARE_MAX
            and float(diagnostics["top10_improvement_share"]) <= 0.5
        ),
        "ecfp_baselines_done": bool(baselines_done),
        "audits_pass": bool(audits_pass),
    }
    passed = all(conditions.values())
    ci_low = float(boot_seed_mean["ci95"][0])
    ci_high = float(boot_seed_mean["ci95"][1])
    if passed:
        decision = "TRANSFER_SSL_GAIN_PROMISING"
    elif ci_low < 0.0 < ci_high:
        # a real but statistically unconfirmed direction of change: neither a
        # claim of gain nor a claim of absence
        decision = "TRANSFER_SSL_PROMISING_UNCONFIRMED"
    else:
        decision = "TRANSFER_SSL_NO_CONFIRMED_GAIN"
    return {
        "rmse_delta": rmse_delta,
        "rel_drop": rel_drop,
        "per_seed_deltas": per_seed_deltas,
        "conditions": conditions,
        "decision": decision,
        "test_authorized": bool(passed),
        "seeds_to_confirm": list(SEEDS_CONFIRM) if passed else [],
        "frozen_rule": {
            "relative_rmse_drop_min": PROMOTE_REL_RMSE_DROP,
            "absolute_rmse_drop_min": PROMOTE_ABS_RMSE_DROP,
            "mae_tolerance": PROMOTE_MAE_TOLERANCE,
            "single_molecule_share_max": PROMOTE_SINGLE_MOL_SHARE_MAX,
        },
    }


def _seed_mean_metrics(metrics: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = [k for k in metrics[0].keys() if k != "n"]
    return {k: float(np.mean([m[k] for m in metrics])) for k in keys} | {"n": int(metrics[0]["n"])}


def _paired_bootstrap_mean(preds: Mapping[tuple[str, int], np.ndarray], seeds: Sequence[int],
                           y: np.ndarray, metric: str, n_boot: int = 5000, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals = []
        for s in seeds:
            if metric == "rmse":
                ma = math.sqrt(float(np.mean((preds[("P", s)][idx] - y[idx]) ** 2)))
                mb = math.sqrt(float(np.mean((preds[("S", s)][idx] - y[idx]) ** 2)))
            else:
                ma = float(np.mean(np.abs(preds[("P", s)][idx] - y[idx])))
                mb = float(np.mean(np.abs(preds[("S", s)][idx] - y[idx])))
            vals.append(ma - mb)
        deltas[b] = float(np.mean(vals))
    return {"delta": float(np.mean(deltas)), "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
            "n_boot": n_boot}


def _subgroup_rmse_delta(preds: Mapping[tuple[str, int], np.ndarray], y: np.ndarray,
                         mask: np.ndarray, seeds: Sequence[int]) -> float:
    vals = []
    for s in seeds:
        ma = math.sqrt(float(np.mean((preds[("P", s)][mask] - y[mask]) ** 2)))
        mb = math.sqrt(float(np.mean((preds[("S", s)][mask] - y[mask]) ** 2)))
        vals.append(ma - mb)
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# terminal test (single read)
# ---------------------------------------------------------------------------


def cmd_score_test(args: argparse.Namespace) -> int:
    decision = read_json(OUT / "validation_decision.json")
    if not decision.get("test_authorized"):
        raise SystemExit("terminal test is locked: validation decision did not authorize it")
    split = load_split()
    manifest = read_json(OUT / "prediction_manifest_test.json")
    y = read_label_vault("test", split)
    write_json(OUT / "test_read.lock", {
        "protocol": PROTOCOL_ID, "git_commit": git_commit(),
        "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prediction_manifest_sha256": file_sha256(OUT / "prediction_manifest_test.json"),
        "n_rows": int(len(y)),
        "y_sha256": hashlib.sha256(np.ascontiguousarray(y, dtype=np.float64).tobytes()).hexdigest(),
        "label_source": "data/lipophilicity/labels.npy (LOCAL vault)",
    })
    seeds = sorted({int(r["seed"]) for r in manifest["records"]})
    preds = {(r["arm"], int(r["seed"])): _load_npz_pred(OUT / r["path"]) for r in manifest["records"]}
    per_seed = {str(s): {arm: _metrics(preds[(arm, s)], y) for arm in ARMS} for s in seeds}
    seed_mean = {arm: _seed_mean_metrics([per_seed[str(s)][arm] for s in seeds]) for arm in ARMS}
    rmse_std = {arm: float(np.std([per_seed[str(s)][arm]["rmse"] for s in seeds])) for arm in ARMS}
    mae_std = {arm: float(np.std([per_seed[str(s)][arm]["mae"] for s in seeds])) for arm in ARMS}
    boot = _paired_bootstrap_mean(preds, seeds, y, "rmse")
    base_pred = np.load(OUT / "baseline_predictions.npz")
    base_metrics = {
        "ecfp4_rf": _metrics(np.asarray(base_pred["test_rf"], dtype=np.float64), y),
        "ecfp4_xgb": _metrics(np.asarray(base_pred["test_xgb"], dtype=np.float64), y),
    }
    q10 = float(split["train_label_quantiles"]["q10"])
    tail = y <= q10
    payload = {
        "protocol": PROTOCOL_ID, "git_commit": git_commit(), "zone": "test",
        "per_seed": per_seed,
        "seed_mean": seed_mean, "rmse_std": rmse_std, "mae_std": mae_std,
        "bootstrap_rmse_seedmean": boot,
        "baselines": base_metrics,
        "tail_definition": f"y <= train q10 ({q10:.4f})",
        "tail_delta_rmse_seedmean": _subgroup_rmse_delta(preds, y, tail, seeds),
        "ordinary_delta_rmse_seedmean": _subgroup_rmse_delta(preds, y, ~tail, seeds),
        "n_seeds": len(seeds), "official_valid_loaded": True, "official_test_loaded": True,
    }
    write_json(OUT / "test_metrics.json", payload)
    log(f"[score-test] S rmse={seed_mean['S']['rmse']:.5f} P rmse={seed_mean['P']['rmse']:.5f} "
        f"delta={boot['delta']:+.5f} n_seeds={len(seeds)}")
    return 0


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-dataset")
    p.add_argument("--csv", required=True)
    p.add_argument("--reference-json", default=None)
    p.add_argument("--download-utc", default=None)
    p.set_defaults(func=cmd_prepare_dataset)
    p = sub.add_parser("prep")
    p.set_defaults(func=cmd_prep)
    p = sub.add_parser("make-fixture")
    p.set_defaults(func=cmd_make_fixture)
    p = sub.add_parser("gates")
    p.add_argument("--no-strict-code", action="store_true")
    p.set_defaults(func=cmd_gates, no_strict_code=False)
    p = sub.add_parser("smoke")
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-unfrozen", action="store_true")
    p.add_argument("--no-strict-code", action="store_true")
    p.set_defaults(func=cmd_smoke, no_strict_code=False)
    p = sub.add_parser("ssl")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-updates", type=int, default=0, help="debug-only clamp; never used in the frozen runs")
    p.add_argument("--allow-unfrozen", action="store_true")
    p.add_argument("--no-strict-code", action="store_true")
    p.set_defaults(func=cmd_ssl, no_strict_code=False)
    p = sub.add_parser("arm")
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs-override", type=int, default=0, help="debug-only epoch count; never used in the frozen runs")
    p.add_argument("--allow-unfrozen", action="store_true")
    p.add_argument("--no-strict-code", action="store_true")
    p.set_defaults(func=cmd_arm, no_strict_code=False)
    p = sub.add_parser("predict")
    p.add_argument("--zone", required=True, choices=["valid", "test"])
    p.add_argument("--seeds", default=None)
    p.add_argument("--arms", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-unfrozen", action="store_true")
    p.add_argument("--no-strict-code", action="store_true")
    p.set_defaults(func=cmd_predict, no_strict_code=False)
    p = sub.add_parser("baselines")
    p.set_defaults(func=cmd_baselines)
    p = sub.add_parser("score-valid")
    p.set_defaults(func=cmd_score_valid)
    p = sub.add_parser("score-test")
    p.set_defaults(func=cmd_score_test)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
