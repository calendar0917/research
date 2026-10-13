"""cscl-bjoc-performance-mechanism-long-v1 driver.

Protocol: ``configs/bjoc_performance_mechanism_long_v1.yaml`` (frozen before
any GPU trajectory).  Track: ``tracks/structure-contribution``.

Stages
------
``splits``               deterministic group-aware split(s), label-blind
``objects``              object caches + Fit-only standardiser, label-blind
``freeze``               protocol freeze (code/config/input hashes)
``verify-inputs``        re-hash every reused frozen input
``stage-a-audit``        full-10k B3 init/recipe parity audit (no labels)
``stage-a-train``        train B3 scratch + R3 on the official 10k, predict valid
``stage-a-predict``      re-forward the frozen B3 soups on official valid
``stage-a-score``        2x2 full-10k table on the exposed official-valid split
``stage-b-ssl``          masked SSL on a new split's Fit rows (label-free)
``stage-b-arm``          one supervised arm on a new split (raw-y L1)
``stage-b-predict``      label-blind Monitor/Locked forwards + hash freeze
``stage-b-score-monitor``the one Monitor label read + gates (per split tag)
``stage-c-gate``         Stage-C CPU information / identity / invariance gate
``stage-c-train``        Stage-C arm (second-moment readout) on a split
``stage-c-predict``      label-blind forwards for the C1 soup
``stage-c-score-monitor``Stage-C Monitor gate
``locked-freeze``        freeze the final candidate prediction set
``locked-score``         the one Locked label read + final table

Discipline: official test is never referenced by any code path in this driver.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import cscl_features as cf  # noqa: E402
import run_bjoc_full10k_pretraining_transfer_v1 as f10k  # noqa: E402
import run_bjoc_masked_pretraining_v1 as mpr  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = bp.PROTOCOL_ID
CONFIG_PATH = _CODE_DIR.parent / "configs" / "bjoc_performance_mechanism_long_v1.yaml"
OUT = REPO_ROOT / "tracks/structure-contribution/results/bjoc_performance_mechanism_long_v1"
F10K = REPO_ROOT / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1"
REC = REPO_ROOT / "tracks/structure-contribution/results/bjoc_recipe_tuning_v1"
MPR = REPO_ROOT / "tracks/structure-contribution/results/bjoc_masked_pretraining_v1"
ENCODED_TRAIN = REPO_ROOT / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt"
CANONICAL_SMILES = REPO_ROOT / "tracks/ksvd/results/zinc_cssd_basis_reuse_v1/train_canonical_smiles.npz"

N_TRAIN = 10000
N_VALID = 1000
SEEDS = (0, 1)
EXTREME_VALID_ROW = 172

# ---------------------------------------------------------------------------
# arm registry
# ---------------------------------------------------------------------------

STAGE_ARMS: dict[str, dict[str, Any]] = {
    "B0": {"init": "scratch", "recipe": "R0", "ssl_mode": None, "model_kind": "base"},
    "B3": {"init": "scratch", "recipe": "R3", "ssl_mode": None, "model_kind": "base"},
    "P0": {"init": "ssl", "recipe": "R0", "ssl_mode": "atom_bond", "model_kind": "base"},
    "P3": {"init": "ssl", "recipe": "R3", "ssl_mode": "atom_bond", "model_kind": "base"},
    "A1": {"init": "ssl", "recipe": "R3", "ssl_mode": "atom_only", "model_kind": "base"},
    "C1": {"init": "ssl", "recipe": "R3", "ssl_mode": "atom_bond", "model_kind": "c1"},
}

#: the Stage-A full-10k B3 arm is named B3F to avoid colliding with split arms.
STAGE_A_RECIPE = "R3"

#: Stage-B/C/D split tags -> salt (frozen in the config)
SPLIT_SALTS = {
    "B": "cscl-bjoc-perfmech-long-v1-B",
    "C": "cscl-bjoc-perfmech-long-v1-C",
}


def split_paths(tag: str) -> dict[str, Path]:
    return {
        "split": OUT / f"split_{tag}.json",
        "objects_fit": OUT / f"objects_fit_{tag}.pt",
        "objects_monitor": OUT / f"objects_monitor_{tag}.pt",
        "objects_locked": OUT / f"objects_locked_{tag}.pt",
        "prep": OUT / f"prep_fit_{tag}.json",
        "objects_manifest": OUT / f"objects_manifest_{tag}.json",
    }


def ssl_paths(tag: str, seed: int, mode: str) -> tuple[Path, Path]:
    stem = f"pretrain_{tag}" + ("_atomonly" if mode == "atom_only" else "") + f"_s{seed}"
    return OUT / f"{stem}.pt", OUT / f"{stem}.json"


def soup_path(tag: str, arm: str, seed: int) -> Path:
    return OUT / f"soup_{arm}_{tag}_s{seed}.pt"


def train_json_path(tag: str, arm: str, seed: int) -> Path:
    return OUT / f"train_{arm}_{tag}_s{seed}.json"


def pred_path(zone: str, tag: str, arm: str, seed: int) -> Path:
    return OUT / "predictions" / f"{zone}_{arm}_{tag}_s{seed}.npz"


def stage_a_pred_path(seed: int) -> Path:
    return OUT / "predictions" / f"valid_B3F_s{seed}.npz"


PRED_MANIFEST = {tag: OUT / f"prediction_manifest_{tag}.json" for tag in SPLIT_SALTS}
MONITOR_LOCK = {tag: OUT / f"monitor_read_{tag}.lock" for tag in SPLIT_SALTS}
MONITOR_Y = {tag: OUT / f"monitor_y_{tag}.npy" for tag in SPLIT_SALTS}
LOCKED_LOCK = {tag: OUT / f"locked_read_{tag}.lock" for tag in SPLIT_SALTS}
LOCKED_Y = {tag: OUT / f"locked_y_{tag}.npy" for tag in SPLIT_SALTS}
FREEZE_JSON = OUT / "protocol_freeze.json"


def log(msg: str) -> None:
    print(msg, flush=True)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(f10k._j(payload), indent=2), encoding="utf-8")


def git_commit() -> str:
    return f10k.git_commit()


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# code / input freeze
# ---------------------------------------------------------------------------


def _code_paths() -> dict[str, Path]:
    return {
        "driver": _CODE_DIR / "run_bjoc_performance_mechanism_long_v1.py",
        "module": _CODE_DIR / "bjoc_perfmech_long_v1.py",
        "config": CONFIG_PATH,
        "base_module": _CODE_DIR / "bjoc_masked_pretraining_v1.py",
        "objects": _CODE_DIR / "structural_relation_objects_v1.py",
        "composer": _CODE_DIR / "structural_relation_composer_v1.py",
        "masked_driver": _CODE_DIR / "run_bjoc_masked_pretraining_v1.py",
        "recipe_driver": _CODE_DIR / "run_bjoc_recipe_tuning_v1.py",
    }


def code_hashes() -> dict[str, str]:
    return {k: bp.file_sha256(p) for k, p in _code_paths().items()}


def _frozen_input_files() -> dict[str, Path]:
    return {
        "objects_train10k": F10K / "objects_train10k.pt",
        "objects_valid": F10K / "objects_valid.pt",
        "prep_train10k": F10K / "prep_train10k.json",
        "pretrain_encoder_s0": F10K / "pretrain_encoder_s0.pt",
        "pretrain_encoder_s1": F10K / "pretrain_encoder_s1.pt",
        "soup_P_seed0": F10K / "soup_P_seed0.pt",
        "soup_P_seed1": F10K / "soup_P_seed1.pt",
        "soup_R3_seed0": REC / "soup_R3_seed0.pt",
        "soup_R3_seed1": REC / "soup_R3_seed1.pt",
        "split_a": MPR / "split_a.json",
        "canonical_smiles": CANONICAL_SMILES,
    }


def audit_inputs() -> dict[str, Any]:
    cfg = load_config()
    frozen = cfg["frozen"]
    expected = {
        "objects_train10k": frozen["objects_train10k_sha256"],
        "objects_valid": frozen["objects_valid_sha256"],
        "prep_train10k": frozen["prep_train10k_sha256"],
        "pretrain_encoder_s0": frozen["pretrain_encoder_s0_sha256"],
        "pretrain_encoder_s1": frozen["pretrain_encoder_s1_sha256"],
        "soup_P_seed0": frozen["soup_P_seed0_sha256"],
        "soup_P_seed1": frozen["soup_P_seed1_sha256"],
        "soup_R3_seed0": frozen["soup_R3_seed0_sha256"],
        "soup_R3_seed1": frozen["soup_R3_seed1_sha256"],
        "split_a": frozen["split_a_sha256"],
        "canonical_smiles": frozen["canonical_smiles_sha256"],
    }
    checks: dict[str, Any] = {}
    ok = True
    for name, path in _frozen_input_files().items():
        exists = Path(path).exists()
        actual = bp.file_sha256(path) if exists else None
        hit = bool(exists and actual == expected[name])
        ok = ok and hit
        checks[name] = {"path": str(Path(path).relative_to(REPO_ROOT)), "exists": exists, "sha256": actual, "matches": hit}
    return {"ok": ok, "checks": checks}


def cmd_verify_inputs(args: argparse.Namespace) -> int:
    audit = audit_inputs()
    write_json(OUT / "input_audit.json", {"protocol": PROTOCOL_ID, "git_commit": git_commit(), **audit})
    log(f"[verify-inputs] ok={audit['ok']}")
    for name, rec in audit["checks"].items():
        log(f"  {name:24s} {'OK' if rec['matches'] else 'MISMATCH'}")
    return 0 if audit["ok"] else 1


def cmd_freeze(args: argparse.Namespace) -> int:
    audit = audit_inputs()
    if not audit["ok"]:
        raise SystemExit("frozen input audit failed; refusing to freeze")
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "config_sha256": bp.file_sha256(CONFIG_PATH),
        "code_hashes": code_hashes(),
        "arm_registry_sha256": hashlib.sha256(json.dumps(STAGE_ARMS, sort_keys=True).encode()).hexdigest(),
        "split_salts": SPLIT_SALTS,
        "input_audit": {k: v["sha256"] for k, v in audit["checks"].items()},
        "input_audit_ok": True,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_test_loaded": False,
    }
    write_json(FREEZE_JSON, payload)
    log(f"[freeze] written {FREEZE_JSON.relative_to(REPO_ROOT)} commit={payload['git_commit'][:12]}")
    return 0


def verify_freeze(*, strict_code: bool = True) -> dict[str, Any]:
    if not FREEZE_JSON.exists():
        raise SystemExit("protocol is not frozen; run `freeze` first")
    frozen = read_json(FREEZE_JSON)
    if frozen["protocol"] != PROTOCOL_ID:
        raise SystemExit("freeze protocol id mismatch")
    if strict_code:
        current = code_hashes()
        drift = {k: (frozen["code_hashes"].get(k), v) for k, v in current.items() if frozen["code_hashes"].get(k) != v}
        if drift:
            raise SystemExit(f"code drift versus freeze: {drift}")
    audit = audit_inputs()
    if not audit["ok"]:
        bad = [k for k, v in audit["checks"].items() if not v["matches"]]
        raise SystemExit(f"frozen input drift: {bad}")
    return frozen


# ---------------------------------------------------------------------------
# splits + objects
# ---------------------------------------------------------------------------


def _overlap(a: Sequence[int], b: Sequence[int]) -> int:
    return int(len(set(int(x) for x in a) & set(int(x) for x in b)))


def build_split_payload(tag: str) -> dict[str, Any]:
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    if len(smiles) != N_TRAIN:
        raise SystemExit(f"canonical smiles table has {len(smiles)} rows")
    split = bp.build_split(smiles, SPLIT_SALTS[tag])
    fit, monitor, locked = split["fit"], split["monitor"], split["locked"]
    split_a = read_json(MPR / "split_a.json")
    overlap = {
        "split_a_fit": _overlap(fit, split_a["fit_rows"]),
        "split_a_monitor": _overlap(monitor, split_a["monitor_rows"]),
        "split_a_locked": _overlap(locked, split_a["locked_rows"]),
    }
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": tag,
        "split_salt": SPLIT_SALTS[tag],
        "n_groups": split["n_groups"],
        "n_fit": int(len(fit)),
        "n_monitor": int(len(monitor)),
        "n_locked": int(len(locked)),
        "fit_rows": [int(x) for x in fit],
        "monitor_rows": [int(x) for x in monitor],
        "locked_rows": [int(x) for x in locked],
        "fit_sha256_rows": bp.rows_sha256(fit),
        "monitor_sha256_rows": bp.rows_sha256(monitor),
        "locked_sha256_rows": bp.rows_sha256(locked),
        "overlap_rows_with_split_a": overlap,
        "duplicate_canonical_smiles_groups": len(split["duplicate_groups"]),
        "duplicate_groups_cross_sides": split["duplicate_groups_cross_sides"],
        "label_information_used": False,
        "monitor_labels_read": False,
        "locked_labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    union = set(map(int, fit)) | set(map(int, monitor)) | set(map(int, locked))
    payload["union_rows"] = len(union)
    return payload


def cmd_splits(args: argparse.Namespace) -> int:
    tag = args.tag
    payload = build_split_payload(tag)
    write_json(split_paths(tag)["split"], payload)
    if tag == "B":
        # split C exists from the start as a frozen plan; its salt is fixed in
        # the config and its rows are only materialised when Stage D is bought.
        pass
    log(
        f"[splits:{tag}] fit={payload['n_fit']} monitor={payload['n_monitor']} locked={payload['n_locked']} "
        f"dup_groups_cross={payload['duplicate_groups_cross_sides']}"
    )
    log(
        f"[splits:{tag}] overlap with A: fit {payload['overlap_rows_with_split_a']['split_a_fit']} "
        f"monitor {payload['overlap_rows_with_split_a']['split_a_monitor']} "
        f"locked {payload['overlap_rows_with_split_a']['split_a_locked']}"
    )
    return 0


def cmd_objects(args: argparse.Namespace) -> int:
    tag = args.tag
    paths = split_paths(tag)
    payload = build_split_payload(tag)
    write_json(paths["split"], payload)
    ds = cf.load_official_train()
    t0 = time.perf_counter()
    fit_objs = bp.build_objects_for_rows(payload["fit_rows"], ds)
    mon_objs = bp.build_objects_for_rows(payload["monitor_rows"], ds)
    lock_objs = bp.build_objects_for_rows(payload["locked_rows"], ds)
    build_seconds = time.perf_counter() - t0
    for objs, rows, key in (
        (fit_objs, payload["fit_rows"], "fit"),
        (mon_objs, payload["monitor_rows"], "monitor"),
        (lock_objs, payload["locked_rows"], "locked"),
    ):
        assert [o.mol_index for o in objs] == [int(i) for i in rows], key
    torch.save(fit_objs, paths["objects_fit"])
    torch.save(mon_objs, paths["objects_monitor"])
    torch.save(lock_objs, paths["objects_locked"])
    prep = bp.fit_standardiser(fit_objs)
    prep["protocol"] = PROTOCOL_ID
    prep["split_tag"] = tag
    write_json(paths["prep"], prep)

    # alignment check on fit + monitor rows only (locked y is not read)
    enc = torch.load(ENCODED_TRAIN, map_location="cpu", weights_only=False)
    check_rows = list(payload["fit_rows"][:200]) + list(payload["monitor_rows"])
    max_diff = 0.0
    for i in check_rows:
        max_diff = max(max_diff, abs(float(ds[i].y.reshape(-1)[0]) - float(enc[i].y.reshape(-1)[0])))
    del enc
    manifest = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": tag,
        "build_seconds": build_seconds,
        "fit_rows_sha256": payload["fit_sha256_rows"],
        "objects_fit_sha256": bp.file_sha256(paths["objects_fit"]),
        "objects_monitor_sha256": bp.file_sha256(paths["objects_monitor"]),
        "objects_locked_sha256": bp.file_sha256(paths["objects_locked"]),
        "prep_fit_sha256": bp.file_sha256(paths["prep"]),
        "alignment_rows_checked": len(check_rows),
        "alignment_max_abs_diff": float(max_diff),
        "locked_labels_read": False,
        "official_test_loaded": False,
    }
    write_json(paths["objects_manifest"], manifest)
    log(f"[objects:{tag}] build {build_seconds:.1f}s alignment max|dy|={max_diff:.2e}")
    return 0


# ---------------------------------------------------------------------------
# init / recipe parity audit (Stage A)
# ---------------------------------------------------------------------------


def _stage_a_audit() -> dict[str, Any]:
    cfg = load_config()
    frozen = cfg["frozen"]
    norm = read_json(F10K / "prep_train10k.json")
    per_seed: dict[str, Any] = {}
    ok = True
    for seed in SEEDS:
        model = bp.build_bjoc_null(seed, norm, "cpu")
        init_full = src.state_dict_hash(model)
        init_group = mp.group_state_hash(model)
        init_encoder = bp.encoder_group_hash(model)
        p_rec = read_json(F10K / f"train_P_seed{seed}.json")
        r3_rec = read_json(REC / f"train_R3_seed{seed}.json")
        order = bp.batch_order_hash(seed, N_TRAIN, bp.BATCH_SIZE, 360)
        checks = {
            "b3_scratch_init_full_matches_P_init": init_full == p_rec["init_full_sha256"] == frozen["p_init_full_sha256"][str(seed)],
            "b3_scratch_init_group_matches_P_init": init_group == p_rec["init_group_state_sha256"] == frozen["p_init_group_state_sha256"][str(seed)],
            "b3_scratch_encoder_matches_P_init_encoder": init_encoder == p_rec["init_encoder_sha256"] == frozen["p_init_encoder_sha256"][str(seed)],
            "b3_never_loads_pretrained_encoder": init_encoder != frozen["p_encoder_used_sha256"][str(seed)],
            "batch_order_matches_recorded_R3": order == r3_rec["batch_order"]["full_hash"] == frozen["r3_recipe"]["full_batch_order_hash"][str(seed)],
            "recipe_epochs": int(r3_rec["epochs"]) == 360,
            "recipe_schedule_cosine": r3_rec["schedule"]["type"] == "cosine",
            "recipe_lr": float(r3_rec["schedule"]["lr0"]) == 1.0e-3,
            "recipe_lr_min": float(r3_rec["schedule"]["lr_min"]) == 1.0e-5,
            "recipe_wd": float(r3_rec["hyper"]["weight_decay"]) == 1.0e-5,
            "recipe_clip": float(r3_rec["hyper"]["grad_clip"]) == 5.0,
            "recipe_soup_last5": r3_rec["soup_rule"].endswith("last 5 epoch states"),
            "default_recipe_matches_frozen_R3": math.isclose(bp.RECIPES["R3"]["epochs"], 360)
            and bp.RECIPES["R3"]["schedule"] == "cosine"
            and math.isclose(bp.RECIPES["R3"]["lr"], 1.0e-3)
            and math.isclose(bp.RECIPES["R3"]["lr_min"], 1.0e-5)
            and math.isclose(bp.RECIPES["R3"]["weight_decay"], 1.0e-5),
            "schedule_function_matches_recorded_R3": abs(
                bp.cosine_lr(0, 28440, 1.0e-3, 1.0e-5) - float(r3_rec["schedule"]["lr_first_step"])
            ) < 1e-12
            and abs(bp.cosine_lr(28440, 28440, 1.0e-3, 1.0e-5) - float(r3_rec["schedule"]["lr_last_step"])) < 1e-9,
        }
        per_seed[str(seed)] = {
            "init_full_sha256": init_full,
            "init_group_state_sha256": init_group,
            "init_encoder_sha256": init_encoder,
            "batch_order_hash_360": order,
            "r3_recorded_soup_sha256": r3_rec["soup_sha256"],
            "checks": checks,
        }
        ok = ok and all(checks.values())
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "per_seed": per_seed,
        "ok": bool(ok),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "stage_a_audit.json", payload)
    return payload


def cmd_stage_a_audit(args: argparse.Namespace) -> int:
    audit = _stage_a_audit()
    log(f"[stage-a-audit] ok={audit['ok']}")
    for seed, rec in audit["per_seed"].items():
        bad = [k for k, v in rec["checks"].items() if not v]
        log(f"  seed {seed}: {'OK' if not bad else 'FAIL ' + str(bad)}")
    return 0 if audit["ok"] else 1


# ---------------------------------------------------------------------------
# training commands
# ---------------------------------------------------------------------------


def _save_soup(payload: Mapping[str, Any], path: Path) -> None:
    state = payload["soup_state"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu().clone() for k, v in state.items()}, path)


def _strip_heavy(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out.pop("soup_state", None)
    out.pop("encoder_state", None)
    return out


def cmd_stage_a_train(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=True)
    seed = int(args.seed)
    smoke_epochs = args.smoke_epochs
    audit = _stage_a_audit()
    if not audit["ok"]:
        raise SystemExit("Stage-A audit failed; refusing to train")
    smoke_dir = OUT / "smoke"
    paths = {s: (smoke_dir / f"train_B3F_s{s}_smoke.json" if smoke_epochs else OUT / f"train_B3F_s{s}.json") for s in SEEDS}
    soups = {s: (smoke_dir / f"soup_B3F_s{s}_smoke.pt" if smoke_epochs else OUT / f"soup_B3F_s{s}.pt") for s in SEEDS}
    if paths[seed].exists() or soups[seed].exists():
        raise SystemExit(f"B3F seed {seed} already exists; retraining is not allowed")
    device = torch.device(args.device)
    norm = read_json(F10K / "prep_train10k.json")
    train_objs = f10k.load_objects(F10K / "objects_train10k.pt")
    y_fit = bp.load_y_rows(ENCODED_TRAIN, list(range(N_TRAIN)))
    payload = bp.train_supervised(
        arm="B3F",
        seed=seed,
        recipe_key=STAGE_A_RECIPE,
        init="scratch",
        train_objs=train_objs,
        rows=list(range(N_TRAIN)),
        y_fit=y_fit,
        norm=norm,
        device=device,
        model_kind="base",
        smoke_epochs=smoke_epochs,
    )
    _save_soup(payload, soups[seed])
    payload["soup_path"] = str(soups[seed].relative_to(REPO_ROOT))
    payload["soup_sha256"] = bp.file_sha256(soups[seed])
    write_json(paths[seed], _strip_heavy(payload))
    log(f"[B3F s{seed}] soup fit {payload['soup_fit_mae']:.5f} wall {payload['wall_seconds']:.0f}s")
    if smoke_epochs:
        return 0
    # label-blind official-valid forward, same job
    valid_objs = f10k.load_objects(F10K / "objects_valid.pt")
    model = bp.build_bjoc_null(seed, norm, device)
    model.load_state_dict(torch.load(soups[seed], map_location=device, weights_only=False))
    pred = bp.predict_model(model, valid_objs, device)
    if not np.isfinite(pred).all():
        raise SystemExit("non-finite B3F predictions")
    out = stage_a_pred_path(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, row=np.arange(N_VALID, dtype=np.int64), pred=pred.astype(np.float32))
    log(f"[B3F s{seed}] valid prediction written {out.relative_to(REPO_ROOT)}")
    return 0


def cmd_stage_a_predict(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    device = torch.device(args.device)
    norm = read_json(F10K / "prep_train10k.json")
    valid_objs = f10k.load_objects(F10K / "objects_valid.pt")
    for seed in [int(s) for s in args.seeds]:
        soup = OUT / f"soup_B3F_s{seed}.pt"
        if not soup.exists():
            raise SystemExit(f"missing {soup}")
        model = bp.build_bjoc_null(seed, norm, device)
        model.load_state_dict(torch.load(soup, map_location=device, weights_only=False))
        pred = bp.predict_model(model, valid_objs, device)
        out = stage_a_pred_path(seed)
        np.savez_compressed(out, row=np.arange(N_VALID, dtype=np.int64), pred=pred.astype(np.float32))
        log(f"[stage-a-predict] seed {seed} -> {out.relative_to(REPO_ROOT)}")
    return 0


def cmd_stage_b_ssl(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=True)
    tag = args.tag
    seed = int(args.seed)
    mode = args.mode
    ssl_pt, ssl_json = ssl_paths(tag, seed, mode)
    if args.smoke_updates:
        ssl_pt = OUT / "smoke" / f"{ssl_pt.stem}_smoke.pt"
        ssl_json = OUT / "smoke" / f"{ssl_json.stem}_smoke.json"
    if ssl_pt.exists() or ssl_json.exists():
        raise SystemExit(f"SSL {tag} seed {seed} mode {mode} already exists; retraining is not allowed")
    paths = split_paths(tag)
    for p in (paths["split"], paths["objects_fit"], paths["prep"]):
        if not p.exists():
            raise SystemExit(f"missing {p}; run splits+objects for tag {tag} first")
    split = read_json(paths["split"])
    fit_objs = f10k.load_objects(paths["objects_fit"])
    assert [int(o.mol_index) for o in fit_objs] == [int(i) for i in split["fit_rows"]]
    ds = cf.load_official_train()
    payload = bp.pretrain_masked(
        seed=seed,
        fit_objs=fit_objs,
        fit_rows=split["fit_rows"],
        raw_dataset=ds,
        device=torch.device(args.device),
        mode=mode,
        smoke_updates=args.smoke_updates,
    )
    ssl_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload["encoder_state"], ssl_pt)
    payload["encoder_path"] = str(ssl_pt.relative_to(REPO_ROOT))
    payload["encoder_file_sha256"] = bp.file_sha256(ssl_pt)
    payload["split_tag"] = tag
    write_json(ssl_json, _strip_heavy(payload))
    log(f"[ssl {tag} s{seed} {mode}] final acc atom {payload['final_acc_atom']:.4f} wall {payload['wall_seconds']:.0f}s")
    return 0


def cmd_stage_b_arm(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=True)
    tag = args.tag
    arm = args.arm
    seed = int(args.seed)
    if arm not in STAGE_ARMS:
        raise SystemExit(f"unknown arm {arm!r}")
    spec = STAGE_ARMS[arm]
    if arm == "A1" and tag != "B":
        raise SystemExit("A1 (atom-only) is only defined for the Stage-B split")
    if arm == "C1":
        raise SystemExit("C1 has its own gate + command (stage-c-train); refusing stage-b-arm")
    smoke_epochs = args.smoke_epochs
    paths = split_paths(tag)
    soup = soup_path(tag, arm, seed)
    tj = train_json_path(tag, arm, seed)
    if smoke_epochs:
        soup = OUT / "smoke" / f"{soup.stem}_smoke.pt"
        tj = OUT / "smoke" / f"{tj.stem}_smoke.json"
    if soup.exists() or tj.exists():
        raise SystemExit(f"{arm} {tag} seed {seed} already exists; retraining is not allowed")
    split = read_json(paths["split"])
    fit_objs = f10k.load_objects(paths["objects_fit"])
    norm = read_json(paths["prep"])
    y_fit = bp.load_y_rows(ENCODED_TRAIN, split["fit_rows"])
    ssl_state = None
    ssl_recorded = None
    if spec["init"] == "ssl":
        ssl_pt, ssl_json = ssl_paths(tag, seed, spec["ssl_mode"])
        if not ssl_pt.exists():
            raise SystemExit(f"missing SSL checkpoint {ssl_pt}")
        ssl_state = torch.load(ssl_pt, map_location="cpu", weights_only=False)
        ssl_recorded = read_json(ssl_json)["encoder_final_sha256"]
    payload = bp.train_supervised(
        arm=arm,
        seed=seed,
        recipe_key=spec["recipe"],
        init=spec["init"],
        train_objs=fit_objs,
        rows=split["fit_rows"],
        y_fit=y_fit,
        norm=norm,
        device=torch.device(args.device),
        ssl_state=ssl_state,
        model_kind=spec["model_kind"],
        smoke_epochs=smoke_epochs,
    )
    if ssl_recorded is not None and payload["encoder_used_sha256"] != ssl_recorded:
        raise SystemExit("encoder used does not match the recorded pretraining state")
    _save_soup(payload, soup)
    payload["soup_path"] = str(soup.relative_to(REPO_ROOT))
    payload["soup_sha256"] = bp.file_sha256(soup)
    payload["split_tag"] = tag
    write_json(tj, _strip_heavy(payload))
    log(f"[{arm} {tag} s{seed}] soup fit {payload['soup_fit_mae']:.5f} wall {payload['wall_seconds']:.0f}s")
    return 0


# ---------------------------------------------------------------------------
# label-blind predictions + freeze
# ---------------------------------------------------------------------------


def _load_arm_model(tag: str, arm: str, seed: int, device: torch.device):
    spec = STAGE_ARMS[arm]
    norm = read_json(split_paths(tag)["prep"])
    if spec["model_kind"] == "base":
        model = bp.build_bjoc_null(seed, norm, device)
        model.load_state_dict(torch.load(soup_path(tag, arm, seed), map_location=device, weights_only=False))
        return model
    _, model, _ = bp.build_c1_from_init(seed, norm, device, None)
    model.load_state_dict(torch.load(soup_path(tag, arm, seed), map_location=device, weights_only=False))
    return model


def cmd_stage_b_predict(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    tag = args.tag
    if MONITOR_LOCK[tag].exists() or LOCKED_LOCK[tag].exists():
        raise SystemExit("labels already read; refusing to rewrite predictions")
    paths = split_paths(tag)
    mon_objs = f10k.load_objects(paths["objects_monitor"])
    lock_objs = f10k.load_objects(paths["objects_locked"])
    device = torch.device(args.device)
    arms = list(args.arms) if args.arms else ["B0", "B3", "P0", "P3"]
    entries: dict[str, Any] = {}
    for arm in arms:
        for seed in [int(s) for s in args.seeds]:
            if not soup_path(tag, arm, seed).exists():
                raise SystemExit(f"missing soup {soup_path(tag, arm, seed)}")
            model = _load_arm_model(tag, arm, seed, device)
            for zone, objs in (("monitor", mon_objs), ("locked", lock_objs)):
                pred = bp.predict_model(model, objs, device)
                if not np.isfinite(pred).all():
                    raise SystemExit(f"non-finite {zone} prediction {arm} {seed}")
                out = pred_path(zone, tag, arm, seed)
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, row=np.arange(len(objs), dtype=np.int64), pred=pred.astype(np.float32))
                entries[f"{zone}:{arm}:{seed}"] = {
                    "path": str(out.relative_to(REPO_ROOT)),
                    "sha256": bp.file_sha256(out),
                    "soup_sha256": bp.file_sha256(soup_path(tag, arm, seed)),
                }
                log(f"[stage-b-predict {tag}] {zone} {arm} s{seed} -> {out.name}")
    manifest = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": tag,
        "entries": entries,
        "monitor_labels_read": False,
        "locked_labels_read": False,
        "official_test_loaded": False,
    }
    write_json(PRED_MANIFEST[tag], manifest)
    return 0


def _verify_prediction_manifest(tag: str) -> dict[str, Any]:
    manifest = read_json(PRED_MANIFEST[tag])
    for key, rec in manifest["entries"].items():
        p = REPO_ROOT / rec["path"]
        if not p.exists() or bp.file_sha256(p) != rec["sha256"]:
            raise SystemExit(f"prediction drift for {key}")
    return manifest


# ---------------------------------------------------------------------------
# scoring helpers
# ---------------------------------------------------------------------------


def _read_npz_pred(path: Path, n: int) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    pred = np.asarray(z["pred"], dtype=np.float64).reshape(-1)
    row = np.asarray(z["row"], dtype=np.int64).reshape(-1)
    if len(pred) != n or not np.array_equal(row, np.arange(n)):
        raise SystemExit(f"row/pred shape mismatch in {path}")
    return pred


def _err(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(pred - y)


def _contrast(new: Mapping[int, np.ndarray], ref: Mapping[int, np.ndarray], y: np.ndarray, tag: str) -> dict[str, Any]:
    seeds = sorted(new)
    e_new = {s: _err(new[s], y) for s in seeds}
    e_ref = {s: _err(ref[s], y) for s in seeds}
    delta = np.mean([e_new[s] - e_ref[s] for s in seeds], axis=0)
    delta_seed_mean = float(np.mean([float((e_new[s] - e_ref[s]).mean()) for s in seeds]))
    mae_new = float(np.mean([float(e_new[s].mean()) for s in seeds]))
    mae_ref = float(np.mean([float(e_ref[s].mean()) for s in seeds]))
    ordinal = y > bp.TAIL_THRESHOLD
    per_seed = {str(s): float((e_new[s] - e_ref[s]).mean()) for s in seeds}
    worst = np.argsort(-np.maximum(e_new[seeds[0]], e_ref[seeds[0]]))[: bp.TOP_K_REMOVED]
    keep = np.ones(len(delta), dtype=bool)
    keep[worst] = False
    return {
        "tag": tag,
        "seeds": seeds,
        "mae_new_seed_mean": mae_new,
        "mae_ref_seed_mean": mae_ref,
        "delta_seed_mean": float(delta_seed_mean),
        "delta_error_vector_mean": float(delta.mean()),
        "per_seed_delta": per_seed,
        "bootstrap": f10k.paired_bootstrap(delta, n=bp.BOOTSTRAP_N, seed=bp.BOOTSTRAP_SEED),
        "ordinary_delta": float(delta[ordinal].mean()),
        "tail_delta": float(delta[~ordinal].mean()),
        "tail_n": int((~ordinal).sum()),
        "rmse_new": float(np.sqrt(np.mean(np.mean([(new[s] - y) ** 2 for s in seeds], axis=0)))),
        "rmse_ref": float(np.sqrt(np.mean(np.mean([(ref[s] - y) ** 2 for s in seeds], axis=0)))),
        "top10_removed_delta": float(delta[keep].mean()),
        "frac_molecules_better": float((delta < 0).mean()),
    }


def _read_monitor_y_once(tag: str, rows: Sequence[int]) -> np.ndarray:
    if MONITOR_LOCK[tag].exists():
        raise SystemExit("Monitor labels already read for this tag")
    y = bp.load_y_rows(ENCODED_TRAIN, rows)
    ds = cf.load_official_train()
    max_diff = 0.0
    for i, v in zip(rows, y):
        max_diff = max(max_diff, abs(float(ds[int(i)].y.reshape(-1)[0]) - float(v)))
    if max_diff > 1e-9:
        raise SystemExit(f"monitor y alignment failed: {max_diff}")
    np.save(MONITOR_Y[tag], y)
    write_json(
        MONITOR_LOCK[tag],
        {
            "protocol": PROTOCOL_ID,
            "split_tag": tag,
            "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_rows": len(rows),
            "y_sha256": bp.array_sha256(y),
            "alignment_max_abs_diff": float(max_diff),
            "prediction_manifest_sha256": bp.file_sha256(PRED_MANIFEST[tag]),
        },
    )
    return y


def _read_locked_y_once(tag: str, rows: Sequence[int]) -> np.ndarray:
    if LOCKED_LOCK[tag].exists():
        raise SystemExit("Locked labels already read for this tag")
    y = bp.load_y_rows(ENCODED_TRAIN, rows)
    ds = cf.load_official_train()
    max_diff = 0.0
    for i, v in zip(rows, y):
        max_diff = max(max_diff, abs(float(ds[int(i)].y.reshape(-1)[0]) - float(v)))
    if max_diff > 1e-9:
        raise SystemExit(f"locked y alignment failed: {max_diff}")
    np.save(LOCKED_Y[tag], y)
    write_json(
        LOCKED_LOCK[tag],
        {
            "protocol": PROTOCOL_ID,
            "split_tag": tag,
            "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_rows": len(rows),
            "y_sha256": bp.array_sha256(y),
            "alignment_max_abs_diff": float(max_diff),
        },
    )
    return y


def cmd_stage_b_score_monitor(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    tag = args.tag
    paths = split_paths(tag)
    split = read_json(paths["split"])
    manifest = _verify_prediction_manifest(tag)
    required = ["B0", "B3", "P0", "P3"]
    for arm in required:
        for seed in SEEDS:
            if f"monitor:{arm}:{seed}" not in manifest["entries"]:
                raise SystemExit(f"missing monitor prediction {arm} {seed}")
    y = _read_monitor_y_once(tag, split["monitor_rows"])
    preds: dict[str, dict[int, np.ndarray]] = {}
    for arm in required:
        preds[arm] = {s: _read_npz_pred(pred_path("monitor", tag, arm, s), len(y)) for s in SEEDS}
    contrasts = {
        "P3_minus_P0": _contrast(preds["P3"], preds["P0"], y, "P3-P0"),
        "B3_minus_B0": _contrast(preds["B3"], preds["B0"], y, "B3-B0"),
        "P3_minus_B3": _contrast(preds["P3"], preds["B3"], y, "P3-B3"),
        "P0_minus_B0": _contrast(preds["P0"], preds["B0"], y, "P0-B0"),
    }
    interaction = float(np.mean([contrasts["P3_minus_B3"]["per_seed_delta"][str(s)] - contrasts["P0_minus_B0"]["per_seed_delta"][str(s)] for s in SEEDS]))
    per_seed_interaction = {
        str(s): contrasts["P3_minus_B3"]["per_seed_delta"][str(s)] - contrasts["P0_minus_B0"]["per_seed_delta"][str(s)] for s in SEEDS
    }
    gates = {
        "recipe_effect": bool(
            contrasts["P3_minus_P0"]["delta_seed_mean"] <= -0.003
            and all(v < 0 for v in contrasts["P3_minus_P0"]["per_seed_delta"].values())
        ),
        "recipe_effect_scratch": bool(
            contrasts["B3_minus_B0"]["delta_seed_mean"] <= -0.003
            and all(v < 0 for v in contrasts["B3_minus_B0"]["per_seed_delta"].values())
        ),
        "ssl_effect_r3": bool(
            contrasts["P3_minus_B3"]["delta_seed_mean"] <= -0.005
            and all(v < 0 for v in contrasts["P3_minus_B3"]["per_seed_delta"].values())
        ),
        "ssl_effect_r0": bool(
            contrasts["P0_minus_B0"]["delta_seed_mean"] <= -0.005
            and all(v < 0 for v in contrasts["P0_minus_B0"]["per_seed_delta"].values())
        ),
    }
    if gates["recipe_effect"]:
        decision = "RECIPE_GAIN_ROBUST_ON_NEW_SPLIT"
    elif gates["ssl_effect_r0"] or gates["ssl_effect_r3"]:
        decision = "RECIPE_GAIN_NOT_ROBUST_ACROSS_SPLITS"
    else:
        decision = "NO_ROBUST_GAIN"
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": tag,
        "contrasts": contrasts,
        "interaction": interaction,
        "per_seed_interaction": per_seed_interaction,
        "gates": gates,
        "decision": decision,
        "ssl_gain_robust_under_r3": gates["ssl_effect_r3"],
        "official_test_loaded": False,
    }
    write_json(OUT / f"stage_b_monitor_{tag}.json", payload)
    log(f"[stage-b-monitor {tag}] decision {decision}")
    for key, c in contrasts.items():
        log(
            f"  {key:12s} delta {c['delta_seed_mean']:+.5f} CI95 [{c['bootstrap']['ci_lo']:+.5f},{c['bootstrap']['ci_hi']:+.5f}] "
            f"per-seed {c['per_seed_delta']} ordinary {c['ordinary_delta']:+.5f} tail {c['tail_delta']:+.5f}"
        )
    return 0


# ---------------------------------------------------------------------------
# Stage C — CPU gate
# ---------------------------------------------------------------------------


def _c1_forward_identity_check(device: torch.device, n_mols: int = 64) -> dict[str, Any]:
    norm = read_json(F10K / "prep_train10k.json")
    objs = f10k.load_objects(F10K / "objects_train10k.pt")[:n_mols]
    base = bp.build_bjoc_null(0, norm, device)
    _, c1, info = bp.build_c1_from_init(0, norm, device, None)
    batch = bp._mv(sro.batch_objects(objs), device)
    with torch.no_grad():
        b = base(batch)
        c = c1(batch)
    diff = float((b - c).abs().max())
    # shared-parameter equality
    bad = []
    base_sd = base.state_dict()
    c1_sd = c1.state_dict()
    for k, v in base_sd.items():
        if k == "reader.lins.0.weight":
            co, ao = int(src.BJOCConfig().compose_out), int(src.BJOCConfig().atom_out)
            blocks = [v[:, 0:co], v[:, co : co + 2 * ao + int(sro.GLOBAL_DIM)]]
            new_v = c1_sd[k]
            if not torch.equal(new_v[:, 0:co], blocks[0]):
                bad.append(k + "[z]")
            if not torch.equal(new_v[:, co + ao :], blocks[1]):
                bad.append(k + "[rest]")
            if not torch.equal(new_v[:, co : co + ao], torch.zeros_like(new_v[:, co : co + ao])):
                bad.append(k + "[c_sq_zero]")
        elif not torch.equal(v.cpu(), c1_sd[k].cpu()):
            bad.append(k)
    return {"identity_max_abs_diff": diff, "shared_parameter_mismatches": bad, "c1_info": info}


def cmd_stage_c_gate(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    device = torch.device(args.device)
    identity = _c1_forward_identity_check(device)
    gate: dict[str, Any] = {"identity": identity}

    # --- empirical novelty: linear reconstructability of c_sq from reader_in
    norm = read_json(F10K / "prep_train10k.json")
    objs = f10k.load_objects(F10K / "objects_train10k.pt")[: args.n_molecules]
    model = bp.build_bjoc_null(0, norm, device)
    model.load_state_dict(torch.load(REC / "soup_R3_seed0.pt", map_location=device, weights_only=False))
    model.eval()
    X, CS = [], []
    with torch.no_grad():
        for i in range(0, len(objs), 256):
            batch = bp._mv(sro.batch_objects(objs[i : i + 256]), device)
            feats = model.forward_features(batch)
            c = feats["c"]
            csq = torch.zeros((int(batch["n_graphs"]), c.shape[1]), device=c.device, dtype=c.dtype)
            csq.index_add_(0, batch["bond_batch"], c * c)
            X.append(feats["reader_in"].detach().cpu().numpy())
            CS.append(csq.detach().cpu().numpy())
    X = np.concatenate(X, 0)
    CS = np.concatenate(CS, 0)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    ntr = int(0.8 * len(X))
    tr, te = perm[:ntr], perm[ntr:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    A = np.concatenate([np.ones((ntr, 1)), (X[tr] - mu) / sd], 1)
    B = np.concatenate([np.ones((len(te), 1)), (X[te] - mu) / sd], 1)
    lam = 1e-2
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ CS[tr])
    p_tr, p_te = A @ W, B @ W
    r2_tr = 1 - ((CS[tr] - p_tr) ** 2).sum(0) / ((CS[tr] - CS[tr].mean(0)) ** 2).sum(0)
    r2_te = 1 - ((CS[te] - p_te) ** 2).sum(0) / ((CS[te] - CS[te].mean(0)) ** 2).sum(0)
    centered_resid = float(np.sqrt(((CS[te] - p_te) ** 2).sum()) / np.sqrt(((CS[te] - CS[te].mean(0)) ** 2).sum()))
    gate["novelty"] = {
        "n_molecules": int(len(X)),
        "linear_probe_train_r2_median": float(np.median(r2_tr)),
        "linear_probe_test_r2_median": float(np.median(r2_te)),
        "linear_probe_test_r2_mean": float(r2_te.mean()),
        "centered_relative_residual": centered_resid,
        "csq_per_dim_std_median": float(np.median(CS.std(0))),
        "csq_zero_std_fraction": float((CS.std(0) < 1e-6).mean()),
        "csq_finite": bool(np.isfinite(CS).all()),
    }
    # explicit existence construction: same sum/count, different square-sum
    v1 = np.asarray([1.0, 1.0])
    v2 = np.asarray([0.0, 2.0])
    gate["algebraic_construction"] = {
        "sum_v1": float(v1.sum()),
        "sum_v2": float(v2.sum()),
        "sqsum_v1": float((v1**2).sum()),
        "sqsum_v2": float((v2**2).sum()),
        "same_first_moment_different_second_moment": bool(v1.sum() == v2.sum() and (v1**2).sum() != (v2**2).sum()),
    }

    # --- gradient activity of the new reader columns + one-step prediction change
    _, c1, _ = bp.build_c1_from_init(0, norm, device, None)
    batch = bp._mv(sro.batch_objects(objs[:128]), device)
    y = torch.as_tensor(bp.load_y_rows(ENCODED_TRAIN, list(range(128))), dtype=torch.float32, device=device)
    c1.train()
    pred0 = c1(batch).detach()
    loss = torch.nn.functional.l1_loss(c1(batch).view(-1), y.view(-1))
    c1.zero_grad()
    loss.backward()
    co, ao = int(src.BJOCConfig().compose_out), int(src.BJOCConfig().atom_out)
    grad_new = c1.reader.lins[0].weight.grad[:, co : co + ao]
    gate["gradient"] = {
        "new_column_grad_abs_max": float(grad_new.abs().max()),
        "new_column_grad_nonzero_frac": float((grad_new != 0).float().mean()),
    }
    opt = torch.optim.Adam(c1.parameters(), lr=1.0e-3)
    opt.step()
    pred1 = c1(batch).detach()
    gate["one_step_prediction_change_max"] = float((pred1 - pred0).abs().max())

    # --- invariance checks (atom permutation + bond endpoint swap + batching)
    inv = _c1_invariance_check(device)
    gate["invariance"] = inv

    # --- static discipline checks
    gate["forbidden_modules"] = src.forbidden_module_scan(c1)
    gate["parameter_audit"] = src.parameter_audit(bp.C1Config(), c1)
    gate["labels_used_in_features"] = False

    cfg = load_config()
    cg = cfg["stage_c"]["cpu_gate"]
    conditions = {
        "identity": identity["identity_max_abs_diff"] <= cg["identity_max_abs_diff_max"] and not identity["shared_parameter_mismatches"],
        "algebraic_construction": gate["algebraic_construction"]["same_first_moment_different_second_moment"],
        "empirical_novelty": gate["novelty"]["linear_probe_test_r2_median"] <= 0.90 and gate["novelty"]["centered_relative_residual"] >= 0.15,
        "nondegenerate": gate["novelty"]["csq_finite"] and gate["novelty"]["csq_zero_std_fraction"] <= 0.10,
        "gradient_activity": gate["gradient"]["new_column_grad_abs_max"] > 0.0 and gate["gradient"]["new_column_grad_nonzero_frac"] > 0.0,
        "one_step_effect": gate["one_step_prediction_change_max"] > 0.0,
        "invariance": all(inv.values()) if inv else False,
        "forbidden_modules": len(gate["forbidden_modules"]) == 0,
        "parameter_budget": gate["parameter_audit"]["actual_parameters"] == bp.C1_EXPECTED_PARAMETERS,
    }
    gate["conditions"] = conditions
    gate["decision"] = "C1_CPU_GATE_PASS" if all(conditions.values()) else "C1_CPU_BLOCKED"
    write_json(OUT / "stage_c_cpu_gate.json", gate)
    log(f"[stage-c-gate] {gate['decision']}")
    for k, v in conditions.items():
        log(f"  {k:22s} {'PASS' if v else 'FAIL'}")
    log(
        f"  novelty: test R2 median {gate['novelty']['linear_probe_test_r2_median']:.4f} "
        f"centered residual {gate['novelty']['centered_relative_residual']:.4f}"
    )
    return 0 if gate["decision"] == "C1_CPU_GATE_PASS" else 1


def _c1_invariance_check(device: torch.device) -> dict[str, bool]:
    """Permutation / endpoint-swap / batching invariance for the C1 forward.

    Raw PyG graphs are relabelled and rebuilt through the frozen object
    builder, so the check matches the builder's canonicalisation contract.
    """
    norm = read_json(F10K / "prep_train10k.json")
    ds = cf.load_official_train()
    rows = [0, 1, 2, 3]
    base_objs = [bp.sro.build_mol_objects(i, ds[i].x.numpy(), ds[i].edge_index.numpy(), ds[i].edge_attr.numpy()) for i in rows]
    _, c1, _ = bp.build_c1_from_init(0, norm, device, None)
    c1.eval()
    rng = np.random.default_rng(0)
    ok = {}

    def forward(objects):
        with torch.no_grad():
            return c1(bp._mv(sro.batch_objects(objects), device)).detach().cpu().numpy()

    y0 = forward(base_objs)
    y_single = {i: forward([base_objs[i]])[0] for i in range(len(base_objs))}
    ok["batch_membership"] = all(float(np.abs(y0[i] - y_single[i]).max()) < 1e-5 for i in range(len(base_objs)))

    # atom-order permutation on the raw graph -> rebuild -> identical prediction
    perm_objs = []
    for i, r in enumerate(rows):
        x = ds[r].x.numpy()
        ei = ds[r].edge_index.numpy()
        ea = ds[r].edge_attr.numpy()
        perm = rng.permutation(x.shape[0])
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        xp = x[perm]
        eip = inv[ei]
        perm_objs.append(bp.sro.build_mol_objects(i, xp, eip, ea))
    yp = forward(perm_objs)
    ok["atom_permutation"] = bool(np.abs(yp - y0).max() < 1e-5)

    # bond endpoint swap on the raw doubled edge list -> identical prediction
    swap_objs = []
    for i, r in enumerate(rows):
        x = ds[r].x.numpy()
        ei = ds[r].edge_index.numpy()[::-1]
        ea = ds[r].edge_attr.numpy()
        swap_objs.append(bp.sro.build_mol_objects(i, x, ei, ea))
    ys = forward(swap_objs)
    ok["bond_endpoint_swap"] = bool(np.abs(ys - y0).max() < 1e-5)
    return ok


# ---------------------------------------------------------------------------
# Stage C — training / prediction / monitor gate
# ---------------------------------------------------------------------------


def cmd_stage_c_train(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=True)
    gate_json = OUT / "stage_c_cpu_gate.json"
    if not gate_json.exists() or read_json(gate_json)["decision"] != "C1_CPU_GATE_PASS":
        raise SystemExit("C1 CPU gate has not passed; refusing to train")
    tag = args.tag
    seed = int(args.seed)
    soup = soup_path(tag, "C1", seed)
    tj = train_json_path(tag, "C1", seed)
    if soup.exists() or tj.exists():
        raise SystemExit("C1 checkpoint already exists; retraining is not allowed")
    paths = split_paths(tag)
    split = read_json(paths["split"])
    fit_objs = f10k.load_objects(paths["objects_fit"])
    norm = read_json(paths["prep"])
    y_fit = bp.load_y_rows(ENCODED_TRAIN, split["fit_rows"])
    ssl_pt, ssl_json = ssl_paths(tag, seed, "atom_bond")
    if not ssl_pt.exists():
        raise SystemExit(f"missing SSL checkpoint {ssl_pt}")
    ssl_state = torch.load(ssl_pt, map_location="cpu", weights_only=False)
    payload = bp.train_supervised(
        arm="C1",
        seed=seed,
        recipe_key="R3",
        init="ssl",
        train_objs=fit_objs,
        rows=split["fit_rows"],
        y_fit=y_fit,
        norm=norm,
        device=torch.device(args.device),
        ssl_state=ssl_state,
        model_kind="c1",
    )
    if payload["encoder_used_sha256"] != read_json(ssl_json)["encoder_final_sha256"]:
        raise SystemExit("C1 encoder does not match the recorded SSL state")
    _save_soup(payload, soup)
    payload["soup_path"] = str(soup.relative_to(REPO_ROOT))
    payload["soup_sha256"] = bp.file_sha256(soup)
    payload["split_tag"] = tag
    write_json(tj, _strip_heavy(payload))
    log(f"[C1 {tag} s{seed}] soup fit {payload['soup_fit_mae']:.5f} wall {payload['wall_seconds']:.0f}s")
    return 0


def cmd_stage_c_predict(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    tag = args.tag
    if MONITOR_LOCK[tag].exists() or LOCKED_LOCK[tag].exists():
        raise SystemExit("labels already read; refusing to rewrite predictions")
    paths = split_paths(tag)
    norm = read_json(paths["prep"])
    mon_objs = f10k.load_objects(paths["objects_monitor"])
    lock_objs = f10k.load_objects(paths["objects_locked"])
    device = torch.device(args.device)
    entries = read_json(PRED_MANIFEST[tag])["entries"] if PRED_MANIFEST[tag].exists() else {}
    for seed in [int(s) for s in args.seeds]:
        soup = soup_path(tag, "C1", seed)
        if not soup.exists():
            raise SystemExit(f"missing C1 soup {soup}")
        _, model, _ = bp.build_c1_from_init(seed, norm, device, None)
        model.load_state_dict(torch.load(soup, map_location=device, weights_only=False))
        for zone, objs in (("monitor", mon_objs), ("locked", lock_objs)):
            pred = bp.predict_model(model, objs, device)
            out = pred_path(zone, tag, "C1", seed)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, row=np.arange(len(objs), dtype=np.int64), pred=pred.astype(np.float32))
            entries[f"{zone}:C1:{seed}"] = {"path": str(out.relative_to(REPO_ROOT)), "sha256": bp.file_sha256(out), "soup_sha256": bp.file_sha256(soup)}
            log(f"[stage-c-predict {tag}] {zone} C1 s{seed} -> {out.name}")
    manifest = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": tag,
        "entries": entries,
        "monitor_labels_read": bool(MONITOR_LOCK[tag].exists()),
        "locked_labels_read": bool(LOCKED_LOCK[tag].exists()),
        "official_test_loaded": False,
    }
    write_json(PRED_MANIFEST[tag], manifest)
    return 0


def cmd_stage_c_score_monitor(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    tag = args.tag
    if not MONITOR_LOCK[tag].exists():
        raise SystemExit("Monitor labels are not read yet; run stage-b-score-monitor first")
    split = read_json(split_paths(tag)["split"])
    y = np.load(MONITOR_Y[tag])
    manifest = _verify_prediction_manifest(tag)
    seeds = [s for s in SEEDS if f"monitor:C1:{s}" in manifest["entries"]]
    if not seeds:
        raise SystemExit("no C1 monitor predictions")
    p3 = {s: _read_npz_pred(pred_path("monitor", tag, "P3", s), len(y)) for s in seeds}
    c1 = {s: _read_npz_pred(pred_path("monitor", tag, "C1", s), len(y)) for s in seeds}
    contrast = _contrast(c1, p3, y, "C1-P3")
    cg = load_config()["stage_c"]["gpu_gate"]
    e_c1 = {s: _err(c1[s], y) for s in seeds}
    e_p3 = {s: _err(p3[s], y) for s in seeds}
    delta = np.mean([e_c1[s] - e_p3[s] for s in seeds], axis=0)
    worst = np.argsort(-np.maximum(e_c1[seeds[0]], e_p3[seeds[0]]))[: bp.TOP_K_REMOVED]
    keep = np.ones(len(delta), dtype=bool)
    keep[worst] = False
    conditions = {
        "seed_mean_delta_le_threshold": contrast["delta_seed_mean"] <= -0.003,
        "ordinary_not_worse": contrast["ordinary_delta"] <= 0.0,
        "top10_removed_negative": float(delta[keep].mean()) < 0.0,
        "rmse_within_tolerance": contrast["rmse_new"] - contrast["rmse_ref"] <= 0.005,
    }
    if all(conditions.values()):
        decision = "C1_MONITOR_GAIN_SUPPORTED"
    elif contrast["delta_seed_mean"] < 0:
        decision = "C1_GAIN_WEAK_OR_MIXED"
    else:
        decision = "C1_NO_PREDICTIVE_GAIN"
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": tag,
        "seeds_scored": seeds,
        "contrast": contrast,
        "conditions": conditions,
        "decision": decision,
        "official_test_loaded": False,
    }
    write_json(OUT / f"stage_c_monitor_{tag}.json", payload)
    log(f"[stage-c-monitor {tag}] decision {decision} delta {contrast['delta_seed_mean']:+.5f}")
    return 0


# ---------------------------------------------------------------------------
# Stage A scoring (exposed official valid)
# ---------------------------------------------------------------------------


def _load_valid_y() -> tuple[np.ndarray, str]:
    y_path = F10K / "valid_y_read_once.npy"
    y = np.asarray(np.load(y_path), dtype=np.float64).reshape(-1)
    lock = read_json(F10K / "valid_read.lock")
    return y, bp.file_sha256(y_path)


def cmd_stage_a_score(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    y, y_sha = _load_valid_y()
    assert len(y) == N_VALID, len(y)
    preds = {
        "B0": {s: _read_npz_pred(F10K / "predictions" / f"valid_B{s}.npz", N_VALID) for s in SEEDS},
        "P0": {s: _read_npz_pred(F10K / "predictions" / f"valid_P{s}.npz", N_VALID) for s in SEEDS},
        "P3": {s: _read_npz_pred(REC / "predictions" / f"valid_R3{s}.npz", N_VALID) for s in SEEDS},
    }
    for s in SEEDS:
        if not stage_a_pred_path(s).exists():
            raise SystemExit(f"missing B3F prediction for seed {s}")
        preds.setdefault("B3", {})[s] = _read_npz_pred(stage_a_pred_path(s), N_VALID)
    contrasts = {
        "G_SSL_R0__P0_minus_B0": _contrast(preds["P0"], preds["B0"], y, "P0-B0"),
        "G_SSL_R3__P3_minus_B3": _contrast(preds["P3"], preds["B3"], y, "P3-B3"),
        "recipe_P3_minus_P0": _contrast(preds["P3"], preds["P0"], y, "P3-P0"),
        "recipe_B3_minus_B0": _contrast(preds["B3"], preds["B0"], y, "B3-B0"),
    }
    interaction = float(np.mean([contrasts["G_SSL_R3__P3_minus_B3"]["per_seed_delta"][str(s)] - contrasts["G_SSL_R0__P0_minus_B0"]["per_seed_delta"][str(s)] for s in SEEDS]))
    g = contrasts["G_SSL_R3__P3_minus_B3"]
    if g["delta_seed_mean"] <= -0.005 and g["bootstrap"]["ci_hi"] < 0:
        decision = "SSL_GAIN_PERSISTS_UNDER_R3"
    elif g["delta_seed_mean"] < 0:
        decision = "SSL_GAIN_SHRINKS_UNDER_R3"
    else:
        decision = "NO_SSL_GAIN_UNDER_R3"
    extreme = {s: {"row_delta": float((_err(preds["P3"][s], y) - _err(preds["B3"][s], y))[EXTREME_VALID_ROW])} for s in SEEDS}
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "valid_y_sha256": y_sha,
        "contrasts": contrasts,
        "interaction": interaction,
        "per_seed_interaction": {str(s): contrasts["G_SSL_R3__P3_minus_B3"]["per_seed_delta"][str(s)] - contrasts["G_SSL_R0__P0_minus_B0"]["per_seed_delta"][str(s)] for s in SEEDS},
        "extreme_row_172": extreme,
        "decision": decision,
        "note": "official valid is an exposed development split; descriptive only",
        "official_test_loaded": False,
    }
    write_json(OUT / "stage_a_2x2.json", payload)
    log(f"[stage-a-score] decision {decision}")
    for key, c in contrasts.items():
        log(
            f"  {key:26s} delta {c['delta_seed_mean']:+.5f} CI95 [{c['bootstrap']['ci_lo']:+.5f},{c['bootstrap']['ci_hi']:+.5f}] "
            f"per-seed {c['per_seed_delta']} ordinary {c['ordinary_delta']:+.5f} tail {c['tail_delta']:+.5f}"
        )
    return 0


# ---------------------------------------------------------------------------
# Locked final
# ---------------------------------------------------------------------------


def cmd_locked_freeze(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    tag = args.tag
    if LOCKED_LOCK[tag].exists():
        raise SystemExit("Locked labels already read; refusing to refreeze")
    manifest = _verify_prediction_manifest(tag)
    arms = ["B0", "B3", "P0", "P3"]
    c1_decision = OUT / f"stage_c_monitor_{tag}.json"
    if c1_decision.exists() and read_json(c1_decision)["decision"] == "C1_MONITOR_GAIN_SUPPORTED":
        arms.append("C1")
    a1_decision = OUT / f"stage_e_monitor_{tag}.json"
    if a1_decision.exists():
        arms.append("A1")
    entries = {}
    for arm in arms:
        for seed in SEEDS:
            key = f"locked:{arm}:{seed}"
            if key not in manifest["entries"]:
                raise SystemExit(f"missing locked prediction {key}; run stage-b-predict first")
            entries[key] = manifest["entries"][key]
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": tag,
        "arms": arms,
        "entries": entries,
        "monitor_decision_sha256": bp.file_sha256(OUT / f"stage_b_monitor_{tag}.json"),
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_test_loaded": False,
    }
    write_json(OUT / f"locked_manifest_{tag}.json", payload)
    log(f"[locked-freeze {tag}] arms {arms}")
    return 0


def cmd_locked_score(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    tag = args.tag
    locked_manifest = read_json(OUT / f"locked_manifest_{tag}.json")
    for key, rec in locked_manifest["entries"].items():
        p = REPO_ROOT / rec["path"]
        if not p.exists() or bp.file_sha256(p) != rec["sha256"]:
            raise SystemExit(f"locked prediction drift for {key}")
    split = read_json(split_paths(tag)["split"])
    y = _read_locked_y_once(tag, split["locked_rows"])
    arms = locked_manifest["arms"]
    preds: dict[str, dict[int, np.ndarray]] = {}
    for arm in arms:
        preds[arm] = {s: _read_npz_pred(pred_path("locked", tag, arm, s), len(y)) for s in SEEDS}
    metrics = {arm: {"per_seed_mae": {str(s): float(_err(preds[arm][s], y).mean()) for s in SEEDS}} for arm in arms}
    for arm in arms:
        m = np.mean([_err(preds[arm][s], y) for s in SEEDS], axis=0)
        metrics[arm]["seed_mean_mae"] = float(m.mean())
        metrics[arm]["median_abs"] = float(np.median(m))
        metrics[arm]["p90_abs"] = float(np.quantile(m, 0.90))
        metrics[arm]["rmse"] = float(np.sqrt(np.mean(np.mean([(preds[arm][s] - y) ** 2 for s in SEEDS], axis=0))))
    contrasts = {}
    for new_arm, ref_arm in (("P3", "P0"), ("B3", "B0"), ("P3", "B3"), ("P0", "B0")):
        if new_arm in preds and ref_arm in preds:
            contrasts[f"{new_arm}_minus_{ref_arm}"] = _contrast(preds[new_arm], preds[ref_arm], y, f"{new_arm}-{ref_arm}")
    if "C1" in preds:
        contrasts["C1_minus_P3"] = _contrast(preds["C1"], preds["P3"], y, "C1-P3")
    if "A1" in preds:
        contrasts["A1_minus_P3"] = _contrast(preds["A1"], preds["P3"], y, "A1-P3")
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": tag,
        "metrics": metrics,
        "contrasts": contrasts,
        "official_test_loaded": False,
    }
    write_json(OUT / f"locked_final_{tag}.json", payload)
    log(f"[locked-score {tag}] done")
    for arm in arms:
        log(f"  {arm}: seed-mean locked MAE {metrics[arm]['seed_mean_mae']:.6f} per-seed {metrics[arm]['per_seed_mae']}")
    for key, c in contrasts.items():
        log(f"  {key:12s} delta {c['delta_seed_mean']:+.5f} CI95 [{c['bootstrap']['ci_lo']:+.5f},{c['bootstrap']['ci_hi']:+.5f}]")
    return 0


# ---------------------------------------------------------------------------
# Stage E (atom-only) monitor gate
# ---------------------------------------------------------------------------


def cmd_stage_e_score_monitor(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    tag = args.tag
    if not MONITOR_LOCK[tag].exists():
        raise SystemExit("Monitor labels are not read yet")
    split = read_json(split_paths(tag)["split"])
    y = np.load(MONITOR_Y[tag])
    manifest = _verify_prediction_manifest(tag)
    seeds = [s for s in SEEDS if f"monitor:A1:{s}" in manifest["entries"]]
    if not seeds:
        raise SystemExit("no A1 monitor predictions")
    a1 = {s: _read_npz_pred(pred_path("monitor", tag, "A1", s), len(y)) for s in seeds}
    p3 = {s: _read_npz_pred(pred_path("monitor", tag, "P3", s), len(y)) for s in seeds}
    contrast = _contrast(a1, p3, y, "A1-P3")
    if contrast["delta_seed_mean"] >= 0.003:
        decision = "BOND_OBJECTIVE_MATTERS"
    elif contrast["delta_seed_mean"] <= -0.003:
        decision = "ATOM_ONLY_BETTER"
    else:
        decision = "BOND_OBJECTIVE_NEUTRAL"
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": tag,
        "seeds_scored": seeds,
        "contrast": contrast,
        "decision": decision,
        "official_test_loaded": False,
    }
    write_json(OUT / f"stage_e_monitor_{tag}.json", payload)
    log(f"[stage-e-monitor {tag}] decision {decision} delta {contrast['delta_seed_mean']:+.5f}")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def cmd_prep(args: argparse.Namespace) -> int:
    """One-shot remote prep: verify inputs, build split(s), freeze, audit."""
    rc = cmd_verify_inputs(args)
    if rc != 0:
        return rc
    for tag in args.tags:
        args.tag = tag
        cmd_splits(args)
        cmd_objects(args)
    cmd_freeze(args)
    rc = cmd_stage_a_audit(args)
    if rc != 0:
        return rc
    log("[prep] complete")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prep")
    prep.add_argument("--tags", nargs="*", default=["B"], choices=sorted(SPLIT_SALTS))

    s = sub.add_parser("splits")
    s.add_argument("--tag", default="B", choices=sorted(SPLIT_SALTS))

    o = sub.add_parser("objects")
    o.add_argument("--tag", default="B", choices=sorted(SPLIT_SALTS))

    sub.add_parser("freeze")
    sub.add_parser("verify-inputs")

    sub.add_parser("stage-a-audit")

    t = sub.add_parser("stage-a-train")
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--device", default="cpu")
    t.add_argument("--smoke-epochs", type=int, default=None)

    t = sub.add_parser("stage-a-predict")
    t.add_argument("--seeds", nargs="*", default=[0, 1])
    t.add_argument("--device", default="cpu")
    sub.add_parser("stage-a-score")

    ssl = sub.add_parser("stage-b-ssl")
    ssl.add_argument("--tag", default="B", choices=sorted(SPLIT_SALTS))
    ssl.add_argument("--seed", type=int, required=True)
    ssl.add_argument("--mode", default="atom_bond", choices=["atom_bond", "atom_only"])
    ssl.add_argument("--device", default="cpu")
    ssl.add_argument("--smoke-updates", type=int, default=None)

    arm = sub.add_parser("stage-b-arm")
    arm.add_argument("--tag", default="B", choices=sorted(SPLIT_SALTS))
    arm.add_argument("--arm", required=True, choices=sorted(STAGE_ARMS))
    arm.add_argument("--seed", type=int, required=True)
    arm.add_argument("--device", default="cpu")
    arm.add_argument("--smoke-epochs", type=int, default=None)

    pr = sub.add_parser("stage-b-predict")
    pr.add_argument("--tag", default="B", choices=sorted(SPLIT_SALTS))
    pr.add_argument("--arms", nargs="*", default=None)
    pr.add_argument("--seeds", nargs="*", default=[0, 1])
    pr.add_argument("--device", default="cpu")

    sc = sub.add_parser("stage-b-score-monitor")
    sc.add_argument("--tag", default="B", choices=sorted(SPLIT_SALTS))

    cg = sub.add_parser("stage-c-gate")
    cg.add_argument("--device", default="cpu")
    cg.add_argument("--n-molecules", type=int, default=4000)

    ct = sub.add_parser("stage-c-train")
    ct.add_argument("--tag", default="B", choices=sorted(SPLIT_SALTS))
    ct.add_argument("--seed", type=int, required=True)
    ct.add_argument("--device", default="cpu")

    cp = sub.add_parser("stage-c-predict")
    cp.add_argument("--tag", default="B", choices=sorted(SPLIT_SALTS))
    cp.add_argument("--seeds", nargs="*", default=[0])
    cp.add_argument("--device", default="cpu")

    cs = sub.add_parser("stage-c-score-monitor")
    cs.add_argument("--tag", default="B", choices=sorted(SPLIT_SALTS))

    es = sub.add_parser("stage-e-score-monitor")
    es.add_argument("--tag", default="B", choices=sorted(SPLIT_SALTS))

    lf = sub.add_parser("locked-freeze")
    lf.add_argument("--tag", default="B", choices=sorted(SPLIT_SALTS))

    ls = sub.add_parser("locked-score")
    ls.add_argument("--tag", default="B", choices=sorted(SPLIT_SALTS))
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {
        "prep": cmd_prep,
        "splits": cmd_splits,
        "objects": cmd_objects,
        "freeze": cmd_freeze,
        "verify-inputs": cmd_verify_inputs,
        "stage-a-audit": cmd_stage_a_audit,
        "stage-a-train": cmd_stage_a_train,
        "stage-a-predict": cmd_stage_a_predict,
        "stage-a-score": cmd_stage_a_score,
        "stage-b-ssl": cmd_stage_b_ssl,
        "stage-b-arm": cmd_stage_b_arm,
        "stage-b-predict": cmd_stage_b_predict,
        "stage-b-score-monitor": cmd_stage_b_score_monitor,
        "stage-c-gate": cmd_stage_c_gate,
        "stage-c-train": cmd_stage_c_train,
        "stage-c-predict": cmd_stage_c_predict,
        "stage-c-score-monitor": cmd_stage_c_score_monitor,
        "stage-e-score-monitor": cmd_stage_e_score_monitor,
        "locked-freeze": cmd_locked_freeze,
        "locked-score": cmd_locked_score,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
