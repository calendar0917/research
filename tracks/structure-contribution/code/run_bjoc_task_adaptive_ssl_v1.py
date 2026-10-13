"""cscl-bjoc-task-adaptive-ssl-v1 driver.

Protocol: ``configs/bjoc_task_adaptive_ssl_v1.yaml`` (frozen before any GPU
trajectory).  Track: ``tracks/structure-contribution``.

Stages
------
``prep``          verify inputs, split-D fingerprint + prior-use audit, guarded
                  objects, Fit-only standardiser, protocol freeze, init parity
``stage-a``        CPU correctness / feasibility gate: frozen-loop replay,
                  SSL wrapper smoke equivalence, mask leak audits, gradient
                  diagnostics, invariances
``ssl``            Fit-only masked SSL for one seed; captures the atom head
``arm``            one supervised arm (P3 baseline or TA-SSL candidate)
``predict``        label-blind Monitor/Locked forwards + hash freeze
``score-monitor``  the single Monitor label read + promotion gate
``locked-freeze``  freeze the final prediction set before the Locked read
``locked-score``   the one Locked read + final table (conditional)
``control-train``  conditional P3-EXTRA compute control (gate must have passed)
``control-score``  descriptive Monitor contrast TA vs P3-EXTRA

Discipline: official test and official valid are never referenced by any code
path; Monitor/Locked structures are built under :class:`ab.LabelGuard` so their
``.y`` is unreachable during preprocessing, SSL, training and prediction; the
Monitor label read and the conditional Locked label read are lock-filed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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

import bjoc_arch_breakthrough_v1 as ab  # noqa: E402
import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import bjoc_task_adaptive_ssl_v1 as tat  # noqa: E402
import cscl_features as cf  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = tat.PROTOCOL_ID
CONFIG_PATH = _CODE_DIR.parent / "configs" / "bjoc_task_adaptive_ssl_v1.yaml"
OUT = REPO_ROOT / "tracks/structure-contribution/results/bjoc_task_adaptive_ssl_v1"
ARCH = REPO_ROOT / "tracks/structure-contribution/results/bjoc_architecture_breakthrough_v1"
MPR = REPO_ROOT / "tracks/structure-contribution/results/bjoc_masked_pretraining_v1"
PERF = REPO_ROOT / "tracks/structure-contribution/results/bjoc_performance_mechanism_long_v1"
ENCODED_TRAIN = REPO_ROOT / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt"
CANONICAL_SMILES = REPO_ROOT / "tracks/ksvd/results/zinc_cssd_basis_reuse_v1/train_canonical_smiles.npz"
ARCH_CONFIG = _CODE_DIR.parent / "configs" / "bjoc_architecture_breakthrough_v1.yaml"

SPLIT_TAG = "D"
N_TRAIN = 10000
SEEDS = (0, 1)
ARMS = ("P3", "TA")


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
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def split_path() -> Path:
    return OUT / "split_D.json"


def objects_path(zone: str) -> Path:
    return OUT / f"objects_{zone}_D.pt"


def prep_path() -> Path:
    return OUT / "prep_fit_D.json"


def objects_manifest_path() -> Path:
    return OUT / "objects_manifest_D.json"


def init_parity_path() -> Path:
    return OUT / "init_parity.json"


def freeze_path() -> Path:
    return OUT / "protocol_freeze.json"


def ssl_paths(seed: int) -> tuple[Path, Path, Path]:
    return (
        OUT / f"pretrain_D_s{seed}.pt",
        OUT / f"pretrain_head_D_s{seed}.pt",
        OUT / f"pretrain_D_s{seed}.json",
    )


def soup_path(arm: str, seed: int) -> Path:
    return OUT / f"soup_{arm}_D_s{seed}.pt"


def train_json_path(arm: str, seed: int) -> Path:
    return OUT / f"train_{arm}_D_s{seed}.json"


def pred_path(zone: str, arm: str, seed: int) -> Path:
    return OUT / "predictions" / f"{zone}_{arm}_D_s{seed}.npz"


PRED_MANIFEST = OUT / "prediction_manifest_D.json"
MONITOR_LOCK = OUT / "monitor_read_D.lock"
MONITOR_Y = OUT / "monitor_y_D.npy"
MONITOR_ROUNDS = OUT / "monitor_rounds_D.json"
LOCKED_LOCK = OUT / "locked_read_D.lock"
LOCKED_Y = OUT / "locked_y_D.npy"
LOCKED_MANIFEST = OUT / "locked_manifest_D.json"
DECISION = OUT / "decision.json"


def _code_paths() -> dict[str, Path]:
    return {
        "driver": _CODE_DIR / "run_bjoc_task_adaptive_ssl_v1.py",
        "module": _CODE_DIR / "bjoc_task_adaptive_ssl_v1.py",
        "config": CONFIG_PATH,
        "base_module": _CODE_DIR / "bjoc_perfmech_long_v1.py",
        "masked_module": _CODE_DIR / "bjoc_masked_pretraining_v1.py",
        "objects": _CODE_DIR / "structural_relation_objects_v1.py",
        "composer": _CODE_DIR / "structural_relation_composer_v1.py",
        "arch_module": _CODE_DIR / "bjoc_arch_breakthrough_v1.py",
    }


def code_hashes() -> dict[str, str]:
    return {k: file_sha256(p) for k, p in _code_paths().items()}


def _frozen_input_files() -> dict[str, Path]:
    # the committed architecture-round config is the pre-registration anchor for
    # split D; the fingerprint payload split_D.json is an optional cross-check
    return {
        "canonical_smiles": CANONICAL_SMILES,
        "encoded_train": ENCODED_TRAIN,
        "arch_config": ARCH_CONFIG,
    }


def audit_inputs() -> dict[str, Any]:
    cfg = load_config()
    expected = {
        "canonical_smiles": cfg["frozen_inputs"]["canonical_smiles_sha256"],
        "encoded_train": cfg["frozen_inputs"]["encoded_train_sha256"],
        "arch_config": None,  # recorded, not pre-frozen (anchor for split D fingerprints)
    }
    checks: dict[str, Any] = {}
    ok = True
    for name, path in _frozen_input_files().items():
        exists = Path(path).exists()
        actual = file_sha256(path) if exists else None
        hit = bool(exists and (expected[name] is None or actual == expected[name]))
        ok = ok and hit
        checks[name] = {
            "path": str(Path(path).relative_to(REPO_ROOT)),
            "exists": exists,
            "sha256": actual,
            "matches": hit,
        }
    return {"ok": ok, "checks": checks}


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def cmd_verify_inputs(args: argparse.Namespace) -> int:
    audit = audit_inputs()
    write_json(OUT / "input_audit.json", {"protocol": PROTOCOL_ID, "git_commit": git_commit(), **audit})
    log(f"[verify-inputs] ok={audit['ok']}")
    for name, rec in audit["checks"].items():
        log(f"  {name:24s} {'OK' if rec['matches'] else 'MISSING'}")
    return 0 if audit["ok"] else 1


def cmd_freeze(args: argparse.Namespace) -> int:
    audit = audit_inputs()
    if not audit["ok"]:
        raise SystemExit("frozen input audit failed; refusing to freeze")
    cfg = load_config()
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "code_hashes": code_hashes(),
        "split_tag": SPLIT_TAG,
        "split_salt": tat.SPLIT_SALT,
        "split_fingerprints": tat.SPLIT_FINGERPRINTS,
        "aux": {
            "every": tat.AUX_EVERY,
            "lambda": tat.AUX_LAMBDA,
            "loss": tat.AUX_LOSS,
            "visit_rule": tat.AUX_VISIT_RULE,
        },
        "parameter_budgets": {
            "inference": bp.EXPECTED_PARAMETERS,
            "atom_head": tat.atom_head_parameters(),
            "training_total": bp.EXPECTED_PARAMETERS + tat.atom_head_parameters(),
        },
        "input_audit": {k: v["sha256"] for k, v in audit["checks"].items()},
        "input_audit_ok": True,
        "label_access": dict(cfg["label_access"]),
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(freeze_path(), payload)
    log(f"[freeze] written commit={payload['git_commit'][:12]}")
    return 0


def verify_freeze(*, strict_code: bool = True) -> dict[str, Any]:
    if not freeze_path().exists():
        raise SystemExit("protocol is not frozen; run `prep` first")
    frozen = read_json(freeze_path())
    if frozen["protocol"] != PROTOCOL_ID:
        raise SystemExit("freeze protocol id mismatch")
    if strict_code:
        current = code_hashes()
        drift = {
            k: (frozen["code_hashes"].get(k), v)
            for k, v in current.items()
            if frozen["code_hashes"].get(k) != v
        }
        if drift:
            raise SystemExit(f"code drift versus freeze: {drift}")
    audit = audit_inputs()
    if not audit["ok"]:
        bad = [k for k, v in audit["checks"].items() if not v["matches"]]
        raise SystemExit(f"frozen input drift: {bad}")
    return frozen


# ---------------------------------------------------------------------------
# split D + prior-use audit
# ---------------------------------------------------------------------------


def _overlap_with(path: Path, fit, monitor, locked) -> dict[str, Any] | None:
    if not path.exists():
        return None
    d = read_json(path)
    other_fit = set(map(int, d.get("fit_rows") or d.get("fit8k") or []))
    other_mon = set(map(int, d.get("monitor_rows", [])))
    other_lock = set(map(int, d.get("locked_rows", [])))
    prev_read = other_mon | other_lock
    return {
        "fit_in_prev_fit": len(set(fit) & other_fit),
        "fit_in_prev_read_zones": len(set(fit) & prev_read),
        "monitor_in_prev_fit": len(set(monitor) & other_fit),
        "monitor_in_prev_read_zones": len(set(monitor) & prev_read),
        "locked_in_prev_fit": len(set(locked) & other_fit),
        "locked_in_prev_read_zones": len(set(locked) & prev_read),
    }


def _scan_prior_d_artifacts() -> list[str]:
    """Any D-tagged artifact other than the pre-registration fingerprint payload."""
    allowed: set[str] = set()
    for path in (ARCH / "split_D.json",):
        try:
            allowed.add(str(path.relative_to(REPO_ROOT)))
        except ValueError:
            pass
    pattern = re.compile(r"(^|_)D(_|\.|$)")
    found: list[str] = []
    results = REPO_ROOT / "tracks/structure-contribution/results"
    for path in sorted(results.glob("*/*")):
        if not path.is_file() or not pattern.search(path.name):
            continue
        rel = str(path.relative_to(REPO_ROOT))
        if rel in allowed or str(path).startswith(str(OUT)):
            continue
        found.append(rel)
    return found


def build_split_payload() -> dict[str, Any]:
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    if len(smiles) != N_TRAIN:
        raise SystemExit(f"canonical smiles table has {len(smiles)} rows")
    split = bp.build_split(smiles, tat.SPLIT_SALT)
    fit = [int(x) for x in split["fit"]]
    monitor = [int(x) for x in split["monitor"]]
    locked = [int(x) for x in split["locked"]]
    got = {
        "n_fit": len(fit),
        "n_monitor": len(monitor),
        "n_locked": len(locked),
        "fit_sha256_rows": bp.rows_sha256(fit),
        "monitor_sha256_rows": bp.rows_sha256(monitor),
        "locked_sha256_rows": bp.rows_sha256(locked),
    }
    for key, expected in tat.SPLIT_FINGERPRINTS.items():
        if got[key] != expected:
            raise SystemExit(f"split D fingerprint mismatch for {key}: {got[key]} != {expected}")
    if int(split["duplicate_groups_cross_sides"]) != 0:
        raise SystemExit("split D duplicate canonical-SMILES groups cross sides")
    # primary anchor: the committed architecture-round pre-registration config
    arch_cfg = yaml.safe_load(ARCH_CONFIG.read_text(encoding="utf-8"))
    reg = arch_cfg["splits"]["fingerprints"]["D"]
    config_match = all(int(reg[k]) == n for k, n in (("n_fit", 8000), ("n_monitor", 1000), ("n_locked", 1000))) and all(
        reg[k] == got[k] for k in ("fit_sha256_rows", "monitor_sha256_rows", "locked_sha256_rows")
    )
    # optional cross-check: the fingerprint payload written by the arch round (if present)
    payload_path = ARCH / "split_D.json"
    arch_match = True
    if payload_path.exists():
        arch_payload = read_json(payload_path)
        arch_match = all(
            int(arch_payload[k]) == n for k, n in (("n_fit", 8000), ("n_monitor", 1000), ("n_locked", 1000))
        ) and all(
            arch_payload[k] == got[k]
            for k in ("fit_sha256_rows", "monitor_sha256_rows", "locked_sha256_rows")
        )
    prior_artifacts = _scan_prior_d_artifacts()
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": SPLIT_TAG,
        "split_salt": tat.SPLIT_SALT,
        "n_groups": int(split["n_groups"]),
        **got,
        "fit_rows": fit,
        "monitor_rows": monitor,
        "locked_rows": locked,
        "union_rows": len(set(fit) | set(monitor) | set(locked)),
        "duplicate_canonical_smiles_groups": int(len(split["duplicate_groups"])),
        "duplicate_groups_cross_sides": int(split["duplicate_groups_cross_sides"]),
        "matches_pre_registered_arch_config": bool(config_match),
        "matches_pre_registered_arch_payload": bool(arch_match) if payload_path.exists() else None,
        "arch_payload_sha256": file_sha256(payload_path) if payload_path.exists() else None,
        "arch_config_sha256": file_sha256(ARCH_CONFIG),
        "prior_use_audit": {
            "not_purchased_in_arch_round": True,
            "prior_d_artifacts_found": prior_artifacts,
            "prior_d_training_detected": bool(prior_artifacts),
            "overlap_with_previous_splits": {
                "A": _overlap_with(MPR / "split_a.json", fit, monitor, locked),
                "B": _overlap_with(PERF / "split_B.json", fit, monitor, locked),
            },
            "disclosure": (
                "internal re-partition of the official ZINC train 10000; these molecules have been used "
                "in historical experiments (including historical full-10k training and previously read "
                "internal monitor/locked blocks); this is NOT an independent external test split and "
                "absolute MAE levels are not comparable to official-split numbers"
            ),
        },
        "label_information_used": False,
        "monitor_labels_read": False,
        "locked_labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if not config_match:
        raise SystemExit("split D does not match the fingerprints pre-registered in the architecture-round config")
    if not arch_match:
        raise SystemExit("split D does not match the architecture-round fingerprint payload")
    if prior_artifacts:
        raise SystemExit(f"split D was already used by prior artifacts: {prior_artifacts}")
    return payload


def cmd_prep(args: argparse.Namespace) -> int:
    rc = cmd_verify_inputs(args)
    if rc != 0:
        return rc
    payload = build_split_payload()
    write_json(split_path(), payload)
    log(
        f"[split:D] fit={payload['n_fit']} monitor={payload['n_monitor']} locked={payload['n_locked']} "
        f"dup_cross={payload['duplicate_groups_cross_sides']} prior_artifacts={len(payload['prior_use_audit']['prior_d_artifacts_found'])}"
    )
    cmd_objects(args)
    cmd_freeze(args)
    rc = cmd_init_parity(args)
    if rc != 0:
        return rc
    log("[prep] complete")
    return 0


# ---------------------------------------------------------------------------
# guarded objects + standardiser
# ---------------------------------------------------------------------------


def _guard_for_split() -> tuple[dict[str, Any], Any, ab.LabelGuard]:
    split = read_json(split_path())
    ds = cf.load_official_train()
    guard = ab.LabelGuard(ds, split["monitor_rows"] + split["locked_rows"])
    return split, ds, guard


def _build_zone_objects(rows: Sequence[int], guard, ds) -> list[sro.MolObjects]:
    objs = []
    for i in rows:
        d = guard[int(i)]
        objs.append(sro.build_mol_objects(int(i), d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy()))
    return objs


def cmd_objects(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    split, ds, guard = _guard_for_split()
    started = time.perf_counter()
    fit_objs = _build_zone_objects(split["fit_rows"], guard, ds)
    mon_objs = _build_zone_objects(split["monitor_rows"], guard, ds)
    lock_objs = _build_zone_objects(split["locked_rows"], guard, ds)
    build_seconds = time.perf_counter() - started
    for objs, rows, name in (
        (fit_objs, split["fit_rows"], "fit"),
        (mon_objs, split["monitor_rows"], "monitor"),
        (lock_objs, split["locked_rows"], "locked"),
    ):
        if [o.mol_index for o in objs] != [int(i) for i in rows]:
            raise SystemExit(f"object row order mismatch for {name}")
        torch.save(objs, objects_path(name))
    prep = bp.fit_standardiser(fit_objs)
    prep["protocol"] = PROTOCOL_ID
    prep["split_tag"] = SPLIT_TAG
    write_json(prep_path(), prep)
    # Fit-only label alignment check (Monitor/Locked .y never accessed here)
    enc = torch.load(ENCODED_TRAIN, map_location="cpu", weights_only=False)
    check_rows = list(split["fit_rows"][:200])
    max_diff = 0.0
    for i in check_rows:
        max_diff = max(max_diff, abs(float(ds[i].y.reshape(-1)[0]) - float(enc[i].y.reshape(-1)[0])))
    del enc
    manifest = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": SPLIT_TAG,
        "build_seconds": build_seconds,
        "fit_rows_sha256": split["fit_sha256_rows"],
        "objects_fit_sha256": file_sha256(objects_path("fit")),
        "objects_monitor_sha256": file_sha256(objects_path("monitor")),
        "objects_locked_sha256": file_sha256(objects_path("locked")),
        "prep_fit_sha256": file_sha256(prep_path()),
        "alignment_rows_checked": len(check_rows),
        "alignment_rows_scope": "Fit rows only",
        "alignment_max_abs_diff": float(max_diff),
        "forbidden_label_access_attempts": len(guard.access_attempts),
        "guarded_rows": len(guard.forbidden_rows),
        "monitor_labels_read": False,
        "locked_labels_read": False,
    }
    if guard.access_attempts:
        raise SystemExit(f"guarded label access attempted during object build: {guard.access_attempts[:3]}")
    if max_diff > 1e-9:
        raise SystemExit(f"Fit label alignment failed: {max_diff}")
    write_json(objects_manifest_path(), manifest)
    log(
        f"[objects:D] {build_seconds:.1f}s alignment max|dy|={max_diff:.2e} "
        f"guarded_attempts={len(guard.access_attempts)}"
    )
    return 0


def cmd_init_parity(args: argparse.Namespace) -> int:
    arch_cfg = yaml.safe_load(ARCH_CONFIG.read_text(encoding="utf-8"))
    frozen = arch_cfg["frozen"]
    norm = read_json(prep_path())
    per_seed: dict[str, Any] = {}
    ok = True
    for seed in SEEDS:
        model = bp.build_bjoc_null(seed, norm, "cpu")
        full = src.state_dict_hash(model)
        group = mp.group_state_hash(model)
        encoder = bp.encoder_group_hash(model)
        checks = {
            "full_init_matches_p_round": full == frozen["p_init_full_sha256"][str(seed)],
            "group_init_matches_p_round": group == frozen["p_init_group_state_sha256"][str(seed)],
            "encoder_init_matches_p_round": encoder == frozen["p_init_encoder_sha256"][str(seed)],
            "parameter_audit": src.parameter_audit(src.BJOCConfig(), model)["actual_parameters"]
            == bp.EXPECTED_PARAMETERS,
            "recipe_R3_frozen": bp.RECIPES["R3"]
            == {"epochs": 360, "schedule": "cosine", "lr": 1.0e-3, "lr_min": 1.0e-5, "weight_decay": 1.0e-5},
        }
        per_seed[str(seed)] = {
            "init_full_sha256": full,
            "init_group_state_sha256": group,
            "init_encoder_sha256": encoder,
            "checks": checks,
        }
        ok = ok and all(checks.values())
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "batch_order_hash_D_s0": bp.batch_order_hash(0, 8000, bp.BATCH_SIZE, 360),
        "per_seed": per_seed,
        "ok": bool(ok),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(init_parity_path(), payload)
    log(f"[init-parity] ok={payload['ok']}")
    return 0 if payload["ok"] else 1


# ---------------------------------------------------------------------------
# Stage A: CPU correctness / feasibility
# ---------------------------------------------------------------------------


def _fit_raw_rows(split: Mapping[str, Any], ds, n: int) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    rows = [int(i) for i in split["fit_rows"][: int(n)]]
    out = []
    for i in rows:
        d = ds[i]
        out.append((i, d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy()))
    return out


def _strip_seconds(curve: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in entry.items() if k != "seconds"} for entry in curve]


def _compare_loops(payload_bp: Mapping[str, Any], payload_tat: Mapping[str, Any]) -> dict[str, Any]:
    curve_a = payload_bp["lr_curve"]
    curve_b = payload_tat["lr_curve"]
    fields = ("train_mae", "lr_first", "lr_last")
    curve_equal = len(curve_a) == len(curve_b) and all(
        all(a[f] == b[f] for f in fields) for a, b in zip(curve_a, curve_b)
    )
    return {
        "soup_state_sha256_bp": tat.state_sha256(payload_bp["soup_state"]),
        "soup_state_sha256_new": tat.state_sha256(payload_tat["soup_state"]),
        "soup_bit_identical": tat.state_sha256(payload_bp["soup_state"])
        == tat.state_sha256(payload_tat["soup_state"]),
        "batch_order_hash_bp": payload_bp["batch_order"]["full_hash"],
        "batch_order_hash_new": payload_tat["batch_order"]["full_hash"],
        "batch_order_identical": payload_bp["batch_order"]["full_hash"]
        == payload_tat["batch_order"]["full_hash"],
        "lr_curve_equal": bool(curve_equal),
        "soup_members_equal": list(payload_bp["soup_members"]) == list(payload_tat["soup_members"]),
        "soup_fit_mae_bp": payload_bp["soup_fit_mae"],
        "soup_fit_mae_new": payload_tat["soup_fit_mae"],
        "updates_bp": payload_bp["updates"],
        "updates_new": payload_tat["updates"],
    }


def cmd_stage_a(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    split, ds, guard = _guard_for_split()
    fit_objs_all = torch.load(objects_path("fit"), map_location="cpu", weights_only=False)
    norm = read_json(prep_path())
    device = torch.device("cpu")
    result: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }

    # --- A1 frozen-loop replay -------------------------------------------
    n_rows = int(args.replay_rows)
    rows = [int(i) for i in split["fit_rows"][:n_rows]]
    objs = fit_objs_all[:n_rows]
    y_fit = bp.load_y_rows(ENCODED_TRAIN, rows)
    epochs = int(args.replay_epochs)
    pb = bp.train_supervised(
        arm="P3-replay-bp",
        seed=0,
        recipe_key="R3",
        init="scratch",
        train_objs=objs,
        rows=rows,
        y_fit=y_fit,
        norm=norm,
        device=device,
        smoke_epochs=epochs,
    )
    pt = tat.train_task_adaptive(
        arm="P3-replay-new",
        seed=0,
        recipe_key="R3",
        init="scratch",
        train_objs=objs,
        rows=rows,
        y_fit=y_fit,
        norm=norm,
        device=device,
        aux_every=None,
        smoke_epochs=epochs,
    )
    repl = _compare_loops(pb, pt)
    repl["n_rows"] = n_rows
    repl["epochs"] = epochs
    repl["init_full_sha256_equal"] = pb["init_full_sha256"] == pt["init_full_sha256"]
    result["A1_replay"] = repl
    log(f"[stage-a A1] soup_identical={repl['soup_bit_identical']} order_identical={repl['batch_order_identical']}")
    if not (
        repl["soup_bit_identical"]
        and repl["batch_order_identical"]
        and repl["lr_curve_equal"]
        and repl["init_full_sha256_equal"]
    ):
        write_json(OUT / "stage_a.json", result)
        raise SystemExit("A1 replay failed: the new loop is not equivalent to bp.train_supervised")

    # --- A2 SSL wrapper equivalence + masked-view audits ------------------
    ssl_rows = [int(i) for i in split["fit_rows"][: int(args.ssl_smoke_molecules)]]
    ssl_objs = fit_objs_all[: int(args.ssl_smoke_molecules)]
    pb_ssl = bp.pretrain_masked(
        seed=0,
        fit_objs=ssl_objs,
        fit_rows=ssl_rows,
        raw_dataset=guard,
        device=device,
        mode="atom_bond",
        smoke_updates=int(args.ssl_smoke_updates),
    )
    pt_ssl = tat.pretrain_masked_with_head(
        seed=0,
        fit_objs=ssl_objs,
        fit_rows=ssl_rows,
        raw_dataset=guard,
        device=device,
        mode="atom_bond",
        smoke_updates=int(args.ssl_smoke_updates),
    )
    ssl_smoke = {
        "updates": int(args.ssl_smoke_updates),
        "n_molecules": int(args.ssl_smoke_molecules),
        "encoder_final_sha256_bp": pb_ssl["encoder_final_sha256"],
        "encoder_final_sha256_wrapper": pt_ssl["encoder_final_sha256"],
        "encoder_bit_identical": pb_ssl["encoder_final_sha256"] == pt_ssl["encoder_final_sha256"],
        "loss_at_steps_equal": pb_ssl["loss_at_steps"] == pt_ssl["loss_at_steps"],
        "curve_equal": _strip_seconds(pb_ssl["curve"]) == _strip_seconds(pt_ssl["curve"]),
        "head_parameters": pt_ssl["atom_head_parameters"],
        "head_sha256": pt_ssl["atom_head_sha256"],
        "labels_read": pt_ssl["labels_read"],
    }
    raw_rows = _fit_raw_rows(split, ds, int(args.audit_molecules))
    swap = tat.mask_swap_audit(raw_rows, n_trials=int(args.swap_trials))
    rebuild = tat.rebuild_audit(raw_rows, n_cases=int(args.rebuild_cases))
    result["A2_ssl"] = {
        "smoke": ssl_smoke,
        "mask_swap": swap,
        "rebuild": rebuild,
        "guarded_label_access_attempts": len(guard.access_attempts),
    }
    log(
        f"[stage-a A2] encoder_identical={ssl_smoke['encoder_bit_identical']} "
        f"swap={swap['verdict']} rebuild={rebuild['verdict']}"
    )
    if not (ssl_smoke["encoder_bit_identical"] and ssl_smoke["loss_at_steps_equal"]):
        write_json(OUT / "stage_a.json", result)
        raise SystemExit("A2 SSL wrapper smoke equality failed")
    if swap["verdict"] != "MASK_SWAP_INVARIANT" or rebuild["verdict"] != "MASKED_REBUILD_LEAK_FREE":
        write_json(OUT / "stage_a.json", result)
        raise SystemExit("A2 masked-view audit failed")
    if guard.access_attempts:
        raise SystemExit(f"guarded label access during Stage A: {guard.access_attempts[:3]}")

    # --- A3 gradient diagnostics at init and after 2 short epochs ---------
    head_state = pt_ssl["atom_head_state"]
    ssl_state_smoke = pt_ssl["encoder_state"]
    g0 = tat.gradient_diagnostics(
        seed=0,
        train_objs=fit_objs_all[: int(args.grad_rows)],
        rows=[int(i) for i in split["fit_rows"][: int(args.grad_rows)]],
        y_fit=bp.load_y_rows(ENCODED_TRAIN, [int(i) for i in split["fit_rows"][: int(args.grad_rows)]]),
        norm=norm,
        device=device,
        raw_dataset=guard,
        ssl_state=ssl_state_smoke,
        atom_head_state=head_state,
        n_batches=int(args.grad_batches),
        batch_size=64,
        visit_index=0,
    )
    short = tat.train_task_adaptive(
        arm="TA-short",
        seed=0,
        recipe_key="R3",
        init="ssl",
        train_objs=fit_objs_all[: int(args.grad_rows)],
        rows=[int(i) for i in split["fit_rows"][: int(args.grad_rows)]],
        y_fit=bp.load_y_rows(ENCODED_TRAIN, [int(i) for i in split["fit_rows"][: int(args.grad_rows)]]),
        norm=norm,
        device=device,
        ssl_state=ssl_state_smoke,
        atom_head_state=head_state,
        aux_every=tat.AUX_EVERY,
        aux_lambda=tat.AUX_LAMBDA,
        raw_dataset=guard,
        smoke_epochs=2,
    )
    short_model = bp.build_bjoc_null(0, norm, device)
    short_model.load_state_dict(short["soup_state"])
    short_head = tat.build_atom_head()
    short_head.load_state_dict(short["final_atom_head_state"])
    short_head.to(device)
    g1 = tat.gradient_diagnostics(
        seed=0,
        train_objs=fit_objs_all[: int(args.grad_rows)],
        rows=[int(i) for i in split["fit_rows"][: int(args.grad_rows)]],
        y_fit=bp.load_y_rows(ENCODED_TRAIN, [int(i) for i in split["fit_rows"][: int(args.grad_rows)]]),
        norm=norm,
        device=device,
        raw_dataset=guard,
        ssl_state=None,
        atom_head_state=head_state,
        n_batches=int(args.grad_batches),
        batch_size=64,
        visit_index=1,
        model=short_model,
        atom_head=short_head,
    )
    result["A3_gradients"] = {
        "at_init": g0,
        "after_2_epochs": g1,
        "short_train_soup_fit_mae": short["soup_fit_mae"],
        "short_train_aux_updates": short["aux_stats"]["aux_updates"],
        "short_train_updates": short["updates"],
    }
    log(
        f"[stage-a A3] init cos={g0['mean_cosine_property_atom']:+.3f} "
        f"ratio={g0['mean_grad_norm_ratio_atom_over_property']:.3f}; "
        f"short cos={g1['mean_cosine_property_atom']:+.3f}"
    )
    if not (g0["all_finite"] and g1["all_finite"] and g0["head_receives_atom_gradient"] and g1["head_receives_atom_gradient"]):
        write_json(OUT / "stage_a.json", result)
        raise SystemExit("A3 gradient diagnostic failed (non-finite or dead auxiliary head)")

    # --- A4 invariants and inference-path identity ------------------------
    inv_rows = 32
    inv_objs = fit_objs_all[:inv_rows]
    model = bp.build_bjoc_null(0, norm, device)
    mp.load_encoder_state(model, ssl_state_smoke, device)
    base_pred = bp.predict_model(model, inv_objs, device)
    relabeled = tat.relabel_objects(inv_objs, guard, seed=5)
    swapped = tat.endpoint_swap_objects(inv_objs, guard)
    relabel_diff = float(np.abs(bp.predict_model(model, relabeled, device) - base_pred).max())
    swap_diff = float(np.abs(bp.predict_model(model, swapped, device) - base_pred).max())
    batch_inv = tat.batch_invariance(model, inv_objs, device)
    replay = bp.build_bjoc_null(0, norm, device)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    replay.load_state_dict(state)
    replay_diff = float(np.abs(bp.predict_model(replay, inv_objs, device) - base_pred).max())
    audit = src.parameter_audit(src.BJOCConfig(), model)
    result["A4_invariance"] = {
        "inference_parameters": int(audit["actual_parameters"]),
        "inference_parameter_exact": bool(audit["parameter_exact"]),
        "state_keys_reference_keys_equal": sorted(model.state_dict().keys())
        == sorted(bp.build_bjoc_null(0, norm, device).state_dict().keys()),
        "has_auxiliary_head_attribute": hasattr(model, "atom_head"),
        "relabel_max_abs_diff": relabel_diff,
        "endpoint_swap_max_abs_diff": swap_diff,
        "batch_membership_max_abs_diff": batch_inv["batch_membership_max_abs_diff"],
        "save_load_replay_max_abs_diff": replay_diff,
    }
    tol = 1.0e-4
    ok_inv = (
        result["A4_invariance"]["inference_parameter_exact"]
        and result["A4_invariance"]["state_keys_reference_keys_equal"]
        and not result["A4_invariance"]["has_auxiliary_head_attribute"]
        and max(relabel_diff, swap_diff, batch_inv["batch_membership_max_abs_diff"], replay_diff) <= tol
    )
    result["A4_invariance"]["all_within_tolerance"] = bool(ok_inv)
    result["ok"] = bool(ok_inv)
    write_json(OUT / "stage_a.json", result)
    log(f"[stage-a A4] invariances ok={ok_inv} max_diff={max(relabel_diff, swap_diff):.2e}")
    if not ok_inv:
        raise SystemExit("A4 invariance gate failed")
    log("[stage-a] complete")
    return 0


# ---------------------------------------------------------------------------
# SSL + supervised arms
# ---------------------------------------------------------------------------


def cmd_ssl(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=True)
    seed = int(args.seed)
    ssl_pt, head_pt, ssl_json = ssl_paths(seed)
    if args.smoke_updates:
        ssl_pt = OUT / "smoke" / f"{ssl_pt.stem}_smoke.pt"
        head_pt = OUT / "smoke" / f"{head_pt.stem}_smoke.pt"
        ssl_json = OUT / "smoke" / f"{ssl_json.stem}_smoke.json"
    if ssl_pt.exists() or head_pt.exists() or ssl_json.exists():
        raise SystemExit(f"SSL D seed {seed} already exists; retraining is not allowed")
    split, ds, guard = _guard_for_split()
    fit_objs = torch.load(objects_path("fit"), map_location="cpu", weights_only=False)
    if [o.mol_index for o in fit_objs] != [int(i) for i in split["fit_rows"]]:
        raise SystemExit("fit object rows do not match the frozen split")
    payload = tat.pretrain_masked_with_head(
        seed=seed,
        fit_objs=fit_objs,
        fit_rows=split["fit_rows"],
        raw_dataset=guard,
        device=torch.device(args.device),
        mode="atom_bond",
        smoke_updates=args.smoke_updates,
    )
    encoder_state = payload.pop("encoder_state")
    head_state = payload.pop("atom_head_state")
    ssl_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder_state, ssl_pt)
    torch.save(head_state, head_pt)
    payload["encoder_path"] = str(ssl_pt.relative_to(REPO_ROOT))
    payload["encoder_file_sha256"] = file_sha256(ssl_pt)
    payload["atom_head_path"] = str(head_pt.relative_to(REPO_ROOT))
    payload["atom_head_file_sha256"] = file_sha256(head_pt)
    payload["split_tag"] = SPLIT_TAG
    payload["forbidden_label_access_attempts"] = len(guard.access_attempts)
    write_json(ssl_json, payload)
    if guard.access_attempts:
        raise SystemExit("guarded label access during SSL")
    if payload["labels_read"]:
        raise SystemExit("SSL unexpectedly read labels")
    log(
        f"[ssl D s{seed}] final acc atom {payload['final_acc_atom']:.4f} wall {payload['wall_seconds']:.0f}s "
        f"head={payload['atom_head_sha256'][:12]}"
    )
    if args.grad_diagnostics:
        grad = tat.gradient_diagnostics(
            seed=seed,
            train_objs=fit_objs[: int(args.grad_rows)],
            rows=list(split["fit_rows"][: int(args.grad_rows)]),
            y_fit=bp.load_y_rows(ENCODED_TRAIN, list(split["fit_rows"][: int(args.grad_rows)])),
            norm=read_json(prep_path()),
            device=torch.device(args.device),
            raw_dataset=guard,
            ssl_state=torch.load(ssl_pt, map_location="cpu", weights_only=False),
            atom_head_state=head_state,
            n_batches=int(args.grad_batches),
            batch_size=64,
            visit_index=0,
        )
        write_json(OUT / f"gradients_ssl_real_s{seed}.json", grad)
    return 0


def cmd_arm(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=True)
    arm = args.arm
    seed = int(args.seed)
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}")
    split, ds, guard = _guard_for_split()
    fit_objs = torch.load(objects_path("fit"), map_location="cpu", weights_only=False)
    if [o.mol_index for o in fit_objs] != [int(i) for i in split["fit_rows"]]:
        raise SystemExit("fit object rows do not match the frozen split")
    norm = read_json(prep_path())
    y_fit = bp.load_y_rows(ENCODED_TRAIN, split["fit_rows"])
    ssl_pt, head_pt, ssl_json = ssl_paths(seed)
    if not ssl_pt.exists():
        raise SystemExit(f"missing SSL checkpoint {ssl_pt}")
    ssl_state = torch.load(ssl_pt, map_location="cpu", weights_only=False)
    ssl_recorded = read_json(ssl_json)["encoder_final_sha256"]

    soup = soup_path(arm, seed)
    tj = train_json_path(arm, seed)
    smoke_epochs = args.smoke_epochs
    if smoke_epochs:
        soup = OUT / "smoke" / f"{soup.stem}_smoke.pt"
        tj = OUT / "smoke" / f"{tj.stem}_smoke.json"
    if soup.exists() or tj.exists():
        raise SystemExit(f"{arm} D seed {seed} already exists; retraining is not allowed")

    if arm == "P3":
        payload = bp.train_supervised(
            arm="P3",
            seed=seed,
            recipe_key="R3",
            init="ssl",
            train_objs=fit_objs,
            rows=split["fit_rows"],
            y_fit=y_fit,
            norm=norm,
            device=torch.device(args.device),
            ssl_state=ssl_state,
            model_kind="base",
            smoke_epochs=smoke_epochs,
        )
        payload["aux_schedule"] = None
    else:
        if not head_pt.exists():
            raise SystemExit(f"missing SSL atom head {head_pt}")
        head_state = torch.load(head_pt, map_location="cpu", weights_only=False)
        ssl_recorded_head = read_json(ssl_json)["atom_head_sha256"]
        if tat.state_sha256(head_state) != ssl_recorded_head:
            raise SystemExit("SSL atom head file does not match the recorded head hash")
        payload = tat.train_task_adaptive(
            arm="TA",
            seed=seed,
            recipe_key="R3",
            init="ssl",
            train_objs=fit_objs,
            rows=split["fit_rows"],
            y_fit=y_fit,
            norm=norm,
            device=torch.device(args.device),
            ssl_state=ssl_state,
            atom_head_state=head_state,
            aux_every=tat.AUX_EVERY,
            aux_lambda=tat.AUX_LAMBDA,
            raw_dataset=guard,
            smoke_epochs=smoke_epochs,
        )
    if payload["encoder_used_sha256"] != ssl_recorded:
        raise SystemExit("encoder used does not match the recorded pretraining state")
    soup.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu().clone() for k, v in payload["soup_state"].items()}, soup)
    payload.pop("soup_state", None)
    payload["soup_path"] = str(soup.relative_to(REPO_ROOT))
    payload["soup_sha256"] = file_sha256(soup)
    payload["split_tag"] = SPLIT_TAG
    payload["ssl_source"] = str(ssl_pt.relative_to(REPO_ROOT))
    payload["ssl_source_sha256"] = file_sha256(ssl_pt)
    payload["ssl_source_encoder_sha256"] = ssl_recorded
    if arm == "TA":
        payload["atom_head_source"] = str(head_pt.relative_to(REPO_ROOT))
        payload["atom_head_source_sha256"] = file_sha256(head_pt)
        final_head = payload.pop("final_atom_head_state", None)
        if final_head is not None:
            payload["head_info"]["final_atom_head_sha256"] = tat.state_sha256(final_head)
    else:
        payload.pop("final_atom_head_state", None)
    write_json(tj, payload)
    log(f"[{arm} D s{seed}] soup fit {payload['soup_fit_mae']:.5f} wall {payload['wall_seconds']:.0f}s")
    return 0


# ---------------------------------------------------------------------------
# label-blind predictions + freeze
# ---------------------------------------------------------------------------


def _load_arm_model(arm: str, seed: int, device: torch.device) -> src.BJOC:
    norm = read_json(prep_path())
    model = bp.build_bjoc_null(seed, norm, device)
    soup = torch.load(soup_path(arm, seed), map_location=device, weights_only=False)
    expected_keys = sorted(model.state_dict().keys())
    if sorted(soup.keys()) != expected_keys:
        raise SystemExit(f"{arm} soup keys are not a BJOC-Null state dict")
    model.load_state_dict(soup)
    return model


def cmd_predict(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    zones = {
        "monitor": torch.load(objects_path("monitor"), map_location="cpu", weights_only=False),
        "locked": torch.load(objects_path("locked"), map_location="cpu", weights_only=False),
    }
    device = torch.device(args.device)
    arms = list(args.arms) if args.arms else list(ARMS)
    seeds = [int(s) for s in args.seeds]
    manifest = read_json(PRED_MANIFEST) if PRED_MANIFEST.exists() else {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": SPLIT_TAG,
        "entries": {},
        "monitor_labels_read": False,
        "locked_labels_read": False,
        "official_test_loaded": False,
    }
    for arm in arms:
        for seed in seeds:
            s = soup_path(arm, seed)
            if not s.exists():
                raise SystemExit(f"missing soup {s}")
            model = _load_arm_model(arm, seed, device)
            for zone, objs in zones.items():
                out = pred_path(zone, arm, seed)
                key = f"{zone}:{arm}:{seed}"
                pred = bp.predict_model(model, objs, device)
                if not np.isfinite(pred).all():
                    raise SystemExit(f"non-finite {zone} prediction {arm} {seed}")
                if out.exists():
                    existing = manifest["entries"].get(key)
                    if existing and existing["sha256"] == file_sha256(out):
                        log(f"[predict] {zone} {arm} s{seed} already frozen")
                        continue
                    raise SystemExit(f"prediction file {out} exists but is not frozen in the manifest")
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, row=np.arange(len(objs), dtype=np.int64), pred=pred.astype(np.float32))
                manifest["entries"][key] = {
                    "path": str(out.relative_to(REPO_ROOT)),
                    "sha256": file_sha256(out),
                    "soup_sha256": file_sha256(s),
                    "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                log(f"[predict] {zone} {arm} s{seed} -> {out.name}")
    manifest["monitor_labels_read"] = bool(MONITOR_LOCK.exists())
    manifest["locked_labels_read"] = bool(LOCKED_LOCK.exists())
    write_json(PRED_MANIFEST, manifest)
    log("[predict] manifest written")
    return 0


def _verify_prediction_manifest(require_complete: bool = True) -> dict[str, Any]:
    manifest = read_json(PRED_MANIFEST)
    for key, rec in manifest["entries"].items():
        p = REPO_ROOT / rec["path"]
        if not p.exists() or file_sha256(p) != rec["sha256"]:
            raise SystemExit(f"prediction drift for {key}")
    if require_complete:
        expected = {f"{zone}:{arm}:{seed}" for zone in ("monitor", "locked") for arm in ARMS for seed in SEEDS}
        missing = expected - set(manifest["entries"])
        if missing:
            raise SystemExit(f"prediction manifest incomplete: {sorted(missing)}")
    return manifest


def _read_npz_pred(path: Path, n: int) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    pred = np.asarray(z["pred"], dtype=np.float64).reshape(-1)
    row = np.asarray(z["row"], dtype=np.int64).reshape(-1)
    if len(pred) != n or not np.array_equal(row, np.arange(n)):
        raise SystemExit(f"row/pred shape mismatch in {path}")
    return pred


def _load_preds(zone: str, arm: str, seeds: Sequence[int], n: int) -> dict[int, np.ndarray]:
    return {int(s): _read_npz_pred(pred_path(zone, arm, int(s)), n) for s in seeds}


def _ensure_monitor_y() -> np.ndarray:
    split = read_json(split_path())
    if MONITOR_LOCK.exists():
        y = np.asarray(np.load(MONITOR_Y), dtype=np.float64).reshape(-1)
        lock = read_json(MONITOR_LOCK)
        if bp.array_sha256(y) != lock["y_sha256"]:
            raise SystemExit("monitor y file drifted from its lock record")
        return y
    rows = split["monitor_rows"]
    y = bp.load_y_rows(ENCODED_TRAIN, rows)
    ds = cf.load_official_train()
    max_diff = 0.0
    for i, v in zip(rows, y):
        max_diff = max(max_diff, abs(float(ds[int(i)].y.reshape(-1)[0]) - float(v)))
    if max_diff > 1e-9:
        raise SystemExit(f"monitor y alignment failed: {max_diff}")
    np.save(MONITOR_Y, y)
    write_json(
        MONITOR_LOCK,
        {
            "protocol": PROTOCOL_ID,
            "split_tag": SPLIT_TAG,
            "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_rows": len(rows),
            "y_sha256": bp.array_sha256(y),
            "alignment_max_abs_diff": float(max_diff),
            "prediction_manifest_sha256": file_sha256(PRED_MANIFEST),
        },
    )
    log(f"[monitor-read] n={len(y)} sha={bp.array_sha256(y)[:12]}")
    return y


def _read_locked_y_once() -> np.ndarray:
    split = read_json(split_path())
    if LOCKED_LOCK.exists():
        raise SystemExit("Locked labels already read for split D")
    rows = split["locked_rows"]
    y = bp.load_y_rows(ENCODED_TRAIN, rows)
    ds = cf.load_official_train()
    max_diff = 0.0
    for i, v in zip(rows, y):
        max_diff = max(max_diff, abs(float(ds[int(i)].y.reshape(-1)[0]) - float(v)))
    if max_diff > 1e-9:
        raise SystemExit(f"locked y alignment failed: {max_diff}")
    np.save(LOCKED_Y, y)
    write_json(
        LOCKED_LOCK,
        {
            "protocol": PROTOCOL_ID,
            "split_tag": SPLIT_TAG,
            "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_rows": len(rows),
            "y_sha256": bp.array_sha256(y),
            "alignment_max_abs_diff": float(max_diff),
            "locked_manifest_sha256": file_sha256(LOCKED_MANIFEST),
        },
    )
    log(f"[locked-read] n={len(y)} sha={bp.array_sha256(y)[:12]}")
    return y


def _stage_a_ok() -> bool:
    if not (OUT / "stage_a.json").exists():
        return False
    return bool(read_json(OUT / "stage_a.json").get("ok", False))


def _audits_ok() -> dict[str, Any]:
    stage_a = read_json(OUT / "stage_a.json")
    objects = read_json(objects_manifest_path())
    split = read_json(split_path())
    checks = {
        "stage_a_ok": bool(stage_a.get("ok", False)),
        "A1_soup_identical": bool(stage_a["A1_replay"]["soup_bit_identical"]),
        "A2_encoder_identical": bool(stage_a["A2_ssl"]["smoke"]["encoder_bit_identical"]),
        "A2_mask_swap_invariant": stage_a["A2_ssl"]["mask_swap"]["verdict"] == "MASK_SWAP_INVARIANT",
        "A2_rebuild_leak_free": stage_a["A2_ssl"]["rebuild"]["verdict"] == "MASKED_REBUILD_LEAK_FREE",
        "A3_gradients_finite": bool(stage_a["A3_gradients"]["at_init"]["all_finite"]),
        "A4_invariance": bool(stage_a["A4_invariance"]["all_within_tolerance"]),
        "object_build_label_guard_clean": int(objects["forbidden_label_access_attempts"]) == 0,
        "split_prior_use_clean": not bool(split["prior_use_audit"]["prior_d_training_detected"]),
        "init_parity_ok": bool(read_json(init_parity_path())["ok"]),
    }
    return {"checks": checks, "ok": bool(all(checks.values()))}


def cmd_score_monitor(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    if MONITOR_LOCK.exists():
        raise SystemExit("Monitor labels already read; refusing a second read")
    audits = _audits_ok()
    if not audits["ok"]:
        raise SystemExit(f"audit gate failed before Monitor read: {audits}")
    _verify_prediction_manifest(require_complete=True)
    y = _ensure_monitor_y()
    n = len(y)
    seeds = list(SEEDS)
    p3 = _load_preds("monitor", "P3", seeds, n)
    ta = _load_preds("monitor", "TA", seeds, n)
    contrast = ab.compute_contrast(ta, p3, y)
    gate = ab.monitor_gate(contrast, two_seed=True)
    conditions = dict(gate["conditions"])
    conditions["audits_ok"] = bool(audits["ok"])
    passed = bool(all(conditions.values()))
    if passed:
        decision = "TASSL_GAIN_PROMISING"
    elif contrast["delta_seed_mean"] < 0:
        decision = "TASSL_WEAK_OR_SPLIT_SENSITIVE"
    else:
        decision = "TASSL_NO_GAIN"
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": SPLIT_TAG,
        "candidate": "TA",
        "reference": "P3",
        "seeds": seeds,
        "contrast_vs_P3": contrast,
        "gate_conditions": conditions,
        "gate_passed": passed,
        "audits": audits,
        "decision": decision,
        "prediction_manifest_sha256": file_sha256(PRED_MANIFEST),
        "monitor_read_utc": read_json(MONITOR_LOCK)["read_utc"],
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "monitor_TA_D.json", payload)
    write_json(
        DECISION,
        {
            "protocol": PROTOCOL_ID,
            "decision": decision,
            "gate_passed": passed,
            "delta_seed_mean": contrast["delta_seed_mean"],
            "per_seed_delta": contrast["per_seed_delta"],
            "bootstrap_ci95": [contrast["bootstrap"]["ci_lo"], contrast["bootstrap"]["ci_hi"]],
            "locked_read_authorized": passed,
            "official_test_loaded": False,
        },
    )
    rounds = read_json(MONITOR_ROUNDS) if MONITOR_ROUNDS.exists() else {"protocol": PROTOCOL_ID, "rounds": []}
    rounds["rounds"].append(
        {
            "round": len(rounds["rounds"]) + 1,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "candidate": "TA",
            "seeds": seeds,
            "decision": decision,
            "delta_seed_mean": contrast["delta_seed_mean"],
            "prediction_manifest_sha256": payload["prediction_manifest_sha256"],
        }
    )
    write_json(MONITOR_ROUNDS, rounds)
    log(f"[score-monitor] {decision} delta {contrast['delta_seed_mean']:+.5f}")
    log(
        f"  CI95 [{contrast['bootstrap']['ci_lo']:+.5f},{contrast['bootstrap']['ci_hi']:+.5f}] "
        f"per-seed {contrast['per_seed_delta']} ordinary {contrast['ordinary_delta']:+.5f} "
        f"top10 {contrast['top10_removed_delta']:+.5f} rmse {contrast['rmse_new'] - contrast['rmse_ref']:+.5f} "
        f"single-share {contrast['max_single_gain_fraction']:.3f}"
    )
    return 0


# ---------------------------------------------------------------------------
# Locked final (conditional)
# ---------------------------------------------------------------------------


def cmd_locked_freeze(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    if LOCKED_LOCK.exists():
        raise SystemExit("Locked labels already read; refusing to refreeze")
    decision = read_json(DECISION)
    if not decision.get("gate_passed"):
        raise SystemExit("Monitor gate did not pass; Locked read is not authorized")
    manifest = _verify_prediction_manifest(require_complete=True)
    entries = {f"locked:{arm}:{seed}": manifest["entries"][f"locked:{arm}:{seed}"] for arm in ARMS for seed in SEEDS}
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": SPLIT_TAG,
        "arms": list(ARMS),
        "arm_seeds": {arm: list(SEEDS) for arm in ARMS},
        "entries": entries,
        "monitor_decision_sha256": file_sha256(DECISION),
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_test_loaded": False,
    }
    write_json(LOCKED_MANIFEST, payload)
    log("[locked-freeze] written")
    return 0


def cmd_locked_score(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    locked_manifest = read_json(LOCKED_MANIFEST)
    for key, rec in locked_manifest["entries"].items():
        p = REPO_ROOT / rec["path"]
        if not p.exists() or file_sha256(p) != rec["sha256"]:
            raise SystemExit(f"locked prediction drift for {key}")
    y = _read_locked_y_once()
    n = len(y)
    preds = {arm: _load_preds("locked", arm, SEEDS, n) for arm in ARMS}
    metrics = {}
    for arm in ARMS:
        per_seed = {str(s): float(np.abs(preds[arm][s] - y).mean()) for s in SEEDS}
        m = np.mean([np.abs(preds[arm][s] - y) for s in SEEDS], axis=0)
        metrics[arm] = {
            "per_seed_mae": per_seed,
            "seed_mean_mae": float(m.mean()),
            "median_abs": float(np.median(m)),
            "p90_abs": float(np.quantile(m, 0.90)),
            "rmse": float(np.sqrt(np.mean(np.mean([(preds[arm][s] - y) ** 2 for s in SEEDS], axis=0)))),
        }
    contrast = ab.compute_contrast(preds["TA"], preds["P3"], y)
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": SPLIT_TAG,
        "metrics": metrics,
        "contrast_TA_minus_P3": contrast,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "locked_final_D.json", payload)
    log("[locked-score] done")
    for arm in ARMS:
        log(f"  {arm}: seed-mean locked MAE {metrics[arm]['seed_mean_mae']:.6f} per-seed {metrics[arm]['per_seed_mae']}")
    log(f"  TA-P3 delta {contrast['delta_seed_mean']:+.5f} CI95 [{contrast['bootstrap']['ci_lo']:+.5f},{contrast['bootstrap']['ci_hi']:+.5f}]")
    return 0


# ---------------------------------------------------------------------------
# conditional compute control
# ---------------------------------------------------------------------------


def control_soup_path(seed: int) -> Path:
    return OUT / f"soup_P3EXTRA_D_s{seed}.pt"


def control_json_path(seed: int) -> Path:
    return OUT / f"train_P3EXTRA_D_s{seed}.json"


def cmd_control_train(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=True)
    decision = read_json(DECISION)
    if not decision.get("gate_passed"):
        raise SystemExit("compute control not authorized: Monitor gate did not pass")
    seed = int(args.seed)
    split, ds, guard = _guard_for_split()
    fit_objs = torch.load(objects_path("fit"), map_location="cpu", weights_only=False)
    norm = read_json(prep_path())
    y_fit = bp.load_y_rows(ENCODED_TRAIN, split["fit_rows"])
    ssl_pt, _, ssl_json = ssl_paths(seed)
    ssl_state = torch.load(ssl_pt, map_location="cpu", weights_only=False)
    ssl_recorded = read_json(ssl_json)["encoder_final_sha256"]
    soup = control_soup_path(seed)
    tj = control_json_path(seed)
    if soup.exists() or tj.exists():
        raise SystemExit(f"P3-EXTRA seed {seed} already exists")
    OUT.mkdir(parents=True, exist_ok=True)
    recipe = {
        "epochs": tat.CONTROL_EPOCHS,
        "schedule": "cosine",
        "lr": 1.0e-3,
        "lr_min": 1.0e-5,
        "weight_decay": 1.0e-5,
    }
    payload = tat.train_task_adaptive(
        arm="P3-EXTRA",
        seed=seed,
        recipe_key="R3",
        recipe_override=recipe,
        init="ssl",
        train_objs=fit_objs,
        rows=split["fit_rows"],
        y_fit=y_fit,
        norm=norm,
        device=torch.device(args.device),
        ssl_state=ssl_state,
        aux_every=None,
    )
    if payload["encoder_used_sha256"] != ssl_recorded:
        raise SystemExit("encoder used does not match the recorded pretraining state")
    torch.save({k: v.detach().cpu().clone() for k, v in payload["soup_state"].items()}, soup)
    payload.pop("soup_state", None)
    payload["soup_path"] = str(soup.relative_to(REPO_ROOT))
    payload["soup_sha256"] = file_sha256(soup)
    payload["split_tag"] = SPLIT_TAG
    payload["ssl_source"] = str(ssl_pt.relative_to(REPO_ROOT))
    payload["ssl_source_sha256"] = file_sha256(ssl_pt)
    payload["control_design"] = (
        "same SSL encoder and scratch init as P3; the R3 per-step cosine schedule is stretched to "
        f"T_max = {tat.CONTROL_UPDATES} updates (22,680 * 1.25, matching the 5,670 auxiliary "
        "forward/backward passes at one extra update-equivalent each); last-5 soup; approximate, "
        "not exact FLOP match, and it sees the Fit labels in 5,670 extra updates"
    )
    write_json(tj, payload)
    log(f"[P3-EXTRA D s{seed}] soup fit {payload['soup_fit_mae']:.5f} wall {payload['wall_seconds']:.0f}s")
    return 0


def cmd_control_score(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    decision = read_json(DECISION)
    if not decision.get("gate_passed"):
        raise SystemExit("compute control not authorized: Monitor gate did not pass")
    # monitor labels are already read (Monitor read is declared and locked)
    y = np.asarray(np.load(MONITOR_Y), dtype=np.float64).reshape(-1)
    n = len(y)
    ta = _load_preds("monitor", "TA", SEEDS, n)
    p3 = _load_preds("monitor", "P3", SEEDS, n)
    extra = {}
    monitor_objs = torch.load(objects_path("monitor"), map_location="cpu", weights_only=False)
    for seed in SEEDS:
        if not control_soup_path(seed).exists():
            continue
        model = bp.build_bjoc_null(seed, read_json(prep_path()), torch.device(args.device))
        model.load_state_dict(torch.load(control_soup_path(seed), map_location=args.device, weights_only=False))
        extra[seed] = bp.predict_model(model, monitor_objs, torch.device(args.device))
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": SPLIT_TAG,
        "note": "descriptive compute-control contrast on the single declared Monitor read",
        "contrast_TA_minus_P3EXTRA": ab.compute_contrast(ta, extra, y) if extra else None,
        "contrast_P3EXTRA_minus_P3": ab.compute_contrast(extra, p3, y) if extra else None,
        "seeds": sorted(extra),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "control_score_D.json", payload)
    log("[control-score] written")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prep")
    prep.set_defaults(cmd="prep")

    sub.add_parser("verify-inputs")
    sub.add_parser("freeze")
    sub.add_parser("init-parity")
    sub.add_parser("objects")

    sa = sub.add_parser("stage-a")
    sa.add_argument("--replay-rows", type=int, default=640)
    sa.add_argument("--replay-epochs", type=int, default=2)
    sa.add_argument("--ssl-smoke-molecules", type=int, default=256)
    sa.add_argument("--ssl-smoke-updates", type=int, default=5)
    sa.add_argument("--audit-molecules", type=int, default=200)
    sa.add_argument("--swap-trials", type=int, default=100)
    sa.add_argument("--rebuild-cases", type=int, default=50)
    sa.add_argument("--grad-rows", type=int, default=256)
    sa.add_argument("--grad-batches", type=int, default=4)

    ssl = sub.add_parser("ssl")
    ssl.add_argument("--seed", type=int, required=True)
    ssl.add_argument("--device", default="cpu")
    ssl.add_argument("--smoke-updates", type=int, default=None)
    ssl.add_argument("--grad-diagnostics", action="store_true")
    ssl.add_argument("--grad-rows", type=int, default=256)
    ssl.add_argument("--grad-batches", type=int, default=4)

    a = sub.add_parser("arm")
    a.add_argument("--arm", required=True, choices=list(ARMS))
    a.add_argument("--seed", type=int, required=True)
    a.add_argument("--device", default="cpu")
    a.add_argument("--smoke-epochs", type=int, default=None)

    pr = sub.add_parser("predict")
    pr.add_argument("--arms", nargs="*", default=None)
    pr.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    pr.add_argument("--device", default="cpu")

    sm = sub.add_parser("score-monitor")
    sm.set_defaults(cmd="score-monitor")

    lf = sub.add_parser("locked-freeze")
    lf.set_defaults(cmd="locked-freeze")
    ls = sub.add_parser("locked-score")
    ls.set_defaults(cmd="locked-score")

    ct = sub.add_parser("control-train")
    ct.add_argument("--seed", type=int, required=True)
    ct.add_argument("--device", default="cpu")
    cs = sub.add_parser("control-score")
    cs.add_argument("--device", default="cpu")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "prep":
        return cmd_prep(args)
    if args.cmd == "verify-inputs":
        return cmd_verify_inputs(args)
    if args.cmd == "freeze":
        return cmd_freeze(args)
    if args.cmd == "init-parity":
        return cmd_init_parity(args)
    if args.cmd == "objects":
        return cmd_objects(args)
    if args.cmd == "stage-a":
        return cmd_stage_a(args)
    if args.cmd == "ssl":
        return cmd_ssl(args)
    if args.cmd == "arm":
        return cmd_arm(args)
    if args.cmd == "predict":
        return cmd_predict(args)
    if args.cmd == "score-monitor":
        return cmd_score_monitor(args)
    if args.cmd == "locked-freeze":
        return cmd_locked_freeze(args)
    if args.cmd == "locked-score":
        return cmd_locked_score(args)
    if args.cmd == "control-train":
        return cmd_control_train(args)
    if args.cmd == "control-score":
        return cmd_control_score(args)
    raise SystemExit(f"unknown command {args.cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())
