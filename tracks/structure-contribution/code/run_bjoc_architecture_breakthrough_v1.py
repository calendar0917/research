"""cscl-bjoc-architecture-breakthrough-v1 driver.

Protocol: ``configs/bjoc_architecture_breakthrough_v1.yaml`` (frozen before
any GPU trajectory).  Track: ``tracks/structure-contribution``.

Stages
------
``prep``            split C/D fingerprints, guarded object + pair caches for C,
                    Fit-only standardiser, init-parity audit, protocol freeze
``objects``         materialise guarded object/pair caches for a split tag (D)
``cpu-gate-glrc``   GLRC CPU correctness / invariance / ownership gate
``cpu-gate-dlp``    DLP CPU correctness / information / invariance gate
``ssl``             Fit-only masked SSL for a split tag + seed
``arm``             one supervised arm (P3 baseline or a candidate)
``predict``         label-blind Monitor/Locked forwards + hash freeze
``score-monitor``   declared Monitor scoring round for one candidate
``locked-freeze``   freeze the final prediction set before the Locked read
``locked-score``    the one Locked label read + final table
``full10k-train``   conditional exposed-valid refit on the official train 10k
``full10k-score``   descriptive official-valid metrics for a refit arm

Discipline: official test is never referenced by any code path in this driver;
Monitor/Locked structures are built under :class:`ab.LabelGuard` so their
``.y`` is unreachable during preprocessing, SSL, training and prediction; the
single Monitor label read and the single Locked label read are lock-filed.
"""

from __future__ import annotations

import argparse
import json
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
import cscl_features as cf  # noqa: E402
import run_bjoc_full10k_pretraining_transfer_v1 as f10k  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = ab.PROTOCOL_ID
CONFIG_PATH = _CODE_DIR.parent / "configs" / "bjoc_architecture_breakthrough_v1.yaml"
OUT = REPO_ROOT / "tracks/structure-contribution/results/bjoc_architecture_breakthrough_v1"
F10K = REPO_ROOT / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1"
REC = REPO_ROOT / "tracks/structure-contribution/results/bjoc_recipe_tuning_v1"
MPR = REPO_ROOT / "tracks/structure-contribution/results/bjoc_masked_pretraining_v1"
PERF = REPO_ROOT / "tracks/structure-contribution/results/bjoc_performance_mechanism_long_v1"
ENCODED_TRAIN = REPO_ROOT / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt"
CANONICAL_SMILES = REPO_ROOT / "tracks/ksvd/results/zinc_cssd_basis_reuse_v1/train_canonical_smiles.npz"

N_TRAIN = 10000
N_VALID = 1000
SEEDS = (0, 1)
SPLIT_TAGS = ("C", "D")
BASE_ARMS = ("P3",)
CANDIDATE_ARMS = ("GLRC", "DLP", "ORGANIZED")
_BASE_READER_IN = int(src.BJOCConfig().reader_in)


def log(msg: str) -> None:
    print(msg, flush=True)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(f10k._j(payload), indent=2), encoding="utf-8")


def git_commit() -> str:
    return f10k.git_commit()


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return f10k.file_sha256(path)


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def split_path(tag: str) -> Path:
    return OUT / f"split_{tag}.json"


def objects_path(zone: str, tag: str) -> Path:
    return OUT / f"objects_{zone}_{tag}.pt"


def pairs_path(zone: str, tag: str) -> Path:
    return OUT / f"pairs_{zone}_{tag}.pt"


def prep_path(tag: str) -> Path:
    return OUT / f"prep_fit_{tag}.json"


def objects_manifest_path(tag: str) -> Path:
    return OUT / f"objects_manifest_{tag}.json"


def ssl_paths(tag: str, seed: int) -> tuple[Path, Path]:
    stem = f"pretrain_{tag}_s{seed}"
    return OUT / f"{stem}.pt", OUT / f"{stem}.json"


def soup_path(tag: str, arm: str, seed: int) -> Path:
    return OUT / f"soup_{arm}_{tag}_s{seed}.pt"


def train_json_path(tag: str, arm: str, seed: int) -> Path:
    return OUT / f"train_{arm}_{tag}_s{seed}.json"


def pred_path(zone: str, tag: str, arm: str, seed: int) -> Path:
    return OUT / "predictions" / f"{zone}_{arm}_{tag}_s{seed}.npz"


PRED_MANIFEST = {tag: OUT / f"prediction_manifest_{tag}.json" for tag in SPLIT_TAGS}
MONITOR_LOCK = {tag: OUT / f"monitor_read_{tag}.lock" for tag in SPLIT_TAGS}
MONITOR_Y = {tag: OUT / f"monitor_y_{tag}.npy" for tag in SPLIT_TAGS}
MONITOR_ROUNDS = {tag: OUT / f"monitor_rounds_{tag}.json" for tag in SPLIT_TAGS}
LOCKED_LOCK = {tag: OUT / f"locked_read_{tag}.lock" for tag in SPLIT_TAGS}
LOCKED_Y = {tag: OUT / f"locked_y_{tag}.npy" for tag in SPLIT_TAGS}
LOCKED_MANIFEST = {tag: OUT / f"locked_manifest_{tag}.json" for tag in SPLIT_TAGS}
FREEZE_JSON = OUT / "protocol_freeze.json"
FULL10K_SCORE = OUT / "full10k_score.json"


# ---------------------------------------------------------------------------
# code / input freeze
# ---------------------------------------------------------------------------


def _code_paths() -> dict[str, Path]:
    return {
        "driver": _CODE_DIR / "run_bjoc_architecture_breakthrough_v1.py",
        "module": _CODE_DIR / "bjoc_arch_breakthrough_v1.py",
        "config": CONFIG_PATH,
        "base_module": _CODE_DIR / "bjoc_perfmech_long_v1.py",
        "masked_module": _CODE_DIR / "bjoc_masked_pretraining_v1.py",
        "objects": _CODE_DIR / "structural_relation_objects_v1.py",
        "composer": _CODE_DIR / "structural_relation_composer_v1.py",
        "f10k_driver": _CODE_DIR / "run_bjoc_full10k_pretraining_transfer_v1.py",
    }


def code_hashes() -> dict[str, str]:
    return {k: file_sha256(p) for k, p in _code_paths().items()}


def _frozen_input_files() -> dict[str, Path]:
    return {
        "canonical_smiles": CANONICAL_SMILES,
        "encoded_train": ENCODED_TRAIN,
        "objects_train10k": F10K / "objects_train10k.pt",
        "prep_train10k": F10K / "prep_train10k.json",
        "pretrain_encoder_s0": F10K / "pretrain_encoder_s0.pt",
        "pretrain_encoder_s1": F10K / "pretrain_encoder_s1.pt",
    }


def audit_inputs() -> dict[str, Any]:
    cfg = load_config()
    frozen = cfg["frozen"]
    expected = {
        "canonical_smiles": frozen["canonical_smiles_sha256"],
        "encoded_train": frozen["encoded_train_sha256"],
        "objects_train10k": frozen["objects_train10k_sha256"],
        "prep_train10k": frozen["prep_train10k_sha256"],
        "pretrain_encoder_s0": frozen["pretrain_encoder_s0_sha256"],
        "pretrain_encoder_s1": frozen["pretrain_encoder_s1_sha256"],
    }
    checks: dict[str, Any] = {}
    ok = True
    for name, path in _frozen_input_files().items():
        exists = Path(path).exists()
        actual = file_sha256(path) if exists else None
        hit = bool(exists and actual == expected[name])
        ok = ok and hit
        checks[name] = {
            "path": str(Path(path).relative_to(REPO_ROOT)),
            "exists": exists,
            "sha256": actual,
            "matches": hit,
        }
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
    cfg = load_config()
    loads = dict(cfg["label_access"])
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "code_hashes": code_hashes(),
        "arm_registry": ab.ARM_KINDS,
        "split_salts": ab.SPLIT_SALTS,
        "pre_registered_split_fingerprints": cfg["splits"]["fingerprints"],
        "dlp": {"rank": ab.DLP_RANK, "buckets": list(ab.DLP_BUCKET_LABELS)},
        "glrc_gate": ab.GLRC_GATE_LAMBDA,
        "parameter_budgets": {
            "P3": bp.EXPECTED_PARAMETERS,
            "GLRC": ab.GLRC_PARAMETERS,
            "DLP": ab.DLP_PARAMETERS,
            "ORGANIZED": bp.EXPECTED_PARAMETERS,
        },
        "input_audit": {k: v["sha256"] for k, v in audit["checks"].items()},
        "input_audit_ok": True,
        "label_access": loads,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_test_loaded": False,
    }
    write_json(FREEZE_JSON, payload)
    log(f"[freeze] written {FREEZE_JSON.relative_to(REPO_ROOT)} commit={payload['git_commit'][:12]}")
    return 0


def verify_freeze(*, strict_code: bool = True) -> dict[str, Any]:
    if not FREEZE_JSON.exists():
        raise SystemExit("protocol is not frozen; run `prep` (or `freeze`) first")
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
# splits / guarded objects / pairs
# ---------------------------------------------------------------------------


def build_split_payload(tag: str) -> dict[str, Any]:
    cfg = load_config()
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    if len(smiles) != N_TRAIN:
        raise SystemExit(f"canonical smiles table has {len(smiles)} rows")
    split = bp.build_split(smiles, ab.SPLIT_SALTS[tag])
    fit = [int(x) for x in split["fit"]]
    monitor = [int(x) for x in split["monitor"]]
    locked = [int(x) for x in split["locked"]]
    expected = cfg["splits"]["fingerprints"][tag]
    got = {
        "n_fit": len(fit),
        "n_monitor": len(monitor),
        "n_locked": len(locked),
        "fit_sha256_rows": bp.rows_sha256(fit),
        "monitor_sha256_rows": bp.rows_sha256(monitor),
        "locked_sha256_rows": bp.rows_sha256(locked),
    }
    for key, value in got.items():
        if expected[key] != value:
            raise SystemExit(f"split {tag} fingerprint mismatch for {key}: {value} != {expected[key]}")
    overlap: dict[str, Any] = {}
    for other, path in (("A", MPR / "split_a.json"), ("B", PERF / "split_B.json")):
        if not path.exists():
            overlap[other] = None
            continue
        d = read_json(path)
        other_mon = set(map(int, d["monitor_rows"]))
        other_lock = set(map(int, d["locked_rows"]))
        prev_read = other_mon | other_lock
        overlap[other] = {
            "fit_in_prev_fit": len(set(fit) & set(map(int, d["fit_rows"]))),
            "fit_in_prev_read_zones": len(set(fit) & prev_read),
            "monitor_in_prev_fit": len(set(monitor) & set(map(int, d["fit_rows"]))),
            "monitor_in_prev_read_zones": len(set(monitor) & prev_read),
            "locked_in_prev_fit": len(set(locked) & set(map(int, d["fit_rows"]))),
            "locked_in_prev_read_zones": len(set(locked) & prev_read),
        }
    union = set(fit) | set(monitor) | set(locked)
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": tag,
        "split_salt": ab.SPLIT_SALTS[tag],
        "n_groups": int(split["n_groups"]),
        **got,
        "fit_rows": fit,
        "monitor_rows": monitor,
        "locked_rows": locked,
        "overlap_rows_with_previous_splits": overlap,
        "union_rows": len(union),
        "duplicate_canonical_smiles_groups": int(len(split["duplicate_groups"])),
        "duplicate_groups_cross_sides": int(split["duplicate_groups_cross_sides"]),
        "label_information_used": False,
        "monitor_labels_read": False,
        "locked_labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "disclosure": (
            "internal re-partition of the official ZINC train 10000; these molecules have been used "
            "in historical experiments (including full-10k training and previously read internal "
            "monitor/locked blocks), so this is NOT an independent external test split"
        ),
    }
    return payload


def cmd_splits(args: argparse.Namespace) -> int:
    for tag in args.tags:
        payload = build_split_payload(tag)
        write_json(split_path(tag), payload)
        log(
            f"[splits:{tag}] fit={payload['n_fit']} monitor={payload['n_monitor']} locked={payload['n_locked']} "
            f"dup_groups_cross={payload['duplicate_groups_cross_sides']}"
        )
    return 0


def _save_pair_cache(cache: Mapping[int, Mapping[str, np.ndarray]], path: Path) -> None:
    payload = {
        int(k): {
            "i": np.asarray(v["i"], dtype=np.int64),
            "j": np.asarray(v["j"], dtype=np.int64),
            "bucket": np.asarray(v["bucket"], dtype=np.int64),
        }
        for k, v in cache.items()
    }
    torch.save(payload, path)


def load_pair_cache(path: Path) -> dict[int, dict[str, np.ndarray]]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return {int(k): {kk: np.asarray(vv) for kk, vv in v.items()} for k, v in raw.items()}


def _guarded_dataset(forbidden_rows: Sequence[int]):
    ds = cf.load_official_train()
    guard = ab.LabelGuard(ds, forbidden_rows)
    return ds, guard


def _build_zone_objects(rows: Sequence[int], guard, ds) -> list[sro.MolObjects]:
    objs = []
    for i in rows:
        d = guard[int(i)]
        objs.append(sro.build_mol_objects(int(i), d.x.numpy(), d.edge_index.numpy(), d.edge_attr.numpy()))
    return objs


def cmd_objects(args: argparse.Namespace) -> int:
    tag = args.tag
    split = read_json(split_path(tag))
    ds, guard = _guarded_dataset(split["monitor_rows"] + split["locked_rows"])
    t0 = time.perf_counter()
    fit_objs = _build_zone_objects(split["fit_rows"], guard, ds)
    mon_objs = _build_zone_objects(split["monitor_rows"], guard, ds)
    lock_objs = _build_zone_objects(split["locked_rows"], guard, ds)
    build_seconds = time.perf_counter() - t0
    for objs, rows, key in (
        (fit_objs, split["fit_rows"], "fit"),
        (mon_objs, split["monitor_rows"], "monitor"),
        (lock_objs, split["locked_rows"], "locked"),
    ):
        if [o.mol_index for o in objs] != [int(i) for i in rows]:
            raise SystemExit(f"object row order mismatch for {key}")
    torch.save(fit_objs, objects_path("fit", tag))
    torch.save(mon_objs, objects_path("monitor", tag))
    torch.save(lock_objs, objects_path("locked", tag))

    pair_summary: dict[str, Any] = {}
    for objs, zone in ((fit_objs, "fit"), (mon_objs, "monitor"), (lock_objs, "locked")):
        cache = ab.build_pair_cache(objs)
        _save_pair_cache(cache, pairs_path(zone, tag))
        pair_summary[zone] = ab.pair_cache_summary(cache)

    prep = bp.fit_standardiser(fit_objs)
    prep["protocol"] = PROTOCOL_ID
    prep["split_tag"] = tag
    write_json(prep_path(tag), prep)

    # Fit-only label alignment check (Monitor/Locked .y is never touched here).
    enc = torch.load(ENCODED_TRAIN, map_location="cpu", weights_only=False)
    check_rows = list(split["fit_rows"][:200])
    max_diff = 0.0
    for i in check_rows:
        max_diff = max(max_diff, abs(float(ds[i].y.reshape(-1)[0]) - float(enc[i].y.reshape(-1)[0])))
    del enc
    manifest = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": tag,
        "build_seconds": build_seconds,
        "fit_rows_sha256": split["fit_sha256_rows"],
        "objects_fit_sha256": file_sha256(objects_path("fit", tag)),
        "objects_monitor_sha256": file_sha256(objects_path("monitor", tag)),
        "objects_locked_sha256": file_sha256(objects_path("locked", tag)),
        "pairs_fit_sha256": file_sha256(pairs_path("fit", tag)),
        "pairs_monitor_sha256": file_sha256(pairs_path("monitor", tag)),
        "pairs_locked_sha256": file_sha256(pairs_path("locked", tag)),
        "prep_fit_sha256": file_sha256(prep_path(tag)),
        "pair_summary": pair_summary,
        "alignment_rows_checked": len(check_rows),
        "alignment_rows_scope": "Fit rows only",
        "alignment_max_abs_diff": float(max_diff),
        "forbidden_label_access_attempts": len(guard.access_attempts),
        "guarded_rows": len(guard.forbidden_rows),
        "locked_labels_read": False,
        "monitor_labels_read": False,
    }
    write_json(objects_manifest_path(tag), manifest)
    log(
        f"[objects:{tag}] build {build_seconds:.1f}s alignment max|dy|={max_diff:.2e} "
        f"pairs {pair_summary['fit']['n_pairs']} fit / {pair_summary['monitor']['n_pairs']} monitor"
    )
    return 0


# ---------------------------------------------------------------------------
# init-parity audit (base P3 init must match the historical P round)
# ---------------------------------------------------------------------------


def cmd_stage_a_audit(args: argparse.Namespace) -> int:
    cfg = load_config()
    frozen = cfg["frozen"]
    norm = read_json(prep_path("C"))
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
            "parameter_audit": src.parameter_audit(src.BJOCConfig(), model)["actual_parameters"] == bp.EXPECTED_PARAMETERS,
            "recipe_R3_frozen": bp.RECIPES["R3"] == {
                "epochs": 360,
                "schedule": "cosine",
                "lr": 1.0e-3,
                "lr_min": 1.0e-5,
                "weight_decay": 1.0e-5,
            },
            "batch_order_matches_recipe_round": bp.batch_order_hash(seed, 8000, bp.BATCH_SIZE, 360)
            == f10k.batch_order_hash(seed, 8000, 128, 360),
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
        "per_seed": per_seed,
        "ok": bool(ok),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "stage_a_audit.json", payload)
    log(f"[stage-a-audit] ok={payload['ok']}")
    return 0 if payload["ok"] else 1


def cmd_prep(args: argparse.Namespace) -> int:
    rc = cmd_verify_inputs(args)
    if rc != 0:
        return rc
    args.tags = list(args.tags)
    cmd_splits(args)
    for tag in args.tags:
        args.tag = tag
        cmd_objects(args)
    cmd_freeze(args)
    rc = cmd_stage_a_audit(args)
    if rc != 0:
        return rc
    log("[prep] complete")
    return 0


# ---------------------------------------------------------------------------
# CPU gates
# ---------------------------------------------------------------------------


def _small_real_objects(n: int = 64) -> tuple[list[sro.MolObjects], np.ndarray, list[int]]:
    split = read_json(split_path("C"))
    rows = [int(i) for i in split["fit_rows"][:n]]
    objs = torch.load(objects_path("fit", "C"), map_location="cpu", weights_only=False)[:n]
    y = bp.load_y_rows(ENCODED_TRAIN, rows)
    return objs, y, rows


def _invariance_predictions(model, objs, device, *, model_kind: str, pairs_by_index=None, lambda_override=None) -> np.ndarray:
    def forward(chunk):
        batch = (
            ab.build_dlp_batch(chunk, pairs_by_index)
            if model_kind == "dlp"
            else sro.batch_objects(chunk)
        )
        batch = bp._mv(batch, torch.device(device))
        with torch.no_grad():
            if model_kind == "glrc" and lambda_override is not None:
                return model(batch, lambda_override=lambda_override).detach().cpu().numpy()
            return model(batch).detach().cpu().numpy()

    y_all = forward(list(objs))
    y_single = np.asarray([forward([o])[0] for o in objs])
    return {"batch_membership_max_abs_diff": float(np.abs(y_all - y_single).max()), "batch": y_all}


def _relabel_objects(objs: Sequence[sro.MolObjects], seed: int = 0) -> list[sro.MolObjects]:
    """Atom-order permutation + bond endpoint swap through the raw builder."""
    ds = cf.load_official_train()
    rng = np.random.default_rng(seed)
    out = []
    for o in objs:
        d = ds[int(o.mol_index)]
        x = d.x.numpy()
        ei = d.edge_index.numpy()
        ea = d.edge_attr.numpy()
        perm = rng.permutation(x.shape[0])
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        out.append(sro.build_mol_objects(int(o.mol_index), x[perm], inv[ei], ea))
    return out


def _endpoint_swap_objects(objs: Sequence[sro.MolObjects]) -> list[sro.MolObjects]:
    ds = cf.load_official_train()
    out = []
    for o in objs:
        d = ds[int(o.mol_index)]
        out.append(sro.build_mol_objects(int(o.mol_index), d.x.numpy(), d.edge_index.numpy()[::-1], d.edge_attr.numpy()))
    return out


def cmd_cpu_gate_glrc(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    device = torch.device(args.device)
    objs, y, rows = _small_real_objects(int(args.n_molecules))
    norm = read_json(prep_path("C"))
    cg = load_config()["stage_a_glrc"]["cpu_gate"]

    base, model, info = ab.build_glrc_from_init(0, norm, device, None)
    model.eval()
    base.eval()
    gate: dict[str, Any] = {"info": info}

    # --- identity at lambda = 0 + exact base parameter parity
    batch = bp._mv(sro.batch_objects(objs), device)
    with torch.no_grad():
        y_base = base(batch)
        y_glrc0 = model(batch)
        lam0 = model.lambda_of(batch, next(model.parameters()).dtype)
    gate["identity"] = {
        "forward_max_abs_diff": float((y_base - y_glrc0).abs().max()),
        "lambda_at_init_abs_max": float(lam0.abs().max()),
        "gate_weight_abs_max": float(model.lambda_gate.weight.abs().max()),
        "gate_bias_abs": float(model.lambda_gate.bias.abs().max()),
        "base_parameter_hash_equal": ab.base_parameter_hash(base)
        == ab.base_parameter_hash(model, exclude_prefixes=("lambda_gate.",)),
        "base_init_hash_equal_p3": info["base_scratch_init_full_sha256"] == src.state_dict_hash(base),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "new_parameters": info["new_parameters"],
    }

    # --- gate gradient activity + real use after optimizer steps
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
    before = float(model(batch).detach().mean())
    for _ in range(2):
        opt.zero_grad()
        loss = torch.nn.functional.l1_loss(model(batch).view(-1), y_t.view(-1))
        loss.backward()
        gate_grads = [p.grad for p in (model.lambda_gate.weight, model.lambda_gate.bias)]
        opt.step()
    finite = all(g is not None and torch.isfinite(g).all() for g in gate_grads)
    gate["optimization"] = {
        "gate_grad_finite": bool(finite),
        "gate_grad_abs_max": float(max(float(g.abs().max()) for g in gate_grads if g is not None)),
        "gate_grad_nonzero": bool(all(float((g != 0).float().sum()) > 0 for g in gate_grads if g is not None)),
        "lambda_abs_max_after_2_steps": float(model.lambda_of(batch, next(model.parameters()).dtype).abs().max()),
        "pred_mean_before": before,
    }
    model.eval()
    with torch.no_grad():
        y_active = model(batch)
        y_zero = model(batch, lambda_override=0.0)
    gate["optimization"]["active_vs_zero_lambda_max_abs_diff"] = float((y_active - y_zero).abs().max())

    # --- conservation of the local residual per molecule
    feats = model.forward_features(batch, lambda_override=0.5)
    m = batch["bond_batch"]
    dw_sum = torch.zeros(int(batch["n_graphs"]), device=device)
    dv_sum = torch.zeros_like(dw_sum)
    for g in range(int(batch["n_graphs"])):
        mm = m == g
        dw_sum[g] = feats["dW"][mm].sum(0).abs().max()
        dv_sum[g] = feats["dV"][mm].sum(0).abs().max()
    gate["conservation"] = {
        "n_molecules": int(batch["n_graphs"]),
        "sum_dW_max_abs": float(dw_sum.max()),
        "sum_dV_max_abs": float(dv_sum.max()),
        "tolerance": 1.0e-3,
    }

    # --- controlled ownership counterexample (marginal-preserving switch)
    inst_a, inst_b = src.counterexample_object_pair()
    assert src.object_marginal_signature(inst_a) == src.object_marginal_signature(inst_b)
    cap = 64
    obj_batch = bp._mv(src.batch_object_instances([inst_a, inst_b]), device)
    with torch.no_grad():
        base_syn = base(obj_batch)
        glrc0_syn = model(obj_batch, lambda_override=0.0)
        glrc1_syn = model(obj_batch, lambda_override=1.0)
        feats_syn = model.forward_features(obj_batch, lambda_override=1.0)
        local_feature_diff = float((feats_syn["dW"][:4] - feats_syn["dW"][4:]).abs().max())
        # null features: re-evaluate with lambda forced to 0
        feats_null = model.forward_features(obj_batch, lambda_override=0.0)
        null_w_identical = float((feats_null["W"][:4] - feats_null["W"][4:]).abs().max())
    gate["ownership_counterexample"] = {
        "marginal_signatures_equal": True,
        "per_bond_token_fingerprints_differ": src.per_bond_junction_tokens(inst_a) != src.per_bond_junction_tokens(inst_b),
        "base_pred_abs_diff": float(abs(base_syn[0] - base_syn[1])),
        "glrc_lambda0_pred_abs_diff": float(abs(glrc0_syn[0] - glrc0_syn[1])),
        "glrc_lambda1_pred_abs_diff": float(abs(glrc1_syn[0] - glrc1_syn[1])),
        "null_W_abs_diff": null_w_identical,
        "local_residual_abs_diff": local_feature_diff,
        "cap_note": f"synthetic instances use <= {cap} atoms; predictions finite: {bool(torch.isfinite(glrc1_syn).all())}",
    }

    # --- invariance (gate active so the local path participates)
    perm_objs = _relabel_objects(objs[:4])
    swap_objs = _endpoint_swap_objects(objs[:4])
    inv = _invariance_predictions(model, objs[:4], device, model_kind="glrc", lambda_override=0.5)
    inv_perm = _invariance_predictions(model, perm_objs, device, model_kind="glrc", lambda_override=0.5)
    inv_swap = _invariance_predictions(model, swap_objs, device, model_kind="glrc", lambda_override=0.5)
    # junction row permutation: valid permutation of (junc_index, junc_bond) rows
    rng = np.random.default_rng(0)
    def junc_permuted(chunk):
        out = []
        for o in chunk:
            perm = rng.permutation(o.junc_index.shape[0])
            new = sro.MolObjects(
                mol_index=o.mol_index, smiles=o.smiles, n_atoms=o.n_atoms,
                atom_feat=o.atom_feat, bond_index=o.bond_index, bond_type=o.bond_type, bond_feat=o.bond_feat,
                junc_index=o.junc_index[perm], junc_bond=o.junc_bond[perm], junc_feat=o.junc_feat[perm],
                global_feat=o.global_feat,
            )
            out.append(new)
        return out
    inv_junc = _invariance_predictions(model, junc_permuted(list(objs[:4])), device, model_kind="glrc", lambda_override=0.5)
    gate["invariance"] = {
        "atom_relabel_max_abs_diff": float(np.abs(inv["batch"] - inv_perm["batch"]).max()),
        "bond_endpoint_swap_max_abs_diff": float(np.abs(inv["batch"] - inv_swap["batch"]).max()),
        "junction_row_permutation_max_abs_diff": float(np.abs(inv["batch"] - inv_junc["batch"]).max()),
        "batch_membership_max_abs_diff": inv["batch_membership_max_abs_diff"],
        "tolerance": cg["invariance_tolerance"],
    }

    gate["forbidden_modules"] = ab.forbidden_module_scan_arch(model)
    gate["no_label_features"] = True
    conditions = {
        "identity": gate["identity"]["forward_max_abs_diff"] <= cg["identity_max_abs_diff"]
        and gate["identity"]["lambda_at_init_abs_max"] == 0.0
        and gate["identity"]["base_parameter_hash_equal"]
        and gate["identity"]["base_init_hash_equal_p3"],
        "gradient_activity": gate["optimization"]["gate_grad_finite"]
        and gate["optimization"]["gate_grad_nonzero"]
        and gate["optimization"]["lambda_abs_max_after_2_steps"] > 0.0,
        "local_path_used": gate["optimization"]["active_vs_zero_lambda_max_abs_diff"] > 0.0,
        "conservation": gate["conservation"]["sum_dW_max_abs"] <= cg["conservation_tolerance"]
        and gate["conservation"]["sum_dV_max_abs"] <= cg["conservation_tolerance"],
        "ownership_counterexample": gate["ownership_counterexample"]["base_pred_abs_diff"] <= cg["counterexample_base_tolerance"]
        and gate["ownership_counterexample"]["glrc_lambda0_pred_abs_diff"] <= cg["counterexample_base_tolerance"]
        and gate["ownership_counterexample"]["glrc_lambda1_pred_abs_diff"] > 0.0
        and gate["ownership_counterexample"]["local_residual_abs_diff"] > cg["counterexample_local_min"]
        and gate["ownership_counterexample"]["per_bond_token_fingerprints_differ"],
        "invariance": max(
            gate["invariance"]["atom_relabel_max_abs_diff"],
            gate["invariance"]["bond_endpoint_swap_max_abs_diff"],
            gate["invariance"]["junction_row_permutation_max_abs_diff"],
            gate["invariance"]["batch_membership_max_abs_diff"],
        )
        <= cg["invariance_tolerance"],
        "parameter_budget": gate["identity"]["parameter_count"] == ab.GLRC_PARAMETERS,
        "forbidden_modules": len(gate["forbidden_modules"]) == 0,
    }
    gate["conditions"] = conditions
    gate["decision"] = "GLRC_CPU_GATE_PASS" if all(conditions.values()) else "GLRC_CPU_BLOCKED"
    write_json(OUT / "cpu_gate_glrc.json", gate)
    log(f"[cpu-gate-glrc] {gate['decision']}")
    for k, v in conditions.items():
        log(f"  {k:22s} {'PASS' if v else 'FAIL'}")
    return 0 if gate["decision"] == "GLRC_CPU_GATE_PASS" else 1


def cmd_cpu_gate_dlp(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    device = torch.device(args.device)
    n = int(args.n_molecules)
    objs, y, rows = _small_real_objects(n)
    pairs = load_pair_cache(pairs_path("fit", "C"))
    norm = read_json(prep_path("C"))
    cg = load_config()["stage_b_dlp"]["cpu_gate"]
    gate: dict[str, Any] = {}

    # --- pair construction brute-force cross-check
    check = []
    for o in objs[:8]:
        rec = ab.pair_arrays_for_object(o)
        n_atoms = int(o.n_atoms)
        adj: list[list[tuple[int, int]]] = [[] for _ in range(n_atoms)]
        for (u, v), bt in zip(np.asarray(o.bond_index), np.asarray(o.bond_type)):
            adj[int(u)].append((int(v), int(bt)))
            adj[int(v)].append((int(u), int(bt)))
        dist = sro._all_pairs_distances(n_atoms, adj)
        expected = []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                d = int(dist[i, j])
                if d >= 3:
                    expected.append((i, j, 0 if d == 3 else (1 if d == 4 else 2)))
        got = list(zip(rec["i"].tolist(), rec["j"].tolist(), rec["bucket"].tolist()))
        refetched = pairs[int(o.mol_index)]
        if got != expected:
            raise SystemExit("pair brute-force cross-check failed")
        if got != list(zip(refetched["i"].tolist(), refetched["j"].tolist(), refetched["bucket"].tolist())):
            raise SystemExit("cached pair arrays differ from fresh build")
        check.append({"mol": int(o.mol_index), "n_pairs": len(got), "ok": True})
    witness = ab.pair_feature_partition_witness()
    gate["information_witness"] = witness

    base, model, info = ab.build_dlp_from_init(0, norm, device, None)
    base.eval()
    model.eval()
    gate["info"] = info
    dbatch = bp._mv(ab.build_dlp_batch(objs, pairs), device)
    with torch.no_grad():
        y_base = base(dbatch)
        y_dlp0 = model(dbatch)
        feats = model.forward_features(dbatch)
    gate["identity"] = {
        "forward_max_abs_diff": float((y_base - y_dlp0).abs().max()),
        "new_reader_columns_abs_max": float(model.reader.lins[0].weight[:, _BASE_READER_IN:].abs().max()),
        "base_parameter_hash_equal": ab.base_parameter_hash(base, exclude_prefixes=("reader.lins.0.weight",))
        == ab.base_parameter_hash(model, exclude_prefixes=("pair_proj.", "reader.lins.0.weight")),
        "old_reader_columns_copied": bool(
            torch.equal(
                model.reader.lins[0].weight[:, :_BASE_READER_IN],
                base.reader.lins[0].weight,
            )
        ),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "new_parameters": info["new_parameters"],
    }
    dlp = feats["dlp"]
    gate["nondegenerate"] = {
        "finite": bool(torch.isfinite(dlp).all()),
        "dim_std_median": float(dlp.std(0).median()),
        "zero_std_fraction": float((dlp.std(0) < 1e-6).float().mean()),
    }

    # --- gradient: reader columns first step, pair projection from the second
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
    bp_step = {}
    for step in range(2):
        opt.zero_grad()
        loss = torch.nn.functional.l1_loss(model(dbatch).view(-1), y_t.view(-1))
        loss.backward()
        g_new = model.reader.lins[0].weight.grad[:, _BASE_READER_IN:]
        g_p = model.pair_proj.weight.grad
        bp_step[step] = {
            "reader_new_columns_grad_nonzero_frac": float((g_new != 0).float().mean()),
            "pair_proj_grad_nonzero_frac": float((g_p != 0).float().mean()),
            "pair_proj_grad_finite": bool(torch.isfinite(g_p).all()),
        }
        opt.step()
    model.eval()
    with torch.no_grad():
        y_after = model(dbatch)
    gate["gradient"] = {
        "step0": bp_step[0],
        "step1": bp_step[1],
        "prediction_change_after_2_steps": float((y_after - y_dlp0).abs().max()),
    }

    # --- invariance
    perm_objs = _relabel_objects(objs[:4])
    swap_objs = _endpoint_swap_objects(objs[:4])
    inv = _invariance_predictions(model, objs[:4], device, model_kind="dlp", pairs_by_index=pairs)
    inv_perm = _invariance_predictions(model, perm_objs, device, model_kind="dlp", pairs_by_index=pairs)
    inv_swap = _invariance_predictions(model, swap_objs, device, model_kind="dlp", pairs_by_index=pairs)
    # pair row permutation inside one molecule
    flipped = {}
    for k, v in pairs.items():
        flipped[k] = {"i": v["i"][::-1].copy(), "j": v["j"][::-1].copy(), "bucket": v["bucket"][::-1].copy()}
    inv_pair = _invariance_predictions(model, objs[:4], device, model_kind="dlp", pairs_by_index=flipped)
    gate["invariance"] = {
        "atom_relabel_max_abs_diff": float(np.abs(inv["batch"] - inv_perm["batch"]).max()),
        "bond_endpoint_swap_max_abs_diff": float(np.abs(inv["batch"] - inv_swap["batch"]).max()),
        "pair_row_order_max_abs_diff": float(np.abs(inv["batch"] - inv_pair["batch"]).max()),
        "batch_membership_max_abs_diff": inv["batch_membership_max_abs_diff"],
        "tolerance": cg["invariance_tolerance"],
    }

    # --- linear recoverability of the DLP block from the frozen reader input
    probe_n = min(len(objs), int(args.probe_molecules))
    X, P = [], []
    with torch.no_grad():
        for i in range(0, probe_n, 256):
            chunk = objs[i : i + 256]
            b = bp._mv(ab.build_dlp_batch(chunk, pairs), device)
            feats = model.forward_features(b)
            co = int(src.BJOCConfig().compose_out)
            ao = int(src.BJOCConfig().atom_out)
            base_reader_in = feats["reader_in"][:, : co + 2 * ao + sro.GLOBAL_DIM]
            X.append(base_reader_in.detach().cpu().numpy())
            P.append(feats["dlp"].detach().cpu().numpy())
    gate["recoverability"] = ab.linear_recoverability(np.concatenate(X, 0), np.concatenate(P, 0), seed=0)

    gate["forbidden_modules"] = ab.forbidden_module_scan_arch(model)
    conditions = {
        "pair_construction": bool(check) and all(c["ok"] for c in check),
        "information_witness": witness["different_pair_features"] and witness["first_second_moments_equal"],
        "identity": gate["identity"]["forward_max_abs_diff"] <= cg["identity_max_abs_diff"]
        and gate["identity"]["new_reader_columns_abs_max"] == 0.0
        and gate["identity"]["base_parameter_hash_equal"],
        "nondegenerate": gate["nondegenerate"]["finite"] and gate["nondegenerate"]["zero_std_fraction"] <= cg["zero_std_fraction_max"],
        "gradient_reader_first_step": gate["gradient"]["step0"]["reader_new_columns_grad_nonzero_frac"] > 0.0,
        "gradient_pair_projection_second_step": gate["gradient"]["step1"]["pair_proj_grad_nonzero_frac"] > 0.0
        and gate["gradient"]["step1"]["pair_proj_grad_finite"],
        "prediction_change": gate["gradient"]["prediction_change_after_2_steps"] > 0.0,
        "recoverability_not_complete": gate["recoverability"]["test_r2_median"] <= cg["recoverability_r2_median_max"],
        "invariance": max(
            gate["invariance"]["atom_relabel_max_abs_diff"],
            gate["invariance"]["bond_endpoint_swap_max_abs_diff"],
            gate["invariance"]["pair_row_order_max_abs_diff"],
            gate["invariance"]["batch_membership_max_abs_diff"],
        )
        <= cg["invariance_tolerance"],
        "parameter_budget": gate["identity"]["parameter_count"] == ab.DLP_PARAMETERS,
        "forbidden_modules": len(gate["forbidden_modules"]) == 0,
    }
    gate["conditions"] = conditions
    gate["decision"] = "DLP_CPU_GATE_PASS" if all(conditions.values()) else "DLP_CPU_BLOCKED"
    write_json(OUT / "cpu_gate_dlp.json", gate)
    log(f"[cpu-gate-dlp] {gate['decision']}")
    for k, v in conditions.items():
        log(f"  {k:34s} {'PASS' if v else 'FAIL'}")
    log(
        f"  recoverability test R2 median {gate['recoverability']['test_r2_median']:.4f} "
        f"residual {gate['recoverability']['centered_relative_residual']:.4f}"
    )
    return 0 if gate["decision"] == "DLP_CPU_GATE_PASS" else 1


# ---------------------------------------------------------------------------
# SSL + supervised arms
# ---------------------------------------------------------------------------


def _guard_for_tag(tag: str):
    split = read_json(split_path(tag))
    ds, guard = _guarded_dataset(split["monitor_rows"] + split["locked_rows"])
    return split, ds, guard


def cmd_ssl(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=True)
    tag = args.tag
    seed = int(args.seed)
    ssl_pt, ssl_json = ssl_paths(tag, seed)
    if args.smoke_updates:
        ssl_pt = OUT / "smoke" / f"{ssl_pt.stem}_smoke.pt"
        ssl_json = OUT / "smoke" / f"{ssl_json.stem}_smoke.json"
    if ssl_pt.exists() or ssl_json.exists():
        raise SystemExit(f"SSL {tag} seed {seed} already exists; retraining is not allowed")
    split, ds, guard = _guard_for_tag(tag)
    fit_objs = torch.load(objects_path("fit", tag), map_location="cpu", weights_only=False)
    if [int(o.mol_index) for o in fit_objs] != [int(i) for i in split["fit_rows"]]:
        raise SystemExit("fit object rows do not match the frozen split")
    payload = bp.pretrain_masked(
        seed=seed,
        fit_objs=fit_objs,
        fit_rows=split["fit_rows"],
        raw_dataset=guard,
        device=torch.device(args.device),
        mode="atom_bond",
        smoke_updates=args.smoke_updates,
    )
    ssl_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload["encoder_state"], ssl_pt)
    payload["encoder_path"] = str(ssl_pt.relative_to(REPO_ROOT))
    payload["encoder_file_sha256"] = file_sha256(ssl_pt)
    payload["split_tag"] = tag
    payload["forbidden_label_access_attempts"] = len(guard.access_attempts)
    payload.pop("encoder_state", None)
    write_json(ssl_json, payload)
    log(f"[ssl {tag} s{seed}] final acc atom {payload['final_acc_atom']:.4f} wall {payload['wall_seconds']:.0f}s")
    return 0


def cmd_arm(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=True)
    tag = args.tag
    arm = args.arm
    seed = int(args.seed)
    kind = ab.ARM_KINDS[arm]
    split = read_json(split_path(tag))
    fit_objs = torch.load(objects_path("fit", tag), map_location="cpu", weights_only=False)
    if [int(o.mol_index) for o in fit_objs] != [int(i) for i in split["fit_rows"]]:
        raise SystemExit("fit object rows do not match the frozen split")
    norm = read_json(prep_path(tag))
    y_fit = bp.load_y_rows(ENCODED_TRAIN, split["fit_rows"])
    ssl_pt, ssl_json = ssl_paths(tag, seed)
    if not ssl_pt.exists():
        raise SystemExit(f"missing SSL checkpoint {ssl_pt}")
    ssl_state = torch.load(ssl_pt, map_location="cpu", weights_only=False)
    ssl_recorded = read_json(ssl_json)["encoder_final_sha256"]

    pairs = None
    if kind == "dlp":
        pairs = load_pair_cache(pairs_path("fit", tag))

    soup = soup_path(tag, arm, seed)
    tj = train_json_path(tag, arm, seed)
    smoke_epochs = args.smoke_epochs
    if smoke_epochs:
        soup = OUT / "smoke" / f"{soup.stem}_smoke.pt"
        tj = OUT / "smoke" / f"{tj.stem}_smoke.json"
    if soup.exists() or tj.exists():
        raise SystemExit(f"{arm} {tag} seed {seed} already exists; retraining is not allowed")

    if kind == "base":
        payload = bp.train_supervised(
            arm=arm,
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
    else:
        payload = ab.train_supervised_arch(
            arm=arm,
            seed=seed,
            recipe_key="R3",
            init="ssl",
            train_objs=fit_objs,
            rows=split["fit_rows"],
            y_fit=y_fit,
            norm=norm,
            device=torch.device(args.device),
            model_kind=kind,
            ssl_state=ssl_state,
            pairs_by_index=pairs,
            smoke_epochs=smoke_epochs,
        )
    if payload["encoder_used_sha256"] != ssl_recorded:
        raise SystemExit("encoder used does not match the recorded pretraining state")
    soup.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu().clone() for k, v in payload["soup_state"].items()}, soup)
    payload.pop("soup_state", None)
    payload["soup_path"] = str(soup.relative_to(REPO_ROOT))
    payload["soup_sha256"] = file_sha256(soup)
    payload["split_tag"] = tag
    write_json(tj, payload)
    log(f"[{arm} {tag} s{seed}] soup fit {payload['soup_fit_mae']:.5f} wall {payload['wall_seconds']:.0f}s")
    return 0


# ---------------------------------------------------------------------------
# label-blind predictions + freezes
# ---------------------------------------------------------------------------


def _load_arm_model(tag: str, arm: str, seed: int, device: torch.device):
    kind = ab.ARM_KINDS[arm]
    norm = read_json(prep_path(tag))
    _, model, _ = ab.build_arm_model(arm, seed, norm, device, None)
    model.load_state_dict(torch.load(soup_path(tag, arm, seed), map_location=device, weights_only=False))
    return model, kind


def cmd_predict(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    tag = args.tag
    paths = {
        "monitor": torch.load(objects_path("monitor", tag), map_location="cpu", weights_only=False),
        "locked": torch.load(objects_path("locked", tag), map_location="cpu", weights_only=False),
    }
    pairs = {zone: load_pair_cache(pairs_path(zone, tag)) for zone in paths}
    device = torch.device(args.device)
    arms = list(args.arms) if args.arms else ["P3", "GLRC"]
    seeds = [int(s) for s in args.seeds]
    smoke = bool(args.smoke)
    manifest = read_json(PRED_MANIFEST[tag]) if PRED_MANIFEST[tag].exists() else {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": tag,
        "entries": {},
        "monitor_labels_read": False,
        "locked_labels_read": False,
        "official_test_loaded": False,
    }
    if smoke:
        arms = ["P3", "GLRC"]
        seeds = [0]
    for arm in arms:
        for seed in seeds:
            s = soup_path(tag, arm, seed)
            if smoke:
                s = OUT / "smoke" / f"soup_{arm}_{tag}_s{seed}_smoke.pt"
            if not s.exists():
                raise SystemExit(f"missing soup {s}")
            model, kind = _load_arm_model(tag, arm, seed, device)
            model.load_state_dict(torch.load(s, map_location=device, weights_only=False))
            for zone, objs in paths.items():
                out = pred_path(zone, tag, arm, seed)
                if smoke:
                    out = OUT / "smoke" / f"{zone}_{arm}_{tag}_s{seed}_smoke.npz"
                key = f"{zone}:{arm}:{seed}"
                pred = ab.predict_model_arch(model, objs, device, model_kind=kind, pairs_by_index=pairs[zone])
                if not np.isfinite(pred).all():
                    raise SystemExit(f"non-finite {zone} prediction {arm} {seed}")
                if out.exists():
                    existing = manifest["entries"].get(key)
                    if existing and existing["sha256"] == file_sha256(out):
                        log(f"[predict {tag}] {zone} {arm} s{seed} already frozen")
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
                log(f"[predict {tag}] {zone} {arm} s{seed} -> {out.name}")
    if not smoke:
        manifest["monitor_labels_read"] = bool(MONITOR_LOCK[tag].exists())
        manifest["locked_labels_read"] = bool(LOCKED_LOCK[tag].exists())
        write_json(PRED_MANIFEST[tag], manifest)
    return 0


def _verify_prediction_manifest(tag: str) -> dict[str, Any]:
    manifest = read_json(PRED_MANIFEST[tag])
    for key, rec in manifest["entries"].items():
        p = REPO_ROOT / rec["path"]
        if not p.exists() or file_sha256(p) != rec["sha256"]:
            raise SystemExit(f"prediction drift for {key}")
    return manifest


def _read_npz_pred(path: Path, n: int) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    pred = np.asarray(z["pred"], dtype=np.float64).reshape(-1)
    row = np.asarray(z["row"], dtype=np.int64).reshape(-1)
    if len(pred) != n or not np.array_equal(row, np.arange(n)):
        raise SystemExit(f"row/pred shape mismatch in {path}")
    return pred


def _ensure_monitor_y(tag: str) -> np.ndarray:
    split = read_json(split_path(tag))
    if MONITOR_LOCK[tag].exists():
        y = np.asarray(np.load(MONITOR_Y[tag]), dtype=np.float64).reshape(-1)
        lock = read_json(MONITOR_LOCK[tag])
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
            "prediction_manifest_sha256": file_sha256(PRED_MANIFEST[tag]),
        },
    )
    log(f"[monitor-read {tag}] n={len(y)} sha={bp.array_sha256(y)[:12]}")
    return y


def _read_locked_y_once(tag: str) -> np.ndarray:
    split = read_json(split_path(tag))
    if LOCKED_LOCK[tag].exists():
        raise SystemExit(f"Locked labels already read for split {tag}")
    rows = split["locked_rows"]
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
            "locked_manifest_sha256": file_sha256(LOCKED_MANIFEST[tag]),
        },
    )
    log(f"[locked-read {tag}] n={len(y)} sha={bp.array_sha256(y)[:12]}")
    return y


# ---------------------------------------------------------------------------
# Monitor scoring (declared rounds)
# ---------------------------------------------------------------------------


def _load_preds(tag: str, arm: str, seeds: Sequence[int], n: int) -> dict[int, np.ndarray]:
    return {int(s): _read_npz_pred(pred_path("monitor", tag, arm, int(s)), n) for s in seeds}


def cmd_score_monitor(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    tag = args.tag
    candidate = args.candidate
    if candidate not in CANDIDATE_ARMS:
        raise SystemExit(f"unknown candidate {candidate!r}")
    manifest = _verify_prediction_manifest(tag)
    y = _ensure_monitor_y(tag)
    n = len(y)
    seeds = sorted(
        {int(key.split(":")[2]) for key in manifest["entries"] if key.startswith("monitor:P3:")}
        & {int(key.split(":")[2]) for key in manifest["entries"] if key.startswith(f"monitor:{candidate}:")}
    )
    if not seeds:
        raise SystemExit("no shared seeds between P3 and the candidate")
    p3 = _load_preds(tag, "P3", seeds, n)
    cand = _load_preds(tag, candidate, seeds, n)
    two_seed = len(seeds) >= 2
    contrast = ab.compute_contrast(cand, p3, y)
    gate = ab.monitor_gate(contrast, two_seed=two_seed)
    if gate["passed"]:
        decision = f"{candidate}_MONITOR_GAIN_SUPPORTED"
    elif contrast["delta_seed_mean"] < 0:
        decision = f"{candidate}_WEAK_OR_MIXED"
    else:
        decision = f"{candidate}_NO_GAIN"
    payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": tag,
        "candidate": candidate,
        "seeds": seeds,
        "two_seed": two_seed,
        "contrast_vs_P3": contrast,
        "gate": gate,
        "decision": decision,
        "prediction_manifest_sha256": file_sha256(PRED_MANIFEST[tag]),
        "monitor_read_utc": read_json(MONITOR_LOCK[tag])["read_utc"] if MONITOR_LOCK[tag].exists() else None,
        "official_test_loaded": False,
    }
    if candidate == "ORGANIZED":
        # mechanism contrast against the GLRC prediction set (same seeds)
        glrc_seeds = sorted(
            {int(k.split(":")[2]) for k in manifest["entries"] if k.startswith("monitor:GLRC:")} & set(seeds)
        )
        if glrc_seeds:
            glrc = _load_preds(tag, "GLRC", glrc_seeds, n)
            payload["contrast_vs_GLRC"] = ab.compute_contrast(cand, glrc, y)
    write_json(OUT / f"monitor_{candidate}_{tag}.json", payload)
    rounds = read_json(MONITOR_ROUNDS[tag]) if MONITOR_ROUNDS[tag].exists() else {"protocol": PROTOCOL_ID, "tag": tag, "rounds": []}
    rounds["rounds"].append(
        {
            "round": len(rounds["rounds"]) + 1,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "candidate": candidate,
            "seeds": seeds,
            "decision": decision,
            "delta_seed_mean": contrast["delta_seed_mean"],
            "prediction_manifest_sha256": payload["prediction_manifest_sha256"],
        }
    )
    rounds["n_selection_rounds"] = len(rounds["rounds"])
    write_json(MONITOR_ROUNDS[tag], rounds)
    log(f"[score-monitor {tag}] {candidate}: {decision} delta {contrast['delta_seed_mean']:+.5f}")
    for key, c in (("vs P3", contrast),):
        log(
            f"  {key}: CI95 [{c['bootstrap']['ci_lo']:+.5f},{c['bootstrap']['ci_hi']:+.5f}] "
            f"per-seed {c['per_seed_delta']} ordinary {c['ordinary_delta']:+.5f} "
            f"top10 {c['top10_removed_delta']:+.5f} rmse {c['rmse_new']-c['rmse_ref']:+.5f} "
            f"single-share {c['max_single_gain_fraction']:.3f}"
        )
    if "contrast_vs_GLRC" in payload:
        g = payload["contrast_vs_GLRC"]
        log(f"  vs GLRC: delta {g['delta_seed_mean']:+.5f} CI95 [{g['bootstrap']['ci_lo']:+.5f},{g['bootstrap']['ci_hi']:+.5f}]")
    return 0


# ---------------------------------------------------------------------------
# Locked final
# ---------------------------------------------------------------------------


def _final_arms(tag: str) -> list[str]:
    manifest = _verify_prediction_manifest(tag)
    present = {key.split(":")[1] for key in manifest["entries"] if key.startswith("monitor:")}
    arms = ["P3"] + [a for a in CANDIDATE_ARMS if a in present]
    return arms


def cmd_locked_freeze(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    tag = args.tag
    if LOCKED_LOCK[tag].exists():
        raise SystemExit("Locked labels already read; refusing to refreeze")
    manifest = _verify_prediction_manifest(tag)
    arms = _final_arms(tag)
    entries: dict[str, Any] = {}
    arm_seeds: dict[str, list[int]] = {}
    for arm in arms:
        seeds = sorted(
            {int(k.split(":")[2]) for k in manifest["entries"] if k.startswith(f"locked:{arm}:")}
        )
        if not seeds:
            raise SystemExit(f"missing locked predictions for {arm}; run predict first")
        arm_seeds[arm] = seeds
        for seed in seeds:
            key = f"locked:{arm}:{seed}"
            if key not in manifest["entries"]:
                raise SystemExit(f"missing locked prediction {key}; run predict first")
            entries[key] = manifest["entries"][key]
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": tag,
        "arms": arms,
        "arm_seeds": arm_seeds,
        "entries": entries,
        "monitor_rounds_sha256": file_sha256(MONITOR_ROUNDS[tag]) if MONITOR_ROUNDS[tag].exists() else None,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_test_loaded": False,
    }
    write_json(LOCKED_MANIFEST[tag], payload)
    log(f"[locked-freeze {tag}] arms {arms}")
    return 0


def cmd_locked_score(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    tag = args.tag
    locked_manifest = read_json(LOCKED_MANIFEST[tag])
    for key, rec in locked_manifest["entries"].items():
        p = REPO_ROOT / rec["path"]
        if not p.exists() or file_sha256(p) != rec["sha256"]:
            raise SystemExit(f"locked prediction drift for {key}")
    y = _read_locked_y_once(tag)
    arms = locked_manifest["arms"]
    arm_seeds = locked_manifest.get("arm_seeds") or {arm: list(SEEDS) for arm in arms}
    preds: dict[str, dict[int, np.ndarray]] = {}
    for arm in arms:
        preds[arm] = {s: _read_npz_pred(pred_path("locked", tag, arm, s), len(y)) for s in arm_seeds[arm]}
    metrics = {}
    for arm in arms:
        seeds = arm_seeds[arm]
        per_seed = {str(s): float(np.abs(preds[arm][s] - y).mean()) for s in seeds}
        m = np.mean([np.abs(preds[arm][s] - y) for s in seeds], axis=0)
        metrics[arm] = {
            "seeds": seeds,
            "per_seed_mae": per_seed,
            "seed_mean_mae": float(m.mean()),
            "median_abs": float(np.median(m)),
            "p90_abs": float(np.quantile(m, 0.90)),
            "rmse": float(np.sqrt(np.mean(np.mean([(preds[arm][s] - y) ** 2 for s in seeds], axis=0)))),
        }
    contrasts: dict[str, Any] = {}
    for candidate in [a for a in arms if a != "P3"]:
        shared = {s: preds["P3"][s] for s in arm_seeds[candidate]}
        contrasts[f"{candidate}_minus_P3"] = ab.compute_contrast(preds[candidate], shared, y)
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
        log(f"  {key:16s} delta {c['delta_seed_mean']:+.5f} CI95 [{c['bootstrap']['ci_lo']:+.5f},{c['bootstrap']['ci_hi']:+.5f}]")
    return 0


# ---------------------------------------------------------------------------
# conditional full-10k refit (exposed official valid, descriptive only)
# ---------------------------------------------------------------------------


def full10k_paths(arm: str, seed: int) -> tuple[Path, Path, Path]:
    soup = OUT / f"soup_{arm}_full10k_s{seed}.pt"
    tj = OUT / f"train_{arm}_full10k_s{seed}.json"
    pred = OUT / "predictions" / f"valid_{arm}_full10k_s{seed}.npz"
    return soup, tj, pred


def cmd_full10k_train(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=True)
    arm = args.arm
    seed = int(args.seed)
    if arm not in ("GLRC", "DLP"):
        raise SystemExit("full-10k refit is only defined for the two candidates")
    kind = ab.ARM_KINDS[arm]
    soup, tj, pred_path_out = full10k_paths(arm, seed)
    if soup.exists() or tj.exists():
        raise SystemExit(f"full-10k {arm} seed {seed} already exists")
    norm = read_json(F10K / "prep_train10k.json")
    train_objs = f10k.load_objects(F10K / "objects_train10k.pt")
    ssl_path = F10K / f"pretrain_encoder_s{seed}.pt"
    ssl_state = torch.load(ssl_path, map_location="cpu", weights_only=False)
    pairs = None
    if kind == "dlp":
        # the full-10k pair cache is built once, from the same 10k object list
        cache_path = OUT / "pairs_full10k.pt"
        if not cache_path.exists():
            _save_pair_cache(ab.build_pair_cache(train_objs), cache_path)
        pairs = load_pair_cache(cache_path)
    payload = ab.train_supervised_arch(
        arm=arm,
        seed=seed,
        recipe_key="R3",
        init="ssl",
        train_objs=train_objs,
        rows=list(range(N_TRAIN)),
        y_fit=bp.load_y_rows(ENCODED_TRAIN, list(range(N_TRAIN))),
        norm=norm,
        device=torch.device(args.device),
        model_kind=kind,
        ssl_state=ssl_state,
        pairs_by_index=pairs,
    )
    torch.save({k: v.detach().cpu().clone() for k, v in payload["soup_state"].items()}, soup)
    payload.pop("soup_state", None)
    payload["soup_path"] = str(soup.relative_to(REPO_ROOT))
    payload["soup_sha256"] = file_sha256(soup)
    payload["ssl_source"] = str(ssl_path.relative_to(REPO_ROOT))
    payload["ssl_source_sha256"] = file_sha256(ssl_path)
    write_json(tj, payload)
    # descriptive official-valid forward (exposed development split)
    valid_objs = f10k.load_objects(F10K / "objects_valid.pt")
    _, model, _ = ab.build_arm_model(arm, seed, norm, torch.device(args.device), None)
    model.load_state_dict(torch.load(soup, map_location=torch.device(args.device), weights_only=False))
    preds = ab.predict_model_arch(model, valid_objs, torch.device(args.device), model_kind=kind, pairs_by_index=pairs)
    pred_path_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path_out, row=np.arange(N_VALID, dtype=np.int64), pred=preds.astype(np.float32))
    log(f"[full10k {arm} s{seed}] soup fit {payload['soup_fit_mae']:.5f}; valid forward written")
    return 0


def cmd_full10k_score(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    y_path = F10K / "valid_y_read_once.npy"
    y = np.asarray(np.load(y_path), dtype=np.float64).reshape(-1)
    refs = {s: _read_npz_pred(REC / "predictions" / f"valid_R3{s}.npz", N_VALID) for s in SEEDS}
    payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "note": "official valid is exposed; descriptive only; never used for selection",
        "reference": "historical full-10k P3 (recipe round) official-valid predictions",
        "arms": {},
        "official_test_loaded": False,
    }
    for arm in ("GLRC", "DLP"):
        preds = {}
        for s in SEEDS:
            _, _, p = full10k_paths(arm, s)
            if p.exists():
                preds[s] = _read_npz_pred(p, N_VALID)
        if not preds:
            continue
        per_seed = {str(s): float(np.abs(preds[s] - y).mean()) for s in preds}
        ref = {s: refs[s] for s in preds}
        payload["arms"][arm] = {
            "per_seed_mae": per_seed,
            "contrast_vs_historical_P3": ab.compute_contrast(preds, ref, y),
        }
    write_json(FULL10K_SCORE, payload)
    log("[full10k-score] written")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prep")
    prep.add_argument("--tags", nargs="*", default=["C"], choices=sorted(SPLIT_TAGS))

    s = sub.add_parser("splits")
    s.add_argument("--tags", nargs="*", default=["C", "D"], choices=sorted(SPLIT_TAGS))

    o = sub.add_parser("objects")
    o.add_argument("--tag", default="C", choices=sorted(SPLIT_TAGS))

    sub.add_parser("freeze")
    sub.add_parser("verify-inputs")
    sub.add_parser("stage-a-audit")

    g = sub.add_parser("cpu-gate-glrc")
    g.add_argument("--device", default="cpu")
    g.add_argument("--n-molecules", type=int, default=64)

    g = sub.add_parser("cpu-gate-dlp")
    g.add_argument("--device", default="cpu")
    g.add_argument("--n-molecules", type=int, default=256)
    g.add_argument("--probe-molecules", type=int, default=2048)

    ssl = sub.add_parser("ssl")
    ssl.add_argument("--tag", default="C", choices=sorted(SPLIT_TAGS))
    ssl.add_argument("--seed", type=int, required=True)
    ssl.add_argument("--device", default="cpu")
    ssl.add_argument("--smoke-updates", type=int, default=None)

    a = sub.add_parser("arm")
    a.add_argument("--tag", default="C", choices=sorted(SPLIT_TAGS))
    a.add_argument("--arm", required=True, choices=sorted(ab.ARM_KINDS))
    a.add_argument("--seed", type=int, required=True)
    a.add_argument("--device", default="cpu")
    a.add_argument("--smoke-epochs", type=int, default=None)

    pr = sub.add_parser("predict")
    pr.add_argument("--tag", default="C", choices=sorted(SPLIT_TAGS))
    pr.add_argument("--arms", nargs="*", default=None)
    pr.add_argument("--seeds", nargs="*", default=[0, 1])
    pr.add_argument("--device", default="cpu")
    pr.add_argument("--smoke", action="store_true")

    sm = sub.add_parser("score-monitor")
    sm.add_argument("--tag", default="C", choices=sorted(SPLIT_TAGS))
    sm.add_argument("--candidate", required=True, choices=sorted(CANDIDATE_ARMS))

    lf = sub.add_parser("locked-freeze")
    lf.add_argument("--tag", default="C", choices=sorted(SPLIT_TAGS))

    ls = sub.add_parser("locked-score")
    ls.add_argument("--tag", default="C", choices=sorted(SPLIT_TAGS))

    ft = sub.add_parser("full10k-train")
    ft.add_argument("--arm", required=True, choices=["GLRC", "DLP"])
    ft.add_argument("--seed", type=int, required=True)
    ft.add_argument("--device", default="cpu")

    sub.add_parser("full10k-score")
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
        "cpu-gate-glrc": cmd_cpu_gate_glrc,
        "cpu-gate-dlp": cmd_cpu_gate_dlp,
        "ssl": cmd_ssl,
        "arm": cmd_arm,
        "predict": cmd_predict,
        "score-monitor": cmd_score_monitor,
        "locked-freeze": cmd_locked_freeze,
        "locked-score": cmd_locked_score,
        "full10k-train": cmd_full10k_train,
        "full10k-score": cmd_full10k_score,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
