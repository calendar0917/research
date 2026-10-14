"""cscl-bjoc-junction-sufficiency-audit-v1 driver.

Protocol: ``configs/bjoc_junction_sufficiency_audit_v1.yaml`` (frozen before
any training).  Track: ``tracks/structure-contribution``.

Subcommands
-----------
``verify-inputs``  recompute split-D fingerprints and verify every reused
                   artifact hash against the source round manifests
``freeze``         write ``protocol_freeze.json`` binding config + code + sources
``stage-a``        CPU correctness / label-isolation gate (17 checks)
``grad-audit``     allocated vs gradient-active parameters, composer column
                   blocks, dead channels, forward timing (A0/A1/A2)
``train``          one supervised R3 trajectory (A0 or A1; A2 is reused)
``predict``        label-blind Monitor predictions + hash-freeze manifest
``score-monitor``  the single Monitor label read of this round + metric table
``b-algebraic``    B1 algebraic blindness proof + B4 Test 2 + B5 synthetic truth
``b-layerwise``    B2 layer-wise representation audit + B4 Tests 1 and 4
``b-collisions``   B3 label-blind real-molecule structure collision search
``finalize``       merge A + B decision tokens into ``decision.json``

Discipline: official test and official valid are never referenced by any code
path.  Locked labels are never read (Locked structures may appear only in the
label-blind Part B structural scan, as allowed by the round protocol).  Monitor
labels are read exactly once, after the new prediction manifest is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
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
import bjoc_junction_sufficiency_models_v1 as jsm  # noqa: E402
import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import bjoc_task_adaptive_ssl_v1 as tat  # noqa: E402
import cscl_features as cf  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = jsm.PROTOCOL_ID
CONFIG_PATH = _CODE_DIR.parent / "configs" / "bjoc_junction_sufficiency_audit_v1.yaml"
RES = REPO_ROOT / "tracks/structure-contribution/results"
OUT = RES / "bjoc_junction_sufficiency_audit_v1"
SRC = RES / "bjoc_task_adaptive_ssl_v1"
ENCODED_TRAIN = REPO_ROOT / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt"
CANONICAL_SMILES = REPO_ROOT / "tracks/ksvd/results/zinc_cssd_basis_reuse_v1/train_canonical_smiles.npz"

SEED = 0
ARMS = ("A0", "A1", "A2")
SPLIT_TAG = "D"
N_TRAIN = 10000

SPLIT_PATH = SRC / "split_D.json"
PREP_PATH = SRC / "prep_fit_D.json"
SSL_PATH = SRC / "pretrain_D_s0.pt"
SSL_JSON = SRC / "pretrain_D_s0.json"
SRC_SOUP = SRC / "soup_P3_D_s0.pt"
SRC_P3_TRAIN = SRC / "train_P3_D_s0.json"
SRC_MON_PRED = SRC / "predictions/monitor_P3_D_s0.npz"
SRC_MON_Y = SRC / "monitor_y_D.npy"
SRC_MON_LOCK = SRC / "monitor_read_D.lock"
SRC_OBJECTS_MANIFEST = SRC / "objects_manifest_D.json"
OBJECT_FILES = {
    "fit": SRC / "objects_fit_D.pt",
    "monitor": SRC / "objects_monitor_D.pt",
    "locked": SRC / "objects_locked_D.pt",
}

FREEZE_PATH = OUT / "protocol_freeze.json"
PRED_MANIFEST = OUT / "prediction_manifest.json"
MONITOR_LOCK = OUT / "monitor_read.lock"
MONITOR_Y = OUT / "monitor_y.npy"
MONITOR_SCORES = OUT / "monitor_scores.json"
DECISION = OUT / "decision.json"


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


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes()).hexdigest()


def git_commit() -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def soup_path(arm: str) -> Path:
    return SRC_SOUP if arm == "A2" else OUT / f"soup_{arm}_D_s0.pt"


def train_json_path(arm: str) -> Path:
    return OUT / f"train_{arm}_D_s0.json"


def pred_path(arm: str) -> Path:
    return OUT / "predictions" / f"monitor_{arm}_D_s0.npz"


def _code_paths() -> dict[str, Path]:
    return {
        "driver": _CODE_DIR / "run_bjoc_junction_sufficiency_audit_v1.py",
        "module": _CODE_DIR / "bjoc_junction_sufficiency_models_v1.py",
        "config": CONFIG_PATH,
        "composer": _CODE_DIR / "structural_relation_composer_v1.py",
        "objects": _CODE_DIR / "structural_relation_objects_v1.py",
        "masked": _CODE_DIR / "bjoc_masked_pretraining_v1.py",
        "perfmech": _CODE_DIR / "bjoc_perfmech_long_v1.py",
        "arch": _CODE_DIR / "bjoc_arch_breakthrough_v1.py",
        "tat": _CODE_DIR / "bjoc_task_adaptive_ssl_v1.py",
    }


def code_hashes() -> dict[str, str]:
    return {k: file_sha256(p) for k, p in _code_paths().items()}


# ---------------------------------------------------------------------------
# source verification
# ---------------------------------------------------------------------------


def audit_sources() -> dict[str, Any]:
    cfg = load_config()
    reuse = cfg["reuse"]
    checks: dict[str, Any] = {}

    def chk(name: str, path: Path, expected: str | None) -> bool:
        exists = path.exists()
        actual = file_sha256(path) if exists else None
        hit = bool(exists and (expected is None or actual == expected))
        checks[name] = {"path": str(path.relative_to(REPO_ROOT)), "exists": exists, "sha256": actual, "matches": hit}
        return hit

    ok = True
    ok &= chk("ssl_encoder_s0", SSL_PATH, reuse["ssl_encoder_seed0"]["file_sha256"])
    ok &= chk("p3_soup_s0", SRC_SOUP, reuse["p3_soup_seed0"]["file_sha256"])
    ok &= chk("objects_fit", OBJECT_FILES["fit"], reuse["fit_objects"]["file_sha256"])
    ok &= chk("objects_monitor", OBJECT_FILES["monitor"], reuse["monitor_objects"]["file_sha256"])
    ok &= chk("p3_monitor_predictions", SRC_MON_PRED, None)
    ok &= chk("p3_train_record", SRC_P3_TRAIN, None)

    # cross-check against the source round's own manifests
    cross: dict[str, Any] = {}
    if SRC_OBJECTS_MANIFEST.exists():
        om = read_json(SRC_OBJECTS_MANIFEST)
        cross["objects_manifest_matches_config"] = (
            om["objects_fit_sha256"] == reuse["fit_objects"]["file_sha256"]
            and om["objects_monitor_sha256"] == reuse["monitor_objects"]["file_sha256"]
            and int(om["forbidden_label_access_attempts"]) == 0
        )
    if SRC_P3_TRAIN.exists():
        t = read_json(SRC_P3_TRAIN)
        cross["p3_train_matches_config"] = (
            t["ssl_source_sha256"] == reuse["ssl_encoder_seed0"]["file_sha256"]
            and t["soup_sha256"] == reuse["p3_soup_seed0"]["file_sha256"]
            and t["init_full_sha256"] == reuse["p3_train_record"]["init_full_sha256"]
            and t["init_group_state_sha256"] == reuse["p3_train_record"]["init_group_state_sha256"]
            and t["init_encoder_sha256"] == reuse["p3_train_record"]["init_encoder_sha256"]
            and t["encoder_used_sha256"] == reuse["ssl_encoder_seed0"]["encoder_used_sha256"]
            and t["batch_order"]["full_hash"] == reuse["p3_train_record"]["batch_order_hash"]
        )
    if SSL_JSON.exists():
        s = read_json(SSL_JSON)
        cross["ssl_json_matches_config"] = s["encoder_final_sha256"] == reuse["ssl_encoder_seed0"]["encoder_used_sha256"]

    # recompute the split from the committed canonical SMILES table
    smiles = [str(x) for x in cf.load_canonical_smiles()]
    if len(smiles) != N_TRAIN:
        raise SystemExit(f"canonical SMILES table has {len(smiles)} rows, expected {N_TRAIN}")
    sp = bp.build_split(smiles, cfg["split"]["salt"])
    fp = cfg["split"]["fingerprints"]
    split_ok = (
        bp.rows_sha256(sp["fit"]) == fp["fit_sha256_rows"]
        and bp.rows_sha256(sp["monitor"]) == fp["monitor_sha256_rows"]
        and bp.rows_sha256(sp["locked"]) == fp["locked_sha256_rows"]
        and len(sp["fit"]) == 8000
        and len(sp["monitor"]) == 1000
        and len(sp["locked"]) == 1000
        and int(sp["duplicate_groups_cross_sides"]) == 0
    )
    cross["split_recomputed_matches_fingerprints"] = bool(split_ok)
    ok &= bool(split_ok)
    return {"ok": bool(ok), "checks": checks, "cross_checks": cross}


def cmd_verify_inputs(args: argparse.Namespace) -> int:
    audit = audit_sources()
    write_json(OUT / "input_audit.json", {"protocol": PROTOCOL_ID, "git_commit": git_commit(), **audit})
    log(f"[verify-inputs] ok={audit['ok']}")
    for name, rec in audit["checks"].items():
        log(f"  {name:26s} {'OK' if rec['matches'] else 'MISSING'}")
    for name, hit in audit["cross_checks"].items():
        log(f"  {name:26s} {'OK' if hit else 'MISSING'}")
    return 0 if audit["ok"] else 1


def cmd_freeze(args: argparse.Namespace) -> int:
    audit = audit_sources()
    if not audit["ok"]:
        raise SystemExit("source audit failed; refusing to freeze")
    cfg = load_config()
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "code_hashes": code_hashes(),
        "split_tag": SPLIT_TAG,
        "split_salt": cfg["split"]["salt"],
        "split_fingerprints": cfg["split"]["fingerprints"],
        "source_hashes": {k: v["sha256"] for k, v in audit["checks"].items()},
        "source_audit_ok": bool(audit["ok"]),
        "arms": list(ARMS),
        "seed": SEED,
        "recipe": "R3",
        "budget_gpu_hard_limit_hours": cfg["budget"]["gpu_hard_limit_hours"],
        "label_access": dict(cfg["label_access"]),
        "locked_read_authorized": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(FREEZE_PATH, payload)
    log(f"[freeze] written commit={payload['git_commit'][:12]}")
    return 0


def verify_freeze(*, strict_code: bool = True) -> dict[str, Any]:
    if not FREEZE_PATH.exists():
        raise SystemExit("protocol is not frozen; run `freeze` first")
    frozen = read_json(FREEZE_PATH)
    if frozen["protocol"] != PROTOCOL_ID:
        raise SystemExit("freeze protocol id mismatch")
    if frozen["config_sha256"] != file_sha256(CONFIG_PATH):
        raise SystemExit("protocol config drifted from the freeze")
    if strict_code:
        current = code_hashes()
        drift = {k: (frozen["code_hashes"].get(k), v) for k, v in current.items() if frozen["code_hashes"].get(k) != v}
        if drift:
            raise SystemExit(f"code drift versus freeze: {drift}")
    return frozen


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------


def load_split() -> dict[str, Any]:
    return read_json(SPLIT_PATH)


def load_norm() -> dict[str, Any]:
    return read_json(PREP_PATH)


def load_objects(zone: str) -> list[sro.MolObjects]:
    path = OBJECT_FILES[zone]
    if not path.exists():
        raise SystemExit(f"missing object cache {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_y_fit() -> np.ndarray:
    split = load_split()
    return bp.load_y_rows(ENCODED_TRAIN, split["fit_rows"])


def load_ssl_state() -> dict[str, dict[str, torch.Tensor]]:
    return torch.load(SSL_PATH, map_location="cpu", weights_only=False)


def load_soup(arm: str) -> Mapping[str, torch.Tensor]:
    path = soup_path(arm)
    if not path.exists():
        raise SystemExit(f"missing soup for {arm}: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def build_loaded_model(arm: str, device: torch.device | str = "cpu") -> jsm.BJOCSufficiency:
    norm = load_norm()
    model = jsm.BJOCSufficiency(
        src.BJOCConfig(),
        np.asarray(norm["global_mean"], dtype=np.float32),
        np.asarray(norm["global_scale"], dtype=np.float32),
        mode=jsm.ARM_MODES[arm],
    )
    model.load_state_dict(load_soup(arm))
    model.to(device)
    model.eval()
    return model


def build_init_model(mode: str, device: torch.device | str = "cpu") -> jsm.BJOCSufficiency:
    _base, model, _info = jsm.build_sufficiency_from_init(SEED, load_norm(), device, mode, load_ssl_state())
    model.eval()
    return model


def predict_objects(model: jsm.BJOCSufficiency, objs: Sequence[sro.MolObjects], device: torch.device | str, batch_size: int = 256) -> np.ndarray:
    device = torch.device(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(objs), int(batch_size)):
            batch = bp._mv(sro.batch_objects(list(objs[i : i + int(batch_size)])), device)
            out.append(model(batch).detach().cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros(0)


def check_object_rows(objs: Sequence[sro.MolObjects], rows: Sequence[int], name: str) -> None:
    if [int(o.mol_index) for o in objs] != [int(r) for r in rows]:
        raise SystemExit(f"{name} object rows do not match the frozen split")


# ---------------------------------------------------------------------------
# Stage A: CPU correctness / isolation gates
# ---------------------------------------------------------------------------


def _permute_junctions(objs: Sequence[sro.MolObjects], seed: int = 0) -> list[sro.MolObjects]:
    rng = np.random.default_rng(int(seed))
    out = []
    for o in objs:
        perm = rng.permutation(o.junc_index.shape[0])
        out.append(
            sro.MolObjects(
                mol_index=o.mol_index,
                smiles=o.smiles,
                n_atoms=o.n_atoms,
                atom_feat=o.atom_feat,
                bond_index=o.bond_index,
                bond_type=o.bond_type,
                bond_feat=o.bond_feat,
                junc_index=o.junc_index[perm],
                junc_bond=o.junc_bond[perm],
                junc_feat=o.junc_feat[perm],
                global_feat=o.global_feat,
            )
        )
    return out


def _perturb_junctions(objs: Sequence[sro.MolObjects], seed: int = 0) -> list[sro.MolObjects]:
    """Random junction features + random junction row order (same counts)."""
    rng = np.random.default_rng(int(seed))
    out = []
    for o in objs:
        jf = rng.normal(size=o.junc_feat.shape).astype(np.float32) * 5.0
        perm = rng.permutation(o.junc_index.shape[0])
        out.append(
            sro.MolObjects(
                mol_index=o.mol_index,
                smiles=o.smiles,
                n_atoms=o.n_atoms,
                atom_feat=o.atom_feat,
                bond_index=o.bond_index,
                bond_type=o.bond_type,
                bond_feat=o.bond_feat,
                junc_index=o.junc_index[perm],
                junc_bond=o.junc_bond[perm],
                junc_feat=jf[perm],
                global_feat=o.global_feat,
            )
        )
    return out


def cmd_stage_a(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    dev = torch.device(args.device)
    split = load_split()
    fit_objs = load_objects("fit")
    mon_objs = load_objects("monitor")
    check_object_rows(fit_objs, split["fit_rows"], "fit")
    check_object_rows(mon_objs, split["monitor_rows"], "monitor")
    norm = load_norm()
    ssl_state = load_ssl_state()
    p3_soup = load_soup("A2")
    frozen_p3 = read_json(SRC_P3_TRAIN)
    small_fit = fit_objs[:8]
    small_batch = bp._mv(sro.batch_objects(small_fit), dev)

    gates: dict[str, Any] = {}

    # -- A2 model (P3 weights) and the parent reference ----------------------
    a2 = build_loaded_model("A2", dev)
    base = bp.build_bjoc_null(SEED, norm, dev)
    base.load_state_dict({k: v.to(dev) for k, v in p3_soup.items()})
    base.eval()
    f_parent = predict_objects(base, small_fit, dev)
    f_a2 = predict_objects(a2, small_fit, dev)
    frozen_pred = np.asarray(np.load(SRC_MON_PRED, allow_pickle=True)["pred"], dtype=np.float64)
    f_a2_monitor = predict_objects(a2, mon_objs, dev)
    gates["G1_A2_equals_parent"] = {
        "pass": bool(float(np.abs(f_parent - f_a2).max()) <= 1e-6),
        "max_abs_diff_small_fit": float(np.abs(f_parent - f_a2).max()),
        "max_abs_diff_vs_frozen_p3_monitor": float(np.abs(f_a2_monitor - frozen_pred).max()),
    }

    # -- per-mode init models -------------------------------------------------
    a0 = build_init_model(jsm.MODE_A0, dev)
    a1 = build_init_model(jsm.MODE_A1, dev)
    feats0 = a0.forward_features(small_batch)
    feats1 = a1.forward_features(small_batch)

    gates["G2_A0_WVq_zero"] = {
        "pass": bool(
            float(feats0["W"].abs().max()) == 0.0
            and float(feats0["V"].abs().max()) == 0.0
            and float(feats0["n_b"].abs().max()) == 0.0
        ),
        "W_max_abs": float(feats0["W"].abs().max()),
        "V_max_abs": float(feats0["V"].abs().max()),
        "n_b_max_abs": float(feats0["n_b"].abs().max()),
    }

    pert = _perturb_junctions(small_fit, seed=0)
    f0 = predict_objects(a0, small_fit, dev)
    f0_pert = predict_objects(a0, pert, dev)
    gates["G3_A0_junction_insensitive"] = {
        "pass": bool(float(np.abs(f0 - f0_pert).max()) <= 1e-6),
        "max_abs_diff": float(np.abs(f0 - f0_pert).max()),
    }

    # -- A1 broadcast invariants ---------------------------------------------
    n_bonds = int(small_batch["bond_feat"].shape[0])
    bond_batch = small_batch["bond_batch"]
    spread_w = 0.0
    spread_v = 0.0
    for g in range(int(small_batch["n_graphs"])):
        m = bond_batch == g
        if int(m.sum()) == 0:
            continue
        spread_w = max(spread_w, float((feats1["W"][m].max(0).values - feats1["W"][m].min(0).values).abs().max()))
        spread_v = max(spread_v, float((feats1["V"][m].max(0).values - feats1["V"][m].min(0).values).abs().max()))
    gates["G4_A1_WV_uniform_within_molecule"] = {
        "pass": bool(spread_w == 0.0 and spread_v == 0.0),
        "W_max_within_molecule_spread": spread_w,
        "V_max_within_molecule_spread": spread_v,
    }

    perm_fit = _permute_junctions(small_fit, seed=1)
    f1 = predict_objects(a1, small_fit, dev)
    f1_perm = predict_objects(a1, perm_fit, dev)
    f2 = predict_objects(a2, small_fit, dev)
    f2_perm = predict_objects(a2, perm_fit, dev)
    gates["G5_A1_junction_row_permutation_invariant"] = {
        "pass": bool(float(np.abs(f1 - f1_perm).max()) <= 1e-6),
        "max_abs_diff": float(np.abs(f1 - f1_perm).max()),
    }
    gates["G6_A2_junction_row_permutation_invariant"] = {
        "pass": bool(float(np.abs(f2 - f2_perm).max()) <= 1e-6),
        "max_abs_diff": float(np.abs(f2 - f2_perm).max()),
    }

    # -- G7: A2 n_b equals an independent count ------------------------------
    jb = small_batch["junc_bond"].detach().cpu().numpy()
    n_expected = np.bincount(jb.reshape(-1), minlength=n_bonds).astype(np.float64)
    feats2 = a2.forward_features(small_batch)
    n_got = feats2["n_b"].detach().cpu().numpy().astype(np.float64)
    gates["G7_A2_per_bond_count_consistent"] = {
        "pass": bool(np.max(np.abs(n_got - n_expected)) == 0.0),
        "max_abs_diff": float(np.max(np.abs(n_got - n_expected))),
    }

    # -- G8/G9: junction-encoder gradient activity ----------------------------
    fit_for_grad = fit_objs[:128]
    y_grad = load_y_fit()[: len(fit_for_grad)]
    grad0 = jsm.gradient_audit(a0, objs=fit_for_grad, y=y_grad, device=dev, mode=jsm.MODE_A0)
    grad1 = jsm.gradient_audit(a1, objs=fit_for_grad, y=y_grad, device=dev, mode=jsm.MODE_A1)
    grad2 = jsm.gradient_audit(a2, objs=fit_for_grad, y=y_grad, device=dev, mode=jsm.MODE_A2)
    gates["G8_A1_A2_junction_encoder_gradient_nonzero"] = {
        "pass": bool(
            grad1["modules"]["junction_encoder"]["grad_norm_total"] not in (None, 0.0)
            and grad1["modules"]["junction_encoder"]["grad_norm_total"] > 0.0
            and grad2["modules"]["junction_encoder"]["grad_norm_total"] not in (None, 0.0)
            and grad2["modules"]["junction_encoder"]["grad_norm_total"] > 0.0
        ),
        "A1_junction_grad_norm": grad1["modules"]["junction_encoder"]["grad_norm_total"],
        "A2_junction_grad_norm": grad2["modules"]["junction_encoder"]["grad_norm_total"],
    }
    g9 = grad0["modules"]["junction_encoder"]
    gates["G9_A0_junction_encoder_gradient_zero"] = {
        "pass": bool(
            (g9["grad_norm_total"] in (None, 0.0))
            and int(g9["active_entries_nonzero_grad"]) == 0
        ),
        "A0_junction_grad_norm": g9["grad_norm_total"],
        "A0_junction_active_entries": int(g9["active_entries_nonzero_grad"]),
        "A0_junction_allocated_parameters": int(g9["allocated_parameters"]),
    }

    # -- G10/G11/G12: relabel / endpoint swap / batch composition -------------
    guard = ab.LabelGuard(cf.load_official_train(), split["monitor_rows"] + split["locked_rows"])
    rel_objs = tat.relabel_objects(small_fit, guard, seed=0)
    swap_objs = tat.endpoint_swap_objects(small_fit, guard)

    def inv_max(model: jsm.BJOCSufficiency, objs: Sequence[sro.MolObjects]) -> float:
        return float(np.abs(predict_objects(model, small_fit, dev) - predict_objects(model, objs, dev)).max())

    def batch_member(model: jsm.BJOCSufficiency, objs: Sequence[sro.MolObjects]) -> float:
        y_all = predict_objects(model, objs, dev)
        y_one = np.asarray([predict_objects(model, [o], dev)[0] for o in objs])
        return float(np.abs(y_all - y_one).max())

    invs: dict[str, Any] = {}
    for name, model in (("A0", a0), ("A1", a1), ("A2", a2)):
        invs[name] = {
            "atom_relabel_max_abs_diff": inv_max(model, rel_objs),
            "bond_endpoint_swap_max_abs_diff": inv_max(model, swap_objs),
            "batch_composition_max_abs_diff": batch_member(model, small_fit),
        }
    worst = max(v for rec in invs.values() for v in rec.values())
    gates["G10_G11_G12_invariances"] = {
        "pass": bool(worst <= 1e-5),
        "per_arm": invs,
        "worst": float(worst),
    }

    # -- G13: save/load replay ------------------------------------------------
    replay: dict[str, Any] = {}
    for name, model in (("A0", a0), ("A1", a1), ("A2", a2)):
        tmp = OUT / f"stage_a_replay_{name}.pt"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), tmp)
        fresh = jsm.BJOCSufficiency(src.BJOCConfig(), np.asarray(norm["global_mean"], dtype=np.float32), np.asarray(norm["global_scale"], dtype=np.float32), mode=jsm.ARM_MODES[name])
        fresh.load_state_dict(torch.load(tmp, map_location="cpu", weights_only=False))
        fresh.to(dev)
        fresh.eval()
        replay[name] = float(np.abs(predict_objects(model, small_fit, dev) - predict_objects(fresh, small_fit, dev)).max())
        tmp.unlink(missing_ok=True)
    gates["G13_save_load_replay"] = {"pass": bool(max(replay.values()) == 0.0), "per_arm_max_abs_diff": replay}

    # -- G14: Fit-only standardisation ----------------------------------------
    src_prep = read_json(PREP_PATH)
    gates["G14_fit_only_standardisation"] = {
        "pass": bool(src_prep.get("fit_only") is True and int(src_prep.get("n_fit", 0)) == 8000),
        "fit_only": src_prep.get("fit_only"),
        "n_fit": src_prep.get("n_fit"),
        "source": str(PREP_PATH.relative_to(REPO_ROOT)),
        "note": "the standardiser is the frozen split-D Fit-only one from the TA round; no Monitor feature enters it",
    }

    # -- G15: label isolation --------------------------------------------------
    om = read_json(SRC_OBJECTS_MANIFEST)
    gates["G15_label_isolation"] = {
        "pass": bool(int(om["forbidden_label_access_attempts"]) == 0 and not MONITOR_LOCK.exists()),
        "objects_manifest_forbidden_access_attempts": int(om["forbidden_label_access_attempts"]),
        "monitor_label_read_this_round": bool(MONITOR_LOCK.exists()),
        "locked_label_read_this_round": False,
        "note": "Monitor labels are read only by score-monitor after the new prediction manifest is frozen; Locked labels are never read",
    }

    # -- G16: no official valid/test code path ---------------------------------
    scan_files = [_CODE_DIR / "run_bjoc_junction_sufficiency_audit_v1.py", _CODE_DIR / "bjoc_junction_sufficiency_models_v1.py"]
    scan_tokens = ["load_official_" + "valid", "load_official_" + "test"]
    hits: list[str] = []
    for p in scan_files:
        text = p.read_text(encoding="utf-8")
        for token in scan_tokens:
            if token in text:
                hits.append(f"{p.name}:{token}")
    gates["G16_no_official_valid_test_path"] = {
        "pass": bool(not hits),
        "hits": hits,
        "dataset_loader_used": "cscl_features.load_official_train only",
        "note": "static scan for official valid/test loader calls; both files also carry explicit never-loaded flags",
    }

    # -- G17: no GNN/attention/iteration --------------------------------------
    g17: dict[str, Any] = {}
    for name, model in (("A0", a0), ("A1", a1), ("A2", a2)):
        g17[name] = src.forbidden_module_scan(model)
    gates["G17_no_forbidden_modules"] = {"pass": bool(all(len(v) == 0 for v in g17.values())), "hits": g17}

    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": SPLIT_TAG,
        "device": str(dev),
        "source_p3_init_hashes": {
            "init_full_sha256": frozen_p3["init_full_sha256"],
            "init_group_state_sha256": frozen_p3["init_group_state_sha256"],
            "init_encoder_sha256": frozen_p3["init_encoder_sha256"],
        },
        "gates": gates,
        "gradient_audit_at_init": {"A0": grad0, "A1": grad1, "A2": grad2},
        "ok": bool(all(rec.get("pass", False) for rec in gates.values())),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "stage_a.json", payload)
    log(f"[stage-a] ok={payload['ok']}")
    for name, rec in gates.items():
        log(f"  {name:52s} {'PASS' if rec.get('pass') else 'FAIL'}")
    return 0 if payload["ok"] else 1


# ---------------------------------------------------------------------------
# parameter / gradient audit
# ---------------------------------------------------------------------------


def _forward_timing(model: jsm.BJOCSufficiency, objs: Sequence[sro.MolObjects], device: torch.device | str, mode: str, repeats: int = 5) -> dict[str, Any]:
    dev = torch.device(device)
    batch = bp._mv(sro.batch_objects(list(objs[:256])), dev)
    model.eval()
    with torch.no_grad():
        model(batch, mode=mode)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(int(repeats)):
            model(batch, mode=mode)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / max(int(repeats), 1)
    return {"n_molecules": len(objs[:256]), "seconds_per_forward": float(dt), "molecules_per_second": float(len(objs[:256]) / max(dt, 1e-12))}


def cmd_grad_audit(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    dev = torch.device(args.device)
    split = load_split()
    fit_objs = load_objects("fit")
    check_object_rows(fit_objs, split["fit_rows"], "fit")
    y_fit = load_y_fit()[:128]
    payload: dict[str, Any] = read_json(OUT / "parameter_gradient_audit.json") if (OUT / "parameter_gradient_audit.json").exists() else {"protocol": PROTOCOL_ID, "git_commit": git_commit()}
    phases = list(args.phase) if args.phase else ["init", "trained"]
    for phase in phases:
        section: dict[str, Any] = {}
        for arm in ARMS:
            if phase == "init":
                _base, model, info = jsm.build_sufficiency_from_init(SEED, load_norm(), dev, jsm.ARM_MODES[arm], load_ssl_state())
            else:
                model = build_loaded_model(arm, dev)
                info = {"soup_path": str(soup_path(arm).relative_to(REPO_ROOT))}
            rec = jsm.gradient_audit(model, objs=fit_objs[:128], y=y_fit, device=dev, mode=jsm.ARM_MODES[arm])
            rec["activation"] = jsm.activation_audit(model, objs=fit_objs[:256], device=dev, mode=jsm.ARM_MODES[arm])
            rec["forward_timing"] = _forward_timing(model, fit_objs[:256], dev, jsm.ARM_MODES[arm])
            rec["info"] = info
            section[arm] = rec
            log(
                f"[grad-audit:{phase}] {arm} active_entries {rec['active_entries_nonzero_grad']}/{rec['allocated_parameters']} "
                f"junction_grad {rec['modules']['junction_encoder']['grad_norm_total']} "
                f"fwd {rec['forward_timing']['seconds_per_forward']:.4f}s"
            )
        payload[f"at_{phase}"] = section
    write_json(OUT / "parameter_gradient_audit.json", payload)
    return 0


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=True)
    arm = str(args.arm)
    if arm not in ("A0", "A1"):
        raise SystemExit("only A0 and A1 are new trajectories; A2 is read-only reuse")
    seed = int(args.seed)
    if seed != SEED:
        raise SystemExit("this round is seed 0 only")
    smoke = args.smoke_epochs is not None
    target_soup = (OUT / "smoke" / f"soup_{arm}_D_s0_smoke.pt") if smoke else soup_path(arm)
    target_json = (OUT / "smoke" / f"train_{arm}_D_s0_smoke.json") if smoke else train_json_path(arm)
    if target_soup.exists() or target_json.exists():
        raise SystemExit(f"{arm} seed {seed} already exists; retraining is not allowed")
    split = load_split()
    fit_objs = load_objects("fit")
    check_object_rows(fit_objs, split["fit_rows"], "fit")
    norm = load_norm()
    y_fit = load_y_fit()
    ssl_state = load_ssl_state()
    payload = jsm.train_supervised_suff(
        arm=arm,
        seed=seed,
        recipe_key="R3",
        init="ssl",
        train_objs=fit_objs,
        rows=split["fit_rows"],
        y_fit=y_fit,
        norm=norm,
        device=torch.device(args.device),
        mode=jsm.ARM_MODES[arm],
        ssl_state=ssl_state,
        smoke_epochs=args.smoke_epochs,
    )
    frozen = read_json(SRC_P3_TRAIN)
    expected_encoder = frozen["encoder_used_sha256"]
    checks = {
        "init_full_matches_p3": payload["init_full_sha256"] == frozen["init_full_sha256"],
        "init_group_matches_p3": payload["init_group_state_sha256"] == frozen["init_group_state_sha256"],
        "init_encoder_matches_p3": payload["init_encoder_sha256"] == frozen["init_encoder_sha256"],
        "encoder_used_matches_ssl_source": payload["encoder_used_sha256"] == expected_encoder,
        "parameter_audit_exact": bool(payload["parameter_audit"]["parameter_exact"]),
        "batch_order_matches_p3": payload["batch_order"]["full_hash"] == frozen["batch_order"]["full_hash"],
        "full_recipe_unchanged": (payload["epochs"] == 360 and payload["updates"] == 22680) if not smoke else True,
    }
    if smoke:
        checks["init_checks_only"] = True
        checks["batch_order_matches_p3"] = payload["batch_order"]["full_matches"]
    payload["p3_matched_checks"] = checks
    if not all(checks.values()):
        raise SystemExit(f"P3-matching checks failed: {checks}")
    soup = target_soup
    soup.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu().clone() for k, v in payload["soup_state"].items()}, soup)
    payload.pop("soup_state", None)
    payload["soup_path"] = str(soup.relative_to(REPO_ROOT))
    payload["soup_sha256"] = file_sha256(soup)
    write_json(target_json, payload)
    log(f"[train:{arm}] soup fit {payload['soup_fit_mae']:.6f} wall {payload['wall_seconds']:.0f}s -> {soup.name}")
    return 0


# ---------------------------------------------------------------------------
# predictions + Monitor scoring
# ---------------------------------------------------------------------------


def cmd_predict(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    if MONITOR_LOCK.exists():
        raise SystemExit("Monitor labels already read in this round; refusing new predictions")
    mon_objs = load_objects("monitor")
    check_object_rows(mon_objs, load_split()["monitor_rows"], "monitor")
    device = torch.device(args.device)
    arms = tuple(args.arms) if args.arms else ARMS
    manifest = read_json(PRED_MANIFEST) if PRED_MANIFEST.exists() else {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": SPLIT_TAG,
        "entries": {},
        "monitor_labels_read": False,
        "locked_labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    for arm in arms:
        model = build_loaded_model(arm, device)
        pred = predict_objects(model, mon_objs, device)
        if not np.isfinite(pred).all():
            raise SystemExit(f"non-finite prediction for {arm}")
        out = pred_path(arm)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, row=np.arange(len(mon_objs), dtype=np.int64), pred=pred.astype(np.float32))
        manifest["entries"][arm] = {
            "path": str(out.relative_to(REPO_ROOT)),
            "sha256": file_sha256(out),
            "soup_sha256": file_sha256(soup_path(arm)),
            "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        log(f"[predict] {arm} -> {out.name} sha={manifest['entries'][arm]['sha256'][:12]}")
    # A2 must reproduce the historical frozen P3 predictions
    frozen = np.asarray(np.load(SRC_MON_PRED, allow_pickle=True)["pred"], dtype=np.float64)
    a2_pred = np.asarray(np.load(pred_path("A2"), allow_pickle=True)["pred"], dtype=np.float64)
    a2_diff = float(np.abs(a2_pred - frozen).max())
    manifest["a2_matches_frozen_p3_max_abs_diff"] = a2_diff
    if a2_diff > 1e-5:
        raise SystemExit(f"A2 does not reproduce the frozen P3 Monitor predictions: {a2_diff}")
    manifest["monitor_labels_read"] = False
    manifest["locked_labels_read"] = False
    write_json(PRED_MANIFEST, manifest)
    log("[predict] manifest written")
    return 0


def _verify_prediction_manifest() -> dict[str, Any]:
    manifest = read_json(PRED_MANIFEST)
    for arm, rec in manifest["entries"].items():
        p = REPO_ROOT / rec["path"]
        if not p.exists() or file_sha256(p) != rec["sha256"]:
            raise SystemExit(f"prediction drift for {arm}")
    missing = set(ARMS) - set(manifest["entries"])
    if missing:
        raise SystemExit(f"prediction manifest incomplete: {sorted(missing)}")
    return manifest


def _read_pred_npz(path: Path, n: int) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    pred = np.asarray(z["pred"], dtype=np.float64).reshape(-1)
    row = np.asarray(z["row"], dtype=np.int64).reshape(-1)
    if len(pred) != n or not np.array_equal(row, np.arange(n)):
        raise SystemExit(f"row/pred shape mismatch in {path}")
    return pred


def _ensure_monitor_y() -> np.ndarray:
    split = load_split()
    if MONITOR_LOCK.exists():
        y = np.asarray(np.load(MONITOR_Y), dtype=np.float64).reshape(-1)
        lock = read_json(MONITOR_LOCK)
        if array_sha256(y) != lock["y_sha256"]:
            raise SystemExit("Monitor y file drifted from its lock record")
        return y
    if SRC_MON_Y.exists() and SRC_MON_LOCK.exists():
        y = np.asarray(np.load(SRC_MON_Y), dtype=np.float64).reshape(-1)
        lock = read_json(SRC_MON_LOCK)
        if array_sha256(y) != lock["y_sha256"]:
            raise SystemExit("source Monitor y file does not match the source lock")
        provenance = "reused the historically exposed split-D Monitor label file (same rows, hash verified)"
    else:  # pragma: no cover - source lock is committed in the TA round
        rows = split["monitor_rows"]
        y = bp.load_y_rows(ENCODED_TRAIN, rows)
        provenance = "read from the committed encoded train cache during this round"
    np.save(MONITOR_Y, y)
    write_json(
        MONITOR_LOCK,
        {
            "protocol": PROTOCOL_ID,
            "split_tag": SPLIT_TAG,
            "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_rows": len(y),
            "y_sha256": array_sha256(y),
            "prediction_manifest_sha256": file_sha256(PRED_MANIFEST),
            "label_provenance": provenance,
            "exposure_disclosure": "split-D Monitor is historically exposed exploratory development data, not a blind validation",
            "official_valid_loaded": False,
            "official_test_loaded": False,
        },
    )
    log(f"[monitor-read] n={len(y)} sha={array_sha256(y)[:12]}")
    return y


def _metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    err = np.abs(pred - y)
    order = np.argsort(-err)
    keep = np.ones(len(err), dtype=bool)
    keep[order[:10]] = False
    ordinary = y > bp.TAIL_THRESHOLD
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "median_abs": float(np.median(err)),
        "p90_abs": float(np.percentile(err, 90)),
        "ordinary_mae": float(err[ordinary].mean()),
        "ordinary_n": int(ordinary.sum()),
        "tail_mae": float(err[~ordinary].mean()),
        "tail_n": int((~ordinary).sum()),
        "top10_removed_mae": float(err[keep].mean()),
        "n": int(len(err)),
    }


def _useful_label(delta: float, ci_hi: float, useful_token: str, no_gain_token: str) -> str:
    return useful_token if (delta <= -0.002 and ci_hi < 0.0) else no_gain_token


def cmd_score_monitor(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    if MONITOR_LOCK.exists():
        raise SystemExit("Monitor labels already read in this round; refusing a second read")
    if not (OUT / "stage_a.json").exists() or not read_json(OUT / "stage_a.json").get("ok", False):
        raise SystemExit("Stage-A gate is not green; refusing the Monitor read")
    manifest = _verify_prediction_manifest()
    y = _ensure_monitor_y()
    n = len(y)
    preds = {arm: _read_pred_npz(pred_path(arm), n) for arm in ARMS}
    table = {arm: _metrics(preds[arm], y) for arm in ARMS}
    for arm in ARMS:
        tj = train_json_path(arm)
        if tj.exists():
            t = read_json(tj)
            table[arm]["fit_mae"] = float(t["soup_fit_mae"])
            table[arm]["training_wall_seconds"] = float(t["wall_seconds"])
            table[arm]["updates"] = int(t["updates"])
        else:
            table[arm]["fit_mae"] = None
            table[arm]["training_wall_seconds"] = None
            table[arm]["updates"] = None
    contrasts: dict[str, Any] = {}
    for new_arm, ref_arm, key in (("A1", "A0", "delta_10"), ("A2", "A1", "delta_21"), ("A2", "A0", "delta_20")):
        c = ab.compute_contrast({SEED: preds[new_arm]}, {SEED: preds[ref_arm]}, y)
        c["mae_new"] = float(table[new_arm]["mae"])
        c["mae_ref"] = float(table[ref_arm]["mae"])
        contrasts[key] = c
    tokens_a = [
        _useful_label(
            contrasts["delta_10"]["delta_seed_mean"],
            contrasts["delta_10"]["bootstrap"]["ci_hi"],
            "J_MOMENTS_USEFUL_EXPLORATORY",
            "J_MOMENTS_NO_CLEAR_GAIN",
        ),
        _useful_label(
            contrasts["delta_21"]["delta_seed_mean"],
            contrasts["delta_21"]["bootstrap"]["ci_hi"],
            "COUNT_CONDITIONING_USEFUL_EXPLORATORY",
            "COUNT_CONDITIONING_NO_CLEAR_GAIN",
        ),
    ]
    for key in ("delta_10", "delta_21", "delta_20"):
        c = contrasts[key]
        if abs(c["delta_seed_mean"]) <= 0.002 and c["bootstrap"]["ci_lo"] < 0.0 < c["bootstrap"]["ci_hi"]:
            tokens_a.append("INCONCLUSIVE_SINGLE_SEED")
            break
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": SPLIT_TAG,
        "seeds": [SEED],
        "n_monitor": n,
        "table": table,
        "contrasts": contrasts,
        "a_tokens": tokens_a,
        "gate": {
            "useful_rule": "delta <= -0.002 and paired bootstrap CI95 upper < 0",
            "bootstrap_resamples": int(bp.BOOTSTRAP_N),
            "bootstrap_unit": "molecule",
            "interpretation_matrix": {
                "delta_10_label": tokens_a[0],
                "delta_21_label": tokens_a[1],
            },
        },
        "prediction_manifest_sha256": file_sha256(PRED_MANIFEST),
        "monitor_read_utc": read_json(MONITOR_LOCK)["read_utc"],
        "exposure_disclosure": "historically exposed exploratory development data",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(MONITOR_SCORES, payload)
    log(f"[score-monitor] MAE A0={table['A0']['mae']:.6f} A1={table['A1']['mae']:.6f} A2={table['A2']['mae']:.6f}")
    for key, c in contrasts.items():
        log(
            f"  {key}: {c['delta_seed_mean']:+.6f} CI95 [{c['bootstrap']['ci_lo']:+.6f},{c['bootstrap']['ci_hi']:+.6f}] "
            f"ordinary {c['ordinary_delta']:+.6f} tail {c['tail_delta']:+.6f} top10 {c['top10_removed_delta']:+.6f}"
        )
    log(f"  tokens: {tokens_a}")
    return 0


# ---------------------------------------------------------------------------
# Part B: algebraic blindness + synthetic known truth
# ---------------------------------------------------------------------------


def cmd_b_algebraic(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    dev = torch.device(args.device)
    model = build_loaded_model("A2", dev)
    model.eval()
    out: dict[str, Any] = {"protocol": PROTOCOL_ID, "git_commit": git_commit()}

    # ---- B1: minimal scalar worked example ---------------------------------
    # 4 bonds (P,P,Q,Q), 4 junctions (X,X,Y,Y), every junction touches two bonds.
    # w values [1,1,2,2] on junctions X1,X2,Y1,Y2.
    w_scalar = np.asarray([1.0, 1.0, 2.0, 2.0])
    mean_w = float(w_scalar.mean())
    inc_a = [(0, 2), (0, 3), (1, 2), (1, 3)]
    inc_b = [(0, 2), (1, 3), (0, 3), (1, 2)]

    def per_bond_null(incidence, n_bonds=4):
        counts = np.zeros(n_bonds)
        for a, b in incidence:
            counts[a] += 1
            counts[b] += 1
        return counts * mean_w

    def per_bond_organized(incidence, n_bonds=4):
        W = np.zeros(n_bonds)
        for j, (a, b) in enumerate(incidence):
            W[a] += w_scalar[j]
            W[b] += w_scalar[j]
        return W

    out["b1_worked_example"] = {
        "bonds": ["P1", "P2", "Q1", "Q2"],
        "junctions": ["X1", "X2", "Y1", "Y2"],
        "w_values": w_scalar.tolist(),
        "assignment_A": inc_a,
        "assignment_B": inc_b,
        "per_bond_null_A": per_bond_null(inc_a).tolist(),
        "per_bond_null_B": per_bond_null(inc_b).tolist(),
        "per_bond_organized_A": per_bond_organized(inc_a).tolist(),
        "per_bond_organized_B": per_bond_organized(inc_b).tolist(),
        "null_identical": bool(np.array_equal(per_bond_null(inc_a), per_bond_null(inc_b))),
        "organized_differs": bool(not np.array_equal(per_bond_organized(inc_a), per_bond_organized(inc_b))),
        "formula_null": "W_e = n_e * mean_{j in G}(w_j) ; V_e = n_e * mean_{j in G}(w_j^2) ; q_e = log1p(n_e)",
        "formula_organized": "W_e = sum_{j touches e} w_j ; V_e = sum_{j touches e} w_j^2",
        "statement": (
            "For fixed b_e, {n_e}, and the graph Junction multiset {w_j}, any reassignment of Junction "
            "content to bonds leaves the Null W_e/V_e unchanged; the Organized W_e/V_e change unless the "
            "reassignment is an automorphism of the incidence structure.  This is a representation-level "
            "(conditional) blindness statement, not a claim that arbitrary reassignments are realisable molecules."
        ),
    }

    # ---- B4 Test 2 + B5 on the historical object counterexample ------------
    a_inst, b_inst = src.counterexample_object_pair()
    batch = src.batch_object_instances([a_inst, b_inst])
    pair_records: dict[str, Any] = {}
    for mode in ARMS:
        feats = model.forward_features(batch, mode=jsm.ARM_MODES[mode])
        W = feats["W"].detach().cpu().numpy()
        V = feats["V"].detach().cpu().numpy()
        pred = model(batch, mode=jsm.ARM_MODES[mode]).detach().cpu().numpy()
        pair_records[mode] = {
            "W_max_abs_diff": float(np.abs(W[:4] - W[4:]).max()),
            "V_max_abs_diff": float(np.abs(V[:4] - V[4:]).max()),
            "pred_max_abs_diff": float(np.abs(pred[0] - pred[1])),
        }
    feats_org = model.forward_features(batch, organization_mode=src.ORGANIZED)
    pred_org = model(batch, organization_mode=src.ORGANIZED).detach().cpu().numpy()
    pair_records["ORGANIZED"] = {
        "W_max_abs_diff": float(np.abs(feats_org["W"].detach()[:4] - feats_org["W"].detach()[4:]).max()),
        "V_max_abs_diff": float(np.abs(feats_org["V"].detach()[:4] - feats_org["V"].detach()[4:]).max()),
        "pred_max_abs_diff": float(np.abs(pred_org[0] - pred_org[1])),
    }
    out["b4_test2_object_counterexample"] = {
        "instance_A_per_bond_junction_tokens": src.per_bond_junction_tokens(a_inst),
        "instance_B_per_bond_junction_tokens": src.per_bond_junction_tokens(b_inst),
        "marginal_signature_A": src.object_marginal_signature(a_inst),
        "marginal_signature_B": src.object_marginal_signature(b_inst),
        "target_marginal_A": a_inst.target_marginal,
        "target_marginal_B": b_inst.target_marginal,
        "target_incidence_A": a_inst.target_incidence,
        "target_incidence_B": b_inst.target_incidence,
        "records": pair_records,
        "reading": "A0/A1/A2 are invariant; the Organized grouping separates the two objects.  The switch happens after encoding.",
    }

    # ---- B5a scaled post-encoding ownership family --------------------------
    # Construction: m bond pairs, each bond touched by exactly two junctions;
    # every junction touches one bond from each pair and sees the same unordered
    # bond-type pair, so the encoded junction multiset is fixed; only the
    # ownership pattern changes.  m junction rows are type X and m are type Y in
    # both instances, so the encoded w multiset is exactly preserved.
    def _pair_instances(m: int, seed: int = 0) -> tuple[src.ObjectInstance, src.ObjectInstance]:
        n_slots = 2 * m
        slots_p = [j for j in range(m) for _ in range(2)]
        slots_q = [m + j for j in range(m) for _ in range(2)]
        junc_tokens = [0] * m + [1] * m
        bond_tokens = [0] * m + [1] * m
        inc_a = [(slots_p[i], slots_q[i]) for i in range(n_slots)]
        rng = np.random.default_rng(int(seed) + 1000 * m)
        for _ in range(200):
            pp = list(rng.permutation(slots_p))
            pq = list(rng.permutation(slots_q))
            inc_b = [(int(pp[i]), int(pq[i])) for i in range(n_slots)]
            ta = src.object_targets(bond_tokens, junc_tokens, inc_a)
            tb = src.object_targets(bond_tokens, junc_tokens, inc_b)
            if ta[0] != tb[0]:
                break
        inst_a = src.ObjectInstance(bond_tokens, junc_tokens, inc_a, 0.0, 0.0)
        inst_b = src.ObjectInstance(bond_tokens, junc_tokens, inc_b, 0.0, 0.0)
        inst_a.target_incidence, inst_a.target_marginal = src.object_targets(bond_tokens, junc_tokens, inc_a)
        inst_b.target_incidence, inst_b.target_marginal = src.object_targets(bond_tokens, junc_tokens, inc_b)
        return inst_a, inst_b

    family_pairs = [_pair_instances(m) for m in (2, 3, 4, 5)]
    fam_flat: list[src.ObjectInstance] = [inst for pair in family_pairs for inst in pair]
    fam_batch = src.batch_object_instances(fam_flat)
    fam_records: list[dict[str, Any]] = []
    with torch.no_grad():
        fam_null = model.forward_features(fam_batch)
        fam_org = model.forward_features(fam_batch, organization_mode=src.ORGANIZED)
    Wn = fam_null["W"].detach().numpy(); Vn = fam_null["V"].detach().numpy()
    Wo = fam_org["W"].detach().numpy(); Vo = fam_org["V"].detach().numpy()
    bb = fam_batch["bond_batch"].numpy()
    fam_pred = model(fam_batch).detach().numpy()
    for k, (ia, ib) in enumerate(family_pairs):
        g0, g1 = 2 * k, 2 * k + 1
        m0, m1 = bb == g0, bb == g1
        fam_records.append(
            {
                "m": k + 2,
                "null_W_max_abs_diff": float(np.abs(Wn[m0] - Wn[m1]).max()),
                "null_V_max_abs_diff": float(np.abs(Vn[m0] - Vn[m1]).max()),
                "null_pred_abs_diff": float(abs(fam_pred[g0] - fam_pred[g1])),
                "organized_W_max_abs_diff": float(np.abs(Wo[m0] - Wo[m1]).max()),
                "organized_V_max_abs_diff": float(np.abs(Vo[m0] - Vo[m1]).max()),
                "organized_pred_abs_diff": float(abs(model(fam_batch, organization_mode=src.ORGANIZED).detach().numpy()[g0] - model(fam_batch, organization_mode=src.ORGANIZED).detach().numpy()[g1])),
                "target_marginal_abs_diff": float(abs(ia.target_marginal - ib.target_marginal)),
                "target_incidence_abs_diff": float(abs(ia.target_incidence - ib.target_incidence)),
                "marginal_signature_equal": src.object_marginal_signature(ia) == src.object_marginal_signature(ib),
            }
        )
    out["b5a_scaled_ownership_family"] = {
        "records": fam_records,
        "reading": (
            "Post-encoding ownership switch with a fixed Junction embedding multiset and fixed per-bond "
            "counts: Null W/V/prediction are invariant; Organized W/V differ and the ownership target "
            "changes while the marginal target does not.  Deterministic, no training."
        ),
    }

    # ---- B5b raw marginal signatures do not fix the encoded w multiset -------
    # Same raw bond/junction token multisets and counts, different pairing -> w
    # can differ because f_J consumes the two incident bond embeddings.  This
    # measures how much ownership leaks into the 'content' before pooling.
    rng = np.random.default_rng(0)
    instances = [src.random_object_instance(rng, n_bonds=6, n_junc=6) for _ in range(400)]
    groups: dict[Any, list[int]] = defaultdict(list)
    for i, inst in enumerate(instances):
        groups[src.object_marginal_signature(inst)].append(i)
    big = src.batch_object_instances(instances)
    with torch.no_grad():
        feats_null = model.forward_features(big)
    W = feats_null["W"].detach().numpy()
    V = feats_null["V"].detach().numpy()
    bond_batch = big["bond_batch"].numpy()
    per_graph_W: list[tuple[float, float, float, float]] = []
    for g in range(len(instances)):
        m = bond_batch == g
        per_graph_W.append((float(W[m].sum()), float((W[m] ** 2).sum()), float(V[m].sum()), float((V[m] ** 2).sum())))
    within: list[dict[str, Any]] = []
    for key, idxs in groups.items():
        if len(idxs) < 2:
            continue
        for ii in range(len(idxs)):
            for jj in range(ii + 1, len(idxs)):
                i, j = idxs[ii], idxs[jj]
                wi, wj = np.asarray(per_graph_W[i]), np.asarray(per_graph_W[j])
                within.append(
                    {
                        "i": int(i),
                        "j": int(j),
                        "null_summary_max_abs_diff": float(np.abs(wi - wj).max()),
                        "target_marginal_diff": float(abs(instances[i].target_marginal - instances[j].target_marginal)),
                        "target_incidence_diff": float(abs(instances[i].target_incidence - instances[j].target_incidence)),
                        "incidence_differs": bool(instances[i].incidence != instances[j].incidence),
                    }
                )
    differing = [r for r in within if r["incidence_differs"]]
    out["b5b_raw_marginal_pairing_leak"] = {
        "n_instances": len(instances),
        "n_marginal_groups": len(groups),
        "n_groups_size_ge2": int(sum(1 for v in groups.values() if len(v) >= 2)),
        "n_within_group_pairs": len(within),
        "n_within_group_pairs_different_incidence": len(differing),
        "null_summary_max_abs_diff_over_all_pairs": float(max((r["null_summary_max_abs_diff"] for r in within), default=0.0)),
        "target_incidence_diff_max": float(max((r["target_incidence_diff"] for r in differing), default=0.0)),
        "target_marginal_diff_max": float(max((r["target_marginal_diff"] for r in within), default=0.0)),
        "examples": sorted(differing, key=lambda r: -r["target_incidence_diff"])[:5],
        "reading": (
            "Ownership is not fully erased at the raw-token level: f_J consumes the two incident bond "
            "embeddings, so the encoded w multiset can depend on which bonds a junction touches.  "
            "The exact algebraic blindness of B1 is conditional on the encoded w multiset being fixed."
        ),
    }
    write_json(OUT / "b1_algebraic.json", out)
    b5_payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "b1_worked_example": out["b1_worked_example"],
        "b4_test2": out["b4_test2_object_counterexample"],
        "b5a_scaled_ownership_family": out["b5a_scaled_ownership_family"],
        "b5b_raw_marginal_pairing_leak": {k: v for k, v in out["b5b_raw_marginal_pairing_leak"].items() if k != "examples"},
    }
    write_json(OUT / "b5_synthetic.json", b5_payload)
    log("[b-algebraic] written")
    return 0


# ---------------------------------------------------------------------------
# Part B: layer-wise representation audit
# ---------------------------------------------------------------------------


def _block_stats(X: np.ndarray) -> dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    mean = X.mean(0)
    std = X.std(0)
    Xc = X - mean
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s / max(float(s.sum()), 1e-30)
    erank = float(np.exp(-np.sum(p * np.log(p + 1e-30))))
    cvar = np.cumsum(s**2) / max(float((s**2).sum()), 1e-30)
    d90 = int(np.searchsorted(cvar, 0.90) + 1)
    d99 = int(np.searchsorted(cvar, 0.99) + 1)
    return {
        "shape": [int(n), int(d)],
        "mean_abs": float(np.abs(mean).mean()),
        "std_mean": float(std.mean()),
        "constant_column_fraction": float((std < 1e-12).mean()),
        "effective_rank": erank,
        "effective_rank_fraction": float(erank / max(d, 1)),
        "pca_dims_90pct": d90,
        "pca_dims_99pct": d99,
    }


def _ridge_r2(Y: np.ndarray, X: np.ndarray, alpha: float = 1.0) -> float:
    """Mean R^2 of ridge regressions from X to each column of Y (fit on all rows)."""
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    X = np.hstack([X, np.ones((len(X), 1))])
    A = X.T @ X + alpha * np.eye(X.shape[1])
    B = np.linalg.solve(A, X.T @ Y)
    pred = X @ B
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-30
    return float(np.mean(1.0 - ss_res / ss_tot))


def _extract_levels(
    model: jsm.BJOCSufficiency,
    objs: Sequence[sro.MolObjects],
    device: torch.device | str,
    mode: str,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    dev = torch.device(device)
    n = len(objs)
    ao, bo, jo = int(model.config.atom_out), int(model.config.bond_out), int(model.config.junc_out)
    a_sum = np.zeros((n, ao)); a_sq = np.zeros((n, ao))
    b_sum = np.zeros((n, bo)); b_sq = np.zeros((n, bo))
    w_sum = np.zeros((n, jo)); w_sq = np.zeros((n, jo))
    z = np.zeros((n, int(model.config.compose_out)))
    reader = np.zeros((n, int(model.config.reader_in)))
    W_all, V_all, n_all = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk = list(objs[i : i + batch_size])
            batch = bp._mv(sro.batch_objects(chunk), dev)
            feats = model.forward_features(batch, mode=mode)
            sl = slice(i, i + len(chunk))
            a_sum[sl] = _index_add_np(feats["a"], batch["atom_batch"], len(chunk))
            a_sq[sl] = _index_add_np(feats["a"] * feats["a"], batch["atom_batch"], len(chunk))
            b_sum[sl] = _index_add_np(feats["b"], batch["bond_batch"], len(chunk))
            b_sq[sl] = _index_add_np(feats["b"] * feats["b"], batch["bond_batch"], len(chunk))
            w_sum[sl] = _index_add_np(feats["w"], batch["junc_batch"], len(chunk))
            w_sq[sl] = _index_add_np(feats["w"] * feats["w"], batch["junc_batch"], len(chunk))
            z[sl] = feats["z"].detach().cpu().numpy()
            reader[sl] = feats["reader_in"].detach().cpu().numpy()
            W_all.append(feats["W"].detach().cpu().numpy())
            V_all.append(feats["V"].detach().cpu().numpy())
            n_all.append(feats["n_b"].detach().cpu().numpy())
    return {
        "a_sum": a_sum, "a_sq": a_sq, "b_sum": b_sum, "b_sq": b_sq, "w_sum": w_sum, "w_sq": w_sq,
        "z": z, "reader_in": reader,
        "W": np.concatenate(W_all, 0), "V": np.concatenate(V_all, 0), "n_b": np.concatenate(n_all, 0),
    }


def _index_add_np(x: torch.Tensor, index: torch.Tensor, n: int) -> np.ndarray:
    out = torch.zeros((n, x.shape[1]), device=x.device, dtype=x.dtype)
    out.index_add_(0, index, x)
    return out.detach().cpu().numpy()


def _level0_features(objs: Sequence[sro.MolObjects]) -> dict[str, np.ndarray]:
    n = len(objs)
    g = np.stack([o.global_feat for o in objs], 0).astype(np.float64)
    atom = np.stack([o.atom_feat[:, :28].sum(0) for o in objs], 0).astype(np.float64)
    deg = np.stack([o.atom_feat[:, 28:33].sum(0) for o in objs], 0).astype(np.float64)
    bond = np.stack([np.bincount(o.bond_type, minlength=4) for o in objs], 0).astype(np.float64)
    # degree 0..4 x endpoints-bonded x n_ring_bonds 0..2 x centre-in-ring
    junc = np.zeros((n, 5 * 2 * 3 * 2), dtype=np.float64)
    for i, o in enumerate(objs):
        for row in o.junc_feat:
            d = min(int(row[0]), 4)
            endpoints_bonded = int(row[1] > 0)
            ring_bonds = min(int(row[3]), 2)
            centre_in_ring = int(row[5] > 0)
            idx = ((d * 2 + endpoints_bonded) * 3 + ring_bonds) * 2 + centre_in_ring
            junc[i, idx] += 1.0
    counts = np.zeros((n, 7), dtype=np.float64)
    for i, o in enumerate(objs):
        c = np.bincount(o.junc_bond.reshape(-1), minlength=o.bond_index.shape[0])
        counts[i, : min(len(np.bincount(c, minlength=7)), 7)] = np.bincount(c, minlength=7)[:7]
    return {"global": g, "atom_hist": atom, "degree_hist": deg, "bond_hist": bond, "junction_desc_hist": junc, "per_bond_count_hist": counts}


def cmd_b_layerwise(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    dev = torch.device(args.device)
    split = load_split()
    fit_objs = load_objects("fit")
    mon_objs = load_objects("monitor")
    check_object_rows(fit_objs, split["fit_rows"], "fit")
    check_object_rows(mon_objs, split["monitor_rows"], "monitor")
    fit_sel = fit_objs[:2000]
    subset = list(fit_sel) + list(mon_objs)
    out: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "subset": {"fit_rows_first": 2000, "monitor_rows": 1000, "n_total": len(subset), "label_blind": True},
    }
    level0 = _level0_features(subset)
    out["level0"] = {k: _block_stats(v) for k, v in level0.items()}

    modes_rep: dict[str, Any] = {}
    p3 = build_loaded_model("A2", dev)
    for mode in ARMS:
        feats = _extract_levels(p3, subset, dev, mode=jsm.ARM_MODES[mode])
        modes_rep[mode] = feats
    out["representation_only_p3_weights"] = {}
    for mode, feats in modes_rep.items():
        counts = np.asarray([o.atom_feat.shape[0] for o in subset], dtype=np.float64)
        bcounts = np.asarray([o.bond_index.shape[0] for o in subset], dtype=np.float64)
        jcounts = np.asarray([o.junc_index.shape[0] for o in subset], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            a_mean = feats["a_sum"] / counts[:, None].clip(min=1)
            a_var = feats["a_sq"] / counts[:, None].clip(min=1) - a_mean**2
            b_mean = feats["b_sum"] / bcounts[:, None].clip(min=1)
            b_var = feats["b_sq"] / bcounts[:, None].clip(min=1) - b_mean**2
            w_mean = feats["w_sum"] / jcounts[:, None].clip(min=1)
            w_var = feats["w_sq"] / jcounts[:, None].clip(min=1) - w_mean**2
        blocks = {
            "atom_mean": a_mean,
            "atom_var": a_var,
            "bond_mean": b_mean,
            "bond_var": b_var,
            "junction_mean": w_mean,
            "junction_var": w_var,
            "composer_sum": feats["z"],
        }
        rec = {k: _block_stats(v) for k, v in blocks.items()}
        rec["reader_blocks"] = {
            "composer_sum": _block_stats(feats["reader_in"][:, :96]),
            "atom_sum": _block_stats(feats["reader_in"][:, 96:192]),
            "atom_sq_sum": _block_stats(feats["reader_in"][:, 192:288]),
            "global": _block_stats(feats["reader_in"][:, 288:]),
        }
        zblock = feats["reader_in"][:, :96]
        ablock = feats["reader_in"][:, 96:192]
        asqblock = feats["reader_in"][:, 192:288]
        gblock = feats["reader_in"][:, 288:]
        rec["reader_block_ridge_r2"] = {
            "composer_sum_from_rest": _ridge_r2(zblock, np.hstack([ablock, asqblock, gblock])),
            "atom_sum_from_rest": _ridge_r2(ablock, np.hstack([zblock, asqblock, gblock])),
            "atom_sq_sum_from_rest": _ridge_r2(asqblock, np.hstack([zblock, ablock, gblock])),
            "global_from_rest": _ridge_r2(gblock, np.hstack([zblock, ablock, asqblock])),
        }
        modes_rep[mode] = {"feats": feats, "summary": rec}
        out["representation_only_p3_weights"][mode] = rec

    # Fit-only standardised block distances
    fit_mask = np.zeros(len(subset), dtype=bool)
    fit_mask[: len(fit_sel)] = True
    blocks3 = {
        "composer_sum": modes_rep["A2"]["feats"]["reader_in"][:, :96],
        "atom_sum": modes_rep["A2"]["feats"]["reader_in"][:, 96:192],
        "atom_sq_sum": modes_rep["A2"]["feats"]["reader_in"][:, 192:288],
        "global": modes_rep["A2"]["feats"]["reader_in"][:, 288:],
    }
    dist_rec: dict[str, Any] = {}
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, len(fit_sel), size=(500, 2))
    for name, X in blocks3.items():
        mu = X[fit_mask].mean(0)
        sd = X[fit_mask].std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Xs = (X - mu) / sd
        d = np.linalg.norm(Xs[pairs[:, 0]] - Xs[pairs[:, 1]], axis=1)
        dist_rec[name] = {
            "quantiles": {str(q): float(np.quantile(d, q)) for q in (0.0, 0.01, 0.05, 0.5, 0.95, 1.0)},
            "mean": float(d.mean()),
        }
    out["fit_standardised_pair_distances"] = dist_rec

    # system-level: trained soups under their own modes
    trained: dict[str, Any] = {}
    for arm in ARMS:
        if not soup_path(arm).exists():
            continue
        m = build_loaded_model(arm, dev)
        feats = _extract_levels(m, subset, dev, mode=jsm.ARM_MODES[arm])
        rec = {
            "composer_sum": _block_stats(feats["z"]),
            "reader_blocks": {
                "composer_sum": _block_stats(feats["reader_in"][:, :96]),
                "atom_sum": _block_stats(feats["reader_in"][:, 96:192]),
                "atom_sq_sum": _block_stats(feats["reader_in"][:, 192:288]),
                "global": _block_stats(feats["reader_in"][:, 288:]),
            },
        }
        trained[arm] = rec
    out["trained_models"] = trained

    # B4 Test 1: junction row permutation on the trained models
    small = list(fit_sel[:16])
    perm = _permute_junctions(small, seed=3)
    test1: dict[str, Any] = {}
    for arm in ARMS:
        if not soup_path(arm).exists():
            continue
        m = build_loaded_model(arm, dev)
        test1[arm] = {
            "max_abs_pred_diff": float(np.abs(predict_objects(m, small, dev) - predict_objects(m, perm, dev)).max()),
        }
    out["b4_test1_junction_row_permutation"] = {
        "records": test1,
        "reading": "known invariance: the Null grouping uses permutation-invariant moments; re-checked, not a new finding",
    }

    # B4 Test 4: block neutralisation on the frozen P3 model
    neutral: dict[str, Any] = {}
    base_pred = predict_objects(p3, small, dev)
    for block, sl in (("composer_sum", slice(0, 96)), ("atom_sum", slice(96, 192)), ("atom_sq_sum", slice(192, 288)), ("global", slice(288, None))):
        batch = bp._mv(sro.batch_objects(small), dev)
        with torch.no_grad():
            feats = p3.forward_features(batch)
            r = feats["reader_in"].clone()
            r[:, sl] = 0.0
            pred = p3.reader(r).view(-1).detach().cpu().numpy()
        neutral[block] = {
            "mean_abs_change": float(np.abs(pred - base_pred).mean()),
            "median_abs_change": float(np.median(np.abs(pred - base_pred))),
            "max_abs_change": float(np.abs(pred - base_pred).max()),
        }
    out["b4_test4_block_neutralisation"] = {
        "records": neutral,
        "caveat": "zeroing a block feeds the reader an out-of-distribution input; a sensitivity diagnostic, not an attribution",
    }
    # store raw per-molecule arrays for audit (ignored by git)
    np.savez_compressed(
        OUT / "b2_level_arrays.npz",
        reader_in_p3_a2=modes_rep["A2"]["feats"]["reader_in"],
        z_p3_a2=modes_rep["A2"]["feats"]["z"],
        fit_mask=fit_mask,
    )
    write_json(OUT / "b2_layerwise.json", out)
    log("[b-layerwise] written")
    return 0


# ---------------------------------------------------------------------------
# Part B: real-molecule structure collision search
# ---------------------------------------------------------------------------


def _structure_signatures(objs: Sequence[sro.MolObjects]) -> dict[str, np.ndarray]:
    n = len(objs)
    g = np.stack([o.global_feat for o in objs], 0).astype(np.float64)
    atom = np.zeros((n, 28)); bond = np.zeros((n, 4)); deg = np.zeros((n, 5)); cnt = np.zeros((n, 7))
    for i, o in enumerate(objs):
        atom[i] = o.atom_feat[:, :28].sum(0)
        bond[i] = np.bincount(o.bond_type, minlength=4)
        deg[i] = o.atom_feat[:, 28:33].sum(0)
        c = np.bincount(o.junc_bond.reshape(-1), minlength=o.bond_index.shape[0])
        cnt[i] = np.bincount(c, minlength=7)[:7]
    return {"global": g, "atom_hist": atom, "bond_hist": bond, "degree_hist": deg, "count_hist": cnt}


def _exact_key(objs_i: Mapping[str, np.ndarray], i: int) -> bytes:
    parts = []
    for k in ("global", "atom_hist", "bond_hist", "degree_hist", "count_hist"):
        parts.append(np.ascontiguousarray(objs_i[k][i], dtype=np.float64).tobytes())
    return b"|".join(parts)


def _canonical_isomorphism_check(ds: Any, i: int, j: int) -> bool:
    """NetworkX GraphMatcher on the two raw molecule graphs (node/edge typed)."""
    import networkx as nx

    def graph_of(row: int):
        d = ds[row]
        x = d.x.numpy().reshape(-1)
        ei = d.edge_index.numpy()
        ea = d.edge_attr.numpy().reshape(-1)
        G = nx.Graph()
        for a in range(len(x)):
            G.add_node(a, t=int(x[a]))
        for k in range(ei.shape[1]):
            u, v = int(ei[0, k]), int(ei[1, k])
            if u < v:
                G.add_edge(u, v, t=int(ea[k]))
        return G

    return nx.is_isomorphic(graph_of(i), graph_of(j), node_match=lambda a, b: a["t"] == b["t"], edge_match=lambda a, b: a["t"] == b["t"])


def cmd_b_collisions(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    dev = torch.device(args.device)
    split = load_split()
    objs = list(load_objects("fit")) + list(load_objects("monitor"))
    if OBJECT_FILES["locked"].exists():
        objs = objs + list(load_objects("locked"))
    n_expected = int(len(split["fit_rows"]) + len(split["monitor_rows"]) + len(split["locked_rows"]))
    if len(objs) != n_expected:
        raise SystemExit(f"structural scan expected {n_expected} train structures, got {len(objs)}")
    row_ids = [int(r) for r in list(split["fit_rows"]) + list(split["monitor_rows"]) + list(split["locked_rows"])]
    bad = [i for i, o in enumerate(objs) if int(o.mol_index) != row_ids[i]]
    if bad:
        raise SystemExit(f"structural scan row alignment failed at positions {bad[:8]}")
    smiles = [str(cf.load_canonical_smiles()[r]) for r in row_ids]
    sig = _structure_signatures(objs)
    out: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "scope": "official train 10000 structure fields only (x / edge_index / edge_attr); no labels read",
        "n_structures": len(objs),
        "label_blind": True,
    }

    # exact collisions
    keys = [_exact_key(sig, i) for i in range(len(objs))]
    by_key: dict[bytes, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        by_key[k].append(i)
    exact_groups = [v for v in by_key.values() if len(v) > 1]
    exact_pairs: list[tuple[int, int]] = []
    same_smiles_pairs = 0
    for group in exact_groups:
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                if smiles[group[a]] != smiles[group[b]]:
                    exact_pairs.append((group[a], group[b]))
                else:
                    same_smiles_pairs += 1
    out["exact_collision_groups"] = len(exact_groups)
    out["exact_collision_pairs_different_smiles"] = len(exact_pairs)
    out["exact_collision_pairs_same_smiles"] = same_smiles_pairs
    out["row_alignment_verified"] = True
    t0 = time.perf_counter()
    ds = cf.load_official_train()
    nxn_checks: list[dict[str, Any]] = []
    for i, j in exact_pairs[:5]:
        iso = _canonical_isomorphism_check(ds, row_ids[i], row_ids[j]) if args.nx_check else None
        nxn_checks.append({"i": i, "j": j, "official_rows": [row_ids[i], row_ids[j]], "canonical_smiles_equal": smiles[i] == smiles[j], "networkx_isomorphic": iso})
    out["exact_pair_isomorphism_checks"] = nxn_checks

    # near candidates
    raw = np.hstack([sig["global"], sig["atom_hist"], sig["bond_hist"], sig["degree_hist"], sig["count_hist"]])
    mu = raw[: len(split["fit_rows"])].mean(0)
    sd = raw[: len(split["fit_rows"])].std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    X = (raw - mu) / sd
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=21, algorithm="auto").fit(X)
    dist, idx = nn.kneighbors(X)
    cand: dict[tuple[int, int], float] = {}
    for i in range(len(X)):
        for d, j in zip(dist[i, 1:], idx[i, 1:]):
            j = int(j)
            if smiles[i] == smiles[j]:
                continue
            a, b = (i, j) if i < j else (j, i)
            if a not in cand or d < cand[(a, b)]:
                cand[(a, b)] = float(d)
    ranked = sorted(cand.items(), key=lambda kv: kv[1])[:200]
    knn_scale = float(np.quantile(dist[:, 1:].reshape(-1), 0.005))
    out["near_search"] = {
        "knn_k": 21,
        "candidate_pairs": len(ranked),
        "knn_distance_0.5pct": knn_scale,
        "min_candidate_distance": float(ranked[0][1]) if ranked else None,
        "n_pairs_at_or_below_knn_0.5pct": int(sum(1 for _, d in ranked if d <= knn_scale)),
    }

    # representation/prediction comparison for the reported pairs
    p3 = build_loaded_model("A2", dev)
    fit_objs = objs[: len(split["fit_rows"])]
    fit_reader = _extract_levels(p3, fit_objs[:2000], dev, mode=jsm.ARM_MODES["A2"])["reader_in"]
    rmu = fit_reader.mean(0); rsd = fit_reader.std(0); rsd = np.where(rsd < 1e-12, 1.0, rsd)
    fit_reader_std = (fit_reader - rmu) / rsd
    # closest distinct-structure reader distance in the Fit subset (strict collision reference)
    rng = np.random.default_rng(0)
    fpairs = rng.integers(0, len(fit_reader), size=(20000, 2))
    fd = np.linalg.norm(fit_reader_std[fpairs[:, 0]] - fit_reader_std[fpairs[:, 1]], axis=1)
    fit_dist_p01 = float(np.quantile(fd, 0.01))
    out["fit_reader_distance_reference"] = {"p01": fit_dist_p01, "median": float(np.median(fd))}

    reported: list[dict[str, Any]] = []
    max_report = int(load_config()["structural_audit"]["collision_search"]["max_reported_pairs"])
    exact_set = {(i, j) for i, j in exact_pairs}
    to_report: list[tuple[int, int, float, bool]] = [(i, j, 0.0, True) for i, j in exact_pairs[:max_report]]
    for (i, j), d in ranked[:max_report]:
        if (i, j) in exact_set:
            continue
        if len(to_report) >= max_report:
            break
        to_report.append((i, j, float(d), False))
    for i, j, d, is_exact in to_report:
        oi, oj = objs[i], objs[j]
        pred_pair = predict_objects(p3, [oi, oj], dev)
        pred_solo = np.asarray([float(predict_objects(p3, [o], dev)[0]) for o in (oi, oj)])
        rr = _extract_levels(p3, [oi, oj], dev, mode=jsm.ARM_MODES["A2"])["reader_in"]
        reader_dist = float(np.linalg.norm((rr[0] - rmu) / rsd - (rr[1] - rmu) / rsd))
        reported.append(
            {
                "i": i,
                "j": j,
                "official_rows": [row_ids[i], row_ids[j]],
                "smiles": [smiles[i], smiles[j]],
                "exact_structural_statistics_collision": bool(is_exact),
                "signature_distance": d,
                "canonical_smiles_differ": smiles[i] != smiles[j],
                "n_atoms": [oi.n_atoms, oj.n_atoms],
                "n_bonds": [int(oi.bond_index.shape[0]), int(oj.bond_index.shape[0])],
                "n_junctions": [int(oi.junc_index.shape[0]), int(oj.junc_index.shape[0])],
                "reader_input_standardised_l2": reader_dist,
                "reader_input_raw_max_abs_diff": float(np.abs(rr[0] - rr[1]).max()),
                "pred_abs_diff": float(abs(pred_pair[0] - pred_pair[1])),
                "pred_abs_diff_singleton": float(abs(pred_solo[0] - pred_solo[1])),
            }
        )
    out["reported_pairs"] = reported
    exact_verified = bool(exact_pairs) and all(r["canonical_smiles_equal"] is False for r in nxn_checks)
    strict_collision = bool(
        exact_verified
        and any(r["exact_structural_statistics_collision"] and r["reader_input_standardised_l2"] <= fit_dist_p01 for r in reported)
    )
    out["verdict"] = "REAL_REPRESENTATION_COLLISION_FOUND" if strict_collision else "REAL_COLLISION_NOT_FOUND_WITHIN_BUDGET"
    out["verdict_rule"] = (
        "REAL_REPRESENTATION_COLLISION_FOUND requires an exact structural-statistics collision between "
        "different canonical SMILES AND a Fit-standardised reader-input distance at or below the 1st "
        "percentile of the Fit distinct-structure reader distance distribution.  Near neighbours are "
        "reported with their distances and are never called identical."
    )
    out["search_seconds"] = float(time.perf_counter() - t0)
    write_json(OUT / "b3_collisions.json", out)
    log(f"[b-collisions] exact_pairs={len(exact_pairs)} verdict={out['verdict']}")
    return 0


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------


def cmd_finalize(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    tokens: list[str] = []
    sections: dict[str, Any] = {"protocol": PROTOCOL_ID, "git_commit": git_commit()}
    if MONITOR_SCORES.exists():
        ms = read_json(MONITOR_SCORES)
        sections["A"] = {"table": ms["table"], "contrasts": ms["contrasts"], "tokens": ms["a_tokens"]}
        tokens.extend(ms["a_tokens"])
    if (OUT / "b1_algebraic.json").exists():
        b1 = read_json(OUT / "b1_algebraic.json")
        blind = bool(
            b1["b4_test2_object_counterexample"]["records"]["A0"]["W_max_abs_diff"] == 0.0
            and b1["b4_test2_object_counterexample"]["records"]["A1"]["W_max_abs_diff"] == 0.0
            and b1["b4_test2_object_counterexample"]["records"]["A2"]["W_max_abs_diff"] == 0.0
            and b1["b4_test2_object_counterexample"]["records"]["ORGANIZED"]["W_max_abs_diff"] > 0.0
        )
        sections["B1_B4_B5"] = {
            "null_ownership_blindness_verified": blind,
            "worked_example_null_identical": b1["b1_worked_example"]["null_identical"],
            "worked_example_organized_differs": b1["b1_worked_example"]["organized_differs"],
            "b5a_scaled_family_max_null_W_diff": max(r["null_W_max_abs_diff"] for r in b1["b5a_scaled_ownership_family"]["records"]),
            "b5a_scaled_family_min_organized_W_diff": min(r["organized_W_max_abs_diff"] for r in b1["b5a_scaled_ownership_family"]["records"]),
            "b5a_scaled_family_min_target_incidence_diff": min(r["target_incidence_abs_diff"] for r in b1["b5a_scaled_ownership_family"]["records"]),
            "b5b_raw_marginal_null_summary_max_abs_diff": b1["b5b_raw_marginal_pairing_leak"]["null_summary_max_abs_diff_over_all_pairs"],
            "b5b_raw_marginal_pairs": b1["b5b_raw_marginal_pairing_leak"]["n_within_group_pairs_different_incidence"],
        }
        family_blind = bool(
            b1["b5a_scaled_ownership_family"]["records"]
            and all(r["null_W_max_abs_diff"] == 0.0 and r["null_V_max_abs_diff"] == 0.0 for r in b1["b5a_scaled_ownership_family"]["records"])
            and all(r["organized_W_max_abs_diff"] > 0.0 for r in b1["b5a_scaled_ownership_family"]["records"])
        )
        sections["B1_B4_B5"]["scaled_family_blind"] = family_blind
        if blind and family_blind:
            tokens.append("NULL_OWNERSHIP_BLINDNESS_VERIFIED")
    if (OUT / "b2_layerwise.json").exists():
        b2 = read_json(OUT / "b2_layerwise.json")
        a2 = b2["representation_only_p3_weights"]["A2"]
        sections["B2"] = {
            "subset": b2["subset"],
            "fit_standardised_pair_distances": b2["fit_standardised_pair_distances"],
            "a2_reader_block_ridge_r2": a2["reader_block_ridge_r2"],
            "a2_junction_block_rank_fraction": a2["junction_mean"]["effective_rank_fraction"],
            "b4_test1_junction_row_permutation": b2.get("b4_test1_junction_row_permutation", {}).get("records", {}),
            "b4_test4_block_neutralisation": {k: v["mean_abs_change"] for k, v in b2.get("b4_test4_block_neutralisation", {}).get("records", {}).items()},
            "trained_models_available": sorted(b2.get("trained_models", {}).keys()),
        }
    if (OUT / "b3_collisions.json").exists():
        b3 = read_json(OUT / "b3_collisions.json")
        sections["B3"] = {
            "exact_collision_groups": b3["exact_collision_groups"],
            "exact_collision_pairs_different_smiles": b3["exact_collision_pairs_different_smiles"],
            "exact_collision_pairs_same_smiles": b3["exact_collision_pairs_same_smiles"],
            "n_structures_scanned": b3["n_structures"],
            "min_reported_distinct_reader_standardised_l2": min(
                (r["reader_input_standardised_l2"] for r in b3["reported_pairs"] if r["canonical_smiles_differ"]),
                default=None,
            ),
            "verdict": b3["verdict"],
            "n_reported_pairs": len(b3["reported_pairs"]),
        }
        tokens.append(b3["verdict"])
    sections["tokens"] = tokens
    sections["locked_read_authorized"] = False
    sections["official_valid_loaded"] = False
    sections["official_test_loaded"] = False
    sections["evidence_boundary"] = (
        "single seed; historically exposed split-D Monitor; decision words describe this round's "
        "observations, not population-level claims"
    )
    write_json(DECISION, sections)
    log(f"[finalize] tokens={tokens}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu", help="torch device (cpu or cuda)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("verify-inputs")
    sub.add_parser("freeze")
    sub.add_parser("stage-a")
    ga = sub.add_parser("grad-audit")
    ga.add_argument("--phase", action="append", choices=["init", "trained"], default=None)
    tr = sub.add_parser("train")
    tr.add_argument("--arm", required=True, choices=["A0", "A1"])
    tr.add_argument("--seed", type=int, default=SEED)
    tr.add_argument("--smoke-epochs", type=int, default=None)
    pr = sub.add_parser("predict")
    pr.add_argument("--arms", nargs="*", default=None)
    sub.add_parser("score-monitor")
    sub.add_parser("b-algebraic")
    sub.add_parser("b-layerwise")
    bcol = sub.add_parser("b-collisions")
    bcol.add_argument("--no-nx-check", dest="nx_check", action="store_false", default=True)
    sub.add_parser("finalize")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    dispatch = {
        "verify-inputs": cmd_verify_inputs,
        "freeze": cmd_freeze,
        "stage-a": cmd_stage_a,
        "grad-audit": cmd_grad_audit,
        "train": cmd_train,
        "predict": cmd_predict,
        "score-monitor": cmd_score_monitor,
        "b-algebraic": cmd_b_algebraic,
        "b-layerwise": cmd_b_layerwise,
        "b-collisions": cmd_b_collisions,
        "finalize": cmd_finalize,
    }
    return int(dispatch[args.command](args))


if __name__ == "__main__":
    raise SystemExit(main())
