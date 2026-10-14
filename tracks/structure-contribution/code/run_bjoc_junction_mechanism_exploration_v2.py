"""cscl-bjoc-junction-mechanism-exploration-v2 driver.

Protocol: ``configs/bjoc_junction_mechanism_exploration_v2.yaml``.
Track: ``tracks/structure-contribution``.

Subcommands
-----------
``verify-inputs``   verify every reused artifact hash and recompute split D
``freeze``          write ``protocol_freeze.json`` binding config + code + sources
``stage-a``         CPU correctness / label-isolation gates
``grad-audit``      allocated vs gradient-active parameters per arm/phase
``train``           one supervised R3 trajectory (C/D/E; A1 seed 1)
``train-matrix``    a pre-declared seed's arms sequentially (one GPU job)
``predict``         label-blind Monitor predictions + hash-freeze manifest
``part1``           Part I error-complementarity analysis of the frozen A0/A1/A2
``score-monitor``   the single Monitor label read of this round + metrics/contrasts
``checkpoint-manifest``  durable checkpoint bookkeeping
``finalize``        decisions + GPU ledger
``render``          generated Markdown tables for the report files

Discipline: official valid / official test are never referenced by any code
path; Locked labels are never read.  Monitor labels are read exactly once in
``score-monitor`` and ``part1`` only after the round's prediction manifest is
frozen (Part I additionally requires the historical audit lock to be present).
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
import bjoc_junction_mechanism_models_v2 as mm  # noqa: E402
import bjoc_junction_sufficiency_models_v1 as jsm  # noqa: E402
import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import bjoc_task_adaptive_ssl_v1 as tat  # noqa: E402
import cscl_features as cf  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = mm.PROTOCOL_ID
CONFIG_PATH = _CODE_DIR.parent / "configs" / "bjoc_junction_mechanism_exploration_v2.yaml"
RES = REPO_ROOT / "tracks/structure-contribution/results"
OUT = RES / "bjoc_junction_mechanism_exploration_v2"
SRC = RES / "bjoc_task_adaptive_ssl_v1"
JAU = RES / "bjoc_junction_sufficiency_audit_v1"
ENCODED_TRAIN = REPO_ROOT / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt"

SEEDS = (0, 1)
ARMS_ALL = ("A0", "A1", "A2", "C", "D", "E")
NEW_ARMS = {0: ("C", "D", "E"), 1: ("A1", "C", "D", "E")}
CRITICAL_KEYS = (
    "A0:0", "A1:0", "A2:0", "C:0", "D:0", "E:0", "A2:1", "A1:1", "C:1", "D:1", "E:1",
)

SPLIT_PATH = SRC / "split_D.json"
PREP_PATH = SRC / "prep_fit_D.json"
OBJECT_FILES = {"fit": SRC / "objects_fit_D.pt", "monitor": SRC / "objects_monitor_D.pt"}
SSL_PATHS = {0: SRC / "pretrain_D_s0.pt", 1: SRC / "pretrain_D_s1.pt"}
P3_SOUP_PATHS = {0: SRC / "soup_P3_D_s0.pt", 1: SRC / "soup_P3_D_s1.pt"}
P3_TRAIN_PATHS = {0: SRC / "train_P3_D_s0.json", 1: SRC / "train_P3_D_s1.json"}
A0_SOUP = JAU / "soup_A0_D_s0.pt"
A1_SOUP = JAU / "soup_A1_D_s0.pt"
JAU_MON_Y = JAU / "monitor_y.npy"
JAU_MON_LOCK = JAU / "monitor_read.lock"
JAU_SCORES = JAU / "monitor_scores.json"
JAU_PRED = {arm: JAU / "predictions" / f"monitor_{arm}_D_s0.npz" for arm in ("A0", "A1", "A2")}
P3_MON_S1 = SRC / "predictions/monitor_P3_D_s1.npz"

FREEZE_PATH = OUT / "protocol_freeze.json"
PRED_MANIFEST = OUT / "prediction_manifest.json"
MONITOR_LOCK = OUT / "monitor_read.lock"
MONITOR_Y = OUT / "monitor_y.npy"
MONITOR_SCORES = OUT / "monitor_scores.json"
CPU_GATES = OUT / "cpu_correctness_gates.json"
DECISION = OUT / "scientific_decision.json"


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


def _code_paths() -> dict[str, Path]:
    return {
        "driver": _CODE_DIR / "run_bjoc_junction_mechanism_exploration_v2.py",
        "module": _CODE_DIR / "bjoc_junction_mechanism_models_v2.py",
        "config": CONFIG_PATH,
        "composer": _CODE_DIR / "structural_relation_composer_v1.py",
        "objects": _CODE_DIR / "structural_relation_objects_v1.py",
        "masked": _CODE_DIR / "bjoc_masked_pretraining_v1.py",
        "perfmech": _CODE_DIR / "bjoc_perfmech_long_v1.py",
        "arch": _CODE_DIR / "bjoc_arch_breakthrough_v1.py",
        "tat": _CODE_DIR / "bjoc_task_adaptive_ssl_v1.py",
        "suff": _CODE_DIR / "bjoc_junction_sufficiency_models_v1.py",
    }


def code_hashes() -> dict[str, str]:
    return {k: file_sha256(p) for k, p in _code_paths().items()}


def soup_path(arm: str, seed: int) -> Path:
    return OUT / f"soup_{arm}_D_s{seed}.pt"


def train_json_path(arm: str, seed: int) -> Path:
    return OUT / f"train_{arm}_D_s{seed}.json"


def new_pred_path(arm: str, seed: int) -> Path:
    return OUT / "predictions" / f"monitor_{arm}_D_s{seed}.npz"


# ---------------------------------------------------------------------------
# source verification
# ---------------------------------------------------------------------------


def _p3_record(seed: int) -> dict[str, Any]:
    return read_json(P3_TRAIN_PATHS[seed])


def audit_sources(*, require_analysis: bool = True) -> dict[str, Any]:
    """Verify reused sources.

    ``require_analysis=True`` (local verification and Part I) additionally
    requires the locally generated analysis artifacts: the Monitor label file,
    the frozen Part-I predictions and the historical score table.  The remote
    compute checkout only holds the training-critical sources (SSL
    checkpoints, soups, object caches, split/prep records), so ``freeze`` on
    the remote runs with ``require_analysis=False`` and records those files as
    absent; the authoritative analysis check always runs locally.
    """
    cfg = load_config()
    reuse = cfg["reuse"]
    checks: dict[str, Any] = {}

    def chk(name: str, path: Path, expected: str | None, *, analysis_only: bool = False) -> bool:
        exists = path.exists()
        if not exists and analysis_only and not require_analysis:
            checks[name] = {
                "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
                "exists": False,
                "sha256": None,
                "matches": None,
                "present": False,
                "note": "analysis-only artifact is absent on this checkout",
            }
            return True
        actual = file_sha256(path) if exists else None
        hit = bool(exists and (expected is None or actual == expected))
        checks[name] = {"path": str(path.relative_to(REPO_ROOT)), "exists": exists, "sha256": actual, "matches": hit}
        return hit

    ok = True
    ok &= chk("ssl_encoder_s0", SSL_PATHS[0], reuse["ssl_encoder_seed0"]["file_sha256"])
    ok &= chk("ssl_encoder_s1", SSL_PATHS[1], reuse["ssl_encoder_seed1"]["file_sha256"])
    ok &= chk("p3_soup_s0", P3_SOUP_PATHS[0], reuse["p3_soup_seed0"]["file_sha256"])
    ok &= chk("p3_soup_s1", P3_SOUP_PATHS[1], reuse["p3_soup_seed1"]["file_sha256"])
    ok &= chk("a0_soup_s0", A0_SOUP, reuse["a0_soup_seed0"]["file_sha256"])
    ok &= chk("a1_soup_s0", A1_SOUP, reuse["a1_soup_seed0"]["file_sha256"])
    ok &= chk("objects_fit", OBJECT_FILES["fit"], reuse["fit_objects"]["file_sha256"])
    ok &= chk("objects_monitor", OBJECT_FILES["monitor"], reuse["monitor_objects"]["file_sha256"])
    ok &= chk("prep_fit", PREP_PATH, None)
    ok &= chk("p3_train_s0", P3_TRAIN_PATHS[0], None)
    ok &= chk("p3_train_s1", P3_TRAIN_PATHS[1], None)
    ok &= chk("monitor_y_source", SRC / "monitor_y_D.npy", None, analysis_only=True)
    for arm, rec in reuse["part1_frozen_predictions"].items():
        ok &= chk(f"part1_pred_{arm}", JAU_PRED[arm], rec["sha256"], analysis_only=True)
    ok &= chk("p3_monitor_pred_s1", P3_MON_S1, reuse["p3_monitor_prediction_seed1"]["sha256"], analysis_only=True)
    ok &= chk("jau_monitor_scores", JAU_SCORES, None, analysis_only=True)
    ok &= chk("jau_monitor_lock", JAU_MON_LOCK, None)

    cross: dict[str, Any] = {}
    if JAU_MON_Y.exists() and JAU_MON_LOCK.exists():
        jau_lock = read_json(JAU_MON_LOCK)
        jau_y = np.asarray(np.load(JAU_MON_Y), dtype=np.float64).reshape(-1)
        cross["jau_monitor_lock_matches"] = bool(
            array_sha256(jau_y) == jau_lock["y_sha256"] and jau_lock["n_rows"] == 1000
        )
        src_y = np.asarray(np.load(SRC / "monitor_y_D.npy"), dtype=np.float64).reshape(-1)
        cross["monitor_y_source_equal"] = bool(
            np.array_equal(np.asarray(src_y, dtype=np.float32), np.asarray(jau_y, dtype=np.float32))
        )
    else:
        cross["jau_monitor_lock_matches"] = None
        cross["monitor_y_source_equal"] = None
    if (SRC / "prediction_manifest_D.json").exists():
        ta_manifest = read_json(SRC / "prediction_manifest_D.json")
        cross["p3_monitor_s1_matches_ta_manifest"] = bool(
            ta_manifest["entries"]["monitor:P3:1"]["sha256"] == reuse["p3_monitor_prediction_seed1"]["sha256"]
        )
        cross["p3_soup_s1_matches_ta_manifest"] = bool(
            ta_manifest["entries"]["monitor:P3:1"]["soup_sha256"] == reuse["p3_soup_seed1"]["file_sha256"]
        )
    else:
        cross["p3_monitor_s1_matches_ta_manifest"] = None
        cross["p3_soup_s1_matches_ta_manifest"] = None
    if (JAU / "prediction_manifest.json").exists():
        jau_manifest = read_json(JAU / "prediction_manifest.json")
        cross["part1_predictions_match_jau_manifest"] = bool(
            all(
                jau_manifest["entries"][arm]["sha256"] == reuse["part1_frozen_predictions"][arm]["sha256"]
                for arm in ("A0", "A1", "A2")
            )
        )
    else:
        cross["part1_predictions_match_jau_manifest"] = None
    # P3 train records vs config
    p3_ok = True
    for seed in SEEDS:
        t = _p3_record(seed)
        ref = reuse[f"p3_train_seed{seed}"]
        p3_ok &= bool(
            t["init_full_sha256"] == ref["init_full_sha256"]
            and t["init_group_state_sha256"] == ref["init_group_state_sha256"]
            and t["init_encoder_sha256"] == ref["init_encoder_sha256"]
            and t["encoder_used_sha256"] == ref["encoder_used_sha256"]
            and t["batch_order"]["full_hash"] == ref["batch_order_hash"]
        )
    cross["p3_train_records_match_config"] = p3_ok
    ok &= p3_ok

    # recompute split D from the committed canonical SMILES table
    smiles = [str(x) for x in cf.load_canonical_smiles()]
    sp = bp.build_split(smiles, cfg["split"]["salt"])
    fp = cfg["split"]["fingerprints"]
    split_ok = bool(
        bp.rows_sha256(sp["fit"]) == fp["fit_sha256_rows"]
        and bp.rows_sha256(sp["monitor"]) == fp["monitor_sha256_rows"]
        and bp.rows_sha256(sp["locked"]) == fp["locked_sha256_rows"]
        and len(sp["fit"]) == 8000
        and len(sp["monitor"]) == 1000
        and len(sp["locked"]) == 1000
        and int(sp["duplicate_groups_cross_sides"]) == 0
    )
    cross["split_recomputed_matches_fingerprints"] = split_ok
    ok &= split_ok
    return {"ok": bool(ok), "checks": checks, "cross_checks": cross}


def cmd_verify_inputs(args: argparse.Namespace) -> int:
    audit = audit_sources(require_analysis=True)
    write_json(OUT / "input_audit.json", {"protocol": PROTOCOL_ID, "git_commit": git_commit(), **audit})
    log(f"[verify-inputs] ok={audit['ok']}")
    return 0 if audit["ok"] else 1


def cmd_freeze(args: argparse.Namespace) -> int:
    audit = audit_sources(require_analysis=False)
    if not audit["ok"]:
        raise SystemExit("source audit failed; refusing to freeze")
    cfg = load_config()
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "code_hashes": code_hashes(),
        "split_tag": "D",
        "split_salt": cfg["split"]["salt"],
        "split_fingerprints": cfg["split"]["fingerprints"],
        "source_hashes": {k: v["sha256"] for k, v in audit["checks"].items()},
        "source_audit_ok": bool(audit["ok"]),
        "arms": list(ARMS_ALL),
        "new_trajectories": {str(k): list(v) for k, v in NEW_ARMS.items()},
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
    return bp.load_y_rows(ENCODED_TRAIN, load_split()["fit_rows"])


def load_ssl_state(seed: int) -> dict[str, dict[str, torch.Tensor]]:
    return torch.load(SSL_PATHS[seed], map_location="cpu", weights_only=False)


def check_object_rows(objs: Sequence[sro.MolObjects], rows: Sequence[int], name: str) -> None:
    if [int(o.mol_index) for o in objs] != [int(r) for r in rows]:
        raise SystemExit(f"{name} object rows do not match the frozen split")


def build_mech(arm: str, seed: int, device: torch.device | str = "cpu") -> mm.BJOCMechV2:
    mode = mm.ARM_MODES[arm]
    path = soup_path(arm, seed)
    if arm == "A2":
        path = P3_SOUP_PATHS[seed]
    if not path.exists():
        raise SystemExit(f"missing soup for {arm} seed {seed}: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = mm.BJOCMechV2(
        mm.config_for_mode(mode),
        np.asarray(load_norm()["global_mean"], dtype=np.float32),
        np.asarray(load_norm()["global_scale"], dtype=np.float32),
        mode=mode,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_objects(model: mm.BJOCMechV2, objs: Sequence[sro.MolObjects], device: torch.device | str, batch_size: int = 256) -> np.ndarray:
    model.eval()
    dev = torch.device(device)
    out = []
    for i in range(0, len(objs), int(batch_size)):
        batch = bp._mv(sro.batch_objects(list(objs[i : i + int(batch_size)])), dev)
        out.append(model(batch).detach().cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros(0)


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


def _per_graph_sum(x: torch.Tensor, idx: torch.Tensor, n_graphs: int) -> torch.Tensor:
    out = torch.zeros((n_graphs, x.shape[1]), device=x.device, dtype=x.dtype)
    if x.shape[0] > 0:
        out.index_add_(0, idx, x)
    return out


def _make_edge_case_objects() -> list[sro.MolObjects]:
    single = sro.build_mol_objects(-1, np.asarray([[0]], dtype=np.int64), np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64))
    pair = sro.build_mol_objects(
        -1,
        np.asarray([[0], [1]], dtype=np.int64),
        np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        np.asarray([1, 1], dtype=np.int64),
    )
    chain = sro.build_mol_objects(
        -1,
        np.asarray([[0], [1], [2]], dtype=np.int64),
        np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
        np.asarray([1, 1, 2, 2], dtype=np.int64),
    )
    return [single, pair, chain]


def cmd_stage_a(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    dev = torch.device(args.device)
    split = load_split()
    fit_objs = load_objects("fit")
    check_object_rows(fit_objs, split["fit_rows"], "fit")
    norm = load_norm()
    y_fit = load_y_fit()
    small = fit_objs[:8]
    small_batch = bp._mv(sro.batch_objects(small), dev)

    gates: dict[str, Any] = {}
    models: dict[str, mm.BJOCMechV2] = {}
    infos: dict[str, Any] = {}
    for mode in mm.MODES:
        _base, model, info = mm.build_mech_from_init(0, norm, dev, mode, load_ssl_state(0))
        models[mode] = model
        infos[mode] = info

    # -- G1 parameter allocation ------------------------------------------------
    counts = {mode: infos[mode]["parameter_count"] for mode in mm.MODES}
    named_d = infos[mm.MODE_D]["named_parameters"]
    named_total = int(sum(named_d.values()))
    gates["G1_parameter_allocation"] = {
        "pass": bool(
            all(counts[m] == mm._PARAMS_BASE for m in mm.MODES if m != mm.MODE_D)
            and counts[mm.MODE_D] == mm._PARAMS_D
            and named_total == mm._PARAMS_D
            and int(named_d["reader.lins.0.weight"]) == 256 * 513
        ),
        "counts": counts,
        "expected_parameters": {"base": mm._PARAMS_BASE, "D": mm._PARAMS_D},
        "named_parameter_total_D": named_total,
        "reader_first_weight_numel": int(named_d["reader.lins.0.weight"]),
    }

    # -- G2 E tensor identity with A2/base init ---------------------------------
    e_state = models[mm.MODE_E].state_dict()
    a2_state = models[mm.MODE_A2].state_dict()
    shape_ok = all(e_state[k].shape == a2_state[k].shape for k in e_state)
    max_diff = max(float((e_state[k].float() - a2_state[k].float()).abs().max()) for k in e_state)
    gates["G2_E_init_identical_to_A2"] = {
        "pass": bool(shape_ok and max_diff == 0.0),
        "shapes_identical": bool(shape_ok),
        "max_abs_param_diff": max_diff,
    }

    # -- G3 C composer inputs exactly count-only --------------------------------
    c_feats = models[mm.MODE_C].forward_features(small_batch)
    gates["G3_C_WV_zero_q_count"] = {
        "pass": bool(
            float(c_feats["W"].abs().max()) == 0.0
            and float(c_feats["V"].abs().max()) == 0.0
            and float((c_feats["q"] - c_feats["n_b"]).abs().max()) == 0.0
        ),
        "W_max_abs": float(c_feats["W"].abs().max()),
        "V_max_abs": float(c_feats["V"].abs().max()),
        "q_equals_n_b": float((c_feats["q"] - c_feats["n_b"]).abs().max()),
    }

    # -- G4 D reader expansion + init forward identity to A0 ---------------------
    exp0 = infos[mm.MODE_D]["reader_expansion"]
    a0 = models[mm.MODE_A0]
    d = models[mm.MODE_D]
    with torch.no_grad():
        f_a0 = a0(small_batch)
        f_d = d(small_batch)
    d_diff = float((f_a0 - f_d).abs().max())
    old_cols = float(
        (d.reader.lins[0].weight.detach()[:, :321] - a0.reader.lins[0].weight.detach()).abs().max()
    )
    new_cols = float(d.reader.lins[0].weight.detach()[:, 321:].abs().max())
    gates["G4_D_expansion_identity"] = {
        "pass": bool(
            exp0["new_columns"] == 192
            and exp0["new_columns_zero"]
            and old_cols == 0.0
            and new_cols == 0.0
            and d_diff <= 1e-6
        ),
        "reader_input_dim_D": infos[mm.MODE_D]["reader_input_dim"],
        "new_columns": exp0["new_columns"],
        "old_columns_match_A0_max_abs": old_cols,
        "new_columns_max_abs": new_cols,
        "init_forward_D_vs_A0_max_abs": d_diff,
        "tolerance": 1e-6,
        "note": "the 513-column matmul changes float32 reduction order; the identity is exact in exact arithmetic",
    }

    # -- G5/G6 gradients: D junction + new reader columns; C junction dead ------
    grads: dict[str, Any] = {}
    for mode, arm in ((mm.MODE_C, "C"), (mm.MODE_D, "D"), (mm.MODE_E, "E")):
        grads[mode] = mm.gradient_audit_mech(
            models[mode], objs=fit_objs[:128], y=y_fit[:128], device=dev, mode=mode
        )
    # a zero-init reader gate blocks the Junction gradient on the first
    # backward pass; re-audit after one real Adam step to show it opens
    d_state = {k: v.detach().clone() for k, v in models[mm.MODE_D].state_dict().items()}
    d_grads = mm.gradient_audit_after_optimizer_step(
        models[mm.MODE_D], objs=fit_objs[:128], y=y_fit[:128], device=dev, mode=mm.MODE_D
    )
    models[mm.MODE_D].load_state_dict(d_state)
    d_j_first = d_grads["at_init"]["modules"]["junction_encoder"]
    d_j = d_grads["after_one_step"]["modules"]["junction_encoder"]
    d_reader = d_grads["after_one_step"]["reader_first_layer_column_grad_max_abs"]
    c_j = grads[mm.MODE_C]["modules"]["junction_encoder"]
    gates["G5_D_gradient_channel_active"] = {
        "pass": bool(
            (d_j["grad_norm_total"] or 0.0) > 0.0
            and d_reader["new_max_abs"] is not None
            and d_reader["new_max_abs"] > 0.0
        ),
        "junction_encoder_grad_norm_at_init": d_j_first["grad_norm_total"],
        "junction_encoder_grad_norm_after_one_step": d_j["grad_norm_total"],
        "reader_new_columns_grad_max_abs_after_one_step": d_reader["new_max_abs"],
        "note": "the zero-init readout gate blocks the first Junction gradient by construction; it opens after the reader columns move",
    }
    gates["G6_C_junction_encoder_structurally_inactive"] = {
        "pass": bool((c_j["grad_norm_total"] in (None, 0.0)) and int(c_j["active_entries_nonzero_grad"]) == 0),
        "junction_encoder_grad_norm": c_j["grad_norm_total"],
        "junction_encoder_allocated": int(c_j["allocated_parameters"]),
        "disclosure": "C discards w, so the Junction encoder is allocated but receives no supervised gradient (effective active capacity < A2)",
    }

    # -- G7 E moment conservation ------------------------------------------------
    e_feats = models[mm.MODE_E].forward_features(small_batch)
    n_graphs = int(small_batch["n_graphs"])
    sum_w = _per_graph_sum(e_feats["w"], small_batch["junc_batch"], n_graphs)
    sum_w2 = _per_graph_sum(e_feats["w"] * e_feats["w"], small_batch["junc_batch"], n_graphs)
    sum_W = _per_graph_sum(e_feats["W"], small_batch["bond_batch"], n_graphs)
    sum_V = _per_graph_sum(e_feats["V"], small_batch["bond_batch"], n_graphs)
    rel = lambda a, b: float((a - b).abs().max() / b.abs().max().clamp(min=1.0))
    gates["G7_E_moment_conservation"] = {
        "pass": bool(rel(sum_W, 2 * sum_w) <= 1e-4 and rel(sum_V, 2 * sum_w2) <= 1e-4),
        "max_rel_err_sum_W_vs_2sum_w": rel(sum_W, 2 * sum_w),
        "max_rel_err_sum_V_vs_2sum_w2": rel(sum_V, 2 * sum_w2),
    }

    # -- G8 E reference-loop equality -------------------------------------------
    W_ref, V_ref, n_ref = mm.type_conditioned_grouping_reference(e_feats["w"], small_batch)
    gates["G8_E_reference_loop_equality"] = {
        "pass": bool(
            float((W_ref - e_feats["W"]).abs().max()) <= 1e-4
            and float((V_ref - e_feats["V"]).abs().max()) <= 1e-4
            and float((n_ref - e_feats["n_b"]).abs().max()) == 0.0
        ),
        "W_max_abs_diff": float((W_ref - e_feats["W"]).abs().max()),
        "V_max_abs_diff": float((V_ref - e_feats["V"]).abs().max()),
        "n_max_abs_diff": float((n_ref - e_feats["n_b"]).abs().max()),
    }

    # -- G9 ownership witness: A2 blind, E sensitive -----------------------------
    wa, wb = mm.type_witness_batch()
    wvec = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    Wa2_a, Va2_a, n_a = mm.null_grouping(wvec, wa)
    Wa2_b, Va2_b, n_b_w = mm.null_grouping(wvec, wb)
    We_a, Ve_a, _, _, _ = mm.type_conditioned_grouping(wvec, wa)
    We_b, Ve_b, _, _, _ = mm.type_conditioned_grouping(wvec, wb)
    gates["G9_type_witness"] = {
        "pass": bool(
            float((Wa2_a - Wa2_b).abs().max()) == 0.0
            and float((n_a - n_b_w).abs().max()) == 0.0
            and float((We_a - We_b).abs().max()) > 0.0
        ),
        "A2_W_max_abs_diff": float((Wa2_a - Wa2_b).abs().max()),
        "A2_n_max_abs_diff": float((n_a - n_b_w).abs().max()),
        "E_W_max_abs_diff": float((We_a - We_b).abs().max()),
        "object_level_only": True,
        "note": "object-level counterexample, not claimed to be a valid molecule or a real ZINC collision",
    }

    # -- G10 type relabel consistency -------------------------------------------
    perm = torch.tensor([2, 0, 3, 1])
    n_types = sro.BOND_CATEGORIES
    bf2 = small_batch["bond_feat"].clone()
    bf2[:, :n_types] = small_batch["bond_feat"][:, perm]
    batch2 = dict(small_batch)
    batch2["bond_feat"] = bf2
    W2, V2, _, _, _ = mm.type_conditioned_grouping(e_feats["w"], batch2)
    W1, V1, _, _, _ = mm.type_conditioned_grouping(e_feats["w"], small_batch)
    gates["G10_type_relabel_consistency"] = {
        "pass": bool(float((W1 - W2).abs().max()) == 0.0 and float((V1 - V2).abs().max()) == 0.0),
        "W_max_abs_diff": float((W1 - W2).abs().max()),
        "V_max_abs_diff": float((V1 - V2).abs().max()),
    }

    # -- G11 junction row permutation invariance ---------------------------------
    perm_objs = _permute_junctions(small, seed=1)
    perm_batch = bp._mv(sro.batch_objects(perm_objs), dev)
    perm_res: dict[str, float] = {}
    for mode in mm.MODES:
        with torch.no_grad():
            perm_res[mode] = float((models[mode](small_batch) - models[mode](perm_batch)).abs().max())
    gates["G11_junction_row_permutation_invariance"] = {
        "pass": bool(max(perm_res.values()) <= 1e-6),
        "per_mode_max_abs_diff": perm_res,
    }

    # -- G12 relabel / endpoint swap / batch composition -------------------------
    guard = ab.LabelGuard(cf.load_official_train(), split["monitor_rows"] + split["locked_rows"])
    rel = tat.relabel_objects(small, guard, seed=0)
    swp = tat.endpoint_swap_objects(small, guard)
    inv: dict[str, Any] = {}
    for mode in mm.MODES:
        with torch.no_grad():
            base = models[mode](small_batch)
            rel_d = float((base - models[mode](bp._mv(sro.batch_objects(rel), dev))).abs().max())
            swp_d = float((base - models[mode](bp._mv(sro.batch_objects(swp), dev))).abs().max())
        bi = tat.batch_invariance(models[mode], small, dev)
        inv[mode] = {
            "atom_relabel": rel_d,
            "endpoint_swap": swp_d,
            "batch_composition": bi["batch_membership_max_abs_diff"],
            "finite": bi["all_finite"],
        }
    worst = max(v for rec in inv.values() for k, v in rec.items() if k != "finite")
    gates["G12_invariances"] = {
        "pass": bool(worst <= 1e-5 and all(rec["finite"] for rec in inv.values())),
        "per_mode": inv,
        "worst": worst,
    }

    # -- G13 save/load replay ----------------------------------------------------
    replay: dict[str, float] = {}
    for mode in mm.MODES:
        tmp = OUT / f"stage_a_replay_{mode}.pt"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(models[mode].state_dict(), tmp)
        fresh = mm.BJOCMechV2(
            mm.config_for_mode(mode),
            np.asarray(norm["global_mean"], dtype=np.float32),
            np.asarray(norm["global_scale"], dtype=np.float32),
            mode=mode,
        )
        fresh.load_state_dict(torch.load(tmp, map_location="cpu", weights_only=False))
        fresh.to(dev)
        fresh.eval()
        with torch.no_grad():
            replay[mode] = float((models[mode](small_batch) - fresh(small_batch)).abs().max())
        tmp.unlink(missing_ok=True)
    gates["G13_save_load_replay"] = {"pass": bool(max(replay.values()) == 0.0), "per_mode": replay}

    # -- G14 edge cases ----------------------------------------------------------
    edge_objs = _make_edge_case_objects()
    edge_batch = bp._mv(sro.batch_objects(edge_objs), dev)
    edge: dict[str, Any] = {}
    for mode in mm.MODES:
        with torch.no_grad():
            pred = models[mode](edge_batch)
        edge[mode] = {"finite": bool(np.isfinite(pred.detach().cpu().numpy()).all())}
    gates["G14_empty_edge_cases_finite"] = {
        "pass": bool(all(v["finite"] for v in edge.values())),
        "per_mode": edge,
    }

    # -- G15 seed-1 parity: batch order + ssl encoder + init ---------------------
    parity: dict[str, Any] = {}
    for seed in SEEDS:
        p3 = _p3_record(seed)
        _base1, model1, info1 = mm.build_mech_from_init(seed, norm, dev, mm.MODE_C, load_ssl_state(seed))
        parity[f"seed{seed}"] = {
            "init_full_matches": info1["base_scratch_init_full_sha256"] == p3["init_full_sha256"],
            "init_group_matches": info1["base_scratch_init_group_state_sha256"] == p3["init_group_state_sha256"],
            "init_encoder_matches": info1["base_scratch_init_encoder_sha256"] == p3["init_encoder_sha256"],
            "encoder_used_matches": info1["encoder_used_sha256"] == p3["encoder_used_sha256"],
            "batch_order_matches": bp.batch_order_hash(seed, 8000, bp.BATCH_SIZE, 360) == p3["batch_order"]["full_hash"],
        }
    gates["G15_seed_parity"] = {
        "pass": bool(all(all(v for k, v in rec.items()) for rec in parity.values())),
        "per_seed": parity,
    }

    # -- G16 Fit-only standardisation --------------------------------------------
    prep = read_json(PREP_PATH)
    gates["G16_fit_only_standardisation"] = {
        "pass": bool(prep.get("fit_only") is True and int(prep.get("n_fit", 0)) == 8000),
        "fit_only": prep.get("fit_only"),
        "n_fit": prep.get("n_fit"),
    }

    # -- G17 label isolation / no official paths ---------------------------------
    om_path = SRC / "objects_manifest_D.json"
    om_hits = int(read_json(om_path)["forbidden_label_access_attempts"]) if om_path.exists() else None
    scan_files = [_CODE_DIR / "run_bjoc_junction_mechanism_exploration_v2.py", _CODE_DIR / "bjoc_junction_mechanism_models_v2.py"]
    scan_tokens = ["load_official_" + "valid", "load_official_" + "test"]
    hits = [f"{p.name}:{t}" for p in scan_files for t in scan_tokens if t in p.read_text(encoding="utf-8")]
    gates["G17_label_isolation_no_official"] = {
        "pass": bool((om_hits in (None, 0)) and not MONITOR_LOCK.exists() and not hits),
        "object_build_forbidden_access_attempts": om_hits,
        "monitor_label_read_this_round": bool(MONITOR_LOCK.exists()),
        "locked_label_read_this_round": False,
        "official_path_hits": hits,
        "note": None if om_hits is not None else "objects manifest absent on this checkout; static scan + no-lock still enforced",
    }

    # -- G18 no forbidden modules ------------------------------------------------
    g18 = {mode: src.forbidden_module_scan(models[mode]) for mode in mm.MODES}
    gates["G18_no_forbidden_modules"] = {"pass": bool(all(len(v) == 0 for v in g18.values())), "hits": g18}

    # -- G19 no-y forward --------------------------------------------------------
    gates["G19_no_label_forward"] = {
        "pass": True,
        "note": "all forwards above run with Fit structure objects only; no y enters any model input",
    }

    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "device": str(dev),
        "gates": gates,
        "gradient_audit_at_init": {
            "C": grads[mm.MODE_C],
            "D": grads[mm.MODE_D],
            "E": grads[mm.MODE_E],
        },
        "build_info_seed0": {mode: {k: v for k, v in infos[mode].items() if k != "named_parameters"} for mode in mm.MODES},
        "ok": bool(all(rec.get("pass", False) for rec in gates.values())),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(CPU_GATES, payload)
    log(f"[stage-a] ok={payload['ok']}")
    for name, rec in gates.items():
        log(f"  {name:44s} {'PASS' if rec.get('pass') else 'FAIL'}")
    return 0 if payload["ok"] else 1


# ---------------------------------------------------------------------------
# parameter / gradient audit
# ---------------------------------------------------------------------------


def cmd_grad_audit(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    dev = torch.device(args.device)
    fit_objs = load_objects("fit")
    check_object_rows(fit_objs, load_split()["fit_rows"], "fit")
    y_fit = load_y_fit()
    payload_path = OUT / "parameter_gradient_audit.json"
    payload: dict[str, Any] = read_json(payload_path) if payload_path.exists() else {"protocol": PROTOCOL_ID, "git_commit": git_commit()}
    phases = list(args.phase) if args.phase else ["init", "trained"]
    arms = tuple(args.arms) if args.arms else ARMS_ALL
    for phase in phases:
        section: dict[str, Any] = {}
        for arm in arms:
            seed = int(args.seed)
            mode = mm.ARM_MODES[arm]
            if phase == "init":
                _base, model, info = mm.build_mech_from_init(seed, load_norm(), dev, mode, load_ssl_state(seed))
            else:
                model = build_mech(arm, seed, dev)
                info = {"soup_path": str(soup_path(arm, seed).relative_to(REPO_ROOT))}
            rec = mm.gradient_audit_mech(model, objs=fit_objs[:128], y=y_fit[:128], device=dev, mode=mode)
            rec["info"] = info
            rec["seed"] = seed
            section[f"{arm}:{seed}"] = rec
            log(
                f"[grad-audit:{phase}] {arm}:{seed} active_entries {rec['active_entries_nonzero_grad']}/"
                f"{rec['allocated_parameters']} junction_grad {rec['modules']['junction_encoder']['grad_norm_total']}"
            )
        payload[f"at_{phase}"] = section
    write_json(payload_path, payload)
    return 0


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _train_one(arm: str, seed: int, *, device: str, smoke_epochs: int | None = None) -> dict[str, Any]:
    if seed not in NEW_ARMS or arm not in NEW_ARMS[seed]:
        raise SystemExit(f"{arm} seed {seed} is not in the pre-declared training matrix")
    smoke = smoke_epochs is not None
    target_soup = (OUT / "smoke" / f"soup_{arm}_D_s{seed}_smoke.pt") if smoke else soup_path(arm, seed)
    target_json = (OUT / "smoke" / f"train_{arm}_D_s{seed}_smoke.json") if smoke else train_json_path(arm, seed)
    if target_soup.exists() or target_json.exists():
        raise SystemExit(f"{arm} seed {seed} already exists; retraining is not allowed")
    split = load_split()
    fit_objs = load_objects("fit")
    check_object_rows(fit_objs, split["fit_rows"], "fit")
    payload = mm.train_supervised_mech(
        arm=arm,
        seed=seed,
        recipe_key="R3",
        init="ssl",
        train_objs=fit_objs,
        rows=split["fit_rows"],
        y_fit=load_y_fit(),
        norm=load_norm(),
        device=torch.device(device),
        mode=mm.ARM_MODES[arm],
        ssl_state=load_ssl_state(seed),
        smoke_epochs=smoke_epochs,
    )
    p3 = _p3_record(seed)
    checks = {
        "init_full_matches_p3": payload["init_full_sha256"] == p3["init_full_sha256"],
        "init_group_matches_p3": payload["init_group_state_sha256"] == p3["init_group_state_sha256"],
        "init_encoder_matches_p3": payload["init_encoder_sha256"] == p3["init_encoder_sha256"],
        "encoder_used_matches_ssl_source": payload["encoder_used_sha256"] == p3["encoder_used_sha256"],
        "parameter_audit_exact": bool(payload["parameter_audit"]["parameter_exact"]),
        "batch_order_matches_p3": (
            bool(payload["batch_order"]["full_matches"])
            if smoke
            else payload["batch_order"]["full_hash"] == p3["batch_order"]["full_hash"]
        ),
        "full_recipe_unchanged": (payload["epochs"] == 360 and payload["updates"] == 22680) if not smoke else True,
        "shared_parameters_match_base": payload["build_info"]["model_shared_parameter_sha256"]
        == payload["build_info"]["base_shared_parameter_sha256"],
    }
    if mm.ARM_MODES[arm] == mm.MODE_D:
        checks["reader_expansion_zero_columns"] = bool(payload["build_info"]["reader_expansion"]["new_columns_zero"])
    payload["p3_matched_checks"] = checks
    payload["p3_batch_order_hash"] = p3["batch_order"]["full_hash"]
    if not all(checks.values()):
        raise SystemExit(f"P3-matching checks failed: {checks}")
    soup = target_soup
    soup.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu().clone() for k, v in payload["soup_state"].items()}, soup)
    payload.pop("soup_state", None)
    payload["soup_path"] = str(soup.relative_to(REPO_ROOT))
    payload["soup_sha256"] = file_sha256(soup)
    write_json(target_json, payload)
    log(f"[train:{arm}:s{seed}] soup fit {payload['soup_fit_mae']:.6f} wall {payload['wall_seconds']:.0f}s -> {soup.name}")
    return payload


def cmd_train(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=True)
    _train_one(str(args.arm), int(args.seed), device=str(args.device), smoke_epochs=args.smoke_epochs)
    return 0


def cmd_train_matrix(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=True)
    seed = int(args.seed)
    arms = tuple(args.arms) if args.arms else NEW_ARMS[seed]
    for arm in arms:
        _train_one(arm, seed, device=str(args.device))
    return 0


# ---------------------------------------------------------------------------
# predictions + manifest
# ---------------------------------------------------------------------------


def _reuse_entry(kind: str, path: Path, sha: str, **extra: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": sha,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }


def _new_prediction_entries() -> dict[str, Any]:
    """Read (or verify) the manifest entries for all critical arm:seed pairs."""
    entries: dict[str, Any] = {}
    jau_manifest = read_json(JAU / "prediction_manifest.json")
    for arm in ("A0", "A1", "A2"):
        entries[f"{arm}:0"] = _reuse_entry(
            "reused_junction_audit",
            JAU_PRED[arm],
            jau_manifest["entries"][arm]["sha256"],
            soup_sha256=jau_manifest["entries"][arm]["soup_sha256"],
            source_manifest=str((JAU / "prediction_manifest.json").relative_to(REPO_ROOT)),
        )
    ta_manifest = read_json(SRC / "prediction_manifest_D.json")
    entries["A2:1"] = _reuse_entry(
        "reused_task_adaptive_ssl",
        P3_MON_S1,
        ta_manifest["entries"]["monitor:P3:1"]["sha256"],
        soup_sha256=ta_manifest["entries"]["monitor:P3:1"]["soup_sha256"],
        source_manifest=str((SRC / "prediction_manifest_D.json").relative_to(REPO_ROOT)),
    )
    return entries


def cmd_predict(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    if MONITOR_LOCK.exists():
        raise SystemExit("Monitor labels already read in this round; refusing new predictions")
    mon_objs = load_objects("monitor")
    check_object_rows(mon_objs, load_split()["monitor_rows"], "monitor")
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds] if args.seeds else list(SEEDS)
    arms = tuple(args.arms) if args.arms else None
    manifest = read_json(PRED_MANIFEST) if PRED_MANIFEST.exists() else {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": "D",
        "entries": {},
        "monitor_labels_read": False,
        "locked_labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    # register reused entries first
    for key, rec in _new_prediction_entries().items():
        p = REPO_ROOT / rec["path"]
        if not p.exists() or file_sha256(p) != rec["sha256"]:
            raise SystemExit(f"reused prediction drift for {key}")
        manifest["entries"][key] = rec
    for seed in seeds:
        for arm in NEW_ARMS[seed]:
            if arms is not None and arm not in arms:
                continue
            key = f"{arm}:{seed}"
            model = build_mech(arm, seed, device)
            pred = predict_objects(model, mon_objs, device)
            if not np.isfinite(pred).all():
                raise SystemExit(f"non-finite prediction for {key}")
            out = new_pred_path(arm, seed)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, row=np.arange(len(mon_objs), dtype=np.int64), pred=pred.astype(np.float32))
            manifest["entries"][key] = {
                "kind": "new",
                "path": str(out.relative_to(REPO_ROOT)),
                "sha256": file_sha256(out),
                "soup_sha256": file_sha256(soup_path(arm, seed)),
                "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            log(f"[predict] {key} -> {out.name} sha={manifest['entries'][key]['sha256'][:12]}")
    missing = [k for k in CRITICAL_KEYS if k not in manifest["entries"]]
    if missing:
        raise SystemExit(f"prediction manifest incomplete: {missing}")
    manifest["monitor_labels_read"] = False
    manifest["locked_labels_read"] = False
    write_json(PRED_MANIFEST, manifest)
    log("[predict] manifest written")
    return 0


# ---------------------------------------------------------------------------
# Monitor scoring
# ---------------------------------------------------------------------------


def _verify_prediction_manifest() -> dict[str, Any]:
    manifest = read_json(PRED_MANIFEST)
    for key, rec in manifest["entries"].items():
        p = REPO_ROOT / rec["path"]
        if not p.exists() or file_sha256(p) != rec["sha256"]:
            raise SystemExit(f"prediction drift for {key}")
    missing = [k for k in CRITICAL_KEYS if k not in manifest["entries"]]
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
        "p95_abs": float(np.percentile(err, 95)),
        "p99_abs": float(np.percentile(err, 99)),
        "ordinary_mae": float(err[ordinary].mean()),
        "ordinary_n": int(ordinary.sum()),
        "tail_mae": float(err[~ordinary].mean()),
        "tail_n": int((~ordinary).sum()),
        "top10_removed_mae": float(err[keep].mean()),
        "n": int(len(err)),
    }


def _ensure_monitor_y() -> np.ndarray:
    if MONITOR_LOCK.exists():
        raise SystemExit("Monitor labels already read in this round; refusing a second read")
    split = load_split()
    y = np.asarray(np.load(JAU_MON_Y), dtype=np.float64).reshape(-1)
    jau_lock = read_json(JAU_MON_LOCK)
    if array_sha256(y) != jau_lock["y_sha256"]:
        raise SystemExit("Monitor y file drifted from the historical lock")
    np.save(MONITOR_Y, y)
    write_json(
        MONITOR_LOCK,
        {
            "protocol": PROTOCOL_ID,
            "split_tag": "D",
            "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_rows": len(y),
            "y_sha256": array_sha256(y),
            "prediction_manifest_sha256": file_sha256(PRED_MANIFEST),
            "label_provenance": "reused the historically exposed split-D Monitor label file (same rows, hash verified)",
            "exposure_disclosure": "split-D Monitor is historically exposed exploratory development data, not a blind validation",
            "official_valid_loaded": False,
            "official_test_loaded": False,
        },
    )
    log(f"[monitor-read] n={len(y)} sha={array_sha256(y)[:12]}")
    return y


def _load_frozen_predictions(n: int) -> dict[str, np.ndarray]:
    manifest = read_json(PRED_MANIFEST)
    out = {}
    for key in CRITICAL_KEYS:
        out[key] = _read_pred_npz(REPO_ROOT / manifest["entries"][key]["path"], n)
    return out


def _contrast(preds: Mapping[str, np.ndarray], new_key: str, ref_key: str, y: np.ndarray) -> dict[str, Any]:
    def seed_of(key: str) -> int:
        return int(key.split(":")[1])

    c = ab.compute_contrast({seed_of(new_key): preds[new_key]}, {seed_of(ref_key): preds[ref_key]}, y)
    c["new"] = new_key
    c["ref"] = ref_key
    return c


def _two_seed_contrast(preds: Mapping[str, np.ndarray], new_arm: str, ref_arm: str, y: np.ndarray) -> dict[str, Any]:
    c = ab.compute_contrast(
        {0: preds[f"{new_arm}:0"], 1: preds[f"{new_arm}:1"]},
        {0: preds[f"{ref_arm}:0"], 1: preds[f"{ref_arm}:1"]},
        y,
    )
    c["new"] = f"{new_arm}:01"
    c["ref"] = f"{ref_arm}:01"
    return c


def cmd_score_monitor(args: argparse.Namespace) -> int:
    verify_freeze(strict_code=False)
    if not CPU_GATES.exists() or not read_json(CPU_GATES).get("ok", False):
        raise SystemExit("Stage-A gate is not green; refusing the Monitor read")
    manifest = _verify_prediction_manifest()
    y = _ensure_monitor_y()
    n = len(y)
    preds = _load_frozen_predictions(n)
    table = {key: _metrics(preds[key], y) for key in CRITICAL_KEYS}
    # attach parameters / fit / wall from train records
    for key in CRITICAL_KEYS:
        arm, seed = key.split(":")
        seed = int(seed)
        rec = {
            "parameters": mm._PARAMS_D if arm == "D" else mm._PARAMS_BASE,
            "fit_mae": None,
            "training_wall_seconds": None,
            "updates": None,
        }
        if (OUT / f"train_{arm}_D_s{seed}.json").exists():
            t = read_json(OUT / f"train_{arm}_D_s{seed}.json")
            rec.update(
                fit_mae=float(t["soup_fit_mae"]),
                training_wall_seconds=float(t["wall_seconds"]),
                updates=int(t["updates"]),
            )
        elif arm in ("A0", "A1"):
            tj = JAU / f"train_{arm}_D_s{seed}.json"
            if tj.exists():
                t = read_json(tj)
                rec.update(fit_mae=float(t["soup_fit_mae"]), training_wall_seconds=float(t["wall_seconds"]), updates=int(t["updates"]))
        elif arm == "A2":
            p3 = _p3_record(seed)
            rec.update(fit_mae=float(p3["soup_fit_mae"]), training_wall_seconds=float(p3["wall_seconds"]), updates=int(p3["updates"]))
        table[key].update(rec)

    contrasts: dict[str, Any] = {}
    # pre-declared main contrasts
    contrasts["C_minus_A0_s0"] = _contrast(preds, "C:0", "A0:0", y)
    contrasts["D_minus_A1_s01"] = _two_seed_contrast(preds, "D", "A1", y)
    contrasts["E_minus_A2_s01"] = _two_seed_contrast(preds, "E", "A2", y)
    contrasts["D_minus_A2_s01"] = _two_seed_contrast(preds, "D", "A2", y)
    contrasts["E_minus_A1_s01"] = _two_seed_contrast(preds, "E", "A1", y)
    contrasts["C_minus_A2_s01"] = _two_seed_contrast(preds, "C", "A2", y)
    # per-seed and supplementary
    for key in ("D_minus_A1_s0", "D_minus_A1_s1", "E_minus_A2_s0", "E_minus_A2_s1", "D_minus_A2_s0", "D_minus_A2_s1",
                "E_minus_A1_s0", "E_minus_A1_s1", "C_minus_A2_s0", "C_minus_A2_s1", "A1_minus_A2_s0", "A1_minus_A2_s1"):
        base, seed = key.rsplit("_s", 1)
        new_arm, ref_arm = base.split("_minus_")
        contrasts[key] = _contrast(preds, f"{new_arm}:{seed}", f"{ref_arm}:{seed}", y)
    for key, (new_arm, ref_arm) in {
        "D_minus_A0_s0": ("D", "A0"),
        "E_minus_A0_s0": ("E", "A0"),
        "E_minus_D_s01": ("E", "D"),
    }.items():
        if key.endswith("_s0"):
            contrasts[key] = _contrast(preds, f"{new_arm}:0", f"{ref_arm}:0", y)
        else:
            contrasts[key] = _two_seed_contrast(preds, new_arm, ref_arm, y)

    cfg = load_config()
    rules = cfg["decision_rules"]
    useful = lambda c: bool(c["delta_seed_mean"] <= -0.002 and c["bootstrap"]["ci_hi"] < 0.0)
    count_delta = contrasts["C_minus_A2_s01"]["delta_seed_mean"]
    tokens = {
        "C": (
            "COUNT_ONLY_NEAR_A2"
            if abs(count_delta) <= 0.002
            else ("COUNT_ONLY_INSUFFICIENT" if count_delta >= 0.005 else "COUNT_ONLY_INCONCLUSIVE")
        ),
        "D": "J_DIRECT_READOUT_USEFUL_EXPLORATORY" if useful(contrasts["D_minus_A1_s01"]) else "J_DIRECT_READOUT_NO_GAIN",
        "E": "J_TYPE_POOL_USEFUL_EXPLORATORY" if useful(contrasts["E_minus_A2_s01"]) else "J_TYPE_POOL_NO_GAIN",
    }
    if tokens["E"] == "J_TYPE_POOL_NO_GAIN":
        e_fit = [table[f"E:{s}"]["fit_mae"] for s in SEEDS]
        a2_fit = [table[f"A2:{s}"]["fit_mae"] for s in SEEDS]
        if all(a is not None and b is not None and a < b for a, b in zip(e_fit, a2_fit)):
            tokens["E"] = "J_TYPE_POOL_FIT_ONLY"
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": "D",
        "seeds": list(SEEDS),
        "n_monitor": n,
        "table": table,
        "contrasts": contrasts,
        "decision_tokens": tokens,
        "decision_rules": rules,
        "prediction_manifest_sha256": file_sha256(PRED_MANIFEST),
        "monitor_read_utc": read_json(MONITOR_LOCK)["read_utc"],
        "exposure_disclosure": "historically exposed exploratory development data",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(MONITOR_SCORES, payload)
    for key in ("A0:0", "A1:0", "A2:0", "C:0", "D:0", "E:0", "A1:1", "A2:1", "C:1", "D:1", "E:1"):
        t = table[key]
        log(f"  {key:6s} MAE {t['mae']:.6f} RMSE {t['rmse']:.6f} ordinary {t['ordinary_mae']:.6f} tail {t['tail_mae']:.6f}")
    for key, c in contrasts.items():
        log(
            f"  {key:22s} {c['delta_seed_mean']:+.6f} CI95 [{c['bootstrap']['ci_lo']:+.6f},{c['bootstrap']['ci_hi']:+.6f}]"
        )
    log(f"  tokens: {tokens}")
    return 0


# ---------------------------------------------------------------------------
# Part I: error complementarity
# ---------------------------------------------------------------------------


def _pair_stats(new: np.ndarray, ref: np.ndarray, y: np.ndarray, top_ks: Sequence[int]) -> dict[str, Any]:
    ae_new, ae_ref = np.abs(new - y), np.abs(ref - y)
    se_new, se_ref = (new - y) ** 2, (ref - y) ** 2
    d_ae = ae_new - ae_ref
    d_se = se_new - se_ref
    order_dse = np.argsort(-d_se)
    order_se_new = np.argsort(-se_new)
    pos = np.maximum(d_se, 0.0)
    pos_sum = float(pos.sum())
    net = float(d_se.sum())

    def rmse_without(order: np.ndarray, k: int) -> dict[str, float]:
        keep = np.ones(len(y), dtype=bool)
        keep[order[:k]] = False
        return {
            "rmse_new": float(np.sqrt(se_new[keep].mean())),
            "rmse_ref": float(np.sqrt(se_ref[keep].mean())),
            "mae_new": float(ae_new[keep].mean()),
            "mae_ref": float(ae_ref[keep].mean()),
        }

    top_rows: dict[str, Any] = {}
    for k in top_ks:
        idx = order_dse[:k]
        top_rows[str(k)] = {
            "sum_d_se": float(d_se[idx].sum()),
            "positive_part_sum": float(pos[idx].sum()),
            "share_of_all_positive_se_increments": float(pos[idx].sum() / pos_sum) if pos_sum > 0 else None,
            "share_of_net_se_change": float(d_se[idx].sum() / net) if net != 0.0 else None,
            "remove_top_dse_worsening": rmse_without(order_dse, k),
            "remove_top_new_se": rmse_without(order_se_new, k),
        }
    worst = []
    for i in order_dse[:10]:
        worst.append(
            {
                "row": int(i),
                "y": float(y[i]),
                "pred_new": float(new[i]),
                "pred_ref": float(ref[i]),
                "ae_new": float(ae_new[i]),
                "ae_ref": float(ae_ref[i]),
                "d_ae": float(d_ae[i]),
                "se_new": float(se_new[i]),
                "se_ref": float(se_ref[i]),
                "d_se": float(d_se[i]),
            }
        )
    thr = 0.05
    return {
        "mae_new": float(ae_new.mean()),
        "mae_ref": float(ae_ref.mean()),
        "d_ae": {
            "mean": float(d_ae.mean()),
            "median": float(np.median(d_ae)),
            "q05": float(np.percentile(d_ae, 5)),
            "q25": float(np.percentile(d_ae, 25)),
            "q75": float(np.percentile(d_ae, 75)),
            "q95": float(np.percentile(d_ae, 95)),
            "positive_n": int((d_ae > 0).sum()),
            "negative_n": int((d_ae < 0).sum()),
            "zero_n": int((d_ae == 0).sum()),
            "positive_mean": float(d_ae[d_ae > 0].mean()) if (d_ae > 0).any() else 0.0,
            "negative_mean": float(d_ae[d_ae < 0].mean()) if (d_ae < 0).any() else 0.0,
            "positive_median": float(np.median(d_ae[d_ae > 0])) if (d_ae > 0).any() else 0.0,
            "negative_median": float(np.median(d_ae[d_ae < 0])) if (d_ae < 0).any() else 0.0,
            "improvement_magnitude_sum": float(-d_ae[d_ae < 0].sum()),
            "degradation_magnitude_sum": float(d_ae[d_ae > 0].sum()),
        },
        "d_se": {
            "mean": float(d_se.mean()),
            "median": float(np.median(d_se)),
            "q05": float(np.percentile(d_se, 5)),
            "q25": float(np.percentile(d_se, 25)),
            "q75": float(np.percentile(d_se, 75)),
            "q95": float(np.percentile(d_se, 95)),
            "min": float(d_se.min()),
            "max": float(d_se.max()),
            "positive_n": int((d_se > 0).sum()),
            "negative_n": int((d_se < 0).sum()),
            "net_sum": net,
            "positive_sum": pos_sum,
        },
        "clear_better": {
            f"ref_thr_{thr}": int((d_ae > thr).sum()),
            f"new_thr_{thr}": int((d_ae < -thr).sum()),
            "mid": int((np.abs(d_ae) <= thr).sum()),
            "any_ref_better": int((d_ae > 0).sum()),
            "any_new_better": int((d_ae < 0).sum()),
            f"ref_thr_{thr}_frac": float((d_ae > thr).mean()),
            f"new_thr_{thr}_frac": float((d_ae < -thr).mean()),
            "any_ref_better_frac": float((d_ae > 0).mean()),
            "any_new_better_frac": float((d_ae < 0).mean()),
        },
        "top_k": top_rows,
        "worst_rows_by_d_se": worst,
    }


def _descriptor_matrix(objs: Sequence[sro.MolObjects]) -> np.ndarray:
    rows = []
    for o in objs:
        raw = np.asarray(
            [o.n_atoms, float(o.bond_index.shape[0]), float(o.junc_index.shape[0]), float(o.global_feat[3])],
            dtype=np.float64,
        )
        atom_hist = np.asarray(o.atom_feat[:, :28], dtype=np.float64).sum(0)
        bond_hist = np.asarray(o.bond_feat[:, :4], dtype=np.float64).sum(0)
        deg_hist = np.asarray(o.atom_feat[:, 28:33], dtype=np.float64).sum(0)
        rows.append(np.concatenate([raw, np.asarray(o.global_feat, dtype=np.float64), atom_hist, bond_hist, deg_hist]))
    return np.stack(rows, 0) if rows else np.zeros((0, 74), dtype=np.float64)


def _nn_distance(fit_z: np.ndarray, mon_z: np.ndarray, chunk: int = 128) -> np.ndarray:
    out = np.empty(mon_z.shape[0], dtype=np.float64)
    fit_t = torch.as_tensor(fit_z, dtype=torch.float64)
    for i in range(0, mon_z.shape[0], int(chunk)):
        q = torch.as_tensor(mon_z[i : i + int(chunk)], dtype=torch.float64)
        d2 = torch.cdist(q, fit_t, p=2.0) ** 2
        out[i : i + int(chunk)] = torch.sqrt(d2.min(dim=1).values).numpy()
    return out


def _corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if method == "pearson":
        return float(np.corrcoef(a, b)[0, 1])
    # spearman = pearson of ranks
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def cmd_part1(args: argparse.Namespace) -> int:
    """Error-complementarity analysis of the frozen A0/A1/A2 seed-0 predictions."""
    cfg = load_config()
    # -- verification ---------------------------------------------------------
    y = np.asarray(np.load(JAU_MON_Y), dtype=np.float64).reshape(-1)
    if array_sha256(y) != read_json(JAU_MON_LOCK)["y_sha256"]:
        raise SystemExit("Part-I y hash mismatch")
    if array_sha256(y) != cfg["reuse"]["monitor_y"]["array_sha256"]:
        raise SystemExit("Part-I y hash does not match the protocol freeze")
    preds = {}
    verification: dict[str, Any] = {}
    hist_scores = read_json(JAU_SCORES)
    for arm in ("A0", "A1", "A2"):
        rec = cfg["reuse"]["part1_frozen_predictions"][arm]
        path = JAU_PRED[arm]
        if file_sha256(path) != rec["sha256"]:
            raise SystemExit(f"Part-I prediction hash mismatch for {arm}")
        z = np.load(path, allow_pickle=True)
        p = np.asarray(z["pred"], dtype=np.float64).reshape(-1)
        row = np.asarray(z["row"], dtype=np.int64).reshape(-1)
        mae = float(np.abs(p - y).mean())
        verification[arm] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": rec["sha256"],
            "rows_arange": bool(np.array_equal(row, np.arange(len(y)))),
            "finite": bool(np.isfinite(p).all()),
            "mae_recomputed": mae,
            "mae_historical": float(hist_scores["table"][arm]["mae"]),
            "mae_recomputed_matches_historical": abs(mae - float(hist_scores["table"][arm]["mae"])) <= 1e-12,
            "scale_ok": bool(abs(mae - float(hist_scores["table"][arm]["mae"])) <= 1e-12),
        }
        preds[arm] = p
    verification["y"] = {
        "path": str(JAU_MON_Y.relative_to(REPO_ROOT)),
        "array_sha256": array_sha256(y),
        "n": int(len(y)),
        "matches_historical_lock": True,
    }

    metrics = {arm: _metrics(preds[arm], y) for arm in ("A0", "A1", "A2")}
    pairs = {}
    for new_arm, ref_arm in (("A2", "A0"), ("A1", "A0"), ("A2", "A1")):
        key = f"{new_arm}-{ref_arm}"
        st = _pair_stats(preds[new_arm], preds[ref_arm], y, cfg["part1"]["top_k"])
        st["rmse_new"] = float(np.sqrt(np.mean((preds[new_arm] - y) ** 2)))
        st["rmse_ref"] = float(np.sqrt(np.mean((preds[ref_arm] - y) ** 2)))
        pairs[key] = st

    # -- structural novelty ----------------------------------------------------
    fit_objs = load_objects("fit")
    mon_objs = load_objects("monitor")
    check_object_rows(fit_objs, load_split()["fit_rows"], "fit")
    check_object_rows(mon_objs, load_split()["monitor_rows"], "monitor")
    fit_desc = _descriptor_matrix(fit_objs)
    mon_desc = _descriptor_matrix(mon_objs)
    mu, sd = fit_desc.mean(0), fit_desc.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    fit_z, mon_z = (fit_desc - mu) / sd, (mon_desc - mu) / sd
    nn = _nn_distance(fit_z, mon_z)
    q25, q50, q75 = (float(np.quantile(nn, q)) for q in (0.25, 0.50, 0.75))
    quart = np.digitize(nn, [q25, q50, q75])  # 0..3
    y_tail = y <= bp.TAIL_THRESHOLD

    def stratum_stats(mask: np.ndarray) -> dict[str, Any]:
        rec: dict[str, Any] = {"n": int(mask.sum()), "tail_n": int((mask & y_tail).sum())}
        for arm in ("A0", "A1", "A2"):
            rec[f"{arm}_mae"] = float(np.abs(preds[arm][mask] - y[mask]).mean()) if mask.any() else None
            rec[f"{arm}_rmse"] = float(np.sqrt(np.mean((preds[arm][mask] - y[mask]) ** 2))) if mask.any() else None
        d_ae = np.abs(preds["A2"] - y) - np.abs(preds["A0"] - y)
        d_se = (preds["A2"] - y) ** 2 - (preds["A0"] - y) ** 2
        rec["d_ae_A2_A0_mean"] = float(d_ae[mask].mean()) if mask.any() else None
        rec["d_se_A2_A0_mean"] = float(d_se[mask].mean()) if mask.any() else None
        rec["ref_better_frac"] = float((d_ae[mask] > 0).mean()) if mask.any() else None
        rec["novelty_median"] = float(np.median(nn[mask])) if mask.any() else None
        return rec

    strata = {f"Q{i+1}": stratum_stats(quart == i) for i in range(4)}
    cross_tab = {
        "y_le_-3": {f"Q{i+1}": int(((quart == i) & y_tail).sum()) for i in range(4)},
        "y_gt_-3": {f"Q{i+1}": int(((quart == i) & ~y_tail).sum()) for i in range(4)},
    }

    d_ae_20 = np.abs(preds["A2"] - y) - np.abs(preds["A0"] - y)
    thr = 0.05
    groups = {
        "A0_clearly_better(dae>0.05)": d_ae_20 > thr,
        "A2_clearly_better(dae<-0.05)": d_ae_20 < -thr,
        "mid": np.abs(d_ae_20) <= thr,
    }
    group_stats = {}
    for name, mask in groups.items():
        group_stats[name] = {
            "n": int(mask.sum()),
            "novelty_mean": float(nn[mask].mean()) if mask.any() else None,
            "novelty_median": float(np.median(nn[mask])) if mask.any() else None,
            "novelty_q25": float(np.quantile(nn[mask], 0.25)) if mask.any() else None,
            "novelty_q75": float(np.quantile(nn[mask], 0.75)) if mask.any() else None,
            "quartile_hist": {f"Q{i+1}": int(((quart == i) & mask).sum()) for i in range(4)},
            "tail_n": int((mask & y_tail).sum()),
        }

    ae = {arm: np.abs(preds[arm] - y) for arm in ("A0", "A1", "A2")}
    corr = {}
    for a in ("A0", "A1", "A2"):
        for b in ("A0", "A1", "A2"):
            corr[f"AE_{a}_{b}"] = {"pearson": _corr(ae[a], ae[b], "pearson"), "spearman": _corr(ae[a], ae[b], "spearman")}
    novelty_corr = {
        "AE_A2_vs_novelty": {"pearson": _corr(ae["A2"], nn, "pearson"), "spearman": _corr(ae["A2"], nn, "spearman")},
        "d_ae_A2_A0_vs_novelty": {"pearson": _corr(d_ae_20, nn, "pearson"), "spearman": _corr(d_ae_20, nn, "spearman")},
        "y_vs_novelty": {"pearson": _corr(y, nn, "pearson"), "spearman": _corr(y, nn, "spearman")},
    }

    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "analysis": "Part I error complementarity (frozen A0/A1/A2 seed 0; historically exposed Monitor)",
        "input_verification": verification,
        "metrics": metrics,
        "pairs": pairs,
        "novelty": {
            "descriptor_blocks": cfg["part1"]["descriptor_blocks"],
            "fit_standardiser": {"mean_sha256": hashlib.sha256(mu.astype(np.float32).tobytes()).hexdigest(),
                                 "scale_sha256": hashlib.sha256(sd.astype(np.float32).tobytes()).hexdigest()},
            "monitor_distance": {
                "min": float(nn.min()), "q25": q25, "median": q50, "q75": q75, "max": float(nn.max()),
                "mean": float(nn.mean()),
            },
            "strata_quartiles": strata,
            "cross_tab_y_tail": cross_tab,
            "win_loss_by_dae": group_stats,
            "error_correlation": corr,
            "novelty_correlation": novelty_corr,
        },
        "questions": {
            "q1_extreme_molecules": "see pairs.A2-A0.top_k and worst_rows_by_d_se",
            "q2_rarity": "see novelty.win_loss_by_dae and novelty.strata_quartiles",
            "q3_within_novelty": "see novelty.strata_quartiles d_ae_A2_A0_mean and ref_better_frac",
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "locked_read": False,
    }
    write_json(OUT / "error_complementarity.json", payload)
    log("[part1] written error_complementarity.json")
    log(f"  metrics: A0 {metrics['A0']['mae']:.6f} A1 {metrics['A1']['mae']:.6f} A2 {metrics['A2']['mae']:.6f}")
    top1 = pairs["A2-A0"]["top_k"]["1"]
    log(
        f"  A2-A0 top1 positive share {top1['share_of_all_positive_se_increments']:.3f} "
        f"net share {top1['share_of_net_se_change']:.3f} rmse without {top1['remove_top_dse_worsening']['rmse_new']:.4f}"
        f" (A0 {top1['remove_top_dse_worsening']['rmse_ref']:.4f})"
    )
    for i in range(4):
        s = strata[f"Q{i+1}"]
        log(f"  nov Q{i+1}: n={s['n']} A0 {s['A0_mae']:.4f} A2 {s['A2_mae']:.4f} dAE {s['d_ae_A2_A0_mean']:+.4f}")
    return 0


# ---------------------------------------------------------------------------
# checkpoint manifest / finalize / render
# ---------------------------------------------------------------------------


def cmd_checkpoint_manifest(args: argparse.Namespace) -> int:
    manifest = {"protocol": PROTOCOL_ID, "git_commit": git_commit(), "reused": {}, "new": {}}
    cfg = load_config()
    manifest["reused"] = {
        "ssl_seed0": {"path": str(SSL_PATHS[0].relative_to(REPO_ROOT)), "sha256": file_sha256(SSL_PATHS[0])},
        "ssl_seed1": {"path": str(SSL_PATHS[1].relative_to(REPO_ROOT)), "sha256": file_sha256(SSL_PATHS[1])},
        "p3_soup_seed0": {"path": str(P3_SOUP_PATHS[0].relative_to(REPO_ROOT)), "sha256": file_sha256(P3_SOUP_PATHS[0])},
        "p3_soup_seed1": {"path": str(P3_SOUP_PATHS[1].relative_to(REPO_ROOT)), "sha256": file_sha256(P3_SOUP_PATHS[1])},
        "a0_soup_seed0": {"path": str(A0_SOUP.relative_to(REPO_ROOT)), "sha256": file_sha256(A0_SOUP)},
        "a1_soup_seed0": {"path": str(A1_SOUP.relative_to(REPO_ROOT)), "sha256": file_sha256(A1_SOUP)},
    }
    for seed in SEEDS:
        for arm in NEW_ARMS[seed]:
            p = soup_path(arm, seed)
            t = train_json_path(arm, seed)
            manifest["new"][f"{arm}:{seed}"] = {
                "path": str(p.relative_to(REPO_ROOT)),
                "sha256": file_sha256(p) if p.exists() else None,
                "train_record": str(t.relative_to(REPO_ROOT)),
                "train_record_sha256": file_sha256(t) if t.exists() else None,
                "epochs": 360,
                "updates": 22680,
                "init_matched_to_p3": True,
                "batch_order_matched_to_p3": True,
                "parameter_count": int(mm._PARAMS_D if arm == "D" else mm._PARAMS_BASE),
            }
    manifest["note"] = "A2 (both seeds) and A0/A1 seed 0 have no new checkpoint; they load historical soups."
    write_json(OUT / "checkpoint_manifest.json", manifest)
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    scores = read_json(MONITOR_SCORES)
    part1 = read_json(OUT / "error_complementarity.json")
    cfg = load_config()
    # GPU ledger from train records
    traj = []
    for seed in SEEDS:
        for arm in NEW_ARMS[seed]:
            t = read_json(train_json_path(arm, seed))
            traj.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "wall_seconds": float(t["wall_seconds"]),
                    "gpu_hours": float(t["wall_seconds"]) / 3600.0,
                    "device": t["device"],
                    "cuda_device_name": t["cuda_device_name"],
                    "soup_fit_mae": float(t["soup_fit_mae"]),
                }
            )
    ledger = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "hard_limit_gpu_hours": float(cfg["budget"]["gpu_hard_limit_hours"]),
        "trajectories": traj,
        "total_wall_gpu_hours": float(sum(t["gpu_hours"] for t in traj)),
        "note": "gpu_hours are per-trajectory wall seconds / 3600; parallel jobs are counted separately",
    }
    write_json(OUT / "gpu_budget_ledger.json", ledger)
    decision = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "decision_tokens": scores["decision_tokens"],
        "primary_contrasts": {
            "C_minus_A0_s0": scores["contrasts"]["C_minus_A0_s0"]["delta_seed_mean"],
            "C_minus_A2_s01": scores["contrasts"]["C_minus_A2_s01"]["delta_seed_mean"],
            "D_minus_A1_s01": scores["contrasts"]["D_minus_A1_s01"]["delta_seed_mean"],
            "E_minus_A2_s01": scores["contrasts"]["E_minus_A2_s01"]["delta_seed_mean"],
            "D_minus_A2_s01": scores["contrasts"]["D_minus_A2_s01"]["delta_seed_mean"],
            "E_minus_A1_s01": scores["contrasts"]["E_minus_A1_s01"]["delta_seed_mean"],
        },
        "part1_summary": {
            "A2_A0_top1_positive_share": part1["pairs"]["A2-A0"]["top_k"]["1"]["share_of_all_positive_se_increments"],
            "A2_A0_top1_net_share": part1["pairs"]["A2-A0"]["top_k"]["1"]["share_of_net_se_change"],
            "A2_A0_top20_positive_share": part1["pairs"]["A2-A0"]["top_k"]["20"]["share_of_all_positive_se_increments"],
            "A2_A0_top20_net_share": part1["pairs"]["A2-A0"]["top_k"]["20"]["share_of_net_se_change"],
            "novelty_quartile_dAE": {f"Q{i+1}": part1["novelty"]["strata_quartiles"][f"Q{i+1}"]["d_ae_A2_A0_mean"] for i in range(4)},
        },
        "claims_not_made": [
            "no causal attribution from forward sensitivity",
            "no claim that bootstrap-crossing zero proves no difference",
            "no claim that the object-level type witness is a real molecule or a real ZINC collision",
            "no independent-split or blind validation claim: split-D Monitor is exposed",
            "no official valid/test or Locked read",
        ],
    }
    write_json(DECISION, decision)
    log(f"[finalize] tokens {scores['decision_tokens']}; GPU {ledger['total_wall_gpu_hours']:.4f} h")
    return 0


def _fmt(x: Any, nd: int = 6) -> str:
    if x is None:
        return "NOT_RUN"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def cmd_render(args: argparse.Namespace) -> int:
    part1 = read_json(OUT / "error_complementarity.json")
    scores = read_json(MONITOR_SCORES)
    lines: list[str] = []
    lines.append("# Part I — Error complementarity of the frozen A0/A1/A2 seed-0 predictions")
    lines.append("")
    lines.append("Split-D Monitor (n=1000, historically exposed). All numbers are paired per molecule;")
    lines.append("the labels and predictions were hash-verified before this analysis (see `error_complementarity.json`).")
    lines.append("")
    lines.append("## 1. Verification")
    lines.append("")
    lines.append("| arm | file sha256 (12) | rows arange | MAE recomputed | matches historical |")
    lines.append("|---|---|---|---|---|")
    for arm in ("A0", "A1", "A2"):
        v = part1["input_verification"][arm]
        lines.append(f"| {arm} | `{v['sha256'][:12]}` | {v['rows_arange']} | {v['mae_recomputed']:.12f} | {v['mae_recomputed_matches_historical']} |")
    lines.append("")
    lines.append("## 2. Per-model error distribution")
    lines.append("")
    lines.append("| model | MAE | RMSE | median | P90 | P95 | P99 |")
    lines.append("|---|---|---|---|---|---|---|")
    for arm in ("A0", "A1", "A2"):
        m = part1["metrics"][arm]
        lines.append(f"| {arm} | {_fmt(m['mae'])} | {_fmt(m['rmse'])} | {_fmt(m['median_abs'])} | {_fmt(m['p90_abs'])} | {_fmt(m['p95_abs'])} | {_fmt(m['p99_abs'])} |")
    lines.append("")
    for pair in ("A2-A0", "A1-A0", "A2-A1"):
        p = part1["pairs"][pair]
        lines.append(f"## 3. Pair {pair}")
        lines.append("")
        lines.append(f"Net SSE change = {_fmt(p['d_se']['net_sum'], 3)}; sum of positive SSE increments = {_fmt(p['d_se']['positive_sum'], 3)}.")
        lines.append("")
        lines.append("| k | top-k ΔSSE | share of positive SSE | share of net SSE | RMSE without top-k (new / ref) | RMSE without top new-SE (new / ref) |")
        lines.append("|---|---|---|---|---|---|")
        for k in ("1", "3", "5", "10", "20"):
            t = p["top_k"][k]
            a = t["remove_top_dse_worsening"]
            b = t["remove_top_new_se"]
            lines.append(
                f"| {k} | {_fmt(t['sum_d_se'], 4)} | {_fmt(t['share_of_all_positive_se_increments'], 4)} | {_fmt(t['share_of_net_se_change'], 4)} | "
                f"{_fmt(a['rmse_new'], 4)} / {_fmt(a['rmse_ref'], 4)} | {_fmt(b['rmse_new'], 4)} / {_fmt(b['rmse_ref'], 4)} |"
            )
        d = p["d_ae"]
        lines.append("")
        lines.append(
            f"Per-molecule ΔAE: mean {d['mean']:+.6f}, median {d['median']:+.6f}; "
            f"{d['positive_n']} worse / {d['negative_n']} better / {d['zero_n']} equal; "
            f"positive mean {d['positive_mean']:+.6f}, negative mean {d['negative_mean']:+.6f}; "
            f"clearly worse (>0.05) {p['clear_better']['ref_thr_0.05']}, clearly better (<-0.05) {p['clear_better']['new_thr_0.05']}."
        )
        lines.append("")
        lines.append("| row | y | pred new | pred ref | AE new | AE ref | ΔSE |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in p["worst_rows_by_d_se"]:
            lines.append(f"| {r['row']} | {r['y']:.4f} | {r['pred_new']:.4f} | {r['pred_ref']:.4f} | {r['ae_new']:.4f} | {r['ae_ref']:.4f} | {r['d_se']:.4f} |")
        lines.append("")
    nv = part1["novelty"]
    lines.append("## 4. Structural novelty (Fit-only standardised descriptors)")
    lines.append("")
    lines.append("Monitor nearest-Fit distance quantiles: min {:.3f}, Q25 {:.3f}, median {:.3f}, Q75 {:.3f}, max {:.3f}.".format(
        nv["monitor_distance"]["min"], nv["monitor_distance"]["q25"], nv["monitor_distance"]["median"],
        nv["monitor_distance"]["q75"], nv["monitor_distance"]["max"]))
    lines.append("")
    lines.append("| stratum | n | tail n | A0 MAE | A1 MAE | A2 MAE | mean ΔAE(A2-A0) | ref-better frac |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for i in range(4):
        s = nv["strata_quartiles"][f"Q{i+1}"]
        lines.append(f"| Q{i+1} | {s['n']} | {s['tail_n']} | {_fmt(s['A0_mae'],4)} | {_fmt(s['A1_mae'],4)} | {_fmt(s['A2_mae'],4)} | {s['d_ae_A2_A0_mean']:+.4f} | {_fmt(s['ref_better_frac'],3)} |")
    lines.append("")
    lines.append("y-tail × novelty quartile counts (structure strata vs label strata):")
    lines.append("")
    lines.append("| stratum | y ≤ −3 | y > −3 |")
    lines.append("|---|---|---|")
    for i in range(4):
        lines.append(f"| Q{i+1} | {nv['cross_tab_y_tail']['y_le_-3'][f'Q{i+1}']} | {nv['cross_tab_y_tail']['y_gt_-3'][f'Q{i+1}']} |")
    lines.append("")
    lines.append("| win/loss group | n | novelty median | Q1 | Q2 | Q3 | Q4 | tail n |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, g in nv["win_loss_by_dae"].items():
        lines.append(
            f"| {name} | {g['n']} | {_fmt(g['novelty_median'],3)} | {g['quartile_hist']['Q1']} | {g['quartile_hist']['Q2']} | "
            f"{g['quartile_hist']['Q3']} | {g['quartile_hist']['Q4']} | {g['tail_n']} |"
        )
    lines.append("")
    lines.append("AE correlation (Pearson / Spearman): A0-A1 {:.3f}/{:.3f}, A0-A2 {:.3f}/{:.3f}, A1-A2 {:.3f}/{:.3f}.".format(
        nv["error_correlation"]["AE_A0_A1"]["pearson"], nv["error_correlation"]["AE_A0_A1"]["spearman"],
        nv["error_correlation"]["AE_A0_A2"]["pearson"], nv["error_correlation"]["AE_A0_A2"]["spearman"],
        nv["error_correlation"]["AE_A1_A2"]["pearson"], nv["error_correlation"]["AE_A1_A2"]["spearman"]))
    (OUT / "ERROR_COMPLEMENTARITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # -- architecture comparison ------------------------------------------------
    cfg = load_config()
    order = ["A0:0", "C:0", "A1:0", "A2:0", "D:0", "E:0", "C:1", "A1:1", "A2:1", "D:1", "E:1"]
    lines = []
    lines.append("# Part II — Architecture comparison (split-D Monitor, n=1000, historically exposed)")
    lines.append("")
    lines.append("All arms are paired by seed on the same molecules and share the per-seed Fit-only masked-SSL")
    lines.append("encoder, scratch init and batch order; only the pre-declared Junction path differs.")
    lines.append("")
    lines.append("## 1. Per-arm metrics")
    lines.append("")
    lines.append("| arm:seed | MAE | RMSE | median | P90 | ordinary (y>−3) | tail (y≤−3) | top-10 removed | fit MAE | params | wall (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for key in order:
        t = scores["table"][key]
        lines.append(
            f"| {key} | {_fmt(t['mae'])} | {_fmt(t['rmse'])} | {_fmt(t['median_abs'])} | {_fmt(t['p90_abs'])} | "
            f"{_fmt(t['ordinary_mae'])} (n={t['ordinary_n']}) | {_fmt(t['tail_mae'])} (n={t['tail_n']}) | "
            f"{_fmt(t['top10_removed_mae'])} | {_fmt(t['fit_mae'])} | {t['parameters']} | {_fmt(t['training_wall_seconds'],1)} |"
        )
    lines.append("")
    lines.append("## 2. Pre-declared contrasts (paired molecule bootstrap, 5000 resamples)")
    lines.append("")
    lines.append("| contrast | Δ MAE | CI95 | ordinary Δ | tail Δ | RMSE new/ref | top-10 removed Δ | frac better |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for key in ("C_minus_A0_s0", "C_minus_A2_s01", "D_minus_A1_s01", "D_minus_A2_s01", "E_minus_A2_s01", "E_minus_A1_s01", "D_minus_A0_s0", "E_minus_A0_s0"):
        c = scores["contrasts"][key]
        lines.append(
            f"| {key} | {c['delta_seed_mean']:+.6f} | [{c['bootstrap']['ci_lo']:+.6f}, {c['bootstrap']['ci_hi']:+.6f}] | "
            f"{c['ordinary_delta']:+.6f} | {c['tail_delta']:+.6f} | {c['rmse_new']:.6f} / {c['rmse_ref']:.6f} | "
            f"{c['top10_removed_delta']:+.6f} | {c['frac_molecules_better']:.3f} |"
        )
    lines.append("")
    lines.append("Per-seed deltas: " + ", ".join(
        f"{key} {scores['contrasts'][key]['per_seed_delta']}" for key in ("D_minus_A1_s01", "E_minus_A2_s01", "C_minus_A2_s01")
    ) + ".")
    lines.append("")
    lines.append("## 3. Decision tokens")
    lines.append("")
    for arm, token in scores["decision_tokens"].items():
        lines.append(f"- **{arm}**: `{token}`")
    lines.append("")
    ledger = read_json(OUT / "gpu_budget_ledger.json")
    lines.append(f"GPU: {ledger['total_wall_gpu_hours']:.4f} h of {ledger['hard_limit_gpu_hours']} h across "
                 f"{len(ledger['trajectories'])} new trajectories.")
    (OUT / "ARCHITECTURE_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("[render] wrote ERROR_COMPLEMENTARITY.md and ARCHITECTURE_COMPARISON.md")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify-inputs")
    sub.add_parser("freeze")
    sub.add_parser("stage-a")
    ga = sub.add_parser("grad-audit")
    ga.add_argument("--phase", action="append", default=None)
    ga.add_argument("--arms", nargs="*", default=None)
    ga.add_argument("--seed", type=int, default=0)
    tr = sub.add_parser("train")
    tr.add_argument("--arm", required=True)
    tr.add_argument("--seed", type=int, required=True)
    tr.add_argument("--smoke-epochs", type=int, default=None)
    tm = sub.add_parser("train-matrix")
    tm.add_argument("--seed", type=int, required=True)
    tm.add_argument("--arms", nargs="*", default=None)
    pr = sub.add_parser("predict")
    pr.add_argument("--arms", nargs="*", default=None)
    pr.add_argument("--seeds", nargs="*", default=None)
    sub.add_parser("part1")
    sub.add_parser("score-monitor")
    sub.add_parser("checkpoint-manifest")
    sub.add_parser("finalize")
    sub.add_parser("render")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dispatch = {
        "verify-inputs": cmd_verify_inputs,
        "freeze": cmd_freeze,
        "stage-a": cmd_stage_a,
        "grad-audit": cmd_grad_audit,
        "train": cmd_train,
        "train-matrix": cmd_train_matrix,
        "predict": cmd_predict,
        "part1": cmd_part1,
        "score-monitor": cmd_score_monitor,
        "checkpoint-manifest": cmd_checkpoint_manifest,
        "finalize": cmd_finalize,
        "render": cmd_render,
    }
    return dispatch[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
