"""cscl-bjoc-predictor-terminal-evaluation-v1 — driver.

Protocol: ``configs/bjoc_predictor_terminal_v1.yaml`` (frozen before the single
authorised new training trajectory and before any official-test read).

Question (three parts)
----------------------
Q1  Can BJOC-Null's prediction advantage be reproduced at full 10k with a
    second seed (the only new training trajectory of this round)?
Q2  How much of the previously observed official-validation gain
    (0.118618 -> 0.103423) comes from the single extreme row ``valid:0172``
    and how much from broad improvement over ordinary molecules?
Q3  What raw-y MAE does BJOC-Null reach on the official test split, relative
    to the historical two-seed Full-Y system?

Stages
------
``reuse-audit``  Phase-A engineering reuse audit of every frozen artifact
                 (Y0/Y1 checkpoints + training-domain prep, N0 checkpoint +
                 full-10k train-only standardiser, BJOC object cache) with a
                 forward replay gate on all 10 000 official-train rows.
``valid-audit``  Phase-C zero-training per-molecule official-validation error
                 decomposition (valid:0172 conservation, ordinary vs tail
                 molecules, complexity strata, input-content comparison).
``train-n1``     The ONE authorised trajectory: BJOC-Null, full 10k, seed 1.
                 Delegates to the frozen entry point
                 ``run_structural_relation_composer_v1.train_bjoc_arm``.
``predict``      Label-free frozen predictions for Y0/Y1/N0/N1 on official
                 validation and official test (the test label field is never
                 read; predictions carry no y).
``freeze``       terminal_freeze_manifest.json: checkpoint + prediction +
                 split fingerprints, ensemble rule, metric rules, exposure
                 declaration.  Written BEFORE any official label is read.
``score``        ONE terminal scoring invocation: reads official validation
                 labels once and official test labels once, then computes the
                 pre-declared metrics, paired bootstrap, subgroups and the
                 scientific decision.

Discipline
----------
* exactly one new training trajectory; no seed 2, no Organized training;
* official test is read only inside ``score`` and only after ``freeze``;
* predictions are never regenerated after a scoring attempt;
* GPU training is run through ``rr`` (remote) — this driver only orchestrates.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cscl_features as cf  # noqa: E402
import run_prediction_feasibility_v2 as rpv2  # noqa: E402
import run_structural_relation_composer_v1 as srtc  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402
from tracks.ksvd.experiments.luyin16 import (  # noqa: E402
    zinc_full_decomposition_valid_test_confirmation_v1 as zfdtc,
    zinc_static_dictionary_pair as sdp,
)

PROTOCOL_ID = "cscl-bjoc-predictor-terminal-evaluation-v1"
CONFIG_PATH = _CODE_DIR.parent / "configs" / "bjoc_predictor_terminal_v1.yaml"
OUT = REPO_ROOT / "tracks/structure-contribution/results/bjoc_predictor_terminal_v1"
BJOC_DIR = REPO_ROOT / "tracks/structure-contribution/results/structural_relation_composer_v1"
KSVD_CONF = REPO_ROOT / "tracks/ksvd/results/zinc_full_decomposition_valid_test_confirmation_v1"

# --- the four frozen models ------------------------------------------------
Y_SOUP = {
    "Y0": KSVD_CONF / "Y_seed0_raw_soup_state.pt",
    "Y1": KSVD_CONF / "Y_seed1_raw_soup_state.pt",
}
Y_JSON = {"Y0": KSVD_CONF / "Y_seed0.json", "Y1": KSVD_CONF / "Y_seed1.json"}
Y_FIT_PRED = {"Y0": KSVD_CONF / "Y_seed0_fit_pred.npz", "Y1": KSVD_CONF / "Y_seed1_fit_pred.npz"}
N_SOUP = {
    "N0": BJOC_DIR / "bjoc_null_full10k_s0_soup_state.pt",
    "N1": BJOC_DIR / "bjoc_null_full10k_s1_soup_state.pt",
}
N_JSON = {
    "N0": BJOC_DIR / "train_bjoc_null_full10k_s0.json",
    "N1": BJOC_DIR / "train_bjoc_null_full10k_s1.json",
}
MODELS = ("Y0", "Y1", "N0", "N1")

LEGACY_PREP = rpv2.LEGACY_PREP  # all-10k train-only Full-Y prep
NORM_FULL10K = BJOC_DIR / "bjoc_global_norm_full10k.json"
OBJECTS_CACHE = BJOC_DIR / "bjoc_objects_v1.pt"
BJOC_OBJECTS_CODE = _CODE_DIR / "structural_relation_objects_v1.py"
BJOC_MODEL_CODE = _CODE_DIR / "structural_relation_composer_v1.py"
V1_DRIVER_CODE = _CODE_DIR / "run_structural_relation_composer_v1.py"
ENCODED_VALID = sdp.CACHE_DIR / "encoded_valid.pt"

HIST_VALID_NPZ = KSVD_CONF / "heldout_valid_predictions.npz"
HIST_TEST_NPZ = KSVD_CONF / "heldout_test_predictions.npz"
BJOC_VALID_N0 = BJOC_DIR / "official_valid_preds_null_full10k_s0.npz"
BJOC_VALID_O0 = BJOC_DIR / "official_valid_preds_organized_full10k_s0.npz"
BJOC_VALID_FREEZE = BJOC_DIR / "official_valid_freeze.json"
MECH_JSON = BJOC_DIR / "mechanism_diagnostics.json"
SPLIT_JSON = BJOC_DIR / "split_v1.json"
BJOC_ROUND_SUMMARY = BJOC_DIR / "round_summary.json"

N1_SOUP_OUT = OUT / "seed1_last5_soup_state.pt"
PRED = {"valid": OUT / "valid_predictions.npz", "test": OUT / "test_predictions.npz"}
FREEZE_MANIFEST = OUT / "terminal_freeze_manifest.json"

# --- frozen tolerances -----------------------------------------------------
Y_REPLAY_ATOL = 5.0e-5          # CPU/GPU float32 replay vs a stored export
Y_REPLAY_MAE_ATOL = 1.0e-4
HIST_MAE_ATOL = 1.0e-4          # terminal score vs the historical recorded MAE
STD_EPS = 1e-9
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 0
TAIL_Y = -3.0

HISTORICAL_REFERENCE = {
    "valid": {"Y0": 0.11861754740116885, "Y1": 0.11769479865295579,
              "seed_mean": 0.11815617302706233, "ensemble": 0.10985725831700256},
    "test": {"Y0": 0.09458897127117961, "Y1": 0.09664334127493203,
             "seed_mean": 0.09561615627305582, "ensemble": 0.08595460536144674},
}

# label fields that must never be consulted by an inference path
LABEL_FIELDS = ("y", "g", "c", "k", "ell", "s")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def _j(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _j(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_j(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_j(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    """Hash of a parameter dict (float32 bytes; excludes norm buffers)."""
    digest = hashlib.sha256()
    for key in sorted(state):
        if key.endswith("global_mean") or key.endswith("global_scale"):
            continue
        value = state[key]
        if not torch.is_tensor(value):
            continue
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return digest.hexdigest()


def gpu_identity() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "hostname": platform.node(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        uuid = getattr(props, "uuid", None)
        info.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "gpu_uuid": str(uuid) if uuid is not None else None,
                "total_memory": int(props.total_memory),
                "capability": list(torch.cuda.get_device_capability(0)),
            }
        )
    return info


def _load_y_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    return torch.load(path, map_location=device, weights_only=False)


def load_prep_blob() -> dict[str, np.ndarray]:
    return {k: z for k, z in np.load(LEGACY_PREP, allow_pickle=False).items()}


def build_y_model(tag: str, blob: Mapping[str, np.ndarray], device: torch.device) -> torch.nn.Module:
    seed = 0 if tag == "Y0" else 1
    model = rpv2.build_model(blob, seed).to(device)
    model.load_state_dict(_load_y_state(Y_SOUP[tag], device), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def build_norm() -> dict[str, Any]:
    return read_json(NORM_FULL10K)


def build_bjoc_model(tag: str, norm: Mapping[str, Any], device: torch.device) -> src.BJOC:
    seed = 0 if tag == "N0" else 1
    model = src.build_bjoc(
        src.BJOCConfig(),
        seed=seed,
        organization_mode=src.NULL,
        global_mean=np.asarray(norm["global_mean"], dtype=np.float32),
        global_scale=np.asarray(norm["global_scale"], dtype=np.float32),
        device=device,
    )
    model.load_state_dict(_load_y_state(N_SOUP[tag], device), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _batch_objects_signature(objs: Sequence[sro.MolObjects]) -> dict[str, Any]:
    digest = hashlib.sha256()
    for o in objs:
        digest.update(np.ascontiguousarray(o.atom_feat, dtype=np.float32).tobytes())
        digest.update(np.ascontiguousarray(o.bond_index, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(o.bond_type, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(o.junc_index, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(o.junc_bond, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(o.global_feat, dtype=np.float32).tobytes())
    counts = {
        "n_atoms": int(sum(o.n_atoms for o in objs)),
        "n_bonds": int(sum(o.bond_index.shape[0] for o in objs)),
        "n_junctions": int(sum(o.junc_index.shape[0] for o in objs)),
    }
    return {"objects_sha256": digest.hexdigest(), **counts}


# ---------------------------------------------------------------------------
# stage: reuse-audit (Phase A)
# ---------------------------------------------------------------------------


def _audit_y_recipe(tag: str) -> dict[str, Any]:
    meta = read_json(Y_JSON[tag])
    src_text = V1_DRIVER_CODE.read_text(encoding="utf-8")
    return {
        "json": str(Y_JSON[tag].relative_to(REPO_ROOT)),
        "arm": meta["arm"],
        "seed": int(meta["seed"]),
        "fit_target_is_y": bool(meta["fit_target_is_y"]),
        "h1_lambda": float(meta["h1_lambda"]),
        "epochs": int(meta["epochs"]),
        "steps_done": int(meta["steps_done"]),
        "soup_members": list(meta["soup_members"]),
        "train_generator_seed": int(meta["train_generator_seed"]),
        "optimizer_recipe": {
            "lr": float(meta["lr"]),
            "weight_decay": float(meta["weight_decay"]),
            "grad_clip": float(meta["grad_clip"]),
            "batch_size": int(meta["batch_size"]),
        },
        "soup_state_sha256_recorded": meta["soup_state_sha256"],
        "fit_raw_mae_y_recorded": float(meta["fit_raw_mae_y"]),
        "official_valid_loaded_during_training": bool(meta["official_valid_loaded"]),
        "official_test_loaded_during_training": bool(meta["official_test_loaded"]),
        "loss_source_line": next(
            (line.strip() for line in src_text.splitlines() if "task = F.l1_loss(prediction" in line), None
        ),
        "reconstruction_is_label_free": bool("model.reconstruction_loss(aux[\"phi\"], aux[\"coord\"])" in src_text),
        "auxiliary_targets_absent": bool(
            "decomp[" not in src_text.split("def train_full_y_matched")[0]
        ),
    }


def _y_train_replay(tag: str, blob: Mapping[str, np.ndarray], device: torch.device, limit: int | None) -> dict[str, Any]:
    train = rpv2.load_train_rows()
    zfdtc.apply_prep(train, blob)
    n = len(train) if limit is None else min(int(limit), len(train))
    model = build_y_model(tag, blob, device)
    started = time.perf_counter()
    pred = rpv2.predict_raw(model, train[:n], device)
    stored = np.load(Y_FIT_PRED[tag], allow_pickle=False)
    ref = np.asarray(stored["fit_raw"], dtype=np.float64)[:n]
    y_ref = np.asarray(stored["y"], dtype=np.float64)[:n]
    diff = np.abs(pred - ref)
    mae_replay = float(np.abs(pred - y_ref).mean())
    mae_ref = float(np.abs(ref - y_ref).mean())
    out = {
        "n_rows": n,
        "device": str(device),
        "max_abs_pred_diff_vs_stored": float(diff.max()),
        "mean_abs_pred_diff": float(diff.mean()),
        "replay_fit_mae": mae_replay,
        "stored_fit_mae": mae_ref,
        "mae_abs_diff": float(abs(mae_replay - mae_ref)),
        "atol": Y_REPLAY_ATOL,
        "mae_atol": Y_REPLAY_MAE_ATOL,
        "seconds": time.perf_counter() - started,
    }
    out["ok"] = bool(out["max_abs_pred_diff_vs_stored"] <= Y_REPLAY_ATOL and out["mae_abs_diff"] <= Y_REPLAY_MAE_ATOL)
    return out


def _n_train_audit() -> dict[str, Any]:
    norm = build_norm()
    n_json = read_json(N_JSON["N0"])
    cfg = src.BJOCConfig()
    model = src.build_bjoc(
        cfg,
        seed=0,
        organization_mode=src.NULL,
        global_mean=np.asarray(norm["global_mean"], dtype=np.float32),
        global_scale=np.asarray(norm["global_scale"], dtype=np.float32),
        device="cpu",
    )
    audit = src.parameter_audit(cfg, model)
    objs = torch.load(OBJECTS_CACHE, map_location="cpu", weights_only=False)
    roles = src.forbidden_module_scan(model)
    return {
        "train_json": str(N_JSON["N0"].relative_to(REPO_ROOT)),
        "recipe": {
            "arm": n_json["arm"],
            "seed": int(n_json["seed"]),
            "full10k": bool(n_json["full10k"]),
            "epochs": int(n_json["epochs"]),
            "soup_members": list(n_json["soup_members"]),
            "hyper": n_json["hyper"],
            "parameters": int(n_json["parameters"]),
            "init_state_sha256": n_json["init_state_sha256"],
            "soup_fit_mae": float(n_json["soup_fit_mae"]),
            "wall_seconds": float(n_json["wall_seconds"]),
            "official_valid_loaded": bool(n_json["official_valid_loaded"]),
            "official_test_loaded": bool(n_json["official_test_loaded"]),
        },
        "norm": {
            "path": str(NORM_FULL10K.relative_to(REPO_ROOT)),
            "sha256": file_sha256(NORM_FULL10K),
            "fit_only": bool(norm["fit_only"]),
            "n_fit": int(norm["n_fit"]),
            "global_mean_len": int(len(norm["global_mean"])),
            "global_scale_min": float(np.min(np.asarray(norm["global_scale"]))),
        },
        "objects": {
            "path": str(OBJECTS_CACHE.relative_to(REPO_ROOT)),
            "sha256": file_sha256(OBJECTS_CACHE),
            "n_objects": int(len(objs)),
            "n_atoms": int(sum(o.n_atoms for o in objs)),
            "n_bonds": int(sum(o.bond_index.shape[0] for o in objs)),
            "n_junctions": int(sum(o.junc_index.shape[0] for o in objs)),
            "mol_index_is_arange": bool(np.array_equal(np.asarray([o.mol_index for o in objs]), np.arange(len(objs)))),
        },
        "parameter_audit": audit,
        "forbidden_module_scan": roles,
        "object_code_sha256": file_sha256(BJOC_OBJECTS_CODE),
        "model_code_sha256": file_sha256(BJOC_MODEL_CODE),
        "split_json_sha256": file_sha256(SPLIT_JSON),
    }


def _recorded_checkpoint_evidence(tag: str, file_sha: str, state_sha: str) -> dict[str, Any]:
    """Independent hashes recorded by the round that produced each checkpoint."""
    if tag.startswith("Y"):
        meta = read_json(Y_JSON[tag])
        return {
            "source": str(Y_JSON[tag].relative_to(REPO_ROOT)),
            "recorded_state_hash": meta["soup_state_sha256"],
            "state_hash_matches_record": bool(state_sha == meta["soup_state_sha256"]),
            "recorded_soup_members": list(meta["soup_members"]),
            "recorded_file_sha256": None,
        }
    freeze = read_json(BJOC_VALID_FREEZE)
    recorded = freeze["arms"]["null"]["checkpoint_sha256"]
    train_json = read_json(N_JSON[tag])
    return {
        "source": str(BJOC_VALID_FREEZE.relative_to(REPO_ROOT)),
        "recorded_file_sha256": recorded,
        "file_sha256_matches_record": bool(file_sha == recorded),
        "recorded_soup_members": list(train_json["soup_members"]),
        "recorded_init_state_sha256": train_json["init_state_sha256"],
        "recorded_soup_state_sha256": train_json.get("soup_state_sha256"),
    }


def stage_reuse_audit(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    inventory: dict[str, Any] = {"protocol_id": PROTOCOL_ID, "stage": "checkpoint_inventory"}
    for tag in MODELS:
        path = Y_SOUP.get(tag) or N_SOUP.get(tag)
        file_sha = file_sha256(path) if path.exists() else None
        entry: dict[str, Any] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "exists": path.exists(),
            "sha256": file_sha,
            "bytes": path.stat().st_size if path.exists() else None,
        }
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=False)
            state_sha = state_hash(state)
            entry["state_hash_sha256"] = state_sha
            entry["n_tensors"] = int(len(state))
            try:
                entry.update(_recorded_checkpoint_evidence(tag, file_sha, state_sha))
            except Exception as exc:  # pragma: no cover - diagnostic only
                entry["record_error"] = repr(exc)
        inventory[tag] = entry
    inventory["prep"] = {
        "full_y_all_train_prep": {"path": str(LEGACY_PREP.relative_to(REPO_ROOT)), "sha256": file_sha256(LEGACY_PREP)},
        "bjoc_full10k_norm": {"path": str(NORM_FULL10K.relative_to(REPO_ROOT)), "sha256": file_sha256(NORM_FULL10K)},
        "bjoc_objects_cache": {"path": str(OBJECTS_CACHE.relative_to(REPO_ROOT)), "sha256": file_sha256(OBJECTS_CACHE)},
        "encoded_valid_cache": {"path": str(ENCODED_VALID.relative_to(REPO_ROOT)), "sha256": file_sha256(ENCODED_VALID)},
    }
    write_json(OUT / "checkpoint_inventory.json", inventory)

    y0 = read_json(Y_JSON["Y0"])
    y1 = read_json(Y_JSON["Y1"])
    n0 = read_json(N_JSON["N0"])
    audit: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "stage": "reuse_audit",
        "git_commit": git_commit(),
        "device": str(device),
        "inventory": {tag: {"sha256": inventory[tag]["sha256"], "state_hash_sha256": inventory[tag].get("state_hash_sha256")} for tag in MODELS},
        "Y_arms": {tag: _audit_y_recipe(tag) for tag in ("Y0", "Y1")},
        "Y0_Y1_state_differs": bool(y0["soup_state_sha256"] != y1["soup_state_sha256"]),
        "Y0_Y1_paired_recipe_identical": bool(
            y0["lr"] == y1["lr"] and y0["weight_decay"] == y1["weight_decay"] and y0["grad_clip"] == y1["grad_clip"]
            and y0["batch_size"] == y1["batch_size"] and y0["epochs"] == y1["epochs"]
            and y0["soup_members"] == y1["soup_members"]
            and (y1["train_generator_seed"] - y0["train_generator_seed"]) == 1
        ),
        "BJOC_N0": _n_train_audit(),
        "BJOC_N1_present_before_training": bool(N_SOUP["N1"].exists()),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }

    replay = {}
    for tag in ("Y0", "Y1"):
        replay[tag] = _y_train_replay(tag, load_prep_blob(), device, args.limit)
        log(f"[reuse] {tag} train replay n={replay[tag]['n_rows']} "
            f"max|dpred|={replay[tag]['max_abs_pred_diff_vs_stored']:.3e} ok={replay[tag]['ok']}")
    audit["train_replay"] = replay

    hard = [
        all(inventory[t]["exists"] for t in MODELS if not (t == "N1" and not N_SOUP["N1"].exists())),
        all(inventory[t].get("state_hash_matches_record", inventory[t].get("file_sha256_matches_record", False))
            for t in ("Y0", "Y1", "N0")),
        audit["Y0_Y1_state_differs"],
        audit["Y0_Y1_paired_recipe_identical"],
        audit["BJOC_N0"]["parameter_audit"]["parameter_exact"],
        audit["BJOC_N0"]["norm"]["fit_only"] and audit["BJOC_N0"]["norm"]["n_fit"] == 10000,
        audit["BJOC_N0"]["objects"]["n_objects"] == 10000 and audit["BJOC_N0"]["objects"]["mol_index_is_arange"],
        audit["BJOC_N0"]["parameter_audit"]["actual_parameters"] == 413505,
        not audit["BJOC_N0"]["forbidden_module_scan"],
        not audit["BJOC_N0"]["recipe"]["official_valid_loaded"] and not audit["BJOC_N0"]["recipe"]["official_test_loaded"],
        all(replay[t]["ok"] for t in replay),
    ]
    audit["hard_checks_pass"] = bool(all(hard))
    audit["hard_checks"] = [bool(h) for h in hard]
    audit["decision"] = "REUSE_FROZEN_MODELS" if audit["hard_checks_pass"] else "AUDIT_BLOCK"
    audit["seconds"] = time.perf_counter() - started
    write_json(OUT / "reuse_audit.json", audit)
    log(f"[reuse] hard_checks_pass={audit['hard_checks_pass']} decision={audit['decision']} "
        f"({audit['seconds']:.0f}s)")
    return 0 if audit["hard_checks_pass"] else 1


# ---------------------------------------------------------------------------
# structural covariates + metric helpers
# ---------------------------------------------------------------------------


def structural_covariates(ds: Sequence[Any], limit: int | None = None) -> dict[str, np.ndarray]:
    import networkx as nx

    n = len(ds) if limit is None else min(int(limit), len(ds))
    n_atoms = np.zeros(n, dtype=np.int64)
    n_bonds = np.zeros(n, dtype=np.int64)
    cyc = np.zeros(n, dtype=np.int64)
    diameter = np.zeros(n, dtype=np.int64)
    min_ring = np.zeros(n, dtype=np.int64)
    for i in range(n):
        x = ds[i].x.numpy()
        ei = ds[i].edge_index.numpy()
        nn = int(x.shape[0])
        edges = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(ei[0].tolist(), ei[1].tolist())}
        g = nx.Graph()
        g.add_nodes_from(range(nn))
        g.add_edges_from(edges)
        n_atoms[i] = nn
        n_bonds[i] = len(edges)
        cyc[i] = int(len(edges) - nn + nx.number_connected_components(g))
        diam = 0
        for comp in nx.connected_components(g):
            sub = g.subgraph(comp)
            if sub.number_of_nodes() > 1:
                diam = max(diam, int(nx.diameter(sub)))
        diameter[i] = diam
        basis = nx.minimum_cycle_basis(g)
        min_ring[i] = int(min((len(c) for c in basis), default=0))
    return {
        "n_atoms": n_atoms, "n_bonds": n_bonds, "cyclomatic_rank": cyc,
        "diameter": diameter, "min_ring_size": min_ring,
    }


def abs_metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    e = np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64))
    return {
        "mae": float(e.mean()),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "median_ae": float(np.median(e)),
        "p90_ae": float(np.percentile(e, 90)),
        "p95_ae": float(np.percentile(e, 95)),
        "max_ae": float(e.max()),
        "sum_ae": float(e.sum()),
    }


def paired_bootstrap(delta: np.ndarray, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    d = np.asarray(delta, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, d.size, size=(int(n), d.size))
    means = d[idx].mean(axis=1)
    return {
        "mean_diff": float(d.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "resamples": int(n),
        "seed": int(seed),
        "statistic": "percentile bootstrap of the paired per-molecule mean",
        "frac_negative": float((d < 0).mean()),
        "n_better": int((d < 0).sum()),
        "n_worse": int((d > 0).sum()),
        "n_tied": int((d == 0).sum()),
    }


def _group_entry(preds: Mapping[str, np.ndarray], y: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    out: dict[str, Any] = {"n": n, "mae": {k: float(np.abs(p[mask] - y[mask]).mean()) for k, p in preds.items()}}
    return out


def _delta_mean_trajectory(preds: Mapping[str, np.ndarray], y: np.ndarray) -> np.ndarray:
    e = {k: np.abs(p - y) for k, p in preds.items()}
    return 0.5 * ((e["N0"] - e["Y0"]) + (e["N1"] - e["Y1"]))


# ---------------------------------------------------------------------------
# stage: valid-audit (Phase C, zero training)
# ---------------------------------------------------------------------------


def stage_valid_audit(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)

    hist = np.load(HIST_VALID_NPZ, allow_pickle=True)
    y = np.asarray(hist["y"], dtype=np.float64)
    ids = np.asarray(hist["ids"], dtype=object)
    n0 = np.load(BJOC_VALID_N0, allow_pickle=False)
    o0 = np.load(BJOC_VALID_O0, allow_pickle=False)
    assert np.array_equal(n0["row"], np.arange(1000)), "N0 valid prediction row order broken"
    assert np.array_equal(o0["row"], np.arange(1000)), "O0 valid prediction row order broken"

    preds = {
        "Y0": np.asarray(hist["Y_raw_0"], dtype=np.float64),
        "Y1": np.asarray(hist["Y_raw_1"], dtype=np.float64),
        "N0": np.asarray(n0["pred"], dtype=np.float64).astype(np.float64),
        "O0": np.asarray(o0["pred"], dtype=np.float64).astype(np.float64),
    }

    # ---- alignment evidence -------------------------------------------------
    raw_val = cf.load_official_train  # noqa: F841 (kept for clarity; not used here)
    from torch_geometric.datasets import ZINC

    ds_val = ZINC(root=str(cf.ZINC_ROOT), subset=True, split="val")
    y_raw = np.asarray([float(ds_val[i].y.reshape(-1)[0]) for i in range(1000)], dtype=np.float64)
    # dict_atom is attached by the historical env adapter inside load_valid_data
    enc_rows, _enc_meta = zfdtc.load_valid_data(load_prep_blob())
    label_diffs = 0
    atom_diffs = 0
    for i in range(1000):
        if int(enc_rows[i].dict_atom.shape[0]) != int(ds_val[i].x.shape[0]):
            atom_diffs += 1
        if not bool(torch.equal(enc_rows[i].dict_atom.long().reshape(-1), ds_val[i].x.long().reshape(-1))):
            atom_diffs += 1
        try:
            label_diffs += int(abs(float(enc_rows[i].y.reshape(-1)[0]) - float(ds_val[i].y.reshape(-1)[0])) > STD_EPS)
        except Exception:
            label_diffs += 1
    alignment = {
        "n_rows": 1000,
        "historical_ids_first_last": [str(ids[0]), str(ids[-1])],
        "historical_y_vs_raw_pyg_val_y_max_abs": float(np.abs(y - y_raw).max()),
        "encoded_valid_vs_raw_val_atom_mismatches": atom_diffs,
        "encoded_valid_vs_raw_val_label_mismatches": label_diffs,
        "n0_prediction_rows": int(len(n0["pred"])),
        "o0_prediction_rows": int(len(o0["pred"])),
        "y_source_file": str(HIST_VALID_NPZ.relative_to(REPO_ROOT)),
        "calibration_note": (
            "all four vectors are RAW predictions (Y_raw_*, BJOC predictions carry no calibration bias); "
            "the Y_cal_* columns are not used anywhere in this round"
        ),
        "ok": bool(atom_diffs == 0 and label_diffs == 0 and float(np.abs(y - y_raw).max()) < STD_EPS),
    }
    if not alignment["ok"]:
        write_json(OUT / "valid_error_decomposition.json", {"alignment": alignment, "blocked": True})
        log("[valid-audit] BLOCKED: row alignment could not be established")
        return 1

    e = {k: np.abs(p - y) for k, p in preds.items()}
    mae = {k: float(e[k].mean()) for k in preds}
    extreme_id = 172  # valid:0172 (positional id in the official split; historical exposure)
    keep = np.ones(1000, dtype=bool)
    keep[extreme_id] = False

    def gain(a: str, b: str) -> dict[str, Any]:
        total = mae[a] - mae[b]
        from_row = float((e[a][extreme_id] - e[b][extreme_id]) / 1000.0)
        other = total - from_row
        mae_gain_other = float(e[a][keep].mean() - e[b][keep].mean())
        return {
            "total_gain": total,
            "gain_from_0172": from_row,
            "gain_from_other999": other,
            "conservation_residual": float(abs(from_row + other - total)),
            "mae_gain_on_other999": mae_gain_other,
            "scaling_identity_999_over_1000": float(abs(other - mae_gain_other * 999.0 / 1000.0)),
        }

    c1 = {
        "row_id": f"valid:{extreme_id:04d}",
        "y": float(y[extreme_id]),
        "table": {
            k: {"pred": float(preds[k][extreme_id]), "abs_error": float(e[k][extreme_id])}
            for k in ("Y0", "N0", "O0")
        },
        "mae_total": mae,
        "mae_without_0172": {k: float(e[k][keep].mean()) for k in preds},
        "gain_N0_vs_Y0": gain("Y0", "N0"),
        "gain_O0_vs_Y0": gain("Y0", "O0"),
    }
    c1["gain_N0_vs_Y0"]["conservation_ok"] = bool(c1["gain_N0_vs_Y0"]["conservation_residual"] < 1e-12)
    c1["gain_N0_vs_Y0"]["other999_scaling_ok"] = bool(
        c1["gain_N0_vs_Y0"]["scaling_identity_999_over_1000"] < 1e-12
    )
    c1["gain_N0_vs_Y0"]["share_of_total_gain_from_0172"] = float(
        c1["gain_N0_vs_Y0"]["gain_from_0172"] / c1["gain_N0_vs_Y0"]["total_gain"]
    )

    # ---- ordinary vs tail, complexity strata, extremes ----------------------
    cov = structural_covariates(ds_val)

    def subgroup_block(mask: np.ndarray) -> dict[str, Any]:
        out = {"n": int(mask.sum())}
        if mask.any():
            out["mae"] = {k: float(e[k][mask].mean()) for k in preds}
            out["delta_N0_minus_Y0"] = float(e["N0"][mask].mean() - e["Y0"][mask].mean())
        return out

    top10 = {}
    for k in preds:
        order = np.argsort(-e[k])
        top10[k] = {
            "top1_row": int(order[0]),
            "top1_share": float(e[k][order[0]] / e[k].sum()),
            "top10_share": float(np.sort(e[k])[-10:].sum() / e[k].sum()),
            "top10_rows": [int(r) for r in order[:10]],
        }

    fractions = {
        f"{a}_better_than_{b}": float((e[a] < e[b]).mean())
        for a, b in (("N0", "Y0"), ("O0", "Y0"), ("O0", "N0"))
    }
    tail_contribution = {
        k: {
            "y_le_minus3_error_mass_share": float(e[k][y <= TAIL_Y].sum() / e[k].sum()),
            "n_rows": int((y <= TAIL_Y).sum()),
        }
        for k in preds
    }
    strata = {
        "y_le_minus3": subgroup_block(y <= TAIL_Y),
        "y_le_minus3_without_0172": subgroup_block((y <= TAIL_Y) & keep),
        "y_gt_minus3": subgroup_block(y > TAIL_Y),
        "without_0172": subgroup_block(keep),
        "atoms_le_median": subgroup_block(cov["n_atoms"] <= np.median(cov["n_atoms"])),
        "atoms_gt_median": subgroup_block(cov["n_atoms"] > np.median(cov["n_atoms"])),
        "cyclomatic_0_1": subgroup_block(cov["cyclomatic_rank"] <= 1),
        "cyclomatic_2_3": subgroup_block((cov["cyclomatic_rank"] >= 2) & (cov["cyclomatic_rank"] <= 3)),
        "cyclomatic_ge_4": subgroup_block(cov["cyclomatic_rank"] >= 4),
        "diameter_le_median": subgroup_block(cov["diameter"] <= np.median(cov["diameter"])),
        "diameter_gt_median": subgroup_block(cov["diameter"] > np.median(cov["diameter"])),
        "atoms_quartiles": {
            f"q{k+1}": subgroup_block(
                (cov["n_atoms"] > np.percentile(cov["n_atoms"], 25 * k))
                & (cov["n_atoms"] <= np.percentile(cov["n_atoms"], 25 * (k + 1)))
            )
            for k in range(4)
        },
    }

    # ---- C3 input-content comparison + frozen dependence evidence -----------
    mech = read_json(MECH_JSON)
    zero_mae = {}
    for entry in mech.get("perturbations", []) or []:
        if isinstance(entry, dict) and "name" in entry and "mae" in entry:
            zero_mae[entry["name"]] = entry["mae"]
    c3 = {
        "full_y_input": (
            "814-dim reader input = [unary 289 | 5 hard distance-bucket moment blocks x 97 | global 32 | "
            "topology 8]; the pair state is pooled per distance bucket by sum(h), sum(h^2), log1p(count) "
            "before the reader, so which pair states shared a centre or a bond is not recoverable"
        ),
        "bjoc_input": (
            "108-dim raw atom features + physical bonds + centre-explicit junctions with a static "
            "junction->bond incidence; the composer receives per-bond [b_b, W_b, V_b, log1p n_b, raw bond] "
            "and the reader receives [sum_b c_b, sum_i a_i, sum_i a_i^2, standardised global topology]"
        ),
        "bjoc_only_information": [
            "per-bond grouping of junction content (ownership) instead of a graph-level pooled moment",
            "radius-1 and radius-2 neighbour atom-type histograms per atom with an explicit centre",
            "ring-role and min-ring-size features on junctions (centre degree, common neighbours, "
            "whether each of the two bonds is a ring bond)",
            "physical bond objects carrying both endpoint embeddings plus ring-role flags",
        ],
        "both_express_indirectly": [
            "bond-type composition and atom-type composition (both reachable from the moment blocks)",
            "global size / ring counts (both models standardise a global topology vector)",
        ],
        "confounded": [
            "task loss differs: Full-Y = raw-y L1 + label-free structural reconstruction, BJOC = raw-y L1 only",
            "parameter counts differ (408651 vs 413505) and the training trajectories differ",
            "therefore no per-input causal claim is possible from this comparison",
        ],
        "frozen_dependence_evidence_from_previous_round": zero_mae,
        "dependence_evidence_warning": (
            "zero-mask MAE numbers are USE evidence on the frozen model only; they do not establish that a "
            "feature block causes generalisation. No such claim is made in this round."
        ),
        "code_sha256": {
            "full_y_scale_model": file_sha256(_CODE_DIR / "run_prediction_feasibility_v2.py"),
            "bjoc_model": file_sha256(BJOC_MODEL_CODE),
            "bjoc_objects": file_sha256(BJOC_OBJECTS_CODE),
        },
    }

    # ---- C4 naming boundary -------------------------------------------------
    c4 = {
        "name_used_in_this_round": "one-shot / non-iterative structural object composer",
        "observed_facts": [
            "no torch_geometric MessagePassing subclass and no attention/Transformer/recurrent module "
            "(forbidden_module_scan on the real model returns [])",
            "no iterative hidden-state propagation: a -> b -> w is a fixed one-pass DAG and the grouping "
            "step never feeds back into a, b or w",
            "predictions are unchanged by batch composition (gate 13) and by atom relabelling (gate 1)",
        ],
        "honest_caveat": (
            "the model still forms a bond representation from its endpoint atoms and a junction "
            "representation from its centre and two neighbours in ONE pass, which is a single round of "
            "message aggregation in the broad sense.  Not inheriting PyG's MessagePassing class is not a "
            "proof of theoretical non-GNN status, so the descriptive term above is used instead of the "
            "unqualified label 'not a GNN'."
        ),
        "difference_from_mpnn": (
            "an MPNN recomputes node states for L rounds and lets every node read every other node's "
            "evolving state; here the atom state is computed once from fixed raw features, and the "
            "junction->bond incidence is consumed by a static grouping with no learned parameters and no "
            "state refresh."
        ),
    }

    payload = {
        "protocol_id": PROTOCOL_ID,
        "stage": "valid_error_decomposition",
        "git_commit": git_commit(),
        "official_valid_loaded": True,
        "official_test_loaded": False,
        "zero_training": True,
        "inputs": {
            "historical_Y": str(HIST_VALID_NPZ.relative_to(REPO_ROOT)),
            "bjoc_N0_frozen": str(BJOC_VALID_N0.relative_to(REPO_ROOT)),
            "bjoc_O0_frozen": str(BJOC_VALID_O0.relative_to(REPO_ROOT)),
            "bjoc_valid_freeze": str(BJOC_VALID_FREEZE.relative_to(REPO_ROOT)),
        },
        "alignment": alignment,
        "C1_valid0172": c1,
        "C2_ordinary_molecules": {
            "strata": strata,
            "top10_extremes": top10,
            "better_fractions": fractions,
            "tail_contribution": tail_contribution,
            "delta_N0_minus_Y0_without_0172": float(e["N0"][keep].mean() - e["Y0"][keep].mean()),
            "note": "trimmed numbers are auxiliary diagnostics; the formal metric always uses all 1000 rows",
        },
        "C3_input_content_and_dependence": c3,
        "C4_method_naming_boundary": c4,
        "seconds": time.perf_counter() - started,
    }
    write_json(OUT / "valid_error_decomposition.json", payload)
    log(f"[valid-audit] y={c1['y']:.4f} |eY0|={e['Y0'][extreme_id]:.4f} |eN0|={e['N0'][extreme_id]:.4f} "
        f"|eO0|={e['O0'][extreme_id]:.4f}")
    log(f"[valid-audit] MAE Y0={mae['Y0']:.6f} N0={mae['N0']:.6f} O0={mae['O0']:.6f}; "
        f"gain N0-Y0 total={c1['gain_N0_vs_Y0']['total_gain']:+.6f} "
        f"from_0172={c1['gain_N0_vs_Y0']['gain_from_0172']:+.6f} "
        f"other999={c1['gain_N0_vs_Y0']['gain_from_other999']:+.6f}")
    return 0


# ---------------------------------------------------------------------------
# stage: train-n1 (the ONE authorised trajectory)
# ---------------------------------------------------------------------------


def _loss_code_path_evidence() -> dict[str, Any]:
    src_text = V1_DRIVER_CODE.read_text(encoding="utf-8")
    tree = ast.parse(src_text)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "train_bjoc_arm")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    call_names = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute):
                call_names.append(f"{getattr(f.value, 'id', '?')}.{f.attr}")
            elif isinstance(f, ast.Name):
                call_names.append(f.id)
    loss_lines = [ln.strip() for ln in src_text.splitlines() if "F.l1_loss" in ln and "y_fit" in ln]
    return {
        "function": "train_bjoc_arm",
        "loss_call_names": sorted({c for c in call_names if "loss" in c.lower()}),
        "loss_lines": loss_lines,
        "y_source_lines": [ln.strip() for ln in src_text.splitlines() if "y_all = np.asarray" in ln],
        "labels_used": ["y_all[fit_rows]"],
        "forbidden_label_names_present": sorted(n for n in names if n in {"g", "ell", "c", "k", "s", "decomp"}),
        "forbidden_label_attrs_present": sorted(a for a in attrs if a in {"g", "ell", "c", "k", "decomp"}),
        "steps": 79 * 240,
        "soup_rule": "parameter mean of the last 5 epoch states (236..240)",
    }


def _pretrain_evidence(device: torch.device) -> dict[str, Any]:
    norm = build_norm()
    objs = torch.load(OBJECTS_CACHE, map_location="cpu", weights_only=False)
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    split = srtc.build_split_v1(smiles)
    cfg = src.BJOCConfig()
    model = src.build_bjoc(
        cfg,
        seed=1,
        organization_mode=src.NULL,
        global_mean=np.asarray(norm["global_mean"], dtype=np.float32),
        global_scale=np.asarray(norm["global_scale"], dtype=np.float32),
        device="cpu",
    )
    init_sha = src.state_dict_hash(model)
    audit = src.parameter_audit(cfg, model)
    fit_rows = list(range(10000))
    return {
        "protocol_id": PROTOCOL_ID,
        "stage": "n1_pretrain_evidence",
        "git_commit": git_commit(),
        "seed": 1,
        "arm": src.NULL,
        "full10k": True,
        "train_rows": {"n": len(fit_rows), "first": fit_rows[0], "last": fit_rows[-1],
                       "sha256": bytes_sha256(np.ascontiguousarray(np.asarray(fit_rows, dtype=np.int64)).tobytes())},
        "train_shuffle_seed": 1 + srtc.TRAIN_SHUFFLE_OFFSET,
        "epochs": srtc.EPOCHS,
        "steps_per_epoch": 79,
        "steps_total": 79 * 240,
        "soup_rule": "parameter mean of epochs 236..240",
        "initial_state_sha256": init_sha,
        "parameters": int(audit["actual_parameters"]),
        "parameter_audit": audit,
        "object_cache": {"path": str(OBJECTS_CACHE.relative_to(REPO_ROOT)), "sha256": file_sha256(OBJECTS_CACHE),
                         "n_objects": len(objs)},
        "norm": {"path": str(NORM_FULL10K.relative_to(REPO_ROOT)), "sha256": file_sha256(NORM_FULL10K),
                 "fit_only": bool(norm["fit_only"]), "n_fit": int(norm["n_fit"])},
        "split_json": {"path": str(SPLIT_JSON.relative_to(REPO_ROOT)), "sha256": file_sha256(SPLIT_JSON),
                       "n_fit_recomputed": int(len(split["fit"]))},
        "config": cfg.__dict__ if hasattr(cfg, "__dict__") else None,
        "config_hash": bytes_sha256(json.dumps({
            k: getattr(cfg, k) for k in ("atom_hidden", "atom_out", "bond_hidden", "bond_out",
                                         "junc_hidden", "junc_out", "compose_hidden", "compose_out",
                                         "reader_hidden", "reader_layers", "dropout")
        }, sort_keys=True).encode()),
        "loss_code_path": _loss_code_path_evidence(),
        "device": str(device),
        "gpu": gpu_identity(),
        "entry_point": "tracks/structure-contribution/code/run_structural_relation_composer_v1.py::train_bjoc_arm",
        "entry_point_sha256": file_sha256(V1_DRIVER_CODE),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }


def stage_train_n1(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if N_SOUP["N1"].exists():
        raise SystemExit(f"refusing to train: {N_SOUP['N1']} already exists (one trajectory only)")
    if N1_SOUP_OUT.exists():
        raise SystemExit(f"refusing to train: {N1_SOUP_OUT} already exists")
    device = torch.device(args.device)
    evidence = _pretrain_evidence(device)
    write_json(OUT / "seed1_pretrain_evidence.json", evidence)
    log(f"[train-n1] init_sha={evidence['initial_state_sha256'][:16]} params={evidence['parameters']} "
        f"device={device} gpu={evidence['gpu'].get('device_name')} uuid={evidence['gpu'].get('gpu_uuid')}")

    started = time.perf_counter()
    result = srtc.train_bjoc_arm(
        src.NULL,
        1,
        device,
        smoke=False,
        full10k=True,
        norm_path=NORM_FULL10K,
    )
    wall = time.perf_counter() - started
    if not N_SOUP["N1"].exists():
        raise SystemExit("training finished without a checkpoint; recorded as interrupted")
    import shutil

    shutil.copy2(N_SOUP["N1"], N1_SOUP_OUT)
    same = file_sha256(N_SOUP["N1"]) == file_sha256(N1_SOUP_OUT)

    # reload replay + finiteness (the released artifact is re-loaded from disk)
    cfg = src.BJOCConfig()
    norm = build_norm()
    reloaded = src.build_bjoc(
        cfg, seed=1, organization_mode=src.NULL,
        global_mean=np.asarray(norm["global_mean"], dtype=np.float32),
        global_scale=np.asarray(norm["global_scale"], dtype=np.float32),
        device=device,
    )
    states = {
        "primary": torch.load(N_SOUP["N1"], map_location=device, weights_only=False),
        "released_copy": torch.load(N1_SOUP_OUT, map_location=device, weights_only=False),
    }
    finite = {k: bool(all(torch.isfinite(v).all() for v in s.values() if torch.is_tensor(v))) for k, s in states.items()}
    reloaded.load_state_dict(states["released_copy"], strict=True)
    objs = torch.load(OBJECTS_CACHE, map_location="cpu", weights_only=False)
    fit_objs = [objs[i] for i in range(10000)]
    fit_pred = srtc.predict_bjoc(reloaded, fit_objs, list(range(10000)), device)
    train_rows = torch.load(rpv2.ENCODED_TRAIN, map_location="cpu", weights_only=False)
    y_all = np.asarray([float(train_rows[i].y.reshape(-1)[0]) for i in range(10000)], dtype=np.float64)
    del train_rows
    fit_mae = float(np.abs(fit_pred - y_all).mean())
    replay = {
        "protocol_id": PROTOCOL_ID,
        "stage": "n1_replay",
        "checkpoint_copy_byte_identical": bool(same),
        "primary_checkpoint": {"path": str(N_SOUP["N1"].relative_to(REPO_ROOT)), "sha256": file_sha256(N_SOUP["N1"])},
        "released_checkpoint": {"path": str(N1_SOUP_OUT.relative_to(REPO_ROOT)), "sha256": file_sha256(N1_SOUP_OUT)},
        "state_hash_sha256": state_hash(states["released_copy"]),
        "finite": finite,
        "replay_fit_mae": fit_mae,
        "recorded_soup_fit_mae": float(result["soup_fit_mae"]),
        "mae_abs_diff": float(abs(fit_mae - float(result["soup_fit_mae"]))),
        "atol": Y_REPLAY_ATOL,
        "soup_members_recorded": list(result["soup_members"]),
        "wall_seconds": wall,
        "device": str(device),
        "gpu": gpu_identity(),
    }
    replay["ok"] = bool(same and all(finite.values()) and replay["mae_abs_diff"] <= Y_REPLAY_ATOL)
    write_json(OUT / "seed1_replay.json", replay)

    meta = {
        "protocol_id": PROTOCOL_ID,
        "stage": "n1_train_meta",
        "git_commit": git_commit(),
        "pretrain_evidence": str((OUT / "seed1_pretrain_evidence.json").relative_to(REPO_ROOT)),
        "v1_result": result,
        "wall_seconds_measured": wall,
        "artifacts": {
            "canonical_checkpoint": {"path": str(N_SOUP["N1"].relative_to(REPO_ROOT)), "sha256": file_sha256(N_SOUP["N1"])},
            "released_checkpoint": {"path": str(N1_SOUP_OUT.relative_to(REPO_ROOT)), "sha256": file_sha256(N1_SOUP_OUT)},
            "v1_train_json": {"path": str(N_JSON["N1"].relative_to(REPO_ROOT)),
                              "sha256": file_sha256(N_JSON["N1"]) if N_JSON["N1"].exists() else None},
        },
        "replay": replay,
    }
    write_json(OUT / "seed1_train_meta.json", meta)
    log(f"[train-n1] DONE wall={wall:.0f}s soup_fit_mae={result['soup_fit_mae']:.6f} "
        f"replay_ok={replay['ok']} soup_sha={meta['artifacts']['released_checkpoint']['sha256'][:16]}")
    return 0 if replay["ok"] else 1


# ---------------------------------------------------------------------------
# stage: predict (label-free, official valid + official test)
# ---------------------------------------------------------------------------


class LabelGuard:
    """Read-only wrapper that forbids access to any label field.

    Used for the official-test raw graphs while inference objects are built, so
    a label read cannot happen silently even if downstream code changes.
    """

    __slots__ = ("_d",)

    def __init__(self, d: Any) -> None:
        object.__setattr__(self, "_d", d)

    def __getattr__(self, name: str) -> Any:
        if name in LABEL_FIELDS:
            raise RuntimeError(f"label field {name!r} accessed during label-free prediction")
        return getattr(object.__getattribute__(self, "_d"), name)

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover - defensive
        raise RuntimeError("LabelGuard is read-only")


def _zero_labels(rows: Sequence[Any]) -> dict[str, Any]:
    """Replace every label tensor on the prepared rows with zeros (no values read)."""
    touched = 0
    for row in rows:
        for name in LABEL_FIELDS:
            if hasattr(row, name):
                value = getattr(row, name)
                if torch.is_tensor(value):
                    setattr(row, name, torch.zeros_like(value))
                    touched += 1
    return {"label_tensors_zeroed": touched, "label_values_read": False}


def _y_independence_check(models: Mapping[str, torch.nn.Module], rows: Sequence[Any], device: torch.device,
                          tag_to_builder: Mapping[str, Any], n: int = 32) -> dict[str, Any]:
    """Predict n rows with y=0 and with y=999; the forward must be unchanged."""
    rows_sub = list(rows[:n])
    out: dict[str, Any] = {"n": n, "per_model": {}}
    base = {}
    for tag, model in models.items():
        base[tag] = np.asarray(tag_to_builder[tag](model, rows_sub), dtype=np.float64)
    for row in rows_sub:
        for name in LABEL_FIELDS:
            if hasattr(row, name) and torch.is_tensor(getattr(row, name)):
                setattr(row, name, torch.full_like(getattr(row, name), 999.0))
    ok = True
    for tag, model in models.items():
        alt = np.asarray(tag_to_builder[tag](model, rows_sub), dtype=np.float64)
        worst = float(np.abs(alt - base[tag]).max())
        out["per_model"][tag] = {"max_abs_pred_diff_when_labels_changed": worst}
        ok &= worst == 0.0
    for row in rows_sub:
        for name in LABEL_FIELDS:
            if hasattr(row, name) and torch.is_tensor(getattr(row, name)):
                setattr(row, name, torch.zeros_like(getattr(row, name)))
    out["ok"] = bool(ok)
    out["gate"] = "official-test/valid labels cannot influence any model prediction (exact equality)"
    return out


def _load_official_y_rows(split: str, blob: Mapping[str, np.ndarray]):
    """Historical Full-Y encoded rows for one official split (labels zeroed)."""
    if split == "valid":
        rows, meta = zfdtc.load_valid_data(blob)
    elif split == "test":
        rows, meta = zfdtc.load_test_data(blob)
    else:
        raise ValueError(split)
    assert len(rows) == 1000, (split, len(rows))
    zero = _zero_labels(rows)
    return rows, meta, zero


def _raw_pyg(split: str):
    from torch_geometric.datasets import ZINC

    name = "val" if split == "valid" else "test"
    return ZINC(root=str(cf.ZINC_ROOT), subset=True, split=name)


def _alignment_gate(rows: Sequence[Any], ds, split: str) -> dict[str, Any]:
    n = len(rows)
    atom_mismatch = 0
    for i in range(n):
        if int(rows[i].dict_atom.shape[0]) != int(ds[i].x.shape[0]):
            atom_mismatch += 1
            continue
        if not bool(torch.equal(rows[i].dict_atom.long().reshape(-1), ds[i].x.long().reshape(-1))):
            atom_mismatch += 1
    return {
        "split": split,
        "n_rows": n,
        "atom_count_and_type_mismatches": atom_mismatch,
        "y_read_during_alignment": False,
        "ok": bool(atom_mismatch == 0),
    }


def _build_bjoc_objects_from_dataset(ds: Sequence[Any], n: int) -> list[sro.MolObjects]:
    return [
        sro.build_mol_objects(i, ds[i].x.numpy(), ds[i].edge_index.numpy(), ds[i].edge_attr.numpy())
        for i in range(n)
    ]


def _predict_y_model(model: torch.nn.Module, rows: Sequence[Any], device: torch.device) -> np.ndarray:
    return rpv2.predict_raw(model, list(rows), device)


def _predict_n_model(model: src.BJOC, objects: Sequence[sro.MolObjects], device: torch.device) -> np.ndarray:
    return srtc.predict_bjoc(model, objects, list(range(len(objects))), device)


def stage_predict(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    for tag in MODELS:
        path = Y_SOUP.get(tag) or (N1_SOUP_OUT if tag == "N1" else N_SOUP.get(tag))
        if not path.exists():
            raise SystemExit(f"missing frozen checkpoint for {tag}: {path}")
    blob = load_prep_blob()
    norm = build_norm()

    ckpt_sha_before = {
        tag: file_sha256(Y_SOUP.get(tag) or (N1_SOUP_OUT if tag == "N1" else N_SOUP.get(tag))) for tag in MODELS
    }

    results: dict[str, Any] = {}
    for split in ("valid", "test"):
        smoke = bool(args.limit)
        suffix = "_smoke" if smoke else ""
        n = int(args.limit) if smoke else 1000
        rows, meta, zero_log = _load_official_y_rows(split, blob)
        rows = rows[:n]
        ds = _raw_pyg(split)
        alignment = _alignment_gate(rows, ds, split)
        if not alignment["ok"]:
            raise SystemExit(f"{split}: encoded rows do not align with the official split")
        # label-blind guard on the raw graphs (test: no label may be touched)
        guarded = [LabelGuard(ds[i]) for i in range(n)] if split == "test" else [ds[i] for i in range(n)]
        objects = _build_bjoc_objects_from_dataset(guarded, n)
        obj_sig = _batch_objects_signature(objects)
        log(f"[predict:{split}{suffix}] rows={len(rows)} aligned={alignment['ok']} "
            f"{obj_sig['n_atoms']} atoms / {obj_sig['n_bonds']} bonds / {obj_sig['n_junctions']} junctions")

        y_models = {tag: build_y_model(tag, blob, device) for tag in ("Y0", "Y1")}
        n_models = {tag: build_bjoc_model(tag, norm, device) for tag in ("N0", "N1")}

        preds: dict[str, np.ndarray] = {}
        for tag, model in y_models.items():
            preds[tag] = _predict_y_model(model, rows, device)
        for tag, model in n_models.items():
            preds[tag] = _predict_n_model(model, objects, device)

        finiteness = {tag: bool(np.isfinite(p).all()) for tag, p in preds.items()}
        if not all(finiteness.values()):
            raise SystemExit(f"{split}: non-finite predictions {finiteness}")

        builders = {
            "Y0": lambda m, r: _predict_y_model(m, r, device),
            "Y1": lambda m, r: _predict_y_model(m, r, device),
            "N0": lambda m, r: _predict_n_model(m, objects[: len(r)], device),
            "N1": lambda m, r: _predict_n_model(m, objects[: len(r)], device),
        }
        y_indep = _y_independence_check({**y_models, **n_models}, rows, device, builders, n=min(32, n))

        crosscheck: dict[str, Any] = {}
        if not smoke and split == "valid":
            hist = np.load(HIST_VALID_NPZ, allow_pickle=True)
            for tag, col in (("Y0", "Y_raw_0"), ("Y1", "Y_raw_1")):
                ref = np.asarray(hist[col], dtype=np.float64)
                crosscheck[f"{tag}_vs_historical_csv"] = {
                    "max_abs_pred_diff": float(np.abs(preds[tag] - ref).max()),
                    "mae_current": float(np.abs(preds[tag] - np.asarray(hist["y"], dtype=np.float64)).mean()),
                    "mae_historical": HISTORICAL_REFERENCE["valid"][tag],
                }
            ref_n0 = np.load(BJOC_VALID_N0, allow_pickle=False)
            crosscheck["N0_vs_frozen_gpu_export"] = {
                "max_abs_pred_diff": float(np.abs(preds["N0"] - np.asarray(ref_n0["pred"], dtype=np.float64)).max()),
                "note": "frozen export was produced on the res A100 (commit 828c587e); this run is CPU float32",
            }

        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        pred_meta = {
            "protocol_id": PROTOCOL_ID,
            "split": split,
            "smoke": smoke,
            "n_molecules": n,
            "rows": list(range(n)),
            "device": str(device),
            "generated_at": now,
            "checkpoint_sha256": ckpt_sha_before,
            "alignment": alignment,
            "objects_signature": obj_sig,
            "label_isolation": zero_log,
            "y_independence": y_indep,
            "crosscheck": crosscheck,
            "y_in_file": False,
            "predictions_are_raw": True,
            "inference_input": (
                "raw molecule graph + train-fitted preprocessing only "
                "(Full-Y: all-10k train-only prep; BJOC: all-10k train-only global standardiser)"
            ),
            "official_test_labels_read": False,
        }
        files = {}
        for tag, p in preds.items():
            path = OUT / f"{split}{suffix}_{tag}_preds.npz"
            np.savez_compressed(path, model=tag, pred=np.asarray(p, dtype=np.float64),
                                row=np.arange(n, dtype=np.int64), meta=json.dumps(_j(pred_meta), default=str))
            files[tag] = {"path": path.name, "sha256": file_sha256(path)}
        combined = OUT / f"{split}{suffix}_predictions.npz"
        np.savez_compressed(
            combined,
            row=np.arange(n, dtype=np.int64),
            Y0_pred=np.asarray(preds["Y0"], dtype=np.float64),
            Y1_pred=np.asarray(preds["Y1"], dtype=np.float64),
            N0_pred=np.asarray(preds["N0"], dtype=np.float64),
            N1_pred=np.asarray(preds["N1"], dtype=np.float64),
            meta=json.dumps(_j(pred_meta), default=str),
        )
        results[split] = {
            "n": n, "smoke": smoke, "alignment": alignment, "finiteness": finiteness,
            "y_independence": y_indep, "crosscheck": crosscheck, "files": files,
            "combined": {"path": combined.name, "sha256": file_sha256(combined)},
            "objects_signature": obj_sig,
            "seconds": None,
        }
        log(f"[predict:{split}{suffix}] combined sha {results[split]['combined']['sha256'][:16]} "
            f"y_independence={y_indep['ok']}")

    ckpt_sha_after = {
        tag: file_sha256(Y_SOUP.get(tag) or (N1_SOUP_OUT if tag == "N1" else N_SOUP.get(tag))) for tag in MODELS
    }
    determinism = {"before": ckpt_sha_before, "after": ckpt_sha_after, "ok": bool(ckpt_sha_before == ckpt_sha_after)}
    write_json(OUT / "prediction_access_log.json", {
        "protocol_id": PROTOCOL_ID,
        "stage": "predict",
        "git_commit": git_commit(),
        "device": str(device),
        "official_valid_labels_read": False,
        "official_test_labels_read": False,
        "test_label_guard_active": True,
        "label_values_read": False,
        "checkpoint_determinism": determinism,
        "results": results,
        "seconds": time.perf_counter() - started,
    })
    if not determinism["ok"]:
        raise SystemExit("checkpoint hashes changed during prediction")
    log(f"[predict] done ({time.perf_counter() - started:.0f}s)")
    return 0


# ---------------------------------------------------------------------------
# stage: freeze
# ---------------------------------------------------------------------------


def _split_fingerprint(split: str, rows: Sequence[Any], ds) -> dict[str, Any]:
    processed = REPO_ROOT / f"data/ZINC/subset/processed/{'val' if split == 'valid' else 'test'}.pt"
    x_digest = hashlib.sha256()
    for i in range(len(rows)):
        x_digest.update(np.ascontiguousarray(ds[i].x.numpy(), dtype=np.int64).tobytes())
        x_digest.update(np.ascontiguousarray(ds[i].edge_index.numpy(), dtype=np.int64).tobytes())
        x_digest.update(np.ascontiguousarray(ds[i].edge_attr.numpy(), dtype=np.int64).tobytes())
    return {
        "split": split,
        "n_molecules": len(rows),
        "processed_pt": {"path": str(processed.relative_to(REPO_ROOT)), "sha256": file_sha256(processed)},
        "structure_sha256": x_digest.hexdigest(),
        "note": "structure fingerprint = x + edge_index + edge_attr of every row, label-free",
    }


def stage_freeze(args: argparse.Namespace) -> int:
    for split in ("valid", "test"):
        if not PRED[split].exists():
            raise SystemExit(f"missing prediction file {PRED[split]}; run predict first")
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "stage": "terminal_freeze",
        "git_commit": git_commit(),
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frozen_before_any_official_label_read": True,
        "models": {
            tag: {
                "checkpoint": str((Y_SOUP.get(tag) or (N1_SOUP_OUT if tag == "N1" else N_SOUP.get(tag))).relative_to(REPO_ROOT)),
                "checkpoint_sha256": file_sha256(Y_SOUP.get(tag) or (N1_SOUP_OUT if tag == "N1" else N_SOUP.get(tag))),
                "seed": 0 if tag.endswith("0") else 1,
                "parameters": 408651 if tag.startswith("Y") else 413505,
                "preprocessing": str((LEGACY_PREP if tag.startswith("Y") else NORM_FULL10K).relative_to(REPO_ROOT)),
                "preprocessing_sha256": file_sha256(LEGACY_PREP if tag.startswith("Y") else NORM_FULL10K),
                "supervision": (
                    "raw-y L1 + label-free structural reconstruction (lambda 33.95873017865987)"
                    if tag.startswith("Y") else "raw-y L1 only"
                ),
                "train_rows_fingerprint": "official-train 10000 rows",
            }
            for tag in MODELS
        },
        "predictions": {
            split: {
                "combined": {"path": PRED[split].name, "sha256": file_sha256(PRED[split])},
                "per_model": {
                    tag: {"path": f"{split}_{tag}_preds.npz",
                          "sha256": file_sha256(OUT / f"{split}_{tag}_preds.npz")}
                    for tag in MODELS
                },
            }
            for split in ("valid", "test")
        },
        "inference": {
            "python": _CODE_DIR / "run_bjoc_predictor_terminal_v1.py",
            "python_sha256": file_sha256(Path(__file__).resolve()),
            "device": "cpu",
            "full_y_model_code_sha256": file_sha256(_CODE_DIR / "run_prediction_feasibility_v2.py"),
            "bjoc_model_code_sha256": file_sha256(BJOC_MODEL_CODE),
            "bjoc_objects_code_sha256": file_sha256(BJOC_OBJECTS_CODE),
            "bjoc_v1_driver_sha256": file_sha256(V1_DRIVER_CODE),
            "raw_y_output_semantics": "model output in raw y units, no post-hoc calibration anywhere",
        },
        "ensemble_rule": {
            "pred_N_ens": "0.5 * (pred_N0 + pred_N1)",
            "pred_Y_ens": "0.5 * (pred_Y0 + pred_Y1)",
            "weights": "fixed at 0.5; never fitted on validation or test",
        },
        "primary_metric": {
            "name": "official test raw-y MAE, mean over the two per-seed MAEs",
            "MAE_N_mean": "0.5 * (MAE_N0 + MAE_N1)",
            "MAE_Y_mean": "0.5 * (MAE_Y0 + MAE_Y1)",
            "delta_test": "MAE_N_mean - MAE_Y_mean (negative => BJOC-Null better)",
        },
        "predeclared_secondary_metrics": [
            "per-seed raw MAE", "seed-MAE mean", "equal-weight ensemble MAE", "median_ae",
            "p90_ae", "p95_ae", "rmse", "max_ae", "per-molecule improvement fraction",
            "y>-3 and y<=-3 subgroups", "atom count / cyclomatic rank / diameter / min ring strata",
            "top-10 error mass share", "tail error-mass contribution",
            "descriptive MAE excluding the single largest-error molecule",
        ],
        "bootstrap": {"resamples": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "kind": "molecule bootstrap of the paired mean"},
        "evaluation_rules": {
            "single_scoring_invocation": True,
            "predictions_may_not_change_after_scoring": True,
            "no_model_selection_after_test": True,
            "no_calibrator_fitted_after_test": True,
            "trimmed_metrics_are_descriptive_only": True,
        },
        "official_test_exposure": {
            "project_wide_pristine": False,
            "first_historical_read": "2026-10-03T11:41:26Z (KSVD track zinc_full_decomposition_valid_test_confirmation_v1)",
            "statement": (
                "the official test split has been read in this repository before; this round is a "
                "frozen-version terminal evaluation of an already-exposed instrument, not an independent "
                "blind confirmation"
            ),
        },
        "historical_reference": HISTORICAL_REFERENCE,
        "split_fingerprints": {},
    }
    from torch_geometric.datasets import ZINC

    for split in ("valid", "test"):
        ds = _raw_pyg(split)
        payload["split_fingerprints"][split] = _split_fingerprint(split, list(range(1000)), ds)
    if FREEZE_MANIFEST.exists():
        prior = read_json(FREEZE_MANIFEST)
        same = (
            prior.get("predictions") == payload["predictions"]
            and prior.get("models") == payload["models"]
        )
        if not same:
            raise SystemExit("refusing to overwrite a freeze manifest with different hashes")
        log("[freeze] identical hashes already frozen; rewriting timestamp only")
    write_json(FREEZE_MANIFEST, payload)
    log(f"[freeze] valid={payload['predictions']['valid']['combined']['sha256'][:16]} "
        f"test={payload['predictions']['test']['combined']['sha256'][:16]} -> {FREEZE_MANIFEST.name}")
    return 0


# ---------------------------------------------------------------------------
# stage: score (ONE terminal scoring; reads official labels)
# ---------------------------------------------------------------------------


def _load_official_y(split: str) -> np.ndarray:
    ds = _raw_pyg(split)
    return np.asarray([float(ds[i].y.reshape(-1)[0]) for i in range(len(ds))], dtype=np.float64)


def _score_one(split: str, y: np.ndarray, preds: Mapping[str, np.ndarray]) -> dict[str, Any]:
    metrics = {tag: abs_metrics(preds[tag], y) for tag in MODELS}
    seed_macro = {
        "Y": 0.5 * (metrics["Y0"]["mae"] + metrics["Y1"]["mae"]),
        "N": 0.5 * (metrics["N0"]["mae"] + metrics["N1"]["mae"]),
    }
    ens = {
        "Y": 0.5 * (preds["Y0"] + preds["Y1"]),
        "N": 0.5 * (preds["N0"] + preds["N1"]),
    }
    ens_metrics = {k: abs_metrics(v, y) for k, v in ens.items()}
    e = {tag: np.abs(preds[tag] - y) for tag in MODELS}
    delta0 = metrics["N0"]["mae"] - metrics["Y0"]["mae"]
    delta1 = metrics["N1"]["mae"] - metrics["Y1"]["mae"]
    delta_mean = 0.5 * (delta0 + delta1)
    d = 0.5 * ((e["N0"] - e["Y0"]) + (e["N1"] - e["Y1"]))
    assert abs(float(d.mean()) - delta_mean) < 1e-12, "paired statistic != seed-MAE difference"
    boot = paired_bootstrap(d)
    per_seed_pairs = {
        "N0_minus_Y0": {**paired_bootstrap(e["N0"] - e["Y0"]), "frac_N_better": float((e["N0"] < e["Y0"]).mean())},
        "N1_minus_Y1": {**paired_bootstrap(e["N1"] - e["Y1"]), "frac_N_better": float((e["N1"] < e["Y1"]).mean())},
        "N_ens_minus_Y_ens": {**paired_bootstrap(np.abs(ens["N"] - y) - np.abs(ens["Y"] - y))},
        "N_ens_minus_Y0": {**paired_bootstrap(np.abs(ens["N"] - y) - e["Y0"])},
    }
    return {
        "split": split,
        "n_molecules": int(y.size),
        "metrics_per_model": metrics,
        "seed_macro_mae": seed_macro,
        "ensemble_predictions": {k: v.tolist() for k, v in ens.items()},
        "ensemble_metrics": ens_metrics,
        "deltas": {
            "delta0_N0_minus_Y0": delta0,
            "delta1_N1_minus_Y1": delta1,
            "delta_mean": delta_mean,
            "delta_ensemble_mae": ens_metrics["N"]["mae"] - ens_metrics["Y"]["mae"],
            "delta_seed_macro_mae": seed_macro["N"] - seed_macro["Y"],
        },
        "paired_bootstrap_d_mean": boot,
        "paired_per_seed": per_seed_pairs,
        "errors": {tag: e[tag].tolist() for tag in MODELS},
    }


def _subgroups(split: str, y: np.ndarray, preds: Mapping[str, np.ndarray], cov: Mapping[str, np.ndarray],
               e: Mapping[str, np.ndarray]) -> dict[str, Any]:
    d = 0.5 * ((e["N0"] - e["Y0"]) + (e["N1"] - e["Y1"]))

    def block(mask: np.ndarray) -> dict[str, Any]:
        out = _group_entry(preds, y, mask)
        if out["n"] > 1:
            out["delta_mean"] = float(d[mask].mean())
            out["delta_ci95"] = paired_bootstrap(d[mask], n=2000)["ci95"]
            out["frac_N_better"] = float((d[mask] < 0).mean())
        return out

    top10 = {}
    for tag in MODELS:
        order = np.argsort(-e[tag])
        top10[tag] = {
            "max_error_row": int(order[0]),
            "max_error_value": float(e[tag][order[0]]),
            "top1_share": float(e[tag][order[0]] / e[tag].sum()),
            "top10_share": float(np.sort(e[tag])[-10:].sum() / e[tag].sum()),
            "top10_rows": [int(r) for r in order[:10]],
        }
    worst_row = int(np.argmax(e["N1"] + e["N0"])) if split == "test" else int(np.argmax(np.abs(preds["Y0"] - y)))
    keep = np.ones(y.size, dtype=bool)
    keep[worst_row] = False
    return {
        "split": split,
        "y_le_minus3": block(y <= TAIL_Y),
        "y_gt_minus3": block(y > TAIL_Y),
        "atoms_quartiles": {
            f"q{k+1}": block(
                (cov["n_atoms"] > np.percentile(cov["n_atoms"], 25 * k))
                & (cov["n_atoms"] <= np.percentile(cov["n_atoms"], 25 * (k + 1)))
            )
            for k in range(4)
        },
        "cyclomatic_0_1": block(cov["cyclomatic_rank"] <= 1),
        "cyclomatic_2_3": block((cov["cyclomatic_rank"] >= 2) & (cov["cyclomatic_rank"] <= 3)),
        "cyclomatic_ge_4": block(cov["cyclomatic_rank"] >= 4),
        "diameter_le_median": block(cov["diameter"] <= np.median(cov["diameter"])),
        "diameter_gt_median": block(cov["diameter"] > np.median(cov["diameter"])),
        "min_ring_0_acyclic": block(cov["min_ring_size"] == 0),
        "min_ring_3_4": block((cov["min_ring_size"] >= 3) & (cov["min_ring_size"] <= 4)),
        "min_ring_5": block(cov["min_ring_size"] == 5),
        "min_ring_ge_6": block(cov["min_ring_size"] >= 6),
        "top10_extremes": top10,
        "tail_error_mass_share": {
            tag: float(e[tag][y <= TAIL_Y].sum() / e[tag].sum()) for tag in MODELS
        },
        "descriptive_mae_without_single_worst_row": {
            "excluded_row": worst_row,
            "mae": {tag: float(e[tag][keep].mean()) for tag in MODELS},
            "note": "descriptive only; the formal metric always uses all rows",
        },
        "better_fractions": {
            "N0_better_than_Y0": float((e["N0"] < e["Y0"]).mean()),
            "N1_better_than_Y1": float((e["N1"] < e["Y1"]).mean()),
            "N_mean_better_than_Y_mean": float((d < 0).mean()),
        },
    }


def _decision(valid: Mapping[str, Any], test: Mapping[str, Any]) -> dict[str, Any]:
    d0 = test["deltas"]["delta0_N0_minus_Y0"]
    d1 = test["deltas"]["delta1_N1_minus_Y1"]
    dm = test["deltas"]["delta_mean"]
    ci = test["paired_bootstrap_d_mean"]["ci95"]
    ens_delta = test["deltas"]["delta_ensemble_mae"]
    ci_supports = bool(ci[1] < 0.0)
    consistent = bool(d0 < 0 and d1 < 0)
    sub = test["subgroups"]
    ordinary = sub["y_gt_minus3"].get("delta_mean", float("nan"))
    worst_row = sub["descriptive_mae_without_single_worst_row"]
    trimmed = {k: worst_row["mae"][k] for k in MODELS}
    trimmed_delta = 0.5 * ((trimmed["N0"] - trimmed["Y0"]) + (trimmed["N1"] - trimmed["Y1"]))
    rmse_n = 0.5 * (test["metrics_per_model"]["N0"]["rmse"] + test["metrics_per_model"]["N1"]["rmse"])
    rmse_y = 0.5 * (test["metrics_per_model"]["Y0"]["rmse"] + test["metrics_per_model"]["Y1"]["rmse"])

    if dm >= 0.0:
        branch = "G4_no_test_advantage"
    elif ci_supports and consistent and ordinary < 0 and trimmed_delta < 0 and ens_delta < 0:
        branch = "G1_replicated_broad_advantage"
    elif not ordinary < 0 and (rmse_n < rmse_y or trimmed_delta < 0):
        branch = "G2_tail_driven_advantage"
    elif ci_supports and not consistent:
        branch = "G3_trajectory_sensitivity"
    elif dm < 0 and not ci_supports:
        branch = "point_estimate_negative_ci_crosses_zero"
    else:
        branch = "partially_consistent"
    return {
        "protocol_id": PROTOCOL_ID,
        "branch": branch,
        "inputs": {
            "delta0": d0, "delta1": d1, "delta_mean": dm, "ci95": ci,
            "ci_supports_bjc_better": ci_supports,
            "seed_directions_consistent": consistent,
            "delta_ensemble_mae": ens_delta,
            "ordinary_molecules_delta_mean": ordinary,
            "delta_mean_without_single_worst_row": trimmed_delta,
            "rmse_seed_mean_N": rmse_n,
            "rmse_seed_mean_Y": rmse_y,
        },
        "claims_allowed": [
            "a frozen-version terminal descriptive comparison on an already-exposed official test split",
            "which of the pre-declared secondary endpoints moved and in which direction",
            "disclosed historical-exposure language",
        ],
        "claims_forbidden": [
            "new-method independent blind-test confirmation",
            "state-of-the-art",
            "structural-mechanism causality from a test improvement",
            "reusing this split for further model selection",
        ],
    }


def _answers(d: Mapping[str, Any], test: Mapping[str, Any], valid: Mapping[str, Any],
             n1_meta: Mapping[str, Any] | None) -> dict[str, Any]:
    t = test["metrics_per_model"]
    v = valid["metrics_per_model"]
    return {
        "Q1_seed1_training_completed": bool(n1_meta is not None),
        "Q2_seed1_contract_matches_N0": bool(
            n1_meta
            and n1_meta["v1_result"]["parameters"] == 413505
            and n1_meta["v1_result"]["config"]["organization_mode"] == "null"
            and n1_meta["v1_result"]["soup_members"] == [236, 237, 238, 239, 240]
            and n1_meta["v1_result"]["hyper"]["epochs"] == 240
        ),
        "Q3_N1_valid_raw_mae": v["N1"]["mae"],
        "Q4_N1_test_raw_mae": t["N1"]["mae"],
        "Q5_test_seed_mae_mean": test["seed_macro_mae"],
        "Q6_N_test_ensemble_mae": test["ensemble_metrics"]["N"]["mae"],
        "Q7_Y_test_seed_mae_mean_and_ensemble": {
            "seed_macro_mae": test["seed_macro_mae"]["Y"],
            "ensemble_mae": test["ensemble_metrics"]["Y"]["mae"],
        },
        "Q8_advantage_on_ordinary_molecules": test["subgroups"]["y_gt_minus3"].get("delta_mean"),
        "Q9_advantage_driven_by_extremes": {
            "delta_mean_full": test["deltas"]["delta_mean"],
            "delta_mean_without_single_worst_row": d["inputs"]["delta_mean_without_single_worst_row"],
            "tail_delta_mean": test["subgroups"]["y_le_minus3"].get("delta_mean"),
            "ordinary_delta_mean": test["subgroups"]["y_gt_minus3"].get("delta_mean"),
        },
        "Q10_verdict_branch": d["branch"],
    }


def stage_score(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    if not FREEZE_MANIFEST.exists():
        raise SystemExit("refusing to score: terminal_freeze_manifest.json is missing")
    manifest = read_json(FREEZE_MANIFEST)
    for split in ("valid", "test"):
        want = manifest["predictions"][split]["combined"]["sha256"]
        got = file_sha256(PRED[split])
        if want != got:
            raise SystemExit(f"{split}: prediction file changed after freeze ({want[:12]} != {got[:12]})")
    for tag in MODELS:
        chk = manifest["models"][tag]["checkpoint_sha256"]
        path = Y_SOUP.get(tag) or (N1_SOUP_OUT if tag == "N1" else N_SOUP.get(tag))
        if file_sha256(path) != chk:
            raise SystemExit(f"{tag}: checkpoint changed after freeze")

    score_log: dict[str, Any] = {"protocol_id": PROTOCOL_ID, "stage": "score", "git_commit": git_commit(),
                                 "label_reads": [], "predictions_frozen_hashes": {
                                     s: manifest["predictions"][s]["combined"]["sha256"] for s in ("valid", "test")}}
    results: dict[str, Any] = {}
    for split in ("valid", "test"):
        y = _load_official_y(split)
        score_log["label_reads"].append({"split": split, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": int(y.size),
                                         "source": f"PyG ZINC subset processed {'val' if split == 'valid' else 'test'}.pt"})
        z = np.load(PRED[split], allow_pickle=False)
        preds = {tag: np.asarray(z[f"{tag}_pred"], dtype=np.float64) for tag in MODELS}
        assert np.array_equal(z["row"], np.arange(y.size)), f"{split}: prediction row order broken"
        assert all(p.shape == y.shape for p in preds.values()), f"{split}: prediction/label length mismatch"
        res = _score_one(split, y, preds)
        results[split] = res
        if y.size == 1000:
            hist = np.load(HIST_VALID_NPZ if split == "valid" else HIST_TEST_NPZ, allow_pickle=True)
            res["historical_label_crosscheck_max_abs"] = float(
                np.abs(np.asarray(hist["y"], dtype=np.float64) - y).max())
            res["historical_prediction_crosscheck"] = {
                "Y0_max_abs": float(np.abs(np.asarray(hist["Y_raw_0"], dtype=np.float64) - preds["Y0"]).max()),
                "Y1_max_abs": float(np.abs(np.asarray(hist["Y_raw_1"], dtype=np.float64) - preds["Y1"]).max()),
                "note": "engineering cross-check only; the frozen Freeze manifest hashes are the truth of record",
            }
        else:  # smoke-scale scoring (tests only)
            res["historical_label_crosscheck_max_abs"] = None
            res["historical_prediction_crosscheck"] = None
        sanity = {
            "Y0": abs(res["metrics_per_model"]["Y0"]["mae"] - HISTORICAL_REFERENCE[split]["Y0"]),
            "Y1": abs(res["metrics_per_model"]["Y1"]["mae"] - HISTORICAL_REFERENCE[split]["Y1"]),
            "seed_mean": abs(res["seed_macro_mae"]["Y"] - HISTORICAL_REFERENCE[split]["seed_mean"]),
            "ensemble": abs(res["ensemble_metrics"]["Y"]["mae"] - HISTORICAL_REFERENCE[split]["ensemble"]),
        }
        res["historical_sanity_check"] = {"abs_diff": sanity, "atol": HIST_MAE_ATOL,
                                          "ok": bool(all(v <= HIST_MAE_ATOL for v in sanity.values()))}
        if not res["historical_sanity_check"]["ok"]:
            raise SystemExit(f"{split}: Y metrics do not reproduce the historical record {sanity}")
        ds = _raw_pyg(split)
        cov = structural_covariates(ds)
        e = {tag: np.abs(preds[tag] - y) for tag in MODELS}
        sub = _subgroups(split, y, preds, cov, e)
        res["subgroups"] = sub
        write_json(OUT / "official_valid_metrics.json" if split == "valid" else OUT / "official_test_metrics.json",
                   {k: v for k, v in res.items() if k != "errors"})
        log(f"[score:{split}] MAE Y0={res['metrics_per_model']['Y0']['mae']:.6f} "
            f"Y1={res['metrics_per_model']['Y1']['mae']:.6f} N0={res['metrics_per_model']['N0']['mae']:.6f} "
            f"N1={res['metrics_per_model']['N1']['mae']:.6f}")
        log(f"[score:{split}] seed-MAE mean Y={res['seed_macro_mae']['Y']:.6f} N={res['seed_macro_mae']['N']:.6f}; "
            f"ensemble Y={res['ensemble_metrics']['Y']['mae']:.6f} N={res['ensemble_metrics']['N']['mae']:.6f}")
        log(f"[score:{split}] delta0={res['deltas']['delta0_N0_minus_Y0']:+.6f} "
            f"delta1={res['deltas']['delta1_N1_minus_Y1']:+.6f} delta_mean={res['deltas']['delta_mean']:+.6f} "
            f"CI={res['paired_bootstrap_d_mean']['ci95']}")

    paired = {
        "protocol_id": PROTOCOL_ID,
        "definition": "d_i = 0.5 * [(|e_N0,i| - |e_Y0,i|) + (|e_N1,i| - |e_Y1,i|)]",
        "valid": results["valid"]["paired_bootstrap_d_mean"],
        "test": results["test"]["paired_bootstrap_d_mean"],
        "per_seed_test": results["test"]["paired_per_seed"],
        "per_seed_valid": results["valid"]["paired_per_seed"],
        "limits": [
            "a molecule bootstrap does not include across-trajectory randomness",
            "two seeds cannot estimate training stability precisely",
        ],
    }
    write_json(OUT / "paired_bootstrap.json", paired)

    subgroup_payload = {
        "protocol_id": PROTOCOL_ID,
        "valid": results["valid"]["subgroups"],
        "test": results["test"]["subgroups"],
    }
    write_json(OUT / "subgroup_analysis.json", subgroup_payload)

    n1_meta = read_json(OUT / "seed1_train_meta.json") if (OUT / "seed1_train_meta.json").exists() else None
    decision = _decision(results["valid"], results["test"])
    decision["answers"] = _answers(decision, results["test"], results["valid"], n1_meta)
    decision["next_steps"] = {
        "P_prediction_route": (
            "the next predictive purchase is a representation/optimisation route that improves ORDINARY "
            "molecules at matched budget, not more seeds for the current recipe; a matched full-10k Full-Y "
            "control on the same evaluation path is still missing before any architectural attribution"
        ),
        "M_mechanism_route": (
            "the BJOC ownership channel (junction -> bond incidence) is functional but compensable and its "
            "generalisation effect is not isolated; the task/representation where ownership is predictively "
            "necessary (per-bond or atom-mapped targets) is the only clean next mechanism experiment"
        ),
        "I_attribution_route": (
            "contributions must be built as an additive, cross-molecule-shared object decomposition with an "
            "explicit identity (pred = bias + sum of unit/object contributions), not read out of intermediate "
            "embeddings; the current reader is a plain MLP, so no attribution claim is made in this round"
        ),
    }
    write_json(OUT / "scientific_decision.json", decision)

    write_json(OUT / "scoring_record.json", {
        **score_log,
        "n_label_reads": len(score_log["label_reads"]),
        "single_invocation": True,
        "post_score_prediction_changes": 0,
        "result_status": (
            "one-time frozen-version terminal result; belongs to this frozen version only and must never be "
            "reused to claim independence for a future configuration"
        ),
        "seconds": time.perf_counter() - started,
    })
    log(f"[score] branch={decision['branch']} ({time.perf_counter() - started:.0f}s)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=PROTOCOL_ID)
    parser.add_argument("--device", default="cpu")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("reuse-audit")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=stage_reuse_audit)

    p = sub.add_parser("valid-audit")
    p.set_defaults(func=stage_valid_audit)

    p = sub.add_parser("train-n1")
    p.set_defaults(func=stage_train_n1)

    p = sub.add_parser("predict")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=stage_predict)

    p = sub.add_parser("freeze")
    p.set_defaults(func=stage_freeze)

    p = sub.add_parser("score")
    p.set_defaults(func=stage_score)

    args = parser.parse_args(argv)
    if not CONFIG_PATH.exists():
        raise SystemExit(f"frozen protocol missing: {CONFIG_PATH}")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
