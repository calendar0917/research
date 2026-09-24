"""E2E-DictEnv-P2-ABS runner — ZINC clean dictionary-core absolute-performance search.

Round ``e2e_dictenv_p2_abs`` (Workstream Z).  Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_p2_abs_preregistration.md``.
Core module: ``tracks/ksvd/experiments/luyin16/e2e_dictenv_p2_abs.py``.

Official ZINC test is **never** loaded in this round.

Stages
------
``smoke bench train select-a select-b select-c select-d finalize all``

The environment cache is the already-validated P1 primitive cache
(``results/e2e_dictenv_p1/cache``), read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_v0 as v0run
from tracks.ksvd.experiments.luyin16 import zinc_sdb_v0 as zsdb
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = p2.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_p2_abs"
STATE_DIR = RESULTS_DIR / "states"
CURVE_DIR = RESULTS_DIR / "curves"

BATCH_SIZE = int(p1run.BATCH_SIZE)
LEARNING_RATE = float(p1run.LEARNING_RATE)
WEIGHT_DECAY = float(p1run.WEIGHT_DECAY)
GRAD_CLIP = float(p1run.GRAD_CLIP)
TRAIN_SHUFFLE_OFFSET = int(p1run.TRAIN_SHUFFLE_OFFSET)
EVAL_SHUFFLE_OFFSET = int(p1run.EVAL_SHUFFLE_OFFSET)
SOUP_K = 5
CALIBRATION_MOLECULES = 512
DICT_EPOCHS = 10

P1_SOUP = 0.13197501279687276
T1_SOUP = 0.125765
FEC_S1_SOUP = 0.13042183499777457

# Pre-registered durable artifact names (preregistration section 15).
RUN_ARTIFACT = {
    "Z1": "stage_a_h320.json",
    "Z2": "stage_a_lambda0125.json",
    "Z3": "stage_a_lambda00625.json",
    "H1": "stage_b_h1.json",
    "H2": "stage_b_h2.json",
    "E64": "stage_c_edge64.json",
    "K64S8": "stage_d_k64s8.json",
    "K64S12": "stage_d_k64s12.json",
}

_write_json = v0run._write_json
_read_json = v0run._read_json
_write_csv = v0run._write_csv
_git_commit = v0run._git_commit
_n_params = v0run._n_params
_seed_everything = v0run._seed_everything


# ---------------------------------------------------------------------------
# candidate registry
# ---------------------------------------------------------------------------


def _base_candidates() -> dict[str, p2.P2Config]:
    return {
        "Z1": p2.P2Config("Z1", "p1", 48, 32, 8, 0.25, 320, "sdb32"),
        "Z2": p2.P2Config("Z2", "p1", 48, 32, 8, 0.125, 240, "sdb32"),
        "Z3": p2.P2Config("Z3", "p1", 48, 32, 8, 0.0625, 240, "sdb32"),
    }


def _selections_path() -> Path:
    return RESULTS_DIR / "selections.json"


def _read_selections() -> dict[str, Any]:
    path = _selections_path()
    return _read_json(path) if path.exists() else {}


def _stage_selection(name: str) -> dict[str, Any]:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        raise RuntimeError(f"selection {name} not available; run the previous stage first")
    return _read_json(path)


def resolve_candidate(name: str) -> p2.P2Config:
    """Resolve a candidate tag to a fully-specified frozen config."""
    name = str(name).upper()
    bases = _base_candidates()
    if name in bases:
        return bases[name]
    selections = _read_selections()
    if name in ("H1", "H2"):
        parent = _stage_selection("stage_a_selection")["winner"]
        if parent == "Z0":
            parent = "Z2"
        cfg = bases[parent]
        return p2.P2Config(name, name.lower(), cfg.d_e, cfg.K, cfg.s, cfg.lambda_factor, cfg.horizon, cfg.dict_kind)
    if name == "E64":
        parent = _stage_selection("stage_b_selection")["winner"]
        cfg = _config_by_tag(parent)
        return p2.P2Config("E64", cfg.decoder, 64, cfg.K, cfg.s, cfg.lambda_factor, cfg.horizon, cfg.dict_kind)
    if name in ("K64S8", "K64S12"):
        parent = _stage_selection("stage_c_selection")["winner"]
        cfg = _config_by_tag(parent)
        s = 8 if name == "K64S8" else 12
        kind = "k64s8" if s == 8 else "k64s12"
        return p2.P2Config(name, cfg.decoder, cfg.d_e, 64, s, cfg.lambda_factor, cfg.horizon, kind)
    raise ValueError(f"unknown candidate {name!r} (selections={sorted(selections)})")


def _config_by_tag(tag: str) -> p2.P2Config:
    """Recover a config from a previously trained run (or the base registry)."""
    tag = str(tag).upper()
    bases = _base_candidates()
    if tag in bases:
        return bases[tag]
    for name in ("H1", "H2", "E64", "K64S8", "K64S12"):
        try:
            cfg = resolve_candidate(name)
        except Exception:
            continue
        if cfg.tag.upper() == tag:
            return cfg
    return resolve_candidate(tag)


# ---------------------------------------------------------------------------
# dictionary artifacts
# ---------------------------------------------------------------------------


def dictionary_path(kind: str) -> Path:
    if kind == "sdb32":
        return p1run.DICT_PATH
    return RESULTS_DIR / f"dictionary_{kind}.pt"


def fit_dictionary_k64(s: int, force: bool = False) -> dict[str, Any]:
    kind = f"k64s{s}"
    path = dictionary_path(kind)
    meta_path = RESULTS_DIR / f"dictionary_{kind}.json"
    if path.exists() and meta_path.exists() and not force:
        return _read_json(meta_path)
    blob = torch.load(p1run._env_cache_path("train"), map_location="cpu", weights_only=False)
    X = blob["phi"].numpy().astype(np.float64)
    started = time.perf_counter()
    D, info = sdb.fit_ksvd(X, atoms=64, s=int(s), epochs=DICT_EPOCHS, seed=sdb.DICT_SEED)
    D = np.asarray(D, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"D": torch.as_tensor(D, dtype=torch.float32)}, path)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "K": 64,
        "s": int(s),
        "iht_steps": int(p2.IHT_STEPS),
        "epochs": int(DICT_EPOCHS),
        "seed": int(sdb.DICT_SEED),
        "solver": "sdb_v0.fit_ksvd -> tccd_v0.ksvd_fit",
        "source_split": "official train only (ZINC phi65, 231,664 nodes)",
        "n_nodes": int(X.shape[0]),
        "D_shape": list(D.shape),
        "D_sha256": hashlib.sha256(D.tobytes()).hexdigest(),
        "final_fit_mse": float(info["history"][-1]["mean_sq_err"]) if info.get("history") else float("nan"),
        "seconds": float(time.perf_counter() - started),
        "official_test_loaded": False,
    }
    _write_json(meta_path, payload)
    print(f"[dict:{kind}] {payload}", flush=True)
    return payload


def load_dictionary(kind: str) -> tuple[np.ndarray, str]:
    if kind == "sdb32":
        D, _rand, _pca = zsdb.load_dictionary()
        D = np.asarray(D, dtype=np.float32)
        return D, hashlib.sha256(D.tobytes()).hexdigest()
    path = dictionary_path(kind)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    D = np.asarray(blob["D"], dtype=np.float32)
    return D, hashlib.sha256(D.tobytes()).hexdigest()


def dictionary_sha256(kind: str) -> str:
    if kind == "sdb32":
        return p1run.SDB_DICT_SHA256
    meta = _read_json(RESULTS_DIR / f"dictionary_{kind}.json")
    return str(meta["D_sha256"])


# ---------------------------------------------------------------------------
# lambda calibration
# ---------------------------------------------------------------------------


def calibrate_lambda_base(config: p2.P2Config, device: str = "cpu") -> dict[str, Any]:
    path = RESULTS_DIR / f"lambda_{config.dict_kind}.json"
    if path.exists():
        return _read_json(path)
    D, _sha = load_dictionary(config.dict_kind)
    model = p2.build_model(config, D, seed=0).to(torch.device(device)).eval()
    data = p1run.load_split("train", subset=CALIBRATION_MOLECULES)
    loader = p1.make_env_loader(data, BATCH_SIZE, False, 0)
    task_sum = rec_sum = 0.0
    n_mol = n_nodes = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(torch.device(device))
            prediction, aux = model(batch, return_aux=True)
            task_sum += float(F.l1_loss(prediction.view(-1), batch.y.view(-1), reduction="sum"))
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"])
            rec_sum += float((((phi - phi_hat) ** 2).sum(dim=1) / ((phi ** 2).sum(dim=1) + v0.EPS)).sum())
            n_mol += int(batch.y.numel())
            n_nodes += int(phi.shape[0])
    task_init = float(task_sum / max(n_mol, 1))
    rec_init = float(rec_sum / max(n_nodes, 1))
    lambda_base = float(task_init / (rec_init + v0.EPS))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "dict_kind": config.dict_kind,
        "molecules": int(CALIBRATION_MOLECULES),
        "task_init_l1": task_init,
        "rec_init": rec_init,
        "lambda_base": lambda_base,
        "p1_reference_lambda_base": float(p2.LAMBDA_BASE_P1),
        "matches_p1_reference": bool(abs(lambda_base - float(p2.LAMBDA_BASE_P1)) < 1.0e-6),
        "official_test_loaded": False,
    }
    _write_json(path, payload)
    return payload


def lambda_for(config: p2.P2Config) -> float:
    if config.dict_kind == "sdb32":
        return float(config.lambda_factor * p2.LAMBDA_BASE_P1)
    return float(config.lambda_factor * calibrate_lambda_base(config)["lambda_base"])


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(state[key].detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def candidate_gate(config: p2.P2Config, device: str = "cpu") -> dict[str, Any]:
    """Cheap per-candidate integrity gate required by pre-registration section 8."""
    import ast

    path = RESULTS_DIR / "gates" / f"{config.tag.upper()}_gate.json"
    if path.exists():
        return _read_json(path)
    device_obj = torch.device(device)
    D, dict_sha = load_dictionary(config.dict_kind)
    model = p2.build_model(config, D, seed=0).to(device_obj)
    batch = p1run._first_batch(p1run.load_split("train", subset=32), device_obj, 32)
    model.train()
    model.zero_grad(set_to_none=True)
    prediction, aux = model(batch, return_aux=True)
    loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lambda_for(config) * model.reconstruction_loss(
        aux["phi"], aux["coord"]
    )
    loss.backward()
    grad_d = float(model.D.grad.norm()) if model.D.grad is not None else 0.0
    with torch.no_grad():
        alpha = model.code(batch.dict_phi)
    l0 = (alpha.abs() > 0).sum(dim=1)
    active = int((alpha.abs() > 0).any(dim=0).sum())
    names: set[str] = set()
    for module in (p2,):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Name):
                names.add(node.id)
    forbidden = sorted(names & {"patch_cont", "atom_shell", "bond_shell", "path_bond_mean", "adjacent_bond_type"})
    total = p2.total_parameter_count(config)
    actual = int(sum(p.numel() for p in model.parameters()))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "candidate": config.tag.upper(),
        "config": config.as_dict(),
        "dictionary_sha256": dict_sha,
        "loss_finite": bool(torch.isfinite(loss)),
        "task_gradient_to_D": grad_d,
        "exact_top_s": bool(int(l0.max()) <= int(config.s)),
        "max_l0": int(l0.max()),
        "active_atoms_on_batch": active,
        "dictionary_not_collapsed": bool(active >= 8),
        "forbidden_names": forbidden,
        "parameters_accounted": total["whole_model"],
        "parameters_actual": actual,
        "within_budget": bool(total["within_budget"] and actual == total["whole_model"]),
        "no_message_passing": True,
        "no_recurrence": True,
        "official_test_loaded": False,
    }
    payload["passed"] = bool(
        payload["loss_finite"]
        and grad_d > 0.0
        and payload["exact_top_s"]
        and payload["dictionary_not_collapsed"]
        and not forbidden
        and payload["within_budget"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)
    if not payload["passed"]:
        raise RuntimeError(f"candidate gate failed for {config.tag}: {payload}")
    print(f"[gate:{config.tag.upper()}] passed grad_D={grad_d:.4f} active={active}", flush=True)
    return payload


def train_candidate(name: str, device: str = "cuda") -> dict[str, Any]:
    config = resolve_candidate(name)
    _canonicalize_run(config.tag)
    run_path_ = run_path(config.tag)
    if run_path_.exists():
        return _read_json(run_path_)
    device_obj = torch.device(device)
    if config.K == 64:
        fit_dictionary_k64(config.s)
    candidate_gate(config, device="cpu")
    D, dict_sha = load_dictionary(config.dict_kind)
    lam = lambda_for(config)
    train_data = p1run.load_split("train")
    valid_data = p1run.load_split("valid")
    model = p2.build_model(config, D, seed=0).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = p1.make_env_loader(train_data, BATCH_SIZE, True, 0 + TRAIN_SHUFFLE_OFFSET)
    eval_loader = p1.make_env_loader(valid_data, BATCH_SIZE, False, 0 + EVAL_SHUFFLE_OFFSET)
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.cuda.reset_peak_memory_stats(device_obj)
    best_mae = float("inf")
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    epoch_states: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, int(config.horizon) + 1):
        model.train()
        task_sum = rec_sum = 0.0
        n_mol = n_nodes = 0
        for batch in loader:
            batch = batch.to(device_obj)
            prediction, aux = model(batch, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            n_mol += int(batch.y.numel())
            phi = aux["phi"]
            phi_hat = model.reconstruct(phi, aux["coord"]).detach()
            rec_sum += float((((phi - phi_hat) ** 2).sum(dim=1) / ((phi ** 2).sum(dim=1) + v0.EPS)).sum())
            n_nodes += int(phi.shape[0])
        train_mae = float(task_sum / max(n_mol, 1))
        train_rec = float(rec_sum / max(n_nodes, 1))
        valid = p1run.evaluate(model, eval_loader, device_obj)
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": train_mae,
                "train_rec": train_rec,
                "valid_mae": float(valid["mae"]),
                "valid_rec": float(valid["rec"]),
                "d_norm": float(model.D.detach().norm()),
            }
        )
        epoch_states[int(epoch)] = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        keep = {i + 1 for i in sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:SOUP_K]}
        for cached in list(epoch_states):
            if cached not in keep:
                del epoch_states[cached]
        if float(valid["mae"]) < best_mae:
            best_mae = float(valid["mae"])
            best_epoch = int(epoch)
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 20 == 0 or epoch == int(config.horizon):
            print(
                f"[{config.tag}] epoch={epoch:03d} train={train_mae:.6f} rec={train_rec:.2e} "
                f"valid={float(valid['mae']):.6f} best={best_mae:.6f}@{best_epoch}",
                flush=True,
            )
    wall = float(time.perf_counter() - started)
    assert best_state is not None
    members = sorted(
        int(i) + 1 for i in sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:SOUP_K]
    )
    soup_state = {
        k: torch.stack([epoch_states[e][k].float() for e in members]).mean(0) for k in epoch_states[members[0]]
    }
    soup_model = p2.build_model(config, D, seed=0)
    soup_model.load_state_dict(soup_state)
    soup_valid = p1run.evaluate(soup_model.to(device_obj), eval_loader, device_obj)
    dbar_init = np.asarray(v0.normalized_dictionary(torch.as_tensor(D)))
    dbar_soup = np.asarray(v0.normalized_dictionary(soup_state["D"].detach().float()))
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, STATE_DIR / f"{config.tag}_raw_state.pt")
    torch.save(soup_state, STATE_DIR / f"{config.tag}_soup_state.pt")
    _write_csv(CURVE_DIR / f"{config.tag}_curve.csv", curve)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "config": config.as_dict(),
        "parameters": p2.total_parameter_count(config),
        "actual_params": int(_n_params(model)),
        "dictionary_sha256": dict_sha,
        "lambda_rec": lam,
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "soup": {
            "members": members,
            "member_valid_mae": [float(curve[e - 1]["valid_mae"]) for e in members],
            "soup_valid_mae": float(soup_valid["mae"]),
        },
        "train_min_mae": float(min(row["train_mae"] for row in curve)),
        "train_mae_at_best": float(curve[best_epoch - 1]["train_mae"]),
        "valid_rec_soup": float(soup_valid["rec"]),
        "dictionary_movement": {
            "soup_vs_init_fro": float(np.linalg.norm(dbar_soup - dbar_init)),
            "soup_vs_init_relative": float(np.linalg.norm(dbar_soup - dbar_init) / (np.linalg.norm(dbar_init) + v0.EPS)),
        },
        "epochs_run": int(len(curve)),
        "wall_clock_s": wall,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2)) if device_obj.type == "cuda" else None,
        "official_test_loaded": False,
    }
    _write_json(run_path_, payload)
    print(
        f"[{config.tag}] best={best_mae:.6f}@{best_epoch} soup={float(soup_valid['mae']):.6f} "
        f"members={members} wall={wall:.1f}s",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def _canonicalize_run(tag: str) -> None:
    """Mirror a pre-rename run JSON onto its pre-registered artifact name."""
    tag = str(tag).upper()
    target = RESULTS_DIR / RUN_ARTIFACT.get(tag, f"{tag}.json")
    legacy = RESULTS_DIR / f"{tag}.json"
    if not target.exists() and legacy.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(legacy.read_bytes())


def run_path(tag: str) -> Path:
    tag = str(tag).upper()
    return RESULTS_DIR / RUN_ARTIFACT.get(tag, f"{tag}.json")


def _soup_of(tag: str) -> float:
    if tag.upper() == "Z0":
        return float(P1_SOUP)
    _canonicalize_run(tag)
    path = run_path(tag)
    if not path.exists():
        raise RuntimeError(f"run {tag} missing")
    return float(_read_json(path)["soup"]["soup_valid_mae"])


def _select(stage: str, tags: Sequence[str]) -> dict[str, Any]:
    values = {tag.upper(): _soup_of(tag) for tag in tags}
    winner = min(values, key=lambda key: values[key])
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "stage": str(stage),
        "candidates": values,
        "winner": winner,
        "winner_soup_valid_mae": float(values[winner]),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"{stage}_selection.json", payload)
    selections = _read_selections()
    selections[str(stage)] = winner
    _write_json(_selections_path(), selections)
    print(f"[select:{stage}] {payload}", flush=True)
    return payload


def select_stage_a() -> dict[str, Any]:
    return _select("stage_a", ["Z0", "Z1", "Z2", "Z3"])


def select_stage_b() -> dict[str, Any]:
    previous = _stage_selection("stage_a_selection")["winner"]
    parent = "Z2" if previous == "Z0" else previous
    return _select("stage_b", [parent, "H1", "H2"])


def select_stage_c() -> dict[str, Any]:
    previous = _stage_selection("stage_b_selection")["winner"]
    return _select("stage_c", [previous, "E64"])


def select_stage_d() -> dict[str, Any]:
    previous = _stage_selection("stage_c_selection")["winner"]
    previous_mae = _soup_of(previous)
    k64 = _soup_of("K64S8")
    improvement = float(previous_mae - k64)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "stage": "stage_d",
        "candidates": {previous: previous_mae, "K64S8": k64},
        "winner": "K64S8" if k64 < previous_mae else previous,
        "k64s8_improvement": improvement,
        "s12_trigger_threshold": 0.001,
        "s12_triggered": bool(improvement >= 0.001),
        "official_test_loaded": False,
    }
    if payload["s12_triggered"]:
        payload["candidates"]["K64S12"] = _soup_of("K64S12")
        payload["winner"] = min(payload["candidates"], key=lambda key: payload["candidates"][key])
    _write_json(RESULTS_DIR / "stage_d_selection.json", payload)
    selections = _read_selections()
    selections["stage_d"] = payload["winner"]
    _write_json(_selections_path(), selections)
    print(f"[select:stage_d] {payload}", flush=True)
    return payload


def finalize() -> dict[str, Any]:
    winner = _stage_selection("stage_d_selection")["winner"]
    config = _config_by_tag(winner)
    _canonicalize_run(config.tag)
    run = _read_json(run_path(config.tag)) if config.tag.upper() != "Z0" else None
    D, sha = load_dictionary(config.dict_kind)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "winning_candidate": config.tag.upper(),
        "config": config.as_dict(),
        "parameters": p2.total_parameter_count(config),
        "dictionary_sha256": sha,
        "lambda_rec": lambda_for(config) if config.tag.upper() != "Z0" else float(p2.LAMBDA_Z1),
        "K": int(config.K),
        "s": int(config.s),
        "iht_steps": int(p2.IHT_STEPS),
        "horizon": int(config.horizon),
        "soup_valid_mae": float(_soup_of(config.tag.upper())),
        "run": run,
        "official_test_loaded": False,
        "zinc_test_read_this_round": False,
        "bands": {
            "exceptional": 0.120,
            "target": 0.125,
            "strong": 0.130,
            "modest": 0.132,
        },
        "anchors_context_only": {
            "P1": float(P1_SOUP),
            "T1": float(T1_SOUP),
            "FEC_S1": float(FEC_S1_SOUP),
        },
    }
    _write_json(RESULTS_DIR / "final_config.json", payload)
    _write_report(payload)
    _write_decision(payload)
    return payload


def _write_report(payload: Mapping[str, Any]) -> None:
    lines = [
        "# E2E-DictEnv-P2-ABS — report",
        "",
        "ZINC clean dictionary-core absolute-performance search.  Official ZINC test was not loaded.",
        "",
        f"* winning candidate: **{payload['winning_candidate']}**",
        f"* official-valid Top-5 soup MAE = **{payload['soup_valid_mae']:.6f}**",
        f"* params = {payload['parameters']['whole_model']}",
        f"* K = {payload['K']}, s = {payload['s']}, lambda = {payload['lambda_rec']}, horizon = {payload['horizon']}",
        "",
        "## Context anchors (different protocols)",
        "",
        f"* P1 0.131975; T1 0.125765 (coarse bypass); FEC-S1 0.130422",
        f"* commit `{payload['git_commit']}`",
    ]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_decision(payload: Mapping[str, Any]) -> None:
    mae = float(payload["soup_valid_mae"])
    if mae <= 0.120:
        band = "exceptional"
    elif mae <= 0.125:
        band = "target band"
    elif mae <= 0.130:
        band = "strong improvement"
    elif mae <= 0.132:
        band = "modest improvement"
    else:
        band = "no material absolute progress"
    lines = [
        "# E2E-DictEnv-P2-ABS — decision",
        "",
        f"```\nZINC_BEST_CLEAN_DICTIONARY_CORE_VALID_MAE = {mae:.6f}  ({band})\n```",
        "",
        f"* winner `{payload['winning_candidate']}`: {payload['config']}",
        f"* P1 reference {P1_SOUP:.6f} -> delta {mae - P1_SOUP:+.6f}",
        f"* T1 (coarse bypass, context) {T1_SOUP:.6f}; FEC-S1 {FEC_S1_SOUP:.6f}",
        f"* official test not read this round",
    ]
    (RESULTS_DIR / "DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# smoke / bench / correctness
# ---------------------------------------------------------------------------


def smoke(device: str = "cuda") -> dict[str, Any]:
    device_obj = torch.device(device)
    config = _base_candidates()["Z2"]
    D, _sha = load_dictionary("sdb32")
    model = p2.build_model(config, D, seed=0).to(device_obj)
    data = p1run.load_split("train", subset=512)
    loader = p1.make_env_loader(data, BATCH_SIZE, True, TRAIN_SHUFFLE_OFFSET)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lam = lambda_for(config)
    losses: list[float] = []
    started = time.perf_counter()
    for step, batch in enumerate(loader):
        batch = batch.to(device_obj)
        prediction, aux = model(batch, return_aux=True)
        rec = model.reconstruction_loss(aux["phi"], aux["coord"])
        loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step >= 3:
            break
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": str(device_obj),
        "config": config.as_dict(),
        "steps": len(losses),
        "losses": losses,
        "lambda_rec": lam,
        "params": _n_params(model),
        "wall_clock_s": float(time.perf_counter() - started),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2)) if device_obj.type == "cuda" else None,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "smoke.json", payload)
    return payload


def bench(name: str = "Z2", batches: int = 20, device: str = "cuda") -> dict[str, Any]:
    device_obj = torch.device(device)
    config = resolve_candidate(name)
    D, _sha = load_dictionary(config.dict_kind)
    model = p2.build_model(config, D, seed=0).to(device_obj)
    data = p1run.load_split("train")
    loader = p1.make_env_loader(data, BATCH_SIZE, True, TRAIN_SHUFFLE_OFFSET)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lam = lambda_for(config)
    if device_obj.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_obj)
    samples = 0
    started = time.perf_counter()
    model.train()
    step = 0
    for batch in loader:
        batch = batch.to(device_obj)
        prediction, aux = model(batch, return_aux=True)
        rec = model.reconstruction_loss(aux["phi"], aux["coord"])
        loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        samples += int(batch.y.numel())
        step += 1
        if step >= int(batches):
            break
    wall = float(time.perf_counter() - started)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "candidate": config.tag,
        "batches": int(batches),
        "samples": int(samples),
        "wall_clock_s": wall,
        "samples_per_s": float(samples / max(wall, 1e-9)),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2)) if device_obj.type == "cuda" else None,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / f"bench_{config.tag}.json", payload)
    print(f"[bench:{config.tag}] {payload}", flush=True)
    return payload


def correctness(device: str = "cuda") -> dict[str, Any]:
    device_obj = torch.device(device)
    config = resolve_candidate("Z2")
    D, _sha = load_dictionary(config.dict_kind)
    model = p2.build_model(config, D, seed=0).to(device_obj)
    batch = p1run._first_batch(p1run.load_split("train", subset=32), device_obj, 32)
    # task gradient to D
    model.train()
    model.zero_grad(set_to_none=True)
    prediction, aux = model(batch, return_aux=True)
    F.l1_loss(prediction.view(-1), batch.y.view(-1)).backward()
    grad_d_task = float(model.D.grad.norm()) if model.D.grad is not None else 0.0
    # exact sparsity
    with torch.no_grad():
        alpha = model.code(batch.dict_phi)
    l0 = (alpha.abs() > 0).sum(dim=1)
    # dictionary not collapsed
    used = int((alpha.abs() > 0).any(dim=0).sum())
    # forbidden names in the module source
    import ast

    names: set[str] = set()
    for module in (p2, p1):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Name):
                names.add(node.id)
    forbidden = sorted(names & {"patch_cont", "atom_shell", "bond_shell", "path_bond_mean", "adjacent_bond_type"})
    total = p2.total_parameter_count(config)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "config": config.as_dict(),
        "task_gradient_to_D": grad_d_task,
        "exact_top_s": bool(int(l0.max()) <= int(config.s)),
        "max_l0": int(l0.max()),
        "active_atoms_on_batch": used,
        "forbidden_names": forbidden,
        "parameters": total,
        "within_budget": bool(total["within_budget"]),
        "no_message_passing": True,
        "no_recurrence": True,
        "official_test_loaded": False,
    }
    payload["all_passed"] = bool(
        grad_d_task > 0.0
        and payload["exact_top_s"]
        and used >= 8
        and not forbidden
        and total["within_budget"]
    )
    _write_json(RESULTS_DIR / "correctness.json", payload)
    if not payload["all_passed"]:
        raise RuntimeError(f"P2-ABS correctness failed: {payload}")
    return payload


def run_all(device: str = "cuda") -> None:
    correctness(device)
    smoke(device)
    for tag in ("Z1", "Z2", "Z3"):
        train_candidate(tag, device)
    select_stage_a()
    for tag in ("H1", "H2"):
        train_candidate(tag, device)
    select_stage_b()
    train_candidate("E64", device)
    select_stage_c()
    train_candidate("K64S8", device)
    if _soup_of("K64S8") <= _soup_of(_stage_selection("stage_c_selection")["winner"]) - 0.001:
        train_candidate("K64S12", device)
    select_stage_d()
    finalize()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["smoke", "bench", "correct", "train", "select-a", "select-b", "select-c", "select-d", "finalize", "all"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidate", default="Z2")
    parser.add_argument("--batches", type=int, default=20)
    args = parser.parse_args(argv)
    if args.stage == "smoke":
        print(json.dumps(smoke(device=args.device), indent=2))
    elif args.stage == "bench":
        print(json.dumps(bench(name=args.candidate, batches=args.batches, device=args.device), indent=2))
    elif args.stage == "correct":
        print(json.dumps(correctness(device=args.device), indent=2))
    elif args.stage == "train":
        print(json.dumps(train_candidate(args.candidate, device=args.device), indent=2))
    elif args.stage == "select-a":
        print(json.dumps(select_stage_a(), indent=2))
    elif args.stage == "select-b":
        print(json.dumps(select_stage_b(), indent=2))
    elif args.stage == "select-c":
        print(json.dumps(select_stage_c(), indent=2))
    elif args.stage == "select-d":
        print(json.dumps(select_stage_d(), indent=2))
    elif args.stage == "finalize":
        print(json.dumps(finalize(), indent=2))
    else:
        run_all(device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
