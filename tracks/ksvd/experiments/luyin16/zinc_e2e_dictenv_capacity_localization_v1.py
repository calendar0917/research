"""Runner for ``e2e_dictenv_capacity_localization_v1``.

Stages
------
``preflight  init-audit  screen-arm  wave1  wave2  summarize  full-run
  mechanism  report  chain``

Four short warm-start screening arms (M0, F, R, G; 40 epochs each) are launched
with bounded concurrency, then the frozen gate selects at most ONE candidate
for a single from-scratch 320-epoch seed-0 run.  CPU only; the official ZINC
test split is never loaded.  Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_capacity_localization_v1_preregistration.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_capacity_localization_v1 as cl
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = cl.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_capacity_localization_v1"
NOTES_DIR = TRACK_ROOT / "notes"
PREREG_PATH = NOTES_DIR / "e2e_dictenv_capacity_localization_v1_preregistration.md"

SCREENING_DIR = RESULTS_DIR / "screening"
FULL_DIR = RESULTS_DIR / "full"
FULL_CHECKPOINT_DIR = FULL_DIR / "checkpoints"
MECHANISM_DIR = RESULTS_DIR / "mechanism"

CSSD_DIR = TRACK_ROOT / "results/e2e_dictenv_common_subspace_dictionary_v1"
CSSD_SOUP_PATH = CSSD_DIR / "training/checkpoints/CSSD-Q1-seed0_soup_state.pt"
CSSD_SUBSPACE_PATH = CSSD_DIR / "common_subspace.json"
CSSD_CURVE_PATH = CSSD_DIR / "training/curve.csv"

PROBES = ("P1_phi_row_shuffle", "P2_node_correspondence", "P3_edge_correspondence", "P4_relation")
PROBE_SEEDS = (4242, 5150, 6262)

_write_json = p2run._write_json
_read_json = p2run._read_json
_git_commit = p2run._git_commit

ARM_DIRS = {
    "M0": SCREENING_DIR / "m0",
    "F": SCREENING_DIR / "fusion",
    "R": SCREENING_DIR / "relation",
    "G": SCREENING_DIR / "readout",
}
ARM_TAGS = {"M0": "M0", "F": "FUSION", "R": "RELATION", "G": "READOUT"}

THREADS = 4
SCREEN_CONCURRENCY = 3
FULL_THREADS = 4


# ---------------------------------------------------------------------------
# small io helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], header: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow([row.get(key, "") for key in header])


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _ensure_dirs() -> None:
    for path in (RESULTS_DIR, SCREENING_DIR, FULL_DIR, FULL_CHECKPOINT_DIR, MECHANISM_DIR):
        path.mkdir(parents=True, exist_ok=True)
    for path in ARM_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def _load_subspace() -> cssd.CommonSubspace:
    payload = _read_json(CSSD_SUBSPACE_PATH)["q1"]
    return cssd.CommonSubspace(
        components=np.asarray(payload["components"], dtype=np.float64),
        rms=np.asarray(payload["rms"], dtype=np.float64),
        kind=str(payload["kind"]),
    )


def _load_soup() -> dict[str, torch.Tensor]:
    return torch.load(CSSD_SOUP_PATH, map_location="cpu", weights_only=False)


def _payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    cl.official_test_blocker(payload)
    return dict(payload)


# ---------------------------------------------------------------------------
# stage: preflight
# ---------------------------------------------------------------------------


def stage_preflight() -> dict[str, Any]:
    _ensure_dirs()
    cl.cpu_only_guard(torch.device("cpu"))
    if not PREREG_PATH.exists():
        raise FileNotFoundError(f"preregistration missing: {PREREG_PATH}")
    for path in (CSSD_SOUP_PATH, CSSD_SUBSPACE_PATH, CSSD_CURVE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"frozen CAP-BASE artifact missing: {path}")
    subspace = _load_subspace()
    if int(subspace.q) != 1:
        raise RuntimeError(f"CAP-BASE subspace is q{int(subspace.q)}, expected q1")
    soup = _load_soup()
    if "D" not in soup or "U" not in soup:
        raise RuntimeError("CAP-BASE soup checkpoint is missing dictionary/subspace buffers")

    # re-verify the frozen CAP-BASE soup MAE on the official-valid split
    model = cl.build_capacity_model(p2run.load_dictionary(cm.H1_CONFIG.dict_kind)[0], 0, subspace, "M0")
    model.load_state_dict(soup)
    valid = p1run.load_split("valid")
    if not valid:
        raise RuntimeError("official-valid data not available")
    loader = p1.make_env_loader(valid, p2run.BATCH_SIZE, False, p2run.EVAL_SHUFFLE_OFFSET)
    device = audit.attach_cpu(int(THREADS))
    soup_mae = cl.evaluate_mae(model, loader, device, cssd.CSSD_MASK)

    budget = cl.parameter_budget(subspace)
    prereg = {
        "protocol_version": PROTOCOL_VERSION,
        "path": str(PREREG_PATH.relative_to(REPO_ROOT)),
        "sha256": _sha256_file(PREREG_PATH),
        "git_commit": _git_commit(),
        "official_test_loaded": False,
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "threads": int(THREADS),
        "official_test_loaded": False,
        "preregistration": prereg,
        "cap_base": {
            "name": cl.CAP_BASE,
            "soup_checkpoint": str(CSSD_SOUP_PATH.relative_to(REPO_ROOT)),
            "soup_checkpoint_sha256": _sha256_file(CSSD_SOUP_PATH),
            "frozen_soup_mae": cl.CAP_BASE_SOUP_MAE,
            "recomputed_soup_mae": float(soup_mae),
            "mae_reproduced": bool(abs(float(soup_mae) - cl.CAP_BASE_SOUP_MAE) < 5e-7),
            "subspace_q": int(subspace.q),
            "subspace_path": str(CSSD_SUBSPACE_PATH.relative_to(REPO_ROOT)),
            "subspace_sha256": _sha256_file(CSSD_SUBSPACE_PATH),
        },
        "historical_reference": {
            "final_clean_sparse_seed0_soup_mae": cl.FINAL_CLEAN_SPARSE_SOUP_MAE,
            "note": "historical performance reference only; never a matched control",
        },
        "data": {
            "train_cache": str(p1run._env_cache_path("train").relative_to(REPO_ROOT)),
            "valid_cache": str(p1run._env_cache_path("valid").relative_to(REPO_ROOT)),
        },
        "candidates": {kind: cl.CAPACITY_SPECS[kind].as_dict() for kind in cl.KINDS},
        "screening": {
            "epochs": cl.SCREEN_EPOCHS,
            "soup_window": list(cl.SCREEN_SOUP_WINDOW),
            "soup_k": cl.SCREEN_SOUP_K,
            "last_k": cl.SCREEN_LAST10,
            "gate_soup_delta": cl.GATE_SOUP_DELTA,
            "gate_last10_delta": cl.GATE_LAST10_DELTA,
            "strong_soup_delta": cl.STRONG_SOUP_DELTA,
            "tie_tolerance": cl.TIE_TOLERANCE,
            "tie_order": list(cl.TIE_ORDER),
        },
        "full_run": {"epochs": cl.FULL_EPOCHS, "seed": cl.FULL_SEED},
        "parameter_budget": budget,
    }
    _write_json(RESULTS_DIR / "preregistration_snapshot.json", prereg)
    _write_json(RESULTS_DIR / "parameter_budget.json", budget)
    _write_json(RESULTS_DIR / "preflight.json", _payload(payload))
    if not payload["cap_base"]["mae_reproduced"]:
        raise RuntimeError("CAP-BASE soup MAE was not reproduced")
    print(
        f"[preflight] prereg={prereg['sha256'][:12]} soup={soup_mae:.6f} "
        f"ratio={budget['added_params_ratio']:.4f}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: initialisation audit (frozen record)
# ---------------------------------------------------------------------------


def stage_init_audit() -> dict[str, Any]:
    _ensure_dirs()
    cl.cpu_only_guard(torch.device("cpu"))
    subspace = _load_subspace()
    soup = _load_soup()
    dictionary = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)[0]
    train = p1run.load_split("train", subset=64)
    valid = p1run.load_split("valid")
    if not train or not valid:
        raise RuntimeError("data not available")
    device = audit.attach_cpu(int(THREADS))
    eval_loader = p1.make_env_loader(valid, p2run.BATCH_SIZE, False, p2run.EVAL_SHUFFLE_OFFSET)
    grad_batch = next(iter(p1.make_env_loader(train, p2run.BATCH_SIZE, False, 0)))

    base = cl.build_capacity_model(dictionary, 0, subspace, "M0")
    base.load_state_dict(soup)
    base_prediction = cl.predictions_for(base, eval_loader, device, cssd.CSSD_MASK)
    targets = np.concatenate([batch.y.numpy() for batch in eval_loader]).reshape(-1)
    base_mae = float(np.mean(np.abs(base_prediction - targets)))

    rows: dict[str, Any] = {}
    for kind in cl.KINDS:
        model = cl.build_capacity_model(dictionary, 0, subspace, kind)
        report = cl.load_capacity_warm_state(model, soup)
        mae = cl.evaluate_mae(model, eval_loader, device, cssd.CSSD_MASK)
        shift = cl.prediction_shift(model, eval_loader, device, base_prediction)
        if kind == "M0":
            norms: dict[str, float] = {}
        else:
            norms = cl.new_module_gradient_norms(model, grad_batch, device)
        finite = all(np.isfinite(value) for value in norms.values())
        positive = all(value > 0.0 for value in norms.values())
        audit_pass = bool(
            report["bit_identical"]
            and abs(float(mae) - base_mae) <= 0.03
            and finite
            and positive
        )
        rows[kind] = {
            "kind": kind,
            "params": int(sum(parameter.numel() for parameter in model.parameters())),
            "added_params": int(model.capacity_parameter_count()),
            "shared_state_keys": int(report["shared_keys"]),
            "shared_state_bit_identical": bool(report["bit_identical"]),
            "reader_columns_copied": int(report.get("reader_columns_copied", 0)),
            "step0_valid_mae": float(mae),
            "step0_mae_delta_vs_base": float(mae) - base_mae,
            "step0_mean_abs_prediction_delta": float(shift["mean_abs_prediction_delta"]),
            "step0_max_abs_prediction_delta": float(shift["max_abs_prediction_delta"]),
            "new_module_gradient_norms": norms,
            "new_module_gradients_finite": finite,
            "new_module_gradients_positive": positive,
            "audit_pass": audit_pass,
        }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "base_valid_mae": base_mae,
        "rows": rows,
        "all_candidates_pass": bool(all(rows[kind]["audit_pass"] for kind in cl.KINDS)),
    }
    _write_json(RESULTS_DIR / "init_audit.json", _payload(payload))
    print(
        f"[init-audit] base={base_mae:.6f} "
        + " ".join(f"{kind}:{rows[kind]['step0_valid_mae']:.6f}" for kind in cl.KINDS),
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: one screening arm (subprocess entry)
# ---------------------------------------------------------------------------


def stage_screen_arm(kind: str, threads: int = THREADS) -> dict[str, Any]:
    kind = str(kind).upper()
    if kind not in cl.KINDS:
        raise ValueError(f"unknown arm {kind!r}")
    out_dir = ARM_DIRS[kind]
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if result_path.exists():
        print(f"[screen-{kind}] cache hit", flush=True)
        return _read_json(result_path)

    cl.cpu_only_guard(torch.device("cpu"))
    cl.official_test_blocker({"official_test_loaded": False})
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    subspace = _load_subspace()
    soup = _load_soup()
    dictionary = p2run.load_dictionary(cm.H1_CONFIG.dict_kind)[0]
    train = p1run.load_split("train")
    valid = p1run.load_split("valid")
    if not train or not valid:
        raise RuntimeError("data not available")

    p2run._seed_everything(cl.FULL_SEED)
    model = cl.build_capacity_model(dictionary, cl.FULL_SEED, subspace, kind)
    warm_report = cl.load_capacity_warm_state(model, soup)

    # step-0 audit, in eval mode so it cannot consume the dropout RNG stream
    device = audit.attach_cpu(int(threads))
    eval_loader = p1.make_env_loader(valid, p2run.BATCH_SIZE, False, p2run.EVAL_SHUFFLE_OFFSET)
    base = cl.build_capacity_model(dictionary, cl.FULL_SEED, subspace, "M0")
    base.load_state_dict(soup)
    base_prediction = cl.predictions_for(base, eval_loader, device, cssd.CSSD_MASK)
    targets = np.concatenate([batch.y.numpy() for batch in eval_loader]).reshape(-1)
    base_mae = float(np.mean(np.abs(base_prediction - targets)))
    step0_mae = cl.evaluate_mae(model, eval_loader, device, cssd.CSSD_MASK)
    shift = cl.prediction_shift(model, eval_loader, device, base_prediction)
    grad_batch = next(iter(p1.make_env_loader(train, p2run.BATCH_SIZE, False, 0)))
    grad_norms = cl.new_module_gradient_norms(model, grad_batch, device)

    # the frozen training loop consumes the global RNG stream from here
    payload = cl.train_screen(
        tag=f"SCREEN-{ARM_TAGS[kind]}",
        model=model,
        dictionary=dictionary,
        subspace=subspace,
        epochs=cl.SCREEN_EPOCHS,
        threads=int(threads),
        train_data=train,
        valid_data=valid,
        seed=cl.FULL_SEED,
        soup_window=cl.SCREEN_SOUP_WINDOW,
        soup_k=cl.SCREEN_SOUP_K,
        last_k=cl.SCREEN_LAST10,
        log=True,
        return_soup_state=True,
    )
    payload.update(
        {
            "git_commit": _git_commit(),
            "arm": kind,
            "warm_start": str(CSSD_SOUP_PATH.relative_to(REPO_ROOT)),
            "warm_state_shared_keys": int(warm_report["shared_keys"]),
            "warm_state_bit_identical": bool(warm_report["bit_identical"]),
            "step0_base_valid_mae": base_mae,
            "step0_valid_mae": float(step0_mae),
            "step0_mae_delta_vs_base": float(step0_mae) - base_mae,
            "step0_mean_abs_prediction_delta": float(shift["mean_abs_prediction_delta"]),
            "step0_max_abs_prediction_delta": float(shift["max_abs_prediction_delta"]),
            "new_module_gradient_norms": grad_norms,
            "new_module_gradient_norm_max": float(max(grad_norms.values())) if grad_norms else 0.0,
            "params": int(sum(parameter.numel() for parameter in model.parameters())),
            "added_params": int(model.capacity_parameter_count()),
        }
    )
    curve = payload.pop("curve")
    _write_csv(
        out_dir / "curve.csv",
        curve,
        ["epoch", "train_mae", "train_rec", "train_rec_term", "valid_mae", "d_norm"],
    )
    soup_state = payload.pop("soup_state")
    _write_json(
        out_dir / "soup.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "kind": kind,
            "members": payload["soup_members"],
            "member_valid_mae": payload["soup_member_valid_mae"],
            "soup_valid_mae": payload["soup_valid_mae"],
        },
    )
    torch.save(soup_state, out_dir / "soup_state.pt")
    _write_json(result_path, _payload(payload))
    print(
        f"[screen-{kind}] soup={payload['soup_valid_mae']:.6f} "
        f"last10={payload['last10_mean_valid_mae']:.6f} best={payload['best_valid_mae']:.6f} "
        f"wall={payload['wall_clock_s']:.0f}s",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# launch helpers
# ---------------------------------------------------------------------------


def _spawn(args: Sequence[str], threads: int, env_extra: Mapping[str, str] | None = None) -> subprocess.Popen:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(int(threads))
    env["MKL_NUM_THREADS"] = str(int(threads))
    env["CUDA_VISIBLE_DEVICES"] = ""
    if env_extra:
        env.update(env_extra)
    command = [sys.executable, "-m", "tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_capacity_localization_v1", *args]
    print(f"[spawn] {' '.join(args)} (threads={threads})", flush=True)
    return subprocess.Popen(command, cwd=str(REPO_ROOT), env=env)


def launch_wave(kinds: Sequence[str], threads: int, concurrency: int) -> None:
    queue = [str(kind) for kind in kinds]
    running: list[tuple[str, subprocess.Popen]] = []
    concurrency = max(1, min(int(concurrency), len(queue)))
    while queue or running:
        while queue and len(running) < concurrency:
            kind = queue.pop(0)
            if (ARM_DIRS[kind] / "result.json").exists():
                print(f"[wave] skip {kind} (cached)", flush=True)
                continue
            running.append(
                (kind, _spawn(["screen-arm", "--kind", kind, "--threads", str(int(threads))], threads))
            )
        time.sleep(2.0)
        for item in list(running):
            kind, process = item
            if process.poll() is not None:
                running.remove(item)
                if process.returncode != 0:
                    raise RuntimeError(f"screening arm {kind} failed with code {process.returncode}")


# ---------------------------------------------------------------------------
# stage: summarize + frozen decision
# ---------------------------------------------------------------------------


def _arm_row(kind: str) -> dict[str, Any]:
    return _read_json(ARM_DIRS[kind] / "result.json")


def stage_summarize() -> dict[str, Any]:
    _ensure_dirs()
    rows = {kind: _arm_row(kind) for kind in cl.KINDS}
    deltas = cl.screening_deltas(rows)
    gates = {kind: cl.capacity_gate(rows[kind], deltas[kind]) for kind in ("F", "R", "G")}
    selection = cl.select_winner(rows, gates)
    budget = cl.parameter_budget(_load_subspace())

    table: list[dict[str, Any]] = []
    for kind in ("M0", "F", "R", "G"):
        row = rows[kind]
        entry: dict[str, Any] = {
            "candidate": kind,
            "params": row["params"],
            "added_params": row["added_params"],
            "step0_mae": row["step0_valid_mae"],
            "step0_prediction_shift": row["step0_mean_abs_prediction_delta"],
            "step0_prediction_shift_max": row["step0_max_abs_prediction_delta"],
            "best_valid": row["best_valid_mae"],
            "best_epoch": row["best_epoch"],
            "screen_soup": row["soup_valid_mae"],
            "last10_mean": row["last10_mean_valid_mae"],
            "new_branch_grad_norm": row["new_module_gradient_norm_max"],
            "seconds_per_epoch": row["seconds_per_epoch"],
            "wall_clock_s": row["wall_clock_s"],
        }
        if kind == "M0":
            entry.update(
                {
                    "delta_soup_vs_M0": 0.0,
                    "delta_last10_vs_M0": 0.0,
                    "gate_pass": "control",
                    "verdict": "CONTROL",
                }
            )
        else:
            entry.update(
                {
                    "delta_soup_vs_M0": deltas[kind]["delta_soup_vs_M0"],
                    "delta_last10_vs_M0": deltas[kind]["delta_last10_vs_M0"],
                    "gate_pass": bool(gates[kind]["passed"]),
                    "verdict": gates[kind]["verdict"],
                }
            )
        table.append(entry)
    header = [
        "candidate",
        "params",
        "added_params",
        "step0_mae",
        "step0_prediction_shift",
        "step0_prediction_shift_max",
        "best_valid",
        "best_epoch",
        "screen_soup",
        "last10_mean",
        "delta_soup_vs_M0",
        "delta_last10_vs_M0",
        "new_branch_grad_norm",
        "gate_pass",
        "verdict",
        "seconds_per_epoch",
        "wall_clock_s",
    ]
    _write_csv(SCREENING_DIR / "screening_summary.csv", table, header)
    decision = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "control": "M0",
        "control_soup": rows["M0"]["soup_valid_mae"],
        "control_last10": rows["M0"]["last10_mean_valid_mae"],
        "rows": table,
        "deltas": deltas,
        "gates": gates,
        "selection": selection,
        "parameter_budget": {
            "cap_base_params": budget["cap_base_params"],
            "candidate_added_params": budget["candidate_added_params"],
            "added_params_ratio": budget["added_params_ratio"],
        },
    }
    _write_json(SCREENING_DIR / "decision.json", _payload(decision))
    print(
        f"[summarize] M0={rows['M0']['soup_valid_mae']:.6f} "
        + " ".join(
            f"{kind}:{deltas[kind]['delta_soup_vs_M0']:+.6f}/{gates[kind]['verdict']}"
            for kind in ("F", "R", "G")
        )
        + f" -> winner={selection['winner']}",
        flush=True,
    )
    return decision


# ---------------------------------------------------------------------------
# stage: full run (only if bought)
# ---------------------------------------------------------------------------


def _cssd_reference_curve() -> list[float]:
    rows = _read_csv_rows(CSSD_CURVE_PATH)
    rows.sort(key=lambda row: int(row["epoch"]))
    return [float(row["valid_mae"]) for row in rows]


def _full_early_stop_callback(reference: Sequence[float], capacity_names: Sequence[str]):
    state = {"worse_streak": 0, "window": []}

    def callback(epoch: int, model: Any, optimizer: Any, curve: list[dict[str, Any]]):
        row = curve[-1]
        valid = float(row["valid_mae"])
        payload: dict[str, Any] = {"epoch": int(epoch), "valid_mae": valid}
        if not np.isfinite(valid) or not np.isfinite(float(row["train_mae"])):
            payload["reason"] = "non_finite"
            return False, payload
        if valid > float(cssd.CATASTROPHIC_MAE):
            payload["reason"] = "divergence"
            return False, payload
        if not bool(torch.isfinite(model.D).all()):
            payload["reason"] = "dictionary_non_finite"
            return False, payload
        if int(epoch) % 20 == 0:
            parameters = dict(model.named_parameters())
            norms = {
                name: float(parameters[name].grad.detach().norm())
                for name in capacity_names
                if parameters[name].grad is not None
            }
            payload["capacity_grad_norms"] = norms
            if norms and all(value == 0.0 for value in norms.values()):
                payload["reason"] = "capacity_gradient_zero"
                return False, payload
        if int(epoch) <= len(reference):
            delta = valid - float(reference[int(epoch) - 1])
            payload["delta_vs_cssd_curve"] = delta
            if delta > 0.03:
                state["worse_streak"] += 1
            else:
                state["worse_streak"] = 0
            state["window"].append(valid)
            if len(state["window"]) > 20:
                state["window"].pop(0)
            improving = True
            if len(state["window"]) == 20:
                last10 = min(state["window"][-10:])
                previous10 = min(state["window"][:10])
                improving = last10 < previous10
            payload["worse_streak"] = state["worse_streak"]
            payload["improving_trend"] = improving
            if state["worse_streak"] >= 40 and not improving:
                payload["reason"] = "matched_cssd_curve_hopeless"
                return False, payload
        return True, payload

    return callback


def stage_full(threads: int = FULL_THREADS, force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    final_path = FULL_DIR / "final.json"
    if final_path.exists() and not force:
        print("[full] cache hit", flush=True)
        return _read_json(final_path)
    decision = _read_json(SCREENING_DIR / "decision.json")
    winner = decision["selection"].get("winner")
    if winner is None:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "device": "cpu",
            "official_test_loaded": False,
            "run": False,
            "reason": "no candidate passed the preregistered capacity gate",
            "selection": decision["selection"],
        }
        _write_json(FULL_DIR / "NOT_RUN.json", _payload(payload))
        print("[full] no winner -> NOT_RUN", flush=True)
        return payload

    cl.cpu_only_guard(torch.device("cpu"))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    subspace = _load_subspace()
    train = p1run.load_split("train")
    valid = p1run.load_split("valid")
    if not train or not valid:
        raise RuntimeError("data not available")

    def factory(dictionary: np.ndarray, seed: int, subspace_: cssd.CommonSubspace) -> cl.CapacityModel:
        return cl.build_capacity_model(dictionary, seed, subspace_, winner)

    reference = _cssd_reference_curve()
    probe = cl.build_capacity_model(p2run.load_dictionary(cm.H1_CONFIG.dict_kind)[0], 0, subspace, winner)
    capacity_names = probe.capacity_parameter_names()
    callback = _full_early_stop_callback(reference, capacity_names)
    tag = f"CAP-{winner}-seed{cl.FULL_SEED}"
    started = time.perf_counter()
    payload = cssd.train_cssd(
        tag=tag,
        epochs=cl.FULL_EPOCHS,
        threads=int(threads),
        out_dir=FULL_CHECKPOINT_DIR,
        subspace=subspace,
        train_data=train,
        valid_data=valid,
        seed=cl.FULL_SEED,
        callback=callback,
        model_factory=factory,
        save_states=True,
        log=True,
    )
    payload.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "device": "cpu",
            "official_test_loaded": False,
            "winner": winner,
            "seed": cl.FULL_SEED,
            "from_scratch": True,
            "selection": decision["selection"],
            "params": int(sum(parameter.numel() for parameter in probe.parameters())),
            "cap_base_params": cl.parameter_budget(subspace)["cap_base_params"],
            "round_seconds": float(time.perf_counter() - started),
        }
    )
    payload["interpretation"] = cl.full_interpretation(float(payload["soup"]["soup_valid_mae"]))
    curve = payload.pop("curve")
    _write_csv(
        FULL_DIR / "curve.csv",
        curve,
        ["epoch", "train_mae", "train_rec", "train_rec_term", "valid_mae", "d_norm"],
    )
    _write_json(
        FULL_DIR / "soup.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "winner": winner,
            "members": payload["soup"]["members"],
            "member_valid_mae": payload["soup"]["member_valid_mae"],
            "soup_valid_mae": payload["soup"]["soup_valid_mae"],
            "interpretation": payload["interpretation"],
        },
    )
    _write_json(final_path, _payload(payload))
    print(
        f"[full] winner={winner} completed={payload['completed']} epochs={payload['epochs_run']} "
        f"soup={payload['soup']['soup_valid_mae']:.6f} "
        f"band={payload['interpretation']['band']}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: mechanism audit (only when the full run exists)
# ---------------------------------------------------------------------------


def _predictions_with_phi_shuffle(
    model: cl.CapacityModel, loader: Any, device: torch.device, mask: audit.AuditMask, seed: int
) -> np.ndarray:
    generator = torch.Generator().manual_seed(int(seed))
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            permutation = torch.randperm(int(batch.dict_phi.shape[0]), generator=generator)
            batch.dict_phi = batch.dict_phi[permutation]
            predictions.append(model(batch, mask=mask).view(-1).cpu().numpy())
    return np.concatenate(predictions) if predictions else np.zeros((0,))


def _probe_delta(
    base_mae: float, probe_mae: float, base_prediction: np.ndarray | None, probe_prediction: np.ndarray | None
) -> dict[str, Any]:
    payload: dict[str, Any] = {"mae": float(probe_mae), "delta_mae": float(probe_mae) - float(base_mae)}
    if base_prediction is not None and probe_prediction is not None:
        shift = np.abs(probe_prediction - base_prediction)
        payload["mean_abs_prediction_delta"] = float(shift.mean())
        payload["max_abs_prediction_delta"] = float(shift.max())
    return payload


def stage_mechanism(threads: int = THREADS, force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    out_path = MECHANISM_DIR / "probes.json"
    if out_path.exists() and not force:
        print("[mechanism] cache hit", flush=True)
        return _read_json(out_path)
    final = _read_json(FULL_DIR / "final.json") if (FULL_DIR / "final.json").exists() else None
    if final is None or not final.get("winner"):
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "run": False,
            "reason": "full winner run not available",
        }
        _write_json(out_path, _payload(payload))
        return payload

    winner = str(final["winner"])
    subspace = _load_subspace()
    state = torch.load(
        FULL_CHECKPOINT_DIR / f"CAP-{winner}-seed{cl.FULL_SEED}_soup_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = cl.build_capacity_model(p2run.load_dictionary(cm.H1_CONFIG.dict_kind)[0], cl.FULL_SEED, subspace, winner)
    model.load_state_dict(state)
    valid = p1run.load_split("valid")
    if not valid:
        raise RuntimeError("valid data not available")
    train_diag = p1run.load_split("train", subset=p2run.BATCH_SIZE)
    if not train_diag:
        raise RuntimeError("train data not available")
    grad_batch = next(iter(p1.make_env_loader(train_diag, p2run.BATCH_SIZE, False, 0)))
    device = audit.attach_cpu(int(threads))
    mask = cssd.CSSD_MASK
    loader = p1.make_env_loader(valid, p2run.BATCH_SIZE, False, p2run.EVAL_SHUFFLE_OFFSET)
    base_prediction = cl.predictions_for(model, loader, device, mask)
    targets = np.concatenate([batch.y.numpy() for batch in loader]).reshape(-1)
    base_mae = float(np.mean(np.abs(base_prediction - targets)))

    rows: list[dict[str, Any]] = []
    # P1 — dictionary coordinate row shuffle (distribution preserving)
    for seed in PROBE_SEEDS:
        prediction = _predictions_with_phi_shuffle(model, loader, device, mask, seed)
        rows.append(
            {
                "probe": "P1_phi_row_shuffle",
                "probe_group": "all",
                "seed": int(seed),
                **_probe_delta(base_mae, float(np.mean(np.abs(prediction - targets))), base_prediction, prediction),
            }
        )
    # P2 / P3 — assignment-preserving node / edge correspondence shuffles
    for name, kind, field in (
        ("P2_node_correspondence", "node", "node"),
        ("P3_edge_correspondence", "edge", "edge"),
    ):
        probe_mask = cm.merge_masks(mask, audit.AuditMask(**{f"use_{field}_shuffle": True}))
        for seed in PROBE_SEEDS:
            audit.prepare_shuffles(valid, kind, (int(seed), int(seed) + 1))
            try:
                result = audit.evaluate_mask(model, loader, device, probe_mask)
            finally:
                audit.clear_shuffles(valid)
            rows.append(
                {
                    "probe": name,
                    "probe_group": kind,
                    "seed": int(seed),
                    **_probe_delta(
                        base_mae, float(result["mae"]), base_prediction, np.asarray(result["predictions"])
                    ),
                }
            )
    # P4 — distribution-preserving relation row shuffle
    for group in ("all", "distance", "overlap", "boundary"):
        for seed in PROBE_SEEDS:
            restore = audit.permute_pair_rows(valid, group, seed)
            try:
                result = audit.evaluate_mask(model, loader, device, mask)
            finally:
                restore()
            rows.append(
                {
                    "probe": "P4_relation",
                    "probe_group": group,
                    "seed": int(seed),
                    **_probe_delta(
                        base_mae, float(result["mae"]), base_prediction, np.asarray(result["predictions"])
                    ),
                }
            )
    # P5 — winner branch disable
    model.capacity_off = True
    disabled_prediction = cl.predictions_for(model, loader, device, mask)
    model.capacity_off = False
    rows.append(
        {
            "probe": "P5_branch_disable",
            "probe_group": winner,
            "seed": 0,
            **_probe_delta(
                base_mae, float(np.mean(np.abs(disabled_prediction - targets))), base_prediction, disabled_prediction
            ),
        }
    )
    _write_csv(
        MECHANISM_DIR / "probes.csv",
        rows,
        [
            "probe",
            "probe_group",
            "seed",
            "mae",
            "delta_mae",
            "mean_abs_prediction_delta",
            "max_abs_prediction_delta",
        ],
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "winner": winner,
        "base_valid_mae": base_mae,
        "rows": rows,
        "candidate_specific": _candidate_specific_diagnostics(model, winner, grad_batch, device, loader),
    }
    _write_json(out_path, _payload(payload))
    _write_json(MECHANISM_DIR / "candidate_specific.json", payload["candidate_specific"])
    print(f"[mechanism] winner={winner} base={base_mae:.6f} probes={len(rows)}", flush=True)
    return payload


def _candidate_specific_diagnostics(
    model: cl.CapacityModel,
    winner: str,
    grad_batch: Any,
    device: torch.device,
    loader: Any,
) -> dict[str, Any]:
    """Winner-specific frozen diagnostics (no loss term, no rescue)."""
    model.eval()
    batch = next(iter(loader)).to(device)
    payload: dict[str, Any] = {"winner": winner}
    if winner == "F":
        with torch.no_grad():
            coord = model.code(batch.dict_phi)
            heads = model.head_products(coord, batch)
            for name, value in heads.items():
                norms = value.norm(dim=2)  # [H, occurrences]
                payload[f"{name}_head_mean_norm"] = norms.mean(dim=1).tolist()
                payload[f"{name}_head_std_norm"] = norms.std(dim=1).tolist()
                flat = value.permute(1, 0, 2).reshape(value.shape[1], -1)
                cosine = torch.nn.functional.cosine_similarity(
                    flat[:, None, :], flat[None, :, :], dim=2
                )
                payload[f"{name}_head_cosine"] = cosine.tolist()
        train_batch = grad_batch.to(device)
        parameters = dict(model.named_parameters())
        model.train(False)
        batch_train = train_batch
        prediction, aux = model(batch_train, mask=cssd.CSSD_MASK, return_aux=True)
        loss = torch.nn.functional.l1_loss(prediction.view(-1), batch_train.y.view(-1)) + float(
            cm.H1_LAMBDA
        ) * model.reconstruction_loss(aux["phi"], aux["coord"])
        model.zero_grad(set_to_none=True)
        loss.backward()
        per_head: dict[str, float] = {}
        for head in range(cl.FUSION_HEADS):
            total = 0.0
            for name in (f"F_NS.{head}.weight", f"F_NC.{head}.weight", f"F_ES.{head}.weight", f"F_EC.{head}.weight"):
                grad = parameters[name].grad
                if grad is not None:
                    total += float(grad.detach().norm() ** 2)
            per_head[f"head{head}"] = float(np.sqrt(total))
        payload["per_head_gradient_norm"] = per_head
        model.zero_grad(set_to_none=True)
        model.train(True)
    elif winner == "R":
        captured: dict[str, torch.Tensor] = {}

        def hook(module, inputs, output):  # noqa: ANN001
            captured["input"] = inputs[0].detach()
            captured["output"] = output.detach()

        handle = model.pair_encoder.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(batch, mask=cssd.CSSD_MASK)
        finally:
            handle.remove()
        with torch.no_grad():
            contributions = model.pair_block_contributions(captured["input"], captured["input"][:, -int(p2.PAIR_HIDDEN) :])
        for index, entry in enumerate(contributions):
            payload[f"block{index}_contribution_norm"] = float(entry["contribution"].norm(dim=1).mean())
            payload[f"block{index}_modulation_mean_abs"] = float(entry["modulation"].abs().mean())
            payload[f"block{index}_shift_mean_abs"] = float(entry["shift"].abs().mean())
    elif winner == "G":
        with torch.no_grad():
            _, aux = model(batch, mask=cssd.CSSD_MASK, return_aux=True)
            pair_batch = batch.batch[batch.pair_index[0]]
            h_env, h_pair = model.summary_values(
                aux["E"], batch.batch, aux["pair_value"], pair_batch, int(batch.global_context.shape[0])
            )
            payload["summary_env_norm_mean"] = float(h_env.norm(dim=1).mean())
            payload["summary_pair_norm_mean"] = float(h_pair.norm(dim=1).mean())
            activation_env = torch.sigmoid(model.summary_gate_env(aux["E"]))
            activation_pair = torch.sigmoid(model.summary_gate_pair(aux["pair_value"]))
            payload["gate_env_mean"] = float(activation_env.mean())
            payload["gate_env_std"] = float(activation_env.std())
            payload["gate_pair_mean"] = float(activation_pair.mean())
            payload["gate_pair_std"] = float(activation_pair.std())
            for name, activation in (("env", activation_env), ("pair", activation_pair)):
                mean_activation = activation.mean(dim=0)
                probabilities = mean_activation / mean_activation.sum().clamp_min(1e-12)
                entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
                payload[f"gate_{name}_entropy"] = entropy
        for kind in ("env", "pair"):
            model.disabled_summaries = {kind}
            disabled = cl.evaluate_mae(model, loader, device, cssd.CSSD_MASK)
            model.disabled_summaries = set()
            payload[f"disable_{kind}_summary_delta_mae"] = float(disabled) - float(
                cl.evaluate_mae(model, loader, device, cssd.CSSD_MASK)
            )
    return payload


# ---------------------------------------------------------------------------
# stage: report tables
# ---------------------------------------------------------------------------


def stage_report() -> dict[str, Any]:
    _ensure_dirs()
    decision = _read_json(SCREENING_DIR / "decision.json")
    init_audit = _read_json(RESULTS_DIR / "init_audit.json")
    budget = _read_json(RESULTS_DIR / "parameter_budget.json")
    summary: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "preregistration": _read_json(RESULTS_DIR / "preregistration_snapshot.json"),
        "parameter_budget": budget,
        "init_audit": init_audit,
        "screening": decision,
    }
    if (FULL_DIR / "final.json").exists():
        final = _read_json(FULL_DIR / "final.json")
        summary["full"] = {
            "winner": final.get("winner"),
            "params": final.get("params"),
            "cap_base_params": final.get("cap_base_params"),
            "best_valid_mae": final.get("best_valid_mae"),
            "best_epoch": final.get("best_epoch"),
            "soup": final.get("soup"),
            "completed": final.get("completed"),
            "epochs_run": final.get("epochs_run"),
            "wall_clock_s": final.get("wall_clock_s"),
            "interpretation": final.get("interpretation"),
        }
    elif (FULL_DIR / "NOT_RUN.json").exists():
        summary["full"] = {"run": False, "reason": _read_json(FULL_DIR / "NOT_RUN.json")["reason"]}
    if (MECHANISM_DIR / "probes.json").exists():
        summary["mechanism"] = _read_json(MECHANISM_DIR / "probes.json")
    _write_json(RESULTS_DIR / "summary.json", _payload(summary))
    _write_analysis_tables(summary)
    print("[report] analysis_tables.md and summary.json written", flush=True)
    return summary


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}"


def _write_analysis_tables(summary: Mapping[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Analysis tables — `e2e_dictenv_capacity_localization_v1`")
    lines.append("")
    prereg = summary["preregistration"]
    lines.append(
        f"Preregistration sha256 `{prereg['sha256'][:12]}…`; commit `{summary['git_commit'][:12]}`; "
        f"CAP-BASE = {cl.CAP_BASE}; official test never loaded."
    )
    lines.append("")
    lines.append("## 1. Parameter budget")
    lines.append("")
    budget = summary["parameter_budget"]
    lines.append(
        f"CAP-BASE {budget['cap_base_params']} params; added-parameter ratio "
        f"{budget['added_params_ratio']:.4f} (max {cl.BUDGET_RATIO_MAX})."
    )
    lines.append("")
    lines.append("| candidate | added | total | relative | preferred | hard ceiling |")
    lines.append("|---|---|---|---|---|---|")
    for kind in ("M0", "F", "R", "G"):
        row = budget["rows"][kind]
        lines.append(
            f"| {kind} | {row['added_params_vs_cap_base']} | {row['total_params']} | "
            f"{row['relative_increase'] * 100:.2f} % | {row['within_preferred']} | {row['within_hard_ceiling']} |"
        )
    lines.append("")
    lines.append("## 2. Initialization audit (step 0, before any training)")
    lines.append("")
    lines.append("| candidate | step0 MAE | Δ vs CAP-BASE | mean shift | max shift | grads > 0 | bit-identical |")
    lines.append("|---|---|---|---|---|---|---|")
    for kind in ("M0", "F", "R", "G"):
        row = summary["init_audit"]["rows"][kind]
        lines.append(
            f"| {kind} | {_fmt(row['step0_valid_mae'])} | {_fmt(row['step0_mae_delta_vs_base'])} | "
            f"{_fmt(row['step0_mean_abs_prediction_delta'], 5)} | {_fmt(row['step0_max_abs_prediction_delta'], 5)} | "
            f"{row['new_module_gradients_positive']} | {row['shared_state_bit_identical']} |"
        )
    lines.append("")
    lines.append("## 3. Screening (40 epochs warm from the CAP-BASE soup, fresh Adam)")
    lines.append("")
    lines.append(
        "| candidate | params | step0 MAE | best valid | best epoch | soup (21–40) | last-10 mean | "
        "Δsoup vs M0 | Δlast10 vs M0 | branch grad | verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in summary["screening"]["rows"]:
        lines.append(
            f"| {row['candidate']} | {row['params']} | {_fmt(row['step0_mae'])} | {_fmt(row['best_valid'])} | "
            f"{row['best_epoch']} | {_fmt(row['screen_soup'])} | {_fmt(row['last10_mean'])} | "
            f"{_fmt(row['delta_soup_vs_M0'])} | {_fmt(row['delta_last10_vs_M0'])} | "
            f"{_fmt(row['new_branch_grad_norm'], 3)} | {row['verdict']} |"
        )
    lines.append("")
    selection = summary["screening"]["selection"]
    lines.append(
        f"Winner: **{selection['winner']}** ({selection['reason']}); passing: "
        f"{selection['passing']}; frozen tie order F > R > G."
    )
    lines.append("")
    full = summary.get("full")
    lines.append("## 4. Full winner run (only if bought)")
    lines.append("")
    if full and full.get("winner"):
        lines.append(
            f"Winner **{full['winner']}**, params {full['params']} (CAP-BASE {full['cap_base_params']}), "
            f"best valid {_fmt(full['best_valid_mae'])} @ {full['best_epoch']}, "
            f"soup {_fmt(full['soup']['soup_valid_mae'])} (members {full['soup']['members']}), "
            f"epochs {full['epochs_run']}, wall {full['wall_clock_s']:.0f}s."
        )
        interpretation = full["interpretation"]
        lines.append("")
        lines.append(
            f"Band **{interpretation['band']}**; Δ vs CAP-BASE {_fmt(interpretation['delta_vs_cap_base'])}, "
            f"Δ vs FINAL-CLEAN sparse {_fmt(interpretation['delta_vs_final_clean_sparse'])}."
        )
    else:
        lines.append(f"Not run: {full.get('reason') if full else 'no decision artifact'}.")
    lines.append("")
    mechanism = summary.get("mechanism")
    lines.append("## 5. Mechanism audit (frozen winner checkpoint)")
    lines.append("")
    if mechanism and mechanism.get("base_valid_mae") is not None:
        lines.append("| probe | group | seed | MAE | Δ MAE | mean shift | max shift |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in mechanism["rows"]:
            lines.append(
                f"| {row['probe']} | {row['probe_group']} | {row['seed']} | {_fmt(row['mae'])} | "
                f"{_fmt(row['delta_mae'])} | {_fmt(row.get('mean_abs_prediction_delta'), 5)} | "
                f"{_fmt(row.get('max_abs_prediction_delta'), 5)} |"
            )
        lines.append("")
        lines.append("Candidate-specific diagnostics: `mechanism/candidate_specific.json`.")
    else:
        lines.append("Not run (no full winner).")
    lines.append("")
    (RESULTS_DIR / "analysis_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# chain / CLI
# ---------------------------------------------------------------------------


def chain(threads: int = THREADS) -> None:
    _ensure_dirs()
    stage_preflight()
    stage_init_audit()
    launch_wave(("M0", "F", "R"), threads, SCREEN_CONCURRENCY)
    launch_wave(("G",), threads, 1)
    decision = stage_summarize()
    if decision["selection"]["winner"] is not None:
        stage_full(threads=FULL_THREADS)
        stage_mechanism(threads=threads)
    stage_report()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "init-audit", "wave1", "wave2", "summarize", "full-run", "mechanism", "report"):
        stage = sub.add_parser(name)
        stage.add_argument("--threads", type=int, default=THREADS)
        if name in ("full-run", "mechanism", "report"):
            stage.add_argument("--force", action="store_true")
    screen = sub.add_parser("screen-arm")
    screen.add_argument("--kind", required=True)
    screen.add_argument("--threads", type=int, default=THREADS)
    chain_parser = sub.add_parser("chain")
    chain_parser.add_argument("--threads", type=int, default=THREADS)

    args = parser.parse_args(argv)
    if args.command == "preflight":
        stage_preflight()
    elif args.command == "init-audit":
        stage_init_audit()
    elif args.command == "screen-arm":
        stage_screen_arm(args.kind, args.threads)
    elif args.command == "wave1":
        launch_wave(("M0", "F", "R"), args.threads, SCREEN_CONCURRENCY)
    elif args.command == "wave2":
        launch_wave(("G",), args.threads, 1)
    elif args.command == "summarize":
        stage_summarize()
    elif args.command == "full-run":
        stage_full(threads=args.threads, force=bool(args.force))
    elif args.command == "mechanism":
        stage_mechanism(threads=args.threads, force=bool(args.force))
    elif args.command == "report":
        stage_report()
    elif args.command == "chain":
        chain(threads=args.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
