"""Post-training, Validation-only mechanism checks on the frozen checkpoints.

No retraining, no checkpoint reselection, no hyper-parameter change, no Test
label.  Three forward modes of the *same* frozen weights are compared:

* ``real``            -- the trained model (baseline replay);
* ``message_off``     -- the message term is forced to zero in both layers;
* ``message_shuffle`` -- inside every molecule only the ``src`` assignment of the
  dynamic edges is permuted (``dst`` multiplicity, ``h0``, all W2 static inputs
  and the bond features are unchanged).

Both interventions create out-of-distribution inputs. They measure the model's
sensitivity to the real message correspondence; they are **not** a causal proof
of an independent message-passing generalisation gain and are never equivalent
to the historical independently trained STATIC control.  No number produced here
is used to change anything.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import torch  # noqa: E402

import dy_common as C  # noqa: E402
import dy_model as M  # noqa: E402

REPLAY_TOL = 1e-4


def load_frozen(path: Path, task: Any, device: torch.device) -> tuple[M.W2DynamicModel, dict[str, Any], str]:
    if not Path(path).is_file():
        raise RuntimeError(f"checkpoint missing: {path}")
    sha = C.sha256_file(Path(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != C.SCHEMA_CHECKPOINT:
        raise RuntimeError(f"checkpoint schema {payload.get('schema')} != {C.SCHEMA_CHECKPOINT}")
    if payload.get("model") != C.MODEL_NAME:
        raise RuntimeError(f"checkpoint model {payload.get('model')} != {C.MODEL_NAME}")
    model, audit = M.build_dynamic_model(
        task.spec, task.norm, C.N_TASKS, seed=int(payload.get("seed", C.SEED)), device=device
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    audit = dict(audit)
    audit["checkpoint_sha256"] = sha
    audit["checkpoint_epoch"] = int(payload.get("epoch", -1))
    audit["checkpoint_global_step"] = int(payload.get("global_step", -1))
    return model, audit, sha


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict_with(
    model: M.W2DynamicModel,
    task: Any,
    rows: np.ndarray,
    device: torch.device,
    *,
    message: str = "real",
    shuffle_seed: int | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    rows = np.asarray(rows, dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(rows), int(batch_size)):
            chunk = rows[start : start + int(batch_size)]
            batch = _to_device(C.PC.collate(task.cache, chunk.tolist()), device)
            logits = model(batch, message=message, shuffle_seed=shuffle_seed)
            if int(logits.shape[1]) != C.N_TASKS:
                raise RuntimeError(f"model returned {logits.shape[1]} outputs")
            out.append(logits.detach().float().cpu().numpy())
    logits = np.concatenate(out, axis=0) if out else np.zeros((0, C.N_TASKS), dtype=np.float32)
    if logits.shape[0] != len(rows):
        raise RuntimeError(f"prediction rows {logits.shape[0]} != {len(rows)}")
    if not np.isfinite(logits).all():
        raise RuntimeError("non-finite logits in the mechanism pass")
    return logits


def report_from_logits(logits: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    probs = (1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))).astype(np.float32)
    rep = C.MET.validation_report(y, probs)
    rep["valid_bce"] = float(C.MET.masked_bce_numpy(logits, y))
    return rep


def logits_change(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = b - a
    rms_a = float(np.sqrt((a * a).mean()))
    rms_d = float(np.sqrt((diff * diff).mean()))
    per_row = np.abs(diff).max(axis=1)
    return {
        "rms_logits_reference": rms_a,
        "rms_logits_difference": rms_d,
        "relative_rms_change": (rms_d / rms_a) if rms_a else None,
        "mean_abs_change": float(np.abs(diff).mean()),
        "max_abs_change": float(np.abs(diff).max()),
        "fraction_molecules_changed": float((per_row > 1e-6).mean()),
    }


def full_validation_usage(model: M.W2DynamicModel, task: Any, device: torch.device) -> dict[str, Any]:
    """Global R1/R2/Rtotal over the whole official Validation split (eval mode)."""
    model.eval()
    rows = np.asarray(task.valid_rows, dtype=np.int64)
    acc = {"h0": 0.0, "h1": 0.0, "h2": 0.0, "d1": 0.0, "d2": 0.0, "dt": 0.0, "n": 0.0}
    finite = 0
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            chunk = rows[start : start + 256]
            batch = _to_device(C.PC.collate(task.cache, chunk.tolist()), device)
            _logits, aux = model(batch, return_aux=True)
            h0, h1, h2 = aux["h0"].double(), aux["h1"].double(), aux["h2"].double()
            acc["h0"] += float((h0 * h0).sum())
            acc["h1"] += float((h1 * h1).sum())
            acc["h2"] += float((h2 * h2).sum())
            acc["d1"] += float(((h1 - h0) ** 2).sum())
            acc["d2"] += float(((h2 - h1) ** 2).sum())
            acc["dt"] += float(((h2 - h0) ** 2).sum())
            acc["n"] += float(h0.numel())
            finite += int(torch.isfinite(h2).sum())
    rms = {k: math.sqrt(acc[k] / acc["n"]) for k in ("h0", "h1", "h2", "d1", "d2", "dt")}
    return {
        "rows": int(len(rows)),
        "atom_entries": int(acc["n"]),
        "rms_h0": rms["h0"],
        "rms_h1": rms["h1"],
        "rms_h2": rms["h2"],
        "rms_h1_minus_h0": rms["d1"],
        "rms_h2_minus_h1": rms["d2"],
        "rms_h2_minus_h0": rms["dt"],
        "R1": rms["d1"] / rms["h0"],
        "R2": rms["d2"] / rms["h1"],
        "Rtotal": rms["dt"] / rms["h0"],
        "finite_fraction": float(finite / acc["n"]),
        "mode": "eval (dropout identity), official Validation rows",
    }


def analyse_checkpoint(
    label: str,
    path: Path,
    task: Any,
    device: torch.device,
    *,
    recorded_valid_ap: float,
) -> dict[str, Any]:
    model, audit, sha = load_frozen(path, task, device)
    y = np.asarray(task.labels[task.valid_rows])
    t0 = time.perf_counter()
    base = predict_with(model, task, task.valid_rows, device, message="real")
    base_report = report_from_logits(base, y)
    replay_delta = abs(float(base_report["ap"]) - float(recorded_valid_ap))
    if replay_delta > REPLAY_TOL:
        raise RuntimeError(
            f"{label}: Validation replay {base_report['ap']} != recorded {recorded_valid_ap} "
            f"(delta {replay_delta})"
        )
    t_base = time.perf_counter() - t0

    variants: dict[str, Any] = {
        "real": {
            "valid_ap": float(base_report["ap"]),
            "valid_bce": float(base_report["valid_bce"]),
            "n_tasks_scored": int(base_report["n_tasks_scored"]),
            "per_task_ap": base_report["per_task_ap"],
            "seconds": float(t_base),
        }
    }
    for mode in ("message_off", "message_shuffle"):
        t1 = time.perf_counter()
        logits = predict_with(
            model,
            task,
            task.valid_rows,
            device,
            message="off" if mode == "message_off" else "shuffle",
            shuffle_seed=int(C.MECHANISM["shuffle_seed"]),
        )
        rep = report_from_logits(logits, y)
        variants[mode] = {
            "valid_ap": float(rep["ap"]),
            "valid_bce": float(rep["valid_bce"]),
            "valid_ap_delta_vs_real": float(rep["ap"]) - float(base_report["ap"]),
            "valid_bce_delta_vs_real": float(rep["valid_bce"]) - float(base_report["valid_bce"]),
            "n_tasks_scored": int(rep["n_tasks_scored"]),
            "per_task_ap": rep["per_task_ap"],
            "per_task_ap_delta_vs_real": [
                None if (a is None or b is None) else float(a) - float(b)
                for a, b in zip(rep["per_task_ap"], base_report["per_task_ap"])
            ],
            "logits_change_vs_real": logits_change(base, logits),
            "seconds": float(time.perf_counter() - t1),
        }
    usage = full_validation_usage(model, task, device)
    del model
    return {
        "label": label,
        "checkpoint": C.track_rel(path),
        "checkpoint_sha256": sha,
        "checkpoint_epoch": audit["checkpoint_epoch"],
        "checkpoint_global_step": audit["checkpoint_global_step"],
        "parameters": int(audit["parameters"]["total"]),
        "recorded_valid_ap": float(recorded_valid_ap),
        "replay_valid_ap": float(base_report["ap"]),
        "replay_abs_delta": float(replay_delta),
        "replay_within_tolerance": bool(replay_delta <= REPLAY_TOL),
        "valid_positive": base_report["positive"],
        "valid_observed": base_report["observed"],
        "variants": variants,
        "full_validation_dynamic_usage": usage,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(C.RUNS / "W2_DYNAMIC"))
    parser.add_argument("--out", default=str(C.ANALYSIS / "mechanism.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoints", default="best,epoch40")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    run_dir = Path(args.run_dir)
    run = C.read_json(run_dir / "run.json")
    if run.get("status") != "COMPLETED":
        raise RuntimeError(f"run not completed: {run.get('status')}")
    if run.get("test_evaluated"):
        raise RuntimeError("run claims a Test evaluation")
    task = C.SD.load_task()

    wanted: dict[str, Path] = {}
    recorded: dict[str, float] = {}
    for key in [x.strip() for x in str(args.checkpoints).split(",") if x.strip()]:
        if key == "best":
            wanted["W2_DYNAMIC_best_valid"] = C.REPO / run["best_checkpoint"]
            recorded["W2_DYNAMIC_best_valid"] = float(run["best_valid_ap"])
        elif key == "epoch40":
            wanted["W2_DYNAMIC_epoch40"] = C.REPO / run["primary_checkpoint"]
            report = run["fixed_epoch_reports"][str(run["final_epoch"])]
            recorded["W2_DYNAMIC_epoch40"] = float(report["ap"])
        else:
            raise KeyError(key)

    out: dict[str, Any] = {
        "schema": C.SCHEMA_MECHANISM,
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "device": str(device),
        "test_evaluated": False,
        "protocol": dict(C.MECHANISM),
        "split": "official Validation only (43,793 rows); no Test row is read or scored",
        "results": {},
        "interpretation_boundary": (
            "message_off / message_shuffle are post-training input interventions. They measure "
            "sensitivity to the real message correspondence, not a causal generalisation gain, and "
            "they are not equivalent to an independently trained capacity-matched STATIC control."
        ),
    }
    for label, path in wanted.items():
        C.log(f"[mechanism] {label} <- {path}")
        out["results"][label] = analyse_checkpoint(
            label, path, task, device, recorded_valid_ap=recorded[label]
        )
        r = out["results"][label]
        C.log(
            f"[mechanism] {label} real {r['variants']['real']['valid_ap']:.6f} "
            f"off {r['variants']['message_off']['valid_ap']:.6f} "
            f"shuffle {r['variants']['message_shuffle']['valid_ap']:.6f} "
            f"Rtotal {r['full_validation_dynamic_usage']['Rtotal']:.4f}"
        )
    out["verdict"] = (
        "ALL_CHECKPOINTS_REPLAYED_AND_INTERVENED"
        if all(v["replay_within_tolerance"] for v in out["results"].values())
        else "REPLAY_MISMATCH"
    )
    C.write_json(Path(args.out), out)
    C.log(f"[mechanism] wrote {args.out}")
    return 0 if out["verdict"] == "ALL_CHECKPOINTS_REPLAYED_AND_INTERVENED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
