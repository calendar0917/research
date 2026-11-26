"""Optional frozen-weight intervention on the historical FIXED1 best checkpoint.

No weight is changed. The SAME FIXED1 state is loaded into (a) the original FIXED1
computation and (b) the MEANFIELD computation (identical state-dict keys/shapes),
and both are scored on the **complete** official Valid split. This separates "what
the intervention does to a trained model" from "what a model trained under the
intervention reaches" (the independent 12-epoch run, reported separately).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import mf_common as R  # noqa: E402
import mf_model as MF  # noqa: E402
import ds_common as C  # noqa: E402
import ds_model as DM  # noqa: E402

DEFAULT_CHECKPOINT = R.FIXED1_ROUND / "runs/FIXED1_seed0/checkpoint_best.pt"


def rms(x: torch.Tensor) -> float:
    return float(x.detach().double().pow(2).mean().sqrt().item()) if x.numel() else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="0 = the complete Valid split")
    parser.add_argument("--out", default=str(R.ANALYSIS / "frozen_intervention.json"))
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        payload = {
            "schema": "fsdp-frozen-intervention-v1",
            "round_id": R.ROUND_ID,
            "status": "SKIPPED",
            "reason": f"checkpoint not available: {ckpt_path}",
        }
        C.write_json(Path(args.out), payload)
        return 0

    task = C.SD.load_task()
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]

    fixed1, _a1 = DM.build_ds_model(
        task.spec, task.norm, C.N_TASKS, arm="FIXED1", seed=int(payload.get("seed", 0)), device=device
    )
    fixed1.load_state_dict(state, strict=True)
    meanfield, _a2 = MF.build_mf_model(
        task.spec, task.norm, C.N_TASKS, seed=int(payload.get("seed", 0)), device=device
    )
    load = meanfield.load_state_dict(state, strict=True)

    rows = np.asarray(task.valid_rows, dtype=np.int64)
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]
    y = np.asarray(task.labels[rows])

    results: dict[str, dict] = {}
    logits: dict[str, np.ndarray] = {}
    for name, model in (("FIXED1", fixed1), ("MEANFIELD", meanfield)):
        model.eval()
        model.to(device)
        rep = C.evaluate_split(
            model, task, device, "valid", y=y, per_task=True, keep_logits=True,
            batch_size=int(args.batch_size), reason=f"frozen_{name}_valid",
        )
        logits[name] = rep["logits"]
        results[name] = {
            "valid_ap": float(rep["ap"]),
            "valid_bce": float(rep["bce"]),
            "n_tasks_scored": int(rep["n_tasks_scored"]),
            "per_task_ap": rep["per_task_ap"],
        }
        C.log(f"frozen {name}: valid AP {rep['ap']:.9f} BCE {rep['bce']:.9f}")

    diff = logits["MEANFIELD"] - logits["FIXED1"]
    mf_ap = results["MEANFIELD"]["per_task_ap"]
    f1_ap = results["FIXED1"]["per_task_ap"]
    per_task_delta = np.asarray(
        [
            (np.nan if (a is None or b is None) else float(a) - float(b))
            for a, b in zip(mf_ap, f1_ap)
        ],
        dtype=np.float64,
    )
    out = {
        "schema": "fsdp-frozen-intervention-v1",
        "round_id": R.ROUND_ID,
        "status": "COMPLETED",
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_valid_ap_recorded": payload.get("valid_ap"),
        "weights_changed": False,
        "state_dict_strict_load": {"unexpected_keys": load.unexpected_keys, "missing_keys": load.missing_keys},
        "valid_rows": int(rows.size),
        "scope": "complete official Valid split" if int(args.limit) == 0 else f"first {int(args.limit)} Valid rows",
        "baseline_fixed1": {k: results["FIXED1"][k] for k in ("valid_ap", "valid_bce", "n_tasks_scored")},
        "intervention_meanfield": {k: results["MEANFIELD"][k] for k in ("valid_ap", "valid_bce", "n_tasks_scored")},
        "delta_valid_ap": results["MEANFIELD"]["valid_ap"] - results["FIXED1"]["valid_ap"],
        "delta_valid_bce": results["MEANFIELD"]["valid_bce"] - results["FIXED1"]["valid_bce"],
        "logit_change_rms": rms(torch.from_numpy(diff)),
        "logit_rms_reference": rms(torch.from_numpy(logits["FIXED1"])),
        "logit_relative_change_rms": (
            rms(torch.from_numpy(diff)) / rms(torch.from_numpy(logits["FIXED1"]))
            if rms(torch.from_numpy(logits["FIXED1"]))
            else None
        ),
        "n_tasks_scored": int(np.isfinite(per_task_delta).sum()),
        "n_tasks_improved": int((per_task_delta > 0).sum()),
        "n_tasks_degraded": int((per_task_delta < 0).sum()),
        "mean_per_task_delta": float(np.nanmean(per_task_delta)),
        "per_task_delta": {
            str(t): (None if d != d else float(d)) for t, d in enumerate(per_task_delta)
        },
        "note": (
            "frozen-weight intervention: identical FIXED1 weights, only the message aggregation rule changes; "
            "reported separately from the independent MEANFIELD training run"
        ),
    }
    C.write_json(Path(args.out), out)
    C.log(
        f"frozen intervention: FIXED1 {out['baseline_fixed1']['valid_ap']:.9f} -> "
        f"MEANFIELD {out['intervention_meanfield']['valid_ap']:.9f} "
        f"(delta {out['delta_valid_ap']:+.6f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
