"""Replay-nondeterminism diagnostic for the frozen encoder.

Question: the frozen LCRC-C1 forward contains ``index_add_`` reductions on CUDA
(``mean_h`` / ``mean_h2``), which PyTorch implements with atomics.  If the same
input produces slightly different logits on a second call, then an
"independent replay == extraction" check must be calibrated against that spread
instead of pretending bitwise equality is available.

This script measures the spread on a small Validation subset:

* extraction with batch 512, twice          -> run-to-run spread
* ``pc_model.predict_logits`` batch 512, twice -> independent path, same batching
* ``pc_model.predict_logits`` batch 256     -> batch-size sensitivity

and per-block ``rho_in`` differences.  It changes no model code and no gate.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    os.environ.setdefault("OMP_NUM_THREADS", "8")


_configure_imports()

import torch  # noqa: E402

import fr_common as C  # noqa: E402
import fr_metrics as FM  # noqa: E402
import fr_model as MODEL  # noqa: E402
import lr_data as DATA  # noqa: E402
import pc_model as BM  # noqa: E402

N_ROWS = 8192


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def _ap(task: Any, rows: np.ndarray, logits: np.ndarray) -> float:
    probs = BM.sigmoid(logits)
    return float(FM.official_ap(task.labels[rows], probs)["value"])


def run(*, device: torch.device, out_dir: Path = C.RUNS / "replay_diag") -> dict[str, Any]:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    task = DATA.load_task()
    model, model_audit = MODEL.load_frozen_model(task, device=device, label="replay_diag")
    rows = np.asarray(task.valid_rows[:N_ROWS], dtype=np.int64)
    rho_dim = int(model_audit["rho_dimension"]["rho_input_dim"])

    a = MODEL.extract_split(model, task, rows, device, split="diag_a", batch_size=512, rho_dim=rho_dim)
    b = MODEL.extract_split(model, task, rows, device, split="diag_b", batch_size=512, rho_dim=rho_dim)
    p512_1 = MODEL.replay_logits(model, task, rows, device, batch_size=512)
    p512_2 = MODEL.replay_logits(model, task, rows, device, batch_size=512)
    p256 = MODEL.replay_logits(model, task, rows, device, batch_size=256)
    # The head cache always holds these frozen logits; two extractions of the same
    # rows must therefore agree closely for the round's pairing argument to hold.
    logits_a = a["logits"].cpu().numpy()
    logits_b = b["logits"].cpu().numpy()
    rho_a = a["rho_in"].cpu().numpy()
    rho_b = b["rho_in"].cpu().numpy()
    layout = model_audit["rho_dimension"]["layout"]
    block_diffs = {}
    for name, (lo, hi) in layout.items():
        block_diffs[name] = _max_abs(rho_a[:, lo:hi], rho_b[:, lo:hi])
    ap_a = _ap(task, rows, logits_a)
    ap_b = _ap(task, rows, logits_b)

    payload = {
        "schema": "lcrc-pcba-frozen-readout-replay-diag-v1",
        "round_id": C.ROUND_ID,
        "phase": "diagnostic_replay_nondeterminism",
        "rows": int(rows.size),
        "rows_sha256": C.sha256_int64(rows),
        "device": str(device),
        "environment": C.environment(),
        "torch_state": {
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "model_is_eval": not bool(model.training),
        "parameters_frozen": not any(p.requires_grad for p in model.parameters()),
        "extraction_vs_extraction": {
            "logits_max_abs_diff": _max_abs(logits_a, logits_b),
            "logits_sha256_equal": bool(a["audit"]["logits_sha256"] == b["audit"]["logits_sha256"]),
            "rho_in_max_abs_diff": _max_abs(rho_a, rho_b),
            "rho_in_block_max_abs_diff": block_diffs,
            "official_ap_first": ap_a,
            "official_ap_second": ap_b,
            "ap_abs_diff": abs(ap_a - ap_b),
        },
        "predict_logits_vs_predict_logits_bs512": {
            "logits_max_abs_diff": _max_abs(p512_1, p512_2),
            "official_ap_first": _ap(task, rows, p512_1),
            "official_ap_second": _ap(task, rows, p512_2),
        },
        "extraction_vs_predict_logits_bs512": {
            "logits_max_abs_diff": _max_abs(logits_a, p512_1),
            "ap_abs_diff": abs(ap_a - _ap(task, rows, p512_1)),
        },
        "extraction_vs_predict_logits_bs256": {
            "logits_max_abs_diff": _max_abs(logits_a, p256),
            "ap_abs_diff": abs(ap_a - _ap(task, rows, p256)),
        },
        "per_batch_reconstruction_max_abs_diff": {
            "first_extraction": a["audit"]["max_abs_rho_reconstruction_diff"],
            "second_extraction": b["audit"]["max_abs_rho_reconstruction_diff"],
        },
        "interpretation": (
            "index_add_ reductions in the frozen forward use CUDA atomics, so the "
            "encoder is not bitwise reproducible across calls; logits can differ in "
            "the last fp32 bits while the AP is stable. This spread bounds the "
            "independent-replay gate, it does not change the model or the cache."
        ),
        "gpu_peak": C.gpu_peak(device),
        "gpu_seconds": float(time.perf_counter() - t0),
        "gpu_hours": float((time.perf_counter() - t0) / 3600.0),
        "finished_utc": C.utcnow(),
        "test_evaluated": False,
    }
    C.write_json(out_dir / "replay_diag.json", payload)
    C.log(
        "replay diag: extract-vs-extract logits {:.3e} rho_in {:.3e} AP {:.3e}; "
        "extract-vs-predict(512) logits {:.3e}; extract-vs-predict(256) logits {:.3e}".format(
            payload["extraction_vs_extraction"]["logits_max_abs_diff"],
            payload["extraction_vs_extraction"]["rho_in_max_abs_diff"],
            payload["extraction_vs_extraction"]["ap_abs_diff"],
            payload["extraction_vs_predict_logits_bs512"]["logits_max_abs_diff"],
            payload["extraction_vs_predict_logits_bs256"]["logits_max_abs_diff"],
        )
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=str(C.RUNS / "replay_diag"))
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.log("CUDA requested but unavailable")
        return 2
    try:
        run(device=device, out_dir=Path(args.out_dir))
        return 0
    except Exception as exc:  # noqa: BLE001 - the failure is the deliverable
        C.write_json(
            Path(args.out_dir) / "replay_diag.json",
            {
                "schema": "lcrc-pcba-frozen-readout-replay-diag-v1",
                "round_id": C.ROUND_ID,
                "all_passed": False,
                "stop_reason": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
            },
        )
        C.log(f"replay diag FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
