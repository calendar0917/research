"""GPU smoke: mechanism, identity and determinism checks before the formal runs.

Cheap by design (~30 s of GPU): it exercises exactly the code paths the formal
phases use, on a small row subset, plus one full Validation-AP replay.

Checks
------
1. The cached ``rho_in`` equals an independent reconstruction and the frozen
   Validation AP reproduces ``AP0`` within ``1e-4`` on the full Validation split.
2. The Train-only standardiser is safe on a constant column (synthetic CPU case).
3. Both residual heads are exactly zero at initialisation and the initial logits
   are bitwise equal to the frozen logits (full-subset check).
4. Twenty head steps run twice from the same seed produce the identical loss
   sequence and the identical final head weights (determinism).
5. The frozen encoder's parameter fingerprint is unchanged and no gradient ever
   reaches it.
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

SMOKE_TRAIN_ROWS = 2048
SMOKE_STEPS = 20


def _train_steps(
    head: torch.nn.Module,
    x: torch.Tensor,
    frozen: torch.Tensor,
    y: torch.Tensor,
    *,
    steps: int,
) -> dict[str, Any]:
    params = list(head.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=float(C.HEAD_RECIPE["learning_rate"]),
        betas=tuple(float(v) for v in C.HEAD_RECIPE["adam_betas"]),
        eps=float(C.HEAD_RECIPE["adam_eps"]),
        weight_decay=float(C.HEAD_RECIPE["weight_decay"]),
    )
    n = int(x.shape[0])
    losses: list[float] = []
    head.train()
    for step in range(int(steps)):
        start = (step * C.BATCH_SIZE) % max(n - 1, 1)
        idx = torch.arange(start, min(start + C.BATCH_SIZE, n), device=x.device)
        logits = frozen[idx] + head(x[idx])
        loss = FM.masked_bce_torch(logits, y[idx])
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite smoke loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    return {
        "losses": losses,
        "loss_sha256": C.sha256_array(np.asarray(losses, dtype=np.float64)),
        "final_fingerprint": FM.head_parameter_fingerprint(head)["combined_sha256"],
    }


def run(*, device: torch.device, out_dir: Path = C.PREFLIGHT_RUN) -> dict[str, Any]:
    t_start = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inherited = C.assert_inherited_constants()
    task = DATA.load_task()
    model, model_audit = MODEL.load_frozen_model(task, device=device, label="frozen_preflight")
    params_before = MODEL.parameter_digests(model)
    rho_dim = int(model_audit["rho_dimension"]["rho_input_dim"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    # ---- 1. full Validation replay (environment + checkpoint + data) --------
    valid = MODEL.extract_split(
        model, task, task.valid_rows, device, split="valid", batch_size=512, rho_dim=rho_dim
    )
    replay = MODEL.replay_logits(model, task, task.valid_rows, device)
    extracted = valid["logits"].cpu().numpy()
    replay_diff = float(np.abs(replay.astype(np.float64) - extracted.astype(np.float64)).max())
    valid_ap = FM.official_ap(task.labels[task.valid_rows], BM.sigmoid(replay))["value"]
    valid_delta = abs(float(valid_ap) - float(C.AP0))
    extracted_ap = FM.official_ap(task.labels[task.valid_rows], BM.sigmoid(extracted))["value"]
    replay_ap_delta = abs(float(valid_ap) - float(extracted_ap))

    # ---- 2. small Train subset: identity + content checks -------------------
    train_rows = np.asarray(task.train_rows[:SMOKE_TRAIN_ROWS], dtype=np.int64)
    train = MODEL.extract_split(
        model, task, train_rows, device, split="train_smoke", batch_size=512, rho_dim=rho_dim
    )
    x = train["rho_in"]
    frozen = train["logits"]
    y = torch.from_numpy(np.ascontiguousarray(task.labels[train_rows])).to(device)

    # synthetic zero-variance standardiser case (device-independent, tiny)
    synthetic = torch.zeros((16, 3), dtype=torch.float32, device=device)
    synthetic[:, 0] = torch.arange(16, device=device)
    synthetic[:, 1] = 7.5  # constant column
    synthetic[:, 2] = torch.linspace(-1.0, 1.0, 16, device=device)
    syn_stats = FM.fit_standardizer(synthetic)
    zero_var_ok = (
        syn_stats["zero_variance_columns"] == [1]
        and float(syn_stats["scale"][1].item()) == 1.0
        and abs(float(syn_stats["mean"][1].item()) - 7.5) < 1e-6
    )

    head_audits: dict[str, Any] = {}
    determinism: dict[str, Any] = {}
    for arm in C.ARM_ORDER:
        # A fresh standardiser on the smoke subset keeps the smoke self-contained;
        # the formal transform is the full-Train fit audited in Phase A.
        stats = FM.fit_standardizer(x)
        x_std = x.clone()
        FM.standardize_(x_std, stats["mean"], stats["scale"])
        head = FM.build_head(arm, int(x_std.shape[1])).to(device)
        audit = FM.zero_init_audit(head, x_std, frozen)
        full_identity = False
        with torch.no_grad():
            full_logits = frozen + head(x_std)
            full_identity = bool(torch.equal(full_logits, frozen))
        audit["full_subset_identity"] = full_identity
        head_audits[arm] = audit
        if arm == "L":
            first = _train_steps(head, x_std, frozen, y, steps=SMOKE_STEPS)
            head2 = FM.build_head(arm, int(x_std.shape[1])).to(device)
            second = _train_steps(head2, x_std, frozen, y, steps=SMOKE_STEPS)
            determinism = {
                "steps": SMOKE_STEPS,
                "loss_sha256_first": first["loss_sha256"],
                "loss_sha256_second": second["loss_sha256"],
                "final_fingerprint_first": first["final_fingerprint"],
                "final_fingerprint_second": second["final_fingerprint"],
                "losses_identical": bool(first["loss_sha256"] == second["loss_sha256"]),
                "weights_identical": bool(first["final_fingerprint"] == second["final_fingerprint"]),
                "loss_first10": first["losses"][:10],
            }
        del head

    params_after = MODEL.parameter_digests(model)
    grads_present = any(p.grad is not None for p in model.parameters())
    gates = {
        "valid_ap_reproduces_ap0": bool(valid_delta <= C.VALID_AP_TOL),
        "extracted_ap_reproduces_ap0": bool(abs(float(extracted_ap) - float(C.AP0)) <= C.VALID_AP_TOL),
        "independent_replay_logits_within_calibrated_bound": bool(
            replay_diff <= C.REPLAY_LOGIT_TOL
        ),
        "independent_replay_valid_ap": bool(replay_ap_delta <= C.REPLAY_AP_TOL),
        "rho_in_reconstruction": bool(valid["audit"]["rho_reconstruction_gate"] and train["audit"]["rho_reconstruction_gate"]),
        "rho_replay": bool(valid["audit"]["rho_replay_gate"] and train["audit"]["rho_replay_gate"]),
        "logits_identity": bool(valid["audit"]["logits_identity_gate"] and train["audit"]["logits_identity_gate"]),
        "rho_dimension": bool(rho_dim == C.RHO_IN_DIM_EXPECTED),
        "standardizer_zero_variance": bool(zero_var_ok),
        "head_identity_both_arms": bool(
            all(head_audits[a]["residual_is_exactly_zero"] and head_audits[a]["full_subset_identity"] for a in C.ARM_ORDER)
        ),
        "head_training_deterministic": bool(
            determinism.get("losses_identical") and determinism.get("weights_identical")
        ),
        "frozen_parameters_unchanged": bool(
            params_after["parameter_fingerprint"] == params_before["parameter_fingerprint"]
        ),
        "no_gradients_on_frozen_parameters": bool(not grads_present),
        "test_evaluated_false": True,
    }
    gates = {k: bool(v) for k, v in gates.items()}
    total_seconds = float(time.perf_counter() - t_start)
    payload = {
        "schema": "lcrc-pcba-frozen-readout-preflight-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "all_passed": bool(all(gates.values())),
        "gates": gates,
        "failed_gates": [k for k, v in gates.items() if not v],
        "device": str(device),
        "environment": C.environment(),
        "gpu_occupancy": C.gpu_occupancy(),
        "inherited": inherited,
        "checkpoint_load": model_audit["checkpoint_load_audit"],
        "rho_dimension": model_audit["rho_dimension"],
        "valid_replay": {
            "rows": int(C.N_VALID),
            "official_ap": float(valid_ap),
            "ap0": float(C.AP0),
            "abs_delta": float(valid_delta),
            "tolerance": float(C.VALID_AP_TOL),
            "independent_replay_max_abs_logit_diff": replay_diff,
            "independent_replay_logit_tolerance": float(C.REPLAY_LOGIT_TOL),
            "extracted_official_ap": float(extracted_ap),
            "independent_replay_vs_extracted_ap_delta": float(replay_ap_delta),
            "independent_replay_ap_tolerance": float(C.REPLAY_AP_TOL),
            "extraction": valid["audit"],
        },
        "train_smoke_subset": {
            "rows": int(SMOKE_TRAIN_ROWS),
            "rows_sha256": train["audit"]["rows_sha256"],
            "extraction": train["audit"],
        },
        "standardizer_zero_variance_case": {
            "columns": syn_stats["zero_variance_columns"],
            "scale": syn_stats["scale"].tolist(),
            "mean": syn_stats["mean"].tolist(),
            "passed": bool(zero_var_ok),
        },
        "heads": head_audits,
        "determinism": determinism,
        "frozen_parameters": {
            "before": params_before,
            "after": params_after,
            "gradients_present": bool(grads_present),
        },
        "gpu_peak": C.gpu_peak(device),
        "gpu_seconds": total_seconds,
        "gpu_hours": float(total_seconds / 3600.0),
        "finished_utc": C.utcnow(),
        "test_evaluated": False,
    }
    C.write_json(out_dir / "preflight.json", payload)
    C.log(
        f"preflight all_passed={payload['all_passed']} validAP={valid_ap:.10f} "
        f"delta={valid_delta:.3e} deterministic={gates['head_training_deterministic']} "
        f"gpu_hours={payload['gpu_hours']:.6f}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=str(C.PREFLIGHT_RUN))
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.log("CUDA requested but unavailable")
        return 2
    try:
        payload = run(device=device, out_dir=Path(args.out_dir))
        return 0 if payload["all_passed"] else 3
    except Exception as exc:  # noqa: BLE001 - the failure is the deliverable
        C.write_json(
            Path(args.out_dir) / "preflight.json",
            {
                "schema": "lcrc-pcba-frozen-readout-preflight-v1",
                "round_id": C.ROUND_ID,
                "all_passed": False,
                "stop_reason": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
            },
        )
        C.log(f"preflight FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
