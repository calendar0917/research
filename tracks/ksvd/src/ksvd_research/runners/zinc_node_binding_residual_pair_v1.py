"""Runner: ZINC node-binding residual pair v1 (fresh Full, paired arms).

One frozen protocol, two arms differing only by the node-binding configuration
(``model.arm = control | residual``).  Both arms start from the same canonical
fresh Full initialisation and run the identical 240-epoch recipe; the fixed
residual coefficients are computed once from that shared initial state on a
fixed 1024-graph train sample.

``model.stage``:
  ``checks`` — CPU-friendly correctness checks (product equivalence, residual
               reduction, identical initial parameters, task-gradient reach);
  ``smoke``  — short GPU plumbing smoke (both arms, no valid / test);
  ``train``  — one formal 240-epoch arm trajectory.

Official ZINC **test** is never instantiated; a granted ``test_access`` is
refused.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

from ksvd_research.runtime.fingerprints import zinc_fingerprints
from ksvd_research.runtime.manifest import RunContext, RunResult
from ksvd_research.runtime.paths import REPO_ROOT, TRACK_ROOT, resolve_path

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.experiments.luyin16 import (  # noqa: E402
    zinc_node_binding_residual_pair_v1 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_node_binding_residual_pair_v1.yaml"

STAGES = ("checks", "smoke", "train")


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_node_binding_residual_pair_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Fresh Full paired node-binding comparison: product control vs fixed-amplitude "
            "additive residual; 240 epochs, fixed last-5 soup, train-median calibration; "
            "official test blocked"
        ),
        run_module=sys.modules[__name__],
    )


def fingerprints(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    data_root = resolve_path(config["data"]["root"])
    payload = zinc_fingerprints(data_root, expected_sizes=EXPECTED_SPLIT_SIZES)
    return payload, payload["split_fingerprint"]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _prepare(device, *, n_graphs: int):
    import torch

    from tracks.ksvd.experiments.luyin16 import zinc_upstream_portfolio_v1 as uprun

    stages.seed_everything(stages.SEED)
    subspace = uprun._load_parent_subspace()
    dictionary = uprun._dictionary_tensor()
    train_data = uprun.load_split("control", "train")
    control = stages.build_pair_model("control", dictionary, subspace)
    coeffs = stages.compute_residual_coefficients(
        control, train_data, device, n_graphs=int(n_graphs)
    )
    control_hash = stages.parameter_state_hash(control)
    del control
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dictionary, subspace, train_data, coeffs, control_hash


def _finite(value: Any) -> bool:
    import math

    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def run(config: Mapping[str, Any], context: RunContext) -> RunResult:
    if context.test_access == "granted":
        raise RuntimeError(
            "node-binding residual pair is non-terminal; it must never run with test access granted"
        )
    model_cfg = config.get("model", {})
    stage = str(model_cfg.get("stage", "checks"))
    if stage not in STAGES:
        raise ValueError(f"model.stage must be one of {STAGES}, got {stage!r}")
    runtime = config.get("runtime", {})
    device_name = str(runtime.get("device", "cpu"))
    print(f"[nodebind] stage={stage} device={device_name}", flush=True)

    if stage == "checks":
        checks = stages.run_checks(n_graphs=int(model_cfg.get("check_graphs", 32)))
        _write_json(context.artifact_dir / "checks.json", checks)
        metrics = {
            "measure": "node_binding_residual_checks",
            "stage": stage,
            "all_passed": bool(checks["all_passed"]),
            "official_valid_loaded": False,
            "official_test_loaded": False,
        }
        _write_json(context.artifact_dir / "metrics.json", metrics)
        print(f"[nodebind] checks all_passed={checks['all_passed']}", flush=True)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/checks.json"])

    from tracks.ksvd.experiments.luyin16 import zinc_upstream_portfolio_v1 as uprun

    device = uprun.resolve_device(device_name)
    if device.type == "cpu":
        import torch

        torch.set_num_threads(int(runtime.get("torch_threads", 8)))

    if stage == "smoke":
        import torch

        n_coeff = int(model_cfg.get("smoke_coeff_graphs", 512))
        n_train = int(model_cfg.get("smoke_train_graphs", 512))
        epochs = int(model_cfg.get("smoke_epochs", 2))
        dictionary, subspace, train_data, coeffs, control_hash = _prepare(device, n_graphs=n_coeff)
        smoke: dict[str, Any] = {"coeffs": coeffs.as_dict(), "arms": {}}
        for arm in stages.ARMS:
            stages.seed_everything(stages.SEED)
            model = stages.build_pair_model(
                arm,
                dictionary,
                subspace,
                eta_s=coeffs.eta_s if arm == "residual" else 0.0,
                eta_a=coeffs.eta_a if arm == "residual" else 0.0,
                gamma=coeffs.gamma if arm == "residual" else 1.0,
            )
            result = stages.train_arm(
                arm,
                model,
                list(train_data[:n_train]),
                None,
                device=device,
                out_dir=context.artifact_dir,
                epochs=epochs,
                diag_epochs=(0, epochs),
            )
            finite = all(_finite(row["train_mae"]) for row in result["curve"]) and all(
                bool(torch.isfinite(p).all()) for p in model.parameters()
            )
            smoke["arms"][arm] = {
                "seconds_per_epoch": float(result["seconds_per_epoch"]),
                "final_train_mae": float(result["curve"][-1]["train_mae"]),
                "finite": bool(finite),
                "peak_gpu_memory_mb": float(result["peak_gpu_memory_mb"]),
                "initial_state_sha256": control_hash,
            }
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        smoke["all_finite"] = bool(all(row["finite"] for row in smoke["arms"].values()))
        smoke["device"] = str(device)
        smoke["official_test_loaded"] = False
        _write_json(context.artifact_dir / "smoke.json", smoke)
        _write_json(stages.EXPORT_DIR / "smoke" / "smoke.json", smoke)
        metrics = {
            "measure": "node_binding_residual_smoke",
            "stage": stage,
            "device": str(device),
            "all_finite": bool(smoke["all_finite"]),
            "seconds_per_epoch_control": float(smoke["arms"]["control"]["seconds_per_epoch"]),
            "seconds_per_epoch_residual": float(smoke["arms"]["residual"]["seconds_per_epoch"]),
            "official_valid_loaded": False,
            "official_test_loaded": False,
        }
        _write_json(context.artifact_dir / "metrics.json", metrics)
        print(f"[nodebind] smoke all_finite={smoke['all_finite']} device={device}", flush=True)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/smoke.json"])

    # stage == train
    import torch

    arm = str(model_cfg.get("arm", "control"))
    if arm not in stages.ARMS:
        raise ValueError(f"model.arm must be one of {stages.ARMS}, got {arm!r}")
    epochs = int(model_cfg.get("epochs", stages.EPOCHS))
    if epochs != stages.EPOCHS:
        raise ValueError(f"the epoch budget is frozen at {stages.EPOCHS}, got {epochs!r}")
    lr = float(model_cfg.get("lr", stages.LR))
    n_coeff = int(model_cfg.get("coeff_graphs", stages.COEFF_GRAPHS))
    dictionary, subspace, train_data, coeffs, control_hash = _prepare(device, n_graphs=n_coeff)
    valid_data = uprun.load_split("control", "valid")

    stages.seed_everything(stages.SEED)
    model = stages.build_pair_model(
        arm,
        dictionary,
        subspace,
        eta_s=coeffs.eta_s if arm == "residual" else 0.0,
        eta_a=coeffs.eta_a if arm == "residual" else 0.0,
        gamma=coeffs.gamma if arm == "residual" else 1.0,
    )
    arm_hash = stages.parameter_state_hash(model)
    if arm_hash != control_hash:
        raise RuntimeError(
            f"INIT_INVALID: {arm} initial parameter state differs from the canonical shared init"
        )
    print(
        f"[nodebind] arm={arm} init_sha256={arm_hash[:16]} eta_s={coeffs.eta_s:.6f} "
        f"eta_a={coeffs.eta_a:.6f} gamma={coeffs.gamma:.6f}",
        flush=True,
    )

    export_dir = stages.EXPORT_DIR / arm
    export_dir.mkdir(parents=True, exist_ok=True)
    out_dir = context.artifact_dir
    result = stages.train_arm(
        arm,
        model,
        train_data,
        valid_data,
        device=device,
        out_dir=out_dir,
        epochs=epochs,
        lr=lr,
        coeffs=coeffs if arm == "residual" else None,
    )

    stages.write_curve(out_dir, result["curve"])
    stages.write_curve(export_dir, result["curve"])
    _write_json(out_dir / "health.json", {"arm": arm, "health": result["health"]})
    _write_json(export_dir / "health.json", {"arm": arm, "health": result["health"]})
    _write_json(out_dir / "coefficients.json", coeffs.as_dict())
    _write_json(export_dir / "coefficients.json", coeffs.as_dict())
    stages.write_valid_predictions(out_dir, result)
    stages.write_valid_predictions(export_dir, result)

    import torch as _torch

    soup_path = export_dir / "soup_state.pt"
    _torch.save(result["soup_state"], soup_path)
    _torch.save(result["soup_state"], out_dir / "soup_state.pt")
    _write_json(
        export_dir / "soup.json",
        {
            "arm": arm,
            "members": result["members"],
            "member_valid_mae": result.get("member_valid_mae", []),
            "soup_epochs": list(stages.SOUP_EPOCHS),
            "official_test_loaded": False,
        },
    )

    summary = {
        "protocol_version": stages.PROTOCOL_VERSION,
        "arm": arm,
        "device": str(device),
        "epochs": int(result["epochs"]),
        "lr": float(result["lr"]),
        "weight_decay": float(result["weight_decay"]),
        "batch_size": int(result["batch_size"]),
        "grad_clip": float(result["grad_clip"]),
        "members": result["members"],
        "member_valid_mae": result.get("member_valid_mae", []),
        "raw_valid_mae": result.get("raw_valid_mae"),
        "calibrated_valid_mae": result.get("calibrated_valid_mae"),
        "raw_train_mae": result.get("raw_train_mae"),
        "calibrated_train_mae": result.get("calibrated_train_mae"),
        "train_fitted_bias": result.get("train_fitted_bias"),
        "eval_gap": result.get("eval_gap"),
        "seconds_per_epoch": float(result["seconds_per_epoch"]),
        "wall_clock_s": float(result["wall_clock_s"]),
        "peak_gpu_memory_mb": float(result["peak_gpu_memory_mb"]),
        "init_state_sha256": arm_hash,
        "coefficients": coeffs.as_dict(),
        "official_valid_loaded": True,
        "official_test_loaded": False,
    }
    _write_json(export_dir / "summary.json", summary)
    _write_json(out_dir / "summary.json", summary)

    metrics = {
        "measure": "calibrated_valid_mae",
        "stage": stage,
        "arm": arm,
        "device": str(device),
        "epochs": int(result["epochs"]),
        "raw_valid_mae": float(result.get("raw_valid_mae", float("nan"))),
        "calibrated_valid_mae": float(result.get("calibrated_valid_mae", float("nan"))),
        "train_fitted_bias": float(result.get("train_fitted_bias", float("nan"))),
        "eval_gap": float(result.get("eval_gap", float("nan"))),
        "seconds_per_epoch": float(result["seconds_per_epoch"]),
        "wall_clock_s": float(result["wall_clock_s"]),
        "peak_gpu_memory_mb": float(result["peak_gpu_memory_mb"]),
        "soup_members": result["members"],
        "official_valid_loaded": True,
        "official_test_loaded": False,
    }
    _write_json(context.artifact_dir / "metrics.json", metrics)
    print(
        f"[nodebind] done arm={arm} raw={metrics['raw_valid_mae']:.6f} "
        f"cal={metrics['calibrated_valid_mae']:.6f} wall={metrics['wall_clock_s']:.0f}s",
        flush=True,
    )
    return RunResult(
        metrics=metrics,
        status="completed",
        artifacts=["artifacts/summary.json", "artifacts/curve.csv", "artifacts/health.json"],
    )


__all__ = ["build_runner", "fingerprints", "run", "stages"]