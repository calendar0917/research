"""Runner: ZINC overnight bottleneck v1 (node-collapse 2x2 + additive topology).

``model.stage``:

* ``checks``       CPU correctness checks (product equivalence, unit-scale
                   ratio, additive-reader structure/gradients, zero-mask).
* ``scale``        freeze the initial node-slot scale ``kappa`` from the
                   canonical fresh seed-0 initialisation (train only).
* ``smoke``        short GPU plumbing smoke (N1 + T0, no valid / test).
* ``train_prefix`` one 80-epoch node arm, checkpointed; no official-valid read.
* ``train``        one 240-epoch arm (fresh, or resumed from the seed-0
                   epoch-80 checkpoint of the same arm).

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
    zinc_overnight_bottleneck_v1 as stages,
)

EXPECTED_SPLIT_SIZES = {"train": 10_000, "val": 1_000, "test": 1_000}
DEFAULT_CONFIG = TRACK_ROOT / "configs/luyin16/zinc_overnight_bottleneck_v1.yaml"

STAGES = ("checks", "scale", "smoke", "train_prefix", "train")

#: frozen arm table: scale factor A, node weight decay factor B, readout.
ARM_SPEC: dict[str, dict[str, Any]] = {
    "N0": {"scale": False, "node_wd": None, "additive": False},
    "N1": {"scale": True, "node_wd": None, "additive": False},
    "N2": {"scale": False, "node_wd": 0.0, "additive": False},
    "N3": {"scale": True, "node_wd": 0.0, "additive": False},
    "T0": {"scale": False, "node_wd": None, "additive": True},
}


def build_runner():
    from ksvd_research.runner_api import Runner

    return Runner(
        name="zinc_overnight_bottleneck_v1",
        default_config=DEFAULT_CONFIG,
        default_study="zinc-context-gap",
        description=(
            "Fresh-Full node-collapse 2x2 (initial slot scale x node weight decay) and one "
            "additive-topology-readout arm; 80-epoch prefix + 240-epoch full; fixed last-5 soup; "
            "train-median calibration; official test blocked"
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


def _finite(value: Any) -> bool:
    import math

    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _resolve_arm_settings(model_cfg: Mapping[str, Any]) -> tuple[str, float, float, bool]:
    import torch  # noqa: F401

    arm = str(model_cfg.get("arm", "N0"))
    if arm not in ARM_SPEC:
        raise ValueError(f"model.arm must be one of {sorted(ARM_SPEC)}, got {arm!r}")
    spec = ARM_SPEC[arm]
    node_wd = spec["node_wd"]
    if model_cfg.get("node_weight_decay") is not None:
        node_wd = float(model_cfg["node_weight_decay"])
    if node_wd is None:
        node_wd = float(stages.WEIGHT_DECAY)
    kappa_cfg = model_cfg.get("kappa")
    kappa = float(kappa_cfg) if kappa_cfg is not None else float("nan")
    return arm, float(node_wd), kappa, bool(spec["additive"])


def _prepare(device, *, n_graphs: int, train_seed: int, scale_seed: int):
    import torch

    from tracks.ksvd.experiments.luyin16 import zinc_upstream_portfolio_v1 as uprun

    stages.seed_everything(int(train_seed))
    subspace = uprun._load_parent_subspace()
    dictionary = uprun._dictionary_tensor()
    train_data = uprun.load_split("control", "train")
    canonical = stages.build_readout_model(dictionary, subspace, seed=int(train_seed), scale_seed=int(scale_seed))
    # The frozen scale rule is computed on CPU so that kappa is device-independent
    # and reproducible across the CPU audit / GPU training regimes.
    manifest = stages.compute_node_scale(canonical, train_data, torch.device("cpu"), n_graphs=int(n_graphs), seed=int(train_seed))
    canonical_hash = stages.parameter_state_hash(canonical)
    del canonical
    return dictionary, subspace, train_data, manifest, canonical_hash


def run(config: Mapping[str, Any], context: RunContext) -> RunResult:
    if context.test_access == "granted":
        raise RuntimeError("zinc_overnight_bottleneck_v1 is non-terminal; it must never run with test access granted")
    model_cfg = config.get("model", {})
    stage = str(model_cfg.get("stage", "checks"))
    if stage not in STAGES:
        raise ValueError(f"model.stage must be one of {STAGES}, got {stage!r}")
    runtime = config.get("runtime", {})
    device_name = str(runtime.get("device", "cpu"))
    train_seed = int(model_cfg.get("train_seed", stages.SEED))
    scale_seed = int(model_cfg.get("scale_seed", stages.SCALE_SEED))
    print(f"[overnight] stage={stage} device={device_name} train_seed={train_seed}", flush=True)

    if stage == "checks":
        checks = stages.run_checks(n_graphs=int(model_cfg.get("check_graphs", 32)))
        _write_json(context.artifact_dir / "checks.json", checks)
        metrics = {
            "measure": "overnight_bottleneck_checks",
            "stage": stage,
            "all_passed": bool(checks["all_passed"]),
            "official_valid_loaded": False,
            "official_test_loaded": False,
        }
        _write_json(context.artifact_dir / "metrics.json", metrics)
        print(f"[overnight] checks all_passed={checks['all_passed']}", flush=True)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/checks.json"])

    from tracks.ksvd.experiments.luyin16 import zinc_upstream_portfolio_v1 as uprun

    device = uprun.resolve_device(device_name)
    if device.type == "cpu":
        import torch

        torch.set_num_threads(int(runtime.get("torch_threads", 8)))

    if stage == "scale":
        dictionary, subspace, train_data, manifest, canonical_hash = _prepare(
            device, n_graphs=int(model_cfg.get("scale_graphs", stages.SCALE_GRAPHS)), train_seed=train_seed, scale_seed=scale_seed
        )
        payload = {"stage": stage, "device": str(device), "canonical_init_sha256": canonical_hash, "manifest": manifest.as_dict()}
        if train_seed == stages.SEED:
            frozen = stages.FROZEN_SCALE_PATH
            if frozen.exists():
                previous = json.loads(frozen.read_text(encoding="utf-8"))
                if abs(float(previous["kappa"]) - manifest.kappa) > 1e-5 * abs(float(previous["kappa"])):
                    raise RuntimeError(f"INIT_INVALID: recomputed kappa {manifest.kappa} != frozen {previous['kappa']}")
            else:
                _write_json(frozen, manifest.as_dict())
            payload["frozen_matches"] = True
        _write_json(context.artifact_dir / "scale.json", payload)
        metrics = {
            "measure": "node_scale_manifest",
            "stage": stage,
            "kappa": float(manifest.kappa),
            "r_init_slot_rms": float(manifest.r_init_slot_rms),
            "official_valid_loaded": False,
            "official_test_loaded": False,
        }
        _write_json(context.artifact_dir / "metrics.json", metrics)
        print(f"[overnight] kappa={manifest.kappa:.6e} r_init={manifest.r_init_slot_rms:.6e}", flush=True)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/scale.json"])

    if stage == "smoke":
        import torch

        dictionary, subspace, train_data, manifest, canonical_hash = _prepare(
            device, n_graphs=int(model_cfg.get("scale_graphs", 128)), train_seed=train_seed, scale_seed=scale_seed
        )
        smoke: dict[str, Any] = {"kappa": manifest.kappa, "arms": {}, "official_test_loaded": False}
        subset = list(train_data[: int(model_cfg.get("smoke_train_graphs", 512))])
        for arm in ("N1", "T0"):
            _arm, node_wd, _, additive = _resolve_arm_settings({"arm": arm})
            stages.seed_everything(train_seed)
            if additive:
                model = stages.build_readout_model(dictionary, subspace, seed=train_seed, scale_seed=scale_seed, additive_topology=True)
            else:
                model = stages.build_node_model(arm, dictionary, subspace, kappa=manifest.kappa, seed=train_seed, scale_seed=scale_seed)
            result = stages.train_arm(
                arm,
                model,
                subset,
                None,
                device=device,
                out_dir=context.artifact_dir,
                epochs=int(model_cfg.get("smoke_epochs", 2)),
                lr=float(model_cfg.get("lr", stages.LR)),
                node_weight_decay=node_wd,
                kappa=manifest.kappa,
                diag_epochs=(0, 1, 2),
                train_seed=train_seed,
            )
            smoke["arms"][arm] = {
                "seconds_per_epoch": float(result["seconds_per_epoch"]),
                "final_train_mae": float(result["curve"][-1]["train_mae"]),
                "finite": bool(_finite(result["curve"][-1]["train_mae"]) and all(bool(torch.isfinite(p).all()) for p in model.parameters())),
                "peak_gpu_memory_mb": float(result["peak_gpu_memory_mb"]),
                "init_state_sha256": stages.parameter_state_hash(model),
            }
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        smoke["all_finite"] = bool(all(row["finite"] for row in smoke["arms"].values()))
        smoke["device"] = str(device)
        _write_json(context.artifact_dir / "smoke.json", smoke)
        metrics = {
            "measure": "overnight_bottleneck_smoke",
            "stage": stage,
            "device": str(device),
            "all_finite": bool(smoke["all_finite"]),
            "official_valid_loaded": False,
            "official_test_loaded": False,
        }
        _write_json(context.artifact_dir / "metrics.json", metrics)
        print(f"[overnight] smoke all_finite={smoke['all_finite']} device={device}", flush=True)
        return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/smoke.json"])

    # train_prefix / train
    import torch

    arm, node_wd, kappa_cfg, additive = _resolve_arm_settings(model_cfg)
    if stage == "train_prefix" and arm not in stages.NODE_ARMS:
        raise ValueError("train_prefix is only for the node arms N0..N3")
    dictionary, subspace, train_data, manifest, canonical_hash = _prepare(
        device, n_graphs=int(model_cfg.get("scale_graphs", stages.SCALE_GRAPHS)), train_seed=train_seed, scale_seed=scale_seed
    )
    if (arm in ("N1", "N3")) or (not _finite(kappa_cfg)):
        kappa = float(manifest.kappa)
    else:
        kappa = float(kappa_cfg)
    frozen_kappa = None
    if arm in ("N1", "N3") and train_seed == stages.SEED:
        frozen = stages.FROZEN_SCALE_PATH
        if frozen.exists():
            previous = json.loads(frozen.read_text(encoding="utf-8"))
            frozen_kappa = float(previous["kappa"])
            if abs(frozen_kappa - kappa) > 1e-5 * abs(frozen_kappa):
                raise RuntimeError(f"INIT_INVALID: arm kappa {kappa} != frozen {frozen_kappa}")
            # use the committed frozen value so seed-0 kappa is identical on every regime
            kappa = frozen_kappa
            print(f"[overnight] using frozen kappa={kappa:.10f} (recomputed {manifest.kappa:.10f})", flush=True)
    train_data_local = train_data
    resume_from = model_cfg.get("resume_from")
    valid_data = uprun.load_split("control", "valid") if stage == "train" else None

    if additive:
        stages.seed_everything(train_seed)
        model = stages.build_readout_model(dictionary, subspace, seed=train_seed, scale_seed=scale_seed, additive_topology=True, head_seed=int(model_cfg.get("head_seed", 0)))
    else:
        stages.seed_everything(train_seed)
        model = stages.build_node_model(arm, dictionary, subspace, kappa=kappa, seed=train_seed, scale_seed=scale_seed)
    arm_hash = stages.parameter_state_hash(model)
    if additive:
        if arm_hash == canonical_hash:
            raise RuntimeError("INIT_INVALID: additive reader did not change the parameter state")
    elif arm_hash != canonical_hash:
        raise RuntimeError(f"INIT_INVALID: {arm} initial parameter state differs from the canonical shared init")
    print(f"[overnight] arm={arm} init_sha256={arm_hash[:16]} node_wd={node_wd:.1e} kappa={kappa:.6e}", flush=True)

    tag = f"{arm}_s{train_seed}" + ("_prefix" if stage == "train_prefix" else "")
    export_dir = stages.EXPORT_DIR / tag
    export_dir.mkdir(parents=True, exist_ok=True)
    out_dir = context.artifact_dir
    checkpoint_path = export_dir / "checkpoint.pt"

    resume_state = None
    if resume_from:
        resume_path = Path(str(resume_from))
        if not resume_path.is_absolute():
            resume_path = REPO_ROOT / resume_path
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint unavailable: {resume_path}")
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if str(resume_state.get("arm")) != arm:
            raise ValueError(f"resume checkpoint arm {resume_state.get('arm')!r} != {arm!r}")
        if int(resume_state.get("epoch", 0)) <= 0:
            raise ValueError("resume checkpoint must have a positive epoch")
        kappa = float(resume_state["kappa"])
        print(f"[overnight] resuming {arm} from {resume_path} epoch={resume_state['epoch']} kappa={kappa:.6e}", flush=True)

    prefix_epochs = int(model_cfg.get("prefix_epochs", stages.PREFIX_EPOCHS))
    if stage == "train_prefix" and prefix_epochs != stages.PREFIX_EPOCHS and not model_cfg.get("allow_short_prefix"):
        raise ValueError(f"the prefix budget is frozen at {stages.PREFIX_EPOCHS} epochs, got {prefix_epochs!r}")
    epochs = int(prefix_epochs if stage == "train_prefix" else model_cfg.get("epochs", stages.EPOCHS))
    if stage == "train" and epochs != stages.EPOCHS:
        raise ValueError(f"the full epoch budget is frozen at {stages.EPOCHS}, got {epochs!r}")
    result = stages.train_arm(
        arm,
        model,
        train_data_local,
        valid_data,
        device=device,
        out_dir=out_dir,
        epochs=epochs,
        lr=float(model_cfg.get("lr", stages.LR)),
        node_weight_decay=node_wd,
        kappa=kappa,
        diag_epochs=tuple(stages.DIAG_EPOCHS if stage == "train" else (0, 1, 10, 20, 40, 80)),
        resume_state=resume_state,
        checkpoint_path=checkpoint_path,
        train_seed=train_seed,
    )

    stages._export_curve(out_dir, result["curve"])
    stages._export_curve(export_dir, result["curve"])
    _write_json(out_dir / "health.json", {"arm": arm, "health": result["health"]})
    _write_json(export_dir / "health.json", {"arm": arm, "health": result["health"]})
    _write_json(out_dir / "scale_manifest.json", manifest.as_dict())
    _write_json(export_dir / "scale_manifest.json", manifest.as_dict())
    _write_json(out_dir / "parameter_groups.json", stages.parameter_group_inventory(model))

    torch.save(result["soup_state"], export_dir / "soup_state.pt")
    torch.save(result["soup_state"], out_dir / "soup_state.pt")
    _write_json(
        export_dir / "soup.json",
        {"arm": arm, "members": result["members"], "member_valid_mae": result.get("member_valid_mae", []), "soup_epochs": list(stages.SOUP_EPOCHS), "official_test_loaded": False},
    )

    summary = {
        "protocol_version": stages.PROTOCOL_VERSION,
        "arm": arm,
        "stage": stage,
        "device": str(device),
        "epochs": int(result["epochs"]),
        "resumed_from": int(result["resumed_from"]),
        "lr": float(result["lr"]),
        "weight_decay": float(result["weight_decay"]),
        "node_weight_decay": float(result["node_weight_decay"]),
        "kappa": float(result["kappa"]),
        "train_seed": int(train_seed),
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
        "canonical_init_sha256": canonical_hash,
        "readout_style": getattr(model, "readout_style", "shared_reader"),
        "frozen_kappa_used": frozen_kappa is not None,
        "recomputed_kappa": float(manifest.kappa),
        "official_valid_loaded": bool(stage == "train"),
        "official_test_loaded": False,
    }
    _write_json(export_dir / "summary.json", summary)
    _write_json(out_dir / "summary.json", summary)
    if stage == "train" and result.get("valid_predictions_raw") is not None:
        stages.write_valid_predictions(export_dir, result)
        stages.write_valid_predictions(out_dir, result)

    metrics = {
        "measure": "calibrated_valid_mae" if stage == "train" else "node_prefix_train_mae",
        "stage": stage,
        "arm": arm,
        "device": str(device),
        "epochs": int(result["epochs"]),
        "raw_valid_mae": float(result.get("raw_valid_mae", float("nan"))),
        "calibrated_valid_mae": float(result.get("calibrated_valid_mae", float("nan"))),
        "final_train_mae": float(result["curve"][-1]["train_mae"]),
        "train_fitted_bias": float(result.get("train_fitted_bias", float("nan"))),
        "seconds_per_epoch": float(result["seconds_per_epoch"]),
        "wall_clock_s": float(result["wall_clock_s"]),
        "kappa": float(result["kappa"]),
        "node_weight_decay": float(result["node_weight_decay"]),
        "official_valid_loaded": bool(stage == "train"),
        "official_test_loaded": False,
    }
    _write_json(context.artifact_dir / "metrics.json", metrics)
    print(f"[overnight] done arm={arm} stage={stage} wall={result['wall_clock_s']:.0f}s", flush=True)
    return RunResult(metrics=metrics, status="completed", artifacts=["artifacts/summary.json", "artifacts/curve.csv", "artifacts/health.json"])


__all__ = ["build_runner", "fingerprints", "run", "stages", "ARM_SPEC"]