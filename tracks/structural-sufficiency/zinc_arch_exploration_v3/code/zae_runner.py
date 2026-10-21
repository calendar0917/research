"""CLI for zinc-arch-exploration-v3: prep / timing / train / ssl."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_V1 = _EXP.parent / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import zae_data as ZD  # noqa: E402
import zae_models as ZM  # noqa: E402
import zae_ssl as ZS  # noqa: E402
import zae_train as ZT  # noqa: E402

ANALYSIS = _EXP / "analysis"


def make_builder(arm: str):
    ssl = True

    def builder(candidate, norm, seed, device):
        return ZM.build_model(arm, norm, seed, device, ssl=ssl)

    return builder


def prep(splits=("train", "valid")) -> dict:
    out = {}
    for split in splits:
        t0 = time.perf_counter()
        data = ZD.build_pair_split(split)
        path = ZD.save_pair_split(data, split)
        out[split] = {
            "path": str(path.relative_to(C.REPO)),
            "sha256": C.sha256_file(path),
            "n_molecules": data["n_molecules"],
            "seconds": float(time.perf_counter() - t0),
            "meta": data["meta"],
        }
        C.log(f"[prep] {split} -> {path} ({out[split]['seconds']:.1f}s)")
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS / "prep_pair.json", out)
    return out


def write_artifacts(out: Path, info: dict) -> dict:
    art = {
        "run_dir": str(out.relative_to(C.REPO)),
        "candidate": info.get("candidate"),
        "seed": info.get("seed"),
        "parameter_count": info.get("parameter_count"),
        "model_info": info.get("model_info"),
        "best_valid": info.get("best_valid"),
        "fixed_objects": info.get("fixed_objects"),
        "recipe": {k: info["recipe"].get(k) for k in
                   ("epochs", "batch_size", "lr", "min_lr", "weight_decay", "warmup_steps_effective",
                    "steps_per_epoch", "updates", "optimizer")},
        "gpu_hours_run_level": info.get("gpu_hours"),
        "wall_seconds": info.get("wall_seconds"),
        "sha256": {
            "best_pt": C.sha256_file(out / "best.pt") if (out / "best.pt").exists() else None,
            "final_pt": C.sha256_file(out / "final.pt") if (out / "final.pt").exists() else None,
            "valid_best_npz": C.sha256_file(out / "predictions" / "valid_best.npz")
            if (out / "predictions" / "valid_best.npz").exists() else None,
            "train_json": C.sha256_file(out / "train.json"),
            "curve_csv": C.sha256_file(out / "curve.csv"),
        },
    }
    C.write_json(out / "artifacts.json", art)
    return art


def timing(arm: str, seed: int, device, *, epochs: int = 3) -> dict:
    out_dir = ZT.run_dir(arm, seed, tag="timing")
    info = ZT.train_supervised(
        arm, seed, device, builder=make_builder(arm), model_info={}, out_dir=out_dir,
        epochs=epochs, force=True,
    )
    curve = info["curve"]
    epoch_seconds = [float(e["seconds"]) for e in curve]
    mean_epoch = float(np.mean(epoch_seconds))
    steady = float(np.mean(epoch_seconds[1:])) if len(epoch_seconds) > 1 else mean_epoch
    steps = int(info["recipe"]["steps_per_epoch"])
    planned_epochs = int(ZT.RECIPE["epochs"])
    startup = float(info["wall_seconds"]) - float(np.sum(epoch_seconds))
    projected = 1.15 * planned_epochs * steady + startup + 60.0
    payload = {
        "protocol_id": ZT.PROTOCOL_ID,
        "arm": arm,
        "seed": int(seed),
        "timing_rows": "full official train 10000 + full valid 1000",
        "epochs_measured": len(epoch_seconds),
        "steps_per_epoch": steps,
        "epoch_seconds_including_full_valid": epoch_seconds,
        "mean_epoch_seconds": mean_epoch,
        "steady_epoch_seconds": steady,
        "planned_epochs": planned_epochs,
        "planned_updates": steps * planned_epochs,
        "projected_seconds_with_15pct_and_overheads": projected,
        "projected_gpu_hours": planned_epochs * steady / 3600.0,
        "projected_gpu_hours_with_15pct": projected / 3600.0,
        "parameters": int(info["parameter_count"]),
        "parameter_audit": info["parameter_audit"],
        "gradient_audit": info["gradient_audit"],
        "best_valid_at_timing_horizon": info["best_valid"],
        "official_test_loaded": False,
        "note": ("steady epoch seconds = mean of epochs 2..N; projection is engineering guidance "
                 "for the declared schedule, not a stop rule"),
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS / f"timing_{arm}.json", payload)
    C.log(json.dumps(payload, indent=2))
    return payload


def train(arm: str, seed: int, device, *, epochs=None, tag=None, force=False,
          init_from: str | None = None) -> dict:
    if arm == "A_SSL":
        base_arm = "A"
        init_path = Path(init_from) if init_from else None
        if init_path is None:
            raise SystemExit("A_SSL needs --init-from runs/ASSL_pretrain_s0/pretrain.pt")
        if not init_path.is_absolute():
            init_path = C.REPO / init_path
        init = ZS.load_pretrained(init_path, device="cpu")
        extra = {
            "ssl_pretrain": {
                "path": str(init_path.relative_to(C.REPO)),
                "sha256": C.sha256_file(init_path),
            }
        }
    else:
        base_arm = arm
        init = None
        extra = None
    out = ZT.run_dir(arm, seed, tag)
    info = ZT.train_supervised(
        arm, seed, device, builder=make_builder(base_arm), model_info={}, out_dir=out,
        epochs=epochs, force=force, init_state=init, extra_info=extra,
    )
    art = write_artifacts(out, info)
    C.log(f"[{arm}] seed={seed} best={info['best_valid']['valid_mae']:.9f}@"
          f"{info['best_valid']['epoch']} params={info['parameter_count']} "
          f"updates={info['recipe']['updates']} wall={info['wall_seconds']:.1f}s "
          f"best_sha={art['sha256']['best_pt']}")
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prep", "timing", "train", "ssl"])
    p.add_argument("--arm", default="A", choices=["A", "B", "A_SSL", "A_attn"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--init-from", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--splits", default="train,valid")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if args.command == "prep":
        prep(tuple(s.strip() for s in args.splits.split(",") if s.strip()))
    elif args.command == "timing":
        timing(args.arm, args.seed, device, epochs=args.epochs or 3)
    elif args.command == "train":
        train(args.arm, args.seed, device, epochs=args.epochs, tag=args.tag,
              force=args.force, init_from=args.init_from)
    else:
        ZS.pretrain(args.seed, device, epochs=args.epochs, tag=args.tag,
                    force=args.force, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
