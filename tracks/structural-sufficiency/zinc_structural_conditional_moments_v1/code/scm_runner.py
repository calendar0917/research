"""Runner for the two SCM arms (seed 0, official train 10k / valid 1k).

The supervised loop is the frozen ``ad_train.train`` with recipe variant B, i.e.
exactly the historical DCC-L seed-0 recipe: AdamW, 600 epochs, 500-step
warm-up, per-step cosine 1e-3 -> 1e-5, wd 1e-5, clip 5, raw-y L1, batch 128,
full valid every epoch, strict best-valid selection, batch order seeded
``seed + 101``.  Nothing in the loop is modified; this module only supplies the
model builder, the relation-aware collation and artifact hashing.

Official test is never constructed or read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import scm_common as S
import scm_relations as R
import scm_models as M

import ad_common as C
import ad_data as D
import ad_train as T1
import dcc_models as MD

RECIPE = "B"
ARMS = {"s_only": ("S-only", False), "s_plus_a": ("S+A", True)}


def run_dir(arm_key: str, seed: int, tag: str | None = None) -> Path:
    prefix = f"{tag}_" if tag else ""
    return S.RUNS / f"{prefix}{arm_key}_s{int(seed)}"


def _sha(path: Path) -> str | None:
    return C.sha256_file(path) if path.exists() else None


def stores() -> dict:
    return {s: R.load_split(s) for s in ("train", "valid")}


def collate_fn() -> object:
    return R.make_split_collate(stores(), T1.collate)


def batch_order_fingerprint(seed: int, n_rows: int = 10000, batch_size: int = 128) -> dict:
    """Fingerprint of the row order the frozen trainer will use in epoch 1."""
    import hashlib
    gen = torch.Generator().manual_seed(int(seed) + 101)
    order = torch.randperm(n_rows, generator=gen).numpy()
    h = hashlib.sha256(np.ascontiguousarray(order, dtype=np.int64).tobytes()).hexdigest()
    return {"generator": "torch.randperm(n_rows, generator=manual_seed(seed+101))",
            "source": "frozen ad_train.train code path (unchanged)",
            "epoch1_order_sha256": h, "n_rows": int(n_rows),
            "batch_size": int(batch_size),
            "steps_per_epoch": int(np.ceil(n_rows / batch_size)),
            "first_batch_first_10_rows": order[:10].tolist()}


def prep() -> dict:
    """Build the label-free relation caches if absent (idempotent)."""
    S.CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in ("train", "valid"):
        payload = R.build_split(split)
        R.save_split(payload, split)
        out[split] = R.validate(R.load_split(split))
    return out


def assert_shared_init(seed: int, norm) -> dict:
    """Both arms must start from the same tensors, equal to the historical state."""
    pair, base = M.shared_pair_builders(norm, seed)
    hashes = {}
    for arm, (model, _) in pair.items():
        ref = MD.build_initial_state(norm, seed)[0]
        sd_a, sd_b = model.dcc.state_dict(), ref.state_dict()
        same = list(sd_a) == list(sd_b) and all(torch.equal(sd_a[k], sd_b[k]) for k in sd_b)
        if not same:
            raise RuntimeError(f"{arm}: backbone differs from the historical DCC-L state")
        hashes[arm] = C.state_hash(model)
    if len(set(hashes.values())) != 1:
        raise RuntimeError(f"arms do not share one initial state: {hashes}")
    return {**base, "arm_initial_state_hashes": hashes}


def write_artifacts(out: Path, info: dict, arm_key: str) -> dict:
    art = {
        "run_dir": str(out.relative_to(S._REPO)),
        "arm_key": arm_key,
        "candidate": info.get("candidate"),
        "arm": info.get("model_info", {}).get("arm"),
        "use_cov": info.get("model_info", {}).get("use_cov"),
        "seed": info.get("seed"),
        "recipe_variant": info.get("recipe_variant"),
        "parameter_count": info.get("parameter_count"),
        "parameter_audit": info.get("parameter_audit"),
        "parameters_breakdown": info.get("model_info", {}).get("parameters_breakdown"),
        "backbone_state_sha256": info.get("model_info", {}).get("backbone_dcc_l_state_sha256"),
        "initial_state_sha256": info.get("model_info", {}).get("initial_state_sha256"),
        "scm_state_sha256": info.get("model_info", {}).get("scm_state_sha256"),
        "bands": info.get("model_info", {}).get("bands"),
        "rank": info.get("model_info", {}).get("rank"),
        "scales": info.get("model_info", {}).get("scales"),
        "best_valid": info.get("best_valid"),
        "fixed_objects": info.get("fixed_objects"),
        "gradient_audit": info.get("gradient_audit"),
        "gpu_hours_run_level": info.get("gpu_hours"),
        "wall_seconds": info.get("wall_seconds"),
        "sha256": {
            "best_pt": _sha(out / "best.pt"),
            "final_pt": _sha(out / "final.pt"),
            "valid_best_npz": _sha(out / "predictions" / "valid_best.npz"),
            "train_json": _sha(out / "train.json"),
            "curve_csv": _sha(out / "curve.csv"),
        },
    }
    C.write_json(out / "artifacts.json", art)
    return art


def train(arm_key: str, seed: int, device, *, epochs: int | None = None,
          tag: str | None = None, force: bool = False, max_rows: int | None = None) -> dict:
    if arm_key not in ARMS:
        raise ValueError(f"unknown arm {arm_key!r}")
    arm, use_cov = ARMS[arm_key]
    norm = D.load_norm()
    init = assert_shared_init(seed, norm)
    out = run_dir(arm_key, seed, tag)
    t0 = time.perf_counter()
    info = T1.train(
        f"SCM_{arm}", seed, device, epochs=epochs, tag=tag, force=force,
        recipe_variant=RECIPE, model_builder=M.make_builder(use_cov),
        out_dir=out, collate_fn=collate_fn(),
        train_rows=None if max_rows is None else np.arange(int(max_rows)),
    )
    info["model_info"]["shared_init_audit"] = init
    info["model_info"]["batch_order"] = batch_order_fingerprint(seed)
    C.write_json(out / "train.json", info)
    art = write_artifacts(out, info, arm_key)
    C.log(
        f"[scm-{arm_key}] seed={seed} best={info['best_valid']['valid_mae']:.9f}"
        f"@{info['best_valid']['epoch']} final={info['fixed_objects']['final']['valid_mae']:.9f} "
        f"params={info['parameter_count']} wall={info['wall_seconds']:.1f}s"
    )
    return info


def timing(seed: int, device, *, epochs: int = 2) -> dict:
    """Measure full-10k epochs (with full valid) for both arms, then project."""
    norm = D.load_norm()
    out = {"protocol_id": S.TRACK_ID, "seed": int(seed), "recipe": RECIPE,
           "timing_rows": "full official train 10000 (no row subset)",
           "epochs_measured": int(epochs), "arms": {}}
    total_projected = 0.0
    for arm_key in ("s_only", "s_plus_a"):
        t0 = time.perf_counter()
        info = train(arm_key, seed, device, epochs=epochs, tag="timing", force=True)
        total = time.perf_counter() - t0
        sec = [float(e["seconds"]) for e in info["curve"]]
        mean_ep = float(np.mean(sec))
        steady = float(np.mean(sec[1:])) if len(sec) > 1 else mean_ep
        startup = float(total) - float(info["wall_seconds"])
        final_eval = float(info["wall_seconds"]) - float(np.sum(sec))
        reserve = 30.0
        projected = 1.15 * 600.0 * mean_ep + startup + final_eval + reserve
        total_projected += projected
        out["arms"][arm_key] = {
            "epoch_seconds_including_full_valid": sec,
            "mean_epoch_seconds": mean_ep,
            "steady_epoch_seconds": steady,
            "steps_per_epoch": int(info["recipe"]["steps_per_epoch"]),
            "startup_seconds": startup,
            "final_eval_seconds": final_eval,
            "process_reserve_seconds": reserve,
            "projected_600ep_seconds_15pct": projected,
            "projected_600ep_gpu_hours": projected / 3600.0,
            "run_wall_seconds": total,
            "parameters": int(info["parameter_count"]),
        }
    out["projected_both_arms_gpu_hours_15pct"] = total_projected / 3600.0
    out["budget_a100_hours"] = float(S.frozen_config().get("budget_a100_hours", 2.5))
    out["timing_cost_hours"] = sum(v["run_wall_seconds"] for v in out["arms"].values()) / 3600.0
    out["projected_total_with_timing"] = out["projected_both_arms_gpu_hours_15pct"] + out["timing_cost_hours"]
    out["fits_budget_with_15pct_margin"] = bool(
        out["projected_total_with_timing"] * 1.15 <= out["budget_a100_hours"])
    out["official_test_loaded"] = False
    S.write_json(S.ANALYSIS / "timing.json", out)
    C.log(json.dumps(out, indent=2))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prep", "timing", "train"])
    p.add_argument("--arm", choices=sorted(ARMS), default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-rows", type=int, default=None,
                   help="CPU smoke only; the formal run always uses all 10,000 rows")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if args.command == "prep":
        if args.max_rows is not None:
            raise SystemExit("--max-rows is a train-smoke option")
        C.log(json.dumps(prep(), indent=2))
    elif args.command == "timing":
        timing(args.seed, device, epochs=args.epochs or 2)
    else:
        if args.arm is None:
            raise SystemExit("--arm is required for train")
        train(args.arm, args.seed, device, epochs=args.epochs, tag=args.tag,
              force=args.force, max_rows=args.max_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
