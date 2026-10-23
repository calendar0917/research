"""Train / measure entry points for the D/S relation-composition probe.

Everything reuses ``ad_train.train`` (the same loop, recipe semantics, best-valid
selection and artifact layout as the historical rounds), with:

* a probe model builder (arms ``D`` and ``S``, identical parameters),
* the ``RCCollator`` (P3N batch + dense padded bond-pair relation),
* the declared R0 recipe (AdamW 1e-3 -> 1e-5, wd 1e-5, warmup 500, cosine,
  clip 5, raw-y L1, batch 128, 600 epochs = 47,400 updates).

Official test is never loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rc_data as RD  # noqa: E402
import rc_models as RM  # noqa: E402
import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402

EXP = _HERE.parent
RUNS = EXP / "runs"
ANALYSIS = EXP / "analysis"
ROUND_ID = "zinc-relation-composition-probe-v1"

#: the declared R0 recipe, byte-identical to zps_runner.RECIPES["R0"]
RECIPE = dict(optimizer="adamw", lr=1.0e-3, min_lr=1.0e-5, weight_decay=1.0e-5,
              warmup_steps=500, epochs=600, dropout=0.0, grad_clip=5.0,
              batch_size=128, valid_batch_size=256, loss="l1")


def run_dir(arm: str, seed: int, tag: str | None = None) -> Path:
    prefix = f"{tag}_" if tag else ""
    return RUNS / f"{prefix}RC_{arm}_s{int(seed)}"


def _builder(arm: str, stats: dict):
    def build(candidate, norm, seed, device):
        return RM.build_model(candidate, norm, seed, device, arm=arm, stats=stats)
    return build


def _collator():
    train_rel = RD.load_split("train")
    valid_rel = RD.load_split("valid")
    return RD.RCCollator({"train": train_rel, "valid": valid_rel}), train_rel["stats"]


def train(arm: str, seed: int, device, *, epochs: int | None = None,
          tag: str | None = None, force: bool = False, reason: str | None = None,
          timing_epochs: int | None = None, batch_size: int | None = None) -> dict:
    if arm not in ("D", "S"):
        raise ValueError(f"unknown arm {arm!r}")
    recipe = dict(RECIPE)
    if batch_size is not None:
        recipe["batch_size"] = int(batch_size)
    if epochs is not None:
        recipe["epochs"] = int(epochs)
    out = run_dir(arm, seed, tag)
    out.mkdir(parents=True, exist_ok=True)
    collator, stats = _collator()
    # declared before the run (matches the search.json convention)
    declaration = {
        "round": ROUND_ID,
        "arm": arm,
        "seed": int(seed),
        "reason": reason,
        "recipe": recipe,
        "declared_epochs": int(recipe["epochs"]),
        "declared_updates": math.ceil(10000 / int(recipe["batch_size"])) * int(recipe["epochs"]),
        "relation_cache_train": str(RD.cache_path("train")),
        "relation_cache_valid": str(RD.cache_path("valid")),
        "relation_stats": stats,
        "official_test_loaded": False,
        "declared_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timing_epochs": timing_epochs,
    }
    C.write_json(out / "search.json", declaration)
    info = T1.train(
        f"RC_{arm}", seed, device,
        epochs=epochs, tag=tag, force=force,
        model_builder=_builder(arm, stats),
        collate_fn=collator,
        recipe_override=recipe,
        out_dir=out,
    )
    C.write_json(out / "search.json", declaration)
    return info


# ---------------------------------------------------------------------------
# cost measurement on the real 10k / 1k splits
# ---------------------------------------------------------------------------
def measure(arm: str = "D", seed: int = 0, device="cuda:0", *,
            epochs: int = 3, batch_size: int = 128, warmup_epochs: int = 1) -> dict:
    """Time complete epochs (train + full valid) on the full splits."""
    collator, stats = _collator()
    train_cache = D.load_cache("train")
    valid_cache = D.load_cache("valid")
    norm = D.load_norm()
    model, minfo = RM.build_model("RC", norm, seed, device, arm=arm, stats=stats)
    n_params = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    y_train = np.asarray(train_cache["labels"], dtype=np.float64)
    y_valid = np.asarray(valid_cache["labels"], dtype=np.float64)
    steps = math.ceil(len(y_train) / batch_size)
    torch.manual_seed(seed + 101)
    gen = torch.Generator().manual_seed(seed + 101)
    total_steps = steps * 600
    records = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(y_train), generator=gen).numpy()
        t_collate = 0.0
        t_transfer = 0.0
        t_fwd = 0.0
        t_bwd = 0.0
        t0 = time.perf_counter()
        for s in range(0, len(order), batch_size):
            rows = order[s:s + batch_size]
            t1 = time.perf_counter()
            batch = collator(train_cache, rows, "RC")
            t2 = time.perf_counter()
            batch = T1._to_device(batch, device)
            t3 = time.perf_counter()
            target = torch.as_tensor(y_train[rows], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = torch.nn.functional.l1_loss(pred.view(-1), target.view(-1))
            t4 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            t5 = time.perf_counter()
            t_collate += t2 - t1
            t_transfer += t3 - t2
            t_fwd += t4 - t3
            t_bwd += t5 - t4
        train_s = time.perf_counter() - t0
        t6 = time.perf_counter()
        model.eval()
        with torch.no_grad(), C.isolated_rng():
            for s in range(0, len(y_valid), 256):
                rows = np.arange(s, min(s + 256, len(y_valid)))
                batch = T1._to_device(collator(valid_cache, rows, "RC", valid=True), device)
                model(batch)
        t_valid = time.perf_counter() - t6
        t7 = time.perf_counter()
        torch.save({"state_dict": C.cpu_state(model)}, out := (RUNS / f"_measure_{arm}_s{seed}" / "tmp.pt"))
        t_save = time.perf_counter() - t7
        rec = {
            "epoch": epoch,
            "train_seconds": train_s,
            "valid_seconds": t_valid,
            "save_seconds": t_save,
            "train_collate_seconds": t_collate,
            "train_transfer_seconds": t_transfer,
            "train_forward_seconds": t_fwd,
            "train_backward_seconds": t_bwd,
            "steps": steps,
            "batch_size": batch_size,
        }
        if epoch > warmup_epochs:
            records.append(rec)
        print(f"[measure {arm} s{seed}] ep={epoch} train={train_s:.2f}s valid={t_valid:.2f}s "
              f"collate={t_collate:.2f} transfer={t_transfer:.2f} fwd={t_fwd:.2f} "
              f"bwd={t_bwd:.2f} save={t_save:.2f}", flush=True)
    warm = records[-1]
    per_epoch = warm["train_seconds"] + warm["valid_seconds"]
    out = {
        "arm": arm,
        "seed": int(seed),
        "device": str(device),
        "parameters": n_params,
        "batch_size": batch_size,
        "steps_per_epoch": steps,
        "warmup_epochs": warmup_epochs,
        "records": records,
        "warm_epoch": warm,
        "warm_seconds_per_epoch": per_epoch,
        "projected_600_epochs_seconds": per_epoch * 600,
        "projected_600_epochs_gpu_hours": per_epoch * 600 / 3600.0,
        "train_seconds_per_epoch": warm["train_seconds"],
        "valid_seconds_per_epoch": warm["valid_seconds"],
        "collate_seconds_per_epoch": warm["train_collate_seconds"],
        "transfer_seconds_per_epoch": warm["train_transfer_seconds"],
        "forward_seconds_per_epoch": warm["train_forward_seconds"],
        "backward_seconds_per_epoch": warm["train_backward_seconds"],
        "save_seconds": warm["save_seconds"],
    }
    (RUNS / f"_measure_{arm}_s{seed}").mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS / f"timing_{arm}_s{seed}.json", out)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prep", "train", "measure", "params"])
    p.add_argument("--arm", default="D")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--reason", default=None)
    p.add_argument("--warmup-epochs", type=int, default=1)
    args = p.parse_args(argv)
    if args.command == "prep":
        RD.prep(("train", "valid"), force=args.force)
    elif args.command == "params":
        import rc_build
        print(json.dumps(rc_build.audit(args.seed, arm=args.arm), indent=2)[:2000])
    elif args.command == "measure":
        res = measure(args.arm, args.seed, args.device, epochs=args.epochs or 3,
                      batch_size=args.batch_size or 128, warmup_epochs=args.warmup_epochs)
        print(json.dumps(res, indent=2))
    else:
        res = train(args.arm, args.seed, args.device, epochs=args.epochs, tag=args.tag,
                    force=args.force, reason=args.reason, batch_size=args.batch_size)
        print(json.dumps({k: v for k, v in res.items() if k != "curve"}, indent=2)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
