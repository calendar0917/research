"""Short supervised schedule for DBT-PA (frozen local stack + adapters).

The recipe is fixed in ``configs/pa_v1.json``.  The formal horizon is *not*
declared in advance: it is computed only from the measured steady-state epoch
time after the narrow timing run (see ``dbt_pa_timing.py`` and the report).
The cosine horizon is frozen at ``E * 157`` updates before the formal launch
and is never reset or changed on valid fluctuations.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
_DBT = _SSUF / "zinc_directed_bond_transformer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_DBT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import dbt_data as BD  # noqa: E402
import dbt_pa_cache as PC  # noqa: E402
import dbt_pa_models as PM  # noqa: E402
from dbt_pa_models import ADAPTER_RANK  # noqa: E402

ANALYSIS = _EXP / "analysis"
RUNS = _EXP / "runs"
CONFIG = _EXP / "configs" / "pa_v1.json"

RECIPE: dict[str, Any] = {
    "optimizer": "adamw",
    "lr": 3.0e-4,
    "min_lr": 3.0e-6,
    "weight_decay": 1.0e-4,
    "warmup_updates": 1000,
    "batch_size": 64,
    "valid_batch_size": 256,
    "eval_train_batch_size": 256,
    "dropout": 0.05,
    "grad_clip": 5.0,
    "loss": "raw-y L1",
    "seed": 0,
    "adapter_rank": 8,
    "max_epochs_cap": 600,
}
STEPS_PER_EPOCH = math.ceil(10000 / RECIPE["batch_size"])


def validate_config() -> dict:
    cfg = C.read_json(CONFIG)
    rec = cfg["recipe"]
    for key, value in RECIPE.items():
        if key in rec and rec[key] != value:
            raise RuntimeError(f"configs/pa_v1.json disagrees on {key}: {rec[key]} != {value}")
    model = cfg["model"]
    if int(model["adapter_rank"]) != ADAPTER_RANK:
        raise RuntimeError("config adapter_rank does not match code")
    if int(model["layers"]) != 4 or int(model["width"]) != 64:
        raise RuntimeError("config changed DBT layers/width; this round forbids it")
    return cfg



def _to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()}


def _lr_at(step: int, total: int, recipe: Mapping[str, Any]) -> float:
    return C.cosine_with_warmup_lr(
        step, total, int(recipe["warmup_updates"]),
        float(recipe["lr"]), float(recipe["min_lr"]),
    )


def steps_per_epoch(n_rows: int = 10000) -> int:
    return int(math.ceil(n_rows / float(RECIPE["batch_size"])))


def _module_sha(model: torch.nn.Module, prefix: str) -> str:
    state = {k: v for k, v in model.state_dict().items() if k.startswith(prefix)}
    return PM.state_dict_sha256(state)


def _param_audit(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    all_params = list(model.parameters())
    total = sum(p.numel() for p in all_params)
    trainable = [p for p in all_params if p.requires_grad]
    frozen = [p for p in all_params if not p.requires_grad]
    opt_params = [p for group in optimizer.param_groups for p in group["params"]]
    opt_ids = {id(p) for p in opt_params}
    train_ids = {id(p) for p in trainable}
    frozen_ids = {id(p) for p in frozen}
    return {
        "parameters_total_inference": int(total),
        "parameters_trainable": int(sum(p.numel() for p in trainable)),
        "parameters_frozen": int(sum(p.numel() for p in frozen)),
        "parameter_tensors_total": int(len(all_params)),
        "parameter_tensors_trainable": int(len(trainable)),
        "parameter_tensors_frozen": int(len(frozen)),
        "optimizer_parameters": int(sum(p.numel() for p in opt_params)),
        "optimizer_covers_exactly_trainable": bool(opt_ids == train_ids),
        "frozen_in_optimizer": bool(bool(opt_ids & frozen_ids)),
        "under_500k_all_inference_weights": bool(total <= 500_000),
        "module_breakdown": {
            name: int(sum(p.numel() for p in mod.parameters()))
            for name, mod in model.named_children()
        },
        "trainable_breakdown": {
            name: int(sum(p.numel() for p in mod.parameters() if p.requires_grad))
            for name, mod in model.named_children()
        },
    }


@torch.no_grad()
def evaluate(model, object_cache, relation, feature, rows: np.ndarray, *,
             device, batch_size: int) -> tuple[np.ndarray, float]:
    model.eval()
    out = []
    with C.isolated_rng():
        for start in range(0, len(rows), int(batch_size)):
            chunk = np.asarray(rows[start:start + int(batch_size)], dtype=np.int64)
            a0, b0, w0 = PC.gather_cached_features(feature, chunk, device)
            batch = _to_device(BD.collate(object_cache, relation, chunk, valid=True), device)
            out.append(model.forward_cached(batch, a0, b0, w0).detach().float().cpu().numpy().reshape(-1))
    pred = np.concatenate(out, 0) if out else np.zeros(0, dtype=np.float32)
    y = np.asarray(object_cache["labels"], dtype=np.float64)[rows]
    return pred, C.mae(pred, y)


def _load_features(model_info: Mapping[str, Any], *, require: bool = True):
    ssl_sha = str(model_info["ssl"]["sha256"])
    enc_sha = str(model_info["encoder_sha256"])
    out = {}
    for split in ("train", "valid"):
        obj = BD.load_object_cache(split)
        struct = PC.structural_fingerprint(obj)
        feat = PC.load_feature_split(
            split, None, ssl_sha256=ssl_sha, encoder_sha256=enc_sha,
            structure_sha256=struct, require=require,
        )
        out[split] = (obj, feat, struct)
    return out


def train(seed: int, device, *, epochs: int, tag: str | None = None,
          force: bool = False, resume: bool = False, save_every: int = 25,
          timing_only: bool = False) -> dict:
    validate_config()
    n_epochs = int(epochs)
    if n_epochs < 1:
        raise ValueError("epochs must be >= 1")
    run_name = f"{tag}_" if tag else ""
    out = RUNS / run_name / f"pa_s{int(seed)}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.json").exists() and not force and not resume:
        C.log(f"[pa s{seed}] already present in {out}; use --force to rerun")
        return C.read_json(out / "train.json")

    train_obj = BD.load_object_cache("train")
    valid_obj = BD.load_object_cache("valid")
    train_rel = BD.load_relation_split("train")
    valid_rel = BD.load_relation_split("valid")
    y_train = np.asarray(train_obj["labels"], dtype=np.float64)
    rows_train = np.arange(len(y_train), dtype=np.int64)
    rows_valid = np.arange(len(valid_obj["labels"]), dtype=np.int64)
    if len(rows_train) != 10000 or len(rows_valid) != 1000:
        raise RuntimeError(f"unexpected split sizes {len(rows_train)}/{len(rows_valid)}")

    model, model_info = PM.build_model_pa(
        D.load_norm(), seed, device, graph_scales=C.read_json(ANALYSIS / "graph_scales.json")
        if (ANALYSIS / "graph_scales.json").exists()
        else C.read_json(_SSUF / "zinc_directed_bond_transformer_v1" / "analysis" / "graph_scales.json"),
        dropout=float(RECIPE["dropout"]),
        adapter_rank=int(RECIPE["adapter_rank"]),
    )
    features = _load_features(model_info, require=True)
    train_feat = features["train"][1]
    valid_feat = features["valid"][1]
    if train_feat is None or valid_feat is None:
        raise RuntimeError("missing train/valid frozen-feature cache")

    params = C.parameter_count(model)
    if params > 500_000:
        raise RuntimeError(f"{params} > 500k inference parameters")
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=float(RECIPE["lr"]),
        weight_decay=float(RECIPE["weight_decay"]),
    )
    audit = _param_audit(model, optimizer)
    if not audit["optimizer_covers_exactly_trainable"] or audit["frozen_in_optimizer"]:
        raise RuntimeError(f"optimizer/freeze audit failed: {audit}")

    spe = steps_per_epoch(len(rows_train))
    total_updates = spe * n_epochs
    gen = torch.Generator().manual_seed(int(seed) + 101)
    curve: list[dict] = []
    best = {"valid_mae": float("inf"), "epoch": -1, "global_step": 0}
    global_step = 0
    start_epoch = 1

    base_sha_before = _module_sha(model, "encoders.")
    adapter_sha_before = _module_sha(model, "adapter_a.")
    trunk_sha_before = _module_sha(model, "layers.")

    if resume and (out / "last.pt").exists():
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.to(device) for k, v in state["state_dict"].items()})
        optimizer.load_state_dict(state["optimizer"])
        gen.set_state(state["gen_state"])
        curve = list(state["curve"])
        best = dict(state["best"])
        global_step = int(state["global_step"])
        start_epoch = int(state["epoch"]) + 1
        if int(state["recipe"]["total_updates"]) != total_updates:
            raise RuntimeError("resume horizon mismatch; cosine would be reset")
        C.log(f"[pa s{seed}] resumed at epoch {state['epoch']} step {global_step}")

    started = time.perf_counter()
    lr = _lr_at(max(global_step, 1), total_updates, RECIPE)
    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        order = torch.randperm(len(rows_train), generator=gen).numpy()
        total_err, n_mol = 0.0, 0
        t_ep = time.perf_counter()
        for start in range(0, len(order), int(RECIPE["batch_size"])):
            rows = order[start:start + int(RECIPE["batch_size"])]
            a0, b0, w0 = PC.gather_cached_features(train_feat, rows, device)
            batch = _to_device(BD.collate(train_obj, train_rel, rows), device)
            target = torch.as_tensor(y_train[rows], dtype=torch.float32, device=device)
            pred = model.forward_cached(batch, a0, b0, w0)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters(), float(RECIPE["grad_clip"])
            )
            if not math.isfinite(float(gn)):
                raise RuntimeError(f"non-finite gradient at epoch {epoch}")
            global_step += 1
            lr = _lr_at(global_step, total_updates, RECIPE)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            total_err += float((pred.detach().view(-1) - target).abs().sum())
            n_mol += len(rows)
        online = total_err / max(n_mol, 1)
        pred_valid, valid_mae = evaluate(
            model, valid_obj, valid_rel, valid_feat, rows_valid,
            device=device, batch_size=int(RECIPE["valid_batch_size"]),
        )
        sec = time.perf_counter() - t_ep
        entry = {
            "epoch": int(epoch), "global_step": int(global_step),
            "online_train_mae": float(online), "valid_mae": float(valid_mae),
            "lr": float(lr), "seconds": float(sec), "steps": int(spe),
        }
        curve.append(entry)
        if valid_mae < best["valid_mae"]:
            best = {"valid_mae": float(valid_mae), "epoch": int(epoch),
                    "global_step": int(global_step)}
            C.atomic_torch_save(
                {"state_dict": C.cpu_state(model), "epoch": int(epoch),
                 "valid_mae": float(valid_mae), "candidate": "DBT-PA", "seed": int(seed),
                 "parameter_count": int(params),
                 "encoder_sha256": model_info["encoder_sha256"],
                 "ssl_sha256": model_info["ssl"]["sha256"],
                 "recipe": {**RECIPE, "epochs": int(n_epochs),
                            "total_updates": int(total_updates),
                            "steps_per_epoch": int(spe)}},
                out / "best.pt",
            )
        C.log(
            f"[pa s{seed}] ep={epoch:04d}/{n_epochs} valid={valid_mae:.6f} "
            f"online={online:.6f} lr={lr:.3e} best={best['valid_mae']:.6f}@{best['epoch']} "
            f"{sec:.2f}s"
        )
        if epoch % int(save_every) == 0 or epoch == n_epochs:
            C.atomic_torch_save(
                {
                    "state_dict": C.cpu_state(model),
                    "optimizer": optimizer.state_dict(),
                    "gen_state": gen.get_state(),
                    "epoch": int(epoch), "global_step": int(global_step),
                    "best": best, "curve": curve, "candidate": "DBT-PA",
                    "seed": int(seed),
                    "recipe": {**RECIPE, "epochs": int(n_epochs),
                               "total_updates": int(total_updates),
                               "steps_per_epoch": int(spe)},
                },
                out / "last.pt",
            )

    wall = time.perf_counter() - started
    if (out / "last.pt").exists():
        C.atomic_torch_save(torch.load(out / "last.pt", map_location="cpu", weights_only=False),
                            out / "final.pt")

    # Evaluate the selected best and the final state from fixed cached objects.
    best_model, _ = PM.build_model_pa(
        D.load_norm(), seed, device,
        graph_scales=C.read_json(ANALYSIS / "graph_scales.json")
        if (ANALYSIS / "graph_scales.json").exists()
        else C.read_json(_SSUF / "zinc_directed_bond_transformer_v1" / "analysis" / "graph_scales.json"),
        dropout=float(RECIPE["dropout"]), adapter_rank=int(RECIPE["adapter_rank"]),
    )
    state = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    C.load_state(best_model, state["state_dict"], device)
    pred_valid, best_valid = evaluate(
        best_model, valid_obj, valid_rel, valid_feat, rows_valid,
        device=device, batch_size=int(RECIPE["valid_batch_size"]),
    )
    _, best_train = evaluate(
        best_model, train_obj, train_rel, train_feat, rows_train,
        device=device, batch_size=int(RECIPE["eval_train_batch_size"]),
    )
    _, final_valid = evaluate(
        model, valid_obj, valid_rel, valid_feat, rows_valid,
        device=device, batch_size=int(RECIPE["valid_batch_size"]),
    )
    _, final_train = evaluate(
        model, train_obj, train_rel, train_feat, rows_train,
        device=device, batch_size=int(RECIPE["eval_train_batch_size"]),
    )
    preds = out / "predictions"
    preds.mkdir(exist_ok=True)
    np.savez(preds / "valid_best.npz", row=rows_valid, pred=pred_valid)

    base_sha_after = _module_sha(model, "encoders.")
    adapter_sha_after = _module_sha(model, "adapter_a.")
    trunk_sha_after = _module_sha(model, "layers.")
    info = {
        "round": "zinc-restricted-adaptation-v1",
        "candidate": "DBT-PA",
        "seed": int(seed),
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "data": {
            "source": "official PyG ZINC subset=True train/valid",
            "n_train": int(len(y_train)), "n_valid": int(len(rows_valid)),
            "official_test_loaded": False,
            "train_label_sha256": C.sha256_array(y_train),
            "valid_label_sha256": C.sha256_array(np.asarray(valid_obj["labels"], np.float64)),
        },
        "model_info": model_info,
        "parameter_audit": audit,
        "feature_cache": {
            split: {
                "path": str(features[split][1]["_path"]),
                "sha256": features[split][1]["_sha256"],
                "identity": {k: v for k, v in features[split][1]["meta"].items()
                             if k in ("feature_version", "split", "ssl_sha256",
                                      "encoder_sha256", "structure_sha256")},
                "build_seconds": features[split][1]["meta"]["build_seconds"],
                "n_atoms_total": features[split][1]["meta"]["n_atoms_total"],
                "n_bonds_total": features[split][1]["meta"]["n_bonds_total"],
                "n_junctions_total": features[split][1]["meta"]["n_junctions_total"],
            }
            for split in ("train", "valid")
        },
        "recipe": {
            **RECIPE, "epochs": int(n_epochs), "steps_per_epoch": int(spe),
            "total_updates": int(total_updates),
            "actual_global_steps": int(global_step),
            "valid_every_epoch": True,
            "selection": "argmin(valid_mae), strict <, ties keep the earlier epoch",
            "cosine_horizon": "E * 157 fixed before launch",
        },
        "schedule_checks": {
            "lr_step_1": float(_lr_at(1, total_updates, RECIPE)),
            "lr_step_warmup": float(_lr_at(int(RECIPE["warmup_updates"]), total_updates, RECIPE)),
            "lr_step_last": float(_lr_at(total_updates, total_updates, RECIPE)),
        },
        "hashes": {
            "encoder_before": base_sha_before,
            "encoder_after": base_sha_after,
            "encoder_unchanged": bool(base_sha_before == base_sha_after),
            "adapter_a_before": adapter_sha_before,
            "adapter_a_after": adapter_sha_after,
            "adapter_a_updated": bool(adapter_sha_before != adapter_sha_after),
            "trunk_before": trunk_sha_before,
            "trunk_after": trunk_sha_after,
            "trunk_updated": bool(trunk_sha_before != trunk_sha_after),
        },
        "best_valid": best,
        "fixed_objects": {
            "best": {"valid_mae": float(best_valid),
                     "train_mae_eval_mode": float(best_train),
                     "epoch": int(best["epoch"])},
            "final": {"valid_mae": float(final_valid),
                      "train_mae_eval_mode": float(final_train),
                      "epoch": int(n_epochs)},
        },
        "curve": curve,
        "wall_seconds": float(wall),
        "gpu_seconds": None if str(device).startswith("cpu") else float(wall),
        "gpu_hours": None if str(device).startswith("cpu") else float(wall) / 3600.0,
        "peak_gpu_mem_mb": float(torch.cuda.max_memory_allocated(device) / 2 ** 20)
        if str(device).startswith("cuda") else None,
        "device": str(device),
        "timing_only": bool(timing_only),
        "status": "COMPLETE" if global_step == total_updates else "INCOMPLETE",
        "notes": [
            "online_train_mae is a within-epoch pre-update running mean",
            "train_mae_eval_mode is a full-10k eval-mode measurement of the fixed checkpoint",
            "the frozen local stack is never in the optimizer and is forced to eval mode",
            "the cosine horizon is E * 157 and is never reset",
        ],
    }
    C.write_json(out / "train.json", info)
    C.write_curve_csv(out / "curve.csv", curve)
    C.log(f"[pa s{seed}] done best={best_valid:.6f}@{best['epoch']} "
          f"final={final_valid:.6f} wall={wall:.1f}s params={params}")
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-every", type=int, default=25)
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    train(args.seed, device, epochs=args.epochs, tag=args.tag, force=args.force,
          resume=args.resume, save_every=args.save_every,
          timing_only=args.tag == "timing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
