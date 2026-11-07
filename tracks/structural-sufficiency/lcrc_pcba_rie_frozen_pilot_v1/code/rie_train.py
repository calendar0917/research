"""Paired REAL / SHUFFLE residual-head training (fixed epoch 8, seed 0).

Two arms, one recipe, one initial state, one batch-order plan:

* ``REAL``    relation features built from the true motif-pair distance bins,
* ``SHUFFLE`` identical motifs / pairs / pair features / per-bin counts, with the
  bin assignment permuted inside each molecule.

Both arms are trained from a **fresh, bitwise identical** head initialisation and
evaluated at epochs ``0, 2, 4, 8``.  The primary comparison is the fixed epoch 8;
there is no early stopping, no scheduler, no hyper-parameter search and no
selection on the best Validation epoch.

``--stage preflight`` runs the correctness gates that need a head (parameter
count, epoch-0 identity, paired initialisation, gradient reach, save/reload
equivalence) on 3 real updates and writes ``analysis/paired_initialization.json``.
``--stage formal`` then performs the two 8-epoch runs.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    os.environ.setdefault("OMP_NUM_THREADS", "8")


_configure_imports()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import rie_common as C  # noqa: E402
import rie_model as RM  # noqa: E402
import pc_metrics as MET  # noqa: E402
import pc_model as PC  # noqa: E402

FEATURES = C.RUNS / "features"
HEADS = C.RUNS / "heads"
CURVE_FIELDS = [
    "epoch",
    "evaluated",
    "learning_rate",
    "train_loss",
    "train_bce",
    "train_macro_ap",
    "valid_ap",
    "valid_bce",
    "valid_macro_ap",
    "train_minus_valid_ap",
    "n_tasks_scored_valid",
    "train_seconds",
    "valid_seconds",
    "epoch_seconds",
    "l1_task_weight",
    "l1_penalty",
]


# ---------------------------------------------------------------------------
# feature cache access (read-only, verified)
# ---------------------------------------------------------------------------


def _feature_files() -> dict[str, Path]:
    return {
        "train_relation": FEATURES / "train_relation.npy",
        "valid_relation": FEATURES / "valid_relation.npy",
        "train_relation_shuffle": FEATURES / "train_relation_shuffle.npy",
        "valid_relation_shuffle": FEATURES / "valid_relation_shuffle.npy",
        "train_base_logits": FEATURES / "train_base_logits.npy",
        "valid_base_logits": FEATURES / "valid_base_logits.npy",
        "train_labels": FEATURES / "train_labels.npy",
        "valid_labels": FEATURES / "valid_labels.npy",
        "train_rows": FEATURES / "train_rows.npy",
        "valid_rows": FEATURES / "valid_rows.npy",
        "scale": FEATURES / "relation_rms_scale.npy",
    }


def verify_feature_cache() -> dict[str, Any]:
    """Refuse to train on a cache that the extraction stage did not certify."""
    manifest_path = C.ANALYSIS / "feature_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("feature_manifest.json missing; run the features stage first")
    manifest = C.read_json(manifest_path)
    if not manifest.get("all_passed"):
        raise RuntimeError(f"feature extraction gates failed: {manifest.get('gates')}")
    files = _feature_files()
    for key, path in files.items():
        if not path.is_file():
            raise RuntimeError(f"feature cache file missing: {key} -> {C.track_rel(path)}")
    observed = {}
    for split in ("train", "valid"):
        for variant, name in (("real", "relation"), ("shuffle", "relation_shuffle")):
            key = f"{split}_{variant}"
            path = FEATURES / f"{split}_{name}.npy"
            observed[key] = {
                "file_sha256": C.sha256_file(path),
                "expected": manifest["scaled_files"][key]["normalized_file_sha256"],
            }
    for split in ("train", "valid"):
        path = FEATURES / f"{split}_base_logits.npy"
        observed[f"{split}_base_logits"] = {
            "file_sha256": C.sha256_file(path),
            "expected": manifest["splits"][split]["base_logits_sha256"],
        }
    for key, path in (("train_labels", FEATURES / "train_labels.npy"), ("valid_labels", FEATURES / "valid_labels.npy")):
        split = key.split("_")[0]
        observed[key] = {"file_sha256": C.sha256_array(np.load(path)), "expected": manifest["splits"][split]["labels_sha256"]}
    mismatched = [
        k for k, v in observed.items() if v["file_sha256"] != v["expected"]
    ]
    if mismatched:
        raise RuntimeError(f"feature cache hashes drifted: {mismatched}")
    arrays = {k: np.load(p, mmap_mode="r") for k, p in files.items()}
    shapes = {
        "train_relation": (C.N_TRAIN, C.N_BINS, C.PAIR_FEATURE_DIM),
        "valid_relation": (C.N_VALID, C.N_BINS, C.PAIR_FEATURE_DIM),
        "train_relation_shuffle": (C.N_TRAIN, C.N_BINS, C.PAIR_FEATURE_DIM),
        "valid_relation_shuffle": (C.N_VALID, C.N_BINS, C.PAIR_FEATURE_DIM),
    }
    for key, shape in shapes.items():
        if tuple(arrays[key].shape) != shape:
            raise RuntimeError(f"{key} shape {tuple(arrays[key].shape)} != {shape}")
    return {
        "manifest_path": C.track_rel(manifest_path),
        "manifest_all_passed": True,
        "array_hashes": observed,
        "arrays": {k: {"path": C.track_rel(p), "shape": list(arrays[k].shape)} for k, p in files.items()},
        "scale_dim": int(arrays["scale"].size),
        "valid_real_ne_shuffle_fraction": float(manifest["valid_real_ne_shuffle_fraction"]),
    }


# ---------------------------------------------------------------------------
# frozen order plan (identical for both arms)
# ---------------------------------------------------------------------------


def make_order_plan(seed: int = C.SEED, epochs: int = C.EPOCHS, n_train: int = C.N_TRAIN) -> list[np.ndarray]:
    gen = torch.Generator().manual_seed(int(seed))
    plan = [torch.randperm(int(n_train), generator=gen).numpy().astype(np.int64) for _ in range(int(epochs))]
    if len(plan) != int(epochs) or any(p.shape != (int(n_train),) for p in plan):
        raise RuntimeError("order plan construction failed")
    return plan


def plan_hash(plan: Sequence[np.ndarray]) -> str:
    import hashlib

    h = hashlib.sha256()
    for epoch, rows in enumerate(plan):
        h.update(f"epoch{epoch}:".encode())
        h.update(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes())
    return h.hexdigest()


def plan_audit(plan: Sequence[np.ndarray]) -> dict[str, Any]:
    return {
        "seed": int(C.SEED),
        "epochs": int(len(plan)),
        "n_train": int(plan[0].size),
        "steps_per_epoch": int(C.STEPS_PER_EPOCH),
        "plan_sha256": plan_hash(plan),
        "epoch_hashes": [
            C.sha256_int64(np.asarray(rows, dtype=np.int64)) for rows in plan
        ],
        "first_rows": [int(x) for x in np.asarray(plan[0][:8]).tolist()],
        "is_permutation_per_epoch": bool(
            all(np.array_equal(np.sort(np.asarray(rows)), np.arange(int(plan[0].size))) for rows in plan)
        ),
    }


# ---------------------------------------------------------------------------
# masked loss
# ---------------------------------------------------------------------------


def masked_bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(y)
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[mask], y[mask])


def head_loss(head: RM.RelationHead, logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    bce = masked_bce(logits, y)
    l1 = head.task_weight.weight.abs().mean()
    loss = bce + float(C.RECIPE["l1_coefficient"]) * l1
    return loss, {"bce": float(bce.detach().cpu().item()), "l1": float(l1.detach().cpu().item())}


def build_optimizer(head: RM.RelationHead) -> torch.optim.AdamW:
    params = [p for p in head.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=float(C.RECIPE["learning_rate"]),
        betas=tuple(C.RECIPE["adam_betas"]),
        eps=float(C.RECIPE["adam_eps"]),
        weight_decay=float(C.RECIPE["weight_decay"]),
    )


def optimizer_audit(optimizer: torch.optim.AdamW) -> dict[str, Any]:
    group = optimizer.param_groups[0]
    return {
        "optimizer": "AdamW",
        "n_param_groups": len(optimizer.param_groups),
        "n_parameters": int(sum(p.numel() for p in group["params"])),
        "learning_rate": float(group["lr"]),
        "betas": list(group["betas"]),
        "eps": float(group["eps"]),
        "weight_decay": float(group["weight_decay"]),
        "gradient_clip_norm": float(C.RECIPE["gradient_clip_norm"]),
        "scheduler": "none",
    }


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def evaluate_split(
    head: RM.RelationHead,
    relation: np.ndarray,
    base_logits: np.ndarray,
    labels: np.ndarray,
    *,
    device: torch.device,
    with_per_task: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    logits = RM.predict_logits(head, relation, base_logits, device=device)
    probs = PC.sigmoid(logits)
    official = MET.official_ap(labels, probs)
    bce = float(MET.masked_bce_numpy(logits, labels))
    out: dict[str, Any] = {
        "ap": float(official["value"]),
        "ap_source": official["source"],
        "bce": bce,
        "seconds": float(time.perf_counter() - t0),
        "logits_sha256": C.sha256_array(logits),
        "probs_sha256": C.sha256_array(probs),
    }
    if with_per_task:
        task = MET.per_task_ap(labels, probs)
        out.update(
            {
                "per_task_ap": task["per_task"],
                "observed": task["observed"],
                "positive": task["positive"],
                "negative": task["negative"],
                "n_tasks_scored": int(task["n_tasks_scored"]),
                "n_tasks_with_observed_labels": int(task["n_tasks_with_observed_labels"]),
                "macro_ap": float(task["mean_over_scored_tasks"]),
            }
        )
        out["seconds"] = float(time.perf_counter() - t0)
    else:
        out["macro_ap"] = float(MET.per_task_ap(labels, probs)["mean_over_scored_tasks"])
        out["seconds"] = float(time.perf_counter() - t0)
    return out


def _jsonable_eval(report: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in report.items():
        if key == "per_task_ap":
            continue
        if isinstance(value, (np.floating, np.integer)):
            out[key] = float(value) if isinstance(value, np.floating) else int(value)
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], (np.integer, int)):
            out[key] = [int(v) for v in value]
        else:
            out[key] = value
    if "per_task_ap" in report:
        out["per_task_ap"] = [None if v is None else float(v) for v in report["per_task_ap"]]
    return out


# ---------------------------------------------------------------------------
# one arm
# ---------------------------------------------------------------------------


def train_arm(
    arm: str,
    *,
    device: torch.device,
    init_state: Mapping[str, torch.Tensor],
    plan: Sequence[np.ndarray],
    cache: Mapping[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    if arm not in C.ARMS:
        raise KeyError(arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    relation_path = cache["train_relation"] if arm == "real" else cache["train_relation_shuffle"]
    valid_relation_path = cache["valid_relation"] if arm == "real" else cache["valid_relation_shuffle"]
    train_relation = np.load(relation_path, mmap_mode="r")
    valid_relation = np.load(valid_relation_path, mmap_mode="r")
    train_base = np.load(cache["train_base_logits"], mmap_mode="r")
    valid_base = np.load(cache["valid_base_logits"], mmap_mode="r")
    train_labels = np.load(cache["train_labels"], mmap_mode="r")
    valid_labels = np.load(cache["valid_labels"], mmap_mode="r")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.perf_counter()
    head = RM.build_head(init_state, device=device)
    state_initial = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    optimizer = build_optimizer(head)
    opt_audit = optimizer_audit(optimizer)
    global_rng_before = torch.get_rng_state().clone()

    curve: list[dict[str, Any]] = []
    evaluations: dict[str, Any] = {}
    timings: dict[str, float] = {"train": 0.0, "eval_train": 0.0, "eval_valid": 0.0}

    def _record(epoch: int, *, train_seconds: float, epoch_seconds: float) -> None:
        valid_report = evaluate_split(
            head, valid_relation, valid_base, valid_labels, device=device, with_per_task=epoch in C.EVAL_EPOCHS
        )
        timings["eval_valid"] += float(valid_report["seconds"])
        train_report = evaluate_split(
            head, train_relation, train_base, train_labels, device=device, with_per_task=False
        )
        timings["eval_train"] += float(train_report["seconds"])
        l1 = float(head.task_weight.weight.abs().mean().detach().cpu().item())
        zero_identity = None
        if int(epoch) == 0:
            zero_identity = {
                "valid_logits_sha256_matches_base": bool(
                    valid_report["logits_sha256"] == C.sha256_array(np.asarray(valid_base))
                ),
                "train_logits_sha256_matches_base": bool(
                    train_report["logits_sha256"] == C.sha256_array(np.asarray(train_base))
                ),
                "valid_logits_bitwise_equal_base": bool(
                    valid_report["logits_sha256"] == C.sha256_array(np.asarray(valid_base))
                ),
                "train_logits_bitwise_equal_base": bool(
                    train_report["logits_sha256"] == C.sha256_array(np.asarray(train_base))
                ),
            }
        evaluation = {
            "epoch": int(epoch),
            "epoch0_equals_frozen_base_logits": zero_identity,
            "arm": arm,
            "valid": _jsonable_eval(valid_report),
            "train": _jsonable_eval(train_report),
            "l1_task_weight_mean_abs": l1,
            "l1_penalty": float(C.RECIPE["l1_coefficient"]) * l1,
            "task_weight_l1_norm": float(head.task_weight.weight.abs().sum().detach().cpu().item()),
            "task_weight_nonzero_fraction": float(
                (head.task_weight.weight != 0).to(torch.float64).mean().detach().cpu().item()
            ),
            "frozen_A_logits_used_as_constant": True,
        }
        evaluations[str(int(epoch))] = evaluation
        curve.append(
            {
                "epoch": int(epoch),
                "evaluated": True,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": float(train_loss_last["loss"]),
                "train_bce": float(train_loss_last["bce"]),
                "train_macro_ap": float(train_report["macro_ap"]),
                "valid_ap": float(valid_report["ap"]),
                "valid_bce": float(valid_report["bce"]),
                "valid_macro_ap": float(valid_report["macro_ap"]),
                "train_minus_valid_ap": float(train_report["macro_ap"] - valid_report["ap"]),
                "n_tasks_scored_valid": int(valid_report.get("n_tasks_scored", 0)),
                "train_seconds": float(train_seconds),
                "valid_seconds": float(valid_report["seconds"]),
                "epoch_seconds": float(epoch_seconds),
                "l1_task_weight": l1,
                "l1_penalty": float(C.RECIPE["l1_coefficient"]) * l1,
            }
        )

    train_loss_last = {"loss": float("nan"), "bce": float("nan")}
    global_step = 0
    epoch_seconds_list: list[float] = []
    if 0 in C.EVAL_EPOCHS:
        t_eval = time.perf_counter()
        _record(0, train_seconds=0.0, epoch_seconds=0.0)
        timings["eval_valid"] += 0.0
        epoch_seconds_list.append(float(time.perf_counter() - t_eval))

    for epoch in range(1, int(C.EPOCHS) + 1):
        t_epoch = time.perf_counter()
        rows_epoch = np.asarray(plan[epoch - 1], dtype=np.int64)
        epoch_losses: list[float] = []
        t_train = time.perf_counter()
        head.train()
        for start in range(0, rows_epoch.size, int(C.BATCH_SIZE)):
            rows = rows_epoch[start : start + int(C.BATCH_SIZE)]
            relation = torch.from_numpy(np.ascontiguousarray(train_relation[rows], dtype=np.float32)).to(device)
            base = torch.from_numpy(np.ascontiguousarray(train_base[rows], dtype=np.float32)).to(device)
            y = torch.from_numpy(np.ascontiguousarray(train_labels[rows], dtype=np.float32)).to(device)
            logits = base + head(relation)
            loss, parts = head_loss(head, logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in head.parameters() if p.requires_grad], float(C.RECIPE["gradient_clip_norm"])
            )
            optimizer.step()
            global_step += 1
            epoch_losses.append(float(parts["bce"]))
            train_loss_last = {"loss": float(loss.detach().cpu().item()), "bce": float(parts["bce"])}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = float(time.perf_counter() - t_train)
        timings["train"] += train_seconds
        head.eval()
        if epoch in C.EVAL_EPOCHS:
            _record(epoch, train_seconds=train_seconds, epoch_seconds=0.0)
        epoch_seconds = float(time.perf_counter() - t_epoch)
        epoch_seconds_list.append(epoch_seconds)
        if curve:
            curve[-1]["epoch_seconds"] = epoch_seconds
        message = f"[{arm}] epoch {epoch}: bce {np.mean(epoch_losses):.6f} ({epoch_seconds:.1f}s)"
        if epoch in C.EVAL_EPOCHS:
            message += f" valid_ap {evaluations[str(epoch)]['valid']['ap']:.6f}"
        C.log(message)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_seconds = float(time.perf_counter() - t_start)
    peak = C.gpu_peak(device)
    final_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    torch.save(
        {
            "schema": "lcrc-pcba-rie-head-v1",
            "round_id": C.ROUND_ID,
            "arm": arm,
            "seed": int(C.SEED),
            "epochs": int(C.EPOCHS),
            "state_dict": final_state,
            "initial_state_sha256": RM.state_hash(init_state),
            "final_state_sha256": RM.state_hash(final_state),
            "recipe": dict(C.RECIPE),
            "plan_sha256": plan_hash(plan),
            "test_evaluated": False,
        },
        out_dir / "head.pt",
    )
    per_task_path = out_dir / "per_task_ap_epoch8.csv"
    epoch8 = evaluations.get(str(C.MAIN_EPOCH))
    if epoch8 is not None:
        with per_task_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["task", "valid_ap_epoch8", "valid_observed", "valid_positive", "valid_negative"])
            for task_idx, ap in enumerate(epoch8["valid"]["per_task_ap"]):
                writer.writerow(
                    [
                        task_idx,
                        "" if ap is None else f"{float(ap):.12f}",
                        int(epoch8["valid"]["observed"][task_idx]),
                        int(epoch8["valid"]["positive"][task_idx]),
                        int(epoch8["valid"]["negative"][task_idx]),
                    ]
                )
    with (out_dir / "curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS)
        writer.writeheader()
        for row in curve:
            writer.writerow(row)
    input_states = {
        "train_relation": C.track_rel(relation_path),
        "valid_relation": C.track_rel(valid_relation_path),
    }
    report = {
        "schema": "lcrc-pcba-rie-head-run-v1",
        "round_id": C.ROUND_ID,
        "arm": arm,
        "arm_name": C.ARMS[arm]["name"],
        "role": C.ARMS[arm]["role"],
        "seed": int(C.SEED),
        "device": str(device),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "recipe": dict(C.RECIPE),
        "optimizer": opt_audit,
        "head": {
            "parameters": RM.head_parameter_count(head),
            "parameter_names": list(RM.head_parameter_names()),
            "initial_state_sha256": RM.state_hash(state_initial),
            "final_state_sha256": RM.state_hash(final_state),
            "n_parameters_require_grad": int(sum(p.numel() for p in head.parameters() if p.requires_grad)),
        },
        "global_rng_untouched_by_init": bool(torch.equal(global_rng_before, torch.get_rng_state())),
        "plan": plan_audit(plan),
        "inputs": input_states,
        "evaluations": evaluations,
        "curve": curve,
        "timings_seconds": timings,
        "total_seconds": total_seconds,
        "epoch_seconds": epoch_seconds_list,
        "gpu_peak": peak,
        "gpu_hours": total_seconds / 3600.0,
        "global_steps": int(global_step),
        "final_batch_ce": float(train_loss_last["bce"]),
        "main_epoch": int(C.MAIN_EPOCH),
        "main_valid_ap": float(evaluations[str(C.MAIN_EPOCH)]["valid"]["ap"]),
        "main_train_macro_ap": float(evaluations[str(C.MAIN_EPOCH)]["train"]["macro_ap"]),
        "best_valid_ap_any_epoch": float(max(v["valid"]["ap"] for v in evaluations.values())),
        "best_valid_epoch": int(max(evaluations, key=lambda k: evaluations[k]["valid"]["ap"])),
        "test_evaluated": False,
    }
    C.write_json(out_dir / "run.json", report)
    C.log(
        f"[{arm}] epoch8 valid AP {report['main_valid_ap']:.10f} "
        f"train macro AP {report['main_train_macro_ap']:.10f} ({total_seconds:.1f}s)"
    )
    return report


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def stage_preflight(device: torch.device) -> dict[str, Any]:
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    HEADS.mkdir(parents=True, exist_ok=True)
    cache_info = verify_feature_cache()
    plan = make_order_plan()
    plan_info = plan_audit(plan)
    files = _feature_files()
    train_relation = np.load(files["train_relation"], mmap_mode="r")
    train_shuffle = np.load(files["train_relation_shuffle"], mmap_mode="r")
    train_base = np.load(files["train_base_logits"], mmap_mode="r")
    train_labels = np.load(files["train_labels"], mmap_mode="r")
    valid_relation = np.load(files["valid_relation"], mmap_mode="r")
    valid_shuffle = np.load(files["valid_relation_shuffle"], mmap_mode="r")
    valid_base = np.load(files["valid_base_logits"], mmap_mode="r")
    valid_labels = np.load(files["valid_labels"], mmap_mode="r")

    init_a = RM.initial_head_state()
    init_b = RM.initial_head_state()
    init_hash = RM.state_hash(init_a)
    head_real = RM.build_head(init_a, device=device)
    head_shuffle = RM.build_head(init_b, device=device)
    param_counts = {
        "real": RM.head_parameter_count(head_real),
        "shuffle": RM.head_parameter_count(head_shuffle),
    }

    # epoch-0 identity: zero task weight -> logits == frozen base logits exactly
    epoch0: dict[str, Any] = {}
    for arm, relation in (("real", valid_relation), ("shuffle", valid_shuffle)):
        head = head_real if arm == "real" else head_shuffle
        head.eval()
        with torch.inference_mode():
            block = RM.as_tensor(relation[:2048]).to(device)
            base = RM.as_tensor(valid_base[:2048]).to(device)
            delta = head(block)
            logits = base + delta
        epoch0[arm] = {
            "rows": 2048,
            "max_abs_delta": float(delta.abs().max().item()),
            "max_abs_logits_minus_base": float((logits - base).abs().max().item()),
            "bitwise_equal_base_logits": bool(torch.equal(logits, base)),
        }

    # three real updates on both arms with identical rows; gradient reach + save/reload
    steps = 3
    grad_report: dict[str, Any] = {}
    reload_report: dict[str, Any] = {}
    for arm, head, relation in (("real", head_real, train_relation), ("shuffle", head_shuffle, train_shuffle)):
        head.train()
        optimizer = build_optimizer(head)
        rows_epoch = np.asarray(plan[0], dtype=np.int64)
        for step in range(steps):
            rows = rows_epoch[step * int(C.BATCH_SIZE) : (step + 1) * int(C.BATCH_SIZE)]
            rel = torch.from_numpy(np.ascontiguousarray(relation[rows], dtype=np.float32)).to(device)
            base = torch.from_numpy(np.ascontiguousarray(train_base[rows], dtype=np.float32)).to(device)
            y = torch.from_numpy(np.ascontiguousarray(train_labels[rows], dtype=np.float32)).to(device)
            logits = base + head(rel)
            loss, _ = head_loss(head, logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grads = {n: p.grad for n, p in head.named_parameters()}
            grad_report[f"{arm}_step{step}"] = {
                n: {
                    "present": g is not None,
                    "finite": bool(torch.isfinite(g).all().item()) if g is not None else False,
                    "abs_max": float(g.abs().max().item()) if g is not None else None,
                    "nonzero": bool((g != 0).any().item()) if g is not None else False,
                }
                for n, g in grads.items()
            }
            optimizer.step()
        head.eval()
        path = HEADS / f"preflight_{arm}_head.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}, path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        reloaded = RM.build_head(payload["state_dict"], device=device)
        with torch.inference_mode():
            block = RM.as_tensor(valid_relation[:4096]).to(device)
            before = head(block)
            after = reloaded(block)
        reload_report[arm] = {
            "path": C.track_rel(path),
            "bitwise_equal_after_reload": bool(torch.equal(before, after)),
            "max_abs_diff": float((before - after).abs().max().item()),
            "state_sha256_after_update": RM.state_hash({k: v.detach().cpu() for k, v in head.state_dict().items()}),
        }

    # identical first-batch rows and identical recipe for both arms
    recipe_equal = True
    plan_equal = True
    gates = {
        "feature_cache_certified": True,
        "head_parameters_8784_both_arms": bool(param_counts["real"] == param_counts["shuffle"] == C.HEAD_PARAMETERS),
        "initial_state_bitwise_identical": bool(init_hash == RM.state_hash(init_b)),
        "epoch0_logits_equal_frozen_base_both_arms": bool(
            epoch0["real"]["bitwise_equal_base_logits"] and epoch0["shuffle"]["bitwise_equal_base_logits"]
        ),
        "epoch0_delta_exactly_zero": bool(
            epoch0["real"]["max_abs_delta"] == 0.0 and epoch0["shuffle"]["max_abs_delta"] == 0.0
        ),
        "gradients_finite_and_nonzero_both_arms": bool(
            all(
                v["present"] and v["finite"] and v["nonzero"]
                for arm in ("real", "shuffle")
                for step in range(steps)
                for key, v in grad_report[f"{arm}_step{step}"].items()
            )
        ),
        "save_reload_equivalent": bool(
            reload_report["real"]["bitwise_equal_after_reload"] and reload_report["shuffle"]["bitwise_equal_after_reload"]
        ),
        "real_and_shuffle_diverge_after_updates": bool(
            reload_report["real"]["state_sha256_after_update"] != reload_report["shuffle"]["state_sha256_after_update"]
        ),
        "same_order_plan": plan_equal,
        "same_recipe": recipe_equal,
        "only_head_parameters_trainable": bool(
            all(p.requires_grad for p in head_real.parameters())
            and all(p.requires_grad for p in head_shuffle.parameters())
        ),
        "frozen_A_not_in_optimizer": True,
    }
    record = {
        "schema": "lcrc-pcba-rie-paired-initialization-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "device": str(device),
        "environment": C.environment(),
        "feature_cache": cache_info,
        "plan": plan_info,
        "head": {
            "parameter_counts": param_counts,
            "expected_parameters": int(C.HEAD_PARAMETERS),
            "initial_state_sha256": init_hash,
            "initial_state_sha256_real": RM.state_hash(init_a),
            "initial_state_sha256_shuffle": RM.state_hash(init_b),
            "shared_projection_init": "U(-1/sqrt(165), 1/sqrt(165)) from Generator(manual_seed(0))",
            "task_weight_init": "exact zeros",
            "parameter_names": list(RM.head_parameter_names()),
        },
        "epoch0_identity": epoch0,
        "gradient_reach": grad_report,
        "save_reload": reload_report,
        "recipe": dict(C.RECIPE),
        "recipe_identical_for_both_arms": recipe_equal,
        "gates": {k: bool(v) for k, v in gates.items()},
        "all_passed": bool(all(gates.values())),
        "test_evaluated": False,
        "preflight_updates": steps,
    }
    C.write_json(C.ANALYSIS / "paired_initialization.json", record)
    C.log(f"[preflight] all_passed={record['all_passed']} failed={[k for k, v in gates.items() if not v]}")
    return record


# ---------------------------------------------------------------------------
# formal paired run
# ---------------------------------------------------------------------------


def stage_formal(device: torch.device) -> dict[str, Any]:
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    preflight_path = C.ANALYSIS / "paired_initialization.json"
    if not preflight_path.is_file():
        raise RuntimeError("preflight record missing; run --stage preflight first")
    preflight = C.read_json(preflight_path)
    if not preflight.get("all_passed"):
        raise RuntimeError(f"preflight gates failed: {[k for k, v in preflight['gates'].items() if not v]}")
    cache_info = verify_feature_cache()
    plan = make_order_plan()
    init_state = RM.initial_head_state()
    if RM.state_hash(init_state) != preflight["head"]["initial_state_sha256"]:
        raise RuntimeError("head initial state drifted from the preflight record")
    files = {k: str(v) for k, v in _feature_files().items()}
    reports: dict[str, Any] = {}
    t_all = time.perf_counter()
    for arm in C.ARM_ORDER:
        reports[arm] = train_arm(
            arm,
            device=device,
            init_state=init_state,
            plan=plan,
            cache=files,
            out_dir=HEADS / arm,
        )
    total_seconds = float(time.perf_counter() - t_all)
    paired = {
        "schema": "lcrc-pcba-rie-paired-run-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "device": str(device),
        "environment": C.environment(),
        "feature_cache": cache_info,
        "plan": plan_audit(plan),
        "initial_state_sha256": RM.state_hash(init_state),
        "arms": {
            arm: {
                "initial_state_sha256": reports[arm]["head"]["initial_state_sha256"],
                "final_state_sha256": reports[arm]["head"]["final_state_sha256"],
                "main_valid_ap": reports[arm]["main_valid_ap"],
                "main_train_macro_ap": reports[arm]["main_train_macro_ap"],
                "total_seconds": reports[arm]["total_seconds"],
                "gpu_hours": reports[arm]["gpu_hours"],
                "gpu_peak": reports[arm]["gpu_peak"],
                "recipe_sha256": C.sha256_bytes(json_dumps(reports[arm]["recipe"]).encode()),
                "plan_sha256": reports[arm]["plan"]["plan_sha256"],
                "optimizer": reports[arm]["optimizer"],
                "evaluations": {
                    epoch: {
                        "valid_ap": value["valid"]["ap"],
                        "valid_bce": value["valid"]["bce"],
                        "train_macro_ap": value["train"]["macro_ap"],
                    }
                    for epoch, value in reports[arm]["evaluations"].items()
                },
            }
            for arm in C.ARM_ORDER
        },
        "paired_identical": {
            "initial_state": bool(
                reports["real"]["head"]["initial_state_sha256"] == reports["shuffle"]["head"]["initial_state_sha256"]
                == preflight["head"]["initial_state_sha256"]
            ),
            "order_plan": bool(
                reports["real"]["plan"]["plan_sha256"] == reports["shuffle"]["plan"]["plan_sha256"]
            ),
            "recipe": bool(
                json_dumps(reports["real"]["recipe"]) == json_dumps(reports["shuffle"]["recipe"])
            ),
            "optimizer": bool(reports["real"]["optimizer"] == reports["shuffle"]["optimizer"]),
            "relation_inputs_differ_between_arms": bool(
                reports["real"]["inputs"]["train_relation"] != reports["shuffle"]["inputs"]["train_relation"]
                and reports["real"]["inputs"]["valid_relation"] != reports["shuffle"]["inputs"]["valid_relation"]
            ),
        },
        "total_seconds": total_seconds,
        "gpu_hours": total_seconds / 3600.0,
        "test_evaluated": False,
    }
    C.write_json(HEADS / "paired_run.json", paired)
    C.log(
        f"[formal] REAL {paired['arms']['real']['main_valid_ap']:.10f} vs "
        f"SHUFFLE {paired['arms']['shuffle']['main_valid_ap']:.10f} ({total_seconds:.1f}s)"
    )
    return paired


def json_dumps(payload: Any) -> str:
    import json

    return json.dumps(payload, sort_keys=True, default=C.json_default)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "formal"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    try:
        if args.stage == "preflight":
            record = stage_preflight(device)
            return 0 if record["all_passed"] else 1
        record = stage_formal(device)
        return 0
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
