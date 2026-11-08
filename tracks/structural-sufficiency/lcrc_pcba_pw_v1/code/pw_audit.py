"""Pair-index build, pre-training checks, cost profile and inference audits.

Subcommands
-----------

``build-pairs``    build the label-free legal-pair index once (CPU, structure only)
``checks``         run the correctness checks on the selected device and write
                   ``analysis/preflight.json`` (a failing check exits 4)
``profile``        measure a real PW training step against a plain W2 step on the
                   same batches and project the 40-epoch GPU-hour cost
``inference``      measure Validation inference time of PW vs W2
``query-entropy``  per-query attention entropy (and its distance to uniform) of a
                   trained checkpoint over the official Validation split

Nothing here trains a formal model.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
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

import pw_common as C  # noqa: E402
import bs_common as BC  # noqa: E402
import bs_model as BSM  # noqa: E402
import bs_train as BT  # noqa: E402
import pw_checks as CHK  # noqa: E402
import pw_pairs as D  # noqa: E402
import pw_model as M  # noqa: E402
import pw_train as T  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402

PREFLIGHT_PATH = C.ANALYSIS / "preflight.json"
PROFILE_PATH = C.ANALYSIS / "profile.json"
INFERENCE_PATH = C.ANALYSIS / "inference_timing.json"
PAIR_BUILD_PATH = C.ANALYSIS / "pair_index_build.json"
ENTROPY_PATH = C.ANALYSIS / "query_entropy.json"
GPU_PATH = C.ANALYSIS / "gpu_schedule.json"


def _abs_path(value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else C.REPO / path


# ---------------------------------------------------------------------------
# build-pairs
# ---------------------------------------------------------------------------


def cmd_build_pairs(args: argparse.Namespace) -> int:
    out = Path(args.out)
    meta = D.build_cache(
        out,
        verify_fraction=float(args.verify_fraction),
        rc_crosscheck_rows=int(args.crosscheck_rows),
    )
    record = {
        "schema": "lcrc-pcba-pw-pair-index-build-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "host_facts": C.host_facts(),
        "cpu_seconds": float(meta["build_seconds"]),
        "labels_read": False,
        "note": (
            "Structure-only build from atom_ptr/bond_ptr/bond_index. The label artifact is "
            "never opened, so the official Test rows enter the index without their targets. "
            "Distances are one BFS per molecule; the training loop never runs BFS."
        ),
        "meta": meta,
    }
    C.write_json(PAIR_BUILD_PATH, record)
    C.log(
        f"[build-pairs] wrote {out} ({meta['n_pairs_stored']} legal pairs, "
        f"{meta['n_pairs_dropped_cross_component']} cross-component pairs dropped) "
        f"in {meta['build_seconds']:.1f}s"
    )
    return 0


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def cmd_checks(args: argparse.Namespace) -> int:
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    task = D.load_task_gated()
    dist = D.load_cache(args.pair_cache)
    counts = np.asarray(dist["pair_counts"])
    with_pairs = np.where(counts > 0)[0]
    without_pairs = np.where(counts == 0)[0]
    if with_pairs.size < int(args.rows):
        raise RuntimeError(f"only {with_pairs.size} rows carry pairs in the cache")
    if without_pairs.size == 0:
        raise RuntimeError("the cache has no molecule without a legal pair; check 11 cannot run")
    plan = BC.SD.make_plan40(task, seed=C.SEED)
    rows = [int(r) for r in np.asarray(plan[0]["batches"][0], dtype=np.int64)]
    if len([r for r in rows if counts[r] > 0]) < 8:
        rows = [int(r) for r in with_pairs[: int(args.rows)]]
    batch = D.collate_pw(task, dist, rows)
    no_pair = D.collate_pw(task, dist, [int(r) for r in without_pairs[: int(args.no_pair_rows)]])
    if device != "cpu":
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        no_pair = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in no_pair.items()}
    grad_batches = []
    for i in range(3):
        g_rows = [int(r) for r in np.asarray(plan[0]["batches"][i], dtype=np.int64)[:64]]
        b = D.collate_pw(task, dist, g_rows)
        grad_batches.append({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()})
    if all(int(b["pair_i"].numel()) == 0 for b in grad_batches):
        raise RuntimeError("the gradient batches carry no legal pairs")
    record = CHK.run_all(
        task.spec,
        task.norm,
        batch,
        rows,
        task=task,
        pairs=dist,
        no_pair_batch=no_pair,
        device=device,
        grad_batches=grad_batches,
    )
    record["real_task_available"] = True
    record["pair_index_available"] = True
    record["batch_rows"] = rows
    record["n_pairs_in_batch"] = int(batch["pair_i"].numel())
    record["no_pair_rows"] = [int(r) for r in without_pairs[: int(args.no_pair_rows)]]
    record["host_facts"] = C.host_facts()
    record["gpu_occupancy"] = C.gpu_occupancy()
    C.write_json(PREFLIGHT_PATH, record)
    for check in record["checks"]:
        C.log(f"[checks] {'PASS' if check['passed'] else 'FAIL'} {check['name']}")
    C.log(f"[checks] {record['n_passed']}/{record['n_checks']} passed on {device}")
    return 0 if record["all_passed"] else 4


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean() * 1e3),
        "p50_ms": float(np.percentile(arr, 50) * 1e3),
        "p90_ms": float(np.percentile(arr, 90) * 1e3),
        "min_ms": float(arr.min() * 1e3),
        "max_ms": float(arr.max() * 1e3),
        "steps_per_second": float(1.0 / arr.mean()),
    }


def _time_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    task: Any,
    dist: Mapping[str, Any] | None,
    rows_list: Sequence[np.ndarray],
    device: torch.device,
    *,
    steps: int,
    warmup: int,
    global_step: int,
) -> dict[str, Any]:
    stager = PinnedStager(device, depth=2)
    clip = float(C.RECIPE["gradient_clip"])
    model.train()
    times: list[float] = []
    collate_times: list[float] = []
    for i in range(int(warmup) + int(steps)):
        rows = rows_list[i % len(rows_list)]
        t0 = time.perf_counter()
        raw = BC.PC.collate(task.cache, rows.tolist()) if dist is None else D.collate_pw(task, dist, rows.tolist())
        t1 = time.perf_counter()
        batch = stager.batch(raw)
        y = stager.labels(task.labels[rows])
        BT.train_step(model, optimizer, batch, y, global_step=int(global_step) + i, clip=clip)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        if i >= int(warmup):
            times.append(float(t2 - t1))
            collate_times.append(float(t1 - t0))
    stager.release()
    return {
        "step": _percentiles(times),
        "collate_cpu": _percentiles(collate_times),
        "measured_steps": int(steps),
        "warmup_steps": int(warmup),
    }


def cmd_profile(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    task = D.load_task_gated()
    dist = D.load_cache(args.pair_cache, require_full=True)
    plan = BC.SD.make_plan40(task, seed=C.SEED)
    rows_list = [np.asarray(plan[0]["batches"][i], dtype=np.int64) for i in range(int(args.batches))]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    record: dict[str, Any] = {
        "schema": C.SCHEMA_PROFILE,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "threads": {"torch": torch.get_num_threads(), "omp_env": os.environ.get("OMP_NUM_THREADS")},
        "batches": int(args.batches),
        "steps": int(args.steps),
        "warmup": int(args.warmup),
        "batch_rows": [int(r.size) for r in rows_list],
        "pairs_per_batch": [int(np.asarray(dist["pair_counts"])[r].sum()) for r in rows_list],
        "runs": {},
    }

    w2, _ = BSM.build_model(C.BASE_MODEL_KEY, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
    w2_opt, _ = BSM.make_optimizer(w2)
    record["runs"]["W2"] = _time_steps(
        w2, w2_opt, task, None, rows_list, device, steps=int(args.steps), warmup=int(args.warmup), global_step=0
    )
    del w2, w2_opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model, audit = M.build_model(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
    opt, _ = BSM.make_optimizer(model)
    record["runs"][C.MODEL_KEY] = _time_steps(
        model, opt, task, dist, rows_list, device, steps=int(args.steps), warmup=int(args.warmup), global_step=0
    )
    record["runs"][C.MODEL_KEY]["parameters"] = int(audit["total_parameters"])
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "cuda":
        record["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        record["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))

    valid_seconds = float(args.valid_seconds)
    train_eval_seconds = float(args.train_eval_seconds)
    steps_per_epoch = C.STEPS_PER_EPOCH
    for key, run in record["runs"].items():
        step_s = float(run["step"]["p50_ms"]) / 1e3
        train_s = step_s * steps_per_epoch * C.TOTAL_EPOCHS
        total = train_s + valid_seconds * C.TOTAL_EPOCHS + train_eval_seconds
        run["projection"] = {
            "step_seconds_p50": step_s,
            "train_seconds_40_epochs": train_s,
            "valid_seconds_40": valid_seconds * C.TOTAL_EPOCHS,
            "train_eval_seconds": train_eval_seconds,
            "total_seconds": total,
            "total_gpu_hours": total / 3600.0,
            "budget_target_hours": float(C.BUDGET["gpu_target_hours"]),
            "budget_hard_hours": float(C.BUDGET["gpu_hard_hours"]),
            "within_target": bool(total / 3600.0 <= float(C.BUDGET["gpu_target_hours"])),
            "within_hard": bool(total / 3600.0 <= float(C.BUDGET["gpu_hard_hours"])),
        }
    base = float(record["runs"]["W2"]["step"]["p50_ms"])
    for key, run in record["runs"].items():
        run["step_overhead_vs_w2"] = float(run["step"]["p50_ms"]) / base - 1.0
    record["w2_reference_step_ms_p50"] = base
    record["within_budget"] = bool(record["runs"][C.MODEL_KEY]["projection"]["within_hard"])
    C.write_json(PROFILE_PATH, record)
    for key, run in record["runs"].items():
        C.log(
            f"[profile] {key} step p50 {run['step']['p50_ms']:.2f} ms "
            f"({run['step_overhead_vs_w2'] * 100:+.1f}% vs W2) "
            f"projected {run['projection']['total_gpu_hours']:.3f} GPU-h "
            f"(target {C.BUDGET['gpu_target_hours']}, hard {C.BUDGET['gpu_hard_hours']})"
        )
    return 0


# ---------------------------------------------------------------------------
# inference timing
# ---------------------------------------------------------------------------


def cmd_inference(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    task = D.load_task_gated()
    dist = D.load_cache(args.pair_cache, require_full=True)
    rows = np.asarray(task.valid_rows, dtype=np.int64)[: int(args.rows)]
    record: dict[str, Any] = {
        "schema": "lcrc-pcba-pw-inference-timing-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "rows": int(rows.size),
        "batch_size": 256,
        "runs": {},
    }
    w2, w2_audit = BSM.build_model(C.BASE_MODEL_KEY, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
    model, audit = M.build_model(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
    for key, built, use_pairs in (("W2", w2, False), (C.MODEL_KEY, model, True)):
        built.eval()
        warm_rows = rows[: min(256, rows.size)]
        if use_pairs:
            T.predict_logits(built, task, dist, warm_rows, device)
        else:
            _predict_plain_w2(built, task, warm_rows, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        if use_pairs:
            T.predict_logits(built, task, dist, rows, device)
        else:
            _predict_plain_w2(built, task, rows, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - t0
        record["runs"][key] = {
            "seconds": float(seconds),
            "rows_per_second": float(rows.size / seconds),
            "milliseconds_per_1k_rows": float(seconds / rows.size * 1e3 * 1e3),
            "parameters": int((w2_audit if not use_pairs else audit)["total_parameters"]),
        }
        record["runs"][key]["relative_to_w2"] = 1.0 if not use_pairs else None
    base = record["runs"]["W2"]["seconds"]
    for key, run in record["runs"].items():
        run["relative_to_w2"] = float(run["seconds"] / base)
    C.write_json(INFERENCE_PATH, record)
    for key, run in record["runs"].items():
        C.log(f"[inference] {key} {run['rows_per_second']:.1f} rows/s ({run['relative_to_w2']:.3f}x W2)")
    return 0


def _predict_plain_w2(model, task, rows: np.ndarray, device) -> np.ndarray:
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    out = []
    with torch.inference_mode():
        for start in range(0, rows.size, 256):
            chunk = rows[start : start + 256]
            raw = BC.PC.collate(task.cache, chunk.tolist())
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
            out.append(model(batch).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, C.N_TASKS), dtype=np.float32)


# ---------------------------------------------------------------------------
# query entropy
# ---------------------------------------------------------------------------


def _nvidia_smi(query: str) -> list[list[str]]:
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001  (diagnostic only)
        return [[f"ERROR: {type(exc).__name__}: {exc}"]]
    if out.returncode != 0:
        return [[f"ERROR: exit {out.returncode}: {(out.stderr or '').strip()[:200]}"]]
    return [[c.strip() for c in line.split(",")] for line in out.stdout.splitlines() if line.strip()]


def cmd_gpu_schedule(args: argparse.Namespace) -> int:
    """Record which GPU is free *before* claiming one, and the claim itself."""
    gpus = _nvidia_smi("gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu")
    apps = _nvidia_smi("compute-apps=pid,gpu_uuid,process_name,used_memory")
    by_uuid: dict[str, list[dict[str, Any]]] = {}
    for row in apps:
        if len(row) < 4 or row[0].startswith("ERROR"):
            continue
        by_uuid.setdefault(str(row[1]), []).append(
            {"pid": row[0], "process_name": row[2], "used_memory_mib": row[3]}
        )
    table: list[dict[str, Any]] = []
    for row in gpus:
        if len(row) < 7 or row[0].startswith("ERROR"):
            table.append({"raw": row})
            continue
        index, uuid = int(row[0]), str(row[1])
        total_mib, used_mib, free_mib = int(row[3]), int(row[4]), int(row[5])
        table.append(
            {
                "index": index,
                "uuid": uuid,
                "name": row[2],
                "memory_total_mib": total_mib,
                "memory_used_mib": used_mib,
                "memory_free_mib": free_mib,
                "utilization_gpu_percent": row[6],
                "compute_apps": by_uuid.get(uuid, []),
                "has_compute_apps": bool(by_uuid.get(uuid)),
            }
        )
    # pool -> physical GPU mapping used by this round
    gpu1 = next((g for g in table if g.get("index") == 1), None)
    gpu0 = next((g for g in table if g.get("index") == 0), None)
    gpu0_safe = bool(
        gpu0
        and not gpu0.get("has_compute_apps")
        and int(gpu0.get("memory_used_mib", 10 ** 9)) < 1024
    )
    decision = {
        "pw_training_pool": "res-gpu1",
        "pw_training_gpu_index": 1,
        "pw_training_gpu_idle_at_check": bool(gpu1 and not gpu1.get("has_compute_apps")),
        "res_gpu0_pool_used": False,
        "res_gpu0_safe_to_use": gpu0_safe,
        "reason": (
            "PW trains on res-gpu1 only after rc-blind released it. res-gpu0 is left alone: "
            "it is only safe when it shows no compute processes and <1 GiB used, and even then "
            "this round has no need for a second GPU."
        ),
        "no_preemption": True,
    }
    record = {
        "schema": "lcrc-pcba-pw-gpu-schedule-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "host_facts": C.host_facts(),
        "gpu_occupancy": C.gpu_occupancy(),
        "nvidia_smi_gpus": table,
        "nvidia_smi_compute_apps_raw": apps,
        "decision": decision,
    }
    C.write_json(GPU_PATH, record)
    C.log(
        f"gpu1 idle={decision['pw_training_gpu_idle_at_check']} gpu0 safe={gpu0_safe} -> {GPU_PATH}"
    )
    return 0


@torch.no_grad()
def cmd_query_entropy(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    task = D.load_task_gated()
    dist = D.load_cache(args.pair_cache, require_full=True)
    checkpoint = _abs_path(args.checkpoint)
    if not checkpoint.is_file():
        raise RuntimeError(f"checkpoint missing: {checkpoint}")
    sha = C.sha256_file(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("model") != C.MODEL_KEY:
        raise RuntimeError(f"checkpoint model {payload.get('model')} != {C.MODEL_KEY}")
    model, audit = M.build_model(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch: missing={missing} unexpected={unexpected}")
    model.eval()

    rows = np.asarray(task.valid_rows, dtype=np.int64)[: int(args.rows)]
    n_queries = C.N_QUERIES
    sum_entropy = np.zeros(n_queries, dtype=np.float64)
    sum_logn = np.zeros(n_queries, dtype=np.float64)
    sum_ratio = np.zeros(n_queries, dtype=np.float64)
    sum_max_weight = np.zeros(n_queries, dtype=np.float64)
    sum_effective = np.zeros(n_queries, dtype=np.float64)
    min_ratio = np.full(n_queries, np.inf, dtype=np.float64)
    max_ratio = np.full(n_queries, -np.inf, dtype=np.float64)
    n_graphs = 0
    n_pairs_total = 0
    n_ratio = np.zeros(n_queries, dtype=np.float64)
    batch_size = int(args.batch_size)
    for start in range(0, rows.size, batch_size):
        chunk = rows[start : start + batch_size].tolist()
        batch = D.collate_pw(task, dist, chunk)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        pair_graph = batch["pair_graph"].to(torch.int64)
        counts = batch["pair_counts"].to(torch.int64)
        keep = counts > 0
        if int(keep.sum().item()) == 0:
            continue
        _a, _b, _scal, h = model.encode_h(batch)
        token = model.branch.pair_tokens(
            h,
            batch["pair_i"].to(torch.int64),
            batch["pair_j"].to(torch.int64),
            batch["pair_bucket"].to(torch.int64),
        )
        _z, attn, _score = model.branch.segment_attention(token, pair_graph, int(batch["n_graphs"]))
        ent = torch.zeros((int(batch["n_graphs"]), n_queries), device=attn.device, dtype=attn.dtype).index_add(
            0, pair_graph, -(attn * torch.log(attn.clamp_min(1e-30)))
        )
        mx = torch.full((int(batch["n_graphs"]), n_queries), -1.0, device=attn.device, dtype=attn.dtype).scatter_reduce(
            0, pair_graph.unsqueeze(1).expand(-1, n_queries), attn, reduce="amax", include_self=True
        )
        ent = ent[keep].double().cpu().numpy()
        mx = mx[keep].double().cpu().numpy()
        c = counts[keep].double().cpu().numpy()
        logn = np.log(c)[:, None]
        ratio = np.where(c[:, None] > 1, ent / np.maximum(logn, 1e-12), np.nan)
        sum_entropy += ent.sum(axis=0)
        sum_logn += (logn * np.ones_like(ent)).sum(axis=0)
        sum_max_weight += mx.sum(axis=0)
        sum_effective += np.exp(ent).sum(axis=0)
        finite = np.isfinite(ratio)
        sum_ratio += np.where(finite, ratio, 0.0).sum(axis=0)
        min_ratio = np.minimum(min_ratio, np.where(finite, ratio, np.inf).min(axis=0))
        max_ratio = np.maximum(max_ratio, np.where(finite, ratio, -np.inf).max(axis=0))
        n_ratio += finite.sum(axis=0)
        n_graphs += int(keep.sum().item())
        n_pairs_total += int(batch["pair_i"].numel())

    per_query = []
    for k in range(n_queries):
        count = float(n_ratio[k]) if n_ratio[k] else 1.0
        per_query.append(
            {
                "query": int(k),
                "mean_entropy_nats": float(sum_entropy[k] / max(n_graphs, 1)),
                "mean_uniform_entropy_nats": float(sum_logn[k] / max(n_graphs, 1)),
                "mean_normalised_entropy": float(sum_ratio[k] / count),
                "min_normalised_entropy": float(min_ratio[k]) if np.isfinite(min_ratio[k]) else None,
                "max_normalised_entropy": float(max_ratio[k]) if np.isfinite(max_ratio[k]) else None,
                "mean_effective_pairs": float(sum_effective[k] / max(n_graphs, 1)),
                "mean_max_attention_weight": float(sum_max_weight[k] / max(n_graphs, 1)),
                "near_uniform_095": bool(float(sum_ratio[k] / count) > 0.95),
            }
        )
    ratios = [q["mean_normalised_entropy"] for q in per_query]
    record = {
        "schema": "lcrc-pcba-pw-query-entropy-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "checkpoint": C.track_rel(checkpoint),
        "checkpoint_sha256": sha,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_kind": payload.get("checkpoint_kind"),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "split": "official Validation rows",
        "rows_requested": int(args.rows),
        "rows_with_pairs": int(n_graphs),
        "n_pairs_total": int(n_pairs_total),
        "n_queries": int(n_queries),
        "definition": (
            "H_k(g) = -sum_ij a_kij log a_kij over the legal pairs of molecule g; the uniform reference is "
            "log n_pairs(g); the normalised entropy H / log n is reported for molecules with n > 1. "
            "'mean_effective_pairs' = mean exp(H_k), the effective number of pairs a query reads."
        ),
        "queries": per_query,
        "mean_normalised_entropy_over_queries": float(sum(ratios) / len(ratios)),
        "min_normalised_entropy_over_queries": float(min(ratios)),
        "max_normalised_entropy_over_queries": float(max(ratios)),
        "all_queries_near_uniform": bool(all(q["near_uniform_095"] for q in per_query)),
        "any_query_collapsed": bool(any(q["mean_normalised_entropy"] < 0.5 for q in per_query)),
        "note": (
            "Inference-only diagnostic: no training, no query re-initialisation, no hyperparameter change. "
            "A query at 1.0 is exactly uniform over the molecule's pairs; a query at 0.0 has collapsed onto "
            "a single pair."
        ),
        "parameters": int(audit["total_parameters"]),
    }
    C.write_json(ENTROPY_PATH, record)
    for q in per_query:
        C.log(
            f"[query-entropy] q{q['query']:02d} H {q['mean_entropy_nats']:.4f} / uniform "
            f"{q['mean_uniform_entropy_nats']:.4f} = {q['mean_normalised_entropy']:.4f} "
            f"eff_pairs {q['mean_effective_pairs']:.2f} max_w {q['mean_max_attention_weight']:.4f}"
        )
    C.log(
        f"[query-entropy] mean normalised entropy {record['mean_normalised_entropy_over_queries']:.4f} "
        f"all near uniform: {record['all_queries_near_uniform']}"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    bp = sub.add_parser("build-pairs")
    bp.add_argument("--out", default=str(C.PAIR_CACHE_DIR))
    bp.add_argument("--verify-fraction", type=float, default=0.002)
    bp.add_argument("--crosscheck-rows", type=int, default=2000)

    ck = sub.add_parser("checks")
    ck.add_argument("--device", default="cuda")
    ck.add_argument("--pair-cache", default=str(C.PAIR_CACHE_DIR))
    ck.add_argument("--rows", type=int, default=128)
    ck.add_argument("--no-pair-rows", type=int, default=6)

    pr = sub.add_parser("profile")
    pr.add_argument("--device", default="cuda")
    pr.add_argument("--pair-cache", default=str(C.PAIR_CACHE_DIR))
    pr.add_argument("--batches", type=int, default=4)
    pr.add_argument("--steps", type=int, default=25)
    pr.add_argument("--warmup", type=int, default=8)
    pr.add_argument("--valid-seconds", type=float, default=8.0)
    pr.add_argument("--train-eval-seconds", type=float, default=60.0)

    inf = sub.add_parser("inference")
    inf.add_argument("--device", default="cuda")
    inf.add_argument("--pair-cache", default=str(C.PAIR_CACHE_DIR))
    inf.add_argument("--rows", type=int, default=8192)

    sub.add_parser("gpu")

    qe = sub.add_parser("query-entropy")
    qe.add_argument("--device", default="cuda")
    qe.add_argument("--pair-cache", default=str(C.PAIR_CACHE_DIR))
    qe.add_argument("--checkpoint", required=True)
    qe.add_argument("--rows", type=int, default=43793)
    qe.add_argument("--batch-size", type=int, default=256)

    args = parser.parse_args(argv)
    if args.mode == "build-pairs":
        return cmd_build_pairs(args)
    if args.mode == "checks":
        return cmd_checks(args)
    if args.mode == "profile":
        return cmd_profile(args)
    if args.mode == "inference":
        return cmd_inference(args)
    if args.mode == "gpu":
        return cmd_gpu_schedule(args)
    if args.mode == "query-entropy":
        return cmd_query_entropy(args)
    raise KeyError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
