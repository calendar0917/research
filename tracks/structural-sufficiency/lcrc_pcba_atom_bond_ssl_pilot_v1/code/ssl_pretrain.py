"""Formal masked atom/bond SSL pretraining on official-Train structures only.

Frozen recipe (``PROTOCOL.md`` §5): Adam, lr 1e-3, wd 1e-5, betas (0.9, 0.999),
eps 1e-8, clip 5.0, batch 128 molecules, 10,000 updates, FP32, dropout 0,
15% atom mask + 15% bond mask, ``L_SSL = CE_atom + CE_bond``.

The exported weights are always the state after update 10,000.  Probe CE is a
diagnostic, never a checkpoint selector.

This program never reads the assay-label artifact: the structure loader opens
only the memory-mapped structure arrays and the split row ids, and the
label-access ledger is written into ``run.json``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import threading
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

import ssl_common as C  # noqa: E402
import ssl_data as D  # noqa: E402
import ssl_model as M  # noqa: E402

CURVE_FIELDS = [
    "update",
    "wall_seconds",
    "train_loss",
    "train_ce_atom",
    "train_ce_bond",
    "train_atom_accuracy",
    "train_bond_accuracy",
    "probe_atom_ce",
    "probe_bond_ce",
    "probe_atom_accuracy",
    "probe_bond_accuracy",
    "probe_atom_prior_ce",
    "probe_bond_prior_ce",
    "probe_atom_prior_ce_masked_targets",
    "probe_bond_prior_ce_masked_targets",
    "probe_masked_atoms",
    "probe_masked_bonds",
    "probe_atom_class_counts",
    "probe_bond_class_counts",
    "grad_norm_mean",
    "grad_norm_max",
    "clip_events",
    "step_ms_mean",
    "step_ms_p50",
    "step_ms_p90",
    "steps_per_second",
    "learning_rate",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "encoder_l2_delta_from_init",
    "notes",
]


class Watchdog:
    def __init__(self, hard_seconds: float):
        self.hard_seconds = float(hard_seconds)
        self.stop = threading.Event()
        self._t0 = time.perf_counter()
        self._timer = threading.Timer(self.hard_seconds, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        self.stop.set()

    def cancel(self) -> None:
        self._timer.cancel()


# ---------------------------------------------------------------------------
# device transfer of the only tensors the network may see
# ---------------------------------------------------------------------------


def _move_view(view: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: view[k].to(device, non_blocking=False) for k in D.MODEL_INPUT_KEYS}


def _probe_eval(
    st: D.Structure,
    model: M.PretextNet,
    priors: Mapping[str, Any],
    device: torch.device,
    *,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Fixed ``visit_index=0`` pretext evaluation on the 4096 Train-Probe rows."""
    model.eval()
    ce_atom_sum = 0.0
    ce_bond_sum = 0.0
    atom_correct = 0
    bond_correct = 0
    atom_n = 0
    bond_n = 0
    atom_targets: list[np.ndarray] = []
    bond_targets: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(st.probe_rows.size), int(batch_size)):
            rows = st.probe_rows[start : start + int(batch_size)]
            pb = D.build_pretext_batch(st, rows, C.PROBE_VISIT_INDEX)
            out = model(
                _move_view(pb.atom_view, device),
                _move_view(pb.bond_view, device),
                pb.atom_ids.to(device),
                pb.bond_ids.to(device),
            )
            if pb.atom_targets.numel():
                logits = out["atom_logits"]
                tgt = pb.atom_targets.to(device)
                ce = torch.nn.functional.cross_entropy(logits, tgt, reduction="sum")
                ce_atom_sum += float(ce)
                atom_correct += int((logits.argmax(1) == tgt).sum())
                atom_n += int(tgt.numel())
                atom_targets.append(pb.atom_targets.numpy().astype(np.int64))
            if pb.bond_targets.numel():
                logits = out["bond_logits"]
                tgt = pb.bond_targets.to(device)
                ce = torch.nn.functional.cross_entropy(logits, tgt, reduction="sum")
                ce_bond_sum += float(ce)
                bond_correct += int((logits.argmax(1) == tgt).sum())
                bond_n += int(tgt.numel())
                bond_targets.append(pb.bond_targets.numpy().astype(np.int64))
    model.train()
    at = np.concatenate(atom_targets) if atom_targets else np.zeros(0, np.int64)
    bt = np.concatenate(bond_targets) if bond_targets else np.zeros(0, np.int64)
    atom_hist = np.bincount(at, minlength=C.N_ATOM_CLASSES).astype(np.int64)
    bond_hist = np.bincount(bt, minlength=C.N_BOND_CLASSES).astype(np.int64)
    atom_ce = float(ce_atom_sum / max(atom_n, 1))
    bond_ce = float(ce_bond_sum / max(bond_n, 1))
    return {
        "probe_atom_ce": atom_ce,
        "probe_bond_ce": bond_ce,
        "probe_atom_accuracy": float(atom_correct / max(atom_n, 1)),
        "probe_bond_accuracy": float(bond_correct / max(bond_n, 1)),
        "probe_atom_prior_ce": float(priors["atom"]["prior_ce_nats"]),
        "probe_bond_prior_ce": float(priors["bond"]["prior_ce_nats"]),
        "probe_atom_prior_ce_masked_targets": D.prior_ce(at, priors["atom"]["frequencies"]),
        "probe_bond_prior_ce_masked_targets": D.prior_ce(bt, priors["bond"]["frequencies"]),
        "probe_masked_atoms": int(atom_n),
        "probe_masked_bonds": int(bond_n),
        "probe_atom_class_counts": [int(x) for x in atom_hist],
        "probe_bond_class_counts": [int(x) for x in bond_hist],
        "probe_atom_majority_accuracy": float(priors["atom"]["majority_fraction"]),
        "probe_bond_majority_accuracy": float(priors["bond"]["majority_fraction"]),
        "probe_rows": int(st.probe_rows.size),
        "probe_visit_index": int(C.PROBE_VISIT_INDEX),
        "probe_rows_sha256": C.sha256_int64(st.probe_rows),
    }


# ---------------------------------------------------------------------------
# shard bookkeeping
# ---------------------------------------------------------------------------


def _new_shuffle_state(gen: torch.Generator, fit_size: int) -> dict[str, Any]:
    perm = torch.randperm(int(fit_size), generator=gen)
    return {
        "pass_index": 0,
        "offset": 0,
        "generator_state": gen.get_state().clone(),
        "perm": perm,
        "perm_sha256": C.sha256_int64(perm.numpy()),
        "n_full_batches": int(int(fit_size) // int(C.PRETRAIN["batch_size"])),
        "dropped_tail_rows": int(int(fit_size) % int(C.PRETRAIN["batch_size"])),
    }


def _next_batch(shuffle: dict[str, Any], gen: torch.Generator, fit_rows: np.ndarray) -> np.ndarray:
    nb = int(shuffle["n_full_batches"])
    while int(shuffle["offset"]) >= nb:
        perm = torch.randperm(int(fit_rows.size), generator=gen)
        shuffle["pass_index"] = int(shuffle["pass_index"]) + 1
        shuffle["offset"] = 0
        shuffle["perm"] = perm
        shuffle["perm_sha256"] = C.sha256_int64(perm.numpy())
        shuffle["generator_state"] = gen.get_state().clone()
    start = int(shuffle["offset"]) * int(C.PRETRAIN["batch_size"])
    rows = fit_rows[shuffle["perm"][start : start + int(C.PRETRAIN["batch_size"])].numpy()]
    shuffle["offset"] = int(shuffle["offset"]) + 1
    shuffle["generator_state"] = gen.get_state().clone()
    return np.asarray(rows, dtype=np.int64)


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------


def _atomic_save(payload: Mapping[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    os.replace(tmp, path)
    return C.sha256_file(path)


def _optimizer_state_cpu(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    import copy

    state = optimizer.state_dict()
    out: dict[str, Any] = {"state": {}, "param_groups": copy.deepcopy(state["param_groups"])}
    for pid, values in state["state"].items():
        out["state"][int(pid)] = {
            k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
            for k, v in values.items()
        }
    return out


def _restore_rng(rng_state: Mapping[str, Any]) -> dict[str, Any]:
    import random

    random.setstate(rng_state["python_random"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"])
    if torch.cuda.is_available() and rng_state.get("torch_cuda"):
        states = [s.detach().cpu() for s in rng_state["torch_cuda"]]
        if len(states) != torch.cuda.device_count():
            raise RuntimeError("RNG state does not match the visible CUDA device count")
        torch.cuda.set_rng_state_all(states)
    return C.rng_state_hashes(C.capture_rng_state())


# ---------------------------------------------------------------------------
# pretraining
# ---------------------------------------------------------------------------


def pretrain(
    st: D.Structure,
    *,
    out_dir: Path,
    device: torch.device,
    updates: int,
    soft_seconds: float,
    hard_seconds: float,
    resume: Mapping[str, Any] | None = None,
    checkpoint_every: int | None = None,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    started_utc = C.utcnow()
    out_dir.mkdir(parents=True, exist_ok=True)
    updates_total = int(C.PRETRAIN["updates"])
    if int(updates) <= 0 or int(updates) > updates_total:
        raise ValueError(f"updates must be in 1..{updates_total}")
    checkpoint_every = int(
        C.PRETRAIN["checkpoint_every_updates"] if checkpoint_every is None else checkpoint_every
    )
    eval_every = int(C.PRETRAIN["eval_every_updates"])

    norm = D.load_train_norm(st)
    priors = D.class_priors(st)
    C.seed_everything(C.SSL_SEED)
    model, pretext_audit = M.build_pretext(st.spec, norm)
    model.to(device)
    initial_encoder_state = M.encoder_state(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(C.PRETRAIN["learning_rate"]),
        betas=tuple(C.PRETRAIN["adam_betas"]),
        eps=float(C.PRETRAIN["adam_eps"]),
        weight_decay=float(C.PRETRAIN["weight_decay"]),
    )
    optimizer_audit = {
        "optimizer": "Adam",
        "learning_rate": float(C.PRETRAIN["learning_rate"]),
        "weight_decay": float(C.PRETRAIN["weight_decay"]),
        "betas": list(C.PRETRAIN["adam_betas"]),
        "eps": float(C.PRETRAIN["adam_eps"]),
        "momentum_buffers_initialised": "zeros (fresh optimizer)",
        "parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    gen = torch.Generator().manual_seed(int(C.SHUFFLE_SEED))
    shuffle = _new_shuffle_state(gen, int(st.fit_rows.size))
    start_update = 0
    curve: list[dict[str, Any]] = []
    rng_start_hashes = C.rng_state_hashes(C.capture_rng_state())
    resumed = False
    if resume is not None:
        payload = resume
        if payload.get("schema") != C.SCHEMA_CHECKPOINT:
            raise RuntimeError("resume payload schema mismatch")
        if dict(payload.get("pretrain_config") or {}) != dict(C.PRETRAIN):
            raise RuntimeError("resume payload was produced under a different pretrain recipe")
        if int(payload.get("updates_total", -1)) != updates_total:
            raise RuntimeError("resume payload has a different update budget")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        gen.set_state(payload["shuffle"]["generator_state"].clone())
        shuffle = {
            "pass_index": int(payload["shuffle"]["pass_index"]),
            "offset": int(payload["shuffle"]["offset"]),
            "generator_state": payload["shuffle"]["generator_state"].clone(),
            "perm": payload["shuffle"]["perm"].clone(),
            "perm_sha256": str(payload["shuffle"]["perm_sha256"]),
            "n_full_batches": int(payload["shuffle"]["n_full_batches"]),
            "dropped_tail_rows": int(payload["shuffle"]["dropped_tail_rows"]),
        }
        if shuffle["perm_sha256"] != C.sha256_int64(shuffle["perm"].numpy()):
            raise RuntimeError("resume permutation hash mismatch")
        if C.sha256_int64(st.probe_rows) != str(payload.get("probe_rows_sha256")):
            raise RuntimeError("resume probe partition mismatch")
        start_update = int(payload["update"])
        curve = list(payload.get("curve") or [])
        restored = _restore_rng(payload["rng_state"])
        if restored["combined_sha256"] != (payload.get("rng_state_hashes") or {}).get("combined_sha256"):
            raise RuntimeError("restored RNG hashes do not match the checkpoint")
        resumed = True
        C.log(f"[ssl] resumed from update {start_update}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    curve_path = out_dir / "curve.csv"
    ckpt_latest = out_dir / "checkpoint_latest.pt"
    watchdog = Watchdog(hard_seconds)
    clip = float(C.PRETRAIN["gradient_clip_norm"])

    def write_curve() -> str:
        with open(curve_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CURVE_FIELDS)
            writer.writeheader()
            for row in curve:
                writer.writerow({k: _cell(row.get(k)) for k in CURVE_FIELDS})
        return C.sha256_file(curve_path)

    def checkpoint_payload(kind: str, update: int) -> dict[str, Any]:
        return {
            "schema": C.SCHEMA_CHECKPOINT,
            "round_id": C.ROUND_ID,
            "kind": kind,
            "dataset": C.DATASET,
            "seed": int(C.SSL_SEED),
            "update": int(update),
            "updates_total": int(updates_total),
            "pretrain_config": dict(C.PRETRAIN),
            "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "encoder_state": M.encoder_state(model),
            "encoder_fingerprint": M.encoder_fingerprint(model),
            "initial_encoder_fingerprint": M.encoder_fingerprint_from_state(initial_encoder_state),
            "optimizer": optimizer_audit,
            "optimizer_state": _optimizer_state_cpu(optimizer),
            "shuffle": {
                "pass_index": int(shuffle["pass_index"]),
                "offset": int(shuffle["offset"]),
                "generator_state": shuffle["generator_state"].clone(),
                "perm": shuffle["perm"].clone(),
                "perm_sha256": str(shuffle["perm_sha256"]),
                "n_full_batches": int(shuffle["n_full_batches"]),
                "dropped_tail_rows": int(shuffle["dropped_tail_rows"]),
                "shuffle_seed": int(C.SHUFFLE_SEED),
            },
            "rng_state": C.capture_rng_state(),
            "rng_state_hashes": C.rng_state_hashes(C.capture_rng_state()),
            "probe_rows_sha256": C.sha256_int64(st.probe_rows),
            "fit_rows_sha256": C.sha256_int64(st.fit_rows),
            "split_partition_sha256": st.split_partition_sha256,
            "feature_spec_hash": st.spec.hash(),
            "norm_fingerprint": norm["norm_fingerprint"],
            "curve": curve,
            "pretext_audit": pretext_audit,
            "git_commit": C.git_commit(),
            "environment": C.environment(),
            "created_utc": C.utcnow(),
            "test_evaluated": False,
        }

    rows_records: list[dict[str, Any]] = []
    window_loss = 0.0
    window_ce_atom = 0.0
    window_ce_bond = 0.0
    window_atom_correct = 0
    window_atom_n = 0
    window_bond_correct = 0
    window_bond_n = 0
    window_grad: list[float] = []
    window_step_ms: list[float] = []
    window_clip = 0
    window_steps = 0
    non_finite_events = 0
    stop_reason = "RUNNING"
    model.train()

    def record_eval(update: int) -> None:
        probe = _probe_eval(st, model, priors, device)
        encoder_delta = M.parameter_change(initial_encoder_state, M.encoder_state(model))
        row = {
            "update": int(update),
            "wall_seconds": float(time.perf_counter() - t_start),
            "train_loss": float(window_loss / max(window_steps, 1)),
            "train_ce_atom": float(window_ce_atom / max(window_steps, 1)),
            "train_ce_bond": float(window_ce_bond / max(window_steps, 1)),
            "train_atom_accuracy": float(window_atom_correct / max(window_atom_n, 1)),
            "train_bond_accuracy": float(window_bond_correct / max(window_bond_n, 1)),
            "probe_atom_ce": probe["probe_atom_ce"],
            "probe_bond_ce": probe["probe_bond_ce"],
            "probe_atom_accuracy": probe["probe_atom_accuracy"],
            "probe_bond_accuracy": probe["probe_bond_accuracy"],
            "probe_atom_prior_ce": probe["probe_atom_prior_ce"],
            "probe_bond_prior_ce": probe["probe_bond_prior_ce"],
            "probe_atom_prior_ce_masked_targets": probe["probe_atom_prior_ce_masked_targets"],
            "probe_bond_prior_ce_masked_targets": probe["probe_bond_prior_ce_masked_targets"],
            "probe_masked_atoms": probe["probe_masked_atoms"],
            "probe_masked_bonds": probe["probe_masked_bonds"],
            "probe_atom_class_counts": ";".join(str(x) for x in probe["probe_atom_class_counts"]),
            "probe_bond_class_counts": ";".join(str(x) for x in probe["probe_bond_class_counts"]),
            "grad_norm_mean": float(np.mean(window_grad)) if window_grad else float("nan"),
            "grad_norm_max": float(np.max(window_grad)) if window_grad else float("nan"),
            "clip_events": int(window_clip),
            "step_ms_mean": float(np.mean(window_step_ms) * 1e3) if window_step_ms else float("nan"),
            "step_ms_p50": float(np.percentile(window_step_ms, 50) * 1e3) if window_step_ms else float("nan"),
            "step_ms_p90": float(np.percentile(window_step_ms, 90) * 1e3) if window_step_ms else float("nan"),
            "steps_per_second": float(window_steps / max(sum(window_step_ms), 1e-9)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
            "encoder_l2_delta_from_init": float(encoder_delta["l2_delta"]),
            "notes": "",
        }
        curve.append(row)
        write_curve()
        C.log(
            f"[ssl] update {update}: loss {row['train_loss']:.4f} "
            f"probeAtomCE {row['probe_atom_ce']:.4f} (prior {row['probe_atom_prior_ce']:.4f}) "
            f"probeBondCE {row['probe_bond_ce']:.4f} (prior {row['probe_bond_prior_ce']:.4f}) "
            f"atomAcc {row['probe_atom_accuracy']:.3f} bondAcc {row['probe_bond_accuracy']:.3f} "
            f"{row['step_ms_mean']:.2f} ms/step"
        )
        reset_window()

    def reset_window() -> None:
        nonlocal window_loss, window_ce_atom, window_ce_bond, window_atom_correct, window_atom_n
        nonlocal window_bond_correct, window_bond_n, window_grad, window_step_ms, window_clip, window_steps
        window_loss = 0.0
        window_ce_atom = 0.0
        window_ce_bond = 0.0
        window_atom_correct = 0
        window_atom_n = 0
        window_bond_correct = 0
        window_bond_n = 0
        window_grad = []
        window_step_ms = []
        window_clip = 0
        window_steps = 0

    if start_update == 0:
        record_eval(0)
    else:
        if not curve:
            raise RuntimeError("resumed run has no curve rows to continue from")

    update = int(start_update)
    try:
        while update < int(updates):
            if watchdog.stop.is_set():
                stop_reason = "HARD_WALL_LIMIT"
                break
            if time.perf_counter() - t_start > float(soft_seconds):
                stop_reason = "SOFT_WALL_LIMIT"
                break
            rows = _next_batch(shuffle, gen, st.fit_rows)
            t0 = time.perf_counter()
            pb = D.build_pretext_batch(st, rows, int(shuffle["pass_index"]))
            loss, stats = model.loss(
                _move_view(pb.atom_view, device),
                _move_view(pb.bond_view, device),
                pb.atom_ids.to(device),
                pb.bond_ids.to(device),
                pb.atom_targets.to(device),
                pb.bond_targets.to(device),
            )
            if not torch.isfinite(loss):
                non_finite_events += 1
                raise RuntimeError(f"non-finite SSL loss at update {update}: {float(loss)}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
            if not math.isfinite(gnorm):
                non_finite_events += 1
                raise RuntimeError(f"non-finite gradient norm at update {update}")
            if gnorm > clip:
                window_clip += 1
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = float(time.perf_counter() - t0)
            update += 1
            window_steps += 1
            window_loss += float(loss.detach())
            if "ce_atom" in stats:
                window_ce_atom += float(stats["ce_atom"])
                window_atom_correct += int(stats["atom_correct"])
                window_atom_n += int(stats["atom_n"])
            if "ce_bond" in stats:
                window_ce_bond += float(stats["ce_bond"])
                window_bond_correct += int(stats["bond_correct"])
                window_bond_n += int(stats["bond_n"])
            window_grad.append(gnorm)
            window_step_ms.append(elapsed)
            if update % checkpoint_every == 0 and update != int(updates):
                payload = checkpoint_payload("pretrain_latest", update)
                sha = _atomic_save(payload, ckpt_latest)
                rows_records.append(
                    {"update": int(update), "checkpoint": C.track_rel(ckpt_latest), "sha256": sha}
                )
            if update % eval_every == 0 or update == int(updates):
                record_eval(update)
    finally:
        watchdog.cancel()

    if update < int(updates) and non_finite_events == 0:
        stop_reason = stop_reason if stop_reason != "RUNNING" else "INCOMPLETE"
    status = "COMPLETED" if update >= int(updates) and non_finite_events == 0 else "INCOMPLETE"
    if status == "COMPLETED":
        stop_reason = f"COMPLETED_{int(updates)}_UPDATES"
    elif non_finite_events:
        stop_reason = "NON_FINITE_EVENT"

    if curve and curve[-1]["update"] != update:
        record_eval(update)

    encoder_delta = M.parameter_change(initial_encoder_state, M.encoder_state(model))
    last = curve[-1] if curve else {}
    gate = {
        "atom_ce": float(last.get("probe_atom_ce", float("nan"))),
        "atom_prior_ce": float(last.get("probe_atom_prior_ce", float("nan"))),
        "atom_margin_nats": float(last.get("probe_atom_prior_ce", 0.0)) - float(last.get("probe_atom_ce", 0.0)),
        "bond_ce": float(last.get("probe_bond_ce", float("nan"))),
        "bond_prior_ce": float(last.get("probe_bond_prior_ce", float("nan"))),
        "bond_margin_nats": float(last.get("probe_bond_prior_ce", 0.0)) - float(last.get("probe_bond_ce", 0.0)),
        "required_margin_nats": float(C.PRETEXT_GATE["min_prior_ce_margin_nats"]),
        "encoders_changed": bool(encoder_delta["l2_delta"] > 0.0 and encoder_delta["all_tensors_changed"]),
        "encoder_l2_delta": float(encoder_delta["l2_delta"]),
        "encoder_relative_l2_delta": float(encoder_delta["relative_l2_delta"]),
    }
    gate["atom_passed"] = bool(gate["atom_margin_nats"] >= gate["required_margin_nats"])
    gate["bond_passed"] = bool(gate["bond_margin_nats"] >= gate["required_margin_nats"])
    gate["passed"] = bool(gate["atom_passed"] and gate["bond_passed"] and gate["encoders_changed"])
    gate["label"] = (
        "PRETEXT_LEARNED"
        if gate["passed"]
        else C.PRETEXT_GATE["failure_label"]
    )
    final_path = out_dir / "checkpoint_final.pt"
    final_payload = checkpoint_payload("pretrain_final", int(update))
    final_payload["pretext_gate"] = gate
    final_sha = _atomic_save(final_payload, final_path)
    curve_sha = write_curve()
    initial_fp = M.encoder_fingerprint_from_state(initial_encoder_state)
    final_fp = M.encoder_fingerprint(model)
    total_seconds = float(time.perf_counter() - t_start)
    run = {
        "schema": C.SCHEMA_RUN,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "stage": "pretrain",
        "seed": int(C.SSL_SEED),
        "status": status,
        "stop_reason": stop_reason,
        "updates_requested": int(updates),
        "updates_total": int(updates_total),
        "updates_completed": int(update),
        "resumed": bool(resumed),
        "dataset": C.DATASET,
        "pretrain_config": dict(C.PRETRAIN),
        "optimizer_audit": optimizer_audit,
        "pretext_audit": pretext_audit,
        "class_priors": priors,
        "norm_fingerprint": norm["norm_fingerprint"],
        "norm_audit": norm["ssl_norm_audit"],
        "data": st.audit,
        "probe": _probe_eval(st, model, priors, device),
        "curve": curve,
        "curve_path": C.track_rel(curve_path),
        "curve_sha256": curve_sha,
        "final_checkpoint": C.track_rel(final_path),
        "final_checkpoint_sha256": final_sha,
        "checkpoints": rows_records,
        "encoder_initial_fingerprint": initial_fp,
        "encoder_final_fingerprint": final_fp,
        "encoder_change": encoder_delta,
        "pretext_gate": gate,
        "non_finite_events": int(non_finite_events),
        "shuffle": {
            "pass_index": int(shuffle["pass_index"]),
            "offset": int(shuffle["offset"]),
            "n_full_batches": int(shuffle["n_full_batches"]),
            "dropped_tail_rows": int(shuffle["dropped_tail_rows"]),
            "shuffle_seed": int(C.SHUFFLE_SEED),
            "perm_sha256": str(shuffle["perm_sha256"]),
            "visits_reached": int(shuffle["pass_index"]) + 1,
        },
        "rng_state_hashes_start": rng_start_hashes,
        "rng_state_hashes_end": C.rng_state_hashes(C.capture_rng_state()),
        "gpu_seconds": total_seconds,
        "gpu_hours": total_seconds / 3600.0,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "soft_seconds": float(soft_seconds),
        "hard_seconds": float(hard_seconds),
        "environment": C.environment(),
        "git_commit": C.git_commit(),
        "started_utc": started_utc,
        "finished_utc": C.utcnow(),
        "test_evaluated": False,
        "test_policy": "official Test rows are never indexed or scored; no label artifact is opened",
    }
    C.write_json(out_dir / "run.json", run)
    C.log(
        f"[ssl] done status={status} updates={update}/{int(updates)} gate={gate['label']} "
        f"atomCE {gate['atom_ce']:.4f} vs prior {gate['atom_prior_ce']:.4f} | "
        f"bondCE {gate['bond_ce']:.4f} vs prior {gate['bond_prior_ce']:.4f} "
        f"gpu_hours={run['gpu_hours']:.4f}"
    )
    return run


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=int(C.PRETRAIN["updates"]))
    parser.add_argument("--soft-seconds", type=float, required=True)
    parser.add_argument("--hard-seconds", type=float, required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        C.write_json(
            out_dir / "run.json",
            {
                "schema": C.SCHEMA_RUN,
                "round_id": C.ROUND_ID,
                "stage": "pretrain",
                "status": "INCOMPLETE",
                "stop_reason": "CUDA_UNAVAILABLE",
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
                "test_evaluated": False,
            },
        )
        return 2
    try:
        st = D.load_structure()
        resume = None
        if args.resume:
            resume = torch.load(Path(args.resume), map_location="cpu", weights_only=False)
        run = pretrain(
            st,
            out_dir=out_dir,
            device=device,
            updates=int(args.updates),
            soft_seconds=float(args.soft_seconds),
            hard_seconds=float(args.hard_seconds),
            resume=resume,
            checkpoint_every=(int(args.checkpoint_every) or None),
        )
        return 0 if run["status"] == "COMPLETED" else 3
    except Exception as exc:  # noqa: BLE001
        C.write_json(
            out_dir / "run.json",
            {
                "schema": C.SCHEMA_RUN,
                "round_id": C.ROUND_ID,
                "stage": "pretrain",
                "status": "FAILED",
                "stop_reason": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "environment": C.environment(),
                "finished_utc": C.utcnow(),
                "test_evaluated": False,
            },
        )
        C.log(f"[ssl] FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
