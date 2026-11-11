"""Correctness gates for the fullscale round (run before any formal step).

Covers the twelve checks required by the execution instruction: parameter counts
and width rule, E1 == frozen collation on real full-cache batches, full-Train
row alignment, 1M-prefix identity, no Validation/Test row in an optimizer
batch, reproducible initialisation/RNG, a single non-iterative atom/bond/local
encoding, finite real loss/gradient/update, full checkpoint save/restore/continue,
official evaluator == NumPy MAE, frozen normalisation, and SHA256 identity of
code and data.

The checkpoint/resume gate runs on the requested device (``cuda`` for the formal
regime); everything else works on CPU or CUDA.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import fs_common as C
import fs_data as D
import fs_train as T

import torch


def _state_equal(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    keys = sorted(set(a) | set(b))
    mismatched = [
        k
        for k in keys
        if k not in a or k not in b or a[k].shape != b[k].shape or not torch.equal(a[k].cpu(), b[k].cpu())
    ]
    return {"n_keys": len(keys), "mismatched": mismatched, "bitwise_equal": not mismatched}


def _optimizer_equal(a: torch.optim.Optimizer, b: torch.optim.Optimizer) -> dict[str, Any]:
    sa, sb = a.state_dict(), b.state_dict()
    bad: list[str] = []
    if sa["param_groups"] != sb["param_groups"]:
        bad.append("param_groups")
    for pid in sorted(set(sa["state"]) | set(sb["state"])):
        x, y = sa["state"].get(pid), sb["state"].get(pid)
        if x is None or y is None or set(x) != set(y):
            bad.append(f"state[{pid}]")
            continue
        for key in x:
            vx, vy = x[key], y[key]
            if torch.is_tensor(vx):
                if vx.shape != vy.shape or not torch.equal(vx.cpu(), vy.cpu()):
                    bad.append(f"state[{pid}].{key}")
            elif vx != vy:
                bad.append(f"state[{pid}].{key}")
    return {"bitwise_equal": not bad, "mismatched": bad}


def _batches(task: Mapping[str, Any], count: int, *, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(int(task["n_rows"])).astype(np.int64)
    return [order[i : i + int(C.BATCH_SIZE)] for i in range(0, int(count) * int(C.BATCH_SIZE), int(C.BATCH_SIZE))]


def _device_tensors(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def run_gates(
    *,
    device: torch.device,
    steps_finite: int = 100,
    steps_resume: int = 40,
    steps_resume_extra: int = 10,
    out_path: Path,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    deterministic = C.apply_deterministic_settings()
    task = D.load_task("PCQM_W2_FULL", verify_hashes=True)
    task_w3 = dict(task)
    task_w3.update(
        {
            "run": "PCQM_W3_FULL",
            "model": "W3",
            "model_name": C.MODEL_NAMES[3],
            "scale": 3,
            "expected_parameters": C.EXPECTED_PARAMETERS[3],
            "widths": dict(C.WIDTHS[3]),
        }
    )
    gates: dict[str, Any] = {}
    spec, norm = task["spec"], task["norm"]

    # 1 — parameter counts, width rule, builder identity, reproducible init
    table = C.param_table()
    built: dict[str, Any] = {}
    for scale in (2, 3):
        model, audit = C.EW_MODEL.build_model(spec, norm, scale=scale, seed=C.MODEL_SEED, device="cpu")
        built[str(scale)] = {
            "parameters": int(audit["total_parameters"]),
            "expected": int(C.EXPECTED_PARAMETERS[scale]),
            "analytic": int(table[str(scale)]["total_parameters"]),
            "state_fingerprint": audit["state_fingerprint"],
            "width_audit_all_match": bool(audit["width_audit"]["all_match"]),
        }
    model_w2_a, audit_a = C.EW_MODEL.build_model(spec, norm, scale=2, seed=0, device="cpu")
    model_w2_b, audit_b = C.EW_MODEL.build_model(spec, norm, scale=2, seed=0, device="cpu")
    init_equal = _state_equal(model_w2_a.state_dict(), model_w2_b.state_dict())
    with torch.random.fork_rng(devices=[]):
        frozen_w2, frozen_audit = C.PQ_MODEL.build_model(spec, norm, seed=0, device="cpu")
    frozen_equal = _state_equal(model_w2_a.state_dict(), frozen_w2.state_dict())
    gates["parameters"] = {
        "analytic_table": table,
        "built": built,
        "w2_builder_equals_historical": frozen_equal,
        "w2_builder_parameters_historical": int(frozen_audit["total_parameters"]),
        "init_reproducible": init_equal,
        "status": "OK"
        if (
            all(int(built[s]["parameters"]) == int(C.EXPECTED_PARAMETERS[int(s)]) for s in built)
            and all(int(built[s]["analytic"]) == int(C.EXPECTED_PARAMETERS[int(s)]) for s in built)
            and all(bool(built[s]["width_audit_all_match"]) for s in built)
            and frozen_equal["bitwise_equal"]
            and init_equal["bitwise_equal"]
        )
        else "FAILED",
    }

    # 2 — E1 collation equals the frozen collation field-by-field
    rng = np.random.default_rng(7)
    probe = np.unique(
        np.concatenate(
            [
                np.arange(0, 5),
                np.arange(999_998, 1_000_003),
                rng.choice(int(task["n_rows"]), size=24, replace=False),
            ]
        )
    )
    collate_checks = []
    for size in (1, 2, 5, 128):
        rows = probe[: min(size, probe.size)].tolist()
        a = C.EW_COLLATE.collate_for(task["train_cache"], rows, implementation="E1")
        b = C.EW_COLLATE.collate_for(task["train_cache"], rows, implementation="frozen")
        cmp = C.EW_COLLATE.compare_batches(a, b)
        collate_checks.append({"batch_size": len(rows), "all_equal": cmp["all_equal"], "missing": cmp["missing"], "extra": cmp["extra"]})
    gates["collate_e1_equals_frozen"] = {
        "checks": collate_checks,
        "status": "OK" if all(c["all_equal"] for c in collate_checks) else "FAILED",
    }

    # 3/4/5 — row alignment, prefix control, no hidden row in optimizer input
    plan = task["row_plan"]
    cache = task["train_cache"]
    gids = np.asarray(cache["gids"], dtype=np.int64)
    prefix = C.read_json(C.ANALYSIS / "fs_cache_prefix_control.json")
    split = task["split_identity"]["split"]
    valid_ids = np.asarray(split["valid"], dtype=np.int64)
    dev_ids = np.asarray(split["test-dev"], dtype=np.int64)
    chal_ids = np.asarray(split["test-challenge"], dtype=np.int64)
    train_ids = np.asarray(split["train"], dtype=np.int64)
    sampled = _batches(task, 20, seed=0)
    batch_gids = np.concatenate([gids[b] for b in sampled])
    gates["row_alignment"] = {
        "n_rows": int(gids.size),
        "gids_sha256": C.index_hash(gids),
        "plan_rows_sha256": plan["rows_sha256"],
        "rows_equal_plan": bool(np.array_equal(gids, np.load(C.PARENT_CACHE / "pcqm_permutation.npy"))),
        "unique": bool(np.unique(gids).size == gids.size),
        "set_equals_official_train": bool(np.array_equal(np.sort(gids), np.sort(train_ids))),
        "prefix_equals_hist_1m": bool(
            np.array_equal(gids[:1_000_000], np.load(C.PARENT_CACHE / "pcqm_subset_rows.npy"))
        ),
        "prefix_control_status": prefix["status"],
        "prefix_control_all_equal": bool(prefix["all_equal"]),
        "sampled_batches": int(len(sampled)),
        "sampled_rows": int(batch_gids.size),
        "sampled_valid_overlap": int(np.intersect1d(batch_gids, valid_ids, assume_unique=False).size),
        "sampled_test_dev_overlap": int(np.intersect1d(batch_gids, dev_ids, assume_unique=False).size),
        "sampled_test_challenge_overlap": int(np.intersect1d(batch_gids, chal_ids, assume_unique=False).size),
        "status": "OK"
        if (
            bool(np.array_equal(gids, np.load(C.PARENT_CACHE / "pcqm_permutation.npy")))
            and bool(np.array_equal(np.sort(gids), np.sort(train_ids)))
            and prefix["status"] == "OK"
            and bool(prefix["all_equal"])
            and int(np.intersect1d(batch_gids, valid_ids).size) == 0
            and int(np.intersect1d(batch_gids, dev_ids).size) == 0
            and int(np.intersect1d(batch_gids, chal_ids).size) == 0
        )
        else "FAILED",
    }

    # 6/7 — one-shot encoding only; count module calls on a real batch
    model_w2_a.eval()
    batch_rows = sampled[0].tolist()
    raw = C.EW_COLLATE.collate_v2(task["train_cache"], batch_rows)
    calls = C.EW_MODEL.count_module_calls(model_w2_a, raw)
    gates["one_shot_forward"] = {
        "module_calls_one_batch": calls,
        "expected": {"atom_encoder": 1, "bond_encoder": 1, "local": 1, "f": 1, "rho": 1, "readout": 1},
        "no_iterative_message_passing": bool(
            calls.get("atom_encoder") == 1 and calls.get("bond_encoder") == 1 and calls.get("local") == 1
        ),
        "status": "OK"
        if calls == {"atom_encoder": 1, "bond_encoder": 1, "local": 1, "f": 1, "rho": 1, "readout": 1}
        else "FAILED",
    }

    # 8 — finite real loss/gradient/update at both widths
    finite: dict[str, Any] = {}
    for scale, t in ((2, task), (3, task_w3)):
        C.seed_everything(int(t["seed"]))
        model, _audit = C.EW_MODEL.build_model(spec, norm, scale=scale, seed=int(t["seed"]), device=device)
        optimizer, _ = T.make_optimizer(model)
        model.train()
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        losses, gnorms = [], []
        for i, rows in enumerate(_batches(t, steps_finite, seed=0)):
            raw = C.EW_COLLATE.collate_v2(t["train_cache"], rows.tolist())
            batch = _device_tensors(raw, device)
            y = torch.from_numpy(np.ascontiguousarray(t["train_y_z"][rows], dtype=np.float32)).to(device)
            out = C.PQ_TRAIN.train_step(
                model,
                optimizer,
                batch,
                y,
                global_step=i,
                steps_epoch=int(t["steps_epoch"]),
                steps_total=int(t["steps_total"]),
                clip=float(C.RECIPE["gradient_clip"]),
            )
            losses.append(float(out["loss"]))
            gnorms.append(float(out["grad_norm"]))
        delta = max(
            float((v.detach().cpu().to(torch.float64) - before[k].to(torch.float64)).abs().max())
            for k, v in model.state_dict().items()
        )
        finite[str(scale)] = {
            "steps": int(steps_finite),
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_mean": float(np.mean(losses)),
            "grad_norm_mean": float(np.mean(gnorms)),
            "grad_norm_max": float(np.max(gnorms)),
            "weights_changed_max_abs": delta,
            "all_finite": bool(np.isfinite(losses).all() and np.isfinite(gnorms).all() and np.isfinite(delta)),
            "status": "OK"
            if np.isfinite(losses).all() and np.isfinite(gnorms).all() and delta > 0
            else "FAILED",
        }
    gates["finite_training"] = finite

    # 9 — checkpoint save -> restore -> continue is bitwise identical
    C.seed_everything(0)
    model, model_audit = C.EW_MODEL.build_model(spec, norm, scale=2, seed=0, device=device)
    optimizer, opt_audit = T.make_optimizer(model)
    model.train()
    batches = _batches(task, steps_resume + steps_resume_extra, seed=11)
    for i, rows in enumerate(batches[:steps_resume]):
        raw = C.EW_COLLATE.collate_v2(task["train_cache"], rows.tolist())
        batch = _device_tensors(raw, device)
        y = torch.from_numpy(np.ascontiguousarray(task["train_y_z"][rows], dtype=np.float32)).to(device)
        C.PQ_TRAIN.train_step(
            model, optimizer, batch, y, global_step=i, steps_epoch=int(task["steps_epoch"]),
            steps_total=int(task["steps_total"]), clip=float(C.RECIPE["gradient_clip"]),
        )
    payload = T.checkpoint_payload(
        task=task,
        run="PCQM_W2_FULL",
        epoch=0,
        global_step=steps_resume,
        model=model,
        optimizer=optimizer,
        model_audit=model_audit,
        valid_report=None,
        kind="gate_resume",
        extra={"optimizer_state_audit": opt_audit},
    )
    reference_losses = []
    for i, rows in enumerate(batches[steps_resume:], start=steps_resume):
        raw = C.EW_COLLATE.collate_v2(task["train_cache"], rows.tolist())
        batch = _device_tensors(raw, device)
        y = torch.from_numpy(np.ascontiguousarray(task["train_y_z"][rows], dtype=np.float32)).to(device)
        out = C.PQ_TRAIN.train_step(
            model, optimizer, batch, y, global_step=i, steps_epoch=int(task["steps_epoch"]),
            steps_total=int(task["steps_total"]), clip=float(C.RECIPE["gradient_clip"]),
        )
        reference_losses.append(float(out["loss"]))
    ref_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    C.seed_everything(123456)  # different state, then restored from the checkpoint
    model2, audit2 = C.EW_MODEL.build_model(spec, norm, scale=2, seed=0, device=device)
    optimizer2, _ = T.make_optimizer(model2)
    restore_audit = T.restore(payload, model2, optimizer2, run="PCQM_W2_FULL", task=task)
    model2.train()
    restored_losses = []
    for i, rows in enumerate(batches[steps_resume:], start=steps_resume):
        raw = C.EW_COLLATE.collate_v2(task["train_cache"], rows.tolist())
        batch = _device_tensors(raw, device)
        y = torch.from_numpy(np.ascontiguousarray(task["train_y_z"][rows], dtype=np.float32)).to(device)
        out = C.PQ_TRAIN.train_step(
            model2, optimizer2, batch, y, global_step=i, steps_epoch=int(task["steps_epoch"]),
            steps_total=int(task["steps_total"]), clip=float(C.RECIPE["gradient_clip"]),
        )
        restored_losses.append(float(out["loss"]))
    state_cmp = _state_equal(ref_state, model2.state_dict())
    opt_cmp = _optimizer_equal(optimizer, optimizer2)
    gates["checkpoint_resume"] = {
        "device": str(device),
        "steps_before_save": int(steps_resume),
        "steps_after_restore": int(steps_resume_extra),
        "restore_audit": restore_audit,
        "losses_reference": reference_losses,
        "losses_restored": restored_losses,
        "max_loss_delta": float(np.max(np.abs(np.asarray(reference_losses) - np.asarray(restored_losses)))),
        "state": state_cmp,
        "optimizer": opt_cmp,
        "status": "OK" if state_cmp["bitwise_equal"] and opt_cmp["bitwise_equal"] else "FAILED",
    }

    # 10 — official evaluator MAE == independent NumPy MAE on full Validation
    report, _ = C.PQ_TRAIN.evaluate_split(model, task, device, which="valid", batch_size=512)
    gates["official_vs_numpy_mae"] = {
        "mae_eV": float(report["mae_eV"]),
        "mae_eV_numpy": float(report["mae_eV_numpy"]),
        "delta": float(report["crosscheck_delta_eV"]),
        "n": int(report["n"]),
        "status": "OK" if float(report["crosscheck_delta_eV"]) <= float(C.GATES["official_vs_numpy_mae_abs_tol"]) else "FAILED",
    }

    # 11 — frozen normalisation constants + inverse roundtrip
    feature_norm = C.read_json(C.PARENT_ANALYSIS / "pcqm_feature_norm.json")
    label_norm = C.read_json(C.PARENT_ANALYSIS / "pcqm_label_norm.json")
    z = C.PQ_DATA.to_z(task["train_y_eV"][:1000], label_norm)
    back = C.PQ_DATA.from_z(z, label_norm)
    gates["normalisation"] = {
        "feature_norm_file_sha256": C.sha256_file(C.PARENT_ANALYSIS / "pcqm_feature_norm.json"),
        "label_norm_file_sha256": C.sha256_file(C.PARENT_ANALYSIS / "pcqm_label_norm.json"),
        "feature_norm_fingerprint": feature_norm["norm_fingerprint"],
        "task_feature_norm_fingerprint": task["feature_norm"]["norm_fingerprint"],
        "label_mean_eV": float(label_norm["mean_eV"]),
        "label_std_eV": float(label_norm["std_eV"]),
        "fit_rows": int(label_norm["n"]),
        "roundtrip_max_abs_error_eV": float(np.max(np.abs(back - task["train_y_eV"][:1000]))),
        "status": "OK"
        if (
            task["feature_norm"]["norm_fingerprint"] == feature_norm["norm_fingerprint"]
            and int(label_norm["n"]) == 300_000
            and float(np.max(np.abs(back - task["train_y_eV"][:1000]))) < 1e-9
        )
        else "FAILED",
    }

    # 12 — code / data identity
    artifacts = {
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "full_cache_sha256": task["cache_manifest"]["full_cache"]["cache_sha256"],
        "rows_sha256": plan["rows_sha256"],
        "spec_hash": task["spec"].hash(),
        "split_dict_sha256": task["split_identity"]["split_dict"]["sha256"],
        "valid_cache_sha256": task["cache_manifest"]["valid_cache"]["cache_sha256"],
        "train1m_control_sha256": task["cache_manifest"]["train1m_control_sha256"],
        "code_files": {},
    }
    for p in sorted(Path(__file__).resolve().parent.glob("fs_*.py")):
        artifacts["code_files"][p.name] = C.sha256_file(p)
    gates["identity"] = {
        **artifacts,
        "status": "OK" if not artifacts["git_dirty"] else "DIRTY_WORKTREE",
    }

    failed = [k for k, v in gates.items() if isinstance(v, Mapping) and v.get("status") not in (None, "OK")]
    payload = {
        "schema": C.SCHEMA_VERIFY,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "device": str(device),
        "deterministic_settings": deterministic,
        "task_load_seconds": float(task["load_seconds"]),
        "gates": gates,
        "failed": failed,
        "status": "OK" if not failed else "FAILED",
        "seconds": float(time.perf_counter() - t_start),
        "environment": C.environment(),
    }
    C.write_json(out_path, C.json_safe(payload))
    C.log(f"[verify] {payload['status']} failed={failed} {payload['seconds']:.1f}s -> {C.track_rel(out_path)}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps-finite", type=int, default=100)
    parser.add_argument("--steps-resume", type=int, default=40)
    parser.add_argument("--steps-resume-extra", type=int, default=10)
    parser.add_argument("--out", default=str(C.ANALYSIS / "fs_verify.json"))
    args = parser.parse_args(argv)
    payload = run_gates(
        device=torch.device(args.device),
        steps_finite=int(args.steps_finite),
        steps_resume=int(args.steps_resume),
        steps_resume_extra=int(args.steps_resume_extra),
        out_path=Path(args.out),
    )
    return 0 if payload["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
