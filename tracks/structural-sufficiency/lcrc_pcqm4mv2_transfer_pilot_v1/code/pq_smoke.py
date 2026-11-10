"""End-to-end preflight (<= 10,000 molecules) for the PCQM4Mv2 pilot.

The smoke test is **not** the first steps of a formal run: it trains a freshly
initialised model for 2 epochs on a 6,000-molecule slice and a 4,000-molecule
Validation slice, checks the items the protocol requires, and throws the result
away.  The formal runs are then started from scratch.

Checks
------
1. PCQM SMILES -> LCRC features: dims match the spec, exact rebuild of sampled
   molecules, S1/S2 counts consistent with the real physical bond graph.
2. Model: one output, expected parameter count, width audit, historical replay
   (``n_out=128`` reproduces the 994,324-parameter W2).
3. Atom renumbering does not change the graph-level prediction.
4. L1 loss + backward + optimizer update work; nothing is NaN/Inf.
5. De-normalisation is exactly invertible and the official MAE equals the
   independent NumPy MAE (both in eV).
6. Validation / hidden-test rows never enter a training batch.
7. Cache mmap read, checkpoint save and full restore (optimizer + RNG + order).
8. Measured step / evaluation cost -> projected GPU-hours for both formal runs.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pq_common as C  # noqa: E402
import pq_data as D  # noqa: E402
import pq_model as M  # noqa: E402
import pq_train as T  # noqa: E402
import pq_verify as V  # noqa: E402

import feature_spec as FS  # noqa: E402
import torch  # noqa: E402
from pf_pipeline import PinnedStager  # noqa: E402


def _build_task(train_tag: str, valid_tag: str, *, seed: int) -> dict[str, Any]:
    spec = FS.load_spec(C.ANALYSIS / "pcqm_spec.json")
    feature_norm = C.read_json(C.ANALYSIS / "pcqm_feature_norm.json")
    label_norm_payload = C.read_json(C.ANALYSIS / "pcqm_label_norm.json")
    train_cache = D.load_cache(C.CACHE / train_tag)
    valid_cache = D.load_cache(C.CACHE / valid_tag)
    labels = np.load(C.CACHE / "pcqm_labels_trainvalid.npy").astype(np.float64)
    train_gids = np.asarray(train_cache["gids"], dtype=np.int64)
    valid_gids = np.asarray(valid_cache["gids"], dtype=np.int64)
    train_y_eV = labels[train_gids]
    valid_y_eV = labels[valid_gids]
    norm = {
        "global_mean": np.asarray(feature_norm["global_mean_list"], dtype=np.float32),
        "global_scale": np.asarray(feature_norm["global_scale_list"], dtype=np.float32),
        "scalar_mean": np.asarray(feature_norm["scalar_mean_list"], dtype=np.float32),
        "scalar_scale": np.asarray(feature_norm["scalar_scale_list"], dtype=np.float32),
    }
    n_rows = int(train_cache["n_molecules"])
    return {
        "run": "SMOKE",
        "spec": spec,
        "norm": norm,
        "feature_norm": feature_norm,
        "label_norm": label_norm_payload,
        "train_cache": train_cache,
        "valid_cache": valid_cache,
        "valid_rows": np.arange(valid_gids.size, dtype=np.int64),
        "valid_y_eV": valid_y_eV,
        "train_y_eV": train_y_eV,
        "train_y_z": D.to_z(train_y_eV, label_norm_payload).astype(np.float32),
        "monitor_rows": np.arange(min(512, n_rows), dtype=np.int64),
        "monitor_y_eV": train_y_eV[: min(512, n_rows)],
        "monitor_plan": {"n": int(min(512, n_rows)), "source": "smoke subset only"},
        "n_rows": n_rows,
        "steps_epoch": int(C.steps_per_epoch(n_rows)),
        "steps_total": int(C.total_steps(n_rows)),
        "seed": int(seed),
        "orders": D.epoch_orders(n_rows, seed=int(seed), epochs=2),
        "evaluator": D._evaluator(),
        "cache_manifest": {"smoke": True},
        "data_isolation": {"smoke": True},
    }


def run_smoke(
    *,
    train_tag: str = "pcqm_smoke_train",
    valid_tag: str = "pcqm_smoke_valid",
    train_epochs: int = 2,
    device_name: str = "cuda",
    out: Path | None = None,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    gates: dict[str, Any] = {}
    details: dict[str, Any] = {}
    task = _build_task(train_tag, valid_tag, seed=C.MODEL_SEED)
    spec = task["spec"]
    n_train, n_valid = int(task["n_rows"]), int(task["valid_y_eV"].size)

    # ---- 1. features / cache -------------------------------------------------
    structure = D.cache_structure_audit(task["train_cache"], spec)
    structure_valid = D.cache_structure_audit(task["valid_cache"], spec)
    sample_rows = [0, 1, n_train // 2, n_train - 1]
    exact = V.exact_sample_check(task["train_cache"], spec, sample_rows)
    exact_valid = V.exact_sample_check(task["valid_cache"], spec, [0, n_valid - 1])
    dims = {
        "atom_feat_dim": [int(np.asarray(task["train_cache"]["atom_feat"]).shape[1]), spec.atom_feat_dim],
        "bond_feat_dim": [int(np.asarray(task["train_cache"]["bond_feat"]).shape[1]), spec.bond_feat_dim],
        "atom_scalar_dim": [int(np.asarray(task["train_cache"]["atom_scalar"]).shape[1]), spec.n_scalars],
        "global_dim": [int(np.asarray(task["train_cache"]["global"]).shape[1]), spec.global_dim],
        "n_nbr_codes": [int(spec.n_nbr_codes), int(FS.lipo_v1_spec().n_nbr_codes)],
        "n_path_codes": [int(spec.n_path_codes), int(FS.lipo_v1_spec().n_path_codes)],
    }
    gates["features_dims_match_spec"] = all(a == b for a, b in dims.values())
    gates["cache_structure_ok"] = bool(structure["ok"] and structure_valid["ok"])
    gates["s2_count_identity"] = bool(structure["formula_s2"]["ok"] and structure_valid["formula_s2"]["ok"])
    gates["s1_equals_2_bonds"] = bool(structure["formula_s2"]["s1_total_equals_2_bonds"])
    gates["exact_rebuild_from_smiles"] = bool(exact["all_fields_exact"] and exact_valid["all_fields_exact"])
    details["features"] = {
        "dims": dims,
        "structure_train": {
            "n_molecules": structure["n_molecules"],
            "n_atoms": structure["n_atoms_total"],
            "n_bonds": structure["n_bonds_total"],
            "n_s1": structure["n_s1_codes"],
            "n_s2": structure["n_s2_paths"],
            "formula_s2": structure["formula_s2"],
        },
        "structure_valid": {
            "n_molecules": structure_valid["n_molecules"],
            "n_s2": structure_valid["n_s2_paths"],
            "formula_s2_ok": structure_valid["formula_s2"]["ok"],
        },
        "exact_rebuild": exact,
        "exact_rebuild_valid": exact_valid,
    }

    # ---- labels / isolation --------------------------------------------------
    labels = np.load(C.CACHE / "pcqm_labels_trainvalid.npy").astype(np.float64)
    train_gids = np.asarray(task["train_cache"]["gids"], dtype=np.int64)
    valid_gids = np.asarray(task["valid_cache"]["gids"], dtype=np.int64)
    isolation = {
        "train_rows": int(train_gids.size),
        "valid_rows": int(valid_gids.size),
        "train_valid_gid_overlap": int(np.intersect1d(train_gids, valid_gids).size),
        "train_labels_finite": bool(np.isfinite(labels[train_gids]).all()),
        "valid_labels_finite": bool(np.isfinite(labels[valid_gids]).all()),
        "label_range_eV": [float(labels[train_gids].min()), float(labels[train_gids].max())],
    }
    gates["labels_finite"] = bool(isolation["train_labels_finite"] and isolation["valid_labels_finite"])
    gates["train_valid_disjoint"] = int(isolation["train_valid_gid_overlap"]) == 0
    details["isolation"] = isolation

    # ---- 2. model ------------------------------------------------------------
    seed_info = C.seed_everything(int(task["seed"]))
    model, audit = M.build_model(spec, task["norm"], seed=int(task["seed"]), device=device)
    optimizer, opt_audit = M.make_optimizer(model)
    replay = M.historical_replay(spec, task["norm"], seed=int(task["seed"]))
    gates["model_parameters_match_analytic"] = audit["parameter_check"].startswith("exact")
    gates["historical_w2_replays_994324"] = bool(replay["matches_historical"])
    gates["width_audit_all_match"] = bool(audit["width_audit"]["all_match"])
    gates["single_output"] = audit["readout_output_shape"] == [1, int(audit["widths"]["pool_hidden"])]
    details["model"] = {
        "parameters": audit["total_parameters"],
        "base_parameters": audit["base_parameters"],
        "readout_parameters": audit["readout_parameters"],
        "analytic": audit["parameter_accounting"],
        "historical_replay": replay,
        "readout_output_shape": audit["readout_output_shape"],
        "widths": audit["widths"],
        "trunk_init_fingerprint": audit["trunk_init_fingerprint"],
        "state_fingerprint": audit["state_fingerprint"],
        "dropout": audit["dropout"],
        "seed_state": seed_info,
        "optimizer": opt_audit,
    }

    # ---- 3. forward, output shape, finiteness, renumbering -------------------
    stager = PinnedStager(device, depth=2)
    rows = np.arange(8, dtype=np.int64)
    batch_cpu = D.collate(task["train_cache"], rows.tolist())
    batch = stager.batch(batch_cpu)
    model.eval()
    with torch.inference_mode():
        z_ref = M.predict_z(model, batch)
    # ``permute_atoms`` is a graph-level (CPU) relabelling in the pinned code.
    perm_batch_cpu, perm = M.permute_atoms(batch_cpu, seed=20261110)
    perm_batch = stager.batch(perm_batch_cpu)
    with torch.inference_mode():
        z_perm = M.predict_z(model, perm_batch)
    diff = float((z_ref - z_perm).abs().max())
    forward = {
        "batch_size": int(rows.size),
        "output_shape": list(z_ref.shape),
        "output_finite": bool(torch.isfinite(z_ref).all()),
        "renumbering_max_abs_dz": diff,
        "renumbering_identity_atoms": bool(torch.equal(perm, torch.arange(perm.numel()))),
        "module_calls": M.count_module_calls(model, batch),
    }
    gates["output_shape_is_batch"] = tuple(z_ref.shape) == (int(rows.size),)
    gates["output_finite"] = bool(torch.isfinite(z_ref).all())
    gates["atom_renumbering_invariant"] = bool(diff <= 1e-4)
    gates["one_encode_one_local_per_forward"] = bool(
        all(v == 1 for v in forward["module_calls"].values())
    )
    details["forward"] = forward

    # ---- 4. loss / backward / optimizer --------------------------------------
    model.train()
    y = stager.labels(task["train_y_z"][rows])
    out = T.train_step(
        model,
        optimizer,
        batch,
        y,
        global_step=0,
        steps_epoch=int(task["steps_epoch"]),
        steps_total=int(task["steps_total"]),
        clip=float(C.RECIPE["gradient_clip"]),
    )
    params_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    y2 = stager.labels(task["train_y_z"][rows])
    out2 = T.train_step(
        model,
        optimizer,
        batch,
        y2,
        global_step=1,
        steps_epoch=int(task["steps_epoch"]),
        steps_total=int(task["steps_total"]),
        clip=float(C.RECIPE["gradient_clip"]),
    )
    moved = sum(1 for n, p in model.named_parameters() if not torch.equal(p, params_before[n]))
    step_report = {
        "loss_step0": out["loss"],
        "loss_step1": out2["loss"],
        "grad_norm_step0": out["grad_norm"],
        "grad_norm_step1": out2["grad_norm"],
        "lr_step0": out["lr"],
        "lr_step1": out2["lr"],
        "parameters_updated": int(moved),
        "n_parameters": int(len(params_before)),
        "finite": bool(math.isfinite(out["loss"]) and math.isfinite(out2["loss"])),
    }
    lr0_expected = C.scheduled_lr(0, steps_epoch=int(task["steps_epoch"]), steps_total=int(task["steps_total"]))
    gates["l1_loss_finite"] = bool(step_report["finite"])
    gates["optimizer_updates_parameters"] = moved > 0
    gates["lr_follows_warmup_schedule"] = bool(abs(out["lr"] - lr0_expected) < 1e-12)
    details["train_step"]["lr_step0_expected"] = float(lr0_expected)
    details["train_step"] = step_report

    # ---- 5. de-normalisation + metric cross-check ---------------------------
    denorm = D.from_z(task["train_y_z"][rows].astype(np.float64), task["label_norm"])
    roundtrip = float(np.abs(denorm - task["train_y_eV"][rows]).max())
    valid_report, valid_pred = T.evaluate_split(model, task, device, which="valid", batch_size=256)
    metric = {
        "denorm_roundtrip_max_abs_eV": roundtrip,
        "official_mae_eV": valid_report["mae_eV"],
        "numpy_mae_eV": valid_report["mae_eV_numpy"],
        "crosscheck_delta_eV": valid_report["crosscheck_delta_eV"],
        "n_valid": valid_report["n"],
        "pred_z_mean": float(valid_pred.mean()),
        "pred_z_std": float(valid_pred.std()),
        "inference_seconds": valid_report["inference_seconds"],
    }
    gates["denormalisation_is_invertible"] = roundtrip <= 1e-9
    gates["official_mae_matches_numpy"] = valid_report["crosscheck_delta_eV"] <= 1e-9
    details["metric"] = metric

    # ---- 6. training batches never contain Validation rows ------------------
    seen: set[int] = set()
    for epoch in (1, 2):
        for batch_rows in D.batches_for_epoch(n_train, epoch, seed=int(task["seed"])):
            seen.update(int(x) for x in batch_rows)
    seen_gids = set(int(g) for g in train_gids[sorted(seen)])
    leak = seen_gids & set(int(g) for g in valid_gids)
    batch_isolation = {
        "epochs_checked": [1, 2],
        "train_rows_seen": int(len(seen)),
        "train_rows_total": int(n_train),
        "gids_seen": int(len(seen_gids)),
        "valid_gids_leaked": int(len(leak)),
        "covers_every_row_at_least_once": bool(len(seen) == n_train),
    }
    gates["no_validation_row_in_training_batches"] = len(leak) == 0
    details["batch_isolation"] = batch_isolation

    # ---- 7. two real epochs + checkpoint / restore ---------------------------
    out_dir = C.CACHE / "smoke_run"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    smoke_task = dict(task)
    smoke_task["orders"] = D.epoch_orders(n_train, seed=int(task["seed"]), epochs=int(train_epochs))
    t_fit = time.perf_counter()
    run_payload = T.train(
        smoke_task,
        run="PCQM_W2_300K",  # only the recipe constants are borrowed; no formal record is written
        epochs=int(train_epochs),
        out_dir=out_dir,
        device=device,
        resume=None,
        soft_seconds=10_000.0,
        hard_seconds=20_000.0,
        eval_batch_size=512,
        allow_partial_epochs=True,
    )
    fit_seconds = float(time.perf_counter() - t_fit)
    curve = run_payload["fixed_epoch_reports"]
    gates["smoke_run_reached_all_epochs"] = len(curve) == int(train_epochs)
    gates["smoke_valid_mae_finite"] = all(math.isfinite(float(v["mae_eV"])) for v in curve.values())

    last_path = out_dir / "checkpoint_last.pt"
    payload = torch.load(last_path, map_location="cpu", weights_only=False)
    model2, _audit2 = M.build_model(spec, task["norm"], seed=int(task["seed"]), device=device)
    optimizer2, _ = M.make_optimizer(model2)
    restore_audit = T.restore(payload, model2, optimizer2, run="PCQM_W2_300K", task=smoke_task)
    with torch.inference_mode():
        z1 = M.predict_z(model.eval(), batch)
        z2 = M.predict_z(model2.eval(), batch)
    resumed_report, _ = T.evaluate_split(model2, smoke_task, device, which="valid", batch_size=256)
    restore_check = {
        "restore_audit": restore_audit,
        "predictions_identical_after_restore": bool(torch.equal(z1, z2)),
        "max_abs_dz_after_restore": float((z1 - z2).abs().max()),
        "valid_mae_after_restore_eV": float(resumed_report["mae_eV"]),
        "valid_mae_before_restore_eV": float(curve[str(int(train_epochs))]["mae_eV"]),
    }
    gates["checkpoint_restore_is_exact"] = bool(restore_check["predictions_identical_after_restore"])
    gates["restore_reproduces_valid_mae"] = abs(
        restore_check["valid_mae_after_restore_eV"] - restore_check["valid_mae_before_restore_eV"]
    ) <= 1e-12
    details["checkpoint"] = restore_check

    # resume one more epoch from the restored state
    t_resume = time.perf_counter()
    resume_task = dict(smoke_task)
    resume_task["orders"] = D.epoch_orders(n_train, seed=int(task["seed"]), epochs=int(train_epochs) + 1)
    resumed = T.train(
        resume_task,
        run="PCQM_W2_300K",
        epochs=int(train_epochs) + 1,
        out_dir=out_dir,
        device=device,
        resume=payload,
        soft_seconds=10_000.0,
        hard_seconds=20_000.0,
        eval_batch_size=512,
        allow_partial_epochs=True,
    )
    resume_seconds = float(time.perf_counter() - t_resume)
    gates["resume_completes"] = resumed["status"] == "COMPLETED" and resumed["epochs_run"] == int(train_epochs) + 1
    details["resume"] = {
        "status": resumed["status"],
        "epochs_run": resumed["epochs_run"],
        "seconds": resume_seconds,
        "resume_audit": resumed["resume_audit"],
    }

    # ---- 8. cost projection -------------------------------------------------
    curves = C.read_json(C.ANALYSIS / "pcqm_data_audit.json")
    step_ms = [float(r["step_ms_p50"]) for r in _curve_rows(out_dir / "train_curve.csv")]
    valid_seconds = [float(r["epoch_valid_seconds"]) for r in _curve_rows(out_dir / "train_curve.csv")]
    median_step_ms = float(np.median(step_ms)) if step_ms else float("nan")
    median_valid_s = float(np.median(valid_seconds)) if valid_seconds else float("nan")
    projections = {}
    for run, rspec in C.RUNS_TABLE.items():
        n = int(rspec["train_size"])
        s_epoch = C.steps_per_epoch(n)
        s_total = C.total_steps(n)
        train_s = s_total * median_step_ms / 1e3
        # Validation cost per epoch scales with the number of validation molecules
        eval_s = 40 * (median_valid_s * (73493 / max(n_valid, 1)) + 0.25 * median_valid_s)
        projections[run] = {
            "steps_per_epoch": s_epoch,
            "steps_total": s_total,
            "projected_train_seconds": float(train_s),
            "projected_eval_seconds": float(eval_s),
            "projected_gpu_hours": float((train_s + eval_s) / 3600.0),
        }
    total_hours = sum(p["projected_gpu_hours"] for p in projections.values())
    cost = {
        "median_step_ms_from_smoke": median_step_ms,
        "median_valid_epoch_seconds_smoke": median_valid_s,
        "smoke_train_seconds": fit_seconds,
        "smoke_resume_seconds": resume_seconds,
        "projections": projections,
        "projected_total_gpu_hours": float(total_hours),
        "budget_target_hours": 6.0,
        "budget_hard_hours": 9.0,
        "within_target": bool(total_hours <= 6.0),
        "within_hard_cap": bool(total_hours <= 9.0),
        "dataset_split_sizes": curves["split_sizes"],
        "peak_rss_mb": C.peak_rss_mb(),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    gates["cost_within_hard_cap"] = bool(cost["within_hard_cap"])
    details["cost"] = cost

    payload_out = {
        "schema": "lcrc-pcqm4mv2-smoke-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "host_facts": C.host_facts(),
        "device": str(device),
        "smoke_molecules": int(n_train + n_valid),
        "smoke_limit": 10_000,
        "within_smoke_limit": bool(n_train + n_valid <= 10_000),
        "train_epochs": int(train_epochs),
        "seconds": float(time.perf_counter() - t_start),
        "details": details,
        "gates": gates,
        "all_gates_pass": bool(all(bool(v) for v in gates.values()) and n_train + n_valid <= 10_000),
    }
    payload_out["status"] = "OK" if payload_out["all_gates_pass"] else "FAILED"
    out = Path(out) if out else C.ANALYSIS / "pcqm_smoke.json"
    C.write_json(out, payload_out)
    C.log(f"[smoke] {payload_out['status']} gates={sum(bool(v) for v in gates.values())}/{len(gates)} -> {C.track_rel(out)}")
    return payload_out


def _curve_rows(path: Path) -> list[dict[str, Any]]:
    import csv

    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", default="pcqm_smoke_train")
    parser.add_argument("--valid-cache", default="pcqm_smoke_valid")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    payload = run_smoke(
        train_tag=args.train_cache,
        valid_tag=args.valid_cache,
        train_epochs=int(args.epochs),
        device_name=args.device,
        out=Path(args.out) if args.out else None,
    )
    return 0 if payload["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
