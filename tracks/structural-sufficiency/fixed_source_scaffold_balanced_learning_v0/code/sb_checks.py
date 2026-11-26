"""Pre-training checks (PROTOCOL section 6) + throughput preflight.

All checks must pass before the formal run.  They are cheap: one bundle load, one
model build, one diagnostic batch, twelve real training steps and two full-split
evaluations.

    0 frozen config / data identity / batch plan
    1 initial state equals the historical FIXED1 initial state (base + full)
    2 architecture / parameter count / inference computation unchanged
       (historical checkpoint reloaded on this CPU reproduces Valid RMSE 0.726678)
    3 every Train molecule participates, in every epoch
    4 every scaffold carries exactly N_train / K_train total weight
    5 mean weight == 1, quantiles, effective sample size
    6 weights depend on Train structure only (label permutations change nothing)
    7 unit weights reproduce the plain MSE bitwise (value and gradient)
    8 Valid metrics are unweighted and reproduce the hand-computed RMSE
    9 preflight timing projection fits the wall-clock budget
"""

from __future__ import annotations

import inspect
import time
from typing import Any

import numpy as np

import sb_common as C

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import cs_model as M  # noqa: E402
import sb_data as D  # noqa: E402
import sb_scaffold as S  # noqa: E402
import sb_train as T  # noqa: E402


def _state_keys(model: torch.nn.Module) -> dict[str, list[int]]:
    return {k: list(v.shape) for k, v in model.state_dict().items()}


def main() -> int:
    checks: dict[str, Any] = {}
    t0 = time.perf_counter()
    frozen = C.read_json(C.CONFIG_DIR / "experiment.json")
    official = S.load_official_rows()
    all_rows = np.concatenate([official["train_rows"], official["valid_rows"]])
    scaffold_by_row = S.scaffold_of_rows(official["smiles"], all_rows)
    k_train = len({scaffold_by_row[int(r)] for r in official["train_rows"]})
    bundle, identity = D.load_bundle()
    device = torch.device("cpu")

    # ---- 0 frozen config / identity / plan ---------------------------------
    wpack = D.train_weights(bundle, official, scaffold_by_row, k_train=int(k_train))
    plan = D.batch_order_plan(C.EPOCHS)
    canon = D.canonical_check(bundle, official)
    checks["0_frozen_identity"] = {
        "k_train_matches_frozen": bool(int(k_train) == int(frozen["weights"]["k_train"])),
        "weight_sha256_matches_frozen": bool(wpack["sha256"] == frozen["weights"]["sha256"]),
        "batch_plan_sha256_matches_frozen": bool(plan["plan_sha256"] == frozen["batch_plan"]["plan_sha256"]),
        "batch_plan_matches_historical_fixed1": bool(plan["plan_sha256"] == C.HIST["train_plan_sha256"]),
        "n_train": int(C.N_TRAIN),
        "n_valid": int(C.N_VALID),
        "splits_available": sorted(bundle.caches.keys()),
        "test_arrays_dropped": identity["test_arrays_dropped_after_audit"],
        "cache_vs_official_smiles_ok": bool(canon["ok"]),
        "target_scaler": {k: bundle.scaler[k] for k in ("mean", "std", "n", "ddof")},
        "target_scaler_matches_frozen_history": bool(
            abs(float(bundle.scaler["mean"]) - 2.1621607144740187) < 1e-12
            and abs(float(bundle.scaler["std"]) - 1.211539840098397) < 1e-12
            and int(bundle.scaler["n"]) == C.N_TRAIN
        ),
        "passed": bool(
            int(k_train) == int(frozen["weights"]["k_train"])
            and wpack["sha256"] == frozen["weights"]["sha256"]
            and plan["plan_sha256"] == C.HIST["train_plan_sha256"]
            and canon["ok"]
            and sorted(bundle.caches.keys()) == ["train", "valid"]
        ),
    }

    # ---- 1 initial state ---------------------------------------------------
    C.set_seeds(C.SEED)
    model, model_audit = M.build_csc(bundle.spec, bundle.norm, arm="A", seed=C.SEED, device=device)
    initial = C.state_fingerprint(C.state_cpu(model))
    checks["1_initial_state"] = {
        "initial_state_sha256": initial,
        "historical_initial_state_sha256": C.HIST["initial_state_sha256"],
        "base_state_sha256": model_audit["base_state_sha256"],
        "historical_base_state_sha256": C.HIST["base_state_sha256"],
        "new_state_sha256": model_audit["new_state_sha256"],
        "initial_matches_historical": bool(initial == C.HIST["initial_state_sha256"]),
        "base_matches_historical": bool(model_audit["base_state_sha256"] == C.HIST["base_state_sha256"]),
        "passed": bool(
            initial == C.HIST["initial_state_sha256"]
            and model_audit["base_state_sha256"] == C.HIST["base_state_sha256"]
        ),
    }

    # ---- 2 architecture / inference invariance ----------------------------
    hist_ckpt = torch.load(C.CSC_RUN / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    hist_state = hist_ckpt["state_dict"]
    keys_now = _state_keys(model)
    keys_hist = {k: list(v.shape) for k, v in hist_state.items()}
    n_params = int(model_audit["n_parameters"])
    n_new = int(model_audit["n_parameters_new"])
    # reload the historical best checkpoint on this machine and re-score Valid
    model_hist, _ = M.build_csc(bundle.spec, bundle.norm, arm="A", seed=C.SEED, device=device)
    model_hist.load_state_dict(hist_state, strict=True)
    hist_epoch_eval = D.evaluate(bundle, model_hist, "valid", device)
    checks["2_architecture_invariance"] = {
        "parameters": n_params,
        "historical_parameters": int(C.HIST["parameters"]),
        "new_parameters": n_new,
        "state_keys_identical": bool(keys_now == keys_hist),
        "state_key_count": int(len(keys_now)),
        "n_gate_modules": int(sum(1 for n, _ in model.named_parameters() if n.startswith("gate."))),
        "parameter_names": sorted(keys_now),
        "historical_checkpoint_epoch": int(hist_ckpt["epoch"]),
        "historical_checkpoint_valid_rmse_recorded": float(hist_ckpt["valid_rmse_raw"]),
        "historical_checkpoint_valid_rmse_recomputed_cpu": float(hist_epoch_eval["rmse_raw"]),
        "delta_vs_recorded": float(abs(hist_epoch_eval["rmse_raw"] - float(hist_ckpt["valid_rmse_raw"]))),
        "tolerance": 1e-6,
        "inference_path": "model(batch) == CSModel.forward -> encode_h0 + 2 structural stages + post_local; no extra module for this arm",
        "passed": bool(
            n_params == int(C.HIST["parameters"])
            and keys_now == keys_hist
            and abs(hist_epoch_eval["rmse_raw"] - float(hist_ckpt["valid_rmse_raw"])) <= 1e-6
        ),
    }
    del model_hist

    # ---- 3 all Train molecules / order ------------------------------------
    per_epoch_ok = 0
    visits = 0
    for epoch in range(C.EPOCHS):
        order = plan["plan"][epoch]
        ok = bool(order.size == C.N_TRAIN and np.unique(order).size == C.N_TRAIN and int(order.min()) == 0
                  and int(order.max()) == C.N_TRAIN - 1)
        per_epoch_ok += int(ok)
        visits += int(order.size)
    checks["3_all_train_molecules"] = {
        "epochs": int(C.EPOCHS),
        "epochs_with_exact_permutation": int(per_epoch_ok),
        "row_visits_total": int(visits),
        "row_visits_expected": int(C.EPOCHS * C.N_TRAIN),
        "all_weights_positive": bool(np.all(wpack["weights"] > 0)),
        "n_zero_weights": int((wpack["weights"] == 0).sum()),
        "passed": bool(per_epoch_ok == C.EPOCHS and visits == C.EPOCHS * C.N_TRAIN and np.all(wpack["weights"] > 0)),
    }

    # ---- 4 per-scaffold total weight --------------------------------------
    audit = S.weight_audit(scaffold_by_row, official["train_rows"], wpack["weights_by_row_id"])
    checks["4_scaffold_total_weight"] = {
        **{k: audit[k] for k in (
            "n_train", "k_scaffolds", "expected_total_per_scaffold", "per_scaffold_total_min",
            "per_scaffold_total_max", "per_scaffold_total_max_abs_deviation",
            "per_scaffold_total_relative_deviation", "all_per_scaffold_totals_equal",
        )},
        "tolerance": "1e-9 relative",
        "passed": bool(audit["all_per_scaffold_totals_equal"]),
    }

    # ---- 5 mean / quantiles / ESS -----------------------------------------
    w = wpack["weights"]
    checks["5_weight_distribution"] = {
        **audit["weight_summary"],
        "mean_weight": float(w.mean()),
        "sum_weight": float(w.sum()),
        "mean_is_one": bool(abs(float(w.mean()) - 1.0) <= 1e-12),
        "effective_sample_size": audit["effective_sample_size"],
        "effective_sample_size_fraction": audit["effective_sample_size_fraction"],
        "uniform_weights_ess": float(C.N_TRAIN),
        "passed": bool(abs(float(w.mean()) - 1.0) <= 1e-12),
    }

    # ---- 6 label independence --------------------------------------------
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    y_train = np.asarray([official["labels"][int(r)] for r in official["train_rows"]], dtype=np.float64)
    base_sha = wpack["sha256"]
    label_perm_sha: set[str] = set()
    for _ in range(200):
        y_perm = rng.permutation(y_train)
        # the weight rule has no label argument; emulate "labels supplied but unused"
        permuted_order = np.argsort(y_perm, kind="stable")
        rows_perm = np.asarray(official["train_rows"], dtype=np.int64)[permuted_order]
        w_perm = S.balanced_weights(scaffold_by_row, rows_perm, n_train=C.N_TRAIN, k_train=int(k_train))
        vec = np.asarray([float(w_perm[int(r)]) for r in np.asarray(official["train_rows"], dtype=np.int64)])
        label_perm_sha.add(C.sha256_array(vec))
    sig = inspect.signature(S.balanced_weights)
    checks["6_weights_label_independent"] = {
        "function_signature": str(sig),
        "signature_has_no_label_argument": bool(not any("label" in p or p in ("y", "targets") for p in sig.parameters)),
        "n_label_permutations": 200,
        "n_distinct_weight_vectors_after_permutation": int(len(label_perm_sha)),
        "weight_sha256": base_sha,
        "all_permutations_reproduce_frozen_weights": bool(label_perm_sha == {base_sha}),
        "weight_inputs": "Train scaffold strings + N_train + K_train only",
        "passed": bool(label_perm_sha == {base_sha}),
    }

    # ---- 7 unit weights == plain MSE -------------------------------------
    C.set_seeds(C.SEED)
    model_a, _ = M.build_csc(bundle.spec, bundle.norm, arm="A", seed=C.SEED, device=device)
    order0 = plan["plan"][0]
    rows0 = order0[: int(C.RECIPE["batch_size"])]
    batch = D.make_batch(bundle, "train", rows0, device)
    y0 = torch.as_tensor(
        np.asarray(
            (bundle.targets_raw["train"][rows0] - bundle.scaler["mean"]) / bundle.scaler["std"], dtype=np.float32
        ),
        device=device,
    )
    ones = torch.ones_like(y0)
    model_a.train()
    pred = model_a(batch).view(-1)          # one train-mode forward, both losses read the same tensor
    loss_w = T.weighted_mse(pred, y0, ones)
    loss_plain = F.mse_loss(pred, y0)
    model_a.zero_grad(set_to_none=True)
    loss_w.backward(retain_graph=True)
    grads_w = {n: p.grad.detach().clone() for n, p in model_a.named_parameters()}
    model_a.zero_grad(set_to_none=True)
    loss_plain.backward()
    grads_plain = {n: p.grad.detach().clone() for n, p in model_a.named_parameters()}
    grad_equal = all(torch.equal(grads_w[n], grads_plain[n]) for n in grads_w)
    max_grad_diff = max(float((grads_w[n] - grads_plain[n]).abs().max()) for n in grads_w)
    checks["7_unit_weights_is_plain_mse"] = {
        "batch_rows": int(rows0.size),
        "forward_mode": "train (dropout active); both losses are computed from the same prediction tensor",
        "weighted_loss_unit_weights": float(loss_w),
        "plain_mse_loss": float(loss_plain),
        "bitwise_equal": bool(torch.equal(loss_w, loss_plain)),
        "gradients_bitwise_equal": bool(grad_equal),
        "n_parameters_compared": int(len(grads_w)),
        "max_abs_grad_diff": max_grad_diff,
        "note": "reduction='mean' over one output per molecule, so weighted MSE with w=1 is exactly F.mse_loss",
        "passed": bool(torch.equal(loss_w, loss_plain) and grad_equal),
    }

    # ---- 8 unweighted Valid metric ---------------------------------------
    model_a.eval()
    rep = D.evaluate(bundle, model_a, "valid", device)
    manual = C.rmse(rep["pred_raw"], rep["y_raw"])
    manual_mae = C.mae(rep["pred_raw"], rep["y_raw"])
    eval_sig = inspect.signature(D.evaluate)
    checks["8_valid_metric_unweighted"] = {
        "evaluate_signature": str(eval_sig),
        "evaluate_accepts_weights": bool("weight" in eval_sig.parameters),
        "n_valid_rows": int(rep["row_ids"].size),
        "reported_rmse": float(rep["rmse_raw"]),
        "hand_computed_rmse": float(manual),
        "abs_diff": float(abs(rep["rmse_raw"] - manual)),
        "reported_mae": float(rep["mae_raw"]),
        "hand_computed_mae": float(manual_mae),
        "mae_abs_diff": float(abs(rep["mae_raw"] - manual_mae)),
        "valid_row_ids_are_sorted_official_ids": bool(np.all(np.diff(rep["row_ids"]) > 0)),
        "passed": bool(abs(rep["rmse_raw"] - manual) < 1e-12 and abs(rep["mae_raw"] - manual_mae) < 1e-12
                       and "weight" not in eval_sig.parameters),
    }

    # ---- 9 preflight timing ----------------------------------------------
    C.set_seeds(C.SEED)
    model_t, _ = M.build_csc(bundle.spec, bundle.norm, arm="A", seed=C.SEED, device=device)
    opt, _ = T.build_optimizer(model_t)
    model_t.train()
    step_times: list[float] = []
    n_steps = 12
    order0 = plan["plan"][0]
    for start in range(0, n_steps * int(C.RECIPE["batch_size"]), int(C.RECIPE["batch_size"])):
        rows = order0[start:start + int(C.RECIPE["batch_size"])]
        t1 = time.perf_counter()
        batch = D.make_batch(bundle, "train", rows, device)
        yb = torch.as_tensor(
            np.asarray((bundle.targets_raw["train"][rows] - bundle.scaler["mean"]) / bundle.scaler["std"], dtype=np.float32),
            device=device,
        )
        wb = torch.as_tensor(wpack["weights"][rows].astype(np.float32), device=device)
        opt.zero_grad(set_to_none=True)
        loss = T.weighted_mse(model_t(batch).view(-1), yb, wb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_t.parameters(), float(C.RECIPE["gradient_clip"]))
        opt.step()
        step_times.append(float(time.perf_counter() - t1))
    t_eval = time.perf_counter()
    D.evaluate(bundle, model_t, "train", device)
    eval_train = float(time.perf_counter() - t_eval)
    t_eval = time.perf_counter()
    D.evaluate(bundle, model_t, "valid", device)
    eval_valid = float(time.perf_counter() - t_eval)
    steps_epoch = C.steps_per_epoch(C.DATASET)
    proj_epoch = float(np.median(step_times)) * steps_epoch + eval_train + eval_valid
    proj_total = proj_epoch * C.EPOCHS
    preflight = {
        "created_utc": C.utcnow(),
        "device": "cpu",
        "torch_threads": int(torch.get_num_threads()),
        "n_parameters": int(model_audit["n_parameters"]),
        "steps_per_epoch": int(steps_epoch),
        "step_seconds_median": float(np.median(step_times)),
        "step_seconds_min": float(np.min(step_times)),
        "step_seconds_max": float(np.max(step_times)),
        "n_timed_steps": int(len(step_times)),
        "eval_seconds": {"train": eval_train, "valid": eval_valid},
        "projected_epoch_seconds": proj_epoch,
        "projected_360_epoch_seconds": proj_total,
        "projected_360_epoch_minutes": proj_total / 60.0,
        "hard_seconds": 2100.0,
        "fits_budget": bool(proj_total < 2100.0),
        "gpu_hours": 0.0,
    }
    C.write_json(C.ANALYSIS_DIR / "preflight_timing.json", preflight)
    checks["9_preflight_timing"] = {
        "projected_360_epoch_seconds": proj_total,
        "projected_360_epoch_minutes": proj_total / 60.0,
        "hard_seconds": 2100.0,
        "fits_budget": bool(proj_total < 2100.0),
        "passed": bool(proj_total < 2100.0),
    }

    all_passed = all(bool(v.get("passed")) for v in checks.values())
    payload = {
        "schema": C.SCHEMA_CHECKS,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "seconds": float(time.perf_counter() - t0),
        "all_passed": all_passed,
        "checks": checks,
    }
    C.write_json(C.ANALYSIS_DIR / "checks.json", payload)
    for name, value in checks.items():
        C.log(f"check {name}: {'PASS' if value.get('passed') else 'FAIL'}")
    C.log(f"checks all_passed={all_passed} in {payload['seconds']:.1f}s")
    return 0 if all_passed else 5


if __name__ == "__main__":
    raise SystemExit(main())
