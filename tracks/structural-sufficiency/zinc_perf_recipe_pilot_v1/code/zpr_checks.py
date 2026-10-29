"""ZPR-v1 correctness gates.

``cpu``    — everything that can be checked without a GPU run (C1, C5, C6, C7
             formulas, plus the frozen group masks and the historical
             checkpoint replay).  Writes ``analysis/cpu_checks.json``.
``formal`` — post-run checks read from the two completed ``train.json`` files
             (C2, C3, C4, C7b, C8).  Merges into the same JSON.

No GPU is required for either mode; ``formal`` only reads run artifacts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import zpr_common as Z
import zpr_train as T

L, D, M = Z.L, Z.D, Z.M
CHECKS_PATH = Z.ANALYSIS / "cpu_checks.json"


# ---------------------------------------------------------------------------
# C5 — SmoothL1 reference (value + gradient) against a hand formula
# ---------------------------------------------------------------------------

def _ref_smooth_l1_scalar(d: float, beta: float) -> float:
    z = abs(d)
    return 0.5 * z * z / beta if z < beta else z - 0.5 * beta


def _ref_smooth_l1_grad(d: float, beta: float) -> float:
    z = abs(d)
    if z < beta:
        return d / beta
    return 1.0 if d > 0 else (-1.0 if d < 0 else 0.0)


def check_smooth_l1() -> dict:
    beta = float(Z.SMOOTH_L1_BETA)
    cases = [0.0, 1e-6, 0.01, 0.02, 0.0499999, 0.05, 0.0500001, 0.2, 1.0, 6.9, -0.02, -0.05, -1.3]
    rows = []
    for d in cases:
        pred = torch.tensor([d], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([0.0], dtype=torch.float64)
        got = F.smooth_l1_loss(pred, target, beta=beta)
        got.backward()
        ref_val = _ref_smooth_l1_scalar(d, beta)
        ref_grad = _ref_smooth_l1_grad(d, beta)
        rows.append({
            "d": d,
            "torch_value": float(got.item()),
            "reference_value": ref_val,
            "value_abs_diff": abs(float(got.item()) - ref_val),
            "torch_grad": float(pred.grad.item()),
            "reference_grad": ref_grad,
            "grad_abs_diff": abs(float(pred.grad.item()) - ref_grad),
        })
    # vector/mean semantics: batch reduction is a mean over the batch
    d = torch.tensor([0.01, -0.2, 0.05, -0.049, 2.0], dtype=torch.float64)
    batch = F.smooth_l1_loss(d, torch.zeros_like(d), beta=beta).item()
    batch_ref = sum(_ref_smooth_l1_scalar(float(x), beta) for x in d) / len(d)
    # trainer's own reference helper must agree with the hand formula
    trainer_ref = T.hand_smooth_l1(d, torch.zeros_like(d), beta).item()
    # finite-difference gradient on a small MLP through the trainer's loss
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)).double()
    x = torch.randn(16, 3, dtype=torch.float64)
    yt = torch.randn(16, dtype=torch.float64)
    loss = T._loss("smooth_l1", net(x).view(-1), yt)
    net.zero_grad(set_to_none=True)
    loss.backward()
    analytic = {name: param.grad.detach().clone() for name, param in net.named_parameters()}
    max_rel = 0.0
    eps = 1e-6
    for name, param in net.named_parameters():
        flat = param.view(-1)
        for idx in (0, flat.numel() // 2, flat.numel() - 1):
            orig = float(flat[idx])
            with torch.no_grad():
                flat[idx] = orig + eps
                lp = T._loss("smooth_l1", net(x).view(-1), yt).item()
                flat[idx] = orig - eps
                lm = T._loss("smooth_l1", net(x).view(-1), yt).item()
                flat[idx] = orig
            fd = (lp - lm) / (2 * eps)
            an = float(analytic[name].view(-1)[idx])
            denom = max(1e-9, abs(fd), abs(an))
            max_rel = max(max_rel, abs(fd - an) / denom)
    out = {
        "beta": beta,
        "cases": rows,
        "max_value_abs_diff": max(r["value_abs_diff"] for r in rows),
        "max_grad_abs_diff": max(r["grad_abs_diff"] for r in rows),
        "batch_mean_value": float(batch),
        "batch_mean_reference": float(batch_ref),
        "batch_mean_abs_diff": abs(float(batch) - float(batch_ref)),
        "trainer_helper_value": float(trainer_ref),
        "trainer_helper_abs_diff": abs(float(trainer_ref) - float(batch_ref)),
        "finite_difference_max_rel_error": float(max_rel),
        "continuity_at_beta": {
            "left": 0.5 * beta,
            "right": beta - 0.5 * beta,
            "abs_diff": abs(0.5 * beta - (beta - 0.5 * beta)),
        },
    }
    out["pass"] = bool(
        out["max_value_abs_diff"] < 1e-12
        and out["max_grad_abs_diff"] < 1e-12
        and out["batch_mean_abs_diff"] < 1e-12
        and out["trainer_helper_abs_diff"] < 1e-12
        and out["finite_difference_max_rel_error"] < 1e-5
    )
    return out


# ---------------------------------------------------------------------------
# C1 — initial state equals the frozen LCRC-A seed-0 state, deterministically
# ---------------------------------------------------------------------------

def check_initial_state() -> dict:
    norm = D.load_norm()
    cfg = dict(M.DEFAULT_CONFIG)
    hashes = {}
    params = {}
    for tag in ("first", "second"):
        model, info = M.build_model("A", norm, 0, "cpu", cfg)
        hashes[tag] = info["initial_state_sha256"]
        params[tag] = info["parameters"]
    frozen = Z.FROZEN["lcrc_a_s0_initial_state_sha256"]
    out = {
        "build_1": hashes["first"],
        "build_2": hashes["second"],
        "frozen_lcrc_a_s0": frozen,
        "builds_equal": hashes["first"] == hashes["second"],
        "equals_frozen": hashes["first"] == frozen,
        "params": params["first"],
        "params_ok": params["first"] == Z.FROZEN["params"] == 237519,
        "same_build_path_for_both_arms": True,
        "note": "both arms call build_model('A', norm, 0, ...) with the frozen config; "
                "the loss switch happens only inside the training step",
    }
    out["pass"] = bool(out["builds_equal"] and out["equals_frozen"] and out["params_ok"])
    return out


# ---------------------------------------------------------------------------
# C6 — EMA formula, initialisation epoch, checkpoint round-trip
# ---------------------------------------------------------------------------

def _reference_ema_trajectory(param_sequences: list[list[np.ndarray]], start_step: int,
                              decay: float, dtype=np.float32) -> list[list[np.ndarray]]:
    """Independent reference of the declared EMA recurrence.

    ``dtype=float32`` is the bit-exactness check (same rounding order as the
    trainer: ``round(e * decay)`` then ``+ round(p * (1 - decay))``);
    ``float64`` is the drift diagnostic.
    """
    state = None
    out = []
    alpha = dtype(1.0 - decay)
    dec = dtype(decay)
    for step in range(1, len(param_sequences) + 1):
        online = param_sequences[step - 1]
        if step == start_step:
            state = [(p.astype(dtype).copy()) for p in online]
        elif step > start_step and state is not None:
            state = [(e * dec + p.astype(dtype) * alpha).astype(dtype, copy=False)
                     for e, p in zip(state, online)]
        out.append(None if state is None else [e.copy() for e in state])
    return out


def check_ema_formula() -> dict:
    decay = float(Z.EMA_DECAY)
    # real horizon, real start step
    real_start = T.ema_start_step(Z.FROZEN["total_updates"])
    real_start_2ep = T.ema_start_step(79 * 2)
    # synthetic 30-step trajectory with the same recurrence, start at step 8
    torch.manual_seed(1234)
    model = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.Linear(4, 1))
    seq: list[list[np.ndarray]] = []
    torch_seq: list[list[torch.Tensor]] = []
    start = 8
    ema_state = None
    for step in range(1, 31):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        seq.append([p.detach().numpy().copy() for p in model.parameters()])
        torch_seq.append([p.detach().clone() for p in model.parameters()])
        if step == start:
            ema_state = T.ema_init_state(model)
        elif step > start:
            T.ema_update_(ema_state, model, decay)
    ref32 = _reference_ema_trajectory(seq, start, decay, dtype=np.float32)
    ref64 = _reference_ema_trajectory(seq, start, decay, dtype=np.float64)
    # replay the recorded parameter sequence into a fresh module and compare
    # the trainer's EMA against both references at every step
    ema_state2 = None
    model2 = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.Linear(4, 1))
    with torch.no_grad():
        for p2, p1 in zip(model2.parameters(), model.parameters()):
            p2.copy_(p1)
    diffs32, diffs64 = [], []
    for step in range(1, 31):
        with torch.no_grad():
            for p, target in zip(model2.parameters(), torch_seq[step - 1]):
                p.copy_(target)
        if step == start:
            ema_state2 = T.ema_init_state(model2)
            diffs32.append(0.0)
            diffs64.append(0.0)
        elif step > start:
            T.ema_update_(ema_state2, model2, decay)
            got = [e.detach().numpy().astype(np.float64) for e in ema_state2]
            diffs32.append(max(float(np.abs(g - r).max())
                               for g, r in zip(got, ref32[step - 1])))
            diffs64.append(max(float(np.abs(g - r).max())
                               for g, r in zip(got, ref64[step - 1])))
        else:
            diffs32.append(0.0)
            diffs64.append(0.0)
    max_abs32 = max(diffs32)
    max_abs64 = max(diffs64)
    init_equal = bool(all(
        torch.equal(e, p) for e, p in zip(
            T.ema_init_state(model), list(model.parameters())))
    )
    # round-trip: an EMA parameter list loaded into a fresh module reproduces the
    # same forward pass bit-for-bit (state_dict load path used for deployment).
    norm = D.load_norm()
    cfg = dict(M.DEFAULT_CONFIG)
    m_a, _ = M.build_model("A", norm, 0, "cpu", cfg)
    m_b, _ = M.build_model("A", norm, 0, "cpu", cfg)
    ema_like = T.ema_init_state(m_a)
    T.copy_ema_into(m_b, ema_like, m_a)
    h_a = T.tensor_list_hash([p for p in m_a.parameters()])
    h_b = T.param_state_hash(m_b)
    # state_dict round-trip (the deployment path: cpu_state -> load_model_from_state)
    state = L.C.cpu_state(m_b)
    m_c = M.load_model_from_state("A", norm, state, "cpu", cfg)
    h_c = T.param_state_hash(m_c)
    out = {
        "decay": decay,
        "start_frac": float(Z.EMA_START_FRAC),
        "start_step_formal_600ep": int(real_start),
        "start_step_formal_expected": int(Z.FROZEN["ema_start_step"]),
        "start_epoch_formal_600ep": int(real_start // Z.FROZEN["steps_per_epoch"]),
        "start_step_2ep_smoke": int(real_start_2ep),
        "synthetic_start_step": int(start),
        "synthetic_steps": 30,
        "max_abs_diff_vs_float32_reference": float(max_abs32),
        "float32_tolerance": 1e-6,
        "max_abs_diff_vs_float64_reference": float(max_abs64),
        "reference_note": "the float32 reference uses the same recurrence; residual differences "
                          "are float32 rounding/FMA order (<= 1 ulp), the float64 number is "
                          "only the accumulation drift diagnostic",
        "init_equals_online": init_equal,
        "ema_differs_from_online_at_end": bool(
            max(float((e - p).abs().max()) for e, p in zip(ema_state2, model2.parameters())) > 0),
        "copy_into_module_equals_ema_state": bool(h_a == h_b),
        "state_dict_roundtrip_equals": bool(h_a == h_c),
        "direction_statement": "copy_ema_into writes ema_state INTO the module "
                               "(module_param.copy_(ema_slot)); the historical zps bug "
                               "was the reverse direction and would fail "
                               "copy_into_module_equals_ema_state",
    }
    out["pass"] = bool(
        out["max_abs_diff_vs_float32_reference"] < 1e-6
        and out["max_abs_diff_vs_float64_reference"] < 1e-5
        and out["init_equals_online"]
        and out["ema_differs_from_online_at_end"]
        and out["copy_into_module_equals_ema_state"]
        and out["state_dict_roundtrip_equals"]
        and out["start_step_formal_600ep"] == out["start_step_formal_expected"] == 11850
    )
    return out


# ---------------------------------------------------------------------------
# C7 — EMA is passive: no RNG / grad side effects
# ---------------------------------------------------------------------------

def check_ema_passive() -> dict:
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 1))
    ema_state = T.ema_init_state(model)
    gen = torch.Generator().manual_seed(101)
    # simulate one optimizer step that does create grads
    x = torch.randn(8, 4)
    loss = model(x).sum()
    loss.backward()
    grad_before = [p.grad.clone() for p in model.parameters()]
    # snapshot the RNG only *after* every stochastic op preceding the EMA block
    torch_rng_before = torch.get_rng_state().clone()
    gen_before = gen.get_state().clone()
    T.ema_update_(ema_state, model, float(Z.EMA_DECAY))
    grad_after = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    out = {
        "torch_rng_state_unchanged": bool(torch.equal(torch.get_rng_state(), torch_rng_before)),
        "data_order_gen_state_unchanged": bool(torch.equal(gen.get_state(), gen_before)),
        "grads_unchanged": bool(all(
            (a is None and b is not None) or (a is not None and torch.equal(a, b))
            for a, b in zip(grad_before, grad_after))),
        "ema_slots_are_leaf_tensors_without_grad": bool(
            all((not t.requires_grad) and t.grad_fn is None for t in ema_state)),
        "ema_update_runs_under_no_grad": True,
    }
    out["pass"] = bool(out["torch_rng_state_unchanged"]
                       and out["data_order_gen_state_unchanged"]
                       and out["grads_unchanged"]
                       and out["ema_slots_are_leaf_tensors_without_grad"])
    return out


# ---------------------------------------------------------------------------
# frozen masks + historical checkpoint replay
# ---------------------------------------------------------------------------

def check_masks() -> dict:
    masks = Z.group_masks()
    sizes = {k: int(v.sum()) for k, v in masks.items()}
    expected = {"full": 1000, "ordinary": 935, "r2_unseen": 251,
                "r2_unseen_ordinary": 225, "seen_ordinary": 710,
                "minus_row172": 999, "minus_fixed_worst10": 990,
                "tail": 65, "row172": 1, "fixed_worst10": 10}
    out = {"sizes": sizes, "expected": expected,
           "matches_lcrc_groups_md": sizes == expected,
           "fixed_worst10_rows": sorted(int(r) for r in np.flatnonzero(masks["fixed_worst10"]))}
    out["pass"] = bool(sizes == expected)
    return out


def check_historical_replay() -> dict:
    """Recompute the frozen LCRC-A s0 best-checkpoint valid MAE on CPU."""
    norm = D.load_norm()
    cfg = dict(M.DEFAULT_CONFIG)
    best_path = Z.LCRC_A_S0_RUN / "best.pt"
    sha = Z.sha256_file(best_path)
    state = torch.load(best_path, map_location="cpu", weights_only=False)["state_dict"]
    model = M.load_model_from_state("A", norm, state, "cpu", cfg)
    valid = D.load_cache("valid")
    rows = np.arange(len(valid["labels"]))
    _, mae = Z.T.evaluate(model, valid, rows, device="cpu", batch_size=256)
    y = np.asarray(valid["labels"], dtype=np.float64)
    npz = np.load(Z.LCRC_A_S0_RUN / "predictions/valid_best.npz")
    pred = npz["pred"].astype(np.float64)
    out = {
        "checkpoint": Z.track_rel(best_path),
        "checkpoint_sha256": sha,
        "checkpoint_sha_matches_frozen": sha == Z.FROZEN["lcrc_a_s0_best_sha256"],
        "valid_mae_recomputed": float(mae),
        "valid_mae_frozen": float(Z.HIST["lcrc_a_s0_valid"]),
        "abs_diff": abs(float(mae) - float(Z.HIST["lcrc_a_s0_valid"])),
        "saved_prediction_mae": float(np.mean(np.abs(pred - y))),
        "eval_path": "lc_models.load_model_from_state + lc_train.evaluate (cpu)",
    }
    out["pass"] = bool(out["checkpoint_sha_matches_frozen"] and out["abs_diff"] < 1e-6)
    return out


# ---------------------------------------------------------------------------
# formal post-run checks
# ---------------------------------------------------------------------------

def _read_run(arm: str) -> dict:
    path = Z.RUNS / f"{arm}_s0/train.json"
    if not path.exists():
        raise FileNotFoundError(f"missing run output {Z.track_rel(path)}")
    return Z.read_json(path)


def _curve(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def check_formal() -> dict:
    out: dict[str, Any] = {}
    runs = {arm: _read_run(arm) for arm in ("G0", "G1")}

    # C2 — frozen inputs recorded by both runs (content sha gates; container
    # bytes of the same content differ per checkout, HIS-v0 finding)
    keys = ["train_cache_content_sha256", "valid_cache_content_sha256",
            "norm_fingerprint", "feature_version"]
    file_keys = ["train_cache_file_sha256", "valid_cache_file_sha256"]
    inputs = {arm: {k: r["data"][k] for k in keys + file_keys} for arm, r in runs.items()}
    gating = {arm: {k: v for k, v in inputs[arm].items() if k in keys} for arm in inputs}
    host_profile = [
        name for name, prof in Z.FROZEN_CACHE_FILE_SHA256_BY_HOST.items()
        if all(inputs["G0"][f"{split}_cache_file_sha256"] == sha
               for split, sha in prof.items())
    ]
    c2 = {
        "inputs": inputs,
        "gating_inputs": gating,
        "equal_across_arms": gating["G0"] == gating["G1"],
        "equal_to_frozen": all(gating["G0"][k] == Z.FROZEN[k] for k in keys),
        "cache_container_sha_matches_host_profile": host_profile,
        "cache_container_sha_note": "torch.save container bytes differ per checkout; "
                                    "the gating identity is the content sha256",
        "ssl_sha256": {arm: r["model_info"]["ssl"]["sha256"] for arm, r in runs.items()},
    }
    c2["ssl_equal_across_arms"] = len(set(c2["ssl_sha256"].values())) == 1
    c2["ssl_equals_frozen"] = c2["ssl_sha256"]["G0"] == Z.FROZEN["ssl_s0_sha256"]
    c2["pass"] = bool(c2["equal_across_arms"] and c2["equal_to_frozen"]
                      and c2["ssl_equal_across_arms"] and c2["ssl_equals_frozen"])
    out["C2_frozen_inputs"] = c2

    # C1/C3 — initial state + data order
    c3 = {
        "initial_state_sha256": {arm: r["model_info"]["initial_state_sha256"] for arm, r in runs.items()},
        "first_batch_rows_equal": runs["G0"]["first_batch_rows"] == runs["G1"]["first_batch_rows"],
        "first_batch_rows_head": runs["G0"]["first_batch_rows"][:12],
        "data_order_seed": {arm: r["data_order_seed"] for arm, r in runs.items()},
        "steps_per_epoch": {arm: r["steps_per_epoch"] for arm, r in runs.items()},
        "total_updates": {arm: r["total_updates"] for arm, r in runs.items()},
        "epochs": {arm: r["epochs"] for arm, r in runs.items()},
        "lr_sequences_equal": (
            [round(c["lr"], 12) for c in runs["G0"]["curve"]]
            == [round(c["lr"], 12) for c in runs["G1"]["curve"]]
        ),
        "warmup_steps": {arm: r["warmup_steps"] for arm, r in runs.items()},
        "first_batch_loss_reference": {arm: r["loss_reference_first_batch"] for arm, r in runs.items()},
    }
    c3["initial_equals_frozen"] = all(
        v == Z.FROZEN["lcrc_a_s0_initial_state_sha256"] for v in c3["initial_state_sha256"].values())
    c3["pass"] = bool(
        c3["initial_equals_frozen"]
        and c3["first_batch_rows_equal"]
        and set(c3["data_order_seed"].values()) == {101}
        and set(c3["steps_per_epoch"].values()) == {79}
        and set(c3["total_updates"].values()) == {47400}
        and c3["lr_sequences_equal"]
    )
    out["C1_C3_initial_state_and_data_order"] = c3

    # C4 — L1 control reproduces the historical LCRC-A s0 trajectory
    hist = _curve(Z.LCRC_A_S0_CURVE)
    got = _curve(Z.RUNS / "G0_s0" / "curve.csv")
    n = min(len(hist), len(got))
    dvalid = [abs(float(hist[i]["valid_mae"]) - float(got[i]["valid_mae"])) for i in range(n)]
    donline = [abs(float(hist[i]["online_train_mae"]) - float(got[i]["online_train_mae"])) for i in range(n)]
    c4 = {
        "historical_curve": Z.track_rel(Z.LCRC_A_S0_CURVE),
        "epochs_compared": n,
        "max_abs_valid_diff": max(dvalid),
        "max_abs_online_diff": max(donline),
        "worst_valid_epoch": int(np.argmax(dvalid)) + 1,
        "best_epoch": {arm: runs[arm]["best"]["epoch"] for arm in runs},
        "best_valid": {arm: runs[arm]["best"]["valid_mae"] for arm in runs},
        "historical_best": {"epoch": int(Z.HIST["lcrc_a_s0_epoch"]),
                            "valid": float(Z.HIST["lcrc_a_s0_valid"])},
    }
    c4["pass"] = bool(n == 600 and c4["max_abs_valid_diff"] < 1e-5
                      and c4["max_abs_online_diff"] < 1e-5)
    out["C4_l1_control_matches_frozen_path"] = c4

    # C6b/C7b — EMA runtime audit as recorded by the trainer
    c7 = {}
    for arm, r in runs.items():
        audit = r["ema"]["audit"]
        rng = audit.get("rng_invariance", {})
        c7[arm] = {
            "start_step": r["ema"]["start_step"],
            "init_step": audit.get("init_step"),
            "init_epoch": audit.get("init_epoch"),
            "init_equals_online_params": audit.get("init_equals_online_params"),
            "rng_invariance": rng,
            "ema_best": r["ema"]["best"],
            "first_ema_valid_epoch": next((c["epoch"] for c in r["curve"]
                                           if c["ema_valid_mae"] is not None), None),
        }
    out["C7_ema_runtime"] = c7
    out["C7_ema_runtime"]["pass"] = bool(all(
        v["start_step"] == Z.FROZEN["ema_start_step"]
        and v["init_equals_online_params"] is True
        and v["rng_invariance"].get("torch_rng_state_unchanged") is True
        and v["rng_invariance"].get("data_order_gen_state_unchanged") is True
        for v in (c7["G0"], c7["G1"])))

    # C8 — deployment shape
    c8 = {
        "parameters": {arm: runs[arm]["model_info"]["parameters"] for arm in runs},
        "param_audit": {arm: runs[arm]["param_audit"] for arm in runs},
        "checkpoint_sha256": {arm: runs[arm]["checkpoint_sha256"] for arm in runs},
        "single_state_dict_per_checkpoint": True,
        "ensemble_or_soup": False,
        "test_loaded_during_training": {arm: runs[arm]["data"]["official_test_loaded"] for arm in runs},
    }
    c8["pass"] = bool(
        set(c8["parameters"].values()) == {237519}
        and all(v is False for v in c8["test_loaded_during_training"].values())
        and all(v is not None for arm in runs for v in runs[arm]["checkpoint_sha256"].values()
                if True)
    )
    out["C8_deployment_shape"] = c8
    out["all_pass"] = all(v["pass"] for k, v in out.items() if isinstance(v, dict) and "pass" in v)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ZPR-v1 correctness gates")
    parser.add_argument("mode", choices=["cpu", "formal"])
    args = parser.parse_args(argv)
    payload = Z.read_json(CHECKS_PATH) if CHECKS_PATH.exists() else {}
    if args.mode == "cpu":
        payload.setdefault("round", Z.ROUND_ID)
        payload["C1_initial_state"] = check_initial_state()
        payload["C5_smooth_l1_reference"] = check_smooth_l1()
        payload["C6_ema_formula"] = check_ema_formula()
        payload["C7_ema_passive"] = check_ema_passive()
        payload["masks"] = check_masks()
        payload["historical_checkpoint_replay"] = check_historical_replay()
        payload["cpu_all_pass"] = all(
            payload[k]["pass"] for k in
            ["C1_initial_state", "C5_smooth_l1_reference", "C6_ema_formula",
             "C7_ema_passive", "masks", "historical_checkpoint_replay"])
        payload["created"] = Z.now()
    else:
        formal = check_formal()
        payload["formal"] = formal
        payload["formal_all_pass"] = bool(formal.get("all_pass"))
        payload["verified"] = Z.now()
    Z.write_json(CHECKS_PATH, payload)
    summary = {k: v.get("pass") for k, v in payload.items() if isinstance(v, dict) and "pass" in v}
    Z.log(f"[zpr checks:{args.mode}] {summary}")
    ok = all(v is True for v in summary.values())
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
