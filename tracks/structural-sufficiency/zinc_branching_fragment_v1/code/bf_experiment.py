"""One P3-BF trajectory: timing -> fixed E -> training -> best-valid freeze -> test.

The GPU budget is an incremental ~60 minutes.  The timing stage runs the same
P3B-initialised model for a few full epochs (train + full valid) and the
horizon is fixed before the formal run; after timing the model is rebuilt from
the identical P3B checkpoint, the new modules are re-seeded, and a fresh
optimizer is created.  No cosine reset, no horizon extension.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
_V2 = _HERE.parents[1] / "zinc_recipe_trunk_v2" / "code"
_PA = _HERE.parents[1] / "zinc_restricted_adaptation_v1"
for _p in (str(_HERE), str(_V1), str(_V2)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import bf_checks  # noqa: E402
import bf_data as B  # noqa: E402
import bf_models as M  # noqa: E402

EXP = _HERE.parent
RUNS = EXP / "runs"
ANALYSIS = EXP / "analysis"
RUN = RUNS / "bf_s0"

SEED = 0
BATCH = 128
STEPS_PER_EPOCH = math.ceil(10000 / BATCH)  # 79
WARMUP = 200
WD = 1e-5
CLIP = 5.0
PEAK_INHERITED, MIN_INHERITED = 1e-4, 1e-6
PEAK_NEW, MIN_NEW = 1e-3, 1e-5
REF_IMPROVEMENT = 5e-4
MIN_EPOCHS = 100
PATIENCE = 100
GROUPS = (
    ("covered_ordinary", 0, 1),
    ("covered_tail", 0, 0),
    ("unseen_ordinary", 1, 1),
    ("unseen_tail", 1, 0),
)
EXPECTED = {
    "valid": {"P3B_s0": 0.07622375367576023, "L_s0": 0.078537152, "DBT_s0": 0.077768402},
    "test": {"P3B_s0": 0.069864841, "L_s0": 0.066266555, "DBT_s0": 0.069401494},
}


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def make_optimizer(model):
    inherited, new = M.param_groups(model)
    return torch.optim.AdamW(
        [
            {"params": [p for _, p in inherited], "lr": PEAK_INHERITED},
            {"params": [p for _, p in new], "lr": PEAK_NEW},
        ],
        weight_decay=WD,
    )


def set_lrs(optimizer, step: int, total_updates: int) -> tuple[float, float]:
    lr_old = C.cosine_with_warmup_lr(step, total_updates, WARMUP, PEAK_INHERITED, MIN_INHERITED)
    lr_new = C.cosine_with_warmup_lr(step, total_updates, WARMUP, PEAK_NEW, MIN_NEW)
    optimizer.param_groups[0]["lr"] = lr_old
    optimizer.param_groups[1]["lr"] = lr_new
    return lr_old, lr_new


def train_epoch(model, optimizer, train_cache, rows, y, gen, *, epoch, total_updates,
                global_step, log_prefix="") -> tuple[float, int, float, float]:
    model.train()
    device = model_device(model)
    order = torch.randperm(len(rows), generator=gen).numpy()
    total_err, n_mol = 0.0, 0
    lr_old = lr_new = 0.0
    for start in range(0, len(order), BATCH):
        chunk = order[start:start + BATCH]
        batch = T1._to_device(B.collate_bf(train_cache, chunk), device)
        target = torch.as_tensor(y[chunk], dtype=torch.float32, device=device)
        pred = model(batch)
        loss = F.l1_loss(pred.view(-1), target.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        if not math.isfinite(float(gn)):
            raise RuntimeError(f"non-finite gradient at epoch {epoch}")
        global_step += 1
        lr_old, lr_new = set_lrs(optimizer, global_step, total_updates)
        optimizer.step()
        total_err += float((pred.detach().view(-1) - target).abs().sum())
        n_mol += len(chunk)
    return total_err / max(n_mol, 1), global_step, lr_old, lr_new


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def evaluate(model, cache, rows, device, batch_size=256):
    return T1.evaluate(model, cache, rows, candidate="T", device=device,
                       batch_size=batch_size, collate_fn=B.collate_bf)


def load_split(split: str):
    cache = D.load_cache(split)
    y = np.asarray(cache["labels"], dtype=np.float64)
    return cache, y, np.arange(len(y), dtype=np.int64)


def gate() -> dict:
    """Refuse GPU work unless the declared CPU representation gates passed."""
    path = ANALYSIS / "cpu_checks.json"
    if not path.exists():
        raise RuntimeError("missing analysis/cpu_checks.json; run bf_checks.py first")
    checks = C.read_json(path)["checks"]
    failed = [k for k, v in checks.items() if not v.get("pass")]
    if failed:
        raise RuntimeError(f"CPU checks failed: {failed}")
    rep = C.read_json(ANALYSIS / "representation_check.json")
    stats = C.read_json(ANALYSIS / "cache_stats.json")
    occ = sum(stats["train"]["meta"]["occurrences_by_family"].values())
    if occ <= 0:
        raise RuntimeError("no real branching objects in train")
    ref = rep["refinement"]
    if ref["n_roots_in_subdivided_groups"] <= 0:
        raise RuntimeError("the new encoding does not split any raw108-confounded root group")
    return {
        "cpu_checks_passed": True,
        "train_branching_occurrences": int(occ),
        "subdivided_roots": int(ref["n_roots_in_subdivided_groups"]),
        "fraction_roots_subdivided": float(ref["fraction_roots_subdivided"]),
    }


# ---------------------------------------------------------------------------
# timing + plan
# ---------------------------------------------------------------------------


def run_timing(norm, train_cache, y_train, rows_train, valid_cache, rows_valid,
               device, *, epochs: int) -> dict:
    model, info = M.build_model(norm, seed=SEED, device=device)
    optimizer = make_optimizer(model)
    gen = torch.Generator().manual_seed(SEED + 101)
    total_updates = STEPS_PER_EPOCH * 600
    global_step = 0
    curve = []
    for ep in range(1, int(epochs) + 1):
        t0 = time.perf_counter()
        online, global_step, lr_old, lr_new = train_epoch(
            model, optimizer, train_cache, rows_train, y_train, gen,
            epoch=ep, total_updates=total_updates, global_step=global_step,
        )
        _, valid_mae = evaluate(model, valid_cache, rows_valid, device)
        sec = time.perf_counter() - t0
        curve.append({
            "epoch": ep, "online_train_mae": float(online),
            "valid_mae": float(valid_mae), "seconds": float(sec),
            "lr_inherited": float(lr_old), "lr_new": float(lr_new),
        })
        C.log(f"[bf-timing] ep={ep} valid={valid_mae:.6f} online={online:.6f} {sec:.2f}s")
    steady = float(np.mean([e["seconds"] for e in curve[1:]])) if len(curve) > 1 else \
        float(curve[0]["seconds"])
    return {
        "candidate": "P3-BF", "seed": SEED, "timing_epochs": int(epochs),
        "curve": curve,
        "steady_epoch_seconds": steady,
        "parameters": info["parameters"],
    }


def plan_from_timing(timing: dict, used_seconds: float, *, budget_seconds: float,
                     reserve_seconds: float, max_epochs: int) -> dict:
    t_epoch = float(timing["steady_epoch_seconds"])
    raw = (budget_seconds - used_seconds - reserve_seconds) / (1.15 * t_epoch)
    e = int(math.floor(raw))
    e = (e // 10) * 10
    e = max(0, min(int(max_epochs), e))
    return {
        "protocol": "zinc-branching-fragment-v1",
        "candidate": "P3-BF",
        "seed": SEED,
        "budget_seconds": float(budget_seconds),
        "reserve_seconds": float(reserve_seconds),
        "used_gpu_seconds_before_plan": float(used_seconds),
        "steady_epoch_seconds": t_epoch,
        "speed_margin": 1.15,
        "raw_epoch_estimate": float(raw),
        "epochs": int(e),
        "steps_per_epoch": STEPS_PER_EPOCH,
        "total_updates": int(STEPS_PER_EPOCH * e),
        "feasible": bool(e >= MIN_EPOCHS),
        "param_groups": {
            "inherited_p3": {"peak_lr": PEAK_INHERITED, "min_lr": MIN_INHERITED},
            "new": {"peak_lr": PEAK_NEW, "min_lr": MIN_NEW},
        },
        "warmup_updates": WARMUP,
        "early_stop": {"min_epochs": MIN_EPOCHS, "improvement": REF_IMPROVEMENT,
                       "patience": PATIENCE},
        "created": C.utcnow(),
    }


# ---------------------------------------------------------------------------
# formal run
# ---------------------------------------------------------------------------


def run_formal(*, epochs: int, norm, train_cache, y_train, rows_train,
               valid_cache, y_valid, rows_valid, device, epoch0: dict) -> dict:
    model, info = M.build_model(norm, seed=SEED, device=device)
    optimizer = make_optimizer(model)
    gen = torch.Generator().manual_seed(SEED + 101)
    total_updates = STEPS_PER_EPOCH * int(epochs)
    RUN.mkdir(parents=True, exist_ok=True)

    # epoch 0 (inherited P3B) is a candidate checkpoint
    valid0, epoch0_valid = evaluate(model, valid_cache, rows_valid, device)
    assert abs(epoch0_valid - epoch0["valid_mae"]) < 1e-6, (epoch0_valid, epoch0)
    C.atomic_torch_save(
        {"state_dict": C.cpu_state(model), "epoch": 0, "valid_mae": float(epoch0_valid),
         "candidate": "P3-BF", "seed": SEED, "source": "inherited P3B_s0 best.pt"},
        RUN / "epoch0.pt",
    )
    C.atomic_torch_save(
        {"state_dict": C.cpu_state(model), "epoch": 0, "valid_mae": float(epoch0_valid),
         "best": {"valid_mae": float(epoch0_valid), "epoch": 0},
         "global_step": 0, "candidate": "P3-BF", "seed": SEED},
        RUN / "best.pt",
    )
    best = {"valid_mae": float(epoch0_valid), "epoch": 0}
    reference = float(epoch0_valid)
    patience = 0
    curve: list[dict] = [{
        "epoch": 0, "online_train_mae": None, "valid_mae": float(epoch0_valid),
        "lr_inherited": None, "lr_new": None, "seconds": None, "steps": 0,
        "reference_valid": float(reference), "patience": 0, "is_best": True,
        "early_stop": False,
    }]
    C.write_curve_csv(RUN / "curve.csv", curve)
    global_step = 0
    stopped_early = False
    started = time.perf_counter()
    for ep in range(1, int(epochs) + 1):
        t0 = time.perf_counter()
        online, global_step, lr_old, lr_new = train_epoch(
            model, optimizer, train_cache, rows_train, y_train, gen,
            epoch=ep, total_updates=total_updates, global_step=global_step,
        )
        _, valid_mae = evaluate(model, valid_cache, rows_valid, device)
        sec = time.perf_counter() - t0
        if reference - float(valid_mae) >= REF_IMPROVEMENT:
            reference = float(valid_mae)
            patience = 0
        else:
            patience += 1
        is_best = float(valid_mae) < best["valid_mae"]
        if is_best:
            best = {"valid_mae": float(valid_mae), "epoch": int(ep)}
            C.atomic_torch_save(
                {"state_dict": C.cpu_state(model), "epoch": int(ep),
                 "valid_mae": float(valid_mae), "candidate": "P3-BF", "seed": SEED,
                 "global_step": int(global_step)},
                RUN / "best.pt",
            )
        entry = {
            "epoch": int(ep), "online_train_mae": float(online),
            "valid_mae": float(valid_mae), "lr_inherited": float(lr_old),
            "lr_new": float(lr_new), "seconds": float(sec),
            "steps": STEPS_PER_EPOCH, "reference_valid": float(reference),
            "patience": int(patience), "is_best": bool(is_best), "early_stop": False,
        }
        curve.append(entry)
        C.atomic_torch_save(
            {"state_dict": C.cpu_state(model), "epoch": int(ep),
             "valid_mae": float(valid_mae), "best": dict(best),
             "global_step": int(global_step), "candidate": "P3-BF", "seed": SEED},
            RUN / "last.pt",
        )
        C.log(f"[bf s{SEED}] ep={ep:03d} valid={valid_mae:.6f} online={online:.6f} "
              f"best={best['valid_mae']:.6f}@{best['epoch']} patience={patience} {sec:.2f}s")
        if ep % 5 == 0 or ep == int(epochs):
            C.write_curve_csv(RUN / "curve.csv", curve)
        if ep >= MIN_EPOCHS and patience >= PATIENCE:
            stopped_early = True
            curve[-1]["early_stop"] = True
            C.write_curve_csv(RUN / "curve.csv", curve)
            C.log(f"[bf s{SEED}] early stop at epoch {ep}: {PATIENCE} epochs without "
                  f">= {REF_IMPROVEMENT} improvement over the stopping reference")
            break
    formal_seconds = time.perf_counter() - started
    epochs_run = len(curve) - 1
    assert global_step == STEPS_PER_EPOCH * epochs_run, (global_step, epochs_run)
    C.atomic_torch_save(
        {"state_dict": C.cpu_state(model), "optimizer": optimizer.state_dict(),
         "gen_state": gen.get_state(), "epoch": int(epochs_run),
         "global_step": int(global_step), "best": dict(best), "curve": curve,
         "candidate": "P3-BF", "seed": SEED, "stopped_early": bool(stopped_early)},
        RUN / "final.pt",
    )
    C.write_curve_csv(RUN / "curve.csv", curve)
    return {
        "model": model, "info": info, "curve": curve, "best": best,
        "epoch0_valid": float(epoch0_valid), "epochs_run": int(epochs_run),
        "global_step": int(global_step), "total_updates": int(total_updates),
        "formal_seconds": float(formal_seconds), "stopped_early": bool(stopped_early),
        "valid0_pred": valid0,
    }


def score_checkpoints(*, info, norm, train_cache, y_train, rows_train,
                      valid_cache, y_valid, rows_valid, device, epoch0_valid) -> dict:
    ck = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)
    best_model, _ = M.build_model(norm, seed=SEED, device=device)
    C.load_state(best_model, ck["state_dict"], device)
    pred_valid, mae_valid = evaluate(best_model, valid_cache, rows_valid, device)
    _, train_eval_best = T1.evaluate(best_model, train_cache, rows_train, candidate="T",
                                     device=device, batch_size=256, collate_fn=B.collate_bf)
    pred_dir = RUN / "predictions"
    pred_dir.mkdir(exist_ok=True)
    np.savez(pred_dir / "valid_best.npz", row=rows_valid, pred=pred_valid)
    final = torch.load(RUN / "final.pt", map_location="cpu", weights_only=False)
    final_model, _ = M.build_model(norm, seed=SEED, device=device)
    C.load_state(final_model, final["state_dict"], device)
    _, mae_valid_final = evaluate(final_model, valid_cache, rows_valid, device)
    _, train_eval_final = T1.evaluate(final_model, train_cache, rows_train, candidate="T",
                                      device=device, batch_size=256, collate_fn=B.collate_bf)
    return {
        "best": {"epoch": int(ck["epoch"]), "valid_mae": float(mae_valid),
                 "valid_mae_stored": float(ck["valid_mae"]),
                 "train_mae_eval_mode": float(train_eval_best)},
        "final": {"epoch": int(final["epoch"]), "valid_mae": float(mae_valid_final),
                  "train_mae_eval_mode": float(train_eval_final)},
        "valid_pred": pred_valid,
        "best_sha256": C.sha256_file(RUN / "best.pt"),
        "final_sha256": C.sha256_file(RUN / "final.pt"),
        "epoch0_valid": float(epoch0_valid),
        "best_is_epoch0": bool(int(ck["epoch"]) == 0),
    }


# ---------------------------------------------------------------------------
# freeze + test / groups
# ---------------------------------------------------------------------------


def ensure_test_caches() -> dict:
    out = {}
    ad_path = D.cache_path("test")
    if not ad_path.exists():
        C.log("[bf-test] building ad_test cache from official PyG (post-freeze)")
        data = D.build_split_v2("test")
        D.save_cache(data, "test")
    out["ad_test"] = {"path": str(ad_path.relative_to(C.REPO)),
                      "sha256": C.sha256_file(ad_path),
                      "size_bytes": int(ad_path.stat().st_size)}
    if not B.cache_path("test").exists():
        C.log("[bf-test] building fragment test cache (structure only, post-freeze)")
        B.save_split("test")
    out["bf_test"] = {"path": str(B.cache_path("test").relative_to(C.REPO)),
                      "sha256": C.sha256_file(B.cache_path("test")),
                      "size_bytes": int(B.cache_path("test").stat().st_size),
                      "meta": B.load_cache("test")["meta"]}
    return out


def read_reference(split: str):
    path = _PA / "analysis" / f"reference_preds_{split}.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1000:
        raise RuntimeError(f"{path}: expected 1000 rows, got {len(rows)}")
    row = np.asarray([int(r["row"]) for r in rows], dtype=np.int64)
    if not np.array_equal(row, np.arange(1000)):
        raise RuntimeError(f"{path}: row ids are not 0..999")
    y = np.asarray([float(r["y_raw"]) for r in rows], dtype=np.float64)
    refs = {name: np.asarray([float(r[f"pred_{name}"]) for r in rows], dtype=np.float64)
            for name in ("P3B_s0", "L_s0", "DBT_s0")}
    for name, pred in refs.items():
        got = float(np.abs(pred - y).mean())
        if abs(got - EXPECTED[split][name]) > 1e-6:
            raise RuntimeError(f"{split} reference {name} MAE {got} != {EXPECTED[split][name]}")
    unseen = np.asarray([int(r["signature_unseen"]) for r in rows], dtype=np.int64)
    ordinary = np.asarray([int(r["ordinary_y_gt_minus3"]) for r in rows], dtype=np.int64)
    if not np.array_equal(ordinary + np.asarray([int(r["tail_y_le_minus3"]) for r in rows]),
                          np.ones(1000, dtype=np.int64)):
        raise RuntimeError(f"{split}: ordinary/tail flags are not complementary")
    if not np.array_equal(ordinary, (y > -3.0).astype(np.int64)):
        raise RuntimeError(f"{split}: ordinary flag disagrees with y")
    return {"rows": rows, "y": y, "refs": refs, "unseen": unseen, "ordinary": ordinary,
            "csv": str(path.relative_to(C.REPO))}


def paired_bootstrap(ae_a: np.ndarray, ae_b: np.ndarray, *, reps: int = 10000,
                     seed: int = 20261022) -> dict:
    delta = np.asarray(ae_a, dtype=np.float64) - np.asarray(ae_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(delta)
    means = np.empty(reps, dtype=np.float64)
    chunk = max(1, min(reps, 2000))
    done = 0
    while done < reps:
        m = min(chunk, reps - done)
        idx = rng.integers(0, n, size=(m, n))
        means[done:done + m] = delta[idx].mean(axis=1)
        done += m
    return {
        "mean_delta_ae": float(delta.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "fraction_candidate_better": float((delta < 0).mean()),
        "reps": int(reps),
        "seed": int(seed),
        "delta_definition": "AE(P3-BF) - AE(reference); negative = P3-BF better",
    }


def group_tables(pred: np.ndarray, ref: dict) -> tuple[list[dict], dict]:
    y = ref["y"]
    ae = {"P3-BF": np.abs(pred.astype(np.float64) - y)}
    for name, p in ref["refs"].items():
        ae[name] = np.abs(p - y)
    rows = []
    for gname, unseen_flag, ordinary_flag in GROUPS:
        mask = (ref["unseen"] == unseen_flag) & (ref["ordinary"] == ordinary_flag)
        item = {"group": gname, "n": int(mask.sum()),
                "mae": {k: float(v[mask].mean()) for k, v in ae.items()}}
        for name in ("P3B_s0", "L_s0", "DBT_s0"):
            item[f"delta_vs_{name}"] = float(item["mae"]["P3-BF"] - item["mae"][name])
        rows.append(item)
    contrasts = {}
    for name in ("P3B_s0", "L_s0", "DBT_s0"):
        contrasts[name] = paired_bootstrap(ae["P3-BF"], ae[name])
    return rows, {"by_group": rows, "contrasts": contrasts,
                  "overall_mae": {k: float(v.mean()) for k, v in ae.items()},
                  "row_172_kept": True,
                  "groups_analysis_only": True}


def freeze(payload: dict) -> dict:
    path = ANALYSIS / "freeze.json"
    if path.exists():
        old = C.read_json(path)
        if old.get("best_sha256") != payload["best_sha256"]:
            raise RuntimeError("freeze.json already exists for a different checkpoint")
        C.log("[bf-freeze] identical freeze already present")
        return old
    C.write_json(path, payload)
    return payload


def run_test(*, best_sha256: str, norm, info, device, epoch0_valid, test_pred_valid=None):
    freeze_doc = C.read_json(ANALYSIS / "freeze.json")
    if freeze_doc["best_sha256"] != C.sha256_file(RUN / "best.pt"):
        raise RuntimeError("best.pt changed after the freeze")
    cache_info = ensure_test_caches()
    test_cache, y_test, rows_test = load_split("test")
    ref = read_reference("test")
    labels = D.official_labels("test")
    if not np.allclose(labels.astype(np.float64), ref["y"], atol=1e-6):
        raise RuntimeError("official test labels do not match the frozen reference rows")
    ck = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)
    model, _ = M.build_model(norm, seed=SEED, device=device)
    C.load_state(model, ck["state_dict"], device)
    pred_test, mae_test = evaluate(model, test_cache, rows_test, device)
    pred_dir = RUN / "predictions"
    np.savez(pred_dir / "test_best.npz", row=rows_test, pred=pred_test)
    rows, meta = group_tables(pred_test, ref)
    p3b_diff = float(np.abs(pred_test - ref["refs"]["P3B_s0"]).max())
    report = {
        "freeze": freeze_doc,
        "best_sha256": C.sha256_file(RUN / "best.pt"),
        "best_epoch": int(ck["epoch"]),
        "best_valid_mae": float(ck["valid_mae"]),
        "test": {"mae": float(mae_test), "n": int(len(y_test)),
                 "test_label_sha256": C.sha256_array(labels)},
        "test_prediction_path": str((pred_dir / "test_best.npz").relative_to(C.REPO)),
        "test_prediction_sha256": C.sha256_array(pred_test),
        "caches": cache_info,
        "groups": rows,
        "overall": meta,
        "p3b_function_same_epoch0": {
            "best_is_epoch0": bool(int(ck["epoch"]) == 0),
            "max_abs_diff_vs_P3B_reference_test": p3b_diff,
        },
        "historically_exposed_test": True,
        "test_used_for_selection": False,
    }
    C.write_json(ANALYSIS / "test_report.json", report)
    C.write_json(ANALYSIS / "groups_test.json", meta)
    # valid groups use the frozen valid predictions
    d = np.load(RUN / "predictions" / "valid_best.npz")
    pred_valid = np.asarray(d["pred"], dtype=np.float64)
    ref_valid = read_reference("valid")
    rows_v, meta_v = group_tables(pred_valid, ref_valid)
    C.write_json(ANALYSIS / "groups_valid.json", meta_v)
    C.write_json(ANALYSIS / "groups_rows.json", {"valid": rows_v, "test": rows})
    return report


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["run", "test"],
                   help="run = timing + formal + valid/train scoring + freeze + test; "
                        "test = score the frozen best checkpoint")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--budget-seconds", type=float, default=3600.0)
    p.add_argument("--reserve-seconds", type=float, default=180.0)
    p.add_argument("--timing-epochs", type=int, default=4)
    p.add_argument("--max-epochs", type=int, default=600)
    p.add_argument("--epochs", type=int, default=None,
                   help="explicit E (only for recovery; recorded in the plan)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    job_start = time.perf_counter()

    if args.command == "test":
        norm = D.load_norm()
        freeze_doc = C.read_json(ANALYSIS / "freeze.json")
        info = freeze_doc["model_info"]
        report = run_test(best_sha256=freeze_doc["best_sha256"], norm=norm, info=info,
                          device=device, epoch0_valid=freeze_doc["epoch0_valid"])
        C.log(f"[bf-test] test MAE {report['test']['mae']:.6f} (n={report['test']['n']})")
        return 0

    gate_info = gate()
    C.log(f"[bf-gate] {gate_info}")
    if (RUN / "train.json").exists() and not args.force:
        raise RuntimeError(f"{RUN}/train.json exists; use --force to rerun or `test` to score")

    norm = D.load_norm()
    train_cache, y_train, rows_train = load_split("train")
    valid_cache, y_valid, rows_valid = load_split("valid")
    C.log("[bf-run] data loaded")

    # ---- epoch 0 (inherited P3B) is fixed before timing -------------------
    ref_model, info = M.build_model(norm, seed=SEED, device=device)
    _, epoch0_valid = evaluate(ref_model, valid_cache, rows_valid, device)
    C.log(f"[bf-run] epoch0 (P3B inherited) valid={epoch0_valid:.9f}")
    if abs(epoch0_valid - M.P3B_VALID) > 1e-6:
        raise RuntimeError(f"epoch0 valid {epoch0_valid} != P3B {M.P3B_VALID}")
    del ref_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- timing -----------------------------------------------------------
    timing = run_timing(norm, train_cache, y_train, rows_train, valid_cache, rows_valid,
                        device, epochs=args.timing_epochs)
    C.write_json(ANALYSIS / "timing.json", timing)
    used = time.perf_counter() - job_start
    plan = plan_from_timing(timing, used, budget_seconds=args.budget_seconds,
                            reserve_seconds=args.reserve_seconds,
                            max_epochs=args.max_epochs)
    if args.epochs is not None:
        plan["epochs"] = int(args.epochs)
        plan["total_updates"] = int(STEPS_PER_EPOCH * args.epochs)
        plan["feasible"] = bool(args.epochs >= MIN_EPOCHS)
        plan["epochs_override"] = True
    plan["epoch0_valid"] = float(epoch0_valid)
    plan["model_info"] = info
    plan["gate"] = gate_info
    plan["used_gpu_seconds_before_plan"] = float(used)
    plan["created"] = C.utcnow()
    C.write_json(ANALYSIS / "plan.json", plan)
    C.log(f"[bf-plan] t_epoch={timing['steady_epoch_seconds']:.3f}s used={used:.1f}s "
          f"E={plan['epochs']} total_updates={plan['total_updates']} feasible={plan['feasible']}")
    if not plan["feasible"]:
        C.log("[bf-plan] infeasible under the declared budget; stopping before formal training")
        return 2

    if device.type == "cuda":
        torch.cuda.empty_cache()
    # ---- formal run -------------------------------------------------------
    formal = run_formal(epochs=plan["epochs"], norm=norm, train_cache=train_cache,
                        y_train=y_train, rows_train=rows_train, valid_cache=valid_cache,
                        y_valid=y_valid, rows_valid=rows_valid, device=device,
                        epoch0={"valid_mae": float(epoch0_valid)})
    scored = score_checkpoints(info=info, norm=norm, train_cache=train_cache,
                               y_train=y_train, rows_train=rows_train,
                               valid_cache=valid_cache, y_valid=y_valid,
                               rows_valid=rows_valid, device=device,
                               epoch0_valid=float(epoch0_valid))
    total_seconds = time.perf_counter() - job_start
    info_out = {
        "candidate": "P3-BF",
        "seed": SEED,
        "round": "zinc-branching-fragment-v1",
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "data": {
            "source": "official PyG ZINC subset=True train/valid",
            "n_train": int(len(y_train)),
            "n_valid": int(len(y_valid)),
            "official_test_loaded_before_freeze": False,
            "train_label_sha256": C.sha256_array(y_train),
            "valid_label_sha256": C.sha256_array(y_valid),
            "train_cache_sha256": C.sha256_file(B.cache_path("train")),
            "valid_cache_sha256": C.sha256_file(B.cache_path("valid")),
        },
        "model_info": info,
        "parameter_count": info["parameters"],
        "parameter_audit": {
            "parameters_in_optimizer": info["parameters"],
            "parameters_inherited": info["parameters_inherited"],
            "parameters_new": info["parameters_new"],
            "all_parameters_require_grad": True,
        },
        "recipe": {
            "epochs": int(formal["epochs_run"]),
            "planned_epochs": int(plan["epochs"]),
            "steps_per_epoch": STEPS_PER_EPOCH,
            "updates": int(formal["total_updates"]),
            "actual_global_steps": int(formal["global_step"]),
            "warmup_updates": WARMUP,
            "peak_lrs": {"inherited_p3": PEAK_INHERITED, "new": PEAK_NEW},
            "min_lrs": {"inherited_p3": MIN_INHERITED, "new": MIN_NEW},
            "weight_decay": WD, "grad_clip": CLIP, "dropout": 0.0,
            "batch_size": BATCH, "loss": "raw-y L1", "valid_every_epoch": True,
            "early_stop": {"min_epochs": MIN_EPOCHS, "improvement": REF_IMPROVEMENT,
                           "patience": PATIENCE, "triggered": bool(formal["stopped_early"])},
        },
        "epoch0": {"valid_mae": float(formal["epoch0_valid"]),
                   "source": info["source"], "checkpoint_sha256": info["sha256"],
                   "checkpoint_epoch": info["checkpoint_epoch"]},
        "best_valid": formal["best"],
        "scored": {k: v for k, v in scored.items() if k != "valid_pred"},
        "curve": formal["curve"],
        "timing": timing,
        "plan": plan,
        "wall_seconds": float(total_seconds),
        "formal_train_seconds": float(formal["formal_seconds"]),
        "gpu_hours": None if device.type == "cpu" else float(total_seconds) / 3600.0,
        "notes": [
            "epoch 0 is the inherited P3B_s0 best-valid checkpoint and a candidate",
            "best-valid uses strict <, ties keep the earlier epoch",
            "online_train_mae is a within-epoch pre-update running mean",
            "train_mae_eval_mode is the full-10k eval-mode score of the fixed checkpoint",
            "no test data was loaded before the freeze",
        ],
    }
    C.write_json(RUN / "train.json", info_out)
    C.log(f"[bf-run] done best={formal['best']['valid_mae']:.6f}@{formal['best']['epoch']} "
          f"test-pending wall={total_seconds:.1f}s")

    freeze_doc = {
        "protocol": "zinc-branching-fragment-v1",
        "candidate": "P3-BF",
        "seed": SEED,
        "git_commit": C.git_commit(),
        "best_sha256": scored["best_sha256"],
        "best_epoch": int(scored["best"]["epoch"]),
        "best_valid_mae": float(scored["best"]["valid_mae"]),
        "best_valid_mae_stored": float(scored["best"]["valid_mae_stored"]),
        "best_is_epoch0": bool(scored["best_is_epoch0"]),
        "epoch0_valid": float(epoch0_valid),
        "init_checkpoint": {
            "path": info["source"], "sha256": info["sha256"],
            "epoch": int(info["checkpoint_epoch"]),
            "valid_mae": float(info["checkpoint_valid_mae"]),
        },
        "plan_epochs": int(plan["epochs"]),
        "epochs_run": int(formal["epochs_run"]),
        "total_updates": int(formal["total_updates"]),
        "actual_global_steps": int(formal["global_step"]),
        "stopped_early": bool(formal["stopped_early"]),
        "params": int(info["parameters"]),
        "recipe": info_out["recipe"],
        "model_info": info,
        "caches": {
            "train": C.sha256_file(B.cache_path("train")),
            "valid": C.sha256_file(B.cache_path("valid")),
        },
        "data": {"train_label_sha256": C.sha256_array(y_train),
                 "valid_label_sha256": C.sha256_array(y_valid)},
        "test_policy": "one descriptive read after this freeze; no test-driven change",
        "frozen_at": C.utcnow(),
    }
    freeze_doc = freeze(freeze_doc)
    C.log(f"[bf-freeze] best sha256 {freeze_doc['best_sha256']} frozen")
    report = run_test(best_sha256=freeze_doc["best_sha256"], norm=norm, info=info,
                      device=device, epoch0_valid=float(epoch0_valid))
    C.log(f"[bf-test] test MAE {report['test']['mae']:.6f}; "
          f"P3B function diff {report['p3b_function_same_epoch0']['max_abs_diff_vs_P3B_reference_test']:.2e}")
    C.log(f"[bf-run] total GPU job wall {time.perf_counter() - job_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
