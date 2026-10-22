"""Shared runner for the two single-model arms of zinc_structural_object_tokens_v1.

``--arm A``  A_joint_BF       P3-BF graph, SSL-start initialisation, one LR,
                             full fixed 600-epoch recipe B cosine.
``--arm B``  B_object_tokens  atom / rooted-branch / chordless-cycle tokens,
                             4 full-graph attention blocks, one reader.

Both arms: full valid every epoch, strict ``<`` best selection (ties keep the
earlier epoch), best/final/last checkpoints, real curves, frozen best after the
schedule, then one descriptive test read.  No early stopping, no horizon
change after launch, no ensemble, no second predictor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
_BF = _SS / "zinc_branching_fragment_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_BF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import bf_data as BFD  # noqa: E402
import bf_models as BFM  # noqa: E402
import ot_data as O  # noqa: E402
import ot_joint as J  # noqa: E402
import ot_models as OM  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"
RUNS = EXP / "runs"

SEED = 0
BATCH = 128
STEPS = math.ceil(10000 / BATCH)  # 79
WARMUP = 500
CLIP = 5.0
PEAK_LR = 1e-3
MIN_LR = 1e-5
WD_A = 1e-5
WD_B = 1e-4
A_EPOCHS = 600
A_TIME_GUARD_SECONDS = 3600.0
B_BUDGET_SECONDS = 2400.0
B_RESERVE_SECONDS = 120.0
B_MAX_EPOCHS = 600
B_MIN_EPOCHS = 300
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
_REF_DIR = _SS / "zinc_restricted_adaptation_v1" / "analysis"


def arm_cfg(arm: str) -> dict:
    if arm == "A":
        return {
            "arm": "A",
            "candidate": "A_joint_BF",
            "run_dir": RUNS / "A_joint_BF_s0",
            "analysis": ANALYSIS / "A_joint_BF",
            "wd": WD_A,
            "python_module": "bf_data",
            "protocol": "zinc-structural-object-tokens-v1/A_joint_BF",
            "collate": BFD.collate_bf,
            "eval_candidate": "P3BF",
            "eval_batch": 256,
        }
    if arm == "B":
        return {
            "arm": "B",
            "candidate": "B_object_tokens",
            "run_dir": RUNS / "B_object_tokens_s0",
            "analysis": ANALYSIS / "B_object_tokens",
            "wd": WD_B,
            "python_module": "ot_data",
            "protocol": "zinc-structural-object-tokens-v1/B_object_tokens",
            "collate": O.collate_ot,
            "eval_candidate": "OT",
            "eval_batch": 128,
        }
    raise ValueError(f"unknown arm {arm!r}")


# ---------------------------------------------------------------------------
# model builders / data
# ---------------------------------------------------------------------------


def build_arm_model(arm: str, norm, device, *, group_mean=None):
    if arm == "A":
        return J.build_model_a(norm, seed=SEED, device=device)
    return OM.build_model(norm, group_mean=group_mean, device=device, seed=SEED)


def load_split(split: str):
    cache = D.load_cache(split)
    y = np.asarray(cache["labels"], dtype=np.float64)
    return cache, y, np.arange(len(y), dtype=np.int64)


def prep_structure_caches(arm: str) -> dict:
    """Build the label-free structural caches for train/valid if missing."""
    out = {}
    if arm == "A":
        BFD.CACHE.mkdir(parents=True, exist_ok=True)
        for split in ("train", "valid"):
            if not BFD.cache_path(split).exists():
                C.log(f"[prep] building BF {split} cache")
                BFD.save_split(split)
            out[split] = {
                "path": str(BFD.cache_path(split).relative_to(C.REPO)),
                "sha256": C.sha256_file(BFD.cache_path(split)),
            }
    else:
        O.CACHE.mkdir(parents=True, exist_ok=True)
        for split in ("train", "valid"):
            if not O.cache_path(split).exists():
                C.log(f"[prep] building object-token {split} cache")
                O.save_split(split)
            out[split] = {
                "path": str(O.cache_path(split).relative_to(C.REPO)),
                "sha256": C.sha256_file(O.cache_path(split)),
                "meta": O.load_cache(split)["meta"],
            }
    return out


def group_means() -> dict:
    pc = O.load_cache("train")
    return {
        "atom": float(pc["meta"]["mean_atoms_per_molecule"]),
        "branch": float(pc["meta"]["mean_branch_tokens_per_molecule"]),
        "cycle": float(pc["meta"]["mean_cycle_tokens_per_molecule"]),
    }


# ---------------------------------------------------------------------------
# optimisation / loop
# ---------------------------------------------------------------------------


def make_optimizer(model, wd: float):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=PEAK_LR, weight_decay=wd)


def set_lr(optimizer, step: int, total_updates: int) -> float:
    lr = C.cosine_with_warmup_lr(step, total_updates, WARMUP, PEAK_LR, MIN_LR)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def train_epoch(model, optimizer, cache, rows, y, gen, *, epoch, total_updates,
                global_step, collate_fn, device):
    model.train()
    order = torch.randperm(len(rows), generator=gen).numpy()
    total_err, n_mol, lr = 0.0, 0, 0.0
    for start in range(0, len(order), BATCH):
        chunk = order[start:start + BATCH]
        batch = T1._to_device(collate_fn(cache, chunk), device)
        target = torch.as_tensor(y[chunk], dtype=torch.float32, device=device)
        pred = model(batch)
        loss = F.l1_loss(pred.view(-1), target.view(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        if not math.isfinite(float(gn)):
            raise RuntimeError(f"non-finite gradient at epoch {epoch}")
        global_step += 1
        lr = set_lr(optimizer, global_step, total_updates)
        optimizer.step()
        total_err += float((pred.detach().view(-1) - target).abs().sum())
        n_mol += len(chunk)
    return total_err / max(n_mol, 1), global_step, lr


def evaluate(model, cache, rows, device, collate_fn, candidate, batch_size):
    return T1.evaluate(model, cache, rows, candidate=candidate, device=device,
                       batch_size=batch_size, collate_fn=collate_fn)


def epoch_time_seconds(model, optimizer, cache, y, rows, valid_cache, valid_rows,
                       gen, *, total_updates, global_step, collate_fn, device,
                       candidate, eval_batch):
    t0 = time.perf_counter()
    online, global_step, lr = train_epoch(
        model, optimizer, cache, rows, y, gen, epoch=0, total_updates=total_updates,
        global_step=global_step, collate_fn=collate_fn, device=device)
    _, valid_mae = evaluate(model, valid_cache, valid_rows, device, collate_fn,
                            candidate, eval_batch)
    return {"online_train_mae": float(online), "valid_mae": float(valid_mae),
            "seconds": float(time.perf_counter() - t0), "lr": float(lr)}, global_step


# ---------------------------------------------------------------------------
# timing + plan
# ---------------------------------------------------------------------------


def run_timing(cfg, norm, device, *, epochs: int, train_cache, y, rows,
               valid_cache, valid_rows, group_mean) -> dict:
    model, info = build_arm_model(cfg["arm"], norm, device, group_mean=group_mean)
    init_hash = C.state_hash(model)
    opt = make_optimizer(model, cfg["wd"])
    gen = torch.Generator().manual_seed(SEED + 101)
    total = STEPS * (A_EPOCHS if cfg["arm"] == "A" else B_MAX_EPOCHS)
    step = 0
    curve = []
    for ep in range(1, int(epochs) + 1):
        rec, step = epoch_time_seconds(
            model, opt, train_cache, y, rows, valid_cache, valid_rows, gen,
            total_updates=total, global_step=step, collate_fn=cfg["collate"],
            device=device, candidate=cfg["eval_candidate"], eval_batch=cfg["eval_batch"])
        rec["epoch"] = ep
        curve.append(rec)
        C.log(f"[timing {cfg['arm']}] ep={ep} valid={rec['valid_mae']:.6f} "
              f"online={rec['online_train_mae']:.6f} {rec['seconds']:.2f}s")
    steady = float(np.mean([r["seconds"] for r in curve[1:]])) if len(curve) > 1 else \
        float(curve[0]["seconds"])
    return {
        "candidate": cfg["candidate"], "seed": SEED, "timing_epochs": int(epochs),
        "curve": curve, "steady_epoch_seconds": steady,
        "parameters": info["parameters"], "model_info": info,
        "init_state_hash": init_hash,
    }


def plan_a(timing: dict, used_seconds: float) -> dict:
    t_epoch = float(timing["steady_epoch_seconds"])
    projected = t_epoch * A_EPOCHS
    plan = {
        "protocol": "zinc-structural-object-tokens-v1/A_joint_BF",
        "candidate": "A_joint_BF", "seed": SEED,
        "steady_epoch_seconds": t_epoch,
        "used_seconds_before_plan": float(used_seconds),
        "epochs": A_EPOCHS, "steps_per_epoch": STEPS,
        "total_updates": int(STEPS * A_EPOCHS),
        "projected_train_seconds": float(projected),
        "time_guard_seconds": A_TIME_GUARD_SECONDS,
        "feasible": bool(projected <= A_TIME_GUARD_SECONDS),
        "peak_lr": PEAK_LR, "min_lr": MIN_LR, "weight_decay": WD_A,
        "warmup_updates": WARMUP,
        "early_stop": None,
        "note": "full declared 600-epoch cosine; no patience, no horizon change",
        "created": C.utcnow(),
    }
    return plan


def plan_b(timing: dict, used_seconds: float) -> dict:
    t_epoch = float(timing["steady_epoch_seconds"])
    raw = (B_BUDGET_SECONDS - used_seconds - B_RESERVE_SECONDS) / (1.15 * t_epoch)
    e = int(math.floor(raw))
    e = (e // 10) * 10
    e = max(0, min(B_MAX_EPOCHS, e))
    return {
        "protocol": "zinc-structural-object-tokens-v1/B_object_tokens",
        "candidate": "B_object_tokens", "seed": SEED,
        "budget_seconds": B_BUDGET_SECONDS,
        "reserve_seconds": B_RESERVE_SECONDS,
        "used_seconds_before_plan": float(used_seconds),
        "steady_epoch_seconds": t_epoch,
        "speed_margin": 1.15,
        "raw_epoch_estimate": float(raw),
        "epochs": int(e), "steps_per_epoch": STEPS,
        "total_updates": int(STEPS * e),
        "min_learning_epochs": B_MIN_EPOCHS,
        "feasible": bool(e >= B_MIN_EPOCHS),
        "peak_lr": PEAK_LR, "min_lr": MIN_LR, "weight_decay": WD_B,
        "warmup_updates": WARMUP,
        "early_stop": None,
        "note": "E_B fixed by measured epoch time only; no valid-based choice",
        "created": C.utcnow(),
    }


# ---------------------------------------------------------------------------
# formal run
# ---------------------------------------------------------------------------


def run_formal(cfg, norm, device, *, epochs: int, train_cache, y, rows,
               valid_cache, y_valid, valid_rows, group_mean, epoch0: dict | None) -> dict:
    run = cfg["run_dir"]
    run.mkdir(parents=True, exist_ok=True)
    model, info = build_arm_model(cfg["arm"], norm, device, group_mean=group_mean)
    if epoch0.get("init_state_hash") is not None:
        assert C.state_hash(model) == epoch0["init_state_hash"], "formal init != timing init"
    opt = make_optimizer(model, cfg["wd"])
    gen = torch.Generator().manual_seed(SEED + 101)
    steps = math.ceil(len(rows) / BATCH)
    total = steps * int(epochs)
    curve: list[dict] = []
    best = None
    if epoch0["include_as_candidate"]:
        pred0, valid0 = evaluate(model, valid_cache, valid_rows, device, cfg["collate"],
                                 cfg["eval_candidate"], cfg["eval_batch"])
        best = {"valid_mae": float(valid0), "epoch": 0}
        C.atomic_torch_save(
            {"state_dict": C.cpu_state(model), "epoch": 0, "valid_mae": float(valid0),
             "candidate": cfg["candidate"], "seed": SEED,
             "source": "SSL-start P3 (w_s = 0)"},
            run / "epoch0.pt")
        C.atomic_torch_save(
            {"state_dict": C.cpu_state(model), "epoch": 0, "valid_mae": float(valid0),
             "best": dict(best), "global_step": 0, "candidate": cfg["candidate"]},
            run / "best.pt")
        curve.append({"epoch": 0, "online_train_mae": None, "valid_mae": float(valid0),
                      "lr": None, "seconds": None, "steps": 0, "is_best": True})
        C.log(f"[{cfg['arm']}] epoch0 (SSL-start P3) valid={valid0:.9f}")
    step = 0
    started = time.perf_counter()
    for ep in range(1, int(epochs) + 1):
        rec, step = epoch_time_seconds(
            model, opt, train_cache, y, rows, valid_cache, valid_rows, gen,
            total_updates=total, global_step=step, collate_fn=cfg["collate"],
            device=device, candidate=cfg["eval_candidate"], eval_batch=cfg["eval_batch"])
        is_best = best is None or rec["valid_mae"] < best["valid_mae"]
        if is_best:
            best = {"valid_mae": float(rec["valid_mae"]), "epoch": int(ep)}
            C.atomic_torch_save(
                {"state_dict": C.cpu_state(model), "epoch": int(ep),
                 "valid_mae": float(rec["valid_mae"]), "candidate": cfg["candidate"],
                 "seed": SEED, "global_step": int(step)}, run / "best.pt")
        entry = {
            "epoch": int(ep), "online_train_mae": float(rec["online_train_mae"]),
            "valid_mae": float(rec["valid_mae"]), "lr": float(rec["lr"]),
            "seconds": float(rec["seconds"]), "steps": steps, "is_best": bool(is_best),
        }
        curve.append(entry)
        C.atomic_torch_save(
            {"state_dict": C.cpu_state(model), "epoch": int(ep),
             "valid_mae": float(rec["valid_mae"]), "best": dict(best),
             "global_step": int(step), "candidate": cfg["candidate"], "seed": SEED},
            run / "last.pt")
        C.log(f"[{cfg['arm']}] ep={ep:03d} valid={rec['valid_mae']:.6f} "
              f"online={rec['online_train_mae']:.6f} lr={rec['lr']:.2e} "
              f"best={best['valid_mae']:.6f}@{best['epoch']} {rec['seconds']:.2f}s")
        if ep % 5 == 0 or ep == int(epochs):
            C.write_curve_csv(run / "curve.csv", curve)
    formal_seconds = time.perf_counter() - started
    assert step == steps * int(epochs), (step, steps, epochs)
    C.atomic_torch_save(
        {"state_dict": C.cpu_state(model), "optimizer": opt.state_dict(),
         "gen_state": gen.get_state(), "epoch": int(epochs), "global_step": int(step),
         "best": dict(best), "curve": curve, "candidate": cfg["candidate"], "seed": SEED},
        run / "final.pt")
    C.write_curve_csv(run / "curve.csv", curve)
    return {"info": info, "curve": curve, "best": best, "epochs_run": int(epochs),
            "global_step": int(step), "total_updates": int(total),
            "formal_seconds": float(formal_seconds)}


def score_checkpoints(cfg, norm, device, *, train_cache, y, rows, valid_cache,
                      valid_rows, group_mean) -> dict:
    run = cfg["run_dir"]
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    best_model, _ = build_arm_model(cfg["arm"], norm, device, group_mean=group_mean)
    C.load_state(best_model, ck["state_dict"], device)
    pred_valid, mae_valid = evaluate(best_model, valid_cache, valid_rows, device,
                                     cfg["collate"], cfg["eval_candidate"], cfg["eval_batch"])
    _, train_best = evaluate(best_model, train_cache, rows, device, cfg["collate"],
                             cfg["eval_candidate"], cfg["eval_batch"])
    if abs(mae_valid - float(ck["valid_mae"])) > 1e-6:
        raise RuntimeError(f"reloaded best valid {mae_valid} != stored {ck['valid_mae']}")
    pred_dir = run / "predictions"
    pred_dir.mkdir(exist_ok=True)
    np.savez(pred_dir / "valid_best.npz", row=valid_rows, pred=pred_valid)
    write_preds_csv(cfg["analysis"] / "preds_valid.csv", valid_rows, y, pred_valid)
    final = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
    final_model, _ = build_arm_model(cfg["arm"], norm, device, group_mean=group_mean)
    C.load_state(final_model, final["state_dict"], device)
    _, mae_final = evaluate(final_model, valid_cache, valid_rows, device, cfg["collate"],
                            cfg["eval_candidate"], cfg["eval_batch"])
    _, train_final = evaluate(final_model, train_cache, rows, device, cfg["collate"],
                              cfg["eval_candidate"], cfg["eval_batch"])
    return {
        "best": {"epoch": int(ck["epoch"]), "valid_mae": float(mae_valid),
                 "valid_mae_stored": float(ck["valid_mae"]),
                 "train_mae_eval_mode": float(train_best)},
        "final": {"epoch": int(final["epoch"]), "valid_mae": float(mae_final),
                  "train_mae_eval_mode": float(train_final)},
        "best_sha256": C.sha256_file(run / "best.pt"),
        "final_sha256": C.sha256_file(run / "final.pt"),
        "last_sha256": C.sha256_file(run / "last.pt"),
        "best_is_epoch0": bool(int(ck["epoch"]) == 0),
    }


# ---------------------------------------------------------------------------
# freeze / test / groups
# ---------------------------------------------------------------------------


def freeze(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = C.read_json(path)
        if old.get("best_sha256") != payload["best_sha256"]:
            raise RuntimeError(f"{path} already freezes a different checkpoint")
        C.log(f"[freeze] identical freeze already present at {path}")
        return old
    C.write_json(path, payload)
    return payload


def ensure_test_caches(arm: str) -> dict:
    out = {}
    ad_path = D.cache_path("test")
    if not ad_path.exists():
        C.log("[test] building ad_test cache from official PyG (post-freeze)")
        data = D.build_split_v2("test")
        D.save_cache(data, "test")
    out["ad_test"] = {"path": str(ad_path.relative_to(C.REPO)),
                      "sha256": C.sha256_file(ad_path),
                      "size_bytes": int(ad_path.stat().st_size)}
    if arm == "A":
        if not BFD.cache_path("test").exists():
            C.log("[test] building BF test cache (structure only, post-freeze)")
            BFD.save_split("test")
        out["structure"] = {"path": str(BFD.cache_path("test").relative_to(C.REPO)),
                            "sha256": C.sha256_file(BFD.cache_path("test")),
                            "meta": BFD.load_cache("test")["meta"]}
    else:
        if not O.cache_path("test").exists():
            C.log("[test] building object-token test cache (structure only, post-freeze)")
            O.save_split("test")
        out["structure"] = {"path": str(O.cache_path("test").relative_to(C.REPO)),
                            "sha256": C.sha256_file(O.cache_path("test")),
                            "meta": O.load_cache("test")["meta"]}
    return out


def read_reference(split: str) -> dict:
    path = _REF_DIR / f"reference_preds_{split}.csv"
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
    return {"rows": row, "y": y, "refs": refs, "unseen": unseen, "ordinary": ordinary,
            "csv": str(path.relative_to(C.REPO))}


def write_preds_csv(path: Path, rows: np.ndarray, y: np.ndarray, pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "y_raw", "pred"])
        for r, t, p in zip(rows, y, pred):
            w.writerow([int(r), f"{float(t):.9f}", f"{float(p):.9f}"])


def paired_bootstrap(ae_a: np.ndarray, ae_b: np.ndarray, *, reps: int = 10000,
                     seed: int = 20261023) -> dict:
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
        "reps": int(reps), "seed": int(seed),
        "delta_definition": "AE(candidate) - AE(reference); negative = candidate better",
    }


def group_tables(pred: np.ndarray, ref: dict, candidate: str) -> tuple[list[dict], dict]:
    y = ref["y"]
    ae = {candidate: np.abs(np.asarray(pred, dtype=np.float64) - y)}
    for name, p in ref["refs"].items():
        ae[name] = np.abs(p - y)
    rows = []
    for gname, unseen_flag, ordinary_flag in GROUPS:
        mask = (ref["unseen"] == unseen_flag) & (ref["ordinary"] == ordinary_flag)
        item = {"group": gname, "n": int(mask.sum()),
                "mae": {k: float(v[mask].mean()) for k, v in ae.items()}}
        for name in ("P3B_s0", "L_s0", "DBT_s0"):
            item[f"delta_vs_{name}"] = float(item["mae"][candidate] - item["mae"][name])
        rows.append(item)
    contrasts = {name: paired_bootstrap(ae[candidate], ae[name])
                 for name in ("P3B_s0", "L_s0", "DBT_s0")}
    return rows, {"by_group": rows, "contrasts": contrasts,
                  "overall_mae": {k: float(v.mean()) for k, v in ae.items()},
                  "groups_analysis_only": True}


def run_test(cfg, norm, device, *, group_mean, freeze_doc) -> dict:
    run = cfg["run_dir"]
    ana = cfg["analysis"]
    if freeze_doc["best_sha256"] != C.sha256_file(run / "best.pt"):
        raise RuntimeError("best.pt changed after the freeze")
    cache_info = ensure_test_caches(cfg["arm"])
    test_cache, y_test, rows_test = load_split("test")
    ref = read_reference("test")
    labels = D.official_labels("test")
    if not np.allclose(labels.astype(np.float64), ref["y"], atol=1e-6):
        raise RuntimeError("official test labels do not match the frozen reference rows")
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model, _ = build_arm_model(cfg["arm"], norm, device, group_mean=group_mean)
    C.load_state(model, ck["state_dict"], device)
    pred_test, mae_test = evaluate(model, test_cache, rows_test, device, cfg["collate"],
                                   cfg["eval_candidate"], cfg["eval_batch"])
    pred_dir = run / "predictions"
    pred_dir.mkdir(exist_ok=True)
    np.savez(pred_dir / "test_best.npz", row=rows_test, pred=pred_test)
    write_preds_csv(ana / "preds_test.csv", rows_test, y_test, pred_test)
    rows, meta = group_tables(pred_test, ref, cfg["candidate"])
    d = np.load(pred_dir / "valid_best.npz")
    pred_valid = np.asarray(d["pred"], dtype=np.float64)
    ref_valid = read_reference("valid")
    rows_v, meta_v = group_tables(pred_valid, ref_valid, cfg["candidate"])
    report = {
        "arm": cfg["arm"], "candidate": cfg["candidate"],
        "freeze": freeze_doc,
        "best_sha256": C.sha256_file(run / "best.pt"),
        "best_epoch": int(ck["epoch"]),
        "best_valid_mae": float(ck["valid_mae"]),
        "test": {"mae": float(mae_test), "n": int(len(y_test)),
                 "test_label_sha256": C.sha256_array(labels)},
        "test_prediction_path": str((pred_dir / "test_best.npz").relative_to(C.REPO)),
        "test_prediction_sha256": C.sha256_array(pred_test),
        "caches": cache_info,
        "groups": rows,
        "overall": meta,
        "historically_exposed_test": True,
        "test_used_for_selection": False,
    }
    C.write_json(ana / "test_report.json", report)
    C.write_json(ana / "groups_test.json", meta)
    C.write_json(ana / "groups_valid.json", meta_v)
    C.write_json(ana / "groups_rows.json", {"valid": rows_v, "test": rows})
    return report


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def gate() -> dict:
    path = ANALYSIS / "cpu_checks.json"
    if not path.exists():
        raise RuntimeError("missing analysis/cpu_checks.json; run ot_checks.py first")
    doc = C.read_json(path)
    failed = [k for k, v in doc["checks"].items() if not v.get("pass")]
    if failed:
        raise RuntimeError(f"CPU checks failed: {failed}")
    return {"checks_passed": doc["n_pass"], "checks_failed": doc["n_fail"]}


def command_run(cfg, args) -> int:
    ana, run = cfg["analysis"], cfg["run_dir"]
    ana.mkdir(parents=True, exist_ok=True)
    gate_info = gate()
    C.log(f"[gate] {gate_info}")
    if (run / "train.json").exists() and not args.force:
        raise RuntimeError(f"{run}/train.json exists; use --force or `test`")
    job_start = time.perf_counter()
    structure = prep_structure_caches(cfg["arm"])
    group_mean = group_means() if cfg["arm"] == "B" else None
    norm = D.load_norm()
    train_cache, y, rows = load_split("train")
    valid_cache, y_valid, valid_rows = load_split("valid")
    device = torch.device(args.device)
    C.log(f"[{cfg['arm']}] data ready ({time.perf_counter() - job_start:.1f}s) on {device}")
    timing = run_timing(cfg, norm, device, epochs=args.timing_epochs, train_cache=train_cache,
                        y=y, rows=rows, valid_cache=valid_cache, valid_rows=valid_rows,
                        group_mean=group_mean)
    C.write_json(ana / "timing.json", timing)
    used = time.perf_counter() - job_start
    plan = plan_a(timing, used) if cfg["arm"] == "A" else plan_b(timing, used)
    if args.epochs is not None:
        plan["epochs"] = int(args.epochs)
        plan["total_updates"] = int(STEPS * plan["epochs"])
        plan["epochs_override"] = True
        plan["feasible"] = bool(plan["epochs"] >= (A_EPOCHS if cfg["arm"] == "A" else B_MIN_EPOCHS))
    plan["gate"] = gate_info
    plan["structure_caches"] = structure
    plan["group_mean"] = group_mean
    plan["model_info"] = timing["model_info"]
    plan["init_state_hash"] = timing["init_state_hash"]
    C.write_json(ana / "plan.json", plan)
    C.log(f"[{cfg['arm']}-plan] t_epoch={timing['steady_epoch_seconds']:.3f}s used={used:.1f}s "
          f"E={plan['epochs']} (>=79*E={plan['total_updates']}) feasible={plan['feasible']}")
    if not plan["feasible"]:
        C.write_json(ana / "status.json", {
            "status": "INCOMPLETE", "reason": "planned horizon infeasible under the declared budget",
            "plan": plan, "created": C.utcnow()})
        C.log(f"[{cfg['arm']}] infeasible plan; stopping before formal training")
        return 2
    for split, items in plan["structure_caches"].items():
        if cfg["arm"] == "B" and split == "train":
            assert abs(items["meta"]["mean_branch_tokens_per_molecule"] - group_mean["branch"]) < 1e-12
    include_epoch0 = cfg["arm"] == "A"
    formal = run_formal(cfg, norm, device, epochs=plan["epochs"], train_cache=train_cache,
                        y=y, rows=rows, valid_cache=valid_cache, y_valid=y_valid,
                        valid_rows=valid_rows, group_mean=group_mean,
                        epoch0={"include_as_candidate": include_epoch0,
                                "init_state_hash": timing["init_state_hash"]})
    scored = score_checkpoints(cfg, norm, device, train_cache=train_cache, y=y, rows=rows,
                               valid_cache=valid_cache, valid_rows=valid_rows,
                               group_mean=group_mean)
    total_seconds = time.perf_counter() - job_start
    info = {
        "arm": cfg["arm"], "candidate": cfg["candidate"], "seed": SEED,
        "protocol": cfg["protocol"],
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "data": {
            "source": "official PyG ZINC subset=True train/valid",
            "n_train": int(len(y)), "n_valid": int(len(y_valid)),
            "official_test_loaded_before_freeze": False,
            "train_label_sha256": C.sha256_array(y),
            "valid_label_sha256": C.sha256_array(y_valid),
        },
        "structure_caches": structure,
        "model_info": formal["info"],
        "parameter_count": formal["info"]["parameters"],
        "group_mean": group_mean,
        "recipe": {
            "epochs": int(formal["epochs_run"]), "planned_epochs": int(plan["epochs"]),
            "steps_per_epoch": int(STEPS), "updates": int(formal["total_updates"]),
            "actual_global_steps": int(formal["global_step"]),
            "warmup_updates": WARMUP, "peak_lr": PEAK_LR, "min_lr": MIN_LR,
            "weight_decay": cfg["wd"], "grad_clip": CLIP, "batch_size": BATCH,
            "loss": "raw-y L1", "valid_every_epoch": True,
            "early_stop": None,
            "dropout": ("encoders/content MLPs 0.0; transformer/reader 0.05"
                        if cfg["arm"] == "B" else "0.0"),
        },
        "best_valid": formal["best"],
        "scored": scored,
        "curve": formal["curve"],
        "timing": timing,
        "plan": plan,
        "wall_seconds": float(total_seconds),
        "formal_train_seconds": float(formal["formal_seconds"]),
        "gpu_hours": None if device.type == "cpu" else float(total_seconds) / 3600.0,
        "notes": [
            "best-valid uses strict <, ties keep the earlier epoch",
            "online_train_mae is a within-epoch pre-update running mean",
            "train_mae_eval_mode is the full-10k eval-mode score of the fixed checkpoint",
            "no test data was loaded before the freeze",
        ],
    }
    C.write_json(run / "train.json", info)
    if (run / "curve.csv").exists():
        (ana / "curve.csv").write_text((run / "curve.csv").read_text(encoding="utf-8"),
                                       encoding="utf-8")
    C.write_json(ana / "status.json", {"status": "TRAINED", "wall_seconds": total_seconds,
                                       "created": C.utcnow()})
    C.log(f"[{cfg['arm']}] trained best={formal['best']['valid_mae']:.6f}@"
          f"{formal['best']['epoch']} wall={total_seconds:.1f}s")

    freeze_doc = {
        "protocol": cfg["protocol"], "arm": cfg["arm"], "candidate": cfg["candidate"],
        "seed": SEED, "git_commit": C.git_commit(),
        "best_sha256": scored["best_sha256"], "best_epoch": int(scored["best"]["epoch"]),
        "best_valid_mae": float(scored["best"]["valid_mae"]),
        "best_valid_mae_stored": float(scored["best"]["valid_mae_stored"]),
        "best_is_epoch0": bool(scored["best_is_epoch0"]),
        "init": formal["info"].get("ssl_encoders") or formal["info"].get("ssl_atom_encoder"),
        "plan_epochs": int(plan["epochs"]), "epochs_run": int(formal["epochs_run"]),
        "total_updates": int(formal["total_updates"]),
        "actual_global_steps": int(formal["global_step"]),
        "params": int(formal["info"]["parameters"]),
        "recipe": info["recipe"],
        "model_info": formal["info"],
        "caches": {k: v.get("sha256") for k, v in plan["structure_caches"].items()},
        "data": {"train_label_sha256": C.sha256_array(y),
                 "valid_label_sha256": C.sha256_array(y_valid)},
        "test_policy": "one descriptive read after this freeze; no test-driven change",
        "frozen_at": C.utcnow(),
    }
    freeze_doc = freeze(ana / "freeze.json", freeze_doc)
    C.log(f"[freeze] {cfg['arm']} best sha256 {freeze_doc['best_sha256']}")
    report = run_test(cfg, norm, device, group_mean=group_mean, freeze_doc=freeze_doc)
    C.log(f"[test] {cfg['arm']} test MAE {report['test']['mae']:.6f}")
    C.write_json(ana / "status.json", {"status": "COMPLETE",
                                       "wall_seconds": float(time.perf_counter() - job_start),
                                       "created": C.utcnow()})
    return 0


def command_test(cfg, args) -> int:
    ana = cfg["analysis"]
    freeze_doc = C.read_json(ana / "freeze.json")
    group_mean = group_means() if cfg["arm"] == "B" else None
    norm = D.load_norm()
    device = torch.device(args.device)
    report = run_test(cfg, norm, device, group_mean=group_mean, freeze_doc=freeze_doc)
    C.log(f"[test] {cfg['arm']} test MAE {report['test']['mae']:.6f}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["run", "test"])
    p.add_argument("--arm", required=True, choices=["A", "B"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--timing-epochs", type=int, default=4)
    p.add_argument("--epochs", type=int, default=None,
                   help="explicit horizon (recovery only; recorded in the plan)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    cfg = arm_cfg(args.arm)
    if args.command == "test":
        return command_test(cfg, args)
    return command_run(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
