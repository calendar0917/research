"""ZINC topology cross-fit diagnostic v1 — fixed residual readout over two Full bases.

Question (pre-registered, single hypothesis): keeping Full's local environment,
dictionaries and relation channels unchanged, does the existing topology25
channel provide a *generalisable* error-correction signal beyond the base
prediction itself?

Design
------
* Outer split: the frozen 8000 fit / 2000 dev molecule-level split of
  ``zinc_joint_dictionary_decision_v1`` (identities and hashes preserved).
* Two ~4000-row sub-folds A/B inside the 8000 fit, seed ``20261003``, balanced
  by raw cycle-penalty stratum over canonical groups; every row kept.
* Two identical Full models (408,651 params): ``F_A`` trains on A only, ``F_B``
  on B only.  All data-dependent prep (standardisers, sdb32 dictionary, common
  subspace, X175 normalisation) is fit on the base training fold only.
* Per-base calibration: median train-fold residual folded into the reader output
  bias exactly once; unfolded raw and single-calibrated predictions both saved.
* Fixed CPU readout (ExtraTreesRegressor, 256 trees, leaf=1, max_features=1.0,
  random_state=0): ``P = [p_base_cal]`` and ``TP = [raw_topology25, p_base_cal]``
  fit on the *other* sub-fold's residual ``r = y - p_base_cal`` and evaluated on
  the outer dev.
* Main comparison on outer dev: equal-weight average of the two base models
  before MAE.  ``gain_base = MAE(base) - MAE(TP)``, ``gain_topo = MAE(P) - MAE(TP)``.

Official test is never instantiated, loaded or evaluated.  No AMP/DDP, FP32.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztf

PROTOCOL_VERSION = "zinc-topology-crossfit-diagnostic-v1"
TRACK_ROOT = zjd.TRACK_ROOT
RESULTS_DIR = TRACK_ROOT / "results/zinc_topology_crossfit_diagnostic_v1"
PREP_DIR = RESULTS_DIR / "prep"
SUBFOLD_SEED = 20261003
TREE_SEED = 0
N_TREES = 256
FOLDS = ("A", "B")
SEVERE_MAX = -2


# ---------------------------------------------------------------------------
# split helpers
# ---------------------------------------------------------------------------


def _sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def outer_split() -> dict[str, Any]:
    split = zjd.build_fold_split()
    saved = json.loads((zjd.PREP_DIR / "split.json").read_text())
    assert _sha_arr(split["fit_idx"]) == saved["fit_idx_sha256"], "outer fit_idx changed"
    assert _sha_arr(split["dev_idx"]) == saved["dev_idx_sha256"], "outer dev_idx changed"
    split["meta"]["reused_from"] = "zinc_joint_dictionary_decision_v1/prep/split.json"
    return split


def build_subfold_split(fit_idx: np.ndarray, pen: np.ndarray, gid: np.ndarray) -> dict[str, Any]:
    fit_set = set(int(i) for i in fit_idx.tolist())
    by_group: dict[int, list[int]] = {}
    for i in fit_set:
        by_group.setdefault(int(gid[i]), []).append(i)
    inconsistent = []
    strata: dict[int, list[tuple[int, list[int], int]]] = {0: [], 1: [], 2: []}
    for g, rows in by_group.items():
        vals = sorted(set(int(pen[r]) for r in rows))
        if len(vals) > 1:
            inconsistent.append({"group": int(g), "penalties": vals})
        p = vals[0]
        s = 0 if p == 0 else (1 if p == -1 else 2)
        strata[s].append((g, rows, p))

    assign: dict[int, str] = {}
    for s in (0, 1, 2):
        groups = sorted(strata[s], key=lambda t: int.from_bytes(
            hashlib.sha256(f"{SUBFOLD_SEED}:{t[0]}".encode()).digest()[:8], "big"))
        a_rows = b_rows = 0
        for g, rows, _p in groups:
            if a_rows <= b_rows:
                assign[g] = "A"; a_rows += len(rows)
            else:
                assign[g] = "B"; b_rows += len(rows)

    a_idx = np.array(sorted(i for g, rows in by_group.items() if assign[g] == "A" for i in rows), np.int64)
    b_idx = np.array(sorted(i for g, rows in by_group.items() if assign[g] == "B" for i in rows), np.int64)
    assert len(a_idx) + len(b_idx) == len(fit_idx)
    assert not (set(a_idx.tolist()) & set(b_idx.tolist()))
    assert not (set(a_idx.tolist()) | set(b_idx.tolist())) - fit_set

    def stats(idx: np.ndarray) -> dict[str, Any]:
        p = pen[idx]
        return {"n_rows": int(len(idx)), "n_groups": len({int(gid[i]) for i in idx.tolist()}),
                "penalty_0": int((p == 0).sum()), "penalty_-1": int((p == -1).sum()),
                "penalty_le_-2": int((p <= -2).sum())}

    meta = {
        "subfold_seed": SUBFOLD_SEED,
        "unit": "canonical_group_id",
        "n_fit_rows": int(len(fit_idx)),
        "A": stats(a_idx), "B": stats(b_idx),
        "inconsistent_group_penalties": inconsistent,
        "a_idx_sha256": _sha_arr(a_idx), "b_idx_sha256": _sha_arr(b_idx),
        "fit_idx_sha256": _sha_arr(fit_idx),
        "official_test_loaded": False,
    }
    return {"A": a_idx, "B": b_idx, "meta": meta, "assign": assign}


# ---------------------------------------------------------------------------
# sub-fold prep (data-dependent objects fit on the base training fold only)
# ---------------------------------------------------------------------------


def build_subfold_blob(sub_idx: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    train_data = p1run.load_split("train")
    valid_data = p1run.load_split("valid")
    prep = zjd.build_fold_inputs(train_data, valid_data, sub_idx)
    return prep, train_data


def save_subfold_blob(prep: Mapping[str, Any], sub_idx: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        fit_idx=np.asarray(sub_idx, np.int64),
        patch_all_mean=prep["patch_all"].mean, patch_all_scale=prep["patch_all"].scale,
        ctx_all_mean=prep["ctx_all"].mean, ctx_all_scale=prep["ctx_all"].scale,
        anchor_all_mean=prep["anchor_all"].mean, anchor_all_scale=prep["anchor_all"].scale,
        topo_all_mean=prep["topo_all"].mean, topo_all_scale=prep["topo_all"].scale,
        patch_fit_mean=prep["patch_fit"].mean, patch_fit_scale=prep["patch_fit"].scale,
        ctx_fit_mean=prep["ctx_fit"].mean, ctx_fit_scale=prep["ctx_fit"].scale,
        anchor_fit_mean=prep["anchor_fit"].mean, anchor_fit_scale=prep["anchor_fit"].scale,
        topo_fit_mean=prep["topo_fit"].mean, topo_fit_scale=prep["topo_fit"].scale,
        D_fit=prep["D_fit"], U_components=prep["subspace_fit"].components, U_rms=prep["subspace_fit"].rms,
        x_mean=prep["x_mean"], x_std=prep["x_std"], block_scale=prep["block_scale"],
    )
    return _sha_arr(np.load(path)["D_fit"])


def load_blob(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# training one Full base
# ---------------------------------------------------------------------------


def train_base(fold: str, sub_idx: np.ndarray, meta_idx: np.ndarray, dev_idx: np.ndarray,
               blob_path: Path, *, epochs: int, seed: int, out_dir: Path, device: torch.device,
               log=print) -> dict[str, Any]:
    blob = load_blob(blob_path)
    assert _sha_arr(np.asarray(sub_idx, np.int64)) == _sha_arr(blob["fit_idx"]), "blob fold mismatch"
    train_data = p1run.load_split("train")
    valid_data = p1run.load_split("valid")
    zjd.apply_prep_blob(train_data, valid_data, blob)
    train_set = set(int(i) for i in sub_idx.tolist())
    meta_set = set(int(i) for i in meta_idx.tolist())
    dev_set = set(int(i) for i in dev_idx.tolist())
    fit_data = [d for i, d in enumerate(train_data) if i in train_set]
    meta_data = [d for i, d in enumerate(train_data) if i in meta_set]
    dev_data = [d for i, d in enumerate(train_data) if i in dev_set]

    zjd.seed_everything(seed)
    model = zjd.make_arm_model("F", blob, seed=seed).to(device)
    init_hash = zjd.parameter_state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=zjd.LR, weight_decay=zjd.WEIGHT_DECAY)
    train_gen = torch.Generator().manual_seed(seed + zjd.TRAIN_SHUFFLE_OFFSET)
    train_loader = torch.utils.data.DataLoader(list(fit_data), batch_size=zjd.BATCH_SIZE, shuffle=True,
                                               generator=train_gen, num_workers=0, collate_fn=p1.env_collate)
    eval_gen = torch.Generator().manual_seed(seed + zjd.EVAL_SHUFFLE_OFFSET)

    soup: dict[int, dict[str, torch.Tensor]] = {}
    soup_epochs = set(range(max(1, int(epochs) - 4), int(epochs) + 1))
    curve, epoch_seconds = [], []
    peak = 0.0
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        es = time.perf_counter()
        model.train()
        task_sum, rec_sum, n_mol, n_steps, gsum = 0.0, 0.0, 0, 0, 0.0
        for batch in train_loader:
            batch = batch.to(device)
            prediction, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
            loss = torch.nn.functional.l1_loss(prediction.view(-1), batch.y.view(-1)) \
                + float(cm.H1_LAMBDA) * model.reconstruction_loss(aux["phi"], aux["coord"])
            optimizer.zero_grad()
            loss.backward()
            gsum += float(torch.nn.utils.clip_grad_norm_(model.parameters(), zjd.GRAD_CLIP))
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            rec_sum += float(model.reconstruction_loss(aux["phi"], aux["coord"]).detach())
            n_mol += int(batch.y.numel()); n_steps += 1
        epoch_seconds.append(float(time.perf_counter() - es))
        curve.append({"epoch": int(epoch), "train_mae": float(task_sum / max(n_mol, 1)),
                      "rec": float(rec_sum / max(n_steps, 1)), "grad_norm": float(gsum / max(n_steps, 1)),
                      "seconds": epoch_seconds[-1]})
        if epoch in soup_epochs:
            soup[epoch] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if device.type == "cuda":
            peak = max(peak, float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)))
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == int(epochs)):
            log(f"[F{fold}] ep={epoch:03d} train={curve[-1]['train_mae']:.6f} "
                f"rec={curve[-1]['rec']:.4g} gnorm={curve[-1]['grad_norm']:.3g} {epoch_seconds[-1]:.1f}s")

    members = sorted(soup)
    soup_state = {k: torch.stack([soup[e][k].float() for e in members]).mean(0) for k in soup[members[0]]}
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    fresh = zjd.make_arm_model("F", blob, seed=seed).to(device)
    fresh.load_state_dict(soup_state)
    fresh.eval()

    # calibration: median train-fold residual, folded into the output bias ONCE
    fit_cal_raw, fit_y = zjd.collect_predictions(fresh, fit_data, device)
    delta = float(np.median(fit_y - fit_cal_raw))
    before = float(fresh.reader.net[4].bias.detach().cpu().item())
    with torch.no_grad():
        fresh.reader.net[4].bias.add_(delta)
    after = float(fresh.reader.net[4].bias.detach().cpu().item())

    def predict(data):
        cal, y = zjd.collect_predictions(fresh, data, device)
        return cal - delta, cal, y  # unfolded raw, single-calibrated, target

    fit_raw, fit_cal, fit_y2 = predict(fit_data)
    meta_raw, meta_cal, meta_y = predict(meta_data)
    dev_raw, dev_cal, dev_y = predict(dev_data)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"soup_state": soup_state, "last_state": last_state,
                "soup_members": members, "delta": delta, "fold": fold, "seed": seed},
               out_dir / f"F_{fold}_state.pt")
    result = {
        "arm": "F", "fold": fold, "seed": int(seed), "epochs": int(epochs),
        "lr": zjd.LR, "weight_decay": zjd.WEIGHT_DECAY, "lambda_rec_structural": float(cm.H1_LAMBDA),
        "soup_members": members, "n_train_rows": int(len(fit_data)),
        "n_meta_rows": int(len(meta_data)), "n_dev_rows": int(len(dev_data)),
        "init_state_sha256": init_hash,
        "soup_state_sha256": zjd.parameter_state_hash(fresh),
        "soup_state_file_sha256": hashlib.sha256((out_dir / f"F_{fold}_state.pt").read_bytes()).hexdigest(),
        "calibration": {"delta": delta, "reader_bias_before": before, "reader_bias_after": after},
        "fit_raw_mae": float(np.abs(fit_raw - fit_y2).mean()),
        "fit_cal_mae": float(np.abs(fit_cal - fit_y2).mean()),
        "meta_raw_mae": float(np.abs(meta_raw - meta_y).mean()),
        "meta_cal_mae": float(np.abs(meta_cal - meta_y).mean()),
        "dev_raw_mae": float(np.abs(dev_raw - dev_y).mean()),
        "dev_cal_mae": float(np.abs(dev_cal - dev_y).mean()),
        "meta_predictions_raw": meta_raw.tolist(), "meta_predictions_cal": meta_cal.tolist(),
        "meta_targets": meta_y.tolist(),
        "dev_predictions_raw": dev_raw.tolist(), "dev_predictions_cal": dev_cal.tolist(),
        "dev_targets": dev_y.tolist(),
        "curve": curve, "epoch_seconds": epoch_seconds,
        "wall_clock_s": float(time.perf_counter() - started),
        "seconds_per_epoch": float(np.mean(epoch_seconds)),
        "peak_gpu_memory_mb": peak, "device": str(device),
        "official_test_loaded": False,
    }
    (out_dir / f"F_{fold}.json").write_text(json.dumps(result, indent=2))
    log(f"[F{fold}] DONE dev_raw={result['dev_raw_mae']:.6f} dev_cal={result['dev_cal_mae']:.6f} "
        f"wall={result['wall_clock_s']:.0f}s")
    return result


# ---------------------------------------------------------------------------
# fixed CPU residual readout
# ---------------------------------------------------------------------------


def raw_topology25() -> np.ndarray:
    train_data = p1run.load_split("train")
    mat, _frame, meta = ztf.matrices_for_split("train", train_data, "hinge")
    assert mat.shape == (10000, 25), mat.shape
    assert meta["mode"] == "hinge" and meta["input_width"] == 25
    return np.asarray(mat, np.float32)


def fit_readout(base: Mapping[str, Any], t25: np.ndarray, meta_idx: np.ndarray, dev_idx: np.ndarray,
                *, log=print) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesRegressor

    meta_cal = np.asarray(base["meta_predictions_cal"], np.float64)
    meta_y = np.asarray(base["meta_targets"], np.float64)
    dev_cal = np.asarray(base["dev_predictions_cal"], np.float64)
    dev_y = np.asarray(base["dev_targets"], np.float64)
    r_meta = meta_y - meta_cal
    T_meta = t25[meta_idx]
    T_dev = t25[dev_idx]

    q_P = ExtraTreesRegressor(n_estimators=N_TREES, min_samples_leaf=1, max_features=1.0,
                              random_state=TREE_SEED, n_jobs=8).fit(meta_cal.reshape(-1, 1), r_meta)
    q_TP = ExtraTreesRegressor(n_estimators=N_TREES, min_samples_leaf=1, max_features=1.0,
                               random_state=TREE_SEED, n_jobs=8).fit(
        np.concatenate([T_meta, meta_cal.reshape(-1, 1)], 1), r_meta)
    p_P = dev_cal + q_P.predict(dev_cal.reshape(-1, 1))
    p_TP = dev_cal + q_TP.predict(np.concatenate([T_dev, dev_cal.reshape(-1, 1)], 1))
    const = float(np.median(r_meta))
    p_const = dev_cal + const
    return {"fold": base["fold"], "r_meta_median": const,
            "meta_mae_base": float(np.abs(meta_cal - meta_y).mean()),
            "dev_mae_base": float(np.abs(dev_cal - dev_y).mean()),
            "dev_mae_P": float(np.abs(p_P - dev_y).mean()),
            "dev_mae_TP": float(np.abs(p_TP - dev_y).mean()),
            "dev_mae_const": float(np.abs(p_const - dev_y).mean()),
            "dev_predictions_base": dev_cal.tolist(), "dev_predictions_P": p_P.tolist(),
            "dev_predictions_TP": p_TP.tolist(), "dev_predictions_const": p_const.tolist(),
            "dev_targets": dev_y.tolist(),
            "dev_gain_base_vs_TP": float(np.abs(dev_cal - dev_y).mean() - np.abs(p_TP - dev_y).mean()),
            "dev_gain_P_vs_TP": float(np.abs(p_P - dev_y).mean() - np.abs(p_TP - dev_y).mean()),
            "dev_gain_base_vs_P": float(np.abs(dev_cal - dev_y).mean() - np.abs(p_P - dev_y).mean()),
            "dev_gain_base_vs_const": float(np.abs(dev_cal - dev_y).mean() - np.abs(p_const - dev_y).mean()),
            "sklearn_version": __import__("sklearn").__version__,
            "params": {"n_estimators": N_TREES, "min_samples_leaf": 1, "max_features": 1.0,
                       "random_state": TREE_SEED, "n_jobs": 8}}


# ---------------------------------------------------------------------------
# coverage diagnostics (cheap, no new fits)
# ---------------------------------------------------------------------------


def coverage(t25: np.ndarray, meta_idx: np.ndarray, dev_idx: np.ndarray, pen: np.ndarray) -> dict[str, Any]:
    T_meta = t25[meta_idx]
    T_dev = t25[dev_idx]
    # exact-class key = rounded raw integer-valued topology statistics
    key_meta = [tuple(np.round(r, 6).tolist()) for r in T_meta]
    meta_classes: dict[tuple, list[int]] = {}
    for k, pos in zip(key_meta, range(len(key_meta))):
        meta_classes.setdefault(k, []).append(pos)
    exact = np.array([1.0 if tuple(np.round(r, 6).tolist()) in meta_classes else 0.0 for r in T_dev])
    # nearest meta neighbour under meta-fit column std (zero-variance ignored)
    std = T_meta.std(0)
    keep = std > 1e-12
    A = (T_meta[:, keep] - T_meta[:, keep].mean(0)) / std[keep]
    B = (T_dev[:, keep] - T_meta[:, keep].mean(0)) / std[keep]
    d2 = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    nn = d2.argmin(1)
    return {"n_dev": int(len(dev_idx)), "exact_class_coverage": float(exact.mean()),
            "exact_hits": int(exact.sum()),
            "dev_severe_exact": {int(i): float(exact[j]) for j, i in enumerate(dev_idx.tolist())
                                 if pen[i] <= SEVERE_MAX},
            "nn_dist_mean": float(np.sqrt(d2[np.arange(len(nn)), nn]).mean()),
            "nn_penalty_agreement": float(np.mean([pen[meta_idx[nn[j]]] == pen[dev_idx[j]]
                                                   for j in range(len(nn))]))}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def mode_prep() -> int:
    pen, _ = zjd._load_penalties()
    handoff = np.load(zjd.HANDOFF / "train.npz", allow_pickle=True)
    gid = handoff["canonical_group_id"].astype(np.int64)
    outer = outer_split()
    sub = build_subfold_split(outer["fit_idx"], pen, gid)
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    (PREP_DIR / "subfold_split.json").write_text(json.dumps(sub["meta"], indent=2))
    t0 = time.perf_counter()
    for fold in FOLDS:
        idx = sub[fold]
        prep, _ = build_subfold_blob(idx)
        path = PREP_DIR / f"fold_{fold}.npz"
        dhash = save_subfold_blob(prep, idx, path)
        print(f"[prep/{fold}] rows={len(idx)} D_fit={dhash[:12]} sec={prep['seconds']:.1f}")
    (PREP_DIR / "prep_meta.json").write_text(json.dumps({
        "subfold": sub["meta"], "fold_A_blob_sha256": hashlib.sha256((PREP_DIR / "fold_A.npz").read_bytes()).hexdigest(),
        "fold_B_blob_sha256": hashlib.sha256((PREP_DIR / "fold_B.npz").read_bytes()).hexdigest(),
        "prep_seconds": float(time.perf_counter() - t0), "official_test_loaded": False}, indent=2))
    return 0


def mode_train(fold: str, epochs: int, device: str) -> int:
    pen, _ = zjd._load_penalties()
    handoff = np.load(zjd.HANDOFF / "train.npz", allow_pickle=True)
    gid = handoff["canonical_group_id"].astype(np.int64)
    outer = outer_split()
    sub = build_subfold_split(outer["fit_idx"], pen, gid)
    sub_idx = sub[fold]
    meta_idx = sub["B" if fold == "A" else "A"]
    result = train_base(fold, sub_idx, meta_idx, outer["dev_idx"], PREP_DIR / f"fold_{fold}.npz",
                        epochs=epochs, seed=zjd.SEED, out_dir=RESULTS_DIR, device=zjd.resolve_device(device))
    print(json.dumps({k: result[k] for k in ("fold", "dev_raw_mae", "dev_cal_mae", "wall_clock_s",
                                              "soup_state_sha256")}, indent=2))
    return 0


def mode_readout() -> int:
    pen, _ = zjd._load_penalties()
    handoff = np.load(zjd.HANDOFF / "train.npz", allow_pickle=True)
    gid = handoff["canonical_group_id"].astype(np.int64)
    outer = outer_split()
    sub = build_subfold_split(outer["fit_idx"], pen, gid)
    t25 = raw_topology25()
    bases = {f: json.loads((RESULTS_DIR / f"F_{f}.json").read_text()) for f in FOLDS}
    readouts = {f: fit_readout(bases[f], t25, sub["B" if f == "A" else "A"], outer["dev_idx"]) for f in FOLDS}
    (RESULTS_DIR / "readouts.json").write_text(json.dumps(readouts, indent=2))
    (RESULTS_DIR / "coverage.json").write_text(json.dumps(
        {"t25_cache": str(ztf.CACHE_ROOT / "train_topology_features.csv"),
         "coverage": coverage(t25, sub["A"], outer["dev_idx"], pen)}, indent=2))
    for f in FOLDS:
        r = readouts[f]
        print(f"[{f}] base={r['dev_mae_base']:.6f} P={r['dev_mae_P']:.6f} TP={r['dev_mae_TP']:.6f} "
              f"gain_base={r['dev_gain_base_vs_TP']:+.6f} gain_topo={r['dev_gain_P_vs_TP']:+.6f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ZINC topology cross-fit diagnostic v1")
    ap.add_argument("--mode", required=True, choices=("prep", "train", "readout"))
    ap.add_argument("--fold", choices=FOLDS)
    ap.add_argument("--epochs", type=int, default=zjd.EPOCHS)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if args.mode == "prep":
        return mode_prep()
    if args.mode == "train":
        assert args.fold, "--fold required"
        return mode_train(args.fold, args.epochs, args.device)
    return mode_readout()


if __name__ == "__main__":
    raise SystemExit(main())
