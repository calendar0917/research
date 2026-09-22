#!/usr/bin/env python
"""TCCD-GLOBAL62-v0 runner.

Stages:

* ``gate0``  : frozen-base reproduction, exact 62-D global context, train-only
               standardization, label-free construction, gradient scope,
               derangement/multiset, bias-only train-only.
* ``stageA`` : train the single seed-0 REAL GLOBAL62 residual head and evaluate
               the evaluation-only GLOBAL-SHUFFLE intervention.

Single seed, internal train/dev only.  Official ZINC valid/test are never
loaded.  No existing baseline is retrained.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v5 as V5
from tracks.ksvd.code import global62_residual as G
from tracks.ksvd.code.run_tccd_v0 import internal_split

OUT_DIR = G.RESULTS_DIR


def _commit() -> str:
    return T.git("rev-parse", "HEAD")


def _save_state(path: Path, state: Any) -> None:
    if state is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    G._torch().save(state, path)


def _load_dataset():
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc

    return _load_zinc(G.REPO_ROOT / "data/ZINC", "train")


# ===========================================================================
# Gate 0
# ===========================================================================
def gate0(args) -> int:
    torch = G._torch()
    refs = G.frozen_reference_values()
    checks: dict[str, Any] = {"official_valid_loaded": False, "official_test_loaded": False}

    # ---- 1/2. frozen v5 PRE soup reproduction + frozen TCCD weights ----
    base = G.frozen_base_predictions(args.device)
    pred = base["pred"]
    cache = base["cache"]
    y = np.asarray(cache.y, dtype=np.float64)
    train_idx, dev_idx = internal_split(len(cache.graph_index))
    train_idx = list(map(int, train_idx))
    dev_idx = list(map(int, dev_idx))
    mae_base = float(np.abs(pred[dev_idx].astype(np.float64) - y[dev_idx]).mean())
    checks["reproduced_dev_mae"] = mae_base
    checks["reference_dev_mae"] = float(refs["pre_soup"])
    checks["reproduction_abs_delta"] = abs(mae_base - float(refs["pre_soup"]))
    checks["reproduction_pass"] = bool(abs(mae_base - float(refs["pre_soup"])) <= G.REPRODUCTION_TOL)
    checks["v5_pre_soup_sha256"] = base["soup_sha256"]
    checks["v2_checkpoint_sha256"] = G.V2_BEST_CHECKPOINT_SHA256
    checks["pair_cache_official_test_loaded"] = cache.meta.get("official_test_loaded")
    checks["pair_cache_v2_sha_ok"] = bool(cache.meta.get("v2_checkpoint_sha256") == G.V2_BEST_CHECKPOINT_SHA256)
    checks["base_pred_finite"] = bool(np.isfinite(pred).all())
    # no TCCD training: the soup state is loaded frozen; a head backward must not
    # touch it, and no frozen model is ever instantiated as trainable.
    soup_state = torch.load(G.V5_PRE_SOUP_STATE, map_location="cpu", weights_only=True)
    frozen_model = V5.StageAModelFactory.build("pre", seed=G.SEED)
    frozen_model.load_state_dict(soup_state)
    frozen_model.requires_grad_(False)
    checks["frozen_tccd_all_requires_grad_false"] = bool(
        all(not p.requires_grad for p in frozen_model.parameters())
    )
    checks["frozen_tccd_never_trained"] = True

    # ---- 3. exact 62-D B-Full global context definition ----
    g62 = G.load_global62_raw()
    raw_all = g62["global_all"]
    checks["global_width"] = int(raw_all.shape[1])
    checks["global_width_ok"] = bool(raw_all.shape[1] == G.GLOBAL_WIDTH)
    checks["global_n_graphs"] = int(raw_all.shape[0])
    checks["global_n_matches_cache"] = bool(raw_all.shape[0] == len(cache.graph_index))
    checks["global_finite_rate"] = float(g62["finite_rate"])
    checks["global_all_finite"] = bool(g62["finite_rate"] == 1.0)
    checks["global_function_provenance"] = G.GLOBAL_FEATURE_FUNCTION
    checks["global_encoder_provenance"] = G.GLOBAL_ENCODER_PROVENANCE
    # width decomposition and histogram normalization
    checks["block_widths"] = {
        "short": int(g62["short"].shape[1]),
        "long": int(g62["long"].shape[1]),
        "attributes": int(g62["attributes"].shape[1]),
    }
    atom = raw_all[:, G.GLOBAL_STRUCTURE_SHORT + G.GLOBAL_STRUCTURE_LONG :][:, : G.GLOBAL_ATOM_BINS]
    bond = raw_all[:, G.GLOBAL_STRUCTURE_SHORT + G.GLOBAL_STRUCTURE_LONG + G.GLOBAL_ATOM_BINS :]
    checks["atom_hist_sums_max_abs_dev"] = float(np.abs(atom.sum(axis=1) - 1.0).max())
    checks["bond_hist_sums_max_abs_dev"] = float(np.abs(bond.sum(axis=1) - 1.0).max())
    checks["histograms_normalized"] = bool(
        checks["atom_hist_sums_max_abs_dev"] <= 1e-5 and checks["bond_hist_sums_max_abs_dev"] <= 1e-5
    )
    # alignment with the TCCD record labels
    checks["label_alignment_max_abs_diff"] = float(np.abs(g62["dataset_y"] - y).max())
    checks["label_alignment_ok"] = bool(np.abs(g62["dataset_y"] - y).max() <= 1e-6)
    # determinism of the global features
    g62_again = G.load_global62_raw()
    checks["global_determinism_max_abs_diff"] = float(np.abs(g62_again["global_all"] - raw_all).max())
    checks["global_deterministic"] = bool(checks["global_determinism_max_abs_diff"] == 0.0)

    # ---- 4. train-only standardization ----
    std = G.fit_train_only_standardizer(raw_all, train_idx)
    Xtr = std.transform(raw_all[train_idx])
    Xdv = std.transform(raw_all[dev_idx])
    checks["standardizer_fit_indices"] = "internal_train_8000_only"
    checks["standardizer_mean_from_train_max_abs_diff"] = float(
        np.abs(std.mean.astype(np.float64) - raw_all[train_idx].astype(np.float64).mean(axis=0)).max()
    )
    nonconst = [i for i in range(G.GLOBAL_WIDTH) if i not in std.constant_columns]
    train_std = raw_all[train_idx].astype(np.float64).std(axis=0)
    checks["standardizer_std_from_train"] = bool(
        np.abs(std.std[nonconst].astype(np.float64) - train_std[nonconst]).max() <= 1e-5
    )
    checks["constant_columns"] = list(std.constant_columns)
    checks["train_mean"] = std.mean.tolist()
    checks["train_std"] = std.std.tolist()
    checks["train_standardized_mean_max_abs"] = float(np.abs(Xtr.mean(axis=0)).max())
    checks["train_standardized_std_max_abs_dev"] = float(
        np.abs(Xtr[:, nonconst].std(axis=0) - 1.0).max()
    )

    # ---- 5. labels never enter feature construction ----
    import copy as _copy

    dataset = _load_dataset()
    probe_idx = [0, 1, 2, 3, 4]
    original = [dataset[i] for i in probe_idx]
    perturbed = []
    for i in probe_idx:
        d = _copy.deepcopy(dataset[i])
        d.y = torch.as_tensor([[float(i) + 123.0]], dtype=d.y.dtype)
        perturbed.append(d)
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import global_feature_views

    views_orig = global_feature_views(original)["global_all"]
    views_pert = global_feature_views(perturbed)["global_all"]
    checks["label_free_max_abs_diff"] = float(np.abs(np.asarray(views_orig) - np.asarray(views_pert)).max())
    checks["label_free"] = bool(checks["label_free_max_abs_diff"] == 0.0)

    # ---- 8/11. gradient scope + bias-only train-only ----
    torch.manual_seed(G.SEED)
    r_train = y[train_idx] - pred[train_idx].astype(np.float64)
    b_star = float(np.median(r_train))
    checks["bias_star"] = b_star
    checks["bias_train_only"] = True
    dev_bias_mae = float(np.abs(pred[dev_idx].astype(np.float64) + b_star - y[dev_idx]).mean())
    checks["dev_mae_bias"] = dev_bias_mae

    model = G.global62_head_class()
    opt = torch.optim.Adam(model.parameters(), lr=G.LR, weight_decay=G.WD)
    xt = torch.as_tensor(Xtr[:64], dtype=torch.float32)
    bt = torch.as_tensor(pred[train_idx[:64]].astype(np.float32))
    yt = torch.as_tensor(y[train_idx[:64]].astype(np.float32))
    opt.zero_grad(set_to_none=True)
    ((bt + model(xt)) - yt).abs().mean().backward()
    grads = {name: float(p.grad.abs().sum()) for name, p in model.named_parameters() if p.grad is not None}
    checks["head_gradients"] = grads
    checks["gradient_reaches_head"] = bool(all(v > 0 for v in grads.values()) and len(grads) == len(list(model.parameters())))
    checks["frozen_tccd_grad_none_after_head_backward"] = bool(
        all(p.grad is None for p in frozen_model.parameters())
    )
    checks["parameter_accounting"] = G.parameter_accounting()
    checks["parameter_accounting_ok"] = bool(G.parameter_accounting()["total"] == 3169)

    # ---- 9/10. shuffle preserves dev multiset, no fixed points ----
    perm = G.derangement(len(dev_idx))
    checks["shuffle_seed"] = G.SHUFFLE_SEED
    checks["shuffle_has_fixed_points"] = bool(np.any(perm == np.arange(len(dev_idx))))
    checks["shuffle_no_fixed_points"] = bool(not np.any(perm == np.arange(len(dev_idx))))
    checks["shuffle_permutation_valid"] = bool(np.array_equal(np.sort(perm), np.arange(len(dev_idx))))
    shuffled_dev = Xdv[perm]
    checks["shuffle_multiset_max_abs_diff"] = float(
        np.abs(np.sort(shuffled_dev, axis=0) - np.sort(Xdv, axis=0)).max()
    )
    checks["shuffle_multiset_preserved"] = bool(checks["shuffle_multiset_max_abs_diff"] == 0.0)
    checks["shuffle_changes_rows"] = bool(float(np.abs(shuffled_dev - Xdv).max()) > 0.0)

    checks["n_train"] = len(train_idx)
    checks["n_dev"] = len(dev_idx)
    checks["split_seed"] = int(T.SPLIT_SEED)

    critical = [
        "reproduction_pass",
        "frozen_tccd_all_requires_grad_false",
        "global_width_ok",
        "global_all_finite",
        "histograms_normalized",
        "label_alignment_ok",
        "global_deterministic",
        "standardizer_std_from_train",
        "label_free",
        "gradient_reaches_head",
        "frozen_tccd_grad_none_after_head_backward",
        "parameter_accounting_ok",
        "shuffle_no_fixed_points",
        "shuffle_multiset_preserved",
    ]
    checks["critical_checks"] = critical
    checks["all_pass"] = bool(all(checks[k] for k in critical))

    payload = {
        "protocol": G.PROTOCOL_VERSION,
        "stage": "gate0",
        "commit": _commit(),
        "device": str(args.device),
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "frozen_references": refs,
        "checks": checks,
    }
    G.write_json(OUT_DIR / "gate0.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0 if checks["all_pass"] else 1


# ===========================================================================
# Stage A
# ===========================================================================
def stage_a(args) -> int:
    if int(args.seed) != 0:
        raise SystemExit("TCCD-GLOBAL62-v0 is a single-seed round; seed 0 only")
    started = time.time()
    refs = G.frozen_reference_values()

    base = G.frozen_base_predictions(args.device)
    pred = base["pred"]
    cache = base["cache"]
    y = np.asarray(cache.y, dtype=np.float64)
    train_idx, dev_idx = internal_split(len(cache.graph_index))
    train_idx = list(map(int, train_idx))
    dev_idx = list(map(int, dev_idx))

    mae_base = float(np.abs(pred[dev_idx].astype(np.float64) - y[dev_idx]).mean())
    if abs(mae_base - float(refs["pre_soup"])) > G.REPRODUCTION_TOL:
        raise SystemExit(f"frozen base reproduction FAILED: {mae_base} vs {refs['pre_soup']}")

    g62 = G.load_global62_raw()
    raw_all = g62["global_all"]
    std = G.fit_train_only_standardizer(raw_all, train_idx)
    Xtr = std.transform(raw_all[train_idx])
    Xdv = std.transform(raw_all[dev_idx])

    r_train = (y[train_idx] - pred[train_idx].astype(np.float64)).astype(np.float32)
    b_star = float(np.median(r_train))
    mae_bias = float(np.abs(pred[dev_idx].astype(np.float64) + b_star - y[dev_idx]).mean())

    res = G.train_residual_head(
        Xtr,
        r_train,
        pred[train_idx].astype(np.float32),
        y[train_idx].astype(np.float32),
        Xdv,
        pred[dev_idx].astype(np.float32),
        y[dev_idx].astype(np.float32),
        seed=G.SEED,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch=args.batch,
        log=print,
    )

    # REAL dev predictions (soup primary, best secondary)
    real_soup = G.head_predictions(res.state_soup, Xdv, pred[dev_idx])
    real_best = G.head_predictions(res.state_best, Xdv, pred[dev_idx])
    # SHUFFLE: derange the dev global-feature rows only; base and head unchanged
    perm = G.derangement(len(dev_idx))
    Xdv_shuf = Xdv[perm]
    shuf_soup = G.head_predictions(res.state_soup, Xdv_shuf, pred[dev_idx])
    shuf_best = G.head_predictions(res.state_best, Xdv_shuf, pred[dev_idx])

    ydev = y[dev_idx]
    mae_real_soup = float(np.abs(real_soup.astype(np.float64) - ydev).mean())
    mae_real_best = float(np.abs(real_best.astype(np.float64) - ydev).mean())
    mae_shuf_soup = float(np.abs(shuf_soup.astype(np.float64) - ydev).mean())
    mae_shuf_best = float(np.abs(shuf_best.astype(np.float64) - ydev).mean())

    g_total = float(mae_base - mae_real_soup)
    g_global = float(mae_bias - mae_real_soup)
    g_bind = float(mae_shuf_soup - mae_real_soup)
    g_total_best = float(mae_base - mae_real_best)
    g_global_best = float(mae_bias - mae_real_best)
    g_bind_best = float(mae_shuf_best - mae_real_best)
    gap_closed = float((mae_base - mae_real_soup) / (mae_base - G.B_FULL_SCALE_REFERENCE))
    verdict = G.preregistered_verdict(g_total, g_global, g_bind)

    # secondary analyses (read-only)
    raw_dev = raw_all[dev_idx]
    raw_train = raw_all[train_idx]
    corr_train = G.residual_correlations(raw_train, (y[train_idx] - pred[train_idx].astype(np.float64)))
    corr_dev = G.residual_correlations(raw_dev, (ydev - pred[dev_idx].astype(np.float64)))
    strata = G.size_stratification(raw_dev[:, 0], ydev, pred[dev_idx], real_soup, shuf_soup)

    payload = {
        "protocol": G.PROTOCOL_VERSION,
        "stage": "A_frozen_global62_residual",
        "commit": _commit(),
        "device": str(args.device),
        "seed": int(args.seed),
        "split_seed": int(T.SPLIT_SEED),
        "n_train": len(train_idx),
        "n_dev": len(dev_idx),
        "v2_checkpoint": str(G.V2_BEST_CHECKPOINT.relative_to(G.REPO_ROOT)),
        "v2_checkpoint_sha256": G.V2_BEST_CHECKPOINT_SHA256,
        "v5_pre_soup_state": str(G.V5_PRE_SOUP_STATE.relative_to(G.REPO_ROOT)),
        "v5_pre_soup_sha256": base["soup_sha256"],
        "pair_cache_meta": cache.meta,
        "global_feature_function": G.GLOBAL_FEATURE_FUNCTION,
        "global_encoder_provenance": G.GLOBAL_ENCODER_PROVENANCE,
        "parameter_accounting": G.parameter_accounting(),
        "init_checksum": res.init_checksum,
        "primary_metric": "top5_soup",
        "train_history": res.train_history,
        "shuffle_seed": G.SHUFFLE_SEED,
        "standardizer": {
            "fit": "internal_train_8000_only",
            "constant_columns": list(std.constant_columns),
            "train_mean": std.mean.tolist(),
            "train_std": std.std.tolist(),
        },
        "frozen_references": refs,
        "results": {
            "BASE": {
                "soup_valid_mae": mae_base,
                "source": "exact frozen TCCD-v5 PRE Top-5 soup forward (reproduced)",
            },
            "BIAS_ONLY": {
                "bias_star": b_star,
                "soup_valid_mae": mae_bias,
                "source": "train-residual median; no optimizer",
            },
            "REAL_GLOBAL62": {
                "best_valid_mae": mae_real_best,
                "soup_valid_mae": mae_real_soup,
                "best_epoch": int(res.best_epoch),
                "soup_members": list(res.soup_members),
                "init_checksum": res.init_checksum,
                "wall_s": float(res.wall_s),
                "source": "new this round (tiny head only)",
            },
            "SHUFFLED_GLOBAL62": {
                "best_valid_mae": mae_shuf_best,
                "soup_valid_mae": mae_shuf_soup,
                "source": "evaluation-only deterministic row derangement on dev global features",
            },
            "B_FULL": {
                "soup_valid_mae": G.B_FULL_SCALE_REFERENCE,
                "source": "existing scale reference only",
            },
        },
        "g_total": g_total,
        "g_global": g_global,
        "g_bind": g_bind,
        "g_total_best": g_total_best,
        "g_global_best": g_global_best,
        "g_bind_best": g_bind_best,
        "gap_closed_fraction": gap_closed,
        "verdict": verdict,
        "secondary": {
            "residual_correlation_train": corr_train,
            "residual_correlation_dev": corr_dev,
            "size_stratification": strata,
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "wall_s": time.time() - started,
    }
    G.write_json(OUT_DIR / "stageA_seed0.json", payload)
    G.write_json(OUT_DIR / "stageA_decision.json", {"protocol": G.PROTOCOL_VERSION, **verdict})
    _save_state(OUT_DIR / "global62_head_seed0_soup.pt", res.state_soup)
    _save_state(OUT_DIR / "global62_head_seed0_best.pt", res.state_best)
    np.save(OUT_DIR / "base_dev_pred.npy", pred[dev_idx])
    np.save(OUT_DIR / "dev_real_soup.npy", real_soup)
    np.save(OUT_DIR / "dev_real_best.npy", real_best)
    np.save(OUT_DIR / "dev_shuffle_soup.npy", shuf_soup)
    np.save(OUT_DIR / "dev_shuffle_best.npy", shuf_best)
    np.save(OUT_DIR / "dev_indices.npy", np.asarray(dev_idx, dtype=np.int64))
    np.save(OUT_DIR / "global62_raw.npy", raw_all)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["gate0", "stageA"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=G.BATCH)
    p.add_argument("--max-epochs", type=int, default=G.MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=G.PATIENCE)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "gate0":
        return gate0(args)
    if args.stage == "stageA":
        return stage_a(args)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
