"""Phase A: the six new XGBoost attribution arms (MolHIV).

Reads the committed label-free bundle (``bundle/sav1_inputs.npz``), so the remote
host never re-derives a representation.  Fits exactly one arm per invocation
(``--arm``), which keeps the remote jobs independent and lets Phase A run in
parallel with Phase B on the same checkout.

The frozen recipe of PROTOCOL.md §6.2 is hard-coded here: no tuning of any kind.

Outputs (``--out-dir``):
  <arm>.json      recipe, metrics, feature usage, timing, provenance
  <arm>_pred.npz  per-molecule Train/Valid/Test logits + row ids
  <arm>_model.json xgboost booster dump (used by Phase D decision-path diagnostics)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

import sga_common as K

#: PROTOCOL.md §6.2 — frozen, shared with ZX0/ZX1 (only the objective changes there)
XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "min_child_weight": 1,
    "tree_method": "hist",
    "random_state": K.SEED,
    "device": "cpu",
    "n_jobs": 8,
}

#: arm -> (extra block names in design order)
ARMS: dict[str, list[str]] = {
    "X-MC": ["MC"],
    "X-MU": ["MU"],
    "X-WL": ["WL"],
    "X-C": ["C"],
    "X-F": ["F"],
    "X-PRES": ["PRES"],
}


def load_bundle(path: Path | None = None) -> dict[str, np.ndarray]:
    path = path or K.INPUT_BUNDLE
    with np.load(path) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def unpack_h(bundle: dict[str, np.ndarray]) -> np.ndarray:
    n_bits = int(bundle["H_n_bits"])
    shape = tuple(int(v) for v in bundle["H_shape"])
    bits = np.unpackbits(bundle["H_bits_packed"], axis=1)[:, :n_bits].astype(np.float32)
    out = np.empty(shape, dtype=np.float32)
    out[:, :n_bits] = bits
    out[:, n_bits:] = np.asarray(bundle["H_desc"], dtype=np.float32)
    return out


def design_rows(
    h: np.ndarray, blocks: list[tuple[str, sp.csr_matrix]], rows: np.ndarray
) -> np.ndarray:
    width = h.shape[1] + sum(m.shape[1] for _, m in blocks)
    out = np.empty((rows.size, width), dtype=np.float32)
    out[:, : h.shape[1]] = h[rows]
    col = h.shape[1]
    for _name, mat in blocks:
        sub = mat[rows]
        block = np.zeros((rows.size, mat.shape[1]), dtype=np.float32)
        coo = sub.tocoo()
        block[coo.row, coo.col] = coo.data
        out[:, col: col + mat.shape[1]] = block
        col += mat.shape[1]
    return out


def metrics(y: np.ndarray, logit: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    p = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logit, dtype=np.float64), -60, 60)))
    out = {
        "n": int(y.size),
        "n_positive": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
    }
    from ogb.graphproppred import Evaluator

    ev = Evaluator(name="ogbg-molhiv")
    ogb_auc = float(ev.eval({"y_true": y.reshape(-1, 1), "y_pred": p.reshape(-1, 1)})["rocauc"])
    out["ogb_evaluator_rocauc"] = ogb_auc
    out["ogb_abs_diff"] = abs(ogb_auc - out["roc_auc"])
    return out


def feature_usage(booster, offset: int, width: int, block_sizes: dict[str, tuple[int, int]]) -> dict[str, Any]:
    gain = booster.get_score(importance_type="total_gain")
    gains = {int(k[1:]): float(v) for k, v in gain.items()}
    total = sum(gains.values())
    struct = {c: g for c, g in gains.items() if c >= offset}
    out = {
        "total_gain": float(total),
        "historical_block_gain": float(total - sum(struct.values())),
        "structural_block_gain": float(sum(struct.values())),
        "structural_block_gain_share": float(sum(struct.values()) / total) if total else None,
        "n_columns_total": int(width),
        "n_columns_historical": int(offset),
        "n_used_features_total": int(len(gains)),
        "n_used_features_structural": int(len(struct)),
        "blocks": {},
    }
    for name, (lo, size) in block_sizes.items():
        cols = {c: g for c, g in struct.items() if lo <= c < lo + size}
        out["blocks"][name] = {
            "offset": int(lo),
            "size": int(size),
            "n_used": int(len(cols)),
            "gain": float(sum(cols.values())),
            "gain_share_of_structure": float(sum(cols.values()) / sum(struct.values())) if struct else None,
        }
    top = sorted(struct.items(), key=lambda kv: -kv[1])[:60]
    out["top_structural_features"] = [
        {"column": int(c), "block_column": int(c - offset), "total_gain": float(g)}
        for c, g in top
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--out-dir", default=str(K.RUNS / "phaseA"))
    ap.add_argument("--bundle", default=str(K.INPUT_BUNDLE))
    ap.add_argument("--smoke", action="store_true",
                    help="ENGINEERING SMOKE ONLY: 20 trees, tagged smoke=true, never analysed")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.perf_counter()

    import xgboost as xgb

    bundle = load_bundle(Path(args.bundle))
    y = np.asarray(bundle["y"], dtype=np.int64)
    splits = {name: np.asarray(bundle[f"rows_{name}"], dtype=np.int64)
              for name in ("train", "valid", "test")}

    extra_names = ARMS[args.arm]
    h = unpack_h(bundle)
    blocks: list[tuple[str, sp.csr_matrix]] = []
    block_sizes: dict[str, tuple[int, int]] = {}
    col = h.shape[1]
    for name in extra_names:
        mat = sp.csr_matrix(
            (
                np.asarray(bundle[f"{name}_data"], dtype=np.float32),
                np.asarray(bundle[f"{name}_indices"], dtype=np.int64),
                np.asarray(bundle[f"{name}_indptr"], dtype=np.int64),
            ),
            shape=(h.shape[0], int(np.asarray(bundle[f"{name}_indices"]).max()) + 1),
        )
        blocks.append((name, mat))
        block_sizes[name] = (col, mat.shape[1])
        col += mat.shape[1]
    width = col
    print(f"[{args.arm}] design width={width} blocks={[(n, m.shape[1]) for n, m in blocks]}", flush=True)

    params = dict(XGB_PARAMS)
    if args.smoke:
        params["n_estimators"] = 20
    Xtr = design_rows(h, blocks, splits["train"])
    ytr = y[splits["train"]]
    model = xgb.XGBClassifier(**params, eval_metric="logloss")
    t_fit = time.perf_counter()
    model.fit(Xtr, ytr.astype(int))
    fit_seconds = time.perf_counter() - t_fit
    del Xtr
    print(f"[{args.arm}] fitted in {fit_seconds:.1f}s", flush=True)

    per_split: dict[str, Any] = {}
    preds: dict[str, np.ndarray] = {}
    for name in ("train", "valid", "test"):
        rows = splits[name]
        Xs = design_rows(h, blocks, rows)
        logit = model.predict(Xs, output_margin=True)
        del Xs
        per_split[name] = metrics(y[rows], logit)
        if name == "test":
            per_split[name]["evidence_label"] = "EXPOSED_DEVELOPMENT_TEST"
        preds[name] = logit.astype(np.float64)

    booster = model.get_booster()
    usage = feature_usage(booster, h.shape[1], width, block_sizes)
    model_path = out_dir / f"{args.arm}{'_smoke' if args.smoke else ''}_model.json"
    booster.save_model(model_path)

    payload = {
        "round_id": K.ROUND_ID,
        "phase": "A: XGBoost gain attribution",
        "arm": args.arm,
        "extra_blocks": extra_names,
        "created_utc": created,
        "recipe": {**params, "eval_metric": "logloss", "objective": "binary:logistic"},
        "smoke": bool(args.smoke),
        "dense_input": True,
        "n_columns": int(width),
        "fit_seconds": float(fit_seconds),
        "metrics": per_split,
        "feature_usage": usage,
        "model_dump": {"path": str(model_path), "size_bytes": int(model_path.stat().st_size)},
        "input_bundle": {"path": K.rel(K.INPUT_BUNDLE), "sha256": K.sha256_file(K.INPUT_BUNDLE)},
        "bundle_meta_sha256": K.sha256_file(K.INPUT_BUNDLE_META) if K.INPUT_BUNDLE_META.exists() else None,
        "git": K.git_state(),
        "environment": K.environment(),
        "seconds": float(time.perf_counter() - t0),
    }
    tag = args.arm + ("_smoke" if args.smoke else "")
    (out_dir / f"{tag}.json").write_text(json.dumps(payload, indent=2, sort_keys=True,
                                                   default=K._json_default) + "\n")
    np.savez_compressed(
        out_dir / f"{tag}_pred.npz",
        train=preds["train"], valid=preds["valid"], test=preds["test"],
        train_rows=splits["train"], valid_rows=splits["valid"], test_rows=splits["test"],
        y=y,
    )
    print(f"[{args.arm}] valid={per_split['valid']['roc_auc']:.4f} test={per_split['test']['roc_auc']:.4f} "
          f"struct_gain_share={usage['structural_block_gain_share']:.3f} "
          f"done in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
