#!/usr/bin/env python
"""跨 seed 的“完整预测 + 冻结 R 干预响应”稳定性（任务书 §12.3 要求）。

只做 CPU 前向，不训练、不重选任何东西。输入必须是同一 arm 的两个 seed；
两 seed 用同一组 Valid 分子（official row id），逐分子比较：

  * 完整预测 pred 的一致性（Pearson / Spearman / 平均绝对差）；
  * 冻结干预响应 Δ = pred(R→ERASED 均值) − pred 的一致性；
  * 干预方向（Δ 符号）在分子级的一致率。

用法：
  uv run python code/ws_seed_stability.py --runs lipo_full_wgine_r1_s0 lipo_full_wgine_r1_s1
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / denom) if denom > 0 else float("nan")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    da = a - a.mean()
    db = b - b.mean()
    denom = np.sqrt((da @ da) * (db @ db))
    return float(da @ db / denom) if denom > 0 else float("nan")


def load_run_terms(run_id: str, device: str):
    """返回 (row_ids, pred_raw, delta_raw)；delta 是冻结 R→ERASED 均值的预测位移。"""
    import ws_analyze as A
    import ws_common as C

    run = json.loads((pathlib.Path("results") / run_id / "run.json").read_text())
    run["run_dir"] = str((pathlib.Path("results") / run_id).resolve())
    payload, _ = A.T.load_prepared(C.load_config()["datasets"][run["dataset"]])
    scaler = C.target_scaler(C.dataset_cache(C.load_config()["datasets"][run["dataset"]]))
    model, spec = A._load_model(run, payload, device)
    data = A.T.split_data(payload, run["partition"], "valid")
    base = A.forward_terms(model, data, device, extra=spec.extra == "attrpair")
    swapped = A.forward_terms(model, data, device, extra=spec.extra == "attrpair",
                              intervention="erased")
    std, mean = scaler["std"], scaler["mean"]
    pred = base["pred"] * std + mean
    pred_i = swapped["pred"] * std + mean
    row_ids = np.asarray(data.row_ids)
    return row_ids, pred, pred_i - pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=2, required=True,
                    help="同一 arm 的两个 seed run_id")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ids_a, pred_a, d_a = load_run_terms(args.runs[0], args.device)
    ids_b, pred_b, d_b = load_run_terms(args.runs[1], args.device)
    common = np.intersect1d(ids_a, ids_b)
    ia = np.searchsorted(np.sort(ids_a), common)
    ib = np.searchsorted(np.sort(ids_b), common)
    oa = np.argsort(ids_a)[ia]
    ob = np.argsort(ids_b)[ib]
    pa, pb, da, db = pred_a[oa], pred_b[ob], d_a[oa], d_b[ob]

    out = {
        "runs": list(args.runs),
        "n_molecules": int(len(common)),
        "prediction": {
            "pearson": _pearson(pa, pb),
            "spearman": _spearman(pa, pb),
            "mean_abs_diff": float(np.mean(np.abs(pa - pb))),
            "rmse_between_seeds": float(np.sqrt(np.mean((pa - pb) ** 2))),
        },
        "intervention_response": {
            "mean_abs_delta_a": float(np.mean(np.abs(da))),
            "mean_abs_delta_b": float(np.mean(np.abs(db))),
            "pearson": _pearson(da, db),
            "spearman": _spearman(da, db),
            "sign_agreement": float(np.mean(np.sign(da) == np.sign(db))),
            "mean_abs_delta_diff": float(np.mean(np.abs(da - db))),
        },
    }
    text = json.dumps(out, indent=1, ensure_ascii=False)
    print(text)
    if args.out:
        pathlib.Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
