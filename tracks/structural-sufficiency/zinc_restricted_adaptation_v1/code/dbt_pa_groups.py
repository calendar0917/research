"""Fixed four-group descriptive error tables for valid and test.

Groups (fixed before looking at the new model's test result):

* ``covered_ordinary``: all atoms' radius-2 content signatures seen in train,
  ``y > -3``;
* ``covered_tail``: all covered, ``y <= -3``;
* ``unseen_ordinary``: at least one unseen radius-2 signature, ``y > -3``;
* ``unseen_tail``: at least one unseen signature, ``y <= -3``.

The label split is analysis-only; it never enters training, sampling or
inference.  Row 172 is kept everywhere; a without-row-172 number is reported
only as a sensitivity note.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
_DBT = _SSUF / "zinc_directed_bond_transformer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_DBT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

ANALYSIS = _EXP / "analysis"
RUN = _EXP / "runs" / "pa_s0"
GROUPS = (
    ("covered_ordinary", 0, True),
    ("covered_tail", 0, False),
    ("unseen_ordinary", 1, True),
    ("unseen_tail", 1, False),
)
MODELS = ("P3B_s0", "L_s0", "DBT_s0", "PA_s0")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_preds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    return np.asarray(d["row"], dtype=np.int64), np.asarray(d["pred"], dtype=np.float64)


def _load_split(split: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rows = _read_csv(ANALYSIS / f"reference_preds_{split}.csv")
    n = len(rows)
    row = np.asarray([int(r["row"]) for r in rows], dtype=np.int64)
    if not np.array_equal(row, np.arange(n)):
        raise RuntimeError(f"{split} reference rows are not 0..{n - 1}")
    y = np.asarray([float(r["y_raw"]) for r in rows], dtype=np.float64)
    preds = {name: np.asarray([float(r[f"pred_{name}"]) for r in rows], dtype=np.float64)
             for name in ("P3B_s0", "L_s0", "DBT_s0")}
    unseen = np.asarray([int(r["signature_unseen"]) for r in rows], dtype=np.int64)
    frac = np.asarray([float(r["r2_unseen_atom_frac"]) for r in rows], dtype=np.float64)
    ordinary = np.asarray([int(r["ordinary_y_gt_minus3"]) for r in rows], dtype=np.int64)
    tail = np.asarray([int(r["tail_y_le_minus3"]) for r in rows], dtype=np.int64)
    if not np.array_equal(unseen, (frac > 0).astype(np.int64)):
        raise RuntimeError(f"{split} signature_unseen flag disagrees with fraction")
    if not np.array_equal(ordinary + tail, np.ones(n, dtype=np.int64)):
        raise RuntimeError(f"{split} ordinary/tail flags are not complementary")
    if not np.array_equal(ordinary, (y > -3.0).astype(np.int64)):
        raise RuntimeError(f"{split} ordinary flag disagrees with y")
    out = {"row": row, "y": y, "unseen": unseen, "ordinary": ordinary, "tail": tail,
           **preds}
    if split == "valid":
        pred_rows, pa = _load_preds(RUN / "predictions" / "valid_best.npz")
        if not np.array_equal(pred_rows, row):
            raise RuntimeError("new valid predictions are not row-aligned")
        out["PA_s0"] = pa
        sha = C.sha256_array(pa)
    else:
        pred_rows, pa = _load_preds(RUN / "predictions" / "test_best.npz")
        if not np.array_equal(pred_rows, row):
            raise RuntimeError("new test predictions are not row-aligned")
        out["PA_s0"] = pa
        sha = C.sha256_array(pa)
    meta = {
        "split": split,
        "n": n,
        "reference_csv": str((ANALYSIS / f"reference_preds_{split}.csv").relative_to(C.REPO)),
        "new_prediction_path": str((RUN / "predictions" /
                                    ("valid_best.npz" if split == "valid" else "test_best.npz")).relative_to(C.REPO)),
        "new_prediction_sha256": sha,
        "labels_analysis_only": True,
        "row_172_kept": True,
    }
    return out, meta


def _tables(data: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    y = data["y"]
    ae = {name: np.abs(data[name] - y) for name in MODELS}
    total_ae = {name: float(ae[name].sum()) for name in MODELS}
    rows = []
    for gname, unseen_flag, ordinary_flag in GROUPS:
        mask = (data["unseen"] == unseen_flag) & (
            data["ordinary"] == 1 if ordinary_flag else data["tail"] == 1
        )
        n = int(mask.sum())
        if n == 0:
            raise RuntimeError(f"empty group {gname}")
        group = {"group": gname, "n": n, "labels": {"signature_unseen": bool(unseen_flag),
                                                    "ordinary_y_gt_minus3": bool(ordinary_flag)},
                 "models": {}}
        for name in MODELS:
            sum_ae = float(ae[name][mask].sum())
            group["models"][name] = {
                "mae": float(ae[name][mask].mean()),
                "sum_abs_error": sum_ae,
                "share_of_total_abs_error": float(sum_ae / total_ae[name]),
            }
        for ref in ("P3B_s0", "L_s0", "DBT_s0"):
            group[f"DBT_minus_{ref}"] = {
                "delta_mae": float(group["models"]["DBT_s0"]["mae"]
                                   - group["models"][ref]["mae"]),
                "delta_sum_abs_error": float(group["models"]["DBT_s0"]["sum_abs_error"]
                                             - group["models"][ref]["sum_abs_error"]),
            }
        for ref in ("P3B_s0", "L_s0", "DBT_s0"):
            group[f"PA_minus_{ref}"] = {
                "delta_mae": float(group["models"]["PA_s0"]["mae"]
                                   - group["models"][ref]["mae"]),
                "delta_sum_abs_error": float(group["models"]["PA_s0"]["sum_abs_error"]
                                             - group["models"][ref]["sum_abs_error"]),
            }
        rows.append(group)

    overall = {"group": "all", "n": int(len(y)), "models": {}}
    for name in MODELS:
        overall["models"][name] = {
            "mae": float(ae[name].mean()),
            "sum_abs_error": total_ae[name],
            "share_of_total_abs_error": 1.0,
        }
    for ref in ("P3B_s0", "L_s0", "DBT_s0"):
        overall[f"DBT_minus_{ref}"] = {
            "delta_mae": float(overall["models"]["DBT_s0"]["mae"]
                               - overall["models"][ref]["mae"]),
        }
        overall[f"PA_minus_{ref}"] = {
            "delta_mae": float(overall["models"]["PA_s0"]["mae"]
                               - overall["models"][ref]["mae"]),
        }
    sensitivity = {"without_row_172": {}}
    keep = data["row"] != 172
    for name in MODELS:
        sensitivity["without_row_172"][name] = {
            "n": int(keep.sum()),
            "mae": float(ae[name][keep].mean()),
            "delta_mae_vs_full": float(ae[name][keep].mean() - ae[name].mean()),
        }
    i172 = int(np.where(data["row"] == 172)[0][0])
    sensitivity["row_172"] = {
        "row": 172, "y": float(y[i172]), "unseen": int(data["unseen"][i172]),
        "tail": int(data["tail"][i172]),
        "predictions": {name: float(data[name][i172]) for name in MODELS},
        "abs_errors": {name: float(ae[name][i172]) for name in MODELS},
        "note": "kept in every headline number; the without-row sensitivity is descriptive",
    }
    return rows, [overall, sensitivity]


def _markdown(split: str, groups: list[dict[str, Any]], overall: dict[str, Any],
              sensitivity: dict[str, Any]) -> str:
    lines = [f"### {split}", "",
             "| group | n | model | MAE | sum AE | share of total AE | ΔMAE vs P3B | ΔMAE vs L | ΔMAE vs DBT |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for g in groups + [overall]:
        for name in MODELS:
            m = g["models"][name]
            d_p3b = d_l = d_dbt = ""
            if name == "DBT_s0":
                d_p3b = f"{g['DBT_minus_P3B_s0']['delta_mae']:+.6f}"
                d_l = f"{g['DBT_minus_L_s0']['delta_mae']:+.6f}"
            elif name == "PA_s0":
                d_p3b = f"{g['PA_minus_P3B_s0']['delta_mae']:+.6f}"
                d_l = f"{g['PA_minus_L_s0']['delta_mae']:+.6f}"
                d_dbt = f"{g['PA_minus_DBT_s0']['delta_mae']:+.6f}"
            lines.append(
                f"| {g['group']} | {g['n']} | {name} | {m['mae']:.6f} | "
                f"{m['sum_abs_error']:.3f} | {m['share_of_total_abs_error']:.3f} | "
                f"{d_p3b} | {d_l} | {d_dbt} |"
            )
    lines += ["", "Sensitivity (descriptive only; row 172 stays in every headline):", "",
              "| model | MAE without row 172 | Δ vs full |", "|---|---:|---:|"]
    for name in MODELS:
        s = sensitivity["without_row_172"][name]
        lines.append(f"| {name} | {s['mae']:.6f} | {s['delta_mae_vs_full']:+.6f} |")
    r = sensitivity["row_172"]
    lines += ["", f"Row 172: y={r['y']:.6f}, unseen={r['unseen']}, tail={r['tail']}, "
              + ", ".join(f"{k} AE={v:.6f}" for k, v in r["abs_errors"].items()) + "."]
    return "\n".join(lines)

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["valid", "test", "both"], default="both")
    args = p.parse_args(argv)
    splits = ("valid", "test") if args.split == "both" else (args.split,)
    chunks = ["# Fixed four-group error tables", "",
              "Signature grouping uses the corrected radius-2 content keys relative to train "
              "(the same keys as the audit's `r2_unseen_atom_frac`). "
              "The ordinary/tail label split is analysis-only and never enters the model.",
              ""]
    meta_all = {}
    for split in splits:
        data, meta = _load_split(split)
        groups, (overall, sensitivity) = _tables(data)
        C.write_json(ANALYSIS / f"groups_{split}.json", {
            "meta": meta, "groups": groups, "overall": overall,
            "sensitivity": sensitivity,
        })
        chunks.append(_markdown(split, groups, overall, sensitivity))
        chunks.append("")
        meta_all[split] = meta
    (ANALYSIS / "groups.md").write_text("\n".join(chunks) + "\n", encoding="utf-8")
    C.write_json(ANALYSIS / "groups_meta.json", meta_all)
    C.log("\n".join(chunks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
