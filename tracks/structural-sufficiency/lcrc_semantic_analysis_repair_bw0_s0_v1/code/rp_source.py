"""Input identity + independent metric recomputation (no training).

Produces ``analysis/source_manifest.json``: for every artifact this round
actually reads, the run id, git commit, config/resolved recipe, data/split/
normaliser/epoch-plan hashes, checkpoint tag + state hash, and prediction-file
hash.  Then independently recomputes best/last RMSE / MAE / bias / Train RMSE
from the stored raw predictions and checks them against each run's
``metrics.json`` (the predecessor's machine table).

Nothing here trains, selects or writes into a predecessor directory.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import rp_common as RP


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run_meta(arm_dir: Path) -> dict[str, Any]:
    m = RP.read_json(arm_dir / "metrics.json")
    summary = m.get("summary", {})
    return {
        "arm_dir": RP.track_rel(arm_dir),
        "metrics_sha256": RP.sha256_file(arm_dir / "metrics.json"),
        "protocol_id": m.get("protocol_id"),
        "git_commit": m.get("git_commit"),
        "hostname": m.get("hostname"),
        "torch": m.get("torch"),
        "family": m.get("family"),
        "recipe": m.get("recipe"),
        "budget": m.get("budget"),
        "dataset": m.get("dataset"),
        "seed": m.get("seed"),
        "parameters": (m.get("model") or {}).get("parameters"),
        "recipe_full": m.get("recipe_full"),
        "target_norm": (m.get("data") or {}).get("target_norm"),
        "train_norm_fingerprint": (m.get("data") or {}).get("train_norm_fingerprint"),
        "train_norm_source": (m.get("data") or {}).get("train_norm_source"),
        "cache_fingerprints": (m.get("data") or {}).get("cache_fingerprints"),
        "row_sha256": (m.get("data") or {}).get("row_sha256"),
        "feature_spec_hash": (m.get("data") or {}).get("feature_spec_hash"),
        "labels_in_structure_cache": (m.get("data") or {}).get("labels_in_structure_cache"),
        "n_fit": (m.get("recipe_full") or {}).get("n_fit"),
        "total_updates": (m.get("recipe_full") or {}).get("total_updates"),
        "best_epoch": (summary.get("best_valid") or {}).get("epoch"),
        "grid_best": summary.get("grid_best"),
        "late_window": summary.get("late_window", {}).get("window"),
        "metric_best_valid": {
            split: {
                "rmse": ((summary.get("best_valid") or {}).get(split) or {}).get("rmse"),
                "mae": ((summary.get("best_valid") or {}).get(split) or {}).get("mae"),
            }
            for split in RP.SPLITS
        },
        "metric_last_epoch": {
            split: {
                "rmse": ((summary.get("last_epoch") or {}).get(split) or {}).get("rmse"),
                "mae": ((summary.get("last_epoch") or {}).get(split) or {}).get("mae"),
            }
            for split in RP.SPLITS
        },
        "test_labels_read": m.get("test_labels_read"),
        "test_access": m.get("test_access"),
    }


def _pred_hashes(arm_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"files": {}}
    for tag in ("best", "last"):
        for split in RP.SPLITS:
            p = arm_dir / f"pred_{split}_{tag}.npz"
            out["files"][f"pred_{split}_{tag}"] = {
                "present": p.is_file(),
                "sha256": RP.sha256_file(p) if p.is_file() else None,
            }
    for tag in ("best", "last"):
        p = arm_dir / f"checkpoint_{tag}.pt"
        out["files"][f"checkpoint_{tag}"] = {
            "present": p.is_file(),
            "sha256": RP.sha256_file(p) if p.is_file() else None,
        }
    return out


def _ckpt_meta(arm_dir: Path, tag: str = "best") -> dict[str, Any]:
    p = arm_dir / f"checkpoint_{tag}.pt"
    if not p.is_file():
        # replay arms keep their pulled checkpoint under data/checkpoints/<arm>/
        alt = RP.CKPT_DIR / arm_dir.name.replace("_N100_lipo_", "_") / f"checkpoint_{tag}.pt"
        if alt.is_file():
            p = alt
    if not p.is_file():
        return {"present": False, "note": "checkpoint on res, not pulled (not needed by this round)"}
    import torch

    ck = torch.load(p, map_location="cpu", weights_only=False)
    return {
        "present": True,
        "sha256": RP.sha256_file(p),
        "family": ck.get("family"),
        "checkpoint": ck.get("checkpoint"),
        "best_epoch": ck.get("best_epoch"),
        "state_hash": ck.get("state_hash"),
        "init_state_hash": ck.get("init_state_hash"),
        "parameters": ck.get("parameters"),
        "epoch_plan_sha256": ck.get("epoch_plan_sha256"),
        "target_norm": ck.get("target_norm"),
    }


# ---------------------------------------------------------------------------
# metric recompute from raw predictions
# ---------------------------------------------------------------------------


def _metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = np.asarray(pred, float) - np.asarray(y, float)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "n": int(err.size),
    }


def recompute_arm(arm_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"arm_dir": RP.track_rel(arm_dir)}
    meta = _run_meta(arm_dir)
    for tag in ("best", "last"):
        key = "best_valid" if tag == "best" else "last_epoch"
        ref = meta[f"metric_{key}"]
        for split in RP.SPLITS:
            p = arm_dir / f"pred_{split}_{tag}.npz"
            if not p.is_file():
                out[f"{tag}_{split}"] = {"present": False}
                continue
            d = np.load(p, allow_pickle=False)
            m = _metrics(np.asarray(d["raw"], float), np.asarray(d["y_true"], float))
            ref_rmse = ((ref.get(split) or {}).get("rmse"))
            ref_mae = ((ref.get(split) or {}).get("mae"))
            out[f"{tag}_{split}"] = {
                "present": True,
                "recomputed": m,
                "metrics_json_rmse": ref_rmse,
                "metrics_json_mae": ref_mae,
                "abs_rmse_diff": (abs(m["rmse"] - ref_rmse) if ref_rmse is not None else None),
                "abs_mae_diff": (abs(m["mae"] - ref_mae) if ref_mae is not None else None),
                "match": bool(
                    ref_rmse is not None and ref_mae is not None
                    and abs(m["rmse"] - ref_rmse) <= 1e-6 and abs(m["mae"] - ref_mae) <= 1e-6
                ),
            }
    out["meta"] = meta
    return out


# ---------------------------------------------------------------------------
# data identity
# ---------------------------------------------------------------------------


def data_identity() -> dict[str, Any]:
    import lp_common as LPC
    import lp_data as LPD
    import lp_source as LPS

    view = LPD.build_view()
    split = LPS.load_split()
    lipo = {
        "dataset": LPC.DATASET,
        "split_method": LPC.OGB_SPLIT_METHOD,
        "sizes": {k: int(len(view.rows[k])) for k in RP.SPLITS},
        "official_split_sizes": dict(LPC.SPLIT_SIZES),
        "row_sha256": split["rows_sha256"],
        "source_file_sha256": split["source_file_sha256"],
        "feature_spec_hash": view.spec.hash(),
        "labels_in_structure_cache": bool(view.caches["train"]["meta"].get("labels_present")),
        "target_norm": dict(view.target_norm),
        "train_norm_source": view.train_norm.get("source"),
        "train_norm_fingerprint": __import__("lcrc_data").norm_fingerprint(view.train_norm),
        "cache_fingerprints": {k: view.caches[k]["meta"]["content_fingerprint"] for k in RP.SPLITS},
        "row_sets_disjoint": bool(
            not (set(view.rows["train"].tolist()) & set(view.rows["valid"].tolist()))
            and not (set(view.rows["train"].tolist()) & set(view.rows["test"].tolist()))
            and not (set(view.rows["valid"].tolist()) & set(view.rows["test"].tolist()))
        ),
        "coverage": sorted(
            set(view.rows["train"].tolist()) | set(view.rows["valid"].tolist()) | set(view.rows["test"].tolist())
        ) == list(range(LPC.N_MOLECULES)),
    }
    # global row-id alignment: model pred rows == official split rows
    align: dict[str, Any] = {}
    for name, arm_dir in {**RP.LIPO_ARMS}.items():
        p = arm_dir / "pred_valid_best.npz"
        if not p.is_file():
            continue
        d = np.load(p, allow_pickle=False)
        rows = np.asarray(d["rows"]).tolist()
        y = np.asarray(d["y_true"], float)
        ok_rows = rows == view.rows["valid"].tolist()
        ok_y = bool(np.max(np.abs(y - view.y["valid"])) <= 1e-9) if len(y) == len(view.y["valid"]) else False
        align[name] = {"rows_match_official_valid": ok_rows, "labels_match": ok_y, "n": len(rows)}
    return {"lipo": lipo, "row_alignment": align}


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def build_manifest() -> dict[str, Any]:
    arms: dict[str, Any] = {}
    recomputed: dict[str, Any] = {}
    for name, d in RP.LIPO_ARMS.items():
        if not (d / "metrics.json").is_file():
            arms[name] = {"present": False, "arm_dir": RP.track_rel(d)}
            continue
        arms[name] = {"present": True, **_run_meta(d), "checkpoint_best": _ckpt_meta(d, "best"),
                      "pred_hashes": _pred_hashes(d)}
        recomputed[name] = recompute_arm(d)
    esol: dict[str, Any] = {}
    for name, d in RP.ESOL_ARMS.items():
        if not (d / "metrics.json").is_file():
            esol[name] = {"present": False, "arm_dir": RP.track_rel(d)}
            continue
        esol[name] = {"present": True, **_run_meta(d), "checkpoint_best": _ckpt_meta(d, "best")}
    payload = {
        "schema": "lcrc-semantic-analysis-repair-source-manifest-v1",
        "protocol_id": RP.PROTOCOL_ID,
        "created_utc": RP.utcnow(),
        "git_commit": RP.git_commit(),
        "code_hashes": RP.code_hashes(),
        "design_basis": {
            "predecessor_round": "lcrc-explicit-semantic-learning-8h-s0-v1",
            "predecessor_commit": "1d6700ea01acbee33e7630f1813c7884d28bee71",
            "note": "remote checkout on res was at ancestor 98da0b75 when this round started",
        },
        "data_identity": data_identity(),
        "lipo_arms": arms,
        "esol_arms": esol,
        "recomputed_metrics": recomputed,
    }
    return payload


def run() -> dict[str, Any]:
    payload = build_manifest()
    RP.write_json(RP.ANALYSIS / "source_manifest.json", payload)
    # a compact metric-recompute table so the audit is readable without the hash wall
    rows = []
    for name, rec in payload["recomputed_metrics"].items():
        for tag in ("best", "last"):
            for split in RP.SPLITS:
                e = rec.get(f"{tag}_{split}", {})
                if e.get("present"):
                    rows.append(
                        {
                            "arm": name,
                            "tag": tag,
                            "split": split,
                            "rmse_recomputed": e["recomputed"]["rmse"],
                            "mae_recomputed": e["recomputed"]["mae"],
                            "rmse_metrics_json": e["metrics_json_rmse"],
                            "mae_metrics_json": e["metrics_json_mae"],
                            "match": e["match"],
                        }
                    )
    import csv

    out = RP.ANALYSIS / "metric_recompute.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["arm"])
        w.writeheader()
        w.writerows(rows)
    n_match = sum(1 for r in rows if r["match"])
    return {
        "schema": payload["schema"],
        "n_arms": len(payload["lipo_arms"]),
        "n_metric_rows": len(rows),
        "n_metric_rows_match": n_match,
        "artifacts": [RP.track_rel(RP.ANALYSIS / "source_manifest.json"), RP.track_rel(out)],
    }


__all__ = [name for name in dir() if not name.startswith("_")]
