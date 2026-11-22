"""Whole-molecule versus direct-cut replacement deltas, old and new arms.

Both quantities are always named explicitly:

* ``whole_molecule_delta`` = ŷ(G_b) - ŷ(G_a)
* ``direct_cut_delta``     = f(C,S_b) - f(C,S_a)

Official Test-touching pairs are dropped without reading Test labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cpo_common as C

OLD = {
    "A_TRC_FACTOR_CUTRESP": ("parent", "FACTOR", C.PARENT_ROUND / "results" / "LIPO_FACTOR_CUTRESP_R0", True),
    "B_CSOI_ADD": ("csoi", "CSOI_ADD", C.CSOI_ROUND / "results" / "CSOI_ADD_LIPO_R0", True),
    "C_CSOI_GLOBAL": ("csoi", "CSOI_GLOBAL", C.CSOI_ROUND / "results" / "CSOI_GLOBAL_LIPO_R0", True),
    "D_CSOI_REL_EXTRA": ("csoi", "CSOI_REL", C.CSOI_ROUND / "results" / "CSOI_REL_LIPO_R0", True),
    "E_GINE_RAW_REF": (None, None, C.PARENT_ROUND / "results" / "GINE_RAW_LIPO_R0", False),
}
NEW = {
    "F_GRAPH": ("parent", "FACTOR", "F_GRAPH_LIPO_R0", True),
    "B_GRAPH": ("csoi", "CSOI_ADD", "B_GRAPH_LIPO_R0", True),
    "C_GRAPH": ("csoi", "CSOI_GLOBAL", "C_GRAPH_LIPO_R0", True),
}
SCOPES = ("train_in_sample_pairs", "train_valid_cross_pairs", "valid_only_pairs")


def _scopes(pack: Mapping[str, Any]) -> dict[str, np.ndarray]:
    split = np.asarray(pack["molecule_split"], np.uint8)
    a = np.asarray(pack["pair_mol_a"], np.int64)
    b = np.asarray(pack["pair_mol_b"], np.int64)
    sa, sb = split[a], split[b]
    allowed = (sa != 2) & (sb != 2)
    return {
        "train_in_sample_pairs": allowed & (sa == 0) & (sb == 0),
        "train_valid_cross_pairs": allowed & (((sa == 0) & (sb == 1)) | ((sa == 1) & (sb == 0))),
        "valid_only_pairs": allowed & (sa == 1) & (sb == 1),
        "_excluded_test_touching": ~allowed,
    }


def _pred_mol(pack: Mapping[str, Any], preds: Mapping[str, np.ndarray]) -> np.ndarray:
    if "train_row_id" in preds:
        return C.predictions_by_molecule(pack, preds)
    # legacy GINE prediction file has no row_id: use the official split order
    train = np.asarray(pack["train_idx"], np.int64)
    valid = np.asarray(pack["valid_idx"], np.int64)
    labels = np.asarray(pack["labels"], np.float64)
    out = np.full(int(pack["n_molecules"]), np.nan, np.float64)
    for rows, split in ((train, "train"), (valid, "valid")):
        p = np.asarray(preds[f"{split}_pred"], np.float64)
        y = np.asarray(preds[f"{split}_label"], np.float64)
        if p.size != rows.size or not np.allclose(y, labels[rows]):
            raise ValueError(f"{split} prediction order does not match the official split")
        out[rows] = p
    return out


def _cluster_for(scope: str, mask: np.ndarray, pack: Mapping[str, Any]) -> np.ndarray:
    a = np.asarray(pack["pair_mol_a"], np.int64)
    b = np.asarray(pack["pair_mol_b"], np.int64)
    split = np.asarray(pack["molecule_split"], np.uint8)
    if scope == "train_valid_cross_pairs":
        return np.where(split[a] == 1, a, b)[mask]
    # component clustering for in-sample and Valid-only pairs
    return np.asarray(pack["pair_component"], np.int64)[mask]


def arm_record(name: str, family: str | None, arm: str | None, path: Path,
               has_cut: bool, cfg: Mapping[str, Any], pack: Mapping[str, Any],
               scopes: Mapping[str, np.ndarray], log=print) -> dict[str, Any] | None:
    try:
        preds = C.load_predictions(path, kind="best")
    except FileNotFoundError:
        log(f"  [skip] {name}: no predictions at {path}")
        return None
    pred = _pred_mol(pack, preds)
    labels = np.asarray(pack["labels"], np.float64)
    a = np.asarray(pack["pair_mol_a"], np.int64)
    b = np.asarray(pack["pair_mol_b"], np.int64)
    true = labels[b] - labels[a]
    whole = pred[b] - pred[a]
    direct = None
    adapter = None
    ckpt = None
    if has_cut and family is not None and arm is not None:
        model, ckpt = C.load_checkpoint(family, cfg, pack, arm, path, device="cpu")
        adapter = C.CutAdapter(model, family, pack, device="cpu")
        adapter.prepare()
        n = a.size
        direct = np.concatenate([
            adapter.direct_pair_delta(np.arange(lo, min(lo + 512, n), dtype=np.int64))
            for lo in range(0, n, 512)
        ]) * float(ckpt["scaler"]["std"])
    out: dict[str, Any] = {
        "run_dir": str(path),
        "family": family, "arm": arm,
        "best_epoch": int(ckpt["epoch"]) if ckpt is not None else None,
        "definitions": {
            "whole_molecule_delta": "ŷ(G_b) - ŷ(G_a), ŷ = best-checkpoint molecule prediction",
            "direct_cut_delta": "f(C,S_b) - f(C,S_a), best-checkpoint cut response, scaled to label units",
        },
        "scopes": {},
    }
    for scope in SCOPES:
        mask = scopes[scope]
        if not mask.any():
            out["scopes"][scope] = {"n": 0}
            continue
        clusters = _cluster_for(scope, mask, pack)
        rec: dict[str, Any] = {
            "n": int(mask.sum()),
            "n_clusters": int(np.unique(clusters).size),
            "whole_molecule_delta": C.cluster_bootstrap_pair_stats(
                true[mask], whole[mask], clusters, n_boot=2000, seed=0),
        }
        if direct is not None:
            rec["direct_cut_delta"] = C.cluster_bootstrap_pair_stats(
                true[mask], direct[mask], clusters, n_boot=2000, seed=0)
            rec["difference_between_definitions"] = {
                "mean_abs_delta_pred_difference": float(np.mean(np.abs(direct[mask] - whole[mask]))),
                "corr_direct_vs_whole_delta_pred": C.pearson(direct[mask], whole[mask]),
            }
        out["scopes"][scope] = rec
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lipo")
    args = ap.parse_args(argv)
    cfg = C.load_config()
    pack = C.load_pack(cfg, args.dataset)
    scopes = _scopes(pack)
    out: dict[str, Any] = {
        "protocol": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "cache_sha256": pack["cache_sha256"],
        "cache_content_sha256": pack.get("content_sha256"),
        "test_policy": "pairs touching official Test are dropped; their labels are never read",
        "pair_directory": {
            "n_pairs_total": int(np.asarray(pack["pair_mol_a"]).size),
            "scopes": {k: int(v.sum()) for k, v in scopes.items() if not k.startswith("_")},
            "excluded_test_touching": int(scopes["_excluded_test_touching"].sum()),
        },
        "cluster_definition": {
            "train_in_sample_pairs": "pair component (in-sample; optimistic for every arm)",
            "train_valid_cross_pairs": "the Valid molecule of the crossing pair",
            "valid_only_pairs": "pair component; n is tiny, descriptive only",
        },
        "arms": {},
    }
    for name, (family, arm, path, has_cut) in OLD.items():
        rec = arm_record(name, family, arm, path, has_cut, cfg, pack, scopes)
        if rec is not None:
            out["arms"][name] = rec
    for name, (family, arm, run_id, has_cut) in NEW.items():
        path = C.run_dir(run_id)
        rec = arm_record(name, family, arm, path, has_cut, cfg, pack, scopes)
        if rec is not None:
            out["arms"][name] = rec
    out_path = C.analysis_dir() / "pair_delta_comparison.json"
    C.write_json(out_path, out)
    print(f"wrote {out_path}")
    for name, rec in out["arms"].items():
        row = []
        for scope in SCOPES:
            s = rec["scopes"].get(scope, {})
            if s.get("n"):
                w = s["whole_molecule_delta"]
                d = s.get("direct_cut_delta", {})
                row.append(f"{scope}: n={s['n']} whole_mae={w['delta_mae']:.3f}"
                           + (f" direct_mae={d['delta_mae']:.3f}" if d else ""))
        print(name, " | ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())