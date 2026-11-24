"""Design matrices, feature groups and the exact additive ledger for HSRF v0."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import hrf_common as C


class Stage0:
    def __init__(self, path: Path | str | None = None):
        p = Path(path) if path is not None else C.DATA_DIR / "hsrf_stage0.npz"
        self.path = p
        self.z = dict(np.load(p, allow_pickle=False))
        self.catalog = C.read_json(C.DATA_DIR / "feature_catalog.json")
        self.n = int(self.z["labels"].size)
        self.rows = {s: np.asarray(self.z[f"rows_{s}"], np.int64)
                     for s in ("train", "valid", "test")}
        self.y = np.asarray(self.z["labels"], np.float64)
        self.train = self.rows["train"]
        self.scaler = C.target_scaler(self.y, self.train)
        self.y_std = (self.y - self.scaler["mean"]) / self.scaler["std"]
        self.base_basis = np.asarray(self.z["base_basis"], np.float64)
        self.base_scale = np.asarray(self.z["base_scale"], np.float64)
        self.base_raw_col = np.asarray(self.z["base_basis_raw_col"], np.int64)
        self.base_knot = np.asarray(self.z["base_basis_knot"], np.int64)
        self.base_lin = np.asarray(self.z["base_lin_index"], np.int64)
        self.base_counts = np.asarray(self.z["base_counts"], np.float64)
        self.rel_counts = np.asarray(self.z["rel_counts"], np.float64)
        self.rel_basis = np.asarray(self.z["rel_basis"], np.float64)
        self.rel_scale = np.asarray(self.z["rel_scale"], np.float64)
        self.rel_raw_col = np.asarray(self.z["rel_basis_raw_col"], np.int64)
        self.rel_knot = np.asarray(self.z["rel_basis_knot"], np.int64)
        self.rel_lin = np.asarray(self.z["rel_lin_index"], np.int64)
        self.null_basis = np.asarray(self.z["null_basis"], np.float64)
        self.null_scale_self = np.asarray(self.z["null_scale_self"], np.float64)
        self.base_raw_meta = self.catalog["base_raw_columns"]
        self.rel_meta = self.catalog["relation_columns"]
        self.base_basis_meta = self.catalog["base_basis"]["columns"]
        self.rel_basis_meta = self.catalog["relation_basis"]["columns"]

    def design(self, arm: str) -> dict[str, Any]:
        """Return X (float64), feature groups and block annotations for one arm."""
        base_all = self.base_basis / self.base_scale[None, :]
        rel_all = self.rel_basis / self.rel_scale[None, :]
        null_all = self.null_basis / self.rel_scale[None, :]
        null_self = self.null_basis / self.null_scale_self[None, :]
        pB = int(self.base_basis.shape[1])
        pR = int(self.rel_basis.shape[1])
        if arm == "F_BASE":
            X = base_all
            groups = self._base_groups(np.arange(pB), np.arange(pB))
        elif arm == "L_BASE":
            sel = self.base_lin
            X = base_all[:, sel]
            groups = self._base_groups(np.arange(sel.size), sel)
        elif arm == "F_REAL":
            X = np.concatenate([base_all, rel_all], axis=1)
            groups = (self._base_groups(np.arange(pB), np.arange(pB))
                      + self._rel_groups(pB + np.arange(pR), np.arange(pR)))
        elif arm == "L_REAL":
            bsel = self.base_lin
            rsel = self.rel_lin
            X = np.concatenate([base_all[:, bsel], rel_all[:, rsel]], axis=1)
            groups = (self._base_groups(np.arange(bsel.size), bsel)
                      + self._rel_groups(bsel.size + np.arange(rsel.size), rsel))
        elif arm == "F_NULL":
            X = np.concatenate([base_all, null_all], axis=1)
            groups = (self._base_groups(np.arange(pB), np.arange(pB))
                      + self._rel_groups(pB + np.arange(pR), np.arange(pR),
                                         block="relation_null"))
        elif arm == "F_NULL_SELF":
            X = np.concatenate([base_all, null_self], axis=1)
            groups = (self._base_groups(np.arange(pB), np.arange(pB))
                      + self._rel_groups(pB + np.arange(pR), np.arange(pR),
                                         block="relation_null_self"))
        else:
            raise KeyError(arm)
        return {"arm": arm, "X": np.ascontiguousarray(X), "groups": groups,
                "n_features": int(X.shape[1])}

    def _base_groups(self, dcols: np.ndarray, bidx: np.ndarray) -> list[dict[str, Any]]:
        dcols = np.asarray(dcols, np.int64)
        bidx = np.asarray(bidx, np.int64)
        raw = self.base_raw_col[bidx]
        out = []
        for rc in np.unique(raw):
            mask = raw == rc
            local_d = dcols[mask]
            local_b = bidx[mask]
            start = int(local_d.min()); stop = int(local_d.max()) + 1
            meta = self.base_raw_meta[int(rc)]
            out.append({
                "group_id": f"base:{int(rc)}",
                "block": "base",
                "kind": meta["kind"],
                "raw_col": int(rc),
                "name": meta["name"],
                "cols": np.arange(start, stop, dtype=np.int64),
                "basis_cols": local_b,
                "knots": [int(self.base_knot[i]) for i in local_b],
                "scales": [float(self.base_scale[i]) for i in local_b],
                "meta": meta,
            })
        return out

    def _rel_groups(self, dcols: np.ndarray, bidx: np.ndarray,
                    block: str = "relation") -> list[dict[str, Any]]:
        dcols = np.asarray(dcols, np.int64)
        bidx = np.asarray(bidx, np.int64)
        raw = self.rel_raw_col[bidx]
        out = []
        for rc in np.unique(raw):
            mask = raw == rc
            local_d = dcols[mask]
            local_b = bidx[mask]
            start = int(local_d.min()); stop = int(local_d.max()) + 1
            meta = self.rel_meta[int(rc)]
            out.append({
                "group_id": f"{block}:{int(rc)}",
                "block": block,
                "kind": "relation",
                "raw_col": int(rc),
                "name": f"rel_{int(rc)}",
                "relation": meta,
                "cols": np.arange(start, stop, dtype=np.int64),
                "basis_cols": local_b,
                "knots": [int(self.rel_knot[i]) for i in local_b],
                "scales": [float(self.rel_scale[i]) for i in local_b],
                "meta": meta,
            })
        return out


def ledger(X: np.ndarray, beta: np.ndarray, groups: Sequence[Mapping[str, Any]],
           train_mean: np.ndarray, y_mean: float, y_std: float,
           z_mean: float = 0.0) -> dict[str, Any]:
    """Exact additive ledger in original label units.

    ``X`` is the design row (already scaled by Train scales), ``beta`` its
    coefficient.  The intercept absorbs the Train-centring term.
    """
    n = X.shape[0]
    intercept = float(y_mean + y_std * (float(z_mean) - float(train_mean @ beta)))
    main_sum = np.zeros(n, np.float64)
    rel_sum = np.zeros(n, np.float64)
    group_values = np.zeros((n, len(groups)), np.float32)
    for gi, g in enumerate(groups):
        cols = np.asarray(g["cols"], np.int64)
        val = y_std * (X[:, cols] @ beta[cols])
        group_values[:, gi] = val.astype(np.float32)
        if str(g["block"]).startswith("relation"):
            rel_sum += val
        else:
            main_sum += val
    recon = intercept + main_sum + rel_sum
    return {"intercept": intercept, "group_values": group_values,
            "main_effect_sum": main_sum, "relation_effect_sum": rel_sum,
            "reconstructed": recon}


def function_curves(stage: Stage0, arm: str, beta: np.ndarray,
                    groups: Sequence[Mapping[str, Any]],
                    top_groups: Sequence[int], group_values: np.ndarray) -> list[dict[str, Any]]:
    """Raw-count response functions for selected groups, with support counts."""
    out = []
    for gi in top_groups:
        g = groups[int(gi)]
        cols = np.asarray(g["cols"], np.int64)
        bidx = np.asarray(g["basis_cols"], np.int64)
        if g["block"].startswith("relation"):
            raw = stage.rel_counts[:, int(g["raw_col"])]
            knots = [int(stage.rel_knot[i]) for i in bidx]
            scales = [float(stage.rel_scale[i]) for i in bidx]
        else:
            raw = stage.base_counts[:, int(g["raw_col"])]
            knots = [int(stage.base_knot[i]) for i in bidx]
            scales = [float(stage.base_scale[i]) for i in bidx]
        hi = int(np.max(raw[stage.train]))
        support = np.bincount(np.asarray(raw[stage.train], np.int64), minlength=hi + 1)
        support_v = np.bincount(np.asarray(raw[stage.rows["valid"]], np.int64),
                                minlength=hi + 1)
        support_t = np.bincount(np.asarray(raw[stage.rows["test"]], np.int64),
                                minlength=hi + 1)
        values = []
        for x in range(0, hi + 1):
            vals = np.asarray([max(x - k, 0) if k > 0 else x for k in knots], np.float64)
            values.append(float(stage.scaler["std"] * np.dot(vals / np.asarray(scales),
                                                             beta[cols])))
        out.append({
            "group_id": g["group_id"], "block": g["block"], "kind": g["kind"],
            "raw_col": int(g["raw_col"]), "name": g["name"],
            "definition": g.get("meta", {}),
            "knots": knots, "scales": scales,
            "counts": list(range(0, hi + 1)),
            "response": values,
            "train_support": support.tolist(),
            "valid_support": support_v.tolist(),
            "test_support": support_t.tolist(),
            "train_max": int(hi),
            "mean_abs_group_contribution": float(np.mean(np.abs(group_values[:, gi]))),
        })
    return out


def coefficient_table(arm: str, beta: np.ndarray,
                      groups: Sequence[Mapping[str, Any]],
                      base_knot: np.ndarray, rel_knot: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for g in groups:
        cols = np.asarray(g["cols"], np.int64)
        bidx = np.asarray(g["basis_cols"], np.int64)
        for j, c in enumerate(cols):
            knot = int((rel_knot if g["block"].startswith("relation") else base_knot)[bidx[j]])
            rows.append({
                "arm": arm, "group_id": g["group_id"], "block": g["block"],
                "kind": g["kind"], "raw_col": int(g["raw_col"]),
                "name": g["name"], "knot": knot,
                "scale": float(g["scales"][j]),
                "beta_scaled": float(beta[c]),
                "beta_raw": float(beta[c] / g["scales"][j]),
            })
    return rows