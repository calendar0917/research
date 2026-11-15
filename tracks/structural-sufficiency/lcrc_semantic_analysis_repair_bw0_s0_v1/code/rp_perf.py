"""Two complete performance tables + initialisation x weight-decay comparisons.

Table **A** = the old protocol's common 50-point update grid (``grid.json``
selection, Val-min / earlier-on-tie).  Table **B** = every-epoch best-Valid
selection (``metrics.json`` ``summary.best_valid``), used for comparison with the
existing B arm, the checkpoints and the edit results.  Both are printed in full;
no cherry-picking across tables.  Real ``last`` and the old late window (the
actual final 20 epochs, not "last 5%") are reported as secondary columns.

Comparisons inside one recipe measure the **initialisation** effect
(B_R vs LS_R, B_W0 vs LS_W0, B_Rwd vs LS_Rwd); comparisons inside one
initialisation measure the **recipe / weight-decay** effect.  Uncertainty is a
500-replicate common Murcko-scaffold ``Exp(1)`` perturbation (no-ring fallback to
the canonical molecule itself, seed 20261115), applied with the *same* weights
to every model after row_id alignment; it is fixed-state sample perturbation and
says nothing about training variance.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

import rp_common as RP

PAIRS_INIT = (
    ("B_R", "LS_R", "R"),
    ("B_W0", "LS_W0", "W0"),
    ("B_Rwd", "LS_Rwd", "Rwd"),
)
GROUPS_BY_INIT = {
    "B": ("B_R", "B_W0", "B_Rwd"),
    "LS": ("LS_R", "LS_W0", "LS_Rwd"),
    "FS": ("FS_R", "FS_W0", "FS_Rwd"),
}


# ---------------------------------------------------------------------------
# per-arm tables
# ---------------------------------------------------------------------------


def _arm(name: str) -> Path | None:
    if name == "B_W0":
        return RP.BW0_DIR if (RP.BW0_DIR / "metrics.json").is_file() else None
    return RP.LIPO_ARMS.get(name)


def _grid_best(d: Path) -> dict[str, Any] | None:
    p = d / "grid.json"
    if not p.is_file():
        return None
    g = RP.read_json(p)
    sel = g.get("selection") or {}
    j = sel.get("j")
    pt = next((x for x in g.get("points", []) if x.get("j") == j), None)
    return {
        "j": j,
        "update": sel.get("update"),
        "valid_rmse": sel.get("valid_rmse"),
        "test_rmse": sel.get("test_rmse"),
        "train_rmse": (pt or {}).get("train_rmse"),
        "n_points": len(g.get("points", [])),
        "grid_values": g.get("grid_values"),
    }


def _summary(d: Path) -> dict[str, Any]:
    m = RP.read_json(d / "metrics.json")
    s = m.get("summary", {})
    out: dict[str, Any] = {"recipe": m.get("recipe"), "budget": m.get("budget"), "family": m.get("family")}
    for key, name in (("best_valid", "best"), ("last_epoch", "last")):
        blk = s.get(key) or {}
        out[name] = {sp: {"rmse": (blk.get(sp) or {}).get("rmse"), "mae": (blk.get(sp) or {}).get("mae"),
                          "bias": (blk.get(sp) or {}).get("bias")} for sp in RP.SPLITS}
        out[name]["epoch"] = blk.get("epoch")
    lw = s.get("late_window") or {}
    out["late_window"] = {
        "window": lw.get("window"),
        "epochs": lw.get("epochs"),
        "splits": {sp: {"rmse": lw.get(f"{sp}_rmse"), "mae": lw.get(f"{sp}_mae")} for sp in RP.SPLITS},
    }
    out["anomalies"] = m.get("anomalies")
    return out


def table_a_and_b() -> dict[str, Any]:
    arms = {}
    for name in list(RP.LIPO_ARMS) + ["B_W0"]:
        d = _arm(name)
        if d is None:
            continue
        arms[name] = {"arm_dir": RP.track_rel(d), "tableA_grid": _grid_best(d), "tableB_best": _summary(d)}
    return arms


# ---------------------------------------------------------------------------
# predictions for paired bootstrap
# ---------------------------------------------------------------------------


def _test_rows_and_preds(name: str):
    d = _arm(name)
    if d is None:
        return None
    p = d / "pred_test_best.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=False)
    return {
        "rows": np.asarray(z["rows"]).tolist(),
        "raw": np.asarray(z["raw"], dtype=np.float64),
        "y": np.asarray(z["y_true"], dtype=np.float64),
    }


def _val_rows_and_preds(name: str):
    d = _arm(name)
    if d is None:
        return None
    p = d / "pred_valid_best.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=False)
    return {"rows": np.asarray(z["rows"]).tolist(), "raw": np.asarray(z["raw"], dtype=np.float64),
            "y": np.asarray(z["y_true"], dtype=np.float64)}


def _murcko_groups(rows: list[int]) -> np.ndarray:
    """Scaffold group id per row (no-ring fallback = the canonical molecule)."""
    import lp_source as LPS
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    raw = LPS.load_raw_artifact()
    groups: dict[str, int] = {}
    out = np.empty(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        smi = str(raw[int(r)]["canonical_smiles"])
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            key = smi
        else:
            try:
                scaf = MurckoScaffold.GetScaffoldForMol(mol)
                key = Chem.MolToSmiles(scaf) if scaf is not None and scaf.GetNumAtoms() > 0 else Chem.MolToSmiles(mol)
            except Exception:  # noqa: BLE001
                key = Chem.MolToSmiles(mol)
        out[i] = groups.setdefault(key, len(groups))
    return out


def _wrmse(pred: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    e2 = (pred - y) ** 2
    return float(math.sqrt(float(np.sum(w * e2) / np.sum(w))))


def bootstrap_pair_diff(name_a: str, name_b: str, *, split: str = "test") -> dict[str, Any]:
    getter = _test_rows_and_preds if split == "test" else _val_rows_and_preds
    A = getter(name_a)
    B = getter(name_b)
    if A is None or B is None:
        return {"status": "MISSING", "a": name_a, "b": name_b, "split": split}
    if A["rows"] != B["rows"]:
        return {"status": "ROW_MISMATCH", "a": name_a, "b": name_b}
    rows = A["rows"]
    groups = _murcko_groups(rows)
    y = A["y"]
    if not np.array_equal(y, B["y"]):
        return {"status": "LABEL_MISMATCH", "a": name_a, "b": name_b}
    n_groups = int(groups.max()) + 1
    rng = np.random.default_rng(RP.BOOTSTRAP_SEED)
    W = rng.exponential(1.0, size=(RP.BOOTSTRAP_REPLICATES, n_groups))
    w_rows = W[:, groups]
    ea2 = (A["raw"] - y) ** 2
    eb2 = (B["raw"] - y) ** 2
    d_reps = np.sqrt((w_rows * ea2).sum(1) / w_rows.sum(1)) - np.sqrt((w_rows * eb2).sum(1) / w_rows.sum(1))
    return {
        "status": "OK",
        "a": name_a,
        "b": name_b,
        "split": split,
        "n": int(len(rows)),
        "n_scaffold_groups": n_groups,
        "rmse_a": _wrmse(A["raw"], y, np.ones_like(y)),
        "rmse_b": _wrmse(B["raw"], y, np.ones_like(y)),
        "delta_rmse": float(_wrmse(A["raw"], y, np.ones_like(y)) - _wrmse(B["raw"], y, np.ones_like(y))),
        "delta_rmse_boot_mean": float(d_reps.mean()),
        "ci95": [float(np.percentile(d_reps, 2.5)), float(np.percentile(d_reps, 97.5))],
        "crosses_zero": bool(np.percentile(d_reps, 2.5) <= 0.0 <= np.percentile(d_reps, 97.5)),
        "replicates": RP.BOOTSTRAP_REPLICATES,
        "seed": RP.BOOTSTRAP_SEED,
    }


# ---------------------------------------------------------------------------
# comparison builder
# ---------------------------------------------------------------------------


def _row(name: str, table: str) -> dict[str, Any] | None:
    d = _arm(name)
    if d is None:
        return None
    if table == "A":
        g = _grid_best(d)
        if not g:
            return None
        return {"arm": name, "selection": f"grid j={g['j']}", "valid_rmse": g["valid_rmse"],
                "test_rmse": g["test_rmse"], "train_rmse": g["train_rmse"]}
    s = _summary(d)
    return {"arm": name, "selection": f"best-epoch {s['best'].get('epoch')}",
            "valid_rmse": s["best"]["valid"]["rmse"], "test_rmse": s["best"]["test"]["rmse"],
            "train_rmse": s["best"]["train"]["rmse"]}


def _cmp(a: str, b: str, table: str) -> dict[str, Any] | None:
    ra, rb = _row(a, table), _row(b, table)
    if ra is None or rb is None:
        return None
    out = {"table": table, "a": a, "b": b}
    for sp in ("valid", "test"):
        ka, kb = f"{sp}_rmse", f"{sp}_rmse"
        va, vb = ra[ka], rb[kb]
        out[sp] = {
            "a": va, "b": vb,
            "diff_a_minus_b": (va - vb) if (va is not None and vb is not None) else None,
            "rel_pct": (100.0 * (va - vb) / vb) if (va is not None and vb not in (None, 0)) else None,
        }
    return out


def run() -> dict[str, Any]:
    tables = table_a_and_b()

    comparisons: dict[str, Any] = {"initialisation": {}, "recipe_within_init": {}}
    for a, b, recipe in PAIRS_INIT:
        for table in ("A", "B"):
            c = _cmp(a, b, table)
            comparisons["initialisation"].setdefault(f"{recipe}_{table}", c)
    for init, names in GROUPS_BY_INIT.items():
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                for table in ("A", "B"):
                    c = _cmp(a, b, table)
                    comparisons["recipe_within_init"].setdefault(f"{init}:{a}-{b}:{table}", c)

    boot: dict[str, Any] = {}
    boot_pairs = [
        ("B_R", "LS_R"), ("B_W0", "LS_W0"), ("B_Rwd", "LS_Rwd"),
        ("B_R", "B_W0"), ("B_R", "B_Rwd"), ("B_W0", "B_Rwd"),
        ("LS_R", "LS_W0"), ("LS_R", "LS_Rwd"), ("LS_W0", "LS_Rwd"),
        ("LS_R", "FS_R"), ("LS_W0", "FS_W0"),
    ]
    for a, b in boot_pairs:
        for split in ("valid", "test"):
            boot[f"{a}-{b}:{split}"] = bootstrap_pair_diff(a, b, split=split)

    # explicit CPU coordinate (corrected)
    exp_path = RP.ANALYSIS / "explicit_baseline_corrected.json"
    explicit = RP.read_json(exp_path) if exp_path.is_file() else None

    payload = {
        "schema": "lcrc-semantic-analysis-repair-performance-v1",
        "protocol_id": RP.PROTOCOL_ID,
        "created_utc": RP.utcnow(),
        "note": (
            "Table A = old 50-point update grid (validation-min, earlier on tie); Table B = every-epoch "
            "best-Valid. Both printed in full. Late window in the old code is the final 20 epochs."
        ),
        "tables": tables,
        "comparisons": comparisons,
        "bootstrap": boot,
        "explicit_reference": explicit,
    }
    RP.write_json(RP.ANALYSIS / "performance_tables.json", payload)

    rows = []
    for name, rec in tables.items():
        for table, key in (("A", "tableA_grid"), ("B", "tableB_best")):
            if table == "A":
                g = rec.get(key)
                if not g:
                    continue
                rows.append({"arm": name, "table": "A_grid", "selection": f"j={g['j']}",
                             "valid_rmse": g["valid_rmse"], "test_rmse": g["test_rmse"],
                             "train_rmse": g["train_rmse"], "last_valid_rmse": None, "last_test_rmse": None})
            else:
                s = rec[key]
                rows.append({"arm": name, "table": "B_best", "selection": f"epoch {s['best'].get('epoch')}",
                             "valid_rmse": s["best"]["valid"]["rmse"], "test_rmse": s["best"]["test"]["rmse"],
                             "train_rmse": s["best"]["train"]["rmse"],
                             "last_valid_rmse": s["last"]["valid"]["rmse"],
                             "last_test_rmse": s["last"]["test"]["rmse"]})
    out_csv = RP.ANALYSIS / "performance_tables.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "table", "selection", "valid_rmse", "test_rmse",
                                           "train_rmse", "last_valid_rmse", "last_test_rmse"])
        w.writeheader()
        w.writerows(rows)
    return {
        "n_arms": len(tables),
        "bootstrap_ok": sum(1 for v in boot.values() if v.get("status") == "OK"),
        "bootstrap_total": len(boot),
        "artifacts": [RP.track_rel(RP.ANALYSIS / "performance_tables.json"), RP.track_rel(out_csv)],
    }


__all__ = [name for name in dir() if not name.startswith("_")]
