#!/usr/bin/env python
"""B4: one-time AuditEval label read + scoring for a frozen B3 seed.

Checks the B3 freeze hashes first.  Official valid and test are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

_HERE = Path(__file__).resolve().parent
_TRACK = _HERE.parent
_REPO = _TRACK.parents[2]
_V1 = _REPO / "tracks" / "structural-sufficiency" / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import audit_common as AC  # noqa: E402

BOOT_N = 10000
BOOT_SEED = 20261026
STRAT_BOOT_N = 2000
TAIL = -3.0


def sha_file(path):
    return C.sha256_file(path)


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def medae(y, p):
    return float(np.median(np.abs(np.asarray(y) - np.asarray(p))))


def boot_ci(values, n_boot=STRAT_BOOT_N, seed=BOOT_SEED):
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return {"n": int(len(v)), "mean": float(v.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def worst_rows(y, p, rows, k=10):
    err = np.abs(np.asarray(y) - np.asarray(p))
    order = np.argsort(err)[::-1]
    top = order[:k]
    n = len(err)
    total = float(err.sum())
    return {
        "n": int(n),
        "top_rows": [int(rows[i]) for i in top],
        "top_errors": [float(err[i]) for i in top],
        "top_contribution_to_mae": float(err[top].sum() / n),
        "mae_without_top": float((total - err[top].sum()) / max(n - k, 1)),
        "mae_all": float(total / n),
    }


def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return {"pearson": 0.0, "spearman": 0.0}
    return {"pearson": float(pearsonr(a, b).statistic),
            "spearman": float(spearmanr(a, b).statistic)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--freeze-dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    t0 = time.time()
    freeze_dir = Path(args.freeze_dir)
    out = Path(args.out) if args.out else freeze_dir / "metrics.json"
    if out.exists() and not args.force:
        print(f"refusing second label read: {out} exists (use --force only to repair an implementation error)")
        return 2
    fz = json.loads((freeze_dir / "freeze.json").read_text())
    if int(fz["seed"]) != int(args.seed):
        raise RuntimeError("freeze seed mismatch")

    # ---- verify every frozen hash before touching labels -----------------
    checks = {}
    for name, rec in fz["predictions"].items():
        path = freeze_dir / f"pred_{name}.npy"
        checks[f"pred_{name}"] = {"exists": path.exists(),
                                  "sha256": sha_file(path) if path.exists() else None,
                                  "expected": rec, "match": path.exists() and sha_file(path) == rec}
    ckpt_path = Path(fz["checkpoint"]["checkpoint"])
    if not ckpt_path.exists():
        ckpt_path = _REPO / ckpt_path
    checks["checkpoint"] = {"path": str(ckpt_path),
                            "sha256": sha_file(ckpt_path),
                            "expected": fz["checkpoint"]["checkpoint_sha256"],
                            "match": sha_file(ckpt_path) == fz["checkpoint"]["checkpoint_sha256"]}
    support_path = Path(fz["support"]["path"])
    if not support_path.exists():
        support_path = _REPO / support_path
    checks["support"] = {"sha256": sha_file(support_path),
                         "expected": fz["support"]["sha256"],
                         "match": sha_file(support_path) == fz["support"]["sha256"]}
    for name, rec in checks.items():
        if not rec["match"]:
            raise RuntimeError(f"freeze hash mismatch: {name}: {rec}")

    # ---- ONE AuditEval label read ----------------------------------------
    # Support rows are the frozen AuditEval order used by B3 predictions.
    sup = np.load(fz["support"]["path"], allow_pickle=False)
    audit_rows = np.asarray(sup["row"], dtype=np.int64)
    cache = D.load_cache("train")
    labels = np.asarray(cache["labels"], dtype=np.float64)
    y = labels[audit_rows]
    label_access = {
        "read_once_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows_sha256": hashlib.sha256(np.ascontiguousarray(audit_rows).tobytes()).hexdigest(),
        "n": int(len(y)),
        "source": "official ZINC train, AuditEval rows only",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }

    preds = {}
    for name in ("raw_audit", "Z_audit", "ZJ_audit", "ZN_audit",
                 "raw_probe", "Z_probe", "ZJ_probe", "ZN_probe"):
        preds[name] = np.load(freeze_dir / f"pred_{name}.npy")
    for arm in ("Z", "ZJ", "ZN"):
        hp = freeze_dir / f"huber_{arm}_audit.npy"
        preds[f"huber_{arm}_audit"] = np.load(hp) if hp.exists() else None

    arms = ("raw", "Z", "ZJ", "ZN", "huber_Z", "huber_ZJ", "huber_ZN")
    metrics = {"seed": int(args.seed), "label_access": label_access,
               "n_audit": int(len(y)), "arms": {}}
    for arm in arms:
        pp = preds.get(f"{arm}_audit")
        if pp is None:
            continue
        metrics["arms"][arm] = {
            "mae": mae(y, pp), "rmse": rmse(y, pp), "median_ae": medae(y, pp),
            "prediction_sha256": hashlib.sha256(np.ascontiguousarray(pp).tobytes()).hexdigest(),
        }
    metrics["gains"] = {
        "Gain_J": metrics["arms"]["Z"]["mae"] - metrics["arms"]["ZJ"]["mae"],
        "Gain_vs_N": metrics["arms"]["ZN"]["mae"] - metrics["arms"]["ZJ"]["mae"],
        "Gain_Z_over_raw": metrics["arms"]["raw"]["mae"] - metrics["arms"]["Z"]["mae"],
        "Gain_J_over_raw": metrics["arms"]["raw"]["mae"] - metrics["arms"]["ZJ"]["mae"],
        "Gain_N_over_raw": metrics["arms"]["raw"]["mae"] - metrics["arms"]["ZN"]["mae"],
        "Gain_huber_J": (metrics["arms"].get("huber_Z", {}).get("mae", float("nan"))
                         - metrics["arms"].get("huber_ZJ", {}).get("mae", float("nan"))),
        "Gain_huber_vs_N": (metrics["arms"].get("huber_ZN", {}).get("mae", float("nan"))
                            - metrics["arms"].get("huber_ZJ", {}).get("mae", float("nan"))),
    }

    # ---- paired bootstrap -------------------------------------------------
    n = len(y)
    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = rng.integers(0, n, size=(BOOT_N, n))
    def boot_gain(a, b):
        ea = np.abs(y - preds[f"{a}_audit"])
        eb = np.abs(y - preds[f"{b}_audit"])
        g = eb[boot_idx].mean(axis=1) - ea[boot_idx].mean(axis=1)
        return {"mean": float(g.mean()), "ci95": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))],
                "fraction_positive": float(np.mean(g > 0))}
    metrics["paired_bootstrap"] = {
        "n": BOOT_N, "seed": BOOT_SEED, "unit": "AuditEval molecule",
        "Gain_J": boot_gain("ZJ", "Z"),
        "Gain_vs_N": boot_gain("ZJ", "ZN"),
        "Gain_Z_over_raw": boot_gain("Z", "raw"),
        "Gain_J_over_raw": boot_gain("ZJ", "raw"),
    }

    # ---- worst rows and removal ------------------------------------------
    for arm in ("raw", "Z", "ZJ", "ZN"):
        metrics.setdefault("worst", {})[arm] = worst_rows(y, preds[f"{arm}_audit"], audit_rows)
    gain_rows = np.abs(y - preds["Z_audit"]) - np.abs(y - preds["ZJ_audit"])
    order = np.argsort(gain_rows)[::-1]
    top10 = order[:10]
    metrics["gain_worst_rows"] = {
        "top_rows": [int(audit_rows[i]) for i in top10],
        "top_gains": [float(gain_rows[i]) for i in top10],
        "top10_gain_contribution": float(gain_rows[top10].sum() / n),
        "gain_without_top10": float(gain_rows[order[10:]].mean()),
        "gain_all": float(gain_rows.mean()),
    }

    # ---- train-fit vs audit gap ------------------------------------------
    # ProbeFit labels are already in the train cache; this is not AuditEval.
    split = json.loads((_TRACK / "configs" / "split_fit8k_probe_audit.json").read_text())
    probe_rows = np.asarray(split["rows"]["probe"], dtype=np.int64)
    y_probe = labels[probe_rows]
    metrics["train_fit_vs_audit"] = {}
    for arm in ("raw", "Z", "ZJ", "ZN"):
        pp = preds[f"{arm}_probe"]
        metrics["train_fit_vs_audit"][arm] = {
            "probe_mae": mae(y_probe, pp),
            "audit_mae": metrics["arms"][arm]["mae"],
            "gap": metrics["arms"][arm]["mae"] - mae(y_probe, pp),
        }

    # ---- correlations -----------------------------------------------------
    metrics["correlations"] = {}
    for a, b in (("Z", "ZJ"), ("ZN", "ZJ"), ("Z", "ZN"), ("raw", "Z"), ("raw", "ZJ")):
        metrics["correlations"][f"{a}_vs_{b}_prediction"] = corr(preds[f"{a}_audit"], preds[f"{b}_audit"])
        ra = y - preds[f"{a}_audit"]
        rb = y - preds[f"{b}_audit"]
        metrics["correlations"][f"{a}_vs_{b}_residual"] = corr(ra, rb)

    # ---- strata (labels only for tail; support/size/ring are label-free) -
    ap = np.asarray(cache["atom_ptr"])
    g = np.asarray(cache["global_feat"])
    support_class = np.asarray(sup["class"])
    n_atoms = ap[audit_rows + 1] - ap[audit_rows]
    rings = g[audit_rows, 6]
    strata = {
        "support": support_class,
        "tail": np.where(y <= TAIL, "tail_y_le_-3", "ordinary_y_gt_-3"),
        "size": np.asarray([f"size_{b}" for b in np.digitize(n_atoms, [21, 25, 29])]),
        "ring_rich": np.where(rings >= 4, "ring_rich_ge4", "ring_lt4"),
    }
    metrics["strata"] = {}
    for sname, sval in strata.items():
        metrics["strata"][sname] = {}
        for level in sorted(set(sval.tolist())):
            mask = sval == level
            if mask.sum() == 0:
                continue
            rec = {"n": int(mask.sum())}
            for arm in ("raw", "Z", "ZJ", "ZN"):
                rec[arm] = boot_ci(np.abs(y[mask] - preds[f"{arm}_audit"][mask]))
            rec["Gain_J"] = boot_ci(np.abs(y[mask] - preds["Z_audit"][mask])
                                    - np.abs(y[mask] - preds["ZJ_audit"][mask]))
            rec["Gain_vs_N"] = boot_ci(np.abs(y[mask] - preds["ZN_audit"][mask])
                                       - np.abs(y[mask] - preds["ZJ_audit"][mask]))
            rec["single_molecule_dominated"] = bool(mask.sum() <= 5)
            metrics["strata"][sname][str(level)] = rec
    metrics["cycle_penalty_stratum"] = {"available": False,
                                         "reason": "no reproducible label-side cycle-penalty reconstruction in this round; omitted rather than guessed"}

    metrics["checks"] = checks
    metrics["runtime_seconds"] = float(time.time() - t0)
    out.write_text(json.dumps(AC.jsonable(metrics), indent=2))
    print(json.dumps({"seed": args.seed, "gains": metrics["gains"],
                      "bootstrap": metrics["paired_bootstrap"],
                      "train_fit_vs_audit": metrics["train_fit_vs_audit"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
