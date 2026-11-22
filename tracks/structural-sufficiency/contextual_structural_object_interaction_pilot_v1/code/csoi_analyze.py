"""CSOI-v1 analysis: main table, paired bootstrap, pair metrics, support, cases.

Only Train/Valid prediction arrays are ever touched.  Official Test labels are
never read: the loader whitelists ``train_*`` / ``valid_*`` keys and asserts the
absence of any test access.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import csoi_common as C
import csoi_data as D

ARMS = {
    "A_TRC_FACTOR_CUTRESP": {
        "kind": "parent_reuse",
        "run_dir": C.PARENT_ROUND / "results" / "LIPO_FACTOR_CUTRESP_R0",
        "arm": "FACTOR",
    },
    "B_CSOI_ADD": {"kind": "trained", "run_id": "CSOI_ADD_LIPO_R0", "arm": "CSOI_ADD"},
    "C_CSOI_GLOBAL": {"kind": "trained", "run_id": "CSOI_GLOBAL_LIPO_R0", "arm": "CSOI_GLOBAL"},
    "D_CSOI_REL": {"kind": "trained", "run_id": "CSOI_REL_LIPO_R0", "arm": "CSOI_REL"},
    "E_GINE_RAW_REF": {
        "kind": "parent_reuse",
        "run_dir": C.PARENT_ROUND / "results" / "GINE_RAW_LIPO_R0",
        "arm": "GINE_RAW",
    },
}
PRIMARY = [("B_CSOI_ADD", "C_CSOI_GLOBAL"), ("C_CSOI_GLOBAL", "D_CSOI_REL")]
CALIBRATION = [("A_TRC_FACTOR_CUTRESP", "B_CSOI_ADD")]
REFERENCE = [("E_GINE_RAW_REF", "B_CSOI_ADD"), ("E_GINE_RAW_REF", "C_CSOI_GLOBAL"),
             ("E_GINE_RAW_REF", "D_CSOI_REL"), ("A_TRC_FACTOR_CUTRESP", "C_CSOI_GLOBAL"),
             ("A_TRC_FACTOR_CUTRESP", "D_CSOI_REL")]
ALLOWED = ("train_", "valid_")


def _pred_path(entry: Mapping[str, Any]) -> tuple[Path, str]:
    if entry["kind"] == "trained":
        return C.run_dir(entry["run_id"]) / "predictions_best.npz", entry["run_id"]
    d = Path(entry["run_dir"])
    for name in ("predictions_best.npz", "predictions.npz"):
        if (d / name).is_file():
            return d / name, d.name
    raise FileNotFoundError(f"no predictions in {d}")


def load_arm(entry: Mapping[str, Any], ref: Mapping[str, Any] | None = None) -> dict[str, Any]:
    path, run_id = _pred_path(entry)
    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.keys())
        touched = [k for k in keys if not k.startswith(ALLOWED)]
        if any(k.startswith("test") for k in keys):
            assert all(k.startswith(ALLOWED) for k in keys if not k.startswith("test")), keys
        out = {"run_id": run_id, "path": str(path),
               "keys_used": [k for k in keys if k.startswith(ALLOWED)],
               "test_keys_present_but_unread": [k for k in keys if k.startswith("test")]}
        for split in ("train", "valid"):
            for field in ("label", "pred"):
                out[f"{split}_{field}"] = np.asarray(npz[f"{split}_{field}"], np.float64)
            key = f"{split}_row_id"
            if key in npz:
                out[key] = np.asarray(npz[key], np.float64)
            else:
                if ref is None:
                    raise KeyError(f"{path} has no {key} and no reference order given")
                lab = out[f"{split}_label"]
                if not np.allclose(lab, ref[f"{split}_label"]):
                    raise ValueError(f"{path}: {split} labels do not match the reference order")
                out[key] = ref[key]
                out["row_id_source"] = f"reference order ({ref['run_id']})"
    assert not touched or all(k.startswith("test") for k in touched)
    man = None
    if entry["kind"] == "trained":
        man = C.read_json(C.run_dir(entry["run_id"]) / "manifest.json")
    else:
        mp = Path(entry["run_dir"]) / "manifest.json"
        man = C.read_json(mp) if mp.is_file() else {}
    out["manifest"] = man
    out["label"] = entry["arm"]
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def arm_metrics(a: Mapping[str, Any]) -> dict[str, Any]:
    y = a["valid_label"]
    p = a["valid_pred"]
    yt = a["train_label"]
    pt = a["train_pred"]
    return {
        "valid_rmse": C.rmse(p, y), "valid_mae": C.mae(p, y),
        "train_rmse": C.rmse(pt, yt), "train_mae": C.mae(pt, yt),
        "valid_pearson": C.pearson(p, y),
        "valid_r2": float(1.0 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)),
        "n_valid": int(y.size), "n_train": int(yt.size),
        "best_epoch": (a["manifest"].get("best") or {}).get("epoch"),
        "params": (a["manifest"].get("params") or {}).get("total")
        or (a["manifest"].get("params") or {}).get("trainable"),
        "gpu_hours": a["manifest"].get("gpu_hours"),
        "wall_seconds": a["manifest"].get("wall_seconds"),
    }


def contrast(a: Mapping[str, Any], b: Mapping[str, Any], reps: int, seed: int) -> dict[str, Any]:
    """Paired bootstrap over Valid molecules.  Positive delta = b better than a."""
    ya, pa = a["valid_label"], a["valid_pred"]
    yb, pb = b["valid_label"], b["valid_pred"]
    if not np.allclose(a["valid_row_id"], b["valid_row_id"]):
        raise ValueError("valid row order mismatch")
    ea = (pa - ya) ** 2
    eb = (pb - yb) ** 2
    res = C.paired_bootstrap(ea, eb, reps=reps, seed=seed)
    res.update({
        "arm_a": a["run_id"], "arm_b": b["run_id"],
        "rmse_a": C.rmse(pa, ya), "rmse_b": C.rmse(pb, yb),
        "delta_rmse": C.rmse(pa, ya) - C.rmse(pb, yb),  # >0 => b better
        "delta_mae": C.mae(pa, ya) - C.mae(pb, yb),
    })
    return res


# ---------------------------------------------------------------------------
# pair metrics (Valid molecules only)
# ---------------------------------------------------------------------------


def _pred_by_mol(arm: Mapping[str, Any], pack: Mapping[str, Any]) -> np.ndarray:
    rows = np.asarray(pack["row_ids"], np.int64)
    idx_of_row = {int(r): i for i, r in enumerate(rows.tolist())}
    pred = np.full(int(pack["n_molecules"]), np.nan)
    for rid, val in zip(arm["valid_row_id"].tolist(), arm["valid_pred"].tolist()):
        pred[idx_of_row[int(rid)]] = val
    for rid, val in zip(arm["train_row_id"].tolist(), arm["train_pred"].tolist()):
        pred[idx_of_row[int(rid)]] = val
    return pred


def pair_scopes(pack: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Pair masks over the frozen train-only pair directory, by label availability.

    The directory is train-only by protocol, so most pairs are in-sample for every
    arm.  Only pairs whose molecules lie in {train, valid} are ever used: any pair
    touching the official Test split is dropped, unread.
    """
    labels = np.asarray(pack["labels"], np.float64)
    split = np.asarray(pack["molecule_split"], np.uint8)
    a = np.asarray(pack["pair_mol_a"], np.int64)
    b = np.asarray(pack["pair_mol_b"], np.int64)
    sa, sb = split[a], split[b]
    allowed = (sa != 2) & (sb != 2)
    return {
        "train_in_sample_pairs": allowed & (sa == 0) & (sb == 0),
        "train_valid_cross_pairs": allowed & (((sa == 0) & (sb == 1))
                                              | ((sa == 1) & (sb == 0))),
        "valid_only_pairs": allowed & (sa == 1) & (sb == 1),
        "_excluded_test_touching": ~allowed,
    }


def pair_metrics(pack: Mapping[str, Any], arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    labels = np.asarray(pack["labels"], np.float64)
    a = np.asarray(pack["pair_mol_a"], np.int64)
    b = np.asarray(pack["pair_mol_b"], np.int64)
    transform = np.asarray(pack["pair_transform"], np.int64)
    component = np.asarray(pack["pair_component"], np.int64)
    scopes = pair_scopes(pack)
    out: dict[str, Any] = {
        "n_pairs_total": int(a.size),
        "scopes": {k: int(v.sum()) for k, v in scopes.items()},
        "scope_note": ("the frozen pair directory is train-only, so "
                       "'train_in_sample_pairs' is optimistic for every arm; "
                       "'valid_only_pairs' is fully out-of-sample but tiny (n=15); "
                       "pairs touching Test are excluded and their labels unread"),
        "by_arm": {},
    }
    for name, arm in arms.items():
        if arm.get("valid_pred") is None or arm.get("train_pred") is None:
            continue
        pred = _pred_by_mol(arm, pack)
        rec: dict[str, Any] = {}
        for scope, mask in scopes.items():
            if scope.startswith("_"):
                continue
            if not mask.any():
                rec[scope] = {"n": 0}
                continue
            dl = labels[b[mask]] - labels[a[mask]]
            dp = pred[b[mask]] - pred[a[mask]]
            rec[scope] = _delta_stats(dl, dp)
            tt = transform[mask]
            comp = component[mask]
            by_t: dict[str, Any] = {}
            for t in np.unique(tt).tolist():
                m = tt == t
                if m.sum() >= 10:
                    by_t[f"transform_{int(t)}"] = {"n": int(m.sum()),
                                                   **_delta_stats(dl[m], dp[m])}
            rec[scope]["by_transform"] = by_t
            multi = np.asarray([len(set(comp[tt == t].tolist())) > 1 for t in tt])
            if multi.any():
                rec[scope]["matched_transforms_different_components"] = {
                    "n": int(multi.sum()), **_delta_stats(dl[multi], dp[multi])}
        out["by_arm"][name] = rec
    return out


def _delta_stats(dl: np.ndarray, dp: np.ndarray) -> dict[str, Any]:
    if dl.size == 0:
        return {"n": 0}
    err = dp - dl
    sign = np.sign(dl) * np.sign(dp)
    return {
        "n": int(dl.size),
        "delta_mae": float(np.mean(np.abs(err))),
        "delta_rmse": float(np.sqrt(np.mean(err ** 2))),
        "delta_pearson": C.pearson(dp, dl),
        "direction_accuracy": float(np.mean(sign > 0)),
        "delta_bias": float(np.mean(err)),
        "delta_true_std": float(np.std(dl)),
    }


# ---------------------------------------------------------------------------
# support strata
# ---------------------------------------------------------------------------


def support_report(pack: Mapping[str, Any], arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    support = D.SupportIndex(pack)
    cov = support.coverage()
    obj = support.object_coverage(pack)
    strata = support.coverage()["strata_counts"]
    per_arm: dict[str, Any] = {}
    # per-molecule stratum: majority stratum over the molecule's Valid parses
    valid_parse = support.valid_parse
    st = support.stratum(valid_parse)
    mol_of_parse = support.parse_mol[valid_parse]
    mol_stratum: dict[int, int] = {}
    for m in np.unique(mol_of_parse).tolist():
        votes = st[mol_of_parse == m]
        mol_stratum[int(m)] = int(np.bincount(votes, minlength=4).argmax())
    for name, arm in arms.items():
        if arm.get("valid_pred") is None:
            continue
        rows = np.asarray(pack["row_ids"], np.int64)
        idx_of_row = {int(r): i for i, r in enumerate(rows.tolist())}
        y = arm["valid_label"]
        p = arm["valid_pred"]
        rid = arm["valid_row_id"].astype(np.int64)
        ms = np.asarray([mol_stratum.get(int(r), -1) for r in rid.tolist()])
        entry = {}
        for s in range(4):
            m = ms == s
            if m.any():
                entry[f"stratum_{s}"] = {"n": int(m.sum()), "rmse": C.rmse(p[m], y[m]),
                                         "mae": C.mae(p[m], y[m])}
        entry["all"] = {"n": int(y.size), "rmse": C.rmse(p, y)}
        per_arm[name] = entry
    return {"coverage": cov, "object_coverage": obj,
            "stratum_definition": {
                "0": "seen S, seen C", "1": "seen S, unseen C",
                "2": "unseen S, seen C", "3": "unseen S, unseen C"},
            "n_valid_parses_per_stratum": strata,
            "by_arm": per_arm}


# ---------------------------------------------------------------------------
# case analysis
# ---------------------------------------------------------------------------


def cases(pack: Mapping[str, Any], arms: Mapping[str, Mapping[str, Any]],
          limit: int = 20) -> dict[str, Any]:
    """Auto-selected matched pairs: Valid-only first, then Valid x Train crossings."""
    labels = np.asarray(pack["labels"], np.float64)
    rows = np.asarray(pack["row_ids"], np.int64)
    split = np.asarray(pack["molecule_split"], np.uint8)
    a = np.asarray(pack["pair_mol_a"], np.int64)
    b = np.asarray(pack["pair_mol_b"], np.int64)
    transform = np.asarray(pack["pair_transform"], np.int64)
    component = np.asarray(pack["pair_component"], np.int64)
    scopes = pair_scopes(pack)
    preds = {name: _pred_by_mol(arm, pack) for name, arm in arms.items()
             if arm.get("valid_pred") is not None and arm.get("train_pred") is not None}
    need = ("B_CSOI_ADD", "C_CSOI_GLOBAL", "D_CSOI_REL")
    if not all(k in preds for k in need):
        return {"note": "trained CSOI arms are not available yet", "cases": {}}

    def build(mask: np.ndarray, scope: str) -> list[dict[str, Any]]:
        idx = np.nonzero(mask)[0]
        recs = []
        for i in idx.tolist():
            dl = float(labels[b[i]] - labels[a[i]])
            rec = {"scope": scope, "pair_index": int(i), "transform": int(transform[i]),
                   "component": int(component[i]),
                   "split_a": int(split[a[i]]), "split_b": int(split[b[i]]),
                   "row_id_a": int(rows[a[i]]), "row_id_b": int(rows[b[i]]),
                   "label_a": float(labels[a[i]]), "label_b": float(labels[b[i]]),
                   "delta_true": dl,
                   "S_a": int(np.asarray(pack["pair_Sa"])[i]),
                   "S_b": int(np.asarray(pack["pair_Sb"])[i]),
                   "C": int(np.asarray(pack["pair_C"])[i]),
                   "cut_bond_row": int(np.asarray(pack["pair_cut_bond_row"])[i]),
                   "delta_pred": {}, "abs_err": {}}
            for name, pred in preds.items():
                dp = float(pred[b[i]] - pred[a[i]])
                rec["delta_pred"][name] = dp
                rec["abs_err"][name] = abs(dp - dl)
            recs.append(rec)
        return recs

    primary = build(scopes["valid_only_pairs"], "valid_only")
    cross = build(scopes["train_valid_cross_pairs"], "train_valid_cross")
    pool = primary + cross
    out: dict[str, Any] = {
        "n_valid_only_pairs": len(primary),
        "n_train_valid_cross_pairs": len(cross),
        "selection_rule": ("rank pairs by the absolute pair-delta error improvement of "
                           "an interaction arm over CSOI_ADD; Valid-only pairs first, "
                           "then Valid x Train crossings; deterministic top-k, no "
                           "hand-picking"),
        "requires_all_of": list(need),
    }
    for inter in ("C_CSOI_GLOBAL", "D_CSOI_REL"):
        wins = sorted(pool, key=lambda r: -(r["abs_err"]["B_CSOI_ADD"] - r["abs_err"][inter]))
        losses = sorted(pool, key=lambda r: (r["abs_err"]["B_CSOI_ADD"] - r["abs_err"][inter]))
        out[f"interaction_wins_{inter}_over_B"] = [
            _case_row(r, inter, sign=1) for r in wins[:limit]
            if r["abs_err"]["B_CSOI_ADD"] - r["abs_err"][inter] > 0]
        out[f"interaction_loses_{inter}_over_B"] = [
            _case_row(r, inter, sign=-1) for r in losses[:limit]
            if r["abs_err"]["B_CSOI_ADD"] - r["abs_err"][inter] < 0]
    out["relation_bias_wins_D_over_C"] = [
        _case_row(r, "D_CSOI_REL", sign=1)
        for r in sorted(pool, key=lambda r: -(r["abs_err"]["C_CSOI_GLOBAL"]
                                              - r["abs_err"]["D_CSOI_REL"]))[:limit]
        if r["abs_err"]["C_CSOI_GLOBAL"] - r["abs_err"]["D_CSOI_REL"] > 0]
    return out


def _case_row(r: Mapping[str, Any], inter: str, sign: int) -> dict[str, Any]:
    base = "C_CSOI_GLOBAL" if inter == "D_CSOI_REL" else "B_CSOI_ADD"
    return {"scope": r["scope"], "pair_index": r["pair_index"],
            "split_a": r["split_a"], "split_b": r["split_b"],
            "row_id_a": r["row_id_a"], "row_id_b": r["row_id_b"],
            "transform": r["transform"], "component": r["component"],
            "S_a": r["S_a"], "S_b": r["S_b"], "C": r["C"],
            "cut_bond_row": r["cut_bond_row"],
            "label_a": r["label_a"], "label_b": r["label_b"],
            "delta_true": r["delta_true"],
            f"delta_pred_{base}": r["delta_pred"][base],
            f"delta_pred_{inter}": r["delta_pred"][inter],
            f"abs_err_{base}": r["abs_err"][base],
            f"abs_err_{inter}": r["abs_err"][inter],
            "gain": r["abs_err"][base] - r["abs_err"][inter]}


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def decide(summary: Mapping[str, Any], sigma_train: float, cfg: Mapping[str, Any]) -> dict[str, Any]:
    dec = cfg["decision"]
    delta_task = float(dec["delta_task_sigma_multiple"]) * sigma_train
    weak = float(dec["weak_signal_sigma_multiple"]) * sigma_train
    out = {"sigma_train": sigma_train, "delta_task": delta_task, "weak_band": weak,
           "rules": "see PROTOCOL.md section 7", "contrasts": {}}
    for key in [f"{a}->{b}" for a, b in PRIMARY + CALIBRATION + REFERENCE]:
        c = summary["contrasts"].get(key)
        if c is None:
            continue
        lo, hi = c["ci_low"], c["ci_high"]
        point = c["delta_rmse"]
        if lo > 0 and point >= delta_task:
            label = "strong_positive"
        elif lo > 0 and point >= weak:
            label = "weak_positive"
        elif hi < 0:
            label = "negative"
        elif lo > 0:
            label = "below_weak_band"
        else:
            label = "no_evidence"
        out["contrasts"][key] = {"delta_rmse": point, "ci": [lo, hi],
                                 "decision": label,
                                 "relative_to_delta_task": point / delta_task}
    prim = out["contrasts"].get("B_CSOI_ADD->C_CSOI_GLOBAL", {})
    out["primary_decision"] = prim.get("decision", "missing")
    out["esol_extension_allowed"] = prim.get("decision") in ("strong_positive", "weak_positive")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lipo")
    ap.add_argument("--reps", type=int, default=int(C.load_config()["decision"]["bootstrap_reps"]))
    ap.add_argument("--seed", type=int, default=int(C.load_config()["decision"]["bootstrap_seed"]))
    args = ap.parse_args(argv)
    cfg = C.load_config()
    pack = D.load_pack(cfg, args.dataset)
    arms: dict[str, dict[str, Any]] = {}
    ref = None
    for name, entry in ARMS.items():
        try:
            arms[name] = load_arm(entry, ref)
        except FileNotFoundError as exc:
            print(f"[skip] {name}: {exc}")
            continue
        except (KeyError, ValueError) as exc:
            print(f"[skip] {name}: {exc}")
            continue
        if ref is None and "valid_row_id" in arms[name]:
            ref = arms[name]
    table = {name: arm_metrics(a) for name, a in arms.items()}
    contrasts = {}
    for a, b in PRIMARY + CALIBRATION + REFERENCE:
        if a in arms and b in arms:
            contrasts[f"{a}->{b}"] = contrast(arms[a], arms[b], args.reps, args.seed)
    sigma_train = float(np.std(np.asarray(pack["labels"])[np.asarray(pack["train_idx"])]))
    summary = {
        "protocol": cfg["protocol_id"],
        "cache_sha256": pack["cache_sha256"], "content_sha256": pack["content_sha256"],
        "bootstrap": {"reps": args.reps, "seed": args.seed,
                      "unit": "Valid molecule", "statistic": "paired RMSE difference",
                      "note": ("single train seed (0) as mandated: train-seed "
                               "uncertainty is NOT covered by this interval")},
        "sigma_train": sigma_train,
        "arms": table,
        "contrasts": contrasts,
    }
    summary["decision"] = decide(summary, sigma_train, cfg)
    C.write_json(C.results_root() / "summary.json", summary)

    rows = ["arm,role,n_train,n_valid,train_rmse,valid_rmse,valid_mae,valid_pearson,"
            "valid_r2,best_epoch,params,gpu_hours,run_id"]
    roles = {"A_TRC_FACTOR_CUTRESP": "parent_reuse", "B_CSOI_ADD": "trained",
             "C_CSOI_GLOBAL": "trained", "D_CSOI_REL": "trained",
             "E_GINE_RAW_REF": "parent_reuse"}
    for name, m in table.items():
        rows.append(",".join(str(x) for x in [
            name, roles[name], m["n_train"], m["n_valid"],
            f"{m['train_rmse']:.6f}", f"{m['valid_rmse']:.6f}", f"{m['valid_mae']:.6f}",
            f"{m['valid_pearson']:.6f}", f"{m['valid_r2']:.6f}",
            m["best_epoch"], m["params"], m["gpu_hours"], arms[name]["run_id"]]))
    (C.results_root() / "main_table.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    C.write_json(C.results_root() / "pair_metrics.json", pair_metrics(pack, arms))
    C.write_json(C.results_root() / "support.json", support_report(pack, arms))
    C.write_json(C.results_root() / "cases.json", cases(pack, arms))
    for name, m in table.items():
        print(f"  {name:22s} valid_rmse={m['valid_rmse']:.4f} mae={m['valid_mae']:.4f}")
    for k, c in contrasts.items():
        print(f"  {k:44s} delta_rmse={c['delta_rmse']:+.4f} "
              f"ci=[{c['ci_low']:+.4f},{c['ci_high']:+.4f}] "
              f"-> {summary['decision']['contrasts'][k]['decision']}")
    print(f"primary: {summary['decision']['primary_decision']} "
          f"(delta_task={summary['decision']['delta_task']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
