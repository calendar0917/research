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


def pair_metrics(pack: Mapping[str, Any], arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    labels = np.asarray(pack["labels"], np.float64)
    train = np.asarray(pack["train_idx"], np.int64)
    valid = np.asarray(pack["valid_idx"], np.int64)
    store = C.PPAIR.PairStore(C.load_config(), pack, labels, None, train,
                              np.asarray(pack["row_ids"]), log=lambda *_a: None)
    a = np.asarray(pack["pair_mol_a"], np.int64)[store.index]
    b = np.asarray(pack["pair_mol_b"], np.int64)[store.index]
    transform = store.transform
    cell = np.asarray(pack["pair_cell"], np.uint8)[store.index]
    rows = np.asarray(pack["row_ids"], np.int64)
    out: dict[str, Any] = {
        "n_train_pairs": int(store.npairs()),
        "scope_note": ("pair directory is train-only; only pairs whose two molecules "
                       "are both Valid are scored"),
        "by_arm": {},
    }
    valid_set = set(valid.tolist())
    keep = np.asarray([int(x) in valid_set and int(y) in valid_set for x, y in zip(a, b)])
    out["n_pairs_with_both_molecules_in_valid"] = int(keep.sum())
    if not keep.any():
        out["note"] = "no train pair has both molecules in Valid"
        return out
    a_k, b_k = a[keep], b[keep]
    dl = labels[b_k] - labels[a_k]
    tt = transform[keep]
    comp = np.asarray(pack["pair_component"], np.int64)[store.index][keep]
    for name, arm in arms.items():
        if arm.get("valid_pred") is None:
            continue
        # predictions are aligned with valid_row_id -> map row_id -> molecule index
        idx_of_row = {int(r): i for i, r in enumerate(rows.tolist())}
        pred = np.full(int(pack["n_molecules"]), np.nan)
        for rid, val in zip(arm["valid_row_id"].tolist(), arm["valid_pred"].tolist()):
            pred[idx_of_row[int(rid)]] = val
        dp = pred[b_k] - pred[a_k]
        diff_comp = np.asarray([len(set(comp[tt == t].tolist())) > 1 for t in tt])
        rec = {"n_pairs_scored": int(keep.sum()), "all": _delta_stats(dl, dp),
               "matched_transforms": {"n_pairs": int(diff_comp.sum()),
                                      **_delta_stats(dl[diff_comp], dp[diff_comp])},
               "same_component_pairs": {"n_pairs": int((~diff_comp).sum()),
                                        **_delta_stats(dl[~diff_comp], dp[~diff_comp])},
               "by_transform": {}}
        for t in np.unique(tt).tolist():
            m = tt == t
            if m.sum() >= 5:
                rec["by_transform"][f"transform_{int(t)}"] = {"n": int(m.sum()),
                                                              **_delta_stats(dl[m], dp[m])}
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
    labels = np.asarray(pack["labels"], np.float64)
    rows = np.asarray(pack["row_ids"], np.int64)
    idx_of_row = {int(r): i for i, r in enumerate(rows.tolist())}
    a = np.asarray(pack["pair_mol_a"], np.int64)
    b = np.asarray(pack["pair_mol_b"], np.int64)
    cell = np.asarray(pack["pair_cell"], np.uint8)
    transform = np.asarray(pack["pair_transform"], np.int64)
    comp = np.asarray(pack["pair_component"], np.int64)
    valid = np.asarray(pack["valid_idx"], np.int64)
    train = np.asarray(pack["train_idx"], np.int64)
    store = C.PPAIR.PairStore(C.load_config(), pack, labels, None, train,
                              rows, log=lambda *_a: None)
    keep = np.isin(store.index, np.nonzero(cell == 0)[0])
    pairs = store.index[keep]
    both_valid = np.asarray([int(a[p]) in set(valid.tolist()) and int(b[p]) in set(valid.tolist())
                             for p in pairs.tolist()])
    pairs = pairs[both_valid]
    if pairs.size == 0:
        return {"note": "no train pairs with both molecules in Valid", "cases": []}
    preds = {}
    for name in ("B_CSOI_ADD", "C_CSOI_GLOBAL", "D_CSOI_REL", "A_TRC_FACTOR_CUTRESP"):
        arm = arms.get(name)
        if arm is None or arm.get("valid_pred") is None:
            continue
        pred = np.full(int(rows.size), np.nan)
        pos = np.asarray([idx_of_row[int(r)] for r in arm["valid_row_id"].tolist()])
        pred[pos] = arm["valid_pred"]
        preds[name] = pred
    dl = labels[b[pairs]] - labels[a[pairs]]
    table = []
    for i, p in enumerate(pairs.tolist()):
        rec = {"pair_index": int(p), "transform": int(transform[p]),
               "component": int(comp[p]), "mol_a": int(a[p]), "mol_b": int(b[p]),
               "row_id_a": int(rows[a[p]]), "row_id_b": int(rows[b[p]]),
               "label_a": float(labels[a[p]]), "label_b": float(labels[b[p]]),
               "delta_true": float(dl[i]),
               "s_fragment_a": int(np.asarray(pack["pair_Sa"])[p]),
               "s_fragment_b": int(np.asarray(pack["pair_Sb"])[p]),
               "c_fragment": int(np.asarray(pack["pair_C"])[p]),
               "cut_bond_row": int(np.asarray(pack["pair_cut_bond_row"])[p]),
               "pred": {}}
        for name, pred in preds.items():
            rec["pred"][name] = {"a": float(pred[a[p]]), "b": float(pred[b[p]]),
                                 "delta": float(pred[b[p]] - pred[a[p]]),
                                 "abs_err": float(abs(pred[b[p]] - pred[a[p]] - dl[i]))}
        table.append(rec)
    # auto-selected rules: interaction (C or D) clearly better than additive (B)
    def gain(rec, inter):
        return rec["pred"]["B_CSOI_ADD"]["abs_err"] - rec["pred"][inter]["abs_err"]

    selection = {}
    for inter in ("C_CSOI_GLOBAL", "D_CSOI_REL"):
        if inter not in preds:
            continue
        ranked = sorted(table, key=lambda r: -gain(r, inter))
        selection[f"interaction_wins_{inter}_over_B"] = [
            {k: r[k] for k in ("pair_index", "row_id_a", "row_id_b", "delta_true",
                               "transform", "component", "s_fragment_a", "s_fragment_b",
                               "c_fragment", "cut_bond_row")}
            | {"abs_err_B": r["pred"]["B_CSOI_ADD"]["abs_err"],
               "abs_err_inter": r["pred"][inter]["abs_err"],
               "gain": gain(r, inter),
               "delta_pred_B": r["pred"]["B_CSOI_ADD"]["delta"],
               "delta_pred_inter": r["pred"][inter]["delta"]}
            for r in ranked[:limit] if gain(r, inter) > 0]
        ranked_bad = sorted(table, key=lambda r: gain(r, inter))
        selection[f"interaction_loses_{inter}_over_B"] = [
            {k: r[k] for k in ("pair_index", "row_id_a", "row_id_b", "delta_true",
                               "transform", "component")}
            | {"gain": gain(r, inter),
               "delta_pred_B": r["pred"]["B_CSOI_ADD"]["delta"],
               "delta_pred_inter": r["pred"][inter]["delta"]}
            for r in ranked_bad[:limit] if gain(r, inter) < 0]
    return {"n_pairs_valid": int(pairs.size),
            "selection_rule": ("rank pairs with both molecules in Valid by "
                               "|pair-delta error| improvement of the interaction arm "
                               "over the additive arm; deterministic top-k"),
            "cases": selection}


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
