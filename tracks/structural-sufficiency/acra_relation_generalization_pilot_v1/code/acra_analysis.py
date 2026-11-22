"""Dev evaluation, support stratification, matched pairs, outlier sensitivity.

No official Valid or Test prediction is produced here.  Matched pairs are the
parent round's real replacement pairs restricted to the official Train rows and
then classified by the frozen Train-sub / Dev split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import acra_common as C
import acra_cache as K
import acra_data as D
import acra_model as M
import acra_split as S
import acra_support_gate as SG

ARMS = ("ACRA_REL_REF", "ACRA_MOMENT_ONLY", "ACRA_HIST_ONLY",
        "ACRA_TOPO_COUPLED", "ACRA_SUPPORT_GATE")
RUN_IDS = {a: f"{a}_S0" for a in ARMS}
REF = "ACRA_REL_REF"
PAIR_PACK = (C.ROUND_DIR.parent / "task_response_composition_s0_v1" / "data"
             / "trc_cache_lipo.npz")
EDGES = (0.0, 0.2, 0.5, 1.0)
SUPPORT_BINS = ((0, 0, "0"), (1, 1, "1"), (2, 4, "2-4"), (5, 9, "5-9"),
                (10, 49, "10-49"), (50, 10 ** 9, "50+"))


def bucket(x: float) -> str:
    if x <= 0:
        return "0"
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        if lo < x <= hi:
            return f"({lo:g},{hi:g}]"
    return ">1"


def load_run(arm: str, pack) -> dict[str, Any]:
    rdir = C.run_dir(RUN_IDS[arm])
    manifest = C.read_json(rdir / "manifest.json")
    with np.load(rdir / "predictions_best_train_sub.npz") as tr, \
            np.load(rdir / "predictions_best_dev.npz") as dv:
        train = {k: np.asarray(tr[k], np.float64) for k in tr.files}
        dev = {k: np.asarray(dv[k], np.float64) for k in dv.files}
    curves = C.read_csv(rdir / "curves.csv")
    import torch
    ckpt = torch.load(rdir / "state_best.pt", map_location="cpu", weights_only=False)
    model = M.ACRAModel(ckpt["config"], *C.official_feature_dims(), ckpt["arm"])
    model.load_pack(pack)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return {"arm": arm, "run_id": RUN_IDS[arm], "manifest": manifest,
            "train": train, "dev": dev, "curves": curves, "model": model,
            "rdir": rdir}


def event_outputs(model: M.ACRAModel, source: D.BatchSource, rows) -> dict[str, np.ndarray]:
    import torch

    rows = np.asarray(rows, np.int64)
    keys = ("r", "scalar_s", "scalar_c", "interaction", "support", "rho", "ev_mol")
    out: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    with torch.no_grad():
        for start in range(0, rows.size, 128):
            part = rows[start:start + 128]
            batch = source.make(part)
            res = model.forward_batch(batch)
            for k in keys:
                if k == "ev_mol":
                    arr = res[k].detach().cpu().numpy()
                else:
                    arr = res[k].detach().double().cpu().numpy()
                out[k].append(arr)
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}


def _pair_data(raw, split, labels):
    with np.load(PAIR_PACK, allow_pickle=False) as handle:
        a = np.asarray(handle["pair_mol_a"], np.int64)
        b = np.asarray(handle["pair_mol_b"], np.int64)
        comp = np.asarray(handle["pair_component"], np.int64)
    official_train = np.zeros(labels.size, bool)
    official_train[np.asarray(raw["train_idx"], np.int64)] = True
    allowed = official_train[a] & official_train[b]
    sa, sb = split[a], split[b]
    masks = {
        "train_sub_train_sub": allowed & (sa == 0) & (sb == 0),
        "train_sub_dev": allowed & (((sa == 0) & (sb == 1)) | ((sa == 1) & (sb == 0))),
        "dev_dev": allowed & (sa == 1) & (sb == 1),
    }
    return a, b, comp, masks


def _pair_compare(deltas, truth, clusters, n_boot=2000, seed=0):
    out = {}
    uniq = np.unique(clusters)
    rng = np.random.default_rng(int(seed))
    for candidate in ARMS:
        if candidate == REF:
            continue
        p_c, p_o = deltas[candidate], deltas[REF]
        observed = float(np.mean(np.abs(p_o - truth)) - np.mean(np.abs(p_c - truth)))
        diffs = []
        for _ in range(int(n_boot)):
            pick = rng.integers(0, uniq.size, uniq.size)
            idx = np.concatenate([np.nonzero(clusters == uniq[j])[0] for j in pick])
            diffs.append(float(np.mean(np.abs(p_o[idx] - truth[idx]))
                               - np.mean(np.abs(p_c[idx] - truth[idx]))))
        out[f"{candidate}_vs_{REF}"] = {
            "delta_mae_difference": observed,
            "lo": float(np.percentile(diffs, 2.5)),
            "hi": float(np.percentile(diffs, 97.5)),
            "positive_means_candidate_better": observed > 0,
            "n_pairs": int(truth.size), "n_clusters": int(uniq.size),
            "reps": int(n_boot),
        }
    return out


def run(cfg: Mapping[str, Any], log=print) -> dict[str, Any]:
    import torch

    pack = K.ensure_pack(cfg, log=log)
    raw = D.load_raw(cfg)
    labels = np.asarray(raw["labels"], np.float64)
    smiles = np.asarray(raw["smiles"])
    split_payload = S.load(cfg)
    split = S.row_split(cfg, labels.size)
    train_sub = np.asarray(split_payload["train_sub_rows"], np.int64)
    dev = np.asarray(split_payload["dev_rows"], np.int64)
    sg_cfg = cfg["composition"]["support_gate"]
    sg = SG.build(pack, train_sub, dev, floor=sg_cfg["rho_floor"],
                  span=sg_cfg["rho_span"], offset=sg_cfg["support_offset"])
    source = D.BatchSource(pack, "cpu", event_support=sg["support_by_event"])

    runs = {arm: load_run(arm, pack) for arm in ARMS}
    for arm, rec in runs.items():
        if not np.array_equal(rec["dev"]["row_id"].astype(np.int64), dev):
            raise ValueError(f"{arm}: Dev prediction rows do not match the frozen split")
        if not np.allclose(rec["dev"]["label"], labels[dev], atol=1e-8):
            raise ValueError(f"{arm}: Dev labels do not match the raw cache")
    truth_dev = labels[dev]
    truth_train = labels[train_sub]

    # ---- main table --------------------------------------------------------
    rows = []
    dev_pred = {}
    for arm, rec in runs.items():
        m = rec["manifest"]
        v = rec["dev"]
        best_epoch = int(m["best"]["epoch"])
        best_row = next(r for r in rec["curves"] if int(r["epoch"]) == best_epoch)
        pred = v["pred"]
        dev_pred[arm] = pred
        rows.append({
            "model": arm,
            "kind": "acra_arm",
            "seed": m["seed"],
            "params": m["params"]["total"],
            "best_epoch": best_epoch,
            "epochs_run": m["epochs"],
            "train_sub_rmse": float(best_row["train_sub_rmse"]),
            "train_sub_mae": float(best_row["train_sub_mae"]),
            "dev_rmse": float(m["best"]["dev_rmse"]),
            "dev_mae": float(m["best"]["dev_mae"]),
            "generalization_gap_rmse": float(m["best"]["dev_rmse"]
                                             - best_row["train_sub_rmse"]),
            "gate": float(m["final"]["gate"]),
            "gamma_mean": (None if m["final"].get("gamma_mean") is None
                           else float(m["final"]["gamma_mean"])),
            "wall_seconds": m["wall_seconds"],
            "gpu_hours": m["gpu_hours"],
            "max_vram_bytes": m["max_vram_bytes"],
            "run_id": RUN_IDS[arm],
            "test_labels_read": False,
        })
    C.write_csv(C.analysis_dir() / "main_table.csv", rows)
    C.write_csv(C.results_root() / "main_table.csv", rows)
    log(f"wrote main_table.csv ({len(rows)} arms)")

    # ---- Dev paired bootstrap vs REF --------------------------------------
    comparisons = {}
    for candidate in ARMS:
        if candidate == REF:
            continue
        key = f"{candidate}_vs_{REF}"
        comparisons[key] = C.paired_bootstrap(dev_pred[REF], dev_pred[candidate],
                                              truth_dev, n_boot=2000, seed=0)
        comparisons[key]["definition"] = ("RMSE(REF) - RMSE(candidate); "
                                          "positive => candidate better")
    log("wrote Dev paired bootstrap comparisons")

    # ---- real matched pairs on official Train rows -------------------------
    a, b, comp, pair_masks = _pair_data(raw, split, labels)
    pred_mol = {arm: np.full(labels.size, np.nan) for arm in ARMS}
    for arm, rec in runs.items():
        for name in ("train", "dev"):
            rows_ = rec[name]["row_id"].astype(np.int64)
            pred_mol[arm][rows_] = rec[name]["pred"]
    pair_out: dict[str, Any] = {
        "source": str(PAIR_PACK.relative_to(C.REPO_ROOT)),
        "policy": "only pairs with both rows in the official Train split; "
                  "official Valid/Test-touching pairs are never used",
        "counts": {k: int(v.sum()) for k, v in pair_masks.items()},
        "scopes": {},
    }
    cross_deltas = {}
    cross_truth = None
    cross_clusters = None
    for scope, mask in pair_masks.items():
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            continue
        true_delta = labels[b[idx]] - labels[a[idx]]
        clusters = (comp[idx] if scope == "train_sub_train_sub"
                    else (a[idx] if split[a[idx]] == 1 else b[idx]))
        rec_scope: dict[str, Any] = {"n": int(idx.size)}
        for arm in ARMS:
            pred_delta = pred_mol[arm][b[idx]] - pred_mol[arm][a[idx]]
            stats = C.paired_delta_stats(true_delta, pred_delta,
                                         clusters if scope == "train_sub_dev" else None,
                                         n_boot=2000, seed=0)
            rec_scope[arm] = stats
            if scope == "train_sub_dev":
                cross_deltas[arm] = pred_delta
                cross_truth = true_delta
                cross_clusters = clusters
        pair_out["scopes"][scope] = rec_scope
    if cross_deltas:
        pair_out["cross_scope_comparisons"] = _pair_compare(
            cross_deltas, cross_truth, cross_clusters)
    C.write_json(C.analysis_dir() / "pair_delta.json", pair_out)

    # ---- support / novelty on Dev -----------------------------------------
    ev_ptr = np.asarray(pack["ev_ptr"], np.int64)
    combo = np.asarray(pack["ev_combo"], np.int64)
    counts = sg["combo_molecule_counts"]
    ev_mol = np.repeat(np.arange(int(pack["n_molecules"])), np.diff(ev_ptr))
    dev_pos = SG.positions_from_rows(pack, dev)
    novelty = np.asarray([
        float(np.mean(counts[combo[ev_ptr[p]:ev_ptr[p + 1]]] == 0))
        for p in dev_pos.tolist()
    ])
    bucket_of = np.asarray([bucket(x) for x in novelty])
    support_block: dict[str, Any] = {
        "protocol_id": cfg["protocol_id"],
        "definition": {
            "novelty": "fraction of a molecule's directed events whose exact "
                       "(S,C,cut-bond) combo has zero Train-sub molecules",
            "support": "distinct Train-sub molecules containing the combo; "
                       "Train-sub LOO, Dev full count",
            "buckets": ["0", "(0,0.2]", "(0.2,0.5]", "(0.5,1]"],
        },
        "n_dev": int(dev.size),
        "support_summary": sg["summary"],
        "novelty_quantiles": C.quantiles(novelty, (0, 5, 25, 50, 75, 95, 100)),
        "buckets": {},
        "confounding": {},
        "event_support_bins": {},
        "arm_comparisons": {},
    }
    for name in ("0", "(0,0.2]", "(0.2,0.5]", "(0.5,1]"):
        mask = bucket_of == name
        if not mask.any():
            continue
        rec: dict[str, Any] = {"n": int(mask.sum())}
        for arm in ARMS:
            rec[arm] = {
                "rmse": C.rmse(dev_pred[arm][mask], truth_dev[mask]),
                "mae": C.mae(dev_pred[arm][mask], truth_dev[mask]),
            }
        rec["rmse_reduction_support_gate_vs_ref"] = (
            rec[REF]["rmse"] - rec["ACRA_SUPPORT_GATE"]["rmse"])
        rec["mae_reduction_support_gate_vs_ref"] = (
            rec[REF]["mae"] - rec["ACRA_SUPPORT_GATE"]["mae"])
        support_block["buckets"][name] = rec
    for arm in ARMS:
        if arm == REF:
            continue
        key = f"{arm}_vs_{REF}"
        support_block["arm_comparisons"][key] = {
            "overall": C.paired_bootstrap(dev_pred[REF], dev_pred[arm], truth_dev,
                                          n_boot=2000, seed=0),
            "bootstrap_unit": "Dev molecule",
        }
        for name in ("0", "(0,0.2]", "(0.2,0.5]", "(0.5,1]"):
            mask = bucket_of == name
            if int(mask.sum()) < 5:
                continue
            support_block["arm_comparisons"][key][name] = C.paired_bootstrap(
                dev_pred[REF][mask], dev_pred[arm][mask], truth_dev[mask],
                n_boot=2000, seed=0)
    # size / cycle-rank confounding
    desc = {
        "n_atoms": np.asarray(pack["mol_n_atoms"], np.float64)[dev_pos],
        "n_bonds": np.asarray(pack["mol_n_bonds"], np.float64)[dev_pos],
        "cycle_rank": np.asarray(pack["mol_cycle_rank"], np.float64)[dev_pos],
    }
    support_block["confounding"] = {
        "pearson_novelty_vs_" + k: C.pearson(novelty, v) for k, v in desc.items()
    }
    for name in ("0", "(0,0.2]", "(0.2,0.5]", "(0.5,1]"):
        mask = bucket_of == name
        if not mask.any():
            continue
        support_block["confounding"][name] = {
            "mean_n_atoms": float(np.mean(desc["n_atoms"][mask])),
            "mean_n_bonds": float(np.mean(desc["n_bonds"][mask])),
            "mean_cycle_rank": float(np.mean(desc["cycle_rank"][mask])),
        }
    # event-level interaction amplitude vs Train support
    for arm in ("ACRA_REL_REF", "ACRA_SUPPORT_GATE"):
        ev = event_outputs(runs[arm]["model"], source, dev)
        support = ev["support"].astype(np.int64)
        scalar = ev["scalar_s"] + ev["scalar_c"]
        for lo, hi, label in SUPPORT_BINS:
            mask = (support >= lo) & (support <= hi)
            if not mask.any():
                continue
            support_block["event_support_bins"].setdefault(label, {})[arm] = {
                "n_events": int(mask.sum()),
                "mean_abs_interaction": float(np.mean(np.abs(ev["interaction"][mask]))),
                "mean_abs_scalar_sum": float(np.mean(np.abs(scalar[mask]))),
                "mean_abs_r": float(np.mean(np.abs(ev["r"][mask]))),
                "mean_rho": float(np.mean(ev["rho"][mask])) if arm == "ACRA_SUPPORT_GATE" else 1.0,
                "negative_r_fraction": float(np.mean(ev["r"][mask] < 0)),
            }
    C.write_json(C.analysis_dir() / "support_generalization.json", support_block)

    # ---- outliers ----------------------------------------------------------
    out: dict[str, Any] = {
        "protocol_id": cfg["protocol_id"],
        "policy": "all Dev molecules retained; no post-hoc deletion is used to claim improvement",
        "arms": {},
        "ranking": {},
        "top5_union_rows": [],
    }
    sq = {arm: (dev_pred[arm] - truth_dev) ** 2 for arm in ARMS}
    abs_err = {arm: np.abs(dev_pred[arm] - truth_dev) for arm in ARMS}
    order = np.argsort(-sq[REF])[:5]
    union = sorted(set(np.concatenate([np.argsort(-sq[a])[:5] for a in ARMS]).tolist()))
    out["top5_union_rows"] = [int(dev[i]) for i in union]
    for arm in ARMS:
        idx = np.argsort(-sq[arm])[:5]
        n = truth_dev.size
        lomo = np.asarray([
            float(np.sqrt(np.sum(sq[arm]) - sq[arm][i]) / (n - 1)) for i in range(n)
        ])
        out["arms"][arm] = {
            "n": int(n),
            "rmse": float(np.sqrt(np.mean(sq[arm]))),
            "mae": float(np.mean(abs_err[arm])),
            "abs_error_quantiles": C.quantiles(abs_err[arm], (50, 75, 90, 95, 99, 100)),
            "max_error_molecule": {
                "dev_index": int(idx[0]),
                "row_id": int(dev[idx[0]]),
                "smiles": str(smiles[dev[idx[0]]]),
                "n_atoms": int(pack["mol_n_atoms"][dev_pos[idx[0]]]),
                "n_bonds": int(pack["mol_n_bonds"][dev_pos[idx[0]]]),
                "cycle_rank": int(pack["mol_cycle_rank"][dev_pos[idx[0]]]),
                "novelty": float(novelty[idx[0]]),
                "pred": float(dev_pred[arm][idx[0]]),
                "label": float(truth_dev[idx[0]]),
                "squared_error": float(sq[arm][idx[0]]),
            },
            "lomo_rmse_range": {
                "min": float(lomo.min()), "max": float(lomo.max()),
                "range": float(lomo.max() - lomo.min()),
            },
            "top5_share_of_total_squared_error": float(np.sum(sq[arm][idx]) / np.sum(sq[arm])),
            "rmse_without_top5_own": float(np.sqrt(
                (np.sum(sq[arm]) - np.sum(sq[arm][idx])) / (n - 5))),
            "rmse_without_top5_union": float(np.sqrt(
                (np.sum(sq[arm]) - np.sum(sq[arm][union])) / (n - len(union)))),
        }
    full_rank = sorted(ARMS, key=lambda a: float(np.sqrt(np.mean(sq[a]))))
    own_excl = {a: float(np.sqrt((np.sum(sq[a]) - np.sum(np.sort(sq[a])[-5:]))
                                 / (truth_dev.size - 5))) for a in ARMS}
    union_excl = {a: out["arms"][a]["rmse_without_top5_union"] for a in ARMS}
    out["ranking"] = {
        "full_dev_rmse": full_rank,
        "without_each_models_own_top5": sorted(ARMS, key=lambda a: own_excl[a]),
        "without_top5_union": sorted(ARMS, key=lambda a: union_excl[a]),
        "own_top5_excluded_rmse": own_excl,
        "union_top5_excluded_rmse": union_excl,
    }
    C.write_json(C.analysis_dir() / "outlier_sensitivity.json", out)

    # ---- summary -----------------------------------------------------------
    summary = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "commit": C.git_commit(),
        "seed": 0,
        "test_labels_read": False,
        "split": {
            "sha256": split_payload["split_sha256"],
            "n_train_sub": split_payload["n_train_sub"],
            "n_dev": split_payload["n_dev"],
            "n_scaffolds_train_sub": split_payload["n_scaffolds_train_sub"],
            "n_scaffolds_dev": split_payload["n_scaffolds_dev"],
        },
        "main_table": rows,
        "dev_paired_bootstrap": comparisons,
        "pair_delta": pair_out,
        "support_generalization_headline": {
            "buckets": {k: {a: v[a] for a in ARMS}
                        for k, v in support_block["buckets"].items()},
            "support_gate_vs_ref": support_block["arm_comparisons"].get(
                "ACRA_SUPPORT_GATE_vs_ACRA_REL_REF", {}),
        },
        "outlier_headline": out["ranking"],
        "checks_all_passed": bool(C.read_json(C.analysis_dir() / "checks.json")["all_passed"]),
        "relation_factorization": C.read_json(C.analysis_dir() / "relation_factorization.json"),
        "relation_term_provenance": C.read_json(
            C.analysis_dir() / "relation_term_provenance.json"),
    }
    if (C.analysis_dir() / "structural_cases.json").is_file():
        summary["structural_cases"] = C.read_json(C.analysis_dir() / "structural_cases.json")
    if (C.analysis_dir() / "pair_delta.json").is_file():
        summary["pair_delta"] = C.read_json(C.analysis_dir() / "pair_delta.json")
    C.write_json(C.results_root() / "summary.json", summary)
    log("wrote support_generalization.json, outlier_sensitivity.json, results/summary.json")
    return {"main_table": rows, "support": support_block, "outliers": out,
            "summary": summary}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    run(C.load_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())