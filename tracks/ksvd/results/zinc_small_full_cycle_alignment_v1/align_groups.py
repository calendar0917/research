"""Small/Full calibration comparison + cycle-penalty group alignment.

Descriptive only.  The single fitted quantity per model is the train-median
scalar output bias b = median_train(y - pred_raw).  No head/scaler/oracle fit,
no backbone update, no test access.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
HANDOFF = REPO / "tracks/ksvd/results/zinc_dictionary_real_data_handoff"
CYCLE = REPO / "tracks/ksvd/results/zinc_long_cycle_audit"
PROBE = REPO / "tracks/ksvd/results/zinc_full_nonlinear_probe_v1"
TOL = 1e-5
PUB = {"small_raw_valid": 0.12105831989174476, "full_raw_valid": 0.1191540920053958,
       "full_cal_valid": 0.11506585458567133}


def mae(p, y):
    return float(np.mean(np.abs(p - y)))


def calibrate(ptr, ytr, pva):
    b = float(np.median(ytr - ptr))
    return b, ptr + b, pva + b


def join_check(lab, y_handoff, split):
    """Return aligned label array + provenance report."""
    rep = {"n_rows": int(len(lab)), "n_unique_subset_index": int(lab["subset_index"].nunique()),
           "subset_index_is_0_based_contiguous": bool(
               np.array_equal(np.sort(lab["subset_index"].to_numpy()), np.arange(len(lab)))),
           "molecule_id_matches_pattern": bool(
               all(f"{split}:{i:04d}" == m for i, m in zip(lab["subset_index"], lab["molecule_id"]))),
           "target_max_abs_diff_vs_handoff_y": float(
               np.max(np.abs(lab["target"].to_numpy() - y_handoff))),
           "label_effective_cycle_snapped_unique": sorted(
               np.unique(lab["label_effective_cycle_snapped"].to_numpy()).tolist()),
           }
    rep["target_identity_ok"] = rep["target_max_abs_diff_vs_handoff_y"] < TOL
    rep["identity_ok"] = (rep["n_unique_subset_index"] == len(lab)
                          and rep["subset_index_is_0_based_contiguous"]
                          and rep["molecule_id_matches_pattern"]
                          and rep["target_identity_ok"])
    return rep


def group_masks(label):
    lc = label["label_effective_cycle_snapped"].to_numpy()
    idx = label["subset_index"].to_numpy()
    g172 = idx == 172
    g0 = (lc == 0.0) & ~g172
    g1 = (lc < 0.0) & ~g172
    unknown = ~(g0 | g1 | g172)
    return {"G0": g0, "G1": g1, "G172": g172, "UNKNOWN": unknown}


def group_table(pred_cal, y, masks):
    rows = []
    abs_all = np.abs(pred_cal - y)
    total = float(abs_all.sum())
    for name, m in masks.items():
        if m.sum() == 0:
            rows.append({"group": name, "n": 0}); continue
        se = pred_cal[m] - y[m]
        contrib = float(abs_all[m].sum() / len(y))
        rows.append({
            "group": name, "n": int(m.sum()),
            "mae": mae(pred_cal[m], y[m]),
            "signed_mean": float(se.mean()), "signed_median": float(np.median(se)),
            "total_mae_contribution": contrib,
            "abs_error_mass_share": float(abs_all[m].sum() / total),
        })
    return rows


def main():
    # ---- raw predictions ----
    ht = np.load(HANDOFF / "train_reader.npz")
    hv = np.load(HANDOFF / "valid_reader.npz")
    sr = np.load(OUT / "small_replay.npz")
    ytr, yva = ht["y"], hv["y"]
    ptr_full, pva_full = ht["p_base"], hv["p_base"]
    ptr_sm, pva_sm = sr["ptr"], sr["pva"]

    b_f, cal_tr_f, cal_va_f = calibrate(ptr_full, ytr, pva_full)
    b_s, cal_tr_s, cal_va_s = calibrate(ptr_sm, ytr, pva_sm)

    models = {
        "Small": {"b": b_s, "cal_tr": cal_tr_s, "cal_va": cal_va_s,
                  "raw_tr": mae(ptr_sm, ytr), "raw_va": mae(pva_sm, yva)},
        "Full": {"b": b_f, "cal_tr": cal_tr_f, "cal_va": cal_va_f,
                 "raw_tr": mae(ptr_full, ytr), "raw_va": mae(pva_full, yva)},
    }
    # A/B/C two seeds from the previous frozen probe (calibrated already)
    paired = pd.read_csv(PROBE / "valid_paired.csv")
    assert np.array_equal(paired["graph_id"].to_numpy(), hv["ids"])
    assert np.max(np.abs(paired["y"].to_numpy() - yva)) < TOL
    assert np.max(np.abs(paired["parent_calibrated"].to_numpy() - cal_va_f)) < 1e-6
    for cfg in ("A", "B", "C"):
        for s in (0, 1):
            models[f"{cfg}_seed{s}"] = {"cal_va": paired[f"{cfg}_seed{s}_calibrated"].to_numpy(),
                                        "raw_va": paired[f"{cfg}_seed{s}_raw"].to_numpy()}

    # ---- identity / join ----
    lab_v = pd.read_csv(CYCLE / "valid_cycle_audit_label.csv")
    lab_t = pd.read_csv(CYCLE / "train_cycle_audit_label.csv")
    jv = join_check(lab_v, yva, "valid")
    jt = join_check(lab_t, ytr, "train")
    if not jv["identity_ok"]:
        raise SystemExit(f"valid join identity failed: {jv}")
    masks_v = group_masks(lab_v)
    masks_t = group_masks(lab_t)
    assert sum(m.sum() for m in masks_v.values()) == len(yva)
    assert masks_v["UNKNOWN"].sum() == 0

    # id172 identity
    i172 = int(np.argmax(np.abs(cal_va_f - yva)))
    id172 = {"valid_row_index": i172, "molecule_id": str(lab_v["molecule_id"].iloc[i172]),
             "y": float(yva[i172]),
             "label_effective_cycle_snapped": float(lab_v["label_effective_cycle_snapped"].iloc[i172]),
             "label_cycle_component": float(lab_v["label_cycle_component"].iloc[i172]),
             "y_without_cycle_label": float(lab_v["y_without_cycle_label"].iloc[i172])}

    # ---- group tables ----
    groups = {}
    for name, m in models.items():
        groups[name] = {
            "valid": group_table(m["cal_va"], yva, masks_v),
            "valid_paired_gain_vs_parent_full": None,
        }
    # train groups (only Small/Full available)
    train_groups = {"Small": group_table(cal_tr_s, ytr, masks_t),
                    "Full": group_table(cal_tr_f, ytr, masks_t)}

    # ---- paired gains (calibrated) ----
    def paired_gain(base_cal, cand_cal, tag, label=lab_v):
        d = np.abs(base_cal - yva) - np.abs(cand_cal - yva)  # >0 cand better
        g172 = i172
        ex = np.ones(len(yva), bool); ex[g172] = False
        out = {"gain_all": float(d.mean()), "gain_ex_id172": float((d.sum() - d[g172]) / (len(d) - 1))}
        lc = label["label_effective_cycle_snapped"].to_numpy()
        idx = label["subset_index"].to_numpy()
        for gname, m in (("G0", (lc == 0) & (idx != 172)), ("G1", (lc < 0) & (idx != 172)),
                         ("G172", idx == 172)):
            out[f"gain_{gname}"] = float(d[m].mean())
            out[f"contrib_change_{gname}"] = float(d[m].sum() / len(yva))
        return out

    full_vs_small = paired_gain(cal_va_s, cal_va_f, "Full_vs_Small")
    pair_gains = {"Full_vs_Small": full_vs_small}
    for cfg in ("A", "B", "C"):
        for s in (0, 1):
            pair_gains[f"{cfg}_seed{s}_vs_Full"] = paired_gain(cal_va_f, models[f"{cfg}_seed{s}"]["cal_va"], cfg)

    # ---- gap to 0.09 ----
    overall_full = mae(cal_va_f, yva)
    grp_full = {r["group"]: r for r in groups["Full"]["valid"]}
    gap = {
        "full_cal_valid_mae": overall_full,
        "gap_to_0.09": overall_full - 0.09,
        "gap_to_0.08": overall_full - 0.08,
        "group_contributions": {k: grp_full[k].get("total_mae_contribution") for k in ("G0", "G1", "G172")},
        "contrib_note": "arithmetic budget if a group were perfectly predicted with others unchanged; "
                        "NOT an achievable gain or learning ceiling",
    }

    # ---- per-graph valid table ----
    out_tbl = pd.DataFrame({
        "graph_id": hv["ids"], "molecule_id": lab_v["molecule_id"],
        "subset_index": lab_v["subset_index"], "y": yva,
        "label_effective_cycle_snapped": lab_v["label_effective_cycle_snapped"],
        "group": np.select([masks_v["G0"], masks_v["G1"], masks_v["G172"]], ["G0", "G1", "G172"], "UNKNOWN"),
        "small_raw": pva_sm, "small_calibrated": cal_va_s,
        "full_raw": pva_full, "full_calibrated": cal_va_f,
    })
    for cfg in ("A", "B", "C"):
        for s in (0, 1):
            out_tbl[f"{cfg}_seed{s}_calibrated"] = paired[f"{cfg}_seed{s}_calibrated"]
    out_tbl.to_csv(OUT / "valid_per_graph_aligned.csv", index=False)

    # ---- group CSV ----
    gcsv = []
    for name, m in models.items():
        for r in groups[name]["valid"]:
            gcsv.append({"model": name, "split": "valid", **r})
    for name in ("Small", "Full"):
        for r in train_groups[name]:
            gcsv.append({"model": name, "split": "train", **r})
    pd.DataFrame(gcsv).to_csv(OUT / "group_table.csv", index=False)

    summary = {
        "before_after": {
            "Small": {"raw_train": models["Small"]["raw_tr"], "cal_train": mae(cal_tr_s, ytr),
                      "raw_valid": models["Small"]["raw_va"], "cal_valid": mae(cal_va_s, yva),
                      "bias_train_median": b_s},
            "Full": {"raw_train": models["Full"]["raw_tr"], "cal_train": mae(cal_tr_f, ytr),
                     "raw_valid": models["Full"]["raw_va"], "cal_valid": mae(cal_va_f, yva),
                     "bias_train_median": b_f},
        },
        "published": PUB,
        "gain_full_vs_small": {
            "raw_valid_gain": mae(pva_sm, yva) - mae(pva_full, yva),
            "cal_valid_gain": mae(cal_va_s, yva) - mae(cal_va_f, yva),
            **full_vs_small,
        },
        "id172": id172, "join_valid": jv, "join_train": jt,
        "groups_valid": groups, "groups_train": train_groups,
        "probe_pair_gains": pair_gains,
        "gap": gap,
        "params": {"small": sr and 106925, "full": 408651,
                   "ratio": 408651 / 106925},
        "official_test_loaded": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("before_after", "gain_full_vs_small", "id172",
                                              "join_valid", "gap")}, indent=1, default=float))
    print("\nFull valid groups:")
    for r in groups["Full"]["valid"]:
        print(r)


if __name__ == "__main__":
    main()