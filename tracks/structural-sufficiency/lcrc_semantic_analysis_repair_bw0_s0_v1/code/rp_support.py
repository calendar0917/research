"""Corrected support / document-frequency statistics (one correct entry point).

Rebuilds the per-molecule occurrence statistics with :mod:`rp_reader` for the
N100 / N50 / N25 Fit partitions, the Fit-LOO document-frequency thresholds, the
per-molecule low-support joint ratio ``r_joint`` and the frozen-tercile risk
bins.  Writes:

* ``analysis/support_corrected.json``      -- per budget/split statistics + verdicts
* ``analysis/support_before_after.csv``    -- the old (buggy) counters vs corrected
* ``analysis/support_molecules.csv``       -- per-molecule row_id / Fit / r_joint / sizes

The old ``p0_audit.json`` numbers are read **only** to build the before/after
table; they are never used to compute a new statistic.
"""

from __future__ import annotations

import csv
from typing import Any

import numpy as np

import rp_common as RP
import rp_reader as RR

BUDGET_FRACTION = {"N100": 1.0, "N50": 0.5, "N25": 0.25}


def _fit_positions(view, budget: str) -> list[int]:
    import se_data as SED

    train_rows = view.rows["train"].tolist()
    if budget == "N100":
        fit = train_rows
    else:
        fit = SED.nested_fit_rows(train_rows, BUDGET_FRACTION[budget])
    pos = {int(r): i for i, r in enumerate(train_rows)}
    return [pos[int(r)] for r in fit]


def _terciles(r_joint: np.ndarray) -> dict[str, Any]:
    if r_joint.size == 0:
        return {"q1": None, "q2": None, "degenerate": True, "reason": "no molecules"}
    q1, q2 = np.quantile(r_joint, [1 / 3, 2 / 3])
    degenerate = bool(float(q1) == float(q2))
    return {
        "q1": float(q1),
        "q2": float(q2),
        "degenerate": degenerate,
        "reason": "equal tercile thresholds (ties dominate) -> bins not estimable" if degenerate else "ok",
        "n": int(r_joint.size),
        "n_nonzero": int((r_joint > 0).sum()),
        "max": float(r_joint.max()),
    }


def _old_numbers() -> dict[str, Any]:
    p = RP.OLD / "analysis/p0_audit.json"
    return RP.read_json(p) if p.is_file() else {}


def run(view=None) -> dict[str, Any]:
    import lp_data as LPD

    view = view or LPD.build_view()
    train = view.caches["train"]
    corrected: dict[str, Any] = {}
    for budget in ("N100", "N50", "N25"):
        fit_pos = _fit_positions(view, budget)
        df = RR.document_frequency_subset(train, fit_pos, name=f"train_{budget}")
        fit_set = set(fit_pos)
        out: dict[str, Any] = {
            "budget": budget,
            "n_fit": len(fit_pos),
            "df_reference_size": {
                "path": len(df["df_path"]),
                "end": len(df["df_end"]),
                "nbr": len(df["df_nbr"]),
                "joint": len(df["df_joint"]),
            },
            "fit_df_paths": df["n_paths"],
            "fit_df_s1": df["n_s1"],
            "splits": {},
        }
        for split in RP.SPLITS:
            cache = view.caches[split]
            if split == "train":
                st = RR.support_stats(cache, df, loo_positions=fit_set, split=f"train[{budget}]")
            else:
                st = RR.support_stats(cache, df, loo_positions=None, split=split)
            r = np.asarray(st.pop("r_joint"), dtype=np.float64)
            terc = _terciles(r) if split == "train" else None
            st["r_joint"] = {
                "n_molecules": int(r.size),
                "n_nonzero": int((r > 0).sum()),
                "frac_molecules_zero": float((r == 0).mean()) if r.size else None,
                "mean": float(r.mean()) if r.size else None,
                "max": float(r.max()) if r.size else None,
            }
            if terc is not None:
                st["terciles"] = terc
            out["splits"][split] = st
        # risk bins: only if the Train-LOO terciles are non-degenerate
        terc = out["splits"]["train"]["terciles"]
        if terc.get("degenerate"):
            out["risk_bins"] = {
                "status": "NOT_ESTIMABLE",
                "reason": terc.get("reason"),
                "q1": terc.get("q1"),
                "q2": terc.get("q2"),
                "note": "frozen terciles keep ties; no alternative quantile/threshold was searched",
            }
        else:
            out["risk_bins"] = {"status": "ESTIMABLE", "q1": terc["q1"], "q2": terc["q2"]}
        corrected[budget] = out

    payload = {
        "schema": "lcrc-semantic-analysis-repair-support-corrected-v1",
        "protocol_id": RP.PROTOCOL_ID,
        "created_utc": RP.utcnow(),
        "git_commit": RP.git_commit(),
        "reader": "rp_reader.MoleculeReader (atom_ptr -> path_ptr/nbr_ptr, atom-level occurrences)",
        "low_support_df": RP.LOW_SUPPORT_DF,
        "budgets": corrected,
        "layout_train": RR.check_layout(train, name="train"),
        "notes": {
            "df_definition": "per-molecule document frequency; each code counts once per molecule",
            "fit_loo": "Fit molecules that built df subtract their own contribution (support=df-1)",
            "joint": "joint = the same (path,end) occurrence pair, never a cartesian product",
            "has_no_path": "molecules with zero S2 paths are reported separately, not as low risk",
        },
    }
    RP.write_json(RP.ANALYSIS / "support_corrected.json", payload)

    # ---- before / after (old buggy counters vs corrected, N100) -----------
    old = _old_numbers()
    rows: list[dict[str, Any]] = []
    n100 = corrected["N100"]
    if old:
        osup = old.get("support_counts", {})
        for split in RP.SPLITS:
            o = osup.get(split, {})
            c = n100["splits"][split]
            occ = c["occurrence_counts"]
            buckets = c["df_buckets_frac"]["path"]
            pairs = [
                ("path_tokens", o.get("path_tokens"), occ["path"]),
                ("nbr_tokens", o.get("nbr_tokens"), occ["nbr"]),
                ("end_tokens", o.get("end_tokens"), occ["end"]),
                ("joint_tokens", o.get("joint_tokens"), occ["joint"]),
                ("path_df1_5_frac", o.get("path_df1_5_frac"), buckets["1-5"]),
                ("path_df6_20_frac", o.get("path_df6_20_frac"), buckets["6-20"]),
                ("path_df_gt20_frac", o.get("path_df_gt20_frac"), buckets[">20"]),
                ("path_unseen_frac", o.get("path_unseen_frac"), buckets["0"]),
                ("joint_unseen_frac", o.get("joint_unseen_frac"), c["df_buckets_frac"]["joint"]["0"]),
                ("nbr_unseen_frac", o.get("nbr_unseen_frac"), c["df_buckets_frac"]["nbr"]["0"]),
            ]
            for metric, vold, vnew in pairs:
                rows.append(
                    {
                        "budget": "N100",
                        "split": split,
                        "metric": metric,
                        "old_value": vold,
                        "corrected_value": vnew,
                    }
                )
        for metric, key in (("path_df_size", "path_df_size"), ("nbr_df_size", "nbr_df_size"),
                            ("end_df_size", "end_df_size"), ("joint_df_size", "joint_df_size")):
            name = {"path_df_size": "path", "nbr_df_size": "nbr", "end_df_size": "end",
                    "joint_df_size": "joint"}[key]
            rows.append({"budget": "N100", "split": "reference_df", "metric": metric,
                         "old_value": osup.get(key), "corrected_value": n100["df_reference_size"][name]})
    out_csv = RP.ANALYSIS / "support_before_after.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["budget", "split", "metric", "old_value", "corrected_value"])
        w.writeheader()
        w.writerows(rows)

    # ---- per-molecule table ----------------------------------------------
    mol_rows: list[dict[str, Any]] = []
    train_rows = view.rows["train"].tolist()
    pos_to_row = {i: int(r) for i, r in enumerate(train_rows)}
    fit_set = set(_fit_positions(view, "N100"))
    df = RR.document_frequency_subset(train, fit_set, name="train_N100")
    reader = RR.MoleculeReader(train, name="train")
    r = np.zeros(reader.n_molecules)
    for m in range(reader.n_molecules):
        jp = reader.joint_pairs(m)
        if jp:
            low = sum(1 for pair in set(jp) if df["df_joint"].get(pair, 0) - (1 if m in fit_set else 0) <= RP.LOW_SUPPORT_DF)
            r[m] = low / len(jp)
    for m in range(reader.n_molecules):
        s = reader.sets(m)
        mol_rows.append(
            {
                "split": "train",
                "row_id": pos_to_row[m],
                "in_fit_N100": int(m in fit_set),
                "n_atoms": int(reader.atom_ptr[m + 1] - reader.atom_ptr[m]),
                "n_path": int(s["n_path"]),
                "n_s1": int(s["n_s1"]),
                "has_no_path": int(s["has_no_path"]),
                "r_joint": float(r[m]),
            }
        )
    mol_csv = RP.ANALYSIS / "support_molecules.csv"
    with open(mol_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mol_rows[0].keys()))
        w.writeheader()
        w.writerows(mol_rows)

    return {
        "artifacts": [
            RP.track_rel(RP.ANALYSIS / "support_corrected.json"),
            RP.track_rel(out_csv),
            RP.track_rel(mol_csv),
        ],
        "n100": {
            "df_reference_size": n100["df_reference_size"],
            "train_occurrences": n100["splits"]["train"]["occurrence_counts"],
            "train_terciles": n100["splits"]["train"]["terciles"],
            "risk_bins": n100["risk_bins"],
        },
    }


__all__ = [name for name in dir() if not name.startswith("_")]
