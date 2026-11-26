"""B3 + C1 + C3: frozen candidate-structure groups over the whole official split.

Only the frozen SMARTS of :mod:`sfl_common` are used.  No SMARTS is added after
looking at a result, no grouping threshold is searched, and the two size /
lipophilicity controls are *fixed at the Train median* before the group effect is
read.  Groups overlap; overlapping membership is reported explicitly and the
net squared-error gain is never double counted.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

import sfl_common as C
import sfl_data as D


def annotate(base: D.Base) -> pd.DataFrame:
    pats = C.frozen_patterns()
    factory, _ = C.basefeature_factory()
    rows = []
    for rec in base.mol.itertuples():
        hits = C.candidate_hits(rec.smiles, pats)
        bf = C.basefeature_hits(rec.smiles, factory)
        fams = C.acid_families(hits)
        desc = C.descriptors(rec.smiles)
        rows.append(
            {
                "row_id": int(rec.row_id),
                "split": rec.split,
                "smiles": rec.smiles,
                "canon_smiles": rec.canon_smiles,
                "y": float(rec.exp),
                "mol_logp": desc["mol_logp"],
                "y_minus_mol_logp": float(rec.exp) - desc["mol_logp"],
                "n_heavy": desc["n_heavy"],
                "n_P": desc["n_P"],
                "n_S": desc["n_S"],
                "tpsa": desc["tpsa"],
                "hbd": desc["hbd"],
                "hba": desc["hba"],
                "formal_charge": desc["formal_charge"],
                "n_charged_atoms": desc["n_charged_atoms"],
                "scaffold": C.scaffold_key(rec.smiles),
                "P_OH": int("P_OH" in hits),
                "P_O_MINUS": int("P_O_MINUS" in hits),
                "P_TRIESTER": int("P_TRIESTER" in hits),
                "C_ACID_OH": int("C_ACID_OH" in hits),
                "C_ACID_O_MINUS": int("C_ACID_O_MINUS" in hits),
                "C_ESTER": int("C_ESTER" in hits),
                "S_ACID_OH": int("S_ACID_OH" in hits),
                "S_ACID_O_MINUS": int("S_ACID_O_MINUS" in hits),
                "families": "+".join(fams) or "NONE",
                "PHOSPHO": int("PHOSPHO" in fams),
                "CARBOXYL": int("CARBOXYL" in fams),
                "SULFONIC": int("SULFONIC" in fams),
                "ANY_ACID": int(bool(fams)),
                "basefeature_NegIonizable": bf["NegIonizable"],
                "basefeature_PosIonizable": bf["PosIonizable"],
            }
        )
    df = pd.DataFrame(rows)
    return df.merge(base.errors[["row_id", "y_raw", "pred_A", "pred_B", "error_A", "error_B", "e_A2", "e_B2", "gain"]],
                    on="row_id", how="inner", validate="one_to_one")


#: frozen group predicates.  Ordered; overlap is reported, not resolved away.
GROUPS: dict[str, str] = {
    "ALL": "all molecules in the split",
    "PHOSPHO_ACID": "P_OH or P_O_MINUS (frozen SMARTS)",
    "PHOSPHO_TRIESTER_ONLY": "P_TRIESTER and no acidic P",
    "CARBOXYL_ACID": "C(=O)OH or C(=O)O-",
    "SULFONIC_ACID": "S(=O)2OH or S(=O)2O-",
    "ANY_ACID": "union of the three acidic families",
    "NO_FROZEN_ACID": "none of the three acidic families",
    "ESTER_ONLY": "C(=O)O-C present and no frozen acid family",
    "RDKIT_NEG_IONIZABLE": "RDKit BaseFeatures NegIonizable >= 1",
    "RDKIT_POS_IONIZABLE": "RDKit BaseFeatures PosIonizable >= 1",
}


def _mask(df: pd.DataFrame, name: str) -> pd.Series:
    return {
        "ALL": pd.Series(True, index=df.index),
        "PHOSPHO_ACID": (df.PHOSPHO == 1),
        "PHOSPHO_TRIESTER_ONLY": (df.P_TRIESTER == 1) & (df.PHOSPHO == 0),
        "CARBOXYL_ACID": (df.CARBOXYL == 1),
        "SULFONIC_ACID": (df.SULFONIC == 1),
        "ANY_ACID": (df.ANY_ACID == 1),
        "NO_FROZEN_ACID": (df.ANY_ACID == 0),
        "ESTER_ONLY": (df.C_ESTER == 1) & (df.ANY_ACID == 0),
        "RDKIT_NEG_IONIZABLE": (df.basefeature_NegIonizable >= 1),
        "RDKIT_POS_IONIZABLE": (df.basefeature_PosIonizable >= 1),
    }[name]


def _stats(d: pd.DataFrame, split: str, group: str) -> dict[str, Any]:
    if d.empty:
        return {
            "group": group, "split": split, "n_molecules": 0, "status": "EMPTY",
        }
    y = d["y"].to_numpy(dtype=float)
    ylp = d["y_minus_mol_logp"].to_numpy(dtype=float)
    lp = d["mol_logp"].to_numpy(dtype=float)
    return {
        "group": group,
        "split": split,
        "status": "OK",
        "n_molecules": int(len(d)),
        "pct_of_split": float(100.0 * len(d) / (3360 if split == "train" else 420)),
        "n_distinct_scaffolds": int(d["scaffold"].nunique()),
        "y_median": float(np.median(y)),
        "y_q1": float(np.percentile(y, 25)),
        "y_q3": float(np.percentile(y, 75)),
        "mol_logp_median": float(np.median(lp)),
        "y_minus_mol_logp_median": float(np.median(ylp)),
        "y_minus_mol_logp_q1": float(np.percentile(ylp, 25)),
        "y_minus_mol_logp_q3": float(np.percentile(ylp, 75)),
        "n_heavy_median": float(np.median(d["n_heavy"].to_numpy(dtype=float))),
        "tpsa_median": float(np.median(d["tpsa"].to_numpy(dtype=float))),
        "err_A_mean": float(d["error_A"].mean()),
        "err_B_mean": float(d["error_B"].mean()),
        "err_A_median": float(d["error_A"].median()),
        "err_B_median": float(d["error_B"].median()),
        "rmse_A": C.rmse(d["error_A"]),
        "rmse_B": C.rmse(d["error_B"]),
        "mae_A": float(d["error_A"].abs().mean()),
        "mae_B": float(d["error_B"].abs().mean()),
        "sum_sq_err_A": float(d["e_A2"].sum()),
        "sum_sq_err_B": float(d["e_B2"].sum()),
        "sum_sqerr_A_minus_B": float(d["gain"].sum()),
        "n_nonzero_formal_charge": int((d["n_charged_atoms"] > 0).sum()),
    }


def group_summary(df: pd.DataFrame, controls: dict[str, float]) -> pd.DataFrame:
    rows = []
    for split in ("train", "valid"):
        d = df[df.split == split]
        for g in GROUPS:
            m = _mask(d, g)
            r = _stats(d[m], split, g)
            r["criterion"] = GROUPS[g]
            # frozen size / lipophilicity control (Train medians)
            r["median_n_heavy_cut"] = controls["n_heavy_median_train"]
            r["median_mol_logp_cut"] = controls["mol_logp_median_train"]
            for lab, cond in (
                ("small", d["n_heavy"] <= controls["n_heavy_median_train"]),
                ("large", d["n_heavy"] > controls["n_heavy_median_train"]),
                ("low_logp", d["mol_logp"] <= controls["mol_logp_median_train"]),
                ("high_logp", d["mol_logp"] > controls["mol_logp_median_train"]),
            ):
                r[f"n_{lab}"] = int((m & cond).sum())
                if int((m & cond).sum()) > 0:
                    r[f"errA_mean_{lab}"] = float(d[m & cond]["error_A"].mean())
                    r[f"errB_mean_{lab}"] = float(d[m & cond]["error_B"].mean())
            rows.append(r)

    # explicit overlap matrix for the Train split (counts, both directions)
    tr = df[df.split == "train"]
    for a in GROUPS:
        for b in GROUPS:
            if GROUPS[a] < GROUPS[b]:
                continue
            n = int((_mask(tr, a) & _mask(tr, b)).sum())
            rows.append(
                {
                    "group": f"OVERLAP::{a}&{b}",
                    "split": "train",
                    "status": "OVERLAP",
                    "n_molecules": n,
                    "criterion": "intersection size (never added into any net gain)",
                }
            )
    return pd.DataFrame(rows)


def train_median_controls(df: pd.DataFrame) -> dict[str, float]:
    tr = df[df.split == "train"]
    return {
        "n_heavy_median_train": float(np.median(tr["n_heavy"].to_numpy(dtype=float))),
        "mol_logp_median_train": float(np.median(tr["mol_logp"].to_numpy(dtype=float))),
    }


def c3_valid_cases(df: pd.DataFrame, cases: pd.DataFrame) -> dict[str, Any]:
    """Do the frozen candidate groups cover the frozen case sets, and is there a
    same-direction prediction bias in the full Valid split?"""
    out: dict[str, Any] = {}
    for setname, col in (("GAIN5", "in_gain5"), ("LOSS5", "in_loss5"), ("COMMON5", "in_common5"), ("UNION", None)):
        sub = cases if col is None else cases[cases[col] == 1]
        out[setname] = {
            "n_cases": int(len(sub)),
            "rows": [int(x) for x in sub["row_id"]],
            "with_PHOSPHO": int(sub["PHOSPHO"] == 1) if "PHOSPHO" in sub else int(
                sub["candidate_families"].str.contains("PHOSPHO").sum()
            ),
            "with_ANY_acid": int(sub["candidate_families"].ne("none").sum()),
            "rows_with_acid": [
                int(r.row_id) for r in sub.itertuples() if str(r.candidate_families) != "none"
            ],
            "n_rdkit_neg_ionizable": int((sub["basefeature_NegIonizable"] >= 1).sum()),
            "n_rdkit_pos_ionizable": int((sub["basefeature_PosIonizable"] >= 1).sum()),
        }

    valid = df[df.split == "valid"]
    # y-quartile stratification is computed on the Valid split itself (descriptive)
    valid = valid.assign(y_quartile=pd.qcut(valid["y"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"]))
    rows = []
    for (g, q), d in valid.groupby([_mask(valid, "ANY_ACID").map({True: "ANY_ACID", False: "NO_ACID"}), "y_quartile"], observed=True):
        rows.append(
            {
                "group": g,
                "y_quartile": str(q),
                "n": int(len(d)),
                "y_mean": float(d["y"].mean()),
                "errA_mean": float(d["error_A"].mean()),
                "errB_mean": float(d["error_B"].mean()),
                "rmse_A": C.rmse(d["error_A"]),
                "rmse_B": C.rmse(d["error_B"]),
                "sum_sqerr_A_minus_B": float(d["gain"].sum()),
            }
        )
    out["valid_y_quartile_by_acid"] = rows
    out["valid_any_acid_rows"] = [int(x) for x in valid[valid.ANY_ACID == 1]["row_id"]]
    out["valid_phospho_rows"] = [int(x) for x in valid[valid.PHOSPHO == 1]["row_id"]]
    out["note"] = (
        "Valid has already been used in the discovery step; these are not "
        "independent confirmations."
    )
    return out
