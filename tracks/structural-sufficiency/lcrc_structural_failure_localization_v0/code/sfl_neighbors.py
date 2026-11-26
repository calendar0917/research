"""B1: label-blind Morgan r=2 nearest neighbours inside the official Train3360."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import sfl_common as C
import sfl_data as D


def morgan_matrix(smiles: list[str]) -> Any:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(C.mol_from_smiles(s)) for s in smiles]
    return fps, gen


def neighbours(base: D.Base, cases: pd.DataFrame, k: int = C.TOP_K) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = base.mol[base.mol.split == "train"].sort_values("row_id").reset_index(drop=True)
    err = base.errors.set_index("row_id")
    train_fps, gen = morgan_matrix(list(train["smiles"]))
    case_fps, _ = morgan_matrix(list(cases["smiles"]))

    from rdkit import DataStructs

    rows = []
    case_diag: list[dict[str, Any]] = []
    for ci, case in cases.reset_index(drop=True).iterrows():
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(case_fps[ci], train_fps), dtype=float)
        # label-blind: rank on similarity only, ties broken by ascending row id
        order = np.lexsort((train["row_id"].to_numpy(), -sims))
        top = order[:k]
        for rank, j in enumerate(top, start=1):
            r = train.iloc[int(j)]
            e = err.loc[int(r["row_id"])]
            rows.append(
                {
                    "case_row_id": int(case["row_id"]),
                    "rank": int(rank),
                    "neighbor_row_id": int(r["row_id"]),
                    "tanimoto_morgan_r2_2048": float(sims[int(j)]),
                    "neighbor_smiles": r["smiles"],
                    "neighbor_canonical_isomeric_smiles": r["canon_smiles"],
                    "neighbor_y_true": float(r["exp"]),
                    "neighbor_pred_A": float(e["pred_A"]),
                    "neighbor_pred_B": float(e["pred_B"]),
                    "neighbor_mol_logp": C.descriptors(r["smiles"])["mol_logp"],
                    "neighbor_candidate_families": "+".join(
                        C.acid_families(C.candidate_hits(r["smiles"]))
                    )
                    or "none",
                }
            )
        diag = {
            "case_row_id": int(case["row_id"]),
            "n_train": int(train.shape[0]),
            "n_sim_ge_0.6": int((sims >= C.SIM_HI).sum()),
            "n_sim_ge_0.8": int((sims >= C.SIM_VHI).sum()),
            "max_sim": float(sims.max()),
            "median_sim": float(np.median(sims)),
            "n_exact_duplicate_canonical_in_train": int(
                (train["canon_smiles"].to_numpy() == case["canonical_isomeric_smiles"]).sum()
            ),
            "n_fp_equivalent_different_molecule": int(
                sum(
                    1
                    for j, fp in enumerate(train_fps)
                    if sims[j] >= 1.0 - 1e-12
                    and train["canon_smiles"].iat[j] != case["canonical_isomeric_smiles"]
                )
            ),
            "top1_sim": float(sims[top[0]]) if len(top) else None,
        }
        case_diag.append(diag)
    df = pd.DataFrame(rows)
    diag_df = pd.DataFrame(case_diag)
    # global duplicate audit inside Train (independent of the cases)
    canon = train["canon_smiles"].to_numpy()
    fp_groups: dict[str, list[int]] = {}
    for j, fp in enumerate(train_fps):
        key = fp.ToBitString()
        fp_groups.setdefault(key, []).append(int(j))
    fp_collisions = [
        {"rows": [int(train["row_id"].iat[x]) for x in g],
         "canonical_smiles": [str(canon[x]) for x in g]}
        for g in fp_groups.values()
        if len(g) > 1 and len({str(canon[x]) for x in g}) > 1
    ]
    audit = {
        "n_train": int(train.shape[0]),
        "fingerprint": "Morgan radius=2, 2048 bits, RDKit default (useChirality=False)",
        "n_duplicate_canonical_smiles_in_train": int(pd.Series(canon).duplicated().sum()),
        "n_fp_equivalent_groups_with_different_smiles": int(len(fp_collisions)),
        "fp_equivalent_groups": fp_collisions[:10],
        "cases": diag_df.to_dict(orient="records"),
    }
    return df, audit
