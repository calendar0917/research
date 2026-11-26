"""B2 + C2: strict real structural edit pairs, re-restricted to the official split.

The pair enumeration itself is *reused read-only* from
``lcrc_lipo_edit_supervision_public122_s0_v1``:

    one single non-ring bond cut, variable fragment of 1-4 heavy atoms,
    core >= 10 heavy atoms covering >= 75 % of each parent,
    explicit labelled-graph isomorphism with an attachment marker,
    verified per pair (``verify_mapping``) in the source round.

This round does **not** relax any criterion.  It only

1. restricts the pair universe to the official Train3360 / Valid420 molecules,
2. re-checks, per pair, that both endpoints are real dataset molecules whose
   stored canonical isomeric SMILES differ,
3. re-checks that the reported variable atoms reproduce the reported variable
   fragment (element multiset, both orientations), which also fixes the
   a/b orientation used for every delta below,
4. attaches the frozen A/B predictions and the frozen candidate-structure tags.

Stereo: nothing is stripped.  Coverage of stereo-bearing molecules is reported
as-is, and the number of pairs whose endpoints carry stereocentres is counted.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import sfl_common as C
import sfl_data as D


def load_pairs() -> tuple[pd.DataFrame, Mapping[str, Any]]:
    pairs = pd.read_csv(C.EDITSUP_PAIRS)
    mappings = C.read_json(C.EDITSUP_DIR / "analysis/pair_mappings.json")
    lut = {str(p["pair_id"]): p for p in mappings["pairs"]}
    return pairs, lut


def _elements(smiles: str) -> list[str]:
    from rdkit import Chem

    m = C.mol_from_smiles(smiles)
    return sorted(a.GetSymbol() for a in m.GetAtoms())


def _fragment_elements(var_smiles: str) -> list[str]:
    """elements of a ``[1*]``-marked fragment SMILES, dummy atom excluded"""
    from rdkit import Chem

    m = Chem.MolFromSmiles(str(var_smiles))
    if m is None:
        return []
    return sorted(a.GetSymbol() for a in m.GetAtoms() if a.GetAtomicNum() != 0)


def build(base: D.Base) -> tuple[pd.DataFrame, dict[str, Any]]:
    pairs, lut = load_pairs()
    split_of = dict(zip(base.mol.row_id, base.mol.split))
    canon = base.mol.set_index("row_id")["canon_smiles"]
    smiles = base.mol.set_index("row_id")["smiles"]
    ylab = base.mol.set_index("row_id")["exp"]
    err = base.errors.set_index("row_id")
    pats = C.frozen_patterns()

    # The pair universe is restricted to official Train/Valid endpoints.  Pairs
    # with a Test endpoint are dropped *without* reading any label or prediction
    # for that molecule (the official scaffold split, not the source round's
    # random122 split, decides membership).
    all_pairs = pairs
    mask = pairs.row_a.isin(split_of) & pairs.row_b.isin(split_of)
    keep = pairs[mask].copy()

    records = []
    checks = {"n_source_pairs": int(len(all_pairs)), "n_train_valid_pairs": int(len(keep)),
              "n_source_pairs_with_test_endpoint_dropped": int((~mask).sum()),
              "orientation_mismatch": 0, "fragment_element_mismatch": [], "canonical_equal": 0,
              "sterocentre_molecules_skipped": 0}

    for _, p in keep.iterrows():
        pid = str(p["pair_id"])
        mp = lut.get(pid)
        if mp is None:
            checks["fragment_element_mismatch"].append({"pair_id": pid, "reason": "no mapping"})
            continue
        ra, rb = int(mp["row_a"]), int(mp["row_b"])
        if (ra, rb) != (int(p["row_a"]), int(p["row_b"])):
            checks["orientation_mismatch"] += 1
        if str(canon.loc[ra]) == str(canon.loc[rb]):
            checks["canonical_equal"] += 1
            continue
        # which mapping orientation reproduces the reported variable fragments?
        frag_a = _elements(smiles.loc[ra])
        use = None
        for swap in (False, True):
            va = mp["var_atoms_a"] if not swap else mp["var_atoms_b"]
            vb = mp["var_atoms_b"] if not swap else mp["var_atoms_a"]
            ea = _fragment_elements(p["var_a"] if not swap else p["var_b"])
            eb = _fragment_elements(p["var_b"] if not swap else p["var_a"])
            from rdkit import Chem

            ma = C.mol_from_smiles(smiles.loc[ra])
            mb = C.mol_from_smiles(smiles.loc[rb])
            el_a = sorted(ma.GetAtomWithIdx(int(i)).GetSymbol() for i in va)
            el_b = sorted(mb.GetAtomWithIdx(int(i)).GetSymbol() for i in vb)
            if el_a == ea and el_b == eb:
                use = swap
                break
        if use is None:
            checks["fragment_element_mismatch"].append({"pair_id": pid, "reason": "elements"})
            continue
        var_a_atoms = mp["var_atoms_a"] if not use else mp["var_atoms_b"]
        var_b_atoms = mp["var_atoms_b"] if not use else mp["var_atoms_a"]
        var_a = p["var_a"] if not use else p["var_b"]
        var_b = p["var_b"] if not use else p["var_a"]

        ha = C.candidate_hits(smiles.loc[ra], pats)
        hb = C.candidate_hits(smiles.loc[rb], pats)
        fam_a = C.acid_families(ha)
        fam_b = C.acid_families(hb)
        gained = sorted(set(fam_b) - set(fam_a))
        lost = sorted(set(fam_a) - set(fam_b))

        # is the gained/lost acidic family actually *at the changed connectivity*?
        def _atoms_of(fams: Sequence[str], hits: Mapping[str, Sequence[int]]) -> set[int]:
            out: set[int] = set()
            for fam, keys in C.ACID_FAMILIES:
                if fam in fams:
                    for k in keys:
                        out.update(int(x) for x in hits.get(k, []))
            return out

        gained_atoms = _atoms_of(gained, hb)
        lost_atoms = _atoms_of(lost, ha)
        gained_in_var = bool(gained_atoms & {int(x) for x in var_b_atoms})
        lost_in_var = bool(lost_atoms & {int(x) for x in var_a_atoms})

        # direction normalisation: "a" = no-acid side, "b" = acid side.  Both a
        # gained acid and a lost acid therefore code the *same* chemical
        # direction (more acidic group at the changed connection site).
        raw_direction = "+".join(gained) + "_APPEARS" if gained and not lost else (
            "+".join(lost) + "_DISAPPEARS" if lost and not gained else
            ("ACID_FAMILY_SWAP" if (gained and lost) else "NO_ACID_CHANGE")
        )
        if gained and not lost:
            orow_a, orow_b, h_a, h_b, fam_na, fam_nb = ra, rb, ha, hb, fam_a, fam_b
        elif lost and not gained:
            orow_a, orow_b, h_a, h_b, fam_na, fam_nb = rb, ra, hb, ha, fam_b, fam_a
        else:
            orow_a, orow_b, h_a, h_b, fam_na, fam_nb = ra, rb, ha, hb, fam_a, fam_b
        direction = "ACID_APPEARS" if (gained or lost) else "NO_ACID_CHANGE"
        triester = ("P_TRIESTER" in ha) or ("P_TRIESTER" in hb)
        phospho_free_acid = ("P_OH" in ha) or ("P_OH" in hb) or ("P_O_MINUS" in ha) or ("P_O_MINUS" in hb)

        da = C.descriptors(smiles.loc[orow_a])
        db = C.descriptors(smiles.loc[orow_b])
        ester_to_acid = bool(("C_ESTER" in h_a) and (not fam_na) and fam_nb)
        delta_y = float(ylab.loc[orow_b] - ylab.loc[orow_a])
        delta_A = float(err.loc[orow_b, "pred_A"] - err.loc[orow_a, "pred_A"])
        delta_B = float(err.loc[orow_b, "pred_B"] - err.loc[orow_a, "pred_B"])

        records.append(
            {
                "pair_id": pid,
                "split_a": split_of[int(orow_a)],
                "split_b": split_of[int(orow_b)],
                # NOTE: the source file's split label belongs to the random122
                # split of that round and is NOT the official scaffold split.
                # Every split-dependent number below uses split_of (official).
                "source_split_combo_random122": p["split_combo"],
                "official_split_combo": (
                    "Train-Train" if (split_of[int(orow_a)] == split_of[int(orow_b)] == "train")
                    else "Valid-Valid" if (split_of[int(orow_a)] == split_of[int(orow_b)] == "valid")
                    else "Train-Valid"
                ),
                "row_a": int(orow_a),
                "row_b": int(orow_b),
                "row_a_is_lower_row_id": int(orow_a < orow_b),
                "smiles_a": smiles.loc[orow_a],
                "smiles_b": smiles.loc[orow_b],
                "direction": direction,
                "raw_direction": raw_direction,
                "gained_families": "+".join(gained) or "",
                "lost_families": "+".join(lost) or "",
                "gained_acid_in_changed_fragment": int(gained_in_var) if gained else -1,
                "lost_acid_in_changed_fragment": int(lost_in_var) if lost else -1,
                "ester_to_acid": int(ester_to_acid),
                "phospho_acid": int(phospho_free_acid),
                "p_triester_endpoint": int(triester),
                "core_smiles": p["core_smiles"],
                "core_heavy": int(p["core_heavy"]),
                "var_a": var_a,
                "var_b": var_b,
                "var_heavy_a": int(p.get("var_heavy_a", len(var_a_atoms))),
                "var_heavy_b": int(p.get("var_heavy_b", len(var_b_atoms))),
                "var_atoms_a": ",".join(str(int(x)) for x in var_a_atoms),
                "var_atoms_b": ",".join(str(int(x)) for x in var_b_atoms),
                "attach_a": int(mp["attach_a"]),
                "attach_b": int(mp["attach_b"]),
                "y_a": float(ylab.loc[orow_a]),
                "y_b": float(ylab.loc[orow_b]),
                "pred_A_a": float(err.loc[orow_a, "pred_A"]),
                "pred_A_b": float(err.loc[orow_b, "pred_A"]),
                "pred_B_a": float(err.loc[orow_a, "pred_B"]),
                "pred_B_b": float(err.loc[orow_b, "pred_B"]),
                "delta_y": delta_y,
                "delta_A": delta_A,
                "delta_B": delta_B,
                "response_error_A": float(delta_A - delta_y),
                "response_error_B": float(delta_B - delta_y),
                "mol_logp_a": da["mol_logp"],
                "mol_logp_b": db["mol_logp"],
                "delta_mol_logp": float(db["mol_logp"] - da["mol_logp"]),
                "tpsa_a": da["tpsa"],
                "tpsa_b": db["tpsa"],
                "n_heavy_a": da["n_heavy"],
                "n_heavy_b": db["n_heavy"],
                "formed_charge_a": da["formal_charge"],
                "formed_charge_b": db["formal_charge"],
                "n_stereo_centers_a": da["n_stereo_centers"],
                "n_stereo_centers_b": db["n_stereo_centers"],
                "changed_fragment_family_pair": f"{var_a}=>{var_b}",
                "class_id_source": p["class_id"],
            }
        )

    df = pd.DataFrame(records)
    checks["n_pairs_analysed"] = int(len(df))
    if len(df):
        checks["official_split_combo_counts"] = df.official_split_combo.value_counts().to_dict()
        checks["n_source_split_labels_disagreeing_with_official"] = int(
            (
                df.source_split_combo_random122.map(
                    {"Train-Train": "Train-Train", "Train-Valid": "Train-Valid", "Valid-Valid": "Valid-Valid"}
                )
                != df.official_split_combo
            ).sum()
        )
    checks["n_pairs_with_gained_acid_in_changed_fragment"] = int(
        ((df["gained_acid_in_changed_fragment"] == 1)).sum()
    ) if len(df) else 0
    checks["n_pairs_stereo_bearing"] = int(
        ((df["n_stereo_centers_a"] > 0) | (df["n_stereo_centers_b"] > 0)).sum()
    ) if len(df) else 0
    checks["n_pairs_forming_charge"] = int(
        ((df["formed_charge_a"] != 0) | (df["formed_charge_b"] != 0)).sum()
    ) if len(df) else 0
    checks["source_file"] = C.rel(C.EDITSUP_PAIRS)
    checks["source_sha256"] = C.sha256_file(C.EDITSUP_PAIRS)
    checks["mapping_file_sha256"] = C.sha256_file(C.EDITSUP_DIR / "analysis/pair_mappings.json")
    return df, checks


# ---------------------------------------------------------------------------
# C2: group a same-direction change and test whether it repeats
# ---------------------------------------------------------------------------


def group_stats(df: pd.DataFrame, name: str, subset: pd.DataFrame) -> dict[str, Any]:
    if subset.empty:
        return {"group": name, "n_pairs": 0, "status": "NO_MATCHED_ANALOG"}
    dy = subset["delta_y"].to_numpy(dtype=float)
    dA = subset["delta_A"].to_numpy(dtype=float)
    dB = subset["delta_B"].to_numpy(dtype=float)
    cores = subset["core_smiles"].tolist()
    edges = list(zip(subset["row_a"].astype(int), subset["row_b"].astype(int)))
    nodes = sorted(set(subset["row_a"].astype(int)) | set(subset["row_b"].astype(int)))
    # direction consistency: fraction of pairs whose delta_y has the group's majority sign
    med = float(np.median(dy))
    sign = np.sign(med) if med != 0 else 0.0
    same = int((np.sign(dy) == sign).sum())
    return {
        "group": name,
        "status": "MATCHED_ANALOG",
        "n_pairs": int(len(subset)),
        "n_distinct_cores": int(len(set(cores))),
        "n_shared_endpoints_components": int(C.components(nodes, edges)),
        "n_molecules": int(len(nodes)),
        "n_train_pairs": int((subset.split_a.eq("train") & subset.split_b.eq("train")).sum()),
        "delta_y_median": float(np.median(dy)),
        "delta_y_q1": float(np.percentile(dy, 25)),
        "delta_y_q3": float(np.percentile(dy, 75)),
        "delta_y_iqr": float(np.percentile(dy, 75) - np.percentile(dy, 25)),
        "delta_y_min": float(dy.min()),
        "delta_y_max": float(dy.max()),
        "delta_y_sign_consistency": float(same / len(dy)),
        "delta_y_majority_sign": float(sign),
        "delta_A_median": float(np.median(dA)),
        "delta_B_median": float(np.median(dB)),
        "response_error_A_median": float(np.median(dA - dy)),
        "response_error_B_median": float(np.median(dB - dy)),
        "response_error_A_iqr": float(np.percentile(dA - dy, 75) - np.percentile(dA - dy, 25)),
        "response_error_B_iqr": float(np.percentile(dB - dy, 75) - np.percentile(dB - dy, 25)),
        "n_pairs_A_right_sign": int((np.sign(dA) == sign).sum()),
        "n_pairs_B_right_sign": int((np.sign(dB) == sign).sum()),
        "n_pairs_A_larger_than_true": int((np.abs(dA) > np.abs(dy)).sum()),
        "n_pairs_B_larger_than_true": int((np.abs(dB) > np.abs(dy)).sum()),
        "n_pairs_A_smaller_than_true": int((np.abs(dA) < np.abs(dy)).sum()),
        "n_pairs_B_smaller_than_true": int((np.abs(dB) < np.abs(dy)).sum()),
        "delta_mol_logp_median": float(np.median(subset["delta_mol_logp"].to_numpy(dtype=float))),
        "counterexamples": [
            {
                "pair_id": str(r.pair_id),
                "row_a": int(r.row_a),
                "row_b": int(r.row_b),
                "delta_y": float(r.delta_y),
                "delta_A": float(r.delta_A),
                "delta_B": float(r.delta_B),
                "core_smiles": str(r.core_smiles),
            }
            for r in subset.itertuples()
            if np.sign(r.delta_y) != sign
        ][:10],
    }


def c2_groups(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Same-direction groups.  Direction is frozen: more acidic group at the
    changed single connection site (``direction == ACID_APPEARS``)."""
    out: list[dict[str, Any]] = []
    appears = df[df.direction == "ACID_APPEARS"]
    at_site = appears[
        (appears.gained_acid_in_changed_fragment == 1) | (appears.lost_acid_in_changed_fragment == 1)
    ]
    for name, sub in (
        ("ANY_ACID_APPEARS", at_site),
        ("CARBOXYL_APPEARS", at_site[at_site.gained_families.eq("CARBOXYL") | at_site.lost_families.eq("CARBOXYL")]),
        ("PHOSPHO_ACID_APPEARS", at_site[at_site.gained_families.str.contains("PHOSPHO") | at_site.lost_families.str.contains("PHOSPHO")]),
        ("SULFONIC_APPEARS", at_site[at_site.gained_families.str.contains("SULFONIC") | at_site.lost_families.str.contains("SULFONIC")]),
        ("ESTER_TO_CARBOXYL", at_site[at_site.ester_to_acid == 1]),
    ):
        out.append(group_stats(df, name, sub))
    return out
