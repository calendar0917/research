"""Phase D runner (ISRD-v1): the aromatic carbon–sulfur family, conditional check, cases.

No new structure family is searched and no other FULL subtype gets its own model.
D1 is descriptive within COARSE-419-positive molecules; D2 adds exactly two tiny
logistic models (CS-BASE / CS-FINE) on that subset; D3 shows a few real molecules.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

import isrd1_common as K
import isrd1_rules as R

CASE_COARSE = 419
CASE_FULL = (2078, 2525, 1788, 2275)
C_FIXED = 0.03


def molecules():
    sys.path.insert(0, str(K.MSDS / "code"))
    import msds_data as D

    return D.load_molecules()


def presence(data: K.Data, gran: str, col: int) -> np.ndarray:
    return data.col(gran, col) > 0


def split_stats(data: K.Data, mask: np.ndarray, base_mask: np.ndarray | None = None) -> dict:
    out = {}
    for s in K.SPLITS:
        r = data.rows[s]
        m = mask[r]
        if base_mask is not None:
            keep = base_mask[r]
            m = m & keep
        y = data.y[r]
        out[s] = {"n": int(m.sum()), "n_positive": int(y[m].sum()),
                  "positive_rate": float(y[m].mean()) if m.sum() else None,
                  "n_scaffolds": int(np.unique(data.scaffold_code[r][m]).size),
                  "n_atoms_median": float(np.median(data.n_atoms[r][m])) if m.sum() else None}
    return out


def b0_logit(data: K.Data) -> tuple[np.ndarray, dict]:
    logit = np.full(data.y.size, np.nan)
    with np.load(K.B2_PRED) as z:
        for k in K.SPLITS:
            logit[data.rows[k]] = z[f"B0|0.03|{k}"].astype(np.float64)
    rec = K.read_json(K.MSDS / "analysis" / "phaseB_linear.json")
    info = {"split_coverage": {k: int(np.isfinite(logit[data.rows[k]]).sum()) for k in K.SPLITS},
            "finite_all": bool(np.all(np.isfinite(logit))),
            "source": K.rel(K.B2_PRED)}
    return logit, info


def subset_model(data: K.Data, rows_sub: dict[str, np.ndarray], X_full: np.ndarray,
                 y: np.ndarray, name: str) -> dict[str, Any]:
    """Fit one fixed-C logistic on the COARSE-419 subset (Train only) and evaluate."""
    tr = rows_sub["train"]
    sw = K.class_balanced_weights(y[tr].astype(np.float64))
    mu = X_full[tr].mean(axis=0)
    sd = X_full[tr].std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X_full - mu) / sd
    fit = K.fit_logistic(Z[tr], y[tr].astype(np.float64), sw, C_FIXED)
    logit = fit["intercept"] + Z @ fit["coef"]
    out = {"arm": name, "C": C_FIXED, "intercept": fit["intercept"],
           "coef_standardised": fit["coef"], "coef_raw": fit["coef"] / sd,
           "intercept_raw": float(fit["intercept"] - np.sum(fit["coef"] * mu / sd)),
           "mean": mu, "sd": sd, "logit": logit, "converged": fit["converged"],
           "splits": {}}
    for s in K.SPLITS:
        r = rows_sub[s]
        m = K.metrics(logit[r], y[r])
        out["splits"][s] = {**m,
                            "evidence_label": "DEVELOPMENT_TEST" if s == "test" else s.upper()}
    return out


def describe_instance(mols, data: K.Data, g: int, nodes: list[int]) -> dict[str, Any]:
    import sga_chem as CHEM

    af = mols.atom_feat[int(mols.atom_offsets[g]): int(mols.atom_offsets[g + 1])]
    bu, bv, bf = mols.bonds(g)
    inside = set(nodes)
    atoms = []
    for i in nodes:
        d = CHEM.decode_atom(af[i], "FULL")
        atoms.append({"node": int(i), "element": d.get("symbol"), "aromatic": d.get("aromatic"),
                      "degree": d.get("degree"), "num_h": d.get("num_h"),
                      "hybridization": d.get("hybridization"), "in_ring": d.get("in_ring")})
    inner, outer = [], []
    for u, v, b in zip(bu, bv, bf):
        bd = CHEM.decode_bond(b)
        if u in inside and v in inside:
            inner.append({"u": int(u), "v": int(v), "bond": bd.get("bond_type")})
        elif u in inside or v in inside:
            other = int(v) if u in inside else int(u)
            od = CHEM.decode_atom(af[other], "FULL")
            outer.append({"inside": int(u if u in inside else v), "outside": other,
                          "outside_element": od.get("symbol"),
                          "outside_aromatic": od.get("aromatic"),
                          "bond": bd.get("bond_type")})
    return {"graph_index": int(g), "smiles": mols.smiles[g], "label": int(data.y[g]),
            "n_atoms": int(data.n_atoms[g]), "atoms": atoms, "internal_bonds": inner,
            "external_bonds": outer}


def main() -> None:
    sys.path.insert(0, str(K.SGA / "code"))
    data = K.Data()
    mols = molecules()
    reg = K.registry()
    with np.load(K.INSTANCES, allow_pickle=True) as z:
        examples = z["examples"]
    out: dict[str, Any] = {"family": {"coarse": CASE_COARSE, "full": list(CASE_FULL)},
                           "D1": {}, "D2": {}, "D3": {}}

    c419 = presence(data, "C", CASE_COARSE)
    full_pres = {c: presence(data, "F", c) for c in CASE_FULL}
    out["D1"]["n_coarse419"] = {s: int(c419[data.rows[s]].sum()) for s in K.SPLITS}
    out["D1"]["coarse419"] = split_stats(data, c419)
    for c in CASE_FULL:
        meta = reg["F"].iloc[c]
        dtid = int(meta["discovered_type_id"])
        pres = full_pres[c]
        rec = {
            "column": int(c), "canonical_key": str(meta["canonical_key"]),
            "family": str(meta["family"]),
            "node_signature": str(meta["node_signature"]),
            "bond_signature": str(meta["bond_signature"]),
            "element_sequence": str(meta["element_sequence"]),
            "num_h_sequence": str(meta["num_h_sequence"]),
            "aromatic_sequence": str(meta["aromatic_sequence"]),
            "degree_sequence": str(meta.get("node_signature")),
            "support_all": split_stats(data, pres & c419),
            "parent_is_superset": bool(np.all(c419[pres])),
            "jaccard_with_siblings": {
                str(c2): float((pres & full_pres[c2]).sum() / max(1, (pres | full_pres[c2]).sum()))
                for c2 in CASE_FULL if c2 != c},
            "n_atoms_present": {s: {"median": float(np.median(data.n_atoms[data.rows[s]][pres[data.rows[s]]]))
                                    if pres[data.rows[s]].sum() else None,
                                    "mean": float(data.n_atoms[data.rows[s]][pres[data.rows[s]]].mean())
                                    if pres[data.rows[s]].sum() else None}
                               for s in K.SPLITS},
            "examples": ([{"graph_index": int(g), "nodes": [int(x) for x in nd],
                           "label": int(data.y[g]), "split": next(
                               s for s in K.SPLITS if g in set(data.rows[s].tolist()))}
                          for g, nd in list(examples[dtid])[:3]]),
        }
        out["D1"][f"F{c}"] = rec

    # size-stratified association (within COARSE 419 positive molecules)
    tr = data.rows["train"]
    qs = np.percentile(data.n_atoms[tr], [25, 50, 75])
    bins = np.digitize(data.n_atoms, qs)
    size_tbl: dict[str, Any] = {"train_quantiles": [float(v) for v in qs],
                                "bins": ["<=q25", "q25-q50", "q50-q75", ">q75"]}
    for c in CASE_FULL:
        rows = []
        for b in range(4):
            for s in K.SPLITS:
                m = (bins[data.rows[s]] == b) & c419[data.rows[s]] & full_pres[c][data.rows[s]]
                y = data.y[data.rows[s]][m]
                rows.append({"bin": b, "split": s, "n": int(m.sum()),
                             "positive_rate": float(y.mean()) if y.size else None})
        size_tbl[f"F{c}"] = rows
    size_tbl["coarse419_all"] = [
        {"bin": b, "split": s,
         "n": int(((bins[data.rows[s]] == b) & c419[data.rows[s]]).sum()),
         "positive_rate": float(data.y[data.rows[s]][(bins[data.rows[s]] == b) & c419[data.rows[s]]].mean())}
        for b in range(4) for s in K.SPLITS]
    out["D1"]["size_stratified"] = size_tbl

    # existing-model performance inside each subtype subset
    b0, b0_info = b0_logit(data)
    b2 = np.zeros(data.y.size)
    with np.load(K.B2_PRED) as z:
        for k in K.SPLITS:
            b2[data.rows[k]] = z[f"B2|0.03|{k}"].astype(np.float64)
    out["D1"]["existing_models"] = {}
    for c in CASE_FULL:
        rec = {}
        for s in ("valid", "test"):
            r = data.rows[s]
            for grp, m in (("present", full_pres[c][r] & c419[r]),
                           ("absent", (~full_pres[c][r]) & c419[r])):
                if m.sum() == 0 or data.y[r][m].min() == data.y[r][m].max():
                    rec[f"{s}_{grp}"] = {"n": int(m.sum()), "auc": None}
                    continue
                rec[f"{s}_{grp}"] = {"n": int(m.sum()),
                                     "b0_auc": float(K.metrics(b0[r][m], data.y[r][m])["roc_auc"]),
                                     "b2_auc": float(K.metrics(b2[r][m], data.y[r][m])["roc_auc"]),
                                     "positive_rate": float(data.y[r][m].mean())}
        out["D1"]["existing_models"][f"F{c}"] = rec
    out["D2"]["b0"] = b0_info

    # ---------------- D2: CS-BASE / CS-FINE ---------------------------------
    sub = {s: data.rows[s][c419[data.rows[s]]] for s in K.SPLITS}
    out["D2"]["subset_sizes"] = {s: {"n": int(sub[s].size),
                                     "n_positive": int(data.y[sub[s]].sum())} for s in K.SPLITS}
    ok = (b0_info["finite_all"] and min(v["n"] for v in out["D2"]["subset_sizes"].values()) >= 50
          and int(data.y[sub["train"]].sum()) >= 30)
    if not ok:
        out["D2"]["status"] = "CONDITIONAL_CHECK_BLOCKED"
    else:
        size = np.log1p(data.n_atoms.astype(np.float64))
        X_base = np.column_stack([b0, size, data.col("C", CASE_COARSE).astype(np.float64)])
        X_fine = np.column_stack([X_base, full_pres[2078].astype(np.float64)])
        base = subset_model(data, sub, X_base, data.y, "CS-BASE")
        fine = subset_model(data, sub, X_fine, data.y, "CS-FINE")
        for name, arm in (("CS-BASE", base), ("CS-FINE", fine)):
            # the model is only defined on the COARSE-419 subset: mask every other row
            masked = np.full(data.y.size, np.nan)
            masked[c419] = arm["logit"][c419]
            np.savez_compressed(K.PREDICTIONS / f"{name}.npz",
                                logit=masked, y=data.y, split_code=data.split_code(),
                                graph_id=np.arange(data.y.size), scaffold=data.scaffold,
                                subset_mask=c419)
        boot = {}
        for s in ("valid", "test"):
            r = sub[s]
            if data.y[r].min() != data.y[r].max():
                boot[s] = K.scaffold_bootstrap(r, data.y, fine["logit"], base["logit"],
                                               data.scaffold_code, n_boot=1000)
        # present vs absent groups within the subset
        groups = {}
        for s in K.SPLITS:
            r = sub[s]
            p = full_pres[2078][r]
            groups[s] = {}
            for grp, m in (("present_2078", p), ("absent_2078", ~p)):
                yy = data.y[r][m]
                entry = {"n": int(m.sum()), "n_positive": int(yy.sum()),
                         "positive_rate": float(yy.mean()) if m.sum() else None,
                         "mean_prob_cs_base": float(K.sigmoid(base["logit"][r][m]).mean())
                         if m.sum() else None,
                         "mean_prob_cs_fine": float(K.sigmoid(fine["logit"][r][m]).mean())
                         if m.sum() else None}
                if m.sum() and yy.min() != yy.max():
                    entry["auc_cs_base"] = float(K.metrics(base["logit"][r][m], yy)["roc_auc"])
                    entry["auc_cs_fine"] = float(K.metrics(fine["logit"][r][m], yy)["roc_auc"])
                groups[s][grp] = entry
        out["D2"].update({
            "status": "EXECUTED",
            "CS-BASE": {k: v for k, v in base.items() if k != "logit"},
            "CS-FINE": {k: v for k, v in fine.items() if k != "logit"},
            "fine_coef_2078_standardised": float(fine["coef_standardised"][3]),
            "fine_coef_2078_raw": float(fine["coef_raw"][3]),
            "fine_coef_2078_standardised_se_approx": None,
            "bootstrap": boot,
            "groups": groups,
        })
        print(f"[D] CS-BASE valid={base['splits']['valid']['roc_auc']:.4f} "
              f"test={base['splits']['test']['roc_auc']:.4f}", flush=True)
        print(f"[D] CS-FINE valid={fine['splits']['valid']['roc_auc']:.4f} "
              f"test={fine['splits']['test']['roc_auc']:.4f}", flush=True)

    # ---------------- D3: cases --------------------------------------------
    cases: dict[str, Any] = {}
    p2078 = full_pres[2078]
    dtid = int(reg["F"].iloc[2078]["discovered_type_id"])
    ex = list(examples[dtid])
    for tag, want in (("support_positive", 1), ("support_negative", 0)):
        pick = next((g for g, nd in ex if data.y[g] == want), None)
        if pick is not None:
            nodes = next(nd for g, nd in ex if g == pick)
            cases[tag] = describe_instance(mols, data, pick, list(nodes))
    # same COARSE, different FULL subtypes
    variants = {}
    for c in CASE_FULL:
        if c == 2078:
            continue
        both = np.flatnonzero(c419 & full_pres[c])
        if both.size:
            g = int(both[0])
            dtid2 = int(reg["F"].iloc[c]["discovered_type_id"])
            nd = next((list(n) for gg, n in examples[dtid2] if gg == g), None)
            variants[f"F{c}"] = describe_instance(mols, data, g, nd) if nd else None
    cases["same_coarse_variants"] = variants
    out["D3"]["cases"] = cases

    # COUNT-ALL computation case: a molecule where the top structure term changes the call
    try:
        with np.load(K.ANALYSIS / "count_all_contributions.npz") as zc:
            top_ids = zc["top20_ids"]
            names = zc["structure_names"]
            coef_raw = zc["coef_raw"]
        with np.load(K.PREDICTIONS / "COUNT-ALL.npz") as zl:
            logit = zl["logit"]
        candidates = np.flatnonzero(c419 & p2078 & (data.y == 1))
        if candidates.size:
            g = int(candidates[np.argmax(logit[candidates])])
            ids = top_ids[g][:8]
            out["D3"]["count_all_case"] = {
                "graph_index": g, "split": next(s for s in K.SPLITS if g in set(data.rows[s].tolist())),
                "smiles": mols.smiles[g], "label": int(data.y[g]),
                "logit": float(logit[g]),
                "top_terms": [{"structure": str(names[i]),
                               "coef_raw": float(coef_raw[2 + i]),
                               "value": float(data.all_structure_matrix()[g, i])}
                              for i in ids],
            }
    except Exception as exc:  # pragma: no cover - analysis-only convenience
        out["D3"]["count_all_case"] = {"error": repr(exc)}

    K.write_json(K.ANALYSIS / "phaseD.json", out)
    print("[D] done", flush=True)


if __name__ == "__main__":
    main()
