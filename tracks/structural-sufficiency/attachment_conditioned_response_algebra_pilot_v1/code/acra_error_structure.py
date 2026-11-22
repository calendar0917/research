"""Error structure on Valid (supporting the failure-class question).

Molecule-level only: whole-molecule labels are the only supervised target.
Buckets are defined by the fraction of a molecule's directed events whose
(S, C, cut-bond) combination never occurred in Train.
"""

from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence

import numpy as np

import acra_common as C
import acra_data as D
import acra_cache as K

PD = C.PD
RUNS = {"ACRA_ADD": "ACRA_ADD_S0", "ACRA_COND": "ACRA_COND_S0", "ACRA_REL": "ACRA_REL_S0"}
EDGES = (0.0, 0.2, 0.5, 0.8, 1.0)


def _bucket(x: float) -> str:
    if x <= 0:
        return "0"
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        if lo < x <= hi:
            return f"({lo:g},{hi:g}]"
    return ">1"


def run(cfg: Mapping[str, Any], log=print) -> dict[str, Any]:
    pack = K.ensure_pack(cfg)
    raw = D.load_raw(cfg)
    splits = D.official_splits(raw)
    labels = np.asarray(raw["labels"], np.float64)
    smiles = np.asarray(raw["smiles"])
    train_mean = float(np.mean(labels[splits["train"]]))
    baseline = np.full(splits["valid"].size, train_mean)

    preds = {}
    for arm, run_id in RUNS.items():
        with np.load(C.run_dir(run_id) / "predictions_best.npz") as handle:
            preds[arm] = {k: np.asarray(handle[k], np.float64) for k in handle.files}
    truth = preds["ACRA_ADD"]["label"]
    if not np.allclose(truth, labels[splits["valid"]], atol=1e-8):
        raise ValueError("prediction labels do not match the official Valid split")

    ev_ptr = np.asarray(pack["ev_ptr"], np.int64)
    combo_ids = np.asarray(pack["ev_combo"], np.int64)
    mol_split = np.full(int(pack["n_molecules"]), 2, np.uint8)
    mol_split[splits["train"]] = 0
    mol_split[splits["valid"]] = 1
    ev_mol = np.repeat(np.arange(int(pack["n_molecules"])), np.diff(ev_ptr))
    train_combo = np.bincount(combo_ids[mol_split[ev_mol] == 0],
                              minlength=int(pack["combo_s"].size))
    novelty = np.asarray([
        float(np.mean(train_combo[combo_ids[ev_ptr[row]:ev_ptr[row + 1]]] == 0))
        for row in splits["valid"].tolist()
    ])

    desc = []
    for row in splits["valid"].tolist():
        mol = D.raw_molecule(raw, row)
        adj = D.adjacency(mol)
        geo = D.bond_geometry(mol, adj)
        n_comp = len(set(PD.connected_components(adj, mol.n_atoms)))
        desc.append({
            "row_id": int(row),
            "smiles": str(smiles[row]),
            "n_atoms": int(mol.n_atoms),
            "n_bonds": int(mol.bond_u.size),
            "n_components": n_comp,
            "cycle_rank": int(mol.bond_u.size - mol.n_atoms + n_comp),
            "ring_bond_fraction": float(np.mean(geo["in_ring"])),
            "n_bridge_bonds": int(geo["is_bridge"].sum()),
            "unseen_combo_fraction": float(novelty[len(desc)]),
        })

    se = {arm: (preds[arm]["pred"] - truth) ** 2 for arm in preds}
    out: dict[str, Any] = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "baseline_train_mean": {"value": train_mean, "rmse": C.rmse(baseline, truth),
                                "mae": C.mae(baseline, truth)},
        "n_valid": int(truth.size),
        "novelty_buckets": {},
        "rings": {},
        "size_quartiles": {},
        "correlations": {},
        "worst_rel_cases": [],
        "test_labels_read": False,
    }
    bucket_of = np.asarray([_bucket(x) for x in novelty])
    for bucket in ["0"] + [f"({lo:g},{hi:g}]" for lo, hi in zip(EDGES[:-1], EDGES[1:])]:
        m = bucket_of == bucket
        if not m.any():
            continue
        rec = {"n": int(m.sum())}
        for arm in preds:
            rec[arm] = {"rmse": C.rmse(preds[arm]["pred"][m], truth[m]),
                        "mae": C.mae(preds[arm]["pred"][m], truth[m])}
        rec["rmse_reduction_rel_vs_add"] = rec["ACRA_ADD"]["rmse"] - rec["ACRA_REL"]["rmse"]
        rec["rmse_reduction_rel_vs_cond"] = rec["ACRA_COND"]["rmse"] - rec["ACRA_REL"]["rmse"]
        out["novelty_buckets"][bucket] = rec
    for name, m in (("has_ring_bonds", np.asarray([d["ring_bond_fraction"] > 0 for d in desc])),
                    ("no_ring_bonds", np.asarray([d["ring_bond_fraction"] == 0 for d in desc])),
                    ("has_bridge_bonds", np.asarray([d["n_bridge_bonds"] > 0 for d in desc])),
                    ("single_component", np.asarray([d["n_components"] == 1 for d in desc])),
                    ("multi_component", np.asarray([d["n_components"] > 1 for d in desc]))):
        if not m.any():
            continue
        out["rings"][name] = {"n": int(m.sum()), **{
            arm: {"rmse": C.rmse(preds[arm]["pred"][m], truth[m]),
                  "mae": C.mae(preds[arm]["pred"][m], truth[m])}
            for arm in preds}}
    sizes = np.asarray([d["n_atoms"] for d in desc])
    qs = np.quantile(sizes, [0.25, 0.5, 0.75])
    for name, m in (("q1_small", sizes <= qs[0]), ("q2", (sizes > qs[0]) & (sizes <= qs[1])),
                    ("q3", (sizes > qs[1]) & (sizes <= qs[2])), ("q4_large", sizes > qs[2])):
        if not m.any():
            continue
        out["size_quartiles"][name] = {
            "n": int(m.sum()), "n_atoms_range": [int(sizes[m].min()), int(sizes[m].max())],
            **{arm: {"rmse": C.rmse(preds[arm]["pred"][m], truth[m]),
                     "mae": C.mae(preds[arm]["pred"][m], truth[m])}
               for arm in preds}}
    for arm in preds:
        abs_err = np.abs(preds[arm]["pred"] - truth)
        out["correlations"][arm] = {
            "pearson_abs_error_vs_unseen_fraction": C.pearson(abs_err, novelty),
            "pearson_abs_error_vs_n_atoms": C.pearson(abs_err, sizes),
            "pearson_abs_error_vs_cycle_rank":
                C.pearson(abs_err, np.asarray([d["cycle_rank"] for d in desc])),
            "pearson_abs_error_vs_ring_bond_fraction":
                C.pearson(abs_err, np.asarray([d["ring_bond_fraction"] for d in desc])),
        }
    order = np.lexsort((np.asarray([d["row_id"] for d in desc]),
                        -np.abs(preds["ACRA_REL"]["pred"] - truth)))
    for i in order[:10].tolist():
        out["worst_rel_cases"].append({
            **desc[i],
            "label": float(truth[i]),
            "pred_rel": float(preds["ACRA_REL"]["pred"][i]),
            "abs_error_rel": float(abs(preds["ACRA_REL"]["pred"][i] - truth[i])),
            "pred_add": float(preds["ACRA_ADD"]["pred"][i]),
            "pred_cond": float(preds["ACRA_COND"]["pred"][i]),
        })
    C.write_json(C.analysis_dir() / "error_structure.json", out)
    log("wrote error_structure.json")
    for bucket, rec in out["novelty_buckets"].items():
        log(f"  novelty {bucket}: n={rec['n']} add={rec['ACRA_ADD']['rmse']:.3f} "
            f"rel={rec['ACRA_REL']['rmse']:.3f}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    _args = ap.parse_args(argv)
    run(C.load_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())