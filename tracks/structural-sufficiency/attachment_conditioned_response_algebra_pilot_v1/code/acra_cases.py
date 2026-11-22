"""Deterministic failure / success cases on Valid (11.5).

Fixed rule: paired per-molecule squared errors on the same Valid molecules.

* COND better than ADD  = top-3 by (SE_ADD - SE_COND), ties by ascending row id
* COND worse than ADD   = top-3 by (SE_COND - SE_ADD), ties by ascending row id
* REL better than COND  = top-3 by (SE_COND - SE_REL)
* REL worse than COND   = top-3 by (SE_REL - SE_COND)

No manual cherry-picking.  Structural descriptors are label-free.
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
TOP = 3


def _load_valid(run_id: str) -> dict[str, np.ndarray]:
    path = C.run_dir(run_id) / "predictions_best.npz"
    with np.load(path) as handle:
        return {k: np.asarray(handle[k], np.float64) for k in handle.files}


def _top(score: np.ndarray, row_id: np.ndarray, k: int = TOP) -> list[int]:
    order = np.lexsort((row_id, -score))
    out = []
    for i in order.tolist():
        if score[i] <= 0:
            break
        out.append(i)
        if len(out) >= k:
            break
    return out


def _describe(mol: PD.Molecule, adj, geo) -> dict[str, Any]:
    return {
        "n_atoms": int(mol.n_atoms),
        "n_bonds": int(mol.bond_u.size),
        "n_components": int(len(set(PD.connected_components(adj, mol.n_atoms)))),
        "cycle_rank": int(mol.bond_u.size - mol.n_atoms
                          + len(set(PD.connected_components(adj, mol.n_atoms)))),
        "n_ring_bonds": int(geo["in_ring"].sum()),
        "n_bridge_bonds": int(geo["is_bridge"].sum()),
    }


def run(cfg: Mapping[str, Any], log=print) -> dict[str, Any]:
    pack = K.ensure_pack(cfg)
    raw = D.load_raw(cfg)
    splits = D.official_splits(raw)
    labels = np.asarray(raw["labels"], np.float64)
    smiles = np.asarray(raw["smiles"])
    valid_rows = np.asarray(splits["valid"], np.int64)

    preds: dict[str, dict[str, np.ndarray]] = {}
    for arm, run_id in RUNS.items():
        path = C.run_dir(run_id) / "predictions_best.npz"
        if not path.is_file():
            log(f"  {run_id}: missing; cases not computed")
            return {"n": 0, "missing": run_id}
        preds[arm] = _load_valid(run_id)
    row_id = preds["ACRA_ADD"]["row_id"]
    for arm in preds:
        if not np.array_equal(row_id, preds[arm]["row_id"]):
            raise ValueError("Valid molecule order differs between arms")
    truth = preds["ACRA_ADD"]["label"]
    se = {arm: (preds[arm]["pred"] - truth) ** 2 for arm in preds}

    ev_ptr = np.asarray(pack["ev_ptr"], np.int64)
    combo_ids = np.asarray(pack["ev_combo"], np.int64)
    ev_split = np.full(int(pack["n_molecules"]), 2, np.uint8)
    ev_split[splits["train"]] = 0
    ev_split[splits["valid"]] = 1
    ev_mol = np.repeat(np.arange(int(pack["n_molecules"])), np.diff(ev_ptr))
    train_combo = np.bincount(combo_ids[ev_split[ev_mol] == 0],
                              minlength=int(pack["combo_s"].size))
    novelty = {}
    for row in valid_rows.tolist():
        ids = combo_ids[ev_ptr[row]:ev_ptr[row + 1]]
        novelty[row] = float(np.mean(train_combo[ids] == 0)) if ids.size else 0.0

    def record(i: int) -> dict[str, Any]:
        row = int(valid_rows[i])
        mol = D.raw_molecule(raw, row)
        adj = D.adjacency(mol)
        geo = D.bond_geometry(mol, adj)
        return {
            "valid_index": int(i),
            "row_id": int(row_id[i]),
            "smiles": str(smiles[row]),
            **_describe(mol, adj, geo),
            "label": float(truth[i]),
            "unseen_combo_fraction": novelty[row],
            "predictions": {arm: float(preds[arm]["pred"][i]) for arm in preds},
            "squared_errors": {arm: float(se[arm][i]) for arm in preds},
            "terms_std": {
                arm: {
                    "e_atom": float(preds[arm]["e_atom_std"][i]),
                    "e_bond": float(preds[arm]["e_bond_std"][i]),
                    "e_rel": float(preds[arm]["e_rel_std"][i]),
                    "beta": float(preds[arm]["beta_std"][i]),
                }
                for arm in preds
            },
        }

    def block(control: str, candidate: str, label: str) -> dict[str, Any]:
        gain = se[control] - se[candidate]
        better = _top(gain, row_id)
        worse = _top(-gain, row_id)
        diff = gain
        rng = np.random.default_rng(0)
        boot = np.asarray([np.mean(rng.choice(diff, diff.size, replace=True))
                           for _ in range(2000)])
        return {
            "control": control,
            "candidate": candidate,
            "definition": label,
            "n_valid": int(gain.size),
            "n_candidate_better": int(np.sum(gain > 0)),
            "n_candidate_worse": int(np.sum(gain < 0)),
            "mean_squared_error_reduction": float(np.mean(gain)),
            "mean_squared_error_reduction_ci": [float(np.percentile(boot, 2.5)),
                                                float(np.percentile(boot, 97.5))],
            "better_cases": [record(i) for i in better],
            "worse_cases": [record(i) for i in worse],
            "rule": "top-3 by paired SE difference; ties by ascending row id; only "
                    "strictly positive differences are listed",
        }

    out = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "selection": "Valid molecules only; deterministic fixed rule; no cherry-picking",
        "improvement_over_control": block("ACRA_ADD", "ACRA_COND",
                                          "SE(ADD) - SE(COND) > 0 means COND better"),
        "degradation_vs_control": block("ACRA_COND", "ACRA_ADD",
                                        "SE(COND) - SE(ADD) > 0 means COND worse"),
        "relation_improvement": block("ACRA_COND", "ACRA_REL",
                                      "SE(COND) - SE(REL) > 0 means REL better"),
        "relation_degradation": block("ACRA_REL", "ACRA_COND",
                                      "SE(REL) - SE(COND) > 0 means REL worse"),
        "test_labels_read": False,
    }
    C.write_json(C.analysis_dir() / "cases.json", out)
    for key in ("improvement_over_control", "degradation_vs_control",
                "relation_improvement", "relation_degradation"):
        b = out[key]
        log(f"  {key}: better={b['n_candidate_better']} worse={b['n_candidate_worse']} "
            f"mean_SE_reduction={b['mean_squared_error_reduction']:.4f}")
    log("wrote cases.json")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    _args = ap.parse_args(argv)
    run(C.load_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())