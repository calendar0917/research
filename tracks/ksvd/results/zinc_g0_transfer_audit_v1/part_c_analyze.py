"""Part C analysis — read the per-molecule tables and produce the report tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
CLASSES = ("joint_seen", "joint_rare_marginals_seen", "structure_rare",
           "semantic_rare", "both_rare")
FOLDS = ("A", "B")
KEY_VARIANTS = ("residual_support", "full_descriptor")


def load_rows():
    z = np.load(OUT / "part_c_per_molecule.npz", allow_pickle=False)
    rows = {}
    for k in z.files:
        f, v, var, s = k.split("|")
        rows[(f, v, var, s)] = [json.loads(x) for x in z[k].tolist()]
    return rows


def frac(rows, key):
    return np.array([r["frac"][key] if r["frac"][key] is not None else np.nan for r in rows], np.float64)


def main() -> int:
    rows = load_rows()
    out = {"class_composition": {}, "exposure_quantiles": {}, "molecule_scale": {}}
    for f in FOLDS:
        for v in ("edge", "node"):
            for var in KEY_VARIANTS:
                for s in ("meta_g0", "dev_g0"):
                    R = rows[(f, v, var, s)]
                    comp = {}
                    for c in CLASSES:
                        x = frac(R, c)
                        comp[c] = {"mean": float(np.nanmean(x)), "p50": float(np.nanmedian(x)),
                                   "max": float(np.nanmax(x)),
                                   "n_mol_gt0": int(np.sum(x > 0)),
                                   "n_mol_ge_0.05": int(np.sum(x >= 0.05)),
                                   "n_mol_ge_0.25": int(np.sum(x >= 0.25))}
                    unseen = np.array([r["unseen_frac"] for r in R], np.float64)
                    comp["_unseen_joint"] = {"mean": float(np.nanmean(unseen)),
                                             "max": float(np.nanmax(unseen)),
                                             "n_mol_gt0": int(np.sum(unseen > 0))}
                    out["class_composition"][f"{f}|{v}|{var}|{s}"] = comp
                    exp = frac(R, "joint_rare_marginals_seen")
                    out["exposure_quantiles"][f"{f}|{v}|{var}|{s}"] = {
                        "mean": float(np.nanmean(exp)), "p50": float(np.nanmedian(exp)),
                        "p90": float(np.nanpercentile(exp, 90)),
                        "p99": float(np.nanpercentile(exp, 99)),
                        "max": float(np.nanmax(exp)),
                        "n_ge_0.25": int(np.sum(exp >= 0.25)),
                        "n_le_0.05": int(np.sum(exp <= 0.05)),
                        "n_gt_0.05": int(np.sum(exp > 0.05)),
                    }
                    out["molecule_scale"][f"{f}|{v}|{var}|{s}"] = {
                        "n_nodes_mean": float(np.mean([r["n_nodes"] for r in R])),
                        "n_edges_mean": float(np.mean([r["n_edges"] for r in R])),
                        "cycle_rank_mean": float(np.mean([r["cycle_rank"] for r in R])),
                        "mae": float(np.mean([r["abs_err"] for r in R])),
                        "signed_mean": float(np.mean([r["err"] for r in R])),
                    }
    (OUT / "part_c_class_composition.json").write_text(json.dumps(out, indent=2))
    # print concise
    for f in FOLDS:
        for v in ("edge", "node"):
            for var in KEY_VARIANTS:
                for s in ("meta_g0", "dev_g0"):
                    comp = out["class_composition"][f"{f}|{v}|{var}|{s}"]
                    exp = out["exposure_quantiles"][f"{f}|{v}|{var}|{s}"]
                    print(f"{f} {v:4s} {var:17s} {s:7s} "
                          f"joint_seen={comp['joint_seen']['mean']:.4f} "
                          f"jrm={comp['joint_rare_marginals_seen']['mean']:.4f} "
                          f"struct_rare={comp['structure_rare']['mean']:.4f} "
                          f"sem_rare={comp['semantic_rare']['mean']:.4f} "
                          f"both_rare={comp['both_rare']['mean']:.4f} "
                          f"unseen={comp['_unseen_joint']['mean']:.4f} "
                          f"exp_max={exp['max']:.3f} n>=0.25={exp['n_ge_0.25']} n>0.05={exp['n_gt_0.05']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())