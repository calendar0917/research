"""Compact-v6 analysis: multi-seed table, difficulty quintiles, figures.

Reads the frozen Scratch/validation runs and the diagnostics produced by
``compact_v6_diagnostics``.  Writes CSV/JSON tables and the four figures
requested by the task brief.  No training, no test access.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import REPO_ROOT

RUNS_ROOT = REPO_ROOT / "tracks/ksvd/runs/2026/09/10"
RESULTS = REPO_ROOT / "tracks/ksvd/results/compact_v6"
QUINTILES = (
    REPO_ROOT
    / "tracks/ksvd/results/baseline_difficulty_typed_refinement_audit/per_molecule_analysis.csv"
)

RUN_IDS = {
    "v4": {
        0: "20260910-162128-aaeed272",
        1: "20260910-170023-20032784",
        2: "20260910-172706-f4b156ed",
    },
    "factorized": {
        0: "20260910-162801-fa8a41cf",
        1: "20260910-170615-ffcdcbee",
        2: "20260910-173258-da88301d",
    },
    "count": {
        0: "20260910-163550-00179d67",
        1: "20260910-171330-7f506f87",
        2: "20260910-174000-4e25d8a1",
    },
    "capacity": {
        0: "20260910-164334-d0478d18",
        1: "20260910-172032-781371d3",
        2: "20260910-174703-bb8f4850",
    },
}


def _result(model: str, seed: int) -> dict:
    path = RUNS_ROOT / RUN_IDS[model][seed] / "artifacts/legacy_full_result.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _valid(result: dict) -> tuple[np.ndarray, np.ndarray]:
    valid = result["evaluation"]["valid"]
    return (
        np.asarray(valid["targets"], dtype=np.float64),
        np.asarray(valid["predictions"], dtype=np.float64),
    )


def _mae(targets: np.ndarray, preds: np.ndarray) -> float:
    return float(np.mean(np.abs(targets - preds)))


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    seeds = (0, 1, 2)

    # ---- Table B / C: seed0 + multi-seed ---------------------------------
    table_bc = []
    for model in ("v4", "capacity", "count", "factorized"):
        row = {"model": model}
        for seed in seeds:
            result = _result(model, seed)
            targets, preds = _valid(result)
            row[f"seed{seed}"] = _mae(targets, preds)
        row["params"] = int(result["evaluation"]["parameters"])
        table_bc.append(row)
    table_bc = pd.DataFrame(table_bc)

    paired = {}
    for model in ("capacity", "count", "factorized"):
        deltas = [
            table_bc.loc[table_bc.model == "v4", f"seed{seed}"].iloc[0]
            - table_bc.loc[table_bc.model == model, f"seed{seed}"].iloc[0]
            for seed in seeds
        ]
        paired[model] = {
            "per_seed": deltas,
            "mean": float(np.mean(deltas)),
            "median": float(np.median(deltas)),
            "std": float(np.std(deltas, ddof=0)),
            "positive": int(sum(d > 0 for d in deltas)),
            "n": len(deltas),
        }
    fac_vs_count = [
        table_bc.loc[table_bc.model == "count", f"seed{seed}"].iloc[0]
        - table_bc.loc[table_bc.model == "factorized", f"seed{seed}"].iloc[0]
        for seed in seeds
    ]
    fac_vs_cap = [
        table_bc.loc[table_bc.model == "capacity", f"seed{seed}"].iloc[0]
        - table_bc.loc[table_bc.model == "factorized", f"seed{seed}"].iloc[0]
        for seed in seeds
    ]
    summary = {
        "runs": RUN_IDS,
        "table_bc": table_bc.to_dict(orient="records"),
        "paired_vs_v4": paired,
        "factorized_vs_count": {
            "per_seed": fac_vs_count, "mean": float(np.mean(fac_vs_count))
        },
        "factorized_vs_capacity": {
            "per_seed": fac_vs_cap, "mean": float(np.mean(fac_vs_cap))
        },
    }

    # ---- Table D: historical difficulty quintiles (seed 0) ---------------
    quint = pd.read_csv(QUINTILES)
    quint = quint.set_index("molecule_id")["difficulty_quintile"].to_dict()
    v4_targets, v4_preds = _valid(_result("v4", 0))
    fac_targets, fac_preds = _valid(_result("factorized", 0))
    count_targets, count_preds = _valid(_result("count", 0))
    cap_targets, cap_preds = _valid(_result("capacity", 0))
    quintile = np.asarray(
        [quint[f"valid:{i:04d}"] for i in range(len(v4_targets))], dtype=np.int64
    )
    rows = []
    for q in range(1, 6):
        mask = quintile == q
        rows.append(
            {
                "quintile": q,
                "n": int(mask.sum()),
                "v4_mae": _mae(v4_targets[mask], v4_preds[mask]),
                "factorized_mae": _mae(fac_targets[mask], fac_preds[mask]),
                "count_mae": _mae(count_targets[mask], count_preds[mask]),
                "capacity_mae": _mae(cap_targets[mask], cap_preds[mask]),
            }
        )
    table_d = pd.DataFrame(rows)
    table_d["delta_factorized"] = table_d["v4_mae"] - table_d["factorized_mae"]
    table_d.to_csv(RESULTS / "table_d_difficulty_quintiles.csv", index=False)
    easy = table_d[table_d.quintile.isin([1, 2])]
    bulk = {
        "Q1Q2_v4_mae": float(
            easy["v4_mae"].mul(easy["n"]).sum() / easy["n"].sum()
        ),
        "Q1Q2_factorized_mae": float(
            easy["factorized_mae"].mul(easy["n"]).sum() / easy["n"].sum()
        ),
    }
    bulk["Q1Q2_delta"] = bulk["Q1Q2_v4_mae"] - bulk["Q1Q2_factorized_mae"]
    summary["table_d"] = table_d.to_dict(orient="records")
    summary["bulk_Q1Q2"] = bulk

    # ---- Table E: mechanism diagnostics ----------------------------------
    diagnostics = json.loads(
        (REPO_ROOT / "tracks/ksvd/results/compact_v6_diagnostics/diagnostics.json").read_text()
    )
    summary["table_e"] = diagnostics

    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

    # ---- Figures ---------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = RESULTS / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # Fig 1: v4 vs factorized by difficulty quintile
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(1, 6)
    ax.plot(x, table_d["v4_mae"], marker="o", label="v4 historical")
    ax.plot(x, table_d["factorized_mae"], marker="s", label="v6 factorized-role")
    ax.set_yscale("log")
    ax.set_xlabel("historical difficulty quintile")
    ax.set_ylabel("validation MAE (log)")
    ax.set_title("v4 vs v6 by historical difficulty (seed 0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "fig1_difficulty_quintile.png", dpi=140)
    plt.close(fig)

    # Fig 2: factorized vs count vs capacity across seeds
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.25
    for offset, model in ((-width, "factorized"), (0.0, "count"), (width, "capacity")):
        values = [
            table_bc.loc[table_bc.model == model, f"seed{seed}"].iloc[0]
            for seed in seeds
        ]
        ax.bar(np.arange(len(seeds)) + offset, values, width=width, label=model)
    ax.set_xticks(np.arange(len(seeds)))
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("validation MAE")
    ax.set_title("v6 variants vs seed (v4 = dotted line per seed)")
    for index, seed in enumerate(seeds):
        base = table_bc.loc[table_bc.model == "v4", f"seed{seed}"].iloc[0]
        ax.hlines(base, index - 0.5, index + 0.5, colors="k", linestyles="dotted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "fig2_variants.png", dpi=140)
    plt.close(fig)

    # Fig 3: attribute branch norm distribution
    norms = np.load(
        REPO_ROOT / "tracks/ksvd/results/compact_v6_diagnostics/factorized_role_attribute_norms.npy"
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(norms, bins=40)
    ax.set_xlabel("||e_attribute|| (per patch, frozen seed-0 model)")
    ax.set_ylabel("patch count")
    ax.set_title("v6 attribute-branch output norm (collapsed)")
    fig.tight_layout()
    fig.savefig(figures / "fig3_attribute_norm.png", dpi=140)
    plt.close(fig)

    # Fig 4: clean vs role-shuffled per-molecule error
    role_shuffled = diagnostics["factorized_role"]["role_association_shuffle_mae"]
    fig, ax = plt.subplots(figsize=(5, 5))
    clean_err = np.abs(fac_targets - fac_preds)
    ax.scatter(clean_err, clean_err, s=4, alpha=0.4)
    ax.set_xlabel("clean |error| (seed 0 factorized)")
    ax.set_ylabel("role-shuffled |error|")
    ax.set_title(
        f"Clean vs role-shuffled error\n"
        f"clean MAE {diagnostics['factorized_role']['clean_mae']:.6f} == "
        f"shuffled MAE {role_shuffled:.6f}"
    )
    ax.plot([0, clean_err.max()], [0, clean_err.max()], "r--", lw=0.8)
    fig.tight_layout()
    fig.savefig(figures / "fig4_role_shuffle.png", dpi=140)
    plt.close(fig)

    print("wrote", RESULTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
