"""Literature comparison matrix for the seven-task round.

Every number in ``analysis/literature_matrix.csv`` is transcribed from a fetched,
hash-pinned public document; the ``locator`` column says exactly where.  Nothing
is recalled from memory and nothing is re-computed.

Grading (pre-registered in ``BENCHMARK_PROTOCOL.md``):

* ``A`` EXACT_SPLIT_MATCH   -- the source uses the *same* Train/Valid/Test row set
                               (today: only ``ogbg-molhiv``, whose official split is
                               exactly the one this round consumes) and the same
                               metric.  Seed count and input featurisation still
                               differ and are recorded.
* ``B`` PROTOCOL_MATCH_ONLY -- same dataset and split *method*, same metric, but the
                               exact indices are not published/verified.
* ``C`` NOT_COMPARABLE      -- different data version, molecule count, split method
                               or metric.  Never ranked against this round.

A seed-0 point estimate is never described as beating a published multi-seed mean.

Usage
-----
    uv run python .../code/mb2_literature.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import mb2_common as C

# ---------------------------------------------------------------------------
# fetched, hash-pinned sources
# ---------------------------------------------------------------------------
SOURCES: dict[str, dict[str, Any]] = {
    "molnet2018": {
        "citation": "Wu et al., Chem. Sci. 9, 513-530 (2018) - MoleculeNet",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5868307/",
        "fetched": "2026-11-02T03:32:00Z",
        "sha256": "776fcf26c13c92943a03e66ea0f87cba4b660e0b588c6d2b504f321fb7ce4d63",
        "bytes": 242320,
        "note": (
            "Table 1 (dataset table: per-dataset molecule count and split method) and Table 3 "
            "(test-subset best conventional vs best graph-based number). One number per model, no "
            "multi-seed statistics, no per-example split indices. The PMC HTML carries a per-request "
            "nonce, so the sha256 changes on re-fetch even when the bytes are the same length "
            "(242320 B); the earlier Sprint-01 fetch had sha256 f5ba3f55... and the two transcripts "
            "were compared row by row."
        ),
    },
    "grover2020": {
        "citation": "Rong et al., NeurIPS 33 (2020) - GROVER, arXiv:2007.02835v2",
        "url": "https://arxiv.org/pdf/2007.02835v2",
        "fetched": "2026-11-02T03:32:00Z",
        "sha256": "d672b7f40e5cef743b19b43b6d2cfec0273b9fc98617bfeb968923f275b9ed81",
        "bytes": 2657733,
        "note": (
            "Table 1 (p. 7): classification ROC-AUC and regression RMSE, mean(std) over 3 runs. "
            "Sec. 4.2: 'we apply the scaffold splitting for all tasks on all datasets', so these "
            "measure a harder regime than the random-split rows of this round."
        ),
    },
    "ogb2020": {
        "citation": "Hu et al., OGB, arXiv:2005.00687v4",
        "url": "https://arxiv.org/pdf/2005.00687v4",
        "fetched": "2026-11-02T03:32:00Z",
        "sha256": "28782efb305d625f4b020ea00599cf3bfd498a67b60fd72777b0116cb3ddb1e1",
        "bytes": 976124,
        "note": (
            "Table 15 (ogbg-molhiv, 10 runs, mean(std)) and Table 20 (ogbg-moltox21) / Table 24 "
            "(ogbg-molbbbp). The checkbox columns ('Add. Feat.' / 'Virt. Node') are mangled by PDF "
            "text extraction, so a row is only cited when the main text fixes it: p. 7/8 states GIN "
            "with additional features and virtual nodes is the best configuration and that 'the "
            "best GIN achieves ... 82.73+-2.02% (5.66 percentage points higher than scaffold)' on "
            "molhiv and 77.57+-0.62% on the moltox21 scaffold split."
        ),
    },
    "chemprop2019": {
        "citation": "Yang et al., J. Chem. Inf. Model. 59, 3370-3389 (2019) - D-MPNN / chemprop, arXiv:1904.01561v5",
        "url": "https://arxiv.org/pdf/1904.01561v5",
        "fetched": "2026-11-02T03:32:00Z",
        "sha256": "f95ac4b49d196bc1b63e00e75fa7dff4709bce1d6ed0967f8d6a496d35a23447",
        "bytes": 2359573,
        "note": (
            "Table 4 attests the dataset sizes (ESOL 1128, FreeSolv 642, Lipophilicity 4200, BACE "
            "1513) used by this round. The per-dataset result panels are figures, not numeric "
            "tables, so no D-MPNN scalar is transcribed from this source."
        ),
    },
}

# ---------------------------------------------------------------------------
# transcribed rows
# ---------------------------------------------------------------------------
# (dataset, source, model, family, metric, value, std, n_seeds, split_method,
#  n_molecules, grade, locator, note)
ROWS: list[dict[str, Any]] = [
    # ---- ESOL ---------------------------------------------------------
    dict(dataset="ESOL", source="molnet2018", model="XGBoost", family="conventional", metric="RMSE", value=0.99,
         split_method="random", n_molecules=1128, grade="B",
         locator="Table 3, Physical chemistry / ESOL / conventional column",
         note="single reported number, no seed spread; indices unpublished"),
    dict(dataset="ESOL", source="molnet2018", model="MPNN", family="graph", metric="RMSE", value=0.58,
         split_method="random", n_molecules=1128, grade="B",
         locator="Table 3, Physical chemistry / ESOL / graph-based column"),
    dict(dataset="ESOL", source="grover2020", model="D-MPNN", family="graph", metric="RMSE", value=0.980, std=0.258,
         n_seeds=3, split_method="scaffold", n_molecules=1128, grade="C",
         locator="Table 1, Regression block, ESOL column, DMPNN row",
         note="scaffold split: a harder regime than this round's random split"),
    dict(dataset="ESOL", source="grover2020", model="AttentiveFP", family="graph", metric="RMSE", value=0.853, std=0.060,
         n_seeds=3, split_method="scaffold", n_molecules=1128, grade="C",
         locator="Table 1, Regression block, ESOL column, AttentiveFP row"),
    dict(dataset="ESOL", source="grover2020", model="GROVER-large", family="graph", metric="RMSE", value=0.831,
         std=0.120, n_seeds=3, split_method="scaffold", n_molecules=1128, grade="C",
         locator="Table 1, Regression block, ESOL column, GROVERlarge row", note="pretrained"),

    # ---- Lipophilicity -------------------------------------------------
    dict(dataset="Lipophilicity", source="molnet2018", model="XGBoost", family="conventional", metric="RMSE", value=0.799,
         split_method="random", n_molecules=4200, grade="B",
         locator="Table 3, Physical chemistry / Lipophilicity / conventional column"),
    dict(dataset="Lipophilicity", source="molnet2018", model="GraphConv", family="graph", metric="RMSE", value=0.655,
         split_method="random", n_molecules=4200, grade="B",
         locator="Table 3, Physical chemistry / Lipophilicity / graph-based column"),
    dict(dataset="Lipophilicity", source="grover2020", model="D-MPNN", family="graph", metric="RMSE", value=0.653, std=0.046,
         n_seeds=3, split_method="scaffold", n_molecules=4200, grade="C",
         locator="Table 1, Regression block, Lipo column, DMPNN row"),
    dict(dataset="Lipophilicity", source="grover2020", model="GROVER-large", family="graph", metric="RMSE", value=0.560,
         std=0.035, n_seeds=3, split_method="scaffold", n_molecules=4200, grade="C",
         locator="Table 1, Regression block, Lipo column, GROVERlarge row", note="pretrained"),

    # ---- FreeSolv ------------------------------------------------------
    dict(dataset="FreeSolv", source="molnet2018", model="XGBoost", family="conventional", metric="RMSE", value=1.74,
         split_method="random", n_molecules=642, grade="B",
         locator="Table 3, Physical chemistry / FreeSolv / conventional column"),
    dict(dataset="FreeSolv", source="molnet2018", model="MPNN", family="graph", metric="RMSE", value=1.15,
         split_method="random", n_molecules=642, grade="B",
         locator="Table 3, Physical chemistry / FreeSolv / graph-based column"),
    dict(dataset="FreeSolv", source="grover2020", model="D-MPNN", family="graph", metric="RMSE", value=2.177, std=0.914,
         n_seeds=3, split_method="scaffold", n_molecules=642, grade="C",
         locator="Table 1, Regression block, FreeSolv column, DMPNN row"),
    dict(dataset="FreeSolv", source="grover2020", model="GROVER-large", family="graph", metric="RMSE", value=1.544,
         std=0.397, n_seeds=3, split_method="scaffold", n_molecules=642, grade="C",
         locator="Table 1, Regression block, FreeSolv column, GROVERlarge row", note="pretrained"),

    # ---- BACE ----------------------------------------------------------
    dict(dataset="BACE", source="molnet2018", model="Random Forest", family="conventional", metric="ROC-AUC", value=0.867,
         split_method="scaffold", n_molecules=1522, grade="C",
         locator="Table 3, Biophysics / BACE / conventional column",
         note="MoleculeNet lists 1522 molecules; this round uses the 1513-molecule cleansed release"),
    dict(dataset="BACE", source="molnet2018", model="Weave", family="graph", metric="ROC-AUC", value=0.806,
         split_method="scaffold", n_molecules=1522, grade="C",
         locator="Table 3, Biophysics / BACE / graph-based column"),
    dict(dataset="BACE", source="grover2020", model="D-MPNN", family="graph", metric="ROC-AUC", value=0.852, std=0.053,
         n_seeds=3, split_method="scaffold", n_molecules=1513, grade="B",
         locator="Table 1, Classification block, BACE column, DMPNN row",
         note="same 1513 molecules, same scaffold protocol, unpublished indices"),
    dict(dataset="BACE", source="grover2020", model="AttentiveFP", family="graph", metric="ROC-AUC", value=0.863, std=0.015,
         n_seeds=3, split_method="scaffold", n_molecules=1513, grade="B",
         locator="Table 1, Classification block, BACE column, AttentiveFP row"),
    dict(dataset="BACE", source="grover2020", model="GROVER-large", family="graph", metric="ROC-AUC", value=0.894,
         std=0.028, n_seeds=3, split_method="scaffold", n_molecules=1513, grade="B",
         locator="Table 1, Classification block, BACE column, GROVERlarge row", note="pretrained"),

    # ---- BBBP ----------------------------------------------------------
    dict(dataset="BBBP", source="molnet2018", model="KernelSVM", family="conventional", metric="ROC-AUC", value=0.729,
         split_method="scaffold", n_molecules=2053, grade="C",
         locator="Table 3, Physiology / BBBP / conventional column",
         note="MoleculeNet lists 2053 molecules; this round uses the 2039-molecule release, so the sample set differs"),
    dict(dataset="BBBP", source="molnet2018", model="GraphConv", family="graph", metric="ROC-AUC", value=0.690,
         split_method="scaffold", n_molecules=2053, grade="C",
         locator="Table 3, Physiology / BBBP / graph-based column"),
    dict(dataset="BBBP", source="grover2020", model="D-MPNN", family="graph", metric="ROC-AUC", value=0.919, std=0.030,
         n_seeds=3, split_method="scaffold", n_molecules=2039, grade="B",
         locator="Table 1, Classification block, BBBP column, DMPNN row",
         note="same 2039 molecules, same scaffold protocol, unpublished indices"),
    dict(dataset="BBBP", source="grover2020", model="AttentiveFP", family="graph", metric="ROC-AUC", value=0.908, std=0.050,
         n_seeds=3, split_method="scaffold", n_molecules=2039, grade="B",
         locator="Table 1, Classification block, BBBP column, AttentiveFP row"),
    dict(dataset="BBBP", source="grover2020", model="GROVER-large", family="graph", metric="ROC-AUC", value=0.940,
         std=0.019, n_seeds=3, split_method="scaffold", n_molecules=2039, grade="B",
         locator="Table 1, Classification block, BBBP column, GROVERlarge row", note="pretrained"),
    dict(dataset="BBBP", source="ogb2020", model="GIN (OGB scaffold)", family="graph", metric="ROC-AUC", value=69.71,
         std=1.92, n_seeds=10, split_method="scaffold (OGB, 80/10/10)", n_molecules=2039, grade="C",
         locator="Table 24 (ogbg-molbbbp), last GIN row",
         note="percentage scale; OGB's own scaffold draw differs from ChemBench's, so not comparable to this round"),

    # ---- Tox21 ---------------------------------------------------------
    dict(dataset="Tox21", source="molnet2018", model="KernelSVM", family="conventional", metric="ROC-AUC", value=0.822,
         split_method="random", n_molecules=8014, grade="C",
         locator="Table 3, Physiology / Tox21 / conventional column",
         note="MoleculeNet Table 1 lists 12 tasks / 8014 molecules; this round uses the 7831-molecule "
              "12-endpoint release, so the sample set differs"),
    dict(dataset="Tox21", source="molnet2018", model="GraphConv", family="graph", metric="ROC-AUC", value=0.829,
         split_method="random", n_molecules=8014, grade="C",
         locator="Table 3, Physiology / Tox21 / graph-based column"),
    dict(dataset="Tox21", source="grover2020", model="D-MPNN", family="graph", metric="ROC-AUC", value=0.826, std=0.023,
         n_seeds=3, split_method="scaffold", n_molecules=7831, grade="C",
         locator="Table 1, Classification block, Tox21 column, DMPNN row",
         note="scaffold split against this round's published random split"),
    dict(dataset="Tox21", source="grover2020", model="GROVER-large", family="graph", metric="ROC-AUC", value=0.831,
         std=0.025, n_seeds=3, split_method="scaffold", n_molecules=7831, grade="C",
         locator="Table 1, Classification block, Tox21 column, GROVERlarge row", note="pretrained"),
    dict(dataset="Tox21", source="ogb2020", model="GIN (OGB scaffold)", family="graph", metric="ROC-AUC", value=77.57,
         std=0.62, n_seeds=10, split_method="scaffold (OGB)", n_molecules=7831, grade="C",
         locator="Table 20 (ogbg-moltox21), GIN row fixed by the main text (p. 8: 'the best GIN for the "
                 "scaffold split (77.57+-0.62% ROC-AUC)')",
         note="percentage scale; OGB scaffold draw, not the ChemBench random draw"),

    # ---- ogbg-molhiv ---------------------------------------------------
    dict(dataset="ogbg-molhiv", source="molnet2018", model="KernelSVM", family="conventional", metric="ROC-AUC", value=0.792,
         split_method="scaffold", n_molecules=41913, grade="C",
         locator="Table 3, Biophysics / HIV / conventional column",
         note="MoleculeNet Table 1 lists 41,913 HIV compounds; OGB's release has 41,127 after its own processing, "
              "so the sample set differs even before the split"),
    dict(dataset="ogbg-molhiv", source="molnet2018", model="GraphConv", family="graph", metric="ROC-AUC", value=0.763,
         split_method="scaffold", n_molecules=41913, grade="C",
         locator="Table 3, Biophysics / HIV / graph-based column"),
    dict(dataset="ogbg-molhiv", source="ogb2020", model="GIN + OGB features + virtual node", family="graph",
         metric="ROC-AUC", value=77.07, std=1.49, n_seeds=10, split_method="scaffold (OGB official)",
         n_molecules=41127, grade="A",
         locator="Table 15 (ogbg-molhiv), last GIN row; main text p. 8 fixes it as the best GIN "
                 "(82.73+-2.02% on the random split, 5.66 points above scaffold)",
         note="EXACT split match: these are the official row ids this round consumes. Still a 10-run mean "
              "with OGB's 9-d node / 3-d edge features, so the comparison mixes model family with input "
              "featurisation and must not be read as a pure architecture comparison."),
    dict(dataset="ogbg-molhiv", source="ogb2020", model="GCN + OGB features + virtual node", family="graph",
         metric="ROC-AUC", value=75.99, std=1.19, n_seeds=10, split_method="scaffold (OGB official)",
         n_molecules=41127, grade="A",
         locator="Table 15 (ogbg-molhiv), third GCN row",
         note="same exact split; the row is the GCN configuration with additional features and virtual node"),
]

DATASET_ORDER = list(C.DATASET_ORDER)


def build() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in ROWS:
        src = SOURCES[row["source"]]
        rows.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "family": row["family"],
                "metric": row["metric"],
                "value": row["value"],
                "std": row.get("std"),
                "n_seeds": row.get("n_seeds"),
                "split_method": row["split_method"],
                "n_molecules": row["n_molecules"],
                "grade": row["grade"],
                "grade_meaning": {
                    "A": "EXACT_SPLIT_MATCH",
                    "B": "PROTOCOL_MATCH_ONLY",
                    "C": "NOT_COMPARABLE",
                }[row["grade"]],
                "source_key": row["source"],
                "citation": src["citation"],
                "source_url": src["url"],
                "source_sha256": src["sha256"],
                "source_fetched": src["fetched"],
                "locator": row["locator"],
                "note": row.get("note", ""),
            }
        )
    return rows


COLUMNS = [
    "dataset",
    "model",
    "family",
    "metric",
    "value",
    "std",
    "n_seeds",
    "split_method",
    "n_molecules",
    "grade",
    "grade_meaning",
    "source_key",
    "citation",
    "source_url",
    "source_sha256",
    "source_fetched",
    "locator",
    "note",
]

EXACT_MATCH_FOUND = {
    "ogbg-molhiv": "YES: OGB paper Table 15 uses the official ogbg-molhiv row ids consumed here (grade A)",
    "ESOL": "NO_EXACT_MATCH_FOUND",
    "Lipophilicity": "NO_EXACT_MATCH_FOUND",
    "FreeSolv": "NO_EXACT_MATCH_FOUND",
    "BACE": "NO_EXACT_MATCH_FOUND",
    "BBBP": "NO_EXACT_MATCH_FOUND",
    "Tox21": "NO_EXACT_MATCH_FOUND",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build the literature matrix")
    parser.add_argument("--out", default=str(C.ANALYSIS / "literature_matrix.csv"))
    args = parser.parse_args(argv)
    rows = build()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    C.write_json(
        C.ANALYSIS / "literature_sources.json",
        {
            "schema": "lcrc-molnet-v2-literature-sources-v1",
            "round_id": C.ROUND_ID,
            "created_utc": C.utcnow(),
            "sources": SOURCES,
            "exact_match_found": EXACT_MATCH_FOUND,
            "n_rows": len(rows),
            "grades": {g: sum(1 for r in rows if r["grade"] == g) for g in "ABC"},
        },
    )
    C.log(f"wrote {C.track_rel(path)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
