"""Literature comparison matrix (P3).

Every number in ``analysis/literature_matrix.csv`` is transcribed from a fetched,
hash-pinned public document; the locator column says exactly where.  Nothing here
is recalled from memory, and nothing is re-computed.

Comparison grading (pre-registered in ``BENCHMARK_PROTOCOL.md``):

* ``A`` EXACT_SPLIT_MATCH      - the source publishes the split *indices* and they
                                are byte-identical to ours (none found).
* ``B`` PROTOCOL_MATCH_ONLY    - same dataset, same split method (random for the
                                three regression sets, scaffold for BACE), same
                                metric, same 80:10:10-ish proportions, but the
                                source does not publish indices we can match.
* ``C`` NOT_COMPARABLE         - different split *method* (e.g. scaffold numbers on
                                a random-split round), different molecule set, or
                                a different metric.

Usage:  uv run python .../code/mb_literature.py
"""

from __future__ import annotations

import argparse
import csv
from typing import Any, Sequence

import mb_common as C

# ---------------------------------------------------------------------------
# fetched, hash-pinned sources
# ---------------------------------------------------------------------------
SOURCES: dict[str, dict[str, Any]] = {
    "molnet2018": {
        "citation": "Wu et al., Chem. Sci. 9, 513-530 (2018) - MoleculeNet",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5868307/",
        "fetched": "2026-11-01T18:05:39Z",
        "sha256": "f5ba3f55520c70fee1f3bd3c23eae3f2dce74cdca8d1df7c7bf0ae6400323d07",
        "bytes": 242320,
        "note": (
            "Table 1 (dataset table: split method per dataset) and Table 3 (best performance per "
            "dataset, conventional vs graph-based). No per-example split indices are published; "
            "the paper reports one number per model, not a multi-seed mean."
        ),
    },
    "grover2020": {
        "citation": "Rong et al., NeurIPS 33 (2020) - GROVER, arXiv:2007.02835v2",
        "url": "https://arxiv.org/pdf/2007.02835v2",
        "fetched": "2026-11-01T18:07:26Z",
        "sha256": "d672b7f40e5cef743b19b43b6d2cfec0273b9fc98617bfeb968923f275b9ed81",
        "bytes": 2657733,
        "note": (
            "Table 1 (p. 7): classification ROC-AUC and regression RMSE with standard deviations. "
            "Sec. 4.2 and Appendix B.1 state: 'we apply the scaffold splitting for all tasks on all "
            "datasets' and 'three independent runs on three random-seeded scaffold splitting'. "
            "Split seeds are not published, so the exact indices cannot be reproduced or matched."
        ),
    },
    "chemprop2019": {
        "citation": "Yang et al., J. Chem. Inf. Model. 59, 3370-3389 (2019) - D-MPNN / chemprop, arXiv:1904.01561v5",
        "url": "https://arxiv.org/pdf/1904.01561v5",
        "fetched": "2026-11-01T18:08:20Z",
        "sha256": "f95ac4b49d196bc1b63e00e75fa7dff4709bce1d6ed0967f8d6a496d35a23447",
        "bytes": 2359573,
        "note": (
            "Table 4 gives the dataset sizes (ESOL 1128, FreeSolv 642, Lipophilicity 4200, BACE 1513 "
            "- identical to the CSVs used here). The per-dataset result panels are figures, not "
            "numeric tables, so no D-MPNN scalar can be transcribed from the text; this source is "
            "therefore only used to attest the molecule counts and the split protocol "
            "(10 randomly-seeded 80:10:10 splits, scaffold split for headline results, random-split "
            "results moved to the Supporting Information)."
        ),
    },
}

# ---------------------------------------------------------------------------
# transcribed rows
# ---------------------------------------------------------------------------
# (dataset, model, family, metric, value, std, split_method, n_rows, source, locator, notes)
ROWS: list[dict[str, Any]] = [
    # ---------------- ESOL (regression, random split, RMSE) ----------------
    dict(dataset="ESOL", model="XGBoost (2D descriptors)", family="conventional", metric="RMSE",
         value=0.99, std=None, split_method="random", n_rows=1128, source="molnet2018",
         locator="Table 3, 'Physical chemistry / ESOL / RMSE: XGBoost 0.99'",
         notes="single reported number, no seed spread"),
    dict(dataset="ESOL", model="MPNN (MoleculeNet)", family="graph", metric="RMSE",
         value=0.58, std=None, split_method="random", n_rows=1128, source="molnet2018",
         locator="Table 3, 'ESOL / RMSE ... graph-based: MPNN 0.58'",
         notes="single reported number, no seed spread"),
    dict(dataset="ESOL", model="GraphConv", family="graph", metric="RMSE",
         value=1.068, std=0.050, split_method="scaffold", n_rows=1128, source="grover2020",
         locator="Table 1, regression panel, row 'GraphConv'",
         notes="scaffold split; 3 random-seeded scaffold splits, seeds unpublished"),
    dict(dataset="ESOL", model="D-MPNN", family="graph", metric="RMSE",
         value=0.980, std=0.258, split_method="scaffold", n_rows=1128, source="grover2020",
         locator="Table 1, regression panel, row 'DMPNN'",
         notes="scaffold split as re-run by GROVER, not the chemprop headline number"),
    dict(dataset="ESOL", model="AttentiveFP", family="graph+attention", metric="RMSE",
         value=0.853, std=0.060, split_method="scaffold", n_rows=1128, source="grover2020",
         locator="Table 1, regression panel, row 'AttentiveFP'",
         notes="scaffold split"),
    dict(dataset="ESOL", model="GROVER-large", family="graph+transformer+SSL", metric="RMSE",
         value=0.831, std=0.120, split_method="scaffold", n_rows=1128, source="grover2020",
         locator="Table 1, regression panel, row 'GROVERlarge'",
         notes="scaffold split, ~100M parameters, self-supervised pre-training on 10M molecules"),
    # ---------------- FreeSolv (regression, random split, RMSE) ----------------
    dict(dataset="FreeSolv", model="XGBoost (2D descriptors)", family="conventional", metric="RMSE",
         value=1.74, std=None, split_method="random", n_rows=643, source="molnet2018",
         locator="Table 3, 'FreeSolv / RMSE: XGBoost 1.74'",
         notes="paper lists 643 compounds; the published CSV holds 642 (one molecule is not in the "
               "released table)"),
    dict(dataset="FreeSolv", model="MPNN (MoleculeNet)", family="graph", metric="RMSE",
         value=1.15, std=None, split_method="random", n_rows=643, source="molnet2018",
         locator="Table 3, 'FreeSolv / RMSE ... graph-based: MPNN 1.15'",
         notes="same 643/642 caveat"),
    dict(dataset="FreeSolv", model="D-MPNN", family="graph", metric="RMSE",
         value=2.177, std=0.914, split_method="scaffold", n_rows=642, source="grover2020",
         locator="Table 1, regression panel, row 'DMPNN'",
         notes="scaffold split; large spread across 3 scaffold draws"),
    dict(dataset="FreeSolv", model="AttentiveFP", family="graph+attention", metric="RMSE",
         value=2.030, std=0.420, split_method="scaffold", n_rows=642, source="grover2020",
         locator="Table 1, regression panel, row 'AttentiveFP'",
         notes="scaffold split"),
    dict(dataset="FreeSolv", model="GROVER-large", family="graph+transformer+SSL", metric="RMSE",
         value=1.544, std=0.397, split_method="scaffold", n_rows=642, source="grover2020",
         locator="Table 1, regression panel, row 'GROVERlarge'",
         notes="scaffold split"),
    # ---------------- Lipophilicity (regression, random split, RMSE) ----------------
    dict(dataset="Lipophilicity", model="XGBoost (2D descriptors)", family="conventional", metric="RMSE",
         value=0.799, std=None, split_method="random", n_rows=4200, source="molnet2018",
         locator="Table 3, 'Lipophilicity / RMSE: XGBoost 0.799'",
         notes="single reported number"),
    dict(dataset="Lipophilicity", model="GraphConv", family="graph", metric="RMSE",
         value=0.655, std=None, split_method="random", n_rows=4200, source="molnet2018",
         locator="Table 3, 'Lipophilicity / RMSE ... graph-based: GC 0.655'",
         notes="best graph-based number in MoleculeNet"),
    dict(dataset="Lipophilicity", model="D-MPNN", family="graph", metric="RMSE",
         value=0.653, std=0.046, split_method="scaffold", n_rows=4200, source="grover2020",
         locator="Table 1, regression panel, row 'DMPNN'",
         notes="scaffold split"),
    dict(dataset="Lipophilicity", model="AttentiveFP", family="graph+attention", metric="RMSE",
         value=0.650, std=0.030, split_method="scaffold", n_rows=4200, source="grover2020",
         locator="Table 1, regression panel, row 'AttentiveFP'",
         notes="scaffold split"),
    dict(dataset="Lipophilicity", model="GROVER-large", family="graph+transformer+SSL", metric="RMSE",
         value=0.560, std=0.035, split_method="scaffold", n_rows=4200, source="grover2020",
         locator="Table 1, regression panel, row 'GROVERlarge'",
         notes="scaffold split"),
    # ---------------- BACE (binary classification, scaffold split, ROC-AUC) ----------------
    dict(dataset="BACE", model="Random forest (ECFP)", family="conventional", metric="ROC-AUC",
         value=0.867, std=None, split_method="scaffold", n_rows=1522, source="molnet2018",
         locator="Table 3, 'BACE / AUC-ROC: RF 0.867' (conventional column)",
         notes="paper lists 1522 compounds; the released CSV used here holds 1513"),
    dict(dataset="BACE", model="Weave", family="graph", metric="ROC-AUC",
         value=0.806, std=None, split_method="scaffold", n_rows=1522, source="molnet2018",
         locator="Table 3, 'BACE / AUC-ROC ... graph-based: Weave 0.806'",
         notes="best graph-based number in MoleculeNet"),
    dict(dataset="BACE", model="GraphConv", family="graph", metric="ROC-AUC",
         value=0.854, std=0.011, split_method="scaffold", n_rows=1513, source="grover2020",
         locator="Table 1, classification panel, row 'GraphConv', column 'BACE'",
         notes="scaffold split, 1513 compounds - same molecule count as this round"),
    dict(dataset="BACE", model="D-MPNN", family="graph", metric="ROC-AUC",
         value=0.852, std=0.053, split_method="scaffold", n_rows=1513, source="grover2020",
         locator="Table 1, classification panel, row 'DMPNN', column 'BACE'",
         notes="scaffold split"),
    dict(dataset="BACE", model="AttentiveFP", family="graph+attention", metric="ROC-AUC",
         value=0.863, std=0.015, split_method="scaffold", n_rows=1513, source="grover2020",
         locator="Table 1, classification panel, row 'AttentiveFP', column 'BACE'",
         notes="scaffold split"),
    dict(dataset="BACE", model="GROVER-large", family="graph+transformer+SSL", metric="ROC-AUC",
         value=0.894, std=0.028, split_method="scaffold", n_rows=1513, source="grover2020",
         locator="Table 1, classification panel, row 'GROVERlarge', column 'BACE'",
         notes="scaffold split, self-supervised pre-training"),
]


def _round_split_method(dataset: str) -> str:
    meta = C.DATASETS[dataset]
    return str(meta["split_method"])


def grade(dataset: str, split_method: str) -> tuple[str, str]:
    """Return (grade, reason) for one literature row against this round."""
    ours = _round_split_method(dataset)
    if split_method != ours:
        return "C", (
            f"source uses a {split_method} split; this round uses the published {ours} repeat-0 "
            f"indices, so the numbers measure a different generalisation regime"
        )
    return "B", (
        f"same dataset, same {ours} split method, same metric and split sizes, but the source does "
        f"not publish the per-example indices (and uses its own seed draw), so index equality "
        f"cannot be verified"
    )


def build_rows() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in ROWS:
        g, reason = grade(row["dataset"], row["split_method"])
        src = SOURCES[row["source"]]
        out.append(
            {
                "dataset": row["dataset"],
                "task": C.DATASETS[row["dataset"]]["task"],
                "primary_metric": row["metric"],
                "model": row["model"],
                "model_family": row["family"],
                "value": f"{row['value']:.3f}",
                "std": "" if row["std"] is None else f"{row['std']:.3f}",
                "source_split_method": row["split_method"],
                "round_split_method": _round_split_method(row["dataset"]),
                "source_rows": row["n_rows"],
                "source_id": row["source"],
                "citation": src["citation"],
                "source_url": src["url"],
                "source_sha256": src["sha256"],
                "locator": row["locator"],
                "split_indices_published": "no",
                "split_indices_match": "NOT_VERIFIABLE",
                "grade": g,
                "grade_reason": reason,
                "notes": row["notes"],
            }
        )
    return out


FIELDS = list(build_rows()[0].keys())


def write_csv(rows: Sequence[dict[str, Any]], path) -> None:
    C.ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="literature comparison matrix")
    parser.parse_args(argv)
    rows = build_rows()
    path = C.ANALYSIS / "literature_matrix.csv"
    write_csv(rows, path)
    summary = {
        "rows": len(rows),
        "by_dataset": {
            d: {
                "n": sum(1 for r in rows if r["dataset"] == d),
                "grades": sorted({r["grade"] for r in rows if r["dataset"] == d}),
            }
            for d in C.DATASET_ORDER
        },
        "grade_counts": {
            g: sum(1 for r in rows if r["grade"] == g) for g in ("A", "B", "C")
        },
        "exact_match_found": any(r["grade"] == "A" for r in rows),
        "verdict": "NO_EXACT_MATCH_FOUND",
        "why": (
            "No published source releases the per-example split indices behind its numbers, and the "
            "published random/scaffold draws are seed-dependent, so A-grade (EXACT_SPLIT_MATCH) is "
            "unreachable for these four datasets. Within-grade B comparisons are still valid for "
            "the protocol-matched rows."
        ),
        "sources": SOURCES,
        "transcription": (
            "values copied by hand from the fetched documents; locators give the table and row. "
            "GROVER's regression panel is a scaffold-split panel, which is why its ESOL/FreeSolv/"
            "Lipophilicity numbers are graded C against a random-split round."
        ),
        "generated_utc": C.utcnow(),
    }
    C.write_json(C.ANALYSIS / "literature_sources.json", summary)
    C.log(f"wrote {C.track_rel(path)} ({len(rows)} rows) and analysis/literature_sources.json")
    for d in C.DATASET_ORDER:
        sub = [r for r in rows if r["dataset"] == d]
        C.log(f"  {d}: {len(sub)} rows, grades {sorted({r['grade'] for r in sub})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
