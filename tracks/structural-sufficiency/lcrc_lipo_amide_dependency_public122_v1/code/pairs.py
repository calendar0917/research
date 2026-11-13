"""Real legal molecule pairs (low-support side evidence, never blocking).

Reuses the **already deduplicated 823 unordered MMP row pairs** of the previous
round (``lcrc_lipo_edit_supervision_public122_s0_v1/analysis/pairs.csv``); no new
pair mining, no expansion, no new training, no transformation-family baseline.

Selection is structural only: ``|n_target(a) - n_target(b)| == 1`` where
``n_target`` is the number of restricted amide instances of protocol section 3
(``ALL`` instances, **not** ``MATCHED``).  The direction always puts the
molecule with more target instances first.

Reported per pair on the *original complete* molecules: ``dy``, ``df`` (frozen
checkpoint ``f(O)``), ``|df - dy|`` and the fixed zero-change baseline ``|dy|``.
Groups: ``P_train`` (both endpoints Train, descriptive only), ``E_T`` (>= 1 Test
endpoint) and ``E_HH`` (Valid-Test / Test-Test; LOW support).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import common as C
import prepare as P

PAIRS_SOURCE = C.EDITS / "analysis/pairs.csv"
PAIRS_EXPECTED_SHA256 = None  # filled from the first run and then frozen in inputs_manifest
EXPECTED = {
    "P_train": {"pairs": 38, "components": 17},
    "E_T": {"pairs": 21, "components": 8},
    "E_HH": {"pairs": 3, "components": 3},
}


def load_pairs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not PAIRS_SOURCE.exists():
        raise FileNotFoundError(f"MILEGACY_PAIRS_MISSING: {C.track_rel(PAIRS_SOURCE)}")
    sha = C.sha256_file(PAIRS_SOURCE)
    rows = P.read_csv(PAIRS_SOURCE)
    out = []
    for r in rows:
        d = dict(r)
        for k in (
            "row_lo",
            "row_hi",
            "row_a",
            "row_b",
            "in_p_train",
            "in_e_t",
            "in_e_hh",
            "eval_component",
            "train_component",
        ):
            d[k] = int(r[k])
        out.append(d)
    return out, {"path": C.track_rel(PAIRS_SOURCE), "sha256": sha, "n_rows": len(out)}


def orient_pair(row_a: int, row_b: int, n_a: int, n_b: int) -> tuple[int, int, int, int]:
    """Direction rule: the molecule with more target instances comes first."""
    if n_a < n_b:
        return int(row_b), int(row_a), int(n_b), int(n_a)
    return int(row_a), int(row_b), int(n_a), int(n_b)


def target_counts(inst: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for r in inst:
        counts[int(r["row_id"])] = counts.get(int(r["row_id"]), 0) + 1
    return counts


def summarise(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    if not rows:
        return {"n_pairs": 0, "n_components": 0}
    residual = np.asarray([abs(float(r["df"]) - float(r["dy"])) for r in rows], dtype=np.float64)
    baseline = np.asarray([abs(float(r["dy"])) for r in rows], dtype=np.float64)
    comps: dict[int, list[float]] = {}
    comp_base: dict[int, list[float]] = {}
    for r in rows:
        comps.setdefault(int(r["eval_component"]), []).append(abs(float(r["df"]) - float(r["dy"])))
        comp_base.setdefault(int(r["eval_component"]), []).append(abs(float(r["dy"])))
    comp_means = [float(np.mean(v)) for v in comps.values()]
    comp_base_means = [float(np.mean(v)) for v in comp_base.values()]
    return {
        "n_pairs": int(len(rows)),
        "n_components": int(len(comps)),
        "pair_equal_mae": float(residual.mean()),
        "component_equal_mae": float(np.mean(comp_means)),
        "zero_change_pair_equal_mae": float(baseline.mean()),
        "zero_change_component_equal_mae": float(np.mean(comp_base_means)),
        "median_abs_dy": float(np.median(baseline)),
        "mean_signed_df": float(np.mean([float(r["df"]) for r in rows])),
        "mean_signed_dy": float(np.mean([float(r["dy"]) for r in rows])),
        "n_pairs_df_closer_than_zero": int(np.sum(residual < baseline)),
        "n_pairs_zero_change_closer": int(np.sum(residual > baseline)),
        "n_pairs_equal": int(np.sum(residual == baseline)),
    }


def main() -> int:
    pairs, source = load_pairs()
    inst = P.read_csv(C.ANALYSIS / "motif_catalog.csv")
    preds = {
        (r["view_id"], r["scope"]): float(r["pred_raw"])
        for r in P.read_csv(C.ANALYSIS / "per_view_predictions.csv")
    }
    y = {int(r["row_id"]): float(r["y"]) for r in P.read_csv(C.ANALYSIS / "reference_o_predictions.csv")}
    split = {
        int(r["row_id"]): str(r["split"]) for r in P.read_csv(C.ANALYSIS / "reference_o_predictions.csv")
    }
    counts = target_counts(inst)

    selected: list[dict[str, Any]] = []
    excluded: dict[str, int] = {}
    for r in pairs:
        a, b = int(r["row_a"]), int(r["row_b"])
        na, nb = counts.get(a, 0), counts.get(b, 0)
        if abs(na - nb) != 1:
            excluded["abs_delta_targets_not_1"] = excluded.get("abs_delta_targets_not_1", 0) + 1
            continue
        a, b, na, nb = orient_pair(a, b, na, nb)
        if a not in y or b not in y:
            excluded["label_missing"] = excluded.get("label_missing", 0) + 1
            continue
        fa = preds[(f"{a}:O", "FULL")]
        fb = preds[(f"{b}:O", "FULL")]
        row = {
            "pair_id": str(r["pair_id"]),
            "row_a": a,
            "row_b": b,
            "split_a": split[a],
            "split_b": split[b],
            "split_combo": "-".join(sorted([split[a], split[b]])),
            "n_target_a": na,
            "n_target_b": nb,
            "in_p_train": int(r["in_p_train"]),
            "in_e_t": int(r["in_e_t"]),
            "in_e_hh": int(r["in_e_hh"]),
            "eval_component": int(r["eval_component"]),
            "y_a": float(y[a]),
            "y_b": float(y[b]),
            "f_a": float(fa),
            "f_b": float(fb),
            "dy": float(y[a] - y[b]),
            "df": float(fa - fb),
            "abs_residual": abs(float(fa - fb) - float(y[a] - y[b])),
            "abs_dy": abs(float(y[a] - y[b])),
            "smiles_a": str(r.get("smiles_a", "")),
            "smiles_b": str(r.get("smiles_b", "")),
        }
        selected.append(row)

    groups = {
        "P_train": [r for r in selected if r["in_p_train"] == 1],
        "E_T": [r for r in selected if r["in_e_t"] == 1],
        "E_HH": [r for r in selected if r["in_e_hh"] == 1],
    }
    summary = {
        "source": source,
        "n_source_pairs": int(source["n_rows"]),
        "n_selected": len(selected),
        "excluded": excluded,
        "selection": "|n_target(a) - n_target(b)| == 1; direction = more targets minus fewer targets",
        "groups": {k: summarise(v, k) for k, v in groups.items()},
    }
    mismatches = {}
    for name, want in EXPECTED.items():
        got = summary["groups"][name]
        if got["n_pairs"] != want["pairs"] or got["n_components"] != want["components"]:
            mismatches[name] = {
                "expected": want,
                "actual": {"pairs": got["n_pairs"], "components": got["n_components"]},
            }
    summary["precount_match"] = not mismatches
    summary["precount_mismatch"] = mismatches

    columns = [
        "pair_id",
        "row_a",
        "row_b",
        "split_a",
        "split_b",
        "split_combo",
        "n_target_a",
        "n_target_b",
        "in_p_train",
        "in_e_t",
        "in_e_hh",
        "eval_component",
        "y_a",
        "y_b",
        "f_a",
        "f_b",
        "dy",
        "df",
        "abs_residual",
        "abs_dy",
        "smiles_a",
        "smiles_b",
    ]
    P.write_csv(C.ANALYSIS / "natural_pairs.csv", selected, columns)
    C.write_json(C.ANALYSIS / "natural_pairs_summary.json", summary)
    C.log(
        "natural pairs: "
        + ", ".join(
            f"{k} {v['n_pairs']}p/{v['n_components']}c mae={v['pair_equal_mae']:.3f} vs |dy|={v['zero_change_pair_equal_mae']:.3f}"
            for k, v in summary["groups"].items()
        )
    )
    if mismatches:
        C.log(f"NATURAL_PAIR_PRECOUNT_MISMATCH: {mismatches}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
