"""Read-only replay of the existing 823-pair Lipo edit catalogue.

The parent round used only the amide 74-pair catalogue and kept same-split
pairs, so it could not see the Train-Valid / Train-Test edits at all.  This
script maps the existing ``lcrc_lipo_edit_supervision_public122_s0_v1``
``analysis/pairs.csv`` to official OGB rows through RDKit canonical SMILES,
re-derives the official split from the row id and computes paired-difference
errors of the **saved full F(G)** predictions.  It never re-mines pairs, never
changes legality and never selects pairs by model behaviour.

Because two pairs can share an endpoint, the support unit is the connected
component of the endpoint graph; component-equal MAE is reported next to the
pair-level MAE, and cells with fewer than 20 components are marked LOW_SUPPORT.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import fu_common as FC  # noqa: E402
import rd_common as RC  # noqa: E402
import rd_train as RT  # noqa: E402

DATASET = "ogbg-mollipo"
TASK = "lipo"
MODELS = ("RESP", "ADD", "JOINT")
SPLITS = ("train", "valid", "test")
LOW_SUPPORT_COMPONENTS = 20


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=str(FC.REPO_ROOT / FC.load_config()["edit_catalogue"]["path"]))
    parser.add_argument("--out", default=str(FC.ROUND_DIR / "analysis"))
    parser.add_argument("--joint-results", default=str(FC.run_dir(TASK, "JOINT")))
    parser.add_argument("--models", default="RESP,ADD,JOINT",
                        help="comma separated; JOINT requires the follow-up unit")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    payload = RC.dataset_cache(DATASET)
    smiles = [str(s) for s in payload["smiles"]]
    labels_all = np.asarray(payload["labels"], dtype=np.float64)
    official = np.empty(len(smiles), dtype=object)
    for split in SPLITS:
        official[np.asarray(payload[f"{split}_idx"], dtype=np.int64)] = split

    from rdkit import Chem

    canon = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles]
    if len(set(canon)) != len(canon):
        raise SystemExit("canonical SMILES are not unique across the official rows")
    row_of = {c: i for i, c in enumerate(canon)}

    pairs, mapping_report = _load_pairs(Path(args.pairs), row_of, Chem)
    pairs, dedup_report = _dedup(pairs)
    components = _components(pairs)
    mapping_report.update(dedup_report)
    mapping_report["n_components_all_pairs"] = int(len(set(components.values())))
    mapping_report["n_endpoints_all_pairs"] = int(len(components))

    wanted = tuple(m.strip() for m in str(args.models).split(",") if m.strip())
    models = _load_models(args.joint_results, payload, labels_all, official, wanted)
    rows, metrics = _metrics(pairs, components, models, labels_all)
    cells = _cells(pairs, components, models, labels_all)
    _write_csv(out / "edit_pairs.csv", rows)
    _write_csv(out / "edit_metrics_by_cell.csv", metrics)
    _write_csv(out / "edit_cells.csv", cells)
    RC.write_json(out / "edit_support.json", {
        "catalogue": str(Path(args.pairs)),
        "catalogue_sha256": RC.sha256_file(Path(args.pairs)),
        "mapping": mapping_report,
        "cells": cells,
        "metrics": metrics,
        "models": {k: v["source"] for k, v in models.items()},
        "low_support_rule": f"component count < {LOW_SUPPORT_COMPONENTS} -> LOW_SUPPORT (descriptive only)",
        "note": ("Train-Train pairs share a Train endpoint with the training set; they are not held-out "
                 "evidence. The only non-Train supports are the 27 Train-Test and 5 Train-Valid pairs, "
                 "which are 7/4 Test/Valid molecules inside 7/4 components. This diagnostic cannot "
                 "establish a chemical task law and is not a model-selection criterion."),
    })
    for cell in cells:
        print(f"{cell['cell']:12s} pairs {cell['n_pairs']:4d} endpoints {cell['n_endpoints']:4d} "
              f"components {cell['n_components']:3d} low_support={cell['low_support']}")
    for m in metrics:
        print(f"{m['model']:5s} {m['cell']:12s} pair_mae {m['pair_mae']:.4f} "
              f"ce_mae {m['component_equal_mae']:.4f} n_comp {m['n_components']}")
    return 0


# ---------------------------------------------------------------------------


def _load_pairs(path: Path, row_of: Mapping[str, int], Chem) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    problems: list[str] = []
    stored_row_mismatch = 0
    split_relabelled = 0
    out: list[dict[str, Any]] = []
    for r in rows:
        ca = Chem.MolToSmiles(Chem.MolFromSmiles(str(r["smiles_a"])))
        cb = Chem.MolToSmiles(Chem.MolFromSmiles(str(r["smiles_b"])))
        a, b = row_of.get(ca), row_of.get(cb)
        if a is None or b is None:
            problems.append(f"{r['pair_id']}: canonical SMILES not in the official rows")
            continue
        if (int(r["row_a"]), int(r["row_b"])) != (a, b):
            stored_row_mismatch += 1
        if (r["split_a"], r["split_b"]) != ("train", "train"):
            split_relabelled += 1
        out.append({"pair_id": str(r["pair_id"]), "row_a": int(a), "row_b": int(b),
                    "stored_split_a": r["split_a"], "stored_split_b": r["split_b"],
                    "class_id": r.get("class_id", ""), "core_smiles": r.get("core_smiles", "")})
    return out, {
        "source_rows": len(rows),
        "mapped_pairs": len(out),
        "unmapped": len(problems),
        "problems": problems[:10],
        "stored_row_columns_agree_with_canonical_mapping": stored_row_mismatch == 0,
        "rows_whose_stored_split_column_is_not_train_train": split_relabelled,
        "note": ("The catalogue's own split_a/split_b columns are not used: the official split is "
                 "re-derived from the mapped OGB row id. row_a/row_b are kept only as a cross-check."),
    }


def _dedup(pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen: dict[tuple[int, int], dict[str, Any]] = {}
    duplicates = 0
    self_pairs = 0
    for p in pairs:
        a, b = p["row_a"], p["row_b"]
        if a == b:
            self_pairs += 1
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            duplicates += 1
            continue
        seen[key] = {**p, "row_lo": key[0], "row_hi": key[1]}
    return list(seen.values()), {"n_unique_pairs": len(seen), "duplicate_pairs": duplicates,
                                 "self_pairs": self_pairs}


def _components(pairs: list[dict[str, Any]]) -> dict[int, int]:
    adj: dict[int, set[int]] = defaultdict(set)
    for p in pairs:
        adj[p["row_lo"]].add(p["row_hi"])
        adj[p["row_hi"]].add(p["row_lo"])
    comp: dict[int, int] = {}
    for node in adj:
        if node in comp:
            continue
        stack = [node]
        comp[node] = node
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in comp:
                    comp[v] = node
                    stack.append(v)
    # canonical component id = smallest row in the component
    groups: dict[int, list[int]] = defaultdict(list)
    for node, root in comp.items():
        groups[root].append(node)
    final: dict[int, int] = {}
    for root, nodes in groups.items():
        cid = min(nodes)
        for node in nodes:
            final[node] = cid
    return final


def _load_models(joint_results: str, payload, labels_all, official, wanted=MODELS) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for model in wanted:
        base = (Path(joint_results) if model == "JOINT" else FC.S0_RESULTS / f"{TASK}_{model}")
        pred_path = base / "predictions_best.npz"
        if not pred_path.is_file():
            raise SystemExit(f"missing predictions for {model}: {pred_path}")
        with np.load(pred_path) as data:
            pm: dict[int, float] = {}
            for split in SPLITS:
                rows = np.asarray(data[f"{split}_row_ids"], dtype=np.int64)
                preds = np.asarray(data[f"{split}_pred_raw"], dtype=np.float64)
                targets = np.asarray(data[f"{split}_target_raw"], dtype=np.float64)
                if not np.array_equal(np.sort(rows), np.sort(np.asarray(payload[f"{split}_idx"], dtype=np.int64))):
                    raise SystemExit(f"{model}: {split} row ids differ from the official split")
                if not np.allclose(targets, labels_all[rows], atol=1e-9):
                    raise SystemExit(f"{model}: {split} labels differ from the official labels")
                if not np.all(np.asarray([str(official[r]) for r in rows.tolist()], dtype=object) == split):
                    raise SystemExit(f"{model}: {split} split membership mismatch")
                pm.update({int(r): float(p) for r, p in zip(rows.tolist(), preds.tolist())})
        out[model] = {
            "pred": pm,
            "source": str(pred_path),
            "source_sha256": RC.sha256_file(pred_path),
            "selected_epoch": int(np.load(pred_path)["selected_epoch"][0]),
        }
    return out


def _cell_name(sa: str, sb: str) -> str:
    ordered = sorted((sa, sb), key=lambda s: SPLITS.index(s))
    return "-".join(s.capitalize() for s in ordered)


def _rows_of_pairs(pairs, components, official) -> list[dict[str, Any]]:
    rows = []
    for p in pairs:
        sa, sb = str(official[p["row_lo"]]), str(official[p["row_hi"]])
        rows.append({**p, "split_lo": sa, "split_hi": sb,
                     "cell": _cell_name(sa, sb),
                     "component_id": int(components[p["row_lo"]])})
    return rows


def _metrics(pairs, components, models, labels_all) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    official = _official_of(pairs)
    rows = _rows_of_pairs(pairs, components, official)
    cells = sorted({r["cell"] for r in rows})
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)
    table = []
    for model, spec in models.items():
        for cell in cells:
            sub = by_cell[cell]
            err = np.asarray([spec["pred"][r["row_hi"]] - spec["pred"][r["row_lo"]]
                              - (labels_all[r["row_hi"]] - labels_all[r["row_lo"]]) for r in sub])
            comp_ids = np.asarray([r["component_id"] for r in sub])
            uniq = sorted(set(comp_ids.tolist()))
            comp_mae = [float(np.mean(np.abs(err[comp_ids == c]))) for c in uniq]
            comp_mse = [float(np.mean(err[comp_ids == c] ** 2)) for c in uniq]
            zero = np.asarray([abs(labels_all[r["row_hi"]] - labels_all[r["row_lo"]]) for r in sub])
            comp_zero = [float(np.mean(zero[comp_ids == c])) for c in uniq]
            table.append({
                "model": model, "cell": cell, "n_pairs": int(len(sub)),
                "n_components": int(len(uniq)),
                "pair_rmse": float(np.sqrt(np.mean(err ** 2))),
                "pair_mae": float(np.mean(np.abs(err))),
                "mean_signed_error": float(np.mean(err)),
                "component_equal_mae": float(np.mean(comp_mae)),
                "component_equal_rmse": float(np.sqrt(np.mean(comp_mse))),
                "zero_change_pair_mae": float(np.mean(zero)),
                "zero_change_component_equal_mae": float(np.mean(comp_zero)),
                "low_support": bool(len(uniq) < LOW_SUPPORT_COMPONENTS),
            })
    out_rows = []
    for r in rows:
        dy = labels_all[r["row_hi"]] - labels_all[r["row_lo"]]
        entry = {"pair_id": r["pair_id"], "cell": r["cell"], "component_id": r["component_id"],
                 "row_lo": r["row_lo"], "row_hi": r["row_hi"],
                 "split_lo": r["split_lo"], "split_hi": r["split_hi"],
                 "y_lo": float(labels_all[r["row_lo"]]), "y_hi": float(labels_all[r["row_hi"]]),
                 "delta_y": float(dy), "class_id": r.get("class_id", "")}
        for model, spec in models.items():
            dp = spec["pred"][r["row_hi"]] - spec["pred"][r["row_lo"]]
            entry[f"delta_pred_{model}"] = float(dp)
            entry[f"error_{model}"] = float(dp - dy)
        out_rows.append(entry)
    return out_rows, table


def _official_of(pairs) -> np.ndarray:
    # the split of every endpoint, re-derived once from the official cache
    payload = RC.dataset_cache(DATASET)
    official = np.empty(len(payload["smiles"]), dtype=object)
    for split in SPLITS:
        official[np.asarray(payload[f"{split}_idx"], dtype=np.int64)] = split
    return official


def _cells(pairs, components, models, labels_all) -> list[dict[str, Any]]:
    official = _official_of(pairs)
    rows = _rows_of_pairs(pairs, components, official)
    out = []
    for cell in sorted({r["cell"] for r in rows}):
        sub = [r for r in rows if r["cell"] == cell]
        endpoints = set()
        for r in sub:
            endpoints.update((r["row_lo"], r["row_hi"]))
        non_train = [e for e in endpoints if official[e] != "train"]
        out.append({
            "cell": cell,
            "n_pairs": len(sub),
            "n_endpoints": len(endpoints),
            "n_components": len({r["component_id"] for r in sub}),
            "n_non_train_endpoints": len(non_train),
            "non_train_endpoints_by_split": {s: int(sum(1 for e in non_train if official[e] == s))
                                             for s in ("valid", "test")},
            "low_support": bool(len({r["component_id"] for r in sub}) < LOW_SUPPORT_COMPONENTS),
            "n_train_endpoints": int(len(endpoints) - len(non_train)),
        })
    return out


def _write_csv(path: Path, rows) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    raise SystemExit(main())
