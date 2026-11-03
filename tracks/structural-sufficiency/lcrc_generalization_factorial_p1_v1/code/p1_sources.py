"""Frozen-input loading, strict alignment and integrity auditing for LCRC-P1.

Read-only with respect to every historical artefact: this module never writes
into ``lcrc_performance_longrun_v1``, ``lcrc_failure_mechanism_p0_v1`` or the
chembench source tree.

Loads:

* the three frozen ChemBench Lipophilicity splits and the shared 4,200-molecule
  ``Lipop.csv.gz`` label table;
* label-free molecule records (rooted radius-2 signatures, Bemis-Murcko
  scaffolds, descriptors, Morgan fingerprints) via the P0 machinery;
* per-molecule predictions of every unit (reused historical run or fresh run of
  this round), strictly aligned to the frozen row order.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from p1_common import (
    EXPECTED_SIZES,
    LIPOP_CSV,
    LIPOP_SPLIT_DIR,
    LONGRUN_RUNS,
    N_MOLECULES,
    P0_CODE,
    PART_ORDER,
    SPLITS,
    TOL,
    TOP_FRACTION,
    read_json,
    row_sha,
    sha256_file,
    track_rel,
    unit_registry,
)
import p1_stats as S

#: the P0 round provides the label-free structural machinery; imported read-only
if str(P0_CODE) not in sys.path:
    sys.path.insert(0, str(P0_CODE))

import p0_structure as PS  # noqa: E402  (frozen, committed, label-free)


class AlignmentError(RuntimeError):
    """Raised when a prediction file is not aligned to the frozen row order."""


# ---------------------------------------------------------------------------
# frozen split
# ---------------------------------------------------------------------------


@dataclass
class FrozenSplit:
    name: str
    longrun_split: str
    path: Path
    file_sha256: str
    rows: dict[str, np.ndarray]
    rows_sha256: dict[str, str]
    labels: dict[str, np.ndarray]
    smiles: list[str]
    molecules: list[Any] = field(repr=False, default_factory=list)
    support: Any = field(repr=False, default=None)

    @property
    def scaffold(self) -> list[str]:
        return [rec.scaffold for rec in self.molecules]


@dataclass
class Unit:
    key: str
    split: str
    arm: str
    seed: int
    origin: str
    run_dir: Path
    train_json: dict[str, Any]
    metrics: dict[str, dict[str, float]]
    rows: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    pred: dict[str, np.ndarray]
    standardised: dict[str, np.ndarray]
    test_rows: np.ndarray | None
    checks: list[dict[str, Any]] = field(default_factory=list)

    def ae(self, part: str) -> np.ndarray:
        return S.absolute_error(self.y[part], self.pred[part])

    def se(self, part: str) -> np.ndarray:
        return S.squared_error(self.y[part], self.pred[part])


# ---------------------------------------------------------------------------
# CSV + molecule table
# ---------------------------------------------------------------------------


def read_csv_table() -> dict[str, Any]:
    frame = pd.read_csv(LIPOP_CSV)
    return {
        "frame": frame,
        "columns": list(frame.columns),
        "n": int(len(frame)),
        "smiles": [str(s) for s in frame["smiles"].tolist()],
        "labels": np.asarray(frame["exp"].to_numpy(), dtype=np.float64),
        "sha256": sha256_file(LIPOP_CSV),
    }


def read_pickle_split(path: Path) -> dict[str, np.ndarray]:
    series = pd.read_pickle(path)
    parts = [list(part) for part in series.to_list()]
    if len(parts) != 3:
        raise ValueError(f"{path}: expected 3 parts, got {len(parts)}")
    return {name: np.asarray(part, dtype=np.int64) for name, part in zip(PART_ORDER, parts)}


def _recorded_rows(train_json: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, int]]:
    """Row hashes/sizes a frozen run recorded for its split, from both places."""
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    record = train_json.get("split_record", {}) or {}
    for part, entry in (record.get("rows") or {}).items():
        if isinstance(entry, Mapping) and entry.get("sha256"):
            hashes[part] = str(entry["sha256"])
            if entry.get("n") is not None:
                sizes[part] = int(entry["n"])
    source = train_json.get("source", {}) or {}
    recorded = source.get("rows_sha256")
    if isinstance(recorded, Mapping):
        for part, value in recorded.items():
            prev = hashes.get(part)
            if prev is not None and prev != value:
                raise RuntimeError(f"conflicting recorded row hashes for {part}: {prev} vs {value}")
            hashes[part] = str(value)
    elif isinstance(recorded, (list, tuple)):
        for part, value in zip(PART_ORDER, recorded):
            prev = hashes.get(part)
            if prev is not None and prev != value:
                raise RuntimeError(f"conflicting recorded row hashes for {part}: {prev} vs {value}")
            hashes[part] = str(value)
    for part in PART_ORDER:
        sizes.setdefault(part, EXPECTED_SIZES[part])
    return hashes, sizes


def expected_rows_hashes() -> dict[str, dict[str, Any]]:
    """Row hashes recorded by the frozen historical runs, per split."""
    out: dict[str, dict[str, Any]] = {}
    for split in ("random122", "random123", "random124"):
        rec = None
        for unit in unit_registry():
            if unit["split"] != split:
                continue
            train_path = unit["run_dir"] / "train.json"
            if not train_path.exists():
                continue
            train_json = read_json(train_path)
            hashes, sizes = _recorded_rows(train_json)
            current = {"sha256": hashes, "n": sizes}
            if rec is None:
                rec = current
            elif rec != current:
                raise RuntimeError(
                    f"{split}: frozen runs disagree about the split rows: {rec} vs {current} ({unit['key']})"
                )
        if rec is None:
            raise RuntimeError(f"{split}: no frozen run records the split rows")
        out[split] = rec
    return out


def load_splits(*, build_support: bool = True, verbose: bool = False) -> tuple[dict[str, FrozenSplit], dict[str, Any]]:
    """Load all three frozen splits and audit their row/label/molecule identity."""
    csv = read_csv_table()
    labels_all = csv["labels"]
    smiles = csv["smiles"]
    checks: list[dict[str, Any]] = []

    unique_smiles = len(set(smiles))
    checks.append(
        {
            "check": "csv_smiles_unique",
            "passed": unique_smiles == N_MOLECULES == len(smiles),
            "detail": {"n_rows": len(smiles), "n_unique_smiles": unique_smiles},
        }
    )
    checks.append(
        {
            "check": "csv_columns",
            "passed": csv["columns"] == ["smiles", "exp"],
            "detail": csv["columns"],
        }
    )

    # molecule table is split-independent: row id -> structure is a bijection
    molecules = PS.build_molecule_table(smiles, verbose=verbose)
    n_parse_fail = int(sum(1 for rec in molecules if not rec.parse_ok))
    checks.append(
        {
            "check": "molecule_table_parse",
            "passed": n_parse_fail == 0,
            "detail": {
                "n_parse_failures": n_parse_fail,
                "n_rows": len(molecules),
                "n_empty_scaffold_fallback": int(sum(1 for r in molecules if r.scaffold_fallback_used)),
                "n_multi_component": int(sum(1 for r in molecules if r.multi_component)),
            },
        }
    )
    scaffolds = [rec.scaffold for rec in molecules]

    expected = expected_rows_hashes()
    splits: dict[str, FrozenSplit] = {}
    for name in ("random122", "random123", "random124"):
        path = LIPOP_SPLIT_DIR / SPLITS[name]["file"]
        file_sha = sha256_file(path)
        rows = read_pickle_split(path)
        rows_sha = {part: row_sha(rows[part]) for part in PART_ORDER}
        sizes = {part: int(len(rows[part])) for part in PART_ORDER}

        checks.append(
            {
                "check": f"split_sizes_{name}",
                "passed": sizes == EXPECTED_SIZES,
                "detail": sizes,
            }
        )
        checks.append(
            {
                "check": f"split_rows_sha256_match_frozen_runs_{name}",
                "passed": all(rows_sha[part] == value for part, value in expected[name]["sha256"].items())
                and all(part in expected[name]["sha256"] for part in ("train", "valid")),
                "detail": {
                    "rows_sha256": rows_sha,
                    "recorded": expected[name]["sha256"],
                    "recorded_parts": sorted(expected[name]["sha256"]),
                },
            }
        )
        checks.append(
            {
                "check": f"split_pickle_sha256_recorded_{name}",
                "passed": all(
                    read_json(unit["run_dir"] / "train.json").get("source", {}).get("split_file_sha256")
                    in (None, file_sha)
                    for unit in unit_registry()
                    if unit["split"] == name and (unit["run_dir"] / "train.json").exists()
                ),
                "detail": file_sha,
            }
        )
        checks.append(
            {
                "check": f"split_rows_unique_{name}",
                "passed": all(len(np.unique(rows[p])) == len(rows[p]) for p in PART_ORDER),
                "detail": sizes,
            }
        )
        union = np.union1d(np.union1d(rows["train"], rows["valid"]), rows["test"])
        intersections = {
            "train_valid": int(len(np.intersect1d(rows["train"], rows["valid"]))),
            "train_test": int(len(np.intersect1d(rows["train"], rows["test"]))),
            "valid_test": int(len(np.intersect1d(rows["valid"], rows["test"]))),
        }
        checks.append(
            {
                "check": f"split_row_isolation_{name}",
                "passed": all(v == 0 for v in intersections.values())
                and len(union) == N_MOLECULES
                and int(union.min()) == 0
                and int(union.max()) == N_MOLECULES - 1,
                "detail": {**intersections, "union": int(len(union))},
            }
        )

        labels = {part: labels_all[rows[part]] for part in PART_ORDER}
        entry = FrozenSplit(
            name=name,
            longrun_split=SPLITS[name]["longrun_split"],
            path=path,
            file_sha256=file_sha,
            rows=rows,
            rows_sha256=rows_sha,
            labels=labels,
            smiles=smiles,
            molecules=molecules,
        )
        if build_support:
            entry.support = PS.build_support(
                group_key=f"LCRC-P1/{name}",
                train_rows=rows["train"],
                query_splits={"train": rows["train"], "valid": rows["valid"], "test": rows["test"]},
                molecules=molecules,
                include_loo=True,
            )
        splits[name] = entry

    # cross-split per-molecule consistency: each split must reconstruct the same
    # row-id -> label vector over the full 4,200-row table, and row ids must be a
    # bijection onto unique SMILES (so a permuted index space would be visible).
    per_split_labels: dict[str, np.ndarray] = {}
    for name, split_obj in splits.items():
        mapping = np.full(N_MOLECULES, np.nan)
        for part in PART_ORDER:
            mapping[split_obj.rows[part]] = split_obj.labels[part]
        per_split_labels[name] = mapping
        checks.append(
            {
                "check": f"split_covers_all_rows_{name}",
                "passed": bool(np.all(np.isfinite(mapping))) and int(np.sum(np.isfinite(mapping))) == N_MOLECULES,
                "detail": int(np.sum(np.isfinite(mapping))),
            }
        )
    reference = splits["random122"]
    for name in ("random123", "random124"):
        other = splits[name]
        same_labels = np.array_equal(per_split_labels["random122"], per_split_labels[name], equal_nan=True)
        same_smiles = all(reference.smiles[i] == other.smiles[i] for i in range(N_MOLECULES))
        checks.append(
            {
                "check": f"cross_split_molecule_identity_random122_vs_{name}",
                "passed": bool(same_labels and same_smiles and unique_smiles == N_MOLECULES),
                "detail": {
                    "same_label_by_row_id": bool(same_labels),
                    "same_smiles_by_row_id": bool(same_smiles),
                    "row_id_to_unique_smiles_is_bijection": bool(unique_smiles == N_MOLECULES),
                    "n_rows_checked": N_MOLECULES,
                    "scaffold_source": "derived deterministically from the row-id SMILES; split-independent",
                },
            }
        )

    audit = {
        "csv": {
            "path": track_rel(LIPOP_CSV),
            "sha256": csv["sha256"],
            "n_rows": csv["n"],
            "columns": csv["columns"],
        },
        "splits": {
            name: {
                "longrun_split": s.longrun_split,
                "path": track_rel(s.path),
                "sha256": s.file_sha256,
                "rows_sha256": s.rows_sha256,
                "sizes": {p: int(len(s.rows[p])) for p in PART_ORDER},
                "n_unique_scaffolds": int(len(set(scaffolds))),
                "n_acyclic_fallback": int(sum(1 for r in molecules if r.scaffold_fallback_used)),
                "train_scaffold_coverage": {
                    "n_train_scaffolds": int(len({scaffolds[i] for i in s.rows["train"]})),
                    "n_valid_scaffolds_seen_in_train": int(
                        len({scaffolds[i] for i in s.rows["valid"]} & {scaffolds[i] for i in s.rows["train"]})
                    ),
                    "n_test_scaffolds_seen_in_train": int(
                        len({scaffolds[i] for i in s.rows["test"]} & {scaffolds[i] for i in s.rows["train"]})
                    ),
                },
            }
            for name, s in splits.items()
        },
        "checks": checks,
    }
    return splits, audit


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------


def strict_align(rows: np.ndarray, values: np.ndarray, expected: np.ndarray, *, label: str) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    expected = np.asarray(expected, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if not np.array_equal(rows, expected):
        raise AlignmentError(f"{label}: prediction rows are not in frozen split order")
    return values.reshape(-1)


def _destandardise(raw: np.ndarray, norm: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(raw, dtype=np.float64).reshape(-1) * float(norm["scale"]) + float(norm["mean"])


def load_unit(spec: Mapping[str, Any], split: FrozenSplit) -> Unit:
    """Load one unit's frozen artefacts and verify its identity and alignment."""
    run_dir = Path(spec["run_dir"])
    train_json = read_json(run_dir / "train.json")
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"check": f"{spec['key']}_{name}", "passed": bool(passed), "detail": detail})

    recorded_split = train_json.get("split")
    add(
        "identity",
        train_json.get("dataset") == "Lipophilicity"
        and recorded_split == split.longrun_split
        and train_json.get("arm") == spec["arm"]
        and int(train_json.get("seed")) == int(spec["seed"]),
        {
            "dataset": train_json.get("dataset"),
            "split": recorded_split,
            "arm": train_json.get("arm"),
            "seed": train_json.get("seed"),
            "expected": {"split": split.longrun_split, "arm": spec["arm"], "seed": spec["seed"]},
        },
    )

    norm = train_json["data"]["target_norm"]
    add(
        "target_norm_train_only",
        norm["source"] == "train_split_only" and norm["valid_or_test_used"] is False and int(norm["n"]) == EXPECTED_SIZES["train"],
        {"source": norm["source"], "n": norm["n"], "mean": norm["mean"], "scale": norm["scale"]},
    )

    recipe = train_json.get("recipe", {})
    add(
        "checkpoint_rule_uses_validation_only",
        recipe.get("validation_used_for_checkpoint_selection") is True
        and recipe.get("test_used_for_any_selection") is False
        and train_json.get("checkpoint", {}).get("selection_uses_test") is False
        and train_json.get("test_used_for_development") is True,
        {
            "validation_used_for_checkpoint_selection": recipe.get("validation_used_for_checkpoint_selection"),
            "test_used_for_any_selection": recipe.get("test_used_for_any_selection"),
            "selection_uses_test": train_json.get("checkpoint", {}).get("selection_uses_test"),
            "test_used_for_development": train_json.get("test_used_for_development"),
        },
    )

    model = train_json.get("model", {})
    if spec["arm"] != "base":
        zero = model.get("extension_zero_init") or {}
        add(
            "extension_zero_init",
            bool(zero.get("passed")) is True,
            zero,
        )
        add(
            "shared_tensors_copied_from_base",
            model.get("base_init_state_hash") is not None and bool(model.get("base_copy")),
            {"base_init_state_hash": model.get("base_init_state_hash"), "base_copy": model.get("base_copy")},
        )

    # predictions: valid is stored standardised, Test is stored de-standardised
    valid_npz = np.load(run_dir / "valid_predictions.npz", allow_pickle=False)
    test_npz = np.load(run_dir / "test_predictions.npz", allow_pickle=False)
    valid_rows = np.asarray(valid_npz["rows"], dtype=np.int64)
    test_rows = np.asarray(test_npz["rows"], dtype=np.int64)
    raw_valid = strict_align(valid_rows, valid_npz["raw"], split.rows["valid"], label=f"{spec['key']}/valid")
    raw_test = strict_align(test_rows, test_npz["raw"], split.rows["test"], label=f"{spec['key']}/test")
    y_valid_npz = strict_align(valid_rows, valid_npz["y_true"], split.rows["valid"], label=f"{spec['key']}/valid/y")
    y_test_npz = strict_align(test_rows, test_npz["y_true"], split.rows["test"], label=f"{spec['key']}/test/y")

    pred_valid = _destandardise(raw_valid, norm)
    pred_test = _destandardise(raw_test, norm)
    if "pred" in test_npz.files:
        pred_test_stored = strict_align(test_rows, test_npz["pred"], split.rows["test"], label=f"{spec['key']}/test/pred")
        max_diff = float(np.max(np.abs(pred_test_stored - pred_test)))
        add("test_pred_is_destandardised_raw", max_diff <= 1e-9, max_diff)
        pred_test = pred_test_stored

    add(
        "y_true_matches_frozen_split_labels_valid",
        float(np.max(np.abs(y_valid_npz - split.labels["valid"]))) == 0.0,
        float(np.max(np.abs(y_valid_npz - split.labels["valid"]))),
    )
    add(
        "y_true_matches_frozen_split_labels_test",
        float(np.max(np.abs(y_test_npz - split.labels["test"]))) == 0.0,
        float(np.max(np.abs(y_test_npz - split.labels["test"]))),
    )
    add("predictions_finite", bool(np.isfinite(pred_valid).all() and np.isfinite(pred_test).all()), {
        "valid_nonfinite": int(np.sum(~np.isfinite(pred_valid))),
        "test_nonfinite": int(np.sum(~np.isfinite(pred_test))),
    })

    recorded = train_json.get("metrics", {})
    recomputed_valid = S.metrics(split.labels["valid"], pred_valid)
    recomputed_test = S.metrics(split.labels["test"], pred_test)
    for part, recomputed in (("valid", recomputed_valid), ("test", recomputed_test)):
        rec = recorded.get(part) or {}
        for metric in ("rmse", "mae"):
            ok = rec.get(metric) is not None and abs(float(rec[metric]) - float(recomputed[metric])) <= TOL
            add(f"metric_reproduction_{part}_{metric}", ok, {
                "recorded": rec.get(metric),
                "recomputed": float(recomputed[metric]),
                "abs_diff": None if rec.get(metric) is None else abs(float(rec[metric]) - float(recomputed[metric])),
            })
        rec_sse = float(rec.get("n", 0)) * float(rec.get("rmse", 0.0)) ** 2
        add(f"metric_reproduction_{part}_sse", abs(rec_sse - float(recomputed["sse"])) <= max(TOL, 1e-6), {
            "recorded": rec_sse,
            "recomputed": float(recomputed["sse"]),
        })
        add(f"n_{part}", int(rec.get("n", -1)) == int(recomputed["n"]), {"recorded": rec.get("n"), "recomputed": int(recomputed["n"])})

    test_eval_path = run_dir / "test_eval.json"
    test_rows_eval = None
    if test_eval_path.exists():
        test_eval = read_json(test_eval_path)
        test_rows_eval = np.asarray(test_eval.get("test_rows", []), dtype=np.int64)
        add(
            "test_eval_rows_match_split",
            test_rows_eval.shape == split.rows["test"].shape and np.array_equal(test_rows_eval, split.rows["test"]),
            int(test_rows_eval.size),
        )
        add(
            "test_eval_metric_reproduction",
            abs(float(test_eval["metrics"]["rmse"]) - float(recomputed_test["rmse"])) <= TOL
            and abs(float(test_eval["metrics"]["mae"]) - float(recomputed_test["mae"])) <= TOL,
            {
                "recorded_rmse": test_eval["metrics"]["rmse"],
                "recomputed_rmse": float(recomputed_test["rmse"]),
            },
        )
        add(
            "test_eval_exposure_label",
            "DEVELOPMENT_TEST" in str(test_eval.get("exposure", "")),
            test_eval.get("exposure"),
        )

    return Unit(
        key=spec["key"],
        split=spec["split"],
        arm=spec["arm"],
        seed=int(spec["seed"]),
        origin=spec["origin"],
        run_dir=run_dir,
        train_json=train_json,
        metrics={"valid": recomputed_valid, "test": recomputed_test},
        rows={"valid": valid_rows, "test": test_rows},
        y={"valid": split.labels["valid"], "test": split.labels["test"]},
        pred={"valid": pred_valid, "test": pred_test},
        standardised={"valid": raw_valid, "test": raw_test},
        test_rows=test_rows_eval,
        checks=checks,
    )


def load_all_units(splits: Mapping[str, FrozenSplit], *, verbose: bool = False) -> dict[tuple[str, str, int], Unit]:
    units: dict[tuple[str, str, int], Unit] = {}
    for spec in unit_registry():
        if not (Path(spec["run_dir"]) / "train.json").exists():
            if verbose:
                print(f"  missing unit {spec['key']} ({track_rel(spec['run_dir'])})", flush=True)
            continue
        unit = load_unit(spec, splits[spec["split"]])
        units[(unit.split, unit.arm, unit.seed)] = unit
        if verbose:
            print(f"  loaded {unit.key} [{unit.origin}]", flush=True)
    return units


# ---------------------------------------------------------------------------
# per-molecule feature frame
# ---------------------------------------------------------------------------


def molecule_frame(split: FrozenSplit) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Valid/Test feature frame with Train-fitted quantile bins."""
    molecules = split.molecules
    scaffolds = [rec.scaffold for rec in molecules]
    train_scaffolds = {scaffolds[i] for i in split.rows["train"]}

    train_loo = split.support.train_loo
    train_heavy = np.asarray([molecules[i].heavy_atoms for i in split.rows["train"]], dtype=np.float64)
    train_tpsa = np.asarray([molecules[i].tpsa for i in split.rows["train"]], dtype=np.float64)
    train_tani = np.asarray(train_loo["max_train_tanimoto"].to_numpy(), dtype=np.float64)

    edges = {
        "heavy_atoms": S.quantile_edges(train_heavy),
        "tpsa": S.quantile_edges(train_tpsa),
        "max_train_tanimoto": S.quantile_edges(train_tani),
    }
    train_bin_counts = {
        key: np.bincount(S.bin_index(values, edges[key]).clip(min=0), minlength=4).tolist()
        for key, values in (
            ("heavy_atoms", train_heavy),
            ("tpsa", train_tpsa),
            ("max_train_tanimoto", train_tani),
        )
    }

    frames = []
    support_query = split.support.query
    for part in ("valid", "test"):
        rows = split.rows[part]
        q = support_query[support_query["split"] == part].reset_index(drop=True)
        if len(q) != len(rows):
            raise RuntimeError(f"{split.name}/{part}: support frame has {len(q)} rows, split has {len(rows)}")
        if not np.array_equal(q["row_id"].to_numpy(dtype=np.int64), rows):
            raise AlignmentError(f"{split.name}/{part}: support frame is not in frozen split order")
        frame = pd.DataFrame(
            {
                "split": split.name,
                "part": part,
                "row_id": rows,
                "y_true": split.labels[part],
                "smiles": q["smiles"].to_numpy(),
                "scaffold": q["scaffold"].to_numpy(),
                "scaffold_seen": [bool(scaffolds[i] in train_scaffolds) for i in rows],
                "heavy_atoms": q["heavy_atoms"].to_numpy(dtype=np.float64),
                "tpsa": q["tpsa"].to_numpy(dtype=np.float64),
                "max_train_tanimoto": q["max_train_tanimoto"].to_numpy(dtype=np.float64),
                "unseen_atom_frac": q["unseen_atom_frac"].to_numpy(dtype=np.float64),
                "rare_atom_frac": q["rare_atom_frac"].to_numpy(dtype=np.float64),
                "median_log1p_df": q["median_log1p_df"].to_numpy(dtype=np.float64),
                "p10_log1p_df": q["p10_log1p_df"].to_numpy(dtype=np.float64),
            }
        )
        for key, edge in edges.items():
            frame[f"{key}_bin"] = S.bin_labels(frame[key].to_numpy(), edge, key)
        frames.append(frame)

    meta = {
        "split": split.name,
        "quantile_edges": {k: [float(x) for x in v] for k, v in edges.items()},
        "quantile_edge_source": "Train molecules of the same split only",
        "max_train_tanimoto_train_side": "leave-one-molecule-out maximum over the other Train molecules",
        "max_train_tanimoto_query_side": "maximum over all Train molecules",
        "train_bin_counts": train_bin_counts,
        "n_train": int(len(split.rows["train"])),
    }
    return pd.concat(frames, ignore_index=True), meta


__all__ = [
    "AlignmentError",
    "FrozenSplit",
    "Unit",
    "read_csv_table",
    "read_pickle_split",
    "expected_rows_hashes",
    "load_splits",
    "strict_align",
    "load_unit",
    "load_all_units",
    "molecule_frame",
]
