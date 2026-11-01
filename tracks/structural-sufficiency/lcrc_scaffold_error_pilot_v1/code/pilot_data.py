"""New Bemis--Murcko scaffold partition and data views for the pilot round.

The split is generated **once** from a fixed group random seed:

    split_seed = 20261102

Bemis--Murcko scaffolds (RDKit, ``includeChirality=False``) define indivisible
groups.  The group order is a ``numpy.random.default_rng`` permutation of the
lexicographically sorted scaffold strings, then groups are greedily assigned to
train / inner-validation / outer-evaluation with 80 / 10 / 10 cutoffs.  Row
order inside each split is ascending row id.  No label is used for assignment.

The split manifest stores every row id, every row's scaffold string, sizes and
SHA256 hashes.  Training views deliberately do **not** build the outer cache and
do **not** load outer labels; the outer partition is materialised only in the
one-shot evaluation step after the checkpoint manifest is frozen.
"""

from __future__ import annotations

import collections
import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import pilot_common as PC

import be_common as OLD  # noqa: E402
import be_data as D  # noqa: E402
import lcrc_data  # noqa: E402
import r42_data as R42D  # noqa: E402


# ---------------------------------------------------------------------------
# raw data
# ---------------------------------------------------------------------------


def load_csv_labels() -> tuple[list[str], np.ndarray]:
    if PC.sha256_file(PC.CSV_PATH) != PC.CSV_SHA256:
        raise PermissionError("Lipophilicity CSV hash drifted")
    with open(PC.CSV_PATH, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != PC.N_MOLECULES:
        raise ValueError("Lipophilicity row count drifted")
    smiles = [str(row["smiles"]) for row in rows]
    y = np.asarray([float(row["exp"]) for row in rows], dtype=np.float64)
    return smiles, y


def bemis_murcko_scaffolds(smiles: list[str]) -> list[str]:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    RDLogger.DisableLog("rdApp.*")
    out: list[str] = []
    for index, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"invalid SMILES at row {index}: {smi!r}")
        out.append(MurckoScaffoldSmiles(mol=mol, includeChirality=False))
    return out


# ---------------------------------------------------------------------------
# split generation / audit
# ---------------------------------------------------------------------------


def _split_stats(rows: Mapping[str, list[int]], y: np.ndarray, scaffolds: list[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for name in ("train", "valid", "test"):
        idx = np.asarray(rows[name], dtype=np.int64)
        values = y[idx]
        groups = {scaffolds[int(i)] for i in idx}
        stats[name] = {
            "n": int(idx.size),
            "n_scaffold_groups": int(len(groups)),
            "y_mean": float(values.mean()),
            "y_std": float(values.std(ddof=0)),
            "y_min": float(values.min()),
            "y_max": float(values.max()),
            "rows_sha256": PC.rows_sha256(idx.tolist()),
        }
    return stats


def generate_split() -> dict[str, Any]:
    smiles, y = load_csv_labels()
    scaffolds = bemis_murcko_scaffolds(smiles)
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for row, scaffold in enumerate(scaffolds):
        groups[str(scaffold)].append(int(row))

    keys = sorted(groups)  # deterministic pre-shuffle order
    rng = np.random.default_rng(PC.SPLIT_SEED)
    order = [int(x) for x in rng.permutation(len(keys)).tolist()]
    cut_train = PC.SPLIT_FRACTIONS[0] * PC.N_MOLECULES
    cut_valid = (PC.SPLIT_FRACTIONS[0] + PC.SPLIT_FRACTIONS[1]) * PC.N_MOLECULES

    rows: dict[str, list[int]] = {"train": [], "valid": [], "test": []}
    for key_index in order:
        group = groups[keys[key_index]]
        if len(rows["train"]) + len(group) <= cut_train:
            rows["train"].extend(group)
        elif len(rows["train"]) + len(rows["valid"]) + len(group) <= cut_valid:
            rows["valid"].extend(group)
        else:
            rows["test"].extend(group)
    rows = {name: sorted(values) for name, values in rows.items()}

    union = set(rows["train"]) | set(rows["valid"]) | set(rows["test"])
    if union != set(range(PC.N_MOLECULES)):
        raise RuntimeError("pilot split does not cover all 4,200 rows")
    if any(
        set(rows[a]).intersection(rows[b])
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))
    ):
        raise RuntimeError("pilot split has overlapping rows")

    split_of_scaffold: dict[str, str] = {}
    for name, values in rows.items():
        for row in values:
            scaffold = scaffolds[row]
            previous = split_of_scaffold.get(scaffold)
            if previous is not None and previous != name:
                raise RuntimeError(f"scaffold crosses splits: {scaffold!r} in {previous} and {name}")
            split_of_scaffold[scaffold] = name

    group_sizes = collections.Counter(len(values) for values in groups.values())
    payload = {
        "protocol_id": PC.PROTOCOL_ID,
        "created_utc": PC.utcnow(),
        "algorithm": (
            "Bemis-Murcko scaffold groups (RDKit MurckoScaffoldSmiles, includeChirality=False); "
            "numpy.random.default_rng(split_seed) permutation of lexicographically sorted "
            "scaffold strings; greedy fill with train<=0.8*N, train+valid<=0.9*N; rows sorted "
            "ascending inside each split; labels not used for assignment"
        ),
        "split_seed": PC.SPLIT_SEED,
        "split_fractions": list(PC.SPLIT_FRACTIONS),
        "n_molecules": PC.N_MOLECULES,
        "sizes": {name: len(values) for name, values in rows.items()},
        "rows": rows,
        "scaffold_by_row": scaffolds,
        "row_sha256": {name: PC.rows_sha256(values) for name, values in rows.items()},
        "scaffold_vector_sha256": PC.vector_sha256_float64(
            np.asarray([float(len(s)) for s in scaffolds], dtype=np.float64)
        ),
        "n_scaffold_groups": int(len(groups)),
        "n_singleton_scaffold_groups": int(group_sizes.get(1, 0)),
        "scaffold_group_size_histogram": {str(k): int(v) for k, v in sorted(group_sizes.items())},
        "scaffold_groups_cross_split": int(0),
        "descriptive_label_stats": _split_stats(rows, y, scaffolds),
        "label_stats_note": "descriptive only; labels were not used to generate or adjust the split",
        "versions": {
            "numpy": np.__version__,
            "rdkit": __import__("rdkit").__version__,
        },
    }
    payload["split_sha256"] = PC.sha256_json(
        {
            "split_seed": payload["split_seed"],
            "algorithm": payload["algorithm"],
            "rows": payload["rows"],
            "scaffold_by_row": payload["scaffold_by_row"],
        }
    )
    return payload


def write_split() -> dict[str, Any]:
    PC.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    payload = generate_split()
    if PC.SPLIT_MANIFEST.exists():
        previous = PC.read_json(PC.SPLIT_MANIFEST)
        if previous.get("split_sha256") != payload["split_sha256"]:
            raise PermissionError("existing pilot split manifest differs; refusing to overwrite a frozen split")
        return previous
    PC.write_json(PC.SPLIT_MANIFEST, payload)
    return payload


def audit_split() -> dict[str, Any]:
    payload = PC.read_json(PC.SPLIT_MANIFEST)
    recomputed_sha = PC.sha256_json(
        {
            "split_seed": payload["split_seed"],
            "algorithm": payload["algorithm"],
            "rows": payload["rows"],
            "scaffold_by_row": payload["scaffold_by_row"],
        }
    )
    smiles, y = load_csv_labels()
    scaffolds = bemis_murcko_scaffolds(smiles)
    rows = {name: [int(x) for x in payload["rows"][name]] for name in ("train", "valid", "test")}
    row_hashes = {name: PC.rows_sha256(values) for name, values in rows.items()}
    split_of_scaffold: dict[str, str] = {}
    cross_split: list[str] = []
    for name, values in rows.items():
        for row in values:
            scaffold = scaffolds[row]
            previous = split_of_scaffold.get(scaffold)
            if previous is not None and previous != name:
                cross_split.append(scaffold)
            split_of_scaffold[scaffold] = name
    union = set(rows["train"]) | set(rows["valid"]) | set(rows["test"])
    disjoint = all(
        set(rows[a]).isdisjoint(rows[b])
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))
    )
    checks = {
        "split_sha_matches": recomputed_sha == payload["split_sha256"],
        "rows_disjoint": bool(disjoint),
        "rows_complete": union == set(range(PC.N_MOLECULES)),
        "sizes_match_manifest": {name: len(rows[name]) for name in rows} == {
            name: int(payload["sizes"][name]) for name in rows
        },
        "row_hashes_match": row_hashes == {name: str(payload["row_sha256"][name]) for name in rows},
        "scaffold_by_row_matches_recomputed": scaffolds == [str(x) for x in payload["scaffold_by_row"]],
        "scaffold_groups_do_not_cross_splits": not cross_split,
        "all_rows_once": sum(len(values) for values in rows.values()) == PC.N_MOLECULES,
    }
    audit = {
        "protocol_id": PC.PROTOCOL_ID,
        "created_utc": PC.utcnow(),
        "split_manifest": PC.track_rel(PC.SPLIT_MANIFEST),
        "split_manifest_sha256": PC.sha256_file(PC.SPLIT_MANIFEST),
        "split_sha256": payload["split_sha256"],
        "split_seed": int(payload["split_seed"]),
        "sizes": {name: len(values) for name, values in rows.items()},
        "row_sha256": row_hashes,
        "n_scaffold_groups": int(len(split_of_scaffold)),
        "n_scaffold_groups_cross_split": int(len(cross_split)),
        "scaffold_groups_cross_split_examples": cross_split[:5],
        "descriptive_label_stats": _split_stats(rows, y, scaffolds),
        "checks": checks,
        "passed": bool(all(checks.values())),
        "note": "outer-evaluation labels are not read here; only the CSV labels used for descriptive stats",
    }
    PC.write_json(PC.SPLIT_AUDIT, audit)
    if not audit["passed"]:
        raise SystemExit(f"pilot split audit failed: {checks}")
    return audit


def load_split() -> dict[str, Any]:
    payload = PC.read_json(PC.SPLIT_MANIFEST)
    recomputed = PC.sha256_json(
        {
            "split_seed": payload["split_seed"],
            "algorithm": payload["algorithm"],
            "rows": payload["rows"],
            "scaffold_by_row": payload["scaffold_by_row"],
        }
    )
    if recomputed != payload["split_sha256"]:
        raise PermissionError("pilot split manifest content changed")
    return payload


# ---------------------------------------------------------------------------
# data views
# ---------------------------------------------------------------------------


def build_pilot_views(*, include_outer: bool) -> dict[str, Any]:
    """Build train/inner-validation views; outer cache only when requested.

    With ``include_outer=False`` the outer-evaluation cache is not built and
    outer labels are not loaded.  This is the only view used by training.
    """
    split_manifest = load_split()
    smiles, y = load_csv_labels()
    rows = {name: [int(x) for x in split_manifest["rows"][name]] for name in ("train", "valid", "test")}
    # The fixed split manifest already contains the scaffold strings; training
    # must not depend on RDKit (the GPU host does not install it).  The local
    # split audit re-derived them with RDKit before the freeze.
    if len(split_manifest["scaffold_by_row"]) != PC.N_MOLECULES:
        raise PermissionError("pilot split scaffold vector has the wrong length")

    base_views = D.load_views(include_test=False)
    cache_all = base_views["cache_built"]["cache"]
    if [str(x) for x in cache_all["molecule_ids"]] != [str(i) for i in range(PC.N_MOLECULES)]:
        raise PermissionError("frozen LCRC cache is not in CSV row order")
    spec = base_views["spec"]

    names = ["train", "valid"] + (["test"] if include_outer else [])
    caches: dict[str, Any] = {}
    for name in names:
        selected = R42D.select_cache(cache_all, rows[name], split_label=f"pilot_{name}")
        caches[name] = D.build_distance_buckets(selected)
        expected = [smiles[int(row)] for row in rows[name]]
        if [str(x) for x in caches[name]["smiles"]] != expected:
            raise ValueError(f"pilot {name} SMILES identity drifted")

    train_norm = lcrc_data.fit_norm(caches["train"], spec, source="pilot_train_structures")
    y_train = y[np.asarray(rows["train"], dtype=np.int64)]
    y_valid = y[np.asarray(rows["valid"], dtype=np.int64)]
    target_norm = {
        "mean": float(y_train.mean()),
        "scale": float(y_train.std(ddof=0)),
        "source": "pilot_train_labels_only",
        "fit_on": "pilot Train3360 only",
        "n": int(y_train.size),
        "valid_or_test_used": False,
    }
    labels = {
        "y_train": np.asarray(y_train, dtype=np.float64),
        "y_valid": np.asarray(y_valid, dtype=np.float64),
        "audit": {
            "source": "frozen Lipophilicity CSV + pilot split manifest",
            "y_train_sha256": PC.vector_sha256_float64(y_train),
            "y_valid_sha256": PC.vector_sha256_float64(y_valid),
        },
    }
    if include_outer:
        y_test = y[np.asarray(rows["test"], dtype=np.int64)]
        labels["audit"]["y_test_sha256"] = PC.vector_sha256_float64(y_test)

    data_override = {
        "split_file": PC.track_rel(PC.SPLIT_MANIFEST),
        "split_file_sha256": PC.sha256_file(PC.SPLIT_MANIFEST),
        "labels_file": PC.track_rel(PC.CSV_PATH),
        "labels_file_sha256": PC.CSV_SHA256,
        "train_rows_sha256": PC.rows_sha256(rows["train"]),
        "valid_rows_sha256": PC.rows_sha256(rows["valid"]),
        "test_rows_sha256": PC.rows_sha256(rows["test"]),
        "split_scope": "pilot_bemis_murcko_scaffold_seed_20261102",
        "train_norm_fingerprint": lcrc_data.norm_fingerprint(train_norm),
        "target_norm": dict(target_norm),
        "test_row_values_used": False,
        "label_scope": (
            "pilot_train_and_inner_valid_only_during_training"
            if not include_outer
            else "outer_labels_loaded_after_manifest_freeze"
        ),
    }
    views = {
        "split": {"rows": rows, "smiles": smiles, "name": "lipo_pilot_scaffold_seed_20261102"},
        "labels": labels,
        "spec": spec,
        "cache_built": {k: v for k, v in base_views["cache_built"].items() if k != "cache"},
        "train_cache": caches["train"],
        "valid_cache": caches["valid"],
        "train_norm": train_norm,
        "train_norm_fingerprint": lcrc_data.norm_fingerprint(train_norm),
        "target_norm": target_norm,
        "split_seed": PC.SPLIT_SEED,
        "split_scope": "pilot_bemis_murcko_scaffold_seed_20261102",
        "data_override": data_override,
        "protocol_override": {
            "pilot_protocol_id": PC.PROTOCOL_ID,
            "pilot_round_id": PC.ROUND_ID,
            "pilot_split_seed": PC.SPLIT_SEED,
            "pilot_split_manifest_sha256": PC.sha256_file(PC.SPLIT_MANIFEST),
        },
    }
    if include_outer:
        views["test_cache"] = caches["test"]
        views["y_test"] = y[np.asarray(rows["test"], dtype=np.int64)]
    return views


__all__ = [
    "load_csv_labels",
    "bemis_murcko_scaffolds",
    "generate_split",
    "write_split",
    "audit_split",
    "load_split",
    "build_pilot_views",
]
