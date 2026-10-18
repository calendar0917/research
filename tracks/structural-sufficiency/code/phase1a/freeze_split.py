"""Freeze the Phase 1a internal Lipophilicity split (structure-only).

This script is deliberately label-blind.  It reads the Lipophilicity SMILES
column, the official DeepChem-2.8.0 train rows (row indices only), and the
label-free row manifest.  It never opens the score-label vault
(``data/lipophilicity/labels.npy``) and never reads ``train_y`` from the old
split JSON.

Outputs
-------
``tracks/structural-sufficiency/results/phase1a/split_phase1a.json``
    Fit / Monitor / Reserved row indices (2352 / 504 / 504), the frozen
    n_small subset, scaffold-group hashes and row-index hashes.

The script reproduces the Phase 0 candidate algorithm exactly and verifies the
row hashes against ``results/phase0/analysis/lipo_feasibility_audit.json`` when
that local evidence file exists.

Run:
    uv run python tracks/structural-sufficiency/code/phase1a/freeze_split.py
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LIPO = REPO / "data/lipophilicity/Lipophilicity.csv"
BJOC = REPO / "tracks/structure-contribution/results/bjoc_cross_task_lipophilicity_v1"
SPLIT = BJOC / "split_lipo_v1.json"
ROW_MANIFEST = BJOC / "row_manifest.json"
RAW_GRAPH = BJOC / "dataset/lipo_raw_v1.json.gz"
PHASE0_AUDIT = REPO / "tracks/structural-sufficiency/results/phase0/analysis/lipo_feasibility_audit.json"
OUT = REPO / "tracks/structural-sufficiency/results/phase1a/split_phase1a.json"
PROTOCOL_ID = "ssrc-phase1a-lipo-v1"
SMALL_TARGET = 250
SMALL_MAX_GROUP = 10
SMALL_SEED = 20261018


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_hash(rows: Sequence[int]) -> str:
    """Phase 0 row-index hash: comma-separated decimal string."""
    return sha256_text(",".join(str(int(r)) for r in rows))


def row_bytes_hash(rows: Sequence[int]) -> str:
    """Legacy BJOC split hash: little-endian int64 bytes, original order."""
    return sha256_bytes(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes())


def group_hash(scaffolds: Sequence[str]) -> str:
    return sha256_text("\n".join(repr(s) for s in scaffolds))


def canonical_and_scaffold(smiles: str) -> tuple[str, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot parse {smiles!r}")
    can = Chem.MolToSmiles(mol)
    scaf = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return can, str(scaf)


def load_official_train_rows() -> tuple[list[int], dict[str, Any]]:
    """Read only row-index fields; drop label payloads at parse time."""

    def drop_labels(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        drop = {"train_y", "train_label_quantiles"}
        return {k: v for k, v in pairs if k not in drop}

    split = json.loads(SPLIT.read_text(), object_pairs_hook=drop_labels)
    train = [int(x) for x in split["train_rows"]]
    if len(train) != 3360 or len(set(train)) != 3360:
        raise ValueError("official train rows unexpected")
    if row_bytes_hash(train) != split["train_sha256_rows"]:
        raise ValueError("official train row hash mismatch")
    meta = {
        "split_json": str(SPLIT.relative_to(REPO)),
        "split_sha256": sha256_file(SPLIT),
        "train_sha256_rows": split["train_sha256_rows"],
        "reference_match": split.get("reference_match"),
        "label_keys_loaded": False,
    }
    return train, meta


def phase0_candidate(train_rows: Sequence[int], scaffolds: Sequence[str]) -> dict[str, list[int]]:
    """Exact Phase 0 candidate algorithm (structure-only, 70/15/15)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for row in train_rows:
        groups[str(scaffolds[row])].append(int(row))
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    targets = {
        "fit": int(round(0.70 * len(train_rows))),
        "monitor": int(round(0.15 * len(train_rows))),
        "reserved": int(round(0.15 * len(train_rows))),
    }
    assigned: dict[str, list[int]] = {"fit": [], "monitor": [], "reserved": []}
    keys = ("fit", "monitor", "reserved")
    for _scaf, rows in ordered:
        capacities = {k: targets[k] - len(assigned[k]) for k in keys}
        dest = max(keys, key=lambda k: (capacities[k], -keys.index(k)))
        assigned[dest].extend(rows)
    for k in assigned:
        assigned[k].sort()
    if sorted(assigned["fit"] + assigned["monitor"] + assigned["reserved"]) != sorted(train_rows):
        raise ValueError("candidate split does not partition official train rows")
    return assigned


def choose_small(fit_rows: Sequence[int], scaffolds: Sequence[str]) -> tuple[list[int], dict[str, Any]]:
    """Label-free scaffold-grouped subset of Fit, target ~250 molecules.

    Pre-registered rule: only scaffold groups with size <= SMALL_MAX_GROUP are
    eligible (so one large scaffold cannot dominate the small-n training set);
    eligible groups are shuffled with a fixed seed keyed by the scaffold string,
    then whole groups are added until the target is reached.  Actual n is
    frozen and shared by all models.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for row in fit_rows:
        groups[str(scaffolds[row])].append(int(row))
    eligible = [(scaf, sorted(rows)) for scaf, rows in groups.items() if len(rows) <= SMALL_MAX_GROUP]
    eligible.sort(key=lambda kv: kv[0])
    rng = random.Random(SMALL_SEED)
    rng.shuffle(eligible)
    selected: list[int] = []
    chosen_groups: list[str] = []
    for scaf, rows in eligible:
        if len(selected) >= SMALL_TARGET:
            break
        selected.extend(rows)
        chosen_groups.append(scaf)
    if len(selected) < SMALL_TARGET:
        # Fallback: add the remaining groups in the same shuffled order.
        for scaf, rows in sorted(groups.items(), key=lambda kv: kv[0]):
            if scaf in chosen_groups:
                continue
            selected.extend(rows)
            chosen_groups.append(scaf)
            if len(selected) >= SMALL_TARGET:
                break
    selected = sorted(selected)
    meta = {
        "target": SMALL_TARGET,
        "actual_n": len(selected),
        "rule": (
            "eligible scaffold groups size <= 10; sort by scaffold string; "
            "shuffle with random.Random(20261018); add whole groups until target"
        ),
        "n_groups": len(chosen_groups),
        "seed": SMALL_SEED,
        "max_group": max(len(groups[g]) for g in chosen_groups),
        "min_group": min(len(groups[g]) for g in chosen_groups),
        "group_size_histogram": dict(sorted(Counter(len(groups[g]) for g in chosen_groups).items())),
    }
    return selected, meta


def main() -> int:
    df = pd.read_csv(LIPO, usecols=["CMPD_CHEMBLID", "smiles"])
    if len(df) != 4200:
        raise ValueError("Lipophilicity row count changed")
    canonical: list[str] = []
    scaffolds: list[str] = []
    for smi in df["smiles"].tolist():
        can, scaf = canonical_and_scaffold(smi)
        canonical.append(can)
        scaffolds.append(scaf)
    if len(set(canonical)) != len(canonical):
        raise ValueError("canonical SMILES duplicates")

    train_rows, split_meta = load_official_train_rows()
    assigned = phase0_candidate(train_rows, scaffolds)
    small_rows, small_meta = choose_small(assigned["fit"], scaffolds)

    group_sets = {k: {scaffolds[r] for r in rows} for k, rows in assigned.items()}
    overlap = {
        "fit_monitor": len(group_sets["fit"] & group_sets["monitor"]),
        "fit_reserved": len(group_sets["fit"] & group_sets["reserved"]),
        "monitor_reserved": len(group_sets["monitor"] & group_sets["reserved"]),
    }
    if any(overlap.values()):
        raise ValueError(f"scaffold overlap: {overlap}")
    small_groups = {scaffolds[r] for r in small_rows}
    if not small_groups <= group_sets["fit"]:
        raise ValueError("small subset is not contained in Fit groups")

    # Compare with the local Phase 0 evidence, when present.
    phase0_match: dict[str, Any]
    if PHASE0_AUDIT.exists():
        audit = json.loads(PHASE0_AUDIT.read_text())
        cand = audit["candidate_internal_split"]
        expected = {
            "fit": cand["rows"]["fit"],
            "monitor": cand["rows"]["valid"],
            "reserved": cand["rows"]["locked"],
        }
        phase0_match = {
            "audit_present": True,
            "same_rows": {k: list(map(int, expected[k])) == assigned[k] for k in assigned},
            "row_hash_match": {
                k: row_hash(assigned[k]) == cand["row_index_sha256"][
                    {"fit": "fit", "monitor": "valid", "reserved": "locked"}[k]
                ]
                for k in assigned
            },
        }
        if not all(phase0_match["same_rows"].values()):
            raise ValueError(f"recomputed split differs from Phase 0 candidate: {phase0_match}")
    else:
        phase0_match = {"audit_present": False}

    payload = {
        "protocol_id": PROTOCOL_ID,
        "status": "FROZEN_STRUCTURE_ONLY",
        "date_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "source": {
            "csv": {"path": str(LIPO.relative_to(REPO)), "sha256": sha256_file(LIPO)},
            "raw_graph": {"path": str(RAW_GRAPH.relative_to(REPO)), "sha256": sha256_file(RAW_GRAPH)},
            "row_manifest": {"path": str(ROW_MANIFEST.relative_to(REPO)), "sha256": sha256_file(ROW_MANIFEST)},
            **split_meta,
        },
        "scope": {
            "labels_read": False,
            "official_valid_test_labels_read": False,
            "internal_reserved_labels_read": False,
            "structure_only": True,
        },
        "n_total": 4200,
        "n_official_train": len(train_rows),
        "rows": assigned,
        "row_index_sha256": {k: row_hash(v) for k, v in assigned.items()},
        "scaffold_group_sha256": {k: group_hash(sorted(group_sets[k])) for k in assigned},
        "n_scaffold_groups": {k: len(group_sets[k]) for k in assigned},
        "scaffold_group_overlap": overlap,
        "small": {
            "rows": small_rows,
            "row_index_sha256": row_hash(small_rows),
            **small_meta,
        },
        "phase0_candidate_check": phase0_match,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    OUT.write_text(text)

    print("SSRC Phase 1a split freeze")
    print(f"  output: {OUT.relative_to(REPO)}")
    print(f"  fit/monitor/reserved: {[len(assigned[k]) for k in ('fit','monitor','reserved')]}")
    print(f"  groups: {payload['n_scaffold_groups']} overlap={overlap}")
    print(f"  small n={small_meta['actual_n']} groups={small_meta['n_groups']} hash={row_hash(small_rows)}")
    print(f"  phase0 check: {phase0_match}")
    print(f"  sha256: {sha256_text(text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
