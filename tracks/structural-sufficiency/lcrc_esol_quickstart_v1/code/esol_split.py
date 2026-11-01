"""One fixed, label-blind Bemis--Murcko scaffold partition for ESOL.

Algorithm (identical to the frozen pilot splitter, with this round's seed):

1. Bemis--Murcko scaffold per molecule (RDKit, ``includeChirality=False``),
   taken from the committed ``esol_scaffolds_v1.json`` artifact;
2. molecules are grouped by *group key* (see the empty-scaffold policy below);
3. group keys are sorted lexicographically, then permuted by
   ``numpy.random.default_rng(split_seed)``;
4. groups are greedily appended to Train while ``<= 0.8*N``, else to
   Validation while ``<= 0.9*N``, else to Test;
5. row ids inside a split are sorted ascending.

No label is read by this module: the split manifest contains row ids, scaffold
strings and group keys only.  No per-split label statistics are computed before
the Test evaluation is frozen (that is a deliberate scope restriction).

Empty-scaffold policy (decided before any training, recorded here)
-----------------------------------------------------------------
Acyclic molecules have the empty Bemis--Murcko scaffold ``""``.  In ESOL, 317 of
1,128 molecules (28.1%) are acyclic, so treating ``""`` as one indivisible group
is *not* a harmless detail: with ``split_seed = 20261103`` it puts the whole
block into Validation and leaves **Test with zero molecules** (that strict
variant is recorded in the manifest as evidence).  A per-row singleton key is
also wrong: ESOL contains 5 duplicated SMILES, and one of those pairs
(sorbitol) would then straddle Train and Test under two different labels.

The pre-registered policy is therefore:

    an acyclic molecule's group key is ``"__ACYCLIC__" + smiles``: no shared
    ring system to protect, but identical SMILES always stay in one split.

Non-empty scaffolds are always kept indivisible.  Both decisions are purely
structural and label-blind, taken before training and never revisited.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

import esol_common as C
import esol_raw as R


def _group_keys(scaffolds: Sequence[str], smiles: Sequence[str], *, acyclic_singleton: bool) -> list[str]:
    if acyclic_singleton:
        return [
            str(scaffold) if scaffold != "" else f"__ACYCLIC__{str(smi)}"
            for scaffold, smi in zip(scaffolds, smiles)
        ]
    return [str(s) for s in scaffolds]


def _assign(group_keys: Sequence[str], n_molecules: int, seed: int) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for row, key in enumerate(group_keys):
        groups.setdefault(str(key), []).append(int(row))
    keys = sorted(groups)
    rng = np.random.default_rng(int(seed))
    order = [int(x) for x in rng.permutation(len(keys)).tolist()]
    cut_train = C.SPLIT_FRACTIONS[0] * n_molecules
    cut_valid = (C.SPLIT_FRACTIONS[0] + C.SPLIT_FRACTIONS[1]) * n_molecules
    rows: dict[str, list[int]] = {"train": [], "valid": [], "test": []}
    for key_index in order:
        group = groups[keys[key_index]]
        if len(rows["train"]) + len(group) <= cut_train:
            rows["train"].extend(group)
        elif len(rows["train"]) + len(rows["valid"]) + len(group) <= cut_valid:
            rows["valid"].extend(group)
        else:
            rows["test"].extend(group)
    return {name: sorted(values) for name, values in rows.items()}


def _sizes(rows: Mapping[str, Sequence[int]]) -> dict[str, int]:
    return {name: len(rows[name]) for name in ("train", "valid", "test")}


def _strict_sizes(scaffolds: Sequence[str], smiles: Sequence[str]) -> dict[str, Any]:
    rows = _assign(_group_keys(scaffolds, smiles, acyclic_singleton=False), len(scaffolds), C.SPLIT_SEED)
    return {
        "policy": 'strict: every molecule with the empty Bemis--Murcko scaffold shares group ""',
        "sizes": _sizes(rows),
        "rejected_because": (
            "the empty-scaffold block (317 rows) is placed into Validation and Test receives "
            "0 molecules, so no one-shot evaluation would be possible"
            if _sizes(rows)["test"] == 0
            else "kept as a record of the strict variant"
        ),
    }


def _duplicate_report(smiles: Sequence[str], rows: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    groups: dict[str, list[int]] = {}
    for row, smi in enumerate(smiles):
        groups.setdefault(str(smi), []).append(int(row))
    split_of_row = {int(row): name for name, values in rows.items() for row in values}
    duplicates = {smi: members for smi, members in groups.items() if len(members) > 1}
    crossing = {
        smi: {"rows": members, "splits": [split_of_row[row] for row in members]}
        for smi, members in duplicates.items()
        if len({split_of_row[row] for row in members}) > 1
    }
    return {
        "n_distinct_smiles": len(groups),
        "n_duplicate_groups": len(duplicates),
        "n_extra_rows": int(sum(len(m) - 1 for m in duplicates.values())),
        "groups": {smi: [int(r) for r in members] for smi, members in duplicates.items()},
        "groups_crossing_splits": crossing,
        "n_groups_crossing_splits": len(crossing),
    }


def generate_split() -> dict[str, Any]:
    csv_payload = R.read_csv()
    scaffolds = R.load_scaffolds()["scaffold_by_row"]
    if len(scaffolds) != len(csv_payload["smiles"]):
        raise ValueError("scaffold vector length differs from the CSV")
    group_keys = _group_keys(scaffolds, csv_payload["smiles"], acyclic_singleton=True)
    rows = _assign(group_keys, len(scaffolds), C.SPLIT_SEED)

    union = set(rows["train"]) | set(rows["valid"]) | set(rows["test"])
    if union != set(range(len(scaffolds))):
        raise RuntimeError("split does not cover every row exactly once")
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if set(rows[a]) & set(rows[b]):
            raise RuntimeError(f"rows shared by {a} and {b}")

    split_of_group: dict[str, str] = {}
    for name, values in rows.items():
        for row in values:
            key = group_keys[int(row)]
            previous = split_of_group.get(key)
            if previous is not None and previous != name:
                raise RuntimeError(f"group {key!r} crosses {previous}/{name}")
            split_of_group[key] = name

    group_sizes: dict[str, int] = {}
    for key in group_keys:
        group_sizes[key] = group_sizes.get(key, 0) + 1
    histogram: dict[str, int] = {}
    for size in group_sizes.values():
        histogram[str(size)] = histogram.get(str(size), 0) + 1

    n_acyclic = int(sum(1 for s in scaffolds if s == ""))
    acyclic_per_split = {
        name: int(sum(1 for row in rows[name] if scaffolds[int(row)] == "")) for name in rows
    }
    benzene_per_split = {
        name: int(sum(1 for row in rows[name] if scaffolds[int(row)] == "c1ccccc1")) for name in rows
    }
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "algorithm": (
            "Bemis-Murcko scaffolds (RDKit MurckoScaffoldSmiles, includeChirality=False) as "
            "indivisible groups; acyclic rows (empty scaffold) form per-molecule singleton groups; "
            "sorted group keys permuted by numpy.random.default_rng(split_seed); greedy fill with "
            "train<=0.8*N, train+valid<=0.9*N; row ids sorted ascending inside each split"
        ),
        "grouping_policy": {
            "non_empty_scaffold": "indivisible group",
            "empty_scaffold": 'group key "__ACYCLIC__" + SMILES (one group per distinct acyclic molecule)',
            "rationale": (
                "317/1128 ESOL molecules are acyclic, so a single empty-scaffold group would "
                "swallow an entire split (the strict variant leaves Test empty and is recorded "
                "below); a per-row singleton key instead lets the 5 duplicated ESOL SMILES "
                "straddle splits, so identical acyclic SMILES are kept together"
            ),
        },
        "strict_empty_scaffold_alternative": _strict_sizes(scaffolds, csv_payload["smiles"]),
        "duplicate_smiles": _duplicate_report(csv_payload["smiles"], rows),
        "split_seed": int(C.SPLIT_SEED),
        "split_fractions": list(C.SPLIT_FRACTIONS),
        "n_molecules": int(len(scaffolds)),
        "sizes": _sizes(rows),
        "rows": {name: [int(x) for x in rows[name]] for name in ("train", "valid", "test")},
        "row_sha256": {name: C.rows_sha256(rows[name]) for name in ("train", "valid", "test")},
        "scaffold_by_row": [str(s) for s in scaffolds],
        "group_key_by_row": [str(k) for k in group_keys],
        "n_groups": int(len(group_sizes)),
        "n_singleton_groups": int(sum(1 for size in group_sizes.values() if size == 1)),
        "group_size_histogram": dict(sorted(histogram.items(), key=lambda kv: int(kv[0]))),
        "n_acyclic_rows": n_acyclic,
        "n_empty_scaffold_rows": n_acyclic,
        "acyclic_rows_per_split": acyclic_per_split,
        "benzene_scaffold_rows_per_split": benzene_per_split,
        "largest_groups": sorted(
            ({"scaffold": key, "n": int(size)} for key, size in group_sizes.items()),
            key=lambda rec: -rec["n"],
        )[:5],
        "scaffold_groups_cross_split": 0,
        "labels_read": False,
        "label_statistics_computed": False,
        "source": {
            "csv_sha256": C.CSV_SHA256,
            "scaffold_artifact": C.track_rel(C.SCAFFOLD_FILE),
            "scaffold_artifact_sha256": C.sha256_file(C.SCAFFOLD_FILE),
        },
        "versions": dict(
            (k, v)
            for k, v in C.env_info().items()
            if k in ("python", "numpy", "rdkit", "platform", "git_commit")
        ),
    }
    payload["split_sha256"] = C.sha256_json(
        {
            "split_seed": payload["split_seed"],
            "algorithm": payload["algorithm"],
            "rows": payload["rows"],
            "group_key_by_row": payload["group_key_by_row"],
        }
    )
    return payload


def write_split() -> dict[str, Any]:
    C.SPLIT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    payload = generate_split()
    if C.SPLIT_MANIFEST.exists():
        previous = C.read_json(C.SPLIT_MANIFEST)
        if previous.get("split_sha256") != payload["split_sha256"]:
            raise PermissionError("existing split manifest differs; refusing to overwrite a frozen split")
        return previous
    C.write_json(C.SPLIT_MANIFEST, payload)
    return payload


def load_split() -> dict[str, Any]:
    payload = C.read_json(C.SPLIT_MANIFEST)
    recomputed = C.sha256_json(
        {
            "split_seed": payload["split_seed"],
            "algorithm": payload["algorithm"],
            "rows": payload["rows"],
            "group_key_by_row": payload["group_key_by_row"],
        }
    )
    if recomputed != payload["split_sha256"]:
        raise PermissionError("split manifest content changed after the freeze")
    if payload.get("labels_read") is not False:
        raise PermissionError("split manifest is not certified label-free")
    return payload


def audit_split() -> dict[str, Any]:
    payload = load_split()
    csv_payload = R.read_csv()
    scaffolds = R.bemis_murcko_scaffolds(csv_payload["smiles"])
    rows = {name: [int(x) for x in payload["rows"][name]] for name in ("train", "valid", "test")}
    group_keys = _group_keys(scaffolds, csv_payload["smiles"], acyclic_singleton=True)
    regenerated = _assign(group_keys, len(scaffolds), C.SPLIT_SEED)
    split_of_group: dict[str, str] = {}
    cross: list[str] = []
    for name, values in rows.items():
        for row in values:
            key = group_keys[row]
            previous = split_of_group.get(key)
            if previous is not None and previous != name:
                cross.append(key)
            split_of_group[key] = name
    union = set(rows["train"]) | set(rows["valid"]) | set(rows["test"])
    checks = {
        "split_sha_matches": C.sha256_json(
            {
                "split_seed": payload["split_seed"],
                "algorithm": payload["algorithm"],
                "rows": payload["rows"],
                "group_key_by_row": payload["group_key_by_row"],
            }
        )
        == payload["split_sha256"],
        "rows_disjoint": all(
            set(rows[a]).isdisjoint(rows[b]) for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))
        ),
        "rows_complete": union == set(range(len(scaffolds))),
        "regeneration_matches_manifest": regenerated == rows,
        "row_hashes_match": {name: C.rows_sha256(rows[name]) for name in rows}
        == {name: str(payload["row_sha256"][name]) for name in rows},
        "sizes_match_manifest": _sizes(rows) == {name: int(payload["sizes"][name]) for name in rows},
        "scaffold_by_row_matches_recomputed": [str(s) for s in payload["scaffold_by_row"]] == scaffolds,
        "group_keys_match_recomputed": [str(k) for k in payload["group_key_by_row"]] == group_keys,
        "groups_do_not_cross_splits": not cross,
        "no_label_field_in_manifest": not any(
            key in payload for key in ("y", "labels", "target", "descriptive_label_stats")
        ),
        "label_statistics_absent": payload.get("label_statistics_computed") is False,
    }
    audit = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "split_manifest": C.track_rel(C.SPLIT_MANIFEST),
        "split_manifest_sha256": C.sha256_file(C.SPLIT_MANIFEST),
        "split_sha256": payload["split_sha256"],
        "split_seed": int(payload["split_seed"]),
        "sizes": _sizes(rows),
        "row_sha256": {name: C.rows_sha256(rows[name]) for name in rows},
        "n_groups": int(len(split_of_group)),
        "n_groups_cross_split": int(len(cross)),
        "cross_split_examples": cross[:5],
        "n_acyclic_rows": int(sum(1 for s in scaffolds if s == "")),
        "acyclic_rows_per_split": {
            name: int(sum(1 for row in rows[name] if scaffolds[row] == "")) for name in rows
        },
        "strict_empty_scaffold_alternative": payload["strict_empty_scaffold_alternative"],
        "duplicate_smiles": payload["duplicate_smiles"],
        "duplicate_smiles_cross_splits": payload["duplicate_smiles"]["n_groups_crossing_splits"],
        "checks": checks,
        "passed": bool(all(checks.values())),
        "note": (
            "the split is label-blind; no per-split label statistic is computed here or before the "
            "one-shot Test evaluation is frozen"
        ),
        "environment": C.env_info(),
    }
    C.write_json(C.SPLIT_AUDIT, audit)
    if not audit["passed"]:
        raise SystemExit(f"ESOL split audit failed: {checks}")
    return audit


__all__ = ["generate_split", "write_split", "load_split", "audit_split"]
