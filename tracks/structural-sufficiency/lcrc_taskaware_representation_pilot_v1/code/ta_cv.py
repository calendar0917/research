"""P2 -- BBBP Train-only, scaffold-grouped 3-fold CV split.

Frozen design (see ``PROTOCOL.md``):

* rows: only the **Train** rows of the published ``BBBP.scaffold122`` split
  (1631 rows).  The published Validation and Test labels are not read, not
  reshaped and not used to guide anything here.
* groups: Bemis-Murcko scaffold, computed with RDKit directly from the locked
  SMILES, ``includeChirality=False`` (the same convention as the standard
  MoleculeNet scaffold splitter).
* acyclic molecules have the empty scaffold; grouping all of them into one
  bucket would put ~100 rows in a single fold, so every acyclic row is its own
  group.  This is fixed *before* looking at any label distribution and recorded.
* assignment: deterministic greedy **stratified** group assignment --
  groups sorted by (size desc, sha256(salt|scaffold)), each group placed in the
  fold that minimises a normalised positive/negative imbalance.  No RNG state is
  involved, so the assignment is reproducible bit-for-bit from the salt alone.
* every constraint is verified: no scaffold group crosses a fold boundary, the
  three folds partition the Train rows exactly, and each fold has both classes.

Usage
-----
    uv run python code/ta_cv.py --build        # build + verify, print the audit
"""

from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence

import numpy as np

import ta_common as C  # noqa: I001

import mb2_source as S

OUT = C.SPLIT_FREEZE
BBBP_SPLIT_FILE = C.V2 / "analysis/source/chembench/BBBP/BBBP.scaffold122.ind.pkl"


def _read_split_pkl(path) -> list[list[int]]:
    import pandas as pd

    series = pd.read_pickle(path)
    return [[int(x) for x in part] for part in series.to_list()]


def bbbp_published_rows() -> dict[str, Any]:
    """The published BBBP split (repeat 122) and its verified hashes."""
    import pandas as pd

    manifest = C.read_json(C.V2 / "analysis/split_manifest_BBBP.json")
    sha = C.sha256_file(BBBP_SPLIT_FILE)
    if sha != manifest["selected_split_sha256"]:
        raise ValueError(f"BBBP split file sha {sha} != Sprint-02 recorded {manifest['selected_split_sha256']}")
    train, valid, test = _read_split_pkl(BBBP_SPLIT_FILE)
    record = {
        "file": BBBP_SPLIT_FILE.name,
        "sha256": sha,
        "rows": {"train": train, "valid": valid, "test": test},
        "rows_sha256": {k: C.row_sha(v) for k, v in (("train", train), ("valid", valid), ("test", test))},
        "sizes": {k: len(v) for k, v in (("train", train), ("valid", valid), ("test", test))},
        "manifest_rows_sha256": manifest["rows_sha256"],
    }
    for name in ("train", "valid", "test"):
        if record["rows_sha256"][name] != manifest["rows_sha256"][name]:
            raise ValueError(f"BBBP/{name}: row hash differs from Sprint 02")
    return record


def murcko_scaffolds(smiles: Sequence[str], rows: Sequence[int]) -> dict[int, str]:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    out: dict[int, str] = {}
    for r in rows:
        mol = Chem.MolFromSmiles(str(smiles[int(r)]))
        if mol is None:  # documented relaxed fallback, mirrors the frozen adapter
            mol = Chem.MolFromSmiles(str(smiles[int(r)]), sanitize=False)
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)
        out[int(r)] = str(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))
    return out


def build_folds(
    rows: Sequence[int],
    labels: Sequence[float],
    scaffolds: Mapping[int, str],
    *,
    folds: int = C.BBBP_CV_FOLDS,
    salt: str = C.BBBP_CV_SALT,
) -> dict[str, Any]:
    """Deterministic greedy stratified group assignment."""
    import hashlib

    label = {int(r): int(round(float(y))) for r, y in zip(rows, labels)}
    groups: dict[str, list[int]] = {}
    acyclic = 0
    for r in rows:
        scaf = scaffolds[int(r)]
        key = scaf if scaf else f"__acyclic__/{int(r)}"
        if not scaf:
            acyclic += 1
        groups.setdefault(key, []).append(int(r))

    def order_key(item: tuple[str, list[int]]) -> tuple:
        key, members = item
        digest = hashlib.sha256(f"{salt}|{key}".encode("utf-8")).hexdigest()
        return (-len(members), digest)

    ordered = sorted(groups.items(), key=order_key)
    total_pos = sum(label[r] for r in rows)
    total_neg = len(rows) - total_pos
    target_pos = total_pos / folds
    target_neg = total_neg / folds
    current = [{"pos": 0, "neg": 0, "rows": []} for _ in range(folds)]
    assignment: dict[str, int] = {}
    for key, members in ordered:
        p = sum(label[r] for r in members)
        n = len(members) - p
        # least-loaded bottleneck scheduling: place the group where it is least
        # overloaded after insertion.  A convex-l2 cost would instead keep
        # filling the already-fullest fold and can leave a fold empty.
        scores = [
            max(
                (current[f]["pos"] + p) / max(target_pos, 1.0),
                (current[f]["neg"] + n) / max(target_neg, 1.0),
            )
            for f in range(folds)
        ]
        fold = int(np.argmin(np.asarray(scores, dtype=np.float64)))
        assignment[key] = fold
        current[fold]["pos"] += p
        current[fold]["neg"] += n
        current[fold]["rows"].extend(members)

    fold_rows = {}
    for f in range(folds):
        members = sorted(current[f]["rows"])
        fold_rows[f] = {
            "rows": members,
            "n": len(members),
            "rows_sha256": C.row_sha(members),
            "n_positive": int(sum(label[r] for r in members)),
            "n_negative": int(len(members) - sum(label[r] for r in members)),
            "positive_rate": float(sum(label[r] for r in members) / max(len(members), 1)),
            "n_scaffold_groups": int(sum(1 for k in assignment if assignment[k] == f)),
        }
    # ---- verification ---------------------------------------------------
    union: list[int] = []
    for f in range(folds):
        union.extend(fold_rows[f]["rows"])
    if sorted(union) != sorted(int(r) for r in rows):
        raise ValueError("folds do not partition the Train rows")
    if len(set(union)) != len(union):
        raise ValueError("a row appears in more than one fold")
    for key, fold in assignment.items():
        members = groups[key]
        present = {f for f in range(folds) if any(r in set(fold_rows[f]["rows"]) for r in members)}
        if present != {fold}:
            raise ValueError(f"scaffold group {key!r} crosses folds: {present}")
    for f in range(folds):
        if fold_rows[f]["n_positive"] == 0 or fold_rows[f]["n_negative"] == 0:
            raise ValueError(f"fold {f} has a single class")
    return {
        "folds": fold_rows,
        "assignment": assignment,
        "group_sizes": {k: len(v) for k, v in groups.items()},
        "n_groups": len(groups),
        "n_acyclic_rows": int(acyclic),
        "targets": {"positive_per_fold": float(target_pos), "negative_per_fold": float(target_neg)},
        "algorithm": (
            "greedy stratified group assignment (least-loaded bottleneck scheduling): groups ordered by "
            "(-size, sha256(salt|scaffold)); each group goes to the fold minimising "
            "max((pos_f+p)/target_pos, (neg_f+n)/target_neg), ties -> lowest fold index; no RNG"
        ),
        "salt": salt,
        "acyclic_policy": "each acyclic (empty Murcko scaffold) row is its own group",
        "scaffold_definition": "RDKit MurckoScaffold.MurckoScaffoldSmiles(includeChirality=False)",
    }


def frozen_folds() -> dict[str, Any]:
    """Load the frozen fold definitions and re-verify every hash."""
    if not OUT.exists():
        raise FileNotFoundError(f"{C.track_rel(OUT)} is missing; run `ta_cv.py --build` first")
    payload = C.read_json(OUT)
    entry = payload["bbbp_inner_cv"]
    for f, item in entry["folds"].items():
        if C.row_sha(item["rows"]) != item["rows_sha256"]:
            raise ValueError(f"BBBP fold {f}: row hash drift against the frozen split")
    return entry


def build() -> dict[str, Any]:
    """Build (or rebuild) ``analysis/split_freeze.json`` for P1 + P2."""
    published = bbbp_published_rows()
    src = S.load_dataset("BBBP", labels="train_valid")
    train_rows = list(published["rows"]["train"])
    if [int(x) for x in src["rows"]["train"]] != train_rows:
        raise ValueError("BBBP Train rows differ between the split file and the source loader")
    y = np.asarray(src["y"]["train"], dtype=np.float64).reshape(-1)
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("BBBP Train labels must be binary")
    scaffolds = murcko_scaffolds(src["smiles"], train_rows)
    folds = build_folds(train_rows, y, scaffolds)

    lipo = {}
    for suffix in (122, 123, 124):
        s = __import__("ta_data").lipo_repeat_split(suffix)
        lipo[str(suffix)] = {
            "file": s["file"],
            "sha256": s["sha256"],
            "sizes": s["sizes"],
            "rows_sha256": s["rows_sha256"],
            "rows": s["rows"],
            "role": "baseline (Sprint 01/02)" if suffix == 122 else "new robustness repeat (P1)",
            "train_positive_rate": None,
        }
        meta = C.dataset_entry("Lipophilicity")
        import pandas as pd

        df = pd.read_csv(S.chembench_path(f"{meta['chembench_task']}/{meta['chembench_task']}.csv.gz"))
        yy = df[list(meta["label_columns"])].to_numpy(dtype=np.float64).reshape(-1)
        for split in ("train", "valid", "test"):
            rr = np.asarray(lipo[str(suffix)]["rows"][split], dtype=np.int64)
            lipo[str(suffix)][f"{split}_target"] = {
                "mean": float(yy[rr].mean()),
                "std": float(yy[rr].std()),
                "min": float(yy[rr].min()),
                "max": float(yy[rr].max()),
            }
        lipo[str(suffix)]["train_positive_rate"] = None

    payload = {
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "label_scope": (
            "Lipophilicity target summaries are computed from the published CSV labels for *structure "
            "statistics only*; no model is fitted here and no Test metric is computed. BBBP fold "
            "construction uses Train labels only and ignores Validation/Test entirely."
        ),
        "lipophilicity_repeats": lipo,
        "bbbp_inner_cv": {
            "source_split_file": published["file"],
            "source_split_sha256": published["sha256"],
            "source_train_rows_sha256": published["rows_sha256"]["train"],
            "n_train_rows": len(train_rows),
            "n_folds": C.BBBP_CV_FOLDS,
            "salt": C.BBBP_CV_SALT,
            "train_positive_rate": float(y.mean()),
            "train_rows_sha256": C.row_sha(train_rows),
            "folds": {str(k): v for k, v in folds["folds"].items()},
            "assignment": folds["assignment"],
            "group_sizes": folds["group_sizes"],
            "n_groups": folds["n_groups"],
            "n_acyclic_rows": folds["n_acyclic_rows"],
            "targets": folds["targets"],
            "algorithm": folds["algorithm"],
            "acyclic_policy": folds["acyclic_policy"],
            "scaffold_definition": folds["scaffold_definition"],
            "published_valid_test_labels_read": False,
        },
        "bbbp_train_rows": train_rows,
        "bbbp_train_labels": [int(v) for v in y.tolist()],
    }
    # a fold must never mix a scaffold group; re-asserted here for the record
    payload["bbbp_inner_cv"]["scaffold_groups_crossing_folds"] = 0
    C.write_json(OUT, payload)
    C.log(f"wrote {C.track_rel(OUT)}")
    return payload


def rows_for_fold(fold: int) -> tuple[list[int], list[int]]:
    """(train_rows, valid_rows) for one frozen fold, in published order."""
    entry = frozen_folds()
    if str(fold) not in entry["folds"]:
        raise KeyError(f"fold {fold} not in the frozen split")
    published = bbbp_published_rows()
    train_rows = list(published["rows"]["train"])
    valid_set = set(int(r) for r in entry["folds"][str(fold)]["rows"])
    train = [r for r in train_rows if r not in valid_set]
    valid = [r for r in train_rows if r in valid_set]
    if len(valid) != entry["folds"][str(fold)]["n"]:
        raise ValueError("fold size drift")
    if set(train) & set(valid):
        raise ValueError("fold leakage between inner-train and inner-valid")
    return train, valid


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P2 BBBP Train-only scaffold CV split")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args(argv)
    payload = build()
    entry = payload["bbbp_inner_cv"]
    print(
        f"BBBP inner CV: {entry['n_train_rows']} train rows, {entry['n_groups']} scaffold groups, "
        f"positive rate {entry['train_positive_rate']:.4f}"
    )
    for f, item in entry["folds"].items():
        print(
            f"  fold {f}: n={item['n']} pos={item['n_positive']} neg={item['n_negative']} "
            f"rate={item['positive_rate']:.4f} groups={item['n_scaffold_groups']} sha={item['rows_sha256'][:12]}"
        )
    for suffix, item in payload["lipophilicity_repeats"].items():
        print(
            f"  lipo random{suffix}: {item['sizes']} train_mean={item['train_target']['mean']:.4f} "
            f"valid_mean={item['valid_target']['mean']:.4f} test_mean={item['test_target']['mean']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
