"""Phase 0.2 — three kinds of "out of vocabulary", from the frozen matrices only.

No enumeration, no training.  Every FULL **instance** of the frozen MSDS-v0
enumeration cache is assigned to exactly one class:

1. ``in_vocab``                  the FULL type is one of the 3 652 frozen columns;
2. ``train_seen_but_not_frozen`` Train document frequency > 0 but no column, with a
   deterministic reason from the frozen selection gates;
3. ``train_never_seen``          Train document frequency 0.

Reported per structural family and split: instance share, molecule coverage and the
COARSE-fallback coverage of the non-``in_vocab`` instances.  Class 2 is never called
"truly unseen": the frozen vocabulary is a *support-and-cap* filter over seen types.
"""

from __future__ import annotations

import csv
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np

import sga_common as K

K.add_prev_code_path()

import msds_common as MC  # noqa: E402
import msds_data as D  # noqa: E402
import msds_vocab as VOCAB  # noqa: E402

FAMILIES = MC.FAMILIES
CLASSES = ("in_vocab", "train_seen_but_not_frozen", "train_never_seen")
REASONS = (
    "parent_coarse_not_retained",
    "below_min_document_frequency",
    "below_min_scaffold_frequency",
    "top_k_truncated_by_family_cap",
)


def main() -> None:
    t0 = time.perf_counter()
    K.ensure_dirs()
    molecules = D.load_molecules()
    splits = {k: np.asarray(v, dtype=np.int64) for k, v in molecules.split.items()}
    scaffolds, _ = D.load_scaffolds(molecules)
    uniq = {s: i for i, s in enumerate(sorted(set(scaffolds)))}
    scaffold_ids = np.asarray([uniq[s] for s in scaffolds], dtype=np.int64)
    n_scaf = int(scaffold_ids.max()) + 1

    discovered = VOCAB.load_discovered(str(K.MSDS_DATA / "phaseA_instances.npz"))
    with np.load(K.MSDS_DATA / "vocab.npz") as z:
        frozen_full = np.asarray(z["full_ids"], dtype=np.int64)
        frozen_coarse = np.asarray(z["coarse_ids"], dtype=np.int64)

    train = splits["train"]
    fstats = VOCAB.type_stats(discovered.full, train, splits["valid"], splits["test"],
                              molecules.label.astype(np.float64), scaffold_ids, n_scaf)
    cstats = VOCAB.type_stats(discovered.coarse, train, splits["valid"], splits["test"],
                              molecules.label.astype(np.float64), scaffold_ids, n_scaf)
    df = fstats["document_frequency_train"]
    sf = fstats["scaffold_frequency_train"]
    parent = discovered.full_parent
    n_full_types = discovered.full.shape[1]

    in_vocab = np.zeros(n_full_types, dtype=bool)
    in_vocab[frozen_full] = True
    coarse_retained = np.zeros(discovered.coarse.shape[1], dtype=bool)
    coarse_retained[frozen_coarse] = True
    parent_retained = coarse_retained[parent]

    # ---- deterministic gate attribution (priority order = the frozen filter order)
    reason = np.full(n_full_types, "", dtype=object)
    reason[train_seen := (df > 0)] = "unclassified"
    gate_fail = {
        "parent_coarse_not_retained": (~parent_retained),
        "below_min_document_frequency": (df < MC.FULL_MIN_DOC_FREQ),
        "below_min_scaffold_frequency": (sf < MC.FULL_MIN_SCAFFOLDS),
        "top_k_truncated_by_family_cap": np.zeros(n_full_types, dtype=bool),
    }
    # reproduce the frozen selection to identify per-family cap truncation
    from msds_vocab import _select  # noqa: F401  (same code path as MSDS-v0)

    eligible_all = (df >= MC.FULL_MIN_DOC_FREQ) & (sf >= MC.FULL_MIN_SCAFFOLDS) & parent_retained
    truncated = np.zeros(n_full_types, dtype=bool)
    for fam in range(len(FAMILIES)):
        fam_types = np.flatnonzero(discovered.full_family == fam)
        keep = _select(fam_types, fstats, discovered.full_keys, MC.FULL_MIN_DOC_FREQ,
                       MC.FULL_MIN_SCAFFOLDS, MC.FULL_MAX_PER_FAMILY,
                       allowed=(discovered.full_family == fam) & parent_retained)
        dropped = np.setdiff1d(fam_types[eligible_all[fam_types]], keep)
        truncated[dropped] = True
    if not np.array_equal(np.sort(np.flatnonzero(eligible_all & ~truncated)), np.sort(frozen_full)):
        raise SystemExit("reproduced eligible set does not match the frozen FULL vocabulary")
    gate_fail["top_k_truncated_by_family_cap"] = truncated

    for cls_reason in REASONS:
        mask = (df > 0) & ~in_vocab & (reason == "unclassified") & gate_fail[cls_reason]
        reason[mask] = cls_reason
    reason[(df > 0) & ~in_vocab & (reason == "unclassified")] = "other_unresolved"
    reason[(df == 0)] = "train_never_seen"
    reason[in_vocab] = "in_vocab"

    # collapse the reasons into the three classes
    cls_of = np.full(n_full_types, "train_seen_but_not_frozen", dtype=object)
    cls_of[reason == "in_vocab"] = "in_vocab"
    cls_of[reason == "train_never_seen"] = "train_never_seen"

    # multi-gate co-occurrence (a type can fail several gates at once)
    cooccur = Counter()
    for cls_reason in REASONS[:3]:
        cooccur[f"{cls_reason}_only"] = int(((reason == cls_reason) &
                                             (np.sum([gate_fail[r] for r in REASONS[:3]], axis=0) == 1)).sum())
    n_gates_failed = np.sum([gate_fail[r] for r in REASONS], axis=0)
    cooccur["types_failing_2_or_more_gates"] = int(((df > 0) & ~in_vocab & (n_gates_failed >= 2)).sum())

    # ---- per split / family instance + molecule statistics
    rows_out: list[dict[str, Any]] = []
    for split_name in ("train", "valid", "test"):
        rows = splits[split_name]
        full_sub = discovered.full[rows]
        counts = np.asarray(full_sub.sum(axis=0)).reshape(-1).astype(np.float64)
        present = np.zeros((rows.size, n_full_types), dtype=bool)
        coo = full_sub.tocoo()
        present[coo.row, coo.col] = True
        coarse_sub = discovered.coarse[rows]
        coarse_counts = np.asarray(coarse_sub.sum(axis=0)).reshape(-1).astype(np.float64)
        for fam in range(len(FAMILIES)):
            fam_mask = discovered.full_family == fam
            total = float(counts[fam_mask].sum())
            for cls in CLASSES:
                sel = fam_mask & (cls_of == cls)
                n_inst = float(counts[sel].sum())
                mol_cov = int(present[:, sel].any(axis=1).sum()) if sel.any() else 0
                rows_out.append({
                    "split": split_name, "family": FAMILIES[fam], "class": cls,
                    "n_types": int(sel.sum()),
                    "n_instances": n_inst,
                    "instance_share": (n_inst / total) if total else None,
                    "n_molecules": mol_cov,
                    "molecule_coverage": (mol_cov / rows.size) if rows.size else None,
                })
            # coarse fallback coverage of the non-in-vocab instances
            not_in = fam_mask & ~in_vocab
            fallback = not_in & parent_retained
            no_fallback = not_in & ~parent_retained
            n_not = float(counts[not_in].sum())
            rows_out.append({
                "split": split_name, "family": FAMILIES[fam], "class": "coarse_fallback_of_not_in_vocab",
                "n_types": int(not_in.sum()),
                "n_instances": n_not,
                "instance_share": (n_not / total) if total else None,
                "n_instances_with_coarse_fallback": float(counts[fallback].sum()),
                "coarse_fallback_coverage": (float(counts[fallback].sum()) / n_not) if n_not else None,
                "n_instances_without_fallback": float(counts[no_fallback].sum()),
                "uncovered_fraction": (float(counts[no_fallback].sum()) / n_not) if n_not else None,
                "n_molecules": int(present[:, not_in].any(axis=1).sum()) if not_in.any() else 0,
                "molecule_coverage": (int(present[:, not_in].any(axis=1).sum()) / rows.size) if not_in.any() else 0.0,
            })
    _write_csv(K.ANALYSIS / "oov_classes.csv", rows_out)

    # ---- reason table (type level, per family)
    reason_rows = []
    for fam in range(len(FAMILIES)):
        fam_mask = discovered.full_family == fam
        for cls in REASONS + ("in_vocab", "train_never_seen"):
            sel = fam_mask & (reason == cls)
            if not sel.any():
                continue
            reason_rows.append({
                "family": FAMILIES[fam], "class": cls, "n_types": int(sel.sum()),
                "median_train_document_frequency": float(np.median(df[sel])),
                "median_train_scaffold_frequency": float(np.median(sf[sel])),
            })
    _write_csv(K.ANALYSIS / "oov_reasons.csv", reason_rows)

    # ---- per-molecule coverage vectors for Phase D (frozen-boundary free)
    full_all = discovered.full
    total_per_mol = np.asarray(full_all.sum(axis=1)).reshape(-1).astype(np.float64)
    in_vocab_cols = np.flatnonzero(in_vocab)
    in_vocab_per_mol = np.asarray(full_all[:, in_vocab_cols].sum(axis=1)).reshape(-1).astype(np.float64)
    seen_not_frozen_cols = np.flatnonzero(np.isin(reason, list(REASONS)))
    seen_not_per_mol = np.asarray(full_all[:, seen_not_frozen_cols].sum(axis=1)).reshape(-1).astype(np.float64)
    never_seen_cols = np.flatnonzero(reason == "train_never_seen")
    never_per_mol = np.asarray(full_all[:, never_seen_cols].sum(axis=1)).reshape(-1).astype(np.float64)
    coarse_fallback_cols = np.flatnonzero(coarse_retained)
    np.savez_compressed(
        K.CACHE / "oov_per_molecule.npz",
        total=total_per_mol, in_vocab=in_vocab_per_mol,
        seen_not_frozen=seen_not_per_mol, never_seen=never_per_mol,
        type_reason=np.asarray([str(x) for x in reason]),
        type_has_coarse_parent_retained=parent_retained,
        n_coarse_retained_columns=np.int64(coarse_fallback_cols.size),
    )

    payload = {
        "round_id": K.ROUND_ID,
        "phase": "0.2 three-way out-of-vocabulary statistics",
        "definition": {
            "in_vocab": "the FULL type is one of the frozen FULL columns",
            "train_seen_but_not_frozen": "Train document frequency > 0 but filtered out by the "
                                         "frozen selection gates (reason in oov_reasons.csv)",
            "train_never_seen": "Train document frequency == 0",
            "coarse_fallback": "a non-in_vocab FULL instance whose COARSE parent is a frozen "
                               "COARSE column; the COARSE block still counts it",
            "warning": "class 2 is NOT 'truly unseen' -- it is seen and deliberately filtered",
        },
        "n_full_types": int(n_full_types),
        "n_in_vocab_types": int(in_vocab.sum()),
        "n_train_seen_types": int((df > 0).sum()),
        "n_train_never_seen_types": int((df == 0).sum()),
        "class_counts": {c: int((cls_of == c).sum()) for c in CLASSES},
        "reason_counts": {r: int((reason == r).sum()) for r in REASONS}
                         | {"in_vocab": int((reason == "in_vocab").sum()),
                            "train_never_seen": int((reason == "train_never_seen").sum())},
        "gate_cooccurrence": dict(cooccur),
        "multi_gate_rule": "reason = first failing gate in the frozen filter order "
                           "(parent retained -> df -> scaffold); co-occurrence is reported separately",
        "instances": rows_out,
        "seconds": float(time.perf_counter() - t0),
        "git": K.git_state(),
        "environment": K.environment(),
    }
    K.write_json(K.ANALYSIS / "phase0_oov.json", payload)
    summary = payload["class_counts"]
    reasons = payload["reason_counts"]
    print(f"[p0.2] types: in_vocab={summary['in_vocab']} seen_not_frozen="
          f"{summary['train_seen_but_not_frozen']} never_seen={summary['train_never_seen']}", flush=True)
    print(f"[p0.2] reasons: " + ", ".join(f"{r}={reasons[r]}" for r in REASONS), flush=True)
    print(f"[p0.2] done in {time.perf_counter() - t0:.1f}s", flush=True)


def _write_csv(path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
