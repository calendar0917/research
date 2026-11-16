"""Diagnostic: is the incomplete control R a support limit or an attempt budget?

Read-only with respect to the formal round.  It never writes into
``results/sq-pilot-small-v1`` and changes no frozen constant; it only *measures*
how fillable the unfilled slots are and (with ``--refit``) what the R arm would
look like if every slot could be filled.

Two questions, both answerable on CPU:

1. For each unfilled slot, enumerate the complete parameter-assignment space of
   that shape (every injective assignment of the values used in the AST into the
   Train vocabulary of that column) and count how many remapped queries clear the
   frozen support floor.  The pre-declared per-slot attempt cap is
   ``sq_search.MAX_R_ATTEMPTS_PER_SLOT``; comparing the two says whether the
   incompleteness is structural or a sampling-budget artefact.
2. With ``--refit``, rebuild R with a larger (explicitly non-pre-declared) cap,
   refit the frozen head presets on CPU and report Val/Test RMSE next to the
   formal A / R / F numbers.  This is an EXPLORATORY number: it is not part of
   the round's decision, it only informs whether a follow-up round should
   pre-register a larger cap.

Usage::

    uv run python tracks/structural-query-learning/code/sq_diag_r_budget.py \
        --dataset ogbg-mollipo [--refit] [--max-attempts 4096]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import sq_ast as A  # noqa: E402
import sq_bank as B  # noqa: E402
import sq_common as C  # noqa: E402
import sq_data as D  # noqa: E402
import sq_search as S  # noqa: E402
import sq_xgb as X  # noqa: E402

SPLITS = ("train", "valid", "test")


def _families(ast: tuple) -> dict[tuple[str, ...], set[int]]:
    used: dict[tuple[str, ...], set[int]] = {}
    for kind, leaf in A.leaf_params(ast):
        if kind == "node" and leaf[0] == "ATOM_EQ":
            used.setdefault(("atom", str(leaf[1])), set()).add(int(leaf[2]))
        elif kind == "node" and leaf[0] == "DEG_EQ":
            used.setdefault(("deg",), set()).add(int(leaf[1]))
        elif kind == "node" and leaf[0] == "MIN_RING_EQ":
            used.setdefault(("ring",), set()).add(int(leaf[1]))
        elif kind == "bond" and leaf[0] == "BOND_EQ":
            used.setdefault(("bond", str(leaf[1])), set()).add(int(leaf[2]))
    return used


def _domain(kind: tuple[str, ...], vocabulary: dict) -> list[int]:
    if kind[0] == "atom":
        return [int(v) for v in vocabulary["atom_values"][int(kind[1])]]
    if kind[0] == "bond":
        return [int(v) for v in vocabulary["bond_values"][int(kind[1])]]
    if kind[0] == "deg":
        return [int(v) for v in vocabulary["degree_values"]]
    return [int(v) for v in vocabulary["ring_values"]]


def enumerate_slot(ast: tuple, train, vocabulary, cache) -> dict[str, object]:
    families = _families(ast)
    spaces = []
    for kind, values in families.items():
        pool = _domain(kind, vocabulary)
        assignments = [
            dict(zip(sorted(values), perm)) for perm in itertools.permutations(pool, len(values))
        ]
        spaces.append((kind, assignments))
    total = 1
    for _, assignments in spaces:
        total *= len(assignments)
    feasible = 0
    best = 0
    for combo in itertools.product(*[a for _, a in spaces]):
        maps = {kind: assignment for (kind, _), assignment in zip(spaces, combo)}
        candidate = S.remap_ast(ast, maps)
        ok, _reason, mask = B.validate_query_on_train(train, candidate, cache=cache)
        if ok:
            feasible += 1
            best = max(best, int((train.graph_counts(mask) > 0).sum()))
    expected_attempts = None if not feasible else int(round(total / feasible))
    tail = (
        None
        if expected_attempts is None
        else float((1.0 - feasible / max(1, total)) ** S.MAX_R_ATTEMPTS_PER_SLOT)
    )
    return {
        "ast": A.canon(ast),
        "assignments": int(total),
        "feasible": int(feasible),
        "feasible_fraction": float(feasible / max(1, total)),
        "best_train_support": int(best),
        "expected_attempts": expected_attempts,
        "p_unfilled_at_declared_cap": tail,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ogbg-mollipo", choices=list(C.DATASETS))
    parser.add_argument("--refit", action="store_true", help="also refit the R arm locally")
    parser.add_argument("--max-attempts", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=X.NUM_BOOST_ROUND)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    dataset = args.dataset
    out = C.RESULTS_DIR / dataset
    cache = D.load_cache(dataset)
    packed = {split: D.packed_split(cache, split) for split in SPLITS}
    vocabulary = D.train_vocabulary(cache)
    cache_masks = B.MaskCache()
    base_records, _ = B.build_base_bank(packed["train"], vocabulary, cache=cache_masks)
    scaler = C.train_target_scaler(cache["labels"][D.split_indices(cache, "train")])
    targets = {
        split: cache["labels"][D.split_indices(cache, split)].astype(np.float64) for split in SPLITS
    }

    formal_r = json.loads((out / "random_library.json").read_text(encoding="utf-8"))
    discovered_payload = json.loads((out / "discovery.json").read_text(encoding="utf-8"))
    discovered = [
        B.QueryRecord(
            ast=A.from_json(item["ast_tree"]),
            label=item["label"],
            source="discovered",
            outer_round=int(item["outer_round"]),
            train_support=int(item["train_support"]),
            train_gain=item.get("train_gain"),
        )
        for item in discovered_payload["discovered"]
    ]
    unfilled = [slot for slot in formal_r["slots"] if not slot["filled"]]
    by_label = {record.label: record for record in discovered}

    print(f"# R-budget diagnosis — {dataset}")
    print(f"formal status {formal_r['status']}: {formal_r['n_filled']}/{formal_r['n_requested']} slots")
    print(f"declared cap {S.MAX_R_ATTEMPTS_PER_SLOT} attempts/slot, support floor {B.MIN_TRAIN_MOLECULES}\n")
    rows = []
    for slot in unfilled:
        record = by_label[slot["source_label"]]
        row = enumerate_slot(record.ast, packed["train"], vocabulary, cache_masks)
        rows.append(row)
        print(json.dumps(row, indent=2))
    if rows:
        worst = max(row["p_unfilled_at_declared_cap"] or 1.0 for row in rows)
        print(
            f"\n=> {len(rows)} unfilled slots; enumeration shows all of them are fillable in principle "
            f"(feasible fraction {min(row['feasible_fraction'] for row in rows):.4f}"
            f"..{max(row['feasible_fraction'] for row in rows):.4f}); per-slot chance of running out of the "
            f"declared {S.MAX_R_ATTEMPTS_PER_SLOT} attempts is up to {worst:.2f}."
        )

    if not args.refit:
        return 0

    print(f"\n# EXPLORATORY refit (not part of the formal round): R with {args.max_attempts} attempts/slot")
    masks_local = B.MaskCache()
    built = S.build_random_library(
        discovered=discovered,
        train_packed=packed["train"],
        vocabulary=vocabulary,
        existing_records=base_records,
        cache=masks_local,
        max_attempts_per_slot=args.max_attempts,
    )
    print(
        f"R rebuilt: {built['status']} {built['n_filled']}/{built['n_requested']}, "
        f"shapes match {built['all_shapes_match']}"
    )
    records_r = list(base_records) + list(built["records"])
    matrices = {
        split: B.design_matrix(packed[split], records_r, cache=masks_local, include_base=True)[0]
        for split in SPLITS
    }
    names = B.design_matrix(packed["train"], records_r, cache=masks_local, include_base=True)[1]
    best = None
    for preset in X.PRESET_ORDER:
        head = X.run_head(
            "R", preset, matrices, targets, scaler, names,
            device=args.device, num_boost_round=args.rounds,
        )
        state = head.best
        print(
            f"  {preset}: val {state['valid_rmse_raw']:.4f} test {state['test_rmse_raw']:.4f} "
            f"train {state['train_rmse_raw']:.4f} @{state['iteration']}"
        )
        key = (state["valid_rmse_raw"], X.PRESET_ORDER.index(preset), state["iteration"])
        if best is None or key < best[0]:
            best = (key, preset, state)
    _, preset, state = best
    formal_a = json.loads((out / "methods_fra.json").read_text(encoding="utf-8"))
    print("\ncomparison (raw RMSE)")
    print("  formal  A : val %.4f test %.4f" % (
        formal_a["A"]["selected"]["valid_rmse_raw"], formal_a["A"]["selected"]["test_rmse_raw"]))
    print("  formal  R : val %.4f test %.4f" % (
        formal_a["R"]["selected"]["valid_rmse_raw"], formal_a["R"]["selected"]["test_rmse_raw"]))
    print("  formal  F : val %.4f test %.4f" % (
        formal_a["F"]["selected"]["valid_rmse_raw"], formal_a["F"]["selected"]["test_rmse_raw"]))
    print(f"  R(complete, {args.max_attempts} attempts): val {state['valid_rmse_raw']:.4f} "
          f"test {state['test_rmse_raw']:.4f} [{preset}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
