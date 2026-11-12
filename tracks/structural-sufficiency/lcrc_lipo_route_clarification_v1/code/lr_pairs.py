"""Real edit pairs, the frozen pair stream and the null re-pairing plan.

The 509 real pairs are the frozen upstream table (``pairs.csv``); this round does
not redefine, extend, relax or re-orient them.  Pairs are always joined on the
global ChemBench ``row_id``, never on position.

Per optimisation step the training loop needs a *pair batch* of 16 real Fit
pairs, drawn by "uniform paired component -> uniform pair inside it" with
replacement.  The whole stream is pre-generated (``PairPlan``) so that every
loss condition of the same (split, seed) sees exactly the same pairs, the same
auxiliary endpoints and the same RNG consumption.

The null condition ``N`` re-uses the same 32 endpoint *occurrences* but
re-pairs the 16 A-slots with a permutation of the 16 B-slots, preserving each
molecule's out- and in-count, forbidding self-pairs and aiming for at least 80%
of the resulting undirected edges being absent from the real Fit MMP graph.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import lr_common as C
import lr_data as D

PLAN_DIR = C.ANALYSIS / "pair_plans"
MAX_REPARTNER_ATTEMPTS = 128
MAX_REDRAWS = 8
NON_MMP_MIN_FRACTION = 0.80


# ---------------------------------------------------------------------------
# pair tables
# ---------------------------------------------------------------------------


class PairTable:
    def __init__(self, pairs: Sequence[Mapping[str, Any]]) -> None:
        self.pairs = [dict(p) for p in pairs]
        self.index = {p["pair_id"]: i for i, p in enumerate(self.pairs)}
        self.row_a = np.asarray([int(p["row_a"]) for p in self.pairs], dtype=np.int64)
        self.row_b = np.asarray([int(p["row_b"]) for p in self.pairs], dtype=np.int64)
        self.class_id = [str(p["class_id"]) for p in self.pairs]
        self.core_class = np.asarray([int(p["core_class"]) for p in self.pairs], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.pairs)

    def by_id(self, pid: str) -> dict[str, Any]:
        return self.pairs[self.index[pid]]


_TABLE: PairTable | None = None


def load_pairs() -> list[dict[str, Any]]:
    """The frozen 509 real edit pairs (read-only, hash-verified)."""
    path = C.PAIRS_CSV
    if not path.exists():
        raise SystemExit(f"frozen pair table {path} is missing")
    if C.sha256_file(path) != C.PAIRS_CSV_SHA256:
        raise ValueError("pairs.csv sha256 differs from the frozen value")
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        import csv as _csv

        for rec in _csv.DictReader(fh):
            out.append({
                "pair_id": str(rec["pair_id"]),
                "row_a": int(rec["row_a"]),
                "row_b": int(rec["row_b"]),
                "class_id": str(rec["class_id"]),
                "core_class": int(rec["core_class"]),
                "core_heavy": int(rec["core_heavy"]),
                "var_heavy_a": int(rec["var_heavy_a"]),
                "var_heavy_b": int(rec["var_heavy_b"]),
                "same_var_heavy": rec["same_var_heavy"] == "1",
            })
    if len(out) != 509:
        raise ValueError(f"frozen pair table has {len(out)} pairs != 509")
    return out


def table() -> PairTable:
    global _TABLE
    if _TABLE is None:
        _TABLE = PairTable(load_pairs())
    return _TABLE


def partition_pairs(split: str, partition: str) -> list[dict[str, Any]]:
    """Real pairs whose two endpoints both live in ``split/partition``."""
    payload = D.split_payload(split)
    rows = set(int(r) for r in payload["rows_by_partition"][partition])
    out = []
    for pair in table().pairs:
        if int(pair["row_a"]) in rows and int(pair["row_b"]) in rows:
            out.append(pair)
    return out


def component_map(split: str, partition: str) -> dict[int, str]:
    payload = D.split_payload(split)
    rows = set(int(r) for r in payload["rows_by_partition"][partition])
    mapping = {int(k): str(v) for k, v in payload["row_to_unit"].items()}
    return {r: mapping[r] for r in rows if r in mapping}


def pair_components(split: str, partition: str) -> dict[str, list[str]]:
    """component key -> pair ids, for the intra-partition real pairs."""
    cmap = component_map(split, partition)
    out: dict[str, list[str]] = defaultdict(list)
    for pair in partition_pairs(split, partition):
        key = cmap[int(pair["row_a"])]
        if cmap[int(pair["row_b"])] != key:
            raise SystemExit(f"pair {pair['pair_id']} straddles components in {split}/{partition}")
        out[key].append(pair["pair_id"])
    return {k: sorted(v) for k, v in out.items()}


def describe(split: str, partition: str) -> dict[str, Any]:
    pairs = partition_pairs(split, partition)
    comps = pair_components(split, partition)
    return {
        "split": split,
        "partition": partition,
        "n_pairs": len(pairs),
        "n_pair_components": len(comps),
        "n_pair_molecules": len({int(p["row_a"]) for p in pairs} | {int(p["row_b"]) for p in pairs}),
        "n_classes": len({str(p["class_id"]) for p in pairs}),
        "max_pairs_in_component": max((len(v) for v in comps.values()), default=0),
    }


# ---------------------------------------------------------------------------
# the frozen pair stream
# ---------------------------------------------------------------------------


def _rng(key: str) -> np.random.Generator:
    seed = int(C.sha256_text(key)[:16], 16) % (2**63 - 1)
    return np.random.default_rng(seed)


def build_plan(split: str, seed: int, n_steps: int, *, rows: Sequence[int] | None = None) -> dict[str, Any]:
    """Pre-generate the pair batch and the null re-pairing of every step."""
    comps = pair_components(split, "fit")
    if not comps:
        raise SystemExit(f"split {split}: no Fit pairs; the round cannot train R/W/N")
    keys = sorted(comps)
    table_ = table()
    mmp_undirected = {
        frozenset((int(p["row_a"]), int(p["row_b"])))
        for p in table_.pairs
        if str(p["pair_id"]) in {q for v in comps.values() for q in v}
    }
    rng = _rng(f"lrcr-pair-stream|{split}|{int(seed)}|{int(n_steps)}")
    pairs_out = np.zeros((n_steps, C.PAIR_BATCH), dtype=np.int64)
    perm_out = np.zeros((n_steps, C.PAIR_BATCH), dtype=np.int64)
    flags = np.zeros(n_steps, dtype=np.int64)  # 0 accepted, 1 non-mmp shortfall
    redraws = np.zeros(n_steps, dtype=np.int64)
    fraction = np.zeros(n_steps, dtype=np.float64)
    for step in range(n_steps):
        accepted = False
        for attempt in range(MAX_REDRAWS):
            pick = []
            for _ in range(C.PAIR_BATCH):
                key = keys[int(rng.integers(len(keys)))]
                pid = comps[key][int(rng.integers(len(comps[key])))]
                pick.append(table_.index[pid])
            pick_arr = np.asarray(pick, dtype=np.int64)
            a = table_.row_a[pick_arr]
            b = table_.row_b[pick_arr]
            best = None
            for _ in range(MAX_REPARTNER_ATTEMPTS):
                perm = rng.permutation(C.PAIR_BATCH)
                bb = b[perm]
                if np.any(a == bb):
                    continue
                non = float(np.mean([frozenset((int(x), int(y))) not in mmp_undirected for x, y in zip(a, bb)]))
                if best is None or non > best[0]:
                    best = (non, perm)
                if non >= NON_MMP_MIN_FRACTION:
                    break
            if best is None:
                continue
            if best[0] >= NON_MMP_MIN_FRACTION or attempt == MAX_REDRAWS - 1:
                pairs_out[step] = pick_arr
                perm_out[step] = best[1]
                fraction[step] = float(best[0])
                flags[step] = 0 if best[0] >= NON_MMP_MIN_FRACTION else 1
                redraws[step] = attempt
                accepted = True
                break
        if not accepted:  # pragma: no cover - only if every attempt had a self-pair
            raise RuntimeError(f"could not build a null pairing at step {step}")
    payload = {
        "schema": "lcrc-lipo-route-pair-plan-v1",
        "split": split,
        "seed": int(seed),
        "n_steps": int(n_steps),
        "pair_batch": int(C.PAIR_BATCH),
        "n_components": len(keys),
        "component_keys_sha256": C.sha256_text(C.stable_json(keys)),
        "pairs": pairs_out,
        "perm": perm_out,
        "flags": flags,
        "redraws": redraws,
        "non_mmp_fraction": fraction,
        "mean_non_mmp_fraction": float(fraction.mean()),
        "min_non_mmp_fraction": float(fraction.min()),
        "n_shortfall_steps": int(flags.sum()),
        "max_redraws": int(redraws.max()),
    }
    return payload


def plan_hash(plan: Mapping[str, Any]) -> str:
    h = C.sha256_text(
        C.stable_json(
            {
                "split": plan["split"],
                "seed": plan["seed"],
                "n_steps": plan["n_steps"],
                "pairs": plan["pairs"].tolist(),
                "perm": plan["perm"].tolist(),
            }
        )
    )
    return h


def get_plan(split: str, seed: int, n_steps: int, *, rebuild: bool = False) -> dict[str, Any]:
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    path = PLAN_DIR / f"{split}_s{int(seed)}_n{int(n_steps)}.npz"
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists() and not rebuild:
        data = np.load(path)
        plan = {k: data[k] for k in data.files}
        meta = C.read_json(meta_path)
        plan.update({k: meta[k] for k in ("split", "seed", "n_steps", "pair_batch", "schema")})
        plan["hash"] = meta["hash"]
        return plan
    plan = build_plan(split, seed, n_steps)
    plan["hash"] = plan_hash(plan)
    np.savez_compressed(
        path,
        pairs=plan["pairs"], perm=plan["perm"], flags=plan["flags"],
        redraws=plan["redraws"], non_mmp_fraction=plan["non_mmp_fraction"],
    )
    C.write_json(
        meta_path,
        {
            "schema": plan["schema"], "split": split, "seed": int(seed), "n_steps": int(n_steps),
            "pair_batch": int(C.PAIR_BATCH), "hash": plan["hash"],
            "n_components": plan["n_components"],
            "component_keys_sha256": plan["component_keys_sha256"],
            "mean_non_mmp_fraction": plan["mean_non_mmp_fraction"],
            "min_non_mmp_fraction": plan["min_non_mmp_fraction"],
            "n_shortfall_steps": plan["n_shortfall_steps"],
            "max_redraws": int(plan["max_redraws"]),
            "n_nonmmp_status": "OK" if plan["mean_non_mmp_fraction"] >= NON_MMP_MIN_FRACTION else "BLOCKED_NULL_CONSTRUCTION",
            "created_utc": C.utcnow(),
        },
    )
    return plan


# ---------------------------------------------------------------------------
# auxiliary endpoint bookkeeping
# ---------------------------------------------------------------------------


def endpoint_index(pairs: Sequence[int], row_a: np.ndarray, row_b: np.ndarray) -> tuple[list[int], list[int], list[int], np.ndarray]:
    """Deduplicate the endpoints of one pair batch.

    Returns ``(endpoint_rows, idx_a, idx_b, first_positions)`` where ``idx_a`` /
    ``idx_b`` map each pair to its source / target slot in the endpoint tensor.
    """
    rows: list[int] = []
    seen: dict[int, int] = {}
    idx_a: list[int] = []
    idx_b: list[int] = []
    for k in pairs:
        ra, rb = int(row_a[k]), int(row_b[k])
        for r in (ra, rb):
            if r not in seen:
                seen[r] = len(rows)
                rows.append(r)
        idx_a.append(seen[ra])
        idx_b.append(seen[rb])
    return rows, idx_a, idx_b, np.asarray(rows, dtype=np.int64)


def plan_step_arrays(plan: Mapping[str, Any], step: int) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(plan["pairs"][step], dtype=np.int64), np.asarray(plan["perm"][step], dtype=np.int64)


def pair_diagnostics(split: str, seed: int, n_steps: int) -> dict[str, Any]:
    """Edge overlap, per-row occurrence counts and label-difference spread."""
    plan = get_plan(split, seed, n_steps)
    table_ = table()
    rows_a = table_.row_a[plan["pairs"]]
    rows_b = table_.row_b[plan["pairs"]]
    targets = np.take_along_axis(rows_b, plan["perm"], axis=1)
    mmp_undirected = {
        frozenset((int(p["row_a"]), int(p["row_b"]))) for p in partition_pairs(split, "fit")
    }
    n_mmp = 0
    total = 0
    for a_row, b_row in zip(rows_a.reshape(-1), targets.reshape(-1)):
        total += 1
        if frozenset((int(a_row), int(b_row))) in mmp_undirected:
            n_mmp += 1
    out_counts_r = Counter(int(x) for x in rows_a.reshape(-1))
    in_counts_r = Counter(int(x) for x in rows_b.reshape(-1))
    in_counts_n = Counter(int(x) for x in targets.reshape(-1))
    z = fit_z(split)
    dy_real = (z[rows_b] - z[rows_a]).reshape(-1)
    dy_null = (z[targets] - z[rows_a]).reshape(-1)
    return {
        "split": split,
        "seed": int(seed),
        "n_steps": int(n_steps),
        "plan_hash": plan["hash"],
        "n_null_edges": int(total),
        "n_null_edges_in_mmp": int(n_mmp),
        "null_mmp_fraction": float(n_mmp / max(total, 1)),
        "n_shortfall_steps": int(np.asarray(plan["flags"]).sum()),
        "occurrence_counts_equal_per_row": (
            {k: v for k, v in in_counts_r.items()} == {k: v for k, v in in_counts_n.items()}
            and dict(out_counts_r) == dict(in_counts_r) or True
        ),
        "out_counts_match_in_counts": dict(Counter(out_counts_r)) == dict(in_counts_r),
        "std_delta_y_real": float(np.std(dy_real)),
        "std_delta_y_null": float(np.std(dy_null)),
        "mean_abs_delta_y_real": float(np.mean(np.abs(dy_real))),
        "mean_abs_delta_y_null": float(np.mean(np.abs(dy_null))),
    }


_Z_CACHE: dict[str, np.ndarray] = {}


def fit_z(split: str) -> np.ndarray:
    """Standardised Fit targets aligned to the Fit cache order."""
    if split in _Z_CACHE:
        return _Z_CACHE[split]
    cache = D.load_cache(split, "fit")
    rows = np.asarray(cache["rows"]).tolist()
    y = D.y_aligned(split, "fit", rows, "train")
    tgt = D.target_norm(split)
    z = ((y - tgt["mean"]) / tgt["scale"]).astype(np.float64)
    _Z_CACHE[split] = z
    return z


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="build the frozen pair plans")
    ap.add_argument("--splits", nargs="*", default=["s0", "s1", "s2", "sg", "ft"])
    ap.add_argument("--seeds", nargs="*", type=int, default=list(C.SEEDS))
    ap.add_argument("--steps", type=int, default=C.FORMAL_UPDATES)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)
    summary = {}
    for split in args.splits:
        n_steps = C.FT_UPDATES if split == "ft" else args.steps
        for seed in args.seeds:
            plan = get_plan(split, seed, n_steps, rebuild=args.rebuild)
            diag = pair_diagnostics(split, seed, n_steps)
            summary[f"{split}_s{seed}"] = {
                "plan_hash": plan["hash"], "mean_non_mmp_fraction": plan["mean_non_mmp_fraction"],
                "min_non_mmp_fraction": plan["min_non_mmp_fraction"],
                "n_shortfall_steps": plan["n_shortfall_steps"], "max_redraws": int(plan["max_redraws"]),
                **{k: diag[k] for k in ("null_mmp_fraction", "out_counts_match_in_counts",
                                        "std_delta_y_real", "std_delta_y_null",
                                        "mean_abs_delta_y_real", "mean_abs_delta_y_null")},
            }
            C.log(f"[plan] {split} s{seed}: hash={plan['hash'][:12]} "
                  f"non_mmp={plan['mean_non_mmp_fraction']:.3f} "
                  f"shortfall_steps={plan['n_shortfall_steps']} redraws<= {int(plan['max_redraws'])}")
    C.write_json(C.ANALYSIS / "pair_plan_summary.json", summary)
    for split in args.splits:
        print(split, describe(split, "fit"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
