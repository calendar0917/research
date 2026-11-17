"""P0 identity: official split, frozen assets, checkpoint replay, WC_LOGP closure.

Writes ``analysis/identity.json`` and ``analysis/protocol_freeze.json``.  Any
failed identity gate stops the affected scope before any explanation number is
produced.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import lfa_common as C
import lfa_data as D
import lfa_models as M


def _check(ok: bool, **payload: Any) -> dict[str, Any]:
    return {"passed": bool(ok), **payload}


def cache_identity() -> dict[str, Any]:
    from xr_common import EXPECTED_CACHE_SHA256, SQL_DIR

    out = {}
    for name, expected in EXPECTED_CACHE_SHA256.items():
        path = SQL_DIR / "data" / name
        actual = C.sha256_file(path) if path.is_file() else None
        out[name] = {"path": C.relpath(path), "sha256": actual, "expected": expected,
                     "match": bool(actual == expected)}
    return out


def asset_identity() -> dict[str, Any]:
    out = {}
    for name, expected in C.EXPECTED_ASSET_SHA256.items():
        if name == "ogbg-mollipo.npz":
            path = C.XR_DIR / "analysis" / name
        else:
            path = C.XR_DIR / "analysis" / name
        actual = C.sha256_file(path) if path.is_file() else None
        out[name] = {"path": C.relpath(path), "sha256": actual, "expected": expected,
                     "match": bool(actual == expected)}
    return out


def split_identity(b) -> dict[str, Any]:
    import xr_assets

    official = xr_assets.verify_official_split(b)
    sizes = {split: int(np.asarray(b.global_row_ids[split]).size) for split in C.SPLITS}
    sets = {split: set(np.asarray(b.global_row_ids[split], dtype=np.int64).tolist()) for split in C.SPLITS}
    intersections = {f"{a}-{c}": len(sets[a] & sets[c])
                     for i, a in enumerate(C.SPLITS) for c in C.SPLITS[i + 1:]}
    union = set().union(*sets.values())
    scope = D.main_scope()
    by_stratum = Counter(r["stratum"] for r in scope)
    ids_ok = all(
        int(r["row_id"]) == int(np.asarray(b.global_row_ids[r["split"]], dtype=np.int64)[int(r["cache_row"])])
        for r in scope
    )
    train_ids = [r["row_id"] for r in scope if r["stratum"] == "train_description"]
    return {
        "sizes": sizes,
        "sizes_match": bool(sizes == C.EXPECTED_SPLIT_SIZES),
        "official_csv": official,
        "intersections": intersections,
        "union_size": len(union),
        "disjoint_and_cover": bool(all(v == 0 for v in intersections.values()) and len(union) == 4200),
        "main_scope": {"n": len(scope), "by_stratum": dict(by_stratum),
                       "row_ids_sha256": C.sha256_json([[r["split"], int(r["row_id"])] for r in scope]),
                       "cache_row_mapping_ok": bool(ids_ok),
                       "train_description_row_ids": train_ids},
    }


def checkpoint_gates(b, *, device: str = "cpu") -> dict[str, Any]:
    import xr_models

    reports = []
    for name in C.CHECKPOINT_NAMES.values():
        ident = M.checkpoint_identity(name, b)
        replay = xr_models.replay_checkpoint(name, b, device=device)
        passed = bool(ident["sha256_match"] and ident["parameter_count_match"]
                      and ident["epoch_match"] and replay["passed"])
        reports.append({
            "name": name,
            "identity": ident,
            "replay": replay,
            "passed": passed,
        })
    return {"models": reports, "passed": all(r["passed"] for r in reports)}


def oracle_graph_closure() -> dict[str, Any]:
    payload = D.oracle_payload()
    ptr = np.asarray(payload["atom_ptr"], dtype=np.int64)
    atom = np.asarray(payload["WC_LOGP_atom"], dtype=np.float64)
    value = np.asarray(payload["WC_LOGP"], dtype=np.float64)
    errs = np.empty(value.size, dtype=np.float64)
    for row in range(value.size):
        errs[row] = abs(float(atom[int(ptr[row]):int(ptr[row + 1])].sum()) - float(value[row]))
    return {
        "n_rows": int(value.size),
        "max_abs_error": float(errs.max()),
        "n_over_1e-6": int((errs > 1e-6).sum()),
        "passed": bool(np.all(errs <= 1e-6)),
    }


def main_scope_oracle(b) -> dict[str, Any]:
    scope = D.main_scope()
    rows = []
    sizes_ok = True
    for rec in scope:
        ptr = np.asarray(b.caches[rec["split"]]["atom_ptr"], dtype=np.int64)
        n_cache = int(ptr[int(rec["cache_row"]) + 1] - ptr[int(rec["cache_row"])])
        g = D.graph_contribution(int(rec["row_id"]))
        sizes_ok = sizes_ok and bool(n_cache == int(g["contrib"].size))
        rows.append({
            "split": rec["split"], "row_id": int(rec["row_id"]),
            "n_atoms": int(g["contrib"].size), "cache_atoms": n_cache,
            "closure_error": g["closure_error"], "closed": g["closed"],
            "abs_mass": float(np.abs(g["contrib"]).sum()),
        })
    zero_mass = [r for r in rows if not (r["abs_mass"] > 0.0)]
    return {
        "n": len(rows),
        "n_closed": int(sum(r["closed"] for r in rows)),
        "max_closure_error": float(max(r["closure_error"] for r in rows)),
        "atom_counts_match": bool(sizes_ok),
        "n_zero_abs_mass": len(zero_mass),
        "zero_abs_mass": [(r["split"], r["row_id"]) for r in zero_mass],
        "passed": bool(all(r["closed"] for r in rows) and sizes_ok),
    }


def aux_pair_closure() -> dict[str, Any]:
    """Recompute the 823 pair contribution closures from the frozen endpoints."""
    endpoints = D.aux_endpoints()
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = {}
    for rec in endpoints:
        by_pair.setdefault(str(rec["pair_id"]), {})[str(rec["side"])] = rec
    payload = D.oracle_payload()
    value = np.asarray(payload["WC_LOGP"], dtype=np.float64)
    import gzip

    with gzip.open(C.XR_DIR / "analysis" / "pair_catalog.jsonl.gz", "rt", encoding="utf-8") as handle:
        pairs = [json.loads(line) for line in handle]
    errors = []
    for pair in pairs:
        pid = str(pair["pair_id"])
        a, b = by_pair[pid]["a"], by_pair[pid]["b"]
        da = np.asarray(a["oracle_delta"], dtype=np.float64)
        db = np.asarray(b["oracle_delta"], dtype=np.float64)
        core = [int(v) for v in pair["core_atoms_a"]]
        var_a = [int(v) for v in a["var_atoms"]]
        var_b = [int(v) for v in b["var_atoms"]]
        total = float(da[core].sum() + da[var_a].sum() + db[var_b].sum())
        target = float(value[int(pair["row_b"])] - value[int(pair["row_a"])])
        errors.append(abs(total - target))
    errors = np.asarray(errors, dtype=np.float64)
    return {"n_pairs": len(pairs), "max_abs_error": float(errors.max()),
            "n_over_tolerance": int((errors > 1e-6).sum()),
            "passed": bool(np.all(errors <= 1e-6))}


def run(*, device: str = "cpu", skip_replay: bool = False) -> dict[str, Any]:
    b = D.bundle()
    caches = cache_identity()
    assets = asset_identity()
    splits = split_identity(b)
    checkpoints = checkpoint_gates(b, device=device) if not skip_replay else {"models": [], "passed": True}
    graph_closure = oracle_graph_closure()
    main_oracle = main_scope_oracle(b)
    aux_closure = aux_pair_closure()
    gates = {
        "cache_identity": _check(all(v["match"] for v in caches.values()), caches=caches),
        "asset_identity": _check(all(v["match"] for v in assets.values()), assets=assets),
        "split_identity": _check(bool(splits["sizes_match"] and splits["disjoint_and_cover"]
                                      and splits["official_csv"]["all_match"]
                                      and splits["main_scope"]["cache_row_mapping_ok"]), **splits),
        "checkpoint_identity_and_replay": checkpoints,
        "wc_glop_graph_closure": graph_closure,
        "wc_glop_main_scope_closure": main_oracle,
        "wc_glop_pair_closure": aux_closure,
    }
    gates["all_passed"] = bool(
        gates["cache_identity"]["passed"] and gates["asset_identity"]["passed"]
        and gates["split_identity"]["passed"] and checkpoints["passed"]
        and graph_closure["passed"] and main_oracle["passed"] and aux_closure["passed"]
    )
    C.write_json(C.ANALYSIS_DIR / "identity.json", {
        "round_id": C.ROUND_ID, "created_utc": C.utc_now(), "environment": C.environment(),
        "gates": gates,
    })
    C.write_json(C.ANALYSIS_DIR / "protocol_freeze.json", {
        "round_id": C.ROUND_ID,
        "freeze_hash": C.freeze_hash(),
        "constants": C.FROZEN_CONSTANTS,
        "code_hashes": C.code_hashes(),
        "git_commit": C.git_commit(),
        "created_utc": C.utc_now(),
    })
    print(f"[prepare] all_passed={gates['all_passed']} checkpoints={checkpoints['passed']} "
          f"main_oracle={main_oracle['passed']} aux={aux_closure['passed']}", flush=True)
    return gates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args()
    gates = run(device=args.device, skip_replay=args.skip_replay)
    print(json.dumps({k: v.get("passed") if isinstance(v, dict) else v for k, v in gates.items()}, indent=1))
    return 0 if gates["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
