"""Stage 2 — build the single shared candidate pool per molecule (CPU).

All methods read the pool file written here; the GPU stages never generate a
candidate of their own.  The stage is label-, pair-, edit- and oracle-blind: it
consumes only the frozen molecule graph and the stage-1 gradient / local-output
fields.

For every molecule it writes

* the generator beams (indices into the pool) and the union pool,
* the model-free selector scores (A, B, D, E) on every pool candidate,
* the model-free selections and their matched random controls,
* the "own candidate set" of every selector, so the shared-pool Top-1 can be
  compared with the Top-1 that selector would have produced alone.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import lfa_common as C
import lfa_data as D
import lfa_search as S

MODEL_FREE_SELECTORS = ("A_LCRC_GRAD", "B_LCRC_LOCAL", "D_TOPOLOGY", "E_GINE_GRAD")


def load_fields(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[(str(rec["split"]), int(rec["cache_row"]))] = rec
    return out


def build_molecule(rec: Mapping[str, Any], fields_lcrc: Mapping[str, Any],
                   fields_gine: Mapping[str, Any], *, scope: str, task: str) -> dict[str, Any]:
    bundle = D.bundle()
    mods = C.source_modules()
    split, cache_row = str(rec["split"]), int(rec["cache_row"])
    mol = mods["as_gate"].SingleMolecule(bundle.caches[split], cache_row, bundle.spec)
    if mol.n_bonds != int(fields_lcrc["n_bonds"]) or mol.n_bonds != int(fields_gine["n_bonds"]):
        raise RuntimeError(f"{split}/{cache_row}: field bond counts disagree with the molecule")
    if mol.n_atoms != int(fields_lcrc["n_atoms"]) or mol.n_atoms != int(fields_gine["n_atoms"]):
        raise RuntimeError(f"{split}/{cache_row}: field atom counts disagree with the molecule")
    fields = {
        "lcrc_grad": np.asarray(fields_lcrc["grad"], dtype=np.float64),
        "lcrc_local": np.asarray(fields_lcrc["contrib"], dtype=np.float64),
        "gine_grad": np.asarray(fields_gine["grad"], dtype=np.float64),
    }
    families = S.generate_families(mol, fields)
    pool = S.shared_pool(families)
    index = {bonds: i for i, bonds in enumerate(pool)}
    fn_by_name = S.score_functions(mol, fields)

    generator_members: dict[str, dict[str, list[int]]] = {}
    for name, fam in families.items():
        by_k: dict[str, list[int]] = {}
        for bonds in fam["scores"]:
            by_k.setdefault(str(len(bonds)), []).append(index[bonds])
        generator_members[name] = {k: sorted(v) for k, v in sorted(by_k.items(), key=lambda kv: int(kv[0]))}

    selector_scores: dict[str, list[float]] = {}
    for sel in MODEL_FREE_SELECTORS:
        name = {
            "A_LCRC_GRAD": "LCRC_GRAD", "B_LCRC_LOCAL": "LCRC_LOCAL",
            "D_TOPOLOGY": "TOPO_S2", "E_GINE_GRAD": "GINE_GRAD",
        }[sel]
        selector_scores[sel] = [float(fn_by_name[name](bonds)) for bonds in pool]

    selections: dict[str, Any] = {}
    own_pools: dict[str, list[int]] = {}
    own_top1: dict[str, Any] = {}
    hits: dict[str, Any] = {}
    s2_for: dict[str, Any] = {}
    for sel in C.SELECTORS:
        own = sorted({i for gen in C.OWN_POOL_GENERATORS[sel]
                      for k, idxs in generator_members.get(gen, {}).items() for i in idxs})
        own_pools[sel] = own
        if sel not in MODEL_FREE_SELECTORS:
            selections[sel] = None
            own_top1[sel] = None
            hits[sel] = None
            s2_for[sel] = None
            continue
        scores = selector_scores[sel]
        best = S.select_top1(pool, scores)
        own_best = S.select_top1([pool[i] for i in own], [scores[i] for i in own]) if own else None
        own_index = None if own_best is None else int(own[own_best])
        selections[sel] = None if best is None else {
            "pool_index": int(best),
            "bonds": [int(v) for v in pool[best]],
            "k": len(pool[best]),
            "score": float(scores[best]),
        }
        own_top1[sel] = None if own_index is None else {
            "pool_index": int(own_index),
            "bonds": [int(v) for v in pool[own_index]],
            "k": len(pool[own_index]),
            "score": float(scores[own_index]),
        }
        hits[sel] = None if (best is None or own_index is None) else bool(best == own_index)
        if best is not None:
            s2_for[sel] = S.s2_counts(mol, pool[best])

    controls: dict[str, Any] = {}
    for sel in MODEL_FREE_SELECTORS:
        top = selections[sel]
        if top is None:
            controls[sel] = {"status": "NO_CANDIDATE", "controls": [], "n_random": 0}
            continue
        controls[sel] = S.matched_controls(
            mol, top["bonds"], seed=S.control_seed(split, int(rec["row_id"]), sel)
        )

    pairs = list(itertools.combinations(C.GENERATORS, 2))
    overlap = {
        f"{a}|{b}": len(set(i for v in generator_members.get(a, {}).values() for i in v)
                        & set(i for v in generator_members.get(b, {}).values() for i in v))
        for a, b in pairs
    }
    sizes = {name: sum(len(v) for v in generator_members.get(name, {}).values()) for name in C.GENERATORS}
    return {
        "scope": scope,
        "task": task,
        "split": split,
        "cache_row": cache_row,
        "row_id": int(rec["row_id"]),
        "n_atoms": int(mol.n_atoms),
        "n_bonds": int(mol.n_bonds),
        "pred_fields": {"lcrc": float(fields_lcrc["pred_raw"]), "gine": float(fields_gine["pred_raw"])},
        "pool": [[int(v) for v in bonds] for bonds in pool],
        "n_pool": len(pool),
        "generator_members": generator_members,
        "generator_sizes": sizes,
        "generator_overlap": overlap,
        "generator_n_evaluated": {k: int(v["n_evaluated"]) for k, v in families.items()},
        "selector_scores": selector_scores,
        "selections": selections,
        "own_pools": own_pools,
        "own_top1": own_top1,
        "hit_own_top1": hits,
        "n_s2": s2_for,
        "controls": controls,
    }


def run(*, fields_dir: Path | str = C.FIELDS_DIR, out_dir: Path | str = C.POOL_DIR,
        scopes: Sequence[str] = (C.SCOPE_MAIN, C.SCOPE_AUX),
        max_molecules: int | None = None) -> dict[str, Any]:
    fields_dir, out_dir = Path(fields_dir), Path(out_dir)
    C.ensure_dir(out_dir)
    t_all = time.time()
    reports: dict[str, Any] = {}
    for scope in scopes:
        task = "wc" if scope == C.SCOPE_MAIN else "lipo"
        fields_lcrc = load_fields(fields_dir / f"lcrc_{task}_{scope}.jsonl")
        fields_gine = load_fields(fields_dir / f"gine_{task}_{scope}.jsonl")
        if set(fields_lcrc) != set(fields_gine):
            missing = sorted(set(fields_lcrc) ^ set(fields_gine))[:5]
            raise RuntimeError(f"{scope}: LCRC/GINE field scopes differ, e.g. {missing}")
        rows = D.field_scope(scope)
        if max_molecules is not None:
            rows = rows[: int(max_molecules)]
        rows = [r for r in rows if (str(r["split"]), int(r["cache_row"])) in fields_lcrc]
        path = out_dir / f"pool_{scope}.jsonl"
        seen, handle = C.open_resumable_jsonl(path, ("split", "cache_row"))
        n_new = n_skip = 0
        t0 = time.time()
        per_k: dict[str, int] = {}
        sizes: list[int] = []
        try:
            for rec in rows:
                key = (str(rec["split"]), int(rec["cache_row"]))
                if key in seen:
                    n_skip += 1
                    continue
                out = build_molecule(
                    rec, fields_lcrc[key], fields_gine[key], scope=scope, task=task
                )
                handle.write(json.dumps(out, sort_keys=True) + "\n")
                handle.flush()
                n_new += 1
                sizes.append(out["n_pool"])
                for bonds in out["pool"]:
                    per_k[str(len(bonds))] = per_k.get(str(len(bonds)), 0) + 1
                if n_new % 100 == 0:
                    print(f"[pool:{scope}] {n_new}/{len(rows)} t={time.time()-t0:.1f}s", flush=True)
        finally:
            handle.close()
        report = {
            "scope": scope,
            "task": task,
            "path": C.relpath(path),
            "sha256": C.sha256_file(path),
            "n_rows": len(rows),
            "n_new": n_new,
            "n_skipped": n_skip,
            "runtime_s": float(time.time() - t0),
            "field_sha256": {
                "lcrc": C.sha256_file(fields_dir / f"lcrc_{task}_{scope}.jsonl"),
                "gine": C.sha256_file(fields_dir / f"gine_{task}_{scope}.jsonl"),
            },
            "pool_size": {
                "n": len(sizes),
                "mean": float(np.mean(sizes)) if sizes else None,
                "median": float(np.median(sizes)) if sizes else None,
                "min": int(np.min(sizes)) if sizes else None,
                "max": int(np.max(sizes)) if sizes else None,
            },
            "pool_candidates_by_k": per_k,
        }
        reports[scope] = report
        print(f"[pool:{scope}] rows={len(rows)} new={n_new} sha={report['sha256'][:12]}", flush=True)
    out = {
        "stage": "pool",
        "reports": reports,
        "code_hashes": C.code_hashes(),
        "freeze_hash": C.freeze_hash(),
        "constants": C.FROZEN_CONSTANTS,
        "runtime_s": float(time.time() - t_all),
        "created_utc": C.utc_now(),
    }
    C.write_json(out_dir / "pool_manifest.json", out)
    print(f"[pool] freeze={out['freeze_hash'][:12]} runtime={out['runtime_s']:.1f}s", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields-dir", default=str(C.FIELDS_DIR))
    parser.add_argument("--out-dir", default=str(C.POOL_DIR))
    parser.add_argument("--scopes", default="main,aux")
    parser.add_argument("--max-molecules", type=int, default=None)
    args = parser.parse_args()
    out = run(
        fields_dir=args.fields_dir, out_dir=args.out_dir,
        scopes=tuple(s for s in args.scopes.split(",") if s),
        max_molecules=args.max_molecules,
    )
    print(json.dumps(out["reports"], indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
