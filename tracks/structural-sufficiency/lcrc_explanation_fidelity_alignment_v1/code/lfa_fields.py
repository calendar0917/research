"""Stage 1 — frozen discovery fields for both model families and both tasks.

One process per model family (``res-gpu0`` = LCRC, ``res-gpu1`` = GINE).  Each
process writes one streaming JSONL per (task, scope): the all-ones gate
gradient, the LCRC per-atom local output, the base prediction and the molecule
size.  No candidate, label, pair, edit region or oracle contribution is read
here; the shared candidate pool is built from these fields on CPU afterwards.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import lfa_common as C
import lfa_data as D
import lfa_models as M

FAMILIES = ("lcrc", "gine")
TASKS = ("wc", "lipo")
MODEL_NAME = {
    ("lcrc", "wc"): "L_WC_LOGP",
    ("lcrc", "lipo"): "L_lipophilicity",
    ("gine", "wc"): "T_WC_LOGP",
    ("gine", "lipo"): "G0_lipophilicity",
}


def field_path(out_dir: Path, family: str, task: str, scope: str) -> Path:
    return out_dir / f"{family}_{task}_{scope}.jsonl"


def run(*, family: str, out_dir: Path | str, device: str = "cuda",
        scopes: Sequence[str] = (C.SCOPE_MAIN, C.SCOPE_AUX),
        max_molecules: int | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    C.ensure_dir(out_dir)
    if family not in FAMILIES:
        raise KeyError(family)
    b = D.bundle()
    t_start = time.time()
    reports: dict[str, Any] = {}
    for scope in scopes:
        rows = D.field_scope(scope)
        if max_molecules is not None:
            rows = rows[: int(max_molecules)]
        for task in TASKS:
            name = MODEL_NAME[(family, task)]
            b_task = D.oracle_bundle() if task == "wc" else b
            model, payload = M.load_model(name, b, device=device)
            before = M.parameter_hash(model)
            path = field_path(out_dir, family, task, scope)
            seen, handle = C.open_resumable_jsonl(path, ("split", "cache_row"))
            n_new = n_skip = n_err = 0
            occ_cache: dict[str, Any] = {}
            t0 = time.time()
            try:
                for rec in rows:
                    key = (str(rec["split"]), int(rec["cache_row"]))
                    if key in seen:
                        n_skip += 1
                        continue
                    split = key[0]
                    try:
                        if family == "lcrc":
                            if split not in occ_cache:
                                occ_cache[split] = D.split_occ(split)
                            mods = C.source_modules()
                            sm = mods["as_gate"].SingleMolecule(b.caches[split], key[1], b.spec)
                            gp = M.lcrc_gradient(
                                model, b.caches[split], occ_cache[split], key[1], b_task
                            )
                            if int(gp["grad_transformed"].size) != sm.n_bonds:
                                raise RuntimeError(
                                    f"grad size {gp['grad_transformed'].size} != bonds {sm.n_bonds}"
                                )
                            contrib = gp["contrib_atom_std"]
                        else:
                            mods = C.source_modules()
                            sm = mods["as_gate"].SingleMolecule(b.caches[split], key[1], b.spec)
                            batch = M.gine_single_batch(b, split, key[1], device=device)
                            gp = M.gine_gradient(model, batch, b_task)
                            if int(gp["grad_transformed"].size) != sm.n_bonds:
                                raise RuntimeError(
                                    f"grad size {gp['grad_transformed'].size} != bonds {sm.n_bonds}"
                                )
                            contrib = None
                        rec_out = {
                            "scope": scope,
                            "task": task,
                            "family": family,
                            "split": split,
                            "cache_row": int(key[1]),
                            "row_id": int(rec["row_id"]),
                            "n_atoms": int(sm.n_atoms),
                            "n_bonds": int(sm.n_bonds),
                            "pred_raw": float(gp["pred_raw"]),
                            "pred_std": float(gp["pred_std"]),
                            "grad": [float(v) for v in gp["grad_transformed"]],
                            "contrib": None if contrib is None else [float(v) for v in contrib],
                        }
                        handle.write(json.dumps(rec_out, sort_keys=True) + "\n")
                        handle.flush()
                        n_new += 1
                    except Exception as exc:  # noqa: BLE001 - one molecule must not kill the stage
                        n_err += 1
                        print(f"[fields:{family}/{task}/{scope}] ERROR {key}: {type(exc).__name__}: {exc}",
                              flush=True)
            finally:
                handle.close()
            after = M.parameter_hash(model)
            report = {
                "family": family,
                "task": task,
                "scope": scope,
                "model": name,
                "checkpoint_epoch": payload.get("epoch"),
                "path": C.relpath(path),
                "sha256": C.sha256_file(path) if path.is_file() else None,
                "n_scope_rows": len(rows),
                "n_new": n_new,
                "n_skipped": n_skip,
                "n_error": n_err,
                "runtime_s": float(time.time() - t0),
                "parameter_hash_before": before,
                "parameter_hash_after": after,
                "parameters_unchanged": bool(before == after),
                "device": device,
            }
            if n_err or before != after:
                raise RuntimeError(f"field stage failed for {family}/{task}/{scope}: {report}")
            reports[f"{family}_{task}_{scope}"] = report
            print(f"[fields:{family}/{task}/{scope}] new={n_new} skip={n_skip} "
                  f"t={report['runtime_s']:.1f}s sha={report['sha256'][:12]}", flush=True)
            del model
    out = {
        "stage": "fields",
        "family": family,
        "device": device,
        "runtime_s": float(time.time() - t_start),
        "reports": reports,
        "code_hashes": C.code_hashes(),
        "freeze_hash": C.freeze_hash(),
        "environment": C.environment(),
        "created_utc": C.utc_now(),
    }
    C.write_json(out_dir / f"fields_manifest_{family}.json", out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--out-dir", default=str(C.FIELDS_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scopes", default="main,aux")
    parser.add_argument("--max-molecules", type=int, default=None)
    args = parser.parse_args()
    out = run(
        family=args.family,
        out_dir=args.out_dir,
        device=args.device,
        scopes=tuple(s for s in args.scopes.split(",") if s),
        max_molecules=args.max_molecules,
    )
    print(json.dumps({k: v for k, v in out.items() if k != "environment"}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
