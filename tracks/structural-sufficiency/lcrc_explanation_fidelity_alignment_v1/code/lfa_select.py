"""Stage 3 — MASK50 selection and finite-dose fidelity for one model family.

One process per family (``res-gpu0`` = LCRC: A, B, C, D; ``res-gpu1`` = GINE:
E, F, D).  Every candidate comes from the shared pool file written by the CPU
stage; this worker never generates a candidate.

Per molecule:

1. one batched forward over the shared pool with ``m_S = 0.5`` on each
   candidate (base + ``|y(1) - y(0.5)|``) -> the MASK50 selector's Top-1 and the
   MASK50 Top-1 of its *own* candidate set (the hit audit);
2. one batched forward with the real fidelity dose ``m_S = 0`` (and the
   half-dose ``m_S = 0.5``) for every method's Top-1 plus all matched random
   controls of this family.

Only the auxiliary scope can skip step 2 (``--no-fidelity``): its report is a
structural localisation diagnostic, not a fidelity measurement.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import lfa_common as C
import lfa_data as D
import lfa_models as M
import lfa_search as S

FAMILY_SELECTORS = {"lcrc": C.MODEL_SELECTORS["LCRC"], "gine": C.MODEL_SELECTORS["GINE"]}
MASK_SELECTOR = {"lcrc": "C_LCRC_MASK50", "gine": "F_GINE_MASK50"}


def _task_of(scope: str) -> str:
    return "wc" if scope == C.SCOPE_MAIN else "lipo"


def _model_of(family: str, scope: str) -> str:
    task = _task_of(scope)
    return C.CHECKPOINT_NAMES[f"{family.upper()}_{task.upper()}"]


def _one_hot_masks(n_bonds: int, bond_sets: Sequence[Sequence[int]], value: float) -> list[np.ndarray]:
    out = []
    for bonds in bond_sets:
        m = np.ones(int(n_bonds), dtype=np.float64)
        ids = np.asarray([int(b) for b in bonds], dtype=np.int64)
        if ids.size:
            m[ids] = float(value)
        out.append(m)
    return out


def _chunks(n_graphs: int, n_bonds: int, n_atoms: int) -> list[list[int]]:
    max_by_bonds = max(1, C.CHUNK_BONDS // max(1, n_bonds))
    max_by_atoms = max(1, C.CHUNK_ATOMS // max(1, n_atoms))
    size = max(1, min(max_by_bonds, max_by_atoms))
    return [list(range(i, min(n_graphs, i + size))) for i in range(0, n_graphs, size)]


def eval_masks(family: str, model, *, cache, row: int, occ, b, split: str,
               masks: Sequence[np.ndarray], n_bonds: int, n_atoms: int,
               sigma: float, mu: float, device) -> dict[str, np.ndarray]:
    pred = np.empty(len(masks), dtype=np.float64)
    local = np.empty(len(masks), dtype=np.float64)
    rho = np.empty(len(masks), dtype=np.float64)
    for idxs in _chunks(len(masks), n_bonds, n_atoms):
        pick = [masks[i] for i in idxs]
        if family == "lcrc":
            gb = M.lcrc_batch(cache, row, occ, len(pick))
            Mt = M.gate_matrix(gb, pick, device=device)
            parts = M.lcrc_forward(model.arm, gb, Mt)
            pred[idxs] = parts["pred_std"] * sigma + mu
            local[idxs] = parts["local_sum"] * sigma
            rho[idxs] = parts["rho"] * sigma
        else:
            batch = M.gine_batch_replicated(b, split, row, len(pick), device=device)
            flat = torch.as_tensor(M.masks_to_flat(pick), dtype=torch.float32, device=device)
            parts = M.gine_forward(model, batch, flat, len(pick))
            pred[idxs] = parts["pred_std"] * sigma + mu
            local[idxs] = np.nan
            rho[idxs] = np.nan
    return {"pred_raw": pred, "local_sum_raw": local, "rho_raw": rho}


def process_molecule(pool_rec: Mapping[str, Any], *, family: str, model,
                     b, b_task, cache, occ, device, with_fidelity: bool) -> dict[str, Any]:
    mods = C.source_modules()
    split, cache_row = str(pool_rec["split"]), int(pool_rec["cache_row"])
    mol = mods["as_gate"].SingleMolecule(cache, cache_row, b.spec)
    n_bonds, n_atoms = int(mol.n_bonds), int(mol.n_atoms)
    if n_bonds != int(pool_rec["n_bonds"]) or n_atoms != int(pool_rec["n_atoms"]):
        raise RuntimeError(f"{split}/{cache_row}: pool record size disagrees with the molecule")
    sigma, mu = float(b_task.scaler["std"]), float(b_task.scaler["mean"])
    pool = [tuple(int(x) for x in bonds) for bonds in pool_rec["pool"]]
    mask_sel = MASK_SELECTOR[family]
    t0 = time.time()

    # -- forward 1: base + every pool candidate at m_S = 0.5 --------------
    ones = np.ones(n_bonds, dtype=np.float64)
    scan_masks = [ones] + _one_hot_masks(n_bonds, pool, C.DOSE_GATE)
    scan = eval_masks(family, model, cache=cache, row=cache_row, occ=occ, b=b, split=split,
                      masks=scan_masks, n_bonds=n_bonds, n_atoms=n_atoms,
                      sigma=sigma, mu=mu, device=device)
    base = float(scan["pred_raw"][0])
    expected_base = float(pool_rec["pred_fields"][family])
    base_err = abs(base - expected_base)
    base_tol = float(C.trace_tolerance(expected_base))
    if base_err > base_tol:
        raise RuntimeError(f"{split}/{cache_row}: base pred {base} != field {expected_base} "
                           f"(err {base_err:.3e} > {base_tol:.3e})")
    mask50_scores = np.abs(base - scan["pred_raw"][1:]).astype(np.float64)
    top_mask = S.select_top1(pool, mask50_scores.tolist())
    own = list(pool_rec["own_pools"][mask_sel])
    own_best = S.select_top1([pool[i] for i in own], [float(mask50_scores[i]) for i in own]) if own else None
    own_index = None if own_best is None else int(own[own_best])
    mask_selection = None if top_mask is None else {
        "pool_index": int(top_mask),
        "bonds": [int(v) for v in pool[top_mask]],
        "k": len(pool[top_mask]),
        "score": float(mask50_scores[top_mask]),
        "mask50_rank0": True,
    }
    mask_own = None if own_index is None else {
        "pool_index": own_index,
        "bonds": [int(v) for v in pool[own_index]],
        "score": float(mask50_scores[own_index]),
    }
    mask_hit = None if (top_mask is None or own_index is None) else bool(top_mask == own_index)
    top_mask_s2 = None if top_mask is None else S.s2_counts(mol, pool[top_mask])

    # -- controls for the MASK50 selector ---------------------------------
    mask_controls = {"status": "NO_CANDIDATE", "controls": [], "n_random": 0}
    if mask_selection is not None:
        mask_controls = S.matched_controls(
            mol, mask_selection["bonds"],
            seed=S.control_seed(split, int(pool_rec["row_id"]), mask_sel),
        )

    # -- forward 2: real fidelity doses -----------------------------------
    methods: dict[str, Any] = {}
    for sel in FAMILY_SELECTORS[family]:
        if sel == mask_sel:
            selection = mask_selection
            score = None if mask_selection is None else float(mask_selection["score"])
            controls = mask_controls
            s2 = top_mask_s2
        else:
            selection = pool_rec["selections"][sel]
            score = None if selection is None else float(selection["score"])
            controls = pool_rec["controls"][sel]
            s2 = pool_rec["n_s2"][sel]
        if selection is None:
            methods[sel] = {"status": "NO_CANDIDATE"}
            continue
        entry: dict[str, Any] = {
            "status": "OK",
            "bonds": [int(v) for v in selection["bonds"]],
            "k": int(len(selection["bonds"])),
            "n_atoms": int(len(mol.bond_atoms(list(selection["bonds"])))),
            "score": score,
            "score_source": "mask50" if sel == mask_sel else "pool",
            "mask50_score": float(mask50_scores[selection["pool_index"]]),
            "n_s2": s2,
            "controls_status": str(controls.get("status")),
            "n_controls": int(len(controls.get("controls", []))),
            "controls_attempts": int(controls.get("attempts", 0)),
            "control_bonds": [[int(x) for x in rec["bonds"]] for rec in controls.get("controls", [])],
        }
        methods[sel] = entry

    if with_fidelity:
        variants: list[tuple[str, str, int | None]] = []   # (selector, kind, control index)
        masks: list[np.ndarray] = [ones]
        for sel in FAMILY_SELECTORS[family]:
            if methods.get(sel, {}).get("status") != "OK":
                continue
            bonds = methods[sel]["bonds"]
            variants.append((sel, "top_zero", None))
            masks.extend(_one_hot_masks(n_bonds, [bonds], 0.0))
            variants.append((sel, "top_half", None))
            masks.extend(_one_hot_masks(n_bonds, [bonds], C.DOSE_GATE))
            for j, ctrl in enumerate(methods[sel]["control_bonds"]):
                variants.append((sel, "control", j))
                masks.extend(_one_hot_masks(n_bonds, [ctrl], 0.0))
        fx = eval_masks(family, model, cache=cache, row=cache_row, occ=occ, b=b, split=split,
                        masks=masks, n_bonds=n_bonds, n_atoms=n_atoms,
                        sigma=sigma, mu=mu, device=device)
        if abs(float(fx["pred_raw"][0]) - base) > base_tol:
            raise RuntimeError(f"{split}/{cache_row}: fidelity base pred disagrees with scan base")
        cursor = 1
        per_sel: dict[str, dict[str, Any]] = {sel: {"zero": [], "half": [], "controls": []}
                                             for sel in FAMILY_SELECTORS[family]}
        for sel, kind, _j in variants:
            rec = {"pred_raw": float(fx["pred_raw"][cursor]),
                   "local_sum_raw": float(fx["local_sum_raw"][cursor]),
                   "rho_raw": float(fx["rho_raw"][cursor])}
            per_sel[sel][{"top_zero": "zero", "top_half": "half", "control": "controls"}[kind]].append(rec)
            cursor += 1
        target_std = float(b_task.scaler["std"])
        for sel in FAMILY_SELECTORS[family]:
            entry = methods.get(sel, {})
            if entry.get("status") != "OK":
                continue
            zero = per_sel[sel]["zero"][0]
            half = per_sel[sel]["half"][0]
            delta0 = base - zero["pred_raw"]
            delta_half = base - half["pred_raw"]
            abs0 = abs(delta0)
            abs_half = abs(delta_half)
            ctrl_abs = np.abs(np.asarray([base - rec["pred_raw"] for rec in per_sel[sel]["controls"]],
                                         dtype=np.float64))
            med_ctrl = float(np.median(ctrl_abs)) if ctrl_abs.size else None
            entry.update({
                "pred_zero": zero["pred_raw"],
                "pred_half": half["pred_raw"],
                "delta0": float(delta0),
                "delta_half": float(delta_half),
                "abs_delta0": float(abs0),
                "abs_delta_half": float(abs_half),
                "delta0_std": float(delta0 / target_std),
                "abs_delta0_std": float(abs0 / target_std),
                "abs_delta_half_std": float(abs_half / target_std),
                "delta0_sign": int(np.sign(delta0)),
                "delta_half_sign": int(np.sign(delta_half)),
                "sign_consistent": bool(np.sign(delta0) == np.sign(delta_half)),
                "half_le_full": bool(abs_half <= abs0 + float(C.trace_tolerance(abs0))),
                "half_over_full": (float(abs_half / abs0) if abs0 > 0 else None),
                "local_sum_raw_1": float(fx["local_sum_raw"][0]) if family == "lcrc" else None,
                "local_sum_raw_0": float(zero["local_sum_raw"]) if family == "lcrc" else None,
                "rho_raw_1": float(fx["rho_raw"][0]) if family == "lcrc" else None,
                "rho_raw_0": float(zero["rho_raw"]) if family == "lcrc" else None,
                "delta_local": float(fx["local_sum_raw"][0] - zero["local_sum_raw"]) if family == "lcrc" else None,
                "delta_rho": float(fx["rho_raw"][0] - zero["rho_raw"]) if family == "lcrc" else None,
                "closure_error": (float(abs(delta0 - (fx["local_sum_raw"][0] - zero["local_sum_raw"])
                                            - (fx["rho_raw"][0] - zero["rho_raw"])))
                                  if family == "lcrc" else None),
                "control_abs_delta0": [float(v) for v in ctrl_abs],
                "random_median_abs_delta0": med_ctrl,
                "random_median_abs_delta0_std": None if med_ctrl is None else float(med_ctrl / target_std),
                "fidelity_advantage": None if med_ctrl is None else float(abs0 - med_ctrl),
                "fidelity_advantage_std": None if med_ctrl is None else float((abs0 - med_ctrl) / target_std),
                "target_std": target_std,
                "missing_controls": bool(ctrl_abs.size == 0),
                "few_controls": bool(ctrl_abs.size < C.MIN_RANDOM),
            })
    else:
        for sel in FAMILY_SELECTORS[family]:
            entry = methods.get(sel, {})
            if entry.get("status") == "OK":
                entry["fidelity"] = "NOT_EVALUATED_AUX_SCOPE"

    return {
        "scope": pool_rec["scope"],
        "task": pool_rec["task"],
        "family": family,
        "split": split,
        "cache_row": cache_row,
        "row_id": int(pool_rec["row_id"]),
        "n_atoms": n_atoms,
        "n_bonds": n_bonds,
        "base_pred_raw": base,
        "base_pred_expected": expected_base,
        "base_pred_error": base_err,
        "base_pred_tolerance": base_tol,
        "mask50_selector": mask_sel,
        "mask50_selection": mask_selection,
        "mask50_own_top1": mask_own,
        "mask50_hit_own_top1": mask_hit,
        "mask50_n_scanned": len(pool),
        "mask50_control_status": mask_controls.get("status"),
        "mask50_n_controls": len(mask_controls.get("controls", [])),
        "mask50_control_bonds": [[int(x) for x in rec["bonds"]] for rec in mask_controls.get("controls", [])],
        "n_s2": {**{k: v for k, v in (pool_rec.get("n_s2") or {}).items() if v is not None},
                 **({} if top_mask_s2 is None else {mask_sel: top_mask_s2})},
        "methods": methods,
        "with_fidelity": bool(with_fidelity),
        "runtime_s": float(time.time() - t0),
    }


def run(*, family: str, pool_path: Path | str, out_dir: Path | str, device: str = "cuda",
        with_fidelity: bool = True, max_molecules: int | None = None,
        pool_manifest: Path | str | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    C.ensure_dir(out_dir)
    pool_path = Path(pool_path)
    manifest_path = Path(pool_manifest) if pool_manifest is not None else (
        Path(out_dir).parent / "pool" / "pool_manifest.json")
    manifest = C.read_json(manifest_path)
    scope = "main" if "pool_main" in pool_path.name else "aux"
    expected_sha = manifest["reports"][scope]["sha256"]
    actual_sha = C.sha256_file(pool_path)
    if actual_sha != expected_sha:
        raise RuntimeError(f"shared pool identity: {actual_sha} != manifest {expected_sha}")
    b = D.bundle()
    b_task = D.oracle_bundle() if scope == C.SCOPE_MAIN else b
    name = _model_of(family, scope)
    model, payload = M.load_model(name, b, device=device)
    before = M.parameter_hash(model)
    path = out_dir / f"select_{family}_{scope}.jsonl"
    seen, handle = C.open_resumable_jsonl(path, ("split", "cache_row"))
    n_new = n_skip = n_err = 0
    occ_cache: dict[str, Any] = {}
    t0 = time.time()
    try:
        with pool_path.open("r", encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = (str(rec["split"]), int(rec["cache_row"]))
                if key in seen:
                    n_skip += 1
                    continue
                if max_molecules is not None and n_new >= int(max_molecules):
                    break
                split = key[0]
                if split not in occ_cache:
                    occ_cache[split] = D.split_occ(split)
                try:
                    out = process_molecule(
                        rec, family=family, model=model, b=b, b_task=b_task,
                        cache=b.caches[split], occ=occ_cache[split], device=device,
                        with_fidelity=with_fidelity,
                    )
                    out["status"] = "OK"
                except Exception as exc:  # noqa: BLE001 - one molecule must not kill the stage
                    n_err += 1
                    out = {"scope": scope, "family": family, "split": split, "cache_row": key[1],
                           "row_id": int(rec["row_id"]), "status": "ERROR",
                           "error": f"{type(exc).__name__}: {exc}"}
                    print(f"[select:{family}/{scope}] ERROR {key}: {out['error']}", flush=True)
                handle.write(json.dumps(out, sort_keys=True) + "\n")
                handle.flush()
                n_new += 1
                if n_new % 50 == 0:
                    print(f"[select:{family}/{scope}] {n_new} t={time.time()-t0:.1f}s", flush=True)
    finally:
        handle.close()
    after = M.parameter_hash(model)
    report = {
        "stage": "select",
        "family": family,
        "scope": scope,
        "model": name,
        "checkpoint_epoch": payload.get("epoch"),
        "pool_path": C.relpath(pool_path),
        "pool_sha256": actual_sha,
        "path": C.relpath(path),
        "sha256": C.sha256_file(path),
        "n_new": n_new,
        "n_skipped": n_skip,
        "n_error": n_err,
        "with_fidelity": bool(with_fidelity),
        "runtime_s": float(time.time() - t0),
        "parameter_hash_before": before,
        "parameter_hash_after": after,
        "parameters_unchanged": bool(before == after),
        "device": device,
        "environment": C.environment(),
        "code_hashes": C.code_hashes(),
        "freeze_hash": C.freeze_hash(),
        "created_utc": C.utc_now(),
    }
    if n_err or before != after:
        report["status"] = "FAILED"
        C.write_json(out_dir / f"select_manifest_{family}_{scope}.json", report)
        raise RuntimeError(f"select stage failed for {family}/{scope}: n_error={n_err}")
    report["status"] = "OK"
    C.write_json(out_dir / f"select_manifest_{family}_{scope}.json", report)
    print(f"[select:{family}/{scope}] done new={n_new} err={n_err} t={report['runtime_s']:.1f}s "
          f"sha={report['sha256'][:12]}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=("lcrc", "gine"))
    parser.add_argument("--scope", required=True, choices=(C.SCOPE_MAIN, C.SCOPE_AUX))
    parser.add_argument("--pool", default=None)
    parser.add_argument("--pool-manifest", default=None)
    parser.add_argument("--out-dir", default=str(C.SELECT_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-fidelity", action="store_true")
    parser.add_argument("--max-molecules", type=int, default=None)
    args = parser.parse_args()
    pool = args.pool or str(C.POOL_DIR / f"pool_{args.scope}.jsonl")
    report = run(
        family=args.family, pool_path=pool, out_dir=args.out_dir, device=args.device,
        with_fidelity=not args.no_fidelity, max_molecules=args.max_molecules,
        pool_manifest=args.pool_manifest,
    )
    print(json.dumps({k: v for k, v in report.items() if k not in ("environment", "code_hashes")},
                     indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
