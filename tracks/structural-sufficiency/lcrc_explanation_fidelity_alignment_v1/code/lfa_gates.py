"""P1 implementation gates for the fidelity--alignment round.

Every gate either reproduces a frozen upstream behaviour (all-ones equivalence,
finite differences, renumbering, WC_LOGP closure) or verifies a property this
round newly relies on (batched GINE forward, the executed 0/0.5 dose path,
label-blind pool construction, shared-pool identity).

A failed gate stops the affected method; it is never bypassed.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import lfa_common as C
import lfa_data as D
import lfa_models as M
import lfa_search as S


def _mols(task_split: str = "valid", n: int = 3) -> list[int]:
    rows = [r for r in D.main_scope() if r["split"] == task_split]
    return [int(r["cache_row"]) for r in rows[: int(n)]]


def _bundles():
    b = D.bundle()
    return b, D.oracle_bundle()


def _pass(ok: bool, **payload: Any) -> dict[str, Any]:
    return {"passed": bool(ok), **payload}


# ---------------------------------------------------------------------------
# 1. all-ones equivalence, both families, both tasks
# ---------------------------------------------------------------------------


def gate_all_ones(*, device="cpu") -> dict[str, Any]:
    import xr_models

    b, b_wc = _bundles()
    out: dict[str, Any] = {"records": [], "passed": True}
    for task, b_task in (("wc", b_wc), ("lipo", b)):
        models = {"lcrc": M.load_model(C.CHECKPOINT_NAMES[f"LCRC_{task.upper()}"], b, device=device)[0],
                  "gine": M.load_model(C.CHECKPOINT_NAMES[f"GINE_{task.upper()}"], b, device=device)[0]}
        for split in ("valid", "test"):
            occ = D.split_occ(split)
            for row in _mols(split, 2):
                cache = b.caches[split]
                ref = xr_models.lcrc_all_ones_check(models["lcrc"], cache, occ, row, b_task)
                gb = M.lcrc_batch(cache, row, occ, 3)
                mt = M.gate_matrix(gb, [np.ones(int(gb.bond_counts[0]))] * 3, device=device)
                batched = M.lcrc_forward(models["lcrc"].arm, gb, mt)["pred_std"] * float(b_task.scaler["std"]) + float(b_task.scaler["mean"])
                batched_ok = bool(np.all(np.abs(batched - batched[0]) <= 1e-4))
                gbatch = M.gine_single_batch(b, split, row, device=device)
                gref = xr_models.gine_all_ones_check(models["gine"], gbatch, b_task)
                n = 3
                gbatched = M.gine_batch_replicated(b, split, row, n, device=device)
                nb = int(gbatched["edge_cat"].shape[0]) // 2 // n
                gflat = torch.ones(n * nb, dtype=torch.float32, device=device)
                gpred = M.gine_forward(models["gine"], gbatched, gflat, n)["pred_std"] * float(b_task.scaler["std"]) + float(b_task.scaler["mean"])
                gbatched_ok = bool(np.all(np.abs(gpred - gpred[0]) <= 1e-4))
                rec = {
                    "task": task, "split": split, "row": row,
                    "lcrc": ref, "lcrc_batched_consistent": batched_ok,
                    "gine": gref, "gine_batched_consistent": gbatched_ok,
                    "passed": bool(ref["passed"] and gref["passed"] and batched_ok and gbatched_ok),
                }
                out["records"].append(rec)
                out["passed"] = out["passed"] and rec["passed"]
    return out


# ---------------------------------------------------------------------------
# 2. new batched GINE forward == previously verified single-graph forward
# ---------------------------------------------------------------------------


def gate_gine_batched_reference(*, device="cpu", seed: int = C.CONTROL_SEED) -> dict[str, Any]:
    import xr_models

    b, b_wc = _bundles()
    model = M.load_model(C.CHECKPOINT_NAMES["GINE_WC"], b, device=device)[0]
    rng = np.random.default_rng([int(seed), 17])
    recs, ok = [], True
    for split in ("valid", "test"):
        for row in _mols(split, 2):
            batch = M.gine_single_batch(b, split, row, device=device)
            nb = int(batch["edge_cat"].shape[0]) // 2
            n = 3
            gates = [rng.uniform(0.0, 1.0, size=nb) for _ in range(n)]
            gb = M.gine_batch_replicated(b, split, row, n, device=device)
            flat = torch.as_tensor(np.concatenate(gates), dtype=torch.float32, device=device)
            got = M.gine_forward(model, gb, flat, n)["pred_std"]
            refs = []
            for g in gates:
                refs.append(float(xr_models.gine_gated_parts(model, batch, torch.as_tensor(g[None, :], dtype=torch.float32, device=device))["pred_std"].item()))
            err = float(np.max(np.abs(got - np.asarray(refs))))
            tol = 1e-5 + 1e-5 * float(np.max(np.abs(refs)))
            recs.append({"split": split, "row": row, "n_graphs": n, "max_abs_error_std": err,
                         "tolerance": tol, "passed": bool(err <= tol)})
            ok = ok and bool(err <= tol)
    return {"records": recs, "passed": bool(ok)}


# ---------------------------------------------------------------------------
# 3. finite differences (gate gradient vs central difference)
# ---------------------------------------------------------------------------


def gate_finite_difference(*, device="cpu") -> dict[str, Any]:
    import xr_models

    b, b_wc = _bundles()
    out = {"records": [], "passed": True}
    for task, b_task in (("wc", b_wc), ("lipo", b)):
        for family, name in (("lcrc", f"LCRC_{task.upper()}"), ("gine", f"GINE_{task.upper()}")):
            model = M.load_model(C.CHECKPOINT_NAMES[name], b, device=device)[0]
            split = "valid"
            occ = D.split_occ(split)
            rows = _mols(split, 2)
            for row in rows:
                nb = int(np.diff(np.asarray(b.caches[split]["bond_ptr"], dtype=np.int64))[row])
                bonds = sorted({0, nb // 2, nb - 1})[:3]
                if family == "lcrc":
                    recs = xr_models.lcrc_finite_difference(model, b.caches[split], occ, row, b_task, bonds)
                else:
                    batch = M.gine_single_batch(b, split, row, device=device)
                    recs = xr_models.gine_finite_difference(model, batch, b_task, bonds)
                ok = all(r["passed"] for r in recs)
                out["records"].append({"task": task, "family": family, "row": row, "checks": recs, "passed": ok})
                out["passed"] = out["passed"] and ok
    return out


# ---------------------------------------------------------------------------
# 4. atom + physical-bond renumbering
# ---------------------------------------------------------------------------


def gate_renumbering(*, device="cpu") -> dict[str, Any]:
    import xr_models

    b, b_wc = _bundles()
    out = {"records": [], "passed": True}
    occ = D.split_occ("valid")
    for task, b_task in (("wc", b_wc), ("lipo", b)):
        for family, name in (("lcrc", f"LCRC_{task.upper()}"), ("gine", f"GINE_{task.upper()}")):
            model = M.load_model(C.CHECKPOINT_NAMES[name], b, device=device)[0]
            rows = _mols("valid", 2)
            if family == "lcrc":
                rep = xr_models.lcrc_renumbering_check(model, b.caches["valid"], occ, rows, b_task)
            else:
                rep = xr_models.gine_renumbering_check(model, b, "valid", rows)
            rep_ok = bool(rep.get("passed", rep.get("pass")))
            out["records"].append({"task": task, "family": family, "report": rep, "passed": rep_ok})
            out["passed"] = out["passed"] and rep_ok
    return out


# ---------------------------------------------------------------------------
# 5. executed dose path (m=1 -> 0 and 0.5) closes with the gate gradient
# ---------------------------------------------------------------------------


def _path_masks(n_bonds: int, bonds: Sequence[int], grid: np.ndarray) -> list[np.ndarray]:
    ids = [int(b) for b in bonds]
    out = []
    for t in grid:
        m = np.ones(n_bonds, dtype=np.float64)
        m[ids] = 1.0 - float(t)
        out.append(m)
    return out


def dose_path_check(family: str, model, *, cache, row: int, occ, b, split: str,
                    bonds: Sequence[int], b_task, device, steps: int = C.DOSE_PATH_STEPS) -> dict[str, Any]:
    mods = C.source_modules()
    mol = mods["as_gate"].SingleMolecule(cache, row, b.spec)
    n_bonds, n_atoms = int(mol.n_bonds), int(mol.n_atoms)
    grid = np.linspace(0.0, 1.0, int(steps) + 1)
    masks = _path_masks(n_bonds, bonds, grid)
    sigma, mu = float(b_task.scaler["std"]), float(b_task.scaler["mean"])
    ids = [int(b) for b in bonds]
    if family == "lcrc":
        gb = M.lcrc_batch(cache, row, occ, len(masks))
        Mt = M.gate_matrix(gb, masks, device=device, requires_grad=True)
        parts = M.lcrc_forward_grad(model.arm, gb, Mt)
        pred_raw = parts["pred_std"] * sigma + mu
        grad = parts["grad"] * sigma
    else:
        batch = M.gine_batch_replicated(b, split, row, len(masks), device=device)
        flat = torch.as_tensor(M.masks_to_flat(masks), dtype=torch.float32, device=device)
        flat.requires_grad_(True)
        parts = M.gine_forward_grad(model, batch, flat, len(masks))
        pred_raw = parts["pred_std"] * sigma + mu
        nb = int(batch["edge_cat"].shape[0]) // 2 // len(masks)
        grad = parts["grad"].reshape(len(masks), nb) * sigma
    # dm/dt = -1 on the selected bonds; other bonds do not move.
    # int_0^1 dpred/dt dt = pred(1) - pred(0), so the reported integral is
    # flipped to the delta orientation pred(0) - pred(1).
    direction = -np.sum(grad[:, ids], axis=1)
    integral = float(-np.trapezoid(direction, grid))
    delta = float(pred_raw[0] - pred_raw[-1])
    err = abs(integral - delta)
    tol = 2e-3 + 2e-3 * max(1.0, abs(delta))
    mid = len(grid) // 2
    half_err = abs(float(pred_raw[mid]) - float(pred_raw[mid]))
    return {
        "family": family, "split": split, "row": int(row), "bonds": ids,
        "steps": int(steps), "pred_t0": float(pred_raw[0]), "pred_t1": float(pred_raw[-1]),
        "pred_t05": float(pred_raw[mid]), "path_delta": delta,
        "integrated_gradient": integral, "closure_abs_error": err, "tolerance": tol,
        "midpoint_mask_equals_half_dose": bool(abs(1.0 - grid[mid] - C.DOSE_GATE) < 1e-12),
        "passed": bool(err <= tol and abs(1.0 - grid[mid] - C.DOSE_GATE) < 1e-12),
    }


def gate_dose_path(*, device="cpu") -> dict[str, Any]:
    b, b_wc = _bundles()
    out = {"records": [], "passed": True}
    for task, b_task in (("wc", b_wc), ("lipo", b)):
        for family, name in (("lcrc", f"LCRC_{task.upper()}"), ("gine", f"GINE_{task.upper()}")):
            model = M.load_model(C.CHECKPOINT_NAMES[name], b, device=device)[0]
            split = "valid"
            occ = D.split_occ(split)
            for row in _mols(split, 2):
                mol = C.source_modules()["as_gate"].SingleMolecule(b.caches[split], row, b.spec)
                gp = (M.lcrc_gradient(model, b.caches[split], occ, row, b_task) if family == "lcrc"
                      else M.gine_gradient(model, M.gine_single_batch(b, split, row, device=device), b_task))
                g = np.abs(np.asarray(gp["grad_transformed"], dtype=np.float64))
                bonds = [int(i) for i in np.argsort(-g)[:2]]
                rec = dose_path_check(family, model, cache=b.caches[split], row=row, occ=occ, b=b,
                                      split=split, bonds=bonds, b_task=b_task, device=device)
                rec["task"] = task
                out["records"].append(rec)
                out["passed"] = out["passed"] and rec["passed"]
    return out


# ---------------------------------------------------------------------------
# 6. gate weight semantics (internal / boundary / unaffected)
# ---------------------------------------------------------------------------


def gate_weight_semantics() -> dict[str, Any]:
    b, _ = _bundles()
    split = "valid"
    occ = D.split_occ(split)
    recs, ok = [], True
    for row in _mols(split, 2):
        mol = C.source_modules()["as_gate"].SingleMolecule(b.caches[split], row, b.spec)
        nb = int(mol.n_bonds)
        cands = [tuple(sorted(mol.shared_endpoint_pairs()[0]))] if mol.shared_endpoint_pairs() else []
        if not cands:
            continue
        bonds = cands[0]
        gb = M.lcrc_batch(b.caches[split], row, occ, 2)
        m0 = np.ones(nb); m0[list(bonds)] = 0.0
        m5 = np.ones(nb); m5[list(bonds)] = C.DOSE_GATE
        Mt = M.gate_matrix(gb, [m0, m5])
        w_all = gb.s2_weight(Mt).detach().cpu().numpy()
        graph_of_path = gb.path_jump.detach().cpu().numpy()
        w0 = w_all[graph_of_path == 0]
        w5 = w_all[graph_of_path == 1]
        counts = S.s2_counts(mol, bonds)
        zero = int(np.sum(np.abs(w0) < 1e-12))
        half = int(np.sum(np.abs(w5 - 0.5) < 1e-12))
        one = int(np.sum(np.abs(w5 - 1.0) < 1e-12))
        expected = {
            "zero_at_full_dose": int(counts["n_s2_affected"]),
            "half_at_half_dose": int(counts["n_s2_boundary"]),
            "one_at_half_dose": int(len(w5) - counts["n_s2_affected"]),
        }
        got = {"zero_at_full_dose": zero, "half_at_half_dose": half, "one_at_half_dose": one}
        rec_ok = got == expected
        recs.append({"split": split, "row": row, "bonds": list(bonds), "n_path": int(len(w5)),
                     "expected": expected, "observed": got, "passed": bool(rec_ok)})
        ok = ok and rec_ok
    # GINE: every physical bond must appear exactly twice per graph copy.
    b2, _ = _bundles()
    batch = M.gine_batch_replicated(b2, split, _mols(split, 1)[0], 3, device="cpu")
    nb = int(batch["edge_cat"].shape[0]) // 2 // 3
    local = torch.arange(nb)
    per_graph = [local + g * nb for g in range(3)]
    edge_bond = torch.cat([torch.cat([x, x]) for x in per_graph], dim=0).numpy()
    cover_ok = bool(all(sorted(edge_bond[g * 2 * nb:(g + 1) * 2 * nb].tolist())
                        == sorted(list(range(g * nb, (g + 1) * nb)) * 2)
                        for g in range(3)))
    recs.append({"family": "gine", "bond_cover_ok": cover_ok, "passed": cover_ok})
    ok = ok and cover_ok
    return {"records": recs, "passed": bool(ok)}


# ---------------------------------------------------------------------------
# 7. label-blind pool construction
# ---------------------------------------------------------------------------

FORBIDDEN = ("label", "oracle", "edit_atoms", "edit_bonds", "edit_", "pair_id", "y_raw",
             "targets_raw", "target_std", "target_values", "true_value", "property_value")


def gate_label_blind(*, device="cpu") -> dict[str, Any]:
    import lfa_pool

    signatures = {}
    ok = True
    for fn in (lfa_pool.build_molecule, S.generate_families, S.score_functions,
               S.matched_controls, S.shared_pool, S.select_top1):
        names = list(inspect.signature(fn).parameters)
        bad = [n for n in names if any(tok in n.lower() for tok in FORBIDDEN)]
        signatures[fn.__name__] = names
        ok = ok and not bad
    # AST-level check: docstrings/comments may *describe* the blindness, but no
    # identifier in the code may reference labels, targets, oracle or edits.
    import ast

    source_tokens = {}
    for mod in ("lfa_search.py", "lfa_pool.py"):
        tree = ast.parse((C.CODE_DIR / mod).read_text(encoding="utf-8"))
        used: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                used.add(node.attr.lower())
            elif isinstance(node, ast.alias):
                used.add(node.name.lower())
                if node.asname:
                    used.add(node.asname.lower())
        hits = sorted(n for n in used if any(tok in n for tok in FORBIDDEN))
        source_tokens[mod] = {"identifiers_checked": len(used), "forbidden_hits": hits,
                              "passed": not hits}
        ok = ok and not hits

    # rebuild three pools with the oracle payload and the labels replaced by
    # shuffled copies: the pools must be byte-identical.
    b, b_wc = _bundles()
    split = "valid"
    occ = D.split_occ(split)
    model_l = M.load_model(C.CHECKPOINT_NAMES["LCRC_WC"], b, device=device)[0]
    model_g = M.load_model(C.CHECKPOINT_NAMES["GINE_WC"], b, device=device)[0]
    rows = [r for r in D.main_scope() if r["split"] == split][:3]
    pools = []
    for shuffle_seed in (None, 12345):
        orig_oracle = D.oracle_payload
        orig_targets = D.target_values
        if shuffle_seed is None:
            rec_payload = orig_oracle()
            D.oracle_payload = lambda: rec_payload
            D.target_values = lambda task, s: np.asarray(orig_targets(task, s))
        else:
            rng = np.random.default_rng([int(shuffle_seed), 3])
            rec_payload = dict(orig_oracle())
            for key in ("WC_LOGP", "WC_LOGP_atom"):
                arr = np.array(rec_payload[key], dtype=np.float64).reshape(-1)
                rec_payload[key] = rng.permutation(arr)
            D.oracle_payload = lambda: rec_payload
            base_targets = {s: np.asarray(orig_targets("wc", s), dtype=np.float64) for s in C.SPLITS}
            D.target_values = lambda task, s: rng.permutation(base_targets[s])
        try:
            mol_pools = []
            for rec in rows:
                cache = b.caches[split]
                row = int(rec["cache_row"])
                gp_l = M.lcrc_gradient(model_l, cache, occ, row, b_wc)
                gp_g = M.gine_gradient(model_g, M.gine_single_batch(b, split, row, device=device), b_wc)
                fields_l = {"n_atoms": gp_l["contrib_atom_std"].size, "n_bonds": gp_l["grad_transformed"].size,
                            "grad": gp_l["grad_transformed"].tolist(), "contrib": gp_l["contrib_atom_std"].tolist(),
                            "pred_raw": gp_l["pred_raw"]}
                fields_g = {"n_atoms": fields_l["n_atoms"], "n_bonds": fields_l["n_bonds"],
                            "grad": gp_g["grad_transformed"].tolist(), "contrib": None,
                            "pred_raw": gp_g["pred_raw"]}
                out = lfa_pool.build_molecule(rec, fields_l, fields_g, scope=C.SCOPE_MAIN, task="wc")
                mol_pools.append({k: out[k] for k in ("split", "cache_row", "n_atoms", "n_bonds", "pool",
                                                      "generator_members", "selector_scores",
                                                      "selections", "controls")})
            pools.append(mol_pools)
        finally:
            D.oracle_payload = orig_oracle
            D.target_values = orig_targets
    identical = pools[0] == pools[1]
    return {"signatures": signatures, "source_tokens": source_tokens,
            "pools_identical_after_shuffle": bool(identical),
            "passed": bool(ok and identical)}


# ---------------------------------------------------------------------------
# 8. WC_LOGP capture permutation equivariance
# ---------------------------------------------------------------------------


def capture(contrib: np.ndarray, atoms: Sequence[int]) -> float | None:
    absd = np.abs(np.asarray(contrib, dtype=np.float64))
    denom = float(absd.sum())
    if not np.isfinite(denom) or denom <= 0.0:
        return None
    ids = [int(v) for v in atoms if 0 <= int(v) < absd.size]
    return float(absd[ids].sum() / denom) if ids else 0.0


def gate_oracle_permutation(*, seed: int = C.CONTROL_SEED) -> dict[str, Any]:
    rng = np.random.default_rng([int(seed), 71])
    b, _ = _bundles()
    recs, ok = [], True
    for rec in [r for r in D.main_scope() if r["split"] == "valid"][:4]:
        g = D.graph_contribution(int(rec["row_id"]))
        contrib = np.asarray(g["contrib"], dtype=np.float64)
        n = int(contrib.size)
        perm = rng.permutation(n)
        contrib_p = np.empty_like(contrib)
        contrib_p[perm] = contrib
        cand = list(range(min(4, n)))
        cand_p = [int(perm[v]) for v in cand]
        c0 = capture(contrib, cand)
        c1 = capture(contrib_p, cand_p)
        err = abs(float(c0) - float(c1))
        sum_err = abs(float(contrib_p.sum()) - g["value"])
        passed = bool(err <= 1e-12 and sum_err <= float(C.oracle_tolerance(g["value"])))
        recs.append({"split": rec["split"], "row_id": int(rec["row_id"]), "capture": c0,
                     "capture_permuted": c1, "capture_abs_error": err,
                     "permuted_sum_closure_error": sum_err, "passed": passed})
        ok = ok and passed
    return {"records": recs, "passed": bool(ok)}


# ---------------------------------------------------------------------------
# 9. pool identity across the two select workers
# ---------------------------------------------------------------------------


def gate_pool_identity() -> dict[str, Any]:
    manifest_path = C.POOL_DIR / "pool_manifest.json"
    if not manifest_path.is_file():
        return {"passed": False, "reason": "pool manifest missing"}
    manifest = C.read_json(manifest_path)
    out: dict[str, Any] = {"pools": {}, "selects": {}, "passed": True}
    for scope in (C.SCOPE_MAIN, C.SCOPE_AUX):
        rep = manifest["reports"].get(scope)
        path = C.POOL_DIR / f"pool_{scope}.jsonl"
        if rep is None or not path.is_file():
            out["pools"][scope] = {"present": False}
            out["passed"] = False
            continue
        actual = C.sha256_file(path)
        out["pools"][scope] = {"sha256": actual, "manifest": rep["sha256"],
                               "match": bool(actual == rep["sha256"])}
        out["passed"] = out["passed"] and bool(actual == rep["sha256"])
    for family in ("lcrc", "gine"):
        for scope in (C.SCOPE_MAIN, C.SCOPE_AUX):
            path = C.SELECT_DIR / f"select_manifest_{family}_{scope}.json"
            if not path.is_file():
                continue
            rep = C.read_json(path)
            key = f"{family}_{scope}"
            out["selects"][key] = {"pool_sha256": rep["pool_sha256"],
                                   "pool_manifest": out["pools"][scope].get("manifest"),
                                   "match": bool(rep["pool_sha256"] == out["pools"][scope].get("manifest"))}
            out["passed"] = out["passed"] and out["selects"][key]["match"]
    return out


def run(*, device: str = "cpu", only: Sequence[str] | None = None) -> dict[str, Any]:
    path = C.ANALYSIS_DIR / "gates.json"
    gates: dict[str, Any] = {}
    if path.is_file():
        existing = C.read_json(path)
        gates.update(existing.get("gates", {}))
        gates.pop("all_passed", None)
    plan = {
        "all_ones": lambda: gate_all_ones(device=device),
        "gine_batched_reference": lambda: gate_gine_batched_reference(device=device),
        "finite_difference": lambda: gate_finite_difference(device=device),
        "renumbering": lambda: gate_renumbering(device=device),
        "dose_path": lambda: gate_dose_path(device=device),
        "weight_semantics": lambda: gate_weight_semantics(),
        "label_blind": lambda: gate_label_blind(device=device),
        "oracle_permutation": lambda: gate_oracle_permutation(),
        "pool_identity": lambda: gate_pool_identity(),
    }
    for name, fn in plan.items():
        if only is not None and name not in only:
            continue
        print(f"[gates] {name} ...", flush=True)
        try:
            gates[name] = fn()
        except Exception as exc:  # noqa: BLE001 - a gate failure is data, not a crash
            gates[name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"[gates] {name} ERROR {gates[name]['error']}", flush=True)
        print(f"[gates] {name} passed={gates[name].get('passed')}", flush=True)
    gates["all_passed"] = bool(all(v.get("passed") for v in gates.values() if isinstance(v, dict)))
    C.write_json(path, {
        "round_id": C.ROUND_ID, "created_utc": C.utc_now(), "environment": C.environment(),
        "device": device, "gates": gates,
    })
    return gates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    only = tuple(s for s in args.only.split(",")) if args.only else None
    gates = run(device=args.device, only=only)
    print(json.dumps({k: (v.get("passed") if isinstance(v, dict) else v) for k, v in gates.items()}, indent=1))
    return 0 if gates["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
