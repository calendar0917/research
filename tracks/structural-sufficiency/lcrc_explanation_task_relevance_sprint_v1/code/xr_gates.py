"""P1 implementation gates, run before any formal P2 evaluation."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import xr_common as C
import xr_assets as A
import xr_discover as D
import xr_models as M


def _sample_endpoints(rows: Sequence[Mapping[str, Any]], n: int, seed: int) -> list[Mapping[str, Any]]:
    rng = np.random.default_rng(int(seed))
    if len(rows) <= n:
        return list(rows)
    # deterministic spread plus random fill
    idx = np.linspace(0, len(rows) - 1, max(2, n // 2), dtype=np.int64).tolist()
    remaining = [i for i in range(len(rows)) if i not in set(idx)]
    rng.shuffle(remaining)
    idx.extend(remaining[: n - len(idx)])
    return [rows[int(i)] for i in sorted(set(idx))[:n]]


def _passed(rec: Mapping[str, Any]) -> bool:
    if "passed" in rec:
        return bool(rec["passed"])
    if "pass" in rec:
        return bool(rec["pass"])
    return bool(all(r.get("passed", False) for r in rec.get("records", [])))


def _choose_fd_bonds(mol, grad: np.ndarray, k: int = 3) -> list[int]:
    g = np.abs(np.asarray(grad, dtype=np.float64).reshape(-1))
    order = np.argsort(-g).tolist()
    out: list[int] = []
    for b in order:
        if int(b) not in out:
            out.append(int(b))
        if len(out) >= k:
            break
    # include a low-magnitude bond when possible
    low = np.argsort(g).tolist()
    for b in low:
        if int(b) not in out:
            out.append(int(b))
            break
    return out


def _edit_consistency(endpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bad = []
    for ep in endpoints:
        atoms = set(int(v) for v in ep["edit_atoms"])
        bonds = set(tuple(sorted(int(x) for x in b)) for b in ep["edit_bonds"])
        bond_atoms = set()
        for u, v in bonds:
            bond_atoms.add(u)
            bond_atoms.add(v)
        attach_bonds = set(tuple(sorted(int(x) for x in b)) for b in ep["attach_bonds"])
        changed = set(tuple(sorted(int(x) for x in b)) for b in ep["changed_core_bonds"])
        reasons = []
        if atoms != bond_atoms:
            reasons.append("edit_atoms != atoms(edit_bonds)")
        if int(ep["edit_atom_count"]) != len(atoms):
            reasons.append("edit_atom_count mismatch")
        if int(ep["edit_bond_count"]) != len(bonds):
            reasons.append("edit_bond_count mismatch")
        if not attach_bonds <= bonds:
            reasons.append("attachment bonds not subset of edit bonds")
        if not changed <= bonds:
            reasons.append("changed core bonds not subset of edit bonds")
        if int(ep["attach_atom"]) not in atoms:
            reasons.append("attachment atom not in edit atoms")
        if reasons:
            bad.append({"endpoint": f"{ep['pair_id']}:{ep['side']}", "reasons": reasons})
    return {"n_endpoints": len(endpoints), "n_bad": len(bad), "bad": bad[:20], "passed": not bad}


def _label_blind_audit() -> dict[str, Any]:
    sig = inspect.signature(D.discover_from_gradient)
    params = list(sig.parameters)
    forbidden = [p for p in params if any(t in p.lower() for t in ("label", "target", "oracle", "edit", "pair", "y_"))]
    src = inspect.getsource(D.discover_from_gradient)
    forbidden_src = [token for token in ("y_a", "y_b", "label", "target", "oracle", "edit_atoms") if token in src]
    return {
        "discover_from_gradient_parameters": params,
        "forbidden_parameters": forbidden,
        "forbidden_tokens_in_source": forbidden_src,
        "passed": bool(not forbidden and not forbidden_src),
        "note": "discovery receives only the molecule graph and a model gradient field",
    }


def run_gates(*, device: str = "cpu", n_sample: int = 8, fd_bonds: int = 3) -> dict[str, Any]:
    C.install_source_paths()
    b = M.bundle(C.DATASET)
    mods = C.source_modules()
    AG, AD = mods["as_gate"], mods["as_data"]
    endpoints = C.read_jsonl_gz(C.PAIR_ENDPOINTS)
    sample = _sample_endpoints(endpoints, int(n_sample), C.SAMPLE_SEED)

    results: dict[str, Any] = {
        "identity": json.loads((C.ANALYSIS_DIR / "data_identity.json").read_text()),
        "label_blind": _label_blind_audit(),
        "edit_consistency": _edit_consistency(endpoints),
        "models": {},
    }
    occs: dict[str, Any] = {}
    models = {
        "measured": {
            "LCRC": M.load_model("L_lipophilicity", b, device=device)[0],
            "GINE": M.load_model("G0_lipophilicity", b, device=device)[0],
        },
        "oracle": {
            "LCRC": M.load_model("L_WC_LOGP", b, device=device)[0],
            "GINE": M.load_model("T_WC_LOGP", b, device=device)[0],
        },
    }
    for task, pair in models.items():
        task_rec: dict[str, Any] = {"all_ones": [], "finite_difference": [], "renumbering": []}
        b_task = b
        if task == "oracle":
            b_task = mods["gr_data"].oracle_bundle(C.DATASET, "WC_LOGP")
        for ep in sample:
            split = str(ep["split"])
            if split not in occs:
                occs[split] = AD.build_split_occ(b.caches[split], b.spec, b.dataset, split)
            cache = b.caches[split]
            occ = occs[split]
            row = int(ep["cache_row"])
            mol = AG.SingleMolecule(cache, row, b.spec)
            gp_l = M.lcrc_gradient_pass(pair["LCRC"], cache, occ, row, b_task)
            all_l = M.lcrc_all_ones_check(pair["LCRC"], cache, occ, row, b_task)
            fd_l = M.lcrc_finite_difference(pair["LCRC"], cache, occ, row, b_task, _choose_fd_bonds(mol, gp_l["grad_transformed"][0], fd_bonds))
            batch = M.gine_batch_for_row(b, split, row, device=device)
            gp_g = M.gine_gradient_pass(pair["GINE"], batch, b_task)
            all_g = M.gine_all_ones_check(pair["GINE"], batch, b_task)
            fd_g = M.gine_finite_difference(pair["GINE"], batch, b_task, _choose_fd_bonds(mol, gp_g["grad_transformed"][0], fd_bonds))
            task_rec["all_ones"].append({"endpoint": f"{ep['pair_id']}:{ep['side']}", "LCRC": all_l, "GINE": all_g})
            task_rec["finite_difference"].extend([{"model": "LCRC", "endpoint": f"{ep['pair_id']}:{ep['side']}", **rec} for rec in fd_l])
            task_rec["finite_difference"].extend([{"model": "GINE", "endpoint": f"{ep['pair_id']}:{ep['side']}", **rec} for rec in fd_g])
        # renumbering is split-local; use the sample endpoint's split/row pairs
        renum_l, renum_g = [], []
        for ep in sample[:4]:
            split = str(ep["split"])
            row = int(ep["cache_row"])
            renum_l.append(M.lcrc_renumbering_check(pair["LCRC"], b.caches[split], occs[split], [row], b_task))
            renum_g.append(M.gine_renumbering_check(pair["GINE"], b, split, [row]))
        task_rec["renumbering"] = {"LCRC": renum_l, "GINE": renum_g}
        results["models"][task] = task_rec

    # aggregate gate booleans
    all_ones_ok = all(
        rec["LCRC"]["passed"] and rec["GINE"]["passed"]
        for task in results["models"].values()
        for rec in task["all_ones"]
    )
    fd_records = [rec for task in results["models"].values() for rec in task["finite_difference"]]
    fd_ok = sum(1 for rec in fd_records if rec["passed"])
    fd_pass_rate = fd_ok / len(fd_records) if fd_records else 0.0
    renum_records = [r for task in results["models"].values() for model in task["renumbering"].values() for r in model]
    renum_ok = all(_passed(r) for r in renum_records)
    results["summary"] = {
        "identity_passed": bool(results["identity"]["all_identity_gates_passed"]),
        "label_blind_passed": bool(results["label_blind"]["passed"]),
        "edit_consistency_passed": bool(results["edit_consistency"]["passed"]),
        "all_ones_passed": bool(all_ones_ok),
        "finite_difference_n": len(fd_records),
        "finite_difference_n_passed": int(fd_ok),
        "finite_difference_pass_rate": float(fd_pass_rate),
        "finite_difference_passed": bool(fd_pass_rate >= 0.90),
        "renumbering_n": len(renum_records),
        "renumbering_passed": bool(renum_ok),
        "oracle_pair_closure_all_passed": bool(results["identity"]["endpoint_summary"]["wc_all_closed"]),
        "oracle_graph_closure_passed": bool(results["identity"]["oracle_graph_closure"]["passed"]),
    }
    results["all_passed"] = bool(
        results["summary"]["identity_passed"]
        and results["summary"]["label_blind_passed"]
        and results["summary"]["edit_consistency_passed"]
        and results["summary"]["all_ones_passed"]
        and results["summary"]["finite_difference_passed"]
        and results["summary"]["renumbering_passed"]
        and results["summary"]["oracle_pair_closure_all_passed"]
        and results["summary"]["oracle_graph_closure_passed"]
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-sample", type=int, default=8)
    parser.add_argument("--fd-bonds", type=int, default=3)
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR / "p1_gates.json"))
    args = parser.parse_args()
    out = run_gates(device=args.device, n_sample=args.n_sample, fd_bonds=args.fd_bonds)
    C.write_json(Path(args.out), out)
    print(json.dumps(out["summary"], indent=1))
    print("all_passed", out["all_passed"])
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
