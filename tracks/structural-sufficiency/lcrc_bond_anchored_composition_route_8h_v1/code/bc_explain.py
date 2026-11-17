"""Stage C explanation evaluation: does the bond-anchored composition localise?

Every measurement here is *descriptive development evidence* (`valid` split of
ogbg-mollipo only); none of it feeds the test-time selection of a model, and the
task verdict never rests on a single deletion curve.

Three questions, each operationalised so that a trivial model cannot pass:

1. **Event specificity** — deleting the events that the composer itself rates
   highest must move the prediction more than deleting the same number of events
   at random (and more than deleting the lowest-rated events).  Measured on the
   composition residual only, so the frozen C1 base is untouched.
2. **Physical-bond necessity** — zeroing the *anchor bond* embedding of the
   deleted events must cost more than zeroing a peripheral bond of the same
   events.  That is the test that "the bond is real" rather than decorative.
3. **Compound-level attribution** — the molecules whose prediction moves most
   must be the molecules whose error falls most, otherwise the residual is not
   buying task information.

Plus two invariants that must hold exactly on a *trained* model: direction
asymmetry ``|c_P - c_P(reversed)| == 0`` and the branch-off identity
``pred(branch=False) == pred_base``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import bc_common as C
import bc_data as D
import bc_models as M

DELETION_FRACTIONS = (0.10, 0.25, 0.50)
RANDOM_DRAWS = 5


def _load_unit(unit: str, device: torch.device) -> tuple[M.BAJC, dict[str, Any], Any, dict[str, Any]]:
    """Load a BAJC checkpoint from a result directory or a unit id."""
    path = Path(unit)
    if path.suffix != ".pt":
        path = path / "checkpoint_best_val.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    task = payload["task"]
    variant = payload["variant"]
    b = D.bundle(task)
    arm = C.modules()["cr_models"].ArmA(b.spec, b.norm)
    model = M.BAJC(arm, variant)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, b, payload, {"checkpoint": str(path)}


def _topk_mask(c: np.ndarray, graph: np.ndarray, n_graphs: int, frac: float, mode: str,
               rng: np.random.Generator | None = None) -> np.ndarray:
    keep = np.ones(c.size, dtype=bool)
    order_by_graph = np.argsort(graph, kind="stable")
    bounds = np.searchsorted(graph[order_by_graph], np.arange(n_graphs + 1))
    for g in range(n_graphs):
        lo, hi = int(bounds[g]), int(bounds[g + 1])
        idx = order_by_graph[lo:hi]
        if idx.size == 0:
            continue
        k = int(round(frac * idx.size))
        if k <= 0:
            continue
        if mode == "top":
            pick = idx[np.argsort(-np.abs(c[idx]), kind="stable")[:k]]
        elif mode == "bottom":
            pick = idx[np.argsort(np.abs(c[idx]), kind="stable")[:k]]
        elif mode == "random":
            if rng is None:
                raise ValueError("random mode needs a generator")
            pick = rng.choice(idx, size=k, replace=False)
        else:
            raise KeyError(mode)
        keep[pick] = False
    return keep


def _predict(model: M.BAJC, b, p3, rows: Sequence[int], device, *, gate_product=None,
             zero_anchor: np.ndarray | None = None, swap: bool = False) -> np.ndarray:
    bt = D.make_batch(b, "valid", rows, p3).to(device)
    p3b = bt.p3
    if swap:
        p3b = dict(p3b)
        p3b["p3_occ"] = p3b["p3_occ"][:, [2, 3, 0, 1]]
    with torch.no_grad():
        out = model(bt.batch, p3b,
                    gate_product=None if gate_product is None else gate_product.to(device),
                    return_parts=True)
    return out["pred_std"].detach().cpu().numpy()


def event_specificity(model: M.BAJC, b, p3, device, *, p3_split=None,
                      n_molecules: int = 420, seed: int = C.BOOTSTRAP_SEED) -> dict[str, Any]:
    order, _, _ = D.eval_order(b, "valid")
    rows = order[: min(n_molecules, order.size)]
    bt = D.make_batch(b, "valid", rows, p3).to(device)
    with torch.no_grad():
        parts = model(bt.batch, bt.p3, return_parts=True)
    c = parts["c_P"].detach().cpu().numpy().astype(np.float64)
    graph = bt.p3["p3_graph"].numpy().astype(np.int64)
    full = parts["pred_std"].detach().cpu().numpy().astype(np.float64)
    base = parts["pred_base_std"].detach().cpu().numpy().astype(np.float64)
    n_graphs = int(bt.n_graphs)
    out: dict[str, Any] = {"n_molecules": int(rows.size), "n_events": int(c.size)}
    rng = np.random.default_rng(int(seed))
    curves = {}
    for frac in DELETION_FRACTIONS:
        entry = {}
        for mode in ("top", "bottom", "random"):
            draws = []
            for d in range(RANDOM_DRAWS if mode == "random" else 1):
                keep = _topk_mask(c, graph, n_graphs, frac, mode, rng if mode == "random" else None)
                gate = torch.from_numpy(keep.astype(np.float32))
                masked = _predict(model, b, p3, rows, device, gate_product=gate)
                delta = full - masked
                draws.append(
                    {
                        "mean_abs_delta": float(np.mean(np.abs(delta))),
                        "max_abs_delta": float(np.max(np.abs(delta))),
                        "mean_abs_delta_over_std": float(np.mean(np.abs(delta)) / (np.std(base) + 1e-12)),
                    }
                )
            if mode == "random":
                entry["random"] = {
                    "draws": len(draws),
                    "mean_abs_delta": float(np.mean([d["mean_abs_delta"] for d in draws])),
                    "max_abs_delta": float(np.max([d["max_abs_delta"] for d in draws])),
                    "dispersion": float(np.std([d["mean_abs_delta"] for d in draws])),
                }
            else:
                entry[mode] = draws[0]
        entry["specificity_top_over_random"] = float(
            entry["top"]["mean_abs_delta"] / max(1e-12, entry["random"]["mean_abs_delta"])
        )
        entry["specificity_top_over_bottom"] = float(
            entry["top"]["mean_abs_delta"] / max(1e-12, entry["bottom"]["mean_abs_delta"])
        )
        curves[f"{frac:.2f}"] = entry
    out["fractions"] = curves

    out["bond_intervention"] = bond_intervention(
        model, b, p3_split, rows, device, n_molecules=rows.size
    )

    # trained-model invariants
    swapped = _predict(model, b, p3, rows, device, swap=True)
    out["direction_asymmetry_max_abs"] = float(np.max(np.abs(swapped - full)))
    with torch.no_grad():
        off = model(bt.batch, bt.p3, branch=False, return_parts=True)
    out["branch_off_identity_max_abs"] = float(
        np.max(np.abs(off["pred_std"].detach().cpu().numpy() - base))
    )
    out["trained_delta_rms_over_teacher_std"] = float(
        np.std(full - base) / (np.std(base) + 1e-12)
    )
    return out


def _bond_lookup(cache, molecule: int) -> dict[tuple[int, int], int]:
    """{(atom_u, atom_v): local bond id} of one molecule, canonical orientation."""
    ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    idx = np.asarray(cache["bond_index"], dtype=np.int64)
    lo, hi = int(ptr[molecule]), int(ptr[molecule + 1])
    out = {}
    for local, (u, v) in enumerate(idx[lo:hi]):
        key = (int(min(u, v)), int(max(u, v)))
        out.setdefault(key, int(local))
    return out


def _event_bonds(cache, p3s: P3.Path3Split, molecule: int) -> dict[str, np.ndarray]:
    """Local bond ids of the first hop, the anchor and the last hop of each event."""
    lo, hi = int(p3s.ptr[molecule]), int(p3s.ptr[molecule + 1])
    atoms = p3s.atoms[lo:hi]
    lookup = _bond_lookup(cache, molecule)
    first = np.array([lookup.get((min(a, b), max(a, b)), -1) for a, b, _c, _d in atoms], dtype=np.int64)
    last = np.array([lookup.get((min(c, d), max(c, d)), -1) for _a, _b, c, d in atoms], dtype=np.int64)
    return {"first": first, "anchor": np.asarray(p3s.anchor[lo:hi], dtype=np.int64), "last": last}


def bond_intervention(model: M.BAJC, b, p3, rows: np.ndarray, device, *,
                      fractions: Sequence[float] = (0.02, 0.05, 0.10, 0.25),
                      n_random: int = 5, n_molecules: int | None = None,
                      seed: int = C.BOOTSTRAP_SEED) -> dict[str, Any]:
    """``CONDITIONAL_S1_S2_EVIDENCE_MASK`` on real physical bonds.

    For every molecule the top-``k`` events by ``|c_P|`` are selected and then the
    *anchor* bond (the b-c bond, the only bond the event shares between its two
    paths) is gated off, keeping the number of distinct gated bonds and their bond
    types matched against a random bond set of the same molecule.
    """
    cache = b.caches["valid"]
    rows = np.asarray(rows)[: (n_molecules or np.asarray(rows).size)]
    bt = D.make_batch(b, "valid", rows, p3).to(device)
    gb = bt.gb
    with torch.no_grad():
        parts = model(bt.batch, bt.p3, return_parts=True)
    c = parts["c_P"].detach().cpu().numpy().astype(np.float64)
    graph = bt.p3["p3_graph"].numpy().astype(np.int64)
    pred_full = parts["pred_std"].detach().cpu().numpy().astype(np.float64)
    n_graphs = int(bt.n_graphs)
    rng = np.random.default_rng(int(seed))
    max_bonds = int(gb.max_bonds)
    occ = bt.p3["p3_occ"].numpy().astype(np.int64)

    def gate_and_predict(bond_sets: dict[int, set[int]]) -> np.ndarray:
        M = torch.ones((n_graphs, max_bonds), dtype=torch.float32)
        for g, bonds in bond_sets.items():
            for local in bonds:
                M[g, int(local)] = 0.0
        M = M.to(device)
        bond_gate, nbr_gate = gb.s1_gate(M)
        path_weight = gb.s2_weight(M)
        # an event is deleted when any of its three physical bonds is gated
        with torch.no_grad():
            out = model(bt.batch, bt.p3, gate=(bond_gate, nbr_gate, path_weight),
                        gate_product=ev_alive(M), return_parts=True)
        return out["pred_std"].detach().cpu().numpy().astype(np.float64)

    # per-event physical bond ids in graph coordinates, as (n_ev, 3)
    ev_bonds = np.zeros((c.size, 3), dtype=np.int64)
    ev_bond_type = np.zeros((c.size, 3), dtype=np.int64)
    ev_molecule = np.zeros(c.size, dtype=np.int64)
    bond_type = np.asarray(cache["bond_feat"])[:, 0] if "bond_feat" in cache else None
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    for position, molecule in enumerate(rows):
        local = _event_bonds(cache, p3, int(molecule))
        lo, hi = int(p3.ptr[int(molecule)]), int(p3.ptr[int(molecule) + 1])
        ev_bonds[lo:hi, 0] = local["first"]
        ev_bonds[lo:hi, 1] = local["anchor"]
        ev_bonds[lo:hi, 2] = local["last"]
        ev_molecule[lo:hi] = position
        if bond_type is not None:
            off = int(bond_ptr[int(molecule)])
            for k in range(3):
                ids = np.clip(ev_bonds[lo:hi, k], 0, None) + off
                ev_bond_type[lo:hi, k] = bond_type[ids]

    ev_graph = graph

    def ev_alive(M: torch.Tensor) -> torch.Tensor:
        """1.0 for every event none of whose three bonds is gated."""
        Mm = M.detach().cpu().numpy()
        alive = np.ones(c.size, dtype=np.float64)
        for k in range(3):
            g = ev_graph
            ids = ev_bonds[:, k]
            valid = ids >= 0
            alive[valid] *= Mm[g[valid], ids[valid]]
        return torch.from_numpy(alive.astype(np.float32)).to(M.device)

    def selected_bonds(mode: str) -> dict[int, set[int]]:
        sets: dict[int, set[int]] = {g: set() for g in range(n_graphs)}
        for g in range(n_graphs):
            mask = ev_graph == g
            if not mask.any():
                continue
            idx = np.where(mask)[0]
            k = max(1, int(round(fraction * idx.size)))
            pick = idx[np.argsort(-np.abs(c[idx]), kind="stable")[:k]]
            column = {"anchor": 1, "first": 0, "last": 2}[mode]
            for e in pick:
                bond = int(ev_bonds[e, column])
                if bond >= 0:
                    sets[g].add(bond)
        return sets

    def matched_random(reference: dict[int, set[int]]) -> dict[int, set[int]]:
        """Random bond sets with the same size and bond-type multiset per molecule."""
        out: dict[int, set[int]] = {}
        off = bond_ptr
        for g, bonds in reference.items():
            molecule = int(rows[g])
            lo, hi = int(off[molecule]), int(off[molecule + 1])
            pool = np.arange(hi - lo)
            if bond_type is not None:
                want = sorted(bond_type[np.asarray(sorted(bonds)) + lo].tolist())
                have = bond_type[lo:hi]
                chosen: list[int] = []
                for t in want:
                    cand = [b for b in pool if have[b] == t and b not in chosen]
                    if not cand:
                        cand = [b for b in pool if b not in chosen]
                    if not cand:
                        break
                    chosen.append(int(rng.choice(cand)))
            else:
                k = min(len(bonds), pool.size)
                chosen = rng.choice(pool, size=k, replace=False).tolist()
            out[g] = set(int(x) for x in chosen)
        return out

    result: dict[str, Any] = {"n_events": int(c.size), "by_fraction": {}}
    for fraction in fractions:
      for mode in ("anchor", "first", "last"):
        reference = selected_bonds(mode)
        pred_masked = gate_and_predict(reference)
        delta = pred_full - pred_masked
        entry = {
            "n_gated_bonds": int(sum(len(v) for v in reference.values())),
            "gated_bond_fraction_of_molecule_mean": float(
                sum(len(v) for v in reference.values())
                / max(1e-12, np.mean(np.diff(bond_ptr[np.asarray(rows)])) * n_graphs)
            ),
            "mean_abs_delta": float(np.mean(np.abs(delta))),
            "max_abs_delta": float(np.max(np.abs(delta))),
            "n_molecules_changed": int(np.sum(np.abs(delta) > 1e-9)),
        }
        draws = []
        for _ in range(n_random):
            control = matched_random(reference)
            d = pred_full - gate_and_predict(control)
            draws.append(float(np.mean(np.abs(d))))
        entry["random_mean_abs_delta"] = float(np.mean(draws))
        entry["random_dispersion"] = float(np.std(draws))
        entry["ratio_over_matched_random"] = float(
            entry["mean_abs_delta"] / max(1e-12, entry["random_mean_abs_delta"])
        )
        entry["reference_mean_abs_residual"] = float(np.mean(np.abs(pred_full)))
        result["by_fraction"].setdefault(f"{fraction:.2f}", {})[mode] = entry
    result["note"] = (
        "anchor = the middle bond shared by both paths of an event; first/last = the two "
        "peripheral hops. Every arm gates the same number of bonds with the same bond-type "
        "multiset (random arm: matched random draw of the same size and types, "
        f"{n_random} draws, seed {seed})."
    )
    return result


def compound_attribution(model: M.BAJC, b, p3, device) -> dict[str, Any]:
    """Do the molecules with the largest contribution have the largest error drop?"""
    from scipy.stats import spearmanr

    order, _, y_raw = D.eval_order(b, "valid")
    bt = D.make_batch(b, "valid", order, p3).to(device)
    with torch.no_grad():
        parts = model(bt.batch, bt.p3, return_parts=True)
    y = np.asarray(y_raw, dtype=np.float64).ravel()
    base = parts["pred_base_std"].detach().cpu().numpy().astype(np.float64)
    delta = parts["delta_comp"].detach().cpu().numpy().astype(np.float64)
    pred = base + delta
    sq_base = (base - y) ** 2
    sq_new = (pred - y) ** 2
    rho_contrib_vs_reduction = spearmanr(np.abs(delta), -(sq_new - sq_base))
    top = np.argsort(-np.abs(delta))[: max(1, int(0.1 * delta.size))]
    return {
        "n": int(delta.size),
        "spearman_abs_contribution_vs_error_reduction": float(rho_contrib_vs_reduction.statistic),
        "spearman_p": float(rho_contrib_vs_reduction.pvalue),
        "rmse_base_std": float(np.sqrt(np.mean(sq_base))),
        "rmse_new_std": float(np.sqrt(np.mean(sq_new))),
        "top_decile_abs_contribution_share": float(
            np.sum(np.abs(delta[top])) / max(1e-12, np.sum(np.abs(delta)))
        ),
        "top_decile_fraction_of_error_reduction": float(
            np.sum(sq_base[top] - sq_new[top]) / max(1e-12, np.sum(sq_base - sq_new))
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Stage C explanation evaluation")
    ap.add_argument("--unit", required=True, help="result directory or checkpoint .pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(C.ANALYSIS_DIR / "explanation_eval.json"))
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    model, b, payload, meta = _load_unit(args.unit, device)
    p3 = D.build_paths3(b)
    report: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "unit": payload.get("unit", meta["checkpoint"]),
        "task": payload["task"],
        "variant": payload["variant"],
        "institution": payload["institution"],
        "seed": payload["seed"],
        "selected_epoch": payload["epoch"],
        "selected_valid_rmse_raw": payload["valid_rmse_raw"],
        "task_invariant": "CONDITIONAL_S1_S2_EVIDENCE_MASK",
        "split": "valid",
    }
    report["event_specificity"] = event_specificity(
        model, b, p3["valid"], device, p3_split=p3["valid"]
    )
    report["compound_attribution"] = compound_attribution(model, b, p3["valid"], device)
    C.atomic_write_json(Path(args.out), report)
    print(json.dumps({k: report[k] for k in ("unit", "variant", "selected_epoch")}, indent=2))
    spec = report["event_specificity"]
    for frac, entry in spec["fractions"].items():
        print(f"  delete {frac}: top/random={entry['specificity_top_over_random']:.2f} "
              f"top/bottom={entry['specificity_top_over_bottom']:.2f}")
    for frac, entry in spec["bond_intervention"]["by_fraction"].items():
        print(f"  bond intervention {frac}: " + " ".join(
            f"{k}={v['ratio_over_matched_random']:.3f}" for k, v in entry.items()))
    print("  direction asymmetry:", spec["direction_asymmetry_max_abs"])
    print("  attribution:", json.dumps(report["compound_attribution"]))
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
