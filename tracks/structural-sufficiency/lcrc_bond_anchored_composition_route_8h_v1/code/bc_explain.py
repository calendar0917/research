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
    bt = D.make_batch(b, "valid", rows, p3)
    p3b = bt.p3
    if swap:
        p3b = dict(p3b)
        p3b["p3_occ"] = p3b["p3_occ"][:, [2, 3, 0, 1]]
    with torch.no_grad():
        out = model(bt.batch.to(device), {k: v.to(device) for k, v in p3b.items()},
                    gate_product=None if gate_product is None else gate_product.to(device),
                    return_parts=True)
    return out["pred_std"].detach().cpu().numpy()


def event_specificity(model: M.BAJC, b, p3, device, *, n_molecules: int = 420,
                      seed: int = C.BOOTSTRAP_SEED) -> dict[str, Any]:
    rows = np.arange(min(n_molecules, int(b.caches["valid"]["n_graphs"])))
    bt = D.make_batch(b, "valid", rows, p3)
    with torch.no_grad():
        parts = model(bt.batch.to(device), {k: v.to(device) for k, v in bt.p3.items()},
                      return_parts=True)
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

    # bond necessity: zero the anchor bond vs a peripheral bond of the same events
    occ = bt.p3["p3_occ"]
    bond_emb = parts["evidence"].new_zeros(0)  # placeholder, real embedding below
    anchor = bt.p3["p3_anchor_bond"].to(device)
    keep_top = torch.from_numpy(_topk_mask(c, graph, n_graphs, 0.25, "top").astype(np.float32)).to(device)
    with torch.no_grad():
        a = model.trunk.atom_encoder(bt.batch["atom_feat"].to(device))
        idx = bt.batch["bond_index"].to(device)
        au, av = a[idx[:, 0]], a[idx[:, 1]]
        bfull = model.trunk.bond_encoder(
            torch.cat([au + av, (au - av).abs(), au * av, bt.batch["bond_feat"].to(device)], dim=1)
        )
        del bond_emb
        p3d = {k: v.to(device) for k, v in bt.p3.items()}
        full_delta = model.compose(parts["evidence"].to(device), bfull, p3d,
                                  n_graphs=n_graphs, gate_product=keep_top)["delta_comp"]
        # anchor bond ablated for the selected events only
        b_anchor = bfull.clone()
        sel = keep_top > 0
        b_anchor[p3d["p3_anchor_bond"][sel]] = 0.0
        anchor_ablate = model.compose(parts["evidence"].to(device), b_anchor, p3d,
                                      n_graphs=n_graphs, gate_product=None)["delta_comp"]
        # peripheral bond (first path's first hop) ablated for the same events
        b_periph = bfull.clone()
        p_first = p3d["p3_bond_first"] if "p3_bond_first" in p3d else None
        if p_first is None:
            periph_shift = None
        else:
            b_periph[p_first[sel]] = 0.0
            periph_shift = model.compose(parts["evidence"].to(device), b_periph, p3d,
                                         n_graphs=n_graphs)["delta_comp"]
    baseline_delta = model.compose(parts["evidence"].to(device), bfull, p3d,
                                   n_graphs=n_graphs)["delta_comp"]
    out["bond_necessity"] = {
        "fraction": 0.25,
        "mean_abs_shift_anchor_ablation": float(
            (anchor_ablate - baseline_delta).abs().mean().item()
        ),
        "mean_abs_shift_peripheral_ablation": None if periph_shift is None else float(
            (periph_shift - baseline_delta).abs().mean().item()
        ),
        "mean_abs_reference_delta": float(baseline_delta.abs().mean().item()),
        "note": "anchor ablation zeroes v(b_e) for the selected events; a peripheral ablation zeroes the first-hop bond of the same events",
    }

    # trained-model invariants
    swapped = _predict(model, b, p3, rows, device, swap=True)
    out["direction_asymmetry_max_abs"] = float(np.max(np.abs(swapped - full)))
    with torch.no_grad():
        off = model(bt.batch.to(device), {k: v.to(device) for k, v in bt.p3.items()},
                    branch=False, return_parts=True)
    out["branch_off_identity_max_abs"] = float(
        np.max(np.abs(off["pred_std"].detach().cpu().numpy() - base))
    )
    out["trained_delta_rms_over_teacher_std"] = float(
        np.std(full - base) / (np.std(base) + 1e-12)
    )
    return out


def compound_attribution(model: M.BAJC, b, p3, device) -> dict[str, Any]:
    """Do the molecules with the largest contribution have the largest error drop?"""
    from scipy.stats import spearmanr

    bt = D.make_batch(b, "valid", np.arange(int(b.caches["valid"]["n_graphs"])), p3)
    with torch.no_grad():
        parts = model(bt.batch.to(device), {k: v.to(device) for k, v in bt.p3.items()},
                      return_parts=True)
    y = bt.batch["y_std"].detach().cpu().numpy().astype(np.float64).ravel()
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
    report["event_specificity"] = event_specificity(model, b, p3["valid"], device)
    report["compound_attribution"] = compound_attribution(model, b, p3["valid"], device)
    C.atomic_write_json(Path(args.out), report)
    print(json.dumps({k: report[k] for k in ("unit", "variant", "selected_epoch")}, indent=2))
    spec = report["event_specificity"]
    for frac, entry in spec["fractions"].items():
        print(f"  delete {frac}: top/random={entry['specificity_top_over_random']:.2f} "
              f"top/bottom={entry['specificity_top_over_bottom']:.2f}")
    print("  bond necessity:", json.dumps(spec["bond_necessity"]))
    print("  direction asymmetry:", spec["direction_asymmetry_max_abs"])
    print("  attribution:", json.dumps(report["compound_attribution"]))
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
