"""Structural mechanism capability: fixed local objects, fixed category counts,
permuted category-to-pair assignment.

This is an input-intervention property check, not chemical performance evidence.
It runs the fitted REF and TOPO checkpoints on real Dev molecules, holds every
edge representation and every local object fixed, preserves the joint relation
category histogram, and permutes which category is attached to which concrete
bond pair.
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Mapping, Sequence

import numpy as np

import acra_common as C
import acra_cache as K
import acra_data as D
import acra_model as M
import acra_split as S


def _direct(model, batch, out, categories=None, gamma=None):
    import torch

    if int(batch["n_pairs"]) == 0:
        return torch.zeros(int(batch["n_molecules"]), device=model.beta.device)
    qe, ke = out["rel_qe"], out["rel_ke"]
    pe, pf = batch["pair_e"], batch["pair_f"]
    dots = 0.5 * ((qe[pe] * ke[pf]).sum(-1) + (qe[pf] * ke[pe]).sum(-1))
    cat = (out["rel_categories"] if categories is None
           else torch.as_tensor(categories, dtype=torch.long, device=qe.device))
    pv = dots / math.sqrt(model.rel_rank)
    if hasattr(model, "rel_bias"):
        pv = pv + model.rel_bias.weight.squeeze(-1)[cat]
    if gamma is not None:
        pv = pv + (torch.as_tensor(gamma, dtype=qe.dtype, device=qe.device)[cat]
                   - 1.0) * dots / math.sqrt(model.rel_rank)
    total = torch.zeros(int(batch["n_molecules"]), device=qe.device).index_add_(
        0, batch["pair_mol"], pv)
    return torch.sigmoid(model.gate_raw) * total / batch["n_bonds"].float().clamp(min=1.0)


def _load(arm: str, pack, dims):
    import torch

    rdir = C.run_dir(f"{arm}_S0")
    ckpt = torch.load(rdir / "state_best.pt", map_location="cpu", weights_only=False)
    model = M.ACRAModel(ckpt["config"], dims[0], dims[1], arm)
    model.load_pack(pack)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt


def run(cfg: Mapping[str, Any], log=print) -> dict[str, Any]:
    import torch

    pack = K.ensure_pack(cfg, log=log)
    raw = D.load_raw(cfg)
    dims = C.official_feature_dims()
    split_payload = S.load(cfg)
    dev = np.asarray(split_payload["dev_rows"], np.int64)
    source = D.BatchSource(pack, "cpu")
    ref, ref_ck = _load("ACRA_REL_REF", pack, dims)
    topo, topo_ck = _load("ACRA_TOPO_COUPLED", pack, dims)
    gamma = topo.gamma().detach().cpu().numpy()
    rng = np.random.default_rng(0)

    cases = []
    for row in dev.tolist():
        batch = source.make([int(row)])
        with torch.no_grad():
            ref_out = ref.forward_batch(batch)
            topo_out = topo.forward_batch(batch)
        cat = ref_out["rel_categories"].cpu().numpy()
        if int(np.unique(cat).size) < 2 or int(batch["n_pairs"]) < 4:
            continue
        perm = rng.permutation(cat)
        with torch.no_grad():
            ref_base = _direct(ref, batch, ref_out)
            ref_perm = _direct(ref, batch, ref_out, categories=perm)
            topo_base = _direct(topo, batch, topo_out, gamma=gamma)
            topo_perm = _direct(topo, batch, topo_out, categories=perm, gamma=gamma)
        delta_ref = float((ref_base - ref_perm).abs().max())
        delta_topo = float((topo_base - topo_perm).abs().max())
        counts = np.bincount(cat, minlength=int(ref.n_rel_categories))
        cases.append({
            "row_id": int(row),
            "n_bonds": int(batch["n_bonds"][0]),
            "n_pairs": int(batch["n_pairs"]),
            "n_categories_used": int(np.unique(cat).size),
            "category_counts_fixed": [int(x) for x in counts],
            "delta_e_rel_ref": delta_ref,
            "delta_e_rel_topo": delta_topo,
            "ref_e_rel": float(ref_base[0]),
            "topo_e_rel": float(topo_base[0]),
            "topo_permuted_e_rel": float(topo_perm[0]),
        })
        if len(cases) >= 200:
            break

    # gamma=1 equivalence on the full Dev split (copied REF weights)
    state = dict(ref.state_dict())
    state["gamma_theta"] = torch.zeros(topo.n_rel_categories)
    topo.load_state_dict(state, strict=True)
    eq = []
    with torch.no_grad():
        for start in range(0, dev.size, 128):
            batch = source.make(dev[start:start + 128])
            a = ref.forward_batch(batch)
            b = topo.forward_batch(batch)
            eq.append(float((a["e_rel"] - b["e_rel"]).abs().max()))
    # restore the trained TOPO weights (not needed after this point, but explicit)
    topo, _ = _load("ACRA_TOPO_COUPLED", pack, dims)

    # a controlled capability vector, clearly separated from trained gamma
    forced = np.asarray([1.3, 0.7, 1.3, 0.7, 1.3, 0.7, 1.3, 0.7], np.float64)
    forced_cases = []
    for case in cases[:50]:
        row = int(case["row_id"])
        batch = source.make([row])
        with torch.no_grad():
            out = topo.forward_batch(batch)
        cat = out["rel_categories"].cpu().numpy()
        if int(np.unique(cat).size) < 2:
            continue
        perm = rng.permutation(cat)
        with torch.no_grad():
            base = _direct(topo, batch, out, gamma=forced)
            changed = _direct(topo, batch, out, categories=perm, gamma=forced)
        forced_cases.append({
            "row_id": row,
            "delta_e_rel_forced_gamma": float((base - changed).abs().max()),
        })

    out = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "intervention": "same graph, same edge representations, same local object "
                        "multiset, same joint relation-category histogram; only the "
                        "category-to-bond-pair assignment is permuted",
        "evidence_kind": "structural capability check; not a chemical prediction claim",
        "test_labels_read": False,
        "ref": {
            "checkpoint": str((C.run_dir("ACRA_REL_REF_S0") / "state_best.pt")
                              .relative_to(C.REPO_ROOT)),
            "best_epoch": int(ref_ck["epoch"]),
            "n_cases": len(cases),
            "max_abs_delta_e_rel": float(max((c["delta_e_rel_ref"] for c in cases), default=0.0)),
            "invariant": bool(all(c["delta_e_rel_ref"] < 1e-5 for c in cases)),
        },
        "topo": {
            "best_epoch": int(topo_ck["epoch"]),
            "trained_gamma": [float(x) for x in gamma.tolist()],
            "trained_gamma_deviation_from_one": float(np.max(np.abs(gamma - 1.0))),
            "n_cases": len(cases),
            "max_abs_delta_e_rel": float(max((c["delta_e_rel_topo"] for c in cases), default=0.0)),
            "n_cases_with_delta_gt_1e-6": int(sum(c["delta_e_rel_topo"] > 1e-6 for c in cases)),
            "trained_gamma_couples_categories": bool(
                any(c["delta_e_rel_topo"] > 1e-6 for c in cases)),
        },
        "gamma_one_equivalence": {
            "n_dev_molecules": int(dev.size),
            "max_abs_delta_e_rel": float(max(eq)),
            "equivalent": bool(max(eq) < 1e-6),
        },
        "forced_nonconstant_gamma_capability": {
            "gamma": [float(x) for x in forced.tolist()],
            "n_cases": len(forced_cases),
            "max_abs_delta_e_rel": float(max((c["delta_e_rel_forced_gamma"]
                                              for c in forced_cases), default=0.0)),
            "note": "engineered capability witness, not a trained model result",
        },
        "cases": cases[:24],
    }
    C.write_json(C.analysis_dir() / "structural_cases.json", out)
    log(f"wrote structural_cases.json: ref max delta="
        f"{out['ref']['max_abs_delta_e_rel']:.3e}, topo max delta="
        f"{out['topo']['max_abs_delta_e_rel']:.3e}, gamma dev="
        f"{out['topo']['trained_gamma_deviation_from_one']:.4f}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    run(C.load_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())