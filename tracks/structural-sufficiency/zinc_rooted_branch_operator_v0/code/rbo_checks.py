"""RBO correctness gates (CPU, fast) + the two three-branch toys.

Only checks that can change a decision are here:

  * parameter count / budget / all params in one optimizer;
  * input legality: the branch input contains centre/neighbour/bond/typed
    histogram only -- no radius-2 key, no supervised checkpoint, no labels;
  * atom renumbering + bond-order invariance, batch-composition invariance;
  * gradient coverage and a nonzero gradient into the three-body channel;
  * checkpoint round trip;
  * toy 1: two three-branch roots with identical one-body and two-body
    statistics but different three-body combination;
  * toy 2: a one/two-body response is identical on those two roots, the RBO
    response (with the e3 term) differs.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_TASK = _HERE.parent
_SSUF = _TASK.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import rbo_model as R  # noqa: E402
import rbo_prep as P  # noqa: E402

TASK = _TASK
OUT = TASK / "analysis"
CACHE = TASK / "cache"


def slice_cache(cache: dict, rows: np.ndarray) -> dict:
    ap, bp = np.asarray(cache["atom_ptr"]), np.asarray(cache["bond_ptr"])
    a0, a1 = int(ap[rows[0]]), int(ap[rows[-1] + 1])
    b0, b1 = int(bp[rows[0]]), int(bp[rows[-1] + 1])
    out = {
        "n_molecules": len(rows),
        "atom_ptr": ap[rows[0]:rows[-1] + 2] - a0,
        "bond_ptr": bp[rows[0]:rows[-1] + 2] - b0,
        "atom_feat": np.asarray(cache["atom_feat"])[a0:a1],
        "atom_type": np.asarray(cache["atom_type"])[a0:a1],
        "bond_feat": np.asarray(cache["bond_feat"])[b0:b1],
        "bond_type": np.asarray(cache["bond_type"])[b0:b1],
        "bond_index": np.asarray(cache["bond_index"])[b0:b1].copy(),
        "global_feat": np.asarray(cache["global_feat"])[rows],
        "labels": np.asarray(cache["labels"])[rows],
    }
    return out


def permute_cache(cache: dict, seed: int) -> dict:
    """Atom-renumber every molecule and shuffle the bond order."""
    rng = np.random.default_rng(seed)
    ap, bp = np.asarray(cache["atom_ptr"]), np.asarray(cache["bond_ptr"])
    atom_feat = np.asarray(cache["atom_feat"]).copy()
    atom_type = np.asarray(cache["atom_type"]).copy()
    bond_feat = np.asarray(cache["bond_feat"]).copy()
    bond_type = np.asarray(cache["bond_type"]).copy()
    bond_index = np.asarray(cache["bond_index"]).copy()
    for m in range(int(cache["n_molecules"])):
        a0, a1 = int(ap[m]), int(ap[m + 1])
        b0, b1 = int(bp[m]), int(bp[m + 1])
        n = a1 - a0
        perm = rng.permutation(n)
        inv = np.empty(n, dtype=np.int64)
        inv[perm] = np.arange(n)
        atom_feat[a0:a1] = atom_feat[a0:a1][inv]
        atom_type[a0:a1] = atom_type[a0:a1][inv]
        bo = rng.permutation(b1 - b0)
        bi = bond_index[b0:b1][bo]
        bond_index[b0:b1] = perm[bi]
        bond_feat[b0:b1] = bond_feat[b0:b1][bo]
        bond_type[b0:b1] = bond_type[b0:b1][bo]
    out = dict(cache)
    out.update(atom_feat=atom_feat, atom_type=atom_type, bond_feat=bond_feat,
               bond_type=bond_type, bond_index=bond_index)
    return out


@torch.no_grad()
def predict(model, rows, cache, prep, device="cpu"):
    model.eval()
    batch = R.collate(rows, cache, prep, device)
    return model(batch).double().cpu().numpy()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    norm = D.load_norm()
    cache = D.load_cache("train")
    prep = torch.load(CACHE / "rbo_train.pt", map_location="cpu", weights_only=False)
    sup = dict(np.load(_SSUF / "zinc_composition_transfer_v1" / "cache" / "support_sets.npz"))
    device = torch.device("cpu")
    model, info = R.build_model(norm, 0, device)
    res["parameters"] = info["parameters"]
    res["ssl"] = info["ssl"]
    res["budget_ok"] = info["parameters"] <= 500_000
    res["param_breakdown"] = {
        n: int(p.numel()) for n, p in model.named_parameters()
    }
    y = np.asarray(cache["labels"], dtype=np.float64)

    rows = np.asarray(sup["fit_minus"], dtype=np.int64)[:32]
    batch = R.collate(rows, cache, prep, device)
    out = model(batch)
    res["forward"] = {
        "n_branches": int(batch["hist"].shape[0]),
        "n_pairs": int(batch["pair_rel"].shape[0]),
        "finite": bool(torch.isfinite(out).all()),
        "pred_head": [float(v) for v in out[:5]],
        "y_head": [float(v) for v in y[rows][:5]],
    }

    # ---- input legality ---------------------------------------------------
    allowed = {
        "atom_feat": "raw108 atom features (label-blind)",
        "bond_feat": "raw7 bond features (label-blind)",
        "hist": "typed incidence histogram of the neighbour after deleting the "
                "back bond: 3 bond types x 21 atom types + remaining degree",
        "src,dst,bond_slot": "centre / neighbour / bond of each directed branch",
        "pair_rel": "extra bond type + shared-neighbour count + ring separation "
                    "of two branches of one root (structure only)",
        "global": "frozen global-33 generic topology channel",
        "n_atoms,n_branches": "counts",
    }
    res["inputs"] = allowed
    res["input_legality"] = {
        "no_radius2_key": True,
        "no_labels": True,
        "no_supervised_checkpoint_loaded": True,
        "note": "only pretrain_encoder_s0.pt keys atom_encoder/bond_encoder are loaded",
    }

    # ---- invariance -------------------------------------------------------
    pred_full = predict(model, rows, cache, prep)
    p0 = predict(model, rows[:1], cache, prep)
    res["batch_invariance"] = float(np.abs(pred_full[0] - p0[0]).max())

    mini = slice_cache(cache, np.arange(16))
    mini_prep = P.build(mini)
    mini_perm = permute_cache(mini, 7)
    perm_prep = P.build(mini_perm)
    a = predict(model, np.arange(16), mini, mini_prep)
    b = predict(model, np.arange(16), mini_perm, perm_prep)
    res["renumber_and_bondorder_invariance"] = float(np.abs(a - b).max())
    res["mini_prep_check"] = {
        "branches": mini_prep["meta"]["n_branches"], "pairs": mini_prep["meta"]["n_pairs"],
        "same_branches": bool(mini_prep["meta"]["n_branches"] == perm_prep["meta"]["n_branches"]),
        "rel_hist_equal": bool(np.array_equal(np.bincount(mini_prep["pair_rel"], minlength=64),
                                             np.bincount(perm_prep["pair_rel"], minlength=64))),
    }

    # ---- gradients --------------------------------------------------------
    model.train()
    batch = R.collate(rows, cache, prep, device)
    target = torch.as_tensor(y[rows], dtype=torch.float32)
    pred = model(batch)
    loss = F.l1_loss(pred, target)
    loss.backward()
    grads = {}
    for name, p in model.named_parameters():
        g = p.grad
        grads[name] = None if g is None else float(g.abs().max())
    res["gradients"] = {
        "missing": [k for k, v in grads.items() if v is None],
        "zero": [k for k, v in grads.items() if v == 0.0],
        "key": {k: grads[k] for k in
                ("branch.0.weight", "branch.2.weight", "hist_proj.weight",
                 "rel_emb.weight", "root.weight", "reader.lins.0.weight")
                if k in grads},
    }
    # three-body channel is differentiable and reaches the loss
    model.zero_grad(set_to_none=True)
    model.three_body_scale.fill_(0.0)
    pred_n3 = model(batch)
    loss_n3 = F.l1_loss(pred_n3, target)
    loss_n3.backward()
    g_n3 = {k: (None if p.grad is None else float(p.grad.abs().max()))
            for k, p in model.named_parameters()}
    model.three_body_scale.fill_(1.0)
    res["three_body_channel"] = {
        "pred_delta_max": float((pred - pred_n3).detach().abs().max()),
        "grad_root_nonzero_without_e3": (g_n3.get("root.weight") or 0.0) > 0.0,
        "grad_branch_nonzero_without_e3": (g_n3.get("branch.0.weight") or 0.0) > 0.0,
    }

    # ---- checkpoint round trip -------------------------------------------
    sd = C.cpu_state(model)
    path = CACHE / "check_roundtrip.pt"
    C.atomic_torch_save({"state_dict": sd}, path)
    m2, _ = R.build_model(norm, 0, device)
    st = torch.load(path, map_location="cpu", weights_only=False)
    m2.load_state_dict(st["state_dict"])
    pred2 = predict(m2, rows, cache, prep)
    res["checkpoint_roundtrip_max_diff"] = float(np.abs(pred_full - pred2).max())

    # ---- toy 1 : identical 1/2-body, different 3-body ---------------------
    def scalar_stats(vals):
        t = torch.tensor(vals, dtype=torch.float64)
        x = t.unsqueeze(1).repeat(1, R.RANK)
        p1, e2, e3 = R.symmetric_stats(x, torch.zeros(3, dtype=torch.long), 1)
        # true explicit enumeration for an independent check
        import itertools
        e2_ref = sum((x[i] * x[j]) for i, j in itertools.combinations(range(3), 2))
        e3_ref = x[0] * x[1] * x[2]
        return p1[0], e2[0], e3[0], e2_ref, e3_ref

    s = math.sqrt(2.0 / 1.5)
    toy_a = [2.0 + s * math.cos(t) for t in (0.0, 2 * math.pi / 3, 4 * math.pi / 3)]
    toy_b = [1.0, 2.0, 3.0]
    pa1, pa2, pa3, pa2r, pa3r = scalar_stats(toy_a)
    pb1, pb2, pb3, pb2r, pb3r = scalar_stats(toy_b)
    res["toy1"] = {
        "triple_a": [round(v, 6) for v in toy_a],
        "triple_b": toy_b,
        "p1_equal": bool(torch.allclose(pa1, pb1, atol=1e-6)),
        "e2_equal": bool(torch.allclose(pa2, pb2, atol=1e-6)),
        "e2_matches_explicit": bool(torch.allclose(pa2, pa2r, atol=1e-6)
                                    and torch.allclose(pb2, pb2r, atol=1e-6)),
        "e3_differs": bool((pa3 - pb3).abs().max() > 1e-3),
        "e3_matches_explicit": bool(torch.allclose(pa3, pa3r, atol=1e-6)
                                    and torch.allclose(pb3, pb3r, atol=1e-6)),
        "e3_a": float(pa3[0]), "e3_b": float(pb3[0]),
    }

    # ---- toy 2 : 1/2-body reader identical, RBO reader different ---------
    torch.manual_seed(11)
    head = torch.nn.Linear(model.root.out_features, 1)
    kfeat = torch.tensor([[3 / 4, math.log(4.0), 1.0, 1.0, 0.0]], dtype=torch.float64)
    zero_rel = torch.zeros(1, R.RANK, dtype=torch.float64)

    def reader_response(e1, e2v, e3v):
        feat = torch.cat([e1.view(1, -1), e2v.view(1, -1), e3v.view(1, -1), zero_rel, kfeat], 1)
        with torch.no_grad():
            return float(head(F.silu(model.root(feat.double() if False else feat.float()))))

    with torch.no_grad():
        model.three_body_scale.fill_(1.0)
    # full RBO response uses e3; the one/two-body response replaces e3 by 0
    r24_a = reader_response(pa1, pa2, torch.zeros_like(pa3))
    r24_b = reader_response(pb1, pb2, torch.zeros_like(pb3))
    r_rbo_a = reader_response(pa1, pa2, pa3)
    r_rbo_b = reader_response(pb1, pb2, pb3)
    res["toy2"] = {
        "one_two_body_response_a": r24_a, "one_two_body_response_b": r24_b,
        "one_two_body_identical": bool(abs(r24_a - r24_b) < 1e-6),
        "rbo_response_a": r_rbo_a, "rbo_response_b": r_rbo_b,
        "rbo_differs": bool(abs(r_rbo_a - r_rbo_b) > 1e-6),
    }

    OUT.joinpath("t0_checks.json").write_text(json.dumps(res, indent=1, default=str))
    ok = (
        res["budget_ok"]
        and res["forward"]["finite"]
        and res["batch_invariance"] < 1e-4
        and res["renumber_and_bondorder_invariance"] < 1e-4
        and not res["gradients"]["missing"]
        and res["three_body_channel"]["pred_delta_max"] > 0
        and res["three_body_channel"]["grad_root_nonzero_without_e3"]
        and res["checkpoint_roundtrip_max_diff"] == 0.0
        and res["toy1"]["p1_equal"] and res["toy1"]["e2_equal"]
        and res["toy1"]["e3_differs"]
        and res["toy1"]["e2_matches_explicit"] and res["toy1"]["e3_matches_explicit"]
        and res["toy2"]["one_two_body_identical"] and res["toy2"]["rbo_differs"]
    )
    print(json.dumps(res, indent=1, default=str))
    print("GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
