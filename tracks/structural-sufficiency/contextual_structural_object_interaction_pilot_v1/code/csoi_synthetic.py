"""Synthetic context-flip sanity task (capability check, not evidence about Lipo).

Two rules are built on the *real* token machinery of the frozen cache:

``R_ADD``   y = a(S)                      -- a purely additive rule: context is irrelevant
``R_FLIP``  y = a(S) * sign(g(C))         -- the same S with a different C flips the
                                             sign of the response, so NO additive model
                                             can fit it while an interaction model can
``R_REL``   y = a(S) * sign(bucket(S) - bucket(C))  -- the flip depends on the *relation*
                                             between S and C distance buckets

The same architecture (frozen hyper-parameters) is trained on each rule for a fixed
step budget with the same optimiser; the only difference is the arm.  The expected
pattern is: ``CSOI_ADD`` fits ``R_ADD`` but cannot fit ``R_FLIP``/``R_REL``, while
``CSOI_GLOBAL``/``CSOI_REL`` fit all three.  This validates that the interaction
mechanism actually conditions on C; it says nothing about the real dataset.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Sequence

import numpy as np

import csoi_common as C
import csoi_data as D
import csoi_model as M


def _features(pack: dict, view: D.TokenView):
    frame_k = np.asarray(pack["frame_k"], np.int64)
    tok_k = frame_k[np.asarray(pack["frag_frame"], np.int64)]
    return tok_k


def _side_features(pack: dict, view: D.TokenView, ids: np.ndarray):
    """(#size-1 objects, mean distance bucket, mean multiplicity) per fragment."""
    starts, counts = view.slices(ids)
    out = np.zeros((ids.size, 4), np.float64)
    for i, (s, c) in enumerate(zip(starts.tolist(), counts.tolist())):
        b = view.frag_bucket[s:s + c].astype(np.float64)
        w = view.frag_weight[s:s + c].astype(np.float64)
        out[i, 0] = float(np.sum(w[b <= 1]))          # near objects
        out[i, 1] = float(np.average(b, weights=w))   # mass-weighted mean bucket
        out[i, 2] = float(w.sum())                    # total mass
        out[i, 3] = float(c)                          # n tokens
    return out


def build_task(pack: dict, view: D.TokenView, n: int, seed: int = 0):
    tok = view.token_count
    s_pool = np.nonzero((tok <= 24) & (tok >= 4))[0]
    c_pool = np.nonzero((tok <= 48) & (tok >= 8))[0]
    rng = np.random.default_rng(seed)
    s_ids = rng.choice(s_pool, n, replace=True)
    c_ids = rng.choice(c_pool, n, replace=True)
    fs = _side_features(pack, view, s_ids)
    fc = _side_features(pack, view, c_ids)
    # additive part: a simple structural summary, centred so that it cannot be
    # captured by a context-independent additive term
    a_s = (fs[:, 0] / 8.0) + 0.5 * (fs[:, 1] - 3.0) / 2.0
    a_s = (a_s - a_s.mean()) / max(1e-9, a_s.std())
    # context sign: balanced split on the context's total object mass (each token
    # carries log(multiplicity), so the pooled representation can read it)
    flip = np.where(fc[:, 2] >= np.median(fc[:, 2]), 1.0, -1.0)
    # relation sign: balanced split on (S bucket - C bucket), i.e. the relative
    # distance of the two sides
    d = fs[:, 1] - fc[:, 1]
    rel_flip = np.where(d >= np.median(d), 1.0, -1.0)
    bond = np.asarray(pack["parse_cut_bond_row"], np.int64)
    bond_choices = np.unique(bond[bond >= 0])
    bond_ids = rng.choice(bond_choices, n, replace=True)
    y = {
        "R_ADD": a_s,
        "R_FLIP": a_s * flip,
        "R_REL": a_s * rel_flip,
    }
    return {"s": s_ids.astype(np.int64), "c": c_ids.astype(np.int64),
            "bond": bond_ids.astype(np.int64), "y": y,
            "flip_balance": {"R_FLIP": float(np.mean(flip > 0)),
                             "R_REL": float(np.mean(rel_flip > 0))},
            "a_s_std": float(a_s.std())}


def run_arm(cfg, pack, atom_dims, bond_dims, arm: str, task: dict, rule: str,
            steps: int, seed: int, device: str = "cpu", log=print) -> dict[str, Any]:
    import torch

    C.set_deterministic(seed)
    model = M.CSOIModel(cfg, atom_dims, bond_dims, arm)
    model.load_pack(pack)
    model = model.float().to(device)
    n = task["s"].size
    n_valid = max(64, n // 8)
    tr = slice(0, n - n_valid)
    va = slice(n - n_valid, n)
    y = task["y"][rule]
    y_tr = (y[tr] - y[tr].mean()) / max(1e-6, float(y[tr].std()))
    y_va_raw = y[va]
    t = {k: torch.as_tensor(v, device=device) for k, v in task.items()
         if isinstance(v, np.ndarray)}
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    bs = 64
    rng = np.random.default_rng(seed + 1)
    idx_all = np.arange(n - n_valid)
    model.train()
    t0 = time.time()
    for step in range(steps):
        sel = rng.choice(idx_all, min(bs, idx_all.size), replace=False)
        f = model.cut_scores(t["s"][sel], t["c"][sel], t["bond"][sel])
        target = torch.as_tensor(y_tr[sel], dtype=f.dtype, device=device)
        loss = torch.mean((f - target) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if log and (step + 1) % max(1, steps // 4) == 0:
            print(f"    {arm:11s} {rule:7s} step {step + 1:4d} loss {float(loss):.4f}")
    model.eval()
    with torch.no_grad():
        f_va = model.cut_scores(t["s"][va], t["c"][va],
                                t["bond"][va]).detach().cpu().numpy()
    pred = f_va * max(1e-6, float(y[tr].std())) + float(y[tr].mean())
    rmse = float(np.sqrt(np.mean((pred - y_va_raw) ** 2)))
    base = float(np.sqrt(np.mean((y_va_raw - y[tr].mean()) ** 2)))
    return {"arm": arm, "rule": rule, "valid_rmse": rmse,
            "valid_rmse_of_mean_predictor": base,
            "normalized_rmse": rmse / max(1e-9, base),
            "corr": C.pearson(pred, y_va_raw),
            "steps": steps, "seconds": time.time() - t0}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lipo")
    ap.add_argument("--n", type=int, default=1536)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="*", default=list(M.CSOI_ARMS))
    ap.add_argument("--rules", nargs="*", default=["R_ADD", "R_FLIP", "R_REL"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    cfg = C.load_config()
    pack = D.ensure_pack(cfg, args.dataset)
    atom_dims, bond_dims = C.official_feature_dims()
    view = D.TokenView(pack)
    task = build_task(pack, view, args.n, seed=args.seed)
    rows = []
    for rule in args.rules:
        for arm in args.arms:
            rows.append(run_arm(cfg, pack, atom_dims, bond_dims, arm, task, rule,
                                args.steps, args.seed, device=args.device))
    table = {(r["arm"], r["rule"]): r for r in rows}
    verdict = {}
    for rule in args.rules:
        add = table.get(("CSOI_ADD", rule))
        inter = [table.get((a, rule)) for a in ("CSOI_GLOBAL", "CSOI_REL")
                 if (a, rule) in table]
        verdict[rule] = {
            "cs_add_normalized_rmse": add["normalized_rmse"] if add else None,
            "best_interaction_normalized_rmse": min(x["normalized_rmse"] for x in inter)
            if inter else None,
            "additive_arm_fails": bool(add and add["normalized_rmse"] > 0.8) if add else None,
            "interaction_arm_succeeds": bool(inter and
                                             min(x["normalized_rmse"] for x in inter) < 0.5),
        }
    out = {"task": "synthetic context-flip sanity task",
           "note": ("capability check on the real token machinery; NOT evidence about "
                    "the Lipo dataset"),
           "n_examples": int(args.n), "steps": int(args.steps), "seed": int(args.seed),
           "device": args.device,
           "flip_balance": task["flip_balance"], "a_s_std": task["a_s_std"],
           "cache_sha256": pack["cache_sha256"],
           "rules": {
               "R_ADD": "y = a(S) (context irrelevant)",
               "R_FLIP": "y = a(S) * sign(g(C)) (same S, different C flips the sign)",
               "R_REL": "y = a(S) * sign(bucket(S) - bucket(C)) (relation flips it)"},
           "rows": rows, "verdict": verdict}
    C.write_json(C.analysis_dir() / "synthetic_flip.json", out)
    for r in rows:
        print(f"  {r['arm']:11s} {r['rule']:7s} norm_rmse={r['normalized_rmse']:.3f} "
              f"corr={r['corr']:+.3f}")
    print(f"-> {C.analysis_dir() / 'synthetic_flip.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
