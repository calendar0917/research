"""CPU-only checks for zinc-recipe-trunk-v2 (target: < ~10 minutes).

1. ``closes`` walk-feature semantics: the v1 expression vs the corrected one on
   two tiny walks, plus the real definition used by ``PathModel``.
2. Corrected P0/P1 checkpoints: 32-walk vs 128-walk predictions and the
   repeat-the-same-tokens normalisation invariance on fixed molecules.  This
   decides whether the split-scale sensitivity is gone and whether sampling
   noise remains; it does **not** trigger a GPU rerun.
3. ``P3N`` (the harness reimplementation used for the recipe cross) vs the
   original ``BJOC`` on the same frozen checkpoint and molecules.
4. New trunks: parameter counts, real forward/backward, gradient coverage,
   cross-graph isolation on ``S`` (attention) and ``M``.
5. Masked-view leak check: changing the *hidden* raw type at a masked position
   must not change any feature the masked branch sees.

Writes ``analysis/cpu_checks.json`` and prints a compact summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_models as M1  # noqa: E402
import ad_train as T1  # noqa: E402
import ad2_models as A2  # noqa: E402

EXP2 = _HERE.parent
ANALYSIS2 = EXP2 / "analysis"
N33 = D.N_ATOM_TYPES


def _norm():
    return D.load_norm()


# ---------------------------------------------------------------------------
# 1. closes semantics
# ---------------------------------------------------------------------------


def check_closes() -> dict:
    # propane C-C-C, walk visits three *different* atoms of the same type
    wa = torch.tensor([[6, 6, 6], [6, 6, 6]], dtype=torch.long)
    anon = torch.tensor([[0, 1, 2], [0, 1, 0]], dtype=torch.long)
    L = torch.tensor([2, 2], dtype=torch.long)
    T = wa.shape[1]
    pos = torch.arange(T).view(1, T)
    valid = pos <= L.view(-1, 1)
    old = (wa == wa[:, :1]) & (pos > 0) & valid
    new = M1.path_token_scalars(wa, anon, L)[..., 3].bool()
    return {
        "walks": {
            "0": "propane 0-1-2 : three distinct atoms, same element, no return",
            "1": "ethane 0-1-0 : one genuine return to the start atom",
        },
        "old_closes_type_identity": old.tolist(),
        "corrected_closes_return_to_start": new.tolist(),
        "old_expression": "closes = (wa == wa[:, :1]) & (pos > 0) & valid",
        "corrected_expression": "closes = (wanon == 0) & (pos > 0) & valid",
        "old_wrong_on_walk0": bool(old[0].any().item()),
        "corrected_right": bool(not new[0].any().item() and new[1].sum().item() == 1),
        "note": (
            "The two walks differ only in whether the third token revisits the "
            "start atom.  The old flag fired on same-type first visits; the "
            "corrected flag marks only the genuine return.  Old P0/P1 runs are "
            "kept unchanged and are not renamed or re-reported as the same model."
        ),
    }


# ---------------------------------------------------------------------------
# 2. corrected P0/P1 readout: 32 vs 128 walks, and repetition invariance
# ---------------------------------------------------------------------------


def _predict_paths(model, cache, rows, cols, *, valid: bool) -> torch.Tensor:
    """The checkpoint's real prediction (P1 includes its attention layer)."""
    if cols is None:
        batch = T1.collate(cache, rows, "P0", valid=valid)
    else:
        batch = T1.collate(cache, rows, "P0", valid=False, walk_cols=cols)
    with torch.no_grad():
        return model(batch)


def _duplicate_walks(batch: dict, repeat: int = 2) -> dict:
    S = int(batch["S"])
    B = int(batch["n_graphs"])
    out = dict(batch)
    for key in ("walk_atom", "walk_bond", "walk_anon", "walk_len", "walk_group", "walk_graph", "walk_local"):
        v = batch[key]
        shape = v.shape
        v = v.reshape(B, S, *shape[1:])
        v = v.repeat_interleave(repeat, dim=1)
        out[key] = v.reshape(B * S * repeat, *shape[1:])
    out["S"] = S * repeat
    return out


def check_path_readout() -> dict:
    norm = _norm()
    valid_cache = D.load_cache("valid")
    train_cache = D.load_cache("train")
    rows = np.arange(8, dtype=np.int64)
    out = {"molecules": 8, "splits": {}, "repeat_invariance": {}}
    for tag, candidate in (("P0", "P0"), ("P1", "P1")):
        path = C.RUNS / f"corr_{tag}_s0" / "best.pt"
        if not path.exists():
            out["splits"][tag] = {"status": f"missing {path}"}
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model, _ = M1.build_model(candidate, norm, 0, "cpu")
        C.load_state(model, ckpt["state_dict"], "cpu")
        model.eval()
        gen = torch.Generator().manual_seed(0)
        cols32 = T1._walk_cols(gen)
        entry = {"checkpoint": str(path.relative_to(C.REPO)), "epoch": int(ckpt["epoch"]),
                 "best_valid": float(ckpt["valid_mae"])}
        for split, cache in (("valid", valid_cache), ("train", train_cache)):
            p32 = _predict_paths(model, cache, rows, cols32, valid=False)
            p128 = _predict_paths(model, cache, rows, None, valid=True)
            y = torch.as_tensor(cache["labels"][rows])
            entry[split] = {
                "pred32": p32.tolist(),
                "pred128": p128.tolist(),
                "label": y.tolist(),
                "mae32_on_these_rows": float((p32 - y).abs().mean()),
                "mae128_on_these_rows": float((p128 - y).abs().mean()),
                "mean_abs_pred_shift_32_to_128": float((p32 - p128).abs().mean()),
                "max_abs_pred_shift_32_to_128": float((p32 - p128).abs().max()),
            }
            batch = T1.collate(cache, rows, "P0", valid=False, walk_cols=cols32)
            batch2 = _duplicate_walks(batch, 2)
            with torch.no_grad():
                h1 = model.encode_paths(batch)
                y1 = model.readout_paths(h1, batch)
                h2 = model.encode_paths(_to_cpu(batch2))
                y2 = model.readout_paths(h2, batch2)
            entry[f"{split}_repeat2_max_abs_diff"] = float((y1 - y2).abs().max())
        out["splits"][tag] = entry
    return out


def _to_cpu(batch):
    return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# 3. P3N vs the original BJOC
# ---------------------------------------------------------------------------


def check_p3_equivalence() -> dict:
    _perf = _V1.parents[1] / "zinc_perf_protocol_v1" / "code"
    if str(_perf) not in sys.path:
        sys.path.insert(0, str(_perf))
    import zpp_p3  # noqa: F401  (sets sys.path for the P3 modules)
    import esr_models as E
    import structural_relation_objects_v1 as sro

    prep = C.read_json(C.P3_PREP)
    bjoc = E.build_arm("B", 0, prep, None, 0.0, torch.device("cpu"))
    ckpt = torch.load(C.REPO / "tracks/structural-sufficiency/zinc_perf_protocol_v1/runs/p3_s0/best.pt",
                      map_location="cpu", weights_only=False)
    bjoc.load_state_dict(ckpt["state_dict"])
    bjoc.eval()
    norm = _norm()
    p3n, _ = A2.build_model("P3N", norm, 0, "cpu")
    p3n.eval()
    p3n.load_bjoc_state(ckpt["state_dict"])

    objs_valid = D.load_p3_objects("valid")[0]
    cache_valid = D.load_cache("valid")
    rows = np.arange(16, dtype=np.int64)
    max_diff = 0.0
    with torch.no_grad():
        for start in range(0, len(rows), 8):
            chunk = rows[start:start + 8]
            b = sro.batch_objects([objs_valid[int(i)] for i in chunk])
            p_orig = bjoc(b).detach().numpy()
            batch = T1.collate(cache_valid, chunk, "T")
            p_new = p3n(batch).detach().numpy()
            max_diff = max(max_diff, float(np.abs(p_orig - p_new).max()))
    return {
        "checkpoint": "tracks/structural-sufficiency/zinc_perf_protocol_v1/runs/p3_s0/best.pt",
        "epoch": int(ckpt["epoch"]),
        "molecules": int(len(rows)),
        "max_abs_prediction_diff_bjoc_vs_p3n": max_diff,
        "parameter_count_p3n": C.parameter_count(p3n),
        "parameter_count_bjoc": C.parameter_count(bjoc),
        "parameter_count_equal": C.parameter_count(p3n) == C.parameter_count(bjoc) == 413505,
        "note": (
            "Same frozen checkpoint, same molecules, original batcher vs the v1 "
            "cached batch layout.  Difference is float32 reassociation only."
        ),
    }


# ---------------------------------------------------------------------------
# 4. new trunks: params, forward/backward, gradients, cross-graph isolation
# ---------------------------------------------------------------------------


def check_trunks() -> dict:
    norm = _norm()
    cache = D.load_cache("valid")
    rows = np.arange(8, dtype=np.int64)
    out = {}
    batch = T1.collate(cache, rows, "M")
    for cand in ("M", "S", "Maux", "Saux"):
        model, info = A2.build_model(cand, norm, 0, "cpu")
        model.train()
        pred = model(batch)
        loss = pred.pow(2).mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        n_params = C.parameter_count(model)
        entry = {
            "parameters": n_params,
            "within_cap": n_params <= 500_000,
            "n_parameter_tensors": sum(1 for _ in model.parameters()),
            "grad_tensors": len(grads),
            "all_grad_finite": bool(all(torch.isfinite(g).all().item() for g in grads)),
            "nonzero_grad_tensors": int(sum(1 for g in grads if torch.count_nonzero(g).item() > 0)),
            "ssl_init": info["ssl"]["source"] if info["ssl"] else None,
            "forward_finite": bool(torch.isfinite(pred).all().item()),
            "pred_shape": list(pred.shape),
        }
        if cand in ("M", "S"):
            # cross-graph isolation: perturb one molecule, the other must not move
            model.eval()
            with torch.no_grad():
                base = model(batch)
                b2 = dict(batch)
                b2["atom_feat"] = batch["atom_feat"].clone()
                atoms0 = int(cache["atom_ptr"][rows[0]])
                atoms1 = int(cache["atom_ptr"][rows[0] + 1])
                b2["atom_feat"][atoms0:atoms1] += 0.5
                pert = model(b2)
            entry["cross_graph_isolation_max_abs_diff_other_molecules"] = float(
                (base[1:] - pert[1:]).abs().max()
            )
            entry["cross_graph_perturbation_self_diff"] = float((base[0] - pert[0]).abs().max())
            # batch-size invariance: same molecule alone vs inside the batch
            with torch.no_grad():
                single = model(T1.collate(cache, rows[:1], "M"))
            entry["batch_vs_single_max_abs_diff"] = float((base[0] - single[0]).abs().max())
        out[cand] = entry
    return out


# ---------------------------------------------------------------------------
# 5. masked-view leak check
# ---------------------------------------------------------------------------


def check_masked_view(visits: int = 4) -> dict:
    import ad2_masked as MK

    sro, mp, cf = MK._modules()
    ds = cf.load_official_train()
    bases, _ = D.load_p3_objects("train")
    i = 3
    base = bases[i]
    d = ds[i]
    x = d.x.reshape(-1).long().numpy()
    ei = d.edge_index.numpy()
    ea = d.edge_attr.reshape(-1).long().numpy()
    s0 = mp.build_masked_sample(base, x, ei, ea, seed=MK.MASK_SEED, visit_index=0)
    x2 = x.copy()
    x2[s0.atom_ids[0]] = (int(x2[s0.atom_ids[0]]) + 5) % 27  # change the *hidden* truth
    s1 = mp.build_masked_sample(base, x2, ei, ea, seed=MK.MASK_SEED, visit_index=0)
    same = {
        "atom_feat": bool(np.array_equal(s0.masked.atom_feat, s1.masked.atom_feat)),
        "bond_feat": bool(np.array_equal(s0.masked.bond_feat, s1.masked.bond_feat)),
        "junc_feat": bool(np.array_equal(s0.masked.junc_feat, s1.masked.junc_feat)),
        "global_feat": bool(np.array_equal(s0.masked.global_feat, s1.masked.global_feat)),
        "target_changed": bool(int(s0.atom_types[0]) != int(s1.atom_types[0])),
    }
    cache = MK.load(visits) if MK.cache_path(visits).exists() else None
    return {
        "molecule": int(i),
        "masked_tensors_unchanged_when_hidden_truth_changes": same,
        "pass": bool(all(v for k, v in same.items() if k != "target_changed") and same["target_changed"]),
        "masked_cache_present": cache is not None,
        "note": (
            "The masked object is rebuilt from the masked raw graph, so the "
            "masked branch cannot read the hidden type from any feature field."
        ),
    }


def main() -> int:
    out = {
        "closes": check_closes(),
        "path_readout": check_path_readout(),
        "p3n_equivalence": check_p3_equivalence(),
        "trunks": check_trunks(),
        "masked_view": check_masked_view(),
    }
    ANALYSIS2.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS2 / "cpu_checks.json", out)
    print("closes:", out["closes"]["old_closes_type_identity"], "->",
          out["closes"]["corrected_closes_return_to_start"])
    for tag, e in out["path_readout"].get("splits", {}).items():
        if "valid" in e:
            print(tag, "valid mean|32-128| =", round(e["valid"]["mean_abs_pred_shift_32_to_128"], 5),
                  "repeat2 max diff =", e["valid_repeat2_max_abs_diff"])
    print("p3n vs bjoc max diff:", out["p3n_equivalence"]["max_abs_prediction_diff_bjoc_vs_p3n"])
    for cand, e in out["trunks"].items():
        print(cand, e["parameters"], "params; isolation",
              e.get("cross_graph_isolation_max_abs_diff_other_molecules"))
    print("masked view pass:", out["masked_view"]["pass"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
