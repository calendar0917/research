"""Four structural case studies on the real replacement directory.

Each case is selected by an explicit numeric rule so the reader can see that the
examples were not hand-picked:

* ``merge``      the pair's response is far from the sum of the isolated word
                 actions (strong composition / merging),
* ``redirect``   the context flips the sign of a word's action,
* ``composed``   both the cut function and the molecule errors look fine while
                 the pair error is large,
* ``pathological`` one transform inside one cell carries clearly disagreeing
                 labels (a duplicate word row, not a model failure).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

import trc_common as C
import trc_prepare as P


def _cut_terms(model, items: Mapping[str, np.ndarray], with_context: bool):
    """f(S|C) and f(S|empty) for every pair endpoint."""
    import torch

    phi = model.frame_phi()
    out = {}
    with torch.no_grad():
        rows = np.concatenate([items["Sa"], items["Sb"]])
        uniq, inv = torch.unique(torch.as_tensor(rows, device=model.device), sorted=True,
                                 return_inverse=True)
        h = model.encode_fragments(uniq, phi)
        n = items["Sa"].size
        e = model._cut_embedding(torch.as_tensor(items["bond"], device=model.device))
        h_a = h[inv[:n]]
        h_b = h[inv[n:]]
        if with_context:
            ctx = torch.as_tensor(items["C"], device=model.device)
            cuniq, cinv = torch.unique(ctx, sorted=True, return_inverse=True)
            hc = model.encode_fragments(cuniq, phi)[cinv]
        else:
            hc = model.h_empty[None, :].expand(n, -1)
        out["a"] = model.cut_scores(hc, h_a, e).double().cpu().numpy()
        out["b"] = model.cut_scores(hc, h_b, e).double().cpu().numpy()
    return out


def _molecule_scores(model, n_mol: int, chunk: int = 512) -> np.ndarray:
    import torch

    out = np.zeros(n_mol, np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, n_mol, chunk):
            rows = torch.arange(start, min(n_mol, start + chunk), device=model.device)
            out[start:start + rows.numel()] = (model.molecule_forward(rows)["f_bar"]
                                               .double().cpu().numpy())
    return out


def build_cases(pack, cfg, labels, model, row_ids: np.ndarray) -> dict[str, Any]:
    cell = np.asarray(pack["pair_cell"])
    pa = np.asarray(pack["pair_mol_a"], np.int64)
    pb = np.asarray(pack["pair_mol_b"], np.int64)
    transform = np.asarray(pack["pair_transform"], np.int64)
    turnq = np.asarray(pack["transform_word_a"], np.int64)
    tunqb = np.asarray(pack["transform_word_b"], np.int64)
    pc = np.asarray(pack["pair_C"], np.int64)
    sa = np.asarray(pack["pair_Sa"], np.int64)
    sb = np.asarray(pack["pair_Sb"], np.int64)
    bond = np.asarray(pack["pair_cut_bond_row"], np.int64)
    y = np.asarray(labels, float)
    sigma = float(np.std(y[np.asarray(pack["train_idx"], np.int64)]))
    dy = (y[pb] - y[pa]) / sigma
    items = {"a": pa, "b": pb, "C": pc, "Sa": sa, "Sb": sb, "bond": bond}
    ctx = _cut_terms(model, items, True)
    iso = _cut_terms(model, items, False)
    delta_full = ctx["b"] - ctx["a"]
    delta_iso = iso["b"] - iso["a"]
    merge = np.abs(delta_full - delta_iso)
    shift_a = np.abs(ctx["a"] - iso["a"])
    shift_b = np.abs(ctx["b"] - iso["b"])
    split_name = np.asarray(["train", "valid", "test", "cross"])[cell]
    keys = np.asarray(pack["fragment_keys"])
    atoms_ptr = np.asarray(pack["parse_S_atoms_ptr"], np.int64)
    flat_atoms = np.asarray(pack["parse_S_atoms"], np.int64)

    def describe(i: int) -> dict[str, Any]:
        return {
            "pair": int(i), "cell": str(split_name[i]),
            "molecule_a_ogb_id": int(row_ids[pa[i]]),
            "molecule_b_ogb_id": int(row_ids[pb[i]]),
            "context_key": str(keys[pc[i]])[:160],
            "word_a_key": str(keys[sa[i]])[:160],
            "word_b_key": str(keys[sb[i]])[:160],
            "delta_y_sigma": float(dy[i]),
            "delta_full_sigma": float(delta_full[i]),
            "delta_isolated_sigma": float(delta_iso[i]),
            "merge_magnitude": float(merge[i]),
            "context_shift_a": float(shift_a[i]),
            "context_shift_b": float(shift_b[i]),
            "cut_error": float(delta_full[i] - dy[i]),
        }

    held = cell != 0
    out: dict[str, Any] = {"sigma_y": sigma, "n_pairs": int(cell.size),
                           "n_held_out_pairs": int(held.sum())}
    order = np.nonzero(held)[0][np.argsort(merge[held])[::-1]]
    out["merge"] = {
        "rule": "largest |delta_full - delta_isolated| on held-out pairs",
        "picks": [describe(int(i)) for i in order[:3]],
    }
    flip = np.nonzero((np.sign(ctx["a"]) * np.sign(iso["a"]) < 0) & held)[0]
    if flip.size:
        best = flip[np.argsort(shift_a[flip])[::-1][:3]]
        block = []
        for i in best.tolist():
            entry = describe(int(i))
            entry["word_a_with_context"] = float(ctx["a"][i])
            entry["word_a_isolated"] = float(iso["a"][i])
            block.append(entry)
        out["redirect"] = {"rule": "context flips the sign of the word's action",
                           "n_sign_flips": int(flip.size), "picks": block}
    mol_pred = _molecule_scores(model, len(y))
    err_a = np.abs(mol_pred[pa] - y[pa])
    err_b = np.abs(mol_pred[pb] - y[pb])
    pair_err = np.abs(delta_full - dy) * sigma
    good_single = ((err_a <= np.percentile(err_a[held], 40))
                   & (err_b <= np.percentile(err_b[held], 40)) & held)
    bad_pair = pair_err >= np.percentile(pair_err[held], 90)
    cand = np.nonzero(good_single & bad_pair)[0][:3]
    picks = []
    for i in cand.tolist():
        entry = describe(i)
        entry["molecule_a_abs_error"] = float(err_a[i])
        entry["molecule_b_abs_error"] = float(err_b[i])
        entry["pair_delta_abs_error"] = float(pair_err[i])
        picks.append(entry)
    out["composed"] = {
        "rule": "both single molecules predicted well (top 40%), pair delta badly "
                "(top 10%), held-out only",
        "n_candidates": int((good_single & bad_pair).sum()), "picks": picks,
    }
    dup: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (c, t) in enumerate(zip(cell.tolist(), transform.tolist())):
        dup[(c, t)].append(i)
    bad = []
    for (c, t), idxs in dup.items():
        if len(idxs) < 3:
            continue
        vals = dy[idxs]
        bad.append((float(np.std(vals)), c, t, idxs))
    bad.sort(key=lambda x: -x[0])
    out["pathological"] = {
        "rule": "transform repeated >=3 times inside one cell with the largest label spread",
        "picks": [],
    }
    for spread, c, t, idxs in bad[:2]:
        entry = describe(int(idxs[0]))
        entry["cell"] = str(split_name[idxs[0]])
        entry.pop("pair", None)
        entry["repeat_count"] = len(idxs)
        entry["label_delta_spread_sigma"] = float(spread)
        entry["label_deltas_sigma"] = [float(dy[i]) for i in idxs]
        entry["molecule_pairs_ogb"] = [[int(row_ids[pa[i]]), int(row_ids[pb[i]])] for i in idxs]
        entry["transform_words"] = [str(keys[int(turnq[t])])[:120], str(keys[int(tunqb[t])])[:120]]
        out["pathological"]["picks"].append(entry)
    del atoms_ptr, flat_atoms
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--run-id", default="LIPO_FACTOR_CUTRESP_R0")
    args = ap.parse_args(argv)
    cfg = dict(C.load_config())
    cfg["cache_stem"] = "trc_cache"
    pack = P.load_pack(cfg, "lipo")
    labels = np.asarray(pack["labels"], float)
    import trc_analyze as A

    models = A.load_models({args.run_id: {"manifest": {"arm": "FACTOR"}}}, cfg, pack, args.device)
    case = build_cases(pack, cfg, labels, models[args.run_id], np.asarray(pack["row_ids"]))
    C.write_json(C.analysis_dir() / "cases.json", case)
    print(json.dumps(case, indent=1)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
