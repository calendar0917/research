"""SUM edit ledger: exact path-score accounting for a matched pair (v2, geometric split).

The TRC cache's ``parse_*_atoms`` fields are stored in the fragment's local atom
numbering and cannot be mapped back to molecule indices without the parent
round's motif machinery.  The ledger therefore uses the geometry of the pair's
own cut bond instead: removing the cut bond splits the molecule into two
components; the smaller one (ties broken by the lower minimum atom index) is the
``S`` side, the other is the ``C`` side.  Every enumerated path is then
classified as S / C / X by its vertex set.  This is a coarser but unambiguous
partition of the same edit, and the identity ``F = beta + F_S + F_C + F_X``
holds exactly.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np

import psc_common as C
import psc_analysis as A
import psc_data as D
import psc_paths as PP
import psc_train as T


def components(raw: Mapping[str, Any], mol: int, bond_row: int) -> list[set[int]]:
    """Graph components of one molecule after deleting its cut bond."""
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    bu = np.asarray(raw["bond_u"], np.int64)
    bv = np.asarray(raw["bond_v"], np.int64)
    b0, b1 = int(bond_ptr[mol]), int(bond_ptr[mol + 1])
    n = int(node_ptr[mol + 1] - node_ptr[mol])
    cut = b0 + int(bond_row)
    u, v = int(bu[cut]), int(bv[cut])
    adj: list[list[int]] = [[] for _ in range(n)]
    for e in range(b0, b1):
        if e == cut:
            continue
        x, y = int(bu[e]), int(bv[e])
        adj[x].append(y)
        adj[y].append(x)
    comps: list[set[int]] = []
    seen_global = np.zeros(n, bool)
    for start in (u, v):
        if seen_global[start]:
            continue
        seen = np.zeros(n, bool)
        stack = [start]
        seen[start] = True
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if not seen[y]:
                    seen[y] = True
                    stack.append(y)
        comps.append(set(int(i) for i in np.nonzero(seen)[0].tolist()))
        seen_global |= seen
    return comps


def align_sides(comps_a: list[set[int]], comps_b: list[set[int]]
               ) -> tuple[tuple[set[int], set[int]], tuple[set[int], set[int]]] | None:
    """Identify the shared C side by equal component size (the TRC pairs share C)."""
    if len(comps_a) != 2 or len(comps_b) != 2:
        return None
    sizes_a = [len(x) for x in comps_a]
    sizes_b = [len(x) for x in comps_b]
    matches = [(i, j) for i in range(2) for j in range(2) if sizes_a[i] == sizes_b[j]]
    if len(matches) != 1:
        return None
    i, j = matches[0]
    C_a, S_a = comps_a[i], comps_a[1 - i]
    C_b, S_b = comps_b[j], comps_b[1 - j]
    if not S_a or not S_b:
        return None
    return (S_a, C_a), (S_b, C_b)


def classify(paths: Sequence[Sequence[int]], scores: np.ndarray, plen: np.ndarray,
             c_l: np.ndarray, S: set[int], Cc: set[int]) -> dict[str, float]:
    out = {"S": 0.0, "C": 0.0, "X": 0.0}
    for seq, sc, l in zip(paths, scores, plen):
        vs = set(int(x) for x in seq)
        w = float(c_l[int(l)])
        if vs <= S:
            out["S"] += w * float(sc)
        elif vs <= Cc:
            out["C"] += w * float(sc)
        else:
            out["X"] += w * float(sc)
    return out


def build(bundle: D.DataBundle, lock: Mapping[str, Any], cfg: Mapping[str, Any],
          n_examples_per_rule: int = 2, device: str = "cpu") -> dict[str, Any]:
    champ = lock["champion"]
    target = champ if A.ARM_SPEC[champ]["readout"] == "SUM" else lock["R_star"]
    rid = A.run_id(target, "D0", 0)
    model, cache, spec, state = A.load_psc_model(rid, bundle, cfg, device)
    c_l = np.asarray(state["c_l"], np.float64)
    scaler = state["scaler"]
    L = cache.L
    split = C.split_spec("D0", bundle.raw)
    pd = A.PairData(bundle, split)
    acra = lock["ACRA_B_star"]
    ref_preds = []
    for seed in (0, 1, 2):
        d = A.prediction_dict(acra, "D0", seed)
        if d:
            ref_preds.append(d)
    if not ref_preds:
        raise SystemExit(f"ledger: no predictions for {acra}")
    delta_true = bundle.labels[pd.pb] - bundle.labels[pd.pa]
    # sanity: the ledger reproduces the forward pass exactly
    held = np.asarray(split["eval_rows"], np.int64)
    max_err = 0.0
    for row in held[:12].tolist():
        paths, atoms, bonds, keys = A.enumerate_row(cache, bundle.raw, row, L)
        s = A.score_paths(model, atoms, bonds, L, device, spec["input"] == "BAG")
        plen = np.array([len(p) - 1 for p in paths])
        ledger = float(model.beta) + float(np.sum(c_l[plen] * s))
        direct = A._predict_one(model, cache, c_l, int(row), spec["input"] == "BAG")
        max_err = max(max_err, abs(ledger - direct))
    if max_err > 1e-3:
        raise AssertionError(f"SUM ledger identity failed: {max_err}")
    # candidate pairs with the predefined attributes
    cut_lists = pd.cut_rows_of_pair()
    trc = A.load_trc()
    t_seen = set(np.unique(pd.dir.T_id[pd.cut_scope == 0]).tolist())
    frag_seen = A.fit_fragment_set(pd.dir, split["fit_rows"], trc)
    cache_rows = np.union1d(bundle.official_train, bundle.official_valid)
    counts = dict(zip(cache_rows.tolist(), A.path_count_dicts(cache, cache_rows)))
    vocab = set(D.fit_path_vocab(cache, split["fit_rows"], L).ids)
    cand = []
    for i in np.nonzero(pd.scope == 2)[0].tolist():
        a, b = int(pd.pa[i]), int(pd.pb[i])
        cuts = cut_lists[i]
        bond0 = int(pd.dir.bond_id[int(cuts[0])])
        al = align_sides(components(bundle.raw, a, bond0),
                         components(bundle.raw, b, bond0))
        min_side = None if al is None else min(len(al[0][0]), len(al[0][1]))
        dp = float(np.mean([np.mean([d[b] - d[a] for d in ref_preds])]))
        da, db = counts[a], counts[b]
        keys = set(da) | set(db)
        num = sum(abs(db.get(k, 0) - da.get(k, 0)) for k in keys if k not in vocab)
        den = sum(abs(db.get(k, 0) - da.get(k, 0)) for k in keys)
        cand.append({
            "pair": i, "a": a, "b": b, "cut_row": int(cuts[0]),
            "delta_true": float(delta_true[i]), "delta_ref_pred": dp,
            "abs_err_ref": float(abs(dp - delta_true[i])),
            "eta": (num / den) if den > 0 else None,
            "alignable": al is not None, "min_side": min_side,
            "big_sides": bool(min_side is not None and min_side >= 3),
            "T_seen": any(int(pd.dir.T_id[c]) in t_seen for c in cuts.tolist()),
            "frag_seen": all(int(pd.dir.word_a[c]) in frag_seen
                             and int(pd.dir.word_b[c]) in frag_seen for c in cuts.tolist()),
        })
    cand = [c for c in cand if np.isfinite(c["abs_err_ref"])]
    if not cand:
        raise SystemExit("ledger: no candidates")
    etas = [c["eta"] for c in cand if c["eta"] is not None]
    high_eta = float(np.quantile(etas, 0.75))
    rules = cfg["analysis"]["case_rules"]
    picked: list[dict[str, Any]] = []
    seen: set[int] = set()

    def take(pred, key, n, why):
        items = [c for c in cand if c["pair"] not in seen and c["alignable"]
                 and c["big_sides"] and pred(c)]
        if not items:
            items = [c for c in cand if c["pair"] not in seen and c["alignable"]
                     and pred(c)]
        items.sort(key=key)
        for c in items[:n]:
            seen.add(c["pair"])
            picked.append({**c, "rule": why})

    unseen_frag_seen = lambda c: (not c["T_seen"]) and c["frag_seen"]
    take(unseen_frag_seen, lambda c: -c["abs_err_ref"],
         rules["t_unseen_frag_seen_improved"],
         "T unseen / fragments seen · largest ACRA error")
    take(unseen_frag_seen, lambda c: c["abs_err_ref"],
         rules["t_unseen_frag_seen_degraded"],
         "T unseen / fragments seen · smallest ACRA error")
    take(lambda c: c["eta"] is not None and c["eta"] >= high_eta,
         lambda c: -c["abs_err_ref"], rules["high_eta_improved"],
         "high eta · largest ACRA error")
    take(lambda c: c["eta"] is not None and c["eta"] >= high_eta,
         lambda c: c["abs_err_ref"], rules["high_eta_degraded"],
         "high eta · smallest ACRA error")
    take(lambda c: c["T_seen"] and c["delta_true"] > 0, lambda c: -c["delta_true"],
         rules["seen_t_positive"], "seen T · largest positive Δ")
    take(lambda c: c["T_seen"] and c["delta_true"] < 0, lambda c: c["delta_true"],
         rules["seen_t_negative"], "seen T · most negative Δ")
    out: dict[str, Any] = {
        "generated_utc": C.now_utc(), "ledger_model": target, "champion": champ,
        "reference": acra, "n_candidates": len(cand),
        "n_alignable": sum(1 for c in cand if c["alignable"]),
        "n_two_sided": sum(1 for c in cand if c["big_sides"]),
        "high_eta_quantile": high_eta,
        "rule_book": rules, "split_convention": (
            "the TRC pair shares its context side; after deleting the cut bond in both "
            "molecules the shared context is the component pair with equal size, the "
            "remaining components are the S sides; paths entirely inside S / C are "
            "labelled S / C, paths touching both are X"),
        "identity_max_abs_std": float(max_err),
        "note": ("exact SUM ledger" if target == champ else
                 "the champion uses POOL, so the ledger is reported for its matched SUM "
                 "twin and does not explain the champion's own predictions"),
        "examples": [],
    }
    for c in picked:
        bond = int(pd.dir.bond_id[c["cut_row"]])
        comps_a = components(bundle.raw, c["a"], bond)
        comps_b = components(bundle.raw, c["b"], bond)
        aligned = align_sides(comps_a, comps_b)
        if aligned is None:
            out["examples"].append({**{k: c[k] for k in ("pair", "a", "b", "delta_true",
                                                          "delta_ref_pred", "eta", "T_seen",
                                                          "frag_seen", "rule")},
                                    "decomposition": "skipped: ambiguous side alignment"})
            continue
        (S_a, C_a), (S_b, C_b) = aligned
        per_side = {}
        for key, row, S, Cc in (("a", c["a"], S_a, C_a), ("b", c["b"], S_b, C_b)):
            paths, atoms, bonds, _ = A.enumerate_row(cache, bundle.raw, row, L)
            scores = A.score_paths(model, atoms, bonds, L, device,
                                   spec["input"] == "BAG")
            plen = np.array([len(p) - 1 for p in paths])
            sums = classify(paths, scores, plen, c_l, S, Cc)
            per_side[key] = {"row": row, "n_paths": int(len(paths)),
                             "n_S": int(sum(1 for p in paths if set(map(int, p)) <= S)),
                             "n_C": int(sum(1 for p in paths if set(map(int, p)) <= Cc)),
                             "sums": sums, "sides": (len(S), len(Cc))}
        dS = per_side["b"]["sums"]["S"] - per_side["a"]["sums"]["S"]
        dC = per_side["b"]["sums"]["C"] - per_side["a"]["sums"]["C"]
        dX = per_side["b"]["sums"]["X"] - per_side["a"]["sums"]["X"]
        out["examples"].append({
            **{k: c[k] for k in ("pair", "a", "b", "delta_true", "delta_ref_pred",
                                 "eta", "T_seen", "frag_seen", "rule")},
            "sides_a": {"S": per_side["a"]["sides"][0], "C": per_side["a"]["sides"][1]},
            "sides_b": {"S": per_side["b"]["sides"][0], "C": per_side["b"]["sides"][1]},
            "aligned_by_component_size": True,
            "n_paths": {"a": per_side["a"]["n_paths"], "b": per_side["b"]["n_paths"]},
            "path_count_share": {
                "S": (per_side["a"]["n_S"] + per_side["b"]["n_S"])
                / max(1, per_side["a"]["n_paths"] + per_side["b"]["n_paths"]),
                "C": (per_side["a"]["n_C"] + per_side["b"]["n_C"])
                / max(1, per_side["a"]["n_paths"] + per_side["b"]["n_paths"]),
            },
            "delta_F_S": dS, "delta_F_C": dC, "delta_F_X": dX,
            "delta_F_total": dS + dC + dX,
            "delta_pred_units": (dS + dC + dX) * scaler["std"],
            "beta": float(model.beta), "label_std": float(scaler["std"]),
        })
    return out


def main() -> int:
    bundle = D.load_bundle()
    cfg = C.load_config()
    lock = C.read_json(A.ANALYSIS / "selection_lock.json")
    out = build(bundle, lock, cfg)
    C.write_json(A.ANALYSIS / "sum_edit_ledger.json", out)
    print(json.dumps({"ledger_model": out["ledger_model"], "examples": len(out["examples"]),
                      "identity": out["identity_max_abs_std"]}, indent=1))
    for e in out["examples"]:
        if "delta_F_total" not in e:
            print(f"{e['rule'][:38]:38s} a={e['a']:4d} b={e['b']:4d} (unaligned)")
            continue
        print(f"{e['rule'][:38]:38s} a={e['a']:4d} b={e['b']:4d} "
              f"Δtrue={e['delta_true']:+.3f} Δpred={e['delta_F_total'] * e['label_std']:+.3f} "
              f"S/C/X={e['delta_F_S'] * e['label_std']:+.3f}/"
              f"{e['delta_F_C'] * e['label_std']:+.3f}/"
              f"{e['delta_F_X'] * e['label_std']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())