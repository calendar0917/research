"""Exploration on the *old* supervision-support package (CPU, no new training).

Base: the two ``sup_minus_s{0,1}`` epoch-600 checkpoints of
``zinc_composition_transfer_v1``. Their supervised fit is ``F_minus`` = 7,549
molecules (``B`` + ``D_minus``), which contains **no** molecule carrying any of
the 90 target keys. `H` (801), `Q_background` (650) and `M` (1,000) are all
out-of-sample for these checkpoints.

`H` is split by *key co-occurrence components* (61 components, largest 7 keys /
89 molecules) into correction-fit, correction-dev and query, so no target key is
shared between the three. Everything is structure-only and fixed-RNG.

Historical exposure (recorded, not removable): the old round used `M` for
best-checkpoint selection and read the `Q_target`/`Q_background` labels after its
checkpoints were frozen. This is an exploration, not a confirmation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import zst_common as Z  # noqa: E402
import zst_head as H  # noqa: E402
import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402

ZCT = Z._SSUF / "zinc_composition_transfer_v1"
MINUS = {0: ZCT / "runs/sup_minus_s0/final.pt", 1: ZCT / "runs/sup_minus_s1/final.pt"}
MINUS_SHA = {0: "2267c95d71a4bedb47d78537d3cb81b1e10503250de63e79a2a88001ac0d15ea",
             1: "38f3eaad698e4ede8450f6a019a046ebb3921d6014ecb5cfd65a150a3ff87e02"}


def component_split(seed: int = 20261122) -> dict:
    z = np.load(ZCT / "cache/support_sets.npz")
    tk = np.asarray(z["target_keys"])
    h_rows = np.asarray(z["H"])
    keys = Z.load_keys()
    _, mol2 = Z.molecule_key_sets(keys)
    sets = {m: set(int(k) for k in mol2[m].tolist()) for m in h_rows.tolist()}
    tk2m = {int(k): [m for m in h_rows.tolist() if int(k) in sets[m]] for k in tk}
    par = {int(k): int(k) for k in tk}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    per_mol: dict[int, list[int]] = {}
    for k, ms in tk2m.items():
        for m in ms:
            per_mol.setdefault(m, []).append(k)
    for ks in per_mol.values():
        for o in ks[1:]:
            ra, rb = find(ks[0]), find(o)
            if ra != rb:
                par[rb] = ra
    comps: dict[int, list[int]] = {}
    for k in tk:
        comps.setdefault(find(int(k)), []).append(int(k))
    comp_list = sorted(comps.values(), key=lambda v: (-len(v), v[0]))
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(comp_list))
    roles = {"C_fit": [], "C_dev": [], "Q_old": []}
    used_mols = set()
    for idx in order:
        comp = comp_list[int(idx)]
        mols = sorted({m for k in comp for m in tk2m[k]} - used_mols)
        if not mols:
            continue
        n = len(mols)
        if len(roles["Q_old"]) < 220 and n <= 90:
            roles["Q_old"] += mols
        elif len(roles["C_dev"]) < 110:
            roles["C_dev"] += mols
        else:
            roles["C_fit"] += mols
        used_mols |= set(mols)
    return {"roles": {k: np.asarray(sorted(v), dtype=np.int64) for k, v in roles.items()},
            "n_components": len(comp_list),
            "components": comp_list}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--steps", type=int, default=H.STEPS)
    args = ap.parse_args()
    z = np.load(ZCT / "cache/support_sets.npz")
    fit_rows = np.asarray(z["fit_minus"], dtype=np.int64)
    q_bg = np.asarray(z["q_background"], dtype=np.int64)
    monitor = np.asarray(z["monitor"], dtype=np.int64)
    cs = component_split()
    roles = dict(cs["roles"])
    roles["Q_bg"] = q_bg
    roles["M"] = monitor
    print("[oldpkg] roles", {k: int(len(v)) for k, v in roles.items()},
          "components", cs["n_components"], flush=True)

    cache = D.load_cache("train")
    keys = Z.load_keys()
    mol1, mol2 = Z.molecule_key_sets(keys)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    y = np.asarray(cache["labels"], dtype=np.float64)
    sv = Z.per_atom_support(fit_rows, mol1, mol2, keys,
                            int(keys["n_key1"]), int(keys["n_key2"]))
    sup2, sup1 = sv["sup2"], sv["sup1"]

    union = np.unique(np.concatenate([roles[k] for k in roles]))
    preds = {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        sha = C.sha256_file(MINUS[seed])
        if sha != MINUS_SHA[seed]:
            raise RuntimeError(f"sup_minus_s{seed} hash mismatch: {sha}")
        model, _ = Z.load_base_model(MINUS[seed], seed, torch.device("cpu"))
        preds[seed] = Z.predict_rows(model, cache, union, torch.device("cpu"))
        print(f"[oldpkg s{seed}] predicted {len(union)} molecules", flush=True)
    idx = {int(r): i for i, r in enumerate(union.tolist())}

    out = {"fit_rows": "F_minus 7549", "n_fit": int(len(fit_rows)),
           "support_hash": Z.support_hash(sup1, sup2),
           "components": {"n": cs["n_components"],
                          "sizes": sorted((len(v) for v in cs["components"]), reverse=True)},
           "sizes": {k: int(len(v)) for k, v in roles.items()},
           "historical_exposure": {
               "M_used_for_best_checkpoint_selection": True,
               "Q_target_and_Q_background_labels_read_in_old_round": True,
               "checkpoint": "epoch 600 final (fixed endpoint, not the old best-M checkpoint)"},
           "seeds": {}}
    per_rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        def build(r):
            rr = np.asarray(r, dtype=np.int64)
            sel = np.asarray([idx[int(x)] for x in rr])
            return H.Features(cache, rr, preds[seed][sel], sv["atom_n1"], sv["atom_n2"],
                              atom_ptr), \
                preds[seed][sel], y[rr]

        fit_raw, b_fit, y_fit = build(roles["C_fit"])
        dev_raw, b_dev, y_dev = build(roles["C_dev"])
        q_raw = {k: build(roles[k]) for k in ("Q_old", "Q_bg", "M")}
        std = H.Standardiser(fit_raw)
        data = {"f_cfit": std.transform(fit_raw), "b_cfit": b_fit, "y_cfit": y_fit,
                "f_cdev": std.transform(dev_raw), "b_cdev": b_dev, "y_cdev": y_dev,
                "f_q": {k: std.transform(v[0]) for k, v in q_raw.items()},
                "b_q": {k: v[1] for k, v in q_raw.items()},
                "y_q": {k: v[2] for k, v in q_raw.items()}}
        tkeys = {"Q_old": np.asarray(z["target_keys"], dtype=np.int64), "Q_B": None}
        rec, trained = H.build_and_train(seed, data, tkeys, mol2, steps=args.steps)
        rec.update({"seed": seed, "base_ckpt_sha256": MINUS_SHA[seed],
            "support_hash": Z.support_hash(sup1, sup2),
            "c_fit": {"n": int(len(y_fit)),
                              "n_gated": int((data["f_cfit"].gate > 0).sum()),
                              "base_mae": float(np.abs(b_fit - y_fit).mean())},
                    "c_dev": {"n": int(len(y_dev)),
                              "n_gated": int((data["f_cdev"].gate > 0).sum()),
                              "base_mae": float(np.abs(b_dev - y_dev).mean())}})
        g = data["f_cfit"].gate > 0
        r = y_fit - b_fit
        rec["c_fit_residual_gated"] = {"mean": float(r[g].mean()), "mean_abs": float(np.abs(r[g]).mean())}
        out["seeds"][str(seed)] = rec
        for qname in ("Q_old", "Q_bg", "M"):
            for kind in ("G", "S"):
                for e in rec["arms"][kind]:
                    per_rows.append([seed, qname, kind, e["init"], e["q"][qname]["delta"],
                                     e["q"][qname]["mae_base"], e["q"][qname]["mae_corrected"],
                                     e["q"][qname]["mean_abs_correction"]])
        for qname in ("Q_old", "Q_bg", "M"):
            b = rec["base_metrics"][qname]
            gd = np.mean([e["q"][qname]["delta"] for e in rec["arms"]["G"]])
            sd = np.mean([e["q"][qname]["delta"] for e in rec["arms"]["S"]])
            print(f"[oldpkg s{seed} {qname}] base={b['mae']:.4f} "
                  f"gated={b['n_gated']} G={gd:+.4f} S={sd:+.4f}", flush=True)
    (Z.RUNS / "records").mkdir(parents=True, exist_ok=True)
    # NB: analysis/oldpkg_records.json + oldpkg_cells.csv are the earlier,
    # pre-A2 exploration artifacts and are left untouched (history is not
    # overwritten); this round writes the *_v2 pair next to them.
    Z.write_json(Z.RUNS / "records" / "oldpkg_records_full.json", out)
    import zst_run_head as R
    Z.write_json(Z.ANALYSIS / "oldpkg_records_v2.json",
                 {"seeds": {k: R.compact(v) if "arms" in v else v for k, v in out["seeds"].items()},
                  "components": out.get("components"),
                  "historical_exposure": out.get("historical_exposure")})
    with open(Z.ANALYSIS / "oldpkg_cells_v2.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "package", "head", "init", "delta", "base_mae", "corrected_mae",
                    "mean_abs_correction"])
        w.writerows(per_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
