"""§5 conditional check: apply the frozen shared head to the full-10k E trunks.

Only runs if the internal result in §4 is positive. Reuses the previous round's
``E_s0`` / ``E_s1`` epoch-600 frozen checkpoints (full-10k supervised, same
architecture, features and SSL recipe — verified from their ``train.json``), and
evaluates on **official valid only**. The head weights are frozen; the support
statistics are recomputed from the real 10k supervised fit; nothing is selected
on valid.

The valid label read is recorded once (sha256 of the array) and reused by every
checkpoint of this round.
"""

from __future__ import annotations

import argparse
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
E_CKPT = {0: ZCT / "runs/E_s0/final.pt", 1: ZCT / "runs/E_s1/final.pt"}
E_SHA = {0: "4be8bce56a30a03cf10fbddf7e344a5a108b8bf314c46b250168b31f70d54681",
         1: "b8348456a447aadf090b857770127e2723c6081b0e4319240b2ec3c7b703c3b4"}
REF_GROUPS = Z._SSUF / "zinc_restricted_adaptation_v1" / "analysis" / "reference_preds_valid.csv"


def four_groups():
    import csv

    with open(REF_GROUPS, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    frac = np.asarray([float(r["r2_unseen_atom_frac"]) for r in rows])
    tail = np.asarray([int(r["tail_y_le_minus3"]) for r in rows])
    unseen = frac > 0
    names = np.where(unseen & (tail == 0), "unseen_ordinary",
                     np.where(unseen & (tail == 1), "unseen_tail",
                              np.where(tail == 1, "seen_tail", "seen_ordinary")))
    return names, np.asarray([float(r["y_raw"]) for r in rows])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--heads", default=str(Z.ANALYSIS / "head_records.json"))
    ap.add_argument("--out", default=str(Z.ANALYSIS / "full10k_valid.json"))
    args = ap.parse_args()

    valid = D.load_cache("valid")
    y_valid = np.asarray(valid["labels"], dtype=np.float64)
    valid_y_sha = __import__("hashlib").sha256(y_valid.tobytes()).hexdigest()
    np.save(Z.ANALYSIS / "valid_y_read_once.npy", y_valid)
    rows_valid = np.arange(len(y_valid), dtype=np.int64)

    keys = Z.load_keys()
    meta = json.load(open(ZCT / "cache/keys_meta.json", encoding="utf-8"))
    atom_start_valid = int(meta["slices"]["valid"]["atom_start"])
    atom_k1, atom_k2, _ = Z.atom_tables(keys, int(keys["n_key2"]))
    train_cache = D.load_cache("train")
    mol1, mol2 = Z.molecule_key_sets(keys)
    sup10k = Z.key_support(np.arange(Z.N_TRAIN), mol2, int(keys["n_key2"])).astype(np.float64)
    names, ref_y = four_groups()
    if not np.allclose(ref_y, y_valid, atol=1e-5):
        raise RuntimeError("valid labels disagree with the frozen reference CSV")

    heads = json.load(open(args.heads, encoding="utf-8"))
    per_rows = []
    out = {"valid_label_sha256": valid_y_sha, "n_valid": int(len(y_valid)),
           "support_hash_10k": Z.support_hash(sup10k), "seeds": {}}
    per_rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        rec = heads[str(seed)]
        ckpt = E_CKPT[seed]
        sha = C.sha256_file(ckpt)
        if sha != E_SHA[seed]:
            raise RuntimeError(f"E_s{seed} checkpoint hash mismatch: {sha}")
        train_rec = json.load(open(ckpt.parent / "train.json", encoding="utf-8"))
        model, _ = Z.load_base_model(ckpt, seed, torch.device("cpu"))
        base = Z.predict_rows(model, valid, rows_valid, torch.device("cpu"))
        feats = H.Features(valid, rows_valid, base, sup10k, atom_k1, atom_k2,
                           np.asarray(valid["atom_ptr"], dtype=np.int64),
                           atom_offset=atom_start_valid)
        entry = {
            "seed": seed, "base_ckpt": str(ckpt), "base_ckpt_sha256": sha,
            "base_fit": "official train 10,000 (supervised)", "n_fit": 10000,
            "base_recipe": {k: train_rec.get(k) for k in
                            ("parameters", "ssl_encoder_sha256", "fit", "arm")},
            "base_metrics": {"mae": float(np.abs(base - y_valid).mean())},
            "n_gated": int((feats.gate > 0).sum()),
            "gate_equals_reference_unseen": bool(np.array_equal(
                feats.gate > 0,
                (names == "unseen_ordinary") | (names == "unseen_tail"))),
        }
        # four old groups
        for g in ("seen_ordinary", "unseen_ordinary", "seen_tail", "unseen_tail"):
            m = names == g
            entry.setdefault("groups", {})[g] = {
                "n": int(m.sum()), "base_mae": float(np.abs(base[m] - y_valid[m]).mean())}
        entry["base_gated_mae"] = float(np.abs(base[feats.gate > 0] - y_valid[feats.gate > 0]).mean())
        entry["base_ungated_mae"] = float(np.abs(base[feats.gate == 0] - y_valid[feats.gate == 0]).mean())
        seed_preds = {}
        for kind in ("G", "S"):
            heads_list = rec["arms"][kind]
            preds = []
            for h in heads_list:
                sd = torch.load(h["checkpoint"], map_location="cpu", weights_only=False)
                model_h = H.HeadS() if kind == "S" else H.HeadG()
                model_h.load_state_dict(sd["state_dict"])
                # the frozen standardiser comes from the head's own base fit
                std = H.Standardiser.__new__(H.Standardiser)
                for k, v in sd["standardiser"].items():
                    setattr(std, {"ctx_mean": "ctx_mu", "ctx_scale": "ctx_sd",
                                  "aux_mean": "aux_mu", "aux_scale": "aux_sd",
                                  "atom_mean": "at_mu", "atom_scale": "at_sd"}[k],
                            np.asarray(v))
                hcorr = H.predict_head(model_h, std.transform(feats))
                preds.append(base + hcorr)
            seed_preds[kind] = preds
            pm = np.mean(preds, axis=0)
            ga = feats.gate > 0
            entry.setdefault("heads", {})[kind] = {
                "mae": float(np.abs(pm - y_valid).mean()),
                "delta": float(np.abs(pm - y_valid).mean() - np.abs(base - y_valid).mean()),
                "mae_by_init": [float(np.abs(p - y_valid).mean()) for p in preds],
                "delta_by_init": [float(np.abs(p - y_valid).mean() - np.abs(base - y_valid).mean())
                                  for p in preds],
                "mean_abs_correction": float(np.abs(pm - base).mean()),
                "changed_fraction": float((np.abs(pm - base) > 1e-9).mean()),
                "gated_delta": float((np.abs(pm[ga] - y_valid[ga])
                                      - np.abs(base[ga] - y_valid[ga])).mean()),
                "groups": {g: float(np.abs(pm[names == g] - y_valid[names == g]).mean())
                           for g in ("seen_ordinary", "unseen_ordinary", "seen_tail", "unseen_tail")},
                "identity_max_abs_diff_ungated": float(np.max(np.abs(pm[~ga] - base[~ga]))),
                "corr_with_residual": float(np.corrcoef(pm - base, y_valid - base)[0, 1]),
            }
        gmean = np.mean(seed_preds["G"], axis=0)
        smean = np.mean(seed_preds["S"], axis=0)
        for i in range(len(y_valid)):
            per_rows.append([seed, i, names[i], int(feats.gate[i] > 0), float(y_valid[i]),
                             float(base[i]), float(gmean[i]), float(smean[i]),
                             entry["heads"]["G"]["delta_by_init"][0] * 0.0])
        out["seeds"][str(seed)] = entry
        print(f"[full10k s{seed}] base={entry['base_metrics']['mae']:.4f} "
              f"gated={entry['n_gated']} "
              f"G={entry['heads']['G']['mae']:.4f} ({entry['heads']['G']['delta']:+.4f}) "
              f"S={entry['heads']['S']['mae']:.4f} ({entry['heads']['S']['delta']:+.4f})", flush=True)
    import csv
    with open(Z.ANALYSIS / "full10k_valid_per_molecule.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "molecule_id", "group", "gated", "y", "base_pred", "G_pred", "S_pred"])
        for r in per_rows:
            w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.6f}", f"{r[5]:.6f}",
                        f"{r[6]:.6f}", f"{r[7]:.6f}"])
    Z.write_json(Path(args.out), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
