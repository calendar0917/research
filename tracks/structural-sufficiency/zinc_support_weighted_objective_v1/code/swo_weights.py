"""Build and freeze the per-training-molecule weight vector W (train-only).

FROZEN FUNCTION (declared before any GPU run; see ``configs/swo_v1.json``):

    vocab      = exact typed radius-2 rooted signatures of all 10,000 TRAINING
                 molecules (same construction and thresholds as
                 ``bjm_common.radius2_support``, i.e. the track's frozen
                 definition) with counts c_h over training ATOMS.
    low4_i     = fraction of molecule i's atoms whose signature has
                 1 <= c_h <= 4   (the track's "low support" band)
    z_i        = clip((low4_i - mean_train) / std_train, -1, +1)
    w_i        = 1 + 0.5 * z_i
    w_i        = w_i / mean(w)                       (exact normalisation)

Only structure enters: no labels, no valid/test molecules, no model errors, no
knowledge of row 172.  The function is frozen by sha256 before training and is
identical for every arm and seed.

Why this shape: ``low4`` is the only per-molecule statistic in this track that is
(a) computable on training molecules, (b) defined by the same vocabulary that
later defines the valid ``radius2_low_support`` / ``radius2_any_unseen`` groups,
and (c) bounded in [0, 1] so the weight stays inside a narrow, conservative band.
The sign is declared up-front: the hypothesis is that structurally rare training
molecules are the ones that teach compositional coverage, so they get MORE weight.
A null or a loss therefore also rules out the opposite reading ("rare examples are
noise"), because the two readings predict opposite signs.
"""

from __future__ import annotations

import argparse

import numpy as np

import swo_common as S


LAMBDA = 0.5          # frozen weight amplitude (weight band is ~[0.68, 1.61])
CLIP_Z = 1.0          # frozen clip on the standardised score
LOW_LO, LOW_HI = 1, 4  # frozen "low support" band (identical to support_tier)


def build() -> dict:
    import ad_data as D
    import bjm_common as BC

    tcache = D.load_cache("train")
    tgraphs = BC.molecules_from_cache(tcache)
    sup = BC.radius2_support(tgraphs, tgraphs)   # vocab = train, query = train
    low4 = np.asarray(sup["low4_frac"], dtype=np.float64)
    unseen = np.asarray(sup["unseen_frac"], dtype=np.float64)
    assert unseen.max() == 0.0, "train molecules must be self-supported by construction"

    mu = float(low4.mean())
    sd = float(low4.std())
    z_raw = (low4 - mu) / sd
    z = np.clip(z_raw, -CLIP_Z, CLIP_Z)
    w = 1.0 + LAMBDA * z
    w = w / w.mean()

    ess = float(w.sum() ** 2 / (w ** 2).sum())
    n_atoms = np.asarray(tcache["n_atoms_graph"], dtype=np.float64)
    stats = {
        "n_train": int(len(w)),
        "vocab_size": int(sup["train_vocab_size"]),
        "low4": {"mean": mu, "std": sd, "median": float(np.median(low4)),
                 "p90": float(np.quantile(low4, 0.90)),
                 "p99": float(np.quantile(low4, 0.99)),
                 "max": float(low4.max()),
                 "frac_exactly_zero": float(np.mean(low4 == 0))},
        "clip_fraction": float(np.mean(np.abs(z_raw) > CLIP_Z)),
        "weights": {"min": float(w.min()), "max": float(w.max()),
                    "mean": float(w.mean()), "std": float(w.std()),
                    "p10": float(np.quantile(w, 0.10)),
                    "p50": float(np.quantile(w, 0.50)),
                    "p90": float(np.quantile(w, 0.90)),
                    "frac_gt_1": float(np.mean(w > 1.0)),
                    "frac_lt_1": float(np.mean(w < 1.0))},
        "effective_sample_size": ess,
        "effective_sample_size_fraction": ess / len(w),
        "corr_low4_n_atoms": float(np.corrcoef(low4, n_atoms)[0, 1]),
        "corr_weight_n_atoms": float(np.corrcoef(w, n_atoms)[0, 1]),
        "weight_sha256": S.sha256_array(w.astype(np.float64)),
        "labels_used": False, "valid_or_test_used": False,
        "frozen_function": "w = 1 + 0.5*clip((low4-mean)/std, -1, 1); w /= mean(w)",
        "frozen_parameters": {"lambda": LAMBDA, "clip_z": CLIP_Z,
                             "low_band": [LOW_LO, LOW_HI],
                             "vocab": "exact typed radius-2 rooted signatures, train 10k"},
    }
    return {"weights": w, "stats": stats}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    out = S.ANALYSIS / "weights_train10k.npz" if args.out is None else S.ANALYSIS / args.out
    meta_path = out.with_suffix(".json")
    if out.exists() and meta_path.exists() and not args.force:
        print(f"[swo-weights] exists ({S.sha256_file(out)[:12]}); use --force to rebuild")
        return 0
    built = build()
    S.ANALYSIS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, w=built["weights"].astype(np.float64))
    meta = dict(built["stats"])
    meta.update({"written": S.now(), "npz": S.rel(out),
                 "npz_sha256": S.sha256_file(out)})
    S.write_json(meta_path, meta)
    st = built["stats"]
    print(f"[swo-weights] n={st['n_train']} vocab={st['vocab_size']} "
          f"low4_mean={st['low4']['mean']:.4f} clip={st['clip_fraction']:.3f} "
          f"w=[{st['weights']['min']:.3f},{st['weights']['max']:.3f}] "
          f"ESS={st['effective_sample_size']:.1f} "
          f"({100*st['effective_sample_size_fraction']:.1f}%) "
          f"sha={st['weight_sha256'][:12]} -> {S.rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
