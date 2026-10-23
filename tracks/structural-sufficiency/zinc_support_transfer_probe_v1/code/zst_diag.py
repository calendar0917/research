"""One targeted diagnostic: is the residual predictable *within* a structural key?

C_fit is split molecule-wise inside each key family 80/20 (deterministic). The
same G/S heads are trained on the 80% and evaluated on the 20%, whose keys are by
construction represented in the training part. If the residual cannot be
predicted even there, the failure of the cross-family correction is not a
transfer problem but a "the residual is mostly label noise" problem.

Also reports the descriptive oracle constant offset per query package (best
constant *after* seeing the package labels) as a ceiling for any group-level
correction, and the train/test correlation of the residual with the coarse
features.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import zst_common as Z  # noqa: E402
import zst_head as H  # noqa: E402
import ad_data as D  # noqa: E402


def main() -> int:
    man = json.load(open(Z.ANALYSIS / "split_manifest.json", encoding="utf-8"))
    tkeys = {k: np.asarray(v, dtype=np.int64) for k, v in man["target_keys"].items()}
    cache = D.load_cache("train")
    keys = Z.load_keys()
    _, mol2 = Z.molecule_key_sets(keys)
    atom_k1, atom_k2, _ = Z.atom_tables(keys, int(keys["n_key2"]))
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    out = {"within_key": {}, "oracle_constant": {}, "feature_correlation": {}}
    for seed in (0, 1):
        data = H.run_base_seed(seed, man, cache, atom_k1, atom_k2, atom_ptr)
        fit = data["f_cfit"]
        gated = fit.gate > 0
        # within-key molecule split: for every C-family key, every 5th molecule
        # (by a fixed hash of the row id) goes to the held-out part
        rng = np.random.default_rng(777)
        hold = np.zeros(len(fit.rows), dtype=bool)
        for k in tkeys["C"].tolist():
            ix = np.flatnonzero([int(k) in set(mol2[int(r)].tolist()) for r in fit.rows])
            if len(ix) > 1:
                perm = rng.permutation(len(ix))
                hold[ix[perm[: max(1, len(ix) // 5)]]] = True
        hold &= gated
        tr = ~hold
        r = data["y_cfit"] - data["b_cfit"]
        parts = {}
        for tag, m in (("train", tr), ("hold", hold)):
            o = object.__new__(H.Features)
            for name in ("ctx", "aux", "atoms", "mask", "gate", "n_unseen", "rows"):
                setattr(o, name, getattr(fit, name)[m])
            parts[tag] = H.attach_tensors(o)
            parts[tag].y = data["y_cfit"][m]
            parts[tag].b = data["b_cfit"][m]
        res = {"n_train": int(tr.sum()), "n_hold": int(hold.sum()),
               "base_mae_hold": float(np.abs(parts["hold"].b - parts["hold"].y).mean())}
        for kind in ("G", "S"):
            model, curve, best, final_state, _s = H.train_head(
                kind, parts["train"], parts["train"].y - parts["train"].b, parts["train"].b,
                0.0, 0, dev=parts["hold"], dev_base=parts["hold"].b, dev_y=parts["hold"].y)
            corr = parts["hold"].b + H.predict_head(model, parts["hold"])
            m = H.metrics(parts["hold"].y, parts["hold"].b, corr)
            mend = (H.HeadS() if kind == "S" else H.HeadG())
            mend.load_state_dict(final_state)
            corr_end = parts["hold"].b + H.predict_head(mend, parts["hold"])
            m_end = H.metrics(parts["hold"].y, parts["hold"].b, corr_end)
            res[kind] = {"best_step": best, "hold_delta": m["delta"],
                         "hold_mae": m["mae_corrected"],
                         "hold_delta_endpoint": m_end["delta"],
                         "endpoint_train_fit": curve[-1]["fit_l1"],
                         "train_fit_at_best": [c for c in curve if c["step"] == best][0]["fit_l1"],
                         "hold_curve": [(c["step"], round(c["dev_mae"], 4)) for c in curve]}
            print(f"[within-key s{seed} {kind}] hold delta {m['delta']:+.4f} "
                  f"best {best} base {res['base_mae_hold']:.4f}", flush=True)
        out["within_key"][str(seed)] = res
        for q in ("Q_A", "Q_B", "Q_bg"):
            base, yy = data["b_q"][q], data["y_q"][q]
            grid = np.linspace(-0.3, 0.3, 1201)
            maes = [np.abs(base + c - yy).mean() for c in grid]
            j = int(np.argmin(maes))
            out["oracle_constant"][f"{q}_s{seed}"] = {
                "base_mae": float(np.abs(base - yy).mean()),
                "best_constant": float(grid[j]),
                "best_mae": float(maes[j]),
                "gain": float(np.abs(base - yy).mean() - maes[j])}
        # correlation of the residual with coarse features on C_fit vs Q
        ctx = data["f_cfit"].ctx
        rr = data["y_cfit"] - data["b_cfit"]
        cor = {"C_fit_fbase_size_unseenfrac_minn1": [
            float(np.corrcoef(ctx[gated, j], rr[gated])[0, 1]) for j in (0, 1, 7, 8)]}
        out["feature_correlation"][str(seed)] = cor
    Z.write_json(Z.ANALYSIS / "diagnostics.json", out)
    print(json.dumps(out["oracle_constant"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
