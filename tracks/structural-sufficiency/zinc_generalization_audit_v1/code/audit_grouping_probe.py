"""Zero-training CPU probe: P3B weights under Null vs organized junction grouping.

This is a counterfactual inference probe, not a trained model.  It asks whether
the *already learned* P3B representation contains enough local junction content
that exposing true ownership at inference changes errors in the rare-environment
subgroup.  It does not establish that a trained organized model would improve.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "zinc_recipe_trunk_v2" / "code"))
sys.path.insert(0, str(_HERE.parents[1] / "zinc_arch_discovery_v1" / "code"))
import audit_common as C  # noqa: E402
import ad2_models as A2  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T  # noqa: E402

ANALYSIS = C.ANALYSIS


class OrganizedProbe(A2.P3NullModel):
    def group_to_bonds(self, w: torch.Tensor, batch):
        jb = batch["junc_bond"]
        n_b = torch.zeros(int(batch["bond_feat"].shape[0]), device=w.device, dtype=w.dtype)
        n_b.index_add_(0, jb.reshape(-1), torch.ones(jb.numel(), device=w.device, dtype=w.dtype))
        W = torch.zeros((int(batch["bond_feat"].shape[0]), w.shape[1]), device=w.device, dtype=w.dtype)
        V = torch.zeros_like(W)
        W.index_add_(0, jb[:, 0], w)
        W.index_add_(0, jb[:, 1], w)
        w2 = w * w
        V.index_add_(0, jb[:, 0], w2)
        V.index_add_(0, jb[:, 1], w2)
        return W, V, n_b


def main() -> int:
    cache = C.load_cache("valid")
    rows = np.arange(int(cache["n_molecules"]), dtype=np.int64)
    norm = D.load_norm()
    model = A2.P3NullModel(norm)
    state = torch.load(C.MODELS["P3B_s0"]["run"] / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval()
    base_pred, base_mae = T.evaluate(model, cache, rows, candidate="P3N", device="cpu", batch_size=256)

    probe = OrganizedProbe(norm)
    probe.load_state_dict(state["state_dict"])
    probe.eval()
    org_pred, org_mae = T.evaluate(probe, cache, rows, candidate="P3N", device="cpu", batch_size=256)

    y = np.asarray(cache["labels"], dtype=np.float64)
    # use the audit table's r2 novelty column to define the fixed high-group
    import csv
    table = list(csv.DictReader(open(ANALYSIS / "per_molecule.csv", encoding="utf-8")))
    r2 = np.asarray([float(r["r2_unseen_atom_frac"]) for r in table])
    hi = r2 > 0.0
    base_ae = np.abs(base_pred - y)
    org_ae = np.abs(org_pred - y)
    delta = org_ae - base_ae
    out = {
        "check": "P3B best checkpoint, Null vs organized junction grouping at inference; no retraining",
        "baseline_mae": float(base_mae),
        "organized_mae": float(org_mae),
        "delta_mae_organized_minus_null": float(org_mae - base_mae),
        "by_r2_novelty": {
            "zero_unseen": {
                "n": int((~hi).sum()),
                "null_mae": float(base_ae[~hi].mean()),
                "organized_mae": float(org_ae[~hi].mean()),
                "delta": float(delta[~hi].mean()),
            },
            "any_unseen": {
                "n": int(hi.sum()),
                "null_mae": float(base_ae[hi].mean()),
                "organized_mae": float(org_ae[hi].mean()),
                "delta": float(delta[hi].mean()),
            },
        },
        "mean_abs_prediction_shift": float(np.abs(org_pred - base_pred).mean()),
        "p90_abs_prediction_shift": float(np.quantile(np.abs(org_pred - base_pred), 0.9)),
        "max_abs_prediction_shift": float(np.abs(org_pred - base_pred).max()),
        "fraction_rows_organized_better": float((delta < 0).mean()),
        "note": "Counterfactual inference only: the checkpoint was trained with Null grouping. A trained organized arm is required for a causal claim.",
    }
    C.write_json(ANALYSIS / "grouping_probe.json", out)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
