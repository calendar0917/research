"""CPU smoke test for the masked structural auxiliary branch.

Runs two real auxiliary steps (batch 32) on the precomputed masked views and
reports the CE values, index ranges and the clean/aux loss ratio.  Used before
the first GPU joint run; no GPU time is billed.
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
        sys.path.insert(0, str(_p))

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad2_models as A2  # noqa: E402
import ad2_runner as R2  # noqa: E402


def main() -> int:
    visits = 4
    spec, _ = R2.make_aux(visits)
    norm = D.load_norm()
    cache = D.load_cache("train")
    out = {"visits": visits, "weight": R2.AUX_WEIGHT, "steps": []}
    for cand in ("Maux", "Saux"):
        model, info = A2.build_model(cand, norm, 0, "cpu")
        model.train()
        for step, row_start in enumerate((0, 37)):
            rows = np.arange(row_start, row_start + 32, dtype=np.int64)
            batch = R2.T1.collate(cache, rows, "Maux")
            y = torch.as_tensor(cache["labels"][rows], dtype=torch.float32)
            clean = model(batch)
            l1 = torch.nn.functional.l1_loss(clean, y)
            total, parts = spec["loss"](model, rows, "cpu", epoch=1 + 2 * step)
            loss = l1 + R2.AUX_WEIGHT * total
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            out["steps"].append({
                "candidate": cand, "step": step, "rows": [int(rows[0]), int(rows[-1])],
                "l1_clean": float(l1), "ce_atom": float(parts["ce_atom"]),
                "ce_bond": float(parts["ce_bond"]),
                "weighted_aux": float(R2.AUX_WEIGHT * total),
                "ratio_weighted_aux_to_l1": float(R2.AUX_WEIGHT * total / l1),
                "all_grads_finite": bool(all(torch.isfinite(g).all().item() for g in grads)),
                "nonzero_grad_tensors": int(sum(1 for g in grads if torch.count_nonzero(g).item() > 0)),
                "grad_tensors": len(grads),
            })
        out.setdefault("parameters", {})[cand] = info["parameters"]
    out["ce_atom_marginal_reference"] = {
        "note": "uniform over the 21 observed atom codes",
        "value": float(np.log(21)),
    }
    out["ce_bond_marginal_reference"] = {
        "note": "uniform over the three observed bond codes",
        "value": float(np.log(3)),
    }
    C.write_json(R2.ANALYSIS2 / "aux_smoke.json", out)
    for s in out["steps"]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
