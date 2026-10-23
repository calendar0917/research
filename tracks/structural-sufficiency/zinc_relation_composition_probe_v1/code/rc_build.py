"""Build the D/S probe on CPU and audit every module's parameter count.

No data and no GPU are needed: this is the design-vs-implementation parameter
check required before any training is declared.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import rc_models as RM  # noqa: E402,F401  (registers the sibling code paths)
import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402

ANALYSIS = _HERE.parent / "analysis"


def audit(seed: int = 0, stats: dict | None = None, arm: str = "D") -> dict:
    stats = stats or {"mean_bonds": 24.9279, "mean_atoms": 23.1664}
    norm = D.load_norm()
    model, info = RM.build_initial_state(norm, seed, stats=stats, arm=arm)
    per_module = {name: int(sum(p.numel() for p in mod.parameters()))
                  for name, mod in model.named_children()}
    per_layer = [
        {name: int(sum(p.numel() for p in sub.parameters())) for name, sub in layer.named_children()}
        for layer in model.core.layers
    ]
    total = int(sum(p.numel() for p in model.parameters()))
    buffers = {k: tuple(v.shape) for k, v in model.named_buffers()}
    out = {
        "arm": arm,
        "seed": int(seed),
        "d_hidden": RM.D_HID,
        "heads": RM.N_HEADS,
        "layers": RM.N_LAYERS,
        "ffn_width": 2 * RM.D_HID,
        "param_total": total,
        "under_500k": bool(total <= 500_000),
        "per_module": per_module,
        "per_layer": per_layer,
        "layer_with_relation_update": [l.update_relation for l in model.core.layers],
        "buffers": {k: list(v) for k, v in buffers.items()},
        "parameter_tensors": sum(1 for _ in model.parameters()),
    }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arm", default="D")
    args = p.parse_args(argv)
    res = audit(args.seed, arm=args.arm)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS / "param_audit.json", res)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
