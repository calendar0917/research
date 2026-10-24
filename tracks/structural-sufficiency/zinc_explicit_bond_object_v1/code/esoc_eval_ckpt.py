"""Evaluate an arbitrary ESOC checkpoint on the official valid split.

Used for interim, matched-epoch reads while a formal run is still in flight.
CPU-only by default so it never disturbs the training GPUs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
for _p in (str(_HERE), str(_SS / "zinc_arch_discovery_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import esoc_data as D  # noqa: E402
import esoc_models as M  # noqa: E402

ANALYSIS = _HERE.parent / "analysis"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    scales = C.read_json(ANALYSIS / "scales.json")
    valid = D.SplitTensors(D.load_cache("valid"), args.device)
    y = valid.labels.double().cpu().numpy()

    rows = []
    for path in args.ckpt:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        arm, seed = ck.get("arm"), int(ck.get("seed", 0))
        net = M.build_model(seed, arm, scales).to(args.device)
        net.load_state_dict(ck["state"])
        net.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, valid.n, 500):
                idx = np.arange(start, min(start + 500, valid.n))
                preds.append(net(valid.batch(idx)).double().cpu().numpy())
        pred = np.concatenate(preds)
        rows.append({
            "path": str(path),
            "arm": arm,
            "seed": seed,
            "epoch": int(ck.get("epoch", -1)),
            "valid_mae": float(np.abs(pred - y).mean()),
            "stored_valid_mae": float(ck.get("valid_mae", float("nan"))),
        })
    rows.sort(key=lambda r: (r["epoch"], r["arm"]))
    out = {"rows": rows}
    by_epoch: dict[int, dict] = {}
    for r in rows:
        by_epoch.setdefault(r["epoch"], {})[r["arm"]] = r["valid_mae"]
    out["by_epoch"] = {str(k): v for k, v in sorted(by_epoch.items())}
    out["real_minus_null"] = {str(k): (v["real"] - v["null"])
                              for k, v in sorted(by_epoch.items())
                              if "real" in v and "null" in v}
    print(json.dumps(out, indent=2))
    if args.out:
        C.write_json(Path(args.out), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
