"""Parameter audit for ESOC-v1 (depth rule: 12 -> 10 -> 8, audit only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
for _p in (str(_HERE), str(_SS / "zinc_arch_discovery_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import esoc_models as M  # noqa: E402
import esoc_train as T  # noqa: E402

ANALYSIS = _HERE.parent / "analysis"


def main() -> int:
    scales = T.read_json(ANALYSIS / "scales.json")
    rows = {}
    chosen = None
    for n_layers in (12, 10, 8):
        model = M.build_model(0, "real", scales, n_layers=n_layers)
        audit = M.param_audit(model)
        rows[f"{n_layers}_blocks"] = audit
        if chosen is None and audit["param_total"] <= 500_000:
            chosen = n_layers
    out = {
        "rule": "12 blocks by default; if the audit exceeds 500k, reduce to 10, then 8; decided only by the audit",
        "variants": rows,
        "chosen_n_layers": chosen,
        "chosen_params": rows[f"{chosen}_blocks"]["param_total"] if chosen else None,
        "under_cap": bool(chosen is not None),
        "real_null_identical_by_construction": True,
        "note": "both arms instantiate the same modules; `arm` only selects the pair-edge input at forward time",
    }
    C.write_json(ANALYSIS / "param_audit.json", out)
    print(json.dumps(out, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
