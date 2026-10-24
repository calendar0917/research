"""CPU smoke test for the structure-sensitivity diagnostic (no train.json needed)."""
import sys, json
from pathlib import Path
import torch
_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
for _p in (str(_HERE), str(_SS / "zinc_arch_discovery_v1" / "code")):
    if _p not in sys.path: sys.path.insert(0, _p)
import ad_common as C, esoc_analyze as A

R = _HERE.parent / "runs" / "_aborted_default"
runs = {}
for arm in ("real", "null"):
    p = R / f"{arm}_s0_best.pt"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    runs[f"{arm}_s0"] = {"arm": arm, "seed": int(ck["seed"]), "dir": str(R), "ckpt": str(p),
                         "pred_best": None, "y_best": None}
# monkeypatch the loader path used by structure_sensitivity
_orig = torch.load
def patched(path, *a, **k):
    if str(path).endswith("/best.pt"):
        for arm in ("real", "null"):
            if f"/{arm}_s0/" in str(path):
                return _orig(str(R / f"{arm}_s0_best.pt"), *a, **k)
    return _orig(path, *a, **k)
torch.load = patched
scales = C.read_json(_HERE.parent / "analysis" / "scales.json")
out = A.structure_sensitivity(runs, scales, n_rows=64, n_perm=2)
torch.load = _orig
print(json.dumps({k: (v if not isinstance(v, dict) else
                      {kk: vv for kk, vv in v.items() if not kk.startswith("per_row")})
                  for k, v in out.items()}, indent=2))
