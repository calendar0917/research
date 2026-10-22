"""Terminal post-selection read of the official ZINC test split.

Same protocol as ``zinc_recipe_trunk_v2/code/ad2_test.py``: every checkpoint was
already frozen and selected on official valid before this file ran; the test split
is read once, nothing is selected, switched, tuned or re-run from the read, and it
does not change any decision recorded in REPORT.md.

Run at the user's request after the round's valid-only decision was written
(PROTOCOL.md amendment A3).  CPU-only inference; no A100 time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_TASK = _HERE.parent
_SSUF = _TASK.parent
for _p in (str(_HERE), str(_SSUF / "zinc_arch_discovery_v1" / "code"),
           str(_SSUF / "zinc_recipe_trunk_v2" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import zct_train as ZT  # noqa: E402

RUNS = _TASK / "runs"
OUT = _TASK / "cache" / "artifacts"

CORE = ["E_s0", "R_s0", "S_s0", "E_s1", "R_s1", "S_s1", "R_s0_k04", "S_s0_k04", "R_s1_k04"]
SUPPORT = ["sup_plus_s0", "sup_minus_s0", "sup_plus_s1", "sup_minus_s1"]
WHICH = {"best": "best.pt", "final": "final.pt", "best_monitor": "best_monitor.pt"}


def which_for(run: str) -> tuple[str, ...]:
    return ("best_monitor", "final") if run.startswith("sup_") else ("best", "final")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=str(OUT / "test_terminal.json"))
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    test = D.load_cache("test")
    norm = D.load_norm()
    rows = np.arange(int(test["n_molecules"]), dtype=np.int64)
    y = np.asarray(test["labels"], dtype=np.float64)
    payload = {
        "protocol": "zinc-composition-transfer-v1",
        "split": "official PyG ZINC test (1000 molecules)",
        "n": int(len(rows)),
        "test_read": ("one pass, after every checkpoint was frozen and selected on official "
                      "valid; requested by the user after the valid-only decision was written; "
                      "descriptive only, nothing selected or changed from it"),
        "test_label_sha256": C.sha256_array(y),
        "runs": {},
    }
    for run in CORE + SUPPORT:
        run_dir = RUNS / run
        if not (run_dir / "train.json").exists():
            continue
        meta = C.read_json(run_dir / "train.json")
        model, info = ZT.build_model(norm, int(meta["seed"]), torch.device(args.device))
        for which in which_for(run):
            ckpt = run_dir / WHICH[which]
            if not ckpt.exists():
                continue
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            C.load_state(model, ck["state_dict"], torch.device(args.device))
            model.eval()
            pred, mae = T1.evaluate(model, test, rows, candidate="P3N",
                                    device=torch.device(args.device), batch_size=256)
            pred = np.asarray(pred, dtype=np.float64)
            name = f"{run}_{which}"
            np.savez_compressed(OUT / f"test_{name}.npz", pred=pred, y=y)
            ae = np.abs(pred - y)
            if run.startswith("sup_"):
                valid_ref = None
            else:
                valid_ref = float(meta["fixed_objects"][which]["valid_mae"])
            payload["runs"][name] = {
                "run": run, "which": which, "arm": meta["arm"] if "arm" in meta else meta["fit"],
                "seed": int(meta["seed"]), "kappa": meta.get("kappa"),
                "epoch": int(ck.get("epoch", -1)),
                "test_mae": float(mae), "valid_mae": valid_ref,
                "test_minus_valid": None if valid_ref is None else float(mae - valid_ref),
                "median_ae": float(np.median(ae)),
                "max_ae": float(ae.max()),
                "max_ae_row": int(np.argmax(ae)),
                "checkpoint": str(ckpt), "checkpoint_sha256": C.sha256_file(ckpt),
                "parameters": int(info["parameters"]),
                "init_state_sha256": meta["init_state_sha256"],
            }
            C.log(f"[test] {name:24s} test={mae:.6f} valid={valid_ref} "
                  f"ep={ck.get('epoch')} med={np.median(ae):.4f}")
    C.write_json(Path(args.out), payload)
    C.log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
