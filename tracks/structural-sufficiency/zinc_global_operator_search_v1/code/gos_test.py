"""Terminal, reporting-only read of the official ZINC test split (GOS-v1).

Two phases in one process, in this order, so the freeze provably precedes the
first test label access:

  1. for every entry of the freeze manifest: rebuild the model from ``best.pt``,
     recompute valid MAE on the full 1,000-molecule valid split and compare with
     the recorded value (must agree to 1e-6); write ``analysis/test_freeze.json``
     including checkpoint sha256.
  2. only then load the test cache (built from official PyG if absent), evaluate
     the frozen checkpoints once and write descriptive scores.

No selection, no epoch change, no recipe change follows from the test numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import gos_data as GD  # noqa: E402
import gos_models as GM  # noqa: E402
import gos_runner as GR  # noqa: E402

ANALYSIS = _EXP / "analysis"
FREEZE = ANALYSIS / "test_freeze.json"
REPORT = ANALYSIS / "test_report.json"
PRED = ANALYSIS / "test_preds"
VALID_TOL = 1e-6


def ensure_test() -> dict:
    if not D.cache_path("test").exists():
        D.prep(("test",), force=False)
    return D.load_cache("test")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=str(ANALYSIS / "freeze_manifest.json"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reread", action="store_true",
                   help="only re-evaluate the already frozen checkpoints (no new selection)")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    rel_train, rel_valid, scale, mean_nb = GR.prepare()
    collate = GD.make_collate(rel_train, rel_valid, scale)
    norm = D.load_norm()
    valid_cache = D.load_cache("valid")
    rows_valid = np.arange(int(valid_cache["labels"].shape[0]))

    if FREEZE.exists() and not args.reread:
        raise SystemExit(f"{FREEZE} already exists: the test read is frozen; use --reread")
    if FREEZE.exists():
        frozen = C.read_json(FREEZE)
        entries = frozen["entries"]
    else:
        entries = C.read_json(Path(args.manifest))["entries"]
        verified = []
        for e in entries:
            ck_path = (C.REPO / e["checkpoint"]).resolve()
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            model, info = GM.build_model(e["route"], norm, int(e["seed"]), device,
                                         mean_n_bonds=mean_nb)
            model.load_state_dict(ck["state_dict"])
            model.eval()
            _pred, mae = T1.evaluate(model, valid_cache, rows_valid, candidate="GOS",
                                     device=device, batch_size=256, collate_fn=collate)
            diff = abs(float(mae) - float(e["recorded_valid"]))
            if diff > VALID_TOL:
                raise SystemExit(f"{e['run']}: recomputed valid {mae:.9f} != recorded "
                                 f"{e['recorded_valid']:.9f} (diff {diff:.2e})")
            verified.append({**e,
                             "valid_recomputed": float(mae),
                             "valid_abs_diff": float(diff),
                             "checkpoint_sha256": C.sha256_file(ck_path),
                             "parameter_count_recomputed": int(info["parameters"]),
                             "epoch_in_checkpoint": int(ck.get("epoch", -1))})
            C.log(f"[freeze] {e['run']} valid={mae:.9f} ok (diff {diff:.2e}) "
                  f"params={info['parameters']} sha={C.sha256_file(ck_path)[:16]}")
        frozen = {
            "protocol": "zinc-global-operator-search-v1",
            "frozen_at": C.utcnow(),
            "git_commit": C.git_commit(),
            "official_test_loaded_before_this_file": False,
            "valid_tolerance": VALID_TOL,
            "selection_rule": "lowest valid MAE (best checkpoint of one declared 600-epoch cosine run)",
            "entries": verified,
        }
        C.write_json(FREEZE, frozen)
        C.log(f"[freeze] wrote {FREEZE}")

    # ---- phase 2: first test access -------------------------------------
    test = ensure_test()
    rel_test = GD.load_rel("test")
    collate_t = GD.make_collate(rel_train, rel_valid, scale, rel_test=rel_test)
    y_test = np.asarray(test["labels"], dtype=np.float64)
    rows = np.arange(len(y_test))
    PRED.mkdir(parents=True, exist_ok=True)
    out = {"protocol": "zinc-global-operator-search-v1", "read_at": C.utcnow(),
           "git_commit": C.git_commit(), "n_test": int(len(y_test)),
           "freeze_file": str(FREEZE.relative_to(C.REPO)), "models": []}
    for e in frozen["entries"]:
        ck_path = (C.REPO / e["checkpoint"]).resolve()
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model, _ = GM.build_model(e["route"], norm, int(e["seed"]), device, mean_n_bonds=mean_nb)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        pred, mae = T1.evaluate(model, test, rows, candidate="GOS", device=device,
                                batch_size=256, collate_fn=collate_t)
        ae = np.abs(pred - y_test)
        np.savez_compressed(PRED / f"{e['run']}_test.npz", row=rows, pred=pred, ae=ae)
        rec = {"run": e["run"], "route": e["route"], "seed": int(e["seed"]),
               "epoch": e["epoch"], "valid_recorded": e["recorded_valid"],
               "valid_recomputed": e["valid_recomputed"], "test_mae": float(mae),
               "test_rmse": float(np.sqrt(np.mean((pred - y_test) ** 2))),
               "test_p90_abs_err": float(np.percentile(ae, 90)),
               "test_median_abs_err": float(np.median(ae))}
        out["models"].append(rec)
        C.log(f"[test] {e['run']} valid={e['valid_recomputed']:.6f} test={mae:.6f}")
    C.write_json(REPORT, out)
    C.log(f"[test] wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
