"""Freeze seed-0 Monitor predictions/checkpoint hashes; never touches Locked."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROUND = _HERE.parent
_SUFF = _ROUND.parent
for _p in (
    str(_HERE),
    str(_SUFF / "zinc_arch_discovery_v1" / "code"),
    str(_SUFF / "zinc_dual_channel_composer_v1" / "code"),
    str(_SUFF / "zinc_dccl_bottleneck_v1" / "code"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import bamf_runner as R  # noqa: E402
import dcb_train as DT  # noqa: E402
import dcc_models as DCC  # noqa: E402
import dcc_runner as DR  # noqa: E402


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_hash_from_dict(sd: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(sd):
        h.update(key.encode())
        h.update(np.ascontiguousarray(sd[key].detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def load_pred(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return np.asarray(obj["monitor_pred"], dtype=np.float64)


def main() -> int:
    split = DT.split_rows()
    cache = D.load_cache("train")
    rows = split["monitor"]
    y = np.asarray(cache["labels"], dtype=np.float64)[rows]
    preds = {}
    for arm in ("G", "I"):
        preds[arm] = load_pred(R.run_dir(arm, 0) / "predictions_best_monitor.pt")
    norm = DT.fit_only_norm(cache, split["fit8k"])
    l_ckpt = _SUFF / "zinc_dccl_bottleneck_v1" / "runs" / "fit8k_s0" / "best_monitor.pt"
    model = DCC.DualChannel(norm)
    ck = torch.load(l_ckpt, map_location="cpu", weights_only=False)
    C.load_state(model, ck["state_dict"], torch.device("cpu"))
    lp, _ = T1.evaluate(model, cache, rows, candidate="DCC-L", device=torch.device("cpu"),
                        batch_size=int(R.RECIPE["valid_batch_size"]), collate_fn=DR.collate_p3n)
    preds["L"] = np.asarray(lp, dtype=np.float64)

    outdir = _ROUND / "analysis"
    csv = outdir / "predictions_seed0_monitor.csv"
    arr = np.column_stack([rows.astype(np.float64), y, preds["G"], preds["I"], preds["L"]])
    np.savetxt(csv, arr, delimiter=",", fmt=["%d", "%.9f", "%.9f", "%.9f", "%.9f"],
               header="train_row,y,G_s0,I_s0,L_s0", comments="")

    files = {}
    for arm in ("G", "I"):
        d = R.run_dir(arm, 0)
        for name in ("best_monitor.pt", "final.pt", "predictions_best_monitor.pt",
                     "predictions_final.pt", "train.json", "curve.csv"):
            p = d / name
            if p.exists():
                files[f"{arm}_s0/{name}"] = {"sha256": sha256_file(p), "bytes": int(p.stat().st_size)}
        ckpt = torch.load(d / "best_monitor.pt", map_location="cpu", weights_only=False)
        files[f"{arm}_s0/best_monitor_state_sha256"] = state_hash_from_dict(ckpt["state_dict"])
    for name in ("best_monitor.pt", "predictions_best_monitor.pt"):
        p = l_ckpt.parent / name
        if p.exists():
            files[f"L_s0/{name}"] = {"sha256": sha256_file(p), "bytes": int(p.stat().st_size)}
    files["L_s0/best_monitor_state_sha256"] = state_hash_from_dict(ck["state_dict"])
    files["predictions_seed0_monitor.csv"] = {"sha256": sha256_file(csv),
                                              "bytes": int(csv.stat().st_size)}

    gates = json.loads((outdir / "monitor_gates.json").read_text())
    out = {
        "protocol_id": R.ROUND_ID,
        "seed": 0,
        "verdict": "STOP_NO_SIGNAL",
        "n_rows": int(len(rows)),
        "train_row_min": int(rows.min()),
        "train_row_max": int(rows.max()),
        "monitor_mae": {k: float(np.abs(v - y).mean()) for k, v in preds.items()},
        "seed0_gate": gates["seed0_gate"],
        "files": files,
        "locked_rows_used": False,
        "locked_values_accessed": False,
        "official_valid_used": False,
        "official_test_used": False,
        "incidental_historical_prediction_file_load": (
            "An earlier analysis pass loaded "
            "zinc_dccl_bottleneck_v1/runs/fit8k_s0/predictions_best_monitor.pt and "
            "extracted only monitor_pred; that file also contains unused locked_pred/"
            "official_valid_pred arrays. Final L predictions here are regenerated from "
            "the checkpoint. No locked/official-valid value was inspected, summarized, "
            "or used in any gate."
        ),
        "formal_run_commit": "a04eb5591ec42e0cde571c4999bb9b4044dff2bf",
    }
    p = outdir / "frozen_seed0.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"monitor_mae": out["monitor_mae"], "csv": str(csv),
                      "gate": out["seed0_gate"], "frozen": str(p)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
