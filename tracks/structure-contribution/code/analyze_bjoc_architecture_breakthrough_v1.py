"""Post-freeze descriptive analysis for cscl-bjoc-architecture-breakthrough-v1.

Reads the frozen artifacts in the round results directory (predictions, labels
of the single declared Monitor read and the single Locked read, soups, objects)
and writes ``analysis_C.json``.  This script makes no model, gate, or selection
decisions; it is descriptive mechanism evidence for the report.

Label policy: the Monitor labels were read at the declared Monitor scoring
round and the Locked labels at the single Locked read; this analysis is run
after both reads and only describes the already-frozen prediction sets.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_arch_breakthrough_v1 as ab  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

OUT = REPO_ROOT / "tracks/structure-contribution/results/bjoc_architecture_breakthrough_v1"


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_pred(zone: str, arm: str, seed: int) -> np.ndarray:
    z = np.load(OUT / "predictions" / f"{zone}_{arm}_C_s{seed}.npz", allow_pickle=True)
    return np.asarray(z["pred"], dtype=np.float64).reshape(-1)


def _lambda_stats(model, objs, device, chunk: int = 512) -> np.ndarray:
    model.eval()
    vals = []
    with torch.no_grad():
        for i in range(0, len(objs), chunk):
            b = bp._mv(sro.batch_objects(list(objs[i : i + chunk])), device)
            vals.append(model.lambda_of(b, next(model.parameters()).dtype).detach().cpu().numpy())
    return np.concatenate(vals, 0) if vals else np.zeros(0, dtype=np.float64)


def _summary(x: np.ndarray) -> dict:
    return {
        "n": int(x.size),
        "mean": float(x.mean()) if x.size else None,
        "std": float(x.std()) if x.size else None,
        "min": float(x.min()) if x.size else None,
        "max": float(x.max()) if x.size else None,
        "abs_mean": float(np.abs(x).mean()) if x.size else None,
        "p10": float(np.quantile(x, 0.10)) if x.size else None,
        "p90": float(np.quantile(x, 0.90)) if x.size else None,
    }


def _zone_objs(zone: str):
    return torch.load(OUT / f"objects_{zone}_C.pt", map_location="cpu", weights_only=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT / "analysis_C.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    norm = read_json(OUT / "prep_fit_C.json")
    zones = {z: _zone_objs(z) for z in ("monitor", "locked")}
    y = {z: np.asarray(np.load(OUT / f"{z}_y_C.npy"), dtype=np.float64).reshape(-1) for z in zones}

    payload: dict = {
        "protocol": ab.PROTOCOL_ID,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "descriptive analysis after the declared Monitor and Locked reads; no selection use",
        "official_test_loaded": False,
    }

    # --- GLRC gate statistics over both zones
    lambda_table: dict = {}
    gate_weights: dict = {}
    for seed in (0, 1):
        _, model, _ = ab.build_glrc_from_init(seed, norm, device, None)
        model.load_state_dict(torch.load(OUT / f"soup_GLRC_C_s{seed}.pt", map_location=device, weights_only=False))
        lam = {z: _lambda_stats(model, zones[z], device) for z in zones}
        both = np.concatenate([lam[z] for z in zones])
        lambda_table[f"seed{seed}"] = {
            "both": _summary(both),
            "monitor": _summary(lam["monitor"]),
            "locked": _summary(lam["locked"]),
        }
        w = model.lambda_gate.weight.detach().cpu().numpy().reshape(-1)
        b = float(model.lambda_gate.bias.detach().cpu().reshape(-1)[0])
        order = np.argsort(-np.abs(w))
        gate_weights[f"seed{seed}"] = {
            "bias": b,
            "top_features": [
                {"name": sro.GLOBAL_FEAT_NAMES[i], "w": float(w[i])} for i in order[:12]
            ],
            "n_nonzero_abs_gt_1e-4": int((np.abs(w) > 1e-4).sum()),
        }
        for z in zones:
            np.save(OUT / f"lambda_GLRC_C_s{seed}_{z}.npy", lam[z].astype(np.float32))
    payload["glrc_lambda"] = lambda_table
    payload["glrc_gate_weights"] = gate_weights

    # --- prediction differences and label-blind statistics
    diffs: dict = {}
    for zone in zones:
        p3 = {s: _read_pred(zone, "P3", s) for s in (0, 1)}
        glrc = {s: _read_pred(zone, "GLRC", s) for s in (0, 1)}
        dlp = {0: _read_pred(zone, "DLP", 0)}
        lam = {s: np.load(OUT / f"lambda_GLRC_C_s{s}_{zone}.npy") for s in (0, 1)}
        zone_out = {}
        for s in (0, 1):
            d = glrc[s] - p3[s]
            zone_out[f"GLRC_s{s}"] = {
                **_summary(d),
                "corr_pred_with_lambda": float(np.corrcoef(d, lam[s])[0, 1]),
                "corr_absdelta_with_abslambda": float(np.corrcoef(np.abs(d), np.abs(lam[s]))[0, 1]),
            }
        d = dlp[0] - p3[0]
        zone_out["DLP_s0"] = _summary(d)
        diffs[zone] = zone_out
    payload["prediction_differences"] = diffs

    # --- locked descriptive score table + gate-quartile breakdown (confirmatory zone)
    lam0 = np.load(OUT / "lambda_GLRC_C_s0_locked.npy")
    yl = y["locked"]
    table = {}
    for arm, seeds in (("P3", (0, 1)), ("GLRC", (0, 1)), ("DLP", (0,))):
        preds = {s: _read_pred("locked", arm, s) for s in seeds}
        err = np.mean([np.abs(preds[s] - yl) for s in seeds], axis=0)
        table[arm] = {"seed_mean_mae": float(err.mean()), "seeds": list(seeds)}
    payload["locked_table"] = table

    p3_err = np.mean([np.abs(_read_pred("locked", "P3", s) - yl) for s in (0, 1)], axis=0)
    glrc_err = np.mean([np.abs(_read_pred("locked", "GLRC", s) - yl) for s in (0, 1)], axis=0)
    delta = glrc_err - p3_err
    q = np.quantile(lam0, [0.0, 0.25, 0.5, 0.75, 1.0])
    quart = np.clip(np.digitize(lam0, q[1:-1]), 0, 3)
    payload["locked_glrc_delta_by_lambda_quartile"] = {
        f"q{k}": {
            "lambda_range": [float(q[k]), float(q[k + 1])],
            "n": int((quart == k).sum()),
            "delta_mae": float(delta[quart == k].mean()) if (quart == k).any() else None,
        }
        for k in range(4)
    }

    # junction-richness subgroups (structure-only descriptor, labels on locked)
    n_junc = np.asarray([len(np.asarray(o.junc_index).reshape(-1)) for o in zones["locked"]], dtype=np.float64)
    t = np.quantile(n_junc, [0.5])
    payload["locked_glrc_delta_by_junction_count"] = {
        "low": {"n": int((n_junc <= t[0]).sum()), "delta_mae": float(delta[n_junc <= t[0]].mean())},
        "high": {"n": int((n_junc > t[0]).sum()), "delta_mae": float(delta[n_junc > t[0]].mean())},
    }

    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[analysis] written {args.out}")
    print(json.dumps(payload["glrc_lambda"], indent=1))
    print(json.dumps(payload["prediction_differences"]["monitor"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
