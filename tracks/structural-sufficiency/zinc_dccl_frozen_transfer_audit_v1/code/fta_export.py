"""Phase A: export the frozen 513-dim reader input of four DCC-L final checkpoints.

Runs the *real* frozen forward (no reimplementation is used for the export; the
independent recomputation is only a verification path), and writes

    runs/z/<fit>_s<seed>.npz   {z (5000,513) f32, pred (5000,) f32, rows_i64}
    analysis/representation_freeze.json

No label is read here.  The Monitor label is read once, after every probe cell
is frozen, by ``fta_score.py``.
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np

import fta_common as K


def main() -> int:
    t_start = time.perf_counter()
    cfg = K.load_config()
    cache = K.load_train_cache()
    rows = K.split_rows()
    pf, ps, split_info = K.probe_split(cache, rows["fit4k"])
    mon = rows["monitor"]
    export_rows = np.concatenate([pf, ps, mon]).astype(np.int64)
    K.log(f"probe split {split_info['n_probe_fit']}/{split_info['n_probe_sel']} "
          f"export rows {len(export_rows)}")

    freeze = {
        "round": K.ROUND_ID,
        "protocol_sha256": K.sha256_file(K.ROUND / "PROTOCOL.md"),
        "config_sha256": K.sha256_file(K.CONFIG_PATH),
        "split": split_info,
        "export_rows": {
            "n": int(len(export_rows)),
            "sha256": K.row_hash(export_rows),
            "order": "probe_fit ++ probe_sel ++ monitor",
            "slices": {"probe_fit": [0, len(pf)],
                       "probe_sel": [len(pf), len(pf) + len(ps)],
                       "monitor": [len(pf) + len(ps), len(export_rows)]},
            "monitor_rows_sha256": K.row_hash(mon),
        },
        "conditions": {},
    }

    for fit, seed in K.CONDITIONS:
        cid = f"{fit}_s{seed}"
        t0 = time.perf_counter()
        model, norm, meta = K.build_trunk(fit, seed)
        z, pred, checks = K.export_z(model, cache, export_rows)
        wall = time.perf_counter() - t0

        if z.shape != (len(export_rows), int(cfg["representation"]["dim"])):
            raise RuntimeError(f"{cid}: bad z shape {z.shape}")
        if not np.isfinite(z).all() or not np.isfinite(pred).all():
            raise RuntimeError(f"{cid}: non-finite z/pred")
        tol = cfg["export"]
        checks["E4_relative_to_max_abs_z"] = (
            checks["E4_batch64_vs_batch256_max_abs"] / max(float(np.abs(z).max()), 1e-12))
        gate = (
            checks["E1_hook_vs_export_max_abs"] <= tol["E1_hook_vs_export_max_abs"]
            and checks["E2_hook_vs_independent_recompute_max_abs"] <= tol["E2_tol"]
            and checks["E3_reader_replay_max_abs"] <= tol["E3_tol"]
            and checks["E4_relative_to_max_abs_z"] <= tol["E4_rel_tol"]
        )
        if not gate:
            raise RuntimeError(f"{cid}: export gates failed {checks}")

        z_sha = K.atomic_npz(K.Z_DIR / f"{cid}.npz", z=z, pred=pred, rows=export_rows)
        freeze["conditions"][cid] = {
            **meta,
            "fit_rows": int(len(rows[fit])),
            "seed": int(seed),
            "norm_source": meta["norm_source"],
            "checks": checks,
            "gates_pass": bool(gate),
            "z_sha256": z_sha,
            "z_shape": list(z.shape),
            "z_stats": {
                "mean_abs": float(np.abs(z).mean()),
                "max_abs": float(np.abs(z).max()),
                "per_block": {
                    name: {
                        "mean": float(z[:, a:b].mean()),
                        "std": float(z[:, a:b].std()),
                        "trace_cov": float(z[:, a:b].astype(np.float64).var(0, ddof=1).sum()),
                    }
                    for name, (a, b) in K.BLOCKS.items()
                },
            },
            "wall_seconds": float(wall),
            "pred_sha256": K.sha256_file(K.Z_DIR / f"{cid}.npz"),
        }
        K.log(f"{cid} z={z.shape} E2={checks['E2_hook_vs_independent_recompute_max_abs']:.1e} "
              f"E3={checks['E3_reader_replay_max_abs']:.1e} "
              f"E4={checks['E4_batch64_vs_batch256_max_abs']:.1e} {wall:.1f}s")
        K.ledger_append({"stage": "export", "condition": cid, "wall_seconds": float(wall),
                         "n_rows": int(len(export_rows)), "device": "cpu"})

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    freeze["wall_seconds"] = float(time.perf_counter() - t_start)
    freeze["peak_rss_mb"] = float(rss_mb)
    freeze["torch_threads"] = 1
    K.write_json(K.ANALYSIS / "representation_freeze.json", freeze)
    K.log(f"wrote analysis/representation_freeze.json ({freeze['wall_seconds']:.1f}s, "
          f"{rss_mb:.0f} MB peak)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
