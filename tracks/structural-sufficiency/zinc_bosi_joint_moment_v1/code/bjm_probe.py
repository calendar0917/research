"""Frozen-weight probes (CPU inference only, no training, no test).

Asks whether the trained BOSI-JM checkpoint actually *uses* the joint-moment
block, in the same spirit as the relation-permutation probe of
``zinc_bosi_architecture_v0``:

1. ``mode_swap``  — recompute the same weights with the *CM* statistic
   (``outer(sum u, sum w)/max(n,1)``) instead of ``J``.  Same parameters, same
   inputs, only the pairing constant changes.  A large move means the trained
   function depends on the pairing; it is **not** a trained-CM result.
2. ``zero_joint`` — zero the 288-dim joint block.  This is an
   out-of-distribution operation (it breaks the Side input statistics), so its
   MAE is reported for completeness only, exactly as v0 cautioned.
3. ``marginal_repairing`` — feed ``w`` from a permuted partner inside each
   half-edge group.  Both marginal multisets are preserved exactly, so only the
   pairing changes.

Usage:
    python .../bjm_probe.py --runs JM_s0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bjm_common as BC  # noqa: E402
import bjm_models as BJM  # noqa: E402


def partner_row_lists(tgt: np.ndarray, n_half: int) -> list[np.ndarray]:
    order = np.argsort(tgt, kind="stable")
    tgt_s = tgt[order]
    starts = np.searchsorted(tgt_s, np.arange(n_half), side="left")
    stops = np.searchsorted(tgt_s, np.arange(n_half), side="right")
    return [order[starts[h]:stops[h]] for h in range(n_half)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", default=["JM_s0"])
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    norm = BC.load_norm()
    vcache = BC.load_cache("valid")
    vrel = BC.load_rel("valid")
    rows = np.arange(len(np.asarray(vcache["labels"])))
    y = np.asarray(vcache["labels"], dtype=np.float64)
    out = {"round": BC.ROUND_ID, "written": BC.now(), "runs": {},
           "official_valid_loaded": True, "official_test_loaded": False,
           "note": "frozen-weight inference probes; no retraining, no selection"}

    for tag in args.runs:
        d = BC.RUNS / tag
        info = BC.read_json(d / "train.json")
        arm = info["arm"]
        seed = int(info["seed"])
        blob = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
        model, _ = BJM.build_model(norm, arm, seed, "cpu")
        model.load_state_dict({k: v for k, v in blob["state_dict"].items()})
        base_pred = BC.predict(model, vcache, vrel, rows, device="cpu")
        base = BC.mae(base_pred, y)
        item = {"arm": arm, "seed": seed, "checkpoint": "best.pt",
                "checkpoint_sha256": BC.sha256_file(d / "best.pt"),
                "recomputed_valid_mae": base,
                "train_json_valid_mae": info["best"]["valid_mae"],
                "abs_diff": abs(base - info["best"]["valid_mae"])}

        if arm == "JM":
            # 1. same weights, CM statistic
            model.mode = "CM"
            cm_pred = BC.predict(model, vcache, vrel, rows, device="cpu")
            model.mode = "JM"
            item["mode_swap_to_CM"] = {
                "valid_mae": BC.mae(cm_pred, y),
                "delta_mae": BC.mae(cm_pred, y) - base,
                "mean_abs_dy": float(np.mean(np.abs(cm_pred - base_pred))),
                "max_abs_dy": float(np.max(np.abs(cm_pred - base_pred))),
                "moved_fraction": float(np.mean(np.abs(cm_pred - base_pred) > 1e-6)),
            }
            # 2. marginal-preserving re-pairing of w
            moved = []
            for start in range(0, len(rows), 128):
                chunk = rows[start:start + 128]
                batch = BD_collate(vcache, vrel, chunk)
                tgt = batch["partner_tgt"].numpy()
                lag = batch["partner_half"].numpy()
                p = lag.copy()
                for r in partner_row_lists(tgt, int(batch["n_half"])):
                    if len(r) >= 2:
                        p[r] = np.roll(lag[r], 1)
                y0 = BC.predict(model, vcache, vrel, chunk, device="cpu")
                model._diag_pairing = torch.from_numpy(p.astype(np.int64))
                y1 = BC.predict(model, vcache, vrel, chunk, device="cpu")
                model._diag_pairing = None
                moved.append(np.abs(y1 - y0))
            moved = np.concatenate(moved)
            item["marginal_repairing"] = {
                "mean_abs_dy": float(moved.mean()),
                "max_abs_dy": float(moved.max()),
                "moved_fraction": float(np.mean(moved > 1e-6)),
                "note": "both marginal multisets preserved; only the pairing changes",
            }
            # 3. zero the joint block (out of distribution; reference only)
            orig = BJM.BOSIJointMoment.relation_stats

            def zeroed(self, batch, x, _orig=orig):
                M, n = _orig(self, batch, x)
                return torch.cat([M[:, :96], torch.zeros_like(M[:, 96:])], dim=1), n

            BJM.BOSIJointMoment.relation_stats = zeroed
            try:
                z_pred = BC.predict(model, vcache, vrel, rows, device="cpu")
            finally:
                BJM.BOSIJointMoment.relation_stats = orig
            item["zero_joint_block"] = {
                "valid_mae": BC.mae(z_pred, y),
                "delta_mae": BC.mae(z_pred, y) - base,
                "mean_abs_dy": float(np.mean(np.abs(z_pred - base_pred))),
                "interpretation": "out-of-distribution zeroing; reference only",
            }
        out["runs"][tag] = item
        BC.log(f"[bjm-probe] {tag}: valid={base:.6f} "
               + (f"CM-swap={item['mode_swap_to_CM']['valid_mae']:.6f} "
                  f"repair_dy={item['marginal_repairing']['mean_abs_dy']:.3e} "
                  if arm == "JM" else ""))

    out_path = Path(args.out) if args.out else BC.ANALYSIS / "probe.json"
    BC.write_json(out_path, out)
    BC.log(f"[bjm-probe] wrote {BC.rel_or_abs(out_path)}")
    return 0


def BD_collate(cache, rel, rows):
    import bosi_data as BD

    return BD.collate(cache, rel, rows)


if __name__ == "__main__":
    raise SystemExit(main())
