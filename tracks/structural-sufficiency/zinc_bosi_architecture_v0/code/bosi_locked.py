"""The single, freeze-gated Locked read.

Refuses to run unless ``analysis/freeze.json`` exists and every recorded hash
still matches.  Writes ``analysis/locked_read.json`` and
``analysis/locked_preds/<run>_<ckpt>.npz``.  Idempotent: a second invocation
verifies the existing artefacts and exits instead of recomputing them.

Frozen-weight interventions written in the same single read (inference only):
  * ``swap_arm``   — an R checkpoint evaluated with B statistics (and vice versa)
  * ``zero_stats`` — the M block of the side input forced to zero
  * ``rel_perm``   — histogram-preserving permutation of the relation labels,
                     for a fixed 256-molecule subset (sensitivity diagnostic)
  * ``side_zero``  — the whole side statistics+counts+closure block zeroed

Usage:
    python .../bosi_locked.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ad_common as C  # noqa: E402
import bosi_common as BC  # noqa: E402
import bosi_data as BD  # noqa: E402
import bosi_models as BM  # noqa: E402

M_START = 2 * BM.WIDTH + BM.RANK
M_STOP = M_START + BD.N_REL * 3 * BM.RANK


def _verify(freeze: dict) -> None:
    checks = {
        "config": (BC.CONFIGS / "bosi_v0.json", freeze["config_sha256"]),
        "protocol": (BC.ROUND / "PROTOCOL.md", freeze["protocol_sha256"]),
        "design": (BC.ROUND / "DESIGN.md", freeze["design_sha256"]),
        "data_cache": (BC.DATA_CACHE, freeze["data_cache_sha256"]),
        "relation_cache": (BC.CACHE / "bosi_rel_train.pt", freeze["relation_cache_sha256"]),
    }
    for name, (path, expected) in checks.items():
        if name == "relation_cache":
            actual = BD.relation_content_hash(
                torch.load(path, map_location="cpu", weights_only=False))
        else:
            actual = BC.sha256_file(path)
        if actual != expected:
            raise SystemExit(f"freeze mismatch on {name}: {actual} != {expected}")
    for tag, run in freeze["runs"].items():
        for ckpt, entry in run["checkpoints"].items():
            path = BC.ROUND / entry["file"]
            if BC.sha256_file(path) != entry["sha256"]:
                raise SystemExit(f"freeze mismatch on {tag}/{ckpt}")


def _patched(model, patch: str):
    """Context manager applying one frozen-weight intervention."""
    import contextlib

    @contextlib.contextmanager
    def cm():
        orig = model.side.forward
        if patch == "zero_stats":
            def fwd(x):
                x = x.clone()
                x[:, M_START:M_STOP] = 0.0
                return orig(x)
        elif patch == "side_zero":
            def fwd(x):
                x = x.clone()
                x[:, 2 * BM.WIDTH:] = 0.0
                return orig(x)
        else:
            raise ValueError(patch)
        model.side.forward = fwd
        try:
            yield
        finally:
            model.side.forward = orig

    return cm()


def main(argv=None) -> int:
    if not (BC.ANALYSIS / "freeze.json").exists():
        raise SystemExit("run bosi_freeze.py first")
    freeze = json.loads((BC.ANALYSIS / "freeze.json").read_text())
    _verify(freeze)
    out_dir = BC.ANALYSIS / "locked_preds"
    lock_path = BC.ANALYSIS / "locked_read.json"
    if lock_path.exists():
        BC.log("[bosi-locked] locked_read.json already exists; verifying and exiting")
        existing = json.loads(lock_path.read_text())
        for tag, entry in existing["predictions"].items():
            for ckpt, meta in entry.items():
                path = BC.ROUND / meta["file"]
                if BC.sha256_file(path) != meta["sha256"]:
                    raise SystemExit(f"locked prediction file changed: {path}")
        BC.log("[bosi-locked] existing artefacts verified, nothing recomputed")
        return 0

    cache = BC.load_train_cache()
    rel = torch.load(BC.CACHE / "bosi_rel_train.pt", map_location="cpu", weights_only=False)
    rows = BC.split_rows()
    locked = rows["locked"]
    y = BC.labels(cache, locked)
    norm = BC.fit_only_norm(cache, rows["fit8k"])
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "read_utc": BC.now(),
        "round": BC.ROUND_ID,
        "locked_rows_sha256": rows["hashes"]["locked"],
        "n_locked": int(len(locked)),
        "freeze_config_sha256": freeze["config_sha256"],
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "note": "internal Locked split, historically exposed; not a new blind test",
        "predictions": {},
        "interventions": {},
    }

    perm_subset = locked[:256]
    rng = np.random.default_rng(20261026)

    for tag, run in freeze["runs"].items():
        seed, arm = int(run["seed"]), run["arm"]
        result["predictions"][tag] = {}
        for ckpt, meta in run["checkpoints"].items():
            model, _ = BM.build_model(norm, "R", seed, "cpu")
            model.arm = arm
            blob = torch.load(BC.ROUND / meta["file"], map_location="cpu", weights_only=False)
            C.load_state(model, blob["state_dict"], "cpu")
            pred = BC.predict(model, cache, rel, locked, device="cpu",
                              batch_size=int(BC.RECIPE["eval_batch_size"]))
            arr = np.asarray(pred, dtype=np.float32)
            path = out_dir / f"{tag}_{ckpt}.npz"
            np.savez(path, row=locked, pred=arr, arm=arm, seed=seed,
                     checkpoint=ckpt, y=y.astype(np.float32))
            result["predictions"][tag][ckpt] = {
                "file": str(path.relative_to(BC.ROUND)),
                "sha256": BC.sha256_file(path),
                "locked_mae": BC.mae(arr, y),
                "checkpoint_sha256": meta["sha256"],
            }
            BC.log(f"[bosi-locked] {tag}/{ckpt} locked MAE {BC.mae(arr, y):.9f}")

            # frozen-weight interventions, best-monitor checkpoint only
            if ckpt != "best_monitor":
                continue
            other = "B" if arm == "R" else "R"
            swap = BC.predict(model, cache, rel, locked, device="cpu",
                              batch_size=int(BC.RECIPE["eval_batch_size"]), arm=other)
            with _patched(model, "zero_stats"):
                zero = BC.predict(model, cache, rel, locked, device="cpu",
                                  batch_size=int(BC.RECIPE["eval_batch_size"]))
            with _patched(model, "side_zero"):
                side0 = BC.predict(model, cache, rel, locked, device="cpu",
                                   batch_size=int(BC.RECIPE["eval_batch_size"]))

            # histogram-preserving relation permutation on a fixed subset
            batch = BD.collate(cache, rel, perm_subset)
            pr = batch["partner_rel"].numpy().copy()
            tgt = batch["partner_tgt"].numpy()
            for h in np.unique(tgt):
                idx = np.where(tgt == h)[0]
                if len(idx) > 1:
                    pr[idx] = pr[idx][rng.permutation(len(idx))]
            bperm = dict(batch)
            bperm["partner_rel"] = torch.from_numpy(pr)
            model.eval()
            with torch.no_grad():
                p0 = model(batch).numpy()
                p1 = model(bperm).numpy()
            base_sub = BC.predict(model, cache, rel, perm_subset, device="cpu",
                                  batch_size=int(BC.RECIPE["eval_batch_size"]))
            result["interventions"][tag] = {
                "checkpoint": "best_monitor",
                "swap_to_arm": other,
                "swap_locked_mae": BC.mae(swap, y),
                "swap_mean_abs_delta": float(np.mean(np.abs(np.asarray(swap) - arr))),
                "swap_max_abs_delta": float(np.max(np.abs(np.asarray(swap) - arr))),
                "zero_stats_locked_mae": BC.mae(zero, y),
                "zero_stats_mean_abs_delta": float(np.mean(np.abs(np.asarray(zero) - arr))),
                "side_zero_locked_mae": BC.mae(side0, y),
                "side_zero_mean_abs_delta": float(np.mean(np.abs(np.asarray(side0) - arr))),
                "rel_perm_subset_n": int(len(perm_subset)),
                "rel_perm_baseline_mae": BC.mae(base_sub, y[:len(perm_subset)]),
                "rel_perm_mean_abs_delta": float(np.mean(np.abs(p1 - p0))),
                "rel_perm_max_abs_delta": float(np.max(np.abs(p1 - p0))),
                "rel_perm_moves_fraction": float(np.mean(np.abs(p1 - p0) > 1e-6)),
            }
    result["frozen"] = True
    BC.write_json(lock_path, result)
    BC.log(f"[bosi-locked] wrote {lock_path} (one-time read complete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
