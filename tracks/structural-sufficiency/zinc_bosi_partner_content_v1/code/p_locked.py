"""The single, freeze-gated Locked read for the partner-content null.

Only ``P`` is evaluated on Locked here.  Every reused model (BOSI-B, BOSI-R,
DCC-L) is taken from the already-recorded, hash-verified v0 files; no
pre-existing model is ever run on Locked again.

Interventions on P's frozen weights (inference only):
  * ``swap_to_B`` / ``swap_to_R`` -- P's trained weights with the historical
    moment formula of the other arm (same parameters, different statistics)
  * ``zero_stats`` -- the 384-d M block of the side input forced to zero
  * ``side_zero``  -- the whole statistics+counts+closure block zeroed

Usage:
    python .../p_locked.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bosi_common as BC  # noqa: E402
import ad_common as C  # noqa: E402
import bosi_data as BD  # noqa: E402
import bosi_models as BM  # noqa: E402

M_START = 2 * BM.WIDTH + BM.RANK
M_STOP = M_START + BD.N_REL * 3 * BM.RANK

def _v0path(text):
    """Resolve a path recorded by the v0 round (relative to that round's root)."""
    p = Path(text)
    return p if p.is_absolute() else (BC.V0 / p)



def _verify(freeze: dict) -> None:
    cfg = json.loads((BC.CONFIGS / "bosi_p_v1.json").read_text())
    if BC.sha256_file(BC.CONFIGS / "bosi_p_v1.json") != freeze["config_sha256"]:
        raise SystemExit("config changed since the freeze")
    if BC.sha256_file(BC.ROUND / "PROTOCOL.md") != freeze["protocol_sha256"]:
        raise SystemExit("protocol changed since the freeze")
    if BC.sha256_file(BC.ROUND / "DESIGN.md") != freeze["design_sha256"]:
        raise SystemExit("design changed since the freeze")
    if BC.sha256_file(BC.DATA_CACHE) != freeze["data_cache_sha256"]:
        raise SystemExit("official-train cache changed since the freeze")
    rel = torch.load(BC.CACHE / "bosi_rel_train.pt", map_location="cpu", weights_only=False)
    if BD.relation_content_hash(rel) != freeze["reused"]["relation_cache"]["content_sha256"]:
        raise SystemExit("relation cache changed since the freeze")
    for name, entry in freeze["reused"].items():
        if name.startswith(("checkpoint/", "locked_pred/", "dccl/")):
            items = [entry] if "file" in entry else list(entry.values())
            for meta in items:
                if BC.sha256_file(_v0path(meta["file"])) != meta["sha256"]:
                    raise SystemExit(f"reused artefact changed: {name}")
    for tag, run in freeze["runs"].items():
        # P runs are recorded relative to this round; reused v0 runs are not
        resolve = BC.resolve_path if tag.startswith("P_") else _v0path
        for ckpt, entry in run["checkpoints"].items():
            if BC.sha256_file(resolve(entry["file"])) != entry["sha256"]:
                raise SystemExit(f"freeze mismatch on {tag}/{ckpt}")
    # the config itself must still describe the same design
    if cfg["design_sha256"] != freeze["design_sha256"]:
        raise SystemExit("config design hash differs from the frozen design")


def _patched(model, patch: str):
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
        raise SystemExit("run p_freeze.py first")
    freeze = json.loads((BC.ANALYSIS / "freeze.json").read_text())
    _verify(freeze)
    lock_path = BC.ANALYSIS / "locked_read.json"
    out_dir = BC.ANALYSIS / "locked_preds"
    if lock_path.exists():
        BC.log("[p-locked] locked_read.json exists; verifying and exiting")
        existing = json.loads(lock_path.read_text())
        for entry in existing["predictions"].values():
            for meta in entry.values():
                if BC.sha256_file(_v0path(meta["file"])) != meta["sha256"]:
                    raise SystemExit("locked prediction file changed")
        BC.log("[p-locked] existing artefacts verified, nothing recomputed")
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
        "training_models_evaluated": ["P"],
        "reused_models_not_reevaluated": ["B_s0", "B_s1", "R_s0", "R_s1", "DCC-L_s0", "DCC-L_s1"],
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "note": "internal Locked split, historically exposed; not a new blind test",
        "predictions": {},
        "interventions": {},
    }

    for tag, run in freeze["runs"].items():
        if not tag.startswith("P_"):
            continue
        seed = int(run["seed"])
        arm = run["arm"]
        result["predictions"][tag] = {}
        for ckpt, meta in run["checkpoints"].items():
            model, _ = BM.build_model(norm, arm, seed, "cpu")
            model.arm = arm
            blob = torch.load(BC.resolve_path(meta["file"]), map_location="cpu",
                              weights_only=False)
            C.load_state(model, blob["state_dict"], "cpu")
            pred = BC.predict(model, cache, rel, locked, device="cpu",
                              batch_size=int(BC.RECIPE["eval_batch_size"]))
            arr = np.asarray(pred, dtype=np.float32)
            path = out_dir / f"{tag}_{ckpt}.npz"
            np.savez(path, row=locked, pred=arr, arm=arm, seed=seed,
                     checkpoint=ckpt, y=y.astype(np.float32))
            result["predictions"][tag][ckpt] = {
                "file": BC.rel_or_abs(path), "sha256": BC.sha256_file(path),
                "locked_mae": BC.mae(arr, y),
                "checkpoint_sha256": meta["sha256"],
            }
            BC.log(f"[p-locked] {tag}/{ckpt} locked MAE {BC.mae(arr, y):.9f}")

            if ckpt != "best_monitor":
                continue
            inter = {"checkpoint": "best_monitor"}
            for other in ("B", "R"):
                swapped = BC.predict(model, cache, rel, locked, device="cpu",
                                     batch_size=int(BC.RECIPE["eval_batch_size"]),
                                     arm=other)
                inter[f"swap_to_{other}_locked_mae"] = BC.mae(swapped, y)
                inter[f"swap_to_{other}_mean_abs_delta"] = float(
                    np.mean(np.abs(np.asarray(swapped) - arr)))
            with _patched(model, "zero_stats"):
                zero = BC.predict(model, cache, rel, locked, device="cpu",
                                  batch_size=int(BC.RECIPE["eval_batch_size"]))
            with _patched(model, "side_zero"):
                side0 = BC.predict(model, cache, rel, locked, device="cpu",
                                   batch_size=int(BC.RECIPE["eval_batch_size"]))
            inter["zero_stats_locked_mae"] = BC.mae(zero, y)
            inter["zero_stats_mean_abs_delta"] = float(np.mean(np.abs(np.asarray(zero) - arr)))
            inter["side_zero_locked_mae"] = BC.mae(side0, y)
            inter["side_zero_mean_abs_delta"] = float(np.mean(np.abs(np.asarray(side0) - arr)))
            result["interventions"][tag] = inter

    result["frozen"] = True
    BC.write_json(lock_path, result)
    BC.log(f"[p-locked] wrote {lock_path} (one-time read complete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
