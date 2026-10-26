"""The single, freeze-gated Locked read of this round.

Only the models trained here are evaluated (BOSI-SSL s0/s1 and, if purchased,
the compute control C s0/s1).  BOSI-B is never re-run: its Locked predictions
are the frozen, hash-verified BOSI v0 arrays.

Descriptive inference-only interventions on the BOSI-SSL checkpoints (they do
not enter the decision rule):

* ``zero_stats``      -- the 384-d partner-moment block of the side input zeroed
* ``partner_shuffle`` -- ``partner_half`` permuted inside each eval batch (same
  relation counts, random partner contents)

Locked is an internal, historically exposed split; this is not a new blind test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import s_common as SC  # noqa: E402
import s_pretext as SP  # noqa: E402
import ad_common as C  # noqa: E402

EVAL_BATCH = int(SC.BC.RECIPE["eval_batch_size"])


def _verify(freeze: dict) -> None:
    if SC.sha256_file(SC.CONFIGS / "bosi_stat_ssl_v1.json") != freeze["config_sha256"]:
        raise SystemExit("config changed since the freeze")
    if SC.sha256_file(SC.ROUND / "PROTOCOL.md") != freeze["protocol_sha256"]:
        raise SystemExit("protocol changed since the freeze")
    if SC.sha256_file(SC.DATA_CACHE) != freeze["data_cache_sha256"]:
        raise SystemExit("official-train cache changed since the freeze")
    rel = SC.load_relation_cache()
    if SC.BD.relation_content_hash(rel) != freeze["relation_cache_sha256"]:
        raise SystemExit("relation cache changed since the freeze")
    for tag, entry in freeze["runs"].items():
        for ckpt, meta in entry["checkpoints"].items():
            if SC.sha256_file(SC.resolve_path(meta["file"])) != meta["sha256"]:
                raise SystemExit(f"freeze mismatch on {tag}/{ckpt}")
    for tag, entry in freeze["pretrain"].items():
        if SC.sha256_file(SC.resolve_path(entry["state_file"])) != entry["state_sha256"]:
            raise SystemExit(f"pretraining state changed: {tag}")


@torch.no_grad()
def _predict(model, cache, rel, rows, device) -> np.ndarray:
    return SC.BC.predict(model, cache, rel, rows, device=device, batch_size=EVAL_BATCH)


@torch.no_grad()
def _predict_shuffled(model, cache, rel, rows, device) -> np.ndarray:
    """Same batches, ``partner_half`` permuted (partner contents scrambled)."""
    model.eval()
    out = []
    for start in range(0, len(rows), EVAL_BATCH):
        chunk = np.asarray(rows[start:start + EVAL_BATCH], dtype=np.int64)
        batch = SC.BD.collate(cache, rel, chunk)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch = SP.shuffle_partner_stats(batch, seed=7 + start)
        out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out, 0)


def main(argv=None) -> int:
    if not (SC.ANALYSIS / "freeze.json").exists():
        raise SystemExit("run s_freeze.py first")
    freeze = json.loads((SC.ANALYSIS / "freeze.json").read_text())
    _verify(freeze)
    lock_path = SC.ANALYSIS / "locked_read.json"
    out_dir = SC.ANALYSIS / "locked_preds"
    if lock_path.exists():
        SC.log("[s-locked] locked_read.json exists; verifying and exiting")
        existing = json.loads(lock_path.read_text())
        for entry in existing["predictions"].values():
            for meta in entry.values():
                p = SC.resolve_path(meta["file"])
                if SC.sha256_file(p) != meta["sha256"]:
                    raise SystemExit("locked prediction file changed")
        SC.log("[s-locked] existing artefacts verified, nothing recomputed")
        return 0

    cache = SC.load_train_cache()
    rel = SC.load_relation_cache()
    rows = SC.split_rows()
    locked = rows["locked"]
    y = SC.labels(cache, locked)
    norm = SC.fit_only_norm(cache)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "read_utc": SC.now(),
        "round": SC.ROUND_ID,
        "locked_rows_sha256": rows["hashes"]["locked"],
        "n_locked": int(len(locked)),
        "training_models_evaluated": sorted(freeze["runs"].keys()),
        "reused_models_not_reevaluated": ["B_s0", "B_s1"],
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "note": "internal Locked split, historically exposed; not a new blind test",
        "predictions": {},
        "interventions": {},
    }

    for tag, run in freeze["runs"].items():
        seed = int(run["seed"])
        result["predictions"][tag] = {}
        for ckpt, meta in run["checkpoints"].items():
            model, _ = SP.build_backbone(norm, seed, "cpu")
            blob = torch.load(SC.resolve_path(meta["file"]), map_location="cpu",
                              weights_only=False)
            C.load_state(model, blob["state_dict"], "cpu")
            pred = np.asarray(_predict(model, cache, rel, locked, "cpu"), dtype=np.float32)
            path = out_dir / f"{tag}_{ckpt}.npz"
            np.savez(path, row=locked, pred=pred, seed=seed, checkpoint=ckpt,
                     init=run["init"]["init"], y=y.astype(np.float32))
            result["predictions"][tag][ckpt] = {
                "file": SC.rel_or_abs(path), "sha256": SC.sha256_file(path),
                "locked_mae": SC.BC.mae(pred, y),
                "checkpoint_sha256": meta["sha256"],
                "step": meta["step"],
            }
            SC.log(f"[s-locked] {tag}/{ckpt} locked MAE {SC.BC.mae(pred, y):.9f}")

            if ckpt != "best_monitor":
                continue
            inter = {"checkpoint": "best_monitor"}
            with SP.zero_moment_block(model):
                zero = np.asarray(_predict(model, cache, rel, locked, "cpu"),
                                  dtype=np.float32)
            inter["zero_stats_locked_mae"] = SC.BC.mae(zero, y)
            inter["zero_stats_mean_abs_delta"] = float(np.mean(np.abs(zero - pred)))
            shuf = _predict_shuffled(model, cache, rel, locked, "cpu")
            inter["partner_shuffle_locked_mae"] = SC.BC.mae(shuf, y)
            inter["partner_shuffle_mean_abs_delta"] = float(np.mean(np.abs(shuf - pred)))
            inter["partner_shuffle_fraction_moved"] = float(np.mean(np.abs(shuf - pred) > 1e-6))
            result["interventions"][tag] = inter

    result["frozen"] = True
    SC.write_json(lock_path, result)
    SC.log(f"[s-locked] wrote {lock_path} (one-time read complete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
