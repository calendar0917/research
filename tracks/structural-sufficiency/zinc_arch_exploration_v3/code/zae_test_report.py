"""Test evaluation of the frozen best-valid checkpoints (zinc-arch-exploration-v3).

Two steps, deliberately separate:

``freeze``  writes ``analysis/test_freeze.json``: the exact list of the round's
            formal arms, their best-valid epoch, checkpoint path and sha256, and
            the valid numbers.  It also records the reference models that are
            reused (not retrained).  Nothing from the official test split is read.
``report``  reads the freeze file (refusing if a checkpoint hash no longer
            matches), builds the test pair cache from raw official test graphs,
            runs exactly those checkpoints, and writes the test table.  The test
            split is an already-exposed benchmark in this repository; these
            numbers are descriptive reporting, not a blind confirmatory read,
            and they cannot change any checkpoint or recipe of this round.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SUF = _EXP.parent
_V1 = _SUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import zae_data as ZD  # noqa: E402
import zae_models as ZM  # noqa: E402
import zae_runner as ZR  # noqa: E402
import zae_analysis as ZA  # noqa: E402
import zae_ssl as ZS  # noqa: E402
import zae_train as ZT  # noqa: E402

ANALYSIS = _EXP / "analysis"
RUNS = _EXP / "runs"
DCC = _SUF / "zinc_dual_channel_composer_v1"
FREEZE = ANALYSIS / "test_freeze.json"
ARMS = ("A_s0", "B_s0", "A_SSL_s0")
REFS = ("P3B_s0", "L_s0", "C_s0")
BOOT = 10_000
BOOT_SEED = 20261022


def freeze() -> dict:
    if FREEZE.exists():
        C.log(f"freeze file already exists: {FREEZE} (not overwritten)")
        return C.read_json(FREEZE)
    rows = []
    for arm in ARMS:
        run = RUNS / arm
        if not (run / "train.json").exists():
            raise SystemExit(f"{arm}: run incomplete; freeze only after all formal runs are done")
        info = C.read_json(run / "train.json")
        rows.append({
            "model": arm,
            "arm": arm.replace("_s0", ""),
            "best_epoch": int(info["best_valid"]["epoch"]),
            "valid_mae": float(info["fixed_objects"]["best"]["valid_mae"]),
            "checkpoint": str((run / "best.pt").relative_to(C.REPO)),
            "checkpoint_sha256": C.sha256_file(run / "best.pt"),
            "parameter_count": int(info["parameter_count"]),
            "recipe": {k: info["recipe"][k] for k in
                       ("epochs", "batch_size", "lr", "min_lr", "weight_decay", "updates")},
            "ssl_pretrain": info.get("extra", {}).get("ssl_pretrain"),
            "selection": "best official-valid MAE, strict <, earlier epoch wins a tie",
        })
    payload = {
        "round": ZT.ROUND_ID,
        "protocol_id": ZT.PROTOCOL_ID,
        "frozen_at": C.utcnow(),
        "git_commit": C.git_commit(),
        "models": rows,
        "references_reused_not_retrained": {
            "P3B_s0": "zinc_recipe_trunk_v2/runs/xBssl_P3N_s0 (best-valid 0.076223751 @ 377)",
            "L_s0": "zinc_dual_channel_composer_v1/runs/L_s0 (best-valid 0.078537150 @ 489)",
            "C_s0": "zinc_dual_channel_composer_v1/runs/C_s0 (best-valid 0.082367432 @ 482)",
        },
        "test_read_policy": (
            "reporting only: no checkpoint, epoch, recipe, subgroup or vote is changed after this "
            "read; the official test split has been read in earlier rounds and is not a blind test"
        ),
    }
    C.write_json(FREEZE, payload)
    C.log(f"froze {len(rows)} checkpoints -> {FREEZE}")
    return payload


def _load_model(arm: str, device):
    norm = D.load_norm()
    base_arm = "A_SSL" if arm == "A_SSL_s0" else arm.replace("_s0", "")
    model, info = ZM.build_model(base_arm, norm, 0, device)
    return model, info


def report() -> int:
    frozen = C.read_json(FREEZE)
    for row in frozen["models"]:
        path = C.REPO / row["checkpoint"]
        if not path.exists():
            raise SystemExit(f"missing checkpoint {path}")
        if C.sha256_file(path) != row["checkpoint_sha256"]:
            raise SystemExit(f"checkpoint hash mismatch for {path}; refusing to score")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # pair cache for the test split (label-free; built from raw official graphs)
    if not ZD.pair_cache_path("test").exists():
        ZD.save_pair_split(ZD.build_pair_split("test"), "test")
    test_cache = D.load_cache("test")
    test_pair = ZD.load_pair_split("test")
    y = np.asarray(test_cache["labels"], dtype=np.float64)
    rows = np.arange(len(y))
    result = {"round": ZT.ROUND_ID, "protocol_id": ZT.PROTOCOL_ID,
              "frozen": frozen, "models": {}, "contrasts": {}}
    preds = {}
    for row in frozen["models"]:
        arm = row["model"]
        model, _info = _load_model(arm, device)
        blob = torch.load(C.REPO / row["checkpoint"], map_location="cpu", weights_only=False)
        C.load_state(model, blob["state_dict"], device)
        pred, mae = ZT.evaluate(model, test_cache, test_pair, rows, device=device, batch_size=128)
        preds[arm] = pred.astype(np.float64)
        result["models"][arm] = {
            **ZA.metrics(preds[arm], y),
            "valid_mae": row["valid_mae"],
            "best_epoch": row["best_epoch"],
            "parameter_count": row["parameter_count"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "mae_recomputed": float(mae),
        }
        C.log(f"[test] {arm} MAE={mae:.9f} (valid {row['valid_mae']:.9f})")
        np.savez(ANALYSIS / "test_preds" / f"{arm}_test.npz", row=rows, pred=pred)
    # reference predictions (existing files, not recomputed)
    for ref in REFS:
        npz = DCC / "analysis" / "test_preds" / f"{ref}_test.npz"
        preds[ref] = ZA.load_npz_pred(npz, len(y))
        result["models"][ref] = {**ZA.metrics(preds[ref], y), "source": str(npz.relative_to(C.REPO))}
    for first in ARMS:
        for second in ("P3B_s0", "L_s0"):
            result["contrasts"][f"{first} - {second}"] = ZA.paired_bootstrap(
                preds[first], preds[second], y, seed=BOOT_SEED)
        for second in ARMS:
            if second != first:
                result["contrasts"][f"{first} - {second}"] = ZA.paired_bootstrap(
                    preds[first], preds[second], y, seed=BOOT_SEED)
    # subgroups with the frozen test masks from the dual-channel round
    tpm = list(csv.DictReader(open(DCC / "analysis" / "test_per_molecule.csv", encoding="utf-8")))
    if len(tpm) != len(y):
        raise RuntimeError("test_per_molecule.csv row count mismatch")
    masks = {
        "signature_unseen": np.asarray([int(r["signature_unseen"]) == 1 for r in tpm]),
        "signature_covered": np.asarray([int(r["signature_unseen"]) == 0 for r in tpm]),
        "ordinary_y_gt_-3": np.asarray([int(r["ordinary_y_gt_minus3"]) == 1 for r in tpm]),
        "tail_y_le_-3": np.asarray([int(r["tail_y_le_minus3"]) == 1 for r in tpm]),
        "rings_ge4": np.asarray([int(r["rings_ge4"]) == 1 for r in tpm]),
        "rings_ge4_and_unseen": np.asarray([
            int(r["rings_ge4"]) == 1 and int(r["signature_unseen"]) == 1 for r in tpm]),
        "P3B_test_worst100": np.asarray([int(r["P3B_test_worst100"]) == 1 for r in tpm]),
    }
    subgroup = {}
    names = list(ARMS) + list(REFS)
    for g, mask in masks.items():
        entry = {"n": int(mask.sum())}
        for k in names:
            entry[k] = float(np.abs(preds[k] - y)[mask].mean())
        subgroup[g] = entry
    result["subgroups"] = subgroup
    C.write_json(ANALYSIS / "test_reporting.json", result)
    lines = ["| model | params | best valid | best ep | test MAE | test RMSE | median AE | p95 AE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k in ("P3B_s0", "L_s0", "C_s0") + ARMS:
        m = result["models"][k]
        lines.append(f"| {k} | {m.get('parameter_count', '')} | {m.get('valid_mae', float('nan')):.9f} | "
                     f"{m.get('best_epoch', '')} | {m['mae']:.9f} | {m['rmse']:.6f} | "
                     f"{m['median_ae']:.6f} | {m['p95_ae']:.6f} |")
    (ANALYSIS / "test_main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    C.log("\n".join(lines))
    C.log(json.dumps(result["contrasts"], indent=1)[:3000])
    return 0


def main(argv=None) -> int:
    p = __import__("argparse").ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["freeze", "report"])
    args = p.parse_args(argv)
    (ANALYSIS / "test_preds").mkdir(parents=True, exist_ok=True)
    if args.command == "freeze":
        freeze()
        return 0
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
