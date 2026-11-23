"""Independent replay of saved best checkpoints against stored Dev predictions."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
import torch

import sr_common as C
import sr_data as D
import sr_eval as E
import sr_fresponse as FR
import sr_struct as ST

sys.path.insert(0, str(C.ROUND_DIR.parent / "r2_sentence_replication_v1" / "code"))
import r2_common as RC  # noqa: E402
import r2_data as RD  # noqa: E402
import gwic_model as GM  # noqa: E402
import gwic_structure as GS  # noqa: E402
import gwic_tensor as GT  # noqa: E402

RUNS = [
    "R2_REF_D0_S0", "E_MOB_D0_S0", "F_MOB_D0_S0", "F_RAW_D0_S0",
    "G_FULL5_D0_S0", "G_EGO2_5_D0_S0", "F_MOB_BIG_D0_S0",
    "R2_REF_D1_S0", "E_MOB_D1_S0", "F_MOB_D1_S0", "F_RAW_D1_S0",
    "G_FULL5_D1_S0", "G_EGO2_5_D1_S0", "F_MOB_BIG_D1_S0",
]


def replay_r2(run: str, split: str, manifest: dict[str, Any]) -> dict[str, Any]:
    run_dir = C.ROUND_DIR / "results" / run
    bundle = RD.load_bundle(split, manifest["atom_scalar_mode"])
    atom_dims, bond_dims = RD.GD.atom_bond_dims()
    blob = torch.load(run_dir / "state_best.pt", map_location="cpu",
                      weights_only=False)
    model = GM.load_model_from_state(atom_dims, bond_dims, blob["state"],
                                     RD.norm_of(bundle), use_words=False,
                                     config=RC.frozen_config(), device="cpu")
    rows = np.union1d(bundle.fit_rows, bundle.eval_rows)
    structs = GS.build_all(bundle.raw, rows)
    preds = {}
    for name, part in (("dev", bundle.eval_rows), ("fit", bundle.fit_rows)):
        out = np.zeros(part.size, np.float64)
        model.eval()
        with torch.no_grad():
            for start in range(0, part.size, 256):
                chunk = part[start:start + 256]
                tensors = GT.collate(chunk.tolist(), structs, bundle.raw, "cpu")
                out[start:start + chunk.size] = (
                    model(tensors, arm="SENTENCE_BASE")["yhat_std"].double().numpy())
        preds[name] = out * float(manifest["scaler"]["std"]) + float(manifest["scaler"]["mean"])
    return {"pred_dev": preds["dev"], "pred_fit": preds["fit"]}


def replay_sr(run: str, split: str, manifest: dict[str, Any]) -> dict[str, Any]:
    bundle = D.load_bundle(split)
    arm = manifest["arm"]
    cfg = C.load_config()
    rows = np.union1d(bundle.fit_rows, bundle.eval_rows)
    structs = ST.build_structures(bundle, rows.tolist())
    atom_dims, bond_dims = D.feature_dims()
    registry = None
    if arm in ("F_RAW", "F_MOB", "F_MOB_BIG"):
        reg = ST.build_registry([structs[int(r)] for r in rows.tolist()],
                                atom_dims, bond_dims)
        registry = FR.registry_tensors(reg, "cpu")
    model, _ = E.load_trained(f"results/{run}", arm, bundle, cfg,
                              device="cpu", registry=registry)
    pred_dev, _ = E.predict_raw(model, arm, bundle.eval_rows, structs, bundle.raw,
                                "cpu", 256, manifest["scaler"])
    pred_fit, _ = E.predict_raw(model, arm, bundle.fit_rows, structs, bundle.raw,
                                "cpu", 256, manifest["scaler"])
    return {"pred_dev": pred_dev, "pred_fit": pred_fit}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=RUNS)
    args = ap.parse_args()
    out = []
    for run in args.runs:
        run_dir = C.ROUND_DIR / "results" / run
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        split = manifest.get("split_name") or manifest.get("split")
        if isinstance(split, dict):
            split = split["name"]
        split = str(split)
        print("replay", run, split)
        res = (replay_r2(run, split, manifest) if run.startswith("R2_REF")
               else replay_sr(run, split, manifest))
        stored = np.load(run_dir / "predictions_best_dev.npz")
        stored_fit = np.load(run_dir / "predictions_best_fit.npz")
        dev = np.sqrt(np.mean((res["pred_dev"] - stored["label"]) ** 2))
        fit = np.sqrt(np.mean((res["pred_fit"] - stored_fit["label"]) ** 2))
        rec = {
            "run_id": run, "split": split,
            "dev_rmse_replay": float(dev),
            "dev_rmse_manifest": float(manifest["dev_rmse"]),
            "dev_abs_delta": abs(float(dev) - float(manifest["dev_rmse"])),
            "fit_rmse_replay": float(fit),
            "fit_rmse_manifest": float(manifest["train_rmse_at_best"]),
            "fit_abs_delta": abs(float(fit) - float(manifest["train_rmse_at_best"])),
            "max_abs_delta_stored_dev": float(np.max(np.abs(res["pred_dev"] - stored["pred"]))),
            "max_abs_delta_stored_fit": float(np.max(np.abs(res["pred_fit"] - stored_fit["pred"]))),
        }
        rec["ok"] = bool(rec["dev_abs_delta"] < 1e-4 and rec["fit_abs_delta"] < 1e-4
                         and rec["max_abs_delta_stored_dev"] < 2e-3
                         and rec["max_abs_delta_stored_fit"] < 2e-3)
        print(" ", rec["ok"], rec["dev_abs_delta"], rec["max_abs_delta_stored_dev"])
        out.append(rec)
    previous = {}
    path = C.ANALYSIS_DIR / "replay.json"
    if path.is_file():
        prior = json.loads(path.read_text(encoding="utf-8"))
        previous = {r["run_id"]: r for r in prior.get("runs", [])}
    previous.update({r["run_id"]: r for r in out})
    merged = [previous[k] for k in sorted(previous)]
    C.write_json(path, {"runs": merged,
                        "all_ok": all(r["ok"] for r in merged)})
    print("all_ok", all(r["ok"] for r in out))
    return 0 if all(r["ok"] for r in out) else 1


if __name__ == "__main__":
    raise SystemExit(main())