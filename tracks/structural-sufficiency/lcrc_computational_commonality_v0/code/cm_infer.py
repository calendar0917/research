"""Materialise the missing frozen Valid/Test logits (read-only forwards).

Only the five core/one extended models whose logits were never written to disk
need this.  Every forward uses the historical constructor of its own round and
loads the recorded best-Validation checkpoint; the replay AP is gated against
the recorded Validation AP before the logits are kept.

``uv run python code/cc_infer.py parta --device cuda``
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import cm_common as C
import dy_mechanism as MECH  # historical W2-DYNAMIC loader/predictor


def _payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"checkpoint missing: {C.rel(path)}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _build(label: str, task: Any, device: torch.device):
    spec = C.MODELS[label]
    fam = spec["family"]
    if fam == "bs":
        model, audit = C.DM.build_w2(task.spec, task.norm, C.N_TASKS, seed=int(spec["seed"]), device="cpu")
        payload = _payload(spec["checkpoint"])
        model.load_state_dict(payload["state_dict"], strict=True)
    elif fam == "ds":
        arm = spec["arch"]
        model, audit = C.DM.build_ds_model(
            task.spec, task.norm, C.N_TASKS, arm=arm, seed=int(spec["seed"]), device="cpu"
        )
        payload = _payload(spec["checkpoint"])
        model.load_state_dict(payload["state_dict"], strict=True)
    elif fam == "dy":
        model, audit, _sha = MECH.load_frozen(spec["checkpoint"], task, device="cpu")
        payload = {"epoch": int(spec["best_epoch"])}
    else:  # pragma: no cover
        raise KeyError(f"{label}: family {fam} needs an external structure cache")
    model.to(device)
    model.eval()
    return model, audit, payload


def _pred(model: Any, task: Any, rows: np.ndarray, device: torch.device) -> np.ndarray:
    """One read-only forward on the frozen LCRC cache batch layout."""
    return C.DC.PC.predict_logits(model, task.cache, np.asarray(rows, dtype=np.int64), device, batch_size=256)


def run(labels: tuple[str, ...], *, device: str = "cpu", tolerance: float = 2e-4) -> dict[str, Any]:
    toc = time.perf_counter()
    task = C.DC.SD.load_task()
    dev = torch.device(device)
    C.PREDICTIONS.mkdir(parents=True, exist_ok=True)
    y_va = np.asarray(task.labels[task.valid_rows], dtype=np.float32)
    y_te = C.y_test()
    records: dict[str, Any] = {}
    for label in labels:
        spec = C.MODELS[label]
        t0 = time.perf_counter()
        model, audit, payload = _build(label, task, dev)
        t_build = time.perf_counter() - t0
        entry: dict[str, Any] = {
            "label": label,
            "arch": spec["arch"],
            "seed": int(spec["seed"]),
            "checkpoint": C.rel(spec["checkpoint"]),
            "checkpoint_sha256": C.sha256_file(spec["checkpoint"]),
            "checkpoint_payload_epoch": int(payload.get("epoch", -1)),
            "parameters_recorded": int(spec["parameters"]),
            "parameters_built": int(sum(p.numel() for p in model.parameters())),
            "device": str(dev),
            "build_seconds": float(t_build),
        }
        for split, y in (("valid", y_va), ("test", y_te)):
            target = C.inference_path(label, split)
            if target.is_file():
                entry[f"{split}_reused"] = True
                entry[f"{split}_path"] = C.rel(target)
                continue
            t1 = time.perf_counter()
            rows = np.asarray(task.valid_rows if split == "valid" else task.test_rows, dtype=np.int64)
            logits = _pred(model, task, rows, dev)
            seconds = time.perf_counter() - t1
            if logits.shape != (rows.size, C.N_TASKS):
                raise RuntimeError(f"{label}/{split} logits shape {logits.shape}")
            ap = C.macro_ap(y, C.score_logits(logits))
            tmp = target.with_suffix(".npy.tmp")
            with open(tmp, "wb") as fh:
                np.save(fh, np.ascontiguousarray(logits, dtype=np.float32))
            tmp.replace(target)
            entry[f"{split}_path"] = C.rel(target)
            entry[f"{split}_logits_sha256"] = C.sha256_array(logits)
            entry[f"{split}_macro_ap_replay"] = float(ap["macro_ap"])
            entry[f"{split}_positive_support"] = int(sum(ap["positive"]))
            entry[f"{split}_seconds"] = float(seconds)
            if split == "valid":
                delta = abs(float(ap["macro_ap"]) - float(spec["valid_ap_recorded"]))
                entry["valid_ap_recorded"] = float(spec["valid_ap_recorded"])
                entry["valid_ap_abs_delta"] = float(delta)
                if delta > tolerance:
                    raise RuntimeError(
                        f"{label}: Validation replay {ap['macro_ap']:.9f} != recorded "
                        f"{spec['valid_ap_recorded']:.9f} (delta {delta:.2e} > {tolerance})"
                    )
        del model
        records[label] = entry
        C.write_json(C.LOCAL / "inference_manifest.json", {
            "schema": "cc-frozen-inference-v1",
            "created_utc": C.utcnow(),
            "test_role": C.TEST_ROLE,
            "device": str(dev),
            "torch": torch.__version__,
            "labels": list(labels),
            "records": records,
            "seconds": float(time.perf_counter() - toc),
        })
        print(f"[infer] {label} done {entry.get('valid_macro_ap_replay')} / {entry.get('test_macro_ap_replay')}", flush=True)
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["parta"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--labels", default=",".join(C.NEEDS_INFERENCE))
    args = ap.parse_args(argv)
    labels = tuple(x for x in args.labels.split(",") if x)
    run(labels, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
