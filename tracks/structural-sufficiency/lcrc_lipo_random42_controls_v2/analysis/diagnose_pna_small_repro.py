"""Diagnostic: does the PNA-small checkpoint reproduce its saved Validation predictions?

Evaluation-only; reads Train/Validation structures and the new checkpoint.  No
Test artifact is touched.  Writes analysis/pna_small_repro_diag.json.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parent.parent / "code"
sys.path.insert(0, str(CODE))

import c2_common as C  # noqa: E402
import c2_data as CD  # noqa: E402
import c2_eval as CE  # noqa: E402
import c2_models as CM  # noqa: E402
import r42_data as D  # noqa: E402
import r42_metrics as MET  # noqa: E402
import r42_native_train as NT  # noqa: E402


def main() -> int:
    device = torch.device("cuda", int(torch.cuda.current_device()))
    split, labels = CD.load_frozen_split_and_labels()
    rows_train = split["rows"]["train"]
    rows_valid = split["rows"]["valid"]
    train_data = D.load_structure_data(rows=rows_train)
    valid_data = D.load_structure_data(rows=rows_valid)
    cfg = CM.pna_small_config()
    target_norm = D.fit_target_scaler(labels.y_train)
    saved = np.load(C.RUNS / "pna_small_s45_s0/valid_predictions.npz")["pred"]
    out = {"device": torch.cuda.get_device_name(device), "runs": {}}
    for precision in ("bf16", "fp32"):
        with tempfile.TemporaryDirectory(prefix="pna_repro_") as tmp:
            model, _ = NT.build_pna(
                cfg,
                device,
                Path(tmp),
                train_data,
                int(valid_data[0].x.shape[1]),
                int(valid_data[0].edge_attr.shape[1]),
            )
            CE._load_checkpoint(model, C.RUNS / "pna_small_s45_s0/checkpoint.pt", device)
            pred = NT.denormalise(
                NT._predict(
                    model,
                    valid_data,
                    kind="pna",
                    device=device,
                    batch_size=int(cfg["batch_size"]),
                    precision=precision,
                ),
                target_norm,
            )
            out["runs"][precision] = {
                "max_abs_diff_vs_saved": float(np.max(np.abs(pred - saved))),
                "rmse_vs_saved": float(np.sqrt(np.mean((pred - saved) ** 2))),
                "metrics": MET.all_metrics(pred, labels.y_valid),
            }
        del model
        torch.cuda.empty_cache()
    C.write_json(C.ANALYSIS / "pna_small_repro_diag.json", out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
