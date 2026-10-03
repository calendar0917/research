"""Deterministic replay of the reduced deployment checkpoints.

Checks, on the full 8000 fit + 2000 dev stream:
  * ``S_M_deploy_state.pt`` reproduces the old S_M raw soup prediction;
  * ``N0_deploy_state.pt`` reproduces the new N0 raw soup prediction;
  * the reduced model is independently loadable from its state dict alone
    (the full models are never required for the deploy path).

Run from the repo root:
    PYTHONPATH=. uv run python tracks/ksvd/results/zinc_zero_binding_baseline_seed0_v1/replay_deploy.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import zinc_structure_semantic_factorial_seed0_v1 as zsf
from tracks.ksvd.experiments.luyin16 import zinc_zero_binding_baseline_seed0_v1 as z0

RESULTS_DIR = z0.RESULTS_DIR
TOL = z0.COMPRESS_TOL


def main() -> int:
    torch.set_num_threads(8)
    blob, decomp, _train_data, fit_data, dev_data, fit_idx, dev_idx, y = z0._load_source_fit_dev()
    y_fit = torch.as_tensor(y[fit_idx], dtype=torch.float32)
    y_dev = torch.as_tensor(y[dev_idx], dtype=torch.float32)
    device = torch.device("cpu")
    report: dict[str, object] = {"official_test_loaded": False, "tolerance": TOL}

    for name, soup_file, factory in (
        ("S_M", "S_M_deploy_state.pt", lambda: zsf.build_factorial_model(blob, zsf.SEED, "sparse", "indep")),
        ("N0", "N0_deploy_state.pt", lambda: z0.build_zero_model(blob, z0.SEED, zero_binding_enabled=True)),
    ):
        deploy = z0.build_deploy_model(torch.load(RESULTS_DIR / soup_file, map_location=device, weights_only=False), device=device)
        deploy.eval()
        full = factory().to(device)
        source_state = (
            z0.load_source_state("S_M", device)
            if name == "S_M"
            else torch.load(RESULTS_DIR / "N0_raw_soup_state.pt", map_location=device, weights_only=False)
        )
        full.load_state_dict(source_state)
        full.eval()
        entry: dict[str, object] = {}
        for split, data, target in (("fit", fit_data, y_fit), ("dev", dev_data, y_dev)):
            ref, _ = zsf.predict(full, data, target, device)
            got, _ = zsf.predict(deploy, data, target, device)
            entry[f"{split}_max_abs_diff"] = float(np.max(np.abs(ref - got)))
        entry["pass"] = bool(max(float(v) for v in entry.values() if isinstance(v, float)) <= TOL)
        entry["deploy_parameters"] = int(sum(p.numel() for p in deploy.parameters()))
        report[name] = entry

    report["pass"] = bool(all(report[k]["pass"] for k in ("S_M", "N0")))
    Path(RESULTS_DIR / "replay_deploy.json").write_text(json.dumps(z0.jsonable(report), indent=2), encoding="utf-8")
    print(json.dumps(z0.jsonable(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
