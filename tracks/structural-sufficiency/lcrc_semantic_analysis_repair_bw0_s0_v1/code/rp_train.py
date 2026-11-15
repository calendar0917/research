"""The single new formal training: ``B_N100_lipo_W0``.

Everything scientific is delegated to the frozen predecessors:

* ``se_train.train_run`` — the exact old training loop (per-step cosine, the
  shared ``round(j*U/50)`` grid, full three-split evaluation every epoch, RNG
  snapshot/restore around evaluation);
* ``LPC.C1_RECIPE`` — AdamW / peak 3e-4 / floor 1e-5 / batch 128 / dropout 0.10 /
  clip 5 / 360 epochs / 9720 updates / ``shuffle_offset`` 917000;
* ``se_common.RECIPES['W0']`` — the only scientific change, ``weight_decay=0``
  (the ``B_N100_lipo_Rwd`` template uses 1e-2, otherwise identical);
* ``lp_model.build_arm(spec, norm, 'B')`` — the original C1 / NO_BOND_MEAN model,
  228 674 parameters, no generator, no materialised tables, no new layers.

**One documented adaptation.** ``se_train`` builds the C1 base with the process
global RNG and never seeds it, so the old runs' initial weights are *not*
reproducible and *not* shared across families (recorded in
``analysis/init_reproducibility.json``).  This round therefore loads the frozen,
documented C1 initial state — ``dissect_model.build_full_state(spec, norm, 0)``
then ``remap_state('C1', ...)``, i.e. ``lp_model.shared_c1_init_state`` — into
``model.base`` before training starts.  That is the protocol the HOME round
already used for the original ``B`` arm, so ``B_W0`` is initialised exactly like
``B_R``.  ``B_Rwd`` cannot be shown to share it; that comparison is reported as
carried by *different* initial states and the limitation is stated, not hidden.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import rp_common as RP

ADAPTATION = {
    "name": "deterministic_shared_c1_init",
    "why": (
        "se_train never seeds the process RNG before building the C1 base, so the initial weights "
        "differ per process and cannot be shared across arms; build_full_state(seed=0)+remap_state('C1') "
        "is the frozen, documented C1 init used by the HOME round's original B arm"
    ),
    "effect_on_comparison": (
        "B_W0 vs B_R share an initial state by construction; B_Rwd (old round) cannot be shown to "
        "share it, so the wd 1e-2 coordinate is compared with a stated initial-state caveat"
    ),
}


def install() -> dict[str, Any]:
    """Monkeypatch the frozen loop in place: seed-shared init + this round's identity."""
    import lp_common as LPC
    import lp_model as LPM
    import se_common as C
    import se_train as TR

    original_build = TR.build_model

    def build_model_with_shared_init(cfg, view, permutations):  # type: ignore[no-untyped-def]
        model = original_build(cfg, view, permutations)
        if cfg["family"] == "B":
            state = LPM.shared_c1_init_state(view.spec, view.train_norm)
            model.base.load_state_dict(state, strict=True)
        return model

    TR.build_model = build_model_with_shared_init
    C.PROTOCOL_ID = RP.PROTOCOL_ID
    C.ROUND_ID = RP.PROTOCOL_ID
    C.STUDY = RP.PROTOCOL_ID
    return {
        "protocol_id": RP.PROTOCOL_ID,
        "old_protocol_id": "lcrc-explicit-semantic-learning-8h-s0-v1",
        "init_adaptation": ADAPTATION,
        "recipe": dict(C.RECIPES["W0"]),
        "c1_recipe": {k: (list(v) if isinstance(v, tuple) else v) for k, v in LPC.C1_RECIPE.items()},
        "expected_parameters": RP.C1_PARAMS,
    }


def resolved_config(device: str) -> dict[str, Any]:
    return {
        "protocol_id": RP.PROTOCOL_ID,
        "stage": "train",
        "family": "B",
        "recipe": "W0",
        "dataset": "lipo",
        "budget": "N100",
        "seed": 0,
        "device_pref": device,
        "out_dir": RP.track_rel(RP.BW0_DIR),
    }


def run(device: str = "cuda", *, smoke: bool = False, epochs: int | None = None,
        resume: bool = False) -> dict[str, Any]:
    import torch

    import se_train as TR

    identity = install()
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cfg = {"family": "B", "recipe": "W0", "dataset": "lipo", "budget": "N100"}
    out_dir = Path(RP.BW0_DIR) / ("smoke" if smoke else "")
    RP.write_json(out_dir / "identity.json", {**identity, "smoke": bool(smoke)})
    payload = TR.train_run(cfg, device=dev, out_dir=out_dir, epochs_override=epochs,
                           smoke=smoke, resume=resume)
    summary = payload.get("summary", {})
    best = summary.get("best_valid") or {}
    grid = summary.get("grid_best") or {}
    return {
        "status": payload.get("status"),
        "out_dir": RP.track_rel(out_dir),
        "best_epoch": best.get("epoch"),
        "best_valid_rmse": (best.get("valid") or {}).get("rmse"),
        "best_test_rmse": (best.get("test") or {}).get("rmse"),
        "grid_valid_rmse": grid.get("valid_rmse"),
        "grid_test_rmse": grid.get("test_rmse"),
        "parameters": (payload.get("model") or {}).get("parameters"),
        "init_state_hash": payload.get("model", {}).get("init_state_hash"),
        "gpu_hours": (payload.get("timing") or {}).get("gpu_hours_charged"),
        "identity": identity,
    }


__all__ = [name for name in dir() if not name.startswith("_")]
