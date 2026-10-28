"""Supervised integration for CJP-SSL v1.

The supervised loop is **not** re-implemented: this module patches
``lc_models.build_model`` for the duration of one call to the frozen
``lc_train.train`` and then restores it.  Therefore batch order, parameter
names, initialisation order, data normalisation, optimiser schedule, valid
selection and checkpoint format are bit-for-bit the frozen LCRC-A trainer.

The patch only changes *which initial state* ``LocalContent`` receives and
whether the historical atom/bond masked-SSL encoders are loaded:

============  ===========  ====================  =====================
arm           load_ssl     LocalContent init      purpose
============  ===========  ====================  =====================
A-REPLAY      yes          seed-initialised      synchronous baseline
SCRATCH       no           seed-initialised      value of old encoder SSL
MARG-SSL      yes          MARG pretrained       LocalContent transfer
JOINT-SSL     yes          JOINT pretrained      joint-vs-marginal test
============  ===========  ====================  =====================
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import cjp_common as C
import cjp_models as CM

L = C.L
M = C.M
T = C.T

MODES = {
    "replay": {"load_ssl": True, "local_state": None},
    "scratch": {"load_ssl": False, "local_state": None},
    "margssl": {"load_ssl": True, "local_state": "MARG"},
    "jointssl": {"load_ssl": True, "local_state": "JOINT"},
}


def run_name(mode: str, seed: int, tag: str | None = None) -> str:
    prefix = f"{tag}_" if tag else ""
    return f"A_{prefix}{mode}_s{int(seed)}"


def _load_local_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    return state, {
        "path": C.track_rel(path),
        "sha256_file": C.sha256_file(path),
        "arm": payload.get("arm"),
        "seed": payload.get("seed"),
        "updates": payload.get("updates"),
        "state_sha256": C.sha256_bytes(
            b"".join(
                np.ascontiguousarray(state[k].cpu().numpy(), dtype=np.float32).tobytes()
                for k in sorted(state)
            )
        ),
    }


def run(
    mode: str,
    seed: int,
    device: str,
    *,
    out_dir: Path | None = None,
    epochs: int | None = None,
    tag: str | None = None,
    pretrain_dir: Path | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    L.require_cuda(device)
    cfg = MODES[mode]
    out = Path(out_dir) if out_dir is not None else C.RUNS / run_name(mode, seed, tag)

    local_state = None
    local_info: dict[str, Any] = {"replaced": False}
    if cfg["local_state"] is not None:
        if pretrain_dir is None:
            pretrain_dir = C.RUNS / (
                f"pretrain_{str(cfg['local_state']).lower()}_s{int(seed)}"
            )
        state_path = Path(pretrain_dir) / "local_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"missing pretrained LocalContent state {state_path}")
        local_state, local_info = _load_local_state(state_path)

    captured: dict[str, Any] = {}
    original_build = M.build_model

    def patched_build(arm, norm, seed_arg, device_arg, cfg_arg=None):
        model, info = CM.build_supervised_model(
            norm,
            seed_arg,
            device_arg,
            cfg_arg,
            load_ssl=bool(cfg["load_ssl"]),
            local_state=local_state,
        )
        captured["info"] = info
        return model, info

    M.build_model = patched_build
    try:
        payload = T.train("A", int(seed), device, epochs=epochs, force=True, out_dir=out)
    finally:
        M.build_model = original_build

    info = captured.get("info", {})
    provenance = {
        "round": C.ROUND_ID,
        "mode": mode,
        "run_name": out.name,
        "seed": int(seed),
        "device": str(device),
        "load_ssl": bool(cfg["load_ssl"]),
        "local_state_source": local_info,
        "model_info": info,
        "deployed_parameters": int(info.get("parameters", -1)),
        "supervised_recipe": dict(C.SUPERVISED_RECIPE),
        "trainer": "lc_train.train (frozen, imported unchanged)",
        "trainer_sha256": C.sha256_file(C.LC_CODE / "lc_train.py"),
        "checkpoint_excludes_pretext_head": True,
        "official_test_loaded": False,
    }
    C.write_json(out / "cjp_provenance.json", provenance)

    # augment the frozen payload (keys and values of the frozen fields untouched)
    payload_path = out / "train.json"
    stored = C.read_json(payload_path)
    stored["cjp"] = provenance
    C.write_json(payload_path, stored)
    L.log(f"[sup {mode} s{seed}] best={stored['best']['valid_mae']:.6f} "
          f"@{stored['best']['epoch']} params={info.get('parameters')}")
    return stored


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CJP-SSL supervised arms")
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--pretrain-dir", default=None)
    args = parser.parse_args(argv)
    res = run(
        args.mode,
        args.seed,
        args.device,
        out_dir=None if args.out is None else Path(args.out),
        epochs=args.epochs,
        tag=args.tag,
        pretrain_dir=None if args.pretrain_dir is None else Path(args.pretrain_dir),
    )
    L.log(f"best valid {res['best']['valid_mae']:.6f} @ {res['best']['epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
