"""Is the frozen training loop's initial state reproducible at all?

``se_train.train_run`` builds the C1 base with the process-global RNG and only
calls ``torch.manual_seed`` *after* the build, so the initial weights depend on
whatever ``torch.initial_seed()`` happened to be.  This module measures that
directly in fresh subprocesses instead of asserting it: it rebuilds the ``B``
model exactly the way ``train_run`` does (``setup_determinism`` -> view ->
permutations -> ``build_model``) and hashes the state, then compares against the
``init_state_hash`` recorded inside the pulled old checkpoints.

Output: ``analysis/init_reproducibility.json``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import rp_common as RP

PROBE = r'''
import sys
sys.path[:0] = {paths!r}
import torch
seed_before = int(torch.initial_seed())
import lp_common as LPC, lp_data as LPD, se_train as TR, se_tables as T
LPC.setup_determinism()
view = LPD.build_view()
perms = T.build_permutations()
model = TR.build_model({{"family": "B", "recipe": "W0", "dataset": "lipo", "budget": "N100"}}, view, perms)
print("@@" + __import__("json").dumps({{"initial_seed": seed_before,
      "state_hash": LPD.state_hash(model.state_dict()),
      "params": int(sum(p.numel() for p in model.parameters()))}}))
'''


def _paths() -> list[str]:
    import rp_common as _rp

    return [str(p) for p in _rp.module_paths()]


def probe() -> dict[str, Any]:
    script = PROBE.format(paths=_paths())
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    line = next((ln for ln in out.stdout.splitlines() if ln.startswith("@@")), None)
    if line is None:
        return {"error": (out.stderr or out.stdout)[-1500:]}
    return json.loads(line[2:])


def recorded_init_hashes() -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    for name, arm in (("B_Rwd", "B_Rwd"), ("LS_Rwd", "LS_Rwd"), ("FS_Rwd", "FS_Rwd")):
        p = RP.CKPT_DIR / arm / "checkpoint_best.pt"
        if not p.is_file():
            out[name] = {"present": False}
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        out[name] = {"present": True, "init_state_hash": ck.get("init_state_hash"),
                     "state_hash": ck.get("state_hash"), "best_epoch": ck.get("best_epoch")}
    return out


def bw0_init_identity() -> dict[str, Any]:
    """Does ``B_N100_lipo_W0`` share the HOME round ``B`` arm's initial state?"""
    import torch

    import lp_data as LPD
    import lp_model as LPM

    out: dict[str, Any] = {}
    view = LPD.build_view()
    c1_state = LPM.shared_c1_init_state(view.spec, view.train_norm)
    out["shared_c1_state_hash"] = LPD.state_hash(c1_state)
    home = RP.HOME / "analysis/results/B/metrics.json"
    out["home_B_recorded_c1_init_state_hash"] = (
        (RP.read_json(home).get("init") or {}).get("c1_init_state_hash") if home.is_file() else None
    )
    out["matches_home_B"] = out["home_B_recorded_c1_init_state_hash"] == out["shared_c1_state_hash"]
    ck_path = RP.BW0_DIR / "checkpoint_best.pt"
    if ck_path.is_file():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model = LPM.build_arm(view.spec, view.train_norm, "B")
        model.base.load_state_dict(c1_state, strict=True)
        out["bw0_recorded_init_state_hash"] = ck.get("init_state_hash")
        out["bw0_reproduced_init_state_hash"] = LPD.state_hash(model.state_dict())
        out["bw0_init_reproduced"] = out["bw0_recorded_init_state_hash"] == out["bw0_reproduced_init_state_hash"]
        out["bw0_parameters"] = ck.get("parameters")
        out["bw0_best_epoch"] = ck.get("best_epoch")
    else:
        out["bw0_recorded_init_state_hash"] = None
    out["note"] = (
        "the two hashes use different state_dict conventions (unprefixed C1 state vs the wrapper's "
        "prefixed state), so equality is established by rebuilding the state, not by comparing strings"
    )
    return out


def run(n_probes: int = 3) -> dict[str, Any]:
    probes = [probe() for _ in range(n_probes)]
    hashes = [p.get("state_hash") for p in probes if "state_hash" in p]
    seeds = [p.get("initial_seed") for p in probes if "initial_seed" in p]
    recorded = recorded_init_hashes()
    reproducible = len(set(hashes)) == 1
    payload = {
        "schema": "lcrc-semantic-analysis-repair-init-reproducibility-v1",
        "protocol_id": RP.PROTOCOL_ID,
        "created_utc": RP.utcnow(),
        "git_commit": RP.git_commit(),
        "question": "does the frozen training loop start from a reproducible initial state?",
        "method": (
            "fresh subprocesses run setup_determinism -> build_view -> build_permutations -> "
            "se_train.build_model(B), exactly the order train_run uses before torch.manual_seed"
        ),
        "probes": probes,
        "distinct_initial_seeds": len(set(seeds)),
        "distinct_state_hashes": len(set(hashes)),
        "reproducible_across_processes": bool(reproducible),
        "recorded_init_hashes": recorded,
        "matches_any_recorded": {
            name: (rec.get("init_state_hash") in set(hashes)) for name, rec in recorded.items()
            if rec.get("present")
        },
        "conclusion": (
            "NOT_REPRODUCIBLE: se_train seeds the global RNG only after building the model, so the old "
            "arms' initial weights depend on the OS entropy of each process.  The 'same non-table "
            "initial state' pairing across B/LS/FS/LP/FP is therefore not established by construction."
            if not reproducible else
            "REPRODUCIBLE: the initial state is identical across fresh processes."
        ),
        "b_w0_initial_state": bw0_init_identity(),
        "this_round_action": (
            "B_N100_lipo_W0 loads the frozen, documented C1 initial state "
            "dissect_model.build_full_state(spec,norm,0) -> remap_state('C1',...) (i.e. "
            "lp_model.shared_c1_init_state) before training, the same init the HOME round's original B arm used."
        ),
    }
    RP.write_json(RP.ANALYSIS / "init_reproducibility.json", payload)
    return {
        "reproducible_across_processes": payload["reproducible_across_processes"],
        "distinct_state_hashes": payload["distinct_state_hashes"],
        "matches_any_recorded": payload["matches_any_recorded"],
        "b_w0_matches_home_B": payload["b_w0_initial_state"].get("matches_home_B"),
        "artifacts": [RP.track_rel(RP.ANALYSIS / "init_reproducibility.json")],
    }


__all__ = [name for name in dir() if not name.startswith("_")]
