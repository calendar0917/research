"""FSR v0 -- fixed-source role separation pilot.

Round: ``fixed-source-role-separation-pilot-v0``

Read-only inheritance (never modified here):

* ``direct_structural_computation_recurrence_v0/code/ds_model.py`` -- the
  FIXED1 implementation (``DSModel``), ``MessageFFN``, ``directed_bonds``,
  ``_agg_source`` and the official W2 constructor path;
* ``ds_common.py`` -- the frozen PCBA recipe, the frozen seed-0 40-epoch order
  plan, the official evaluator wrappers, the data identity and the
  checkpoint/state helpers;
* ``results/runs/FIXED1_seed0/{train_curve.csv,per_task_ap.csv}`` -- the real
  historical FIXED1 seed-0 curve used as the same-horizon reference.

This round changes exactly one thing: each of the two FIXED1 blocks receives an
independent low-rank branch ``A(B(x))`` added to the pre-nonlinearity linear
term, with ``x = z + m`` (arm ``TIED``) or ``x = z - m`` (arm ``RSSC``).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
ROUND = CODE_DIR.parent
TRACK = ROUND.parent
REPO = TRACK.parents[1]
SS = REPO / "tracks/structural-sufficiency"

SRC_ROUND = SS / "direct_structural_computation_recurrence_v0"
SRC_CODE = SRC_ROUND / "code"
for _p in (str(SRC_CODE), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ds_common as C  # noqa: E402  (frozen recipe / data / plan / evaluation)
import ds_model as DSM  # noqa: E402  (FIXED1 model)
import ds_checks as DCHK  # noqa: E402  (stand-in normaliser / feature spec loader)
import pc_metrics as MET  # noqa: E402

ANALYSIS = ROUND / "analysis"
CONFIGS = ROUND / "configs"
RUNS = ROUND / "runs"
RESULTS = ROUND / "results"

ROUND_ID = "fixed-source-role-separation-pilot-v0"
PROTOCOL_ID = "fsr-v0"
SCHEMA_RUN = "fsr-run-v1"
SCHEMA_CHECKPOINT = "fsr-checkpoint-v1"
SCHEMA_CHECKS = "fsr-checks-v1"
SCHEMA_PREFLIGHT = "fsr-preflight-v1"
SCHEMA_MECHANISM = "fsr-mechanism-v1"
SCHEMA_MANIFEST = "fsr-checkpoint-manifest-v1"

TEST_ROLE = "NOT_RUN / FORBIDDEN_THIS_ROUND"

SEED = 0
ARMS = ("TIED", "RSSC")
ARM_ORDER = ARMS
RANK = 32
WIDTH = 128
BLOCK_HIDDEN = 256
PREFERRED_HORIZON = 12
FALLBACK_HORIZON = 8
HORIZON_GRID = (12, 8)
TRAIN_EVAL_EPOCHS = (4, 8, 12)

PARAMS_FIXED1 = 1_128_468
PARAMS_W2 = 994_324
PARAMS_INHERITED_NEW = 134_144
PARAMS_PER_BLOCK_LOWRANK = WIDTH * RANK + RANK * BLOCK_HIDDEN  # 12,288
PARAMS_ADD = 2 * PARAMS_PER_BLOCK_LOWRANK  # 24,576
PARAMS_TOTAL = PARAMS_FIXED1 + PARAMS_ADD  # 1,153,044
EXPECTED = {
    "w2_base": 848_096,
    "w2_readout": 146_228,
    "w2_total": PARAMS_W2,
    "bond_proj": 1_792,
    "blocks_0_inherited": 66_176,
    "blocks_1_inherited": 66_176,
    "inherited_new": PARAMS_INHERITED_NEW,
    "lowrank_0": PARAMS_PER_BLOCK_LOWRANK,
    "lowrank_1": PARAMS_PER_BLOCK_LOWRANK,
    "lowrank_total": PARAMS_ADD,
    "total": PARAMS_TOTAL,
}

#: Historical initial-state fingerprints of the inherited FIXED1 model at seed 0
#: (recorded by the source round; read-only).
HIST_FIXED1_INIT = {
    "base_state_sha256": "5f7101a8f378e7c9283caaade8d61eee733c0c9f3cfb65b5584c8a1e495ef63d",
    "new_state_sha256": "f7c24ee48bc71c9c7d1d958bac6b249b583762c97b11ca07af21a62fe5441177",
    "initial_state_sha256": "3d313189bd322eb6c2b1c9075851b3b5f31b7c2e82318c5b4217319d4bd50e5c",
    "n_state_tensors": 71,
    "parameters": PARAMS_FIXED1,
    "n_epochs_available": 25,
}

HIST = {
    "source_round": "direct_structural_computation_recurrence_v0",
    "arm": "FIXED1",
    "seed": 0,
    "status": "COMPLETED",
    "epochs_available": 25,
    "run_json": SRC_ROUND / "results/runs/FIXED1_seed0/run.json",
    "curve": SRC_ROUND / "results/runs/FIXED1_seed0/train_curve.csv",
    "per_task": SRC_ROUND / "results/runs/FIXED1_seed0/per_task_ap.csv",
    "best_valid_ap_25": 0.27584552136653284,
    "w2_best_valid_ap_40": 0.265273,
}

#: Historical 25-epoch FIXED1 curve was produced on the same physical GPU model
#: (A100-SXM4-40GB) and the same torch build; the regime is recorded per run.
HIST_REGIME = {
    "host": "res (a100-2)",
    "backend": "process",
    "gpu": "NVIDIA A100-SXM4-40GB",
    "torch": "2.5.1+cu124",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def is_lowrank(name: str) -> bool:
    return str(name).startswith("lowrank.")


def is_inherited_new(name: str) -> bool:
    return C.is_new_parameter(str(name))


def is_frs_new(name: str) -> bool:
    return is_inherited_new(name) or is_lowrank(name)


def frs_trunk_subset(state: Mapping[str, Any]) -> dict[str, Any]:
    """The W2 trunk used for the fingerprint: excludes *every* added tensor."""
    return {k: v for k, v in state.items() if not is_frs_new(k)}


def frs_new_subset(state: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in state.items() if is_frs_new(k)}


def inherited_new_subset(state: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in state.items() if is_inherited_new(k)}


def lowrank_subset(state: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in state.items() if is_lowrank(k)}


def arm_out_dir(arm: str) -> Path:
    return RUNS / f"{arm}_seed{SEED}"


def utcnow() -> str:
    return C.utcnow()


def write_json(path: Path, payload: Any) -> None:
    C.write_json(Path(path), payload)


def read_json(path: Path) -> Any:
    return C.read_json(Path(path))


def sha256_file(path: Path) -> str:
    return C.sha256_file(Path(path))


def log(msg: str) -> None:
    C.log(msg)


def parameter_groups(model: Any) -> dict[str, int]:
    groups = {
        "w2_base": 0,
        "w2_readout": 0,
        "bond_proj": 0,
        "blocks_0_inherited": 0,
        "blocks_1_inherited": 0,
        "lowrank_0": 0,
        "lowrank_1": 0,
    }
    for name, p in model.named_parameters():
        n = int(p.numel())
        if name.startswith("bond_proj."):
            groups["bond_proj"] += n
        elif name.startswith("blocks.0."):
            groups["blocks_0_inherited"] += n
        elif name.startswith("blocks.1."):
            groups["blocks_1_inherited"] += n
        elif name.startswith("lowrank.0."):
            groups["lowrank_0"] += n
        elif name.startswith("lowrank.1."):
            groups["lowrank_1"] += n
        elif name.startswith("readout."):
            groups["w2_readout"] += n
        else:
            groups["w2_base"] += n
    groups["inherited_new"] = (
        groups["bond_proj"] + groups["blocks_0_inherited"] + groups["blocks_1_inherited"]
    )
    groups["lowrank_total"] = groups["lowrank_0"] + groups["lowrank_1"]
    groups["w2_total"] = groups["w2_base"] + groups["w2_readout"]
    groups["total"] = groups["w2_total"] + groups["inherited_new"] + groups["lowrank_total"]
    return groups


def parameter_audit(model: Any) -> dict[str, Any]:
    groups = parameter_groups(model)
    checks = {k: int(groups.get(k, -1)) == int(v) for k, v in EXPECTED.items()}
    return {
        "groups": groups,
        "expected": dict(EXPECTED),
        "checks": checks,
        "all_match": bool(all(checks.values())),
        "total": int(groups["total"]),
        "n_parameter_tensors": int(len(list(model.parameters()))),
        "n_state_tensors": int(len(model.state_dict())),
    }


def initial_fingerprints(model: Any) -> dict[str, Any]:
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return {
        "base_state_sha256": C.state_fingerprint(frs_trunk_subset(state)),
        "new_state_sha256": C.state_fingerprint(inherited_new_subset(state)),
        "lowrank_state_sha256": C.state_fingerprint(lowrank_subset(state)),
        "initial_state_sha256": C.state_fingerprint(state),
        "n_state_tensors": int(len(state)),
    }


# ---------------------------------------------------------------------------
# historical FIXED1 same-horizon reference
# ---------------------------------------------------------------------------


def load_hist_curve() -> list[dict[str, Any]]:
    path = Path(HIST["curve"])
    if not path.is_file():
        raise FileNotFoundError(f"historical FIXED1 curve missing: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    out = []
    for r in rows:
        out.append(
            {
                "epoch": int(r["epoch"]),
                "valid_ap": float(r["valid_ap"]),
                "valid_bce": float(r["valid_bce"]),
                "train_ap": float(r["train_ap"]),
                "train_bce": float(r["train_bce"]),
                "learning_rate": float(r["learning_rate"]),
                "epoch_seconds": float(r["epoch_seconds"]),
            }
        )
    return out


def hist_prefix(horizon: int) -> dict[str, Any]:
    rows = [r for r in load_hist_curve() if r["epoch"] <= int(horizon)]
    if len(rows) < int(horizon):
        raise RuntimeError(f"historical FIXED1 curve has {len(rows)} < {horizon} epochs")
    best = max(rows, key=lambda r: (r["valid_ap"], -r["epoch"]))
    last3 = rows[-3:]
    return {
        "arm": "FIXED1",
        "source": "historical_seed0_run (direct_structural_computation_recurrence_v0)",
        "horizon": int(horizon),
        "epochs_available": len(rows),
        "best_epoch": int(best["epoch"]),
        "best_valid_ap": float(best["valid_ap"]),
        "endpoint_epoch": int(rows[-1]["epoch"]),
        "endpoint_valid_ap": float(rows[-1]["valid_ap"]),
        "endpoint_valid_bce": float(rows[-1]["valid_bce"]),
        "last3_mean_valid_ap": float(np.mean([r["valid_ap"] for r in last3])),
        "last3_epochs": [int(r["epoch"]) for r in last3],
        "curve": rows,
    }


def hist_per_task(horizon: int, epoch: int) -> dict[int, dict[str, Any]]:
    """Per-task official Valid AP of the historical FIXED1 run at one epoch."""
    path = Path(HIST["per_task"])
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "valid" and int(r["epoch"]) == int(epoch)]
    if not rows:
        raise RuntimeError(f"historical FIXED1 has no valid per-task row at epoch {epoch}")
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        out[int(r["task"])] = {
            "ap": float(r["ap"]),
            "observed": int(r["observed"]),
            "positive": int(r["positive"]),
            "negative": int(r["negative"]),
        }
    return out


def hist_run_identity() -> dict[str, Any]:
    run = read_json(Path(HIST["run_json"]))
    return {
        "status": run.get("status"),
        "horizon": run.get("horizon"),
        "best_valid_ap": run.get("best_valid_ap"),
        "best_epoch": run.get("best_epoch"),
        "git_commit": run.get("git_commit"),
        "environment": run.get("environment"),
        "host_facts": run.get("host_facts"),
        "base_state_sha256": (run.get("model_audit") or {}).get("base_state_sha256"),
        "new_state_sha256": (run.get("model_audit") or {}).get("new_state_sha256"),
        "initial_state_sha256": (run.get("model_audit") or {}).get("initial_state_sha256"),
        "parameters": run.get("parameters"),
    }


# ---------------------------------------------------------------------------
# data identity (inherited gate, recorded once)
# ---------------------------------------------------------------------------


def data_identity(task: Any, hist: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(C.SOURCE)
    keys = (
        "feature_spec_hash",
        "norm_fingerprint",
        "split_partition_sha256",
        "label_artifact_sha256",
        "cache_sha256",
    )
    got = {k: source.get(k) for k in keys}
    hist_run = hist.get("data_identity") or {}
    checks = {k: (got.get(k) == hist_run.get(k)) for k in keys if k in hist_run}
    counts = {
        "n_train_rows": int(len(task.train_rows)),
        "n_valid_rows": int(len(task.valid_rows)),
        "n_test_rows": int(len(task.test_rows)),
        "n_tasks": int(C.N_TASKS),
        "labels_shape": list(task.labels.shape),
        "test_labels_all_nan": bool(np.isnan(task.labels[task.test_rows]).all()),
    }
    return {
        "current": got,
        "historical": hist_run,
        "checks": checks,
        "all_match": bool(all(checks.values())) if checks else None,
        "counts": counts,
        "count_checks": {
            "n_train_rows": counts["n_train_rows"] == int(C.N_TRAIN),
            "n_valid_rows": counts["n_valid_rows"] == int(C.N_VALID),
            "n_test_rows": counts["n_test_rows"] == int(C.N_TEST),
            "test_labels_all_nan": counts["test_labels_all_nan"],
        },
        "note": (
            "Inherited loader re-verifies feature/norm/split/label/cache identities. "
            "The full cache audit is not repeated; the recorded historical identity is "
            "compared value by value."
        ),
    }
