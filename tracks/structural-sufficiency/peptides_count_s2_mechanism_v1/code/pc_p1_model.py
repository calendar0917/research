"""P1 model construction: frozen LCRC-C1 plus the explicit 2048 -> 10 count branch."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import pc_common as C
import bp_model as BM  # noqa: E402

ARM_SPECS: dict[str, dict[str, Any]] = {
    "A0": {"counts": None, "epochs": 240, "label": "BASE: original LCRC-C1"},
    "A1": {"counts": "type", "epochs": 240, "label": "TYPE: C1 + Linear(2048,10)(log1p(TYPE-S2))"},
    "A2": {"counts": "full", "epochs": 240, "label": "FULL: C1 + Linear(2048,10)(log1p(FULL-S2))"},
    "A3": {"counts": "null", "epochs": 240, "label": "NULL: C1 + Linear(2048,10)(log1p(NULL-S2))"},
    "P1A": {"counts": None, "epochs": 480, "label": "P1a: original LCRC-C1, 480-epoch cosine horizon"},
}
EXPECTED_BASE_PARAMETERS = 464739
EXPECTED_BRANCH_PARAMETERS = 2048 * 10


class CountBranchC1(nn.Module):
    """C1 logits plus a bias-free zero-initialised linear count readout."""

    def __init__(self, base: nn.Module, in_dim: int = C.N_BUCKETS, n_out: int = C.N_LABELS):
        super().__init__()
        self.base = base
        self.count_linear = nn.Linear(int(in_dim), int(n_out), bias=False)
        nn.init.zeros_(self.count_linear.weight)

    def forward(self, batch: dict[str, torch.Tensor], counts: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.base(batch)
        if counts is not None:
            logits = logits + self.count_linear(torch.log1p(counts))
        return logits


def norm_payload(task: dict) -> dict[str, Any]:
    return {
        **task["norm"],
        "continuous_atom": {
            "dim": int(task["cont_norm"].get("dim", 0)),
            "mean": task["cont_norm"].get("mean", []),
            "scale": task["cont_norm"].get("scale", []),
            "fingerprint": task["cont_norm"].get("fingerprint", ""),
        },
    }


def historical_init_hash(seed: int) -> str | None:
    """Init hash recorded by the frozen seed-matched historical checkpoint."""
    paths = [
        C.REPO
        / "tracks/structural-sufficiency/lcrc_peptides_diagnostic_v1/runs/peptide_c1_seed0_240ep/checkpoint.pt",
        C.REPO
        / f"tracks/structural-sufficiency/lcrc_peptides_diagnostic_v1/history/peptide_seed{int(seed)}/checkpoint.pt",
    ]
    for path in paths:
        if not path.is_file():
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        info = ckpt.get("model_info", {})
        if int(info.get("seed", -1)) == int(seed) and info.get("init_state_hash"):
            return str(info["init_state_hash"])
    return None


def build_arm(task: dict, arm: str, seed: int, device: torch.device | str) -> tuple[nn.Module, dict[str, Any]]:
    if arm not in ARM_SPECS:
        raise KeyError(arm)
    spec = ARM_SPECS[arm]
    device = torch.device(device)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    base, base_info = BM.build_c1(
        task["spec"],
        norm_payload(task),
        int(task["n_out"]),
        int(seed),
        device,
        cont_slice=task["cont_slice"],
        cont_mean=np.asarray(task["cont_norm"].get("mean", []), dtype=np.float32),
        cont_scale=np.asarray(task["cont_norm"].get("scale", []), dtype=np.float32),
    )
    base_parameters = int(sum(p.numel() for p in base.parameters()))
    base_state_hash = BM.state_fingerprint(base.state_dict())
    if base_parameters != EXPECTED_BASE_PARAMETERS:
        raise RuntimeError(f"base parameters {base_parameters} != {EXPECTED_BASE_PARAMETERS}")
    hist_hash = historical_init_hash(int(seed))
    structural = {
        "arm": arm,
        "label": spec["label"],
        "counts_kind": spec["counts"],
        "epochs": int(spec["epochs"]),
        "seed": int(seed),
        "base_parameters": base_parameters,
        "expected_base_parameters": EXPECTED_BASE_PARAMETERS,
        "base_parameters_match": base_parameters == EXPECTED_BASE_PARAMETERS,
        "base_init_state_hash": base_info["init_state_hash"],
        "base_state_hash_after_load": base_state_hash,
        "historical_init_state_hash": hist_hash,
        "base_init_matches_history": bool(hist_hash is not None and base_info["init_state_hash"] == hist_hash),
        "feature_spec_hash": task["spec"].hash(),
        "norm_fingerprint": task["norm"]["norm_fingerprint"],
        "local_layout": base_info["local_layout"],
        "full_init_state_hash": base_info["full_init_state_hash"],
        "shared_with_full": base_info["shared_with_full"],
    }
    if not structural["base_init_matches_history"]:
        raise RuntimeError(
            "base initialisation does not match the frozen historical seed-matched checkpoint: "
            f"{base_info['init_state_hash']} vs {hist_hash}"
        )
    if spec["counts"] is None:
        model = base
        info = {
            **structural,
            "parameters": base_parameters,
            "branch_parameters": 0,
            "total_parameters": base_parameters,
            "expected_total_parameters": base_parameters,
            "branch_init_zero": True,
            "branch_init_weight_sha256": None,
            "base_info": base_info,
        }
    else:
        model = CountBranchC1(base).to(device)
        branch_parameters = int(sum(p.numel() for p in model.count_linear.parameters()))
        if branch_parameters != EXPECTED_BRANCH_PARAMETERS:
            raise RuntimeError(f"branch parameters {branch_parameters} != {EXPECTED_BRANCH_PARAMETERS}")
        weight = model.count_linear.weight.detach().cpu().numpy()
        zero = bool(np.all(weight == 0.0))
        if not zero:
            raise RuntimeError("count branch was not zero-initialised")
        info = {
            **structural,
            "parameters": base_parameters,
            "branch_parameters": branch_parameters,
            "total_parameters": base_parameters + branch_parameters,
            "expected_total_parameters": base_parameters + EXPECTED_BRANCH_PARAMETERS,
            "branch_init_zero": zero,
            "branch_init_weight_sha256": C.sha256_array(weight),
            "full_state_hash_at_init": BM.state_fingerprint(model.state_dict()),
            "base_info": base_info,
        }
    if info["total_parameters"] != info["expected_total_parameters"]:
        raise RuntimeError("total parameter count mismatch")
    return model, info


def load_counts(kind: str) -> dict[str, np.ndarray]:
    key = f"counts_{kind}"
    out: dict[str, np.ndarray] = {}
    for split in C.ALLOWED_SPLITS:
        with np.load(C.ANALYSIS / f"features_{split}.npz") as z:
            if key not in z:
                raise KeyError(f"{key} missing from features_{split}.npz")
            out[split] = np.asarray(z[key], dtype=np.float32)
            out[f"{split}_gids"] = np.asarray(z["gids"], dtype=np.int64)
            out[f"{split}_n_paths"] = np.asarray(z["n_paths"], dtype=np.int64)
    return out


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def predict_logits(
    model: nn.Module,
    cache: dict,
    rows: np.ndarray,
    counts: np.ndarray | None,
    device: torch.device,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    from bp_data import collate

    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = np.asarray(rows[start : start + batch_size], dtype=np.int64)
            batch = to_device(collate(cache, chunk.tolist()), device)
            cb = None
            if counts is not None:
                cb = torch.as_tensor(counts[chunk], dtype=torch.float32, device=device)
            if cb is None:
                logits = model(batch)
            else:
                logits = model(batch, cb)
            out.append(logits.detach().cpu().numpy().astype(np.float32))
    if not out:
        return np.zeros((0, C.N_LABELS), dtype=np.float32)
    logits_np = np.concatenate(out, axis=0)
    if logits_np.shape != (len(rows), C.N_LABELS):
        raise RuntimeError(f"logits shape {logits_np.shape} != {(len(rows), C.N_LABELS)}")
    return logits_np


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float64)


def macro_ap(y: np.ndarray, probs: np.ndarray) -> tuple[float, list[float]]:
    from sklearn.metrics import average_precision_score

    aps = [float(average_precision_score(y[:, j], probs[:, j])) for j in range(y.shape[1])]
    return float(np.mean(aps)), aps


__all__ = [
    "ARM_SPECS",
    "CountBranchC1",
    "EXPECTED_BASE_PARAMETERS",
    "EXPECTED_BRANCH_PARAMETERS",
    "build_arm",
    "historical_init_hash",
    "load_counts",
    "macro_ap",
    "norm_payload",
    "predict_logits",
    "sigmoid_np",
    "to_device",
]
