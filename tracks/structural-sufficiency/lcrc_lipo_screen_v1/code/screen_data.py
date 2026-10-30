"""Frozen data access, batching and label-blind prediction.

Everything here is structural: the caches are label-free, the collated batch
has no label/target field, and prediction never receives a target.  Target
standardisation is Fit-only and is written once by the protocol freeze step.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

import screen_common as C
from screen_common import prep_data


# ---------------------------------------------------------------------------
# frozen caches
# ---------------------------------------------------------------------------


def load_caches(spec=None) -> dict[str, dict[str, Any]]:
    """Load the Fit and Monitor structural caches (never Reserved)."""
    spec = spec or C.load_frozen_spec()
    out = {}
    for split in ("fit", "monitor"):
        out[split] = C.load_frozen_cache(split, spec)
    return out


def load_target_norm(*, verify: bool = True) -> dict[str, Any]:
    if not C.NORM_TARGET.exists():
        raise FileNotFoundError(
            f"missing {C.track_rel(C.NORM_TARGET)}; run `run_screen.py freeze-protocol` first"
        )
    norm = C.read_json(C.NORM_TARGET)
    if norm.get("source") != "fit_labels_only" or norm.get("monitor_or_reserved_used") is not False:
        raise PermissionError("target normalisation is not certified Fit-only")
    if int(norm.get("n", -1)) != C.SPLIT_SIZES["fit"]:
        raise PermissionError("target normalisation does not cover exactly Fit2352")
    if verify:
        import fit_labels

        fit = fit_labels.load_fit_labels()
        stats = fit_labels.fit_label_stats(fit["y"])
        for key in ("mean", "scale", "fit_label_vector_sha256"):
            if norm[key] != stats[key]:
                raise PermissionError(f"target normalisation {key} does not match the Fit label vector")
    return norm


def save_target_norm() -> dict[str, Any]:
    import fit_labels

    fit = fit_labels.load_fit_labels()
    stats = fit_labels.fit_label_stats(fit["y"])
    stats["label_artifact_sha256"] = fit["labels_sha256"]
    stats["label_audit"] = fit["audit"]
    C.write_json(C.NORM_TARGET, stats)
    return stats


# ---------------------------------------------------------------------------
# batch order
# ---------------------------------------------------------------------------


def shuffle_batches(n: int, batch_size: int, generator: torch.Generator) -> list[list[int]]:
    perm = torch.randperm(int(n), generator=generator).tolist()
    return [perm[i : i + int(batch_size)] for i in range(0, len(perm), int(batch_size))]


def batch_order_hash(seed: int, n: int = C.SPLIT_SIZES["fit"], epochs: int = C.EPOCHS) -> str:
    generator = torch.Generator().manual_seed(int(seed) + C.TRAIN_SHUFFLE_OFFSET)
    h = hashlib.sha256()
    for _ in range(int(epochs)):
        for idx in shuffle_batches(n, C.BATCH_SIZE, generator):
            h.update(np.ascontiguousarray(np.asarray(idx, dtype=np.int64)).tobytes())
    return h.hexdigest()


def epoch_batches(seed: int, epoch: int) -> list[list[int]]:
    """Deterministic batch rows for one epoch without replaying earlier epochs."""
    if epoch < 1:
        raise ValueError("epoch must be >= 1")
    generator = torch.Generator().manual_seed(int(seed) + C.TRAIN_SHUFFLE_OFFSET)
    batches: list[list[int]] = []
    for _ in range(int(epoch)):
        batches = shuffle_batches(C.SPLIT_SIZES["fit"], C.BATCH_SIZE, generator)
    return batches


# ---------------------------------------------------------------------------
# target assembly
# ---------------------------------------------------------------------------


def normalised_target(y_fit: np.ndarray, rows: Sequence[int], norm: Mapping[str, Any]) -> torch.Tensor:
    y = np.asarray(y_fit, dtype=np.float64)[np.asarray(rows, dtype=np.int64)]
    z = (y - float(norm["mean"])) / float(norm["scale"])
    return torch.as_tensor(z, dtype=torch.float32)


def denormalise(pred: np.ndarray, norm: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(pred, dtype=np.float64) * float(norm["scale"]) + float(norm["mean"])


def raw_contribution_terms(
    contrib: np.ndarray, rho: np.ndarray, norm: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Map the standardised-space A decomposition to raw units.

    The constant Fit mean is absorbed into ``rho`` so that
    ``contrib_raw + rho_raw == pred_raw`` exactly (up to float rounding).
    """
    scale = float(norm["scale"])
    mean = float(norm["mean"])
    return (
        np.asarray(contrib, dtype=np.float64) * scale,
        np.asarray(rho, dtype=np.float64) * scale + mean,
    )


# ---------------------------------------------------------------------------
# collation and prediction
# ---------------------------------------------------------------------------


def collate(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    return prep_data.collate(cache, rows)


def to_device(batch: Mapping[str, Any], device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def monitor_rows(caches: Mapping[str, Mapping[str, Any]]) -> list[int]:
    return [int(r) for r in caches["monitor"]["rows"]]


def monitor_rows_sha256(caches: Mapping[str, Mapping[str, Any]]) -> str:
    rows = monitor_rows(caches)
    return hashlib.sha256(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()


@torch.no_grad()
def predict_cache(
    model: torch.nn.Module,
    cache: Mapping[str, Any],
    device,
    *,
    batch_size: int = 256,
    return_aux: bool = False,
) -> dict[str, Any]:
    """Label-blind prediction over a structural cache, in cache order."""
    was_training = model.training
    model.eval()
    n = int(cache["n_molecules"])
    preds: list[np.ndarray] = []
    contribs: list[np.ndarray] = []
    rhos: list[np.ndarray] = []
    for start in range(0, n, int(batch_size)):
        rows = list(range(start, min(start + int(batch_size), n)))
        batch = to_device(collate(cache, rows), device)
        if return_aux:
            y, aux = model(batch, return_aux=True)
            contribs.append(aux["contrib"].detach().cpu().numpy())
            rhos.append(aux["rho"].detach().cpu().numpy())
        else:
            y = model(batch)
        preds.append(y.detach().cpu().numpy().reshape(-1))
    out = {"pred": np.concatenate(preds).astype(np.float64), "rows": list(range(n))}
    if return_aux:
        out["contrib"] = np.concatenate(contribs).astype(np.float64)
        out["rho"] = np.concatenate(rhos).astype(np.float64)
    if was_training:
        model.train()
    return out


def atom_counts(cache: Mapping[str, Any], rows: Iterable[int] | None = None) -> np.ndarray:
    atom_ptr = np.asarray(cache["atom_ptr"])
    if rows is None:
        return np.diff(atom_ptr).astype(np.int64)
    rows = np.asarray(list(rows), dtype=np.int64)
    return (atom_ptr[rows + 1] - atom_ptr[rows]).astype(np.int64)


__all__ = [
    "load_caches",
    "load_target_norm",
    "save_target_norm",
    "shuffle_batches",
    "batch_order_hash",
    "epoch_batches",
    "normalised_target",
    "denormalise",
    "raw_contribution_terms",
    "collate",
    "to_device",
    "monitor_rows",
    "monitor_rows_sha256",
    "predict_cache",
    "atom_counts",
]
