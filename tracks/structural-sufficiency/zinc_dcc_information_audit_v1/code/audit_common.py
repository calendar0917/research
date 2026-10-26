"""Shared plumbing for the ZINC DCC-L information audit v1.

Reads only the official-train structural cache and the frozen P3 objects.
The official valid/test caches are never opened; a torch.load guard records
every path and refuses anything that looks like a valid/test cache or label
file outside the explicit train/audit split files.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_TRACK = _HERE.parent
_SUFF = _TRACK.parent
_REPO = _SUFF.parents[1]
_V1 = _SUFF / "zinc_arch_discovery_v1" / "code"
_V2 = _SUFF / "zinc_recipe_trunk_v2" / "code"
_DCC = _SUFF / "zinc_dual_channel_composer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_V2), str(_DCC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import ad_common as AC  # noqa: E402
import ad_data as AD  # noqa: E402
import ad_train as AT  # noqa: E402
import dcc_models as DM  # noqa: E402

TRACK = _TRACK
ANALYSIS = TRACK / "analysis"
CONFIGS = TRACK / "configs"
TRAIN_CACHE = _SUFF / "zinc_arch_discovery_v1" / "cache" / "ad_train.pt"
VALID_CACHE = _SUFF / "zinc_arch_discovery_v1" / "cache" / "ad_valid.pt"
TEST_CACHE = _SUFF / "zinc_arch_discovery_v1" / "cache" / "ad_test.pt"
NORM_PATH = _SUFF / "zinc_arch_discovery_v1" / "analysis" / "norm.json"
P3_OBJECTS = _REPO / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/objects_train10k.pt"
P3_PREP = _REPO / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/prep_train10k.json"
P3_SSL_S0 = _REPO / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s0.pt"
P3_SSL_S1 = _REPO / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s1.pt"

CHECKPOINTS = {
    "D0": {
        "seed": 0,
        "path": _SUFF / "zinc_dual_channel_composer_v1" / "runs" / "L_s0" / "best.pt",
        "source": "local historical DCC-L L_s0 best-valid checkpoint",
        "recorded_sha256": "fab04ec0d5fd56f338379d7dcb223e31208338a8a89a335a25a6f0c4f5226cd5",
        "recorded_epoch": 489,
        "recorded_valid_mae": 0.07853714964160463,
    },
    "D1": {
        "seed": 1,
        "path": TRACK / "analysis" / "weights" / "B_R0_s1_best.pt",
        "source": "res:zinc_performance_search_v1/runs/B_R0_s1/best.pt",
        "recorded_sha256": "c444f371fe5b5ec5eeb33f984784016b9b87c8af4648e6e733727eb47d34ef1d",
        "recorded_epoch": 424,
        "recorded_valid_mae": 0.07383690453314921,
    },
    "D2": {
        "seed": 2,
        "path": TRACK / "analysis" / "weights" / "B_R0_s2_best.pt",
        "source": "res:zinc_performance_search_v1/runs/B_R0_s2/best.pt",
        "recorded_sha256": "a54897620bcc33163949ce23719cadeab1be4e854bde45e9f5d3d0f08a803019",
        "recorded_epoch": 392,
        "recorded_valid_mae": 0.074677,
    },
}

# ---------------------------------------------------------------------------
# provenance / access guard
# ---------------------------------------------------------------------------

_LOADED: list[dict[str, Any]] = []
_FORBIDDEN_FRAGMENTS = ("ad_valid.pt", "ad_test.pt", "/valid/", "/test/")


def _looks_forbidden(path: str) -> bool:
    p = str(path).replace("\\", "/")
    if p.endswith("ad_valid.pt") or p.endswith("ad_test.pt"):
        return True
    if "/results/luyin16/" in p and "valid" in p.lower():
        return True
    return False


def install_load_guard() -> None:
    """Wrap torch.load to record paths and refuse official valid/test caches."""
    if getattr(torch.load, "_audit_guard", False):
        return
    original = torch.load

    def guarded(f, *args, **kwargs):
        path = str(f)
        if _looks_forbidden(path):
            raise RuntimeError(f"forbidden valid/test cache access: {path}")
        obj = original(f, *args, **kwargs)
        rec = {"path": path, "kind": "torch.load", "ok": True}
        if isinstance(obj, dict):
            rec["keys"] = sorted(str(k) for k in obj.keys())[:32]
            rec["has_labels"] = "labels" in obj
        _LOADED.append(rec)
        return obj

    guarded._audit_guard = True
    torch.load = guarded


def loaded_paths() -> list[dict[str, Any]]:
    return list(_LOADED)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default) + "\n",
                    encoding="utf-8")


def _json_default(obj: Any):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_audit_config() -> dict:
    return read_json(CONFIGS / "audit_v1.json")


# ---------------------------------------------------------------------------
# data / model
# ---------------------------------------------------------------------------

def load_train_cache() -> dict:
    install_load_guard()
    if not TRAIN_CACHE.exists():
        raise FileNotFoundError(TRAIN_CACHE)
    cache = torch.load(TRAIN_CACHE, map_location="cpu", weights_only=False)
    if cache.get("split") != "train" or int(cache["n_molecules"]) != 10000:
        raise RuntimeError("train cache contract changed")
    return cache


def load_norm() -> dict:
    d = read_json(NORM_PATH)
    for k in ("global_mean", "global_scale", "root_scalar_mean", "root_scalar_scale",
              "block_scalar_mean", "block_scalar_scale"):
        d[k] = np.asarray(d[k], dtype=np.float32)
    return d


def build_frozen_model(norm: Mapping[str, Any], device="cpu", *, organized_local=True) -> DM.DualChannel:
    model = DM.DualChannel(norm, dropout=0.0)
    model.organized_local = bool(organized_local)
    model.to(device)
    model.eval()
    return model


def load_checkpoint(name: str, norm: Mapping[str, Any], device="cpu", *,
                    organized_local=True) -> tuple[DM.DualChannel, dict]:
    meta = CHECKPOINTS[name]
    path = Path(meta["path"])
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint {path}")
    actual = sha256_file(path)
    if actual != meta["recorded_sha256"]:
        raise RuntimeError(f"{name}: checkpoint sha256 {actual} != recorded {meta['recorded_sha256']}")
    install_load_guard()
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj["state_dict"] if "state_dict" in obj else obj
    model = build_frozen_model(norm, device=device, organized_local=organized_local)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{name}: state keys mismatch missing={missing} unexpected={unexpected}")
    if AC.parameter_count(model) != 494193:
        raise RuntimeError(f"{name}: parameter count {AC.parameter_count(model)} != 494193")
    info = dict(meta)
    info.update({
        "loaded_sha256": actual,
        "epoch": int(obj.get("epoch", -1)),
        "stored_valid_mae": float(obj.get("valid_mae", float("nan"))),
        "candidate": obj.get("candidate"),
        "seed_in_checkpoint": int(obj.get("seed", -1)),
        "parameters": int(AC.parameter_count(model)),
        "state_key_count": len(state),
        "state_hash": AC.state_hash(model),
    })
    return model, info


def collate_rows(cache: Mapping[str, Any], rows: Sequence[int], *, device="cpu") -> dict:
    batch = AT.collate(cache, np.asarray(rows, dtype=np.int64), "P3N", valid=False)
    return AT._to_device(batch, torch.device(device))


@torch.no_grad()
def capture_forward(model: DM.DualChannel, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replicate DualChannel.forward while returning every intermediate tensor.

    This is a wrapper, not a reimplementation: it calls the model's own encoder,
    grouping, composer and reader modules in the same order.
    """
    a, b, w = model.encoders(batch)
    Wg, Vg, n_b = model.group_to_bonds(w, batch)
    c_glob = model.group_composer(
        torch.cat([b, Wg, Vg, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
    )
    Wl, Vl, _ = model.local_grouping(w, batch, organized=bool(model.organized_local))
    c_loc = model.local_composer(
        torch.cat([b, Wl, Vl, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
    )
    B = int(batch["n_graphs"])
    z = c_glob.new_zeros((B, c_glob.shape[1])).index_add_(0, batch["bond_batch"], c_glob)
    zl = c_loc.new_zeros((B, c_loc.shape[1])).index_add_(0, batch["bond_batch"], c_loc)
    zl2 = c_loc.new_zeros((B, c_loc.shape[1])).index_add_(0, batch["bond_batch"], c_loc * c_loc)
    a_sum = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a)
    a_sq = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a * a)
    g = (batch["global"] - model.global_mean) / model.global_scale
    z_full = torch.cat([z, a_sum, a_sq, g, zl, zl2], dim=1)
    pred = model.reader(z_full).view(-1)
    return {
        "a": a, "b": b, "w": w,
        "Wg": Wg, "Vg": Vg, "Wl": Wl, "Vl": Vl, "n_b": n_b,
        "c_glob": c_glob, "c_loc": c_loc,
        "z_glob": z, "zl": zl, "zl2": zl2,
        "a_sum": a_sum, "a_sq": a_sq, "g": g,
        "z": z_full, "pred": pred,
    }


@torch.no_grad()
def predict_rows(model: DM.DualChannel, cache: Mapping[str, Any], rows: Sequence[int],
                 *, batch_size: int = 128, device="cpu") -> dict[str, np.ndarray]:
    """Run the frozen model over rows and return per-molecule tensors as numpy."""
    rows = np.asarray(rows, dtype=np.int64)
    out: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(rows), batch_size):
        part = rows[start:start + batch_size]
        batch = collate_rows(cache, part, device=device)
        cap = capture_forward(model, batch)
        for k, v in cap.items():
            out.setdefault(k, []).append(v.detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def predict_smoke(model: DM.DualChannel, cache: Mapping[str, Any], rows: Sequence[int],
                  *, device="cpu") -> tuple[np.ndarray, np.ndarray]:
    batch = collate_rows(cache, rows, device=device)
    cap = capture_forward(model, batch)
    direct = model(batch)
    return cap["pred"].detach().cpu().numpy(), direct.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def participation_ratio(cov: np.ndarray) -> float:
    cov = np.asarray(cov, dtype=np.float64)
    ev = np.linalg.eigvalsh((cov + cov.T) / 2.0)
    ev = np.clip(ev, 0.0, None)
    s = float(ev.sum())
    if s <= 0:
        return 0.0
    return float(s * s / float(np.sum(ev * ev)))


def spectrum_summary(x: np.ndarray) -> dict:
    """Covariance spectrum of rows of x (already centred if desired)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    n = max(1, x.shape[0] - 1)
    cov = (x.T @ x) / n
    ev = np.clip(np.linalg.eigvalsh((cov + cov.T) / 2.0)[::-1], 0.0, None)
    total = float(ev.sum())
    if total <= 0:
        return {"n": int(x.shape[0]), "dim": int(x.shape[1]), "participation_ratio": 0.0,
                "top1": 0.0, "top5": 0.0, "top10": 0.0, "dims90": 0, "dims95": 0,
                "trace": 0.0, "zero_eigenvalues": int(len(ev))}
    cum = np.cumsum(ev) / total
    return {
        "n": int(x.shape[0]), "dim": int(x.shape[1]),
        "participation_ratio": float(total * total / float(np.sum(ev * ev))),
        "top1": float(ev[0] / total),
        "top5": float(ev[:5].sum() / total),
        "top10": float(ev[:10].sum() / total),
        "dims90": int(np.searchsorted(cum, 0.90) + 1),
        "dims95": int(np.searchsorted(cum, 0.95) + 1),
        "trace": float(total),
        "zero_eigenvalues": int(np.sum(ev <= 1e-12 * max(total, 1.0))),
        "eigenvalues_top20": ev[:20].tolist(),
    }


def pooled_within_spectrum(states: np.ndarray, bond_ptr: np.ndarray) -> dict:
    """Centre each molecule's bond states, concatenate, and return the spectrum."""
    states = np.asarray(states, dtype=np.float64)
    bp = np.asarray(bond_ptr, dtype=np.int64)
    pieces = []
    for i in range(len(bp) - 1):
        h = states[bp[i]:bp[i + 1]]
        if h.shape[0] == 0:
            continue
        pieces.append(h - h.mean(axis=0, keepdims=True))
    if not pieces:
        return {"n": 0, "dim": int(states.shape[1]), "participation_ratio": 0.0}
    return spectrum_summary(np.concatenate(pieces, axis=0))


def constant_channel_fraction(x: np.ndarray, *, atol: float = 1e-6) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 1.0
    return float(np.mean(x.std(axis=0) < atol))


def within_molecule_bond_summary(states: np.ndarray, bond_ptr: np.ndarray,
                                 *, max_molecules: int | None = None) -> dict:
    """Per-molecule variation of a [n_bonds, d] state array."""
    states = np.asarray(states, dtype=np.float64)
    bp = np.asarray(bond_ptr, dtype=np.int64)
    n_mol = len(bp) - 1
    if max_molecules is not None:
        n_mol = min(n_mol, int(max_molecules))
    prs, cv, pair_rms, pair_norm, frac_small = [], [], [], [], []
    for i in range(n_mol):
        h = states[bp[i]:bp[i + 1]]
        n = h.shape[0]
        if n < 2:
            continue
        mu = h.mean(axis=0, keepdims=True)
        hc = h - mu
        cov = (hc.T @ hc) / max(n - 1, 1)
        prs.append(participation_ratio(cov))
        rms = float(np.sqrt(np.mean(np.sum(hc * hc, axis=1))))
        mean_norm = float(np.sqrt(np.mean(np.sum(h * h, axis=1))))
        cv.append(rms / max(mean_norm, 1e-12))
        # pairwise RMS
        d2 = np.sum((h[:, None, :] - h[None, :, :]) ** 2, axis=-1)
        iu = np.triu_indices(n, k=1)
        pair_rms.append(float(np.sqrt(np.mean(d2[iu]))))
        pair_norm.append(float(np.sqrt(np.mean(d2[iu]))) / max(mean_norm, 1e-12))
        frac_small.append(float(np.mean(np.sqrt(d2[iu]) < 1e-2 * max(mean_norm, 1e-12))))
    def q(a):
        if not a:
            return {"mean": None, "median": None, "p10": None, "p90": None, "n": 0}
        a = np.asarray(a, dtype=np.float64)
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
                "n": int(len(a))}
    return {
        "per_molecule_participation_ratio": q(prs),
        "per_molecule_cv": q(cv),
        "per_molecule_pairwise_rms": q(pair_rms),
        "per_molecule_pairwise_rms_normalized": q(pair_norm),
        "per_molecule_frac_pairs_below_1pct_mean_norm": q(frac_small),
        "constant_channel_fraction": constant_channel_fraction(states),
    }


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj
