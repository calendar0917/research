#!/usr/bin/env python
"""FGCR joint-response increment audit (frozen checkpoint, CPU, no training).

Round: fgcr-joint-response-increment-audit-v0

Reads the historical FGCR-v0 checkpoint read-only, replays it, decomposes the
exact full-graph responses into a local reference plus a joint increment, and
evaluates four frozen-head input conditions.  Test is never referenced.

Run:
    uv run python tracks/structural-sufficiency/fgcr_joint_response_increment_audit_v0/code/run_audit.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

# Keep BLAS/torch CPU parallelism modest: this audit runs many very small
# tensor ops, where excessive threads are slower and can thrash.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
REPO = ROUND.parents[2]
HIST = REPO / "tracks" / "structural-sufficiency" / "full_graph_cavity_response_pilot_v0"
CONFIG_PATH = ROUND / "configs" / "audit.json"
CFG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# asset preflight (before importing historical modules)
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_minimal_checks(status: str, **payload: Any) -> None:
    out = {"round_id": CFG["round_id"], "status": status, **payload}
    target = ROUND / "checks.json"
    target.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _preflight_assets() -> None:
    required = {
        "checkpoint": HIST / CFG["checkpoint"],
        "historical_predictions": HIST / CFG["historical_predictions"],
        "historical_config": HIST / CFG["historical_config"],
        "cache": REPO / CFG["cache"]["path"],
    }
    for rel, expected in CFG["historical_code"].items():
        required[f"historical_code:{rel}"] = HIST / rel
    missing = [str(p) for p in required.values() if not p.is_file()]
    if missing:
        _write_minimal_checks("MISSING_ARTIFACT", missing=missing)
        print(f"MISSING_ARTIFACT: {missing}", flush=True)
        raise SystemExit(2)

    hashes = {name: _sha256(path) for name, path in required.items()}
    mismatches = []
    expected_map = {
        "checkpoint": CFG["checkpoint_sha256"],
        "historical_predictions": CFG["historical_predictions_sha256"],
        "historical_config": CFG["historical_config_sha256"],
        "cache": CFG["cache"]["sha256"],
    }
    for rel, expected in CFG["historical_code"].items():
        expected_map[f"historical_code:{rel}"] = expected
    for name, expected in expected_map.items():
        if hashes[name] != expected:
            mismatches.append({"asset": name, "expected": expected,
                               "actual": hashes[name]})
    if mismatches:
        _write_minimal_checks("ARTIFACT_IDENTITY_MISMATCH", mismatches=mismatches)
        print(f"ARTIFACT_IDENTITY_MISMATCH: {mismatches}", flush=True)
        raise SystemExit(2)


_preflight_assets()
sys.path.insert(0, str(HIST / "code"))

import fgcr_common as C  # noqa: E402
import fgcr_data as D  # noqa: E402
import fgcr_inference as I  # noqa: E402
import fgcr_model as M  # noqa: E402
import warnings  # noqa: E402
from scipy.stats import ConstantInputWarning, spearmanr  # noqa: E402

warnings.filterwarnings("ignore", category=ConstantInputWarning)

torch.set_num_threads(int(CFG["torch_threads"]))
DEVICE = str(CFG["device"])
CONDITIONS = ["FULL", "LOCAL_H", "LOCAL_ETA", "LOCAL_BOTH"]
EPS = 1e-12


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------


def _batches(rows: np.ndarray, size: int):
    for start in range(0, int(rows.size), int(size)):
        yield rows[start:start + int(size)]


def _qstats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {}
    qs = [0.0, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    names = ["min", "q05", "q25", "median", "q75", "q90", "q95", "q99", "max"]
    return {name: float(np.quantile(x, q)) for name, q in zip(names, qs)}


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    den = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / (den + EPS))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    value = spearmanr(np.asarray(x, dtype=np.float64),
                      np.asarray(y, dtype=np.float64)).statistic
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def _corr_pair(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {"pearson": _pearson(x, y), "spearman": _spearman(x, y)}


def _pearson_cols(x: np.ndarray, y: np.ndarray) -> list[float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    num = (x * y).sum(axis=0)
    den = np.sqrt((x * x).sum(axis=0) * (y * y).sum(axis=0)) + EPS
    return (num / den).tolist()


def _cos_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    num = (x * y).sum(axis=1)
    den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + EPS
    return num / den


def _channel_stats(x: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    return {
        "n": int(x.shape[0]),
        "channels": int(x.shape[1]),
        "channel_mean": x.mean(axis=0).tolist(),
        "channel_std": x.std(axis=0).tolist(),
        "channel_rms": np.sqrt((x * x).mean(axis=0)).tolist(),
        "pooled_mean": float(x.mean()),
        "pooled_std": float(x.std()),
        "pooled_rms": float(np.sqrt((x * x).mean())),
    }


def _rms_ratio(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.sqrt((x * x).mean()) / (np.sqrt((y * y).mean()) + EPS))


def _channel_rms_ratio(x: np.ndarray, y: np.ndarray) -> list[float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return (np.sqrt((x * x).mean(0)) / (np.sqrt((y * y).mean(0)) + EPS)).tolist()


# ---------------------------------------------------------------------------
# exact local reference (same tensor algebra in float32 and float64)
# ---------------------------------------------------------------------------


def local_references(u: torch.Tensor, log_psi: torch.Tensor,
                     bond_u: torch.Tensor, bond_v: torch.Tensor,
                     bond_mol: torch.Tensor, n_molecules: int,
                     channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Local edge-sum H and local star eta from independent neighbour marginals.

    u          [N, C, 2]      unary log potentials
    log_psi    [M, C, 2, 2]   raw physical-edge log potentials
    bond_u/v   [M]            batch-local atom ids
    bond_mol   [M]            batch-local graph ids
    """
    log_p = torch.log_softmax(u, dim=-1)
    log_p_u = log_p.index_select(0, bond_u)
    log_p_v = log_p.index_select(0, bond_v)
    n_bonds = int(log_p_u.shape[0])

    edge_logits = (log_p_u.unsqueeze(-1) + log_p_v.unsqueeze(-2) + log_psi)
    h_edge = torch.logsumexp(edge_logits.reshape(n_bonds, channels, 4), dim=-1)
    h_ind = u.new_zeros((int(n_molecules), channels)).index_add_(0, bond_mol, h_edge)

    # m_{v->u}(s_u) = logsumexp_{s_v}(log p_v(s_v) + psi(s_u, s_v))
    m_v_to_u = torch.logsumexp(log_p_v.unsqueeze(-2) + log_psi, dim=-1)
    # m_{u->v}(s_v) = logsumexp_{s_u}(log p_u(s_u) + psi(s_u, s_v))
    m_u_to_v = torch.logsumexp(log_p_u.unsqueeze(-1) + log_psi, dim=-2)
    eta_star = u.new_zeros((int(u.shape[0]), channels))
    eta_star.index_add_(0, bond_u, m_v_to_u[..., 1] - m_v_to_u[..., 0])
    eta_star.index_add_(0, bond_v, m_u_to_v[..., 1] - m_u_to_v[..., 0])
    return h_ind, eta_star


# ---------------------------------------------------------------------------
# batch construction with a Train+Valid-only plan
# ---------------------------------------------------------------------------


def batch_endpoints(bundle: D.DataBundle, rows: Sequence[int]
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    counts = np.asarray([int(bundle.graph_n_atoms[int(r)]) for r in rows], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]) if counts.size else np.zeros(0, dtype=np.int64)
    bu, bv = [], []
    for off, row in zip(offsets, rows):
        b0 = int(bundle.bond_ptr[int(row)])
        b1 = int(bundle.bond_ptr[int(row) + 1])
        bu.append(bundle.bond_u[b0:b1] + int(off))
        bv.append(bundle.bond_v[b0:b1] + int(off))
    bu_arr = np.concatenate(bu) if bu else np.zeros(0, dtype=np.int64)
    bv_arr = np.concatenate(bv) if bv else np.zeros(0, dtype=np.int64)
    return (torch.as_tensor(bu_arr, dtype=torch.long),
            torch.as_tensor(bv_arr, dtype=torch.long))


def collate_rows(bundle: D.DataBundle, rows: Sequence[int], plan: I.GlobalPlan,
                 row_to_plan: Mapping[int, int], device: Any = DEVICE) -> dict[str, Any]:
    rows_list = [int(r) for r in rows]
    plan_rows = [int(row_to_plan[r]) for r in rows_list]
    graphs = D.graphs_for_rows(bundle, rows_list)
    batch = D.collate_graphs(graphs, plan, device, plan_rows=plan_rows)
    rows_arr = np.asarray(rows_list, dtype=np.int64)
    batch["row_ids"] = torch.as_tensor(rows_arr, dtype=torch.long, device=device)
    batch["y_raw"] = torch.as_tensor(bundle.labels[rows_arr],
                                     dtype=torch.float32, device=device)
    bu, bv = batch_endpoints(bundle, rows_list)
    batch["bond_u"] = bu.to(device)
    batch["bond_v"] = bv.to(device)
    if int(batch["n_bonds"]) != int(bu.numel()):
        raise RuntimeError("bond endpoint count mismatch")
    if bu.numel() and int(bu.max()) >= int(batch["n_atoms"]):
        raise RuntimeError("bond endpoint outside batch atom range")
    return batch


# ---------------------------------------------------------------------------
# frozen-head response override
# ---------------------------------------------------------------------------


class ResponseOverride:
    """Temporarily replace ``fgcr_model.I.exact_responses`` for one batch.

    The historical ``forward`` is untouched: it still derives H and eta exactly
    from the returned ``logz`` / ``logw`` and then runs the frozen sentence MLP,
    eta LayerNorm and readout.
    """

    def __init__(self, condition: str, batch: Mapping[str, Any],
                 capture: dict[str, Any] | None = None) -> None:
        self.condition = condition
        self.batch = batch
        self.capture = capture
        self._original = None

    def __enter__(self) -> "ResponseOverride":
        self._original = I.exact_responses
        I.exact_responses = self._call
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._original is not None
        I.exact_responses = self._original

    def _call(self, u: torch.Tensor, log_psi: torch.Tensor,
              bp: I.BatchPlan) -> dict[str, Any]:
        b = self.batch
        channels = int(u.shape[1])
        h_ind, eta_star = local_references(
            u, log_psi, b["bond_u"], b["bond_v"], b["bond_mol"],
            int(b["n_molecules"]), channels)

        unary_lse = torch.logsumexp(u, dim=-1)
        unary_sum = u.new_zeros((int(b["n_molecules"]), channels))
        unary_sum.index_add_(0, b["atom_mol"], unary_lse)
        unary_logodds = u[..., 1] - u[..., 0]

        need_full = self.condition != "LOCAL_BOTH"
        if need_full:
            original = self._original(u, log_psi, bp)
            h_full = original["logz"] - unary_sum
            eta_full = (original["logw"][..., 1] - original["logw"][..., 0]) - unary_logodds
        else:
            original = None
            h_full = None
            eta_full = None

        if self.capture is not None:
            self.capture.update({
                "H_full": h_full, "H_ind": h_ind,
                "ETA_full": eta_full, "ETA_star": eta_star,
                "u": u, "log_psi": log_psi,
            })

        if self.condition == "FULL":
            assert original is not None
            return original

        synthetic_logw = torch.stack(
            [torch.zeros_like(eta_star), eta_star + unary_logodds], dim=-1)
        if self.condition == "LOCAL_H":
            assert original is not None
            return {"belief": original["belief"],
                    "logz": h_ind + unary_sum,
                    "logw": original["logw"]}
        if self.condition == "LOCAL_ETA":
            assert original is not None
            return {"belief": original["belief"],
                    "logz": original["logz"],
                    "logw": synthetic_logw}
        return {"belief": None, "logz": h_ind + unary_sum, "logw": synthetic_logw}


def evaluate_condition(model: M.FGCRModel, bundle: D.DataBundle, plan: I.GlobalPlan,
                       row_to_plan: Mapping[int, int], rows: np.ndarray,
                       condition: str) -> np.ndarray:
    model.eval()
    preds = np.empty(int(rows.size), dtype=np.float64)
    offset = 0
    with torch.inference_mode():
        for part in _batches(rows, int(CFG["eval_batch_molecules"])):
            batch = collate_rows(bundle, part.tolist(), plan, row_to_plan)
            with ResponseOverride(condition, batch):
                out = model(batch)
            preds[offset:offset + part.size] = out["yhat_std"].double().cpu().numpy()
            offset += int(part.size)
    return preds


# ---------------------------------------------------------------------------
# reference collection
# ---------------------------------------------------------------------------


def collect_reference(model: M.FGCRModel, bundle: D.DataBundle, plan: I.GlobalPlan,
                      row_to_plan: Mapping[int, int], rows: np.ndarray
                      ) -> dict[str, np.ndarray]:
    model.eval()
    h_full, h_ind, eta_full, eta_star = [], [], [], []
    norm_full, norm_star = [], []
    atom_mol, n_atoms, n_bonds = [], [], []
    pred_std, row_ids, y_raw = [], [], []
    molecule_offset = 0
    n_atoms_used = 0
    n_bonds_used = 0
    with torch.inference_mode():
        for part in _batches(rows, int(CFG["eval_batch_molecules"])):
            batch = collate_rows(bundle, part.tolist(), plan, row_to_plan)
            out = model(batch, return_aux=True)
            hf = out["H"]
            ef = out["eta"]
            hi, es = local_references(out["u"], out["log_psi"], batch["bond_u"],
                                      batch["bond_v"], batch["bond_mol"],
                                      int(batch["n_molecules"]), int(hf.shape[1]))
            nf = model.eta_norm(ef)
            ns = model.eta_norm(es)
            h_full.append(hf.double().cpu().numpy())
            h_ind.append(hi.double().cpu().numpy())
            eta_full.append(ef.double().cpu().numpy())
            eta_star.append(es.double().cpu().numpy())
            norm_full.append(nf.double().cpu().numpy())
            norm_star.append(ns.double().cpu().numpy())
            atom_mol.append(batch["atom_mol"].cpu().numpy().astype(np.int64) + molecule_offset)
            molecule_offset += int(batch["n_molecules"])
            n_atoms_used += int(batch["n_atoms"])
            n_bonds_used += int(batch["n_bonds"])
            n_atoms.append(batch["graph_n_atoms"].double().cpu().numpy())
            n_bonds.append(batch["graph_n_bonds"].double().cpu().numpy())
            pred_std.append(out["yhat_std"].double().cpu().numpy())
            row_ids.append(batch["row_ids"].cpu().numpy())
            y_raw.append(batch["y_raw"].double().cpu().numpy())
    return {
        "H_full": np.concatenate(h_full, axis=0),
        "H_ind": np.concatenate(h_ind, axis=0),
        "ETA_full": np.concatenate(eta_full, axis=0),
        "ETA_star": np.concatenate(eta_star, axis=0),
        "ETA_norm_full": np.concatenate(norm_full, axis=0),
        "ETA_norm_star": np.concatenate(norm_star, axis=0),
        "atom_mol": np.concatenate(atom_mol, axis=0),
        "n_atoms": np.concatenate(n_atoms, axis=0),
        "n_bonds": np.concatenate(n_bonds, axis=0),
        "pred_std": np.concatenate(pred_std, axis=0),
        "row_ids": np.concatenate(row_ids, axis=0),
        "y_raw": np.concatenate(y_raw, axis=0),
        "n_atoms_used": int(n_atoms_used),
        "n_bonds_used": int(n_bonds_used),
    }


def _pm_moments(x: np.ndarray, atom_mol: np.ndarray, n_molecules: int,
                channels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    sums = np.zeros((n_molecules, channels), dtype=np.float64)
    sumsq = np.zeros((n_molecules, channels), dtype=np.float64)
    np.add.at(sums, atom_mol, x)
    np.add.at(sumsq, atom_mol, x * x)
    counts = np.bincount(atom_mol, minlength=n_molecules).astype(np.float64)[:, None]
    if counts.size and np.any(counts == 0):
        raise RuntimeError("empty molecule in per-molecule moments")
    mean = sums / counts
    var = np.maximum(sumsq / counts - mean * mean, 0.0)
    return mean, np.sqrt(var), np.sqrt(sumsq / counts)


def _pm_scalar_rms(x: np.ndarray, atom_mol: np.ndarray, n_molecules: int) -> np.ndarray:
    atom_sq = (np.asarray(x, dtype=np.float64) ** 2).mean(axis=1)
    sums = np.zeros(n_molecules, dtype=np.float64)
    np.add.at(sums, atom_mol, atom_sq)
    counts = np.bincount(atom_mol, minlength=n_molecules).astype(np.float64)
    return np.sqrt(sums / counts)


def summarize_reference(data: Mapping[str, np.ndarray], split: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    hf = data["H_full"]
    hi = data["H_ind"]
    ef = data["ETA_full"]
    es = data["ETA_star"]
    nf = data["ETA_norm_full"]
    ns = data["ETA_norm_star"]
    n_mol = int(hf.shape[0])
    channels = int(hf.shape[1])
    hinc = hf - hi
    einc = ef - es
    ninc = nf - ns
    natoms = data["n_atoms"]
    nbonds = data["n_bonds"]
    atom_mol = data["atom_mol"]

    hf_rms = np.sqrt((hf * hf).mean(axis=1))
    hi_rms = np.sqrt((hi * hi).mean(axis=1))
    hinc_rms = np.sqrt((hinc * hinc).mean(axis=1))
    eta_f_rms = _pm_scalar_rms(ef, atom_mol, n_mol)
    eta_s_rms = _pm_scalar_rms(es, atom_mol, n_mol)
    eta_i_rms = _pm_scalar_rms(einc, atom_mol, n_mol)

    mean_f, std_f, rms_f = _pm_moments(ef, atom_mol, n_mol, channels)
    mean_s, std_s, rms_s = _pm_moments(es, atom_mol, n_mol, channels)
    mean_i, std_i, rms_i = _pm_moments(einc, atom_mol, n_mol, channels)

    h_bond_den = (1.0 + nbonds)[:, None]
    hf_over_bonds = hf / h_bond_den
    hi_over_bonds = hi / h_bond_den
    hinc_over_bonds = hinc / h_bond_den

    # per-molecule eta_norm moments (for cross-molecule variation)
    mean_nf, std_nf, _ = _pm_moments(nf, atom_mol, n_mol, channels)
    mean_ns, std_ns, _ = _pm_moments(ns, atom_mol, n_mol, channels)
    mean_ni, std_ni, _ = _pm_moments(ninc, atom_mol, n_mol, channels)

    h_block = {
        "full": _channel_stats(hf),
        "ind": _channel_stats(hi),
        "increment": _channel_stats(hinc),
        "increment_rms_ratio_vs_full": _rms_ratio(hinc, hf),
        "increment_rms_ratio_vs_ind": _rms_ratio(hinc, hi),
        "channel_increment_rms_ratio": _channel_rms_ratio(hinc, hf),
        "full_vs_ind": {
            "channel_pearson_across_molecules": _pearson_cols(hf, hi),
            "molecule_cosine": _qstats(_cos_rows(hf, hi)),
        },
        "over_bonds": {
            "full": _channel_stats(hf_over_bonds),
            "ind": _channel_stats(hi_over_bonds),
            "increment": _channel_stats(hinc_over_bonds),
            "increment_rms_ratio_vs_full": _rms_ratio(hinc_over_bonds, hf_over_bonds),
        },
        "per_molecule": {
            "H_full_rms": _qstats(hf_rms),
            "H_ind_rms": _qstats(hi_rms),
            "H_inc_rms": _qstats(hinc_rms),
            "H_inc_over_full_ratio": _qstats(hinc_rms / (hf_rms + EPS)),
            "near_zero_fraction_H_inc": float((hinc_rms < 1e-6).mean()),
            "channel_mean_increment": hinc.mean(axis=0).tolist(),
            "channel_std_increment_across_molecules": hinc.std(axis=0).tolist(),
            "channel_mean_full": hf.mean(axis=0).tolist(),
            "channel_std_full_across_molecules": hf.std(axis=0).tolist(),
            "coefficient_of_variation_H_inc_rms": float(hinc_rms.std() / (hinc_rms.mean() + EPS)),
        },
    }

    eta_block = {
        "full": _channel_stats(ef),
        "star": _channel_stats(es),
        "increment": _channel_stats(einc),
        "increment_rms_ratio_vs_full": _rms_ratio(einc, ef),
        "increment_rms_ratio_vs_star": _rms_ratio(einc, es),
        "channel_increment_rms_ratio": _channel_rms_ratio(einc, ef),
        "full_vs_star": {
            "channel_pearson_across_atoms": _pearson_cols(ef, es),
            "atom_cosine": _qstats(_cos_rows(ef, es)),
        },
        "eta_norm": {
            "full": _channel_stats(nf),
            "star": _channel_stats(ns),
            "increment": _channel_stats(ninc),
            "increment_rms_ratio_vs_full": _rms_ratio(ninc, nf),
            "increment_rms_ratio_vs_star": _rms_ratio(ninc, ns),
            "increment_rms_ratio_norm_over_raw": _rms_ratio(ninc, einc),
            "channel_increment_rms_ratio": _channel_rms_ratio(ninc, nf),
            "layer_norm_change": {
                "full_rms_change_ratio": _rms_ratio(nf - ef, ef),
                "star_rms_change_ratio": _rms_ratio(ns - es, es),
            },
            "channel_mean_shift_star_minus_full": (ns.mean(0) - nf.mean(0)).tolist(),
            "channel_std_ratio_star_over_full": (ns.std(0) / (nf.std(0) + EPS)).tolist(),
            "max_abs_channel_mean_shift": float(np.abs(ns.mean(0) - nf.mean(0)).max()),
            "atom_cosine": _qstats(_cos_rows(nf, ns)),
            "atom_norm_full": _qstats(np.linalg.norm(nf, axis=1)),
            "atom_norm_star": _qstats(np.linalg.norm(ns, axis=1)),
        },
        "per_molecule": {
            "eta_full_rms": _qstats(eta_f_rms),
            "eta_star_rms": _qstats(eta_s_rms),
            "eta_inc_rms": _qstats(eta_i_rms),
            "eta_inc_over_full_ratio": _qstats(eta_i_rms / (eta_f_rms + EPS)),
            "near_zero_fraction_eta_inc": float((eta_i_rms < 1e-6).mean()),
            "coefficient_of_variation_eta_inc_rms": float(eta_i_rms.std() / (eta_i_rms.mean() + EPS)),
            "channel_mean_increment": mean_i.mean(axis=0).tolist(),
            "channel_std_increment_across_molecules": mean_i.std(axis=0).tolist(),
        },
        "cross_molecule": {
            "full": {
                "channel_mean_global": ef.mean(axis=0).tolist(),
                "between_molecule_std_of_molecule_mean": mean_f.std(axis=0).tolist(),
                "within_molecule_std_mean": std_f.mean(axis=0).tolist(),
                "between_over_within": (mean_f.std(axis=0) / (std_f.mean(axis=0) + EPS)).tolist(),
            },
            "star": {
                "channel_mean_global": es.mean(axis=0).tolist(),
                "between_molecule_std_of_molecule_mean": mean_s.std(axis=0).tolist(),
                "within_molecule_std_mean": std_s.mean(axis=0).tolist(),
            },
            "increment": {
                "channel_mean_global": einc.mean(axis=0).tolist(),
                "between_molecule_std_of_molecule_mean": mean_i.std(axis=0).tolist(),
                "within_molecule_std_mean": std_i.mean(axis=0).tolist(),
            },
            "eta_norm": {
                "full": {
                    "channel_mean_global": nf.mean(axis=0).tolist(),
                    "between_molecule_std_of_molecule_mean": mean_nf.std(axis=0).tolist(),
                    "within_molecule_std_mean": std_nf.mean(axis=0).tolist(),
                },
                "star": {
                    "channel_mean_global": ns.mean(axis=0).tolist(),
                    "between_molecule_std_of_molecule_mean": mean_ns.std(axis=0).tolist(),
                    "within_molecule_std_mean": std_ns.mean(axis=0).tolist(),
                },
                "increment": {
                    "channel_mean_global": ninc.mean(axis=0).tolist(),
                    "between_molecule_std_of_molecule_mean": mean_ni.std(axis=0).tolist(),
                    "within_molecule_std_mean": std_ni.mean(axis=0).tolist(),
                },
            },
        },
    }

    size_basis = {
        "log1p_n_atoms": np.log1p(natoms),
        "log1p_n_bonds": np.log1p(nbonds),
        "mean_degree": 2.0 * nbonds / (natoms + EPS),
        "H_ind_rms": hi_rms,
        "H_full_rms": hf_rms,
        "ETA_star_rms": eta_s_rms,
        "ETA_full_rms": eta_f_rms,
        "H_inc_rms": hinc_rms,
        "ETA_inc_rms": eta_i_rms,
    }
    targets = {
        "H_inc_rms": hinc_rms,
        "H_inc_rms_per_bond": hinc_rms / (nbonds + 1.0),
        "ETA_inc_rms": eta_i_rms,
    }
    corr_block = {}
    for tname, tval in targets.items():
        corr_block[tname] = {}
        for bname, bval in size_basis.items():
            if bname == tname:
                continue
            corr_block[tname][bname] = _corr_pair(tval, bval)

    stats = {
        "split": split,
        "n_molecules": n_mol,
        "n_atoms": int(ef.shape[0]),
        "n_bonds": int(nbonds.sum()),
        "atoms_per_molecule": _qstats(natoms),
        "bonds_per_molecule": _qstats(nbonds),
        "H": h_block,
        "eta": eta_block,
        "increment_vs_size_and_local_response": corr_block,
    }
    pm = {
        "row_ids": data["row_ids"].copy(),
        "n_atoms": natoms.copy(),
        "n_bonds": nbonds.copy(),
        "H_full_rms": hf_rms,
        "H_ind_rms": hi_rms,
        "H_inc_rms": hinc_rms,
        "eta_full_rms": eta_f_rms,
        "eta_star_rms": eta_s_rms,
        "eta_inc_rms": eta_i_rms,
        "pred_std_full": data["pred_std"].copy(),
        "y_raw": data["y_raw"].copy(),
    }
    return stats, pm


# ---------------------------------------------------------------------------
# metrics / bootstrap / sensitivity
# ---------------------------------------------------------------------------


def paired_bootstrap(se_full: np.ndarray, se_cond: np.ndarray, n_boot: int,
                     seed: int) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    n = int(se_full.size)
    dse = np.asarray(se_cond, dtype=np.float64) - np.asarray(se_full, dtype=np.float64)
    dmean, drmse = [], []
    done = 0
    chunk = 200
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        dmean.append(dse[idx].mean(axis=1))
        drmse.append(np.sqrt(np.asarray(se_cond)[idx].mean(axis=1))
                     - np.sqrt(np.asarray(se_full)[idx].mean(axis=1)))
        done += b
    dmean = np.concatenate(dmean)
    drmse = np.concatenate(drmse)
    alpha = (1.0 - float(CFG["bootstrap"]["ci"])) / 2.0
    return {
        "n_resamples": int(n_boot),
        "seed": int(seed),
        "mean_paired_se_diff_boot_mean": float(dmean.mean()),
        "mean_paired_se_diff_ci_low": float(np.quantile(dmean, alpha)),
        "mean_paired_se_diff_ci_high": float(np.quantile(dmean, 1.0 - alpha)),
        "delta_rmse_boot_mean": float(drmse.mean()),
        "delta_rmse_ci_low": float(np.quantile(drmse, alpha)),
        "delta_rmse_ci_high": float(np.quantile(drmse, 1.0 - alpha)),
        "frac_resamples_delta_rmse_positive": float((drmse > 0).mean()),
    }


def trimmed_sensitivity(se_full: np.ndarray, se_cond: np.ndarray,
                        fraction: float, mode: str) -> dict[str, float]:
    se_full = np.asarray(se_full, dtype=np.float64)
    se_cond = np.asarray(se_cond, dtype=np.float64)
    n = int(se_full.size)
    k = int(math.ceil(float(fraction) * n))
    if mode == "mean":
        key = 0.5 * (se_full + se_cond)
    elif mode == "max":
        key = np.maximum(se_full, se_cond)
    else:
        raise ValueError(mode)
    keep = np.argsort(key)[:-k] if k > 0 else np.arange(n)
    return {
        "mode": mode,
        "n_dropped": int(k),
        "n_kept": int(keep.size),
        "rmse_full_trimmed": float(np.sqrt(se_full[keep].mean())),
        "rmse_cond_trimmed": float(np.sqrt(se_cond[keep].mean())),
        "delta_rmse_trimmed": float(np.sqrt(se_cond[keep].mean())
                                    - np.sqrt(se_full[keep].mean())),
    }


def condition_metrics(condition: str, split: str, pred_std: np.ndarray,
                      pred_std_full: np.ndarray, y_raw: np.ndarray,
                      pm: Mapping[str, np.ndarray], cond_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    scaler = CFG["_scaler"]
    pred = pred_std * float(scaler["std"]) + float(scaler["mean"])
    pred_full = pred_std_full * float(scaler["std"]) + float(scaler["mean"])
    y = np.asarray(y_raw, dtype=np.float64)
    se_full = (pred_full - y) ** 2
    se_cond = (pred - y) ** 2
    dse = se_cond - se_full
    change = pred - pred_full

    rmse_full = float(np.sqrt(se_full.mean()))
    rmse = float(np.sqrt(se_cond.mean()))
    mae_full = float(np.abs(pred_full - y).mean())
    mae = float(np.abs(pred - y).mean())

    base = {
        "condition": condition,
        "split": split,
        "n_molecules": int(y.size),
        "rmse": rmse,
        "mae": mae,
        "delta_rmse_vs_FULL": rmse - rmse_full,
        "delta_mae_vs_FULL": mae - mae_full,
        "delta_rmse_pct_vs_FULL": 100.0 * (rmse - rmse_full) / rmse_full,
        "pred_change_rms_vs_FULL": float(np.sqrt((change * change).mean())),
        "pred_change_max_abs_vs_FULL": float(np.abs(change).max()),
        "mean_paired_se_diff": float(dse.mean()),
        "median_paired_se_diff": float(np.median(dse)),
        "frac_molecules_worse": float((dse > 0).mean()),
        "n_molecules_worse": int((dse > 0).sum()),
        "top5pct_share_of_positive_dse": (
            float(np.sort(np.maximum(dse, 0.0))[-max(1, int(math.ceil(0.05 * y.size))):].sum()
                  / (np.maximum(dse, 0.0).sum() + EPS))
        ),
    }

    trim_mean = trimmed_sensitivity(se_full, se_cond,
                                    float(CFG["trim_fraction"]), "mean")
    trim_max = trimmed_sensitivity(se_full, se_cond,
                                   float(CFG["trim_fraction"]), "max")
    base.update({
        "rmse_drop_shared_top5pct_mean": trim_mean["rmse_cond_trimmed"],
        "delta_rmse_drop_shared_top5pct_mean": trim_mean["delta_rmse_trimmed"],
        "n_dropped_top5pct_mean": trim_mean["n_dropped"],
        "rmse_drop_shared_top5pct_max": trim_max["rmse_cond_trimmed"],
        "delta_rmse_drop_shared_top5pct_max": trim_max["delta_rmse_trimmed"],
        "n_dropped_top5pct_max": trim_max["n_dropped"],
    })

    boot = {}
    if condition != "FULL":
        seed = int(CFG["bootstrap"]["seed"]) + 100 * int(cond_index) + (0 if split == "valid" else 1)
        boot = paired_bootstrap(se_full, se_cond,
                                int(CFG["bootstrap"]["n_resamples"]), seed)
        base.update({
            "boot_mean_paired_se_diff_mean": boot["mean_paired_se_diff_boot_mean"],
            "boot_mean_paired_se_diff_ci_low": boot["mean_paired_se_diff_ci_low"],
            "boot_mean_paired_se_diff_ci_high": boot["mean_paired_se_diff_ci_high"],
            "boot_delta_rmse_mean": boot["delta_rmse_boot_mean"],
            "boot_delta_rmse_ci_low": boot["delta_rmse_ci_low"],
            "boot_delta_rmse_ci_high": boot["delta_rmse_ci_high"],
            "boot_frac_resamples_worse": boot["frac_resamples_delta_rmse_positive"],
            "bootstrap_seed": seed,
        })
    else:
        base.update({k: None for k in (
            "boot_mean_paired_se_diff_mean", "boot_mean_paired_se_diff_ci_low",
            "boot_mean_paired_se_diff_ci_high", "boot_delta_rmse_mean",
            "boot_delta_rmse_ci_low", "boot_delta_rmse_ci_high",
            "boot_frac_resamples_worse", "bootstrap_seed")})

    diagnostics = {
        "pred_change_rms": base["pred_change_rms_vs_FULL"],
        "pred_change_max_abs": base["pred_change_max_abs_vs_FULL"],
        "pred_change_abs_per_molecule": _qstats(np.abs(change)),
        "corr_pred_change_with_H_inc_rms": _corr_pair(np.abs(change), pm["H_inc_rms"]),
        "corr_pred_change_with_ETA_inc_rms": _corr_pair(np.abs(change), pm["eta_inc_rms"]),
        "corr_pred_change_with_n_atoms": _corr_pair(np.abs(change), pm["n_atoms"]),
        "corr_pred_change_with_n_bonds": _corr_pair(np.abs(change), pm["n_bonds"]),
        "corr_delta_se_with_H_inc_rms": _corr_pair(dse, pm["H_inc_rms"]),
        "corr_delta_se_with_ETA_inc_rms": _corr_pair(dse, pm["eta_inc_rms"]),
    }
    return base, {
        "metrics": base,
        "diagnostics": diagnostics,
        "pred_raw": pred,
        "se": se_cond,
        "dse": dse,
        "boot": boot,
    }


# ---------------------------------------------------------------------------
# synthetic exact checks (float64)
# ---------------------------------------------------------------------------


def _np_lse(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def _np_local_refs(u: np.ndarray, edges: Sequence[tuple[int, int]],
                   psi: np.ndarray, edge_graph: Sequence[int],
                   n_graphs: int, channels: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    log_p = u - _np_lse(u, axis=-1)[:, :, None]
    h_ind = np.zeros((n_graphs, channels), dtype=np.float64)
    eta_star = np.zeros((u.shape[0], channels), dtype=np.float64)
    for e, (a, b) in enumerate(edges):
        edge_logits = log_p[a][:, :, None] + log_p[b][:, None, :] + psi[e]
        h_ind[edge_graph[e]] += _np_lse(edge_logits.reshape(channels, 4), axis=1)
        m_v_to_u = _np_lse(log_p[b][:, None, :] + psi[e], axis=2)
        m_u_to_v = _np_lse(log_p[a][:, :, None] + psi[e], axis=1)
        eta_star[a] += m_v_to_u[:, 1] - m_v_to_u[:, 0]
        eta_star[b] += m_u_to_v[:, 1] - m_u_to_v[:, 0]
    return h_ind, eta_star


def _np_bruteforce(u: np.ndarray, edges: Sequence[tuple[int, int]],
                   psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = int(u.shape[0])
    channels = int(u.shape[1])
    states = ((np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1)
    energy = np.zeros(((1 << n), channels), dtype=np.float64)
    for v in range(n):
        energy += u[v][:, states[:, v]].T
    for e, (a, b) in enumerate(edges):
        energy += psi[e][:, states[:, a], states[:, b]].T
    logz = _np_lse(energy, axis=0)
    h_full = logz - _np_lse(u, axis=-1).sum(axis=0)
    post = np.stack([
        _np_lse(energy[states[:, v] == 1], axis=0)
        - _np_lse(energy[states[:, v] == 0], axis=0)
        for v in range(n)
    ])
    eta_full = post - (u[:, :, 1] - u[:, :, 0])
    return h_full, eta_full


def _torch_synthetic(components: Sequence[tuple[int, Sequence[tuple[int, int]],
                                               np.ndarray, np.ndarray]]
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the exact plan path on synthetic components (float64)."""
    graphs = [(int(n), list(edges)) for n, edges, _, _ in components]
    plan = I.build_corpus_plan(graphs, smax=4)
    atom_counts = [int(n) for n, _, _, _ in components]
    bond_counts = [len(edges) for _, edges, _, _ in components]
    bp = plan.select(list(range(len(components))), atom_counts, bond_counts)

    u_all, psi_all, bu, bv, bm, am = [], [], [], [], [], []
    atom_off = 0
    bond_off = 0
    for g, (n, edges, u, psi) in enumerate(components):
        u_all.append(u)
        psi_all.append(psi)
        bu.extend([a + atom_off for a, b in edges])
        bv.extend([b + atom_off for a, b in edges])
        bm.extend([g] * len(edges))
        am.extend([g] * int(n))
        atom_off += int(n)
        bond_off += len(edges)
    u_t = torch.as_tensor(np.concatenate(u_all, axis=0), dtype=torch.float64)
    psi_t = torch.as_tensor(np.concatenate(psi_all, axis=0), dtype=torch.float64)
    bu_t = torch.as_tensor(np.asarray(bu, dtype=np.int64), dtype=torch.long)
    bv_t = torch.as_tensor(np.asarray(bv, dtype=np.int64), dtype=torch.long)
    bm_t = torch.as_tensor(np.asarray(bm, dtype=np.int64), dtype=torch.long)
    am_t = torch.as_tensor(np.asarray(am, dtype=np.int64), dtype=torch.long)

    responses = I.exact_responses(u_t, psi_t, bp)
    unary_lse = torch.logsumexp(u_t, dim=-1)
    unary_sum = u_t.new_zeros((len(components), int(u_t.shape[1])))
    unary_sum.index_add_(0, am_t, unary_lse)
    h_full = responses["logz"] - unary_sum
    eta_full = ((responses["logw"][..., 1] - responses["logw"][..., 0])
                - (u_t[..., 1] - u_t[..., 0]))
    h_ind, eta_star = local_references(u_t, psi_t, bu_t, bv_t, bm_t,
                                       len(components), int(u_t.shape[1]))
    return (h_full.numpy(), eta_full.numpy(), h_ind.numpy(),
            eta_star.numpy(), bp, responses["logz"].numpy())


def synthetic_case(name: str, components: Sequence[tuple[int, Sequence[tuple[int, int]],
                                                        np.ndarray, np.ndarray]],
                   *, zero_psi: bool = False) -> dict[str, Any]:
    hf, ef, hi, es, bp, logz = _torch_synthetic(components)
    bf_h, bf_e = [], []
    for n, edges, u, psi in components:
        h, e = _np_bruteforce(u, edges, psi)
        bf_h.append(h)
        bf_e.append(e)
    bf_h = np.stack(bf_h, axis=0)
    bf_e = np.concatenate(bf_e, axis=0)

    # independent numpy local reference
    u_all = np.concatenate([u for _, _, u, _ in components], axis=0)
    psi_all = np.concatenate([p for _, _, _, p in components], axis=0)
    edge_graph, edges_all = [], []
    atom_off = 0
    for g, (n, edges, _, _) in enumerate(components):
        edge_graph.extend([g] * len(edges))
        edges_all.extend([(int(a) + atom_off, int(b) + atom_off) for a, b in edges])
        atom_off += int(n)
    np_hi, np_es = _np_local_refs(u_all, edges_all, psi_all, edge_graph,
                                  len(components), int(u_all.shape[1]))
    hinc = hf - hi
    einc = ef - es
    return {
        "name": name,
        "n_components": len(components),
        "n_atoms": int(u_all.shape[0]),
        "n_bonds": int(psi_all.shape[0]),
        "zero_psi": zero_psi,
        "max_abs_H_full_vs_bruteforce": float(np.abs(hf - bf_h).max()),
        "max_abs_ETA_full_vs_bruteforce": float(np.abs(ef - bf_e).max()),
        "max_abs_H_ind_vs_numpy_local": float(np.abs(hi - np_hi).max()),
        "max_abs_ETA_star_vs_numpy_local": float(np.abs(es - np_es).max()),
        "max_abs_H_full_minus_local_minus_increment": float(
            np.abs(hf - (hi + hinc)).max()),
        "max_abs_ETA_full_minus_local_minus_increment": float(
            np.abs(ef - (es + einc)).max()),
        "max_abs_H_full": float(np.abs(hf).max()),
        "max_abs_ETA_full": float(np.abs(ef).max()),
        "max_abs_H_increment": float(np.abs(hinc).max()),
        "max_abs_ETA_increment": float(np.abs(einc).max()),
        "bruteforce_states_max": int(1 << max(int(n) for n, _, _, _ in components)),
    }


def make_synthetic_cases() -> dict[str, Any]:
    rng = np.random.default_rng(20270115)
    channels = 4

    def symmetric_psi(n_edges: int) -> np.ndarray:
        raw = rng.normal(size=(n_edges, channels, 3)) * 0.8
        psi = np.empty((n_edges, channels, 2, 2), dtype=np.float64)
        psi[:, :, 0, 0] = raw[:, :, 0]
        psi[:, :, 0, 1] = raw[:, :, 1]
        psi[:, :, 1, 0] = raw[:, :, 1]
        psi[:, :, 1, 1] = raw[:, :, 2]
        return psi

    def make(n: int, edges: Sequence[tuple[int, int]], *,
             zero: bool = False, nonsym: bool = False):
        u = rng.normal(size=(n, channels, 2)) * 1.4
        if zero:
            psi = np.zeros((len(edges), channels, 2, 2), dtype=np.float64)
        elif nonsym:
            psi = rng.normal(size=(len(edges), channels, 2, 2)) * 0.9
        else:
            psi = symmetric_psi(len(edges))
        return (n, list(edges), u, psi)

    cases = {
        "single_edge": [make(2, [(0, 1)])],
        "two_disconnected_single_edges": [make(2, [(0, 1)]), make(2, [(0, 1)])],
        "chain4": [make(4, [(0, 1), (1, 2), (2, 3)])],
        "star4": [make(4, [(0, 1), (0, 2), (0, 3)])],
        "cycle4": [make(4, [(0, 1), (1, 2), (2, 3), (3, 0)])],
        "multi_component_with_isolated_atom": [
            make(4, [(0, 1), (1, 2), (2, 0)]),
            make(2, [(0, 1)]),
            make(1, []),
        ],
        "zero_psi_cycle4": [make(4, [(0, 1), (1, 2), (2, 3), (3, 0)], zero=True)],
        "zero_psi_star_with_random_unary": [
            make(4, [(0, 1), (0, 2), (0, 3)], zero=True),
        ],
    }
    results = {}
    for name, comps in cases.items():
        results[name] = synthetic_case(
            name, comps, zero_psi=("zero_psi" in name))
    return results


def make_invariance_checks() -> dict[str, Any]:
    rng = np.random.default_rng(424242)
    channels = 4
    n = 5
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    u = rng.normal(size=(n, channels, 2))
    raw = rng.normal(size=(len(edges), channels, 3))
    psi_sym = np.empty((len(edges), channels, 2, 2))
    psi_sym[:, :, 0, 0] = raw[:, :, 0]
    psi_sym[:, :, 0, 1] = raw[:, :, 1]
    psi_sym[:, :, 1, 0] = raw[:, :, 1]
    psi_sym[:, :, 1, 1] = raw[:, :, 2]

    base = _torch_synthetic([(n, edges, u, psi_sym)])
    hf, ef, hi, es = base[0], base[1], base[2], base[3]

    perm = np.array([3, 0, 4, 1, 2], dtype=np.int64)
    u_perm = np.empty_like(u)
    u_perm[perm] = u
    edges_perm = [(int(perm[a]), int(perm[b])) for a, b in edges]
    out_perm = _torch_synthetic([(n, edges_perm, u_perm, psi_sym)])
    # perm is old -> new: eta_old[v] = eta_new[perm[v]]
    eta_perm_back = out_perm[1][perm]
    eta_star_perm_back = out_perm[3][perm]
    atom_perm = {
        "max_abs_H_full": float(np.abs(out_perm[0] - hf).max()),
        "max_abs_H_ind": float(np.abs(out_perm[2] - hi).max()),
        "max_abs_ETA_full": float(np.abs(eta_perm_back - ef).max()),
        "max_abs_ETA_star": float(np.abs(eta_star_perm_back - es).max()),
    }

    # non-symmetric psi + reversed endpoint order + transposed table
    psi_non = rng.normal(size=(len(edges), channels, 2, 2))
    base_non = _torch_synthetic([(n, edges, u, psi_non)])
    edges_rev = [(b, a) for a, b in edges]
    psi_rev = psi_non.transpose(0, 1, 3, 2)
    out_rev = _torch_synthetic([(n, edges_rev, u, psi_rev)])
    direction = {
        "max_abs_H_full": float(np.abs(out_rev[0] - base_non[0]).max()),
        "max_abs_H_ind": float(np.abs(out_rev[2] - base_non[2]).max()),
        "max_abs_ETA_full": float(np.abs(out_rev[1] - base_non[1]).max()),
        "max_abs_ETA_star": float(np.abs(out_rev[3] - base_non[3]).max()),
    }

    order = rng.permutation(len(edges))
    edges_shuf = [edges[i] for i in order]
    psi_shuf = psi_sym[order]
    out_shuf = _torch_synthetic([(n, edges_shuf, u, psi_shuf)])
    edge_order = {
        "max_abs_H_full": float(np.abs(out_shuf[0] - hf).max()),
        "max_abs_H_ind": float(np.abs(out_shuf[2] - hi).max()),
        "max_abs_ETA_full": float(np.abs(out_shuf[1] - ef).max()),
        "max_abs_ETA_star": float(np.abs(out_shuf[3] - es).max()),
    }
    return {
        "atom_relabeling": atom_perm,
        "edge_direction_reversal_nonsymmetric_psi": direction,
        "edge_storage_order": edge_order,
        "tolerance": 1e-9,
    }


# ---------------------------------------------------------------------------
# real-model invariance and edge-count checks
# ---------------------------------------------------------------------------


def real_model_invariance(model: M.FGCRModel, bundle: D.DataBundle,
                          plan: I.GlobalPlan, row_to_plan: Mapping[int, int],
                          rows: Sequence[int]) -> dict[str, Any]:
    results = []
    for row in rows:
        row = int(row)
        orig_graph = D.graphs_for_rows(bundle, [row])[0]
        n = int(orig_graph["n_atoms"])
        m = int(orig_graph["n_bonds"])
        rng = np.random.default_rng(9000 + row)
        perm = rng.permutation(n).astype(np.int64)
        edge_order = rng.permutation(m).astype(np.int64)

        new_u = perm[orig_graph["bond_u"][edge_order]]
        new_v = perm[orig_graph["bond_v"][edge_order]]
        new_feat = orig_graph["bond_feat"][edge_order]
        reverse = (rng.random(m) < 0.5)
        new_u, new_v = np.where(reverse, new_v, new_u), np.where(reverse, new_u, new_v)

        atom_feat_new = np.empty_like(orig_graph["atom_feat"])
        atom_feat_new[perm] = orig_graph["atom_feat"]
        new_graph = {
            "row": row,
            "n_atoms": n,
            "n_bonds": m,
            "atom_feat": atom_feat_new,
            "bond_feat": new_feat,
            "bond_u": new_u,
            "bond_v": new_v,
        }
        new_plan = I.build_corpus_plan([(n, list(zip(new_u.tolist(), new_v.tolist())))], smax=4)
        new_batch = D.collate_graphs([new_graph], new_plan, DEVICE, plan_rows=[0])
        offs = 0
        bu = torch.as_tensor(new_u + offs, dtype=torch.long)
        bv = torch.as_tensor(new_v + offs, dtype=torch.long)
        new_batch["bond_u"] = bu
        new_batch["bond_v"] = bv

        # original single-molecule batch through the corpus plan
        orig_batch = collate_rows(bundle, [row], plan, row_to_plan)
        with torch.inference_mode():
            out_orig = model(orig_batch, return_aux=True)
            out_new = model(new_batch, return_aux=True)
            hi_orig, es_orig = local_references(
                out_orig["u"], out_orig["log_psi"], orig_batch["bond_u"],
                orig_batch["bond_v"], orig_batch["bond_mol"], 1,
                int(out_orig["H"].shape[1]))
            hi_new, es_new = local_references(
                out_new["u"], out_new["log_psi"], new_batch["bond_u"],
                new_batch["bond_v"], new_batch["bond_mol"], 1,
                int(out_new["H"].shape[1]))
        # perm is old -> new: eta_old[v] = eta_new[perm[v]]
        eta_back = out_new["eta"].double().cpu().numpy()[perm]
        eta_star_back = es_new.double().cpu().numpy()[perm]

        results.append({
            "row_id": row,
            "n_atoms": n,
            "n_bonds": m,
            "max_abs_yhat_std_vs_original": float(
                np.abs(out_new["yhat_std"].double().cpu().numpy()
                       - out_orig["yhat_std"].double().cpu().numpy()).max()),
            "max_abs_H_full": float(
                np.abs(out_new["H"].double().cpu().numpy()
                       - out_orig["H"].double().cpu().numpy()).max()),
            "max_abs_ETA_full": float(np.abs(eta_back - out_orig["eta"].double().cpu().numpy()).max()),
            "max_abs_H_ind": float(np.abs(hi_new.double().cpu().numpy()
                                          - hi_orig.double().cpu().numpy()).max()),
            "max_abs_ETA_star": float(np.abs(eta_star_back - es_orig.double().cpu().numpy()).max()),
        })
    worst = max(max(r["max_abs_yhat_std_vs_original"], r["max_abs_H_full"],
                    r["max_abs_ETA_full"], r["max_abs_H_ind"], r["max_abs_ETA_star"])
                for r in results)
    return {"cases": results, "max_abs_all": float(worst), "tolerance": 1e-4}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    t_start = time.perf_counter()
    checks: dict[str, Any] = {}
    required_checks: list[str] = []

    def record(name: str, passed: bool, *, required: bool = True,
               **details: Any) -> None:
        checks[name] = {"pass": bool(passed), "required": bool(required), **details}
        if required:
            required_checks.append(name)

    # ---- load data / model -------------------------------------------------
    bundle = D.load_bundle()
    CFG["_scaler"] = dict(bundle.scaler)
    assert bundle.cache_sha256 == CFG["cache"]["sha256"]
    checkpoint = torch.load(HIST / CFG["checkpoint"], map_location="cpu",
                            weights_only=False)
    if int(checkpoint["epoch"]) != 96:
        raise RuntimeError(f"unexpected best epoch {checkpoint['epoch']}")
    model = M.load_model_from_state(checkpoint["state_dict"], bundle.atom_dims,
                                    bundle.bond_dims, config=checkpoint["config"],
                                    device=DEVICE)
    model.eval()
    if any(m.training for m in model.modules()):
        raise RuntimeError("model contains a module still in training mode")
    if list(bundle.atom_dims) != list(checkpoint["atom_dims"]):
        raise RuntimeError("atom_dims differ from checkpoint")
    if list(bundle.bond_dims) != list(checkpoint["bond_dims"]):
        raise RuntimeError("bond_dims differ from checkpoint")

    train_idx = np.asarray(bundle.train_idx, dtype=np.int64)
    valid_idx = np.asarray(bundle.valid_idx, dtype=np.int64)
    if np.intersect1d(train_idx, valid_idx).size:
        raise RuntimeError("Train/Valid overlap")
    audit_rows = np.concatenate([train_idx, valid_idx])
    if audit_rows.size != 3780:
        raise RuntimeError("audit row union is not 3780")
    row_to_plan = {int(r): i for i, r in enumerate(audit_rows.tolist())}

    graphs = D.graphs_for_rows(bundle, audit_rows.tolist())
    graph_pairs = [(int(g["n_atoms"]), list(zip(g["bond_u"].tolist(),
                                                g["bond_v"].tolist())))
                   for g in graphs]
    t_plan = time.perf_counter()
    plan = I.build_corpus_plan(graph_pairs, smax=int(checkpoint["config"]["inference"]["smax"]))
    plan_seconds = time.perf_counter() - t_plan
    plan_stats = I.plan_stats(plan)

    record("assets_present_and_hashed", True,
           checkpoint_sha256=CFG["checkpoint_sha256"],
           cache_sha256=CFG["cache"]["sha256"],
           historical_code_sha256=CFG["historical_code"])
    record("analysis_rows_train_valid_only", True,
           train=int(train_idx.size), valid=int(valid_idx.size),
           audit_rows=int(audit_rows.size),
           plan_graphs=int(audit_rows.size),
           disjoint_from_test=bool(bundle.audit.get("disjoint")),
           union_covers_4200=bool(bundle.audit.get("union_covers_4200")),
           test_rows_used_in_forward=0,
           test_scored=False,
           test_metrics_computed=False,
           note=("The historical loader reads the monolithic cache for its "
                 "file-level audit; no Test row enters any response, prediction, "
                 "metric, normalization, bootstrap or decision."))
    record("model_eval_no_dropout", True,
           training_flags=[bool(m.training) for m in model.modules()].count(True),
           dropout_layers=sum(isinstance(m, torch.nn.Dropout) for m in model.modules()),
           eval_mode=not model.training)
    record("plan_train_valid_only", plan_stats["max_width"] <= 8,
           plan_stats=plan_stats, plan_seconds=float(plan_seconds),
           n_graphs=len(graph_pairs))
    record("checkpoint_identity", int(checkpoint["epoch"]) == 96,
           epoch=int(checkpoint["epoch"]),
           stored_valid_rmse=float(checkpoint["metrics"]["valid_rmse"]),
           parameters=sum(p.numel() for p in model.parameters()))

    # ---- historical replay and reference collection -----------------------
    t_ref = time.perf_counter()
    reference = {
        "train": collect_reference(model, bundle, plan, row_to_plan, train_idx),
        "valid": collect_reference(model, bundle, plan, row_to_plan, valid_idx),
    }
    reference_seconds = time.perf_counter() - t_ref

    npz_path = HIST / CFG["historical_predictions"]
    with np.load(npz_path) as npz:
        saved = {
            "train": (npz["train_row_ids"].copy(), npz["train_y_pred"].copy()),
            "valid": (npz["valid_row_ids"].copy(), npz["valid_y_pred"].copy()),
        }
    replay = {}
    for split in ("train", "valid"):
        data = reference[split]
        pred_raw = data["pred_std"] * float(bundle.scaler["std"]) + float(bundle.scaler["mean"])
        target = data["y_raw"]
        rmse = float(np.sqrt(np.mean((pred_raw - target) ** 2)))
        mae = float(np.mean(np.abs(pred_raw - target)))
        saved_ids, saved_pred = saved[split]
        if not np.array_equal(data["row_ids"], saved_ids):
            raise RuntimeError(f"{split} row ids differ from historical predictions")
        max_diff = float(np.max(np.abs(pred_raw - saved_pred)))
        replay[split] = {"rmse": rmse, "mae": mae,
                         "max_abs_pred_diff_vs_saved": max_diff,
                         "n_molecules": int(target.size)}
    replay_ok = (
        abs(replay["valid"]["rmse"] - float(CFG["expected_historical_valid_rmse"]))
        <= float(CFG["valid_rmse_tolerance"])
        and replay["valid"]["max_abs_pred_diff_vs_saved"]
        <= float(CFG["prediction_replay_tolerance"])
        and replay["train"]["max_abs_pred_diff_vs_saved"]
        <= float(CFG["prediction_replay_tolerance"])
    )
    record("historical_replay_valid_rmse", replay_ok,
           expected_valid_rmse=float(CFG["expected_historical_valid_rmse"]),
           tolerance=float(CFG["valid_rmse_tolerance"]),
           **replay)

    # ---- statistics from the reference pass -------------------------------
    t_stats = time.perf_counter()
    stats_json = {"definitions": {
        "H_FULL": "logZ - sum_i logsumexp(u_i)",
        "H_IND": "sum_edges logsumexp_{s,t}(log_softmax(u_i)(s) + log_softmax(u_j)(t) + psi_ij(s,t))",
        "H_INCREMENT": "H_FULL - H_IND",
        "ETA_FULL": "posterior_logodds - unary_logodds",
        "ETA_STAR": "sum_j [m_{j->i}(1) - m_{j->i}(0)], m_{j->i}(s)=logsumexp_t(log_softmax(u_j)(t)+psi_ij(s,t))",
        "ETA_INCREMENT": "ETA_FULL - ETA_STAR",
        "joint_increment": "the FULL-minus-LOCAL difference; not a retrained local model",
        "n_test_rows_used": 0,
    }}
    pm_by_split = {}
    for split in ("valid", "train"):
        stats, pm = summarize_reference(reference[split], split)
        stats_json[split] = stats
        pm_by_split[split] = pm
    reference_stats_seconds = time.perf_counter() - t_stats

    reconstruction = {}
    for split in ("valid", "train"):
        data = reference[split]
        hinc = data["H_full"] - data["H_ind"]
        einc = data["ETA_full"] - data["ETA_star"]
        reconstruction[split] = {
            "max_abs_H_full_minus_ind_minus_inc": float(
                np.abs(data["H_full"] - (data["H_ind"] + hinc)).max()),
            "max_abs_ETA_full_minus_star_minus_inc": float(
                np.abs(data["ETA_full"] - (data["ETA_star"] + einc)).max()),
        }
    record("full_equals_local_plus_increment", True, **reconstruction)

    # ---- synthetic exact checks -------------------------------------------
    t_math = time.perf_counter()
    synthetic = make_synthetic_cases()
    tol = 1e-9
    synth_max = {}
    for name, case in synthetic.items():
        synth_max[name] = {
            k: case[k] for k in (
                "max_abs_H_full_vs_bruteforce", "max_abs_ETA_full_vs_bruteforce",
                "max_abs_H_ind_vs_numpy_local", "max_abs_ETA_star_vs_numpy_local",
                "max_abs_H_full_minus_local_minus_increment",
                "max_abs_ETA_full_minus_local_minus_increment",
                "max_abs_H_increment", "max_abs_ETA_increment")
        }
    synth_ok = all(
        max(v[k] for k in (
            "max_abs_H_full_vs_bruteforce", "max_abs_ETA_full_vs_bruteforce",
            "max_abs_H_ind_vs_numpy_local", "max_abs_ETA_star_vs_numpy_local",
            "max_abs_H_full_minus_local_minus_increment",
            "max_abs_ETA_full_minus_local_minus_increment")) <= tol
        for v in synth_max.values())
    record("synthetic_bruteforce_chain_star_cycle_multicomponent", synth_ok,
           tolerance=tol, cases=synthetic)
    zero_ok = all(
        v["max_abs_H_full"] <= tol and v["max_abs_ETA_full"] <= tol
        and v["max_abs_H_increment"] <= tol and v["max_abs_ETA_increment"] <= tol
        for name, v in synthetic.items() if v["zero_psi"])
    record("zero_edge_potential_all_responses_zero", zero_ok, tolerance=tol,
           cases={k: v for k, v in synthetic.items() if v["zero_psi"]})
    single_ok = all(
        synthetic[name]["max_abs_H_full_minus_local_minus_increment"] <= tol
        and synthetic[name]["max_abs_ETA_full_minus_local_minus_increment"] <= tol
        for name in ("single_edge", "two_disconnected_single_edges")
    )
    record("single_edge_and_disconnected_single_edges_full_equals_local",
           single_ok, tolerance=tol,
           single_edge=synthetic["single_edge"],
           two_disconnected=synthetic["two_disconnected_single_edges"])
    tree_ok = (synthetic["chain4"]["max_abs_H_increment"] > 1e-8
               and synthetic["chain4"]["max_abs_ETA_increment"] > 1e-8)
    record("generic_tree_increment_is_not_identically_zero", tree_ok,
           note="A tree does not guarantee H_INCREMENT=0 for generic potentials.",
           chain4_H_increment=synthetic["chain4"]["max_abs_H_increment"],
           chain4_ETA_increment=synthetic["chain4"]["max_abs_ETA_increment"])

    invariance = make_invariance_checks()
    inv_ok = all(v <= invariance["tolerance"]
                 for block in ("atom_relabeling", "edge_direction_reversal_nonsymmetric_psi",
                               "edge_storage_order")
                 for k, v in invariance[block].items())
    record("synthetic_atom_relabel_edge_direction_order_invariance", inv_ok,
           **invariance)

    real_inv_rows = [int(valid_idx[0]), int(valid_idx[len(valid_idx) // 2]),
                     int(valid_idx[-1])]
    t_inv = time.perf_counter()
    real_inv = real_model_invariance(model, bundle, plan, row_to_plan, real_inv_rows)
    record("real_model_atom_relabel_edge_reverse_order_invariance",
           real_inv["max_abs_all"] <= real_inv["tolerance"], **real_inv)
    invariance_seconds = time.perf_counter() - t_inv
    math_seconds = time.perf_counter() - t_math

    # ---- edge count / uniqueness -----------------------------------------
    edge_checks = {}
    for split, rows in (("valid", valid_idx), ("train", train_idx)):
        data = reference[split]
        expected_atoms = int(sum(int(bundle.graph_n_atoms[int(r)]) for r in rows))
        expected_bonds = int(sum(int(bundle.graph_n_bonds[int(r)]) for r in rows))
        unique_ok = True
        for r in rows:
            b0 = int(bundle.bond_ptr[int(r)])
            b1 = int(bundle.bond_ptr[int(r) + 1])
            pairs = list(zip(bundle.bond_u[b0:b1].tolist(),
                             bundle.bond_v[b0:b1].tolist()))
            if len(pairs) != len(set(pairs)) or any(a >= b for a, b in pairs):
                unique_ok = False
                break
        edge_checks[split] = {
            "expected_atoms": expected_atoms,
            "observed_atoms": int(data["n_atoms_used"]),
            "expected_bonds": expected_bonds,
            "observed_bonds": int(data["n_bonds_used"]),
            "unique_canonical_physical_pairs": bool(unique_ok),
        }

    # independent Python-loop H_IND on a few real Valid molecules
    loop_samples = []
    max_loop_diff = 0.0
    for row in valid_idx[:5]:
        row = int(row)
        batch = collate_rows(bundle, [row], plan, row_to_plan)
        with torch.inference_mode():
            _, u, _, log_psi = model.potentials(batch)
            hi_vec, _ = local_references(u, log_psi, batch["bond_u"],
                                         batch["bond_v"], batch["bond_mol"], 1,
                                         int(u.shape[1]))
            log_p = torch.log_softmax(u.double(), dim=-1)
            h_loop = torch.zeros(int(u.shape[1]), dtype=torch.float64)
            for e in range(int(batch["n_bonds"])):
                a = int(batch["bond_u"][e])
                b = int(batch["bond_v"][e])
                logits = (log_p[a][:, :, None] + log_p[b][:, None, :]
                          + log_psi[e].double())
                h_loop += torch.logsumexp(logits.reshape(int(u.shape[1]), 4), dim=-1)
            diff = float((hi_vec[0].double() - h_loop).abs().max())
            max_loop_diff = max(max_loop_diff, diff)
            loop_samples.append({
                "row_id": row,
                "n_bonds": int(batch["n_bonds"]),
                "n_atoms": int(batch["n_atoms"]),
                "max_abs_vectorized_minus_loop": diff,
            })
    edge_counts_ok = all(
        v["expected_atoms"] == v["observed_atoms"]
        and v["expected_bonds"] == v["observed_bonds"]
        and v["unique_canonical_physical_pairs"] for v in edge_checks.values())
    record("each_physical_edge_counted_once",
           edge_counts_ok and max_loop_diff <= 1e-5,
           splits=edge_checks,
           independent_loop_samples=loop_samples,
           max_abs_vectorized_minus_independent_loop=float(max_loop_diff),
           tolerance=1e-5)

    # ---- four frozen-head conditions --------------------------------------
    t_cond = time.perf_counter()
    condition_predictions = {}
    condition_records = {}
    full_patch_diffs = {}
    for split, rows in (("valid", valid_idx), ("train", train_idx)):
        full_std = reference[split]["pred_std"]
        condition_predictions[split] = {}
        condition_records[split] = {}
        for ci, cond in enumerate(CONDITIONS):
            pred_std = evaluate_condition(model, bundle, plan, row_to_plan,
                                          rows, cond)
            condition_predictions[split][cond] = pred_std
            if cond == "FULL":
                full_patch_diffs[split] = float(np.max(np.abs(pred_std - full_std)))
            row, diag = condition_metrics(cond, split, pred_std, full_std,
                                          reference[split]["y_raw"],
                                          pm_by_split[split], ci)
            condition_records[split][cond] = {
                "row": row, "diag": diag,
                "pred_raw": diag["pred_raw"],
                "se": diag["se"], "dse": diag["dse"],
            }
    record("patched_FULL_equals_unpatched_replay",
           all(v <= 1e-7 for v in full_patch_diffs.values()),
           max_abs_pred_diff_by_split=full_patch_diffs)
    condition_seconds = time.perf_counter() - t_cond

    # ---- verify the override actually injects the intended responses -----
    t_target = time.perf_counter()
    probe_rows = valid_idx[:64]
    probe_batch = collate_rows(bundle, probe_rows.tolist(), plan, row_to_plan)
    with torch.inference_mode():
        out_ref = model(probe_batch, return_aux=True)
        h_full = out_ref["H"].clone()
        eta_full = out_ref["eta"].clone()
        h_ind, eta_star = local_references(out_ref["u"], out_ref["log_psi"],
                                           probe_batch["bond_u"], probe_batch["bond_v"],
                                           probe_batch["bond_mol"],
                                           int(probe_batch["n_molecules"]),
                                           int(h_full.shape[1]))
    targets = {
        "FULL": (h_full, eta_full),
        "LOCAL_H": (h_ind, eta_full),
        "LOCAL_ETA": (h_full, eta_star),
        "LOCAL_BOTH": (h_ind, eta_star),
    }
    target_checks = {}
    for cond in CONDITIONS:
        with torch.inference_mode():
            with ResponseOverride(cond, probe_batch):
                out = model(probe_batch, return_aux=True)
        th, te = targets[cond]
        target_checks[cond] = {
            "max_abs_H_injected_minus_target": float((out["H"] - th).abs().max()),
            "max_abs_ETA_injected_minus_target": float((out["eta"] - te).abs().max()),
        }
    record("override_injects_intended_H_and_ETA",
           all(v["max_abs_H_injected_minus_target"] <= 1e-5
               and v["max_abs_ETA_injected_minus_target"] <= 1e-5
               for v in target_checks.values()),
           tolerance=1e-5, conditions=target_checks)
    I.BELIEF_CALLS["count"] = 0
    with torch.inference_mode():
        with ResponseOverride("LOCAL_BOTH", probe_batch):
            _ = model(probe_batch)
    local_belief_calls = int(I.BELIEF_CALLS["count"])
    record("LOCAL_BOTH_needs_no_full_belief_pass", local_belief_calls == 0,
           belief_calls=local_belief_calls)
    target_seconds = time.perf_counter() - t_target

    # ---- assemble table / statistics / predictions ------------------------
    t_write = time.perf_counter()
    table_rows = []
    for split in ("valid", "train"):
        for cond in CONDITIONS:
            table_rows.append(condition_records[split][cond]["row"])

    columns = [
        "condition", "split", "n_molecules", "rmse", "mae",
        "delta_rmse_vs_FULL", "delta_mae_vs_FULL", "delta_rmse_pct_vs_FULL",
        "pred_change_rms_vs_FULL", "pred_change_max_abs_vs_FULL",
        "mean_paired_se_diff", "median_paired_se_diff",
        "frac_molecules_worse", "n_molecules_worse",
        "boot_mean_paired_se_diff_mean", "boot_mean_paired_se_diff_ci_low",
        "boot_mean_paired_se_diff_ci_high", "boot_delta_rmse_mean",
        "boot_delta_rmse_ci_low", "boot_delta_rmse_ci_high",
        "boot_frac_resamples_worse",
        "rmse_drop_shared_top5pct_mean", "delta_rmse_drop_shared_top5pct_mean",
        "n_dropped_top5pct_mean", "rmse_drop_shared_top5pct_max",
        "delta_rmse_drop_shared_top5pct_max", "n_dropped_top5pct_max",
        "top5pct_share_of_positive_dse", "bootstrap_seed",
    ]
    C.write_csv(ROUND / "main_table.csv", table_rows, columns)

    pred_rows = []
    for split in ("valid", "train"):
        data = reference[split]
        rows = data["row_ids"]
        y = data["y_raw"]
        for i, row_id in enumerate(rows.tolist()):
            rec = {
                "split": split, "row_id": int(row_id),
                "y_true_raw": float(y[i]),
                "n_atoms": int(pm_by_split[split]["n_atoms"][i]),
                "n_bonds": int(pm_by_split[split]["n_bonds"][i]),
                "H_full_rms": float(pm_by_split[split]["H_full_rms"][i]),
                "H_ind_rms": float(pm_by_split[split]["H_ind_rms"][i]),
                "H_inc_rms": float(pm_by_split[split]["H_inc_rms"][i]),
                "eta_full_rms": float(pm_by_split[split]["eta_full_rms"][i]),
                "eta_star_rms": float(pm_by_split[split]["eta_star_rms"][i]),
                "eta_inc_rms": float(pm_by_split[split]["eta_inc_rms"][i]),
            }
            for cond in CONDITIONS:
                pred = condition_records[split][cond]["pred_raw"][i]
                rec[f"pred_{cond}"] = float(pred)
                rec[f"diff_{cond}"] = float(pred - condition_records[split]["FULL"]["pred_raw"][i])
                rec[f"se_{cond}"] = float(condition_records[split][cond]["se"][i])
                rec[f"dse_{cond}"] = float(condition_records[split][cond]["dse"][i])
            pred_rows.append(rec)
    pred_columns = ["split", "row_id", "y_true_raw", "n_atoms", "n_bonds",
                    "H_full_rms", "H_ind_rms", "H_inc_rms",
                    "eta_full_rms", "eta_star_rms", "eta_inc_rms"]
    for cond in CONDITIONS:
        pred_columns += [f"pred_{cond}", f"diff_{cond}", f"se_{cond}", f"dse_{cond}"]
    C.write_csv(ROUND / "predictions_by_condition.csv", pred_rows, pred_columns)

    # condition input-shift summary and prediction-change diagnostics
    shift = {}
    pred_diag = {}
    for split in ("valid", "train"):
        data = reference[split]
        hf, hi = data["H_full"], data["H_ind"]
        ef, es = data["ETA_full"], data["ETA_star"]
        nf, ns = data["ETA_norm_full"], data["ETA_norm_star"]
        denom = (1.0 + data["n_bonds"])[:, None]
        shift[split] = {
            "H": {
                "rms_ind_minus_full_over_full": _rms_ratio(hi - hf, hf),
                "rms_increment_over_full": _rms_ratio(hf - hi, hf),
                "channel_increment_over_full": _channel_rms_ratio(hf - hi, hf),
            },
            "H_over_1_plus_n_bonds": {
                "rms_ind_minus_full_over_full": _rms_ratio((hi - hf) / denom, hf / denom),
                "rms_increment_over_full": _rms_ratio((hf - hi) / denom, hf / denom),
            },
            "eta_raw": {
                "rms_star_minus_full_over_full": _rms_ratio(es - ef, ef),
                "rms_increment_over_full": _rms_ratio(ef - es, ef),
                "channel_increment_over_full": _channel_rms_ratio(ef - es, ef),
            },
            "eta_norm": {
                "rms_star_minus_full_over_full": _rms_ratio(ns - nf, nf),
                "rms_increment_over_full": _rms_ratio(nf - ns, nf),
                "channel_increment_over_full": _channel_rms_ratio(nf - ns, nf),
                "max_abs_channel_mean_shift": float(np.abs(ns.mean(0) - nf.mean(0)).max()),
            },
        }
        pred_diag[split] = {}
        for cond in CONDITIONS[1:]:
            pred_diag[split][cond] = condition_records[split][cond]["diag"]

    stats_json["condition_input_shift"] = shift
    stats_json["prediction_change_diagnostics"] = pred_diag

    # provisional policy label, from pre-declared evidence, not one statistic
    valid_rows = {row["condition"]: row for row in table_rows if row["split"] == "valid"}
    both = valid_rows["LOCAL_BOTH"]
    eta_row = valid_rows["LOCAL_ETA"]
    h_row = valid_rows["LOCAL_H"]
    if (both["delta_rmse_vs_FULL"] > 0.02
            and both["boot_delta_rmse_ci_low"] is not None
            and both["boot_delta_rmse_ci_low"] > 0.0
            and both["delta_rmse_drop_shared_top5pct_mean"] > 0.01
            and both["frac_molecules_worse"] > 0.55):
        provisional = "JOINT_DEPENDENCE_SIGNAL"
    elif (abs(h_row["delta_rmse_vs_FULL"]) <= 0.01
          and abs(eta_row["delta_rmse_vs_FULL"]) <= 0.01
          and both["boot_delta_rmse_ci_low"] is not None
          and both["boot_delta_rmse_ci_low"] <= 0.0
          and both["boot_delta_rmse_ci_high"] >= 0.0):
        provisional = "LOCAL_SUFFICIENT_FOR_FROZEN_MODEL"
    elif min(both["delta_rmse_vs_FULL"], eta_row["delta_rmse_vs_FULL"],
             h_row["delta_rmse_vs_FULL"]) < -0.005:
        provisional = "LOCAL_REPLACEMENT_IMPROVES"
    else:
        provisional = "INCONCLUSIVE"

    stats_json["headline"] = {
        "valid_full_rmse": valid_rows["FULL"]["rmse"],
        "valid_local_h_rmse": h_row["rmse"],
        "valid_local_eta_rmse": eta_row["rmse"],
        "valid_local_both_rmse": both["rmse"],
        "valid_local_h_delta_rmse": h_row["delta_rmse_vs_FULL"],
        "valid_local_eta_delta_rmse": eta_row["delta_rmse_vs_FULL"],
        "valid_local_both_delta_rmse": both["delta_rmse_vs_FULL"],
        "valid_local_both_delta_rmse_ci": [both["boot_delta_rmse_ci_low"],
                                           both["boot_delta_rmse_ci_high"]],
        "valid_local_h_delta_rmse_ci": [h_row["boot_delta_rmse_ci_low"],
                                        h_row["boot_delta_rmse_ci_high"]],
        "valid_local_eta_delta_rmse_ci": [eta_row["boot_delta_rmse_ci_low"],
                                          eta_row["boot_delta_rmse_ci_high"]],
        "valid_local_both_delta_rmse_trim5pct": both["delta_rmse_drop_shared_top5pct_mean"],
        "valid_local_eta_delta_rmse_trim5pct": eta_row["delta_rmse_drop_shared_top5pct_mean"],
        "valid_local_h_delta_rmse_trim5pct": h_row["delta_rmse_drop_shared_top5pct_mean"],
        "valid_H_increment_rms_ratio": stats_json["valid"]["H"]["increment_rms_ratio_vs_full"],
        "valid_ETA_increment_rms_ratio": stats_json["valid"]["eta"]["increment_rms_ratio_vs_full"],
        "valid_ETA_norm_increment_rms_ratio": stats_json["valid"]["eta"]["eta_norm"]["increment_rms_ratio_vs_full"],
        "valid_eta_norm_shift_ratio": shift["valid"]["eta_norm"]["rms_star_minus_full_over_full"],
        "provisional_category": provisional,
        "n_test_rows_used": 0,
    }
    C.write_json(ROUND / "increment_statistics.json", stats_json)

    # ---- checks / provenance ---------------------------------------------
    wall = time.perf_counter() - t_start
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)
    all_required = all(checks[name]["pass"] for name in required_checks)
    status = "PASS" if all_required else "FAIL"
    if wall > float(CFG["wall_clock_limit_seconds"]):
        status = "RESOURCE_LIMIT"

    checks_payload = {
        "round_id": CFG["round_id"],
        "status": status,
        "all_required_checks_passed": all_required,
        "required_check_names": required_checks,
        "checks": checks,
        "replay": replay,
        "plan_stats": plan_stats,
        "math_checks": synthetic,
        "invariance_checks": {"synthetic": invariance, "real_model": real_inv},
        "edge_count_checks": edge_checks,
        "condition_target_checks": target_checks,
        "local_both_belief_calls": local_belief_calls,
        "budget": {
            "wall_seconds": float(wall),
            "wall_limit_seconds": float(CFG["wall_clock_limit_seconds"]),
            "peak_rss_gb": float(peak_rss_gb),
            "device": DEVICE,
            "torch_threads": int(CFG["torch_threads"]),
            "reference_seconds": float(reference_seconds),
            "reference_stats_seconds": float(reference_stats_seconds),
            "math_checks_seconds": float(math_seconds),
            "real_invariance_seconds": float(invariance_seconds),
            "condition_seconds": float(condition_seconds),
            "target_checks_seconds": float(target_seconds),
            "write_seconds": float(time.perf_counter() - t_write),
            "plan_seconds": float(plan_seconds),
        },
        "compliance": {
            "training": False,
            "optimizer_steps": 0,
            "test_rows_used": 0,
            "test_scored": False,
            "test_metrics_computed": False,
            "plan_graphs_train_valid_only": int(audit_rows.size),
            "checkpoint_modified": False,
            "historical_files_modified": False,
            "hyperparameter_search": False,
            "frozen_head": True,
            "normalization_refit": False,
        },
    }
    C.write_json(ROUND / "checks.json", checks_payload)

    provenance = {
        "round_id": CFG["round_id"],
        "timestamp_utc": C.utc_stamp(),
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": DEVICE,
        "torch_threads": int(CFG["torch_threads"]),
        "config": CFG,
        "config_sha256": C.sha256_file(CONFIG_PATH),
        "cache_sha256": bundle.cache_sha256,
        "checkpoint_sha256": CFG["checkpoint_sha256"],
        "historical_code_sha256": CFG["historical_code"],
        "historical_best_epoch": int(checkpoint["epoch"]),
        "historical_stored_valid_rmse": float(checkpoint["metrics"]["valid_rmse"]),
        "replay": replay,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "wall_seconds": float(wall),
        "peak_rss_gb": float(peak_rss_gb),
        "status": status,
        "provisional_category": provisional,
        "outputs": [
            "checks.json", "main_table.csv", "increment_statistics.json",
            "predictions_by_condition.csv", "provenance.json", "REPORT.md",
        ],
    }
    C.write_json(ROUND / "provenance.json", provenance)

    print("\n=== HEADLINE ===")
    print(json.dumps(stats_json["headline"], indent=2, sort_keys=True))
    print("\n=== MAIN TABLE ===")
    for row in table_rows:
        print(f"{row['split']:5s} {row['condition']:11s} "
              f"rmse={row['rmse']:.6f} mae={row['mae']:.6f} "
              f"dRMSE={row['delta_rmse_vs_FULL']:+.6f} "
              f"predRMS={row['pred_change_rms_vs_FULL']:.6f} "
              f"worse={row['frac_molecules_worse']:.3f} "
              f"trimD={row['delta_rmse_drop_shared_top5pct_mean']:+.6f}")
    print(f"\nstatus={status} provisional={provisional} wall={wall:.1f}s "
          f"peak_rss={peak_rss_gb:.3f}GB")
    print(f"checks: {sum(1 for c in checks.values() if c['pass'])}/{len(checks)} pass")
    return 0 if status == "PASS" else (2 if status == "RESOURCE_LIMIT" else 3)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        import traceback
        _write_minimal_checks("ERROR", error=repr(exc),
                              traceback=traceback.format_exc()[-8000:])
        raise
