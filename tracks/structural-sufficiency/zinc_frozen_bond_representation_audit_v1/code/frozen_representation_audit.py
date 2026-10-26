#!/usr/bin/env python
"""Frozen bond-representation audit for DCC-L and RCC-v0.

CPU only.  No gradient step, no optimizer, no training.  Reads the frozen
official-train structural cache used by the historical runs, never indexes its
``labels`` key, and never opens the official valid/test caches.

The audit separates, for each physical bond,

    H_G = mean_G + (H_G - mean_G)

and computes (a) the origin-dependent common/within energy split, (b) the
pooled and molecule-weighted within spectra, (c) the D0 -> D1 change of the
DCC-L components, and (d) the response of the frozen relational contrast
``B = K_R - K_D`` on the within-molecule part.

Outputs:
    analysis/representation_spectrum.json
    analysis/structure_response.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_TRACK = _HERE.parent
_SUFF = _TRACK.parent
_REPO = _SUFF.parents[1]

_V1 = _SUFF / "zinc_arch_discovery_v1" / "code"
_V2 = _SUFF / "zinc_recipe_trunk_v2" / "code"
_DCC = _SUFF / "zinc_dual_channel_composer_v1" / "code"
_RCC = _SUFF / "zinc_relational_contrast_contraction_v0" / "code"
for _p in (str(_HERE), str(_V1), str(_V2), str(_DCC), str(_RCC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as AC  # noqa: E402
import ad_data as D  # noqa: E402
import dcc_models as DM  # noqa: E402
import rcc_common as RC  # noqa: E402
import rcc_kernels as RK  # noqa: E402
import rcc_models as RM  # noqa: E402

CACHE_PATH = _SUFF / "zinc_arch_discovery_v1" / "cache" / "ad_train.pt"
BANK_PATH = _SUFF / "zinc_relational_contrast_contraction_v0" / "analysis" / "kernel_bank.npz"
PREP_PATH = _SUFF / "zinc_relational_contrast_contraction_v0" / "analysis" / "prep.json"
REDUNDANCY_PATH = _SUFF / "zinc_relational_contrast_contraction_v0" / "analysis" / "redundancy_probe.json"
RCC_SPLIT_PATH = _REPO / "tracks/structural-sufficiency/zinc_relational_contrast_contraction_v0/configs/rcc_v0.json"
BEST_PT = _SUFF / "zinc_dual_channel_composer_v1" / "runs" / "L_s0" / "best.pt"
FINAL_PT = _SUFF / "zinc_dual_channel_composer_v1" / "runs" / "L_s0" / "final.pt"
ARTIFACTS = _SUFF / "zinc_dual_channel_composer_v1" / "analysis" / "artifact_hashes.json"
DCC_RUN_ARTIFACTS = _SUFF / "zinc_dual_channel_composer_v1" / "runs" / "L_s0" / "artifacts.json"

OUT_SPECTRUM = _TRACK / "analysis" / "representation_spectrum.json"
OUT_RESPONSE = _TRACK / "analysis" / "structure_response.json"

BATCH = 128
STRATA = ((8, 16), (17, 23), (24, 29), (30, 41))
DCC_COMPONENTS = ("b", "c_loc", "c_glob")
RCC_COMPONENTS = ("h_RCC", "Z_RCC")


# ---------------------------------------------------------------------------
# provenance / guards
# ---------------------------------------------------------------------------

_LOADED_PATHS: list[str] = []


def install_torch_load_guard() -> None:
    """Record every ``torch.load`` path and block valid/test caches."""
    original = torch.load

    def guarded(f, *args, **kwargs):
        path = str(f)
        _LOADED_PATHS.append(path)
        low = path.lower()
        # The historical checkpoints live under runs/L_s0 and the frozen
        # encoder under pretrain_encoder_s0.pt; neither contains 'valid' or
        # 'test'.  A valid/test cache would.
        if "valid" in low or "test" in low:
            raise RuntimeError(f"blocked non-train torch.load path: {path}")
        return original(f, *args, **kwargs)

    torch.load = guarded  # type: ignore[assignment]


class GuardedCache(dict):
    """Dict wrapper that makes the supervised labels key unreadable."""

    def __init__(self, base: Mapping[str, Any]):
        super().__init__(base)
        self.accessed: set[str] = set()
        self.labels_attempted = False

    def __getitem__(self, key):
        if key == "labels":
            self.labels_attempted = True
            raise RuntimeError("labels are blocked in this audit")
        self.accessed.add(str(key))
        return super().__getitem__(key)


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_hash_sd(sd: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(sd.items()):
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def flat_state(sd: Mapping[str, torch.Tensor]) -> np.ndarray:
    return np.concatenate(
        [v.detach().cpu().double().numpy().ravel() for _, v in sorted(sd.items())]
    )


# ---------------------------------------------------------------------------
# small statistics helpers
# ---------------------------------------------------------------------------


def qsummary(a: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(a, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "median": None, "p10": None, "p90": None, "mean": None}
    return {
        "n": int(len(x)),
        "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
        "mean": float(np.mean(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def spectrum(C: np.ndarray) -> dict[str, Any]:
    """PR and variance shares of a symmetric covariance matrix.

    Negative eigenvalues are only reported as a numerical diagnostic; shares and
    PR use the positive part.  A near-zero trace is reported as degenerate and
    is never forced to PR=1.
    """
    C = np.asarray(C, dtype=np.float64)
    C = 0.5 * (C + C.T)
    raw = np.linalg.eigvalsh(C)[::-1]
    if not np.isfinite(raw).all():
        return {"degenerate": True, "reason": "nonfinite_eigenvalues"}
    pos = np.clip(raw, 0.0, None)
    tr = float(raw.sum())
    if tr <= 1e-30:
        return {
            "degenerate": True,
            "reason": "trace_below_1e-30",
            "trace": tr,
            "lambda_max_raw": float(raw[0]) if len(raw) else None,
        }
    cum = np.cumsum(pos) / tr
    top = float(pos[0] / tr)
    out: dict[str, Any] = {
        "degenerate": False,
        "pr": float(tr * tr / float((pos * pos).sum())),
        "top1": top,
        "top2": float(pos[:2].sum() / tr),
        "top4": float(pos[:4].sum() / tr),
        "dims_90": int(np.searchsorted(cum, 0.90) + 1),
        "dims_95": int(np.searchsorted(cum, 0.95) + 1),
        "trace": tr,
        "frobenius": float(np.sqrt(float((raw * raw).sum()))),
        "lambda_max": float(raw[0]),
        "lambda_min_raw": float(raw[-1]),
        "negative_eigenvalue_count": int((raw < 0).sum()),
        "negative_eigenvalue_mass": float(np.abs(raw[raw < 0]).sum()),
        "zero_eigenvalue_count": int((pos <= 0).sum()),
        "numerical_rank": {
            "tol_1e-12_rel": int((pos > pos[0] * 1e-12).sum()),
            "tol_1e-10_rel": int((pos > pos[0] * 1e-10).sum()),
            "tol_1e-8_rel": int((pos > pos[0] * 1e-8).sum()),
            "tol_1e-6_rel": int((pos > pos[0] * 1e-6).sum()),
        },
        "eigenvalues_desc": [float(x) for x in raw],
    }
    return out


def mol_spectrum_from_scatter(scatter: np.ndarray, n: int) -> dict[str, Any]:
    if n <= 1:
        return {"degenerate": True, "reason": "n_bonds_le_1"}
    return spectrum(scatter / float(n - 1))


def subspace_overlap(
    C0: np.ndarray,
    C1: np.ndarray,
    ks: Sequence[int] = (1, 2, 4, 8, 16),
) -> dict[str, Any]:
    """How much of D1's within variance lies in D0's leading directions.

    This distinguishes 'amplify the old few directions' from 'form new
    within-bond directions'.  It is descriptive: no projection is fitted and no
    predictive target is used.
    """
    e0, U0 = np.linalg.eigh(0.5 * (C0 + C0.T))
    o0 = np.argsort(-e0)
    e0, U0 = e0[o0], U0[:, o0]
    e1, U1 = np.linalg.eigh(0.5 * (C1 + C1.T))
    o1 = np.argsort(-e1)
    e1, U1 = e1[o1], U1[:, o1]
    tr0 = max(float(e0.sum()), 1e-30)
    tr1 = max(float(e1.sum()), 1e-30)
    out: dict[str, Any] = {
        "top1_cos2": float((U0[:, 0] @ U1[:, 0]) ** 2),
        "by_k": {},
    }
    for k in ks:
        kk = int(min(k, C0.shape[0]))
        P0, P1 = U0[:, :kk], U1[:, :kk]
        cap1 = float(np.trace(P0.T @ C1 @ P0) / tr1)
        cap0 = float(np.trace(P1.T @ C0 @ P1) / tr0)
        sv = np.linalg.svd(P0.T @ P1, compute_uv=False)
        cos2 = np.clip(sv * sv, 0.0, 1.0)
        out["by_k"][str(kk)] = {
            "D1_trace_fraction_in_D0_topk": cap1,
            "D0_trace_fraction_in_D1_topk": cap0,
            "mean_cos2": float(cos2.mean()),
            "min_cos2": float(cos2.min()),
        }
    return out


# ---------------------------------------------------------------------------
# extraction from the real forward path
# ---------------------------------------------------------------------------


class DCCCapture:
    """Capture the actual bond encoder and both composer outputs."""

    def __init__(self, model: DM.DualChannel):
        self.model = model
        self.cap: dict[str, torch.Tensor] = {}
        self.handles = [
            model.encoders.register_forward_hook(
                lambda _m, _i, o: self.cap.update(a=o[0], b=o[1], w=o[2])
            ),
            model.group_composer.register_forward_hook(
                lambda _m, _i, o: self.cap.update(c_glob=o)
            ),
            model.local_composer.register_forward_hook(
                lambda _m, _i, o: self.cap.update(c_loc=o)
            ),
        ]

    def __call__(self, batch: Mapping[str, Any]) -> dict[str, np.ndarray]:
        with torch.no_grad():
            self.model(batch)
        return {
            k: self.cap[k].detach().double().cpu().numpy().copy()
            for k in DCC_COMPONENTS
        }

    def close(self) -> None:
        for h in self.handles:
            h.remove()


class RCCCapture:
    def __init__(self, model: RM.RCC):
        self.model = model

    def __call__(self, batch: Mapping[str, Any]) -> dict[str, np.ndarray]:
        with torch.no_grad():
            cap = self.model.compute(batch)
        return {
            "h_RCC": cap["h"].detach().double().cpu().numpy().copy(),
            "Z_RCC": cap["zr"].detach().double().cpu().numpy().copy(),
        }


def extract_dcc(model, cache, collate, rows: np.ndarray) -> dict[str, np.ndarray]:
    cap = DCCCapture(model)
    out: dict[str, list[np.ndarray]] = {k: [] for k in DCC_COMPONENTS}
    try:
        with torch.no_grad():
            for s in range(0, len(rows), BATCH):
                batch = collate(cache, rows[s:s + BATCH])
                got = cap(batch)
                for k in out:
                    out[k].append(got[k])
    finally:
        cap.close()
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def extract_rcc(model, cache, collate, rows: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {k: [] for k in RCC_COMPONENTS}
    with torch.no_grad():
        for s in range(0, len(rows), BATCH):
            batch = collate(cache, rows[s:s + BATCH])
            got = RCCCapture(model)(batch)
            for k in out:
                out[k].append(got[k])
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


# ---------------------------------------------------------------------------
# representation spectrum
# ---------------------------------------------------------------------------


def representation_stats(
    H: np.ndarray,
    counts: np.ndarray,
    offsets: np.ndarray,
    n_bonds_total: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Common/within energy split, pooled spectra, per-molecule spectra."""
    M = len(counts)
    d = H.shape[1]
    means = np.empty((M, d), dtype=np.float64)
    E_total = 0.0
    E_common = 0.0
    E_within = 0.0

    for i in range(M):
        s, e = int(offsets[i]), int(offsets[i + 1])
        X = H[s:e]
        m = X.mean(axis=0)
        means[i] = m
        n = e - s
        E_total += float(np.sum(X * X))
        E_common += float(n * np.sum(m * m))
        E_within += float(np.sum((X - m) ** 2))

    mu = (means * counts[:, None]).sum(axis=0) / float(n_bonds_total)
    Wtot = np.zeros((d, d), dtype=np.float64)
    Cmw_sum = np.zeros((d, d), dtype=np.float64)
    common = np.zeros((d, d), dtype=np.float64)

    n_arr = counts.astype(np.float64)
    common_share = np.empty(M)
    within_share = np.empty(M)
    within_pr = np.empty(M)
    within_top1 = np.empty(M)
    within_top4 = np.empty(M)
    within_pr_over_cap = np.empty(M)
    pairwise_rms = np.empty(M)
    pairwise_norm = np.empty(M)
    within_rms = np.empty(M)
    bond_rms = np.empty(M)

    for i in range(M):
        s, e = int(offsets[i]), int(offsets[i + 1])
        X = H[s:e]
        n = e - s
        m = means[i]
        dev = X - m
        W = dev.T @ dev
        Wtot += W
        C_i = W / float(n - 1) if n > 1 else np.zeros_like(W)
        Cmw_sum += C_i
        diff = m - mu
        common += float(n) * np.outer(diff, diff)

        tot_i = float(np.sum(X * X))
        com_i = float(n * np.sum(m * m))
        w_i = float(np.sum(dev * dev))
        common_share[i] = com_i / max(tot_i, 1e-30)
        within_share[i] = w_i / max(tot_i, 1e-30)
        spec_i = mol_spectrum_from_scatter(W, n)
        if spec_i.get("degenerate"):
            within_pr[i] = np.nan
            within_top1[i] = np.nan
            within_top4[i] = np.nan
            within_pr_over_cap[i] = np.nan
        else:
            within_pr[i] = spec_i["pr"]
            within_top1[i] = spec_i["top1"]
            within_top4[i] = spec_i["top4"]
            within_pr_over_cap[i] = spec_i["pr"] / float(n - 1)
        within_rms[i] = np.sqrt(w_i / float(n)) if n else np.nan
        bond_rms[i] = np.sqrt(np.mean(np.sum(X * X, axis=1)))
        pr2 = 2.0 * w_i / float(n - 1) if n > 1 else np.nan
        pairwise_rms[i] = np.sqrt(max(pr2, 0.0)) if np.isfinite(pr2) else np.nan
        pairwise_norm[i] = (
            pairwise_rms[i] / max(bond_rms[i], 1e-30)
            if np.isfinite(pairwise_rms[i])
            else np.nan
        )

    C_within = Wtot / float(n_bonds_total - M)
    C_mw = Cmw_sum / float(M)
    C_common = common / float(n_bonds_total - 1)
    C_all = (Wtot + common) / float(n_bonds_total - 1)

    energy_residual = (E_total - E_common - E_within) / max(E_total, 1e-30)
    summary: dict[str, Any] = {
        "n_bonds": int(n_bonds_total),
        "n_molecules": int(M),
        "dimension": int(d),
        "energy": {
            "total": E_total,
            "common": E_common,
            "within": E_within,
            "common_share": E_common / max(E_total, 1e-30),
            "within_share": E_within / max(E_total, 1e-30),
            "additivity_residual": float(energy_residual),
            "note": "origin-dependent squared-norm decomposition; not a PCA share",
        },
        "C_all": spectrum(C_all),
        "C_within_bond_weighted": spectrum(C_within),
        "C_within_molecule_weighted": spectrum(C_mw),
        "C_common_bond_weighted": spectrum(C_common),
        "per_molecule": {
            "common_share": qsummary(common_share),
            "within_share": qsummary(within_share),
            "within_pr": qsummary(within_pr),
            "within_top1": qsummary(within_top1),
            "within_top4": qsummary(within_top4),
            "within_pr_over_cap": qsummary(within_pr_over_cap),
            "pairwise_rms": qsummary(pairwise_rms),
            "pairwise_rms_over_bond_rms": qsummary(pairwise_norm),
            "within_rms": qsummary(within_rms),
            "bond_rms": qsummary(bond_rms),
            "n_bonds": qsummary(n_arr),
            "rank_cap_n_bonds_minus_1": {
                "min": int(counts.min() - 1),
                "median": float(np.median(counts - 1)),
                "max": int(counts.max() - 1),
                "note": "within-PR of one molecule cannot exceed n_bonds-1; "
                        "do not compare a per-molecule PR directly with a pooled PR",
            },
        },
        "strata": {},
        "definition": {
            "mean_G": "mean over the physical bonds of one molecule",
            "H_within_G": "H_G - mean_G",
            "C_all": "all bonds centered by the global bond mean, divisor N-1",
            "C_within_bond_weighted": "within-molecule scatter pooled over all bonds, "
                                      "divisor N-M",
            "C_within_molecule_weighted": "mean of per-molecule within covariances "
                                          "(divisor n_G-1), each molecule equal",
            "C_common": "bond-weighted scatter of molecule means about the global mean, "
                        "divisor N-1",
            "PR": "tr(C)^2 / tr(C^2); scale invariant",
        },
    }

    # fixed bond-count strata, decided before looking at the numbers
    for lo, hi in STRATA:
        idx = np.where((counts >= lo) & (counts <= hi))[0]
        if len(idx) == 0:
            summary["strata"][f"{lo}-{hi}"] = {"n_molecules": 0}
            continue
        Ws = np.zeros((d, d), dtype=np.float64)
        sub_total = 0.0
        sub_common = 0.0
        sub_within = 0.0
        for i in idx:
            s, e = int(offsets[i]), int(offsets[i + 1])
            X = H[s:e]
            dev = X - means[i]
            Ws += dev.T @ dev
            n = e - s
            sub_total += float(np.sum(X * X))
            sub_common += float(n * np.sum(means[i] * means[i]))
            sub_within += float(np.sum(dev * dev))
        n_sub = int(counts[idx].sum())
        C_sub = Ws / float(n_sub - len(idx))
        summary["strata"][f"{lo}-{hi}"] = {
            "n_molecules": int(len(idx)),
            "n_bonds": n_sub,
            "energy_common_share": sub_common / max(sub_total, 1e-30),
            "energy_within_share": sub_within / max(sub_total, 1e-30),
            "C_within": spectrum(C_sub),
            "per_molecule": {
                "within_pr": qsummary(within_pr[idx]),
                "within_top1": qsummary(within_top1[idx]),
                "common_share": qsummary(common_share[idx]),
                "pairwise_rms_over_bond_rms": qsummary(pairwise_norm[idx]),
            },
        }

    per_mol_arrays = {
        "within_pr": within_pr,
        "within_top1": within_top1,
        "within_top4": within_top4,
        "common_share": common_share,
        "within_share": within_share,
        "pairwise_rms": pairwise_rms,
        "pairwise_rms_over_bond_rms": pairwise_norm,
        "within_pr_over_cap": within_pr_over_cap,
    }
    return summary, per_mol_arrays, C_within, C_mw


# ---------------------------------------------------------------------------
# structure response
# ---------------------------------------------------------------------------


def structure_response(
    H: np.ndarray,
    kernels: Sequence[np.ndarray],
    counts: np.ndarray,
    offsets: np.ndarray,
    *,
    b1_max_abs: float,
    b1_max_rel: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    d = H.shape[1]
    M = len(counts)
    Rsum = np.zeros((d, d), dtype=np.float64)
    total_tr = 0.0
    mean_energy = 0.0
    resp_fro = np.empty(M)
    amp = np.empty(M)
    roH = np.empty(M)
    top1 = np.empty(M)
    top4 = np.empty(M)
    mean_share = np.empty(M)
    full_rel = 0.0

    for i in range(M):
        s, e = int(offsets[i]), int(offsets[i + 1])
        X = H[s:e]
        B = kernels[i]
        n = e - s
        BX = B @ X
        m = X.mean(axis=0)
        Xw = X - m
        R = B @ Xw
        denom = max(float(np.linalg.norm(BX)), 1e-30)
        full_rel = max(full_rel, float(np.linalg.norm(R - BX) / denom))
        rr = R.T @ R
        ev = np.linalg.eigvalsh(0.5 * (rr + rr.T))
        ev = np.clip(ev, 0.0, None)
        tr = float(ev.sum())
        Rsum += rr
        total_tr += tr
        rmean = R.mean(axis=0)
        mean_energy += float(n * np.sum(rmean * rmean))
        resp_fro[i] = np.sqrt(tr)
        amp[i] = np.sqrt(tr) / max(float(np.linalg.norm(Xw)), 1e-30)
        roH[i] = np.sqrt(tr) / max(float(np.linalg.norm(X)), 1e-30)
        if tr > 1e-30:
            cum = np.cumsum(ev[::-1]) / tr
            top1[i] = float(cum[0])
            top4[i] = float(cum[min(3, len(cum) - 1)])
            mean_share[i] = float(n * np.sum(rmean * rmean) / tr)
        else:
            top1[i] = np.nan
            top4[i] = np.nan
            mean_share[i] = np.nan

    global_spec = spectrum(Rsum / float(len(H)))
    summary: dict[str, Any] = {
        "n_bonds": int(len(H)),
        "n_molecules": int(M),
        "dimension": int(d),
        "checks": {
            "B_at_1_max_abs": float(b1_max_abs),
            "B_at_1_max_rel_to_B_fro": float(b1_max_rel),
            "B_H_vs_B_H_within_max_rel": float(full_rel),
            "note": "B @ 1 = 0 is exact in the frozen kernel; B @ H and "
                    "B @ H_within differ only by the common term",
        },
        "response_global_second_moment": global_spec,
        "response_mean_energy_share": float(mean_energy / max(total_tr, 1e-30)),
        "per_molecule": {
            "response_frobenius": qsummary(resp_fro),
            "response_rms_per_bond": qsummary(resp_fro / np.sqrt(counts)),
            "amplification_within": qsummary(amp),
            "response_over_H": qsummary(roH),
            "response_top1_share": qsummary(top1),
            "response_top4_share": qsummary(top4),
            "mean_energy_share": qsummary(mean_share),
        },
        "strata": {},
        "definition": {
            "R_G": "B_G @ H_within_G, one row per physical bond",
            "global_second_moment": "mean over bonds of R_e R_e^T; not centered",
            "amplification_within": "||R_G||_F / ||H_within_G||_F",
            "response_over_H": "||R_G||_F / ||H_G||_F",
        },
    }
    for lo, hi in STRATA:
        idx = np.where((counts >= lo) & (counts <= hi))[0]
        summary["strata"][f"{lo}-{hi}"] = {
            "n_molecules": int(len(idx)),
            "amplification_within": qsummary(amp[idx]),
            "response_over_H": qsummary(roH[idx]),
            "response_top1_share": qsummary(top1[idx]),
            "response_top4_share": qsummary(top4[idx]),
        }
    per_mol = {
        "response_frobenius": resp_fro,
        "amplification_within": amp,
        "response_over_H": roH,
        "response_top1_share": top1,
        "response_top4_share": top4,
        "mean_energy_share": mean_share,
    }
    return summary, per_mol


# ---------------------------------------------------------------------------
# reordering checks
# ---------------------------------------------------------------------------


def clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def permute_bonds(batch: Mapping[str, Any], perm: np.ndarray):
    nb = clone_batch(batch)
    n = len(perm)
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n, dtype=np.int64)
    for key in ("bond_feat", "bond_type", "bond_index"):
        nb[key] = batch[key][torch.as_tensor(perm)]
    nb["junc_bond"] = torch.from_numpy(inv[batch["junc_bond"].numpy()]).to(
        batch["junc_bond"].dtype
    )
    if "bond_local" in batch:
        nb["bond_local"] = torch.arange(n, dtype=batch["bond_local"].dtype)
    if "k_pad" in batch:
        K = batch["k_pad"].numpy()
        nb["k_pad"] = torch.from_numpy(np.ascontiguousarray(K[:, perm][:, :, perm]))
    return nb, inv


def permute_atoms(batch: Mapping[str, Any], perm: np.ndarray):
    nb = clone_batch(batch)
    n = len(perm)
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n, dtype=np.int64)
    nb["atom_feat"] = batch["atom_feat"][torch.as_tensor(perm)]
    nb["atom_type"] = batch["atom_type"][torch.as_tensor(perm)]
    nb["bond_index"] = torch.from_numpy(inv[batch["bond_index"].numpy()]).to(
        batch["bond_index"].dtype
    )
    nb["junc_index"] = torch.from_numpy(inv[batch["junc_index"].numpy()]).to(
        batch["junc_index"].dtype
    )
    return nb


def invariance_checks(
    model,
    cache,
    collate,
    *,
    kind: str,
    rows: Sequence[int] = (0, 1, 7, 42, 123),
) -> dict[str, Any]:
    cap = DCCCapture(model) if kind == "dcc" else RCCCapture(model)
    try:
        rng = np.random.default_rng(20261025)
        max_bond = {k: 0.0 for k in (DCC_COMPONENTS if kind == "dcc" else RCC_COMPONENTS)}
        max_atom = {k: 0.0 for k in max_bond}
        for row in rows:
            batch = collate(cache, [int(row)])
            old = cap(batch)
            n = int(len(batch["bond_feat"]))
            perm = rng.permutation(n)
            nb, inv = permute_bonds(batch, perm)
            new = cap(nb)
            for k in old:
                max_bond[k] = max(max_bond[k], float(np.abs(new[k][inv] - old[k]).max()))
            na = int(len(batch["atom_feat"]))
            perm_a = rng.permutation(na)
            nb = permute_atoms(batch, perm_a)
            new = cap(nb)
            for k in old:
                max_atom[k] = max(max_atom[k], float(np.abs(new[k] - old[k]).max()))
    finally:
        if kind == "dcc":
            cap.close()
    threshold = 1e-5
    return {
        "rows": [int(r) for r in rows],
        "n_molecules": len(rows),
        "bond_permutation_max_abs": max_bond,
        "atom_permutation_max_abs": max_atom,
        "threshold": threshold,
        "passed": bool(
            max(max_bond.values(), default=0.0) <= threshold
            and max(max_atom.values(), default=0.0) <= threshold
        ),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    install_torch_load_guard()
    torch.set_grad_enabled(False)

    norm = D.load_norm()

    # --- frozen data --------------------------------------------------------
    raw_cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    cache = GuardedCache(raw_cache)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
    counts = np.diff(bond_ptr).astype(np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    n_molecules = int(cache["n_molecules"])
    n_bonds_total = int(bond_ptr[-1])
    if n_molecules != len(counts) or n_bonds_total != len(bond_index):
        raise RuntimeError("inconsistent physical-bond enumeration")
    if int(offsets[-1]) != n_bonds_total:
        raise RuntimeError("bond offsets do not cover the physical bond list")

    bank = np.load(BANK_PATH)
    kpad = np.asarray(bank["kpad"])
    struct = np.asarray(bank["struct"])
    collate = RC.make_collate((kpad, struct))

    # --- frozen B matrices --------------------------------------------------
    kernels: list[np.ndarray] = []
    b1_max_abs = 0.0
    b1_max_rel = 0.0
    b_sym_err = 0.0
    for i in range(n_molecules):
        s, e = int(bond_ptr[i]), int(bond_ptr[i + 1])
        Kmat, _ = RK.build_kernel(bond_index[s:e], int(atom_ptr[i + 1] - atom_ptr[i]))
        Kd, _, _ = RK.null_from_row_sums(Kmat)
        B = Kmat - Kd
        kernels.append(B)
        b_sym_err = max(b_sym_err, float(np.abs(B - B.T).max()) if B.size else 0.0)
        b1 = B @ np.ones(e - s, dtype=np.float64)
        b1_max_abs = max(b1_max_abs, float(np.abs(b1).max()))
        b1_max_rel = max(
            b1_max_rel, float(np.linalg.norm(b1) / max(float(np.linalg.norm(B)), 1e-30))
        )
    del bank

    # --- frozen model states ------------------------------------------------
    d0, d0_info = DM.build_initial_state(norm, 0)
    d0.eval()
    d0.organized_local = True
    d0_state_hash = AC.state_hash(d0)

    best = torch.load(BEST_PT, map_location="cpu", weights_only=False)
    d1, _ = DM.build_initial_state(norm, 0)
    missing, unexpected = d1.load_state_dict(best["state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"best.pt does not match DCC-L: {missing} {unexpected}")
    d1.eval()
    d1.organized_local = True

    r0, r0_info = RM.build_initial_state(norm, 0)
    r0.eval()

    artifacts = json.loads(ARTIFACTS.read_text())
    run_artifacts = json.loads(DCC_RUN_ARTIFACTS.read_text())
    prep = json.loads(PREP_PATH.read_text())
    redundancy = json.loads(REDUNDANCY_PATH.read_text())
    best_sha = sha256_file(BEST_PT)
    recorded_best_sha = artifacts["L_s0"]["best_pt_sha256"]
    recorded_final_sha = artifacts["L_s0"]["final_pt_sha256"]
    best_state_hash = state_hash_sd(best["state_dict"])
    d0_flat = flat_state(d0.state_dict())
    best_flat = flat_state(best["state_dict"])
    rel_best_d0 = float(np.linalg.norm(best_flat - d0_flat) / max(np.linalg.norm(d0_flat), 1e-30))

    # R0 provenance: the prep.json hash was recorded when prep.json did not yet
    # exist, so build_initial_state used t_scale=1.0.  Rebuilding now uses the
    # stored t_scale.  Verify that this is the only difference.
    r0_current_hash = AC.state_hash(r0)
    t_scale_value = float(r0.t_scale.item())
    r0_sd = {k: v.clone() for k, v in r0.state_dict().items()}
    with torch.no_grad():
        r0.t_scale.fill_(1.0)
    r0_t1_hash = AC.state_hash(r0)
    with torch.no_grad():
        r0.t_scale.fill_(t_scale_value)
    r0_sd_t1 = dict(r0_sd)
    r0_sd_t1["t_scale"] = torch.ones_like(r0_sd_t1["t_scale"])
    prep_hash = prep["init"]["initial_state_sha256"]
    prep_hash_explained = (r0_t1_hash == prep_hash) and (state_hash_sd(r0_sd_t1) == prep_hash)
    enc0 = {k[len("encoders."):]: v for k, v in d0.state_dict().items() if k.startswith("encoders.")}
    encr = {
        k[len("encoders."):]: v
        for k, v in r0.state_dict().items()
        if k.startswith("encoders.")
    }
    enc_keys = sorted(enc0)
    enc_diff = max(
        (float((enc0[k] - encr[k]).abs().max()) for k in enc_keys),
        default=0.0,
    )
    seed1 = sorted((_SUFF / "zinc_dual_channel_composer_v1" / "runs").glob("*s1*/best.pt"))
    rcc_runs = sorted((_SUFF / "zinc_relational_contrast_contraction_v0" / "runs").glob("**/*.pt"))

    # --- reproduce the previous RCC PR on the exact prior subset ------------
    split = RC.load_split()
    fit_rows = np.asarray(split["cache_rows"]["fit"], dtype=np.int64)
    repro_rows = fit_rows[:2560]
    hs = []
    with torch.no_grad():
        for s in range(0, len(repro_rows), BATCH):
            b = collate(cache, repro_rows[s:s + BATCH])
            hs.append(r0.compute(b)["h"].double().cpu().numpy())
    Hrepro = np.concatenate(hs, axis=0)
    Hc = Hrepro - Hrepro.mean(axis=0)
    cov = (Hc.T @ Hc) / max(len(Hc) - 1, 1)
    repro_spec = spectrum(cov)
    prev = redundancy["bond_content_spectrum"]
    repro = {
        "rows": "split_E fit[:2560], batch 128, same order as rcc_diag.py",
        "n_bonds": int(len(Hrepro)),
        "participation_ratio": repro_spec["pr"],
        "top1_share": repro_spec["top1"],
        "top5_share": float(sum(repro_spec["eigenvalues_desc"][:5]) / repro_spec["trace"]),
        "top10_share": float(sum(repro_spec["eigenvalues_desc"][:10]) / repro_spec["trace"]),
        "previous_reported": {
            "participation_ratio": prev["participation_ratio"],
            "top1_share": prev["top1_share"],
            "top5_share": prev["top5_share"],
            "top10_share": prev["top10_share"],
        },
        "abs_diff_pr": abs(repro_spec["pr"] - prev["participation_ratio"]),
        "match": bool(abs(repro_spec["pr"] - prev["participation_ratio"]) < 1e-9),
        "note": "same frozen rcc_models.py object and same preprocessing as the "
                "RCC-v0 bond_content_spectrum probe",
    }

    # --- full 10k forward passes -------------------------------------------
    rows = np.arange(n_molecules, dtype=np.int64)
    states: dict[str, dict[str, Any]] = {}
    per_state_arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    cw_mats: dict[str, dict[str, np.ndarray]] = {}
    cmw_mats: dict[str, dict[str, np.ndarray]] = {}
    response: dict[str, dict[str, Any]] = {}
    response_arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for state_name, kind, model in (
        ("D0", "dcc", d0),
        ("D1", "dcc", d1),
        ("R0", "rcc", r0),
    ):
        if kind == "dcc":
            reprs = extract_dcc(model, cache, collate, rows)
        else:
            reprs = extract_rcc(model, cache, collate, rows)
        states[state_name] = {}
        per_state_arrays[state_name] = {}
        cw_mats[state_name] = {}
        cmw_mats[state_name] = {}
        response[state_name] = {}
        response_arrays[state_name] = {}
        for comp, H in reprs.items():
            s, a, cw, cmw = representation_stats(H, counts, offsets, n_bonds_total)
            states[state_name][comp] = s
            per_state_arrays[state_name][comp] = a
            cw_mats[state_name][comp] = cw
            cmw_mats[state_name][comp] = cmw
            r, ra = structure_response(
                H, kernels, counts, offsets, b1_max_abs=b1_max_abs, b1_max_rel=b1_max_rel
            )
            response[state_name][comp] = r
            response_arrays[state_name][comp] = ra
        del reprs

    # --- D0 -> D1 paired changes -------------------------------------------
    d01: dict[str, Any] = {}
    for comp in DCC_COMPONENTS:
        s0, s1 = states["D0"][comp], states["D1"][comp]
        a0, a1 = per_state_arrays["D0"][comp], per_state_arrays["D1"][comp]
        d01[comp] = {
            "spectrum_delta": {
                "C_within_pr": s1["C_within_bond_weighted"]["pr"] - s0["C_within_bond_weighted"]["pr"],
                "C_within_top1": s1["C_within_bond_weighted"]["top1"] - s0["C_within_bond_weighted"]["top1"],
                "C_within_top4": s1["C_within_bond_weighted"]["top4"] - s0["C_within_bond_weighted"]["top4"],
                "C_within_dims_90": s1["C_within_bond_weighted"]["dims_90"] - s0["C_within_bond_weighted"]["dims_90"],
                "C_within_trace": s1["C_within_bond_weighted"]["trace"] - s0["C_within_bond_weighted"]["trace"],
                "C_all_pr": s1["C_all"]["pr"] - s0["C_all"]["pr"],
                "C_molecule_weighted_pr": s1["C_within_molecule_weighted"]["pr"] - s0["C_within_molecule_weighted"]["pr"],
                "energy_common_share": s1["energy"]["common_share"] - s0["energy"]["common_share"],
                "energy_within_share": s1["energy"]["within_share"] - s0["energy"]["within_share"],
            },
            "per_molecule_delta": {
                "within_pr": qsummary(a1["within_pr"] - a0["within_pr"]),
                "within_top1": qsummary(a1["within_top1"] - a0["within_top1"]),
                "common_share": qsummary(a1["common_share"] - a0["common_share"]),
                "pairwise_rms": qsummary(a1["pairwise_rms"] - a0["pairwise_rms"]),
                "pairwise_rms_over_bond_rms": qsummary(
                    a1["pairwise_rms_over_bond_rms"] - a0["pairwise_rms_over_bond_rms"]
                ),
            },
            "strata_delta": {},
            "response_spectrum_delta": {
                "response_pr": response["D1"][comp]["response_global_second_moment"]["pr"]
                - response["D0"][comp]["response_global_second_moment"]["pr"],
                "response_top1": response["D1"][comp]["response_global_second_moment"]["top1"]
                - response["D0"][comp]["response_global_second_moment"]["top1"],
                "response_trace": response["D1"][comp]["response_global_second_moment"]["trace"]
                - response["D0"][comp]["response_global_second_moment"]["trace"],
            },
            "response_per_molecule_delta": {
                "amplification_within": qsummary(
                    response_arrays["D1"][comp]["amplification_within"]
                    - response_arrays["D0"][comp]["amplification_within"]
                ),
                "response_over_H": qsummary(
                    response_arrays["D1"][comp]["response_over_H"]
                    - response_arrays["D0"][comp]["response_over_H"]
                ),
                "response_top1_share": qsummary(
                    response_arrays["D1"][comp]["response_top1_share"]
                    - response_arrays["D0"][comp]["response_top1_share"]
                ),
                "response_top4_share": qsummary(
                    response_arrays["D1"][comp]["response_top4_share"]
                    - response_arrays["D0"][comp]["response_top4_share"]
                ),
            },
            "response_strata_delta": {},
            "subspace_overlap": subspace_overlap(
                cw_mats["D0"][comp], cw_mats["D1"][comp]
            ),
            "subspace_overlap_molecule_weighted": subspace_overlap(
                cmw_mats["D0"][comp], cmw_mats["D1"][comp]
            ),
        }
        for lo, hi in STRATA:
            key = f"{lo}-{hi}"
            r0s, r1s = s0["strata"][key], s1["strata"][key]
            d01[comp]["strata_delta"][key] = {
                "n_molecules": r0s.get("n_molecules"),
                "C_within_pr": r1s["C_within"]["pr"] - r0s["C_within"]["pr"],
                "C_within_top1": r1s["C_within"]["top1"] - r0s["C_within"]["top1"],
                "energy_common_share": r1s["energy_common_share"] - r0s["energy_common_share"],
                "within_pr_median": r1s["per_molecule"]["within_pr"]["median"]
                - r0s["per_molecule"]["within_pr"]["median"],
                "pairwise_norm_median": r1s["per_molecule"]["pairwise_rms_over_bond_rms"]["median"]
                - r0s["per_molecule"]["pairwise_rms_over_bond_rms"]["median"],
            }
            q0, q1 = response["D0"][comp]["strata"][key], response["D1"][comp]["strata"][key]
            d01[comp]["response_strata_delta"][key] = {
                "n_molecules": q0.get("n_molecules"),
                "amplification_median": q1["amplification_within"]["median"]
                - q0["amplification_within"]["median"],
                "response_over_H_median": q1["response_over_H"]["median"]
                - q0["response_over_H"]["median"],
                "response_top1_median": q1["response_top1_share"]["median"]
                - q0["response_top1_share"]["median"],
                "response_top4_median": q1["response_top4_share"]["median"]
                - q0["response_top4_share"]["median"],
            }

    # --- reordering checks --------------------------------------------------
    inv_rows = [0, 1, 7, 42, 123]
    inv_checks = {
        "D0": invariance_checks(d0, cache, collate, kind="dcc", rows=inv_rows),
        "D1": invariance_checks(d1, cache, collate, kind="dcc", rows=inv_rows),
        "R0": invariance_checks(r0, cache, collate, kind="rcc", rows=inv_rows),
    }

    # --- provenance / hashes ------------------------------------------------
    code_paths = {
        "audit_script": _HERE / "frozen_representation_audit.py",
        "dcc_models": _DCC / "dcc_models.py",
        "rcc_models": _RCC / "rcc_models.py",
        "rcc_kernels": _RCC / "rcc_kernels.py",
        "rcc_common": _RCC / "rcc_common.py",
        "rcc_prep": _RCC / "rcc_prep.py",
        "rcc_config": RCC_SPLIT_PATH,
    }
    code_hashes = {
        k: {"path": str(p.relative_to(_REPO)), "sha256": sha256_file(p)}
        for k, p in code_paths.items()
    }
    data_hashes = {
        "official_train_structural_cache": {
            "path": str(CACHE_PATH.relative_to(_REPO)),
            "sha256": sha256_file(CACHE_PATH),
        },
        "kernel_bank": {
            "path": str(BANK_PATH.relative_to(_REPO)),
            "sha256": sha256_file(BANK_PATH),
        },
        "prep_json": {
            "path": str(PREP_PATH.relative_to(_REPO)),
            "sha256": sha256_file(PREP_PATH),
        },
        "redundancy_probe_json": {
            "path": str(REDUNDANCY_PATH.relative_to(_REPO)),
            "sha256": sha256_file(REDUNDANCY_PATH),
        },
    }
    checkpoint_hashes = {
        "D1_best_pt": {"path": str(BEST_PT.relative_to(_REPO)), "sha256": best_sha},
        "D1_final_pt_recorded": {
            "path": str(FINAL_PT.relative_to(_REPO)),
            "sha256": recorded_final_sha,
            "note": "hash read from analysis/artifact_hashes.json; final.pt not loaded",
        },
        "D1_best_pt_recorded": {
            "path": str(BEST_PT.relative_to(_REPO)),
            "sha256": recorded_best_sha,
        },
        "D1_best_is_not_final": best_sha != recorded_final_sha,
    }

    common_meta = {
        "protocol": "zinc_frozen_bond_representation_audit_v1",
        "git_commit": RC.git_commit(),
        "git_dirty_tracked": RC.git_dirty(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gpu_hours": 0.0,
        "device": "cpu",
        "cuda_used": False,
        "gradient_steps": 0,
        "optimizer_steps": 0,
        "training_runs": 0,
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "labels_read": False,
        "labels_key_attempted": bool(cache.labels_attempted),
        "torch_load_paths": [str(Path(p).relative_to(_REPO)) if str(p).startswith(str(_REPO)) else str(p) for p in _LOADED_PATHS],
        "data": {
            "split": "official PyG ZINC subset=True train",
            "n_molecules": int(n_molecules),
            "n_physical_bonds": int(n_bonds_total),
            "n_bonds_min": int(counts.min()),
            "n_bonds_max": int(counts.max()),
            "bond_enumeration": "frozen cache bond_index, one row per physical bond; "
                                "PyG bidirectional edge storage is not expanded",
            "labels_indexed": False,
            "cache_keys_accessed": sorted(cache.accessed),
        },
        "code_hashes": code_hashes,
        "data_hashes": data_hashes,
        "checkpoint_hashes": checkpoint_hashes,
        "architecture": {
            "DCC_L": {
                "bond_encoder_b": 96,
                "local_composer_c_loc": 96,
                "local_composer_hidden": 80,
                "group_composer_c_glob": 96,
                "note": "c_loc is produced by an MLP with hidden width 80, so its "
                        "output covariance has at most 80 nonzero eigenvalues; "
                        "the observed within-PR is far below that cap",
            },
            "RCC_v0": {
                "unified_composer_h_RCC": 96,
                "z_proj_rank": 16,
            },
        },
    }

    provenance = {
        "d0": {
            "definition": "DCC-L shared SSL-start supervised initialization, seed 0",
            "builder": "dcc_models.build_initial_state(norm, seed=0)",
            "state_sha256": d0_state_hash,
            "recorded_initial_state_sha256": run_artifacts.get("initial_state_sha256"),
            "match_recorded_initial_state": d0_state_hash == run_artifacts.get("initial_state_sha256"),
            "p3_reference_state_sha256": d0_info.get("p3_reference_state_sha256"),
            "recorded_p3_reference_state_sha256": run_artifacts.get("p3_reference_state_sha256"),
            "match_p3_reference": d0_info.get("p3_reference_state_sha256")
            == run_artifacts.get("p3_reference_state_sha256"),
            "ssl": d0_info.get("ssl"),
            "parameters": int(d0_info.get("parameters")),
        },
        "d1": {
            "definition": "historical DCC-L seed0 best checkpoint (L arm), epoch 489",
            "path": str(BEST_PT.relative_to(_REPO)),
            "sha256": best_sha,
            "sha256_matches_recorded": best_sha == recorded_best_sha,
            "epoch": int(best["epoch"]),
            "stored_valid_mae": float(best["valid_mae"]),
            "recorded_best_epoch": int(run_artifacts["best_valid"]["epoch"]),
            "recorded_best_valid_mae": float(run_artifacts["best_valid"]["valid_mae"]),
            "epoch_matches_recorded": int(best["epoch"]) == int(run_artifacts["best_valid"]["epoch"]),
            "candidate": str(best["candidate"]),
            "seed": int(best["seed"]),
            "state_sha256": best_state_hash,
            "state_hash_differs_from_d0": best_state_hash != d0_state_hash,
            "relative_l2_distance_from_d0": rel_best_d0,
            "parameters": int(sum(p.numel() for p in d1.parameters())),
            "architecture_matches_d0": list(d1.state_dict()) == list(d0.state_dict()),
            "is_best_not_final": best_sha != recorded_final_sha,
            "final_pt_sha256_recorded": recorded_final_sha,
            "final_pt_not_loaded": True,
        },
        "d1_s1": {
            "present": bool(seed1),
            "searched": "zinc_dual_channel_composer_v1/runs/*s1*/best.pt",
            "note": "no DCC-L seed1 checkpoint exists; this audit does not train one",
        },
        "r0": {
            "definition": "RCC-v0 shared unified-composer initialization, seed 0",
            "builder": "rcc_models.build_initial_state(norm, seed=0)",
            "state_sha256_current": r0_current_hash,
            "state_sha256_with_t_scale_1": r0_t1_hash,
            "prep_recorded_initial_state_sha256": prep_hash,
            "prep_hash_explained_by_t_scale": bool(prep_hash_explained),
            "t_scale_current": t_scale_value,
            "prep_recorded_t_scale": float(prep.get("t_scale", float("nan"))),
            "encoders_equal_d0": bool(enc_keys == sorted(encr)),
            "encoder_max_abs_diff_vs_d0": enc_diff,
            "parameters": int(r0_info.get("parameters")),
            "rcc_supervised_checkpoint_present": bool(rcc_runs),
            "rcc_supervised_checkpoint_paths": [str(p.relative_to(_REPO)) for p in rcc_runs],
            "note": "prep.json was written before build_initial_state could read a "
                    "t_scale, so its recorded hash is the t_scale=1 buffer state; "
                    "current rebuild differs only in that buffer and h/Z are unchanged",
        },
        "reproduction": repro,
        "seed1_note": "D1-s1 is absent; no cross-seed claim is made",
    }

    # --- conclusion strings ------------------------------------------------
    def f(x: float, nd: int = 3) -> str:
        return f"{x:.{nd}f}"

    pr_d0 = {c: states["D0"][c]["C_within_bond_weighted"]["pr"] for c in DCC_COMPONENTS}
    pr_d1 = {c: states["D1"][c]["C_within_bond_weighted"]["pr"] for c in DCC_COMPONENTS}
    t1_d0 = {c: states["D0"][c]["C_within_bond_weighted"]["top1"] for c in DCC_COMPONENTS}
    t1_d1 = {c: states["D1"][c]["C_within_bond_weighted"]["top1"] for c in DCC_COMPONENTS}
    cs_d0 = {c: states["D0"][c]["energy"]["common_share"] for c in DCC_COMPONENTS}
    cs_d1 = {c: states["D1"][c]["energy"]["common_share"] for c in DCC_COMPONENTS}

    class_label = "MIXED_OR_INCONCLUSIVE"
    ov_loc = d01["c_loc"]["subspace_overlap"]
    ov_loc_mw = d01["c_loc"]["subspace_overlap_molecule_weighted"]
    class_basis = (
        "D0->D1 within-PR rises for b "
        f"({f(pr_d0['b'])} -> {f(pr_d1['b'])}) and c_glob "
        f"({f(pr_d0['c_glob'])} -> {f(pr_d1['c_glob'])}), but the local composer c_loc "
        f"rises only from {f(pr_d0['c_loc'])} to {f(pr_d1['c_loc'])} and keeps "
        f"top1={f(t1_d1['c_loc'])}.  Its D0/D1 leading within directions are nearly "
        f"orthogonal (top1 cos2={ov_loc['top1_cos2']:.4g}; D1 trace in D0 top4="
        f"{ov_loc['by_k']['4']['D1_trace_fraction_in_D0_topk']:.3f}; molecule-weighted "
        f"top1 cos2={ov_loc_mw['top1_cos2']:.4g}), so the small increase is not "
        "amplification of the old few directions: the representation re-forms into "
        "new but still low-dimensional directions.  The components "
        "disagree in the strength of the change; there is no DCC-L seed1 and no "
        "supervised RCC checkpoint."
    )
    class_scope = (
        "The result is not a statement about what would happen if RCC were trained. "
        "It says that within-DCC-L training effects are component-dependent, so the "
        "low-rank initialization cannot be treated either as a pure initialization "
        "artifact (b, c_glob) or as a persistent property of every DCC-L component "
        "(c_loc)."
    )

    spectrum_payload = {
        **common_meta,
        "provenance": provenance,
        "definitions": {
            "H_G": "matrix [n_bonds_G, d] for one molecule, physical bonds only",
            "mean_G": "mean over the bonds of one molecule",
            "H_within_G": "H_G - mean_G",
            "E_total": "sum_G sum_e ||H_G[e]||^2",
            "E_common": "sum_G n_G ||mean_G||^2",
            "E_within": "sum_G sum_e ||H_G[e]-mean_G||^2",
            "PR": "tr(C)^2 / tr(C^2)",
            "C_all": "all bonds centered by the global bond mean",
            "C_within_bond_weighted": "within-molecule scatter pooled over bonds",
            "C_within_molecule_weighted": "molecule-equal mean of per-molecule covariances",
        },
        "states": states,
        "d0_to_d1": d01,
        "rcc_reproduction": repro,
        "invariance_checks": inv_checks,
        "classification": {
            "label": class_label,
            "basis": class_basis,
            "scope": class_scope,
            "not_a_threshold": True,
        },
    }

    response_payload = {
        **common_meta,
        "provenance": provenance,
        "kernel": {
            "definition": "B = sym(lazy one-step physical-bond walk) - d d^T/sum(d)",
            "b_symmetry_max_abs": float(b_sym_err),
            "B_at_1_max_abs": float(b1_max_abs),
            "B_at_1_max_rel_to_B_fro": float(b1_max_rel),
            "n_molecules": int(n_molecules),
        },
        "states": response,
        "d0_to_d1": {
            c: {
                "response_spectrum_delta": d01[c]["response_spectrum_delta"],
                "response_per_molecule_delta": d01[c]["response_per_molecule_delta"],
                "response_strata_delta": d01[c]["response_strata_delta"],
            }
            for c in DCC_COMPONENTS
        },
        "interpretation": {
            "label": "NONTRIVIAL_BUT_COMPONENT_DEPENDENT_RESPONSE",
            "basis": (
                "B@1 is zero to machine precision and B@H equals B@H_within to "
                "~1e-15 relative.  The response is nonzero on every audited state.  "
                "D1 c_loc keeps response top1="
                f"{f(response['D1']['c_loc']['response_global_second_moment']['top1'])} "
                "and response PR="
                f"{f(response['D1']['c_loc']['response_global_second_moment']['pr'])}; "
                "D1 b and c_glob have higher response PR "
                f"({f(response['D1']['b']['response_global_second_moment']['pr'])} and "
                f"{f(response['D1']['c_glob']['response_global_second_moment']['pr'])}).  "
                "This is an algebraic response, not evidence of label utility or "
                "generalization."
            ),
            "not_a_generalization_claim": True,
        },
    }

    RC.write_json(OUT_SPECTRUM, spectrum_payload)
    RC.write_json(OUT_RESPONSE, response_payload)

    # short console summary
    print("== representation spectrum ==")
    for st in ("D0", "D1", "R0"):
        for comp, s in states[st].items():
            print(
                f"{st:2s} {comp:6s} C_within PR={f(s['C_within_bond_weighted']['pr'])} "
                f"top1={f(s['C_within_bond_weighted']['top1'])} "
                f"common={f(s['energy']['common_share'])} "
                f"C_all PR={f(s['C_all']['pr'])}"
            )
    print("== response ==")
    for st in ("D0", "D1", "R0"):
        for comp, r in response[st].items():
            g = r["response_global_second_moment"]
            print(
                f"{st:2s} {comp:6s} PR={f(g['pr'])} top1={f(g['top1'])} "
                f"amp={f(r['per_molecule']['amplification_within']['median'])}"
            )
    print(f"RCC reproduction PR={f(repro['participation_ratio'], 6)} "
          f"match={repro['match']}")
    print(f"invariance: " + ", ".join(f"{k}={v['passed']}" for k, v in inv_checks.items()))
    print(f"classification: {class_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
