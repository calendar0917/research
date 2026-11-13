"""Mechanism panels: what the M / S / A switches do inside the frozen C1 model.

Read-only.  One split of one small dataset is materialised at a time; the large
(ogbg-molhiv) cache is never touched here.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

import lcpr_common as C
import lcpr_data as D


def _fixed_rows(n_rows: int, k: int, salt: int) -> list[int]:
    """Deterministic row choice by hashed order (no RNG, no label use)."""
    h = (np.arange(n_rows, dtype=np.int64) * 2654435761 + int(salt)) % (2**31 - 1)
    return np.sort(np.argsort(h, kind="stable")[: int(k)]).tolist()


def _joint_slice(model) -> slice:
    """Column slice of the joint (J) block inside ``path_agg``."""
    layout = model.local._layout()          # noqa: SLF001 - frozen layout
    rank = int(model.local.rank)
    pos = 0
    for kind in layout:
        dim = 8 if kind == "closure" else rank
        if kind == "joint":
            return slice(pos, pos + dim)
        pos += dim
    return slice(0, 0)


class _eval_mode:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.was_training = self.model.training
        self.model.eval()
        return self

    def __exit__(self, *exc):
        self.model.train(self.was_training)


@torch.no_grad()
def panel_m(model, dataset: str = "Lipophilicity", *, masks: int = 256,
            k_train: int = 128, k_valid: int = 64, k_test: int = 64) -> dict[str, Any]:
    """Dropout panel: train-time moments/predictions versus the eval forward.

    For each fixed row set, ``masks`` independent dropout masks are applied to
    the *clean* local representation ``h``; the panel records the mean/var of the
    resulting predictions and of the two pooling statistics used by ``rho``
    (``mean h`` and ``mean h²``), i.e. exactly the M-family definitions.
    """
    p = float(model.dropout_p)
    keep = 1.0 - p
    dev = next(model.parameters()).device
    out: dict[str, Any] = {"masks": int(masks), "dropout_p": p, "splits": {}}
    for split, k in (("train", k_train), ("valid", k_valid), ("test", k_test)):
        cache = D.load_cache(dataset, split)
        rows = _fixed_rows(int(cache["n_molecules"]), k, salt=7)
        import lcpr_analysis as A

        y_true_all, _ = A.split_labels(dataset, split)
        batch = D.to_device(D.collate(cache, rows), dev)
        with _eval_mode(model):
            ev = model(batch, return_aux=True)
            h = ev["h_clean"]
            acc = {key: torch.zeros((len(rows),), dtype=h.dtype, device=dev)
                   for key in ("y", "y2", "mh", "mh2")}
            mh_clean = ev["mean_h"]
            mh2_clean = ev["mean_h2"]
            for _ in range(int(masks)):
                mask = (torch.rand_like(h) > p).to(h.dtype)
                h_m = h * mask / keep
                rec = model(batch, return_aux=True, h_override=h_m)
                acc["y"] += rec["y"]
                acc["y2"] += rec["y"] ** 2
                acc["mh"] += rec["mean_h"].mean(dim=1)
                acc["mh2"] += rec["mean_h2"].mean(dim=1)
            mc_mean = acc["y"] / masks
            mc_var = (acc["y2"] / masks - mc_mean ** 2).clamp(min=0)
            mh_m = acc["mh"] / masks
            mh2_m = acc["mh2"] / masks
        mh2_c = mh2_clean.mean(dim=1)
        out["splits"][split] = {
            "rows": [int(r) for r in rows],
            "y_true": np.asarray(y_true_all, dtype=float)[rows].tolist(),
            "pred_eval": ev["y"].cpu().numpy().tolist(),
            "pred_mc_mean": mc_mean.cpu().numpy().tolist(),
            "pred_mc_std": torch.sqrt(mc_var).cpu().numpy().tolist(),
            "mean_h_clean": mh_clean.mean(dim=1).cpu().numpy().tolist(),
            "mean_h2_clean": mh2_c.cpu().numpy().tolist(),
            "mean_h_mask": mh_m.cpu().numpy().tolist(),
            "mean_h2_mask": mh2_m.cpu().numpy().tolist(),
            "summary": {
                #: pooled ratios use totals per split, which is the quantity the
                #: pooling statistics actually aggregate (per-row ratios divide by
                #: near-zero means and are uninformative)
                "mean_h2_ratio_mask_over_clean": float(mh2_m.sum() / mh2_c.sum().clamp(min=1e-12)),
                "mean_h_ratio_mask_over_clean": float(mh_m.sum() / mh_clean.mean(dim=1).sum().clamp(min=1e-12)),
                "pred_mc_std_mean": float(torch.sqrt(mc_var).mean()),
                "pred_mc_mean_minus_eval_mean": float((mc_mean - ev["y"]).mean()),
                "keep_prob": keep,
            },
        }
    return out


@torch.no_grad()
def panel_s(model, dataset: str = "Lipophilicity", *, k_valid: int = 64, k_test: int = 64,
            kappa_mults: Sequence[float] = (1e6, 4.0, 1.0, 0.1, 0.001)) -> dict[str, Any]:
    """Support panel: per-row sensitivity of the J block to the support rule.

    The contraction strength is swept through ``kappa_eff = kappa * mult``:
    ``mult = 1`` is the frozen rule the arm was trained with, ``mult -> inf`` is
    the frozen marginal operator (B5, ``a -> 0``) and ``mult -> 0`` approaches the
    FULL operator (``a -> 1``).  Every quantity is reported as a change against
    ``mult = 1``, i.e. against the operator the checkpoint actually uses.
    """
    import lcpr_analysis as A

    support = D.load_support(dataset)
    base_kappa = float(support["kappa"])
    dev = next(model.parameters()).device
    jslice = _joint_slice(model)
    out: dict[str, Any] = {"kappa_mults": [float(m) for m in kappa_mults], "kappa": base_kappa,
                           "a_bar_k1": float(support["a_bar_k1"]),
                           "a_bar_k4": float(support["a_bar_k4"]), "splits": {}}
    s2_all = A.s2_support_by_molecule(dataset)
    for split, k in (("valid", k_valid), ("test", k_test)):
        cache = D.load_cache(dataset, split)
        rows = _fixed_rows(int(cache["n_molecules"]), k, salt=11)
        batch = D.to_device(D.collate(cache, rows, path_count=support["path_count"][split]), dev)
        y_true = np.asarray(A.split_labels(dataset, split)[0], dtype=float)[rows]
        ref = None
        per: dict[str, Any] = {}
        with _eval_mode(model):
            for mult in kappa_mults:
                kappa_eff = float(base_kappa * float(mult))
                model.local.shrink_kappa.fill_(min(kappa_eff, 1e12))
                model.local.shrink_abar.fill_(float(support["a_bar_k1"]))
                rec = model(batch, return_aux=True)
                joint = rec["path_agg"][:, jslice]
                per[f"{mult:g}"] = {"pred": rec["y"].cpu().numpy(), "joint": joint.cpu().numpy()}
                if float(mult) == 1.0:
                    ref = per[f"{mult:g}"]
        if ref is None:
            raise ValueError("kappa_mults must include 1.0 (the frozen rule)")
        s2 = s2_all[split][np.asarray(rows)]
        recs = []
        for i in range(len(rows)):
            d_j = float(np.linalg.norm(per["1e+06"]["joint"][i] - ref["joint"][i]))
            d_j_full = float(np.linalg.norm(per["0.001"]["joint"][i] - ref["joint"][i]))
            d_p = float(per["1e+06"]["pred"][i] - ref["pred"][i])
            recs.append({
                "row": int(rows[i]),
                "s2_support_mean": float(s2[i, 0]) if np.isfinite(s2[i, 0]) else None,
                "s2_support_min": float(s2[i, 1]) if np.isfinite(s2[i, 1]) else None,
                "delta_j_norm": d_j,
                "delta_j_norm_toward_full": d_j_full,
                "delta_pred": d_p,
                "delta_se": float((per["1e+06"]["pred"][i] - y_true[i]) ** 2
                                  - (ref["pred"][i] - y_true[i]) ** 2),
                "delta_se_toward_full": float((per["0.001"]["pred"][i] - y_true[i]) ** 2
                                              - (ref["pred"][i] - y_true[i]) ** 2),
                "delta_pred_marginal": d_p,
                "abs_error_rule": float(abs(ref["pred"][i] - y_true[i])),
            })
        js = [r["delta_j_norm"] for r in recs]
        out["splits"][split] = {
            "rows": recs,
            "summary": {
                "delta_j_norm_mean": float(np.mean(js)),
                "delta_j_norm_toward_full_mean": float(np.mean([r["delta_j_norm_toward_full"] for r in recs])),
                "delta_sse_toward_full": float(np.sum([r["delta_se_toward_full"] for r in recs])),
                "delta_j_norm_zero_frac": float(np.mean(np.asarray(js) == 0.0)),
                "delta_sse_alpha0_minus_1": float(np.sum([r["delta_se"] for r in recs])),
                "delta_pred_abs_mean": float(np.mean([abs(r["delta_pred"]) for r in recs])),
                "rows_with_support": int(sum(1 for r in recs if r["s2_support_mean"] is not None)),
            },
        }
    return out


def panel_a(train_payload: Mapping[str, Any]) -> dict[str, Any]:
    """A-family panel from recorded gradient diagnostics (no extra compute)."""
    diag = train_payload.get("diagnostics") or {}
    steps = diag.get("grad_steps") or diag.get("gradients") or []
    return {"steps": steps, "aux_r2": train_payload.get("aux_r2") or diag.get("aux_r2"),
            "n_steps": len(steps)}
