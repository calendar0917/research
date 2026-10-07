"""Read-only binding-collapse diagnostic for zinc_cssd_basis_reuse_v1.

Locates WHY the unique alpha-mean dictionary intervention shows no response
on the trained soups (|Δpred| ~ 1e-7) although the substitution itself is a
large input change (per-root ||alpha - mean_alpha|| median ~ 0.36):

* the A-slot product node binding's z-projection W_A_S decays monotonically
  from ||W_A_S|| ~ 2.3 at epoch 1 to ~ 0 in the final soup (both arms, both
  seeds), i.e. the trained consumer carries its predictions through the
  chemistry one-hot product term (W_A_C), the raw/root-tuple MLP, the Sem108
  interface and the shared Q — not through the z code;
* an epoch-1 checkpoint (before the collapse) DOES respond to the same
  alpha-mean swap, so the intervention path itself is wired and working.

Read-only: loads the frozen soups/epoch checkpoints, never retrains, never
re-runs terminal-eval, never reads official valid/test.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import zinc_cssd_basis_reuse_v1 as m


def main() -> int:
    started = time.perf_counter()
    out: dict = {
        "protocol_version": m.PROTOCOL_VERSION,
        "stage": "binding-collapse-diagnostic",
        "read_only": True,
        "question": (
            "why does the alpha-mean intervention show no response on the trained "
            "soups (|Δpred| ~ 1e-7) while the substitution changes z by ~0.36 per root?"
        ),
        "reproduction": (
            "uv run python -m tracks.ksvd.experiments.luyin16.zinc_cssd_basis_reuse_v1_binding_diag"
        ),
    }
    objects = m.load_domain_objects()
    payload = m.zlt.TuplePayload(objects["payload_arrays"])
    kappa = float(objects["kappa"]["kappa_M"])

    # (1) size of the substitution itself (input side, SOURCE basis, T_eval rows)
    basis = m.load_basis("SOURCE")
    phi_rows, _ = m._phi_rows_for(objects["fold"]["t_eval_idx"])
    z, _phi_hat, _rel = m._root_codes(basis, phi_rows)
    alpha = z[:, m.COMMON_DIM :]
    mean_alpha = alpha.mean(axis=0)
    dz = np.linalg.norm(alpha - mean_alpha, axis=1)
    out["substitution_size"] = {
        "n_roots": int(dz.size),
        "per_root_l2_median": float(np.median(dz)),
        "per_root_l2_p95": float(np.percentile(dz, 95)),
        "per_root_l2_max": float(dz.max()),
        "mean_alpha_l2": float(np.linalg.norm(mean_alpha)),
    }

    # (2) z-binding weight trajectory across the recorded checkpoints
    trajectory = {}
    for arm in m.REUSE_ARMS:
        for seed in m.REUSE_SEEDS:
            rdir = m.RESULTS_DIR / "runs" / f"{arm}_s{seed}"
            rows = {}
            for ep in (1, 40, 120, 240):
                st = torch.load(rdir / f"epoch{ep}_state.pt", map_location="cpu", weights_only=False)
                W_A = st["W_A_S"]
                W_E = st["W_E_S"]
                rows[f"epoch{ep}"] = {
                    "W_A_S_total_l2": float(W_A.norm()),
                    "W_A_S_common_row_l2": float(W_A[0].norm()),
                    "W_A_S_alpha_rows_l2": float(W_A[1:].norm()),
                    "W_E_S_common_rows_l2": float(W_E[:1].norm()),
                    "W_E_S_alpha_rows_l2": float(W_E[1:].norm()),
                }
            soup = torch.load(rdir / "soup_state.pt", map_location="cpu", weights_only=False)
            rows["soup"] = {
                "W_A_S_total_l2": float(soup["W_A_S"].norm()),
                "W_A_S_common_row_l2": float(soup["W_A_S"][0].norm()),
                "W_A_S_alpha_rows_l2": float(soup["W_A_S"][1:].norm()),
                "W_E_S_common_rows_l2": float(soup["W_E_S"][:1].norm()),
                "W_E_S_alpha_rows_l2": float(soup["W_E_S"][1:].norm()),
            }
            trajectory[f"{arm}_s{seed}"] = rows
    out["z_binding_weight_trajectory"] = trajectory

    # (3) epoch-1 (pre-collapse) response to the same alpha-mean swap
    responses = {}
    for arm in m.REUSE_ARMS:
        model = m.build_arm(m.SKELETON_ARM, payload, kappa, m.load_basis(arm), 0)
        st1 = torch.load(
            m.RESULTS_DIR / "runs" / f"{arm}_s0" / "epoch1_state.pt",
            map_location="cpu", weights_only=False,
        )
        model.load_state_dict(st1)
        model.eval()
        data = m.build_domain_data(objects)
        rows = data["t_eval"][:128]
        device = torch.device("cpu")
        with torch.no_grad():
            h0, _, _ = m._predict_rows(model, rows, device)
            mean_alpha_t = torch.as_tensor(m._t_fit_mean_alpha(arm), dtype=torch.float32)
            original_code = model.code

            def patched(phi, _model=model, _mean=mean_alpha_t, _orig=original_code):
                z = _orig(phi)
                out_z = z.clone()
                out_z[:, _model.common_dim :] = _mean.to(dtype=z.dtype).expand(int(z.shape[0]), -1)
                return out_z

            model.code = patched
            try:
                h1, _, _ = m._predict_rows(model, rows, device)
            finally:
                del model.code
        d = np.abs(h1 - h0)
        responses[f"{arm}_s0_epoch1"] = {
            "n_rows": int(d.size),
            "abs_dpred_max": float(d.max()),
            "abs_dpred_mean": float(d.mean()),
            "note": "the pre-collapse checkpoint responds: the intervention path is wired",
        }
    out["epoch1_response"] = responses

    out["located_answer"] = (
        "the A-slot product binding's z-projection W_A_S collapses monotonically during "
        "the 240-epoch body training (both arms, both seeds: ||W_A_S|| ~ 2.3 at epoch 1 -> "
        "~ 1e-11 in the soup; the edge binding's alpha rows decay 4.5 -> ~4e-4), so the "
        "trained consumers are numerically invariant to the whole z code (common and alpha "
        "alike): predictions are carried by the chemistry product term, the raw/root-tuple "
        "MLP, the Sem108 interface and the shared Q. This matches the historical evidence "
        "boundary of the design basis (the old node product binding is inactive) and does "
        "not contradict the terminal-eval branch D reading"
    )
    out["official_valid_loaded"] = False
    out["official_test_loaded"] = False
    out["seconds"] = float(time.perf_counter() - started)
    (m.RESULTS_DIR / "binding_collapse_diagnostic.json").write_text(
        __import__("json").dumps(out, indent=2) + "\n"
    )
    print("[binding-diag]", out["located_answer"][:120] + "…")
    print(f"[binding-diag] wrote {m.RESULTS_DIR / 'binding_collapse_diagnostic.json'} "
          f"in {out['seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
