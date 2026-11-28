"""Phase 0 — recompute the frozen COUNT-ALL local contributions (no refit).

The ISRD-v1 ``local_contributions()`` divided the standardised design by ``sd`` a
second time when forming ``contrib = (x - mu) / sd * w_raw`` with
``w_raw = w_std / sd``.  Prediction closure itself used the correct raw-unit
parametrisation, so AUC/AP are unaffected; only the contribution magnitudes,
concentration and rankings were wrong.

This script reads the frozen coefficients/predictions, recomputes two exact
decompositions (zero baseline and mean-centred baseline), verifies both against
the stored logits, and writes the corrected concentration statistics together
with the old (buggy) values for the ERRATA.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy.sparse as sp

import sfl_common as K

CHUNK = 1500


def main() -> None:
    t0 = time.perf_counter()
    data = K.HIVData()
    phase_c = K.read_json(K.ISRD_PHASEC)
    arm = phase_c["count_all"]
    w_std = np.asarray(arm["coef_standardised"], dtype=np.float64)
    w_raw = np.asarray(arm["coef_raw"], dtype=np.float64)
    mu = np.asarray(arm["mean"], dtype=np.float64)
    sd = np.asarray(arm["sd"], dtype=np.float64)
    b_std = float(arm["intercept_standardised"])
    b_raw = float(arm["intercept_raw"])
    stored = np.load(K.ISRD_PRED / "COUNT-ALL.npz")["logit"].astype(np.float64)
    assert w_std.size == K.HIV_WIDTH and w_raw.size == K.HIV_WIDTH

    # identity vs the frozen prediction table
    if not np.array_equal(np.load(K.ISRD_PRED / "COUNT-ALL.npz")["y"], data.y):
        raise ValueError("COUNT-ALL y identity mismatch")
    if not np.array_equal(np.load(K.ISRD_PRED / "COUNT-ALL.npz")["graph_id"],
                          np.arange(data.n)):
        raise ValueError("COUNT-ALL graph id identity mismatch")

    X = data.design_sparse()
    n = data.n
    w_struct = w_raw[2:]
    w_struct_std = w_std[2:]
    mu_s, sd_s = mu[2:], sd[2:]
    parent = K.full_parent_columns()
    n_groups = K.HIV_N_C + int((parent < 0).sum())
    group_of = np.empty(K.HIV_N_STRUCT, dtype=np.int64)
    group_of[:K.HIV_N_C] = np.arange(K.HIV_N_C)
    orphan = parent < 0
    group_of[K.HIV_N_C:] = np.where(orphan, K.HIV_N_C + np.cumsum(orphan) - 1, parent)
    G = sp.csr_matrix((np.ones(K.HIV_N_STRUCT), (group_of, np.arange(K.HIV_N_STRUCT))),
                      shape=(n_groups, K.HIV_N_STRUCT))

    names = [f"C[{i}]" for i in range(K.HIV_N_C)] + \
            [f"F[{i}]" for i in range(K.HIV_N_F)]
    nan = np.full((n, 3), np.nan)
    z_rec_zero = np.empty(n)
    z_rec_mean = np.empty(n)
    z_rec_old = np.empty(n)          # reconstruction from the *buggy* formula (should NOT close)
    old_abs_total = np.empty(n)
    old_top = {k: np.empty(n) for k in (5, 10, 20)}
    abs_total = np.empty(n)
    top = {k: np.empty(n) for k in (5, 10, 20)}
    remaining_signed = np.empty(n)
    remaining_abs = np.empty(n)
    n_active = np.empty(n, dtype=np.int32)
    group_abs = np.empty((n, n_groups))
    struct_abs = np.zeros(K.HIV_N_STRUCT)
    struct_signed = np.zeros(K.HIV_N_STRUCT)
    active = np.zeros(K.HIV_N_STRUCT, dtype=np.int64)
    del nan

    for lo in range(0, n, CHUNK):
        hi = min(n, lo + CHUNK)
        x = X[lo:hi].toarray().astype(np.float64)          # [chunk, 4408]
        xs = x[:, 2:]
        # --- exact raw-unit decomposition (zero structure baseline) -------
        c_zero = xs * w_struct
        # --- exact mean-centred decomposition -----------------------------
        c_mean = (xs - mu_s) / sd_s * w_struct_std
        size_raw = x[:, :2] @ w_raw[:2]
        size_mean = (x[:, :2] - mu[:2]) / sd[:2] @ w_std[:2]
        z_rec_zero[lo:hi] = b_raw + size_raw + c_zero.sum(axis=1)
        z_rec_mean[lo:hi] = b_std + size_mean + c_mean.sum(axis=1)
        # --- historical (buggy) contribution magnitude --------------------
        c_old = (xs - mu_s) / sd_s * w_struct
        old_abs = np.abs(c_old)
        old_abs_total[lo:hi] = old_abs.sum(axis=1)
        old_sort = np.argsort(-old_abs, axis=1)
        for k in (5, 10, 20):
            old_top[k][lo:hi] = np.take_along_axis(old_abs, old_sort[:, :k], axis=1).sum(1)
        z_rec_old[lo:hi] = b_raw + size_raw + c_old.sum(axis=1)
        del c_old, old_abs, old_sort

        abs_c = np.abs(c_zero)
        abs_total[lo:hi] = abs_c.sum(axis=1)
        n_active[lo:hi] = (xs > 0).sum(axis=1)
        order = np.argsort(-abs_c, axis=1)
        for k in (5, 10, 20):
            top[k][lo:hi] = np.take_along_axis(abs_c, order[:, :k], axis=1).sum(1)
        keep = np.ones_like(abs_c, dtype=bool)
        np.put_along_axis(keep, order[:, :20], False, axis=1)
        remaining_signed[lo:hi] = np.where(keep, c_zero, 0.0).sum(axis=1)
        remaining_abs[lo:hi] = np.where(keep, abs_c, 0.0).sum(axis=1)
        g = np.asarray(c_zero @ G.T)
        group_abs[lo:hi] = np.abs(g)
        struct_abs += abs_c.sum(axis=0)
        struct_signed += c_zero.sum(axis=0)
        active += (xs > 0).sum(axis=0)
        del x, xs, c_zero, c_mean, abs_c, order, g, keep

    def split_mean(a: np.ndarray, split: str) -> float:
        return float(np.mean(a[data.rows[split]]))

    shares = {k: top[k] / np.maximum(abs_total, 1e-12) for k in top}
    old_shares = {k: old_top[k] / np.maximum(old_abs_total, 1e-12) for k in old_top}
    gsum = group_abs.sum(axis=1)
    gorder = np.argsort(-group_abs, axis=1)
    gtop = {k: np.take_along_axis(group_abs, gorder[:, :k], axis=1).sum(1)
            for k in (5, 10, 20)}
    gshares = {k: gtop[k] / np.maximum(gsum, 1e-12) for k in gtop}

    summary: dict[str, Any] = {
        "arm": "COUNT-ALL", "n_molecules": n, "n_structure_columns": K.HIV_N_STRUCT,
        "n_coarse_groups": int(n_groups),
        "reconstruction_max_abs_error_zero_baseline": float(np.max(np.abs(z_rec_zero - stored))),
        "reconstruction_max_abs_error_mean_centred": float(np.max(np.abs(z_rec_mean - stored))),
        "reconstruction_max_abs_error_old_buggy": float(np.max(np.abs(z_rec_old - stored))),
        "intercept_raw": b_raw, "intercept_standardised": b_std,
        "zero_baseline": {
            "share_top5_mean": {s: split_mean(shares[5], s) for s in K.SPLITS},
            "share_top10_mean": {s: split_mean(shares[10], s) for s in K.SPLITS},
            "share_top20_mean": {s: split_mean(shares[20], s) for s in K.SPLITS},
            "abs_total_mean": {s: split_mean(abs_total, s) for s in K.SPLITS},
            "remaining_abs_mean": {s: split_mean(remaining_abs, s) for s in K.SPLITS},
            "remaining_signed_mean": {s: split_mean(remaining_signed, s) for s in K.SPLITS},
            "n_active_structures_mean": {s: split_mean(n_active, s) for s in K.SPLITS},
            "n_active_structures_median": {
                s: float(np.median(n_active[data.rows[s]])) for s in K.SPLITS},
            "coarse_family_share_top5_mean": {s: split_mean(gshares[5], s) for s in K.SPLITS},
            "coarse_family_share_top10_mean": {s: split_mean(gshares[10], s) for s in K.SPLITS},
            "coarse_family_share_top20_mean": {s: split_mean(gshares[20], s) for s in K.SPLITS},
            "coarse_family_abs_total_mean": {s: split_mean(gsum, s) for s in K.SPLITS},
            "n_active_coarse_groups_mean": {
                s: float(np.mean((group_abs[data.rows[s]] > 0).sum(axis=1)))
                for s in K.SPLITS},
        },
        "old_buggy": {
            "share_top5_mean": {s: split_mean(old_shares[5], s) for s in K.SPLITS},
            "share_top10_mean": {s: split_mean(old_shares[10], s) for s in K.SPLITS},
            "share_top20_mean": {s: split_mean(old_shares[20], s) for s in K.SPLITS},
            "abs_total_mean": {s: split_mean(old_abs_total, s) for s in K.SPLITS},
        },
        "n_nonzero_structure_terms": int((struct_abs > 0).sum()),
        "input_identity": {
            "bundle_sha256": K.sha256_file(K.BUNDLE),
            "stored_logit_sha256": K.sha256_file(K.ISRD_PRED / "COUNT-ALL.npz"),
            "coef_raw_sha256": K.sha256_array(w_raw),
            "coef_std_sha256": K.sha256_array(w_std),
            "mean_sha256": K.sha256_array(mu),
            "sd_sha256": K.sha256_array(sd),
            "graph_id": "0..41126 = bundle row = global graph id",
        },
        "top_structures_by_abs_contribution_total": [],
        "group_definition": (
            "COARSE column i and all FULL columns whose frozen parent is i; "
            "orphan FULL columns (parent outside vocabulary) are singleton groups; "
            "group contribution = exact sum of its model terms"),
        "seconds": float(time.perf_counter() - t0),
        "git": K.git_state(), "environment": K.environment(),
    }
    order = np.argsort(-struct_abs)[:30]
    reg = K.registry()
    for i in order:
        gran = "C" if i < K.HIV_N_C else "F"
        col = int(i if gran == "C" else i - K.HIV_N_C)
        meta = reg[gran].iloc[col]
        summary["top_structures_by_abs_contribution_total"].append({
            "structure": names[i], "granularity": gran, "column": col,
            "canonical_key": str(meta["canonical_key"]),
            "family": str(meta["family"]),
            "coef_raw": float(w_struct[i]),
            "n_active_molecules": int(active[i]),
            "abs_contribution_total": float(struct_abs[i]),
            "signed_contribution_total": float(struct_signed[i]),
        })
    K.write_json(K.ANALYSIS / "phase0_contributions.json", summary)
    print(f"[P0] closure zero-baseline={summary['reconstruction_max_abs_error_zero_baseline']:.2e} "
          f"mean-centred={summary['reconstruction_max_abs_error_mean_centred']:.2e} "
          f"old-buggy={summary['reconstruction_max_abs_error_old_buggy']:.2e}")
    for s in K.SPLITS:
        print(f"[P0] {s}: new top5/10/20="
              f"{summary['zero_baseline']['share_top5_mean'][s]:.4f}/"
              f"{summary['zero_baseline']['share_top10_mean'][s]:.4f}/"
              f"{summary['zero_baseline']['share_top20_mean'][s]:.4f} "
              f"abs_total={summary['zero_baseline']['abs_total_mean'][s]:.1f} "
              f"family_top5/10/20="
              f"{summary['zero_baseline']['coarse_family_share_top5_mean'][s]:.4f}/"
              f"{summary['zero_baseline']['coarse_family_share_top10_mean'][s]:.4f}/"
              f"{summary['zero_baseline']['coarse_family_share_top20_mean'][s]:.4f}")
    print(f"[P0] done in {summary['seconds']:.1f}s")


if __name__ == "__main__":
    main()
