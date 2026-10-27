"""Descriptive statistics of the frozen representations (no labels, no gates).

Reports, per checkpoint and per block: covariance trace / share, participation
ratio, finite-sample numerical rank; and linear CKA across seeds / scales.
These are descriptive only -- a low rank is *not* evidence of missing
information and never enters the decision.
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np

import fta_common as K


def condition_slice(freeze: dict, name: str) -> slice:
    a, b = freeze["export_rows"]["slices"][name]
    return slice(a, b)


def main() -> int:
    t0 = time.perf_counter()
    cfg = K.load_config()
    freeze = json.loads((K.ANALYSIS / "representation_freeze.json").read_text())
    zs = {}
    for fit, seed in K.CONDITIONS:
        cid = f"{fit}_s{seed}"
        zs[cid] = np.load(K.Z_DIR / f"{cid}.npz")["z"]

    out = {
        "round": K.ROUND_ID,
        "note": ("descriptive only; low participation ratio / numerical rank is not "
                 "interpreted as insufficient information"),
        "row_sets": {},
        "cka": {},
    }

    for rowset in ("probe_fit", "monitor"):
        sl = condition_slice(freeze, rowset)
        block_out = {}
        for fit, seed in K.CONDITIONS:
            cid = f"{fit}_s{seed}"
            z = zs[cid][sl]
            var = K.block_variance_summary(z)
            spec = {}
            for name, (a, b) in K.BLOCKS.items():
                spec[name] = K.spectral_summary(z[:, a:b])
                spec[name]["participation_ratio"] = K.participation_ratio(
                    np.linalg.svd(z[:, a:b] - z[:, a:b].mean(0), compute_uv=False)
                )
            spec["full_513"] = K.spectral_summary(z)
            block_out[cid] = {"variance": var, "spectra": spec}
        out["row_sets"][rowset] = block_out

        # linear CKA across conditions (raw z, row-centred inside the function)
        cka_full = {}
        cka_blocks = {}
        for c1, c2 in itertools.combinations([f"{f}_s{s}" for f, s in K.CONDITIONS], 2):
            cka_full[f"{c1}|{c2}"] = K.linear_cka(zs[c1][sl], zs[c2][sl])
            cka_blocks[f"{c1}|{c2}"] = {
                name: K.linear_cka(zs[c1][sl][:, a:b], zs[c2][sl][:, a:b])
                for name, (a, b) in K.BLOCKS.items()
            }
        out["cka"][rowset] = {"full_513": cka_full, "blocks": cka_blocks}

        # secondary: CKA after the per-condition probe-fit standardisation the
        # head actually sees (probe_fit statistics applied to this row set)
        pf = condition_slice(freeze, "probe_fit")
        std = {}
        for fit, seed in K.CONDITIONS:
            cid = f"{fit}_s{seed}"
            mu, sd = K.standardizer(zs[cid][pf])
            std[cid] = K.apply_standardizer(zs[cid][sl], mu, sd)
        out["cka"][rowset]["standardized_full_513"] = {
            f"{c1}|{c2}": K.linear_cka(std[c1], std[c2])
            for c1, c2 in itertools.combinations(list(std), 2)
        }

    out["block_layout"] = {k: list(v) for k, v in K.BLOCKS.items()}
    out["wall_seconds"] = float(time.perf_counter() - t0)
    K.write_json(K.ANALYSIS / "representation_stats.json", out)
    K.ledger_append({"stage": "stats", "wall_seconds": out["wall_seconds"], "device": "cpu"})

    lines = ["# representation statistics (descriptive)", ""]
    cid0 = "fit4k_s0"
    lines.append("| block | dim | fit4k_s0 var | share | PR | rank(1e-6) | rank99 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    v = out["row_sets"]["probe_fit"][cid0]
    for name, (a, b) in K.BLOCKS.items():
        s = v["spectra"][name]
        lines.append(f"| {name} | {b - a} | {v['variance']['blocks'][name]['trace_cov']:.2f} | "
                     f"{v['variance']['blocks'][name]['trace_share']:.3f} | "
                     f"{s['participation_ratio']:.2f} | {s['numerical_rank_tol_1e-06']} | "
                     f"{s['rank_99pct_energy']} |")
    s = v["spectra"]["full_513"]
    lines.append(f"| full_513 | 513 | {v['variance']['total_trace_cov']:.2f} | 1.000 | "
                 f"{s['participation_ratio']:.2f} | {s['numerical_rank_tol_1e-06']} | "
                 f"{s['rank_99pct_energy']} |")
    lines += ["", "## linear CKA (probe_fit rows, full 513)", "", "| pair | CKA |", "|---|---:|"]
    for k, val in out["cka"]["probe_fit"]["full_513"].items():
        lines.append(f"| {k} | {val:.4f} |")
    (K.ANALYSIS / "representation_stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    K.log(f"wrote analysis/representation_stats.json ({out['wall_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
