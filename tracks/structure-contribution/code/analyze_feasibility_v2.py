"""cscl-prediction-feasibility-v2 — Phase-1 CPU audits (no training).

``analyze``      Error-structure audit of the exported cscl-v1 dev predictions
                 (O-rich / O-unit vs raw y) + subgroup/confound tables.
``topo-audit``   Verify the base573 topology25 block is all-zero and that the
                 verified label-free hinge25 cache reproduces the encoded
                 topology_features input up to float32 inversion error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cscl_features as cf  # noqa: E402
from tracks.ksvd.experiments.luyin16 import (  # noqa: E402
    typed_cycle_probe_v1 as tcp,
    zinc_joint_dictionary_decision_v1 as zjd,
    zinc_topology_features as ztf,
    zinc_static_dictionary_pair as sdp,
)

PROTOCOL_ID = "cscl-prediction-feasibility-v2"
RESULTS_CPU = REPO_ROOT / "tracks/structure-contribution/results/feasibility_v2_cpu"
V1_EXPORT = REPO_ROOT / "tracks/structure-contribution/results/cscl_v1_gpu/dev_preds_raw_units.npz"
LEGACY_PREP = (
    REPO_ROOT
    / "tracks/ksvd/results/zinc_full_decomposition_valid_test_confirmation_v1/all_train_prep.npz"
)


def _j(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _j(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_j(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_j(payload), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# error-structure audit
# ---------------------------------------------------------------------------


def run_analyze(out_dir: Path, log=print) -> dict[str, Any]:
    dataset = cf.load_official_train()
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    idx = cf.build_split_indices(smiles)
    dev = np.sort(idx["dev"])
    y_all = np.array([float(d.y.reshape(-1)[0]) for d in dataset], dtype=np.float64)

    z = np.load(V1_EXPORT)
    assert np.array_equal(z["row"], dev), "v1 export rows must equal sorted dev"
    y = z["y"].astype(np.float64)
    assert np.array_equal(y, y_all[dev]), "exported y must equal dataset y"
    arms = {"orich": z["pred_orich"].astype(np.float64), "ounit": z["pred_ounit"].astype(np.float64)}
    errs = {k: np.abs(v - y) for k, v in arms.items()}

    mols, _, _ = cf.load_or_build_units_cache(REPO_ROOT / "data/cache/cscl_v1_units.pt", cf.extract_all)
    from cscl_units import KIND_RING, unique_undirected_bonds

    import networkx as nx

    struct = []
    for i in dev:
        d = dataset[int(i)]
        x = d.x.numpy().reshape(-1)
        ei = d.edge_index.numpy()
        ea = d.edge_attr.numpy().reshape(-1)
        bonds = unique_undirected_bonds(
            [(int(ei[0, k]), int(ei[1, k])) for k in range(ei.shape[1])], [int(v) for v in ea]
        )
        g = nx.Graph()
        g.add_nodes_from(range(len(x)))
        g.add_edges_from([(u, v) for u, v, _ in bonds])
        m = mols[int(i)]
        ring_units = sum(1 for k in m.unit_kinds if k == KIND_RING)
        ring_total = 0
        for k, ua in enumerate(m.unit_atoms):
            n_intra = int(round(float(m.desc[k, 28:32].sum())))
            c = n_intra - len(ua) + 1
            if c > 0:
                ring_total += c
        struct.append(
            {
                "n_atoms": int(len(x)),
                "n_bonds": len(bonds),
                "cyclomatic": len(bonds) - len(x) + nx.number_connected_components(g),
                "ring_units": int(ring_units),
                "ring_total": int(ring_total),
            }
        )
    S = {k: np.array([r[k] for r in struct], dtype=np.int64) for k in struct[0]}

    def dist(e: np.ndarray) -> dict[str, Any]:
        srt = np.sort(e)[::-1]
        out: dict[str, Any] = {
            "mae": float(e.mean()),
            "rmse": float(np.sqrt((e**2).mean())),
            "median": float(np.median(e)),
            "p75": float(np.percentile(e, 75)),
            "p90": float(np.percentile(e, 90)),
            "p95": float(np.percentile(e, 95)),
            "p99": float(np.percentile(e, 99)),
            "max": float(e.max()),
        }
        for k in (1, 5, 10, 20, 50):
            out[f"top{k}_error_mass_share"] = float(srt[:k].sum() / e.sum())
        return out

    def subgroup(name: str, mask: np.ndarray) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n": int(mask.sum()),
            "y_mean": float(y[mask].mean()),
            "y_std": float(y[mask].std()),
        }
        for k, e in errs.items():
            out[f"mae_{k}"] = float(e[mask].mean())
        out["diff_ounit_minus_orich"] = out["mae_ounit"] - out["mae_orich"]
        return out

    qy = np.quantile(y, [0.25, 0.5, 0.75, 0.9, 0.99])
    bands = {
        "y<=q25": y <= qy[0],
        "q25_50": (y > qy[0]) & (y <= qy[1]),
        "q50_75": (y > qy[1]) & (y <= qy[2]),
        "q75_90": (y > qy[2]) & (y <= qy[3]),
        "q90_99": (y > qy[3]) & (y <= qy[4]),
        "q99_100": y > qy[4],
    }
    groups: dict[str, Any] = {
        "y_bands": {k: subgroup(k, m) for k, m in bands.items()},
        "cyclomatic": {
            f"cyc[{lo},{hi})": subgroup("c", (S["cyclomatic"] >= lo) & (S["cyclomatic"] < hi))
            for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 100))
        },
        "ring_units": {
            f"ru[{lo},{hi})": subgroup("r", (S["ring_units"] >= lo) & (S["ring_units"] < hi))
            for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 100))
        },
        "ring_total": {
            f"rt[{lo},{hi})": subgroup("t", (S["ring_total"] >= lo) & (S["ring_total"] < hi))
            for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 100))
        },
        "n_atoms": {
            f"na[{lo},{hi})": subgroup("n", (S["n_atoms"] >= lo) & (S["n_atoms"] < hi))
            for lo, hi in ((0, 15), (15, 22), (22, 30), (30, 100))
        },
    }
    # confound control: ring_units x y-band interaction (same y-range comparison)
    inter = {}
    for band_nm, bm in (("y>-1", y > -1), ("y(-3,-1]", (y <= -1) & (y > -3)), ("y<=-3", y <= -3)):
        for grp_nm, gm in (("ru0_1", S["ring_units"] < 2), ("ru2plus", S["ring_units"] >= 2)):
            m = bm & gm
            if int(m.sum()) == 0:
                continue
            inter[f"{band_nm}|{grp_nm}"] = subgroup("i", m)

    top = {}
    for k, e in errs.items():
        order = np.argsort(e)[::-1][:20]
        top[k] = {
            "rows": [int(dev[i]) for i in order[:20]],
            "abs_errors": [float(e[i]) for i in order[:20]],
            "y": [float(y[i]) for i in order[:20]],
            "n_atoms": [int(S["n_atoms"][i]) for i in order[:20]],
            "ring_units": [int(S["ring_units"][i]) for i in order[:20]],
            "frac_y<=-3": float((y[order] <= -3).mean()),
            "frac_ring_units>=2": float((S["ring_units"][order] >= 2).mean()),
        }

    err_corr = {
        "pearson_orich_ounit": float(np.corrcoef(errs["orich"], errs["ounit"])[0, 1]),
        "pearson_abserr_y_orich": float(np.corrcoef(errs["orich"], y)[0, 1]),
        "pearson_abserr_y_ounit": float(np.corrcoef(errs["ounit"], y)[0, 1]),
        "pearson_abserr_natoms_orich": float(np.corrcoef(errs["orich"], S["n_atoms"])[0, 1]),
    }
    restricted = {}
    for thr in (-3.0, -2.0, -1.0):
        m = y > thr
        restricted[f"y>{thr:g}"] = {
            "n": int(m.sum()),
            **{f"mae_{k}": float(e[m].mean()) for k, e in errs.items()},
            **{f"error_mass_share_{k}": float(e[m].sum() / e.sum()) for k, e in errs.items()},
        }

    report = {
        "protocol": PROTOCOL_ID,
        "source_export": str(V1_EXPORT.relative_to(REPO_ROOT)),
        "note": "per-arm dev_preds_{arm}_s0.npz files are the known defective standardized-unit export; this uses the corrected raw-unit merged export (REPORT §6)",
        "n_dev": int(len(dev)),
        "y_distribution": {
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
            "quantiles": {f"p{q}": float(np.percentile(y, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
            "frac_y<=-3": float((y <= -3).mean()),
        },
        "error_distribution": {k: dist(e) for k, e in errs.items()},
        "subgroups": groups,
        "ring_units_x_yband": inter,
        "top20_errors": top,
        "correlations": err_corr,
        "restricted_mae": restricted,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "v1_error_structure.json", report)
    log(json.dumps({k: report[k] for k in ("error_distribution", "correlations")}, indent=1))
    return report


# ---------------------------------------------------------------------------
# topology25 audit
# ---------------------------------------------------------------------------


def run_topo_audit(out_dir: Path, log=print, n_sample: int = 160) -> dict[str, Any]:
    dataset = cf.load_official_train()
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    idx = cf.build_split_indices(smiles)

    # 1) base573 topology block all-zero?
    rng = np.random.default_rng(0)
    sample = sorted(set(rng.choice(10000, size=n_sample - 40, replace=False).tolist()) | set(int(i) for i in idx["dev"][:40]))
    max_abs = 0.0
    for i in sample:
        base, _t, _c = tcp.graph_feature_row(dataset[int(i)])
        max_abs = max(max_abs, float(np.abs(base[412:437]).max()))
    log(f"[topo] base573 topology block [412:437] max|.| over {len(sample)} rows = {max_abs}")

    # 2) verified hinge25 cache reproduces the encoded topology_features input
    frame = ztf.load_cache("train")
    T_raw = np.stack([ztf.raw_vector(frame.iloc[i], "hinge") for i in range(10000)]).astype(np.float32)
    blob = {k: z for k, z in np.load(LEGACY_PREP, allow_pickle=False).items()}
    topo_all = zjd.Std(blob["topo_all_mean"], blob["topo_all_scale"])
    train = list(torch.load(sdp.CACHE_DIR / "encoded_train.pt", map_location="cpu", weights_only=False))
    enc_T = np.concatenate([d.topology_features.reshape(1, -1).numpy() for d in train], 0)
    recon = topo_all.inverse(enc_T)
    max_recon = float(np.abs(recon - T_raw).max())
    log(f"[topo] max|raw hinge - inverse(encoded topo25)| = {max_recon:.3e}")

    # 3) train-only refit sanity for future use (no labels involved anywhere)
    report = {
        "protocol": PROTOCOL_ID,
        "base573_topology_block": {"slice": [412, 437], "max_abs_over_sample": max_abs, "n_sample": len(sample), "all_zero": bool(max_abs == 0.0)},
        "hinge_cache": {
            "rows": int(len(frame)),
            "width": int(ztf.raw_width("hinge")),
            "definition": "spectrum(L3..>10) + mcb stats + cycle_rank + [L, L^2, ReLU(L-3)..ReLU(L-10)] — pure graph invariants, label-free",
            "raw_range": [float(T_raw.min()), float(T_raw.max())],
        },
        "encoded_consistency": {
            "max_abs_inverse_vs_raw_hinge": max_recon,
            "interpretation": "float32 round-trip precision; encoded topology25 IS the historical Full-Y input, same hinge25 definition",
        },
        "label_free": True,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "topology25_audit.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=PROTOCOL_ID)
    parser.add_argument("mode", choices=("analyze", "topo-audit"))
    parser.add_argument("--out", type=Path, default=RESULTS_CPU)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "analyze":
        run_analyze(args.out)
    else:
        run_topo_audit(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
