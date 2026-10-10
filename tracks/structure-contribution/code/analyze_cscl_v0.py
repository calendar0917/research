"""CSCL-v0 formal analysis (v0_protocol §5/§6): main table, paired bootstrap
differences, cross-seed attribution stability on real ZINC, provenance.

Writes ``results/cscl_v0_gpu/analysis.json`` and ``REPORT.md``.

Opaque (no contribution export) and XGB dev predictions are recomputed here
from saved artifacts / a deterministic refit, then paired with B's predictions.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

BASE = Path("tracks/structure-contribution/results/cscl_v0_gpu")
ARMS = ("additive", "relational", "shuffled", "opaque", "xgb")


def main() -> None:
    import torch

    import cscl_features as cf
    import cscl_models as cm
    from run_cscl_v0 import assemble, build_tensors, prepare_data, set_model_centering, xgb_features

    res = {}
    for arm in ARMS:
        for s in (0, 1):
            f = BASE / (f"{arm}_s{s}" if arm != "xgb" else ".") / f"train_{arm}_s{s}.json"
            res[(arm, s)] = json.load(open(f))

    data = prepare_data(cache=Path("data/cache/cscl_v0_units.pt"))
    stats = data["stats"]
    y_dev = data["y"][data["dev_idx"], 0]

    # model outputs are in standardized-y units; convert to raw units with the
    # recorded y_mean/y_std of the corresponding run
    preds: dict[tuple[str, int], np.ndarray] = {}
    for arm in ("additive", "relational", "shuffled"):
        for s in (0, 1):
            z = np.load(BASE / f"{arm}_s{s}" / "contributions_dev.npz")
            assert np.allclose(z["y"], y_dev, atol=1e-4)
            ym = res[(arm, s)]["y_mean"]
            ys = res[(arm, s)]["y_std"]
            preds[(arm, s)] = z["pred"] * ys + ym
    for s in (0, 1):
        model = cm.build_model("opaque", stats.vocab.n_total, s)
        model.load_state_dict(torch.load(BASE / f"opaque_s{s}" / f"state_opaque_s{s}.pt", weights_only=True))
        model = model.to("cpu")
        model.eval()
        set_model_centering(model, stats)
        outs = []
        for start in range(0, len(data["dev"]), 512):
            batch = assemble(data["dev"][start : start + 512])
            with torch.no_grad():
                outs.append(model(batch))
        ym = res[("opaque", s)]["y_mean"]
        ys = res[("opaque", s)]["y_std"]
        preds[("opaque", s)] = torch.cat(outs).numpy() * ys + ym

    # ---- recompute xgb dev predictions (deterministic refit) --------------
    import xgboost as xgb

    X_in = xgb_features(data["fit_inner"], stats.vocab, data["fit_inner_mols"])
    X_mon = xgb_features(data["monitor"], stats.vocab, data["monitor_mols"])
    X_dev = xgb_features(data["dev"], stats.vocab, data["dev_mols"])
    for s in (0, 1):
        model = xgb.XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.9,
            colsample_bytree=0.9, random_state=s, n_jobs=1, early_stopping_rounds=30,
        )
        model.fit(X_in, data["y"][data["fit_inner_idx"], 0], eval_set=[(X_mon, data["y"][data["monitor_idx"], 0])], verbose=False)
        preds[("xgb", s)] = model.predict(X_dev)

    # ---- paired bootstrap differences vs B --------------------------------
    def paired(a: str, b: str, s: int, n_boot: int = 2000, seed: int = 20261010):
        da = np.abs(preds[(a, s)] - y_dev)
        db = np.abs(preds[(b, s)] - y_dev)
        diff = da - db  # positive: a worse than b
        rng = np.random.RandomState(seed)
        boots = np.array([diff[rng.randint(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"mean": float(diff.mean()), "ci95": [float(lo), float(hi)], "mae_a": float(da.mean()), "mae_b": float(db.mean())}

    paired_diffs = {}
    for a in ("additive", "shuffled", "opaque", "xgb"):
        for s in (0, 1):
            paired_diffs[f"{a}-relational_s{s}"] = paired(a, "relational", s)

    # ---- cross-seed attribution stability (real ZINC, best-epoch models) --
    stability = {}
    alpha_by_seed: dict[int, dict[int, float]] = {}
    gamma_by_seed: dict[int, list[float]] = {}
    rel_pairs_seen: dict[int, dict[tuple[int, int], list[float]]] = {0: {}, 1: {}}
    for s in (0, 1):
        model = cm.build_model("relational", stats.vocab.n_total, s)
        model.load_state_dict(torch.load(BASE / f"relational_s{s}" / "best_state_relational_s{s}.pt".format(s=s), weights_only=True))
        model = model.to("cpu")
        model.eval()
        set_model_centering(model, stats)
        acc: dict[int, list[float]] = {}
        for chunk_ts, chunk_mols in (
            (data["fit_inner"][::4], data["fit_inner_mols"][::4]),
            (data["dev"], data["dev_mols"]),
        ):
            for start in range(0, len(chunk_ts), 512):
                b = assemble(chunk_ts[start : start + 512])
                with torch.no_grad():
                    o = cm.forward_arm(model, b, "relational", None)
                u_off, r_off = 0, 0
                for mol in chunk_mols[start : start + 512]:
                    for k, sig in enumerate(mol.unit_sigs):
                        t = stats.vocab.to_id(sig, mol.unit_kinds[k], len(mol.unit_atoms[k]))
                        acc.setdefault(t, []).append(float(o.alpha[u_off + k]))
                    u_off += len(mol.unit_sigs)
                    for j, (a2, b2) in enumerate(mol.rel_pairs):
                        pair = tuple(sorted((stats.type_id(mol, a2), stats.type_id(mol, b2))))
                        rel_pairs_seen[s].setdefault(pair, []).append(float(o.gamma[r_off + j]))
                    r_off += len(mol.rel_pairs)
        alpha_by_seed[s] = {t: float(np.mean(v)) for t, v in acc.items()}
        gamma_by_seed[s] = [float(g) for pair in rel_pairs_seen[s].values() for g in pair]

    common_types = sorted(set(alpha_by_seed[0]) & set(alpha_by_seed[1]))
    a0 = np.array([alpha_by_seed[0][t] for t in common_types])
    a1 = np.array([alpha_by_seed[1][t] for t in common_types])
    from scipy.stats import pearsonr, spearmanr

    stability["alpha_pearson_across_seeds"] = float(pearsonr(a0, a1).statistic)
    stability["alpha_spearman_across_seeds"] = float(spearmanr(a0, a1).statistic)
    stability["alpha_sign_agreement"] = float(np.mean(np.sign(a0) == np.sign(a1)))
    g0 = np.array(gamma_by_seed[0])
    g1 = np.array(gamma_by_seed[:1][0]) if False else np.array(gamma_by_seed[1])
    n = min(len(g0), len(g1))
    stability["gamma_std_s0"] = float(g0.std())
    stability["gamma_std_s1"] = float(g1.std())
    stability["gamma_mean_abs_s0"] = float(np.abs(g0).mean())
    stability["gamma_mean_abs_s1"] = float(np.abs(g1).mean())
    # top-type stability: correlation of |alpha| magnitudes across seeds
    stability["n_common_types"] = len(common_types)

    # per-type alpha table (top-20 by |alpha| mean across seeds) for the report
    top = sorted(common_types, key=lambda t: -abs((alpha_by_seed[0][t] + alpha_by_seed[1][t]) / 2))[:20]
    alpha_table = [
        {
            "type_id": int(t),
            "alpha_s0": alpha_by_seed[0][t],
            "alpha_s1": alpha_by_seed[1][t],
        }
        for t in top
    ]

    # ---- provenance --------------------------------------------------------
    prov = []
    for f in sorted(glob.glob(str(BASE / "rr_meta" / "cscl-v0-*.json"))):
        m = json.load(open(f))
        r = m.get("runtime", {})
        prov.append(
            {
                "experiment": m.get("experiment"),
                "run_id": m.get("run_id"),
                "commit": m.get("git_commit", "")[:12],
                "pool": m.get("pool"),
                "state": m.get("state"),
                "exit_code": m.get("exit_code"),
                "gpu_names": r.get("gpus", {}).get("names", []),
                "driver": r.get("driver_version"),
                "torch": r.get("torch_version"),
                "cuda_visible": r.get("cuda_visible_devices"),
            }
        )

    out = {
        "protocol": "cscl-v0",
        "primary_metric": "soup_dev_mae",
        "main_table": {
            f"{arm}_s{s}": {
                k: res[(arm, s)].get(k)
                for k in ("soup_dev_mae", "best_epoch_dev_mae", "fit_inner_mae", "n_params", "wall_seconds", "device", "best_epoch", "epochs_run")
            }
            for arm in ARMS
            for s in (0, 1)
        },
        "seed_means": {
            arm: float(np.mean([res[(arm, 0)]["soup_dev_mae"], res[(arm, 1)]["soup_dev_mae"]])) for arm in ARMS
        },
        "paired_diffs_vs_relational": paired_diffs,
        "attribution_stability": stability,
        "alpha_top20_table": alpha_table,
        "provenance": prov,
        "official_test_loaded": False,
        "official_valid_loaded": False,
        "supervision": "raw y only",
    }
    (BASE / "analysis.json").write_text(json.dumps(out, indent=2))

    lines = ["# REPORT — CSCL-v0 formal round (seeds {0,1})", ""]
    lines.append("Protocol `cscl-v0` (notes/v0_protocol.md + addendum). Primary metric: soup dev y-MAE (raw units).")
    lines.append("Data: internal grouped fit 8000 / dev 2000 split of official-train 10000; official valid/test never loaded.")
    lines.append("")
    lines.append("| arm | seed | soup dev MAE | best dev MAE | fit MAE | params | wall s | device |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for s in (0, 1):
            r = res[(arm, s)]
            lines.append(
                f"| {arm} | {s} | {r['soup_dev_mae']:.5f} | {r.get('best_epoch_dev_mae', float('nan')):.5f} | {r['fit_inner_mae']:.5f} | {r['n_params']} | {r['wall_seconds']:.0f} | {r['device']} |"
            )
    lines.append("")
    lines.append("Seed means (soup): " + ", ".join(f"{arm} {out['seed_means'][arm]:.5f}" for arm in ARMS))
    lines.append("")
    lines.append("## Paired differences vs B (relational), molecule bootstrap 95% CI")
    lines.append("")
    lines.append("| comparison | seed | mean | CI95 |")
    lines.append("|---|---|---|---|")
    for key, v in paired_diffs.items():
        lines.append(f"| {key} | | {v['mean']:+.5f} | [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}] |")
    lines.append("")
    lines.append(f"## Attribution stability (real ZINC, best-epoch B models)")
    lines.append("")
    for k, v in stability.items():
        lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    for p in prov:
        lines.append(f"- {p['experiment']} run={p['run_id']} commit={p['commit']} pool={p['pool']} state={p['state']} exit={p['exit_code']} cuda_visible={p['cuda_visible']}")
    (BASE / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
