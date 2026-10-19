"""Phase 1e Stage A2/A3 — error attribution and zero-training routing.

A2 asks whether the frozen models' Monitor error is concentrated on molecules
whose **label** carries an artificial cycle penalty, and whether the models
already predict that penalty.

A3 evaluates, without training, the fixed conditional rule
``RDKit large ring -> R1_size_cap else R_incidence`` as a *post-hoc diagnostic*.
It is not a new model result and it does not select anything.

Both stages use only frozen predictions.  No label term ever becomes a model
input; the cycle-free target is used only as a descriptive counterfactual.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for _p in (HERE, REPO / "tracks/structural-sufficiency/code/phase1b"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import phase1e_common as C  # noqa: E402

OUT_ERRORS = C.RESULTS_1E / "error_attribution.json"
OUT_ROUTING = C.RESULTS_1E / "zero_training_routing.json"
N_BOOT = 5000
BOOT_SEED = 20261021


def _ratio(num: float, den: float) -> float | None:
    return None if den == 0 else float(num / den)


def _pct_contrib(ae: np.ndarray, mask: np.ndarray) -> float:
    total = float(ae.sum())
    return float(ae[np.asarray(mask, dtype=bool)].sum() / total) if total > 0 else 0.0


def _load_strong_baseline(mon: C.MonitorTable) -> dict[str, Any] | None:
    """Stage B2 arm, if it ran: frozen predictions + the freeze record's hashes."""
    npz = C.RESULTS_1E / "baseline" / "monitor_pred_s0.npz"
    freeze = C.RESULTS_1E / "baseline" / "monitor_prediction_freeze.json"
    score = C.RESULTS_1E / "strong_baseline.json"
    if not (npz.exists() and freeze.exists() and score.exists()):
        return None
    rec = C.read_json(freeze)
    if C.sha256_file(npz) != rec["prediction_sha256"]:
        raise SystemExit("strong-baseline prediction file no longer matches its freeze record")
    with np.load(npz) as z:
        pred = np.asarray(z["pred"], dtype=np.float64)
    if pred.shape != mon.y.shape:
        raise SystemExit("strong-baseline prediction length mismatch")
    return {"key": "P3E_seed0", "pred": pred, "freeze": rec, "score": C.read_json(score)}


def _topk_stats(ae: np.ndarray, delta: np.ndarray) -> dict[str, Any]:
    """Tail concentration of a signed per-molecule difference (positive = worse)."""
    order = np.argsort(-np.abs(delta), kind="stable")
    out: dict[str, Any] = {}
    for frac in (0.01, 0.05):
        k = max(1, int(np.ceil(frac * delta.size)))
        top = order[:k]
        share = float(delta[top].sum() / delta.sum()) if delta.sum() != 0 else float("nan")
        rest = np.delete(delta, top)
        out[f"top{int(frac * 100)}pct_share"] = share
        out[f"mean_drop_top{int(frac * 100)}pct"] = float(rest.mean())
    out["abs_error_share_top1pct"] = float(np.sort(ae)[::-1][: max(1, delta.size // 100)].sum() / ae.sum())
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)

    mon = C.load_monitor()
    feat = C.align_features(C.load_features(), mon.rows)
    audit = C.read_json(C.RESULTS_1E / "label_audit.json")

    smiles = C.load_canonical_smiles()
    _, y_all = C.load_zinc_pool()
    comp = C.reconstruct_label_components(smiles)
    lab = C.recover_label_cycle(y_all, comp["logP"], comp["SA"])

    # Monitor-row index into the 10,000-row official-train subset.
    idx10k = mon.rows.astype(np.int64)
    y = mon.y
    y_nocycle = lab["y_without_cycle"][idx10k]
    cycle_comp = lab["cycle_component"][idx10k]
    label_excess = lab["excess"][idx10k]

    large = feat["large_cycles_gt6"] >= 1            # RDKit SSSR, n=40
    penalised = label_excess > 0                     # label-implied, n=43
    penalty_only = penalised & ~large

    groups = {
        "all": np.ones_like(large),
        "label_penalised": penalised,
        "label_complement": ~penalised,
        "rdkit_large": large,
        "penalty_only_rdkit_missed": penalty_only,
    }

    models = list(C.RUN_FILES)
    strong = _load_strong_baseline(mon)
    preds: dict[str, np.ndarray] = dict(mon.pred)
    if strong is not None:
        preds[strong["key"]] = strong["pred"]
        models.append(strong["key"])
    errors: dict[str, Any] = {}
    for key in models:
        pred = preds[key]
        resid = pred - y
        ae = np.abs(resid)
        per_group: dict[str, Any] = {}
        for gname, mask in groups.items():
            st = C.group_stats(ae, resid, mask)
            st["abs_error_contribution"] = float(mask.mean() * st["mae"])
            st["share_of_total_abs_error"] = _pct_contrib(ae, mask)
            # Descriptive counterfactual: the same prediction against the
            # cycle-free target.  Never a model score.
            m = np.asarray(mask, dtype=bool)
            st["mae_vs_cycle_free_target"] = float(np.abs(pred - y_nocycle)[m].mean())
            if m.any() and abs(float(cycle_comp[m].mean())) > 1e-9:
                st["cycle_offset_capture"] = float(
                    (pred - y_nocycle)[m].mean() / cycle_comp[m].mean()
                )
            else:
                st["cycle_offset_capture"] = None
            per_group[gname] = st
        errors[key] = {
            "params": C.PARAMS.get(key, 413505 if key == "P3E_seed0" else None),
            "forward_flops": C.FORWARD_FLOPS.get(key),
            "information_regime": (
                "STAGE_B2_STRONG_BASELINE (full graph + junction objects + structure-only "
                "global descriptors incl. ring-size fractions; Fit-only masked SSL)"
                if key == "P3E_seed0"
                else "SSRC_E2_PATCH_OR_GRAPH_ARM"
            ),
            "groups": per_group,
            "tail": _topk_stats(ae, resid),
        }

    # Paired comparisons against the frozen reference G_ref.
    comparisons: dict[str, Any] = {}
    ref = mon.pred["G_ref_seed0"]
    for key in models:
        if key == "G_ref_seed0":
            continue
        delta = np.abs(preds[key] - y) - np.abs(ref - y)
        row: dict[str, Any] = {"definition": "mean |e_candidate| - |e_Gref|; positive = candidate worse"}
        for gname, mask in groups.items():
            vals = delta[np.asarray(mask, dtype=bool)]
            row[gname] = C.paired_bootstrap_mean(vals, n_boot=N_BOOT, seed=BOOT_SEED)
            row[gname]["win_rate_candidate_better"] = float(np.mean(vals < 0)) if vals.size else None
        row["tail"] = _topk_stats(np.abs(preds[key] - y), delta)
        comparisons[key] = row

    # A2 summary of the label side.
    label_summary = {
        "n_label_penalised": int(penalised.sum()),
        "n_rdkit_large": int(large.sum()),
        "mean_cycle_component_penalised": float(cycle_comp[penalised].mean()),
        "mean_abs_cycle_component_penalised": float(np.abs(cycle_comp[penalised]).mean()),
        "mean_abs_y_without_cycle_penalised": float(np.abs(y_nocycle[penalised]).mean()),
        "cycle_share_of_label_magnitude_penalised": _ratio(
            float(np.abs(cycle_comp[penalised]).mean()),
            float(np.abs(cycle_comp[penalised]).mean()) + float(np.abs(y_nocycle[penalised]).mean()),
        ),
        "n_penalty_levels": {int(k): int(v) for k, v in zip(*np.unique(label_excess, return_counts=True))},
    }

    errors_payload = {
        "protocol_id": "ssrc-phase1e-zinc-v1",
        "stage": "A2_error_attribution",
        "scope": {
            "monitor_rows": int(mon.rows.size),
            "post_hoc_on_reused_monitor": True,
            "confirmatory": False,
            "reserved_labels_read": 0,
            "official_valid_test_read": 0,
        },
        "label_summary": label_summary,
        "group_definitions": {
            "label_penalised": "official train target carries a non-zero GVAE cycle penalty",
            "rdkit_large": "Phase 1d split: large_cycles_gt6 >= 1 (RDKit GetSymmSSSR max ring > 6)",
            "penalty_only_rdkit_missed": "label penalised but RDKit SSSR max ring <= 6",
        },
        "models": errors,
        "comparisons_vs_G_ref": comparisons,
        "counterfactual_note": (
            "mae_vs_cycle_free_target is the same frozen prediction compared with "
            "y - cycle_component. It is a descriptive attribution, not a model "
            "score and not a permitted input or post-hoc correction."
        ),
        "stage_b2_arm": None if strong is None else {
            "key": strong["key"],
            "monitor_mae": strong["score"]["monitor_mae"],
            "prediction_sha256": strong["freeze"]["prediction_sha256"],
            "soup_sha256": strong["freeze"]["soup_sha256"],
            "information_note": (
                "this arm sees structure-only global descriptors that include ring-size "
                "fractions (ring_size7_frac, ring_size8plus_frac) and cyclomatic rank, "
                "which the SSRC arms do not receive; its ring-tail advantage is therefore "
                "partly an information-access difference, not an architecture effect"
            ),
        },
        "label_audit_crosscheck": {
            "monitor_confusion": audit["monitor_confusion"],
            "permutation_stability": {
                k: v for k, v in audit["permutation_stability"].items() if k != "unstable_rows_10k"
            },
        },
    }
    C.write_json(OUT_ERRORS, errors_payload)

    # ------------------------------------------------------------------ A3
    cap = mon.pred["R1_size_cap_seed0"]
    inc = mon.pred["R_incidence_seed0"]
    routed_rdkit = np.where(large, cap, inc)
    routed_oracle = np.where(penalised, cap, inc)
    routed_union = np.where(large | penalised, cap, inc)  # equals routed_oracle here

    def route_block(pred: np.ndarray, route: np.ndarray, name: str) -> dict[str, Any]:
        ae = np.abs(pred - y)
        out = {
            "name": name,
            "monitor_mae": float(ae.mean()),
            "monitor_rmse": float(np.sqrt((ae**2).mean())),
            "groups": {
                g: C.group_stats(ae, pred - y, m) for g, m in groups.items()
            },
            "route_sizes": {
                "to_R1_size_cap": int(np.asarray(route, dtype=bool).sum()),
                "to_R_incidence": int((~np.asarray(route, dtype=bool)).sum()),
            },
        }
        return out

    def cmp(pred: np.ndarray, other: np.ndarray, name: str) -> dict[str, Any]:
        delta = np.abs(pred - y) - np.abs(other - y)
        return {
            "name": name,
            "delta": C.paired_bootstrap_mean(delta, n_boot=N_BOOT, seed=BOOT_SEED),
            "delta_penalised": C.paired_bootstrap_mean(
                delta[penalised], n_boot=N_BOOT, seed=BOOT_SEED
            ),
            "delta_complement": C.paired_bootstrap_mean(
                delta[~penalised], n_boot=N_BOOT, seed=BOOT_SEED
            ),
        }

    ref_ae = np.abs(ref - y)
    route_ae = np.abs(routed_rdkit - y)
    # Conditional-execution FLOPs lower bound: per-molecule weighted mean of the
    # two frozen forward FLOPs.  A batched implementation cannot go below the
    # cost of the larger sub-model for its own molecules, so this is a bound.
    f_inc = C.FORWARD_FLOPS["R_incidence_seed0"]
    f_cap = C.FORWARD_FLOPS["R1_size_cap_seed0"]
    frac_cap = float(large.mean())
    flops_lower_bound = frac_cap * f_cap + (1.0 - frac_cap) * f_inc
    gref_flops = C.FORWARD_FLOPS["G_ref_seed0"]

    routing_payload = {
        "protocol_id": "ssrc-phase1e-zinc-v1",
        "stage": "A3_zero_training_routing",
        "status": "POST_HOC_DIAGNOSTIC_ON_A_REUSED_MONITOR",
        "not_a_new_model_result": True,
        "rule": "if RDKit large ring (large_cycles_gt6 >= 1) -> R1_size_cap, else R_incidence",
        "expected_value_phase1d": 0.1922,
        "routed_rdkit": route_block(routed_rdkit, large, "rdkit_route"),
        "routed_label_oracle": route_block(routed_oracle, penalised, "label_penalty_oracle_route"),
        "single_models": {
            "G_ref": float(ref_ae.mean()),
            "R1_size_cap": float(np.abs(cap - y).mean()),
            "R_incidence": float(np.abs(inc - y).mean()),
        },
        "comparisons": [
            cmp(routed_rdkit, ref, "routed - G_ref"),
            cmp(routed_rdkit, cap, "routed - R1_size_cap"),
            cmp(routed_rdkit, inc, "routed - R_incidence"),
            cmp(routed_oracle, ref, "label_oracle_route - G_ref"),
        ],
        "routing_comparison_detail": {
            "delta_vs_Gref_penalised": C.paired_bootstrap_mean(
                (route_ae - ref_ae)[penalised], n_boot=N_BOOT, seed=BOOT_SEED
            ),
            "delta_vs_Gref_complement": C.paired_bootstrap_mean(
                (route_ae - ref_ae)[~penalised], n_boot=N_BOOT, seed=BOOT_SEED
            ),
        },
        "compute": {
            "forward_flops_R_incidence": f_inc,
            "forward_flops_R1_size_cap": f_cap,
            "forward_flops_G_ref": gref_flops,
            "fraction_routed_to_cap": frac_cap,
            "flops_per_molecule_lower_bound": flops_lower_bound,
            "flops_ratio_vs_G_ref_lower_bound": flops_lower_bound / gref_flops,
            "flops_ratio_single_R1_size_cap_vs_G_ref": f_cap / gref_flops,
            "note": (
                "per-molecule weighted average is a theoretical lower bound. A real "
                "batched implementation must run homogeneous batches, so the "
                "achievable cost is closer to the max of the two sub-model costs "
                "for each populated group; the router itself is a ring-size count "
                "and is negligible."
            ),
        },
        "interpretation_guard": (
            "The routing rule was derived from the same Monitor that produced the "
            "component models. Its MAE must not be quoted as a new method result, "
            "and it must not be upgraded to a proposal unless it closes the G_ref gap."
        ),
    }
    C.write_json(OUT_ROUTING, routing_payload)

    cap_mae = float(np.abs(cap - y).mean())
    print(f"[A2] label-penalised n={int(penalised.sum())} rdkit-large n={int(large.sum())}")
    for key in ("G_ref_seed0", "R1_size_cap_seed0", "R_incidence_seed0"):
        g = errors[key]["groups"]
        print(
            f"[A2] {key:22s} all={g['all']['mae']:.4f} "
            f"complement={g['label_complement']['mae']:.4f} "
            f"penalised={g['label_penalised']['mae']:.4f} "
            f"pen_share={g['label_penalised']['share_of_total_abs_error']:.3f}"
        )
    print(
        f"[A3] routed MAE={float(route_ae.mean()):.4f} "
        f"(R1_size_cap {cap_mae:.4f}, G_ref {float(ref_ae.mean()):.4f}) "
        f"flops_lb={flops_lower_bound / gref_flops:.3f}x G_ref"
    )
    print(f"[A2/A3] wrote {OUT_ERRORS.relative_to(REPO)}, {OUT_ROUTING.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
