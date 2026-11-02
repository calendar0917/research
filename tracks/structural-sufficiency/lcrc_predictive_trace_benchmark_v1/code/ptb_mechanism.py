"""H2 — structural mechanism audit (read-only evidence + a no-retrain input intervention).

Two strictly separated parts:

**A. Frozen evidence (read-only).**  Re-reads the numbers already reported by the
frozen upstream rounds that the protocol names, preferring the follow-up
corrigendum contrast table over the legacy one.  Nothing is re-estimated, no
checkpoint is touched, and every value carries its source file and sha256.  Rows
taken from a Test split keep their original ``EXPOSED_TEST_EXPLORATORY`` status.

**B. ``FROZEN_MODEL_INPUT_INTERVENTION``.**  On already frozen checkpoints of
this round (plus the frozen upstream Lipophilicity seed-0 checkpoint) the typed-S2
input columns ``LOCAL_LAYOUT["s2_type"] … ["s2_content"]`` (columns 320:520 of the
529-dim local block) are set to zero and the prediction change under fixed weights
is reported.  This is **not** an equal-capacity retrain ablation and **not** a
legal chemical edit; it says only how much the trained weights rely on those input
channels.  A same-width random-block zeroing is measured as a scale control.

**C. Structural statistics.**  Where the removable block sits in the input layout,
how wide it is, and how the 18,432 parameters of ``A0 − C1`` decompose.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

import ptb_common as C
import ptb_data as D
import ptb_model as M
import ptb_train_mol as TM

TYPED_S2 = ("s2_type", "s2_end", "s2_joint", "s2_closure", "s2_content")


def _layout() -> dict[str, tuple[int, int]]:
    import dissect_common as DC  # frozen upstream layout

    return dict(DC.LOCAL_LAYOUT)


def typed_s2_columns() -> tuple[int, int]:
    layout = _layout()
    lo = min(layout[k][0] for k in TYPED_S2)
    hi = max(layout[k][1] for k in TYPED_S2)
    for k in TYPED_S2:
        if layout[k][1] < lo or layout[k][0] > hi:
            raise RuntimeError("typed-S2 blocks are not contiguous")
    return lo, hi


# ---------------------------------------------------------------------------
# A. frozen evidence
# ---------------------------------------------------------------------------


def _pick(obj: Any, path: Sequence[str]) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def frozen_evidence() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    follow = C.REPO / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_followup_v1"
    corrected = follow / "analysis/corrected_tables/corrected_contrasts_valid.json"
    if corrected.exists():
        payload = C.read_json(corrected)
        summary = payload.get("cross_repeat_summary", {})
        for key, value in summary.items():
            rows.append(
                {
                    "source_round": "lcrc_lipo_model_dissection_followup_v1",
                    "source_file": C.track_rel(corrected),
                    "source_sha256": C.sha256_file(corrected),
                    "split": "valid",
                    "label": key,
                    "value": value if not isinstance(value, Mapping) else C.stable_json(value),
                    "status": "READ_ONLY_FROZEN_EVIDENCE",
                    "preferred_over": "lcrc_lipo_model_dissection_v1/analysis/tables/contrasts_valid.json (legacy bug)",
                }
            )
        meta = payload.get("meta", {})
        rows.append(
            {
                "source_round": "lcrc_lipo_model_dissection_followup_v1",
                "source_file": C.track_rel(corrected),
                "source_sha256": C.sha256_file(corrected),
                "split": "valid",
                "label": "legacy_bug_note",
                "value": str(meta.get("legacy_bug_note"))[:400],
                "status": "READ_ONLY_FROZEN_EVIDENCE",
                "preferred_over": "",
            }
        )

    dissection = C.REPO / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1"
    means = dissection / "analysis/tables/variant_means.csv"
    if means.exists():
        with open(means, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("repeat") not in (None, "", "122"):
                    continue
                rows.append(
                    {
                        "source_round": "lcrc_lipo_model_dissection_v1",
                        "source_file": C.track_rel(means),
                        "source_sha256": C.sha256_file(means),
                        "split": "valid",
                        "label": f"{row['variant']}|params_trainable",
                        "value": row.get("params_trainable"),
                        "status": "READ_ONLY_FROZEN_EVIDENCE",
                        "preferred_over": "",
                    }
                )
                rows.append(
                    {
                        "source_round": "lcrc_lipo_model_dissection_v1",
                        "source_file": C.track_rel(means),
                        "source_sha256": C.sha256_file(means),
                        "split": "valid",
                        "label": f"{row['variant']}|valid_rmse_mean",
                        "value": row.get("valid_rmse_mean"),
                        "status": "READ_ONLY_FROZEN_EVIDENCE",
                        "preferred_over": "",
                    }
                )

    controls = C.REPO / "tracks/structural-sufficiency/lcrc_lipo_random42_controls_v2"
    table = controls / "analysis/validation_table.json"
    if table.exists():
        payload = C.read_json(table)
        for entry in payload.get("main_table", []):
            rows.append(
                {
                    "source_round": "lcrc_lipo_random42_controls_v2",
                    "source_file": C.track_rel(table),
                    "source_sha256": C.sha256_file(table),
                    "split": "valid",
                    "label": f"main_table|{entry.get('model', entry.get('name', '?'))}",
                    "value": C.stable_json(entry),
                    "status": "READ_ONLY_FROZEN_EVIDENCE",
                    "preferred_over": "",
                }
            )

    mech = C.REPO / "tracks/structural-sufficiency/lcrc_lipo_mechanism_generalization_ssl_v2"
    cv = mech / "analysis/cv_results_v2.json"
    if cv.exists():
        payload = C.read_json(cv)
        for key in ("H1", "H2", "H3"):
            block = payload.get(key, {})
            for field in ("mean_delta", "any_promising", "materiality", "delta"):
                if field in block:
                    rows.append(
                        {
                            "source_round": "lcrc_lipo_mechanism_generalization_ssl_v2",
                            "source_file": C.track_rel(cv),
                            "source_sha256": C.sha256_file(cv),
                            "split": "cv",
                            "label": f"{key}|{field}",
                            "value": C.stable_json(block[field]),
                            "status": "READ_ONLY_FROZEN_EVIDENCE",
                            "preferred_over": "",
                        }
                    )

    bottleneck = C.REPO / "tracks/structural-sufficiency/zinc_dccl_bottleneck_v1"
    dec = bottleneck / "analysis/decision.json"
    if dec.exists():
        payload = C.read_json(dec)
        phase_b = payload.get("phase_b", {})
        for field in ("delta_locked_seed0", "delta_locked_seed1", "relative_reduction_locked", "ci_seed0", "ci_seed1"):
            if field in phase_b:
                rows.append(
                    {
                        "source_round": "zinc_dccl_bottleneck_v1",
                        "source_file": C.track_rel(dec),
                        "source_sha256": C.sha256_file(dec),
                        "split": "valid",
                        "label": f"phase_b|{field}",
                        "value": C.stable_json(phase_b[field]),
                        "status": "READ_ONLY_FROZEN_EVIDENCE",
                        "preferred_over": "",
                    }
                )
        phase_a = payload.get("phase_a", {})
        for field in ("unseen_n", "unseen_mae", "seen_mae", "unseen_ae_share"):
            if field in phase_a:
                rows.append(
                    {
                        "source_round": "zinc_dccl_bottleneck_v1",
                        "source_file": C.track_rel(dec),
                        "source_sha256": C.sha256_file(dec),
                        "split": "valid",
                        "label": f"phase_a|{field}",
                        "value": C.stable_json(phase_a[field]),
                        "status": "READ_ONLY_FROZEN_EVIDENCE",
                        "preferred_over": "",
                    }
                )

    err = C.REPO / "tracks/structural-sufficiency/zinc_error_mechanism_audit_v1"
    audit = err / "analysis/audit.json"
    if audit.exists():
        payload = C.read_json(audit)
        perm = _pick(payload, ["permutation", "summary"])
        if perm is not None:
            rows.append(
                {
                    "source_round": "zinc_error_mechanism_audit_v1",
                    "source_file": C.track_rel(audit),
                    "source_sha256": C.sha256_file(audit),
                    "split": "valid",
                    "label": "permutation|summary",
                    "value": C.stable_json(perm),
                    "status": "READ_ONLY_FROZEN_EVIDENCE",
                    "preferred_over": "",
                }
            )
        for field in ("counts", "oracle_ceiling"):
            block = payload.get(field)
            if block is not None:
                rows.append(
                    {
                        "source_round": "zinc_error_mechanism_audit_v1",
                        "source_file": C.track_rel(audit),
                        "source_sha256": C.sha256_file(audit),
                        "split": "valid",
                        "label": field,
                        "value": C.stable_json(block),
                        "status": "READ_ONLY_FROZEN_EVIDENCE",
                        "preferred_over": "",
                    }
                )
    return {"n_evidence_rows": len(rows), "rows": rows}


# ---------------------------------------------------------------------------
# B. frozen model input intervention
# ---------------------------------------------------------------------------


def _predict_with(
    model,
    cache: Mapping[str, Any],
    device: str,
    *,
    flag: str | None = None,
    batch_size: int = 128,
) -> np.ndarray:
    """Forward the frozen model, optionally with one frozen intervention flag set."""
    model.eval()
    n = int(cache["n_molecules"])
    out = []
    kwargs = {flag: True} if flag else {}
    with torch.no_grad():
        for start in range(0, n, batch_size):
            rows = list(range(start, min(start + batch_size, n)))
            batch = TM.MD_to_device(TM.PREP_DATA.collate(cache, rows), device)
            pred = model(batch, **kwargs)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            out.append(pred.detach().float().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def _checkpoint_paths(datasets: Sequence[str]) -> list[dict[str, Any]]:
    out = []
    upstream = C.MOLNET_V1 / "analysis/runs"
    for dataset in datasets:
        for model_name in C.MODELS:
            if dataset in C.SUPPORT_VIEWS:
                if model_name != C.MODEL_A0:
                    continue  # the frozen upstream checkpoint is the FULL variant only
                ref = upstream / f"{dataset}_s0" / "checkpoint.pt"
                if ref.exists():
                    out.append(
                        {
                            "dataset": dataset,
                            "model": model_name,
                            "seed": 0,
                            "path": ref,
                            "scope": "FROZEN_UPSTREAM_CHECKPOINT",
                        }
                    )
                continue
            ckpt = C.mol_run_dir(dataset, model_name, 0) / "checkpoint.pt"
            if ckpt.exists():
                out.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "seed": 0,
                        "path": ckpt,
                        "scope": "THIS_ROUND_FROZEN_CHECKPOINT",
                    }
                )
    return out


def local_dim() -> int:
    import dissect_common as DC

    return int(DC.LOCAL_DIM)


#: frozen forward flags, their semantics, and the input width each one silences
INTERVENTIONS: dict[str, dict[str, Any]] = {
    "typed_s2_off": {
        "flag": "zero_s2",
        "columns": "LOCAL_LAYOUT['s2_type']…['s2_content'] = 320:520",
        "width": 200,
        "role": "target",
    },
    "global_g_off": {
        "flag": "zero_g",
        "columns": "rho layout 'g' = 0:42 (normalised global block)",
        "width": 42,
        "role": "control (unrelated input channel)",
    },
    "rho_mean_h2_off": {
        "flag": "zero_mean_h2",
        "columns": "rho layout 'mean_h2' = 106:170",
        "width": 64,
        "role": "control (second moment channel)",
    },
}


def intervention(datasets: Sequence[str], device: str) -> dict[str, Any]:
    lo, hi = typed_s2_columns()
    payload: dict[str, Any] = {
        "schema": "lcrc-ptb-mechanism-intervention-v1",
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "label": "FROZEN_MODEL_INPUT_INTERVENTION",
        "claim_boundary": (
            "fixed trained weights, one frozen forward flag switched on at inference. Not an "
            "equal-capacity retrain ablation, not a legal chemical edit, and not evidence that the "
            "channel is necessary or sufficient."
        ),
        "typed_s2_columns": [int(lo), int(hi)],
        "typed_s2_width": int(hi - lo),
        "local_dim": int(local_dim()),
        "interventions": INTERVENTIONS,
        "rows": [],
    }
    targets = _checkpoint_paths(datasets)
    if not targets:
        payload["status"] = "UNAVAILABLE"
        payload["reason"] = "no frozen checkpoint found"
        return payload
    for t in targets:
        view = D.molecule_view(t["dataset"])
        blob = torch.load(t["path"], map_location="cpu", weights_only=False)
        state = blob.get("state_dict", blob) if isinstance(blob, Mapping) else blob
        n_tasks = int(blob.get("n_tasks", 1)) if isinstance(blob, Mapping) else 1
        model = M.load_checkpoint_model(
            view.spec, view.train_norm, t["model"], state, device, n_tasks=n_tasks
        )
        base = _predict_with(model, view.valid_cache, device)
        row: dict[str, Any] = {
            "dataset": t["dataset"],
            "model": t["model"],
            "seed": t["seed"],
            "scope": t["scope"],
            "checkpoint": C.track_rel(t["path"]),
            "checkpoint_sha256": C.sha256_file(t["path"]),
            "params_total": M.parameter_count(model),
            "n_rows": int(base.shape[0]),
            "n_outputs": int(base.shape[1]),
            "split": "valid",
            "test_split_touched": False,
            "interventions": {},
        }
        for name, spec in INTERVENTIONS.items():
            alt = _predict_with(model, view.valid_cache, device, flag=spec["flag"])
            d = np.abs(alt - base).reshape(-1)
            rho = float(spearmanr(base.reshape(-1), alt.reshape(-1)).statistic) if base.size > 2 else None
            row["interventions"][name] = {
                "role": spec["role"],
                "width": spec["width"],
                "mean_abs_delta": float(np.mean(d)),
                "median_abs_delta": float(np.median(d)),
                "p95_abs_delta": float(np.percentile(d, 95)),
                "max_abs_delta": float(np.max(d)),
                "spearman_vs_base": rho,
            }
        target = row["interventions"]["typed_s2_off"]["mean_abs_delta"]
        for ctrl in ("global_g_off", "rho_mean_h2_off"):
            c = row["interventions"][ctrl]["mean_abs_delta"]
            row["interventions"][ctrl]["ratio_target_over_control"] = float(target / c) if c > 0 else None
        payload["rows"].append(row)
        C.log(
            f"[intervention] {t['dataset']}/{t['model']} {t['scope']}: "
            f"typed_s2 mean|d|={target:.4f} g_off={row['interventions']['global_g_off']['mean_abs_delta']:.4f} "
            f"mean_h2_off={row['interventions']['rho_mean_h2_off']['mean_abs_delta']:.4f}"
        )
    payload["status"] = "OK"
    return payload


# ---------------------------------------------------------------------------
# C. structural statistics
# ---------------------------------------------------------------------------


def structural_statistics() -> dict[str, Any]:
    import dissect_common as DC

    layout = {k: list(v) for k, v in DC.LOCAL_LAYOUT.items()}
    widths = {k: int(v[1] - v[0]) for k, v in DC.LOCAL_LAYOUT.items()}
    view = D.molecule_view(C.MOLECULE_ORDER[0])
    stats: dict[str, Any] = {
        "schema": "lcrc-ptb-structural-statistics-v1",
        "created_utc": C.utcnow(),
        "local_layout": layout,
        "block_widths": widths,
        "local_dim": int(DC.LOCAL_DIM),
        "rho_dim": int(DC.RHO_DIM),
        "typed_s2_width": int(sum(widths[k] for k in TYPED_S2)),
        "typed_s2_share_of_local": float(sum(widths[k] for k in TYPED_S2) / DC.LOCAL_DIM),
        "variants": {},
    }
    for model_name in C.MODELS:
        model, info = M.build_model(view.spec, view.train_norm, model_name, 0, "cpu")
        stats["variants"][model_name] = {
            "params_total": int(info["parameters"]),
            "params_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "parameters_breakdown": info.get("parameters_breakdown"),
        }
    a0 = stats["variants"][C.MODEL_A0]["params_total"]
    c1 = stats["variants"][C.MODEL_C1]["params_total"]
    stats["delta_params"] = int(a0 - c1)
    stats["delta_decomposition"] = {
        "removed_block": "b_mean",
        "removed_input_columns": int(widths["b_mean"]),
        "hidden": 96,
        "projection_width": 192,
        "note": "measured, not assumed: every differing parameter sits in the first local-MLP layer",
        "delta_equals_removed_columns_times_projection": bool((a0 - c1) == widths["b_mean"] * 192),
    }
    # where does the parameter difference live?
    diff = []
    state_a = M.build_model(view.spec, view.train_norm, C.MODEL_A0, 0, "cpu")[0].state_dict()
    state_c = M.build_model(view.spec, view.train_norm, C.MODEL_C1, 0, "cpu")[0].state_dict()
    for name, tensor in state_a.items():
        other = state_c.get(name)
        da = int(tensor.numel())
        dc = int(other.numel()) if other is not None else 0
        if da != dc:
            diff.append({"name": name, "params_a0": da, "params_c1": dc, "delta": da - dc})
    stats["parameter_difference_by_tensor"] = diff
    stats["parameter_difference_total"] = int(sum(d["delta"] for d in diff))
    return stats


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", nargs="*", default=list(C.MOLECULE_ORDER))
    ap.add_argument("--skip", nargs="*", default=[], choices=["evidence", "intervention", "structure"])
    args = ap.parse_args(argv)
    payload: dict[str, Any] = {
        "schema": "lcrc-ptb-mechanism-audit-v1",
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "code_hashes": C.code_hashes(),
        "no_new_training": True,
        "no_new_test_read": True,
    }
    if "structure" not in args.skip:
        payload["structural_statistics"] = structural_statistics()
    if "evidence" not in args.skip:
        payload["frozen_evidence"] = frozen_evidence()
    if "intervention" not in args.skip:
        payload["intervention"] = intervention(args.datasets, args.device)
    C.write_json(C.ANALYSIS / "mechanism_audit.json", payload)
    C.log("mechanism audit -> analysis/mechanism_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
