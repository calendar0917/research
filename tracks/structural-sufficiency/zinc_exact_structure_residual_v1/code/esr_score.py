"""Post-freeze scoring / analysis for zinc-exact-structure-residual-v1.

Why this module exists
---------------------
The frozen runner (``esr_runner.py``, run commit ``8df7986``) declares its
registered run matrix with human-readable config names (``E@0.1``) while the
on-disk run keys use ``E_eta0.1``.  Its ``score``/``analyze`` orchestration
therefore cannot map a registered config to a prediction file.  That is a
stage-plumbing bug; it does not touch training, predictions or any scientific
constant.

The freeze binds the code hashes of the runner / data / model / config modules,
so rather than editing the frozen runner after training this module reads the
frozen artifacts and reuses the frozen helpers (``verify_freeze``, ``_err``,
``_contrast``, ``valid_labels``, ``parse_run_key``, ``model_for_run``).  The
protocol constants and the decision rule are read from the frozen config and the
frozen ``protocol_freeze.json``.  No prediction or training artifact is
modified; the valid labels remain the single hash-gated read.

Outputs use the artifact names declared in PROTOCOL.md:
``analysis/metrics.json``, ``contrasts.json``, ``decision.json``,
``mechanism.json``, ``groups.json``, ``gpu_ledger.json`` and
``run_manifest.tsv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parent
REPO = _CODE_DIR.parents[3]
for _p in (_CODE_DIR, str(REPO), str(REPO / "tracks/structure-contribution/code")):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

import esr_data as D  # noqa: E402
import esr_runner as R  # noqa: E402

ANALYSIS = R.ANALYSIS
PRED = R.PRED
RUNS = R.RUNS
CURVES = R.CURVES


def config_pairs(phase: int) -> tuple[list[str], list[str]]:
    """Registered config strings -> canonical run keys."""
    cfg = R.load_config()
    if phase == 1:
        declared = [str(v) for v in cfg["runs"]["phase1"]["configs"]]
        configs = []
        for item in declared:
            if item == "B":
                configs.append("B")
            else:
                arm, _, eta = item.partition("@")
                configs.append(R.run_key(arm, float(eta)))
        return declared, configs
    # phase 2: only the purchased configurations
    p1 = R.read_json(R.DECISION_JSON)
    configs = ["B"] + list(p1["phase2_plan"]["purchase_configs"])
    return configs, configs


def load_predictions() -> dict[str, np.ndarray]:
    freeze = R.read_json(R.PRED_FREEZE)
    R.require(freeze["frozen_before_valid_read"] is True, "prediction freeze not pre-label")
    out: dict[str, np.ndarray] = {}
    for key, entry in freeze["prediction_files"].items():
        path = REPO / entry["path"]
        R.require(R.file_sha256(path) == entry["sha256"], f"prediction drift: {key}")
        arr = np.load(path)["pred"].astype(np.float64)
        R.require(arr.shape == (R.N_VALID,), f"bad prediction shape {key}")
        out[key] = arr
    return out


def score(phase: int) -> dict[str, Any]:
    R.verify_freeze()
    R.require(R.PRED_FREEZE.exists(), "freeze predictions first")
    y, y_meta = R.valid_labels()
    cfg = R.load_config()
    declared, configs = config_pairs(phase)
    available = set(R.read_json(R.PRED_FREEZE)["prediction_files"])
    if phase == 1:
        seeds = [int(s) for s in cfg["runs"]["phase1"]["seeds"]]
        score_json, contrast_json, decision_json = R.SCORE_JSON, R.CONTRAST_JSON, R.DECISION_JSON
    else:
        R.require(R.DECISION_JSON.exists(), "phase-1 decision missing")
        p1 = R.read_json(R.DECISION_JSON)
        R.require(p1["phase2_plan"]["authorized"], "phase 2 was not authorized by the phase-1 rule")
        seeds = sorted({int(s) for s in cfg["runs"]["phase2"]["seeds"]} | {int(s) for s in cfg["runs"]["phase1"]["seeds"]})
        seeds = [s for s in seeds if all(f"valid_{k}_s{s}" in available for k in configs)]
        score_json = ANALYSIS / "metrics_phase2.json"
        contrast_json = ANALYSIS / "contrasts_phase2.json"
        decision_json = ANALYSIS / "decision_phase2.json"
    R.require(len(seeds) >= 2, "score needs at least two seeds")
    preds = load_predictions()

    models: dict[str, Any] = {}
    for key in configs:
        per_seed = {str(s): R.f10k.row_metrics(preds[f"valid_{key}_s{s}"], y) for s in seeds}
        models[key] = {
            "mae_seed_mean": float(np.mean([per_seed[str(s)]["mae"] for s in seeds])),
            "per_seed": per_seed,
            "params": R.read_json(RUNS / f"{key}_s{seeds[0]}" / "train.json")["parameter_audit"]["total_parameters"],
            "wall_seconds": {str(s): R.read_json(RUNS / f"{key}_s{s}" / "train.json")["wall_seconds"] for s in seeds},
            "soup_fit_mae": {str(s): R.read_json(RUNS / f"{key}_s{s}" / "train.json")["soup_fit_mae"] for s in seeds},
        }

    b_errs = [R._err(preds[f"valid_B_s{s}"], y) for s in seeds]
    b_mean_err = np.mean(np.stack(b_errs, 0), axis=0)
    worst = np.argsort(-b_mean_err)[: int(cfg["decision"]["drop_reference_worst_k"])]
    keep = np.ones(R.N_VALID, dtype=bool)
    keep[worst] = False
    rows_ordinary = np.where(y > float(cfg["decision"]["ordinary_threshold"]))[0]
    rows_tail = np.where(y <= float(cfg["decision"]["ordinary_threshold"]))[0]

    contrasts: dict[str, Any] = {}
    for key in configs:
        if key == "B":
            continue
        c_errs = [R._err(preds[f"valid_{key}_s{s}"], y) for s in seeds]
        full = R._contrast(c_errs, b_errs)
        full["per_seed"] = {
            str(s): float(R._err(preds[f"valid_{key}_s{s}"], y).mean() - R._err(preds[f"valid_B_s{s}"], y).mean())
            for s in seeds
        }
        full["all_seeds_improve"] = bool(all(v < 0.0 for v in full["per_seed"].values()))
        full["seed_mean_gain"] = float(models["B"]["mae_seed_mean"] - models[key]["mae_seed_mean"])
        full["ordinary"] = R._contrast(c_errs, b_errs, rows_ordinary)
        full["tail"] = R._contrast(c_errs, b_errs, rows_tail)
        full["drop_B_worst_k"] = {
            "k": int(len(worst)),
            "rows": [int(i) for i in worst.tolist()],
            **R._contrast(c_errs, b_errs, np.where(keep)[0]),
        }
        contrasts[f"{key}_minus_B"] = full
    for eta in cfg["prior"]["etas"]:
        ek, hk = R.run_key("E", float(eta)), R.run_key("H", float(eta))
        if ek in configs and hk in configs:
            e_errs = [R._err(preds[f"valid_{ek}_s{s}"], y) for s in seeds]
            h_errs = [R._err(preds[f"valid_{hk}_s{s}"], y) for s in seeds]
            contrasts[f"{hk}_minus_{ek}"] = {
                **R._contrast(h_errs, e_errs),
                "per_seed": {
                    str(s): float(R._err(preds[f"valid_{hk}_s{s}"], y).mean() - R._err(preds[f"valid_{ek}_s{s}"], y).mean())
                    for s in seeds
                },
            }

    R.write_json(
        score_json,
        {
            "protocol": R.PROTOCOL_ID,
            "phase": phase,
            "git_commit": R.git_commit(),
            "valid_labels": y_meta,
            "seeds": seeds,
            "declared_configs": declared,
            "run_keys": configs,
            "models": models,
            "rows_ordinary": int(rows_ordinary.size),
            "rows_tail": int(rows_tail.size),
            "drop_B_worst_k_rows": [int(i) for i in worst.tolist()],
            "official_test_loaded": False,
        },
    )
    R.write_json(contrast_json, {"protocol": R.PROTOCOL_ID, "phase": phase, "contrasts": contrasts})

    gain_min = float(cfg["decision"]["family_vs_B"]["mean_valid_mae_gain_min"])
    decision: dict[str, Any] = {
        "protocol": R.PROTOCOL_ID,
        "phase": phase,
        "git_commit": R.git_commit(),
        "thresholds": cfg["decision"],
        "seeds": seeds,
        "run_keys": configs,
        "B_seed_mean_valid_mae": models["B"]["mae_seed_mean"],
        "families": {},
        "triggers": {},
    }
    for family in ("E", "H"):
        fam: dict[str, Any] = {}
        for eta in cfg["prior"]["etas"]:
            key = R.run_key(family, float(eta))
            c = contrasts.get(f"{key}_minus_B")
            if c is None or key not in models:
                continue
            checks = {
                "both_seeds_improve": bool(c["all_seeds_improve"]),
                "seed_mean_gain_ge_threshold": bool(c["seed_mean_gain"] >= gain_min),
                "ordinary_not_reversed": bool(c["ordinary"]["delta"] < 0.0),
                "drop_B_worst_k_not_reversed": bool(c["drop_B_worst_k"]["delta"] < 0.0),
            }
            fam[str(eta)] = {
                "declared_config": f"{family}@{eta}",
                "run_key": key,
                "mae_seed_mean": models[key]["mae_seed_mean"],
                "per_seed_mae": {s: models[key]["per_seed"][s]["mae"] for s in models[key]["per_seed"]},
                "seed_mean_gain_vs_B": c["seed_mean_gain"],
                "per_seed_delta_vs_B": c["per_seed"],
                "ordinary_delta": c["ordinary"]["delta"],
                "drop_B_worst_k_delta": c["drop_B_worst_k"]["delta"],
                "bootstrap_all": c["bootstrap"],
                "checks": checks,
                "trigger": bool(all(checks.values())),
            }
        triggers = [float(e) for e, v in fam.items() if v["trigger"]]
        best = max(triggers, key=lambda e: fam[str(e)]["seed_mean_gain_vs_B"]) if triggers else None
        decision["families"][family] = fam
        decision["triggers"][family] = {"triggered": bool(triggers), "frozen_eta": best}
    triggered = [f for f in ("E", "H") if decision["triggers"][f]["triggered"]]
    decision["any_trigger"] = bool(triggered)
    decision["families_triggered"] = triggered
    decision["phase2_plan"] = {
        "authorized": bool(triggered) and phase == 1,
        "purchase_B": True,
        "purchase_configs": [R.run_key(f, decision["triggers"][f]["frozen_eta"]) for f in triggered],
        "seeds": [int(s) for s in cfg["runs"]["phase2"]["seeds"]],
        "ssl_seeds": [int(s) for s in cfg["runs"]["phase2"]["ssl_seeds_to_train"]],
        "note": "only families passing the pre-registered multi-seed rule are purchased; no eta search beyond the two declared levels",
    }
    if phase == 2:
        decision["phase1_reference"] = R.read_json(R.DECISION_JSON)["triggers"]
    R.write_json(decision_json, decision)

    R.log(f"[score p{phase}] B seed-mean {models['B']['mae_seed_mean']:.6f}")
    for key in configs[1:]:
        c = contrasts[f"{key}_minus_B"]
        R.log(
            f"[score p{phase}] {key}: mae {models[key]['mae_seed_mean']:.6f} gain {c['seed_mean_gain']:+.6f} "
            f"all {c['all_seeds_improve']} ord {c['ordinary']['delta']:+.6f} "
            f"drop10 {c['drop_B_worst_k']['delta']:+.6f}"
        )
    for f in ("E", "H"):
        R.log(f"[score p{phase}] {f}: triggered={decision['triggers'][f]['triggered']} eta={decision['triggers'][f]['frozen_eta']}")
    return decision


# ---------------------------------------------------------------------------
# mechanism / groups / ledger
# ---------------------------------------------------------------------------


def analyze(device_str: str = "cpu") -> None:
    R.require(R.DECISION_JSON.exists(), "run score first")
    cfg = R.load_config()
    y, _ = R.valid_labels()
    prep = R.read_json(R.PREP_JSON)
    vocab = R.load_vocab()
    valid_objs = R.load_objects_valid()
    assigned = R.load_assigned("valid")
    device = torch.device(device_str)
    seeds = [int(s) for s in cfg["runs"]["phase1"]["seeds"]]
    _, configs = config_pairs(1)
    cuts = prep["train_size_terciles"]

    mechanism: dict[str, Any] = {"protocol": R.PROTOCOL_ID, "runs": {}}
    for key in configs:
        arm, eta = R.parse_run_key(key)
        for s in seeds:
            train_rec = R.read_json(RUNS / f"{key}_s{s}" / "train.json")
            R._write_curve(CURVES / f"{key}_s{s}.csv", train_rec["curve"])
            if arm == "B":
                continue
            model, _, _ = R.model_for_run(key, s, device)
            residual = model.atom_residual
            with torch.no_grad():
                a = residual.correction_matrix()
                norms = a.norm(dim=0)
                depth = residual.depth
                per_type = {
                    "mean_norm_all": float(norms.mean()),
                    "mean_norm_depth1": float(norms[depth == 1].mean()),
                    "mean_norm_depth2": float(norms[depth == 2].mean()),
                    "max_norm": float(norms.max()),
                    "active_fraction": float((norms > 1e-12).float().mean()),
                    "mean_beta_abs": float(residual.beta.detach().abs().mean()),
                    "P_column_norm_min": float(residual.P.detach().norm(dim=0).min()),
                    "P_column_norm_max": float(residual.P.detach().norm(dim=0).max()),
                }
                prior, _ = residual.prior()
                t = np.concatenate([np.asarray(x["t_star"]) for x in assigned])
                used = t[t >= 0]
                occ_norm = float(norms[used].mean()) if used.size else 0.0
                atom_types = torch.as_tensor(t, dtype=torch.long, device=device)
                coeff = residual.beta.index_select(0, atom_types.clamp(min=0))
                coeff = coeff * (atom_types >= 0).unsqueeze(1).to(coeff.dtype)
                atom_norm = (coeff @ residual.P.t()).norm(dim=1).cpu().numpy()
            pred_full = np.load(PRED / f"valid_{key}_s{s}.npz")["pred"].astype(np.float64)
            with torch.no_grad():
                saved = residual.beta.detach().clone()
                residual.beta.zero_()
                pred_zero = R.M.predict(model, valid_objs, assigned, device).astype(np.float64)
                residual.beta.copy_(saved)
            prior_curve = [row["prior"] for row in train_rec["curve"] if "prior" in row and row["prior"] is not None]
            mechanism["runs"][f"{key}_s{s}"] = {
                "prior_value": float(prior),
                "lambda": train_rec["prior"]["lambda"],
                "per_type": per_type,
                "occurrence_weighted_correction_norm_valid": occ_norm,
                "mean_atom_correction_norm_valid": float(atom_norm.mean()),
                "max_atom_correction_norm_valid": float(atom_norm.max()),
                "valid_zero_residual_mae": float(np.abs(pred_zero - y).mean()),
                "valid_full_mae": float(np.abs(pred_full - y).mean()),
                "dependence_delta_mae": float(np.abs(pred_zero - y).mean() - np.abs(pred_full - y).mean()),
                "dependence_delta_note": "positive = zeroing the trained residual worsens valid MAE; this is dependence, not net benefit vs B",
                "grad_norm_P_last": train_rec["curve"][-1].get("grad_norm_P"),
                "grad_norm_beta_last": train_rec["curve"][-1].get("grad_norm_beta"),
                "prior_last_epoch": train_rec["curve"][-1].get("prior"),
                "prior_logged_curve_mean": float(np.mean(prior_curve)) if prior_curve else None,
                "prior_logged_curve_mean_note": "mean of the 20-epoch logged R values; the JSON field mean_R_per_step is 0.0 due to a frozen logging bug (see AMENDMENT.md)",
                "sh2": float(R.read_json(R.sh2_path(s))["sh2"]),
                "K": int(vocab["n_types"]),
            }
    R.write_json(R.MECH_JSON, mechanism)

    preds = {key: {s: np.load(PRED / f"valid_{key}_s{s}.npz")["pred"].astype(np.float64) for s in seeds} for key in configs}
    group_rows: dict[str, list[int]] = {}
    per_mol = []
    for i, obj in enumerate(valid_objs):
        labels = D.molecule_group_labels(
            obj, assigned[i], np.asarray(assigned[i]["support"]), float(y[i]), size_tercile_cuts=cuts
        )
        per_mol.append(labels)
        for name in (
            "ordinary", "tail", "large_ring",
            "low_support_none", "low_support_partial", "low_support_majority",
            "fallback_none", "fallback_partial", "fallback_high",
            "size_small", "size_medium", "size_large",
        ):
            if labels.get(name):
                group_rows.setdefault(name, []).append(i)
    groups: dict[str, Any] = {"protocol": R.PROTOCOL_ID, "thresholds": prep["group_thresholds"], "groups": {}}
    for name, rows in group_rows.items():
        if not rows:
            continue
        idx = np.asarray(rows)
        entry: dict[str, Any] = {"n": int(idx.size), "fraction": float(idx.size / R.N_VALID), "models": {}, "contrasts": {}}
        for key in configs:
            mae = {str(s): float(R._err(preds[key][s], y)[idx].mean()) for s in seeds}
            entry["models"][key] = {"mae_seed_mean": float(np.mean(list(mae.values()))), "per_seed": mae}
        for key in configs[1:]:
            entry["contrasts"][f"{key}_minus_B"] = R._contrast(
                [R._err(preds[key][s], y)[idx] for s in seeds],
                [R._err(preds["B"][s], y)[idx] for s in seeds],
            )
        groups["groups"][name] = entry
    groups["molecule_table_summary"] = {
        "low_support_fraction_mean": float(np.mean([m["low_support_fraction"] for m in per_mol])),
        "fallback_fraction_mean": float(np.mean([m["fallback_fraction"] for m in per_mol])),
        "zero_correction_fraction_mean": float(np.mean([m["zero_correction_fraction"] for m in per_mol])),
        "n_atoms_mean": float(np.mean([m["n_atoms"] for m in per_mol])),
        "group_sizes": {name: len(rows) for name, rows in group_rows.items()},
    }
    R.write_json(R.GROUPS_JSON, groups)

    ledger = build_ledger(configs, seeds)
    R.write_json(R.LEDGER_JSON, ledger)
    write_manifest(configs, seeds)
    R.log(f"[analyze] wrote mechanism/groups/ledger; phase-1 GPU {ledger['phase1_gpu_hours']:.4f} h")


def build_ledger(configs: Sequence[str], seeds: Sequence[int]) -> dict[str, Any]:
    rows = []
    total = 0.0
    for path in sorted(RUNS.glob("*_s*/train.json")):
        rec = R.read_json(path)
        rows.append({
            "run": rec["run_key"], "seed": rec["seed"], "stage": "train",
            "wall_seconds": rec["wall_seconds"], "gpu": rec.get("cuda_device_name", ""),
        })
        total += float(rec["wall_seconds"])
    predict_seconds = 0.0
    predict_rows = []
    for path in sorted(PRED.glob("valid_*_s*.json")):
        rec = R.read_json(path)
        w = float(rec.get("wall_seconds", 0.0))
        predict_seconds += w
        predict_rows.append({"run": rec["run_key"], "seed": rec["seed"], "wall_seconds": w})
    smoke = R.read_json(R.SMOKE_JSON) if R.SMOKE_JSON.exists() else None
    smoke_seconds = float(sum(r["wall_seconds"] for r in smoke["records"])) if smoke else 0.0
    ssl_rows = []
    for path in sorted(R.SSL_DIR.glob("pretraining_seed*.json")):
        rec = R.read_json(path)
        ssl_rows.append({"seed": int(rec["seed"]), "wall_seconds": float(rec["wall_seconds"])})
    ssl_total = float(sum(r["wall_seconds"] for r in ssl_rows))
    phase1_train = sum(
        float(R.read_json(RUNS / f"{k}_s{s}" / "train.json")["wall_seconds"])
        for k in configs for s in seeds
        if (RUNS / f"{k}_s{s}" / "train.json").exists()
    )
    phase1_predict = sum(
        float(R.read_json(PRED / f"valid_{k}_s{s}.json").get("wall_seconds", 0.0))
        for k in configs for s in seeds
        if (PRED / f"valid_{k}_s{s}.json").exists()
    )
    prep = R.read_json(R.PREP_JSON)
    return {
        "protocol": R.PROTOCOL_ID,
        "gpu_stages": rows,
        "predictions": predict_rows,
        "ssl_pretraining": ssl_rows,
        "smoke_seconds": smoke_seconds,
        "train_seconds": total,
        "predict_seconds_total": predict_seconds,
        "ssl_seconds": ssl_total,
        "gpu_seconds_total": total + predict_seconds + smoke_seconds + ssl_total,
        "gpu_hours_total": (total + predict_seconds + smoke_seconds + ssl_total) / 3600.0,
        "phase1_gpu_hours": (phase1_train + phase1_predict + smoke_seconds) / 3600.0,
        "cpu_preprocessing_seconds": prep["timing_seconds"]["total_prep"],
        "cpu_patch_key_seconds": prep["timing_seconds"]["patch_keys_both_splits"],
        "cap_gpu_hours": 8.0,
        "accounting": "sum of process wall on both GPUs, including smoke and prediction forwards; scheduler overhead not included",
    }


def write_manifest(configs: Sequence[str], seeds: Sequence[int]) -> None:
    lines = ["run_key\tseed\tarm\teta\tparams\tsoup_fit_mae\tsoup_sha256\twall_seconds\tpred_sha256\tvalid_mae"]
    decision = R.read_json(R.DECISION_JSON)
    metrics = R.read_json(R.SCORE_JSON)
    for key in configs:
        for s in seeds:
            rec = R.read_json(RUNS / f"{key}_s{s}" / "train.json")
            pred = R.read_json(PRED / f"valid_{key}_s{s}.json")
            mae = metrics["models"][key]["per_seed"][str(s)]["mae"]
            lines.append(
                "\t".join(
                    [
                        key, str(s), rec["arm"], "" if rec["eta"] is None else f"{rec['eta']:g}",
                        str(rec["parameter_audit"]["total_parameters"]), f"{rec['soup_fit_mae']:.9f}",
                        rec["soup_sha256"], f"{rec['wall_seconds']:.3f}", pred["pred_sha256"], f"{mae:.9f}",
                    ]
                )
            )
    lines.append("")
    lines.append(f"# decision phase 1: E={decision['triggers']['E']} H={decision['triggers']['H']}")
    (R.EXP_DIR / "run_manifest.tsv").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("score")
    p.add_argument("--phase", type=int, default=1, choices=[1, 2])
    p.set_defaults(func=lambda a: (score(int(a.phase)), 0)[1])
    sub.add_parser("analyze").set_defaults(func=lambda a: (analyze(a.device), 0)[1])
    p = sub.add_parser("all")
    p.add_argument("--phase", type=int, default=1, choices=[1, 2])
    p.set_defaults(func=lambda a: (score(int(a.phase)), analyze(a.device), 0)[2])
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
