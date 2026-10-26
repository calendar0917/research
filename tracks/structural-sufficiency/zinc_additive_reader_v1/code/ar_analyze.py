"""Freeze, one-shot Locked read, and frozen analysis.

Phase 1 (pilot, A_s0 + D_s0 only) only reads Monitor metrics and applies the
pre-registered stop rules; Locked is never touched.  When all four formal runs
exist, this script

1. freezes every checkpoint / prediction / config / protocol hash,
2. performs exactly one Locked read (predictions and A diagnostics are saved),
3. computes the pre-registered paired, group, tail and decision analyses.

Re-running the script after the one-shot read loads the saved Locked arrays and
never re-opens a checkpoint for Locked inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ar_common as C  # noqa: E402
import ar_models as M  # noqa: E402

for _p in (str(C.V1), str(C.DCC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_train as T1  # noqa: E402
import dcc_runner as DR  # noqa: E402

BEST = "best_monitor"
FINAL = "final"
MODELS = ("A", "D")
SEEDS = (0, 1)
DCC_TAG = {0: "C_s0", 1: "C_s1"}


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def _sha(path: Path) -> str | None:
    return C.sha256_file(path) if path.exists() else None


def _bootstrap_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)


def paired_bootstrap(ae_a: np.ndarray, ae_b: np.ndarray, label: str,
                     n_boot: int = 10000) -> dict:
    ae_a = np.asarray(ae_a, dtype=np.float64).reshape(-1)
    ae_b = np.asarray(ae_b, dtype=np.float64).reshape(-1)
    n = len(ae_a)
    rng = np.random.default_rng(_bootstrap_seed(label))
    means = np.empty(n_boot, dtype=np.float64)
    chunk = 2000
    for start in range(0, n_boot, chunk):
        k = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(k, n))
        means[start:start + k] = ae_a[idx].mean(axis=1) - ae_b[idx].mean(axis=1)
    return {
        "delta_mae": float(ae_a.mean() - ae_b.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "frac_a_better": float(np.mean(ae_a < ae_b)),
        "median_delta_ae": float(np.median(ae_a - ae_b)),
        "n": int(n),
        "n_boot": int(n_boot),
        "bootstrap_seed": _bootstrap_seed(label),
    }


def top10_sensitivity(ae_a: np.ndarray, ae_b: np.ndarray) -> dict:
    ae_a = np.asarray(ae_a, dtype=np.float64)
    ae_b = np.asarray(ae_b, dtype=np.float64)
    order = np.argsort(-np.maximum(ae_a, ae_b))
    keep = np.ones(len(ae_a), dtype=bool)
    keep[order[:10]] = False
    return {
        "delta_mae_drop_top10": float(ae_a[keep].mean() - ae_b[keep].mean()),
        "mae_a_drop_top10": float(ae_a[keep].mean()),
        "mae_b_drop_top10": float(ae_b[keep].mean()),
        "dropped_rows": [int(i) for i in order[:10]],
    }


# ---------------------------------------------------------------------------
# graph construction for Fit8k support (exact typed radius-2 signatures)
# ---------------------------------------------------------------------------
def molecule_graphs(cache: dict, rows: np.ndarray):
    at = np.asarray(cache["atom_type"], dtype=np.int64)
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)
    out = []
    for i in np.asarray(rows, dtype=np.int64):
        a0, a1 = int(ap[i]), int(ap[i + 1])
        b0, b1 = int(bp[i]), int(bp[i + 1])
        atoms = at[a0:a1]
        bonds = [(int(bi[e, 0]), int(bi[e, 1]), int(bt[e])) for e in range(b0, b1)]
        out.append((atoms, bonds))
    return out


def radius2_signature_sets(graphs):
    sets, per_atom = [], []
    for atoms, bonds in graphs:
        nbrs = [[] for _ in range(len(atoms))]
        for u, v, t in bonds:
            nbrs[u].append((t, v))
            nbrs[v].append((t, u))
        one = [
            f"r1({int(atoms[i])};" + ",".join(
                sorted(f"{t}-{int(atoms[j])}" for t, j in nbrs[i])) + ")"
            for i in range(len(atoms))
        ]
        two = [
            f"r2({one[i]};" + ",".join(
                sorted(f"{t}-{one[j]}" for t, j in nbrs[i])) + ")"
            for i in range(len(atoms))
        ]
        sets.append(set(two))
        per_atom.append(two)
    return sets, per_atom


def locked_support(cache: dict, fit_rows: np.ndarray, locked_rows: np.ndarray) -> dict:
    fit_graphs = molecule_graphs(cache, fit_rows)
    locked_graphs = molecule_graphs(cache, locked_rows)
    fit_sets, _ = radius2_signature_sets(fit_graphs)
    freq = Counter()
    for s in fit_sets:
        for k in s:
            freq[k] += 1
    _, atom_sigs = radius2_signature_sets(locked_graphs)
    unseen_frac, low_frac, supported_frac, counts = [], [], [], []
    for sigs in atom_sigs:
        c = [freq.get(s, 0) for s in sigs]
        n = max(len(c), 1)
        counts.append(c)
        unseen_frac.append(float(np.mean([x == 0 for x in c])) if c else 0.0)
        low_frac.append(float(np.mean([1 <= x <= 4 for x in c])) if c else 0.0)
        supported_frac.append(float(np.mean([x >= 5 for x in c])) if c else 0.0)
    groups = []
    for u, s in zip(unseen_frac, supported_frac):
        if s >= 1.0 - 1e-12:
            groups.append("all_supported")
        elif u == 0.0:
            groups.append("low_support")
        else:
            groups.append("any_unseen")
    return {
        "unseen_frac": np.asarray(unseen_frac),
        "low4_frac": np.asarray(low_frac),
        "supported_frac": np.asarray(supported_frac),
        "min_support": np.asarray([min(c) if c else 0 for c in counts]),
        "median_support": np.asarray([float(np.median(c)) if c else 0.0 for c in counts]),
        "groups": np.asarray(groups),
        "n_fit_molecules": int(len(fit_rows)),
        "n_unique_fit_environments": int(len(freq)),
        "group_counts": {k: int(v) for k, v in Counter(groups).items()},
    }


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------
def run_tags() -> list[str]:
    out = []
    for arm in MODELS:
        for seed in SEEDS:
            tag = f"{arm}_s{seed}"
            if (C.RUNS / tag / "train.json").exists():
                out.append(tag)
    return out


def freeze(runs: list[str]) -> dict:
    path = C.ANALYSIS / "freeze.json"
    files: dict = {}
    for tag in runs:
        d = C.RUNS / tag
        files[tag] = {
            name: _sha(d / name)
            for name in ("best_monitor.pt", "final.pt", "predictions_best_monitor.pt",
                         "predictions_final.pt", "train.json", "curve.csv")
        }
    hist = {}
    for seed in SEEDS:
        d = C.DCCL_RUNS / f"fit8k_s{seed}"
        hist[f"C_s{seed}"] = {
            name: _sha(d / name)
            for name in ("best_monitor.pt", "final.pt", "predictions_best_monitor.pt",
                         "train.json")
        }
    payload = {
        "protocol_id": C.load_config()["protocol_id"],
        "frozen_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "config_sha256": _sha(C.CONFIG),
        "protocol_md_sha256": _sha(C.ROUND / "PROTOCOL.md"),
        "new_runs": files,
        "historical_runs": hist,
        "locked_read": False,
        "official_test_loaded": False,
        "official_valid_loaded": False,
    }
    if path.exists():
        old = C.read_json(path)
        for key in ("new_runs", "historical_runs", "config_sha256", "protocol_md_sha256"):
            if old.get(key) != payload.get(key):
                raise RuntimeError(
                    f"freeze mismatch in {key}: checkpoints or configuration changed after freeze"
                )
        return old
    C.write_json(path, payload)
    return payload


# ---------------------------------------------------------------------------
# one-shot Locked read
# ---------------------------------------------------------------------------
@torch.no_grad()
def _locked_forward(model, cache, rows, device, *, additive: bool) -> dict:
    model.eval()
    preds, wS, psi, snorm = [], [], [], []
    for start in range(0, len(rows), 256):
        chunk = np.asarray(rows[start:start + 256], dtype=np.int64)
        batch = T1._to_device(DR.collate_p3n(cache, chunk, "P3N", valid=True), device)
        comp = model.local_components(batch)
        y = model.additive_terms(comp).view(-1) if additive else model(batch)
        preds.append(y.detach().float().cpu().numpy())
        if additive:
            wS.append(model.local_term(comp).detach().float().cpu().numpy())
            psi.append(model.global_term(comp).detach().float().cpu().numpy())
            snorm.append(comp["S"].detach().float().cpu().norm(dim=1).numpy())
    out = {"pred": np.concatenate(preds).astype(np.float64)}
    if additive:
        out["wS"] = np.concatenate(wS).astype(np.float64)
        out["psi"] = np.concatenate(psi).astype(np.float64)
        out["S_norm"] = np.concatenate(snorm).astype(np.float64)
    return out


def one_shot_locked(runs: list[str], cache, rows: np.ndarray, freeze_info: dict) -> dict:
    meta_path = C.ANALYSIS / "locked_read.json"
    diag_dir = C.ANALYSIS / "locked_preds"
    diag_dir.mkdir(parents=True, exist_ok=True)
    if meta_path.exists():
        meta = C.read_json(meta_path)
        if meta.get("read_count") != 1:
            raise RuntimeError("locked_read.json exists but read_count != 1")
        return meta

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    norm = C.fit_only_norm(cache, C.split_rows()["fit8k"])
    payload = {
        "read_count": 1,
        "read_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "config_sha256": _sha(C.CONFIG),
        "device": str(device),
        "rows": int(len(rows)),
        "rows_sha256": C.sha256_array(rows, "<i8"),
        "files": {},
        "official_test_loaded": False,
        "official_valid_loaded": False,
    }
    payload["reused_existing"] = []
    payload["computed_this_invocation"] = []
    for tag in runs:
        arm = tag.split("_")[0]
        seed = int(tag.split("_s")[1])
        train = C.read_json(C.RUNS / tag / "train.json")
        for ck_tag in (BEST, FINAL):
            ck_path = (C.RUNS / tag / "best_monitor.pt" if ck_tag == BEST
                       else C.RUNS / tag / "final.pt")
            f = diag_dir / f"{tag}_{ck_tag}.npz"
            key = f"{tag}_{ck_tag}"
            if f.exists():
                z = np.load(f, allow_pickle=False)
                missing = {"pred", "rows", "checkpoint"} - set(z.files)
                if missing:
                    raise RuntimeError(f"existing Locked file {f} is incomplete: {missing}")
                if not np.array_equal(np.asarray(z["rows"]), rows):
                    raise RuntimeError(f"existing Locked file {f} has different rows")
                payload["reused_existing"].append(key)
                payload["files"][key] = {
                    "path": str(f.relative_to(C.REPO)),
                    "sha256": _sha(f),
                    "checkpoint_sha256": _sha(ck_path),
                    "reused_existing": True,
                }
                continue
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            model = M.build_arm(arm, norm, seed, device)[0]
            model.load_state_dict(ck["state_dict"])
            model.to(device).eval()
            out = _locked_forward(model, cache, rows, device, additive=(arm == "A"))
            out["rows"] = rows
            out["checkpoint"] = ck_tag
            out["checkpoint_epoch"] = int(ck["epoch"])
            out["checkpoint_step"] = int(ck["step"])
            out["checkpoint_monitor_mae"] = float(
                ck["monitor_mae"] if "monitor_mae" in ck
                else train["eval_reads"][ck_tag]["monitor_mae"])
            np.savez_compressed(f, **out)
            payload["computed_this_invocation"].append(key)
            payload["files"][key] = {
                "path": str(f.relative_to(C.REPO)),
                "sha256": _sha(f),
                "checkpoint_sha256": _sha(ck_path),
                "reused_existing": False,
            }
    C.write_json(meta_path, payload)
    return payload


def load_locked(tag: str, ck_tag: str = BEST) -> dict:
    f = C.ANALYSIS / "locked_preds" / f"{tag}_{ck_tag}.npz"
    if not f.exists():
        raise FileNotFoundError(f"missing one-shot Locked array {f}")
    z = np.load(f, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# metrics / comparisons
# ---------------------------------------------------------------------------
def model_entries(runs: list[str], cache, locked: np.ndarray, y_locked: np.ndarray,
                  support: dict) -> dict:
    entries = {}
    for tag in runs:
        train = C.read_json(C.RUNS / tag / "train.json")
        arm = tag.split("_")[0]
        entry = {
            "tag": tag, "arm": arm, "seed": int(train["seed"]),
            "parameters": int(train["parameters"]),
            "monitor_mae": float(train["eval_reads"][BEST]["monitor_mae"]),
            "monitor_mae_final": float(train["eval_reads"][FINAL]["monitor_mae"]),
            "train_fit_mae_best": float(train["eval_reads"][BEST]["train_fit_mae_eval_mode"]),
            "train_fit_mae_final": float(train["eval_reads"][FINAL]["train_fit_mae_eval_mode"]),
            "best_epoch": int(train["best_monitor"]["epoch"]),
            "best_step": int(train["best_monitor"]["step"]),
            "gpu_hours": float(train["gpu_hours"]),
            "wall_seconds": float(train["wall_seconds"]),
            "host": train.get("host"),
            "device_name": train.get("device_name"),
            "git_commit": train.get("git_commit"),
            "norm_hash": train.get("norm_hash"),
            "total_updates_run": int(train["total_updates_run"]),
            "curve_len": int(train["epochs_run"]),
            "official_test_loaded": bool(train["official_test_loaded"]),
            "official_valid_loaded": bool(train["official_valid_loaded"]),
            "locked_read_during_training": bool(train["locked_read"]),
        }
        for ck_tag in (BEST, FINAL):
            z = load_locked(tag, ck_tag)
            pred = np.asarray(z["pred"], dtype=np.float64)
            ae = np.abs(pred - y_locked)
            entry[f"locked_mae_{ck_tag}"] = float(ae.mean())
            entry[f"locked_ae_{ck_tag}"] = ae
            entry[f"locked_pred_{ck_tag}"] = pred
            if arm == "A":
                entry[f"wS_{ck_tag}"] = np.asarray(z["wS"], dtype=np.float64)
                entry[f"psi_{ck_tag}"] = np.asarray(z["psi"], dtype=np.float64)
                entry[f"S_norm_{ck_tag}"] = np.asarray(z["S_norm"], dtype=np.float64)
        # Canonical aliases used by the comparison/decision code.
        entry["locked_mae_best"] = entry["locked_mae_best_monitor"]
        entry["locked_ae_best"] = entry["locked_ae_best_monitor"]
        entry["locked_pred_best"] = entry["locked_pred_best_monitor"]
        entries[tag] = entry
    return entries


def _group_masks(support: dict, y_locked: np.ndarray) -> dict:
    g = support["groups"]
    return {
        "all": np.ones(len(g), dtype=bool),
        "all_supported": g == "all_supported",
        "low_support": g == "low_support",
        "any_unseen": g == "any_unseen",
        "ordinary_y_gt_-3": y_locked > -3.0,
        "extreme_tail_y_le_-3": y_locked <= -3.0,
    }


def _pair_record(name: str, ae_a: np.ndarray, ae_b: np.ndarray,
                 masks: dict) -> dict:
    rec = {"label": name, "overall": paired_bootstrap(ae_a, ae_b, name)}
    rec["overall"]["top10"] = top10_sensitivity(ae_a, ae_b)
    rec["groups"] = {}
    for gname, mask in masks.items():
        if gname == "all":
            continue
        rec["groups"][gname] = paired_bootstrap(ae_a[mask], ae_b[mask], f"{name}:{gname}")
        rec["groups"][gname]["n"] = int(mask.sum())
    return rec


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------
def decide(entries: dict, runs: list[str], masks: dict, support: dict) -> dict:
    seeds = sorted({int(entries[t]["seed"]) for t in runs if entries[t]["arm"] in MODELS})
    reasons = []
    collapsed = {}
    for seed in seeds:
        e = entries.get(f"A_s{seed}")
        if e is None:
            collapsed[seed] = None
            continue
        wS = e[f"wS_{BEST}"]; psi = e[f"psi_{BEST}"]
        collapsed[seed] = {
            "std_wS": float(np.std(wS)),
            "std_psi": float(np.std(psi)),
            "ratio_std_wS_over_psi": float(np.std(wS) / max(np.std(psi), 1e-12)),
            "local_share_mean": float(np.mean(np.abs(wS)) /
                                      max(np.mean(np.abs(wS)) + np.mean(np.abs(psi)), 1e-12)),
        }
    # explicit zero-local MAE needs labels; computed by caller and injected under
    # entries[tag]["locked_mae_zero_local_best"]
    for seed in seeds:
        e = entries.get(f"A_s{seed}")
        if e is None:
            continue
        collapsed[seed]["mae_zero_local"] = float(e.get("locked_mae_zero_local_best", float("nan")))
        collapsed[seed]["delta_zero_local"] = float(
            e.get("locked_mae_zero_local_best", float("nan")) - e["locked_mae_best"]
        )
        collapsed[seed]["collapsed"] = bool(
            collapsed[seed]["ratio_std_wS_over_psi"] < 0.05
            and abs(collapsed[seed]["delta_zero_local"]) < 0.001
        )

    def mean_mae(tag_prefix: str, key: str = "locked_mae_best") -> float:
        vals = [entries[f"{tag_prefix}_s{s}"][key] for s in seeds
                if f"{tag_prefix}_s{s}" in entries]
        return float(np.mean(vals))

    a_mean, d_mean, c_mean = mean_mae("A"), mean_mae("D"), mean_mae("C")
    gain_a_vs_d = d_mean - a_mean
    gain_a_vs_c = c_mean - a_mean
    gain_d_vs_c = c_mean - d_mean
    per_seed_a_better_d = all(entries[f"A_s{s}"]["locked_mae_best"]
                              < entries[f"D_s{s}"]["locked_mae_best"] for s in seeds)
    per_seed_a_better_c = all(entries[f"A_s{s}"]["locked_mae_best"]
                              < entries[f"C_s{s}"]["locked_mae_best"] for s in seeds)

    ordinary_gain_c = []
    ordinary_gain_d = []
    top10_gain_c = []
    top10_gain_d = []
    for s in seeds:
        a = entries[f"A_s{s}"]; d = entries[f"D_s{s}"]; c = entries[f"C_s{s}"]
        m = masks["ordinary_y_gt_-3"]
        ordinary_gain_c.append(float(c["locked_ae_best"][m].mean() - a["locked_ae_best"][m].mean()))
        ordinary_gain_d.append(float(d["locked_ae_best"][m].mean() - a["locked_ae_best"][m].mean()))
        top10_gain_c.append(top10_sensitivity(c["locked_ae_best"], a["locked_ae_best"])["delta_mae_drop_top10"])
        top10_gain_d.append(top10_sensitivity(d["locked_ae_best"], a["locked_ae_best"])["delta_mae_drop_top10"])
    ordinary_gain_c_mean = float(np.mean(ordinary_gain_c))
    ordinary_gain_d_mean = float(np.mean(ordinary_gain_d))
    # top10_sensitivity(reference, candidate) already returns reference - candidate,
    # i.e. a positive number when A is better; do not negate it again.
    top10_gain_c_mean = float(np.mean(top10_gain_c))
    top10_gain_d_mean = float(np.mean(top10_gain_d))

    group_gains = {}
    for gname in ("low_support", "any_unseen"):
        gains_c = [entries[f"C_s{s}"]["locked_ae_best"][masks[gname]].mean()
                   - entries[f"A_s{s}"]["locked_ae_best"][masks[gname]].mean() for s in seeds]
        gains_d = [entries[f"D_s{s}"]["locked_ae_best"][masks[gname]].mean()
                   - entries[f"A_s{s}"]["locked_ae_best"][masks[gname]].mean() for s in seeds]
        group_gains[gname] = {
            "gain_vs_C_per_seed": [float(x) for x in gains_c],
            "gain_vs_D_per_seed": [float(x) for x in gains_d],
            "gain_vs_C_mean": float(np.mean(gains_c)),
            "gain_vs_D_mean": float(np.mean(gains_d)),
            "n": int(masks[gname].sum()),
            "stable": bool(all(x > 0 for x in gains_c) and all(x > 0 for x in gains_d)
                           and np.mean(gains_c) >= 0.003 and np.mean(gains_d) >= 0.003),
        }

    any_collapse = any(v is not None and v.get("collapsed") for v in collapsed.values())
    promising_group = any(v["stable"] for v in group_gains.values())
    promising = bool(
        per_seed_a_better_d and per_seed_a_better_c
        and gain_a_vs_d >= 0.003 and gain_a_vs_c >= 0.003
        and ordinary_gain_c_mean >= 0.002 and ordinary_gain_d_mean >= 0.002
        and top10_gain_c_mean > 0 and top10_gain_d_mean > 0
        and promising_group
        and not any_collapse
    )
    delta_a_d = a_mean - d_mean
    delta_a_c = a_mean - c_mean
    additivity_bad = bool(any_collapse or delta_a_d >= 0.003 or delta_a_c >= 0.003)
    capacity = bool(
        not promising and not additivity_bad
        and gain_a_vs_c >= 0.003 and gain_d_vs_c >= 0.003
        and (abs(delta_a_d) < 0.003
             or not all((entries[f"A_s{s}"]["locked_mae_best"]
                         - entries[f"D_s{s}"]["locked_mae_best"]) > 0 for s in seeds))
    )
    if promising:
        verdict = "PROMISING_READER_BIAS"
    elif additivity_bad:
        verdict = "ADDITIVITY_NOT_SUPPORTED"
    elif capacity:
        verdict = "CAPACITY_ONLY"
    else:
        verdict = "INCONCLUSIVE"

    reasons.append(f"A_mean={a_mean:.6f} D_mean={d_mean:.6f} C_mean={c_mean:.6f}")
    reasons.append(f"gain_A_vs_D={gain_a_vs_d:.6f} gain_A_vs_C={gain_a_vs_c:.6f} "
                   f"gain_D_vs_C={gain_d_vs_c:.6f}")
    reasons.append(f"per-seed A<D={per_seed_a_better_d} A<C={per_seed_a_better_c}; "
                   f"ordinary gains vs C/D={ordinary_gain_c_mean:.6f}/{ordinary_gain_d_mean:.6f}; "
                   f"top10 gains vs C/D={top10_gain_c_mean:.6f}/{top10_gain_d_mean:.6f}")
    reasons.append(f"group stability={ {k: v['stable'] for k, v in group_gains.items()} }; "
                   f"any_collapse={any_collapse}")
    return {
        "verdict": verdict,
        "reasons": reasons,
        "inputs": {
            "seeds": seeds,
            "A_mean": a_mean, "D_mean": d_mean, "C_mean": c_mean,
            "gain_A_vs_D": gain_a_vs_d, "gain_A_vs_C": gain_a_vs_c,
            "gain_D_vs_C": gain_d_vs_c,
            "per_seed_A_better_D": per_seed_a_better_d,
            "per_seed_A_better_C": per_seed_a_better_c,
            "ordinary_gain_vs_C_mean": ordinary_gain_c_mean,
            "ordinary_gain_vs_D_mean": ordinary_gain_d_mean,
            "top10_gain_vs_C_mean": top10_gain_c_mean,
            "top10_gain_vs_D_mean": top10_gain_d_mean,
            "group_gains": group_gains,
            "collapse": collapsed,
        },
        "rules": C.load_config()["decision_rules"],
    }


# ---------------------------------------------------------------------------
# report tables
# ---------------------------------------------------------------------------
def write_tables(summary: dict) -> None:
    lines = []
    lines.append("# zinc_additive_reader_v1 — machine-generated main tables\n")
    lines.append("Frozen checkpoints only. MAE is raw-y L1, lower is better; "
                 "delta = candidate - reference (negative = candidate better).\n")
    lines.append("## Per-run metrics\n")
    lines.append("| run | params | Monitor | Locked (best) | Locked (final) | "
                 "train@best | best ep | GPU-h |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for tag, e in summary["runs"].items():
        lines.append(f"| {tag} | {e['parameters']} | {e['monitor_mae']:.6f} | "
                     f"{e['locked_mae_best']:.6f} | {e['locked_mae_final']:.6f} | "
                     f"{e['train_fit_mae_best']:.6f} | {e['best_epoch']} | {e['gpu_hours']:.4f} |")
    lines.append("\n## Locked groups (best Monitor checkpoint)\n")
    lines.append("| run | all | all-supported | low-support | any-unseen | ordinary | tail y<=-3 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for tag, e in summary["runs"].items():
        g = e["locked_group_mae_best"]
        lines.append(f"| {tag} | {g['all']:.6f} | {g['all_supported']:.6f} | "
                     f"{g['low_support']:.6f} | {g['any_unseen']:.6f} | "
                     f"{g['ordinary_y_gt_-3']:.6f} | {g['extreme_tail_y_le_-3']:.6f} |")
    lines.append("\n## Paired comparisons (best checkpoint)\n")
    lines.append("| comparison | scope | delta MAE | 95% bootstrap CI | frac molecules better | n |")
    lines.append("|---|---|---:|---|---:|---:|")
    for name, rec in summary["comparisons"].items():
        for scope in ("overall", "all_supported", "low_support", "any_unseen",
                      "ordinary_y_gt_-3", "extreme_tail_y_le_-3"):
            r = rec["overall"] if scope == "overall" else rec["groups"].get(scope)
            if r is None:
                continue
            lines.append(f"| {name} | {scope} | {r['delta_mae']:+.6f} | "
                         f"[{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}] | "
                         f"{r['frac_a_better']:.3f} | {r['n']} |")
    lines.append("\n## A pathway diagnostics (Locked, best checkpoint)\n")
    lines.append("| run | std(wS) | std(psi) | corr(wS,y) | corr(psi,y) | "
                 "local share | MAE full | MAE zero-local | delta | collapsed |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for tag, d in summary["A_diagnostics"].items():
        lines.append(f"| {tag} | {d['std_wS']:.4f} | {d['std_psi']:.4f} | "
                     f"{d['corr_wS_y']:+.4f} | {d['corr_psi_y']:+.4f} | "
                     f"{d['local_share_mean']:.4f} | {d['mae_full']:.6f} | "
                     f"{d['mae_zero_local']:.6f} | {d['delta_zero_local']:+.6f} | "
                     f"{d['collapsed']} |")
    lines.append(f"\n## Decision\n\n**{summary['decision']['verdict']}**\n")
    for r in summary["decision"]["reasons"]:
        lines.append(f"- {r}")
    lines.append("")
    (C.ANALYSIS / "tables.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# pilot
# ---------------------------------------------------------------------------
def pilot() -> dict:
    cfg = C.load_config()
    tags = run_tags()
    if not all(t in tags for t in ("A_s0", "D_s0")):
        raise SystemExit("pilot needs A_s0 and D_s0 train.json")
    a = C.read_json(C.RUNS / "A_s0" / "train.json")
    d = C.read_json(C.RUNS / "D_s0" / "train.json")
    c = C.read_json(C.DCCL_RUNS / "fit8k_s0" / "train.json")
    am = float(a["eval_reads"][BEST]["monitor_mae"])
    dm = float(d["eval_reads"][BEST]["monitor_mae"])
    cm = float(c["best_monitor"]["monitor_mae"])
    stop_1 = (am > dm + 0.005) and (am > cm + 0.010)
    stop_2 = (am > cm + 0.010) and (dm > cm + 0.010)
    stop = bool(stop_1 or stop_2)
    payload = {
        "stage": "pilot",
        "monitor_only": True,
        "locked_read": False,
        "official_test_loaded": False,
        "official_valid_loaded": False,
        "A_s0_monitor": am, "D_s0_monitor": dm, "DCCL_s0_monitor": cm,
        "A_minus_D": am - dm, "A_minus_DCCL": am - cm, "D_minus_DCCL": dm - cm,
        "stop_rule_1_fires": bool(stop_1),
        "stop_rule_2_fires": bool(stop_2),
        "stop": stop,
        "next": "stop (no seed 1, no Locked read)" if stop else "train A_s1 and D_s1",
        "gpu_hours": float(a["gpu_hours"] + d["gpu_hours"]),
        "runs_present": tags,
    }
    C.write_json(C.ANALYSIS / "pilot.json", payload)
    C.log(json.dumps(payload, indent=2))
    return payload


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot", action="store_true",
                   help="only evaluate the two-run Monitor pilot and exit")
    args = p.parse_args(argv)

    tags = run_tags()
    if args.pilot or not all(t in tags for t in ("A_s0", "D_s0", "A_s1", "D_s1")):
        pilot()
        return 0

    cfg = C.load_config()
    cache = C.load_train_cache()
    sp = C.split_rows()
    y_locked = np.asarray(cache["labels"], dtype=np.float64)[sp["locked"]]
    support = locked_support(cache, sp["fit8k"], sp["locked"])

    freeze_info = freeze(tags)
    locked_meta = one_shot_locked(tags, cache, sp["locked"], freeze_info)

    entries = model_entries(tags, cache, sp["locked"], y_locked, support)

    # historical DCC-L frozen predictions (bootstrap only; never retrained)
    hist_entries = {}
    for seed in SEEDS:
        tag = DCC_TAG[seed]
        pred_file = C.DCCL_RUNS / f"fit8k_s{seed}" / "predictions_best_monitor.pt"
        z = torch.load(pred_file, map_location="cpu", weights_only=False)
        if not np.array_equal(np.asarray(z["locked_rows"]), sp["locked"]):
            raise RuntimeError("historical Locked rows differ from the frozen split")
        pred = np.asarray(z["locked_pred"], dtype=np.float64)
        ae = np.abs(pred - y_locked)
        mon_pred = np.asarray(z["monitor_pred"], dtype=np.float64)
        mon_rows = np.asarray(z["monitor_rows"], dtype=np.int64)
        mon_ae = np.abs(mon_pred - np.asarray(cache["labels"])[mon_rows])
        hist_train = C.read_json(C.DCCL_RUNS / f"fit8k_s{seed}" / "train.json")
        hist_entries[tag] = {
            "tag": tag, "arm": "C", "seed": seed,
            "parameters": int(cfg["baseline_history"]["parameters"]),
            "monitor_mae": float(mon_ae.mean()),
            "monitor_mae_final": float(hist_train["eval_reads"]["final"]["monitor_mae"]),
            "train_fit_mae_best": float("nan"),
            "train_fit_mae_final": float("nan"),
            "best_epoch": int(hist_train["best_monitor"]["epoch"]),
            "best_step": int(hist_train["best_monitor"]["step"]),
            "gpu_hours": float(hist_train["gpu_hours"]),
            "wall_seconds": float(hist_train["wall_seconds"]),
            "host": "res (historical, not retrained)",
            "device_name": "NVIDIA A100-SXM4-40GB",
            "git_commit": hist_train.get("git_commit", "historical"),
            "norm_hash": "historical dcb_train fit8k norm",
            "total_updates_run": int(hist_train["total_updates"]),
            "curve_len": int(hist_train["epochs_run"]),
            "official_test_loaded": bool(hist_train["official_test_loaded"]),
            "official_valid_loaded": True,
            "locked_read_during_training": True,
            "locked_mae_best": float(ae.mean()),
            "locked_ae_best": ae,
            "locked_pred_best": pred,
            "predictions_sha256": _sha(pred_file),
            "note": "frozen historical DCC-L best-Monitor checkpoint; no retraining; "
                    "that round also read official valid descriptively, this round never does",
        }
        expected = float(cfg["baseline_history"]["fit8k_locked_mae"][str(seed)])
        # Recomputing from the stored per-molecule predictions can differ from
        # the metric stored in train.json by a few ulps (~5e-9); 1e-7 is still
        # far tighter than any scientific effect of interest.
        if abs(float(ae.mean()) - expected) > 1e-7:
            raise RuntimeError(f"historical Locked MAE mismatch for seed {seed}")
        expected_monitor = float(cfg["baseline_history"]["fit8k_monitor_mae"][str(seed)])
        if abs(float(mon_ae.mean()) - expected_monitor) > 1e-7:
            raise RuntimeError(f"historical Monitor MAE mismatch for seed {seed}")
    entries.update(hist_entries)
    # historical final is not part of the frozen reference table; use best for both
    for tag, e in hist_entries.items():
        e["locked_mae_final"] = e["locked_mae_best"]
        e["locked_ae_final"] = e["locked_ae_best"]

    masks = _group_masks(support, y_locked)
    # per-run group MAEs and A diagnostics
    for tag, e in entries.items():
        if "locked_ae_best" not in e:
            continue
        e["locked_group_mae_best"] = {
            gname: float(e["locked_ae_best"][mask].mean())
            for gname, mask in masks.items()
        }
        if e["arm"] == "A":
            y = y_locked
            wS = e[f"wS_{BEST}"]; psi = e[f"psi_{BEST}"]; ae = e[f"locked_ae_{BEST}"]
            e["locked_mae_zero_local_best"] = float(np.abs(psi - y).mean())
            diag = {
                "std_wS": float(np.std(wS)),
                "std_psi": float(np.std(psi)),
                "mean_wS": float(np.mean(wS)),
                "mean_psi": float(np.mean(psi)),
                "corr_wS_y": float(np.corrcoef(wS, y)[0, 1]),
                "corr_psi_y": float(np.corrcoef(psi, y)[0, 1]),
                "corr_wS_psi": float(np.corrcoef(wS, psi)[0, 1]),
                "local_share_mean": float(np.mean(np.abs(wS)) /
                                          max(np.mean(np.abs(wS)) + np.mean(np.abs(psi)), 1e-12)),
                "mae_full": float(ae.mean()),
                "mae_zero_local": float(np.abs(psi - y).mean()),
                "delta_zero_local": float(np.abs(psi - y).mean() - ae.mean()),
                "S_norm_mean": float(np.mean(e[f"S_norm_{BEST}"])),
                "ratio_std_wS_over_psi": float(np.std(wS) / max(np.std(psi), 1e-12)),
            }
            diag["collapsed"] = bool(diag["ratio_std_wS_over_psi"] < 0.05
                                     and abs(diag["delta_zero_local"]) < 0.001)
            e["A_diagnostics"] = diag

    # comparisons
    comparisons = {}
    for seed in SEEDS:
        comparisons[f"A_s{seed} vs D_s{seed}"] = _pair_record(
            f"A_s{seed} vs D_s{seed}",
            entries[f"A_s{seed}"]["locked_ae_best"], entries[f"D_s{seed}"]["locked_ae_best"], masks)
        comparisons[f"A_s{seed} vs DCCL_s{seed}"] = _pair_record(
            f"A_s{seed} vs DCCL_s{seed}",
            entries[f"A_s{seed}"]["locked_ae_best"], entries[f"C_s{seed}"]["locked_ae_best"], masks)
        comparisons[f"D_s{seed} vs DCCL_s{seed}"] = _pair_record(
            f"D_s{seed} vs DCCL_s{seed}",
            entries[f"D_s{seed}"]["locked_ae_best"], entries[f"C_s{seed}"]["locked_ae_best"], masks)
    # seed means (paired by seed only for descriptive bootstrap)
    for arm, label in (("A", "A mean vs D mean"), ("A", "A mean vs DCCL mean"),
                       ("D", "D mean vs DCCL mean")):
        pass
    ae_a = np.mean([entries[f"A_s{s}"]["locked_ae_best"] for s in SEEDS], axis=0)
    ae_d = np.mean([entries[f"D_s{s}"]["locked_ae_best"] for s in SEEDS], axis=0)
    ae_c = np.mean([entries[f"C_s{s}"]["locked_ae_best"] for s in SEEDS], axis=0)
    comparisons["seed-mean A vs D"] = _pair_record("seed-mean A vs D", ae_a, ae_d, masks)
    comparisons["seed-mean A vs DCCL"] = _pair_record("seed-mean A vs DCCL", ae_a, ae_c, masks)
    comparisons["seed-mean D vs DCCL"] = _pair_record("seed-mean D vs DCCL", ae_d, ae_c, masks)

    decision = decide(entries, [t for t in tags], masks, support)
    # attach final-checkpoint auxiliary (descriptive only)
    final_aux = {}
    for tag, e in entries.items():
        if f"locked_mae_{FINAL}" in e:
            final_aux[tag] = {
                "locked_mae_final": e[f"locked_mae_{FINAL}"],
                "monitor_mae_final": e.get("monitor_mae_final"),
            }

    total_gpu_hours = float(sum(e.get("gpu_hours", 0.0) for e in entries.values()
                                if e["arm"] in MODELS))
    smoke_files = sorted((C.RUNS_SMOKE).glob("*/smoke.json")) if C.RUNS_SMOKE.exists() else []
    smoke_gpu_hours = 0.0
    smoke_records = []
    for f in smoke_files:
        try:
            rec = C.read_json(f)
            smoke_gpu_hours += float(rec.get("gpu_hours", 0.0))
            smoke_records.append({"path": str(f.relative_to(C.REPO)),
                                  "gpu_hours": float(rec.get("gpu_hours", 0.0)),
                                  "projected_gpu_hours_per_run": float(
                                      rec.get("projected_gpu_hours_per_run", 0.0))})
        except Exception:
            pass
    summary = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "config_sha256": _sha(C.CONFIG),
        "protocol_md_sha256": _sha(C.ROUND / "PROTOCOL.md"),
        "freeze": freeze_info,
        "locked_read": locked_meta,
        "split": {
            "hashes": sp["hashes"],
            "fit8k": int(len(sp["fit8k"])), "monitor": int(len(sp["monitor"])),
            "locked": int(len(sp["locked"])),
        },
        "support": {
            "group_counts": support["group_counts"],
            "n_fit_molecules": support["n_fit_molecules"],
            "n_unique_fit_environments": support["n_unique_fit_environments"],
            "definition": cfg["analysis"]["locked_groups"],
        },
        "locked_label_summary": {
            "mean": float(y_locked.mean()), "sd": float(y_locked.std()),
            "n_tail_y_le_-3": int((y_locked <= -3).sum()),
        },
        "runs": {k: {kk: vv for kk, vv in v.items()
                     if not kk.startswith(("locked_ae_", "locked_pred_", "wS_", "psi_", "S_norm_"))}
                 for k, v in entries.items()},
        "A_diagnostics": {t: e["A_diagnostics"] for t, e in entries.items()
                          if e["arm"] == "A"},
        "comparisons": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                        for k, v in comparisons.items()},
        "final_checkpoint_auxiliary": final_aux,
        "decision": decision,
        "budget": {
            "hard_cap_gpu_hours": float(cfg["budget"]["gpu_hours_hard_cap"]),
            "formal_gpu_hours": total_gpu_hours,
            "smoke_gpu_hours": smoke_gpu_hours,
            "total_gpu_hours_including_smoke": total_gpu_hours + smoke_gpu_hours,
            "smoke_runs": smoke_records,
            "within_cap": bool(total_gpu_hours + smoke_gpu_hours
                               <= float(cfg["budget"]["gpu_hours_hard_cap"])),
            "note": "formal cost = sum of ar_train wall_seconds / 3600; smoke cost = sum of the committed local/remote runs_smoke/*/smoke.json wall clocks. Remote rr occupancy (including failed jobs) is reported separately in the REPORT.",
        },
        "official_test_loaded": False,
        "official_valid_loaded": False,
    }
    C.write_json(C.ANALYSIS / "summary.json", summary)
    C.write_json(C.ANALYSIS / "decision.json", decision)
    write_tables(summary)
    C.log(f"[analysis] verdict={decision['verdict']} writes=analysis/summary.json,"
          f"decision.json,tables.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
