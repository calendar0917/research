"""ZINC CSSD performance triage v1: read-only error attribution on the saved
predictions of ``zinc_cssd_consumer_replacement_v1``.

Round: ``zinc_cssd_performance_triage_v1`` (analysis only — no training, no
GPU, no model loading, no forward replays, no official valid/test).

The single question this round answers for the next budget decision: to lower
the complete ``y_raw`` MAE, should the next training budget go to the Q cycle
path or to the dictionary consumer's g generalisation / readout — and is any
one concrete performance candidate purchased by the current evidence?

Method (all from saved NumPy/JSON products; ``evaluate_run`` /
``terminal_eval`` / ``load_round_objects`` of the source module are never
called because they would load model objects this analysis does not need):

* rows come from the source fold: ``fit_idx`` (8001) and
  ``dev_idx = sorted union(select_idx, confirm_idx)`` (1999); the saved
  ``gid``/``y`` columns are checked row-by-row against
  ``targets.npz`` at those positions (``gid`` is ``canonical_group_id``,
  *not* ``subset_index`` and not assumed unique; every row is kept).
* error synthesis with the "prediction - truth" sign convention::

      e_g = h - g,  e_Q = q_raw - c,  e_y = e_g + e_Q  (≈ y_raw - y)

  Component MAEs are never added into a budget; the net effect of the current
  Q error is ``B_Q = mean(|e_g + e_Q| - |e_g|)`` and the composition
  cancellation is ``cancellation_gap = mean(|e_g| + |e_Q| - |e_y|)``.
* four mutually exclusive label-cycle groups ``k=0 / k=-1 / k=-2 / k<=-3``
  with contributions ``C_y(G)``, ``B_Q(G)``, ``C_g(G)``, ``Delta_y(G)``
  normalised by the FULL split row count ``N`` (never the in-group ``n``),
  plus exact add-back checks against the full-split totals.
* dev top-20 by the two-seed mean DICT ``|e_y|`` (ties by ``subset_index``),
  top-2/5/20 contributions to the full MAE / ``B_Q``, and exclusion scores
  as sensitivity only (the primary metric is always the full 1999 rows).
* the located long-cycle pair gid=3775 / gid=1424 keeps its historical
  localisation from the q-spotcheck note; every other top-20 row is marked
  ``unknown`` — this round does not start a new molecule audit.
* two text errata of the previous round's REPORT are recorded; no historical
  number, roster or record is modified.

True ``g``/``c`` are used only inside this offline decomposition; they are
never a deployment plan, an input feature, or a reachable-performance claim.
The oracle-Q g-MAE is not a lower bound for the complete model.

Usage (CPU, seconds)::

    python -m tracks.ksvd.experiments.luyin16.zinc_cssd_performance_triage_v1
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROTOCOL_VERSION = "zinc-cssd-performance-triage-v1"
RESULT_SLUG = "zinc_cssd_performance_triage_v1"
TRACK_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = TRACK_ROOT / "results" / RESULT_SLUG

#: read-only inputs (paths identical to the source module's RESULTS_DIR /
#: SOURCE_DIR constants, verified against commit 32b8c96 of
#: zinc_cssd_consumer_replacement_v1.py)
SOURCE_DIR = TRACK_ROOT / "results" / "zinc_cssd_nonlinear_binding_v1"
CONSUMER_DIR = TRACK_ROOT / "results" / "zinc_cssd_consumer_replacement_v1"

ARMS = ("RAW", "DICT")
SEEDS = (0, 1)
RUN_NAMES = tuple(f"{arm}_s{seed}" for arm in ARMS for seed in SEEDS)

#: the frozen shared Q soup hash recorded by all four run manifests
EXPECTED_Q_SOUP_SHA = "14d175e371e31f00bd043bcece1e5a44150bd6c9cadea6b2eed3ec86f4f41b21"

#: FP32/save-precision tolerance for the stored identities
IDENTITY_TOL = 1e-5

GROUPS: tuple[tuple[str, Any], ...] = (
    ("k=0", lambda k: k == 0),
    ("k=-1", lambda k: k == -1),
    ("k=-2", lambda k: k == -2),
    ("k<=-3", lambda k: k <= -3),
)

#: historical localisation (notes/zinc_cssd_nonlinear_binding_v1_q_spotcheck.md §5)
LOCATED_GIDS: dict[int, str] = {
    3775: (
        "T25 unseen singleton class (its class fell into select, not the 8001 fit "
        "pool) + ordering-dependent label penalty (q_spotcheck §5; = audit "
        "train:3776)"
    ),
    1424: (
        "T25 input conflict (identical T25 vector carries k=0 at fit row "
        "position 1270) + long-cycle tail beyond T25 expressivity "
        "(q_spotcheck §5)"
    ),
}
UNKNOWN_LOCALIZATION = "unknown (not located this round; no new molecule audit)"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=TRACK_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")
    return {"head": head, "dirty": bool(dirty), "dirty_files": dirty.splitlines()}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def round_split_idx(fold: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """fit = the saved fit fold; dev = sorted union(select, confirm)."""
    fit = np.sort(np.asarray(fold["fit_idx"], np.int64))
    dev = np.sort(np.union1d(
        np.asarray(fold["select_idx"], np.int64),
        np.asarray(fold["confirm_idx"], np.int64),
    ))
    return {"fit": fit, "dev": dev}


def _load_split_predictions(run: str, split: str) -> dict[str, np.ndarray]:
    path = CONSUMER_DIR / "runs" / run / f"{split}_predictions.npz"
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _mae(a: np.ndarray) -> float:
    return float(np.mean(np.abs(a)))


def _totals(eg: np.ndarray, eQ: np.ndarray) -> dict[str, float]:
    ey = eg + eQ
    return {
        "n": int(ey.size),
        "MAE_y": _mae(ey),
        "MAE_g": _mae(eg),
        "MAE_Q": _mae(eQ),
        "B_Q": float(np.mean(np.abs(ey) - np.abs(eg))),
        "cancellation_gap": float(np.mean(np.abs(eg) + np.abs(eQ) - np.abs(ey))),
        "opposite_sign_fraction": float(
            np.mean(np.sign(eg) * np.sign(eQ) < 0)
        ),
        "diag_MAE_h_plus_c_minus_g": _mae(eg),   # identity: == MAE_g
        "diag_MAE_g_plus_q_minus_c": _mae(eQ),   # identity: == MAE_Q
    }


def _group_rows(
    k: np.ndarray, eg: np.ndarray, eQ: np.ndarray, N: int,
    eg_pair: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Per-group metrics + contributions (normalised by the FULL split N).

    ``eg_pair`` (same arm's RAW partner, same seed) enables Delta_y(G).
    Returns (rows, add-back residuals).
    """
    ey = eg + eQ
    rows: list[dict[str, Any]] = []
    sum_cy = sum_bq = sum_cg = 0.0
    sum_dy = 0.0
    for name, pred in GROUPS:
        m = pred(k)
        n = int(m.sum())
        cy = float(np.abs(ey)[m].sum() / N)
        bq = float((np.abs(ey)[m] - np.abs(eg)[m]).sum() / N)
        cg = float(np.abs(eg)[m].sum() / N)
        row = {
            "group": name, "n": n,
            "mae_y_in_group": _mae(ey[m]),
            "mae_g_in_group": _mae(eg[m]),
            "mae_q_in_group": _mae(eQ[m]),
            "C_y": cy, "B_Q_group": bq, "C_g": cg,
        }
        if eg_pair is not None:
            dy = float((np.abs(ey) - np.abs(eg_pair + eQ))[m].sum() / N)
            row["Delta_y_group"] = dy
            sum_dy += dy
        rows.append(row)
        sum_cy += cy
        sum_bq += bq
        sum_cg += cg
    return rows, {
        "addback_C_y_residual": sum_cy - _mae(ey),
        "addback_B_Q_residual": sum_bq - float(np.mean(np.abs(ey) - np.abs(eg))),
        "addback_C_g_residual": sum_cg - _mae(eg),
        "addback_Delta_y_residual": sum_dy - (
            float(np.mean(np.abs(ey) - np.abs(eg_pair + eQ))) if eg_pair is not None else 0.0
        ),
        "groups_cover_all_rows": sum(int(pred(k).sum()) for _, pred in GROUPS) == int(k.size),
    }


def analyze(out_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- (0) inputs, code state, fold view --------------------------------
    fold_path = SOURCE_DIR / "fold.npz"
    targets_path = SOURCE_DIR / "targets.npz"
    terminal_eval_path = CONSUMER_DIR / "terminal_eval.json"
    consumer_source_manifest_path = CONSUMER_DIR / "source_manifest.json"

    with np.load(fold_path, allow_pickle=False) as z:
        fold = {k: np.asarray(z[k]) for k in z.files}
    with np.load(targets_path, allow_pickle=False) as z:
        targets = {k: np.asarray(z[k]) for k in (
            "y", "g", "c", "k", "ell", "s", "gid"
        )}
    split_idx = round_split_idx(fold)
    terminal_eval = _read_json(terminal_eval_path)
    consumer_source_manifest = _read_json(consumer_source_manifest_path)

    git_state = _git_state()
    inputs_manifest = {
        "code_revision": git_state,
        "read_only": True,
        "source_round": "zinc_cssd_consumer_replacement_v1 (promote commit 32b8c96, "
                        "four formal training runs at revision a308e3490bfe)",
        "inputs": {
            "fold.npz": sha256_file(fold_path),
            "targets.npz": sha256_file(targets_path),
            "terminal_eval.json": sha256_file(terminal_eval_path),
            "source_manifest.json (consumer round)": sha256_file(
                consumer_source_manifest_path
            ),
        },
        "per_run_inputs": {},
        "fold_view": {
            split: {
                "n": int(idx.size),
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(idx, np.int64).tobytes()
                ).hexdigest(),
                "definition": (
                    "the source round's committed fit fold"
                    if split == "fit"
                    else "sorted union(select_idx, confirm_idx) — the historical "
                         "development rows; a DEVELOPMENT COMPARISON set, never a "
                         "new independent confirm"
                ),
            }
            for split, idx in split_idx.items()
        },
        "disjoint_fit_dev": bool(
            not np.intersect1d(split_idx["fit"], split_idx["dev"]).size
        ),
        "cover_10000": bool(
            np.union1d(split_idx["fit"], split_idx["dev"]).size == targets["y"].size
        ),
        "gid_semantics": (
            "gid is canonical_group_id, not subset_index and not assumed unique; "
            "every sample row is kept (no gid->single-row mapping)"
        ),
    }

    # ---- (1) per-run loading, alignment, identity, error synthesis --------
    alignment: dict[str, Any] = {
        "identity_y_equals_g_plus_c_max": float(
            np.abs(targets["y"] - (targets["g"] + targets["c"])).max()
        ),
        "identity_g_equals_ell_plus_s_max": float(
            np.abs(targets["g"] - (targets["ell"] + targets["s"])).max()
        ),
    }
    q_soup_hashes: dict[str, str] = {}
    q_ref: dict[str, np.ndarray] = {}
    errors: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    totals: dict[str, dict[str, dict[str, Any]]] = {"fit": {}, "dev": {}}
    group_table: dict[str, dict[str, Any]] = {}
    terminal_repro: dict[str, dict[str, Any]] = {}

    for run in RUN_NAMES:
        run_manifest = _read_json(CONSUMER_DIR / "runs" / run / "manifest.json")
        q_soup_hashes[run] = str(run_manifest["q_soup_sha256"])
        inputs_manifest["per_run_inputs"][run] = {
            "manifest.json": sha256_file(CONSUMER_DIR / "runs" / run / "manifest.json"),
            "fit_predictions.npz": sha256_file(
                CONSUMER_DIR / "runs" / run / "fit_predictions.npz"
            ),
            "dev_predictions.npz": sha256_file(
                CONSUMER_DIR / "runs" / run / "dev_predictions.npz"
            ),
            "soup_state_sha256": str(run_manifest["soup_state_sha256"]),
        }
        for split in ("fit", "dev"):
            pred = _load_split_predictions(run, split)
            idx = split_idx[split]
            g = targets["g"][idx]
            c = targets["c"][idx]
            y = targets["y"][idx]
            k = targets["k"][idx]
            gid = targets["gid"][idx]
            h = pred["h"].astype(np.float64)
            q = pred["q_raw"].astype(np.float64)
            y_raw = pred["y_raw"].astype(np.float64)

            checks = {
                "gid_matches_targets_rows": bool(np.array_equal(pred["gid"], gid)),
                "y_matches_targets_rows": bool(
                    np.array_equal(pred["y"].astype(np.float64), y)
                ),
                "y_raw_minus_h_plus_q_raw_max": float(
                    np.abs(y_raw - (h + q)).max()
                ),
                "e_y_minus_y_raw_minus_y_max": float(
                    np.abs((h - g + q - c) - (y_raw - y)).max()
                ),
                "q_raw_bit_identical_to_reference": None,  # reference for the first run
            }
            if split in q_ref:
                checks["q_raw_bit_identical_to_reference"] = bool(
                    np.array_equal(q, q_ref[split])
                )
            else:
                q_ref[split] = q

            eg = h - g
            eQ = q - c
            errors.setdefault(run, {})[split] = {"e_g": eg, "e_Q": eQ, "e_y": eg + eQ}
            tot = _totals(eg, eQ)
            tot["identity_h_equals_ell_hat_plus_s_hat_max"] = (
                float(np.abs(h - (pred["ell_hat"].astype(np.float64)
                                  + pred["s_hat"].astype(np.float64))).max())
                if "ell_hat" in pred else None
            )
            totals[split][run] = tot
            alignment[f"{run}/{split}"] = checks

            # reproduce the one-shot terminal_eval dev MAEs at full precision
            if split == "dev":
                published = terminal_eval["main_table"][run]["mae"]
                terminal_repro[run] = {
                    "saved_MAE_y_raw": tot["MAE_y"],
                    "terminal_eval_MAE_y_raw": float(published["y_raw"]),
                    "y_raw_match_within_1e-8": bool(
                        abs(tot["MAE_y"] - float(published["y_raw"])) <= 1e-8
                    ),
                    "y_raw_repro_abs_diff": abs(
                        tot["MAE_y"] - float(published["y_raw"])
                    ),
                    "saved_MAE_g_raw": tot["MAE_g"],
                    "terminal_eval_MAE_g_raw": float(published["g_raw"]),
                    "g_raw_match_within_1e-8": bool(
                        abs(tot["MAE_g"] - float(published["g_raw"])) <= 1e-8
                    ),
                    "g_raw_repro_abs_diff": abs(
                        tot["MAE_g"] - float(published["g_raw"])
                    ),
                    "tolerance_note": (
                        "the terminal-eval MAEs were accumulated in torch FP32; "
                        "this replay averages the saved FP32 arrays in float64, so "
                        "agreement is expected at the ~1e-10 level, never exact"
                    ),
                }

    # ---- (2) frozen-Q object checks ---------------------------------------
    q_checks = {
        "per_run_manifest_q_soup_sha256": q_soup_hashes,
        "all_four_runs_share_one_q_hash": bool(
            len(set(q_soup_hashes.values())) == 1
        ),
        "q_hash_equals_expected_frozen_soup": bool(
            set(q_soup_hashes.values()) == {EXPECTED_Q_SOUP_SHA}
        ),
        "q_hash_equals_consumer_source_manifest": bool(
            set(q_soup_hashes.values())
            == {consumer_source_manifest["q_soup"]["state_sha256"]}
        ),
        "q_raw_bit_identical_across_runs": {
            split: bool(
                len({
                    _load_split_predictions(run, split)["q_raw"].tobytes()
                    for run in RUN_NAMES
                }) == 1
            )
            for split in ("fit", "dev")
        },
    }

    # ---- (3) seed-mean totals and the published expectation check --------
    seed_mean: dict[str, dict[str, Any]] = {}
    for split in ("fit", "dev"):
        for arm in ARMS:
            per_seed = [totals[split][f"{arm}_s{seed}"] for seed in SEEDS]
            seed_mean[f"{arm}/{split}"] = {
                key: float(np.mean([ps[key] for ps in per_seed]))
                for key in ("MAE_y", "MAE_g", "MAE_Q", "B_Q",
                            "cancellation_gap", "opposite_sign_fraction")
            }
        # paired DICT - RAW delta on the full split
        for seed in SEEDS:
            seed_mean[f"delta_s{seed}/{split}"] = float(
                totals[split][f"DICT_s{seed}"]["MAE_y"]
                - totals[split][f"RAW_s{seed}"]["MAE_y"]
            )
        seed_mean[f"avg_delta/{split}"] = float(np.mean([
            seed_mean[f"delta_s{seed}/{split}"] for seed in SEEDS
        ]))
    expectation = {
        "note": "expected values come from the previous round's REPORT rounding; "
                "the saved full-precision values above are authoritative",
        "dict_dev_mean_MAE_y": {
            "expected_approx": 0.11994, "saved": seed_mean["DICT/dev"]["MAE_y"]},
        "dict_dev_mean_MAE_g": {
            "expected_approx": 0.09257, "saved": seed_mean["DICT/dev"]["MAE_g"]},
        "dict_dev_mean_B_Q": {
            "expected_approx": 0.02737, "saved": seed_mean["DICT/dev"]["B_Q"]},
    }

    # ---- (4) k-group contributions (per run + seed means, both splits) ---
    csv_rows: list[dict[str, Any]] = []
    group_json: dict[str, Any] = {"splits": {}}
    addback: dict[str, Any] = {}
    for split in ("fit", "dev"):
        idx = split_idx[split]
        k = targets["k"][idx]
        N = int(idx.size)
        split_rows: dict[str, Any] = {"per_run": {}, "seed_mean": {}}
        for arm in ARMS:
            per_seed_rows = []
            for seed in SEEDS:
                run = f"{arm}_s{seed}"
                eg = errors[run][split]["e_g"]
                eQ = errors[run][split]["e_Q"]
                eg_pair = (
                    errors[f"RAW_s{seed}"][split]["e_g"] if arm == "DICT" else None
                )
                rows, resid = _group_rows(k, eg, eQ, N, eg_pair=eg_pair)
                addback[f"{run}/{split}"] = resid
                split_rows["per_run"][run] = rows
                per_seed_rows.append(rows)
                for r in rows:
                    csv_rows.append({
                        "split": split, "arm": arm, "seed": seed,
                        **{kk: r[kk] for kk in (
                            "group", "n", "mae_y_in_group", "mae_g_in_group",
                            "mae_q_in_group", "C_y", "B_Q_group", "C_g",
                        )},
                        **({"Delta_y_group": r["Delta_y_group"]}
                           if "Delta_y_group" in r else {}),
                    })
            # seed-mean rows: each metric averaged over seeds (never an ensemble)
            mean_rows = []
            for gi, (name, _) in enumerate(GROUPS):
                row: dict[str, Any] = {"group": name}
                for key in ("n", "mae_y_in_group", "mae_g_in_group", "mae_q_in_group",
                            "C_y", "B_Q_group", "C_g", "Delta_y_group"):
                    vals = [per_seed_rows[si][gi][key] for si in range(len(SEEDS))
                            if key in per_seed_rows[si][gi]]
                    if vals:
                        row[key] = float(np.mean(vals))
                mean_rows.append(row)
                csv_rows.append({
                    "split": split, "arm": arm, "seed": -1, **{
                        kk: row[kk] for kk in row if kk != "group"},
                    "group": name,
                })
            split_rows["seed_mean"][arm] = mean_rows
        group_json["splits"][split] = split_rows

    # add-back checks (all runs; Delta_y only defined for the DICT pairing)
    for split in ("fit", "dev"):
        for run in RUN_NAMES:
            resid = addback[f"{run}/{split}"]
            ok = (
                abs(resid["addback_C_y_residual"]) < 1e-9
                and abs(resid["addback_B_Q_residual"]) < 1e-9
                and abs(resid["addback_C_g_residual"]) < 1e-9
                and resid["groups_cover_all_rows"]
            )
            if run.startswith("DICT"):
                ok = ok and abs(resid["addback_Delta_y_residual"]) < 1e-9
            addback[f"{run}/{split}"]["all_addback_checks_pass"] = bool(ok)

    # ---- (5) dev top-20 by two-seed mean DICT |e_y| -----------------------
    idx = split_idx["dev"]
    gid = targets["gid"][idx]
    k_dev = targets["k"][idx]
    y_dev = targets["y"][idx]
    c_dev = targets["c"][idx]
    q_dev = q_ref["dev"]
    mean_abs_ey_dict = np.mean([
        np.abs(errors[f"DICT_s{seed}"]["dev"]["e_y"]) for seed in SEEDS
    ], axis=0)
    order = sorted(range(idx.size), key=lambda i: (-mean_abs_ey_dict[i], int(idx[i])))

    top_rows = []
    for rank, i in enumerate(order[:20], start=1):
        row: dict[str, Any] = {
            "rank": rank,
            "subset_index": int(idx[i]),
            "gid": int(gid[i]),
            "k": int(k_dev[i]),
            "y": float(y_dev[i]),
            "c": float(c_dev[i]),
            "q_raw": float(q_dev[i]),
            "mean_abs_e_y_dict": float(mean_abs_ey_dict[i]),
            "localization": LOCATED_GIDS.get(int(gid[i]), UNKNOWN_LOCALIZATION),
        }
        for run in RUN_NAMES:
            eg = errors[run]["dev"]["e_g"][i]
            eQ = errors[run]["dev"]["e_Q"][i]
            h = eg + float(targets["g"][idx][i])
            row[f"h_{run}"] = float(h)
            row[f"e_g_{run}"] = float(eg)
            row[f"e_y_{run}"] = float(eg + eQ)
            row[f"abs_e_y_{run}"] = float(abs(eg + eQ))
        row["e_Q_shared"] = float(errors["DICT_s0"]["dev"]["e_Q"][i])
        for seed in SEEDS:
            row[f"pairdiff_s{seed}"] = float(
                abs(errors[f"DICT_s{seed}"]["dev"]["e_y"][i])
                - abs(errors[f"RAW_s{seed}"]["dev"]["e_y"][i])
            )
        row["pairdiff_mean"] = float(np.mean([
            row[f"pairdiff_s{seed}"] for seed in SEEDS
        ]))
        top_rows.append(row)

    N = int(idx.size)
    topk: dict[str, Any] = {}
    for K in (2, 5, 20):
        m = np.zeros(N, bool)
        m[order[:K]] = True
        topk[f"top{K}"] = {
            "C_y_DICT_seed_mean": float(mean_abs_ey_dict[m].sum() / N),
            "C_y_RAW_seed_mean": float(np.mean([
                np.abs(errors[f"RAW_s{seed}"]["dev"]["e_y"])[m].sum() / N
                for seed in SEEDS
            ])),
            "B_Q_DICT_seed_mean": float(np.mean([
                (np.abs(errors[f"DICT_s{seed}"]["dev"]["e_y"])[m].sum()
                 - np.abs(errors[f"DICT_s{seed}"]["dev"]["e_g"])[m].sum()) / N
                for seed in SEEDS
            ])),
            "Delta_y_seed_mean": float(np.mean([
                (np.abs(errors[f"DICT_s{seed}"]["dev"]["e_y"])[m].sum()
                 - np.abs(errors[f"RAW_s{seed}"]["dev"]["e_y"])[m].sum()) / N
                for seed in SEEDS
            ])),
            "share_of_DICT_dev_B_Q": None,  # filled after B_Q known
            "sensitivity_MAE_excluding_topK_DICT_seed_mean": float(np.mean([
                np.abs(errors[f"DICT_s{seed}"]["dev"]["e_y"])[~m].mean()
                for seed in SEEDS
            ])),
            "sensitivity_MAE_excluding_topK_RAW_seed_mean": float(np.mean([
                np.abs(errors[f"RAW_s{seed}"]["dev"]["e_y"])[~m].mean()
                for seed in SEEDS
            ])),
            "sensitivity_note": (
                "sensitivity only; the primary metric is always the full 1999-row "
                "MAE — these numbers are never a performance claim"
            ),
        }
    dict_dev_bq = seed_mean["DICT/dev"]["B_Q"]
    for K in (2, 5, 20):
        topk[f"top{K}"]["share_of_DICT_dev_B_Q"] = float(
            topk[f"top{K}"]["B_Q_DICT_seed_mean"] / dict_dev_bq
        )

    # the located pair's current contributions
    located: dict[str, Any] = {}
    for g in (3775, 1424):
        pos = int(np.where(targets["gid"] == g)[0][0])
        hits = np.where(idx == pos)[0]
        if hits.size == 0:
            located[str(g)] = {
                "subset_index": pos, "in_dev": False,
                "localization": LOCATED_GIDS[g],
            }
            continue
        i = int(hits[0])
        located[str(g)] = {
            "subset_index": pos,
            "in_dev": True,
            "k": int(k_dev[i]),
            "y": float(y_dev[i]),
            "c": float(c_dev[i]),
            "per_run_abs_e_y": {
                run: float(abs(errors[run]["dev"]["e_y"][i])) for run in RUN_NAMES
            },
            "per_run_abs_e_g": {
                run: float(abs(errors[run]["dev"]["e_g"][i])) for run in RUN_NAMES
            },
            "abs_e_Q_shared": float(abs(errors["DICT_s0"]["dev"]["e_Q"][i])),
            "C_y_contribution_DICT_seed_mean": float(
                mean_abs_ey_dict[i] / N
            ),
            "B_Q_contribution_DICT_seed_mean": float(np.mean([
                (abs(errors[f"DICT_s{seed}"]["dev"]["e_y"][i])
                 - abs(errors[f"DICT_s{seed}"]["dev"]["e_g"][i])) / N
                for seed in SEEDS
            ])),
            "localization": LOCATED_GIDS[g],
        }

    # ---- (6) component cancellation gap (erratum support) -----------------
    # dev: from the terminal_eval.json saved full-precision component MAEs
    # (the dev prediction files have no per-row ell_hat/s_hat by design and no
    # forward replay is allowed this round).
    component_gap = {"dev": {}, "fit": {}}
    for run in RUN_NAMES:
        mae = terminal_eval["main_table"][run]["mae"]
        component_gap["dev"][run] = {
            "MAE_ell": float(mae["ell"]),
            "MAE_s": float(mae["s"]),
            "MAE_g": totals["dev"][run]["MAE_g"],
            "component_cancellation_gap": float(mae["ell"]) + float(mae["s"])
            - totals["dev"][run]["MAE_g"],
        }
        fd = terminal_eval["main_table"][run]["fit_diagnostics"]
        component_gap["fit"][run] = {
            "MAE_ell": float(fd["fit_ell_mae"]),
            "MAE_s": float(fd["fit_s_mae"]),
            "MAE_g": totals["fit"][run]["MAE_g"],
            "component_cancellation_gap": float(fd["fit_ell_mae"])
            + float(fd["fit_s_mae"]) - totals["fit"][run]["MAE_g"],
        }
    for arm in ARMS:
        for split in ("fit", "dev"):
            component_gap[split][f"{arm}_seed_mean"] = {
                key: float(np.mean([
                    component_gap[split][f"{arm}_s{seed}"][key] for seed in SEEDS
                ]))
                for key in ("MAE_ell", "MAE_s", "MAE_g",
                            "component_cancellation_gap")
            }
    component_gap["reading"] = (
        "ell/s are the disclosed COMP auxiliary components (s is the residual "
        "item, never 'pure SA'); the gap change is an error-composition change "
        "only and proves no regularisation / semantic-decoupling mechanism; no "
        "per-row component predictions exist on dev so no correlation or "
        "opposite-sign ratio is reported"
    )

    errata = [
        {
            "id": "consumer-replacement-report-fit-dev-wording",
            "text": "The consumer-replacement REPORT §2 said 'fit/dev 方向一致'; the "
                    "correct reading is the opposite direction: fit y_raw DICT is "
                    "slightly WORSE (0.04083/0.03778 vs 0.03962/0.03514) while dev "
                    "is slightly better — fit and dev move in opposite directions, "
                    "and the two-seed dev advantage must not be read as a stable "
                    "regularisation gain.",
        },
        {
            "id": "consumer-replacement-report-s-component-wording",
            "text": "The consumer-replacement REPORT §8c listed 's 分量 DICT 略好' as "
                    "a direction; on the four-run means the s MAE did NOT improve "
                    "(RAW ≈ 0.08107 vs DICT ≈ 0.08138) and ell is worse (0.05357 vs "
                    "0.05634); only g improved (0.09359 vs 0.09257). The component "
                    "cancellation gap grew ≈ 0.0410 → 0.0451, an error-composition "
                    "change, not evidence of regularisation/semantic decoupling.",
        },
    ]

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "analyze",
        "seconds": None,
        "inputs": inputs_manifest,
        "alignment": alignment,
        "frozen_q": q_checks,
        "terminal_eval_reproduction": terminal_repro,
        "totals": totals,
        "seed_mean": seed_mean,
        "expectation_check": expectation,
        "groups": group_json,
        "addback_checks": addback,
        "top20": {"rows": top_rows, "topk": topk},
        "located_gids": located,
        "component_cancellation_gap": component_gap,
        "errata_previous_round": errata,
        "identity_tolerance": IDENTITY_TOL,
        "sign_convention": "e_g = h - g; e_Q = q_raw - c; e_y = e_g + e_Q ≈ y_raw - y",
        "scope": (
            "read-only attribution on saved predictions; true g/c are offline "
            "diagnostics only, never a deployment plan, input feature or "
            "reachable-performance claim; dev is the historical development "
            "comparison, not an independent confirm; no official valid/test"
        ),
    }
    summary["seconds"] = time.perf_counter() - started

    # ---- (7) write outputs -------------------------------------------------
    (out_dir / "source_manifest.json").write_text(
        json.dumps(inputs_manifest, indent=2), encoding="utf-8"
    )
    (out_dir / "error_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _fieldnames = ["split", "arm", "seed", "group", "n", "mae_y_in_group",
                   "mae_g_in_group", "mae_q_in_group", "C_y", "B_Q_group", "C_g",
                   "Delta_y_group"]
    with open(out_dir / "group_contributions.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_fieldnames)
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)
    top_fields = ["rank", "subset_index", "gid", "k", "y", "c", "q_raw",
                  "mean_abs_e_y_dict"] \
        + [f"h_{run}" for run in RUN_NAMES] \
        + [f"e_g_{run}" for run in RUN_NAMES] \
        + ["e_Q_shared"] \
        + [f"e_y_{run}" for run in RUN_NAMES] \
        + [f"abs_e_y_{run}" for run in RUN_NAMES] \
        + [f"pairdiff_s{seed}" for seed in SEEDS] + ["pairdiff_mean",
                                                     "localization"]
    with open(out_dir / "top20.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=top_fields)
        w.writeheader()
        for r in top_rows:
            w.writerow(r)
    return summary


def main() -> int:
    result = analyze()
    print(json.dumps({
        "seconds": result["seconds"],
        "dict_dev_mean_MAE_y": result["seed_mean"]["DICT/dev"]["MAE_y"],
        "dict_dev_mean_MAE_g": result["seed_mean"]["DICT/dev"]["MAE_g"],
        "dict_dev_mean_B_Q": result["seed_mean"]["DICT/dev"]["B_Q"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
