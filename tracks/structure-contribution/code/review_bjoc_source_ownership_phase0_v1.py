#!/usr/bin/env python3
"""Independent re-analysis of ``bjoc_source_ownership_v1`` Phase 0.

Read-only w.r.t. the original namespace.  Three stages:

  recompute  rebuild the matched-control per-junction signed/absolute responses
             from the frozen sample + soups + RNG + matching rule and verify
             exact reproduction of the stored historical values;
  analyze    per-seed / cluster / conservative-row statistics, absolute matched
             controls, and Phase-0C permutation-null + sign-semantics corrections;
  decide     apply the pre-declared review decision rule (PROTOCOL.md).

No labels, no training, no GPU, no official test / Locked access.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_source_attribution_feasibility_v1 as bsaf  # noqa: E402
import bjoc_source_ownership_v1 as own  # noqa: E402
import bjoc_structure_explanation_audit_v1 as v1  # noqa: E402
import run_bjoc_source_ownership_v1 as rso  # noqa: E402  (frozen intervention/ranking code)

TRACK = REPO_ROOT / "tracks/structure-contribution"
ORIG = TRACK / "results" / "bjoc_source_ownership_v1"
BSAF = TRACK / "results" / "bjoc_source_attribution_feasibility_v1"
OUT = TRACK / "results" / "bjoc_source_ownership_phase0_review_v1"

SEEDS = (0, 1)
PHASE0_SEEDS = (0, 1)
N_BOOTSTRAP = 5000
N_PERMUTATION = 1000
PHASE0_RANDOM_DRAWS = 16
PHASE0_MATCH_CONTROLS = 8
PHASE0_TOP_K = 4
BOOTSTRAP_SEED = 0
RNG_INDEPENDENT_PERM = 20261015
RNG_JOINT_PERM = 20261016


def log(msg: str) -> None:
    print(msg, flush=True)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return v1.file_sha256(path)


# ---------------------------------------------------------------------------
# statistics helpers (corrected units)
# ---------------------------------------------------------------------------


def bootstrap_mean_ci(values: np.ndarray, n: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    rng = np.random.default_rng(seed)
    draws = rng.choice(arr, size=(n, arr.size), replace=True).mean(1)
    return {
        "mean": float(arr.mean()),
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)),
        "median": float(np.median(arr)),
        "n": int(arr.size),
    }


def one_sided_sign(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    k = int((arr > 0).sum())
    p = float(sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0**n)) if n else float("nan")
    return {
        "n": n,
        "n_positive": k,
        "n_negative": int((arr < 0).sum()),
        "frac_positive": float(k / n) if n else float("nan"),
        "p_one_sided": p,
    }


def cluster_bootstrap_ci(per_row: Mapping[Any, np.ndarray], n: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> dict[str, float]:
    """Row-cluster bootstrap: a drawn molecule contributes all its seed values."""
    rows = list(per_row)
    a = np.full(len(rows), np.nan)
    b = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        v = np.asarray(per_row[r], dtype=np.float64)
        a[i] = v[0]
        if v.size > 1:
            b[i] = v[1]
    allv = np.concatenate([v[~np.isnan(v)] for v in (a, b)]) if rows else np.zeros(0)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for t in range(n):
        idx = rng.integers(0, len(rows), size=len(rows))
        va, vb = a[idx], b[idx]
        total = np.nansum(va) + np.nansum(vb)
        cnt = int(np.sum(~np.isnan(va))) + int(np.sum(~np.isnan(vb)))
        means[t] = total / cnt if cnt else np.nan
    return {
        "mean": float(np.nanmean(allv)) if allv.size else float("nan"),
        "ci95_lower": float(np.nanquantile(means, 0.025)),
        "ci95_upper": float(np.nanquantile(means, 0.975)),
        "n_rows": len(rows),
        "n_values": int(allv.size),
    }


def row_unit(vals_by_seed_row: Mapping[tuple[int, int], float], seeds: tuple[int, ...] = SEEDS) -> dict[int, np.ndarray]:
    """row -> array of the available seed values (mean over seeds is the unit)."""
    out: dict[int, list[float]] = defaultdict(list)
    for (seed, row), v in vals_by_seed_row.items():
        if v is None or seed not in seeds:
            continue
        out[row].append(float(v))
    return {row: np.asarray(v, dtype=np.float64) for row, v in out.items()}


def summarize_values(vals_by_seed_row: Mapping[tuple[int, int], float], seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """Summmarise with per-seed, pooled, cluster and conservative-row units."""
    out: dict[str, Any] = {"per_seed": {}, "pooled_molecule_seed": {}, "row_unit": {}}
    for seed in seeds:
        per_seed = np.asarray([v for (s, _), v in vals_by_seed_row.items() if s == seed and v is not None], dtype=np.float64)
        out["per_seed"][str(seed)] = {**bootstrap_mean_ci(per_seed), **one_sided_sign(per_seed)}
    pooled = np.asarray([v for v in vals_by_seed_row.values() if v is not None], dtype=np.float64)
    out["pooled_molecule_seed"] = {**bootstrap_mean_ci(pooled), **one_sided_sign(pooled)}
    rows = row_unit(vals_by_seed_row, seeds)
    out["row_unit"] = cluster_bootstrap_ci(rows)
    row_means = np.asarray([v.mean() for v in rows.values()], dtype=np.float64)
    out["row_unit"].update(one_sided_sign(row_means))
    complete = {r: v for r, v in rows.items() if len(v) == 2}
    comp_means = np.asarray([v.mean() for v in complete.values()], dtype=np.float64)
    out["row_unit_complete_pairs"] = {**bootstrap_mean_ci(comp_means), **one_sided_sign(comp_means)}
    return out


# ---------------------------------------------------------------------------
# stage 1 — recompute matched controls (signed + absolute) with reproduction
# ---------------------------------------------------------------------------


def recompute() -> dict[str, Any]:
    orig = read_json(ORIG / "phase0_results.json")
    hist_b = {(int(r["seed"]), int(r["row"])): r for r in orig["records_b"]}
    sample = read_json(BSAF / "sample_manifest.json")
    diag_rows = [int(r) for r in sample["diagnostic_rows"]]
    ref_rows = [int(r) for r in sample["reference_rows"]]
    objs = v1.load_objects()
    norm = v1.load_norm()
    log("[recompute] corpus signature counts ...")
    counts = own.signature_corpus_counts(objs)
    corpus_total = int(sum(counts.values()))
    records: list[dict[str, Any]] = []
    check: dict[str, Any] = {
        "top4_sets_mismatch": [],
        "top4_response_max_abs_diff": 0.0,
        "random_response_max_abs_diff": 0.0,
        "matched_signed_gap_max_abs_diff": 0.0,
        "coverage_mismatch": [],
        "dev_exact_mismatch": [],
        "n_molecule_seed_units": 0,
    }
    for seed in PHASE0_SEEDS:
        model = v1.load_model(seed, norm)
        ref_w_map = rso._reference_w_map(model, objs, ref_rows)
        ref_valid = [
            (int(r), sample["reference_stats"][str(int(r))], ref_w_map[int(r)])
            for r in ref_rows
            if int(r) in ref_w_map
        ]
        for row in diag_rows:
            stats = sample["diagnostic_stats"][str(int(row))]
            ref_row, _, w_ref = min(
                ref_valid,
                key=lambda t: (
                    abs(t[1]["n_atoms"] - stats["n_atoms"])
                    + abs(t[1]["n_bonds"] - stats["n_bonds"])
                    + abs(t[1]["n_junctions"] - stats["n_junctions"]),
                    t[0],
                ),
            )
            batch = v1.make_batch(objs, [int(row)])
            feats = v1.run_features(model, batch)
            pack = bsaf.pack_from_features(model, batch, feats, 0)
            n_j = int(pack["n_junctions"])
            if n_j < 1:
                continue
            w = pack["w"]
            ig = bsaf.ig_scores(model, pack, w.mean(0))
            dev = own.dev_ranking(w).numpy()
            rare = own.rarity_ranking(objs[int(row)], counts, corpus_total)
            ig_abs = ig["ig"].abs().numpy()
            k = min(PHASE0_TOP_K, n_j)
            scores = {"IG": ig_abs, "dev": dev, "rare": rare}
            sets = {name: rso._top_k(arr, k) for name, arr in scores.items()}

            # identical RNG consumption order as the frozen phase0 driver
            rng = np.random.default_rng(70_000_000 + seed * 100_000 + int(row))
            complement = [i for i in range(n_j) if i not in set(sets["IG"])]
            rand_signed: list[float] = []
            for _ in range(PHASE0_RANDOM_DRAWS):
                if complement:
                    draw = rng.choice(np.asarray(complement), size=min(k, len(complement)), replace=False)
                    rand_signed.extend(rso._subset_response(model, pack, w_ref, draw).tolist())

            sig_resp = {name: rso._subset_response(model, pack, w_ref, idxs) for name, idxs in sets.items()}

            sig_by_j = {j: own.junction_signature(objs[int(row)], j) for j in range(n_j)}
            deg_by_j = {j: own.center_degree(objs[int(row)], j) for j in range(n_j)}
            junctions: list[dict[str, Any]] = []
            gaps_signed: list[float] = []
            gaps_abs: list[float] = []
            cover_hits = cover_total = 0
            dev_exact_hits = dev_exact_total = 0
            for pos, j in enumerate(sets["IG"]):
                cover_total += 1
                ctrl = own.matched_controls(
                    objs[int(row)],
                    j,
                    dev,
                    sig_by_j[j],
                    deg_by_j[j],
                    sig_by_j[j][3],
                    set(sets["IG"]),
                    PHASE0_MATCH_CONTROLS,
                    rng,
                )
                if not ctrl:
                    continue
                cover_hits += 1
                dev_exact_total += 1
                dev_exact_hits += sum(1 for c in ctrl if int(np.floor(dev[c] * 10.0)) == int(np.floor(dev[j] * 10.0)))
                ctrl_resp = rso._subset_response(model, pack, w_ref, ctrl)
                top_signed = float(sig_resp["IG"][pos])
                g_sig = top_signed - float(ctrl_resp.mean())
                g_abs = abs(top_signed) - float(np.abs(ctrl_resp).mean())
                gaps_signed.append(g_sig)
                gaps_abs.append(g_abs)
                junctions.append(
                    {
                        "j": int(j),
                        "top_signed": top_signed,
                        "top_abs": abs(top_signed),
                        "ctrl_signed_mean": float(ctrl_resp.mean()),
                        "ctrl_abs_mean": float(np.abs(ctrl_resp).mean()),
                        "ctrl_signed": [float(v) for v in ctrl_resp],
                        "n_ctrl": len(ctrl),
                        "gap_signed": g_sig,
                        "gap_abs": g_abs,
                        "dev_decile_exact": any(
                            int(np.floor(dev[c] * 10.0)) == int(np.floor(dev[j] * 10.0)) for c in ctrl
                        ),
                    }
                )

            h = hist_b[(seed, row)]
            if set(h["top4_sets"]["IG"]) != set(sets["IG"]):
                check["top4_sets_mismatch"].append([seed, row])
            for name in ("IG", "dev", "rare"):
                mine = float(np.abs(sig_resp[name]).mean())
                check["top4_response_max_abs_diff"] = max(
                    check["top4_response_max_abs_diff"], abs(mine - float(h["top4_response"][name]))
                )
            mine_rand = float(np.abs(np.asarray(rand_signed, dtype=np.float64)).mean()) if rand_signed else None
            if (mine_rand is None) != (h["random_response"] is None):
                check["random_response_max_abs_diff"] = float("inf")
            elif mine_rand is not None:
                check["random_response_max_abs_diff"] = max(
                    check["random_response_max_abs_diff"], abs(mine_rand - float(h["random_response"]))
                )
            mine_gap = float(np.mean(gaps_signed)) if gaps_signed else None
            if (mine_gap is None) != (h["matched_control_gap"] is None):
                check["matched_signed_gap_max_abs_diff"] = float("inf")
            elif mine_gap is not None:
                check["matched_signed_gap_max_abs_diff"] = max(
                    check["matched_signed_gap_max_abs_diff"], abs(mine_gap - float(h["matched_control_gap"]))
                )
            if (cover_hits, cover_total) != (int(h["matched_cover_hits"]), int(h["matched_cover_total"])):
                check["coverage_mismatch"].append([seed, row, cover_hits, cover_total])
            if (dev_exact_hits, dev_exact_total) != (int(h["matched_dev_exact_hits"]), int(h["matched_dev_exact_total"])):
                check["dev_exact_mismatch"].append([seed, row, dev_exact_hits, dev_exact_total])
            check["n_molecule_seed_units"] += 1
            records.append(
                {
                    "seed": seed,
                    "row": int(row),
                    "reference_row": int(ref_row),
                    "n_junctions": n_j,
                    "top4_sets": sets,
                    "resp_abs_mean": {name: float(np.abs(v).mean()) for name, v in sig_resp.items()},
                    "resp_signed_mean": {name: float(v.mean()) for name, v in sig_resp.items()},
                    "random_abs_mean": mine_rand,
                    "random_signed_mean": float(np.mean(rand_signed)) if rand_signed else None,
                    "gap_signed": mine_gap,
                    "gap_abs": float(np.mean(gaps_abs)) if gaps_abs else None,
                    "matched_cover_hits": cover_hits,
                    "matched_cover_total": cover_total,
                    "dev_exact_hits": dev_exact_hits,
                    "dev_exact_total": dev_exact_total,
                    "junctions": junctions,
                }
            )
        log(f"[recompute] seed {seed} done ({len(records)} units)")
    check["top4_response_max_abs_diff"] = float(check["top4_response_max_abs_diff"])
    check["random_response_max_abs_diff"] = float(check["random_response_max_abs_diff"])
    check["matched_signed_gap_max_abs_diff"] = float(check["matched_signed_gap_max_abs_diff"])
    check["reproduction_pass"] = bool(
        not check["top4_sets_mismatch"]
        and not check["coverage_mismatch"]
        and not check["dev_exact_mismatch"]
        and check["top4_response_max_abs_diff"] == 0.0
        and check["random_response_max_abs_diff"] == 0.0
        and check["matched_signed_gap_max_abs_diff"] <= 1e-12
    )
    payload = {
        "review": "bjoc_source_ownership_phase0_review_v1",
        "source_of_truth": {
            "phase0_results_sha256": _sha(ORIG / "phase0_results.json"),
            "world_git_commit": rso.git_commit() if hasattr(rso, "git_commit") else None,
        },
        "reproduction_check": check,
        "records": records,
        "labels_used": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "review_matched_records.json", payload)
    log(f"[recompute] reproduction pass: {check['reproduction_pass']}")
    return payload


# ---------------------------------------------------------------------------
# stage 2 — corrected statistics
# ---------------------------------------------------------------------------


def _matched_arrays(records: list[dict[str, Any]]) -> dict[str, dict[tuple[int, int], float]]:
    out = {
        "gap_signed": {},
        "gap_abs": {},
        "top_signed": {},
        "top_abs": {},
        "ctrl_signed": {},
        "ctrl_abs": {},
    }
    for r in records:
        key = (int(r["seed"]), int(r["row"]))
        out["gap_signed"][key] = r["gap_signed"]
        out["gap_abs"][key] = r["gap_abs"]
        js = r["junctions"]
        if js:
            out["top_signed"][key] = float(np.mean([j["top_signed"] for j in js]))
            out["top_abs"][key] = float(np.mean([j["top_abs"] for j in js]))
            out["ctrl_signed"][key] = float(np.mean([j["ctrl_signed_mean"] for j in js]))
            out["ctrl_abs"][key] = float(np.mean([j["ctrl_abs_mean"] for j in js]))
    return out


def _hist_contrasts(field: str, other: str) -> dict[tuple[int, int], float]:
    orig = read_json(ORIG / "phase0_results.json")
    out = {}
    for r in orig["records_b"]:
        if r[field] is None or r[other] is None:
            continue
        out[(int(r["seed"]), int(r["row"]))] = float(r[field]) - float(r[other])
    return out


def _phase0c(orig: dict[str, Any]) -> list[dict[str, Any]]:
    return list(orig["records_c"])


def _sig_tables(records_c: list[dict[str, Any]]) -> dict[str, Any]:
    unc = sorted({tuple(r["signature"]) for r in records_c}, key=lambda s: -sum(1 for r in records_c if tuple(r["signature"]) == s))
    sig_to_id = {s: i for i, s in enumerate(unc)}
    rec_sig_ids = np.asarray([sig_to_id[tuple(r["signature"])] for r in records_c], dtype=np.int64)
    class_sample = np.bincount(rec_sig_ids, minlength=len(unc))
    hist = read_json(ORIG / "phase0_decision.json")
    corpus_map = {tuple(c["signature"]): int(c["corpus_count"]) for c in hist["phase0_c"]["classes"]}
    class_freq = np.asarray([corpus_map.get(s, 0) / 346358.0 for s in unc], dtype=np.float64)
    frequent_mask = np.asarray([corpus_map.get(s, 0) >= 100 for s in unc], dtype=bool)
    return {
        "n_classes": len(unc),
        "sig_ids": rec_sig_ids,
        "class_sample": class_sample,
        "class_freq": class_freq,
        "frequent_mask": frequent_mask,
    }


def _by_mol(records_c: list[dict[str, Any]]) -> dict[tuple[int, int], list[int]]:
    out: dict[tuple[int, int], list[int]] = {}
    for i, r in enumerate(records_c):
        out.setdefault((int(r["seed"]), int(r["row"])), []).append(i)
    return out


def _by_row(records_c: list[dict[str, Any]]) -> dict[int, dict[int, list[int]]]:
    out: dict[int, dict[int, list[int]]] = {}
    for i, r in enumerate(records_c):
        out.setdefault(int(r["row"]), {}).setdefault(int(r["seed"]), []).append(i)
    return out


def _enrichment_stat(sig_ids: np.ndarray, class_freq: np.ndarray, frequent_mask: np.ndarray, top_flags: np.ndarray, n_classes: int) -> float:
    n_top = int(top_flags.sum())
    if n_top == 0:
        return 0.0
    obs = np.bincount(sig_ids[top_flags], minlength=n_classes).astype(np.float64)
    exp = n_top * class_freq
    mask = frequent_mask & (exp >= 3.0)
    if not mask.any():
        return 0.0
    return float(np.max(obs[mask] / np.maximum(exp[mask], 1e-9)))


def _sign_stat(sig_ids: np.ndarray, class_sample: np.ndarray, values: np.ndarray, threshold: float | None) -> float:
    v = values if threshold is None else values - threshold
    s = np.sign(v)
    num = np.bincount(sig_ids, weights=s, minlength=len(class_sample))
    cnt = np.bincount(sig_ids, minlength=len(class_sample))
    mask = class_sample >= 30
    if not mask.any():
        return 0.0
    agree = np.abs(num[mask] / np.maximum(cnt[mask], 1))
    return float(np.sum(cnt[mask] * agree) / np.sum(cnt[mask]))


def _top_flags(records_c: list[dict[str, Any]], by_mol: Mapping[tuple[int, int], list[int]]) -> np.ndarray:
    top = np.zeros(len(records_c), dtype=bool)
    ig_abs = np.asarray([r["ig_abs"] for r in records_c], dtype=np.float64)
    for idxs in by_mol.values():
        k = max(1, int(round(0.1 * len(idxs))))
        best = np.argsort(-ig_abs[idxs], kind="stable")[:k]
        for t in best:
            top[idxs[int(t)]] = True
    return top


def _permutation_stats(
    records_c: list[dict[str, Any]],
    *,
    joint: bool,
    rng_seed: int,
    draws: int = N_PERMUTATION,
) -> dict[str, Any]:
    tables = _sig_tables(records_c)
    sig_ids = tables["sig_ids"]
    n_classes = tables["n_classes"]
    centered = np.asarray([r["ig_centered"] for r in records_c], dtype=np.float64)
    ig_abs = np.asarray([r["ig_abs"] for r in records_c], dtype=np.float64)
    median_centered = float(np.median(centered))
    by_mol = _by_mol(records_c)
    top_obs = _top_flags(records_c, by_mol)

    def stats(values: np.ndarray, top: np.ndarray) -> tuple[float, float, float]:
        return (
            _enrichment_stat(sig_ids, tables["class_freq"], tables["frequent_mask"], top, n_classes),
            _sign_stat(sig_ids, tables["class_sample"], values, median_centered),  # as-implemented
            _sign_stat(sig_ids, tables["class_sample"], values, None),  # reported semantics
        )

    obs = stats(centered, top_obs)
    rng = np.random.default_rng(rng_seed)
    null = {"enrichment": [], "sign_as_implemented": [], "sign_reported": []}
    if joint:
        by_row = _by_row(records_c)
        for _ in range(draws):
            perm_top = top_obs.copy()
            perm_vals = centered.copy()
            for row, seed_idx in by_row.items():
                seeds = sorted(seed_idx)
                if len(seeds) == 2 and len(seed_idx[seeds[0]]) == len(seed_idx[seeds[1]]):
                    m = len(seed_idx[seeds[0]])
                    perm = rng.permutation(m)
                    for s in seeds:
                        idxs = np.asarray(seed_idx[s], dtype=np.int64)
                        k = max(1, int(round(0.1 * m)))
                        best = np.argsort(-ig_abs[idxs][perm], kind="stable")[:k]
                        flags = np.zeros(m, dtype=bool)
                        flags[best] = True
                        perm_top[idxs] = flags
                        perm_vals[idxs] = centered[idxs][perm]
                else:  # fall back to independent permutation (no pair to preserve)
                    for s in seeds:
                        idxs = np.asarray(seed_idx[s], dtype=np.int64)
                        perm = rng.permutation(len(idxs))
                        k = max(1, int(round(0.1 * len(idxs))))
                        best = np.argsort(-ig_abs[idxs][perm], kind="stable")[:k]
                        flags = np.zeros(len(idxs), dtype=bool)
                        flags[best] = True
                        perm_top[idxs] = flags
                        perm_vals[idxs] = centered[idxs][perm]
            e, s_impl, s_rep = stats(perm_vals, perm_top)
            null["enrichment"].append(e)
            null["sign_as_implemented"].append(s_impl)
            null["sign_reported"].append(s_rep)
    else:
        for _ in range(draws):
            perm_top = top_obs.copy()
            perm_vals = centered.copy()
            for idxs in by_mol.values():  # historical iteration order
                perm = rng.permutation(len(idxs))
                k = max(1, int(round(0.1 * len(idxs))))
                best = np.argsort(-ig_abs[idxs][perm], kind="stable")[:k]
                flags = np.zeros(len(idxs), dtype=bool)
                flags[best] = True
                perm_top[idxs] = flags
                perm_vals[idxs] = centered[idxs][perm]
            e, s_impl, s_rep = stats(perm_vals, perm_top)
            null["enrichment"].append(e)
            null["sign_as_implemented"].append(s_impl)
            null["sign_reported"].append(s_rep)

    def pval(null_vals: list[float], observed: float) -> float:
        arr = np.asarray(null_vals, dtype=np.float64)
        return float((1 + int(np.sum(arr >= observed))) / (1 + len(arr)))

    return {
        "joint": bool(joint),
        "rng_seed": int(rng_seed),
        "draws": int(draws),
        "median_centered_ig": median_centered,
        "observed": {
            "enrichment": obs[0],
            "sign_as_implemented": obs[1],
            "sign_reported_semantics": obs[2],
        },
        "null_p95": {k: float(np.quantile(v, 0.95)) for k, v in null.items()},
        "p": {
            "enrichment": pval(null["enrichment"], obs[0]),
            "sign_as_implemented": pval(null["sign_as_implemented"], obs[1]),
            "sign_reported_semantics": pval(null["sign_reported"], obs[2]),
        },
    }


def analyze() -> dict[str, Any]:
    orig = read_json(ORIG / "phase0_results.json")
    matched = read_json(OUT / "review_matched_records.json")
    records = matched["records"]
    arr = _matched_arrays(records)

    # ---- R1: signed vs absolute matched -----------------------------------
    direction = []
    for r in records:
        if r["gap_signed"] is None or r["gap_abs"] is None:
            continue
        if abs(r["gap_signed"]) < 1e-15 or abs(r["gap_abs"]) < 1e-15:
            continue
        direction.append(1.0 if np.sign(r["gap_signed"]) == np.sign(r["gap_abs"]) else 0.0)
    coverage_hits = float(sum(r["matched_cover_hits"] for r in records))
    coverage_total = float(sum(r["matched_cover_total"] for r in records))
    dev_hits = float(sum(r["dev_exact_hits"] for r in records))
    dev_total = float(sum(r["dev_exact_total"] for r in records))
    r1 = {
        "signed_gap": summarize_values(arr["gap_signed"]),
        "absolute_gap": summarize_values(arr["gap_abs"]),
        "top_vs_controls_abs_means": {
            "top_abs": summarize_values(arr["top_abs"])["pooled_molecule_seed"],
            "ctrl_abs": summarize_values(arr["ctrl_abs"])["pooled_molecule_seed"],
        },
        "direction_consistency": {
            "n_units": len(direction),
            "n_same_sign": int(np.sum(np.asarray(direction) > 0)),
            "fraction": float(np.mean(direction)) if direction else None,
            "p_one_sided_same_sign": one_sided_sign(np.asarray(direction))["p_one_sided"] if direction else None,
        },
        "coverage": {
            "junction_coverage": coverage_hits / coverage_total if coverage_total else None,
            "cover_hits": int(coverage_hits),
            "cover_total": int(coverage_total),
            "dev_decile_exact_fraction": dev_hits / dev_total if dev_total else None,
            "dev_exact_hits": int(dev_hits),
            "dev_exact_total": int(dev_total),
            "per_seed_coverage": {},
        },
    }
    for seed in SEEDS:
        ch = sum(r["matched_cover_hits"] for r in records if r["seed"] == seed)
        ct = sum(r["matched_cover_total"] for r in records if r["seed"] == seed)
        dh = sum(r["dev_exact_hits"] for r in records if r["seed"] == seed)
        dt = sum(r["dev_exact_total"] for r in records if r["seed"] == seed)
        r1["coverage"]["per_seed_coverage"][str(seed)] = {
            "junction_coverage": ch / ct,
            "dev_decile_exact_fraction": dh / dt if dt else None,
        }

    # ---- R2: unit corrections for all contrasts ---------------------------
    orig_recs = {(int(r["seed"]), int(r["row"])): r for r in orig["records_b"]}
    contrasts: dict[str, dict[tuple[int, int], float]] = {
        "matched_signed": arr["gap_signed"],
        "matched_absolute": arr["gap_abs"],
    }
    contrasts["IG_minus_dev"] = {
        k: float(r["top4_response"]["IG"] - r["top4_response"]["dev"]) for k, r in orig_recs.items()
    }
    contrasts["IG_minus_rare"] = {
        k: float(r["top4_response"]["IG"] - r["top4_response"]["rare"]) for k, r in orig_recs.items()
    }
    contrasts["IG_minus_random"] = {
        k: float(r["top4_response"]["IG"] - r["random_response"]) for k, r in orig_recs.items() if r["random_response"] is not None
    }
    # absolute-response versions of the ranking contrasts (recomputed only for IG/dev/rare)
    con_abs: dict[tuple[int, int], float] = {}
    for r in records:
        con_abs[(r["seed"], r["row"])] = float(r["resp_abs_mean"]["IG"] - r["resp_abs_mean"]["dev"])
    contrasts["IG_minus_dev_abs_resp"] = con_abs
    con_abs_rare: dict[tuple[int, int], float] = {}
    for r in records:
        con_abs_rare[(r["seed"], r["row"])] = float(r["resp_abs_mean"]["IG"] - r["resp_abs_mean"]["rare"])
    contrasts["IG_minus_rare_abs_resp"] = con_abs_rare
    r2 = {name: summarize_values(vals) for name, vals in contrasts.items()}
    r2["_note"] = (
        "pooled_molecule_seed = historical unit; row_unit = 256 molecule identities "
        "(both seeds resampled together); row_unit_complete_pairs requires both seeds"
    )

    # ---- R3: 0C null + sign semantics ------------------------------------
    records_c = _phase0c(orig)
    independent = _permutation_stats(records_c, joint=False, rng_seed=RNG_INDEPENDENT_PERM)
    joint = _permutation_stats(records_c, joint=True, rng_seed=RNG_JOINT_PERM)
    per_seed = {}
    for seed in SEEDS:
        subset = [r for r in records_c if r["seed"] == seed]
        per_seed[str(seed)] = _permutation_stats(subset, joint=False, rng_seed=RNG_INDEPENDENT_PERM)
    r3 = {
        "historical": {
            k: orig["phase0_c"][k] for k in ("enrichment_stat", "enrichment_p", "sign_stat", "sign_p", "n_classes", "n_classes_frequent")
        },
        "independent_reseed": independent,
        "joint_cross_seed": joint,
        "per_seed": per_seed,
        "implementation_vs_reported": {
            "implemented": "sign(v_j - median(v)) with v = pooled within-molecule-centered IG",
            "reported": "sign(centered IG_j)",
            "median_v": independent["median_centered_ig"],
            "as_implemented_sign_stat": independent["observed"]["sign_as_implemented"],
            "reported_semantics_sign_stat": independent["observed"]["sign_reported_semantics"],
        },
    }

    payload = {
        "review": "bjoc_source_ownership_phase0_review_v1",
        "source_sha256": {
            "phase0_results.json": _sha(ORIG / "phase0_results.json"),
            "review_matched_records.json": _sha(OUT / "review_matched_records.json"),
        },
        "reproduction": read_json(OUT / "review_matched_records.json")["reproduction_check"],
        "R1_signed_vs_absolute": r1,
        "R2_units": r2,
        "R3_phase0c": r3,
        "labels_used": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "review_results.json", payload)
    log("[analyze] written")
    return payload


# ---------------------------------------------------------------------------
# stage 3 — decision
# ---------------------------------------------------------------------------


def decide() -> dict[str, Any]:
    res = read_json(OUT / "review_results.json")
    r1 = res["R1_signed_vs_absolute"]
    r2 = res["R2_units"]
    r3 = res["R3_phase0c"]

    p1 = r2["IG_minus_dev"]["row_unit"]
    p1_pass = bool(p1["mean"] > 0 and p1["ci95_lower"] > 0 and p1["p_one_sided"] < 0.01)
    p2 = r2["matched_absolute"]["row_unit"]
    p2_pass = bool(p2["mean"] > 0 and p2["ci95_lower"] > 0 and p2["p_one_sided"] < 0.05)
    joint = r3["joint_cross_seed"]
    p3_enrich = joint["p"]["enrichment"] < 0.05
    p3_sign_reported = joint["p"]["sign_reported_semantics"] < 0.05
    p3_sign_impl = joint["p"]["sign_as_implemented"] < 0.05
    p3_pass = bool(p3_enrich and p3_sign_reported)

    if p1_pass and p2_pass and p3_pass:
        verdict = "SOURCE_SIGNAL_CONFIRMED_WITH_CORRECTED_STATISTICS"
    elif (not p1_pass) or ((not p2_pass) and (not p3_pass)):
        verdict = "SOURCE_SIGNAL_INCONCLUSIVE"
    else:
        verdict = "SOURCE_SIGNAL_PARTIALLY_SUPPORTED"

    payload = {
        "review": "bjoc_source_ownership_phase0_review_v1",
        "propositions": {
            "P1_IG_beyond_dev_corrected_unit": {
                "pass": p1_pass,
                "evidence": p1,
                "rule": "row-unit mean > 0 and CI95 lower > 0 and one-sided sign p < 0.01",
                "per_seed": {s: r2["IG_minus_dev"]["per_seed"][s] for s in r2["IG_minus_dev"]["per_seed"]},
            },
            "P2_absolute_matched_gap_corrected_unit": {
                "pass": p2_pass,
                "evidence": p2,
                "rule": "row-unit mean > 0 and CI95 lower > 0 and one-sided sign p < 0.05",
                "signed_counterpart": r2["matched_signed"]["row_unit"],
                "direction_consistency": r1["direction_consistency"],
                "coverage": r1["coverage"],
            },
            "P3_phase0c_corrected_null": {
                "pass": p3_pass,
                "enrichment_joint_p": joint["p"]["enrichment"],
                "sign_reported_joint_p": joint["p"]["sign_reported_semantics"],
                "sign_as_implemented_joint_p": joint["p"]["sign_as_implemented"],
                "rule": "joint-permutation p < 0.05 for enrichment AND reported-semantics sign",
                "independent_null_reproduction": r3["independent_reseed"]["p"],
            },
        },
        "verdict": verdict,
        "historical_verdict_untouched": "SOURCE_SIGNAL_BEYOND_SIMPLE_BASELINES",
        "labels_used": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "review_decision.json", payload)
    log(f"[decide] {verdict}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(prog="review_bjoc_source_ownership_phase0_v1")
    parser.add_argument("stage", choices=["recompute", "analyze", "decide", "all"])
    args = parser.parse_args()
    t0 = time.perf_counter()
    if args.stage in ("recompute", "all"):
        recompute()
    if args.stage in ("analyze", "all"):
        analyze()
    if args.stage in ("decide", "all"):
        decide()
    log(f"done in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
