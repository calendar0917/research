#!/usr/bin/env python3
"""E2E-DictEnv-A1 — post-hoc Gate-0 stratum diagnostics (label-free, official train only).

The frozen G0.3 continuity gate failed (REAL code-space AUC 0.5344 < 0.70) and the
round stopped at Gate 0 with verdict ``REPRESENTATION_NOT_QUALIFIED``.  This script
does **not** re-open that verdict: it only characterises *why* the frozen stratum
behaves the way it does, so a future pre-registration can replace the metric with a
sound one.

Everything here is label-free and reads the frozen, already-audited artifacts of
this round (``tracks/ksvd/results/e2e_dictenv_a1``) plus the raw official-train
cache.  The official ZINC test split is never touched.

Reported per stratum (near = top-800 attributed-WL cosine pairs, tail = next 800,
random = the frozen (size, rootcat)-matched control):
  * WL-cosine distribution of the pairs and the pool-wide tie structure,
  * size / root-category / molecule-disjointness composition,
  * x-space and code-space distance means per arm,
  * near-vs-control AUCs reproduced exactly, then restricted to molecule-disjoint
    and/or tie-free pairs,
  * graded Spearman(WL cosine, -distance) over sampled same-class pairs.

Output: ``tracks/ksvd/results/e2e_dictenv_a1/gate0_stratum_diagnostics.json``
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.code.run_tccd_v0 import _auc_paired  # noqa: E402  (the frozen gate's AUC)
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_a1 as a1  # noqa: E402
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_a1 as run  # noqa: E402


def _auc(near_dist: np.ndarray, rand_dist: np.ndarray) -> float:
    """Unpaired AUC: P(distance(near) < distance(control)) over all pairs."""
    near = np.asarray(near_dist, dtype=np.float64)[:, None]
    rand = np.asarray(rand_dist, dtype=np.float64)[None, :]
    return float((near < rand).mean() + 0.5 * (near == rand).mean())


def _angular(table: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """1 - cos between code vectors (scale invariant)."""
    idx = np.asarray(pairs, dtype=np.int64)
    a, b = table[idx[:, 0]], table[idx[:, 1]]
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return 1.0 - np.einsum("ij,ij->i", a, b) / np.maximum(na * nb, 1e-30)


def _distances(space: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.zeros(0, dtype=np.float64)
    idx = np.asarray(pairs, dtype=np.int64)
    return np.linalg.norm(space[idx[:, 0]] - space[idx[:, 1]], axis=1)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 3:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(left)), np.argsort(np.argsort(right)))[0, 1])


def main() -> int:
    started = time.perf_counter()
    data = run.continuity_pool_data()
    sim = np.asarray(data["similarity"], dtype=np.float64)
    sizes = np.asarray(data["sizes"], dtype=np.int64)
    rootcats = np.asarray(data["rootcats"], dtype=np.int64)
    keys = list(data["keys"])
    molecules = np.asarray([mol for mol, _root in data["pool"]], dtype=np.int64)
    n_pool = sizes.shape[0]

    near_pairs = list(data["near_pairs"])
    tail_pairs = list(data["tail_pairs"])
    random_pairs = list(data["random_pairs"])
    iso_pairs = list(data["isomorphic_pairs"])

    def pair_stats(pairs: list[tuple[int, int]]) -> dict:
        if not pairs:
            return {"n_pairs": 0}
        idx = np.asarray(pairs, dtype=np.int64)
        cos = np.asarray([sim[i, j] for i, j in pairs], dtype=np.float64)
        return {
            "n_pairs": len(pairs),
            "wl_cosine_mean": float(cos.mean()),
            "wl_cosine_min": float(cos.min()),
            "frac_cosine_ge_0.9999": float((cos >= 0.9999).mean()),
            "frac_cosine_eq_1.0": float((cos >= 1.0 - 1e-12).mean()),
            "size_equal_fraction": float((sizes[idx[:, 0]] == sizes[idx[:, 1]]).mean()),
            "size_mean_pair": float((0.5 * (sizes[idx[:, 0]] + sizes[idx[:, 1]])).mean()),
            "rootcat_equal_fraction": float((rootcats[idx[:, 0]] == rootcats[idx[:, 1]]).mean()),
            "matched_class_fraction": float(
                (
                    (sizes[idx[:, 0]] == sizes[idx[:, 1]])
                    & (rootcats[idx[:, 0]] == rootcats[idx[:, 1]])
                ).mean()
            ),
            "same_molecule_fraction": float((molecules[idx[:, 0]] == molecules[idx[:, 1]]).mean()),
            "same_slot_key_fraction": float(np.mean([keys[i] == keys[j] for i, j in pairs])),
        }

    # ---- pool-wide tie structure of the attributed-WL fingerprint -------------
    upper = np.triu(np.ones((n_pool, n_pool), dtype=bool), k=1)
    distinct_key = np.asarray(
        [[keys[i] != keys[j] for j in range(n_pool)] for i in range(n_pool)], dtype=bool
    )
    pool_pairs_mask = upper & distinct_key
    pool_cos = sim[pool_pairs_mask]
    pool_sizes = sizes.astype(np.float64)
    tie_structure = {
        "pool_size_mean": float(pool_sizes.mean()),
        "pool_size_median": float(np.median(pool_sizes)),
        "pool_size_ge10_fraction": float((pool_sizes >= 10).mean()),
        "n_pool_pairs_distinct_key": int(pool_pairs_mask.sum()),
        "n_pairs_cosine_ge_0.9999": int((pool_cos >= 0.9999).sum()),
        "n_pairs_cosine_eq_1.0": int((pool_cos >= 1.0 - 1e-12).sum()),
        "cosine_top1pct": float(np.quantile(pool_cos, 0.99)),
        "cosine_median": float(np.median(pool_cos)),
        "cosine_mean": float(pool_cos.mean()),
        "near_selection_is_inside_tie": bool((pool_cos >= 1.0 - 1e-12).sum() >= len(near_pairs)),
        "near_cosine_threshold": float(min(s for *_rest, s in data["near"])),
        "tail_cosine_threshold_min": float(min(s for *_rest, s in data["tail"])),
    }

    # ---- per-arm distances ---------------------------------------------------
    widths = {"TOPO": a1.PHI_DIM, "INDEP": a1.A1_DIM, "REAL": a1.A1_DIM}
    arms = tuple(data["arms"])
    strata = {"near": near_pairs, "tail": tail_pairs, "random": random_pairs}
    distance_table: dict[str, dict[str, dict[str, float]]] = {}
    auc_table: dict[str, dict[str, float]] = {}
    restricted: dict[str, dict[str, float]] = {}
    for arm in arms:
        rows = np.asarray(data["rows"][arm], dtype=np.float64)
        width = widths[arm]
        x_space, code_space = rows[:, :width], rows[:, width:]
        distance_table[arm] = {}
        for name, pairs in strata.items():
            distance_table[arm][name] = {
                "x_mean": float(_distances(x_space, pairs).mean()) if pairs else float("nan"),
                "code_mean": float(_distances(code_space, pairs).mean()) if pairs else float("nan"),
            }
        d_near_x = _distances(x_space, near_pairs)
        d_rand_x = _distances(x_space, random_pairs)
        d_near_c = _distances(code_space, near_pairs)
        d_rand_c = _distances(code_space, random_pairs)
        d_tail_x = _distances(x_space, tail_pairs)
        d_tail_rand_x = _distances(x_space, data["tail_random"])
        d_tail_c = _distances(code_space, tail_pairs)
        d_tail_rand_c = _distances(code_space, data["tail_random"])
        auc_table[arm] = {
            # the frozen gate uses the paired statistic; both are reported
            "frozen_paired_x_auc": _auc_paired(d_near_x, d_rand_x),
            "frozen_paired_code_auc": _auc_paired(d_near_c, d_rand_c),
            "unpaired_x_auc": _auc(d_near_x, d_rand_x),
            "unpaired_code_auc": _auc(d_near_c, d_rand_c),
            "tail_paired_x_auc": _auc_paired(d_tail_x, d_tail_rand_x),
            "tail_paired_code_auc": _auc_paired(d_tail_c, d_tail_rand_c),
        }
        mol_disjoint_near = [p for p in near_pairs if molecules[p[0]] != molecules[p[1]]]
        mol_disjoint_rand = [p for p in random_pairs if molecules[p[0]] != molecules[p[1]]]
        disjoint_near_x = _distances(x_space, mol_disjoint_near)
        disjoint_rand_x = _distances(x_space, mol_disjoint_rand)
        disjoint_near_c = _distances(code_space, mol_disjoint_near)
        disjoint_rand_c = _distances(code_space, mol_disjoint_rand)
        # scale-invariant and scale-matched variants of the same comparison
        ang_near = _angular(code_space, near_pairs)
        ang_rand = _angular(code_space, random_pairs)
        norm_near = np.linalg.norm(code_space[np.asarray(near_pairs, dtype=np.int64)[:, 0]], axis=1)
        norm_rand = np.linalg.norm(code_space[np.asarray(random_pairs, dtype=np.int64)[:, 0]], axis=1)
        ratio_near = np.abs(norm_near - norm_rand) / np.maximum(norm_near + norm_rand, 1e-30)
        keep_ratio = ratio_near <= 0.05
        restricted[arm] = {
            "n_near_mol_disjoint": len(mol_disjoint_near),
            "n_random_mol_disjoint": len(mol_disjoint_rand),
            "mol_disjoint_paired_x_auc": _auc_paired(disjoint_near_x, disjoint_rand_x),
            "mol_disjoint_paired_code_auc": _auc_paired(disjoint_near_c, disjoint_rand_c),
            "mol_disjoint_near_code_mean": float(disjoint_near_c.mean()),
            "mol_disjoint_random_code_mean": float(disjoint_rand_c.mean()),
            "angular_paired_code_auc": _auc_paired(ang_near, ang_rand),
            "angular_near_code_mean": float(ang_near.mean()),
            "angular_random_code_mean": float(ang_rand.mean()),
            "code_norm_near_mean": float(norm_near.mean()),
            "code_norm_random_mean": float(norm_rand.mean()),
            "norm_ratio_matched_fraction": float(keep_ratio.mean()),
            "norm_ratio_matched_paired_code_auc": _auc_paired(
                d_near_c[keep_ratio], d_rand_c[keep_ratio]
            ) if keep_ratio.sum() >= 3 else None,
            "norm_ratio_matched_n": int(keep_ratio.sum()),
        }

    # ---- graded relation over sampled same-class pairs -----------------------
    rng = np.random.default_rng(run.CONTINUITY_SEED + 1)
    sample = np.column_stack(
        [rng.integers(0, n_pool, size=run.POSTHOC_PAIRS), rng.integers(0, n_pool, size=run.POSTHOC_PAIRS)]
    )
    keep = (sample[:, 0] < sample[:, 1]) & np.asarray(
        [keys[i] != keys[j] for i, j in sample], dtype=bool
    )
    pairs = sample[keep]
    cosines = np.asarray([sim[i, j] for i, j in pairs], dtype=np.float64)
    equal_size = np.asarray([sizes[i] == sizes[j] for i, j in pairs], dtype=bool)
    mol_disjoint = np.asarray([molecules[i] != molecules[j] for i, j in pairs], dtype=bool)
    graded: dict[str, dict[str, float]] = {}
    for arm in arms:
        rows = np.asarray(data["rows"][arm], dtype=np.float64)
        width = widths[arm]
        x_space, code_space = rows[:, :width], rows[:, width:]
        idx = np.asarray(pairs, dtype=np.int64)
        x_dist = np.linalg.norm(x_space[idx[:, 0]] - x_space[idx[:, 1]], axis=1)
        c_dist = np.linalg.norm(code_space[idx[:, 0]] - code_space[idx[:, 1]], axis=1)
        strict = equal_size & mol_disjoint
        graded[arm] = {
            "n_sampled_pairs": int(pairs.shape[0]),
            "n_equal_size_mol_disjoint": int(strict.sum()),
            "spearman_cosine_vs_neg_xdist_all": _spearman(cosines, -x_dist),
            "spearman_cosine_vs_neg_codedist_all": _spearman(cosines, -c_dist),
            "spearman_cosine_vs_neg_xdist_equal_size_mol_disjoint": _spearman(
                cosines[strict], -x_dist[strict]
            ),
            "spearman_cosine_vs_neg_codedist_equal_size_mol_disjoint": _spearman(
                cosines[strict], -c_dist[strict]
            ),
            "x_dist_mean_equal_size": float(x_dist[equal_size].mean()),
            "x_dist_mean_unequal_size": float(x_dist[~equal_size].mean()),
            "code_dist_mean_equal_size": float(c_dist[equal_size].mean()),
            "code_dist_mean_unequal_size": float(c_dist[~equal_size].mean()),
        }

    payload = {
        "protocol_version": a1.PROTOCOL_VERSION,
        "kind": "posthoc (local analysis of pulled frozen artifacts; cannot change the verdict)",
        "gate_verdict": "REPRESENTATION_NOT_QUALIFIED",
        "frozen_gate": {
            "metric": "real.code_auc",
            "threshold": a1.CONTINUITY_AUC_PASS,
            "measured": 0.534375,
        },
        "pool": {"n_pool": int(n_pool), "seed": int(run.CONTINUITY_SEED), "rounds": int(a1.CONTINUITY_ROUNDS)},
        "strata": {
            "near": pair_stats(near_pairs),
            "tail": pair_stats(tail_pairs),
            "random_matched": pair_stats(random_pairs),
            "tail_random_matched": pair_stats(list(data["tail_random"])),
            "isomorphic_control": pair_stats(iso_pairs),
        },
        "tie_structure": tie_structure,
        "distances": distance_table,
        "aucs": auc_table,
        "restricted_aucs": restricted,
        "graded": graded,
        "official_test_loaded": False,
        "seconds": float(time.perf_counter() - started),
    }

    out_path = run.RESULTS_DIR / "gate0_stratum_diagnostics.json"
    out_path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("strata", "tie_structure", "aucs", "restricted_aucs", "graded")}, indent=1))
    print(f"wrote {out_path} in {payload['seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
