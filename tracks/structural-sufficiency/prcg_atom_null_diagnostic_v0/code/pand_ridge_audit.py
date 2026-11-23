"""PRCG-ATOM-NULL part A: audit the PRCG-v0 predictive merge screen.

The frozen PRCG-v0 discovery screen (``RoundScreener``) computes a held-out
MSE gain for adding one candidate count column to a ridge model.  It
residualises the candidate against the current design ``X``, then uses

    beta = resid_tr^T y / (resid_tr^T resid_tr + lambda).

The strict augmented ridge, with the candidate column appended to the design
and the same lambda on its coefficient, has the (Frisch-Waugh) denominator

    c_tr^T resid_tr + lambda.

The two denominators differ by ``lambda * ||S||_P^2``, where
``S = (X^T X + lambda P)^{-1} X^T c``.  This script replays every predictive
discovery round, scores every original candidate both ways (including a direct
solve of the augmented normal equations), and records whether the selected
rules, their signs, or the candidate rankings change.

Read-only with respect to PRCG-v0.  CPU only; no neural network is trained.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import pand_common as P

np.set_printoptions(precision=6, suppress=True)


# ---------------------------------------------------------------------------
# strict augmented ridge
# ---------------------------------------------------------------------------
def _penalty(p: int, lam: float) -> np.ndarray:
    pen = np.ones(int(p), np.float64)
    pen[0] = 0.0
    return lam * pen


def strict_scores(x_matrix: np.ndarray, y_std: np.ndarray, fold: np.ndarray,
                  candidate_matrix: np.ndarray, lam: float = 5.0
                  ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Direct augmented-ridge held-out improvements, plus the Schur diagnostics.

    Returns
    -------
    direct : (2, M) array
        Base held-out MSE minus the held-out MSE of the *directly solved*
        augmented ridge, for each fold direction and candidate.
    formula : (2, M) array
        The same quantity using the Frisch-Waugh denominator
        ``c_tr^T resid_tr + lambda``; must equal ``direct`` up to round-off.
    detail : dict
        Per-fold/candidate residual, denominator and coefficient diagnostics.
    """
    X = np.asarray(x_matrix, np.float64)
    y = np.asarray(y_std, np.float64)
    fold = np.asarray(fold, np.int8)
    C = np.asarray(candidate_matrix, np.float64)
    N, M = C.shape
    p = int(X.shape[1])
    pen = _penalty(p, float(lam))
    direct = np.zeros((2, M), np.float64)
    formula = np.zeros((2, M), np.float64)
    detail = {
        "resid_sq": np.zeros((2, M)),
        "c_dot_resid": np.zeros((2, M)),
        "gap": np.zeros((2, M)),
        "gap_identity": np.zeros((2, M)),
        "den_old": np.zeros((2, M)),
        "den_strict": np.zeros((2, M)),
        "beta_old": np.zeros((2, M)),
        "beta_strict": np.zeros((2, M)),
        "base_mse": np.zeros(2),
        "direct_coef": np.zeros((2, p + 1, M)),
    }
    for side in (0, 1):
        tr = fold != side
        te = fold == side
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        gram = Xtr.T @ Xtr + np.diag(pen)
        base_coef = np.linalg.solve(gram, Xtr.T @ ytr)
        base_te = Xte @ base_coef
        base_mse = float(np.mean((yte - base_te) ** 2))
        detail["base_mse"][side] = base_mse

        # Frisch-Waugh residualisation of every candidate against X
        S = np.linalg.solve(gram, Xtr.T @ C[tr])
        resid_tr = C[tr] - Xtr @ S
        resid_te = C[te] - Xte @ S
        num = resid_tr.T @ ytr
        resid_sq = np.einsum("ij,ij->j", resid_tr, resid_tr)
        c_dot = np.einsum("ij,ij->j", C[tr], resid_tr)
        gap = c_dot - resid_sq
        gap_identity = np.einsum("jk,jk->k", S, pen[:, None] * S)
        den_old = resid_sq + float(lam)
        den_strict = c_dot + float(lam)
        beta_old = num / den_old
        beta_strict = num / den_strict
        pred_old = base_te[:, None] + resid_te * beta_old[None, :]
        pred_strict = base_te[:, None] + resid_te * beta_strict[None, :]
        # The first is literally the PRCG-v0 RoundScreener formula; the second
        # is the strict augmented-ridge Schur-complement formula.
        formula_old = base_mse - np.mean((yte[:, None] - pred_old) ** 2, axis=0)
        formula[side] = base_mse - np.mean((yte[:, None] - pred_strict) ** 2, axis=0)

        for j in range(M):
            ztr = np.concatenate([Xtr, C[tr, j:j + 1]], axis=1)
            zte = np.concatenate([Xte, C[te, j:j + 1]], axis=1)
            pen_aug = np.concatenate([pen, [float(lam)]])
            gram_aug = ztr.T @ ztr + np.diag(pen_aug)
            coef = np.linalg.solve(gram_aug, ztr.T @ ytr)
            pred = zte @ coef
            direct[side, j] = base_mse - float(np.mean((yte - pred) ** 2))
            detail["direct_coef"][side, :, j] = coef
        detail["resid_sq"][side] = resid_sq
        detail["c_dot_resid"][side] = c_dot
        detail["gap"][side] = gap
        detail["gap_identity"][side] = gap_identity
        detail["den_old"][side] = den_old
        detail["den_strict"][side] = den_strict
        detail["beta_old"][side] = beta_old
        detail["beta_strict"][side] = beta_strict
    return direct, formula, detail


# ---------------------------------------------------------------------------
# replay helpers
# ---------------------------------------------------------------------------
def _build_round(disc: P.GD.GrammarDiscovery,
                 partitions: Mapping[int, P.G.Partition]
                 ) -> dict[str, Any]:
    cand, groups = disc.enumerate_round(partitions)
    legal = {k: c for k, c in cand.items()
             if c.parent_size <= P.G.MAX_WORD_ATOMS
             and c.n_molecules >= P.GD.MIN_FREQ_MOLECULES}
    ranked = sorted(legal.values(), key=lambda c: (-c.n_molecules, c.pattern))
    shortlist = ranked[:P.GD.CANDIDATE_CAP]

    vocab: dict[str, int] = {}
    for r in disc.rows.tolist():
        for w in partitions[int(r)].words:
            vocab.setdefault(w.cert, len(vocab))
    x = np.zeros((disc.rows.size, len(vocab)), np.float64)
    for i, r in enumerate(disc.rows.tolist()):
        for w in partitions[int(r)].words:
            j = vocab.get(w.cert)
            if j is not None:
                x[i, j] += 1.0
    feats = P.GD._column_standardise(np.concatenate([x, disc.graph], axis=1))
    y = disc.y
    y = (y - y.mean()) / (y.std() if y.std() > 1e-12 else 1.0)
    C = np.zeros((disc.rows.size, len(shortlist)), np.float64)
    for i, g in enumerate(groups):
        for j, c in enumerate(shortlist):
            items = g.get(c.pattern)
            if items:
                C[i, j] = len(P.G.greedy_select(items))
    return {"cand": cand, "groups": groups, "legal": legal, "shortlist": shortlist,
            "feats": feats, "y": y, "C": C, "vocab": vocab}


def _select(candidates: Sequence[P.GD.Candidate], score: np.ndarray,
            eligible: np.ndarray, used: set[str], resolver,
            kind: str = "predictive") -> tuple[P.GD.Candidate | None, str | None,
                                               list[dict[str, Any]]]:
    """The exact PRCG-v0 constraint-aware selection loop."""
    index = {c.pattern: j for j, c in enumerate(candidates)}
    mean = 0.5 * (score[0] + score[1])
    if kind == "predictive":
        order = sorted((c for c, ok in zip(candidates, eligible) if ok),
                       key=lambda c: (-float(mean[index[c.pattern]]),
                                      -c.n_molecules, c.pattern))
    else:
        order = sorted((c for c, ok in zip(candidates, eligible) if ok),
                       key=lambda c: (-c.n_molecules, c.pattern))
    rejected: list[dict[str, Any]] = []
    for c in order:
        parent_cert, why = resolver(c.pattern)
        if why is not None or parent_cert in used:
            rejected.append({"pattern": c.pattern,
                             "reason": why or "parent type already in grammar"})
            continue
        return c, parent_cert, rejected
    return None, None, rejected


def _ranking_metrics(candidates: Sequence[P.GD.Candidate], old: np.ndarray,
                     strict: np.ndarray, eligible_old: np.ndarray,
                     eligible_strict: np.ndarray) -> dict[str, Any]:
    n = len(candidates)
    if n < 2:
        return {"n_candidates": n}
    old_mean = 0.5 * (old[0] + old[1])
    strict_mean = 0.5 * (strict[0] + strict[1])
    # Spearman without scipy: ranks with average ties is overkill here because
    # gains are continuous; use stable ranks and Pearson correlation.
    def rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size, np.float64)
        r[order] = np.arange(v.size)
        return r

    ar, br = rank(old_mean), rank(strict_mean)
    if ar.std() > 0 and br.std() > 0:
        spearman = float(np.corrcoef(ar, br)[0, 1])
    else:
        spearman = 0.0
    do = np.sign(old_mean[:, None] - old_mean[None, :])
    ds = np.sign(strict_mean[:, None] - strict_mean[None, :])
    mask = np.triu(np.ones((n, n), bool), 1)
    concordant = int(np.sum((do == ds) & mask))
    discordant = int(np.sum((do != ds) & mask))
    pairs = n * (n - 1) // 2
    kendall = (concordant - discordant) / max(1, pairs)

    def top(score: np.ndarray, eligible: np.ndarray, k: int) -> list[str]:
        items = [c for c, ok in zip(candidates, eligible) if ok]
        idx = {c.pattern: j for j, c in enumerate(candidates)}
        items.sort(key=lambda c: (-float(score[idx[c.pattern]]), -c.n_molecules, c.pattern))
        return [c.pattern for c in items[:k]]

    old_ranked = top(old_mean, eligible_old, n)
    strict_ranked = top(strict_mean, eligible_strict, n)
    top5_old, top5_strict = old_ranked[:5], strict_ranked[:5]
    return {
        "n_candidates": n,
        "n_eligible_old": int(eligible_old.sum()),
        "n_eligible_strict": int(eligible_strict.sum()),
        "spearman": spearman,
        "kendall": float(kendall),
        "n_pairwise_inversions": discordant,
        "n_pairs": pairs,
        "top1_old": old_ranked[0] if old_ranked else None,
        "top1_strict": strict_ranked[0] if strict_ranked else None,
        "top1_changed": bool(old_ranked and strict_ranked and old_ranked[0] != strict_ranked[0]),
        "top5_overlap": len(set(top5_old) & set(top5_strict)),
        "old_ranked": old_ranked,
        "strict_ranked": strict_ranked,
    }


# ---------------------------------------------------------------------------
def replay() -> dict[str, Any]:
    t0 = time.time()
    cfg = P.load_config()
    parent_cfg = P.parent_config()
    lam = float(cfg["ridge_audit"]["lambda"])
    bundle = P.D.load_bundle()
    fit_rows = np.asarray(bundle.fit_rows, np.int64)
    dev_rows = np.asarray(bundle.eval_rows, np.int64)
    all_rows = np.concatenate([fit_rows, dev_rows])
    molecules = P.D.build_molecules(bundle.raw, all_rows)
    y_std = bundle.labels_std()
    trace = P.read_json(P.PARENT_DIR / "grammar" / "discovery_trace.json")["predictive"]
    parent_grammar = P.PD.load_grammar(P.PARENT_DIR / "grammar" / "predictive_grammar.json")

    disc = P.GD.GrammarDiscovery(molecules, fit_rows, y_std[fit_rows],
                                 bundle.graph_scalars[fit_rows], bundle.scaffold_by_row,
                                 log=lambda *_: None)
    partitions = {int(r): P.G.initial_partition(molecules[int(r)])
                  for r in fit_rows.tolist()}
    grammar = P.G.Grammar()
    rounds: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    replay_old_diff = 0.0
    pattern_mismatch = 0
    parent_mismatch = 0
    applications_mismatch = 0
    direct_formula_max = 0.0
    gap_identity_max = 0.0

    resolver_cache: dict[str, tuple[str, str | None]] = {}

    for step in range(len(trace)):
        rd = _build_round(disc, partitions)
        candidates = rd["shortlist"]
        old = P.GD.RoundScreener(rd["feats"], rd["y"], disc.fold, rd["C"]).score()
        # RoundScreener prepends an unpenalised intercept; strict_scores gets
        # exactly the same design matrix so only the candidate treatment differs.
        x_design = np.concatenate([np.ones((rd["feats"].shape[0], 1)), rd["feats"]],
                                  axis=1)
        direct, formula, detail = strict_scores(x_design, rd["y"], disc.fold,
                                                rd["C"], lam=lam)
        # direct solve and Schur formula must agree
        direct_formula_max = max(direct_formula_max,
                                 float(np.max(np.abs(direct - formula))))
        gap_identity_max = max(gap_identity_max,
                               float(np.max(np.abs(detail["gap"]
                                                    - detail["gap_identity"]))))
        old_mean = 0.5 * (old[0] + old[1])
        strict_mean = 0.5 * (direct[0] + direct[1])
        eligible_old = ((old[0] > P.GD.DECISION_IMPROVEMENT)
                        & (old[1] > P.GD.DECISION_IMPROVEMENT)
                        & (np.asarray([min(c.fold_molecules) for c in candidates])
                           >= P.GD.MIN_FOLD_MOLECULES)
                        & (np.asarray([min(c.fold_scaffolds) for c in candidates])
                           >= P.GD.MIN_FOLD_SCAFFOLDS))
        eligible_strict = ((direct[0] > P.GD.DECISION_IMPROVEMENT)
                           & (direct[1] > P.GD.DECISION_IMPROVEMENT)
                           & (np.asarray([min(c.fold_molecules) for c in candidates])
                              >= P.GD.MIN_FOLD_MOLECULES)
                           & (np.asarray([min(c.fold_scaffolds) for c in candidates])
                              >= P.GD.MIN_FOLD_SCAFFOLDS))

        def resolve(pattern: str) -> tuple[str, str | None]:
            if pattern not in resolver_cache:
                resolver_cache[pattern] = disc.resolve_parents(
                    rd["cand"], pattern, rd["groups"])
            return resolver_cache[pattern]

        used = grammar.parent_certs
        selected, parent_cert, rejected = _select(candidates, old, eligible_old,
                                                  used, resolve)
        if selected is None:
            raise RuntimeError(f"round {step}: replay found no selectable merge")
        sj = candidates.index(selected)
        stored = trace[step]
        stored_rule = stored["rule"]
        if selected.pattern != stored_rule["pattern"]:
            pattern_mismatch += 1
        if parent_cert != stored_rule["parent_cert"]:
            parent_mismatch += 1
        replay_old_diff = max(replay_old_diff,
                              abs(float(old[0, sj]) - float(stored["selected_improvement"][0])),
                              abs(float(old[1, sj]) - float(stored["selected_improvement"][1])))

        strict_selected, strict_parent, strict_rejected = _select(
            candidates, direct, eligible_strict, used, resolve)
        rank = _ranking_metrics(candidates, old, direct, eligible_old, eligible_strict)
        selected_rank_strict = None
        if rank.get("strict_ranked") and selected.pattern in rank["strict_ranked"]:
            selected_rank_strict = int(rank["strict_ranked"].index(selected.pattern) + 1)
        selected_rank_old = None
        if rank.get("old_ranked") and selected.pattern in rank["old_ranked"]:
            selected_rank_old = int(rank["old_ranked"].index(selected.pattern) + 1)

        sign_changed = bool((old[0, sj] > 0) != (direct[0, sj] > 0)
                            or (old[1, sj] > 0) != (direct[1, sj] > 0))
        n_sign_changes = int(np.sum(((old[0] > 0) != (direct[0] > 0))
                                    | ((old[1] > 0) != (direct[1] > 0))))
        delta = direct[:, sj] - old[:, sj]
        selected_mean_diff = float(0.5 * (delta[0] + delta[1]))
        material_reasons = []
        if abs(selected_mean_diff) > 1e-6:
            material_reasons.append("selected mean |diff| > 1e-6")
        if sign_changed:
            material_reasons.append("selected fold sign change")
        if strict_selected is not None and strict_selected.pattern != selected.pattern:
            material_reasons.append("constraint-aware selection changed")
        if rank["top1_changed"]:
            material_reasons.append("raw top-1 changed")
        if rank.get("spearman", 1.0) < 0.999:
            material_reasons.append("Spearman < 0.999")

        selected_den = {}
        for side in (0, 1):
            selected_den[str(side)] = {
                "resid_sq": float(detail["resid_sq"][side, sj]),
                "c_dot_resid": float(detail["c_dot_resid"][side, sj]),
                "gap": float(detail["gap"][side, sj]),
                "gap_identity": float(detail["gap_identity"][side, sj]),
                "den_old": float(detail["den_old"][side, sj]),
                "den_strict": float(detail["den_strict"][side, sj]),
                "den_ratio_strict_over_old": float(detail["den_strict"][side, sj]
                                                    / detail["den_old"][side, sj]),
                "beta_old": float(detail["beta_old"][side, sj]),
                "beta_strict": float(detail["beta_strict"][side, sj]),
                "beta_direct": float(detail["direct_coef"][side, -1, sj]),
                "improvement_old_formula": float(old[side, sj]),
                "improvement_direct": float(direct[side, sj]),
            }

        round_summary = {
            "round": step,
            "n_legal": len(rd["legal"]),
            "n_candidates": len(candidates),
            "n_eligible_old": int(eligible_old.sum()),
            "n_eligible_strict": int(eligible_strict.sum()),
            "selected": {
                "pattern": selected.pattern,
                "parent_cert": parent_cert,
                "parent_size": selected.parent_size,
                "n_molecules": selected.n_molecules,
                "n_scaffolds": selected.n_scaffolds,
                "fold_molecules": list(selected.fold_molecules),
                "fold_scaffolds": list(selected.fold_scaffolds),
                "improvement_old": [float(old[0, sj]), float(old[1, sj])],
                "mean_old": float(old_mean[sj]),
                "improvement_strict": [float(direct[0, sj]), float(direct[1, sj])],
                "mean_strict": float(strict_mean[sj]),
                "delta_fold": [float(delta[0]), float(delta[1])],
                "delta_mean": selected_mean_diff,
                "both_positive_old": bool(old[0, sj] > 0 and old[1, sj] > 0),
                "both_positive_strict": bool(direct[0, sj] > 0 and direct[1, sj] > 0),
                "fold_sign_changed": sign_changed,
                "rank_old": selected_rank_old,
                "rank_strict": selected_rank_strict,
            },
            "strict_selected": None if strict_selected is None else {
                "pattern": strict_selected.pattern,
                "parent_cert": strict_parent,
                "n_molecules": strict_selected.n_molecules,
                "improvement_old": [float(old[0, candidates.index(strict_selected)]),
                                    float(old[1, candidates.index(strict_selected)])],
                "improvement_strict": [float(direct[0, candidates.index(strict_selected)]),
                                       float(direct[1, candidates.index(strict_selected)])],
            },
            "selection_changed": bool(strict_selected is None
                                      or strict_selected.pattern != selected.pattern),
            "n_fold_sign_changes_all": n_sign_changes,
            "candidate_gain_diff": {
                "mean_abs": float(np.mean(np.abs(strict_mean - old_mean))),
                "max_abs": float(np.max(np.abs(strict_mean - old_mean))),
                "median_abs": float(np.median(np.abs(strict_mean - old_mean))),
            },
            "denominator_gap_all": {
                "abs_max": float(np.max(np.abs(detail["gap"]))),
                "abs_median": float(np.median(np.abs(detail["gap"]))),
                "positive_fraction": float(np.mean(detail["gap"] > 0)),
                "den_strict_over_old_min": float(np.min(detail["den_strict"]
                                                         / detail["den_old"])),
                "den_strict_over_old_max": float(np.max(detail["den_strict"]
                                                         / detail["den_old"])),
            },
            "selected_denominator": selected_den,
            "ranking": {k: v for k, v in rank.items()
                        if k not in ("old_ranked", "strict_ranked")},
            "ranking_top10": {"old": rank.get("old_ranked", [])[:10],
                              "strict": rank.get("strict_ranked", [])[:10]},
            "material": bool(material_reasons),
            "material_reasons": material_reasons,
        }
        rounds.append(round_summary)
        for j, c in enumerate(candidates):
            candidate_rows.append({
                "round": step, "pattern": c.pattern, "parent_size": c.parent_size,
                "n_molecules": c.n_molecules, "n_scaffolds": c.n_scaffolds,
                "old_fold0": float(old[0, j]), "old_fold1": float(old[1, j]),
                "old_mean": float(old_mean[j]),
                "strict_fold0": float(direct[0, j]), "strict_fold1": float(direct[1, j]),
                "strict_mean": float(strict_mean[j]),
                "diff_fold0": float(direct[0, j] - old[0, j]),
                "diff_fold1": float(direct[1, j] - old[1, j]),
                "diff_mean": float(strict_mean[j] - old_mean[j]),
                "eligible_old": bool(eligible_old[j]),
                "eligible_strict": bool(eligible_strict[j]),
                "gap_fold0": float(detail["gap"][0, j]),
                "gap_fold1": float(detail["gap"][1, j]),
                "den_old_fold0": float(detail["den_old"][0, j]),
                "den_strict_fold0": float(detail["den_strict"][0, j]),
                "selected_old": bool(j == sj),
                "selected_strict": bool(strict_selected is not None
                                        and candidates.index(strict_selected) == j),
            })

        # advance the *original* grammar exactly as PRCG-v0 did
        rule = P.G.Rule(index=len(grammar), left_cert=selected.left_cert,
                        right_cert=selected.right_cert, pattern=selected.pattern,
                        parent_cert=parent_cert, parent_size=selected.parent_size,
                        n_crossing=selected.n_crossing, origin="predictive")
        grammar.add(rule)
        n_applied = 0
        for r in fit_rows.tolist():
            new_part, n, _ = P.G.apply_rule(molecules[int(r)], partitions[int(r)], rule)
            partitions[int(r)] = new_part
            n_applied += n
        if n_applied != int(stored["applications_total"]):
            applications_mismatch += 1

    # ---- replay integrity --------------------------------------------------
    rules_match = all(grammar.rules[i].pattern == parent_grammar.rules[i].pattern
                      and grammar.rules[i].parent_cert == parent_grammar.rules[i].parent_cert
                      for i in range(min(len(grammar), len(parent_grammar))))
    selected_diffs = [abs(r["selected"]["mean_strict"] - r["selected"]["mean_old"])
                      for r in rounds]
    strict_positive = sum(1 for r in rounds if r["selected"]["both_positive_strict"])
    old_positive = sum(1 for r in rounds if r["selected"]["both_positive_old"])
    sign_changes = sum(1 for r in rounds if r["selected"]["fold_sign_changed"])
    selection_changes = sum(1 for r in rounds if r["selection_changed"])
    material = [r for r in rounds if r["material"]]
    worst = sorted(rounds, key=lambda r: -abs(r["selected"]["delta_mean"]))[:8]
    gap_all = np.concatenate([np.asarray([r["denominator_gap_all"]["abs_max"],
                                         r["denominator_gap_all"]["abs_median"]])
                              for r in rounds])

    payload = {
        "protocol_id": P.PROTOCOL_ID,
        "part": "A_ridge_audit",
        "generated_utc": P.C.now_utc(),
        "git_commit": P.git_commit(),
        "parent_protocol_id": parent_cfg["protocol_id"],
        "parent_config_sha256": P.PARENT_CONFIG_SHA256,
        "parent_trace": "grammar/discovery_trace.json",
        "method": {
            "lambda": lam,
            "candidate_penalty": float(cfg["ridge_audit"]["candidate_penalty"]),
            "base_penalty": cfg["ridge_audit"]["base_penalty"],
            "design": "X = [intercept, standardised current word counts + 8 graph scalars]; candidate column = raw greedy-application count; y = standardised Fit labels",
            "old_denominator": "resid_tr^T resid_tr + lambda (frozen RoundScreener)",
            "strict_denominator": "c_tr^T resid_tr + lambda (Frisch-Waugh for the augmented ridge)",
            "direct_reference": "solve (Z^T Z + lambda diag([0,1,...,1,1])) a = Z^T y for Z = [X, c] separately per candidate and fold",
        },
        "replay_integrity": {
            "n_rounds": len(rounds),
            "selected_pattern_mismatches": int(pattern_mismatch),
            "selected_parent_mismatches": int(parent_mismatch),
            "applications_mismatches": int(applications_mismatch),
            "max_abs_selected_gain_diff_vs_trace": float(replay_old_diff),
            "final_grammar_matches_parent": bool(rules_match),
            "n_final_rules": len(grammar),
            "max_abs_direct_minus_schur_formula": float(direct_formula_max),
            "max_abs_gap_identity_residual": float(gap_identity_max),
        },
        "summary": {
            "n_selected": len(rounds),
            "n_both_positive_old": int(old_positive),
            "n_both_positive_strict": int(strict_positive),
            "n_selected_fold_sign_changes": int(sign_changes),
            "n_constraint_aware_selection_changes": int(selection_changes),
            "max_abs_selected_mean_gain_diff": float(max(selected_diffs)) if selected_diffs else 0.0,
            "mean_abs_selected_mean_gain_diff": float(np.mean(selected_diffs)) if selected_diffs else 0.0,
            "n_material_rounds": len(material),
            "material_rounds": [r["round"] for r in material],
            "worst_rounds_by_selected_gain_diff": [
                {"round": r["round"], "pattern": r["selected"]["pattern"],
                 "n_molecules": r["selected"]["n_molecules"],
                 "old": r["selected"]["improvement_old"],
                 "strict": r["selected"]["improvement_strict"],
                 "delta_mean": r["selected"]["delta_mean"],
                 "selection_changed": r["selection_changed"],
                 "spearman": r["ranking"].get("spearman")}
                for r in worst],
            "denominator_gap": {
                "note": "gap = c_tr^T resid_tr - resid_tr^T resid_tr = lambda * ||S||_P^2 >= 0; strict denominator = old denominator + gap",
                "abs_max_over_rounds": float(gap_all.max()),
                "abs_median_over_rounds": float(np.median(gap_all)),
            },
        },
        "rounds": rounds,
    }
    P.write_json(P.ANALYSIS_DIR / "ridge_audit.json", payload)
    # candidate-level audit table (all original candidates, all rounds)
    header = list(candidate_rows[0].keys()) if candidate_rows else []
    P.write_csv(P.ANALYSIS_DIR / "ridge_candidates.csv", candidate_rows)
    print(json.dumps({
        "n_rounds": len(rounds),
        "replay_pattern_mismatch": pattern_mismatch,
        "replay_max_gain_diff": replay_old_diff,
        "both_positive_old": old_positive,
        "both_positive_strict": strict_positive,
        "selected_sign_changes": sign_changes,
        "selection_changes": selection_changes,
        "max_selected_mean_gain_diff": payload["summary"]["max_abs_selected_mean_gain_diff"],
        "material_rounds": payload["summary"]["material_rounds"],
        "direct_minus_formula_max": direct_formula_max,
        "gap_identity_max": gap_identity_max,
        "wall_seconds": time.time() - t0,
        "written": str(P.ANALYSIS_DIR / "ridge_audit.json"),
    }, indent=1))
    return payload


def main() -> int:
    replay()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())