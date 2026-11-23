"""PRCG-v0 grammar discovery: predictive-residual merges vs frequency merges.

Both grammars share the same molecule structures, the same candidate definition
(adjacent word pair, attachment pattern, parent <= 5 atoms) and the same merge
constraints.  They differ only in the selection score:

* predictive: the mean two-direction held-out MSE improvement of a ridge model
  ``Ridge(y ~ [X, x_m])`` over ``Ridge(y ~ X)``, where ``X`` is the current word
  count matrix plus the eight graph scalars and ``x_m`` the non-overlapping
  occurrence count of the merge pattern.
* frequency:  the number of distinct D0-Fit molecules in which the merge pattern
  occurs.  No label is touched.

Everything here is label-free except the predictive score, which sees only the
2856 D0-Fit labels.  D0 Dev labels are never read by this module.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import prcg_graph as G

# ---- frozen constants, all set before any result was seen -------------------
RIDGE_LAMBDA = 5.0
CANDIDATE_CAP = 400
MAX_RULES = 32
MIN_FOLD_MOLECULES = 8
MIN_FOLD_SCAFFOLDS = 4
FOLD_SEED = 20261201
MIN_FREQ_MOLECULES = 2
DECISION_IMPROVEMENT = 0.0


@dataclass
class Candidate:
    pattern: str
    left_cert: str
    right_cert: str
    parent_cert: str = ""
    parent_size: int = 0
    n_crossing: int = 0
    n_molecules: int = 0
    n_scaffolds: int = 0
    fold_molecules: tuple[int, int] = (0, 0)
    fold_scaffolds: tuple[int, int] = (0, 0)
    improvement: tuple[float, float] = (0.0, 0.0)

    @property
    def mean_improvement(self) -> float:
        return 0.5 * (self.improvement[0] + self.improvement[1])

    def as_dict(self) -> dict[str, Any]:
        return {"pattern": self.pattern, "left_cert": self.left_cert,
                "right_cert": self.right_cert, "parent_cert": self.parent_cert,
                "parent_size": self.parent_size, "n_crossing_bonds": self.n_crossing,
                "n_molecules": self.n_molecules, "n_scaffolds": self.n_scaffolds,
                "fold_molecules": list(self.fold_molecules),
                "fold_scaffolds": list(self.fold_scaffolds),
                "improvement": list(self.improvement),
                "mean_improvement": self.mean_improvement}


def fold_assignment(rows: Sequence[int], scaffold_by_row: Mapping[str, str],
                    seed: int = FOLD_SEED) -> np.ndarray:
    """Two scaffold-disjoint folds inside the D0 Fit rows (label-free).

    Sorted scaffold keys are permuted with a frozen seed and each whole group is
    assigned to the currently smaller fold, so no scaffold appears on both
    sides.
    """
    rows = np.asarray(rows, np.int64)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows.tolist()):
        groups[str(scaffold_by_row[str(r)])].append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(keys))
    fold = np.zeros(rows.size, np.int8)
    size = [0, 0]
    for i in order.tolist():
        side = 0 if size[0] <= size[1] else 1
        members = groups[keys[i]]
        fold[np.asarray(members, np.int64)] = side
        size[side] += len(members)
    return fold


def _column_standardise(mat: np.ndarray) -> np.ndarray:
    mean = mat.mean(axis=0)
    scale = mat.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (mat - mean) / scale


class RoundScreener:
    """All candidates of one grammar round, scored by ridge cross-fitting.

    ``X`` (current word counts + 8 graph scalars, standardised) is fitted once
    per fold direction; each candidate is then added as one extra column, with
    the extra coefficient shrunk by the *same* frozen L2 strength.  The
    residualisation of the candidate column against ``X`` is the conditioning
    step that removes the part explained by current word frequency alone.
    """

    def __init__(self, x_matrix: np.ndarray, y_std: np.ndarray, fold: np.ndarray,
                 candidate_matrix: np.ndarray) -> None:
        self.N, self.M = candidate_matrix.shape
        self.y = np.asarray(y_std, np.float64)
        self.fold = np.asarray(fold, np.int8)
        self.X = np.concatenate([np.ones((self.N, 1)), np.asarray(x_matrix, np.float64)], axis=1)
        self.C = np.asarray(candidate_matrix, np.float64)
        self.p = int(self.X.shape[1])
        self.lam = float(RIDGE_LAMBDA)
        pen = np.ones(self.p)
        pen[0] = 0.0                     # unpenalised intercept
        self._state: dict[int, dict[str, Any]] = {}
        for side in (0, 1):
            tr = self.fold != side
            te = self.fold == side
            Xtr, ytr = self.X[tr], self.y[tr]
            gram = Xtr.T @ Xtr + self.lam * np.diag(pen)
            a = np.linalg.solve(gram, Xtr.T @ ytr)
            self._state[side] = {"tr": tr, "te": te, "Xtr": Xtr, "Xte": self.X[te],
                                 "gram": gram, "base": self.X[te] @ a,
                                 "y": ytr, "y_te": self.y[te]}

    def score(self) -> np.ndarray:
        out = np.zeros((2, self.M), np.float64)
        for side in (0, 1):
            st = self._state[side]
            base_mse = float(np.mean((st["y_te"] - st["base"]) ** 2))
            S = np.linalg.solve(st["gram"], st["Xtr"].T @ self.C[st["tr"]])
            resid_tr = self.C[st["tr"]] - st["Xtr"] @ S
            resid_te = self.C[st["te"]] - st["Xte"] @ S
            beta = (resid_tr.T @ st["y"]) / (np.einsum("ij,ij->j", resid_tr, resid_tr)
                                             + self.lam)
            new_mse = np.mean((st["y_te"][:, None] - (st["base"][:, None]
                                                     + resid_te * beta[None, :])) ** 2, axis=0)
            out[side] = base_mse - new_mse
        return out


class GrammarDiscovery:
    def __init__(self, molecules: Mapping[int, G.Molecule], rows: Sequence[int],
                 labels_std: np.ndarray, graph_scalars: np.ndarray,
                 scaffold_by_row: Mapping[str, str],
                 log: Callable[[str], None] = print) -> None:
        self.mol = molecules
        self.rows = np.asarray(rows, np.int64)
        self.index = {int(r): i for i, r in enumerate(self.rows.tolist())}
        self.y = np.asarray(labels_std, np.float64)
        self.graph = np.asarray(graph_scalars, np.float64)
        self.scaffold = {int(r): str(scaffold_by_row[str(r)]) for r in self.rows.tolist()}
        self.fold = fold_assignment(self.rows, scaffold_by_row)
        self.log = log
        self.trace: list[dict[str, Any]] = []

    # -- enumeration --------------------------------------------------------
    def enumerate_round(self, partitions: Mapping[int, G.Partition]
                        ) -> tuple[dict[str, Candidate], list[dict[str, list[G.MergeOccurrence]]]]:
        """All legal merge patterns of the current partitions (one pass)."""
        first: dict[str, G.MergeOccurrence] = {}
        seen_rows: dict[str, set[int]] = defaultdict(set)
        groups_per_mol: list[dict[str, list[G.MergeOccurrence]]] = []
        for r in self.rows.tolist():
            mol = self.mol[int(r)]
            occ = G.merge_occurrences(mol, partitions[int(r)].words)
            groups: dict[str, list[G.MergeOccurrence]] = defaultdict(list)
            for o in occ:
                groups[o.pattern].append(o)
                seen_rows[o.pattern].add(int(r))
                first.setdefault(o.pattern, o)
            groups_per_mol.append(dict(groups))
        cand: dict[str, Candidate] = {}
        for key, o in first.items():
            rows_with = seen_rows[key]
            sides = [int(self.fold[self.index[r]]) for r in rows_with]
            cand[key] = Candidate(
                pattern=key, left_cert=o.left_cert, right_cert=o.right_cert,
                parent_size=len(o.left_support) + len(o.right_support),
                n_crossing=len(o.crossing), n_molecules=len(rows_with),
                n_scaffolds=len({self.scaffold[r] for r in rows_with}),
                fold_molecules=(sum(1 for s in sides if s == 0),
                                sum(1 for s in sides if s == 1)),
                fold_scaffolds=(
                    len({self.scaffold[r] for r in rows_with if self.fold[self.index[r]] == 0}),
                    len({self.scaffold[r] for r in rows_with if self.fold[self.index[r]] == 1})))
        return cand, groups_per_mol

    def resolve_parents(self, cand: Mapping[str, Candidate], key: str,
                        groups_per_mol: Sequence[Mapping[str, list[G.MergeOccurrence]]]
                        ) -> tuple[str, str | None]:
        """Parent certificate of one pattern, verified over every occurrence."""
        cert: str | None = None
        for i, r in enumerate(self.rows.tolist()):
            items = groups_per_mol[i].get(key)
            if not items:
                continue
            mol = self.mol[int(r)]
            for o in items:
                pc = G.parent_signature(mol, o).cert
                if cert is None:
                    cert = pc
                elif pc != cert:
                    return "", "inconsistent parent certificate across occurrences"
        return (cert or ""), None

    # -- discovery loop -----------------------------------------------------
    def run(self, kind: str, max_rules: int = MAX_RULES, n_rules: int | None = None
            ) -> tuple[G.Grammar, list[dict[str, Any]]]:
        assert kind in ("predictive", "frequency")
        grammar = G.Grammar()
        partitions = {int(r): G.initial_partition(self.mol[int(r)]) for r in self.rows.tolist()}
        target = int(n_rules if n_rules is not None else max_rules)
        steps: list[dict[str, Any]] = []
        t0 = time.time()
        for step in range(target):
            t_round = time.time()
            cand, groups = self.enumerate_round(partitions)
            used = grammar.parent_certs
            legal = {k: c for k, c in cand.items()
                     if c.parent_size <= G.MAX_WORD_ATOMS and c.n_molecules >= MIN_FREQ_MOLECULES}
            if not legal:
                self.log(f"[{kind}] round {step}: no legal candidate -> stop early")
                break
            ranked = sorted(legal.values(), key=lambda c: (-c.n_molecules, c.pattern))
            shortlist = ranked if kind == "frequency" else ranked[:CANDIDATE_CAP]

            # current word-count matrix (+ the eight frozen graph scalars)
            vocab: dict[str, int] = {}
            for r in self.rows.tolist():
                for w in partitions[int(r)].words:
                    vocab.setdefault(w.cert, len(vocab))
            X = np.zeros((self.rows.size, len(vocab)), np.float64)
            for i, r in enumerate(self.rows.tolist()):
                for w in partitions[int(r)].words:
                    j = vocab.get(w.cert)
                    if j is not None:
                        X[i, j] += 1.0
            feats = _column_standardise(np.concatenate([X, self.graph], axis=1))
            y = self.y
            y = (y - y.mean()) / (y.std() if y.std() > 1e-12 else 1.0)

            if kind == "predictive":
                C = np.zeros((self.rows.size, len(shortlist)), np.float64)
                for i, g in enumerate(groups):
                    for j, c in enumerate(shortlist):
                        items = g.get(c.pattern)
                        if items:
                            C[i, j] = len(G.greedy_select(items))
                improvement = RoundScreener(feats, y, self.fold, C).score()
                for j, c in enumerate(shortlist):
                    c.improvement = (float(improvement[0, j]), float(improvement[1, j]))
                eligible = [c for c in shortlist
                            if c.improvement[0] > DECISION_IMPROVEMENT
                            and c.improvement[1] > DECISION_IMPROVEMENT
                            and min(c.fold_molecules) >= MIN_FOLD_MOLECULES
                            and min(c.fold_scaffolds) >= MIN_FOLD_SCAFFOLDS]
            else:
                eligible = list(shortlist)

            selected: Candidate | None = None
            rejected: list[dict[str, Any]] = []
            for c in sorted(eligible, key=lambda c: (
                    -c.mean_improvement if kind == "predictive" else 0.0,
                    -c.n_molecules, c.pattern)):
                parent_cert, why = self.resolve_parents(cand, c.pattern, groups)
                if why is not None or parent_cert in used:
                    rejected.append({"pattern": c.pattern, "reason": why or "parent type already in grammar"})
                    continue
                c.parent_cert = parent_cert
                selected = c
                break
            if selected is None:
                self.log(f"[{kind}] round {step}: no selectable merge -> stop "
                         f"({len(eligible)} eligible, {len(rejected)} rejected)")
                break
            best = selected
            rule = G.Rule(index=len(grammar), left_cert=best.left_cert,
                          right_cert=best.right_cert, pattern=best.pattern,
                          parent_cert=best.parent_cert, parent_size=best.parent_size,
                          n_crossing=best.n_crossing, origin=kind)
            grammar.add(rule)
            n_applied = 0
            mismatches = 0
            for r in self.rows.tolist():
                new_part, n, flags = G.apply_rule(self.mol[int(r)], partitions[int(r)], rule)
                partitions[int(r)] = new_part
                n_applied += n
                if not flags["parent_cert_ok"]:
                    mismatches += 1
            record = {
                "kind": kind, "round": step, "rule": rule.as_dict(),
                "n_candidates_legal": len(legal), "n_shortlist": len(shortlist),
                "n_eligible": len(eligible), "rejected_before_selection": rejected,
                "selected_n_molecules": best.n_molecules,
                "selected_n_scaffolds": best.n_scaffolds,
                "selected_fold_molecules": list(best.fold_molecules),
                "selected_fold_scaffolds": list(best.fold_scaffolds),
                "selected_improvement": list(best.improvement),
                "selected_mean_improvement": best.mean_improvement,
                "applications_total": int(n_applied),
                "parent_cert_mismatch_molecules": int(mismatches),
                "n_word_types": len(vocab), "round_seconds": time.time() - t_round,
                "improvement_quantiles": ({
                    "min": float(np.min(improvement)), "median": float(np.median(improvement)),
                    "max": float(np.max(improvement)),
                    "fraction_positive_both": float(np.mean((improvement[0] > 0)
                                                            & (improvement[1] > 0))),
                } if kind == "predictive" else None),
            }
            steps.append(record)
            self.trace.append(record)
            self.log(f"[{kind}] round {step}: size={rule.parent_size} mols={best.n_molecules} "
                     f"impl={best.improvement} applied={n_applied} types={len(vocab)} "
                     f"({record['round_seconds']:.1f}s)")
        self.log(f"[{kind}] done: {len(grammar)} rules in {time.time() - t0:.1f}s")
        return grammar, steps


# ---------------------------------------------------------------------------
# grammar-level statistics
# ---------------------------------------------------------------------------
def apply_all(molecules: Mapping[int, G.Molecule], rows: Sequence[int], grammar: G.Grammar
              ) -> dict[int, G.Partition]:
    return {int(r): grammar.apply(molecules[int(r)])[0]
            for r in np.asarray(rows, np.int64).tolist()}


def grammar_stats(molecules: Mapping[int, G.Molecule], rows: Sequence[int], grammar: G.Grammar,
                  scaffold_by_row: Mapping[str, str]) -> dict[str, Any]:
    rows = [int(r) for r in np.asarray(rows, np.int64).tolist()]
    sizes: dict[int, int] = defaultdict(int)
    type_rows: dict[str, set[int]] = defaultdict(set)
    type_scaf: dict[str, set[str]] = defaultdict(set)
    type_size: dict[str, int] = {}
    applied: list[int] = []
    n_words = 0
    multi_port = 0
    for r in rows:
        mol = molecules[r]
        part, log = grammar.apply(mol, record_log=True)
        applied.append(sum(log["n_applied"]))
        for w in part.words:
            sizes[w.size] += 1
            type_rows[w.cert].add(r)
            type_scaf[w.cert].add(str(scaffold_by_row[str(r)]))
            type_size[w.cert] = w.size
            n_words += 1
            ports = [a for a in w.support
                     if any(y not in w.support for y, _ in mol.adj[a])]
            if len(ports) > 1:
                multi_port += 1
    counts = np.asarray([len(v) for v in type_rows.values()], np.int64)
    scaf = np.asarray([len(type_scaf[k]) for k in type_rows], np.int64)
    by_size: dict[int, int] = defaultdict(int)
    for cert, n in type_size.items():
        by_size[n] += 1
    return {
        "n_molecules": len(rows), "n_words": int(n_words),
        "mean_words_per_molecule": float(n_words / max(1, len(rows))),
        "size_histogram": {int(k): int(v) for k, v in sorted(sizes.items())},
        "size_fractions": {int(k): float(v / max(1, n_words)) for k, v in sorted(sizes.items())},
        "n_word_types": len(type_rows),
        "types_by_size": {int(k): int(v) for k, v in sorted(by_size.items())},
        "multi_port_word_fraction": float(multi_port / max(1, n_words)),
        "mean_applications_per_molecule": float(np.mean(applied)),
        "type_molecule_support_mean": float(counts.mean()) if counts.size else 0.0,
        "type_molecule_support_median": float(np.median(counts)) if counts.size else 0.0,
        "type_cross_scaffold_reuse_fraction": float(np.mean(scaf >= 2)) if scaf.size else 0.0,
        "singleton_type_fraction": float(np.mean(counts == 1)) if counts.size else 0.0,
        "type_support": {k: int(len(v)) for k, v in type_rows.items()},
        "type_scaffold_support": {k: int(len(v)) for k, v in type_scaf.items()},
        "type_size": {k: int(v) for k, v in type_size.items()},
    }