"""Train-only joint support set Omega, bipartite support graph, marginal projector.

Frozen definitions (see ``PROTOCOL.md`` section 3):

* ``p = path_cat``, ``e = path_end63`` — the production S2 integer codes.
* ``Omega`` = the set of distinct ``(p, e)`` pairs that actually occur in the
  **Train** split.  Nothing is built from Valid/Test and no unobserved
  Cartesian product is materialised.
* Bipartite support graph: left = observed ``p`` types, right = observed ``e``
  types, one edge per Omega element.
* Association design matrix ``X``: one row per Omega element and exactly two
  unit entries per row — the column of that element's ``p`` and the column of
  that element's ``e``.  Columns are ``[P | E]``, so ``X`` is the bipartite
  incidence matrix and ``span(X)`` (as vectors over Omega elements, type-equal
  weights) is exactly the set of additive functions ``a_p + b_e``.
* ``X = U S V^T`` (float64), ``P = U U^T`` projects onto that additive
  subspace; ``I - P`` is the non-additive joint-computation subspace.

Everything here is deterministic, label-free and Train-only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import sj_common as C

SVD_TOL = 1e-10

#: ``pair_mode`` codes for the dense ``code -> J`` tables
MODE_OMEGA = 0
MODE_MARGINAL_LS = 1
MODE_TYPE_MEAN = 2


# ---------------------------------------------------------------------------
# production S2 records of one split cache
# ---------------------------------------------------------------------------


def split_records(cache: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Flatten the S2 records of a split cache to per-record arrays.

    ``mol_of_path`` is the molecule that owns the record's root atom; record
    order is the production cache order and has no effect on anything below.
    """
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    n_atoms = int(atom_ptr[-1])
    if path_ptr.shape != (n_atoms + 1,):
        raise ValueError(f"path_ptr shape {path_ptr.shape} != {(n_atoms + 1,)}")
    counts = np.diff(path_ptr)
    if int(counts.sum()) != int(np.asarray(cache["path_cat"]).size):
        raise ValueError("path record count does not match path_ptr")
    root_atom = np.repeat(np.arange(n_atoms, dtype=np.int64), counts)
    mol_of_atom = np.repeat(np.arange(atom_ptr.size - 1, dtype=np.int64), np.diff(atom_ptr))
    mol_of_path = mol_of_atom[root_atom] if root_atom.size else np.zeros(0, dtype=np.int64)
    return {
        "path_cat": np.asarray(cache["path_cat"], dtype=np.int64),
        "path_end63": np.asarray(cache["path_end63"], dtype=np.int64),
        "root_atom": root_atom,
        "mol_of_path": mol_of_path,
        "path_counts": counts,
        "n_molecules": np.int64(atom_ptr.size - 1),
        "n_atoms": np.int64(n_atoms),
    }


def pair_code(p: np.ndarray, e: np.ndarray, n_nbr_codes: int) -> np.ndarray:
    return np.asarray(p, dtype=np.int64) * int(n_nbr_codes) + np.asarray(e, dtype=np.int64)


# ---------------------------------------------------------------------------
# support object
# ---------------------------------------------------------------------------


@dataclass
class Support:
    task_key: str
    dataset: str
    n_path_codes: int
    n_nbr_codes: int
    n_bond_types: int
    # Omega (sorted by (p, e))
    omega_p: np.ndarray
    omega_e: np.ndarray
    omega_code: np.ndarray
    omega_count: np.ndarray
    omega_docfreq: np.ndarray
    # type vocabularies
    p_types: np.ndarray
    e_types: np.ndarray
    # design / projector (float64)
    X: np.ndarray
    U: np.ndarray
    s: np.ndarray
    P: np.ndarray
    pinvX: np.ndarray
    rank: int
    n_components: int
    comp_p: np.ndarray
    comp_e: np.ndarray
    # dense code -> index tables (both are O(n_path_codes * n_nbr_codes))
    index_table: np.ndarray  # int32, -1 = not in Omega
    pair_mode: np.ndarray  # int8, 0 = in Omega, 1 = marginal LS fallback, 2 = mean fallback
    p_col: np.ndarray  # int32, -1 = p type never seen in Train
    e_col: np.ndarray  # int32, -1 = e type never seen in Train
    # geometry
    n_rectangles: int
    coverage: dict[str, Any] = field(default_factory=dict)
    molecule_coverage: dict[str, np.ndarray] = field(default_factory=dict)
    legality: dict[str, Any] = field(default_factory=dict)
    build_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_omega(self) -> int:
        return int(self.omega_code.size)

    @property
    def n_p(self) -> int:
        return int(self.p_types.size)

    @property
    def n_e(self) -> int:
        return int(self.e_types.size)

    @property
    def joint_dim(self) -> int:
        return int(self.n_omega - self.rank)

    @property
    def n_columns(self) -> int:
        return int(self.n_p + self.n_e)

    def decode(self, p: int, e: int) -> dict[str, Any]:
        """Decode one ``(p, e)`` pair into the frozen readable fields."""
        nb = int(self.n_bond_types)
        return {
            "p": int(p),
            "e": int(e),
            "middle_atom_type": int(p) // (nb * nb),
            "bond_1": (int(p) // nb) % nb + 1,   # root -> middle bond type
            "bond_2": int(p) % nb + 1,           # middle -> end bond type (from p)
            "end_atom_type": int(e) // nb,
            "bond_2_from_e": int(e) % nb + 1,
            "bond_2_repeated_field_consistent": bool(int(p) % nb == int(e) % nb),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "task": self.task_key,
            "dataset": self.dataset,
            "n_path_codes": int(self.n_path_codes),
            "n_nbr_codes": int(self.n_nbr_codes),
            "n_bond_types": int(self.n_bond_types),
            "n_omega": int(self.n_omega),
            "n_p_types": int(self.n_p),
            "n_e_types": int(self.n_e),
            "n_design_columns": int(self.n_columns),
            "n_connected_components": int(self.n_components),
            "rank_X": int(self.rank),
            "rank_identity_P_plus_E_minus_components": int(self.n_columns - self.n_components),
            "joint_subspace_dim": int(self.joint_dim),
            "n_rectangles": int(self.n_rectangles),
            "occurrence_total": int(self.omega_count.sum()),
            "docfreq_total": int(self.omega_docfreq.sum()),
            "n_omega_docfreq_ge_3": int((self.omega_docfreq >= 3).sum()),
            "n_omega_docfreq_ge_20": int((self.omega_docfreq >= C.CONTRAST_MIN_PAIR_DOCFREQ).sum()),
            "svd_singular_min": float(self.s.min()) if self.s.size else None,
            "svd_singular_max": float(self.s.max()) if self.s.size else None,
            "coverage": self.coverage,
            "legality": self.legality,
            "build": self.build_meta,
        }


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def _components(n_p: int, n_e: int, p_index: np.ndarray, e_index: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Connected components of the bipartite support graph (union-find)."""
    n = n_p + n_e
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in zip(p_index.tolist(), e_index.tolist()):
        ra, rb = find(a), find(n_p + b)
        if ra != rb:
            parent[ra] = rb
    labels = np.asarray([find(i) for i in range(n)], dtype=np.int64)
    uniq = np.unique(labels)
    canonical = {int(u): rank for rank, u in enumerate(sorted(uniq.tolist(), key=lambda u: int(np.flatnonzero(labels == u)[0])))}
    canon = np.asarray([canonical[int(v)] for v in labels.tolist()], dtype=np.int32)
    return int(uniq.size), canon[:n_p], canon[n_p:]


def _rectangle_count(n_p: int, n_e: int, p_index: np.ndarray, e_index: np.ndarray) -> int:
    """Number of distinct 2x2 rectangles ``(p1,e1),(p1,e2),(p2,e1),(p2,e2)`` in Omega."""
    B = np.zeros((n_e, n_p), dtype=np.int64)
    B[e_index, p_index] = 1
    common = B @ B.T
    c = common[np.triu_indices(n_e, k=1)]
    return int((c * (c - 1) // 2).sum())


def _coverage(rec: Mapping[str, np.ndarray], index_table: np.ndarray, n_nbr_codes: int) -> tuple[dict[str, Any], np.ndarray]:
    code = pair_code(rec["path_cat"], rec["path_end63"], n_nbr_codes)
    seen = index_table[code] >= 0
    n_mol = int(rec["n_molecules"])
    mol = rec["mol_of_path"]
    n_paths_mol = np.bincount(mol, minlength=n_mol).astype(np.int64) if mol.size else np.zeros(n_mol, np.int64)
    n_seen_mol = np.bincount(mol[seen], minlength=n_mol).astype(np.int64) if seen.any() else np.zeros(n_mol, np.int64)
    frac = np.where(n_paths_mol > 0, n_seen_mol / np.maximum(n_paths_mol, 1), np.nan).astype(np.float64)
    stats = {
        "n_occurrences": int(code.size),
        "occurrence_coverage": float(seen.mean()) if seen.size else 1.0,
        "n_occurrences_unseen": int((~seen).sum()),
        "molecule_fraction_with_any_unseen_pair": float((n_seen_mol < n_paths_mol).mean()) if n_mol else 0.0,
        "molecule_fraction_with_any_seen_pair": float((n_seen_mol > 0).mean()) if n_mol else 0.0,
        "molecule_fraction_without_s2": float((n_paths_mol == 0).mean()) if n_mol else 0.0,
        "n_molecules_without_s2": int((n_paths_mol == 0).sum()),
        "paths_per_molecule_mean": float(n_paths_mol.mean()) if n_mol else 0.0,
    }
    return stats, frac


def build(
    task_key: str,
    train_cache: Mapping[str, Any],
    *,
    valid_cache: Mapping[str, Any] | None = None,
    test_cache: Mapping[str, Any] | None = None,
    spec: Any = None,
) -> Support:
    """Build the Train-only support object for one task."""
    if task_key not in C.TASKS:
        raise KeyError(task_key)
    if spec is None:
        import mb2_data as MD  # frozen loader

        spec = MD.frozen_spec()
    n_path_codes = int(spec.n_path_codes)
    n_nbr_codes = int(spec.n_nbr_codes)
    n_bond = int(spec.n_bond_types)

    tr = split_records(train_cache)
    pc, pe, mol = tr["path_cat"], tr["path_end63"], tr["mol_of_path"]
    code = pair_code(pc, pe, n_nbr_codes)

    uniq, counts = np.unique(code, return_counts=True)
    omega_p = uniq // n_nbr_codes
    omega_e = uniq % n_nbr_codes

    # independent support: number of distinct Train molecules containing the pair
    n_train_mol = int(tr["n_molecules"])
    pair_view = np.unique(code * (n_train_mol + 1) + mol)  # distinct (pair, molecule)
    pair_view = pair_view // (n_train_mol + 1)  # back to the pair code
    rows_unique, df = np.unique(pair_view, return_counts=True)
    if not np.array_equal(rows_unique, uniq):
        raise RuntimeError("document-frequency key set differs from Omega")
    docfreq = df.astype(np.int64)
    if not np.all(docfreq <= n_train_mol):
        raise RuntimeError("document frequency exceeds the number of Train molecules")

    p_types = np.unique(omega_p)
    e_types = np.unique(omega_e)
    p_index = np.searchsorted(p_types, omega_p)
    e_index = np.searchsorted(e_types, omega_e)

    n_comp, comp_p, comp_e = _components(p_types.size, e_types.size, p_index, e_index)
    n_rect = _rectangle_count(p_types.size, e_types.size, p_index, e_index)

    # design matrix X: exactly two unit entries per row (its p column and its e column)
    n_rows = int(uniq.size)
    X = np.zeros((n_rows, int(p_types.size + e_types.size)), dtype=np.float64)
    X[np.arange(n_rows), p_index] = 1.0
    X[np.arange(n_rows), int(p_types.size) + e_index] = 1.0

    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    keep = s > (SVD_TOL * float(s.max()) if s.size else 0.0)
    U = np.ascontiguousarray(U[:, keep])
    s = np.ascontiguousarray(s[keep])
    Vt = np.ascontiguousarray(Vt[keep, :])
    rank = int(U.shape[1])
    P = U @ U.T
    pinvX = (Vt.T / s) @ U.T  # (P+E, K)

    # dense code -> Omega index / fallback mode tables
    index_table = np.full(n_path_codes * n_nbr_codes, -1, dtype=np.int32)
    index_table[uniq] = np.arange(n_rows, dtype=np.int32)
    p_col = np.full(n_path_codes, -1, dtype=np.int32)
    p_col[p_types] = np.arange(p_types.size, dtype=np.int32)
    e_col = np.full(n_nbr_codes, -1, dtype=np.int32)
    e_col[e_types] = np.arange(e_types.size, dtype=np.int32)

    all_p = np.repeat(np.arange(n_path_codes, dtype=np.int64), n_nbr_codes)
    all_e = np.tile(np.arange(n_nbr_codes, dtype=np.int64), n_path_codes)
    known = (p_col[all_p] >= 0) & (e_col[all_e] >= 0)
    same_comp = np.zeros(all_p.size, dtype=bool)
    same_comp[known] = comp_p[p_col[all_p[known]]] == comp_e[e_col[all_e[known]]]
    pair_mode = np.full(all_p.size, 2, dtype=np.int8)
    pair_mode[known & same_comp] = 1
    pair_mode[index_table >= 0] = 0

    # legality: bond_type_2 is encoded in both p and e
    bt2_p = omega_p % n_bond
    bt2_e = omega_e % n_bond
    p_by_bt2 = np.bincount(np.arange(n_path_codes) % n_bond, minlength=n_bond)
    e_by_bt2 = np.bincount(np.arange(n_nbr_codes) % n_bond, minlength=n_bond)
    possible_legal = int(np.sum(p_by_bt2 * e_by_bt2))
    legality = {
        "rule": "bond_type_2 appears in both p and e; legal iff (p % n_bond_types) == (e % n_bond_types)",
        "repeated_key_field": "bond_type_2 (the middle->end bond) is encoded in path_cat AND path_end63",
        "n_omega": int(n_rows),
        "n_omega_legal": int((bt2_p == bt2_e).sum()),
        "all_omega_legal": bool(np.all(bt2_p == bt2_e)),
        "n_possible_legal_type_pairs": possible_legal,
        "n_possible_type_pairs": int(n_path_codes * n_nbr_codes),
        "observed_fraction_of_legal_type_pairs": float(n_rows / max(possible_legal, 1)),
        "fallback_mode_counts": {
            "omega": int((pair_mode == 0).sum()),
            "marginal_ls": int((pair_mode == 1).sum()),
            "type_equal_mean": int((pair_mode == 2).sum()),
        },
    }

    coverage: dict[str, Any] = {}
    molecule_coverage: dict[str, np.ndarray] = {}
    for split, cache in (("train", train_cache), ("valid", valid_cache), ("test", test_cache)):
        if cache is None:
            continue
        rec = split_records(cache)
        stats, frac = _coverage(rec, index_table, n_nbr_codes)
        codes_split = pair_code(rec["path_cat"], rec["path_end63"], n_nbr_codes)
        idx = index_table[codes_split]
        per_pair = np.bincount(idx[idx >= 0], minlength=n_rows).astype(np.int64)
        stats["n_molecules"] = int(rec["n_molecules"])
        stats["n_omega_rows_with_any_occurrence"] = int((per_pair > 0).sum())
        stats["role"] = "definition split (Train-only support)" if split == "train" else "descriptive coverage only"
        coverage[split] = stats
        molecule_coverage[split] = frac.astype(np.float32)

    build_meta = {
        "source": "Train split S2 records only",
        "omega_order": "ascending (path_cat, path_end63)",
        "svd": f"numpy.linalg.svd(float64), keep s > {SVD_TOL} * s_max",
        "n_records_train": int(code.size),
        "n_train_molecules": int(tr["n_molecules"]),
        "n_train_atoms": int(tr["n_atoms"]),
        "n_train_molecules_without_s2": int(
            (np.bincount(tr["mol_of_path"], minlength=int(tr["n_molecules"])) == 0).sum()
        ),
    }

    return Support(
        task_key=task_key,
        dataset=str(C.TASKS[task_key]["dataset"]),
        n_path_codes=n_path_codes,
        n_nbr_codes=n_nbr_codes,
        n_bond_types=n_bond,
        omega_p=omega_p,
        omega_e=omega_e,
        omega_code=uniq,
        omega_count=counts.astype(np.int64),
        omega_docfreq=docfreq,
        p_types=p_types,
        e_types=e_types,
        X=X,
        U=U,
        s=s,
        P=P,
        pinvX=pinvX,
        rank=rank,
        n_components=n_comp,
        comp_p=comp_p,
        comp_e=comp_e,
        index_table=index_table,
        pair_mode=pair_mode,
        p_col=p_col,
        e_col=e_col,
        n_rectangles=n_rect,
        coverage=coverage,
        molecule_coverage=molecule_coverage,
        legality=legality,
        build_meta=build_meta,
    )


# ---------------------------------------------------------------------------
# persistence / verification
# ---------------------------------------------------------------------------


def to_artifact(sup: Support) -> dict[str, np.ndarray]:
    art = {
        "omega_p": sup.omega_p,
        "omega_e": sup.omega_e,
        "omega_code": sup.omega_code,
        "omega_count": sup.omega_count,
        "omega_docfreq": sup.omega_docfreq,
        "p_types": sup.p_types,
        "e_types": sup.e_types,
        "X": sup.X,
        "U": sup.U,
        "s": sup.s,
        "P": sup.P,
        "pinvX": sup.pinvX,
        "comp_p": sup.comp_p,
        "comp_e": sup.comp_e,
        "index_table": sup.index_table,
        "pair_mode": sup.pair_mode,
        "p_col": sup.p_col,
        "e_col": sup.e_col,
    }
    for split, arr in sup.molecule_coverage.items():
        art[f"molecule_coverage_{split}"] = arr
    return art


def artifact_hash(sup: Support) -> str:
    digest = hashlib.sha256()
    for key, arr in sorted(to_artifact(sup).items()):
        a = np.ascontiguousarray(np.asarray(arr))
        digest.update(key.encode("utf-8"))
        digest.update(str(a.shape).encode("utf-8"))
        digest.update(str(a.dtype).encode("utf-8"))
        digest.update(a.tobytes())
    return digest.hexdigest()


def save(sup: Support, path: Any) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **to_artifact(sup))
    return C.sha256_file(p)


def load_artifact(path: Any) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def check_against_artifact(sup: Support, path: Any) -> dict[str, Any]:
    """Recompute-vs-recorded identity check (fails loudly on any drift)."""
    art = load_artifact(path)
    mine = to_artifact(sup)
    checks: dict[str, Any] = {}
    ok = True
    for key in (
        "omega_p", "omega_e", "omega_code", "omega_count", "omega_docfreq", "p_types", "e_types",
        "index_table", "pair_mode", "p_col", "e_col", "comp_p", "comp_e",
    ):
        same = bool(np.array_equal(np.asarray(art[key]), np.asarray(mine[key])))
        checks[key] = same
        ok &= same
    for key in ("X", "U", "P", "pinvX", "s"):
        a = np.asarray(art[key], dtype=np.float64)
        b = np.asarray(mine[key], dtype=np.float64)
        diff = float(np.max(np.abs(a - b))) if a.size else 0.0
        checks[f"{key}_max_abs_diff"] = diff
        ok &= diff <= 1e-12
    return {"all_identical": bool(ok), "checks": checks}


# ---------------------------------------------------------------------------
# task-level wiring (data views)
# ---------------------------------------------------------------------------


def task_caches(task_key: str) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], Any]:
    """(view, train_cache, valid_cache, test_cache, spec) for one task."""
    if task_key == "hiv":
        ds = str(C.TASKS["hiv"]["dataset"])
        view = C.PD.molecule_view(ds)
        test_cache, _y_test, _rows = C.PD.molecule_test(ds)
        return view, view.train_cache, view.valid_cache, test_cache, view.spec
    if task_key == "lipo":
        rep = int(C.TASKS["lipo"]["repeat"])
        view = C.DD.build_view(rep)
        test_cache, _y_test, _smiles, _rows = C.DD.load_test(rep)
        return view, view.train_cache, view.valid_cache, test_cache, view.spec
    raise KeyError(task_key)


def build_task_support(task_key: str, *, verify_artifact: bool = True) -> Support:
    _view, train_cache, valid_cache, test_cache, spec = task_caches(task_key)
    sup = build(task_key, train_cache, valid_cache=valid_cache, test_cache=test_cache, spec=spec)
    if verify_artifact:
        path = C.ANALYSIS / "support" / f"{task_key}_support.npz"
        if path.exists():
            res = check_against_artifact(sup, path)
            if not res["all_identical"]:
                raise RuntimeError(f"{task_key}: recomputed support differs from the recorded artifact: {res}")
    return sup


__all__ = [n for n in dir() if not n.startswith("_")]


# ---------------------------------------------------------------------------
# CLI: (re)build the Train-only support artefacts and the support graph
# ---------------------------------------------------------------------------

FROZEN_DEFINITIONS = {
    "grouping": "p = path_cat, e = path_end63 (production S2 integer codes, no new fields)",
    "omega": "the set of distinct (p, e) pairs that actually occur in the Train split",
    "design": (
        "X has one row per Omega element and exactly two unit entries per row: the column of its p "
        "and the column of its e (bipartite incidence matrix, |P|+|E| columns)"
    ),
    "svd": "float64 reduced SVD of X; U = left singular vectors with s > 1e-10 * s_max",
    "projection": "P = U U^T; J_add = P J; J_int = J - P J",
    "weights": "type-equal: every Omega element has weight 1 (no occurrence / molecule weighting)",
    "legality": "bond_type_2 is encoded in both p and e; only p % n_bond == e % n_bond is legal",
    "unseen_policy": "no new learnable parameter; the fallback is a deterministic function of J_table",
    "fallback": {
        "same_component": (
            "least-squares marginal fit Jhat = argmin ||X Theta - J_table||_F; J(p,e) = Theta[p] + Theta[e]"
        ),
        "otherwise": "type-equal mean over the Omega rows of J_table",
    },
}

ARTIFACT_RULE = (
    "each task recomputes Omega + X + SVD from the frozen caches and must match the recorded .npz "
    "bit-for-bit (arrays) / 1e-12 (float64 matrices)"
)


def write_artifacts(task_keys: Sequence[str], *, out_dir=None, graph_path=None) -> dict[str, Any]:
    out = Path(out_dir) if out_dir else C.ANALYSIS / "support"
    out.mkdir(parents=True, exist_ok=True)
    graph = Path(graph_path) if graph_path else C.ANALYSIS / "support_graph.json"
    payload: dict[str, Any] = {
        "schema": "lcrc-support-joint-function-support-graph-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "artifact_rule": ARTIFACT_RULE,
        "frozen_definitions": FROZEN_DEFINITIONS,
        "tasks": {},
    }
    for task_key in task_keys:
        sup = build_task_support(task_key, verify_artifact=False)
        path = out / f"{task_key}_support.npz"
        save(sup, path)
        entry = dict(sup.summary())
        entry["artifact"] = {
            "path": C.track_rel(path),
            "sha256": C.sha256_file(path),
            "content_hash": artifact_hash(sup),
        }
        payload["tasks"][task_key] = entry
        C.log(
            f"{task_key}: |Omega|={sup.n_omega} rank(X)={sup.rank} joint_dim={sup.joint_dim} "
            f"rectangles={sup.n_rectangles} -> {C.track_rel(path)}"
        )
    graph.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    C.log(f"support graph -> {C.track_rel(graph)}")
    return payload


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="build the Train-only joint support + support graph")
    ap.add_argument("--tasks", nargs="+", default=list(C.TASK_ORDER))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--graph", default=None)
    args = ap.parse_args(argv)
    payload = write_artifacts(args.tasks, out_dir=args.out_dir, graph_path=args.graph)
    print(C.stable_json({t: {"n_omega": v["n_omega"], "rank_X": v["rank_X"],
                             "joint_subspace_dim": v["joint_subspace_dim"],
                             "n_rectangles": v["n_rectangles"]}
                         for t, v in payload["tasks"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
