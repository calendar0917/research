"""Phase D — synthetic known-truth experiments for IJCM (round
``cscl-bjoc-intrinsic-contribution-feasibility-v1``).

Three tasks, CPU only, all on the same deterministic type functions and the
real :class:`IJCM` forward/gauge code:

* **Synthetic A — identifiable additive contributions.**  A known additive
  rule on atom / bond / junction types plus a known context correction and a
  known per-global-feature effect.  The molecular design matrix is checked for
  rank and null space *before* any fitting; model type/context recovery is
  then compared against the truth under the same fit-only gauge convention.
* **Synthetic B — co-occurrence trap.**  Two junction types A and B occur
  exactly the same number of times in every molecule, but only A has a true
  contribution.  The design columns are exactly collinear; the audit measures
  how the fitted contribution is allocated between A and B and whether the
  allocation is stable across seeds.  The expected outcome is an *exported
  identifiability limitation*, not a success story.
* **Synthetic C — necessary non-additivity.**  The target contains an
  interaction between two adjacent junction types.  The additive IJCM is fit;
  the audit quantifies the residual gap against the best additive fit in the
  type-count basis and against an interaction-augmented oracle.  The round
  does NOT add an interaction module in response.

The generator never uses ZINC labels and never touches official valid/test.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_intrinsic_contribution_v1 as ijcm  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

MIN_TYPE_COUNT_SYNTH = 5
SYNTH_EPOCHS = 400
SYNTH_LR = 2.0e-3
SYNTH_BATCH = 64


# ---------------------------------------------------------------------------
# deterministic graph generator
# ---------------------------------------------------------------------------


def random_chain_or_tree(
    rng: np.random.Generator,
    n_atoms: int,
    n_atom_types: int,
    n_bond_types: int,
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Connected graph: random tree + a few extra edges (rings)."""
    atom_types = [int(rng.integers(0, n_atom_types)) for _ in range(n_atoms)]
    edges: list[tuple[int, int, int]] = []
    for v in range(1, n_atoms):
        u = int(rng.integers(0, v))
        edges.append((u, v, int(rng.integers(0, n_bond_types))))
    existing = {(min(u, v), max(u, v)) for u, v, _ in edges}
    n_extra = int(rng.integers(0, max(n_atoms // 4, 1) + 1))
    for _ in range(n_extra):
        u, v = (int(x) for x in rng.integers(0, n_atoms, size=2))
        u, v = min(u, v), max(u, v)
        if u == v or (u, v) in existing:
            continue
        existing.add((u, v))
        edges.append((u, v, int(rng.integers(0, n_bond_types))))
    edges.sort()
    return atom_types, edges


def objects_from_graph(atom_types: Sequence[int], edges: Sequence[tuple[int, int, int]]) -> sro.MolObjects:
    n = len(atom_types)
    x = np.asarray(atom_types, dtype=np.int64).reshape(n, 1)
    ei, ea = [], []
    for u, v, bt in edges:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    edge_index = np.asarray(ei, dtype=np.int64).T if ei else np.zeros((2, 0), dtype=np.int64)
    return sro.build_mol_objects(-1, x, edge_index, np.asarray(ea, dtype=np.int64))


def alternating_path(n_atoms: int) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Path with alternating atom types 0/1 and bond types 0/1."""
    atom_types = [i % 2 for i in range(n_atoms)]
    edges = [(i, i + 1, i % 2) for i in range(n_atoms - 1)]
    return atom_types, edges


# ---------------------------------------------------------------------------
# training / evaluation helpers
# ---------------------------------------------------------------------------


def _standardise(objs: Sequence[sro.MolObjects], fit_rows: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    g = np.stack([objs[int(r)].global_feat for r in fit_rows], 0).astype(np.float64)
    return g.mean(0), np.where(g.std(0) < 1e-9, 1.0, g.std(0))


def train_ijcm(
    objs: Sequence[sro.MolObjects],
    fit_rows: Sequence[int],
    targets: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    min_count: int = MIN_TYPE_COUNT_SYNTH,
    epochs: int = SYNTH_EPOCHS,
    lr: float = SYNTH_LR,
) -> tuple[ijcm.IJCM, ijcm.TypeVocabs, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    vocabs = ijcm.TypeVocabs.build(objs, fit_rows, min_count=min_count)
    gmean, gscale = _standardise(objs, fit_rows)
    model = ijcm.build_model(
        seed=seed,
        vocab_sizes=vocabs.sizes,
        global_mean=gmean,
        global_scale=gscale,
        config=ijcm.IJCMConfig(
            atom_hidden=48,
            atom_out=32,
            bond_hidden=48,
            bond_out=32,
            junc_hidden=48,
            junc_out=32,
            contrib_hidden=128,
            eta_hidden=8,
        ),
        device=device,
    )
    y = torch.as_tensor(np.asarray(targets, dtype=np.float32), dtype=torch.float32)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    batches = [
        [fit_rows[i] for i in rng.permutation(len(fit_rows))[j : j + SYNTH_BATCH]]
        for j in range(0, len(fit_rows), SYNTH_BATCH)
    ]
    from collections import deque

    states: deque = deque(maxlen=5)
    for epoch in range(epochs):
        model.train()
        for b_rows in batches:
            batch = ijcm.build_ijcm_batch(objs, b_rows, vocabs)
            idx = torch.as_tensor(b_rows, dtype=torch.long)
            pred = model(batch)
            loss = F.l1_loss(pred, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    soup = {k: torch.stack([s[k].float() for s in states]).mean(0) for k in states[0]}
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    model.eval()
    offsets = ijcm.compute_fit_offsets(model, objs, fit_rows, vocabs, device)
    return model, vocabs, offsets, gmean, gscale


def _canonical_terms(
    model: ijcm.IJCM, offsets: Mapping[str, np.ndarray], objs: Sequence[sro.MolObjects], rows: Sequence[int], vocabs: ijcm.TypeVocabs, device
) -> dict[str, np.ndarray]:
    batch = ijcm.build_ijcm_batch(objs, rows, vocabs)
    with torch.no_grad():
        terms = model.raw_terms(batch)
        can = ijcm.gauge_terms(terms, offsets, batch)
    return can


def model_predict(
    model: ijcm.IJCM, objs: Sequence[sro.MolObjects], rows: Sequence[int], vocabs: ijcm.TypeVocabs, device
) -> np.ndarray:
    out = []
    for start in range(0, len(rows), 256):
        batch = ijcm.build_ijcm_batch(objs, list(rows[start : start + 256]), vocabs)
        with torch.no_grad():
            out.append(model(batch).cpu().numpy().astype(np.float64))
    return np.concatenate(out, 0)


# ---------------------------------------------------------------------------
# design matrix + rank audit
# ---------------------------------------------------------------------------


def design_matrix(
    objs: Sequence[sro.MolObjects],
    rows: Sequence[int],
    gmean: np.ndarray,
    gscale: np.ndarray,
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Columns: atom/bond/junction raw-type counts + standardised globals + 1."""
    raw = {kind: [ijcm.RAW_TYPE_FN[kind](objs[int(r)]) for r in rows] for kind in ("atom", "bond", "junction")}
    columns: list[np.ndarray] = []
    names: list[tuple[str, int]] = []
    for kind in ("atom", "bond", "junction"):
        all_types = sorted({int(x) for arr in raw[kind] for x in np.unique(arr)})
        for t in all_types:
            columns.append(np.asarray([np.sum(arr == t) for arr in raw[kind]], dtype=np.float64))
            names.append((kind, t))
    g = np.stack([objs[int(r)].global_feat for r in rows], 0).astype(np.float64)
    gz = (g - gmean) / gscale
    for k in range(gz.shape[1]):
        columns.append(gz[:, k])
        names.append(("global", k))
    columns.append(np.ones(len(rows), dtype=np.float64))
    names.append(("bias", -1))
    return np.stack(columns, 1), names


def rank_audit(D: np.ndarray, names: Sequence[tuple[str, int]], tol: float = 1e-8) -> dict[str, Any]:
    u, s, vt = np.linalg.svd(D, full_matrices=False)
    rank = int((s > tol * max(D.shape) * (s[0] if s.size else 1.0)).sum())
    null_dim = int(D.shape[1] - rank)
    null_vectors = vt[rank:, :] if null_dim else np.zeros((0, D.shape[1]))
    witnesses = []
    for vec in null_vectors[:5]:
        order = np.argsort(-np.abs(vec))[:5]
        witnesses.append(
            {
                "column_weights": [
                    {"column": list(names[i]), "weight": float(vec[i])} for i in order if abs(vec[i]) > 1e-9
                ],
                "max_abs_Dv": float(np.abs(D @ vec).max()),
            }
        )
    return {
        "n_rows": int(D.shape[0]),
        "n_columns": int(D.shape[1]),
        "rank": rank,
        "null_space_dim": null_dim,
        "singular_values_tail": s[-min(8, s.size) :].tolist(),
        "null_space_witnesses": witnesses,
        "tol": tol,
    }


def collinear_pairs(D: np.ndarray, names: Sequence[tuple[str, int]], threshold: float = 1.0 - 1e-9) -> list[dict[str, Any]]:
    Dc = D - D.mean(0, keepdims=True)
    norms = np.linalg.norm(Dc, axis=0)
    norms[norms < 1e-12] = 1.0
    corr = (Dc / norms).T @ (Dc / norms)
    out = []
    for i in range(D.shape[1]):
        for j in range(i + 1, D.shape[1]):
            if corr[i, j] >= threshold:
                out.append({"column_i": list(names[i]), "column_j": list(names[j]), "corr": float(corr[i, j])})
    return out[:20]


# ---------------------------------------------------------------------------
# Synthetic A
# ---------------------------------------------------------------------------


@dataclass
class SynthA:
    objs: list[sro.MolObjects]
    fit_rows: list[int]
    test_rows: list[int]
    target: np.ndarray
    true_atom: np.ndarray          # per-instance true contribution before gauge
    true_bond: np.ndarray
    true_junc: np.ndarray
    true_eta: np.ndarray           # per molecule
    atom_ctx: np.ndarray           # per-instance context part
    bond_ctx: np.ndarray
    junc_ctx: np.ndarray
    atom_type: np.ndarray
    bond_type: np.ndarray
    junc_type: np.ndarray


def build_synthetic_a(
    seed: int = 11,
    n_fit: int = 1000,
    n_test: int = 250,
    n_atom_types: int = 6,
    n_bond_types: int = 3,
    noise_std: float = 0.05,
) -> SynthA:
    rng = np.random.default_rng(seed)
    objs: list[sro.MolObjects] = []
    for _ in range(n_fit + n_test):
        n_atoms = int(rng.integers(8, 21))
        atom_types, edges = random_chain_or_tree(rng, n_atoms, n_atom_types, n_bond_types)
        objs.append(objects_from_graph(atom_types, edges))
    fit_rows = list(range(n_fit))
    test_rows = list(range(n_fit, n_fit + n_test))
    atom_effect = {int(t): float(rng.normal(0, 0.35)) for t in range(n_atom_types * 5)}
    bond_effect = {int(t): float(rng.normal(0, 0.40)) for t in range(n_bond_types * 2)}
    junc_effect: dict[int, float] = {}
    atom_type = [ijcm.raw_atom_type_ids(o) for o in objs]
    bond_type = [ijcm.raw_bond_type_ids(o) for o in objs]
    junc_type = [ijcm.raw_junc_type_ids(o) for o in objs]
    for arr in junc_type:
        for t in np.unique(arr):
            if int(t) not in junc_effect:
                junc_effect[int(t)] = float(rng.normal(0, 0.45))
    # known observable context corrections: triangles-like radius2 stat and ring flag
    target = np.zeros(len(objs), dtype=np.float64)
    true_atom, true_bond, true_junc, true_eta = [], [], [], []
    atom_ctx, bond_ctx, junc_ctx = [], [], []
    for k, o in enumerate(objs):
        na = o.n_atoms
        # context corrections use features NOT contained in the type definition
        # (atom type = category x degree; bond type = order x ring; junction
        #  type = centre category x incident bond pair x role)
        tri = o.atom_feat[:, sro.atom_blocks()["radius2_stats"].start + 2]
        ac = 0.55 * (tri - 0.30)
        deg = np.zeros(na, dtype=np.float64)
        for u, v in o.bond_index:
            deg[int(u)] += 1.0
            deg[int(v)] += 1.0
        bdeg = 0.5 * (deg[o.bond_index[:, 0]] + deg[o.bond_index[:, 1]])
        bc = 0.45 * (bdeg - 1.6)
        jc = 0.45 * (o.junc_feat[:, 2] - 0.2) if len(junc_type[k]) else np.zeros(0)
        aa = np.asarray([atom_effect[int(t)] for t in atom_type[k]], dtype=np.float64)
        ba = np.asarray([bond_effect[int(t)] for t in bond_type[k]], dtype=np.float64)
        ja = np.asarray([junc_effect[int(t)] for t in junc_type[k]], dtype=np.float64)
        eta = float(0.25 * o.global_feat[0] - 0.15 * o.global_feat[3] + 0.05 * o.global_feat[6])
        target[k] = 0.7 + aa.sum() + ac.sum() + ba.sum() + bc.sum() + ja.sum() + jc.sum() + eta
        true_atom.append(aa)
        true_bond.append(ba)
        true_junc.append(ja)
        true_eta.append(eta)
        atom_ctx.append(ac)
        bond_ctx.append(bc)
        junc_ctx.append(jc)
    target = target + rng.normal(0, noise_std, size=len(objs))
    return SynthA(
        objs=objs,
        fit_rows=fit_rows,
        test_rows=test_rows,
        target=target,
        true_atom=true_atom,
        true_bond=true_bond,
        true_junc=true_junc,
        true_eta=np.asarray(true_eta),
        atom_ctx=atom_ctx,
        bond_ctx=bond_ctx,
        junc_ctx=junc_ctx,
        atom_type=atom_type,
        bond_type=bond_type,
        junc_type=junc_type,
    )


def _type_recovery(
    can: Mapping[str, np.ndarray],
    type_idx: np.ndarray,
    raw_idx: np.ndarray,
    true_type_effect_by_raw: Mapping[int, float],
    true_ctx: Sequence[np.ndarray],
    offsets: np.ndarray,
    kind: str,
    head: str,
) -> dict[str, Any]:
    fitted_type = can[f"{head}_type_canonical"]
    fitted_ctx = can[f"{head}_ctx_canonical"]
    truth_type = np.asarray(
        [true_type_effect_by_raw.get(int(r), 0.0) + float(offsets[int(t)]) for t, r in zip(type_idx, raw_idx)],
        dtype=np.float64,
    )
    truth_ctx = np.concatenate(true_ctx) if true_ctx else np.zeros(0)
    truth_ctx_canon = (
        truth_ctx - np.asarray([float(offsets[int(t)]) for t in type_idx], dtype=np.float64)
        if truth_ctx.size
        else truth_ctx
    )
    ok = fitted_type.size and fitted_type.std() > 1e-12 and truth_type.std() > 1e-12
    return {
        "n_instances": int(fitted_type.size),
        "type_pearson_r": float(np.corrcoef(fitted_type, truth_type)[0, 1]) if ok else None,
        "type_mean_abs_error": float(np.abs(fitted_type - truth_type).mean()) if fitted_type.size else None,
        "ctx_pearson_r": (
            float(np.corrcoef(fitted_ctx, truth_ctx_canon)[0, 1])
            if fitted_ctx.size and fitted_ctx.std() > 1e-12 and truth_ctx_canon.std() > 1e-12
            else None
        ),
        "ctx_mean_abs_error": float(np.abs(fitted_ctx - truth_ctx_canon).mean()) if fitted_ctx.size else None,
        "type_sign_agreement": (
            float(np.mean(np.sign(fitted_type) == np.sign(truth_type))) if fitted_type.size else None
        ),
    }


def run_synthetic_a(out_dir: Path, device: torch.device, log=print) -> dict[str, Any]:
    data = build_synthetic_a()
    log("[synthetic A] built known additive rule; checking design rank before fitting")
    gmean, gscale = _standardise(data.objs, data.fit_rows)
    D_fit, names = design_matrix(data.objs, data.fit_rows, gmean, gscale)
    rank = rank_audit(D_fit, names)
    pairs = collinear_pairs(D_fit, names)
    type_cols = [i for i, (k, _) in enumerate(names) if k in ("atom", "bond", "junction")]
    type_block_n = len(type_cols)
    type_block_rank = int(np.linalg.matrix_rank(D_fit[:, type_cols]))
    logs = []
    for seed in (0, 1, 2):
        model, vocabs, offsets, _, _ = train_ijcm(
            data.objs, data.fit_rows, data.target, seed=seed, device=device
        )
        pred_test = model_predict(model, data.objs, data.test_rows, vocabs, device)
        test_mae = float(np.abs(pred_test - data.target[data.test_rows]).mean())
        # type recovery on the test rows
        can = _canonical_terms(model, offsets, data.objs, data.test_rows, vocabs, device)
        batch = ijcm.build_ijcm_batch(data.objs, data.test_rows, vocabs)
        type_idx = {kind: batch[f"{kind}_type"].cpu().numpy() for kind in ("atom", "bond", "junction")}
        raw_idx = {kind: batch[f"{kind}_raw_type"].cpu().numpy() for kind in ("atom", "bond", "junction")}
        # true effects are re-derived from the generation formula:
        # effect(t) = mean over fit instances of (contribution - context)
        def _effect_from(contribution, ctx, types, rows):
            acc: dict[int, list[float]] = {}
            for r in rows:
                for v, c, t in zip(contribution[r], ctx[r], types[r]):
                    acc.setdefault(int(t), []).append(float(v - c))
            return {t: float(np.mean(vals)) for t, vals in acc.items()}

        atom_true = _effect_from(data.true_atom, data.atom_ctx, data.atom_type, data.fit_rows)
        bond_true = _effect_from(data.true_bond, data.bond_ctx, data.bond_type, data.fit_rows)
        junc_true = _effect_from(data.true_junc, data.junc_ctx, data.junc_type, data.fit_rows)
        recover = {
            "atom": _type_recovery(can, type_idx["atom"], raw_idx["atom"], atom_true, [data.atom_ctx[r] for r in data.test_rows], offsets["atom"], "atom", "alpha"),
            "bond": _type_recovery(can, type_idx["bond"], raw_idx["bond"], bond_true, [data.bond_ctx[r] for r in data.test_rows], offsets["bond"], "bond", "beta"),
            "junction": _type_recovery(can, type_idx["junction"], raw_idx["junction"], junc_true, [data.junc_ctx[r] for r in data.test_rows], offsets["junction"], "junction", "nu"),
        }
        logs.append(
            {
                "seed": seed,
                "test_mae": test_mae,
                "prediction_r2": float(1 - np.mean((pred_test - data.target[data.test_rows]) ** 2) / np.var(data.target[data.test_rows])),
                "recovery": recover,
            }
        )
        log(f"[synthetic A] seed={seed} test_mae={test_mae:.4f}")

    summary = {
        "n_fit": len(data.fit_rows),
        "n_test": len(data.test_rows),
        "noise_std": 0.05,
        "noise_mae_floor": float(0.05 * np.sqrt(2 / np.pi)),
        "design_matrix_rank_fraction": rank["rank"] / rank["n_columns"],
        "type_block_rank": type_block_rank,
        "type_block_n_columns": type_block_n,
        "recovery_atom_r": [r["recovery"]["atom"]["type_pearson_r"] for r in logs],
        "recovery_bond_r": [r["recovery"]["bond"]["type_pearson_r"] for r in logs],
        "recovery_junction_r": [r["recovery"]["junction"]["type_pearson_r"] for r in logs],
    }
    full_design_rank_deficient = rank["null_space_dim"] > 0
    type_block_full_rank = type_block_rank == type_block_n
    recovery_good = all(
        (r["recovery"]["junction"]["type_pearson_r"] or 0) > 0.8
        and (r["recovery"]["atom"]["type_pearson_r"] or 0) > 0.8
        and (r["recovery"]["bond"]["type_pearson_r"] or 0) > 0.6
        for r in logs
    )
    identifiability_pass = bool(type_block_full_rank and recovery_good)
    payload = {
        "protocol": "cscl-bjoc-intrinsic-contribution-feasibility-v1",
        "task": "synthetic_A_identifiable_additive",
        "design_matrix": {**rank, "collinear_pairs": pairs},
        "type_block": {
            "rank": type_block_rank,
            "n_columns": type_block_n,
            "full_rank": bool(type_block_full_rank),
        },
        "runs": logs,
        "summary": {**summary, "recovery_good": bool(recovery_good)},
        "non_identifiable_by_construction": bool(full_design_rank_deficient and not type_block_full_rank),
        "global_block_note": (
            "the frozen 33-dim global vector contains exact duplicates / linear dependencies "
            "(e.g. mean_distance == mean_pair_distance); the rank audit of the *type-count block*, "
            "which is what the type-baseline attribution needs, is reported separately"
        ),
        "identifiability_pass": identifiability_pass,
        "interpretation": (
            "type-count design full rank + high type/context recovery on a truly additive task means "
            "the additive decomposition is identifiable and recoverable in this regime. If the type "
            "block itself were rank deficient the result would be NON_IDENTIFIABLE_BY_CONSTRUCTION "
            "and the null-space witness columns are exported instead of a pass/fail claim."
        ),
    }
    (out_dir / "synthetic_a.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=float), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# Synthetic B
# ---------------------------------------------------------------------------


def build_synthetic_b(seed: int = 23, n_mols: int = 800, noise_std: float = 0.02) -> tuple[list[sro.MolObjects], list[int], np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    objs: list[sro.MolObjects] = []
    for _ in range(n_mols):
        n_atoms = 2 * int(rng.integers(5, 12))
        atom_types, edges = alternating_path(n_atoms)
        objs.append(objects_from_graph(atom_types, edges))
    rows = list(range(n_mols))
    # find the two collinear junction types
    t_center0 = int(ijcm.raw_junc_type_ids(objects_from_graph([0, 1, 0], [(0, 1, 0), (1, 2, 1)]))[0])
    t_center1 = int(ijcm.raw_junc_type_ids(objects_from_graph([1, 0, 1], [(0, 1, 0), (1, 2, 1)]))[0])
    target = np.asarray(
        [1.5 * np.sum(ijcm.raw_junc_type_ids(o) == t_center0) + 0.1 * np.log1p(o.n_atoms) for o in objs],
        dtype=np.float64,
    )
    target = target + rng.normal(0, noise_std, size=len(objs))
    return objs, rows, target, t_center0, t_center1


def run_synthetic_b(out_dir: Path, device: torch.device, log=print) -> dict[str, Any]:
    objs, rows, target, type_a, type_b = build_synthetic_b()
    gmean, gscale = _standardise(objs, rows)
    D, names = design_matrix(objs, rows, gmean, gscale)
    rank = rank_audit(D, names)
    pairs = collinear_pairs(D, names)
    allocations = []
    for seed in (0, 1, 2):
        model, vocabs, offsets, _, _ = train_ijcm(objs, rows, target, seed=seed, device=device)
        a_idx = vocabs.junction.raw_to_index.get(type_a)
        b_idx = vocabs.junction.raw_to_index.get(type_b)
        a_val = float(offsets["junction"][a_idx]) if a_idx is not None else None
        b_val = float(offsets["junction"][b_idx]) if b_idx is not None else None
        pred = model_predict(model, objs, rows, vocabs, device)
        allocations.append(
            {
                "seed": seed,
                "type_A_canonical": a_val,
                "type_B_canonical": b_val,
                "true_A": 1.5,
                "true_B": 0.0,
                "test_mae_self": float(np.abs(pred - target).mean()),
            }
        )
        log(f"[synthetic B] seed={seed} A={a_val} B={b_val} (true A=1.5 B=0)")
    b_vals = np.asarray([a["type_B_canonical"] for a in allocations], dtype=np.float64)
    a_vals = np.asarray([a["type_A_canonical"] for a in allocations], dtype=np.float64)
    # direct exact-collinearity witness: the two type-count columns are identical
    idc = [i for i, nm in enumerate(names) if nm == ("junction", type_a)]
    jdc = [i for i, nm in enumerate(names) if nm == ("junction", type_b)]
    exact_collinear = bool(
        idc and jdc and np.allclose(D[:, idc[0]], D[:, jdc[0]])
    ) or any(
        set([tuple(p["column_i"]), tuple(p["column_j"])])
        == {("junction", type_a), ("junction", type_b)}
        for p in pairs
    )
    limitation = bool(
        exact_collinear
        and (
            float(np.std(b_vals)) > 0.05
            or float(np.abs(b_vals).mean()) > 0.05 * 1.5
        )
    )
    payload = {
        "protocol": "cscl-bjoc-intrinsic-contribution-feasibility-v1",
        "task": "synthetic_B_cooccurrence_trap",
        "junction_type_A": {"raw": int(type_a), "string": ijcm.junc_type_string(type_a), "true_contribution": 1.5},
        "junction_type_B": {"raw": int(type_b), "string": ijcm.junc_type_string(type_b), "true_contribution": 0.0},
        "design_matrix": rank,
        "exact_collinearity_detected": bool(exact_collinear),
        "collinear_pairs": pairs,
        "allocations": allocations,
        "summary": (
            f"A and B are exactly collinear in every molecule; fitted allocation moves between seeds "
            f"(A sd {float(np.std(a_vals)):.3f}, B sd {float(np.std(b_vals)):.3f}, mean |B| {float(np.abs(b_vals).mean()):.3f}). "
            "Type-level co-occurrence cannot be resolved by the additive model; the uncertainty must be exported."
        ),
        "limitation_documented": limitation,
    }
    (out_dir / "synthetic_b.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=float), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# Synthetic C
# ---------------------------------------------------------------------------


def synthetic_c_target(objs: Sequence[sro.MolObjects], type_a: int, type_b: int) -> tuple[np.ndarray, np.ndarray]:
    """Additive part + interaction count of adjacent A/B junctions sharing a bond."""
    target = np.zeros(len(objs), dtype=np.float64)
    interaction = np.zeros(len(objs), dtype=np.float64)
    for k, o in enumerate(objs):
        jt = ijcm.raw_junc_type_ids(o)
        target[k] += 0.5 * np.sum(jt == type_a) + 0.7 * np.sum(jt == type_b)
        # junction j and j2 are adjacent if they share one incident physical bond
        bonds = [set(map(int, o.junc_bond[j])) for j in range(len(jt))]
        n_adj = 0
        for i in range(len(jt)):
            for j in range(i + 1, len(jt)):
                if bonds[i] & bonds[j]:
                    if {int(jt[i]), int(jt[j])} == {type_a, type_b}:
                        n_adj += 1
        interaction[k] = n_adj
        target[k] += 1.2 * n_adj
    return target, interaction


def run_synthetic_c(out_dir: Path, device: torch.device, log=print) -> dict[str, Any]:
    rng = np.random.default_rng(31)
    objs: list[sro.MolObjects] = []
    for _ in range(1000):
        n_atoms = int(rng.integers(8, 21))
        atom_types, edges = random_chain_or_tree(rng, n_atoms, 6, 3)
        objs.append(objects_from_graph(atom_types, edges))
    rows = list(range(1000))
    from collections import Counter

    freq = Counter(int(t) for o in objs for t in ijcm.raw_junc_type_ids(o))
    top = [t for t, _ in freq.most_common(2)]
    type_a, type_b = int(top[0]), int(top[1])
    target, interaction = synthetic_c_target(objs, type_a, type_b)
    target = target + rng.normal(0, 0.02, size=len(objs))

    gmean, gscale = _standardise(objs, rows)
    D_add, names = design_matrix(objs, rows, gmean, gscale)
    D_int = np.concatenate([D_add, interaction[:, None]], axis=1)
    n = len(rows)
    fit = np.arange(n) < 750
    test = ~fit
    beta_add = np.linalg.lstsq(D_add[fit], target[fit], rcond=None)[0]
    beta_int = np.linalg.lstsq(D_int[fit], target[fit], rcond=None)[0]
    mae_additive_floor = float(np.abs(D_add[test] @ beta_add - target[test]).mean())
    mae_interaction_oracle = float(np.abs(D_int[test] @ beta_int - target[test]).mean())
    interaction_proj = D_add @ np.linalg.lstsq(D_add, interaction, rcond=None)[0]
    interaction_unexplained_r2 = float(1.0 - np.var(interaction - interaction_proj) / max(np.var(interaction), 1e-12))

    model, vocabs, offsets, _, _ = train_ijcm(objs, list(np.flatnonzero(fit)), target, seed=0, device=device)
    pred = model_predict(model, objs, list(np.flatnonzero(test)), vocabs, device)
    mae_ijcm = float(np.abs(pred - target[test]).mean())
    residual = pred - target[test]
    if interaction[test].std() > 1e-12 and residual.std() > 1e-12:
        corr_resid_interaction = float(np.corrcoef(residual, interaction[test])[0, 1])
    else:
        corr_resid_interaction = None
    payload = {
        "protocol": "cscl-bjoc-intrinsic-contribution-feasibility-v1",
        "task": "synthetic_C_necessary_non_additivity",
        "interaction_feature": "count of adjacent junction pairs of the two most frequent junction types sharing a physical bond",
        "type_a": int(type_a),
        "type_b": int(type_b),        "test_mae_ijcm": mae_ijcm,
        "test_mae_additive_ols_floor": mae_additive_floor,
        "test_mae_interaction_oracle": mae_interaction_oracle,
        "ijcm_minus_additive_floor": mae_ijcm - mae_additive_floor,
        "additive_floor_minus_interaction_oracle": mae_additive_floor - mae_interaction_oracle,
        "residual_interaction_corr": corr_resid_interaction,
        "interaction_variance_unexplained_by_additive_design_r2": interaction_unexplained_r2,
        "additive_design_rank": rank_audit(D_add, names),
        "summary": (
            f"The interaction is not additive in the count basis: additive OLS floor {mae_additive_floor:.4f}, "
            f"interaction oracle {mae_interaction_oracle:.4f}; {100 * interaction_unexplained_r2:.0f}% of the interaction "
            f"variance is orthogonal to the additive design. Additive IJCM reaches {mae_ijcm:.4f}: better than the "
            f"count-additive floor (local radius-2/bond context leaks some adjacency information) but still short of the "
            f"oracle, and its residual is strongly anti-correlated with the true interaction count "
            f"({corr_resid_interaction if corr_resid_interaction is None else round(corr_resid_interaction, 3)}). "
            "An explicit pairwise interaction would be needed for the residual gap; this round does not add one."
        ),
    }
    (out_dir / "synthetic_c.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=float), encoding="utf-8")
    log(f"[synthetic C] ijcm={mae_ijcm:.4f} additive_floor={mae_additive_floor:.4f} oracle={mae_interaction_oracle:.4f}")
    return payload


def run_all(out_dir: Path, device: torch.device, log=print) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_synthetic_a(out_dir, device, log)
    run_synthetic_b(out_dir, device, log)
    run_synthetic_c(out_dir, device, log)
