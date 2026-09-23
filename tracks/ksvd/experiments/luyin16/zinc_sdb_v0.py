"""SDB-v0 runner — Sparse Structural Dictionary Binding.

Round: ``sdb-v0``.  Pre-registration: ``tracks/ksvd/notes/sdb_v0_preregistration.md``
(frozen).  Official ZINC **test is never loaded** (`test_policy = no_test`).

Stages
------
``dict gate1 sanity synthetic gate2 report all``

* ``dict``      — fit the frozen ``D_KSVD(K=32, s=8)`` on the FSAR pure-topology
  coordinate cache (plus the random and rank-32 PCA references).
* ``gate1``     — label-free dictionary-domain gate (``E_phi`` / ``E_bind`` /
  health / semantic audit).  Uses **no** ``y``.
* ``sanity``    — Stage-0 implementation correctness tests.
* ``synthetic`` — assignment-only positive / marginal-only negative controls.
* ``gate2``     — FSAR mechanism-preservation gate (frozen ``M0`` + linear
  residual arms), only meaningful after ``gate1`` PASS.

Every stage reuses the FSAR-R2-AR0 feature cache and the durable FSAR ``M0``
soup states; no historical baseline is re-trained.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2
from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0_mechanism as mech
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_fsar_r2_ar0 as runner

REPO_ROOT = shead.REPO_ROOT
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/sdb_v0"
DICT_PATH = RESULTS_DIR / "dictionary.pt"
DICT_META = RESULTS_DIR / "dictionary_meta.json"
STAGE1_JSON = RESULTS_DIR / "stage1_label_free.json"
STAGE1_MD = RESULTS_DIR / "STAGE1_REPORT.md"
STAGE2_JSON = RESULTS_DIR / "stage2_mechanism.json"
STAGE2_MD = RESULTS_DIR / "STAGE2_REPORT.md"
SANITY_JSON = RESULTS_DIR / "sanity.json"
SYNTHETIC_JSON = RESULTS_DIR / "synthetic_controls.json"
DECISION_JSON = RESULTS_DIR / "decision.json"

PROTOCOL_VERSION = "sdb_v0"

_write_json = shead._write_json
_read_json = shead._read_json
_git_commit = shead._git_commit

PHI_GROUPS = {
    "root_basis": (0, 11),
    "node_mean": (11, 22),
    "node_std": (22, 33),
    "edge_mean": (33, 48),
    "edge_std": (48, 63),
    "patch_size": (63, 65),
}

# frozen Stage-1 thresholds (pre-registration §8)
STAGE1_E_BIND_PASS = 0.10
STAGE1_E_BIND_STOP = 0.50
STAGE1_PCA_SLACK = 3.0
STAGE1_USED_PASS = 24
STAGE1_USED_STOP = 16

# frozen Stage-2 thresholds (pre-registration §9)
STAGE2_RECOVERY_PASS = 0.75
STAGE2_RECOVERY_STOP = 0.50
STAGE2_DENSE_SLACK = 0.005
STAGE2_SHUFFLE_DEGRADE = 0.02
PROTOCOL_DRIFT_TOL = 1.0e-6


# ---------------------------------------------------------------------------
# data / dictionary
# ---------------------------------------------------------------------------


def build_datasets():
    return runner.build_datasets()


def _concatenate_phi(molecules: Sequence[Any]) -> np.ndarray:
    return np.concatenate([np.asarray(m.phi, dtype=np.float64) for m in molecules], axis=0)


def fit_dictionary(
    epochs: int = 10,
    max_fit_atoms: int | None = None,
    force: bool = False,
    log: bool = True,
) -> dict[str, Any]:
    if DICT_PATH.exists() and DICT_META.exists() and not force:
        meta = _read_json(DICT_META)
        if meta.get("protocol_version") == PROTOCOL_VERSION:
            if log:
                print(f"[dict] reuse {DICT_PATH}", flush=True)
            return meta
    train, _valid, _scalers, _meta = build_datasets()
    started = time.perf_counter()
    X = _concatenate_phi(train)
    if max_fit_atoms is not None and int(max_fit_atoms) < X.shape[0]:
        rng = np.random.default_rng(int(sdb.DICT_SEED))
        take = rng.choice(X.shape[0], size=int(max_fit_atoms), replace=False)
        X_fit = X[np.sort(take)]
    else:
        X_fit = X
    if log:
        print(f"[dict] fitting K-SVD K={sdb.K_ATOMS} s={sdb.SPARSITY} on {X_fit.shape[0]} atoms", flush=True)
    D_ksvd, info = sdb.fit_ksvd(X_fit, atoms=sdb.K_ATOMS, s=sdb.SPARSITY, epochs=epochs, seed=sdb.DICT_SEED)
    D_rand = sdb.random_normalized_dictionary()
    pca = sdb.fit_pca_rank(X_fit, sdb.K_ATOMS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "D_ksvd": torch.as_tensor(D_ksvd, dtype=torch.float64),
            "D_rand": torch.as_tensor(D_rand, dtype=torch.float64),
            "pca_mean": torch.as_tensor(pca.mean, dtype=torch.float64),
            "pca_components": torch.as_tensor(pca.components, dtype=torch.float64),
        },
        DICT_PATH,
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "dict_seed": int(sdb.DICT_SEED),
        "atoms": int(sdb.K_ATOMS),
        "sparsity": int(sdb.SPARSITY),
        "phi_dim": int(sdb.PHI_DIM),
        "n_fit_atoms": int(X_fit.shape[0]),
        "n_train_atoms": int(X.shape[0]),
        "ksvd_epochs": int(epochs),
        "ksvd_final_fit_mse": float(info["history"][-1]["mean_sq_err"]) if info.get("history") else float("nan"),
        "git_commit": _git_commit(),
        "seconds": float(time.perf_counter() - started),
        "official_test_loaded": False,
    }
    _write_json(DICT_META, payload)
    if log:
        print(f"[dict] done ({payload['seconds']:.1f}s) {payload}", flush=True)
    return payload


def load_dictionary() -> tuple[np.ndarray, np.ndarray, sdb.PCARank]:
    if not DICT_PATH.exists():
        raise RuntimeError("dictionary missing; run the `dict` stage first")
    blob = torch.load(DICT_PATH, map_location="cpu", weights_only=True)
    D_ksvd = blob["D_ksvd"].numpy().astype(np.float64)
    D_rand = blob["D_rand"].numpy().astype(np.float64)
    pca = sdb.PCARank(
        mean=blob["pca_mean"].numpy().astype(np.float64),
        components=blob["pca_components"].numpy().astype(np.float64),
    )
    return D_ksvd, D_rand, pca


def _codes(molecules: Sequence[Any], D: np.ndarray, s: int | None = None) -> list[np.ndarray]:
    return sdb.codes_for_molecules(D, molecules, s=(sdb.SPARSITY if s is None else int(s)))


def _flatten_codes(codes: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=np.float64) for a in codes], axis=0)


def _cyclomatic_per_atom(molecules: Sequence[Any]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for molecule in molecules:
        rank = max(int(molecule.n_edges) - int(molecule.n_nodes) + 1, 0)
        out.append(np.full(int(molecule.n_nodes), float(rank), dtype=np.float64))
    return out


# ---------------------------------------------------------------------------
# Stage 1 — label-free dictionary-domain gate
# ---------------------------------------------------------------------------


def _semantic_audit(
    D: np.ndarray, molecules: Sequence[Any], codes: Sequence[np.ndarray]
) -> dict[str, Any]:
    phi = np.concatenate([np.asarray(m.phi, dtype=np.float64) for m in molecules], axis=0)
    alpha = _flatten_codes(codes)
    cycle = np.concatenate(_cyclomatic_per_atom(molecules), axis=0)
    per_atom: list[dict[str, Any]] = []
    for k in range(int(D.shape[1])):
        activation = np.abs(alpha[:, k])
        if not np.any(activation > 0):
            per_atom.append({"atom": k, "used": False})
            continue
        thresh = np.quantile(activation[activation > 0], 0.99)
        top = activation >= max(thresh, 1e-12)
        mean_phi = phi[top].mean(axis=0)
        groups = {name: float(mean_phi[lo:hi].mean()) for name, (lo, hi) in PHI_GROUPS.items()}
        per_atom.append(
            {
                "atom": k,
                "used": True,
                "n_top_atoms": int(top.sum()),
                "mean_patch_nodes": float(np.expm1(mean_phi[63])),
                "mean_patch_edges": float(np.expm1(mean_phi[64])),
                "mean_molecule_cycle_rank": float(cycle[top].mean()),
                "mean_group_value": groups,
                "support_frequency": float((activation > 0).mean()),
            }
        )
    return {"n_atoms": int(D.shape[1]), "per_atom": per_atom}


def stage1(force: bool = False) -> dict[str, Any]:
    if STAGE1_JSON.exists() and not force:
        return _read_json(STAGE1_JSON)
    fit_dictionary()
    train, valid, _scalers, _meta = build_datasets()
    D_ksvd, D_rand, pca = load_dictionary()

    alpha_train = _codes(train, D_ksvd)
    alpha_valid = _codes(valid, D_ksvd)
    alpha_train_dense = _codes(train, D_ksvd, s=sdb.K_ATOMS)
    alpha_valid_dense = _codes(valid, D_ksvd, s=sdb.K_ATOMS)
    alpha_train_rand = _codes(train, D_rand)
    alpha_valid_rand = _codes(valid, D_rand)
    pca_train = [pca.dense_codes(np.asarray(m.phi, dtype=np.float64)) for m in train]
    pca_valid = [pca.dense_codes(np.asarray(m.phi, dtype=np.float64)) for m in valid]

    X_train = _concatenate_phi(train)
    X_valid = _concatenate_phi(valid)

    e_phi = {
        "ksvd_train": sdb.structural_relative_error(X_train, _flatten_codes(alpha_train) @ D_ksvd.T),
        "ksvd_dev": sdb.structural_relative_error(X_valid, _flatten_codes(alpha_valid) @ D_ksvd.T),
        "rand_train": sdb.structural_relative_error(X_train, _flatten_codes(alpha_train_rand) @ D_rand.T),
        "rand_dev": sdb.structural_relative_error(X_valid, _flatten_codes(alpha_valid_rand) @ D_rand.T),
        "pca_train": sdb.structural_relative_error(X_train, _flatten_codes(pca_train) @ pca.components, mean=pca.mean),
        "pca_dev": sdb.structural_relative_error(X_valid, _flatten_codes(pca_valid) @ pca.components, mean=pca.mean),
    }

    e_bind = {
        "ksvd_train": sdb.binding_relative_error(D_ksvd, train, alpha_train),
        "ksvd_dev": sdb.binding_relative_error(D_ksvd, valid, alpha_valid),
        "ksvd_dense_train": sdb.binding_relative_error(D_ksvd, train, alpha_train_dense),
        "ksvd_dense_dev": sdb.binding_relative_error(D_ksvd, valid, alpha_valid_dense),
        "rand_train": sdb.binding_relative_error(D_rand, train, alpha_train_rand),
        "rand_dev": sdb.binding_relative_error(D_rand, valid, alpha_valid_rand),
        "pca_train": sdb.binding_relative_error(pca.components.T, train, pca_train),
        "pca_dev": sdb.binding_relative_error(pca.components.T, valid, pca_valid),
    }

    health = {
        "ksvd_train": sdb.dictionary_health(_flatten_codes(alpha_train)),
        "ksvd_dev": sdb.dictionary_health(_flatten_codes(alpha_valid)),
    }
    max_l0, within = sdb.exact_sparsity(_flatten_codes(alpha_valid))
    health["exact_sparsity_valid"] = {"max_l0": int(max_l0), "all_within_s": bool(within), "s": int(sdb.SPARSITY)}

    audit = _semantic_audit(D_ksvd, valid, alpha_valid)

    used = int(health["ksvd_train"]["used_atoms"])
    e_bind_ksvd_dev = float(e_bind["ksvd_dev"]["aggregate"])
    e_bind_rand_dev = float(e_bind["rand_dev"]["aggregate"])
    e_bind_pca_dev = float(e_bind["pca_dev"]["aggregate"])
    e_phi_ksvd_dev = float(e_phi["ksvd_dev"])
    e_phi_rand_dev = float(e_phi["rand_dev"])

    stop_reasons: list[str] = []
    if e_phi_ksvd_dev >= e_phi_rand_dev:
        stop_reasons.append("e_phi_ksvd_dev >= e_phi_rand_dev")
    if used < STAGE1_USED_STOP:
        stop_reasons.append(f"used_atoms {used} < {STAGE1_USED_STOP}")
    if e_bind_ksvd_dev > STAGE1_E_BIND_STOP:
        stop_reasons.append(f"e_bind_ksvd_dev {e_bind_ksvd_dev:.4f} > {STAGE1_E_BIND_STOP}")

    pass_reasons_fail: list[str] = []
    if not (e_bind_ksvd_dev <= STAGE1_E_BIND_PASS):
        pass_reasons_fail.append(f"e_bind_ksvd_dev {e_bind_ksvd_dev:.4f} > {STAGE1_E_BIND_PASS}")
    if not (e_bind_ksvd_dev < e_bind_rand_dev):
        pass_reasons_fail.append("e_bind_ksvd_dev >= e_bind_rand_dev")
    if not (used >= STAGE1_USED_PASS):
        pass_reasons_fail.append(f"used_atoms {used} < {STAGE1_USED_PASS}")

    if stop_reasons:
        verdict = "STOP"
        reasons = stop_reasons
    elif not pass_reasons_fail:
        verdict = "PASS"
        reasons = []
    else:
        verdict = "INSPECT"
        reasons = pass_reasons_fail

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "verdict": verdict,
        "reasons": reasons,
        "thresholds": {
            "e_bind_pass": STAGE1_E_BIND_PASS,
            "e_bind_stop": STAGE1_E_BIND_STOP,
            "used_pass": STAGE1_USED_PASS,
            "used_stop": STAGE1_USED_STOP,
        },
        "references_not_gated": {
            "pca_slack_reported": STAGE1_PCA_SLACK,
            "e_bind_ksvd_over_pca_dev": float(e_bind_ksvd_dev / (e_bind_pca_dev + 1e-30)),
            "e_bind_ksvd_dense_dev": float(e_bind["ksvd_dense_dev"]["aggregate"]),
            "amendment": "A1",
        },
        "e_phi": e_phi,
        "e_bind": e_bind,
        "health": health,
        "semantic_audit": audit,
        "split": {"n_train": int(len(train)), "n_valid": int(len(valid))},
        "official_test_loaded": False,
        "reused_artifacts": ["fsar_r2_ar0 cache", "fsar_r2_ar0 dictionary-free phi"],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(STAGE1_JSON, payload)
    _write_stage1_markdown(payload)
    return payload


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_stage1_markdown(payload: Mapping[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# SDB-v0 Stage 1 — label-free dictionary-domain gate\n")
    lines.append(f"Verdict: **{payload['verdict']}**")
    if payload["reasons"]:
        lines.append(f"\nReasons: {', '.join(payload['reasons'])}")
    lines.append("\n## Structural reconstruction `E_phi`\n")
    lines.append("| reference | train | dev |")
    lines.append("|---|---:|---:|")
    for ref in ("ksvd", "rand", "pca"):
        lines.append(
            f"| {ref} | {_fmt(payload['e_phi'][ref + '_train'])} | {_fmt(payload['e_phi'][ref + '_dev'])} |"
        )
    lines.append("\n## Binding reconstruction `E_bind = ||C_phi - R C_code||^2 / ||C_phi||^2`\n")
    lines.append("| reference | train agg | dev agg | dev per-mol median |")
    lines.append("|---|---:|---:|---:|")
    for ref in ("ksvd", "ksvd_dense", "rand", "pca"):
        if ref + "_dev" not in payload["e_bind"]:
            continue
        lines.append(
            f"| {ref} | {_fmt(payload['e_bind'][ref + '_train']['aggregate'])} "
            f"| {_fmt(payload['e_bind'][ref + '_dev']['aggregate'])} "
            f"| {_fmt(payload['e_bind'][ref + '_dev']['per_molecule_median'])} |"
        )
    ref = payload.get("references_not_gated", {})
    lines.append(
        f"\n*Sparse-vs-dense on the same learned `D`: `E_bind(sparse s=8) / E_bind(dense s=K)` = "
        f"{_fmt(payload['e_bind']['ksvd_dev']['aggregate'] / (ref.get('e_bind_ksvd_dense_dev', float('nan')) + 1e-30))}. "
        f"PCA reference ratio `E_bind_ksvd_dev / E_bind_pca_dev` = {_fmt(ref.get('e_bind_ksvd_over_pca_dev', float('nan')))} "
        f"(PCA is a near-lossless dense rank-32 affine projection; reported, not gated — Amendment A1).*"
    )
    h = payload["health"]["ksvd_dev"]
    lines.append("\n## Dictionary health (dev codes)\n")
    lines.append(
        f"- used atoms: {h['used_atoms']} / {h['atoms']} (dead {h['dead_atoms']})\n"
        f"- row coverage: {_fmt(h['row_coverage'])}\n"
        f"- top-1 / top-8 coefficient mass: {_fmt(h['top1_mass_mean'])} / {_fmt(h['top8_mass_mean'])}\n"
        f"- support entropy (normalized): {_fmt(h['support_entropy_normalized'])}\n"
        f"- exact sparsity: max l0 {payload['health']['exact_sparsity_valid']['max_l0']} "
        f"(within s={payload['health']['exact_sparsity_valid']['s']}: "
        f"{payload['health']['exact_sparsity_valid']['all_within_s']})"
    )
    lines.append("\n## Dictionary semantic audit (interpretation only)\n")
    used_atoms = [row for row in payload["semantic_audit"]["per_atom"] if row.get("used")]
    used_atoms.sort(key=lambda row: -row["support_frequency"])
    lines.append("| atom | support freq | mean patch nodes | mean patch edges | mean cycle rank | root_basis | node_mean | node_std | edge_mean | edge_std |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in used_atoms[:12]:
        g = row["mean_group_value"]
        lines.append(
            f"| {row['atom']} | {_fmt(row['support_frequency'])} | {_fmt(row['mean_patch_nodes'])} "
            f"| {_fmt(row['mean_patch_edges'])} | {_fmt(row['mean_molecule_cycle_rank'])} "
            f"| {_fmt(g['root_basis'])} | {_fmt(g['node_mean'])} | {_fmt(g['node_std'])} "
            f"| {_fmt(g['edge_mean'])} | {_fmt(g['edge_std'])} |"
        )
    lines.append("\n*Top-12 dictionary atoms by support frequency; the semantic audit is report-only and never an input.*\n")
    lines.append(f"\n`official_test_loaded = {payload['official_test_loaded']}`\n")
    STAGE1_MD.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Stage 0 — sanity + synthetic controls
# ---------------------------------------------------------------------------


def sanity() -> dict[str, Any]:
    from tracks.ksvd.tests import test_sdb_v0 as suite

    checks: dict[str, Any] = {}
    failures: list[str] = []
    for name in sorted(dir(suite)):
        if not name.startswith("check_"):
            continue
        function = getattr(suite, name)
        try:
            function()
            checks[name] = True
        except Exception as error:  # noqa: BLE001
            checks[name] = False
            failures.append(f"{name}: {error}")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "checks": checks,
        "n_checks": int(len(checks)),
        "n_pass": int(sum(1 for value in checks.values() if value)),
        "all_pass": bool(not failures),
        "failures": failures,
        "official_test_loaded": False,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(SANITY_JSON, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


def _toy_molecule(n: int, edges: list[tuple[int, int]], atom_idx: Sequence[int], y: float) -> r2.MoleculeFeatures:
    from tracks.ksvd.code.graph import from_edges

    graph = from_edges(int(n), list(edges))
    phi = r2.build_phi(graph)
    atom_idx = np.asarray(atom_idx, dtype=np.int64)
    edge_types = [1] * len(edges)
    marginal = r2.build_A(atom_idx, edge_types, int(n))
    return r2.MoleculeFeatures(
        phi=phi.astype(np.float32),
        atom_idx=atom_idx,
        A=marginal.astype(np.float32),
        n_nodes=int(n),
        n_edges=int(len(edges)),
        y=float(y),
    )


def _synthetic_positive() -> dict[str, Any]:
    """Fixed marginals, chemistry placement changes only; label = endpoint N."""
    D, _Drand, _pca = load_dictionary()
    hetero = 1  # N-like category, distinct from carbon 0
    examples: list[r2.MoleculeFeatures] = []
    for n in range(4, 9):
        path = [(i, i + 1) for i in range(n - 1)]
        for pos in range(n):
            atoms = [0] * n
            atoms[pos] = hetero
            y = 1.0 if pos in (0, n - 1) else 0.0
            examples.append(_toy_molecule(n, path, atoms, y))
    train = [m for i, m in enumerate(examples) if i % 2 == 0]
    valid = [m for i, m in enumerate(examples) if i % 2 == 1]
    codes_train = _codes(train, D)
    codes_valid = _codes(valid, D)
    c_train = sdb.binding_matrices_for_molecules(codes_train, train)
    c_valid = sdb.binding_matrices_for_molecules(codes_valid, valid)
    scale, mask = sdb.fit_rms_scaler(c_train)
    protocol = dict(shead.OPTIMIZED_PROTOCOL)
    protocol.update({"max_epochs": 240, "patience": 40})
    base_train = np.zeros(len(train), dtype=np.float64)
    base_valid = np.zeros(len(valid), dtype=np.float64)
    y_train = np.asarray([m.y for m in train], dtype=np.float64)
    y_valid = np.asarray([m.y for m in valid], dtype=np.float64)
    result = sdb.train_linear_residual(
        sdb.apply_rms_scaler(c_train, scale, mask), base_train, y_train,
        sdb.apply_rms_scaler(c_valid, scale, mask), base_valid, y_valid,
        seed=0, protocol=protocol, width=int(sdb.K_ATOMS),
    )
    # marginal-only reference: C_D replaced by a constant (zero) statistic.
    marginal_mae = float(np.mean(np.abs(y_valid - y_train.mean())))
    return {
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "dict_branch_soup_mae": float(result["top5_soup_valid_mae"]),
        "dict_branch_best_mae": float(result["best_valid_mae"]),
        "marginal_only_mae": marginal_mae,
        "W_norm": float(result["W_soup_norm"]),
        "pass": bool(result["top5_soup_valid_mae"] < 0.25 * max(marginal_mae, 1e-9)),
    }


def _synthetic_negative() -> dict[str, Any]:
    """Label depends only on the topology marginal; placement randomised."""
    D, _Drand, _pca = load_dictionary()
    rng = np.random.default_rng(7)
    hetero = 1
    examples: list[r2.MoleculeFeatures] = []
    for n in range(4, 9):
        path = [(i, i + 1) for i in range(n - 1)]
        for _ in range(2):
            atoms = [0] * n
            atoms[int(rng.integers(0, n))] = hetero
            y = float(n)  # marginal-only target (does not depend on placement)
            examples.append(_toy_molecule(n, path, atoms, y))
    train = [m for i, m in enumerate(examples) if i % 2 == 0]
    valid = [m for i, m in enumerate(examples) if i % 2 == 1]
    codes_train = _codes(train, D)
    codes_valid = _codes(valid, D)
    c_train = sdb.binding_matrices_for_molecules(codes_train, train)
    c_valid = sdb.binding_matrices_for_molecules(codes_valid, valid)
    scale, mask = sdb.fit_rms_scaler(c_train)
    protocol = dict(shead.OPTIMIZED_PROTOCOL)
    y_train = np.asarray([m.y for m in train], dtype=np.float64)
    y_valid = np.asarray([m.y for m in valid], dtype=np.float64)
    base_train = np.zeros(len(train), dtype=np.float64)
    base_valid = np.zeros(len(valid), dtype=np.float64)
    result = sdb.train_linear_residual(
        sdb.apply_rms_scaler(c_train, scale, mask), base_train, y_train,
        sdb.apply_rms_scaler(c_valid, scale, mask), base_valid, y_valid,
        seed=0, protocol=protocol, width=int(sdb.K_ATOMS),
    )
    return {
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "dict_branch_soup_mae": float(result["top5_soup_valid_mae"]),
        "mean_baseline_mae": float(np.mean(np.abs(y_valid - y_train.mean()))),
        "W_norm": float(result["W_soup_norm"]),
    }


def synthetic_controls() -> dict[str, Any]:
    positive = _synthetic_positive()
    negative = _synthetic_negative()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "positive_assignment_only": positive,
        "negative_marginal_only": negative,
        "all_pass": bool(positive["pass"]),
        "official_test_loaded": False,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(SYNTHETIC_JSON, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)
    return payload


# ---------------------------------------------------------------------------
# Stage 2 — FSAR mechanism-preservation gate
# ---------------------------------------------------------------------------


def _train_dense_joint(
    c_phi_train: np.ndarray,
    base_train: np.ndarray,
    y_train: np.ndarray,
    c_phi_valid: np.ndarray,
    base_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int,
    protocol: Mapping[str, Any],
    *,
    width: int = sdb.K_ATOMS,
    categories: int = sdb.ATOM_CATEGORIES,
) -> dict[str, Any]:
    """Learned 32-D coordinate control: ``C_z = W^T C_phi`` with ``W`` trained."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    phi_dim = int(c_phi_train.shape[1])
    W = torch.as_tensor(
        sdb.random_normalized_dictionary(phi_dim, width, int(seed) + 17), dtype=torch.float64
    ).clone()
    W.requires_grad_(True)
    readout = torch.zeros(width, categories, dtype=torch.float64)
    readout.requires_grad_(True)
    c_train = torch.as_tensor(np.asarray(c_phi_train), dtype=torch.float64)
    c_valid = torch.as_tensor(np.asarray(c_phi_valid), dtype=torch.float64)
    base_train_t = torch.as_tensor(np.asarray(base_train), dtype=torch.float64)
    base_valid_t = torch.as_tensor(np.asarray(base_valid), dtype=torch.float64)
    residual = torch.as_tensor(np.asarray(y_train) - np.asarray(base_train), dtype=torch.float64)
    y_valid_t = torch.as_tensor(np.asarray(y_valid), dtype=torch.float64)

    with torch.no_grad():
        z_train0 = torch.einsum("blj,lk->bkj", c_train, W)
        scale = z_train0.pow(2).mean(dim=0).sqrt().clamp_min(r2.SCALER_FLOOR)
        mask = (z_train0.pow(2).mean(dim=0) > r2.SCALER_FLOOR ** 2).to(torch.float64)

    def _z(c_phi: "torch.Tensor") -> "torch.Tensor":
        return torch.einsum("blj,lk->bkj", c_phi, W) / scale[None, :, :] * mask[None, :, :]

    optimizer = torch.optim.Adam(
        [W, readout],
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    n = int(c_train.shape[0])
    batch_size = int(protocol["batch_size"])
    max_epochs = int(protocol["max_epochs"])
    patience = int(protocol["patience"])
    clip = float(protocol["gradient_clip_norm"])
    generator = torch.Generator().manual_seed(int(seed) + int(protocol.get("train_shuffle_seed_offset", 0)))

    def _valid_mae() -> float:
        prediction = base_valid_t + (_z(c_valid) * readout).sum(dim=(1, 2))
        return float((prediction - y_valid_t).abs().mean())

    best_mae = float("inf")
    best_epoch = 1
    best_state = None
    stale = 0
    top5: list[tuple[float, int, dict[str, "torch.Tensor"]]] = []
    for epoch in range(1, max_epochs + 1):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            index = order[start : start + batch_size]
            prediction = (_z(c_train[index]) * readout).sum(dim=(1, 2))
            loss = torch.nn.functional.l1_loss(prediction, residual[index])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W, readout], clip)
            optimizer.step()
        valid_mae = _valid_mae()
        state = {"W": W.detach().clone(), "readout": readout.detach().clone()}
        top5.append((float(valid_mae), int(epoch), state))
        top5.sort(key=lambda item: (item[0], item[1]))
        top5 = top5[:5]
        if valid_mae < best_mae:
            best_mae = float(valid_mae)
            best_epoch = int(epoch)
            best_state = state
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    assert best_state is not None
    soup = {
        "W": torch.stack([item[2]["W"] for item in top5], dim=0).mean(dim=0),
        "readout": torch.stack([item[2]["readout"] for item in top5], dim=0).mean(dim=0),
    }

    def _mae_with(state: Mapping[str, "torch.Tensor"]) -> float:
        z = torch.einsum("blj,lk->bkj", c_valid, state["W"]) / scale[None, :, :] * mask[None, :, :]
        prediction = base_valid_t + (z * state["readout"]).sum(dim=(1, 2))
        return float((prediction - y_valid_t).abs().mean())

    return {
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "top5_epochs": [int(item[1]) for item in top5],
        "top5_soup_valid_mae": float(_mae_with(soup)),
        "W_soup_norm": float(soup["W"].norm()),
        "trainable_params": int(W.numel() + readout.numel()),
        "state": soup,
    }


def stage2(seeds: Sequence[int] = (0, 1, 2), force: bool = False) -> dict[str, Any]:
    if STAGE2_JSON.exists() and not force:
        return _read_json(STAGE2_JSON)
    stage1_payload = stage1()
    if stage1_payload["verdict"] == "STOP":
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "verdict": "SKIPPED_STAGE1_STOP",
            "official_test_loaded": False,
        }
        _write_json(STAGE2_JSON, payload)
        return payload

    train, valid, scalers, _meta = build_datasets()
    D_ksvd, D_rand, _pca = load_dictionary()
    protocol = dict(shead.OPTIMIZED_PROTOCOL)

    train_arrays = mech._node_arrays(train, scalers)
    valid_arrays = mech._node_arrays(valid, scalers)
    c_phi_train = train_arrays.C_raw
    c_phi_valid = valid_arrays.C_raw
    y_train = train_arrays.y
    y_valid = valid_arrays.y

    codes_ksvd_train = _codes(train, D_ksvd)
    codes_ksvd_valid = _codes(valid, D_ksvd)
    c_d_train = sdb.binding_matrices_for_molecules(codes_ksvd_train, train)
    c_d_valid = sdb.binding_matrices_for_molecules(codes_ksvd_valid, valid)
    scale_d, mask_d = sdb.fit_rms_scaler(c_d_train)
    stat_d_train = sdb.apply_rms_scaler(c_d_train, scale_d, mask_d)
    stat_d_valid = sdb.apply_rms_scaler(c_d_valid, scale_d, mask_d)

    codes_rand_train = _codes(train, D_rand)
    codes_rand_valid = _codes(valid, D_rand)
    c_r_train = sdb.binding_matrices_for_molecules(codes_rand_train, train)
    c_r_valid = sdb.binding_matrices_for_molecules(codes_rand_valid, valid)
    scale_r, mask_r = sdb.fit_rms_scaler(c_r_train)
    stat_r_train = sdb.apply_rms_scaler(c_r_train, scale_r, mask_r)
    stat_r_valid = sdb.apply_rms_scaler(c_r_valid, scale_r, mask_r)

    # frozen random dense projection control: C_z = W^T C_phi, W fixed normalised.
    W_frozen = sdb.random_normalized_dictionary(int(r2.PHI_DIM), int(sdb.K_ATOMS), int(sdb.DICT_SEED) + 3)
    c_zf_train = np.einsum("blj,lk->bkj", c_phi_train, W_frozen)
    c_zf_valid = np.einsum("blj,lk->bkj", c_phi_valid, W_frozen)
    scale_zf, mask_zf = sdb.fit_rms_scaler(c_zf_train)
    stat_zf_train = sdb.apply_rms_scaler(c_zf_train, scale_zf, mask_zf)
    stat_zf_valid = sdb.apply_rms_scaler(c_zf_valid, scale_zf, mask_zf)

    h5 = _read_json(mech.MECH_DIR / "h5_frozen_m0.json")
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        seed = int(seed)
        m0_train = mech._batched_prediction("M0", seed, train, scalers, base_only=True)["base"]
        m0_valid = mech._batched_prediction("M0", seed, valid, scalers, base_only=True)["base"]
        m0_valid_mae = float(np.mean(np.abs(y_valid - m0_valid)))
        m0_valid_mae_h5 = float(h5["per_seed"][str(seed)]["M0_valid_mae"])

        def _arm(stat_train, stat_valid, width):
            return sdb.train_linear_residual(
                stat_train, m0_train, y_train, stat_valid, m0_valid, y_valid,
                seed=seed, protocol=protocol, width=int(width),
            )

        arm_phi = _arm(train_arrays.C_tilde, valid_arrays.C_tilde, r2.PHI_DIM)
        arm_dict = _arm(stat_d_train, stat_d_valid, sdb.K_ATOMS)
        arm_rand = _arm(stat_r_train, stat_r_valid, sdb.K_ATOMS)
        arm_dense = _train_dense_joint(
            c_phi_train, m0_train, y_train, c_phi_valid, m0_valid, y_valid, seed, protocol,
        )
        arm_dense_frozen = _arm(stat_zf_train, stat_zf_valid, sdb.K_ATOMS)

        # assignment-shuffle mechanism probe on the trained dict branch.
        shuffled_codes = sdb.shuffle_codes_against_chemistry(codes_ksvd_valid, valid, seed=seed + 991)
        c_d_shuf = sdb.binding_matrices_for_molecules(shuffled_codes, valid)
        stat_d_shuf = sdb.apply_rms_scaler(c_d_shuf, scale_d, mask_d)
        weight = arm_dict["W_soup"]
        pred_clean = m0_valid + np.einsum("bkj,kj->b", stat_d_valid, weight.numpy())
        pred_shuf = m0_valid + np.einsum("bkj,kj->b", stat_d_shuf, weight.numpy())
        mae_clean = float(np.mean(np.abs(y_valid - pred_clean)))
        mae_shuf = float(np.mean(np.abs(y_valid - pred_shuf)))
        branch_std = float(np.std(pred_clean - m0_valid))

        m_phi = float(arm_phi["top5_soup_valid_mae"])
        m_dict = float(arm_dict["top5_soup_valid_mae"])
        m_dense = float(arm_dense["top5_soup_valid_mae"])
        m_dense_frozen = float(arm_dense_frozen["top5_soup_valid_mae"])
        m_rand = float(arm_rand["top5_soup_valid_mae"])
        denom = m0_valid_mae - m_phi
        recovery = float((m0_valid_mae - m_dict) / denom) if denom > 0 else float("nan")
        per_seed[str(seed)] = {
            "M0_valid_mae": m0_valid_mae,
            "M0_valid_mae_h5": m0_valid_mae_h5,
            "protocol_drift_M0": float(abs(m0_valid_mae - m0_valid_mae_h5)),
            "phi65_soup_valid_mae": m_phi,
            "h5_phi65_soup_valid_mae": float(h5["per_seed"][str(seed)]["frozen_M0_plus_B_soup_valid_mae"]),
            "phi65_protocol_drift": float(abs(m_phi - float(h5["per_seed"][str(seed)]["frozen_M0_plus_B_soup_valid_mae"]))),
            "dict32_soup_valid_mae": m_dict,
            "rand32_soup_valid_mae": m_rand,
            "dense32_soup_valid_mae": m_dense,
            "dense32_frozen_soup_valid_mae": m_dense_frozen,
            "recovery_oracle": recovery,
            "dense_slack_dict_minus_dense": float(m_dict - m_dense),
            "dict_branch_std": branch_std,
            "shuffle_clean_mae": mae_clean,
            "shuffle_permuted_mae": mae_shuf,
            "shuffle_degradation": float(mae_shuf - mae_clean),
            "dict_trainable_params": int(arm_dict["trainable_params"]),
            "dense_trainable_params": int(arm_dense["trainable_params"]),
            "top5_epochs_dict": arm_dict["top5_epochs"],
        }
        print(f"[stage2 seed{seed}] recovery={recovery:.3f} dict={m_dict:.6f} phi65={m_phi:.6f} "
              f"dense={m_dense:.6f} dense_frozen={m_dense_frozen:.6f} rand={m_rand:.6f} "
              f"shuffle_deg={mae_shuf - mae_clean:.4f}", flush=True)

    recoveries = [row["recovery_oracle"] for row in per_seed.values()]
    mean_recovery = float(np.mean(recoveries))
    mean_shuffle = float(np.mean([row["shuffle_degradation"] for row in per_seed.values()]))
    alive = bool(all(row["dict_branch_std"] > 1e-3 for row in per_seed.values()))
    dense_ok = bool(all(row["dense_slack_dict_minus_dense"] <= STAGE2_DENSE_SLACK for row in per_seed.values()))
    drift = float(max(row["phi65_protocol_drift"] for row in per_seed.values()))
    if mean_recovery < STAGE2_RECOVERY_STOP:
        verdict = "STOP"
    elif (mean_recovery >= STAGE2_RECOVERY_PASS and mean_shuffle >= STAGE2_SHUFFLE_DEGRADE and alive
          and drift <= PROTOCOL_DRIFT_TOL):
        verdict = "PASS"
    else:
        verdict = "INSPECT"
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "verdict": verdict,
        "mean_recovery_oracle": mean_recovery,
        "mean_shuffle_degradation": mean_shuffle,
        "dict_branch_alive": alive,
        "dict_within_dense_slack": dense_ok,
        "phi65_protocol_drift_max": drift,
        "protocol_drift_tolerance": PROTOCOL_DRIFT_TOL,
        "thresholds": {
            "recovery_pass": STAGE2_RECOVERY_PASS,
            "recovery_stop": STAGE2_RECOVERY_STOP,
            "dense_slack": STAGE2_DENSE_SLACK,
            "shuffle_degrade": STAGE2_SHUFFLE_DEGRADE,
        },
        "per_seed": per_seed,
        "reused_artifacts": [
            "fsar_r2_ar0 cache",
            "r2ar0_m0_seed{0,1,2}_top5_soup.pt",
            "h5_frozen_m0.json",
        ],
        "baselines_not_rerun": ["FSAR MB/MM", "strict-static S0", "TCCD", "DTX", "SDPK", "SRDA"],
        "official_test_loaded": False,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(STAGE2_JSON, payload)
    _write_stage2_markdown(payload)
    return payload


def _write_stage2_markdown(payload: Mapping[str, Any]) -> None:
    if payload.get("verdict") == "SKIPPED_STAGE1_STOP":
        STAGE2_MD.write_text("# SDB-v0 Stage 2 — SKIPPED (Stage 1 STOP)\n", encoding="utf-8")
        return
    lines = ["# SDB-v0 Stage 2 — FSAR mechanism-preservation gate\n"]
    lines.append(f"Verdict: **{payload['verdict']}**\n")
    lines.append(f"mean recovery (vs frozen phi65 oracle): {_fmt(payload['mean_recovery_oracle'])}")
    lines.append(f"\nmean assignment-shuffle MAE degradation: {_fmt(payload['mean_shuffle_degradation'])}")
    lines.append(f"\ndict within dense slack: {payload['dict_within_dense_slack']}; alive: {payload['dict_branch_alive']}")
    lines.append(f"\nphi65 protocol drift (max): {_fmt(payload['phi65_protocol_drift_max'])}\n")
    lines.append("| seed | M0 | phi65 | dict32 | dense32 | dense32_frozen | rand32 | recovery | shuffle_deg |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for seed, row in payload["per_seed"].items():
        lines.append(
            f"| {seed} | {_fmt(row['M0_valid_mae'])} | {_fmt(row['phi65_soup_valid_mae'])} "
            f"| {_fmt(row['dict32_soup_valid_mae'])} | {_fmt(row['dense32_soup_valid_mae'])} "
            f"| {_fmt(row['dense32_frozen_soup_valid_mae'])} | {_fmt(row['rand32_soup_valid_mae'])} "
            f"| {_fmt(row['recovery_oracle'])} | {_fmt(row['shuffle_degradation'])} |"
        )
    lines.append("\n*All arms are a frozen `M0` soup plus a single linear assignment residual. "
                 "`dict32` total params = 65x32 (frozen D) + 32x28 (readout); `dense32` = 65x32 (trained W) + 32x28.*\n")
    STAGE2_MD.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# report / CLI
# ---------------------------------------------------------------------------


def report() -> dict[str, Any]:
    payload: dict[str, Any] = {"protocol_version": PROTOCOL_VERSION, "git_commit": _git_commit()}
    if STAGE1_JSON.exists():
        payload["stage1"] = _read_json(STAGE1_JSON)
    if STAGE2_JSON.exists():
        payload["stage2"] = _read_json(STAGE2_JSON)
    if SANITY_JSON.exists():
        payload["sanity"] = _read_json(SANITY_JSON)
    if SYNTHETIC_JSON.exists():
        payload["synthetic"] = _read_json(SYNTHETIC_JSON)
    _write_json(DECISION_JSON, payload)
    print(json.dumps({k: (v.get("verdict") if isinstance(v, dict) else None) for k, v in payload.items()}, indent=2))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SDB-v0 runner")
    parser.add_argument("stage", choices=["dict", "gate1", "sanity", "synthetic", "gate2", "report", "all"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-fit-atoms", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "dict":
        fit_dictionary(epochs=args.epochs, max_fit_atoms=args.max_fit_atoms, force=args.force)
    elif args.stage == "gate1":
        stage1(force=args.force)
    elif args.stage == "sanity":
        sanity()
    elif args.stage == "synthetic":
        synthetic_controls()
    elif args.stage == "gate2":
        stage2(seeds=tuple(args.seeds), force=args.force)
    elif args.stage == "report":
        report()
    elif args.stage == "all":
        sanity()
        synthetic_controls()
        if stage1(force=args.force)["verdict"] != "STOP":
            stage2(seeds=tuple(args.seeds), force=args.force)
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
