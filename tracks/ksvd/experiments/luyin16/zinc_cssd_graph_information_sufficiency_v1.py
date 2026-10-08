"""ZINC CSSD graph information sufficiency v1.

Round: with the frozen CSSD dictionary and the frozen DICT consumers of
``zinc_cssd_consumer_replacement_v1`` (DICT_s0 / DICT_s1) held fixed, does the
graph-level aggregation ``R_G`` (the reader input, verified 814D) discard
task-relevant information about the three-way binding between CSSD
dictionary-atom supports and the REAL radius-2 patch co-coverage of physical
atoms?

Witness (frozen before any labelled scoring; see
``tracks/ksvd/notes/zinc_cssd_graph_information_sufficiency_v1.md`` and
``tracks/ksvd/protocols/zinc-cssd-graph-information-sufficiency-v1.yaml``):

* ``alpha`` = the deployed CSSD operator math on raw ``dict_phi`` rows
  (``tied_iht_codes(Dbar, phi - (phi@U)@U^T, s=8, steps=10)``; frozen basis);
  ``b[v,k] = 1[alpha[v,k] != 0]``;
* ``C(x)`` = roots whose radius-2 patch covers the REAL original atom ``x``
  (from the env cache ``env_occ_node`` / ``env_occ_root``; root row == original
  atom row per molecule);
* ``n[x,k] = sum_{v in C(x)} b[v,k]``;
* ``W_k = sum_x C(n[x,k],3) / (sum_x C(|C(x)|,3) + 1e-9)`` in R^32.
* SHAM: support rows permuted WITHIN each molecule (seed 20261021 + gid) with
  the real incidence untouched — every molecule-level marginal (per-atom-k
  usage counts, |C(x)|, denominator) preserved, only the support—coverage
  binding destroyed.

The witness / ``R_G`` export stages are label-free (they never import
targets); labelled scoring happens only in the ``heads`` stage.  The deployed
model's ``DeployFull.code()`` returns a zero placeholder, so ``aux["coord"]``
is NEVER used as alpha; alpha always comes from the frozen basis + phi rows and
is checked against the deployed ``cssd_decode`` path.

Stages (``--stage``):
* ``witness-export``  label-free W_real/W_shuf + diagnostics for all 10000 rows
* ``restore-checks``  frozen-run restore, hashes, prediction replay, alpha
                      operator equivalence, batch/relabel invariances
* ``rg-export``       capture the true reader input R (814D) for fit+dev, both
                      seeds (label-free)
* ``heads``           the frozen 3-arm residual-head protocol (labelled)
* ``report``          assemble the report manifest from the artifacts

Official valid/test are never instantiated anywhere in this module.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import zinc_cssd_basis_reuse_v1 as zreuse
from tracks.ksvd.experiments.luyin16 import zinc_cssd_consumer_replacement_v1 as repl
from tracks.ksvd.experiments.luyin16 import zinc_cssd_nonlinear_binding_v1 as parent
from tracks.ksvd.experiments.luyin16 import zinc_full_cycle_target_decomposition_v1 as zftd
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1 as prev,
)

PROTOCOL_VERSION = "zinc-cssd-graph-information-sufficiency-v1"
RESULT_SLUG = "zinc_cssd_graph_information_sufficiency_v1"
TRACK_ROOT = repl.TRACK_ROOT
RESULTS_DIR = TRACK_ROOT / "results" / RESULT_SLUG
SOURCE_DIR = repl.SOURCE_DIR
SMILES_TABLE = repl.SMILES_TABLE

# ---- frozen CSSD constants (source round) ---------------------------------
CSSD_Q = repl.CSSD_Q                    # 1
CSSD_K_ATOMS = repl.CSSD_K_ATOMS        # 32
CSSD_SPARSITY = repl.CSSD_SPARSITY      # 8
CSSD_IHT_STEPS = repl.CSSD_IHT_STEPS    # 10
W_DIM = int(CSSD_K_ATOMS)
R_DIM = 814                             # verified reader input width of the frozen consumer

# ---- frozen witness constants ---------------------------------------------
WITNESS_EPS = 1e-9
SHUFFLE_SEED_BASE = 20261021

# ---- frozen head protocol --------------------------------------------------
HEAD_HIDDEN = (13, 13)
HEAD_LR = 1e-3
HEAD_MAX_EPOCHS = 1500
HEAD_PATIENCE = 300
HEAD_INIT_SEED = 20261021
SEL_FRACTION = 0.2
N_OOF_FOLDS = 5
OOF_SEED = 20261021
N_BOOT = 2000
BOOT_SEED = 20261021
SEEDS = (0, 1)
ARMS = ("R_only", "R_W_shuffled", "R_W_real")
REPLAY_TOL = 1e-4

# frozen decision thresholds (never moved after seeing labels)
GO_DG_R = 0.0010
GO_DG_SHAM = 0.0005
DECISION_SCALE = 0.003
MECHANISM_GATE_L1 = 0.1
MECHANISM_GATE_FRACTION = 0.95
TOP_SHARE_GATE = 0.50

write_json = parent.write_json
read_json = parent.read_json
file_sha256 = parent.file_sha256
array_sha256 = parent.array_sha256
state_hash = parent.state_hash
resolve_device = parent.resolve_device


# ---------------------------------------------------------------------------
# pure witness math (no data / model dependencies; synthetic-testable)
# ---------------------------------------------------------------------------


def choose3(counts: np.ndarray) -> np.ndarray:
    """``C(m,3) = m(m-1)(m-2)/6`` elementwise on a non-negative int array."""
    m = np.asarray(counts, dtype=np.int64)
    out = (m * (m - 1) * (m - 2)) // 6
    return np.where(m >= 0, out, 0)


def witness_from_supports(
    b: np.ndarray,
    occ_node: np.ndarray,
    occ_root: np.ndarray,
    n_roots: int,
    eps: float = WITNESS_EPS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``W_k = sum_x C(n[x,k],3) / (sum_x C(|C(x)|,3) + eps)``.

    ``b`` is ``[n_roots, K]`` binary supports; ``occ_node``/``occ_root`` are the
    real patch-incidence pairs (both index rows 0..n_roots-1).  Zero-denominator
    molecules safely return the zero vector with the denominator recorded.
    """
    b = np.asarray(b)
    if b.ndim != 2 or b.shape[0] != int(n_roots):
        raise RuntimeError(f"support shape {b.shape} incompatible with n_roots={n_roots}")
    occ_node = np.asarray(occ_node, dtype=np.int64)
    occ_root = np.asarray(occ_root, dtype=np.int64)
    if occ_node.shape != occ_root.shape:
        raise RuntimeError("occ_node/occ_root length mismatch")
    if occ_node.size and (occ_node.min() < 0 or occ_node.max() >= int(n_roots)
                          or occ_root.min() < 0 or occ_root.max() >= int(n_roots)):
        raise RuntimeError("occ indices out of range")
    n_x = np.bincount(occ_node, minlength=int(n_roots)).astype(np.int64)
    nk = np.zeros((int(n_roots), int(b.shape[1])), dtype=np.int64)
    if occ_node.size:
        np.add.at(nk, occ_node, b[occ_root].astype(np.int64))
    numerator = choose3(nk).sum(axis=0).astype(np.float64)
    denominator = float(choose3(n_x).sum())
    if denominator > 0:
        W = numerator / (denominator + eps)
    else:
        W = np.zeros(int(b.shape[1]), dtype=np.float64)
    diag = {
        "denominator": float(denominator),
        "n_atoms_ge3_coverage": int((n_x >= 3).sum()),
        "n_roots": int(n_roots),
        "n_occ": int(occ_node.size),
    }
    return W, diag


def sham_permutation(n_roots: int, gid: int) -> np.ndarray:
    """Fixed within-molecule support-row permutation (target-independent)."""
    rng = np.random.default_rng(int(SHUFFLE_SEED_BASE) + int(gid))
    return rng.permutation(int(n_roots)).astype(np.int64)


def sham_supports(b: np.ndarray, gid: int) -> tuple[np.ndarray, int]:
    """Permuted support rows + the fixed-point count of the permutation."""
    n = int(b.shape[0])
    perm = sham_permutation(n, gid)
    fixed = int((perm == np.arange(n)).sum())
    return np.ascontiguousarray(b[perm]), fixed


def alpha_from_phi(phi_rows: np.ndarray, basis_parts: Mapping[str, Any]) -> np.ndarray:
    """The deployed CSSD alpha (exactly ``cssd_decode``'s math, no model)."""
    U = basis_parts["U"]
    Dbar = basis_parts["Dbar"]
    phi = torch.as_tensor(np.asarray(phi_rows), dtype=torch.float32)
    c = phi @ U
    r = phi - c @ U.t()
    alpha = v0.tied_iht_codes(Dbar, r, s=CSSD_SPARSITY, steps=CSSD_IHT_STEPS)
    return alpha.numpy().astype(np.float64)


def molecule_witness(
    phi_rows: np.ndarray,
    occ_node: np.ndarray,
    occ_root: np.ndarray,
    gid: int,
    basis_parts: Mapping[str, Any],
) -> dict[str, Any]:
    """Real + sham witness and diagnostics for one molecule (label-free)."""
    alpha = alpha_from_phi(phi_rows, basis_parts)
    b = (alpha != 0).astype(np.int64)
    W_real, diag = witness_from_supports(b, occ_node, occ_root, int(phi_rows.shape[0]))
    b_shuf, n_fixed = sham_supports(b, gid)
    W_shuf, _ = witness_from_supports(b_shuf, occ_node, occ_root, int(phi_rows.shape[0]))
    return {
        "W_real": W_real,
        "W_shuf": W_shuf,
        "diag": {
            **diag,
            "gid": int(gid),
            "n_fixed_points": int(n_fixed),
            "alpha_nnz_mean": float((alpha != 0).sum(axis=1).mean()) if alpha.size else 0.0,
            "support_rows_distinct": int(len({row.tobytes() for row in b})) if b.shape[0] else 0,
            "l1_real_shuf": float(np.abs(W_real - W_shuf).sum()),
        },
    }


def remap_molecule(
    phi: np.ndarray,
    atom: np.ndarray,
    occ_node: np.ndarray,
    occ_root: np.ndarray,
    occ_shell: np.ndarray,
    perm: np.ndarray,
) -> dict[str, np.ndarray]:
    """Relabel a molecule's nodes: the new index of old node ``v`` is
    ``perm[v]`` (test helper).

    Every node-addressed field is remapped consistently: ``new_phi[perm[v]] =
    phi[v]`` (i.e. ``new_phi = phi[inv]`` with ``inv = perm^-1``) and every
    occurrence pair ``(root, node) -> (perm[root], perm[node])``, so any
    permutation-invariant statistic must be unchanged.
    """
    perm = np.asarray(perm, dtype=np.int64)
    n = int(np.asarray(phi).shape[0])
    if sorted(perm.tolist()) != list(range(n)):
        raise RuntimeError("perm is not a permutation")
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    old_occ_node = np.asarray(occ_node, dtype=np.int64)
    return {
        "phi": np.asarray(phi)[inv],
        "atom": np.asarray(atom)[inv],
        "occ_node": perm[old_occ_node],
        "occ_root": perm[np.asarray(occ_root, dtype=np.int64)],
        "occ_shell": np.asarray(occ_shell)[inv[old_occ_node]],
    }


# ---------------------------------------------------------------------------
# shared loaders (label-free)
# ---------------------------------------------------------------------------


def load_basis_parts() -> dict[str, Any]:
    ro = repl.load_round_objects()
    parts = repl.frozen_basis_parts(ro["basis"])
    parts["round_objects"] = ro
    return parts


def basis_only(parts: Mapping[str, Any]) -> dict[str, Any]:
    return {k: parts[k] for k in ("U", "common_rms", "Dbar")}


def smiles_of_row() -> np.ndarray:
    with np.load(SMILES_TABLE, allow_pickle=False) as z:
        return np.asarray(z["smiles"])


_node_size_cache: np.ndarray | None = None


def node_counts() -> np.ndarray:
    """Per-molecule node counts of the 10000 train rows (label-free)."""
    global _node_size_cache
    if _node_size_cache is None:
        _phi, _atom, node_sizes = prev._env_phi_atom()
        _node_size_cache = np.asarray(node_sizes, np.int64)
    return _node_size_cache


# ---------------------------------------------------------------------------
# stage: witness export (label-free)
# ---------------------------------------------------------------------------


def stage_witness_export(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    basis_parts = basis_only(load_basis_parts())
    train_data = zftd.load_train_only()
    if len(train_data) != 10000:
        raise RuntimeError("train-only cache length mismatch")
    # the row's global train index IS its position in the train-only cache
    # (build_prepared_data sets local_mol_id = the source fold index); verify
    # this identity on a sample of the committed fit fold before use.
    ro = repl.load_round_objects()
    pm, fit_data, _dev_data = repl.build_round_data(ro)
    fit_idx = np.asarray(ro["fit_idx"], np.int64)
    for pos in (0, 1, 2, len(fit_data) // 2, len(fit_data) - 1):
        if int(fit_data[pos].local_mol_id.reshape(-1)[0].item()) != int(fit_idx[pos]):
            raise RuntimeError("local_mol_id / train-row-position identity violated")
    W_real = np.zeros((10000, W_DIM), np.float64)
    W_shuf = np.zeros((10000, W_DIM), np.float64)
    diag_rows: list[dict[str, Any]] = []
    n_den_zero = 0
    for row_id, d in enumerate(train_data):
        phi = d.dict_phi.numpy().astype(np.float64)
        occ_node = d.env_occ_node.numpy()
        occ_root = d.env_occ_root.numpy()
        gid = row_id
        w = molecule_witness(phi, occ_node, occ_root, gid, basis_parts)
        W_real[row_id] = w["W_real"]
        W_shuf[row_id] = w["W_shuf"]
        diag_rows.append(w["diag"])
        if w["diag"]["denominator"] == 0:
            n_den_zero += 1
        if (row_id + 1) % 2000 == 0:
            log(f"[witness-export] {row_id + 1}/10000 rows")
    l1 = np.array([r["l1_real_shuf"] for r in diag_rows], np.float64)
    denom_pos = np.array([r["denominator"] > 0 for r in diag_rows], bool)
    frac_nonzero = float((l1[denom_pos] > 0).mean()) if denom_pos.any() else 0.0
    gate = {
        "mean_l1_real_shuf": float(l1.mean()),
        "median_l1_real_shuf": float(np.median(l1)),
        "p95_l1_real_shuf": float(np.percentile(l1, 95)),
        "fraction_nonzero_given_den_pos": frac_nonzero,
        "n_den_zero": int(n_den_zero),
        "passed": bool(l1.mean() >= MECHANISM_GATE_L1 and frac_nonzero >= MECHANISM_GATE_FRACTION),
    }
    np.savez_compressed(
        out_dir / "witness_export.npz",
        W_real=W_real, W_shuf=W_shuf,
        gid=np.array([r["gid"] for r in diag_rows], np.int64),
        row_id=np.arange(10000, dtype=np.int64),
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "witness-export",
        "label_free": True,
        "definitions": {
            "alpha": "tied_iht_codes(Dbar, phi - (phi@U)@U^T, s=8, steps=10), frozen basis, raw dict_phi",
            "support": "b[v,k] = 1[alpha[v,k] != 0]",
            "co_coverage": "C(x) = env_occ roots covering original atom x (unbatched; root row == atom row)",
            "witness": "W_k = sum_x C(n[x,k],3) / (sum_x C(|C(x)|,3) + 1e-9)",
            "sham": "support rows permuted within molecule, seed 20261021+gid, incidence untouched",
        },
        "diagnostics": {
            "mean_denominator": float(np.mean([r["denominator"] for r in diag_rows])),
            "mean_fixed_points": float(np.mean([r["n_fixed_points"] for r in diag_rows])),
            "alpha_nnz_mean": float(np.mean([r["alpha_nnz_mean"] for r in diag_rows])),
            "mean_support_rows_distinct": float(np.mean([r["support_rows_distinct"] for r in diag_rows])),
            "mean_W_real": W_real.mean(axis=0).tolist(),
            "std_W_real": W_real.std(axis=0).tolist(),
        },
        "mechanism_gate": gate,
        "seconds": float(time.perf_counter() - started),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    parts_full = load_basis_parts()
    manifest["basis_sha256"] = parts_full["hashes"]
    write_json(out_dir / "witness_export.json", manifest)
    log(f"[witness-export] gate passed={gate['passed']} mean_l1={gate['mean_l1_real_shuf']:.4f} "
        f"den_zero={n_den_zero} in {manifest['seconds']:.1f}s")
    return manifest


# ---------------------------------------------------------------------------
# stage: restore checks (label-free)
# ---------------------------------------------------------------------------


def _restore_model(
    arm: str, seed: int, *, basis_parts: Mapping[str, Any], payload, kappa: float
) -> tuple[nn.Module, Mapping[str, Any]]:
    model = repl.build_arm(arm, payload, kappa, basis_only(basis_parts), int(seed))
    run = repl.load_run(arm, seed, out_dir=repl.RESULTS_DIR)
    model.load_state_dict({k: v for k, v in run["soup_state"].items()}, strict=True)
    model.eval()
    return model, run


def _run_state_hashes() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for seed in SEEDS:
        run_dir = repl.RESULTS_DIR / "runs" / f"DICT_s{seed}"
        out[f"DICT_s{seed}"] = {
            "soup_state_hash": state_hash(torch.load(run_dir / "soup_state.pt", map_location="cpu", weights_only=False)),
            "init_state_hash": state_hash(torch.load(run_dir / "init_state.pt", map_location="cpu", weights_only=False)),
        }
    q_meta = read_json(SOURCE_DIR / "Q_meta.json")
    out["Q_soup_state_hash"] = state_hash(
        torch.load(SOURCE_DIR / "Q_soup_state.pt", map_location="cpu", weights_only=False)
    )
    out["Q_meta_soup_state_hash"] = q_meta["soup_state_sha256"]
    return out


def stage_restore_checks(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = load_basis_parts()
    ro = parts["round_objects"]
    pm, fit_data, dev_data = repl.build_round_data(ro)
    payload = prev.TuplePayload(ro["objects"]["payload_arrays"])
    kappa = float(ro["objects"]["kappa"]["kappa_M"])
    basis_parts = basis_only(parts)

    checks: dict[str, Any] = {}

    # (1) frozen-object hashes vs the frozen-round provenance manifests
    src = read_json(repl.RESULTS_DIR / "source_manifest.json")
    checks["basis_U_sha_matches_frozen_manifest"] = src["cssd_basis"]["U_sha256"] == parts["hashes"]["U"]
    checks["basis_D_sha_matches_frozen_manifest"] = src["cssd_basis"]["D_sha256"] == parts["hashes"]["D"]
    checks["basis_common_rms_sha_matches_frozen_manifest"] = (
        src["cssd_basis"]["common_rms_sha256"] == parts["hashes"]["common_rms"]
    )
    q_meta = read_json(SOURCE_DIR / "Q_meta.json")
    checks["q_soup_state_hash_matches_meta"] = state_hash(
        torch.load(SOURCE_DIR / "Q_soup_state.pt", map_location="cpu", weights_only=False)
    ) == q_meta["soup_state_sha256"]
    if not checks["q_soup_state_hash_matches_meta"]:
        raise RuntimeError("frozen Q soup state hash mismatch")

    # (2) alpha operator equivalence: frozen-basis math vs deployed cssd_decode
    model, _run = _restore_model("DICT", 0, basis_parts=parts, payload=payload, kappa=kappa)
    max_phi_hat_diff = 0.0
    for d in fit_data[:16]:
        phi = d.dict_phi.numpy().astype(np.float32)
        _z, phi_hat_ref, _rel = zreuse._root_codes(ro["basis"], phi)
        with torch.no_grad():
            phi_hat_dep = model.local_tuple.cssd_decode(torch.as_tensor(phi, dtype=torch.float32)).numpy()
        max_phi_hat_diff = max(max_phi_hat_diff, float(np.abs(phi_hat_ref.astype(np.float32) - phi_hat_dep).max()))
    checks["alpha_operator_equivalence_max_abs_phi_hat"] = max_phi_hat_diff
    if max_phi_hat_diff > 1e-6:
        raise RuntimeError(f"alpha operator mismatch {max_phi_hat_diff}")

    # (3) prediction replay vs the saved per-run npz (fit + dev, both seeds)
    replay: dict[str, Any] = {}
    for seed in SEEDS:
        model, _run = _restore_model("DICT", seed, basis_parts=parts, payload=payload, kappa=kappa)
        saved_fit = dict(np.load(repl.RESULTS_DIR / "runs" / f"DICT_s{seed}" / "fit_predictions.npz"))
        saved_dev = dict(np.load(repl.RESULTS_DIR / "runs" / f"DICT_s{seed}" / "dev_predictions.npz"))
        replay_fit = repl._predict_rows(model, fit_data, torch.device("cpu"))
        replay_dev = repl._predict_rows(model, dev_data, torch.device("cpu"))
        entry = {
            "fit_h_max_abs_diff": float(np.abs(replay_fit["h"] - saved_fit["h"]).max()),
            "dev_h_max_abs_diff": float(np.abs(replay_dev["h"] - saved_dev["h"]).max()),
            "fit_ell_max_abs_diff": float(np.abs(replay_fit["ell_hat"] - saved_fit["ell_hat"]).max()),
        }
        # the saved dev npz deliberately carries no per-row component predictions
        # (dev_predictions.npz = gid/y/y_raw/y_cal/h/q_raw); h (the g readout) is
        # the replay anchor on dev.
        if max(entry.values()) > REPLAY_TOL:
            raise RuntimeError(f"DICT_s{seed} replay mismatch {entry}")
        replay[f"s{seed}"] = entry
    checks["prediction_replay"] = replay

    # (4) witness batch invariance: per-molecule vs collated-batch computation
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1

    train_data = zftd.load_train_only()
    rows = [0, 5, 9]
    chunk = [train_data[i] for i in rows]
    single = [
        molecule_witness(
            d.dict_phi.numpy().astype(np.float64),
            d.env_occ_node.numpy(), d.env_occ_root.numpy(),
            int(i), basis_parts,
        )["W_real"]
        for i, d in zip(rows, chunk)
    ]
    batch = p1.env_collate(list(chunk))
    alpha_batch = alpha_from_phi(batch.dict_phi.numpy().astype(np.float64), basis_parts)
    sizes = [int(d.dict_phi.shape[0]) for d in chunk]
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    W_batched = []
    for j, d in enumerate(chunk):
        lo, hi = int(ptr[j]), int(ptr[j + 1])
        off = int(batch.ptr[j].item()) if hasattr(batch, "ptr") else lo
        occ_node_b = batch.env_occ_node.numpy() - off
        occ_root_b = batch.env_occ_root.numpy() - off
        keep = (occ_node_b >= 0) & (occ_node_b < hi - lo) & (occ_root_b >= 0) & (occ_root_b < hi - lo)
        b = (alpha_batch[lo:hi] != 0).astype(np.int64)
        W_b, _ = witness_from_supports(b, occ_node_b[keep], occ_root_b[keep], hi - lo)
        W_batched.append(W_b)
    batch_diff = float(max(np.abs(single[j] - W_batched[j]).max() for j in range(len(rows))))
    checks["witness_batch_invariance_max_abs"] = batch_diff
    if batch_diff > 1e-12:
        raise RuntimeError(f"witness batch invariance failed {batch_diff}")

    # (5) real-molecule node relabel invariance
    d = train_data[7]
    n = int(d.dict_phi.shape[0])
    perm = np.random.default_rng(777).permutation(n)
    rem = remap_molecule(
        d.dict_phi.numpy().astype(np.float64), d.dict_atom.numpy(),
        d.env_occ_node.numpy(), d.env_occ_root.numpy(), d.env_occ_shell.numpy(), perm,
    )
    gid = 7  # d = train_data[7]: the global train row index IS the list position
    w_orig = molecule_witness(d.dict_phi.numpy().astype(np.float64), d.env_occ_node.numpy(),
                              d.env_occ_root.numpy(), gid, basis_parts)
    w_relab = molecule_witness(rem["phi"], rem["occ_node"], rem["occ_root"], gid, basis_parts)
    relabel_diff = float(np.abs(w_orig["W_real"] - w_relab["W_real"]).max())
    checks["witness_relabel_invariance_max_abs"] = relabel_diff
    if relabel_diff > 1e-12:
        raise RuntimeError(f"witness relabel invariance failed {relabel_diff}")

    # (6) label isolation: the witness artifact carries no target fields
    checks["witness_export_fields"] = sorted(np.load(out_dir / "witness_export.npz").files)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "restore-checks",
        "label_free": True,
        "checks": checks,
        "run_state_hashes": _run_state_hashes(),
        "basis_sha256": parts["hashes"],
        "seconds": float(time.perf_counter() - started),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "restore_checks.json", manifest)
    log(f"[restore-checks] all checks passed in {manifest['seconds']:.1f}s")
    return manifest


# ---------------------------------------------------------------------------
# stage: R_G export (label-free)
# ---------------------------------------------------------------------------


class _ReaderInputHook:
    """Captures the TRUE reader input R_G (never aux['coord'])."""

    def __init__(self, model: nn.Module) -> None:
        self.features: list[np.ndarray] = []
        self.handle = model.reader.register_forward_pre_hook(self._hook)

    def _hook(self, _module: nn.Module, inputs: tuple) -> None:
        self.features.append(inputs[0].detach().cpu().numpy().astype(np.float64))

    def close(self) -> None:
        self.handle.remove()


@torch.no_grad()
def _forward_capture(model: nn.Module, rows: Sequence[Any], device: torch.device) -> dict[str, np.ndarray]:
    hook = _ReaderInputHook(model)
    hs, ells, ss = [], [], []
    model.eval()
    try:
        for start in range(0, len(rows), repl.BATCH_SIZE):
            chunk = list(rows[start:start + repl.BATCH_SIZE])
            batch = zftd.make_batch(chunk, list(range(len(chunk))), torch.zeros(len(chunk)), device)
            prediction = model(batch, mask=cm.C6_MASK)
            components = model.reader.components()
            hs.append(prediction.detach().cpu().numpy().astype(np.float64))
            ells.append(components[:, 0].detach().cpu().numpy().astype(np.float64))
            ss.append(components[:, 1].detach().cpu().numpy().astype(np.float64))
    finally:
        hook.close()
    R = np.concatenate(hook.features, axis=0)
    return {"R": R, "h": np.concatenate(hs), "ell_hat": np.concatenate(ells), "s_hat": np.concatenate(ss)}


def stage_rg_export(*, out_dir: Path = RESULTS_DIR, device_name: str = "cpu", log: Any = print) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    parts = load_basis_parts()
    ro = parts["round_objects"]
    pm, fit_data, dev_data = repl.build_round_data(ro)
    payload = prev.TuplePayload(ro["objects"]["payload_arrays"])
    kappa = float(ro["objects"]["kappa"]["kappa_M"])

    summary: dict[str, Any] = {}
    for seed in SEEDS:
        model, _run = _restore_model("DICT", seed, basis_parts=parts, payload=payload, kappa=kappa)
        model = model.to(device)
        cap_fit = _forward_capture(model, fit_data, device)
        cap_dev = _forward_capture(model, dev_data, device)
        saved_fit = dict(np.load(repl.RESULTS_DIR / "runs" / f"DICT_s{seed}" / "fit_predictions.npz"))
        saved_dev = dict(np.load(repl.RESULTS_DIR / "runs" / f"DICT_s{seed}" / "dev_predictions.npz"))
        replay = {
            "fit_h_max_abs_diff": float(np.abs(cap_fit["h"] - saved_fit["h"]).max()),
            "dev_h_max_abs_diff": float(np.abs(cap_dev["h"] - saved_dev["h"]).max()),
        }
        if max(replay.values()) > REPLAY_TOL:
            raise RuntimeError(f"DICT_s{seed} rg-export replay mismatch {replay}")
        n_fit_nodes = sum(int(d.dict_phi.shape[0]) for d in fit_data)
        n_dev_nodes = sum(int(d.dict_phi.shape[0]) for d in dev_data)
        if cap_fit["R"].shape != (n_fit_nodes, R_DIM) or cap_dev["R"].shape != (n_dev_nodes, R_DIM):
            raise RuntimeError(
                f"unexpected R shape fit={cap_fit['R'].shape} dev={cap_dev['R'].shape} "
                f"expected ({n_fit_nodes},{R_DIM})/({n_dev_nodes},{R_DIM})"
            )
        np.savez_compressed(
            out_dir / f"rg_export_s{seed}.npz",
            R_fit=cap_fit["R"].astype(np.float32),
            R_dev=cap_dev["R"].astype(np.float32),
            h_fit=cap_fit["h"], h_dev=cap_dev["h"],
            ell_fit=cap_fit["ell_hat"], ell_dev=cap_dev["ell_hat"],
            s_fit=cap_fit["s_hat"], s_dev=cap_dev["s_hat"],
        )
        summary[f"s{seed}"] = {
            "R_fit_shape": list(cap_fit["R"].shape),
            "R_dev_shape": list(cap_dev["R"].shape),
            "replay": replay,
        }
        log(f"[rg-export] seed {seed}: R fit {cap_fit['R'].shape} dev {cap_dev['R'].shape} replay {replay}")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "rg-export",
        "label_free": True,
        "capture_method": "forward_pre_hook on model.reader (the true R_G; never aux['coord'])",
        "reader_input_dim": R_DIM,
        "summary": summary,
        "seconds": float(time.perf_counter() - started),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "rg_export.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# stage: heads (the only labelled stage)
# ---------------------------------------------------------------------------


def build_head(input_dim: int, hidden: tuple[int, ...] = HEAD_HIDDEN, seed: int = HEAD_INIT_SEED) -> nn.Sequential:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for width in hidden:
            layers.append(nn.Linear(previous, int(width)))
            layers.append(nn.ReLU())
            previous = int(width)
        layers.append(nn.Linear(previous, 1))
        return nn.Sequential(*layers)


def head_param_count(input_dim: int) -> int:
    return int(sum(p.numel() for p in build_head(input_dim).parameters()))


def _standardize(x_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x_train.mean(dim=0)
    var = x_train.var(dim=0, unbiased=False)
    scale = torch.where(var > 0, var.sqrt(), torch.ones_like(var))
    return mean, scale


def train_residual_head(
    x: torch.Tensor,
    residual: torch.Tensor,
    x_sel: torch.Tensor,
    residual_sel: torch.Tensor,
) -> nn.Sequential:
    head = build_head(int(x.shape[1]))
    opt = torch.optim.Adam(head.parameters(), lr=HEAD_LR)
    best_mae = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    for _epoch in range(HEAD_MAX_EPOCHS):
        head.train()
        opt.zero_grad()
        loss = F.l1_loss(head(x).view(-1), residual)
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            sel_mae = float((head(x_sel).view(-1) - residual_sel).abs().mean())
        if sel_mae < best_mae - 1e-12:
            best_mae = sel_mae
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= HEAD_PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("selection never improved; head protocol degenerate")
    head.load_state_dict(best_state)
    return head


def grouped_folds(smiles: np.ndarray, n_folds: int = N_OOF_FOLDS, seed: int = OOF_SEED) -> np.ndarray:
    """Fixed canonical-SMILES-grouped fold assignment (per row)."""
    groups = sorted(set(str(s) for s in smiles))
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(groups))
    chunks = np.array_split(order, int(n_folds))
    fold_of_group: dict[str, int] = {}
    for f, chunk in enumerate(chunks):
        for gi in chunk:
            fold_of_group[groups[int(gi)]] = f
    return np.array([fold_of_group[str(s)] for s in smiles], dtype=np.int64)


def _selection_split(train_rows: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Grouped 20% selection carve-out of the given training rows."""
    train_groups = sorted(set(str(groups[i]) for i in train_rows))
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(train_groups))
    n_sel = max(1, int(round(SEL_FRACTION * len(train_groups))))
    sel_groups = {train_groups[int(g)] for g in order[:n_sel]}
    is_sel = np.array([str(groups[i]) in sel_groups for i in train_rows], bool)
    return train_rows[is_sel], train_rows[~is_sel]


def group_paired_bootstrap(
    diff_matrix: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict[str, Any]:
    """Group-paired bootstrap of per-seed MAE differences (rows aligned).

    ``diff_matrix`` is ``[n_rows, n_series]`` of per-row signed MAE
    contributions (|e_A| - |e_B|); each draw resamples canonical-SMILES groups
    once (shared across series), computes each series' resampled MAE
    difference, averages the series, and the CI is taken over draws.
    """
    diff_matrix = np.asarray(diff_matrix, np.float64)
    uniq: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        uniq.setdefault(str(g), []).append(i)
    group_lists = [np.asarray(v, np.int64) for v in uniq.values()]
    sums = np.stack([diff_matrix[idx].sum(axis=0) for idx in group_lists])  # [G, m]
    counts = np.array([len(idx) for idx in group_lists], np.float64)
    rng = np.random.default_rng(int(seed))
    G = len(group_lists)
    means = np.zeros((int(n_boot), diff_matrix.shape[1]), np.float64)
    for b_i in range(int(n_boot)):
        pick = rng.integers(0, G, size=G)
        w = np.bincount(pick, minlength=G).astype(np.float64)
        denom = float((w * counts).sum())
        means[b_i] = (w[:, None] * sums).sum(axis=0) / denom
    point = diff_matrix.mean(axis=0)
    seed_mean = means.mean(axis=1)
    lo, hi = np.percentile(seed_mean, [2.5, 97.5])
    return {
        "point_per_series": point.tolist(),
        "bootstrap_mean": float(seed_mean.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_gt_0": float((seed_mean > 0).mean()),
        "n_boot": int(n_boot),
        "n_groups": int(G),
    }


def stage_heads(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ro = repl.load_round_objects()
    fit_idx = np.asarray(ro["fit_idx"], np.int64)
    dev_idx = np.asarray(ro["dev_idx"], np.int64)
    wit = np.load(out_dir / "witness_export.npz")
    W_real_all, W_shuf_all = wit["W_real"], wit["W_shuf"]
    smiles_all = smiles_of_row()

    targets = np.load(SOURCE_DIR / "targets.npz")
    g_all = np.asarray(targets["g"], np.float64)
    k_all = np.asarray(targets["k"], np.int64)

    W_real_fit, W_real_dev = W_real_all[fit_idx], W_real_all[dev_idx]
    W_shuf_fit, W_shuf_dev = W_shuf_all[fit_idx], W_shuf_all[dev_idx]
    g_fit_true = g_all[fit_idx]
    smiles_fit = smiles_all[fit_idx]
    smiles_dev = smiles_all[dev_idx]
    fold_of_row = grouped_folds(smiles_fit)
    arm_block = {"R_only": None, "R_W_shuffled": W_shuf_fit, "R_W_real": W_real_fit}

    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        exp = np.load(out_dir / f"rg_export_s{seed}.npz")
        R_fit = exp["R_fit"].astype(np.float64)
        R_dev = exp["R_dev"].astype(np.float64)
        h_fit = np.asarray(exp["h_fit"], np.float64)
        h_dev = np.asarray(exp["h_dev"], np.float64)
        residual_fit = g_fit_true - h_fit
        x_R_fit = torch.as_tensor(R_fit, dtype=torch.float32)
        x_R_dev = torch.as_tensor(R_dev, dtype=torch.float32)

        oof_residual = {arm: np.zeros(len(fit_idx), np.float64) for arm in ARMS}
        for k in range(N_OOF_FOLDS):
            eval_rows = np.flatnonzero(fold_of_row == k)
            train_rows = np.flatnonzero(fold_of_row != k)
            sel_rows, fit_rows = _selection_split(train_rows, smiles_fit, OOF_SEED + 1 + k)
            mean, scale = _standardize(x_R_fit[fit_rows])
            for arm in ARMS:
                block = arm_block[arm]
                if block is None:
                    x_fit, x_eval = x_R_fit, x_R_fit[eval_rows]
                else:
                    w_fit = torch.as_tensor(block, dtype=torch.float32)
                    x_fit = torch.cat([x_R_fit, w_fit], dim=1)
                    x_eval = torch.cat([x_R_fit[eval_rows], w_fit[eval_rows]], dim=1)
                x_fit_n = (x_fit - mean) / scale
                head = train_residual_head(
                    x_fit_n[fit_rows], torch.as_tensor(residual_fit[fit_rows], np.float32),
                    x_fit_n[sel_rows], torch.as_tensor(residual_fit[sel_rows], np.float32),
                )
                with torch.no_grad():
                    oof_residual[arm][eval_rows] = head(x_eval).view(-1).numpy().astype(np.float64)
            log(f"[heads] seed {seed} fold {k + 1}/{N_OOF_FOLDS} done")

        # dev heads: trained on ALL fit rows (same grouped selection protocol)
        sel_rows, fit_rows = _selection_split(np.arange(len(fit_idx)), smiles_fit, OOF_SEED + 100)
        mean, scale = _standardize(x_R_fit[fit_rows])
        dev_residual: dict[str, np.ndarray] = {}
        dev_blocks = {"R_only": None, "R_W_shuffled": W_shuf_dev, "R_W_real": W_real_dev}
        for arm in ARMS:
            block = arm_block[arm]
            if block is None:
                x_fit = x_R_fit
            else:
                x_fit = torch.cat([x_R_fit, torch.as_tensor(block, dtype=torch.float32)], dim=1)
            x_fit_n = (x_fit - mean) / scale
            head = train_residual_head(
                x_fit_n[fit_rows], torch.as_tensor(residual_fit[fit_rows], np.float32),
                x_fit_n[sel_rows], torch.as_tensor(residual_fit[sel_rows], np.float32),
            )
            dev_block = dev_blocks[arm]
            if dev_block is None:
                x_dev = x_R_dev
            else:
                x_dev = torch.cat([x_R_dev, torch.as_tensor(dev_block, dtype=torch.float32)], dim=1)
            with torch.no_grad():
                dev_residual[arm] = head((x_dev - mean) / scale).view(-1).numpy().astype(np.float64)

        per_seed[f"s{seed}"] = {
            "oof_residual": oof_residual,
            "dev_residual": {arm: dev_residual[arm].tolist() for arm in ARMS},
            "h_fit": h_fit.tolist(),
            "h_dev": h_dev.tolist(),
        }
        log(f"[heads] seed {seed}: OOF + dev heads done in {time.perf_counter() - started:.1f}s")

    scoring = _score(
        per_seed, fit_idx, dev_idx,
        g_fit_true=g_fit_true,
        k_all=k_all, smiles_fit=smiles_fit, smiles_dev=smiles_dev,
        out_dir=out_dir,
        log=log,
    )
    results = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "heads",
        "arm_param_counts": {arm: head_param_count(814 if arm == "R_only" else 846) for arm in ARMS},
        "folds": {"n_oof_folds": N_OOF_FOLDS, "oof_seed": OOF_SEED, "grouping": "canonical SMILES"},
        "scoring": scoring,
        "per_seed_row_residuals_saved": True,
        "seconds": float(time.perf_counter() - started),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    np.savez_compressed(
        out_dir / "heads_residuals.npz",
        **{
            f"oof_residual_s{seed}_{arm}": per_seed[f"s{seed}"]["oof_residual"][arm]
            for seed in SEEDS for arm in ARMS
        },
        **{
            f"dev_residual_s{seed}_{arm}": np.asarray(per_seed[f"s{seed}"]["dev_residual"][arm], np.float64)
            for seed in SEEDS for arm in ARMS
        },
        g_fit_true=g_fit_true,
        g_dev_true=g_all[dev_idx],
        fit_idx=fit_idx, dev_idx=dev_idx,
    )
    write_json(out_dir / "heads_results.json", results)
    return results


def _score(
    per_seed: Mapping[str, Any],
    fit_idx: np.ndarray,
    dev_idx: np.ndarray,
    *,
    g_fit_true: np.ndarray,
    k_all: np.ndarray,
    smiles_fit: np.ndarray,
    smiles_dev: np.ndarray,
    out_dir: Path = RESULTS_DIR,
    log: Any = print,
) -> dict[str, Any]:
    dev_idx = np.asarray(dev_idx, np.int64)
    targets = np.load(SOURCE_DIR / "targets.npz")
    g_dev_true = np.asarray(targets["g"], np.float64)[dev_idx]
    y_dev_true = np.asarray(targets["y"], np.float64)[dev_idx]
    q_devs = {
        seed: np.asarray(np.load(repl.RESULTS_DIR / "runs" / f"DICT_s{seed}" / "dev_predictions.npz")["q_raw"], np.float64)
        for seed in SEEDS
    }
    sizes = node_counts()
    sizes_fit, sizes_dev = sizes[fit_idx], sizes[dev_idx]
    k_fit, k_dev = k_all[fit_idx], k_all[dev_idx]

    def mae(err: np.ndarray) -> float:
        return float(np.abs(err).mean())

    def strata(err: np.ndarray, k: np.ndarray, node_sizes: np.ndarray) -> dict[str, Any]:
        k0 = k == 0
        t = np.quantile(node_sizes, [1 / 3, 2 / 3])
        small = node_sizes <= t[0]
        large = node_sizes > t[1]
        mid = ~(small | large)
        return {
            "k0_mae": mae(err[k0]), "k0_n": int(k0.sum()),
            "size_small_mae": mae(err[small]), "size_mid_mae": mae(err[mid]), "size_large_mae": mae(err[large]),
        }

    # ---- OOF contrasts (primary)
    n_fit = len(fit_idx)
    g_frozen_oof = np.stack([np.asarray(per_seed[f"s{s}"]["h_fit"], np.float64) for s in SEEDS])
    g_pred_oof = {
        arm: np.stack([
            g_frozen_oof[si] + np.asarray(per_seed[f"s{s}"]["oof_residual"][arm], np.float64)
            for si, s in enumerate(SEEDS)
        ])
        for arm in ARMS
    }
    abs_err = {arm: np.abs(g_pred_oof[arm] - g_fit_true[None, :]) for arm in ARMS}
    oof: dict[str, Any] = {
        "mae": {
            arm: {
                **{f"s{seed}": mae(abs_err[arm][si]) for si, seed in enumerate(SEEDS)},
                "two_seed_mean": float(abs_err[arm].mean()),
            }
            for arm in ARMS
        }
    }
    diff_R = abs_err["R_only"] - abs_err["R_W_real"]
    diff_sham = abs_err["R_W_shuffled"] - abs_err["R_W_real"]
    oof["deltas"] = {
        "dG_R": {
            "per_seed": {f"s{seed}": float(diff_R[si].mean()) for si, seed in enumerate(SEEDS)},
            "two_seed_mean": float(diff_R.mean()),
            "bootstrap": group_paired_bootstrap(diff_R.T, smiles_fit),
        },
        "dG_sham": {
            "per_seed": {f"s{seed}": float(diff_sham[si].mean()) for si, seed in enumerate(SEEDS)},
            "two_seed_mean": float(diff_sham.mean()),
            "bootstrap": group_paired_bootstrap(diff_sham.T, smiles_fit),
        },
    }
    per_mol_gain = diff_R.mean(axis=0)
    total_gain = float(per_mol_gain.sum())
    order = np.argsort(per_mol_gain)[::-1]
    oof["top20_gain_share"] = float(per_mol_gain[order[:20]].sum() / total_gain) if total_gain > 0 else None
    oof["strata_R_W_real_vs_R_only"] = {
        f"s{seed}": strata(abs_err["R_W_real"][si] - abs_err["R_only"][si], k_fit, sizes_fit)
        for si, seed in enumerate(SEEDS)
    }

    # ---- dev contrasts (exploratory secondary)
    g_pred_dev = {
        arm: np.stack([
            np.asarray(per_seed[f"s{s}"]["h_dev"], np.float64)
            + np.asarray(per_seed[f"s{s}"]["dev_residual"][arm], np.float64)
            for si, s in enumerate(SEEDS)
        ])
        for arm in ARMS
    }
    y_pred_dev = {
        arm: np.stack([g_pred_dev[arm][si] + q_devs[seed] for si, seed in enumerate(SEEDS)])
        for arm in ARMS
    }
    abs_dev_g = {arm: np.abs(g_pred_dev[arm] - g_dev_true[None, :]) for arm in ARMS}
    abs_dev_y = {arm: np.abs(y_pred_dev[arm] - y_dev_true[None, :]) for arm in ARMS}
    dev: dict[str, Any] = {
        "mae_g": {
            arm: {
                **{f"s{seed}": mae(abs_dev_g[arm][si]) for si, seed in enumerate(SEEDS)},
                "two_seed_mean": float(abs_dev_g[arm].mean()),
            }
            for arm in ARMS
        },
        "mae_y": {
            arm: {
                **{f"s{seed}": mae(abs_dev_y[arm][si]) for si, seed in enumerate(SEEDS)},
                "two_seed_mean": float(abs_dev_y[arm].mean()),
            }
            for arm in ARMS
        },
    }
    dev_diff_R = abs_dev_g["R_only"] - abs_dev_g["R_W_real"]
    dev_diff_sham = abs_dev_g["R_W_shuffled"] - abs_dev_g["R_W_real"]
    dev["deltas"] = {
        "dG_R": {
            "per_seed": {f"s{seed}": float(dev_diff_R[si].mean()) for si, seed in enumerate(SEEDS)},
            "two_seed_mean": float(dev_diff_R.mean()),
            "bootstrap": group_paired_bootstrap(dev_diff_R.T, smiles_dev),
        },
        "dG_sham": {
            "per_seed": {f"s{seed}": float(dev_diff_sham[si].mean()) for si, seed in enumerate(SEEDS)},
            "two_seed_mean": float(dev_diff_sham.mean()),
            "bootstrap": group_paired_bootstrap(dev_diff_sham.T, smiles_dev),
        },
    }
    dev["strata_R_W_real_vs_R_only"] = {
        f"s{seed}": strata(abs_dev_g["R_W_real"][si] - abs_dev_g["R_only"][si], k_dev, sizes_dev)
        for si, seed in enumerate(SEEDS)
    }

    # ---- frozen decision
    wit_manifest = read_json(Path(out_dir) / "witness_export.json")
    gate_passed = bool(wit_manifest.get("mechanism_gate", {}).get("passed", False))
    dG_R = oof["deltas"]["dG_R"]
    dG_sham = oof["deltas"]["dG_sham"]
    seeds_same_direction = bool(
        (dG_R["per_seed"]["s0"] > 0) == (dG_R["per_seed"]["s1"] > 0)
        and (dG_sham["per_seed"]["s0"] > 0) == (dG_sham["per_seed"]["s1"] > 0)
    )
    top_share = oof["top20_gain_share"]
    top_share_ok = top_share is not None and top_share < TOP_SHARE_GATE
    go = bool(
        gate_passed and seeds_same_direction
        and dG_R["two_seed_mean"] >= GO_DG_R
        and dG_sham["two_seed_mean"] >= GO_DG_SHAM
        and dG_sham["bootstrap"]["ci95_low"] > 0
        and top_share_ok
    )
    branch = "A" if go else ("C" if not gate_passed else "B")
    decision = {
        "mechanism_gate_passed": gate_passed,
        "seeds_same_direction": seeds_same_direction,
        "top20_gain_share": top_share,
        "top_share_ok": top_share_ok,
        "dG_R_two_seed_mean": dG_R["two_seed_mean"],
        "dG_sham_two_seed_mean": dG_sham["two_seed_mean"],
        "decision_scale_threshold": DECISION_SCALE,
        "branch": branch,
        "branch_meaning": {
            "A": "aggregation information gap supported (real witness beats R-only and matched sham)",
            "B": "no evidence the consumer misses this dictionary-patch statistic",
            "C": "round did not stand (mechanism/equivalence/restore failure)",
        }[branch],
    }
    log(f"[score] OOF dG_R {dG_R['two_seed_mean']:+.5f} dG_sham {dG_sham['two_seed_mean']:+.5f} "
        f"-> branch {branch}")
    return {"oof": oof, "dev": dev, "decision": decision}


# ---------------------------------------------------------------------------
# stage: report manifest
# ---------------------------------------------------------------------------


def stage_report(*, out_dir: Path = RESULTS_DIR, log: Any = print) -> dict[str, Any]:
    out_dir = Path(out_dir)
    heads = read_json(out_dir / "heads_results.json")
    wit = read_json(out_dir / "witness_export.json")
    restore = read_json(out_dir / "restore_checks.json")
    rg = read_json(out_dir / "rg_export.json")
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "report",
        "decision": heads["scoring"]["decision"],
        "oof_mae": heads["scoring"]["oof"]["mae"],
        "oof_deltas": heads["scoring"]["oof"]["deltas"],
        "oof_strata": heads["scoring"]["oof"]["strata_R_W_real_vs_R_only"],
        "oof_top20_gain_share": heads["scoring"]["oof"]["top20_gain_share"],
        "dev_mae_g": heads["scoring"]["dev"]["mae_g"],
        "dev_mae_y": heads["scoring"]["dev"]["mae_y"],
        "dev_deltas": heads["scoring"]["dev"]["deltas"],
        "mechanism_gate": wit["mechanism_gate"],
        "restore_checks": restore["checks"],
        "rg_summary": rg["summary"],
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(out_dir / "report_manifest.json", manifest)
    log(f"[report] branch {manifest['decision']['branch']}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("witness-export", "restore-checks", "rg-export", "heads", "report"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "witness-export":
        stage_witness_export(out_dir=out_dir)
    elif args.stage == "restore-checks":
        stage_restore_checks(out_dir=out_dir)
    elif args.stage == "rg-export":
        stage_rg_export(out_dir=out_dir, device_name=args.device)
    elif args.stage == "heads":
        stage_heads(out_dir=out_dir)
    elif args.stage == "report":
        stage_report(out_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
