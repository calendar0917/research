"""Sections 4 and 5: exact state replay and exact edit-record validation.

Section 4 verifies the equivalence rewrite ``F(G) = H(U(G), g(G))`` on a fixed
hash sample of molecules and the increment identity

``F(B) - F(A) = H(U(A) + dU, g(A) + dg) - H(U(A), g(A))``

where ``dU`` is **recomputed from the signed root-record multiset** of the two
real molecules and the target is the *independent* frozen full-Valid W3
prediction (not ``H(U(B))``, so the check cannot be satisfied by construction).

Section 5 verifies four structural claims:

1. atom renumbering does not change the root-record multiset (nor ``U``);
2. equal records give equal ``h`` and equal ``phi(h)``;
3. the signed record increment reproduces ``dU``;
4. node-count and graph-descriptor changes are accounted for.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

import ssa_common as C
import ssa_records as REC
import ssa_state as S


def _err(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return float(np.max(np.abs(d))), float(np.mean(np.abs(d)))


def hash_subset(ids: np.ndarray, *, cap: int, salt: str) -> np.ndarray:
    """Positions of the ``cap`` smallest fixed hashes, in ascending position order."""
    order = np.argsort(np.asarray([C.hash64(salt, int(x)) for x in ids.tolist()], np.uint64),
                       kind="stable")
    return np.asarray(sorted(order[: int(cap)].tolist()), np.int64)


def _signed_phi(machine: S.StateMachine, states: Mapping[str, Any], atoms: np.ndarray,
                signs: np.ndarray) -> tuple[np.ndarray, float]:
    """``sum_v sign_v phi(h_v)`` grouped by record class (one representative each)."""
    hi = states["rec_hi"][atoms]
    lo = states["rec_lo"][atoms]
    order = np.lexsort((lo, hi))
    ah, al, sg = hi[order], lo[order], signs[order]
    new = np.r_[True, (ah[1:] != ah[:-1]) | (al[1:] != al[:-1])]
    st = np.flatnonzero(new)
    w = np.add.reduceat(sg, st)
    rep = atoms[order][st]
    phi = machine.phi_atoms(np.asarray(states["h"][rep], np.float64), states["scores"][rep])
    spread = 0.0
    if st.size < sg.size:
        for j in range(st.size):
            lo_j = int(st[j])
            hi_j = int(st[j + 1]) if j + 1 < st.size else sg.size
            if hi_j - lo_j > 1:
                hh = np.asarray(states["h"][atoms[order][lo_j:hi_j]], np.float64)
                spread = max(spread, float(np.abs(hh - hh[0]).max()))
    return w @ phi, spread


def section4(machine: S.StateMachine, states: Mapping[str, Any], U: np.ndarray,
             frozen_pred_eV: np.ndarray) -> dict[str, Any]:
    mol_ids = np.asarray(states["mol_ids"], np.int64)
    pos = hash_subset(mol_ids, cap=C.MAX_REPLAY_MOLECULES, salt="replay")
    idx = pos
    out = machine.readout(U[idx], states["g"][idx])
    aux = states["aux"]
    n_atoms = np.diff(states["atom_off"])[idx]
    rec: dict[str, Any] = {
        "n_molecules": int(idx.size),
        "selection": (
            f"first {C.MAX_REPLAY_MOLECULES} sample molecules in fixed hash order; "
            "no label, prediction or error enters the choice"
        ),
        "errors": {},
    }
    checks = {
        "pred_eV_vs_batch_forward": (out["pred_eV"], states["pred_model_eV"][idx]),
        "pred_eV_vs_frozen_full_valid_pass": (out["pred_eV"], frozen_pred_eV[idx]),
        "logits_vs_batch_forward": (out["logits"], states["z_model"][idx]),
        "sum_f_h": (out["contrib"], np.asarray(aux["contrib"])[idx, 0]),
        "rho_output": (out["rho"], np.asarray(aux["rho"])[idx, 0]),
        "attention_residual_output": (out["residual"], np.asarray(aux["residual_logits"])[idx, 0]),
        "mean_h": (out["mean_h"], np.asarray(aux["mean_h"])[idx]),
        "mean_h2": (out["mean_h2"], np.asarray(aux["mean_h2"])[idx]),
        "head_pooled_vector": (out["pooled"], np.asarray(aux["attention_pooled"])[idx]),
        "log1p_n": (out["log1p_n"], np.log1p(n_atoms)),
        "node_count": (out["n"], n_atoms.astype(np.float64)),
    }
    worst = 0.0
    for name, (a, b) in checks.items():
        mx, mean = _err(a, b)
        rec["errors"][name] = {"max_abs": mx, "mean_abs": mean,
                               "reference_scale": float(np.max(np.abs(np.asarray(b))) if
                                                        np.asarray(b).size else 0.0)}
        if name.startswith("pred_eV"):
            worst = max(worst, mx)
    rec["max_prediction_error_eV"] = worst
    rec["tolerance_eV"] = C.TOL_STATE_RECONSTRUCTION_EV
    rec["passes"] = bool(worst <= C.TOL_STATE_RECONSTRUCTION_EV)
    rec["note"] = (
        "the reconstruction is float64 while the frozen forward is float32; the "
        "residual is float32 rounding, not a missing term. This is an equivalent "
        "rewrite of the existing computation, not a performance improvement."
    )
    order = np.argsort(np.abs(out["pred_eV"] - states["pred_model_eV"][idx])[::-1])[:3]
    rec["examples"] = [
        {
            "mol_id": int(mol_ids[idx[k]]),
            "n_atoms": int(n_atoms[k]),
            "state_l2_norm": float(np.linalg.norm(U[idx[k]])),
            "state_nonzero_entries": int(np.count_nonzero(U[idx[k]])),
            "reconstructed_eV": float(out["pred_eV"][k]),
            "frozen_eV": float(states["pred_model_eV"][idx[k]]),
            "abs_error_eV": float(abs(out["pred_eV"][k] - states["pred_model_eV"][idx[k]])),
        }
        for k in order.tolist()
    ]
    return rec


def same_record_check(machine: S.StateMachine, states: Mapping[str, Any], *,
                      max_molecules: int = C.MAX_GROUP_MOLECULES) -> dict[str, Any]:
    """Equal record digest => equal ``h`` and equal ``phi(h)`` (float32 spread)."""
    mol_ids = np.asarray(states["mol_ids"], np.int64)
    sub = hash_subset(mol_ids, cap=max_molecules, salt="group")
    atom_start = np.asarray(states["atom_start"], np.int64)
    atom_stop = np.asarray(states["atom_stop"], np.int64)
    parts = [np.arange(atom_start[i], atom_stop[i]) for i in sub.tolist()]
    atoms = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    hi = states["rec_hi"][atoms]
    lo = states["rec_lo"][atoms]
    order = np.lexsort((lo, hi))
    hi_s, lo_s, atoms_s = hi[order], lo[order], atoms[order]
    new = np.r_[True, (hi_s[1:] != hi_s[:-1]) | (lo_s[1:] != lo_s[:-1])]
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], hi_s.size]
    counts = ends - starts
    multi = np.flatnonzero(counts >= 2)
    h = states["h"]
    max_h = 0.0
    n_atoms_checked = 0
    cross = 0
    for g in multi.tolist():
        aa = atoms_s[starts[g]:ends[g]]
        hh = np.asarray(h[aa], np.float64)
        max_h = max(max_h, float(np.abs(hh - hh[0]).max()))
        n_atoms_checked += int(aa.size)
        pos = np.searchsorted(sub, np.searchsorted(atom_start, aa, side="right") - 1)
        if np.unique(pos).size > 1:
            cross += 1
    n_phi = min(200, int(multi.size))
    max_phi = 0.0
    for g in multi[:n_phi].tolist():
        aa = atoms_s[starts[g]:ends[g]]
        ph = machine.phi_atoms(np.asarray(h[aa], np.float64), states["scores"][aa])
        max_phi = max(max_phi, float(np.abs(ph - ph[0]).max()))
    return {
        "n_molecules": int(sub.size),
        "n_atoms": int(atoms.size),
        "n_distinct_records": int(starts.size),
        "n_repeated_record_groups": int(multi.size),
        "n_repeated_atoms": int(n_atoms_checked),
        "n_cross_molecule_groups": int(cross),
        "max_abs_h_deviation": float(max_h),
        "max_abs_phi_deviation": float(max_phi),
        "phi_groups_checked": int(n_phi),
        "tolerance_h": float(C.TOL_SAME_RECORD_H),
        "passes": bool(max_h <= C.TOL_SAME_RECORD_H),
        "note": (
            "records are compared as exact BLAKE2b-128 digests of canonical "
            "bytes; no learned-embedding or approximate-equality test is used. "
            "The residual spread is float32 reassociation inside index_add_."
        ),
    }


def renumbering_check(w3: Mapping[str, Any], states: Mapping[str, Any],
                      machine: S.StateMachine) -> dict[str, Any]:
    import ew_collate as EC

    n = min(int(C.RENUMBER_MOLECULES), int(states["n_mol"]))
    sub = hash_subset(np.asarray(states["mol_ids"], np.int64), cap=n, salt="renumber")
    rows = states["rows"][sub]
    cache = w3["cache"]
    EM = w3["imports"][4]
    batch = EC.collate_v2(cache, rows.tolist())
    perm, _p = EM.permute_atoms(batch)
    r0 = REC.extract_batch_records(batch)
    r1 = REC.extract_batch_records(perm)
    sig_ok = True
    n_graphs = int(batch["n_graphs"])
    for g in range(n_graphs):
        m0 = REC.multiset_counts(r0["hi"][r0["atom_start"][g]:r0["atom_stop"][g]],
                                 r0["lo"][r0["atom_start"][g]:r0["atom_stop"][g]])
        m1 = REC.multiset_counts(r1["hi"][r1["atom_start"][g]:r1["atom_stop"][g]],
                                 r1["lo"][r1["atom_start"][g]:r1["atom_stop"][g]])
        if not REC.multiset_equal(m0, m1):
            sig_ok = False
            break
    fw0 = S.forward_atoms(w3["model"], batch)
    fw1 = S.forward_atoms(w3["model"], perm)
    U0 = machine.graph_state(fw0["h"], fw0["scores"], fw0["atom_start"])
    U1 = machine.graph_state(fw1["h"], fw1["scores"], fw1["atom_start"])
    z0 = fw0["logits"]
    z1 = fw1["logits"]
    return {
        "n_molecules": int(n_graphs),
        "record_multiset_identical": bool(sig_ok),
        "max_abs_state_delta": float(np.abs(U0 - U1).max()),
        "max_rel_state_delta": float(np.abs(U0 - U1).max() / max(np.abs(U0).max(), 1e-12)),
        "max_abs_prediction_delta_eV": float(np.abs(z0 - z1).max() * machine.label_std_eV),
        "note": (
            "atom renumbering changes neither the record multiset nor the state "
            "up to float32 reassociation; the predictions differ only by float32 "
            "summation order."
        ),
    }


def record_increment_check(machine: S.StateMachine, states: Mapping[str, Any], U: np.ndarray,
                           rect_idx: np.ndarray, rect: Mapping[str, np.ndarray],
                           mol_of: Mapping[str, np.ndarray],
                           mol_index: Mapping[int, int]) -> dict[str, Any]:
    """``dU`` recomputed from the signed record multiset vs ``U(B) - U(A)``."""
    atom_start = np.asarray(states["atom_start"], np.int64)
    atom_stop = np.asarray(states["atom_stop"], np.int64)
    abs_err: list[float] = []
    scales: list[float] = []
    spread = 0.0
    node_ok = 0
    node_bad = 0
    n_nonzero = 0
    for i in rect_idx.tolist():
        mA = mol_index[int(mol_of["m1"][i])]
        mB = mol_index[int(mol_of["m2"][i])]
        atoms = np.r_[np.arange(atom_start[mA], atom_stop[mA]),
                      np.arange(atom_start[mB], atom_stop[mB])]
        signs = np.r_[-np.ones(atom_stop[mA] - atom_start[mA], np.float64),
                      np.ones(atom_stop[mB] - atom_start[mB], np.float64)]
        rec, sp = _signed_phi(machine, states, atoms, signs)
        spread = max(spread, sp)
        target = U[mB] - U[mA]
        scale = float(np.max(np.abs(target)))
        abs_err.append(float(np.max(np.abs(rec - target))))
        scales.append(scale)
        if scale > 0:
            n_nonzero += 1
        dn_state = float(U[mB, 0] - U[mA, 0])
        dn_true = float((atom_stop[mB] - atom_start[mB]) - (atom_stop[mA] - atom_start[mA]))
        if abs(dn_state - dn_true) < 1e-9:
            node_ok += 1
        else:
            node_bad += 1
    ae = np.asarray(abs_err)
    sc = np.asarray(scales)
    tol = C.TOL_RECORD_DELTA_U_ABS + C.TOL_RECORD_DELTA_U_REL * sc
    return {
        "n_pairs_checked": int(ae.size),
        "n_nonzero_state_increments": int(n_nonzero),
        "max_abs_increment_error": float(ae.max()) if ae.size else 0.0,
        "mean_abs_increment_error": float(ae.mean()) if ae.size else 0.0,
        "max_rel_increment_error": float((ae / np.maximum(sc, 1e-12)).max()) if ae.size else 0.0,
        "n_pairs_within_tolerance": int((ae <= tol).sum()),
        "pass_fraction": float(np.mean(ae <= tol)) if ae.size else float("nan"),
        "tolerance": {"abs": C.TOL_RECORD_DELTA_U_ABS, "rel": C.TOL_RECORD_DELTA_U_REL},
        "max_within_record_h_spread": float(spread),
        "node_count_accounted": int(node_ok),
        "node_count_failed": int(node_bad),
        "node_count_change_nonzero_fraction": float(np.mean(
            (atom_stop - atom_start)[[mol_index[int(mol_of["m2"][i])]
                                      for i in rect_idx.tolist()]]
            != (atom_stop - atom_start)[[mol_index[int(mol_of["m1"][i])]
                                         for i in rect_idx.tolist()]]
        )) if ae.size else float("nan"),
        "note": (
            "dU is rebuilt as sum_k w_k * phi(h_rep_k) over the signed record "
            "multiset: one representative per record class, the signed "
            "multiplicity as weight. U(B) - U(A) is only the comparison target."
        ),
    }


def increment_replay(machine: S.StateMachine, states: Mapping[str, Any], U: np.ndarray,
                     rect_idx: np.ndarray, rect: Mapping[str, np.ndarray],
                     mol_of: Mapping[str, np.ndarray], mol_index: Mapping[int, int],
                     frozen_pred_eV_mol: np.ndarray, *, max_rectangles: int = 512) -> dict[str, Any]:
    """Record-derived ``dU`` + ``dg`` reproduces the frozen ``F(B)`` from ``A``."""
    atom_start = np.asarray(states["atom_start"], np.int64)
    atom_stop = np.asarray(states["atom_stop"], np.int64)
    order = np.argsort(np.asarray([C.hash64("increment", int(i)) for i in rect_idx.tolist()],
                                  np.uint64), kind="stable")
    sel = np.asarray(sorted(order[: int(max_rectangles)].tolist()), np.int64)
    sel = rect_idx[sel]
    err_pred = 0.0
    err_delta = 0.0
    dg_norm = []
    n = 0
    for i in sel.tolist():
        mA = mol_index[int(mol_of["m1"][i])]
        mB = mol_index[int(mol_of["m2"][i])]
        atoms = np.r_[np.arange(atom_start[mA], atom_stop[mA]),
                      np.arange(atom_start[mB], atom_stop[mB])]
        signs = np.r_[-np.ones(atom_stop[mA] - atom_start[mA], np.float64),
                      np.ones(atom_stop[mB] - atom_start[mB], np.float64)]
        dU, _sp = _signed_phi(machine, states, atoms, signs)
        gA = states["g"][mA]
        gB = states["g"][mB]
        moved = float(machine.readout((U[mA] + dU)[None, :], (gA + (gB - gA))[None, :])["pred_eV"][0])
        f_b = float(frozen_pred_eV_mol[mB])
        f_a = float(frozen_pred_eV_mol[mA])
        err_pred = max(err_pred, abs(moved - f_b))
        err_delta = max(err_delta, abs((moved - f_a) - (f_b - f_a)))
        dg_norm.append(float(np.linalg.norm(gB - gA)))
        n += 1
    return {
        "n_pairs": int(n),
        "max_abs_error_prediction_eV": float(err_pred),
        "max_abs_error_double_difference_eV": float(err_delta),
        "tolerance_eV": float(C.TOL_STATE_RECONSTRUCTION_EV),
        "passes": bool(err_pred <= C.TOL_STATE_RECONSTRUCTION_EV),
        "descriptor_change_norm_median": float(np.median(dg_norm)) if dg_norm else float("nan"),
        "descriptor_change_nonzero_fraction": float(np.mean(np.asarray(dg_norm) > 0)) if dg_norm else float("nan"),
        "target": "frozen full-Valid W3 prediction of the real molecule B",
        "statement": (
            "F(B) is reproduced from U(A), the signed root-record increment and "
            "the descriptor increment: F(B) - F(A) = H(U(A)+dU, g(A)+dg) - "
            "H(U(A), g(A)). dU never uses U(B)."
        ),
    }


def run(w3: Mapping[str, Any], obs: Mapping[str, Any], machine: S.StateMachine,
        states: Mapping[str, Any], U: np.ndarray, sample: Mapping[str, Any],
        sel: Mapping[str, Any], rect_idx: np.ndarray,
        frozen_pred_eV_mol: np.ndarray) -> dict[str, Any]:
    t0 = time.perf_counter()
    rect = sel["rect"]
    mol_index = {int(m): i for i, m in enumerate(np.asarray(states["mol_ids"]).tolist())}
    mol_of = sample["molecule_cols"]
    out = {
        "section4_replay": section4(machine, states, U, frozen_pred_eV_mol),
        "same_record": same_record_check(machine, states),
        "renumbering": renumbering_check(w3, states, machine),
        "record_increment": record_increment_check(machine, states, U, rect_idx, rect,
                                                   mol_of, mol_index),
        "increment_replay": increment_replay(machine, states, U, rect_idx, rect, mol_of,
                                             mol_index, frozen_pred_eV_mol),
    }
    out["timing_seconds"] = float(time.perf_counter() - t0)
    return out
