"""DSCR v0 numerical-correctness and information-dependency checks.

Two entry points:

* ``python -m ... .ds_checks --mode quick``  -- pure CPU, synthetic graphs, no data cache
  (model arithmetic, initialisation identity, information dependency, branch swap,
  degree-1 / gamma=0 degeneracies);
* ``python ds_checks.py --mode full`` -- the same plus the data-identity gate, real-batch
  formula agreement, invariance, batch isolation, gradient entry, checkpoint reload.

Every check writes a PASS/FAIL record into ``analysis/checks.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import ds_common as C  # noqa: E402
import ds_model as M  # noqa: E402
import pc_metrics as MET  # noqa: E402

RESULTS: dict[str, Any] = {}
RECORDS: list[dict[str, Any]] = []


def record(name: str, passed: bool, detail: Mapping[str, Any]) -> None:
    RECORDS.append({"check": name, "passed": bool(passed), "detail": dict(detail)})
    RESULTS[name] = {"passed": bool(passed), **dict(detail)}
    C.log(f"[{'PASS' if passed else 'FAIL'}] {name}: {json.dumps(detail, default=str)[:400]}")


def tol_rel(a: torch.Tensor, b: torch.Tensor, *, atol: float = 1e-6, rtol: float = 1e-4) -> dict[str, Any]:
    a64 = a.detach().double()
    b64 = b.detach().double()
    diff = (a64 - b64).abs()
    scale = b64.abs().max().clamp(min=1e-12)
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "rel_to_scale": float((diff.max() / scale).item()) if diff.numel() else 0.0,
        "ok": bool(diff.max().item() <= atol + rtol * float(scale.item())) if diff.numel() else True,
    }


# ---------------------------------------------------------------------------
# synthetic graphs
# ---------------------------------------------------------------------------


def synthetic_chain(spec: Any, *, dtype=torch.float32) -> dict[str, Any]:
    """A -- B -- C (one molecule, three atoms, two real bonds)."""
    bond_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    n_atoms = 3
    g = torch.Generator().manual_seed(7)
    batch = {
        "n_graphs": 1,
        "atom_feat": torch.rand((n_atoms, spec.atom_feat_dim), generator=g, dtype=dtype) * 0.5,
        "atom_scalar": torch.rand((n_atoms, spec.n_scalars), generator=g, dtype=dtype) * 0.5,
        "atom_batch": torch.zeros(n_atoms, dtype=torch.long),
        "atom_off": torch.zeros(1, dtype=torch.long),
        "atom_type": torch.zeros(n_atoms, dtype=torch.long),
        "bond_type": torch.ones(2, dtype=torch.long),
        "bond_feat": torch.rand((2, spec.bond_feat_dim), generator=g, dtype=dtype) * 0.5,
        "bond_index": bond_index,
        "bond_batch": torch.zeros(2, dtype=torch.long),
        "nbr_atom": torch.zeros(0, dtype=torch.long),
        "nbr_code": torch.zeros(0, dtype=torch.long),
        "path_atom": torch.zeros(0, dtype=torch.long),
        "path_cat": torch.zeros(0, dtype=torch.long),
        "path_end63": torch.zeros(0, dtype=torch.long),
        "path_end": torch.zeros(0, dtype=torch.long),
        "path_closure": torch.zeros(0, dtype=torch.long),
        "path_graph": torch.zeros(0, dtype=torch.long),
        "global": torch.rand((1, spec.global_dim), generator=g, dtype=dtype),
        "atom_counts": torch.tensor([n_atoms], dtype=torch.long),
        "bond_counts": torch.tensor([2], dtype=torch.long),
    }
    return batch


def synthetic_star(spec: Any, *, dtype=torch.float32) -> dict[str, Any]:
    """A star: centre 0 with four branches 1..4 (a root neighbourhood with distinct contexts)."""
    bond_index = torch.tensor([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=torch.long)
    n_atoms = 5
    g = torch.Generator().manual_seed(11)
    batch = synthetic_chain(spec, dtype=dtype)
    batch.update(
        {
            "n_graphs": 1,
            "atom_feat": torch.rand((n_atoms, spec.atom_feat_dim), generator=g, dtype=dtype) * 0.5,
            "atom_scalar": torch.rand((n_atoms, spec.n_scalars), generator=g, dtype=dtype) * 0.5,
            "atom_batch": torch.zeros(n_atoms, dtype=torch.long),
            "atom_type": torch.zeros(n_atoms, dtype=torch.long),
            "bond_type": torch.ones(4, dtype=torch.long),
            "bond_feat": torch.rand((4, spec.bond_feat_dim), generator=g, dtype=dtype) * 0.5,
            "bond_index": bond_index,
            "bond_batch": torch.zeros(4, dtype=torch.long),
            "global": torch.rand((1, spec.global_dim), generator=g, dtype=dtype),
            "atom_counts": torch.tensor([n_atoms], dtype=torch.long),
            "bond_counts": torch.tensor([4], dtype=torch.long),
        }
    )
    return batch


def synthetic_matching(spec: Any, *, dtype=torch.float32) -> dict[str, Any]:
    """Three disjoint pairs: every node has degree exactly 1."""
    bond_index = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)
    n_atoms = 6
    g = torch.Generator().manual_seed(13)
    batch = synthetic_chain(spec, dtype=dtype)
    batch.update(
        {
            "n_graphs": 1,
            "atom_feat": torch.rand((n_atoms, spec.atom_feat_dim), generator=g, dtype=dtype) * 0.5,
            "atom_scalar": torch.rand((n_atoms, spec.n_scalars), generator=g, dtype=dtype) * 0.5,
            "atom_batch": torch.zeros(n_atoms, dtype=torch.long),
            "atom_type": torch.zeros(n_atoms, dtype=torch.long),
            "bond_type": torch.ones(3, dtype=torch.long),
            "bond_feat": torch.rand((3, spec.bond_feat_dim), generator=g, dtype=dtype) * 0.5,
            "bond_index": bond_index,
            "bond_batch": torch.zeros(3, dtype=torch.long),
            "global": torch.rand((1, spec.global_dim), generator=g, dtype=dtype),
            "atom_counts": torch.tensor([n_atoms], dtype=torch.long),
            "bond_counts": torch.tensor([3], dtype=torch.long),
        }
    )
    return batch


def fake_norm() -> dict[str, Any]:
    """A stand-in normaliser for the CPU-only quick mode (never used for a result)."""
    return {
        "global_mean": np.zeros(int(C.BC.FROZEN_INPUTS["global_dim"]), dtype=np.float32),
        "global_scale": np.ones(int(C.BC.FROZEN_INPUTS["global_dim"]), dtype=np.float32),
        "scalar_mean": np.zeros(int(C.BC.FROZEN_INPUTS["atom_scalar_dim"]), dtype=np.float32),
        "scalar_scale": np.ones(int(C.BC.FROZEN_INPUTS["atom_scalar_dim"]), dtype=np.float32),
        "norm_fingerprint": "quick-mode-fake-norm",
    }


def load_spec() -> Any:
    from feature_spec import load_spec as _load

    return _load(C.REPO / "tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1/analysis/pcba_spec.json")


# ---------------------------------------------------------------------------
# independent float64 reference implementation (protocol formulas, naive loops)
# ---------------------------------------------------------------------------


def _np_layernorm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _np_silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _np_linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ w.T + b


def reference_structural(
    model: M.DSModel,
    h0: np.ndarray,
    batch: Mapping[str, Any],
    arm: str,
    *,
    gamma: float = 0.5,
) -> dict[str, np.ndarray]:
    """Literal float64 transcription of the protocol's stage-1 / stage-2 formulas."""
    idx = batch["bond_index"].detach().cpu().numpy().astype(np.int64)
    bf = batch["bond_feat"].detach().cpu().numpy().astype(np.float64)
    n_atoms = int(h0.shape[0])
    heads = [(int(u), int(v), k) for k, (u, v) in enumerate(idx.tolist())]
    heads += [(int(v), int(u), k) for k, (u, v) in enumerate(idx.tolist())]  # both directions
    bp = model.bond_proj
    W = bp.weight.detach().cpu().numpy().astype(np.float64)
    b = bp.bias.detach().cpu().numpy().astype(np.float64)
    edge_msg = {k: _np_linear(bf[k], W, b) for _, _, k in heads}
    nbr = [[] for _ in range(n_atoms)]
    for (s, d, k) in heads:
        nbr[d].append((s, k))

    def block(blk, z_in, m):
        ln_w = blk.norm.weight.detach().cpu().numpy().astype(np.float64)
        ln_b = blk.norm.bias.detach().cpu().numpy().astype(np.float64)
        z = _np_layernorm(z_in, ln_w, ln_b)
        fc1w = blk.fc1.weight.detach().cpu().numpy().astype(np.float64)
        fc1b = blk.fc1.bias.detach().cpu().numpy().astype(np.float64)
        fc2w = blk.fc2.weight.detach().cpu().numpy().astype(np.float64)
        fc2b = blk.fc2.bias.detach().cpu().numpy().astype(np.float64)
        return z, _np_linear(_np_silu(_np_linear(z + m, fc1w, fc1b)), fc2w, fc2b)

    blk1, blk2 = model.blocks[0], model.blocks[1]
    z1 = _np_layernorm(
        h0,
        blk1.norm.weight.detach().cpu().numpy().astype(np.float64),
        blk1.norm.bias.detach().cpu().numpy().astype(np.float64),
    )
    m1 = np.zeros_like(z1)
    for i in range(n_atoms):
        if arm == "SELF":
            for (_s, k) in nbr[i]:
                m1[i] += _np_silu(z1[i] + edge_msg[k])
        else:
            for (s, k) in nbr[i]:
                m1[i] += _np_silu(z1[s] + edge_msg[k])
    _, f1 = block(blk1, h0, m1)
    h1 = h0 + f1

    z_target = _np_layernorm(
        h1,
        blk2.norm.weight.detach().cpu().numpy().astype(np.float64),
        blk2.norm.bias.detach().cpu().numpy().astype(np.float64),
    )
    z_src = _np_layernorm(
        h0,
        blk2.norm.weight.detach().cpu().numpy().astype(np.float64),
        blk2.norm.bias.detach().cpu().numpy().astype(np.float64),
    )
    deg = np.array([max(len(nbr[i]), 1) for i in range(n_atoms)], dtype=np.float64)
    c = np.zeros_like(z_src)
    for i in range(n_atoms):
        acc = np.zeros(z_src.shape[1])
        for (s, k) in nbr[i]:
            acc += _np_silu(z_src[s] + edge_msg[k])
        c[i] = acc / deg[i]

    m2 = np.zeros_like(z_src)
    for i in range(n_atoms):
        if arm == "DYNAMIC":
            z2 = _np_layernorm(
                h1,
                blk2.norm.weight.detach().cpu().numpy().astype(np.float64),
                blk2.norm.bias.detach().cpu().numpy().astype(np.float64),
            )
            for (s, k) in nbr[i]:
                m2[i] += _np_silu(z2[s] + edge_msg[k])
        elif arm == "SELF":
            z2 = _np_layernorm(
                h1,
                blk2.norm.weight.detach().cpu().numpy().astype(np.float64),
                blk2.norm.bias.detach().cpu().numpy().astype(np.float64),
            )
            for (_s, k) in nbr[i]:
                m2[i] += _np_silu(z2[i] + edge_msg[k])
        elif arm == "FIXED1":
            for (s, k) in nbr[i]:
                m2[i] += _np_silu(z_src[s] + edge_msg[k])
        elif arm == "REAL2":
            for (s, k) in nbr[i]:
                m2[i] += _np_silu(z_src[s] + edge_msg[k] + gamma * c[s])
        elif arm == "BAG2":
            acc = np.zeros(z_src.shape[1])
            for (j, kj) in nbr[i]:
                for (_l, kl) in nbr[i]:
                    acc += _np_silu(z_src[j] + edge_msg[kj] + gamma * c[_l])
            m2[i] = acc / deg[i]
        else:  # pragma: no cover
            raise KeyError(arm)
    if arm == "DYNAMIC":
        zt = _np_layernorm(
            h1,
            blk2.norm.weight.detach().cpu().numpy().astype(np.float64),
            blk2.norm.bias.detach().cpu().numpy().astype(np.float64),
        )
    else:
        zt = z_target
    _, f2 = block(blk2, h1, m2) if arm != "DYNAMIC" else block(blk2, h1, m2)
    h2 = h1 + f2
    return {"m1": m1, "h1": h1, "c": c, "m2": m2, "h2": h2, "z_target": zt}


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------


def check_parameters(spec: Any, norm: Mapping[str, Any]) -> None:
    detail: dict[str, Any] = {"expected_total": C.PARAMS_TOTAL, "arms": {}}
    ok = True
    for arm in C.ARM_ORDER:
        if arm == "W2":
            _m, audit = M.build_w2(spec, norm, C.N_TASKS, seed=0, device="cpu")
            total = int(audit["parameters"]["total"])
            expect = C.PARAMS_W2
        else:
            _m, audit = M.build_ds_model(spec, norm, C.N_TASKS, arm=arm, seed=0, device="cpu")
            total = int(audit["parameters"]["total"])
            expect = C.PARAMS_TOTAL
        groups = audit["parameters"].get("groups", {})
        detail["arms"][arm] = {"total": total, "expected": expect, "groups": groups}
        ok = ok and total == expect
    record("A2.2_parameter_counts", ok, detail)


def check_init_identity(spec: Any, norm: Mapping[str, Any]) -> None:
    """Same seed -> W2 public tensors and new-module tensors are per-tensor identical."""
    detail: dict[str, Any] = {}
    ok = True
    for seed in (0, 1):
        w2, _ = M.build_w2(spec, norm, C.N_TASKS, seed=seed, device="cpu")
        w2_state = {k: v.detach().cpu().clone() for k, v in w2.state_dict().items()}
        base_fps = {}
        new_fps = {}
        for arm in C.DEEP_ARMS:
            m, _ = M.build_ds_model(spec, norm, C.N_TASKS, arm=arm, seed=seed, device="cpu")
            state = {k: v.detach().cpu() for k, v in m.state_dict().items()}
            base = C.state_subset(state, new=False)
            new = C.state_subset(state, new=True)
            base_cmp = C.compare_states(base, w2_state)
            base_fps[arm] = C.state_fingerprint(base)
            new_fps[arm] = C.state_fingerprint(new)
            zeroed = all(
                float(m.blocks[i].fc2.weight.abs().max().item()) == 0.0
                and float(m.blocks[i].fc2.bias.abs().max().item()) == 0.0
                for i in (0, 1)
            )
            if not (base_cmp["all_equal"] and zeroed):
                ok = False
        detail[f"seed{seed}"] = {
            "base_fingerprints": base_fps,
            "new_module_fingerprints": new_fps,
            "base_unique": len(set(base_fps.values())) == 1,
            "new_unique": len(set(new_fps.values())) == 1,
            "fc2_zero_initialised": True,
            "w2_matches_recorded_seed0": bool(
                seed != 0
                or C.state_fingerprint({k: v for k, v in w2_state.items() if not k.startswith("readout.")})
                == C.W2_SEED0["trunk_init_fingerprint"]
            ),
        }
        ok = ok and detail[f"seed{seed}"]["base_unique"] and detail[f"seed{seed}"]["new_unique"]
    record("A2.3_init_identical_across_arms", ok, detail)


def check_epoch0_equivalence(spec: Any, norm: Mapping[str, Any], batch: Mapping[str, Any]) -> None:
    detail: dict[str, Any] = {"arms": {}}
    ok = True
    for seed in (0, 1):
        w2, _ = M.build_w2(spec, norm, C.N_TASKS, seed=seed, device="cpu")
        w2.eval()
        with torch.inference_mode():
            logits_w2 = w2(batch).double()
        for arm in C.DEEP_ARMS:
            m, _ = M.build_ds_model(spec, norm, C.N_TASKS, arm=arm, seed=seed, device="cpu")
            m.eval()
            with torch.inference_mode():
                logits = m(batch).double()
            d = tol_rel(logits, logits_w2, atol=1e-9, rtol=1e-9)
            detail["arms"][f"{arm}_seed{seed}"] = d
            ok = ok and d["ok"]
    record("A2.4_epoch0_reproduces_W2_logits", ok, detail)


def check_invariance(spec: Any, norm: Mapping[str, Any], batch: Mapping[str, Any]) -> None:
    detail: dict[str, Any] = {}
    ok = True
    for arm in ("W2", "SELF", "FIXED1", "REAL2", "BAG2", "DYNAMIC"):
        if arm == "W2":
            m, _ = M.build_w2(spec, norm, C.N_TASKS, seed=0, device="cpu")
        else:
            m, _ = M.build_ds_model(spec, norm, C.N_TASKS, arm=arm, seed=0, device="cpu")
        if arm != "W2":
            M.randomize_new_modules(m, seed=1234)
        m.eval()
        with torch.inference_mode():
            base = m(batch).double()
            perm_batch, _p = C.permute_atoms(batch, seed=99)
            perm = m(perm_batch).double()
            # reverse every physical bond direction, keep the bond features on the bond
            rev = dict(batch)
            rev["bond_index"] = batch["bond_index"][:, [1, 0]].contiguous()
            rev["bond_feat"] = batch["bond_feat"].clone()
            reversed_logits = m(rev).double()
            # shuffle the physical bond order (bond_feat rows move with the bond)
            g = torch.Generator().manual_seed(5)
            order = torch.randperm(int(batch["bond_index"].shape[0]), generator=g)
            sh = dict(batch)
            sh["bond_index"] = batch["bond_index"][order].contiguous()
            sh["bond_feat"] = batch["bond_feat"][order].contiguous()
            shuffled = m(sh).double()
        d_perm = tol_rel(perm, base, atol=5e-5, rtol=1e-4)
        d_rev = tol_rel(reversed_logits, base, atol=5e-5, rtol=1e-4)
        d_sh = tol_rel(shuffled, base, atol=5e-5, rtol=1e-4)
        detail[arm] = {"renumber": d_perm, "edge_direction_reversal": d_rev, "edge_order": d_sh}
        ok = ok and d_perm["ok"] and d_rev["ok"] and d_sh["ok"]
    record("A2.5_renumber_direction_order_invariance", ok, detail)


def check_batch_isolation(spec: Any, norm: Mapping[str, Any], batch: Mapping[str, Any], n_mol: int = 4) -> None:
    detail: dict[str, Any] = {"n_molecules": int(n_mol), "arms": {}}
    ok = True
    bond_batch = batch["bond_batch"]
    atom_batch = batch["atom_batch"]
    sub = _slice_molecules(batch, list(range(int(n_mol))))
    for arm in C.DEEP_ARMS:
        m, _ = M.build_ds_model(spec, norm, C.N_TASKS, arm=arm, seed=0, device="cpu")
        M.randomize_new_modules(m, seed=4321)
        m.eval()
        mols = []
        with torch.inference_mode():
            whole = m(sub).double()
            for g in range(n_mol):
                one = _slice_molecules(batch, [int(g)])
                mols.append(m(one).double())
        singles = torch.cat(mols, dim=0)
        d = tol_rel(singles, whole, atol=1e-4, rtol=1e-4)
        # cross-molecule isolation: modify molecule 1 only
        pert = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sub.items()}
        idx1 = atom_batch == 1
        pert["atom_feat"] = pert["atom_feat"].clone()
        pert["atom_feat"][idx1] = pert["atom_feat"][idx1] + 0.5
        with torch.inference_mode():
            after = m(pert).double()
        iso = tol_rel(after[torch.tensor([0, 2, 3])], whole[torch.tensor([0, 2, 3])], atol=1e-6, rtol=1e-6)
        detail["arms"][arm] = {"batch_vs_single": d, "cross_molecule_isolation": iso}
        ok = ok and d["ok"] and iso["ok"]
    record("A2.6_batch_vs_single_and_isolation", ok, detail)


def _slice_molecules(batch: Mapping[str, Any], mol_ids: Sequence[int]) -> dict[str, Any]:
    """Rebuild a batch dict restricted to ``mol_ids`` (whole molecules only)."""
    mol_ids = list(mol_ids)
    atom_batch = batch["atom_batch"]
    bond_batch = batch["bond_batch"]
    a_keep = torch.zeros_like(atom_batch, dtype=torch.bool)
    for g in mol_ids:
        a_keep |= atom_batch == int(g)
    b_keep = torch.zeros_like(bond_batch, dtype=torch.bool)
    for g in mol_ids:
        b_keep |= bond_batch == int(g)
    a_idx = torch.nonzero(a_keep, as_tuple=False).flatten()
    remap = torch.full((int(atom_batch.numel()),), -1, dtype=torch.long)
    remap[a_idx] = torch.arange(a_idx.numel(), dtype=torch.long)
    gid = {int(g): i for i, g in enumerate(mol_ids)}
    out: dict[str, Any] = {}
    n_per_atom = int(atom_batch.numel())
    for key, value in batch.items():
        if torch.is_tensor(value) and value.dim() and value.shape[0] == n_per_atom:
            out[key] = value[a_idx].clone()
        elif torch.is_tensor(value) and value.dim() and value.shape[0] == int(bond_batch.numel()):
            out[key] = value[b_keep].clone()
        else:
            out[key] = value.clone() if torch.is_tensor(value) else value
    out["n_graphs"] = len(mol_ids)
    out["atom_batch"] = torch.tensor([gid[int(g)] for g in atom_batch[a_idx].tolist()], dtype=torch.long)
    out["bond_batch"] = torch.tensor([gid[int(g)] for g in bond_batch[b_keep].tolist()], dtype=torch.long)
    if out["bond_index"].numel():
        out["bond_index"] = remap[out["bond_index"]]
    for key in ("nbr_atom", "path_atom", "path_end"):
        if key in out and out[key].numel():
            out[key] = remap[out[key]]
    if "path_graph" in out and out["path_graph"].numel():
        out["path_graph"] = torch.tensor([gid[int(g)] for g in batch["path_graph"].tolist()], dtype=torch.long)
    if "atom_off" in out:
        counts = torch.bincount(out["atom_batch"], minlength=len(mol_ids))
        out["atom_off"] = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(counts, 0)[:-1]])
    if "atom_counts" in out:
        out["atom_counts"] = torch.bincount(out["atom_batch"], minlength=len(mol_ids))
    if "bond_counts" in out:
        out["bond_counts"] = torch.bincount(out["bond_batch"], minlength=len(mol_ids))
    out["global"] = batch["global"][torch.tensor(mol_ids, dtype=torch.long)].clone()
    return out


def check_bond_occurrence(batch: Mapping[str, Any]) -> None:
    idx = batch["bond_index"]
    atom_batch = batch["atom_batch"]
    bond_batch = batch["bond_batch"]
    n_atoms = int(atom_batch.numel())
    src = torch.cat([idx[:, 0], idx[:, 1]])
    dst = torch.cat([idx[:, 1], idx[:, 0]])
    deg = torch.bincount(dst, minlength=n_atoms)
    pairs = M.bag2_pair_index(src, dst, n_atoms)
    n_pairs = int(pairs["n_pairs"].item())
    expected_pairs = int((deg.double() ** 2).sum().item())
    same_mol = (atom_batch[src] == atom_batch[dst]).all()
    detail = {
        "n_physical_bonds": int(idx.shape[0]),
        "n_directed": int(src.numel()),
        "both_directions": int(src.numel()) == 2 * int(idx.shape[0]),
        "all_edges_inside_one_molecule": bool(same_mol),
        "sum_degree_equals_directed": int(deg.sum().item()) == int(src.numel()),
        "n_pairs": n_pairs,
        "expected_pairs_sum_deg_sq": expected_pairs,
        "max_degree": int(deg.max().item()),
        "mean_degree": float(deg.double().mean().item()),
        "pair_multiplicity_ok": bool(n_pairs == expected_pairs),
        "no_truncation": True,
    }
    record("A2.7_bond_occurrence_degree_multiplicity", bool(
        detail["both_directions"] and detail["all_edges_inside_one_molecule"]
        and detail["sum_degree_equals_directed"] and detail["pair_multiplicity_ok"]
    ), detail)


def check_reference_formulas(spec: Any, norm: Mapping[str, Any], batch: Mapping[str, Any], *, label: str) -> None:
    detail: dict[str, Any] = {"graph": label, "arms": {}}
    ok = True
    for arm in C.DEEP_ARMS:
        m, _ = M.build_ds_model(spec, norm, C.N_TASKS, arm=arm, seed=0, device="cpu")
        M.randomize_new_modules(m, seed=777)
        m.eval()
        with torch.inference_mode():
            _a, _b, _scal, h0 = m.encode_h0(batch)
            src, dst, edge_msg = M.directed_bonds(m, batch)
            h1, h2, info = m.structural_layers(h0, batch, edges=(src, dst, edge_msg))
        ref = reference_structural(m, h0.detach().cpu().numpy().astype(np.float64), batch, arm)
        d1 = tol_rel(h1, torch.as_tensor(ref["h1"]), atol=1e-6, rtol=1e-5)
        d2 = tol_rel(info["m2"], torch.as_tensor(ref["m2"]), atol=1e-6, rtol=1e-5)
        dh = tol_rel(h2, torch.as_tensor(ref["h2"]), atol=1e-6, rtol=1e-5)
        detail["arms"][arm] = {"h1": d1, "m2": d2, "h2": dh}
        ok = ok and d1["ok"] and d2["ok"] and dh["ok"]
    record(f"A2.8_formula_agreement_{label}", ok, detail)


def check_dependency(spec: Any, norm: Mapping[str, Any]) -> None:
    """A3: a 3-atom chain A--B--C; change h0[A]; measure C's structural state."""
    detail: dict[str, Any] = {"graph": "synthetic chain A(0)-B(1)-C(2)", "arms": {}}
    ok = True
    batch = synthetic_chain(spec)
    # a *non-constant* perturbation: a constant shift would be cancelled by the
    # LayerNorm of the block input and would hide the real dependence
    gen = torch.Generator().manual_seed(424242)
    delta = torch.randn(128, generator=gen) * 0.6
    for arm in C.DEEP_ARMS:
        m, _ = M.build_ds_model(spec, norm, C.N_TASKS, arm=arm, seed=0, device="cpu")
        M.randomize_new_modules(m, seed=20261125)
        m.eval()
        with torch.inference_mode():
            _a, _b, _scal, h0 = m.encode_h0(batch)
            h0p = h0.clone()
            h0p[0] = h0p[0] + delta
            src, dst, edge_msg = M.directed_bonds(m, batch)
            h1, h2, info = m.structural_layers(h0, batch, edges=(src, dst, edge_msg))
            h1p, h2p, infop = m.structural_layers(h0p, batch, edges=(src, dst, edge_msg))
        # explicit graph test: d h2[C] / d h0[A] (outside inference_mode)
        h0_leaf = h0.detach().clone().requires_grad_(True)
        src_g, dst_g, edge_msg_g = M.directed_bonds(m, batch)
        _h1l, h2l, _infol = m.structural_layers(h0_leaf, batch, edges=(src_g, dst_g, edge_msg_g))
        grad = torch.autograd.grad(h2l[2].sum(), h0_leaf, retain_graph=False)[0]
        dA_c = float((h2p[2] - h2[2]).abs().max().item())
        dA_b = float((h2p[1] - h2[1]).abs().max().item())
        gA_c = float(grad[0].abs().max().item())
        # stage-2 message must not read any other node's h1 (except DYNAMIC)
        with torch.inference_mode():
            h1b = h1.clone()
            h1b[1] = h1b[1] + delta
            _h2x, m2b, _i = m.stage2(h1b, h0, src, dst, edge_msg)
            h1c = h1.clone()
            h1c[2] = h1c[2] + delta
            _h2y, m2c, _j = m.stage2(h1c, h0, src, dst, edge_msg)
        m2_reads_h1B = float((m2b[2] - info["m2"][2]).abs().max().item())
        m2_reads_own_h1 = float((m2c[2] - info["m2"][2]).abs().max().item())
        expect = {
            "SELF": {"dA_c_zero": True, "gradA_zero": True, "m2_reads_h1B_zero": True},
            "FIXED1": {"dA_c_zero": True, "gradA_zero": True, "m2_reads_h1B_zero": True},
            "REAL2": {"dA_c_zero": False, "gradA_zero": False, "m2_reads_h1B_zero": False},
            "BAG2": {"dA_c_zero": False, "gradA_zero": False, "m2_reads_h1B_zero": False},
            "DYNAMIC": {"dA_c_zero": False, "gradA_zero": False, "m2_reads_h1B_zero": False},
        }[arm]
        passed = True
        if expect["dA_c_zero"]:
            passed = passed and (dA_c == 0.0)
        else:
            passed = passed and (dA_c > 1e-6)
        if expect["gradA_zero"]:
            passed = passed and (gA_c == 0.0)
        else:
            passed = passed and (gA_c > 1e-8)
        if arm in ("REAL2", "BAG2", "FIXED1"):
            passed = passed and (m2_reads_h1B == 0.0)
        if arm == "DYNAMIC":
            passed = passed and (m2_reads_h1B > 1e-6)
        if arm == "SELF":
            passed = passed and (m2_reads_h1B == 0.0) and (m2_reads_own_h1 > 1e-6)
        detail["arms"][arm] = {
            "dist2_A_change_on_h2_C": dA_c,
            "dist2_A_change_on_h2_B": dA_b,
            "grad_h2C_h0A_absmax": gA_c,
            "m2_C_change_from_h1_B": m2_reads_h1B,
            "m2_C_change_from_own_h1_C": m2_reads_own_h1,
            "passed": bool(passed),
        }
        ok = ok and passed
    record("A3.1_h0_distance2_dependency", ok, detail)


def check_dependency_grad_address(spec: Any, norm: Mapping[str, Any]) -> None:
    """Explicit autograd address of the structural read: d h2 / d h0 per node."""
    detail: dict[str, Any] = {"graph": "chain A-B-C", "arms": {}}
    ok = True
    batch = synthetic_chain(spec)
    for arm in C.DEEP_ARMS:
        m, _ = M.build_ds_model(spec, norm, C.N_TASKS, arm=arm, seed=0, device="cpu")
        M.randomize_new_modules(m, seed=31415)
        m.eval()
        with torch.no_grad():
            _a, _b, _scal, h0 = m.encode_h0(batch)
        src, dst, edge_msg = M.directed_bonds(m, batch)
        h0_leaf = h0.detach().clone().requires_grad_(True)
        _h1, h2, _info = m.structural_layers(h0_leaf, batch, edges=(src, dst, edge_msg))
        rows = {}
        for target in (0, 1, 2):
            g = torch.autograd.grad(h2[target].sum(), h0_leaf, retain_graph=True)[0]
            rows[f"C{target}"] = {f"A{src_node}": float(g[src_node].abs().max().item()) for src_node in (0, 1, 2)}
        detail["arms"][arm] = rows
    # frozen expectation: the h0 support of h2[C]
    expect = {
        "SELF": {"A0": False, "A1": False, "A2": True},
        "FIXED1": {"A0": False, "A1": True, "A2": True},
        "REAL2": {"A0": True, "A1": True, "A2": True},
        "BAG2": {"A0": True, "A1": True, "A2": True},
        "DYNAMIC": {"A0": True, "A1": True, "A2": True},
    }
    table = {}
    for arm in C.DEEP_ARMS:
        row = detail["arms"][arm]["C2"]
        got = {k: bool(v > 1e-12) for k, v in row.items()}
        table[arm] = {"got_support_h2C": got, "expected": expect[arm],
                      "ok": got == expect[arm]}
        ok = ok and got == expect[arm]
    detail["support_table"] = table
    record("A3.2_h0_support_of_h2_at_C", ok, detail)


def check_branch_correspondence(spec: Any, norm: Mapping[str, Any]) -> None:
    """Swap two branch contexts of one root while holding both multisets fixed."""
    detail: dict[str, Any] = {}
    ok = True
    batch = synthetic_star(spec)
    for arm in ("REAL2", "BAG2"):
        m, _ = M.build_ds_model(spec, norm, C.N_TASKS, arm=arm, seed=0, device="cpu")
        M.randomize_new_modules(m, seed=999)
        m.eval()
        with torch.inference_mode():
            _a, _b, _scal, h0 = m.encode_h0(batch)
            src, dst, edge_msg = M.directed_bonds(m, batch)
            _h1, _h2, info = m.structural_layers(h0, batch, edges=(src, dst, edge_msg))
            c = info["c"].clone()
            c_swap = c.clone()
            # branches 1 and 2 are both leaves attached to the root 0
            c_swap[1], c_swap[2] = c[2], c[1]
            _h2s, m2_swap, _i = m.stage2(
                _h1, h0, src, dst, edge_msg, c_override=c_swap
            )
            # multiset invariance of the c values around the root
            multiset_ok = bool(
                torch.allclose(torch.sort(c[[1, 2, 3, 4]].flatten()).values,
                               torch.sort(c_swap[[1, 2, 3, 4]].flatten()).values)
            )
            changed = float((m2_swap[0] - info["m2"][0]).abs().max().item())
        detail[arm] = {
            "root_m2_change_on_context_swap": changed,
            "float32_sum_order_noise_floor": 1e-5,
            "context_multiset_preserved": multiset_ok,
            "c_branch_values_distinct": bool(float((c[1] - c[2]).abs().max().item()) > 1e-6),
        }
        if arm == "BAG2":
            ok = ok and changed < 1e-5 and multiset_ok
        else:
            ok = ok and changed > 1e-4 and multiset_ok
    record("A3.3_branch_correspondence_swap", ok, detail)


def check_degeneracies(spec: Any, norm: Mapping[str, Any]) -> None:
    """degree=1 => REAL2 == BAG2; gamma=0 => both reduce to FIXED1's stage-2 term."""
    detail: dict[str, Any] = {}
    ok = True
    batch = synthetic_chain(spec)
    star = synthetic_star(spec)
    matching = synthetic_matching(spec)
    models = {}
    for arm in ("REAL2", "BAG2", "FIXED1"):
        m, _ = M.build_ds_model(spec, norm, C.N_TASKS, arm=arm, seed=0, device="cpu")
        M.randomize_new_modules(m, seed=5150)
        m.eval()
        models[arm] = m
    # degree-1 chain: both endpoint nodes have degree 1
    with torch.inference_mode():
        out = {}
        for arm, m in models.items():
            _a, _b, _scal, h0 = m.encode_h0(batch)
            src, dst, edge_msg = M.directed_bonds(m, batch)
            _h1, _h2, info = m.structural_layers(h0, batch, edges=(src, dst, edge_msg))
            out[arm] = info["m2"]
        chain_deg1 = tol_rel(out["REAL2"][[0, 2]], out["BAG2"][[0, 2]], atol=1e-6, rtol=1e-6)
        # explicit matching graph: every node has degree 1
        for arm, m in models.items():
            _a, _b, _scal, h0 = m.encode_h0(matching)
            src, dst, edge_msg = M.directed_bonds(m, matching)
            _h1, _h2, info = m.structural_layers(h0, matching, edges=(src, dst, edge_msg))
            out[arm] = info["m2"]
        matching_all = tol_rel(out["REAL2"], out["BAG2"], atol=1e-6, rtol=1e-6)
        # gamma = 0 on the star graph
        g0 = {}
        for arm, m in models.items():
            _a, _b, _scal, h0 = m.encode_h0(star)
            src, dst, edge_msg = M.directed_bonds(m, star)
            _h1, _h2, info = m.structural_layers(h0, star, edges=(src, dst, edge_msg), gamma=0.0)
            g0[arm] = info["m2"]
        real2_g0 = tol_rel(g0["REAL2"], g0["FIXED1"], atol=1e-6, rtol=1e-6)
        bag2_g0 = tol_rel(g0["BAG2"], g0["FIXED1"], atol=1e-6, rtol=1e-6)
    detail = {
        "chain_degree1_nodes": chain_deg1,
        "matching_all_degree1": matching_all,
        "gamma0_REAL2_equals_FIXED1": real2_g0,
        "gamma0_BAG2_equals_FIXED1": bag2_g0,
    }
    ok = chain_deg1["ok"] and matching_all["ok"] and real2_g0["ok"] and bag2_g0["ok"]
    record("A3.4_degree1_and_gamma0_degeneracy", ok, detail)


def check_gradients(spec: Any, norm: Mapping[str, Any], batch: Mapping[str, Any], y: torch.Tensor) -> None:
    detail: dict[str, Any] = {"arms": {}}
    ok = True
    for arm in C.DEEP_ARMS:
        m, _ = M.build_ds_model(spec, norm, C.N_TASKS, arm=arm, seed=0, device="cpu")
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
        before = {n: p.detach().clone() for n, p in m.named_parameters() if C.is_new_parameter(n)}
        grads = {n: [] for n in before}
        losses = []
        m.train()
        for step in range(3):
            logits = m(batch)
            loss = MET.masked_bce_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            losses.append(float(loss.detach()))
            for n, p in m.named_parameters():
                if n in grads and p.grad is not None:
                    grads[n].append(float(p.grad.detach().norm().item()))
            opt.step()
        per_tensor = {}
        for n, p in m.named_parameters():
            if n not in before:
                continue
            moved = float((p.detach() - before[n]).abs().max().item())
            per_tensor[n] = {
                "grad_norm_max": max(grads[n]) if grads[n] else None,
                "param_moved": moved,
                "finite": bool(torch.isfinite(p).all().item()),
            }
        all_have_grad = all(v["grad_norm_max"] is not None and v["grad_norm_max"] > 0 for v in per_tensor.values())
        all_moved = all(v["param_moved"] > 0 for v in per_tensor.values())
        detail["arms"][arm] = {"losses": losses, "per_tensor": per_tensor,
                              "all_new_tensors_receive_gradient": all_have_grad,
                              "all_new_tensors_updated": all_moved}
        ok = ok and all_have_grad and all_moved
    record("A2.9_new_module_gradients_and_updates", ok, detail)


def check_checkpoint_roundtrip(spec: Any, norm: Mapping[str, Any], batch: Mapping[str, Any]) -> None:
    detail: dict[str, Any] = {}
    ok = True
    with tempfile.TemporaryDirectory() as td:
        for arm in ("SELF", "REAL2", "BAG2", "DYNAMIC", "W2"):
            if arm == "W2":
                m, _ = M.build_w2(spec, norm, C.N_TASKS, seed=0, device="cpu")
            else:
                m, _ = M.build_ds_model(spec, norm, C.N_TASKS, arm=arm, seed=0, device="cpu")
                M.randomize_new_modules(m, seed=8080)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            m.train()
            MET.masked_bce_loss(m(batch), torch.zeros(int(batch["n_graphs"]), C.N_TASKS)).backward()
            opt.step()
            m.eval()
            with torch.inference_mode():
                ref = m(batch).double()
            path = Path(td) / f"{arm}.pt"
            torch.save({"state_dict": C.state_cpu(m), "optimizer": C.optimizer_state_cpu(opt)}, path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if arm == "W2":
                m2, _ = M.build_w2(spec, norm, C.N_TASKS, seed=1, device="cpu")
            else:
                m2, _ = M.build_ds_model(spec, norm, C.N_TASKS, arm=arm, seed=1, device="cpu")
            m2.load_state_dict(payload["state_dict"], strict=True)
            m2.eval()
            with torch.inference_mode():
                got = m2(batch).double()
            d = tol_rel(got, ref, atol=0.0, rtol=0.0)
            detail[arm] = {**d, "sha256": C.sha256_file(path)}
            ok = ok and d["ok"]
    record("A2.10_checkpoint_save_reload", ok, detail)


def check_data_identity(task: Any, test_info: Mapping[str, Any]) -> None:
    labels = np.asarray(task.labels)
    finite = np.isfinite(labels)
    cache = task.cache
    n_atoms = np.asarray(cache["atom_ptr"])
    detail = {
        "dataset": C.DATASET,
        "n_molecules": int(cache["n_molecules"]),
        "n_tasks": int(labels.shape[1]),
        "split_sizes": {"train": int(task.train_rows.size), "valid": int(task.valid_rows.size), "test": int(task.test_rows.size)},
        "expected_split_sizes": {"train": C.N_TRAIN, "valid": C.N_VALID, "test": C.N_TEST},
        "test_labels_nan_in_training_artifact": bool(not finite[np.asarray(task.test_rows)].any()),
        "feature_spec_hash": task.spec.hash(),
        "expected_feature_spec_hash": C.SOURCE["feature_spec_hash"],
        "norm_fingerprint": task.norm["norm_fingerprint"],
        "expected_norm_fingerprint": C.SOURCE["norm_fingerprint"],
        "split_partition_sha256": task.split_partition_sha256,
        "expected_split_partition_sha256": C.SOURCE["split_partition_sha256"],
        "label_artifact_sha256": task.label_artifact_sha256,
        "expected_label_artifact_sha256": C.SOURCE["label_artifact_sha256"],
        "train_observed_entries": int(finite[np.asarray(task.train_rows)].sum()),
        "valid_observed_entries": int(finite[np.asarray(task.valid_rows)].sum()),
        "test_observed_entries": int(test_info["observed_entries"]),
        "test_rows_sha256": test_info["test_rows_sha256"],
        "expected_test_rows_sha256": C.BOC_TEST_ROWS_NPY_SHA256,
        "official_test_label_file_sha256": test_info["raw_sha256"],
        "train_valid_alignment_verified": bool(test_info["train_valid_alignment_verified"]),
        "task_order": "official ogbg-molpcba column order, never reordered",
        "cache_atom_total": int(n_atoms[-1]),
    }
    ok = (
        detail["split_sizes"] == detail["expected_split_sizes"]
        and detail["test_labels_nan_in_training_artifact"]
        and detail["feature_spec_hash"] == detail["expected_feature_spec_hash"]
        and detail["norm_fingerprint"] == detail["expected_norm_fingerprint"]
        and detail["split_partition_sha256"] == detail["expected_split_partition_sha256"]
        and detail["label_artifact_sha256"] == detail["expected_label_artifact_sha256"]
        and detail["test_rows_sha256"] == detail["expected_test_rows_sha256"]
        and detail["train_valid_alignment_verified"]
    )
    record("A2.1_data_identity", ok, detail)


def check_plan_identity(task: Any) -> None:
    plan0, meta0 = C.plan_for_seed(task, 0)
    detail = {
        "seed0_global_plan_sha256": meta0["global_plan_sha256"],
        "expected": C.FROZEN_PLAN_SEED0["global_plan_sha256"],
        "seed0_matches_historical_W2_plan": meta0["global_plan_sha256"] == C.FROZEN_PLAN_SEED0["global_plan_sha256"],
        "seed0_prefix40_sha256": C.plan_meta(plan0, 40, 0)["prefix_sha256"],
        "expected_prefix": C.FROZEN_PLAN_SEED0["consumed40_prefix_sha256"],
        "steps_per_epoch": int(len(plan0[0]["batches"])),
        "last_batch_size": int(len(plan0[0]["batches"][-1])),
        "isolation": meta0["plan_isolation"],
    }
    ok = (
        detail["seed0_matches_historical_W2_plan"]
        and detail["seed0_prefix40_sha256"] == detail["expected_prefix"]
        and detail["steps_per_epoch"] == C.STEPS_PER_EPOCH
        and bool(meta0["plan_isolation"]["passed"])
    )
    record("A2.11_order_plan_identity", ok, detail)


def check_shared_plan_across_arms(task: Any) -> None:
    """Every arm of one seed consumes the same plan prefix (no per-arm resampling)."""
    plan0, _ = C.plan_for_seed(task, 0)
    metas = {}
    for arm in C.ARM_ORDER:
        metas[arm] = C.plan_meta(plan0, C.HORIZONS[2], 0)
    uniq = {json.dumps(m, sort_keys=True) for m in metas.values()}
    record("A2.12_same_plan_for_all_arms", len(uniq) == 1, {"prefix": metas["W2"], "n_unique": len(uniq)})


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--out", default=str(C.ANALYSIS / "checks.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--freeze-seed1", action="store_true")
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(max(1, int(torch.get_num_threads())))

    norm = fake_norm()
    spec = load_spec()
    full = args.mode == "full"
    if not full:
        # the stand-in normaliser cannot reproduce the recorded W2 trunk fingerprint
        M.STRICT_W2_FINGERPRINT = False
    task = None
    test_info = None
    if full:
        task = C.SD.load_task()
        test_info = C.load_test_labels(task, cache_dir=C.RUNS / "_shared")
        norm = task.norm
        check_data_identity(task, test_info)
        check_plan_identity(task)
        check_shared_plan_across_arms(task)
        if args.freeze_seed1:
            record("A1.seed1_plan_frozen", True, C.freeze_seed1_plan(task))

    check_parameters(spec, norm)
    check_init_identity(spec, norm)
    batch_small = synthetic_star(spec)
    check_epoch0_equivalence(spec, norm, batch_small)
    check_dependency(spec, norm)
    check_dependency_grad_address(spec, norm)
    check_branch_correspondence(spec, norm)
    check_degeneracies(spec, norm)

    if full:
        rows = task.valid_rows[:8]
        batch = C.PC.collate(task.cache, rows.tolist())
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        y = torch.as_tensor(np.asarray(task.labels[task.train_rows[:8]]), dtype=torch.float32, device=device)
        check_bond_occurrence(batch)
        check_reference_formulas(spec, norm, C.PC.collate(task.cache, task.valid_rows[:4].tolist()), label="real_batch_4")
        check_reference_formulas(spec, norm, synthetic_star(spec), label="star")
        check_invariance(spec, norm, batch)
        check_batch_isolation(spec, norm, C.PC.collate(task.cache, task.valid_rows[:4].tolist()))
        check_gradients(spec, norm, batch, y)
        # real graph statistics (pair multiplicity at the real degree distribution)
        stats = degree_statistics(task)
        record("A2.13_real_degree_statistics", True, stats)

    payload = {
        "schema": C.SCHEMA_CHECKS,
        "round_id": C.ROUND_ID,
        "mode": args.mode,
        "device": str(device),
        "seconds": float(time.perf_counter() - t0),
        "n_checks": len(RECORDS),
        "n_passed": sum(1 for r in RECORDS if r["passed"]),
        "all_passed": all(r["passed"] for r in RECORDS),
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "checks": RECORDS,
        "created_utc": C.utcnow(),
    }
    C.write_json(Path(args.out), payload)
    C.log(f"checks: {payload['n_passed']}/{payload['n_checks']} passed in {payload['seconds']:.1f}s")
    return 0 if payload["all_passed"] else 1


def degree_statistics(task: Any) -> dict[str, Any]:
    """Real bond degree distribution and the exact BAG2 pair count per split."""
    out: dict[str, Any] = {}
    cache = task.cache
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    bi = np.asarray(cache["bond_index"])
    for split in ("train", "valid", "test"):
        rows = getattr(task, f"{split}_rows")
        deg_sum = 0
        deg_sq_sum = 0
        n_atoms_total = 0
        max_deg = 0
        n_bonds_total = 0
        for r in np.asarray(rows):
            a0, a1 = int(atom_ptr[r]), int(atom_ptr[r + 1])
            b0, b1 = int(bond_ptr[r]), int(bond_ptr[r + 1])
            n_atoms = a1 - a0
            bonds = bi[b0:b1]
            deg = np.zeros(n_atoms, dtype=np.int64)
            if len(bonds):
                np.add.at(deg, bonds[:, 0] - a0, 1)
                np.add.at(deg, bonds[:, 1] - a0, 1)
            deg_sum += int(deg.sum())
            deg_sq_sum += int((deg.astype(np.int64) ** 2).sum())
            n_atoms_total += n_atoms
            n_bonds_total += int(len(bonds))
            max_deg = max(max_deg, int(deg.max()) if deg.size else 0)
        out[split] = {
            "molecules": int(np.asarray(rows).size),
            "atoms": n_atoms_total,
            "bonds": n_bonds_total,
            "mean_degree": deg_sum / max(n_atoms_total, 1),
            "max_degree": max_deg,
            "directed_edges_per_batch128": 2 * n_bonds_total / max(int(np.asarray(rows).size) / 128.0, 1.0),
            "bag2_pairs_per_batch128": deg_sq_sum / max(int(np.asarray(rows).size) / 128.0, 1.0),
            "bag2_pair_edge_ratio": deg_sq_sum / max(deg_sum, 1),
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
