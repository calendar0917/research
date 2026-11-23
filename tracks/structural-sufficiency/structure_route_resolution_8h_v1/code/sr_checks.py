"""Pre-training checks for the route-resolution sprint.

Runs on CPU (some checks build float64 models).  Every check records the real
error, not only a PASS flag.  The formal-run gate is ``all_pass``.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

import sr_common as C
import sr_data as D
import sr_eval as EV
import sr_fresponse as FR
import sr_gnn as G
import sr_model as M
import sr_struct as ST
import sr_tensor as T

import rice_energy as RE
import rice_model as RM
import rice_structure as RS

CHECKS: list[dict[str, Any]] = []


def record(cid: str, group: str, description: str, fn: Callable[[], dict[str, Any]],
           *, required: bool = True) -> None:
    t0 = time.time()
    row: dict[str, Any] = {"id": cid, "group": group, "description": description,
                           "required": bool(required)}
    try:
        payload = fn()
        row.update(payload)
        row["status"] = "PASS" if payload.get("pass", True) else "FAIL"
    except Exception as exc:  # noqa: BLE001 - a failed check is data
        row["status"] = "FAIL"
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = float(time.time() - t0)
    CHECKS.append(row)


def _synth_mol(atom_feat: np.ndarray, bonds: Sequence[tuple[int, int, tuple[int, int, int]]],
               row: int = 0) -> dict[str, Any]:
    atom_feat = np.asarray(atom_feat, np.int64)
    n_atoms = int(atom_feat.shape[0])
    bond_u = np.asarray([b[0] for b in bonds], np.int64)
    bond_v = np.asarray([b[1] for b in bonds], np.int64)
    bond_feat = np.asarray([b[2] for b in bonds], np.int64).reshape(-1, 3)
    return {"row": row, "n_atoms": n_atoms, "n_bonds": int(len(bonds)),
            "atom_feat": atom_feat, "bond_feat": bond_feat,
            "bond_u": bond_u, "bond_v": bond_v, "smiles": "",
            "graph_scalars": np.zeros(8, np.float64)}


def _star4() -> Any:
    atom = np.zeros((4, 9), np.int64)
    atom[:, 0] = [6, 6, 6, 6]
    atom[:, 3] = 1
    bonds = [(0, 1, (0, 0, 0)), (0, 2, (0, 0, 0)), (0, 3, (0, 0, 0))]
    return RS.build_molecule(_synth_mol(atom, bonds))


def _smallest_molecule(bundle: Any) -> tuple[int, Any]:
    rows = D.all_rows(bundle).tolist()
    sizes = [(int(D.view(bundle, r)["n_atoms"]), r) for r in rows[:400]]
    _n, row = min(sizes)
    struct = ST.build_structures(bundle, [row])[row]
    return row, struct


def _batch_ef(rows: Sequence[int], structs: Mapping[int, Any], raw: Mapping[str, Any],
              registry: ST.TypeRegistry, device: Any = "cpu") -> dict[str, Any]:
    batch = T.collate_ef(list(rows), structs, raw, device)
    return batch


def _inclusion_kappa(psi: Callable[[frozenset], float], n: int,
                     root: int) -> dict[frozenset, float]:
    """Full subset-lattice Mobius transform restricted to v in T."""
    order = list(range(n))
    kap: dict[frozenset, float] = {}
    subsets = []
    for k in range(1, n + 1):
        for comb in itertools.combinations(order, k):
            if root in comb:
                subsets.append(frozenset(comb))
    subsets.sort(key=lambda s: (len(s), sorted(s)))
    for S in subsets:
        total = 0.0
        for T in subsets:
            if not T <= S:
                continue
            total += ((-1.0) ** (len(S) - len(T))) * psi(T)
        kap[S] = total
    return kap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR / "checks.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args()

    C.assert_pins()
    cfg = C.load_config()
    bundles: dict[str, Any] = {}
    fits: dict[str, Any] = {}

    def b0() -> Any:
        if "D0" not in bundles:
            bundles["D0"] = D.load_bundle("D0")
        return bundles["D0"]

    # -- A. data identity and label isolation -------------------------------
    def a1() -> dict[str, Any]:
        out = {}
        for split in ("D0", "D1"):
            b = D.load_bundle(split)
            bundles[split] = b
            ident = D.split_identity(b)
            assert ident["n_fit"] == 2856 and ident["n_eval"] == 504
            assert not np.intersect1d(b.fit_rows, b.eval_rows).size
            assert np.union1d(b.fit_rows, b.eval_rows).size == 3360
            assert np.isfinite(b.labels[np.union1d(b.fit_rows, b.eval_rows)]).all()
            assert not np.isfinite(b.labels[b.official_valid]).any()
            out[split] = {k: ident[k] for k in
                          ("identity_sha256", "n_fit", "n_eval", "scaler")}
        assert out["D0"]["identity_sha256"] == C.D0_IDENTITY
        assert out["D1"]["identity_sha256"] == C.D1_IDENTITY
        return {"pass": True, "identities": out}

    record("A1", "data", "D0/D1 identity, sizes, coverage and Valid-label isolation", a1)

    def a2() -> dict[str, Any]:
        b0_ = b0()
        b1 = bundles.get("D1") or D.load_bundle("D1")
        bundles["D1"] = b1
        same_fit = bool(np.array_equal(b0_.fit_rows, b1.fit_rows))
        same_eval = bool(np.array_equal(b0_.eval_rows, b1.eval_rows))
        scaler_diff = abs(float(b0_.scaler["std"]) - float(b1.scaler["std"]))
        assert not same_fit and not same_eval
        assert scaler_diff > 1e-9
        return {"pass": True, "fit_rows_equal": same_fit,
                "eval_rows_equal": same_eval, "scaler_std_abs_diff": scaler_diff}

    record("A2", "data", "D0 and D1 differ; per-split Fit-only scaler is recomputed", a2)

    # -- B. historical energy replay ---------------------------------------
    if not args.skip_replay:
        def b1() -> dict[str, Any]:
            hist_dir = C.RICE_DIR / "results" / "RICE_CUMULANT_S0"
            manifest = C.read_json(hist_dir / "manifest.json")
            blob = torch.load(hist_dir / "state_best.pt", map_location="cpu",
                              weights_only=False)
            b = b0()
            rows = np.union1d(b.fit_rows, b.eval_rows).tolist()
            structs = ST.build_structures(b, rows)
            atom_dims_, bond_dims_ = D.feature_dims()
            model, info = M.build_model("E_MOB", atom_dims_, bond_dims_, 0,
                                       D.norm_of(b), cfg)
            state = {f"rice.{k}": v for k, v in blob["state"].items()}
            model.load_state_dict(state, strict=True)
            pred, _ = EV.predict_raw(model, "E_MOB", b.eval_rows, structs, b.raw,
                                     "cpu", 256, b.scaler)
            got = C.rmse(pred, b.labels[b.eval_rows])
            want = float(manifest["best_dev_rmse"])
            err = abs(got - want)
            return {"pass": err < 2e-4, "historical_dev_rmse": want,
                    "replayed_dev_rmse": got, "max_abs_error": err,
                    "tolerance": 2e-4,
                    "note": "CPU float32 replay of the frozen GPU checkpoint"}
        record("B1", "energy", "frozen RICE_CUMULANT D0 checkpoint replays its Dev RMSE", b1)

    b = b0()
    row, toy = _smallest_molecule(b)
    atom_dims, bond_dims = D.feature_dims()
    reg = ST.build_registry([toy], atom_dims, bond_dims)
    toy_batch = T.collate_ef([row], {row: toy}, b.raw, "cpu")

    def c1() -> dict[str, Any]:
        model, _ = M.build_model("E_MOB", atom_dims, bond_dims, 0, D.norm_of(b), cfg)
        model.double()
        out = model(toy_batch, return_aux=True)
        worst = 0.0
        n = 0
        for size in (2, 3, 4):
            atoms = toy.atoms[size]
            kappa = out["kappa"][size]
            for i in range(atoms.shape[0]):
                root = int(atoms[i, 0])
                total = out["u_flat"].double()[root]
                # phi == sum of stored kappa over connected root-containing
                # proper subsets, using the model's own sub2/sub3 maps
                expected = out["kappa"][1][root].clone() + out["kappa"][size][i]
                if size >= 3:
                    for j in range(toy.sub2[size].shape[1]):
                        hit = int(toy.sub2[size][i, j])
                        if hit >= 0:
                            expected = expected + out["kappa"][2][hit]
                if size == 4:
                    for j in range(toy.sub3[4].shape[1]):
                        hit = int(toy.sub3[4][i, j])
                        if hit >= 0:
                            expected = expected + out["kappa"][3][hit]
                worst = max(worst, float((expected - out["phi"][size][i]).abs().max()))
                n += 1
        return {"pass": worst <= 1e-9, "max_abs_error": worst, "supports": n}
    record("C1", "mobius", "float64 Phi(S,v) == sum of connected kappa on every "
                           "stored support (independent subset sum)", c1)

    def c2() -> dict[str, Any]:
        model, _ = M.build_model("E_MOB", atom_dims, bond_dims, 0, D.norm_of(b), cfg)
        model.double()
        out = model(toy_batch, return_aux=True)
        u = out["u"].detach().numpy()
        j = out["j"].detach().numpy()
        bond_list = [(int(a), int(bb), int(k)) for k, (a, bb) in
                     enumerate(zip(toy.bond_u, toy.bond_v))]
        worst_phi = 0.0
        worst_kappa = 0.0
        n = 0
        for size in (2, 3, 4):
            atoms = toy.atoms[size]
            for i in range(atoms.shape[0]):
                root = int(atoms[i, 0])
                support = [int(x) for x in atoms[i] if x >= 0]
                ref = RE.brute_force_kappa(u, j, toy.n_atoms, bond_list, root,
                                           list(range(toy.n_atoms)))
                key = frozenset(support)
                worst_phi = max(worst_phi,
                                float(np.abs(ref["phi"][key].reshape(-1)
                                             - out["phi"][size][i].detach().numpy()).max()))
                worst_kappa = max(worst_kappa,
                                  float(np.abs(ref["kappa"][key].reshape(-1)
                                               - out["kappa"][size][i].detach().numpy()).max()))
                n += 1
        return {"pass": max(worst_phi, worst_kappa) <= 1e-9,
                "max_phi_error": worst_phi, "max_kappa_error": worst_kappa,
                "supports": n, "molecule_row": int(row)}
    record("C2", "mobius", "vectorised float64 Phi/kappa == brute force (all stored supports)",
           c2)

    def c3() -> dict[str, Any]:
        model, _ = M.build_model("E_MOB", atom_dims, bond_dims, 0, D.norm_of(b), cfg)
        model.double()
        out = model(toy_batch, return_aux=True)
        ref = RE.brute_force_kappa(out["u"].detach().numpy(), out["j"].detach().numpy(),
                                   toy.n_atoms,
                                   [(int(a), int(bb), int(k)) for k, (a, bb) in
                                    enumerate(zip(toy.bond_u, toy.bond_v))],
                                   0, list(range(toy.n_atoms)))
        worst = 0.0
        n = 0
        for S, value in ref["kappa"].items():
            if len(S) >= 2 and not RE.reference_is_connected(
                    sorted(S), [(int(a), int(bb), int(k)) for k, (a, bb) in
                                enumerate(zip(toy.bond_u, toy.bond_v))]):
                worst = max(worst, float(np.abs(value).max()))
                n += 1
        return {"pass": worst <= 1e-9, "max_abs_kappa_disconnected": worst,
                "disconnected_supports": n}
    record("C3", "mobius", "disconnected root-anchored supports have zero kappa", c3)

    def c4() -> dict[str, Any]:
        u = np.asarray([[0.0, 0.31], [0.2, -0.17], [-0.1, 0.47], [0.23, -0.29]],
                       np.float64)[:, None, :]
        j = np.asarray([[0.8, -0.35, 0.45], [0.42, -0.19, 0.63],
                        [0.67, -0.28, 0.51]], np.float64)[:, None, :]
        bond_list = [(0, 1, 0), (1, 2, 1), (1, 3, 2)]
        leaf = RE.brute_force_kappa(u, j, 4, bond_list, 0, [0, 1, 2, 3])
        center = RE.brute_force_kappa(u, j, 4, bond_list, 1, [0, 1, 2, 3])
        got_leaf = leaf["kappa"][frozenset({0, 1, 2, 3})].reshape(-1)
        got_center = center["kappa"][frozenset({0, 1, 2, 3})].reshape(-1)
        want_leaf = np.asarray([0.009789413752, 0.000340153203])
        err = float(np.abs(got_leaf - want_leaf).max())
        center_err = float(np.abs(got_center).max())
        return {"pass": err <= 1e-9 and center_err <= 1e-9,
                "leaf_kappa4": got_leaf.tolist(), "expected_leaf": want_leaf.tolist(),
                "max_abs_error": err, "center_kappa4_abs_max": center_err,
                "edges": [[0, 1], [1, 2], [1, 3]]}
    record("C4", "mobius", "frozen 4-star counterexample (leaf order-4 non-zero, centre zero)",
           c4)

    def c5() -> dict[str, Any]:
        model, _ = M.build_model("E_MOB", atom_dims, bond_dims, 0, D.norm_of(b), cfg)
        model.double()
        out = model(toy_batch, return_aux=True)
        root = 0
        # order >= 2 values must be unchanged when only the root unary moves
        u = out["u"].detach().clone()
        j = out["j"].detach().clone()
        u[root] += 0.37
        sizes = (2, 3, 4)
        batch2 = dict(toy_batch)
        phi2 = {s: RE.compute_phi(u, j, batch2[f"atoms_{s}"], batch2[f"bonds_{s}"],
                                  batch2[f"bslot_{s}"]) for s in sizes}
        kappa2 = RE.mobius(u.reshape(toy.n_atoms, model.rice.kappa_dim), phi2,
                           {s: batch2[f"atoms_{s}"] for s in sizes},
                           {s: batch2[f"sub2_{s}"] for s in (3, 4)},
                           batch2["sub3_4"])
        worst = 0.0
        n = 0
        for s in sizes:
            mask = batch2[f"atoms_{s}"][:, 0] == root
            if mask.any():
                worst = max(worst, float((kappa2[s][mask] - out["kappa"][s][mask])
                                         .abs().max()))
                n += int(mask.sum())
        return {"pass": worst <= 1e-9, "max_abs_change_order_ge2": worst,
                "supports": n}
    record("C5", "mobius", "root unary change leaves every order>=2 kappa fixed", c5)

    # -- D. free response --------------------------------------------------
    def d1() -> dict[str, Any]:
        model, _ = M.build_model("F_MOB", atom_dims, bond_dims, 0, D.norm_of(b), cfg)
        model.set_registry(FR.registry_tensors(reg, "cpu"))
        model.double()
        out = model(toy_batch, return_aux=True)
        psi_hi = out["psi"]
        worst = 0.0
        n = 0
        for size in (2, 3, 4):
            atoms = toy.atoms[size]
            for i in range(atoms.shape[0]):
                root = int(atoms[i, 0])
                expected = out["psi1"][root].clone() + out["kappa"][size][i]
                for j in range(toy.sub2[size].shape[1] if size >= 3 else 0):
                    hit = int(toy.sub2[size][i, j])
                    if hit >= 0:
                        expected = expected + out["kappa"][2][hit]
                if size == 4:
                    for j in range(toy.sub3[4].shape[1]):
                        hit = int(toy.sub3[4][i, j])
                        if hit >= 0:
                            expected = expected + out["kappa"][3][hit]
                worst = max(worst, float((expected - psi_hi[size][i]).abs().max()))
                n += 1
        return {"pass": worst <= 1e-9, "max_abs_error": worst, "supports": n,
                "singleton_definition": "psi({v},v) is the size-1 response"}
    record("D1", "free", "float64 psi == sum of connected kappa on every stored support", d1)

    def d2() -> dict[str, Any]:
        st = _star4()
        reg2 = ST.build_registry([st], atom_dims, bond_dims)
        batch = T.collate_ef([0], {0: st}, None, "cpu")
        model, _ = M.build_model("F_RAW", atom_dims, bond_dims, 0, D.norm_of(b), cfg)
        model.set_registry(FR.registry_tensors(reg2, "cpu"))
        # explicit positive construction: psi[0] = SiLU(|S|)
        with torch.no_grad():
            for module in (model.free.atom_projection, model.free.bond_projection):
                module.weight.zero_()
                module.bias.zero_()
            model.free.support_mlp[0].weight.zero_()
            model.free.support_mlp[0].bias.zero_()
            model.free.support_mlp[0].weight[0, 112:116] = 1.0
            model.free.support_mlp[2].weight.zero_()
            model.free.support_mlp[2].bias.zero_()
            model.free.support_mlp[2].weight[0, 0] = 1.0
        out = model(batch, return_aux=True)
        psi_values = {}
        for size in (1, 2, 3, 4):
            vals = out["psi1"] if size == 1 else out["psi"][size]
            psi_values[size] = float(vals[:, 0].mean())
        ideal = {k: float(k / (1.0 + math.exp(-k))) for k in (1, 2, 3, 4)}
        psi_err = max(abs(psi_values[k] - ideal[k]) for k in ideal)
        # centre-rooted 4-star and 3-star kappa
        kap = out["kappa"]
        a4 = batch["atoms_4"]
        centre4 = [i for i in range(a4.shape[0])
                   if int(a4[i, 0]) == 0 and set(int(x) for x in a4[i] if x >= 0) == {0, 1, 2, 3}]
        a3 = batch["atoms_3"]
        centre3 = [i for i in range(a3.shape[0])
                   if int(a3[i, 0]) == 0 and set(int(x) for x in a3[i] if x >= 0) == {0, 1, 2}]
        k4 = float(kap[4][centre4[0], 0])
        k3 = float(kap[3][centre3[0], 0])
        return {"pass": psi_err <= 2e-6 and abs(k4 - (-0.091388092013)) <= 2e-5
                and abs(k3 - 0.065592647186) <= 2e-5,
                "psi_values": psi_values, "psi_ideal": ideal,
                "psi_max_abs_error": psi_err,
                "centre_kappa4": k4, "expected_kappa4": -0.091388092013,
                "centre_kappa3": k3, "expected_kappa3": 0.065592647186}
    record("D2", "free", "explicit-weight positive control psi=SiLU(|S|), centre star kappa3/4",
           d2)

    def d3() -> dict[str, Any]:
        # extension psi(S,v)=psi(C_v(S),v): disconnected kappa is exactly zero,
        # and the connected-only recursion equals the full subset transform.
        edges = [(0, 1), (1, 2), (3, 4)]
        n = 6
        root = 0

        def comp(S: frozenset) -> frozenset:
            seen = {root}
            if root not in S:
                return frozenset()
            stack = [root]
            while stack:
                x = stack.pop()
                for a, b in edges:
                    if a == x and b in S and b not in seen:
                        seen.add(b)
                        stack.append(b)
                    if b == x and a in S and a not in seen:
                        seen.add(a)
                        stack.append(a)
            return frozenset(seen)

        def psi(S: frozenset) -> float:
            if root not in S:
                return 0.0
            c = sorted(comp(S))
            return float(sum(math.sin(1.7 * (i + 1)) for i in c) / (1.0 + 0.1 * len(c)))

        kap = _inclusion_kappa(psi, n, root)
        disconnected = [S for S in kap if len(S) >= 2 and comp(S) != S]
        worst = max((abs(kap[S]) for S in disconnected), default=0.0)
        connected = sorted([S for S in kap if comp(S) == S],
                           key=lambda s: (len(s), sorted(s)))
        rec: dict[frozenset, float] = {}
        for S in connected:
            total = psi(S)
            for T in connected:
                if T < S:
                    total -= rec[T]
            rec[S] = total
        rec_err = max(abs(rec[S] - kap[S]) for S in connected)
        return {"pass": worst <= 1e-12 and rec_err <= 1e-12,
                "max_abs_disconnected_kappa": worst,
                "connected_vs_full_max_abs_error": rec_err,
                "disconnected_sets": len(disconnected)}
    record("D3", "free", "disconnected extension: every disconnected kappa is zero",
           d3)

    # -- E. initialisation contracts ---------------------------------------
    def e1() -> dict[str, Any]:
        models = {}
        infos = {}
        for arm in ("E_RAW", "E_MOB", "F_RAW", "F_MOB"):
            models[arm], infos[arm] = M.build_model(arm, atom_dims, bond_dims, 3,
                                                    D.norm_of(b), cfg)
        def state_hash(model: Any) -> str:
            return C.module_hashes({"m": model})
        e_raw = state_hash(models["E_RAW"])
        e_mob = state_hash(models["E_MOB"])
        f_raw = state_hash(models["F_RAW"])
        f_mob = state_hash(models["F_MOB"])
        # F models have different keys by construction; compare paired arms only
        assert e_raw == e_mob, "E_RAW and E_MOB initial states differ"
        assert f_raw == f_mob, "F_RAW and F_MOB initial states differ"
        common = ("atom_emb", "bond_emb", "order_norm", "sentence_mlp",
                  "readout_mlp", "beta")
        shared_ok = all(infos["E_MOB"]["init_hashes"][k]
                        == infos["F_MOB"]["init_hashes"][k] for k in common)
        assert shared_ok, "E/F common head initialisation differs"
        return {"pass": True, "e_pair_equal": True, "f_pair_equal": True,
                "common_head_equal": shared_ok,
                "common_head_modules": list(common),
                "params": {a: infos[a]["parameters"]["total"] for a in infos}}
    record("E1", "init", "RAW/MOB initial state identical; E/F common head identical", e1)

    def e2() -> dict[str, Any]:
        # BIG response comes from the same-seed SMALL initial state; the big
        # head is identical between E_BIG and F_BIG.
        big = {}
        info = {}
        for arm in ("E_MOB_BIG", "F_MOB_BIG"):
            big[arm], info[arm] = M.build_family_models(
                cfg, atom_dims, bond_dims, 5, D.norm_of(b), arm)
        small = {}
        for arm in ("E_MOB", "F_MOB"):
            small[arm], _ = M.build_model(arm, atom_dims, bond_dims, 5,
                                          D.norm_of(b), cfg)
        def compare(dst: Any, src: Any, prefixes: Sequence[str]) -> bool:
            a = dst.state_dict()
            s = src.state_dict()
            for k, v in s.items():
                if any(k.startswith(p) for p in prefixes):
                    if k not in a or a[k].shape != v.shape or not torch.equal(a[k], v):
                        return False
            return True
        response_ok = (compare(big["E_MOB_BIG"], small["E_MOB"],
                               ("rice.atom_emb.", "rice.bond_emb.", "rice.atom_mlp.",
                                "rice.bond_mlp."))
                       and compare(big["F_MOB_BIG"], small["F_MOB"],
                                   ("rice.atom_emb.", "rice.bond_emb.", "free.")))
        head_ok = (big["F_MOB_BIG"].state_dict()["rice.sentence_mlp.0.weight"].shape
                   == big["E_MOB_BIG"].state_dict()["rice.sentence_mlp.0.weight"].shape)
        same_head = all(torch.equal(big["E_MOB_BIG"].state_dict()[k],
                                    big["F_MOB_BIG"].state_dict()[k])
                        for k in big["E_MOB_BIG"].state_dict()
                        if k.startswith(("rice.order_norm.", "rice.sentence_mlp.",
                                         "rice.readout_mlp."))
                        or k in ("rice.beta", "rice.graph_mean", "rice.graph_scale"))
        params_ok = (info["E_MOB_BIG"]["parameters"]["total"] == 395438
                     and info["F_MOB_BIG"]["parameters"]["total"] == 396118)
        return {"pass": response_ok and head_ok and same_head and params_ok,
                "response_from_small": response_ok, "big_head_equal": same_head,
                "params": {a: info[a]["parameters"]["total"] for a in info},
                "expected": {"E_MOB_BIG": 395438, "F_MOB_BIG": 396118}}
    record("E2", "init", "BIG arms: response copied from SMALL, shared BIG head, "
                           "395438/396118 parameters", e2)

    # -- F. invariance ------------------------------------------------------
    def _permuted(rows: Sequence[int], bundle: Any, rng: np.random.Generator) -> dict[int, Any]:
        views = []
        for r in rows:
            v = D.view(bundle, int(r))
            n = int(v["n_atoms"])
            m = int(v["n_bonds"])
            perm = rng.permutation(n)
            border = rng.permutation(m)
            inv = np.empty(n, np.int64)
            inv[perm] = np.arange(n)
            atom = np.asarray(v["atom_feat"], np.int64)[perm]
            u = inv[np.asarray(v["bond_u"], np.int64)][border]
            w = inv[np.asarray(v["bond_v"], np.int64)][border]
            bf = np.asarray(v["bond_feat"], np.int64)[border]
            mol = {"row": int(r), "n_atoms": n, "n_bonds": m, "atom_feat": atom,
                   "bond_feat": bf, "bond_u": u, "bond_v": w, "smiles": "",
                   "graph_scalars": bundle.graph_scalars[int(r)]}
            views.append(mol)
        return {int(m["row"]): RS.build_molecule(m) for m in views}

    def f1() -> dict[str, Any]:
        rows = D.all_rows(b)[:6].tolist()
        structs = ST.build_structures(b, rows)
        rng = np.random.default_rng(11)
        alt = _permuted(rows, b, rng)
        out = {}
        for arm in ("E_MOB", "F_MOB"):
            reg_a = ST.build_registry([structs[r] for r in rows], atom_dims, bond_dims)
            reg_b = ST.build_registry([alt[r] for r in rows], atom_dims, bond_dims)
            model, _ = M.build_model(arm, atom_dims, bond_dims, 2, D.norm_of(b), cfg)
            model.eval()
            preds = []
            for reg_use, st_use in ((reg_a, structs), (reg_b, alt)):
                if arm[0] == "F":
                    model.set_registry(FR.registry_tensors(reg_use, "cpu"))
                batch = T.collate_ef(rows, st_use, b.raw, "cpu")
                with torch.no_grad():
                    preds.append(model(batch)["yhat_std"].double().numpy())
            out[arm] = float(np.abs(preds[0] - preds[1]).max())
        return {"pass": max(out.values()) <= 2e-5, "max_abs_prediction_delta": out}
    record("F1", "invariance", "atom relabelling + bond reordering leaves E/F predictions",
           f1)

    def f2() -> dict[str, Any]:
        rows = D.all_rows(b)[:6].tolist()
        structs = ST.build_structures(b, rows)
        reg = ST.build_registry([structs[r] for r in rows], atom_dims, bond_dims)
        out = {}
        for arm in ("E_MOB", "F_MOB"):
            model, _ = M.build_model(arm, atom_dims, bond_dims, 4, D.norm_of(b), cfg)
            if arm[0] == "F":
                model.set_registry(FR.registry_tensors(reg, "cpu"))
            model.eval()
            together = T.collate_ef(rows, structs, b.raw, "cpu")
            with torch.no_grad():
                pred_all = model(together)["yhat_std"].double().numpy()
            single = []
            for r in rows:
                batch = T.collate_ef([r], structs, b.raw, "cpu")
                with torch.no_grad():
                    single.append(float(model(batch)["yhat_std"][0]))
            out[arm] = float(np.abs(pred_all - np.asarray(single)).max())
        return {"pass": max(out.values()) <= 2e-5, "max_abs_batch_delta": out}
    record("F2", "invariance", "batch composition does not change E/F predictions", f2)

    # -- G. range controls -------------------------------------------------
    g_rows = D.all_rows(b)[:6].tolist()
    g_structs = ST.build_structures(b, g_rows)
    for r in g_rows:
        G.prepare_ranges(g_structs[r])
    g_model, g_info = G.build_model("G_FULL5", atom_dims, bond_dims, 7,
                                    D.norm_of(b), cfg)
    g_model.eval()

    def g1() -> dict[str, Any]:
        full = G.collate(g_rows, g_structs, "FULL", "cpu")
        e2 = G.collate(g_rows, g_structs, "EGO2", "cpu")
        e3 = G.collate(g_rows, g_structs, "EGO3", "cpu")
        res = {}
        with torch.no_grad():
            for t in (1, 2):
                a = g_model(full, n_layers=t, return_h=True)
                c = g_model(e2, n_layers=t, return_h=True)
                res[f"ego2_t{t}_root"] = float((a["roots"] - c["roots"]).abs().max())
                res[f"ego2_t{t}_pred"] = float((a["yhat_std"] - c["yhat_std"]).abs().max())
            for t in (1, 2, 3):
                a = g_model(full, n_layers=t, return_h=True)
                c = g_model(e3, n_layers=t, return_h=True)
                res[f"ego3_t{t}_root"] = float((a["roots"] - c["roots"]).abs().max())
                res[f"ego3_t{t}_pred"] = float((a["yhat_std"] - c["yhat_std"]).abs().max())
        return {"pass": max(res.values()) <= 2e-5, "max_abs": res,
                "note": "root vectors and readout compared; not the complete 5-layer nets"}
    record("G1", "range", "t<=2 FULL==EGO2 and t<=3 FULL==EGO3 (root and prediction)", g1)

    def g2() -> dict[str, Any]:
        full = G.collate(g_rows, g_structs, "FULL", "cpu")
        e2 = G.collate(g_rows, g_structs, "EGO2", "cpu")
        a = g_model(full, n_layers=2)
        c = g_model(e2, n_layers=2)
        ga = torch.autograd.grad((a["yhat_std"] - 1).pow(2).sum(),
                                 list(g_model.parameters()), retain_graph=True,
                                 allow_unused=True)
        gc = torch.autograd.grad((c["yhat_std"] - 1).pow(2).sum(),
                                 list(g_model.parameters()), allow_unused=True)
        worst = max(float((x - y).abs().max()) for x, y in zip(ga, gc)
                    if x is not None and y is not None)
        return {"pass": worst <= 5e-4, "max_abs_gradient_difference": worst,
                "note": "float32; truncation t=2"}
    record("G2", "range", "truncated t=2 FULL/EGO2 gradients through readout+trunk agree", g2)

    def g3() -> dict[str, Any]:
        # pick a molecule of diameter <= 2
        chosen = None
        for r in g_rows:
            st = g_structs[r]
            adj = G._adjacency(st.n_atoms, st.bond_u, st.bond_v)
            diam = max(max(G._bfs(st.n_atoms, adj, x, st.n_atoms)) for x in range(st.n_atoms))
            if diam <= 2:
                chosen = r
                break
        if chosen is None:
            return {"pass": True, "skipped": "no diameter<=2 molecule in the small subset",
                    "note": "SKIP recorded as PASS only for this structural predicate"}
        full = G.collate([chosen], g_structs, "FULL", "cpu")
        e2 = G.collate([chosen], g_structs, "EGO2", "cpu")
        with torch.no_grad():
            a = g_model(full, n_layers=5, return_h=True)
            c = g_model(e2, n_layers=5, return_h=True)
        e = float((a["roots"] - c["roots"]).abs().max())
        return {"pass": e <= 2e-5, "row": int(chosen), "max_abs_root_delta": e}
    record("G3", "range", "diameter<=2 molecule: five-layer FULL and EGO2 root vectors agree", g3)

    def g4() -> dict[str, Any]:
        chosen = g_rows[0]
        st = g_structs[chosen]
        adj = G._adjacency(st.n_atoms, st.bond_u, st.bond_v)
        dist = G._bfs(st.n_atoms, adj, 0, st.n_atoms)
        far = [i for i in range(st.n_atoms) if dist[i] >= 3]
        if not far:
            return {"pass": True, "skipped": "no atom at distance>=3 from atom 0"}
        alt = copy.deepcopy(st)
        alt.atom_feat = st.atom_feat.copy()
        alt.atom_feat[far[0], 0] = (int(alt.atom_feat[far[0], 0]) + 1) % atom_dims[0]
        alt.ego2 = G.ego_block(alt, 2)
        alt.ego3 = G.ego_block(alt, 3)
        full = G.collate([chosen], {chosen: st}, "FULL", "cpu")
        ego = G.collate([chosen], {chosen: st}, "EGO2", "cpu")
        full2 = G.collate([chosen], {chosen: alt}, "FULL", "cpu")
        ego2 = G.collate([chosen], {chosen: alt}, "EGO2", "cpu")
        with torch.no_grad():
            a0 = g_model(full, n_layers=5, return_h=True)["roots"][0]
            a1 = g_model(full2, n_layers=5, return_h=True)["roots"][0]
            b0 = g_model(ego, n_layers=5, return_h=True)["roots"][0]
            b1 = g_model(ego2, n_layers=5, return_h=True)["roots"][0]
        full_d = float((a1 - a0).abs().max())
        ego_d = float((b1 - b0).abs().max())
        return {"pass": full_d > 1e-4 and ego_d <= 2e-5, "row": int(chosen),
                "far_atom": int(far[0]), "distance": int(dist[far[0]]),
                "full_root_delta": full_d, "ego_root_delta": ego_d}
    record("G4", "range", "distance>=3 input changes FULL root, leaves EGO2 root unchanged", g4)

    def g5() -> dict[str, Any]:
        st = g_structs[g_rows[0]]
        ok = {}
        for radius, block in ((2, st.ego2), (3, st.ego3)):
            n_ego = int(block["ptr"].size - 1)
            ok[f"r{radius}_roots"] = n_ego == st.n_atoms
            ok[f"r{radius}_one_root_per_atom"] = bool(
                np.array_equal(block["atom"][block["ptr"][:-1]],
                               np.arange(st.n_atoms, dtype=np.int32)))
            # no cross-ego edge: src/dst of an edge lie in the same ego interval
            seg = np.searchsorted(block["ptr"], np.arange(block["ptr"][-1]), side="right") - 1
            ok[f"r{radius}_no_cross_edge"] = bool(np.all(seg[block["src"]] == seg[block["dst"]]))
            # every induced bond appears twice (both directions)
            pairs = {}
            for a, b, bid in zip(block["src"].tolist(), block["dst"].tolist(),
                                 block["bond"].tolist()):
                pairs[(a, b, bid)] = pairs.get((a, b, bid), 0) + 1
            ok[f"r{radius}_directed_pairs"] = all(v == 1 for v in pairs.values())
        inflation = {
            "r2_atom": float(st.ego2["atom"].size / st.n_atoms),
            "r3_atom": float(st.ego3["atom"].size / st.n_atoms),
            "r2_edge": float(st.ego2["src"].size / max(1, 2 * st.n_bonds)),
            "r3_edge": float(st.ego3["src"].size / max(1, 2 * st.n_bonds)),
        }
        return {"pass": all(ok.values()), "checks": ok, "inflation": inflation}
    record("G5", "range", "ego packing: one root per atom, no cross-ego edge, all induced bonds",
           g5)

    def g6() -> dict[str, Any]:
        a, ia = G.build_model("G_FULL5", atom_dims, bond_dims, 9, D.norm_of(b), cfg)
        c, ic = G.build_model("G_EGO2_5", atom_dims, bond_dims, 9, D.norm_of(b), cfg)
        same = C.module_hashes({"m": a}) == C.module_hashes({"m": c})
        return {"pass": same, "init_equal": same,
                "params": ia["parameters"]["total"],
                "expected_roles": "both arms share the complete initial state_dict"}
    record("G6", "init", "G_FULL5 and G_EGO2_5 initialise to byte-identical weights", g6)

    # -- H. trainability ---------------------------------------------------
    def h1() -> dict[str, Any]:
        rows = D.all_rows(b)[:8].tolist()
        structs = ST.build_structures(b, rows)
        reg = ST.build_registry([structs[r] for r in rows], atom_dims, bond_dims)
        y = torch.as_tensor((b.labels[np.asarray(rows)] - b.scaler["mean"])
                            / b.scaler["std"], dtype=torch.float32)
        report = {}
        specs = [("E_MOB", M.build_model, None),
                 ("F_MOB", M.build_model, reg),
                 ("G_EGO2_5", G.build_model, None)]
        for arm, builder, reg_use in specs:
            model, info = builder(arm, atom_dims, bond_dims, 1, D.norm_of(b), cfg)
            if reg_use is not None:
                model.set_registry(FR.registry_tensors(reg, "cpu"))
            model.train()
            if arm.startswith("G_"):
                batch = G.collate(rows, structs, G.mode_of(arm), "cpu")
            else:
                batch = T.collate_ef(rows, structs, b.raw, "cpu")
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
            losses = []
            for _ in range(12):
                out = model(batch)
                loss = ((out["yhat_std"] - y) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.detach()))
            grads = {k: float(v.abs().max()) for k, v in
                     ((n, p.grad) for n, p in model.named_parameters() if p.grad is not None)}
            finite = all(math.isfinite(v) for v in grads.values())
            nonzero = sum(1 for v in grads.values() if v > 0)
            report[arm] = {"loss_start": losses[0], "loss_end": losses[-1],
                           "finite_gradients": finite, "nonzero_grad_tensors": nonzero,
                           "n_grad_tensors": len(grads),
                           "descent": losses[-1] < losses[0]}
        return {"pass": all(v["finite_gradients"] and v["nonzero_grad_tensors"] > 0
                            for v in report.values()), "runs": report}
    record("H1", "trainability", "first-batch gradients finite/non-zero for E/F/G", h1)

    def h2() -> dict[str, Any]:
        rows = D.all_rows(b)[:64].tolist()
        structs = ST.build_structures(b, rows)
        reg = ST.build_registry([structs[r] for r in rows], atom_dims, bond_dims)
        y = torch.as_tensor((b.labels[np.asarray(rows)] - b.scaler["mean"])
                            / b.scaler["std"], dtype=torch.float32)
        out = {}
        for arm in ("E_MOB", "F_MOB", "G_EGO2_5"):
            if arm.startswith("G"):
                model, _ = G.build_model(arm, atom_dims, bond_dims, 2, D.norm_of(b), cfg)
                batch = G.collate(rows, structs, G.mode_of(arm), "cpu")
            else:
                model, _ = M.build_model(arm, atom_dims, bond_dims, 2, D.norm_of(b), cfg)
                if arm[0] == "F":
                    model.set_registry(FR.registry_tensors(reg, "cpu"))
                batch = T.collate_ef(rows, structs, b.raw, "cpu")
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            losses = []
            for _ in range(20):
                loss = ((model(batch)["yhat_std"] - y) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.detach()))
            out[arm] = {"start": losses[0], "end": losses[-1],
                        "ratio": losses[-1] / max(1e-12, losses[0])}
        return {"pass": all(v["ratio"] < 0.999 for v in out.values()), "runs": out}
    record("H2", "trainability", "20 optimiser steps reduce the fit MSE in each family", h2)

    payload = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "pins": C.assert_pins(),
        "checks": CHECKS,
        "n_checks": len(CHECKS),
        "n_pass": int(sum(1 for c in CHECKS if c["status"] == "PASS")),
        "all_pass": bool(all(c["status"] == "PASS" for c in CHECKS if c.get("required"))),
    }
    C.write_json(args.out, payload)
    print(json.dumps({"n_checks": payload["n_checks"], "n_pass": payload["n_pass"],
                      "all_pass": payload["all_pass"]}, indent=1))
    for c in CHECKS:
        if c["status"] != "PASS":
            print(f"FAIL {c['id']}: {c.get('error', json.dumps(C.jsonable(c))[:400])}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())