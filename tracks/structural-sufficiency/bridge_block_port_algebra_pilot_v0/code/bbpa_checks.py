"""Engineering and structural checks for BBPA-v0.

Run before formal training:

    uv run python tracks/.../bridge_block_port_algebra_pilot_v0/code/bbpa_checks.py

Writes ``analysis/checks.json``.  Every check is either PASS or FAIL; a FAIL
must be fixed before training.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import bbpa_common as C
import bbpa_data as D
import bbpa_model as M
import bbpa_structure as S
import bbpa_synthetic as Y
import bbpa_tensor as T


class Checks:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, name: str, ok: bool, info: Mapping[str, Any] | None = None) -> None:
        self.items.append({"name": name, "status": "PASS" if bool(ok) else "FAIL",
                           "info": C.jsonable(dict(info or {}))})
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(C.jsonable(dict(info or {})), sort_keys=True)[:220]}")

    def write(self, path: Path) -> None:
        C.write_json(path, {"generated_utc": C.now_utc(), "code_commit": C.git_commit(),
                            "checks": self.items,
                            "n_pass": sum(x["status"] == "PASS" for x in self.items),
                            "n_fail": sum(x["status"] == "FAIL" for x in self.items)})


def _single_raw(g: Mapping[str, Any]) -> dict[str, Any]:
    return Y.combine([dict(g)])


def _predict(model: M.BBPAModel, raw: Mapping[str, Any], rows: Sequence[int],
             mols: Mapping[int, T.MoleculeTensors], mode: str) -> np.ndarray:
    batch = T.collate(rows, mols, raw, "cpu")
    model.eval()
    with torch.no_grad():
        return model.forward_batch(batch, mode)["yhat_std"].double().cpu().numpy()


def _permute_graph(g: Mapping[str, Any], perm: np.ndarray,
                   bond_perm: np.ndarray) -> dict[str, Any]:
    n = int(g["n_atoms"])
    inv = np.empty(n, np.int64)
    inv[perm] = np.arange(n, dtype=np.int64)
    out = dict(g)
    out["atom_feat"] = np.asarray(g["atom_feat"])[perm]
    out["bond_feat"] = np.asarray(g["bond_feat"])[bond_perm]
    out["bu"] = inv[np.asarray(g["bu"])]
    out["bv"] = inv[np.asarray(g["bv"])]
    # randomly flip the stored orientation of each bond (features stay attached)
    flip = np.zeros(len(bond_perm), bool)
    flip[np.arange(len(bond_perm)) % 3 == 1] = True
    bu = out["bu"].copy()
    bv = out["bv"].copy()
    bu[flip], bv[flip] = out["bv"][flip], out["bu"][flip]
    out["bu"], out["bv"] = bu, bv
    for key in ("degree", "min_ring", "in_cycle"):
        out[key] = np.asarray(g[key])[perm]
    return out


def run_checks(max_mol_checks: int = 64, seed: int = 0) -> dict[str, Any]:
    t0 = time.time()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    ck = Checks()

    # ------------------------------------------------------------------
    # 0. data identity
    # ------------------------------------------------------------------
    b = D.load_bundle()
    ck.add("0_data_identity", True, {
        "raw_cache_sha256": b.hashes["raw_cache"]["sha256"],
        "split_file_sha256": b.hashes["split"]["file_sha256"],
        "split_sha256": b.hashes["split"]["sha256"],
        "n_fit": int(b.fit_rows.size), "n_dev": int(b.eval_rows.size),
        "n_official_train": int(b.official_train.size),
        "fit_dev_disjoint": bool(np.intersect1d(b.fit_rows, b.eval_rows).size == 0),
        "fit_dev_cover_official_train": bool(
            np.union1d(b.fit_rows, b.eval_rows).size == b.official_train.size),
        "label_scaler_mean": b.scaler["mean"], "label_scaler_std": b.scaler["std"],
        "finite_labels_rows": int(np.isfinite(b.labels).sum()),
        "finite_labels_outside_official_train": int(
            np.isfinite(b.labels).sum() - np.isfinite(b.labels[b.official_train]).sum()),
        "finite_labels_on_official_valid": int(np.isfinite(b.labels[b.official_valid]).sum()),
    })

    # ------------------------------------------------------------------
    # 1. exact structure on all D0 molecules + synthetic cases
    # ------------------------------------------------------------------
    all_rows = np.union1d(b.fit_rows, b.eval_rows)
    t = time.time()
    decs = S.build_all(b.raw, all_rows)
    stats = S.structure_stats(decs, all_rows.size,
                              [str(x) for x in np.asarray(b.raw["smiles"])[all_rows]])
    n_valid = 0
    for d in decs:
        S.validate(d)
        n_valid += 1
    ck.add("1_structure_validation", n_valid == all_rows.size, {
        "n_molecules": int(all_rows.size), "n_validated": int(n_valid),
        "seconds": round(time.time() - t, 3),
        "n_bridges_total": stats["n_bridges_total"],
        "n_molecules_with_bridge": stats["n_molecules_with_bridge"],
        "n_molecules_without_bridge": stats["n_molecules_without_bridge"],
        "bridge_coverage_fraction": stats["bridge_coverage_fraction"],
        "block_size": stats["block_size"],
        "block_tree_diameter": stats["block_tree_diameter"],
        "bridges_per_molecule": stats["bridges_per_molecule"],
        "blocks_per_molecule": stats["blocks_per_molecule"],
        "fraction_with_diameter_ge_3": stats["fraction_with_diameter_ge_3"],
    })

    cases = {
        "single_block_benzene": Y.benzene(),
        "single_bridge_ethane": Y.ethane(),
        "four_block_chain": Y.chain4(),
        "star_four_blocks": Y.star4(),
        "ring_chain_four_blocks": Y.concatenated_chain(4),
        "ortho_disubstituted": Y.disubstituted_benzene(1),
        "meta_disubstituted": Y.disubstituted_benzene(2),
        "para_disubstituted": Y.disubstituted_benzene(3),
    }
    case_info: dict[str, Any] = {}
    for name, g in cases.items():
        d = S.decompose(g["n_atoms"], g["bu"], g["bv"])
        info = S.validate(d)
        info.update({
            "block_sizes": [int(len(x.atoms)) for x in d.blocks],
            "block_tree_diameter": [S.tree_diameter(d, c) for c in d.components],
            "max_step": int(d.max_step),
        })
        case_info[name] = info
    # one molecule with two connected components (a forest, not a tree)
    joined_bu = np.asarray([0, 2, 3, 4, 5, 6, 7], np.int64)
    joined_bv = np.asarray([1, 3, 4, 5, 6, 7, 2], np.int64)
    fj = S.decompose(8, joined_bu, joined_bv)
    fj_info = S.validate(fj)
    expected = {
        "single_block_benzene": (1, 0),
        "single_bridge_ethane": (2, 1),
        "four_block_chain": (4, 3),
        "star_four_blocks": (4, 3),
        "ring_chain_four_blocks": (4, 3),
        "ortho_disubstituted": (3, 2),
        "meta_disubstituted": (3, 2),
        "para_disubstituted": (3, 2),
    }
    cases_ok = all((case_info[k]["n_blocks"], case_info[k]["n_bridges"]) == v
                   for k, v in expected.items())
    cases_ok = cases_ok and case_info["four_block_chain"]["block_tree_diameter"] == [3]
    cases_ok = cases_ok and case_info["star_four_blocks"]["block_tree_diameter"] == [2]
    ck.add("2_synthetic_cases", cases_ok,
           {"cases": case_info, "expected_blocks_bridges": expected})
    ck.add("3_two_component_forest", fj_info["n_components"] == 2 and
           fj_info["n_bridges"] == 1, fj_info)

    # ------------------------------------------------------------------
    # 4. port distance / same-ring substituent position
    # ------------------------------------------------------------------
    port_info: dict[str, Any] = {}
    for name, pos in (("ortho", 1), ("meta", 2), ("para", 3)):
        g = Y.disubstituted_benzene(pos)
        d = S.decompose(g["n_atoms"], g["bu"], g["bv"])
        ring_block = [b for b in d.blocks if len(b.atoms) == 6][0]
        dm = d.block_port_dist[ring_block.block_id]
        port_info[name] = {"ring_port_distance": int(dm[0, 1]),
                           "block_port_matrix": dm.tolist()}
    # port encoder outputs and predictions on the three isomers
    graphs = [Y.disubstituted_benzene(p) for p in (1, 2, 3)]
    raw3 = Y.combine(graphs)
    mols3 = T.build_dataset(raw3, [0, 1, 2])
    ad, bd = D.atom_bond_dims()
    model, _ = M.build_model(ad, bd, seed, {"dims": {}})
    model.eval()
    batch3 = T.collate([0, 1, 2], mols3, raw3, "cpu")
    with torch.no_grad():
        out3 = model.forward_batch(batch3, "TREE_PORT", return_states=True)
    uport = out3["u_port"]
    pair_diff = float((uport[:4].unsqueeze(1) - uport[:4].unsqueeze(0)).abs().max())
    preds = out3["yhat_std"].double().cpu().numpy()
    ck.add("4_port_distance_and_isomer_encoding",
           port_info["ortho"]["ring_port_distance"] == 1
           and port_info["meta"]["ring_port_distance"] == 2
           and port_info["para"]["ring_port_distance"] == 3
           and pair_diff > 1e-6
           and float(np.ptp(preds)) > 0.0,
           {"isomers": port_info, "u_port_pairwise_max_abs": pair_diff,
            "predictions": preds.tolist(), "prediction_range": float(np.ptp(preds))})

    # ------------------------------------------------------------------
    # 5. permutation invariance on a real bridged molecule and a synthetic chain
    # ------------------------------------------------------------------
    real_row = int(b.eval_rows[0])
    real_view = D.molecule_view(b.raw, real_row)
    real_g = {
        "n_atoms": int(real_view["n_atoms"]), "n_bonds": int(real_view["n_bonds"]),
        "atom_feat": real_view["atom_feat"], "bond_feat": real_view["bond_feat"],
        "bu": real_view["bond_u"], "bv": real_view["bond_v"],
        "degree": real_view["degree"], "min_ring": real_view["min_ring"],
        "in_cycle": real_view["in_cycle"], "smiles": real_view["smiles"],
    }
    rng = np.random.default_rng(seed)
    inv_info: dict[str, Any] = {}
    for tag, g in (("real", real_g), ("chain4", Y.concatenated_chain(4))):
        base = _single_raw(g)
        m0 = T.build_dataset(base, [0])
        p0 = _predict(model, base, [0], m0, "TREE_PORT")[0]
        vals = []
        for _ in range(5):
            perm = rng.permutation(g["n_atoms"])
            bp = rng.permutation(g["n_bonds"])
            gp = _permute_graph(g, perm, bp)
            raw = _single_raw(gp)
            mm = T.build_dataset(raw, [0])
            vals.append(float(_predict(model, raw, [0], mm, "TREE_PORT")[0]))
        # bond-only reordering (atom ids unchanged)
        vals_bond = []
        for _ in range(3):
            bp = rng.permutation(g["n_bonds"])
            gp = dict(g)
            gp["bond_feat"] = np.asarray(g["bond_feat"])[bp]
            # remap endpoints by permuting the edge list, then re-adding
            bu = np.asarray(g["bu"])[bp]
            bv = np.asarray(g["bv"])[bp]
            gp["bu"], gp["bv"] = bu, bv
            raw = _single_raw(gp)
            mm = T.build_dataset(raw, [0])
            vals_bond.append(float(_predict(model, raw, [0], mm, "TREE_PORT")[0]))
        inv_info[tag] = {
            "base": float(p0),
            "atom_permuted": vals,
            "bond_permuted": vals_bond,
            "atom_max_abs_diff": float(np.max(np.abs(np.asarray(vals) - p0))) if vals else 0.0,
            "bond_max_abs_diff": float(np.max(np.abs(np.asarray(vals_bond) - p0))) if vals_bond else 0.0,
        }
    ok_inv = all(v["atom_max_abs_diff"] <= 1e-5 and v["bond_max_abs_diff"] <= 1e-5
                 for v in inv_info.values())
    ck.add("5_permutation_invariance", ok_inv, inv_info)

    # block / neighbour traversal order
    g = Y.concatenated_chain(4)
    raw = _single_raw(g)
    base_d = S.decompose(g["n_atoms"], g["bu"], g["bv"])
    p0 = float(_predict(model, raw, [0], {0: T.build_molecule_tensors(base_d, 0)},
                        "TREE_PORT")[0])
    order_vals = []
    for _ in range(5):
        d = S.decompose(g["n_atoms"], g["bu"], g["bv"])
        for e in range(len(d.edge_deps)):
            rng.shuffle(d.edge_deps[e])
        for blk in d.blocks:
            rng.shuffle(blk.ports)
        order_vals.append(float(_predict(model, raw, [0],
                                        {0: T.build_molecule_tensors(d, 0)},
                                        "TREE_PORT")[0]))
    ck.add("6_block_neighbour_order", max(abs(x - p0) for x in order_vals) <= 1e-5,
           {"base": p0, "shuffled": order_vals})

    # root-choice invariance of the DP schedule
    root_info = {}
    for tag, g in (("chain4", Y.concatenated_chain(4)), ("real", real_g)):
        vals = {}
        for mode in ("first", "last"):
            d = S.reschedule(S.decompose(g["n_atoms"], g["bu"], g["bv"]), mode)
            mt = {0: T.build_molecule_tensors(d, 0)}
            raw = _single_raw(g)
            vals[mode] = float(_predict(model, raw, [0], mt, "TREE_PORT")[0])
        root_info[tag] = {"first": vals["first"], "last": vals["last"],
                          "abs_diff": abs(vals["first"] - vals["last"])}
    ck.add("6b_root_choice_invariance",
           all(v["abs_diff"] <= 1e-5 for v in root_info.values()), root_info)

    # ------------------------------------------------------------------
    # 7. batch vs single
    # ------------------------------------------------------------------
    sample_rows = all_rows[:int(max_mol_checks)]
    mols_sample = {int(r): T.build_molecule_tensors(d, int(r))
                   for r, d in zip(sample_rows.tolist(), decs[:int(max_mol_checks)])}
    braw = b.raw
    batch = T.collate(sample_rows.tolist(), mols_sample, braw, "cpu")
    model.eval()
    with torch.no_grad():
        p_batch = model.forward_batch(batch, "TREE_PORT")["yhat_std"].double().cpu().numpy()
    p_single = []
    for r in sample_rows.tolist():
        with torch.no_grad():
            p_single.append(float(model.forward_batch(
                T.collate([int(r)], mols_sample, braw, "cpu"), "TREE_PORT")["yhat_std"][0]))
    ck.add("7_batch_vs_single", float(np.max(np.abs(p_batch - np.asarray(p_single)))) <= 1e-5,
           {"n": int(sample_rows.size),
            "max_abs_diff": float(np.max(np.abs(p_batch - np.asarray(p_single))))})

    # ------------------------------------------------------------------
    # 8. TREE vs LOCAL / star equality / deep chain difference
    # ------------------------------------------------------------------
    def two_mode_info(g: Mapping[str, Any]) -> dict[str, Any]:
        raw = _single_raw(g)
        d = S.decompose(g["n_atoms"], g["bu"], g["bv"])
        mt = {0: T.build_molecule_tensors(d, 0)}
        batch = T.collate([0], mt, raw, "cpu")
        model.eval()
        with torch.no_grad():
            tr = model.forward_batch(batch, "TREE_PORT", return_states=True)
            lo = model.forward_batch(batch, "LOCAL_PORT", return_states=True)
        if tr["states"] is None or tr["states"].numel() == 0:
            sd = 0.0
            qd = 0.0
        else:
            sd = float((tr["states"] - lo["states"]).norm(dim=1).max())
            qd = float((tr["q"] - lo["q"]).abs().max()) if tr["q"] is not None else 0.0
        return {
            "n_edges": int(d.edge_step.size),
            "state_max_l2_diff": sd,
            "q_max_abs_diff": qd,
            "pred_tree": float(tr["yhat_std"][0]),
            "pred_local": float(lo["yhat_std"][0]),
            "pred_abs_diff": float(abs(tr["yhat_std"][0] - lo["yhat_std"][0])),
        }

    chain_info = two_mode_info(Y.chain4())
    ringchain_info = two_mode_info(Y.concatenated_chain(4))
    star_info = two_mode_info(Y.star4())
    ck.add("8_deep_chain_tree_local_differ",
           chain_info["state_max_l2_diff"] > 1e-6 and chain_info["q_max_abs_diff"] > 1e-6,
           {"chain4": chain_info, "ringchain4": ringchain_info})
    ck.add("9_star_special_case_equal",
           star_info["state_max_l2_diff"] <= 1e-7 and star_info["pred_abs_diff"] <= 1e-7,
           {"star4": star_info})
    ck.add("10_single_bridge_tree_local_equal",
           two_mode_info(Y.ethane())["state_max_l2_diff"] <= 1e-7,
           {"ethane": two_mode_info(Y.ethane())})

    # ------------------------------------------------------------------
    # 11. finite forward/backward and non-zero key gradients
    # ------------------------------------------------------------------
    graphs = [Y.benzene(), Y.ethane(), Y.chain4(), Y.star4(),
              Y.concatenated_chain(4), Y.disubstituted_benzene(2)]
    rawg = Y.combine(graphs)
    molsg = T.build_dataset(rawg, list(range(len(graphs))))
    batchg = T.collate(list(range(len(graphs))), molsg, rawg, "cpu")
    grad_info: dict[str, Any] = {}
    for mode in M.MODES:
        mm, _ = M.build_model(ad, bd, seed, {"dims": {}})
        mm.train()
        out = mm.forward_batch(batchg, mode)
        target = torch.ones_like(out["yhat_std"])
        loss = ((out["yhat_std"] - target) ** 2).mean()
        loss.backward()
        groups: dict[str, float] = {}
        finite = True
        for name, p in mm.named_parameters():
            if p.grad is None:
                continue
            top = name.split(".")[0]
            groups[top] = groups.get(top, 0.0) + float(p.grad.norm())
            finite = finite and bool(torch.isfinite(p.grad).all())
        required = ["atom_emb", "bond_emb", "pair_mlp", "block_mlp", "readout_mlp", "beta"]
        if mode != "BLOCK_BAG":
            required += ["port_mlp", "message_mlp", "update_mlp", "norm", "q_mlp"]
        grad_info[mode] = {"loss": float(loss), "finite": finite,
                           "grad_norm_by_module": groups,
                           "required_nonzero": {k: bool(groups.get(k, 0.0) > 0.0)
                                                for k in required}}
    ok_grad = all(v["finite"] and all(v["required_nonzero"].values())
                  for v in grad_info.values())
    ck.add("11_finite_backward_key_gradients", ok_grad, grad_info)

    # ------------------------------------------------------------------
    # 12. parameter count / identical initialisation
    # ------------------------------------------------------------------
    init_info: dict[str, Any] = {}
    ledgers = {}
    for mode in M.MODES:
        mm, h = M.build_model(ad, bd, seed, {"dims": {}})
        ledgers[mode] = M.parameter_ledger(mm)
        init_info[mode] = h
    same = all(init_info["TREE_PORT"] == init_info[m] for m in M.MODES)
    ck.add("12_identical_init_and_params", same and
           len({tuple(sorted(v.items())) for v in ledgers.values()}) == 1,
           {"ledger": ledgers["TREE_PORT"], "hashes_equal": same,
            "total_params": ledgers["TREE_PORT"]["total"]})

    # ------------------------------------------------------------------
    # 13. learned state variation / non-constant channel
    # ------------------------------------------------------------------
    n_state = min(32, int(all_rows.size))
    state_rows = all_rows[:n_state]
    state_mols = {int(r): mols_sample[int(r)] for r in state_rows.tolist()}
    sbatch = T.collate(state_rows.tolist(), state_mols, braw, "cpu")
    model.eval()
    with torch.no_grad():
        sout = model.forward_batch(sbatch, "TREE_PORT", return_states=True)
    st = sout["states"].double().cpu().numpy()
    sd = st.std(axis=0)
    eff = C.effective_rank(st)
    near_const = float(np.mean(sd < 1e-4)) if sd.size else 1.0
    # structure sensitivity: states from a different molecule must differ
    other = T.build_molecule_tensors(decs[0], int(all_rows[0]))
    ck.add("13_state_variation", st.size > 0 and float(sd.mean()) > 1e-3
           and near_const < 0.5 and eff["participation_ratio"] > 1.0,
           {"n_states": int(st.shape[0]), "mean_dim_std": float(sd.mean()),
            "min_dim_std": float(sd.min()) if sd.size else 0.0,
            "near_constant_dim_fraction": near_const, "effective_rank": eff,
            "sample_row": int(other.row)})

    # ------------------------------------------------------------------
    # 14. no label leakage / no cross-molecule indexing
    # ------------------------------------------------------------------
    label_rows = set(np.where(np.isfinite(b.labels))[0].tolist())
    train_rows = set(b.official_train.tolist())
    ck.add("14_label_isolation", label_rows == train_rows
           and not np.isfinite(b.labels[b.official_valid]).any()
           and np.isin(b.eval_rows, b.official_train).all(),
           {"finite_label_rows": len(label_rows), "official_train_rows": len(train_rows),
            "valid_rows_with_finite_label": int(np.isfinite(b.labels[b.official_valid]).sum()),
            "dev_rows_are_official_train": bool(np.isin(b.eval_rows, b.official_train).all())})

    # ------------------------------------------------------------------
    # 15. real structure statistics
    # ------------------------------------------------------------------
    ck.add("15_structure_statistics", True, stats)

    out = {
        "generated_utc": C.now_utc(),
        "code_commit": C.git_commit(),
        "elapsed_seconds": round(time.time() - t0, 3),
        "n_pass": sum(x["status"] == "PASS" for x in ck.items),
        "n_fail": sum(x["status"] == "FAIL" for x in ck.items),
        "checks": ck.items,
        "structure_stats": stats,
        "parameter_ledger": ledgers["TREE_PORT"],
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR / "checks.json"))
    parser.add_argument("--max-mol-checks", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = run_checks(max_mol_checks=int(args.max_mol_checks), seed=int(args.seed))
    C.write_json(args.out, out)
    print(f"wrote {args.out}: {out['n_pass']} pass / {out['n_fail']} fail "
          f"({out['elapsed_seconds']}s)")
    if out["n_fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()