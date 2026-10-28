"""CPU gates for LCRC-v1.

Run before any GPU job:

    python .../lc_cpu_checks.py

Writes ``analysis/cpu_checks.json``.  The checks cover parameter/module audits,
SSL provenance, the absence of any junction path, permutation / bond-direction /
batching invariance, the exact semantics of the rooted S1/S2 and pair-relation
inputs, boundary cases, R/N identical initialisation and data order, the
synthetic correspondence test, real train-only relation coverage and a
ridge-redundancy probe, finite gradients, a short CPU descent, checkpoint
round-trips, and official-test isolation.
"""

from __future__ import annotations

import math
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

import lc_common as L
import lc_data as D
import lc_models as M

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


class Checker:
    def __init__(self):
        self.results: list[dict[str, Any]] = []

    def run(self, cid: str, fn: Callable[[], Any]) -> None:
        try:
            details = fn()
            self.results.append({"id": cid, "pass": True, "details": details})
            print(f"[PASS] {cid}")
        except Exception as exc:  # noqa: BLE001
            self.results.append({
                "id": cid, "pass": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
            print(f"[FAIL] {cid}: {type(exc).__name__}: {exc}")

    @property
    def ok(self) -> bool:
        return all(r["pass"] for r in self.results)


def _cache_from_molecules(mols, labels=None) -> dict:
    cache = D._concat_molecules(mols)
    cache["split"] = "synthetic"
    cache["n_molecules"] = len(mols)
    cache["labels"] = np.asarray(
        labels if labels is not None else np.zeros(len(mols)), dtype=np.float32
    )
    cache["meta"] = {"feature_version": L.FEATURE_VERSION, "split": "synthetic"}
    return cache


def _forward_pred(model, cache, rows) -> np.ndarray:
    model.eval()
    with torch.no_grad(), L.C.isolated_rng():
        batch = D.collate(cache, rows)
        return model(batch).cpu().numpy().reshape(-1)


def _build_norm(train_cache) -> dict:
    return D.fit_norm(train_cache)


def _tensor_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and bool(torch.equal(a, b))


def main() -> int:
    L.ensure_dir(L.ANALYSIS)
    # Build train/valid caches if needed.  The test split is never touched.
    prep = D.prep(("train", "valid"))
    train = D.load_cache("train")
    valid = D.load_cache("valid")
    norm = D.load_norm()
    objs_train, _ = L.D.load_p3_objects("train")
    objs_valid, _ = L.D.load_p3_objects("valid")
    ck = Checker()

    # ------------------------------------------------------------------
    # G1 - parameters, modules, SSL provenance, no junction path
    # ------------------------------------------------------------------
    models = {arm: M.build_model(arm, norm, 0, "cpu") for arm in ("A", "R", "N")}

    def g1():
        a_model, a_info = models["A"]
        r_model, r_info = models["R"]
        n_model, n_info = models["N"]
        assert a_info["parameters"] < r_info["parameters"]
        assert r_info["parameters"] == n_info["parameters"]
        for arm, (model, info) in models.items():
            assert info["parameters"] <= 500_000
            assert not info["has_junction_module"]
            assert info["state_keys_with_junction"] == []
            assert not any("junction" in n or "junc" in n for n, _ in model.named_modules())
            assert not any("junction" in n or "junc" in n for n, _ in model.named_parameters())
            assert not any("junction" in k or "junc" in k for k in model.state_dict())
        # R and N are bit-identical at initialisation.
        sd_r = {k: v.detach().clone() for k, v in r_model.state_dict().items()}
        sd_n = {k: v.detach().clone() for k, v in n_model.state_dict().items()}
        assert set(sd_r) == set(sd_n)
        for k in sd_r:
            assert _tensor_equal(sd_r[k], sd_n[k]), f"R/N differ at {k}"
        # Shared A/R tensors are bit-identical (relation is appended last).
        sd_a = {k: v.detach().clone() for k, v in a_model.state_dict().items()}
        shared = set(sd_a) & set(sd_r)
        assert shared
        for k in shared:
            assert _tensor_equal(sd_a[k], sd_r[k]), f"A/R shared tensor differs at {k}"
        # SSL tensors really equal the frozen file.
        ssl = torch.load(M.ssl_path(0), map_location="cpu", weights_only=False)
        for arm, (model, info) in models.items():
            for name in ("atom_encoder", "bond_encoder"):
                sub = getattr(model, name)
                for k, v in sub.state_dict().items():
                    assert torch.equal(v.cpu(), ssl[name][k]), f"{arm} {name}.{k} != SSL"
        # Zero-init relation head -> step-0 R/N equals A.
        mini = _cache_from_molecules([D.build_molecule_from_object(objs_train[i]) for i in range(4)])
        rows = np.arange(4)
        p_a = _forward_pred(a_model, mini, rows)
        p_r = _forward_pred(r_model, mini, rows)
        p_n = _forward_pred(n_model, mini, rows)
        d_ra = float(np.max(np.abs(p_r - p_a)))
        d_rn = float(np.max(np.abs(p_r - p_n)))
        assert d_ra == 0.0 and d_rn == 0.0, (d_ra, d_rn)
        # Source-level guard: the model file cannot contain a junction encoder call.
        src = (Path(__file__).parent / "lc_models.py").read_text()
        for token in ("junction_encoder", "junc_feat", "junc_index", "P3Encoders"):
            assert token not in src, f"forbidden token {token} in lc_models.py"
        return {
            "parameters": {arm: info["parameters"] for arm, (_m, info) in models.items()},
            "breakdown": {arm: info["parameters_breakdown"] for arm, (_m, info) in models.items()},
            "initial_state_sha256": {arm: info["initial_state_sha256"] for arm, (_m, info) in models.items()},
            "ssl": a_info["ssl"],
            "has_junction_module": False,
            "state_keys_with_junction": [],
            "r_vs_n_initial_max_abs": 0.0,
            "r_vs_a_step0_max_abs": d_ra,
            "n_vs_a_step0_max_abs": d_rn,
            "forbidden_source_tokens_absent": True,
        }

    ck.run("G1_parameters_modules_ssl_no_junction", g1)

    # ------------------------------------------------------------------
    # G2 - invariance to renumbering, bond direction, batch composition
    # ------------------------------------------------------------------
    def g2():
        idx = [0, 1, 2, 3]
        mols = [D.build_molecule_from_object(objs_train[i]) for i in idx]
        base = _cache_from_molecules(mols)
        rows = np.arange(4)
        out = {}
        for arm, (model, _info) in models.items():
            p0 = _forward_pred(model, base, rows)
            # atom renumbering of molecule 0
            n0 = mols[0]["n_atoms"]
            rng = np.random.default_rng(20261108)
            perm = rng.permutation(int(n0))
            m0p = D.build_molecule_from_object(objs_train[idx[0]], perm=perm)
            cache_p = _cache_from_molecules([m0p] + mols[1:])
            pp = _forward_pred(model, cache_p, rows)
            # bond direction reversal for all molecules
            mols_rev = [D.build_molecule_from_object(objs_train[i], reverse_bonds=True) for i in idx]
            cache_rev = _cache_from_molecules(mols_rev)
            pr = _forward_pred(model, cache_rev, rows)
            # batch split 2 + 2 and order swap
            p_left = _forward_pred(model, base, [0, 1])
            p_right = _forward_pred(model, base, [2, 3])
            p_split = np.concatenate([p_left, p_right])
            p_swap = _forward_pred(model, base, [3, 1, 2, 0])
            out[arm] = {
                "renumber_max_abs": float(np.max(np.abs(pp - p0))),
                "bond_reverse_max_abs": float(np.max(np.abs(pr - p0))),
                "batch_split_max_abs": float(np.max(np.abs(p_split - p0))),
                "batch_order_max_abs": float(np.max(np.abs(p_swap - p0[[3, 1, 2, 0]]))),
            }
            assert out[arm]["renumber_max_abs"] < 2e-5
            assert out[arm]["bond_reverse_max_abs"] < 2e-5
            assert out[arm]["batch_split_max_abs"] < 2e-5
            assert out[arm]["batch_order_max_abs"] < 2e-5
        return out

    ck.run("G2_permutation_bond_direction_batching", g2)

    # ------------------------------------------------------------------
    # G3 - exact semantics of S1/S2, distance/ring categories, boundaries
    # ------------------------------------------------------------------
    def g3():
        import networkx as nx

        n_check = 200
        nbr_mismatch = 0
        path_mismatch = 0
        pair_mismatch = 0
        scalar_mismatch = 0
        joint_rank1_frac = []
        n_pairs_total = 0
        n_paths_total = 0
        for mi in range(n_check):
            obj = objs_train[mi]
            mol = D.build_molecule_from_object(obj)
            at = mol["atom_type"]
            bi = mol["bond_index"].astype(np.int64)
            bt = mol["bond_type"].astype(np.int64)
            adj = D._adjacency(int(mol["n_atoms"]), bi, bt)
            # S1 codes
            expect_codes = []
            for v in range(int(mol["n_atoms"])):
                expect_codes.extend([int(at[u]) * 3 + int(b) - 1 for u, b, _e in adj[v]])
            if not np.array_equal(np.asarray(expect_codes, dtype=np.uint8), mol["nbr_code"]):
                nbr_mismatch += 1
            # S2 records (multiset per root)
            for v in range(int(mol["n_atoms"])):
                s, e = int(mol["path_ptr"][v]), int(mol["path_ptr"][v + 1])
                got = sorted(zip(mol["path_cat"][s:e].tolist(),
                                 mol["path_end63"][s:e].tolist(),
                                 mol["path_end"][s:e].tolist(),
                                 mol["path_closure"][s:e].tolist()))
                exp = []
                for w, b1, _e in adj[v]:
                    for u, b2, _e2 in adj[w]:
                        if u == v:
                            continue
                        exp.append((int(at[w]) * 9 + (int(b1) - 1) * 3 + (int(b2) - 1),
                                    int(at[u]) * 3 + (int(b2) - 1), u,
                                    1 if u in [x[0] for x in adj[v]] else 0))
                if got != sorted(exp):
                    path_mismatch += 1
                    break
            n_paths_total += len(mol["path_cat"])
            # pair semantics
            dist = D._all_pairs_dist(int(mol["n_atoms"]), adj)
            graph = nx.Graph()
            graph.add_nodes_from(range(int(mol["n_atoms"])))
            graph.add_edges_from((int(u), int(v)) for u, v in bi)
            blocks = [set(c) for c in nx.biconnected_components(graph) if len(c) >= 3]
            mem = [set() for _ in range(int(mol["n_atoms"]))]
            for k, blk in enumerate(blocks):
                for a in blk:
                    mem[a].add(k)
            exp_pairs = {}
            for i in range(int(mol["n_atoms"])):
                for j in range(i + 1, int(mol["n_atoms"])):
                    if dist[i, j] < 3:
                        continue
                    ring = 1 if (mem[i] & mem[j]) else 0
                    exp_pairs[(i, j)] = D._dist_bucket(int(dist[i, j])) * 2 + ring
            got_pairs = {(int(i), int(j)): int(c) for i, j, c in
                         zip(mol["pair_i"], mol["pair_j"], mol["pair_cat"])}
            if exp_pairs != got_pairs:
                pair_mismatch += 1
            n_pairs_total += len(got_pairs)
            # scalars
            exp_n_d2 = float(np.sum(dist[0] == 2))
            if abs(float(mol["atom_scalar"][0, 0]) - math.log1p(exp_n_d2)) > 1e-6:
                scalar_mismatch += 1
            # jointness of the 189-way path table: fraction explained by rank-1
            if len(mol["path_cat"]):
                counts = np.bincount(mol["path_cat"], minlength=189).reshape(21, 9).astype(np.float64)
                if counts.sum() > 1:
                    sv = np.linalg.svd(counts, compute_uv=False)
                    joint_rank1_frac.append(float(sv[0] ** 2 / max((sv ** 2).sum(), 1e-12)))
        assert nbr_mismatch == 0
        assert path_mismatch == 0
        assert pair_mismatch == 0
        assert scalar_mismatch == 0

        # Boundary cases built from arrays.
        def one_atom():
            atom_feat = np.zeros((1, L.ATOM_FEAT_DIM), np.float32)
            atom_feat[0, 0] = 1.0
            atom_feat[0, 28 + 0] = 1.0
            atom_feat[0, 106] = 1.0  # terminal
            return D.build_from_arrays(atom_feat, np.array([0]), np.zeros((0, 2), np.int64),
                                       np.zeros(0, np.int64), np.zeros((0, L.BOND_FEAT_DIM), np.float32),
                                       np.zeros(L.GLOBAL_DIM, np.float32))

        def chain4():
            n = 4
            atom_feat = np.zeros((n, L.ATOM_FEAT_DIM), np.float32)
            for i in range(n):
                atom_feat[i, 0] = 1.0
                atom_feat[i, 28 + min(i, 3)] = 1.0
            atom_feat[:, 106] = 1.0
            atom_feat[0, 106] = 1.0
            atom_feat[-1, 106] = 1.0
            bi = np.asarray([[0, 1], [1, 2], [2, 3]], np.int64)
            bt = np.ones(3, np.int64)
            bf = np.zeros((3, L.BOND_FEAT_DIM), np.float32)
            bf[:, 0] = 1.0
            return D.build_from_arrays(atom_feat, np.zeros(n, np.int64), bi, bt, bf,
                                       np.zeros(L.GLOBAL_DIM, np.float32))

        def ring6():
            n = 6
            atom_feat = np.zeros((n, L.ATOM_FEAT_DIM), np.float32)
            for i in range(n):
                atom_feat[i, 0] = 1.0
                atom_feat[i, 28 + 2] = 1.0
                atom_feat[i, 33] = 1.0
                atom_feat[i, 34 + 3] = 1.0
                atom_feat[i, 40] = 2.0
                atom_feat[i, 40 + 1] = 1.0
            bi = np.asarray([[i, (i + 1) % n] for i in range(n)], np.int64)
            bt = np.ones(n, np.int64)
            bf = np.zeros((n, L.BOND_FEAT_DIM), np.float32)
            bf[:, 0] = 1.0
            bf[:, 4] = 1.0
            return D.build_from_arrays(atom_feat, np.zeros(n, np.int64), bi, bt, bf,
                                       np.zeros(L.GLOBAL_DIM, np.float32))

        def two_components():
            n = 4
            atom_feat = np.zeros((n, L.ATOM_FEAT_DIM), np.float32)
            for i in range(n):
                atom_feat[i, 0] = 1.0
                atom_feat[i, 28 + 1] = 1.0
            bi = np.asarray([[0, 1], [2, 3]], np.int64)
            bt = np.ones(2, np.int64)
            bf = np.zeros((2, L.BOND_FEAT_DIM), np.float32)
            bf[:, 0] = 1.0
            return D.build_from_arrays(atom_feat, np.zeros(n, np.int64), bi, bt, bf,
                                       np.zeros(L.GLOBAL_DIM, np.float32))

        boundary = {}
        for name, mol in (("one_atom", one_atom()), ("chain4", chain4()),
                          ("ring6", ring6()), ("two_components", two_components())):
            cache = _cache_from_molecules([mol])
            for arm, (model, _info) in models.items():
                pred = _forward_pred(model, cache, [0])
                assert np.isfinite(pred).all()
            boundary[name] = {
                "n_atoms": int(mol["n_atoms"]), "n_pairs": int(len(mol["pair_i"])),
                "n_paths": int(len(mol["path_cat"])),
                "pair_categories": sorted(set(mol["pair_cat"].tolist())),
                "pred_finite": True,
            }
        # The multiplicative path term must distinguish identical marginals
        # with a different (middle/bond, end/bond) co-occurrence.
        local = models["A"][0].local
        with torch.no_grad():
            cat_a = torch.as_tensor([0, 1], dtype=torch.long)
            end_a = torch.as_tensor([0, 1], dtype=torch.long)
            cat_b = torch.as_tensor([0, 1], dtype=torch.long)
            end_b = torch.as_tensor([1, 0], dtype=torch.long)
            def path_sum(cat, end):
                joint = local.joint_proj(local.path_u(cat) * local.path_v(end))
                return (local.path_emb(cat).sum(0), local.end_emb(end).sum(0), joint.sum(0),
                        torch.cat([local.path_emb(cat), local.end_emb(end), joint], dim=1).sum(0))
            a_marg, a_end, a_joint, a_all = path_sum(cat_a, end_a)
            b_marg, b_end, b_joint, b_all = path_sum(cat_b, end_b)
            joint_diff = float((a_joint - b_joint).abs().max())
            all_diff = float((a_all - b_all).abs().max())
            marg_same = float((a_marg - b_marg).abs().max()) + float((a_end - b_end).abs().max())
        assert marg_same == 0.0
        assert joint_diff > 1e-6 and all_diff > 1e-6

        # ring6: opposite atoms are d=3 in the same cycle block -> category 1
        ring_cats = sorted(set(ring6()["pair_cat"].tolist()))
        assert ring_cats == [1], ring_cats
        # chain4 has exactly the (0,3) pair at distance 3, no ring block.
        chain = chain4()
        assert len(chain["pair_i"]) == 1 and int(chain["pair_cat"][0]) == 0
        # two_components has unreachable cross pairs, mapped to the >=6 bucket.
        comp_cats = sorted(set(two_components()["pair_cat"].tolist()))
        assert comp_cats == [4], comp_cats
        return {
            "n_molecules_checked": n_check,
            "nbr_mismatch": nbr_mismatch,
            "path_mismatch": path_mismatch,
            "pair_mismatch": pair_mismatch,
            "scalar_mismatch": scalar_mismatch,
            "n_pairs_total": n_pairs_total,
            "n_paths_total": n_paths_total,
            "path_joint_rank1_energy_mean": float(np.mean(joint_rank1_frac)),
            "path_joint_rank1_energy_p90": float(np.percentile(joint_rank1_frac, 90)),
            "path_joint_is_not_exactly_rank1": bool(np.mean(joint_rank1_frac) < 0.999999),
            "synthetic_joint_interaction_max_abs": joint_diff,
            "synthetic_marginal_sum_max_abs": marg_same,
            "synthetic_full_path_sum_max_abs": all_diff,
            "boundary": boundary,
            "unreachable_maps_to_ge6": True,
        }

    ck.run("G3_rooted_stats_pair_categories_boundary", g3)

    # ------------------------------------------------------------------
    # G4 - identical R/N init and identical data order
    # ------------------------------------------------------------------
    def g4():
        r_model, r_info = models["R"]
        n_model, n_info = models["N"]
        assert r_info["initial_state_sha256"] == n_info["initial_state_sha256"]
        orders = []
        for _ in range(2):
            gen = torch.Generator().manual_seed(0 + 101)
            per_epoch = [torch.randperm(10000, generator=gen).numpy().copy() for _ in range(600)]
            orders.append(per_epoch)
        same = all(np.array_equal(orders[0][e], orders[1][e]) for e in range(600))
        assert same
        first = orders[0][0][:128]
        # The trainer source must use exactly this generator seed.
        src = (Path(__file__).parent / "lc_train.py").read_text()
        assert "Generator().manual_seed(int(seed) + 101)" in src
        assert "torch.randperm(len(rows_train), generator=gen)" in src
        return {
            "seed": 0,
            "data_order_seed": 101,
            "identical_over_600_epochs": bool(same),
            "first_batch_rows_sha256": L.C.sha256_array(first.astype(np.int64)),
            "first_batch_head": first[:10].tolist(),
            "r_n_state_hash": r_info["initial_state_sha256"],
        }

    ck.run("G4_r_n_identical_init_and_data_order", g4)

    # ------------------------------------------------------------------
    # G5 - synthetic correspondence test
    # ------------------------------------------------------------------
    def g5():
        h = torch.as_tensor(np.random.default_rng(20261108).normal(size=(8, 64)), dtype=torch.float32)
        pairs = torch.as_tensor(np.asarray(
            [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [4, 6]], dtype=np.int64))
        pi, pj = pairs[:, 0], pairs[:, 1]
        # Both assignments use the same pair set and therefore the same z multiset.
        cat_a = torch.as_tensor(np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64))
        cat_b = torch.as_tensor(np.asarray([2, 2, 1, 1, 0, 0], dtype=np.int64))
        graph = torch.zeros(6, dtype=torch.int64)
        rel = M.RelationHead(M.DEFAULT_CONFIG)
        z = rel.pair_content(h, pi, pj)
        Ma, na, ea = rel.aggregate(z, graph, cat_a, 1, "R")
        Mb, nb, eb = rel.aggregate(z, graph, cat_b, 1, "R")
        Na, _, _ = rel.aggregate(z, graph, cat_a, 1, "N")
        Nb, _, _ = rel.aggregate(z, graph, cat_b, 1, "N")
        # random non-zero head so Delta can differ
        with torch.no_grad():
            rel.head.weight.normal_(0, 0.1)
            rel.head.bias.normal_(0, 0.01)
        da = rel.readout(Ma, na, ea)
        db = rel.readout(Mb, nb, eb)
        # pair order permutation must not change R or N
        perm = torch.as_tensor(np.asarray([3, 0, 5, 2, 1, 4], dtype=np.int64))
        Ma2, _, _ = rel.aggregate(z[perm], graph[perm], cat_a[perm], 1, "R")
        r_diff = float((Ma - Mb).abs().max())
        n_diff = float((Na - Nb).abs().max())
        delta_diff = float((da - db).abs())
        assert r_diff > 1e-5
        assert n_diff == 0.0
        assert delta_diff > 1e-6
        assert float((Ma - Ma2).abs().max()) == 0.0
        assert np.array_equal(na.numpy(), nb.numpy())
        return {
            "same_z_multiset": True,
            "same_category_counts": True,
            "counts": na.view(-1).tolist(),
            "R_content_max_abs_difference": r_diff,
            "N_content_max_abs_difference": n_diff,
            "R_delta_abs_difference": delta_diff,
            "pair_order_invariant": True,
        }

    ck.run("G5_synthetic_correspondence", g5)

    # ------------------------------------------------------------------
    # G6 - real train relation coverage + redundancy probe
    # ------------------------------------------------------------------
    def g6():
        r_model, _ = models["R"]
        n_model, _ = models["N"]
        rows = np.arange(512, dtype=np.int64)
        cache = train
        r_model.eval()
        n_model.eval()
        counts_all, exists_all, diff_all = [], [], []
        Ms, hsum, hsq, hmean, hmean2, gs, ns = [], [], [], [], [], [], []
        with torch.no_grad(), L.C.isolated_rng():
            for start in range(0, len(rows), 64):
                chunk = rows[start:start + 64]
                batch = D.collate(cache, chunk)
                _, aux_r = r_model(batch, return_aux=True)
                _, aux_n = n_model(batch, return_aux=True)
                M_r = aux_r["M"].numpy()
                M_n = aux_n["M"].numpy()
                n = aux_r["n"].numpy()
                ex = aux_r["exists"].numpy()
                counts_all.append(n)
                exists_all.append(ex)
                diff = np.linalg.norm(M_r - M_n, axis=2)
                diff_all.append(diff)
                Ms.append(M_r.reshape(len(chunk), -1))
                h = aux_r["h"].numpy()
                atom_batch = batch["atom_batch"].numpy()
                B = len(chunk)
                s = np.zeros((B, h.shape[1]), dtype=np.float64)
                sq = np.zeros((B, h.shape[1]), dtype=np.float64)
                nn = np.zeros(B, dtype=np.float64)
                for b in range(B):
                    sel = atom_batch == b
                    s[b] = h[sel].astype(np.float64).sum(0)
                    sq[b] = (h[sel].astype(np.float64) ** 2).sum(0)
                    nn[b] = sel.sum()
                hsum.append(s)
                hsq.append(sq)
                hmean.append(s / nn[:, None])
                hmean2.append(sq / nn[:, None])
                gs.append(batch["global"].numpy().astype(np.float64))
                ns.append(nn)
        n = np.concatenate(counts_all, 0)
        ex = np.concatenate(exists_all, 0)
        diff = np.concatenate(diff_all, 0)
        M = np.concatenate(Ms, 0)
        X = np.concatenate([
            np.concatenate(hsum, 0), np.concatenate(hsq, 0),
            np.concatenate(hmean, 0), np.concatenate(hmean2, 0),
            np.concatenate(gs, 0), np.concatenate(ns, 0)[:, None],
            np.log1p(np.concatenate(ns, 0))[:, None],
        ], axis=1)
        # held-out ridge reconstruction of the R relation content from A's
        # pooled local features
        rng = np.random.default_rng(20261108)
        perm = rng.permutation(len(M))
        tr, te = perm[: len(M) // 2], perm[len(M) // 2:]
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 1e-8] = 1.0
        Xtr = np.concatenate([(X[tr] - mu) / sd, np.ones((len(tr), 1))], 1)
        Xte = np.concatenate([(X[te] - mu) / sd, np.ones((len(te), 1))], 1)
        alpha = 1.0
        A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
        W = np.linalg.solve(A, Xtr.T @ M[tr])
        pred = Xte @ W
        ss_res = ((M[te] - pred) ** 2).sum(0)
        ss_tot = ((M[te] - M[te].mean(0)) ** 2).sum(0)
        r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
        valid_dim = ss_tot > 1e-8
        # permutation control
        W0 = np.linalg.solve(A, Xtr.T @ M[tr][rng.permutation(len(tr))])
        pred0 = Xte @ W0
        r2_control = 1.0 - ((M[te] - pred0) ** 2).sum(0) / np.maximum(ss_tot, 1e-12)
        r2_cat = r2.reshape(6, 32)
        valid_cat = valid_dim.reshape(6, 32)
        per_cat = [float(r2_cat[c][valid_cat[c]].mean()) if valid_cat[c].any() else None
                   for c in range(6)]
        common_cats = [0, 2, 4]  # categories present in every sampled molecule
        common_mean = float(r2_cat[common_cats][valid_cat[common_cats]].mean()) if common_cats else None
        mean_nonempty = float(np.mean([diff[i][ex[i] > 0].mean() if ex[i].any() else 0.0
                                       for i in range(len(diff))]))
        frac_two = float(np.mean(ex.sum(1) >= 2))
        frac_diff = float(np.mean([(diff[i][ex[i] > 0] > 1e-6).any() for i in range(len(diff))]))
        # correspondence-vs-AE probe is done in analysis on trained models.
        # h participation ratio at random init (RCC-style degeneracy diagnostic)
        H = np.concatenate(hmean, 0)
        ev = np.linalg.eigvalsh(H.T @ H / len(H))
        pr = float((ev.sum() ** 2) / max((ev ** 2).sum(), 1e-12))
        return {
            "n_molecules": int(len(M)),
            "pairs_per_molecule_mean": float((n.sum(1)).mean()),
            "pairs_per_molecule_min": float(n.sum(1).min()),
            "pairs_per_molecule_max": float(n.sum(1).max()),
            "category_nonempty_fraction": ex.mean(0).tolist(),
            "category_mean_count": n.mean(0).tolist(),
            "molecules_with_ge2_nonempty_categories": frac_two,
            "molecules_with_content_difference_R_vs_N": frac_diff,
            "mean_R_vs_N_content_l2_over_nonempty": mean_nonempty,
            "R_content_rms": float(np.sqrt((M ** 2).mean())),
            "ridge_R2_mean_all_dims": float(np.mean(r2)),
            "ridge_R2_mean_valid_dims": float(r2[valid_dim].mean()),
            "ridge_R2_per_category": per_cat,
            "ridge_R2_common_categories_mean": common_mean,
            "ridge_R2_valid_dims_per_category": valid_cat.sum(1).tolist(),
            "ridge_R2_permutation_control_mean": float(np.mean(r2_control[valid_dim])),
            "h_participation_ratio_mean_at_init": pr,
            "X_dim": int(X.shape[1]),
        }

    ck.run("G6_real_relation_coverage_and_redundancy", g6)

    # ------------------------------------------------------------------
    # G7 - finite gradients, short descent, checkpoint round-trip
    # ------------------------------------------------------------------
    def g7():
        rows = np.arange(64, dtype=np.int64)
        cache = train
        out = {}
        for arm in ("A", "R"):
            model, info = M.build_model(arm, norm, 0, "cpu")
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
            losses = []
            grads_step = {}
            for step in range(30):
                batch = D.collate(cache, rows[step % 2 * 32:(step % 2 + 1) * 32])
                target = torch.as_tensor(cache["labels"][rows[step % 2 * 32:(step % 2 + 1) * 32]],
                                         dtype=torch.float32)
                model.train()
                pred = model(batch)
                loss = F.l1_loss(pred, target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if step < 2:
                    grads_step[step] = {
                        n: float(p.grad.abs().sum()) for n, p in model.named_parameters()
                    }
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                losses.append(float(loss))
            first5 = float(np.mean(losses[:5]))
            last5 = float(np.mean(losses[-5:]))
            assert last5 < first5
            if arm == "R":
                assert grads_step[0]["rel.head.weight"] > 0
                assert grads_step[0]["rel.phi.lins.0.weight"] == 0.0
                assert grads_step[1]["rel.phi.lins.0.weight"] > 0
            # checkpoint round-trip
            state = L.C.cpu_state(model)
            model2 = M.load_model_from_state(arm, norm, state, "cpu")
            p1 = _forward_pred(model, cache, rows[:8])
            p2 = _forward_pred(model2, cache, rows[:8])
            assert float(np.max(np.abs(p1 - p2))) == 0.0
            out[arm] = {
                "parameters": info["parameters"],
                "loss_first5": first5,
                "loss_last5": last5,
                "descent": first5 - last5,
                "head_grad_step0": grads_step.get(0, {}).get("rel.head.weight"),
                "phi_grad_step0": grads_step.get(0, {}).get("rel.phi.lins.0.weight"),
                "phi_grad_step1": grads_step.get(1, {}).get("rel.phi.lins.0.weight"),
                "checkpoint_reload_max_abs": float(np.max(np.abs(p1 - p2))),
            }
        return out

    ck.run("G7_gradients_short_descent_checkpoint", g7)

    # ------------------------------------------------------------------
    # G8 - official-test isolation
    # ------------------------------------------------------------------
    def g8():
        train_src = (Path(__file__).parent / "lc_train.py").read_text()
        data_src = (Path(__file__).parent / "lc_data.py").read_text()
        assert "official_test_loaded" in train_src
        assert "official_test_loaded\": False" in train_src
        assert 'split not in ("train", "valid")' in data_src
        # prep refuses test even if someone asks for it
        refused = False
        try:
            D.prep(("test",))
        except Exception:
            refused = True
        assert refused
        # no test cache exists yet
        test_cache_exists = D.cache_path("test").exists()
        assert not test_cache_exists
        # cache meta labels only for train/valid
        assert train["meta"]["labels_loaded"] is True
        assert valid["meta"]["labels_loaded"] is True
        prep_meta = D.load_cache("train")["meta"]
        assert not any(k.startswith("junc") for k in train.keys())
        return {
            "prep_splits": ["train", "valid"],
            "test_prep_refused": True,
            "test_cache_exists": test_cache_exists,
            "official_test_loaded": False,
            "train_labels_loaded": bool(prep_meta["labels_loaded"]),
            "labels_sha256": prep_meta["labels_sha256"],
        }

    ck.run("G8_official_test_isolation", g8)

    result = {
        "round": L.ROUND_ID,
        "git_commit": L.C.git_commit(),
        "environment": L.env_summary(),
        "prep": prep,
        "all_pass": ck.ok,
        "checks": ck.results,
        "created": L.now(),
    }
    L.write_json(L.ANALYSIS / "cpu_checks.json", result)
    L.log(f"CPU checks: {sum(r['pass'] for r in ck.results)}/{len(ck.results)} pass")
    return 0 if ck.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
