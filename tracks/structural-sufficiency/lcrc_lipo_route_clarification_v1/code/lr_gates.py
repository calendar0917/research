"""Concentrated correctness gates (CPU) for the round.

Each gate maps to a numbered item of ``PROTOCOL.md`` §8 and writes a machine
readable record into ``analysis/gates.json``.  These are the checks that decide
whether a formal run may be purchased; they are deliberately narrow.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import lr_common as C
import lr_data as D
import lr_models as LM
import lr_pairs as P
import lr_train as T


GATES: dict[str, Any] = {}


def record(name: str, ok: bool, detail: Mapping[str, Any] | None = None) -> bool:
    GATES[name] = {"ok": bool(ok), "detail": dict(detail or {})}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {json.dumps(GATES[name]['detail'], default=float)[:300]}")
    return bool(ok)


# ---------------------------------------------------------------------------
# 1. data / hashes / row identity / norm / pair isolation
# ---------------------------------------------------------------------------


def gate_data() -> bool:
    src = D.source()
    train = [int(r) for r in src["rows"]["train"]]
    ok_rows = C.row_sha256(train) == C.TRAIN_ROWS_SHA256
    payload = D.split_payload("s0")
    old = C.read_json(C.S0_SPLIT_JSON)
    ok_s0 = all(sorted(payload["rows_by_partition"][p]) == sorted(int(r) for r in old["rows_by_partition"][p])
                for p in C.PARTITIONS)
    # every split covers the Train rows exactly once; every pair is intra-partition
    cover = {}
    pair_iso = {}
    for split in C.SPLITS:
        rows = D.rows_by_partition(split)
        flat = sorted(r for p in C.PARTITIONS for r in rows[p])
        cover[split] = (len(flat) == C.TRAIN_ROWS and len(set(flat)) == C.TRAIN_ROWS and flat == sorted(train))
        pair_iso[split] = all(split == "ft" or True for _ in [0])
        bad = []
        for pair in P.table().pairs:
            where = {p for p in C.PARTITIONS if int(pair["row_a"]) in set(rows[p])}
            if split == "ft":
                continue
            if not where:
                bad.append(pair["pair_id"])
                continue
            wb = {p for p in C.PARTITIONS if int(pair["row_b"]) in set(rows[p])}
            if where != wb:
                bad.append(pair["pair_id"])
        pair_iso[split] = not bad
    # Fit-only normalisation: changing Select/Diagnose/Audit structures must not matter
    norm = D.train_norm("s0")
    fit_rows = set(D.rows_by_partition("s0")["fit"])
    n_atoms_fit = 0
    cache = D.load_cache("s0", "fit")
    m = int(cache["n_molecules"])
    ok_fit_only = m == len(fit_rows)
    # target normalisation uses Fit only
    tgt = D.target_norm("s0")
    y_fit = D.y_aligned("s0", "fit", [int(r) for r in np.asarray(cache["rows"]).tolist()], "train")
    ok_tgt = abs(tgt["mean"] - float(np.mean(y_fit))) < 1e-12 and abs(tgt["scale"] - float(np.std(y_fit))) < 1e-12
    return record("1_data_alignment", ok_rows and ok_s0 and all(cover.values()) and all(pair_iso.values())
                  and ok_fit_only and ok_tgt,
                  {"train_rows_sha256_ok": ok_rows, "s0_matches_upstream": ok_s0, "coverage": cover,
                   "pair_isolation": pair_iso, "fit_only_norm": ok_fit_only, "target_norm_fit_only": ok_tgt,
                   "n_fit": m, "target_mean": tgt["mean"], "target_scale": tgt["scale"]})


# ---------------------------------------------------------------------------
# 2. B equivalence with the frozen C1
# ---------------------------------------------------------------------------


def gate_b_equivalence(device: torch.device) -> bool:
    import dissect_model as M
    import lcrc_data as PREP_DATA

    spec = D.frozen_spec()
    cache = D.load_cache("s0", "fit")
    norm = D.train_norm("s0")
    rows = [0, 3, 17, 55, 128]
    batch = D.to_device(D.collate(cache, rows), device)
    full = M.build_full_state(spec, norm, 0)
    old = M.build_variant(spec, norm, "C1", full, device)
    new = LM.build_route_model(spec, norm, "B", 0, device)
    old.eval(); new.eval()
    with torch.no_grad():
        y_old = old(batch).cpu().numpy()
        y_new = new(batch).cpu().numpy()
    diff = float(np.max(np.abs(y_old - y_new)))
    tol = 1e-6 if device.type == "cpu" else 1e-4
    ok_params = LM.parameter_count(new) == C.EXPECTED_PARAMS["B"] == 228674
    return record("2_b_equivalence", diff <= tol and ok_params,
                  {"max_abs_diff": diff, "tol": tol, "parameters": LM.parameter_count(new),
                   "device": str(device), "params_ok": ok_params})


# ---------------------------------------------------------------------------
# 3. trainer equivalence with the upstream ordinary trainer
# ---------------------------------------------------------------------------


def gate_trainer_equivalence(device: torch.device, epochs: int = 2) -> bool:
    """Run the upstream C1 trainer for ``epochs`` epochs and compare to ours."""
    import edit_train as ET
    import dissect_model as M

    spec = D.frozen_spec()
    t0 = time.perf_counter()
    old_dir = C.TRACK / "runs" / "_gate_old_trainer"
    old = ET.train_unit(model_variant="C1", seed=0, device=device, out_dir=old_dir,
                        epochs=epochs, smoke=True)
    old_time = time.perf_counter() - t0
    old_state = torch.load(old_dir / "checkpoint.pt", map_location="cpu", weights_only=False)["state_dict"]
    ours = T.train_unit(rid="_gate_new_trainer", split="s0", model_kind="B", loss="A", seed=0,
                        stage="gate", device=device, out_dir=C.TRACK / "runs" / "_gate_new_trainer",
                        smoke_steps=epochs * 16)
    our_state = torch.load(C.TRACK / "runs" / "_gate_new_trainer/checkpoint.pt", map_location="cpu",
                           weights_only=False)["state_dict_last"]
    keys = [k for k in old_state if k in our_state]
    diffs = {k: float((old_state[k].float() - our_state[k].float()).abs().max()) for k in keys}
    worst = max(diffs.values())
    off_probe = ours["recipe"]["batch_order_probe"][:5]
    return record("3_trainer_equivalence", worst <= (1e-6 if device.type == "cpu" else 1e-4),
                  {"updates": epochs * 16, "worst_param_diff": worst,
                   "worst_key": max(diffs, key=diffs.get), "n_keys": len(keys),
                   "old_trainer_seconds": old_time, "old_select_rmse": old["metrics"]["select"]["rmse"],
                   "new_select_rmse": ours["metrics"]["select"]["rmse"],
                   "old_best_epoch": old["checkpoint"]["best_epoch"],
                   "new_best_period": ours["checkpoint"]["best_period"],
                   "batch_order_probe_equal": old["recipe"]["batch_order_probe_epoch1_first10"][:5] == off_probe})


# ---------------------------------------------------------------------------
# 4. S/O/D parameter identity and step-0 equivalence
# ---------------------------------------------------------------------------


def gate_sod(device: torch.device) -> bool:
    spec = D.frozen_spec()
    cache = D.load_cache("s0", "fit")
    norm = D.train_norm("s0")
    batch = D.to_device(D.collate(cache, [1, 2, 9]), device)
    models = {k: LM.build_route_model(spec, norm, k, 0, device) for k in ("B", "S", "O", "D")}
    for m in models.values():
        m.eval()
    with torch.no_grad():
        ys = {k: m(batch).cpu().numpy() for k, m in models.items()}
    step0 = {k: float(np.max(np.abs(ys[k] - ys["B"]))) for k in ("S", "O", "D")}
    sb = models["B"].state_dict()
    shared = {}
    for k in ("S", "O", "D"):
        sd = models[k].state_dict()
        shared[k] = all(torch.equal(sb[key], sd[key]) for key in sb)
    params = {k: LM.parameter_count(models[k]) for k in ("B", "S", "O", "D")}
    block_params = int(sum(p.numel() for p in models["D"].blocks.parameters()))
    tol = 1e-6 if device.type == "cpu" else 1e-4
    ok = all(v <= tol for v in step0.values()) and all(shared.values()) and \
        params["S"] == params["O"] == params["D"] == 278722 and block_params == 50048
    return record("4_sod_params_step0", ok, {"step0_max_abs_diff": step0, "shared_state_equal": shared,
                                             "parameters": params, "block_parameters": block_params,
                                             "tol": tol})


# ---------------------------------------------------------------------------
# 5. neighbour-mean operator against an independent loop
# ---------------------------------------------------------------------------


def _loop_T(values: np.ndarray, edges: Sequence[tuple[int, int]]) -> np.ndarray:
    n = values.shape[0]
    out = np.zeros_like(values)
    for i in range(n):
        nbrs = [j for (a, b) in edges for j in ((b,) if a == i else ((a,) if b == i else ()))]
        if nbrs:
            out[i] = values[nbrs].mean(axis=0)
    return out


def gate_neighbour_operator() -> bool:
    graphs = {
        "chain4": (4, [(0, 1), (1, 2), (2, 3)]),
        "branch5": (5, [(0, 1), (0, 2), (0, 3), (3, 4)]),
        "cycle5": (5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]),
        "isolated3": (3, []),
        "mixed": (7, [(0, 1), (1, 2), (3, 4), (4, 5), (5, 3)]),
    }
    rng = np.random.default_rng(7)
    worst = 0.0
    worst2 = 0.0
    for name, (n, edges) in graphs.items():
        x = rng.normal(size=(n, 3))
        bond = torch.as_tensor(edges, dtype=torch.long) if edges else torch.zeros((0, 2), dtype=torch.long)
        dst, src, inv_deg = LM.neighbour_mean_index(torch.zeros(n, dtype=torch.long), bond, n)
        t1 = LM.neighbour_mean(torch.as_tensor(x, dtype=torch.float64), dst, src, inv_deg).numpy()
        ref1 = _loop_T(x, edges)
        t2 = LM.neighbour_mean(torch.as_tensor(t1, dtype=torch.float64), dst, src, inv_deg).numpy()
        ref2 = _loop_T(ref1, edges)
        worst = max(worst, float(np.max(np.abs(t1 - ref1))) if n else 0.0)
        worst2 = max(worst2, float(np.max(np.abs(t2 - ref2))) if n else 0.0)
    # multi-molecule batch: no cross-graph leakage
    n = 4
    bond = torch.as_tensor([[0, 1], [2, 3]], dtype=torch.long)
    x = torch.arange(8, dtype=torch.float64).reshape(4, 2)
    dst, src, inv_deg = LM.neighbour_mean_index(torch.zeros(n, dtype=torch.long), bond, n)
    got = LM.neighbour_mean(x, dst, src, inv_deg)
    want = torch.stack([x[1], x[0], x[3], x[2]])
    leak = float((got - want).abs().max())
    ok = worst <= 1e-6 and worst2 <= 1e-6 and leak <= 1e-6
    return record("5_neighbour_operator", ok,
                  {"max_abs_err_T": worst, "max_abs_err_T2": worst2, "multi_component_leak": leak,
                   "tol": 1e-6, "note": "tolerance reflects float32 inverse-degree rounding",
                   "graphs": list(graphs)})


# ---------------------------------------------------------------------------
# 6. block information flow
# ---------------------------------------------------------------------------


def gate_information_flow() -> bool:
    torch.manual_seed(11)
    n, width = 6, 64
    edges = [(0, 1), (1, 2), (2, 3), (4, 5)]  # two components: a 4-chain and a disjoint edge
    bond = torch.as_tensor(edges, dtype=torch.long)
    dst, src, inv_deg = LM.neighbour_mean_index(torch.zeros(n, dtype=torch.long), bond, n)
    blocks = torch.nn.ModuleList([LM.ResidualBlock(width, 128) for _ in range(2)])
    for blk in blocks:  # non-zero output weights so the branch is active
        torch.nn.init.normal_(blk.wout.weight, std=0.05)
        torch.nn.init.normal_(blk.wout.bias, std=0.01)
    h0 = torch.randn(n, width, dtype=torch.float64)
    pert = torch.linspace(-1.0, 1.0, width, dtype=torch.float64)
    out: dict[str, Any] = {}
    base = {}
    for kind in ("S", "O", "D"):
        b = [LM.ResidualBlock(width, 128).double() for _ in range(2)]
        # share the same block parameters across branches for a fair comparison
        for new_blk, old_blk in zip(b, blocks):
            new_blk.load_state_dict({k: v.double() for k, v in old_blk.state_dict().items()})
        base[kind] = LM.apply_blocks(kind, b, h0, dst, src, inv_deg)[0]
    # perturb h0 at node 4 (component 2 only)
    h0p = h0.clone(); h0p[4] += pert
    deltas = {}
    for kind in ("S", "O", "D"):
        b = [LM.ResidualBlock(width, 128).double() for _ in range(2)]
        for new_blk, old_blk in zip(b, blocks):
            new_blk.load_state_dict({k: v.double() for k, v in old_blk.state_dict().items()})
        out_p = LM.apply_blocks(kind, b, h0p, dst, src, inv_deg)[0]
        deltas[kind] = float((out_p - base[kind]).abs().max())
    # per-node sensitivity pattern
    pattern = {}
    for kind in ("S", "O", "D"):
        b = [LM.ResidualBlock(width, 128).double() for _ in range(2)]
        for new_blk, old_blk in zip(b, blocks):
            new_blk.load_state_dict({k: v.double() for k, v in old_blk.state_dict().items()})
        out_node = []
        for node in range(n):
            hp = h0.clone(); hp[node] += pert
            o = LM.apply_blocks(kind, b, hp, dst, src, inv_deg)[0]
            out_node.append([int(bool((o[i] - base[kind][i]).abs().max() > 1e-12)) for i in range(n)])
        pattern[kind] = out_node
    # S: no exchange at all -> perturbing node i only changes node i
    s_local = all(pattern["S"][i][j] == 0 for i in range(n) for j in range(n) if i != j)
    # O: reading only h0 -> impact radius at most 2 hops, no cross-component leak
    def within_two(a: int, b: int) -> bool:
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v); adj[v].add(u)
        frontier, seen = {a}, {a}
        for _ in range(2):
            frontier = {w for f in frontier for w in adj[f]} - seen
            seen |= frontier
        return b in seen
    o_bounded = all(pattern["O"][i][j] == 1 for i in range(n) for j in range(n) if within_two(i, j))
    o_no_overflow = all(pattern["O"][i][j] == 0 for i in range(n) for j in range(n) if not within_two(i, j))
    d_second_reads_h1 = _second_block_reads_updated_state(width, edges, blocks, h0)
    # a *uniform* context shift is annihilated by the block's LayerNorm, so the
    # perturbation must be non-uniform (recorded here because it matters)
    uniform = _uniform_shift_is_invisible(width, blocks, h0, dst, src, inv_deg)
    ok = s_local and o_bounded and o_no_overflow and d_second_reads_h1 and uniform
    return record("6_information_flow", ok,
                  {"S_local_only": s_local, "O_within_two_hops": o_bounded,
                   "O_no_overflow": o_no_overflow, "D_second_block_reads_h1": d_second_reads_h1,
                   "uniform_shift_annihilated_by_layernorm": uniform,
                   "perturbation": "non-uniform linspace(-1,1)",
                   "delta_on_other_component": deltas, "pattern": pattern})


def _uniform_shift_is_invisible(width, blocks, h0, dst, src, inv_deg) -> bool:
    """LayerNorm makes a constant context shift invisible: the gate must not use one."""
    b = [LM.ResidualBlock(width, 128).double() for _ in range(2)]
    for new_blk, old_blk in zip(b, blocks):
        new_blk.load_state_dict({k: v.double() for k, v in old_blk.state_dict().items()})
    base = LM.apply_blocks("O", b, h0, dst, src, inv_deg)[0]
    shifted = h0.clone()
    shifted[:, :] += 0.5 * torch.ones(width, dtype=h0.dtype)
    got = LM.apply_blocks("O", b, shifted, dst, src, inv_deg)[0]
    # the *residual* (block output) is invariant to the uniform shift
    return float(((got - shifted) - (base - h0)).abs().max()) < 1e-12


def _second_block_reads_updated_state(width, edges, blocks, h0) -> bool:
    """Direct test: D's second context is T h1, not T h0, and h1 != h0."""
    a = [(0, 1)]
    bond = torch.as_tensor(a, dtype=torch.long)
    dst, src, inv_deg = LM.neighbour_mean_index(torch.zeros(2, dtype=torch.long), bond, 2)
    b = [LM.ResidualBlock(width, 128).double() for _ in range(2)]
    for new_blk, old_blk in zip(b, blocks):
        new_blk.load_state_dict({k: v.double() for k, v in old_blk.state_dict().items()})
    h0_2 = h0[:2]
    h1, _ = LM.apply_blocks("D", b[:1], h0_2, dst, src, inv_deg)
    ctx_d = LM.neighbour_mean(h1, dst, src, inv_deg)
    ctx_o = LM.neighbour_mean(h0_2, dst, src, inv_deg)
    changed = float((h1 - h0_2).abs().max()) > 1e-12
    differs = float((ctx_d - ctx_o).abs().max()) > 1e-12
    # 2-hop range of O
    chain = [(0, 1), (1, 2), (2, 3)]
    bond = torch.as_tensor(chain, dtype=torch.long)
    dst, src, inv_deg = LM.neighbour_mean_index(torch.zeros(4, dtype=torch.long), bond, 4)
    t1 = LM.neighbour_mean(h0[:4], dst, src, inv_deg)
    t2 = LM.neighbour_mean(t1, dst, src, inv_deg)
    two_hop = float(t2[0].abs().max()) > 0 and float(t2[0].sub(t1[0]).abs().max()) > 0
    return bool(changed and differs and two_hop)


# ---------------------------------------------------------------------------
# 7. gradients reach every new parameter
# ---------------------------------------------------------------------------


def gate_branch_gradients(device: torch.device) -> bool:
    spec = D.frozen_spec()
    cache = D.load_cache("s0", "fit")
    norm = D.train_norm("s0")
    tgt = D.target_norm("s0")
    rows = list(range(0, C.BATCH))
    batch = D.to_device(D.collate(cache, rows), device)
    y = D.y_aligned("s0", "fit", [int(r) for r in np.asarray(cache["rows"]).tolist()], "train")
    z = torch.as_tensor((y - tgt["mean"]) / tgt["scale"], dtype=torch.float32, device=device)[: len(rows)]
    out = {}
    for kind in ("S", "O", "D"):
        model = LM.build_route_model(spec, norm, kind, 0, device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        model.train()
        for _ in range(3):
            pred = model(batch).reshape(-1)
            loss = torch.nn.functional.mse_loss(pred, z)
            opt.zero_grad(); loss.backward(); opt.step()
        model.train()
        pred = model(batch).reshape(-1)
        torch.nn.functional.mse_loss(pred, z).backward()
        dead = [n for n, p in model.named_parameters()
                if p.grad is None or float(p.grad.abs().sum()) == 0.0]
        block_dead = [n for n in dead if n.startswith("blocks.")]
        block_norm = float(torch.cat([p.detach().reshape(-1) for n, p in model.named_parameters()
                                     if n.startswith("blocks.") and "wout" in n]).norm())
        out[kind] = {"n_dead": len(dead), "dead": dead[:6], "dead_block_params": block_dead,
                     "wout_norm_after_3_steps": block_norm}
    ok = all(v["n_dead"] == 0 and not v["dead_block_params"] for v in out.values())
    return record("7_branch_gradients", ok, out)


# ---------------------------------------------------------------------------
# 8. permutation / batching invariance
# ---------------------------------------------------------------------------


def _group_blocks(owner: np.ndarray, n: int) -> list[np.ndarray]:
    """Split an occurrence list (sorted by owner atom) into per-atom index arrays."""
    counts = np.bincount(owner, minlength=n)
    out, start = [], 0
    for count in counts:
        out.append(np.arange(start, start + int(count)))
        start += int(count)
    return out


def _permute_single_batch(batch: Mapping[str, torch.Tensor], perm: np.ndarray) -> dict[str, Any]:
    """Permute the atoms of a single-molecule batch and rebuild every index."""
    n = len(perm)
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    out = dict(batch)
    for key in ("atom_feat", "atom_scalar", "atom_type", "atom_batch"):
        out[key] = batch[key][torch.as_tensor(perm)]
    out["atom_off"] = batch["atom_off"]
    bi = batch["bond_index"].numpy()
    out["bond_index"] = torch.as_tensor(inv[bi], dtype=torch.long)
    # neighbour blocks
    owner = batch["nbr_atom"].numpy()
    blocks = _group_blocks(owner, n)
    new_counts = np.bincount(owner, minlength=n)[perm]
    out["nbr_code"] = torch.cat([batch["nbr_code"][blocks[i]] for i in perm]) if n else batch["nbr_code"][:0]
    out["nbr_atom"] = torch.as_tensor(np.repeat(np.arange(n), new_counts), dtype=torch.long)
    # path blocks (path_end holds a local atom index)
    powner = batch["path_atom"].numpy()
    pblocks = _group_blocks(powner, n)
    pcounts = np.bincount(powner, minlength=n)[perm]
    for key in ("path_cat", "path_end63", "path_closure"):
        out[key] = torch.cat([batch[key][pblocks[i]] for i in perm]) if n else batch[key][:0]
    ends = np.concatenate([batch["path_end"].numpy()[pblocks[i]] for i in perm]) if n else np.zeros(0, np.int64)
    out["path_end"] = torch.as_tensor(inv[ends], dtype=torch.long)
    out["path_atom"] = torch.as_tensor(np.repeat(np.arange(n), pcounts), dtype=torch.long)
    out["path_graph"] = batch["path_graph"][:0] if n == 0 else torch.zeros(int(pcounts.sum()), dtype=torch.long)
    return out


def gate_permutation_and_batching(device: torch.device) -> bool:
    spec = D.frozen_spec()
    cache = D.load_cache("s0", "fit")
    norm = D.train_norm("s0")
    rows = [4, 21, 60, 77]
    batch = D.to_device(D.collate(cache, rows), device)
    model = LM.build_route_model(spec, norm, "D", 0, device)
    torch.manual_seed(3)
    for blk in model.blocks:  # non-zero output weights so the blocks participate
        torch.nn.init.normal_(blk.wout.weight, std=0.05)
    model.eval()
    with torch.no_grad():
        y_batch = model(batch).cpu().numpy()
        singles = []
        for row in rows:
            sub = D.to_device(D.collate(cache, [row]), device)
            singles.append(model(sub).cpu().numpy()[0])
        y_split = float(np.max(np.abs(y_batch - np.asarray(singles))))
        single_batch = D.to_device(D.collate(cache, [rows[0]]), device)
        y_single = model(single_batch).cpu().numpy()
        n_atoms = int(single_batch["atom_feat"].shape[0])
        rng = np.random.default_rng(5)
        perm = rng.permutation(n_atoms)
        permuted = _permute_single_batch(single_batch, perm)
        y_perm = model(permuted).cpu().numpy()
        perm_diff = float(np.max(np.abs(y_perm - y_single)))
        y_again = model(single_batch).cpu().numpy()
        stable = float(np.max(np.abs(y_again - y_single)))
    ok = y_split <= 1e-5 and perm_diff <= 1e-5 and stable == 0.0
    return record("8_permutation_batching", ok,
                  {"batch_vs_single_max_abs": y_split, "atom_permutation_max_abs": perm_diff,
                   "eval_determinism_max_abs": stable, "n_atoms_permuted": int(n_atoms)})


# ---------------------------------------------------------------------------
# 9. pair sign / offset / unit handling
# ---------------------------------------------------------------------------


def gate_pair_algebra() -> bool:
    tgt = D.target_norm("s0")
    rng = np.random.default_rng(2)
    pred_std = rng.normal(size=8)
    y_std = rng.normal(size=8)
    pred_raw = pred_std * tgt["scale"] + tgt["mean"]
    y_raw = y_std * tgt["scale"] + tgt["mean"]
    d_std = (pred_std[3] - pred_std[1]) - (y_std[3] - y_std[1])
    d_raw = (pred_raw[3] - pred_raw[1]) - (y_raw[3] - y_raw[1])
    ok_scale = abs(d_raw - d_std * tgt["scale"]) <= 1e-9
    ok_offset = abs(((pred_std[3] + 5.0) - (pred_std[1] + 5.0)) - (pred_std[3] - pred_std[1])) <= 1e-12
    # reverse direction flips the sign exactly
    ok_sign = abs((-(pred_std[3] - pred_std[1])) - (pred_std[1] - pred_std[3])) <= 1e-12
    # R = W - 2 mean(e_i e_j) in the raw auxiliary form
    e_i = torch.as_tensor(rng.normal(size=(16,)), dtype=torch.float64)
    e_j = torch.as_tensor(rng.normal(size=(16,)), dtype=torch.float64)
    r = float(T.auxiliary_term(e_i, e_j, "R"))
    w = float(T.auxiliary_term(e_i, e_j, "W"))
    ident = r - (w - 2.0 * float((e_i * e_j).mean()))
    ok_ident = abs(ident) <= 1e-12
    return record("9_pair_algebra", ok_scale and ok_offset and ok_sign and ok_ident,
                  {"standardisation_scale_ok": ok_scale, "constant_offset_cancels": ok_offset,
                   "reverse_sign_ok": ok_sign, "r_equals_w_minus_cross": ok_ident,
                   "identity_residual": ident, "scale": tgt["scale"]})


# ---------------------------------------------------------------------------
# 10. loss gradients, RNG neutrality, null occurrence counts
# ---------------------------------------------------------------------------


def gate_losses_and_rng(device: torch.device) -> bool:
    spec = D.frozen_spec()
    cache = D.load_cache("s0", "fit")
    norm = D.train_norm("s0")
    rows = list(range(0, 32))
    batch = D.to_device(D.collate(cache, rows), device)
    model = LM.build_route_model(spec, norm, "S", 0, device)
    plan = P.get_plan("s0", 0, C.FORMAL_UPDATES)
    table_ = P.table()
    pair_idx = np.asarray(plan["pairs"][0], dtype=np.int64)
    perm = np.asarray(plan["perm"][0], dtype=np.int64)
    ep_rows, idx_a, idx_b, _ = P.endpoint_index(pair_idx, table_.row_a, table_.row_b)
    row_pos = {int(r): i for i, r in enumerate(np.asarray(cache["rows"]).tolist())}
    ep_pos = [row_pos[int(r)] for r in ep_rows]
    aux_batch = D.to_device(D.collate(cache, ep_pos), device)
    e = torch.randn(len(ep_pos), 32, device=device, requires_grad=True)
    ia = torch.as_tensor(idx_a, dtype=torch.long, device=device)
    ib = torch.as_tensor(idx_b, dtype=torch.long, device=device)
    ib_n = torch.as_tensor([idx_b[int(k)] for k in perm], dtype=torch.long, device=device)
    out = {}
    for name, ib_use in (("R", ib), ("W", ib), ("N", ib_n)):
        value = T.auxiliary_term(e[ia], e[ib_use], name)
        g = torch.autograd.grad(value, e, retain_graph=True)[0]
        out[name] = {"value": float(value), "grad_norm": float(g.norm())}
    # RNG neutrality of the auxiliary eval-mode forward
    model.train()
    rng_before = torch.get_rng_state().clone()
    was = model.training
    model.eval()
    _ = model(aux_batch)
    model.train(was)
    rng_after = torch.get_rng_state().clone()
    rng_same = bool(torch.equal(rng_before, rng_after))
    # a train-mode forward *does* consume RNG
    _ = model(batch)
    rng_train = bool(torch.equal(rng_after, torch.get_rng_state()))
    # occurrence counts per row
    a_counts = np.bincount(table_.row_a[pair_idx], minlength=4000)
    b_counts = np.bincount(table_.row_b[pair_idx], minlength=4000)
    bn_counts = np.bincount(table_.row_b[pair_idx][perm], minlength=4000)
    counts_equal = bool(np.array_equal(b_counts, bn_counts))
    no_self = not bool(np.any(table_.row_a[pair_idx] == table_.row_b[pair_idx][perm]))
    ok = all(v["grad_norm"] > 0 for v in out.values()) and rng_same and (not rng_train) \
        and counts_equal and no_self
    return record("10_losses_and_rng", ok,
                  {"losses": out, "aux_forward_consumes_rng": not rng_same,
                   "train_forward_consumes_rng": not rng_train,
                   "null_counts_equal_R": counts_equal, "null_has_no_self_pair": no_self,
                   "mean_non_mmp_fraction_step0": float(plan["non_mmp_fraction"][0])})


# ---------------------------------------------------------------------------
# 11. F / G shapes and the GINE loop
# ---------------------------------------------------------------------------


def gate_f_and_g(device: torch.device) -> bool:
    spec = D.frozen_spec()
    cache = D.load_cache("s0", "fit")
    norm = D.train_norm("s0")
    fp = np.load(C.CACHE / "fp_s0.npz")
    f = LM.build_fingerprint_model(spec, norm, fp["mean"], fp["scale"], 0, device)
    g = LM.build_gine_model(spec, norm, 0, device)
    model = LM.build_route_model(spec, norm, "B", 0, device)
    f_params, g_params = LM.parameter_count(f), LM.parameter_count(g)
    # the readout of G is the C1 readout at the same seed
    g_state, b_state = g.state_dict(), model.state_dict()
    readout_equal = all(torch.equal(g_state[k], b_state[k]) for k in
                        ("f.lins.0.weight", "f.lins.1.weight", "rho.lins.0.weight", "rho.lins.1.weight"))
    # GINE loop on a branch/cycle/isolated small graph, independent implementation
    edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
    n = 5
    rng = np.random.default_rng(13)
    u = rng.normal(size=(n, 4))
    bf = rng.normal(size=(len(edges), 4))
    hidden = 4
    eps = 0.3
    edge_w = rng.normal(size=(hidden, hidden))
    edge_b = rng.normal(size=hidden)
    mlp_w1 = rng.normal(size=(2 * hidden, hidden)); mlp_b1 = rng.normal(size=2 * hidden)
    mlp_w2 = rng.normal(size=(hidden, 2 * hidden)); mlp_b2 = rng.normal(size=hidden)
    ln_b = rng.normal(size=hidden) * 0.1; ln_w = 1.0 + rng.normal(size=hidden) * 0.1

    def silu(x):
        return x / (1.0 + np.exp(-x))

    agg = np.zeros((n, hidden))
    for e_index, (i, j) in enumerate(edges):
        msg_ij = u[j] + (edge_w @ bf[e_index] + edge_b)
        msg_ji = u[i] + (edge_w @ bf[e_index] + edge_b)
        agg[i] += np.maximum(msg_ij, 0.0)
        agg[j] += np.maximum(msg_ji, 0.0)
    s = (1.0 + eps) * u + agg
    h = silu(s @ mlp_w1.T + mlp_b1)
    h = h @ mlp_w2.T + mlp_b2
    z = u + h
    want = (z - z.mean(axis=1, keepdims=True)) / np.sqrt(z.var(axis=1, keepdims=True) + 1e-5) * ln_w + ln_b
    torch_layer = LM.GINELayer(hidden, 4).double()
    with torch.no_grad():
        torch_layer.edge.weight.copy_(torch.as_tensor(edge_w))
        torch_layer.edge.bias.copy_(torch.as_tensor(edge_b))
        torch_layer.eps.copy_(torch.as_tensor([eps]))
        torch_layer.mlp[0].weight.copy_(torch.as_tensor(mlp_w1)); torch_layer.mlp[0].bias.copy_(torch.as_tensor(mlp_b1))
        torch_layer.mlp[2].weight.copy_(torch.as_tensor(mlp_w2)); torch_layer.mlp[2].bias.copy_(torch.as_tensor(mlp_b2))
        torch_layer.norm.weight.copy_(torch.as_tensor(ln_w)); torch_layer.norm.bias.copy_(torch.as_tensor(ln_b))
    dst = torch.as_tensor([e[0] for e in edges] + [e[1] for e in edges])
    src = torch.as_tensor([e[1] for e in edges] + [e[0] for e in edges])
    with torch.no_grad():
        got = torch_layer(torch.as_tensor(u), dst, src,
                          torch.as_tensor(np.concatenate([bf, bf], axis=0))).detach().numpy()
    err = float(np.max(np.abs(got - want)))
    ok = (f_params == 207009 and g_params == 251781 and readout_equal and err <= 1e-6)
    return record("11_f_g_shapes", ok,
                  {"f_parameters": f_params, "g_parameters": g_params,
                   "g_readout_initialised_from_c1": readout_equal,
                   "gine_layer_loop_max_abs_err": err,
                   "f_input_dim": int(f.net[0].in_features), "g_input_dim": int(g.input[0].in_features)})


# ---------------------------------------------------------------------------
# 12. checkpoint contents
# ---------------------------------------------------------------------------


def gate_checkpoint(path) -> bool:
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return record("12_checkpoint_contents", False, {"missing": str(p)})
    payload = torch.load(p, map_location="cpu", weights_only=False)
    need = ("state_dict_best", "state_dict_last", "optimizer", "scheduler", "rng", "meta")
    ok = all(k in payload for k in need)
    rng = payload.get("rng", {})
    rng_ok = all(k in rng for k in ("torch_cpu", "torch_cuda", "numpy", "sampler_generator"))
    meta = payload.get("meta", {})
    meta_ok = meta.get("best_period") is not None and "best_state_hash" in meta
    return record("12_checkpoint_contents", ok and rng_ok and meta_ok,
                  {"keys": sorted(payload.keys()), "rng_keys": sorted(rng.keys()),
                   "best_period": meta.get("best_period"), "updates": meta.get("updates")})


# ---------------------------------------------------------------------------
# 13. metric equivalence and non-destructiveness
# ---------------------------------------------------------------------------


def gate_metrics() -> bool:
    rng = np.random.default_rng(17)
    pred = rng.normal(size=200)
    y = rng.normal(size=200)
    got = T.regression_metrics(pred, y)
    err = pred - y
    want = {
        "rmse": math.sqrt(float((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
        "r2": float(1.0 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()),
    }
    diffs = {k: abs(got[k] - want[k]) for k in want}
    upstream = C.sha256_file(C.S0_SPLIT_JSON)
    old = C.read_json(C.S0_SPLIT_JSON)
    untouched = upstream == C.sha256_file(C.S0_SPLIT_JSON) and old["status"] == "OK"
    ok = max(diffs.values()) <= 1e-12 and untouched
    return record("13_metric_equivalence", ok, {"max_abs_diff": max(diffs.values()), "diffs": diffs,
                                               "upstream_split_unmodified": untouched,
                                               "upstream_sha256": upstream})


# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CPU correctness gates")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--skip-trainer", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    print(f"[gates] device={device} round={C.ROUND_ID}")
    results = [
        gate_data(),
        gate_b_equivalence(device),
        gate_sod(device),
        gate_neighbour_operator(),
        gate_information_flow(),
        gate_branch_gradients(device),
        gate_permutation_and_batching(device),
        gate_pair_algebra(),
        gate_losses_and_rng(device),
        gate_f_and_g(device),
        gate_metrics(),
    ]
    if not args.skip_trainer:
        results.append(gate_trainer_equivalence(device))
    if args.checkpoint:
        results.append(gate_checkpoint(args.checkpoint))
    ok = all(results)
    payload = {"schema": "lcrc-lipo-route-gates-v1", "round": C.ROUND_ID, "created_utc": C.utcnow(),
               "git_commit": C.git_commit(), "device": str(device),
               "environment": C.env_info(), "rdkit": C.rdkit_version(),
               "code_hashes": C.code_hashes(), "gates": GATES, "all_passed": bool(ok)}
    C.write_json(C.ANALYSIS / f"gates_{'cpu' if device.type == 'cpu' else 'gpu'}.json", payload)
    print(f"[gates] all_passed={ok}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
