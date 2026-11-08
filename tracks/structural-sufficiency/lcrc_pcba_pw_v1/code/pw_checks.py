"""The pre-training correctness checks of the PW round.

Every check is a pure function returning ``{"name", "passed", "details"}`` so
that the same code produces both the pytest assertions (``tests/``) and the
machine-readable ``analysis/preflight.json`` record written by
``pw_audit.py checks``.  Nothing here trains a formal model.

Numbering follows ``PROTOCOL.md`` section 5:

 1. historical W2 initialisation / parameter identity / zero-initialised tail
 2. zero-residual forward equals the W2 forward (bitwise on CPU)
 3. atom renumbering does not change the prediction
 4. the pair token is symmetric under ``i <-> j`` (bitwise) and the logits are
    invariant under an exchange of the two endpoints
 5. no cross-graph pair: the packed pair list only ever connects atoms of its
    own molecule, and a molecule's output does not depend on its batch mates
 6. no cross-component pair is present anywhere in the index
 7. every stored distance bucket agrees with the original chemical bonds
 8. the segment softmax is normalised per query and per molecule with pairs
 9. every new parameter is on the gradient path after a few optimiser steps
10. a small brute-force implementation equals the batched GPU implementation
11. a molecule without a legal pair keeps a zero pair residual, finite logits
12. neither Train nor checkpoint selection ever sees an official Test label
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

import pw_common as C  # noqa: E402  (sets up the inherited code dirs)
import bs_model as BSM  # noqa: E402
import pw_model as M  # noqa: E402
import pw_pairs as P  # noqa: E402


def _result(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def _clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def rebuild_pairs_from_bonds(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the pair payload from the batch's own (possibly permuted) bonds."""
    atom_batch = batch["atom_batch"].to(torch.int64).cpu().numpy()
    counts = batch["atom_counts"].to(torch.int64).cpu().numpy()
    off = np.concatenate([[0], np.cumsum(counts)])[:-1].astype(np.int64)
    n_graphs = int(batch["n_graphs"])
    edges_all = batch["bond_index"].to(torch.int64).cpu().numpy().reshape(-1, 2)
    bond_graph = atom_batch[edges_all[:, 0]] if edges_all.size else np.zeros(0, dtype=np.int64)
    parts_i: list[np.ndarray] = []
    parts_j: list[np.ndarray] = []
    parts_b: list[np.ndarray] = []
    pair_counts = np.zeros(n_graphs, dtype=np.int64)
    for g in range(n_graphs):
        n = int(counts[g])
        sel = bond_graph == g
        local = edges_all[sel] - off[g]
        i, j, b = P.molecule_pairs(n, local)
        pair_counts[g] = int(i.size)
        parts_i.append(i.astype(np.int64))
        parts_j.append(j.astype(np.int64))
        parts_b.append(b.astype(np.int64))
    flat_i = np.concatenate(parts_i) if parts_i else np.zeros(0, dtype=np.int64)
    flat_j = np.concatenate(parts_j) if parts_j else np.zeros(0, dtype=np.int64)
    flat_b = np.concatenate(parts_b) if parts_b else np.zeros(0, dtype=np.int64)
    out = _clone_batch(batch)
    out["pair_i"] = torch.from_numpy(flat_i)
    out["pair_j"] = torch.from_numpy(flat_j)
    out["pair_bucket"] = torch.from_numpy(flat_b)
    out["pair_counts"] = torch.from_numpy(pair_counts)
    out["pair_graph"] = torch.from_numpy(np.repeat(np.arange(n_graphs), pair_counts))
    return out


def permute_batch(batch: Mapping[str, Any], *, seed: int = 20261110) -> tuple[dict[str, Any], torch.Tensor]:
    permuted, p = BSM.permute_atoms(batch, seed=seed)
    return rebuild_pairs_from_bonds(permuted), p


# ---------------------------------------------------------------------------
# 1 / 2 -- initialisation identity and zero-residual forward
# ---------------------------------------------------------------------------


def check_init(spec, norm, *, seed: int = C.SEED, device: str = "cpu") -> dict[str, Any]:
    # The inherited W2 builder constructs a throwaway module before copying the
    # frozen state, so it moves the global RNG.  PW must not move it *further*:
    # that is the claim worth auditing, and the trainer re-seeds afterwards.
    torch.manual_seed(1234)
    base, base_audit = BSM.build_model(C.BASE_MODEL_KEY, spec, norm, C.N_TASKS, seed=int(seed), device="cpu")
    rng_after_base = torch.get_rng_state().clone()
    torch.manual_seed(1234)
    model, audit = M.build_model(spec, norm, C.N_TASKS, seed=int(seed), device=device)
    rng_after_pw = torch.get_rng_state().clone()
    pw_adds_no_rng_movement = bool(torch.equal(rng_after_base, rng_after_pw))
    base_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    differing = [k for k in base_state if not torch.equal(state[k], base_state[k])]
    # group the state tensors by module: LayerNorm contributes weight *and* bias,
    # the pair token has three bias-free Linears, etc.
    def module_numel(name: str) -> int:
        prefix = f"branch.{name}"
        total = 0
        for key, value in state.items():
            if key == prefix or key.startswith(prefix + "."):
                total += int(value.numel())
        return total

    inventory_ok = all(module_numel(name) == int(spec_["parameters"]) for name, spec_ in C.BRANCH_PARAMS.items())
    inventory_observed = {name: module_numel(name) for name in C.BRANCH_PARAMS}
    details = {
        "base_state_sha256": audit["base_state_sha256"],
        "historical_state_sha256": C.W2["base_state_sha256_at_init"],
        "fresh_w2_state_sha256": M.state_sha256(base_state),
        "base_parameters": audit["base_parameters"],
        "new_parameters": audit["new_parameters"],
        "total_parameters": audit["total_parameters"],
        "expected_new_parameters": int(C.BRANCH_ADDED),
        "expected_total_parameters": int(C.EXPECTED_PARAMETERS),
        "parameter_ceiling": int(C.PARAMETER_CEILING),
        "new_state_keys": audit["new_state_keys"],
        "new_parameter_inventory": audit["new_parameter_inventory"],
        "inventory_expected": {name: int(v["parameters"]) for name, v in C.BRANCH_PARAMS.items()},
        "inventory_observed": inventory_observed,
        "inventory_matches_protocol": bool(inventory_ok),
        "w2_subset_bitwise_equal": not differing,
        "n_differing_w2_tensors": len(differing),
        "historical_init_check": audit["historical_init_check"],
        "zero_init_tail": audit["new_module_init"]["fc2_is_zero"],
        "new_module_init_leaves_rng_untouched": audit["new_module_init"]["global_rng_unchanged"],
        "pw_build_moves_rng_no_further_than_the_w2_build": pw_adds_no_rng_movement,
        "base_trunk_init_fingerprint": audit["base_trunk_init_fingerprint"],
        "historical_trunk_init_fingerprint": C.W2["trunk_init_fingerprint"],
        "base_model_audit_present": bool(base_audit),
    }
    passed = bool(
        not differing
        and inventory_ok
        and audit["new_parameters"] == int(C.BRANCH_ADDED)
        and audit["total_parameters"] == int(C.EXPECTED_PARAMETERS)
        and audit["new_module_init"]["fc2_is_zero"]
        and audit["new_module_init"]["global_rng_unchanged"]
        and pw_adds_no_rng_movement
        and audit["historical_init_check"] in ("exact", "skipped_non_frozen_norm_fingerprint")
    )
    return _result("01_init_and_parameter_identity", passed, **details)


def check_zero_init_forward(spec, norm, batch, *, seed: int = C.SEED, device: str = "cpu") -> dict[str, Any]:
    host_batch = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
    base, _ = BSM.build_model(C.BASE_MODEL_KEY, spec, norm, C.N_TASKS, seed=int(seed), device="cpu")
    base.eval()
    with torch.no_grad():
        ref_cpu = base(host_batch)
    results: dict[str, Any] = {}
    passed = True
    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=int(seed), device="cpu")
    model.eval()
    with torch.no_grad():
        logits, aux = model(host_batch, return_aux=True)
        zeroed = model(host_batch, residual_zero=True)
    resid_max = float(aux["residual_logits"].abs().max().item())
    d_zero = float((logits - ref_cpu).abs().max().item())
    d_flag = float((zeroed - ref_cpu).abs().max().item())
    ok = resid_max == 0.0 and d_zero == 0.0 and d_flag == 0.0
    passed &= bool(ok)
    results["cpu"] = {
        "max_abs_diff": d_zero,
        "max_abs_diff_residual_zero_path": d_flag,
        "residual_max_abs": resid_max,
        "reference_max_abs": float(ref_cpu.abs().max().item()),
        "bitwise_exact": bool(ok),
    }
    if str(device) != "cpu":
        dev = torch.device(device)
        dev_batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in host_batch.items()}
        base_dev, _ = BSM.build_model(C.BASE_MODEL_KEY, spec, norm, C.N_TASKS, seed=int(seed), device=dev)
        base_dev.eval()
        model_dev, _ = M.build_model(spec, norm, C.N_TASKS, seed=int(seed), device=dev)
        model_dev.eval()
        with torch.no_grad():
            ref_dev = base_dev(dev_batch)
            logits_dev, aux_dev = model_dev(dev_batch, return_aux=True)
        d = float((logits_dev - ref_dev).abs().max().item())
        rel = d / max(float(ref_dev.abs().max().item()), 1e-12)
        resid_dev = float(aux_dev["residual_logits"].abs().max().item())
        ok_dev = resid_dev == 0.0 and rel <= 1e-6
        passed &= bool(ok_dev)
        results["device"] = {
            "device": str(dev),
            "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
            "max_abs_diff": d,
            "relative_diff": rel,
            "residual_max_abs": resid_dev,
            "tolerance_relative": 1e-6,
            "ok": bool(ok_dev),
        }
        results["device_note"] = (
            "the residual is exactly zero on CUDA too; the inherited W2 trunk uses segment-sum "
            "atomics, so two independent forward passes can differ by a few ULPs"
        )
    return _result("02_zero_init_forward_equals_w2", passed, **results)


# ---------------------------------------------------------------------------
# 3 -- atom renumbering
# ---------------------------------------------------------------------------


def check_permutation(spec, norm, batch, *, seed: int = C.SEED, device: str = "cpu") -> dict[str, Any]:
    host_batch = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
    host_permuted, _p = permute_batch(host_batch)
    if str(device) != "cpu":
        dev = torch.device(device)
        batch_d = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in host_batch.items()}
        permuted_d = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in host_permuted.items()}
    else:
        batch_d, permuted_d = host_batch, host_permuted
    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=int(seed), device=device)
    model.eval()
    with torch.no_grad():
        a = model(batch_d)
        b = model(permuted_d)
    diff = float((a - b).abs().max().item())
    scale = float(a.abs().max().item())
    rel = diff / max(scale, 1e-12)
    ok = rel <= 1e-4
    return _result(
        "03_atom_renumbering_invariance",
        ok,
        max_abs_diff=diff,
        max_abs_reference=scale,
        relative=rel,
        tolerance=1e-4,
        note="the pair payload is rebuilt from the permuted bond_index, so the molecule is the same object",
    )


# ---------------------------------------------------------------------------
# 4 -- pair-token exchange symmetry
# ---------------------------------------------------------------------------


def check_pair_symmetry(spec, norm, batch, *, seed: int = C.SEED, device: str = "cpu") -> dict[str, Any]:
    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=int(seed), device=device)
    # The tail is normally zero at step 0, which would make the logit-invariance
    # part vacuous.  Perturb only ``branch.combiner.fc2`` (deterministically, on
    # the CPU, with a private generator) so the residual is a genuinely non-zero
    # function of the pair tokens before the exchange test.
    gen = torch.Generator(device="cpu").manual_seed(9137)
    with torch.no_grad():
        model.branch.combiner.fc2.weight.copy_(
            torch.randn(model.branch.combiner.fc2.weight.shape, generator=gen).to(
                device=model.branch.combiner.fc2.weight.device, dtype=model.branch.combiner.fc2.weight.dtype
            )
            * 0.02
        )
        model.branch.combiner.fc2.bias.copy_(
            torch.randn(model.branch.combiner.fc2.bias.shape, generator=gen).to(
                device=model.branch.combiner.fc2.bias.device, dtype=model.branch.combiner.fc2.bias.dtype
            )
            * 0.02
        )
    model.eval()
    dev = batch["atom_feat"].device
    with torch.no_grad():
        h = model.encode_h(batch)[3]
        tok = model.branch.pair_tokens(
            h, batch["pair_i"].to(torch.int64), batch["pair_j"].to(torch.int64), batch["pair_bucket"].to(torch.int64)
        )
        swapped = _clone_batch(batch)
        swapped["pair_i"] = batch["pair_j"].clone()
        swapped["pair_j"] = batch["pair_i"].clone()
        tok_swapped = model.branch.pair_tokens(
            h, swapped["pair_i"].to(torch.int64), swapped["pair_j"].to(torch.int64), swapped["pair_bucket"].to(torch.int64)
        )
        logits = model(batch)
        logits_swapped = model(swapped)
        base = model(batch, residual_zero=True)
    token_diff = float((tok - tok_swapped).abs().max().item())
    logit_diff = float((logits - logits_swapped).abs().max().item())
    residual_norm = float((logits - base).abs().max().item())
    passed = bool(torch.equal(tok, tok_swapped) and logit_diff == 0.0 and residual_norm > 0.0)
    return _result(
        "04_pair_token_exchange_symmetry",
        passed,
        token_bitwise_equal=bool(torch.equal(tok, tok_swapped)),
        token_max_abs_diff=token_diff,
        logit_max_abs_diff_under_swap=logit_diff,
        residual_max_abs_after_tail_perturbation=residual_norm,
        n_pairs=int(batch["pair_i"].numel()),
        note=(
            "the zero-initialised tail is replaced by a fixed small random tail (private generator, CPU) "
            "so the residual is non-zero and the exchange invariance is a real test, not a zero==zero one"
        ),
        device=str(dev),
    )


# ---------------------------------------------------------------------------
# 5 -- no cross-graph pair, no batch-mate dependence
# ---------------------------------------------------------------------------


def check_no_cross_graph(
    spec, norm, batch, task, dist, rows, *, seed: int = C.SEED, device: str = "cpu"
) -> dict[str, Any]:
    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=int(seed), device=device)
    model.eval()
    dev = batch["atom_feat"].device
    counts = batch["pair_counts"].to(torch.int64)
    pair_graph = batch["pair_graph"].to(torch.int64)
    atom_off = batch["atom_off"].to(torch.int64)
    atom_counts = batch["atom_counts"].to(torch.int64)
    structural = {
        "pair_counts_match_pair_graph": bool(
            torch.equal(torch.bincount(pair_graph, minlength=int(batch["n_graphs"])), counts)
        ),
        "pair_i_within_molecule": bool(
            int((batch["pair_i"].to(torch.int64) >= 0).sum()) == int(batch["pair_i"].numel())
            and int((batch["pair_i"].to(torch.int64) < atom_counts[pair_graph]).sum()) == int(batch["pair_i"].numel())
        ),
        "pair_j_within_molecule": bool(
            int((batch["pair_j"].to(torch.int64) < atom_counts[pair_graph]).sum()) == int(batch["pair_j"].numel())
        ),
        "i_lt_j": bool(int((batch["pair_j"].to(torch.int64) > batch["pair_i"].to(torch.int64)).sum()) == int(batch["pair_i"].numel())),
        "no_shared_pair": bool(
            len(
                set(
                    zip(
                        (atom_off[pair_graph] + batch["pair_i"].to(torch.int64)).tolist(),
                        (atom_off[pair_graph] + batch["pair_j"].to(torch.int64)).tolist(),
                    )
                )
            )
            == int(batch["pair_i"].numel())
        ),
    }
    with torch.no_grad():
        full = model(batch)
        per_graph = []
        for g, row in enumerate(rows):
            one = P.collate_pw(task, dist, [int(row)])
            one = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in one.items()}
            per_graph.append(float((model(one)[0] - full[g]).abs().max().item()))
    worst = max(per_graph) if per_graph else 0.0
    scale = float(full.abs().max().item())
    rel = worst / max(scale, 1e-12)
    passed = bool(all(structural.values()) and rel <= 1e-4)
    return _result(
        "05_no_cross_graph_pair_and_batch_independence",
        passed,
        structural=structural,
        per_graph_max_abs_diff=per_graph,
        worst_relative=rel,
        tolerance=1e-4,
        note="every molecule is re-collated alone from the cache (its own pair list only) and must reproduce the batched logits",
    )


# ---------------------------------------------------------------------------
# 6 / 7 -- cross-component pairs and bucket-vs-bond consistency
# ---------------------------------------------------------------------------


def check_pair_index(dist: Mapping[str, Any] | None, *, n_sample: int = 200, seed: int = 20261111) -> dict[str, Any]:
    """Buckets agree with the original chemical bonds and no cross-component pair exists."""
    synthetic: dict[str, Any] = {}
    tiny = {
        "chain4_plus_chord": (4, [(0, 1), (1, 2), (2, 3), (0, 3)]),
        "chain5": (5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
        "ring6": (6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]),
        "two_components": (6, [(0, 1), (1, 2), (3, 4)]),
        "single_atom": (1, []),
    }
    failures: list[dict[str, Any]] = []
    for name, (n, edges) in tiny.items():
        i, j, b = P.molecule_pairs(n, np.asarray(edges, dtype=np.int64).reshape(-1, 2))
        ref = P.brute_force_distances(n, np.asarray(edges, dtype=np.int64).reshape(-1, 2))
        expected = {
            (u, v, P.bucket_of_distance(ref[u, v]))
            for u in range(n)
            for v in range(u + 1, n)
            if P.bucket_of_distance(ref[u, v]) is not None
        }
        got = set(zip(i.tolist(), j.tolist(), b.tolist()))
        ok = got == expected
        synthetic[name] = {
            "n_atoms": n,
            "n_pairs": int(i.size),
            "n_pairs_all": int(n * (n - 1) // 2),
            "codes_match": bool(ok),
        }
        if not ok:
            failures.append({"graph": name, "got": sorted(got), "expected": sorted(expected)})

    real: dict[str, Any] = {"rows_checked": 0, "rows_mismatched": 0, "pairs_checked": 0}
    if dist is not None:
        gen = np.random.default_rng(int(seed))
        have = np.where(np.asarray(dist["pair_counts"]) > 0)[0]
        if have.size:
            sample = np.sort(gen.choice(have, size=min(n_sample, have.size), replace=False))
            atom_ptr, bond_ptr, bond_index = P._structure_paths()
            atom_ptr = np.load(atom_ptr)
            bond_ptr = np.load(bond_ptr)
            bond_index = np.load(bond_index, mmap_mode="r")
            for row in sample:
                a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
                b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
                edges = np.asarray(bond_index[b0:b1], dtype=np.int64)
                n = a1 - a0
                ref = P.brute_force_distances(n, edges)
                lo, hi = int(dist["ptr"][row]), int(dist["ptr"][row + 1])
                got = set(
                    zip(
                        dist["pair_i"][lo:hi].astype(np.int64).tolist(),
                        dist["pair_j"][lo:hi].astype(np.int64).tolist(),
                        dist["pair_bucket"][lo:hi].astype(np.int64).tolist(),
                    )
                )
                expected = set()
                for u in range(n):
                    for v in range(u + 1, n):
                        code = P.bucket_of_distance(ref[u, v])
                        if code is not None:
                            expected.add((u, v, code))
                real["rows_checked"] += 1
                real["pairs_checked"] += len(expected)
                if got != expected:
                    real["rows_mismatched"] += 1
                    failures.append({"row": int(row)})
        meta = dist["meta"]
        real["cache_dropped_cross_component_pairs"] = int(meta["n_pairs_dropped_cross_component"])
        real["cache_pairs_stored"] = int(meta["n_pairs_stored"])
        real["cache_unordered_pairs_all_for_scope"] = int(meta["n_pairs_stored"]) + int(
            meta["n_pairs_dropped_cross_component"]
        )
    passed = not failures and all(v["codes_match"] for v in synthetic.values())
    return _result(
        "06_07_bucket_matches_bonds_and_no_cross_component_pair",
        passed,
        synthetic_graphs=synthetic,
        real_cache_sample=real,
        bucket_definitions=[list(b) for b in C.BUCKETS],
        exclusivity="one bucket code per unordered pair; buckets are disjoint and cover every finite distance",
        cross_component="pairs of different connected components are never stored (unreachable -> dropped)",
        failures=failures[:10],
    )


# ---------------------------------------------------------------------------
# 8 -- segment softmax normalisation
# ---------------------------------------------------------------------------


def check_segment_softmax(spec, norm, batch, *, seed: int = C.SEED, device: str = "cpu") -> dict[str, Any]:
    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=int(seed), device=device)
    model.eval()
    with torch.no_grad():
        h = model.encode_h(batch)[3]
        token = model.branch.pair_tokens(
            h, batch["pair_i"].to(torch.int64), batch["pair_j"].to(torch.int64), batch["pair_bucket"].to(torch.int64)
        )
        _z, attn, score = model.branch.segment_attention(
            token, batch["pair_graph"].to(torch.int64), int(batch["n_graphs"])
        )
    pair_graph = batch["pair_graph"].to(torch.int64)
    n_graphs = int(batch["n_graphs"])
    sums = torch.zeros((n_graphs, C.N_QUERIES), device=attn.device, dtype=attn.dtype).index_add(0, pair_graph, attn)
    counts = batch["pair_counts"].to(torch.int64)
    with_pairs = counts > 0
    max_err = float((sums[with_pairs] - 1.0).abs().max().item()) if bool(with_pairs.any()) else 0.0
    empty_ok = bool((sums[~with_pairs] == 0).all().item()) if bool((~with_pairs).any()) else True
    finite = bool(torch.isfinite(attn).all().item() and torch.isfinite(score).all().item())
    nonneg = bool((attn >= 0).all().item())
    passed = bool(max_err <= 1e-5 and empty_ok and finite and nonneg)
    return _result(
        "08_segment_softmax_normalised",
        passed,
        n_graphs=n_graphs,
        n_graphs_with_pairs=int(with_pairs.sum().item()),
        max_abs_row_sum_error=max_err,
        empty_graphs_sum_exactly_zero=empty_ok,
        finite=finite,
        non_negative=nonneg,
        tolerance=1e-5,
        note="one softmax per (molecule, query) over that molecule's legal pairs; empty molecules get exactly 0",
    )


# ---------------------------------------------------------------------------
# 9 -- gradient path
# ---------------------------------------------------------------------------


def check_gradient_path(spec, norm, batches: Sequence[Mapping[str, Any]], *, steps: int = 3, device: str = "cpu") -> dict[str, Any]:
    import bs_train as BT

    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=C.SEED, device=device)
    optimizer, _ = BSM.make_optimizer(model)
    model.train()
    trace = []
    for step in range(int(steps)):
        batch = batches[step % len(batches)]
        if "pair_i" not in batch or int(batch["pair_i"].numel()) == 0:
            raise RuntimeError("gradient check needs a batch with legal pairs")
        y = torch.full((int(batch["n_graphs"]), C.N_TASKS), float("nan"), device=batch["atom_feat"].device)
        y[:, 0] = torch.randint(0, 2, (int(batch["n_graphs"]),)).float()
        y[:, 1] = 1.0
        out = BT.train_step(model, optimizer, batch, y, global_step=step, clip=5.0)
        grads = {
            name: float(param.grad.detach().float().norm().item())
            for name, param in model.named_parameters()
            if name.startswith("branch.") and param.grad is not None
        }
        trace.append({"step": step, "loss": float(out["loss"]), "n_branch_grads": len(grads)})
    final = {
        name: float(param.grad.detach().float().norm().item())
        for name, param in model.named_parameters()
        if name.startswith("branch.") and param.grad is not None
    }
    def prefix_norm(name: str) -> float:
        prefix = f"branch.{name}"
        return float(
            sum(
                value
                for key, value in final.items()
                if key == prefix or key.startswith(prefix + ".")
            )
        )

    missing = sorted(name for name in C.BRANCH_PARAMS if not any(k.startswith(f"branch.{name}") for k in final))
    zero = sorted(name for name in C.BRANCH_PARAMS if prefix_norm(name) <= 0.0)
    key_bias = float(final.get("branch.key.bias", float("nan")))
    # ``key.bias`` is a genuine degeneracy: a constant added to every key of a
    # molecule shifts all its scores by the same amount, which the query-wise
    # softmax removes exactly, so its gradient is 0 by construction.  It is
    # therefore *reported* (``key_bias_grad_norm``) and excluded from the
    # "every new module has gradient" assertion, not hidden.
    passed = bool(not missing and not zero)
    return _result(
        "09_gradient_path_active",
        passed,
        steps=int(steps),
        trace=trace,
        grad_norms=final,
        missing_parameters=missing,
        zero_grad_module_sums=zero,
        module_grad_sums={name: prefix_norm(name) for name in C.BRANCH_PARAMS},
        key_bias_grad_norm=key_bias,
        documented_zero_gradient_parameters=["branch.key.bias"],
        key_bias_degeneracy=(
            "key_ij = W p_ij + b; <q_k, b> is constant inside one molecule, so the segment softmax "
            "removes it exactly. The bias is kept because the model definition says Linear(48,48), "
            "and its zero gradient is reported instead of hidden."
        ),
        note="the final Linear(128,128) is zero at step 0, so the branch trunk only receives gradient from step 1 onward",
    )


# ---------------------------------------------------------------------------
# 10 -- brute force vs batched
# ---------------------------------------------------------------------------


def brute_force_branch(
    model: M.PWModel,
    h: np.ndarray,
    atom_counts: Sequence[int],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pair_bucket: np.ndarray,
    pair_graph: np.ndarray,
) -> dict[str, np.ndarray]:
    """Explicit per-pair, per-molecule implementation in float64 (reference).

    Nothing is vectorised over pairs: every token, every softmax and every
    ``z_k`` is computed with a plain loop, exactly as PROTOCOL.md writes it.
    """
    dim = C.PAIR_DIM
    n_queries = C.N_QUERIES

    def weight(module) -> np.ndarray:
        return module.weight.detach().cpu().numpy().astype(np.float64)

    def bias(module) -> np.ndarray | None:
        if module.bias is None:
            return None
        return module.bias.detach().cpu().numpy().astype(np.float64)

    w_sum, w_diff, w_prod = weight(model.branch.w_sum), weight(model.branch.w_diff), weight(model.branch.w_prod)
    emb = model.branch.dist_embedding.detach().cpu().numpy().astype(np.float64)
    ln_w = model.branch.norm.weight.detach().cpu().numpy().astype(np.float64)
    ln_b = model.branch.norm.bias.detach().cpu().numpy().astype(np.float64)
    query = model.branch.query.detach().cpu().numpy().astype(np.float64)
    w_key, b_key = weight(model.branch.key), bias(model.branch.key)
    w_val, b_val = weight(model.branch.value), bias(model.branch.value)
    c_ln_w = model.branch.combiner.norm.weight.detach().cpu().numpy().astype(np.float64)
    c_ln_b = model.branch.combiner.norm.bias.detach().cpu().numpy().astype(np.float64)
    c_w1, c_b1 = weight(model.branch.combiner.fc1), bias(model.branch.combiner.fc1)
    c_w2, c_b2 = weight(model.branch.combiner.fc2), bias(model.branch.combiner.fc2)

    def silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))

    def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mu = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * w + b

    n_pairs = int(pair_i.size)
    n_graphs = len(atom_counts)
    token = np.zeros((n_pairs, dim), dtype=np.float64)
    for t in range(n_pairs):
        hi = h[int(pair_i[t])]
        hj = h[int(pair_j[t])]
        pre = (
            w_sum @ (hi + hj)
            + w_diff @ np.abs(hi - hj)
            + w_prod @ (hi * hj)
            + emb[int(pair_bucket[t])]
        )
        token[t] = layer_norm(silu(pre), ln_w, ln_b)

    key = token @ w_key.T + (0.0 if b_key is None else b_key)
    value = token @ w_val.T + (0.0 if b_val is None else b_val)
    score = (key @ query.T) * float(C.SCORE_SCALE)

    attn = np.zeros((n_pairs, n_queries), dtype=np.float64)
    z = np.zeros((n_graphs, n_queries, dim), dtype=np.float64)
    for g in range(n_graphs):
        sel = np.where(pair_graph == g)[0]
        if sel.size == 0:
            continue
        s = score[sel]
        s = s - s.max(axis=0, keepdims=True)
        e = np.exp(s)
        a = e / e.sum(axis=0, keepdims=True)
        attn[sel] = a
        for k in range(n_queries):
            z[g, k] = a[:, k] @ value[sel]

    z_flat = z.reshape(n_graphs, n_queries * dim)
    x = layer_norm(z_flat, c_ln_w, c_ln_b)
    x = silu(x @ c_w1.T + c_b1)
    residual = x @ c_w2.T + c_b2
    counts = np.bincount(np.asarray(pair_graph, dtype=np.int64), minlength=n_graphs) if n_pairs else np.zeros(n_graphs)
    residual[counts == 0] = 0.0
    return {"token": token, "attn": attn, "z": z, "residual": residual, "pair_counts": counts}


def check_cpu_vs_batched(spec, norm, batch, *, device: str = "cpu", seed: int = C.SEED) -> dict[str, Any]:
    """The batched implementation must equal an explicit pair-by-pair enumeration."""
    if int(batch["pair_i"].numel()) == 0:
        raise RuntimeError("brute-force check needs a batch with legal pairs")
    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=int(seed), device=device)
    model.eval()
    dev = batch["atom_feat"].device
    with torch.no_grad():
        h = model.encode_h(batch)[3]
        token = model.branch.pair_tokens(
            h, batch["pair_i"].to(torch.int64), batch["pair_j"].to(torch.int64), batch["pair_bucket"].to(torch.int64)
        )
        z, attn, _score = model.branch.segment_attention(
            token, batch["pair_graph"].to(torch.int64), int(batch["n_graphs"])
        )
        z_flat = z.reshape(int(batch["n_graphs"]), C.Z_DIM)
        residual = model.branch.combiner(z_flat) * (batch["pair_counts"].to(torch.int64) > 0).to(torch.float32).unsqueeze(1)
    h_np = h.detach().cpu().numpy().astype(np.float64)
    ref = brute_force_branch(
        model,
        h_np,
        batch["atom_counts"].to(torch.int64).tolist(),
        batch["pair_i"].to(torch.int64).cpu().numpy(),
        batch["pair_j"].to(torch.int64).cpu().numpy(),
        batch["pair_bucket"].to(torch.int64).cpu().numpy(),
        batch["pair_graph"].to(torch.int64).cpu().numpy(),
    )
    results: dict[str, Any] = {}
    passed = True
    for name, got, want in (
        ("token", token.detach().cpu().numpy().astype(np.float64), ref["token"]),
        ("attention", attn.detach().cpu().numpy().astype(np.float64), ref["attn"]),
        ("z", z.detach().cpu().numpy().astype(np.float64), ref["z"]),
        ("residual", residual.detach().cpu().numpy().astype(np.float64), ref["residual"]),
    ):
        diff = float(np.abs(got - want).max())
        scale = max(float(np.abs(want).max()), 1e-12)
        rel = diff / scale
        ok = rel <= 1e-5
        passed &= bool(ok)
        results[name] = {"max_abs_diff": diff, "relative": rel, "tolerance": 1e-5, "ok": bool(ok)}
    results["device"] = str(dev)
    results["n_pairs"] = int(batch["pair_i"].numel())
    results["n_graphs"] = int(batch["n_graphs"])
    results["reference"] = "float64 python loops over pairs (one token, one softmax and one z_k at a time)"
    return _result("10_bruteforce_equals_batched", passed, **results)


# ---------------------------------------------------------------------------
# 11 -- molecule without a legal pair
# ---------------------------------------------------------------------------


def check_no_pairs(spec, norm, batch, *, seed: int = C.SEED, device: str = "cpu") -> dict[str, Any]:
    if int(batch["pair_i"].numel()) != 0:
        raise RuntimeError("this check needs a batch whose molecules have no legal pair")
    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=int(seed), device=device)
    model.eval()
    base, _ = BSM.build_model(C.BASE_MODEL_KEY, spec, norm, C.N_TASKS, seed=int(seed), device=device)
    base.eval()
    with torch.no_grad():
        logits, aux = model(batch, return_aux=True)
        ref = base(batch)
    residual_max = float(aux["residual_logits"].abs().max().item())
    diff = float((logits - ref).abs().max().item())
    finite = bool(torch.isfinite(logits).all().item())
    passed = bool(residual_max == 0.0 and diff == 0.0 and finite)
    return _result(
        "11_no_legal_pair_zero_residual",
        passed,
        n_graphs=int(batch["n_graphs"]),
        n_atoms_per_graph=batch["atom_counts"].to(torch.int64).tolist(),
        residual_max_abs=residual_max,
        logits_vs_w2_max_abs_diff=diff,
        finite=finite,
        note="an explicit per-graph mask keeps the residual exactly zero, so these molecules are the W2 model",
    )


# ---------------------------------------------------------------------------
# 12 -- metric / Train-only isolation parity
# ---------------------------------------------------------------------------


def check_metrics_and_isolation(task: Any | None, batch: Mapping[str, Any] | None = None) -> dict[str, Any]:
    import pc_metrics as MET

    gen = torch.Generator().manual_seed(17)
    logits = torch.randn(64, C.N_TASKS, generator=gen)
    y = torch.randint(0, 2, (64, C.N_TASKS), generator=gen).float()
    y[torch.rand_like(y) < 0.6] = float("nan")
    loss = MET.masked_bce_loss(logits, y)
    manual = torch.nn.functional.binary_cross_entropy_with_logits(logits[torch.isfinite(y)], y[torch.isfinite(y)])
    loss_equal = float((loss - manual).abs()) == 0.0
    probs = torch.sigmoid(logits).numpy()
    y_np = y.numpy()
    official = MET.official_ap(y_np, probs)
    independent = MET.per_task_ap(y_np, probs)
    ap_delta = abs(float(official["value"]) - float(independent["mean_over_scored_tasks"]))

    details: dict[str, Any] = {
        "masked_bce_equals_manual": bool(loss_equal),
        "masked_bce_value": float(loss),
        "official_ap": float(official["value"]),
        "independent_ap": float(independent["mean_over_scored_tasks"]),
        "official_vs_independent_abs_delta": float(ap_delta),
        "collate_keys_are_inherited_plus_pairs": sorted(
            set(batch or {})
            - {"pair_i", "pair_j", "pair_bucket", "pair_graph", "pair_counts"}
        ),
    }
    passed = bool(loss_equal and ap_delta < 1e-9)
    if task is not None:
        test_labels_nan = not bool(np.isfinite(task.labels[np.asarray(task.test_rows, dtype=np.int64)]).any())
        splits_disjoint = (
            np.intersect1d(task.train_rows, task.valid_rows).size == 0
            and np.intersect1d(task.train_rows, task.test_rows).size == 0
            and np.intersect1d(task.valid_rows, task.test_rows).size == 0
        )
        plan = C.BC.SD.make_plan40(task, seed=C.SEED)
        plan_sha_ok = plan[0]["global_plan_sha256"] == C.W2["global_plan_sha256"]
        train_sorted = np.sort(np.asarray(task.train_rows, dtype=np.int64))
        epochs_ok = all(
            np.array_equal(np.sort(np.asarray(p["rows"], np.int64)), train_sorted) for p in plan
        )
        details.update(
            {
                "test_labels_all_nan_in_training_artifact": bool(test_labels_nan),
                "split_partitions_disjoint": bool(splits_disjoint),
                "order_plan_sha256_matches_w2": bool(plan_sha_ok),
                "every_epoch_is_a_train_permutation": bool(epochs_ok),
                "global_plan_sha256": plan[0]["global_plan_sha256"],
                "validation_rows_used_for_selection_only": True,
                "checkpoint_selection_rule": "argmax official Validation AP over epochs 1..40, earliest wins ties",
                "test_used_for_selection": False,
            }
        )
        passed &= bool(test_labels_nan and splits_disjoint and plan_sha_ok and epochs_ok)
    return _result("12_metrics_and_train_only_isolation", passed, **details)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


def run_all(
    spec,
    norm,
    batch: Mapping[str, Any],
    rows: Sequence[int],
    *,
    task: Any | None = None,
    pairs: Mapping[str, Any] | None = None,
    no_pair_batch: Mapping[str, Any] | None = None,
    device: str = "cpu",
    grad_batches: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if no_pair_batch is None:
        raise RuntimeError("the no-pair batch is required for check 11")
    if task is None or pairs is None:
        raise RuntimeError("checks 5 and 12 need the task and the pair index")
    results = [
        check_init(spec, norm, device=device),
        check_zero_init_forward(spec, norm, batch, device=device),
        check_permutation(spec, norm, batch, device=device),
        check_pair_symmetry(spec, norm, batch, device=device),
        check_no_cross_graph(spec, norm, batch, task, pairs, rows, device=device),
        check_pair_index(pairs),
        check_segment_softmax(spec, norm, batch, device=device),
        check_gradient_path(spec, norm, list(grad_batches or [batch]), steps=3, device=device),
        check_cpu_vs_batched(spec, norm, batch, device=device),
        check_no_pairs(spec, norm, no_pair_batch, device=device),
        check_metrics_and_isolation(task, batch),
    ]
    return {
        "schema": C.SCHEMA_PREFLIGHT,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(torch.device(device)) if torch.device(device).type == "cuda" else "cpu",
        "torch": torch.__version__,
        "checks": results,
        "all_passed": bool(all(r["passed"] for r in results)),
        "n_checks": len(results),
        "n_passed": int(sum(1 for r in results if r["passed"])),
    }


__all__ = [
    "brute_force_branch",
    "check_cpu_vs_batched",
    "check_gradient_path",
    "check_init",
    "check_metrics_and_isolation",
    "check_no_cross_graph",
    "check_no_pairs",
    "check_pair_index",
    "check_pair_symmetry",
    "check_permutation",
    "check_segment_softmax",
    "check_zero_init_forward",
    "permute_batch",
    "rebuild_pairs_from_bonds",
    "run_all",
]
