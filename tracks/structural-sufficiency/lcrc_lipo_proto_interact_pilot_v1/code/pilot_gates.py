"""CPU correctness gates for the pilot round.

Twelve gates, all label-scope and implementation checks.  They must pass before
any GPU run is launched.  Each gate is falsifiable and records the numbers it
compared, so a future reader can re-run ``run_pilot.py gates`` and diff the
result.  Nothing here changes a scientific choice: a failure is only ever fixed
by fixing the implementation.
"""

from __future__ import annotations

import math
import platform
from typing import Any, Callable

import numpy as np
import torch

import pilot_common as C
import pilot_models as M
import pilot_pairs as P


C_PAIR_SEMANTICS = {"PAIR_TRUE": "true", "PAIR_NULL": "null"}


class GateFailure(AssertionError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(message)


def gate_fold_identity() -> dict[str, Any]:
    """1. Fold-2 Train/Dev rows equal the frozen v2 fold; no overlap."""
    view = C.load_fold_view()
    folds = C.folds_payload()
    fold_entry = [f for f in folds["folds"] if int(f["fold"]) == C.FOLD][0]
    dev_positions = [int(x) for x in fold_entry["rows_cache_positions"]]
    dev_rows = [int(x) for x in fold_entry["rows_global"]]
    _require(list(view.dev_positions) == dev_positions, "Dev positions differ from the frozen fold file")
    _require(
        [int(view.dev_cache["molecule_ids"][i]) for i in range(len(dev_positions))] == dev_rows,
        "Dev molecule ids differ from the frozen fold rows",
    )
    expected_train = [p for p in range(2352) if p not in set(dev_positions)]
    _require(list(view.train_positions) == expected_train, "Train positions are not the fold complement")
    _require(not (set(view.train_positions) & set(view.dev_positions)), "Train/Dev overlap")
    _require(len(view.y_train) == C.N_TRAIN, "Train label vector length is not 1568")
    _require(len(view.y_dev) == C.N_DEV, "Dev label vector length is not 784")
    _require(int(view.train_cache["n_molecules"]) == C.N_TRAIN, "Train cache size is not 1568")
    _require(int(view.dev_cache["n_molecules"]) == C.N_DEV, "Dev cache size is not 784")
    if len({int(view.dev_cache["molecule_ids"][i]) for i in range(len(dev_positions))}) != C.N_DEV:
        raise GateFailure("Dev cache contains duplicate molecules")
    return {
        "fold": C.FOLD,
        "n_train": len(view.train_positions),
        "n_dev": len(view.dev_positions),
        "dev_positions_sha256": view.audit()["dev_positions_sha256"],
        "train_positions_sha256": view.audit()["train_positions_sha256"],
        "frozen_fold_row_sha256_int64": fold_entry["row_sha256_int64"],
        "no_overlap": True,
        "label_vector_sha256": view.target_norm["label_vector_sha256"],
    }


def gate_target_norm_train_only() -> dict[str, Any]:
    """2. Target standardisation uses only the Train fold labels."""
    view = C.load_fold_view()
    norm = view.target_norm
    _require(norm["source"] == "fold_train_labels_only", "target norm source is not the fold train labels")
    _require(int(norm["n"]) == C.N_TRAIN, "target norm was not fitted on 1568 molecules")
    _require(norm["monitor_or_reserved_used"] is False, "target norm used held-out labels")
    recomputed_mean = float(np.mean(view.y_train))
    recomputed_scale = float(np.std(view.y_train))
    _require(abs(recomputed_mean - float(norm["mean"])) < 1e-12, "target mean mismatch")
    _require(abs(recomputed_scale - float(norm["scale"])) < 1e-12, "target scale mismatch")
    dev_mean = float(np.mean(view.y_dev))
    # A Dev-derived normalisation would give a different mean; assert it differs.
    _require(abs(dev_mean - recomputed_mean) > 1e-6, "Dev labels are indistinguishable from Train labels")
    return {
        "mean": float(norm["mean"]),
        "scale": float(norm["scale"]),
        "scale_definition": norm["scale_definition"],
        "n": int(norm["n"]),
        "dev_mean_differs": True,
        "feature_norm_source": view.train_norm["source"],
        "feature_norm_fit_split": view.train_norm["fit_split"],
        "feature_norm_valid_or_test_used": bool(view.train_norm["valid_or_test_used"]),
    }


def gate_base_init_matches_lcrc() -> dict[str, Any]:
    """3. BASE matches the frozen LCRC initialisation for the same seed, tensor by tensor."""
    spec = C.V2C.load_frozen_spec()
    view = C.load_fold_view()
    model = M.build_base(spec, view.train_norm, C.SEED, "cpu")
    torch.manual_seed(C.SEED)
    reference = C.prep_model.LCRC(spec, view.train_norm, C.prep_model.DEFAULT_CONFIG)
    a_state, b_state = reference.state_dict(), model.state_dict()
    _require(sorted(a_state) == sorted(b_state), "BASE state-dict keys differ from LCRC")
    mismatched = [k for k in a_state if not torch.equal(a_state[k], b_state[k])]
    _require(not mismatched, f"BASE init differs from LCRC on {mismatched}")
    v2_model = C.V2M.build_a(spec, view.train_norm, C.SEED, "cpu")
    v2_mismatched = [k for k in a_state if not torch.equal(a_state[k], v2_model.state_dict()[k])]
    _require(not v2_mismatched, f"BASE init differs from the v2 build_a on {v2_mismatched}")
    return {
        "n_tensors": len(a_state),
        "parameters": M.parameter_count(model),
        "state_hash": C.state_dict_fingerprint(a_state),
        "matches_lcrc": True,
        "matches_v2_build_a": True,
        "seed": C.SEED,
    }


def gate_proto_shared_init() -> dict[str, Any]:
    """4. PROTO shares every non-``f`` tensor with BASE; only ``f`` differs."""
    spec = C.V2C.load_frozen_spec()
    view = C.load_fold_view()
    model, base = M.build_proto(spec, view.train_norm, C.SEED, "cpu")
    audit = M.shared_init_audit(base, model)
    _require(audit["all_shared_bitwise_identical"], f"PROTO shared init mismatch: {audit['mismatched']}")
    model_only = set(audit["model_only_keys"])
    expected = {"f.prototypes", "f.experts", "f.bias"}
    _require(model_only <= expected, f"PROTO added unexpected tensors: {sorted(model_only - expected)}")
    _require(expected <= model_only, f"PROTO is missing prototype tensors: {sorted(expected - model_only)}")
    base_only = set(audit["base_only_keys"])
    _require(base_only == {"f.lins.0.weight", "f.lins.0.bias", "f.lins.1.weight", "f.lins.1.bias"},
             f"unexpected BASE-only tensors {sorted(base_only)}")
    _require(isinstance(model.f, M.ProtoReadout), "PROTO head is not ProtoReadout")
    _require(not any(n.startswith(("pair_",)) for n in model.state_dict()), "PROTO must not carry pair modules")
    return {
        **audit,
        "base_parameters": M.parameter_count(base),
        "proto_parameters": M.parameter_count(model),
        "prototype_parameters": int(model.f.prototypes.numel() + model.f.experts.numel() + model.f.bias.numel()),
        "only_f_replaced": True,
    }


def gate_pair_identity() -> dict[str, Any]:
    """5. PAIR_TRUE / PAIR_NULL: same structure, same parameter count, same init."""
    spec = C.V2C.load_frozen_spec()
    view = C.load_fold_view()
    true_model = M.build_pair(spec, view.train_norm, C.SEED, "cpu")
    null_model = M.build_pair(spec, view.train_norm, C.SEED, "cpu")
    equality = M.pair_state_equality(true_model, null_model)
    _require(equality["same_keys"], f"PAIR arm key mismatch: {equality['mismatched']}")
    _require(equality["identical"], f"PAIR init mismatch on {equality['mismatched']}")
    _require(
        M.parameter_count(true_model) == M.parameter_count(null_model),
        "PAIR parameter counts differ",
    )
    head = true_model.pair_head[2]
    _require(bool(torch.all(head.bias == 0)), "pair head bias is not zero-initialised")
    return {
        "parameter_count": M.parameter_count(true_model),
        "pair_head_parameters": int(sum(p.numel() for p in true_model.pair_head.parameters())),
        "pair_z_parameters": int(sum(p.numel() for p in true_model.pair_z.parameters())),
        "identical_initialisation": True,
        "last_layer_weight_std": float(head.weight.detach().std()),
        "last_layer_bias_all_zero": True,
        "n_tensors": equality["n_tensors"],
    }


def gate_pair_legality() -> dict[str, Any]:
    """6. PAIR_TRUE pairs are unique ``u < v`` with true distance 3/4/5."""
    view = C.load_fold_view()
    report: dict[str, Any] = {"splits": {}}
    for name, cache in (("train", view.train_cache), ("dev", view.dev_cache)):
        pairs = P.build_pair_cache(cache, semantics="true")
        atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
        bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
        bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
        ptr = np.asarray(pairs["pair_ptr"], dtype=np.int64)
        u, v, b = pairs["pair_u"], pairs["pair_v"], pairs["pair_bin"]
        # Independent recomputation of the distances for a deterministic subset.
        rng = np.random.default_rng(0)
        sample = sorted(int(x) for x in rng.choice(int(cache["n_molecules"]), size=40, replace=False))
        checked = 0
        for m in sample:
            a0, a1 = int(atom_ptr[m]), int(atom_ptr[m + 1])
            dist = P.bfs_distances(a1 - a0, bond_index[int(bond_ptr[m]) : int(bond_ptr[m + 1])])
            sl = slice(int(ptr[m]), int(ptr[m + 1]))
            for pu, pv, pb in zip(u[sl].tolist(), v[sl].tolist(), b[sl].tolist()):
                _require(pu < pv, f"molecule {m}: pair is not u < v")
                _require(
                    int(dist[pu, pv]) == C.PAIR_DISTANCES[pb],
                    f"molecule {m}: pair ({pu},{pv}) distance {int(dist[pu, pv])} != bin {pb}",
                )
                checked += 1
        # No pair may exist at distance 3/4/5 that we missed.
        for m in sample:
            a0, a1 = int(atom_ptr[m]), int(atom_ptr[m + 1])
            dist = P.bfs_distances(a1 - a0, bond_index[int(bond_ptr[m]) : int(bond_ptr[m + 1])])
            sl = slice(int(ptr[m]), int(ptr[m + 1]))
            found = set(zip(u[sl].tolist(), v[sl].tolist()))
            for i in range(a1 - a0):
                for j in range(i + 1, a1 - a0):
                    d = int(dist[i, j])
                    if 3 <= d <= 5:
                        _require((i, j) in found, f"molecule {m}: missing pair ({i},{j}) at distance {d}")
        report["splits"][name] = {
            "n_pairs": int(u.size),
            "n_molecules_checked": len(sample),
            "pairs_verified": int(checked),
            "coverage": P.coverage(pairs),
        }
    return report


def gate_pair_null_content() -> dict[str, Any]:
    """7. PAIR_NULL keeps the (u, v) set; only per-molecule bin labels permute."""
    view = C.load_fold_view()
    report: dict[str, Any] = {"splits": {}}
    for name, cache in (("train", view.train_cache), ("dev", view.dev_cache)):
        true_pairs = P.build_pair_cache(cache, semantics="true")
        null_pairs = P.build_pair_cache(cache, semantics="null")
        content = P.same_pair_content(true_pairs, null_pairs)
        _require(content["pair_ptr_identical"], "null pair pointer changed")
        _require(content["pair_u_identical"], "null pair u indices changed")
        _require(content["pair_v_identical"], "null pair v indices changed")
        _require(content["bin_is_per_molecule_permutation"], "null bins are not a per-molecule permutation")
        _require(content["n_pairs_identical"], "null pair count changed")
        _require(
            content["per_molecule_bin_multiset_preserved"],
            "null bins are not a per-molecule permutation of the true counts",
        )
        # Determinism: rebuilding gives the same table.
        again = P.build_pair_cache(cache, semantics="null")
        _require(
            np.array_equal(np.asarray(null_pairs["pair_bin"]), np.asarray(again["pair_bin"])),
            "null bin assignment is not deterministic",
        )
        # Per-molecule, not global.
        per_mol = np.asarray(null_pairs["null_permutations"], dtype=np.int64)
        _require(len({tuple(r) for r in per_mol.tolist()}) > 1, "null used a single global permutation")
        report["splits"][name] = content
    report["salt"] = C.PAIR_NULL_SALT
    return report


def gate_permutation_invariance() -> dict[str, Any]:
    """8. Atom relabelling leaves BASE/PROTO/PAIR_TRUE/PAIR_NULL predictions unchanged."""
    view = C.load_fold_view()
    spec = C.V2C.load_frozen_spec()
    torch.manual_seed(1234)
    base = M.build_base(spec, view.train_norm, C.SEED, "cpu").eval()
    proto, _ = M.build_proto(spec, view.train_norm, C.SEED, "cpu")
    proto = proto.eval()
    pair = M.build_pair(spec, view.train_norm, C.SEED, "cpu").eval()

    rows = list(range(0, 6))
    small = C.V2D.select_cache(view.train_cache, rows, split_label="gate_perm_small")
    renumbered = P.permute_cache_atoms(small, seed=7)
    # Atom features must move with the relabelled index, not with the position.
    true_pairs = P.build_pair_cache(small, semantics="true")
    true_pairs_perm = P.build_pair_cache(renumbered, semantics="true")
    null_pairs = P.build_pair_cache(small, semantics="null")
    null_pairs_perm = P.build_pair_cache(renumbered, semantics="null")

    out: dict[str, Any] = {}
    with torch.no_grad():
        for name, model, use_pairs in (
            ("BASE_G0", base, False),
            (C.PROTO_NAME, proto, False),
            ("PAIR_TRUE", pair, True),
        ):
            ref = model(P.make_batch(small, rows, true_pairs if use_pairs else None)).numpy()
            perm = model(P.make_batch(renumbered, rows, true_pairs_perm if use_pairs else None)).numpy()
            diff = float(np.abs(ref - perm).max())
            _require(diff < 1e-4, f"{name}: prediction changed under atom relabelling ({diff})")
            out[name] = {"max_abs_diff": diff}

        null_ref = pair(P.make_batch(small, rows, null_pairs)).numpy()
        null_perm = pair(P.make_batch(renumbered, rows, null_pairs_perm)).numpy()
        diff = float(np.abs(null_ref - null_perm).max())
        _require(diff < 1e-4, f"PAIR_NULL: prediction changed under atom relabelling ({diff})")
        out["PAIR_NULL"] = {"max_abs_diff": diff}
        # The molecule-level null permutation must not depend on atom order.
        _require(
            np.array_equal(
                np.asarray(null_pairs["null_permutations"]), np.asarray(null_pairs_perm["null_permutations"])
            ),
            "null permutation table depends on atom indexing",
        )
        # The pair *list order* depends on atom numbering, so compare the
        # per-molecule multiset of assigned bins instead of the flat sequence.
        def _per_molecule_bins(pair_cache, molecule_cache):
            ptr = np.asarray(pair_cache["pair_ptr"], dtype=np.int64)
            bins = np.asarray(pair_cache["pair_bin"], dtype=np.int64)
            out = []
            for m in range(int(ptr.size) - 1):
                block = bins[int(ptr[m]) : int(ptr[m + 1])]
                out.append(sorted(np.bincount(block, minlength=3).tolist()))
            return out

        _require(
            _per_molecule_bins(null_pairs, small) == _per_molecule_bins(null_pairs_perm, renumbered),
            "null bin assignment changed under atom relabelling",
        )
    out["null_permutation_table_index_independent"] = True
    out["n_molecules"] = len(rows)
    out["verdict"] = "all arms are permutation invariant within 1e-4"
    return out


def gate_gradients() -> dict[str, Any]:
    """9. Prototype and pair parameters receive finite, non-zero gradients."""
    spec = C.V2C.load_frozen_spec()
    view = C.load_fold_view()
    out: dict[str, Any] = {}

    proto, _ = M.build_proto(spec, view.train_norm, C.SEED, "cpu")
    optimizer = torch.optim.AdamW(proto.parameters(), lr=C.PEAK_LR, weight_decay=C.WEIGHT_DECAY)
    rows = list(range(0, C.BATCH_SIZE))
    for _ in range(2):
        batch = P.make_batch(view.train_cache, rows)
        target = torch.as_tensor(
            (view.y_train[rows] - float(view.target_norm["mean"])) / float(view.target_norm["scale"]),
            dtype=torch.float32,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(proto(batch).view(-1), target.view(-1))
        loss.backward()
        for name in ("f.prototypes", "f.experts", "f.bias"):
            grad = dict(proto.named_parameters())[name].grad
            _require(grad is not None and bool(torch.isfinite(grad).all()), f"{name}: non-finite/missing grad")
            _require(bool(grad.abs().sum() > 0), f"{name}: zero gradient")
        optimizer.step()
        out.setdefault("proto", {})[f"loss_step_{_ + 1}"] = float(loss)
    out["proto"]["parameters_updated"] = True

    pair = M.build_pair(spec, view.train_norm, C.SEED, "cpu")
    pairs = P.build_pair_cache(view.train_cache, semantics="true")
    optimizer = torch.optim.AdamW(pair.parameters(), lr=C.PEAK_LR, weight_decay=C.WEIGHT_DECAY)
    for step in range(2):
        batch = P.make_batch(view.train_cache, rows, pairs)
        target = torch.as_tensor(
            (view.y_train[rows] - float(view.target_norm["mean"])) / float(view.target_norm["scale"]),
            dtype=torch.float32,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(pair(batch).view(-1), target.view(-1))
        loss.backward()
        grads = dict(pair.named_parameters())
        for name in ("pair_z.weight", "pair_z.bias", "pair_head.0.weight", "pair_head.0.bias",
                     "pair_head.2.weight", "pair_head.2.bias"):
            grad = grads[name].grad
            _require(grad is not None and bool(torch.isfinite(grad).all()), f"{name}: non-finite/missing grad")
            _require(bool(grad.abs().sum() > 0), f"{name}: zero gradient")
        optimizer.step()
        out.setdefault("pair", {})[f"loss_step_{step + 1}"] = float(loss)
    out["pair"]["parameters_updated"] = True
    out["n_rows"] = len(rows)
    return out


def gate_zero_residual() -> dict[str, Any]:
    """10. Zeroing the PAIR residual reproduces the same-weight BASE prediction."""
    view = C.load_fold_view()
    spec = C.V2C.load_frozen_spec()
    pair = M.build_pair(spec, view.train_norm, C.SEED, "cpu")
    # Perturb the pair head so the check is not vacuous.
    with torch.no_grad():
        pair.pair_head[2].weight.normal_(0.0, 1e-2)
        pair.pair_head[2].bias.normal_(0.0, 1e-2)
    pair.eval()
    pairs = P.build_pair_cache(view.dev_cache, semantics="true")
    from pilot_train import zero_residual_check

    result = zero_residual_check(pair, view.dev_cache, pairs, device=torch.device("cpu"))
    _require(result["max_abs_diff_zeroed_vs_base"] < 1e-5, "zero residual does not reduce to BASE")
    _require(result["pair_residual_max_abs"] > 0, "pair residual is identically zero after perturbation")
    return result


def gate_smoke_steps() -> dict[str, Any]:
    """11. Two mini-batch forward/backward/optimizer steps per arm, finite loss."""
    spec = C.V2C.load_frozen_spec()
    view = C.load_fold_view()
    rows = list(range(0, C.BATCH_SIZE))
    out: dict[str, Any] = {}
    for arm in C.ARMS:
        semantics = C_PAIR_SEMANTICS.get(arm)
        pairs = P.build_pair_cache(view.train_cache, semantics=semantics) if semantics else None
        model, _meta = M.build_arm(arm, spec, view.train_norm, C.SEED, "cpu", dropout=C.DROPOUT)
        optimizer = torch.optim.AdamW(model.parameters(), lr=C.PEAK_LR, weight_decay=C.WEIGHT_DECAY)
        losses = []
        model.train()
        for step in range(2):
            batch = P.make_batch(view.train_cache, rows, pairs)
            target = torch.as_tensor(
                (view.y_train[rows] - float(view.target_norm["mean"])) / float(view.target_norm["scale"]),
                dtype=torch.float32,
            )
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = torch.nn.functional.mse_loss(pred.view(-1), target.view(-1))
            _require(math.isfinite(float(loss)), f"{arm}: non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            optimizer.step()
            losses.append(float(loss))
        _require(float(loss) < 50.0, f"{arm}: smoke loss did not stay bounded ({loss})")
        out[arm] = {
            "losses": losses,
            "parameters": M.parameter_count(model),
            "finite": True,
            "dropout_p": C.DROPOUT,
        }
    return out


def gate_batch_permutation_and_single() -> dict[str, Any]:
    """12. Batch order, batch composition and single-molecule batches are consistent."""
    spec = C.V2C.load_frozen_spec()
    view = C.load_fold_view()
    torch.manual_seed(99)
    rows = [3, 11, 42, 137, 250]
    out: dict[str, Any] = {}
    for arm in C.ARMS:
        semantics = C_PAIR_SEMANTICS.get(arm)
        pairs = P.build_pair_cache(view.dev_cache, semantics=semantics) if semantics else None
        model, _meta = M.build_arm(arm, spec, view.train_norm, C.SEED, "cpu", dropout=0.0)
        model.eval()
        with torch.no_grad():
            joint = model(P.make_batch(view.dev_cache, rows, pairs)).numpy()
            reversed_order = model(P.make_batch(view.dev_cache, list(reversed(rows)), pairs)).numpy()[::-1]
            singles = np.asarray(
                [float(model(P.make_batch(view.dev_cache, [r], pairs)).numpy().reshape(-1)[0]) for r in rows]
            )
        d_order = float(np.abs(joint - reversed_order).max())
        d_single = float(np.abs(joint - singles).max())
        _require(d_order < 1e-4, f"{arm}: batch order changes predictions ({d_order})")
        _require(d_single < 1e-4, f"{arm}: single-molecule batch differs ({d_single})")
        out[arm] = {"max_abs_diff_reordered": d_order, "max_abs_diff_single": d_single}
    return out


GATES: tuple[tuple[str, str, Callable[[], dict[str, Any]]], ...] = (
    ("fold_identity", "Fold 2 Train/Dev rows equal the frozen v2 fold, no overlap", gate_fold_identity),
    ("target_norm_train_only", "Target standardisation reads Train labels only", gate_target_norm_train_only),
    ("base_init_matches_lcrc", "BASE matches the frozen LCRC initialisation tensor by tensor", gate_base_init_matches_lcrc),
    ("proto_shared_init", "PROTO shares all non-f tensors with BASE", gate_proto_shared_init),
    ("pair_identity", "PAIR_TRUE and PAIR_NULL have identical structure/init", gate_pair_identity),
    ("pair_legality", "PAIR_TRUE pairs are unique u<v at graph distance 3/4/5", gate_pair_legality),
    ("pair_null_content", "PAIR_NULL keeps (u,v) and permutes only per-molecule bins", gate_pair_null_content),
    ("permutation_invariance", "Atom relabelling leaves predictions unchanged", gate_permutation_invariance),
    ("gradients", "PROTO/PAIR parameters receive finite non-zero gradients", gate_gradients),
    ("zero_residual", "Zeroed PAIR residual reproduces the BASE prediction", gate_zero_residual),
    ("smoke_steps", "2 mini-batch forward/backward/optimizer steps per arm", gate_smoke_steps),
    ("batch_invariance", "Batch order / single-molecule batches agree", gate_batch_permutation_and_single),
)


def run_gates(*, write: bool = True) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    ok = True
    for name, description, fn in GATES:
        try:
            detail = fn()
            results.append({"name": name, "description": description, "ok": True, "detail": detail})
            C.log(f"[gate] PASS {name}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            results.append(
                {
                    "name": name,
                    "description": description,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            C.log(f"[gate] FAIL {name}: {type(exc).__name__}: {exc}")
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "config_hash": C.config_hash(C.load_config()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cpu_only": True,
        "n_gates": len(results),
        "n_passed": sum(1 for r in results if r["ok"]),
        "ok": ok,
        "gates": results,
    }
    if write:
        C.write_json(C.CPU_GATES, payload)
    return payload


__all__ = ["GATES", "run_gates", "GateFailure"]
