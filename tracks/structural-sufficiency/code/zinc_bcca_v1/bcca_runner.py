"""ZINC bond-conditioned context attention v1 — runner.

Protocol: ``results/zinc_bond_conditioned_context_attention_v1/PROTOCOL.md``
(``zinc-bond-conditioned-context-attention-v1``).

Stages
------
``audit``          read-only hash / init / encoder / batch-order / P3-valid replay
                   audit of the frozen full-10k artifacts (no valid labels, no
                   official test of any kind).
``gates``          CPU correctness gates: init identity, toy algebra, symmetry,
                   cross-molecule isolation, multi-step gradients, shared
                   C/G attention path, frozen-loop parity, label isolation.
``flops``          GEMM FLOPs + analytic pair-op counts on one fixed 128-graph batch.
``smoke``          2-epoch train-only smoke and throughput projection (never scored).
``train``          one arm/seed: frozen R3, 360 epochs, all 10,000 rows.
``predict-valid``  freeze official-valid predictions (labels not read).
``run-arm``        both seeds of one arm, sequentially.
``mechanism``      label-free attention diagnostics + C->G and uniform forward.
``score``          the single official-valid label read and frozen contrast table.

There is no official-test stage in this round.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import math
import platform
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for _p in (HERE, REPO / "tracks/structure-contribution/code", REPO / "tracks/structural-sufficiency/code/phase2b"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bcca_common as C  # noqa: E402
import bcca_models as BM  # noqa: E402
import phase2b_common as P2B  # noqa: E402
import phase2a_runner as R2A  # noqa: E402

import bjoc_perfmech_long_v1 as bp  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stored(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover - non-git launch
        return "unknown"


def _device(name: str):
    import torch

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False")
    return device


def _env(device: Any) -> dict[str, Any]:
    import torch

    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "device": str(device),
        "cuda_device_name": cuda_name,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": torch.version.cuda,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "git_commit": _git_commit(),
        "utc": _now(),
    }


@contextlib.contextmanager
def _no_labels():
    """Make the official valid/test label loaders unreachable at this stage."""
    saved = (P2B.load_valid_y, P2B.load_test_y, P2B._official_y)

    def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("label access is blocked at this stage")

    P2B.load_valid_y, P2B.load_test_y, P2B._official_y = boom, boom, boom
    try:
        yield
    finally:
        P2B.load_valid_y, P2B.load_test_y, P2B._official_y = saved


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def cmd_audit(args: argparse.Namespace) -> int:
    out = C.OUT / "p3_reuse_audit.json"
    if out.exists():
        raise SystemExit("audit record exists; refusing to rewrite")
    cfg = C.load_config()
    checks: dict[str, Any] = {"protocol_id": C.PROTOCOL_ID, "stage": "audit", "created_utc": _now()}

    # --- frozen input hashes ------------------------------------------------
    inputs = {}
    for name, spec in cfg["frozen_inputs"].items():
        path = REPO / spec["path"]
        if "sha256" in spec:
            inputs[name] = C.verify_file(name, path, spec["sha256"])
        else:
            if not path.exists():
                raise SystemExit(f"missing frozen artifact {name}: {path}")
            inputs[name] = {"name": name, "path": spec["path"], "sha256": C.sha256_file(path)}
    checks["frozen_inputs"] = inputs

    # --- frozen model/object/SSL source hashes ------------------------------
    code_files = {
        "structural_relation_composer_v1": REPO / "tracks/structure-contribution/code/structural_relation_composer_v1.py",
        "structural_relation_objects_v1": REPO / "tracks/structure-contribution/code/structural_relation_objects_v1.py",
        "bjoc_masked_pretraining_v1": REPO / "tracks/structure-contribution/code/bjoc_masked_pretraining_v1.py",
        "phase2a_models": REPO / "tracks/structural-sufficiency/code/phase2a/phase2a_models.py",
    }
    code_now = {k: C.sha256_file(p) for k, p in code_files.items()}
    code_expected = dict(cfg["audit_expectations"]["code_hashes"])
    checks["code_hashes"] = {
        "current": code_now,
        "expected_when_p3_was_evaluated": code_expected,
        "all_match": all(code_now[k] == code_expected[k] for k in code_now),
        "bjoc_perfmech_long_v1_current": C.sha256_file(
            REPO / "tracks/structure-contribution/code/bjoc_perfmech_long_v1.py"
        ),
    }
    if not checks["code_hashes"]["all_match"]:
        raise SystemExit(f"frozen model/object source changed: {code_now}")
    here = Path(__file__).resolve().parent
    checks["round_code"] = {
        "bcca_models.py": C.sha256_file(here / "bcca_models.py"),
        "bcca_common.py": C.sha256_file(here / "bcca_common.py"),
        "bcca_runner.py": C.sha256_file(here / "bcca_runner.py"),
        "config": C.sha256_file(C.CONFIG_PATH),
        "protocol": C.sha256_file(C.PROTOCOL_PATH),
    }

    # --- data ---------------------------------------------------------------
    train_objs = C.load_objects("objects_train10k")
    valid_objs = C.load_objects("objects_valid")
    if [int(o.mol_index) for o in train_objs] != list(range(C.N_TRAIN)):
        raise SystemExit("objects_train10k row order is not 0..9999")
    if [int(o.mol_index) for o in valid_objs] != list(range(C.N_VALID)):
        raise SystemExit("objects_valid row order is not 0..999")
    y = C.load_train_y()
    checks["data"] = {
        "n_train_objects": len(train_objs),
        "n_valid_objects": len(valid_objs),
        "train_y_sha256_float64": C.sha256_array(y, "float64"),
        "standardiser": {"fit_only": True, "n_fit": C.N_TRAIN},
        "label_free": True,
    }

    # --- per-seed init / encoder / batch order ------------------------------
    per_seed: dict[str, Any] = {}
    for seed in C.SEEDS:
        entry: dict[str, Any] = {}
        for arm in BM.VARIANTS:
            model, info = C.build_model(arm, seed, "cpu")
            entry[arm] = {
                "parameters": info["parameters"],
                "checks": info["checks"],
                "shared_state_sha256": info["build"]["shared_state_sha256"],
                "extra_init": info["build"]["extra_init"],
                "module_audit_pass": info["module_audit"]["pass"],
            }
            del model
        # C/G full-state identity and shared K/T identity
        import torch

        _, c_model, c_info = BM.build_variant("C", seed=seed, norm=C.load_norm(), ssl_state=C.ssl_state(seed)[0])
        _, g_model, g_info = BM.build_variant("G", seed=seed, norm=C.load_norm(), ssl_state=C.ssl_state(seed)[0])
        _, f_model, f_info = BM.build_variant("F", seed=seed, norm=C.load_norm(), ssl_state=C.ssl_state(seed)[0])
        cs, gs = c_model.state_dict(), g_model.state_dict()
        state_equal = set(cs) == set(gs) and all(torch.equal(cs[k], gs[k]) for k in cs)
        entry["C_G_state_bitwise_equal"] = bool(state_equal)
        entry["K_C_equals_K_G"] = bool(torch.equal(c_model.attn.k.weight, g_model.attn.k.weight))
        entry["T_F_equals_K_C"] = bool(torch.equal(f_model.ffn.t.weight, c_model.attn.k.weight))
        entry["Q_C_zero"] = float(c_model.attn.q.weight.abs().sum())
        entry["Q_G_zero"] = float(g_model.attn.q.weight.abs().sum())
        entry["U_F_zero"] = float(f_model.ffn.u.weight.abs().sum())
        entry["C_G_shared_equal"] = bool(
            c_info["shared_state_sha256"] == g_info["shared_state_sha256"]
        )
        # batch order over the full 360-epoch R3 schedule
        gen = torch.Generator().manual_seed(int(seed) + C.TRAIN_SHUFFLE_OFFSET)
        h = hashlib.sha256()
        for _ in range(C.EPOCHS):
            perm = torch.randperm(C.N_TRAIN, generator=gen).tolist()
            for i in range(0, C.N_TRAIN, C.BATCH_SIZE):
                h.update(np.ascontiguousarray(np.asarray(perm[i : i + C.BATCH_SIZE], dtype=np.int64)).tobytes())
        entry["batch_order_sha256"] = h.hexdigest()
        entry["batch_order_matches_r3"] = h.hexdigest() == cfg["audit_expectations"]["batch_order_sha256"][str(seed)]
        per_seed[str(seed)] = entry
    checks["per_seed"] = per_seed

    # --- historical P3 soups: hash + label-free valid replay ----------------
    device = _device(args.device)
    recorded_mae = {str(s): float(cfg["audit_expectations"]["historical_valid_mae"][str(s)]) for s in C.SEEDS}
    p3: dict[str, Any] = {}
    for seed in C.SEEDS:
        model, info = C.build_p3(seed, device)
        params = int(sum(p.numel() for p in model.parameters()))
        pred, secs = C.predict(model, valid_objs, device)
        if not np.isfinite(pred).all():
            raise SystemExit("non-finite P3 replay prediction")
        hist, hist_rec = C.p3_valid_pred(seed)
        entry = {
            "soup": info["soup"],
            "parameters": params,
            "parameter_exact_413505": params == 413505,
            "module_audit_pass": info["module_audit"]["pass"],
            "init_checks": info["checks"],
            "recorded_historical_valid_mae": recorded_mae[str(seed)],
            "replay_seconds": secs,
            "labels_read": False,
            "official_test_loaded": False,
        }
        if hist is not None:
            max_abs = float(np.abs(pred - hist).max())
            entry["branch"] = "frozen_array_present"
            entry["frozen_array_sha256"] = hist_rec["array_sha256_float32"]
            entry["frozen_file_sha256"] = hist_rec["historical_file_sha256"]
            entry["replay_max_abs_diff_vs_frozen"] = max_abs
            entry["replay_within_5e-5"] = max_abs <= 5e-5
            if max_abs > 5e-5:
                raise SystemExit(f"P3 seed {seed} valid replay differs from the frozen npz: {max_abs:.3e}")
        else:
            path = C.OUT / f"p3_valid_pred_replay_s{seed}.npz"
            if path.exists():
                raise SystemExit(f"replay array already exists but no audit record: {path}")
            np.savez_compressed(path, row=np.arange(C.N_VALID, dtype=np.int64),
                                pred=pred.astype(np.float32))
            arr = np.asarray(pred, dtype=np.float32)
            entry["branch"] = "regenerated_label_free_from_hash_verified_soup"
            entry["replay_file"] = _stored(path)
            entry["replay_file_sha256"] = C.sha256_file(path)
            entry["replay_array_sha256"] = C.sha256_array(arr)
            entry["replay_within_5e-5"] = True
            entry["note"] = (
                "historical npz absent on this checkout; the frozen soup reproduces it on CPU "
                "(Phase 2b: max|d| 1.9e-6/2.9e-6); MAE is verified against the recorded "
                "historical value at the score stage, after the single label read"
            )
        p3[str(seed)] = entry
        log(f"[audit] P3 s{seed}: params={params} branch={entry['branch']}")

    # --- execution regime vs the P3/Phase 2b regime -------------------------
    env = _env(device)
    regime = cfg["audit_expectations"]["p3_execution_regime"]
    import torch

    regime_checks = {
        "torch": env["torch"] == regime["torch"],
        "python": env["python"] == regime["python"],
        "torch_cuda": env["torch_cuda"] == regime["torch_cuda"],
        "gpu_name": env["cuda_device_name"] == regime["gpu_name"],
    }
    checks["regime"] = {
        "observed": env,
        "frozen_p3_regime": regime,
        "checks": regime_checks,
        "regime_matches": all(regime_checks.values()),
        "p3_baseline_required": not all(regime_checks.values()),
        "note": (
            "a regime mismatch would require two same-round P3 baselines trained before the "
            "single official-valid label read (PROTOCOL §6 provenance branch)"
        ),
    }

    conditions = {
        "frozen_inputs_verified": True,
        "frozen_source_matches_p3_revision": checks["code_hashes"]["all_match"],
        "all_p3_soups_match": all(
            p3[str(s)]["soup"]["sha256"] == cfg["audit_expectations"]["p3_soup_sha256"][str(s)] for s in C.SEEDS
        ),
        "all_init_checks_pass": all(
            all(per_seed[str(s)][arm]["checks"].values()) for s in C.SEEDS for arm in BM.VARIANTS
        ),
        "all_batch_orders_match": all(per_seed[str(s)]["batch_order_matches_r3"] for s in C.SEEDS),
        "all_parameter_counts_exact": all(
            per_seed[str(s)][arm]["parameters"] == C.EXPECTED_PARAMETERS[arm]
            for s in C.SEEDS for arm in BM.VARIANTS
        ),
        "all_module_audits_pass": all(
            per_seed[str(s)][arm]["module_audit_pass"] for s in C.SEEDS for arm in BM.VARIANTS
        ),
        "C_G_init_bitwise_equal": all(per_seed[str(s)]["C_G_state_bitwise_equal"] for s in C.SEEDS),
        "K_shared_C_G_F": all(
            per_seed[str(s)]["K_C_equals_K_G"] and per_seed[str(s)]["T_F_equals_K_C"] for s in C.SEEDS
        ),
        "zero_init_as_designed": all(
            per_seed[str(s)]["Q_C_zero"] == 0.0
            and per_seed[str(s)]["Q_G_zero"] == 0.0
            and per_seed[str(s)]["U_F_zero"] == 0.0
            for s in C.SEEDS
        ),
        "p3_valid_replay_within_5e-5": all(p3[str(s)]["replay_within_5e-5"] for s in C.SEEDS),
        "official_test_not_loaded": True,
    }
    checks["p3"] = p3
    checks["conditions"] = conditions
    checks["all_pass"] = all(conditions.values())
    checks["official_valid_labels_read"] = False
    checks["official_test_loaded"] = False
    checks["environment"] = env
    checks["git_commit"] = _git_commit()
    C.write_once_json(out, checks)
    log(f"[audit] all_pass={checks['all_pass']} regime_matches={checks['regime']['regime_matches']} -> {out}")
    return 0 if checks["all_pass"] else 1


# ---------------------------------------------------------------------------
# CPU gates
# ---------------------------------------------------------------------------


def _train_objs(n: int) -> list[Any]:
    objs = C.load_objects("objects_train10k")
    return list(objs[:n])


def _batch(objs: Sequence[Any]) -> dict[str, Any]:
    return sro.batch_objects(list(objs))


def _toy_system(n_bonds: int = 4, n_junc: int = 4, seed: int = 7):
    """Toy incidence system with non-uniform attention (hand-built, no chemistry)."""
    import torch

    rng = np.random.default_rng(seed)
    b = torch.as_tensor(rng.normal(size=(n_bonds, BM.DIM)), dtype=torch.float64)
    w = torch.as_tensor(rng.normal(size=(n_junc, BM.DIM)), dtype=torch.float64)
    junc_bond = torch.as_tensor([[0, 2], [0, 3], [1, 2], [1, 3]], dtype=torch.long)
    batch = {
        "n_graphs": 1,
        "bond_batch": torch.zeros(n_bonds, dtype=torch.long),
        "junc_batch": torch.zeros(n_junc, dtype=torch.long),
        "bond_feat": torch.zeros((n_bonds, sro.BOND_FEAT_DIM), dtype=torch.float64),
        "bond_index": torch.zeros((n_bonds, 2), dtype=torch.long),
        "junc_feat": torch.zeros((n_junc, sro.JUNC_FEAT_DIM), dtype=torch.float64),
        "junc_index": torch.zeros((n_junc, 3), dtype=torch.long),
        "junc_bond": junc_bond,
        "atom_feat": torch.zeros((1, sro.ATOM_FEAT_DIM), dtype=torch.float64),
        "atom_batch": torch.zeros(1, dtype=torch.long),
        "global_feat": torch.zeros((1, sro.GLOBAL_DIM), dtype=torch.float64),
    }
    n_b = BM.bond_junction_counts(batch).to(torch.float64)
    return b, w, batch, n_b


def _toy_attention(seed: int = 0):
    """A float64 attention module with deliberately non-uniform, non-symmetric Q/K."""
    import torch

    attn = BM.BondJunctionAttention().double()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        attn.q.weight.copy_(torch.randn(attn.q.weight.shape, dtype=torch.float64, generator=g))
        attn.k.weight.copy_(torch.randn(attn.k.weight.shape, dtype=torch.float64, generator=g))
    return attn


def _nonlinear_readout(W: Any, V: Any) -> float:
    import torch

    return float((torch.linalg.vector_norm(W, dim=1) ** 4).sum() + (torch.linalg.vector_norm(V, dim=1) ** 2).sum())


def cmd_gates(args: argparse.Namespace) -> int:
    out = C.OUT / "cpu_gates_this_round.json"
    if out.exists():
        raise SystemExit("gate record exists; refusing to rewrite")
    import torch
    import torch.nn.functional as F

    gates: dict[str, Any] = {"protocol_id": C.PROTOCOL_ID, "stage": "gates", "created_utc": _now()}
    failures: list[str] = []

    def gate(name: str, passed: bool, detail: Any) -> None:
        gates[name] = {"pass": bool(passed), "detail": detail}
        if not passed:
            failures.append(name)

    cfg = C.load_config()
    norm = C.load_norm()
    ssl0, ssl_rec0 = C.ssl_state(0)
    objs = _train_objs(32)
    batch = _batch(objs)

    # --- G1: zero-init function identity with P3 -----------------------------
    models: dict[str, Any] = {}
    infos: dict[str, Any] = {}
    for arm in ("P3", "C", "G", "F"):
        _, model, info = BM.build_variant(arm, seed=0, norm=norm, ssl_state=ssl0)
        model.eval()
        models[arm], infos[arm] = model, info
    with torch.no_grad():
        feats = {arm: models[arm].forward_features(batch, return_attention=(arm in ("C", "G")))
                 for arm in ("P3", "C", "G", "F")}
        preds = {arm: models[arm](batch) for arm in ("P3", "C", "G", "F")}
    wv = {}
    for arm in ("C", "G", "F"):
        w_diff = float((feats[arm]["W"] - feats["P3"]["W"]).abs().max())
        v_diff = float((feats[arm]["V"] - feats["P3"]["V"]).abs().max())
        w_scale = float(feats["P3"]["W"].abs().max())
        v_scale = float(feats["P3"]["V"].abs().max())
        wv[arm] = {
            "W_max_abs_diff": w_diff,
            "V_max_abs_diff": v_diff,
            "W_max_rel_diff": w_diff / max(w_scale, 1e-12),
            "V_max_rel_diff": v_diff / max(v_scale, 1e-12),
            "W_abs_scale": w_scale,
            "V_abs_scale": v_scale,
            "n_b_max_abs_diff": float((feats[arm]["n_b"] - feats["P3"]["n_b"]).abs().max()),
        }
    pred_diff = {arm: float((preds[arm] - preds["P3"]).abs().max()) for arm in ("C", "G", "F")}
    gate(
        "G1_zero_init_matches_p3",
        max(pred_diff.values()) <= 5e-5
        and max(v["n_b_max_abs_diff"] for v in wv.values()) == 0.0
        and max(v["W_max_rel_diff"] for v in wv.values()) <= 1e-5
        and max(v["V_max_rel_diff"] for v in wv.values()) <= 1e-5,
        {
            "W_V": wv,
            "prediction_max_abs_diff": pred_diff,
            "tolerances": {"prediction_abs": 5e-5, "W_V_relative": 1e-5, "n_b_abs": 0.0},
            "error_source": "float32 softmax normalisation 1/J vs division by the integer count",
        },
    )

    # --- G2: init state identity / parameter counts / provenance -------------
    import bjoc_masked_pretraining_v1 as mp

    _, c0, _ = BM.build_variant("C", seed=0, norm=norm, ssl_state=ssl0)
    _, g0, _ = BM.build_variant("G", seed=0, norm=norm, ssl_state=ssl0)
    _, f0, _ = BM.build_variant("F", seed=0, norm=norm, ssl_state=ssl0)
    cs, gs = c0.state_dict(), g0.state_dict()
    state_equal = set(cs) == set(gs) and all(torch.equal(cs[k], gs[k]) for k in cs)
    shared_equal = {
        arm: BM.shared_state_sha256(m) == BM.shared_state_sha256(models["P3"])
        for arm, m in (("C", c0), ("G", g0), ("F", f0))
    }
    params = {arm: BM.parameter_audit(m) for arm, m in models.items()}
    g2 = {
        "C_G_state_bitwise_equal": bool(state_equal),
        "shared_matches_P3": shared_equal,
        "K_C_equals_K_G": bool(torch.equal(c0.attn.k.weight, g0.attn.k.weight)),
        "T_F_equals_K_C": bool(torch.equal(f0.ffn.t.weight, c0.attn.k.weight)),
        "Q_C_zero": float(c0.attn.q.weight.abs().sum()),
        "Q_G_zero": float(g0.attn.q.weight.abs().sum()),
        "U_F_zero": float(f0.ffn.u.weight.abs().sum()),
        "parameters": {arm: params[arm]["actual_parameters"] for arm in params},
        "expected": C.EXPECTED_PARAMETERS,
        "reader_in_dim": {arm: params[arm]["reader_in_dim"] for arm in params},
        "reader_layers": {arm: len(models[arm].reader.lins) for arm in models},
        "normaliser_fit_only": bool(norm.get("fit_only")),
        "encoder_files": {"s0": ssl_rec0["sha256"]},
    }
    gate(
        "G2_init_identity_parameters_provenance",
        state_equal
        and all(shared_equal.values())
        and g2["K_C_equals_K_G"]
        and g2["T_F_equals_K_C"]
        and g2["Q_C_zero"] == 0.0
        and g2["Q_G_zero"] == 0.0
        and g2["U_F_zero"] == 0.0
        and all(g2["parameters"][a] == C.EXPECTED_PARAMETERS[a] for a in g2["parameters"])
        and g2["reader_in_dim"]["P3"] == 321
        and g2["reader_in_dim"]["C"] == 321
        and all(v == 3 for v in g2["reader_layers"].values()),
        g2,
    )

    # --- G3: toy algebra (non-uniform A) ------------------------------------
    attn = _toy_attention(seed=3)
    b, w, toy_batch, n_b = _toy_system()
    with torch.no_grad():
        Wc, Vc, aux_c = attn(b, w, toy_batch, n_b, mode="C", return_attention=True)
        Wg, Vg, aux_g = attn(b, w, toy_batch, n_b, mode="G", return_attention=True)
    A, p = aux_c["A"], aux_g["p"]
    n_bonds = int(n_b.numel())
    assert torch.all(n_b == n_b[0])  # the toy system is complete: every bond has 2 junctions
    cons = BM.moment_conservation(Wc, Vc, Wg, Vg, toy_batch)
    safe = n_b.clamp_min(1.0).unsqueeze(1)
    slack_W = float(((Wc / safe) ** 2 <= (Vc / safe) + 1e-10).all())
    per_dim_margin = float(((Vc / safe) - (Wc / safe) ** 2).min())
    disp = {
        "C": BM.content_dispersion(Wc, Vc, n_b, toy_batch),
        "G": BM.content_dispersion(Wg, Vg, n_b, toy_batch),
    }
    # A is per-bond row; build the row-mean of p for comparison
    row = aux_c["row"]
    p_row = aux_g["p"][aux_c["col"]]
    alloc_diff = torch.zeros(n_bonds, dtype=torch.float64)
    alloc_diff.index_add_(0, row, (A - p_row).abs())
    alloc_diff = float(alloc_diff.max())
    # nonlinear witness: total moments are conserved, the per-bond allocation is not
    lin_C = float(Wc.sum(0).sum())
    lin_G = float(Wg.sum(0).sum())
    nl_C = _nonlinear_readout(Wc, Vc)
    nl_G = _nonlinear_readout(Wg, Vg)
    g3 = {
        "moment_conservation": cons,
        "uniform_softmax_row_sum_error": float((A.reshape(n_bonds, -1).sum(1) - 1).abs().max()),
        "V_over_n_ge_W_over_n_squared": bool(slack_W),
        "min_per_dimension_jensen_margin": per_dim_margin,
        "per_bond_allocation_max_L1_difference_C_vs_G": alloc_diff,
        "content_dispersion": disp,
        "linear_readout_C": lin_C,
        "linear_readout_G": lin_G,
        "linear_readout_abs_diff": abs(lin_C - lin_G),
        "nonlinear_readout_C": nl_C,
        "nonlinear_readout_G": nl_G,
        "nonlinear_separation_relative": abs(nl_C - nl_G) / max(abs(nl_C), abs(nl_G), 1e-12),
        "note": "toy algebra only; no chemistry, no trained model",
    }
    gate(
        "G3_toy_algebra_moment_allocation",
        cons["W_max_relative_error"] <= 1e-10
        and cons["V_max_relative_error"] <= 1e-10
        and g3["uniform_softmax_row_sum_error"] <= 1e-12
        and slack_W
        and alloc_diff > 0.1
        and disp["C"]["W_content_dispersion"] > 0.0
        and disp["G"]["W_content_dispersion"] <= 1e-12
        and g3["linear_readout_abs_diff"] <= 1e-9
        and g3["nonlinear_separation_relative"] > 1e-6,
        g3,
    )

    # --- G4: symmetry, ordering, cross-molecule isolation, empty junction ---
    mols = [o for _, o in R2A._synthetic_molecules()]
    rng = np.random.default_rng(20261019)
    relabel_max = 0.0
    swap_max = 0.0
    bond_perm_max = 0.0
    junc_perm_max = 0.0
    with torch.no_grad():
        for obj in mols:
            perm = rng.permutation(obj.n_atoms)
            relabelled = R2A._reindex_objects(obj, perm)
            b0, b1 = _batch([obj]), _batch([relabelled])
            for arm in ("C", "G", "F"):
                relabel_max = max(relabel_max, abs(float(models[arm](b0)[0]) - float(models[arm](b1)[0])))
            pairs = [(int(u), int(v), int(t)) for (u, v), t in zip(obj.bond_index, obj.bond_type)]
            ei, ea = [], []
            for u, v, t in pairs:
                ei += [[v, u], [u, v]]
                ea += [t, t]
            atom_type = obj.atom_feat[:, : sro.ATOM_CATEGORIES].argmax(1)
            swapped = sro.build_mol_objects(obj.mol_index, atom_type, np.asarray(ei).T, np.asarray(ea))
            bs = _batch([swapped])
            for arm in ("C", "G", "F"):
                swap_max = max(swap_max, abs(float(models[arm](b0)[0]) - float(models[arm](bs)[0])))
            for arm in ("C", "G", "F"):
                obj_b = _permute_bonds(obj, rng.permutation(obj.bond_index.shape[0]))
                d = abs(float(models[arm](b0)[0]) - float(models[arm](_batch([obj_b]))[0]))
                bond_perm_max = max(bond_perm_max, d)
                obj_j = _permute_junctions(obj, rng.permutation(obj.junc_index.shape[0]))
                d = abs(float(models[arm](b0)[0]) - float(models[arm](_batch([obj_j]))[0]))
                junc_perm_max = max(junc_perm_max, d)

        order = [2, 0, 1]
        b_all = _batch(mols)
        b_perm = _batch([mols[i] for i in order])
        batch_reorder = {
            arm: float((models[arm](b_all)[order] - models[arm](b_perm)).abs().max()) for arm in ("C", "G", "F")
        }

        # cross-molecule isolation: perturb molecule 0, molecule 1 must not move
        a_obj, b_obj = mols[0], mols[1]
        two = _batch([a_obj, b_obj])
        perturbed = sro.MolObjects(**{**a_obj.__dict__})
        af = perturbed.atom_feat.copy()
        af[:, sro.ATOM_CATEGORIES : sro.ATOM_CATEGORIES + sro.ATOM_DEGREE_SLOTS] += 1.0
        perturbed.atom_feat = af
        two_p = _batch([perturbed, b_obj])
        isolation = {}
        for arm in ("C", "G", "F"):
            W0 = models[arm].forward_features(two)["W"]
            W1 = models[arm].forward_features(two_p)["W"]
            nb0 = _batch([a_obj])["bond_feat"].shape[0]
            isolation[arm] = {
                "other_molecule_W_change": float((W0[nb0:] - W1[nb0:]).abs().max()),
                "perturbed_molecule_W_change": float((W0[:nb0] - W1[:nb0]).abs().max()),
                "other_molecule_pred_change": float(abs(float(models[arm](two)[1]) - float(models[arm](two_p)[1]))),
            }

        # empty junction (2 atoms, 1 bond) and a single-bond molecule
        tiny = sro.build_mol_objects(
            -1, np.asarray([0, 1])[:, None], np.asarray([[0, 1], [1, 0]]), np.asarray([0, 0])
        )
        bt = _batch([tiny])
        tiny_res = {}
        for arm in ("C", "G", "F"):
            feats_t = models[arm].forward_features(bt)
            pred_t = models[arm](bt)
            tiny_res[arm] = {
                "n_junctions": int(tiny.junc_index.shape[0]),
                "W_abs_sum": float(feats_t["W"].abs().sum()),
                "V_abs_sum": float(feats_t["V"].abs().sum()),
                "pred_finite": bool(torch.isfinite(pred_t).all()),
            }
        # physical bonds counted once and n_e consistency
        n_from_rows = int(tiny.bond_index.shape[0])
        total_n = float(feats["P3"]["n_b"].sum())
        total_j = int(batch["junc_index"].shape[0])
        # reference: the frozen P3 Null path has the same float32 reduction-order
        # noise under a bond permutation (index_add_ order), so this is not an
        # attention-specific effect.
        p3_bond_order = max(
            abs(float(models["P3"](_batch([obj]))[0])
                - float(models["P3"](_batch([_permute_bonds(obj, rng.permutation(obj.bond_index.shape[0]))]))[0]))
            for obj in mols
        )
    g4 = {
        "atom_relabel_max_abs": relabel_max,
        "P3_reference_bond_order_max_abs": p3_bond_order,
        "edge_direction_swap_max_abs": swap_max,
        "bond_order_permutation_max_abs": bond_perm_max,
        "junction_order_permutation_max_abs": junc_perm_max,
        "molecule_batch_reorder": batch_reorder,
        "cross_molecule_isolation": isolation,
        "empty_junction_single_bond": tiny_res,
        "physical_bonds_per_molecule_once": {
            "tiny_bonds": n_from_rows,
            "sum_n_e_equals_2_times_junctions": total_n == 2.0 * total_j,
        },
        "tolerances": {"symmetry": 5e-5, "isolation": 0.0},
        "tolerance_note": "5e-5 float32 scatter-add reordering; the frozen P3 arm shows the same order",
    }
    gate(
        "G4_symmetry_isolation_empty_sets",
        relabel_max <= 5e-5
        and swap_max <= 5e-5
        and bond_perm_max <= 5e-5
        and junc_perm_max <= 5e-5
        and max(batch_reorder.values()) <= 5e-5
        and all(v["other_molecule_W_change"] == 0.0 and v["other_molecule_pred_change"] == 0.0
                for v in isolation.values())
        and all(v["pred_finite"] and v["W_abs_sum"] == 0.0 and v["V_abs_sum"] == 0.0
                for v in tiny_res.values())
        and total_n == 2.0 * total_j,
        g4,
    )

    # --- G5: multi-step gradients and execution path ------------------------
    grad_probe: dict[str, Any] = {}
    for arm in ("C", "G", "F"):
        model = models[arm]
        model.train()
        target = torch.linspace(-1.0, 1.0, int(batch["n_graphs"]))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        extra_names = [n for n, _ in model.named_parameters() if n.startswith(("attn.", "ffn."))]
        seq = []
        for step in range(1, 7):
            opt.zero_grad()
            loss = F.l1_loss(model(batch), target)
            loss.backward()
            grads = {n: float(p.grad.detach().norm()) if p.grad is not None else None
                     for n, p in model.named_parameters() if n in extra_names}
            seq.append({"step": step, "loss": float(loss), "grad_norms": grads})
            opt.step()
        path = {}
        base_pred = model(batch).detach()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n not in extra_names:
                    continue
                saved = p.detach().clone()
                p.add_(1e-3)
                path[n] = float((model(batch).detach() - base_pred).abs().max())
                p.copy_(saved)
        model.eval()
        grad_probe[arm] = {
            "steps": seq,
            "first_step_zero": sorted([n for n, v in seq[0]["grad_norms"].items() if v == 0.0]),
            "second_step_nonzero": sorted([n for n, v in seq[1]["grad_norms"].items() if v and v > 0.0]),
            "final_step_all_nonzero": all(v and v > 0.0 for v in seq[-1]["grad_norms"].values()),
            "path_sensitivity_after_6_steps": path,
            "all_extra_on_path": all(v > 0.0 for v in path.values()),
        }
    expected_zero = {"C": ["attn.k.weight"], "G": ["attn.k.weight"], "F": ["ffn.t.weight"]}
    expected_nonzero = {"C": ["attn.q.weight"], "G": ["attn.q.weight"], "F": ["ffn.u.weight"]}
    g5 = {
        "per_arm": grad_probe,
        "expected_first_step_zero": expected_zero,
        "expected_first_step_nonzero": expected_nonzero,
    }
    g5_pass = True
    for arm in ("C", "G", "F"):
        p = grad_probe[arm]
        g5_pass &= p["first_step_zero"] == expected_zero[arm]
        g5_pass &= all(n in p["second_step_nonzero"] for n in expected_zero[arm])
        g5_pass &= all(n in p["second_step_nonzero"] for n in expected_nonzero[arm])
        g5_pass &= p["final_step_all_nonzero"]
        g5_pass &= p["all_extra_on_path"]
    gate("G5_multistep_gradients_and_path", bool(g5_pass), g5)

    # --- G6: shared C/G attention computation, G only marginalises ----------
    _, c1, _ = BM.build_variant("C", seed=1, norm=norm, ssl_state=C.ssl_state(1)[0])
    _, g1, _ = BM.build_variant("G", seed=1, norm=norm, ssl_state=C.ssl_state(1)[0])
    c1.eval()
    g1.eval()
    with torch.no_grad():
        fc = c1.forward_features(batch, return_attention=True)
        fg = g1.forward_features(batch, return_attention=True)
        fc_g = c1.forward_features(batch, mode="G", return_attention=True)
    A_same = bool(torch.equal(fc["attn"]["A"], fg["attn"]["A"]))
    A_same_cross_mode = bool(torch.equal(fc["attn"]["A"], fc_g["attn"]["A"]))
    # independent marginalisation from the returned A and p
    row, col, A = fc["attn"]["row"], fc["attn"]["col"], fc["attn"]["A"]
    p = fc_g["attn"]["p"]
    bb, jb = batch["bond_batch"], batch["junc_batch"]
    n_graphs = int(batch["n_graphs"])
    num = torch.zeros(p.numel(), dtype=torch.float64)
    num.index_add_(0, col, fc["n_b"].double()[row] * A.double())
    den = torch.zeros(n_graphs, dtype=torch.float64)
    den.index_add_(0, bb, fc["n_b"].double())
    p_ref = num / den[jb].clamp_min(1e-12)
    p_match = float((p_ref - p.double()).abs().max())
    gw = torch.zeros((n_graphs, BM.DIM), dtype=torch.float64)
    gw.index_add_(0, jb, fc["w"].double() * p_ref.unsqueeze(1))
    W_ref = fc["n_b"].double().unsqueeze(1) * gw[bb]
    W_match = float((W_ref - fc_g["W"].double()).abs().max())
    cons = BM.moment_conservation(fc["W"], fc["V"], fc_g["W"], fc_g["V"], batch)
    disp_c = BM.content_dispersion(fc["W"], fc["V"], fc["n_b"], batch)
    disp_g = BM.content_dispersion(fc_g["W"], fc_g["V"], fc_g["n_b"], batch)
    g6 = {
        "A_identical_C_vs_G_same_weights": A_same,
        "A_identical_C_vs_C_marginal_mode": A_same_cross_mode,
        "p_recomputed_match": p_match,
        "W_G_recomputed_match": W_match,
        "moment_conservation_C_vs_G_same_weights": cons,
        "content_dispersion_C": disp_c,
        "content_dispersion_G": disp_g,
        "single_recipe_for_all_arms": bool(
            set(cfg["arms"]) == set(C.EXPECTED_PARAMETERS)
            and "epochs" not in cfg["arms"]["C"]
            and cfg["recipe"]["epochs"] == C.EPOCHS
        ),
    }
    gate(
        "G6_shared_attention_path_and_marginalisation",
        A_same
        and A_same_cross_mode
        and p_match <= 1e-6
        and W_match <= 1e-5
        and cons["W_max_relative_error"] <= 1e-6
        and cons["V_max_relative_error"] <= 1e-6
        and disp_c["W_content_dispersion"] > 0.0
        and disp_g["W_content_dispersion"] <= 1e-6
        and g6["single_recipe_for_all_arms"],
        g6,
    )

    # --- G7: training-loop parity with the frozen R3 loop -------------------
    synth = [o for _, o in R2A._synthetic_molecules()]
    y_small = np.linspace(-1.0, 1.0, len(synth))
    ref = bp.train_supervised(
        arm="P3",
        seed=0,
        recipe_key="R3",
        init="ssl",
        train_objs=synth,
        rows=list(range(len(synth))),
        y_fit=y_small,
        norm=norm,
        device=torch.device("cpu"),
        ssl_state=ssl0,
        model_kind="base",
        smoke_epochs=2,
    )
    mine = _train_loop(
        arm="P3",
        seed=0,
        train_objs=synth,
        rows=list(range(len(synth))),
        y_fit=y_small,
        norm=norm,
        device=torch.device("cpu"),
        smoke_epochs=2,
        save_soup=False,
        diag_rows=None,
    )
    max_state = max(float((ref["soup_state"][k].float() - mine["soup_state"][k].float()).abs().max())
                    for k in ref["soup_state"])
    g7 = {
        "max_abs_state_diff": max_state,
        "within_1e-4": max_state <= 1e-4,
        "abs_soup_fit_mae_diff": abs(ref["soup_fit_mae"] - mine["soup_fit_mae"]),
        "soup_fit_mae_within_1e-5": abs(ref["soup_fit_mae"] - mine["soup_fit_mae"]) <= 1e-5,
        "ref_updates": ref["updates"],
        "mine_updates": mine["updates"],
        "ref_batch_order": ref["batch_order"]["full_hash"],
        "mine_batch_order": mine["batch_order"]["full_hash"],
    }
    gate(
        "G7_frozen_loop_parity_p3_mode",
        g7["within_1e-4"] and g7["soup_fit_mae_within_1e-5"]
        and g7["ref_updates"] == g7["mine_updates"]
        and g7["ref_batch_order"] == g7["mine_batch_order"],
        g7,
    )

    # --- G8: label isolation ------------------------------------------------
    runner_src = Path(__file__).read_text(encoding="utf-8")
    forbidden_calls = _label_isolation_scan(runner_src)
    guarded = {
        fn: "_no_labels()" in _function_body(runner_src, fn)
        for fn in ("cmd_train", "cmd_predict_valid", "cmd_mechanism", "cmd_smoke")
    }
    blocked = False
    try:
        with _no_labels():
            C.load_valid_y()
    except RuntimeError:
        blocked = True
    g8 = {
        "runner_forbidden_calls": forbidden_calls,
        "stage_functions_use_guard": guarded,
        "label_loader_blocked_under_guard": bool(blocked),
        "official_test_loaded": False,
        "official_valid_loaded": False,
    }
    gate(
        "G8_label_isolation",
        forbidden_calls["pass"] and all(guarded.values()) and blocked,
        g8,
    )

    # --- G9: module and parameter audit for every arm/seed ------------------
    audits = {}
    for seed in C.SEEDS:
        state, _ = C.ssl_state(seed)
        for arm in BM.NEW_VARIANTS:
            _, model, _ = BM.build_variant(arm, seed=seed, norm=norm, ssl_state=state)
            audits[f"{arm}_s{seed}"] = {
                "parameter_audit": BM.parameter_audit(model),
                "module_audit": BM.module_audit(model),
            }
    gate(
        "G9_module_and_parameter_audit",
        all(v["parameter_audit"]["parameter_exact"] and v["module_audit"]["pass"] for v in audits.values()),
        audits,
    )

    gates["failures"] = failures
    gates["all_pass"] = not failures
    gates["environment"] = _env(torch.device("cpu"))
    gates["git_commit"] = _git_commit()
    C.write_once_json(out, gates)
    log(f"[gates] all_pass={gates['all_pass']} failures={failures} -> {out}")
    return 0 if not failures else 1


def _label_isolation_scan(src: str) -> dict[str, Any]:
    """AST scan: no train/predict/mechanism/smoke stage may call a label loader."""
    import ast

    out: dict[str, Any] = {}
    for fn in ("cmd_train", "cmd_predict_valid", "cmd_mechanism", "cmd_smoke"):
        body = _function_body(src, fn)
        tree = ast.parse(body)
        calls: list[str] = []
        splits: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if isinstance(f, ast.Attribute):
                calls.append(f.attr)
            elif isinstance(f, ast.Name):
                calls.append(f.id)
            for kw in node.keywords:
                if kw.arg == "split" and isinstance(kw.value, ast.Constant) and kw.value.value in ("test", "val"):
                    splits.append(str(kw.value.value))
        loader_names = ("load_valid_y", "load_test_y", "_official_y", "load_test_dataset")
        out[fn] = {
            "label_loader_calls": sorted({c for c in calls if c in loader_names}),
            "split_calls": sorted(set(splits)),
        }
    out["pass"] = all(
        not v["label_loader_calls"] and not v["split_calls"] for v in out.values() if isinstance(v, dict)
    )
    return out


def _function_body(src: str, name: str) -> str:
    start = src.find(f"def {name}(")
    if start < 0:
        return ""
    end = src.find("\ndef ", start + 1)
    return src[start:] if end < 0 else src[start:end]


def _permute_bonds(obj: Any, perm: np.ndarray) -> Any:
    """Permute physical-bond rows and remap the junction incidence."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    new = sro.MolObjects(**{**obj.__dict__})
    new.bond_index = obj.bond_index[perm]
    new.bond_type = obj.bond_type[perm]
    new.bond_feat = obj.bond_feat[perm]
    new.junc_bond = inv[obj.junc_bond.reshape(-1)].reshape(obj.junc_bond.shape)
    return new


def _permute_junctions(obj: Any, perm: np.ndarray) -> Any:
    new = sro.MolObjects(**{**obj.__dict__})
    new.junc_index = obj.junc_index[perm]
    new.junc_bond = obj.junc_bond[perm]
    new.junc_feat = obj.junc_feat[perm]
    return new


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------


def cmd_flops(args: argparse.Namespace) -> int:
    out = C.OUT / "flops_this_round.json"
    if out.exists():
        raise SystemExit("flop record exists; refusing to rewrite")
    train_objs = C.load_objects("objects_train10k")
    batch = _batch(train_objs[:128])
    norm = C.load_norm()
    state, _ = C.ssl_state(0)
    results: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "flops",
        "n_graphs": int(batch["n_graphs"]),
        "note": (
            "GEMM counter sees only matmuls; pair gather/mul ops are counted analytically. "
            "cpu_forward_seconds is an unoptimised CPU wall clock and is not a GPU cost "
            "estimate; the GPU projection comes from the smoke stage."
        ),
    }
    import torch
    import time

    for arm in ("P3", "F", "G", "C"):
        _, model, _ = BM.build_variant(arm, seed=0, norm=norm, ssl_state=state)
        model.eval()
        flops = BM.gemm_flops(model, batch)
        ops = BM.attention_op_counts(batch)
        with torch.no_grad():
            for _ in range(2):
                model(batch)
            t0 = time.perf_counter()
            for _ in range(7):
                model(batch)
            secs = (time.perf_counter() - t0) / 7.0
        results[arm] = {
            "gemm_flops": flops["total"],
            "gemm_by_module": flops["by_module"],
            "uses_attention": arm in ("C", "G"),
            "attention_ops": ops if arm in ("C", "G") else None,
            "cpu_forward_seconds": secs,
            "parameters": int(sum(p.numel() for p in model.parameters())),
        }
    base = results["P3"]["gemm_flops"]
    for arm in ("F", "G", "C"):
        results[arm]["gemm_ratio_vs_p3"] = results[arm]["gemm_flops"] / base
        results[arm]["cpu_forward_ratio_vs_p3"] = results[arm]["cpu_forward_seconds"] / results["P3"]["cpu_forward_seconds"]
    C.write_once_json(out, results)
    log(f"[flops] P3 {base/1e9:.4f}G F {results['F']['gemm_ratio_vs_p3']:.4f}x "
        f"G {results['G']['gemm_ratio_vs_p3']:.4f}x C {results['C']['gemm_ratio_vs_p3']:.4f}x")
    return 0


# ---------------------------------------------------------------------------
# training loop (frozen R3 semantics, model injected)
# ---------------------------------------------------------------------------


def _extra_grad_norms(model: Any) -> dict[str, float]:
    out = {}
    for name, p in model.named_parameters():
        if not name.startswith(("attn.", "ffn.")):
            continue
        out[name] = 0.0 if p.grad is None else float(p.grad.detach().norm())
    return out


def _extra_param_norms(model: Any) -> dict[str, float]:
    return {name: float(p.detach().norm()) for name, p in model.named_parameters()
            if name.startswith(("attn.", "ffn."))}


def _attention_snapshot(model: Any, objs: Sequence[Any], device: Any) -> dict[str, Any]:
    import torch

    model.eval()
    batch = bp._mv(sro.batch_objects(list(objs)), device)
    with torch.no_grad():
        _, feats = model(batch, return_features=True, return_attention=True)
    stats = BM.attention_diagnostics(feats["attn"], batch)
    stats.update(BM.content_dispersion(feats["W"], feats["V"], feats["n_b"], batch))
    model.train()
    return stats


def _train_loop(
    *,
    arm: str,
    seed: int,
    train_objs: Sequence[Any],
    rows: Sequence[int],
    y_fit: np.ndarray,
    norm: Mapping[str, Any],
    device: Any,
    smoke_epochs: int | None = None,
    save_soup: bool = True,
    diag_rows: Sequence[int] | None = None,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    recipe = bp.RECIPES[C.RECIPE_KEY]
    model, build_info = C.build_model(arm, seed, device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    if n_params != C.EXPECTED_PARAMETERS[arm]:
        raise SystemExit(f"parameter audit failed: {n_params} != {C.EXPECTED_PARAMETERS[arm]}")

    init_extra = _extra_param_norms(model)
    epochs = int(recipe["epochs"]) if smoke_epochs is None else int(smoke_epochs)
    if smoke_epochs is not None and not (1 <= epochs <= 5):
        raise SystemExit("smoke epochs must be in [1, 5]")
    n = int(len(rows))
    steps_per_epoch = math.ceil(n / C.BATCH_SIZE)
    total_updates = epochs * steps_per_epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=float(recipe["lr"]), weight_decay=float(recipe["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=float(recipe["lr_min"])
    )

    gen = torch.Generator().manual_seed(int(seed) + C.TRAIN_SHUFFLE_OFFSET)
    states: deque = deque(maxlen=C.SOUP_LAST)
    curve: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    grad_probes: dict[str, Any] = {}
    order_hash = hashlib.sha256()
    lr_deviation_max = 0.0
    grad_norm_max = 0.0
    clip_events = 0
    total_mol = 0
    step_seconds: list[float] = []
    diag_batch = None
    diag_objects = None
    if diag_rows is not None and arm in BM.ATTENTION_VARIANTS:
        diag_objects = [train_objs[int(i)] for i in diag_rows]
    snapshot_epochs = {1, C.EPOCHS // 4, C.EPOCHS // 2, 3 * C.EPOCHS // 4, C.EPOCHS}
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        lr_first = lr_last = None
        lr_sum = 0.0
        t_ep = time.perf_counter()
        for indices in bp.shuffle_batches(n, C.BATCH_SIZE, gen):
            global_step = (epoch - 1) * steps_per_epoch + steps + 1
            order_hash.update(np.ascontiguousarray(np.asarray(indices, dtype=np.int64)).tobytes())
            t_step = time.perf_counter()
            batch = bp._mv(sro.batch_objects([train_objs[i] for i in indices]), device)
            y = torch.as_tensor(y_fit[indices], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            if global_step in (1, 2, 11) or (epoch == epochs and steps + 1 == steps_per_epoch):
                grad_probes[str(global_step)] = {
                    "epoch": epoch,
                    "step_in_epoch": steps + 1,
                    "loss": float(loss),
                    "extra_grad_norms": _extra_grad_norms(model),
                }
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP))
            if not math.isfinite(gn):
                raise RuntimeError(f"non-finite gradient norm at epoch {epoch} {arm} seed {seed}")
            grad_norm_max = max(grad_norm_max, gn)
            if gn > C.GRAD_CLIP:
                clip_events += 1
            actual_lr = float(optimizer.param_groups[0]["lr"])
            expected_lr = bp.schedule_lr(recipe, steps + (epoch - 1) * steps_per_epoch, total_updates)
            lr_deviation_max = max(lr_deviation_max, abs(actual_lr - expected_lr))
            if lr_first is None:
                lr_first = actual_lr
            lr_last = actual_lr
            lr_sum += actual_lr
            optimizer.step()
            scheduler.step()
            step_seconds.append(time.perf_counter() - t_step)
            total += float((pred.view(-1) - y.view(-1)).abs().sum())
            n_mol += len(indices)
            steps += 1
        train_mae = total / max(n_mol, 1)
        total_mol += n_mol
        curve.append(
            {
                "epoch": epoch,
                "updates": epoch * steps,
                "train_mae": train_mae,
                "lr_first": lr_first,
                "lr_last": lr_last,
                "lr_mean": lr_sum / max(steps, 1),
                "seconds": time.perf_counter() - t_ep,
            }
        )
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if epoch in snapshot_epochs and diag_objects is not None:
            snap = _attention_snapshot(model, diag_objects, device)
            snapshots.append({"epoch": epoch, **snap})
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            log(f"[{arm} s{seed}] ep={epoch:03d} train={train_mae:.6f} lr={lr_last:.3e} {curve[-1]['seconds']:.1f}s")
        if not math.isfinite(train_mae):
            raise RuntimeError(f"non-finite loss at epoch {epoch} {arm} seed {seed}")
    wall = time.perf_counter() - started

    members, soup = bp._last_soup(list(states), epochs)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = bp.predict_model(model, train_objs, device)
    soup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if not np.isfinite(fit_pred).all():
        raise RuntimeError("non-finite fit prediction after soup")

    # first-step expectation: K/T are exactly zero (Q/U gate the path); after
    # one non-zero update the other matrix must receive a non-zero gradient.
    grad_check: dict[str, Any] = {}
    if arm in BM.ATTENTION_VARIANTS:
        s1 = grad_probes.get("1", {}).get("extra_grad_norms", {})
        s2 = grad_probes.get("2", {}).get("extra_grad_norms", {})
        grad_check = {
            "q_nonzero_first_step": s1.get("attn.q.weight", 0.0) > 0.0,
            "k_zero_first_step": s1.get("attn.k.weight", None) == 0.0,
            "k_nonzero_second_step": s2.get("attn.k.weight", 0.0) > 0.0,
        }
        if not all(grad_check.values()):
            raise RuntimeError(f"attention gradient check failed for {arm} s{seed}: {grad_check}")
    elif arm == "F":
        s1 = grad_probes.get("1", {}).get("extra_grad_norms", {})
        s2 = grad_probes.get("2", {}).get("extra_grad_norms", {})
        grad_check = {
            "u_nonzero_first_step": s1.get("ffn.u.weight", 0.0) > 0.0,
            "t_zero_first_step": s1.get("ffn.t.weight", None) == 0.0,
            "t_nonzero_second_step": s2.get("ffn.t.weight", 0.0) > 0.0,
        }
        if not all(grad_check.values()):
            raise RuntimeError(f"FFN gradient check failed for F s{seed}: {grad_check}")

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "arm": arm,
        "seed": int(seed),
        "recipe_key": C.RECIPE_KEY,
        "recipe": dict(recipe),
        "init": "full10k_masked_ssl",
        "epochs": epochs,
        "updates": int(total_updates),
        "steps_per_epoch": steps_per_epoch,
        "n_train_rows": n,
        "soup_rule": f"parameter mean of the last {C.SOUP_LAST} epoch states",
        "soup_members": members,
        "soup_fit_mae": float(np.abs(fit_pred - y_fit).mean()),
        "schedule": {
            "type": recipe["schedule"],
            "per": "step",
            "warmup": "none",
            "lr0": float(recipe["lr"]),
            "lr_min": float(recipe["lr_min"]),
            "t_max_updates": total_updates,
            "lr_first_step": curve[0]["lr_first"],
            "lr_last_step": curve[-1]["lr_last"],
            "cosine_formula_max_abs_deviation": lr_deviation_max,
        },
        "build": build_info["build"],
        "checks": build_info["checks"],
        "parameter_audit": build_info["parameter_audit"],
        "module_audit": build_info["module_audit"],
        "init_full_sha256": build_info["build"]["scratch"]["init_full_sha256"],
        "init_group_state_sha256": build_info["build"]["scratch"]["init_group_state_sha256"],
        "init_encoder_sha256": build_info["build"]["scratch"]["init_encoder_sha256"],
        "encoder_used_sha256": build_info["encoder_used_sha256"],
        "batch_order": {
            "full_hash": order_hash.hexdigest(),
            "seed": int(seed) + C.TRAIN_SHUFFLE_OFFSET,
        },
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_norm_max": grad_norm_max,
            "lr_schedule_deviation_max": lr_deviation_max,
        },
        "grad_probes": grad_probes,
        "gradient_expectation": grad_check,
        "extra_param_norms_init": init_extra,
        "extra_param_norms_final": _extra_param_norms(model),
        "attention_snapshots": snapshots,
        "step_seconds": {
            "mean": float(np.mean(step_seconds)),
            "median": float(np.median(step_seconds)),
            "p90": float(np.percentile(step_seconds, 90)),
            "n": len(step_seconds),
        },
        "lr_curve": curve,
        "wall_seconds": wall,
        "molecules_seen": int(total_mol),
        "device": str(device),
        "cuda_device_name": None
        if getattr(device, "type", "cpu") == "cpu"
        else __import__("torch").cuda.get_device_name(0),
        "torch": __import__("torch").__version__,
        "python": platform.python_version(),
        "labels_used": "official train 10000 raw y only",
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "smoke": bool(smoke_epochs is not None),
    }
    return {**payload, "soup_state": soup_state}


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def _audit_requires_p3_baseline() -> bool:
    path = C.OUT / "p3_reuse_audit.json"
    if not path.exists():
        return False
    return bool(C.read_json(path)["regime"]["p3_baseline_required"])


def cmd_smoke(args: argparse.Namespace) -> int:
    out = C.OUT / "smoke_this_round.json"
    if out.exists():
        raise SystemExit("smoke record exists; refusing to rewrite")
    device = _device(args.device)
    train_objs = C.load_objects("objects_train10k")
    rows = list(range(1024))
    objs = train_objs[: len(rows)]
    y = C.load_train_y()[rows]
    norm = C.load_norm()
    results: dict[str, Any] = {"protocol_id": C.PROTOCOL_ID, "stage": "smoke", "rows": len(rows),
                               "epochs": 2, "batch_size": C.BATCH_SIZE, "device": str(device)}
    projected = {}
    with _no_labels():
        for arm in ("P3", "F", "G", "C"):
            rec = _train_loop(
                arm=arm,
                seed=0,
                train_objs=objs,
                rows=rows,
                y_fit=y,
                norm=norm,
                device=device,
                smoke_epochs=2,
                save_soup=False,
                diag_rows=None,
            )
            rec.pop("soup_state")
            per_update = rec["step_seconds"]["mean"]
            results[arm] = {**rec, "projected_formal_seconds": per_update * C.UPDATES}
            projected[arm] = per_update * C.UPDATES
            log(f"[smoke] {arm}: step={per_update*1e3:.2f}ms projected={projected[arm]/60:.1f}min")
    planned = 2 * (projected["F"] + projected["G"] + projected["C"])  # two seeds, three arms
    margin = 1.20
    results["budget"] = {
        "projected_formal_gpu_seconds_no_margin": planned,
        "margin": margin,
        "projected_with_margin_seconds": planned * margin,
        "projected_with_margin_gpu_hours": planned * margin / 3600.0,
        "cap_gpu_hours": float(C.load_config()["budget"]["gpu_hours_cap"]),
        "ok": planned * margin / 3600.0 <= float(C.load_config()["budget"]["gpu_hours_cap"]),
        "per_arm_seconds": projected,
    }
    results["environment"] = _env(device)
    C.write_once_json(out, results)
    log(f"[smoke] projected {results['budget']['projected_with_margin_gpu_hours']:.3f} GPU-h "
        f"(cap {results['budget']['cap_gpu_hours']}) ok={results['budget']['ok']}")
    return 0 if results["budget"]["ok"] else 1


# ---------------------------------------------------------------------------
# formal training
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    arm, seed = args.arm, int(args.seed)
    if arm not in BM.VARIANTS:
        raise SystemExit(f"unknown arm {arm!r}")
    if seed not in C.SEEDS:
        raise SystemExit(f"unknown seed {seed}")
    if arm == "P3" and not _audit_requires_p3_baseline():
        raise SystemExit("P3 baseline training is only allowed if the audit flagged a regime mismatch")
    if C.train_path(arm, seed).exists() or C.soup_path(arm, seed).exists():
        raise SystemExit("arm already trained; retraining is not allowed")

    device = _device(args.device)
    train_objs = C.load_objects("objects_train10k")
    y = C.load_train_y()
    norm = C.load_norm()
    cfg = C.load_config()
    with _no_labels():
        payload = _train_loop(
            arm=arm,
            seed=seed,
            train_objs=train_objs,
            rows=list(range(C.N_TRAIN)),
            y_fit=y,
            norm=norm,
            device=device,
            smoke_epochs=None,
            save_soup=True,
            diag_rows=list(range(C.BATCH_SIZE)),
        )
    if payload["epochs"] != C.EPOCHS or payload["updates"] != C.UPDATES:
        raise SystemExit("formal run did not use the frozen epoch budget")
    if payload["batch_order"]["full_hash"] != cfg["audit_expectations"]["batch_order_sha256"][str(seed)]:
        raise SystemExit(
            f"batch order mismatch: {payload['batch_order']['full_hash']} != "
            f"{cfg['audit_expectations']['batch_order_sha256'][str(seed)]}"
        )
    soup = payload.pop("soup_state")
    import torch

    torch.save(soup, C.soup_path(arm, seed))
    record = dict(payload)
    record["soup_path"] = _stored(C.soup_path(arm, seed))
    record["soup_sha256"] = C.sha256_file(C.soup_path(arm, seed))
    record["train_objects_sha256"] = C.sha256_file(REPO / cfg["frozen_inputs"]["objects_train10k"]["path"])
    record["train_y_sha256"] = C.sha256_array(y, "float64")
    record["environment"] = _env(device)
    record["git_commit"] = _git_commit()
    C.write_once_json(C.train_path(arm, seed), record)
    log(
        f"[train {arm} s{seed}] params {record['parameter_audit']['actual_parameters']} "
        f"soup fit {record['soup_fit_mae']:.6f} wall {record['wall_seconds']:.0f}s"
    )
    return 0


# ---------------------------------------------------------------------------
# official-valid prediction + freeze
# ---------------------------------------------------------------------------


def cmd_predict_valid(args: argparse.Namespace) -> int:
    arm, seed = args.arm, int(args.seed)
    if arm not in C.NEW_ARMS:
        raise SystemExit(f"unknown arm {arm!r}")
    freeze_path = C.valid_freeze_path(arm, seed)
    if freeze_path.exists():
        raise SystemExit("valid predictions already frozen; refusing to rewrite")
    if not C.train_path(arm, seed).exists() or not C.soup_path(arm, seed).exists():
        raise SystemExit("arm is not trained")
    record = C.read_json(C.train_path(arm, seed))
    if record.get("smoke"):
        raise SystemExit("smoke run cannot be used for a frozen prediction")
    if C.sha256_file(C.soup_path(arm, seed)) != record["soup_sha256"]:
        raise SystemExit("soup changed after training")

    device = _device(args.device)
    valid_objs = C.load_objects("objects_valid")
    cfg = C.load_config()
    with _no_labels():
        model, info = C.build_model(arm, seed, device)
        soup = C.load_soup(model, C.soup_path(arm, seed))
        pred, seconds = C.predict(model, valid_objs, device)
    if not np.isfinite(pred).all():
        raise SystemExit("non-finite valid prediction")
    np.savez_compressed(C.valid_pred_path(arm, seed), row=np.arange(C.N_VALID, dtype=np.int64),
                        pred=pred.astype(np.float32))
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "valid_predict_freeze",
        "arm": arm,
        "seed": seed,
        "prediction_file": _stored(C.valid_pred_path(arm, seed)),
        "prediction_sha256": C.sha256_file(C.valid_pred_path(arm, seed)),
        "prediction_array_sha256": C.sha256_array(pred),
        "soup": soup,
        "parameters": info["parameters"],
        "inference_seconds": seconds,
        "n": C.N_VALID,
        "labels_read": False,
        "official_valid_labels_read": False,
        "official_test_loaded": False,
        "environment": _env(device),
        "created_utc": _now(),
    }
    C.write_once_json(freeze_path, payload)
    log(f"[predict-valid {arm} s{seed}] sha {payload['prediction_sha256'][:12]} {seconds:.3f}s")
    return 0


# ---------------------------------------------------------------------------
# mechanism diagnostics (label-free, before the valid label read)
# ---------------------------------------------------------------------------


def cmd_mechanism(args: argparse.Namespace) -> int:
    out = C.OUT / "mechanism_this_round.json"
    if out.exists():
        raise SystemExit("mechanism record exists; refusing to rewrite")
    import torch

    cfg = C.load_config()
    device = _device(args.device)
    valid_objs = C.load_objects("objects_valid")
    # every formal soup and frozen prediction must already exist
    for arm in C.NEW_ARMS:
        for seed in C.SEEDS:
            if not C.soup_path(arm, seed).exists() or not C.valid_freeze_path(arm, seed).exists():
                raise SystemExit(f"missing frozen run for {arm} s{seed}")

    report: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "mechanism",
        "created_utc": _now(),
        "labels_read": False,
        "official_test_loaded": False,
        "n_valid_structures": len(valid_objs),
        "per_seed": {},
    }
    plan = {
        "C": ("C", "G", "uniform"),
        "G": ("G", "uniform", "C"),
    }
    with _no_labels():
        for seed in C.SEEDS:
            entry: dict[str, Any] = {}
            for arm in ("C", "G"):
                train_rec = C.read_json(C.train_path(arm, seed))
                model, info = C.build_model(arm, seed, device)
                soup = C.load_soup(model, C.soup_path(arm, seed))
                if soup["sha256"] != train_rec["soup_sha256"]:
                    raise SystemExit("soup changed after training")
                model.eval()
                mode_stats: dict[str, Any] = {}
                preds: dict[str, np.ndarray] = {}
                for mode in plan[arm]:
                    batch_preds = []
                    agg: list[dict[str, Any]] = []
                    cons: list[dict[str, float]] = []
                    for i in range(0, len(valid_objs), 128):
                        chunk = list(valid_objs[i : i + 128])
                        batch = bp._mv(sro.batch_objects(chunk), device)
                        with torch.no_grad():
                            pred, feats = model(
                                batch, mode=mode, return_features=True, return_attention=True
                            )
                        batch_preds.append(pred.detach().cpu().numpy())
                        agg.append(BM.attention_diagnostics(feats["attn"], batch))
                        agg[-1].update(BM.content_dispersion(feats["W"], feats["V"], feats["n_b"], batch))
                        if arm == "C" and mode == "C":
                            with torch.no_grad():
                                feats_g = model.forward_features(batch, mode="G")
                            cons.append(BM.moment_conservation(
                                feats["W"], feats["V"], feats_g["W"], feats_g["V"], batch
                            ))
                    pred_arr = np.concatenate(batch_preds, 0).astype(np.float32)
                    if not np.isfinite(pred_arr).all():
                        raise SystemExit("non-finite mechanism prediction")
                    preds[mode] = pred_arr.astype(np.float64)
                    mode_stats[mode] = {
                        "D_molecule_mean": _nanmean([a["D_molecule_mean"] for a in agg]),
                        "D_molecule_weighted": _nanmean([a["D_molecule_weighted"] for a in agg]),
                        "entropy_norm_mean": _nanmean([a["entropy_norm_mean"] for a in agg]),
                        "max_weight_mean": _nanmean([a["max_weight_mean"] for a in agg]),
                        "W_content_dispersion": _nanmean([a["W_content_dispersion"] for a in agg]),
                        "V_content_dispersion": _nanmean([a["V_content_dispersion"] for a in agg]),
                        "n_pairs": int(sum(a["n_pairs"] for a in agg)),
                    }
                    path = C.mech_pred_path(arm, seed, mode)
                    if path.exists():
                        raise SystemExit(f"mechanism prediction already exists: {path}")
                    np.savez_compressed(path, row=np.arange(C.N_VALID, dtype=np.int64), pred=pred_arr)
                    mode_stats[mode]["prediction_file"] = _stored(path)
                    mode_stats[mode]["prediction_sha256"] = C.sha256_file(path)
                    mode_stats[mode]["prediction_array_sha256"] = C.sha256_array(pred_arr)
                # primary frozen prediction must be reproduced by this path
                freeze = C.read_json(C.valid_freeze_path(arm, seed))
                primary = plan[arm][0]
                src = np.load(REPO / freeze["prediction_file"])["pred"]
                mode_stats["primary_matches_frozen_valid"] = {
                    "max_abs_diff": float(np.abs(preds[primary] - src).max()),
                    "within_1e-5": float(np.abs(preds[primary] - src).max()) <= 1e-5,
                }
                deltas = {}
                for other in plan[arm][1:]:
                    d = np.abs(preds[primary] - preds[other])
                    deltas[f"{primary}_vs_{other}"] = {
                        "mean_abs": float(d.mean()),
                        "max_abs": float(d.max()),
                        "p99_abs": float(np.percentile(d, 99)),
                    }
                entry[arm] = {
                    "soup": soup,
                    "parameters": info["parameters"],
                    "modes": mode_stats,
                    "prediction_deltas": deltas,
                    "moment_conservation_C_vs_G_same_weights": {
                        "W_max_relative_error": max((c["W_max_relative_error"] for c in cons), default=None),
                        "V_max_relative_error": max((c["V_max_relative_error"] for c in cons), default=None),
                        "W_mean_relative_error": _nanmean([c["W_mean_relative_error"] for c in cons]),
                        "V_mean_relative_error": _nanmean([c["V_mean_relative_error"] for c in cons]),
                    } if cons else None,
                }
            report["per_seed"][str(seed)] = entry
    # pre-registered inertness verdict (descriptive, never used to select an arm)
    inert = {}
    for seed in C.SEEDS:
        c = report["per_seed"][str(seed)]["C"]
        m = c["modes"]["C"]
        d_uniform = c["prediction_deltas"]["C_vs_uniform"]["mean_abs"]
        inert[str(seed)] = {
            "D_molecule_mean": m["D_molecule_mean"],
            "entropy_norm_mean": m["entropy_norm_mean"],
            "mean_abs_prediction_change_vs_uniform": d_uniform,
            "attention_inert": bool(
                m["D_molecule_mean"] is not None
                and m["D_molecule_mean"] <= cfg["mechanism"]["inert_D_max"]
                and m["entropy_norm_mean"] >= cfg["mechanism"]["inert_entropy_min"]
                and d_uniform <= cfg["mechanism"]["inert_prediction_change_max"]
            ),
        }
    report["inertness"] = inert
    report["attention_participates"] = not all(v["attention_inert"] for v in inert.values())
    report["environment"] = _env(device)
    report["git_commit"] = _git_commit()
    C.write_once_json(out, report)
    for seed in C.SEEDS:
        log(f"[mechanism] s{seed}: inert={inert[str(seed)]}")
    return 0


def _nanmean(values: Sequence[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------------
# official-valid scoring (single label read)
# ---------------------------------------------------------------------------


def _frozen_preds() -> dict[str, dict[int, np.ndarray]]:
    cfg = C.load_config()
    preds: dict[str, dict[int, np.ndarray]] = {name: {} for name in C.ALL_MODELS}
    for seed in C.SEEDS:
        hist, rec = C.p3_valid_pred(seed)
        if hist is not None:
            preds["P3"][seed] = hist
        else:
            path = C.OUT / f"p3_valid_pred_replay_s{seed}.npz"
            audit = C.read_json(C.OUT / "p3_reuse_audit.json")["p3"][str(seed)]
            if not path.exists() or C.sha256_file(path) != audit["replay_file_sha256"]:
                raise SystemExit("frozen P3 replay array missing or changed")
            with np.load(path) as z:
                pred = np.asarray(z["pred"], dtype=np.float64)
            if C.sha256_array(pred) != audit["replay_array_sha256"]:
                raise SystemExit("P3 replay array hash mismatch")
            preds["P3"][seed] = pred
        for arm in C.NEW_ARMS:
            freeze = C.read_json(C.valid_freeze_path(arm, seed))
            path = REPO / freeze["prediction_file"]
            if C.sha256_file(path) != freeze["prediction_sha256"]:
                raise SystemExit(f"valid prediction changed after freeze: {path}")
            with np.load(path) as z:
                pred = np.asarray(z["pred"], dtype=np.float64)
            if pred.shape != (C.N_VALID,):
                raise SystemExit(f"unexpected valid prediction shape {pred.shape}")
            if C.sha256_array(pred) != freeze["prediction_array_sha256"]:
                raise SystemExit(f"valid prediction array hash mismatch: {path}")
            preds[arm][seed] = pred
    return preds


def cmd_score(args: argparse.Namespace) -> int:
    out_metrics = C.OUT / "valid_metrics.json"
    out_decision = C.OUT / "valid_decision.json"
    if out_decision.exists():
        raise SystemExit("valid decision exists; refusing to rewrite")
    cfg = C.load_config()
    if not (C.OUT / "mechanism_this_round.json").exists():
        raise SystemExit("mechanism diagnostics must be frozen before scoring")
    preds = _frozen_preds()

    combined = C.OUT / "valid_prediction_freeze.json"
    if not combined.exists():
        entries = {}
        for name, by_seed in preds.items():
            for seed, pred in by_seed.items():
                entries[f"{name}{seed}"] = {
                    "array_sha256": C.sha256_array(pred),
                    "source": "historical_frozen" if name == "P3" else "this_round_frozen",
                }
        C.write_once_json(combined, {
            "protocol_id": C.PROTOCOL_ID,
            "stage": "valid_prediction_freeze_combined",
            "entries": entries,
            "n": len(entries),
            "valid_labels_read_at_freeze": False,
            "created_utc": _now(),
        })
    else:
        frozen = C.read_json(combined)
        for name, by_seed in preds.items():
            for seed, pred in by_seed.items():
                if frozen["entries"][f"{name}{seed}"]["array_sha256"] != C.sha256_array(pred):
                    raise SystemExit("prediction array changed since the combined freeze")
    mechanism_hash = C.sha256_file(C.OUT / "mechanism_this_round.json")

    lock = C.OUT / "valid_read.lock"
    recover = bool(getattr(args, "recover", False))
    if lock.exists() and not recover:
        raise SystemExit(
            "valid label read already consumed (valid_read.lock exists); "
            "a second read requires score --recover and is recorded as a deviation"
        )
    y = C.load_valid_y()
    if lock.exists():
        prior = C.read_json(lock)
        read_count = int(prior.get("read_count", 1)) + 1
        C.write_json(C.OUT / "score_recovery_deviation.json", {
            "protocol_id": C.PROTOCOL_ID,
            "stage": "score_recovery",
            "created_utc": _now(),
            "reason": "score re-run after a consumed valid-label read",
            "prior_read_utc": prior.get("read_utc"),
            "read_count": read_count,
            "note": "post-hoc recovery; reported as a deviation in REPORT.md",
        })
    else:
        read_count = 1
    C.write_json(lock, {
        "protocol_id": C.PROTOCOL_ID,
        "valid_labels_read": True,
        "read_count": read_count,
        "y_source": "official PyG ZINC subset=True split=val; historical array hash verified",
        "historical_y_path": cfg["label_provenance"]["valid_y"]["historical_path"],
        "y_sha256_float32": C.sha256_array(y),
        "y_sha256_float64": C.sha256_array(y, "float64"),
        "prediction_freeze_sha256": C.sha256_file(combined),
        "mechanism_freeze_sha256": mechanism_hash,
        "read_utc": _now(),
        "reread": read_count > 1,
    })

    valid_objs = C.load_objects("objects_valid")
    longest, penalty = C.cycle_proxy(valid_objs)
    masks = {
        "all": np.ones(C.N_VALID, dtype=bool),
        "ordinary_y_gt_minus3": y > C.TAIL_THRESHOLD,
        "tail_y_le_minus3": y <= C.TAIL_THRESHOLD,
        "penalty_proxy": penalty,
        "penalty_complement": ~penalty,
        "without_extreme_valid172": np.arange(C.N_VALID) != C.EXTREME_VALID_ROW,
    }

    p3_mae = {str(s): float(np.abs(preds["P3"][s] - y).mean()) for s in C.SEEDS}
    p3_diff = {str(s): abs(p3_mae[str(s)] - float(cfg["audit_expectations"]["historical_valid_mae"][str(s)]))
               for s in C.SEEDS}
    if max(p3_diff.values()) > 1e-5:
        raise SystemExit(f"frozen P3 valid MAE does not reproduce the historical MAE: {p3_diff}")

    contrasts = {
        "C_minus_P3": C.contrast("C", "P3", preds, y, masks),
        "G_minus_P3": C.contrast("G", "P3", preds, y, masks),
        "F_minus_P3": C.contrast("F", "P3", preds, y, masks),
        "C_minus_G": C.contrast("C", "G", preds, y, masks),
        "C_minus_F": C.contrast("C", "F", preds, y, masks),
        "G_minus_F": C.contrast("G", "F", preds, y, masks),
    }
    thresholds = cfg["thresholds"]
    promising = {name: C.performance_promising(contrasts[f"{name}_minus_P3"], thresholds)
                 for name in ("F", "G", "C")}
    conditioning = C.conditioning_promising(contrasts["C_minus_G"], thresholds)
    capacity = C.capacity_claim(contrasts["C_minus_F"])

    metrics = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "valid_score",
        "created_utc": _now(),
        "n": C.N_VALID,
        "primary_metric": "raw-y MAE, seed-mean of two single-model seeds",
        "labels": {
            "source": "official PyG ZINC subset=True split=val; historical array hash verified",
            "array_sha256_float32": C.sha256_array(y),
            "array_sha256_float64": C.sha256_array(y, "float64"),
            "read_count_this_round": read_count,
        },
        "models": P2B.metric_block(preds, y),
        "p3_replay_check": {
            "per_seed_mae": p3_mae,
            "recorded_historical_mae": {str(s): float(cfg["audit_expectations"]["historical_valid_mae"][str(s)])
                                        for s in C.SEEDS},
            "abs_diff": p3_diff,
            "within_1e-5": True,
            "branch": "frozen_array" if C.p3_valid_pred(0)[0] is not None else "regenerated_replay",
        },
        "contrasts": contrasts,
        "threshold_evaluation": {
            "performance_promising": promising,
            "conditioning_promising": conditioning,
            "capacity_claim_C_better_than_F": capacity,
        },
        "subgroups": P2B.subgroup_metrics(preds, y, masks),
        "masks": {name: {"n": int(mask.sum()), "rows_sha256": C.sha256_array(np.flatnonzero(mask), "int64")}
                  for name, mask in masks.items()},
        "penalty_proxy": {
            "definition": "max(len(c) for c in nx.cycle_basis(G)) > 6",
            "longest_cycle_histogram": {str(int(k)): int((longest == k).sum()) for k in np.unique(longest)},
            "n_penalised": int(penalty.sum()),
            "order_sensitive_proxy": True,
            "not_a_model_input": True,
        },
        "extreme_row": {"row": C.EXTREME_VALID_ROW, "y": float(y[C.EXTREME_VALID_ROW])},
        "official_test_loaded": False,
        "mechanism_freeze_sha256": mechanism_hash,
        "git_commit": _git_commit(),
    }

    mechanism = C.read_json(C.OUT / "mechanism_this_round.json")
    labels = _decision_labels(promising, conditioning, capacity, contrasts, mechanism)
    decision = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "valid_decision",
        "created_utc": _now(),
        "thresholds": thresholds,
        "performance_promising": promising,
        "conditioning_promising": conditioning,
        "capacity_claim_C_better_than_F": capacity,
        "seed_mean_mae": {name: C.seed_mean_mae(preds, name, y) for name in C.ALL_MODELS},
        "primary_gain_vs_p3": {name: contrasts[f"{name}_minus_P3"]["mean_gain"] for name in ("F", "G", "C")},
        "gain_G_C": contrasts["C_minus_G"]["mean_gain"],
        "gain_F_C": contrasts["C_minus_F"]["mean_gain"],
        "attention_participates": bool(mechanism.get("attention_participates")),
        "inertness": mechanism.get("inertness"),
        "labels": labels,
        "primary_label": labels[0] if labels else "INCONCLUSIVE",
        "valid_labels_read_count": read_count,
        "official_test_loaded": False,
        "no_test_access_in_this_round": True,
        "git_commit": _git_commit(),
    }
    C.write_once_json(out_metrics, metrics)
    C.write_once_json(out_decision, decision)
    log(f"[score] labels={labels}")
    for name in ("F", "G", "C"):
        c = contrasts[f"{name}_minus_P3"]
        log(f"  {name}-P3: mean={c['mean_gain']:+.6f} per-seed={c['per_seed_gain']} "
            f"ci={c['bootstrap_mae_delta']['ci95_low']:+.6f}..{c['bootstrap_mae_delta']['ci95_high']:+.6f}")
    log(f"  C-G: {contrasts['C_minus_G']['mean_gain']:+.6f}   C-F: {contrasts['C_minus_F']['mean_gain']:+.6f}")
    return 0


def _decision_labels(promising: Mapping[str, Any], conditioning: Mapping[str, Any],
                     capacity: Mapping[str, Any], contrasts: Mapping[str, Any],
                     mechanism: Mapping[str, Any]) -> list[str]:
    """Pre-registered multi-axis reading (PROTOCOL §9); first label is primary."""
    labels: list[str] = []
    inert_all = all(v.get("attention_inert") for v in mechanism.get("inertness", {}).values())
    any_promising = any(v["pass"] for v in promising.values())
    if inert_all:
        labels.append("E_ATTENTION_NUMERICALLY_INERT")
    if promising["C"]["pass"] and conditioning["pass"] and capacity["pass"]:
        labels.append("A_CONDITIONAL_ALLOCATION_PROMISING")
    if promising["G"]["pass"] and not (conditioning["pass"] and capacity["pass"]):
        labels.append("B_GLOBAL_REWEIGHTING_PROMISING")
    if promising["F"]["pass"] and not (conditioning["pass"] and capacity["pass"]):
        labels.append("C_CAPACITY_GAIN_DOMINATES")
    if mechanism.get("attention_participates") and not any_promising:
        labels.append("D_ATTENTION_ACTIVE_NO_USEFUL_GAIN")
    sign_conflict = any(
        len({v > 0 for v in contrasts[f"{name}_minus_P3"]["per_seed_gain"].values()}) > 1
        for name in ("F", "G", "C")
    )
    if not labels or sign_conflict:
        labels.append("F_INCONCLUSIVE")
    return labels


# ---------------------------------------------------------------------------
# convenience wrappers / CLI
# ---------------------------------------------------------------------------


def cmd_prep(args: argparse.Namespace) -> int:
    sub = argparse.Namespace(device=args.device)
    cmd_audit(sub)
    cmd_gates(sub)
    cmd_flops(sub)
    return 0


def cmd_run_arm(args: argparse.Namespace) -> int:
    if args.arm == "P3" and not _audit_requires_p3_baseline():
        raise SystemExit("P3 baseline training is only allowed after a regime-mismatch audit")
    for seed in C.SEEDS:
        if not (C.train_path(args.arm, seed).exists() and C.soup_path(args.arm, seed).exists()):
            cmd_train(argparse.Namespace(arm=args.arm, seed=seed, device=args.device))
        if args.arm in C.NEW_ARMS and not C.valid_freeze_path(args.arm, seed).exists():
            cmd_predict_valid(argparse.Namespace(arm=args.arm, seed=seed, device=args.device))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(C.OUT), help="results directory")
    sub = p.add_subparsers(dest="command", required=True)

    for name in ("audit", "gates", "flops", "prep"):
        a = sub.add_parser(name)
        a.add_argument("--device", default="cpu")
    s = sub.add_parser("smoke")
    s.add_argument("--device", default="cuda")
    t = sub.add_parser("train")
    t.add_argument("--arm", required=True, choices=list(BM.VARIANTS))
    t.add_argument("--seed", required=True, type=int, choices=list(C.SEEDS))
    t.add_argument("--device", default="cuda")
    sc = sub.add_parser("score")
    sc.add_argument("--recover", action="store_true", help="re-run after a consumed label read (recorded as a deviation)")

    pv = sub.add_parser("predict-valid")
    pv.add_argument("--arm", required=True, choices=list(C.NEW_ARMS))
    pv.add_argument("--seed", required=True, type=int, choices=list(C.SEEDS))
    pv.add_argument("--device", default="cuda")
    ra = sub.add_parser("run-arm")
    ra.add_argument("--arm", required=True, choices=list(BM.VARIANTS))
    ra.add_argument("--device", default="cuda")
    m = sub.add_parser("mechanism")
    m.add_argument("--device", default="cpu")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    C.OUT = Path(args.out)
    C.OUT.mkdir(parents=True, exist_ok=True)
    return {
        "audit": cmd_audit,
        "gates": cmd_gates,
        "flops": cmd_flops,
        "prep": cmd_prep,
        "smoke": cmd_smoke,
        "train": cmd_train,
        "predict-valid": cmd_predict_valid,
        "run-arm": cmd_run_arm,
        "mechanism": cmd_mechanism,
        "score": cmd_score,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
