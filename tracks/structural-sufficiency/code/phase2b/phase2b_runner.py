"""SSRC Phase 2b runner — standard ZINC-12k validation of the frozen F/M arms.

Protocol: ``results/phase2b/PROTOCOL.md`` (``ssrc-phase2b-zinc-v1``).

Stages
------
``audit``          read-only hash/parameter/init/batch-order/valid-replay audit of
                   the historical full-10k P3 checkpoints, the two SSL encoders and
                   the official-valid/test instruments (no test labels).
``gates``          CPU correctness gates, including a bitwise loop-parity check
                   against the frozen Phase 2a training loop and a replay of the
                   frozen Phase 2a F/M Monitor predictions.
``flops``          forward GEMM FLOPs for P3 / F / M on one fixed 128-graph batch.
``smoke``          2-epoch timing/numerics run on a 1,024-row prefix (never scored).
``train``          one formal arm/seed: frozen R3, 360 epochs, all 10,000 rows.
``predict-valid``  freeze official-valid predictions (no labels read).
``score-valid``    the single official-valid label read + frozen promotion rule.
``test-predict``   ONLY if a finalist was promoted: freeze its two test predictions.
``score-test``     ONLY after both test predictions are frozen: read test labels.

Every stage refuses to overwrite an existing artefact.
"""

from __future__ import annotations

import argparse
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
for _p in (HERE, REPO / "tracks/structure-contribution/code", REPO / "tracks/structural-sufficiency/code/phase2a"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import phase2b_common as C  # noqa: E402
import phase2a_models as M  # noqa: E402
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


def _train_path(arm: str, seed: int) -> Path:
    return C.OUT / f"train_{arm}_s{seed}.json"


def _soup_path(arm: str, seed: int) -> Path:
    return C.OUT / f"soup_{arm}_s{seed}.pt"


def _valid_pred_path(arm: str, seed: int) -> Path:
    return C.OUT / f"valid_pred_{arm}_s{seed}.npz"


def _valid_freeze_path(arm: str, seed: int) -> Path:
    return C.OUT / f"valid_freeze_{arm}_s{seed}.json"


def _test_pred_path(arm: str, seed: int) -> Path:
    return C.OUT / f"test_pred_{arm}_s{seed}.npz"


def _test_freeze_path(arm: str, seed: int) -> Path:
    return C.OUT / f"test_freeze_{arm}_s{seed}.json"


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

    return {
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if getattr(device, "type", "cpu") == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "git_commit": _git_commit(),
        "utc": _now(),
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def cmd_audit(args: argparse.Namespace) -> int:
    out = C.OUT / "p3_reuse_audit.json"
    if out.exists():
        raise SystemExit("audit record exists; refusing to rewrite")

    cfg = C.load_config()
    checks: dict[str, Any] = {"protocol_id": C.PROTOCOL_ID, "stage": "audit", "created_utc": _now()}

    # --- frozen input hashes -------------------------------------------------
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

    # --- code hashes vs the P3 evaluation revision ---------------------------
    code_files = {
        "structural_relation_composer_v1": REPO / "tracks/structure-contribution/code/structural_relation_composer_v1.py",
        "structural_relation_objects_v1": REPO / "tracks/structure-contribution/code/structural_relation_objects_v1.py",
        "bjoc_masked_pretraining_v1": REPO / "tracks/structure-contribution/code/bjoc_masked_pretraining_v1.py",
        "phase2a_models": HERE.parent / "phase2a/phase2a_models.py",
    }
    code_now = {k: C.sha256_file(p) for k, p in code_files.items()}
    code_expected = {k: v for k, v in cfg["audit_expectations"]["code_hashes"].items() if k in code_now}
    checks["code_hashes"] = {
        "current": code_now,
        "expected_when_p3_was_evaluated": code_expected,
        "all_match": code_now == code_expected,
    }
    if not checks["code_hashes"]["all_match"]:
        raise SystemExit(f"model/object code hash differs from the P3 evaluation revision: {code_now}")

    # --- data ----------------------------------------------------------------
    train_objs = C.load_objects(C.OBJECTS_TRAIN, cfg["frozen_inputs"]["objects_train10k"]["sha256"])
    valid_objs = C.load_objects(C.OBJECTS_VALID, cfg["frozen_inputs"]["objects_valid"]["sha256"])
    if [int(o.mol_index) for o in train_objs] != list(range(C.N_TRAIN)):
        raise SystemExit("objects_train10k row order is not 0..9999")
    if [int(o.mol_index) for o in valid_objs] != list(range(C.N_VALID)):
        raise SystemExit("objects_valid row order is not 0..999")
    y = C.load_train_y()
    checks["data"] = {
        "n_train_objects": len(train_objs),
        "n_valid_objects": len(valid_objs),
        "train_y_sha256": C.sha256_array(y, "float64"),
        "train_y_stats": {"min": float(y.min()), "max": float(y.max()), "mean": float(y.mean())},
        "standardiser": {
            "file": str(C.PREP_TRAIN.relative_to(REPO)),
            "fit_only": True,
            "n_fit": C.N_TRAIN,
        },
    }
    # raw train targets must equal the official PyG train targets (no label drift)
    import cscl_features as cf

    ds = cf.load_official_train()
    if len(ds) != C.N_TRAIN:
        raise SystemExit("official train split size changed")
    y_official = np.asarray([float(ds[i].y.reshape(-1)[0]) for i in range(C.N_TRAIN)], dtype=np.float64)
    y_match = bool(np.array_equal(y, y_official))
    checks["data"]["train_y_matches_official_pyg"] = y_match
    checks["data"]["train_y_max_abs_diff_vs_pyg"] = float(np.abs(y - y_official).max())
    if not y_match:
        raise SystemExit("supervised train targets differ from the official PyG targets")

    # --- historical P3: hashes, parameters, replay ---------------------------
    p3: dict[str, Any] = {}
    device = _device(args.device)
    recorded_valid = C.read_json(C.RECIPE / "confirmation_metrics.json")["contrast"]["per_seed_mae_new"]
    for seed in C.SEEDS:
        model, info = C.build_p3(seed, device)
        params = int(sum(p.numel() for p in model.parameters()))
        forbidden = M.src.forbidden_module_scan(model)
        pred, secs = C.predict(model, valid_objs, device)
        prov = C.historical_prediction(f"valid_P3{seed}")
        array_sha = C.sha256_array(pred)
        if array_sha != prov["array_sha256_float32"]:
            raise SystemExit(
                f"regenerated P3 seed {seed} valid array does not match the historical hash: "
                f"{array_sha} != {prov['array_sha256_float32']}"
            )
        hist_path = REPO / prov["historical_path"]
        max_abs = None
        if hist_path.exists():
            with np.load(hist_path) as z:
                hist = np.asarray(z["pred"], dtype=np.float64)
            max_abs = float(np.abs(pred - hist).max())
        p3_pred_path = C.OUT / f"p3_valid_pred_s{seed}.npz"
        p3_freeze_path = C.OUT / f"p3_valid_freeze_s{seed}.json"
        if p3_freeze_path.exists():
            fr = C.read_json(p3_freeze_path)
            if (
                fr["prediction_array_sha256"] != array_sha
                or C.sha256_file(p3_pred_path) != fr["prediction_sha256"]
            ):
                raise SystemExit("regenerated P3 valid prediction changed after its freeze")
        else:
            np.savez_compressed(
                p3_pred_path, row=np.arange(C.N_VALID, dtype=np.int64), pred=pred.astype(np.float32)
            )
            C.write_once_json(
                p3_freeze_path,
                {
                    "protocol_id": C.PROTOCOL_ID,
                    "stage": "p3_valid_regenerated_from_frozen_soup",
                    "arm": "P3",
                    "seed": seed,
                    "prediction_file": _stored(p3_pred_path),
                    "prediction_sha256": C.sha256_file(p3_pred_path),
                    "prediction_array_sha256": array_sha,
                    "soup": info["soup"],
                    "historical_path": prov["historical_path"],
                    "historical_file_sha256": prov["historical_file_sha256"],
                    "historical_array_sha256": prov["array_sha256_float32"],
                    "bit_identical_to_historical_array": True,
                    "labels_read": False,
                    "created_utc": _now(),
                },
            )
        rec = C.p3_run(seed)
        batch_ok = rec["batch_order"]["full_hash"] == cfg["audit_expectations"]["batch_order_sha256"][str(seed)]
        p3[str(seed)] = {
            "soup": info["soup"],
            "parameters": params,
            "parameter_exact_413505": params == 413505,
            "forbidden_modules": forbidden,
            "init": {
                "init_full_sha256": rec["init"]["init_full_sha256"],
                "init_group_state_sha256": rec["init"]["init_group_state_sha256"],
                "encoder_used_sha256": rec["init"]["encoder_used_sha256"],
                "pretrain_encoder_file_sha256": rec["init"]["pretrain_encoder_file_sha256"],
                "matches_this_round_rebuild": info["checks"],
            },
            "recipe": {
                "key": C.RECIPE_KEY,
                "epochs": rec["epochs"],
                "updates": rec["updates"],
                "batch_size": rec["hyper"]["batch_size"],
                "lr": rec["hyper"]["lr"],
                "weight_decay": rec["hyper"]["weight_decay"],
                "grad_clip": rec["hyper"]["grad_clip"],
                "loss": rec["hyper"]["loss"],
                "soup_members": rec["soup_members"],
                "batch_order_matches": batch_ok,
            },
            "valid_replay": {
                "array_sha256_matches_history": True,
                "prediction_array_sha256": array_sha,
                "historical_array_sha256": prov["array_sha256_float32"],
                "historical_file_sha256": prov["historical_file_sha256"],
                "max_abs_pred_diff_vs_historical_npz": max_abs,
                "recorded_mae_from_confirmation_metrics": float(recorded_valid[f"seed{seed}"]),
                "regenerated_prediction_file": _stored(p3_pred_path),
                "seconds": secs,
                "labels_read": False,
            },
        }
        log(f"[audit] P3 seed {seed}: params={params} replay max|d|={max_abs:.3e}")

    conditions = {
        "all_soups_hash_match": all(
            p3[str(s)]["soup"]["sha256"] == cfg["audit_expectations"]["p3_soup_sha256"][str(s)] for s in C.SEEDS
        ),
        "all_413505_parameters": all(p3[str(s)]["parameter_exact_413505"] for s in C.SEEDS),
        "no_forbidden_modules": all(not p3[str(s)]["forbidden_modules"] for s in C.SEEDS),
        "all_inits_match_r3_record": all(
            all(p3[str(s)]["init"]["matches_this_round_rebuild"].values()) for s in C.SEEDS
        ),
        "all_batch_orders_match": all(p3[str(s)]["recipe"]["batch_order_matches"] for s in C.SEEDS),
        "p3_valid_arrays_bit_identical_to_history": all(
            p3[str(s)]["valid_replay"]["array_sha256_matches_history"] for s in C.SEEDS
        ),
        "p3_valid_replay_checked_against_local_npz": all(
            p3[str(s)]["valid_replay"]["max_abs_pred_diff_vs_historical_npz"] in (None, 0.0) for s in C.SEEDS
        ),
        "historical_valid_mae_recorded": all(
            abs(
                p3[str(s)]["valid_replay"]["recorded_mae_from_confirmation_metrics"]
                - cfg["audit_expectations"]["historical_valid_mae"][str(s)]
            )
            <= 1e-12
            for s in C.SEEDS
        ),
        "code_hash_matches_p3_revision": checks["code_hashes"]["all_match"],
        "train_y_matches_official": y_match,
    }
    checks["p3"] = p3
    checks["conditions"] = conditions
    checks["all_pass"] = all(conditions.values())

    # --- historical test predictions: hashes and recorded metrics only -------
    test_hashes = {}
    for seed in C.SEEDS:
        prov = C.historical_prediction(f"test_P3{seed}")
        test_hashes[str(seed)] = {
            "historical_path": prov["historical_path"],
            "historical_file_sha256": prov["historical_file_sha256"],
            "array_sha256": prov["array_sha256_float32"],
            "n": C.N_TEST,
            "historical_file_present_on_compute_checkout": (REPO / prov["historical_path"]).exists(),
            "array_regenerated_and_verified_at_test_predict_stage": True,
        }
    metrics = C.read_json(C.TERMINAL / "official_test_metrics.json")
    key_mae = {name: rec.get("mae") for name, rec in metrics.get("per_model", {}).items() if name in ("P30", "P31")}
    checks["historical_test"] = {
        "prediction_hashes": test_hashes,
        "recorded_mae": key_mae,
        "recorded_seed_mean_mae": metrics.get("seed_mean", {}).get("P3", {}).get("mae"),
        "test_labels_read": False,
        "note": "metrics copied from the historical terminal round; not recomputed here",
    }
    for seed in C.SEEDS:
        expected = C.load_config()["audit_expectations"]["historical_test_mae"][str(seed)]
        got = key_mae.get(f"P3{seed}")
        if got is None or abs(float(got) - float(expected)) > 1e-9:
            raise SystemExit(f"historical P3 test MAE record mismatch for seed {seed}: {got} != {expected}")
    checks["historical_test"]["mae_matches_config"] = True

    # --- official-test structure fingerprint (structures only) --------------
    from run_bjoc_pretraining_mechanism_terminal_test_v1 import _test_structure_fingerprint, load_test_dataset

    test_ds = load_test_dataset()
    fp = _test_structure_fingerprint(test_ds)
    expected_fp = cfg["audit_expectations"]["test_structure_fingerprint"]
    checks["test_structure"] = {
        "fingerprint": fp,
        "expected": expected_fp,
        "processed_pt_matches": fp["processed_pt_sha256"] == expected_fp["processed_pt_sha256"],
        "structure_sha256_matches": fp["structure_sha256"] == expected_fp["structure_sha256"],
        "labels_read": False,
    }
    checks["conditions"]["test_structure_fingerprint_matches"] = bool(
        checks["test_structure"]["processed_pt_matches"] and checks["test_structure"]["structure_sha256_matches"]
    )
    checks["all_pass"] = all(checks["conditions"].values())

    checks["environment"] = _env(device)
    C.write_once_json(out, checks)
    log(f"[audit] all_pass={checks['all_pass']} -> {out}")
    return 0 if checks["all_pass"] else 1


# ---------------------------------------------------------------------------
# CPU gates
# ---------------------------------------------------------------------------


def _synthetic_objs(n: int = 270) -> list[sro.MolObjects]:
    base = [o for _, o in R2A._synthetic_molecules()]
    return [base[i % len(base)] for i in range(n)]


def _batch(objs: Sequence[sro.MolObjects]) -> dict[str, Any]:
    return sro.batch_objects(list(objs))


def cmd_gates(args: argparse.Namespace) -> int:
    out = C.OUT / "cpu_gates_phase2b.json"
    if out.exists():
        raise SystemExit("gate record exists; refusing to rewrite")
    import torch

    gates: dict[str, Any] = {"protocol_id": C.PROTOCOL_ID, "stage": "gates", "created_utc": _now()}
    failures: list[str] = []

    def gate(name: str, passed: bool, detail: Any) -> None:
        gates[name] = {"pass": bool(passed), "detail": detail}
        if not passed:
            failures.append(name)

    # G1: per-seed rebuild parity with the historical P3 initialisation.
    rebuild = {}
    for arm in C.CANDIDATES:
        for seed in C.SEEDS:
            _, info = C.build_arm(arm, seed, "cpu")
            rebuild[f"{arm}_s{seed}"] = {
                "parameters": info["parameters"],
                "checks": info["checks"],
                "encoder_used_sha256": info["encoder_used_sha256"],
            }
    gate(
        "G1_per_seed_init_and_ssl_parity",
        all(all(v["checks"].values()) for v in rebuild.values())
        and all(v["parameters"] == C.EXPECTED_PARAMETERS[k.split("_")[0]] for k, v in rebuild.items()),
        rebuild,
    )

    # G2: Phase 2a F/M definitions replay their frozen Monitor predictions.
    fit_norm = C.read_json(R2A.C.BASELINE_DIR / "prep_fit.json")
    monitor_objs_path = R2A.C.BASELINE_DIR / "objects_monitor.pt"
    phase2a_soups = {f"{a}_s{s}": C.TRACK / f"results/phase2a/train_{a}_s{s}_soup.pt" for a in C.CANDIDATES for s in C.SEEDS}
    missing_baseline = [str(p) for p in [monitor_objs_path, *phase2a_soups.values()] if not p.exists()]
    if missing_baseline and not args.allow_missing_baseline:
        raise SystemExit(f"missing Phase 1e/2a replay artifacts: {missing_baseline}")
    replay = {}
    if missing_baseline:
        replay = {"skipped_local_only": missing_baseline}
    else:
        fit_objs, _ = R2A._objects(R2A.C.load_split().monitor, which="monitor")
        for arm in C.CANDIDATES:
            kind = C.ARM_KINDS[arm]
            for seed in C.SEEDS:
                path = C.TRACK / f"results/phase2a/monitor_pred_{arm}_s{seed}.npz"
                if not path.exists():
                    replay[f"{arm}_s{seed}"] = {"skipped": "no frozen phase2a prediction"}
                    continue
                _, model, _ = M.build_local_model(kind, seed=seed, norm=fit_norm, device="cpu")
                soup = torch.load(phase2a_soups[f"{arm}_s{seed}"], map_location="cpu", weights_only=False)
                model.load_state_dict(soup, strict=True)
                frozen = np.load(path)["pred"]
                pred, _ = C.predict(model, fit_objs, "cpu")
                replay[f"{arm}_s{seed}"] = {"max_abs_diff": float(np.abs(pred - frozen).max())}
    gate(
        "G2_phase2a_frozen_predictions_replay",
        True if missing_baseline else all(v.get("max_abs_diff", 9) <= 1e-5 for v in replay.values()),
        replay,
    )

    # G3: batch order over the official 10k reproduces the historical R3 hash.
    order = {}
    for seed in C.SEEDS:
        gen = torch.Generator().manual_seed(int(seed) + C.TRAIN_SHUFFLE_OFFSET)
        h = hashlib.sha256()
        for _ in range(C.EPOCHS):
            perm = torch.randperm(C.N_TRAIN, generator=gen).tolist()
            for i in range(0, C.N_TRAIN, C.BATCH_SIZE):
                h.update(np.ascontiguousarray(np.asarray(perm[i : i + C.BATCH_SIZE], dtype=np.int64)).tobytes())
        order[str(seed)] = h.hexdigest()
    expected_order = C.load_config()["audit_expectations"]["batch_order_sha256"]
    gate("G3_batch_order_matches_r3", all(order[str(s)] == expected_order[str(s)] for s in C.SEEDS), order)

    # G4: loop parity against the frozen Phase 2a loop (bitwise, tiny CPU run).
    objs = _synthetic_objs(270)
    y_small = np.linspace(-1.0, 1.0, len(objs))
    parity = {}
    ssl_fit = None
    if not R2A.C.SSL_STATE.exists():
        if not args.allow_missing_baseline:
            raise SystemExit(f"missing Phase 1e Fit-only SSL state {R2A.C.SSL_STATE}")
        parity = {"skipped_local_only": str(R2A.C.SSL_STATE)}
    else:
        ssl_fit, _ = R2A.C.load_ssl_state()
    for arm in (C.CANDIDATES if ssl_fit is not None else ()):
        kind = C.ARM_KINDS[arm]
        ref = R2A.train_arm(
            arm=arm,
            kind=kind,
            seed=0,
            train_objs=objs,
            rows=list(range(len(objs))),
            y_fit=y_small,
            norm=fit_norm,
            device=torch.device("cpu"),
            ssl_state=ssl_fit,
            smoke_epochs=2,
        )
        mine = _train_loop(
            arm=arm,
            kind=kind,
            seed=0,
            train_objs=objs,
            rows=list(range(len(objs))),
            y_fit=y_small,
            norm=fit_norm,
            device=torch.device("cpu"),
            ssl_state=ssl_fit,
            ssl_expected=None,
            smoke_epochs=2,
            save_soup=False,
        )
        max_abs = max(
            float((ref["soup_state"][k].float() - mine["soup_state"][k].float()).abs().max()) for k in ref["soup_state"]
        )
        parity[arm] = {
            "max_abs_state_diff": max_abs,
            "ref_soup_fit_mae": ref["soup_fit_mae"],
            "mine_soup_fit_mae": mine["soup_fit_mae"],
            "ref_updates": ref["updates"],
            "mine_updates": mine["updates"],
            "ref_batch_order": ref["batch_order"]["full_hash"],
            "mine_batch_order": mine["batch_order"]["full_hash"],
        }
    gate(
        "G4_loop_parity_with_phase2a",
        True
        if ssl_fit is None
        else all(
            v["max_abs_state_diff"] == 0.0 and v["ref_batch_order"] == v["mine_batch_order"]
            for v in parity.values()
        ),
        parity,
    )

    # G5: relabelling / edge-direction / batch-order invariance + F blindness.
    mols = [o for _, o in R2A._synthetic_molecules()]
    g_models = {}
    for arm in C.CANDIDATES:
        _, g_models[arm], _ = M.build_local_model(C.ARM_KINDS[arm], seed=0, norm=fit_norm, device="cpu")
        g_models[arm].eval()
    with torch.no_grad():
        relabel = {}
        for obj in mols:
            rng = np.random.default_rng(12345 + int(obj.mol_index))
            perm = rng.permutation(obj.n_atoms)
            relabelled = R2A._reindex_objects(obj, perm)
            b0, b1 = _batch([obj]), _batch([relabelled])
            relabel[int(obj.mol_index)] = {
                arm: abs(float(g_models[arm](b0)[0]) - float(g_models[arm](b1)[0])) for arm in C.CANDIDATES
            }
        swap = 0.0
        for obj in mols:
            pairs = [(int(u), int(v), int(t)) for (u, v), t in zip(obj.bond_index, obj.bond_type)]
            ei, ea = [], []
            for u, v, t in pairs:
                ei += [[v, u], [u, v]]
                ea += [t, t]
            atom_type = obj.atom_feat[:, : sro.ATOM_CATEGORIES].argmax(1)
            swapped = sro.build_mol_objects(obj.mol_index, atom_type, np.asarray(ei).T, np.asarray(ea))
            b0, b1 = _batch([obj]), _batch([swapped])
            for arm in C.CANDIDATES:
                swap = max(swap, abs(float(g_models[arm](b0)[0]) - float(g_models[arm](b1)[0])))
        order = [2, 0, 1]
        b_all = _batch(mols)
        b_perm = _batch([mols[i] for i in order])
        reorder = {
            arm: float((g_models[arm](b_all)[order] - g_models[arm](b_perm)).abs().max()) for arm in C.CANDIDATES
        }
        base_batch = _batch(mols)
        b_index = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base_batch.items()}
        b_index["bond_index"] = torch.roll(b_index["bond_index"], shifts=1, dims=0)
        b_feat = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base_batch.items()}
        b_feat["bond_feat"] = torch.roll(b_feat["bond_feat"], shifts=3, dims=0)
        sens = {}
        for arm in C.CANDIDATES:
            a0 = g_models[arm].encode_atoms(base_batch)
            sens[arm] = {
                "bond_index_roll": float((g_models[arm].encode_atoms(b_index) - a0).abs().max()),
                "bond_feat_roll": float((g_models[arm].encode_atoms(b_feat) - a0).abs().max()),
            }
    gate(
        "G5_permutation_edge_direction_batch_invariance",
        max(max(r.values()) for r in relabel.values()) <= 1e-5
        and swap <= 1e-5
        and max(reorder.values()) <= 1e-5,
        {"relabelling_max_abs": relabel, "edge_direction_swap_max_abs": swap, "batch_reorder_max_abs": reorder},
    )
    gate(
        "G6_F_adjacency_insensitive",
        sens["F"]["bond_index_roll"] <= 1e-7 and sens["F"]["bond_feat_roll"] <= 1e-7,
        sens["F"],
    )
    gate(
        "G7_M_adjacency_and_bond_feature_sensitive",
        sens["M"]["bond_index_roll"] > 1e-4 and sens["M"]["bond_feat_roll"] > 1e-4,
        sens["M"],
    )

    allowed = R2A.ALLOWED_BATCH_KEYS
    base_batch = _batch(mols)
    gate("G8_no_label_input", set(base_batch.keys()) <= allowed, sorted(base_batch.keys()))

    # G9: new parameters receive non-zero gradient; predictions are finite.
    grad = {}
    for arm in C.CANDIDATES:
        _, model, _ = M.build_local_model(C.ARM_KINDS[arm], seed=0, norm=fit_norm, device="cpu")
        y = torch.zeros(base_batch["n_graphs"], dtype=torch.float32)
        loss = torch.nn.functional.l1_loss(model(base_batch).view(-1), y)
        loss.backward()
        missing = [n for n, p in model.named_parameters() if n.startswith("local.") and (p.grad is None or float(p.grad.abs().sum()) == 0)]
        grad[arm] = {"zero_local_gradients": missing, "loss_finite": bool(torch.isfinite(loss))}
    gate("G9_local_gradients", all(not v["zero_local_gradients"] and v["loss_finite"] for v in grad.values()), grad)

    # G10: parameter inventory matches the frozen Phase 2a numbers.
    inventories = {}
    for arm in C.CANDIDATES:
        _, model, _ = M.build_local_model(C.ARM_KINDS[arm], seed=0, norm=fit_norm, device="cpu")
        inventories[arm] = {
            "actual": int(sum(p.numel() for p in model.parameters())),
            "expected": C.EXPECTED_PARAMETERS[arm],
            "local": M.local_parameters(C.ARM_KINDS[arm]),
        }
    gate(
        "G10_parameter_counts",
        all(v["actual"] == v["expected"] for v in inventories.values())
        and inventories["F"]["local"] == 49984
        and inventories["M"]["local"] == 50752,
        inventories,
    )

    gates["failures"] = failures
    gates["all_pass"] = not failures
    gates["environment"] = _env(torch.device("cpu"))
    C.write_once_json(out, gates)
    log(f"[gates] all_pass={gates['all_pass']} failures={failures} -> {out}")
    return 0 if not failures else 1


# ---------------------------------------------------------------------------
# training loop (line-for-line Phase 2a semantics; SSL check parameterised)
# ---------------------------------------------------------------------------


def _grad_group_norms(model: Any) -> dict[str, float]:
    groups: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        parts = name.split(".")
        group = parts[0]
        if group == "local":
            group = "local." + (parts[1] if parts[1] == "bond_proj" else ".".join(parts[1:3]))
        groups[group] = groups.get(group, 0.0) + float(p.grad.detach().pow(2).sum())
    return {k: math.sqrt(v) for k, v in sorted(groups.items())}


def _train_loop(
    *,
    arm: str,
    kind: str,
    seed: int,
    train_objs: Sequence[sro.MolObjects],
    rows: Sequence[int],
    y_fit: np.ndarray,
    norm: Mapping[str, Any],
    device: Any,
    ssl_state: Mapping[str, Mapping[str, Any]],
    ssl_expected: Mapping[str, Any] | None,
    smoke_epochs: int | None = None,
    save_soup: bool = False,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    recipe = bp.RECIPES[C.RECIPE_KEY]
    _, model, build_info = M.build_local_model(kind, seed=seed, norm=norm, device="cpu")
    n_params = M.parameter_audit(model)["actual_parameters"]
    expected_params = C.EXPECTED_PARAMETERS[arm]
    if n_params != expected_params:
        raise SystemExit(f"parameter audit failed: {n_params} != {expected_params}")

    enc_init = bp.encoder_group_hash(model)
    trunk_init = M.trunk_param_sha256(model)
    group_init = M.group_state_sha256(model)
    local_init = M.local_state_sha256(model.local)
    local_block_init = M.local_block_state_sha256(model.local)
    import bjoc_masked_pretraining_v1 as mp

    mp.load_encoder_state(model, ssl_state, "cpu")
    encoder_used = bp.encoder_group_hash(model)
    if ssl_expected is not None and encoder_used != ssl_expected["encoder_used_sha256"]:
        raise SystemExit(f"encoder hash mismatch after SSL load: {encoder_used}")
    model = model.to(device)

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
    order_hash = hashlib.sha256()
    lr_deviation_max = 0.0
    grad_norm_max = 0.0
    clip_events = 0
    total_mol = 0
    first_step_grads: dict[str, float] | None = None
    final_step_grads: dict[str, float] | None = None
    step_seconds: list[float] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        lr_first = lr_last = None
        lr_sum = 0.0
        t_ep = time.perf_counter()
        for indices in bp.shuffle_batches(n, C.BATCH_SIZE, gen):
            order_hash.update(np.ascontiguousarray(np.asarray(indices, dtype=np.int64)).tobytes())
            t_step = time.perf_counter()
            batch = bp._mv(sro.batch_objects([train_objs[i] for i in indices]), device)
            y = torch.as_tensor(y_fit[indices], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            if first_step_grads is None:
                first_step_grads = _grad_group_norms(model)
            if epoch == epochs:
                final_step_grads = _grad_group_norms(model)
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

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "arm": arm,
        "kind": kind,
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
        "build": build_info,
        "init_full_sha256": trunk_init,
        "init_group_state_sha256": group_init,
        "init_encoder_sha256": enc_init,
        "encoder_used_sha256": encoder_used,
        "local_init_sha256": local_init,
        "local_block_init_sha256": local_block_init,
        "parameter_audit": M.parameter_audit(model),
        "batch_order": {
            "full_hash": order_hash.hexdigest(),
            "seed": int(seed) + C.TRAIN_SHUFFLE_OFFSET,
        },
        "anomalies": {
            "grad_clip_events": int(clip_events),
            "grad_norm_max": grad_norm_max,
            "lr_schedule_deviation_max": lr_deviation_max,
        },
        "grad_group_norms_first_step": first_step_grads,
        "grad_group_norms_final_step": final_step_grads,
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
    if ssl_expected is not None:
        payload["ssl_record"] = dict(ssl_expected)
    return {**payload, "soup_state": soup_state}


# ---------------------------------------------------------------------------
# flops
# ---------------------------------------------------------------------------


def cmd_flops(args: argparse.Namespace) -> int:
    out = C.OUT / "flops_phase2b.json"
    if out.exists():
        raise SystemExit("flop record exists; refusing to rewrite")

    train_objs = C.load_objects(C.OBJECTS_TRAIN, C.load_config()["frozen_inputs"]["objects_train10k"]["sha256"])
    batch = _batch(train_objs[:128])
    results: dict[str, Any] = {"protocol_id": C.PROTOCOL_ID, "stage": "flops", "n_graphs": int(batch["n_graphs"])}
    for arm in ("P3", "F", "M"):
        _, model, _ = M.build_local_model(C.ARM_KINDS[arm], seed=0, norm=C.load_norm(), device="cpu")
        results[arm] = M.forward_gemm_flops(model, batch)
    base = results["P3"]["total"]
    for arm in ("F", "M"):
        results[arm]["ratio_vs_p3"] = results[arm]["total"] / base
    results["parameters"] = {arm: C.EXPECTED_PARAMETERS[arm] for arm in ("P3", "F", "M")}
    C.write_once_json(out, results)
    log(f"[flops] P3 {base/1e9:.3f}G F {results['F']['ratio_vs_p3']:.3f}x M {results['M']['ratio_vs_p3']:.3f}x")
    return 0


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def cmd_smoke(args: argparse.Namespace) -> int:
    out = C.OUT / "smoke_phase2b.json"
    if out.exists():
        raise SystemExit("smoke record exists; refusing to rewrite")
    device = _device(args.device)
    train_objs = C.load_objects(C.OBJECTS_TRAIN, C.load_config()["frozen_inputs"]["objects_train10k"]["sha256"])
    rows = list(range(1024))
    objs = train_objs[: len(rows)]
    y = C.load_train_y()[rows]
    norm = C.load_norm()
    results: dict[str, Any] = {"protocol_id": C.PROTOCOL_ID, "stage": "smoke", "rows": len(rows), "epochs": 2}
    for arm in C.CANDIDATES:
        state, ssl_rec = C.ssl_state(0)
        rec = _train_loop(
            arm=arm,
            kind=C.ARM_KINDS[arm],
            seed=0,
            train_objs=objs,
            rows=rows,
            y_fit=y,
            norm=norm,
            device=device,
            ssl_state=state,
            ssl_expected={"encoder_used_sha256": self_encoder_sha(arm, 0), "file_sha256": ssl_rec["sha256"]},
            smoke_epochs=2,
            save_soup=False,
        )
        rec.pop("soup_state")
        per_update = rec["step_seconds"]["mean"]
        results[arm] = {
            **rec,
            "projected_formal_step_seconds": per_update,
            "projected_formal_updates": C.UPDATES,
            "projected_formal_seconds": per_update * C.UPDATES,
        }
        log(f"[smoke] {arm}: step={per_update*1e3:.2f}ms projected={per_update*C.UPDATES/60:.1f}min")
    results["environment"] = _env(device)
    C.write_once_json(out, results)
    return 0


def self_encoder_sha(arm: str, seed: int) -> str:
    rec = C.p3_run(seed)["init"]
    return rec["encoder_used_sha256"]


# ---------------------------------------------------------------------------
# formal training
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    arm, seed = args.arm, int(args.seed)
    if arm not in C.CANDIDATES:
        raise SystemExit(f"unknown arm {arm!r}")
    if seed not in C.SEEDS:
        raise SystemExit(f"unknown seed {seed}")
    if _train_path(arm, seed).exists() or _soup_path(arm, seed).exists():
        raise SystemExit("arm already trained; retraining is not allowed")

    device = _device(args.device)
    train_objs = C.load_objects(C.OBJECTS_TRAIN, C.load_config()["frozen_inputs"]["objects_train10k"]["sha256"])
    rows = list(range(C.N_TRAIN))
    y = C.load_train_y()
    norm = C.load_norm()
    state, ssl_rec = C.ssl_state(seed)
    expected = C.load_config()["audit_expectations"]
    payload = _train_loop(
        arm=arm,
        kind=C.ARM_KINDS[arm],
        seed=seed,
        train_objs=train_objs,
        rows=rows,
        y_fit=y,
        norm=norm,
        device=device,
        ssl_state=state,
        ssl_expected={
            "encoder_used_sha256": expected["encoder_used_sha256"][str(seed)],
            "file_sha256": ssl_rec["sha256"],
        },
        smoke_epochs=None,
    )
    if payload["epochs"] != C.EPOCHS or payload["updates"] != C.UPDATES:
        raise SystemExit("formal run did not use the frozen epoch budget")
    if payload["batch_order"]["full_hash"] != expected["batch_order_sha256"][str(seed)]:
        raise SystemExit(
            f"batch order mismatch: {payload['batch_order']['full_hash']} != "
            f"{expected['batch_order_sha256'][str(seed)]}"
        )
    soup = payload.pop("soup_state")
    import torch

    torch.save(soup, _soup_path(arm, seed))
    record = dict(payload)
    record["soup_path"] = _stored(_soup_path(arm, seed))
    record["soup_sha256"] = C.sha256_file(_soup_path(arm, seed))
    record["train_objects_sha256"] = C.sha256_file(C.OBJECTS_TRAIN)
    record["train_rows"] = {"n": C.N_TRAIN, "sha256": C.sha256_array(np.arange(C.N_TRAIN), "int64")}
    record["train_y_sha256"] = C.sha256_array(y, "float64")
    record["ssl_record"] = {**ssl_rec, "expected_encoder_used_sha256": expected["encoder_used_sha256"][str(seed)]}
    record["environment"] = _env(device)
    C.write_once_json(_train_path(arm, seed), record)
    log(
        f"[train {arm} s{seed}] params {record['parameter_audit']['actual_parameters']} "
        f"soup fit {record['soup_fit_mae']:.6f} wall {record['wall_seconds']:.0f}s "
        f"batch_order_ok=True"
    )
    return 0


# ---------------------------------------------------------------------------
# official-valid prediction + freeze
# ---------------------------------------------------------------------------


def cmd_predict_valid(args: argparse.Namespace) -> int:
    arm, seed = args.arm, int(args.seed)
    if arm not in C.CANDIDATES:
        raise SystemExit(f"unknown arm {arm!r}")
    freeze_path = _valid_freeze_path(arm, seed)
    if freeze_path.exists():
        raise SystemExit("valid predictions already frozen; refusing to rewrite")
    if not _train_path(arm, seed).exists() or not _soup_path(arm, seed).exists():
        raise SystemExit("arm is not trained")
    record = C.read_json(_train_path(arm, seed))
    if record["smoke"]:
        raise SystemExit("smoke run cannot be used for a frozen prediction")
    if C.sha256_file(_soup_path(arm, seed)) != record["soup_sha256"]:
        raise SystemExit("soup changed after training")

    device = _device(args.device)
    valid_objs = C.load_objects(C.OBJECTS_VALID, C.load_config()["frozen_inputs"]["objects_valid"]["sha256"])
    model, info = C.build_arm(arm, seed, device)
    soup = C.load_soup(model, _soup_path(arm, seed))
    pred, seconds = C.predict(model, valid_objs, device)
    if not np.isfinite(pred).all():
        raise SystemExit("non-finite valid prediction")
    np.savez_compressed(_valid_pred_path(arm, seed), row=np.arange(C.N_VALID, dtype=np.int64), pred=pred.astype(np.float32))
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "valid_predict_freeze",
        "arm": arm,
        "seed": seed,
        "prediction_file": _stored(_valid_pred_path(arm, seed)),
        "prediction_sha256": C.sha256_file(_valid_pred_path(arm, seed)),
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
    log(f"[predict-valid {arm} s{seed}] MAE<frozen> sha {payload['prediction_sha256'][:12]} {seconds:.3f}s")
    return 0


# ---------------------------------------------------------------------------
# official-valid scoring + frozen promotion rule
# ---------------------------------------------------------------------------


def _frozen_valid_predictions() -> dict[str, dict[int, np.ndarray]]:
    preds: dict[str, dict[int, np.ndarray]] = {"P3": {}, "F": {}, "M": {}}
    for seed in C.SEEDS:
        freeze = C.OUT / f"p3_valid_freeze_s{seed}.json"
        if not freeze.exists():
            raise SystemExit(f"missing regenerated P3 valid freeze {freeze}; run the audit first")
        rec = C.read_json(freeze)
        path = REPO / rec["prediction_file"]
        if C.sha256_file(path) != rec["prediction_sha256"]:
            raise SystemExit("regenerated P3 valid prediction changed after freeze")
        with np.load(path) as z:
            pred = np.asarray(z["pred"], dtype=np.float64)
        if C.sha256_array(pred) != rec["prediction_array_sha256"]:
            raise SystemExit("regenerated P3 valid prediction array hash mismatch")
        if pred.shape != (C.N_VALID,):
            raise SystemExit(f"unexpected P3 valid prediction shape {pred.shape}")
        preds["P3"][seed] = pred
        for arm in C.CANDIDATES:
            freeze = _valid_freeze_path(arm, seed)
            if not freeze.exists():
                raise SystemExit(f"missing frozen valid prediction {freeze}")
            rec = C.read_json(freeze)
            path = REPO / rec["prediction_file"]
            if C.sha256_file(path) != rec["prediction_sha256"]:
                raise SystemExit(f"valid prediction changed after freeze: {path}")
            with np.load(path) as z:
                pred = np.asarray(z["pred"], dtype=np.float64)
            if pred.shape != (C.N_VALID,):
                raise SystemExit(f"unexpected valid prediction shape {pred.shape}")
            if C.sha256_array(pred) != rec["prediction_array_sha256"]:
                raise SystemExit(f"valid prediction array hash mismatch: {path}")
            preds[arm][seed] = pred
    return preds


def _candidate_eval(
    cand: str,
    preds: Mapping[str, Mapping[int, np.ndarray]],
    y: np.ndarray,
    masks: Mapping[str, np.ndarray],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    rule = cfg["promotion_rule"]
    p3 = preds["P3"]
    means = {k: float(np.mean([np.abs(preds[k][s] - y).mean() for s in C.SEEDS])) for k in preds}
    per_seed_gain = {
        str(s): float(np.abs(p3[s] - y).mean() - np.abs(preds[cand][s] - y).mean()) for s in C.SEEDS
    }
    mean_gain = float(np.mean(list(per_seed_gain.values())))
    d = C.seed_averaged_delta(preds[cand], p3, y)
    fav_order = np.argsort(-d)
    p3_err_order = C.error_rank(p3, y)
    drop_top1 = C.top_removed_gain(d, fav_order[:1])
    drop_top10 = C.top_removed_gain(d, p3_err_order[:10])
    complement = masks["penalty_complement"]
    complement_gain = float(d[complement].mean())
    checks = {
        "mean_gain_ge_0.003": mean_gain >= float(rule["practical_gain_mean_mae"]),
        "both_seeds_positive": all(v > 0 for v in per_seed_gain.values()),
        "drop_top1_favourable_ge_0.002": drop_top1 >= float(rule["robust_gain_after_drop_top1_favourable"]),
        "drop_top10_p3_error_ge_0.002": drop_top10 >= float(rule["robust_gain_after_drop_top10_p3_error"]),
        "penalty_complement_gain_positive": complement_gain > 0,
    }
    return {
        "candidate": cand,
        "mae_seed_mean": means[cand],
        "p3_mae_seed_mean": means["P3"],
        "per_seed_gain": per_seed_gain,
        "mean_gain": mean_gain,
        "bootstrap": C.paired_bootstrap(d),
        "drop_top1_favourable_gain": drop_top1,
        "drop_top10_p3_error_gain": drop_top10,
        "penalty_complement_gain": complement_gain,
        "checks": checks,
        "qualifies": all(checks.values()),
    }


def _select_finalist(evals: Mapping[str, Mapping[str, Any]], mp: Mapping[str, Any]) -> tuple[str | None, str]:
    """Frozen §7.1 selection: at most one finalist, never both."""
    f_ok, m_ok = bool(evals["F"]["qualifies"]), bool(evals["M"]["qualifies"])
    if f_ok and m_ok:
        if mp["separated"]:
            return "M", "both qualified; M separated from F"
        return "F", "both qualified; M not separated -> simpler F"
    if f_ok:
        return "F", "only F qualified"
    if m_ok:
        return "M", "only M qualified"
    return None, "no candidate reached the frozen practical band"


def cmd_score_valid(args: argparse.Namespace) -> int:
    out_metrics = C.OUT / "valid_metrics.json"
    out_decision = C.OUT / "valid_decision.json"
    if out_decision.exists():
        raise SystemExit("valid decision exists; refusing to rewrite")

    cfg = C.load_config()
    preds = _frozen_valid_predictions()

    # combined freeze is written BEFORE the label read
    combined = C.OUT / "valid_prediction_freeze.json"
    if not combined.exists():
        entries = {}
        for key, by_seed in preds.items():
            for seed, pred in by_seed.items():
                entries[f"{key}{seed}"] = {
                    "array_sha256": C.sha256_array(pred),
                    "source": "historical_frozen_regenerated_from_soup" if key == "P3" else "phase2b_frozen",
                }
        C.write_once_json(
            combined,
            {
                "protocol_id": C.PROTOCOL_ID,
                "stage": "valid_prediction_freeze_combined",
                "entries": entries,
                "n": len(entries),
                "valid_labels_read_at_freeze": False,
                "created_utc": _now(),
            },
        )
    else:
        frozen = C.read_json(combined)
        for key, by_seed in preds.items():
            for seed, pred in by_seed.items():
                if frozen["entries"][f"{key}{seed}"]["array_sha256"] != C.sha256_array(pred):
                    raise SystemExit("prediction array changed since the combined freeze")

    y = C.load_valid_y()
    lock = C.OUT / "valid_read.lock"
    if lock.exists():
        prior = C.read_json(lock)
        read_count = int(prior.get("read_count", 1)) + 1
    else:
        read_count = 1
    C.write_json(
        lock,
        {
            "protocol_id": C.PROTOCOL_ID,
            "valid_labels_read": True,
            "read_count": read_count,
            "y_source": "official PyG ZINC subset=True split=val; historical array hash verified",
            "historical_y_path": cfg["label_provenance"]["valid_y"]["historical_path"],
            "y_sha256_float32": C.sha256_array(y),
            "y_sha256_float64": C.sha256_array(y, "float64"),
            "prediction_freeze_sha256": C.sha256_file(combined),
            "read_utc": _now(),
            "reread": read_count > 1,
        },
    )

    valid_objs = C.load_objects(C.OBJECTS_VALID, cfg["frozen_inputs"]["objects_valid"]["sha256"])
    longest, penalty = C.cycle_proxy(valid_objs)
    masks = {
        "all": np.ones(C.N_VALID, dtype=bool),
        "ordinary_y_gt_minus3": y > C.TAIL_THRESHOLD,
        "tail_y_le_minus3": y <= C.TAIL_THRESHOLD,
        "penalty_proxy": penalty,
        "penalty_complement": ~penalty,
        "without_extreme_valid172": np.arange(C.N_VALID) != C.EXTREME_VALID_ROW,
    }

    metrics = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "valid_score",
        "created_utc": _now(),
        "n": C.N_VALID,
        "primary_metric": "raw-y MAE, seed-mean of two single-model seeds",
        "labels": {
            "source": "official PyG ZINC subset=True split=val; historical array hash verified",
            "historical_y_path": cfg["label_provenance"]["valid_y"]["historical_path"],
            "array_sha256_float32": C.sha256_array(y),
            "array_sha256_float64": C.sha256_array(y, "float64"),
            "read_count_this_round": read_count,
        },
        "models": C.metric_block(preds, y),
        "subgroups": C.subgroup_metrics(preds, y, masks),
        "masks": {
            name: {
                "n": int(mask.sum()),
                "rows_sha256": C.sha256_array(np.flatnonzero(mask), "int64"),
            }
            for name, mask in masks.items()
        },
        "penalty_proxy": {
            "definition": "max(len(c) for c in nx.cycle_basis(G)) > 6",
            "longest_cycle_histogram": {
                str(int(k)): int((longest == k).sum()) for k in np.unique(longest)
            },
            "n_penalised": int(penalty.sum()),
            "order_sensitive_proxy": True,
            "not_a_model_input": True,
        },
        "extreme_row": {
            "row": C.EXTREME_VALID_ROW,
            "y": float(y[C.EXTREME_VALID_ROW]),
        },
    }

    evals = {cand: _candidate_eval(cand, preds, y, masks, cfg) for cand in C.CANDIDATES}
    # MSE tail diagnostic (no threshold gate)
    for cand in C.CANDIDATES:
        d_mse = C.mse_delta(preds[cand], preds["P3"], y)
        top = int(np.argmax(np.abs(d_mse)))
        metrics[f"mse_delta_{cand}"] = {
            "bootstrap": C.paired_bootstrap(d_mse),
            "top_abs_contributor_row": top,
            "top_abs_contributor_share": float(abs(d_mse[top]) / max(abs(d_mse).sum(), 1e-12)),
            "p3_error_top1_share_of_abs_delta": float(abs(d_mse[top]) / max(np.abs(d_mse).sum(), 1e-12)),
        }

    # Gain_MP uses F as reference and M as candidate.
    d_mp = C.seed_averaged_delta(preds["M"], preds["F"], y)
    mp = {
        "definition": "MAE(F) - MAE(M); positive = M better",
        "per_seed_gain": {
            str(s): float(np.abs(preds["F"][s] - y).mean() - np.abs(preds["M"][s] - y).mean()) for s in C.SEEDS
        },
        "bootstrap": C.paired_bootstrap(d_mp),
        "penalty_complement_gain": float(d_mp[masks["penalty_complement"]].mean()),
    }
    mp["mean_gain"] = float(np.mean(list(mp["per_seed_gain"].values())))
    rule = cfg["promotion_rule"]
    mp["checks"] = {
        "mean_gain_ge_0.002": mp["mean_gain"] >= float(rule["mp_separation_mean_mae"]),
        "both_seeds_positive": all(v > 0 for v in mp["per_seed_gain"].values()),
        "penalty_complement_positive": mp["penalty_complement_gain"] > 0,
    }
    mp["separated"] = all(mp["checks"].values())

    finalist, reason = _select_finalist(evals, mp)

    if finalist is None:
        performance = ["VALID_GAIN_NOT_REPLICATED"]
        mechanism: list[str] = []
        mechanism_note = "no increment to attribute; mechanism labels are not estimable"
    else:
        performance = ["VALID_GAIN_REPLICATED_ON_OFFICIAL_VALID"]
        if finalist == "M" and mp["separated"]:
            mechanism = ["MP_INCREMENT_SUPPORTED"]
            mechanism_note = "M separated from F on official valid"
        else:
            mechanism = ["CAPACITY_GAIN_WITHOUT_MP_SPECIFICITY"]
            mechanism_note = "finalist is capacity-only or M is not separated from F"

    decision = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "valid_decision",
        "created_utc": _now(),
        "rule": cfg["promotion_rule"],
        "candidates": evals,
        "gain_mp": mp,
        "finalist": finalist,
        "selection_reason": reason,
        "promotion_authorized": finalist is not None,
        "performance_verdicts": performance,
        "mechanism_verdicts": mechanism,
        "mechanism_note": mechanism_note,
        "no_test_if_not_promoted": finalist is None,
        "valid_labels_read_count": read_count,
        "git_commit": _git_commit(),
    }
    C.write_once_json(out_metrics, metrics)
    C.write_once_json(out_decision, decision)
    log(f"[score-valid] finalist={finalist} ({reason})")
    for cand in C.CANDIDATES:
        e = evals[cand]
        log(
            f"  {cand}: gain={e['mean_gain']:+.6f} per-seed={e['per_seed_gain']} "
            f"qualifies={e['qualifies']} checks={e['checks']}"
        )
    log(f"  Gain_MP={mp['mean_gain']:+.6f} separated={mp['separated']}")
    return 0


# ---------------------------------------------------------------------------
# official-test prediction (only after promotion) + scoring
# ---------------------------------------------------------------------------


class _LabelGuard:
    """Read-only dataset wrapper that raises on any label-field access."""

    __slots__ = ("_d",)

    def __init__(self, dataset: Any):
        object.__setattr__(self, "_d", dataset)

    def __len__(self) -> int:
        return len(self._d)

    def __getitem__(self, i: int) -> Any:
        d = self._d[int(i)]

        class _Struct:
            __slots__ = ("x", "edge_index", "edge_attr")

            def __init__(self, data: Any):
                self.x = data.x
                self.edge_index = data.edge_index
                self.edge_attr = data.edge_attr

        return _Struct(d)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"label access forbidden at test-prediction time: {name}")


def _test_fingerprint(ds: Any) -> dict[str, Any]:
    digest = hashlib.sha256()
    for i in range(C.N_TEST):
        digest.update(np.ascontiguousarray(ds[i].x.numpy(), dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(ds[i].edge_index.numpy(), dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(ds[i].edge_attr.numpy(), dtype=np.int64).tobytes())
    processed = REPO / "data/ZINC/subset/processed/test.pt"
    return {
        "n_molecules": C.N_TEST,
        "processed_pt": _stored(processed),
        "processed_pt_sha256": C.sha256_file(processed),
        "structure_sha256": digest.hexdigest(),
    }


def _finalist_from_decision() -> tuple[str, dict[str, Any]]:
    path = C.OUT / "valid_decision.json"
    if not path.exists():
        raise SystemExit("no valid decision: run score-valid first")
    decision = C.read_json(path)
    finalist = decision.get("finalist")
    if not finalist or not decision.get("promotion_authorized"):
        raise SystemExit("no finalist was promoted on official valid; official test must not be touched")
    return finalist, decision


def cmd_test_predict(args: argparse.Namespace) -> int:
    finalist, decision = _finalist_from_decision()
    if args.arm and args.arm != finalist:
        raise SystemExit(f"requested arm {args.arm} is not the frozen finalist {finalist}")
    out = C.OUT / "test_structure_audit.json"
    if out.exists():
        raise SystemExit("test structure audit exists; refusing to rewrite")
    for seed in C.SEEDS:
        if _test_freeze_path(finalist, seed).exists():
            raise SystemExit("test predictions already frozen; refusing to rewrite")

    cfg = C.load_config()
    from run_bjoc_pretraining_mechanism_terminal_test_v1 import load_test_dataset

    ds = _LabelGuard(load_test_dataset())
    fp = _test_fingerprint(ds)
    expected = cfg["audit_expectations"]["test_structure_fingerprint"]
    checks = {
        "processed_pt_matches": fp["processed_pt_sha256"] == expected["processed_pt_sha256"],
        "structure_sha256_matches": fp["structure_sha256"] == expected["structure_sha256"],
    }
    if not all(checks.values()):
        raise SystemExit(f"official-test structure fingerprint changed: {checks}")

    objs = [
        sro.build_mol_objects(int(i), ds[i].x.numpy(), ds[i].edge_index.numpy(), ds[i].edge_attr.numpy())
        for i in range(C.N_TEST)
    ]
    # Regenerate the historical P3 test predictions from the frozen soups (test
    # structures only, no labels) and require bit-identical historical arrays.
    # This stage is only reachable after promotion.
    p3_entries: dict[str, dict[str, Any]] = {}
    for seed in C.SEEDS:
        freeze_path = C.OUT / f"test_p3_freeze_s{seed}.json"
        if freeze_path.exists():
            raise SystemExit("P3 test predictions already frozen; refusing to rewrite")
        model_p3, info_p3 = C.build_p3(seed, "cpu")
        p3_pred, p3_seconds = C.predict(model_p3, objs, "cpu")
        prov = C.historical_prediction(f"test_P3{seed}")
        array_sha = C.sha256_array(p3_pred)
        if array_sha != prov["array_sha256_float32"]:
            raise SystemExit(
                f"regenerated P3 seed {seed} test array does not match the historical hash: "
                f"{array_sha} != {prov['array_sha256_float32']}"
            )
        pred_path = C.OUT / f"test_p3_pred_s{seed}.npz"
        np.savez_compressed(pred_path, row=np.arange(C.N_TEST, dtype=np.int64), pred=p3_pred.astype(np.float32))
        payload = {
            "protocol_id": C.PROTOCOL_ID,
            "stage": "test_p3_regenerated_from_frozen_soup",
            "arm": "P3",
            "seed": seed,
            "prediction_file": _stored(pred_path),
            "prediction_sha256": C.sha256_file(pred_path),
            "prediction_array_sha256": array_sha,
            "soup": info_p3["soup"],
            "historical_file_sha256": prov["historical_file_sha256"],
            "historical_array_sha256": prov["array_sha256_float32"],
            "bit_identical_to_historical_array": True,
            "n": C.N_TEST,
            "labels_read": False,
            "official_test_labels_read": False,
            "inference_seconds": p3_seconds,
            "created_utc": _now(),
        }
        C.write_once_json(freeze_path, payload)
        p3_entries[str(seed)] = payload
        log(f"[test-predict P3 s{seed}] sha {payload['prediction_sha256'][:12]} {p3_seconds:.3f}s")

    device = _device(args.device)
    entries = {}
    for seed in C.SEEDS:
        model, info = C.build_arm(finalist, seed, device)
        soup = C.load_soup(model, _soup_path(finalist, seed))
        pred, seconds = C.predict(model, objs, device)
        if not np.isfinite(pred).all():
            raise SystemExit("non-finite test prediction")
        np.savez_compressed(_test_pred_path(finalist, seed), row=np.arange(C.N_TEST, dtype=np.int64), pred=pred.astype(np.float32))
        payload = {
            "protocol_id": C.PROTOCOL_ID,
            "stage": "test_predict_freeze",
            "arm": finalist,
            "seed": seed,
            "prediction_file": _stored(_test_pred_path(finalist, seed)),
            "prediction_sha256": C.sha256_file(_test_pred_path(finalist, seed)),
            "prediction_array_sha256": C.sha256_array(pred),
            "soup": soup,
            "parameters": info["parameters"],
            "inference_seconds": seconds,
            "n": C.N_TEST,
            "labels_read": False,
            "official_test_labels_read": False,
            "trained_on_official_train_10000_only": True,
            "environment": _env(device),
            "created_utc": _now(),
        }
        C.write_once_json(_test_freeze_path(finalist, seed), payload)
        entries[str(seed)] = payload["prediction_sha256"]
        log(f"[test-predict {finalist} s{seed}] sha {payload['prediction_sha256'][:12]} {seconds:.3f}s")
    C.write_once_json(
        out,
        {
            "protocol_id": C.PROTOCOL_ID,
            "stage": "test_structure_audit_and_frozen_predictions_before_label_read",
            "fingerprint": fp,
            "checks": checks,
            "finalist": finalist,
            "p3_prediction_sha256": {s: p3_entries[s]["prediction_sha256"] for s in p3_entries},
            "finalist_prediction_sha256": entries,
            "valid_decision_sha256": C.sha256_file(C.OUT / "valid_decision.json"),
            "labels_read": False,
            "created_utc": _now(),
        },
    )
    return 0


def cmd_score_test(args: argparse.Namespace) -> int:
    finalist, decision = _finalist_from_decision()
    out_metrics = C.OUT / "test_metrics.json"
    out_decision = C.OUT / "test_decision.json"
    if out_decision.exists():
        raise SystemExit("test decision exists; refusing to rewrite")
    cfg = C.load_config()
    preds: dict[str, dict[int, np.ndarray]] = {"P3": {}, finalist: {}}
    for seed in C.SEEDS:
        p3_freeze = C.read_json(C.OUT / f"test_p3_freeze_s{seed}.json")
        p3_path = REPO / p3_freeze["prediction_file"]
        if C.sha256_file(p3_path) != p3_freeze["prediction_sha256"]:
            raise SystemExit("regenerated P3 test prediction changed after freeze")
        with np.load(p3_path) as z:
            p3_pred = np.asarray(z["pred"], dtype=np.float64)
        if C.sha256_array(p3_pred) != p3_freeze["prediction_array_sha256"]:
            raise SystemExit("regenerated P3 test prediction array hash mismatch")
        if p3_pred.shape != (C.N_TEST,):
            raise SystemExit(f"unexpected P3 test prediction shape {p3_pred.shape}")
        preds["P3"][seed] = p3_pred
        freeze = C.read_json(_test_freeze_path(finalist, seed))
        path = REPO / freeze["prediction_file"]
        if C.sha256_file(path) != freeze["prediction_sha256"]:
            raise SystemExit("test prediction changed after freeze")
        with np.load(path) as z:
            pred = np.asarray(z["pred"], dtype=np.float64)
        if C.sha256_array(pred) != freeze["prediction_array_sha256"]:
            raise SystemExit("test prediction array hash mismatch")
        preds[finalist][seed] = pred

    combined = C.OUT / "test_prediction_freeze.json"
    if not combined.exists():
        C.write_once_json(
            combined,
            {
                "protocol_id": C.PROTOCOL_ID,
                "stage": "test_prediction_freeze_combined",
                "finalist": finalist,
                "entries": {
                    f"{k}{s}": {
                        "array_sha256": C.sha256_array(preds[k][s]),
                        "source": "historical_frozen_regenerated_from_soup" if k == "P3" else "phase2b_frozen",
                    }
                    for k in preds
                    for s in C.SEEDS
                },
                "official_test_labels_read_at_freeze": False,
                "created_utc": _now(),
            },
        )
    y = C.load_test_y()
    lock = C.OUT / "test_read.lock"
    if lock.exists():
        prior = C.read_json(lock)
        read_count = int(prior.get("read_count", 1)) + 1
    else:
        read_count = 1
    C.write_json(
        lock,
        {
            "protocol_id": C.PROTOCOL_ID,
            "official_test_labels_read": True,
            "read_count": read_count,
            "y_source": "official PyG ZINC subset=True split=test; historical array hash verified",
            "historical_y_path": cfg["label_provenance"]["test_y"]["historical_path"],
            "y_sha256_float32": C.sha256_array(y),
            "y_sha256_float64": C.sha256_array(y, "float64"),
            "prediction_freeze_sha256": C.sha256_file(combined),
            "read_utc": _now(),
            "reread": read_count > 1,
        },
    )

    d = C.seed_averaged_delta(preds[finalist], preds["P3"], y)
    per_seed_gain = {
        str(s): float(np.abs(preds["P3"][s] - y).mean() - np.abs(preds[finalist][s] - y).mean())
        for s in C.SEEDS
    }
    mean_gain = float(np.mean(list(per_seed_gain.values())))
    boot = C.paired_bootstrap(d)
    rule = cfg["test_rule"]
    checks = {
        "mean_gain_ge_0.003": mean_gain >= float(rule["seed_mean_gain_min"]),
        "both_seeds_positive": all(v > 0 for v in per_seed_gain.values()),
        "bootstrap_ci95_low_gt_0": boot["ci95_low"] > float(rule["bootstrap_ci95_lower_must_exceed"]),
    }
    confirmed = all(checks.values())

    masks = {
        "all": np.ones(C.N_TEST, dtype=bool),
        "ordinary_y_gt_minus3": y > C.TAIL_THRESHOLD,
        "tail_y_le_minus3": y <= C.TAIL_THRESHOLD,
    }
    d_mse = C.mse_delta(preds[finalist], preds["P3"], y)
    top = int(np.argmax(np.abs(d_mse)))
    metrics = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "test_score",
        "created_utc": _now(),
        "n": C.N_TEST,
        "finalist": finalist,
        "primary_metric": "raw-y MAE on official ZINC-12k test 1000, seed-mean of two single models",
        "labels": {
            "source": "official PyG ZINC subset=True split=test; historical array hash verified",
            "historical_y_path": cfg["label_provenance"]["test_y"]["historical_path"],
            "array_sha256_float32": C.sha256_array(y),
            "array_sha256_float64": C.sha256_array(y, "float64"),
            "read_count_this_round": read_count,
            "historically_exposed": True,
        },
        "models": C.metric_block(preds, y),
        "subgroups": C.subgroup_metrics(preds, y, masks),
        "paired_vs_p3": {
            "per_seed_gain": per_seed_gain,
            "mean_gain": mean_gain,
            "bootstrap": boot,
            "different_molecules_improved_fraction": boot["better_fraction"],
        },
        "mse_tail": {
            "bootstrap": C.paired_bootstrap(d_mse),
            "top_abs_contributor_row": top,
            "top_abs_contributor_share": float(abs(d_mse[top]) / max(np.abs(d_mse).sum(), 1e-12)),
        },
        "test_is_first_blind_test": False,
        "test_read_used_for_selection": False,
    }
    decision_payload = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "test_decision",
        "finalist": finalist,
        "rule": cfg["test_rule"],
        "checks": checks,
        "mean_gain": mean_gain,
        "per_seed_gain": per_seed_gain,
        "bootstrap": boot,
        "performance_verdicts": ["STANDARD_ZINC_GAIN_CONFIRMED"] if confirmed else ["NO_PRACTICAL_ADVANTAGE"],
        "baseline_p3_seed_mean_mae": float(np.mean([np.abs(preds["P3"][s] - y).mean() for s in C.SEEDS])),
        "finalist_seed_mean_mae": float(np.mean([np.abs(preds[finalist][s] - y).mean() for s in C.SEEDS])),
        "git_commit": _git_commit(),
        "created_utc": _now(),
        "no_test_driven_followup": True,
    }
    C.write_once_json(out_metrics, metrics)
    C.write_once_json(out_decision, decision_payload)
    log(
        f"[score-test] {finalist}: MAE={decision_payload['finalist_seed_mean_mae']:.6f} "
        f"P3={decision_payload['baseline_p3_seed_mean_mae']:.6f} gain={mean_gain:+.6f} "
        f"confirmed={confirmed}"
    )
    return 0


# ---------------------------------------------------------------------------
# convenience wrappers (sequential stages in one remote job)
# ---------------------------------------------------------------------------


def cmd_prep(args: argparse.Namespace) -> int:
    """audit -> gates -> flops, in one CPU job, refusing to overwrite."""
    sub = argparse.Namespace(device=args.device, allow_missing_baseline=False)
    cmd_audit(sub)
    cmd_gates(sub)
    cmd_flops(sub)
    return 0


def cmd_run_arm(args: argparse.Namespace) -> int:
    """train + freeze valid predictions for both seeds of one arm."""
    for seed in C.SEEDS:
        sub = argparse.Namespace(arm=args.arm, seed=seed, device=args.device)
        if not (_train_path(args.arm, seed).exists() and _soup_path(args.arm, seed).exists()):
            cmd_train(sub)
        if not _valid_freeze_path(args.arm, seed).exists():
            cmd_predict_valid(sub)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(C.OUT), help="results directory (defaults to results/phase2b)")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit")
    a.add_argument("--device", default="cpu")

    g = sub.add_parser("gates")
    g.add_argument("--device", default="cpu")
    g.add_argument(
        "--allow-missing-baseline",
        action="store_true",
        help="local debugging only: skip checks needing gitignored Phase 1e artifacts",
    )

    f = sub.add_parser("flops")
    f.add_argument("--device", default="cpu")

    s = sub.add_parser("smoke")
    s.add_argument("--device", default="cuda")

    t = sub.add_parser("train")
    t.add_argument("--arm", required=True, choices=sorted(C.CANDIDATES))
    t.add_argument("--seed", required=True, type=int, choices=list(C.SEEDS))
    t.add_argument("--device", default="cuda")

    pv = sub.add_parser("predict-valid")
    pv.add_argument("--arm", required=True, choices=sorted(C.CANDIDATES))
    pv.add_argument("--seed", required=True, type=int, choices=list(C.SEEDS))
    pv.add_argument("--device", default="cuda")

    sv = sub.add_parser("score-valid")
    sv.add_argument("--device", default="cpu")

    pp = sub.add_parser("prep")
    pp.add_argument("--device", default="cpu")

    ra = sub.add_parser("run-arm")
    ra.add_argument("--arm", required=True, choices=sorted(C.CANDIDATES))
    ra.add_argument("--device", default="cuda")

    tp = sub.add_parser("test-predict")
    tp.add_argument("--arm", default=None, choices=sorted(C.CANDIDATES))
    tp.add_argument("--device", default="cuda")

    st = sub.add_parser("score-test")
    st.add_argument("--device", default="cpu")
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
        "run-arm": cmd_run_arm,
        "smoke": cmd_smoke,
        "train": cmd_train,
        "predict-valid": cmd_predict_valid,
        "score-valid": cmd_score_valid,
        "test-predict": cmd_test_predict,
        "score-test": cmd_score_test,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
