"""ZINC structural-token attention v1 — runner.

Protocol: ``results/zinc_structural_token_attention_v1/PROTOCOL.md``
Round id: ``zinc-structural-token-attention-v1``

Stages
------
``audit``       read-only hash/init/encoder/P3-replay/regime audit (no labels)
``gates``       focused CPU correctness gates
``flops``       fixed-batch forward GEMM FLOPs and token/pair counts
``prep``        audit + gates + flops
``smoke``       2-epoch timing on 1,024 train rows for S/T/R (no valid labels)
``train``       one formal arm/seed, R3, 360 epochs
``run-seed``    sequential formal arms for one seed (used on one GPU)
``predict-all`` six label-free valid predictions + per-file hashes
``mechanism``   label-free response statistics for the six soups
``freeze``      predict-all + mechanism + combined pre-score freeze
``score``       the single official-valid label read and decision
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for _p in (
    HERE,
    REPO / "tracks/structural-sufficiency/code/phase2b",
    REPO / "tracks/structure-contribution/code",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sta_common as C  # noqa: E402
import sta_data as D  # noqa: E402
import sta_models as M  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

import bjoc_perfmech_long_v1 as bp  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stored(path: Path) -> str:
    try:
        return str(Path(path).relative_to(REPO))
    except ValueError:
        return str(path)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _device(name: str):
    import torch

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False")
    return device


def _env(device: Any) -> dict[str, Any]:
    import torch

    out = {
        "device": str(device),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "git_commit": _git_commit(),
        "utc": _now(),
    }
    if getattr(device, "type", "cpu") == "cuda":
        out["gpu_name"] = torch.cuda.get_device_name(0)
        try:
            out["gpu_count"] = int(torch.cuda.device_count())
        except Exception:
            out["gpu_count"] = None
        try:
            raw = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).strip()
            out["driver"] = raw.splitlines()[0].strip() if raw else None
        except Exception:
            out["driver"] = None
    return out


def _train_path(arm: str, seed: int) -> Path:
    return C.OUT / f"train_{arm}_s{seed}.json"


def _soup_path(arm: str, seed: int) -> Path:
    return C.OUT / f"soup_{arm}_s{seed}.pt"


def _pred_path(arm: str, seed: int) -> Path:
    return C.OUT / f"valid_pred_{arm}_s{seed}.npz"


def _pred_freeze_path(arm: str, seed: int) -> Path:
    return C.OUT / f"valid_freeze_{arm}_s{seed}.json"


def _write_npz_once(path: Path, pred: np.ndarray) -> dict[str, Any]:
    if path.exists():
        with np.load(path) as z:
            old = np.asarray(z["pred"], dtype=np.float64)
        if old.shape != pred.shape or not np.allclose(old, pred, atol=0.0, rtol=0.0):
            raise SystemExit(f"refusing to overwrite a different prediction file {path}")
    else:
        np.savez(path, pred=pred)
    with np.load(path) as z:
        arr = np.asarray(z["pred"], dtype=np.float64)
    return {
        "path": _stored(path),
        "file_sha256": C.sha256_file(path),
        "array_sha256_float32": C.sha256_array(arr),
        "array_sha256_float64": C.sha256_array(arr, "float64"),
        "n": int(arr.size),
    }


def _read_pred(path: Path) -> np.ndarray:
    with np.load(path) as z:
        return np.asarray(z["pred"], dtype=np.float64)


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def cmd_audit(args: argparse.Namespace) -> int:
    out = C.OUT / "p3_reuse_audit.json"
    if out.exists():
        raise SystemExit("audit record exists; refusing to rewrite")

    cfg = C.load_config()
    device = _device(args.device)
    checks: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "stage": "audit",
        "created_utc": _now(),
        "inputs": {},
        "code": {},
        "p3": {},
        "batch_order": {},
        "regime": {},
    }

    for name, spec in cfg["frozen_inputs"].items():
        path = REPO / spec["path"]
        checks["inputs"][name] = C.verify_file(name, path, spec["sha256"])

    for name, rel in cfg["audit_expectations"].get("code_hashes", {}).items():
        path = REPO / rel
        got = C.sha256_file(path)
        if got != cfg["audit_expectations"]["code_hashes_expected"][name]:
            raise SystemExit(f"frozen source hash mismatch for {name}: {got}")
        checks["code"][name] = {"path": rel, "sha256": got}
    for name in ("sta_models.py", "sta_data.py", "sta_common.py", "sta_runner.py"):
        path = HERE / name
        checks["code"][f"round:{name}"] = {"path": _stored(path), "sha256": C.sha256_file(path)}

    valid_objs = C.load_objects("objects_valid")
    if len(valid_objs) != C.N_VALID:
        raise SystemExit(f"valid objects {len(valid_objs)} != {C.N_VALID}")

    for seed in C.SEEDS:
        model, info = C.build_p3_reference(seed, "cpu")
        pred = bp.predict_model(model, valid_objs, "cpu").astype(np.float64)
        hist, prov = C.historical_valid_pred(seed)
        max_diff = float(np.max(np.abs(pred - hist)))
        replay_path = C.OUT / f"p3_valid_replay_s{seed}.npz"
        # label-free replay artefact; overwrite only if identical
        if replay_path.exists():
            old = _read_pred(replay_path)
            if old.shape != pred.shape or float(np.max(np.abs(old - pred))) > 1e-7:
                raise SystemExit(f"existing P3 replay differs for seed {seed}")
        else:
            np.savez(replay_path, pred=pred)
        checks["p3"][str(seed)] = {
            "parameters": info["parameters"],
            "scratch": info["scratch"],
            "encoder_used_sha256": info["encoder_used_sha256"],
            "soup": info["soup"],
            "historical_prediction": prov,
            "replay_path": _stored(replay_path),
            "replay_file_sha256": C.sha256_file(replay_path),
            "replay_array_sha256_float32": C.sha256_array(pred),
            "historical_array_sha256_float32": prov["array_sha256_float32"],
            "max_abs_replay_minus_historical": max_diff,
            "replay_agrees_5e-5": bool(max_diff <= 5e-5),
        }
        if max_diff > 5e-5:
            raise SystemExit(f"P3 replay mismatch for seed {seed}: {max_diff}")

    exp = cfg["audit_expectations"]
    for seed in C.SEEDS:
        got = bp.batch_order_hash(seed, C.N_TRAIN, C.BATCH_SIZE, C.EPOCHS)
        checks["batch_order"][str(seed)] = {
            "full_hash": got,
            "expected": exp["batch_order_sha256"][str(seed)],
            "match": got == exp["batch_order_sha256"][str(seed)],
        }
        if got != exp["batch_order_sha256"][str(seed)]:
            raise SystemExit(f"batch order hash mismatch for seed {seed}")

    env = _env(device)
    expected_regime = exp.get("p3_execution_regime", {})
    compare_keys = ("gpu_name", "driver", "torch", "torch_cuda", "python")
    if getattr(device, "type", "cpu") == "cuda":
        regime_match = all(
            env.get(k) == expected_regime.get(k) for k in compare_keys if expected_regime.get(k) is not None
        )
        p3_baseline_required: bool | None = not regime_match
        regime_status = "GPU_PROBED"
    else:
        regime_match = None
        p3_baseline_required = None
        regime_status = "CPU_ONLY_NO_GPU_PROBE"
    checks["regime"] = {
        "observed": env,
        "expected": expected_regime,
        "compare_keys": list(compare_keys),
        "match": regime_match,
        "status": regime_status,
        "p3_baseline_required": p3_baseline_required,
        "branch": "A_REUSE_FROZEN_P3"
        if regime_match is True
        else ("B_RETRAIN_P3_REQUIRED" if regime_match is False else "UNPROBED_CPU"),
    }

    checks["status"] = (
        "PASS"
        if all(x.get("replay_agrees_5e-5", True) for x in checks["p3"].values())
        and all(x["match"] for x in checks["batch_order"].values())
        else "FAIL"
    )
    C.write_once_json(out, checks)
    log(
        f"[audit] status={checks['status']} branch={checks['regime']['branch']} "
        f"replay_max={max(v['max_abs_replay_minus_historical'] for v in checks['p3'].values()):.3e}"
    )
    if checks["status"] != "PASS":
        raise SystemExit("audit failed")
    return 0


# ---------------------------------------------------------------------------
# CPU gates
# ---------------------------------------------------------------------------


def _tiny_objects() -> list[sro.MolObjects]:
    """Label-free synthetic objects built with the frozen object builder."""
    out: list[sro.MolObjects] = []
    # 4-atom path: 0-1-2-3
    x = np.zeros((4, 1), dtype=np.int64)
    edges = [(0, 1), (1, 2), (2, 3)]
    ei = np.asarray([[u, v] for u, v in edges] + [[v, u] for u, v in edges], dtype=np.int64).T
    ea = np.zeros((ei.shape[1],), dtype=np.int64)
    out.append(sro.build_mol_objects(0, x, ei, ea, smiles="C-C-C-C"))
    # 4-atom star: 0 centre connected to 1,2,3 (three junctions)
    edges = [(0, 1), (0, 2), (0, 3)]
    ei = np.asarray([[u, v] for u, v in edges] + [[v, u] for u, v in edges], dtype=np.int64).T
    ea = np.zeros((ei.shape[1],), dtype=np.int64)
    out.append(sro.build_mol_objects(1, x, ei, ea, smiles="C(C)(C)C"))
    # one atom, no bond / junction
    out.append(
        sro.build_mol_objects(
            2,
            np.zeros((1, 1), dtype=np.int64),
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            smiles="C",
        )
    )
    # two atoms, one bond, no junction
    out.append(
        sro.build_mol_objects(
            3,
            np.zeros((2, 1), dtype=np.int64),
            np.asarray([[0, 1], [1, 0]], dtype=np.int64).T,
            np.zeros((2,), dtype=np.int64),
            smiles="CC",
        )
    )
    return out


def _direct_model(variant: str, seed: int = 0) -> M.STAObjectModel:
    cfg = M.src.BJOCConfig()
    norm_mean = np.zeros(sro.GLOBAL_DIM, dtype=np.float32)
    norm_scale = np.ones(sro.GLOBAL_DIM, dtype=np.float32)
    model = M.STAObjectModel(cfg, norm_mean, norm_scale, variant)
    M.init_new_modules(model, seed)
    return model


def _direct_base(seed: int = 0) -> M.src.BJOC:
    cfg = M.src.BJOCConfig()
    norm_mean = np.zeros(sro.GLOBAL_DIM, dtype=np.float32)
    norm_scale = np.ones(sro.GLOBAL_DIM, dtype=np.float32)
    return M.src.BJOC(cfg, norm_mean, norm_scale, M.src.NULL)


def _gate(gates: list[dict[str, Any]], name: str, ok: bool, detail: Any) -> None:
    gates.append({"name": name, "pass": bool(ok), "detail": detail})


def _mha_state(model: M.STAObjectModel) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for li, block in enumerate(model.blocks):
        mix = block.mix
        for attr in ("q_proj", "k_proj", "v_proj", "out_proj"):
            lin = getattr(mix, attr)
            out[f"layer{li}.{attr}.weight"] = M.tensor_sha256(lin.weight)
            out[f"layer{li}.{attr}.bias"] = M.tensor_sha256(lin.bias)
    return out


def _common_state(model: M.STAObjectModel) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, p in model.named_parameters():
        if name.startswith(("type_embedding", "reader.")) or ".norm" in name or ".ffn." in name:
            out[name] = M.tensor_sha256(p)
        if name.startswith(("atom_encoder.", "bond_encoder.", "junction_encoder.", "group_composer.")):
            out[name] = M.tensor_sha256(p)
    return out


def _permuted_object(obj: sro.MolObjects, seed: int = 0) -> sro.MolObjects:
    """Atom renumbering through the frozen raw-object builder."""
    rng = np.random.default_rng(int(seed))
    n = int(obj.n_atoms)
    perm = rng.permutation(n)  # old -> new
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    # Recover raw atom types and bond types from the frozen one-hot features.
    atom_type = np.argmax(obj.atom_feat[:, : sro.ATOM_CATEGORIES], axis=1).astype(np.int64)
    bond_type = np.argmax(obj.bond_feat[:, : sro.BOND_CATEGORIES], axis=1).astype(np.int64)
    x = atom_type[perm].reshape(-1, 1)
    u = inv[obj.bond_index[:, 0]]
    v = inv[obj.bond_index[:, 1]]
    ei = np.asarray([[a, b] for a, b in zip(u, v)] + [[b, a] for a, b in zip(u, v)], dtype=np.int64).T
    ea = np.concatenate([bond_type, bond_type], axis=0)
    return sro.build_mol_objects(obj.mol_index, x, ei, ea, smiles=obj.smiles)


def cmd_gates(args: argparse.Namespace) -> int:
    out = C.OUT / "cpu_gates_this_round.json"
    if out.exists():
        raise SystemExit("gate record exists; refusing to rewrite")
    import torch

    gates: list[dict[str, Any]] = []
    toy = _tiny_objects()
    cache = D.build_relation_cache(toy)
    batch = D.batch_objects_sta(toy, list(range(len(toy))), cache)

    # G1 parameter counts and reader replacement
    param_ok = True
    param_detail: dict[str, Any] = {}
    for arm in C.ARMS:
        model = _direct_model(arm)
        audit = M.parameter_audit(model)
        mod = M.module_audit(model)
        param_detail[arm] = {"audit": audit, "module_audit": mod}
        param_ok = param_ok and audit["parameter_exact"] and mod["pass"]
        param_ok = param_ok and audit["groups"]["reader"] == M.READER_PARAMETERS
        expected_blocks = 2 * M.BLOCK_PARAMETERS + (M.RELATION_PARAMETERS if arm == "R" else 0)
        param_ok = param_ok and audit["groups"]["blocks"] == expected_blocks
        param_ok = param_ok and audit["groups"]["type_embedding"] == M.TYPE_PARAMETERS
        param_ok = param_ok and not audit["old_reader_present"]
    param_ok = param_ok and M.EXPECTED_PARAMETERS == {
        "S": 483137,
        "T": 483137,
        "R": 483289,
    }
    _gate(gates, "G1_parameters_and_no_old_reader", param_ok, param_detail)

    # G2 P3 z bypass bitwise parity against the original BJOC reader input
    base = _direct_base()
    sta_s = _direct_model("S")
    M.copy_shared_(sta_s, base)
    with torch.no_grad():
        b_base = base.forward_features(batch)["reader_in"]
        b_sta = sta_s.forward_features(batch)["reader_in_p3"]
    z_diff = float((b_base - b_sta).abs().max())
    _gate(
        gates,
        "G2_zP3_bypass_matches_original_reader_input",
        b_base.shape == b_sta.shape == (len(toy), 321) and z_diff == 0.0,
        {"shape": list(b_sta.shape), "max_abs_diff": z_diff},
    )

    # G3 T/R step-0 identity and zero R bias
    t_model = _direct_model("T")
    r_model = _direct_model("R")
    # Explicitly copy every T tensor that exists in R; the R-only bias tables
    # stay at their zero initialisation.  This makes the step-0 identity
    # independent of construction order.
    with torch.no_grad():
        t_state = t_model.state_dict()
        r_state = r_model.state_dict()
        for key in r_state:
            if key in t_state:
                r_state[key].copy_(t_state[key])
        r_model.load_state_dict(r_state, strict=True)
        for block in r_model.blocks:
            block.mix.struct_bias.type_bias.zero_()
            block.mix.struct_bias.dist_bias.zero_()
            block.mix.struct_bias.overlap_bias.zero_()
    with torch.no_grad():
        pt = t_model(batch)
        pr = r_model(batch)
    tr_diff = float((pt - pr).abs().max())
    mha_same = _mha_state(t_model) == _mha_state(r_model)
    common_same = _common_state(t_model) == _common_state(r_model)
    bias_zero = all(
        float(b.mix.struct_bias.type_bias.abs().sum())
        + float(b.mix.struct_bias.dist_bias.abs().sum())
        + float(b.mix.struct_bias.overlap_bias.abs().sum())
        == 0.0
        for b in r_model.blocks
    )
    _gate(
        gates,
        "G3_T_R_zero_bias_step0_identity",
        tr_diff == 0.0 and mha_same and common_same and bias_zero,
        {
            "prediction_max_abs_diff": tr_diff,
            "mha_state_equal": mha_same,
            "common_state_equal": common_same,
            "bias_zero": bias_zero,
        },
    )

    # G4 hand-checked relation bins on the 4-atom path
    path = toy[0]
    rec = D.relation_tables(path)
    n_a, n_b = 4, 3
    # token order: atoms 0..3, bonds 0..2, junctions 0..1
    # path bonds: (0,1),(1,2),(2,3); junctions: (0,1,2),(1,2,3)
    expected = {}
    expected[("A0", "A3")] = (3, 0)
    expected[("B0", "B2")] = (1, 0)  # {0,1} vs {2,3}: min dist 1
    expected[("B0", "B1")] = (0, 1)  # share atom 1
    expected[("J0", "J1")] = (0, 2)  # share atoms 1,2
    expected[("A3", "J0")] = (1, 0)  # atom 3 -> {0,1,2}: nearest atom 2
    expected[("A0", "J1")] = (1, 0)  # atom 0 -> {1,2,3}: nearest atom 1
    exp_map = {
        "A0": 0,
        "A1": 1,
        "A2": 2,
        "A3": 3,
        "B0": n_a + 0,
        "B1": n_a + 1,
        "B2": n_a + 2,
        "J0": n_a + n_b + 0,
        "J1": n_a + n_b + 1,
    }
    hand = {}
    hand_ok = True
    for (x, y), (ed, eo) in expected.items():
        i, j = exp_map[x], exp_map[y]
        got = (int(rec["dist_bin"][i, j]), int(rec["overlap_bin"][i, j]))
        hand[f"{x}-{y}"] = {"expected": [ed, eo], "got": list(got)}
        hand_ok = hand_ok and got == (ed, eo)
    _gate(gates, "G4_relation_bins_hand_checked_path", hand_ok, hand)

    # G5 disconnected relation bin on a synthetic two-component molecule
    disc = sro.MolObjects(
        mol_index=99,
        smiles="CC.CC",
        n_atoms=4,
        atom_feat=np.zeros((4, sro.ATOM_FEAT_DIM), dtype=np.float32),
        bond_index=np.asarray([[0, 1], [2, 3]], dtype=np.int64),
        bond_type=np.zeros(2, dtype=np.int64),
        bond_feat=np.zeros((2, sro.BOND_FEAT_DIM), dtype=np.float32),
        junc_index=np.zeros((0, 3), dtype=np.int64),
        junc_bond=np.zeros((0, 2), dtype=np.int64),
        junc_feat=np.zeros((0, sro.JUNC_FEAT_DIM), dtype=np.float32),
        global_feat=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
    )
    drec = D.relation_tables(disc)
    # atom 0 component to bond {2,3}: disconnected -> bin 5
    disc_ok = int(drec["dist_bin"][0, 4 + 1]) == D.DIST_DISCONNECTED
    _gate(
        gates,
        "G5_disconnected_distance_bin",
        disc_ok,
        {"dist_bin_A0_B1": int(drec["dist_bin"][0, 5]), "expected": D.DIST_DISCONNECTED},
    )

    # G6 atom renumbering, bond endpoint swap, junction endpoint swap
    real_objs = C.load_objects("objects_train10k")
    real = real_objs[0]
    real_cache = D.build_relation_cache([real])
    real_batch = D.batch_objects_sta([real], [0], real_cache)
    model = C.build_model("R", 0, "cpu")[0]
    model.eval()
    with torch.no_grad():
        p0 = float(model(real_batch)[0])
    perm_obj = _permuted_object(real, seed=0)
    perm_cache = D.build_relation_cache([perm_obj])
    with torch.no_grad():
        p1 = float(model(D.batch_objects_sta([perm_obj], [0], perm_cache))[0])

    import copy

    bond_swap = copy.deepcopy(real)
    bond_swap.bond_index = bond_swap.bond_index[:, ::-1].copy()
    with torch.no_grad():
        p2 = float(model(D.batch_objects_sta([bond_swap], [0], D.build_relation_cache([bond_swap])))[0])

    junc_swap = copy.deepcopy(real)
    junc_swap.junc_index = junc_swap.junc_index[:, ::-1].copy()
    junc_swap.junc_bond = junc_swap.junc_bond[:, ::-1].copy()
    with torch.no_grad():
        p3 = float(model(D.batch_objects_sta([junc_swap], [0], D.build_relation_cache([junc_swap])))[0])

    inv_diffs = {
        "atom_renumber": abs(p1 - p0),
        "bond_endpoint_swap": abs(p2 - p0),
        "junction_endpoint_swap": abs(p3 - p0),
    }
    _gate(
        gates,
        "G6_object_invariances_float32",
        max(inv_diffs.values()) <= 2e-5,
        {"baseline": p0, "max_abs_diff": inv_diffs, "tolerance": 2e-5},
    )

    # G7 token row permutation with synchronised relation matrices
    with torch.no_grad():
        h0 = torch.randn(len(toy), batch["tmax"], M.DIM)
        h0 = h0 * batch["token_valid"].unsqueeze(-1)
        _, sums0, _ = model.encode_tokens(
            h0,
            batch["token_valid"],
            batch["token_type_padded"],
            batch["rel_dist"].long(),
            batch["rel_overlap"].long(),
        )
    perm = torch.stack(
        [
            torch.randperm(batch["tmax"], generator=torch.Generator().manual_seed(17 + g))
            for g in range(len(toy))
        ]
    )
    h0p = torch.stack([h0[g, perm[g]] for g in range(len(toy))])
    validp = torch.stack([batch["token_valid"][g, perm[g]] for g in range(len(toy))])
    typep = torch.stack([batch["token_type_padded"][g, perm[g]] for g in range(len(toy))])
    distp = torch.stack([batch["rel_dist"][g][perm[g]][:, perm[g]] for g in range(len(toy))]).long()
    overp = torch.stack([batch["rel_overlap"][g][perm[g]][:, perm[g]] for g in range(len(toy))]).long()
    with torch.no_grad():
        _, sums1, _ = model.encode_tokens(h0p, validp, typep, distp, overp)
    perm_diff = float((sums0 - sums1).abs().max())
    _gate(
        gates,
        "G7_token_permutation_with_relation_sync",
        perm_diff <= 2e-5,
        {"max_abs_type_sum_diff": perm_diff, "tolerance": 2e-5},
    )

    # G8 batch composition / padding / cross-molecule isolation
    mixed_cache = D.build_relation_cache(toy)
    with torch.no_grad():
        p_single = torch.stack(
            [model(D.batch_objects_sta([o], [0], D.build_relation_cache([o]))) for o in toy]
        ).view(-1)
        p_mixed = model(D.batch_objects_sta(toy, list(range(len(toy))), mixed_cache))
    # perturb molecule 1 only; molecule 0 prediction must not move
    toy2 = [copy.deepcopy(toy[0]), copy.deepcopy(toy[1])]
    toy2[1].atom_feat = toy2[1].atom_feat + 0.5
    with torch.no_grad():
        p_pert = model(D.batch_objects_sta(toy2, [0, 1], D.build_relation_cache(toy2)))
    comp_diff = float((p_single - p_mixed).abs().max())
    cross_diff = float(abs(float(p_pert[0]) - float(p_mixed[0])))
    _gate(
        gates,
        "G8_batch_composition_padding_cross_molecule",
        comp_diff <= 2e-5 and cross_diff <= 2e-5,
        {
            "single_vs_mixed_max_abs_diff": comp_diff,
            "cross_molecule_max_abs_diff": cross_diff,
            "tolerance": 2e-5,
        },
    )

    # G9 empty bond / junction boundary
    empty_ok = True
    empty_detail = {}
    for idx, name in ((2, "one_atom_no_bond"), (3, "two_atoms_one_bond_no_junction")):
        b = D.batch_objects_sta([toy[idx]], [0], D.build_relation_cache([toy[idx]]))
        with torch.no_grad():
            pred, feats = model(b, return_features=True)
        finite = bool(torch.isfinite(pred).all() and torch.isfinite(feats["reader_in"]).all())
        sums = feats["type_sums"][0]
        zero_j = float(sums[2].abs().max())
        empty_detail[name] = {
            "prediction": float(pred[0]),
            "finite": finite,
            "sum_J_max_abs": zero_j,
            "n_tokens": int(b["n_tokens"]),
        }
        empty_ok = empty_ok and finite and zero_j == 0.0
    _gate(gates, "G9_empty_bond_junction_boundary", empty_ok, empty_detail)

    # G10 one/two-update gradients and branch participation
    grad_model = C.build_model("R", 0, "cpu")[0]
    grad_model.train()
    opt = torch.optim.Adam(grad_model.parameters(), lr=1e-3)
    y = torch.linspace(-1.0, 1.0, len(toy))
    probes = {}
    grad_ok = True
    for step in (1, 2):
        pred = grad_model(batch)
        loss = torch.nn.functional.l1_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        groups = _grad_group_norms(grad_model)
        rgrad = _r_bias_grad_norms(grad_model)
        probes[str(step)] = {"loss": float(loss), "groups": groups, "r_bias": rgrad}
        finite = all(math.isfinite(v) for v in groups.values())
        finite = finite and all(
            math.isfinite(float(part["abs_sum"])) for rec in rgrad.values() for part in rec.values()
        )
        grad_ok = grad_ok and finite
        grad_ok = grad_ok and groups.get("type_embedding", 0.0) > 0.0
        grad_ok = grad_ok and groups.get("reader", 0.0) > 0.0
        grad_ok = grad_ok and groups.get("token_ffn", 0.0) > 0.0
        grad_ok = grad_ok and groups.get("mix", 0.0) > 0.0
        if step == 2:
            grad_ok = grad_ok and any(
                float(part["abs_sum"]) > 0 for rec in rgrad.values() for part in rec.values()
            )
        opt.step()
    _gate(
        gates,
        "G10_two_update_gradients_new_branch",
        grad_ok,
        {"probes": probes, "note": "loss uses synthetic y only; no valid labels"},
    )

    # G11 batch-order hashes match R3
    exp = C.load_config()["audit_expectations"]["batch_order_sha256"]
    order = {str(s): bp.batch_order_hash(s, C.N_TRAIN, C.BATCH_SIZE, C.EPOCHS) for s in C.SEEDS}
    _gate(
        gates,
        "G11_batch_order_matches_R3",
        all(order[str(s)] == exp[str(s)] for s in C.SEEDS),
        order,
    )

    # G12 P3 frozen provenance is present (full replay lives in audit)
    p3_ok = True
    p3_detail = {}
    for s in C.SEEDS:
        path = C.p3_soup_path(s)
        sha = C.sha256_file(path)
        p3_detail[str(s)] = {"path": _stored(path), "sha256": sha, "expected": C.p3_soup_sha_expected(s)}
        p3_ok = p3_ok and sha == C.p3_soup_sha_expected(s)
    _gate(gates, "G12_frozen_P3_soups_present", p3_ok, p3_detail)

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": _now(),
        "n_gates": len(gates),
        "n_pass": sum(1 for g in gates if g["pass"]),
        "gates": gates,
        "all_pass": all(g["pass"] for g in gates),
    }
    C.write_once_json(out, payload)
    log(f"[gates] {payload['n_pass']}/{payload['n_gates']} pass")
    if not payload["all_pass"]:
        for g in gates:
            if not g["pass"]:
                log(f"  FAIL {g['name']}: {json.dumps(g['detail'])[:500]}")
        raise SystemExit("CPU gates failed")
    return 0


# ---------------------------------------------------------------------------
# gradients / parameter norms
# ---------------------------------------------------------------------------


def _group_name(name: str) -> str:
    if name.startswith("atom_encoder."):
        return "atom_encoder"
    if name.startswith("bond_encoder."):
        return "bond_encoder"
    if name.startswith("junction_encoder."):
        return "junction_encoder"
    if name.startswith("group_composer."):
        return "group_composer"
    if name.startswith("type_embedding"):
        return "type_embedding"
    if ".mix.struct_bias." in name:
        return "r_struct_bias"
    if ".mix." in name:
        return "mix"
    if ".ffn." in name:
        return "token_ffn"
    if ".norm" in name:
        return "ln"
    if name.startswith("reader."):
        return "reader"
    return "other"


def _grad_group_norms(model: Any) -> dict[str, float]:
    sums: dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        key = _group_name(name)
        sums[key] = sums.get(key, 0.0) + float(p.grad.detach().pow(2).sum())
    return {k: math.sqrt(v) for k, v in sums.items()}


def _param_group_norms(model: Any) -> dict[str, float]:
    sums: dict[str, float] = {}
    for name, p in model.named_parameters():
        key = _group_name(name)
        sums[key] = sums.get(key, 0.0) + float(p.detach().pow(2).sum())
    return {k: math.sqrt(v) for k, v in sums.items()}


def _r_bias_grad_norms(model: Any) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if model.variant != "R":
        return out
    for li, block in enumerate(model.blocks):
        sb = block.mix.struct_bias
        rec = {}
        for name, p in (
            ("type_bias", sb.type_bias),
            ("dist_bias", sb.dist_bias),
            ("overlap_bias", sb.overlap_bias),
        ):
            if p.grad is None:
                rec[name] = {"abs_sum": 0.0, "max_abs": 0.0}
            else:
                rec[name] = {
                    "abs_sum": float(p.grad.detach().abs().sum()),
                    "max_abs": float(p.grad.detach().abs().max()) if p.grad.numel() else 0.0,
                }
        out[f"layer{li}"] = rec
    return out


# ---------------------------------------------------------------------------
# flops
# ---------------------------------------------------------------------------


def cmd_flops(args: argparse.Namespace) -> int:
    out = C.OUT / "flops_this_round.json"
    if out.exists():
        raise SystemExit("FLOP record exists; refusing to rewrite")
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    objs = C.load_objects("objects_train10k")
    cache = D.build_relation_cache(objs)
    rows = list(range(128))
    batch = D.batch_objects_sta(objs, rows, cache)
    payload: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": _now(),
        "n_graphs": len(rows),
        "n_tokens": int(batch["n_tokens"]),
        "relation_bin_counts": D.relation_bin_counts(cache, rows),
        "arms": {},
    }
    for arm in C.ARMS:
        model = _direct_model(arm)
        model.eval()
        counter = FlopCounterMode(display=False)
        try:
            with torch.no_grad(), counter:
                model(batch)
            table = counter.get_flop_counts()
            total = int(counter.get_total_flops())
            prefix = type(model).__name__
            by_module = {}
            for name, _child in model.named_children():
                key = f"{prefix}.{name}"
                if key in table:
                    by_module[name] = int(sum(int(v) for v in table[key].values()))
            payload["arms"][arm] = {
                "total_forward_flops": total,
                "by_module": by_module,
                "parameters": int(sum(p.numel() for p in model.parameters())),
            }
        except Exception as exc:  # pragma: no cover - surfaced as a gate failure
            payload["arms"][arm] = {"error": f"{type(exc).__name__}: {exc}"}
    payload["note"] = (
        "FlopCounterMode analytic GEMM/conv FLOPs on one fixed 128-molecule train batch; "
        "gather/elementwise/embedding work and scatter reductions are not fully counted. "
        "No wall-clock or GPU-cost claim is made from this record."
    )
    C.write_once_json(out, payload)
    log(f"[flops] {json.dumps({k: v.get('total_forward_flops') for k, v in payload['arms'].items()})}")
    return 0


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def _require_branch_a() -> dict[str, Any]:
    path = C.OUT / "p3_reuse_audit.json"
    if not path.exists():
        raise SystemExit("P3 reuse audit is missing; run prep/audit first")
    audit = C.read_json(path)
    if audit.get("regime", {}).get("p3_baseline_required") is not False:
        raise SystemExit("P3 reuse Branch A is not established; refusing to train a non-comparable round")
    return audit


def _save_train_payload(payload: Mapping[str, Any], soup_state: Mapping[str, Any]) -> None:
    import torch

    arm = str(payload["arm"])
    seed = int(payload["seed"])
    train_path = _train_path(arm, seed)
    soup_path = _soup_path(arm, seed)
    if train_path.exists() or soup_path.exists():
        raise SystemExit(f"refusing to overwrite existing training artefacts for {arm} s{seed}")
    torch.save(soup_state, soup_path)
    record = dict(payload)
    record["soup_path"] = _stored(soup_path)
    record["soup_sha256"] = C.sha256_file(soup_path)
    C.write_once_json(train_path, record)
    log(
        f"[{arm} s{seed}] saved soup {record['soup_sha256'][:12]} "
        f"fit={record['soup_fit_mae']:.6f} wall={record['wall_seconds']:.0f}s"
    )


def cmd_smoke(args: argparse.Namespace) -> int:
    out = C.OUT / "smoke_this_round.json"
    if out.exists():
        raise SystemExit("smoke record exists; refusing to rewrite")
    _require_branch_a()
    device = _device(args.device)
    objs = C.load_objects("objects_train10k")
    y = C.load_train_y()
    n_rows = 1024
    subset = objs[:n_rows]
    rows = list(range(n_rows))
    cache = D.build_relation_cache(subset)
    cfg = C.load_config()
    cap_seconds = float(cfg["budget"]["gpu_hours_cap"]) * 3600.0
    records: dict[str, Any] = {}
    smoke_wall = 0.0
    for arm in C.ARMS:
        log(f"[smoke] {arm} 2 epochs x {n_rows} rows")
        payload = _train_loop(
            arm=arm,
            seed=0,
            train_objs=subset,
            rows=rows,
            y_fit=y[:n_rows],
            device=device,
            cache=cache,
            smoke_epochs=2,
            save_soup=False,
        )
        payload.pop("soup_state", None)
        smoke_wall += float(payload["wall_seconds"])
        records[arm] = {
            "wall_seconds": payload["wall_seconds"],
            "step_seconds": payload["step_seconds"],
            "updates": payload["updates"],
            "soup_fit_mae": payload["soup_fit_mae"],
            "training_health": payload["training_health"],
        }
    mean_step = {arm: float(records[arm]["step_seconds"]["mean"]) for arm in C.ARMS}
    projected_formal = 2.0 * sum(mean_step.values()) * float(C.UPDATES)
    reserve = smoke_wall + 300.0
    projected_total = 1.20 * projected_formal + reserve
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": _now(),
        "device": str(device),
        "env": _env(device),
        "rows": n_rows,
        "epochs": 2,
        "per_arm": records,
        "mean_step_seconds": mean_step,
        "formal_runs": 6,
        "projected_formal_seconds": projected_formal,
        "projected_formal_gpu_hours": projected_formal / 3600.0,
        "smoke_wall_seconds": smoke_wall,
        "overhead_reserve_seconds": reserve,
        "margin_factor": 1.20,
        "projected_total_seconds": projected_total,
        "projected_total_gpu_hours": projected_total / 3600.0,
        "cap_seconds": cap_seconds,
        "cap_gpu_hours": cap_seconds / 3600.0,
        "pass": bool(projected_total <= cap_seconds),
        "note": (
            "Smoke weights are temporary and never scored. No valid label is read. "
            "Projection uses the mean measured per-update time for each arm on 1,024 "
            "train rows, two seeds per arm, plus smoke wall time and a 300 s overhead "
            "reserve; 1.20 is the pre-registered failure margin."
        ),
    }
    C.write_once_json(out, payload)
    log(
        f"[smoke] projected formal {payload['projected_formal_gpu_hours']:.3f} GPU-h; "
        f"with margin/overhead {payload['projected_total_gpu_hours']:.3f} / "
        f"{payload['cap_gpu_hours']:.3f} GPU-h -> pass={payload['pass']}"
    )
    return 0


# ---------------------------------------------------------------------------
# formal training
# ---------------------------------------------------------------------------


def _train_loop(
    *,
    arm: str,
    seed: int,
    train_objs: Sequence[Any],
    rows: Sequence[int],
    y_fit: np.ndarray,
    device: Any,
    cache: Sequence[Mapping[str, np.ndarray]],
    smoke_epochs: int | None = None,
    save_soup: bool = True,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    recipe = bp.RECIPES[C.RECIPE_KEY]
    model, build_info = C.build_model(arm, seed, device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    if n_params != C.EXPECTED_PARAMETERS[arm]:
        raise SystemExit(f"parameter audit failed: {n_params} != {C.EXPECTED_PARAMETERS[arm]}")

    init_norms = _param_group_norms(model)
    epochs = int(recipe["epochs"]) if smoke_epochs is None else int(smoke_epochs)
    if smoke_epochs is not None and not (1 <= epochs <= 5):
        raise SystemExit("smoke epochs must be in [1, 5]")
    n = int(len(train_objs))
    if n != len(rows):
        raise SystemExit(f"rows/train_objs length mismatch: {len(rows)} vs {n}")
    steps_per_epoch = math.ceil(n / C.BATCH_SIZE)
    total_updates = epochs * steps_per_epoch
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(recipe["lr"]), weight_decay=float(recipe["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=float(recipe["lr_min"])
    )

    gen = torch.Generator().manual_seed(int(seed) + C.TRAIN_SHUFFLE_OFFSET)
    states: deque = deque(maxlen=C.SOUP_LAST)
    curve: list[dict[str, Any]] = []
    grad_probes: dict[str, Any] = {}
    order_hash = hashlib.sha256()
    lr_deviation_max = 0.0
    grad_norm_max = 0.0
    clip_events = 0
    total_mol = 0
    step_seconds: list[float] = []
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
            batch = D.batch_objects_sta(train_objs, indices, cache)
            batch = D.move_batch(batch, device)
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
                    "groups": _grad_group_norms(model),
                    "r_bias": _r_bias_grad_norms(model),
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
        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            log(
                f"[{arm} s{seed}] ep={epoch:03d} train={train_mae:.6f} "
                f"lr={lr_last:.3e} {curve[-1]['seconds']:.1f}s"
            )
        if not math.isfinite(train_mae):
            raise RuntimeError(f"non-finite loss at epoch {epoch} {arm} seed {seed}")
    wall = time.perf_counter() - started

    members, soup = bp._last_soup(list(states), epochs)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred, fit_seconds = C.predict_sta(model, train_objs, cache, device, batch_size=256)
    if not np.isfinite(fit_pred).all():
        raise RuntimeError("non-finite fit prediction after soup")
    soup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} if save_soup else None

    # ---- training-health and gradient expectations ------------------------
    g1 = grad_probes.get("1", {}).get("groups", {})
    g2 = grad_probes.get("2", {}).get("groups", {})
    r1 = grad_probes.get("1", {}).get("r_bias", {})
    r2 = grad_probes.get("2", {}).get("r_bias", {})
    r_nonzero_1 = any(rec["abs_sum"] > 0.0 for layer in r1.values() for rec in layer.values())
    r_nonzero_2 = any(rec["abs_sum"] > 0.0 for layer in r2.values() for rec in layer.values())
    grad_check = {
        "type_embedding_nonzero_first_step": g1.get("type_embedding", 0.0) > 0.0,
        "reader_nonzero_first_step": g1.get("reader", 0.0) > 0.0,
        "token_ffn_nonzero_first_step": g1.get("token_ffn", 0.0) > 0.0,
        "mix_nonzero_first_step": g1.get("mix", 0.0) > 0.0,
        "mix_nonzero_second_step": g2.get("mix", 0.0) > 0.0,
    }
    if arm == "R":
        grad_check["r_bias_grad_nonzero_step1_or_2"] = bool(r_nonzero_1 or r_nonzero_2)
    if not all(grad_check.values()):
        raise RuntimeError(f"gradient expectation failed for {arm} s{seed}: {grad_check}")
    relation_audit = M.relation_parameter_audit(model)
    if arm == "R":
        if relation_audit.get("total_parameters") != M.RELATION_PARAMETERS:
            raise RuntimeError(
                f"R relation parameters {relation_audit.get('total_parameters')} != {M.RELATION_PARAMETERS}"
            )
        nonzero = sum(
            int(rec[table]["nonzero"])
            for rec in relation_audit.values()
            if isinstance(rec, dict) and "type_bias" in rec
            for table in ("type_bias", "dist_bias", "overlap_bias")
        )
        if nonzero == 0:
            raise RuntimeError(f"R bias stayed exactly zero after training for seed {seed}")
        relation_audit["total_nonzero_entries"] = nonzero

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "arm": arm,
        "seed": int(seed),
        "recipe_key": C.RECIPE_KEY,
        "recipe": dict(recipe),
        "init": "full10k_masked_ssl_then_round_arm",
        "epochs": epochs,
        "updates": int(total_updates),
        "steps_per_epoch": steps_per_epoch,
        "n_train_rows": n,
        "soup_rule": f"parameter mean of the last {C.SOUP_LAST} epoch states",
        "soup_members": members,
        "soup_fit_mae": float(np.abs(fit_pred - y_fit).mean()),
        "soup_fit_seconds": float(fit_seconds),
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
        "relation_cache_sha256": D.relation_cache_sha256(cache),
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
        "extra_param_norms_init": init_norms,
        "extra_param_norms_final": _param_group_norms(model),
        "relation_bias_audit": relation_audit,
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
        "training_health": {
            "finite_loss": True,
            "finite_gradients": True,
            "gradient_expectation": grad_check,
        },
    }
    return {**payload, "soup_state": soup_state}


def cmd_train(args: argparse.Namespace) -> int:
    _require_branch_a()
    arm, seed = str(args.arm), int(args.seed)
    if arm not in C.ARMS or seed not in C.SEEDS:
        raise SystemExit(f"unknown arm/seed {arm}/{seed}")
    if _train_path(arm, seed).exists() or _soup_path(arm, seed).exists():
        raise SystemExit(f"training artefacts already exist for {arm} s{seed}")
    device = _device(args.device)
    objs = C.load_objects("objects_train10k")
    y = C.load_train_y()
    cache = D.build_relation_cache(objs)
    payload = _train_loop(
        arm=arm,
        seed=seed,
        train_objs=objs,
        rows=list(range(len(objs))),
        y_fit=y,
        device=device,
        cache=cache,
        smoke_epochs=None,
        save_soup=True,
    )
    soup_state = payload.pop("soup_state")
    if soup_state is None:
        raise RuntimeError("formal training returned no soup state")
    _save_train_payload(payload, soup_state)
    return 0


def cmd_run_seed(args: argparse.Namespace) -> int:
    _require_branch_a()
    device = _device(args.device)
    seed = int(args.seed)
    if seed not in C.SEEDS:
        raise SystemExit(f"unknown seed {seed}")
    order = [x.strip() for x in str(args.order).split(",") if x.strip()]
    if not order or any(a not in C.ARMS for a in order):
        raise SystemExit(f"bad arm order {order}")
    objs = C.load_objects("objects_train10k")
    y = C.load_train_y()
    cache = D.build_relation_cache(objs)
    for arm in order:
        tp, sp = _train_path(arm, seed), _soup_path(arm, seed)
        if tp.exists() and sp.exists():
            log(f"[{arm} s{seed}] artefacts exist, skip")
            continue
        if tp.exists() or sp.exists():
            raise SystemExit(f"partial artefacts exist for {arm} s{seed}; refusing to continue")
        log(f"[run-seed] start {arm} s{seed} on {device}")
        payload = _train_loop(
            arm=arm,
            seed=seed,
            train_objs=objs,
            rows=list(range(len(objs))),
            y_fit=y,
            device=device,
            cache=cache,
            smoke_epochs=None,
            save_soup=True,
        )
        soup_state = payload.pop("soup_state")
        if soup_state is None:
            raise RuntimeError("formal training returned no soup state")
        _save_train_payload(payload, soup_state)
    return 0


# ---------------------------------------------------------------------------
# label-free valid predictions and mechanism record
# ---------------------------------------------------------------------------


def _check_train_and_soup(arm: str, seed: int) -> dict[str, Any]:
    tp, sp = _train_path(arm, seed), _soup_path(arm, seed)
    if not tp.exists() or not sp.exists():
        raise SystemExit(f"missing training artefacts for {arm} s{seed}")
    rec = C.read_json(tp)
    if rec.get("soup_sha256") != C.sha256_file(sp):
        raise SystemExit(f"soup hash changed since training for {arm} s{seed}")
    return rec


def cmd_predict_all(args: argparse.Namespace) -> int:
    device = _device(args.device)
    valid_objs = C.load_objects("objects_valid")
    if len(valid_objs) != C.N_VALID:
        raise SystemExit(f"valid objects {len(valid_objs)} != {C.N_VALID}")
    cache = D.build_relation_cache(valid_objs)
    prediction_entries: dict[str, dict[str, Any]] = {}
    for arm in C.ARMS:
        prediction_entries[arm] = {}
        for seed in C.SEEDS:
            _check_train_and_soup(arm, seed)
            pred_path = _pred_path(arm, seed)
            freeze_path = _pred_freeze_path(arm, seed)
            if pred_path.exists() and freeze_path.exists():
                freeze = C.read_json(freeze_path)
                if freeze.get("file_sha256") != C.sha256_file(pred_path):
                    raise SystemExit(f"existing prediction file changed for {arm} s{seed}")
                pred = _read_pred(pred_path)
                log(f"[predict] {arm} s{seed} exists, verify only")
            else:
                model, _info = C.build_model(arm, seed, device)
                C.load_soup(model, _soup_path(arm, seed))
                pred, seconds = C.predict_sta(model, valid_objs, cache, device, batch_size=128)
                pred_rec = _write_npz_once(pred_path, pred)
                freeze = {
                    "protocol_id": C.PROTOCOL_ID,
                    "created_utc": _now(),
                    "arm": arm,
                    "seed": int(seed),
                    "split": "official PyG ZINC subset=True val (development, exposed)",
                    "n": int(pred.size),
                    "prediction": pred_rec,
                    "soup_path": _stored(_soup_path(arm, seed)),
                    "soup_sha256": C.sha256_file(_soup_path(arm, seed)),
                    "train_record": _stored(_train_path(arm, seed)),
                    "train_record_sha256": C.sha256_file(_train_path(arm, seed)),
                    "inference_seconds": float(seconds),
                    "official_test_loaded": False,
                    "valid_labels_loaded_at_predict": False,
                }
                C.write_once_json(freeze_path, freeze)
                log(
                    f"[predict] {arm} s{seed} n={pred.size} "
                    f"hash={pred_rec['array_sha256_float32'][:12]} {seconds:.2f}s"
                )
            prediction_entries[arm][str(seed)] = {
                "prediction": freeze["prediction"],
                "freeze_path": _stored(freeze_path),
                "freeze_file_sha256": C.sha256_file(freeze_path),
                "soup_sha256": freeze["soup_sha256"],
                "train_record_sha256": freeze["train_record_sha256"],
            }

    combined = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": _now(),
        "stage": "valid_prediction_freeze",
        "split": "official PyG ZINC subset=True val (development, exposed)",
        "n": C.N_VALID,
        "predictions": prediction_entries,
        "official_test_loaded": False,
        "valid_labels_loaded": False,
        "note": (
            "Six single-model predictions, no seed or arm ensemble. Each prediction "
            "array and freeze file is hashed before the score stage reads any valid label."
        ),
    }
    combined["combined_sha256"] = hashlib.sha256(
        json.dumps(prediction_entries, sort_keys=True).encode()
    ).hexdigest()
    out = C.OUT / "valid_prediction_freeze.json"
    if out.exists():
        old = C.read_json(out)
        if old.get("combined_sha256") != combined["combined_sha256"]:
            raise SystemExit("existing combined prediction freeze differs")
    else:
        C.write_once_json(out, combined)
    log(f"[predict] combined freeze {combined['combined_sha256'][:12]}")
    return 0


def _attention_stats(attn: Any, valid: Any) -> dict[str, Any]:
    import torch

    g, h, t, _ = attn.shape
    valid = valid.bool()
    n_keys = valid.sum(dim=1).to(torch.float64).clamp_min(1.0)
    log_den = torch.log(n_keys.clamp_min(2.0))
    ent = -(attn.double() * torch.log(attn.double().clamp_min(1e-30))).sum(dim=-1)
    norm_ent = ent / log_den[:, None, None]
    max_w = attn.double().max(dim=-1).values
    mask = valid[:, None, :].expand(g, h, t)
    eff = torch.exp(ent)
    return {
        "n_query_tokens": int(mask.sum()),
        "n_key_tokens": int(valid.sum()),
        "normalised_entropy_mean": float(norm_ent[mask].mean()),
        "max_weight_mean": float(max_w[mask].mean()),
        "effective_keys_mean": float(eff[mask].mean()),
        "attention_sum_max_abs_error": float((attn.double().sum(-1) - 1.0).abs().max()),
    }


def cmd_mechanism(args: argparse.Namespace) -> int:
    out = C.OUT / "mechanism_this_round.json"
    if out.exists():
        raise SystemExit("mechanism record exists; refusing to rewrite")
    import torch

    device = _device(args.device)
    valid_objs = C.load_objects("objects_valid")
    n_rows = 128
    subset = valid_objs[:n_rows]
    rows = list(range(n_rows))
    cache = D.build_relation_cache(subset)
    batch = D.batch_objects_sta(subset, rows, cache)
    batch = D.move_batch(batch, device)
    records: dict[str, Any] = {}
    for arm in C.ARMS:
        records[arm] = {}
        for seed in C.SEEDS:
            _check_train_and_soup(arm, seed)
            model, _info = C.build_model(arm, seed, device)
            C.load_soup(model, _soup_path(arm, seed))
            model.eval()
            with torch.no_grad():
                pred, feats = model(batch, return_features=True, need_attention=True)
            rec: dict[str, Any] = {
                "arm": arm,
                "seed": int(seed),
                "n_molecules": n_rows,
                "prediction_abs_mean": float(pred.abs().mean()),
                "prediction_std": float(pred.std()),
                "token_counts": [int(x) for x in batch["token_counts"].sum(0).tolist()],
                "tmax": int(batch["tmax"]),
                "parameter_norms": _param_group_norms(model),
                "labels_used": "none",
            }
            if arm in ("T", "R"):
                layers = []
                for li, attn in enumerate(feats["attention"]):
                    if attn is None:
                        layers.append({"layer": li, "attention": None})
                    else:
                        layers.append({"layer": li, **_attention_stats(attn, batch["token_valid"])})
                rec["attention_layers"] = layers
            if arm == "S":
                gamma = model.blocks[0].mix.gamma.detach()
                rec["gamma_layer0"] = {
                    "mean": float(gamma.mean()),
                    "std": float(gamma.std()),
                    "min": float(gamma.min()),
                    "max": float(gamma.max()),
                }
                gamma = model.blocks[1].mix.gamma.detach()
                rec["gamma_layer1"] = {
                    "mean": float(gamma.mean()),
                    "std": float(gamma.std()),
                    "min": float(gamma.min()),
                    "max": float(gamma.max()),
                }
            if arm == "R":
                rec["relation_bias"] = M.relation_parameter_audit(model)
                # label-free same-weight ablation: zero the structural bias only
                sb_params = [p for name, p in model.named_parameters() if ".mix.struct_bias." in name]
                saved = [p.detach().clone() for p in sb_params]
                with torch.no_grad():
                    for p in sb_params:
                        p.zero_()
                    pred_no_bias = model(batch)
                    for p, old in zip(sb_params, saved):
                        p.copy_(old)
                rec["bias_zeroed_prediction_change"] = {
                    "mean_abs": float((pred - pred_no_bias).abs().mean()),
                    "max_abs": float((pred - pred_no_bias).abs().max()),
                }
                train_rec = C.read_json(_train_path(arm, seed))
                rec["training_gradient_probes"] = train_rec.get("grad_probes", {})
            records[arm][str(seed)] = rec
            log(f"[mechanism] {arm} s{seed} done")
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": _now(),
        "stage": "mechanism",
        "split": "official PyG ZINC subset=True val structures only (first 128); no labels",
        "n_molecules": n_rows,
        "relation_bin_counts": D.relation_bin_counts(cache, rows),
        "records": records,
        "official_test_loaded": False,
        "valid_labels_loaded": False,
        "note": (
            "Response-magnitude diagnostics only. They are not performance, chemistry, "
            "faithfulness or causal claims. S has no cross-token attention; its gamma "
            "is reported as a capacity-control readout."
        ),
    }
    C.write_once_json(out, payload)
    log(f"[mechanism] wrote {_stored(out)} sha={C.sha256_file(out)[:12]}")
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    pred_freeze = C.OUT / "valid_prediction_freeze.json"
    mech = C.OUT / "mechanism_this_round.json"
    if not pred_freeze.exists():
        cmd_predict_all(argparse.Namespace(device=args.device))
    if not mech.exists():
        cmd_mechanism(argparse.Namespace(device=args.device))
    pred = C.read_json(pred_freeze)
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": _now(),
        "stage": "pre_score_freeze",
        "valid_prediction_freeze_path": _stored(pred_freeze),
        "valid_prediction_freeze_sha256": C.sha256_file(pred_freeze),
        "prediction_combined_sha256": pred["combined_sha256"],
        "mechanism_path": _stored(mech),
        "mechanism_sha256": C.sha256_file(mech),
        "prediction_array_sha256_float32": {
            arm: {
                str(seed): pred["predictions"][arm][str(seed)]["prediction"]["array_sha256_float32"]
                for seed in C.SEEDS
            }
            for arm in C.ARMS
        },
        "valid_labels_loaded": False,
        "official_test_loaded": False,
        "note": "Both the six predictions and the label-free mechanism record are frozen before score.",
    }
    out = C.OUT / "pre_score_freeze.json"
    if out.exists():
        old = C.read_json(out)
        if old.get("valid_prediction_freeze_sha256") != payload["valid_prediction_freeze_sha256"]:
            raise SystemExit("existing pre-score freeze differs")
        if old.get("mechanism_sha256") != payload["mechanism_sha256"]:
            raise SystemExit("existing pre-score mechanism freeze differs")
    else:
        C.write_once_json(out, payload)
    log(
        f"[freeze] pred={payload['valid_prediction_freeze_sha256'][:12]} "
        f"mech={payload['mechanism_sha256'][:12]}"
    )
    return 0


# ---------------------------------------------------------------------------
# score (single valid label read)
# ---------------------------------------------------------------------------


def _decision_primary(
    practical: Mapping[str, Mapping[str, Any]],
    consistent: Mapping[str, bool],
) -> tuple[str, str]:
    s_ok = bool(practical["S"]["pass"])
    t_ok = bool(practical["T"]["pass"])
    r_ok = bool(practical["R"]["pass"])
    t_s = bool(consistent["T_vs_S"])
    r_s = bool(consistent["R_vs_S"])
    r_t = bool(consistent["R_vs_T"])
    if s_ok and not t_s and not r_s:
        return (
            "S_CAPACITY_READOUT_DOMINATES",
            "S passes the practical line while T and R show no consistent extra advantage over S; "
            "read this as a readout/capacity reallocation effect, not as evidence that attention is useful.",
        )
    if r_ok and r_s and r_t:
        return (
            "R_STRUCTURAL_RELATION_CANDIDATE",
            "R passes the practical line and is consistently positive over both S and T; this supports "
            "further study of whether the structural relation bias improves composition. It is not a "
            "novelty or causal claim.",
        )
    if t_ok and t_s and not r_t:
        return (
            "T_OBJECT_ATTENTION_NO_EXTRA_STRUCTURE",
            "T passes the practical line and is consistently positive over S, while R is not consistently "
            "better than T; this supports ordinary object interaction, not an extra structural-bias effect.",
        )
    if r_ok and r_s and not r_t:
        return (
            "R_STRUCTURE_ADDS_OVER_S_NOT_T",
            "R passes the practical line and is consistently positive over S but not over T; the evidence "
            "does not separate the structural bias from the ordinary object-attention mechanism.",
        )
    if t_ok and not t_s:
        return (
            "T_OBJECT_ATTENTION_NO_GAIN_OVER_S",
            "T passes the practical line vs P3 but is not consistently better than S; the gain cannot be "
            "attributed to object attention rather than the shared readout/capacity change.",
        )
    if r_ok and not r_s:
        return (
            "R_STRUCTURE_NO_GAIN_OVER_S",
            "R passes the practical line vs P3 but is not consistently better than S; the structural bias "
            "is not separated from the shared readout/capacity change.",
        )
    if not (s_ok or t_ok or r_ok):
        return (
            "NO_OBJECT_ATTENTION_PRACTICAL_GAIN",
            "No arm reaches the pre-registered practical line against P3. Close this object-attention "
            "architecture screen; do not add layers, heads, relation buckets or seeds to rescue it.",
        )
    return (
        "INCONCLUSIVE_MIXED_CONTRASTS",
        "At least one arm reaches the practical line, but the pre-registered specificity contrasts are "
        "mixed. Report this as not-detected / unresolved, not as equivalence.",
    )


def cmd_score(args: argparse.Namespace) -> int:
    recover = bool(getattr(args, "recover", False))
    pre_path = C.OUT / "pre_score_freeze.json"
    if not pre_path.exists():
        raise SystemExit("pre-score freeze is missing; run freeze first")
    pre = C.read_json(pre_path)
    pred_freeze_path = REPO / pre["valid_prediction_freeze_path"]
    mech_path = REPO / pre["mechanism_path"]
    if C.sha256_file(pred_freeze_path) != pre["valid_prediction_freeze_sha256"]:
        raise SystemExit("valid prediction freeze changed after pre-score freeze")
    if C.sha256_file(mech_path) != pre["mechanism_sha256"]:
        raise SystemExit("mechanism record changed after pre-score freeze")

    valid_objs = C.load_objects("objects_valid")
    pred_freeze = C.read_json(pred_freeze_path)
    preds: dict[str, dict[int, np.ndarray]] = {}
    pred_hashes: dict[str, dict[str, str]] = {}
    for arm in C.ARMS:
        preds[arm] = {}
        pred_hashes[arm] = {}
        for seed in C.SEEDS:
            entry = pred_freeze["predictions"][arm][str(seed)]
            path = REPO / entry["prediction"]["path"]
            if C.sha256_file(path) != entry["prediction"]["file_sha256"]:
                raise SystemExit(f"prediction file changed for {arm} s{seed}")
            pred = _read_pred(path)
            if C.sha256_array(pred) != entry["prediction"]["array_sha256_float32"]:
                raise SystemExit(f"prediction array hash mismatch for {arm} s{seed}")
            preds[arm][seed] = pred
            pred_hashes[arm][str(seed)] = entry["prediction"]["array_sha256_float32"]

    # Historical P3 reference predictions (label-free files, exposed development split).
    p3_preds: dict[int, np.ndarray] = {}
    p3_prov: dict[str, Any] = {}
    for seed in C.SEEDS:
        p3_preds[seed], p3_prov[str(seed)] = C.historical_valid_pred(seed)
    preds["P3"] = p3_preds

    lock_path = C.OUT / "valid_read.lock"
    if lock_path.exists() and not recover:
        raise SystemExit("valid_read.lock exists; score refuses a second read without --recover")
    lock_payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": _now(),
        "read_count": 1,
        "reread": False,
        "recover": recover,
        "pre_score_freeze_sha256": C.sha256_file(pre_path),
        "valid_prediction_freeze_sha256": pre["valid_prediction_freeze_sha256"],
        "mechanism_sha256": pre["mechanism_sha256"],
        "prediction_array_sha256_float32": pred_hashes,
        "split": "official PyG ZINC subset=True val, rows 0..999",
        "official_test_loaded": False,
    }
    if lock_path.exists() and recover:
        old = C.read_json(lock_path)
        C.write_json(
            C.OUT / "score_recovery_deviation.json",
            {
                "protocol_id": C.PROTOCOL_ID,
                "created_utc": _now(),
                "reason": "score --recover after a consumed read;",
                "old_lock": old,
                "note": "Reported as a deviation; the pre-registered decision remains the one written after this read.",
            },
        )
        lock_payload["read_count"] = int(old.get("read_count", 1)) + 1
        lock_payload["reread"] = True
        C.write_json(lock_path, lock_payload)
    else:
        C.write_once_json(lock_path, lock_payload)

    # The single label read of this round.
    y = C.load_valid_y()
    if y.shape != (C.N_VALID,):
        raise SystemExit(f"valid labels shape {y.shape} != {(C.N_VALID,)}")

    # P3 reference and label-free replay consistency.
    expected_p3 = C.load_config()["audit_expectations"]["historical_valid_mae"]
    p3_metrics = {}
    replay_checks = {}
    for seed in C.SEEDS:
        m = C.abs_metrics(p3_preds[seed], y)
        p3_metrics[str(seed)] = m
        if abs(m["mae"] - float(expected_p3[str(seed)])) > 1e-5:
            raise SystemExit(
                f"historical P3 MAE does not reproduce for seed {seed}: "
                f"{m['mae']} vs {expected_p3[str(seed)]}"
            )
        replay_path = C.OUT / f"p3_valid_replay_s{seed}.npz"
        if not replay_path.exists():
            raise SystemExit(f"missing P3 replay for seed {seed}")
        replay = _read_pred(replay_path)
        replay_checks[str(seed)] = {
            "max_abs_replay_minus_historical": float(np.max(np.abs(replay - p3_preds[seed]))),
            "replay_file_sha256": C.sha256_file(replay_path),
        }
    p3_mean = float(np.mean([p3_metrics[str(s)]["mae"] for s in C.SEEDS]))

    arm_metrics: dict[str, Any] = {}
    for arm in C.ARMS:
        per_seed = {str(s): C.abs_metrics(preds[arm][s], y) for s in C.SEEDS}
        arm_metrics[arm] = {
            "per_seed": per_seed,
            "seed_mean_mae": float(np.mean([per_seed[str(s)]["mae"] for s in C.SEEDS])),
            "seed_mean_rmse": float(np.mean([per_seed[str(s)]["rmse"] for s in C.SEEDS])),
        }

    p3_worst = C.worst_indices(p3_preds, y, C.TOP_K_REMOVED)
    contrast_pairs = [
        ("S", "P3"),
        ("T", "P3"),
        ("R", "P3"),
        ("T", "S"),
        ("R", "S"),
        ("R", "T"),
    ]
    contrasts = {}
    for cand, ref in contrast_pairs:
        key = f"{cand}_vs_{ref}"
        contrasts[key] = C.contrast_payload(cand, ref, preds, y, p3_worst)

    practical = {arm: C.practical_pass(contrasts[f"{arm}_vs_P3"]) for arm in C.ARMS}
    consistent = {key: C.consistent_positive(contrasts[key]) for key in ("T_vs_S", "R_vs_S", "R_vs_T")}
    primary, interpretation = _decision_primary(practical, consistent)

    masks = {
        "ordinary_gt_-3": np.asarray(y, dtype=np.float64) > C.TAIL_THRESHOLD,
        "tail_le_-3": np.asarray(y, dtype=np.float64) <= C.TAIL_THRESHOLD,
        "drop_p3_worst10": np.ones(y.size, dtype=bool),
        "without_row172": np.ones(y.size, dtype=bool),
    }
    masks["drop_p3_worst10"][p3_worst] = False
    if C.EXTREME_VALID_ROW < y.size:
        masks["without_row172"][C.EXTREME_VALID_ROW] = False
    longest, cycle_penalty = C.cycle_proxy(valid_objs)
    masks["cycle_penalty_proxy"] = cycle_penalty
    masks["no_cycle_penalty_proxy"] = ~cycle_penalty
    subgroups = C.subgroup_metrics(preds, y, masks)

    metrics = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": _now(),
        "split": "official PyG ZINC subset=True val (development, exposed; not a blind test)",
        "n": C.N_VALID,
        "official_test_loaded": False,
        "p3_reference": {
            "provenance": p3_prov,
            "per_seed": p3_metrics,
            "seed_mean_mae": p3_mean,
            "expected_per_seed": expected_p3,
            "replay_checks": replay_checks,
        },
        "arms": arm_metrics,
        "contrasts": contrasts,
        "p3_worst10_rows": p3_worst,
        "extreme_valid_row": C.EXTREME_VALID_ROW,
        "subgroups": subgroups,
        "cycle_proxy": {
            "definition": "max(len(c) for c in networkx.cycle_basis(G)) > 6; order-sensitive, reporting only",
            "n_penalised": int(cycle_penalty.sum()),
            "max_longest_cycle": int(longest.max()) if longest.size else None,
        },
        "bootstrap": {
            "resamples": C.BOOTSTRAP_N,
            "seed": C.BOOTSTRAP_SEED,
            "unit": "molecule; two seeds resampled synchronously",
            "note": "CI describes sample fluctuation for these two training instances only; it does not cover all training randomness.",
        },
        "note": "Positive gain = reference error minus candidate error; lower MAE is better.",
    }
    C.write_once_json(C.OUT / "valid_metrics.json", metrics)

    decision = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": _now(),
        "stage": "score",
        "primary_label": primary,
        "interpretation": interpretation,
        "practical_line": {
            "mean_gain": C.PRACTICAL_MEAN_GAIN,
            "both_seeds_positive": True,
            "ordinary_and_drop_p3_worst10_positive": True,
            "paired_ci_low_gt_zero": True,
        },
        "practical": practical,
        "consistent_positive_contrasts": consistent,
        "labels": [primary],
        "not_equivalence_note": (
            "A non-significant contrast is reported as 'no detected gain', never as equivalence; "
            "the CI only covers molecule sampling for these two training instances."
        ),
        "claims_note": (
            "This is an exploratory screen on an already-exposed official-valid split. It is not a "
            "blind test, not a novelty claim, and not a comparison to GRIT/ESA+ official-test numbers."
        ),
    }
    C.write_once_json(C.OUT / "valid_decision.json", decision)

    # Run manifest (hashes only; arrays stay local / on the compute host).
    manifest_path = C.OUT / "run_manifest.tsv"
    if not manifest_path.exists():
        lines = [
            "stage\tarm\tseed\tupdates\tsoup_sha256\tprediction_array_sha256\tseed_mae\twall_seconds\tbatch_order_sha256\tgit_commit"
        ]
        for seed in C.SEEDS:
            lines.append(
                "reference\tP3\t{}\t0\t{}\t{}\t{:.17g}\t0\t{}\t{}".format(
                    seed,
                    C.p3_soup_sha_expected(seed),
                    p3_prov[str(seed)]["array_sha256_float32"],
                    p3_metrics[str(seed)]["mae"],
                    C.load_config()["audit_expectations"]["batch_order_sha256"][str(seed)],
                    _git_commit(),
                )
            )
        for arm in C.ARMS:
            for seed in C.SEEDS:
                train_rec = C.read_json(_train_path(arm, seed))
                freeze = C.read_json(_pred_freeze_path(arm, seed))
                lines.append(
                    "train+predict\t{}\t{}\t{}\t{}\t{}\t{:.17g}\t{:.1f}\t{}\t{}".format(
                        arm,
                        seed,
                        train_rec["updates"],
                        train_rec["soup_sha256"],
                        freeze["prediction"]["array_sha256_float32"],
                        arm_metrics[arm]["per_seed"][str(seed)]["mae"],
                        train_rec["wall_seconds"],
                        train_rec["batch_order"]["full_hash"],
                        _git_commit(),
                    )
                )
        manifest_path.write_text("\n".join(lines) + "\n")

    # Access log: this round's single score read plus the earlier pre-freeze probe.
    access_log = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": _now(),
        "entries": [
            {
                "when": "before protocol freeze (2026-10-19 UTC)",
                "stage": "local environment probe",
                "what": "called the frozen valid-label loader once to check shape / hash plumbing; printed shape and three values",
                "used_for_architecture_or_selection": False,
                "model_predictions_read": False,
            },
            {
                "when": _now(),
                "stage": "score",
                "what": "single official-valid label read for the frozen six predictions",
                "read_count": int(lock_payload["read_count"]),
                "reread": bool(lock_payload["reread"]),
            },
        ],
        "note": (
            "The official-valid split was already exposed in earlier rounds. This round records its own "
            "accesses and does not claim that the whole research program has only one valid access."
        ),
        "official_test_loaded": False,
    }
    C.write_json(C.OUT / "valid_access_log.json", access_log)

    log(
        f"[score] P3={p3_mean:.6f} "
        + " ".join(f"{a}={arm_metrics[a]['seed_mean_mae']:.6f}" for a in C.ARMS)
        + f" primary={primary}"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_prep(args: argparse.Namespace) -> int:
    cmd_audit(argparse.Namespace(device=args.device))
    cmd_gates(argparse.Namespace(device="cpu"))
    cmd_flops(argparse.Namespace(device="cpu"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(C.OUT) if hasattr(C, "OUT") else None)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("audit", "gates", "flops", "prep"):
        a = sub.add_parser(name)
        a.add_argument("--device", default="cpu")
    s = sub.add_parser("smoke")
    s.add_argument("--device", default="cuda")
    t = sub.add_parser("train")
    t.add_argument("--arm", required=True, choices=list(C.ARMS))
    t.add_argument("--seed", required=True, type=int, choices=list(C.SEEDS))
    t.add_argument("--device", default="cuda")
    rs = sub.add_parser("run-seed")
    rs.add_argument("--seed", required=True, type=int, choices=list(C.SEEDS))
    rs.add_argument("--order", default="S,T,R")
    rs.add_argument("--device", default="cuda")
    pa = sub.add_parser("predict-all")
    pa.add_argument("--device", default="cpu")
    m = sub.add_parser("mechanism")
    m.add_argument("--device", default="cpu")
    f = sub.add_parser("freeze")
    f.add_argument("--device", default="cpu")
    sc = sub.add_parser("score")
    sc.add_argument("--recover", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "out", None):
        C.OUT = Path(args.out).resolve()
    C.OUT.mkdir(parents=True, exist_ok=True)
    return {
        "audit": cmd_audit,
        "gates": cmd_gates,
        "flops": cmd_flops,
        "prep": cmd_prep,
        "smoke": cmd_smoke,
        "train": cmd_train,
        "run-seed": cmd_run_seed,
        "predict-all": cmd_predict_all,
        "mechanism": cmd_mechanism,
        "freeze": cmd_freeze,
        "score": cmd_score,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
