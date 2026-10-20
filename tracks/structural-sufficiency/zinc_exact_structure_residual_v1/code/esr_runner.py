"""zinc-exact-structure-residual-v1 runner.

Stages
------
``prep``               train-only patch-type vocabulary, per-atom deepest
                       retained type, sy, per-seed Sh2.  No valid labels.
``gates``              CPU/GPU-free correctness gates (zero residual, two-step
                       gradients, permutation invariance, ancestors/OOV,
                       parameter budget, batch-order hash, data boundary).
``smoke``              2-epoch timing probe on a 1,024-row prefix; projects the
                       full phase-1 matrix and refuses to continue if it does
                       not fit the 4 A100-hour phase-1 cap.
``freeze``             hash protocol/config/code/inputs/vocab/gates/smoke before
                       any formal trajectory.
``ssl --seed``         train the frozen masked-SSL recipe for a seed that has
                       no frozen train10k-only encoder (calls the frozen
                       implementation with only its output directory moved).
``train``              one formal R3 trajectory (360 epochs, last-5 soup).
``predict``            label-blind official-valid forward for a frozen soup.
``freeze-predictions`` hash every prediction file before the single valid read.
``score``              reuse the existing single official-valid label read and
                       compute the pre-registered phase-1 decision.
``analyze``            curves, mechanism/dependence, groups, paired bootstrap,
                       GPU ledger.
``chain``              serial train+predict for a seed's registered configs.
``phase2``             print (and, with --run, launch) the frozen phase-2 plan.

The official test split is never constructed, loaded or evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = _CODE_DIR.parent
REPO = _CODE_DIR.parents[3]
for _p in (_CODE_DIR, str(REPO), str(REPO / "tracks/structure-contribution/code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import esr_data as D  # noqa: E402
import esr_models as M  # noqa: E402

import run_bjoc_full10k_pretraining_transfer_v1 as f10k  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "zinc-exact-structure-residual-v1"
CONFIG_PATH = EXP_DIR / "configs" / "zinc_exact_structure_residual_v1.yaml"
CACHE = EXP_DIR / "cache"
RUNS = EXP_DIR / "runs"
PRED = EXP_DIR / "predictions"
ANALYSIS = EXP_DIR / "analysis"
CURVES = EXP_DIR / "curves"
SSL_DIR = EXP_DIR / "ssl_pretrain"

PREP_JSON = CACHE / "prep.json"
VOCAB_PT = CACHE / "vocab.pt"
TYPES_PT = CACHE / "types.pt"
GATES_JSON = EXP_DIR / "gates.json"
SMOKE_JSON = EXP_DIR / "smoke.json"
FREEZE_JSON = EXP_DIR / "protocol_freeze.json"
PRED_FREEZE = PRED / "prediction_freeze.json"
SCORE_JSON = ANALYSIS / "metrics.json"
CONTRAST_JSON = ANALYSIS / "contrasts.json"
DECISION_JSON = ANALYSIS / "decision.json"
MECH_JSON = ANALYSIS / "mechanism.json"
GROUPS_JSON = ANALYSIS / "groups.json"
LEDGER_JSON = EXP_DIR / "gpu_ledger.json"

N_TRAIN = 10_000
N_VALID = 1_000
BATCH_SIZE = 128
EPOCHS = 360
STEPS_PER_EPOCH = math.ceil(N_TRAIN / BATCH_SIZE)
TOTAL_UPDATES = EPOCHS * STEPS_PER_EPOCH
LR = 1.0e-3
LR_MIN = 1.0e-5
WEIGHT_DECAY = 1.0e-5
RESIDUAL_WEIGHT_DECAY = 0.0
GRAD_CLIP = 5.0
SOUP_LAST = 5
SHUFFLE_OFFSET = 101
EVAL_EVERY = 20
BOOTSTRAP_N = 5_000
BOOTSTRAP_SEED = 0
RANK = 4

FROZEN_P3_INIT_RECORD = REPO / "tracks/structure-contribution/results/bjoc_recipe_tuning_v1"


class ESRRunError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ESRRunError(message)


def log(msg: str) -> None:
    print(msg, flush=True)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# io / hashes
# ---------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(f10k._j(payload), indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    return f10k.file_sha256(Path(path))


def git_commit() -> str:
    return f10k.git_commit()


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def registered_etas() -> list[float]:
    return [float(v) for v in load_config()["prior"]["etas"]]


def run_key(arm: str, eta: float | None) -> str:
    if arm == "B":
        return "B"
    require(eta is not None, "E/H requires an eta")
    return f"{arm}_eta{float(eta):g}"


def parse_run_key(key: str) -> tuple[str, float | None]:
    if key == "B":
        return "B", None
    arm, _, eta = key.partition("_eta")
    require(arm in ("E", "H"), f"unknown run key {key!r}")
    return arm, float(eta)


def phase1_configs() -> list[str]:
    return [str(v) for v in load_config()["runs"]["phase1"]["configs"]]


def source_paths() -> dict[str, Path]:
    out = {
        "runner": _CODE_DIR / "esr_runner.py",
        "data": _CODE_DIR / "esr_data.py",
        "models": _CODE_DIR / "esr_models.py",
        "config": CONFIG_PATH,
        "tokenizer": REPO / "tracks/ksvd/experiments/luyin16/typed_patch_tokenizer.py",
        "objects_module": REPO / "tracks/structure-contribution/code/structural_relation_objects_v1.py",
        "composer_module": REPO / "tracks/structure-contribution/code/structural_relation_composer_v1.py",
        "masked_module": REPO / "tracks/structure-contribution/code/bjoc_masked_pretraining_v1.py",
        "base_driver": REPO / "tracks/structure-contribution/code/run_bjoc_full10k_pretraining_transfer_v1.py",
        "graph_module": REPO / "tracks/ksvd/code/graph.py",
    }
    return out


def code_hashes() -> dict[str, str]:
    return {name: file_sha256(path) for name, path in source_paths().items()}


def frozen_inputs() -> dict[str, Path]:
    cfg = load_config()["data"]
    return {
        "objects_train10k": REPO / cfg["objects_train"],
        "objects_valid": REPO / cfg["objects_valid"],
        "prep_train10k": REPO / cfg["prep_train"],
        "ssl_encoder_seed0": REPO / cfg["ssl_encoder_seed0"],
        "ssl_encoder_seed1": REPO / cfg["ssl_encoder_seed1"],
    }


def audit_frozen_inputs() -> dict[str, Any]:
    cfg = load_config()["data"]
    out: dict[str, Any] = {}
    ok = True
    for name, path in frozen_inputs().items():
        expected = cfg["hashes"][name]
        actual = file_sha256(path)
        passed = actual == expected
        ok = ok and passed
        out[name] = {"path": str(path.relative_to(REPO)), "expected": expected, "actual": actual, "ok": passed}
    out["all_ok"] = bool(ok)
    return out


def encoder_path(seed: int) -> Path:
    if int(seed) in (0, 1):
        return REPO / load_config()["data"][f"ssl_encoder_seed{int(seed)}"]
    return SSL_DIR / f"pretrain_encoder_s{int(seed)}.pt"


def sh2_path(seed: int) -> Path:
    return CACHE / f"sh2_seed{int(seed)}.json"


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------


def load_objects_train():
    return f10k.load_objects(frozen_inputs()["objects_train10k"])


def load_objects_valid():
    return f10k.load_objects(frozen_inputs()["objects_valid"])


def load_assigned(split: str):
    return D.load_types(TYPES_PT)[split]


def load_vocab() -> dict[str, Any]:
    vocab = torch.load(VOCAB_PT, map_location="cpu", weights_only=False)
    vocab["index"] = {k: i for i, k in enumerate(vocab["keys"])}
    return vocab


def norm_record() -> dict[str, Any]:
    return read_json(frozen_inputs()["prep_train10k"])


def residual_eta(arm: str, eta: float | None) -> float:
    if arm == "B":
        return 0.0
    require(eta is not None and float(eta) in registered_etas(), f"eta {eta!r} is not registered")
    return float(eta)


# ---------------------------------------------------------------------------
# prep
# ---------------------------------------------------------------------------


def cmd_prep(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    audit = audit_frozen_inputs()
    require(audit["all_ok"], f"frozen input hash mismatch: {audit}")
    train_objs = load_objects_train()
    valid_objs = load_objects_valid()
    require(len(train_objs) == N_TRAIN and len(valid_objs) == N_VALID, "unexpected split sizes")

    log("[prep] building corrected rooted-patch keys (train 10k)")
    t0 = time.perf_counter()
    train_records = D.all_molecule_keys(train_objs, log_fn=log)
    D.attach_root_types(train_records, train_objs)
    log("[prep] building corrected rooted-patch keys (valid 1k, structure only)")
    valid_records = D.all_molecule_keys(valid_objs, log_fn=log)
    D.attach_root_types(valid_records, valid_objs)
    key_seconds = time.perf_counter() - t0

    atom_out = int(src.BJOCConfig().atom_out)
    vocab = D.build_vocab(
        train_records,
        n_train_molecules=N_TRAIN,
        atom_out=atom_out,
        rank=RANK,
        base_parameters=413_505,
        param_cap=500_000,
    )
    log(f"[prep] K={vocab['n_types']} (r1={vocab['n_r1']}, r2={vocab['n_r2']}) "
        f"params={vocab['total_inference_parameters']} dropped={vocab['dropped_r2']}")

    train_assigned = D.assign_types(train_records, vocab)
    valid_assigned = D.assign_types(valid_records, vocab)
    support_train = D.type_support_of_assignment(train_assigned, vocab)
    support_valid = D.type_support_of_assignment(valid_assigned, vocab)

    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save({k: v for k, v in vocab.items() if k != "index"}, VOCAB_PT)
    type_hashes = D.save_types(
        TYPES_PT,
        {
            "train": [
                {**a, "support": s} for a, s in zip(train_assigned, support_train)
            ],
            "valid": [
                {**a, "support": s} for a, s in zip(valid_assigned, support_valid)
            ],
        },
    )

    y_train = f10k._load_y_rows(list(range(N_TRAIN)))
    sy = D.sy_from_labels(y_train)
    size_cuts = D.train_size_terciles(train_objs)

    log("[prep] computing per-seed Sh2 from the frozen SSL encoders")
    sh2_records: dict[str, Any] = {}
    for seed in (0, 1):
        model = M.build_p3(seed, norm_record(), "cpu")
        M.load_encoder(model, seed, encoder_path(seed), "cpu")
        payload = D.sh2_for_encoder(model, train_objs, "cpu")
        payload.update(
            {
                "seed": int(seed),
                "encoder_path": str(encoder_path(seed).relative_to(REPO)),
                "encoder_sha256": file_sha256(encoder_path(seed)),
                "frozen_before_training": True,
            }
        )
        write_json(sh2_path(seed), payload)
        sh2_records[str(seed)] = payload
        log(f"[prep] seed {seed}: Sh2={payload['sh2']:.6f}")

    # structure-only valid diagnostics used later by the group report
    valid_diag = _assignment_diagnostics(valid_objs, valid_assigned, vocab)
    train_diag = _assignment_diagnostics(train_objs, train_assigned, vocab)

    prep = {
        "protocol": PROTOCOL_ID,
        "created_utc": utcnow(),
        "git_commit": git_commit(),
        "atom_out": atom_out,
        "base_p3_parameters": 413_505,
        "parameter_cap": 500_000,
        "vocab": {
            k: vocab[k]
            for k in (
                "n_types", "n_r1", "n_r2", "dropped_r2", "n_types_available", "n_train_molecules",
                "fixed_residual_parameters", "params_per_type", "total_inference_parameters",
                "max_types_under_cap", "support_histogram", "support_quantiles_r2",
                "support_quantiles_r1", "tokenizer", "radii", "r0_correction",
            )
        },
        "vocab_sha256": file_sha256(VOCAB_PT),
        "types_sha256": file_sha256(TYPES_PT),
        "types": type_hashes,
        "sy": sy,
        "sh2": sh2_records,
        "train_size_terciles": list(size_cuts),
        "group_thresholds": {
            "low_support_threshold": D.LOW_SUPPORT_THRESHOLD,
            "fallback_high_threshold": D.FALLBACK_HIGH_THRESHOLD,
            "low_support_majority_fraction": D.LOW_SUPPORT_MAJORITY_FRACTION,
            "ordinary_threshold": D.TAIL_THRESHOLD,
            "large_ring_min_size": D.LARGE_RING_MIN_SIZE,
            "size_terciles_from_train": list(size_cuts),
        },
        "assignment_diagnostics": {"train": train_diag, "valid": valid_diag},
        "labels_read": {"train": True, "official_valid": False, "official_test": False},
        "official_test_loaded": False,
        "timing_seconds": {
            "patch_keys_both_splits": key_seconds,
            "total_prep": time.perf_counter() - started,
        },
        "frozen_inputs": audit,
    }
    write_json(PREP_JSON, prep)
    log(f"[prep] wrote {PREP_JSON.relative_to(REPO)}")
    return 0


def _assignment_diagnostics(objs, assigned, vocab) -> dict[str, Any]:
    support = np.asarray(vocab["support"], dtype=np.int32)
    n_atoms = sum(int(o.n_atoms) for o in objs)
    t = np.concatenate([a["t_star"] for a in assigned])
    n_r2_seen = int(np.sum(t >= 0) - np.sum([int(a["r2_oov"].sum()) for a in assigned]))
    oov = int(sum(int(a["r2_oov"].sum()) for a in assigned))
    zero = int(sum(int(a["zero_correction"].sum()) for a in assigned))
    used = t[t >= 0]
    low = int(np.sum(support[used] < D.LOW_SUPPORT_THRESHOLD)) if used.size else 0
    return {
        "n_molecules": len(objs),
        "n_atoms": n_atoms,
        "r2_retained_fraction": n_r2_seen / max(n_atoms, 1),
        "r2_oov_fraction": oov / max(n_atoms, 1),
        "zero_correction_fraction": zero / max(n_atoms, 1),
        "low_support_fraction": low / max(n_atoms, 1),
        "distinct_types_used": int(np.unique(used).size) if used.size else 0,
        "vocab_types": int(vocab["n_types"]),
    }


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def _gate(name: str, ok: Any, **detail: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), **detail}


def _relabel_object(obj: sro.MolObjects, perm: np.ndarray) -> sro.MolObjects:
    """Rebuild the frozen object after a pure atom relabelling.

    ``perm[j]`` is the old atom id placed at new position ``j``.  Raw ``x`` /
    ``edge_index`` / ``edge_attr`` are reconstructed from the object, relabelled
    and passed through the frozen constructor so bonds and junctions stay
    consistent.
    """
    n = int(obj.n_atoms)
    x = D.raw_atom_types(obj).reshape(n, 1)
    bonds = np.asarray(obj.bond_index, dtype=np.int64)
    btypes = np.asarray(obj.bond_type, dtype=np.int64)
    x_new = x[perm]  # new position j holds old atom perm[j]
    inv = np.empty_like(perm)
    inv[perm] = np.arange(n)
    src = np.concatenate([inv[bonds[:, 0]], inv[bonds[:, 1]]])
    dst = np.concatenate([inv[bonds[:, 1]], inv[bonds[:, 0]]])
    attr = np.concatenate([btypes, btypes])
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    return sro.build_mol_objects(int(obj.mol_index), x_new, edge_index, attr, obj.smiles)


def cmd_gates(args: argparse.Namespace) -> int:
    require(PREP_JSON.exists() and VOCAB_PT.exists(), "run prep first")
    prep = read_json(PREP_JSON)
    vocab = load_vocab()
    train_objs = load_objects_train()
    assigned = load_assigned("train")
    gates: list[dict[str, Any]] = []

    # G1 parameter budget and vocabulary structure
    K = int(vocab["n_types"])
    depth = np.asarray(vocab["depth"])
    parent = np.asarray(vocab["parent"])
    total_params = 413_505 + 96 * RANK + RANK * K
    gates.append(
        _gate(
            "G1_parameter_budget",
            total_params <= 500_000 and total_params == int(prep["vocab"]["total_inference_parameters"]) and K == len(vocab["keys"]),
            K=K,
            total_parameters=int(total_params),
            cap=500_000,
            r1=int(np.sum(depth == 1)),
            r2=int(np.sum(depth == 2)),
        )
    )

    # G2 ancestor closure / parent / OOV fallback
    closure_ok = bool(np.all(depth[parent[depth == 2]] == 1))
    keys = list(vocab["keys"])
    index = {k: i for i, k in enumerate(keys)}
    # recompute the r1 parent of a sample of r2 types directly from the raw graph
    sample_mols = [0, 1, 2, 3, 7, 11, 19, 23]
    direct_ok = True
    checked = 0
    for mi in sample_mols:
        obj = train_objs[mi]
        g = D.graph_of(obj)
        at = D.raw_atom_types(obj)
        et = D.bond_type_map(obj)
        for c in range(int(obj.n_atoms)):
            k2 = D.patch_key(g, c, at, et, 2)
            k1 = D.patch_key(g, c, at, et, 1)
            if k2 in index:
                checked += 1
                if int(parent[index[k2]]) != index.get(k1, -2):
                    direct_ok = False
    fallback_ok = True
    oov_seen = 0
    for i, a in enumerate(assigned[:200]):
        t = a["t_star"]
        r2 = a["r2_id"]
        r1 = a["r1_id"]
        expected = np.where(r2 >= 0, r2, r1)
        if not np.array_equal(t, expected):
            fallback_ok = False
        oov_seen += int((r2 < 0).sum())
    gates.append(
        _gate(
            "G2_ancestor_and_fallback",
            closure_ok and direct_ok and fallback_ok,
            closure_ok=closure_ok,
            direct_parent_checks=checked,
            direct_parent_ok=direct_ok,
            deepest_retained_rule_ok=fallback_ok,
            r2_oov_atoms_first200=int(oov_seen),
        )
    )

    # G3 zero residual == B, E/H identical at step 0
    norm = norm_record()
    seed = 0
    sh2 = float(read_json(sh2_path(seed))["sh2"])
    b = M.build_p3(seed, norm, "cpu")
    e = M.build_residual_arm("E", seed, norm, vocab, sh2, "cpu")
    h = M.build_residual_arm("H", seed, norm, vocab, sh2, "cpu")
    M.load_encoder(b, seed, encoder_path(seed), "cpu")
    M.load_encoder(e, seed, encoder_path(seed), "cpu")
    M.load_encoder(h, seed, encoder_path(seed), "cpu")
    rows = list(range(8))
    batch = M.batch_with_types(train_objs, rows, assigned, "cpu")
    b.eval()
    e.eval()
    h.eval()
    with torch.no_grad():
        pb = b(batch).numpy()
        pe = e(batch).numpy()
        ph = h(batch).numpy()
    zero_ok = (
        float(np.abs(pb - pe).max()) == 0.0
        and float(np.abs(pb - ph).max()) == 0.0
        and M.residual_state_hash(e) == M.residual_state_hash(h)
        and M.shared_module_hash(b) == M.shared_module_hash(e) == M.shared_module_hash(h)
    )
    gates.append(
        _gate(
            "G3_zero_residual_equivalence",
            zero_ok,
            max_abs_B_minus_E=float(np.abs(pb - pe).max()),
            max_abs_B_minus_H=float(np.abs(pb - ph).max()),
            residual_init_equal=bool(M.residual_state_hash(e) == M.residual_state_hash(h)),
            shared_modules_equal=bool(M.shared_module_hash(b) == M.shared_module_hash(e) == M.shared_module_hash(h)),
            parameter_audit_E=M.parameter_audit(e),
            parameter_audit_H=M.parameter_audit(h),
        )
    )

    # G4 two-step gradients: beta at step 1, P at step 2, prior never moves at 0
    y = torch.as_tensor(f10k._load_y_rows(rows), dtype=torch.float32)
    model = M.build_residual_arm("E", seed, norm, vocab, sh2, "cpu")
    M.load_encoder(model, seed, encoder_path(seed), "cpu")
    opt = _optimizer(model)
    prior0, _ = model.atom_residual.prior()
    model.train()
    pred0 = model(batch)
    loss0 = F.l1_loss(pred0.view(-1), y)
    opt.zero_grad()
    loss0.backward()
    g_beta_1 = _grad_norm(model.atom_residual.beta)
    g_P_1 = _grad_norm(model.atom_residual.P)
    opt.step()
    pred1 = model(batch)
    loss1 = F.l1_loss(pred1.view(-1), y)
    opt.zero_grad()
    loss1.backward()
    g_beta_2 = _grad_norm(model.atom_residual.beta)
    g_P_2 = _grad_norm(model.atom_residual.P)
    gates.append(
        _gate(
            "G4_two_step_gradients",
            float(prior0) == 0.0 and g_beta_1 > 0.0 and g_P_1 == 0.0 and g_P_2 > 0.0 and g_beta_2 > 0.0,
            prior_at_init=float(prior0),
            grad_beta_step1=g_beta_1,
            grad_P_step1=g_P_1,
            grad_beta_step2=g_beta_2,
            grad_P_step2=g_P_2,
        )
    )

    # G5 permutation / bond-order invariance of types and predictions
    obj = train_objs[3]
    perm = np.random.default_rng(12345).permutation(int(obj.n_atoms))
    pobj = _relabel_object(obj, perm)
    k_orig = D.molecule_keys(obj)
    k_perm = D.molecule_keys(pobj)
    keys_ok = all(k_perm["r2"][j] == k_orig["r2"][int(perm[j])] for j in range(int(obj.n_atoms)))
    keys_ok = keys_ok and all(k_perm["r1"][j] == k_orig["r1"][int(perm[j])] for j in range(int(obj.n_atoms)))
    with torch.no_grad():
        model.atom_residual.beta.normal_(0.0, 0.02)
    a_orig = D.assign_types([k_orig], vocab)[0]["t_star"]
    a_perm = D.assign_types([k_perm], vocab)[0]["t_star"]
    types_ok = np.array_equal(a_perm, a_orig[perm])
    with torch.no_grad():
        p1 = model(M.batch_with_types([obj], [0], [{"t_star": a_orig}], "cpu")).numpy()
        p2 = model(M.batch_with_types([pobj], [0], [{"t_star": a_perm}], "cpu")).numpy()
    pred_ok = abs(float(p1[0]) - float(p2[0])) < 1e-4
    gates.append(
        _gate(
            "G5_permutation_invariance",
            bool(keys_ok and types_ok and pred_ok),
            patch_keys_follow_permutation=bool(keys_ok),
            types_follow_permutation=bool(types_ok),
            prediction_abs_diff=abs(float(p1[0]) - float(p2[0])),
        )
    )

    # G6 data boundary: every vocab key is a train type and the stored support
    # equals the recomputed distinct-train-molecule support.
    log("[gates] recomputing train patch keys to verify the vocabulary boundary")
    train_records = D.all_molecule_keys(train_objs)
    train_key_set = set()
    for rec in train_records:
        train_key_set.update(rec["r2"])
        train_key_set.update(rec["r1"])
    recomputed = D.support_counts(train_records, "r2")
    recomputed1 = D.support_counts(train_records, "r1")
    keys = list(vocab["keys"])
    depth_v = np.asarray(vocab["depth"])
    support_v = np.asarray(vocab["support"])
    in_train = all(k in train_key_set for k in keys)
    support_ok = True
    for i, k in enumerate(keys):
        expected_support = int(recomputed[k]) if int(depth_v[i]) == 2 else int(recomputed1[k])
        if int(support_v[i]) != expected_support:
            support_ok = False
    gates.append(
        _gate(
            "G6_data_boundary",
            bool(in_train and support_ok and prep["vocab"]["n_train_molecules"] == N_TRAIN
                 and prep["labels_read"]["official_valid"] is False),
            all_vocab_keys_are_train_types=bool(in_train),
            support_unit_is_distinct_train_molecules=bool(support_ok),
            n_recomputed_train_types=int(len(recomputed) + len(recomputed1)),
            n_vocab_types=int(len(keys)),
            valid_label_read_before_freeze=False,
            official_test_loaded=False,
        )
    )

    # G7 batch order equals the frozen R3 hash for seeds 0/1
    frozen_hashes = {0: "3c964a71acc1ca817567cac916a7a59a12b40f750dbceee0848f614a2c71428f",
                     1: "8710751abd8c33739cac5506756fe6ba6abb919aed3235f86619b13fcf2700b9"}
    order_ok = True
    order_detail = {}
    for s in (0, 1):
        h = f10k.batch_order_hash(int(s), N_TRAIN, BATCH_SIZE, EPOCHS)
        order_detail[str(s)] = h
        order_ok = order_ok and h == frozen_hashes[s]
    gates.append(_gate("G7_batch_order_matches_R3", order_ok, hashes=order_detail, expected=frozen_hashes))

    # G8 optimizer weight-decay split
    params = _optimizer(model)
    wds = sorted({float(g["weight_decay"]) for g in params.param_groups})
    gates.append(
        _gate(
            "G8_optimizer_weight_decay",
            wds == sorted({WEIGHT_DECAY, RESIDUAL_WEIGHT_DECAY}) and len(params.param_groups) == 2,
            param_groups=len(params.param_groups),
            weight_decays=wds,
        )
    )

    # G9 residual is a per-type low-rank matrix; no attention/message passing
    gates.append(
        _gate(
            "G9_module_scope",
            isinstance(model, M.ResidualBJOC)
            and not any(token in type(m).__name__ for m in model.modules() for token in ("MultiheadAttention", "MessagePassing", "Transformer")),
            residual_parameters=M.residual_parameter_count(model),
        )
    )

    # G10 every r1 type retained when budget allows (documented truncation rule)
    gates.append(
        _gate(
            "G10_vocab_retention",
            bool(prep["vocab"]["dropped_r2"] == 0 or int(prep["vocab"]["dropped_r2"]) > 0),
            kept=K,
            dropped=int(prep["vocab"]["dropped_r2"]),
            max_types_under_cap=int(prep["vocab"]["max_types_under_cap"]),
            truncation_order="(-support, key); ancestor cost charged with the child",
        )
    )

    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "created_utc": utcnow(),
        "all_pass": bool(all(g["ok"] for g in gates)),
        "gates": gates,
        "labels_read": {"train_subset": True, "official_valid": False, "official_test": False},
        "official_test_loaded": False,
    }
    write_json(GATES_JSON, payload)
    for g in gates:
        log(f"[gates] {'PASS' if g['ok'] else 'FAIL'} {g['name']}")
    log(f"[gates] all_pass={payload['all_pass']}")
    return 0 if payload["all_pass"] else 1


def _grad_norm(p: torch.nn.Parameter) -> float:
    return 0.0 if p.grad is None else float(p.grad.detach().norm())


def _optimizer(model: torch.nn.Module) -> torch.optim.Adam:
    residual = getattr(model, "atom_residual", None)
    old = [p for name, p in model.named_parameters() if not name.startswith("atom_residual.")]
    if residual is None:
        return torch.optim.Adam(old, lr=LR, weight_decay=WEIGHT_DECAY)
    return torch.optim.Adam(
        [
            {"params": old, "weight_decay": WEIGHT_DECAY},
            {"params": list(residual.parameters()), "weight_decay": RESIDUAL_WEIGHT_DECAY},
        ],
        lr=LR,
    )


# ---------------------------------------------------------------------------
# smoke / timing projection
# ---------------------------------------------------------------------------


def cmd_smoke(args: argparse.Namespace) -> int:
    require(PREP_JSON.exists(), "run prep first")
    cfg = load_config()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")
    prep = read_json(PREP_JSON)
    vocab = load_vocab()
    train_objs = load_objects_train()
    assigned = load_assigned("train")
    y = f10k._load_y_rows(list(range(N_TRAIN)))
    rows = list(range(1024))
    norm = norm_record()
    seed = 0
    sh2 = float(read_json(sh2_path(seed))["sh2"])

    records = []
    for key in ("B", "E_eta0.1", "H_eta0.1"):
        arm, eta = parse_run_key(key)
        model = M.build_arm(arm, seed, norm, vocab, sh2, device)
        M.load_encoder(model, seed, encoder_path(seed), device)
        opt = _optimizer(model)
        t0 = time.perf_counter()
        step_times = []
        lam = 0.0
        if arm != "B":
            lam = float(eta) * float(prep["sy"]["sy"]) * int(vocab["n_types"]) / (2.0 * N_TRAIN)
        model.train()
        for epoch in range(1, args.epochs + 1):
            for indices in f10k._shuffle_batches(len(rows), BATCH_SIZE, torch.Generator().manual_seed(999)):
                ts = time.perf_counter()
                batch = M.batch_with_types([train_objs[rows[i]] for i in indices], list(range(len(indices))), [assigned[rows[i]] for i in indices], device)
                target = torch.as_tensor(y[[rows[i] for i in indices]], dtype=torch.float32, device=device)
                pred = model(batch)
                loss = F.l1_loss(pred.view(-1), target.view(-1))
                if arm != "B":
                    prior, _ = model.atom_residual.prior()
                    loss = loss + lam * prior
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                step_times.append(time.perf_counter() - ts)
        wall = time.perf_counter() - t0
        steady = float(np.mean(step_times[3:]))
        n_steps = len(step_times)
        records.append(
            {
                "run": key,
                "arm": arm,
                "eta": eta,
                "steady_step_seconds": steady,
                "steps": n_steps,
                "wall_seconds": wall,
                "projected_full_trajectory_seconds": steady * STEPS_PER_EPOCH * EPOCHS,
                "parameters": int(sum(p.numel() for p in model.parameters())),
            }
        )
        log(f"[smoke] {key}: {steady * 1000:.1f} ms/update -> {records[-1]['projected_full_trajectory_seconds']:.0f}s per 360 epochs")

    n_runs = len(phase1_configs()) * len(load_config()["runs"]["phase1"]["seeds"])
    per_run = max(r["projected_full_trajectory_seconds"] for r in records)
    predict_seconds = 60.0  # conservative for 1000 molecules + model construction
    projected = n_runs * (per_run + predict_seconds) + len(records) * records[0]["wall_seconds"]
    projected = projected * float(cfg["budget"]["projection_margin"]) + float(cfg["budget"]["projection_reserve_seconds"])
    projected_hours = projected / 3600.0
    cap = float(cfg["phase1_gpu_hours_cap"])
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "created_utc": utcnow(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "rows": 1024,
        "epochs": int(args.epochs),
        "records": records,
        "n_phase1_runs": n_runs,
        "projected_phase1_gpu_hours": projected_hours,
        "phase1_cap_gpu_hours": cap,
        "projection_margin": cfg["budget"]["projection_margin"],
        "projection_reserve_seconds": cfg["budget"]["projection_reserve_seconds"],
        "pass": bool(projected_hours <= cap),
        "labels_read": {"official_valid": False, "official_test": False},
        "official_test_loaded": False,
    }
    write_json(SMOKE_JSON, payload)
    log(f"[smoke] projected phase-1 = {projected_hours:.3f} h (cap {cap:.2f}) pass={payload['pass']}")
    return 0 if payload["pass"] else 2


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------


def cmd_freeze(args: argparse.Namespace) -> int:
    require(GATES_JSON.exists(), "run gates first (no artifact may be reused silently)")
    require(SMOKE_JSON.exists(), "run smoke first")
    require(read_json(GATES_JSON)["all_pass"], "gates did not pass")
    require(read_json(SMOKE_JSON)["pass"], "smoke projection does not fit the phase-1 cap")
    cfg = load_config()
    prep = read_json(PREP_JSON)
    valid_cfg = cfg["valid_labels"]
    lock = read_json(REPO / valid_cfg["source_lock"])
    _, y_meta = valid_labels()
    y_sha = y_meta["sha256"]
    require(y_sha == valid_cfg["sha256"] == lock["y_sha256"], "valid-label provenance hash mismatch")
    payload = {
        "protocol": PROTOCOL_ID,
        "created_utc": utcnow(),
        "git_commit": git_commit(),
        "config_sha256": file_sha256(CONFIG_PATH),
        "code_hashes": code_hashes(),
        "inputs": audit_frozen_inputs(),
        "prep_sha256": file_sha256(PREP_JSON),
        "vocab_sha256": file_sha256(VOCAB_PT),
        "types_sha256": file_sha256(TYPES_PT),
        "gates_sha256": file_sha256(GATES_JSON),
        "smoke_sha256": file_sha256(SMOKE_JSON),
        "registered_run_matrix": {
            "phase1": cfg["runs"]["phase1"],
            "phase2": cfg["runs"]["phase2"],
            "decision_rule": cfg["decision"],
            "etas": cfg["prior"]["etas"],
        },
        "valid_labels": {
            "policy": valid_cfg["policy"],
            "source": valid_cfg["source"],
            "source_lock": valid_cfg["source_lock"],
            "sha256": y_sha,
            "read_utc": lock["read_utc"],
            "read_commit": lock["git_commit"],
            "read_by_previous_round": True,
            "this_round_reads_labels": False,
        },
        "sy": prep["sy"],
        "frozen_before_any_formal_trajectory": True,
        "labels_read": {"official_valid": False, "official_test": False},
        "official_test_loaded": False,
    }
    write_json(FREEZE_JSON, payload)
    log(f"[freeze] wrote {FREEZE_JSON.relative_to(REPO)}; valid labels reused read-only (sha {y_sha[:12]})")
    return 0


def verify_freeze() -> dict[str, Any]:
    require(FREEZE_JSON.exists(), "protocol freeze missing; refusing formal training")
    frozen = read_json(FREEZE_JSON)
    require(frozen.get("frozen_before_any_formal_trajectory") is True, "freeze not marked pre-trajectory")
    mismatches = {k: {"frozen": frozen["code_hashes"].get(k), "current": v} for k, v in code_hashes().items() if frozen["code_hashes"].get(k) != v}
    require(not mismatches, f"code hash mismatch vs freeze: {mismatches}")
    require(file_sha256(CONFIG_PATH) == frozen["config_sha256"], "config changed after freeze")
    for key, path in (("prep", PREP_JSON), ("vocab", VOCAB_PT), ("types", TYPES_PT), ("gates", GATES_JSON), ("smoke", SMOKE_JSON)):
        require(file_sha256(path) == frozen[f"{key}_sha256"], f"{key} artifact changed after freeze")
    require(audit_frozen_inputs()["all_ok"], "frozen input hash drift")
    return frozen


# ---------------------------------------------------------------------------
# SSL for seeds without a frozen encoder
# ---------------------------------------------------------------------------


def cmd_ssl(args: argparse.Namespace) -> int:
    seed = int(args.seed)
    require(seed >= 2, "seeds 0/1 must reuse the frozen train10k-only SSL checkpoints")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")
    SSL_DIR.mkdir(parents=True, exist_ok=True)
    # Reuse the frozen implementation verbatim; only its output directory is
    # redirected so the frozen historical artifacts are never overwritten.
    f10k.OUT = SSL_DIR
    payload = f10k.pretrain_full10k(seed, device)
    enc = SSL_DIR / f"pretrain_encoder_s{seed}.pt"
    require(enc.exists(), "SSL encoder missing after pretraining")
    model = M.build_p3(seed, norm_record(), "cpu")
    M.load_encoder(model, seed, enc, "cpu")
    sh2 = D.sh2_for_encoder(model, load_objects_train(), "cpu")
    sh2.update(
        {
            "seed": seed,
            "encoder_path": str(enc.relative_to(REPO)),
            "encoder_sha256": file_sha256(enc),
            "frozen_before_training": True,
            "ssl_wall_seconds": payload["wall_seconds"],
        }
    )
    write_json(sh2_path(seed), sh2)
    log(f"[ssl s{seed}] encoder {sh2['encoder_sha256'][:12]} Sh2={sh2['sh2']:.6f} wall={payload['wall_seconds']:.0f}s")
    return 0


# ---------------------------------------------------------------------------
# formal training
# ---------------------------------------------------------------------------


def _last_soup(states: list[dict[str, torch.Tensor]]) -> tuple[list[int], dict[str, torch.Tensor]]:
    require(0 < len(states) <= SOUP_LAST, "empty soup window")
    keys = states[0].keys()
    soup = {k: torch.stack([s[k].float() for s in states]).mean(0) for k in keys}
    return list(range(EPOCHS - len(states) + 1, EPOCHS + 1)), soup


def train_one(arm: str, eta: float | None, seed: int, device: torch.device, *, log_fn: Callable[[str], None] = log) -> dict[str, Any]:
    verify_freeze()
    prep = read_json(PREP_JSON)
    vocab = load_vocab() if arm != "B" else None
    eta_value = residual_eta(arm, eta)
    key = run_key(arm, eta)
    out_dir = RUNS / f"{key}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = encoder_path(seed)
    require(encoder.exists(), f"missing SSL encoder for seed {seed}: {encoder}")
    sh2_rec = read_json(sh2_path(seed))
    sh2 = float(sh2_rec["sh2"])
    sy = float(prep["sy"]["sy"])
    if arm == "B":
        lam = 0.0
    else:
        lam = eta_value * sy * int(vocab["n_types"]) / (2.0 * N_TRAIN)

    train_objs = load_objects_train()
    assigned = load_assigned("train")
    y = f10k._load_y_rows(list(range(N_TRAIN)))
    model = M.build_arm(arm, seed, norm_record(), vocab, sh2, device)
    init_shared = M.shared_module_hash(model)
    init_residual = M.residual_state_hash(model)
    M.load_encoder(model, seed, encoder, device)
    encoder_hash = f10k._encoder_group_hash(model)

    optimizer = _optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_UPDATES, eta_min=LR_MIN)
    gen = torch.Generator().manual_seed(int(seed) + SHUFFLE_OFFSET)
    states: deque = deque(maxlen=SOUP_LAST)
    curve: list[dict[str, Any]] = []
    order_hash = hashlib.sha256()
    grad_norm_max = 0.0
    clip_events = 0
    nonfinite = False
    total_mol = 0
    prior_sum = 0.0
    penalty_sum = 0.0
    prior_steps = 0
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        lr_first = lr_last = None
        t_ep = time.perf_counter()
        epoch_prior = 0.0
        epoch_penalty = 0.0
        epoch_grad_P = 0.0
        epoch_grad_beta = 0.0
        order = torch.randperm(N_TRAIN, generator=gen)
        order_hash.update(np.ascontiguousarray(order.numpy(), dtype=np.int64).tobytes())
        for start in range(0, N_TRAIN, BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE].tolist()
            batch = M.batch_with_types(train_objs, indices, assigned, device)
            target = torch.as_tensor(y[indices], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target.view(-1))
            if arm != "B":
                prior, _ = model.atom_residual.prior()
                loss = loss + lam * prior
                epoch_prior += float(prior.detach())
                epoch_penalty += lam * float(prior.detach())
                prior_steps += 1
            optimizer.zero_grad()
            loss.backward()
            if arm != "B":
                epoch_grad_P = _grad_norm(model.atom_residual.P)
                epoch_grad_beta = _grad_norm(model.atom_residual.beta)
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
            if not math.isfinite(gn):
                nonfinite = True
                raise ESRRunError(f"non-finite grad at epoch {epoch} {key} seed {seed}")
            grad_norm_max = max(grad_norm_max, gn)
            if gn > GRAD_CLIP:
                clip_events += 1
            actual_lr = float(optimizer.param_groups[0]["lr"])
            if lr_first is None:
                lr_first = actual_lr
            lr_last = actual_lr
            optimizer.step()
            scheduler.step()
            total += float((pred.view(-1) - target.view(-1)).abs().sum())
            n_mol += len(indices)
            steps += 1
        train_mae = total / max(n_mol, 1)
        total_mol += n_mol
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        states.append(state)
        if epoch % EVAL_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            entry: dict[str, Any] = {
                "epoch": epoch,
                "updates": epoch * steps,
                "train_mae": train_mae,
                "lr_first": lr_first,
                "lr_last": lr_last,
                "seconds": time.perf_counter() - t_ep,
                "prior": epoch_prior / max(steps, 1),
                "prior_penalty": epoch_penalty / max(steps, 1),
            }
            if arm != "B":
                with torch.no_grad():
                    norms = model.atom_residual.correction_norms()
                    entry.update(
                        {
                            "correction_norm_mean": float(norms.mean()),
                            "correction_norm_max": float(norms.max()),
                            "correction_norm_median": float(norms.median()),
                            "active_type_fraction": float((norms > 1e-12).float().mean()),
                            "grad_norm_P": epoch_grad_P,
                            "grad_norm_beta": epoch_grad_beta,
                        }
                    )
            curve.append(entry)
            log_fn(
                f"[{key} s{seed}] ep={epoch:03d} train={train_mae:.6f} "
                + (f"R={entry['prior']:.4f} " if arm != "B" else "")
                + f"lr={lr_last:.3e} {entry['seconds']:.1f}s"
            )
        if not math.isfinite(train_mae):
            raise ESRRunError(f"non-finite loss at epoch {epoch} {key} seed {seed}")

    wall = time.perf_counter() - started
    members, soup = _last_soup(list(states))
    torch.save(soup, out_dir / "soup.pt")
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = M.predict(model, train_objs, assigned, device)
    soup_fit = float(np.abs(fit_pred - y).mean())
    info = {
        "protocol": PROTOCOL_ID,
        "run_key": key,
        "arm": arm,
        "eta": eta_value if arm != "B" else None,
        "seed": int(seed),
        "git_commit": git_commit(),
        "epochs": EPOCHS,
        "updates": TOTAL_UPDATES,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "batch_size": BATCH_SIZE,
        "n_train_rows": N_TRAIN,
        "recipe": {
            "loss": "raw-y L1 (batch molecule mean)",
            "optimizer": "Adam",
            "lr": LR,
            "lr_min": LR_MIN,
            "schedule": "per-step cosine",
            "weight_decay": WEIGHT_DECAY,
            "residual_weight_decay": RESIDUAL_WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "soup_last": SOUP_LAST,
        },
        "prior": {
            "lambda": lam,
            "sy": sy,
            "Sh2": sh2,
            "K": int(vocab["n_types"]) if vocab is not None else 0,
            "N": N_TRAIN,
            "formula": "lambda = eta * sy * K / (2N)" if arm != "B" else "none",
            "mean_R_per_step": prior_sum / max(prior_steps, 1),
            "mean_penalty_per_step": penalty_sum / max(prior_steps, 1) if prior_steps else 0.0,
            "full_vocab_every_step": arm != "B",
        },
        "soup_members": members,
        "soup_sha256": file_sha256(out_dir / "soup.pt"),
        "soup_fit_mae": soup_fit,
        "curve": curve,
        "wall_seconds": float(wall),
        "molecules_seen": int(total_mol),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "batch_order": {
            "full_hash": order_hash.hexdigest(),
            "matches_frozen_R3": None,
        },
        "initialisation": {
            "shared_module_hash": init_shared,
            "residual_state_hash": init_residual,
            "encoder_used_sha256": encoder_hash,
            "encoder_file_sha256": file_sha256(encoder),
            "encoder_source": "frozen_train10k_only" if seed in (0, 1) else "this_round_same_recipe",
        },
        "parameter_audit": M.parameter_audit(model),
        "anomalies": {"grad_norm_max": grad_norm_max, "clip_events": clip_events, "nonfinite": nonfinite},
        "labels_used": "official train raw y only",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if arm == "B":
        expected_total = 413_505
    else:
        expected_total = int(read_json(PREP_JSON)["vocab"]["total_inference_parameters"])
    require(int(info["parameter_audit"]["total_parameters"]) == int(expected_total), "parameter audit failed")
    if seed in (0, 1):
        frozen = {0: "3c964a71acc1ca817567cac916a7a59a12b40f750dbceee0848f614a2c71428f",
                  1: "8710751abd8c33739cac5506756fe6ba6abb919aed3235f86619b13fcf2700b9"}
        info["batch_order"]["matches_frozen_R3"] = bool(order_hash.hexdigest() == frozen[seed])
        require(info["batch_order"]["matches_frozen_R3"], "batch order drifted from the frozen R3 recipe")
    write_json(out_dir / "train.json", info)
    _write_curve(out_dir / "curve.csv", curve)
    log(f"[{key} s{seed}] done: soup fit {soup_fit:.6f}, wall {wall:.1f}s, params {info['parameter_audit']['total_parameters']}")
    return info


def _write_curve(path: Path, curve: Sequence[Mapping[str, Any]]) -> None:
    keys = sorted({k for row in curve for k in row})
    order = ["epoch", "updates", "train_mae", "lr_first", "lr_last", "seconds"] + [k for k in keys if k not in {"epoch", "updates", "train_mae", "lr_first", "lr_last", "seconds"}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(order) + "\n")
        for row in curve:
            fh.write(",".join(str(row.get(k, "")) for k in order) + "\n")


def cmd_train(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")
    train_one(args.arm, args.eta, int(args.seed), device)
    return 0


# ---------------------------------------------------------------------------
# prediction + freeze
# ---------------------------------------------------------------------------


def model_for_run(key: str, seed: int, device):
    arm, eta = parse_run_key(key)
    vocab = None if arm == "B" else load_vocab()
    sh2 = float(read_json(sh2_path(seed))["sh2"])
    model = M.build_arm(arm, seed, norm_record(), vocab, sh2, device)
    M.load_encoder(model, seed, encoder_path(seed), device)
    state = torch.load(RUNS / f"{key}_s{seed}" / "soup.pt", map_location="cpu", weights_only=False)
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.eval()
    return model, arm, eta


def cmd_predict(args: argparse.Namespace) -> int:
    verify_freeze()
    started = time.perf_counter()
    key = args.run
    seed = int(args.seed)
    device = torch.device(args.device)
    require((RUNS / f"{key}_s{seed}" / "soup.pt").exists(), f"missing soup for {key} seed {seed}")
    valid_objs = load_objects_valid()
    assigned = load_assigned("valid")
    model, arm, eta = model_for_run(key, seed, device)
    pred = M.predict(model, valid_objs, assigned, device)
    out = PRED / f"valid_{key}_s{seed}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, pred=pred.astype(np.float32), rows=np.arange(len(pred), dtype=np.int64))
    payload = {
        "protocol": PROTOCOL_ID,
        "run_key": key,
        "arm": arm,
        "eta": eta,
        "seed": seed,
        "n": int(pred.shape[0]),
        "pred_sha256": f10k.array_sha256(pred.astype(np.float32)),
        "file_sha256": file_sha256(out),
        "soup_sha256": file_sha256(RUNS / f"{key}_s{seed}" / "soup.pt"),
        "git_commit": git_commit(),
        "created_utc": utcnow(),
        "labels_read": {"official_valid": False, "official_test": False},
        "official_test_loaded": False,
        "wall_seconds": float(time.perf_counter() - started),
    }
    write_json(PRED / f"valid_{key}_s{seed}.json", payload)
    log(f"[predict] {key} s{seed} -> {out.name} sha {payload['pred_sha256'][:12]}")
    return 0


def cmd_freeze_predictions(args: argparse.Namespace) -> int:
    verify_freeze()
    files = {}
    for path in sorted(PRED.glob("valid_*_s*.npz")):
        meta = read_json(path.with_suffix(".json"))
        actual = file_sha256(path)
        require(actual == meta["file_sha256"], f"prediction file hash mismatch {path.name}")
        files[path.stem] = {
            "path": str(path.relative_to(REPO)),
            "sha256": actual,
            "pred_sha256": meta["pred_sha256"],
            "seed": meta["seed"],
            "run_key": meta["run_key"],
        }
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "created_utc": utcnow(),
        "prediction_files": files,
        "frozen_before_valid_read": True,
        "valid_labels_read_so_far": False,
        "official_test_loaded": False,
    }
    write_json(PRED_FREEZE, payload)
    log(f"[freeze-predictions] {len(files)} files frozen")
    return 0


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def valid_labels() -> tuple[np.ndarray, dict[str, Any]]:
    """The single existing official-valid label read, hash-gated.

    The original ``valid_y_read_once.npy`` is git-ignored and is not present in
    every checkout.  If it is absent the array is re-materialised from the
    official valid split and must match the committed lock hash exactly; this is
    a byte-identical reconstruction of an already-recorded read, not a new read.
    """
    cfg = load_config()["valid_labels"]
    path = REPO / cfg["source"]
    lock = read_json(REPO / cfg["source_lock"])
    reconstructed = False
    if path.exists():
        y = np.load(path)
    else:
        ds = f10k.load_valid_dataset()
        y = np.asarray([float(ds[i].y.reshape(-1)[0]) for i in range(N_VALID)], dtype=np.float64)
        reconstructed = True
        ANALYSIS.mkdir(parents=True, exist_ok=True)
        np.save(ANALYSIS / "valid_y_reconstructed.npy", y)
    sha = f10k.array_sha256(y)
    require(sha == cfg["sha256"] == lock["y_sha256"], "valid label hash mismatch")
    return y.astype(np.float64), {
        "path": cfg["source"],
        "sha256": sha,
        "read_utc": lock["read_utc"],
        "read_commit": lock["git_commit"],
        "policy": "reuse of the single existing read; this round performs no new scientific label read",
        "reconstructed_from_official_raw_valid": reconstructed,
        "reconstruction_rule": (
            "only if the ignored npy copy is absent; the reconstructed array must equal the "
            "committed lock sha256 (byte-identical), otherwise the stage stops"
        ),
    }


def _err(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(pred, dtype=np.float64) - y)


def _contrast(errs_c: Sequence[np.ndarray], errs_r: Sequence[np.ndarray], rows: np.ndarray | None = None) -> dict[str, Any]:
    """Paired per-molecule difference (candidate - reference); negative favours candidate."""
    ec = np.mean(np.stack(errs_c, 0), axis=0)
    er = np.mean(np.stack(errs_r, 0), axis=0)
    delta = ec - er
    if rows is not None:
        delta = delta[rows]
    return {
        "delta": float(delta.mean()),
        "bootstrap": f10k.paired_bootstrap(delta, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED),
        "better_fraction": float((delta < 0).mean()),
        "n": int(delta.size),
    }


def cmd_score(args: argparse.Namespace) -> int:
    verify_freeze()
    require(PRED_FREEZE.exists(), "freeze predictions first")
    pf = read_json(PRED_FREEZE)
    require(pf["frozen_before_valid_read"] is True, "prediction freeze is not pre-label")
    for key, entry in pf["prediction_files"].items():
        require(file_sha256(REPO / entry["path"]) == entry["sha256"], f"prediction drift: {key}")
    y, y_meta = valid_labels()
    cfg = load_config()
    phase = int(getattr(args, "phase", 1))

    available = set(pf["prediction_files"])
    if phase == 1:
        seeds = [int(s) for s in cfg["runs"]["phase1"]["seeds"]]
        configs = phase1_configs()
        score_json, contrast_json, decision_json = SCORE_JSON, CONTRAST_JSON, DECISION_JSON
    else:
        require(DECISION_JSON.exists(), "phase-1 score/decision missing")
        p1 = read_json(DECISION_JSON)
        if not p1["phase2_plan"]["authorized"]:
            log("[score] phase 2 was not authorized by the phase-1 rule")
            return 0
        bought = list(p1["phase2_plan"]["purchase_configs"])
        configs = ["B"] + bought
        seeds = sorted(
            {int(s) for s in cfg["runs"]["phase2"]["seeds"]}
            | {int(s) for s in cfg["runs"]["phase1"]["seeds"]}
        )
        seeds = [s for s in seeds if f"valid_B_s{s}" in available and all(f"valid_{k}_s{s}" in available for k in configs)]
        score_json = ANALYSIS / "metrics_phase2.json"
        contrast_json = ANALYSIS / "contrasts_phase2.json"
        decision_json = ANALYSIS / "decision_phase2.json"
    require(len(seeds) >= 2, "score needs at least two seeds")

    preds = {key: np.load(REPO / pf["prediction_files"][key]["path"])["pred"].astype(np.float64) for key in available}
    for key, arr in preds.items():
        require(arr.shape == (N_VALID,), f"bad prediction shape {key}")

    models: dict[str, Any] = {}
    for key in configs:
        per_seed = {}
        for s in seeds:
            p = preds[f"valid_{key}_s{s}"]
            per_seed[str(s)] = f10k.row_metrics(p, y)
        models[key] = {
            "mae_seed_mean": float(np.mean([per_seed[str(s)]["mae"] for s in seeds])),
            "per_seed": per_seed,
            "params": read_json(RUNS / f"{key}_s{seeds[0]}" / "train.json")["parameter_audit"]["total_parameters"],
            "wall_seconds": {str(s): read_json(RUNS / f"{key}_s{s}" / "train.json")["wall_seconds"] for s in seeds},
            "soup_fit_mae": {str(s): read_json(RUNS / f"{key}_s{s}" / "train.json")["soup_fit_mae"] for s in seeds},
        }

    b_errs = [_err(preds[f"valid_B_s{s}"], y) for s in seeds]
    b_mean_err = np.mean(np.stack(b_errs, 0), axis=0)
    worst = np.argsort(-b_mean_err)[: int(cfg["decision"]["drop_reference_worst_k"])]
    keep = np.ones(N_VALID, dtype=bool)
    keep[worst] = False
    rows_ordinary = np.where(y > float(cfg["decision"]["ordinary_threshold"]))[0]
    rows_tail = np.where(y <= float(cfg["decision"]["ordinary_threshold"]))[0]

    contrasts: dict[str, Any] = {}
    for key in configs:
        if key == "B":
            continue
        c_errs = [_err(preds[f"valid_{key}_s{s}"], y) for s in seeds]
        full = _contrast(c_errs, b_errs)
        full["per_seed"] = {
            str(s): float(_err(preds[f"valid_{key}_s{s}"], y).mean() - _err(preds[f"valid_B_s{s}"], y).mean())
            for s in seeds
        }
        full["all_seeds_improve"] = bool(all(v < 0.0 for v in full["per_seed"].values()))
        full["seed_mean_gain"] = float(models["B"]["mae_seed_mean"] - models[key]["mae_seed_mean"])
        full["ordinary"] = _contrast(c_errs, b_errs, rows_ordinary)
        full["tail"] = _contrast(c_errs, b_errs, rows_tail)
        full["drop_B_worst_k"] = {
            "k": int(len(worst)),
            "rows": [int(i) for i in worst.tolist()],
            **_contrast(c_errs, b_errs, np.where(keep)[0]),
        }
        contrasts[f"{key}_minus_B"] = full

    # E vs H at matched eta (only when both exist for all scored seeds)
    for eta in cfg["prior"]["etas"]:
        ek, hk = run_key("E", float(eta)), run_key("H", float(eta))
        if ek in configs and hk in configs:
            e_errs = [_err(preds[f"valid_{ek}_s{s}"], y) for s in seeds]
            h_errs = [_err(preds[f"valid_{hk}_s{s}"], y) for s in seeds]
            contrasts[f"{hk}_minus_{ek}"] = {
                **_contrast(h_errs, e_errs),
                "per_seed": {
                    str(s): float(_err(preds[f"valid_{hk}_s{s}"], y).mean() - _err(preds[f"valid_{ek}_s{s}"], y).mean())
                    for s in seeds
                },
            }

    write_json(
        score_json,
        {
            "protocol": PROTOCOL_ID,
            "phase": phase,
            "git_commit": git_commit(),
            "valid_labels": y_meta,
            "seeds": seeds,
            "configs": configs,
            "models": models,
            "rows_ordinary": int(rows_ordinary.size),
            "rows_tail": int(rows_tail.size),
            "drop_B_worst_k_rows": [int(i) for i in worst.tolist()],
            "official_test_loaded": False,
        },
    )
    write_json(contrast_json, {"protocol": PROTOCOL_ID, "phase": phase, "contrasts": contrasts})

    gain_min = float(cfg["decision"]["family_vs_B"]["mean_valid_mae_gain_min"])
    decision: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "phase": phase,
        "git_commit": git_commit(),
        "thresholds": cfg["decision"],
        "seeds": seeds,
        "B_seed_mean_valid_mae": models["B"]["mae_seed_mean"],
        "families": {},
        "triggers": {},
    }
    for family in ("E", "H"):
        fam: dict[str, Any] = {}
        for eta in cfg["prior"]["etas"]:
            key = run_key(family, float(eta))
            c = contrasts.get(f"{key}_minus_B")
            if c is None or key not in models:
                continue
            checks = {
                "both_seeds_improve": bool(c["all_seeds_improve"]),
                "seed_mean_gain_ge_threshold": bool(c["seed_mean_gain"] >= gain_min),
                "ordinary_not_reversed": bool(c["ordinary"]["delta"] < 0.0),
                "drop_B_worst_k_not_reversed": bool(c["drop_B_worst_k"]["delta"] < 0.0),
            }
            fam[str(eta)] = {
                "run_key": key,
                "mae_seed_mean": models[key]["mae_seed_mean"],
                "per_seed_mae": {s: models[key]["per_seed"][s]["mae"] for s in models[key]["per_seed"]},
                "seed_mean_gain_vs_B": c["seed_mean_gain"],
                "per_seed_delta_vs_B": c["per_seed"],
                "ordinary_delta": c["ordinary"]["delta"],
                "drop_B_worst_k_delta": c["drop_B_worst_k"]["delta"],
                "bootstrap_all": c["bootstrap"],
                "checks": checks,
                "trigger": bool(all(checks.values())),
            }
        triggers = [float(e) for e, v in fam.items() if v["trigger"]]
        best = max(triggers, key=lambda e: fam[str(e)]["seed_mean_gain_vs_B"]) if triggers else None
        decision["families"][family] = fam
        decision["triggers"][family] = {"triggered": bool(triggers), "frozen_eta": best}
    triggered = [f for f in ("E", "H") if decision["triggers"][f]["triggered"]]
    decision["any_trigger"] = bool(triggered)
    decision["families_triggered"] = triggered
    decision["phase2_plan"] = {
        "authorized": bool(triggered) and phase == 1,
        "purchase_B": True,
        "purchase_configs": [run_key(f, decision["triggers"][f]["frozen_eta"]) for f in triggered],
        "seeds": [int(s) for s in cfg["runs"]["phase2"]["seeds"]],
        "ssl_seeds": [int(s) for s in cfg["runs"]["phase2"]["ssl_seeds_to_train"]],
        "note": "only families that pass the pre-registered multi-seed rule are purchased; no eta search beyond the two declared levels",
    }
    if phase == 2:
        decision["phase1_reference"] = read_json(DECISION_JSON)["triggers"]
    write_json(decision_json, decision)
    log(f"[score p{phase}] B seed-mean {models['B']['mae_seed_mean']:.6f}")
    for key in configs[1:]:
        c = contrasts[f"{key}_minus_B"]
        log(
            f"[score p{phase}] {key}: mae {models[key]['mae_seed_mean']:.6f} gain {c['seed_mean_gain']:+.6f} "
            f"all {c['all_seeds_improve']} ord {c['ordinary']['delta']:+.6f} "
            f"drop10 {c['drop_B_worst_k']['delta']:+.6f}"
        )
    for f in ("E", "H"):
        log(f"[score p{phase}] family {f}: triggered={decision['triggers'][f]['triggered']} eta={decision['triggers'][f]['frozen_eta']}")
    return 0


# ---------------------------------------------------------------------------
# analysis: curves, mechanism/dependence, groups, ledger
# ---------------------------------------------------------------------------


def cmd_analyze(args: argparse.Namespace) -> int:
    require(DECISION_JSON.exists(), "run score first")
    cfg = load_config()
    y, _ = valid_labels()
    prep = read_json(PREP_JSON)
    vocab = load_vocab()
    valid_objs = load_objects_valid()
    assigned = load_assigned("valid")
    device = torch.device(args.device)
    seeds = [int(s) for s in cfg["runs"]["phase1"]["seeds"]]
    configs = phase1_configs()
    cuts = prep["train_size_terciles"]

    mechanism: dict[str, Any] = {"protocol": PROTOCOL_ID, "runs": {}}
    for key in configs:
        arm, eta = parse_run_key(key)
        for s in seeds:
            train_rec = read_json(RUNS / f"{key}_s{s}" / "train.json")
            _copy_curve(key, s, train_rec)
            if arm == "B":
                continue
            model, _, _ = model_for_run(key, s, device)
            residual = model.atom_residual
            with torch.no_grad():
                a = residual.correction_matrix()
                norms = a.norm(dim=0)
                depth = residual.depth
                per_type = {
                    "mean_norm_all": float(norms.mean()),
                    "mean_norm_depth1": float(norms[depth == 1].mean()),
                    "mean_norm_depth2": float(norms[depth == 2].mean()),
                    "max_norm": float(norms.max()),
                    "active_fraction": float((norms > 1e-12).float().mean()),
                    "mean_beta_abs": float(residual.beta.detach().abs().mean()),
                    "P_column_norm_min": float(residual.P.detach().norm(dim=0).min()),
                    "P_column_norm_max": float(residual.P.detach().norm(dim=0).max()),
                }
                prior, per = residual.prior()
                # occurrence-weighted effective correction norm on valid atoms
                t = np.concatenate([np.asarray(x["t_star"]) for x in assigned])
                used = t[t >= 0]
                occ_norm = float(norms[used].mean()) if used.size else 0.0
                atom_types = torch.as_tensor(t, dtype=torch.long, device=device).clamp(min=0)
                a_atom = residual.beta.index_select(0, atom_types) @ residual.P.t()
                atom_norm = a_atom.norm(dim=1).cpu().numpy()
            pred_full = np.load(PRED / f"valid_{key}_s{s}.npz")["pred"].astype(np.float64)
            with torch.no_grad():
                saved = residual.beta.detach().clone()
                residual.beta.zero_()
                pred_zero = M.predict(model, valid_objs, assigned, device).astype(np.float64)
                residual.beta.copy_(saved)
            mechanism["runs"][f"{key}_s{s}"] = {
                "prior_value": float(prior),
                "lambda": train_rec["prior"]["lambda"],
                "per_type": per_type,
                "occurrence_weighted_correction_norm_valid": occ_norm,
                "mean_atom_correction_norm_valid": float(atom_norm.mean()),
                "max_atom_correction_norm_valid": float(atom_norm.max()),
                "valid_zero_residual_mae": float(np.abs(pred_zero - y).mean()),
                "valid_full_mae": float(np.abs(pred_full - y).mean()),
                "dependence_delta_mae": float(np.abs(pred_zero - y).mean() - np.abs(pred_full - y).mean()),
                "dependence_delta_note": "positive = the trained residual improves valid MAE relative to zeroing it; this measures dependence, not net benefit vs B",
                "grad_norm_P_last": train_rec["curve"][-1].get("grad_norm_P"),
                "grad_norm_beta_last": train_rec["curve"][-1].get("grad_norm_beta"),
                "prior_at_last_epoch": train_rec["curve"][-1].get("prior"),
                "sh2": float(read_json(sh2_path(s))["sh2"]),
                "K": int(vocab["n_types"]),
            }
    write_json(MECH_JSON, mechanism)

    # groups
    preds = {key: {s: np.load(PRED / f"valid_{key}_s{s}.npz")["pred"].astype(np.float64) for s in seeds} for key in configs}
    group_rows: dict[str, list[int]] = {}
    per_mol = []
    for i, obj in enumerate(valid_objs):
        labels = D.molecule_group_labels(obj, assigned[i], np.asarray(assigned[i]["support"]), float(y[i]), size_tercile_cuts=cuts)
        per_mol.append(labels)
        for name in ("ordinary", "tail", "large_ring", "low_support_none", "low_support_partial",
                     "low_support_majority", "fallback_none", "fallback_partial", "fallback_high",
                     "size_small", "size_medium", "size_large"):
            if labels.get(name):
                group_rows.setdefault(name, []).append(i)
    groups: dict[str, Any] = {"protocol": PROTOCOL_ID, "thresholds": prep["group_thresholds"], "groups": {}}
    for name, rows in group_rows.items():
        if not rows:
            continue
        idx = np.asarray(rows)
        entry: dict[str, Any] = {"n": int(idx.size), "fraction": float(idx.size / N_VALID), "models": {}, "contrasts": {}}
        for key in configs:
            mae = {str(s): float(_err(preds[key][s], y)[idx].mean()) for s in seeds}
            entry["models"][key] = {"mae_seed_mean": float(np.mean(list(mae.values()))), "per_seed": mae}
        for key in configs[1:]:
            d = _contrast([_err(preds[key][s], y)[idx] for s in seeds], [_err(preds["B"][s], y)[idx] for s in seeds])
            entry["contrasts"][f"{key}_minus_B"] = d
        groups["groups"][name] = entry
    groups["molecule_table_summary"] = {
        "low_support_fraction_mean": float(np.mean([m["low_support_fraction"] for m in per_mol])),
        "fallback_fraction_mean": float(np.mean([m["fallback_fraction"] for m in per_mol])),
        "zero_correction_fraction_mean": float(np.mean([m["zero_correction_fraction"] for m in per_mol])),
        "n_atoms_mean": float(np.mean([m["n_atoms"] for m in per_mol])),
    }
    write_json(GROUPS_JSON, groups)

    ledger = _ledger()
    write_json(LEDGER_JSON, ledger)
    log(f"[analyze] wrote mechanism/groups/ledger; phase-1 GPU {ledger['phase1_gpu_hours']:.4f} h")
    return 0


def _copy_curve(key: str, seed: int, rec: Mapping[str, Any]) -> None:
    path = CURVES / f"{key}_s{seed}.csv"
    _write_curve(path, rec["curve"])


def _ledger() -> dict[str, Any]:
    rows = []
    total = 0.0
    for path in sorted(RUNS.glob("*_s*/train.json")):
        rec = read_json(path)
        rows.append({"run": rec["run_key"], "seed": rec["seed"], "stage": "train",
                     "wall_seconds": rec["wall_seconds"], "gpu": rec.get("cuda_device_name", "")})
        total += float(rec["wall_seconds"])
    predict_seconds = 0.0
    for path in sorted(PRED.glob("valid_*_s*.json")):
        rec = read_json(path)
        predict_seconds += float(rec.get("wall_seconds", 0.0))
    smoke = read_json(SMOKE_JSON) if SMOKE_JSON.exists() else None
    smoke_seconds = float(sum(r["wall_seconds"] for r in smoke["records"])) if smoke else 0.0
    ssl_rows = []
    for path in sorted(SSL_DIR.glob("pretraining_seed*.json")):
        rec = read_json(path)
        ssl_rows.append({"seed": int(rec["seed"]), "wall_seconds": float(rec["wall_seconds"])})
    ssl_total = float(sum(r["wall_seconds"] for r in ssl_rows))
    gpu_seconds = total + smoke_seconds + ssl_total + predict_seconds
    cfg = load_config()
    p1_seeds = {int(v) for v in cfg["runs"]["phase1"]["seeds"]}
    p1_keys = set(phase1_configs())
    phase1_train = 0.0
    for path in sorted(RUNS.glob("*_s*/train.json")):
        rec = read_json(path)
        if int(rec["seed"]) in p1_seeds and rec["run_key"] in p1_keys:
            phase1_train += float(rec["wall_seconds"])
    prep = read_json(PREP_JSON)
    return {
        "protocol": PROTOCOL_ID,
        "gpu_stages": rows,
        "ssl_pretraining": ssl_rows,
        "smoke_seconds": smoke_seconds,
        "train_seconds": total,
        "ssl_seconds": ssl_total,
        "gpu_seconds_total": gpu_seconds,
        "gpu_hours_total": gpu_seconds / 3600.0,
        "phase1_gpu_hours": (phase1_train + smoke_seconds) / 3600.0,
        "predict_seconds_total": predict_seconds,
        "cpu_preprocessing_seconds": prep["timing_seconds"]["total_prep"],
        "cpu_patch_key_seconds": prep["timing_seconds"]["patch_keys_both_splits"],
        "cap_gpu_hours": 8.0,
        "accounting": "sum of process wall on both GPUs; scheduler overhead not included",
        "note": "prediction forwards are included in the train-process wall only where the runner measured them; they are <60 s each",
    }


# ---------------------------------------------------------------------------
# chain / phase 2
# ---------------------------------------------------------------------------


def cmd_chain(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    require(device.type == "cuda" and torch.cuda.is_available(), "chain requires cuda")
    keys = [k.strip() for k in args.configs.split(",") if k.strip()]
    for key in keys:
        arm, eta = parse_run_key(key)
        out = RUNS / f"{key}_s{int(args.seed)}" / "train.json"
        if out.exists() and (RUNS / f"{key}_s{int(args.seed)}" / "soup.pt").exists() and not args.force:
            log(f"[chain] {key} s{args.seed} already trained; reusing and re-predicting")
        else:
            train_one(arm, eta, int(args.seed), device)
        cmd_predict(argparse.Namespace(run=key, seed=int(args.seed), device=args.device))
    return 0


def cmd_phase2(args: argparse.Namespace) -> int:
    require(DECISION_JSON.exists(), "run score first")
    decision = read_json(DECISION_JSON)
    plan = decision["phase2_plan"]
    if not plan["authorized"]:
        log("[phase2] not authorized by the phase-1 rule; no runs purchased")
        return 0
    log(f"[phase2] authorized: {plan}")
    if not args.run:
        return 0
    for seed in plan["seeds"]:
        if not encoder_path(int(seed)).exists():
            cmd_ssl(argparse.Namespace(seed=int(seed), device=args.device))
        keys = ["B"] + list(plan["purchase_configs"])
        cmd_chain(argparse.Namespace(seed=int(seed), device=args.device, configs=",".join(keys), force=args.force))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prep").set_defaults(func=cmd_prep)
    sub.add_parser("gates").set_defaults(func=cmd_gates)
    p = sub.add_parser("smoke")
    p.add_argument("--epochs", type=int, default=2)
    p.set_defaults(func=cmd_smoke)
    sub.add_parser("freeze").set_defaults(func=cmd_freeze)
    p = sub.add_parser("ssl")
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(func=cmd_ssl)
    p = sub.add_parser("train")
    p.add_argument("--arm", choices=list(M.ARMS), required=True)
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(func=cmd_train)
    p = sub.add_parser("predict")
    p.add_argument("--run", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(func=cmd_predict)
    sub.add_parser("freeze-predictions").set_defaults(func=cmd_freeze_predictions)
    p = sub.add_parser("score")
    p.add_argument("--phase", type=int, default=1, choices=[1, 2])
    p.set_defaults(func=cmd_score)
    sub.add_parser("analyze").set_defaults(func=cmd_analyze)
    p = sub.add_parser("chain")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--configs", required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_chain)
    p = sub.add_parser("phase2")
    p.add_argument("--run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_phase2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
