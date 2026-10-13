"""cscl-bjoc-strc-predictive-feasibility-v1 driver.

Protocol: ``configs/bjoc_strc_predictive_feasibility_v1.yaml`` (frozen before
any GPU trajectory).  Track: ``tracks/structure-contribution``.

Single scientific question
--------------------------
On the already-successful masked-pretrained BJOC-Null (P), can a *tiny*,
cross-molecule-shared, explicitly additive structural residual channel
(157 parameters: 28 shared atom-type scalars + 129 shared junction-context
scalars) trained jointly under the identical raw-y L1 recipe improve the
official-valid single-model MAE?

Everything here is matched to the frozen P round
(``results/bjoc_full10k_pretraining_transfer_v1``): exactly the same SSL
encoder checkpoints, the same scratch composer/reader initialisation, the
same batch order, the same optimizer/loss/soup.  The *only* difference of the
STRC arm is the zero-initialised alpha/beta channel:

    y_hat = f_theta(G) + s_a * sum_i alpha[type_i]
                       + s_j * sum_j beta[context_j]

Stages
------
``vocab``              train-only junction-context vocabulary + frozen scales
``design-audit``       label-blind count-matrix rank / coverage audit (A6)
``cpu-precheck``       A1-A4 correctness + gradient + sharing gates
``synthetic``          A5 planted additive task (full-rank and collinear)
``freeze``             protocol freeze manifest (hashes + gates)
``train``              STRC arm S seed 0/1 on the frozen recipe (GPU)
``predict``            label-blind official-valid predictions + base/residual
``train-predict``      label-blind? no: official-train predictions (train y only)
``freeze-predictions`` hash every prediction file before the valid label read
``score``              the ONE official-valid label read + metrics + bootstrap
``params``             alpha/beta export, support, identifiability caveats
``error-analysis``     tail / ordinary / top-k / extreme-row / strata
``decision``           the frozen scientific decision
``stage-cpu-prep``     vocab + design-audit + cpu-precheck + synthetic + freeze

Official test is never referenced, loaded, hashed or scored in this round.
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
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_masked_pretraining_v1 as mp  # noqa: E402
import bjoc_strc_v1 as strc  # noqa: E402
import run_bjoc_full10k_pretraining_transfer_v1 as f10k  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-bjoc-strc-predictive-feasibility-v1"
CONFIG_PATH = _CODE_DIR.parent / "configs" / "bjoc_strc_predictive_feasibility_v1.yaml"
OUT = REPO_ROOT / "tracks/structure-contribution/results/bjoc_strc_predictive_feasibility_v1"
F10K = REPO_ROOT / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1"
TERMINAL = REPO_ROOT / "tracks/structure-contribution/results/bjoc_pretraining_mechanism_terminal_test_v1"

# ---- official data (no internal split; test never referenced) --------------
N_TRAIN = 10000
N_VALID = 1000
SEEDS = (0, 1)

# ---- frozen downstream recipe (identical to the P round) --------------------
BATCH_SIZE = 128
EPOCHS = 240
SOUP_LAST = 5
TRAIN_SHUFFLE_OFFSET = 101
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
STEPS_PER_EPOCH = math.ceil(N_TRAIN / BATCH_SIZE)  # 79
UPDATES = EPOCHS * STEPS_PER_EPOCH                    # 18960

# ---- frozen analysis constants ---------------------------------------------
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 0
TOP_K_REMOVED = (1, 5, 10)
TAIL_THRESHOLD = -3.0
EXTREME_VALID_ROW = 172
CURVE_LOG_EVERY = 20

# ---- pre-registered purchase gate (official valid only) ---------------------
GAIN_THRESHOLD = -0.004          # seed-mean S - P must be at most this
RMSE_TOLERANCE = 0.005           # seed-mean S RMSE <= P RMSE + this
BASE_GRAD_TOLERANCE = 1.0e-6     # float32 numerical-precision agreement floor
SEED0_BASELINE_MAE = 0.084785663
SEED1_BASELINE_MAE = 0.082313002
P_SEED_MEAN_MAE = 0.083549333
P_SEED_MEAN_RMSE = 0.5 * (0.2787137334413769 + 0.1813011166141071)

# ---- artifacts --------------------------------------------------------------
VOCAB_JSON = OUT / "context_vocab.json"
COUNT_MATRIX = OUT / "count_matrix_train10k.npz"
DESIGN_AUDIT = OUT / "design_matrix_audit.json"
CPU_PRECHECK = OUT / "cpu_precheck.json"
SYNTHETIC_JSON = OUT / "synthetic_task.json"
FREEZE_JSON = OUT / "protocol_freeze.json"
TRAIN_JSON = OUT / "train_S_seed{seed}.json"
SOUP_PATH = OUT / "soup_S_seed{seed}.pt"
PRED_DIR = OUT / "predictions"
PRED_ACCESS = OUT / "prediction_access_log.json"
PRED_FREEZE = OUT / "prediction_freeze.json"
VALID_LOCK = OUT / "valid_read.lock"
VALID_Y_COPY = OUT / "valid_y_read_once.npy"
VALID_METRICS = OUT / "valid_metrics.json"
PAIRED_BOOTSTRAP = OUT / "paired_bootstrap.json"
PARAMS_JSON = OUT / "parameter_analysis.json"
TRAIN_METRICS = OUT / "train_metrics.json"
ERROR_ANALYSIS = OUT / "error_analysis.json"
SCIENTIFIC_DECISION = OUT / "scientific_decision.json"


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# small io / hashing helpers
# ---------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(f10k._j(payload), indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    return f10k.file_sha256(path)


def git_commit() -> str:
    return f10k.git_commit()


def _code_hashes() -> dict[str, str]:
    return {
        "driver": file_sha256(_CODE_DIR / "run_bjoc_strc_v1.py"),
        "module": file_sha256(_CODE_DIR / "bjoc_strc_v1.py"),
        "base_driver": file_sha256(_CODE_DIR / "run_bjoc_full10k_pretraining_transfer_v1.py"),
        "base_module": file_sha256(_CODE_DIR / "bjoc_masked_pretraining_v1.py"),
        "objects": file_sha256(_CODE_DIR / "structural_relation_objects_v1.py"),
        "composer": file_sha256(_CODE_DIR / "structural_relation_composer_v1.py"),
        "features": file_sha256(_CODE_DIR / "cscl_features.py"),
        "config": file_sha256(CONFIG_PATH),
    }


def _verify_inputs() -> dict[str, Any]:
    """Re-hash every frozen input this round reuses (no valid/test labels)."""
    manifest = read_json(F10K / "objects_manifest.json")
    terminal = read_json(TERMINAL / "model_reuse_audit.json")
    freeze = read_json(F10K / "prediction_freeze.json")
    checks: dict[str, Any] = {}

    def check(name: str, path: Path, expected: str) -> None:
        exists = Path(path).exists()
        actual = file_sha256(path) if exists else None
        checks[name] = {
            "path": str(Path(path).relative_to(REPO_ROOT)),
            "exists": bool(exists),
            "sha256": actual,
            "expected": expected,
            "ok": bool(exists and actual == expected),
        }

    check("objects_train10k", F10K / "objects_train10k.pt", manifest["objects_train10k_sha256"])
    check("objects_valid", F10K / "objects_valid.pt", manifest["objects_valid_sha256"])
    check("prep_train10k", F10K / "prep_train10k.json", manifest["prep_train10k_sha256"])
    for seed in SEEDS:
        enc_path = F10K / f"pretrain_encoder_s{seed}.pt"
        pre_rec = read_json(F10K / f"pretraining_seed{seed}.json")
        enc = torch.load(enc_path, map_location="cpu", weights_only=False)
        combined: dict[str, torch.Tensor] = {}
        for prefix, group in (("atom", "atom_encoder"), ("bond", "bond_encoder"), ("junc", "junction_encoder")):
            for k, v in enc[group].items():
                combined[f"{prefix}:{k}"] = v
        state_hash = f10k.state_group_sha256(combined)
        recorded = terminal["input_hash_checks"][f"pretrain_encoder_s{seed}"]["recorded"]
        checks[f"pretrain_encoder_s{seed}"] = {
            "path": str(enc_path.relative_to(REPO_ROOT)),
            "exists": bool(enc_path.exists()),
            "file_sha256": file_sha256(enc_path),
            "encoder_state_sha256": state_hash,
            "recorded_encoder_state_sha256": recorded,
            "recorded_training_sha256": pre_rec["encoder_final_sha256"],
            "ok": bool(state_hash == recorded == pre_rec["encoder_final_sha256"]),
        }
        check(
            f"soup_P_seed{seed}",
            F10K / f"soup_P_seed{seed}.pt",
            terminal["checkpoint_records"][f"P{seed}"]["checkpoint_sha256"],
        )
        check(
            f"valid_P{seed}_preds",
            F10K / "predictions" / f"valid_P{seed}.npz",
            freeze["prediction_files"][f"P{seed}"]["sha256"],
        )
    checks["all_ok"] = all(v.get("ok") for k, v in checks.items() if k != "all_ok")
    return checks


# ---------------------------------------------------------------------------
# model construction (matched P initialisation replay)
# ---------------------------------------------------------------------------


def _build_strc_model(seed: int, device: torch.device | str = "cpu") -> tuple[strc.STRCBJOC, dict[str, Any]]:
    """Rebuild the exact P supervision-phase initialisation, then wrap STRC.

    Order of operations is identical to ``f10k._train_downstream('P', seed)``:
    ``_make_downstream_model`` (seed_everything + scratch BJOC init), then the
    frozen SSL encoder state.  The STRC parameters are created afterwards from
    zeros and never consume the global RNG, so the base initialisation is
    bit-identical to the recorded P init.
    """
    norm = read_json(F10K / "prep_train10k.json")
    vocab = read_json(VOCAB_JSON)
    p_rec = read_json(F10K / f"train_P_seed{seed}.json")
    pretrain_rec = read_json(F10K / f"pretraining_seed{seed}.json")

    base = f10k._make_downstream_model(seed, norm, device)
    scratch_full_sha = src.state_dict_hash(base)
    scratch_group_sha = mp.group_state_hash(base)

    ssl_path = F10K / f"pretrain_encoder_s{seed}.pt"
    ssl_state = torch.load(ssl_path, map_location="cpu", weights_only=False)
    mp.load_encoder_state(base, ssl_state, device)
    encoder_used_sha = f10k._encoder_group_hash(base)

    model = strc.wrap_strc(base, vocab, device)
    audit = strc.strc_parameter_audit(model)
    groups = strc.strc_group_state_sha256(model)
    zero_alpha = bool(torch.equal(model.residual.alpha.detach().cpu(), torch.zeros(strc.ATOM_VOCAB)))
    zero_beta = bool(torch.equal(model.residual.beta.detach().cpu(), torch.zeros(strc.BETA_SIZE)))
    info = {
        "seed": int(seed),
        "scratch_init_full_sha256": scratch_full_sha,
        "scratch_init_group_sha256": scratch_group_sha,
        "recorded_p_init_full_sha256": p_rec["init_full_sha256"],
        "recorded_p_init_group_sha256": p_rec["init_group_state_sha256"],
        "scratch_equals_recorded_p_init": scratch_full_sha == p_rec["init_full_sha256"],
        "composer_reader_equals_recorded_p_init": scratch_group_sha == p_rec["init_group_state_sha256"],
        "encoder_file_sha256": file_sha256(ssl_path),
        "encoder_used_sha256": encoder_used_sha,
        "encoder_recorded_sha256": pretrain_rec["encoder_final_sha256"],
        "encoder_equals_recorded": encoder_used_sha == pretrain_rec["encoder_final_sha256"],
        "recorded_p_encoder_used_sha256": p_rec.get("encoder_used_sha256"),
        "encoder_equals_recorded_p": encoder_used_sha == p_rec.get("encoder_used_sha256"),
        "alpha_zero_at_init": zero_alpha,
        "beta_zero_at_init": zero_beta,
        "residual_group_sha256": groups["residual"],
        "base_full_group_sha256": groups["base_full"],
        "base_encoder_group_sha256": groups["base_encoder"],
        "base_composer_reader_group_sha256": groups["base_composer_reader"],
        "parameter_audit": audit,
    }
    return model, info


def _predict_strc(model: strc.STRCBJOC, objs: Sequence[sro.MolObjects], device: torch.device | str = "cpu") -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(objs), 256):
            batch = f10k._mv(sro.batch_objects(list(objs[i : i + 256])), torch.device(device))
            out.append(model(batch).detach().cpu().numpy())
    return np.concatenate(out, 0)


def _predict_strc_parts(
    model: strc.STRCBJOC, objs: Sequence[sro.MolObjects], device: torch.device | str = "cpu"
) -> dict[str, np.ndarray]:
    """Base prediction, residual and both count vectors (label-blind)."""
    model.eval()
    base_parts, resid_parts, a_parts, c_parts = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(objs), 256):
            batch = f10k._mv(sro.batch_objects(list(objs[i : i + 256])), torch.device(device))
            _, feats = model(batch, return_features=True)
            base_parts.append(feats["base_pred"].detach().cpu().numpy())
            resid_parts.append(feats["residual"].detach().cpu().numpy())
            a_parts.append(feats["a_counts"].detach().cpu().numpy())
            c_parts.append(feats["c_counts"].detach().cpu().numpy())
    return {
        "base_pred": np.concatenate(base_parts, 0),
        "residual": np.concatenate(resid_parts, 0),
        "a_counts": np.concatenate(a_parts, 0).astype(np.int16),
        "c_counts": np.concatenate(c_parts, 0).astype(np.int16),
    }


# ---------------------------------------------------------------------------
# Stage: train-only context vocabulary
# ---------------------------------------------------------------------------


def cmd_vocab(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    train_objs = f10k.load_objects(F10K / "objects_train10k.pt")
    vocab = strc.build_context_vocabulary(train_objs)
    write_json(VOCAB_JSON, vocab)
    log(
        f"[vocab] {vocab['distinct_contexts']} distinct contexts over "
        f"{vocab['n_molecules']} molecules; kept {len(vocab['vocab'])}; "
        f"OTHER {vocab['other_junctions']} junctions ({1 - vocab['covered_fraction']:.3f}); "
        f"s_a {vocab['s_a']:.6f} s_j {vocab['s_j']:.6f}; sha {vocab['vocab_sha256'][:16]}"
    )
    return 0


# ---------------------------------------------------------------------------
# Stage: design-matrix audit (A6, label-blind)
# ---------------------------------------------------------------------------


def _count_matrix(objs: Sequence[sro.MolObjects], vocab: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    residual = strc.STRCResidual(vocab)
    a_rows, c_rows = [], []
    for i in range(0, len(objs), 512):
        batch = sro.batch_objects(list(objs[i : i + 512]))
        a_counts, c_counts = residual.count_vectors(batch)
        a_rows.append(a_counts.numpy().astype(np.int64))
        c_rows.append(c_counts.numpy().astype(np.int64))
    return np.concatenate(a_rows, 0), np.concatenate(c_rows, 0)


def cmd_design_audit(args: argparse.Namespace) -> int:
    vocab = read_json(VOCAB_JSON)
    train_objs = f10k.load_objects(F10K / "objects_train10k.pt")
    a_counts, c_counts = _count_matrix(train_objs, vocab)
    assert a_counts.shape == (N_TRAIN, strc.ATOM_VOCAB), a_counts.shape
    assert c_counts.shape == (N_TRAIN, strc.BETA_SIZE), c_counts.shape

    s_a, s_j = float(vocab["s_a"]), float(vocab["s_j"])
    x_scaled = np.concatenate([s_a * a_counts, s_j * c_counts], axis=1).astype(np.float64)
    x_float = np.concatenate([a_counts, c_counts], axis=1).astype(np.float64)

    # rank / conditioning of the design actually used by the model
    u, sv, vt = np.linalg.svd(x_scaled - x_scaled.mean(0, keepdims=True), full_matrices=False)
    tol = sv.max() * max(x_scaled.shape) * np.finfo(np.float64).eps
    rank = int((sv > tol).sum())
    null_dim = int(x_scaled.shape[1] - rank)
    corr = np.corrcoef(x_float, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    flat = np.abs(corr).ravel()
    pairs = np.argsort(-flat)[:10]
    top_pairs = []
    for p in pairs:
        i, j = int(p // corr.shape[0]), int(p % corr.shape[0])
        if i < j:
            top_pairs.append({"i": i, "j": j, "abs_corr": float(abs(corr[i, j]))})
    n_js = np.asarray([o.junc_index.shape[0] for o in train_objs])

    atom_counts = a_counts.sum(0)
    ctx_counts = c_counts.sum(0)
    per_type = [
        {
            "type": int(t),
            "occurrences": int(atom_counts[t]),
            "molecules": int((a_counts[:, t] > 0).sum()),
        }
        for t in range(strc.ATOM_VOCAB)
    ]
    per_ctx = [
        {
            "slot": int(k),
            "context": (list(vocab["vocab"][k]) if k < len(vocab["vocab"]) else "OTHER"),
            "occurrences": int(ctx_counts[k]),
            "molecules": int((c_counts[:, k] > 0).sum()),
        }
        for k in range(strc.BETA_SIZE)
    ]
    zero_types = [d["type"] for d in per_type if d["occurrences"] == 0]
    zero_ctx = [d["slot"] for d in per_ctx if d["occurrences"] == 0]
    nonzero_sv = sv[sv > tol]
    exact_corr_pairs = [p for p in top_pairs if p["abs_corr"] >= 1.0 - 1e-12]
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "vocab_sha256": vocab["vocab_sha256"],
        "n_molecules": N_TRAIN,
        "n_columns": int(x_scaled.shape[1]),
        "rank": rank,
        "null_dim": null_dim,
        "zero_support_columns": int((~(x_scaled != 0).any(0)).sum()),
        "condition_number": float(sv.max() / max(sv[-1], 1e-300)),
        "condition_number_nonzero_spectrum": float(nonzero_sv.max() / max(nonzero_sv.min(), 1e-300)),
        "singular_values_summary": {
            "top5": sv[:5].tolist(),
            "tail5": sv[-5:].tolist(),
            "median": float(np.median(sv)),
            "nonzero_count": int(nonzero_sv.shape[0]),
        },
        "max_abs_pairwise_correlation": top_pairs,
        "exactly_collinear_pairs": exact_corr_pairs,
        "atom_counts": per_type,
        "context_counts": per_ctx,
        "zero_support_atom_types": zero_types,
        "zero_support_context_slots": zero_ctx,
        "note_zero_support": (
            "atom types 21-27 are raw PyG classes not observed in the official train10k; type 27 is the "
            "masked-pretraining token. Their alpha entries receive zero gradient and stay at zero; the "
            "frozen 28-slot vocabulary is retained for contract fidelity"
        ),
        "note_collinearity": (
            "exact duplicate count directions among supported columns make the individual coefficients "
            "non-identifiable; only fitted predictions are claimed"
        ),
        "other_fraction_of_junctions": float(ctx_counts[strc.OTHER_SLOT] / max(ctx_counts.sum(), 1)),
        "mean_n_atoms": float(np.mean([o.n_atoms for o in train_objs])),
        "mean_n_junctions": float(n_js.mean()),
        "s_a": s_a,
        "s_j": s_j,
        "count_matrix_sha256_a": f10k.array_sha256(a_counts),
        "count_matrix_sha256_c": f10k.array_sha256(c_counts),
        "labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    np.savez_compressed(
        COUNT_MATRIX,
        a_counts=a_counts.astype(np.int16),
        c_counts=c_counts.astype(np.int16),
    )
    write_json(DESIGN_AUDIT, payload)
    log(
        f"[design-audit] rank {rank}/{x_scaled.shape[1]} null {null_dim} "
        f"cond {payload['condition_number']:.3g} OTHER {payload['other_fraction_of_junctions']:.4f}"
    )
    return 0


# ---------------------------------------------------------------------------
# Stage: CPU correctness + mechanism pre-check (A1-A4)
# ---------------------------------------------------------------------------


def _relabelled_objects(row: int) -> tuple[sro.MolObjects, sro.MolObjects]:
    ds = f10k.load_train_dataset()
    base = f10k.load_objects(F10K / "objects_train10k.pt")[row]
    d0 = ds[int(base.mol_index)]
    perm = list(reversed(range(base.n_atoms)))
    inv = {old: new for new, old in enumerate(perm)}
    x_p = np.asarray(d0.x).copy().reshape(-1)
    x_new = np.zeros_like(x_p)
    for old in range(base.n_atoms):
        x_new[inv[old]] = x_p[old]
    rows = [(inv[int(d0.edge_index[0, c])], inv[int(d0.edge_index[1, c])]) for c in range(np.asarray(d0.edge_index).shape[1])]
    ei_p = np.asarray(rows, dtype=np.int64).T
    return base, sro.build_mol_objects(0, x_new, ei_p, np.asarray(d0.edge_attr))


def _forbidden_source_scan() -> list[str]:
    text = (_CODE_DIR / "bjoc_strc_v1.py").read_text(encoding="utf-8")
    tokens = ["MessagePassing", "Transformer", "MultiheadAttention", "GRU", "LSTM", "nn.GNN"]
    return [tok for tok in tokens if tok in text]


def cmd_cpu_precheck(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    vocab = read_json(VOCAB_JSON)
    if strc.vocabulary_sha256(vocab) != vocab["vocab_sha256"]:
        raise SystemExit("context vocabulary hash mismatch")
    train_objs = f10k.load_objects(F10K / "objects_train10k.pt")
    y_fit = f10k._load_y_rows(list(range(N_TRAIN)))
    rng = np.random.default_rng(20261013)
    sample_rows = np.sort(rng.choice(N_TRAIN, 128, replace=False))
    sample_objs = [train_objs[int(i)] for i in sample_rows]
    sample_batch = sro.batch_objects(sample_objs)

    checks: dict[str, Any] = {}
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        model, info = _build_strc_model(seed, device)
        audit = info["parameter_audit"]

        # --- A1: initialisation equality -----------------------------------
        p_ref = f10k._make_downstream_model(seed, read_json(F10K / "prep_train10k.json"), device)
        ssl_state = torch.load(F10K / f"pretrain_encoder_s{seed}.pt", map_location="cpu", weights_only=False)
        mp.load_encoder_state(p_ref, ssl_state, device)
        pred_strc = _predict_strc(model, sample_objs, device)
        pred_ref = f10k.predict_bjoc(p_ref, sample_objs, device)
        init_max_diff = float(np.abs(pred_strc - pred_ref).max())
        with torch.no_grad():
            resid0 = model.residual(sample_batch).abs().max().item()
            parts = model(sample_batch, return_features=True)[1]

        # --- A2: invariance, counting, checkpoint round-trip ---------------
        base0, obj_perm = _relabelled_objects(int(sample_rows[0]))
        with torch.no_grad():
            p_base = model(sro.batch_objects([base0])).item()
            p_perm = model(sro.batch_objects([obj_perm])).item()
        with torch.no_grad():
            p_multi = model(sro.batch_objects(sample_objs[:4])).numpy()
        relabel_diff = float(abs(p_base - p_perm))
        batch_diff = float(np.abs(p_multi[0] - p_base))

        tmp_ckpt = OUT / f".cpu_precheck_seed{seed}.pt"
        torch.save(model.state_dict(), tmp_ckpt)
        model2, _ = _build_strc_model(seed, device)
        model2.load_state_dict(torch.load(tmp_ckpt, map_location="cpu", weights_only=False))
        reload_pred = _predict_strc(model2, sample_objs, device)
        reload_diff = float(np.abs(reload_pred - pred_strc).max())
        tmp_ckpt.unlink()

        a_sum_ok = bool((parts["a_counts"].sum(1).numpy() == np.asarray([o.n_atoms for o in sample_objs])).all())
        c_sum_ok = bool((parts["c_counts"].sum(1).numpy() == np.asarray([o.junc_index.shape[0] for o in sample_objs])).all())

        finite_params = all(bool(torch.isfinite(p).all()) for p in model.parameters())

        # --- A3/A4: gradients, sharing, one-step control --------------------
        model.train()
        gen = torch.Generator().manual_seed(seed + TRAIN_SHUFFLE_OFFSET)
        first = f10k._shuffle_batches(N_TRAIN, BATCH_SIZE, gen)[0]
        fb = f10k._mv(sro.batch_objects([train_objs[i] for i in first]), device)
        fy = torch.as_tensor(y_fit[first], dtype=torch.float32, device=device)
        model.zero_grad()
        pred_fb = model(fb)
        loss_fb = F.l1_loss(pred_fb, fy)
        loss_fb.backward()
        alpha_grad = model.residual.alpha.grad.detach().clone()
        beta_grad = model.residual.beta.grad.detach().clone()
        base_grads_strc = {k: p.grad.detach().clone() for k, p in model.base.named_parameters() if p.grad is not None}
        grad_finite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        clip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
        alpha_nonzero = int((alpha_grad != 0).sum())
        beta_nonzero = int((beta_grad != 0).sum())
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        all_ids = {id(p) for p in model.parameters()}
        opt.step()
        alpha_changed = bool(not torch.equal(model.residual.alpha.detach().cpu(), torch.zeros(strc.ATOM_VOCAB)))
        beta_changed = bool(not torch.equal(model.residual.beta.detach().cpu(), torch.zeros(strc.BETA_SIZE)))

        # zero-residual control: base-only gradient must equal the STRC gradient
        control, _ = _build_strc_model(seed, device)
        control.load_state_dict({k: v for k, v in model2.state_dict().items()})
        control.zero_grad()
        loss_ctrl = F.l1_loss(control.base(fb), fy)
        loss_ctrl.backward()
        ctrl_grads = {k: p.grad.detach().clone() for k, p in control.base.named_parameters() if p.grad is not None}
        base_grad_max_diff = float(
            max((base_grads_strc[k] - ctrl_grads[k]).abs().max().item() for k in base_grads_strc)
        )
        base_grad_scale = float(max(ctrl_grads[k].abs().max().item() for k in ctrl_grads))
        base_grad_relative = base_grad_max_diff / max(base_grad_scale, 1e-12)
        # independent base-only one-step optimizer with the same base state
        opt_ctrl = torch.optim.Adam(control.base.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        ctrl_clip = float(torch.nn.utils.clip_grad_norm_(control.base.parameters(), GRAD_CLIP))
        opt_ctrl.step()
        base_step_hash_equal = src.state_dict_hash(model.base) == src.state_dict_hash(control.base)
        base_step_max_diff = float(
            max((p.detach().cpu() - q.detach().cpu()).abs().max().item() for p, q in zip(model.base.parameters(), control.base.parameters()))
        )

        # per-molecule sharing identity: d y_hat / d alpha[t] == s_a * n_t
        single_obj = [train_objs[int(sample_rows[0])]]
        single_batch = sro.batch_objects(single_obj)
        model3, _ = _build_strc_model(seed, device)
        model3.zero_grad()
        single_pred = model3(single_batch)[0]
        single_pred.backward()
        a_counts_single, c_counts_single = model3.residual.count_vectors(single_batch)
        alpha_expect = model3.residual.s_a * a_counts_single[0].detach()
        beta_expect = model3.residual.s_j * c_counts_single[0].detach()
        alpha_share_diff = float((model3.residual.alpha.grad.detach() - alpha_expect).abs().max().item())
        beta_share_diff = float((model3.residual.beta.grad.detach() - beta_expect).abs().max().item())

        per_seed[str(seed)] = {
            "init": info,
            "parameter_audit": audit,
            "init_equality_max_abs_diff": init_max_diff,
            "zero_residual_max_abs": float(resid0),
            "relabelling_max_abs_diff": relabel_diff,
            "batch_composition_max_abs_diff": batch_diff,
            "checkpoint_reload_max_abs_diff": reload_diff,
            "atom_count_identity": a_sum_ok,
            "junction_count_identity": c_sum_ok,
            "params_finite": finite_params,
            "grads_finite": grad_finite,
            "alpha_grad_nonzero_entries": alpha_nonzero,
            "beta_grad_nonzero_entries": beta_nonzero,
            "alpha_changed_after_step": alpha_changed,
            "beta_changed_after_step": beta_changed,
            "base_grad_control_max_abs_diff": base_grad_max_diff,
            "base_grad_control_relative_to_max_grad": base_grad_relative,
            "base_grad_control_tolerance": BASE_GRAD_TOLERANCE,
            "clip_norm_first_batch": clip_norm,
            "control_clip_norm_first_batch": ctrl_clip,
            "clip_active": bool(clip_norm > GRAD_CLIP or ctrl_clip > GRAD_CLIP),
            "base_step_control_hash_equal": bool(base_step_hash_equal),
            "base_step_control_max_abs_diff": base_step_max_diff,
            "alpha_sharing_identity_max_abs_diff": alpha_share_diff,
            "beta_sharing_identity_max_abs_diff": beta_share_diff,
            "optimizer_contains_all_parameters": opt_ids == all_ids,
            "optimizer_has_new_parameters": id(model.residual.alpha) in opt_ids and id(model.residual.beta) in opt_ids,
            "forbidden_modules": src.forbidden_module_scan(model),
            "forbidden_source_tokens": _forbidden_source_scan(),
            "first_batch_rows_sha256": f10k.rows_sha256(first),
        }
        checks[f"seed{seed}"] = {
            "init_provenance": info["scratch_equals_recorded_p_init"] and info["composer_reader_equals_recorded_p_init"],
            "encoder_provenance": info["encoder_equals_recorded"] and info["encoder_equals_recorded_p"],
            "zero_init": info["alpha_zero_at_init"] and info["beta_zero_at_init"],
            "init_equality": init_max_diff <= 1e-6 and float(resid0) == 0.0,
            "parameter_budget": audit["parameter_exact"],
            "relabelling_invariance": relabel_diff < 1e-6,
            "batch_composition_consistency": batch_diff < 1e-6,
            "checkpoint_round_trip": reload_diff == 0.0,
            "counting_identity": a_sum_ok and c_sum_ok,
            "finite": finite_params and grad_finite,
            "gradient_activity": alpha_nonzero > 0 and beta_nonzero > 0 and alpha_changed and beta_changed,
            "base_gradient_control": base_grad_max_diff <= BASE_GRAD_TOLERANCE,
            "optimizer_membership": opt_ids == all_ids and id(model.residual.alpha) in opt_ids and id(model.residual.beta) in opt_ids,
            "sharing_identity": alpha_share_diff <= 1e-6 and beta_share_diff <= 1e-6,
            "no_forbidden_modules": not per_seed[str(seed)]["forbidden_modules"] and not per_seed[str(seed)]["forbidden_source_tokens"],
        }
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "vocab_sha256": vocab["vocab_sha256"],
        "sample": {"rows_sha256": f10k.rows_sha256(sample_rows.tolist()), "n": int(len(sample_rows))},
        "per_seed": per_seed,
        "checks": checks,
        "passed": bool(all(all(v.values()) for v in checks.values())),
        "decision": "CPU_PRECHECK_PASSED" if all(all(v.values()) for v in checks.values()) else "STRC_CPU_BLOCKED",
        "labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(CPU_PRECHECK, payload)
    log(f"[cpu-precheck] {payload['decision']}")
    if payload["decision"] != "CPU_PRECHECK_PASSED":
        log(json.dumps(f10k._j(checks), indent=1))
    return 0


# ---------------------------------------------------------------------------
# Stage: synthetic additive-task mechanism check (A5)
# ---------------------------------------------------------------------------


def _synthetic_molecules(rng: np.random.Generator, n: int, n_types: int = 12, pair_duplicate: bool = False) -> list[sro.MolObjects]:
    out: list[sro.MolObjects] = []
    for _ in range(n):
        n_atoms = int(rng.integers(6, 17))
        types = [int(t) for t in rng.integers(0, n_types, size=n_atoms)]
        edges: list[tuple[int, int, int]] = []
        for v in range(1, n_atoms):
            u = int(rng.integers(0, v))
            edges.append((u, v, int(rng.integers(0, 2))))
        n_extra = int(rng.integers(0, 3))
        existing = {(u, v) for u, v, _ in edges}
        for _ in range(n_extra):
            u, v = (int(x) for x in rng.integers(0, n_atoms, size=2))
            u, v = min(u, v), max(u, v)
            if u != v and (u, v) not in existing:
                existing.add((u, v))
                edges.append((u, v, int(rng.integers(0, 2))))
        edge_list = list(edges)
        if pair_duplicate:
            # exact per-molecule collinearity: count(type2) == count(type1).
            # Remove every initial type-2 first, then add one pendant type-2
            # atom for every type-1 atom.
            types = [3 if t == 2 else t for t in types]
            for a, t in enumerate(list(types)):
                if t == 1:
                    new_id = len(types)
                    types.append(2)
                    edge_list.append((a, new_id, 0))
        types_np = np.asarray(types, dtype=np.int64)
        x = types_np.reshape(len(types_np), 1)
        ei, ea = [], []
        for u, v, bt in edge_list:
            ei.extend([(u, v), (v, u)])
            ea.extend([bt, bt])
        edge_index = np.asarray(ei, dtype=np.int64).T if ei else np.zeros((2, 0), dtype=np.int64)
        out.append(sro.build_mol_objects(-1, x, edge_index, np.asarray(ea, dtype=np.int64)))
    return out


def _synthetic_design(objs: Sequence[sro.MolObjects], vocab: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    a_count, c_count = _count_matrix(objs, vocab)
    s_a, s_j = float(vocab["s_a"]), float(vocab["s_j"])
    return np.concatenate([s_a * a_count, s_j * c_count], axis=1), np.concatenate([a_count, c_count], axis=1)


def cmd_synthetic(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    vocab = read_json(VOCAB_JSON)
    rng = np.random.default_rng(20261013)
    n = 1024
    objs = _synthetic_molecules(rng, n)
    x_scaled, _ = _synthetic_design(objs, vocab)

    s_a, s_j = float(vocab["s_a"]), float(vocab["s_j"])
    w_a = np.zeros(strc.ATOM_VOCAB)
    for t, v in ((1, 2.0), (3, -1.5), (5, 0.8), (7, -0.6)):
        w_a[t] = v
    w_c = np.zeros(strc.BETA_SIZE)
    for slot, v in ((0, 1.2), (1, -0.9), (2, 0.7), (3, 0.5), (4, -0.4), (strc.OTHER_SLOT, 0.4)):
        w_c[slot] = v
    theta_true = np.concatenate([w_a / s_a, w_c / s_j])
    y_true = x_scaled @ theta_true
    sigma = 0.05
    y = y_true + rng.normal(0.0, sigma, size=n)

    theta_hat, _, rank, sv = np.linalg.lstsq(x_scaled, y, rcond=None)
    fitted = x_scaled @ theta_hat
    residual_rms = float(np.sqrt(np.mean((fitted - y_true) ** 2)))
    u, s, vt = np.linalg.svd(x_scaled - x_scaled.mean(0, keepdims=True), full_matrices=False)
    tol = s.max() * max(x_scaled.shape) * np.finfo(np.float64).eps
    row_space = vt[: int((s > tol).sum())]
    proj_true = row_space @ (theta_true / max(np.linalg.norm(theta_true), 1e-12))
    proj_hat = row_space @ (theta_hat / max(np.linalg.norm(theta_hat), 1e-12))
    identifiable_cos = float(proj_true @ proj_hat)
    corr_hat_true = float(np.corrcoef(theta_hat, theta_true)[0, 1])

    # gradient fit of the real STRC channel through the real forward path
    base = src.build_bjoc(
        src.BJOCConfig(), seed=7, organization_mode=src.NULL,
        global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32),
        device="cpu",
    )
    for p in base.parameters():
        p.requires_grad_(False)
    model = strc.wrap_strc(base, vocab, "cpu")
    model.residual.alpha.data.normal_(0.0, 1.0, generator=torch.Generator().manual_seed(11))
    model.residual.beta.data.normal_(0.0, 1.0, generator=torch.Generator().manual_seed(12))
    optimizer = torch.optim.Adam([model.residual.alpha, model.residual.beta], lr=5e-2)
    batch_size = 128
    y_t = torch.as_tensor(y, dtype=torch.float32)
    gen_t = torch.Generator().manual_seed(3)
    for _step in range(400):
        order = torch.randperm(n, generator=gen_t).tolist()
        for i in range(0, n, batch_size):
            idx = order[i : i + batch_size]
            sub = [objs[j] for j in idx]
            batch = sro.batch_objects(sub)
            pred = model(batch)
            loss = F.l1_loss(pred, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        resid_final = model.residual(sro.batch_objects(objs)).numpy()
    base_pred = f10k.predict_bjoc(base, objs, "cpu")
    target_resid = y - base_pred
    ols_target, _, _, _ = np.linalg.lstsq(x_scaled, target_resid, rcond=None)
    ols_target_fit = x_scaled @ ols_target
    def _r2(pred, target):
        return float(1.0 - np.mean((pred - target) ** 2) / max(np.mean((target - target.mean()) ** 2), 1e-12))
    grad_fit_r2 = _r2(resid_final, target_resid)
    ols_fit_r2 = _r2(ols_target_fit, target_resid)
    alpha_l1 = float(np.abs(model.residual.alpha.detach().numpy() - w_a / s_a).mean())
    beta_l1 = float(np.abs(model.residual.beta.detach().numpy() - w_c / s_j).mean())

    # deliberately collinear task: count(type2) == count(type1) by construction
    objs_dup = _synthetic_molecules(rng, 512, pair_duplicate=True)
    x_dup, _ = _synthetic_design(objs_dup, vocab)
    rank_dup = int(np.linalg.matrix_rank(x_dup))
    null_dup = int(x_dup.shape[1] - rank_dup)
    y_dup = x_dup @ theta_true + rng.normal(0.0, sigma, size=len(objs_dup))
    theta_dup, _, _, _ = np.linalg.lstsq(x_dup, y_dup, rcond=None)
    dup_fit_rms = float(np.sqrt(np.mean((x_dup @ theta_dup - x_dup @ theta_true) ** 2)))
    # explicit null direction e_type1 - e_type2 must satisfy X v ~= 0
    null_vec = np.zeros(x_dup.shape[1])
    null_vec[1] = 1.0
    null_vec[2] = -1.0
    null_leak = float(np.abs(x_dup @ null_vec).max())
    shifted_theta = theta_dup.copy()
    shifted_theta[1] += 1.0
    shifted_theta[2] -= 1.0
    shifted_predictions_diff = float(np.abs(x_dup @ (shifted_theta - theta_dup)).max())

    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "vocab_sha256": vocab["vocab_sha256"],
        "n_molecules": n,
        "noise_sigma": sigma,
        "design": {"n_columns": int(x_scaled.shape[1]), "rank": rank, "null_dim": int(x_scaled.shape[1] - rank)},
        "full_rank_recovery": {
            "residual_rms_vs_truth": residual_rms,
            "corr_theta_hat_theta_true": corr_hat_true,
            "identifiable_cosine": identifiable_cos,
            "passed": bool(residual_rms <= 3 * sigma and identifiable_cos > 0.99 and corr_hat_true > 0.99),
        },
        "gradient_channel_fit": {
            "l1_steps": 400,
            "r2_vs_target_residual": grad_fit_r2,
            "ols_r2_vs_target_residual": ols_fit_r2,
            "alpha_mean_abs_error_vs_planted": alpha_l1,
            "beta_mean_abs_error_vs_planted": beta_l1,
            "passed": bool(grad_fit_r2 > 0.8 and grad_fit_r2 >= ols_fit_r2 - 0.05),
        },
        "collinear_task": {
            "n_molecules": len(objs_dup),
            "rank": rank_dup,
            "null_dim": null_dup,
            "null_direction_e1_minus_e2_max_abs_Xv": null_leak,
            "shift_along_null_changes_predictions_max_abs": shifted_predictions_diff,
            "fitted_truth_rms": dup_fit_rms,
            "predictions_still_recovered": bool(dup_fit_rms <= 3 * sigma),
            "identifiability_note": (
                "with count(type2) == count(type1) the atom-type effects are not separately identifiable; "
                "only the fitted predictions, not the individual coefficients, are claimed"
            ),
        },
        "causal_claim": "function-form check only; no chemical causal meaning is claimed for any fitted coefficient",
        "labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(SYNTHETIC_JSON, payload)
    log(
        f"[synthetic] rank {rank}/{x_scaled.shape[1]} resid_rms {residual_rms:.4f} "
        f"grad_r2 {grad_fit_r2:.3f} (ols {ols_fit_r2:.3f}) dup_rank {rank_dup}"
    )
    return 0


# ---------------------------------------------------------------------------
# Stage: protocol freeze
# ---------------------------------------------------------------------------


def cmd_freeze(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = _verify_inputs()
    vocab = read_json(VOCAB_JSON)
    design = read_json(DESIGN_AUDIT)
    precheck = read_json(CPU_PRECHECK)
    synthetic = read_json(SYNTHETIC_JSON)
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_hashes": _code_hashes(),
        "inputs": inputs,
        "vocabulary": {
            "path": str(VOCAB_JSON.relative_to(REPO_ROOT)),
            "sha256": file_sha256(VOCAB_JSON),
            "vocab_sha256": vocab["vocab_sha256"],
            "size": strc.BETA_SIZE,
            "keep": strc.CONTEXT_KEEP,
            "other_slot": strc.OTHER_SLOT,
            "s_a": vocab["s_a"],
            "s_j": vocab["s_j"],
            "covered_fraction": vocab["covered_fraction"],
            "rule": "train10k junction contexts, count once, sort by (-count, lexicographic), keep 128, rest OTHER",
        },
        "design_audit": {
            "path": str(DESIGN_AUDIT.relative_to(REPO_ROOT)),
            "sha256": file_sha256(DESIGN_AUDIT),
            "rank": design["rank"],
            "null_dim": design["null_dim"],
            "condition_number": design["condition_number"],
        },
        "cpu_precheck": {
            "path": str(CPU_PRECHECK.relative_to(REPO_ROOT)),
            "sha256": file_sha256(CPU_PRECHECK),
            "decision": precheck["decision"],
        },
        "synthetic": {
            "path": str(SYNTHETIC_JSON.relative_to(REPO_ROOT)),
            "sha256": file_sha256(SYNTHETIC_JSON),
            "full_rank_passed": synthetic["full_rank_recovery"]["passed"],
            "gradient_passed": synthetic["gradient_channel_fit"]["passed"],
            "collinear_detected": synthetic["collinear_task"]["null_dim"] > 0,
        },
        "model": {
            "name": "STRC = pretrained BJOC-Null + s_a*sum(alpha[type]) + s_j*sum(beta[context])",
            "base": "identical to P: same scratch init, same SSL encoder checkpoints, same forward",
            "new_parameters": strc.NEW_PARAMETERS,
            "total_parameters": strc.TOTAL_PARAMETERS,
            "initialisation": "alpha=0, beta=0 (first forward equals P initialisation)",
            "shared_parameter_rule": "one scalar per raw atom type (28); one scalar per frozen junction context (128 + OTHER)",
            "scales": {"s_a": vocab["s_a"], "s_j": vocab["s_j"], "rule": "s_a=1/mean_train(n_atoms); s_j=1/max(mean_train(n_junctions),1)"},
        },
        "training": {
            "arm": "S",
            "seeds": list(SEEDS),
            "train_rows": [0, N_TRAIN - 1],
            "recipe": {
                "optimizer": "Adam",
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "grad_clip": GRAD_CLIP,
                "loss": "raw-y L1 only",
                "scheduler": "none",
                "dropout": 0.0,
                "epochs": EPOCHS,
                "steps_per_epoch": STEPS_PER_EPOCH,
                "updates": UPDATES,
                "soup": f"parameter mean of the last {SOUP_LAST} epoch states",
                "train_shuffle_offset": TRAIN_SHUFFLE_OFFSET,
            },
            "matched_to": "results/bjoc_full10k_pretraining_transfer_v1 soup_P_seed{0,1}.pt",
        },
        "evaluation": {
            "primary_metric": "official-valid raw-y MAE, seed-mean of the two single-model MAEs",
            "baseline": {"P_seed0": SEED0_BASELINE_MAE, "P_seed1": SEED1_BASELINE_MAE, "P_seed_mean": P_SEED_MEAN_MAE},
            "purchase_gate": {
                "seed_mean_delta_max": GAIN_THRESHOLD,
                "both_seed_deltas_negative": True,
                "ordinary_not_worse": "S ordinary MAE <= P ordinary MAE",
                "tail_strictly_better": "S tail MAE < P tail MAE",
                "rmse_tolerance": RMSE_TOLERANCE,
                "top10_removed_negative": True,
                "not_driven_by_valid_0172": True,
                "correctness_gates": True,
            },
            "bootstrap": {"n_resamples": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "unit": "molecule"},
            "freeze_order": "all prediction files hashed before the single valid label read",
            "tail_threshold": TAIL_THRESHOLD,
            "extreme_row": f"valid:{EXTREME_VALID_ROW:04d}",
        },
        "budget": {
            "wall_clock_minutes_total": 390,
            "gpu_hours_max": 0.8,
            "gpus_max": 1,
            "reserved_for_reporting_minutes": 35,
            "no_test_access": True,
        },
        "bans": [
            "ensemble as headline; prediction calibration; label clipping; target transforms; tail-weighted loss",
            "new MLP / calibration path / count-conditioned scale a(counts)*f+b(counts)",
            "extra structural features; junction / three-body / four-body modules; GNN/Transformer/attention/iteration",
            "external data; another context vocabulary size; a second scale rule; a third seed",
            "official test access in any form; using the exposed-test 0.073957 or per-molecule test errors",
            "re-tuning after the valid read; deleting rows; replacing the worst row",
        ],
        "decision_vocabulary": [
            "STRC_DEV_GAIN_SUPPORTED",
            "STRC_GAIN_WEAK_OR_MIXED",
            "STRC_NO_PREDICTIVE_GAIN",
            "STRC_CPU_BLOCKED",
        ],
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    import yaml

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    frozen = cfg.get("frozen", {}) or {}
    mismatches = []
    for key, value in (
        ("context_keep", strc.CONTEXT_KEEP),
        ("beta_size", strc.BETA_SIZE),
        ("new_parameters", strc.NEW_PARAMETERS),
        ("total_parameters", strc.TOTAL_PARAMETERS),
        ("lr", LR),
        ("weight_decay", WEIGHT_DECAY),
        ("batch_size", BATCH_SIZE),
        ("grad_clip", GRAD_CLIP),
        ("epochs", EPOCHS),
        ("updates", UPDATES),
        ("train_shuffle_offset", TRAIN_SHUFFLE_OFFSET),
        ("gain_threshold", GAIN_THRESHOLD),
        ("rmse_tolerance", RMSE_TOLERANCE),
        ("base_gradient_tolerance", BASE_GRAD_TOLERANCE),
        ("vocab_sha256", vocab["vocab_sha256"]),
        ("objects_train10k_sha256", inputs["objects_train10k"]["sha256"]),
        ("objects_valid_sha256", inputs["objects_valid"]["sha256"]),
        ("prep_train10k_sha256", inputs["prep_train10k"]["sha256"]),
        ("pretrain_encoder_s0_sha256", inputs["pretrain_encoder_s0"]["encoder_state_sha256"]),
        ("pretrain_encoder_s1_sha256", inputs["pretrain_encoder_s1"]["encoder_state_sha256"]),
        ("pretrain_encoder_s0_file_sha256", inputs["pretrain_encoder_s0"]["file_sha256"]),
        ("pretrain_encoder_s1_file_sha256", inputs["pretrain_encoder_s1"]["file_sha256"]),
        ("soup_P_seed0_sha256", inputs["soup_P_seed0"]["sha256"]),
        ("soup_P_seed1_sha256", inputs["soup_P_seed1"]["sha256"]),
    ):
        if key in frozen and frozen[key] != value:
            mismatches.append({"key": key, "config": frozen[key], "actual": value})
    payload["config_frozen_block_mismatches"] = mismatches
    write_json(FREEZE_JSON, payload)
    if mismatches:
        raise SystemExit(f"config frozen block mismatches: {mismatches}")
    if not (inputs["all_ok"] and precheck["decision"] == "CPU_PRECHECK_PASSED"):
        raise SystemExit("inputs or CPU pre-check failed; refusing to freeze")
    log(f"[freeze] protocol frozen; driver {payload['code_hashes']['driver'][:16]}")
    return 0


def verify_freeze() -> dict[str, Any]:
    if not FREEZE_JSON.exists():
        raise SystemExit("protocol not frozen; run freeze first")
    payload = read_json(FREEZE_JSON)
    code_paths = {
        "driver": _CODE_DIR / "run_bjoc_strc_v1.py",
        "module": _CODE_DIR / "bjoc_strc_v1.py",
        "base_driver": _CODE_DIR / "run_bjoc_full10k_pretraining_transfer_v1.py",
        "base_module": _CODE_DIR / "bjoc_masked_pretraining_v1.py",
        "objects": _CODE_DIR / "structural_relation_objects_v1.py",
        "composer": _CODE_DIR / "structural_relation_composer_v1.py",
        "features": _CODE_DIR / "cscl_features.py",
        "config": CONFIG_PATH,
    }
    for key, entry in payload["code_hashes"].items():
        actual = file_sha256(code_paths[key])
        if actual != entry:
            raise SystemExit(f"code hash mismatch for {key}: {actual} != {entry}")
    if read_json(CPU_PRECHECK)["decision"] != "CPU_PRECHECK_PASSED":
        raise SystemExit("CPU pre-check not passed; GPU training is blocked")
    return payload


# ---------------------------------------------------------------------------
# Stage: STRC training (GPU)
# ---------------------------------------------------------------------------


def _last_soup(states: list[dict[str, torch.Tensor]]) -> tuple[list[int], dict[str, torch.Tensor]]:
    members = list(range(EPOCHS - len(states) + 1, EPOCHS + 1))
    keys = states[0].keys()
    soup = {k: torch.stack([s[k].float() for s in states]).mean(0) for k in keys}
    return members, soup


def _train_strc(seed: int, device: torch.device, *, log_fn: Callable[[str], None] = log) -> dict[str, Any]:
    verify_freeze()
    vocab = read_json(VOCAB_JSON)
    train_objs = f10k.load_objects(F10K / "objects_train10k.pt")
    y_fit = f10k._load_y_rows(list(range(N_TRAIN)))
    model, info = _build_strc_model(seed, device)
    audit = info["parameter_audit"]
    if not audit["parameter_exact"]:
        raise SystemExit(f"parameter audit failed: {audit}")
    p_rec = read_json(F10K / f"train_P_seed{seed}.json")
    if not (info["scratch_equals_recorded_p_init"] and info["composer_reader_equals_recorded_p_init"]):
        raise SystemExit("base initialisation does not replay the recorded P initialisation")
    if not (info["alpha_zero_at_init"] and info["beta_zero_at_init"]):
        raise SystemExit("STRC parameters are not zero-initialised")
    if model.residual.vocab_sha256 != vocab["vocab_sha256"]:
        raise SystemExit("model vocabulary does not match the frozen vocabulary")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(int(seed) + TRAIN_SHUFFLE_OFFSET)
    states: deque = deque(maxlen=SOUP_LAST)
    curve: list[dict[str, Any]] = []
    order_hash = hashlib.sha256()
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        t_ep = time.perf_counter()
        for indices in f10k._shuffle_batches(N_TRAIN, BATCH_SIZE, gen):
            order_hash.update(np.ascontiguousarray(np.asarray(indices, dtype=np.int64)).tobytes())
            batch = f10k._mv(sro.batch_objects([train_objs[i] for i in indices]), device)
            y = torch.as_tensor(y_fit[indices], dtype=torch.float32, device=device)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += float((pred.view(-1) - y.view(-1)).abs().sum())
            n_mol += len(indices)
            steps += 1
        curve.append({"epoch": epoch, "updates": epoch * steps, "train_mae": total / max(n_mol, 1), "seconds": time.perf_counter() - t_ep})
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if epoch % CURVE_LOG_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            log_fn(f"[S s{seed}] ep={epoch:03d} train={curve[-1]['train_mae']:.6f} {curve[-1]['seconds']:.1f}s")
        if not np.isfinite(curve[-1]["train_mae"]):
            raise RuntimeError(f"non-finite loss at epoch {epoch} seed {seed}")
    wall = time.perf_counter() - started
    members, soup = _last_soup(list(states))
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = _predict_strc(model, train_objs, device)
    soup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(soup_state, OUT / f"soup_S_seed{seed}.pt")
    batch_order_hash = order_hash.hexdigest()
    same_order = batch_order_hash == p_rec["batch_order_hash"]
    if not same_order:
        raise SystemExit("STRC batch order differs from the frozen P batch order")
    if not info["scratch_equals_recorded_p_init"]:
        raise SystemExit("base init differs from P")
    payload = {
        "protocol": PROTOCOL_ID,
        "arm": "S",
        "seed": int(seed),
        "git_commit": git_commit(),
        "definition": "pretrained BJOC-Null (P init) + zero-init shared-type residual channel",
        "epochs": EPOCHS,
        "updates": int(EPOCHS * STEPS_PER_EPOCH),
        "steps_per_epoch": STEPS_PER_EPOCH,
        "soup_rule": f"parameter mean of the last {SOUP_LAST} epoch states",
        "soup_members": members,
        "hyper": {
            "optimizer": "Adam",
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "batch_size": BATCH_SIZE,
            "loss": "raw-y L1 only",
            "scheduler": "none",
            "dropout": 0.0,
            "train_shuffle_seed": int(seed) + TRAIN_SHUFFLE_OFFSET,
        },
        "init": info,
        "parameter_audit": audit,
        "same_init_as_P": info["scratch_equals_recorded_p_init"],
        "same_composer_reader_init_as_P": info["composer_reader_equals_recorded_p_init"],
        "same_encoder_as_P": info["encoder_equals_recorded_p"],
        "same_batch_order_as_P": bool(same_order),
        "batch_order_hash": batch_order_hash,
        "p_recorded_batch_order_hash": p_rec["batch_order_hash"],
        "wall_seconds": wall,
        "soup_fit_mae": float(np.abs(fit_pred - y_fit).mean()),
        "soup_alpha_sha256": f10k.state_group_sha256({"alpha": soup_state["residual.alpha"]}),
        "soup_beta_sha256": f10k.state_group_sha256({"beta": soup_state["residual.beta"]}),
        "soup_alpha_absmean": float(soup_state["residual.alpha"].abs().mean()),
        "soup_beta_absmean": float(soup_state["residual.beta"].abs().mean()),
        "curve": curve,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "labels_used": "official train 10000 raw y only",
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / f"train_S_seed{seed}.json", payload)
    log_fn(f"[S s{seed}] soup fit {payload['soup_fit_mae']:.5f} wall {wall:.0f}s")
    return payload


def cmd_train(args: argparse.Namespace) -> int:
    _train_strc(int(args.seed), torch.device(args.device))
    return 0


# ---------------------------------------------------------------------------
# Stage: predictions + freeze
# ---------------------------------------------------------------------------


def _load_soup_model(seed: int, device: torch.device | str = "cpu") -> strc.STRCBJOC:
    soup_path = OUT / f"soup_S_seed{seed}.pt"
    if not soup_path.exists():
        raise SystemExit(f"missing checkpoint {soup_path}")
    model, _ = _build_strc_model(seed, device)
    model.load_state_dict(torch.load(soup_path, map_location=device, weights_only=False))
    model.eval()
    return model


def cmd_predict(args: argparse.Namespace) -> int:
    verify_freeze()
    device = torch.device(args.device)
    valid_objs = f10k.load_objects(F10K / "objects_valid.pt")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    for seed in SEEDS:
        model = _load_soup_model(seed, device)
        pred = _predict_strc(model, valid_objs, device)
        parts = _predict_strc_parts(model, valid_objs, device)
        if not np.isfinite(pred).all():
            raise RuntimeError("non-finite predictions")
        path = PRED_DIR / f"valid_S{seed}.npz"
        np.savez_compressed(path, row=np.arange(N_VALID, dtype=np.int64), pred=pred.astype(np.float32))
        parts_path = PRED_DIR / f"valid_S{seed}_parts.npz"
        np.savez_compressed(
            parts_path,
            row=np.arange(N_VALID, dtype=np.int64),
            base_pred=parts["base_pred"].astype(np.float32),
            residual=parts["residual"].astype(np.float32),
            a_counts=parts["a_counts"],
            c_counts=parts["c_counts"],
        )
        entries[f"S{seed}"] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": file_sha256(path),
            "parts_path": str(parts_path.relative_to(REPO_ROOT)),
            "parts_sha256": file_sha256(parts_path),
            "checkpoint_sha256": file_sha256(OUT / f"soup_S_seed{seed}.pt"),
        }
        log(f"[predict] S{seed} valid written (checkpoint {entries[f'S{seed}']['checkpoint_sha256'][:12]})")
    write_json(
        PRED_ACCESS,
        {
            "protocol": PROTOCOL_ID,
            "git_commit": git_commit(),
            "device": str(device),
            "models": ["S0", "S1"],
            "entries": entries,
            "valid_labels_read": False,
            "official_test_loaded": False,
        },
    )
    return 0


def cmd_train_predict(args: argparse.Namespace) -> int:
    """Official-train predictions for S and P (train labels only; no valid/test)."""
    verify_freeze()
    device = torch.device(args.device)
    train_objs = f10k.load_objects(F10K / "objects_train10k.pt")
    y_fit = f10k._load_y_rows(list(range(N_TRAIN)))
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        model = _load_soup_model(seed, device)
        pred = _predict_strc(model, train_objs, device)
        np.savez_compressed(PRED_DIR / f"train_S{seed}.npz", row=np.arange(N_TRAIN, dtype=np.int64), pred=pred.astype(np.float32))
        p_model = f10k._make_downstream_model(seed, read_json(F10K / "prep_train10k.json"), device)
        p_model.load_state_dict(torch.load(F10K / f"soup_P_seed{seed}.pt", map_location=device, weights_only=False))
        p_pred = f10k.predict_bjoc(p_model, train_objs, device)
        np.savez_compressed(PRED_DIR / f"train_P{seed}.npz", row=np.arange(N_TRAIN, dtype=np.int64), pred=p_pred.astype(np.float32))
        log(f"[train-predict] seed {seed} S fit {np.abs(pred - y_fit).mean():.6f} P fit {np.abs(p_pred - y_fit).mean():.6f}")
    return 0


def cmd_freeze_predictions(args: argparse.Namespace) -> int:
    verify_freeze()
    if VALID_LOCK.exists():
        raise SystemExit("valid labels were already read; refusing to re-freeze predictions")
    files: dict[str, Any] = {}
    for key in ("S0", "S1", "P0", "P1"):
        path = PRED_DIR / f"valid_{key}.npz"
        if not path.exists():
            raise SystemExit(f"missing prediction file {path}")
        with np.load(path) as z:
            if not np.isfinite(z["pred"]).all():
                raise SystemExit(f"non-finite prediction in {path}")
        files[key] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path)}
    ckpts = {p.stem: file_sha256(p) for p in sorted(OUT.glob("soup_S_*.pt"))}
    p_freeze = read_json(F10K / "prediction_freeze.json")
    for key in ("P0", "P1"):
        expected = p_freeze["prediction_files"][key]["sha256"]
        if files[key]["sha256"] != expected:
            raise SystemExit(f"P reference prediction hash mismatch for {key}")
    combined = hashlib.sha256()
    for key in sorted(files):
        combined.update(key.encode())
        combined.update(files[key]["sha256"].encode())
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "n_prediction_files": len(files),
        "prediction_files": files,
        "checkpoints": ckpts,
        "combined_prediction_sha256": combined.hexdigest(),
        "frozen_before_valid_read": True,
        "valid_labels_read": False,
        "official_test_loaded": False,
    }
    write_json(PRED_FREEZE, payload)
    log(f"[freeze-predictions] {len(files)} files combined {payload['combined_prediction_sha256'][:16]}")
    return 0


# ---------------------------------------------------------------------------
# Stage: the ONE official-valid label read + score
# ---------------------------------------------------------------------------


def _load_pred(path: Path) -> np.ndarray:
    with np.load(path) as z:
        return z["pred"].astype(np.float64)


def _contrast(new_paths: Sequence[Path], ref_paths: Sequence[Path], y: np.ndarray, tag: str) -> dict[str, Any]:
    preds_new = [_load_pred(p) for p in new_paths]
    preds_ref = [_load_pred(p) for p in ref_paths]
    e_new = np.mean([np.abs(p - y) for p in preds_new], axis=0)
    e_ref = np.mean([np.abs(p - y) for p in preds_ref], axis=0)
    delta = e_new - e_ref
    out = {
        "tag": tag,
        "delta_seed_mean": float(delta.mean()),
        "seed_mean_mae_new": float(np.mean([np.abs(p - y).mean() for p in preds_new])),
        "seed_mean_mae_ref": float(np.mean([np.abs(p - y).mean() for p in preds_ref])),
        "per_seed_delta": {f"seed{s}": float(np.abs(preds_new[i] - y).mean() - np.abs(preds_ref[i] - y).mean()) for i, s in enumerate(SEEDS)},
        "seed_mean_rmse_new": float(np.mean([np.sqrt(np.mean((p - y) ** 2)) for p in preds_new])),
        "seed_mean_rmse_ref": float(np.mean([np.sqrt(np.mean((p - y) ** 2)) for p in preds_ref])),
        "bootstrap": f10k.paired_bootstrap(delta),
        "ordinary_y_gt_-3": {"n": int((y > TAIL_THRESHOLD).sum()), "delta": float(delta[y > TAIL_THRESHOLD].mean())},
        "tail_y_le_-3": {"n": int((y <= TAIL_THRESHOLD).sum()), "delta": float(delta[y <= TAIL_THRESHOLD].mean())},
        "better_fraction": float((delta < 0).mean()),
    }
    for k in TOP_K_REMOVED:
        out.update(f10k.top_k_removed(delta, e_new, e_ref, k))
    extreme = EXTREME_VALID_ROW
    out["extreme_row"] = {
        "row": f"valid:{extreme:04d}",
        "y": float(y[extreme]),
        "delta_row": float(delta[extreme]),
        "delta_without_row": float(np.delete(delta, extreme).mean()),
        "share_of_total_delta": float(delta[extreme] / 1000.0 / delta.mean()) if delta.mean() != 0 else None,
    }
    return out


def cmd_score(args: argparse.Namespace) -> int:
    if VALID_LOCK.exists():
        raise SystemExit("official-valid labels already read once; refusing a second read")
    verify_freeze()
    freeze = read_json(PRED_FREEZE)
    for key, entry in freeze["prediction_files"].items():
        if file_sha256(REPO_ROOT / entry["path"]) != entry["sha256"]:
            raise SystemExit(f"frozen prediction hash mismatch: {key}")
    if not freeze["frozen_before_valid_read"]:
        raise SystemExit("prediction freeze is not marked pre-label")

    val_ds = f10k.load_valid_dataset()
    y = np.asarray([float(val_ds[i].y.reshape(-1)[0]) for i in range(N_VALID)], dtype=np.float64)
    np.save(VALID_Y_COPY, y)
    enc = torch.load(f10k.ENCODED_VALID, map_location="cpu", weights_only=False)
    y_enc = np.asarray([float(enc[i].y.reshape(-1)[0]) for i in range(N_VALID)], dtype=np.float64)
    del enc
    y_crosscheck = float(np.abs(y - y_enc).max())
    if y_crosscheck != 0.0:
        raise SystemExit("raw official-valid y and the encoded-valid cache disagree")
    VALID_LOCK.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL_ID,
                "git_commit": git_commit(),
                "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "n_valid": N_VALID,
                "valid_rows_sha256": f10k.rows_sha256(range(N_VALID)),
                "y_sha256": f10k.array_sha256(y),
                "y_raw_vs_encoded_max_abs": y_crosscheck,
                "prediction_freeze_sha256": file_sha256(PRED_FREEZE),
                "extreme_row": f"valid:{EXTREME_VALID_ROW:04d}",
                "extreme_row_y": float(y[EXTREME_VALID_ROW]),
                "official_test_loaded": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    keys = ["S0", "S1", "P0", "P1"]
    pred_paths = {k: PRED_DIR / f"valid_{k}.npz" for k in keys}
    preds = {k: _load_pred(v) for k, v in pred_paths.items()}
    models = {k: f10k.row_metrics(preds[k], y) for k in keys}
    seed_mean = {}
    for arm in ("S", "P"):
        ks = [f"{arm}{s}" for s in SEEDS]
        seed_mean[arm] = {
            "mae": float(np.mean([models[k]["mae"] for k in ks])),
            "rmse": float(np.mean([models[k]["rmse"] for k in ks])),
            "median_abs": float(np.mean([models[k]["median_abs"] for k in ks])),
            "p90_abs": float(np.mean([models[k]["p90_abs"] for k in ks])),
            "p95_abs": float(np.mean([models[k]["p95_abs"] for k in ks])),
            "p99_abs": float(np.mean([models[k]["p99_abs"] for k in ks])),
            "per_seed_mae": [models[k]["mae"] for k in ks],
        }
    paired = {"S_minus_P": _contrast([pred_paths["S0"], pred_paths["S1"]], [pred_paths["P0"], pred_paths["P1"]], y, "S-P")}
    fit_gaps = {}
    for arm, base in (("S", OUT), ("P", F10K)):
        for s in SEEDS:
            rec = read_json(base / f"train_{arm}_seed{s}.json")
            fit_gaps[f"{arm}{s}"] = {
                "fit_mae": rec["soup_fit_mae"],
                "valid_mae": models[f"{arm}{s}"]["mae"],
                "fit_minus_valid": rec["soup_fit_mae"] - models[f"{arm}{s}"]["mae"],
            }
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "n_valid": N_VALID,
        "models": models,
        "seed_mean": seed_mean,
        "paired": paired,
        "fit_valid_gap": fit_gaps,
        "y_raw_vs_encoded_max_abs": y_crosscheck,
        "single_read": True,
        "official_test_loaded": False,
    }
    write_json(VALID_METRICS, payload)
    write_json(PAIRED_BOOTSTRAP, {"protocol": PROTOCOL_ID, "paired": paired, "n_resamples": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED})
    for k, m in models.items():
        log(f"[score] {k}: valid MAE {m['mae']:.6f} RMSE {m['rmse']:.6f}")
    log(f"[score] S seed-mean MAE {seed_mean['S']['mae']:.6f} vs P {seed_mean['P']['mae']:.6f}")
    return 0


# ---------------------------------------------------------------------------
# Stage: train metrics (train labels only)
# ---------------------------------------------------------------------------


def cmd_train_metrics(args: argparse.Namespace) -> int:
    if not VALID_LOCK.exists():
        raise SystemExit("run score first so the train comparison is frozen after the primary result")
    y = f10k._load_y_rows(list(range(N_TRAIN)))
    tail = y <= TAIL_THRESHOLD
    payload: dict[str, Any] = {"protocol": PROTOCOL_ID, "git_commit": git_commit(), "arms": {}}
    for arm in ("S", "P"):
        per = {}
        for s in SEEDS:
            p = _load_pred(PRED_DIR / f"train_{arm}{s}.npz")
            e = np.abs(p - y)
            per[f"{arm}{s}"] = {
                "mae": float(e.mean()),
                "mae_ordinary": float(e[~tail].mean()),
                "mae_tail": float(e[tail].mean()),
                "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            }
        e_mean = np.mean([np.abs(_load_pred(PRED_DIR / f"train_{arm}{s}.npz") - y) for s in SEEDS], axis=0)
        payload["arms"][arm] = {
            "per_seed": per,
            "seed_mean_mae": float(e_mean.mean()),
            "seed_mean_mae_ordinary": float(e_mean[~tail].mean()),
            "seed_mean_mae_tail": float(e_mean[tail].mean()),
            "signed_tail": float(np.mean([_load_pred(PRED_DIR / f"train_{arm}{s}.npz")[tail] - y[tail] for s in SEEDS])),
        }
    payload["S_minus_P_train"] = {
        "seed_mean_delta": payload["arms"]["S"]["seed_mean_mae"] - payload["arms"]["P"]["seed_mean_mae"],
        "ordinary_delta": payload["arms"]["S"]["seed_mean_mae_ordinary"] - payload["arms"]["P"]["seed_mean_mae_ordinary"],
        "tail_delta": payload["arms"]["S"]["seed_mean_mae_tail"] - payload["arms"]["P"]["seed_mean_mae_tail"],
    }
    write_json(TRAIN_METRICS, payload)
    log(f"[train-metrics] {payload['S_minus_P_train']}")
    return 0


# ---------------------------------------------------------------------------
# Stage: alpha/beta parameter analysis (label-blind)
# ---------------------------------------------------------------------------


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra**2).sum()) * float((rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def cmd_params(args: argparse.Namespace) -> int:
    vocab = read_json(VOCAB_JSON)
    design = read_json(DESIGN_AUDIT)
    tables = {}
    for s in SEEDS:
        state = torch.load(OUT / f"soup_S_seed{s}.pt", map_location="cpu", weights_only=False)
        tables[s] = {
            "alpha": state["residual.alpha"].numpy().astype(np.float64),
            "beta": state["residual.beta"].numpy().astype(np.float64),
        }
    alpha0, alpha1 = tables[0]["alpha"], tables[1]["alpha"]
    beta0, beta1 = tables[0]["beta"], tables[1]["beta"]
    atom_support = {d["type"]: d["molecules"] for d in design["atom_counts"]}
    ctx_support = {d["slot"]: d["molecules"] for d in design["context_counts"]}
    top_alpha = sorted(range(strc.ATOM_VOCAB), key=lambda t: -abs(alpha0[t] + alpha1[t]))[:8]
    top_beta = sorted(range(strc.BETA_SIZE), key=lambda k: -abs(beta0[k] + beta1[k]))[:8]
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "vocab_sha256": vocab["vocab_sha256"],
        "alpha": {"seed0": alpha0.tolist(), "seed1": alpha1.tolist()},
        "beta": {"seed0": beta0.tolist(), "seed1": beta1.tolist()},
        "cross_seed": {
            "alpha_spearman": _spearman(alpha0, alpha1),
            "alpha_pearson": float(np.corrcoef(alpha0, alpha1)[0, 1]),
            "beta_spearman": _spearman(beta0, beta1),
            "beta_pearson": float(np.corrcoef(beta0, beta1)[0, 1]),
            "alpha_max_abs_diff": float(np.abs(alpha0 - alpha1).max()),
            "beta_max_abs_diff": float(np.abs(beta0 - beta1).max()),
        },
        "top_entries": {
            "alpha": [{"type": int(t), "seed0": float(alpha0[t]), "seed1": float(alpha1[t]), "train_molecules": int(atom_support.get(int(t), 0))} for t in top_alpha],
            "beta": [
                {
                    "slot": int(k),
                    "context": (list(vocab["vocab"][k]) if k < len(vocab["vocab"]) else "OTHER"),
                    "seed0": float(beta0[k]),
                    "seed1": float(beta1[k]),
                    "train_molecules": int(ctx_support.get(int(k), 0)),
                }
                for k in top_beta
            ],
        },
        "atom_effects_at_scale": {
            "s_a": vocab["s_a"],
            "per_type_atom_shift": (float(vocab["s_a"]) * alpha0).tolist(),
        },
        "identifiability_caveats": [
            "alpha/beta are jointly trained with an opaque nonlinear reader: non-uniqueness and compensation are expected",
            "train-internal count collinearity (rank %d/%d, null dim %d) makes individual coefficients only partially identified"
            % (design["rank"], strc.ATOM_VOCAB + strc.BETA_SIZE, design["null_dim"]),
            "cross-seed agreement is a descriptive diagnostic, not a success gate and not a causal claim",
            "these are explicit additive residual contributions inside the model, not verified chemical contributions",
        ],
        "official_test_loaded": False,
    }
    write_json(PARAMS_JSON, payload)
    log(f"[params] alpha spearman {payload['cross_seed']['alpha_spearman']:.3f} beta spearman {payload['cross_seed']['beta_spearman']:.3f}")
    return 0


# ---------------------------------------------------------------------------
# Stage: error structure (Phase D)
# ---------------------------------------------------------------------------


def cmd_error_analysis(args: argparse.Namespace) -> int:
    if not VALID_LOCK.exists():
        raise SystemExit("run score first")
    y = np.load(VALID_Y_COPY)
    assert f10k.array_sha256(y) == read_json(VALID_LOCK)["y_sha256"]
    metrics = read_json(VALID_METRICS)
    valid_objs = f10k.load_objects(F10K / "objects_valid.pt")
    preds = {"S": [_load_pred(PRED_DIR / f"valid_S{s}.npz") for s in SEEDS], "P": [_load_pred(PRED_DIR / f"valid_P{s}.npz") for s in SEEDS]}
    e = {arm: np.mean([np.abs(p - y) for p in preds[arm]], axis=0) for arm in preds}
    delta = e["S"] - e["P"]
    tail = y <= TAIL_THRESHOLD
    parts = {s: {k: v for k, v in np.load(PRED_DIR / f"valid_S{s}_parts.npz").items()} for s in SEEDS}
    resid_mean = np.mean([parts[s]["residual"] for s in SEEDS], axis=0)
    base_mean = np.mean([parts[s]["base_pred"] for s in SEEDS], axis=0)
    n_junc = np.asarray([o.junc_index.shape[0] for o in valid_objs])
    n_atoms = np.asarray([o.n_atoms for o in valid_objs])

    top_k = {}
    for k in TOP_K_REMOVED:
        worst = np.argsort(-np.maximum(e["S"], e["P"]))[:k]
        keep = np.ones(N_VALID, dtype=bool)
        keep[worst] = False
        top_k[f"top{k}"] = {"delta_removed": float(delta[keep].mean()), "rows": [int(i) for i in worst]}
    order = np.argsort(-e["S"])
    error_share = {f"top{k}_share_of_S_error": float(e["S"][order[:k]].sum() / e["S"].sum()) for k in (1, 5, 10)}

    strata = {}
    for name, arr in (("n_junctions", n_junc), ("n_atoms", n_atoms)):
        med = float(np.median(arr))
        for side, mask in (("low", arr <= med), ("high", arr > med)):
            strata[f"{name}_{side}"] = {
                "n": int(mask.sum()),
                "delta": float(delta[mask].mean()),
                "S_mae": float(e["S"][mask].mean()),
                "P_mae": float(e["P"][mask].mean()),
            }
    residual_stats = {
        "seed_mean_abs_residual": float(np.abs(resid_mean).mean()),
        "seed_mean_abs_residual_ordinary": float(np.abs(resid_mean[~tail]).mean()),
        "seed_mean_abs_residual_tail": float(np.abs(resid_mean[tail]).mean()),
        "mean_abs_base": float(np.abs(base_mean).mean()),
        "residual_to_base_ratio": float(np.abs(resid_mean).mean() / max(np.abs(base_mean).mean(), 1e-12)),
        "signed_residual_mean": float(resid_mean.mean()),
        "signed_residual_mean_ordinary": float(resid_mean[~tail].mean()),
        "signed_residual_mean_tail": float(resid_mean[tail].mean()),
        "p99_abs_residual": float(np.quantile(np.abs(resid_mean), 0.99)),
    }
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "y_sha256": f10k.array_sha256(y),
        "tail": {
            "n": int(tail.sum()),
            "S_mae": float(e["S"][tail].mean()),
            "P_mae": float(e["P"][tail].mean()),
            "delta": float(delta[tail].mean()),
            "S_signed_mean": float(np.mean([preds["S"][i][tail] - y[tail] for i in range(2)])),
            "P_signed_mean": float(np.mean([preds["P"][i][tail] - y[tail] for i in range(2)])),
        },
        "ordinary": {"n": int((~tail).sum()), "S_mae": float(e["S"][~tail].mean()), "P_mae": float(e["P"][~tail].mean()), "delta": float(delta[~tail].mean())},
        "extreme_row": {
            "row": f"valid:{EXTREME_VALID_ROW:04d}",
            "y": float(y[EXTREME_VALID_ROW]),
            "S_pred_seed0": float(preds["S"][0][EXTREME_VALID_ROW]),
            "S_pred_seed1": float(preds["S"][1][EXTREME_VALID_ROW]),
            "P_pred_seed0": float(preds["P"][0][EXTREME_VALID_ROW]),
            "P_pred_seed1": float(preds["P"][1][EXTREME_VALID_ROW]),
            "delta_row": float(delta[EXTREME_VALID_ROW]),
        },
        "top_k_removed": top_k,
        "error_concentration": error_share,
        "strata": strata,
        "residual_stats": residual_stats,
        "fit_valid_gap": metrics["fit_valid_gap"],
        "official_test_loaded": False,
    }
    write_json(ERROR_ANALYSIS, payload)
    log(f"[error-analysis] tail delta {payload['tail']['delta']:+.6f} ordinary delta {payload['ordinary']['delta']:+.6f}")
    return 0


# ---------------------------------------------------------------------------
# Stage: frozen decision
# ---------------------------------------------------------------------------


def cmd_decision(args: argparse.Namespace) -> int:
    if not VALID_METRICS.exists():
        raise SystemExit("run score first")
    if not ERROR_ANALYSIS.exists() or not TRAIN_METRICS.exists():
        raise SystemExit("run error-analysis and train-metrics first")
    metrics = read_json(VALID_METRICS)
    precheck = read_json(CPU_PRECHECK)
    paired = metrics["paired"]["S_minus_P"]
    s = metrics["seed_mean"]["S"]
    p = metrics["seed_mean"]["P"]
    err = read_json(ERROR_ANALYSIS) if ERROR_ANALYSIS.exists() else None
    both_seeds_negative = all(v < 0 for v in paired["per_seed_delta"].values())
    conditions = {
        "cpu_correctness_gates": precheck["decision"] == "CPU_PRECHECK_PASSED",
        "seed_mean_gain_at_least_0_004": paired["delta_seed_mean"] <= GAIN_THRESHOLD,
        "both_seed_deltas_negative": both_seeds_negative,
        "ordinary_not_worse": err is not None and err["ordinary"]["S_mae"] <= err["ordinary"]["P_mae"],
        "tail_strictly_better": err is not None and err["tail"]["S_mae"] < err["tail"]["P_mae"],
        "rmse_within_tolerance": s["rmse"] <= p["rmse"] + RMSE_TOLERANCE,
        "top10_removed_still_negative": paired["delta_top10_removed"] < 0,
        "not_driven_by_valid_0172": paired["extreme_row"]["delta_without_row"] < 0,
    }
    failures = [k for k, v in conditions.items() if not v]
    if conditions["cpu_correctness_gates"] and all(conditions.values()):
        decision = "STRC_DEV_GAIN_SUPPORTED"
    elif not conditions["cpu_correctness_gates"]:
        decision = "STRC_CPU_BLOCKED"
    elif paired["delta_seed_mean"] < 0:
        decision = "STRC_GAIN_WEAK_OR_MIXED"
    else:
        decision = "STRC_NO_PREDICTIVE_GAIN"
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "conditions": conditions,
        "failed_conditions": failures,
        "metrics": {
            "S_seed_mean_mae": s["mae"],
            "P_seed_mean_mae": p["mae"],
            "delta_seed_mean": paired["delta_seed_mean"],
            "per_seed_delta": paired["per_seed_delta"],
            "seed_mean_rmse_delta": s["rmse"] - p["rmse"],
            "bootstrap_ci95": paired["bootstrap"],
            "ordinary_delta": None if err is None else err["ordinary"]["delta"],
            "tail_delta": None if err is None else err["tail"]["delta"],
            "top10_removed_delta": paired["delta_top10_removed"],
            "delta_without_extreme_row": paired["extreme_row"]["delta_without_row"],
        },
        "purchase_gate": {
            "seed_mean_delta_max": GAIN_THRESHOLD,
            "rmse_tolerance": RMSE_TOLERANCE,
            "baseline": {"P_seed0": SEED0_BASELINE_MAE, "P_seed1": SEED1_BASELINE_MAE, "P_seed_mean": P_SEED_MEAN_MAE},
        },
        "decision": decision,
        "stop_rule": (
            "STRC_DEV_GAIN_SUPPORTED keeps S as the next-stage development candidate but does not query official test; "
            "STRC_GAIN_WEAK_OR_MIXED / STRC_NO_PREDICTIVE_GAIN stop the shared additive residual route permanently "
            "(no vocabulary/scale/tail-weighting/third-seed follow-up); STRC_CPU_BLOCKED stops before any GPU work."
        ),
        "official_valid_result_only": True,
        "official_test_loaded": False,
    }
    write_json(SCIENTIFIC_DECISION, payload)
    log(f"[decision] {decision} failures={failures}")
    return 0


# ---------------------------------------------------------------------------
# remote CPU chain
# ---------------------------------------------------------------------------


def cmd_stage_cpu_prep(args: argparse.Namespace) -> int:
    if not VOCAB_JSON.exists():
        cmd_vocab(args)
    else:
        vocab = read_json(VOCAB_JSON)
        if strc.vocabulary_sha256(vocab) != vocab["vocab_sha256"]:
            raise SystemExit("existing context vocabulary failed its hash")
    if not DESIGN_AUDIT.exists():
        cmd_design_audit(args)
    # the pre-check, synthetic audit and freeze are deterministic and cheap;
    # always re-run them so a corrected revision cannot silently reuse a
    # blocked pre-check artifact from an earlier deployment
    cmd_cpu_precheck(args)
    cmd_synthetic(args)
    cmd_freeze(argparse.Namespace())
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("vocab").set_defaults(func=cmd_vocab)
    sub.add_parser("design-audit").set_defaults(func=cmd_design_audit)
    sub.add_parser("cpu-precheck").set_defaults(func=cmd_cpu_precheck)
    sub.add_parser("synthetic").set_defaults(func=cmd_synthetic)
    sub.add_parser("freeze").set_defaults(func=cmd_freeze)
    p = sub.add_parser("train")
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(func=cmd_train)
    sub.add_parser("predict").set_defaults(func=cmd_predict)
    sub.add_parser("train-predict").set_defaults(func=cmd_train_predict)
    sub.add_parser("freeze-predictions").set_defaults(func=cmd_freeze_predictions)
    sub.add_parser("score").set_defaults(func=cmd_score)
    sub.add_parser("train-metrics").set_defaults(func=cmd_train_metrics)
    sub.add_parser("params").set_defaults(func=cmd_params)
    sub.add_parser("error-analysis").set_defaults(func=cmd_error_analysis)
    sub.add_parser("decision").set_defaults(func=cmd_decision)
    sub.add_parser("stage-cpu-prep").set_defaults(func=cmd_stage_cpu_prep)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
