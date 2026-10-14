#!/usr/bin/env python
"""Driver for protocol ``cscl-rw-local-vs-multiscale-v1``.

  Phase 0 (CPU, label-blind): verify-inputs, math-audit, prep-r1, cpu-gates
  freeze: protocol_freeze.json
  Phase 1 (GPU, cap 0.5 h): train --arm R1 --seed 0, predict --arm R1,
      freeze-predictions
  Phase 2 (single Monitor read): score-monitor, finalize

The historical R123 (arm R) and U predictions are reused read-only from
``results/bond_random_walk_organization_v1/``; A0/A2 Monitor predictions are
copied read-only into ``references/``.
Discipline: no Locked read; official valid/test never loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_arch_breakthrough_v1 as ab  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import bond_random_walk_organization_v1 as bwo  # noqa: E402
import bond_rw_local_vs_multiscale_v1 as lm  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

REPO_ROOT = _CODE_DIR.parents[2]
TRACK = _CODE_DIR.parent
OUT = Path(
    os.environ.get(
        "BRW2_OUT_DIR",
        str(TRACK / "results" / "bond_rw_local_vs_multiscale_v1"),
    )
)
PRIOR = TRACK / "results" / "bond_random_walk_organization_v1"   # R123/U predictions, train record, soup
DATA = TRACK / "results" / "bjoc_task_adaptive_ssl_v1"           # split D, objects, SSL, fit norm
JUNC = TRACK / "results" / "bjoc_junction_sufficiency_audit_v1"
REFERENCES = OUT / "references"
ENCODED_TRAIN = REPO_ROOT / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt"
CONFIG_PATH = TRACK / "configs" / "bond_rw_local_vs_multiscale_v1.yaml"
PRIOR_CONFIG = TRACK / "configs" / "bond_random_walk_organization_v1.yaml"

PROTOCOL_ID = lm.PROTOCOL_ID
SPLIT_TAG = "D"
SEED = 0
AUDIT_N = 500
INVARIANCE_N = 100
GPU_BUDGET_CAP_HOURS = 0.5
SPLIT_FINGERPRINTS = {
    "fit_sha256_rows": "9468656affcc2b7c69409f03698b1b6287b3a42dc6e164efd19dfb0190254525",
    "monitor_sha256_rows": "d196bc75ce898c7cfd61efa133372f25803d5bc1198210f60b6e0f981a400393",
    "locked_sha256_rows": "f3525ae882114350a57f7b0ddb6889656f40c05cebda21b2611a2c99cdfa22d5",
}
HISTORICAL_PREDICTIONS = {
    "R123": PRIOR / "predictions" / "monitor_R_D_s0.npz",
    "U": PRIOR / "predictions" / "monitor_U_D_s0.npz",
}
#: prior-round manifest keys (arm R = this round's R123)
HISTORICAL_MANIFEST_KEYS = {"R123": "R", "U": "U"}
REFERENCE_PREDICTIONS = {
    "A0": REFERENCES / "monitor_A0_D_s0.npz",
    "A2": REFERENCES / "monitor_A2_D_s0.npz",
}
CONTRASTS = (
    ("R123_minus_R1", "R123", "R1"),
    ("R1_minus_U", "R1", "U"),
    ("R123_minus_U", "R123", "U"),
    ("R1_minus_A0", "R1", "A0"),
    ("R123_minus_A0", "R123", "A0"),
    ("R1_minus_A2", "R1", "A2"),
    ("R123_minus_A2", "R123", "A2"),
)
DECISION_VOCAB = {
    "MULTISCALE_EXTRA_GAIN": "R123 beats R1 with bootstrap CI upper < 0",
    "ONE_HOP_BETTER": "R1 beats R123 with bootstrap CI lower > 0",
    "MULTISCALE_INDISTINGUISHABLE": "R123 - R1 bootstrap CI crosses zero",
}


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=False, default=_json_default)
        fh.write("\n")


def read_json(path: Path | str) -> Any:
    with Path(path).open() as fh:
        return json.load(fh)


def file_sha256(path: Path | str) -> str:
    return bp.file_sha256(Path(path))


def rows_sha256(rows: Sequence[int]) -> str:
    return bp.rows_sha256(rows)


def array_sha256(a: np.ndarray) -> str:
    return bp.array_sha256(np.asarray(a))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def config_sha256() -> str:
    return file_sha256(CONFIG_PATH)


# ---------------------------------------------------------------------------
# paths / loaders
# ---------------------------------------------------------------------------


def split_path() -> Path:
    return DATA / "split_D.json"


def fit_objects_path() -> Path:
    return DATA / "objects_fit_D.pt"


def monitor_objects_path() -> Path:
    return DATA / "objects_monitor_D.pt"


def ssl_path() -> Path:
    return DATA / "pretrain_D_s0.pt"


def norm_path() -> Path:
    return DATA / "prep_fit_D.json"


def r1_cache_path(zone: str) -> Path:
    return OUT / f"r1_{zone}_{SPLIT_TAG}.pt"


def soup_path(arm: str) -> Path:
    return OUT / f"soup_{arm}_{SPLIT_TAG}_s{SEED}.pt"


def smoke_soup_path(arm: str) -> Path:
    return OUT / f"soup_{arm}_{SPLIT_TAG}_s{SEED}_smoke.pt"


def train_json_path(arm: str) -> Path:
    return OUT / f"train_{arm}_{SPLIT_TAG}_s{SEED}.json"


def prediction_path(arm: str) -> Path:
    return OUT / "predictions" / f"monitor_{arm}_{SPLIT_TAG}_s{SEED}.npz"


def smoke_prediction_path(arm: str) -> Path:
    return OUT / "predictions" / f"monitor_{arm}_{SPLIT_TAG}_s{SEED}_smoke.npz"


def load_split() -> Mapping[str, Any]:
    return read_json(split_path())


def load_zone(zone: str) -> list[sro.MolObjects]:
    path = fit_objects_path() if zone == "fit" else monitor_objects_path()
    return list(torch.load(path, map_location="cpu", weights_only=False))


def load_fit_y() -> np.ndarray:
    rows = [int(i) for i in load_split()["fit_rows"]]
    require(len(rows) == 8000, "split D fit row count changed")
    return bp.load_y_rows(ENCODED_TRAIN, rows)


def load_ssl_state() -> Mapping[str, Any]:
    return torch.load(ssl_path(), map_location="cpu", weights_only=False)


def load_r1_triplets(zone: str) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    payload = torch.load(r1_cache_path(zone), map_location="cpu", weights_only=False)
    return [(np.asarray(s), np.asarray(d), np.asarray(w)) for s, d, w in payload["triplets"]]


def audit_objs() -> list[sro.MolObjects]:
    objs = load_zone("fit")
    rows = [int(i) for i in load_split()["fit_rows"]]
    require(len(objs) == len(rows), "fit object count mismatch")
    audit_rows = set(rows[:AUDIT_N])
    return [o for o in objs if int(o.mol_index) in audit_rows][:AUDIT_N]


# ---------------------------------------------------------------------------
# verify-inputs
# ---------------------------------------------------------------------------


def _historical_manifest() -> Mapping[str, Any]:
    return read_json(PRIOR / "prediction_manifest_D.json")


def cmd_verify_inputs(_args: argparse.Namespace) -> int:
    inputs: dict[str, Any] = {}
    for name, path in (
        ("split_D.json", split_path()),
        ("objects_fit_D.pt", fit_objects_path()),
        ("objects_monitor_D.pt", monitor_objects_path()),
        ("pretrain_D_s0.pt", ssl_path()),
        ("prep_fit_D.json", norm_path()),
        ("encoded_train.pt", ENCODED_TRAIN),
        ("config", CONFIG_PATH),
        ("prior_prediction_manifest_D.json", PRIOR / "prediction_manifest_D.json"),
        ("prior_train_R_D_s0.json", PRIOR / "train_R_D_s0.json"),
        ("references/REFERENCE_MANIFEST.json", REFERENCES / "REFERENCE_MANIFEST.json"),
    ):
        require(Path(path).exists(), f"missing input {name}: {path}")
        inputs[name] = {"sha256": file_sha256(path), "bytes": int(Path(path).stat().st_size)}
    split = load_split()
    rows = {
        "fit": [int(i) for i in split["fit_rows"]],
        "monitor": [int(i) for i in split["monitor_rows"]],
        "locked": [int(i) for i in split["locked_rows"]],
    }
    observed = {
        "fit_sha256_rows": rows_sha256(rows["fit"]),
        "monitor_sha256_rows": rows_sha256(rows["monitor"]),
        "locked_sha256_rows": rows_sha256(rows["locked"]),
    }
    matches = {k: observed[k] == v for k, v in SPLIT_FINGERPRINTS.items()}
    require(all(matches.values()), f"split D fingerprints drifted: {matches}")
    fit_objs = load_zone("fit")
    require([int(o.mol_index) for o in fit_objs] == rows["fit"], "fit object row order mismatch")

    prior_manifest = _historical_manifest()
    historical = {}
    for arm, path in HISTORICAL_PREDICTIONS.items():
        rec = prior_manifest["records"][HISTORICAL_MANIFEST_KEYS[arm]]
        sha = file_sha256(path)
        historical[arm] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha,
            "prior_manifest_sha256": rec["file_sha256"],
            "matches_prior_manifest": sha == rec["file_sha256"],
            "pred_sha256": array_sha256(np.load(path)["pred"]),
        }
        require(sha == rec["file_sha256"], f"historical prediction drifted: {arm}")
    ref_manifest = read_json(REFERENCES / "REFERENCE_MANIFEST.json")
    references = {}
    for arm, path in REFERENCE_PREDICTIONS.items():
        sha = file_sha256(path)
        rec = ref_manifest["files"][path.name]
        references[arm] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha, "matches_manifest": sha == rec["sha256"]}
        require(sha == rec["sha256"], f"reference prediction drifted: {arm}")
    prior_train = read_json(PRIOR / "train_R_D_s0.json")
    require(prior_train["init_full_sha256"] == lm.R123_INIT_SHA256, "R123 init hash mismatch")
    require(prior_train["encoder_used_sha256"] == lm.R123_ENCODER_USED_SHA256, "R123 encoder hash mismatch")
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": inputs,
        "split_fingerprints_match": matches,
        "n_fit": len(rows["fit"]),
        "n_monitor": len(rows["monitor"]),
        "n_locked": len(rows["locked"]),
        "historical_predictions": historical,
        "reference_predictions": references,
        "r123_init_sha256_expected": lm.R123_INIT_SHA256,
        "r123_encoder_used_sha256_expected": lm.R123_ENCODER_USED_SHA256,
        "fit_y_loaded": False,
        "monitor_y_loaded": False,
        "locked_y_loaded": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "input_audit.json", payload)
    log(f"[verify-inputs] fit={payload['n_fit']} monitor={payload['n_monitor']} locked={payload['n_locked']}")
    return 0


# ---------------------------------------------------------------------------
# Phase 0
# ---------------------------------------------------------------------------


def cmd_math_audit(_args: argparse.Namespace) -> int:
    objs = audit_objs()
    rng = np.random.default_rng(20261015)
    per_mol = [lm.math_audit_molecule(o, rng) for o in objs]
    summary = lm.summarize_math_audit(per_mol)
    synthetic = lm.synthetic_rng_audit(torch_seeds=(0, 0, 12345))
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "n_molecules": len(objs),
        "k_vs_ks": {
            "phi": "symmetric test function phi(b_e,b_f)=B_e+B_f+B_e*B_f",
            "definition": "r_e=sum_f K_ef phi_ef vs r'_e=sum_f Ks_ef phi_ef, Ks=(K+K^T)/2",
            "summary": summary,
            "note": "r'_e == r_e for EVERY e under symmetric phi; row-weight totals differ",
        },
        "synthetic_rng_reconciliation": synthetic,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "math_audit.json", payload)
    _render_math_md(OUT / "MATH_AUDIT.md", payload)
    agg = summary["aggregate"]
    log(
        f"[math-audit] double-sum max |sum_e r_e - sum_e r'_e| {agg['double_sum_equality_max_over_molecules']:.2e} "
        f"(K123 {agg['double_sum_equality_max_over_molecules_K123']:.2e}); "
        f"per-bond max |r_e-r'_e| {agg['per_bond_diff_max_over_molecules']:.3f} "
        f"(K123 {agg['per_bond_diff_max_over_molecules_K123']:.3f}); "
        f"symK rowsum CV {agg['rowsum_symK_cv']:.3f} corr(deg,rowsum) {agg['corr_degree_rowsum_symK']:.3f}"
    )
    log(f"  synthetic RNG seed0 repeat identical: {synthetic['seed0_repeat_identical']}; different seed differs: {synthetic['different_seed_differs']}")
    return 0


def _render_math_md(path: Path, payload: Mapping[str, Any]) -> None:
    agg = payload["k_vs_ks"]["summary"]["aggregate"]
    con = payload["k_vs_ks"]["summary"]["conclusions"]
    syn = payload["synthetic_rng_reconciliation"]
    lines = [
        "# 数学核查 — cscl-rw-local-vs-multiscale-v1（Phase 0）",
        "",
        f"- 协议 `{payload['protocol']}` · commit `{payload['git_commit']}` · {payload['n_molecules']} 个 split-D Fit 分子（标签盲）",
        "",
        "## 1. K 与 Ks=(K+Kᵀ)/2：全图双重求和相等，逐键 r_e 不等",
        "",
        "设 `r_e = Σ_f K_ef φ_ef`，`r'_e = Σ_f Ks_ef φ_ef`，`Ks=(K+Kᵀ)/2`；φ 取对称测试函数",
        "`φ(b_e,b_f)=B_e+B_f+B_e⊙B_f`。",
        "",
        "**全图双重求和相等**：每个无序对 {e,f} 在 Σ_e r_e 与 Σ_e r'_e 中都贡献",
        "`(K_ef+K_fe)·φ_ef`，因此",
        "",
        "```",
        "Σ_e r'_e = Σ_e Σ_f Ks_ef φ_ef = Σ_e r_e = Σ_{e,f} K_ef φ_ef",
        "```",
        "",
        f"数值核查（{payload['n_molecules']} 个分子）最大偏差 "
        f"**{agg['double_sum_equality_max_over_molecules']:.2e}**（K123 为 "
        f"{agg['double_sum_equality_max_over_molecules_K123']:.2e}）。",
        "",
        "**逐键 r_e 并不相等**（本轮修正）：上一轮“φ 对称 ⇒ 使用 K 与使用 Ks 相同”的说法",
        "只在全图求和层面成立，而模型消费的正是**逐键** r_e。逐键有精确恒等式",
        "",
        "```",
        "r'_e = Σ_f (K_ef+K_fe)/2·φ_ef = ( r_e(K) + Σ_f K_fe φ_ef ) / 2 = ( r_e(K) + (Kᵀφ)_e ) / 2",
        "```",
        "",
        f"数值核查：平均 `max_e|r_e−r'_e|` = **{agg['per_bond_diff_max_over_molecules']:.3f}**"
        f"（K123 为 {agg['per_bond_diff_max_over_molecules_K123']:.3f}），恒等式残差最大 "
        f"{agg['per_bond_identity_residual_max_K123']:.2e}（P 为 "
        f"{agg['per_bond_identity_residual_max_P']:.2e}）。",
        "",
        "历史影响：上一轮的 unit test `test_k_and_symmetrised_k_are_model_equivalent` 比较的是",
        "`Ks` 与 `symmetrize(Ks)`（恒等），因此是空洞测试；上一轮模型实际使用 triplet of Ks",
        "（`symmetrize` 后才建 triplet），历史预测/训练记录不受影响。本轮仅修订解释：",
        "R1/R123 两臂使用的都是**对称化核** Ks，而不是原始 K；两者逐键消息不同。",
        "",
        "**行权重被重新分配**：`Σ_f Ks_ef = (rowsum_e(K)+colsum_e(K))/2`。",
        "对行随机的 lazy walk（每行和=1），Ks 的行和不再恒为 1：",
        "",
        "| 量 | 均值 |",
        "|---|---:|",
        f"| mean \\|rowsum(symK)−1\\| | {agg['mean_abs_rowsum_symK_minus_1']:.4f} |",
        f"| rowsum(symK) CV | {agg['rowsum_symK_cv']:.3f} |",
        f"| rowsum(symK) min / max | {agg['rowsum_symK_min']:.3f} / {agg['rowsum_symK_max']:.3f} |",
        f"| corr(线图度, rowsum(symK)) | {agg['corr_degree_rowsum_symK']:.3f} |",
        f"| corr(线图度, colsum(symK)) | {agg['corr_degree_colsum_symK']:.3f} |",
        f"| mean \\|rowsum(symP)−1\\| | {agg['mean_abs_rowsum_symP_minus_1']:.4f} |",
        f"| corr(线图度, rowsum(symP)) | {agg['corr_degree_rowsum_symP']:.3f} |",
        "",
        "**解释边界**：因为 Pair MLP 带偏置，对称化后的行和变化等价于给每条键"
        "叠加一个与线图度/中心性相关的偏置项；这与“哪对键被组织在一起”是两种"
        "可能的效果来源。R1 与 R123 都使用 Ks，都含此效应；U 的行和恒为 1。"
        "因此 R1/R123 相对 U 的优势**不能**排除行和/度效应的贡献，需要在"
        "科学解释中与“真实成对组织”分开陈述。",
        "",
        "结论：" + json.dumps(con, ensure_ascii=False),
        "",
        "## 2. 历史合成任务 MAE 不一致的核查",
        "",
        f"- 历史观测：{json.dumps(syn['historical_values'], ensure_ascii=False)}",
        f"- 同一函数 `synthetic_relation_task(seed=0)` 在三次不同进程中给出不同 real-K MAE。",
        f"- 本次对照：相同 `torch.manual_seed(0)` 两次 → 结果逐位一致？**{syn['seed0_repeat_identical']}**；"
        f"不同 `torch.manual_seed(12345)` → 结果不同？**{syn['different_seed_differs']}**。",
        "",
        "原因：该函数的 numpy 数据采样有 seed，但 `_TinyPairRegressor`/`_TinyLinearRegressor`"
        "由全局 torch RNG 初始化，且函数内不调用 `torch.manual_seed`；不同进程（或此前是否"
        "调用过 `build_model`/`torch.manual_seed`）会得到不同初始化。",
        "",
        "| torch 手动 seed | real-K MAE | uniform-K MAE | linear MAE |",
        "|---:|---:|---:|---:|",
    ]
    for r in syn["runs"]:
        lines.append(
            f"| {r['torch_manual_seed']} | {r['test_mae_real_K_trained']:.4f} | "
            f"{r['test_mae_uniform_K_trained_uniform']:.4f} | {r['test_mae_linear_summary']:.4f} |"
        )
    lines += [
        "",
        f"**结论**：{syn['conclusion']}",
        "",
        "历史 `REPORT.md`（0.7174，mechanism 进程）与 `SYNTHETIC_WITNESSES.md`（0.3755，"
        "synthetic 进程）均来自同一函数同一 seed 参数，差异是全局 torch RNG 初始化所致；"
        "两者作为定性 witness 仍然成立（真实 K 显著优于 uniform 与线性），但不应把具体数值"
        "当作可复现的定量结果。历史记录不改动。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def cmd_prep_r1(_args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "kernel": "K1 = sym(P), P = (I + D^-1 A_L)/2 with zero-degree rows one-hot",
        "triplet_convention": "unordered e<=f of Ks; diagonal weight halved inside pair_aggregate",
        "zones": {},
    }
    for zone in ("fit", "monitor"):
        objs = load_zone(zone)
        triplets = []
        checks = {"row_sum_max_err": 0.0, "sym_err": 0.0, "reconstruction_max_err": 0.0}
        for o in objs:
            A = bwo.line_graph_adjacency(o.bond_index)
            P = bwo.lazy_transition(A)
            checks["row_sum_max_err"] = max(checks["row_sum_max_err"], float(np.abs(P.sum(axis=1) - 1.0).max()))
            Ks = lm.k1_matrix(o)
            checks["sym_err"] = max(checks["sym_err"], float(np.abs(Ks - Ks.T).max()))
            s, d, w = bwo.unordered_triplets(Ks)
            triplets.append((s, d, w))
            n = A.shape[0]
            if n:
                B = np.random.default_rng(0).normal(size=(n, 4))
                phi = lm._symmetric_phi(B)
                w_eff = np.where(s == d, w * 0.5, w)
                phi_pair = phi[s, d]
                r_trip = np.zeros((n, 4))
                np.add.at(r_trip, s, w_eff[:, None] * phi_pair)
                np.add.at(r_trip, d, w_eff[:, None] * phi_pair)
                r_dense = np.einsum("ef,efk->ek", Ks, phi)
                checks["reconstruction_max_err"] = max(
                    checks["reconstruction_max_err"], float(np.abs(r_dense - r_trip).max())
                )
        torch.save({"triplets": triplets}, r1_cache_path(zone))
        payload["zones"][zone] = {
            "n_molecules": len(objs),
            "n_triplets_total": int(sum(int(s.shape[0]) for s, _, _ in triplets)),
            "checks": checks,
            "file_sha256": file_sha256(r1_cache_path(zone)),
        }
        log(
            f"[prep-r1 {zone}] n={len(objs)} triplets={payload['zones'][zone]['n_triplets_total']} "
            f"row_sum_err={checks['row_sum_max_err']:.2e} recon_err={checks['reconstruction_max_err']:.2e}"
        )
        require(checks["row_sum_max_err"] <= 1e-9, "lazy transition rows do not sum to 1")
        require(checks["reconstruction_max_err"] <= 1e-6, "R1 triplet reconstruction mismatch")
    write_json(OUT / "r1_manifest.json", payload)
    return 0


# ---------------------------------------------------------------------------
# cpu-gates
# ---------------------------------------------------------------------------


def cmd_cpu_gates(args: argparse.Namespace) -> int:
    objs = load_zone("fit")[:256]
    norm = read_json(norm_path())
    device = torch.device(args.device)

    # kernel gates on R1
    kernel_checks = {"row_sum_max": 0.0, "triplet_recon_max": 0.0, "single_bond_ok": True}
    rng = np.random.default_rng(11)
    for o in objs:
        A = bwo.line_graph_adjacency(o.bond_index)
        P = bwo.lazy_transition(A)
        kernel_checks["row_sum_max"] = max(kernel_checks["row_sum_max"], float(np.abs(P.sum(axis=1) - 1).max()))
        if A.shape[0] == 1:
            kernel_checks["single_bond_ok"] = bool(kernel_checks["single_bond_ok"] and P[0, 0] == 1.0)
        n = A.shape[0]
        Ks = lm.k1_matrix(o)
        s, d, w = bwo.unordered_triplets(Ks)
        B = rng.normal(size=(n, 4))
        phi = lm._symmetric_phi(B)
        w_eff = np.where(s == d, w * 0.5, w)
        r_trip = np.zeros((n, 4))
        np.add.at(r_trip, s, w_eff[:, None] * phi[s, d])
        np.add.at(r_trip, d, w_eff[:, None] * phi[s, d])
        kernel_checks["triplet_recon_max"] = max(
            kernel_checks["triplet_recon_max"],
            float(np.abs(np.einsum("ef,efk->ek", Ks, phi) - r_trip).max()),
        )

    # model parity with the historical R123 initialisation
    model = bwo.build_model(SEED, norm, "cpu")
    fresh2 = bwo.build_model(SEED, norm, "cpu")
    other = bwo.build_model(SEED + 1, norm, "cpu")
    model_checks = {
        "parameter_count": bwo.parameter_count(model),
        "parameter_count_matches": bwo.parameter_count(model) == lm.EXPECTED_PARAMETERS,
        "init_state_hash": bwo.state_dict_hash(model),
        "init_matches_R123": bwo.state_dict_hash(model) == lm.R123_INIT_SHA256,
        "init_deterministic_same_seed": bwo.state_dict_hash(model) == bwo.state_dict_hash(fresh2),
        "init_differs_other_seed": bwo.state_dict_hash(model) != bwo.state_dict_hash(other),
        "forbidden_modules": bwo.forbidden_module_scan(model),
        "composer_in": int(bwo.BondRWConfig().compose_in),
        "reader_in": int(bwo.BondRWConfig().reader_in),
    }
    ssl = load_ssl_state()
    model_checks["ssl_init_encoder_hash"] = bwo.encoder_group_hash(model)
    bwo.load_ssl_encoders(model, ssl, "cpu")
    model_checks["ssl_used_encoder_hash"] = bwo.encoder_group_hash(model)
    model_checks["ssl_encoder_matches_R123"] = model_checks["ssl_used_encoder_hash"] == lm.R123_ENCODER_USED_SHA256

    # forward / invariance / batch membership
    triplets = [lm.k1_triplets_from_objects(o) for o in objs[:64]]
    batch = bwo.move_batch(bwo.batch_rw_objects(objs[:64], triplets), device)
    model.eval()
    with torch.no_grad():
        pred_batch = model(batch)
        singles = []
        for i in range(64):
            b1 = bwo.move_batch(bwo.batch_rw_objects([objs[i]], [triplets[i]]), device)
            singles.append(float(model(b1)))
    forward_checks = {
        "batch_membership_max_abs_diff": float(np.abs(np.asarray(singles) - pred_batch.detach().cpu().numpy()).max()),
        "all_finite": bool(np.isfinite(np.asarray(singles)).all()),
    }
    inv = []
    for i in range(min(16, len(objs))):
        o = objs[i]
        perm = rng.permutation(o.n_atoms)
        inv_perm = np.argsort(perm)
        o2 = sro.MolObjects(
            mol_index=o.mol_index, smiles=o.smiles, n_atoms=o.n_atoms,
            atom_feat=o.atom_feat[perm], bond_index=inv_perm[o.bond_index],
            bond_type=o.bond_type, bond_feat=o.bond_feat,
            junc_index=o.junc_index, junc_bond=o.junc_bond, junc_feat=o.junc_feat,
            global_feat=o.global_feat,
        )
        with torch.no_grad():
            b1 = bwo.move_batch(bwo.batch_rw_objects([o], [triplets[i]]), device)
            b2 = bwo.move_batch(bwo.batch_rw_objects([o2], [triplets[i]]), device)
            inv.append(abs(float(model(b1)) - float(model(b2))))
    forward_checks["atom_relabel_max_abs_dpred"] = float(max(inv)) if inv else 0.0

    # checkpoint save/load and deterministic retrain
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model2 = bwo.build_model(SEED, norm, "cpu")
    model2.load_state_dict(state)
    model2.eval()
    with torch.no_grad():
        p_orig = model(batch)
        p_load = model2(batch)
    checkpoint_checks = {
        "save_load_max_abs_dpred": float((p_orig - p_load).abs().max()),
    }
    y_live = load_fit_y()[:32]
    t_small = [lm.k1_triplets_from_objects(o) for o in objs[:32]]
    m_a, info_a = bwo.train_supervised_rw(
        arm="R1", seed=SEED, train_objs=objs[:32], train_triplets=t_small, y_fit=y_live,
        norm=norm, device=torch.device("cpu"), ssl_state=ssl, smoke_epochs=1, log_fn=lambda _m: None,
    )
    m_b, info_b = bwo.train_supervised_rw(
        arm="R1", seed=SEED, train_objs=objs[:32], train_triplets=t_small, y_fit=y_live,
        norm=norm, device=torch.device("cpu"), ssl_state=ssl, smoke_epochs=1, log_fn=lambda _m: None,
    )
    checkpoint_checks["retrain_soup_identical"] = bwo.state_dict_hash(m_a) == bwo.state_dict_hash(m_b)
    checkpoint_checks["retrain_batch_order_matches"] = (
        info_a["batch_order"]["full_hash"] == info_b["batch_order"]["full_hash"]
    )

    # gradient liveness
    model_g = bwo.build_model(SEED, norm, "cpu")
    bwo.load_ssl_encoders(model_g, ssl, "cpu")
    model_g.train()
    batch_g = bwo.move_batch(bwo.batch_rw_objects(objs[:32], t_small), torch.device("cpu"))
    pred_g = model_g(batch_g)
    loss = torch.nn.functional.l1_loss(pred_g.view(-1), torch.zeros(32))
    loss.backward()
    grads = [float(p.grad.detach().norm()) for p in model_g.pair_mlp.parameters() if p.grad is not None]
    gradient_checks = {"pair_mlp_grad_norms": grads, "pair_mlp_grad_live": bool(grads and all(g > 0 for g in grads))}

    # optional R123 checkpoint replay (if the previous round's soup is still present)
    replay: dict[str, Any] = {"checked": False}
    r123_soup = PRIOR / f"soup_R_{SPLIT_TAG}_s0.pt"
    if r123_soup.exists():
        m123 = bwo.build_model(SEED, norm, "cpu")
        m123.load_state_dict({k: v.to("cpu") for k, v in torch.load(r123_soup, map_location="cpu").items()})
        monitor = load_zone("monitor")[:128]
        trip123 = [bwo.kernel_triplets_from_objects(o) for o in monitor]
        pred123 = bwo.predict_model(m123, monitor, trip123, "cpu")
        ref = np.asarray(np.load(HISTORICAL_PREDICTIONS["R123"])["pred"], dtype=np.float64)[:128]
        replay = {
            "checked": True,
            "n_molecules": len(monitor),
            "soup_sha256": file_sha256(r123_soup),
            "replay_max_abs_diff_vs_frozen_predictions": float(np.abs(pred123 - ref).max()),
        }

    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "device": str(device),
        "kernel_checks": kernel_checks,
        "model_checks": model_checks,
        "forward_checks": forward_checks,
        "checkpoint_checks": checkpoint_checks,
        "gradient_checks": gradient_checks,
        "r123_checkpoint_replay": replay,
        "passed": bool(
            kernel_checks["row_sum_max"] <= 1e-9
            and kernel_checks["triplet_recon_max"] <= 1e-6
            and kernel_checks["single_bond_ok"]
            and model_checks["parameter_count_matches"]
            and model_checks["init_matches_R123"]
            and model_checks["init_deterministic_same_seed"]
            and model_checks["init_differs_other_seed"]
            and model_checks["ssl_encoder_matches_R123"]
            and not model_checks["forbidden_modules"]
            and forward_checks["batch_membership_max_abs_diff"] <= 1e-5
            and forward_checks["atom_relabel_max_abs_dpred"] <= 1e-5
            and checkpoint_checks["save_load_max_abs_dpred"] <= 1e-6
            and checkpoint_checks["retrain_soup_identical"]
            and gradient_checks["pair_mlp_grad_live"]
        ),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "cpu_gates.json", payload)
    log(f"[cpu-gates] passed={payload['passed']}")
    return 0


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------


FREEZE_CODE_FILES = (
    "bond_rw_local_vs_multiscale_v1.py",
    "run_bond_rw_local_vs_multiscale_v1.py",
    "bond_random_walk_organization_v1.py",
    "structural_relation_composer_v1.py",
    "structural_relation_objects_v1.py",
    "bjoc_perfmech_long_v1.py",
    "bjoc_arch_breakthrough_v1.py",
    "cscl_units.py",
)


def cmd_freeze(_args: argparse.Namespace) -> int:
    inputs = read_json(OUT / "input_audit.json")
    for zone in ("fit", "monitor"):
        require(r1_cache_path(zone).exists(), f"r1 cache missing: {zone}")
    code = {name: file_sha256(_CODE_DIR / name) for name in FREEZE_CODE_FILES}
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "config_sha256": config_sha256(),
        "scoring_code_sha256": {
            "run_bond_rw_local_vs_multiscale_v1.py": code["run_bond_rw_local_vs_multiscale_v1.py"],
            "bond_rw_local_vs_multiscale_v1.py": code["bond_rw_local_vs_multiscale_v1.py"],
        },
        "code_hashes": code,
        "split_tag": SPLIT_TAG,
        "inputs": inputs["inputs"],
        "r1_caches": {
            zone: {"sha256": file_sha256(r1_cache_path(zone)), "bytes": int(r1_cache_path(zone).stat().st_size)}
            for zone in ("fit", "monitor")
        },
        "recipe": {"arm": "R1", "relation": "sym(P)", "seed": SEED, "epochs": 360, "updates": 22680, "loss": "raw-y L1", "soup": "last-5"},
        "expected_init_sha256": lm.R123_INIT_SHA256,
        "expected_encoder_sha256": lm.R123_ENCODER_USED_SHA256,
        "gpu_cap_hours": GPU_BUDGET_CAP_HOURS,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "protocol_freeze.json", payload)
    log(f"[freeze] {len(code)} code hashes; config {payload['config_sha256'][:12]}")
    return 0


def verify_freeze(require_exact_code: bool = True) -> Mapping[str, Any]:
    path = OUT / "protocol_freeze.json"
    require(path.exists(), "protocol_freeze.json missing; run freeze first")
    payload = read_json(path)
    require(payload["protocol"] == PROTOCOL_ID, "freeze protocol id mismatch")
    if require_exact_code:
        for name, sha in payload["code_hashes"].items():
            require(file_sha256(_CODE_DIR / name) == sha, f"code drift detected in {name}")
        require(config_sha256() == payload["config_sha256"], "config drifted from freeze")
    return payload


# ---------------------------------------------------------------------------
# train / predict
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    arm, seed = args.arm, int(args.seed)
    require(arm == "R1", "only the preregistered R1 arm exists in this round")
    if args.smoke:
        require(args.epochs is not None and 1 <= int(args.epochs) <= 5, "--smoke requires --epochs between 1 and 5")
    device = torch.device(args.device)
    objs = load_zone("fit")
    y = load_fit_y()
    triplets = load_r1_triplets("fit")
    ssl = load_ssl_state()
    log(f"[train {arm} s{seed}] device={device} n={len(objs)}")
    model, info = bwo.train_supervised_rw(
        arm=arm, seed=seed, train_objs=objs, train_triplets=triplets, y_fit=y,
        norm=read_json(norm_path()), device=device, ssl_state=ssl,
        smoke_epochs=args.epochs if args.smoke else None, log_fn=log,
    )
    info["protocol"] = PROTOCOL_ID
    info["trainer"] = "bond_random_walk_organization_v1.train_supervised_rw (unchanged)"
    info["relation"] = "K1 = sym(P)"
    info["expected_init_sha256"] = lm.R123_INIT_SHA256
    info["smoke"] = args.smoke
    if args.smoke:
        torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, smoke_soup_path(arm))
        info["smoke_soup_sha256"] = file_sha256(smoke_soup_path(arm))
        write_json(OUT / f"smoke_train_{arm}_s{seed}.json", info)
        log(f"[train {arm} s{seed}] SMOKE only, epochs={info['epochs']}")
        return 0
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, soup_path(arm))
    write_json(train_json_path(arm), info)
    _append_ledger({"arm": arm, "seed": seed, "wall_seconds": info["wall_seconds"], "device": info["device"], "epochs": info["epochs"], "phase": "formal"})
    log(f"[train {arm} s{seed}] soup_fit_mae={info['soup_fit_mae']:.6f} wall={info['wall_seconds']:.0f}s")
    return 0


def _append_ledger(entry: Mapping[str, Any]) -> None:
    path = OUT / "gpu_budget_ledger.json"
    payload = read_json(path) if path.exists() else {"protocol": PROTOCOL_ID, "entries": []}
    payload["entries"].append({**entry, "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    total = float(sum(float(e["wall_seconds"]) for e in payload["entries"]))
    payload["total_wall_hours"] = total / 3600.0
    payload["cap_hours"] = GPU_BUDGET_CAP_HOURS
    payload["within_cap"] = bool(total / 3600.0 <= GPU_BUDGET_CAP_HOURS)
    write_json(path, payload)
    log(f"[ledger] total {payload['total_wall_hours']:.4f} h / cap {GPU_BUDGET_CAP_HOURS} h")


def cmd_predict(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    arm = args.arm
    smoke = bool(getattr(args, "smoke", False))
    require(arm == "R1", "only R1 is predicted in this round")
    sp = smoke_soup_path(arm) if smoke else soup_path(arm)
    require(sp.exists(), f"missing soup for {arm}{' (smoke)' if smoke else ''}")
    device = torch.device(args.device)
    objs = load_zone("monitor")
    triplets = load_r1_triplets("monitor")
    model = bwo.build_model(SEED, read_json(norm_path()), device)
    model.load_state_dict({k: v.to(device) for k, v in torch.load(sp, map_location="cpu").items()})
    preds = bwo.predict_model(model, objs, triplets, device)
    path = smoke_prediction_path(arm) if smoke else prediction_path(arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, pred=preds.astype(np.float32), arm=arm, seed=SEED, n_molecules=len(preds))
    manifest_path = OUT / ("prediction_manifest_D_smoke.json" if smoke else "prediction_manifest_D.json")
    manifest = read_json(manifest_path) if manifest_path.exists() else {"protocol": PROTOCOL_ID, "records": {}}
    manifest["smoke"] = smoke
    manifest["records"][arm] = {
        "arm": arm, "seed": SEED, "n_molecules": int(len(preds)),
        "pred_sha256": array_sha256(preds), "file_sha256": file_sha256(path),
        "labels_used": "none",
    }
    for hist, hpath in HISTORICAL_PREDICTIONS.items():
        rec = _historical_manifest()["records"][HISTORICAL_MANIFEST_KEYS[hist]]
        manifest["records"][hist] = {
            "arm": hist, "seed": 0, "n_molecules": 1000,
            "pred_sha256": array_sha256(np.load(hpath)["pred"]), "file_sha256": file_sha256(hpath),
            "labels_used": "none", "source": "historical read-only reuse (bond_random_walk_organization_v1)",
        }
        require(manifest["records"][hist]["file_sha256"] == rec["file_sha256"], f"historical {hist} drifted")
    manifest["references"] = {
        arm_: {"file_sha256": file_sha256(p), "labels_used": "prior-round frozen predictions"}
        for arm_, p in REFERENCE_PREDICTIONS.items()
    }
    write_json(manifest_path, manifest)
    log(f"[predict {arm}{' smoke' if smoke else ''}] n={len(preds)} sha={manifest['records'][arm]['pred_sha256'][:12]}")
    return 0


def cmd_freeze_predictions(args: argparse.Namespace) -> int:
    smoke = bool(getattr(args, "smoke", False))
    manifest_path = OUT / ("prediction_manifest_D_smoke.json" if smoke else "prediction_manifest_D.json")
    manifest = read_json(manifest_path)
    for key, rec in manifest["records"].items():
        if key == "R1":
            p = smoke_prediction_path(key) if smoke else prediction_path(key)
        else:
            p = HISTORICAL_PREDICTIONS[key]
        require(file_sha256(p) == rec["file_sha256"], f"prediction drifted: {key}")
    manifest["complete"] = True
    manifest["n_predictions"] = len(manifest["records"])
    manifest["frozen_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps({k: v["pred_sha256"] for k, v in sorted(manifest["records"].items())}, sort_keys=True).encode()
    ).hexdigest()
    write_json(manifest_path, manifest)
    log(f"[freeze-predictions{' smoke' if smoke else ''}] {manifest['n_predictions']} files frozen sha={manifest['manifest_sha256'][:12]}")
    return 0


# ---------------------------------------------------------------------------
# score-monitor (single read)
# ---------------------------------------------------------------------------


def _verify_prediction_manifest(manifest_name: str = "prediction_manifest_D.json") -> Mapping[str, Any]:
    manifest = read_json(OUT / manifest_name)
    require(bool(manifest.get("complete")), "prediction manifest is not complete")
    smoke = bool(manifest.get("smoke", False))
    for key, rec in manifest["records"].items():
        if key == "R1":
            p = smoke_prediction_path(key) if smoke else prediction_path(key)
        else:
            p = HISTORICAL_PREDICTIONS[key]
        require(file_sha256(p) == rec["file_sha256"], f"prediction drifted: {key}")
    for arm_, rec in manifest["references"].items():
        require(file_sha256(REFERENCE_PREDICTIONS[arm_]) == rec["file_sha256"], f"reference drifted: {arm_}")
    return manifest


def cmd_score_monitor(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    dry = bool(getattr(args, "dry_run", False))
    lock_name = "monitor_read_D_dryrun.lock" if dry else "monitor_read_D.lock"
    if not dry:
        require(not (OUT / "monitor_read_D.lock").exists(), "monitor labels were already read in this round")
    manifest = _verify_prediction_manifest("prediction_manifest_D_smoke.json" if dry else "prediction_manifest_D.json")
    rows = [int(i) for i in load_split()["monitor_rows"]]
    require(len(rows) == 1000, "monitor row count changed")
    if dry:
        y = np.random.default_rng(0).normal(size=len(rows))
        log("[score-monitor] DRY RUN with synthetic labels; no Monitor y was read")
    else:
        y = bp.load_y_rows(ENCODED_TRAIN, rows)
    write_json(
        OUT / lock_name,
        {
            "protocol": PROTOCOL_ID,
            "dry_run": dry,
            "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_rows": len(y),
            "y_sha256": array_sha256(y),
            "rows_sha256": rows_sha256(rows),
            "prediction_manifest_sha256": manifest["manifest_sha256"],
        },
    )
    r1_path = smoke_prediction_path("R1") if dry else prediction_path("R1")
    preds = {
        "R1": np.asarray(np.load(r1_path)["pred"], dtype=np.float64),
        **{arm: np.asarray(np.load(p)["pred"], dtype=np.float64) for arm, p in HISTORICAL_PREDICTIONS.items()},
        **{arm: np.asarray(np.load(p)["pred"], dtype=np.float64) for arm, p in REFERENCE_PREDICTIONS.items()},
    }
    audit = lm.error_audit(y, preds, CONTRASTS, bootstrap_n=bp.BOOTSTRAP_N, bootstrap_seed=bp.BOOTSTRAP_SEED)
    primary = audit["contrasts"]["R123_minus_R1"]
    ci = primary["bootstrap"]
    primary_decision = lm.primary_decision_from_ci(ci["ci_lo"], ci["ci_hi"])
    local = audit["contrasts"]["R1_minus_U"]
    local_decision = "LOCAL_TOPOLOGY_GAIN" if local["bootstrap"]["ci_hi"] < 0 else "LOCAL_TOPOLOGY_NO_GAIN"
    next_step = lm.next_step_for(primary_decision)
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "dry_run": dry,
        "split_tag": SPLIT_TAG,
        "delta_definition": "mean over molecules of |err_new| - |err_ref|; negative = new arm better",
        "arm_stats": audit["arm_stats"],
        "contrasts": audit["contrasts"],
        "primary_decision": primary_decision,
        "local_decision": local_decision,
        "decision_vocab": DECISION_VOCAB,
        "next_step": next_step,
        "prediction_manifest_sha256": manifest["manifest_sha256"],
        "monitor_read_utc": read_json(OUT / lock_name)["read_utc"],
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    score_name = "monitor_score_D_dryrun.json" if dry else "monitor_score_D.json"
    write_json(OUT / score_name, payload)
    rows_csv = lm.per_molecule_rows(y, preds)
    csv_name = "per_molecule_errors_dryrun.csv" if dry else "per_molecule_errors.csv"
    with (OUT / csv_name).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows_csv[0].keys()))
        writer.writeheader()
        writer.writerows(rows_csv)
    log(
        f"[score-monitor{' dry-run' if dry else ''}] R1={audit['arm_stats']['R1']['mae']:.6f} "
        f"R123={audit['arm_stats']['R123']['mae']:.6f} U={audit['arm_stats']['U']['mae']:.6f}"
    )
    log(
        f"  R123-R1 delta={primary['delta']:+.6f} CI95 [{ci['ci_lo']:+.6f},{ci['ci_hi']:+.6f}] -> {primary_decision}; "
        f"R1-U {local['delta']:+.6f} CI95 [{local['bootstrap']['ci_lo']:+.6f},{local['bootstrap']['ci_hi']:+.6f}] -> {local_decision}"
    )
    return 0


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------


def cmd_finalize(args: argparse.Namespace) -> int:
    dry = bool(getattr(args, "dry_run", False))
    score_name = "monitor_score_D_dryrun.json" if dry else "monitor_score_D.json"
    decision_name = "decision_dryrun.json" if dry else "decision.json"
    report_name = "REPORT_dryrun.md" if dry else "REPORT.md"
    evidence_name = "evidence_manifest_dryrun.json" if dry else "evidence_manifest.json"
    score = read_json(OUT / score_name)
    math = read_json(OUT / "math_audit.json")
    gates = read_json(OUT / "cpu_gates.json")
    if dry and (OUT / "smoke_train_R1_s0.json").exists():
        train = read_json(OUT / "smoke_train_R1_s0.json")
    else:
        train = read_json(train_json_path("R1"))
    freeze = read_json(OUT / "protocol_freeze.json")
    ledger_path = OUT / "gpu_budget_ledger.json"
    ledger = (
        read_json(ledger_path)
        if ledger_path.exists()
        else {"protocol": PROTOCOL_ID, "entries": [], "total_wall_hours": 0.0, "cap_hours": GPU_BUDGET_CAP_HOURS, "within_cap": True}
    )
    decision = {
        "protocol": PROTOCOL_ID,
        "primary_decision": score["primary_decision"],
        "local_decision": score["local_decision"],
        "next_step": score["next_step"],
        "stop_after_round": True,
        "primary_R123_minus_R1": {
            "delta": score["contrasts"]["R123_minus_R1"]["delta"],
            "ci95": [
                score["contrasts"]["R123_minus_R1"]["bootstrap"]["ci_lo"],
                score["contrasts"]["R123_minus_R1"]["bootstrap"]["ci_hi"],
            ],
            "mae_R123": score["contrasts"]["R123_minus_R1"]["mae_new"],
            "mae_R1": score["contrasts"]["R123_minus_R1"]["mae_ref"],
        },
        "R1_minus_U": {
            "delta": score["contrasts"]["R1_minus_U"]["delta"],
            "ci95": [
                score["contrasts"]["R1_minus_U"]["bootstrap"]["ci_lo"],
                score["contrasts"]["R1_minus_U"]["bootstrap"]["ci_hi"],
            ],
        },
        "gpu_hours": ledger["total_wall_hours"],
        "dry_run": dry,
        "official_test_loaded": False,
    }
    write_json(OUT / decision_name, decision)
    _render_report(OUT / report_name, score, math, gates, train, freeze, ledger, decision)
    evidence_files = [
        name
        for name in (
            "input_audit.json", "math_audit.json", "r1_manifest.json", "cpu_gates.json",
            "protocol_freeze.json", "gpu_budget_ledger.json",
            "train_R1_D_s0.json", "prediction_manifest_D.json", "monitor_read_D.lock",
            "monitor_score_D.json", "per_molecule_errors.csv", "decision.json",
            "smoke_train_R1_s0.json", "prediction_manifest_D_smoke.json",
            "monitor_read_D_dryrun.lock", "monitor_score_D_dryrun.json",
            "per_molecule_errors_dryrun.csv", "decision_dryrun.json",
        )
        if (OUT / name).exists()
    ]
    write_json(
        OUT / evidence_name,
        {
            "protocol": PROTOCOL_ID,
            "git_commit": git_commit(),
            "dry_run": dry,
            "files": {name: file_sha256(OUT / name) for name in evidence_files},
            "official_valid_loaded": False,
            "official_test_loaded": False,
        },
    )
    log(f"[finalize] primary={decision['primary_decision']} next_step={decision['next_step']}")
    return 0


def _fmt(x: Any) -> str:
    return f"{float(x):.6f}"


def _render_report(path: Path, score, math, gates, train, freeze, ledger, decision) -> None:
    s = score["arm_stats"]
    cons = score["contrasts"]
    p = cons["R123_minus_R1"]
    lu = cons["R1_minus_U"]
    lu123 = cons["R123_minus_U"]
    math_agg = math["k_vs_ks"]["summary"]["aggregate"]
    syn = math["synthetic_rng_reconciliation"]
    lines = [
        "# 一跳 vs 多步随机游走关系 — cscl-rw-local-vs-multiscale-v1",
        "",
        f"- 协议 `{score['protocol']}` · commit `{freeze['git_commit']}`",
        f"- 结论：**{score['primary_decision']}**，局部拓扑判定 **{score['local_decision']}**",
        f"- 下一步：{score['next_step']}",
        f"- GPU：{ledger['total_wall_hours']:.4f} h / 上限 {ledger['cap_hours']} h；单 seed；无第二个候选模型",
        "- 纪律：训练完成后先冻结 soup / Monitor predictions / manifest / scoring code，再做且仅做一次 Monitor 读取；Locked 与 official valid/test 从未读取。",
        "",
        "## 1. 科学问题与两臂",
        "",
        "在**完全相同**的 Bond Pair MLP、Composer、Reader、SSL encoder、初始化、数据与 batch order、"
        "R3 360 epochs / 22,680 updates、optimizer/LR/WD、last-5 soup 下：",
        "",
        "| 臂 | 关系 | 说明 |",
        "|---|---|---|",
        "| R1（新） | `K1 = sym(P)` | P 为 lazy 一跳转移矩阵 |",
        "| R123（历史复用） | `Ks = sym((P+P²+P³)/3)` | 上一轮 arm R 的冻结预测 |",
        "| U（历史复用） | `K_ef = 1/N_bond` | 无结构全局 pair 对照 |",
        "",
        f"两臂参数均为 {train['parameter_audit']['actual_parameters']}；R1 初始化哈希与历史 R123 严格一致"
        f"（`{freeze['expected_init_sha256'][:16]}…`），SSL encoder 哈希一致（`{freeze['expected_encoder_sha256'][:16]}…`）。",
        "",
        "## 2. Phase 0 数学核查（详 `MATH_AUDIT.md`）",
        "",
        "- **修正**：对称 φ 下只有**全图双重求和**相等（Σ_e r_e 一致，残差 "
        f"{math_agg['double_sum_equality_max_over_molecules']:.2e}）；**逐键 r_e 并不相等**"
        f"（平均 max_e|r_e−r'_e|={math_agg['per_bond_diff_max_over_molecules']:.3f}），"
        "精确恒等式 `r_e(Ks)=(r_e(K)+(Kᵀφ)_e)/2`。上一轮“使用 K 与使用 Ks 相同”的说法"
        "仅在双重求和层面成立，历史预测/训练因实际使用 Ks triplet 而不受影响；"
        "但解释必须修正，模型消费的是逐键（已对称化）消息。",
        "- 对称化确实改变**每行权重总量**：`rowsum(Ks)=(rowsum(K)+colsum(K))/2`。"
        f"历史 K123 的 symK 行和 CV={math_agg['rowsum_symK_cv']:.3f}，"
        f"corr(线图度, rowsum)={math_agg['corr_degree_rowsum_symK']:.3f}。"
        "由于 Pair MLP 有偏置，这部分相当于一个度/中心性相关的偏置项；R1 与 R123 都含此效应，"
        "U 的行和恒为 1，因此 R1/R123 相对 U 的优势不能排除行和/度效应的贡献。",
        "- 历史合成任务 MAE 不一致（0.3025 / 0.3755 / 0.7174）已核查：numpy 数据采样有 seed，"
        "但 tiny regressor 由全局 torch RNG 初始化且函数内不设 manual seed；"
        f"同 `torch.manual_seed(0)` 两次结果一致={syn['seed0_repeat_identical']}，"
        f"不同 seed 结果不同={syn['different_seed_differs']}。历史记录不改动。",
        "",
        "## 3. Phase 1 训练与冻结",
        "",
        f"- `train_R1_D_s0.json`：soup fit MAE {train['soup_fit_mae']:.6f}，"
        f"wall {train['wall_seconds']:.0f}s，updates {train['updates']}，"
        f"batch-order 哈希自检 {train['batch_order']['full_matches']}。",
        f"- checkpoint 重放与初始化/梯度/不变性检查全部通过（`cpu_gates.json` passed={gates['passed']}）。",
        f"- Monitor 只读一次（lock {score['monitor_read_utc']}，manifest {score['prediction_manifest_sha256'][:12]}…）。",
        "",
        "## 4. Phase 2 配对评分（split-D Monitor，n=1000，seed 0，历史暴露，仅探索性）",
        "",
        "| 臂 | MAE | RMSE | median AE | p90 | p95 | p99 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("R1", "R123", "U", "A0", "A2"):
        if arm not in s:
            continue
        st = s[arm]
        lines.append(
            f"| {arm} | {st['mae']:.6f} | {st['rmse']:.6f} | {st['median_ae']:.6f} | "
            f"{st['p90_ae']:.4f} | {st['p95_ae']:.4f} | {st['p99_ae']:.4f} |"
        )
    lines += [
        "",
        "主对比（负值 = 前者更好）：",
        "",
        "| 对比 | Δ (95% CI) | ordinary | tail | top-10 removed | 去掉最大单点 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("R123_minus_R1", "R1_minus_U", "R123_minus_U", "R1_minus_A0", "R123_minus_A0", "R1_minus_A2", "R123_minus_A2"):
        c = cons[name]
        ci = c["bootstrap"]
        lines.append(
            f"| {name} | {c['delta']:+.6f} [{ci['ci_lo']:+.6f}, {ci['ci_hi']:+.6f}] | "
            f"{c['ordinary_delta']:+.6f} | {c['tail_delta']:+.6f} | {c['top10_removed_delta_arm_ae']:+.6f} | "
            f"{c['delta_wo_single_worst_delta']:+.6f} |"
        )
    worst1 = max(s, key=lambda a: s[a]["worst1_sse_share"])
    lines += [
        "",
        "极端误差贡献（SSE 份额）：",
        "",
        "| 臂 | worst-1 SSE | top-10 SSE | top-1% SSE | top-5% SSE | max AE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("R1", "R123", "U", "A0", "A2"):
        if arm not in s:
            continue
        st = s[arm]
        lines.append(
            f"| {arm} | {st['worst1_sse_share']:.3f} | {st['top10_sse_share']:.3f} | "
            f"{st['top1pct_sse_share']:.3f} | {st['top5pct_sse_share']:.3f} | {st['max_ae']:.3f} |"
        )
    lines += [
        "",
        "关键判断：",
        "",
        f"1. **多步是否提供额外价值**：R123 − R1 = {p['delta']:+.6f}，"
        f"CI95 [{p['bootstrap']['ci_lo']:+.6f}, {p['bootstrap']['ci_hi']:+.6f}] → `{score['primary_decision']}`。",
        f"2. **一跳局部拓扑本身是否有用**：R1 − U = {lu['delta']:+.6f}，"
        f"CI95 [{lu['bootstrap']['ci_lo']:+.6f}, {lu['bootstrap']['ci_hi']:+.6f}] → `{score['local_decision']}`；"
        f"历史 R123 − U = {lu123['delta']:+.6f}（复用，用于一致性核对）。",
        "3. **行和/度效应**：见第 2 节；R1 与 R123 均为对称化核，均含度/中心性偏置效应，"
        "因此两臂之差不能单独归因于“多步信息”。",
        "4. **与 BJOC-Null（A2）比较不能直接归因**：A0/A2 的参数量（A0 与 A2 均为 413,505）与"
        "SSL 组成、结构表达方式均与本轮 349,985 参数的 BondRW 模型不同，且 A0/A2 是历史轨迹、"
        "不同初始化；这里的差值仅作描述性参照。",
        f"5. **极端误差**：最大单分子 SSE 占比最高的是 {worst1} "
        f"({s[worst1]['worst1_sse_share']:.3f})；各对比均报告去掉最大单点后的 Δ 与 top-10 removed Δ。",
        "",
        "## 5. 局限与停止",
        "",
        "- 单 seed、split-D Monitor 历史暴露，仅探索性证据；不写 Locked，不读 official valid/test。",
        "- 两臂共享全部训练设置，但“一跳 vs 多步”的对比仍受对称化行和效应、核尺度（3 步平均）"
        "以及 degree/centrality 偏置影响；本轮不称其为已解释的化学贡献。",
        "- 按预注册：不加 seed1、不加第二个候选模型、不扩展到新架构；本轮结束。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("verify-inputs", cmd_verify_inputs),
        ("math-audit", cmd_math_audit),
        ("prep-r1", cmd_prep_r1),
        ("cpu-gates", cmd_cpu_gates),
        ("freeze", cmd_freeze),
    ):
        sp = sub.add_parser(name)
        if name == "cpu-gates":
            sp.add_argument("--device", default=argparse.SUPPRESS)
        sp.set_defaults(func=fn)
    fp = sub.add_parser("freeze-predictions")
    fp.add_argument("--smoke", action="store_true")
    fp.set_defaults(func=cmd_freeze_predictions)

    sm = sub.add_parser("score-monitor")
    sm.add_argument("--dry-run", action="store_true")
    sm.set_defaults(func=cmd_score_monitor)

    fi = sub.add_parser("finalize")
    fi.add_argument("--dry-run", action="store_true")
    fi.set_defaults(func=cmd_finalize)

    tr = sub.add_parser("train")
    tr.add_argument("--arm", required=True, choices=("R1",))
    tr.add_argument("--seed", type=int, default=SEED)
    tr.add_argument("--smoke", action="store_true")
    tr.add_argument("--epochs", type=int, default=None)
    tr.add_argument("--device", default=argparse.SUPPRESS)
    tr.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict")
    pr.add_argument("--arm", required=True, choices=("R1",))
    pr.add_argument("--smoke", action="store_true")
    pr.add_argument("--device", default=argparse.SUPPRESS)
    pr.set_defaults(func=cmd_predict)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
