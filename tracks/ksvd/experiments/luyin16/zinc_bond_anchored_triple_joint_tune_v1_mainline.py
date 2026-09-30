"""Runner stages for BondAnchoredTriple-JointTune-v1.

Round ``e2e_dictenv_bond_anchored_triple_joint_tune_v1`` (study
``zinc-context-gap``).  Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_bond_anchored_triple_joint_tune_v1_preregistration.md``.

Stages: ``structure`` -> ``references`` -> ``preflight`` -> ``correctness`` ->
``train`` -> ``analysis``.  Local CPU; official valid is evaluation only and the
official test split is never instantiated.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_bond_anchored_triple_joint_tune_v1 as jt
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_bond_anchored_triple_v1 as bat
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = jt.PROTOCOL_VERSION
RESULTS_DIR = jt.RESULTS_DIR
CHECKPOINT_DIR = jt.CHECKPOINT_DIR
PREREG_PATH = jt.PREREG_PATH

_write_json = jt._write_json
_read_json = jt._read_json
_sha256_file = jt._sha256_file
_git_commit = p2run._git_commit

VERDICT_PROMISING = jt.VERDICT_PROMISING
VERDICT_NO_SIGNAL = jt.VERDICT_NO_SIGNAL
VERDICT_INVALID = jt.VERDICT_INVALID
PROMISING_THRESHOLD = jt.PROMISING_THRESHOLD


def _ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], header: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in header})


def _preregistration_payload() -> dict[str, Any]:
    return {
        "path": str(PREREG_PATH.relative_to(REPO_ROOT)),
        "sha256": _sha256_file(PREREG_PATH),
        "sha256_prefix": _sha256_file(PREREG_PATH)[:12],
    }


def _require_preregistration(references: Mapping[str, Any]) -> None:
    if references["preregistration"]["sha256"] != _sha256_file(PREREG_PATH):
        raise RuntimeError("pre-registration changed after references; refusing")


# ---------------------------------------------------------------------------
# stage: structure (data-only triples; no trainable weights)
# ---------------------------------------------------------------------------


def stage_structure(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "structure_report.json"
    if path.exists() and not force:
        report = _read_json(path)
        if report.get("keys", {}).get("train") == jt._structure_key("train") and report.get(
            "keys", {}
        ).get("valid") == jt._structure_key("valid"):
            return report
    started = time.perf_counter()
    train_structure = jt.build_or_load_structure("train", force=force)
    valid_structure = jt.build_or_load_structure("valid", force=force)
    counts = {
        "train": {
            "graphs": int(train_structure.n_graphs),
            "pairs": int(train_structure.n_pairs),
            "triples": int(train_structure.n_triples),
        },
        "valid": {
            "graphs": int(valid_structure.n_graphs),
            "pairs": int(valid_structure.n_pairs),
            "triples": int(valid_structure.n_triples),
        },
    }
    for split, expected in jt.EXPECTED_STRUCTURE_COUNTS.items():
        if counts[split] != {key: int(value) for key, value in expected.items()}:
            raise RuntimeError(f"{split} structure counts {counts[split]} != {expected}")
    cross_check: dict[str, Any] = {}
    for split, structure in (("train", train_structure), ("valid", valid_structure)):
        cache = bat.build_or_load_cache(split)
        cross_check[split] = {
            "bat_cache_path": str(bat.cache_path(split).relative_to(REPO_ROOT)),
            "bat_cache_key": bat._cache_key(split),
            "counts_equal": bool(
                int(cache.n_pairs) == structure.n_pairs and int(cache.n_triples) == structure.n_triples
            ),
            "triple_ij_equal": bool(torch.equal(cache.triple_ij.long(), structure.triple_ij.long())),
            "triple_ik_equal": bool(torch.equal(cache.triple_ik.long(), structure.triple_ik.long())),
            "triple_jk_equal": bool(torch.equal(cache.triple_jk.long(), structure.triple_jk.long())),
        }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "preregistration": _preregistration_payload(),
        "module_sha256": jt.module_sha256(),
        "keys": {"train": jt._structure_key("train"), "valid": jt._structure_key("valid")},
        "paths": {
            "train": str(jt.structure_path("train").relative_to(REPO_ROOT)),
            "valid": str(jt.structure_path("valid").relative_to(REPO_ROOT)),
        },
        "counts": counts,
        "expected_counts": {
            key: {k: int(v) for k, v in value.items()}
            for key, value in jt.EXPECTED_STRUCTURE_COUNTS.items()
        },
        "bat_cache_cross_check": cross_check,
        "provenance": {
            "train": train_structure.provenance,
            "valid": valid_structure.provenance,
        },
        "seconds": float(time.perf_counter() - started),
    }
    jt.official_test_blocker(payload)
    _write_json(path, payload)
    print(
        f"[structure] train triples={counts['train']['triples']} "
        f"valid triples={counts['valid']['triples']}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: references (provenance, accounting, hot start, parent replay)
# ---------------------------------------------------------------------------


def stage_references(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "references.json"
    if path.exists() and not force:
        payload = _read_json(path)
        if (
            payload.get("preregistration", {}).get("sha256") == _sha256_file(PREREG_PATH)
            and payload.get("module_sha256") == jt.module_sha256()
        ):
            return payload
    started = time.perf_counter()
    parent = jt.parent_provenance()
    m1_soup = jt.m1_soup_provenance()
    model = jt.build_joint_model(seed=0)
    accounting = jt.configure_trainable(model)
    if not accounting["passed"]:
        raise RuntimeError(f"parameter accounting failed: {accounting['passed']}")
    breakdown = jt.parameter_breakdown(model)
    valid_data = p1run.load_split("valid")
    valid_structure = jt.build_or_load_structure("valid")
    hot_start = jt.hot_start_check(model, valid_data, valid_structure)
    replay = bat.replay_parent_valid_mae()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "preregistration": _preregistration_payload(),
        "module_sha256": jt.module_sha256(),
        "git_commit": _git_commit(),
        "parent": parent,
        "m1_soup": m1_soup,
        "parameter_accounting": accounting,
        "parameter_breakdown": breakdown,
        "hot_start": hot_start,
        "parent_replay_background": replay,
        "thresholds": {
            "M_start_expected": float(jt.M_START_EXPECTED),
            "M_parent_expected": float(jt.M_PARENT_EXPECTED),
            "hot_start_tolerance": float(jt.HOT_START_TOLERANCE),
            "online_cache_tolerance": float(jt.ONLINE_CACHE_TOLERANCE),
            "promising_threshold": float(jt.PROMISING_THRESHOLD),
        },
        "seconds": float(time.perf_counter() - started),
    }
    jt.official_test_blocker(payload)
    _write_json(path, payload)
    if not hot_start["passed"]:
        raise RuntimeError(
            f"online hot start {hot_start['M_start_online']} != {jt.M_START_EXPECTED}; "
            "refusing to train"
        )
    print(
        f"[references] M_start(online)={hot_start['M_start_online']:.15f} "
        f"diff={hot_start['abs_diff']:.3e} M_parent(replay)={replay['replay_valid_mae']:.15f}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: preflight
# ---------------------------------------------------------------------------


def stage_preflight(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "preflight.json"
    if path.exists() and not force:
        return _read_json(path)
    references = _read_json(RESULTS_DIR / "references.json")
    _require_preregistration(references)
    encoded_audit = _read_json(p1run.ENCODED_AUDIT)
    train_size = int(encoded_audit["n_train"])
    valid_size = int(encoded_audit["n_valid"])
    if (train_size, valid_size) != (jt.TRAIN_SIZE, jt.VALID_SIZE):
        raise RuntimeError(f"split sizes {(train_size, valid_size)} != {(jt.TRAIN_SIZE, jt.VALID_SIZE)}")
    config_checks = {
        "epochs": (int(jt.EPOCHS), 40),
        "batch_size": (int(jt.BATCH_SIZE), 128),
        "learning_rate": (float(jt.LEARNING_RATE), 1.0e-4),
        "weight_decay": (float(jt.WEIGHT_DECAY), 1.0e-5),
        "grad_clip": (float(jt.GRAD_CLIP), 5.0),
        "soup_epoch_lo": (int(jt.SOUP_EPOCH_LO), 21),
        "soup_epoch_hi": (int(jt.SOUP_EPOCH_HI), 40),
        "soup_k": (int(jt.SOUP_K), 5),
        "train_shuffle_seed": (int(0 + jt.TRAIN_SHUFFLE_OFFSET), 91011),
        "eval_shuffle_seed": (int(0 + jt.EVAL_SHUFFLE_OFFSET), 91012),
        "threads": (int(jt.THREADS), 8),
        "promising_threshold": (float(jt.PROMISING_THRESHOLD), 0.120),
    }
    for name, (measured, expected) in config_checks.items():
        if measured != expected:
            raise RuntimeError(f"frozen config drift for {name}: {measured} != {expected}")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "preregistration": references["preregistration"],
        "module_sha256": jt.module_sha256(),
        "data": {
            "train_size": int(train_size),
            "valid_size": int(valid_size),
            "expected_train": int(jt.TRAIN_SIZE),
            "expected_valid": int(jt.VALID_SIZE),
            "official_test_loaded": False,
        },
        "training": {
            "seed": 0,
            "epochs": int(jt.EPOCHS),
            "batch_size": int(jt.BATCH_SIZE),
            "optimizer": "Adam",
            "learning_rate": float(jt.LEARNING_RATE),
            "weight_decay": float(jt.WEIGHT_DECAY),
            "grad_clip": float(jt.GRAD_CLIP),
            "train_shuffle_seed": int(0 + jt.TRAIN_SHUFFLE_OFFSET),
            "eval_shuffle_seed": int(0 + jt.EVAL_SHUFFLE_OFFSET),
            "soup_window": [int(jt.SOUP_EPOCH_LO), int(jt.SOUP_EPOCH_HI)],
            "soup_k": int(jt.SOUP_K),
            "objective": "graph_level_L1",
            "reconstruction_term": "excluded (constant w.r.t. trainable params)",
            "device": "cpu",
            "threads": int(jt.THREADS),
        },
        "parameter_accounting": references["parameter_accounting"],
        "structure_keys": {"train": jt._structure_key("train"), "valid": jt._structure_key("valid")},
    }
    jt.official_test_blocker(payload)
    _write_json(path, payload)
    print("[preflight] ok", flush=True)
    return payload


# ---------------------------------------------------------------------------
# stage: correctness
# ---------------------------------------------------------------------------


def stage_correctness(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "correctness.json"
    if path.exists() and not force:
        payload = _read_json(path)
        if payload.get("module_sha256") == jt.module_sha256():
            return payload
    references = _read_json(RESULTS_DIR / "references.json")
    _require_preregistration(references)
    started = time.perf_counter()
    train_structure = jt.build_or_load_structure("train")
    valid_structure = jt.build_or_load_structure("valid")
    train_data = p1run.load_split("train")
    valid_data = p1run.load_split("valid")
    suite = jt.run_correctness_suite(
        train_structure=train_structure,
        valid_structure=valid_structure,
        train_data=train_data,
        valid_data=valid_data,
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "preregistration": references["preregistration"],
        "module_sha256": jt.module_sha256(),
        "seconds": float(time.perf_counter() - started),
        **suite,
    }
    jt.official_test_blocker(payload)
    _write_json(path, payload)
    print(f"[correctness] all_passed={suite['all_passed']}", flush=True)
    if not suite["all_passed"]:
        raise RuntimeError("correctness suite failed; refusing to train")
    return payload


# ---------------------------------------------------------------------------
# stage: train
# ---------------------------------------------------------------------------


def stage_train(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    final_path = RESULTS_DIR / "run_seed0.json"
    if final_path.exists() and not force:
        print("[train] cache hit", flush=True)
        return _read_json(final_path)
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    if not correctness.get("all_passed"):
        raise RuntimeError("correctness gates not passed; refusing to train")
    references = _read_json(RESULTS_DIR / "references.json")
    _require_preregistration(references)
    if references.get("module_sha256") != jt.module_sha256():
        raise RuntimeError("module changed after references; refusing to train")
    jt.cpu_guard(torch.device("cpu"))
    train_structure = jt.build_or_load_structure("train")
    valid_structure = jt.build_or_load_structure("valid")
    train_data = p1run.load_split("train")
    valid_data = p1run.load_split("valid")
    started = time.perf_counter()
    result = jt.train_joint_tune(
        train_data,
        valid_data,
        train_structure,
        valid_structure,
        seed=0,
        epochs=int(jt.EPOCHS),
        batch_size=int(jt.BATCH_SIZE),
        threads=int(jt.THREADS),
        checkpoint_dir=CHECKPOINT_DIR,
        log=True,
    )
    result.update(
        {
            "git_commit": _git_commit(),
            "round_seconds": float(time.perf_counter() - started),
            "preregistration": references["preregistration"],
            "module_sha256": jt.module_sha256(),
            "structure_keys": {
                "train": jt._structure_key("train"),
                "valid": jt._structure_key("valid"),
            },
        }
    )
    jt.official_test_blocker(result)
    _write_json(final_path, result)
    _write_csv(
        RESULTS_DIR / "curve_seed0.csv",
        result["curve"],
        ("epoch", "train_mae", "valid_mae", "epoch_seconds"),
    )
    soup_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "members": result["soup"]["members"],
        "member_valid_mae": result["soup"]["member_valid_mae"],
        "soup_valid_mae": result["soup"]["soup_valid_mae"],
        "soup_epoch_window": result["soup"]["soup_epoch_window"],
        "soup_k": result["soup"]["soup_k"],
        "averaged_trainable_tensors": result["soup"]["averaged_trainable_tensors"],
        "averaged_trainable_params": result["soup"]["averaged_trainable_params"],
        "frozen_members_identical": result["soup"]["frozen_members_identical"],
        "buffers_members_identical": result["soup"]["buffers_members_identical"],
        "start_valid_mae": result["start_valid_mae"],
        "best_valid_mae": result["best_valid_mae"],
        "best_epoch": result["best_epoch"],
        "source_files_unchanged": result["source_files_unchanged"],
        "frozen_subset_unchanged": result["frozen_subset_unchanged"],
    }
    _write_json(RESULTS_DIR / "soup.json", soup_payload)
    print(
        f"[train] best={result['best_valid_mae']:.6f}@{result['best_epoch']} "
        f"soup={result['soup']['soup_valid_mae']:.6f} wall={result['wall_clock_s']:.1f}s",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# stage: analysis
# ---------------------------------------------------------------------------


def _verdict(soup_mae: float, complete: bool, correctness: Mapping[str, Any], hot_start_ok: bool) -> str:
    if not complete or not hot_start_ok:
        return VERDICT_INVALID
    if not correctness.get("all_passed"):
        return VERDICT_INVALID
    if soup_mae <= PROMISING_THRESHOLD:
        return VERDICT_PROMISING
    return VERDICT_NO_SIGNAL


def stage_analysis(run_id: str | None = None) -> dict[str, Any]:
    _ensure_dirs()
    references = _read_json(RESULTS_DIR / "references.json")
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    run = _read_json(RESULTS_DIR / "run_seed0.json")
    soup_mae = float(run["soup"]["soup_valid_mae"])
    m_start = float(references["hot_start"]["M_start_online"])
    m_parent = float(references["parent_replay_background"]["replay_valid_mae"])
    horizon = float(_read_json(RESULTS_DIR / "preflight.json")["training"]["epochs"])
    last_n = int(min(10, horizon))
    completed = bool(run.get("completed")) and int(run.get("epochs_run", 0)) == int(horizon)
    hot_start_ok = bool(references["hot_start"]["passed"]) and bool(correctness["all_passed"])
    verdict = _verdict(soup_mae, completed, correctness, hot_start_ok)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "run_id": run_id,
        "verdict": verdict,
        "M_joint_soup": soup_mae,
        "M_start": m_start,
        "M_parent": m_parent,
        "M_parent_recorded": float(jt.M_PARENT_EXPECTED),
        "Delta_vs_start": float(soup_mae - m_start),
        "Delta_vs_parent": float(soup_mae - m_parent),
        "promising_threshold": float(PROMISING_THRESHOLD),
        "promising": bool(soup_mae <= PROMISING_THRESHOLD),
        "best_valid_mae": float(run["best_valid_mae"]),
        "best_epoch": int(run["best_epoch"]),
        "soup_members": [int(v) for v in run["soup"]["members"]],
        "soup_member_valid_mae": [float(v) for v in run["soup"]["member_valid_mae"]],
        f"last_{last_n}_valid_mean": float(run["last_10_valid_mean"]),
        "final_train_mae": float(run["final_train_mae"]),
        "epochs_run": int(run["epochs_run"]),
        "completed": completed,
        "wall_clock_s": float(run["wall_clock_s"]),
        "seconds_per_epoch": float(run["seconds_per_epoch"]),
        "first_epoch_seconds": float(run["first_epoch_seconds"]),
        "estimated_remaining_s_after_epoch1": float(run["estimated_remaining_s_after_epoch1"]),
        "peak_rss_mb": float(run["peak_rss_mb"]),
        "parameter_accounting": references["parameter_accounting"],
        "parameter_breakdown": references["parameter_breakdown"],
        "hot_start": references["hot_start"],
        "source_hashes_before": run["source_hashes_before"],
        "source_hashes_after": run["source_hashes_after"],
        "source_files_unchanged": bool(run["source_files_unchanged"]),
        "frozen_subset_before": run["frozen_subset_before"],
        "frozen_subset_after": run["frozen_subset_after"],
        "frozen_subset_unchanged": bool(run["frozen_subset_unchanged"]),
        "correctness_all_passed": bool(correctness["all_passed"]),
        "correctness_checks": {
            key: bool(value.get("passed", False)) for key, value in correctness["checks"].items()
        },
        "parent": references["parent"],
        "m1_soup": references["m1_soup"],
        "structure_keys": run["structure_keys"],
        "scope_note": jt.SCOPE_NOTE,
        "device": "cpu",
    }
    jt.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "summary.json", payload)
    _write_report(payload, run)
    _write_decision(payload)
    return payload


def _fmt(value: Any, digits: int = 9) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _write_report(summary: Mapping[str, Any], run: Mapping[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# BondAnchoredTriple-JointTune-v1 — joint representation + composition fine-tune\n")
    lines.append(
        f"- run id: `{summary.get('run_id')}`\n"
        f"- verdict: **{summary['verdict']}**\n"
        f"- soup valid MAE `M_joint_soup = {_fmt(summary['M_joint_soup'])}`\n"
        f"- online hot start `M_start = {_fmt(summary['M_start'])}` "
        f"(expected {_fmt(summary['hot_start']['M_start_expected'])})\n"
        f"- `Delta_vs_start = {summary['Delta_vs_start']:+.9f}` (overall joint adaptation)\n"
        f"- parent replay `M_parent = {_fmt(summary['M_parent'])}` (background only)\n"
        f"- best valid `{_fmt(summary['best_valid_mae'])}` @ epoch {summary['best_epoch']}\n"
        f"- last-10-epochs valid mean `{_fmt(summary['last_10_valid_mean'])}`\n"
        f"- final train MAE `{_fmt(summary['final_train_mae'])}`\n"
        f"- soup members `{summary['soup_members']}` / member valid "
        f"`{[round(v, 6) for v in summary['soup_member_valid_mae']]}`\n"
        f"- {summary['epochs_run']} epochs completed={summary['completed']}, wall "
        f"{summary['wall_clock_s']:.1f}s ({summary['seconds_per_epoch']:.2f}s/epoch, first epoch "
        f"{summary['first_epoch_seconds']:.1f}s, estimated remaining after epoch 1 "
        f"{summary['estimated_remaining_s_after_epoch1']:.0f}s), peak RSS "
        f"{summary['peak_rss_mb']:.1f} MB\n"
        f"- device: `cpu`, 8 threads (local execution this round; no remote A100 work)\n"
    )
    lines.append("\n## Scope\n")
    lines.append(
        "- one candidate, seed 0, fixed 40 epochs, batch 128, new Adam lr 1e-4, coupled wd 1e-5, "
        "clip 5 over exactly the unfrozen whitelist, graph-level L1, Top-5 soup over epochs 21-40;\n"
        "- the shared dictionary coordinates (`D`, `U`, `common_rms`), the semantic interface, the C6 "
        "rule, `W_A_S`/`W_A_C`, `node_encoder`, `global_encoder`, `topology_encoder` and the six "
        "BAT-v1 normalisation buffers are frozen;\n"
        "- the frozen feature extractor stays in eval mode (original backend dropout off); the dynamic "
        "chain is recomputed online with autograd every step (no cached `E`/`p_ij`/`z_old` training "
        "input);\n"
        "- no M0 retraining, matched control, shuffle, ablation, branch-off, mechanism arm, extra seed, "
        "learning-rate search, early stop or horizon extension; official test never instantiated.\n"
    )
    lines.append("\n## Frozen sources\n")
    parent = summary["parent"]
    m1 = summary["m1_soup"]
    lines.append(
        f"- Sem108 backbone `{parent['path']}` sha256 `{parent['checkpoint_sha256']}`, canonical state "
        f"`{parent['canonical_state_sha256']}` ({parent['state_keys']} keys, {parent['params']} params, "
        f"old Reader {parent['reader_params']})\n"
        f"- BAT-v1 M1 soup `{m1['path']}` sha256 `{m1['checkpoint_sha256']}`, canonical state "
        f"`{m1['canonical_state_sha256']}` ({len(m1['state_keys'])} keys), members `{m1['soup_members']}` "
        f"valid `{[round(v, 6) for v in m1['soup_member_valid_mae']]}`; buffers match "
        f"`standardizers.json`: {all(m1['buffers_match_standardizers_json'].values())}\n"
        f"- source checkpoint files unchanged before/after: `{summary['source_files_unchanged']}`\n"
    )
    lines.append("\n## Parameter accounting (measured)\n")
    accounting = summary["parameter_accounting"]
    lines.append(
        f"- trainable `{accounting['trainable_params']}` params / `{accounting['trainable_tensors']}` "
        f"tensors; frozen `{accounting['frozen_params']}` params; total registered "
        f"`{accounting['total_registered_params']}`; removed old 302-D Reader "
        f"`{accounting['old_reader_removed_params']}` params (not registered)\n"
    )
    for group, value in summary["parameter_breakdown"]["groups"].items():
        lines.append(f"  - `{group}`: {value['numel']} params\n")
    lines.append(
        f"- frozen subset unchanged before/after: `{summary['frozen_subset_unchanged']}` "
        f"(params sha `{summary['frozen_subset_before']['frozen_params_sha256'][:16]}`, buffers sha "
        f"`{summary['frozen_subset_before']['buffers_sha256'][:16]}`, fixed coordinates sha "
        f"`{summary['frozen_subset_before']['fixed_coordinate_sha256'][:16]}`)\n"
    )
    lines.append("\n## Correctness\n")
    for key, value in summary["correctness_checks"].items():
        lines.append(f"- `{key}`: {value}\n")
    hot = summary["hot_start"]
    lines.append(
        f"- online hot start `{_fmt(hot['M_start_online'], 15)}` vs `{_fmt(hot['M_start_expected'], 15)}` "
        f"(abs diff {hot['abs_diff']:.3e}, tolerance {hot['tolerance']:.1e})\n"
    )
    lines.append("\n## Curve (every 5th epoch)\n")
    lines.append("\n| epoch | train MAE | valid MAE | s/epoch |\n|---:|---:|---:|---:|\n")
    for row in run["curve"]:
        if row["epoch"] % 5 == 0 or row["epoch"] == 1 or row["epoch"] == int(run["epochs_run"]):
            lines.append(
                f"| {row['epoch']} | {row['train_mae']:.6f} | {row['valid_mae']:.6f} | "
                f"{row['epoch_seconds']:.2f} |\n"
            )
    lines.append("\n## Interpretation boundary\n")
    lines.append(f"{summary['scope_note']}\n")
    (RESULTS_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def _write_decision(summary: Mapping[str, Any]) -> None:
    verdict = str(summary["verdict"])
    if verdict == VERDICT_PROMISING:
        body = (
            "The fixed 40-epoch joint adaptation cleared the frozen `M_joint_soup <= 0.120` gate. "
            "Per the pre-registration this is a report-and-stop outcome: a future matched-control or "
            "full-budget confirmation needs a NEW pre-registration.  No automatic extra seed, "
            "extension or official-test read is bought here."
        )
    elif verdict == VERDICT_NO_SIGNAL:
        body = (
            "The run is complete and valid but `M_joint_soup > 0.120`.  This closes the combination "
            "candidate: with the shared dictionary coordinates fixed and the representation + "
            "composition jointly adapted for 40 epochs, there is no strong validation signal.  No "
            "rescue (lr, unfreeze scope, initialisation, normalisation, horizon, seed 1) is "
            "authorised; a future step needs a new pre-registration."
        )
    else:
        body = (
            f"The round is `{verdict}`; no performance verdict is issued.  Repair the blocking "
            "condition and re-run only under this same pre-registration."
        )
    lines = [
        "# Decision — BondAnchoredTriple-JointTune-v1\n\n",
        f"- run id: `{summary.get('run_id')}`\n",
        f"- verdict: **{verdict}**\n",
        f"- `M_joint_soup = {summary['M_joint_soup']:.9f}`, `M_start = {summary['M_start']:.9f}`, "
        f"`Delta_vs_start = {summary['Delta_vs_start']:+.9f}`, "
        f"`M_parent = {summary['M_parent']:.9f}` (background)\n\n",
        body,
        "\n\n## Forbidden without a new pre-registration\n\n",
        "- seed 1/2/3 of this candidate, or a rerun with changed lr / unfreeze scope / initialisation / "
        "normalisation / soup window / horizon;\n",
        "- matched control, M0 retraining, shuffle, ablation, branch-off, mechanism arm or dead-node "
        "binding rescue purchased off this result;\n",
        "- official test access (never loaded in this round);\n",
        "- treating this single performance candidate as evidence about the triple operator's "
        "independent effect or the irreplaceability of the three-environment mechanism / dictionary "
        "coordinates.\n",
    ]
    (RESULTS_DIR / "DECISION.md").write_text("".join(lines), encoding="utf-8")


__all__ = [
    "PROTOCOL_VERSION",
    "VERDICT_PROMISING",
    "VERDICT_NO_SIGNAL",
    "VERDICT_INVALID",
    "PROMISING_THRESHOLD",
    "stage_structure",
    "stage_references",
    "stage_preflight",
    "stage_correctness",
    "stage_train",
    "stage_analysis",
]
