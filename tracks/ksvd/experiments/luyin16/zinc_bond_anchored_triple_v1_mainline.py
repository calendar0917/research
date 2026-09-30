"""Runner stages for BondAnchoredTriple-v1 (``zinc_bond_anchored_triple_v1_mainline``).

Round ``e2e_dictenv_bond_anchored_triple_v1`` (study ``zinc-context-gap``).
Pre-registration:
``tracks/ksvd/notes/e2e_dictenv_bond_anchored_triple_v1_preregistration.md``.

Stages: ``references`` -> ``preflight`` -> ``cache`` -> ``correctness`` ->
``train`` -> ``analysis``.  CPU-first; official valid is evaluation only and
the official test split is never instantiated.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_bond_anchored_triple_v1 as bat
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = "e2e_dictenv_bond_anchored_triple_v1"
RESULTS_DIR = bat.RESULTS_DIR
NOTES_DIR = bat.NOTES_DIR
PREREG_PATH = bat.PREREG_PATH
CHECKPOINT_DIR = bat.CHECKPOINT_DIR

_write_json = bat._write_json
_read_json = bat._read_json
_sha256_file = bat._sha256_file
_git_commit = p2run._git_commit

VERDICT_PROMISING = "FROZEN_TRIPLE_SCREEN_PROMISING"
VERDICT_NO_SIGNAL = "FROZEN_TRIPLE_SCREEN_NO_STRONG_SIGNAL"
PROMISING_THRESHOLD = 0.120

LEGACY_NOTE = (
    "This is a performance screen, not a matched control.  Delta is the "
    "difference against the historical, unmatched parent soup; it is not the "
    "treatment effect of the triple operator (the Reader is re-initialised, "
    "the inputs are standardised and the 80-epoch budget differs)."
)


def _ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], header: Sequence[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in header})


# ---------------------------------------------------------------------------
# stage: references
# ---------------------------------------------------------------------------


def stage_references(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "references.json"
    if path.exists() and not force:
        return _read_json(path)
    started = time.perf_counter()
    parent = bat.build_parent()
    provenance = bat.parent_provenance(parent)
    replay = bat.replay_parent_valid_mae(parent)
    accounting = bat.parameter_accounting()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "preregistration": {
            "path": str(PREREG_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(PREREG_PATH),
            "sha256_prefix": _sha256_file(PREREG_PATH)[:12],
        },
        "module_sha256": bat.module_sha256(),
        "parent": provenance,
        "replay": replay,
        "parameter_accounting": accounting,
        "thresholds": {
            "M_parent": float(bat.PARENT_HISTORICAL_SOUP_MAE),
            "promising_threshold": float(PROMISING_THRESHOLD),
            "replay_tolerance": 1.0e-7,
        },
        "seconds": float(time.perf_counter() - started),
    }
    bat.official_test_blocker(payload)
    _write_json(path, payload)
    print(
        f"[references] parent replay={replay['replay_valid_mae']:.15f} "
        f"diff={replay['abs_diff']:.3e} state={provenance['canonical_state_sha256'][:8]}",
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
    if references["preregistration"]["sha256"] != _sha256_file(PREREG_PATH):
        raise RuntimeError("pre-registration changed after references; refusing")
    encoded_audit = _read_json(p1run.ENCODED_AUDIT)
    train_size = int(encoded_audit["n_train"])
    valid_size = int(encoded_audit["n_valid"])
    cache_files = {
        split: {
            "path": str(bat.cache_path(split).relative_to(REPO_ROOT)),
            "exists": bat.cache_path(split).exists(),
            "key": bat._cache_key(split),
        }
        for split in ("train", "valid")
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "preregistration": {
            "path": str(PREREG_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(PREREG_PATH),
        },
        "module_sha256": bat.module_sha256(),
        "data": {
            "train_size": int(train_size),
            "valid_size": int(valid_size),
            "expected_train": int(bat.TRAIN_SIZE),
            "expected_valid": int(bat.VALID_SIZE),
            "official_test_loaded": False,
        },
        "epochs": int(bat.EPOCHS),
        "batch_size": int(bat.BATCH_SIZE),
        "seed": 0,
        "optimizer": "Adam",
        "learning_rate": float(bat.LEARNING_RATE),
        "weight_decay": float(bat.WEIGHT_DECAY),
        "grad_clip": float(bat.GRAD_CLIP),
        "soup_window": [int(bat.SOUP_EPOCH_LO), int(bat.SOUP_EPOCH_HI)],
        "soup_k": int(bat.SOUP_K),
        "parameter_accounting": _read_json(RESULTS_DIR / "references.json")["parameter_accounting"],
        "cache_keys": {split: bat._cache_key(split) for split in ("train", "valid")},
        "cache_files": cache_files,
    }
    if train_size != 10_000 or valid_size != 1_000:
        raise RuntimeError("unexpected split sizes (expected 10000 / 1000)")
    bat.official_test_blocker(payload)
    _write_json(path, payload)
    print("[preflight] ok", flush=True)
    return payload


# ---------------------------------------------------------------------------
# stage: cache
# ---------------------------------------------------------------------------


def stage_cache(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "cache_report.json"
    if path.exists() and not force:
        report = _read_json(path)
        if (
            report.get("cache_keys", {}).get("train") == bat._cache_key("train")
            and report.get("cache_keys", {}).get("valid") == bat._cache_key("valid")
            and bat.cache_path("train").exists()
            and bat.cache_path("valid").exists()
        ):
            return report
    started = time.perf_counter()
    train_cache = bat.build_or_load_cache("train", force=force)
    valid_cache = bat.build_or_load_cache("valid", force=force)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "cache_keys": {
            "train": bat._cache_key("train"),
            "valid": bat._cache_key("valid"),
        },
        "train": train_cache.provenance,
        "valid": valid_cache.provenance,
        "counts": {
            "train_graphs": train_cache.n_graphs,
            "train_pairs": train_cache.n_pairs,
            "train_triples": train_cache.n_triples,
            "valid_graphs": valid_cache.n_graphs,
            "valid_pairs": valid_cache.n_pairs,
            "valid_triples": valid_cache.n_triples,
        },
        "seconds": float(time.perf_counter() - started),
    }
    bat.official_test_blocker(payload)
    _write_json(path, payload)
    print(
        f"[cache] train triples={train_cache.n_triples} valid triples={valid_cache.n_triples}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: standardizers (train-only, fixed)
# ---------------------------------------------------------------------------


def stage_standardizers(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "standardizers.json"
    if path.exists() and not force:
        payload = _read_json(path)
        if (
            payload.get("module_sha256") == bat.module_sha256()
            and payload.get("cache_keys", {}).get("train") == bat._cache_key("train")
            and payload.get("preregistration", {}).get("sha256") == _sha256_file(PREREG_PATH)
        ):
            return payload
    references = _read_json(RESULTS_DIR / "references.json")
    if references["preregistration"]["sha256"] != _sha256_file(PREREG_PATH):
        raise RuntimeError("pre-registration changed after references; refusing")
    started = time.perf_counter()
    train_cache = bat.build_or_load_cache("train")
    stats = bat.fit_standardizers(train_cache, seed=0)
    accounting = bat.parameter_accounting()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "preregistration": {"sha256": _sha256_file(PREREG_PATH)},
        "module_sha256": bat.module_sha256(),
        "cache_keys": {"train": bat._cache_key("train")},
        "statistics": stats.as_payload(),
        "parameter_accounting": accounting,
        "seconds": float(time.perf_counter() - started),
    }
    bat.official_test_blocker(payload)
    _write_json(path, payload)
    print(
        f"[standardizers] near-constant p={stats.near_constant['pair']['near_constant_count']} "
        f"old={stats.near_constant['z_old']['near_constant_count']} "
        f"z3={stats.near_constant['z3_init']['near_constant_count']}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# stage: correctness
# ---------------------------------------------------------------------------


def stage_correctness(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    path = RESULTS_DIR / "correctness.json"
    if path.exists() and not force:
        return _read_json(path)
    preflight = _read_json(RESULTS_DIR / "preflight.json")
    if preflight["preregistration"]["sha256"] != _sha256_file(PREREG_PATH):
        raise RuntimeError("pre-registration changed after preflight; refusing")
    started = time.perf_counter()
    parent = bat.build_parent()
    train_cache = bat.build_or_load_cache("train")
    valid_cache = bat.build_or_load_cache("valid")
    stats = _read_json(RESULTS_DIR / "standardizers.json")["statistics"]
    suite = bat.run_correctness_suite(
        train_cache=train_cache,
        valid_cache=valid_cache,
        stats=stats,
        parent=parent,
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "module_sha256": bat.module_sha256(),
        "seconds": float(time.perf_counter() - started),
        **suite,
    }
    bat.official_test_blocker(payload)
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
    preflight = _read_json(RESULTS_DIR / "preflight.json")
    if preflight["preregistration"]["sha256"] != _sha256_file(PREREG_PATH):
        raise RuntimeError("pre-registration changed after preflight; refusing to train")
    standardizers = _read_json(RESULTS_DIR / "standardizers.json")
    if standardizers.get("module_sha256") != bat.module_sha256():
        raise RuntimeError("module changed after standardizer fit; refusing to train")
    bat.cpu_guard(torch.device("cpu"))
    train_cache = bat.build_or_load_cache("train")
    valid_cache = bat.build_or_load_cache("valid")
    parent_before = audit.state_sha256(bat.load_parent_state())
    checkpoint_sha_before = _sha256_file(bat.PARENT_SOUP_PATH)
    started = time.perf_counter()
    result = bat.train_mainline(
        train_cache,
        valid_cache,
        standardizers["statistics"],
        seed=0,
        epochs=int(bat.EPOCHS),
        batch_size=int(bat.BATCH_SIZE),
        threads=int(bat.THREADS),
        checkpoint_dir=CHECKPOINT_DIR,
        log=True,
    )
    result.update(
        {
            "git_commit": _git_commit(),
            "round_seconds": float(time.perf_counter() - started),
            "parent_state_sha256_before": parent_before,
            "parent_state_sha256_after": audit.state_sha256(bat.load_parent_state()),
            "parent_checkpoint_sha256_before": checkpoint_sha_before,
            "parent_checkpoint_sha256_after": _sha256_file(bat.PARENT_SOUP_PATH),
            "cache_keys": {
                "train": bat._cache_key("train"),
                "valid": bat._cache_key("valid"),
            },
        }
    )
    result["parent_state_unchanged"] = bool(
        result["parent_state_sha256_before"] == result["parent_state_sha256_after"]
        and result["parent_checkpoint_sha256_before"] == result["parent_checkpoint_sha256_after"]
    )
    bat.official_test_blocker(result)
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
        "best_valid_mae": result["best_valid_mae"],
        "best_epoch": result["best_epoch"],
        "parent_state_unchanged": result["parent_state_unchanged"],
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


def _verdict(soup_mae: float, complete: bool, correctness: Mapping[str, Any]) -> str:
    if not complete:
        return "INCOMPLETE_INVALID"
    if not correctness.get("all_passed"):
        return "INVALID_CORRECTNESS_FAILED"
    if soup_mae <= PROMISING_THRESHOLD:
        return VERDICT_PROMISING
    return VERDICT_NO_SIGNAL


def stage_analysis(run_id: str | None = None) -> dict[str, Any]:
    _ensure_dirs()
    references = _read_json(RESULTS_DIR / "references.json")
    correctness = _read_json(RESULTS_DIR / "correctness.json")
    run = _read_json(RESULTS_DIR / "run_seed0.json")
    soup_mae = float(run["soup"]["soup_valid_mae"])
    parent_mae = float(references["replay"]["replay_valid_mae"])
    delta = float(soup_mae - parent_mae)
    verdict = _verdict(soup_mae, bool(run.get("completed")), correctness)
    curve = run["curve"]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "run_id": run_id,
        "verdict": verdict,
        "M_soup": soup_mae,
        "M_parent_replay": parent_mae,
        "M_parent_recorded": float(bat.PARENT_HISTORICAL_SOUP_MAE),
        "delta_soup_minus_parent": delta,
        "promising_threshold": float(PROMISING_THRESHOLD),
        "promising": bool(soup_mae <= PROMISING_THRESHOLD),
        "best_valid_mae": float(run["best_valid_mae"]),
        "best_epoch": int(run["best_epoch"]),
        "soup_members": [int(v) for v in run["soup"]["members"]],
        "soup_member_valid_mae": [float(v) for v in run["soup"]["member_valid_mae"]],
        "last_20_valid_mean": float(run["last_20_valid_mean"]),
        "final_train_mae": float(run["final_train_mae"]),
        "epochs_run": int(run["epochs_run"]),
        "completed": bool(run.get("completed")),
        "wall_clock_s": float(run["wall_clock_s"]),
        "seconds_per_epoch": float(run["seconds_per_epoch"]),
        "peak_rss_mb": float(run["peak_rss_mb"]),
        "parameter_accounting": references["parameter_accounting"],
        "parent_state_unchanged": bool(run.get("parent_state_unchanged")),
        "correctness_all_passed": bool(correctness.get("all_passed")),
        "correctness_checks": {
            key: bool(value.get("passed", value.get("all_requires_grad_false", False)))
            for key, value in correctness["checks"].items()
        },
        "parent": references["parent"],
        "cache": {
            "train": run["cache_keys"]["train"],
            "valid": run["cache_keys"]["valid"],
        },
        "scope_note": LEGACY_NOTE,
        "device": "cpu",
    }
    bat.official_test_blocker(payload)
    _write_json(RESULTS_DIR / "summary.json", payload)
    _write_report(payload, run)
    _write_decision(payload)
    return payload


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _write_report(summary: Mapping[str, Any], run: Mapping[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# BondAnchoredTriple-v1 — mainline performance screen\n")
    lines.append(
        f"- run id: `{summary.get('run_id')}`\n"
        f"- verdict: **{summary['verdict']}**\n"
        f"- soup valid MAE `M_soup = {_fmt(summary['M_soup'], 9)}`\n"
        f"- parent replay valid MAE `M_parent = {_fmt(summary['M_parent_replay'], 9)}` "
        f"(historical {_fmt(summary['M_parent_recorded'], 9)})\n"
        f"- `Delta = M_soup - M_parent = {summary['delta_soup_minus_parent']:+.9f}` "
        f"(unmatched screen difference)\n"
        f"- best valid `{_fmt(summary['best_valid_mae'], 9)}` @ epoch {summary['best_epoch']}\n"
        f"- last-20-epochs valid mean `{_fmt(summary['last_20_valid_mean'], 9)}`\n"
        f"- final train MAE `{_fmt(summary['final_train_mae'], 9)}`\n"
        f"- soup members `{summary['soup_members']}` / member valid "
        f"`{[round(v, 6) for v in summary['soup_member_valid_mae']]}`\n"
        f"- {summary['epochs_run']} epochs completed={summary['completed']}, "
        f"wall {summary['wall_clock_s']:.1f}s ({summary['seconds_per_epoch']:.2f}s/epoch), "
        f"peak RSS {summary['peak_rss_mb']:.1f} MB\n"
        f"- device: `{summary['device']}` (remote A100 unavailable this round: "
        f"NVML/driver `GPU0000:4B:00.0: Unknown Error`)\n"
    )
    lines.append("\n## Scope\n")
    lines.append(
        "- one candidate, seed 0, fixed 80 epochs, batch 128, Adam lr 1e-3, "
        "wd 1e-5, clip 5, Top-5 soup over epochs 41-80;\n"
        "- the parent `CSSD-Sem108 + C6` soup is frozen and never retrained; "
        "`p_ij` and `z_old` are cached under `no_grad` and the training model "
        "contains no parent parameter;\n"
        "- no third-environment shuffle, code/atom/semantic shuffle, mechanism "
        "or ablation arm; no second seed; no hyper-parameter search; no 320-epoch "
        "extension; official test never instantiated.\n"
    )
    lines.append("\n## Frozen parent\n")
    parent = summary["parent"]
    lines.append(
        f"- checkpoint `{parent['checkpoint_path']}` sha256 "
        f"`{parent['checkpoint_sha256']}`\n"
        f"- canonical state sha256 `{parent['canonical_state_sha256']}` "
        f"({parent['state_keys']} keys)\n"
        f"- params {parent['total_params']} (reader {parent['reader_params']}), "
        f"pair width {parent['pair_token_dim']}, env width {parent['environment_dim']}, "
        f"reader input {parent['reader_input_dim']}\n"
        f"- mask `{parent['mask']['signature']}`\n"
        f"- parent state unchanged before/after training: "
        f"`{summary['parent_state_unchanged']}`\n"
    )
    lines.append("\n## Candidate\n")
    accounting = summary["parameter_accounting"]
    lines.append(
        f"- F 48->64->32: {accounting['F_params']} params; new Reader "
        f"{accounting['reader_params']} params ({accounting['reader_increment']:+d} "
        f"vs the replaced parent Reader); trainable {accounting['trainable_params']}; "
        f"full model {accounting['full_model_params']}\n"
        f"- triple objects: one per real bond anchor `(i, j)` (`i < j`) times every "
        f"third node `k`, `m*(n-2)` per graph, zero summary for degenerate graphs\n"
        f"- `z3 = [mean(t); mean(t*t)]` over that graph's triples; fixed train-only "
        f"standardizers for `p`, `z_old` and the initial `z3`\n"
    )
    lines.append("\n## Correctness\n")
    for key, value in summary["correctness_checks"].items():
        lines.append(f"- `{key}`: {value}\n")
    lines.append("\n## Curve (every 10th epoch)\n")
    lines.append("\n| epoch | train MAE | valid MAE | s/epoch |\n|---:|---:|---:|---:|\n")
    for row in run["curve"]:
        if row["epoch"] % 10 == 0 or row["epoch"] == 1 or row["epoch"] == int(run["epochs_run"]):
            lines.append(
                f"| {row['epoch']} | {row['train_mae']:.6f} | {row['valid_mae']:.6f} | "
                f"{row['epoch_seconds']:.2f} |\n"
            )
    lines.append("\n" + str(summary["scope_note"]) + "\n")
    (RESULTS_DIR / "REPORT.md").write_text("".join(lines), encoding="utf-8")


def _write_decision(summary: Mapping[str, Any]) -> None:
    verdict = str(summary["verdict"])
    if verdict == VERDICT_PROMISING:
        body = (
            "The 80-epoch frozen-representation screen cleared the frozen "
            "`M_soup <= 0.120` gate.  Per the pre-registration this is a "
            "report-and-stop outcome: the result justifies a future matched "
            "control / full-budget follow-up under a NEW pre-registration.  "
            "No extra training, seed or architecture work is bought here."
        )
    elif verdict == VERDICT_NO_SIGNAL:
        body = (
            "The run is complete and valid but `M_soup > 0.120`.  This closes "
            "the round: there is no strong 80-epoch signal for the frozen "
            "three-environment static composition.  No rescue (width, "
            "initialisation, shuffle, seed 1, extended horizon) is authorised; "
            "a future step needs a new pre-registration."
        )
    else:
        body = (
            f"The round is `{verdict}`; no performance verdict is issued.  "
            "Repair the blocking condition and re-run only under this same "
            "pre-registration."
        )
    lines = [
        "# Decision — BondAnchoredTriple-v1\n\n",
        f"- run id: `{summary.get('run_id')}`\n",
        f"- verdict: **{verdict}**\n",
        f"- `M_soup = {summary['M_soup']:.9f}`, `M_parent = {summary['M_parent_replay']:.9f}`, "
        f"`Delta = {summary['delta_soup_minus_parent']:+.9f}`\n\n",
        body,
        "\n\n## Forbidden without a new pre-registration\n\n",
        "- seed 1/2/3 of this candidate, or a rerun with changed width/init/normalisation;\n",
        "- any third-environment/code/atom/semantic shuffle, mechanism or ablation arm;\n",
        "- a 320-epoch or otherwise extended horizon, or a changed 0.120 gate;\n",
        "- official test access (never loaded in this round).\n",
    ]
    (RESULTS_DIR / "DECISION.md").write_text("".join(lines), encoding="utf-8")


__all__ = [
    "PROTOCOL_VERSION",
    "VERDICT_PROMISING",
    "VERDICT_NO_SIGNAL",
    "PROMISING_THRESHOLD",
    "stage_references",
    "stage_preflight",
    "stage_cache",
    "stage_standardizers",
    "stage_correctness",
    "stage_train",
    "stage_analysis",
]
