"""E2E-DictEnv-H1-Clarity-Audit runner — CPU-only information-flow / mechanism audit.

Round ``e2e_dictenv_h1_clarity_audit`` (Workstream Z).
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_h1_clarity_audit_preregistration.md``.
Core module: ``tracks/ksvd/experiments/luyin16/e2e_dictenv_h1_clarity_audit.py``.

Stages
------
``inventory budget baseline frozen adapt matched report all-one-pass``

* ``inventory``  — Phase A static information-flow inventory + provenance checks
* ``budget``     — CPU runtime / memory budget (valid inference, train epoch)
* ``baseline``   — B0: exact CPU replay of the H1 soup checkpoint on official-valid
* ``frozen``     — Phase B: all pre-registered frozen interventions
* ``adapt``      — Tier 1 warm-start adaptation from the H1 checkpoint
* ``matched``    — Tier 2 matched from-scratch CPU retraining (1 baseline + finalists)
* ``report``     — machine-readable summary + report/decision drafts

CPU only.  Official ZINC test is never loaded.  ``all-one-pass`` deliberately
stops after ``frozen``: Tier 1/2 are decision-gated and must be launched
explicitly after reviewing the frozen table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_v0 as v0run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = audit.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_h1_clarity_audit"
CHECKPOINT = TRACK_ROOT / "results/e2e_dictenv_p2_abs/states/H1_soup_state.pt"

_write_json = p2run._write_json
_read_json = p2run._read_json
_write_csv = p2run._write_csv
_git_commit = p2run._git_commit
_sha256 = v0run._sha256

#: injected by the audit module; kept as module attribute for testability.
device = torch.device("cpu")


# ---------------------------------------------------------------------------
# shared context
# ---------------------------------------------------------------------------


def _load_valid():
    valid = p1run.load_split("valid")
    loader = p1.make_env_loader(valid, int(p2run.BATCH_SIZE), False, int(p2run.EVAL_SHUFFLE_OFFSET))
    return valid, loader


def _load_h1_model(threads: int) -> tuple[audit.AuditModel, np.ndarray, str]:
    audit.attach_cpu(int(threads))
    dictionary, dict_sha = p2run.load_dictionary(audit.H1_CONFIG.dict_kind)
    model = audit.build_audit_model(dictionary, seed=0)
    state = audit.load_h1_soup_state(CHECKPOINT)
    model.load_state_dict(state)
    model.eval()
    return model, dictionary, dict_sha


def _prediction_digest(predictions: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(predictions.astype(np.float64)).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Phase A — inventory
# ---------------------------------------------------------------------------


def stage_inventory() -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    inventory = audit.information_inventory()
    provenance = {
        "global_context": audit.verify_global_context_provenance(),
        "pair_relation": audit.verify_relation_groups(),
        "topology_features": audit.verify_topology_groups(),
    }
    inventory["provenance_checks"] = provenance
    inventory["git_commit"] = _git_commit()
    inventory["generated_at_unix"] = time.time()
    _write_json(RESULTS_DIR / "information_inventory.json", inventory)
    print(
        f"[inventory] entries={inventory['summary']['n_entries']} "
        f"global_ctx_ok={provenance['global_context']['all_coordinates_affine_consistent']} "
        f"relation_ok={provenance['pair_relation']['all_passed']} "
        f"topology_ok={provenance['topology_features']['all_coordinates_affine_consistent']}",
        flush=True,
    )
    return inventory


# ---------------------------------------------------------------------------
# runtime budget
# ---------------------------------------------------------------------------


def stage_budget(threads: int, scaling_threads: Sequence[int] = (2, 4, 8)) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    nproc = int(os.cpu_count() or 1)
    mem = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    mem[key] = float(rest.strip().split()[0]) / 1024.0
    except Exception:  # pragma: no cover
        pass

    model, dictionary, dict_sha = _load_h1_model(int(threads))
    valid, loader = _load_valid()
    rss_before = audit._rss_mb()
    started = time.perf_counter()
    replay = audit.evaluate_mask(model, loader, device, None)
    valid_seconds = float(time.perf_counter() - started)
    rss_after = audit._rss_mb()

    train = p1run.load_split("train")
    lam = float(audit.H1_LAMBDA)
    epoch_records: dict[str, Any] = {}
    for thread_count in sorted({int(t) for t in scaling_threads} | {int(threads)}):
        audit.attach_cpu(thread_count)
        torch.manual_seed(0)
        model_e = audit.build_audit_model(dictionary, seed=0)
        model_e.load_state_dict(audit.load_h1_soup_state(CHECKPOINT))
        model_e.train()
        optimizer = torch.optim.Adam(model_e.parameters(), lr=1e-3, weight_decay=1e-5)
        run_loader = p1.make_env_loader(train, 128, True, int(p2run.TRAIN_SHUFFLE_OFFSET))
        epoch_started = time.perf_counter()
        n_mol = 0
        for batch in run_loader:
            prediction, aux = model_e(batch, mask=None, return_aux=True)
            rec = model_e.reconstruction_loss(aux["phi"], aux["coord"])
            loss = torch.nn.functional.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_e.parameters(), float(p2run.GRAD_CLIP))
            optimizer.step()
            n_mol += int(batch.y.numel())
        epoch_seconds = float(time.perf_counter() - epoch_started)
        epoch_records[str(thread_count)] = {
            "threads": int(thread_count),
            "molecules": int(n_mol),
            "seconds": epoch_seconds,
            "samples_per_second": float(n_mol / max(epoch_seconds, 1e-9)),
            "rss_mb": float(audit._rss_mb()),
        }
        if thread_count == int(threads):
            print(f"[budget] epoch@{thread_count}t {epoch_seconds:.2f}s", flush=True)

    reference = epoch_records[str(int(threads))]
    full = float(reference["seconds"])
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "torch_threads_chosen": int(threads),
        "nproc": nproc,
        "mem_total_mb": float(mem.get("MemTotal", float("nan"))),
        "mem_available_mb": float(mem.get("MemAvailable", float("nan"))),
        "checkpoint_sha256": _sha256(CHECKPOINT),
        "dictionary_sha256": dict_sha,
        "valid_inference": {
            "molecules": int(replay["n_molecules"]),
            "seconds": valid_seconds,
            "mae": float(replay["mae"]),
            "rss_before_mb": float(rss_before),
            "rss_after_mb": float(rss_after),
        },
        "train_epoch": epoch_records,
        "epoch_seconds_estimate_single_process": full,
        "epochs_320_seconds_estimate_single_process": float(320.0 * full + 320.0 * valid_seconds),
        "peak_rss_mb_process": audit._peak_rss_mb(),
        "concurrency_guidance": {
            "threads_per_process": int(threads),
            "max_concurrent_from_cores": int(max(1, nproc // max(int(threads), 1))),
            "max_concurrent_from_ram_mb": int(
                max(1, int(mem.get("MemAvailable", 0.0) or 0.0) // max(audit._rss_mb(), 1.0))
            ),
        },
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "runtime_budget.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Phase B — baseline replay + frozen interventions
# ---------------------------------------------------------------------------


def _baseline_payload(model, loader, dictionary_sha: str, replay: Mapping[str, Any]) -> dict[str, Any]:
    historical = float(audit.H1_HISTORICAL_SOUP_MAE)
    cpu_mae = float(replay["mae"])
    return {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "checkpoint_path": str(CHECKPOINT.relative_to(REPO_ROOT)),
        "checkpoint_sha256": _sha256(CHECKPOINT),
        "state_sha256": audit.state_sha256(audit.load_h1_soup_state(CHECKPOINT)),
        "dictionary_sha256": dictionary_sha,
        "config": audit.H1_CONFIG.as_dict(),
        "soup_members": list(audit.H1_SOUP_MEMBERS),
        "lambda_rec": float(audit.H1_LAMBDA),
        "official_valid_mae": cpu_mae,
        "historical_soup_valid_mae": historical,
        "abs_diff_vs_historical": abs(cpu_mae - historical),
        "relative_diff_vs_historical": abs(cpu_mae - historical) / historical,
        "prediction_mean": float(replay["prediction_mean"]),
        "prediction_std": float(replay["prediction_std"]),
        "prediction_sha256": _prediction_digest(replay["predictions"]),
        "total_parameters": int(p2run._n_params(model)),
        "official_test_loaded": False,
    }


def _frozen_context(threads: int):
    model, _dictionary, dict_sha = _load_h1_model(int(threads))
    valid, loader = _load_valid()
    started = time.perf_counter()
    replay = audit.evaluate_mask(model, loader, device, None)
    runtime = float(time.perf_counter() - started)
    return model, valid, loader, dict_sha, replay, runtime


def _run_table(
    model: audit.AuditModel,
    loader: Any,
    valid: Sequence[Any],
    device_obj: torch.device,
    registered: Sequence[audit.Intervention],
    baseline_mae: float,
    baseline_predictions: np.ndarray,
    fill_policy: Mapping[str, torch.Tensor] | None,
    controls: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identity_row: dict[str, Any] | None = None
    for intervention in registered:
        if intervention.name in controls:
            if identity_row is None:
                identity_row = audit.run_intervention(
                    model,
                    loader,
                    valid,
                    device_obj,
                    audit.Intervention("IDENTITY", "control"),
                    baseline_predictions,
                    fill_policy=None,
                )
                identity_row["intervention"] = "IDENTITY"
            row = dict(identity_row)
            row["intervention"] = intervention.name
            row["category"] = intervention.category
            row["mask"] = intervention.mask.as_dict()
            row["graph_shuffle"] = intervention.graph_shuffle
            row["use_fill"] = bool(intervention.use_fill)
            row["notes"] = intervention.notes + " (control: identity mask, shared run)"
            row["control_reused"] = True
            row["runtime_seconds"] = 0.0
        else:
            row = audit.run_intervention(
                model, loader, valid, device_obj, intervention, baseline_predictions, fill_policy
            )
            row["control_reused"] = False
        rows.append(row)
        print(
            f"[{row.get('use_fill') and 'fill' or ('shuf' if intervention.graph_shuffle else 'zero')}] "
            f"{row['intervention']:>4} mae={row['intervention_mae']:.6f} "
            f"d={row['intervention_mae'] - baseline_mae:+.6f} "
            f"|dp|={row.get('mean_abs_prediction_delta', float('nan')):.5f} "
            f"r={row.get('prediction_correlation', float('nan')):.4f}",
            flush=True,
        )
    return audit.summarise_interventions(rows, baseline_mae)


def _write_table(name: str, payload_prefix: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    payload = {**payload_prefix, "rows": list(rows)}
    _write_json(RESULTS_DIR / f"{name}.json", payload)
    _write_csv(
        RESULTS_DIR / f"{name}.csv",
        [
            {
                "intervention": row["intervention"],
                "category": row["category"],
                "baseline_mae": row["baseline_mae"],
                "intervention_mae": row["intervention_mae"],
                "delta_mae": row["delta_mae"],
                "mean_abs_prediction_delta": row.get("mean_abs_prediction_delta"),
                "prediction_correlation": row.get("prediction_correlation"),
                "runtime_seconds": row["runtime_seconds"],
                "load_bearing": row["load_bearing"],
                "notes": row["notes"],
            }
            for row in rows
        ],
    )


def stage_frozen(threads: int, include_extended: bool = True) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model, valid, loader, dict_sha, replay, runtime = _frozen_context(int(threads))
    baseline = _baseline_payload(model, loader, dict_sha, replay)
    baseline["runtime_seconds"] = runtime
    baseline["peak_rss_mb"] = audit._peak_rss_mb()
    # provenance: plain forward vs the audit path with an identity mask
    identity = audit.evaluate_mask(model, loader, device, audit.AuditMask())
    baseline["audit_identity_mask_bit_identical"] = bool(
        np.array_equal(identity["predictions"], replay["predictions"])
    )
    baseline["audit_identity_mask_max_abs_diff"] = float(
        np.max(np.abs(identity["predictions"] - replay["predictions"]))
    )
    # independent replay through the frozen P1 evaluation helper (same weights)
    p1_replay = p1run.evaluate(model, loader, device)
    baseline["p1_evaluate_mae"] = float(p1_replay["mae"])
    baseline["p1_evaluate_rec"] = float(p1_replay["rec"])
    baseline["p1_evaluate_max_abs_diff"] = float(
        np.max(np.abs(p1_replay["predictions"] - replay["predictions"]))
    )
    _write_json(RESULTS_DIR / "baseline_replay.json", baseline)
    print(
        f"[baseline] cpu_valid_mae={baseline['official_valid_mae']:.9f} "
        f"historical={baseline['historical_soup_valid_mae']:.9f} "
        f"bit_identical={baseline['audit_identity_mask_bit_identical']}",
        flush=True,
    )

    fill_policy = audit.build_fill_policy(model, loader, device)
    _write_json(
        RESULTS_DIR / "fill_policy.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "source": "official valid, identity-mask pass of the H1 soup checkpoint",
            "blocks": {key: list(value.shape) for key, value in sorted(fill_policy.items())},
            "official_test_loaded": False,
        },
    )

    baseline_predictions = replay["predictions"]
    baseline_mae = float(baseline["official_valid_mae"])
    shared = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "baseline": baseline,
        "official_test_loaded": False,
    }
    controls = ("A0", "G0", "T0", "N0", "R0", "P0")
    stage_start = time.perf_counter()

    def _timed(rows: list[dict[str, Any]], table: str, extra: Mapping[str, Any]) -> None:
        _write_table(
            table,
            {
                **shared,
                **extra,
                "rows_evaluated": len(rows),
                "wall_clock_s": round(time.perf_counter() - stage_start, 3),
            },
            rows,
        )

    zero_rows = _run_table(
        model,
        loader,
        valid,
        device,
        audit.interventions(include_extended=include_extended),
        baseline_mae,
        baseline_predictions,
        None,
        controls,
    )
    _timed(zero_rows, "frozen_interventions", {})
    fill_rows = _run_table(
        model,
        loader,
        valid,
        device,
        audit.interventions_fill(),
        baseline_mae,
        baseline_predictions,
        fill_policy,
        (),
    )
    _timed(fill_rows, "frozen_interventions_fill", {"probe": "mean_fill"})
    shuffle_rows = _run_table(
        model,
        loader,
        valid,
        device,
        audit.interventions_graph_shuffle(),
        baseline_mae,
        baseline_predictions,
        None,
        (),
    )
    _timed(
        shuffle_rows,
        "frozen_interventions_graph_shuffle",
        {"probe": "cross_molecule_row_shuffle"},
    )
    readout_rows = _run_table(
        model,
        loader,
        valid,
        device,
        audit.interventions_readout_shuffle(),
        baseline_mae,
        baseline_predictions,
        None,
        (),
    )
    _timed(
        readout_rows,
        "frozen_interventions_readout_shuffle",
        {"probe": "cross_graph_readout_row_shuffle"},
    )
    relation_rows = _run_table(
        model,
        loader,
        valid,
        device,
        audit.interventions_relation_shuffle(),
        baseline_mae,
        baseline_predictions,
        None,
        (),
    )
    _timed(
        relation_rows,
        "frozen_interventions_relation_shuffle",
        {"probe": "cross_pair_relation_row_shuffle"},
    )
    return {
        "zero": zero_rows,
        "fill": fill_rows,
        "graph_shuffle": shuffle_rows,
        "readout_shuffle": readout_rows,
        "relation_shuffle": relation_rows,
        "baseline": baseline,
        "frozen_stage_wall_clock_s": round(time.perf_counter() - stage_start, 3),
    }


# ---------------------------------------------------------------------------
# Phase C — Tier 1 adaptation
# ---------------------------------------------------------------------------


def stage_adapt(
    threads: int,
    names: Sequence[str],
    epochs: int,
    extend_epochs: int = 40,
    seed: int = 0,
) -> dict[str, Any]:
    out_dir = RESULTS_DIR / "adaptation"
    out_dir.mkdir(parents=True, exist_ok=True)
    train = p1run.load_split("train")
    valid, _loader = _load_valid()
    init_state = audit.load_h1_soup_state(CHECKPOINT)
    summary: dict[str, Any] = {}
    for name in names:
        if name not in audit.PHASE_C_CANDIDATES:
            raise KeyError(f"unknown adaptation candidate {name!r}")
        mask, notes = audit.PHASE_C_CANDIDATES[name]
        path = out_dir / f"{name}.json"
        if path.exists():
            payload = _read_json(path)
        else:
            payload = audit.train_cpu(
                tag=name,
                mask=mask,
                epochs=int(epochs),
                threads=int(threads),
                out_dir=out_dir,
                init_state=init_state,
                train_data=train,
                valid_data=valid,
                seed=int(seed),
            )
            payload["git_commit"] = _git_commit()
            payload["notes"] = notes
            _write_json(path, payload)
        summary[name] = {
            "notes": notes,
            "epochs": payload["epochs_run"],
            "best_valid_mae": payload["best_valid_mae"],
            "final_valid_mae": payload["final_valid_mae"],
            "soup_valid_mae": payload["soup"]["soup_valid_mae"],
            "wall_clock_s": payload["wall_clock_s"],
            "mask": payload["mask"],
            "json": str(path.relative_to(REPO_ROOT)),
        }
        print(f"[adapt] {name} soup={summary[name]['soup_valid_mae']:.6f}", flush=True)
    aggregate = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "tier": "Tier 1 warm-start adaptation",
        "epoch_budget": int(epochs),
        "seed": int(seed),
        "candidates": summary,
        "official_test_loaded": False,
    }
    if "M0" in summary:
        control = summary["M0"]["soup_valid_mae"]
        for name, row in summary.items():
            row["delta_vs_continuation"] = float(row["soup_valid_mae"] - control)
    _write_json(out_dir / "adaptation_summary.json", aggregate)
    return aggregate


# ---------------------------------------------------------------------------
# Phase C — Tier 2 matched retraining
# ---------------------------------------------------------------------------


def stage_matched_one(threads: int, name: str, epochs: int, seed: int = 0) -> dict[str, Any]:
    out_dir = RESULTS_DIR / "matched_cpu"
    out_dir.mkdir(parents=True, exist_ok=True)
    train = p1run.load_split("train")
    valid, _loader = _load_valid()
    mask = None if name == "BASE" else audit.candidate_mask(name)
    notes = (
        "matched CPU baseline (H1 architecture, from scratch)"
        if mask is None
        else audit.PHASE_C_CANDIDATES[name][1]
    )
    payload = audit.train_cpu(
        tag=f"{name}_e{epochs}",
        mask=mask,
        epochs=int(epochs),
        threads=int(threads),
        out_dir=out_dir,
        init_state=None,
        train_data=train,
        valid_data=valid,
        seed=int(seed),
    )
    payload["git_commit"] = _git_commit()
    payload["candidate"] = name
    payload["notes"] = notes
    _write_json(out_dir / f"{name}_e{epochs}.json", payload)
    print(
        f"[matched:{name}] soup={payload['soup']['soup_valid_mae']:.6f} "
        f"best={payload['best_valid_mae']:.6f}@{payload['best_epoch']} "
        f"wall={payload['wall_clock_s']:.0f}s",
        flush=True,
    )
    return payload


def stage_matched(
    threads: int,
    names: Sequence[str],
    epochs: int,
    concurrency: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    running: list[tuple[str, subprocess.Popen]] = []
    results: dict[str, Any] = {}
    queue = list(names)
    threads = max(1, int(threads))
    concurrency = max(1, min(int(concurrency), len(queue)))
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["CUDA_VISIBLE_DEVICES"] = ""
    while queue or running:
        while queue and len(running) < concurrency:
            name = queue.pop(0)
            command = [
                sys.executable,
                "-m",
                "tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_h1_clarity_audit",
                "matched-one",
                "--candidate",
                str(name),
                "--threads",
                str(threads),
                "--epochs",
                str(int(epochs)),
                "--seed",
                str(int(seed)),
            ]
            print(f"[matched] launch {name} (threads={threads})", flush=True)
            running.append(
                (
                    name,
                    subprocess.Popen(command, cwd=str(REPO_ROOT), env=env),
                )
            )
        time.sleep(2.0)
        for item in list(running):
            name, process = item
            if process.poll() is not None:
                running.remove(item)
                if process.returncode != 0:
                    raise RuntimeError(f"matched run {name} failed with code {process.returncode}")
                results[name] = _read_json(RESULTS_DIR / "matched_cpu" / f"{name}_e{epochs}.json")
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "tier": "Tier 2 matched CPU retraining",
        "epoch_budget": int(epochs),
        "threads_per_process": int(threads),
        "concurrency": int(concurrency),
        "candidates": {
            name: {
                "notes": payload["notes"],
                "mask": payload["mask"],
                "best_valid_mae": payload["best_valid_mae"],
                "best_epoch": payload["best_epoch"],
                "final_valid_mae": payload["final_valid_mae"],
                "soup_valid_mae": payload["soup"]["soup_valid_mae"],
                "wall_clock_s": payload["wall_clock_s"],
                "seconds_per_epoch": payload["seconds_per_epoch"],
                "json": f"matched_cpu/{name}_e{epochs}.json",
            }
            for name, payload in results.items()
        },
        "official_test_loaded": False,
    }
    if "BASE" in results:
        baseline = results["BASE"]["soup"]["soup_valid_mae"]
        for name, row in summary["candidates"].items():
            row["delta_vs_cpu_baseline"] = float(row["soup_valid_mae"] - baseline)
    _write_json(RESULTS_DIR / "matched_cpu" / "matched_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


#: intervention id -> (channel label, probe id preferred for the verdict).
#: The preferred probe is distribution-preserving whenever one exists; the zero
#: probe is kept only as the cross-check because it is an OOD input corruption.
CHANNEL_VERDICTS = {
    "A1": ("anchor (full 62)", "A1"),
    "A2": ("anchor patch marginals (atom+bond mass)", "A2"),
    "A3": ("anchor root atom identity", "A3"),
    "A4": ("anchor patch atom mass", "A4"),
    "A5": ("anchor patch bond mass", "A5"),
    "A6": ("anchor size", "A6"),
    "G1": ("graph-level chemistry marginal (atom+bond histogram)", "GS1"),
    "G2": ("graph-level topology summary (short+long)", "GS2"),
    "G3": ("graph-level global62 (full)", "GS3"),
    "EG2": ("graph-level bond histogram", "EG2"),
    "T1": ("topology25 cycle-spectrum bypass", "GS4"),
    "N1": ("node structure-semantic binding (slot input)", "N1"),
    "N2": ("edge structure-semantic binding (role input)", "N2"),
    "N3": ("node alpha<->atom correspondence", "N3"),
    "N4": ("edge role<->bond-type correspondence", "N4"),
    "N6": ("dictionary coordinate alpha", "N6"),
    "R1": ("relation: distance+overlap+boundary (no path count)", "R1"),
    "R2": ("relation: overlap block", "RS2"),
    "R3": ("relation: boundary block", "RS3"),
    "R4": ("relation: log path count", "RS4"),
    "R5": ("relation: full 15-D", "RS5"),
    "P1": ("unary second moment", "PS2"),
    "P2": ("pair second moment", "PS5"),
    "P3": ("unary + pair second moments", "PS9"),
    "P4": ("unary + pair counts", "PS3+PS6"),
    "EP1": ("unary first moment", "PS1"),
    "EP2": ("unary pool (all blocks)", "PS7"),
    "EP3": ("pair readout (all blocks)", "PS8"),
    "EB1": ("global encoder output", "EB1"),
    "EB2": ("distance gate", "EB2"),
    "EB3": ("pair composition (projected E)", "EB3"),
}


def _verdict_class(delta: float) -> str:
    if delta >= 0.10:
        return "strongly_load_bearing"
    if delta >= 0.02:
        return "moderately_used"
    if delta >= 0.005:
        return "weakly_used"
    return "weak_or_dormant"


def stage_report() -> dict[str, Any]:
    zero_path = RESULTS_DIR / "frozen_interventions.json"
    if not zero_path.exists():
        raise RuntimeError("frozen_interventions.json missing; run the frozen stage first")
    tables = {"zero": _read_json(zero_path)}
    for name in ("fill", "graph_shuffle", "readout_shuffle", "relation_shuffle"):
        path = RESULTS_DIR / f"frozen_interventions_{name}.json"
        if path.exists():
            tables[name] = _read_json(path)

    def _index(table: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        return {row["intervention"]: row for row in table.get("rows", [])}

    indices = {name: _index(table) for name, table in tables.items()}

    def _delta(table: str, name: str) -> float | None:
        row = indices.get(table, {}).get(name)
        return None if row is None else float(row["delta_mae"])

    channel_rows: list[dict[str, Any]] = []
    for identifier, (label, probe) in CHANNEL_VERDICTS.items():
        if "+" in probe:  # combined count probe: sum of the two count rows
            parts = probe.split("+")
            delta = sum(
                value for value in (_delta("readout_shuffle", part) for part in parts) if value is not None
            )
            probe_used = "readout_shuffle:count(sum)"
        elif probe in indices["graph_shuffle"]:
            delta = _delta("graph_shuffle", probe)
            probe_used = "graph_shuffle"
        elif probe in indices["relation_shuffle"]:
            delta = _delta("relation_shuffle", probe)
            probe_used = "relation_shuffle"
        elif probe in indices["readout_shuffle"]:
            delta = _delta("readout_shuffle", probe)
            probe_used = "readout_shuffle"
        else:
            delta = _delta("fill", probe)
            probe_used = "fill" if delta is not None else "zero"
            if delta is None:
                delta = _delta("zero", probe)
        zero_delta = _delta("zero", identifier)
        fill_delta = _delta("fill", identifier)
        channel_rows.append(
            {
                "id": identifier,
                "channel": label,
                "verdict_probe": probe_used,
                "delta_primary": None if delta is None else float(delta),
                "delta_zero": zero_delta,
                "delta_fill": fill_delta,
                "verdict": None if delta is None else _verdict_class(float(delta)),
                "zero_probe_inflated": bool(
                    zero_delta is not None
                    and delta is not None
                    and float(zero_delta) > 2.0 * max(float(delta), 1e-9)
                ),
            }
        )
    channel_rows.sort(key=lambda row: (row["delta_primary"] is None, -(row["delta_primary"] or 0.0)))

    classes: dict[str, list[str]] = {
        "strongly_load_bearing": [],
        "moderately_used": [],
        "weakly_used": [],
        "weak_or_dormant": [],
    }
    for row in channel_rows:
        if row["verdict"]:
            classes[row["verdict"]].append(row["id"])

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "baseline": tables["zero"]["baseline"],
        "channel_verdicts": channel_rows,
        "classes": classes,
        "tables": {
            name: [
                {
                    "intervention": row["intervention"],
                    "category": row["category"],
                    "delta_mae": row["delta_mae"],
                    "mean_abs_prediction_delta": row.get("mean_abs_prediction_delta"),
                    "prediction_correlation": row.get("prediction_correlation"),
                    "load_bearing_zero_probe": row["load_bearing"],
                }
                for row in sorted(table["rows"], key=lambda item: float(item["delta_mae"]), reverse=True)
            ]
            for name, table in tables.items()
        },
        "dormant_blocks": [row["id"] for row in channel_rows if row["verdict"] == "weak_or_dormant"],
        "adaptation_available": (RESULTS_DIR / "adaptation" / "adaptation_summary.json").exists(),
        "matched_cpu_available": (RESULTS_DIR / "matched_cpu" / "matched_summary.json").exists(),
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "summary.json", payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all-one-pass",
        choices=[
            "inventory",
            "budget",
            "baseline",
            "frozen",
            "adapt",
            "matched",
            "matched-one",
            "report",
            "all-one-pass",
        ],
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate", default="BASE")
    parser.add_argument("--no-extended", action="store_true")
    parser.add_argument("--candidates", default=None, help="comma-separated candidate list")
    args = parser.parse_args(argv)

    if args.stage == "inventory":
        print(json.dumps(stage_inventory(), indent=2))
    elif args.stage == "budget":
        print(json.dumps(stage_budget(args.threads), indent=2))
    elif args.stage == "baseline":
        model, valid, loader, dict_sha, replay, runtime = _frozen_context(args.threads)
        payload = _baseline_payload(model, loader, dict_sha, replay)
        payload["runtime_seconds"] = runtime
        payload["peak_rss_mb"] = audit._peak_rss_mb()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _write_json(RESULTS_DIR / "baseline_replay.json", payload)
        print(json.dumps(payload, indent=2))
    elif args.stage == "frozen":
        result = stage_frozen(args.threads, not args.no_extended)
        print(
            json.dumps(
                {name: len(rows) for name, rows in result.items() if isinstance(rows, list)},
                indent=2,
            )
        )
    elif args.stage == "adapt":
        names = (
            [name.strip() for name in str(args.candidates).split(",") if name.strip()]
            if args.candidates
            else ["M0", "C1", "C2", "C3"]
        )
        print(json.dumps(stage_adapt(args.threads, names, args.epochs, seed=args.seed), indent=2))
    elif args.stage == "matched":
        names = (
            [name.strip() for name in str(args.candidates).split(",") if name.strip()]
            if args.candidates
            else ["BASE", "C1", "C3"]
        )
        print(
            json.dumps(stage_matched(args.threads, names, args.epochs, args.concurrency, args.seed), indent=2)
        )
    elif args.stage == "matched-one":
        print(json.dumps(stage_matched_one(args.threads, args.candidate, args.epochs, args.seed), indent=2))
    elif args.stage == "report":
        print(json.dumps(stage_report(), indent=2))
    else:
        stage_inventory()
        stage_budget(args.threads)
        stage_frozen(args.threads, not args.no_extended)
        stage_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
