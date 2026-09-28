"""E2E-DictEnv-Clean-Mechanism-v1 runner — CPU-only multi-seed mechanism round.

Round ``e2e_dictenv_clean_mechanism_v1`` (Workstream Z).
Core module: ``tracks/ksvd/experiments/luyin16/e2e_dictenv_clean_mechanism_v1.py``.
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_clean_mechanism_v1_preregistration.md``.

Stages
------
``preflight  train-one  launch-a  gate-a  stage-b  stage-c-frozen  stage-c-adapt
 stage-c-train  gate-c  stage-c-extend  stage-d-groups  stage-d-adapt
 stage-d-decide  stage-d-train  gate-d  stage-d-extend  final-clean
 stage-f-train  gate-f  stage-f-extend  report  chain``

``chain`` executes the preregistered order unattended: it launches only the runs
permitted by the frozen gates and stops expanding a branch as soon as the gate
says stop.  Every stage is idempotent and resumable (existing artifacts are
skipped).

CPU only.  Official ZINC test is never loaded.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = cm.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_clean_mechanism_v1"
OLD_MATCHED = TRACK_ROOT / "results/e2e_dictenv_h1_clarity_audit/matched_cpu"
STAGE_DIRS = {
    "a": RESULTS_DIR / "stage_a_cpu_baseline",
    "c": RESULTS_DIR / "stage_c_independence",
    "d": RESULTS_DIR / "stage_d_relation",
    "f": RESULTS_DIR / "stage_f_dictionary_specificity",
}
CHECKPOINT = TRACK_ROOT / "results/e2e_dictenv_p2_abs/states/H1_soup_state.pt"

_write_json = p2run._write_json
_read_json = p2run._read_json
_git_commit = p2run._git_commit
_seed_everything = p2run._seed_everything

#: reusable seed-0 artifacts of the closed clarity-audit round (verified bit-exact).
REUSED: dict[tuple[str, int], Path] = {
    ("BASE", 0): OLD_MATCHED / "BASE_e320.json",
    ("C6", 0): OLD_MATCHED / "C6_e320.json",
    ("C1", 0): OLD_MATCHED / "C1_e320.json",
}

EPOCHS = 320
THREADS = 4
CONCURRENCY = 3
ADAPT_EPOCHS = 20
ADAPT_EXTEND_EPOCHS = 40


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in STAGE_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def artifact_json(stage: str, tag: str, epochs: int) -> Path:
    return STAGE_DIRS[str(stage)] / f"{tag}_e{epochs}.json"


def find_artifact(tag: str, seed: int, epochs: int = EPOCHS) -> dict[str, Any] | None:
    """Locate a run artifact (reused clarity-audit seed 0 or a new run)."""
    if (tag, int(seed)) in REUSED and int(epochs) == EPOCHS:
        path = REUSED[(tag, int(seed))]
        if path.exists():
            payload = _read_json(path)
            return {
                "arm": str(tag),
                "seed": int(seed),
                "epochs": int(epochs),
                "json": path,
                "payload": payload,
                "soup_state": path.with_name(f"{path.stem}_soup_state.pt"),
                "source": "reused_clarity_audit",
                "spec": cm.CLEAN_MECH_ARMS.get(str(tag)).as_dict() if str(tag) in cm.CLEAN_MECH_ARMS else None,
            }
    for stage in ("a", "c", "d", "f"):
        path = artifact_json(stage, tag, epochs)
        if path.exists():
            payload = _read_json(path)
            return {
                "arm": str(tag),
                "seed": int(seed),
                "epochs": int(epochs),
                "json": path,
                "payload": payload,
                "soup_state": path.parent / f"{tag}_soup_state.pt",
                "source": str(stage),
                "spec": payload.get("arm_spec"),
            }
    return None


def soup_mae(entry: Mapping[str, Any]) -> float:
    return float(entry["payload"]["soup"]["soup_valid_mae"])


def _spec_for(entry: Mapping[str, Any]) -> cm.CleanMechSpec:
    spec = entry.get("spec")
    if spec is None:
        return cm.CLEAN_MECH_ARMS[str(entry["arm"])]
    return cm.CleanMechSpec(**spec)


def load_checkpoint_model(entry: Mapping[str, Any], threads: int = THREADS):
    device = audit.attach_cpu(int(threads))
    dictionary, dict_sha = p2run.load_dictionary("sdb32")
    spec = _spec_for(entry)
    model = cm.build_clean_mech_model(dictionary, seed=0, spec=spec)
    state = audit.load_h1_soup_state(entry["soup_state"])
    model.load_state_dict(state)
    model.eval()
    return model, spec, cm.arm_mask(spec), dictionary, dict_sha, device


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def stage_preflight(threads: int = THREADS, epochs_sample: int = 1) -> dict[str, Any]:
    _ensure_dirs()
    device = audit.attach_cpu(int(threads))
    if device.type != "cpu":
        raise RuntimeError("preflight is CPU-only")
    dictionary, dict_sha = p2run.load_dictionary("sdb32")
    valid = p1run.load_split("valid")
    loader = p1.make_env_loader(valid, int(p2run.BATCH_SIZE), False, int(p2run.EVAL_SHUFFLE_OFFSET))
    started = time.perf_counter()
    replay = audit.evaluate_mask(cm.build_clean_mech_model(dictionary, 0), loader, device, None)
    valid_seconds = float(time.perf_counter() - started)

    train = p1run.load_split("train")
    model = audit.build_audit_model(dictionary, seed=0)
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(p2run.LEARNING_RATE), weight_decay=float(p2run.WEIGHT_DECAY)
    )
    run_loader = p1.make_env_loader(train, int(p2run.BATCH_SIZE), True, int(p2run.TRAIN_SHUFFLE_OFFSET))
    epoch_started = time.perf_counter()
    n_mol = 0
    for batch in run_loader:
        prediction, aux = model(batch, mask=None, return_aux=True)
        rec = model.reconstruction_loss(aux["phi"], aux["coord"])
        loss = torch.nn.functional.l1_loss(prediction.view(-1), batch.y.view(-1)) + float(cm.H1_LAMBDA) * rec
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(p2run.GRAD_CLIP))
        optimizer.step()
        n_mol += int(batch.y.numel())
    epoch_seconds = float(time.perf_counter() - epoch_started)
    mem = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    mem[key] = float(rest.strip().split()[0]) / 1024.0
    except Exception:  # pragma: no cover
        pass
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "cpu_count": int(os.cpu_count() or 1),
        "mem_total_mb": float(mem.get("MemTotal", float("nan"))),
        "mem_available_mb": float(mem.get("MemAvailable", float("nan"))),
        "threads_per_process": int(threads),
        "concurrency": int(CONCURRENCY),
        "dictionary_sha256": dict_sha,
        "checkpoint_sha256": p2run._sha256(CHECKPOINT) if CHECKPOINT.exists() else None,
        "valid_inference": {
            "molecules": int(replay["n_molecules"]),
            "seconds": valid_seconds,
            "mae": float(replay["mae"]),
        },
        "train_epoch": {
            "molecules": int(n_mol),
            "seconds": epoch_seconds,
            "samples_per_second": float(n_mol / max(epoch_seconds, 1e-9)),
        },
        "expected_seconds_per_epoch_at_concurrency": epoch_seconds * 1.55,
        "estimated_epochs_320_seconds_single_process": float(320.0 * epoch_seconds),
        "peak_rss_mb_process": float(audit._peak_rss_mb()),
        "reused_seed0": {
            "BASE": str(REUSED[("BASE", 0)].relative_to(REPO_ROOT)),
            "C6": str(REUSED[("C6", 0)].relative_to(REPO_ROOT)),
            "C1": str(REUSED[("C1", 0)].relative_to(REPO_ROOT)),
        },
    }
    _write_json(RESULTS_DIR / "runtime_budget.json", payload)
    print(
        f"[preflight] epoch={epoch_seconds:.2f}s valid={valid_seconds:.2f}s "
        f"threads={threads} concurrency={CONCURRENCY}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_one(job: Mapping[str, Any], threads: int = THREADS) -> dict[str, Any]:
    _ensure_dirs()
    spec = cm.CleanMechSpec(**dict(job["spec"]))
    stage = str(job["stage"])
    epochs = int(job["epochs"])
    out_dir = STAGE_DIRS[stage]
    json_path = artifact_json(stage, str(job["tag"]), epochs)
    if json_path.exists() and not bool(job.get("force")):
        payload = _read_json(json_path)
        print(f"[train] skip existing {job['tag']} e{epochs}", flush=True)
        return payload
    train = p1run.load_split("train")
    valid = p1run.load_split("valid")
    init_state = None
    if job.get("warm_start"):
        init_state = audit.load_h1_soup_state(str(job["warm_start"]))
    started = time.perf_counter()
    payload = cm.train_spec(
        spec,
        epochs=epochs,
        threads=int(threads),
        out_dir=out_dir,
        train_data=train,
        valid_data=valid,
        seed=int(job["seed"]),
        init_state=init_state,
        tag=str(job["tag"]),
    )
    payload.update(
        {
            "git_commit": _git_commit(),
            "device": "cpu",
            "official_test_loaded": False,
            "job": dict(job),
            "round_seconds": float(time.perf_counter() - started),
        }
    )
    _write_json(json_path, payload)
    print(
        f"[train] {job['tag']} seed={job['seed']} e{epochs} "
        f"soup={payload['soup']['soup_valid_mae']:.6f} "
        f"best={payload['best_valid_mae']:.6f}@{payload['best_epoch']} "
        f"wall={payload['wall_clock_s']:.0f}s",
        flush=True,
    )
    return payload


def launch(jobs: Sequence[Mapping[str, Any]], threads: int = THREADS, concurrency: int = CONCURRENCY) -> None:
    """Run training jobs as subprocesses with bounded concurrency (resumable)."""
    queue = [dict(job) for job in jobs]
    if not queue:
        return
    running: list[tuple[Mapping[str, Any], subprocess.Popen]] = []
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(int(threads))
    env["MKL_NUM_THREADS"] = str(int(threads))
    env["CUDA_VISIBLE_DEVICES"] = ""
    concurrency = max(1, min(int(concurrency), len(queue)))
    while queue or running:
        while queue and len(running) < concurrency:
            job = queue.pop(0)
            epochs = int(job["epochs"])
            tag = str(job["tag"])
            stage = str(job["stage"])
            if artifact_json(stage, tag, epochs).exists() and not bool(job.get("force")):
                print(f"[launch] skip {tag}", flush=True)
                continue
            command = [
                sys.executable,
                "-m",
                "tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_clean_mechanism_v1",
                "train-one",
                "--job",
                json.dumps(job),
                "--threads",
                str(int(threads)),
            ]
            print(f"[launch] {tag} (threads={threads})", flush=True)
            running.append((job, subprocess.Popen(command, cwd=str(REPO_ROOT), env=env)))
        time.sleep(2.0)
        for item in list(running):
            job, process = item
            if process.poll() is not None:
                running.remove(item)
                if process.returncode != 0:
                    raise RuntimeError(f"training job {job.get('tag')} failed with code {process.returncode}")


# ---------------------------------------------------------------------------
# job builders
# ---------------------------------------------------------------------------


def _train_job(
    stage: str,
    spec: cm.CleanMechSpec,
    seed: int,
    epochs: int,
    tag: str,
    *,
    warm_start: str | Path | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "kind": "train",
        "stage": str(stage),
        "spec": spec.as_dict(),
        "seed": int(seed),
        "epochs": int(epochs),
        "tag": str(tag),
        "warm_start": None if warm_start is None else str(warm_start),
        "notes": str(notes),
    }


def stage_a_jobs(seeds: Sequence[int] = (1, 2)) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for seed in seeds:
        for tag in ("BASE", "C6"):
            jobs.append(_train_job("a", cm.CLEAN_MECH_ARMS[tag], int(seed), EPOCHS, f"{tag}_seed{seed}"))
    return jobs


def stage_a_fallback_jobs(seeds: Sequence[int] = (1, 2)) -> list[dict[str, Any]]:
    return [
        _train_job("a", cm.CLEAN_MECH_ARMS["C1"], int(seed), EPOCHS, f"C1_seed{seed}")
        for seed in seeds
    ]


def clean_base_arm(clean_base: str, role: str) -> str:
    suffix = "NODE-INDEP" if role == "node" else "EDGE-INDEP"
    return f"{clean_base}-{suffix}"


def stage_c_train_jobs(clean_base: str, seeds: Sequence[int] = (0,)) -> list[dict[str, Any]]:
    jobs = []
    for role in ("node", "edge"):
        tag = clean_base_arm(clean_base, role)
        for seed in seeds:
            jobs.append(_train_job("c", cm.CLEAN_MECH_ARMS[tag], int(seed), EPOCHS, f"{tag}_seed{seed}"))
    return jobs


def stage_c_adapt_jobs(clean_base: str, epochs: int = ADAPT_EPOCHS) -> list[dict[str, Any]]:
    base = find_artifact(clean_base, 0)
    if base is None:
        raise RuntimeError(f"clean base seed0 artifact missing for {clean_base}")
    warm = base["soup_state"]
    jobs = [
        _train_job(
            "c",
            cm.CLEAN_MECH_ARMS[clean_base],
            0,
            int(epochs),
            f"{clean_base}_M0_adapt{epochs}",
            warm_start=warm,
            notes="matched continuation control on the clean-base mask",
        )
    ]
    for role in ("node", "edge"):
        tag = clean_base_arm(clean_base, role)
        jobs.append(
            _train_job(
                "c",
                cm.CLEAN_MECH_ARMS[tag],
                0,
                int(epochs),
                f"{tag}_adapt{epochs}",
                warm_start=warm,
                notes=f"{role} independence null warm-start adaptation",
            )
        )
    return jobs


def stage_d_adapt_jobs(clean_base: str, epochs: int = ADAPT_EPOCHS) -> list[dict[str, Any]]:
    base = find_artifact(clean_base, 0)
    if base is None:
        raise RuntimeError(f"clean base seed0 artifact missing for {clean_base}")
    warm = base["soup_state"]
    jobs = []
    for kind in ("REL-DIST", "REL-DIST-BOUNDARY"):
        jobs.append(
            _train_job(
                "d",
                cm.CLEAN_MECH_ARMS[kind],
                0,
                int(epochs),
                f"{kind}_adapt{epochs}",
                warm_start=warm,
                notes=f"{kind} warm-start adaptation screen",
            )
        )
    return jobs


def stage_d_train_jobs(kind: str, seeds: Sequence[int] = (0,)) -> list[dict[str, Any]]:
    return [
        _train_job("d", cm.CLEAN_MECH_ARMS[kind], int(seed), EPOCHS, f"{kind}_seed{seed}")
        for seed in seeds
    ]


def stage_f_jobs(final_mask_kind: str, node_binding: str, edge_binding: str, seeds: Sequence[int] = (0,)) -> list[dict[str, Any]]:
    spec = cm.CleanMechSpec(
        "FINAL-CLEAN-DENSE-TIED",
        final_mask_kind,
        node_binding=node_binding,
        edge_binding=edge_binding,
        coding="dense_tied",
    )
    return [
        _train_job("f", spec, int(seed), EPOCHS, f"FINAL-CLEAN-DENSE-TIED_seed{seed}")
        for seed in seeds
    ]


# ---------------------------------------------------------------------------
# Stage A gate
# ---------------------------------------------------------------------------


def collect_seed_deltas(arm: str, baseline: str = "BASE", max_seeds: int = 4) -> dict[int, float]:
    deltas: dict[int, float] = {}
    for seed in range(int(max_seeds)):
        candidate = find_artifact(arm, seed)
        reference = find_artifact(baseline, seed)
        if candidate is None or reference is None:
            continue
        deltas[int(seed)] = soup_mae(candidate) - soup_mae(reference)
    return deltas


def stage_gate_a(threads: int = THREADS, concurrency: int = CONCURRENCY, launch_missing: bool = True) -> dict[str, Any]:
    _ensure_dirs()
    gate_path = STAGE_DIRS["a"] / "gate_a.json"
    deltas = collect_seed_deltas("C6")
    if len(deltas) < 3 and launch_missing:
        missing = [seed for seed in (1, 2) if seed not in deltas]
        if missing:
            launch(stage_a_jobs(missing), threads, concurrency)
            deltas = collect_seed_deltas("C6")
    if len(deltas) < 3:
        gate: dict[str, Any] = {
            "verdict": "INSUFFICIENT_SEEDS",
            "adopted": False,
            "per_seed": {int(seed): float(value) for seed, value in deltas.items()},
            "note": "the frozen gate needs seeds 0, 1, 2",
        }
    else:
        gate = cm.c6_gate(deltas)
    if gate.get("verdict") == "UNSTABLE" and 3 not in deltas and launch_missing:
        launch(stage_a_jobs((3,)), threads, concurrency)
        deltas = collect_seed_deltas("C6")
        gate = cm.c6_gate(deltas)
    clean_base = "C6" if gate.get("adopted") else None
    if gate.get("verdict") == "INSUFFICIENT_SEEDS":
        clean_base = None
    fallback: dict[str, Any] = {}
    if clean_base is None:
        if launch_missing:
            missing = [s for s in (1, 2) if find_artifact("C1", s) is None]
            if missing:
                launch(stage_a_fallback_jobs(missing), threads, concurrency)
        c1_deltas = collect_seed_deltas("C1")
        if c1_deltas:
            fallback = cm.c6_gate(c1_deltas)
            fallback["arm"] = "C1"
            if fallback.get("adopted"):
                clean_base = "C1"
        if clean_base is None:
            clean_base = "BASE"
    entries = {}
    for arm in ("BASE", "C1", "C6"):
        entries[arm] = {}
        for seed in range(4):
            entry = find_artifact(arm, seed)
            if entry is not None:
                entries[arm][int(seed)] = {
                    "soup_valid_mae": soup_mae(entry),
                    "source": entry["source"],
                    "json": str(entry["json"].relative_to(REPO_ROOT)),
                }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "c6_gate": gate,
        "c1_fallback_gate": fallback or None,
        "per_arm": entries,
        "clean_base": None if clean_base is None else str(clean_base),
        "reason": (
            "insufficient seeds for the frozen gate"
            if clean_base is None
            else ("C6 adopted by the frozen gate" if clean_base == "C6" else f"C6 not adopted; fallback selected {clean_base}")
        ),
    }
    _write_json(gate_path, payload)
    if clean_base is not None:
        _write_json(
        RESULTS_DIR / "clean_base.json",
        {
            "clean_base": str(clean_base),
            "c6_gate": gate,
            "c1_fallback_gate": fallback or None,
            "git_commit": _git_commit(),
            "official_test_loaded": False,
        },
    )
    print(f"[gate-a] verdict={gate.get('verdict')} clean_base={clean_base}", flush=True)
    return payload


def read_clean_base() -> str:
    path = RESULTS_DIR / "clean_base.json"
    if not path.exists():
        raise RuntimeError("clean_base.json missing; run gate-a first")
    return str(_read_json(path)["clean_base"])


# ---------------------------------------------------------------------------
# Stage B — mechanism revalidation probes
# ---------------------------------------------------------------------------


def _probe_rows(
    model: cm.CleanMechModel,
    loader: Any,
    valid: Sequence[Any],
    device: torch.device,
    base_mask: audit.AuditMask | None,
    fill_policy: Mapping[str, torch.Tensor],
    interventions: Sequence[audit.Intervention],
    *,
    use_fill: bool,
    baseline_predictions: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for intervention in interventions:
        merged = dataclasses.replace(intervention, mask=cm.merge_masks(base_mask, intervention.mask))
        row = audit.run_intervention(
            model,
            loader,
            valid,
            device,
            merged,
            baseline_predictions,
            fill_policy if use_fill else None,
        )
        row["probe"] = intervention.name
        rows.append(row)
    return rows


def stage_b(threads: int = THREADS, arms: Sequence[str] | None = None) -> dict[str, Any]:
    _ensure_dirs()
    out_dir = RESULTS_DIR / "stage_b_mechanism"
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_base = read_clean_base()
    if arms is None:
        arms = ("BASE", clean_base)
    summary: dict[str, Any] = {}
    for arm in dict.fromkeys(arms):
        for seed in range(4):
            entry = find_artifact(arm, seed)
            if entry is None:
                continue
            tag = f"{arm}_seed{seed}"
            model, spec, arm_mask_value, _dictionary, _sha, device = load_checkpoint_model(entry, threads)
            valid = p1run.load_split("valid")
            loader = p1.make_env_loader(valid, int(p2run.BATCH_SIZE), False, int(p2run.EVAL_SHUFFLE_OFFSET))
            fill_policy = audit.build_fill_policy(model, loader, device, mask=None)
            per_view: dict[str, Any] = {}
            for view in ("own_mask", "identity"):
                base_mask = arm_mask_value if view == "own_mask" else None
                start = time.perf_counter()
                replay = audit.evaluate_mask(model, loader, device, base_mask)
                tables = {
                    "zero": _probe_rows(
                        model, loader, valid, device, base_mask, fill_policy,
                        audit.interventions(), use_fill=False,
                        baseline_predictions=replay["predictions"],
                    ),
                    "fill": _probe_rows(
                        model, loader, valid, device, base_mask, fill_policy,
                        audit.interventions_fill(), use_fill=True,
                        baseline_predictions=replay["predictions"],
                    ),
                    "graph_shuffle": _probe_rows(
                        model, loader, valid, device, base_mask, fill_policy,
                        audit.interventions_graph_shuffle(), use_fill=False,
                        baseline_predictions=replay["predictions"],
                    ),
                    "readout_shuffle": _probe_rows(
                        model, loader, valid, device, base_mask, fill_policy,
                        audit.interventions_readout_shuffle(), use_fill=False,
                        baseline_predictions=replay["predictions"],
                    ),
                    "relation_shuffle": _probe_rows(
                        model, loader, valid, device, base_mask, fill_policy,
                        audit.interventions_relation_shuffle(), use_fill=False,
                        baseline_predictions=replay["predictions"],
                    ),
                }
                payload = {
                    "protocol_version": PROTOCOL_VERSION,
                    "git_commit": _git_commit(),
                    "device": "cpu",
                    "official_test_loaded": False,
                    "arm": str(arm),
                    "seed": int(seed),
                    "spec": spec.as_dict(),
                    "view": str(view),
                    "baseline_valid_mae": float(replay["mae"]),
                    "tables": tables,
                    "wall_clock_s": float(time.perf_counter() - start),
                }
                _write_json(out_dir / f"{tag}_{view}.json", payload)
                per_view[view] = {
                    "baseline_valid_mae": float(replay["mae"]),
                    "deltas": {
                        name: {row["probe"]: float(row["delta_mae"]) for row in rows}
                        for name, rows in tables.items()
                    },
                }
                print(f"[stage-b] {tag} {view} baseline={replay['mae']:.6f}", flush=True)
            diagnostics = cm.dictionary_diagnostics(
                model,
                loader,
                device,
                dictionary_init=p2run.load_dictionary("sdb32")[0],
            )
            summary[tag] = {"spec": spec.as_dict(), "views": per_view, "dictionary": diagnostics}
    _write_json(out_dir / "mechanism_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Stage C — analytic independence null
# ---------------------------------------------------------------------------


def stage_c_frozen(threads: int = THREADS, arms: Sequence[str] | None = None) -> dict[str, Any]:
    _ensure_dirs()
    if arms is None:
        arms = ("BASE", read_clean_base())
    else:
        arms = tuple(str(arm) for arm in arms)
    merged: dict[str, Any] = {}
    for arm in dict.fromkeys(arms):
        for seed in range(4):
            entry = find_artifact(arm, seed)
            if entry is None:
                continue
            tag = f"{arm}_seed{seed}"
            model, spec, arm_mask_value, dictionary, _sha, device = load_checkpoint_model(entry, threads)
            valid = p1run.load_split("valid")
            loader = p1.make_env_loader(valid, int(p2run.BATCH_SIZE), False, int(p2run.EVAL_SHUFFLE_OFFSET))
            baseline = audit.evaluate_mask(model, loader, device, arm_mask_value)
            variants = {}
            for node_binding, edge_binding, name in (
                ("indep", "paired", "NODE_INDEP"),
                ("paired", "indep", "EDGE_INDEP"),
                ("indep", "indep", "BOTH_INDEP"),
            ):
                variants[name] = cm.evaluate_binding_variant(
                    dictionary,
                    model.state_dict(),
                    loader,
                    device,
                    arm_mask_value,
                    node_binding=node_binding,
                    edge_binding=edge_binding,
                    baseline_predictions=baseline["predictions"],
                )
            norms = cm.independence_norm_stats(model, loader, device)
            payload = {
                "protocol_version": PROTOCOL_VERSION,
                "git_commit": _git_commit(),
                "device": "cpu",
                "official_test_loaded": False,
                "arm": str(arm),
                "seed": int(seed),
                "spec": spec.as_dict(),
                "baseline_valid_mae": float(baseline["mae"]),
                "variants": variants,
                "norms": norms,
            }
            _write_json(STAGE_DIRS["c"] / f"frozen_{tag}.json", payload)
            merged[tag] = payload
            print(
                f"[stage-c-frozen] {tag} node_indep_delta="
                f"{variants['NODE_INDEP']['valid_mae'] - baseline['mae']:+.6f} "
                f"edge_indep_delta={variants['EDGE_INDEP']['valid_mae'] - baseline['mae']:+.6f}",
                flush=True,
            )
    _write_json(RESULTS_DIR / "node_edge_independence_diagnostics.json", merged)
    return merged


def stage_c_adapt(threads: int = THREADS, concurrency: int = CONCURRENCY, epochs: int = ADAPT_EPOCHS) -> dict[str, Any]:
    clean_base = read_clean_base()
    jobs = stage_c_adapt_jobs(clean_base, epochs)
    launch(jobs, threads, concurrency)
    control = find_artifact(f"{clean_base}_M0_adapt{epochs}", 0, epochs)
    if control is None:
        return {}
    control_mae = soup_mae(control)
    summary: dict[str, Any] = {"control": control_mae, "clean_base": clean_base, "epochs": int(epochs)}
    for role in ("node", "edge"):
        tag = f"{clean_base_arm(clean_base, role)}_adapt{epochs}"
        entry = find_artifact(tag, 0, epochs)
        if entry is None:
            continue
        summary[role] = {
            "soup_valid_mae": soup_mae(entry),
            "delta_vs_control": soup_mae(entry) - control_mae,
            "best_epoch": entry["payload"]["best_epoch"],
            "final_valid_mae": entry["payload"]["final_valid_mae"],
            "json": str(entry["json"].relative_to(REPO_ROOT)),
        }
    _write_json(STAGE_DIRS["c"] / f"adaptation_summary_e{epochs}.json", summary)
    print(f"[stage-c-adapt] {json.dumps(summary, default=str)[:400]}", flush=True)
    return summary


def stage_c_train(threads: int = THREADS, concurrency: int = CONCURRENCY) -> None:
    clean_base = read_clean_base()
    launch(stage_c_train_jobs(clean_base), threads, concurrency)


def stage_gate_c() -> dict[str, Any]:
    clean_base = read_clean_base()
    base_entry = find_artifact(clean_base, 0)
    if base_entry is None:
        raise RuntimeError(f"clean base seed0 missing for {clean_base}")
    base_mae = soup_mae(base_entry)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "clean_base": str(clean_base),
        "base_soup_valid_mae": base_mae,
    }
    for role in ("node", "edge"):
        tag = clean_base_arm(clean_base, role)
        deltas: dict[int, float] = {}
        for seed in range(4):
            entry = find_artifact(tag, seed)
            if entry is not None:
                deltas[int(seed)] = soup_mae(entry) - base_mae
        gate = cm.independence_gate(role, deltas) if deltas else {"verdict": "NO_DATA"}
        adaptation = None
        adapt_path = STAGE_DIRS["c"] / f"adaptation_summary_e{ADAPT_EPOCHS}.json"
        if adapt_path.exists():
            adaptation = _read_json(adapt_path).get(role)
        payload[role] = {"gate": gate, "adaptation": adaptation}
    _write_json(STAGE_DIRS["c"] / "gate_c.json", payload)
    print(f"[gate-c] {json.dumps({k: v for k, v in payload.items() if k in ('node', 'edge')}, default=str)[:500]}", flush=True)
    return payload


def stage_c_extend(threads: int = THREADS, concurrency: int = CONCURRENCY) -> None:
    clean_base = read_clean_base()
    gate = _read_json(STAGE_DIRS["c"] / "gate_c.json") if (STAGE_DIRS["c"] / "gate_c.json").exists() else {}
    for role in ("node", "edge"):
        verdict = (gate.get(role) or {}).get("gate", {}).get("verdict")
        if verdict == "EXTEND_SEEDS":
            launch(
                stage_c_train_jobs(clean_base, seeds=(1, 2)),
                threads,
                concurrency,
            )


# ---------------------------------------------------------------------------
# Stage D — relation simplification
# ---------------------------------------------------------------------------


def stage_d_groups() -> dict[str, Any]:
    _ensure_dirs()
    provenance = audit.verify_relation_groups()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "official_test_loaded": False,
        "used_slice_indices": list(p1.P1_RELATION_INDICES),
        "groups": {name: {"used_slice": [low, high], "raw": audit.RELATION_RAW_LAYOUT.get(name)} for name, (low, high) in audit.RELATION_GROUPS.items()},
        "raw_layout": {name: list(bounds) for name, bounds in audit.RELATION_RAW_LAYOUT.items()},
        "provenance_checks": provenance,
    }
    _write_json(RESULTS_DIR / "relation_groups.json", payload)
    print(f"[stage-d-groups] all_passed={provenance.get('all_passed')}", flush=True)
    return payload


def stage_d_adapt(threads: int = THREADS, concurrency: int = CONCURRENCY, epochs: int = ADAPT_EPOCHS) -> dict[str, Any]:
    clean_base = read_clean_base()
    launch(stage_d_adapt_jobs(clean_base, epochs), threads, concurrency)
    control = find_artifact(f"{clean_base}_M0_adapt{epochs}", 0, epochs)
    control_mae = soup_mae(control) if control is not None else None
    summary: dict[str, Any] = {"clean_base": clean_base, "epochs": int(epochs), "control": control_mae}
    for kind in ("REL-DIST", "REL-DIST-BOUNDARY"):
        entry = find_artifact(f"{kind}_adapt{epochs}", 0, epochs)
        if entry is None:
            continue
        summary[kind] = {
            "soup_valid_mae": soup_mae(entry),
            "delta_vs_control": None if control_mae is None else soup_mae(entry) - control_mae,
            "best_epoch": entry["payload"]["best_epoch"],
            "final_valid_mae": entry["payload"]["final_valid_mae"],
            "curve": entry["payload"]["curve"],
        }
    _write_json(STAGE_DIRS["d"] / f"adaptation_summary_e{epochs}.json", summary)
    print(f"[stage-d-adapt] {json.dumps(summary, default=str)[:400]}", flush=True)
    return summary


def _recovery_trend(curve: Sequence[Mapping[str, Any]], last: int = 5) -> bool:
    values = [float(row["valid_mae"]) for row in curve[-int(last) :]]
    return len(values) >= 2 and values[-1] <= values[0]


def stage_d_decide() -> dict[str, Any]:
    clean_base = read_clean_base()
    adapt = _read_json(STAGE_DIRS["d"] / f"adaptation_summary_e{ADAPT_EPOCHS}.json")
    decision = {"clean_base": clean_base, "screening": adapt, "candidate": None, "reason": ""}
    for kind in ("REL-DIST", "REL-DIST-BOUNDARY"):
        row = adapt.get(kind)
        if row is None:
            continue
        delta = row.get("delta_vs_control")
        if delta is None:
            continue
        if float(delta) > 0.02 and not _recovery_trend(row.get("curve", [])):
            decision.setdefault("rejected", []).append(kind)
            continue
        decision["candidate"] = kind
        decision["reason"] = f"{kind} survived the 20-epoch screen (delta={float(delta):+.5f})"
        break
    if decision["candidate"] is None:
        decision["candidate"] = "REL-FULL-CLEAN"
        decision["reason"] = "no simplification survived the screen; keep the clean-base relation"
    _write_json(STAGE_DIRS["d"] / "screen_decision.json", decision)
    print(f"[stage-d-decide] candidate={decision['candidate']}", flush=True)
    return decision


def stage_d_train(threads: int = THREADS, concurrency: int = CONCURRENCY) -> None:
    decision = _read_json(STAGE_DIRS["d"] / "screen_decision.json")
    kind = str(decision["candidate"])
    if kind == "REL-FULL-CLEAN":
        return
    launch(stage_d_train_jobs(kind), threads, concurrency)


def stage_gate_d() -> dict[str, Any]:
    clean_base = read_clean_base()
    base_entry = find_artifact(clean_base, 0)
    base_mae = soup_mae(base_entry)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "clean_base": str(clean_base),
        "base_soup_valid_mae": base_mae,
    }
    for kind in ("REL-DIST", "REL-DIST-BOUNDARY"):
        deltas: dict[int, float] = {}
        for seed in range(4):
            entry = find_artifact(kind, seed)
            if entry is not None:
                deltas[int(seed)] = soup_mae(entry) - base_mae
        payload[kind] = {
            "per_seed_delta": {int(seed): float(value) for seed, value in deltas.items()},
            "gate": cm.relation_gate(float(np.mean(list(deltas.values())))) if deltas else {"verdict": "NO_DATA"},
        }
    _write_json(STAGE_DIRS["d"] / "gate_d.json", payload)
    print(f"[gate-d] {json.dumps({k: v['gate']['verdict'] for k, v in payload.items() if k.startswith('REL')}, default=str)}", flush=True)
    return payload


def stage_d_extend(threads: int = THREADS, concurrency: int = CONCURRENCY) -> None:
    gate = _read_json(STAGE_DIRS["d"] / "gate_d.json")
    for kind in ("REL-DIST", "REL-DIST-BOUNDARY"):
        row = gate.get(kind) or {}
        if (row.get("gate") or {}).get("verdict") == "EXTEND_SEEDS":
            missing = [seed for seed in (1, 2) if find_artifact(kind, seed) is None]
            if missing:
                launch(stage_d_train_jobs(kind, missing), threads, concurrency)


# ---------------------------------------------------------------------------
# final clean composition + Stage F
# ---------------------------------------------------------------------------


def _adopted_map() -> dict[str, Any]:
    clean_base = read_clean_base()
    result: dict[str, Any] = {"clean_base": clean_base, "node_indep": False, "edge_indep": False, "relation": None}
    gate_c_path = STAGE_DIRS["c"] / "gate_c.json"
    if gate_c_path.exists():
        gate_c = _read_json(gate_c_path)
        for role, key in (("node", "node_indep"), ("edge", "edge_indep")):
            row = gate_c.get(role) or {}
            per_seed = (row.get("gate") or {}).get("per_seed") or {}
            values = [float(value) for value in per_seed.values()]
            if not values:
                continue
            if role == "node":
                result[key] = bool(max(values) <= 0.005 and float(np.mean(values)) <= 0.005)
            else:
                result[key] = bool(float(np.mean(values)) <= 0.003)
    gate_d_path = STAGE_DIRS["d"] / "gate_d.json"
    if gate_d_path.exists():
        gate_d = _read_json(gate_d_path)
        for kind in ("REL-DIST", "REL-DIST-BOUNDARY"):
            row = gate_d.get(kind) or {}
            values = [float(value) for value in ((row.get("per_seed_delta") or {}).values())]
            if values and float(np.mean(values)) <= 0.005:
                result["relation"] = kind
                break
    return result


def stage_final_clean() -> dict[str, Any]:
    adopted = _adopted_map()
    clean_base = adopted["clean_base"]
    mask_kind = clean_base if adopted["relation"] is None else str(adopted["relation"])
    node_binding = "indep" if adopted["node_indep"] else "paired"
    edge_binding = "indep" if adopted["edge_indep"] else "paired"
    spec = cm.CleanMechSpec("FINAL-CLEAN", mask_kind, node_binding=node_binding, edge_binding=edge_binding)
    _write_json(
        RESULTS_DIR / "final_clean.json",
        {
            "adopted": adopted,
            "final_spec": spec.as_dict(),
            "git_commit": _git_commit(),
            "official_test_loaded": False,
        },
    )
    print(f"[final-clean] spec={spec.as_dict()}", flush=True)
    return {"adopted": adopted, "final_spec": spec.as_dict()}


def _final_sparse_reference() -> dict[str, Any] | None:
    """Locate (or train) the seed-0 FINAL-CLEAN sparse reference."""
    final = _read_json(RESULTS_DIR / "final_clean.json")
    adopted = final["adopted"]
    clean_base = adopted["clean_base"]
    # single-simplification cases can reuse an existing run
    if adopted["relation"] is None and not adopted["node_indep"] and not adopted["edge_indep"]:
        return find_artifact(clean_base, 0)
    if adopted["relation"] is not None and not adopted["node_indep"] and not adopted["edge_indep"]:
        return find_artifact(str(adopted["relation"]), 0)
    if adopted["relation"] is None and (adopted["node_indep"] or adopted["edge_indep"]) and not (
        adopted["node_indep"] and adopted["edge_indep"]
    ):
        role = "node" if adopted["node_indep"] else "edge"
        return find_artifact(clean_base_arm(clean_base, role), 0)
    return None


def stage_final_sparse(threads: int = THREADS, concurrency: int = CONCURRENCY) -> None:
    if _final_sparse_reference() is not None:
        return
    final = _read_json(RESULTS_DIR / "final_clean.json")
    spec = cm.CleanMechSpec(**final["final_spec"])
    job = _train_job("f", spec, 0, EPOCHS, "FINAL-CLEAN-SPARSE_seed0", notes="combined final clean sparse reference")
    launch([job], threads, concurrency)


def stage_f_train(threads: int = THREADS, concurrency: int = CONCURRENCY) -> None:
    final = _read_json(RESULTS_DIR / "final_clean.json")
    spec = cm.CleanMechSpec(**final["final_spec"])
    jobs = stage_f_jobs(spec.mask_kind, spec.node_binding, spec.edge_binding, seeds=(0,))
    launch(jobs, threads, concurrency)


def stage_gate_f() -> dict[str, Any]:
    final = _read_json(RESULTS_DIR / "final_clean.json")
    spec = cm.CleanMechSpec(**final["final_spec"])
    reference = _final_sparse_reference()
    if reference is None:
        reference = find_artifact("FINAL-CLEAN-SPARSE", 0)
    if reference is None:
        raise RuntimeError("final sparse reference missing; run stage-final-sparse first")
    sparse_mae = soup_mae(reference)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
        "final_spec": spec.as_dict(),
        "sparse_reference": {"json": str(reference["json"].relative_to(REPO_ROOT)), "source": reference["source"], "soup_valid_mae": sparse_mae},
        "deltas": {},
    }
    for seed in range(4):
        entry = find_artifact("FINAL-CLEAN-DENSE-TIED", seed)
        if entry is None:
            continue
        dense_mae = soup_mae(entry)
        payload["deltas"][int(seed)] = {
            "dense_soup_valid_mae": dense_mae,
            "sparse_soup_valid_mae": sparse_mae,
            "delta_dense_minus_sparse": dense_mae - sparse_mae,
        }
    values = [float(row["delta_dense_minus_sparse"]) for row in payload["deltas"].values()]
    payload["gate"] = cm.specificity_gate(float(np.mean(values))) if values else {"verdict": "NO_DATA"}
    _write_json(STAGE_DIRS["f"] / "gate_f.json", payload)
    print(f"[gate-f] {json.dumps(payload['gate'], default=str)}", flush=True)
    return payload


def stage_f_extend(threads: int = THREADS, concurrency: int = CONCURRENCY) -> None:
    gate = _read_json(STAGE_DIRS["f"] / "gate_f.json").get("gate", {})
    if gate.get("verdict") != "SPARSE_SPECIFIC_CANDIDATE":
        return
    final = _read_json(RESULTS_DIR / "final_clean.json")
    spec = cm.CleanMechSpec(**final["final_spec"])
    missing = [seed for seed in (1, 2) if find_artifact("FINAL-CLEAN-DENSE-TIED", seed) is None]
    if missing:
        launch(stage_f_jobs(spec.mask_kind, spec.node_binding, spec.edge_binding, missing), threads, concurrency)


def stage_report() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": "cpu",
        "official_test_loaded": False,
    }
    for name in ("runtime_budget", "clean_base", "final_clean", "relation_groups", "node_edge_independence_diagnostics"):
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            payload[name] = _read_json(path)
    for name, stage in (("gate_a", "a"), ("gate_c", "c"), ("gate_d", "d"), ("gate_f", "f")):
        path = STAGE_DIRS[stage] / f"{name}.json"
        if path.exists():
            payload[name] = _read_json(path)
    _write_json(RESULTS_DIR / "summary.json", payload)
    print(json.dumps({key: type(value).__name__ for key, value in payload.items()}, indent=2), flush=True)
    return payload


# ---------------------------------------------------------------------------
# chain
# ---------------------------------------------------------------------------


def run_chain(threads: int = THREADS, concurrency: int = CONCURRENCY, overlap: bool = True) -> None:
    _ensure_dirs()
    stage_preflight(threads)
    jobs = stage_a_jobs((1, 2))
    if overlap:
        provisional = [job for job in stage_c_train_jobs("C6") if job["seed"] == 0]
        jobs = jobs + provisional
    launch(jobs, threads, concurrency)
    gate_a = stage_gate_a(threads, concurrency, launch_missing=True)
    clean_base = str(gate_a["clean_base"])
    if overlap and clean_base != "C6":
        print(f"[chain] overlap runs were provisional on C6; canonical controls use {clean_base}", flush=True)
    stage_c_frozen(threads)
    stage_c_adapt(threads, concurrency)
    stage_c_train(threads, concurrency)
    stage_gate_c()
    stage_c_extend(threads, concurrency)
    stage_gate_c()
    stage_c_frozen(threads)
    stage_d_groups()
    stage_d_adapt(threads, concurrency)
    stage_d_decide()
    stage_d_train(threads, concurrency)
    stage_gate_d()
    stage_d_extend(threads, concurrency)
    stage_gate_d()
    stage_final_clean()
    stage_final_sparse(threads, concurrency)
    stage_f_train(threads, concurrency)
    stage_gate_f()
    stage_f_extend(threads, concurrency)
    stage_gate_f()
    stage_b(threads)
    stage_report()
    print("[chain] complete", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="chain",
        choices=[
            "preflight", "train-one", "launch-a", "gate-a", "stage-b",
            "stage-c-frozen", "stage-c-adapt", "stage-c-train", "gate-c", "stage-c-extend",
            "stage-d-groups", "stage-d-adapt", "stage-d-decide", "stage-d-train", "gate-d", "stage-d-extend",
            "final-clean", "stage-final-sparse", "stage-f-train", "gate-f", "stage-f-extend",
            "report", "status", "chain",
        ],
    )
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--job", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--no-overlap", action="store_true")
    parser.add_argument("--arms", default=None)
    args = parser.parse_args(argv)

    if args.stage == "preflight":
        print(json.dumps(stage_preflight(args.threads), indent=2))
    elif args.stage == "train-one":
        job = json.loads(str(args.job))
        print(json.dumps(train_one(job, args.threads), indent=2))
    elif args.stage == "launch-a":
        seeds = [int(value) for value in str(args.seeds or "1,2").split(",") if value.strip()]
        launch(stage_a_jobs(seeds), args.threads, args.concurrency)
    elif args.stage == "gate-a":
        print(json.dumps(stage_gate_a(args.threads, args.concurrency), indent=2))
    elif args.stage == "stage-b":
        arms = None if not args.arms else [value.strip() for value in args.arms.split(",")]
        print(json.dumps(stage_b(args.threads, arms), indent=2, default=str))
    elif args.stage == "stage-c-frozen":
        arms = None if not args.arms else [value.strip() for value in args.arms.split(",")]
        print(json.dumps(stage_c_frozen(args.threads, arms), indent=2, default=str))
    elif args.stage == "stage-c-adapt":
        print(json.dumps(stage_c_adapt(args.threads, args.concurrency), indent=2, default=str))
    elif args.stage == "stage-c-train":
        stage_c_train(args.threads, args.concurrency)
    elif args.stage == "gate-c":
        print(json.dumps(stage_gate_c(), indent=2, default=str))
    elif args.stage == "stage-c-extend":
        stage_c_extend(args.threads, args.concurrency)
    elif args.stage == "stage-d-groups":
        print(json.dumps(stage_d_groups(), indent=2))
    elif args.stage == "stage-d-adapt":
        print(json.dumps(stage_d_adapt(args.threads, args.concurrency), indent=2, default=str))
    elif args.stage == "stage-d-decide":
        print(json.dumps(stage_d_decide(), indent=2, default=str))
    elif args.stage == "stage-d-train":
        stage_d_train(args.threads, args.concurrency)
    elif args.stage == "gate-d":
        print(json.dumps(stage_gate_d(), indent=2, default=str))
    elif args.stage == "stage-d-extend":
        stage_d_extend(args.threads, args.concurrency)
    elif args.stage == "final-clean":
        print(json.dumps(stage_final_clean(), indent=2))
    elif args.stage == "stage-final-sparse":
        stage_final_sparse(args.threads, args.concurrency)
    elif args.stage == "stage-f-train":
        stage_f_train(args.threads, args.concurrency)
    elif args.stage == "gate-f":
        print(json.dumps(stage_gate_f(), indent=2, default=str))
    elif args.stage == "stage-f-extend":
        stage_f_extend(args.threads, args.concurrency)
    elif args.stage == "report":
        print(json.dumps(stage_report(), indent=2, default=str))
    elif args.stage == "status":
        print(json.dumps(_status(), indent=2, default=str))
    else:
        run_chain(args.threads, args.concurrency, overlap=not args.no_overlap)
    return 0


def _status() -> dict[str, Any]:
    payload: dict[str, Any] = {"results_dir": str(RESULTS_DIR)}
    for stage, path in STAGE_DIRS.items():
        payload[stage] = sorted(item.name for item in path.glob("*_e*.json")) if path.exists() else []
    for name in ("runtime_budget.json", "clean_base.json", "final_clean.json", "summary.json", "relation_groups.json", "node_edge_independence_diagnostics.json"):
        item = RESULTS_DIR / name
        payload[name] = item.exists()
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
