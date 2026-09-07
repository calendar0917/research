"""The ``research`` CLI: the KSVD run control plane.

Typical flow::

    uv run research doctor
    uv run research context
    uv run research run zinc_patch_path_pooling --study zinc-context-gap \\
        --purpose "..." --set model.epochs=2
    uv run research show <run_id>
    uv run research compare --study zinc-context-gap
    uv run research promote <run_id>

The CLI owns every cross-cutting concern (argparse, config loading/snapshots,
run directories, manifests, environment capture, git capture, teeing of
stdout/stderr).  Runners receive the resolved config and a RunContext and
return metrics/artifacts.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import io
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from .runners import RunnerError, get_runner, list_runners
from .runtime import (
    capture_environment,
    capture_git_state,
    config_hash,
    create_run_directory,
    find_duplicate_run,
    list_runs,
    load_local_run,
    load_run,
    new_run_id,
    parse_override_spec,
    promote_run,
    run_fingerprint,
)
from .runtime.config import apply_override, load_base_config
from .runtime.control import (
    load_protocol,
    load_state,
    load_study,
    protocol_for_study,
)
from .runtime.policy import resolve_test_access
from .runtime.git_state import porcelain_status, write_untracked_snapshots
from .runtime.manifest import (
    RunContext,
    RunSpec,
    build_manifest,
    write_manifest,
)
from .runtime.paths import (
    TRACK_ROOT,
    protocols_root,
    records_root,
    resolve_path,
    runs_root,
    studies_root,
)
from .runtime.records import (
    list_claims,
    list_decisions,
    new_claim,
    new_decision,
    set_claim_status,
)
from .runtime.run_store import (
    locate_manifest,
    write_config_resolved,
    write_metrics_file,
    write_patch,
)
from .runtime.serialization import load_yaml

STATUS_EXIT_OK = 0
STATUS_EXIT_WARNING = 1
STATUS_EXIT_ERROR = 2

RUN_MODES = ("scratch", "screen", "confirm", "terminal")


class _Tee:
    """Write to the real stream and a log file simultaneously."""

    def __init__(self, stream: io.TextIOBase, path: Path) -> None:
        self._stream = stream
        self._file = path.open("w", encoding="utf-8")

    def write(self, text: str) -> int:
        self._file.write(text)
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()
        try:
            self._file.flush()
        except ValueError:
            pass

    def close(self) -> None:
        try:
            self._file.close()
        except ValueError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    try:
        code = handler(args)
    except (RunnerError, FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return STATUS_EXIT_ERROR
    return int(code or 0)


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(handler=lambda args: parser.print_help())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research",
        description="KSVD run control plane: doctor, context, run, promote, compare.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="verify the control plane and environment")
    doctor.set_defaults(handler=_cmd_doctor)

    context = sub.add_parser("context", help="print a compact orientation context")
    context.set_defaults(handler=_cmd_context)

    runs = sub.add_parser("runs", help="list recorded runs (local + promoted)")
    runs.add_argument("--study", default=None)
    runs.add_argument("--status", default=None)
    runs.add_argument("--limit", type=int, default=20)
    runs.add_argument(
        "--source",
        choices=("all", "local", "promoted"),
        default="all",
        help="local runs (git-ignored executions), promoted records (git-tracked), or both",
    )
    runs.add_argument("--json", action="store_true")
    runs.set_defaults(handler=_cmd_runs)

    show = sub.add_parser("show", help="show one run manifest and metrics")
    show.add_argument("run_id")
    show.add_argument("--json", action="store_true")
    show.set_defaults(handler=_cmd_show)

    run = sub.add_parser("run", help="execute one registered runner")
    run.add_argument("runner")
    run.add_argument("--study", default=None, help="study id; defaults to runner default study")
    run.add_argument("--candidate", default=None, help="candidate id; defaults to config protocol_id")
    run.add_argument("--purpose", default="", help="one line stating why this run exists")
    run.add_argument("--config", type=Path, default=None, help="override base YAML config")
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--mode", choices=RUN_MODES, default="scratch")
    run.add_argument("--force", action="store_true", help="re-run despite an identical run")
    run.add_argument("--dry-run", action="store_true", help="resolve and print the run plan only")
    run.set_defaults(handler=_cmd_run)

    promote = sub.add_parser("promote", help="create a git-tracked record for a completed run")
    promote.add_argument("run_id")
    promote.add_argument("--note", default=None)
    promote.set_defaults(handler=_cmd_promote)

    compare = sub.add_parser("compare", help="rank comparable runs; INCOMPARABLE otherwise")
    compare.add_argument("run_ids", nargs="*")
    compare.add_argument("--study", default=None)
    compare.add_argument("--metric", default=None, help="override ranking metric key")
    compare.add_argument(
        "--source",
        choices=("all", "local", "promoted"),
        default="all",
        help="which run sources to compare against",
    )
    compare.add_argument("--override", action="store_true", help="rank despite mismatches (warned)")
    compare.add_argument("--json", action="store_true")
    compare.set_defaults(handler=_cmd_compare)

    claim = sub.add_parser("claim", help="manage claim records")
    claim_sub = claim.add_subparsers(dest="claim_command", required=True)
    claim_new = claim_sub.add_parser("new")
    claim_new.add_argument("--statement", required=True)
    claim_new.add_argument("--study", default=None)
    claim_new.add_argument("--status", default="tentative")
    claim_new.add_argument("--scope", default=None)
    claim_new.add_argument("--evidence", action="append", default=[])
    claim_new.add_argument("--contradicting", action="append", default=[])
    claim_new.add_argument("--limitation", action="append", default=[])
    claim_new.add_argument("--implication", default=None)
    claim_new.set_defaults(handler=_cmd_claim_new)
    claim_list = claim_sub.add_parser("list")
    claim_list.set_defaults(handler=_cmd_claim_list)
    claim_status = claim_sub.add_parser("status")
    claim_status.add_argument("claim_id")
    claim_status.add_argument("status")
    claim_status.set_defaults(handler=_cmd_claim_status)

    decision = sub.add_parser("decision", help="manage decision records")
    decision_sub = decision.add_subparsers(dest="decision_command", required=True)
    decision_new = decision_sub.add_parser("new")
    decision_new.add_argument("--decision", required=True)
    decision_new.add_argument("--because", required=True)
    decision_new.add_argument("--study", default=None)
    decision_new.add_argument("--alternative", action="append", default=[])
    decision_new.add_argument("--revisit-if", default=None)
    decision_new.set_defaults(handler=_cmd_decision_new)
    decision_list = decision_sub.add_parser("list")
    decision_list.set_defaults(handler=_cmd_decision_list)

    _add_common(parser)
    return parser


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def _doctor_check(name: str, ok: bool, detail: str, severity: str = "ok") -> tuple[str, bool, str]:
    if not ok:
        severity = "error"
    return name, ok, detail, severity


def _cmd_doctor(args: argparse.Namespace) -> int:
    checks: list[tuple[str, bool, str]] = []

    git = capture_git_state()
    checks.append(("git.repo", git.commit is not None, git.commit or "no git available"))
    checks.append(
        ("git.dirty", True, f"{len(git.untracked)} untracked, diff_hash={git.diff_hash[:12] or 'clean'}")
    )

    state = load_state()
    checks.append(("control.state", True, f"STATE.yaml at {TRACK_ROOT / 'STATE.yaml'} ({len(state)} keys)"))
    checks.append(("control.protocols", protocols_root().is_dir(), str(protocols_root())))
    checks.append(("control.studies", studies_root().is_dir(), str(studies_root())))
    checks.append(("control.records", records_root().is_dir(), str(records_root())))
    checks.append(("control.runs", runs_root().is_dir(), str(runs_root())))

    environment = capture_environment()
    missing = [name for name, version in environment["packages"].items() if version is None]
    checks.append(
        ("env.python", environment["python"].startswith("3.12"), environment["python"]),
    )
    checks.append(
        (
            "env.packages",
            not missing,
            f"{len(environment['packages'])}/{len(list(environment['packages']))} reportable versions"
            + (f"; missing: {', '.join(missing)}" if missing else ""),
        ),
    )

    try:
        runners = list_runners()
        checks.append(("runners.registry", bool(runners), ", ".join(runner["name"] for runner in runners)))
    except Exception as exc:  # noqa: BLE001 - doctor reports failures, not raises
        checks.append(("runners.registry", False, repr(exc)))

    from .runtime.fingerprints import zinc_fingerprints as _zfp

    zinc_root = resolve_path("data/ZINC")
    try:
        payload = _zfp(zinc_root, expected_sizes={"train": 10_000, "val": 1_000, "test": 1_000})
        sizes = payload["split"]["sizes"]
        available = all(size == expected for size, expected in zip((sizes[k] for k in ("train", "val", "test")), (10_000, 1_000, 1_000)))
        checks.append(("data.zinc", available, f"sizes={sizes} raw_files={len(payload['raw_files'])}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("data.zinc", False, repr(exc)))

    ok_count = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        marker = "ok" if ok else "FAIL"
        print(f"[{marker}] {name}: {detail}")
    print(f"doctor: {ok_count}/{len(checks)} checks ok")
    if ok_count == len(checks):
        return STATUS_EXIT_OK
    return STATUS_EXIT_ERROR


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------


def _cmd_context(args: argparse.Namespace) -> int:
    lines: list[str] = []
    state = load_state()

    git = capture_git_state()
    status_now = porcelain_status() or ""
    dirty_lines = [line for line in status_now.splitlines() if line.strip()]
    lines.append("# KSVD research context")
    lines.append(f"phase: {state.get('phase', 'unknown')}")
    lines.append(
        f"git: {git.commit[:8] if git.commit else 'none'} dirty={git.dirty} "
        f"dirty_files={len(dirty_lines)} diff_hash={(git.diff_hash or 'clean')[:12]}"
    )
    if dirty_lines:
        lines.append("git untracked/modified (first 10):")
        lines.extend(f"  {line}" for line in dirty_lines[:10])
    lines.append("")
    lines.append(f"active studies: {', '.join(state.get('active_studies', []))}")
    lines.append(f"focus study: {state.get('focus_study')}")
    lines.append("")
    lines.append("## study / protocol rules")
    for study_id in state.get("active_studies", []):
        try:
            study = load_study(study_id)
            protocol = protocol_for_study(study_id)
        except FileNotFoundError:
            lines.append(f"- {study_id}: MISSING")
            continue
        question = str(study.get("question", "")).replace("\n", " ")[:100]
        metric = protocol.get("metric", {}).get("ranking_metric", "?")
        direction = protocol.get("metric", {}).get("metric_direction", "?")
        test_policy = protocol.get("test_policy", "unguarded")
        lines.append(f"- {study_id}: metric={metric} ({direction}) test_policy={test_policy}")
        lines.append(f"  question: {question}")
    lines.append("## files worth reading")
    for relative in (
        "STATE.yaml",
        "protocols/zinc-context-gap.yaml",
        "protocols/molhiv-cross-scaffold-interaction.yaml",
        "studies/zinc-context-gap.yaml",
        "studies/molhiv-cross-scaffold-interaction.yaml",
    ):
        path = TRACK_ROOT / relative
        lines.append(f"- {path} ({'exists' if path.is_file() else 'MISSING'})")
    lines.append("")
    lines.append("## baseline / candidate")
    baseline = state.get("baseline_record") or state.get("baseline_legacy")
    lines.append(f"baseline: {baseline or 'not promoted yet'}")
    lines.append(f"current candidate: {state.get('current_candidate')}")
    lines.append("")
    lines.append("## recent runs")
    runs = list_runs(limit=8, source="all")
    if runs:
        for run in runs:
            details = ",".join(
                f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in (run.get("metrics") or {}).items()
            )[:110]
            source = run.get("source") or "local"
            if run.get("promoted") and source == "local":
                source = "local+promoted"
            lines.append(
                f"- {run['run_id']} [source={source} {run.get('mode')}/{run.get('status')}] "
                f"{run.get('study_id')} {details or '-'}"
            )
    else:
        lines.append("- none yet; run `uv run research run ...`")
    lines.append("")
    lines.append("## claims")
    for claim_id, path in list_claims()[:6]:
        payload = load_yaml(path)
        lines.append(f"- {claim_id} [{payload.get('status')}] {str(payload.get('statement'))[:70]}")
    lines.append("## decisions")
    for decision_id, path in list_decisions()[:6]:
        payload = load_yaml(path)
        lines.append(f"- {decision_id}: {str(payload.get('decision'))[:60]}")
    lines.append("")
    lines.append("## open questions")
    open_questions = state.get("open_questions") or []
    lines.extend(f"- {question}" for question in open_questions[:12])
    lines.append("")
    lines.append("## next queue")
    next_queue = state.get("next_queue") or []
    lines.extend(f"- {item}" for item in next_queue[:12])
    lines.append("")
    lines.append("## guardrails")
    guardrails = state.get("guardrails") or []
    lines.extend(f"- {item}" for item in guardrails[:12])
    print("\n".join(lines))
    return STATUS_EXIT_OK


# ---------------------------------------------------------------------------
# runs / show
# ---------------------------------------------------------------------------


def _runs_metric_summary(run: dict[str, Any]) -> str:
    metrics = run.get("metrics") or {}
    parts = [
        f"{key}={value:.4f}" if isinstance(value, (int, float)) else f"{key}={value}"
        for key, value in metrics.items()
    ]
    return ", ".join(parts)[:60]


def _cmd_runs(args: argparse.Namespace) -> int:
    runs = list_runs(
        study_id=args.study,
        status=args.status,
        limit=args.limit,
        source=args.source,
    )
    if args.json:
        print(json.dumps(runs, indent=2, ensure_ascii=False))
        return STATUS_EXIT_OK
    if not runs:
        print("no runs recorded")
        return STATUS_EXIT_OK
    print(f"{'run_id':<28} {'source':<10} {'mode':<8} {'status':<10} {'study':<26} {'metrics':<64} purpose")
    for run in runs:
        source = str(run.get("source") or "?")
        if run.get("promoted"):
            source = "local+promoted" if source == "local" else "promoted"
        purpose = (run.get("purpose") or "")[:30]
        print(
            f"{run['run_id']:<28} {source:<10} {run.get('mode', '-'):<8} "
            f"{run.get('status', '-'):<10} {str(run.get('study_id') or '-'):<26} "
            f"{_runs_metric_summary(run):<64} {purpose}"
        )
    return STATUS_EXIT_OK


def _cmd_show(args: argparse.Namespace) -> int:
    manifest = load_run(args.run_id)
    if args.json:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return STATUS_EXIT_OK
    git = manifest.get("git") or {}
    source = manifest.get("source") or "local"
    local_available = source == "local"
    print(f"run_id:        {manifest['run_id']}")
    print(
        "source:        "
        + source
        + (" (promoted record: durable, git-tracked)" if manifest.get("promoted") else "")
        + ("; local artifacts: unavailable" if not local_available else "")
    )
    print(f"runner:        {manifest.get('runner') or '-'}")
    print(f"study:         {manifest.get('study_id')}  protocol: {manifest.get('protocol_id')}")
    print(f"candidate:     {manifest.get('candidate_id')}")
    print(f"mode:          {manifest.get('mode')}  status: {manifest.get('status')}")
    print(f"purpose:       {manifest.get('purpose') or '-'}")
    print(f"started:       {manifest.get('started_at')}  ended: {manifest.get('ended_at')}")
    print(f"runtime:       {manifest.get('runtime_seconds')} s")
    print(f"config_hash:   {manifest.get('config_hash') or '-'}")
    print(f"protocol_hash: {manifest.get('protocol_hash') or 'legacy-unknown'}")
    print(f"git:           commit={git.get('commit')} dirty={git.get('dirty')}")
    print(f"diff_hash:     {git.get('diff_hash') or 'clean'}")
    print(f"test_access:   {manifest.get('test_access') or '-'}")
    print(f"seeds:         {manifest.get('seeds') or []}")
    print("metrics:")
    for key, value in (manifest.get("metrics") or {}).items():
        print(f"  {key}: {value}")
    if local_available:
        run_dir = locate_manifest(args.run_id).parent
        print(f"run directory: {run_dir}")
        if run_dir.is_dir():
            for child in sorted(run_dir.iterdir()):
                suffix = "/" if child.is_dir() else ""
                print(f"  {child.name}{suffix}")
        else:
            print("  (not present)")
    if not local_available and manifest.get("record_path"):
        print(f"record:        {manifest['record_path']}")
        print("local artifacts: unavailable (execution output not present on this machine)")
    if manifest.get("error"):
        print(f"error: {manifest['error']}")
    return STATUS_EXIT_OK


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _resolve_run_plan(args: argparse.Namespace) -> dict[str, Any]:
    runner = get_runner(args.runner)
    study_id = args.study or runner.default_study or "unassigned"
    study = None
    protocol = None
    if study_id != "unassigned":
        study = load_study(study_id)
        protocol = protocol_for_study(study_id)

    if args.config:
        base_config = load_base_config(resolve_path(args.config))
    else:
        base_config = load_base_config(resolve_path(runner.default_config))

    config = dict(base_config)
    for spec in args.set:
        key, value = parse_override_spec(spec)
        config = apply_override(config, key, value)
    if args.seed is not None:
        config = apply_override(config, "seed", args.seed)

    test_access = resolve_test_access(protocol, args.mode)
    candidate_id = args.candidate or str(config.get("protocol_id") or "unassigned")
    protocol_id = str(protocol.get("id")) if protocol else "unassigned"
    seeds = [int(config.get("seed", 0))]

    dataset_payload, split_fingerprint = runner.fingerprints(config)
    git = capture_git_state()
    config_hash_value = config_hash(config)
    spec_fingerprint = run_fingerprint(
        runner=runner.name,
        study_id=study_id,
        protocol_id=protocol_id,
        candidate_id=candidate_id,
        config=config,
        seeds=seeds,
        git=git.to_dict(),
        dataset_fingerprint=dataset_payload,
        split_fingerprint=split_fingerprint,
    )
    return {
        "runner": runner,
        "study_id": study_id,
        "study": study,
        "protocol": protocol,
        "config": config,
        "candidate_id": candidate_id,
        "protocol_id": protocol_id,
        "seeds": seeds,
        "dataset_fingerprint": dataset_payload,
        "split_fingerprint": split_fingerprint,
        "git": git,
        "config_hash": config_hash_value,
        "fingerprint": spec_fingerprint,
        "test_access": test_access,
        "blocked": test_access == "blocked",
    }


def _cmd_run(args: argparse.Namespace) -> int:
    plan = _resolve_run_plan(args)
    if args.dry_run:
        print(f"runner:       {plan['runner'].name}")
        print(f"study:        {plan['study_id']}")
        print(f"protocol:     {plan['protocol_id']}")
        print(f"candidate:    {plan['candidate_id']}")
        print(f"mode:         {args.mode}")
        print(f"seed:         {plan['seeds']}")
        print(f"config_hash:  {plan['config_hash']}")
        print(f"fingerprint:  {plan['fingerprint'][:16]}... ({plan['fingerprint']})")
        print(f"test_access:  {plan['test_access']}")
        print(f"base_config:  {plan['runner'].default_config}")
        print(f"overrides:    {args.set}")
        return STATUS_EXIT_OK

    existing = find_duplicate_run(
        list_runs(status="completed", source="all"), plan["fingerprint"]
    )
    if existing and not args.force:
        print(
            f"identical run already exists: {existing['run_id']} "
            f"(metrics={existing.get('metrics')})"
        )
        print("rerun with --force to execute anyway")
        return STATUS_EXIT_ERROR

    run_id = new_run_id()
    run_dir = create_run_directory(run_id)
    started = datetime.now()
    git = plan["git"]
    spec = RunSpec(
        run_id=run_id,
        study_id=plan["study_id"],
        candidate_id=plan["candidate_id"],
        protocol_id=plan["protocol_id"],
        mode=args.mode,
        purpose=args.purpose,
        runner=plan["runner"].name,
        config_hash=plan["config_hash"],
        dataset_fingerprint=plan["dataset_fingerprint"],
        split_fingerprint=plan["split_fingerprint"],
        seeds=plan["seeds"],
        test_access=plan["test_access"],
        fingerprint=plan["fingerprint"],
        run_dir=run_dir,
        git=git.to_dict(),
        environment=capture_environment(),
    )
    manifest = build_manifest(spec=spec, started_at=started, status="running", config=plan["config"])
    write_manifest(run_dir, manifest)
    write_config_resolved(run_dir, plan["config"])
    write_patch(run_dir, git.patch)
    write_untracked_snapshots(run_dir, git.untracked)

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    context = RunContext(
        run_id=run_id,
        run_dir=run_dir,
        artifact_dir=Path(run_dir) / "artifacts",
        mode=args.mode,
        protocol=plan["protocol"],
        study=plan["study"],
        test_access=plan["test_access"],
    )
    metrics: dict[str, Any] = {}
    status = "failed"
    error: str | None = None
    started_wall = datetime.now()
    out_tee = _Tee(sys.stdout, stdout_path)
    err_tee = _Tee(sys.stderr, stderr_path)
    try:
        with redirect_stdout(out_tee), redirect_stderr(err_tee):
            result = plan["runner"].run(plan["config"], context)
            metrics = result.metrics
            status = result.status
    except Exception as exc:  # noqa: BLE001 - record everything, surface summary
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(file=sys.stderr)
    finally:
        out_tee.close()
        err_tee.close()
    ended = datetime.now()
    runtime_seconds = (ended - started_wall).total_seconds()
    if status != "completed":
        error = error or f"runner returned status {status!r}"

    write_metrics_file(run_dir, metrics)
    final_manifest = build_manifest(
        spec=spec,
        started_at=started,
        status=status,
        metrics=metrics,
        ended_at=ended,
        runtime_seconds=runtime_seconds,
        error=error,
        config=plan["config"],
    )
    write_manifest(run_dir, final_manifest)

    print(f"run {run_id} {status} ({runtime_seconds:.1f}s) {run_dir}", flush=True)
    if status == "completed":
        print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    else:
        print(f"error: {error}", flush=True)
        return STATUS_EXIT_ERROR
    return STATUS_EXIT_OK


# ---------------------------------------------------------------------------
# promote / compare
# ---------------------------------------------------------------------------


def _cmd_promote(args: argparse.Namespace) -> int:
    # Promotion needs the local execution truth (resolved config, run
    # directory, provenance); a promoted record is never promoted again.
    manifest = load_local_run(args.run_id)
    if manifest.get("status") != "completed":
        print(f"run {args.run_id} is not completed ({manifest.get('status')}); refusing to promote")
        return STATUS_EXIT_ERROR
    manifest_path = locate_manifest(args.run_id)
    config_path = manifest_path.parent / "config.resolved.yaml"
    config = load_yaml(config_path) if config_path.is_file() else None
    record = promote_run(
        manifest,
        config=config,
        run_dir=manifest_path.parent,
        meta={"note": args.note} if args.note else {},
    )
    path = records_root() / "runs" / f"{record['run_id']}.json"
    print(f"promoted {record['run_id']} -> {path}")
    return STATUS_EXIT_OK


def metric_for_protocol(protocol_id: str) -> tuple[str, str]:
    """Return (ranking metric key, direction) declared by a protocol.

    Schema priority: ``metric.ranking_metric`` / ``metric.metric_direction``
    (current schema) -> legacy top-level ``ranking_metric`` /
    ``metric_direction`` -> defaults (``valid_mae`` / ``lower_is_better``).
    An unrecognized direction is a hard error: silently ranking with the
    wrong polarity would produce a confidently wrong scientific answer.
    """
    try:
        protocol = load_protocol(protocol_id)
    except FileNotFoundError:
        protocol = {}
    metric_config = protocol.get("metric") or {}
    if not isinstance(metric_config, dict):
        raise ValueError(
            f"protocol {protocol_id!r} has non-mapping 'metric'; cannot read ranking metric"
        )
    metric = str(
        metric_config.get("ranking_metric")
        or protocol.get("ranking_metric")
        or "valid_mae"
    )
    direction = str(
        metric_config.get("metric_direction")
        or protocol.get("metric_direction")
        or "lower_is_better"
    )
    if direction not in {"lower_is_better", "higher_is_better"}:
        raise ValueError(
            f"protocol {protocol_id!r} declares invalid metric_direction {direction!r}; "
            "refusing to rank with unknown polarity"
        )
    return metric, direction


def _cmd_compare(args: argparse.Namespace) -> int:
    source = getattr(args, "source", "all")
    if args.study:
        manifests = [
            run
            for run in list_runs(study_id=args.study, source=source)
            if run.get("status") == "completed"
        ]
    else:
        manifests = [load_run(run_id, source=source) for run_id in args.run_ids]
    if not manifests:
        print("no completed runs to compare")
        return STATUS_EXIT_ERROR

    if args.metric:
        metric = args.metric
    else:
        metric_names = {
            metric_for_protocol(str(manifest.get("protocol_id") or "none"))[0]
            for manifest in manifests
        }
        if len(metric_names) != 1:
            print(
                "runs disagree on the ranking metric "
                f"({', '.join(sorted(metric_names))}); pass --metric <key>"
            )
            return STATUS_EXIT_ERROR
        metric = sorted(metric_names)[0]

    groups, compatible = partition_comparable(manifests)
    if not compatible and not args.override:
        print("INCOMPARABLE: runs differ on protocol_id / dataset_fingerprint / split_fingerprint:")
        for key, group in sorted(groups.items()):
            value = manifest_metric(group[0], metric)
            print(
                f"  protocol={key[0]} dataset={key[1][:12]} split={key[2][:12]} "
                f"runs={[row['run_id'] for row in group]} value={value}"
            )
        print("pass --override to rank anyway (the comparison then ignores comparability)")
        return STATUS_EXIT_ERROR

    if not compatible and args.override:
        print("WARNING: comparing across different protocol/dataset/split fingerprints")

    direction_groups: dict[str, list[str]] = {}
    for manifest in manifests:
        if manifest_metric(manifest, metric) is None:
            continue
        direction = metric_for_protocol(str(manifest.get("protocol_id") or "none"))[1]
        direction_groups.setdefault(direction, []).append(str(manifest.get("run_id")))
    if len(direction_groups) > 1:
        print("METRIC DIRECTION CONFLICT: runs assign different polarity to the same metric")
        print(f"  metric: {metric}")
        for direction, run_ids in sorted(direction_groups.items()):
            print(f"  {direction}: {', '.join(run_ids)}")
        print("refusing to rank; use --override only for protocol/dataset/split comparability")
        return STATUS_EXIT_ERROR
    direction = next(iter(direction_groups), "lower_is_better")

    missing = [
        str(manifest.get("run_id"))
        for manifest in manifests
        if manifest_metric(manifest, metric) is None
    ]
    if missing:
        print("MISSING METRIC:")
        for run_id in missing:
            print(f"  {run_id}")
        print(f"metric={metric!r}: runs without this metric receive no rank")

    rows = [
        (manifest, direction)
        for manifest in manifests
        if manifest_metric(manifest, metric) is not None
    ]
    if not rows:
        print(f"no run has metric {metric!r}; no ranking possible")
        return STATUS_EXIT_ERROR

    rows.sort(
        key=lambda item: float(manifest_metric(item[0], metric)),
        reverse=direction == "higher_is_better",
    )
    for rank, (manifest, _direction) in enumerate(rows, start=1):
        value = manifest_metric(manifest, metric)
        value_text = f"{value:.6f}"
        print(
            f"#{rank:<2} {manifest['run_id']}  {metric}={value_text} "
            f"(protocol={manifest.get('protocol_id')} mode={manifest.get('mode')} "
            f"test_access={manifest.get('test_access')})"
        )
    return STATUS_EXIT_OK


def manifest_metric(manifest: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read exactly the requested metric.  No cross-metric fallback.

    Scientific rule: what you ask for is what you read; a missing metric is
    missing, never silently replaced with a different metric.
    """
    return (manifest.get("metrics") or {}).get(key, default)


def comparability_key(manifest: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(manifest.get("protocol_id")),
        str(manifest.get("dataset_fingerprint", {}).get("dataset_fingerprint", "?")),
        str(manifest.get("split_fingerprint")),
    )


def partition_comparable(
    manifests: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str, str], list[dict[str, Any]]], bool]:
    """Group runs by comparability key; ``compatible`` is True iff one group."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for manifest in manifests:
        groups.setdefault(comparability_key(manifest), []).append(manifest)
    return groups, len(groups) == 1


# ---------------------------------------------------------------------------
# claim / decision
# ---------------------------------------------------------------------------


def _cmd_claim_new(args: argparse.Namespace) -> int:
    record = new_claim(
        statement=args.statement,
        study_id=args.study,
        status=args.status,
        scope=args.scope,
        supporting_evidence=args.evidence,
        contradicting_evidence=args.contradicting,
        limitations=args.limitation,
        implication=args.implication,
    )
    print(f"created claim {record['claim_id']} -> records/claims/{record['claim_id']}.yaml")
    return STATUS_EXIT_OK


def _cmd_claim_list(args: argparse.Namespace) -> int:
    for claim_id, path in list_claims():
        payload = load_yaml(path)
        print(f"{claim_id} [{payload.get('status')}] {str(payload.get('statement'))[:90]}")
    return STATUS_EXIT_OK


def _cmd_claim_status(args: argparse.Namespace) -> int:
    record = set_claim_status(args.claim_id, args.status)
    print(f"{record['claim_id']} -> {record['status']}")
    return STATUS_EXIT_OK


def _cmd_decision_new(args: argparse.Namespace) -> int:
    record = new_decision(
        decision=args.decision,
        because=args.because,
        alternatives_rejected=args.alternative,
        revisit_if=args.revisit_if,
        study_id=args.study,
    )
    print(f"created decision {record['decision_id']} -> records/decisions/{record['decision_id']}.yaml")
    return STATUS_EXIT_OK


def _cmd_decision_list(args: argparse.Namespace) -> int:
    for decision_id, path in list_decisions():
        payload = load_yaml(path)
        print(f"{decision_id}: {str(payload.get('decision'))[:90]}")
    return STATUS_EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
