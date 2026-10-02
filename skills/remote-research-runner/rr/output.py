"""Human-readable rendering.  Agents should prefer ``--json``."""

from __future__ import annotations

from .util import human_age, now_iso


def _short(ts: str | None) -> str:
    if not ts:
        return "-"
    return ts.replace("T", " ")[:19]


def render_hosts(rows: list[dict]) -> str:
    if not rows:
        return "(no hosts configured)"
    lines = []
    for r in rows:
        offline = " offline" if r.get("offline") else ""
        lines.append(
            f"{r['host']:<8} backend={r['backend']:<8} ssh={r['ssh_alias']:<8} "
            f"pool={r.get('default_pool') or '-':<14} code={r['code']}{offline}"
        )
    return "\n".join(lines)


def render_doctor(report: dict) -> str:
    lines = [
        f"host      {report['host']}  ({report['ssh_alias']}, backend={report['backend']})"
    ]
    for check in report.get("checks", []):
        mark = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL"}.get(check["status"], "?   ")
        lines.append(f"  [{mark}] {check['name']:<18} {check['detail']}")
    pools = report.get("pools") or []
    if pools:
        lines.append("pools:")
        for p in pools:
            unconstrained = not p.get("eligible_nodes") and not p.get("min_driver")
            ready = bool(p.get("provisioned")) and (unconstrained or bool(p.get("compatible_nodes")))
            state = "ready" if ready else "not-ready"
            nodes = ",".join(p.get("compatible_nodes") or []) or ("host-level" if unconstrained else "-")
            lines.append(
                f"  {p['pool']:<18} {state:<10} torch={p.get('torch_backend')} "
                f"min_driver={p.get('min_driver') or '-'} nodes={nodes}"
            )
    summary = report.get("summary", {})
    lines.append(
        f"doctor: {summary.get('ok', 0)} ok / {summary.get('warn', 0)} warn / {summary.get('fail', 0)} fail"
    )
    return "\n".join(lines)


def render_jobs(rows: list[dict], errors: list[dict] | None = None) -> str:
    if not rows and not errors:
        return "(no runs)"
    lines = []
    if rows:
        header = (
            f"{'EXPERIMENT':<24} {'RUN':<10} {'HOST':<7} {'BACKEND':<7} {'POOL':<15} "
            f"{'STATE':<10} {'JOB/PID':<10} {'NODE':<8} {'GPU':<4} {'COMMIT':<12} CREATED"
        )
        lines.append(header)
        for r in rows:
            name = (r.get("experiment") or "-") + (" [legacy]" if r.get("legacy") else "")
            run_short = "-" if r.get("legacy") else (r.get("run_id") or "-")[-8:]
            lines.append(
                f"{name:<24} "
                f"{run_short:<10} "
                f"{(r.get('host') or '-'):<7} "
                f"{(r.get('backend') or '-'):<7} "
                f"{(r.get('pool') or '-'):<15} "
                f"{(r.get('state') or '-'):<10} "
                f"{str(r.get('job') or '-'):<10} "
                f"{str(r.get('node') or '-'):<8} "
                f"{str(r.get('gpu_count') if r.get('gpu_count') is not None else '-'):<4} "
                f"{(r.get('commit') or '-')[:12]:<12} "
                f"{_short(r.get('created_at'))}"
            )
    if errors:
        lines.append("unreachable hosts (results are partial):")
        for e in errors:
            lines.append(f"  {e.get('host')}: {e.get('code')}: {e.get('message')}")
    return "\n".join(lines)


def render_status(s: dict) -> str:
    lines = [
        f"experiment   {s.get('experiment')}",
        f"run_id       {s.get('run_id')}",
        f"host         {s.get('host')}  backend={s.get('backend')}  pool={s.get('pool')}",
        f"state        {s.get('state')}" + ("  (live)" if s.get("live") else ""),
        f"job          {s.get('job') or s.get('job_id') or '-'}"
        + (f"  scheduler={s.get('scheduler_state')}" if s.get("scheduler_state") else ""),
        f"node         {s.get('node') or '-'}",
        f"gpu          {s.get('gpu_count') if s.get('gpu_count') is not None else '-'}"
        + (f"  {','.join(s.get('gpu_names') or [])}" if s.get("gpu_names") else "")
        + (f"  CUDA_VISIBLE_DEVICES={s.get('cuda_visible_devices')}" if s.get("cuda_visible_devices") else ""),
        f"driver       {s.get('driver_version') or '-'}",
        f"torch        {s.get('torch_version') or '-'} (cuda {s.get('torch_cuda') or '-'})",
        f"commit       {(s.get('commit') or '-')}"
        + (f"  branch={s.get('branch')}" if s.get("branch") else "")
        + ("  DIRTY" if s.get("dirty") else ""),
        f"started      {_short(s.get('started_at'))}",
        f"completed    {_short(s.get('completed_at'))}",
        f"exit_code    {s.get('exit_code')}",
    ]
    if s.get("failure_reason"):
        lines.append(f"failure      {s.get('failure_reason')}")
    if s.get("reason"):
        lines.append(f"queue_reason {s.get('reason')}")
    if s.get("elapsed"):
        lines.append(f"elapsed      {s.get('elapsed')}")
    return "\n".join(lines)


def render_pull(p: dict) -> str:
    lines = [f"pulled {p.get('run_id')} → {p.get('dest')}"]
    if p.get("pulled"):
        lines.append("result paths:")
        lines.extend(f"  {x}" for x in p["pulled"])
    if p.get("missing"):
        lines.append("missing on remote (declared but not found):")
        lines.extend(f"  {x}" for x in p["missing"])
    if not p.get("declared"):
        lines.append("(no result paths declared; only meta/logs pulled)")
    return "\n".join(lines)


def render_deploy(d: dict) -> str:
    if d.get("dry_run"):
        return f"dry-run: would deploy {d.get('commit', '')[:12]} to {d['host']}"
    extra = f" (incremental from {(d.get('remote_commit_was') or 'none')[:12]})" if d.get("incremental") else ""
    return f"deployed {d['host']}: {d.get('commit', '')[:12]}{extra}"


def render_run(r: dict) -> str:
    meta = r["meta"]
    return (
        f"launched {meta.get('experiment')} on {meta.get('host')} "
        f"(pool={meta.get('pool')}, backend={meta.get('backend')})\n"
        f"run_id: {meta.get('run_id')}\n"
        f"state:  {meta.get('state')}\n"
        f"follow: rr logs {meta.get('experiment')} --follow   |   rr status {meta.get('experiment')}"
    )