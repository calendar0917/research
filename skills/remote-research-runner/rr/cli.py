"""`rr` command line interface.

Stable, non-interactive, JSON-capable.  This is the single user-facing entry
point for remote research execution; the Skill and (future) MCP wrap it.
"""

from __future__ import annotations

import argparse
import sys
import time

from . import __version__
from .config import load_config
from .errors import RRException
from . import output as o
from . import services as svc
from .util import write_json_stdout


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rr",
        description="Research remote execution interface (host/pool/regime abstraction over SSH, Git, Slurm, uv).",
    )
    p.add_argument("--version", action="version", version=f"rr {__version__}")
    p.add_argument("--config", help="path to a config override (TOML)")
    p.add_argument("--json", action="store_true", help="machine-readable JSON output")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("hosts", help="list configured hosts")

    d = sub.add_parser("doctor", help="check a host end to end")
    d.add_argument("host")
    d.add_argument("--no-probe", action="store_true", help="skip node driver cache/probe entirely")
    d.add_argument(
        "--refresh",
        action="store_true",
        help="submit short Slurm driver-probe jobs (default: non-invasive cache only)",
    )

    dep = sub.add_parser("deploy", help="deploy the current committed revision")
    dep.add_argument("host")
    dep.add_argument("--pool", help="sync the environment for this pool")
    dep.add_argument(
        "--allow-dirty",
        action="store_true",
        help="proceed despite tracked local modifications (they are NOT deployed;"
        " the remote still runs its committed revision)",
    )
    dep.add_argument("--dry-run", action="store_true")

    r = sub.add_parser("run", help="launch an experiment (process or Slurm backend)")
    r.add_argument("host")
    r.add_argument("experiment")
    r.add_argument("--pool")
    r.add_argument("--gpus", type=int)
    r.add_argument("--cpus", type=int)
    r.add_argument("--mem")
    r.add_argument("--time", dest="time_limit")
    r.add_argument("--result", action="append", dest="result_paths", metavar="REMOTE_PATH",
                   help="declare a remote result path to sync on `rr pull` (repeatable)")
    r.add_argument(
        "--allow-dirty",
        action="store_true",
        help="proceed despite tracked local modifications (they are NOT run; the"
        " remote executes its committed revision, so uncommitted edits are invisible)",
    )
    r.add_argument(
        "--allow-stale",
        action="store_true",
        help="run the remote's existing commit/deployment even if it differs (smoke tests only)",
    )

    j = sub.add_parser("jobs", help="list runs across hosts")
    j.add_argument("hosts", nargs="*", help="restrict to these hosts")

    s = sub.add_parser("status", help="show one run")
    s.add_argument("ident")
    s.add_argument("--host")
    s.add_argument("--run-id")

    lg = sub.add_parser("logs", help="show run logs")
    lg.add_argument("ident")
    lg.add_argument("--host")
    lg.add_argument("--run-id")
    lg.add_argument("-n", "--lines", type=int, default=40)
    lg.add_argument("--stderr", action="store_true")
    lg.add_argument("--follow", action="store_true")

    c = sub.add_parser("cancel", help="cancel a run")
    c.add_argument("ident")
    c.add_argument("--host")
    c.add_argument("--run-id")

    pl = sub.add_parser("pull", help="sync run outputs back locally")
    pl.add_argument("ident")
    pl.add_argument("--host")
    pl.add_argument("--run-id")
    pl.add_argument("--dest")
    pl.add_argument("--path", action="append", dest="paths", metavar="REMOTE_PATH")
    pl.add_argument("--light", action="store_true", help="skip checkpoint-like files")

    return p


def _split_command(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split `rr run ... -- cmd` at the first bare `--`."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def _extract_globals(argv: list[str]) -> tuple[list[str], bool, str | None]:
    """Allow --json/--config to appear after the subcommand too."""
    json_flag = False
    config_path = None
    out: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--json":
            json_flag = True
        elif arg == "--config":
            i += 1
            if i < len(argv):
                config_path = argv[i]
        elif arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
        else:
            out.append(arg)
        i += 1
    return out, json_flag, config_path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    pre, command = _split_command(argv)
    pre, json_flag, config_path = _extract_globals(pre)
    parser = _build_parser()
    args = parser.parse_args(pre)
    config = load_config(config_path or args.config)
    use_json = json_flag or bool(args.json)

    try:
        if args.cmd == "hosts":
            rows = svc.list_hosts(config)
            if use_json:
                write_json_stdout({"ok": True, "hosts": rows})
            else:
                print(o.render_hosts(rows))
            return 0

        if args.cmd == "doctor":
            report = svc.doctor(config, args.host, probe=not args.no_probe, refresh=args.refresh)
            if use_json:
                write_json_stdout({"ok": report.get("ok", False), **report})
            else:
                print(o.render_doctor(report))
            return 0 if report.get("ok") else 1

        if args.cmd == "deploy":
            result = svc.deploy(
                config,
                args.host,
                pool_name=args.pool,
                allow_dirty=args.allow_dirty,
                dry_run=args.dry_run,
            )
            if use_json:
                write_json_stdout({"ok": True, **result})
            else:
                print(o.render_deploy(result))
            return 0

        if args.cmd == "run":
            result = svc.run(
                config,
                args.host,
                args.experiment,
                command,
                pool_name=args.pool,
                gpus=args.gpus,
                cpus=args.cpus,
                mem=args.mem,
                time_limit=args.time_limit,
                result_paths=args.result_paths,
                allow_dirty=args.allow_dirty,
                allow_stale=args.allow_stale,
            )
            if use_json:
                write_json_stdout({"ok": True, **result})
            else:
                print(o.render_run(result))
            return 0

        if args.cmd == "jobs":
            report = svc.list_jobs_report(config, args.hosts or None)
            if use_json:
                write_json_stdout({"ok": True, **report})
            else:
                print(o.render_jobs(report["jobs"], report.get("errors")))
            return 0

        if args.cmd == "status":
            s = svc.status(config, args.ident, host_name=args.host, run_id=args.run_id)
            if use_json:
                write_json_stdout({"ok": True, **s})
            else:
                print(o.render_status(s))
            return 0 if s.get("state") != "failed" else 1

        if args.cmd == "logs":
            if args.follow:
                return _follow(config, args)
            result = svc.logs(
                config,
                args.ident,
                host_name=args.host,
                run_id=args.run_id,
                lines=args.lines,
                which="stderr" if args.stderr else "stdout",
            )
            if use_json:
                write_json_stdout({"ok": True, **result})
            else:
                print(result["text"])
            return 0

        if args.cmd == "cancel":
            result = svc.cancel(config, args.ident, host_name=args.host, run_id=args.run_id)
            if use_json:
                write_json_stdout({"ok": True, **result})
            else:
                print(f"cancelled {result.get('run_id')} ({result.get('job_id') or result.get('pid') or ''})")
            return 0

        if args.cmd == "pull":
            result = svc.pull(
                config,
                args.ident,
                host_name=args.host,
                run_id=args.run_id,
                dest=args.dest,
                paths=args.paths,
                light=args.light,
            )
            if use_json:
                write_json_stdout({"ok": True, **result})
            else:
                print(o.render_pull(result))
            return 0

        parser.error(f"unknown command {args.cmd}")
        return 2

    except RRException as exc:
        if use_json:
            write_json_stdout(exc.as_dict())
        else:
            print(f"rr: {exc.message}", file=sys.stderr)
            for key, value in exc.details.items():
                print(f"  {key}: {value}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("rr: interrupted", file=sys.stderr)
        return 130


def follow_should_stop(
    live: dict,
    meta_state: str | None,
    idle_polls: int,
    *,
    max_idle: int = 15,
) -> bool:
    """Decide whether `rr logs --follow` should stop.

    Terminal states (including the Slurm states reconciled from sacct) stop
    immediately so follow never hangs on a CANCELLED/OOM/TIMEOUT job.  PENDING
    is a legitimate long wait; anything else that is not live gets a bounded
    grace period.
    """
    from .meta import is_terminal

    if is_terminal(meta_state) or is_terminal(live.get("state")):
        return True
    if live.get("live"):
        return False
    if live.get("state") == "pending":
        return False
    return idle_polls >= max_idle


def _follow(config, args) -> int:
    from .backends import get_backend
    from .meta import is_terminal

    host, meta = svc.resolve_run(config, args.ident, host_name=args.host, run_id=args.run_id)
    backend = get_backend(host, config.ssh_opts)
    run_dir = backend.run_dir_of(meta)
    which = "stderr" if args.stderr else "stdout"
    seen = ""
    idle = 0

    def emit(text: str) -> None:
        nonlocal seen
        if text == seen:
            return
        sys.stdout.write(text[len(seen):] if text.startswith(seen) else text)
        sys.stdout.flush()
        seen = text

    try:
        while True:
            emit(backend.tail_logs(run_dir, which, args.lines, meta))
            try:
                live = backend.live_state(meta)
            except RRException:
                live = {"state": meta.get("state"), "live": False}
            try:
                updated = backend.reconcile(meta, live)
            except RRException:
                updated = None
            if updated is not None:
                meta = updated
            if follow_should_stop(live, meta.get("state"), idle):
                if not is_terminal(meta.get("state")) and not live.get("live"):
                    sys.stdout.write(
                        f"\n[rr] job is no longer live and no terminal state was recorded "
                        f"(state={live.get('state')}); stopping follow\n"
                    )
                    sys.stdout.flush()
                emit(backend.tail_logs(run_dir, which, args.lines, meta))
                break
            idle = 0 if live.get("live") else idle + 1
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())