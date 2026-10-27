"""Freeze ``configs/zbo_v1.json`` after the CPU gates and before any GPU run.

This does **not** rerun gates and does **not** change any scientific constant.
It only updates the code hashes / freeze timestamp after the trainer and helper
scripts are final, and re-asserts that ``analysis/cpu_checks.json`` passed.

Usage:
    python .../zbo_freeze.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import zbo_common as BC  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> int:
    gate_path = BC.ANALYSIS / "cpu_checks.json"
    if not gate_path.exists():
        raise FileNotFoundError(f"missing CPU gates {gate_path}")
    checks = BC.read_json(gate_path)
    if not checks.get("all_pass", False):
        raise RuntimeError("CPU gates did not pass; refusing to freeze GPU authorization")
    if not BC.CONFIG.exists():
        raise FileNotFoundError(f"missing base config {BC.CONFIG}; run zbo_checks.py")

    cfg = BC.read_json(BC.CONFIG)
    code_hashes = {p.name: BC.sha256_file(p) for p in sorted(HERE.glob("*.py"))}
    try:
        head = subprocess.check_output(
            ["git", "-C", str(BC.REPO), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(BC.REPO), "status", "--porcelain",
             "--untracked-files=no"], text=True).strip())
    except Exception:
        head, dirty = None, None

    cfg["frozen_utc"] = BC.now()
    cfg["frozen_before_commit"] = True
    cfg["freeze_note"] = (
        "Frozen after analysis/cpu_checks.json passed, before the commit that "
        "contains this config. The formal run's train.json records the clean "
        "git HEAD at launch; this field may point at the parent commit."
    )
    cfg["git_head_at_freeze"] = head
    cfg["git_dirty_at_freeze"] = dirty
    cfg["code_sha256"] = code_hashes
    cfg["cpu_gates"]["sha256"] = BC.sha256_file(gate_path)
    cfg["cpu_gates"]["all_pass"] = True
    cfg["cpu_gates"]["failed_checks"] = []
    cfg["stop_gates"]["gpu_authorized"] = True
    cfg["stop_gates"]["reason_if_stopped"] = None
    BC.write_json(BC.CONFIG, cfg)
    print(f"[freeze] wrote {BC.rel_or_abs(BC.CONFIG)}")
    print(f"[freeze] code files: {len(code_hashes)}")
    print(f"[freeze] gate sha256: {cfg['cpu_gates']['sha256']}")
    print(f"[freeze] git_head_at_freeze: {head} dirty={dirty}")
    print("[freeze] GPU authorized; commit this state before formal runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
