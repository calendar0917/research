"""Dirty-code provenance: untracked source must enter the run's code identity.

The most dangerous high-frequency AI research situation is a *new source file
that was never git-added* when an experiment runs.  It must change the code
identity (``code_state_hash``), therefore the run fingerprint, therefore
duplicate detection.
"""

from __future__ import annotations

from ksvd_research.runtime.git_state import (
    GitState,
    UntrackedFile,
    capture_git_state,
    code_state_hash_from_git_dict,
    compute_code_state_hash,
    write_untracked_snapshots,
)
from ksvd_research.runtime.paths import REPO_ROOT
from ksvd_research.runtime.run_store import run_fingerprint


def _state(commit: str = "c0ffee", diff_hash: str = "", untracked=()) -> GitState:
    return GitState(
        commit=commit,
        dirty=bool(diff_hash or untracked),
        diff_hash=diff_hash,
        patch="",
        untracked=tuple(untracked),
        code_state_hash=compute_code_state_hash(commit, diff_hash, untracked),
    )


def _u(path: str, sha: str, size: int = 1, snapshot: bool = True) -> UntrackedFile:
    return UntrackedFile(path=path, sha256=sha, size=size, snapshot=snapshot)


def test_untracked_content_affects_code_state_hash():
    left = _state(untracked=(_u("foo.py", "sha-A"),))
    right = _state(untracked=(_u("foo.py", "sha-B"),))
    assert left.code_state_hash != right.code_state_hash


def test_untracked_code_state_is_deterministic():
    a = _state(untracked=(_u("b.py", "s2"), _u("a.py", "s1")))
    b = _state(untracked=(_u("a.py", "s1"), _u("b.py", "s2")))
    assert a.code_state_hash == b.code_state_hash
    assert a.code_state_hash == compute_code_state_hash("c0ffee", "", [_u("b.py", "s2"), _u("a.py", "s1")])


def test_tracked_diff_changes_code_state_hash():
    clean = _state(commit="c0ffee", diff_hash="")
    dirty = _state(commit="c0ffee", diff_hash="d1d2d3")
    assert clean.code_state_hash != dirty.code_state_hash


def test_legacy_string_untracked_entries_still_deterministic():
    git = {"commit": "abc", "diff_hash": "", "untracked": ("b.py", "a.py")}
    assert code_state_hash_from_git_dict(git) == code_state_hash_from_git_dict(
        {"commit": "abc", "diff_hash": "", "untracked": ("a.py", "b.py")}
    )
    assert code_state_hash_from_git_dict(git) != code_state_hash_from_git_dict(
        {"commit": "abc", "diff_hash": "", "untracked": ("a.py", "c.py")}
    )


def test_run_fingerprint_changes_when_untracked_source_changes():
    base = dict(
        runner="r",
        study_id="s",
        protocol_id="p",
        candidate_id="c",
        config={"a": 1},
        seeds=[0],
        dataset_fingerprint={"dataset_fingerprint": "d"},
        split_fingerprint="sp",
    )
    git_a = _state(untracked=(_u("foo.py", "sha-A", 3),)).to_dict()
    git_b = _state(untracked=(_u("foo.py", "sha-B", 3),)).to_dict()
    fp_a = run_fingerprint(**base, git=git_a)
    fp_b = run_fingerprint(**base, git=git_b)
    assert fp_a != fp_b
    # stable: same untracked content gives the same fingerprint
    assert fp_a == run_fingerprint(**base, git=git_a)


def test_capture_git_state_includes_untracked_hash_and_snapshot():
    probe = REPO_ROOT / f"tmp-ksvd-probe-{__import__('secrets').token_hex(4)}.py"
    try:
        probe.write_text("value_a = 1\n", encoding="utf-8")
        state = capture_git_state()
        matches = [entry for entry in state.untracked if entry.path == probe.name]
        assert matches, "probe file should be reported as untracked"
        entry = matches[0]
        assert entry.size == probe.stat().st_size
        assert entry.sha256
        assert entry.snapshot is True
        assert state.code_state_hash == code_state_hash_from_git_dict(state.to_dict())
    finally:
        probe.unlink(missing_ok=True)


def test_write_untracked_snapshots_stays_safe(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    probe = REPO_ROOT / f"tmp-ksvd-probe-{__import__('secrets').token_hex(4)}.py"
    try:
        probe.write_text("x = 2\n", encoding="utf-8")
        entry = UntrackedFile(path=probe.name, sha256="ignored", size=1, snapshot=True)
        written = write_untracked_snapshots(run_dir, [entry])
        assert len(written) == 1
        assert (run_dir / "git.untracked" / probe.name).read_text(encoding="utf-8") == "x = 2\n"
    finally:
        probe.unlink(missing_ok=True)
    traversal = UntrackedFile(path="../../outside.txt", sha256="x", size=1, snapshot=True)
    assert write_untracked_snapshots(run_dir, [traversal]) == []


def test_large_or_binary_untracked_files_never_snapshotted():
    entry = UntrackedFile(path="data/cache.bin", sha256="sha", size=50, snapshot=False)
    state = _state(untracked=(entry,))
    assert state.code_state_hash
