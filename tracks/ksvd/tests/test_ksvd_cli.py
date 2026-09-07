"""Unit tests for CLI-level behavior: compare partitions, context output size."""

from __future__ import annotations


from ksvd_research.cli import comparability_key, partition_comparable


def _run(index: str, protocol: str, dataset: str, split: str) -> dict:
    return {
        "run_id": index,
        "status": "completed",
        "protocol_id": protocol,
        "dataset_fingerprint": {"dataset_fingerprint": dataset},
        "split_fingerprint": split,
        "metrics": {"valid_mae": float(len(index))},
    }


def test_comparability_key_requires_triple_match():
    assert comparability_key(_run("a", "p1", "d1", "s1")) == ("p1", "d1", "s1")
    assert comparability_key(_run("a", "p1", "d2", "s1")) != comparability_key(_run("a", "p1", "d1", "s1"))


def test_partition_same_group_is_comparable():
    groups, compatible = partition_comparable(
        [_run("a", "zinc-context-gap", "d", "s"), _run("b", "zinc-context-gap", "d", "s")]
    )
    assert compatible and list(groups.keys()) == [("zinc-context-gap", "d", "s")]


def test_partition_mismatch_is_incomparable():
    manifests = [
        _run("a", "zinc-context-gap", "d1", "s"),
        _run("b", "zinc-context-gap", "d2", "s"),
    ]
    groups, compatible = partition_comparable(manifests)
    assert not compatible and len(groups) == 2


def test_partition_ignores_mode_and_seed_for_comparability():
    left = _run("a", "p", "d", "s")
    right = _run("b", "p", "d", "s")
    right["mode"] = "terminal"
    right["seeds"] = [0, 1]
    groups, compatible = partition_comparable([left, right])
    assert compatible


def test_context_output_is_bounded(capsys):
    from ksvd_research.cli import _cmd_context
    from argparse import Namespace

    assert _cmd_context(Namespace()) == 0
    output = capsys.readouterr().out
    line_count = len(output.splitlines())
    assert 40 <= line_count <= 260, f"context has {line_count} lines"
    assert "zinc-context-gap" in output
