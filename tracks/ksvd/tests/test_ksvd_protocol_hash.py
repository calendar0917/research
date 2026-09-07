"""Protocol content must bind to a run's identity and comparability.

The same ``protocol_id`` with a modified YAML file must not silently look
comparable to an earlier run; the semantic hash catches that.
"""

from __future__ import annotations

import yaml

from ksvd_research.cli import partition_comparable
from ksvd_research.runtime.control import load_protocol, protocol_hash

PROTOCOL_ONE = """
# formatting comments must not matter
id: demo-protocol
metric:
  ranking_metric: valid_mae
  metric_direction: lower_is_better
test_policy: terminal
seed_semantics:
  seeds: [0]
"""

PROTOCOL_ONE_RESTYLED = """
id: demo-protocol

metric:
  metric_direction: lower_is_better
  ranking_metric: valid_mae

# different comments, different whitespace
test_policy: terminal

seed_semantics:
  seeds: [0]
"""

PROTOCOL_TWO = """
id: demo-protocol
metric:
  ranking_metric: valid_mae
  metric_direction: lower_is_better
test_policy: terminal
seed_semantics:
  seeds: [0, 1]
"""


def _parse(text: str) -> dict:
    payload = yaml.safe_load(text)
    assert isinstance(payload, dict)
    return payload


def test_protocol_semantic_hash_is_deterministic():
    parsed_one, parsed_two = _parse(PROTOCOL_ONE), _parse(PROTOCOL_ONE)
    assert protocol_hash(parsed_one) == protocol_hash(parsed_two)
    zinc = load_protocol("zinc-context-gap")
    assert protocol_hash(zinc) == protocol_hash(zinc)
    assert protocol_hash(zinc) != protocol_hash(load_protocol("molhiv-cross-scaffold-interaction"))


def test_protocol_formatting_change_keeps_hash():
    assert protocol_hash(_parse(PROTOCOL_ONE)) == protocol_hash(_parse(PROTOCOL_ONE_RESTYLED))


def test_protocol_scientific_change_changes_hash():
    assert protocol_hash(_parse(PROTOCOL_ONE)) != protocol_hash(_parse(PROTOCOL_TWO))


def test_protocol_hash_participates_in_comparability():
    def manifest(run_id: str, protocol_hash: str) -> dict:
        return {
            "run_id": run_id,
            "status": "completed",
            "protocol_id": "demo-protocol",
            "protocol_hash": protocol_hash,
            "dataset_fingerprint": {"dataset_fingerprint": "d"},
            "split_fingerprint": "s",
            "metrics": {"valid_mae": 0.1},
        }

    h = protocol_hash(_parse(PROTOCOL_ONE))
    groups, compatible_inner = partition_comparable(
        [manifest("a", h), manifest("b", h)]
    )
    assert compatible_inner and len(groups) == 1
    groups_two, compatible = partition_comparable(
        [manifest("a", h), manifest("b", protocol_hash(_parse(PROTOCOL_TWO)))]
    )
    assert not compatible
    assert len(groups_two) == 2
    # old runs without a hash are NOT comparable to new hashed runs
    _, compatible_legacy = partition_comparable(
        [manifest("old", "legacy-unknown"), manifest("new", h)]
    )
    assert not compatible_legacy
    # ...but two legacy-unknown runs remain comparable to each other
    _, compatible_old = partition_comparable(
        [manifest("old", "legacy-unknown"), manifest("older", "legacy-unknown")]
    )
    assert compatible_old
