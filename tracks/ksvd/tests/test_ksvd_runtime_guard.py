"""Control-plane loading: studies, protocols, claims, decisions.

Test-access policy semantics live in ``test_ksvd_policy.py``; this file
covers control-plane file loading and record primitives.
"""

from __future__ import annotations

import pytest

from ksvd_research.runtime.control import load_protocol, load_study, protocol_for_study
from ksvd_research.runtime.records import new_claim, new_decision, set_claim_status


# ---------------------------------------------------------------------------
# real control-plane files
# ---------------------------------------------------------------------------


def test_active_study_protocols_load():
    for study_id in ("zinc-context-gap", "molhiv-cross-scaffold-interaction"):
        study = load_study(study_id)
        assert study["active"] is True
        protocol = protocol_for_study(study_id)
        assert protocol["id"] == study_id
        assert protocol["metric"]["metric_direction"] in (
            "higher_is_better",
            "lower_is_better",
        )
        assert protocol["seed_semantics"]["seeds"]


def test_protocol_has_no_architecture():
    for protocol_id in ("zinc-context-gap", "molhiv-cross-scaffold-interaction"):
        protocol = load_protocol(protocol_id)
        text = str(protocol).lower()
        assert "architecture" not in [key.lower() for key in protocol]
        assert "model" not in [key.lower() for key in protocol.keys()]
        assert "patch_hidden" not in text and "embedding" not in text


# ---------------------------------------------------------------------------
# claims / decisions
# ---------------------------------------------------------------------------


def test_claim_and_decision_records(tmp_path, monkeypatch):
    import ksvd_research.runtime.records as records

    monkeypatch.setattr(records, "records_root", lambda: tmp_path)
    claim = new_claim(
        statement="ZINC patch-path pooling preserves exact patch identity",
        study_id="zinc-context-gap",
        status="tentative",
        scope="zinc valid only",
        supporting_evidence=["run-a"],
        limitations=["single seed"],
        implication="token family stays relevant",
    )
    assert claim["claim_id"].startswith("claim-")
    updated = set_claim_status(claim["claim_id"], "supported")
    assert updated["status"] == "supported"
    with pytest.raises(ValueError):
        new_claim(statement="x", status="purple")

    decision = new_decision(
        decision="use terminal-only test policy",
        because="protocol requires it",
        alternatives_rejected=["unrestricted test access"],
        revisit_if="protocol changes",
        study_id="zinc-context-gap",
    )
    assert decision["decision_id"].startswith("decision-")
    with pytest.raises(ValueError):
        new_decision(decision="d", because="")
