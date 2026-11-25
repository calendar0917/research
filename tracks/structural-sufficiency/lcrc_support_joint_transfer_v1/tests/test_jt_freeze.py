from __future__ import annotations

import json

import numpy as np
import pytest

import jt_common as C
import jt_support as S


def test_frozen_config_matches_payload():
    frozen = C.read_json(C.ANALYSIS / "protocol_freeze.json")
    assert frozen["schema"] == "lcrc-support-joint-transfer-freeze-v1"
    assert frozen["config_payload"] == C.config_payload()
    assert frozen["config_sha256"] == C.sha256_file(C.CONFIGS / "experiment.json")
    payload = {k: v for k, v in frozen.items() if k != "freeze_hash"}
    assert frozen["freeze_hash"] == C.sha256_text(json.dumps(payload, sort_keys=True, default=str))


def test_lipo_support_is_train_only_and_low_rank():
    support = S.load_support(C.ANALYSIS / "support/lipo_support.npz")
    assert int(support.n_omega) == 108
    assert int(support.rank) == 48
    assert int(support.n_components) == 4
    assert int(support.n_rectangles) == 441
    assert support.legality["all_omega_legal"] is True
    assert np.allclose(support.P, support.P.T, atol=1e-12)
    assert np.allclose(support.P @ support.P, support.P, atol=1e-12)
    assert support.coverage["train"]["molecule_fraction_with_any_seen_pair"] == 1.0


def test_pcba_support_is_train_only_and_low_rank():
    support = S.load_support(C.ANALYSIS / "support/pcba_support.npz")
    assert int(support.n_omega) == 214
    assert int(support.rank) == 67
    assert int(support.n_components) == 4
    assert int(support.n_rectangles) == 2585
    assert support.legality["all_omega_legal"] is True
    assert np.allclose(support.P, support.P.T, atol=1e-12)
    assert support.coverage["train"]["molecule_fraction_with_any_seen_pair"] > 0.999
