"""Training-protocol contracts: plan identity, selection rule, checkpoint resume.

No training is run here. The tests pin the parts that must not drift between
the two arms and the historical W2 reference: the batch plan hashes, the
best-Validation selection rule, and checkpoint round-tripping.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import bs_model as BSM
import s3_common as C
import s3_model as M
import s3_train as T


# ---------------------------------------------------------------------------
# frozen reference identity
# ---------------------------------------------------------------------------


def _historical_run() -> dict:
    if not C.W2_RUN_JSON.is_file():
        pytest.skip("historical W2 run is not present in this checkout")
    return json.loads(C.W2_RUN_JSON.read_text(encoding="utf-8"))


def test_w2_reference_matches_the_historical_run():
    run = _historical_run()
    assert int(run["best_epoch"]) == C.W2["best_epoch"] == 24
    assert float(run["best_valid_ap"]) == pytest.approx(C.W2["best_valid_ap"], abs=0)
    assert C.W2["best_checkpoint_sha256"] == run["best_checkpoint_sha256"]
    assert C.W2["best_valid_ap"] == pytest.approx(0.26527331610028143, abs=0)
    best = run["fixed_epoch_reports"][str(C.W2["best_epoch"])]
    assert float(best["ap"]) == pytest.approx(C.W2["best_valid_ap"], abs=0)
    e40 = run["fixed_epoch_reports"][str(run["final_epoch"])]
    assert float(e40["ap"]) == pytest.approx(C.W2["ap40"], abs=0)
    assert C.W2["ap40"] == pytest.approx(0.26186691813350904, abs=0)
    assert run["batch_plan"]["global_plan_sha256"] == C.W2["global_plan_sha256"]
    assert run["batch_plan"]["prefix_sha256"] == C.W2["consumed40_prefix_sha256"]
    assert C.W2["global_plan_sha256"] == "06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6"


@pytest.mark.slow
def test_batch_plan_hash_is_the_historical_plan():
    task = C.load_task_with_s3()
    plan = C.BC.SD.make_plan40(task, seed=int(C.SEED))
    meta = T.plan_meta(plan, C.TOTAL_EPOCHS, int(C.SEED))
    assert meta["global_plan_sha256"] == C.W2["global_plan_sha256"]
    assert meta["prefix_sha256"] == C.W2["consumed40_prefix_sha256"]
    assert int(meta["epochs"]) == 40
    assert int(meta["batch_size"]) == 128
    assert int(C.STEPS_PER_EPOCH) * 40 == int(C.TOTAL_STEPS_40)


# ---------------------------------------------------------------------------
# the frozen selection rule
# ---------------------------------------------------------------------------


def test_best_valid_selection_rule():
    # strictly better AP wins, at any later epoch
    assert C.better_candidate(0.266, 0.265, 25, 24)
    # equal AP at a later epoch does nothing: the earliest epoch wins ties
    assert not C.better_candidate(0.265, 0.265, 30, 24)
    # strictly worse never wins, even at an earlier epoch
    assert not C.better_candidate(0.264, 0.265, 2, 24)
    # the first evaluated epoch always becomes the incumbent
    assert C.better_candidate(0.1, None, 1, None)
    # a small float difference is a real improvement (no tolerance band)
    assert C.better_candidate(0.26527332, 0.26527331, 40, 39)


def test_selection_is_read_from_the_run_record():
    import csv

    run = _historical_run()
    if not C.W2_CURVE.is_file():
        pytest.skip("historical W2 curve is not present in this checkout")
    with open(C.W2_CURVE, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    curve = {int(r["epoch"]): float(r["valid_ap"]) for r in rows if r.get("valid_ap")}
    if not curve:
        pytest.skip("historical run has no fixed-epoch curve")
    best_epoch = None
    best_ap = None
    for epoch in sorted(curve):
        if C.better_candidate(curve[epoch], best_ap, epoch, best_epoch):
            best_epoch, best_ap = epoch, curve[epoch]
    assert best_epoch == int(run["best_epoch"])
    assert best_ap == pytest.approx(float(run["best_valid_ap"]), abs=1e-12)


# ---------------------------------------------------------------------------
# checkpoint round-trip
# ---------------------------------------------------------------------------


def _fake_plan(n_epochs: int = 2) -> list[dict]:
    return [
        {
            "epoch": e,
            "row_order_sha256": C.sha256_bytes(f"epoch-{e}".encode()),
            "global_plan_sha256": C.sha256_bytes(b"plan"),
            "batches": [],
        }
        for e in range(1, n_epochs + 1)
    ]


def test_checkpoint_roundtrip_restores_state_optimizer_and_rng(spec, norm):
    C.seed_everything(int(C.SEED))
    model, audit = M.build_model("F", spec, norm, C.N_TASKS, seed=0, device="cpu")
    optimizer, _ = BSM.make_optimizer(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    loss = sum(p.float().pow(2).sum() for p in model.parameters())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    plan = _fake_plan()
    rng_before = C.capture_rng_state()
    payload = T.checkpoint_payload(
        model_key="F",
        epoch=2,
        global_step=1,
        model=model,
        optimizer=optimizer,
        rng_state=rng_before,
        plan=T.plan_meta(plan, 2, int(C.SEED)),
        model_audit=audit,
        valid_ap=0.25,
        valid_report=None,
        kind="unit_test",
    )
    assert payload["schema"] == C.SCHEMA_CHECKPOINT
    assert payload["test_evaluated"] is False
    fresh, _ = M.build_model("F", spec, norm, C.N_TASKS, seed=0, device="cpu")
    fresh_opt, _ = BSM.make_optimizer(fresh)
    info = T.restore(payload, fresh, fresh_opt, model_key="F")
    assert info["resumed_from_epoch"] == 2
    assert info["resumed_global_step"] == 1
    state_a = M.state_sha256({k: v.detach().clone() for k, v in model.state_dict().items()})
    state_b = M.state_sha256({k: v.detach().clone() for k, v in fresh.state_dict().items()})
    assert state_a == state_b
    assert C.rng_state_hashes(C.capture_rng_state()) == payload["rng_state_hashes"]
    # the restored optimizer keeps the step count, so the next AdamW update matches
    for group_a, group_b in zip(optimizer.param_groups, fresh_opt.param_groups):
        assert group_a["lr"] == group_b["lr"]


def test_restore_rejects_a_foreign_checkpoint(spec, norm):
    payload = {"schema": C.SCHEMA_CHECKPOINT, "round_id": C.ROUND_ID, "model": "S3", "global_step": 0}
    model, _ = M.build_model("F", spec, norm, C.N_TASKS, seed=0, device="cpu")
    opt, _ = BSM.make_optimizer(model)
    with pytest.raises(RuntimeError, match="model"):
        T.restore(payload, model, opt, model_key="F")


# ---------------------------------------------------------------------------
# losses and bookkeeping helpers
# ---------------------------------------------------------------------------


def test_masked_bce_matches_the_manual_formula():
    logits = torch.tensor([[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]])
    labels = torch.tensor([[1.0, 0.0, float("nan")], [0.0, 1.0, float("nan")]])
    loss = C.BC.MET.masked_bce_loss(logits, labels)
    def bce(z: float, t: float) -> float:
        return max(z, 0.0) - z * t + float(np.log1p(np.exp(-abs(z))))

    manual = (bce(0.0, 1.0) + bce(1.0, 0.0) + bce(2.0, 0.0) + bce(-2.0, 1.0)) / 4.0
    assert float(loss) == pytest.approx(float(manual), rel=1e-6)
    # a wholly unobserved row contributes nothing and does not produce NaN
    all_nan = torch.full((1, 3), float("nan"))
    loss2 = C.BC.MET.masked_bce_loss(torch.zeros(1, 3), all_nan)
    assert float(loss2) == 0.0
    assert torch.isfinite(loss2)


def test_collate_s3_offsets_paths_per_row():
    rows = [5, 9]
    row_index = np.full(10, -1, dtype=np.int64)
    row_index[5] = 0
    row_index[9] = 3
    s3 = {
        "row_index": row_index,
        "rows": np.asarray([5, 9], dtype=np.int64),
        "row_atom_ptr": np.asarray([0, 3, 3, 3, 4], dtype=np.int64),
        "s3_row_atoms": np.asarray([3, 0, 0, 1], dtype=np.int64),
        "s3_row_paths": np.asarray([2, 1, 0, 1], dtype=np.int64),
        "s3_atom_ptr": np.asarray([0, 2, 3, 3, 4], dtype=np.int64),
        "s3_end": np.asarray([2, 1, 0, 2], dtype=np.int64),
        "s3_mid1_type": np.asarray([1, 2, 3, 4], dtype=np.int64),
        "s3_mid2_type": np.asarray([5, 6, 7, 8], dtype=np.int64),
        "s3_bond1": np.asarray([1, 2, 3, 4], dtype=np.int64),
        "s3_bond2": np.asarray([2, 3, 4, 1], dtype=np.int64),
        "s3_bond3": np.asarray([3, 4, 1, 2], dtype=np.int64),
        "s3_dist": np.asarray([3, 2, 1, 3], dtype=np.int64),
    }
    out = C.collate_s3(s3, rows)
    # row 5 has three paths (rows 0, 1, 2), row 9 has one (row 3)
    # row 5 exercises offsets: its three atoms have 2, 1 and 0 paths
    assert out["s3_path_atom"].shape == (4,)
    assert out["s3_path_atom"].tolist() == [0, 0, 1, 3]
    # endpoints are batch-global atom ids, so the second molecule is offset by 3
    assert out["s3_path_end"].tolist() == [2, 1, 0, 5]
    assert out["atom_counts"].tolist() == [3, 1]
    assert int(out["s3_path_atom"].max()) == 3  # local ids in the concatenated atom block
    assert out["s3_bond1"].tolist() == [1, 2, 3, 4]
    assert out["s3_dist"].tolist() == [3, 2, 1, 3]


def test_batch_isolation_flags_a_cross_molecule_path():
    batch = {
        "atom_batch": torch.tensor([0, 0, 0, 1, 1]),
        "s3_path_atom": torch.tensor([0, 1, 3]),
        "s3_path_end": torch.tensor([1, 0, 4]),
    }
    report = C.check_s3_batch_isolation(batch)
    assert report["n_paths"] == 3
    assert report["isolated"] is True
    bad = dict(batch)
    bad["s3_path_end"] = torch.tensor([1, 3, 4])
    report_bad = C.check_s3_batch_isolation(bad)
    assert report_bad["isolated"] is False
    assert report_bad["cross_molecule"] == 1


def test_s3_scope_is_train_plus_valid_only():
    class FakeTask:
        train_rows = np.asarray([0, 1, 2], dtype=np.int64)
        valid_rows = np.asarray([3, 4], dtype=np.int64)
        test_rows = np.asarray([5, 6], dtype=np.int64)

    rows = C.s3_scope_rows(FakeTask())
    assert rows.tolist() == [0, 1, 2, 3, 4]

    class LeakyTask(FakeTask):
        train_rows = np.asarray([0, 1, 5], dtype=np.int64)

    with pytest.raises(RuntimeError, match="Test"):
        C.s3_scope_rows(LeakyTask())


# ---------------------------------------------------------------------------
# decision rules
# ---------------------------------------------------------------------------


def test_decision_rule_thresholds_are_pre_registered():
    rules = C.DECISION_RULES
    assert rules["promising"]["d_best_s3_min"] == 0.005
    assert rules["promising"]["d_mech_min"] == 0.003
    assert rules["promising"]["mean_delta_support_ge20_min"] == 0.003
    assert rules["promising"]["improved_support_ge20_min"] == 60
    assert rules["promising"]["support_ge20_total"] == 107
    assert rules["no_clear_gain"]["d_best_s3_max"] == 0.002
    assert rules["regression"]["d_best_s3_max"] == -0.002
    assert rules["promising"]["label"] == "PROMISING_S3_SIGNAL"


def _stats(mean_delta: float, improved: int) -> dict:
    return {"mean_delta": mean_delta, "improved": improved, "n_scorable": 107}


def test_decision_labels():
    import s3_compare as K

    promising = K.decide(d_best_s3=0.006, d_best_f=0.001, d_mech=0.004, task_stats_ge20=_stats(0.004, 70))
    assert promising["label"] == "PROMISING_S3_SIGNAL"
    # one missing condition is enough to leave the promising branch
    weak = K.decide(d_best_s3=0.006, d_best_f=0.001, d_mech=0.002, task_stats_ge20=_stats(0.004, 70))
    assert weak["label"] != "PROMISING_S3_SIGNAL"
    narrow = K.decide(d_best_s3=0.006, d_best_f=0.001, d_mech=0.004, task_stats_ge20=_stats(0.001, 70))
    assert narrow["label"] != "PROMISING_S3_SIGNAL"
    regression = K.decide(d_best_s3=-0.003, d_best_f=0.001, d_mech=-0.002, task_stats_ge20=_stats(-0.001, 40))
    assert regression["label"] == "REGRESSION"
    none = K.decide(d_best_s3=0.0005, d_best_f=0.0004, d_mech=0.0001, task_stats_ge20=_stats(0.0002, 55))
    assert none["label"] == "NO_CLEAR_GAIN"
    capacity = K.decide(d_best_s3=0.004, d_best_f=0.0035, d_mech=0.0005, task_stats_ge20=_stats(0.001, 60))
    assert capacity["label"] == "CAPACITY_ONLY"
    # S3 beats W2 but the FFN control is close: an ordinary capacity effect
    capacity2 = K.decide(d_best_s3=0.003, d_best_f=0.0001, d_mech=0.0029, task_stats_ge20=_stats(0.002, 65))
    assert capacity2["label"] == "CAPACITY_ONLY"
    # S3 has a material edge over the FFN but the high-support tasks do not improve
    mixed = K.decide(d_best_s3=0.003, d_best_f=0.0001, d_mech=0.0035, task_stats_ge20=_stats(0.002, 65))
    assert mixed["label"] == "WEAK_OR_MIXED"
