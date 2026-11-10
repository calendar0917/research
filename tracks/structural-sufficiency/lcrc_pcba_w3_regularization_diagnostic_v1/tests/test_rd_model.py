"""Both runs are the historical W3 with exactly one regularization change each."""

import numpy as np
import pytest
import torch

import rd_common as C
import rd_model as M

pytestmark = pytest.mark.slow


def test_parameter_counts_are_exactly_the_historical_w3(models):
    for key in C.RUN_ORDER:
        model, audit = models[key]
        total = int(sum(p.numel() for p in model.parameters()))
        trunk = int(sum(p.numel() for n, p in model.named_parameters() if not n.startswith("readout.")))
        readout = int(sum(p.numel() for n, p in model.named_parameters() if n.startswith("readout.")))
        assert total == C.EXPECTED_PARAMETERS == 2120844
        assert trunk == C.EXPECTED_BASE_PARAMETERS == 1809408
        assert readout == C.EXPECTED_READOUT_PARAMETERS == 311436
        assert audit["parameters_exact"] is True
        assert audit["width_audit_all_match"] is True


def test_widths_are_the_w3_widths(models):
    expected = {
        "atom_hidden": 384,
        "atom_out": 288,
        "bond_hidden": 480,
        "bond_out": 288,
        "stat_rank": 96,
        "local_hidden": 576,
        "local_width": 192,
        "rho_hidden": 192,
        "scorer_hidden": 96,
        "pool_hidden": 264,
        "n_heads": 4,
        "readout_out": 128,
    }
    for key in C.RUN_ORDER:
        observed = models[key][1]["width_audit"]["observed"]
        assert observed["atom_hidden"]["out"] == expected["atom_hidden"]
        assert observed["atom_out"]["out"] == expected["atom_out"]
        assert observed["bond_hidden"]["out"] == expected["bond_hidden"]
        assert observed["bond_out"]["out"] == expected["bond_out"]
        assert observed["stat_rank"]["nbr_emb"][1] == expected["stat_rank"]
        assert observed["local_hidden"]["out"] == expected["local_hidden"]
        assert observed["local_width"]["out"] == expected["local_width"]
        assert observed["rho_hidden"]["out"] == expected["rho_hidden"]
        assert observed["pool_hidden"]["out"] == expected["pool_hidden"]
        assert observed["readout_out"]["out"] == expected["readout_out"]
        assert all(h["out"] == expected["scorer_hidden"] for h in observed["scorer_hidden"])
        assert observed["n_heads"] == 4


def test_initial_state_is_bitwise_identical(models):
    fps = {k: C.state_fingerprint(M.BM.state_cpu(models[k][0])) for k in C.RUN_ORDER}
    assert len(set(fps.values())) == 1
    a, b = (models[k][0].state_dict() for k in C.RUN_ORDER)
    assert list(a) == list(b)
    for key in a:
        assert a[key].shape == b[key].shape
        assert torch.equal(a[key], b[key])


def test_only_the_two_config_values_differ():
    audit = C.assert_shared_recipe()
    assert audit["ok"]
    assert [C.RUNS[k]["dropout"] for k in C.RUN_ORDER] == [0.10, 0.20]
    assert [C.RUNS[k]["weight_decay"] for k in C.RUN_ORDER] == [0.010, 0.001]


def test_dropout_rate_is_on_the_real_forward_path(models, batch):
    rates = {}
    for key in C.RUN_ORDER:
        model, audit = models[key]
        assert audit["dropout"]["n_dropout_modules"] == 1
        assert audit["dropout"]["module"] == "drop"
        assert float(model.drop.p) == pytest.approx(C.RUNS[key]["dropout"])
        rates[key] = float(model.drop.p)
        probe = M.dropout_probe(model, batch, repeats=8)
        assert probe["passed"], probe
        assert probe["eval_zero_fraction"] == 0.0
        assert probe["p_effective"] == pytest.approx(C.RUNS[key]["dropout"])
    assert rates["W3_WD10"] == pytest.approx(0.10)
    assert rates["W3_DO20"] == pytest.approx(0.20)


def test_set_dropout_refuses_a_second_dropout_module(models):
    model = models["W3_WD10"][0]
    model.drop2 = torch.nn.Dropout(0.1)
    with pytest.raises(RuntimeError):
        M.set_dropout(model, 0.3)
    del model.drop2


def test_eval_mode_forward_is_identical_between_runs(models, batch):
    out = {}
    for key in C.RUN_ORDER:
        model = models[key][0]
        model.eval()
        with torch.no_grad():
            out[key] = model(batch).clone()
    a, b = (out[k] for k in C.RUN_ORDER)
    assert torch.equal(a, b)
    assert float((a - b).abs().max()) == 0.0


def test_train_mode_differs_and_is_stochastic(models, batch):
    probe = M.intervention_probe({k: models[k][0] for k in C.RUN_ORDER}, batch, seed=C.SEED)
    assert probe["passed"], probe
    assert probe["eval_logits_identical"] is True
    assert probe["train_logits_differ"] is True
    for key in C.RUN_ORDER:
        assert M.train_forward_is_stochastic(models[key][0], batch) == {
            "train_forwards_differ": True,
            "eval_forwards_identical": True,
        }


def test_adamw_weight_decay_is_read_back(models):
    for key in C.RUN_ORDER:
        model = models[key][0]
        opt, audit = M.make_optimizer(model, weight_decay=C.RUNS[key]["weight_decay"])
        assert audit["n_param_groups"] == 1
        assert audit["all_params_one_group"] is True
        assert audit["weight_decay_read_back"] == [C.RUNS[key]["weight_decay"]]
        assert opt.param_groups[0]["weight_decay"] == pytest.approx(C.RUNS[key]["weight_decay"])
        assert opt.param_groups[0]["betas"] == (0.9, 0.999)
        assert opt.param_groups[0]["eps"] == pytest.approx(1e-8)
        assert audit["n_parameters"] == C.EXPECTED_PARAMETERS


def test_weight_decay_intervention_changes_the_update(models, torch_batch):
    import copy

    batch, y = torch_batch
    clones = {k: copy.deepcopy(models[k][0]) for k in C.RUN_ORDER}
    probe = M.weight_decay_update_probe(clones, batch, y, steps=1)
    assert probe["passed"], probe
    assert probe["weight_decay_ratio"] == pytest.approx(10.0)
    assert probe["decay_multipliers_differ"] is True


def test_training_order_matches_the_historical_w3_plan(plan, task):
    hist_path = C.BS_RUNS / "W3" / "training_order.json"
    if not hist_path.is_file():  # pragma: no cover - historical runs are pulled locally
        pytest.skip("historical W3 training_order.json is not present on this host")
    hist = C.read_json(hist_path)
    mine = {int(p["epoch"]): p["row_order_sha256"] for p in plan}
    theirs = {int(e["epoch"]): e["row_order_sha256"] for e in hist["summary"]["epochs_detail"]}
    assert mine == theirs
    assert plan[0]["global_plan_sha256"] == C.SHARED_RECIPE["batch_plan_sha256"]
    assert all(len(p["batches"]) == C.STEPS_PER_EPOCH for p in plan)
    assert len(plan) == C.TOTAL_EPOCHS
    isolation = C.SD.plan_summary(plan, task)["isolation"]
    assert isolation["passed"] is True
    assert all(c["valid_rows_in_epoch"] == 0 and c["test_rows_in_epoch"] == 0 for c in isolation["checks"])
