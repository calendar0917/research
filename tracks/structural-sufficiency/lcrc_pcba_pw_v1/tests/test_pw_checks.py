"""The eleven pre-training checks, plus the inherited training-parity assertions."""

from __future__ import annotations

import numpy as np
import torch

import pw_common as C
import pw_checks as CHK
import pw_model as M
from conftest import TINY_GRAPHS, synthetic_batch


def test_check_init(spec, norm):
    result = CHK.check_init(spec, norm, device="cpu")
    assert result["passed"], result["details"]
    assert result["details"]["total_parameters"] == C.EXPECTED_PARAMETERS


def test_check_zero_init_forward(spec, norm, tiny_batch):
    result = CHK.check_zero_init_forward(spec, norm, tiny_batch, device="cpu")
    assert result["passed"], result["details"]


def test_check_permutation(spec, norm, tiny_batch):
    result = CHK.check_permutation(spec, norm, tiny_batch, device="cpu")
    assert result["passed"], result["details"]


def test_check_pair_symmetry(spec, norm, tiny_batch):
    result = CHK.check_pair_symmetry(spec, norm, tiny_batch, device="cpu")
    assert result["passed"], result["details"]
    assert result["details"]["token_bitwise_equal"] is True
    assert result["details"]["residual_max_abs_after_tail_perturbation"] > 0.0


def test_check_no_cross_graph(spec, norm, task, pair_cache, pair_rows):
    batch = __import__("pw_pairs").collate_pw(task, pair_cache, pair_rows[:16])
    result = CHK.check_no_cross_graph(spec, norm, batch, task, pair_cache, pair_rows[:16], device="cpu")
    assert result["passed"], result["details"]


def test_check_pair_index(pair_cache):
    result = CHK.check_pair_index(pair_cache, n_sample=12)
    assert result["passed"], result["details"]
    assert result["details"]["real_cache_sample"]["rows_mismatched"] == 0


def test_check_segment_softmax(spec, norm, tiny_batch):
    result = CHK.check_segment_softmax(spec, norm, tiny_batch, device="cpu")
    assert result["passed"], result["details"]


def test_check_gradient_path(spec, norm, real_batch):
    result = CHK.check_gradient_path(spec, norm, [real_batch], steps=3, device="cpu")
    assert result["passed"], result["details"]


def test_check_cpu_vs_batched(spec, norm, tiny_batch):
    result = CHK.check_cpu_vs_batched(spec, norm, tiny_batch, device="cpu")
    assert result["passed"], result["details"]
    for key in ("token", "attention", "z", "residual"):
        assert result["details"][key]["relative"] <= 1e-5


def test_check_no_pairs(spec, norm, tiny_no_pair_batch):
    result = CHK.check_no_pairs(spec, norm, tiny_no_pair_batch, device="cpu")
    assert result["passed"], result["details"]


def test_check_metrics_and_isolation(task):
    result = CHK.check_metrics_and_isolation(task)
    assert result["passed"], result["details"]
    assert result["details"]["global_plan_sha256"] == C.W2["global_plan_sha256"]


def test_run_all_cpu(spec, norm, task, pair_cache, pair_rows, no_pair_batch):
    import pw_pairs as D

    batch = D.collate_pw(task, pair_cache, pair_rows[:32])
    record = CHK.run_all(
        spec,
        norm,
        batch,
        pair_rows[:32],
        task=task,
        pairs=pair_cache,
        no_pair_batch=no_pair_batch,
        device="cpu",
        grad_batches=[batch],
    )
    assert record["all_passed"], [
        (c["name"], c["details"]) for c in record["checks"] if not c["passed"]
    ]
    assert record["n_checks"] == 11


# ---------------------------------------------------------------------------
# inherited recipe / data parity
# ---------------------------------------------------------------------------


def test_recipe_is_the_w2_recipe():
    recipe = C.recipe_for_model()
    assert recipe["epochs"] == 40
    assert recipe["batch_size"] == 128
    assert recipe["seed"] == 0
    assert float(recipe["gradient_clip"]) == 5.0
    assert float(recipe["dropout"]) == 0.10
    assert recipe["lr_schedule"]["steps_per_epoch"] == 2738
    assert recipe["lr_schedule"]["total_steps_40_epochs"] == 109_520
    assert recipe["branch"]["pair_layers"] == 1
    assert recipe["no_message_passing"] is True


def test_order_plan_is_the_frozen_w2_plan(task):
    plan = C.BC.SD.make_plan40(task, seed=C.SEED)
    assert plan[0]["global_plan_sha256"] == C.W2["global_plan_sha256"]


def test_test_labels_absent_from_the_training_artifact(task):
    test_rows = np.asarray(task.test_rows, dtype=np.int64)
    assert not np.isfinite(task.labels[test_rows]).any()


def test_metrics_parity():
    import pc_metrics as MET

    gen = torch.Generator().manual_seed(5)
    logits = torch.randn(32, C.N_TASKS, generator=gen)
    y = torch.randint(0, 2, (32, C.N_TASKS), generator=gen).float()
    loss = MET.masked_bce_loss(logits, y)
    manual = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    assert float((loss - manual).abs()) == 0.0


def test_synthetic_single_atom_graph_is_finite(spec, norm):
    batch = synthetic_batch(graphs=[TINY_GRAPHS["single_atom"]], seed=9)
    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    model.eval()
    with torch.no_grad():
        logits = model(batch)
    assert torch.isfinite(logits).all()
    assert int(batch["pair_i"].numel()) == 0


def test_no_pair_sampling_or_bucketing_hidden_in_the_model():
    """The branch must expose its whole pair list; nothing is subsampled."""
    source = (C.CODE_DIR / "pw_model.py").read_text(encoding="utf-8")
    for banned in ("topk(", "argsort(", "randperm(", "multinomial(", "sample("):
        assert banned not in source, banned
