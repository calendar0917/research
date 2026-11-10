"""Evaluation integrity, checkpoint round-trip and resume semantics."""

import numpy as np
import pytest
import torch

import rd_common as C
import rd_model as M
import rd_train as T

pytestmark = pytest.mark.slow


def _tiny_train_args(task, run_key="W3_WD10", epochs=1):
    return dict(run_key=run_key, seed=C.SEED, epochs=epochs, require_frozen_plan=False,
                soft_seconds=600.0, hard_seconds=900.0)


def test_evaluation_preserves_model_optimizer_and_rng(tiny_task, tiny_splits):
    task = tiny_task
    model, _audit = M.build_model("W3_WD10", task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
    opt, _ = M.make_optimizer(model, weight_decay=C.RUNS["W3_WD10"]["weight_decay"])
    C.seed_everything(C.SEED)
    rows = np.asarray(tiny_splits["train"].rows[:32], dtype=np.int64)
    batch = C.PC.collate(task.cache, [int(r) for r in rows])
    y = torch.as_tensor(task.labels[rows], dtype=torch.float32)
    C.BT.train_step(model, opt, batch, y, global_step=0, clip=5.0)

    before_params = C.state_fingerprint(M.BM.state_cpu(model))
    before_opt = T.optimizer_fingerprint(opt)
    before_grads = T.gradient_fingerprint(model)
    before_rng = C.rng_state_hashes(C.capture_rng_state())["combined_sha256"]

    out = T.evaluate_all_splits(
        model, opt, task, tiny_splits, torch.device("cpu"), epoch=1, global_step=1, learning_rate=3e-4,
        batch_sizes={"train": 64, "valid": 64, "test": 64},
    )
    assert out["integrity"]["passed"] is True
    assert out["integrity"]["split_order"] == ["train", "valid", "test"]
    assert C.state_fingerprint(M.BM.state_cpu(model)) == before_params
    assert T.optimizer_fingerprint(opt) == before_opt
    assert T.gradient_fingerprint(model) == before_grads
    assert C.rng_state_hashes(C.capture_rng_state())["combined_sha256"] == before_rng
    for name in C.SPLITS:
        assert out["splits"][name]["rows"] == int(tiny_splits[name].rows.size)


def test_checkpoint_roundtrip_restores_model_optimizer_and_rng(tiny_task, tiny_splits, tiny_plan, tmp_path):
    task = tiny_task
    model, audit = M.build_model("W3_WD10", task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
    opt, opt_audit = M.make_optimizer(model, weight_decay=C.RUNS["W3_WD10"]["weight_decay"])
    C.seed_everything(C.SEED)
    rows = np.asarray(tiny_splits["train"].rows[:32], dtype=np.int64)
    batch = C.PC.collate(task.cache, [int(r) for r in rows])
    y = torch.as_tensor(task.labels[rows], dtype=torch.float32)
    C.BT.train_step(model, opt, batch, y, global_step=0, clip=5.0)

    payload = T.checkpoint_payload(
        run_key="W3_WD10", epoch=1, global_step=1, model=model, optimizer=opt,
        rng_state=C.capture_rng_state(), plan=T.plan_meta(tiny_plan, 1, C.SEED),
        model_audit=audit, optimizer_audit=opt_audit,
        scores={"valid_ap": 0.25}, kind="last_epoch_resumable",
    )
    path = tmp_path / "checkpoint_last.pt"
    sha = T.atomic_save(payload, path)
    assert sha == C.sha256_file(path)

    model2, _ = M.build_model("W3_WD10", task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
    opt2, _ = M.make_optimizer(model2, weight_decay=C.RUNS["W3_WD10"]["weight_decay"])
    C.seed_everything(12345)  # deliberately different RNG before restoring
    restored = T.restore_state(
        torch.load(path, map_location="cpu", weights_only=False), model2, opt2,
        run_key="W3_WD10", plan=tiny_plan, order_seed=C.SEED,
    )
    assert restored["resumed_from_epoch"] == 1
    assert restored["plan_prefix_verified"] is True
    assert C.state_fingerprint(M.BM.state_cpu(model2)) == C.state_fingerprint(M.BM.state_cpu(model))
    assert T.optimizer_fingerprint(opt2) == T.optimizer_fingerprint(opt)
    assert C.rng_state_hashes(C.capture_rng_state())["combined_sha256"] == payload["rng_state_hashes"]["combined_sha256"]
    # a checkpoint from the other run must be refused
    with pytest.raises(RuntimeError):
        T.restore_state(
            torch.load(path, map_location="cpu", weights_only=False), model2, opt2,
            run_key="W3_DO20", plan=tiny_plan, order_seed=C.SEED,
        )


def test_best_valid_tie_takes_the_earlier_epoch(tmp_path):
    metrics = []
    for epoch, ap in ((1, 0.30), (2, 0.35), (3, 0.35), (4, 0.34)):
        for split in C.SPLITS:
            metrics.append({"epoch": epoch, "split": split, "ap_official": ap if split == "valid" else 0.0})
    strata = ["support_ge20"] * C.N_TASKS
    T.write_csv(tmp_path / "epoch_metrics.csv", T.EPOCH_FIELDS, [
        {f: r.get(f, "") for f in T.EPOCH_FIELDS} for r in metrics
    ])
    T.write_csv(tmp_path / "epoch_log.csv", T.LOG_FIELDS, [
        {f: ({"epoch": e, "is_best_epoch": False}.get(f, 0)) for f in T.LOG_FIELDS} for e in (1, 2, 3, 4)
    ])
    T.write_csv(tmp_path / "per_task_ap.csv", T.TASK_FIELDS, [
        {
            "epoch": e,
            "split": s,
            "task": t,
            "ap": "",
            "observed": 0,
            "positive": 0,
            "negative": 0,
            "train_positive": 0,
            "valid_positive": 0,
            "stratum": strata[t],
        }
        for e in (1, 2, 3, 4)
        for s in C.SPLITS
        for t in range(C.N_TASKS)
    ])
    completed = T.load_completed(tmp_path, upto_epoch=4, strata=strata)
    assert completed["best_epoch"] == 2
    assert completed["best_valid_ap"] == pytest.approx(0.35)


def test_resume_keeps_completed_rows_and_does_not_retrain(tiny_task, tiny_splits, tmp_path):
    task = tiny_task
    args = _tiny_train_args(task, epochs=1)
    run1 = T.train(task, tiny_splits, out_dir=tmp_path, device=torch.device("cpu"), **args)
    assert run1["status"] == "COMPLETED"
    assert run1["epochs_run"] == 1
    first_metrics = T.read_csv(tmp_path / "epoch_metrics.csv", T.EPOCH_FIELDS)
    first_tasks = T.read_csv(tmp_path / "per_task_ap.csv", T.TASK_FIELDS)
    assert len(first_metrics) == len(C.SPLITS)
    assert len(first_tasks) == len(C.SPLITS) * C.N_TASKS
    epoch1_ap = {r["split"]: r["ap_official"] for r in first_metrics}
    ckpt = torch.load(tmp_path / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 1 and ckpt["global_step"] == 2

    # resume with a longer horizon: epoch 1 must not be retrained nor duplicated
    run2 = T.train(
        task, tiny_splits, out_dir=tmp_path, device=torch.device("cpu"),
        resume=ckpt, **_tiny_train_args(task, epochs=2)
    )
    assert run2["status"] == "COMPLETED"
    assert run2["epochs_run"] == 2
    assert run2["eval_epochs_completed"] == [1, 2]
    metrics = T.read_csv(tmp_path / "epoch_metrics.csv", T.EPOCH_FIELDS)
    tasks = T.read_csv(tmp_path / "per_task_ap.csv", T.TASK_FIELDS)
    assert len(metrics) == 2 * len(C.SPLITS)
    assert len(tasks) == 2 * len(C.SPLITS) * C.N_TASKS
    epoch1_after = {r["split"]: r["ap_official"] for r in metrics if int(float(r["epoch"])) == 1}
    assert epoch1_after == epoch1_ap
    assert run2["resume_audit"]["resumed_from_epoch"] == 1
    assert run2["best_epoch"] in (1, 2)
