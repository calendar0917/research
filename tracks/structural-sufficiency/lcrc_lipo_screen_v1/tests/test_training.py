from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import screen_common as C
import screen_data as D
import screen_models as M
import screen_training as T


def _optimizer(model):
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.TOTAL_UPDATES, eta_min=C.LR_MIN)
    return opt, sch


def _step(model, opt, sch, batch):
    target = torch.linspace(-1, 1, int(batch["n_graphs"]), dtype=torch.float32)
    loss = F.mse_loss(model(batch).view(-1), target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
    opt.step()
    sch.step()
    return float(loss)


def test_recipe_constants_match_config():
    cfg = C.load_config()
    assert cfg["training"]["seeds"] == list(C.SEEDS)
    assert cfg["training"]["epochs"] == C.EPOCHS == 360
    assert cfg["training"]["batch_size"] == C.BATCH_SIZE == 128
    assert cfg["training"]["peak_lr"] == C.LR == 1e-3
    assert cfg["training"]["min_lr"] == C.LR_MIN == 1e-5
    assert cfg["training"]["weight_decay"] == C.WEIGHT_DECAY == 1e-5
    assert cfg["training"]["grad_clip"] == C.GRAD_CLIP == 5.0
    assert cfg["training"]["checkpoint"] == "mean_of_last_5_epoch_end_state_dicts"
    assert cfg["training"]["early_stopping"] is False
    assert cfg["evaluation"]["monitor_used_for_selection"] is False
    assert C.STEPS_PER_EPOCH * C.EPOCHS == C.TOTAL_UPDATES


def test_batch_order_is_deterministic_and_epoch_local():
    assert D.batch_order_hash(0) == D.batch_order_hash(0)
    assert D.batch_order_hash(0) != D.batch_order_hash(1)
    batches = D.epoch_batches(0, 1)
    assert len(batches) == C.STEPS_PER_EPOCH
    assert sum(len(b) for b in batches) == C.SPLIT_SIZES["fit"]
    assert len(batches[-1]) == C.SPLIT_SIZES["fit"] - (C.STEPS_PER_EPOCH - 1) * C.BATCH_SIZE
    flat = [r for b in batches for r in b]
    assert len(flat) == len(set(flat)) == C.SPLIT_SIZES["fit"]
    gen = torch.Generator().manual_seed(C.TRAIN_SHUFFLE_OFFSET)
    assert batches == D.shuffle_batches(C.SPLIT_SIZES["fit"], C.BATCH_SIZE, gen)


def test_save_restore_reproduces_next_step(spec, norm, caches, tmp_path):
    batches = D.epoch_batches(0, 1)[:2]
    b1 = D.collate(caches["fit"]["cache"], batches[0])
    b2 = D.collate(caches["fit"]["cache"], batches[1])

    model, _ = M.build_model("A", spec, norm, 0, "cpu")
    opt, sch = _optimizer(model)
    gen = torch.Generator().manual_seed(C.TRAIN_SHUFFLE_OFFSET)
    _step(model, opt, sch, b1)
    T.save_training_state(tmp_path / "state.pt", model, opt, sch, epoch=1, generator=gen)
    _step(model, opt, sch, b2)
    live = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model2, _ = M.build_model("A", spec, norm, 0, "cpu")
    opt2, sch2 = _optimizer(model2)
    state = T.load_training_state(tmp_path / "state.pt", model2, opt2, sch2)
    assert state["epoch"] == 1
    _step(model2, opt2, sch2, b2)
    for key in live:
        assert torch.equal(live[key], model2.state_dict()[key])


def test_target_denormalisation_and_raw_decomposition():
    target_norm = D.load_target_norm(verify=True)
    y = np.asarray([-1.48, 0.0, 2.157857142857143, 4.5], dtype=np.float64)
    z = D.normalised_target(y, [0, 1, 2, 3], target_norm).numpy()
    y_back = D.denormalise(z, target_norm)
    # The trainer stores the standardised target in float32; 1e-6 raw units is
    # well below the measurement scale.
    assert np.allclose(y_back, y, atol=1e-6)
    contrib = np.asarray([-0.5, 0.0, 0.3, 1.1])
    rho = z - contrib
    contrib_raw, rho_raw = D.raw_contribution_terms(contrib, rho, target_norm)
    pred_raw = D.denormalise(contrib + rho, target_norm)
    assert np.allclose(contrib_raw + rho_raw, pred_raw, atol=1e-6)


def test_soup_is_elementwise_mean(spec, norm):
    model, _ = M.build_model("C", spec, norm, 0, "cpu")
    states = []
    for i in range(C.SOUP_LAST):
        states.append({k: v + (i + 1) * 0.01 for k, v in model.state_dict().items()})
    soup = M.average_states(states)
    for key in soup:
        expected = torch.stack([s[key] for s in states]).mean(0)
        assert torch.equal(soup[key], expected)
    identity = M.average_states([model.state_dict()] * C.SOUP_LAST)
    for key in identity:
        # Float32 summation of 5 identical values can differ in the last ulp.
        assert float((identity[key] - model.state_dict()[key].float()).abs().max()) < 1e-6


def test_formal_recipe_cannot_be_overridden():
    T.TrainConfig(arm="A", seed=0, device="cuda").validate()
    bad = [
        dict(epochs=120),
        dict(batch_size=64),
        dict(learning_rate=3e-4),
        dict(min_lr=1e-6),
        dict(weight_decay=1e-4),
        dict(grad_clip=1.0),
        dict(dropout=0.1),
        dict(soup_last=1),
        dict(fit_limit=128),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            T.TrainConfig(arm="A", seed=0, device="cuda", **kwargs).validate()
    with pytest.raises(ValueError):
        T.TrainConfig(arm="A", seed=3, device="cuda").validate()
    with pytest.raises(ValueError):
        T.TrainConfig(arm="D", seed=0, device="cuda").validate()


def test_smoke_mode_is_restricted():
    T.TrainConfig(arm="A", seed=0, device="cpu", smoke=True, fit_limit=256, epochs=12).validate()
    with pytest.raises(PermissionError):
        T.TrainConfig(arm="A", seed=0, device="cuda", smoke=True, fit_limit=256, epochs=12).validate()
    with pytest.raises(ValueError):
        T.TrainConfig(arm="A", seed=0, device="cpu", smoke=True, fit_limit=16, epochs=12).validate()
    with pytest.raises(ValueError):
        T.TrainConfig(arm="A", seed=0, device="cpu", smoke=True, fit_limit=256, epochs=400).validate()


def test_cuda_training_requires_matching_authorization(tmp_path, monkeypatch):
    import json

    import run_screen

    probe = tmp_path / "gpu_authorization.json"
    monkeypatch.setattr(run_screen, "GPU_AUTHORIZATION", probe)
    with pytest.raises(PermissionError, match="GPU_AUTHORIZATION_MISSING"):
        run_screen.require_gpu_authorization()
    with pytest.raises(PermissionError, match="GPU_AUTHORIZATION_MISSING"):
        run_screen.require_gpu_authorization("A_s0")

    base = {
        "authorized": True,
        "protocol_id": C.PROTOCOL_ID,
        "protocol_freeze_sha256": C.sha256_file(C.PROTOCOL_FREEZE),
        "max_gpu_hours": 1.5,
        "authorized_runs": [f"{arm}_s{seed}" for arm in C.NN_ARMS for seed in C.SEEDS],
        "researcher_note": "unit-test probe",
    }
    probe.write_text(json.dumps(base), encoding="utf-8")
    assert run_screen.require_gpu_authorization("A_s0")["max_gpu_hours"] == 1.5

    probe.write_text(json.dumps({**base, "protocol_freeze_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(PermissionError, match="does not match"):
        run_screen.require_gpu_authorization("A_s0")

    probe.write_text(json.dumps({**base, "max_gpu_hours": 2.0}), encoding="utf-8")
    with pytest.raises(PermissionError, match="exceed"):
        run_screen.require_gpu_authorization("A_s0")

    probe.write_text(json.dumps({**base, "authorized_runs": ["A_s0"]}), encoding="utf-8")
    with pytest.raises(PermissionError, match="not covered"):
        run_screen.require_gpu_authorization("B_s1")


def test_config_hash_is_pinned():
    cfg = C.load_config()
    assert cfg["protocol_id"] == C.PROTOCOL_ID
    assert cfg["status"] == "FROZEN_PRE_GPU"
    if C.PROTOCOL_FREEZE.exists():
        frozen = C.read_json(C.PROTOCOL_FREEZE)
        assert frozen["config_hash"] == C.config_hash(cfg)
