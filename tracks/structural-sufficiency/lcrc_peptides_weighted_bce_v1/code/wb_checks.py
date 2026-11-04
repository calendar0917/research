"""Preflight correctness gates for the weighted-BCE intervention.

The same functions back ``tests/test_weighted_bce.py`` and the committed
``analysis/gates.json``.  No GPU or model training is needed here.
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
import wb_common as C
import wb_loss as L
import wb_pos_weight as PW
from bp_metrics import masked_bce_loss


def _check(name: str, fn: Callable[[], dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    try:
        detail = fn()
        ok = bool(detail.pop("passed", True))
        return ok, detail
    except Exception as exc:  # noqa: BLE001 - recorded as a failed gate
        return False, {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def check_pos_weight() -> dict[str, Any]:
    payload = C.read_json(C.POS_WEIGHT_JSON)
    y_train = np.load(C.TRAIN_LABELS)
    y_valid = np.load(C.VALID_LABELS)
    verified = PW.verify_payload(payload, y_train)
    labels = tuple(payload["label_order"])
    if labels != tuple(C.LABELS):
        raise AssertionError(f"label order {labels} != {C.LABELS}")

    p = np.asarray(payload["pos_weight"], dtype=np.float64)
    pos = np.asarray([row["train_pos"] for row in payload["labels"]], dtype=np.float64)
    neg = np.asarray([row["train_neg"] for row in payload["labels"]], dtype=np.float64)
    if not np.array_equal(p, neg / pos):
        raise AssertionError("pos_weight is not exactly N_negative / N_positive")

    valid_pos = y_valid.astype(np.float64).sum(axis=0)
    valid_neg = y_valid.shape[0] - valid_pos
    valid_ratio = valid_neg / valid_pos
    merged = np.concatenate([y_train.astype(np.float64), y_valid.astype(np.float64)], axis=0)
    merged_pos = merged.sum(axis=0)
    merged_ratio = (merged.shape[0] - merged_pos) / merged_pos
    if np.array_equal(p, valid_ratio):
        raise AssertionError("pos_weight equals Validation ratios")
    if np.array_equal(p, merged_ratio):
        raise AssertionError("pos_weight equals Train+Validation ratios")
    if payload.get("valid_used") is not False or payload.get("test_used") is not False:
        raise AssertionError("payload records Validation/Test usage")
    bad_keys = [k for k in payload if "valid_pos" in k or "valid_neg" in k or "test_pos" in k]
    if bad_keys:
        raise AssertionError(f"payload contains Validation/Test counts: {bad_keys}")
    if list(inspect.signature(PW.label_counts).parameters) != ["y_train", "labels"]:
        raise AssertionError("label_counts signature is not Train-labels-only")

    return {
        "passed": True,
        "pos_weight_sha256": C.sha256_file(C.POS_WEIGHT_JSON),
        "payload_canonical_sha256": PW.payload_sha256(payload),
        "counts_match_exact_ratio": True,
        "valid_ratio_equal": False,
        "merged_ratio_equal": False,
        "valid_used": False,
        "test_used": False,
        "label_order": list(labels),
        "pos_weight": [float(x) for x in p],
        "train_pos": [int(x) for x in pos],
        "train_neg": [int(x) for x in neg],
        "n_train": int(y_train.shape[0]),
        "n_valid_used_for_negative_control": int(y_valid.shape[0]),
        "train_labels_array_sha256": verified["train_labels_array_sha256"],
    }


def check_ones_equivalence() -> dict[str, Any]:
    torch.manual_seed(17)
    logits64 = torch.randn(23, C.N_LABELS, dtype=torch.float64) * 2.0
    targets = (torch.rand(23, C.N_LABELS, dtype=torch.float64) > 0.45).to(torch.float64)
    ones = torch.ones(C.N_LABELS, dtype=torch.float64)

    x1 = logits64.clone().requires_grad_(True)
    new = L.weighted_bce_with_logits(x1, targets, ones)
    new.backward()
    grad_new = x1.grad.detach().clone()

    x2 = logits64.clone().requires_grad_(True)
    old = masked_bce_loss(x2, targets)
    old.backward()
    grad_old = x2.grad.detach().clone()

    x3 = logits64.clone().requires_grad_(True)
    ref = F.binary_cross_entropy_with_logits(x3, targets, reduction="mean")
    ref.backward()
    grad_ref = x3.grad.detach().clone()

    value_delta = float(abs(float(new) - float(old)))
    ref_delta = float(abs(float(new) - float(ref)))
    grad_delta = float((grad_new - grad_old).abs().max())
    grad_ref_delta = float((grad_new - grad_ref).abs().max())
    if value_delta != 0.0 or ref_delta != 0.0 or grad_delta != 0.0 or grad_ref_delta != 0.0:
        raise AssertionError(
            f"ones-equivalence failed: value {value_delta}, ref {ref_delta}, "
            f"grad {grad_delta}, grad_ref {grad_ref_delta}"
        )

    # float32 (the production dtype) must also agree to numerical precision.
    x32 = logits64.to(torch.float32).requires_grad_(True)
    t32 = targets.to(torch.float32)
    new32 = L.weighted_bce_with_logits(x32, t32, torch.ones(C.N_LABELS))
    new32.backward()
    with torch.no_grad():
        ref32 = F.binary_cross_entropy_with_logits(
            x32.detach(), t32, reduction="mean"
        )
    float32_value_delta = float(abs(float(new32.detach()) - float(ref32)))
    if float32_value_delta > 1e-6:
        raise AssertionError(f"float32 ones-equivalence value delta {float32_value_delta}")
    if not torch.isfinite(x32.grad).all():
        raise AssertionError("float32 ones-equivalence gradient is non-finite")

    return {
        "passed": True,
        "value_delta_float64": value_delta,
        "grad_delta_float64": grad_delta,
        "value_delta_float32": float32_value_delta,
        "ordinary_bce_delta_float64": ref_delta,
        "ordinary_grad_delta_float64": grad_ref_delta,
    }


def check_extreme_imbalance() -> dict[str, Any]:
    torch.manual_seed(23)
    x = torch.tensor(
        [
            [-3.0, 2.0, 0.5, -1.5, 4.0, -2.0, 1.0, 0.0, 3.0, -4.0],
            [1.5, -0.5, -2.5, 2.0, -1.0, 0.25, -3.5, 1.25, -0.75, 2.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    y = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    weights = torch.tensor(
        [0.5, 100.0, 1.0, 20.0, 0.1, 3.0, 8.0, 50.0, 2.0, 0.01],
        dtype=torch.float64,
    )
    got = L.weighted_bce_with_logits(x, y, weights)
    per = -(
        weights * y * F.logsigmoid(x)
        + (1.0 - y) * F.logsigmoid(-x)
    )
    expected = per.mean()
    delta = float(abs(float(got) - float(expected)))
    if delta > 1e-12:
        raise AssertionError(f"extreme-imbalance manual delta {delta}")

    # Single manually worked element: x=2, y=1, w=100 -> 100*softplus(-2).
    one = L.weighted_bce_with_logits(
        torch.tensor([[2.0]], dtype=torch.float64),
        torch.tensor([[1.0]], dtype=torch.float64),
        torch.tensor([100.0], dtype=torch.float64),
    )
    manual_one = float(100.0 * np.log1p(np.exp(-2.0)))
    one_delta = float(abs(float(one) - manual_one))
    if one_delta > 1e-12:
        raise AssertionError(f"single-element manual delta {one_delta}")
    return {
        "passed": True,
        "batch_shape": [3, C.N_LABELS],
        "manual_delta": delta,
        "single_element_delta": one_delta,
        "manual_formula": "-[w*y*logsigmoid(x) + (1-y)*logsigmoid(-x)], mean",
    }


def check_missing_label_mask() -> dict[str, Any]:
    torch.manual_seed(29)
    x = torch.randn(5, C.N_LABELS, dtype=torch.float64, requires_grad=True)
    y = (torch.rand(5, C.N_LABELS, dtype=torch.float64) > 0.5).to(torch.float64)
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[0, 2] = False
    mask[3, 7] = False
    mask[4, 0] = False
    y_missing = y.clone()
    y_missing[~mask] = float("nan")
    weights = torch.arange(0.5, 10.5, dtype=torch.float64)
    got = L.weighted_bce_with_logits(x, y_missing, weights)

    safe = torch.where(mask, y, torch.zeros_like(y))
    per = -(
        weights * safe * F.logsigmoid(x)
        + (1.0 - safe) * F.logsigmoid(-x)
    )
    expected = per[mask].mean()
    delta = float(abs(float(got) - float(expected)))
    if delta > 1e-12:
        raise AssertionError(f"masked manual delta {delta}")
    if tuple(per.shape) != tuple(x.shape):
        raise AssertionError("elementwise shape changed")
    got.backward()
    if not torch.isfinite(x.grad).all():
        raise AssertionError("masked gradient is non-finite")
    if float(x.grad[~mask].abs().sum()) != 0.0:
        raise AssertionError("masked positions received non-zero gradient")

    # All-missing is a zero-loss sentinel, matching the historical helper.
    all_missing = torch.full_like(y, float("nan"))
    zero = L.weighted_bce_with_logits(x.detach(), all_missing, weights)
    if float(zero) != 0.0:
        raise AssertionError(f"all-missing loss {float(zero)} != 0")
    return {
        "passed": True,
        "masked_manual_delta": delta,
        "masked_gradient_zero_at_missing": True,
        "elementwise_shape_preserved": list(per.shape),
        "all_missing_loss": float(zero),
        "n_masked": int((~mask).sum()),
    }


def check_finite_grad_update() -> dict[str, Any]:
    torch.manual_seed(31)
    model = torch.nn.Linear(C.N_LABELS, C.N_LABELS)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    x = torch.randn(11, C.N_LABELS)
    y = (torch.rand(11, C.N_LABELS) > 0.7).to(torch.float32)
    w = torch.linspace(0.2, 20.0, C.N_LABELS, dtype=torch.float32)
    before = [p.detach().clone() for p in model.parameters()]
    logits = model(x)
    loss = L.weighted_bce_with_logits(logits, y, w)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    finite = all(bool(torch.isfinite(g).all()) for g in grads) and bool(torch.isfinite(loss))
    opt.step()
    changed = any(not torch.equal(a, b.detach()) for a, b in zip(before, model.parameters(), strict=True))
    if not finite or not changed:
        raise AssertionError(f"finite={finite}, parameters_changed={changed}")
    return {
        "passed": True,
        "loss": float(loss.detach()),
        "gradients_finite": finite,
        "parameters_updated": changed,
        "grad_norm": float(torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))),
    }


def check_guard() -> dict[str, Any]:
    before = len(C.GUARD["attempts"])
    blocked = False
    error_type = None
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "peptides_func_test.pt"
        try:
            with open(target, "w", encoding="utf-8") as fh:
                fh.write("must not be created")
        except Exception as exc:  # noqa: BLE001
            blocked = True
            error_type = type(exc).__name__
    after = len(C.GUARD["attempts"])
    # The selftest is not a training-run access; leave the shared guard clean.
    del C.GUARD["attempts"][before:after]
    if not blocked or after != before + 1:
        raise AssertionError(f"guard did not block/record: blocked={blocked}, before={before}, after={after}")
    return {
        "passed": True,
        "blocked": blocked,
        "error_type": error_type,
        "recorded_attempts": after - before,
        "guard_state_after": len(C.GUARD["attempts"]),
    }


def check_source_no_test_access() -> dict[str, Any]:
    files = [
        C.CODE_DIR / "wb_p2_train.py",
        C.CODE_DIR / "wb_compute_pos_weight.py",
    ]
    tokens = ("peptides_func_test", "peptides_func_y_test", "test_predictions", "y_test", "_peptide_test_loader")
    hits: dict[str, list[str]] = {}
    for path in files:
        text = path.read_text(encoding="utf-8")
        found = [token for token in tokens if token in text]
        if found:
            hits[path.name] = found
    if hits:
        raise AssertionError(f"entry-point source mentions Test artifacts: {hits}")
    return {
        "passed": True,
        "files_checked": [C.track_rel(p) for p in files],
        "forbidden_tokens": list(tokens),
        "hits": {},
    }


def check_label_order() -> dict[str, Any]:
    payload = C.read_json(C.POS_WEIGHT_JSON)
    import csv

    with open(C.LABEL_STATISTICS, "r", encoding="utf-8", newline="") as fh:
        stats = list(csv.DictReader(fh))
    labels = [row["label"] for row in stats]
    if labels != list(C.LABELS):
        raise AssertionError(f"label statistics order {labels} != {list(C.LABELS)}")
    if payload["label_order"] != list(C.LABELS):
        raise AssertionError("pos_weight label order mismatch")
    weights = payload["pos_weight"]
    if len(weights) != C.N_LABELS or not all(np.isfinite(weights)):
        raise AssertionError("pos_weight is not 10 finite values")
    return {
        "passed": True,
        "label_order": list(C.LABELS),
        "n_labels": C.N_LABELS,
        "dtype": "float64 in JSON, cast to float32 on the logits device",
        "shape": [C.N_LABELS],
    }


def run_all() -> dict[str, Any]:
    specs: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("pos_weight_train_only_and_exact", check_pos_weight),
        ("label_order_dimension", check_label_order),
        ("ones_equals_ordinary_bce_value_and_grad", check_ones_equivalence),
        ("extreme_imbalance_manual_formula", check_extreme_imbalance),
        ("missing_label_mask_no_broadcast", check_missing_label_mask),
        ("finite_gradient_optimizer_update", check_finite_grad_update),
        ("test_access_guard_blocks", check_guard),
        ("entrypoint_source_no_test_access", check_source_no_test_access),
    ]
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    for name, fn in specs:
        ok, detail = _check(name, fn)
        checks[name] = bool(ok)
        details[name] = detail
    return {
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "generated_utc": C.utcnow(),
        "all_passed": bool(all(checks.values())),
        "checks": checks,
        "details": details,
        "environment": C.environment(),
    }


__all__ = ["run_all"]
