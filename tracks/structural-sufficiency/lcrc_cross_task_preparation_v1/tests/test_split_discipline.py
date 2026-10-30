from __future__ import annotations

import inspect

import cpu_gates as G
import lcrc_common as C


def test_source_gate_pass():
    report = G.gate_sources()
    assert report["status"] == "PASS"
    assert report["manifest"]["split"]["union_equals_official_train_rows"] is True
    assert report["manifest"]["split"]["row_1646_in_frozen_union"] is False


def test_frozen_split_scope_flags(split):
    scope = split["payload"]["scope"]
    assert scope["labels_read"] is False
    assert scope["structure_only"] is True
    assert scope["official_valid_test_labels_read"] is False


def test_no_forbidden_label_paths_are_opened():
    # Static source inspection: the audit may mention forbidden filenames in
    # documentation or negative checks, but must not open/read them.
    import re

    text = ""
    for path in (C.CODE_DIR / "adapters.py", C.CODE_DIR / "audit_cpu.py", C.CODE_DIR / "cpu_gates.py"):
        text += path.read_text(encoding="utf-8")
    assert not re.search(r"(?:open|read_json|torch\.load|np\.load)\s*\([^)\n]*labels_fit_monitor", text)
    assert not re.search(r"(?:open|read_json|torch\.load|np\.load)\s*\([^)\n]*labels\.npy", text)


def test_audit_scope_is_cpu_and_train_only():
    source = inspect.getsource(__import__("audit_cpu"))
    assert "official-train-only" in source
    assert "torch.cuda.is_available()" in source
    assert "optimizer.step" not in source
    assert "loss.backward" not in source
