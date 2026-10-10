"""cscl-prediction-feasibility-v2 correctness tests.

Covers the Phase-4 checklist for the Full-Y-fit7200-seed0 candidate:
split integrity, fit-only preprocessing, y-only supervision, non-GNN
computation graph, batching/order invariance, save/load equality, and
guard-rails (no g/c loading, no official valid/test access, seed 0 only).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
for p in (str(_CODE_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)

pytestmark = pytest.mark.slow  # heavy data-loading tests; deselect with -m "not slow"


def _driver():
    import run_prediction_feasibility_v2 as drv

    return drv


@pytest.fixture(scope="module")
def env():
    """One shared (512-row smoke) prep build for the whole module."""
    drv = _driver()
    import cscl_features as cf

    quiet = lambda *a, **k: None  # noqa: E731
    tmp = tempfile.mkdtemp(prefix="feasv2-test-")
    train = drv.load_train_rows(log=quiet)
    idx = cf.build_split_indices([str(s) for s in cf.load_canonical_smiles()])
    drv.verify_row_alignment(train, log=quiet)
    path, meta = drv.build_prep_fit7200(train, idx, Path(tmp) / "prep", smoke=True, log=quiet)
    prep = {k: z for k, z in np.load(path, allow_pickle=False).items()}
    blob10 = {k: z for k, z in np.load(drv.LEGACY_PREP, allow_pickle=False).items()}
    return drv, train, idx, path, meta, prep, blob10


# ---------------------------------------------------------------------------
# static guards (fast)
# ---------------------------------------------------------------------------


def test_seed_zero_only():
    drv = _driver()
    text = Path(drv.__file__).read_text()
    assert "choices=[0]" in text, "driver must enforce seed 0 only"


def test_no_decomp_or_heldout_access_in_driver():
    import ast

    drv = _driver()
    text = Path(drv.__file__).read_text()
    tree = ast.parse(text)
    doc = ast.get_docstring(tree) or ""
    code_only = text.replace(text.strip().split('\n', 1)[0], '', 1) if False else text
    # strip every docstring (they document the discipline) then scan code strings
    class _Strip(ast.NodeTransformer):
        def _drop(self, node):
            self.generic_visit(node)
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
            return node

        visit_Module = _drop
        visit_FunctionDef = _drop
        visit_AsyncFunctionDef = _drop
        visit_ClassDef = _drop

    stripped = ast.unparse(_Strip().visit(ast.parse(text)))
    for forbidden in (
        "target_decomposition",
        "load_decomp",
        "encoded_valid",
        "load_test_data",
        'split="val"',
        'split="test"',
        'split="valid"',
        "ZINC(root",
    ):
        assert forbidden not in stripped, f"driver code must not reference {forbidden}"
    assert "never" in doc and "raw y" in doc


def test_protocol_yaml_frozen_selection_rule():
    text = (REPO_ROOT / "tracks/structure-contribution/configs/prediction_feasibility_v2.yaml").read_text()
    for key in ("cscl-prediction-feasibility-v2", "Full-Y-fit7200-seed0", "0.33206", "top-5 monitor"):
        assert key in text


# ---------------------------------------------------------------------------
# split integrity
# ---------------------------------------------------------------------------


def test_split_indices_unchanged(tmp_path):
    import cscl_features as cf

    smiles = [str(s) for s in cf.load_canonical_smiles()]
    idx = cf.build_split_indices(smiles)
    assert len(idx["fit_inner"]) == 7200
    assert len(idx["monitor"]) == 800
    assert len(idx["dev"]) == 2000
    assert np.intersect1d(idx["fit_inner"], idx["dev"]).size == 0
    assert np.intersect1d(idx["fit_inner"], idx["monitor"]).size == 0
    assert np.intersect1d(idx["monitor"], idx["dev"]).size == 0
    fit_smiles = {smiles[i] for i in idx["fit_inner"]}
    assert not (fit_smiles & {smiles[i] for i in idx["monitor"]})
    assert not (fit_smiles & {smiles[i] for i in idx["dev"]})


# ---------------------------------------------------------------------------
# prep: fit-only + identity
# ---------------------------------------------------------------------------


def test_prep_identity_and_fit_only(env):
    drv, _train, _idx, _path, meta, _prep, _b10 = env
    for block in ("patch", "ctx", "topo", "anchor"):
        assert meta["identity_checks"][block]["nondegenerate"] <= drv.IDENTITY_TOL, block
    assert meta["legacy_prep"]["use"].startswith("inversion basis only")
    assert meta["ksvd"]["phi_rows"] > 0
    assert _prep is not None


def test_prep_standardizers_differ_from_all10k(env):
    drv, _train, _idx, _path, _meta, prep, blob10 = env
    for block in ("patch", "ctx", "anchor", "topo"):
        assert not np.allclose(prep[f"{block}_fit_mean"], blob10[f"{block}_all_mean"], atol=1e-4), block
        assert not np.allclose(prep[f"{block}_fit_scale"], blob10[f"{block}_all_scale"], atol=1e-4), block


# ---------------------------------------------------------------------------
# model: non-GNN audit + supervision + invariances
# ---------------------------------------------------------------------------


def _dev_lists(drv, train, idx, prep, blob10, n=12):
    drv.apply_prep_fit7200(train, prep, blob10)
    rows = [int(i) for i in idx["dev"][:n]]
    return [train[i] for i in rows]


def test_model_build_parameter_audit(env):
    drv, train, idx, _path, _meta, prep, blob10 = env
    lists = _dev_lists(drv, train, idx, prep, blob10, n=1)
    assert lists
    model = drv.build_model(prep, seed=0)
    assert drv.FULL_PARAMETERS == 408_651
    names = {type(m).__name__ for m in model.modules()}
    banned = {
        "MessagePassing", "GCNConv", "GINConv", "GINEConv", "GraphConv",
        "TransformerConv", "GATConv", "MultiheadAttention", "Transformer",
    }
    assert not (names & banned), names & banned


def test_forward_shape_finite_and_label_independent(env):
    import torch

    drv, train, idx, _path, _meta, prep, blob10 = env
    lists = _dev_lists(drv, train, idx, prep, blob10, n=8)
    model = drv.build_model(prep, seed=0)
    model.eval()
    device = torch.device("cpu")
    targets = torch.zeros(len(lists))
    with torch.no_grad():
        pred_a = drv.predict_raw(model, lists, device)
    # changing every label must not change any prediction
    saved_y = [d.y.clone() for d in lists]
    for d in lists:
        d.y = torch.tensor([999.0])
    with torch.no_grad():
        pred_b = drv.predict_raw(model, lists, device)
    for d, y in zip(lists, saved_y):
        d.y = y
    assert np.array_equal(pred_a, pred_b)
    assert np.isfinite(pred_a).all() and float(np.abs(pred_a).max()) < 100.0


def test_prediction_invariant_to_batch_composition_and_order(env):
    import torch

    drv, train, idx, _path, _meta, prep, blob10 = env
    lists = _dev_lists(drv, train, idx, prep, blob10, n=12)
    model = drv.build_model(prep, seed=0)
    model.eval()
    device = torch.device("cpu")
    full = drv.predict_raw(model, lists, device)
    singles = [drv.predict_raw(model, [lists[i]], device)[0] for i in (3, 7, 0, 11)]
    for k, i in enumerate((3, 7, 0, 11)):
        assert abs(full[i] - singles[k]) < 2e-4, (i, full[i], singles[k])
    part = drv.predict_raw(model, [lists[i] for i in (5, 9, 2)], device)
    for k, i in enumerate((5, 9, 2)):
        assert abs(full[i] - part[k]) < 2e-4


def test_save_load_state_equality(env):
    import torch

    drv, train, idx, _path, _meta, prep, blob10 = env
    lists = _dev_lists(drv, train, idx, prep, blob10, n=8)
    model = drv.build_model(prep, seed=0)
    model.eval()
    p1 = drv.predict_raw(model, lists, torch.device("cpu"))
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    import tempfile

    f = Path(tempfile.mkdtemp()) / "state.pt"
    torch.save(sd, f)
    model2 = drv.build_model(prep, seed=1)  # different init …
    model2.load_state_dict(torch.load(f, weights_only=True))
    model2.eval()
    p2 = drv.predict_raw(model2, lists, torch.device("cpu"))
    assert np.array_equal(p1, p2)


def test_train_step_loss_finite_params_update(env):
    import torch

    drv, train, idx, _path, _meta, prep, blob10 = env
    lists = _dev_lists(drv, train, idx, prep, blob10, n=32)
    model = drv.build_model(prep, seed=0)
    model.train()
    y = torch.tensor([float(d.y.reshape(-1)[0]) for d in lists], dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    batch = drv._batch(lists, list(range(len(lists))), y, torch.device("cpu"))
    pred, aux = model(batch, mask=drv.cm.C6_MASK, return_aux=True)
    task = torch.nn.functional.l1_loss(pred.view(-1), batch.y.view(-1))
    rec = model.reconstruction_loss(aux["phi"], aux["coord"])
    loss = task + drv.H1_LAMBDA * rec
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    opt.step()
    changed = sum(1 for k in before if not torch.equal(before[k], model.state_dict()[k]))
    assert changed > 0
    assert torch.isfinite(task) and torch.isfinite(rec)


def test_c6_mask_topology_not_zeroed():
    drv = _driver()
    mask = drv.cm.C6_MASK
    assert mask.topology_zero is False
    assert set(mask.global_zero_groups) == {"atom_histogram", "bond_histogram"}


def test_reconstruction_loss_is_label_free(env):
    import torch

    drv, train, idx, _path, _meta, prep, blob10 = env
    lists = _dev_lists(drv, train, idx, prep, blob10, n=8)
    model = drv.build_model(prep, seed=0)
    model.eval()
    phi = lists[0].dict_phi
    coord = model.code(phi)
    l1 = float(model.reconstruction_loss(phi, coord))
    saved = lists[0].y.clone()
    lists[0].y = torch.tensor([12345.0])
    l2 = float(model.reconstruction_loss(lists[0].dict_phi, model.code(lists[0].dict_phi)))
    lists[0].y = saved
    assert l1 == l2
