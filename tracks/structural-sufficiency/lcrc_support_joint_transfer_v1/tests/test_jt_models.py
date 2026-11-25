from __future__ import annotations

import numpy as np
import torch

import jt_common as C
import jt_models as M
import jt_support as S

S._prepare()
M.prepare_imports()

import cr_common as CC  # noqa: E402
import cr_data as D  # noqa: E402
import cr_models as CM  # noqa: E402


def test_lipo_joint_and_additive_construction_and_forward():
    bundle = D.build_bundle("ogbg-mollipo")
    support = S.load_support(C.ANALYSIS / "support/lipo_support.npz")
    canonical = CM.canonical_c1_state(bundle.spec, bundle.norm)
    CC.set_global_seeds(0)
    baseline = CM.ArmA(bundle.spec, bundle.norm)
    baseline.load_canonical(canonical)
    joint, joint_info = M.build_lipo_model(bundle, support, "C1_JOINT")
    add, add_info = M.build_lipo_model(bundle, support, "C1_ADD")

    assert joint_info["parameters"] == 215714
    assert add_info["parameters"] == 213794
    assert joint_info["joint_module_parameters"] == 108 * 32
    assert add_info["joint_module_parameters"] == 48 * 32
    assert M.compare_parameter_states(M.common_parameter_state(baseline), M.common_parameter_state(joint))["all_equal"]
    assert M.compare_parameter_states(M.common_parameter_state(baseline), M.common_parameter_state(add))["all_equal"]

    rows = np.arange(8, dtype=np.int64)
    batch = bundle.collate(bundle.caches["train"], rows)
    baseline.eval()
    joint.eval()
    add.eval()
    with torch.no_grad():
        p0 = baseline(batch).numpy()
        pj = joint(batch).numpy()
        pa = add(batch).numpy()
    assert np.max(np.abs(p0 - pj)) < 1e-5
    # ADD is in the marginal subspace at init, exactly in float64.
    theta = add.trunk.local.Theta.detach().double().numpy()
    table = np.asarray(support.U, dtype=np.float64) @ theta
    residual = table - support.P @ table
    assert np.max(np.abs(residual)) < 1e-8
    assert np.max(np.abs(pa - p0)) > 1e-7  # removing joint_proj is visible


def test_joint_table_is_the_materialised_original_projection():
    bundle = D.build_bundle("ogbg-mollipo")
    support = S.load_support(C.ANALYSIS / "support/lipo_support.npz")
    joint, _ = M.build_lipo_model(bundle, support, "C1_JOINT")
    canonical = CM.canonical_c1_state(bundle.spec, bundle.norm)
    CC.set_global_seeds(0)
    baseline = CM.ArmA(bundle.spec, bundle.norm)
    baseline.load_canonical(canonical)
    original = M.materialize_joint(baseline.trunk.local, support).cpu().numpy()
    table = M.table_of(joint)
    assert table.shape == (108, 32)
    # Materialisation is deterministic and the JOINT table starts exactly at J0.
    again = M.materialize_joint(baseline.trunk.local, support).cpu().numpy()
    assert np.array_equal(original, again)
    assert np.allclose(table, original, atol=1e-6)
