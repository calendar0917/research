"""Unit tests for round lcrc-relation-role-preserving-pilot-v1.

Fast: no 360-epoch trajectory is trained.  The tests assert the invariants that
make the four formal units interpretable:

* the round identity, the frozen recipe and the declared protocol rules;
* the REAL role is the frozen topology-block ownership test and nothing else;
* the CONTROL role is a pure function of the typed-S2 code (no atoms, no bonds,
  no row id, no label, no oracle) and is invariant under graph relabelling;
* the two role groups are a partition of the S2 occurrences, their union is the
  frozen C1 aggregation, and with ``gamma == 1`` the model is the C1 model;
* both models have identical parameter counts and identical initial state, with
  exactly ``2 * 200`` parameters more than C1;
* the decision logic maps pre-declared statistics to the five verdict strings.

The expensive whole-split gates live in ``rp_checks`` and run in the ``checks``
stage; here they are exercised on a couple of molecules only.
"""

from __future__ import annotations

import hashlib
import inspect
import json

import numpy as np
import pytest

import rp_common as C


# ---------------------------------------------------------------------------
# identity / configuration
# ---------------------------------------------------------------------------


def test_identity_constants() -> None:
    assert C.ROUND_ID == "lcrc-relation-role-preserving-pilot-v1"
    assert C.PROTOCOL_ID == C.ROUND_ID
    assert C.STUDY_ID == C.ROUND_ID
    assert C.TRAIN_SEED == 0
    assert C.TASKS == ("lipo", "wc_logp")
    assert C.MODELS == ("RR_REAL", "RR_CONTROL")
    assert C.DATASET == "ogbg-mollipo"
    assert C.EXPECTED_SPLIT_SIZES == {"train": 3360, "valid": 420, "test": 420}
    # the frozen C1 shapes this round must not touch
    assert C.WIDTH == 64
    assert C.S2_TYPED_DIM == 104
    assert C.S2_CONTENT_DIM == 96
    assert C.S2_ROLE_DIM == 200
    assert C.LOCAL_INPUT_DIM == 433
    assert C.INIT_SEED_CANONICAL == 0


def test_paths_point_into_this_round_and_reuse_is_read_only() -> None:
    assert C.CODE_DIR == C.ROUND_DIR / "code"
    assert C.RESULTS_DIR == C.ROUND_DIR / "results"
    assert C.ANALYSIS_DIR == C.ROUND_DIR / "analysis"
    assert C.CACHE_DIR == C.ROUND_DIR / "cache"
    for path in (C.SQL_CODE, C.RC_CODE, C.PREP_CODE, C.DISSECT_CODE, C.PRED_CODE):
        assert path.is_dir(), path
    for path in (C.PRED_DIR, C.RC_DIR, C.PREP_DIR, C.DISSECT_DIR, C.ROUTE_DIR):
        assert C.ROUND_DIR not in path.parents
        assert path != C.ROUND_DIR


def test_frozen_command_inputs_are_untouched() -> None:
    for path in (C.ORACLE_CONTRIB_NPZ, C.WC_LOG_PRED_CSV, C.ROUTE_STAGE1):
        assert path.is_file(), path
    for split in C.SPLITS:
        cache = C.PRED_CACHE / f"ogbg_mollipo_{split}_topology-routing-cache-v1.npz"
        assert cache.is_file(), cache


def test_recipe_is_the_frozen_rf_recipe() -> None:
    recipe = C.RECIPE
    assert recipe["epochs"] == 360
    assert recipe["batch_size"] == 128
    assert recipe["eval_batch_size"] == 256
    assert recipe["peak_lr"] == pytest.approx(1e-3)
    assert recipe["min_lr"] == pytest.approx(1e-5)
    assert recipe["warmup_epochs"] == 5
    assert recipe["weight_decay"] == pytest.approx(1e-5)
    assert recipe["gradient_clip"] == pytest.approx(5.0)
    assert recipe["dropout"] == pytest.approx(0.10)
    assert recipe["amp"] is False
    assert recipe["early_stopping"] is False
    assert recipe["soup"] is False
    assert recipe["test_policy"].startswith("always")


def test_lr_schedule_matches_the_shared_implementation() -> None:
    import rp_train as T

    out = T.schedule_audit()
    assert out["n_steps"] == T.total_steps(3360) == 360 * 27
    assert out["max_abs_diff"] == 0.0
    assert out["first_step_lr"] == pytest.approx(1e-4)
    assert out["last_step_lr"] == pytest.approx(1e-5, rel=1e-4)


def test_protocol_declares_the_round_rules() -> None:
    import yaml

    protocol = yaml.safe_load(C.ROUND_DIR.joinpath("protocols", f"{C.PROTOCOL_ID}.yaml").read_text())
    assert protocol["test_policy"] == "always"
    assert protocol["validation_selection_policy"]["forbidden"].startswith("selecting an epoch by Test")
    assert protocol["split"]["sizes"]["ogbg-mollipo"] == {"train": 3360, "valid": 420, "test": 420}
    assert protocol["seed_semantics"]["seeds"] == [0]
    assert protocol["comparability"]["ranking_metric"] == "valid_rmse"
    assert protocol["prototype_decision_scales"]["prediction_retention"]
    assert "atom index" in " ".join(protocol["role_split"]["forbidden"])


def test_study_and_config_parse() -> None:
    import yaml

    study = yaml.safe_load(C.ROUND_DIR.joinpath("studies", f"{C.STUDY_ID}.yaml").read_text())
    assert study["id"] == C.STUDY_ID
    assert [c["id"] for c in study["candidates"]] == list(C.MODELS)
    config = yaml.safe_load(C.FROZEN_CONFIG.read_text())
    assert config["protocol_id"] == C.PROTOCOL_ID
    assert config["seed"] == 0
    assert config["model"] in C.MODELS
    assert config["epochs"] == 360


# ---------------------------------------------------------------------------
# the CONTROL role: a pure function of the typed-S2 code
# ---------------------------------------------------------------------------


def test_control_hash_is_a_pure_function_of_the_type_triple() -> None:
    import rp_data as D

    params = list(inspect.signature(D.control_hash_role).parameters)
    assert params == ["path_cat", "end63", "closure"]
    # deterministic and stable
    assert D.control_hash_role(12, 3, 1) == D.control_hash_role(12, 3, 1)
    assert D.control_hash_role(12, 3, 1) in (0, 1)
    # the frozen salt is the one recorded in the manifest / protocol
    assert C.CONTROL_SALT.startswith(C.ROUND_ID)
    assert (hashlib.blake2b(f"{C.CONTROL_SALT}|12|3|1".encode(), digest_size=8).digest()[0] & 1) is not None
    # one role per type: identical triples cannot receive different roles
    table = {
        "key": np.asarray([D.triple_key(12, 3, 1)]),
        "role": np.asarray([D.control_hash_role(12, 3, 1)], np.uint8),
    }
    roles = D.roles_from_triples(table, np.asarray([12, 12]), np.asarray([3, 3]), np.asarray([1, 1]))
    assert roles.tolist() == [int(table["role"][0])] * 2


def test_control_table_is_frozen_and_complete(bundle) -> None:
    import rp_data as D

    if not C.control_manifest_path().is_file():
        pytest.skip("prepare stage has not frozen the CONTROL table in this checkout")
    manifest = C.read_json(C.control_manifest_path())
    assert manifest["labels_used"] is False
    assert manifest["oracle_used"] is False
    assert manifest["atom_index_used"] is False
    assert manifest["row_id_used"] is False
    table = D.load_control_table(verify=True)
    assert table["role"].size == table["key"].size == int(manifest["n_types"])
    assert set(np.unique(table["role"]).tolist()) <= {0, 1}
    # the frozen table is exactly the hash function, recomputed
    recomputed = np.asarray(
        [D.control_hash_role(pc, e, k) for pc, e, k in zip(table["path_cat"], table["path_end63"], table["path_closure"])],
        dtype=np.uint8,
    )
    assert np.array_equal(recomputed, table["role"])
    # every observed type of every split is covered (no unseen type at train time)
    for split in C.SPLITS:
        cache = bundle.caches[split]
        roles = D.roles_from_triples(table, cache["path_cat"], cache["path_end63"], cache["path_closure"])
        assert roles.size == int(np.asarray(cache["path_cat"]).size)


def test_real_role_is_exactly_the_frozen_owner_test() -> None:
    """REAL role == (owner(e1) != owner(e2)), read from the frozen routing cache."""
    for split in C.SPLITS:
        path = C.PRED_CACHE / f"ogbg_mollipo_{split}_topology-routing-cache-v1.npz"
        with np.load(path, allow_pickle=False) as mesh:
            cross = np.asarray(mesh["path_cross"], dtype=np.int64)
            o1 = np.asarray(mesh["path_owner1"], dtype=np.int64)
            o2 = np.asarray(mesh["path_owner2"], dtype=np.int64)
        assert np.array_equal(cross, (o1 != o2).astype(np.int64))
        # internal + cross == all occurrences, with both classes non-empty
        assert cross.sum() > 0 and cross.sum() < cross.size


# ---------------------------------------------------------------------------
# relabelling invariance (one toy molecule, cheap)
# ---------------------------------------------------------------------------


def _toy_caches(rr):
    """One real molecule, in its original labelling and in a relabelled copy."""
    MOD = C.modules()
    adapters = MOD["adapters"]
    lcrc = MOD["lcrc_data"]
    sq_data = MOD["sq_data"]
    graph = adapters.lipo_row_to_graph(sq_data.lipo_raw_row(rr.lipo_raw, 0), 0, rr.spec)
    base = lcrc.build_cache([graph], rr.spec, split="toy-base")
    rng = np.random.default_rng(7)
    perm = rng.permutation(int(graph.atom_type.size))
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    edges = inv[np.stack([graph.bond_u, graph.bond_v], axis=1)]
    order = rng.permutation(edges.shape[0])
    edges = edges[order]
    relabelled_graph = MOD["molecular_graph"].build_molecular_graph(
        atom_type=np.asarray(graph.atom_type)[perm],
        bond_index=edges,
        bond_type=np.asarray(graph.bond_type)[order],
        atom_feat=np.asarray(graph.atom_feat)[perm],
        bond_feat=np.asarray(graph.bond_feat)[order],
        global_feat=None if graph.global_feat is None else np.asarray(graph.global_feat),
        molecule_id=str(graph.molecule_id),
        smiles=str(graph.smiles),
        raw_atom_charge=None if graph.raw_atom_charge is None else np.asarray(graph.raw_atom_charge)[perm],
        raw_atom_h=None if graph.raw_atom_h is None else np.asarray(graph.raw_atom_h)[perm],
        bond_stereo=None if graph.bond_stereo is None else np.asarray(graph.bond_stereo)[order],
    )
    other = lcrc.build_cache([relabelled_graph], rr.spec, split="toy-relabelled")
    return base, other


def _type_role_counts(cache) -> dict[tuple[int, int, int], int]:
    import rp_data as D

    table = D.load_control_table()
    triples = list(
        zip(
            np.asarray(cache["path_cat"], dtype=np.int64).tolist(),
            np.asarray(cache["path_end63"], dtype=np.int64).tolist(),
            np.asarray(cache["path_closure"], dtype=np.int64).tolist(),
        )
    )
    assert len(triples) == int(np.asarray(cache["path_cat"]).size)
    counts: dict[tuple[int, int, int], int] = {}
    for key in triples:
        counts[key] = counts.get(key, 0) + 1
    assert all(D.control_hash_role(*key) in (0, 1) for key in counts)
    return counts


def test_control_role_is_invariant_under_atom_and_bond_relabelling(bundle) -> None:
    base, other = _toy_caches(bundle.cr)
    assert _type_role_counts(base) == _type_role_counts(other)
    # CONTROL is a partition of every occurrence into two classes
    import rp_data as D

    table = D.load_control_table()
    for cache in (base, other):
        roles = D.roles_from_triples(table, cache["path_cat"], cache["path_end63"], cache["path_closure"])
        assert roles.size == int(np.asarray(cache["path_cat"]).size)
        assert set(np.unique(roles).tolist()) <= {0, 1}


# ---------------------------------------------------------------------------
# model: C1 equivalence at gamma = 1, parameter identity
# ---------------------------------------------------------------------------


def test_gamma_one_is_the_frozen_c1_aggregation(bundle) -> None:
    """Q_internal + Q_cross equals the frozen C1 S2 column block, atom by atom."""
    import torch

    import rp_models as M

    canonical = M.canonical_state(bundle.spec, bundle.norm)
    batch = bundle.collate("valid", [0, 1, 2])
    ones = (torch.ones(C.S2_ROLE_DIM), torch.ones(C.S2_ROLE_DIM))
    for variant in C.MODELS:
        model = M.build_model(variant, bundle.spec, bundle.norm, canonical, device="cpu").eval()
        role_key = C.MODEL_ROLE_KEY[variant]
        with torch.no_grad():
            _pred, parts = model(batch, role_key=role_key, gammas=ones, return_aux=True)
        assert torch.allclose(parts["q_internal"] + parts["q_cross"], parts["q_total"], atol=1e-6)
        # the local input's S2 block is exactly the summed evidence
        s2 = parts["local_input"][:, 96 + 96 + 32 : 96 + 96 + 32 + C.S2_ROLE_DIM]
        assert torch.allclose(s2, parts["q_total"], atol=1e-6)
        # A gamma of 1 reproduces the default forward pass bit for bit
        with torch.no_grad():
            base = model(batch, role_key=role_key)
        assert torch.equal(base, _pred)


def test_parameter_identity_between_the_two_models() -> None:
    import rp_models as M

    MOD = C.modules()
    rr = MOD["rr_data"].build_bundle(verify=False)
    canonical = M.canonical_state(rr.spec, rr.norm)
    reports = {}
    states = {}
    for variant in C.MODELS:
        model = M.build_model(variant, rr.spec, rr.norm, canonical, device="cpu")
        reports[variant] = model.parameter_report()
        states[variant] = {k: v.clone() for k, v in model.state_dict().items()}
        assert bool((model.gamma_internal == 1).all()) and bool((model.gamma_cross == 1).all())
    assert reports["RR_REAL"]["total"] == reports["RR_CONTROL"]["total"]
    assert reports["RR_REAL"]["total"] - int(M.CM.FULL_C1_PARAMS) == 2 * C.S2_ROLE_DIM
    assert set(states["RR_REAL"]) == set(states["RR_CONTROL"])
    for key in states["RR_REAL"]:
        assert bool(np.array_equal(states["RR_REAL"][key].numpy(), states["RR_CONTROL"][key].numpy())), key
    # the trainable C1 part is exactly the C1 parameter count
    assert int(sum(v.numel() for v in M.build_model("RR_REAL", rr.spec, rr.norm, canonical, device="cpu").canonical_parameters().values())) == int(M.CM.FULL_C1_PARAMS)


# ---------------------------------------------------------------------------
# the cheap versions of the expensive gates
# ---------------------------------------------------------------------------


def test_role_partition_and_s2_conservation_gates(bundle) -> None:
    import rp_checks as K

    partition = K.gates_role_partition(bundle)
    assert partition["pass"], partition
    conservation = K.gate_s2_conservation(bundle, max_batches=1)
    assert conservation["pass"], {k: v for k, v in conservation.items() if k != "per_split"}


def test_relabelling_gate_on_two_molecules(bundle) -> None:
    import rp_checks as K

    out = K.gate_relabelling(bundle, n_per_split=1)
    assert out["pass"], {k: v for k, v in out.items() if k not in ("molecules",)}


def test_recipe_and_gate_constants() -> None:
    assert C.RETENTION_FACTOR == pytest.approx(1.10)
    assert C.BOOTSTRAP_DRAWS == 2000
    assert C.MIN_ORACLE_CONTROLS == 8
    assert C.MAX_ORACLE_CONTROLS == 16
    assert 0.0 < C.ORACLE_POSITIVE_FRACTION <= 1.0
    assert C.INCREMENT_PROB == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# analysis semantics (the artefacts behind REPORT.md)
# ---------------------------------------------------------------------------


def test_paired_bootstrap_sign_convention() -> None:
    import rp_analysis as A

    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    real = {"row_id": np.arange(300), "y_raw": y, "pred_raw": y + rng.normal(size=300) + 1.0}
    control = {"row_id": np.arange(300), "y_raw": y, "pred_raw": y + rng.normal(size=300)}
    out = A.paired_bootstrap(real, control)
    assert out["rmse_real"] > out["rmse_control"]
    assert out["diff_real_minus_control"] > 0.0     # positive => CONTROL is better
    assert out["prob_real_better"] < 0.5
    assert out["positive_ci_excludes_zero_in_favour_of_real"] is False
    same = A.paired_bootstrap(control, control)
    assert abs(same["diff_real_minus_control"]) < 1e-12
    assert 0.0 < same["prob_real_better"] < 1.0
    with pytest.raises(RuntimeError):
        A.paired_bootstrap(real, {**control, "row_id": np.arange(300)[::-1]})


def test_decision_verdict_mapping() -> None:
    import rp_analysis as A

    def retention(preserved: bool, control: bool) -> dict:
        return {
            "prediction_preserved": preserved,
            "control_also_preserved": control,
            "both_preserved": preserved and control,
            "models": {"RR_REAL": {}, "RR_CONTROL": {}},
        }

    def bootstrap(credit: bool) -> dict:
        ci = [-0.01, -0.001] if credit else [-0.01, 0.02]
        return {
            "comparisons": {
                f"lipo/{split}": {
                    "diff_real_minus_control": -0.005,
                    "ci95_diff": ci,
                    "prob_real_better": 0.99 if credit else 0.6,
                    "n": 420,
                }
                for split in C.SPLITS
            }
        }

    no_oracle = {"available": False, "paired_real_minus_control": {}}
    cases = [
        (True, True, True, "ROLE_INCREMENT_SHOWN"),
        (True, True, False, "ONLY_C1_READOUT_RECOVERY"),
        (False, False, False, "ROLE_NOT_SUPPORTED"),
        (True, False, False, "UNRESOLVED"),
    ]
    units = {(task, v): {} for task in C.TASKS for v in C.MODELS}
    for preserved, control, credit, verdict in cases:
        out = A.build_decision(units, retention(preserved, control), bootstrap(credit), {}, no_oracle)
        assert out["verdict"] == verdict, (preserved, control, credit, out["verdict"])
        assert out["flags"]["TEST_USED_FOR_SELECTION"] is False
    # an incomplete round is UNRESOLVED whatever the numbers say
    partial = {("lipo", "RR_REAL"): {}}
    out = A.build_decision(partial, retention(True, True), bootstrap(True), {}, no_oracle)
    assert out["verdict"] == "UNRESOLVED"


def test_role_group_audit_totals_match_the_frozen_cache(bundle) -> None:
    if not (C.ANALYSIS_DIR / "role_groups.json").is_file():
        pytest.skip("prepare stage has not run in this checkout")
    payload = C.read_json(C.ANALYSIS_DIR / "role_groups.json")
    assert payload["all_pass"] is True
    for split in C.SPLITS:
        total = int(np.asarray(bundle.caches[split]["path_cat"]).size)
        for variant in C.MODELS:
            entry = payload["splits"][split][variant]
            assert entry["n_occurrences"] == total
            assert entry["n_group0"] + entry["n_group1"] == total
            assert entry["pass"] is True


def test_report_is_generated_from_artifacts() -> None:
    decision_path = C.ANALYSIS_DIR / "decision.json"
    report_path = C.ROUND_DIR / "REPORT.md"
    if not decision_path.is_file() or not report_path.is_file():
        pytest.skip("aggregate stage has not run in this checkout")
    text = report_path.read_text()
    payload = json.loads(decision_path.read_text())
    real = payload["retention"]["models"]["RR_REAL"]
    # the ratio printed in the report must be the artefact's ratio
    assert f"{real['valid_ratio_over_c1']:.4f}" in text
    assert payload["verdict"] in text
    assert "DEVELOPMENT_TEST" in text
    assert (C.ROUND_DIR / "DECISION.md").is_file()


def test_run_remote_references_this_round() -> None:
    text = C.ROUND_DIR.joinpath("run_remote.sh").read_text()
    assert C.ROUND_ID in text
    assert "RESEARCH_EXTRA_RUNNERS=rp_runner" in text
    assert "lcrc_relation_interface_routing_pilot_v1/code" in text
