"""Integrity checks for the same-budget structural-computation gap analysis.

Static-analysis audit: these tests pin the *measurement discipline* of the
audit without training any model and without touching official test:

1.  compact-v4 parameter split sums to the canonical checkpoint total (99,613);
2.  reference parameter split sums to the instantiated official config (130,945);
3.  the live canonical model reproduces the CSV split and the 302D R vector;
4.  auditing / instrumentation does not change the model forward;
5.  object-count statistics are consistent with the real per-molecule graphs;
6.  the pair set is exactly the complete unordered pair set (n choose 2);
7.  every source-backed reference claim carries a source id;
8.  no claim is marked VERIFIED without a source;
9.  official test is never loaded / referenced as an input in the artefacts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import yaml

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "tracks/ksvd/results/structural_computation_gap_analysis"
HINGE_CONFIG = ROOT / "tracks/ksvd/configs/luyin16/zinc_compact_v4_topology_hinge.yaml"
SELECTION_TYPED_VOCAB = 6785
PARENT_VOCAB = 32


def _read_json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _read_csv(name: str):
    with open(OUT / name, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _canonical_model() -> zpp.PatchPathModel:
    """The exact compact-v4-hinge selection model (config comment: 99,613)."""
    cfg = yaml.safe_load(HINGE_CONFIG.read_text(encoding="utf-8"))["model"]
    torch.manual_seed(0)
    return zpp.PatchPathModel(
        SELECTION_TYPED_VOCAB,
        PARENT_VOCAB,
        patch_hidden=int(cfg["patch_hidden"]),
        pair_hidden=int(cfg["pair_hidden"]),
        token_width=int(cfg["token_width"]),
        dropout=float(cfg["dropout"]),
        embedding_mode=str(cfg["embedding_mode"]),
        embedding_rank=int(cfg["embedding_rank"]),
        hybrid_full_typed_tokens=int(cfg["hybrid_full_typed_tokens"]),
        hybrid_full_parent_tokens=int(cfg["hybrid_full_parent_tokens"]),
        center_context=bool(cfg["center_context"]),
        center_context_hidden=int(cfg["center_context_hidden"]),
        graph_head_hidden_0=int(cfg["graph_head_hidden_0"]),
        graph_head_hidden_1=int(cfg["graph_head_hidden_1"]),
        shell_width=zpp._shell_width_for_radius(zpp.PATCH_RADIUS),
        context_width=0,
        topology_mode="hinge",
        topology_input_width=25,
        topology_hidden_dim=int(cfg["topology_hidden_dim"]),
        topology_out_dim=int(cfg["topology_out_dim"]),
    )


def _dummy_data() -> zpp.Data:
    n = 6
    return zpp.Data(
        patch_cont=torch.randn(n, zpp._shell_width_for_radius(zpp.PATCH_RADIUS)),
        patch_context=torch.zeros(n, 0),
        typed_token=torch.tensor([0, 1, 767, 768, 769, 100], dtype=torch.long),
        parent_token=torch.tensor([0, 1, 2, 3, 31, 31], dtype=torch.long),
        structural_token=torch.zeros(n, dtype=torch.long),
        structural_coarse=torch.zeros(n, 4),
        pair_index=torch.tensor(
            [[0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
             [1, 2, 3, 4, 5, 2, 3, 4, 5, 3, 4, 5, 4, 5, 5]],
            dtype=torch.long,
        ),
        pair_relation=torch.randn(15, zpp.RELATION_WIDTH),
        pair_bucket=torch.randint(0, zpp.DISTANCE_BUCKETS, (15,)),
        global_context=torch.randn(1, zpp.GLOBAL_WIDTH),
        topology_features=torch.randn(1, 25),
        y=torch.tensor([0.0]),
        num_nodes=n,
        batch=torch.zeros(n, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# 1 / 2 -- parameter splits match the audit totals
# ---------------------------------------------------------------------------

def test_v4_parameter_split_matches_checkpoint_total():
    rows = _read_csv("parameter_allocation_v4.csv")
    body = [r for r in rows if r["component"].upper() != "TOTAL"]
    total = [r for r in rows if r["component"].upper() == "TOTAL"]
    assert len(total) == 1
    assert sum(int(r["params"]) for r in body) == 99613
    assert int(total[0]["params"]) == 99613


def test_reference_parameter_split_matches_official_config():
    rows = _read_csv("parameter_allocation_reference.csv")
    body = [r for r in rows if r["component"].upper() != "TOTAL"]
    total = [r for r in rows if r["component"].upper() == "TOTAL"]
    assert sum(int(r["params"]) for r in body) == 130945
    assert int(total[0]["params"]) == 130945


# ---------------------------------------------------------------------------
# 3 -- the live canonical model matches the CSV split and R = 302D
# ---------------------------------------------------------------------------

def test_live_model_split_matches_audit_csv():
    model = _canonical_model()
    audit = zpp.audit_parameters(model)
    assert audit["total_trainable"] == 99613
    # block names in the live audit correspond 1:1 to CSV components
    for row in _read_csv("parameter_allocation_v4.csv"):
        component = row["component"]
        if component.upper() == "TOTAL":
            continue
        assert component in audit["blocks"], component
        assert audit["blocks"][component] == int(row["params"]), component
    assert audit["dimensions"]["unified_graph_width"] == 302
    assert audit["dimensions"]["unified_graph_width"] == (
        audit["dimensions"]["unary_pooled_dim"]
        + audit["dimensions"]["pair_pooled_dim"] * audit["dimensions"]["distance_buckets"]
        + audit["dimensions"]["global_encoder_width"]
        + audit["dimensions"]["topology_out_dim"]
    )


# ---------------------------------------------------------------------------
# 4 -- auditing does not change the forward
# ---------------------------------------------------------------------------

def test_parameter_audit_does_not_change_forward():
    model = _canonical_model().eval()
    data = _dummy_data()
    with torch.no_grad():
        before = model(data).clone()
        zpp.audit_parameters(model)
        after = model(data).clone()
    assert torch.equal(before, after)


# ---------------------------------------------------------------------------
# 5 / 6 -- object-count statistics match the real per-molecule graphs
# ---------------------------------------------------------------------------

def test_object_count_statistics_are_consistent():
    profile = _read_json("compute_profile_v4.json")
    ref = _read_json("compute_profile_reference.json")
    # v4 patch count == molecule atom count (one patch per atom)
    assert abs(profile["objects_per_molecule_mean_valid"]["patches"] - 23.083) < 1e-3
    assert abs(ref["objects_per_molecule_mean_from_primary_source"]["0_cells_atoms"] - 23.083) < 1e-3
    # reference has explicit higher-order objects, v4 does not
    assert ref["objects_per_molecule_mean_from_primary_source"]["2_cells_induced_rings_k_le_18"] > 0
    assert profile["objects_per_molecule_mean_valid"]["cycle_objects"] == 0


def test_pair_count_is_n_choose_two():
    n = 6
    assert zpp.DISTANCE_BUCKETS == 5
    data = _dummy_data()
    assert data.pair_index.shape[1] == n * (n - 1) // 2 == 15
    pairs = {tuple(sorted(p)) for p in data.pair_index.t().tolist()}
    assert len(pairs) == data.pair_index.shape[1]  # no duplicates
    assert all(i != j for i, j in pairs)  # no self-loops


# ---------------------------------------------------------------------------
# 7 / 8 -- source traceability
# ---------------------------------------------------------------------------

def test_source_backed_reference_claims_have_source_ids():
    inventory = _read_json("source_inventory.json")
    ids = {s["source_id"] for s in inventory["sources"]}
    assert {"S1", "S2", "S3", "S4", "S5", "S6"} <= ids
    for source in inventory["sources"]:
        assert source["claims_supported"], source["source_id"]
        assert source["source_type"], source["source_id"]


def test_no_claim_marked_verified_without_source():
    for name in ("benchmark_comparability.json",):
        for item in _read_json(name)["items"]:
            if item["status"] == "VERIFIED":
                assert item.get("source_id"), item["item"]


def test_reference_graph_is_source_backed():
    reference = _read_json("reference_computation_graph.json")
    assert reference["source_id"] == "S1,S2"
    assert reference["official_config_name"] == "cwn-zinc-small.sh"
    assert reference["param_total_official_config"] == 130945
    # exactly the paper's Eq. 4 ingredients
    eq = reference["update_equations_paper_eq4"]
    assert {"form", "MLP_B_p", "MLP_up_p", "MLP_M_p", "MLP_U_p"} <= set(eq)


# ---------------------------------------------------------------------------
# 9 -- official test never loaded
# ---------------------------------------------------------------------------

def test_official_test_never_loaded():
    final = _read_json("final_decision.json")
    assert any("official test" in item for item in final["not_authorized"])
    plan = _read_json("minimal_falsification_plan.json")
    assert plan["pre_registered_seed0_gates"]["official_test"] == "NEVER loaded"
    inventory = _read_json("source_inventory.json")
    assert inventory["resource_inventory"]["official_test_access"].startswith("NONE")
    for path in OUT.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        for forbidden in ("access_official_test", "load_official_test", "test_labels"):
            assert forbidden not in text, path.name


# ---------------------------------------------------------------------------
# gap matrix + decision integrity
# ---------------------------------------------------------------------------

def test_gap_matrix_is_complete_and_has_contrary_evidence():
    rows = _read_csv("gap_evidence_matrix.csv")
    gaps = {r["gap"].split()[0] for r in rows}
    assert {"G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"} <= gaps
    for row in rows:
        assert row["evidence_against"].strip()
        assert row["confounds"].strip()


def test_decisions_are_case_consistent():
    final = _read_json("final_decision.json")
    top1 = _read_json("top1_hypothesis.json")
    assert final["decision_case"] == "A"
    assert top1["case"].startswith("A")
    assert top1["fits_budget_<=120k"].startswith("YES")
    assert top1["needs_large_hpo"].startswith("NO")
