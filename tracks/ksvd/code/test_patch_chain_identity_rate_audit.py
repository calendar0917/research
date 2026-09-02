"""Self-tests for patch-chain identity and bit-rate accounting."""
from __future__ import annotations

from .run_patch_chain_identity_rate_audit import classify, identity_bit_bounds


def _row(**updates: float | bool) -> dict[str, float | bool]:
    row: dict[str, float | bool] = {
        "mapped_patch_adjacency_match_rate": 1.0,
        "mapped_transition_slot_map_match_rate": 1.0,
        "resampled_exact_ordered_chain_match": 1.0,
        "resampled_patch_set_jaccard": 1.0,
        "resampled_local_vector_row_match": 1.0,
        "resampled_transition_map_match": 1.0,
        "absolute_edge_coverage_delta": 0.0,
        "absolute_pair_coverage_delta": 0.0,
        "absolute_raw_full_rmse_delta": 0.0,
        "identity_chain_lower_bits": 500.0,
        "identity_fixed_width_bits": 600.0,
        "raw_naive_lower_bits": 1000.0,
        "raw_unique_lower_bits": 1100.0,
        "direct_enumerative_exact_bits": 1150.0,
        "ksvd_break_even_bits_per_coefficient": 9.0,
        "direct_bitset_bits": 1225.0,
        "base_invariants": True,
        "relabeled_invariants": True,
    }
    row.update(updates)
    return row


def main() -> int:
    bits = identity_bit_bounds(
        n_nodes=50, patch_size=10, overlap=3, patch_count=18
    )
    assert 800 <= bits["chain_aware_lower_bits"] <= 850
    assert bits["fixed_width_ordered_membership_bits"] == 1080
    passed = classify([_row()])
    assert passed["classification"] == "PATCH_CHAIN_IDENTITY_RATE_SUPPORTED"
    failed = classify(
        [
            _row(
                resampled_exact_ordered_chain_match=0.0,
                raw_unique_lower_bits=1500.0,
                ksvd_break_even_bits_per_coefficient=3.0,
            )
        ]
    )
    assert failed["classification"] == "REVISE_PATCH_CHAIN_REPRESENTATION_CLAIM"
    assert (
        failed["strict_relabel_classification"]
        == "SAMPLER_NOT_STRICTLY_RELABEL_EQUIVARIANT"
    )
    assert (
        failed["raw_rate_classification"]
        == "RAW_CHAIN_NOT_BIT_COMPETITIVE_WITH_ADJACENCY"
    )
    print("patch_chain_identity_rate_audit self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
