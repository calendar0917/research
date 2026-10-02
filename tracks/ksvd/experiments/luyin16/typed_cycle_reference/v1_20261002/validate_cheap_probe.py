"""Synthetic acceptance checks, not a ZINC property experiment."""
import json
from pathlib import Path
import numpy as np
from cheap_property_probe import run_probe, outer_folds
from cheap_probe_features import edge_type_counts, ring_features
from typed_cycle_reference import witness_pair, relabel


def main():
    rng = np.random.default_rng(13)
    n = 180
    base = rng.normal(size=(n, 4))
    extra = rng.normal(size=(n, 2))
    y = 0.2 * base[:, 0] + extra[:, 0] + rng.normal(scale=.03, size=n)
    groups = np.arange(n)
    positive = run_probe(base, extra, y, groups)
    null = run_probe(base, np.zeros((n, 2)), y, groups)
    assert positive['gate'] == 'BUY_ONE_TYPED_CYCLE_SCREEN'
    assert null['gain'] == 0.0
    repeated_groups = np.repeat(np.arange(60), 3)
    for fit, dev, heldout in outer_folds(repeated_groups):
        sets = [set(repeated_groups[ids]) for ids in (fit, dev, heldout)]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        assert len(fit) + len(dev) + len(heldout) == n
    a, b = witness_pair()
    ua, ta = ring_features(a)
    ub, tb = ring_features(b)
    assert np.array_equal(ua, ub)
    assert np.array_equal(edge_type_counts(a), edge_type_counts(b))
    assert not np.array_equal(ta, tb)
    for _ in range(20):
        moved = relabel(a, rng.permutation(len(a.atoms)))
        um, tm = ring_features(moved)
        assert np.array_equal(ua, um) and np.array_equal(ta, tm)
        assert np.array_equal(edge_type_counts(a), edge_type_counts(moved))
    report = {
        'scope': 'synthetic helper checks only; no ZINC labels or PyTorch training',
        'official_test_loaded': False,
        'positive_control_gain': positive['gain'],
        'null_control_gain': null['gain'],
        'group_leakage_checks': 'passed',
        'raw_typed_feature_relabeling': 'passed',
        'witness_typed_feature_L2': float(np.linalg.norm(ta-tb)),
        'all_gates_passed': True,
    }
    Path(__file__).with_name('cheap_probe_acceptance.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
