"""Key scientific checks for lcrc-lipo-s3-locality-public122-s0-v1.

Focused on the pieces this round actually relies on: the provable local
signature, the two structural vectors, the correction operator, the NULL3 mean
field, the frozen initialisation / batch order, and the evaluation plumbing.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND / "code"))

import common as C  # noqa: E402
import data as D  # noqa: E402
import features as F  # noqa: E402
import metrics as M  # noqa: E402
import model as MD  # noqa: E402
import signature as SIG  # noqa: E402

torch.set_num_threads(2)
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset():
    return D.load_dataset(include_test=True)


@pytest.fixture(scope="module")
def feats():
    return F.load_features()


def single_molecule_cache(cache, mol: int, perm: np.ndarray | None = None) -> dict:
    """Slice one molecule and optionally renumber its atoms (perm[new] = old)."""
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    nbr_ptr = np.asarray(cache["nbr_ptr"], dtype=np.int64)
    a0, a1 = int(atom_ptr[mol]), int(atom_ptr[mol + 1])
    b0, b1 = int(bond_ptr[mol]), int(bond_ptr[mol + 1])
    n = a1 - a0
    perm = np.arange(n, dtype=np.int64) if perm is None else np.asarray(perm, dtype=np.int64)
    if sorted(perm.tolist()) != list(range(n)):
        raise ValueError("perm must be a permutation")
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n, dtype=np.int64)

    atom_feat = np.asarray(cache["atom_feat"])[a0:a1][perm]
    atom_type = np.asarray(cache["atom_type"])[a0:a1][perm]
    atom_scalar = np.asarray(cache["atom_scalar"])[a0:a1][perm]

    old_bi = np.asarray(cache["bond_index"])[b0:b1]
    old_bf = np.asarray(cache["bond_feat"])[b0:b1]
    old_bt = np.asarray(cache["bond_type"])[b0:b1]
    new_bonds = []
    for k, (u, v) in enumerate(old_bi.tolist()):
        nu, nv = int(inv[u]), int(inv[v])
        new_bonds.append((min(nu, nv), max(nu, nv), k))
    new_bonds.sort()
    bond_index = np.asarray([[b[0], b[1]] for b in new_bonds], dtype=np.int64).reshape(-1, 2)
    order = [b[2] for b in new_bonds]
    bond_feat = old_bf[order]
    bond_type = old_bt[order]

    # S1 codes per new root
    codes = []
    nptr = []
    offset = 0
    for i in range(n):
        p = int(perm[i])
        seg = np.asarray(cache["nbr_code"])[int(nbr_ptr[a0 + p]) : int(nbr_ptr[a0 + p + 1])]
        nptr.append(offset)
        codes.extend(int(x) for x in seg.tolist())
        offset += len(seg)
    nptr.append(offset)

    # S2 records per new root
    pptr = []
    cats, ends63, ends, closures = [], [], [], []
    offset = 0
    for i in range(n):
        p = int(perm[i])
        lo, hi = int(path_ptr[a0 + p]), int(path_ptr[a0 + p + 1])
        pptr.append(offset)
        cats.extend(int(x) for x in np.asarray(cache["path_cat"])[lo:hi].tolist())
        ends63.extend(int(x) for x in np.asarray(cache["path_end63"])[lo:hi].tolist())
        ends.extend(int(inv[int(x)]) for x in np.asarray(cache["path_end"])[lo:hi].tolist())
        closures.extend(int(x) for x in np.asarray(cache["path_closure"])[lo:hi].tolist())
        offset += hi - lo
    pptr.append(offset)

    return {
        "atom_ptr": np.asarray([0, n], dtype=np.int64),
        "bond_ptr": np.asarray([0, len(order)], dtype=np.int64),
        "nbr_ptr": np.asarray(nptr, dtype=np.int64),
        "path_ptr": np.asarray(pptr, dtype=np.int64),
        "atom_feat": atom_feat,
        "atom_type": atom_type,
        "atom_scalar": atom_scalar,
        "bond_feat": bond_feat,
        "bond_type": bond_type,
        "bond_index": bond_index,
        "nbr_code": np.asarray(codes, dtype=np.int32),
        "path_cat": np.asarray(cats, dtype=np.int32),
        "path_end63": np.asarray(ends63, dtype=np.int32),
        "path_end": np.asarray(ends, dtype=np.int64),
        "path_closure": np.asarray(closures, dtype=np.int8),
        "global": np.asarray(cache["global"])[mol].reshape(1, -1),
        "rows": np.asarray([int(np.asarray(cache["rows"])[mol])], dtype=np.int64),
        "n_molecules": 1,
    }


def cache_of(dataset, split):
    return dataset[f"{split}_cache"]


def split_of(dataset, row):
    for split in C.SPLITS:
        if row in set(dataset["rows"][split]):
            return split
    raise KeyError(row)


# ---------------------------------------------------------------------------
# 1. data identity
# ---------------------------------------------------------------------------


def test_source_split_and_train_only_normalisation(dataset):
    src = C.source_with_labels("none")
    assert tuple(len(src["rows"][s]) for s in C.SPLITS) == C.SPLIT_SIZES
    for split, expected in zip(C.SPLITS, C.SPLIT_ROWS_SHA256):
        assert C.row_sha256(src["rows"][split]) == expected
    assert src["csv_sha256"] == C.LIPOP_CSV_SHA256
    assert src["split_file_sha256"] == C.SPLIT_FILE_SHA256
    assert D.frozen_spec().hash() == C.FEATURE_SPEC_HASH

    y = np.asarray(dataset["y_train"], dtype=np.float64)
    norm = dataset["target_norm"]
    assert abs(float(norm["mean"]) - float(y.mean())) < 1e-12
    assert abs(float(norm["scale"]) - float(y.std(ddof=0))) < 1e-12
    assert norm["valid_or_test_used"] is False

    train_norm = dataset["train_norm"]
    assert train_norm["valid_or_test_used"] is False
    cached = C.read_json(C.CACHE / "manifest.json")["train_norm"]
    assert C.PREP_DATA.norm_fingerprint(train_norm) == C.PREP_DATA.norm_fingerprint(cached)


def test_cache_identity_matches_baseline(dataset):
    manifest = C.read_json(C.CACHE / "manifest.json")
    for split in C.SPLITS:
        rec = manifest["splits"][split]
        assert rec["same_as_baseline_reuse_checked"] is True
        assert rec["labels_present"] is False
        assert rec["rows_sha256"] == C.row_sha256(dataset["rows"][split])
    assert manifest["target_norm"]["mean"] == pytest.approx(float(dataset["target_norm"]["mean"]), abs=0.0)


# ---------------------------------------------------------------------------
# 2. signature: identity-free, multiset, forward coverage
# ---------------------------------------------------------------------------


def test_signature_blocks_are_identity_free(dataset):
    audit = SIG.signature_field_audit()
    excluded = " ".join(audit["excluded"]).lower()
    for token in ("atom id", "path_end", "molecule id", "smiles", "split", "target"):
        assert token in excluded
    assert audit["blocks"]["A"].startswith("atom_feat[root]")
    for k in SIG.BLOCKS:
        assert k in audit["blocks"]

    # behavioural check: identity fields of the cache never enter a descriptor
    cache = cache_of(dataset, "train")
    one = single_molecule_cache(cache, 4)
    one["rows"] = np.asarray([999999], dtype=np.int64)
    one["molecule_id"] = "identity-probe"
    one["smiles"] = "C" * 7
    base = single_molecule_cache(cache, 4)
    assert SIG.molecule_digests(SIG.molecule_view(base, 0)) == SIG.molecule_digests(
        SIG.molecule_view(one, 0)
    )
    assert SIG.global_digest(SIG.molecule_view(base, 0)) == SIG.global_digest(SIG.molecule_view(one, 0))


def test_signature_invariant_under_atom_renumbering(dataset):
    cache = cache_of(dataset, "train")
    rng = np.random.default_rng(7)
    for mol in (0, 5, 11):
        view = SIG.molecule_view(cache, mol)
        digests = sorted(SIG.molecule_digests(view))
        n = int(view["n_atoms"])
        perm = rng.permutation(n)
        permuted = single_molecule_cache(cache, mol, perm)
        pview = SIG.molecule_view(permuted, 0)
        assert sorted(SIG.molecule_digests(pview)) == digests
        assert SIG.multiset_digest(SIG.molecule_digests(pview)) == SIG.multiset_digest(digests)
        assert SIG.global_bytes(pview) == SIG.global_bytes(view)


def test_same_signature_gives_same_forward_inputs(dataset, feats):
    """The five frozen pairs plus a cross-molecule signature-matched sample."""
    cache_by_split = {s: cache_of(dataset, s) for s in C.SPLITS}
    feat_by_split = {s: F.split_features(feats, s, cache_by_split[s]) for s in C.SPLITS}
    d = C.DM
    model = d.load_checkpoint_model(
        D.frozen_spec(),
        dataset["train_norm"],
        C.VARIANT,
        torch.load(C.BASELINE_BEST, map_location="cpu", weights_only=False)["state_dict"],
        DEVICE,
    )
    rows_of = dataset["rows"]
    with torch.no_grad():
        for ra, rb in C.ALIAS_PAIRS:
            sa, sb = split_of(dataset, ra), split_of(dataset, rb)
            ba = D.to_device(D.collate(cache_by_split[sa], [rows_of[sa].index(ra)], feat_by_split[sa]), DEVICE)
            bb = D.to_device(D.collate(cache_by_split[sb], [rows_of[sb].index(rb)], feat_by_split[sb]), DEVICE)
            ya, aux_a = model(ba, return_aux=True)
            yb, aux_b = model(bb, return_aux=True)
            xa = aux_a["local_input"].numpy()
            xb = aux_b["local_input"].numpy()
            ha = aux_a["h"].numpy()
            hb = aux_b["h"].numpy()
            assert xa.shape == xb.shape
            from scipy.optimize import linear_sum_assignment

            cost = np.max(np.abs(xa[:, None, :] - xb[None, :, :]), axis=2)
            ri, ci = linear_sum_assignment(cost)
            assert cost[ri, ci].max() <= C.ALIAS_PRED_TOL
            cost_h = np.max(np.abs(ha[:, None, :] - hb[None, :, :]), axis=2)
            ri, ci = linear_sum_assignment(cost_h)
            assert cost_h[ri, ci].max() <= C.ALIAS_PRED_TOL
            assert abs(float(ya) - float(yb)) <= C.ALIAS_PRED_TOL


# ---------------------------------------------------------------------------
# 3. the frozen alias audit (produced by code/p0.py)
# ---------------------------------------------------------------------------


def test_alias_audit_gate():
    audit = C.read_json(C.ANALYSIS / "alias_audit.json")
    s = audit["alias_pair_summary"]
    assert s["n_pairs"] == 5
    assert s["n_signature_multiset_equal"] == 5
    assert s["n_global42_equal"] == 5
    assert s["n_prediction_equal_within_tol"] == 5
    assert s["n_non_isomorphic_raw_graph"] == 5
    assert s["n_u3_separated"] == 4
    assert s["u3_matches_design_expectation"]
    assert s["n_u2_separated"] == 0
    rec = {tuple(sorted((r["row_a"], r["row_b"]))): r for r in audit["alias_pairs"]}
    assert rec[C.ALIAS_EXPECTED_UNSEPARATED]["u3_separated"] is False
    for pair in C.ALIAS_EXPECTED_SEPARATED:
        assert rec[pair]["u3_separated"] is True
    assert audit["forward_identity"]["sample"]["passed"] is True
    assert audit["groups"]["enumeration"]["n_local_groups"] == 87
    assert audit["groups"]["enumeration"]["n_full_groups"] == 86
    assert audit["groups"]["local_plus_global42"]["n_groups_signature_verified"] == 86
    assert audit["groups"]["non_isomorphic_alias_groups"]["n_additional_full_input_alias_pairs_beyond_the_five"] == 0

    baseline = audit["baseline_A"]
    assert baseline["checkpoint"]["best_epoch"] == C.BASELINE_EXPECTED["best_epoch"]
    assert abs(baseline["checkpoint"]["best_metric_value"] - C.BASELINE_EXPECTED["valid_rmse_raw"]) < 1e-12
    assert abs(baseline["splits"]["test"]["rmse"] - C.BASELINE_EXPECTED["test_rmse_raw"]) < 1e-6
    assert baseline["row_id_join_max_abs_diff"] <= 1e-5
    assert baseline["design_note_predictions_max_abs_diff"] <= C.ALIAS_PRED_TOL


# ---------------------------------------------------------------------------
# 4. U3 / U2 definitions
# ---------------------------------------------------------------------------


def test_u3_hand_computed_counts():
    # chain 0-1-2-3-4-5-6 : root 0 has exactly one distance-3 atom (3)
    chain = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], dtype=np.int64)
    atom_type = np.asarray([2, 2, 2, 7, 2, 2, 2], dtype=np.int64)
    dist = F.all_pairs_distances(7, chain)
    u3 = F.u3_counts(atom_type, dist)
    assert u3[0, 7] == 1 and u3[0].sum() == 1
    assert u3[3, 2] == 2  # atoms 0 and 6 are at distance 3 from atom 3
    assert u3[6, 7] == 1
    assert u3[1, 2] == 1  # atom 4 is the only distance-3 partner of atom 1
    assert u3[2].sum() == 1  # only atom 5

    # star with a 3-bond arm: 0 centre, 1,2,3 arms, 4 attached to 3
    star = np.asarray([[0, 1], [0, 2], [0, 3], [3, 4]], dtype=np.int64)
    atom_type = np.asarray([2, 7, 7, 2, 7], dtype=np.int64)
    dist = F.all_pairs_distances(5, star)
    u3 = F.u3_counts(atom_type, dist)
    assert u3[0].sum() == 0 and u3[3].sum() == 0  # nothing at distance 3
    assert u3[1, 7] == 1 and u3[2, 7] == 1  # atom 4 is at distance 3 from atoms 1 and 2
    assert u3[4, 7] == 2  # atoms 1 and 2 are both at distance 3 from atom 4

    # disconnected: 0-1-2-3 plus 4-5, distance-3 across components never exists
    disc = np.asarray([[0, 1], [1, 2], [2, 3], [4, 5]], dtype=np.int64)
    atom_type = np.asarray([2, 2, 2, 7, 2, 2], dtype=np.int64)
    dist = F.all_pairs_distances(6, disc)
    assert dist[0, 4] == -1
    u3 = F.u3_counts(atom_type, dist)
    assert u3[0, 7] == 1 and u3[0].sum() == 1
    assert u3[4].sum() == 0


def test_u2_keeps_repeated_occurrences_and_triangles(dataset):
    cache = cache_of(dataset, "train")
    atom_type = np.asarray(cache["atom_type"])
    end63 = np.asarray(cache["path_end63"])
    path_ptr = np.asarray(cache["path_ptr"])
    n_atoms = int(cache["atom_ptr"][-1])
    u2 = F.u2_counts(n_atoms, path_ptr - path_ptr[0], end63)
    # exact identity: U2 is the histogram of path_end63 // 5
    for root in range(0, 40):
        expect = np.bincount(end63[path_ptr[root] : path_ptr[root + 1]] // 5, minlength=16)
        assert np.array_equal(u2[root], expect)
    # a triangle keeps its closure occurrences (root has two non-backtracking paths
    # that end on the triangle partner, both with closure == 1)
    view = SIG.molecule_view(cache, 0)
    tri_roots = 0
    for mol in range(int(cache["n_molecules"])):
        v = SIG.molecule_view(cache, mol)
        n = int(v["n_atoms"])
        dist = F.all_pairs_distances(n, np.asarray(v["bond_index"]))
        closure = np.asarray(v["path_closure"])
        pptr = np.asarray(v["path_ptr"])
        for root in range(n):
            seg = closure[int(pptr[root]) : int(pptr[root + 1])]
            if seg.size and int(seg.sum()) > 0:
                tri_roots += 1
        if tri_roots:
            break
    assert tri_roots > 0
    # U2 counts duplicate end atoms, so a root with two paths to the same end
    # atom has count 2, not 1
    found = False
    for mol in range(int(cache["n_molecules"])):
        v = SIG.molecule_view(cache, mol)
        n = int(v["n_atoms"])
        pptr = np.asarray(v["path_ptr"])
        ends = np.asarray(v["path_end63"])
        for root in range(n):
            seg = ends[int(pptr[root]) : int(pptr[root + 1])]
            if seg.size and len(set((seg // 5).tolist())) < len(seg.tolist()):
                found = True
                break
        if found:
            break
    assert found, "expected at least one repeated end type among S2 occurrences"
    assert int(view["n_atoms"]) > 0


def test_u2_equals_independent_adjacency_enumeration(dataset):
    cache = cache_of(dataset, "train")
    for mol in (0, 3, 9):
        view = SIG.molecule_view(cache, mol)
        n = int(view["n_atoms"])
        adj = F.molecule_adjacency(n, np.asarray(view["bond_index"]))
        atom_type = np.asarray(view["atom_type"])
        expect = np.zeros((n, 16), dtype=np.int64)
        for v in range(n):
            for w in adj[v]:
                for u in adj[w]:
                    if u == v:
                        continue
                    expect[v, int(atom_type[u])] += 1
        pptr = np.asarray(view["path_ptr"])
        got = F.u2_counts(n, pptr, np.asarray(view["path_end63"]))
        assert np.array_equal(got, expect)


def test_feature_normalisation_is_train_only(feats):
    norm3 = feats["norm_s3"]
    norm2 = feats["norm_s2"]
    for norm in (norm3, norm2):
        assert norm["fit_split"] == "train"
        assert norm["log1p"] is True
        assert norm["ddof"] == 0
        assert len(norm["mean"]) == 16 and len(norm["scale"]) == 16
        assert all(float(s) > 0 for s in norm["scale"])

    cache = D.load_cache("train")
    raw = F.split_raw_vectors(cache)["u3"].astype(np.float64)
    x = np.log1p(raw)
    assert np.allclose(np.asarray(norm3["mean"]), x.mean(0), atol=1e-10)
    assert np.allclose(np.asarray(norm3["scale"]), x.std(0, ddof=0), atol=1e-10)


# ---------------------------------------------------------------------------
# 5. NULL3 operator
# ---------------------------------------------------------------------------


def test_null3_matches_naive_loop_and_never_mean_then_silu(dataset, feats):
    cache = cache_of(dataset, "train")
    f = F.split_features(feats, "train", cache)
    model, _info = MD.build_model(D.frozen_spec(), dataset["train_norm"], DEVICE, "NULL3")
    torch.manual_seed(11)
    model.s3.W2.weight.data.normal_(0.0, 0.2)
    model.s3.b2.data.normal_(0.0, 0.05)
    batch = D.to_device(D.collate(cache, list(range(6)), f), DEVICE)
    h = torch.randn(batch["atom_feat"].shape[0], 64, requires_grad=True)
    s = batch["s3"].clone().requires_grad_(True)
    group = MD._group_key(batch)

    opt = MD.null_field(model.s3, h, s, group)
    naive = MD.null_field_naive(model.s3, h, s, group)
    assert torch.allclose(opt, naive, atol=1e-5, rtol=0)

    params = [model.s3.W2.weight, model.s3.W_h.weight, model.s3.W_s.weight, h, s]
    g_opt = torch.autograd.grad(opt.sum(), params, retain_graph=True)
    g_naive = torch.autograd.grad(naive.sum(), params, retain_graph=True)
    for a, b in zip(g_opt, g_naive):
        assert torch.allclose(a, b, atol=1e-3, rtol=1e-4)

    # not the mean-vector shortcut
    wrong = torch.zeros_like(opt)
    for g in torch.unique(group).tolist():
        idx = torch.nonzero(group == g, as_tuple=False).view(-1)
        mean_s = s[idx].mean(0, keepdim=True).expand(idx.numel(), s.shape[1])
        wrong[idx] = model.s3(h[idx], mean_s)
    assert float((opt - wrong).abs().max()) > 1e-3


def test_null3_degenerates_to_local3(dataset, feats):
    cache = cache_of(dataset, "train")
    f = F.split_features(feats, "train", cache)
    model, _info = MD.build_model(D.frozen_spec(), dataset["train_norm"], DEVICE, "NULL3")
    torch.manual_seed(13)
    model.s3.W2.weight.data.normal_(0.0, 0.2)
    batch = D.to_device(D.collate(cache, list(range(4)), f), DEVICE)
    h = torch.randn(batch["atom_feat"].shape[0], 64)
    s = batch["s3"].clone()
    group = torch.arange(h.shape[0], device=DEVICE)  # every group is a singleton
    assert torch.allclose(
        MD.null_field(model.s3, h, s, group), model.s3(h, s), atol=1e-6, rtol=0
    )
    # an all-equal group is the same operator up to float32 accumulation order
    s_equal = s.clone()
    s_equal[:] = s_equal[0]
    group_two = torch.zeros(h.shape[0], dtype=torch.long, device=DEVICE)
    assert torch.allclose(
        MD.null_field(model.s3, h, s_equal, group_two), model.s3(h, s_equal), atol=1e-5, rtol=1e-5
    )


def test_null3_does_not_mix_molecules(dataset, feats):
    cache = cache_of(dataset, "train")
    f = F.split_features(feats, "train", cache)
    model, _info = MD.build_model(D.frozen_spec(), dataset["train_norm"], DEVICE, "NULL3")
    torch.manual_seed(17)
    model.s3.W2.weight.data.normal_(0.0, 0.2)
    batch = D.to_device(D.collate(cache, [0, 1], f), DEVICE)
    h = torch.randn(batch["atom_feat"].shape[0], 64)
    s = batch["s3"].clone()
    group = MD._group_key(batch)
    # perturbing molecule 1's s vectors must not change molecule 0's correction
    n0 = int(batch["atom_counts"][0])
    s2 = s.clone()
    s2[n0:] = s2[n0:] + 1.0
    out_a = MD.null_field(model.s3, h, s, group)
    out_b = MD.null_field(model.s3, h, s2, group)
    assert torch.equal(out_a[:n0], out_b[:n0])
    assert not torch.equal(out_a[n0:], out_b[n0:])


# ---------------------------------------------------------------------------
# 6. model / initialisation / RNG
# ---------------------------------------------------------------------------


def test_parameter_counts_and_shared_initialisation(dataset):
    spec = D.frozen_spec()
    norm = dataset["train_norm"]
    infos = {}
    for arm in C.ARMS:
        model, info = MD.build_model(spec, norm, DEVICE, arm)
        assert MD.parameter_count(model) == C.EXPECTED_PARAMS == 233378
        assert info["base_parameters"] == C.BASE_PARAMS == 228674
        assert info["corrector_parameters"] == C.CORRECTOR_PARAMS == 4704
        assert info["base_shared_state_equal"]["all_equal"]
        assert info["init_state_hash"].startswith(C.BASELINE_EXPECTED["init_state_hash_prefix"])
        zeros = {
            "W2": float(model.s3.W2.weight.abs().sum()),
            "b2": float(model.s3.b2.abs().sum()),
        }
        assert zeros["W2"] == 0.0 and zeros["b2"] == 0.0
        infos[arm] = info
    assert len({infos[arm]["init_state_hash"] for arm in C.ARMS}) == 1
    assert len({infos[arm]["corrector_state_hash"] for arm in C.ARMS}) == 1


def test_zero_corrector_reproduces_c1(dataset, feats):
    spec = D.frozen_spec()
    norm = dataset["train_norm"]
    cache = cache_of(dataset, "train")
    f = F.split_features(feats, "train", cache)
    model, _ = MD.build_model(spec, norm, DEVICE, "LOCAL3")
    base = C.DM.DissectLCRC(spec, norm, C.VARIANT)
    base.load_state_dict(MD.base_state_dict(model), strict=True)
    base.eval()
    model.eval()
    batch = D.to_device(D.collate(cache, [0, 1, 2, 3, 4], f), DEVICE)
    with torch.no_grad():
        assert float((base(batch) - model(batch)).abs().max()) == 0.0


def test_gradient_reaches_corrector_in_the_expected_order(dataset, feats):
    spec = D.frozen_spec()
    norm = dataset["train_norm"]
    cache = cache_of(dataset, "train")
    f = F.split_features(feats, "train", cache)
    y = np.asarray(dataset["y_train"], dtype=np.float64)
    z = torch.as_tensor(
        (y[:4] - float(dataset["target_norm"]["mean"])) / float(dataset["target_norm"]["scale"]), dtype=torch.float32
    )
    model, _ = MD.build_model(spec, norm, DEVICE, "LOCAL3")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
    batch = D.to_device(D.collate(cache, [0, 1, 2, 3], f), DEVICE)
    loss = ((model(batch).reshape(-1) - z) ** 2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    assert float(model.s3.W2.weight.grad.abs().sum()) > 0.0  # output projection moves at step 1
    assert float(model.s3.W_s.weight.grad.abs().sum()) == 0.0  # W2 == 0 blocks it at step 1
    assert float(model.s3.W_h.weight.grad.abs().sum()) == 0.0
    opt.step()
    loss2 = ((model(batch).reshape(-1) - z) ** 2).mean()
    opt.zero_grad(set_to_none=True)
    loss2.backward()
    assert float(model.s3.W_s.weight.grad.abs().sum()) > 0.0
    assert float(model.s3.W_h.weight.grad.abs().sum()) > 0.0
    assert float(model.s3.b2.grad.abs().sum()) > 0.0


def test_batch_order_and_rng_identity(dataset):
    manifest = C.read_json(C.ANALYSIS / "batch_order_manifest.json")
    assert manifest["shuffle_generator_seed"] == 917000
    assert manifest["baseline_manifest_match"] is True
    assert manifest["sha256"] == C.read_json(C.BASELINE_ROUND / "analysis" / "schedule_manifest.json")["batch_order"]["sha256"]
    ref = C.read_json(C.ANALYSIS / "baseline_reference.json")
    assert len(ref["rng"]["epoch1_step_hashes"]) == 27


# ---------------------------------------------------------------------------
# 7. invariance: renumbering, batching, dropout
# ---------------------------------------------------------------------------


def test_forward_invariance(dataset, feats):
    spec = D.frozen_spec()
    norm = dataset["train_norm"]
    cache = cache_of(dataset, "train")
    f = F.split_features(feats, "train", cache)
    model, _ = MD.build_model(spec, norm, DEVICE, "LOCAL3")
    torch.manual_seed(21)
    model.s3.W2.weight.data.normal_(0.0, 0.1)
    model.eval()

    drops = {"n": 0}
    model.drop.register_forward_hook(lambda *_: drops.__setitem__("n", drops["n"] + 1))

    mol = 7
    n = int(np.asarray(cache["atom_ptr"])[mol + 1] - np.asarray(cache["atom_ptr"])[mol])
    rng = np.random.default_rng(3)
    perm = rng.permutation(n)
    one = single_molecule_cache(cache, mol, None)
    one_perm = single_molecule_cache(cache, mol, perm)
    f_one = {
        "rows": one["rows"],
        "atom_ptr": one["atom_ptr"],
        "s3": f["s3"][int(np.asarray(cache["atom_ptr"])[mol]) : int(np.asarray(cache["atom_ptr"])[mol + 1])],
        "s2": f["s2"][int(np.asarray(cache["atom_ptr"])[mol]) : int(np.asarray(cache["atom_ptr"])[mol + 1])],
    }
    perm_idx = int(np.asarray(cache["atom_ptr"])[mol]) + perm
    f_perm = {
        "rows": one_perm["rows"],
        "atom_ptr": one_perm["atom_ptr"],
        "s3": f["s3"][perm_idx],
        "s2": f["s2"][perm_idx],
    }
    b1 = D.to_device(D.collate(one, [0], f_one), DEVICE)
    b2 = D.to_device(D.collate(one_perm, [0], f_perm), DEVICE)
    with torch.no_grad():
        y1 = model(b1)
        y2 = model(b2)
        assert abs(float(y1) - float(y2)) <= 1e-5

        # single vs batch, and batch order
        rows = [3, mol, 5, 9]
        full = D.to_device(D.collate(cache, rows, f), DEVICE)
        yb = model(full).reshape(-1)
        permuted_batch = D.to_device(D.collate(cache, rows[::-1], f), DEVICE)
        yb_rev = model(permuted_batch).reshape(-1)
        assert torch.allclose(yb, yb_rev.flip(0), atol=1e-6, rtol=0)
        single = []
        for r in rows:
            b = D.to_device(D.collate(cache, [r], f), DEVICE)
            single.append(float(model(b)))
        assert np.allclose(single, yb.numpy(), atol=1e-5)
    assert drops["n"] == 3 + 1 + len(rows)  # invariance forwards + batch + singles


def test_checkpoint_roundtrip(dataset, feats):
    spec = D.frozen_spec()
    norm = dataset["train_norm"]
    model, info = MD.build_model(spec, norm, DEVICE, "LOCAL3")
    torch.manual_seed(31)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ckpt = {"state_dict": state, "arm": "LOCAL3", "best_epoch": 1}
    reloaded = MD.load_trained_model(spec, norm, ckpt, DEVICE)
    cache = cache_of(dataset, "valid")
    f = F.split_features(feats, "valid", cache)
    a = D.predict_scores(model, cache, DEVICE, batch_size=64, feats=f)
    b = D.predict_scores(reloaded, cache, DEVICE, batch_size=64, feats=f)
    assert np.allclose(a, b, atol=1e-6)
    assert MD.parameter_count(reloaded) == C.EXPECTED_PARAMS


# ---------------------------------------------------------------------------
# 8. evaluation plumbing
# ---------------------------------------------------------------------------


def test_evaluation_components_and_raw_units(dataset):
    ref = C.read_json(C.ANALYSIS / "baseline_reference.json")
    scale = float(dataset["target_norm"]["scale"])
    mean = float(dataset["target_norm"]["mean"])
    rows = dataset["rows"]["test"]
    y = np.asarray(dataset["y_test"], dtype=np.float64)
    assert ref["replication"]["splits"]["test"]["n"] == 420
    # raw conversion happens exactly once: standardised -> raw -> metric
    std = (y - mean) / scale
    once = std * scale + mean
    assert np.allclose(once, y, atol=1e-12)
    assert abs(M.regression_metrics(once, y)["rmse"]) < 1e-12
    assert len(rows) == 420

    comp = M.build_eval_components(
        rows=dataset["rows"]["train"] + dataset["rows"]["valid"] + dataset["rows"]["test"],
        baseline_component={int(r): int(r) for r in dataset["rows"]["train"] + dataset["rows"]["valid"] + dataset["rows"]["test"]},
        same_input_groups=[C.ALIAS_PAIRS[0]],
    )
    assert comp[int(C.ALIAS_PAIRS[0][0])] == comp[int(C.ALIAS_PAIRS[0][1])]
    assert comp[int(C.ALIAS_PAIRS[0][0])] != comp[int(C.ALIAS_PAIRS[1][0])]

    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    pred = {"72": y + 1.0}
    with np.errstate(all="ignore"):
        out = M.bootstrap_split(
            predictions=pred,
            rows=rows,
            y=y,
            component=comp,
            rng=rng,
            replicates=50,
        )
    assert out["72"]["rmse"].shape == (50,)


def test_alias_case_labels_are_never_filtered():
    audit = C.read_json(C.ANALYSIS / "alias_audit.json")
    rows = {r["row_a"] for r in audit["alias_pairs"]} | {r["row_b"] for r in audit["alias_pairs"]}
    assert rows == {294, 1665, 748, 913, 963, 4186, 1467, 2674, 2102, 3823}
    same_label = [r for r in audit["alias_pairs"] if abs(r["y_diff"]) == 0.0]
    assert len(same_label) == 1 and (same_label[0]["row_a"], same_label[0]["row_b"]) == (748, 913)
