from __future__ import annotations


import adapters as A
import cpu_gates as G
import lcrc_common as C


def test_expected_hashes_match_frozen_sources():
    manifest = A.source_manifest()
    assert manifest["hashes"]["raw_graph"] == C.EXPECTED_HASHES["lipo_raw"]
    assert manifest["hashes"]["frozen_split"] == C.EXPECTED_HASHES["lipo_split"]
    assert manifest["hashes"]["row_manifest"] == C.EXPECTED_HASHES["row_manifest"]
    assert manifest["hashes"]["csv"] == C.EXPECTED_HASHES["lipo_csv"]
    assert manifest["label_isolation"]["label_values_parsed"] is False
    assert manifest["label_isolation"]["labels_fit_monitor_opened"] is False


def test_frozen_split_counts_hashes_and_union(split):
    assert {k: len(v) for k, v in split["split"].items()} == {"fit": 2352, "monitor": 504, "reserved": 504}
    assert split["hashes"]["fit"] == C.EXPECTED_ROW_HASHES["fit"]
    assert split["hashes"]["monitor"] == C.EXPECTED_ROW_HASHES["monitor"]
    assert split["hashes"]["reserved"] == C.EXPECTED_ROW_HASHES["reserved"]
    sets = {k: set(v) for k, v in split["split"].items()}
    assert not (sets["fit"] & sets["monitor"])
    assert not (sets["fit"] & sets["reserved"])
    assert not (sets["monitor"] & sets["reserved"])
    assert len(sets["fit"] | sets["monitor"] | sets["reserved"]) == 3360
    assert sorted(sets["fit"] | sets["monitor"] | sets["reserved"]) == split["union"]


def test_scaffold_groups_are_disjoint(row_manifest, split):
    scaffold = row_manifest["scaffold_sha256"]
    groups = {k: {scaffold[int(i)] for i in split["split"][k]} for k in ("fit", "monitor", "reserved")}
    assert len(groups["fit"]) == 725
    assert len(groups["monitor"]) == 421
    assert len(groups["reserved"]) == 422
    assert not (groups["fit"] & groups["monitor"])
    assert not (groups["fit"] & groups["reserved"])
    assert not (groups["monitor"] & groups["reserved"])


def test_official_split_reader_reads_indices_not_labels():
    official = A.load_lipo_official_row_indices()
    assert len(official["train_rows"]) == 3360
    assert len(official["valid_rows"]) == 420
    assert len(official["test_rows"]) == 420
    assert official["_audit"]["label_values_parsed"] is False
    assert set(official) >= {"train_rows", "valid_rows", "test_rows", "_audit"}


def test_row_1646_is_not_silently_in_scope(split):
    # The frozen 3360-row union is exactly the official train set.  Row 1646
    # (a 4-component ChEMBL structure) is outside this round's structure scope.
    union = set(split["union"])
    assert 1646 not in union
    manifest = A.source_manifest()
    assert manifest["split"]["row_1646_in_frozen_union"] is False
    assert manifest["split"]["row_1646_official_membership"] in {"valid", "test"}
    # Do not convert or introspect its structure: it is outside this round's
    # official-train-only structure scope.


def test_row_manifest_alignment_sample(raw_payload, split, row_manifest):
    sample = split["split"]["fit"][:64] + split["split"]["monitor"][:16] + split["split"]["reserved"][:16]
    report = G.gate_row_manifest_alignment(raw_payload["rows"], {"fit": sample, "monitor": [], "reserved": []}, row_manifest)
    # The helper expects all three keys; the sample above is routed through fit only.
    assert report["status"] in {"PASS", "BLOCKED"}
    if report["status"] == "BLOCKED":
        assert report["rdkit_available"] is False
    else:
        assert report["mismatches"] == []
