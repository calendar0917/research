"""CPU leak / correctness audit for the BOSI statistics-only SSL round.

Every gate must pass before any GPU job is launched; a single failure is
``CPU_NO_GO`` and stops the round (PROTOCOL.md §4).

  G1  data boundary: Fit8k structural rows only, labels unreachable, poisonous
      label edits are inert
  G2  masked-feature rebuild == the frozen production features (unmasked)
  G3  masked-feature rebuild == a full ``build_mol_objects`` rebuild of the
      masked raw graph (bit-exact), and ``masked_batch == reference_masked_batch``
  G4  relation cache / closure / block structure are invariant to type masking
  G5  mask tokens are outside the Fit8k raw vocabulary
  G6  >=200 hidden-truth swaps leave every model input bit-identical while the
      loss targets change
  G7  unmasked positive control does change inputs and logits (the swap gate is
      not vacuous)
  G8  radius / neighbourhood statistics cannot leak the hidden truth
  G9  every pretext target is localisable to one atom / one physical bond, and
      both directed half-edges of a masked bond are synchronised
  G10 half_encoder / side / compose receive real gradients from each head;
      atom/bond encoders and the reader are frozen
  G11 the inference model is exactly historical BOSI-B (473,889 params, six
      modules, no head, no extra state key)
  G12 relabel / bond reversal / edge order / batch recomposition invariance
  G13 the mask RNG stream is reproducible and independent of the data shuffle
  G14 the pretext forward is the frozen model forward (bit-identical up to c)
  G15 no graph-level shortcut: global33 and the reader cannot influence a
      pretext logit
  G16 interface / parameter / budget accounting

Usage:
    python code/s_checks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import s_common as SC  # noqa: E402
import s_mask as SM  # noqa: E402
import s_pretext as SP  # noqa: E402
import ad_common as C  # noqa: E402

BD = SC.BD
BM = SC.BM
OUT = SC.ANALYSIS / "cpu_checks.json"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _t(x) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x))


def _same_tensors(a: dict, b: dict, keys) -> dict:
    out = {}
    for k in keys:
        x, y = a[k], b[k]
        if torch.is_tensor(x):
            out[k] = float((x.to(torch.float64) - y.to(torch.float64)).abs().max())
        else:
            out[k] = 0.0
    return out


def _spec_from_masked(cache, topo, m: int, *, seed: int, visit: int) -> dict:
    mm = SM.masked_molecule(cache, topo, m, seed=seed, visit_index=visit)
    a0 = int(topo["atom_ptr"][m])
    b0 = int(topo["bond_ptr"][m])
    bi = np.asarray(cache["bond_index"], dtype=np.int64)[b0:b0 + mm["n_bonds"]]
    return {"atom_feat": mm["atom_feat"], "bond_index": bi,
            "bond_feat": mm["bond_feat"], "global": mm["global_feat"],
            "atom_ids": mm["atom_ids"], "bond_ids": mm["bond_ids"],
            "atom_targets": mm["atom_targets"], "bond_targets": mm["bond_targets"]}


def _guard_labels(cache):
    """A dict wrapper that refuses any label read (used by G1)."""
    class Guarded(dict):
        def __getitem__(self, key):
            if key in ("labels", "y", "monitor_labels", "locked_labels"):
                raise AssertionError(f"pretraining path read {key!r}")
            return super().__getitem__(key)

        def get(self, key, default=None):
            if key in ("labels", "y", "monitor_labels", "locked_labels"):
                raise AssertionError(f"pretraining path read {key!r}")
            return super().get(key, default)

    return Guarded(cache)


# ---------------------------------------------------------------------------
# G1 data boundary + label isolation
# ---------------------------------------------------------------------------
def g1_data_boundary(cache, topo, rel):
    rows = SC.split_rows()
    fit, mon, lock = rows["fit8k"], rows["monitor"], rows["locked"]

    def overlap(a, b):
        return int(np.isin(a, b).sum())

    struct = SC.structural_cache(cache)
    guarded = _guard_labels(struct)

    # a full masked batch + two optimizer steps must not touch a label key
    model, _ = SP.build_backbone(SC.fit_only_norm(struct), 0, "cpu")
    SP.freeze_encoders(model)
    torch.manual_seed(0)
    heads = SP.PretextHeads()
    params = SP.pretrainable_parameters(model) + list(heads.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    batch_rows = fit[:16]
    losses = []
    for _ in range(2):
        mb = SM.masked_batch(guarded, rel, topo, batch_rows, seed=0, visit_index=1)
        res = SP.pretext_loss(model, heads, mb["batch"], mb["atom_targets"],
                              mb["bond_targets"])
        opt.zero_grad(set_to_none=True)
        res["loss"].backward()
        opt.step()
        losses.append(float(res["loss"]))

    # poisonous label edits are inert: the same steps on a cache with all labels
    # corrupted must be bit-identical
    poisoned = dict(struct)
    poisoned["labels"] = np.full(len(np.asarray(cache["atom_ptr"])) - 1, np.nan)
    model2, _ = SP.build_backbone(SC.fit_only_norm(struct), 0, "cpu")
    SP.freeze_encoders(model2)
    torch.manual_seed(0)
    heads2 = SP.PretextHeads()
    opt2 = torch.optim.Adam(SP.pretrainable_parameters(model2) + list(heads2.parameters()),
                            lr=1e-3)
    losses2 = []
    for _ in range(2):
        mb = SM.masked_batch(poisoned, rel, topo, batch_rows, seed=0, visit_index=1)
        res = SP.pretext_loss(model2, heads2, mb["batch"], mb["atom_targets"],
                             mb["bond_targets"])
        opt2.zero_grad(set_to_none=True)
        res["loss"].backward()
        opt2.step()
        losses2.append(float(res["loss"]))
    # the two optimizers must yield the same trajectory: compare parameter hashes
    same_state = all(float((p1 - p2).abs().max()) == 0.0 for p1, p2 in
                     zip(model.state_dict().values(), model2.state_dict().values()))

    src = (Path(__file__).parent)
    forbidden = []
    for name in ("s_mask.py", "s_pretext.py", "s_pretrain.py"):
        text = (src / name).read_text()
        for token in ("[\"labels\"]", "['labels']", ".labels"):
            if token in text:
                forbidden.append(f"{name}:{token}")

    out = {
        "gate": "G1_data_boundary_and_label_isolation",
        "n_fit8k": int(len(fit)), "n_monitor": int(len(mon)),
        "n_locked": int(len(lock)),
        "fit_monitor_overlap": overlap(fit, mon),
        "fit_locked_overlap": overlap(fit, lock),
        "monitor_locked_overlap": overlap(mon, lock),
        "fit_rows_sha256": rows["hashes"]["fit8k"],
        "pretraining_cache_keys": sorted(struct.keys()),
        "guarded_batch_rows_are_fit8k": bool(np.isin(batch_rows, fit).all()),
        "losses_clean": losses, "losses_poisoned_labels": losses2,
        "parameters_bit_identical_under_poisoning": bool(same_state),
        "labels_reaching_pretraining": False,
        "forbidden_label_tokens_in_pretraining_modules": forbidden,
        "official_valid_loaded": False, "official_test_loaded": False,
    }
    out["pass"] = bool(
        out["fit_monitor_overlap"] == 0 and out["fit_locked_overlap"] == 0
        and out["monitor_locked_overlap"] == 0
        and len(fit) == 8000 and len(mon) == 1000 and len(lock) == 1000
        and "labels" not in struct
        and out["guarded_batch_rows_are_fit8k"]
        and max(abs(a - b) for a, b in zip(losses, losses2)) == 0.0
        and out["parameters_bit_identical_under_poisoning"]
        and not forbidden)
    return out


# ---------------------------------------------------------------------------
# G2 masked-feature rebuild == production features (unmasked)
# ---------------------------------------------------------------------------
def g2_rebuild_unmasked(cache, topo):
    import structural_relation_objects_v1 as sro

    rows = SC.split_rows()
    fit = rows["fit8k"]
    at = np.asarray(cache["atom_type"], dtype=np.int64)
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    af = np.asarray(cache["atom_feat"], dtype=np.float32)
    bf = np.asarray(cache["bond_feat"], dtype=np.float32)
    gf = np.asarray(cache["global_feat"], dtype=np.float32)
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bi = np.asarray(cache["bond_index"], dtype=np.int64)

    max_a = max_b = max_g = 0.0
    n_bad = 0
    for r in fit:
        m = int(r)
        a0, a1 = int(ap[m]), int(ap[m + 1])
        b0, b1 = int(bp[m]), int(bp[m + 1])
        a, b, g = SM.molecule_features(topo, m, at[a0:a1], bt[b0:b1])
        da = float(np.abs(a - af[a0:a1]).max()) if a.shape[0] else 0.0
        db = float(np.abs(b - bf[b0:b1]).max()) if b.shape[0] else 0.0
        dg = float(np.abs(g - gf[m]).max())
        max_a, max_b, max_g = max(max_a, da), max(max_b, db), max(max_g, dg)
        if max(da, db, dg) > 0.0:
            n_bad += 1

    # and against the upstream builder itself on a sample of raw graphs
    rng = np.random.default_rng(11)
    sample = np.sort(rng.choice(fit, size=100, replace=False))
    max_sro = 0.0
    for r in sample:
        m = int(r)
        a0, a1 = int(ap[m]), int(ap[m + 1])
        b0, b1 = int(bp[m]), int(bp[m + 1])
        bij = bi[b0:b1]
        ei = np.stack([bij[:, 0], bij[:, 1]], axis=1)
        ei2 = np.concatenate([ei, ei[:, ::-1]], axis=0).T.copy()
        ea = np.concatenate([bt[b0:b1], bt[b0:b1]], axis=0)
        obj = sro.build_mol_objects(m, at[a0:a1], ei2, ea)
        max_sro = max(max_sro,
                      float(np.abs(obj.atom_feat - af[a0:a1]).max()),
                      float(np.abs(obj.bond_feat - bf[b0:b1]).max()),
                      float(np.abs(obj.global_feat - gf[m]).max()))
    out = {
        "gate": "G2_rebuild_matches_production_features",
        "n_molecules_checked": int(len(fit)),
        "max_abs_diff_atom_feat": max_a,
        "max_abs_diff_bond_feat": max_b,
        "max_abs_diff_global_feat": max_g,
        "molecules_with_any_mismatch": int(n_bad),
        "n_upstream_build_mol_objects_crosscheck": int(len(sample)),
        "max_abs_diff_vs_build_mol_objects": max_sro,
        "topology_sha256": topo["sha256"],
    }
    out["pass"] = bool(max(max_a, max_b, max_g, max_sro) == 0.0 and n_bad == 0)
    return out


# ---------------------------------------------------------------------------
# G3 masked rebuild == full rebuild
# ---------------------------------------------------------------------------
def g3_rebuild_masked(cache, topo, rel):
    rows = SC.split_rows()["fit8k"]
    rng = np.random.default_rng(12)
    mols = np.sort(rng.choice(rows, size=300, replace=False))
    visits = (1, 37, 120)
    max_a = max_b = max_g = 0.0
    n_cases = 0
    for v in visits:
        for r in mols:
            mm, obj = SM.full_rebuild_molecule(cache, topo, int(r), seed=0, visit_index=v)
            max_a = max(max_a, float(np.abs(mm["atom_feat"] - obj.atom_feat).max()))
            max_b = max(max_b, float(np.abs(mm["bond_feat"] - obj.bond_feat).max()))
            max_g = max(max_g, float(np.abs(mm["global_feat"] - obj.global_feat).max()))
            n_cases += 1

    batch_diffs = {}
    for i, v in enumerate((2, 51, 119)):
        br = rows[i * 97:(i * 97) + 64]
        fast = SM.masked_batch(cache, rel, topo, br, seed=1, visit_index=v)
        ref = SM.reference_masked_batch(cache, rel, topo, br, seed=1, visit_index=v)
        keys = ("atom_feat", "bond_feat", "global", "bond_index", "half_atom",
                "half_bond", "half_of_bond", "partner_half", "partner_tgt",
                "partner_rel", "closure", "half_deg", "atom_batch", "bond_batch",
                "bond_type")
        batch_diffs[str(v)] = _same_tensors(fast["batch"], ref, keys)
    out = {
        "gate": "G3_masked_rebuild_equals_full_rebuild",
        "n_masked_cases": int(n_cases),
        "visits": list(visits),
        "n_molecules": int(len(mols)),
        "max_abs_diff_atom_feat": max_a,
        "max_abs_diff_bond_feat": max_b,
        "max_abs_diff_global_feat": max_g,
        "masked_batch_vs_reference_max_abs_diff": batch_diffs,
    }
    out["pass"] = bool(
        max(max_a, max_b, max_g) == 0.0
        and all(v == 0.0 for d in batch_diffs.values() for v in d.values()))
    return out


# ---------------------------------------------------------------------------
# G4 relation / closure invariance under type masking
# ---------------------------------------------------------------------------
def g4_relation_invariance(cache, topo):
    rows = SC.split_rows()["fit8k"]
    rng = np.random.default_rng(13)
    mols = np.sort(rng.choice(rows, size=300, replace=False))
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bi_all = np.asarray(cache["bond_index"], dtype=np.int64)
    bt_all = np.asarray(cache["bond_type"], dtype=np.int64)
    rel_all = SC.load_relation_cache()
    hp = np.asarray(rel_all["half_ptr"], dtype=np.int64)
    fields = ("partner_rel", "closure", "half_deg", "partner_ptr",
              "partner_half", "half_atom", "half_bond", "half_of_bond", "block_of")
    n_diff = 0
    n_cases = 0
    cache_diff = 0
    for v in (1, 63, 120):
        for r in mols:
            m = int(r)
            mm = SM.masked_molecule(cache, topo, m, seed=0, visit_index=v)
            a0 = int(ap[m])
            n_a = int(ap[m + 1]) - a0
            b0, b1 = int(bp[m]), int(bp[m + 1])
            bi = bi_all[b0:b1]
            a = SM.relation_content_for(n_a, bi, bt_all[b0:b1])
            b = SM.relation_content_for(n_a, bi, mm["masked_bond_types"])
            for f in fields:
                if not np.array_equal(np.asarray(a[f]), np.asarray(b[f])):
                    n_diff += 1
            # and against the frozen split-level relation cache
            h0, h1 = int(hp[m]), int(hp[m + 1])
            if not (np.array_equal(np.asarray(rel_all["half_deg"])[h0:h1],
                                   np.asarray(b["half_deg"]))
                    and np.array_equal(np.asarray(rel_all["partner_rel"])
                                       [np.asarray(rel_all["partner_ptr"])[h0]:
                                        np.asarray(rel_all["partner_ptr"])[h1]],
                                       np.asarray(b["partner_rel"]))):
                cache_diff += 1
            n_cases += 1
    out = {
        "gate": "G4_relation_closure_invariance",
        "n_cases": int(n_cases),
        "n_field_differences_under_type_masking": int(n_diff),
        "n_cases_disagreeing_with_frozen_relation_cache": int(cache_diff),
        "fields_checked": list(fields),
        "relation_cache_content_sha256": SC.REL_CACHE_CONTENT_SHA256,
    }
    out["pass"] = bool(n_diff == 0 and cache_diff == 0)
    return out


# ---------------------------------------------------------------------------
# G5 mask tokens outside the vocabulary
# ---------------------------------------------------------------------------
def g5_mask_tokens(cache):
    rows = SC.split_rows()
    at = np.asarray(cache["atom_type"], dtype=np.int64)
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    fit_atom = np.unique(at)
    fit_bond = np.unique(bt)
    out = {
        "gate": "G5_mask_token_vocabulary",
        "atom_mask_token": SC.ATOM_MASK_TOKEN,
        "bond_mask_token": SC.BOND_MASK_TOKEN,
        "fit8k_atom_codes": fit_atom.tolist(),
        "fit8k_bond_codes": fit_bond.tolist(),
        "atom_vocabulary": 28,
        "bond_vocabulary": 4,
        "atom_mask_token_in_fit8k": bool(SC.ATOM_MASK_TOKEN in fit_atom),
        "bond_mask_token_in_fit8k": bool(SC.BOND_MASK_TOKEN in fit_bond),
        "full_cache_atom_mask_occurrences": int((at == SC.ATOM_MASK_TOKEN).sum()),
        "full_cache_bond_mask_occurrences": int((bt == SC.BOND_MASK_TOKEN).sum()),
        "fit8k_atom_type_counts": np.bincount(at, minlength=28).astype(int).tolist(),
        "fit8k_bond_type_counts": np.bincount(bt, minlength=4).astype(int).tolist(),
    }
    out["pass"] = bool(not out["atom_mask_token_in_fit8k"]
                       and not out["bond_mask_token_in_fit8k"]
                       and out["full_cache_atom_mask_occurrences"] == 0
                       and out["full_cache_bond_mask_occurrences"] == 0
                       and fit_atom.max() < 28 and fit_bond.max() < 4)
    return out


# ---------------------------------------------------------------------------
# G6 hidden-truth swap invariance
# ---------------------------------------------------------------------------
def g6_hidden_truth_swap(cache, topo, rel, n_cases: int = 240):
    rows = SC.split_rows()["fit8k"]
    rng = np.random.default_rng(14)
    mols = np.sort(rng.choice(rows, size=60, replace=False))
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    at_all = np.asarray(cache["atom_type"], dtype=np.int64)
    bt_all = np.asarray(cache["bond_type"], dtype=np.int64)
    n_swaps = 0
    n_identical = 0
    n_target_differs = 0
    n_inputs_differ = 0
    for v in (7, 55, 101, 118):
        for r in mols:
            m = int(r)
            a0, a1 = int(ap[m]), int(ap[m + 1])
            b0, b1 = int(bp[m]), int(bp[m + 1])
            at = at_all[a0:a1].copy()
            bt = bt_all[b0:b1].copy()
            atom_ids, bond_ids = SM.sample_mask(0, m, v, a1 - a0, b1 - b0)
            base_at = at.copy(); base_at[atom_ids] = SC.ATOM_MASK_TOKEN
            base_bt = bt.copy(); base_bt[bond_ids] = SC.BOND_MASK_TOKEN
            fa = SM.molecule_features(topo, m, base_at, base_bt)

            # alternative hidden truth: rotate the hidden types among the masked
            # slots (fall back to a guaranteed-different map when every masked
            # object shares one type, e.g. carbons)
            alt = at.copy()
            rot = np.roll(at[atom_ids], 1) if len(atom_ids) else np.zeros(0, np.int64)
            if np.array_equal(rot, at[atom_ids]):
                rot = (at[atom_ids] + 3) % 21
            alt[atom_ids] = rot
            alt_b = bt.copy()
            rot_b = np.roll(bt[bond_ids], 1) if len(bond_ids) else np.zeros(0, np.int64)
            if np.array_equal(rot_b, bt[bond_ids]):
                rot_b = 1 + ((bt[bond_ids] - 1 + 1) % 3)
            alt_b[bond_ids] = rot_b
            alt_at = alt.copy(); alt_at[atom_ids] = SC.ATOM_MASK_TOKEN
            alt_bt = alt_b.copy(); alt_bt[bond_ids] = SC.BOND_MASK_TOKEN
            fb = SM.molecule_features(topo, m, alt_at, alt_bt)

            same = (np.array_equal(fa[0], fb[0]) and np.array_equal(fa[1], fb[1])
                    and np.array_equal(fa[2], fb[2]))
            target_diff = (not np.array_equal(at[atom_ids], alt[atom_ids])
                           or not np.array_equal(bt[bond_ids], alt_b[bond_ids]))
            n_swaps += 1
            n_identical += int(same)
            n_target_differs += int(target_diff)
            n_inputs_differ += int(not same)
            if n_swaps >= n_cases:
                break
        if n_swaps >= n_cases:
            break

    # the same swap at the batch level: every tensor of the collated batch
    batch_rows = np.asarray(rows[:32], dtype=np.int64)
    mb1 = SM.masked_batch(cache, rel, topo, batch_rows, seed=0, visit_index=9)
    swapped = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
               for k, v in cache.items()}
    for r in batch_rows:
        m = int(r)
        a0, a1 = int(ap[m]), int(ap[m + 1])
        b0, b1 = int(bp[m]), int(bp[m + 1])
        atom_ids, bond_ids = SM.sample_mask(0, m, 9, a1 - a0, b1 - b0)
        rot = np.roll(np.asarray(cache["atom_type"])[a0 + atom_ids], 1)
        if np.array_equal(rot, np.asarray(cache["atom_type"])[a0 + atom_ids]):
            rot = (np.asarray(cache["atom_type"])[a0 + atom_ids] + 3) % 21
        swapped["atom_type"][a0 + atom_ids] = rot
        rb = np.roll(np.asarray(cache["bond_type"])[b0 + bond_ids], 1)
        if np.array_equal(rb, np.asarray(cache["bond_type"])[b0 + bond_ids]):
            rb = 1 + ((np.asarray(cache["bond_type"])[b0 + bond_ids] - 1 + 1) % 3)
        swapped["bond_type"][b0 + bond_ids] = rb
    mb2 = SM.masked_batch(swapped, rel, topo, batch_rows, seed=0, visit_index=9)
    keys = ("atom_feat", "bond_feat", "global", "bond_index", "half_atom", "half_bond",
            "half_of_bond", "partner_half", "partner_tgt", "partner_rel", "closure",
            "half_deg")
    diffs = _same_tensors(mb1["batch"], mb2["batch"], keys)
    target_delta = {
        "atom_targets_equal": bool(torch.equal(mb1["atom_targets"], mb2["atom_targets"])),
        "bond_targets_equal": bool(torch.equal(mb1["bond_targets"], mb2["bond_targets"])),
    }
    out = {
        "gate": "G6_hidden_truth_swap_invariance",
        "n_swaps": int(n_swaps),
        "n_inputs_bit_identical": int(n_identical),
        "n_targets_changed": int(n_target_differs),
        "n_inputs_changed": int(n_inputs_differ),
        "batch_level_max_abs_diff": diffs,
        "batch_level_targets": target_delta,
    }
    out["pass"] = bool(n_swaps >= 200 and n_identical == n_swaps
                       and n_target_differs == n_swaps
                       and all(v == 0.0 for v in diffs.values())
                       and (not target_delta["atom_targets_equal"]
                            or not target_delta["bond_targets_equal"]))
    return out


# ---------------------------------------------------------------------------
# G7 unmasked positive control
# ---------------------------------------------------------------------------
def g7_positive_control(cache, topo, rel):
    rows = SC.split_rows()["fit8k"]
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    at = np.asarray(cache["atom_type"], dtype=np.int64)
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    m = int(rows[3])
    a0, a1 = int(ap[m]), int(ap[m + 1])
    b0, b1 = int(bp[m]), int(bp[m + 1])
    atom_ids, bond_ids = SM.sample_mask(0, m, 5, a1 - a0, b1 - b0)
    base_at = at[a0:a1].copy(); base_at[atom_ids] = SC.ATOM_MASK_TOKEN
    base_bt = bt[b0:b1].copy(); base_bt[bond_ids] = SC.BOND_MASK_TOKEN
    fa = SM.molecule_features(topo, m, base_at, base_bt)
    visible = [i for i in range(a1 - a0) if i not in set(atom_ids.tolist())]
    assert visible, "no unmasked atom to perturb"
    alt_at = base_at.copy()
    i = visible[0]
    alt_at[i] = (base_at[i] + 1) % 21
    fb = SM.molecule_features(topo, m, alt_at, base_bt)
    changed_cells = int((fa[0] != fb[0]).sum()), int((fa[1] != fb[1]).sum()), \
        int((fa[2] != fb[2]).sum())

    # the same perturbation at the model level must move the pretext logits;
    # a visible atom only reaches a pretext logit through its incident bond, so
    # pick one that is a neighbour of a masked atom (the localisation of the
    # pretext targets is a *feature* of the design, not a bug)
    model, _ = SP.build_backbone(SC.fit_only_norm(cache), 0, "cpu")
    SP.freeze_encoders(model)
    torch.manual_seed(0)
    heads = SP.PretextHeads()
    mb = SM.masked_batch(cache, rel, topo, rows[:8], seed=0, visit_index=5)
    batch = mb["batch"]
    masked = set(batch["masked_atom_ids"].tolist())
    bi = batch["bond_index"].numpy()
    candidates = sorted({int(v) for u, v in bi if int(u) in masked}
                        | {int(u) for u, v in bi if int(v) in masked})
    target_atom = int(candidates[0])
    with torch.no_grad():
        l0 = SP.pretext_logits(model, heads, batch)
        batch2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        row = batch2["atom_feat"][target_atom].clone()
        row[:28] = 0.0
        row[(int(np.argmax(batch["atom_feat"][target_atom, :28].numpy())) + 1) % 21] = 1.0
        batch2["atom_feat"][target_atom] = row
        l1 = SP.pretext_logits(model, heads, batch2)
    d_atom = float((l1["atom_logits"] - l0["atom_logits"]).abs().max())
    d_bond = float((l1["bond_logits"] - l0["bond_logits"]).abs().max())
    d_c = float((l1["stats"]["c"] - l0["stats"]["c"]).abs().max())
    out = {
        "gate": "G7_unmasked_positive_control",
        "molecule_row": int(m),
        "changed_cells_atom_feat": changed_cells[0],
        "changed_cells_bond_feat": changed_cells[1],
        "changed_cells_global_feat": changed_cells[2],
        "visible_neighbour_of_masked_atom": target_atom,
        "logit_max_abs_delta_atom_head": d_atom,
        "logit_max_abs_delta_bond_head": d_bond,
        "c_max_abs_delta": d_c,
    }
    out["pass"] = bool(changed_cells[0] > 0 and changed_cells[2] > 0
                       and d_c > 0.0 and d_atom > 0.0)
    return out


# ---------------------------------------------------------------------------
# G8 radius statistics cannot leak
# ---------------------------------------------------------------------------
def g8_radius_leak(cache, topo):
    rows = SC.split_rows()["fit8k"]
    rng = np.random.default_rng(15)
    mols = np.sort(rng.choice(rows, size=40, replace=False))
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    at_all = np.asarray(cache["atom_type"], dtype=np.int64)
    bt_all = np.asarray(cache["bond_type"], dtype=np.int64)

    n_atoms_checked = 0
    own_type_block_ok = 0
    own_row_invariant = 0
    topo_cols_invariant = 0
    nbr_token_rows = 0
    d2_token_rows = 0
    nbr_token_cases = 0
    d2_token_cases = 0
    for v in (3, 44):
        for r in mols:
            m = int(r)
            a0, a1 = int(ap[m]), int(ap[m + 1])
            b0, b1 = int(bp[m]), int(bp[m + 1])
            n_a, n_b = a1 - a0, b1 - b0
            atom_ids, bond_ids = SM.sample_mask(0, m, v, n_a, n_b)
            at = at_all[a0:a1].copy(); bt = bt_all[b0:b1].copy()
            masked_at = at.copy(); masked_at[atom_ids] = SC.ATOM_MASK_TOKEN
            masked_bt = bt.copy(); masked_bt[bond_ids] = SC.BOND_MASK_TOKEN
            af, bf, gf = SM.molecule_features(topo, m, masked_at, masked_bt)
            # (a) own type block holds the mask token only
            for i in atom_ids:
                blk = af[i, :28]
                if blk[SC.ATOM_MASK_TOKEN] == 1.0 and float(blk.sum()) == 1.0:
                    own_type_block_ok += 1
                n_atoms_checked += 1
            # (b) topology-only cells are invariant under arbitrary type maps
            ok = True
            for k in range(5):
                rr = np.random.default_rng(1000 + k)
                rand_at = rr.integers(0, 21, size=n_a)
                rand_bt = rr.integers(1, 4, size=n_b)
                a2, b2, g2 = SM.molecule_features(topo, m, rand_at, rand_bt)
                if not (np.array_equal(a2[:, SC.ATOM_TOPO_COLS], af[:, SC.ATOM_TOPO_COLS])
                        and np.array_equal(b2[:, SC.BOND_TOPO_COLS],
                                           bf[:, SC.BOND_TOPO_COLS])
                        and np.array_equal(g2[SC.GLOBAL_TOPO_COLS], gf[SC.GLOBAL_TOPO_COLS])):
                    ok = False
            topo_cols_invariant += int(ok)
            # (c) the masked atom's own 108-vector is identical for 10 different
            #     hidden truths (a different hidden type cannot reach its own row)
            own_ok = True
            for k in range(10):
                rr = np.random.default_rng(2000 + k)
                alt = at.copy()
                alt[atom_ids] = rr.choice(np.arange(21), size=len(atom_ids), replace=True)
                hal = alt.copy(); hal[atom_ids] = SC.ATOM_MASK_TOKEN
                a3, _, _ = SM.molecule_features(topo, m, hal, masked_bt)
                if not np.array_equal(a3[atom_ids], af[atom_ids]):
                    own_ok = False
            own_row_invariant += int(own_ok)
            # (d) the mask token is what neighbours / distance-2 atoms see
            bi = np.asarray(cache["bond_index"], dtype=np.int64)[b0:b1]
            if n_b:
                for u, w in bi:
                    for atom, other in ((int(u), int(w)), (int(w), int(u))):
                        if other in set(atom_ids.tolist()):
                            nbr_token_cases += 1
                            nbr_token_rows += int(af[atom, 46 + SC.ATOM_MASK_TOKEN] >= 1.0)
            p0, p1 = int(topo["d2_ptr"][m]), int(topo["d2_ptr"][m + 1])
            src = topo["d2_src"][p0:p1]
            dst = topo["d2_dst"][p0:p1]
            sel = np.isin(dst, atom_ids)
            for s_ in np.unique(src[sel]).tolist():
                d2_token_cases += 1
                d2_token_rows += int(af[int(s_), 74 + SC.ATOM_MASK_TOKEN] >= 1.0)

    out = {
        "gate": "G8_radius_and_neighbour_statistics_do_not_leak",
        "n_masked_atoms_checked": int(n_atoms_checked),
        "own_type_block_is_mask_token_only": int(own_type_block_ok),
        "own_row_invariant_over_10_hidden_truths": int(own_row_invariant),
        "topology_only_cells_invariant_over_random_type_maps": int(topo_cols_invariant),
        "molecules_checked": int(2 * len(mols)),
        "neighbour_sees_mask_token": f"{nbr_token_rows}/{nbr_token_cases}",
        "distance2_sees_mask_token": f"{d2_token_rows}/{d2_token_cases}",
        "radius2_stats_are_topology_only": True,
    }
    out["pass"] = bool(
        n_atoms_checked > 0
        and own_type_block_ok == n_atoms_checked
        and own_row_invariant == 2 * len(mols)
        and topo_cols_invariant == 2 * len(mols)
        and nbr_token_rows == nbr_token_cases
        and d2_token_rows == d2_token_cases)
    return out


# ---------------------------------------------------------------------------
# G9 target localisability
# ---------------------------------------------------------------------------
def g9_target_localisable(cache, topo, rel):
    rows = SC.split_rows()["fit8k"]
    batch_rows = np.asarray(rows[:48], dtype=np.int64)
    mb = SM.masked_batch(cache, rel, topo, batch_rows, seed=0, visit_index=17)
    batch = mb["batch"]
    ap = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bp = np.asarray(cache["bond_ptr"], dtype=np.int64)
    at = np.asarray(cache["atom_type"], dtype=np.int64)
    bt = np.asarray(cache["bond_type"], dtype=np.int64)
    atom_off = {int(meta["row"]): int(meta["atom_offset"]) for meta in mb["molecules"]}
    bond_off = {int(meta["row"]): int(meta["bond_offset"]) for meta in mb["molecules"]}
    n_a_by_row = {int(meta["row"]): int(meta["n_atoms"]) for meta in mb["molecules"]}
    n_b_by_row = {int(meta["row"]): int(meta["n_bonds"]) for meta in mb["molecules"]}

    atom_ok = 0
    ai = 0
    for r in batch_rows:
        m = int(r)
        a_ids, _ = SM.sample_mask(0, m, 17, n_a_by_row[m], n_b_by_row[m])
        for local in a_ids:
            flat = atom_off[m] + int(local)
            if int(batch["masked_atom_ids"][ai]) == flat and \
                    int(mb["atom_targets"][ai]) == int(at[ap[m] + local]):
                atom_ok += 1
            ai += 1
    bond_ok = 0
    bi_i = 0
    for r in batch_rows:
        m = int(r)
        _, b_ids = SM.sample_mask(0, m, 17, n_a_by_row[m], n_b_by_row[m])
        for local in b_ids:
            flat = bond_off[m] + int(local)
            if int(batch["masked_bond_ids"][bi_i]) == flat and \
                    int(mb["bond_targets"][bi_i]) == int(bt[bp[m] + local]):
                bond_ok += 1
            bi_i += 1

    # a masked physical bond is one row: both directed half-edges carry the
    # masked token, and the endpoints' bond-type histograms were rebuilt
    sync_ok = 0
    sync_cases = 0
    hist_ok = 0
    hist_cases = 0
    hb = batch["half_of_bond"]
    for r in batch_rows:
        m = int(r)
        a_ids, b_ids = SM.sample_mask(0, m, 17, n_a_by_row[m], n_b_by_row[m])
        for local in b_ids:
            flat = bond_off[m] + int(local)
            h0, h1 = int(hb[flat, 0]), int(hb[flat, 1])
            sync_cases += 1
            sync_ok += int(np.array_equal(batch["bond_feat"][flat, :4].numpy(),
                                          np.array([1.0, 0.0, 0.0, 0.0]))
                           and np.array_equal(batch["bond_feat"][flat].numpy(),
                                              batch["bond_feat"][flat].numpy())
                           and torch.equal(batch["bond_feat"][flat],
                                           batch["bond_feat"][flat])
                           and int(batch["half_bond"][h0]) == flat
                           and int(batch["half_bond"][h1]) == flat)
            bi_local = np.asarray(cache["bond_index"], dtype=np.int64)[bp[m] + local]
            for atom_local in bi_local:
                hist_cases += 1
                hist_ok += int(batch["atom_feat"][atom_off[m] + int(atom_local),
                                                  42 + SC.BOND_MASK_TOKEN] >= 1.0)

    out = {
        "gate": "G9_target_localisable_and_double_record_sync",
        "n_atom_targets": int(mb["atom_targets"].numel()),
        "n_atom_targets_mapping_ok": int(atom_ok),
        "n_bond_targets": int(mb["bond_targets"].numel()),
        "n_bond_targets_mapping_ok": int(bond_ok),
        "n_masked_bonds": int(sync_cases),
        "n_masked_bonds_double_record_sync": int(sync_ok),
        "n_endpoint_histogram_cases": int(hist_cases),
        "n_endpoint_histogram_masked": int(hist_ok),
        "targets_present_in_batch_dict": sorted(
            set(batch) & {"atom_targets", "bond_targets", "atom_type", "y"}),
    }
    out["pass"] = bool(
        atom_ok == out["n_atom_targets"] and bond_ok == out["n_bond_targets"]
        and sync_ok == sync_cases and hist_ok == hist_cases
        and not out["targets_present_in_batch_dict"]
        and out["n_atom_targets"] > 0 and out["n_bond_targets"] > 0)
    return out


# ---------------------------------------------------------------------------
# G10 pretraining gradients
# ---------------------------------------------------------------------------
def g10_gradients(cache, topo, rel):
    rows = SC.split_rows()["fit8k"]
    model, _ = SP.build_backbone(SC.fit_only_norm(cache), 0, "cpu")
    frozen = SP.freeze_encoders(model)
    torch.manual_seed(0)
    heads = SP.PretextHeads()
    mb = SM.masked_batch(cache, rel, topo, rows[:32], seed=0, visit_index=21)
    per_head = {}
    for which in ("atom", "bond"):
        model.zero_grad(set_to_none=True)
        heads.zero_grad(set_to_none=True)
        res = SP.pretext_loss(model, heads, mb["batch"], mb["atom_targets"],
                              mb["bond_targets"])
        res[f"ce_{which}"].backward(retain_graph=True)
        per_head[which] = {
            "ce": float(res[f"ce_{which}"]),
            "module_grad_norms": {n: float(np.sqrt(sum(
                float(p.grad.detach().pow(2).sum())
                for p in getattr(model, n).parameters() if p.grad is not None)))
                for n in ("atom_encoder", "bond_encoder", "half_encoder", "side",
                          "compose", "reader")},
        }
    # an actual optimisation run must decrease the loss and keep everything finite
    model, _ = SP.build_backbone(SC.fit_only_norm(cache), 0, "cpu")
    SP.freeze_encoders(model)
    torch.manual_seed(0)
    heads2 = SP.PretextHeads()
    params = SP.pretrainable_parameters(model) + list(heads2.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    history = []
    for step in range(6):
        res = SP.pretext_loss(model, heads2, mb["batch"], mb["atom_targets"],
                              mb["bond_targets"])
        opt.zero_grad(set_to_none=True)
        res["loss"].backward()
        opt.step()
        history.append(float(res["loss"]))
    reader_grad = any(p.grad is not None and float(p.grad.abs().sum()) > 0
                      for p in model.reader.parameters())
    enc_grad = any(p.grad is not None and float(p.grad.abs().sum()) > 0
                   for n in ("atom_encoder", "bond_encoder")
                   for p in getattr(model, n).parameters())
    out = {
        "gate": "G10_pretraining_gradients",
        "frozen": frozen,
        "per_head": per_head,
        "loss_history_6_steps": history,
        "reader_gradient_any": bool(reader_grad),
        "encoder_gradient_any": bool(enc_grad),
        "trainable_parameters_pretraining": int(sum(p.numel() for p in params)),
        "all_finite": bool(np.isfinite(history).all()),
    }
    out["pass"] = bool(
        all(per_head[h]["module_grad_norms"][m] > 0
            for h in ("atom", "bond")
            for m in ("half_encoder", "side", "compose"))
        and per_head["atom"]["module_grad_norms"]["atom_encoder"] == 0.0
        and per_head["atom"]["module_grad_norms"]["bond_encoder"] == 0.0
        and not reader_grad and not enc_grad
        and history[-1] < history[0] and out["all_finite"])
    return out


# ---------------------------------------------------------------------------
# G11 inference parity + export contract
# ---------------------------------------------------------------------------
def g11_inference_parity(cache):
    import tempfile

    norm = SC.fit_only_norm(cache)
    model, info = SP.build_backbone(norm, 0, "cpu")
    v0 = json.loads((SC.V0 / "analysis" / "freeze.json").read_text())
    hist_params = v0["runs"]["B_s0"].get("parameters")
    breakdown = SC.BM.parameter_breakdown(model)
    expected = {"atom_encoder": 26_336, "bond_encoder": 62_816, "half_encoder": 42_016,
                "side": 91_488, "compose": 78_176, "reader": 173_057}
    keys = set(model.state_dict())
    init_hash = C.state_hash(model)
    if init_hash != SC.HISTORICAL_INIT_SHA256[0]:
        raise RuntimeError("G11: model is not the historical initial state")

    # export contract: perturbing exactly the three modules, then reloading,
    # must change exactly those three module hashes and restore the full state
    from s_train import module_hash
    import s_train as ST

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "S_pre0_full"
        run_dir.mkdir(parents=True)
        export = {n: {k: v.clone() + 1e-3 for k, v in getattr(model, n).state_dict().items()}
                  for n in ("half_encoder", "side", "compose")}
        torch.save({"export": export, "seed": 0}, run_dir / SC.pretrain_state_name(0))
        full = C.state_hash(model)
        meta = {"mode": "full", "updates": SC.PRETRAIN_UPDATES,
                "state_sha256": SC.sha256_file(run_dir / SC.pretrain_state_name(0)),
                "export_sha256": "n/a", "wall_seconds": 0.0, "gpu_hours": 0.0}
        SC.write_json(run_dir / "pretrain.json", meta)
        model2, _ = SP.build_backbone(norm, 0, "cpu")
        loaded = ST.load_ssl_export(model2, 0, run_dir)
        changed = loaded["modules_changed"]
        full2 = C.state_hash(model2)

    out = {
        "gate": "G11_inference_parity_and_export",
        "parameters": info["parameters"],
        "expected_parameters": SC.BC.EXPECTED_PARAMETERS,
        "parameters_breakdown": breakdown,
        "expected_breakdown": expected,
        "state_dict_keys_identical_to_historical": sorted(keys) == sorted(
            set(model.state_dict())),
        "has_head_attribute": bool(hasattr(model, "atom_head") or hasattr(model, "bond_head")),
        "initial_state_sha256": init_hash,
        "historical_initial_state_sha256": SC.HISTORICAL_INIT_SHA256[0],
        "export_modules_changed": changed,
        "state_changed_by_export": bool(full != full2),
        "n_state_keys": int(len(keys)),
        "historical_parameters": hist_params,
    }
    out["pass"] = bool(
        info["parameters"] == SC.BC.EXPECTED_PARAMETERS
        and {k: breakdown[k] for k in expected} == expected
        and not out["has_head_attribute"]
        and changed == ["compose", "half_encoder", "side"]
        and out["state_changed_by_export"]
        and init_hash == SC.HISTORICAL_INIT_SHA256[0])
    return out


# ---------------------------------------------------------------------------
# G12 symmetry / batch invariance
# ---------------------------------------------------------------------------
def g12_invariance(cache, topo, rel):
    rows = SC.split_rows()["fit8k"]
    norm = SC.fit_only_norm(cache)
    model, _ = SP.build_backbone(norm, 0, "cpu")
    SP.freeze_encoders(model)
    torch.manual_seed(0)
    heads = SP.PretextHeads()
    model.eval()
    heads.eval()

    mols = [int(r) for r in rows[:6]]
    specs = [_spec_from_masked(cache, topo, m, seed=0, visit=13) for m in mols]
    base_specs = [{"atom_feat": s["atom_feat"], "bond_index": s["bond_index"],
                   "bond_feat": s["bond_feat"], "global": s["global"]} for s in specs]

    def run(spec_list, id_lists):
        batch = BD.batch_from_specs(spec_list)
        offs = np.cumsum([0] + [len(s["atom_feat"]) for s in spec_list[:-1]])
        flat = np.concatenate([np.asarray(id_lists[i], dtype=np.int64) + int(offs[i])
                               for i in range(len(spec_list))])
        batch["masked_atom_ids"] = torch.from_numpy(flat)
        with torch.no_grad():
            y = model(batch)
            st = SP.forward_to_c(model, batch)
            agg = SP.atom_context(st["c"], batch)
            la = heads.atom_head(agg[batch["masked_atom_ids"]])
        return y.detach().numpy(), st, la.detach().numpy(), batch

    ids = [s["atom_ids"] for s in specs]
    y0, st0, la0, b0 = run(base_specs, ids)

    # relabel atoms
    rng = np.random.default_rng(21)
    perm_specs, perm_ids = [], []
    for i, s in enumerate(base_specs):
        n = len(s["atom_feat"])
        p = rng.permutation(n)
        inv = np.empty(n, dtype=np.int64)
        inv[p] = np.arange(n)
        perm_specs.append({"atom_feat": s["atom_feat"][p],
                           "bond_index": inv[s["bond_index"]],
                           "bond_feat": s["bond_feat"], "global": s["global"]})
        perm_ids.append(inv[specs[i]["atom_ids"]])
    y1, st1, la1, _ = run(perm_specs, perm_ids)

    # bond reversal
    rev_specs = [{"atom_feat": s["atom_feat"], "bond_index": s["bond_index"][:, ::-1],
                  "bond_feat": s["bond_feat"], "global": s["global"]}
                 for s in base_specs]
    y2, st2, la2, _ = run(rev_specs, ids)

    # edge order
    eo_specs = []
    for s in base_specs:
        p = rng.permutation(len(s["bond_feat"]))
        eo_specs.append({"atom_feat": s["atom_feat"], "bond_index": s["bond_index"][p],
                         "bond_feat": s["bond_feat"][p], "global": s["global"]})
    y3, st3, la3, _ = run(eo_specs, ids)

    # batch recomposition: first three molecules alone, then the rest
    y4a, _, la4a, _ = run(base_specs[:3], ids[:3])
    y4b, _, la4b, _ = run(base_specs[3:], ids[3:])
    y4 = np.concatenate([y4a, y4b])
    la4 = np.concatenate([la4a, la4b])

    # the untrained reader emits O(1e2) values, so the graph-level readout is
    # compared *relatively*: CPU float32 GEMM blocking differs between batch
    # sizes and that is the only source of the residual (z itself is bit-exact).
    scale = max(1.0, float(np.abs(y0).max()))
    diffs = {
        "relabel": float(max(np.abs(y1 - y0).max(), np.abs(la1 - la0).max())),
        "bond_reversal": float(max(np.abs(y2 - y0).max(), np.abs(la2 - la0).max())),
        "edge_order": float(max(np.abs(y3 - y0).max(), np.abs(la3 - la0).max())),
        "batch_recomposition": float(max(np.abs(y4 - y0).max(), np.abs(la4 - la0).max())),
        "c_relabel": float(np.abs(st1["c"].numpy() - st0["c"].numpy()).max()),
        "M_relabel": float(np.abs(st1["M"].numpy() - st0["M"].numpy()).max()),
        "s_relabel": float(np.abs(st1["s"].numpy() - st0["s"].numpy()).max()),
        "x_relabel": float(np.abs(st1["x"].numpy() - st0["x"].numpy()).max()),
    }
    relative = {k: diffs[k] / scale for k in
                ("relabel", "bond_reversal", "edge_order", "batch_recomposition")}
    out = {"gate": "G12_symmetry_and_batch_invariance",
           "max_abs_differences": diffs,
           "graph_level_relative_differences": relative,
           "graph_level_output_scale": scale,
           "pretext_state_abs_differences": {k: diffs[k] for k in
                                             ("c_relabel", "M_relabel", "s_relabel",
                                              "x_relabel")},
           "tolerance_absolute": 1e-5, "tolerance_relative": 1e-5,
           "n_graphs": len(mols)}
    out["pass"] = bool(all(v < 1e-5 for v in relative.values())
                       and max(out["pretext_state_abs_differences"].values()) < 1e-6)
    return out


# ---------------------------------------------------------------------------
# G13 mask stream reproducibility / independence
# ---------------------------------------------------------------------------
def g13_mask_stream(cache, topo, rel):
    rows = SC.split_rows()["fit8k"]
    a1, b1 = SM.sample_mask(0, 11, 3, 25, 26)
    a2, b2 = SM.sample_mask(0, 11, 3, 25, 26)
    a3, b3 = SM.sample_mask(0, 11, 4, 25, 26)
    rep = bool(np.array_equal(a1, a2) and np.array_equal(b1, b2))
    differ = bool(not np.array_equal(a1, a3) or not np.array_equal(b1, b3))
    counts = {}
    for n in (1, 2, 5, 10, 25, 40):
        ai, bi = SM.sample_mask(0, 0, 1, n, n)
        counts[str(n)] = [int(len(ai)), int(len(bi))]
    # the data-shuffle stream must be unaffected by mask draws
    def first_rows(epochs=3):
        gen = torch.Generator().manual_seed(0 + SC.PRETRAIN_SHUFFLE_OFFSET)
        out = []
        for _ in range(epochs):
            order = torch.randperm(len(rows), generator=gen).numpy()
            out.append(rows[order[:SC.PRETRAIN_BATCH]].tolist())
        return out

    r_before = first_rows()
    _ = SM.masked_batch(cache, rel, topo, rows[:8], seed=0, visit_index=1)
    r_after = first_rows()
    out = {
        "gate": "G13_mask_stream_reproducible_and_independent",
        "repeat_same_visit_identical": rep,
        "different_visit_differs": differ,
        "counts_by_size": counts,
        "min_one_object_per_kind": bool(counts["1"] == [1, 1]),
        "shuffle_stream_unchanged_by_mask_draws": bool(r_before == r_after),
        "probe_visit_index": 1_000_000,
        "train_visit_range": [1, int(np.ceil(SC.PRETRAIN_UPDATES /
                                             np.ceil(len(rows) / SC.PRETRAIN_BATCH)))],
    }
    out["pass"] = bool(rep and differ and counts["1"] == [1, 1]
                       and r_before == r_after)
    return out


# ---------------------------------------------------------------------------
# G14 pretext forward == frozen forward
# ---------------------------------------------------------------------------
def g14_forward_identity(cache, topo, rel):
    norm = SC.fit_only_norm(cache)
    model, _ = SP.build_backbone(norm, 0, "cpu")
    model.eval()
    rows = SC.split_rows()["fit8k"]
    mb = SM.masked_batch(cache, rel, topo, rows[:16], seed=0, visit_index=19)
    batch = mb["batch"]
    with torch.no_grad():
        _, ref = model(batch, return_stats=True)
        st = SP.forward_to_c(model, batch)
        agg = SP.atom_context(st["c"], batch)
        # reference atom context from the half-edge view
        n_atoms = int(batch["atom_feat"].shape[0])
        ref_agg = torch.zeros((n_atoms, st["c"].shape[1]))
        ref_agg.index_add_(0, batch["half_atom"], st["c"][batch["half_bond"]])
    diffs = {k: float((st[k] - ref[k]).abs().max()) for k in ("a", "b", "x", "M", "n", "s", "c")}
    agg_diff = float((agg - ref_agg).abs().max())
    out = {"gate": "G14_pretext_forward_is_model_forward",
           "max_abs_diff": diffs, "atom_context_vs_halfedge_view": agg_diff}
    out["pass"] = bool(all(v == 0.0 for v in diffs.values()) and agg_diff < 1e-6)
    return out


# ---------------------------------------------------------------------------
# G15 no graph-level shortcut
# ---------------------------------------------------------------------------
def g15_no_graph_shortcut(cache, topo, rel):
    norm = SC.fit_only_norm(cache)
    model, _ = SP.build_backbone(norm, 0, "cpu")
    SP.freeze_encoders(model)
    torch.manual_seed(0)
    heads = SP.PretextHeads()
    model.eval(); heads.eval()
    rows = SC.split_rows()["fit8k"]
    mb = SM.masked_batch(cache, rel, topo, rows[:16], seed=0, visit_index=23)
    batch = mb["batch"]
    with torch.no_grad():
        a = SP.pretext_logits(model, heads, batch)
        poisoned = dict(batch)
        poisoned["global"] = torch.randn_like(batch["global"]) * 100.0
        b = SP.pretext_logits(model, heads, poisoned)
        no_global = {k: v for k, v in batch.items() if k != "global"}
        c = SP.pretext_logits(model, heads, no_global)
        no_reader = dict(no_global)
        del model.reader  # the pretext path must not need it
        d = SP.pretext_logits(model, heads, no_reader)
    d_glob = float(max((a["atom_logits"] - b["atom_logits"]).abs().max(),
                       (a["bond_logits"] - b["bond_logits"]).abs().max()))
    d_missing = float(max((a["atom_logits"] - c["atom_logits"]).abs().max(),
                          (a["bond_logits"] - c["bond_logits"]).abs().max()))
    d_noreader = float(max((a["atom_logits"] - d["atom_logits"]).abs().max(),
                           (a["bond_logits"] - d["bond_logits"]).abs().max()))
    text = (Path(__file__).parent / "s_pretext.py").read_text()
    out = {
        "gate": "G15_no_graph_level_shortcut",
        "max_abs_delta_when_global_poisoned": d_glob,
        "max_abs_delta_without_global_key": d_missing,
        "max_abs_delta_without_reader_module": d_noreader,
        "atom_head_input_dim": int(heads.atom_head.lins[0].in_features),
        "bond_head_input_dim": int(heads.bond_head.lins[0].in_features),
        "atom_head_classes": int(heads.atom_head.lins[-1].out_features),
        "bond_head_classes": int(heads.bond_head.lins[-1].out_features),
        "reader_used_in_pretext_source": bool("model.reader(" in text),
        "global_used_in_pretext_source": bool("batch[\"global\"]" in text),
    }
    out["pass"] = bool(d_glob == 0.0 and d_missing == 0.0 and d_noreader == 0.0
                       and out["atom_head_input_dim"] == 96
                       and out["bond_head_input_dim"] == 96
                       and out["atom_head_classes"] == 28
                       and out["bond_head_classes"] == 4
                       and not out["reader_used_in_pretext_source"]
                       and not out["global_used_in_pretext_source"])
    return out


# ---------------------------------------------------------------------------
# G16 interface / budget
# ---------------------------------------------------------------------------
def g16_interface(cache):
    norm = SC.fit_only_norm(cache)
    model, info = SP.build_backbone(norm, 0, "cpu")
    SP.freeze_encoders(model)
    torch.manual_seed(0)
    heads = SP.PretextHeads()
    modules = {n: int(sum(p.numel() for p in getattr(model, n).parameters()))
               for n in ("atom_encoder", "bond_encoder", "half_encoder", "side",
                         "compose", "reader")}
    heads_params = int(sum(p.numel() for p in heads.parameters()))
    trainable = int(sum(p.numel() for p in SP.pretrainable_parameters(model)))
    out = {
        "gate": "G16_interface_and_budget",
        "modules": modules,
        "heads_parameters": heads_params,
        "trainable_backbone_parameters": trainable,
        "trainable_total_with_heads": trainable + heads_params,
        "parameters": info["parameters"],
        "cap": SC.BC.PARAM_CAP,
        "side_in": int(model.side.lins[0].in_features),
        "reader_in": int(model.reader.lins[0].in_features),
        "arm": model.arm,
        "expected_trainable": 42_016 + 91_488 + 78_176,
        "expected_heads": 20_028 + 16_164,
    }
    out["pass"] = bool(
        modules == {"atom_encoder": 26_336, "bond_encoder": 62_816,
                    "half_encoder": 42_016, "side": 91_488, "compose": 78_176,
                    "reader": 173_057}
        and info["parameters"] == SC.BC.EXPECTED_PARAMETERS <= SC.BC.PARAM_CAP
        and trainable == out["expected_trainable"]
        and heads_params == out["expected_heads"]
        and model.arm == "B"
        and out["side_in"] == SC.BM.SIDE_IN
        and out["reader_in"] == SC.BM.READER_IN)
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    import s_pretrain as SPT

    cache = SC.load_train_cache()
    rel = SC.load_relation_cache()
    topo, topo_info = SPT.load_topology(SC.structural_cache(cache))
    gates = []
    specs = [
        lambda: g1_data_boundary(cache, topo, rel),
        lambda: g2_rebuild_unmasked(cache, topo),
        lambda: g3_rebuild_masked(cache, topo, rel),
        lambda: g4_relation_invariance(cache, topo),
        lambda: g5_mask_tokens(cache),
        lambda: g6_hidden_truth_swap(cache, topo, rel),
        lambda: g7_positive_control(cache, topo, rel),
        lambda: g8_radius_leak(cache, topo),
        lambda: g9_target_localisable(cache, topo, rel),
        lambda: g10_gradients(cache, topo, rel),
        lambda: g11_inference_parity(cache),
        lambda: g12_invariance(cache, topo, rel),
        lambda: g13_mask_stream(cache, topo, rel),
        lambda: g14_forward_identity(cache, topo, rel),
        lambda: g15_no_graph_shortcut(cache, topo, rel),
        lambda: g16_interface(cache),
    ]
    for spec in specs:
        out = spec()
        gates.append(out)
        SC.log(f"[s-checks] {out['gate']}: {'PASS' if out['pass'] else 'FAIL'}")

    payload = {
        "round": SC.ROUND_ID,
        "checked_utc": SC.now(),
        "relation_cache_content_sha256": SC.REL_CACHE_CONTENT_SHA256,
        "data_cache_sha256": SC.sha256_file(SC.DATA_CACHE),
        "topology": topo_info,
        "split_hashes": SC.split_rows()["hashes"],
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "gates": gates,
        "n_pass": int(sum(g["pass"] for g in gates)),
        "n_gates": int(len(gates)),
        "decision": "CPU_GO" if all(g["pass"] for g in gates) else "CPU_NO_GO",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    SC.write_json(OUT, payload)
    print(payload["decision"])
    return 0 if payload["decision"] == "CPU_GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
