# Pre-next-round integration checkpoint (2026-09-18)

This note records what the `integration/pre-next-research-cleanup` pass merged
into the main research chain, and the provenance decisions taken. It changes no
scientific conclusion and starts no experiment.

## Starting point

* Old `main` baseline: `7372025` ("consolidate identity incremental-information
  study").
* Main research chain: `exp/bnull-sab-path-ablation` (`21db4d0`), 66 commits
  ahead of `7372025`.
* Two branches carried durable assets that were **not** on the main chain:
  * `exp/molhiv-shared-bag-patch-encoder` (base `132f100`)
  * `exp/structural-encoder-mechanism-decomposition` (base `5b19132`)
* Integration branch created from `exp/bnull-sab-path-ablation`.

## Disposition of the divergent assets

### A. `exp/molhiv-shared-bag-patch-encoder` -> **merged** (nothing superseded)

| asset | disposition |
|---|---|
| `experiments/luyin16/molhiv_shared_bag_patch_encoder.py` | merged (unique) |
| `notes/molhiv_shared_bag_patch_encoder.md` | merged (unique) |
| `records/claims/claim-molhiv-shared-bag-transfer-20260916.yaml` | merged (unique) |
| `records/decisions/decision-molhiv-shared-bag-transfer-20260916.yaml` | merged (unique) |
| `tests/test_molhiv_shared_bag_patch_encoder.py` | merged (unique) |

This is the vocabulary-free shared B-Bag transfer to OGBG-MolHIV (281,205 params;
typed identity 839,456 -> 0) with a one-shot frozen official-test read. None of it
existed on the main chain.

### B. `exp/structural-encoder-mechanism-decomposition` -> **merged / partially superseded**

| asset | disposition |
|---|---|
| `experiments/luyin16/structural_patch_encoder.py`, `zinc_shared_bag_patch_encoder.py` (B-bag code, `5b19132`) | **already on the main chain** (`5b19132` is an ancestor of `exp/bnull-sab-path-ablation`) — not re-merged |
| `experiments/.../zinc_shared_bag_patch_encoder.py` `test` stage (`terminal_test`) + `4802ba8` | merged (unique) |
| `notes/structural_encoder_mechanism_decomposition.md` | merged (unique) |
| `notes/structural_encoder_mechanism_decomposition_test_closure.md` | merged (unique) |
| `records/claims/claim-structural-encoder-mechanism-decomposition-20260915.yaml` | merged (unique) |
| `records/decisions/decision-structural-encoder-mechanism-decomposition-20260915.yaml` | merged (unique) |
| `tests/test_shared_bag_patch_encoder.py` (terminal-test additions) | merged (unique) |

So the *code* for the B-bag control was in the main chain, but its **records,
notes and the one-shot official-test closure were not** — those are durable
scientific provenance and were merged.

### C. Intentionally archived

Nothing was deleted or archived. Every divergent asset is either merged or was
already reachable from the main chain.

## Facts surfaced by the integration

### 1. A second ZINC official-test read already existed before the "test read" round

`notes/structural_encoder_mechanism_decomposition_test_closure.md` (2026-09-15,
commit `4802ba8`, tag `sbpe-test`) reports a one-shot official ZINC test read of
the **B-bag** control: 2-seed Top-5 soup test MAE **0.098742** (vs frozen cell A
0.106717). The later `notes/local_token_null_test_read.md` (2026-09-18) describes
its own B-Null result (0.097731) as "for the first time in this track, a single
model below the 0.10 target". That phrasing is not literally correct: the B-bag
closure had already reported a sub-0.10 single-model soup three days earlier.
Both records are kept as written (no conclusion changes); the honest statement
is that **two independent controls reached sub-0.10 soup, on different branches**.

The integration therefore makes the **complete set of spent official-test reads**
visible for the first time (see `STATE.yaml:official_test_reads`):

* ZINC: cell A typed (2026-09-14), B-bag control (2026-09-15), local-token-null
  B-Null/Constant-16 (2026-09-18).
* MolHIV: typed RPC (2026-09-14), shared B-Bag transfer (2026-09-16), B-Null
  closure (2026-09-18), B-Null small-head probe (2026-09-18).

### 2. Dangling decision reference repaired

The pre-integration `STATE.yaml` pointed at
`decision-zinc-cell-a-test-closure-and-molhiv-parameter-attribution-20260914`,
but **no such claim/decision file was ever committed** — the cell-A official-test
closure exists only as
`notes/compact_v4_recurrent_pair_centre_capacity_test_closure.md` plus its test
module. The rebuilt STATE now points at that note. The discrepancy is recorded
here rather than papered over with a newly-invented record.

### 3. `STATE.yaml` was an experiment-history dump

The old `STATE.yaml` was 284,498 bytes: 35 answered `open_questions`, 57
mostly-DONE `next_queue` items and full copies of 54 claims / 47 decisions.
It has been rebuilt as a small live pointer (~14 kB) that stores only IDs, paths
and one-line summaries; the scientific text stays in `records/` and `notes/`.
No record was deleted.

### 4. Future-dated durable records

16 claim/decision files carry dates later than both their introducing git commit
and the real system clock. They were **not** re-dated or renamed (that would
break the id chain and cannot be verified). They are documented in
`records/PROVENANCE_DATES.md` and enforced by `uv run research verify`.

## Verification

See the final report for the exact command results (`uv sync --frozen`,
`research doctor`, `research verify`, fast pytest subset, ruff, remote
preflight).
