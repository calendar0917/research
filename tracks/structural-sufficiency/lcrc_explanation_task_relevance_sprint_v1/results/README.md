# Raw worker outputs

The two formal workers wrote 1646 endpoint records each (823 pairs × 2
endpoints), one complete gzip JSONL candidate log per job, and a worker summary.
The plain `endpoints.jsonl` files are local/remote working artefacts (~55–59 MB
each) and are intentionally not committed; the committed `endpoints.jsonl.gz`
files are byte-identical decompressions and are the audit copies.  The complete
candidate logs are committed because they prove the Top-3 were not re-picked
after seeing labels, edit regions or oracle contributions.

Hashes of both plain and gzip forms are in
`../analysis/raw_results_hashes.json`; `../analysis/run_manifest.json` records
the same execution and the GPU-hour accounting.

The workers ran on `res` (`a100-2`), process backend, pools `res-gpu0`
(Main A / measured) and `res-gpu1` (Main B / oracle), commit
`a16b222466cb`, with 0 errors and 0.048 GPU·h total.
