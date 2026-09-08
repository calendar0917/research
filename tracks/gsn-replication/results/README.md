# results — 本轨唯一数字源

`code/run_official.py` 每个 (config, seed) 写一份 `results/<config>__seed<seed>.json`；
git-ignored。汇总用 `code/audit_results.py`（并排论文数字）。

JSON schema（摘要）:
```json
{
  "protocol_id": "gsn-official-social-v1",
  "config": "IMDBBINARY--gsn-e",
  "dataset": "IMDBBINARY", "variant": "gsn-e",
  "seed": 0, "folds": [0..9],
  "in_paper": true, "expected_paper": {"mean": 0.778, "std": 0.033},
  "metric_semantics": "10-fold mean of test acc at epoch with best mean across folds",
  "official_repo_head": "6cce24a…",
  "config_sha256": "…",
  "metrics": {"best_test_mean": …, "best_test_std": …, "num_params": …},
  "cmd": [ …官方 CLI，可复现… ]
}
```

重要数字（论文语义）不要重复写在 markdown 正文，以结果 JSON + audit 输出为准。
