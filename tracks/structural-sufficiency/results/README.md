# results/

本轨实验结果唯一入口。

## Phase 0

- 人读报告：`phase0/REPORT.md`
- 机器可读数字：
  - `phase0/analysis/phase0_results.json`（CPU toy / 碰撞 / oracle）
  - `phase0/analysis/lipo_feasibility_audit.json`（协议 / 成本 / 标签隔离）
  - `phase0/analysis/train_only_er_information_audit.json`（train-only LOO / 置换）
- 代码：`phase0/code/`
- 测试：`phase0/tests/`

> 依仓库约定，`results/**/*.json` 是本地证据（git-ignored），fresh clone
> 不包含；用上面三个脚本可逐字重生成。`REPORT.md` 保留关键数字与判据。

本轮没有性能主表；不要把这些数字与其它轨的结果混表。
未来 pilot 的 `protocol_id` 与 registry 在 freeze 后再建立。
