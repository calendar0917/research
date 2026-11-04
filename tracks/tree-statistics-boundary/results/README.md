# results/

本 track 的第一轮数字**不在**这个目录，而是在：

- `../analysis/tables/` — 表 A–D（CSV + JSON）
- `../analysis/{fits,predictions,test,freeze,p0}/` — 每次拟合、预测、冻结与门槛记录
- `../records/runs/` — 控制面 promote 后的不可变记录
- `../REPORT.md` — 人读汇总

保留本目录是为了遵守仓库 `tracks/*/results/` 的目录约定，不在此另存数字副本，
避免同一结果出现两份互相漂移的副本。

控制面本地执行现场在 `../runs/`（git-ignored）。
