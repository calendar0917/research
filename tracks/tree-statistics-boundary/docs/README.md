# Tree Models and Explicit Graph Statistics — 轨内说明

- Track 入口：[TRACK.md](../TRACK.md)
- 协议：[PROTOCOL.md](../PROTOCOL.md)
- 第一轮报告：[REPORT.md](../REPORT.md)

## 第一轮（`tsb-lipo-feature-ladder-v1`）一句话

在 76 列受控全局组成/拓扑统计 `G0` 之上，加入 rooted typed-WL round 0–2
（二值或计数）在 Lipophilicity 三个已发布随机划分上都降低 Validation RMSE
（均值 −0.15，三个划分同向、bootstrap 区间不含 0）；计数 vs 二值几乎没有差别
（均值 +0.0007，方向不一致）；round 3 的增量小且区间跨 0；Morgan radius-2
计数指纹比 typed-WL 计数更差（均值 +0.022，三个划分同向）。
以上只是对差值与方向的转述，不构成机制结论。

## 跑法

```bash
export RESEARCH_TRACK_ROOT=tracks/tree-statistics-boundary
export RESEARCH_EXTRA_RUNNER_PATHS=tracks/tree-statistics-boundary/code
export RESEARCH_EXTRA_RUNNERS=tsb_runner
uv run research run tsb_lipo_feature_ladder_v1 --study tsb-lipo-feature-ladder-v1 \
  --purpose "..." --set stage=features|p0|p1|p2|freeze|test|report
```

`test` 必须 `--mode terminal`，且只能在 `freeze` 之后运行一次。
