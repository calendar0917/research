# 交互式流水线可视化

## 打开

```bash
# 重新导出数据 + 生成 HTML
cd tracks/ksvd
python -m code.viz_pipeline_data
python -m code.build_viz_html

# 浏览器打开（推荐）
xdg-open results/viz_pipeline/index.html
# 或 file:///.../tracks/ksvd/results/viz_pipeline/index.html
```

`index.html` **内嵌**了 `data.json`，可直接用 `file://` 打开，无需本地服务器。

## 六个标签页

| 页 | 内容 |
|----|------|
| 1 Seeds | 度分层种子、预算、覆盖率、不硬删 |
| 2 Walk | 逐步播放游走；候选转移；边权软降权 |
| 3 Patches | 诱导 \(G[S]\)；是否含 C4 |
| 4 Dictionary | \(Y,D,X\) 形状；原子热力图；usage |
| 5 Encode | OMP 系数条；图向量 \(s_G\) readout |
| 6 B0 vs RW | 并排对比 1-hop 星形 vs RW patch |

可切换 **C4 demo / C8 demo** 图。

## 文件

| 文件 | 说明 |
|------|------|
| `index.html` | 交互页面 |
| `data.json` | 轨迹与 KSVD 导出 |
| `code/viz_pipeline_data.py` | 导出 |
| `code/build_viz_html.py` | 打包 HTML |
