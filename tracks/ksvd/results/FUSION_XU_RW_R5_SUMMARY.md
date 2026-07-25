# Multi-walk RW node structure + GIN fusion (Xu global e*)

## Setup

| 项 | 值 |
|----|-----|
| 协议 | Xu：全局 e\* = argmax mean_fold Acc |
| 数据 | MUTAG |
| patch | 每节点 **r=5** 条 RW，p=0.5, q=2, L=6, m=8 |
| 系数 | 每条 walk → OMP 得 \(x_i\)；\(s_v=\mathrm{mean}_i x_i\) 再取 \|·\| |
| 字典 | train fold 共享 D；patch 可子采样至 8000 |
| 融合 | concat / gate（跳过 residual；gin_only 用历史 89.4%） |

实现：`node_struct.py`（num_walks + pool）· `run_fusion_xu.py --num_walks 5`

## 重建误差（诊断，不可与 B0 比优劣）

- train recon ≈ **0.09–0.12**（RW 族内部）
- encode patch_recon ≈ **0.06–0.09**
- 仅说明「这类 patch 能否被 D 拟合」，**不**用来证明比 1-hop 更好

## 结果

| 方法 | Acc % @ e\* |
|------|-------------|
| gin_only（ref） | **89.4 ± 5.8** |
| RW r=5 + **concat** | **88.3 ± 6.7** |
| RW r=5 + **gate** | **88.8 ± 4.4** |

对照历史（同协议）：

| 设定 | concat | gate |
|------|--------|------|
| B0 单邻域 | 90.4 | 89.4 |
| RW r=1 | 84.6 | **90.9** |
| RW r=5 mean | 88.3 | 88.8 |

## 解读

1. **多 walk 把单 walk concat 从 84.6 拉回 88.3**（更稳），但 **仍未稳定超过 gin_only**。  
2. **r=5 gate 未超过 r=1 gate（90.9）**——多采样不保证增益；可能平均掉了判别 walk，或 MUTAG 增益空间已饱和。  
3. recon 只作 fold 日志诊断，**主结论仍看 Acc**。  
4. 节点系数定义已对齐理论：\(s_v=\mathrm{pool}_i(\mathrm{OMP}(D,y_i))\)。

## 复现

```bash
python -m code.run_fusion_xu --patch RW --num_walks 5 --pool mean \
  --p 0.5 --q 2 --walk_length 6 --max_nodes 8 --variants concat,gate
```
