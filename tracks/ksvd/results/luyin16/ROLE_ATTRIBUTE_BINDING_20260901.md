# Role × attribute binding：2026-09-01 快速结论

本轮目的不是优化分数，而是把导师所说的“结构—属性融合”拆成可证伪的对象：

```text
拓扑 → structural role
属性 → compact atom/bond semantic channel
融合 → P(role, attribute)
中心化 → P(role, attribute) − P(role)P(attribute)
null → 每个 radius-2 patch 内独立打乱属性实体
```

所有表示均为完整 radius-2 ego 的 all-center 统计；角色不读取 atom/bond
attributes。没有 K-SVD、attention、Optuna 或 official test。

## 结果

| 协议 | S | S+marginals | S+raw joint | S+centered binding | true−shuffle |
|---|---:|---:|---:|---:|---:|
| official-train scaffold folds（3 折均值） | 0.6939 | 0.7077 | 0.7137 | 0.7393 | +0.0701 |
| official-train → official-valid（冻结） | 0.7775 | 0.7846 | 0.7953 | 0.8017 | +0.0014 |

内部三折中，centered binding 相对 marginals 为 `+0.0315`、相对 S 为
`+0.0454`，三折均胜；换 XGBoost seed `[0,1,2]` 后
`S+binding=0.7380`、`S=0.6981`。节点 binding 平均 `0.7405`，边 binding
平均 `0.7158`，但 node+edge 在 official-valid 的绝对分数最高。

冻结 official-valid 后，绝对增量仍在：`S+binding−S=+0.0243`、
`S+binding−S+marginals=+0.0172`。不过 centered binding 相对两次 shuffle
只有 `+0.0014`，低于预注册 `+0.003` 门槛；其中一个 shuffle 略高于 true。
相反，未中心化 raw joint 的 true−shuffle 为 `+0.0170`，通过该 split 的
控制门槛。

## 当前判定

1. **结构—属性联合表示有任务增量**：不是简单把结构和属性边际拼起来；
   raw joint 和 centered binding 都在冻结 split 上高于 marginals/S。
2. **binding 机制尚未完全锁定**：内部 scaffold folds 的 role–attribute
   对应很强，但在 official-valid 上 centered residual 的 shuffle 证据不足。
   不能把 `0.8017` 全部命名为“因果 binding 增益”。
3. **K-SVD 暂不应加入**：当前最强信息增量来自融合对象定义；先不扫描 K/T、
   字典或 cross-attention。

## 下一步（单一机制实验）

固定当前协议，只比较节点角色的定义：`shell-only`、`shell×induced-degree`
和当前 `shell×induced-degree×cycle`，同时保留 raw/binding/shuffle 三组。若
official-valid 仍只有 raw joint 而 centered binding 不稳定，则采用 raw joint
作为可解释融合基线；若 centered binding 也稳定，再考虑低维压缩（PCA/SVD）
而不是直接上 K-SVD。

原始摘要：

- [`role_attribute_binding_screen/summary.json`](role_attribute_binding_screen/summary.json)
- [`role_attribute_binding_screen_seedconfirm/summary.json`](role_attribute_binding_screen_seedconfirm/summary.json)
- [`role_attribute_binding_screen_official_valid/summary.json`](role_attribute_binding_screen_official_valid/summary.json)
