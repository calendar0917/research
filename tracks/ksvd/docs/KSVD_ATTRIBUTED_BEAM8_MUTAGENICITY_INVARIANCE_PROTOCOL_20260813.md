# Attributed Beam8/Mutagenicity relabel-invariance protocol

> 日期：2026-08-13  
> 状态：旧 binary-order/typed-payload invariance 失败后注册。

对前 512 张图各做 3 次固定节点重编号，使用新 attributed Beam8：

- atom type 进入全局 stable-WL initial color；
- bond type 进入 WL neighbor message；
- patch slots 用 root-preserving、node-colored、edge-typed exact canonicalization；
- token 显式编码每个 slot 的 atom type及每对 slot 的 bond type。

硬 gate：ordered token rows、token multiset、compact TRUE readout 三项 match rate 必须均为
1.0。映射回原 ID 的 exact chain match 只作诊断：attributed automorphism 内选择不同节点实例
可以合法，但必须产生相同 token/relation 表示。

通过后，旧 typed classification 作废并由 attributed 版本重跑；失败则停止 Mutagenicity typed
Beam8 分类路线。

