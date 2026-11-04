# configs/

| 文件 | 内容 |
|---|---|
| `tsb-lipo-feature-ladder-v1.yaml` | 冻结的机器可读协议：臂、划分、样本量、WL 属性、Morgan、XGBoost、bootstrap、Test 政策、预算 |
| `feature_spec.json` | 五臂列定义（由 `features` stage 生成，P0 冻结）。G0 的 76 个列名、Morgan 列名、每轮词表哈希、每 (臂, 划分) 列数 |

`tsb_runner._frozen_config()` 会把 YAML 与 `code/tsb_common.py` 逐项比对，
任何不一致都是硬错误。修改语义必须换 `protocol_id`。

已知记录缺陷：`feature_spec.json` 的 WL `blocks[*].dim` 恒为 1（打包占位），
真实维度见 `arms[*].per_split[*].n_columns`。为保持预注册哈希不变，未修改。
