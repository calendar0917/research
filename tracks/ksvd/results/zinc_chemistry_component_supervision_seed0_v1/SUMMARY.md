## Summary (Chinese, ~400 chars)

化学分量监督 seed0 实验完成。将读取器末尾 39→1 线性替换为 39→2 拆分（SUM/COMP 初始预测与原 M 一致 ≤1e-5，两臂 state 完全一致）。SUM 优化 L_g，COMP 优化 L_g+0.5(L_ell+L_s)，ell=(logP−MU)/sigma_logP，s=g−ell。g 恒等式 float64 校验通过 (4.4e-16)；s 与 SA 偏差 epsilon 在 G0-fit MAE 0.001512 溢出阈值，不称“纯 SA”。

COMP 对 SUM 校准 g-MAE 取得显著增益：G0 +0.009974 CI [+0.0065,+0.0135]，overall +0.009873 CI [+0.0064,+0.0136]，3/3 门槛通过 → COMPONENT_SUPERVISION_CANDIDATE。两分量均 FIT_ADEQUATE，但残余化学分量 s 为 dev-G0 差距主体 (MAE 0.083 vs ell 0.054)。误差抵消显著 (61.9% 异号，triangle_gap 0.050)。

资源：两臂并行 c05 A100，UUID 分离 (0D:00 vs 1D:00)，各 1 GPU，642/670s，GPU 计 0.364h＜0.8h。不读 official-valid/test。

决策：通过门槛冻结一次二次确认设计（seed1，同 fold/init/recipe）；按 s dev-gap 引出一条定向新设计。关闭新增 seed/折/变体，本轮不自动续航。
