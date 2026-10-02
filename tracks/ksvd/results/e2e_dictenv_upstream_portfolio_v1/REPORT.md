# ZINC upstream portfolio v1 报告（决策优先）

- 轮次：`zinc-upstream-portfolio-v1`，study `zinc-context-gap`
- 代码 revision：`4c2667ee0ace`（分支 `upstream-portfolio-v1`）；CODE stage-0 代理 revision `574cc091`
- 父模型：`E2E-DictEnv-Scale-v1` Full `LatentScaleSEM108`，408,651 参数，已发布 soup valid `0.1191540920053958`
  - checkpoint sha256 `17f5fcc3cdf32462657006ec89ebf2ecc72ab6fbc954b926af248aca374574eb`
  - split fingerprint `58c69506df857bd8452481c6dcdd608ad230ba1a1aa90624850068fdd8faf28a`
- official test 在本轮任何 run 中都未实例化。

## 结论

**本轮没有值得完整训练的候选。三臂全部关闭，不购买任何新鲜初始化的 320 轮 screen。**

| stage | arm | 初始化 / 协议 | 同协议 matched comparator | raw valid | calibrated valid | 相对增益 | 分组 | 状态 |
|---|---|---|---|---|---|---|---|---|
| cached conditional | CODE | 冻结父模型 + 固定凸 H39+z 头 | 重放父 H39 `0.115023784` | — | `0.115068616` | **−0.000045** | 3/5 | **关闭** `CODE_CONDITIONAL_STOP` |
| warm80 | CONTROL | 父 soup，80 ep，lr 1e-4，末 5 轮 soup | 即 matcher 本身 | `0.114414` | `0.114399` | — | — | 参考 |
| warm80 | DROP | control + 训练期原子 mask p=0.10 | CONTROL cal `0.114399` | `0.117198` | `0.115925` | **−0.001526** | 1/5 | **关闭** |
| warm80 | R3 | control + 36 维 exact shell-3 块 | CONTROL cal `0.114399` | `0.113842` | `0.113868` | **+0.000531** | 4/5 | **关闭** |
| fresh320 | — | — | — | — | — | — | — | **未运行**（无候选过门） |

> 不同协议不混算改善：cached conditional 是冻结父模型上的线性/凸探针，warm80 是新 runner 下从同一父 soup 出发的全模型短训练。每行内部才是 matched 比较（calibrated 对 calibrated，同一 renderer、同一 split）。

## 门判定（预注册，四条全满足才买入）

门槛：候选 calibrated valid `≤0.111`；相对同轮 CONTROL calibrated 下降 `≥0.004`；固定 `id%5`
五分组至少 4 组下降；去掉历史 valid `id172` 后描述性增益 `≥0.002`。

| arm | cal ≤0.111 | gain ≥0.004 | groups ≥4/5 | ex172 ≥0.002 | verdict |
|---|---|---|---|---|---|
| DROP | 0.115925 ✗ | −0.001526 ✗ | 1/5 ✗ | −0.001496 ✗ | **FAIL** |
| R3 | 0.113868 ✗ | +0.000531 ✗ | 4/5 ✓ | +0.000442 ✗ | **FAIL** |

固定分组增益（control − arm，正=臂更好）：
- DROP：`[−0.00093, +0.00049, −0.00216, −0.00267, −0.00237]`（1 组正）
- R3：`[−0.00042, +0.00050, +0.00178, +0.00013, +0.00068]`（4 组正）

CODE 门槛：valid `≤0.112` 且相对 H39 增益 `≥0.003`；实测 `0.115069 / −0.000045`，未过，记录
`CODE_CONDITIONAL_STOP`（只关闭该固定代理，不宣称字典码的理论上限）。

## 实际运行

### CODE stage-0（本地 CPU，run `20261002-150804-a1c659d2`，revision `574cc091`）
`H39_PHYSICAL` 是建立在第二个 reader hidden `[1, H2]`（post-ReLU）上的拟合；重拟合精确复现历史基准
`0.11502378425375083`（refit `0.11502413546485733`）。追加码矩 `z=[Σc, Σc²]`（`c=rho*alpha`，K=288，
宽 576，两个 train-only RMS 尺度）得 `0.11506861603035269`，增益 `−4.483e-05`，分组 3/5，solver
`CONVERGED`（gap `6.165e-07`，λ=1e-5）。

### 共享 warm-80 pilot（res-2 Slurm，节点 c05，A100-PCIE-40GB，driver 525.85.12，torch 2.5.1+cu124）
三臂均 warm-start 同一冻结父 soup，全模型训练 80 epoch（train 10000、batch 128、MAE、Adam lr
1e-4、wd 1e-5、clip 5），固定末 5 轮（76..80）soup，各自折叠 train-median 输出 bias，再对全部 1000
行 official valid 重算。control 全局只买一次。

| run（控制面） | arm | params | soup sha256 | s/epoch | wall s | peak MB | bias Δ |
|---|---|---|---|---|---|---|---|
| `20261002-152611-e30fc31a` | control | 408651 | `ad93338e…` | 10.52 | 855.0 | 745.5 | −0.003617 |
| `20261002-152626-c036a7b8` | drop | 408651 | `6fb948a9…` | 10.70 | 869.0 | 746.3 | −0.016984 |
| `20261002-154105-9f1bd08f` | r3 | 420963 | `8cb931a3…` | 10.28 | 835.4 | 745.1 | −0.003256 |

R3 学到的 train-only 三块 RMS 尺度为 `[0.483, 1.495, 0.165]`（atom / shell2-3 bond /
within-shell3 bond），即 shell-2→3 键块承担最大尺度。两个短 smoke（`up-smoke-control`、
`up-smoke2`）因 `Tensor is not JSON serializable` 报告 bug 失败，`_jsonable` 修复后
`up-smoke3`（`20261002-152330-2e7250e6`）通过并给出稳态估时（≈11.9 s/epoch，724 MB）用于买入
pilot。两并发 Slurm 作业（quota 2）重叠执行；r3 首跑额外构建自身 shell-3 原始缓存。预算未耗尽，
无遗留后台任务。

## 解释

- CODE 的矩读出在冻结 `[1,H39]` 探针之上没有增量信息；`c @ V_L == E` 逐 batch 校验通过，因此这是
  “该代理下矩已被线性张成 / 不预测”的真实负结果，不是接线错误。
- DROP 是方向一致的正则化损失：训练期原子 dropout 反而使 calibrated valid 略升，仅 1/5 组下降。
- R3 是唯一方向为正的臂（+0.000531，4/5 组），也最贵（+36 融合列、420,963 参数），但幅度约为门
  槛的 1/8，ex-172 仅为描述性下限的 ~1/4；与此前 radius-3 VQ 代理（`0.115479`）方向一致。
- 无候选过门，故 fresh-320 正确地**不运行**。

## 复现命令

```bash
uv run pytest -q tracks/ksvd/tests/test_upstream_portfolio_v1.py
uv run research run zinc_upstream_portfolio_v1 --mode stage0-code --set runtime.device=cpu
rr run res-2 up-pilot-control --gpus 1 --cpus 4 \
  --result tracks/ksvd/results/e2e_dictenv_upstream_portfolio_v1 -- \
  uv run --no-sync research run zinc_upstream_portfolio_v1 --mode pilot \
  --purpose "shared control" --set model.stage=pilot --set model.arm=control --set runtime.device=cuda:0
# model.arm=drop / r3 同理；rr status / rr logs / rr pull
```

父权重按 sha256 加载（`17f5fcc3…`），从不重训；`data/**`、`tracks/*/results/**` 缓存为 git-ignored，
以未跟踪文件复制到 res-2；`H39_PHYSICAL.npz` 需 force-add（`.npz` 被忽略）。

## 产物

- 结果：`tracks/ksvd/results/e2e_dictenv_upstream_portfolio_v1/`（`pilot_summary.json`、
  `pilot_{control,drop,r3}.json`、`valid_predictions.npz`、`stage0_code.json`、本报告、`DECISION.md`）
- promoted runs：`records/runs/20261002-150804-a1c659d2.json`、`20261002-152611-e30fc31a.json`、
  `20261002-152626-c036a7b8.json`、`20261002-154105-9f1bd08f.json`
- rr jobs：`up-pilot-control-20261002-152756-17b74fbe`、`up-pilot-drop-20261002-152811-0fbb708a`、
  `up-pilot-r3-20261002-152829-fae8154f`
- claim：`claim-zinc-upstream-portfolio-v1-no-purchased-candidate-20261002`
- decision：`decision-zinc-upstream-portfolio-v1-stop-no-candidate-20261002`
- 预注册：`tracks/ksvd/notes/zinc_upstream_portfolio_v1_preregistration.md`
- 分析：`tracks/ksvd/notes/zinc_upstream_portfolio_v1_analysis.md`