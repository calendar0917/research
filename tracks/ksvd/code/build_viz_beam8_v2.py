"""Build interactive single-page HTML for the v2 Beam8 continuous-cover scheme.

Narrative-first, closed-loop version: the takeaway tab walks the whole pipeline
(sample -> compress -> stitch -> complete -> output) with "how it computes" and
"what it stores" per step, plus provenance for every headline number. Then a
scan demo, a new stitch demo, a compare tab, and a collapsible appendix.

Reads results/viz_beam8_v2/beam8_trace.json (real sampler trace) and embeds the
audit numbers from PATCH_CHAIN_ARBITRARINESS_ERROR_BUDGET_AUDIT_20260802.md and
KSVD_CONTINUOUS_COVER_EXPLORATION_20260801.md.

Usage:
  python -m code.build_viz_beam8_v2
  → results/viz_beam8_v2/index.html
"""

from __future__ import annotations

import json
from pathlib import Path

_TRACK = Path(__file__).resolve().parents[1]
OUT = _TRACK / "results" / "viz_beam8_v2"


def main() -> int:
    trace_path = OUT / "beam8_trace.json"
    if not trace_path.exists():
        print("Missing beam8_trace.json; run python -m code.viz_beam8_trace first")
        return 1
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    html = TEMPLATE.replace("/*__TRACE__*/", json.dumps(trace, ensure_ascii=False))
    out = OUT / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")
    print(f"open in browser: file://{out.resolve()}")
    return 0


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>连续扫描一张图：怎么切、怎么算、怎么拼、怎么存</title>
<style>
  :root {
    --bg: #0d1117;
    --panel: #161b22;
    --panel2: #1c212b;
    --border: #2d333b;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --red: #f85149;
    --amber: #d29922;
    --purple: #bc8cff;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    font-family: "Segoe UI", system-ui, "PingFang SC", "Microsoft YaHei", sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.62;
  }
  header {
    padding: 16px 26px; border-bottom: 1px solid var(--border);
    background: var(--panel); position: sticky; top: 0; z-index: 20;
    display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
  }
  header h1 { font-size: 1.02rem; margin: 0; font-weight: 700; }
  header .sub { color: var(--muted); font-size: 0.78rem; }
  .tabs { display: flex; gap: 6px; flex-wrap: wrap; margin-left: auto; }
  .tab {
    background: transparent; border: 1px solid var(--border); color: var(--muted);
    padding: 6px 14px; border-radius: 999px; cursor: pointer; font-size: 0.82rem;
  }
  .tab.active { background: var(--accent); border-color: var(--accent); color: #0d1117; font-weight: 600; }
  .tab:hover:not(.active) { border-color: var(--accent); color: var(--text); }
  main { padding: 22px 26px 60px; max-width: 1060px; margin: 0 auto; }
  section.panel { display: none; }
  section.panel.active { display: block; }

  .hero h2 { font-size: 1.6rem; margin: 2px 0 12px; font-weight: 800; line-height: 1.35; }
  .hero .lede { font-size: 1.0rem; color: #c9d1d9; max-width: 780px; }
  .hero .lede b { color: #fff; }

  h3.sec { font-size: 1.12rem; margin: 30px 0 12px; font-weight: 700; border-left: 3px solid var(--accent); padding-left: 12px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 18px 20px; font-size: 0.9rem; }
  .card p { margin: 6px 0; }
  .mini { font-size: 0.78rem; color: var(--muted); }
  .tag { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.7rem; font-weight: 700; vertical-align: middle; }
  .tag.ok { background: rgba(63,185,80,0.15); color: var(--green); }
  .tag.no { background: rgba(248,81,73,0.15); color: var(--red); }
  .tag.opt { background: rgba(188,140,255,0.15); color: var(--purple); }
  .quote {
    background: var(--panel2); border: 1px solid var(--border); border-left: 3px solid var(--green);
    padding: 14px 18px; border-radius: 12px; font-size: 0.98rem; margin: 22px 0;
  }
  .quote b { color: #fff; }

  /* pipeline */
  .pipe { margin: 10px 0; }
  .parrow { text-align: center; color: var(--muted); font-size: 1.05rem; line-height: 1; margin: 4px 0; }
  .pnode { border: 1px solid var(--border); border-radius: 12px; padding: 10px 16px; background: var(--panel2); font-weight: 700; font-size: 0.9rem; text-align: center; }
  .pnode.in { border-color: rgba(63,185,80,0.5); }
  .pnode.out { border-color: rgba(188,140,255,0.5); background: rgba(188,140,255,0.08); }
  .pnode .mini { font-weight: 400; }
  .pcard { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 16px 18px; }
  .pcard .ptitle { font-weight: 800; font-size: 0.98rem; margin-bottom: 8px; }
  .pcard ul { margin: 6px 0 0; padding-left: 20px; }
  .pcard li { margin: 5px 0; font-size: 0.88rem; }
  .pcard .how { color: var(--accent); font-weight: 600; }
  .pcard .store { color: var(--purple); font-weight: 600; }
  .pcard .why { color: var(--green); font-weight: 600; }

  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px; margin: 16px 0; }
  .stat { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 16px 18px; }
  .stat .v { font-size: 1.45rem; font-weight: 800; color: var(--green); letter-spacing: -0.01em; }
  .stat .v small { font-size: 0.95rem; color: var(--muted); font-weight: 600; }
  .stat .k { font-size: 0.84rem; color: var(--muted); margin-top: 2px; }
  .stat .why { font-size: 0.76rem; color: var(--muted); margin-top: 8px; border-top: 1px dashed var(--border); padding-top: 8px; }
  .stat .why b { color: #c9d1d9; font-weight: 600; }

  .bars { margin: 6px 0; }
  .bar-row { display: flex; align-items: center; gap: 10px; margin: 5px 0; font-size: 0.82rem; }
  .bar-row .lab { width: 128px; color: var(--muted); text-align: right; flex-shrink: 0; }
  .bar-row .lab b { color: var(--text); }
  .bar-track { flex: 1; height: 14px; background: #0b1117; border-radius: 7px; overflow: hidden; }
  .bar-track i { display: block; height: 100%; border-radius: 7px; }
  .bar-row .val { width: 58px; font-weight: 700; }

  .wf-row { display: grid; grid-template-columns: 190px 1fr 62px; gap: 10px; align-items: center; margin: 4px 0; font-size: 0.82rem; }
  .wf-row .lab { color: var(--muted); text-align: right; }
  .wf-track { position: relative; height: 16px; background: #0b1117; border-radius: 4px; }
  .wf-bar { position: absolute; top: 2px; height: 12px; border-radius: 3px; }
  .wf-row .val { text-align: right; font-weight: 700; }

  details { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px 18px; margin: 12px 0; }
  details summary { cursor: pointer; font-weight: 700; font-size: 0.9rem; }
  details summary:hover { color: var(--accent); }
  details[open] summary { margin-bottom: 12px; }
  table { border-collapse: collapse; width: 100%; font-size: 0.8rem; }
  th, td { padding: 6px 10px; border-bottom: 1px solid var(--border); text-align: right; }
  th { color: var(--muted); font-weight: 600; }
  th:first-child, td:first-child { text-align: left; }
  tr:hover td { background: rgba(88,166,255,0.05); }
  .hl { color: var(--green); font-weight: 700; }
  .kv { display: grid; grid-template-columns: 150px 1fr; gap: 3px 12px; }
  .kv span:first-child { color: var(--muted); }
  h4.sub { font-size: 0.88rem; color: var(--muted); margin: 16px 0 6px; font-weight: 600; }

  .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; padding: 10px 14px; background: var(--panel); border: 1px solid var(--border); border-radius: 12px; margin-bottom: 12px; }
  .controls label { font-size: 0.8rem; color: var(--muted); display: flex; align-items: center; gap: 6px; }
  select, input[type=range] { accent-color: var(--accent); }
  select { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; }
  button.btn { background: var(--accent); color: #0d1117; border: none; border-radius: 8px; padding: 6px 14px; cursor: pointer; font-size: 0.82rem; font-weight: 600; }
  button.btn.secondary { background: var(--border); color: var(--text); }
  button.btn:disabled { opacity: 0.4; cursor: not-allowed; }
  #svg-graph { width: 100%; height: 480px; background: #0b1117; border: 1px solid var(--border); border-radius: 14px; display: block; }
  .node-label { font-size: 10px; fill: #fff; font-weight: 600; pointer-events: none; text-anchor: middle; }
  .legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 0.76rem; color: var(--muted); margin: 10px 0; }
  .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }
  .legend .sw { border-radius: 2px; }
  .ptiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin: 12px 0; }
  .ptile { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; }
  .ptile .v { font-size: 1.2rem; font-weight: 800; }
  .ptile .k { font-size: 0.75rem; color: var(--muted); }
  .grid { display: grid; gap: 14px; }
  .grid.three { grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
  .grid.two { grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); }
  .scan-grid { display: grid; grid-template-columns: 1.4fr 1fr; gap: 14px; }
  @media (max-width: 960px) { .scan-grid { grid-template-columns: 1fr; } }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; }
  .chip { border: 1px solid var(--border); border-radius: 999px; padding: 5px 12px; font-size: 0.8rem; background: var(--panel2); }
  .chip .k { color: var(--muted); margin-right: 5px; }
  footer { padding: 14px 26px; color: var(--muted); font-size: 0.72rem; border-top: 1px solid var(--border); }
</style>
</head>
<body>
<header>
  <div>
    <h1>连续扫描一张图：怎么切、怎么算、怎么拼、怎么存</h1>
    <div class="sub">图切分 · 连续重叠扫描 · 重建质量 · 2026-08-02</div>
  </div>
  <div class="tabs" id="tabs"></div>
</header>
<main id="main"></main>
<footer id="footer"></footer>
<script>
const TRACE = /*__TRACE__*/;

// ---------------- 数据 ----------------
const pipeline = [
  { io: "in", name: "输入", body: ["72 张合成图，每张 50 个节点（正则 / 小世界 / 分块社区）", "无标签，只研究图本身：切格 → 表示 → 拼回"] },
  { io: "step", n: 1, name: "采样 Beam8：把图切成窗格，一格接一格", tag: "核心", body: [
    ["how", "怎么算", "起点 = 一条还没看过的边，围着它开第一格。"],
    ["how", "怎么算", "之后每格：把上一格的 10 个节点里「带走哪 3 个」的所有搭法（C(10,3)=120 种）筛出连通的 8 个候选，各自扩展成一整格。"],
    ["how", "怎么算", "打分：新增真实边 → 新点对 → 新节点 → 格内边，选最优的一格。"],
    ["store", "存什么", "一串窗格（每格 10 个节点 + 中心）+ 重叠节点的 slot 对应表（前一格第几号 ↔ 后一格第几号）。"],
    ["why", "数字来由", "逐窗口与正式实现一致（见「怎么扫」演示的完整候选表）。"],
  ]},
  { io: "step", n: 2, name: "压缩 KSVD（可选）：减小存储", tag: "可选", body: [
    ["how", "怎么算", "每格 10 节点 → 45 个数（10 节点两两之间有没有边）；在训练格上学 24 个共享模板，每格只用稀疏的 3 个系数表示。"],
    ["store", "存什么", "24×45 的共享模板字典 + 每格 3 个系数（原本要存 45 个值）。"],
    ["why", "数字来由", "K24/T3 是容量-误差的合理平衡点；加码 K/T 收益递减（见附录）。"],
  ]},
  { io: "step", n: 3, name: "拼接 stitching：把窗格拼回整图", body: [
    ["how", "怎么算", "按重叠对应表，把每格放回原图坐标；一条边被多格看到就取平均（uniform mean）。"],
    ["why", "数字来由", "逐格演示见「怎么拼」。"],
  ]},
  { io: "step", n: 4, name: "补全 completion（可选）：处理没直接看到的边", tag: "可选", body: [
    ["how", "怎么算", "用训练集的图，把没直接看到的边补齐（只用训练图，不用真值）。"],
  ]},
  { io: "out", name: "输出：整图重建", body: [
    "怎么量：全图 RMSE（所有点对的重建误差，越低越好）、真实边覆盖（看到多少真实边）、点对覆盖。",
  ]},
];

const stats = [
  { v: "0.385 → 0.324", k: "完整管线重建误差（全图 RMSE）", why: "<b>来由</b>：同一批 72 张图、同样加 KSVD 压缩和补全，只换扫描方式：旧扫描链 0.385 → Beam8 0.324，约 −16%。" },
  { v: "76.5% → 87.3%", k: "真实边覆盖率（未压缩）", why: "<b>来由</b>：只把扫描方式从「盯目标边」换成「候选择优」，全部图上的边覆盖提升 11 个百分点。" },
  { v: "≈0.28 s/图", k: "扫描耗时", why: "<b>来由</b>：8 个候选已拿到绝大部分收益；候选加到 32 只多约 0.1pt 覆盖，却贵约 6.6 倍。" },
];

const strategies = [
  { name: "无脑往前走", cover: 0.5176, note: "最朴素，沿图一直走" },
  { name: "沿边界向外扩展", cover: 0.5977, note: "从当前位置往外扩一圈" },
  { name: "盯着没看过的边", cover: 0.7650, note: "下一步总想覆盖一条新边" },
  { name: "候选择优 Beam8", cover: 0.8734, note: "摆出 8 个方向各试一遍，选新增最多", ok: true },
];
const beams = [
  { cell: "只盯目标边", sec: 0.025, edge: 0.7650, rmse: 0.3054 },
  { cell: "候选择优 B8", sec: 0.279, edge: 0.8734, rmse: 0.2205, pick: true },
  { cell: "候选择优 B16", sec: 0.505, edge: 0.8803, rmse: 0.2139 },
  { cell: "候选择优 B32", sec: 0.954, edge: 0.8844, rmse: 0.2102 },
  { cell: "候选择优 B32+R2", sec: 1.849, edge: 0.8845, rmse: 0.2101 },
];
const waterfall = [
  { name: "旧扫描 · 不压缩", rmse: 0.3544 },
  { name: "旧扫描 · +压缩", rmse: 0.4358 },
  { name: "旧扫描 · +补全", rmse: 0.3845 },
  { name: "Beam8 · 不压缩", rmse: 0.2200 },
  { name: "Beam8 · +压缩", rmse: 0.3349 },
  { name: "Beam8 · +压缩+补全", rmse: 0.3241, pick: true },
];

const appendix = {
  spec: [
    ["窗口大小 patch size", "10 个节点"],
    ["相邻窗口重叠", "3 个节点"],
    ["预算系数 budget multiplier", "1.5"],
    ["起点", "一条还没看过的边（target-edge seed）"],
    ["下一窗口的选择", "8 个候选、每候选一次扩展（Beam8_R1）"],
    ["候选打分", "新边 → 新点对 → 新节点 → 格内边"],
    ["节点排序", "按构造顺序 construction ordering"],
    ["重叠对应关系", "显式记录 slot-to-slot 映射"],
    ["可选：稀疏压缩", "KSVD K=24, T=3, updates=25"],
    ["拼接", "均匀取平均 uniform mean"],
    ["可选：训练图补全", "train-only structural ridge"],
  ],
  rejected: [
    ["重叠一半", "50% overlap", "用相同成本探索的新区域更少，被 s10/o3 支配"],
    ["无限制候选搜索", "Beam32/R2", "收益极小，耗时却是 B8 的 6.6 倍"],
    ["窗口内节点保持同位置", "slot-persistent ordering", "像图像一样对齐位置，反而破坏重建几何"],
    ["用拉普拉斯谱做全局排序", "Fiedler total ordering", "谱坐标有并列、跨图不稳定，不能当绝对坐标"],
    ["把上一窗口信息喂进解码器", "linear transition decoder", "有弱信号但低于收益门槛，且不如只看当前窗口"],
    ["给重叠位置加权", "slot reliability weighting", "几乎无效果（约 0.03%）"],
  ],
  rounds: [
    { r: 1, name: "能不能连续扫？", verdict: "早期：失败", ok: false, note: "无脑往前走/沿边界扩展都留缺口，block 图上尤其差" },
    { r: 2, name: "盯目标边能救吗？", verdict: "通过：单链可行", ok: true, note: "跨区域桥接有效，但只是起点" },
    { r: 3, name: "压缩算法能再省吗？", verdict: "中期：差一点", ok: false, note: "KSVD 有益但收益 5.4%，低于 10% 强门槛" },
    { r: 4, name: "像图像一样对齐窗口？", verdict: "否决", ok: false, note: "对齐位置让重建明显变差" },
    { r: 5, name: "把上一窗口信息喂解码器？", verdict: "否决", ok: false, note: "弱信号，无实际收益" },
    { r: 6, name: "用拉普拉斯谱排序？", verdict: "否决", ok: false, note: "谱坐标不能当唯一绝对坐标" },
    { r: "v2", name: "到底怎么扫最好？", verdict: "采纳 Beam8", ok: true, note: "候选择优修复主要瓶颈，全链路误差约 −16%" },
  ],
  boundaries: [
    "Beam8 比旧法贵约 11 倍，但单图仍只有约 0.28 秒。",
    "略偏向观察真实边、少看非边点对（pair coverage 降约 3 个百分点）。",
    "方法假设能看到完整图（用于压缩/表示），不是缺边时的边预测。",
    "目前结果在合成图上；拿到导师真实 50-node 数据后要整套重跑。",
    "若真实图之间有共享节点身份，必须加「整图对齐邻接」baseline。",
    "本轮不涉及下游分类任务。",
  ],
  nextReal: [
    "数据是否连通、是否共享节点身份",
    "用冻结的 Beam8 跑 node/edge/pair 覆盖，与旧法对比",
    "记录候选搜索耗时、窗口数、重叠、连通性",
    "重建上限（RAW stitching ceiling）",
    "压缩前后 held-out 重构对比（INIT/FINAL/PCA3）",
  ],
};

const TABS = [
  ["read", "一图看懂"],
  ["scan", "怎么扫"],
  ["stitch", "怎么拼"],
  ["compare", "关键数字"],
  ["detail", "附录"],
];

const $ = (id) => document.getElementById(id);
const fmt = (x, d = 4) => Number(x).toFixed(d);
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";

// ---------------- 通用小组件 ----------------
function barRow(label, value, opts = {}) {
  const max = opts.max || 1;
  const p = Math.max(0, Math.min(100, value / max * 100)).toFixed(1);
  const color = opts.color || "linear-gradient(90deg,#58a6ff,#bc8cff)";
  const lab = opts.bold ? `<b>${label}</b>` : label;
  return `<div class="bar-row"><span class="lab">${lab}</span>
    <div class="bar-track"><i style="width:${p}%;background:${color}"></i></div>
    <span class="val${opts.cls || ""}">${pct(value)}</span></div>`;
}

function table(headers, rows, opts = {}) {
  const th = headers.map(h => `<th>${h}</th>`).join("");
  const trs = rows.map((row, i) => `<tr${opts.hl === i ? ' class="hl"' : ""}>${row.map(c => `<td>${c}</td>`).join("")}</tr>`).join("");
  return `<table><thead><tr>${th}</tr></thead><tbody>${trs}</tbody></table>`;
}

function waterfallView(rows) {
  const max = Math.max(...rows.map(r => r.rmse)) * 1.12;
  return rows.map(r => {
    const x = (r.rmse / max) * 100;
    const color = r.pick ? "#3fb950" : "#58a6ff";
    return `<div class="wf-row">
      <span class="lab">${r.name}</span>
      <div class="wf-track"><div class="wf-bar" style="width:${x.toFixed(1)}%;background:${color}"></div></div>
      <span class="val"${r.pick ? ' style="color:var(--green)"' : ""}>${fmt(r.rmse)}</span></div>`;
  }).join("");
}

// ---------------- Tab 1：一图看懂（闭环） ----------------
function renderRead() {
  const pipe = pipeline.map(p => {
    if (p.io === "in" || p.io === "out") {
      return `<div class="pnode ${p.io}">${p.name}<div class="mini">${p.body[0]}</div></div>`;
    }
    const tag = p.tag ? ` <span class="tag ${p.tag === "核心" ? "ok" : "opt"}">${p.tag}</span>` : "";
    const bullets = p.body.map(([cls, label, txt]) => `<li><span class="${cls}">${label}：</span>${txt}</li>`).join("");
    return `<div class="pcard"><div class="ptitle">${p.n}. ${p.name}${tag}</div><ul>${bullets}</ul></div>`;
  }).join(`<div class="parrow">↓</div>`);

  const statCards = stats.map(s => `
    <div class="stat">
      <div class="v">${s.v}</div>
      <div class="k">${s.k}</div>
      <div class="why">${s.why}</div>
    </div>`).join("");

  return `
  <div class="hero">
    <h2>一张图太大，我们把它切成小窗格、一格一格连续扫过，再拼回完整图。怎么切、怎么算、怎么拼、怎么存？</h2>
    <div class="lede">
      我们沿整条流水线逐个验证。最重要的发现：<b>最值得优化的环节是「下一步扫哪一格」</b>——它比后面用多复杂的压缩算法都更能决定重建质量。
    </div>
  </div>

  <h3 class="sec">整条流水线（闭环）</h3>
  <div class="pipe">${pipe}</div>

  <h3 class="sec">三个关键数字，每个都说明来路</h3>
  <div class="stats">${statCards}</div>

  <div class="quote">
    <b>一句话总结：</b>先让扫描去选信息密度更高的连续窗口，比在旧扫描上继续堆更复杂的算法更有效。
  </div>
  `;
}

// ---------------- Tab 2：怎么扫（算法演示） ----------------
let state = { step: 0, autoplay: false, timer: null };
let stitchState = { step: 0, autoplay: false, timer: null };

function edgeKey(a, b) { return a < b ? a + "-" + b : b + "-" + a; }

function coveredEdgeSet(stepIndex) {
  const s = new Set();
  for (let i = 0; i <= stepIndex; i++) {
    const ns = new Set(TRACE.steps[i].nodes);
    TRACE.edges.forEach(([u, v]) => { if (ns.has(u) && ns.has(v)) s.add(edgeKey(u, v)); });
  }
  return s;
}

function stepEdgesSet(stepIndex) {
  const s = new Set();
  const ns = new Set(TRACE.steps[stepIndex].nodes);
  TRACE.edges.forEach(([u, v]) => { if (ns.has(u) && ns.has(v)) s.add(edgeKey(u, v)); });
  return s;
}

function drawGraph(svg, opts) {
  const step = opts.step;
  const current = new Set(step.nodes);
  const prev = new Set(opts.prevNodes || []);
  const overlap = new Set(opts.overlap || []);
  const covered = opts.coveredEdges;
  const stepEdges = opts.stepEdges;
  const nodeStyle = opts.nodeStyle || "scan";
  let eHtml = "", nHtml = "";
  TRACE.edges.forEach(([u, v]) => {
    const k = edgeKey(u, v);
    const p1 = TRACE.layout[String(u)], p2 = TRACE.layout[String(v)];
    let stroke = "#243044", sw = 1.5, op = 0.4;
    if (covered.has(k)) { stroke = "#58a6ff"; sw = 2; op = 0.8; }
    if (stepEdges.has(k)) { stroke = "#3fb950"; sw = 3; op = 1; }
    eHtml += `<line x1="${p1[0]}" y1="${p1[1]}" x2="${p2[0]}" y2="${p2[1]}" stroke="${stroke}" stroke-width="${sw}" opacity="${op}" stroke-linecap="round"/>`;
  });
  TRACE.layout && Object.keys(TRACE.layout).forEach(u => {
    u = Number(u);
    const p = TRACE.layout[String(u)];
    let fill = "#2d333b", r = 10;
    if (nodeStyle === "scan") {
      if (overlap.has(u)) { fill = "#d29922"; r = 13; }
      else if (current.has(u)) { fill = "#bc8cff"; r = 13; }
      else if (prev.has(u)) { fill = "#3fb950"; r = 11; }
    }
    if (u === step.center && nodeStyle === "scan") { stroke = "#f0f6fc"; strokeWidth = 2.5; }
    nHtml += `<circle cx="${p[0]}" cy="${p[1]}" r="${r}" fill="${fill}" stroke="#0d1117" stroke-width="1.5"/><text x="${p[0]}" y="${p[1]}" class="node-label">${u}</text>`;
  });
  svg.innerHTML = eHtml + nHtml;
}

function scanControls(idx) {
  return `
    <div class="controls">
      <button class="btn secondary" id="btnPrev" ${idx === 0 ? "disabled" : ""}>◀ 上一格</button>
      <button class="btn" id="btnNext" ${idx === TRACE.steps.length - 1 ? "disabled" : ""}>下一格 ▶</button>
      <button class="btn secondary" id="btnPlay">${state.autoplay ? "暂停" : "自动播放"}</button>
      <label>跳到第 <select id="selStep">${TRACE.steps.map((s, i) => `<option value="${i}" ${i === idx ? "selected" : ""}>${i + 1} 格</option>`).join("")}</select>格</label>
    </div>`;
}

function stitchControls(idx) {
  return `
    <div class="controls">
      <button class="btn secondary" id="sPrev" ${idx === 0 ? "disabled" : ""}>◀ 上一格</button>
      <button class="btn" id="sNext" ${idx === TRACE.steps.length - 1 ? "disabled" : ""}>下一格 ▶</button>
      <button class="btn secondary" id="sPlay">${stitchState.autoplay ? "暂停" : "自动播放"}</button>
      <label>拼到第 <select id="sSel">${TRACE.steps.map((s, i) => `<option value="${i}" ${i === idx ? "selected" : ""}>${i + 1} 格</option>`).join("")}</select>格</label>
    </div>`;
}

function scanStory(step, idx) {
  if (idx === 0) {
    return `<div class="card"><h4 style="margin-top:0">第 1 格</h4><p>起点 = 一条还没看过的边。围着这条边扩展成 10 节点的窗格。<span class="mini">（此步无候选搜索）</span></p></div>`;
  }
  const best = step.best_score;
  const total = step.retained_total;
  const overlap = step.overlap_nodes || [];
  return `<div class="card">
    <h4 style="margin-top:0">第 ${idx + 1} 格是怎么选出来的</h4>
    <ol style="margin:6px 0;padding-left:18px;font-size:0.86rem">
      <li><b>搭法</b>：上一格 10 节点里「带走哪 3 个」，共 ${total} 种，筛掉不连通的。</li>
      <li><b>试选</b>：按「指向未看边的潜力」保留前 8 个候选，各自扩展成一整格。</li>
      <li><b>结果</b>：选中「新增 ${best[0]} 条真实边、${best[2]} 个新节点」的那格；与上一格重叠 ${overlap.length} 个节点。</li>
    </ol>
    <details>
      <summary>看完整候选与打分</summary>
      <div style="margin:8px 0" class="mini">打分顺序：新增边 → 新点对 → 新节点 → 格内边数。</div>
      ${table(["选中", "带走哪些节点", "新边", "新点对", "新节点", "格内边"], step.completed.map((c, i) => {
        const sel = c.retained.join(",") === (step.selected_retained || []).join(",");
        return [sel ? "✓" : "", c.retained.join(","), c.score[0], c.score[1], c.score[2], c.score[3]];
      }))}
    </details>
  </div>`;
}

function renderScan() {
  const idx = state.step;
  const step = TRACE.steps[idx];
  const progress = step.progress;
  const tiles = `
    <div class="ptiles">
      <div class="ptile"><div class="v">${pct(progress.edge_cover)}</div><div class="k">已看到的真实边</div></div>
      <div class="ptile"><div class="v">${pct(progress.node_cover)}</div><div class="k">已看到的节点</div></div>
      <div class="ptile"><div class="v">${idx > 0 ? step.overlap_nodes.length : "—"} <span class="mini">/ 3</span></div><div class="k">与上一格重叠</div></div>
      <div class="ptile"><div class="v">${idx + 1}</div><div class="k">当前第几格</div></div>
    </div>`;
  return `
    <div class="scan-grid">
      <div>
        ${scanControls(idx)}
        <svg id="svg-graph" viewBox="0 0 960 600"></svg>
        <div class="legend">
          <span><i style="background:#bc8cff"></i>当前窗口</span>
          <span><i style="background:#3fb950"></i>上一窗口</span>
          <span><i style="background:#d29922"></i>重叠（共享）节点</span>
          <span class="sw"><i style="background:#3fb950;border-radius:0;width:12px;height:3px"></i>这格新看到的边</span>
          <span class="sw"><i style="background:#58a6ff;border-radius:0;width:12px;height:3px"></i>已看到的边</span>
        </div>
        ${tiles}
      </div>
      <div>
        ${scanStory(step, idx)}
        <div class="card mini" style="margin-top:12px">
          演示用一张 40 节点的合成图（3 个稠密社区 + 稀疏连接）跑真实 Beam8 算法，与正式实现逐窗口一致。窗口 = 10 节点的局部子图，相邻窗口共享 3 个节点。
        </div>
      </div>
    </div>`;
}

// ---------------- Tab 3：怎么拼（拼接演示） ----------------
function edgeStats() {
  const map = {};
  TRACE.edges.forEach(([u, v]) => { map[edgeKey(u, v)] = 0; });
  const firstSeen = {};
  TRACE.steps.forEach((s, i) => {
    const ns = new Set(s.nodes);
    TRACE.edges.forEach(([u, v]) => {
      if (ns.has(u) && ns.has(v)) {
        const k = edgeKey(u, v);
        map[k] += 1;
        if (firstSeen[k] === undefined) firstSeen[k] = i;
      }
    });
  });
  return { map, firstSeen };
}
const ESTATS = (() => { const { map, firstSeen } = edgeStats(); return { multiplicity: map, firstSeen }; })();

function renderStitch() {
  const idx = stitchState.step;
  const totalEdges = TRACE.edges.length;
  const coveredNow = TRACE.steps.slice(0, idx + 1).reduce((acc, s) => {
    const ns = new Set(s.nodes);
    TRACE.edges.forEach(([u, v]) => { if (ns.has(u) && ns.has(v)) acc.add(edgeKey(u, v)); });
    return acc;
  }, new Set());
  const count = coveredNow.size;
  const multiCount = Object.values(ESTATS.multiplicity).filter(v => v > 1).length;
  const maxMulti = Math.max(...Object.values(ESTATS.multiplicity));

  const tiles = `
    <div class="ptiles">
      <div class="ptile"><div class="v">${count} <span class="mini">/ ${totalEdges}</span></div><div class="k">已拼上的真实边</div></div>
      <div class="ptile"><div class="v">${pct(count / totalEdges)}</div><div class="k">拼图完成度</div></div>
      <div class="ptile"><div class="v">${multiCount}</div><div class="k">被 ≥2 格看到的边（取平均）</div></div>
      <div class="ptile"><div class="v">${maxMulti}</div><div class="k">一条边最多被几格看到</div></div>
    </div>`;

  return `
    <div class="scan-grid">
      <div>
        ${stitchControls(idx)}
        <svg id="svg-graph" viewBox="0 0 960 600"></svg>
        <div class="legend">
          <span class="sw"><i style="background:#3fb950;border-radius:0;width:12px;height:3px"></i>这一格新拼上的边</span>
          <span class="sw"><i style="background:#58a6ff;border-radius:0;width:12px;height:3px"></i>之前已拼上的边</span>
          <span class="sw"><i style="background:#243044;border-radius:0;width:12px;height:3px"></i>还没看到的边</span>
        </div>
        ${tiles}
      </div>
      <div>
        <div class="card">
          <h4 style="margin-top:0">拼图是怎么做的</h4>
          <ol style="margin:6px 0;padding-left:18px;font-size:0.86rem">
            <li>每格都自带 <b>10 个节点的坐标位置</b>（即这些节点在整图里的编号）。</li>
            <li>按 <b>重叠对应表</b>，把相邻两格对齐——共享的 3 个节点就是对齐点。</li>
            <li>一格一格「贴」回整图：每条边第一次看到就记下，<b>被多格看到的边取平均</b>。</li>
            <li>全部贴完，得到整图重建。图上灰色 = 扫描没覆盖到的边（重建时会漏/错）。</li>
          </ol>
        </div>
        <div class="card mini" style="margin-top:12px">
          此演示逐格累加同一份 Beam8 trace：绿色 = 当前格新贡献的边，蓝色 = 之前已拼上，灰色 = 整条扫描链都没看到的边。重叠边取平均能提高一致性（这是「记账」correspondence 的用途）。
        </div>
      </div>
    </div>`;
}

// ---------------- Tab 4：关键数字 ----------------
function renderCompare() {
  const strategyBars = strategies.map(s =>
    barRow(s.name + (s.ok ? " ✓" : ""), s.cover, { bold: s.ok, cls: s.ok ? " hl" : "", color: s.ok ? "linear-gradient(90deg,#3fb950,#2ea043)" : "linear-gradient(90deg,#58a6ff,#8b949e)" })
  ).join("");
  const beamRows = beams.map((b, i) => [b.cell, fmt(b.sec), pct(b.edge), fmt(b.rmse)]);

  return `
  <h3 class="sec">1. 四种扫描方式：覆盖的真实边比例（越高越好）</h3>
  <div class="card"><div class="bars">${strategyBars}</div>
    <p class="mini"><b>数字怎么来的：</b>72 张图逐张扫描后统计「窗格内出现的真实边 / 全部真实边」。后两行（盯目标边 / Beam8）来自同一组对照实验，前两行来自早期探索（口径略有差异，供直觉对比）。</p>
  </div>

  <h3 class="sec">2. 端到端重建误差：旧法 vs Beam8（越低越好）</h3>
  <div class="card">
    ${waterfallView(waterfall)}
    <p class="mini"><b>数字怎么来的：</b>全图 RMSE = 对所有点对，比较「拼接重建的边」与「真实边」的差异，取均方根。同一批 72 张图、同样加 KSVD 压缩和补全：旧扫描链 0.385 → Beam8 0.324。</p>
  </div>

  <h3 class="sec">3. 候选越多越好吗？</h3>
  <div class="card">
    ${table(["策略", "耗时 (秒/图)", "边覆盖", "重建误差"], beamRows, { hl: 1 })}
    <p class="mini"><b>数字怎么来的：</b>固定 s10/o3/m1.5，只调候选数。候选 8 → 16 → 32 收益越来越小；8 个候选已拿到绝大部分收益、便宜 6.6 倍，因此冻结 B8。</p>
  </div>
  `;
}

// ---------------- Tab 5：附录 ----------------
function renderDetail() {
  return `
  <h3 class="sec">研究者向：完整参数与数据</h3>

  <details>
    <summary>v2 完整配置</summary>
    ${table(["组件", "取值"], appendix.spec)}
  </details>

  <details>
    <summary>覆盖与容量审计</summary>
    <h4 class="sub">参数不是随便选的：旧配置被两个 cell 同时支配</h4>
    ${table(["cell", "窗口数", "观察槽位", "边覆盖", "点对覆盖", "重建误差"], [
      ["旧 s10/o5/m1.5", "17.65", "794.4", pct(0.6870), pct(0.4925), fmt(0.3530)],
      ["s10/o3/m1.5", "17.65", "794.4", pct(0.7629), pct(0.5599), fmt(0.3064)],
      ["s12/o4/m1.5", "12.00", "792.0", pct(0.7424), pct(0.5492), fmt(0.3177)],
    ], { hl: 1 })}
    <p class="mini">同样的观察成本，三分之一重叠比一半重叠探索到更多新区域，因此 50% 重叠被淘汰。</p>
    <h4 class="sub">KSVD 容量（K24/T3）已是合理 Pareto 点</h4>
    <div class="kv">
      <span>当前</span><span>字典 1080 个标量 · 每图 52.96 个系数</span>
      <span>加大容量</span><span>K24/T4 误差 0.346（+33% 系数）、K32/T4 误差 0.337</span>
      <span>更多训练轮</span><span>25→50 轮只降 0.07%</span>
    </div>
  </details>

  <details>
    <summary>压缩与补全的精确数字</summary>
    ${table(["分支", "窗口误差", "观测 RMSE", "全图 RMSE", "全图 recall"], [
      ["旧 target o5", fmt(0.4928), fmt(0.3653), fmt(0.4367), pct(0.5909)],
      ["target o3", fmt(0.5021), fmt(0.3760), fmt(0.4162), pct(0.6442)],
      ["Beam8 o3", fmt(0.4319), fmt(0.3489), fmt(0.3349), pct(0.8137)],
    ], { hl: 2 })}
    <h4 class="sub">可选补全（train-only structural completion）</h4>
    <div class="kv">
      <span>Beam8 不压缩</span><span>0.2200 → 补全后 0.2031</span>
      <span>Beam8 + 压缩</span><span>0.3349 → 补全后 0.3241</span>
    </div>
  </details>

  <details>
    <summary>七轮探索历程</summary>
    ${table(["轮", "命题", "结论", "要点"], appendix.rounds.map(r => [r.r, r.name, `<span class="tag ${r.ok ? "ok" : "no"}">${r.verdict}</span>`, r.note]))}
  </details>

  <details>
    <summary>边界与下一步</summary>
    <h4 class="sub">尚未解决的边界</h4>
    <ol style="padding-left:18px;font-size:0.84rem">
      ${appendix.boundaries.map(b => `<li>${b}</li>`).join("")}
    </ol>
    <h4 class="sub">拿到真实 50-node 数据后优先做（无标签）</h4>
    <ol style="padding-left:18px;font-size:0.84rem">
      ${appendix.nextReal.map(n => `<li>${n}</li>`).join("")}
    </ol>
  </details>
  `;
}

// ---------------- 路由 ----------------
const BUILDERS = {
  read: renderRead,
  scan: () => `<div id="scan-root">${renderScan()}</div>`,
  stitch: () => `<div id="stitch-root">${renderStitch()}</div>`,
  compare: renderCompare,
  detail: renderDetail,
};

let current = "read";

function initTabs() {
  const tabs = $("tabs");
  tabs.innerHTML = "";
  TABS.forEach(([id, label]) => {
    const b = document.createElement("button");
    b.className = "tab" + (id === current ? " active" : "");
    b.textContent = label;
    b.onclick = () => { current = id; initTabs(); render(); };
    tabs.appendChild(b);
  });
}

function wire(id, idx, setStep) {
  const step = TRACE.steps[idx];
  const prevNodes = idx > 0 ? TRACE.steps[idx - 1].nodes : [];
  const nodeStyle = id === "scan" ? "scan" : "plain";
  drawGraph($("svg-graph"), {
    step, prevNodes,
    overlap: idx > 0 ? step.overlap_nodes || [] : [],
    coveredEdges: coveredEdgeSet(idx),
    stepEdges: stepEdgesSet(idx),
    nodeStyle,
  });
  const sel = $(id === "scan" ? "selStep" : "sSel");
  if (sel) sel.onchange = (e) => { setStep(+e.target.value); render(); };
  const prev = $(id === "scan" ? "btnPrev" : "sPrev");
  if (prev) prev.onclick = () => { setStep(Math.max(0, idx - 1)); render(); };
  const next = $(id === "scan" ? "btnNext" : "sNext");
  if (next) next.onclick = () => { setStep(Math.min(TRACE.steps.length - 1, idx + 1)); render(); };
  const play = $(id === "scan" ? "btnPlay" : "sPlay");
  const st = id === "scan" ? state : stitchState;
  if (play) play.onclick = () => {
    if (st.autoplay) { st.autoplay = false; clearInterval(st.timer); render(); return; }
    st.autoplay = true;
    st.timer = setInterval(() => {
      if (st.step >= TRACE.steps.length - 1) { st.autoplay = false; clearInterval(st.timer); st.step = 0; }
      else st.step += 1;
      render();
    }, 1100);
    render();
  };
}

function render() {
  const main = $("main");
  const panel = document.createElement("section");
  panel.className = "panel active";
  panel.innerHTML = BUILDERS[current]();
  main.innerHTML = "";
  main.appendChild(panel);
  if (current === "scan") requestAnimationFrame(() => wire("scan", state.step, (v) => { state.step = v; }));
  if (current === "stitch") requestAnimationFrame(() => wire("stitch", stitchState.step, (v) => { stitchState.step = v; }));
  $("footer").textContent =
    "数据来源：PATCH_CHAIN_ARBITRARINESS_ERROR_BUDGET_AUDIT_20260802.md · KSVD_CONTINUOUS_COVER_EXPLORATION_20260801.md · marginal_candidate_cover.py 真实采样 trace（beam8_trace.json）";
}

initTabs();
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
