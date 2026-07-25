"""Build interactive HTML with embedded pipeline data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_TRACK = Path(__file__).resolve().parents[1]
OUT = _TRACK / "results" / "viz_pipeline"


def main() -> int:
    data_path = OUT / "data.json"
    if not data_path.exists():
        print("Run: python -m code.viz_pipeline_data first")
        return 1
    data = json.loads(data_path.read_text(encoding="utf-8"))
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False))
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
<title>RW → KSVD 图级流水线可视化</title>
<style>
  :root {
    --bg: #0f1419;
    --panel: #1a2332;
    --border: #2d3a4d;
    --text: #e7ecf3;
    --muted: #8b9bb4;
    --accent: #3b82f6;
    --seed: #f59e0b;
    --walk: #22c55e;
    --patch: #a855f7;
    --danger: #ef4444;
    --edge: #64748b;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: "Segoe UI", system-ui, sans-serif;
    background: var(--bg); color: var(--text); min-height: 100vh;
  }
  header {
    padding: 14px 20px; border-bottom: 1px solid var(--border);
    background: var(--panel); display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
  }
  header h1 { font-size: 1.1rem; margin: 0; font-weight: 600; }
  header .sub { color: var(--muted); font-size: 0.85rem; }
  .tabs { display: flex; gap: 6px; flex-wrap: wrap; }
  .tab {
    background: transparent; border: 1px solid var(--border); color: var(--muted);
    padding: 6px 12px; border-radius: 8px; cursor: pointer; font-size: 0.85rem;
  }
  .tab.active { background: var(--accent); border-color: var(--accent); color: #fff; }
  .tab:hover:not(.active) { border-color: var(--accent); color: var(--text); }
  main { display: grid; grid-template-columns: 1fr 340px; gap: 0; min-height: calc(100vh - 60px); }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  #canvas-wrap {
    padding: 16px; display: flex; flex-direction: column; gap: 10px;
    border-right: 1px solid var(--border);
  }
  #svg {
    width: 100%; height: min(480px, 55vh); background: #0b1017;
    border: 1px solid var(--border); border-radius: 12px;
  }
  .controls {
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
    padding: 10px 12px; background: var(--panel); border-radius: 10px;
    border: 1px solid var(--border);
  }
  .controls label { font-size: 0.8rem; color: var(--muted); display: flex; align-items: center; gap: 6px; }
  select, input[type=range] { accent-color: var(--accent); }
  select {
    background: var(--bg); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 8px;
  }
  button.btn {
    background: var(--accent); color: #fff; border: none; border-radius: 8px;
    padding: 6px 12px; cursor: pointer; font-size: 0.85rem;
  }
  button.btn.secondary { background: var(--border); }
  button.btn:disabled { opacity: 0.4; cursor: not-allowed; }
  #side {
    padding: 16px; overflow-y: auto; background: #121a24;
  }
  #side h2 { font-size: 0.95rem; margin: 0 0 8px; }
  #side h3 { font-size: 0.8rem; color: var(--muted); margin: 14px 0 6px; text-transform: uppercase; letter-spacing: 0.04em; }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 12px; margin-bottom: 10px; font-size: 0.85rem; line-height: 1.45;
  }
  .kv { display: grid; grid-template-columns: 110px 1fr; gap: 4px 8px; }
  .kv span:first-child { color: var(--muted); }
  .bar-row { display: flex; align-items: center; gap: 8px; margin: 3px 0; font-size: 0.75rem; }
  .bar-row .lab { width: 52px; color: var(--muted); }
  .bar {
    flex: 1; height: 10px; background: #0b1017; border-radius: 4px; overflow: hidden;
  }
  .bar i { display: block; height: 100%; background: linear-gradient(90deg, #3b82f6, #a855f7); }
  .heat {
    display: inline-grid; gap: 1px; background: var(--border); padding: 1px;
    border-radius: 4px; margin: 4px 6px 4px 0;
  }
  .heat div { width: 14px; height: 14px; }
  .legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 0.75rem; color: var(--muted); }
  .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.75rem; background: #243044; }
  .tag.ok { background: #14532d; color: #86efac; }
  .tag.no { background: #450a0a; color: #fca5a5; }
  .node-label { font-size: 11px; fill: #fff; font-weight: 600; pointer-events: none; }
  footer { padding: 8px 16px; color: var(--muted); font-size: 0.75rem; border-top: 1px solid var(--border); }
</style>
</head>
<body>
<header>
  <div>
    <h1>图级 RW → KSVD 流水线</h1>
    <div class="sub">Coverage 采样 · 诱导子图 · 共享字典 · readout（交互演示）</div>
  </div>
  <div class="tabs" id="tabs"></div>
</header>
<main>
  <div id="canvas-wrap">
    <div class="controls" id="controls"></div>
    <svg id="svg" viewBox="0 0 500 400"></svg>
    <div class="legend">
      <span><i style="background:#f59e0b"></i>种子/起点</span>
      <span><i style="background:#22c55e"></i>当前 walk / S</span>
      <span><i style="background:#a855f7"></i>诱导边</span>
      <span><i style="background:#64748b"></i>未走边</span>
      <span><i style="background:#ef4444"></i>已降权边（越淡权越小）</span>
    </div>
  </div>
  <aside id="side"></aside>
</main>
<footer id="footer"></footer>
<script>
const DATA = /*__DATA__*/;

let state = {
  step: 0,
  graphKey: "c4",
  walkId: 0,
  stepT: 0,
  patchId: 0,
  atomId: 0,
  playing: false,
  timer: null,
};

const steps = DATA.steps_ui || [];

function $(id) { return document.getElementById(id); }

function initTabs() {
  const tabs = $("tabs");
  tabs.innerHTML = "";
  steps.forEach((s, i) => {
    const b = document.createElement("button");
    b.className = "tab" + (i === state.step ? " active" : "");
    b.textContent = s.title;
    b.onclick = () => { state.step = i; state.playing = false; clearInterval(state.timer); render(); };
    tabs.appendChild(b);
  });
}

function G() { return DATA.graphs[state.graphKey]; }

function edgeKey(a, b) {
  return a < b ? a + "-" + b : b + "-" + a;
}

function weightColor(w, maxW=1) {
  // low weight -> red/faded
  const t = Math.max(0, Math.min(1, w / maxW));
  const alpha = 0.25 + 0.75 * t;
  return `rgba(239,68,68,${alpha.toFixed(2)})`;
}

function drawGraph(opts) {
  const g = G();
  const svg = $("svg");
  const layout = g.layout;
  const highlightS = new Set(opts.S || []);
  const seed = opts.seed;
  const traj = opts.trajEdges || []; // list of [u,v]
  const trajSet = new Set(traj.map(e => edgeKey(e[0], e[1])));
  const induced = opts.induced !== false;
  const weights = opts.weights || null;
  const pathPrefix = opts.pathPrefix || null; // nodes in order so far

  let edgesHtml = "";
  g.edges.forEach(([u, v]) => {
    const k = edgeKey(u, v);
    const p1 = layout[String(u)], p2 = layout[String(v)];
    let stroke = "#334155", sw = 2, op = 0.9;
    const both = highlightS.has(u) && highlightS.has(v);
    if (weights && weights[k] !== undefined) {
      stroke = weightColor(weights[k]);
      sw = 2.5;
    }
    if (trajSet.has(k)) {
      stroke = "#22c55e";
      sw = 3.5;
      op = 1;
    } else if (both && induced && highlightS.size) {
      stroke = "#a855f7";
      sw = 3;
    }
    edgesHtml += `<line x1="${p1[0]}" y1="${p1[1]}" x2="${p2[0]}" y2="${p2[1]}"
      stroke="${stroke}" stroke-width="${sw}" opacity="${op}" stroke-linecap="round"/>`;
  });

  // path order numbers
  let pathHtml = "";
  if (pathPrefix && pathPrefix.length > 1) {
    for (let i = 0; i < pathPrefix.length - 1; i++) {
      const u = pathPrefix[i], v = pathPrefix[i+1];
      const p1 = layout[String(u)], p2 = layout[String(v)];
      const mx = (p1[0]+p2[0])/2, my = (p1[1]+p2[1])/2;
      pathHtml += `<circle cx="${mx}" cy="${my}" r="8" fill="#14532d" stroke="#22c55e"/>
        <text x="${mx}" y="${my+3}" text-anchor="middle" class="node-label" font-size="9">${i+1}</text>`;
    }
  }

  let nodesHtml = "";
  g.nodes.forEach(u => {
    const p = layout[String(u)];
    let fill = "#3b82f6";
    let r = 14;
    if (highlightS.has(u)) { fill = "#22c55e"; r = 16; }
    if (seed !== undefined && u === seed) { fill = "#f59e0b"; r = 18; }
    if (opts.current === u) { fill = "#eab308"; r = 19; }
    nodesHtml += `<circle cx="${p[0]}" cy="${p[1]}" r="${r}" fill="${fill}"
      stroke="#0f172a" stroke-width="2"/>
      <text x="${p[0]}" y="${p[1]+4}" text-anchor="middle" class="node-label">${u}</text>`;
  });

  svg.innerHTML = edgesHtml + pathHtml + nodesHtml;
}

function bars(values, labels) {
  const max = Math.max(...values.map(Math.abs), 1e-9);
  return values.map((v, i) => {
    const lab = labels ? labels[i] : ("a"+i);
    const pct = (Math.abs(v) / max * 100).toFixed(1);
    return `<div class="bar-row"><span class="lab">${lab}</span>
      <div class="bar"><i style="width:${pct}%"></i></div>
      <span>${v.toFixed(3)}</span></div>`;
  }).join("");
}

function heatMatrix(mat) {
  let flat = mat.flat();
  const max = Math.max(...flat.map(Math.abs), 1e-9);
  const n = mat.length;
  let cells = "";
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = mat[i][j];
      const t = Math.abs(v) / max;
      const c = v >= 0
        ? `rgba(59,130,246,${(0.15+0.85*t).toFixed(2)})`
        : `rgba(239,68,68,${(0.15+0.85*t).toFixed(2)})`;
      cells += `<div style="background:${c}" title="${i},${j}: ${v.toFixed(3)}"></div>`;
    }
  }
  return `<div class="heat" style="grid-template-columns:repeat(${n},14px)">${cells}</div>`;
}

function renderControls() {
  const c = $("controls");
  const g = G();
  const sample = g.sample;
  let html = `
    <label>图
      <select id="selGraph">
        <option value="c4" ${state.graphKey==="c4"?"selected":""}>C4 demo (有四元环)</option>
        <option value="c8" ${state.graphKey==="c8"?"selected":""}>C8 demo (无四元环)</option>
      </select>
    </label>`;

  const sid = steps[state.step].id;
  if (sid === "walk" || sid === "patches" || sid === "encode") {
    html += `<label>Walk
      <select id="selWalk">
        ${sample.walks.map((w,i)=>`<option value="${i}" ${i===state.walkId?"selected":""}>#${i} start=${w.start} |S|=${w.S.length} C4=${w.has_c4}</option>`).join("")}
      </select>
    </label>`;
  }
  if (sid === "walk") {
    const w = sample.walks[state.walkId];
    const maxT = w.steps.length;
    html += `<label>步骤 <input type="range" id="rngStep" min="0" max="${maxT}" value="${Math.min(state.stepT,maxT)}"/>
      <span id="stepLab">${Math.min(state.stepT,maxT)}/${maxT}</span></label>
      <button class="btn" id="btnPlay">${state.playing ? "暂停" : "播放游走"}</button>
      <button class="btn secondary" id="btnReset">重置</button>`;
  }
  if (sid === "patches") {
    html += `<label>Patch
      <select id="selPatch">
        ${sample.walks.map((w,i)=>`<option value="${i}" ${i===state.patchId?"selected":""}>patch ${i} S=[${w.S}] C4=${w.has_c4}</option>`).join("")}
      </select>
    </label>`;
  }
  if (sid === "dict" && g.ksvd) {
    html += `<label>Atom
      <select id="selAtom">
        ${g.ksvd.atoms.map((a,i)=>`<option value="${i}" ${i===state.atomId?"selected":""}>atom ${i} usage=${a.usage.toFixed(2)}</option>`).join("")}
      </select>
    </label>`;
  }
  if (sid === "encode" && g.ksvd) {
    html += `<label>Patch 系数
      <select id="selPatch">
        ${g.ksvd.patch_coefs.map((p,i)=>`<option value="${i}" ${i===state.patchId?"selected":""}>patch ${i} recon=${p.recon_err.toFixed(3)}</option>`).join("")}
      </select>
    </label>`;
  }
  c.innerHTML = html;

  const sg = $("selGraph");
  if (sg) sg.onchange = (e) => { state.graphKey = e.target.value; state.walkId=0; state.stepT=0; state.patchId=0; render(); };
  const sw = $("selWalk");
  if (sw) sw.onchange = (e) => { state.walkId = +e.target.value; state.stepT=0; render(); };
  const sp = $("selPatch");
  if (sp) sp.onchange = (e) => { state.patchId = +e.target.value; render(); };
  const sa = $("selAtom");
  if (sa) sa.onchange = (e) => { state.atomId = +e.target.value; render(); };
  const rs = $("rngStep");
  if (rs) rs.oninput = (e) => { state.stepT = +e.target.value; $("stepLab").textContent = state.stepT + "/" + rs.max; renderCanvasOnly(); updateSidePartial(); };
  const bp = $("btnPlay");
  if (bp) bp.onclick = togglePlay;
  const br = $("btnReset");
  if (br) br.onclick = () => { state.stepT = 0; state.playing = false; clearInterval(state.timer); render(); };
}

function togglePlay() {
  const g = G();
  const w = g.sample.walks[state.walkId];
  if (state.playing) {
    state.playing = false;
    clearInterval(state.timer);
    render();
    return;
  }
  state.playing = true;
  state.timer = setInterval(() => {
    if (state.stepT >= w.steps.length) {
      state.playing = false;
      clearInterval(state.timer);
      render();
      return;
    }
    state.stepT += 1;
    renderCanvasOnly();
    updateSidePartial();
    const rs = $("rngStep");
    if (rs) { rs.value = state.stepT; $("stepLab").textContent = state.stepT + "/" + rs.max; }
  }, 650);
  render();
}

function renderCanvasOnly() {
  // lightweight redraw for animation
  const sid = steps[state.step].id;
  if (sid === "walk") drawWalkStep();
}

function updateSidePartial() {
  if (steps[state.step].id === "walk") {
    // re-render side for walk step info only - full renderSide is fine
    renderSide();
  }
}

function drawWalkStep() {
  const g = G();
  const w = g.sample.walks[state.walkId];
  const t = Math.min(state.stepT, w.steps.length);
  const path = w.walk.slice(0, t + 1);
  const traj = w.traj_edges.slice(0, t);
  const S = new Set(path);
  // approximate weights: start from weights_before of this walk, apply decay for completed steps
  let weights = Object.assign({}, w.weights_before);
  const decay = DATA.config.edge_decay;
  for (let i = 0; i < t; i++) {
    const e = w.traj_edges[i];
    const k = edgeKey(e[0], e[1]);
    if (weights[k] !== undefined) weights[k] *= decay;
  }
  const cur = path[path.length - 1];
  drawGraph({
    S: [...S],
    seed: w.start,
    trajEdges: traj,
    pathPrefix: path,
    weights,
    current: cur,
    induced: false,
  });
}

function renderSide() {
  const side = $("side");
  const g = G();
  const cfg = DATA.config;
  const sid = steps[state.step].id;
  let html = `<h2>${steps[state.step].title}</h2>`;

  if (sid === "seeds") {
    const starts = g.sample.walks.map(w => w.start);
    html += `<div class="card">
      <div class="kv">
        <span>图</span><span>${g.name} · ${g.label}</span>
        <span>全局含 C4</span><span class="tag ${g.has_c4_global?"ok":"no"}">${g.has_c4_global}</span>
        <span>种子策略</span><span>${cfg.seed_policy}</span>
        <span>预算 max_walks</span><span>${cfg.max_walks}</span>
        <span>实际 walks</span><span>${g.sample.n_walks}</span>
        <span>轨迹覆盖</span><span>${(g.sample.traj_cover*100).toFixed(1)}%</span>
        <span>硬删边</span><span class="tag ok">否（仅软降权）</span>
      </div>
    </div>
    <h3>本图种子（各 walk 起点）</h3>
    <div class="card">起点序列: <b>[${starts.join(", ")}]</b><br/>
    度分层轮转抽样，不必每个节点都走。</div>
    <h3>参数</h3>
    <div class="card">p=${cfg.p}, q=${cfg.q}, L=${cfg.walk_length}, m=${cfg.max_nodes}, decay=${cfg.edge_decay}</div>`;
    drawGraph({ seed: starts[0], S: starts });
  }

  if (sid === "walk") {
    const w = g.sample.walks[state.walkId];
    const t = Math.min(state.stepT, w.steps.length);
    const step = t > 0 ? w.steps[t-1] : null;
    html += `<div class="card">
      <div class="kv">
        <span>Walk</span><span>#${state.walkId}</span>
        <span>起点 seed</span><span style="color:var(--seed)">${w.start}</span>
        <span>当前步</span><span>${t} / ${w.steps.length}</span>
        <span>节点序列</span><span>[${w.walk.slice(0,t+1).join(" → ")}]</span>
        <span>最终 S</span><span>[${w.S.join(", ")}]</span>
        <span>S 含 C4</span><span class="tag ${w.has_c4?"ok":"no"}">${w.has_c4}</span>
        <span>走完后 cover</span><span>${(w.cover_after*100).toFixed(1)}%</span>
      </div>
    </div>`;
    if (step) {
      html += `<h3>本步转移</h3><div class="card">
        ${step.from} → ${step.to}，边权(走前)=${step.edge_weight_before.toFixed(3)}<br/>
        候选 (node, α, w, score):<br/>
        <code style="font-size:0.75rem">${JSON.stringify(step.candidates)}</code>
      </div>`;
    } else {
      html += `<div class="card">拖动滑条或点「播放游走」逐步查看。绿边=已走轨迹；橙点=起点；黄点=当前位置。</div>`;
    }
    html += `<h3>软降权</h3><div class="card">每走完一条边：weight *= ${cfg.edge_decay}。边不会从图中删除。下一 walk 更少走已走边。</div>`;
    drawWalkStep();
  }

  if (sid === "patches") {
    const w = g.sample.walks[state.patchId];
    html += `<div class="card">
      <div class="kv">
        <span>Patch</span><span>#${state.patchId}</span>
        <span>S</span><span>[${w.S.join(", ")}]</span>
        <span>诱导子图</span><span>G[S] 仅 S 内部边（紫色）</span>
        <span>含 C4</span><span class="tag ${w.has_c4?"ok":"no"}">${w.has_c4}</span>
        <span>‖y‖</span><span>${w.y_norm.toFixed(3)}</span>
      </div>
    </div>
    <h3>全部 patches</h3><div class="card">`;
    g.sample.walks.forEach((ww,i) => {
      html += `<div>patch ${i}: S=[${ww.S}] <span class="tag ${ww.has_c4?"ok":"no"}">C4=${ww.has_c4}</span></div>`;
    });
    html += `</div>
    <div class="card">每个 patch 展平为向量 y（邻接 pad 上三角），堆成矩阵 Y 的列，供 KSVD 使用。</div>`;
    drawGraph({ S: w.S, seed: w.start, trajEdges: w.traj_edges, induced: true });
  }

  if (sid === "dict") {
    if (!g.ksvd) {
      html += `<div class="card">当前图无训练字典（C8 用 C4 的 D 编码，见 Encode）。请切换到 C4 图查看 D。</div>`;
      drawGraph({});
    } else {
      const k = g.ksvd;
      const atom = k.atoms[state.atomId];
      html += `<div class="card">
        <div class="kv">
          <span>Y</span><span>${k.Y_shape.join(" × ")}（特征 × patch 数）</span>
          <span>D</span><span>${k.D_shape.join(" × ")}（特征 × 原子）</span>
          <span>X</span><span>${k.X_shape.join(" × ")}</span>
          <span>重构 ‖Y-DX‖/‖Y‖</span><span>${k.recon_rel.toFixed(4)}</span>
        </div>
      </div>
      <h3>Atom ${state.atomId}</h3>
      <div class="card">
        usage=${atom.usage.toFixed(3)}, energy=${atom.energy.toFixed(3)}<br/>
        列向量 reshape 为 ${atom.matrix.length}×${atom.matrix.length} 对称热力（上三角字典分量）：<br/>
        ${heatMatrix(atom.matrix)}
      </div>
      <h3>所有原子 usage</h3>
      <div class="card">${bars(k.atoms.map(a=>a.usage), k.atoms.map((_,i)=>"a"+i))}</div>
      <div class="card">K-SVD：交替 (1) 固定 D 稀疏编码 X；(2) 逐列更新 D。原子=「可复用的局部结构基」。</div>`;
      // show graph with all S union
      const allS = new Set();
      g.sample.walks.forEach(w => w.S.forEach(x => allS.add(x)));
      drawGraph({ S: [...allS] });
    }
  }

  if (sid === "encode") {
    if (g.ksvd) {
      const k = g.ksvd;
      const pc = k.patch_coefs[state.patchId] || k.patch_coefs[0];
      const w = g.sample.walks[state.patchId] || g.sample.walks[0];
      html += `<div class="card">
        <div class="kv">
          <span>编码</span><span>x = OMP(D, y)，稀疏度 T</span>
          <span>Patch</span><span>#${state.patchId} S=[${w.S}]</span>
          <span>patch 重构误差</span><span>${pc.recon_err.toFixed(4)}</span>
        </div>
      </div>
      <h3>|系数| |x|</h3>
      <div class="card">${bars(pc.abs_coef, pc.abs_coef.map((_,i)=>"a"+i))}</div>
      <h3>图向量 s_G（readout）</h3>
      <div class="card">
        dim=${k.s_G.length}<br/>
        含 rich 统计 + energy + usage（录音「按能量 readout」）<br/>
        ${bars(k.s_G.slice(0, Math.min(18, k.s_G.length)), null)}
        <span class="sub">（仅显示前 ${Math.min(18,k.s_G.length)} 维）</span>
      </div>
      <div class="card">分类时用 s_G → 标准化 + 线性分类器。字典 D 只在 train 图上学习。</div>`;
      drawGraph({ S: w.S, seed: w.start, trajEdges: w.traj_edges, induced: true });
    } else if (g.encode_with_c4_D) {
      const pc = g.encode_with_c4_D[state.patchId] || g.encode_with_c4_D[0];
      const w = g.sample.walks[state.patchId] || g.sample.walks[0];
      html += `<div class="card">C8 使用 <b>在 C4 上训练的 D</b> 编码（演示迁移）。</div>
        <h3>|系数|</h3><div class="card">${bars(pc.abs_coef, pc.abs_coef.map((_,i)=>"a"+i))}</div>
        <div class="card">S=[${w.S}] C4=${w.has_c4}</div>`;
      drawGraph({ S: w.S, seed: w.start, induced: true });
    }
  }

  if (sid === "compare") {
    const b0 = g.b0;
    const b0c4 = b0.filter(p => p.has_c4).length;
    const rwc4 = g.sample.walks.filter(w => w.has_c4).length;
    html += `<div class="card">
      <div class="kv">
        <span>B0 patches</span><span>${b0.length}（每点一星形）</span>
        <span>B0 含 C4 的 patch 数</span><span class="tag ${b0c4?"ok":"no"}">${b0c4}</span>
        <span>RW patches</span><span>${g.sample.walks.length}</span>
        <span>RW 含 C4 的 patch 数</span><span class="tag ${rwc4?"ok":"no"}">${rwc4}</span>
      </div>
    </div>
    <h3>B0 示例（点 0 的 1-hop）</h3>
    <div class="card">中心=0 时 S=[${(b0.find(p=>p.center===0)||b0[0]).S}] has_C4=${(b0.find(p=>p.center===0)||b0[0]).has_c4}<br/>
    环上点的 1-hop 诱导多为路径 P3，看不见闭合四元环。</div>
    <h3>切换</h3>
    <div class="card">
      <button class="btn secondary" id="btnB0">显示 B0 @ center 0</button>
      <button class="btn" id="btnRW">显示 RW patch 0</button>
    </div>`;
    const p0 = b0.find(p => p.center === 0) || b0[0];
    drawGraph({ S: p0.S, seed: p0.center, induced: true });
    setTimeout(() => {
      const b1 = $("btnB0");
      const b2 = $("btnRW");
      if (b1) b1.onclick = () => {
        const p = g.b0.find(x => x.center === 0) || g.b0[0];
        drawGraph({ S: p.S, seed: p.center, induced: true });
      };
      if (b2) b2.onclick = () => {
        const w = g.sample.walks[0];
        drawGraph({ S: w.S, seed: w.start, trajEdges: w.traj_edges, induced: true });
      };
    }, 0);
  }

  side.innerHTML = html;
}

function render() {
  initTabs();
  renderControls();
  renderSide();
  $("footer").textContent =
    `配置: p=${DATA.config.p} q=${DATA.config.q} L=${DATA.config.walk_length} m=${DATA.config.max_nodes} decay=${DATA.config.edge_decay} · ` +
    `硬删边=否 · 数据: results/viz_pipeline/data.json`;
}

render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
