(function () {
  function round(value) {
    return Number(value).toFixed(3);
  }

  function nodeStats(rows, lambda) {
    const G = rows.reduce((sum, row) => sum + row.g, 0);
    const H = rows.reduce((sum, row) => sum + row.h, 0);
    return { G, H, weight: -G / (H + lambda) };
  }

  function initCalculator(root) {
    const data = JSON.parse(root.querySelector('script[type="application/json"]').textContent);
    const lambda = Number(root.dataset.lambda || 1);
    const gamma = Number(root.dataset.gamma || 0);
    const output = root.querySelector('.calc-output');
    const parent = nodeStats(data, lambda);

    root.querySelectorAll('button[data-cut]').forEach((button) => {
      button.addEventListener('click', () => {
        const cut = Number(button.dataset.cut);
        const left = data.filter((row) => row.x <= cut);
        const right = data.filter((row) => row.x > cut);
        const leftStats = nodeStats(left, lambda);
        const rightStats = nodeStats(right, lambda);
        const gain = 0.5 * (
          (leftStats.G ** 2) / (leftStats.H + lambda) +
          (rightStats.G ** 2) / (rightStats.H + lambda) -
          (parent.G ** 2) / (parent.H + lambda)
        ) - gamma;
        root.querySelectorAll('button[data-cut]').forEach((item) => {
          item.setAttribute('aria-pressed', item === button ? 'true' : 'false');
        });
        output.innerHTML = `<strong>x &le; ${cut}</strong>：` +
          `左叶 G=${round(leftStats.G)}，w*=${round(leftStats.weight)}；` +
          `右叶 G=${round(rightStats.G)}，w*=${round(rightStats.weight)}。<br>` +
          `正则化后的 gain = <strong>${round(gain)}</strong>；` +
          (gain > 0 ? '值得保留这个切分。' : '不值得保留这个切分。');
      });
    });
  }

  function init() {
    document.querySelectorAll('.xgb-calculator').forEach(initCalculator);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
}());
