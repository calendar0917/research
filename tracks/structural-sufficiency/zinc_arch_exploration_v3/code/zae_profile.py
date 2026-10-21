"""One-off GPU timing breakdown for the exploration-v3 arms (diagnostic)."""
import sys, time, numpy as np, torch
R = 'tracks/structural-sufficiency/zinc_arch_exploration_v3/code'
V1 = 'tracks/structural-sufficiency/zinc_arch_discovery_v1/code'
sys.path[:0] = [R, V1]
import ad_common as C, ad_data as D, zae_data as ZD, zae_models as ZM
dev = torch.device('cuda:0')
norm = D.load_norm(); vc = D.load_cache('valid'); vp = ZD.load_pair_split('valid')
rows = np.arange(64)
t0 = time.perf_counter(); b = ZD.collate_pair(vc, vp, rows); print('collate ms', (time.perf_counter()-t0)*1e3)
b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
for arm in ('A','B'):
    m, _ = ZM.build_model(arm, norm, 0, dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad(); y = m(b); y.abs().mean().backward(); opt.step()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(10):
        opt.zero_grad(); y = m(b); y.abs().mean().backward(); opt.step()
    torch.cuda.synchronize(); print(arm, 'step ms', (time.perf_counter()-t0)/10*1e3)
    if arm == 'B':
        # part timing
        with torch.no_grad():
            a,bb,w = m.encoders(b); s = m.endpoint(a, w, b)
            torch.cuda.synchronize(); t=time.perf_counter()
            for _ in range(10): ops = ZM.build_kernels(b['adj_dense'], b['adj_typed'], b['node_mask'])
            torch.cuda.synchronize(); print('  kernels ms', (time.perf_counter()-t)/10*1e3)
            t=time.perf_counter()
            for _ in range(10): y = m(b)
            torch.cuda.synchronize(); print('  fwd ms', (time.perf_counter()-t)/10*1e3)
