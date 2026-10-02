"""Repository integration scaffold, source-audited but not run with PyTorch here.

Call export_split only from a NEW registered research runner. Does not fit or
retrain the backbone, prepare dictionaries, or load official test.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256_file(path):
    hasher=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            hasher.update(block)
    return hasher.hexdigest()


def export_split(split, destination):
    if split not in ('train','valid'):
        raise RuntimeError('official test is forbidden')
    import torch
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_scale_v1 as scale
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
    from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
    torch.set_num_threads(8)
    checkpoint=scale.CHECKPOINT_DIR / 'SCALE-FULL-seed0_soup_state.pt'
    if not checkpoint.is_file():
        raise RuntimeError('existing Full soup absent: do not retrain it for this screen')
    checkpoint_sha=sha256_file(checkpoint)
    model=scale._soup_model().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    expected=10000 if split=='train' else 1000
    data=p1run.load_split(split)
    if len(data)!=expected:
        raise RuntimeError('unexpected official subset size')
    captured=[]
    # Actual tensor reaching the old reader, not a hand-rebuilt approximation.
    def capture(_module,args):
        value=args[0]
        if value.ndim!=2 or value.shape[1]!=814:
            raise RuntimeError('Full reader seam no longer has width 814')
        captured.append(value.detach().cpu().double().numpy().copy())
    handle=model.reader.register_forward_pre_hook(capture)
    labels,predictions=[],[]
    try:
        with torch.no_grad():
            for batch in p1.make_env_loader(data,128,False,0):
                prediction=model(batch,mask=cm.C6_MASK).view(-1)
                predictions.append(prediction.cpu().double().numpy())
                labels.append(batch.y.view(-1).cpu().double().numpy())
    finally:
        handle.remove()
    r,y,pred=np.concatenate(captured),np.concatenate(labels),np.concatenate(predictions)
    if r.shape!=(expected,814):
        raise RuntimeError('reader hook call count or graph row alignment mismatch')
    path=Path(destination)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise RuntimeError('refusing to overwrite feature cache')
    # Use only stable official row IDs here. Agent can attach the existing
    # verified train canonical group_ids after checking its row provenance.
    np.savez(path,R=r,y=y,p_base=pred,ids=np.arange(expected),
             split=np.asarray(split),official_test_loaded=np.asarray(False),
             frozen_backbone=np.asarray(True),checkpoint_sha=np.asarray(checkpoint_sha))
    report=dict(split=split,n_graphs=expected,R_width=814,device='cpu',threads=8,
                checkpoint_path=str(checkpoint),checkpoint_sha=checkpoint_sha,
                cache_sha=sha256_file(path),original_MAE=float(np.mean(abs(y-pred))),
                canonical_duplicate_grouping='not attached by scaffold',
                official_test_loaded=False,
                note='Integration scaffold; Agent must add split/cache/revision provenance and replay gates')
    path.with_suffix('.provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
