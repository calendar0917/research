"""PyTorch deployment scaffold; must be tested in the repository environment.

This file was syntax-checked, not executed with PyTorch in the source audit.
Replace the old Full reader only after its state is loaded and cached fitting
has been frozen. Save state_dict, not a pickled module instance.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn


class GraphPrototypeReadout(nn.Module):
    def __init__(self, fitted_npz):
        super().__init__()
        with np.load(Path(fitted_npz),allow_pickle=False) as z:
            values = z['inverse_root'] @ z['coef'][1:]
            offset = float(z['target_median'])+float(z['coef'][0])
            for key in ('mean','scale','weights','centers'):
                self.register_buffer(key,torch.tensor(z[key],dtype=torch.float64))
            self.register_buffer('keep',torch.tensor(z['keep'],dtype=torch.bool))
            self.register_buffer('bandwidth_squared',torch.tensor(float(z['bandwidth_squared']),dtype=torch.float64))
            self.register_buffer('values',torch.tensor(values,dtype=torch.float64))
            self.register_buffer('offset',torch.tensor(offset,dtype=torch.float64))
        if self.mean.numel()!=814 or self.centers.shape[0]!=256:
            raise RuntimeError('fitted dictionary is not the frozen Full prototype spec')

    def forward(self, raw):
        if raw.ndim!=2 or raw.shape[1]!=814:
            raise RuntimeError('Full reader seam changed')
        z=((torch.asinh(raw.double())-self.mean)/self.scale*self.weights)[:,self.keep]
        d2=(z.square().sum(1,keepdim=True)+self.centers.square().sum(1).unsqueeze(0)
            -2*z@self.centers.t()).clamp_min(0.)
        kernel=torch.exp(-d2/(2*self.bandwidth_squared))
        return (kernel@self.values+self.offset).to(raw.dtype)
