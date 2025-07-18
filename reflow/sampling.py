# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""

import torch
import logging
from typing import Any
from reflow.utils import get_dcp_t

def init_sample(flow, sample_shape, z=None, device='cuda'):
    """Initialize samples."""
    if z is None:
        shape = sample_shape
        z0 = flow.get_z0(torch.zeros(shape, device=device), train=False).to(device)
        x = z0.detach().clone()
    else:
        shape = z.shape
        x = z
    return x, shape

@torch.no_grad()
def get_flow_sampler(flow, use_ode_sampler=None, device='cuda'):
    """
    Get rectified flow sampler

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    if use_ode_sampler is None:
        use_ode_sampler = flow.use_ode_sampler

    @torch.no_grad()
    def asm_one_step_sampler(model, hazy=None, **kwargs: Any,):
        """one_step_sampler.

        Args:
        model: A velocity model.
        z: If present, generate samples from latent code `z`.
        Returns:
        samples, number of function evaluations.
        """
        shape = hazy.shape
        x = hazy

        ### one step
        eps = flow.eps # default: 1e-3

        
        t = torch.from_numpy(get_dcp_t(x.squeeze().permute(1,2,0).cpu().numpy(), A1=False))[None, None, :, :].float().to(device)                
        t = torch.ones(shape[0], device=device) * t     
        
        if flow.cfg.flow.refine_t:
            pred,pred_t = flow.model_forward_wrapper(model, x, t, **kwargs) ### Copy from models/utils.py 
        else:
            pred = flow.model_forward_wrapper(model, x, t, **kwargs) ### Copy from models/utils.py 
            pred_t = None
        
        if 'x1' in flow.cfg.flow.consistency:   # predict x1
            x = pred
        else:
            # x = x + pred * ((flow.T-init_t) - eps)
            if flow.cfg.flow.refine_t:
                x = x + pred * ((flow.T-pred_t))
            else:
                x = x + pred * ((flow.T-t))
            # x = x - pred * ((t))*0.25
            # x = x - pred * t
        # x = inverse_scaler(x.clamp_(-1, 1))   # [0, 1]
        
        x = torch.clamp(x, 0., 1.)
        
        nfe = 1
        return x, pred_t
    
    @torch.no_grad()
    def asm_N_step_sampler(model, hazy=None, **kwargs: Any,):
        """one_step_sampler.

        Args:
        model: A velocity model.
        z: If present, generate samples from latent code `z`.
        Returns:
        samples, number of function evaluations.
        """
        shape = hazy.shape
        x = hazy
        ### Uniform
         
        # dt = 1.0 / flow.sample_N
        eps = flow.eps # default: 1e-3

        
        for i in list(range(flow.sample_N)):
            t = torch.from_numpy(get_dcp_t(x.squeeze().permute(1,2,0).cpu().numpy(), A1=False))[None, None, :, :].float().to(device)                
            t = torch.ones(shape[0], device=device) * t     
            if flow.cfg.flow.refine_t:
                pred,pred_t = flow.model_forward_wrapper(model, x, t, **kwargs) ### Copy from models/utils.py 
            else:
                pred = flow.model_forward_wrapper(model, x, t, **kwargs) ### Copy from models/utils.py 
                pred_t = None
            
            if 'x1' in flow.cfg.flow.consistency:   # predict x1
                x = pred
            else:
                # x = x + pred * ((flow.T-init_t) - eps)
                if flow.cfg.flow.refine_t:
                    x = x + pred * ((flow.T-pred_t)/ flow.sample_N)
                else:
                    x = x + pred * ((flow.T-t))
            
            x = torch.clamp(x, 0., 1.)
            
        return x, pred_t
    



    sample_N = flow.sample_N
    logging.info(f'Type of Sampler: {use_ode_sampler}; sample_N: {sample_N}')
    if use_ode_sampler == 'asm_one_step': 
        return asm_one_step_sampler
    elif use_ode_sampler == 'asm_N_step': 
        return asm_N_step_sampler
    else:
        assert False, 'Not Implemented!'
