from typing import Any, Iterable, Tuple, Union
import numpy as np
import torch
from torch import Tensor, nn
import logging
from models.ema import ExponentialMovingAverage
from reflow.losses import get_rectified_flow_loss_fn
import random 
from reflow.utils import get_dcp_t

class RectifiedFlow():
    def __init__(self, model=None, ema_model=None, cfg=None):
        self.cfg = cfg
        self.model = model
        ## init ema model
        if ema_model == None:
            # self.ema_model = copy.deepcopy(self.model)
            self.ema_model = ExponentialMovingAverage(self.model.parameters(), decay=self.cfg.model.ema_rate)
        else:
            self.ema_model = ema_model
        self.device = self.cfg.device
        
        if 'lpips' in self.cfg.training.loss_type:
            import lpips
            self.lpips_model = lpips.LPIPS(net='vgg')
            self.lpips_model = self.lpips_model.cuda()
            for p in self.lpips_model.parameters():
                p.requires_grad = False
        ## parameters
        self.eps = self.cfg.flow.eps
        self.use_ode_sampler = self.cfg.sampling.use_ode_sampler
        self.init_type = self.cfg.sampling.init_type
        self.sample_N = self.cfg.sampling.sample_N
        self.ode_tol = self.cfg.sampling.ode_tol
        self.noise_scale = self.cfg.sampling.init_noise_scale
        try:
            self.flow_t_schedule = int(self.cfg.flow.flow_t_schedule)
        except:
            self.flow_t_schedule = self.cfg.flow.flow_t_schedule
        self.flow_alpha_t = self.cfg.flow.flow_alpha_t
        ## get loss function
        self.loss_fn = get_rectified_flow_loss_fn(self.cfg.training.reduce_mean, self.cfg.training.loss_type)
        self.loss_fn_t = get_rectified_flow_loss_fn(self.cfg.training.reduce_mean, 'l2') ## loss for refined_t
        if self.cfg.flow.use_teacher:
            self.loss_fn_teach = get_rectified_flow_loss_fn(self.cfg.training.reduce_mean, self.cfg.training.loss_type)
        # Initialize the _T instance variable to a default value
        self._T = 1.
        ## x0 randomness
        if 'warmup' in self.cfg.training.x0_randomness:
            self.warmup_iters = int(self.cfg.training.x0_randomness.split('_')[-1])
            logging.info(f'x0_randomness warmup type: {self.cfg.training.x0_randomness}; warmup_iters: {self.warmup_iters}')
        else:
            logging.info(f'x0_randomness type: {self.cfg.training.x0_randomness}')

        logging.info(f'Init. Distribution Variance: {self.noise_scale}')
        logging.info(f'SDE Sampler Variance: 0 for flow')
        logging.info(f'ODE Tolerence: {self.ode_tol}')

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        self._T = value

    def lpips_forward_wrapper(self, x, y, size=224):
        # resize (B, C, H, W) to (B, C, 224, 224)
        if size > 0:
            x = nn.functional.interpolate(x, size=(size, size), mode='bilinear', align_corners=True)
            y = nn.functional.interpolate(y, size=(size, size), mode='bilinear', align_corners=True)
        return self.lpips_model(x, y, normalize=True) if 'lpips' in self.cfg.training.loss_type else None
        

    def model_forward_wrapper(
        self,
        model: nn.Module,
        x: Tensor,
        t: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Wrapper for the model call"""
        if self.cfg.flow.refine_t:
            model_output,output_t = model(x= x, time_cond=t*999,use_refine=True) 
            model_output = model_output[0] if isinstance(model_output, tuple) else model_output
            return model_output, output_t
        else:
            model_output = model(x, t*999)
            model_output = model_output[0] if isinstance(model_output, tuple) else model_output
            return model_output
    
    

    
    def get_asm_distill_pair(self, batch):
        self.data = batch['gt']
        self.noise = batch['A']# [:,:,None,None]
        self.xt = batch['hazy']
        self.t = batch['t']
        self.gt_t = batch['gt_t']
        
    def get_asm_reflow_pair(self, batch): 
        self.xt = batch['hazy']
        self.t = batch['t']
        self.noise = batch['A']
    
    def get_asm_data_pair(self, batch):
        # construct data pair
        self.data = batch['gt']
        self.noise = batch['A']# [:,:,None,None]
        self.xt = batch['hazy']
        self.t = batch['t']
        self.gt_t = batch['gt_t']
        

    def pred_batch_outputs(self, **kwargs: Any,) -> Tuple[Tensor, Tensor]:
        """ Get the predicted and target values for computing the loss. """
        # get prediction with score model
        if self.cfg.flow.refine_t:
            predicted, predicted_t = self.model_forward_wrapper(
                self.model,
                self.xt,
                self.t,
                **kwargs,
            )
            target = self.data - self.noise
            self.t = predicted_t
            return predicted, target, predicted_t, self.gt_t

        else:
            predicted = self.model_forward_wrapper(
                self.model,
                self.xt,
                self.t,
                **kwargs,
            )
            target = self.data - self.noise
    
            return predicted, target

    def pseudo_predict(self, **kwargs: Any,):
        """ Get the predicted and target values for computing the loss. """
        with torch.no_grad():
            # get prediction with score model
            if self.cfg.flow.refine_t:
                pseudo_predicted, refined_t = self.model_forward_wrapper(
                    self.model_teacher,
                    self.xt,
                    self.t,
                    **kwargs,
                )
                pseudo_predicted = pseudo_predicted.detach()
            else:
                pseudo_predicted = self.model_forward_wrapper(
                    self.model_teacher,
                    self.xt,
                    self.t,
                    **kwargs,
                ).detach()
        if self.cfg.flow.refine_t:
            # Pseudo clean  
            pseudo_clean = self.xt + (1-refined_t)*pseudo_predicted 
            
            # Psuedo hazy 
            pseudo_hazy = self.xt - (refined_t)*pseudo_predicted # (1.0 + random.random()*0.5)
        else:
        # Pseudo clean  
            pseudo_clean = self.xt + (1-self.t)*pseudo_predicted 
            
            # Psuedo hazy 
            pseudo_hazy = self.xt - (self.t)*pseudo_predicted # (1.0 + random.random()*0.5)
        # pseudo_hazy = torch.ones(self.xt.shape).cuda()

        # gamma correction for illuminence 
        self.xt = torch.pow(self.xt.clamp_(0,1), random.random()*1.5 + 1.5).float().detach()
        
        # batch_size = self.xt.shape[0]
        # self.t = torch.stack([torch.from_numpy(get_dcp_t(self.xt[i].permute(1, 2, 0).cpu().numpy(), A1=True))[None, :, :] for i in range(batch_size)], dim=0).float().to(self.xt.device)
        if self.cfg.flow.refine_t:
            predicted,pred_t = self.model_forward_wrapper(
                self.model,
                self.xt,
                self.t,
                **kwargs,
            )
            # self.t = pred_t
        else:
            predicted = self.model_forward_wrapper(
                self.model,
                self.xt,
                self.t,
                **kwargs,
            )
        
        # target = self.data - self.noise
        target = pseudo_clean - pseudo_hazy
        # predicted = pseudo_hazy + (1-pseudo_t) * predicted

        return predicted, target.detach()

    def teacher_predict(self, **kwargs: Any,):
        """ Get the predicted and target values for computing the loss. """
        
        with torch.no_grad():
            # get prediction with score model
            if self.cfg.flow.refine_t:
                predicted_teacher, predicted_teacher_t = self.model_forward_wrapper(
                    self.model_teacher,
                    self.data,
                    self.gt_t,
                    **kwargs,
                )
                self.gt_t = predicted_teacher_t
            else:
                predicted_teacher = self.model_forward_wrapper(
                self.model_teacher,
                self.data,
                self.gt_t,
                **kwargs,
                )
            
        return predicted_teacher
    
    def train_step(self, batch, current_training_step: int, augment_pipe=None, **kwargs: Any,):
        """Performs a training step"""
        ### get loss
        '''
        batch: Clean data.
        current_training_step: global training step
        '''
        # self.randomness_schedule(current_training_step)
        ## augment pipeline: edm --> https://github.com/NVlabs/edm/blob/main/training/augment.py

        loss_t = None
        ## reflow
        if self.cfg.flow.reflow: 
            self.get_asm_reflow_pair(batch)
            predicted, target = self.pseudo_predict(**kwargs)


        ## distill
        elif self.cfg.flow.use_teacher: 
            self.get_asm_distill_pair(batch)
            if self.cfg.flow.refine_t:
                predicted, target,_,_ = self.pred_batch_outputs(**kwargs)
            else:
                predicted, target= self.pred_batch_outputs(**kwargs)
            target = self.teacher_predict(**kwargs)

        ## pretrained
        else:  
            self.get_asm_data_pair(batch)
            if self.cfg.flow.refine_t:
                predicted, target,predicted_t,target_t = self.pred_batch_outputs(**kwargs)
                loss_t = self.loss_fn_t(self,predicted_t,target_t)
                w = 0.5
            else:
                predicted, target = self.pred_batch_outputs(**kwargs)

        loss = self.loss_fn(self, predicted, target)
        if loss_t is not None:
            loss = loss + w * loss_t
        return loss
       
        
