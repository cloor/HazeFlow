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

# Lint as: python3
"""Training Rectified Flow on CIFAR-10 with DDPM++."""

import ml_collections
from configs.default_cifar10_configs import get_default_configs


def get_config():
    config = get_default_configs()
    config.work_dir = ''
    config.expr = ''

    # training
    training = config.training
    training.snapshot_freq = 10000
    training.reduce_mean = True
    training.loss_type = 'lpips+l2' # NOTE: l1, l2, lpips, lpips+l1, lpips+l2, lpips+charbonnier
    training.x0_randomness = 'fix_0' # NOTE: whether to use random x0: unpaired data
    training.resume_from = '' # NOTE: path to the checkpoint
    training.accumulation_steps = 1 # NOTE: gradient accumulation
    training.record_iters = 500 # NOTE: record training loss every record_iters

    training.snapshot_sampling = 1000
    training.weight_schedule = 'uniform' # NOTE: snr, snr_inv, snr+1, karras, truncated-snr, uniform
    training.progress = False

    # flow
    config.flow = flow = ml_collections.ConfigDict()
    flow.eps = 1e-3
    flow.flow_t_schedule = 'uniform' # NOTE; t0, t1, uniform, or an integer k > 1
    flow.flow_alpha_t = 'uniform' # NOTE; t0, t1, uniform, or an integer k > 1
    flow.pre_train_model = '' # NOTE: path to the pre-trained model
    flow.h_flip = False
    flow.use_teacher = ''
    flow.reflow = False
    flow.refine_t = True
    # consistency model
    flow.consistency = '' # NOTE: (x1, v) \cross (cd, ct) or empty
    flow.ema_schedule = 'fixed'
    flow.step_schedule = 'fixed'
    flow.initial_timesteps = 1

    # sampling
    sampling = config.sampling
    sampling.ckpt = ''
    sampling.method = 'rectified_flow'
    sampling.init_type = 'gaussian' 
    sampling.init_noise_scale = 1.0
    sampling.use_ode_sampler = 'dcp_euler'
    sampling.ode_tol = 1e-5
    sampling.batch_size = 1
    sampling.sample_N = 20
    
    # data generation
    sampling.direction = 'from_z0'
    sampling.total_number_of_samples = 50000
    sampling.train_subset = 'all' # NOTE: all, random_100
    sampling.class_label = -1
    # heun
    sampling.rho = 1.

    # data
    data = config.data
    data.centered = True # [-1, 1]
    data.image_size = 256
    data.use_aug = False
    data.num_classes = 0 # NOTE: number of classes, 0 for unconditional
    data.reflow_data_root = '' # NOTE: whether to use reflow: use paired data from (k-1) flow
    data.custom_data_root = '' # NOTE: path to the custom dataset
    data.test_data_root = ''
    
    # model
    model = config.model
    model.name='hazeflow'
    model.scale_by_sigma = False
    model.ema_rate = 0.999999
    model.dropout = 0.15
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
    model.num_res_blocks = 2
    # model.ch_mult = (1, 2, 2, 2)
    # model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = False
    model.img_conditional = True
    model.fir = False
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'none'
    model.progressive_input = 'none'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.
    # model.embedding_type = 'positional'
    model.embedding_type='img'
    model.fourier_scale = 16
    model.conv_size = 3

    return config
