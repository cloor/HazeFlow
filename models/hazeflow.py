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
# import sys
# sys.path.append("../../")
# sys.path.append("../")
# sys.path.append(".")
# from absl import app
# from absl import flags
# from ml_collections.config_flags import config_flags
# import utils, layers, layerspp_hazy, normalization
from . import utils, layers, layerspp_hazy, normalization

import torch.nn as nn
import functools
import torch
import numpy as np

from .utils import register_model

ResnetBlockDDPM = layerspp_hazy.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp_hazy.ResnetBlockBigGANpp
Combine = layerspp_hazy.Combine
conv3x3 = layerspp_hazy.conv3x3
conv1x1 = layerspp_hazy.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@register_model(name='hazeflow')
class HazeFlow(nn.Module):
  """HazeFlow model"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.act = act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

    self.nf = nf = config.model.nf
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

    self.conditional = conditional = config.model.conditional  # noise-conditional
    self.img_conditional = config.model.img_conditional
    fir = config.model.fir
    fir_kernel = config.model.fir_kernel
    self.skip_rescale = skip_rescale = config.model.skip_rescale
    self.resblock_type = resblock_type = config.model.resblock_type.lower()
    self.progressive = progressive = config.model.progressive.lower()
    self.progressive_input = progressive_input = config.model.progressive_input.lower()
    self.embedding_type = embedding_type = config.model.embedding_type.lower()
    init_scale = config.model.init_scale
    assert progressive in ['none', 'output_skip', 'residual']
    assert progressive_input in ['none', 'input_skip', 'residual']
    assert embedding_type in ['fourier', 'positional', 'img']
    combine_method = config.model.progressive_combine.lower()
    combiner = functools.partial(Combine, method=combine_method)

    modules = []
    # timestep/noise_level embedding; only for continuous training
    if embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      assert config.training.continuous or config.training.sde=='rectified_flow', "Fourier features are only used for continuous training."

      modules.append(layerspp_hazy.GaussianFourierProjection(
        embedding_size=nf, scale=config.model.fourier_scale
      ))
      embed_dim = 2 * nf

    elif embedding_type == 'positional':
      embed_dim = nf

    elif embedding_type == 'img': 
      embed_dim =nf
    else:
      raise ValueError(f'embedding type {embedding_type} unknown.')

    if conditional:
      modules.append(nn.Linear(embed_dim, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)

    if self.img_conditional: 
      modules.append(nn.Conv2d(1, embed_dim, 1, 1))
      modules.append(nn.Conv2d(embed_dim, embed_dim, 1, 1))

    AttnBlock = functools.partial(layerspp_hazy.AttnBlockpp,
                                  init_scale=init_scale,
                                  skip_rescale=skip_rescale)

    Upsample = functools.partial(layerspp_hazy.Upsample,
                                 with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if progressive == 'output_skip':
      self.pyramid_upsample = layerspp_hazy.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive == 'residual':
      pyramid_upsample = functools.partial(layerspp_hazy.Upsample,
                                           fir=fir, fir_kernel=fir_kernel, with_conv=True)

    Downsample = functools.partial(layerspp_hazy.Downsample,
                                   with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if progressive_input == 'input_skip':
      self.pyramid_downsample = layerspp_hazy.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive_input == 'residual':
      pyramid_downsample = functools.partial(layerspp_hazy.Downsample,
                                             fir=fir, fir_kernel=fir_kernel, with_conv=True)

    if resblock_type == 'ddpm':
      ResnetBlock = functools.partial(ResnetBlockDDPM,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf)

    elif resblock_type == 'biggan':
      ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                      act=act,
                                      dropout=dropout,
                                      fir=fir,
                                      fir_kernel=fir_kernel,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf )

    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')

    # Downsampling block
    
    channels = config.data.num_channels
    if config.flow.refine_t:
        channels = 4
    if progressive_input != 'none':
      input_pyramid_ch = channels

    modules.append(conv3x3(channels, nf))
    hs_c = [nf]

    in_ch = nf
    attn = 0
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch

        # if all_resolutions[i_level] in attn_resolutions:
        if attn == 4:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)

      if i_level != num_resolutions - 1:
        if resblock_type == 'ddpm':
          modules.append(Downsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlock(down=True, in_ch=in_ch))
          attn += 1
        if progressive_input == 'input_skip':
          modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
          if combine_method == 'cat':
            in_ch *= 2

        elif progressive_input == 'residual':
          modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
          input_pyramid_ch = in_ch

        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    pyramid_ch = 0
    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        save=hs_c.pop()
        modules.append(ResnetBlock(in_ch=in_ch ,
                                   out_ch=out_ch, save=save))
        in_ch = out_ch

      # if all_resolutions[i_level] in attn_resolutions:
      if attn ==4:
        modules.append(AttnBlock(channels=in_ch))

      if progressive != 'none':
        if i_level == num_resolutions - 1:
          if progressive == 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
            pyramid_ch = channels
          elif progressive == 'residual':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, in_ch, bias=True))
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name.')
        else:
          if progressive == 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
            pyramid_ch = channels
          elif progressive == 'residual':
            modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name')

      if i_level != 0:
        if resblock_type == 'ddpm':
          modules.append(Upsample(in_ch=in_ch))
        else:
          if in_ch ==nf*2 and save==nf:
            modules.append(ResnetBlock(in_ch=nf, up=True,save=nf))
            attn -=1
          else:
            modules.append(ResnetBlock(in_ch=in_ch, up=True))
            attn -=1
    assert not hs_c

    if progressive != 'output_skip':
      modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                  num_channels=in_ch, eps=1e-6))
      modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

    self.all_modules = nn.ModuleList(modules)


  def forward(self, x, time_cond,use_skip=True, use_refine=False):
    
    if use_skip:
        skip = x.clone().detach()
    if use_refine:
        x = torch.cat([x,time_cond],dim=1)
    # timestep/noise_level embedding; only for continuous training
    modules = self.all_modules
    m_idx = 0
    if self.embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      used_sigmas = time_cond
      temb = modules[m_idx](torch.log(used_sigmas))
      m_idx += 1

    elif self.embedding_type == 'positional':
      # Sinusoidal positional embeddings.
      timesteps = time_cond
      used_sigmas = self.sigmas[time_cond.long()]
      temb = layers.get_timestep_embedding(timesteps, self.nf)

    elif self.embedding_type == 'img':
      transmission = time_cond
      temb = transmission

    else:
      raise ValueError(f'embedding type {self.embedding_type} unknown.')

    if self.conditional:
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    elif self.img_conditional:
      temb = modules[m_idx](time_cond)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None

    if not self.config.data.centered:
      # If input data is in [0, 1]
      x = 2 * x - 1.

    # Downsampling block
    input_pyramid = None
    if self.progressive_input != 'none':
      input_pyramid = x
    
    hs = [modules[m_idx](x)]
    ts = [temb]
    m_idx += 1
    attn = 0
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h, temb= modules[m_idx](hs[-1], temb)
        m_idx += 1
        # if h.shape[-1] in self.attn_resolutions:
        if attn == 4:
          h = modules[m_idx](h)
          m_idx += 1

        hs.append(h)
        ts.append(temb)
      if i_level != self.num_resolutions - 1:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](hs[-1])
          m_idx += 1
        else:
          h ,temb= modules[m_idx](hs[-1], temb) ### D
          ts.append(temb)
          m_idx += 1 
          attn += 1

        if self.progressive_input == 'input_skip':
          input_pyramid = self.pyramid_downsample(input_pyramid)
          h = modules[m_idx](input_pyramid, h)
          m_idx += 1

        elif self.progressive_input == 'residual':
          input_pyramid = modules[m_idx](input_pyramid)
          m_idx += 1
          if self.skip_rescale:
            input_pyramid = (input_pyramid + h) / np.sqrt(2.)
          else:
            input_pyramid = input_pyramid + h
          h = input_pyramid

        hs.append(h)
    
    
    h = hs[-1]
    h , temb= modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h , temb= modules[m_idx](h, temb)
    m_idx += 1
    # ts.pop()
    pyramid = None

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        
        h , temb= modules[m_idx](torch.cat([h, hs.pop()], dim=1), ts.pop())
        m_idx += 1

      # if h.shape[-1] in self.attn_resolutions:
      if attn ==4:
        h = modules[m_idx](h)
        m_idx += 1

      if self.progressive != 'none':
        if i_level == self.num_resolutions - 1:
          if self.progressive == 'output_skip':
            pyramid = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
          elif self.progressive == 'residual':
            pyramid = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
          else:
            raise ValueError(f'{self.progressive} is not a valid name.')
        else:
          if self.progressive == 'output_skip':
            pyramid = self.pyramid_upsample(pyramid)
            pyramid_h = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid_h = modules[m_idx](pyramid_h)
            m_idx += 1
            pyramid = pyramid + pyramid_h
          elif self.progressive == 'residual':
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
            if self.skip_rescale:
              pyramid = (pyramid + h) / np.sqrt(2.)
            else:
              pyramid = pyramid + h
            h = pyramid
          else:
            raise ValueError(f'{self.progressive} is not a valid name')

      if i_level != 0:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](h)
          m_idx += 1
        else:
          h , temb= modules[m_idx](h, ts[-1])
          m_idx += 1
          attn -= 1
    assert not hs

    if self.progressive == 'output_skip':
      h = pyramid
    else:
      h = self.act(modules[m_idx](h))
      m_idx += 1
      h = modules[m_idx](h)
      m_idx += 1

    assert m_idx == len(modules)
    if self.config.model.scale_by_sigma:
      used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
      h = h / used_sigmas
    if use_skip:
        if use_refine:
            return h[:,:3,:,:] + skip, h[:,3:,:,:]
        else:
            return h[:,:3,:,:] + skip
    else:
        if use_refine:
            return h[:,:3,:,:], h[:,3:,:,:]
        else: 
            return h[:,:3,:,:]
