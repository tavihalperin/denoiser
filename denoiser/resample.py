# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import math

import torch as th
from torch.nn import functional as F
import numpy as np

def sinc(t):
    """sinc.

    :param t: the input tensor
    """
    # torch_sinc=th.where(t == 0, th.tensor(1., device=t.device, dtype=t.dtype), th.sin(t) / t)
    np_sinc=np.where(t == 0, 1., np.sin(t) / t)
    # assert np.all(np_sinc == torch_sinc.numpy())
    return th.from_numpy(np_sinc)
        # th.where(t == 0, th.tensor(1., device=t.device, dtype=t.dtype), th.sin(t) / t)


def kernel_upsample2(zeros=56):
    """kernel_upsample2.

    """
    win = th.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def upsample2(x, zeros=56):
    """
    Upsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    # *other, time = x.shape
    # time = x.detach().numpy().shape[-1]
    kernel = kernel_upsample2(zeros).to(x).detach().numpy().shape[-1]
    kernel = th.from_numpy(np.array([1]*(1+kernel), dtype=np.float32).reshape(1,1,kernel+1))
    # print(kernel.shape)
    out = F.conv1d(x, kernel, padding=zeros)
    # out = out[..., 1:].view(*other, time)
    y = th.cat([x, out], dim=-1)
    return y#.view(*other, -1)

win = th.hann_window(4 * 56 + 1, periodic=False).numpy()
winodd = th.from_numpy(win[1::2])


def kernel_downsample2(zeros=56):
    """kernel_downsample2.

    """
    # win = th.hann_window(4 * zeros + 1, periodic=False)
    # winodd = win[1::2]
    # t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t = np.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros) * np.pi
    # t.mul_(math.pi)
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def downsample2(x, zeros=56):
    """
    Downsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    # if x.shape[-1] % 2 != 0:
    #     x = F.pad(x, (0, 1))
    # xeven = x[..., ::2]
    # xodd = x[..., 1::2]
    xeven = F.avg_pool1d(x, 2)
    out = xeven
    # xodd = F.avg_pool1d(x, 2)
    # *other, time = xodd.shape
    # kernel = kernel_downsample2(zeros).to(x).shape[-1]

    # kernel = th.from_numpy(np.array([1]*(1+kernel), dtype=np.float32).reshape(1,1,kernel+1))
    # out = xeven# + F.conv1d(xodd, kernel, padding=zeros)
    # out = out[..., :-1]
    # print(out.shape)

    # out = out.view(*other, time)
    # print(out.shape)
    return out * .5 #out.view(*other, -1).mul(0.5)
