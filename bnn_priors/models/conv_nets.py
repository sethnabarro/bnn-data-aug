"""
Adapted from https://github.com/ratschlab/bnn_priors/blob/main/bnn_priors/models/conv_nets.py
"""
import torch.nn as nn

from .layers import Conv2d
from .base import ClassificationModel
from .dense_nets import LinearPrior
from .. import prior

__all__ = ('Conv2dPrior', 'ClassificationConvNet')


def Conv2dPrior(in_channels, out_channels, kernel_size=3, stride=1,
            padding=0, dilation=1, groups=1, padding_mode='zeros',
            prior_w=prior.Normal, loc_w=0., std_w=1., prior_b=prior.Normal,
            loc_b=0., std_b=1., scaling_fn=None, weight_prior_params={}, bias_prior_params={},
                which_conv=Conv2d, rec_field_scaling=True):
    if scaling_fn is None:
        def scaling_fn(std, dim):
            return std/dim**0.5
    kernel_size = nn.modules.utils._pair(kernel_size)
    bias_prior = prior_b((out_channels,), 0., std_b, **bias_prior_params) if prior_b is not None else None
    maybe_rec_field_size = kernel_size[0] * kernel_size[1] if rec_field_scaling else 1.
    return which_conv(weight_prior=prior_w((out_channels, in_channels//groups, kernel_size[0], kernel_size[1]),
                                       loc_w, scaling_fn(std_w, in_channels * maybe_rec_field_size), **weight_prior_params),
                  bias_prior=bias_prior,
                 stride=stride, padding=padding, dilation=dilation,
                  groups=groups, padding_mode=padding_mode)


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


def ClassificationConvNet(in_channels, img_height, out_features, width, depth=3, softmax_temp=1.,
                          prior_w=prior.Normal, loc_w=0., std_w=2**.5, std_w_logits=None,
                          prior_b=prior.Normal, loc_b=0., std_b=1.,
                          scaling_fn=None, weight_prior_params={}, bias_prior_params={},
                          data_aug_prior=False, rec_field_scaling=True):
    assert depth >= 2, "We can't have less than two layers"
    layers = [Reshape(-1, in_channels, img_height, img_height),
              Conv2dPrior(in_channels, width, kernel_size=3, padding=1, prior_w=prior_w, loc_w=loc_w,
                       std_w=std_w, prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                       scaling_fn=scaling_fn, weight_prior_params=weight_prior_params,
                        bias_prior_params=bias_prior_params, rec_field_scaling=rec_field_scaling),
            nn.ReLU(), nn.MaxPool2d(2)]
    for _ in range(depth-2):
        layers.append(Conv2dPrior(width, width, kernel_size=3, padding=1, prior_w=prior_w, loc_w=loc_w,
                       std_w=std_w, prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                       scaling_fn=scaling_fn, weight_prior_params=weight_prior_params,
                        bias_prior_params=bias_prior_params, rec_field_scaling=rec_field_scaling))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
    layers.append(nn.Flatten())
    reshaped_size = width*(img_height//2**(depth-1))**2
    layers.append(LinearPrior(reshaped_size, out_features, prior_w=prior_w, loc_w=loc_w,
                       std_w=std_w_logits or std_w, prior_b=prior_b, loc_b=loc_b,
                       scaling_fn=scaling_fn, weight_prior_params=weight_prior_params,
                        bias_prior_params=bias_prior_params))
    return ClassificationModel(nn.Sequential(*layers), softmax_temp, data_aug_prior=data_aug_prior)
