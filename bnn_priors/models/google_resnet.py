# Parts of this file are taken from https://github.com/google-research/google-research/blob/master/cold_posterior_bnn/models.py
# licensed under the Apache License, Version 2.0
"""
Adapted from https://github.com/ratschlab/bnn_priors/blob/main/bnn_priors/models/google_resnet.py
"""
import torch.nn as nn

from .conv_nets import Conv2dPrior
from .dense_nets import LinearPrior
from .base import ClassificationModel
from .. import prior


class BasicBlock(nn.Module):
    def __init__(self, in_filters, filters, stride, conv_kwargs, norm, activation):
        super().__init__()
        self.main = nn.Sequential(
            Conv2dPrior(in_filters, filters, kernel_size=3, padding=1, stride=stride, **conv_kwargs),
            norm(filters),
            activation(),
            Conv2dPrior(filters, filters, kernel_size=3, padding=1, stride=1, **conv_kwargs),
            norm(filters))

        if stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                Conv2dPrior(in_filters, filters, kernel_size=1, padding=0, stride=stride, **conv_kwargs),
                norm(filters),
            )
        self.output_activation = activation()

    def forward(self, x):
        y = self.main(x)
        z = self.shortcut(x)
        return self.output_activation(y + z)


def ResNet(softmax_temp=1., depth=20, num_classes=10,
           prior_w=prior.Normal, loc_w=0., std_w=2**.5, std_w_logits=None,
           prior_b=prior.Normal, loc_b=0., std_b=1.,
           scaling_fn=None, bn=True, weight_prior_params={}, bias_prior_params={},
           conv_prior_w=prior.Normal,
           data_aug_prior=False,
           rec_field_scaling=True):
    activation_fn = nn.ReLU
    conv_kwargs = dict(
        prior_w=conv_prior_w, loc_w=loc_w, std_w=std_w,
        prior_b=None, scaling_fn=scaling_fn,
        weight_prior_params=weight_prior_params,
        bias_prior_params=bias_prior_params,
        rec_field_scaling=rec_field_scaling)
    norm = (nn.BatchNorm2d if bn else nn.Identity)

    # Main network code
    num_res_blocks = (depth - 2) // 6
    filters = 16
    if (depth - 2) % 6 != 0:
        raise ValueError('depth must be 6n+2 (e.g. 20, 32, 44).')

    layers = [
        Conv2dPrior(3, filters, kernel_size=3, padding=1, stride=1, **conv_kwargs),
        norm(filters),
        activation_fn()]

    for stack in range(3):
        stride = (1 if stack == 0 else 2)
        prev_filters = filters
        filters *= stride

        layers.append(BasicBlock(
            prev_filters, filters, stride, conv_kwargs, norm, activation_fn))

        for _ in range(num_res_blocks-1):
            layers.append(BasicBlock(
                filters, filters, 1, conv_kwargs, norm, activation_fn))

    layers += [
        nn.AvgPool2d(8),
        nn.Flatten(),
        LinearPrior(filters, num_classes,
                    prior_w=prior_w, loc_w=loc_w, std_w=std_w_logits or std_w,
                    prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                    scaling_fn=scaling_fn,
                    weight_prior_params=weight_prior_params,
                    bias_prior_params=bias_prior_params)]
    return ClassificationModel(nn.Sequential(*layers),
                               softmax_temp=softmax_temp,
                               data_aug_prior=data_aug_prior)
