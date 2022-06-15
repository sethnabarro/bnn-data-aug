"""
Adapted from https://github.com/ratschlab/bnn_priors/blob/main/bnn_priors/models/dense_nets.py
"""
from torch import nn, Tensor

from .layers import Linear
from .base import ClassificationModel

from .. import prior

__all__ = ('LinearPrior', 'ClassificationDenseNet', 'LogisticRegression')


def LinearPrior(in_dim, out_dim, prior_w=prior.Normal, loc_w=0., std_w=1.,
                     prior_b=prior.Normal, loc_b=0., std_b=1., scaling_fn=None,
               weight_prior_params={}, bias_prior_params={}):
    if scaling_fn is None:
        def scaling_fn(std, dim):
            return std/dim**0.5
    return Linear(prior_w((out_dim, in_dim), loc_w, scaling_fn(std_w, in_dim), **weight_prior_params),
                  prior_b((out_dim,), 0., std_b), **bias_prior_params)


def ClassificationDenseNet(in_features, out_features, width, depth=3, softmax_temp=1.,
                           prior_w=prior.Normal, loc_w=0., std_w=2**.5, std_w_logits=None,
                           prior_b=prior.Normal, loc_b=0., std_b=1., bn=False,
                           scaling_fn=None, weight_prior_params={}, bias_prior_params={},
                           data_aug_prior=False):
    maybe_norm = (nn.BatchNorm1d if bn else nn.Identity)
    layers = [nn.Flatten(),
              LinearPrior(in_features, width, prior_w=prior_w, loc_w=loc_w,
                       std_w=std_w, prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                       scaling_fn=scaling_fn, weight_prior_params=weight_prior_params,
                        bias_prior_params=bias_prior_params),
              maybe_norm(width),
              nn.ReLU()]
    for _ in range(depth-2):
        layers.append(LinearPrior(width, width, prior_w=prior_w, loc_w=loc_w,
                       std_w=std_w, prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                       scaling_fn=scaling_fn, weight_prior_params=weight_prior_params,
                        bias_prior_params=bias_prior_params))
        layers.append(maybe_norm(width))
        layers.append(nn.ReLU())
    layers.append(LinearPrior(width, out_features, prior_w=prior_w, loc_w=loc_w,
                       std_w=std_w_logits or std_w, prior_b=prior_b, loc_b=loc_b, std_b=std_b,
                       scaling_fn=scaling_fn, weight_prior_params=weight_prior_params,
                        bias_prior_params=bias_prior_params))
    return ClassificationModel(nn.Sequential(*layers), softmax_temp, data_aug_prior=data_aug_prior)


def LogisticRegression(in_features, out_features, softmax_temp=1.,
                       prior_w=prior.Normal, loc_w=0., std_w=2**.5,
                       prior_b=prior.Normal, loc_b=0., std_b=1.,
                       scaling_fn=None, weight_prior_params={}, bias_prior_params={},
                       data_aug_prior=False):
    return ClassificationModel(LinearPrior(
        in_features, out_features,
        prior_w=prior_w, loc_w=loc_w, std_w=std_w,
        prior_b=prior_b, loc_b=loc_b, std_b=std_b,
        scaling_fn=scaling_fn, weight_prior_params=weight_prior_params,
        bias_prior_params=bias_prior_params), softmax_temp=softmax_temp,
        data_aug_prior=data_aug_prior)
