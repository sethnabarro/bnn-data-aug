"""
Adapted from https://github.com/ratschlab/bnn_priors/blob/main/bnn_priors/prior/loc_scale.py
"""
import torch.distributions as td

from .base import Prior


__all__ = ('LocScale', 'Normal', 'Improper', 'get_prior')


class LocScale(Prior):
    """Prior with a `loc` and `scale` parameters, implemented as a reparameterised
    version of loc=0 and scale=1.

    Arguments:
       shape (torch.Size): shape of the parameter
       loc (float, torch.Tensor, prior.Prior): location
       scale (float, torch.Tensor, prior.Prior): scale
    """
    def __init__(self, shape, loc, scale):
        super().__init__(shape, loc=loc, scale=scale)


class Normal(LocScale):
    _dist = td.Normal


class Improper(Normal):
    "Improper prior that samples like a Normal"
    def log_prob(self):
        return 0.0

def get_prior(prior_name):
    priors = {"gaussian": Normal,
              "datadrivencorrnormal": Normal,
              "improper": Improper}
    assert prior_name in priors
    return priors[prior_name]
