"""
Adapted from https://github.com/ratschlab/bnn_priors/blob/main/bnn_priors/models/base.py
"""
from .. import prior
from torch import nn
import torch
import abc
from typing import Dict, Union, Tuple
from collections import OrderedDict

__all__ = ('ClassificationModel', 'AbstractModel')


class AbstractModel(nn.Module, abc.ABC):
    """A model that can be used with our SGLD and Pyro MCMC samplers.

    Arguments:
       num_data: the total number of data points, for minibatching
       net: neural net to evaluate to get the latent function
    """
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def log_prior(self):
        "log p(params)"
        lp = sum(p.log_prob() for _, p in prior.named_priors(self))
        if isinstance(lp, float):
            return torch.tensor(lp)
        return lp

    @abc.abstractmethod
    def likelihood_dist(self, f: torch.Tensor):
        "representation of p(y | f, params)"
        pass

    def forward(self, x: torch.Tensor):
        "representation of p(y | x, params)"
        if self.data_aug_prior:
            # Collapse batch and augmentation dims of input
            # for forward pass
            x_shp = list(x.shape)   # batch_size x n_aug_samples x n_channel x H x W
            x_rs = x.view([x_shp[0] * x_shp[1]] + x_shp[2:])
            f = self.net(x_rs)

            # Separate batch and augmentation dims of output
            f = f.view([x_shp[0], x_shp[1]] + list(f.shape[1:]))
        else:
            f = self.net(x)
        return self.likelihood_dist(f)

    def log_likelihood(self, x: torch.Tensor, y: torch.Tensor, eff_num_data):
        """
        unbiased minibatch estimate of log-likelihood
        log p(y | x, self.parameters)
        """
        ll, _ = self._log_likelihood_preds(x, y, eff_num_data)
        return ll

    def log_likelihood_avg(self, x: torch.Tensor, y: torch.Tensor):
        """
        unbiased minibatch estimate of log-likelihood per datapoint
        """
        lla, _ = self._log_likelihood_preds(x, y, eff_num_data=1)
        return lla

    def _log_likelihood_preds(self, x: torch.Tensor, y: torch.Tensor, eff_num_data: int):
        # compute batch size using zeroth dim of inputs (log_prob_batch doesn't work with RaoB).
        assert x.shape[0] == y.shape[0]
        batch_size = x.shape[0]
        preds = self(x)
        return preds.log_prob(y).sum() * (eff_num_data / batch_size), preds

    def potential(self, x, y, eff_num_data):
        """
        There are two subtly different ways of altering the "temperature".
        The Wenzel et al. approach is to apply a temperature (here, T) to both the prior and likelihood together.
        However, the VI approach is to in effect replicate each datapoint multiple times (data_mult)
        """
        return - (self.log_likelihood(x, y, eff_num_data) + self.log_prior())

    def _split_potential_preds(self, x, y, eff_num_data):
        lla, preds = self._log_likelihood_preds(x, y, eff_num_data=1)
        loss = -lla
        log_prior = self.log_prior()
        potential_avg = loss - log_prior/eff_num_data
        return loss, log_prior, potential_avg, preds

    @abc.abstractmethod
    def split_potential_and_acc(self, x, y, eff_num_data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.distributions.Distribution]:
        pass

    def potential_avg(self, x, y, eff_num_data):
        "-log p(y, params | x)"
        return -(self.log_likelihood_avg(x, y) + self.log_prior()/eff_num_data)

    def params_dict(self):
        return OrderedDict((n, p.detach()) for (n, p) in self.named_parameters())

    def sample_all_priors(self):
        for _, v in prior.named_priors(self):
            v.sample()


class ClassificationModel(AbstractModel):
    """Model for classification using a Categorical likelihood.
    Arguments:
       num_data: the total number of data points, for minibatching
       softmax_temp (float_like or Prior): the temperature of the softmax output
                 likelihood
       net: modules to evaluate to get the latent function
    """
    def __init__(self, net: nn.Module,
                 softmax_temp: Union[float, torch.Tensor, prior.Prior]=1.,
                 data_aug_prior: [bool, str] = False):
        super().__init__(net)
        self.softmax_temp = softmax_temp
        assert data_aug_prior in (None, False, True, 'logits', 'probs'), \
            f"Arg `data_aug_prior` value not valid: {data_aug_prior}"
        self.data_aug_prior = "logits" if data_aug_prior is True else data_aug_prior

    def likelihood_dist(self, f: torch.Tensor):
        sm_temp = prior.value_or_call(self.softmax_temp)
        if self.data_aug_prior:
            if self.data_aug_prior == 'probs':
                # Average predicted class probs over data augmentation samples
                p_y_cond_x_xa = torch.softmax(f/sm_temp, dim=-1)
                p_y_cond_x = p_y_cond_x_xa.mean(dim=1)
                return torch.distributions.Categorical(probs=p_y_cond_x)
            elif self.data_aug_prior == 'logits':
                # Average in logit space
                f = f.mean(dim=1)
        return torch.distributions.Categorical(logits=f/sm_temp)

    def acc_mse(self, preds, y):
        return torch.argmax(preds.logits, dim=1).eq(y).to(torch.float32)

    def split_potential_and_acc(self, x, y, eff_num_data):
        loss, log_prior, potential_avg, preds = (
            self._split_potential_preds(x, y, eff_num_data))
        acc = torch.argmax(preds.logits, dim=1).eq(y).to(torch.float32)
        return loss, log_prior, potential_avg, acc, preds
