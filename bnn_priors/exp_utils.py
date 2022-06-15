"""
Adapted from https://github.com/ratschlab/bnn_priors/blob/main/bnn_priors/exp_utils.py
"""
import math
import h5py
import numpy as np
import time
from sklearn.metrics import average_precision_score, roc_auc_score
import torch as t
import bnn_priors.data
from bnn_priors.models import ClassificationDenseNet, ResNet, ClassificationConvNet
import bnn_priors.models
from bnn_priors.prior import get_prior
from bnn_priors.third_party.calibration_error import ece, ace, rmsce
import warnings
import sacred
from pathlib import Path

from typing import Dict, Iterable, Tuple


def device(device: str):
    if device == "try_cuda":
        if t.cuda.is_available():
            return t.device("cuda:0")
        else:
            return t.device("cpu")
    return t.device(device)


def get_data(data: str, device: t.device, data_kwargs: [dict, None] = None):
    data_kwargs = data_kwargs or {}
    if data == "cifar10":
        dataset = bnn_priors.data.CIFAR10(device=device, **data_kwargs)
    elif data == "cifar10_augmented":
        dataset = bnn_priors.data.CIFAR10Augmented(device=device,
                                                   download=True,
                                                   **data_kwargs)
    elif data.split('-')[0] == "cifar10_data_aug_prior":
        n_aug_samples, n_aug_samples_test, do_test_aug = \
            parse_data_aug_name(data)
        dataset = bnn_priors.data.CIFAR10DataAugmentPrior(n_aug_samples,
                                                          device=device,
                                                          n_mc_aug_test=n_aug_samples_test,
                                                          test_time_aug=do_test_aug,
                                                          **data_kwargs)
    elif data.split('-')[0] == "cifar10_fix_data_aug_prior":
        n_aug_samples, n_aug_subsamp, n_aug_samples_test, n_aug_subsamp_test, do_test_aug, seed = \
            parse_fix_data_aug_name(data)
        dataset = bnn_priors.data.CIFAR10DataAugmentPrior(n_aug_samples,
                                                          n_subsample=n_aug_subsamp,
                                                          device=device,
                                                          seed=seed,
                                                          n_mc_aug_test=n_aug_samples,
                                                          n_subsample_test=n_aug_subsamp_test,
                                                          test_time_aug=do_test_aug,
                                                          **data_kwargs)
    elif data == "mnist":
        dataset = bnn_priors.data.MNIST(device=device, download=True)
    elif data == "rotated_mnist":
        dataset = bnn_priors.data.RotatedMNIST(device=device)
    elif data.split('-')[0] == "mnist_data_aug_prior":
        n_aug_samples, n_aug_samples_test, do_test_aug = \
            parse_data_aug_name(data)
        dataset = bnn_priors.data.MNISTDataAugmentPrior(n_aug_samples,
                                                        device=device,
                                                        n_mc_aug_test=n_aug_samples_test,
                                                        test_time_aug=do_test_aug,
                                                        **data_kwargs)
    elif data.split('-')[0] == "mnist_fix_data_aug_prior":
        n_aug_samples, n_aug_subsamp, n_aug_samples_test, n_aug_subsamp_test, do_test_aug, seed = \
            parse_fix_data_aug_name(data)
        dataset = bnn_priors.data.MNISTDataAugmentPrior(n_aug_samples,
                                                        n_subsample=n_aug_subsamp,
                                                        device=device,
                                                        seed=seed,
                                                        n_mc_aug_test=n_aug_samples,
                                                        n_subsample_test=n_aug_subsamp_test,
                                                        test_time_aug=do_test_aug,
                                                        **data_kwargs)
    elif data.split('-')[0] == "fashion_mnist_data_aug_prior":
        n_aug_samples, n_aug_samples_test, do_test_aug = \
            parse_data_aug_name(data)
        dataset = bnn_priors.data.FashionMNISTDataAugmentPrior(n_aug_samples,
                                                               device=device,
                                                               n_mc_aug_test=n_aug_samples_test,
                                                               test_time_aug=do_test_aug,
                                                               **data_kwargs)
    elif data.split('-')[0] == "fashion_mnist_fix_data_aug_prior":
        n_aug_samples, n_aug_subsamp, n_aug_samples_test, n_aug_subsamp_test, do_test_aug, seed = \
            parse_fix_data_aug_name(data)
        dataset = bnn_priors.data.FashionMNISTDataAugmentPrior(n_aug_samples,
                                                               n_subsample=n_aug_subsamp,
                                                               device=device,
                                                               seed=seed,
                                                               n_mc_aug_test=n_aug_samples,
                                                               n_subsample_test=n_aug_subsamp_test,
                                                               test_time_aug=do_test_aug,
                                                               **data_kwargs)
    elif data == "fashion_mnist":
        dataset = bnn_priors.data.FashionMNIST(device=device, **data_kwargs)
    else:
        raise ValueError(f"Unknown data='{data}'")
    return dataset


def parse_fix_data_aug_name(dataname):
    """
    Process dataset name string to get number of augmentation samples in finite orbit case,
    whether to augment at test time, if so number of test augmentation samples
    and augmentation seed.
    """
    assert len(dataname.split('-')) == 3 or \
              (len(dataname.split('-')) == 4 and 'test' in dataname.split('-')[3]), \
        f"For data aug prior with fixed augmentations dataset name should be " \
        f"'<dataset>_fix_data_aug_prior-<n_aug_samples>-<seed>' or " \
        f"'<dataset>_fix_data_aug_prior-<n_aug_samples>-<seed>-test<n_aug_samples_test>', " \
        f"but got '{dataname}'. E.g. 'cifar10_fix_data_aug_prior-8' or 'mnist_fix_data_aug_prior-4-test4'"
    n_aug_samples = dataname.split('-')[1]
    if "." in n_aug_samples:
        n_aug_samples, n_aug_subsamp = [int(n) for n in n_aug_samples.split(".")]
    else:
        n_aug_samples = int(n_aug_samples)
        n_aug_subsamp = None
    seed = int(dataname.split('-')[2])
    do_test_aug = (len(dataname.split('-')) == 4) and ('test' in dataname.split('-')[3])
    if do_test_aug:
        n_aug_samples_test = dataname.split('-test')[1]
        if "." in n_aug_samples_test:
            n_aug_samples_test, n_aug_subsamp_test = [int(n) for n in n_aug_samples_test.split(".")]
        else:
            n_aug_samples_test = int(n_aug_samples_test)
            n_aug_subsamp_test = None
    else:
        n_aug_samples_test = None
        n_aug_subsamp_test = None
    return n_aug_samples, n_aug_subsamp, n_aug_samples_test, n_aug_subsamp_test, do_test_aug, seed


def parse_data_aug_name(dataname):
    assert len(dataname.split('-')) in (2, 3), \
        f"For data aug prior without fixed augmentations dataset name should be " \
        f"'<dataset>_data_aug_prior-<n_aug_samples>' or " \
        f"'<dataset>_data_aug_prior-<n_aug_samples>-test<n_aug_samples_test>', " \
        f"but got '{dataname}'. E.g. 'cifar10_data_aug_prior-8' or 'mnist_data_aug_prior-4-test4'"
    n_aug_samples = int(dataname.split('-')[1])
    do_test_aug = (len(dataname.split('-')) == 3) and ('test' in dataname.split('-')[2])
    n_aug_samples_test = int(dataname.split('-test')[1]) if do_test_aug else None
    return n_aug_samples, n_aug_samples_test, do_test_aug


def is_for_data_aug_prior(dataset):
    if hasattr(dataset, 'for_data_augment_prior'):
        return dataset.for_data_augment_prior
    return False


def he_initialize(model):
    for name, param in model.named_parameters():
        if "weight_prior.p" in name:
            t.nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='relu')
        elif "bias_prior.p" in name:
            bound = 1 / math.sqrt(param.size(0))
            t.nn.init.uniform_(param.data, -bound, bound)


def he_zerobias_initialize(model):
    for name, param in model.named_parameters():
        if "weight_prior.p" in name:
            t.nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='relu')
        elif "bias_prior.p" in name:
            t.nn.init.zeros_(param.data)


def he_uniform_initialize(model):
    for name, param in model.named_parameters():
        if "weight_prior.p" in name:
            if "conv" in name or "shortcut" in name or param.dim() == 4:
                t.nn.init.kaiming_uniform_(param.data, a=math.sqrt(5))
            elif "linear" in name or param.dim() == 2:
                bound = 1 / math.sqrt(param.size(1))
                t.nn.init.uniform_(param.data, -bound, bound)
            else:
                raise NotImplementedError(name)
        elif "bias_prior.p" in name:
            if "conv" in name or "shortcut" in name:
                raise NotImplementedError(name)
            elif "linear" in name or param.dim() == 1:
                bound = 1 / math.sqrt(param.size(0))
                t.nn.init.uniform_(param.data, -bound, bound)
            else:
                raise NotImplementedError(name)


class DummyModule(t.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def get_model(x_train, y_train, model, width, depth, weight_prior, weight_loc,
              weight_scale, weight_scale_logits, bias_prior, bias_loc, bias_scale, batchnorm,
              weight_prior_params, bias_prior_params, data_aug_prior, rec_field_scaling,
              softmax_temp, no_prior_scaling):
    if no_prior_scaling:
        def no_scaling(std, dim):
            return std
        scaling_fn = no_scaling
    else:
        if weight_prior in ["cauchy"]:
            # NOTE: Cauchy and anything with infinite variance should use this
            scaling_fn = lambda std, dim: std/dim
        else:
            scaling_fn = lambda std, dim: std/dim**0.5
    weight_prior = get_prior(weight_prior)
    bias_prior = get_prior(bias_prior)
    if model == "classificationdensenet":
        infeat = x_train.size(-1) if len(x_train.shape) < 4 else np.prod(x_train.shape[-3:])
        net = ClassificationDenseNet(infeat, y_train.max()+1, width, depth, softmax_temp=softmax_temp,
                                     prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale,
                                     std_w_logits=weight_scale_logits, bn=batchnorm, prior_b=bias_prior,
                                     loc_b=bias_loc, std_b=bias_scale, scaling_fn=scaling_fn,
                                     weight_prior_params=weight_prior_params,
                                     bias_prior_params=bias_prior_params,
                                     data_aug_prior=data_aug_prior).to(x_train)
    elif model == "classificationconvnet":
        is_da_prior = bool(data_aug_prior)
        if len(x_train.shape) == 4 + int(is_da_prior):
            in_channels = x_train.shape[1 + int(is_da_prior)]
            img_height = x_train.shape[-2]
        else:
            in_channels = 1
            img_height = int(math.sqrt(x_train.shape[-1]))
        network_class = ClassificationConvNet
        net = network_class(in_channels, img_height, y_train.max()+1, width, depth, softmax_temp=softmax_temp,
                            prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale, std_w_logits=weight_scale_logits,
                            prior_b=bias_prior, loc_b=bias_loc, std_b=bias_scale, scaling_fn=scaling_fn,
                            weight_prior_params=weight_prior_params, bias_prior_params=bias_prior_params,
                            data_aug_prior=data_aug_prior, rec_field_scaling=rec_field_scaling).to(x_train)
    elif model == "googleresnet":
        net = ResNet(prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale, std_w_logits=weight_scale_logits,
                     depth=20, conv_prior_w=weight_prior, prior_b=bias_prior, loc_b=bias_loc, std_b=bias_scale,
                     scaling_fn=scaling_fn, bn=batchnorm, softmax_temp=softmax_temp,
                     weight_prior_params=weight_prior_params, bias_prior_params=bias_prior_params,
                     data_aug_prior=data_aug_prior, rec_field_scaling=rec_field_scaling).to(x_train)
    elif model == "googleresnet_depth":
        net = ResNet(prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale, std_w_logits=weight_scale_logits,
                     depth=depth, conv_prior_w=weight_prior, prior_b=bias_prior, loc_b=bias_loc, std_b=bias_scale,
                     scaling_fn=scaling_fn, bn=batchnorm, softmax_temp=softmax_temp,
                     weight_prior_params=weight_prior_params, bias_prior_params=bias_prior_params,
                     data_aug_prior=data_aug_prior, rec_field_scaling=rec_field_scaling).to(x_train)
    elif model == "logistic":
        net = bnn_priors.models.LogisticRegression(
            x_train.size(-1), y_train.size(-1), softmax_temp=softmax_temp,
            prior_w=weight_prior, loc_w=weight_loc, std_w=weight_scale,
            prior_b=bias_prior, loc_b=bias_loc, std_b=bias_scale, scaling_fn=scaling_fn,
            weight_prior_params=weight_prior_params, bias_prior_params=bias_prior_params,
            data_aug_prior=data_aug_prior).to(x_train)
    else:
        raise ValueError("model="+model)

    if x_train.device != t.device("cpu"):
        # For some reason, this increases GPU utilization and decreases CPU
        # utilization. The end result is much faster.
        the_net = t.nn.DataParallel(net.net)
    else:
        the_net = DummyModule(net.net)
    del net.net
    net.net = the_net
    return net


def _n_samples_dict(samples):
    n_samples = min(len(v) for _, v in samples.items())

    if not all((len(v) == n_samples) for _, v in samples.items()):
        warnings.warn("Not all samples have the same length. "
                      "Setting n_samples to the minimum.")
    return n_samples

def sample_iter(samples):
    for i in range(_n_samples_dict(samples)):
        yield dict((k, v[i]) for k, v in samples.items())


def evaluate_model(model: bnn_priors.models.AbstractModel,
                   dataloader_test: Iterable[Tuple[t.Tensor, t.Tensor]],
                   samples: Dict[str, t.Tensor],
                   likelihood_eval: bool, accuracy_eval: bool, calibration_eval: bool,
                   label_entropy_eval: bool, eval_acc_vs_n_samples: bool = False,
                   skip_first: int = 0, n_batches: [int, None] = None,
                   eval_aug_probs_std: bool = False):
    if hasattr(dataloader_test.dataset, "tensors"):
        labels = dataloader_test.dataset.tensors[1].cpu()
    elif hasattr(dataloader_test.dataset, "targets"):
        labels = t.tensor(dataloader_test.dataset.targets).cpu()
    else:
        raise ValueError("I cannot find the labels in the dataloader.")

    if n_batches is None:
        n_batches = len(dataloader_test)
    else:
        labels = labels[:n_batches * dataloader_test.batch_size]
    N, *possibly_D = labels.shape
    E = _n_samples_dict(samples)

    if len(possibly_D) == 0:
        n_classes_or_funs = labels.max().item() + 1
    else:
        n_classes_or_funs, = possibly_D

    lps = t.zeros((E, N), dtype=t.float64, device='cpu')
    acc_data = t.zeros((E, N, n_classes_or_funs), dtype=t.float64, device='cpu')
    eval_aug_probs_std = eval_aug_probs_std and bool(model.data_aug_prior)
    if eval_aug_probs_std:
        n_aug_samples = next(iter(dataloader_test))[0].shape[1]
        acc_data_aug_samples = \
            t.zeros((E, N, n_aug_samples, n_classes_or_funs),
                    dtype=t.float64, device='cpu')

    device = next(iter(model.parameters())).device
    n_samples = _n_samples_dict(samples)
    verbose = n_samples > 100
    for sample_i, sample in enumerate(sample_iter(samples)):
        if verbose:
            if sample_i < skip_first:
                print(f'Skipping sample {sample_i + 1}')
                continue
            else:
                skip_first_str = f"(Skipped first {skip_first})." if skip_first > 0 else ""
                print(f'Sample {sample_i + 1 - skip_first} of {n_samples - skip_first}. {skip_first_str}')

        with t.no_grad():
            model.load_state_dict(sample)
            i = 0
            for batch_id, (batch_x, batch_y) in enumerate(dataloader_test):
                if batch_id == n_batches:
                    break
                # if sample_i < 2:
                #     import matplotlib.pyplot as plt
                #     x = batch_x.numpy()[0, 0]
                #     x -= np.min(x)
                #     x /= np.max(x)
                #     plt.imshow(np.transpose(x, [1, 2, 0]))
                #     plt.savefig(f'ttcheck_test{batch_id}_{sample_i}.png', bbox_inches='tight')
                #     plt.close('all')
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_x)
                if isinstance(preds, t.distributions.Categorical):
                    acc_data_batch = preds.logits # B,n_classes
                    lps_batch = preds.log_prob(batch_y)
                elif isinstance(preds, t.distributions.Normal):
                    acc_data_batch = preds.mean  # B,n_latent
                    lps_batch = preds.log_prob(batch_y).sum(-1)
                else:
                    raise ValueError(f"unknown likelihood {type(preds)}")

                if calibration_eval:
                    if not isinstance(preds, t.distributions.Categorical):
                        raise ValueError("Cannot calculate calibration metrics "
                                         f"for predictions of type {type(preds)}")

                next_i = i+len(batch_x)
                lps[sample_i, i:next_i] = lps_batch.detach()
                acc_data[sample_i, i:next_i, :] = acc_data_batch.detach()
                if eval_aug_probs_std:
                    x_shp = list(batch_x.shape)  # batch_size x n_aug_samples x n_channel x H x W
                    x_rs = batch_x.view([x_shp[0] * x_shp[1]] + x_shp[2:])
                    f = model.net(x_rs)
                    acc_data_aug_samples[sample_i, i:next_i, :] = \
                        f.view([x_shp[0], x_shp[1]] + list(f.shape[1:]))
                i = next_i

    def _log_space_mean(tensor, dim):
        return tensor.logsumexp(dim) - math.log(tensor.size(dim))

    # lps: n_samples,len(dataset)
    lps_each_model = lps.mean(1)
    lp_ensemble = _log_space_mean(lps, 0).mean()

    def get_preds_acc_mse(raw_preds):
        # acc_data: n_samples,len(dataset), (n_classes or n_outputs)
        if isinstance(preds, t.distributions.Categorical):
            ensemble_preds = t.distributions.Categorical(logits=_log_space_mean(raw_preds, 0))
            last_preds = t.distributions.Categorical(logits=raw_preds[-1])

            # TODO: put assert statement back in that ensemble preds have similar acc

        elif isinstance(preds, t.distributions.Normal):
            ensemble_preds = t.distributions.Normal(raw_preds.mean(0), t.ones_like(raw_preds[0]))
            last_preds = t.distributions.Normal(raw_preds[-1], ensemble_preds.scale)
        acc_mse_ensemble = model.acc_mse(ensemble_preds, labels).mean(0)
        acc_mse_last = model.acc_mse(last_preds, labels).mean(0)
        return ensemble_preds, last_preds, acc_mse_ensemble, acc_mse_last

    ensemble_preds, last_preds, acc_or_mse_ensemble, acc_or_mse_last = \
        get_preds_acc_mse(acc_data)
    if eval_acc_vs_n_samples:
        n_samples_eval = []
        accs_eval = []
        for nsamp in range(1, n_samples + 1):
            _, _, samples_acc, _ = get_preds_acc_mse(acc_data[:nsamp])
            n_samples_eval.append(nsamp)
            accs_eval.append(samples_acc.item())

    if calibration_eval:
        probs_mean = ensemble_preds.probs.numpy()
        eces = ece(labels, probs_mean)
        aces = ace(labels, probs_mean)
        rmsces = rmsce(labels, probs_mean)

    if label_entropy_eval:
        assert isinstance(preds, t.distributions.Categorical),\
            "Label entropy calculation currently supported for " \
            "categorical likelihoods only."
        entropy_ensemble = ensemble_preds.entropy().mean()
        entropy_last = last_preds.entropy().mean()
        pred_sort_mean_ensemble = \
            t.sort(ensemble_preds.probs, -1)[0].mean(0)
        pred_sort_mean_last = \
            t.sort(last_preds.probs, -1)[0].mean(0)

    if eval_aug_probs_std:
        assert isinstance(preds, t.distributions.Categorical),\
            "Label std calculation currently supported for " \
            "categorical likelihoods only."
        aug_std_mean = \
            t.distributions.Categorical(logits=_log_space_mean(acc_data_aug_samples, 0)).probs.std(-2).mean()

    results = {}
    if eval_acc_vs_n_samples:
        # Evaluate accuracy for different numbers of samples
        results['accs_vs_samples'] = list(zip(n_samples_eval, accs_eval))

    if likelihood_eval:
        results["lp_ensemble"] = lp_ensemble.item()
        results["lp_last"] = lps_each_model[-1].item()
    if accuracy_eval:
        results["acc_ensemble"] = acc_or_mse_ensemble.item()
        results["acc_last"] = acc_or_mse_last.item()
    if calibration_eval:
        results["ece"] = eces.mean().item()
        results["ace"] = aces.mean().item()
        results["rmsce"] = rmsces.mean().item()
    if label_entropy_eval:
        results["ent_ensemble"] = entropy_ensemble.item()
        results["ent_last"] = entropy_last.item()
        results["pred_sort_mean_ensemble"] = pred_sort_mean_ensemble.tolist()
        results["pred_sort_mean_last"] = pred_sort_mean_last.tolist()
    if eval_aug_probs_std:
        results["aug_prob_std"] = aug_std_mean.item()
    return results


def evaluate_ood(model, dataloader_train, dataloader_test, samples):

    loaders = {"train": dataloader_train, "eval": dataloader_test}
    probs = {"train": [], "eval": []}
    aurocs = []
    auprcs = []

    for sample in sample_iter(samples):
        with t.no_grad():
            model.load_state_dict(sample)
            for dataset in ["train", "eval"]:
                probs_sample = []
                for batch_x, batch_y in loaders[dataset]:
                    pred = model(batch_x)
                    # shape: len(batch) x n_classes
                    probs_sample.append(pred.probs.cpu().numpy())
                # shape: len(dataset) x n_classes
                probs_sample = np.concatenate(probs_sample, axis=0)
                probs[dataset].append(probs_sample)

    for dataset in ["train", "eval"]:
        # axis=0 -> over samples of the model
        probs[dataset] = np.mean(probs[dataset], axis=0)
        # axis=-1 -> over class probabilities
        probs[dataset] = np.max(probs[dataset], axis=-1)
        
    labels = np.concatenate([np.ones_like(probs["train"]), np.zeros_like(probs["eval"])])
    probs_cat = np.concatenate([probs["train"], probs["eval"]])
    auroc = roc_auc_score(labels, probs_cat)
    auprc = average_precision_score(labels, probs_cat)
    
    results = {}

    results["auroc"] = float(auroc)
    results["auprc"] = float(auprc)
    
    return results


def evaluate_marglik(model, train_samples, eval_samples):
    log_priors = []

    n_samples = _n_samples_dict(train_samples)
    assert n_samples == _n_samples_dict(eval_samples)
    for train_sample, eval_sample in zip(sample_iter(train_samples),
                                         sample_iter(eval_samples)):
        # Ideally we would only use eval_sample, but the (possibly hierarchical)
        # model has more parameters than eval_sample. This way we start with
        # parameters in `train_sample` and overwrite them with all parameters
        # that are in `eval_sample`
        sampled_state_dict = {**train_sample, **eval_sample}
        with t.no_grad():
            model.load_state_dict(sampled_state_dict)
            log_prior = model.log_prior().item()
            log_priors.append(log_prior)
        
    log_priors = t.tensor(log_priors)
    
    results = {}
    results["simple_logmarglik"] = log_priors.logsumexp(dim=0).item() - math.log(n_samples)
    results["mean_loglik"] = log_priors.mean().item()
    results["simple_marglik"] = log_priors.exp().mean().item()
    return results


class HDF5ModelSaver:
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        self.chunk_size = 1
        self.__i = 0
        self.__init_dsets = True

    def __enter__(self):
        self.f = h5py.File(self.path, self.mode, libver="latest",
                           rdcc_nbytes=0)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.flush()  # Make sure to save all in-memory cache before closing
        self.f.close()

    def add_state_dict(self, state_dict, step):
        d = {k: v.cpu().detach().unsqueeze(0).numpy()
             for (k, v) in state_dict.items()}
        d["steps"] = np.array([step], dtype=np.int64)
        d["timestamps"] = np.array([time.time()], dtype=np.float64)
        self._extend_dict(d)

    def _extend_dict(self, d):
        length = self._assign_dict_current(d)
        self.__i += length

    def _assign_dict_current(self, d) -> int:
        """
       Assigns all data from dictionary `d` to the positions
        `self.__i:self.__i+self.chunk_size` in the HDF5 file.
        """
        if self.__init_dsets:
            for k, v in d.items():
                self._create_dset(k, v.shape[1:], v.dtype)
            # Set the h5 file to Single Reader Multiple Writer (SWMR) mode,
            # the metrics can be read during the run.
            # No more datasets can be created after that.
            # This will prevent any new keys from being added using `add_scalar`.
            self.f.swmr_mode = True
            self.__init_dsets = False

        i = self.__i
        length = None
        for k, value in d.items():
            if length is None:
                length = len(value)
            else:
                assert length == len(value), "lengths unequal"

            dset = self.f[k]
            if i + length >= len(dset):
                dset.resize(i + length, axis=0)
            dset[i:i+length] = value
        assert length is not None
        return length

    def _create_dset(self, name, shape, dtype):
        # int64 stores NaN as -2**63
        assert dtype in [np.float32, np.float64, np.int64], "accepts NaNs"

        return self.f.create_dataset(
            name, dtype=dtype,
            shape=(0,                *shape),
            chunks=(self.chunk_size, *shape),
            maxshape=(None,          *shape),
            fletcher32=True, fillvalue=np.nan)

    def flush(self):
        self.f.flush()

    def load_samples(self, idx=slice(None), keep_steps=True):
        try:
            self.f.flush()
        except:
            pass
        return load_samples(self.path, idx=idx, keep_steps=keep_steps)


class HDF5Metrics(HDF5ModelSaver):
    def __init__(self, path, mode, chunk_size=8*1024):
        super().__init__(path, mode)
        self.chunk_size = chunk_size
        self._step = -2**63
        self._cache = {}
        self._chunk_i = -1
        self.last_flush = time.time()

    def add_scalar(self, name, value, step, dtype=None):
        if step > self._step:
            self._chunk_i += 1
            if self._chunk_i >= self.chunk_size:
                self._extend_dict(self._cache)
                self._chunk_i = 0
                self._scrub_cache()

            self._step = step
            self._append("steps", step, np.int64)
            self._append("timestamps", time.time(), np.float64)

        elif step < self._step:
            raise ValueError(f"step went backwards ({self._step} -> {step})")
        self._append(name, value, type(value))

    def _scrub_cache(self):
        for v in self._cache.values():
            v[:] = np.nan

    def _append(self, name, value, dtype):
        try:
            arr = self._cache[name]
        except KeyError:
            arr = self._cache[name] = np.empty(self.chunk_size, dtype=dtype)
            arr[:] = np.nan
        arr[self._chunk_i] = value

    def flush(self, every_s=0):
        "flush every `every_s` seconds"
        if self._chunk_i < 0:
            return  # Nothing to flush

        now = time.time()
        if every_s <= 0 or now - self.last_flush > every_s:
            self.last_flush = now
            trimmed_cache = {k: v[:self._chunk_i+1] for k, v in self._cache.items()}
            self._assign_dict_current(trimmed_cache)
            self.f.flush()


def load_samples(path, idx=slice(None), keep_steps=True):
    try:
        with h5py.File(path, "r", swmr=True) as f:
            if keep_steps:
                return {k: t.from_numpy(np.asarray(v[idx]))
                        for k, v in f.items()}
            else:
                return {k: t.from_numpy(np.asarray(v[idx]))
                        for k, v in f.items()
                        if k not in ["steps", "timestamps"]}
    except OSError:
        samples = t.load(path)
        return {k: v[idx] for k, v in samples.items()}


def sneaky_artifact(_run, name):
    """modifed `artifact_event` from `sacred.observers.FileStorageObserver`
    Returns path to the name.
    """
    obs = _run.observers[0]
    assert isinstance(obs, sacred.observers.FileStorageObserver)
    obs.run_entry["artifacts"].append(name)
    obs.save_json(obs.run_entry, "run.json")
    return Path(obs.dir)/name


def reject_samples_(samples, metrics_file):
    is_sample = (metrics_file["acceptance/is_sample"][:] == 1)
    try:
        rejected_arr = metrics_file["acceptance/rejected"][is_sample]
    except KeyError:
        return samples
    assert np.all((rejected_arr == 0) | (rejected_arr == 1))
    metrics_steps = metrics_file["steps"][is_sample]
    rejected = {int(s): bool(r) for (s, r) in zip(metrics_steps, rejected_arr)}

    for i in range(_n_samples_dict(samples)):
        sample_step = samples["steps"][i].item()
        if rejected[sample_step]:
            for k in samples.keys():
                samples[k][i] = samples[k][i-1]
    return samples
