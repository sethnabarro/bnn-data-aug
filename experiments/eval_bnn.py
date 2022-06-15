"""
Evaluation script for the BNN experiments with different data sets and priors.
Adapted from https://github.com/ratschlab/bnn_priors/blob/main/experiments/eval_bnn.py
"""
import os
import json
import h5py

import numpy as np
import torch as t
from pathlib import Path
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver

from bnn_priors.data import CIFAR10
from bnn_priors import prior
from bnn_priors.inference import SGLDRunner
from bnn_priors import exp_utils
from bnn_priors.exp_utils import get_prior

# Makes CUDA faster
if t.cuda.is_available():
    t.backends.cudnn.benchmark = True

TMPDIR = "/tmp"

ex = Experiment("bnn_evaluation")
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def config():
    config_file = None
    eval_data = None
    eval_samples = None
    likelihood_eval = True
    accuracy_eval = True
    calibration_eval = False
    ood_eval = False
    marglik_eval = False
    label_entropy_eval = False
    eval_acc_vs_n_samples = False
    skip_first = 0
    data_aug_prior_type = None
    normtype = None
    data_kwargs = None
    rec_field_scaling = True
    weight_scale_logits = None
    softmax_temp = 1.
    n_batches = None
    eval_on_train_set = False
    eval_aug_probs_std = False
    weight_prior_params = {}
    bias_prior_params = {}
    no_prior_scaling = False

    assert config_file is not None, "No config_file provided"
    ex.add_config(config_file)  # Adds config entries from the previous script
    run_dir = os.path.dirname(config_file)
    eval_dir = os.path.join(run_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    ex.observers.append(FileStorageObserver(eval_dir))

device = ex.capture(exp_utils.device)
get_model = ex.capture(exp_utils.get_model)
evaluate_model = ex.capture(exp_utils.evaluate_model)
evaluate_ood = ex.capture(exp_utils.evaluate_ood)
evaluate_marglik = ex.capture(exp_utils.evaluate_marglik)

@ex.capture
def get_eval_data(data, eval_data, data_kwargs):
    # TODO load synthetic data if present
    if eval_data is not None:
        return exp_utils.get_data(eval_data, device(), eval(str(data_kwargs)))
    else:
        return exp_utils.get_data(data, device(), eval(str(data_kwargs)))


@ex.capture
def get_train_data(data):
    return exp_utils.get_data(data, device())


@ex.automain
def main(config_file, batch_size, n_samples, run_dir, eval_data, data, skip_first, eval_samples,
         likelihood_eval, accuracy_eval, calibration_eval, ood_eval, marglik_eval, label_entropy_eval,
         data_aug_prior_type, eval_acc_vs_n_samples, n_batches, eval_on_train_set):
    assert skip_first < n_samples, "We don't have that many samples to skip"
    run_dir = Path(run_dir)
    with open(run_dir/"run.json") as infile:
        run_data = json.load(infile)

    assert "samples.pt" in run_data["artifacts"], "No samples found"

    samples = exp_utils.load_samples(run_dir/"samples.pt",
                                     idx=np.s_[skip_first:])
    with h5py.File(run_dir/"metrics.h5", "r") as metrics_file:
        exp_utils.reject_samples_(samples, metrics_file)
    del samples["steps"]
    del samples["timestamps"]

    if eval_data is None:
        eval_data = data
    data = get_eval_data()
    data_aug_prior_default = exp_utils.is_for_data_aug_prior(data)
    x_train = data.norm.train_X
    y_train = data.norm.train_y

    x_test = data.norm.test_X
    y_test = data.norm.test_y

    data_aug_prior_type = data_aug_prior_default if data_aug_prior_type is None else data_aug_prior_type
    model = get_model(x_train=x_train, y_train=y_train,
                      data_aug_prior=data_aug_prior_type)

    model.eval()

    if batch_size is None:
        batch_size = len(data.norm.test)
    else:
        batch_size = min(batch_size, len(data.norm.test))
    n_workers = (0 if isinstance(data.norm.train, t.utils.data.TensorDataset) else 3)
    ds_norm = data.norm.train if eval_on_train_set else data.norm.test
    dataloader_eval = t.utils.data.DataLoader(ds_norm,
                                              batch_size=batch_size,
                                              num_workers=n_workers,
                                              shuffle=False)   # No shuffle to compare diff models on same batch

    if calibration_eval and not (eval_data[:7] == "cifar10" or eval_data[-5:] == "mnist"):
        raise NotImplementedError("The calibration is not defined for this type of data.")

    if ood_eval and not (eval_data[:7] == "cifar10" or eval_data[-5:] == "mnist"):
        raise NotImplementedError("The OOD error is not defined for this type of data.")

    results = evaluate_model(model=model,
                             dataloader_test=dataloader_eval,
                             samples=samples,
                             n_batches=n_batches)

    if ood_eval:
        train_data = get_train_data()
        dataloader_train = t.utils.data.DataLoader(train_data.norm.test, batch_size=batch_size)
        ood_results = evaluate_ood(model=model,
                                   dataloader_train=dataloader_train,
                                   dataloader_test=dataloader_eval,
                                   samples=samples)
        results = {**results, **ood_results}

    if marglik_eval:
        if eval_samples is None:
            eval_samples = samples
        else:
            eval_samples = exp_utils.load_samples(eval_samples, idx=np.s_[skip_first:])
            del eval_samples["steps"]
            del eval_samples["timestamps"]
        marglik_results = evaluate_marglik(model=model, train_samples=samples,
                                           eval_samples=eval_samples)
        results = {**results, **marglik_results}

    return results
