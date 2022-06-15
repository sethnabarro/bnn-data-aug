"""
Training script for the BNN experiments with different data sets and priors.
Adapted from https://github.com/ratschlab/bnn_priors/blob/main/experiments/train_bnn.py
"""

import os
import math
import uuid
import json
import contextlib

import numpy as np
import torch as t
from pathlib import Path
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import FileStorageObserver

import bnn_priors.inference
import bnn_priors.inference_reject
from bnn_priors import exp_utils
from bnn_priors.exp_utils import is_for_data_aug_prior

# Makes CUDA faster
if t.cuda.is_available():
    t.backends.cudnn.benchmark = True

TMPDIR = "/tmp"

ex = Experiment("bnn_training")
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def config():
    data = "mnist"
    inference = "SGLD"
    model = "classificationdensenet"
    width = 100
    depth = 3
    weight_prior = "gaussian"
    bias_prior = "gaussian"
    weight_loc = 0.
    weight_scale = 2.**0.5
    weight_scale_logits = None  # For last linear layer weights, will use `weight_scale` if not given
    bias_loc = 0.
    bias_scale = 1.
    weight_prior_params = {}
    bias_prior_params = {}
    if not isinstance(weight_prior_params, dict):
        weight_prior_params = json.loads(weight_prior_params)
    if not isinstance(bias_prior_params, dict):
        bias_prior_params = json.loads(bias_prior_params)
    n_samples = 1000
    warmup = 2000
    burnin = 2000
    skip = 5
    epochs_per_eval = 1
    metrics_skip = 10
    cycles =  5
    temperature = 1.0
    sampling_decay = "cosine"
    momentum = 0.9
    precond_update = 1
    lr = 5e-4
    init_method = "he"
    load_samples = None
    batch_size = None
    reject_samples = False
    batchnorm = False
    device = "try_cuda"
    save_samples = True
    progressbar = True
    data_kwargs = None
    rec_field_scaling = True
    no_prior_scaling = False
    data_aug_prior_type = None
    softmax_temp = 1.
    run_id = uuid.uuid4().hex
    log_dir = str(Path(__file__).resolve().parent.parent/"logs")
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        ex.observers.append(FileStorageObserver(log_dir))
    

device = ex.capture(exp_utils.device)
get_model = ex.capture(exp_utils.get_model)

@ex.capture
def get_data(data, batch_size, _run, data_kwargs):
    return exp_utils.get_data(data, device(), eval(str(data_kwargs)))


@ex.capture
def evaluate_model(model, dataloader_test, samples):
    return exp_utils.evaluate_model(
        model=model, dataloader_test=dataloader_test, samples=samples,
        likelihood_eval=True, accuracy_eval=True,
        calibration_eval=True, label_entropy_eval=True, eval_acc_vs_n_samples=False)


@ex.automain
def main(inference, model, width, n_samples, warmup, init_method,
         burnin, skip, metrics_skip, cycles, temperature, momentum,
         precond_update, lr, batch_size, load_samples, save_samples,
         reject_samples, run_id, log_dir, sampling_decay, progressbar,
         data_aug_prior_type, epochs_per_eval, _run, _log):
    assert inference in ["SGLD", "VerletSGLD", "VerletSGLDReject"]
    assert width > 0
    assert n_samples > 0
    assert cycles > 0
    assert temperature >= 0

    data = get_data()

    x_train = data.norm.train_X
    y_train = data.norm.train_y

    x_test = data.norm.test_X
    y_test = data.norm.test_y

    data_aug_prior_type = is_for_data_aug_prior(data) if data_aug_prior_type is None else data_aug_prior_type
    model = get_model(x_train=x_train, y_train=y_train, data_aug_prior=data_aug_prior_type)

    if load_samples is None:
        if init_method == "he":
            exp_utils.he_initialize(model)
        elif init_method == "he_uniform":
            exp_utils.he_uniform_initialize(model)
        elif init_method == "he_zerobias":
            exp_utils.he_zerobias_initialize(model)
        elif init_method == "prior":
            pass
        else:
            raise ValueError(f"unknown init_method={init_method}")
    else:
        state_dict = exp_utils.load_samples(load_samples, idx=-1, keep_steps=False)
        model_sd = model.state_dict()
        for k in state_dict.keys():
            if k not in model_sd:
                _log.warning(f"key {k} not in model, ignoring")
                del state_dict[k]
            elif model_sd[k].size() != state_dict[k].size():
                _log.warning(f"key {k} size mismatch, model={model_sd[k].size()}, loaded={state_dict[k].size()}")
                state_dict[k] = model_sd[k]

        missing_keys = set(model_sd.keys()) - set(state_dict.keys())
        _log.warning(f"The following keys were not found in loaded state dict: {missing_keys}")
        model_sd.update(state_dict)
        model.load_state_dict(model_sd)
        del state_dict
        del model_sd

    if save_samples:
        model_saver_fn = (lambda: exp_utils.HDF5ModelSaver(
            exp_utils.sneaky_artifact(_run, "samples.pt"), "w"))
    else:
        @contextlib.contextmanager
        def model_saver_fn():
            yield None

    with exp_utils.HDF5Metrics(
            exp_utils.sneaky_artifact(_run, "metrics.h5"), "w") as metrics_saver,\
         model_saver_fn() as model_saver:
        if inference == "SGLD":
            runner_class = bnn_priors.inference.SGLDRunner
        elif inference == "VerletSGLD":
            runner_class = bnn_priors.inference.VerletSGLDRunner
        elif inference == "VerletSGLDReject":
            runner_class = bnn_priors.inference_reject.VerletSGLDRunnerReject

        assert (n_samples * skip) % cycles == 0
        sample_epochs = n_samples * skip // cycles
        epochs_per_cycle = warmup + burnin + sample_epochs
        if batch_size is None:
            batch_size = len(data.norm.train)
        # Disable parallel loading for `TensorDataset`s.
        num_workers = (0 if isinstance(data.norm.train, t.utils.data.TensorDataset) else 4)
        dataloader = t.utils.data.DataLoader(data.norm.train, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=num_workers)
        dataloader_test = t.utils.data.DataLoader(data.norm.test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
        mcmc = runner_class(model=model, dataloader=dataloader, dataloader_test=dataloader_test, epochs_per_cycle=epochs_per_cycle,
                            warmup_epochs=warmup, sample_epochs=sample_epochs, learning_rate=lr,
                            skip=skip, metrics_skip=metrics_skip, sampling_decay=sampling_decay, cycles=cycles, temperature=temperature,
                            momentum=momentum, precond_update=precond_update, epochs_per_eval=epochs_per_eval,
                                metrics_saver=metrics_saver, model_saver=model_saver, reject_samples=reject_samples)

        mcmc.run(progressbar=progressbar)
    samples = mcmc.get_samples()
    model.eval()

    batch_size = min(batch_size, len(data.norm.test))
    dataloader_test = t.utils.data.DataLoader(data.norm.test, batch_size=batch_size)

    return evaluate_model(model, dataloader_test, samples)
