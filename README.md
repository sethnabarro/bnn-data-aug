# Data Augmentation in Bayesian Neural Networks

[![](https://img.shields.io/badge/arXiv-2106.05586-red)](https://arxiv.org/abs/2106.05586)

Codebase for the BNN experiments in [_Data augmentation in Bayesian neural networks and the cold posterior effect_](https://openreview.net/pdf?id=rZEM7ULs5x5), by Seth Nabarro, Stoil Ganev, Adrià Garriga-Alonso, Vincent Fortuin, Mark van der Wilk, Laurence Aitchison. This code is adapted from the `bnn_priors` library ([paper](https://www.sciencedirect.com/science/article/pii/S2665963821000270), [github](https://github.com/ratschlab/bnn_priors)), though significant parts of the library not relevant to our paper have been removed. See the `bnn_priors` github for the full version.

All the experiments needed to reproduce Figure 4 of our paper can be run using the `run_mnist.sh` and `run_cifar10.sh`  scripts in `experiments/bayes_data_aug/`. These will run both the MCMC and the evaluation of the resulting samples. Make sure the `repo_dir` variable is set to the absolute path to the `bnn-data-aug/` directory, and `python_exec` points to a python executable in an environment with the necessary dependencies installed.

## Dependencies

Use the `requirements.txt`.

## Outputs

The output, including model samples, diagnostics and evaluation results, will be written to the subdirectory `bnn-data-aug/results/<dataset_name>/<date>/<run_id>`, where `run_id` is automatically generated to avoid overwriting previous results.

## Experiment Configs

The runs for Figure 4 require only the `data` and `extra_args` variables in `experiments/bayes_data_aug/run_*.sh` to be altered. Different augmentation averaging schemes can be specified through the `data` variable. The configurations should be set as follows:

| Train Avg | Test Avg | Augmentation      | `data=`                               | `extra_args=`                | Linestyle in Fig. 4 |
|-----------|----------|-------------------|---------------------------------------|------------------------------|---------------------|
| None      | None     | None              | "<dataset>"                           | ""                           | Dashed, black       |
| None      | None     | Full orbit        | "<dataset>_data_aug_prior-1"          | ""                           | Solid, black        |
| Logits    | Logits   | Full orbit        | "<dataset>_data_aug_prior-8-test8"    | "data_aug_prior_type=logits" | Solid, purple       |
| Logits    | Logits   | Finite orbit of 8 | "<dataset>_fix_data_aug_prior-8-<aug_seed>-test8" | "data_aug_prior_type=logits" | Solid, purple       |
| Probs     | Probs    | Full orbit        | "<dataset>_data_aug_prior-8-test8"    | "data_aug_prior_type=probs"  | Solid, green        |
| Probs     | Probs    | Finite orbit of 8 | "<dataset>_fix_data_aug_prior-8-<aug_seed>-test8" | "data_aug_prior_type=probs"  | Solid, green        |
| None      | Logits   | Full orbit        | "<dataset>_data_aug_prior-1-test8"    | "data_aug_prior_type=logits" | Faded, purple       |
| None      | Probs    | Full orbit        | "<dataset>_data_aug_prior-1-test8"    | "data_aug_prior_type=probs"  | Faded, green        |

Where <dataset> is "cifar10" or "mnist" accordingly, and <aug_seed> is the seed for random augmentation samples. Other variables such as the architecture, learning rates etc are already set to the correct values in `run_mnist.sh` and `run_cifar10.sh`.

## SGD

To run the configs with SGD rather than SGLD, set `priors=("improper")` and `temps=(0.0)` in `run_*.sh`.

## Time to run

On a NVIDIA RTX6000 GPU, the CIFAR10 experiments with no augmentation averaging take around 20 hours, those which average over eight samples per input take around 2 days. For MNIST, these runs take around 6 and 20 hours respectively.

## How to cite

```bibtex
@article{nabarro2021data,
  title={Data augmentation in Bayesian neural networks and the cold posterior effect},
  author={Nabarro, Seth and Ganev, Stoil and Garriga-Alonso, Adri{\`a} and Fortuin, Vincent and van der Wilk, Mark and Aitchison, Laurence},
  journal={arXiv preprint arXiv:2106.05586},
  year={2021}}
```

