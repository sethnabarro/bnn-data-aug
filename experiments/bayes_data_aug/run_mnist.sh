#!/bin/bash

: '
Run experiments to test the relation between cold posterior effect and data augmentation.
Adapted from `run_experiment.sh` in bnn_priors library (https://github.com/ratschlab/bnn_priors/blob/main/experiments/run_experiment.sh).
'

python_exec="/vol/bitbucket/$(whoami)/envs/check_bnn_env/bin/python"                   # Path to python exec
repo_dir="/homes/$(whoami)/code/bnn_data_aug/"     # Path to repo directory
gpus="0"                                                                     # Cuda device ID

# Inference config
data="mnist_data_aug_prior-1"
data_kwargs="{'download': True}"      # can use "{'load': True}" or "{'save': True}" if using fixed augments
temps=( 0.001 )         # Temperatures
priors=( gaussian )   # "improper" for sgd
num_cycles=60         # Number of repeats of step size schedule
warmup=45             # How many epochs in each cycle before collecting samples
burnin=0
n_samples=300         # Total number of posterior samples
momentum=0.994
scales=( 4.24 )       # 1.41 2.82 Prior variance scale
batchnorm="False"
model="classificationdensenet"  # Architecture
lrs=(0.005)
n_repeats=1

# Any other args can be written here.
# For augmentation avging with invariance prior:
# can specify "data_aug_prior_type=probs" or "data_aug_prior_type=logits"
# Can leave as empty string if not doing augmentation averaging.
extra_args="data_aug_prior_type=logits"

# Where results will be stored
logdir="$repo_dir/results/$data/$(date +'%y%m%d')/"

# Argparser
# NOTE: any args given via command line will override options specified above.
for i in "$@"
do
case $i in
    -d=*|--data=*)
      data="${i#*=}"
      shift
      ;;
    -g=*|--gpus=*)
      gpus="${i#*=}"
      shift
      ;;
    -t=*|--temperatures=*)
      temp_str="${i#*=}"

      # Read into an array named "temps"
      IFS=' ' read -r -a temps <<< "$temp_str"
      shift
      ;;
    *)
      echo "Unknown arg: ${i}"
      ;;
  esac
done

# Must be in same directory as train_bnn.py
cd "$repo_dir/experiments"

for i in {1..$n_repeats} # number of repeats
do
    for prior in "${priors[@]}"
    do
        for scale in "${scales[@]}"
        do
            for temp in "${temps[@]}"
            do
                for lr in "${lrs[@]}"
                do
                  cmd="CUDA_VISIBLE_DEVICES=$gpus PYTHONPATH=$PYTHONPATH:$repo_dir $python_exec train_bnn.py with weight_prior=$prior data=$data inference=VerletSGLDReject model=$model warmup=$warmup burnin=$burnin skip=1 n_samples=$n_samples lr=$lr momentum=$momentum weight_scale=$scale cycles=$num_cycles batch_size=128 temperature=$temp save_samples=True progressbar=True log_dir=$logdir batchnorm=$batchnorm data_kwargs=\"$data_kwargs\" $extra_args"
                  printf "\n\nRunning: $cmd\n\n"
                  eval $cmd
                  printf "Finished\n\n"
                done
            done
        done
    done
done
