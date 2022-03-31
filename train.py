from continual_tts.utils.limit_threads import *

import argparse
import copy
import random

from continual_tts.utils.generic import load_params, get_trainer, \
    get_experiment_name, get_all_hp_combinations


def run(args, params):
    # Get trainer
    Trainer = get_trainer(params["trainer"])

    # If single run: just run for the current config
    if params["tunable_parameters"] is None:
        experiment_name = get_experiment_name(params)
        trainer = Trainer(args, params, experiment_name)
        trainer.run()

    # Otherwise: compute all possible configurations and
    # run the trainer for each hyper-parameter config
    else:
        params_subset = {k: params[k] for k in params["tunable_parameters"]}
        hp_permutations = get_all_hp_combinations(params_subset)
        random.shuffle(hp_permutations)

        for itr_hp, hp_comb in enumerate(hp_permutations):
            # Create experiment params
            params_exp = copy.copy(params)
            params_exp.update(hp_comb)

            # Set experiment's name
            experiment_name = get_experiment_name(params_exp)

            # Run trainer
            trainer = Trainer(args, params_exp, experiment_name)
            trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs_dir', type=str, default='./out')
    parser.add_argument('--dataset_root', type=str,
                        default='./benchmarks/datasets')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--params_path', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--wandb_proj', type=str, default="")
    parser.add_argument('--save_results', action='store_true', default=False)
    args = parser.parse_args()
    
    # Load params from YAML file
    params = load_params(args.params_path)
    params = params[args.config]

    params["num_workers"] = args.num_workers

    run(args, params)
