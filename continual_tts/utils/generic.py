import yaml
import random
import numpy as np
import torch
import importlib
import itertools
import time


def load_params(yml_file_path):
    """ Loads param file and returns a dictionary.  """
    with open(yml_file_path, "r") as yaml_file:
        params = yaml.safe_load(yaml_file)

    return params


def set_random_seed(seed):
    print("Setting random seed: ", seed)
    random.seed(seed)
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False


def get_all_hp_combinations(selected_params):
    """ Computes all hyper-parameter combinations and returns them
        as a list of dictionaries.
    """
    keys, values = zip(*selected_params.items())
    permutations_dicts = [dict(zip(keys, v)) for v in
                          itertools.product(*values)]
    return permutations_dicts


def get_trainer(trainer_name):
    """ Returns trainer
    :param trainer_name:
    :return: Trainer
    """
    trainer_ = importlib.import_module(f"cl_tts.trainers.{trainer_name}")
    Trainer = getattr(trainer_, "Trainer")

    return Trainer


def get_experiment_name(params):
    experiment_name = f'{params["trainer"]}_'
    experiment_name += f'{params["benchmark"]}_'
    experiment_name += f'{params["model"]}_'
    experiment_name += f's{params["seed"]}_'

    t_suff = time.strftime("%m%d%H%M%S")
    experiment_name += f'{t_suff}'

    return experiment_name


def update_config(config, params):
    for k in params.keys():
        setattr(config, k, params[k])

    return config
