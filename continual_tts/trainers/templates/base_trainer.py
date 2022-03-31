import os
import yaml
import torch
from trainer import get_optimizer

from continual_tts.utils.generic import set_random_seed
from continual_tts.models import get_coqui_model
from continual_tts.benchmarks import get_benchmark


class BaseTrainer:
    """
    Base class Trainer. All trainers should inherit from this class.
    """
    def __init__(self, args, params, experiment_name):
        self.params = params

        # Seed
        set_random_seed(params["seed"])

        # Set compute device
        self.device = torch.device("cuda" if
                                   torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

        # Params
        self.args = args
        self.params = params
        self.ds_path = os.path.join(params["datasets_root"],
                                    params["dataset_name"])
        self.experiment_name = experiment_name
        self.config, self.model, self.vocoder = \
            get_coqui_model(params, self.ds_path)

        self.model.vocoder = self.vocoder

        # Initialize benchmark
        self.benchmark = get_benchmark(params, self.ds_path,
                                       self.config, self.model.ap,
                                       self.model.tokenizer)
        # Update config
        self.config.log_to_wandb = self.args.wandb_proj != ""

        # Optimizer
        self.optimizer = get_optimizer(
            optimizer_name=self.config.optimizer,
            optimizer_params=self.config.optimizer_params,
            lr=self.config.lr,
            model=self.model,
        )

        # Criterion
        self.criterion = self.model.get_criterion()

        # Results
        if self.args.save_results:
            # Results path
            results_path = os.path.join(self.args.outputs_dir,
                                        "outputs", self.experiment_name)
            self.results_path = results_path
            os.makedirs(results_path, exist_ok=True)

            # Checkpoints path
            self.checkpoints_path = os.path.join(self.results_path,
                                                 "checkpoints")
            os.makedirs(self.checkpoints_path, exist_ok=True)

            # Save a copy of params to the results folder
            output_params_yml_path = os.path.join(self.results_path,
                                                  "params.yml")
            with open(output_params_yml_path, 'w') as outfile:
                yaml.dump(self.params, outfile, default_flow_style=False)

    def run(self):
        raise NotImplementedError()

    def generate_samples(self):
        raise NotImplementedError()
