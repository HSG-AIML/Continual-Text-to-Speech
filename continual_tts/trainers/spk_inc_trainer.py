import wandb
import torch
import os

from .templates import BaseCLTrainer


class Trainer(BaseCLTrainer):
    def __init__(self, args, params, experiment_name):
        super().__init__(args, params, experiment_name)

    def run(self):
        for itr_exp, exp in enumerate(self.benchmark.train_stream):
            self.strategy.train(exp, num_workers=self.args.num_workers)

            # Save results
            if self.args.save_results:
                ckpt_path = os.path.join(self.checkpoints_path,
                                         f"ckpt_{itr_exp}.pt")
                torch.save(self.strategy.model.state_dict(), ckpt_path)

        if self.config.log_to_wandb:
            wandb.finish()
