from typing import Dict, Callable

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from avalanche.training.templates.base_sgd import BaseSGDTemplate
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator


class BaseTTSStrategy(BaseSGDTemplate):
    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            params: Dict,
            forward_func: Callable,
            criterion_func: Callable,
            collator: Callable,
            *,
            num_workers: int = 4,
            device="cpu",
            plugins=None,
            evaluator: EvaluationPlugin = default_evaluator,
    ):
        """

        :param model:
        :param optimizer:
        :param params:
        :param forward_func:
        :param criterion_func:
        :param device:
        :param plugins:
        :param evaluator:
        """
        super().__init__(model, optimizer,
                         train_mb_size=params["train_mb_size"],
                         train_epochs=params["train_epochs"],
                         eval_mb_size=params["eval_mb_size"],
                         device=device, plugins=plugins, evaluator=evaluator,
                         eval_every=params["eval_every"],
                         peval_mode=params["peval_mode"])

        self.forward_func = forward_func
        self.criterion_func = criterion_func
        self.collator = collator
        self.num_workers = num_workers

    def training_epoch(self, **kwargs):
        raise NotImplementedError

    def make_train_dataloader(self, **kwargs):
        self.dataloader = DataLoader(
            self.experience.dataset,
            collate_fn=self.collator,
            batch_size=self.train_mb_size,
            sampler=None,  # For now no sampler is supported
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            shuffle=True
        )

    @property
    def is_eval(self):
        """True if the strategy is in evaluation mode."""
        return not self.is_training

    def model_adaptation(self, model=None):
        return self.model

    def make_optimizer(self, **kwargs):
        pass

    def _unpack_minibatch(self):
        """Move to device"""
        for k in self.mbatch[0].keys():
            self.mbatch[0][k] = self.mbatch[0][k].to(self.device)
        self.mbatch[1] = self.mbatch[1].to(self.device)

    def forward(self):
        """ Forward function for `self.mbatch`
            mbatch[0]: mini batch data
            mbatch[1]: speakers
        """
        return self.forward_func(
            self.model,
            self.mbatch[0],
            self.mbatch[1]
        )

    def criterion(self):
        """ Criterion function for self.mb_out
        """
        return self.criterion_func(self.mb_output,
                                   self.mbatch[0],
                                   self.mbatch[1])
