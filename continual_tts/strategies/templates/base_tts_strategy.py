from typing import Sequence, Optional
import torch
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
            criterion: Module,
            train_mb_size: int = 1,
            train_epochs: int = 1,
            eval_mb_size: int = 1,
            device="cpu",
            plugins: Optional[Sequence["SupervisedPlugin"]] = None,
            evaluator: EvaluationPlugin = default_evaluator,
            eval_every=-1,
            peval_mode="epoch",
    ):
        """

        :param model:
        :param optimizer:
        :param train_mb_size:
        :param train_epochs:
        :param eval_mb_size:
        :param device:
        :param plugins:
        :param evaluator:
        :param eval_every:
        :param peval_mode:
        """
        super().__init__(model, optimizer,
                         train_mb_size=train_mb_size,
                         train_epochs=train_epochs,
                         eval_mb_size=eval_mb_size,
                         device=device,
                         plugins=plugins,
                         evaluator=evaluator,
                         eval_every=eval_every,
                         peval_mode=peval_mode)

        self._criterion = criterion
        self.mb_outputs = None
        self.mb_loss_dict = None

    def training_epoch(self, **kwargs):
        raise NotImplementedError

    def make_train_dataloader(
            self, num_workers=0, shuffle=True, pin_memory=True,
            persistent_workers=False, sampler=None, **kwargs
    ):
        self.dataloader = DataLoader(
            self.experience.dataset,
            collate_fn=self.experience.dataset.collate_fn,
            batch_size=self.train_mb_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle
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
        self.mbatch = self.model.format_batch(self.mbatch)

        # Add speaker embedding to the batch
        speaker_embeddings = [
            self.model.speaker_manager.get_d_vectors_by_speaker(spk) for spk in
            self.mbatch["speaker_names"]]
        speaker_embeddings = torch.FloatTensor(speaker_embeddings).squeeze(1)
        self.mbatch["d_vectors"] = speaker_embeddings.to(self.device)

        # Move to compute device
        for k in self.mbatch.keys():
            if isinstance(self.mbatch[k], torch.Tensor):
                self.mbatch[k] = self.mbatch[k].to(self.device)

    def forward(self):
        """ Forward function for `self.mbatch`
            self.mbatch: mini batch data
        """
        outputs, loss_dict = self.model.train_step(
            self.mbatch,
            self._criterion
        )

        return outputs, loss_dict
