from typing import Sequence, Optional

from torch.nn import Module
from torch.optim import Optimizer

from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.evaluation import default_evaluator

from .templates import BaseTTSStrategy


class Naive(BaseTTSStrategy):
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
        super().__init__(model, optimizer, criterion,
                         train_mb_size=train_mb_size,
                         train_epochs=train_epochs,
                         eval_mb_size=eval_mb_size,
                         device=device,
                         plugins=plugins,
                         evaluator=evaluator,
                         eval_every=eval_every,
                         peval_mode=peval_mode)

    def training_epoch(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.optimizer.zero_grad()
            self.loss = 0

            # Forward
            self._before_forward(**kwargs)
            self.mb_outputs, self.mb_loss_dict = self.forward()
            self._after_forward(**kwargs)

            self.loss += self.mb_loss_dict["loss"]

            # Loss & Backward
            self._before_backward(**kwargs)
            self.loss.backward()
            self._after_backward(**kwargs)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer.step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)
