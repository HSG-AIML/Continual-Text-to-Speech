from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue


class LossMetric(PluginMetric[float]):
    """
    This metrics computes and logs TTS loss.
    """

    def __init__(self):
        """
        Initialize the metric
        """
        super().__init__()

        self._loss_value = 0.0

    def reset(self) -> None:
        """
        Reset the metric
        """
        self._loss_value = 0.0

    def result(self) -> float:
        """
        Emit the result
        """
        return self._loss_value

    def before_training_iteration(
        self, strategy: "SupervisedTemplate"
    ) -> None:
        """
        Reset the accuracy before the epoch begins
        """
        self.reset()

    def after_training_iteration(
        self, strategy: "SupervisedTemplate"
    ) -> "MetricValue":
        """
        Update the accuracy metric with the current
        predictions and targets
        """
        self._loss_value = strategy.loss.item()
        return self._package_result(strategy)

    def _package_result(self, strategy):
        """Taken from `GenericPluginMetric`, check that class out!"""
        metric_value = self._loss_value
        plot_x_position = strategy.clock.train_iterations
        metric_name = "Loss/train_phase/train_stream/"

        return [MetricValue(self, metric_name, metric_value,
                            plot_x_position)]

    def __str__(self):
        """
        Here you can specify the name of your metric
        """
        return "Loss"
