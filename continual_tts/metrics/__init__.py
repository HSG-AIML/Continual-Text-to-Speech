from .loss_metric import LossMetric


def get_metrics(metric_names, params):
    metric_list = []
    for metric_name in metric_names:
        if metric_name == "loss":
            metric_list.append(LossMetric())
        else:
            raise NotImplementedError()
    return metric_list
