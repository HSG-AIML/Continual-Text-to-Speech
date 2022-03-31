from .sample_synthesizer import SampleSynthesizer
from .gradient_clipper import GradClipper


def get_plugins(plugin_name):
    metric_list = []
    for plugin_name in plugin_name:
        if plugin_name == "sample_synthesizer":
            metric_list.append(SampleSynthesizer())
        elif plugin_name == "grad_clipper":
            metric_list.append(GradClipper())
        else:
            raise NotImplementedError()
    return metric_list
