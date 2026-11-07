import copy

__all__ = ['build_metric']

from .rec_metric import RecMetric

from .rec_metric_char_verify import RecCharVerifyMetric

support_dict = ['RecMetric', 'RecCharVerifyMetric']


def build_metric(config):
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'metric only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class