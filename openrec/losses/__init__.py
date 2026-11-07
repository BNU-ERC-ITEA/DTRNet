import copy
from importlib import import_module
from torch import nn

name_to_module = {
    'CharWiseVerifyExistDisplayLoss': '.charwise_verify_loss_exist_display',
}


def build_loss(config):
    config = copy.deepcopy(config)
    module_name = config.pop('name')

    if module_name in globals():
        module_class = globals()[module_name]
    else:
        assert module_name in name_to_module, Exception(
            '{} is not supported. The losses in {} are supportes'.format(
                module_name, list(name_to_module.keys())))
        module_path = name_to_module[module_name]
        module = import_module(module_path, package=__package__)
        module_class = getattr(module, module_name)

    return module_class(**config)


class GTCLoss(nn.Module):

    def __init__(self,
                 gtc_loss,
                 gtc_weight=1.0,
                 ctc_weight=1.0,
                 zero_infinity=True,
                 **kwargs):
        super(GTCLoss, self).__init__()
        # 动态构建CTCLoss
        ctc_config = {'name': 'CTCLoss', 'zero_infinity': zero_infinity}
        self.ctc_loss = build_loss(ctc_config)
        # 构建GTC损失
        self.gtc_loss = build_loss(gtc_loss)
        self.gtc_weight = gtc_weight
        self.ctc_weight = ctc_weight

    def forward(self, predicts, batch):
        ctc_loss = self.ctc_loss(predicts['ctc_pred'],
                                 [None] + batch[-2:])['loss']
        gtc_loss = self.gtc_loss(predicts['gtc_pred'], batch[:-2])['loss']
        return {
            'loss': self.ctc_weight * ctc_loss + self.gtc_weight * gtc_loss,
            'ctc_loss': ctc_loss,
            'gtc_loss': gtc_loss
        }
