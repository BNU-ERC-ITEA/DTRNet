import torch.nn as nn
from importlib import import_module

__all__ = ['build_decoder']

class_to_module = {

    'CharWiseVerifyStyleDecoder': '.charwise_verify_style_decoder',


}


def build_decoder(config):

    module_name = config.pop('name')

    # Check if the class is defined in current module (e.g., GTCDecoder)
    if module_name in globals():
        module_class = globals()[module_name]
    else:
        if module_name not in class_to_module:
            raise ValueError(f'Unsupported decoder: {module_name}')
        module_str = class_to_module[module_name]
        # Dynamically import the module and get the class
        module = import_module(module_str, package=__package__)
        module_class = getattr(module, module_name)

    return module_class(**config)


class GTCDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 gtc_decoder,
                 ctc_decoder,
                 detach=True,
                 infer_gtc=False,
                 out_channels=0,
                 **kwargs):
        super(GTCDecoder, self).__init__()
        self.detach = detach
        self.infer_gtc = infer_gtc
        if infer_gtc:
            gtc_decoder['out_channels'] = out_channels[0]
            ctc_decoder['out_channels'] = out_channels[1]
            gtc_decoder['in_channels'] = in_channels
            ctc_decoder['in_channels'] = in_channels
            self.gtc_decoder = build_decoder(gtc_decoder)
        else:
            ctc_decoder['in_channels'] = in_channels
            ctc_decoder['out_channels'] = out_channels
        self.ctc_decoder = build_decoder(ctc_decoder)

    def forward(self, x, data=None):
        ctc_pred = self.ctc_decoder(x.detach() if self.detach else x,
                                    data=data)
        if self.training or self.infer_gtc:
            gtc_pred = self.gtc_decoder(x.flatten(2).transpose(1, 2),
                                        data=data)
            return {'gtc_pred': gtc_pred, 'ctc_pred': ctc_pred}
        else:
            return ctc_pred


class GTCDecoderTwo(nn.Module):

    def __init__(self,
                 in_channels,
                 gtc_decoder,
                 ctc_decoder,
                 infer_gtc=False,
                 out_channels=0,
                 **kwargs):
        super(GTCDecoderTwo, self).__init__()
        self.infer_gtc = infer_gtc
        gtc_decoder['out_channels'] = out_channels[0]
        ctc_decoder['out_channels'] = out_channels[1]
        gtc_decoder['in_channels'] = in_channels
        ctc_decoder['in_channels'] = in_channels
        self.gtc_decoder = build_decoder(gtc_decoder)
        self.ctc_decoder = build_decoder(ctc_decoder)

    def forward(self, x, data=None):
        x_ctc, x_gtc = x
        ctc_pred = self.ctc_decoder(x_ctc, data=data)
        if self.training or self.infer_gtc:
            gtc_pred = self.gtc_decoder(x_gtc.flatten(2).transpose(1, 2),
                                        data=data)
            return {'gtc_pred': gtc_pred, 'ctc_pred': ctc_pred}
        else:
            return ctc_pred
