import math
import torch

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.trans_class(trans_name="quantize")
@sefi_component.config_params_handler(handler_target=ComponentType.TRANS, name="quantize",
                                      handler_type=ComponentConfigHandlerType.TRANS_CONFIG_PARAMS, use_default_handler=True,
                                      params={
                                      })
class Transform():
    def __init__(self, args={}):
        self.args = args

    @sefi_component.trans(name="quantize", is_inclass=True)
    def img_transform(self, img):
        if not torch.is_tensor(img):
            img = torch.from_numpy(img)
        return _quantize_img(img, depth=self.args.get("quantize_depth", 8))


def _quantize_img(im, depth=8):
    assert torch.is_tensor(im)
    N = int(math.pow(2, depth))
    im = (im * N).round()
    im = im / N
    return im
