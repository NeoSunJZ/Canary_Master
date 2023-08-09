import math
import torch

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.trans_class(trans_name="quantize")
@sefi_component.config_params_handler(handler_target=ComponentType.TRANS, name="quantize",
                                      handler_type=ComponentConfigHandlerType.TRANS_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                      })
class Transform():
    def __init__(self, quantize_depth=3):
        self.quantize_depth = quantize_depth

    @sefi_component.trans(name="quantize", is_inclass=True)
    def img_transform(self, imgs):
        result = []
        for img in imgs:
            if not torch.is_tensor(img):
                img = torch.from_numpy(img)
            result.append(self._quantize_img(img, depth=self.quantize_depth))
        return result

    @staticmethod
    def _quantize_img(im, depth=3):
        assert torch.is_tensor(im)
        im /= 255
        N = int(math.pow(2, depth))
        im = (im * N).round()
        im = im / N
        im *= 255
        return im
