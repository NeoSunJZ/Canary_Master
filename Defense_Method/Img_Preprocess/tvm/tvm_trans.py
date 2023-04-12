import numpy as np
import torch
from PIL import Image

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

from Defense_Method.Img_Preprocess.tvm.tvm import reconstruct as tvm

sefi_component = SEFIComponent()


@sefi_component.trans_class(trans_name="tvm")
@sefi_component.config_params_handler(handler_target=ComponentType.TRANS, name="tvm",
                                      handler_type=ComponentConfigHandlerType.TRANS_CONFIG_PARAMS, use_default_handler=True,
                                      params={
                                      })
class Transform():
    def __init__(self, pixel_drop_rate=0.05, tvm_method='none', tvm_weight=0.03):
        self.pixel_drop_rate = pixel_drop_rate
        self.tvm_method = tvm_method
        self.tvm_weight = tvm_weight

    @sefi_component.trans(name="tvm", is_inclass=True)
    def img_transform(self, imgs):
        result = []
        for img in imgs:
            if not torch.is_tensor(img):
                img = torch.from_numpy(img)
            img = img / 255
            img = tvm(
                img, self.pixel_drop_rate, self.tvm_method, self.tvm_weight
            )*255
            result.append(img)
        return result
