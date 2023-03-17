import numpy as np
import torch
from PIL import Image

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

from Defense_Method.Img_Preprocess.tvm.tvm import reconstruct as tvm

sefi_component = SEFIComponent()


@sefi_component.trans_class(trans_name="tvm")
@sefi_component.config_params_handler(handler_target=ComponentType.TRANS, name="tvm",
                                      args_type=ComponentConfigHandlerType.TRANS_PARAMS, use_default_handler=True,
                                      params={
                                      })
class Transform():
    def __init__(self, args={}):
        self.args = args
        self.pixel_drop_rate = self.args.get('pixel_drop_rate', 0.05)
        self.tvm_method = self.args.get('tvm_method', 'none')
        self.tvm_weight = self.args.get('tvm_weight', 0.03)

    @sefi_component.trans(name="tvm", is_inclass=True)
    def img_transform(self, img):
        if not torch.is_tensor(img):
            img = torch.from_numpy(img)
        img = img / 255
        im = tvm(
            img, self.pixel_drop_rate, self.tvm_method, self.tvm_weight
        )*255
        return im
