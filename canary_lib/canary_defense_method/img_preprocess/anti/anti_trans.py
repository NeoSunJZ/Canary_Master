import numpy as np
import torch
from canary_lib.canary_defense_method.img_preprocess.anti.anti import anti_adversary_wrapper

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.trans_class(trans_name="anti")
@sefi_component.config_params_handler(handler_target=ComponentType.TRANS, name="anti",
                                      handler_type=ComponentConfigHandlerType.TRANS_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                      })
class Transform():
    def __init__(self, model, device='cuda'):
        self.device = device
        self.model = model

    @sefi_component.trans(name="anti", is_inclass=True)
    def img_transform(self, imgs):
        anti = anti_adversary_wrapper(model=self.model, k=100, alpha=1/255)
        result = []
        for img in imgs:
            if not torch.is_tensor(img):
                img = torch.from_numpy(img)
            img = anti.forward(img)[0].permute(1, 2, 0)*255
            img = torch.from_numpy(np.clip(img.detach().cpu().numpy(), 0, 255).astype(np.float32))
            result.append(img)
        return result
