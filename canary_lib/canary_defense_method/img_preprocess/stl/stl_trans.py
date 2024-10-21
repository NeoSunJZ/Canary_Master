import torch
from canary_lib.canary_defense_method.img_preprocess.stl import stl

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.trans_class(trans_name="stl")
@sefi_component.config_params_handler(handler_target=ComponentType.TRANS, name="stl",
                                      handler_type=ComponentConfigHandlerType.TRANS_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                      })
class Transform():
    def __init__(self, device='cuda', dataset='cifar', npy_name='64_p8_lm0.1'):
        self.device = device
        self.dataset = dataset
        self.npy_name = npy_name

    @sefi_component.trans(name="stl", is_inclass=True)
    def img_transform(self, imgs):
        defense = stl.STL(self.device, self.dataset, self.npy_name)
        result = []
        for img in imgs:
            if not torch.is_tensor(img):
                img = torch.from_numpy(img)
            result.append(defense.forward(img))
        return result

