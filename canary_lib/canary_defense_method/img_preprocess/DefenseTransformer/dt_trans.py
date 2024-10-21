import torch

from canary_lib.canary_defense_method.img_preprocess.DefenseTransformer.dt import DefenseTransformer
from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()
trans_name = "dt"


@sefi_component.trans_class(trans_name=trans_name)
@sefi_component.config_params_handler(handler_target=ComponentType.TRANS, name=trans_name,
                                      handler_type=ComponentConfigHandlerType.TRANS_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                      })
class Transform():
    def __init__(self):
        pass

    @sefi_component.trans(name=trans_name, is_inclass=True)
    def img_transform(self, imgs):
        dt = DefenseTransformer()
        result = []
        for img in imgs:
            if not torch.is_tensor(img):
                img = torch.from_numpy(img)
            result.append(dt.dt_img(img.permute(2, 0, 1).unsqueeze(0).cuda())[0].permute(1, 2, 0))
        return result
