import torch

from canary_lib.canary_defense_method.img_preprocess.MagNet.magnet import magnet_purify, get_defensive_model
from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()
trans_name = "magnet"


@sefi_component.trans_class(trans_name=trans_name)
@sefi_component.config_params_handler(handler_target=ComponentType.TRANS, name=trans_name,
                                      handler_type=ComponentConfigHandlerType.TRANS_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                      })
class Transform():
    def __init__(self):
        self.magnet = get_defensive_model(
            defensive_models_path="../canary_lib/canary_defense_method/img_preprocess/MagNet/",
            defensive_model_name='defensive-model-2', dataset_name='cifar-10', device='cuda')

    @sefi_component.trans(name=trans_name, is_inclass=True)
    def img_transform(self, imgs):
        result = []
        for img in imgs:
            if not torch.is_tensor(img):
                img = torch.from_numpy(img)
            result.append(magnet_purify(img.permute(2, 0, 1).unsqueeze(0).cuda()/255, self.magnet)[0].permute(1, 2, 0)*255)
        return result
