import torch

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
from Defense_Method.Img_Preprocess.quilting.quilting import quilting
from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.trans_class(trans_name="quilting")
@sefi_component.config_params_handler(handler_target=ComponentType.TRANS, name="quilting",
                                      handler_type=ComponentConfigHandlerType.TRANS_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                      })
class Transform():
    def __init__(self, quilting_size=8, kemeans=5):
        self.quilting_size = quilting_size
        self.kemeans = kemeans

    @sefi_component.trans(name="quilting", is_inclass=True)
    def img_transform(self, imgs):
        result = []
        for img in imgs:
            if not torch.is_tensor(img):
                img = torch.from_numpy(img)
            result.append(quilting(img, self.quilting_size, self.kemeans))
        return result

