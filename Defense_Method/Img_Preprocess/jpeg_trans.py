import numpy as np
import torch
from torchvision.transforms import ToPILImage, ToTensor

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
from PIL import Image

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.trans_class(trans_name="jpeg")
@sefi_component.config_params_handler(handler_target=ComponentType.TRANS, name="jpeg",
                                      handler_type=ComponentConfigHandlerType.TRANS_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                      })
class Transform():
    def __init__(self, args={}):
        self.args = args

    @sefi_component.trans(name="jpeg", is_inclass=True)
    def img_transform(self, img):
        if not torch.is_tensor(img):
            img = torch.from_numpy(img)
        return _jpeg_compression(img)


def _jpeg_compression(im):
    assert torch.is_tensor(im)
    im = im.numpy()
    im = im.astype(np.uint8)
    im = Image.fromarray(im)
    # 图片展示
    # im.show()
    # im = ToPILImage()(im)
    savepath = BytesIO()
    im.save(savepath, "JPEG", quality=75)
    im = Image.open(savepath)
    im = ToTensor()(im)
    im = im.permute(1, 2, 0)
    im *= 255
    return im
