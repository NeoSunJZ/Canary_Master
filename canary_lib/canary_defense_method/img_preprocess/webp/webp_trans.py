import numpy as np
import torch
from torchvision.transforms import ToTensor

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
from PIL import Image

from canary_sefi.core.component.component_decorator import SEFIComponent
from canary_sefi.core.component.component_enum import ComponentType, ComponentConfigHandlerType

sefi_component = SEFIComponent()


@sefi_component.trans_class(trans_name="webp")
@sefi_component.config_params_handler(handler_target=ComponentType.TRANS, name="webp",
                                      handler_type=ComponentConfigHandlerType.TRANS_CONFIG_PARAMS,
                                      use_default_handler=True,
                                      params={
                                      })
class Transform():
    def __init__(self, quality=60):
        self.quality = quality

    @sefi_component.trans(name="webp", is_inclass=True)
    def img_transform(self, imgs):
        result = []
        for img in imgs:
            if not torch.is_tensor(img):
                img = torch.from_numpy(img)
            result.append(self.webp_compression(img, self.quality))
        return result

    @staticmethod
    def webp_compression(im, quality):
        assert torch.is_tensor(im)
        im = im.cpu().detach().numpy()
        im = im.astype(np.uint8)
        im = Image.fromarray(im)
        savepath = BytesIO()
        im.save(savepath, "webp", quality=quality)
        im = Image.open(savepath)
        # im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im = ToTensor()(im)
        im = im.permute(1, 2, 0)
        im *= 255
        return im
