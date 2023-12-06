import cv2
import numpy as np

from canary_sefi.handler.image_handler.img_utils import img_size_uniform_fix


def show_cam_on_image(img,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if img is not None:
        if np.max(img) > 1:
            raise Exception("The input image should np.float32 in the range [0, 1]")
        cam = heatmap + img
    else:
        cam = heatmap
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def get_cam_img(img, cam):
    if img is not None:
        img, cam = img_size_uniform_fix(img, cam, True)
    return show_cam_on_image(img / 255.0, cam, use_rgb=True)