import base64
from io import BytesIO

from matplotlib import pyplot as plt

from CANARY_SEFI.handler.image_handler.img_crm_hander import show_cam_on_image
from CANARY_SEFI.handler.image_handler.img_utils import get_img_diff


def img_diff_plt_builder(original_img, adversarial_img):
    figure = plt.figure(facecolor='w')

    ax_1 = figure.add_subplot(131)
    ax_1.set_title('Original')
    ax_1.imshow(original_img)
    ax_1.axis('off')

    ax_2 = figure.add_subplot(132)
    ax_2.set_title('Adversarial')
    ax_2.imshow(adversarial_img)
    ax_2.axis('off')

    ax_3 = figure.add_subplot(133)
    ax_3.set_title('Adversarial-Original')
    ax_3.imshow(get_img_diff(original_img, adversarial_img), cmap=plt.cm.gray)
    ax_3.margins(0, 0)
    ax_3.axis('off')

    figure.tight_layout()
    return figure


def cam_diff_plt_builder(original_img, adversarial_img, ori_cam, adv_cam, title=None):
    figure = plt.figure(facecolor='w')
    figure.suptitle(title)

    ax_1 = figure.add_subplot(131)
    ax_1.set_title('Original CAM')
    ax_1.imshow(show_cam_on_image(original_img / 255, ori_cam, True))
    ax_1.axis('off')

    ax_2 = figure.add_subplot(132)
    ax_2.set_title('Adversarial CAM')
    ax_2.imshow(show_cam_on_image(adversarial_img / 255, adv_cam, True))
    ax_2.axis('off')

    figure.tight_layout()
    return figure


def get_base64_by_plt(figure):
    f = figure.gcf()
    buffer = BytesIO()
    f.savefig(buffer, format='png', bbox_inches="tight", pad_inches=0.0)
    return "data:image/png;base64," + str(base64.b64encode(buffer.getvalue()), "utf-8")


def show_plt(figure):
    figure.show()