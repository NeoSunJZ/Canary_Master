import base64
from io import BytesIO

import numpy as np
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
    ax_3.imshow(get_img_diff(original_img, np.clip(adversarial_img, 0, 255).astype(np.int8)), cmap=plt.cm.gray)
    ax_3.margins(0, 0)
    ax_3.axis('off')

    figure.tight_layout()
    return figure


def cam_diff_plt_builder(img, true_class_cams, inference_class_cams, info):
    model_name, atk_name, ori_img_id, adv_img_file_id, ori_label, ori_inference_label, adv_inference_label = info
    ori_img, adv_img = img
    figure = plt.figure(facecolor='w')
    figure.suptitle('Gradient-weighted Class Activation Mapping\n Inference Model {}'.format(model_name), fontsize=12)

    ori_cam, adv_cam = true_class_cams

    ax_1 = figure.add_subplot(221)
    ax_1.set_title('Original Img (ID:{})\nTarget Category {}\n(GT Label)'.format(ori_img_id, ori_label), fontsize=8)
    ax_1.imshow(show_cam_on_image(ori_img / 255, ori_cam, True))
    ax_1.axis('off')

    ax_2 = figure.add_subplot(222)
    ax_2.set_title('Adv Img (ID:{})\n({})\nTarget Category {}\n(GT Label)'.format(adv_img_file_id, atk_name, ori_label), fontsize=8)
    ax_2.imshow(show_cam_on_image(adv_img / 255, adv_cam, True))
    ax_2.axis('off')

    ori_cam, adv_cam = inference_class_cams

    ax_3 = figure.add_subplot(223)
    ax_3.set_title('Original Img (ID:{})\nTarget Category {}\n(Inference Label)'.format(ori_img_id, ori_inference_label), fontsize=8)
    ax_3.imshow(show_cam_on_image(ori_img / 255, ori_cam, True))
    ax_3.axis('off')

    ax_4 = figure.add_subplot(224)
    ax_4.set_title('Adv Img (ID:{})\n({})\nTarget Category {}\n(Inference Label)'.format(adv_img_file_id, atk_name, adv_inference_label), fontsize=8)
    ax_4.imshow(show_cam_on_image(adv_img / 255, adv_cam, True))
    ax_4.axis('off')

    figure.tight_layout()
    return figure


def get_base64_by_plt(figure):
    f = figure.gcf()
    buffer = BytesIO()
    f.savefig(buffer, format='png', bbox_inches="tight", pad_inches=0.0)
    return "data:image/png;base64," + str(base64.b64encode(buffer.getvalue()), "utf-8")


def show_plt(figure):
    figure.show()