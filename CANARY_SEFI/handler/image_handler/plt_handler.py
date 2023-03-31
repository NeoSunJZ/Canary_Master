import base64
import os
from io import BytesIO

import numpy as np
from colorama import Fore
from matplotlib import figure
import matplotlib.pyplot as plt

from CANARY_SEFI.core.config.config_manager import config_manager
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from CANARY_SEFI.handler.image_handler.img_crm_hander import show_cam_on_image
from CANARY_SEFI.handler.image_handler.img_io_handler import pic_buffer_to_base64
from CANARY_SEFI.handler.image_handler.img_utils import get_img_diff
from CANARY_SEFI.task_manager import task_manager


def img_diff_fig_builder(original_img, adversarial_img):
    adversarial_img = np.clip(adversarial_img, 0, 255).astype(np.uint8)
    fig = figure.Figure(facecolor='w')

    ax_1 = fig.add_subplot(131)
    ax_1.set_title('Original')
    ax_1.imshow(original_img)
    ax_1.axis('off')

    ax_2 = fig.add_subplot(132)
    ax_2.set_title('Adversarial')
    ax_2.imshow(adversarial_img)
    ax_2.axis('off')

    ax_3 = fig.add_subplot(133)
    ax_3.set_title('Adversarial-Original')
    ax_3.imshow(get_img_diff(original_img, adversarial_img))
    ax_3.margins(0, 0)
    ax_3.axis('off')

    fig.tight_layout()
    return fig


def cam_diff_fig_builder(img, true_class_cams, inference_class_cams, info):
    model_name, atk_name, ori_img_id, adv_img_file_id, ori_label, ori_inference_label, adv_inference_label = info
    ori_img, adv_img = img
    fig = figure.Figure(facecolor='w')
    fig.suptitle('Gradient-weighted Class Activation Mapping\n Inference Model {}'.format(model_name), fontsize=12)

    ori_cam, adv_cam = true_class_cams

    ax_1 = fig.add_subplot(221)
    ax_1.set_title('Original Img (ID:{})\nTarget Category {}\n(GT Label)'.format(ori_img_id, ori_label), fontsize=8)
    ax_1.imshow(show_cam_on_image(ori_img / 255, ori_cam, True))
    ax_1.axis('off')

    ax_2 = fig.add_subplot(222)
    ax_2.set_title('Adv Img (ID:{})\n({})\nTarget Category {}\n(GT Label)'.format(adv_img_file_id, atk_name, ori_label), fontsize=8)
    ax_2.imshow(show_cam_on_image(adv_img / 255, adv_cam, True))
    ax_2.axis('off')

    ori_cam, adv_cam = inference_class_cams

    ax_3 = fig.add_subplot(223)
    ax_3.set_title('Original Img (ID:{})\nTarget Category {}\n(Inference Label)'.format(ori_img_id, ori_inference_label), fontsize=8)
    ax_3.imshow(show_cam_on_image(ori_img / 255, ori_cam, True))
    ax_3.axis('off')

    ax_4 = fig.add_subplot(224)
    ax_4.set_title('Adv Img (ID:{})\n({})\nTarget Category {}\n(Inference Label)'.format(adv_img_file_id, atk_name, adv_inference_label), fontsize=8)
    ax_4.imshow(show_cam_on_image(adv_img / 255, adv_cam, True))
    ax_4.axis('off')

    fig.tight_layout()
    return fig


def get_base64_by_fig(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches="tight", pad_inches=0.0)
    result = pic_buffer_to_base64(buffer)
    fig.clf()
    del buffer
    return result


def save_pic_by_fig(file_path, file_name, fig):
    full_path = task_manager.base_temp_path + "pic/" + file_path
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    fig.savefig(full_path + file_name + ".PNG", format='png', bbox_inches="tight", pad_inches=0.0)


def show_plt(fig):
    fig_plt = plt.figure()
    fig_plt.canvas.manager.canvas.figure = fig
    fig_plt.show()


def figure_show_handler(fig, file_path=None, file_name=None):
    action = config_manager.config.get("system", {}).get("save_fig_model", "no_action")
    if action == "no_action":
        return
    elif action == "save_img_file":
        assert file_path, file_name
        save_pic_by_fig(file_path, file_name, fig)
    elif action == "show_base64_on_terminal":
        reporter.console_log(get_base64_by_fig(fig), Fore.CYAN, type="Info")
    elif action == "show_on_window":
        show_plt(fig)

