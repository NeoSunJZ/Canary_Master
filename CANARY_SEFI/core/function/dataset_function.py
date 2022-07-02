import cv2

from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from CANARY_SEFI.core.config.config_manager import config_manager
from CANARY_SEFI.entity.dataset_info_entity import DatasetType
from CANARY_SEFI.evaluator.logger.adv_logger import find_adv_log
from CANARY_SEFI.evaluator.logger.dataset_logger import add_img_log
from CANARY_SEFI.handler.csv_handler.csv_io_handler import get_log_data_to_file
from CANARY_SEFI.handler.image_handler.img_io_handler import get_pic_nparray_from_dataset


def default_image_getter(img_name, dataset_path, dataset_seed, dataset_size=None, with_label=False):
    # 默认getter仅支持全文件名读取
    if with_label:
        return get_pic_nparray_from_dataset(dataset_path, img_name), None
    else:
        return get_pic_nparray_from_dataset(dataset_path, img_name)


# 两种读入方式任选其一:
# 传入dataset_size，则根据dataset_size划分数据集子集，并读入全部的子集（特别的，全部读入则传入数据集原始大小即可）
# 传入传入img_list，则根据img_list读入指定图片
# 如果没有自行指定image getter，则默认的image getter只支持第二种，且传入的img_list必须是图片文件名

def dataset_image_reader(iterator, dataset_info):
    # 对抗样本数据集读入
    if dataset_info.dataset_type == DatasetType.ADVERSARIAL_EXAMPLE:
        adv_dataset_image_reader(iterator, dataset_info)
        return
    # 若数据集自行指定了getter，则不使用我们自己的getter
    # 默认getter仅支持全文件名读取
    dataset_component = SEFI_component_manager.dataset_list.get(dataset_info.dataset_name)
    image_getter = default_image_getter

    is_default_image_getter = True
    if dataset_component is not None:
        custom_image_getter = dataset_component.get("dataset_getter_handler")
        if custom_image_getter is not None:
            image_getter = custom_image_getter
            is_default_image_getter = False

    dataset_path = config_manager.config.get("dataset", {}).get(dataset_info.dataset_name, {}).get("path", None)

    if dataset_info.dataset_size is not None:
        if not is_default_image_getter:
            for img_cursor in range(dataset_info.dataset_size):
                img, img_label = image_getter(img_cursor, dataset_path, dataset_info.dataset_seed, dataset_info.dataset_size,
                                              with_label=True)

                # 检查尺寸
                img = limit_img_size(img)

                # 写入日志
                img_log_id = add_img_log(dataset_info.dataset_log_id, img_label, img_cursor)

                iterator(img, img_log_id, img_label)
        else:
            raise Exception("The default dataset image getter only supports reading by specifying the item list")
    if dataset_info.img_cursor_list is not None:
        for img_cursor in dataset_info.img_name_list:
            img, img_label = image_getter(img_cursor, dataset_path, dataset_seed=None, dataset_size=None, with_label=True)

            # 检查尺寸
            img = limit_img_size(img)

            # 写入日志
            img_log_id = add_img_log(dataset_info.dataset_log_id, img_label, img_cursor)

            iterator(img, img_log_id, img_label)


def limit_img_size(img):
    # 检查尺寸
    limited_img_size = config_manager.config.get("system", {}).get("limited_read_img_size", None)
    if limited_img_size is not None:
        height, width = img.shape[0], img.shape[1]
        if height >= limited_img_size and height >= width:  # 高度超限
            scale = height / limited_img_size
            width_size = int(width / scale)
            # 缩放之
            img = cv2.resize(img, (width_size, limited_img_size), interpolation=cv2.INTER_CUBIC)
        elif width >= limited_img_size and width >= height:  # 宽度超限
            scale = width / limited_img_size
            height_size = int(height / scale)
            # 缩放之
            img = cv2.resize(img, (limited_img_size, height_size), interpolation=cv2.INTER_CUBIC)
    return img


def adv_dataset_image_reader(iterator, dataset_info):
    adv_img_cursor_list = dataset_info.img_cursor_list
    # 读入数据集位置
    adv_dataset_temp_path = config_manager.config.get("temp", "Dataset_Temp/")
    for i in range(len(adv_img_cursor_list)):
        adv_log = find_adv_log(adv_img_cursor_list[i])[0]
        adv_img_id = adv_log[0]
        adv_batch = adv_log[1]
        adv_filename = adv_log[5]

        adv_file_path = adv_dataset_temp_path + adv_batch + "/"

        img = default_image_getter(adv_filename, adv_file_path, None, None, False)
        iterator(img, adv_img_id, None)
