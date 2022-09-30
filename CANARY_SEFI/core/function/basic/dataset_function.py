import cv2

from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from CANARY_SEFI.core.config.config_manager import config_manager
from CANARY_SEFI.core.function.helper.system_log import global_system_log
from CANARY_SEFI.core.function.helper.task_thread import task_thread
from CANARY_SEFI.entity.dataset_info_entity import DatasetType
from CANARY_SEFI.evaluator.logger.adv_example_file_info_handler import find_adv_example_file_log_by_id
from CANARY_SEFI.evaluator.logger.img_file_info_handler import add_img_log
from CANARY_SEFI.handler.image_handler.img_io_handler import get_pic_nparray_from_dataset


def default_image_getter(img_name, dataset_path, dataset_seed, dataset_size=None, with_label=False):
    # 默认getter仅支持全文件名读取
    if with_label:
        return get_pic_nparray_from_dataset(dataset_path, img_name), None
    else:
        return get_pic_nparray_from_dataset(dataset_path, img_name)


def init_dataset_image_reader(dataset_info):
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
    return image_getter, is_default_image_getter, dataset_path


# 两种读入方式任选其一:
# 传入dataset_size，则根据dataset_size划分数据集子集，并读入全部的子集（特别的，全部读入则传入数据集原始大小即可）
# 传入传入img_list，则根据img_list读入指定图片
# 如果没有自行指定image getter，则默认的image getter只支持第二种，且传入的img_list必须是图片文件名
def dataset_image_reader(iterator, dataset_info, completed_num=0):

    # 对抗样本数据集读入
    if dataset_info.dataset_type != DatasetType.NORMAL:
        adv_dataset_image_reader(iterator, dataset_info)
        return

    image_getter, is_default_image_getter, dataset_path = init_dataset_image_reader(dataset_info)

    def handler_img(_img, _img_label, _img_cursor):
        # 检查尺寸
        _img = limit_img_size(_img)
        # 写入日志
        img_log_id = add_img_log(_img_label, _img_cursor)
        # 调用迭代器,传入图片
        iterator(_img, img_log_id, _img_label)
        # 完成数量增加
        global_system_log.update_completed_num(1)

    if dataset_info.dataset_size is not None:
        if not is_default_image_getter:
            for img_cursor in range(completed_num, dataset_info.dataset_size):
                img, img_label = image_getter(img_cursor, dataset_path, dataset_info.dataset_seed, dataset_info.dataset_size,
                                              with_label=True)
                handler_img(img, img_label, img_cursor)
        else:
            raise Exception("The default dataset image getter only supports reading by specifying the item list")
    if dataset_info.img_cursor_list is not None:
        for img_cursor in dataset_info.img_name_list[completed_num:]:
            img, img_label = image_getter(img_cursor, dataset_path, dataset_seed=None, dataset_size=None, with_label=True)
            handler_img(img, img_label, img_cursor)


def dataset_single_image_reader(dataset_info, ori_img_cursor=0):
    image_getter, is_default_image_getter, dataset_path = init_dataset_image_reader(dataset_info)
    if dataset_info.dataset_size is not None:
        if not is_default_image_getter:
            img, img_label = image_getter(ori_img_cursor, dataset_path, dataset_info.dataset_seed, dataset_info.dataset_size,
                                            with_label=True)
            # 检查尺寸
            img = limit_img_size(img)
            return img
        else:
            raise Exception("The default dataset image getter only supports reading by specifying the item list")


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
    adv_img_type = dataset_info.dataset_type
    adv_img_cursor_list = dataset_info.img_cursor_list

    for i in range(len(adv_img_cursor_list)):
        adv_file_log = find_adv_example_file_log_by_id(adv_img_cursor_list[i])

        img = adv_dataset_single_image_reader(adv_file_log, adv_img_type)
        iterator(img, adv_img_cursor_list[i], None)
        global_system_log.update_completed_num(1)


def adv_dataset_single_image_reader(adv_file_log, adv_img_type):
    adv_dataset_temp_path = config_manager.config.get("temp", "Dataset_Temp/")
    adv_file_path = adv_dataset_temp_path + adv_file_log["batch_id"] + "/"
    if adv_img_type == DatasetType.ADVERSARIAL_EXAMPLE_IMG:
        img = get_pic_nparray_from_dataset(adv_file_path, adv_file_log["adv_img_filename"], is_numpy_array_file=False)
    elif adv_img_type == DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA:
        img = get_pic_nparray_from_dataset(adv_file_path, adv_file_log["adv_raw_nparray_filename"],
                                           is_numpy_array_file=True)
    else:
        raise RuntimeError("[ Logic Error ] [ READ DATASET IMG ] Wrong dataset type!")
    return img
