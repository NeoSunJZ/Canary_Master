from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from CANARY_SEFI.core.config.config_manager import config_manager
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

def dataset_image_reader(batch_token, iterator, dataset_name, dataset_size=None, img_list=None):
    # 对抗样本数据集读入
    if dataset_name == "ADVERSARIAL_EXAMPLE":
        adv_dataset_image_reader(batch_token, iterator)
        return
    # 若数据集自行指定了getter，则不使用我们自己的getter
    # 默认getter仅支持全文件名读取
    dataset_component = SEFI_component_manager.dataset_list.get(dataset_name)
    image_getter = default_image_getter

    is_default_image_getter = True
    if dataset_component is not None:
        custom_image_getter = dataset_component.get("dataset_getter_handler")
        if custom_image_getter is not None:
            image_getter = custom_image_getter
            is_default_image_getter = False

    dataset_path = config_manager.config.get("dataset", {}).get(dataset_name, {}).get("path", None)

    if dataset_size is not None and img_list is not None:
        raise Exception("Only one of the two methods can be selected: 'specify dataset subset item' and 'specify dataset subset size'!")
    elif dataset_size is None and img_list is None:
        raise Exception("Either 'specify dataset subset item' or 'specify dataset subset size' must be specified!")
    if dataset_size is not None:
        if not is_default_image_getter:
            for i in range(dataset_size):
                img, img_label = image_getter(i, dataset_path, batch_token, dataset_size, with_label=True)
                iterator(img, i, img_label)
        else:
            raise Exception("The default dataset image getter only supports reading by specifying the item list")
    if img_list is not None:
        for i in img_list:
            img, img_label = image_getter(i, dataset_path, seed=None, dataset_size=None, with_label=True)
            iterator(img, i, img_label)


def adv_dataset_image_reader(batch_token, iterator):
    # 消除batch_token的_ADV
    batch_token = batch_token.replace('_ADV', '')
    # 读入数据集位置
    adv_dataset_temp_path = config_manager.config.get("temp", "Dataset_Temp/")
    # 读入日志
    log = get_log_data_to_file("attack_log_" + batch_token + ".csv")

    for i in range(len(log)):
        img = default_image_getter(log["atk_adv_name"][i], adv_dataset_temp_path + batch_token + "/", None, None, False)
        iterator(img, i, log["ori_label"][i])