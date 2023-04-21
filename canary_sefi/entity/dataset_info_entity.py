from enum import Enum

from canary_sefi.core.config.config_manager import config_manager


class DatasetType(Enum):
    NORMAL = "NORMAL"
    NORMAL_VAL = "NORMAL"
    NORMAL_TRAIN = "NORMAL"
    NORMAL_TEST = "NORMAL"
    ADVERSARIAL_EXAMPLE_IMG = "ADVERSARIAL_EXAMPLE_IMG"
    ADVERSARIAL_EXAMPLE_RAW_DATA = "ADVERSARIAL_EXAMPLE_RAW_DATA"
    TRANSFORM_IMG = "TRANSFORM_IMG"
    TRANSFORM_RAW_DATA = "TRANSFORM_RAW_DATA"


class DatasetInfo(object):
    def __init__(self, dataset_name, dataset_extra_info=None, dataset_type=DatasetType.NORMAL,
                 dataset_seed=None, dataset_size=None, img_cursor_list=None):
        if dataset_extra_info is None:
            dataset_extra_info = {}

        if not isinstance(dataset_type, DatasetType):
            if dataset_type is None:
                dataset_type = DatasetType.NORMAL
            if dataset_type == "VAL":
                dataset_type = DatasetType.NORMAL_VAL
            elif dataset_type == "TRAIN":
                dataset_type = DatasetType.NORMAL_TRAIN
            elif dataset_type == "TEST":
                dataset_type = DatasetType.NORMAL_TEST
            elif dataset_type == "ADV_IMG":
                dataset_type = DatasetType.ADVERSARIAL_EXAMPLE_IMG
            elif dataset_type == "ADV_RAW_DATA":
                dataset_type = DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA
            elif dataset_type == "TRANSFORM_IMG":
                dataset_type = DatasetType.TRANSFORM_IMG
            elif dataset_type == "TRANSFORM_RAW_DATA":
                dataset_type = DatasetType.TRANSFORM_RAW_DATA
            else:
                raise Exception("[SEFI] DatasetConfigError: Dataset Type must be an enum type DatasetType "
                                "or string 'VAL', 'TRAIN', 'TEST', 'ADV_IMG', 'ADV_RAW_DATA', 'TRANSFORM_IMG' or 'TRANSFORM_RAW_DATA'!")

        if dataset_name is None and dataset_type.value == "NORMAL":
            raise Exception("[SEFI] DatasetConfigError: Dataset Name must be specified!")

        if img_cursor_list is None and (dataset_type is DatasetType.ADVERSARIAL_EXAMPLE_IMG or dataset_type is DatasetType.ADVERSARIAL_EXAMPLE_RAW_DATA):
            raise Exception("[SEFI] DatasetConfigError: Adv Examples Dataset Img Cursor List must be specified!")

        self.dataset_name = dataset_name
        self.dataset_type = dataset_type

        # 数据集存储路径
        if dataset_type.value == "NORMAL":
            self.dataset_path = dataset_extra_info.get('path', config_manager.config.get("datasetPath") +
                                                        dataset_extra_info.get('folder', dataset_name))
            self.n_classes = dataset_extra_info.get('n_classes', None)
        elif dataset_type.value.find("ADV") != -1:
            self.dataset_path = None
            self.n_classes = None

        self.dataset_seed = dataset_seed
        self.dataset_size = dataset_size
        # 是否灰度
        self.is_gray = dataset_extra_info.get('is_gray', False)
        self.img_cursor_list = img_cursor_list
        if img_cursor_list is not None:
            self.dataset_size = len(img_cursor_list)
