from enum import Enum


class DatasetType(Enum):
    NORMAL = "NORMAL"
    ADVERSARIAL_EXAMPLE_IMG = "ADVERSARIAL_EXAMPLE_IMG"
    ADVERSARIAL_EXAMPLE_RAW_DATA = "ADVERSARIAL_EXAMPLE_RAW_DATA"
    TRANSFORM_IMG = "TRANSFORM_IMG"
    TRANSFORM_RAW_DATA = "TRANSFORM_RAW_DATA"


class DatasetInfo(object):
    def __init__(self, dataset_name, dataset_seed=None, dataset_size=None, img_cursor_list=None):
        if (dataset_size is not None and dataset_seed is not None) and img_cursor_list is not None:
            raise Exception(
                "[ Config Error ] Only one of the two methods can be selected: 'specify dataset subset item' and 'specify dataset subset size'!")
        elif dataset_size is None and img_cursor_list is None:
            raise Exception("[ Config Error ] Either 'specify dataset subset item' or 'specify dataset subset size' must be specified!")

        self.dataset_name = dataset_name
        self.dataset_seed = dataset_seed
        self.dataset_size = dataset_size
        self.img_cursor_list = img_cursor_list

        if img_cursor_list is not None:
            self.dataset_size = len(img_cursor_list)

        self.dataset_type = DatasetType.NORMAL
        self.n_classes = None
