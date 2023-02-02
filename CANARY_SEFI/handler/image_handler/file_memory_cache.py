from CANARY_SEFI.core.config.config_manager import config_manager
from CANARY_SEFI.core.function.basic.dataset_function import dataset_single_image_reader
from CANARY_SEFI.evaluator.logger.img_file_info_handler import find_img_log_by_id


# 缓存增强
class FileMemoryCache:
    def __init__(self, ):
        self.ori_img_list = {}

    def get_ori_img(self, dataset_info, ori_img_id):
        # 若禁用内存缓存增强
        if not config_manager.config.get("system", {}).get("use_file_memory_cache", False):
            ori_img_log = find_img_log_by_id(ori_img_id)
            return dataset_single_image_reader(dataset_info, ori_img_cursor=ori_img_log['ori_img_cursor'])

        ori_img_data = self.ori_img_list.get(ori_img_id, None)
        if ori_img_data is None:
            ori_img_log = find_img_log_by_id(ori_img_id)
            ori_img, ori_label = dataset_single_image_reader(dataset_info, ori_img_cursor=ori_img_log['ori_img_cursor'])
            # 存入临时缓存
            self.ori_img_list[ori_img_id] = {
                "ori_img": ori_img,
                "ori_label": ori_label,
            }
        else:
            ori_img = ori_img_data.get("ori_img")
            ori_label = ori_img_data.get("ori_label")
        return ori_img, ori_label


file_memory_cache = FileMemoryCache()
