from canary_sefi.task_manager import task_manager
from canary_sefi.handler.tools.cuda_memory_tools import check_cuda_memory_alloc_status


class BatchListIterator:

    @staticmethod
    def model_list_iterator(model_list, model_config, img_proc_config, function):
        for model_name in model_list:
            model_args = {} if model_config is None else model_config.get(model_name, {})
            img_proc_args = {} if img_proc_config is None else img_proc_config.get(model_name, {})

            function(model_name, model_args, img_proc_args, run_device=task_manager.run_device)

            check_cuda_memory_alloc_status(empty_cache=True)

    @staticmethod
    def attack_list_iterator(attacker_list, attacker_config, model_config, img_proc_config, function):
        for atk_name in attacker_list:
            atk_args = attacker_config.get(atk_name, {})

            def _function(model_name, model_args, img_proc_args, run_device):
                function(atk_name, atk_args, model_name, model_args, img_proc_args, run_device)
            model_list = attacker_list[atk_name]

            BatchListIterator.model_list_iterator(model_list, model_config, img_proc_config, _function)

    @staticmethod
    def defense_list_iterator(defense_list, defense_config, model_config, img_proc_config, function):
        for def_name in defense_list:
            def_args = defense_config.get(def_name, {})

            def _function(model_name, model_args, img_proc_args, run_device):
                function(def_name, def_args, model_name, model_args, img_proc_args, run_device)
            model_list = defense_list[def_name]

            BatchListIterator.model_list_iterator(model_list, model_config, img_proc_config, _function)

    @staticmethod
    def get_singleton_model_list(attacker_list):
        model_name_set = set()
        for atk_name in attacker_list:
            for model_name in attacker_list[atk_name]:
                model_name_set.add(model_name)
        return list(model_name_set)
