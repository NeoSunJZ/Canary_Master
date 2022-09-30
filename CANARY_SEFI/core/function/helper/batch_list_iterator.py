from CANARY_SEFI.handler.tools.cuda_memory_tools import check_cuda_memory_alloc_status


class BatchListIterator:

    @staticmethod
    def model_list_iterator(model_list, model_config, img_proc_config, function):
        for model_name in model_list:
            model_args = model_config.get(model_name, {})
            img_proc_args = img_proc_config.get(model_name, {})

            function(model_name, model_args, img_proc_args)

            check_cuda_memory_alloc_status(empty_cache=True)

    @staticmethod
    def attack_list_iterator(attacker_list, attacker_config, model_config, img_proc_config, function):
        for atk_name in attacker_list:
            atk_args = attacker_config.get(atk_name, {})

            def _function(model_name, model_args, img_proc_args):
                function(atk_name, atk_args, model_name, model_args, img_proc_args)
            model_list = attacker_list[atk_name]

            BatchListIterator.model_list_iterator(model_list, model_config, img_proc_config, _function)

    @staticmethod
    def get_singleton_model_list(attacker_list):
        model_name_set = set()
        for atk_name in attacker_list:
            for model_name in attacker_list[atk_name]:
                model_name_set.add(model_name)
        return list(model_name_set)
