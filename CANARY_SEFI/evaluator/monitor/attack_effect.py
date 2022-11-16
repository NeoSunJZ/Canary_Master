import time


def time_cost_statistics(logger):
    def decorated(func):
        def wrapper(*args, **kwargs):
            t_start = time.perf_counter()
            result = func(*args, **kwargs)
            t_end = time.perf_counter()

            # 写入必要日志
            logger.cost_time = str(t_end - t_start)
            return result

        return wrapper
    return decorated


def model_query_statistics(logger, query_type):
    def hook_fn(module, grad_input, grad_output):
        logger.query_num[query_type] += 1
    return hook_fn