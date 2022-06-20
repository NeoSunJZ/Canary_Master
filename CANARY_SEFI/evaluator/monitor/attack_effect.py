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
