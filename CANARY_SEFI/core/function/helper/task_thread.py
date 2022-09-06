from concurrent.futures import ThreadPoolExecutor


class SingleTaskThread:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers = 1)
        self.execute_task_list = {}
        self.stop_flag = False

    def execute_task(self, task_id, *args, **kwargs):
        for task_id in self.execute_task_list:
            if not self.execute_task_list[task_id].done():
                raise RuntimeError("Other tasks running")
        self.execute_task_list[task_id] = self.executor.submit(*args, **kwargs)

    def stop_task(self, task_id):
        task = self.execute_task_list.get(task_id, None)
        if task is not None and not task.done():
            self.stop_flag = True

    def sys_check_task_status(self):
        if self.stop_flag:
            self.stop_flag = False
            raise Exception("STOP SIGNAL")


task_thread = SingleTaskThread()
