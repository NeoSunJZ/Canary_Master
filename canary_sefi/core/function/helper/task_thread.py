from concurrent.futures import ThreadPoolExecutor


class SingleTaskThread:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers = 1)
        self.execute_task_list = {}
        self.stop_flag = False
        self.watch_dog = 10

    def execute_task(self, task_token, *args, **kwargs):
        for index in self.execute_task_list:
            if not self.execute_task_list[index].done():
                raise RuntimeError("[ Logic Error ] Other tasks running")
        self.execute_task_list[task_token] = self.executor.submit(*args, **kwargs)

    def stop_task(self, task_token):
        task = self.execute_task_list.get(task_token, None)
        if task is not None and not task.done():
            self.stop_flag = True

    def sys_check_task_status(self):
        if self.stop_flag:
            self.stop_flag = False
            raise Exception("[ STOP SIGNAL ]")


task_thread = SingleTaskThread()
monitor_thread = SingleTaskThread()
