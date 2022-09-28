from colorama import Fore
from eagerpy import torch
from tqdm import tqdm

from CANARY_SEFI.core.function.helper.realtime_reporter import reporter


def check_cuda_memory_alloc_status(empty_cache=False):
    if torch.cuda.is_available():
        show_cuda_memory_alloc_status()
        if empty_cache:
            reporter.console_log("[CUDA-REPORT] 正在释放CUDA缓存", Fore.CYAN, type="DEBUG", show_batch=True, show_step_sequence=True)
            torch.cuda.empty_cache()
            show_cuda_memory_alloc_status()
    else:
        reporter.console_log("[CUDA-REPORT] CUDA未就绪", Fore.RED, show_batch=True, show_step_sequence=True)


def show_cuda_memory_alloc_status():
    log = "[CUDA-REPORT] 当前Torch.Tensor已分配显存 {} MB, 最大分配的显存 {} MB, 当前进程所分配的显存缓冲区 {} MB" \
        .format(torch.cuda.memory_allocated() / 1024 / 1024, torch.cuda.max_memory_allocated() / 1024 / 1024,
                torch.cuda.memory_reserved() / 1024 / 1024)
    reporter.console_log(log, Fore.GREEN, type="DEBUG", show_batch=True, show_step_sequence=True)