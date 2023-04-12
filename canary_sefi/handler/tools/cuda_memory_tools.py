from colorama import Fore
from eagerpy import torch
from tqdm import tqdm

from canary_sefi.core.function.helper.realtime_reporter import reporter


def check_cuda_memory_alloc_status(empty_cache=False, show_status=False):
    if torch.cuda.is_available():
        if show_status:
            show_cuda_memory_alloc_status()
        if empty_cache:
            reporter.console_log("[CUDA-REPORT] CUDA Cache being released", Fore.CYAN, type="DEBUG", show_task=False, show_step_sequence=False)
            torch.cuda.empty_cache()
            if show_status:
                show_cuda_memory_alloc_status()
    else:
        reporter.console_log("[CUDA-REPORT] CUDA is not AVAILABLE", Fore.RED, show_task=False, show_step_sequence=False)


def show_cuda_memory_alloc_status():
    log = "[CUDA-REPORT] Current allocated memory {} MB, maximum allocated memory {} MB, reserved memory {} MB." \
        .format(torch.cuda.memory_allocated() / 1024 / 1024, torch.cuda.max_memory_allocated() / 1024 / 1024,
                torch.cuda.memory_reserved() / 1024 / 1024)
    reporter.console_log(log, Fore.GREEN, type="DEBUG", show_task=False, show_step_sequence=False)