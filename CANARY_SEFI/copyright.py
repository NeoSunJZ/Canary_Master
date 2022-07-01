from colorama import Fore, Style
from tqdm import tqdm


def print_logo():
    tqdm.write(Fore.GREEN + "\n")
    tqdm.write(" POWERED BY")
    tqdm.write("\n")
    tqdm.write(" ███████╗ ███████╗ ███████╗ ██╗")
    tqdm.write(" ██╔════╝ ██╔════╝ ██╔════╝ ██║")
    tqdm.write(" ███████╗ █████╗   █████╗   ██║")
    tqdm.write(" ╚════██║ ██╔══╝   ██╔══╝   ██║")
    tqdm.write(" ███████║ ███████╗ ██║      ██║")
    tqdm.write(" ╚══════╝ ╚══════╝ ╚═╝      ╚═╝")
    tqdm.write("\n")
    tqdm.write(" Security Evaluation Fast Integration Framework")
    tqdm.write(" V20220630.beta C.2  \n")
    tqdm.write(" Design By Jiazheng Sun [BIT] \n")
    tqdm.write(Style.RESET_ALL)
