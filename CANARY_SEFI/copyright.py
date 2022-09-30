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
    tqdm.write(" V20220929.beta E.9  \n")
    tqdm.write(" Design By Jiazheng Sun [BIT] \n")
    tqdm.write(" Contributor: Li Chen \ Rong Huang \ DongLi Tan \ Zhi Qu \ Chenxiao Xia \n")
    tqdm.write(Style.RESET_ALL)
