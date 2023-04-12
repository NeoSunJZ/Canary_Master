from colorama import Fore, Style
from tqdm import tqdm


def get_system_version():
    return "1.0.1"


def print_logo(color=Fore.BLUE):
    tqdm.write(color)
    tqdm.write(r"    ________________   __   _ __  ")
    tqdm.write(r"   / __/ __/ __/  _/  / /  (_) /  ")
    tqdm.write(r"  _\ \/ _// _/_/ /   / /__/ / _ \ ")
    tqdm.write(r" /___/___/_/ /___/  /____/_/_.__/ ")
    tqdm.write(" Security Evaluation Fast Integration Framework [Library]")
    tqdm.write(" V20230411.beta 内部代号[1]["+get_system_version()+"] \n")
    tqdm.write(" Design By Jiazheng Sun")
    tqdm.write(" Contributor: Li Chen \ Chenxiao Xia \ Zhang DA \ Xiong Wenqi \ Rong Huang \ DongLi Tan \ Zhi Qu")
    tqdm.write(" Referenced Component Source: ")
    tqdm.write(" - Model -")
    tqdm.write(" - Github Project: Torchvision [https://github.com/pytorch/pytorch]")
    tqdm.write(" - Github Project: PyTorch_CIFAR10 [https://github.com/huyvnphan/PyTorch_CIFAR10]")
    tqdm.write(" - Github Project: CNN-for-Fashion-MNIST [https://github.com/wzyjsha-00/CNN-for-Fashion-MNIST]")
    tqdm.write(" - Github Project: PyTorch CIFAR Models [https://github.com/chenyaofo/pytorch-cifar-models]")
    tqdm.write("\n - Attack Method - ")
    tqdm.write(" - Github Project: CleverHans [https://github.com/cleverhans-lab/cleverhans]")
    tqdm.write(" - Github Project: FoolBox [https://github.com/bethgelab/foolbox]")
    tqdm.write("\n -- SEFI LIBRARY LOADED --")
    tqdm.write(Style.RESET_ALL)