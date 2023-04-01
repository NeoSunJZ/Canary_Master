from colorama import Fore, Style
from tqdm import tqdm


def get_system_version():
    return "2.0.1"

def print_logo(color=Fore.GREEN):
    tqdm.write(color + "\n")
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
    tqdm.write(" V20230328.beta 内部代号[1]["+get_system_version()+"] \n")

    tqdm.write(" Copyright 2022 BIT ")
    tqdm.write(" Design By Jiazheng Sun")
    tqdm.write(" Contributor: Li Chen \ Chenxiao Xia \ Zhang DA \ Xiong Wenqi \ Rong Huang \ DongLi Tan \ Zhi Qu \n")
    tqdm.write(" Licensed under the Apache License, Version 2.0 (the 'License')      ")
    tqdm.write(" you may not use this file except in compliance with the License.    ")
    tqdm.write(" You may obtain a copy of the License at ")
    tqdm.write("          http://www.apache.org/licenses/LICENSE-2.0                 ")
    tqdm.write(" Unless required by applicable law or agreed to in writing, software ")
    tqdm.write(" distributed under the License is distributed on an 'AS IS' BASIS,   ")
    tqdm.write(" WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.")
    tqdm.write(" See the License for the specific language governing permissions and  ")
    tqdm.write(" limitations under the License.")
    tqdm.write(Style.RESET_ALL)