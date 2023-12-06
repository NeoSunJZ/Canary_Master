from colorama import Fore, Style
from tqdm import tqdm


def get_system_version():
    return "2.1.2"


def get_system_build_date():
    return "20231206"


def get_system_build_times():
    return "1"

def version():
    print_logo()


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
    tqdm.write(" Security Evaluation Fast Integration ")
    tqdm.write(" V{}.beta 内部版本[{}][{}] \n"
               .format(get_system_version(), get_system_build_date(), get_system_build_times()))

    tqdm.write(" Copyright © 2023 BIT (Jiazheng Sun etc.) ")
    tqdm.write(" Design By Jiazheng Sun")
    tqdm.write(" Contributor: Chenxiao Xia \ Da Zhang \ Wenqi Xiong \ Jing Liu \ Shujie Hu etc.")
    tqdm.write("      See     https://neosunjz.github.io/Canary/about              \n")
    tqdm.write(" Licensed under the Apache License, Version 2.0 (the 'License')      ")
    tqdm.write(" you may not use this file except in compliance with the License.    ")
    tqdm.write(" You may obtain a copy of the License at ")
    tqdm.write("             http://www.apache.org/licenses/LICENSE-2.0              ")
    tqdm.write(" Unless required by applicable law or agreed to in writing, software ")
    tqdm.write(" distributed under the License is distributed on an 'AS IS' BASIS,   ")
    tqdm.write(" WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.")
    tqdm.write(" See the License for the specific language governing permissions and  ")
    tqdm.write(" limitations under the License.")
    tqdm.write(Style.RESET_ALL)