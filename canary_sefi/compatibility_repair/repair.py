from colorama import Fore

from canary_sefi.compatibility_repair.repair_v2_0_3 import repair_v2_0_3
from canary_sefi.compatibility_repair.repair_v2_0_0 import repair_v2_0_0
from canary_sefi.copyright import get_system_version

repair_program_list = {
    "2.0.0": repair_v2_0_0,
    "2.0.3": repair_v2_0_3,
}


def compatibility_repair(data_version, database_list):
    skip = True
    for k, v in repair_program_list.items():
        if skip is True and (data_version == k or data_version is None) and data_version != get_system_version():
            print("DATA RECORD VERSION is LOWER than the current version!".format(k))
            skip = False

        if not skip:
            print("Repairing database compatibility... [Ver.{}]".format(k))
            v(database_list)
    if skip:
        print("DATA RECORD VERSION is UP-TO-DATA".format(k))