from .adv_gan import adv_gan_attacker
from .boundary_attack import boundary_attacker
from .gen_attack import gen_attacker
from .hop_skip_jump_attack import hsj_attacker
from .local_search_attack import ls_attacker
from .qfool import qfool_attacker
from .spatial_attack import spatial_attacker
from .spsa import sps_attacker
from .tremba import tremba_attacker

attacker_list = [
    adv_gan_attacker,
    boundary_attacker,
    gen_attacker,
    hsj_attacker,
    ls_attacker,
    sps_attacker,
    spatial_attacker,
    qfool_attacker,
    tremba_attacker
]
