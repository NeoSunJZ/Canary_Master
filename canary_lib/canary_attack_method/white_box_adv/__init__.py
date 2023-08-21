from .cw import *
from .deepfool import *
from .ead import *
from .fgsm_family import *
from .jsma import *
from .l_bfgs import *
from .ssah import *
from .uap import *
from .deep_fusing import *

attacker_list = [
    cw_attacker,
    deepfool_attacker,
    ead_attacker,
    fgm_attacker,
    fgsm_attacker,
    i_fgsm_attacker,
    mi_fgsm_attacker,
    ni_fgsm_attacker,
    mi_fgsm_lpips,
    pgd_attacker,
    si_fgsm_attacker,
    v_mi_fgsm_attacker,
    jsma_attacker,
    l_bfgs_attacker,
    ssah_attacker,
    uap_attacker,
    deep_fusing_attacker
]
