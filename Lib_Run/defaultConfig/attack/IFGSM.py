import numpy as np

name = "I_FGSM"
object_list = ["I_FGSM"]
config = {
    "attacker_config": {
        "I_FGSM": {
            "clip_min": 0,
            "clip_max": 1,
            "eps_iter": 2.5 * ((16 / 255) / 100),
            "nb_iter": 100,
            "norm": np.inf,
            "attack_type": "UNTARGETED",
            "epsilon": 16 / 255,
        }
    }
}
