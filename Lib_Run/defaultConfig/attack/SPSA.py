import numpy as np

name = "SPSA"
object_list = ["SPSA"]
config = {
    "attacker_config": {
        "SPSA": {
            "clip_min": 0,
            "clip_max": 1,
            "epsilon": 16 / 255,
            "norm": np.inf,
            "attack_type": 'UNTARGETED',
            "tlabel": -1,
            "nb_iter": 100,
            "early_stop_loss_threshold": None,
            "learning_rate": 0.01,
            "delta": 0.01,
            "spsa_samples": 100,
            "spsa_iters": 1,
        }
    }
}