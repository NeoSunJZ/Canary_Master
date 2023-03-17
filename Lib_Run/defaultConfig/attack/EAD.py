name = "EAD"
object_list = ["EAD"]
config = {
    "attacker_config": {
        "EAD": {
            "attack_type": "UNTARGETED",
            "tlabel": None,
            "clip_min": 0.,
            "clip_max": 1.,
            "kappa": 0,
            "init_const": 0.001,
            "lr": 0.02,
            "binary_search_steps": 5,
            "max_iters": 1000,
            "beta": 0.01,
            "EN": True,
            "num_classes": 1000,
        }
    }
}
