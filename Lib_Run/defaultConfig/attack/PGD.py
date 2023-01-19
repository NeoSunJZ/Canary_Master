name = "PGD"
object_list = ["PGD"]
config = {
    "attacker_config": {
        "PGD": {
            "clip_min": 0,
            "clip_max": 1,
            "eps_iter": 2.5 * ((16 / 255) / 100),
            "nb_iter": 100,
            "attack_type": "UNTARGETED",
            "epsilon": 16 / 255,
        }
    }
}
