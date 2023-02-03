name = "LSA"
object_list = ["LSA"]
config = {
    "attacker_config": {
        "LSA": {
            "max_iter": 80,
            "clip_min": 0,
            "clip_max": 1,
            "attack_type": 'UNTARGETED',
            "tlabel": -1,
            "r": 1.5,
            "p": 0.25,
            "d": 5,
            "t": 5,
            "mini_batch": 16
        }
    }
}
