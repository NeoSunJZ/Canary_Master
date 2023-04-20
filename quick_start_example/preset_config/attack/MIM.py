name = "MIM"
object_list = ["MI_FGSM"]
config = {
    "attacker_config": {
        "MI_FGSM": {
            "clip_min": 0,
            "clip_max": 1,
            "epsilon": 1 / 255,
            "T": 100,
            "attack_type": "UNTARGETED",
            "tlabel": None
        }
    }
}
