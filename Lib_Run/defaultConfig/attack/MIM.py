name = "MIM"
object_list = ["MI-FGSM"]
config = {
    "attacker_config": {
        "MI-FGSM": {
            "pixel_min": 0,
            "pixel_max": 1,
            "alpha": 2.5 * ((1 / 255) / 100),
            "epsilon": 1 / 255,
            "T": 100,
            "attack_type": "UNTARGETED",
            "tlabel": None
        }
    }
}
