name = "VMIM"
object_list = ["VMI_FGSM"]
config = {
    "attacker_config": {
        "VMI_FGSM": {
            "clip_min": 0,
            "clip_max": 1,
            "epsilon": 16 / 255,
            "T": 100,
            "attack_type": "UNTARGETED",
            "tlabel": None
        }
    }
}
