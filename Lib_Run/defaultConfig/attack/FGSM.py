name = "FGSM"
object_list = ["FGSM"]
config = {
    "attacker_config": {
        "FGSM": {
            "clip_min": 0,
            "clip_max": 1,
            "epsilon": 16 / 255,
        }
    }
}
