name = "DeepFool"
object_list = ["DeepFool"]
config = {
    "attacker_config": {
        "DeepFool": {
            "pixel_min": 0,
            "pixel_max": 1,
            "p": "l-2",
            "num_classes": 1000,
            "overshoot": 0.02,
            "max_iter": 100,
            "candidates": 20,
            "attack_type": "UNTARGETED"
        }
    }
}
