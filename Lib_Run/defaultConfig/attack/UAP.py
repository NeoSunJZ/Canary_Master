name = "UAP"
object_list = ["UAP"]
config = {
    "attacker_config": {
        "UAP": {
            "num_classes": 1000,
            "pixel_min": 0,
            "pixel_max": 1,
            "attack_type": "UNTARGETED",
            "xi": 4 / 255,
            "p": "l-inf",
            "delta": 0,
            "overshoot": 0.02,
            "max_iter_uni": 20,
            "max_iter_df": 1000
        }
    }
}
