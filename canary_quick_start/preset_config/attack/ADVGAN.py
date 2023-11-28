name = "ADVGAN"
object_list = ["AdvGan"]
config = {
    "attacker_config": {
        "AdvGan": {
            "model_num_labels": 1000,
            "image_nc": 3,
            "clip_min": 0,
            "clip_max": 1,
            "epochs": 400,
            "attack_type": "UNTARGETED"
        }
    }
}