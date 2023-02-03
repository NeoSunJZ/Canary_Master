name = "SA"
object_list = ["SpatialAttack"]
config = {
    "attacker_config": {
        "SpatialAttack": {
            "attack_type": "UNTARGETED",
            "clip_min": 0,
            "clip_max": 1,
            "max_translation": 3,
            "max_rotation": 30,
            "num_translations": 5,
            "num_rotations": 5,
            "grid_search": True,
            "random_steps": 100
        }
    }
}