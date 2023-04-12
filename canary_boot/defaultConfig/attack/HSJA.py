name = "HSJA"
object_list = ["HopSkipJumpAttack"]
config = {
    "attacker_config": {
        "HopSkipJumpAttack": {
            "clip_min": 0,
            "clip_max": 1,
            "attack_type": "UNTARGETED",
            "gamma": 0.1,
            "steps": 64,
            "max_gradient_eval_steps": 10000,
            "initial_gradient_eval_steps": 100,
            "constraint": "linf"
        }
    },
}