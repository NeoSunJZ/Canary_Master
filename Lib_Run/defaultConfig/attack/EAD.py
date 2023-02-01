name = "EAD"
object_list = ["EAD"]
config = {
    "attacker_config": {
        "EAD": {
            "attack_type": 'UNTARGETED',
            "tlabel": None,
            "clip_min": 0,
            "clip_max": 1,
            "epsilon": 16/255,
            "binary_search_steps": 9,
            "steps": 1000,
            "initial_stepsize": 0.01,
            "confidence": 0.0,
            "initial_const": 0.001,
            "regularization": 0.01,
            "decision_rule": 'EN',
            "abort_early": True
        }
    }
}
