name = "BA"
object_list = ["BA"]
config = {
    "attacker_config": {
        "BA": {
            "clip_min": 0,
            "clip_max": 1,
            "attack_type": "UNTARGETED",
            "max_iterations": 10000,
            "spherical_step": 1e-2,
            "source_step_convergence": 1e-07,
            "step_adaptation": 1.5,
            "source_step": 1e-2,
            "update_stats_every_k": 10,
        }
    }
}
