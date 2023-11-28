from canary_sefi.core.function.enum.multi_db_mode_enum import MultiDatabaseMode
from canary_sefi.core.function.helper.multi_db import use_multi_database
from canary_sefi.core.function.helper.recovery import global_recovery
from canary_sefi.service.security_evaluation import SecurityEvaluation
from canary_sefi.task_manager import task_manager
from utils import load_test_config

# Load Canary Lib into SEFI Component Manager
# If the user defines their own components (including attack methods, models, or dataset loaders),
# they need to be loaded into the SEFI Component Manager to be effective
from canary_sefi.core.component.component_manager import SEFI_component_manager
from canary_lib import canary_lib  # Canary Lib
SEFI_component_manager.add_all(canary_lib)

# In freedom mode, you should need to select a preset configuration to build this task configuration.
# If the preset configuration does not meet your requirements,
# you may need to add a new preset configuration or use freedom mode.
# Please read the documentation carefully when writing the configuration.
if __name__ == "__main__":
    # Build this task configuration from the pre-set configuration.
    config = load_test_config(
        attack_config="IFGSM",
        data_config="ILSVRC2012-1000-SEED",
        model_config="IMAGENET-15",
        attack_batch_config="IFGSM-IMAGENET-15-RTX3090",
        model_batch_config="IMAGENET-15-RTX3090"
    )

    # Initialize the task,
    # which will create a new folder for storing images and generate a new SQLite database for the task
    task_manager.init_task(show_logo=True, run_device="cuda")

    # If your task is accidentally interrupted,
    # whether it is a proactive termination or an unexpected error, it can be recovered through recovery mode
    # You need to fill in the token of the task to the "task_token"

    # There can only be one task initialization and recovery, otherwise an error will occur
    # (therefore, we have annotated the recovery mode code to avoid errors)
    # ------------------
    # global_recovery.start_recovery_mode(task_token="", show_logo=True, run_device="cuda")
    # ------------------

    # Set the current mode to a simple database.
    # Please read the document about multiple databases.
    use_multi_database(mode=MultiDatabaseMode.SIMPLE)

    # Build security testing and execute.
    security_evaluation = SecurityEvaluation(config)
    security_evaluation.attack_full_test()
