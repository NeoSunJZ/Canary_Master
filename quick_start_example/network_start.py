from flask_cors import CORS
from canary_sefi.exec_network import create_app
from canary_sefi.core.function.helper.realtime_reporter import reporter

# Load Canary Lib into SEFI Component Manager
# If the user defines their own components (including attack methods, models, or dataset loaders),
# they need to be loaded into the SEFI Component Manager to be effective
from canary_sefi.core.component.component_manager import SEFI_component_manager
from canary_lib import canary_lib  # Canary Lib

SEFI_component_manager.add_all(canary_lib)

# In network mode, you can directly start it
# Related operations need to rely on Canary Web to complete.
if __name__ == "__main__":
    app = create_app()
    CORS(app, resources=r'/*')
    reporter.get_socket(app).run(app, host="0.0.0.0")
