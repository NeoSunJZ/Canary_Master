from flask_cors import CORS
from CANARY_SEFI.exec_network import create_app
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter
from component_manager import init_component_manager

app = create_app()
CORS(app, resources=r'/*')

websocket = reporter.get_socket(app)


if __name__ == "__main__":
    init_component_manager()
    websocket.run(app, host="0.0.0.0")