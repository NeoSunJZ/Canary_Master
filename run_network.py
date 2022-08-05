from flask_cors import CORS
from CANARY_SEFI.core.exec_network import create_app
from CANARY_SEFI.handler.socket.socket import get_socket

app = create_app()
CORS(app, resources=r'/*')

websocket = get_socket(app)

from CANARY_SEFI.core.component.component_manager import SEFI_component_manager
from Model.Vision_Transformer import sefi_component as vision_transformer_model

SEFI_component_manager.add(vision_transformer_model)


# 攻击方案
# CW
from Attack_Method.white_box_adv.CW import sefi_component as cw_attacker
SEFI_component_manager.add(cw_attacker)

if __name__ == "__main__":
    websocket.run(app, host="0.0.0.0")