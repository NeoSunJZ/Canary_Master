from flask_cors import CORS
from CANARY_SEFI.exec_network import create_app
from CANARY_SEFI.core.function.helper.realtime_reporter import reporter

app = create_app()
CORS(app, resources=r'/*')

websocket = reporter.get_socket(app)

from CANARY_SEFI.core.component.component_manager import SEFI_component_manager

# 模型
# Alexnet
from Model.Alexnet_ImageNet import sefi_component as alexnet_model
SEFI_component_manager.add(alexnet_model)
# VGG16
from Model.VGG16_ImageNet import sefi_component as vgg_16_model
SEFI_component_manager.add(vgg_16_model)

# 数据集
# IMAGENET2012
from Dataset.ImageNet2012.dataset_loader import sefi_component as imgnet2012_dataset
SEFI_component_manager.add(imgnet2012_dataset)

# 攻击方案
# CW
from Attack_Method.white_box_adv.CW import sefi_component as cw_attacker
SEFI_component_manager.add(cw_attacker)
# MI-FGSM
from Attack_Method.white_box_adv.MI_FGSM import sefi_component as mi_fgsm_attacker
SEFI_component_manager.add(mi_fgsm_attacker)

if __name__ == "__main__":
    websocket.run(app, host="0.0.0.0")