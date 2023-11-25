# 模型(API)
from .baidu_api import sefi_component as baidu_api
from .alibaba_api import sefi_component as alibaba_api
from .huawei_api import sefi_component as huawei_api
from .tencent_api import sefi_component as tencent_api
from .remote_model import sefi_component as remote_model

model_list = [
    baidu_api, alibaba_api, huawei_api, tencent_api, remote_model
]