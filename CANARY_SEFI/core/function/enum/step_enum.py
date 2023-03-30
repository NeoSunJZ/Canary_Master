from enum import Enum


class Step(Enum):
    INIT = {'step_name': "初始化"}

    MODEL_INFERENCE_CAPABILITY_TEST = {'step_name': "模型预测能力测试"}
    MODEL_INFERENCE_CAPABILITY_TEST_MIGRATED = {'step_name': "模型预测能力测试(迁移)"}
    MODEL_INFERENCE_CAPABILITY_EVALUATION = {'step_name': "模型预测能力评估"}

    ADV_EXAMPLE_GENERATE = {'step_name': "对抗样本生成"}
    ATTACK_ADV_EXAMPLE_COMPARATIVE_TEST = {'step_name': "攻击方法生成对抗样本综合对比测试(图像相似性/模型注意力差异对比/像素差异对比)"}
    ATTACK_ADV_EXAMPLE_DA_AND_COST_EVALUATION = {'step_name': "攻击方法生成对抗样本图像相似性(扰动距离)/生成代价评估"}

    ATTACK_DEFLECTION_CAPABILITY_TEST = {'step_name': "攻击方法推理偏转效果/模型注意力偏转效果测试"}
    ATTACK_DEFLECTION_CAPABILITY_EVALUATION = {'step_name': "攻击方法推理偏转效果/模型注意力偏转效果评估"}

    ADV_EXAMPLE_GENERATE_WITH_PERTURBATION_INCREMENT = {'step_name': "递增扰动的对抗样本生成"}
    ATTACK_DEFLECTION_CAPABILITY_TEST_WITH_PERTURBATION_INCREMENT = \
        {'step_name': "攻击方法推理偏转效果/模型注意力偏转效果测试(扰动递增)"}
    ATTACK_ADV_EXAMPLE_COMPARATIVE_TEST_WITH_PERTURBATION_INCREMENT = \
        {'step_name': "攻击方法生成对抗样本综合对比测试(图像相似性/模型注意力差异对比/像素差异对比)(扰动递增)"}
    ATTACK_EVALUATION_WITH_PERTURBATION_INCREMENT = {'step_name': "攻击方法效果评估(扰动递增)"}

    MODEL_SECURITY_SYNTHETICAL_CAPABILITY_EVALUATION = {'step_name': "模型综合安全能力评估"}
    ATTACK_SYNTHETICAL_CAPABILITY_EVALUATION = {'step_name': "攻击方法综合能力评估"}

    DEFENSE_ADVERSARIAL_TRAINING = {'step_name': "对抗训练"}
    DEFENSE_NORMAL_EFFECTIVENESS_EVALUATION = {'step_name': "防御方法在干净样本上的有效性评估"}
    DEFENSE_ADVERSARIAL_EFFECTIVENESS_EVALUATION = {'step_name': "防御方法在对抗样本上的有效性评估"}

    TRANS_DEFLECTION_CAPABILITY_TEST = {'step_name': "图片预处理防御样本-攻击方法推理偏转效果/模型注意力偏转效果测试"}
    TRANS_DEFLECTION_CAPABILITY_EVALUATION = {'step_name': "图片预处理防御样本-攻击方法推理偏转效果/模型注意力偏转效果评估"}

    def step_name(self):
        return self.value.get('step_name')
