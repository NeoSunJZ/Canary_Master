from enum import Enum


class Step(Enum):
    INIT = {'step_name': "初始化"}

    MODEL_INFERENCE_CAPABILITY_TEST = {'step_name': "模型预测能力测试"}
    MODEL_INFERENCE_CAPABILITY_EVALUATION = {'step_name': "模型预测能力评估"}

    ADV_EXAMPLE_GENERATE = {'step_name': "对抗样本生成"}
    ATTACK_ADV_EXAMPLE_DA_TEST = {'step_name': "攻击方法生成对抗样本质量测试"}
    ATTACK_ADV_EXAMPLE_DA_EVALUATION = {'step_name': "攻击方法生成对抗样本质量评估"}
    ATTACK_DEFLECTION_CAPABILITY_TEST = {'step_name': "攻击方法偏转效果测试"}
    ATTACK_DEFLECTION_CAPABILITY_EVALUATION = {'step_name': "攻击方法偏转效果评估"}

    ADV_EXAMPLE_GENERATE_WITH_PERTURBATION_INCREMENT = {'step_name': "递增扰动的对抗样本生成"}
    ATTACK_DEFLECTION_CAPABILITY_TEST_WITH_PERTURBATION_INCREMENT = {'step_name': "攻击方法偏转效果测试(扰动递增)"}
    ATTACK_ADV_EXAMPLE_DA_TEST_WITH_PERTURBATION_INCREMENT = {'step_name': "攻击方法生成对抗样本质量测试(扰动递增)"}
    ATTACK_EVALUATION_WITH_PERTURBATION_INCREMENT = {'step_name': "攻击方法效果评估(扰动递增)"}

    MODEL_SECURITY_SYNTHETICAL_CAPABILITY_EVALUATION = {'step_name': "模型综合安全能力评估"}

    def step_name(self):
        return self.value.get('step_name')
