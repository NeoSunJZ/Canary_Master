from enum import Enum


class Step(Enum):
    INIT = {'step_name': "初始化"}
    MODEL_CAPABILITY_TEST = {'step_name': "模型基线能力测试"}
    ADV_IMG_BUILD_AND_EVALUATION = {'step_name': "对抗样本生成与质量评估"}
    EXPLORE_ATTACK_PERTURBATION = {'step_name': "攻击扰动探索"}
    EXPLORE_ATTACK_PERTURBATION_ATTACK_TEST = {'step_name': "攻击扰动探索攻击测试"}
    EXPLORE_ATTACK_PERTURBATION_ATTACK_EVALUATION = {'step_name': "攻击扰动探索攻击评估"}
    ATTACK_CAPABILITY_TEST = {'step_name': "攻击能力测试"}
    ATTACK_CAPABILITY_EVALUATION = {'step_name': "攻击能力评估"}
    MODEL_CAPABILITY_EVALUATION = {'step_name': "模型能力评估"}
    MODEL_SECURITY_SYNTHETICAL_CAPABILITY_EVALUATION = {'step_name': "模型综合安全能力评估"}

    def step_name(self):
        return self.value.get('step_name')
