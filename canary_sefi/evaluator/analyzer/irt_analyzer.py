from __future__ import print_function, division
import numpy as np
from tqdm import tqdm


def _log_lognormal(param):
    # 对数正态分布的概率密度分布的对数
    return np.log(1.0 / param) + _log_normal(np.log(param))


def _log_normal(param):
    # 正态分布的概率密度分布的对数
    return param ** 2 * -0.5


def _param_den(slop, threshold, guess):
    # 项目参数联合概率密度
    return _log_normal(threshold) + _log_lognormal(slop) + 4 * np.log(guess) + 16 * np.log(1 - guess)


def logistic(slop, threshold, guess, theta):
    # logistic函数
    return guess + (1 - guess) / (1.0 + np.exp(-1 * (slop * theta + threshold)))


def loglik(slop, threshold, guess, theta, scores, axis=1):
    # 对数似然函数
    p = logistic(slop, threshold, guess, theta)
    p[p <= 0] = 1e-10
    p[p >= 1] = 1 - 1e-10
    loglik = np.sum(scores * np.log(p) + (1 - scores) * np.log(1 - p), axis=axis)
    return loglik


def _tran_theta(slop, threshold, guess, theta, next_theta, scores):
    # 特质的转移函数
    pi = (loglik(slop, threshold, guess, next_theta, scores) + _log_normal(next_theta)[:, 0]) - (
        loglik(slop, threshold, guess, theta, scores) + _log_normal(theta)[:, 0])
    pi = np.exp(pi)
    # 下步可省略
    pi[pi > 1] = 1
    return pi


def _tran_item_para(slop, threshold, guess, next_slop, next_threshold, next_guess, theta, scores):
    # 项目参数的转移函数
    nxt = loglik(next_slop, next_threshold, next_guess, theta, scores, 0) + _param_den(next_slop, next_threshold,
                                                                                       next_guess)
    now = loglik(slop, threshold, guess, theta, scores, 0) + _param_den(slop, threshold, guess)
    pi = nxt - now
    pi.shape = pi.shape[1]
    pi = np.exp(pi)
    # 下步可省略
    pi[pi > 1] = 1
    return pi


def mcmc(chain_size, scores):
    # 样本量
    person_size = scores.shape[0]
    # 项目量
    item_size = scores.shape[1]
    # 潜在特质初值
    theta = np.zeros((person_size, 1))
    # 斜率初值
    slop = np.ones((1, item_size))
    # 阈值初值
    threshold = np.zeros((1, item_size))
    # 猜测参数初值
    guess = np.zeros((1, item_size)) + 0.1
    # 参数储存记录
    theta_list = np.zeros((chain_size, len(theta)))
    slop_list = np.zeros((chain_size, item_size))
    threshold_list = np.zeros((chain_size, item_size))
    guess_list = np.zeros((chain_size, item_size))
    for i in tqdm(range(chain_size)):
        next_theta = np.random.normal(theta, 1)
        theta_pi = _tran_theta(slop, threshold, guess, theta, next_theta, scores)
        theta_r = np.random.uniform(0, 1, len(theta))
        theta[theta_r <= theta_pi] = next_theta[theta_r <= theta_pi]
        theta_list[i] = theta[:, 0]
        next_slop = np.random.normal(slop, 0.3)
        # 防止数值溢出
        next_slop[next_slop < 0] = 1e-10
        next_threshold = np.random.normal(threshold, 0.3)
        next_guess = np.random.uniform(guess - 0.03, guess + 0.03)
        # 防止数值溢出
        next_guess[next_guess <= 0] = 1e-10
        next_guess[next_guess >= 1] = 1 - 1e-10
        param_pi = _tran_item_para(slop, threshold, guess, next_slop, next_threshold, next_guess, theta, scores)
        param_r = np.random.uniform(0, 1, item_size)
        slop[0][param_r <= param_pi] = next_slop[0][param_r <= param_pi]
        threshold[0][param_r <= param_pi] = next_threshold[0][param_r <= param_pi]
        guess[0][param_r <= param_pi] = next_guess[0][param_r <= param_pi]
        slop_list[i] = slop[0]
        threshold_list[i] = threshold[0]
        guess_list[i] = guess[0]
    return theta_list, slop_list, threshold_list, guess_list

def result_2_score_mapper(evaluation_results, evaluation_type):
    scores_list = []
    for result in evaluation_results:
        score = {}
        if evaluation_type == "model":
            score['model_name'] = result['model_name']
        else:
            score['atk_name'] = result['atk_name']
        # Model Capability
        if evaluation_type == "model":
            score['model_capability'] = {}
            score['model_capability']['model_ACC'] = result['model_ACC']
            score['model_capability']['model_F1'] = result['model_F1']
            score['model_capability']['model_Conf'] = result['model_Conf']
        # Attack Capability /  Model Robustness
        # Attack Effects
        score['attack_effects'] = {}
        score['attack_effects']['MR'] = result['MR']
        score['attack_effects']['AIAC'] = result['AIAC']
        score['attack_effects']['ARTC'] = result['ARTC']
        score['attack_effects']['ACAMC_A'] = 0 if result['ACAMC_A'] <= 1 / 2 else 2 * result['ACAMC_A'] - 1
        score['attack_effects']['ACAMC_T'] = 0 if result['ACAMC_A'] <= 1 / 2 else 2 * result['ACAMC_T'] - 1
        if evaluation_type == "attack":
            score['attack_effects']['OTR'] = result['OTR']
        # Disturbance-aware Cost
        score['da_cost'] = {}
        score['da_cost']['APCR'] = result['APCR']
        score['da_cost']['AED'] = 10 * result['AED'] if result['AED'] <= 1 / 10 else 1
        score['da_cost']['AMD'] = (255 / 32) * result['AMD'] if result['AMD'] <= 32 / 255 else 1
        score['da_cost']['AED_HF'] = 10 * result['AED_HF'] if result['AED_HF'] <= 1 / 10 else 1
        score['da_cost']['AED_LF'] = 10 * result['AED_LF'] if result['AED_LF'] <= 1 / 10 else 1
        score['da_cost']['ADMS'] = (5 / 2) * result['ADMS'] if result['ADMS'] <= 2 / 5 else 1
        score['da_cost']['ALMS'] = (5 / 2) * result['ALMS'] if result['ALMS'] <= 2 / 5 else 1
        # Calculate Cost
        if evaluation_type == "attack":
            score['calculate_cost'] = {}
            AQN = ((result['AQN_F'] + result['AQN_B']) / 2) if result['AQN_B'] > 0 else result['AQN_F']
            score['calculate_cost']['AQN'] = (1 / 50000) * AQN if AQN <= 50000 else 1
            score['calculate_cost']['ACT'] = 0.1 if result['ACT'] == "verySlow" else 0.3 if result['ACT'] == "slow" else 0.5 if result['ACT'] == "normal" else 0.7 if result['ACT'] == "fast" else 0.9 if result['ACT'] == "veryFast" else 0

        if evaluation_type == "attack":
            score['attack_effects']['ACAMC_A'] = 1 - score['attack_effects']['ACAMC_A']
            score['attack_effects']['ACAMC_T'] = 1 - score['attack_effects']['ACAMC_T']

            score['da_cost']['APCR'] = 1 - score['da_cost']['APCR']
            score['da_cost']['AED'] = 1 - score['da_cost']['AED']
            score['da_cost']['AMD'] = 1 - score['da_cost']['AMD']
            score['da_cost']['AED_HF'] = 1 - score['da_cost']['AED_HF']
            score['da_cost']['AED_LF'] = 1 - score['da_cost']['AED_LF']
            score['da_cost']['ADMS'] = 1 - score['da_cost']['ADMS']
            score['da_cost']['ALMS'] = 1 - score['da_cost']['ALMS']

            score['calculate_cost']['AQN'] = 1 - score['calculate_cost']['AQN']
            score['calculate_cost']['ACT'] = 1 - score['calculate_cost']['ACT']
        else:
            score['attack_effects']['MR'] = 1 - score['attack_effects']['MR']
            score['attack_effects']['AIAC'] = 1 - score['attack_effects']['AIAC']
            score['attack_effects']['ARTC'] = 1 - score['attack_effects']['ARTC']

        scores_list.append(score)
    return scores_list


def subitem_irt_calculator(scores_list, subitem):
    scores = []
    for item in scores_list:
        scores.append(list(item[subitem].values()))
    scores_np = np.array(scores).transpose()

    scores = []
    for item in scores_np:
        scores.append(normalization(item))
    return irt_calculator(np.array(scores).transpose())


def irt_score_sort(scores_list, irt_score, evaluation_type):
    irt_score_map = {}
    for i in range(len(scores_list)):
        name = scores_list[i]['model_name' if evaluation_type == "model" else 'atk_name']
        irt_score_map[name] = irt_score[i]
    sort_result = sorted(irt_score_map.items(), key=lambda s: s[1])
    print(sort_result)


def synthetical_capability_irt_score_analyzer(evaluation_results, evaluation_type, mapper=result_2_score_mapper):
    scores_list = mapper(evaluation_results, evaluation_type)
    final_irt_result = []
    # Model Capability
    if evaluation_type == "model":
        irt_result_model_capability = subitem_irt_calculator(scores_list, "model_capability")
        final_irt_result.append(irt_result_model_capability[0])
        irt_score_sort(scores_list, irt_result_model_capability[0], evaluation_type)

    # Attack Capability /  Model Robustness
    irt_result_attack_effects = subitem_irt_calculator(scores_list, "attack_effects")
    final_irt_result.append(irt_result_attack_effects[0])
    irt_score_sort(scores_list, irt_result_attack_effects[0], evaluation_type)

    if evaluation_type == "attack":
        irt_result_calculate_cost = subitem_irt_calculator(scores_list, "calculate_cost")
        final_irt_result.append(irt_result_calculate_cost[0])
        irt_score_sort(scores_list, irt_result_calculate_cost[0], evaluation_type)

    irt_result_da_cost = subitem_irt_calculator(scores_list, "da_cost")
    final_irt_result.append(irt_result_da_cost[0])
    irt_score_sort(scores_list, irt_result_da_cost[0], evaluation_type)

    final_irt_result = np.array(final_irt_result).transpose()
    irt_score = irt_calculator(final_irt_result)
    irt_score_sort(scores_list, irt_score[0], evaluation_type)


def normalization(data):
    min_value = min(data)
    max_value = max(data)
    if min_value == max_value:
        return data
    new_list = []
    for i in data:
        new_list.append(round((i - min_value) / (max_value - min_value), 2))
    return new_list


def irt_calculator(scores):
    print(scores)
    thetas, slops, thresholds, guesses = mcmc(60000, scores=scores)
    est_theta = np.mean(thetas[30000:], axis=0)
    est_slop = np.mean(slops[30000:], axis=0)
    est_threshold = np.mean(thresholds[18000:], axis=0)
    est_guess = np.mean(guesses[18000:], axis=0)
    return normalization(est_theta), est_theta, est_slop, est_threshold, est_guess