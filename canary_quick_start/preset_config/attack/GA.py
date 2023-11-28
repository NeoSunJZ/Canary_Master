name = "GA"
object_list = ["GA"]
config = {
    "attacker_config": {
        "GA": {
            "clip_min": 0,
            "clip_max": 1,
            "step": 1000,
            "attack_type": "TARGETED",
            "tlabel": None,
            "epsilon": None,
            "population": 10,  # 种群数量，这个算法中是随机选择的一定范围内的样本数量，foolbox默认10，论文中实验采用6但测试的是Mnist和cifar
            "mutation_probability": 0.1,
            "mutation_range": 0.15,  # 变异概率和变异范围，论文和foolbox都取值为0.1和0.15，要注意实际上他们是一个自适应量，这里只是初始值
            "sampling_temperature": 0.3,  # 采样温度，用于计算遗传算法中的选择概率，论文中没有给出数值，foolbox默认0.3
            "reduced_dims": None,  # 是否减少搜索空间的维度，默认none，这个参数是论文中提出的，因为高维搜索空间需要更多成本，所以考虑缩减。
        }
    }
}
