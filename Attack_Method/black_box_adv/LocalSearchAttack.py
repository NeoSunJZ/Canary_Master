import torch
import numpy as np

from CANARY_SEFI.core.component.component_decorator import SEFIComponent
from CANARY_SEFI.core.component.component_enum import ComponentConfigHandlerType, ComponentType

sefi_component = SEFIComponent()


@sefi_component.attacker_class(attack_name="LocalSearchAttack", perturbation_budget_var_name="epsilon")
@sefi_component.config_params_handler(handler_target=ComponentType.ATTACK, name="LocalSearchAttack",
                                      args_type=ComponentConfigHandlerType.ATTACK_PARAMS, use_default_handler=True,
                                      params={
                                          "epsilon": {"desc": "扰动大小", "type": "FLOAT", "def": "0.2"},
                                          "clip_min": {"desc": "对抗样本像素上界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "clip_max": {"desc": "对抗样本像素下界(与模型相关)", "type": "FLOAT", "required": "true"},
                                          "attack_type": {"desc": "攻击类型", "type": "SELECT",
                                                          "selector": [{"value": "TARGETED", "name": "靶向"},
                                                                       {"value": "UNTARGETED", "name": "非靶向"}],
                                                          "required": "true"},
                                          "tlabel": {"desc": "靶向攻击目标标签(分类标签)(仅TARGETED时有效)", "type": "INT"},
                                          "max_iter": {"desc": "最大迭代次数(整数)", "type": "INT", "def": "150"},
                                          "r": {"desc": "超参r ∈ [0,2]", "type": "FLOAT", "def": "1.5"},
                                          "p": {"desc": "超参（取值可为任意正实数）", "type": "FLOAT", "def": "0.25"},
                                          "d": {"desc": "局部搜索的半径", "type": "INT", "def": "5"},
                                          "t": {"desc": "所能干扰的像素数", "type": "INT", "def": "5"},
                                          "mini_batch": {"desc": "最大同时预测数", "type": "INT", "def": "16"},
                                      })
class LocalSearchAttack():
    def __init__(self, model, run_device, max_iter=150, epsilon=0.2, clip_min=-3, clip_max=3, attack_type='UNTARGETED',
                 tlabel=-1, r=1.5, p=0.25, d=5, t=5, mini_batch=16):
        self.model = model  # 待攻击的白盒模型
        self.epsilon = epsilon  # 以无穷范数作为约束，设置最大值
        self.clip_min = clip_min  # 像素值的下限
        self.clip_max = clip_max  # 像素值的上限
        self.attack_type = attack_type  # 攻击类型：靶向 or 非靶向
        self.tlabel = tlabel
        self.device = run_device if run_device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.R = max_iter
        self.r = r
        self.p = p
        self.d = d
        self.t = t
        assert 0 <= self.r <= 2
        self.mini_batch = mini_batch
        # 参数有问题  可能是0~255
        # Simple Black-Box Adversarial Perturbations for Deep Networks 函数命名也完全和论文一致
        # perturbation factor p 扰动系数
        # two perturbation parameters p ∈ R and r ∈ [0,2],
        # a budget U ∈ N on the number of trials
        # the half side length of the neighborhood square d ∈ N,
        # the number of pixels perturbed at each round t ∈ N,

    @sefi_component.attack(name="LocalSearchAttack", is_inclass=True, support_model=["vision_transformer"])
    def attack(self, imgs, ori_labels):

        min_, max_ = self.clip_min, self.clip_max

        # 强制拷贝 避免针对adv_img的修改也影响adversary.original
        adv_img = imgs.clone()
        original_label = ori_labels[0]
        _, channels, h, w = imgs.size()

        # 正则化到[-0.5,0.5]区间内
        def normalize(im):

            im = im - (min_ + max_) / 2
            im = im / (max_ - min_)

            LB = -1 / 2
            UB = 1 / 2
            return im, LB, UB

        def unnormalize(im):
            # import pdb
            # pdb.set_trace()
            im = im * (max_ - min_)
            im = im + (min_ + max_) / 2
            return im

        # 归一化
        adv_img, LB, UB = normalize(adv_img)

        # 随机选择一部分像素点 总数不超过全部的10% 最大为128个点
        def random_locations():
            n = int(0.1 * h * w)
            n = min(n, 128)
            locations = np.random.permutation(h * w)[:n]
            p_x = locations % w
            p_y = locations // w
            pxy = list(zip(p_x, p_y))
            pxy = np.array(pxy)
            return pxy

        # 针对图像的每个信道的点[x,y]同时进行修改 修改的值为p * np.sign(Im[location]) 类似FGSM的一次迭代
        # 不修改Ii的图像 返回修改后的图像
        def pert(Ii, p, x, y):
            Im = Ii.clone()
            # location = [1, x, y]
            # location.insert(1, slice(None))
            # location = tuple(0, :, x, y)

            Im[0, :, x, y] = p * torch.sign(Im[0, :, x, y])
            return Im

        # 截断 确保assert LB <= r * Ibxy <= UB 但是也有可能阶段失败退出 因此可以适当扩大配置的原始数据范围
        # 这块的实现没有完全参考论文
        def cyclic(r, Ibxy):

            # logger.info("cyclic Ibxy:{}".format(Ibxy))
            result = r * Ibxy
            # logger.info("cyclic result:{}".format(result))

            """
            foolbox的实现 存在极端情况  本来只是有一个元素超过UB，结果一减 都完蛋了
            if result.any() < LB:
                #result = result/r + (UB - LB)
                logger.info("cyclic result:{}".format(result))
                result = result + (UB - LB)
                logger.info("cyclic result:{}".format(result))
                #result=LB
            elif result.any()  > UB:
                #result = result/r - (UB - LB)
                logger.info("cyclic result:{}".format(result))
                result = result - (UB - LB)
                logger.info("cyclic result:{}".format(result))
                #result=UB
            """

            if result.any() < LB:
                result = result + (UB - LB)
            elif result.any() > UB:
                result = result - (UB - LB)

            result = result.clip(LB, UB)

            # logger.info("cyclic result:{}".format(result))

            # assert LB <= np.all(result) <= UB

            return result

        Ii = adv_img
        PxPy = random_locations()
        # 循环攻击轮
        for try_time in range(self.R):
            # 重新排序 随机选择不不超过128个点
            PxPy = PxPy[np.random.permutation(len(PxPy))[:128]]
            L = [pert(Ii, self.p, x, y) for x, y in PxPy]

            # 批量返回预测结果 获取原始图像标签的概率
            def score(Its):
                # mini_batch = 16
                iters = len(Its) // self.mini_batch
                scores = np.array([])
                Its = torch.cat(Its, 0)
                Its = unnormalize(Its)
                for i in range(iters):
                    scores = np.append(scores, self.model(Its[self.mini_batch * i: self.mini_batch * (i + 1), :, :, :])[:,
                                               original_label].cpu().detach().numpy())

                return scores  # np.stack(scores, 0)# torch.cat(scores, 0)

            # 选择影响力最大的t个点进行扰动 抓主要矛盾 这里需要注意 np.argsort是升序 所以要取倒数的几个
            scores = score(L)

            indices = np.argsort(scores)[:-self.t]

            PxPy_star = PxPy[indices]

            for x, y in PxPy_star:
                # 每个颜色通道的[x，y]点进行扰动并截断 扰动算法即放大r倍
                for b in range(channels):
                    location = [0, x, y]
                    location.insert(1, b)
                    location = tuple(location)
                    Ii[location] = cyclic(self.r, Ii[location])

            f = self.model(unnormalize(Ii))

            adv_label = torch.argmax(f).cpu().item()
            if self.attack_type == 'UNTARGETED':
                if adv_label != original_label:
                    return unnormalize(adv_img)
            else:
                if adv_label == self.tlabel:
                    return unnormalize(adv_img)

            # 扩大搜索范围，把原有点周围2d乘以2d范围内的点都拉进来 去掉超过【w，h】的点
            # "{Update a neighborhood of pixel locations for the next round}"

            PxPy = [
                (x, y)
                for _a, _b in PxPy_star
                for x in range(_a - self.d, _a + self.d + 1)
                for y in range(_b - self.d, _b + self.d + 1)]
            PxPy = [(x, y) for x, y in PxPy if 0 <= x < w and 0 <= y < h]
            PxPy = list(set(PxPy))
            PxPy = np.array(PxPy)

        return unnormalize(adv_img)

