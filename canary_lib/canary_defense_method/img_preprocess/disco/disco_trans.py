import torch

import inr


class Disco:
    def trans(self, img):
        device = torch.device('cuda')
        disco_path = 'epoch-best.pth'
        defense = inr.INR(device, disco_path, height=224, width=224)
        return defense.forward(img)
