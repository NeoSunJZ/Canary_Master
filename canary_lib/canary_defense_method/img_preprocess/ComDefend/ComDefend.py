import torch
import torch.nn as nn

class encoder(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(encoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 12, kernel_size=3, padding=1)
        )
        # Weight initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)

    def forward(self, x):
        return self.features(x)


class decoder(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(decoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        # # Weight initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)

    def forward(self, x):
        return self.features(x)


class ComDefend:
    def __init__(self):
        enc = encoder()
        dec = decoder()
        self.enc = enc.cuda()
        self.dec = dec.cuda()
        checkpoint = torch.load('../canary_lib/canary_defense_method/img_preprocess/ComDefend/cifar_comdefend_best.pth')
        self.enc.load_state_dict(checkpoint['enc'])
        self.dec.load_state_dict(checkpoint['dec'])

    def com_img(self, img):
        linear_code = self.enc(img)
        # noisy_code = linear_code - torch.randn(linear_code.size()).cuda() * 20/255
        noisy_code = linear_code
        binary_code = torch.round(torch.sigmoid(noisy_code))
        binary_code = torch.sigmoid(noisy_code)
        recons_x = self.dec(binary_code)
        return recons_x

