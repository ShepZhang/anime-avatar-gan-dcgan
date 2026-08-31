import torch
from torch import nn

# --------- 小工具 ---------
def sn(layer, use_sn=True):
    return nn.utils.spectral_norm(layer) if use_sn else layer

class MinibatchStdDev(nn.Module):
    """
    最简单的 minibatch stddev：按 batch 求标量 std，扩成 1xHxW 拼到通道上
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        B, C, H, W = x.shape
        if B == 1:
            # batch=1 时退化为常数 0
            std = x.new_zeros(B, 1, H, W)
        else:
            std = x.float().var(dim=0, unbiased=False).add(self.eps).sqrt()    # CxHxW
            std = std.mean().view(1, 1, 1, 1).repeat(B, 1, H, W)               # Bx1xHxW
        return torch.cat([x, std.to(x.dtype)], dim=1)

# --------- 生成器：Upsample + Conv 去棋盘格 ---------
class Generator(nn.Module):
    def __init__(self, noise_dim=100, use_sn=False):
        super().__init__()
        self.noise_dim = noise_dim

        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 4*4*512),
            nn.BatchNorm1d(4*4*512),
            nn.ReLU(True),
        )

        def up(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                sn(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), use_sn),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            )

        self.net = nn.Sequential(
            # 4x4 -> 8 -> 16 -> 32 -> 64
            up(512, 256),   # 8x8
            up(256, 128),   # 16x16
            up(128, 64),    # 32x32
            up(64, 32),     # 64x64
            sn(nn.Conv2d(32, 3, kernel_size=3, padding=1), use_sn),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        # z: [B, Z, 1, 1] 或 [B, Z]
        if z.dim() == 4:  # [B, Z, 1, 1] -> [B, Z]
            z = z.view(z.size(0), -1)
        x = self.fc(z).view(-1, 512, 4, 4)
        return self.net(x)

# --------- 判别器：raw logits + MinibatchStdDev（无 Sigmoid） ---------
class Discriminator(nn.Module):
    def __init__(self, use_sn=True):
        super().__init__()

        C = 64
        self.features = nn.Sequential(
            sn(nn.Conv2d(3,   C,   3, stride=2, padding=1), use_sn), nn.LeakyReLU(0.2, True),   # 64->32
            sn(nn.Conv2d(C,   C*2, 3, stride=2, padding=1), use_sn), nn.LeakyReLU(0.2, True),   # 32->16
            sn(nn.Conv2d(C*2, C*4, 3, stride=2, padding=1), use_sn), nn.LeakyReLU(0.2, True),   # 16->8
            sn(nn.Conv2d(C*4, C*4, 3, stride=2, padding=1), use_sn), nn.LeakyReLU(0.2, True),   # 8->4
        )

        # 加一个非常轻的 MinibatchStdDev
        self.mstd = MinibatchStdDev()  # +1 channel
        self.conv_last = sn(nn.Conv2d(C*4 + 1, C*4, 3, padding=1), use_sn)
        self.act_last  = nn.LeakyReLU(0.2, True)

        self.head = sn(nn.Linear(4*4*C*4, 1), use_sn)  # raw logits

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.mstd(x)
        x = self.act_last(self.conv_last(x))
        x = x.view(x.size(0), -1)
        out = self.head(x)             # [B, 1], raw logits（不要 Sigmoid）
        return out
