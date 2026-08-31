import os, copy, math, numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from create_dataset import My_dataset, save_img
from model64 import Generator, Discriminator

# =========================
# 配置
# =========================
DATA_DIR   = r"./preprocessed_square_128_png_2"
EPOCHS     = 500
BATCH_SIZE = 32
Z_DIM      = 100

LOG_FILE   = "training_log.txt"
SAMPLE_DIR = "./sample"
os.makedirs(SAMPLE_DIR, exist_ok=True)
SAVE_EVERY = 50
GRID_NROW, GRID_ROWS = 8, 4   # 8x4=32

# 学习率（TTUR：稍微压低 D）
G_LR, D_LR = 2e-4, 6e-5
BETAS      = (0.5, 0.999)

# Instance Noise（更轻更短）
NOISE_WARMUP = 0.15
INSTANCE_NOISE_SIGMA0 = 0.02

# Lazy R1（更轻）
USE_R1      = True
R1_GAMMA    = 5.0
R1_INTERVAL = 16   # 每 16 步加一次，权重乘 interval 等效

# G:D 步数比（抑制 D 过强）
G_STEPS = 2

# 早停（基于 D(sigmoid) 的 margin）
MARGIN_THR = 0.60
REDUCE_PATIENCE, STOP_PATIENCE = 12, 8
MARGIN_SMA_WIN = 5

# EMA
EMA_DECAY = 0.999

# DataLoader
NUM_WORKERS, PIN_MEMORY = 2, True

# =========================
# 变换 & DiffAugment（轻度）
# =========================
def build_transform():
    return transforms.Compose([
        transforms.Resize(72, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(64),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

def _color_jitter(x, b, c, s):
    # x∈[-1,1]，全部可导
    x = x + b                      # brightness
    x = x * c                      # contrast
    mean = x.mean(dim=1, keepdim=True)
    x = (x - mean) * s + mean      # saturation
    return x.clamp(-1, 1)

def _translate(x, tx, ty):
    B, C, H, W = x.size()
    theta = x.new_zeros(B, 2, 3)
    theta[:, 0, 0] = 1; theta[:, 1, 1] = 1
    theta[:, 0, 2] = tx * 2
    theta[:, 1, 2] = ty * 2
    grid = torch.nn.functional.affine_grid(theta, size=x.size(), align_corners=False)
    return torch.nn.functional.grid_sample(x, grid, mode='bilinear',
                                           padding_mode='zeros', align_corners=False)

def sample_diffaug_params(B, device, bright=0.05, contr=0.06, sat=0.06, trans=0.04):
    # real/fake 同步一套参数
    b  = (torch.rand(1, 1, 1, 1, device=device) - 0.5) * 2 * bright
    c  = 1.0 + (torch.rand(1, 1, 1, 1, device=device) - 0.5) * 2 * contr
    s  = 1.0 + (torch.rand(1, 1, 1, 1, device=device) - 0.5) * 2 * sat
    tx = (torch.rand(1, device=device) - 0.5) * 2 * trans
    ty = (torch.rand(1, device=device) - 0.5) * 2 * trans
    return dict(b=b, c=c, s=s, tx=tx, ty=ty)

def apply_diffaugment(x, params):
    B = x.size(0)
    x = _color_jitter(x, params["b"], params["c"], params["s"])
    tx = params["tx"].expand(B)
    ty = params["ty"].expand(B)
    return _translate(x, tx, ty)

# =========================
# 其它工具
# =========================
def instance_sigma(global_step, total_steps, warmup_ratio=NOISE_WARMUP, sigma0=INSTANCE_NOISE_SIGMA0):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if global_step >= warmup_steps: return 0.0
    return sigma0 * (1.0 - global_step / warmup_steps)

def add_instance_noise(x, sigma):
    if sigma <= 0: return x
    return (x + torch.randn_like(x) * sigma).clamp(-1, 1)

def early_action(margins, reduced_once):
    k = REDUCE_PATIENCE if not reduced_once else STOP_PATIENCE
    if len(margins) < k: return "none"
    tail = margins[-k:]
    sma  = np.mean(margins[-min(MARGIN_SMA_WIN, len(margins)):])
    if all(m > MARGIN_THR for m in tail) and sma > MARGIN_THR:
        return "reduce" if not reduced_once else "stop"
    return "none"

def set_requires_grad(module, flag: bool):
    for p in module.parameters(): p.requires_grad_(flag)

# =========================
# 训练
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("Epoch,Step,d_loss,g_loss,D_real,D_fake\n")

    dataset = My_dataset(DATA_DIR, transform=build_transform())
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    G, D = Generator(Z_DIM).to(device), Discriminator().to(device)
    G.train(); D.train()
    G_ema = copy.deepcopy(G).to(device); set_requires_grad(G_ema, False)

    g_opt = torch.optim.Adam(G.parameters(), lr=G_LR, betas=BETAS)
    d_opt = torch.optim.Adam(D.parameters(), lr=D_LR, betas=BETAS)

    total_steps = EPOCHS * len(loader)
    global_step = 0
    margin_history, reduced_once = [], False

    best_epoch_g, best_g_loss = None, float("inf")
    fixed_noise = torch.randn(GRID_NROW * GRID_ROWS, Z_DIM, 1, 1, device=device)

    for epoch in range(1, EPOCHS + 1):
        D_real_sum = D_fake_sum = g_loss_sum = 0.0
        seen = 0

        for step, real in enumerate(loader, start=1):
            global_step += 1
            real = real.to(device, non_blocking=True)
            bsz  = real.size(0)
            sigma = instance_sigma(global_step, total_steps)

            # --------- D step ---------
            z = torch.randn(bsz, Z_DIM, 1, 1, device=device)
            with torch.no_grad():
                fake = G(z)

            real_in = add_instance_noise(real, sigma)
            fake_in = add_instance_noise(fake, sigma)

            aug_params = sample_diffaug_params(bsz, device)  # 同步增强
            real_aug = apply_diffaugment(real_in, aug_params)
            fake_aug = apply_diffaugment(fake_in, aug_params)

            real_aug.requires_grad_(USE_R1)

            real_out = D(real_aug)   # raw logits
            fake_out = D(fake_aug)

            # HingeD
            d_loss = F.relu(1.0 - real_out).mean() + F.relu(1.0 + fake_out).mean()

            # Lazy R1（等效乘 interval）
            if USE_R1 and (global_step % R1_INTERVAL == 0):
                grad_real = torch.autograd.grad(outputs=real_out.sum(), inputs=real_aug,
                                                create_graph=True, retain_graph=True, only_inputs=True)[0]
                r1 = grad_real.view(bsz, -1).pow(2).sum(1).mean()
                d_loss = d_loss + 0.5 * R1_GAMMA * R1_INTERVAL * r1

            d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            d_opt.step()

            with torch.no_grad():
                D_real_sum += torch.sigmoid(real_out).mean().item()
                D_fake_sum += torch.sigmoid(fake_out).mean().item()
            seen += 1

            # --------- G step (2x) ---------
            set_requires_grad(D, False)
            cur_g_loss = None
            for _ in range(G_STEPS):
                z2 = torch.randn(bsz, Z_DIM, 1, 1, device=device)
                fake2 = G(z2)
                fake2_aug = apply_diffaugment(fake2, sample_diffaug_params(bsz, device))
                out_fake = D(fake2_aug)
                # HingeG
                g_loss = -out_fake.mean()
                g_opt.zero_grad(set_to_none=True)
                g_loss.backward()
                g_opt.step()
                cur_g_loss = g_loss  # 记录最后一次

                # EMA 累积
                with torch.no_grad():
                    for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                        p_ema.mul_(EMA_DECAY).add_(p, alpha=1.0 - EMA_DECAY)
                    for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                        b_ema.copy_(b)

            set_requires_grad(D, True)

            g_loss_sum += cur_g_loss.detach().item()

            if step % 5 == 0:
                print(f"Epoch[{epoch}/{EPOCHS}] d:{d_loss.item():.4f} g:{cur_g_loss.item():.4f} "
                      f"D_real:{torch.sigmoid(real_out).mean().item():.3f} "
                      f"D_fake:{torch.sigmoid(fake_out).mean().item():.3f}")
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{epoch},{step},{d_loss.item():.6f},{cur_g_loss.item():.6f},"
                            f"{torch.sigmoid(real_out).mean().item():.6f},{torch.sigmoid(fake_out).mean().item():.6f}\n")

        # ===== 每 epoch：样张 & 早停 & 最佳点 =====
        if epoch % SAVE_EVERY == 0:
            with torch.no_grad():
                sample = G_ema(fixed_noise)
            save_img(sample, os.path.join(SAMPLE_DIR, f"fake_images_{epoch:03d}.png"),
                     nrow=GRID_NROW, padding=2, pad_value=1.0, outer_border=1)

        D_real_epoch = D_real_sum / max(1, seen)
        D_fake_epoch = D_fake_sum / max(1, seen)
        margin_epoch = D_real_epoch - D_fake_epoch
        margin_history.append(margin_epoch)

        g_loss_epoch = g_loss_sum / max(1, seen)  # hinge 下越小（更负）越好
        if g_loss_epoch < best_g_loss:
            best_g_loss = g_loss_epoch
            best_epoch_g = epoch
            torch.save(G.state_dict(),     "./generator_best.pth")
            torch.save(G_ema.state_dict(), "./generator_ema_best.pth")
            with torch.no_grad():
                best_sample = G_ema(fixed_noise)
            save_img(best_sample, os.path.join(SAMPLE_DIR, f"best_fake_images_epoch{epoch:03d}.png"),
                     nrow=GRID_NROW, padding=2, pad_value=1.0, outer_border=1)
            save_img(best_sample, os.path.join(SAMPLE_DIR, "best_fake_images_latest.png"),
                     nrow=GRID_NROW, padding=2, pad_value=1.0, outer_border=1)
            print(f"[Best @ {epoch}] g_epoch={best_g_loss:.4f} ✓ saved")

        act = early_action(margin_history, reduced_once)
        if act == "reduce":
            for g in d_opt.param_groups: g["lr"] *= 0.5
            reduced_once = True
            print(f"[TTUR] Reduce D lr -> {d_opt.param_groups[0]['lr']:.2e} (margin={margin_epoch:.3f})")
        elif act == "stop":
            print(f"[Early Stop] margin>{MARGIN_THR} persists after LR reduce. Stop at epoch {epoch}.")
            break

    torch.save(G.state_dict(),     "./generator.pth")
    torch.save(G_ema.state_dict(), "./generator_ema.pth")
    torch.save(D.state_dict(),     "./discriminator.pth")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
