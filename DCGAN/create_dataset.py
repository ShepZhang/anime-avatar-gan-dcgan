import os
from PIL import Image, ImageOps
from torchvision.utils import make_grid
import torch
import torch.utils.data as data

class My_dataset(data.Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.transform = transform
        self.exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        self.files = [f for f in os.listdir(self.path) if f.lower().endswith(self.exts)]
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        fp = os.path.join(self.path, self.files[index])
        img = Image.open(fp)
        # 方向&透明通道处理，统一到RGB
        img = ImageOps.exif_transpose(img)
        if img.mode in ("P", "RGBA"):
            bg = Image.new("RGBA", img.size, (255,255,255,255))
            bg.alpha_composite(img.convert("RGBA"))
            img = bg.convert("RGB")
        else:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

def save_img(tensor, fp, nrow=8, padding=2, pad_value=1.0, outer_border=1):
    """
    将[-1,1]的批量图像保存为网格：
      - 固定 nrow 列，行数由 batch 大小 / nrow 决定
      - 白色分隔（pad_value=1.0），padding=2 像素
      - 外圈薄灰边框，outer_border=1 像素
    """
    # 归一化到[0,1]，设定网格分隔线为白色
    grid = make_grid(
        tensor, nrow=nrow, padding=padding,
        normalize=True, value_range=(-1, 1),
        pad_value=pad_value
    )
    # 转成 PIL
    ndarr = (grid.permute(1, 2, 0).cpu().clamp(0, 1) * 255).to(torch.uint8).numpy()
    im = Image.fromarray(ndarr)

    # 外圈薄灰边框（可选）
    if outer_border and outer_border > 0:
        im = ImageOps.expand(im, border=outer_border, fill=(190, 190, 190))

    im.save(fp)
