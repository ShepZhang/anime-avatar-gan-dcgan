'''
数据增强---锐化
'''
import os
from PIL import Image, ImageFilter
from tqdm import tqdm

# ====== 配置（直接改这里） ======下·
INPUT_DIR = r"C:\Users\ap\Desktop\image_1500"        # 原始图片文件夹路径
OUTPUT_DIR = r"./sharpened" # 输出图片文件夹路径
TARGET_SIZE = 64                     # 压缩后的目标边长
# UnsharpMask 锐化参数
RADIUS = 1.2       # 半径，越大锐化范围越广（推荐 1.0 ~ 1.5）
PERCENT = 150      # 强度，越大越锐（推荐 120 ~ 180）
THRESHOLD = 3      # 阈值，越小边缘锐化更明显
# 保存格式设置
SAVE_FORMAT = "png"   # 可选 "png"（无损）或 "jpg"（有损）
JPEG_QUALITY = 95     # 仅 jpg 有效
PNG_COMPRESS = 6      # 仅 png 有效（0~9，越大体积越小）
# ============================

os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def center_crop_resize(im: Image.Image, size: int) -> Image.Image:
    w, h = im.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    im = im.crop((left, top, left + m, top + m))
    return im.resize((size, size), Image.LANCZOS)

def sharpen_unsharp(im: Image.Image, radius: float, percent: int, threshold: int) -> Image.Image:
    return im.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

def main():
    files = [f for f in os.listdir(INPUT_DIR)
             if os.path.splitext(f)[1].lower() in IMG_EXTS]

    print(f"🔧 共 {len(files)} 张图，将裁剪 → 缩放到 {TARGET_SIZE}×{TARGET_SIZE} → 锐化")

    for fname in tqdm(files):
        in_path = os.path.join(INPUT_DIR, fname)
        stem, _ = os.path.splitext(fname)
        out_ext = ".png" if SAVE_FORMAT.lower() == "png" else ".jpg"
        out_name = f"{stem}_sharp{out_ext}"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        try:
            with Image.open(in_path) as im:
                im = im.convert("RGB")
                im = center_crop_resize(im, TARGET_SIZE)
                im = sharpen_unsharp(im, radius=RADIUS, percent=PERCENT, threshold=THRESHOLD)

                if SAVE_FORMAT.lower() == "png":
                    im.save(out_path, format="PNG", compress_level=PNG_COMPRESS)
                else:
                    im.save(out_path, format="JPEG", quality=JPEG_QUALITY, subsample=0, optimize=True)
        except Exception as e:
            print(f"⚠️ 跳过 {fname}: {e}")

    print(f"✅ 完成！输出目录：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
