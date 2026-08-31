import os
from PIL import Image, ImageEnhance
from tqdm import tqdm

# ====== 配置（直接改这里） ======
INPUT_DIR = r"C:/Users/ap/Desktop/image_1500"          # 原始图片文件夹路径
OUTPUT_DIR = r"./contrast"    # 输出图片文件夹路径
TARGET_SIZE = 64                       # 压缩后的目标边长
CONTRAST_FACTOR = 1.30                 # 对比度增强倍数（1.0=不变；>1增强；<1减弱）
# 保存格式设置
SAVE_FORMAT = "png"    # 可选 "png"（无损）或 "jpg"（有损）
JPEG_QUALITY = 95      # 仅 jpg 有效（80~95推荐）
PNG_COMPRESS = 6       # 仅 png 有效（0~9，越大体积越小）
# ============================

os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def center_crop_resize(im: Image.Image, size: int) -> Image.Image:
    """中心裁剪为正方形后，LANCZOS 高质量缩放到 size×size"""
    w, h = im.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    im = im.crop((left, top, left + m, top + m))
    return im.resize((size, size), Image.LANCZOS)

def enhance_contrast(im: Image.Image, factor: float) -> Image.Image:
    """对比度增强：factor=1不变，>1增强，<1减弱"""
    return ImageEnhance.Contrast(im).enhance(factor)

def main():
    files = [f for f in os.listdir(INPUT_DIR)
             if os.path.splitext(f)[1].lower() in IMG_EXTS]

    print(f"🔧 共 {len(files)} 张图：中心裁剪 → 缩放到 {TARGET_SIZE}×{TARGET_SIZE} → 对比度 ×{CONTRAST_FACTOR}")

    for fname in tqdm(files):
        in_path = os.path.join(INPUT_DIR, fname)
        stem, _ = os.path.splitext(fname)
        out_ext = ".png" if SAVE_FORMAT.lower() == "png" else ".jpg"
        out_name = f"{stem}_ctr{out_ext}"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        try:
            with Image.open(in_path) as im:
                im = im.convert("RGB")  # 统一三通道
                im = center_crop_resize(im, TARGET_SIZE)
                im = enhance_contrast(im, CONTRAST_FACTOR)

                if SAVE_FORMAT.lower() == "png":
                    im.save(out_path, format="PNG", compress_level=PNG_COMPRESS)
                else:
                    im.save(out_path, format="JPEG", quality=JPEG_QUALITY, subsample=0, optimize=True)
        except Exception as e:
            print(f"⚠️ 跳过 {fname}: {e}")

    print(f"✅ 完成！输出目录：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
