'''
数据增强---高斯模糊
'''
import os
from PIL import Image, ImageFilter
from tqdm import tqdm

# === 配置参数 ===
input_dir = "C:/Users/ap/Desktop/image_1500"           # 原始图片目录
output_dir = "./gaussian_blur_images"  # 输出目录
target_size = (64, 64)
blur_radius = 1.0  # 高斯模糊半径（0.5 ~ 2 一般较合理）

os.makedirs(output_dir, exist_ok=True)

# === 裁剪为中心正方形 + 缩放到目标大小 ===
def center_crop_resize(image):
    width, height = image.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    cropped = image.crop((left, top, left + min_dim, top + min_dim))
    return cropped.resize(target_size, Image.LANCZOS)

# === 应用高斯模糊 ===
def apply_gaussian_blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

# === 主处理流程 ===
image_exts = ['.jpg', '.jpeg', '.png']
file_list = [f for f in os.listdir(input_dir) if any(f.lower().endswith(ext) for ext in image_exts)]

print(f"🔧 正在处理 {len(file_list)} 张图像，每张将添加高斯模糊并压缩至 64×64...")

for filename in tqdm(file_list):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_blur.jpg")

    with Image.open(input_path).convert("RGB") as img:
        cropped = center_crop_resize(img)
        blurred = apply_gaussian_blur(cropped)
        blurred.save(output_path, quality=95)  # 用较高质量保存，避免过度压缩

print(f"✅ 已处理完成，图像保存至 {output_dir}")
