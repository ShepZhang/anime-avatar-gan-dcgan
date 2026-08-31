'''
数据增强---锐化+对比度增强
'''
import os
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

# === 配置参数 ===
input_dir = "C:/Users/ap/Desktop/image_1500"
output_dir = "./cropped_1500_images"
target_size = (64, 64)
contrast_range = (1.2, 1.5)  # 对比度增强范围（1.0为原始）

os.makedirs(output_dir, exist_ok=True)

# === 居中裁剪为正方形 + 缩放 ===
def center_crop_resize(image):
    width, height = image.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    cropped = image.crop((left, top, right, bottom))
    return cropped.resize(target_size, Image.LANCZOS)

# === 对比度增强 + 锐化 ===
def enhance_contrast_sharpen(img):
    # 随机对比度增强
    factor = random.uniform(*contrast_range)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # 应用锐化滤镜
    sharpened = img.filter(ImageFilter.SHARPEN)
    return sharpened

# === 主处理流程 ===
import random
image_exts = ['.jpg', '.jpeg', '.png']
file_list = [f for f in os.listdir(input_dir) if any(f.lower().endswith(ext) for ext in image_exts)]

print(f"🔧 正在增强 {len(file_list)} 张图像（对比度+锐化）...")

for filename in tqdm(file_list):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_enhanced.jpg")

    with Image.open(input_path).convert("RGB") as img:
        cropped = center_crop_resize(img)
        cropped.save(output_path)

print(f"✅ 处理完成，增强图像已保存至 {output_dir}")
