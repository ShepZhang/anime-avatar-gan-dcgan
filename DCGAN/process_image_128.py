import os, math
from PIL import Image, ImageOps
from tqdm import tqdm

input_dir  = "C:/Users/ap/Desktop/new_10_image/image2"
output_dir = "./preprocessed_square_128_png_2"   # 建议先存 128 或 256
square_size = 128                              # 可改为 256
os.makedirs(output_dir, exist_ok=True)

def to_square_center(img):
    # 处理 EXIF 方向，防止横着/倒着
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top  = (h - m) // 2
    img = img.crop((left, top, left + m, top + m))
    # 下采样优先用 LANCZOS（Pillow>=10 等价于 Image.Resampling.LANCZOS）
    return img.resize((square_size, square_size), Image.LANCZOS)

for name in tqdm(os.listdir(input_dir)):
    if not name.lower().endswith((".jpg",".jpeg",".png",".webp")):
        continue
    path = os.path.join(input_dir, name)
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            sq = to_square_center(im)
            # 尽量用 PNG，避免二次压缩伪影
            out = os.path.join(output_dir, os.path.splitext(name)[0] + ".png")
            sq.save(out)  # PNG 无损
            # 若必须用 JPG，请至少这样：
            # sq.save(out_jpg, format="JPEG", quality=95, subsampling=0, optimize=True)
    except Exception as e:
        print("skip:", name, e)
