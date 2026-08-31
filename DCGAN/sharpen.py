# -*- coding: utf-8 -*-
"""
批量锐化（数据增强）- 路径写死版
- 递归读取 IN_DIR 下的 jpg/png/jpeg/webp/bmp
- 输出到 OUT_DIR，并在文件名追加后缀
- 默认使用 Unsharp Mask（更自然），可切换到 3x3 锐化卷积
"""

import os
from pathlib import Path
import numpy as np
import cv2

# =========================
# 路径写死（按需修改）
# =========================
IN_DIR  = r"./preprocessed_square_128_png_2"          # 原始图片所在文件夹
OUT_DIR = r"./sharpen_128"    # 锐化后图片输出文件夹

# =========================
# 锐化参数（常用区间见注释）
# =========================
MODE = "unsharp"  # "unsharp" 或 "kernel"（3x3卷积核锐化）

# Unsharp Mask 参数（动漫头像常用：amount 1.2~1.6，radius 0.8~1.6，threshold 0~3）
USM_AMOUNT    = 1.4   # 锐化强度
USM_RADIUS    = 1.2   # 模糊半径（越大越平滑，边缘越明显）
USM_THRESHOLD = 1.0   # 阈值（像素差小于阈值不锐化，抑制噪点），单位：像素值（0~255）

# Kernel 锐化强度（固定 3x3 核，strength>1 叠加锐化）
KERNEL_STRENGTH = 1.0

# 支持的图片后缀
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------- 工具函数：兼容中文路径的读写 ----------
def imread_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    return img

def imwrite_unicode(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    # 默认使用原扩展名编码
    ok, buf = cv2.imencode(ext if ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"] else ".png", img)
    if not ok:
        # 回退为 .png
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise RuntimeError(f"Failed to encode image for {path}")
        path = path.with_suffix(".png")
    buf.tofile(str(path))
    return path


# ---------- 锐化实现 ----------
def unsharp_mask(img_bgr: np.ndarray, amount=1.5, radius=1.0, threshold=0.0):
    """
    Unsharp Mask（反遮罩锐化）
    - amount: 锐化强度（1.2~1.8 常用）
    - radius: 高斯模糊半径（0.8~2.0 常用）
    - threshold: 小于该差值的像素不锐化（0~3 可抑制噪点）
    """
    # 转 float32 并归一化
    img = img_bgr.astype(np.float32) / 255.0

    # 计算高斯核大小（奇数）
    k = max(3, int(round(radius * 3) * 2 + 1))
    blur = cv2.GaussianBlur(img, (k, k), sigmaX=radius, borderType=cv2.BORDER_REPLICATE)
    highpass = img - blur

    if threshold > 0:
        # 将阈值缩放到 [0,1]
        thr = threshold / 255.0
        mask = (np.abs(highpass) >= thr).astype(np.float32)
        highpass *= mask

    sharp = img + amount * highpass
    sharp = np.clip(sharp, 0.0, 1.0)
    sharp = (sharp * 255.0 + 0.5).astype(np.uint8)
    return sharp


def kernel_sharpen_3x3(img_bgr: np.ndarray, strength=1.0):
    """
    经典 3x3 锐化核：[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    strength>1 时叠加效果（谨慎过大，易产生噪点和光晕）
    """
    base_kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]], dtype=np.float32)
    # 简单叠加实现强度
    kernel = base_kernel.copy()
    if strength > 1.0:
        times = int(round(strength - 1.0))
        for _ in range(times):
            img_bgr = cv2.filter2D(img_bgr, -1, base_kernel, borderType=cv2.BORDER_REPLICATE)
        return img_bgr
    else:
        return cv2.filter2D(img_bgr, -1, kernel, borderType=cv2.BORDER_REPLICATE)


# ---------- 主流程 ----------
def process_one(img_path: Path, out_root: Path):
    rel = img_path.relative_to(IN_DIR)
    out_path = out_root / rel
    suffix = out_path.suffix
    stem = out_path.stem

    img = imread_unicode(img_path)
    if img is None:
        print(f"[跳过] 读取失败：{img_path}")
        return

    # 统一转换为 BGR 3 通道
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        # 去掉 alpha，防止锐化引入边缘伪影
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if MODE == "unsharp":
        out_img = unsharp_mask(img, amount=USM_AMOUNT, radius=USM_RADIUS, threshold=USM_THRESHOLD)
        tag = f"_usm_a{USM_AMOUNT:g}_r{USM_RADIUS:g}_t{USM_THRESHOLD:g}"
    elif MODE == "kernel":
        out_img = kernel_sharpen_3x3(img, strength=KERNEL_STRENGTH)
        tag = f"_k3_s{KERNEL_STRENGTH:g}"
    else:
        raise ValueError("MODE 必须为 'unsharp' 或 'kernel'")

    save_path = out_path.with_name(f"{stem}{tag}{suffix}")
    imwrite_unicode(save_path, out_img)
    print(f"[OK] {img_path} -> {save_path}")


def main():
    in_root = Path(IN_DIR)
    out_root = Path(OUT_DIR)
    if not in_root.exists():
        raise FileNotFoundError(f"输入目录不存在：{in_root}")

    files = [p for p in in_root.rglob("*") if p.suffix.lower() in EXTS]
    print(f"[INFO] 共发现 {len(files)} 张图片。模式: {MODE}")
    for i, p in enumerate(files, 1):
        try:
            process_one(p, out_root)
        except Exception as e:
            print(f"[ERR] {p}: {e}")
        if i % 50 == 0:
            print(f"[进度] 已处理 {i}/{len(files)}")

    print("[DONE] 全部处理完成。")


if __name__ == "__main__":
    main()
