import argparse, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageOps, ImageEnhance, ImageFilter

# 兼容 Pillow 新旧版本的重采样常量
try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC


def is_image(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def make_square(img: Image.Image, mode: str = "center-crop", bgcolor=(255, 255, 255)) -> Image.Image:
    """将任意宽高图片变成正方形：center-crop 或 pad（留白填充）"""
    w, h = img.size
    if mode == "pad":
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), bgcolor)
        # 若是带 alpha，先转到 RGB（丢弃 alpha；如需保留，可自行改成 RGBA）
        if img.mode != "RGB":
            img = img.convert("RGB")
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas
    else:  # center-crop
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        return img.crop((left, top, left + side, top + side))


def unsharp(img: Image.Image, radius: float, percent: int, threshold: int) -> Image.Image:
    """Unsharp Mask（建议半径 1.0~1.6，percent 100~180，threshold 0~6）"""
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def process_one(src: Path, dst_root: Path, args) -> str:
    try:
        # 读图 & 纠正 EXIF 方向
        with Image.open(src) as im0:
            im = ImageOps.exif_transpose(im0)
            # 统一到 RGB（训练通常不需要 alpha 通道；如需保留可改成 RGBA 流程）
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGB")

            # 变正方形
            im = make_square(im, mode=args.square, bgcolor=tuple(args.pad_bg))

            # 缩放到目标尺寸
            # reducing_gap 可以提高抗锯齿质量（Pillow>=9.1）
            im = im.resize((args.size, args.size), RESAMPLE_BICUBIC, reducing_gap=3.0)

            # 锐化（Unsharp Mask）
            if args.unsharp is not None:
                r, p, t = args.unsharp
                im = unsharp(im, r, p, t)

            # 提升饱和度
            if abs(args.sat - 1.0) > 1e-3:
                im = ImageEnhance.Color(im).enhance(args.sat)

            # 可选再微调整体清晰度（和 unsharp 作用不同）
            if abs(args.sharp - 1.0) > 1e-3:
                im = ImageEnhance.Sharpness(im).enhance(args.sharp)

            # 输出路径（保留子目录结构）
            rel = src.relative_to(args.input)
            out = (dst_root / rel).with_suffix("." + args.format.lower())
            out.parent.mkdir(parents=True, exist_ok=True)

            save_kwargs = {}
            fmt = args.format.lower()
            if fmt in ("jpg", "jpeg"):
                # 高质量 JPEG，适合训练：关闭色度子采样，优化/渐进编码
                save_kwargs.update(dict(quality=args.quality, optimize=True, progressive=True, subsampling=0))
            elif fmt == "png":
                save_kwargs.update(dict(compress_level=args.png_compress))
            elif fmt == "webp":
                save_kwargs.update(dict(quality=args.quality, method=6))

            im.save(out, format=args.format.upper(), **save_kwargs)
        return f"OK: {src}"
    except Exception as e:
        return f"FAIL: {src} -> {e}"


def main():
    parser = argparse.ArgumentParser(
        description="批量将图片压到固定尺寸（默认 128），并进行锐化+提升饱和度。"
    )
    parser.add_argument("--input", type=Path, required=True, help="输入图片文件夹")
    parser.add_argument("--output", type=Path, required=True, help="输出文件夹")
    parser.add_argument("--size", type=int, default=128, help="目标边长（默认 128）")
    parser.add_argument("--square", choices=["center-crop", "pad"], default="center-crop",
                        help="正方形方式：中心裁剪 或 留白填充（默认 center-crop）")
    parser.add_argument("--pad-bg", type=int, nargs=3, default=(255, 255, 255),
                        help="当 square=pad 时的背景色，RGB 三个值（默认 255 255 255）")

    # 饱和度/锐度
    parser.add_argument("--sat", type=float, default=1.15, help="饱和度增强系数，1=不变（默认 1.15）")
    parser.add_argument("--sharp", type=float, default=1.00, help="整体锐度增强（ImageEnhance.Sharpness），默认 1.0 不启用")
    parser.add_argument("--unsharp", type=float, nargs=3, metavar=("RADIUS", "PERCENT", "THRESHOLD"),
                        default=(1.2, 130, 3),
                        help="UnsharpMask 参数：半径 百分比 阈值（默认 1.2 130 3；设为 0 0 0 关闭）")

    # 输出格式
    parser.add_argument("--format", choices=["png", "jpg", "jpeg", "webp"], default="png",
                        help="输出图片格式（默认 png）")
    parser.add_argument("--quality", type=int, default=92, help="jpg/webp 质量（默认 92）")
    parser.add_argument("--png-compress", type=int, default=6, help="PNG 压缩等级 0-9（默认 6）")

    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="并发线程数（默认 CPU 数）")

    args = parser.parse_args()

    # 关闭 unsharp 的方法：--unsharp 0 0 0
    if args.unsharp and args.unsharp == (0.0, 0.0, 0.0):
        args.unsharp = None

    # 收集所有图片
    all_files = [p for p in args.input.rglob("*") if p.is_file() and is_image(p)]
    if not all_files:
        print("未在输入目录中发现图片文件。")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"共发现 {len(all_files)} 张图片，开始处理…")

    # 并发处理
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one, p, args.output, args) for p in all_files]
        for i, fu in enumerate(as_completed(futures), 1):
            msg = fu.result()
            results.append(msg)
            if i % 50 == 0 or i == len(all_files):
                print(f"[{i}/{len(all_files)}] {msg}")

    # 简报
    fail = [r for r in results if r.startswith("FAIL")]
    print(f"完成：成功 {len(results) - len(fail)}，失败 {len(fail)}")
    if fail:
        print("失败列表示例（前 10 条）：")
        for r in fail[:10]:
            print("  ", r)


if __name__ == "__main__":
    main()
