# Anime Avatar Generation with GAN & DCGAN

这是一个面向动漫头像生成场景的 GAN/DCGAN 实验项目，包含基础 GAN、DCGAN、图像预处理、多轮训练日志、模型权重、生成样图和相关数据归档。

## 项目概览

- 使用全连接 GAN 作为教学与效果对照基线。
- 使用 DCGAN 作为主要实验模型，生成 64×64 动漫头像。
- 对锐化、饱和度、对比度、高斯模糊等数据处理方案进行了多组实验。
- 主训练配置运行至 500 epoch，并保存普通、最佳和 EMA 生成器权重。
- 训练过程保留生成器/判别器损失、真假样本评分、阶段样图和最终权重。

## 目录结构

```text
GAN/                 基础 GAN 源码、权重和生成样图
DCGAN/               DCGAN 源码、主训练权重、日志和生成样图
datasets/            相关数据与历史实验归档（Git LFS）
docs/                训练笔记
requirements.txt     Python 依赖
DATASET_MANIFEST.md  数据归档说明和来源
SHA256SUMS.txt        大文件完整性校验值
```

## 推荐模型

推理或继续实验时，建议优先检查：

- `DCGAN/generator_ema_best.pth`：EMA 平滑后的最佳生成器权重。
- `DCGAN/generator_best.pth`：训练过程中保存的最佳普通生成器权重。
- `DCGAN/generator.pth`：第 500 epoch 的最终生成器权重。
- `DCGAN/discriminator.pth`：第 500 epoch 的判别器权重。

当前保存的最佳样图位于 `DCGAN/sample/best_fake_images_latest.png`。

## 获取大文件

模型权重和数据压缩包通过 Git LFS 管理。克隆前请安装 Git LFS：

```powershell
git lfs install
git clone https://github.com/ShepZhang/anime-avatar-gan-dcgan.git
```

如需恢复主 DCGAN 数据集，将 `datasets/dcgan-preprocessed-square-128-v2.zip` 解压至 `DCGAN/`，然后在 `DCGAN` 目录运行训练程序。

## 环境与运行

推荐使用 Python 3.10：

```powershell
pip install -r requirements.txt
cd DCGAN
python main.py
```

模型权重使用 PyTorch 序列化格式。只应加载本项目生成或来源可信的 `.pth` 文件。

## 数据与来源说明

项目 README 原先引用了以下公开资源：

- [Anime Faces（Kaggle）](https://www.kaggle.com/datasets/soumikrakshit/anime-faces)
- [Anime Face Dataset 论文](https://arxiv.org/pdf/1907.13394.pdf)

本仓库同时包含实习期间形成的本地筛选集和预处理结果，不能保证每张图片均来自同一来源。仓库因此默认保持私有；在改为公开仓库或用于商业用途之前，应重新核对第三方代码、图片与模型权重的授权条件。

## 项目状态

这是实习期间形成的实验性原型和历史备份，重点是保存完整实验过程并支持后续复现，不代表生产环境模型。
