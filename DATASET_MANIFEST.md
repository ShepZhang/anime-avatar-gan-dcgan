# 数据与实验归档清单

所有 ZIP 文件均位于 `datasets/`，并通过 Git LFS 保存。压缩包保留原始目录名称，解压后可直接检查其内部结构。

| 归档文件 | 原始内容 | 约计大小 |
| --- | --- | ---: |
| `dcgan-preprocessed-square-128.zip` | 主要 128/64 像素预处理数据、多轮训练结果及记录 | 223.1 MB |
| `dcgan-preprocessed-square-128-v2.zip` | 主训练脚本当前引用的数据目录 | 5.9 MB |
| `dcgan-contrast.zip` | 对比度实验数据与结果 | 45.6 MB |
| `dcgan-sharpened.zip` | 锐化实验数据与结果 | 27.5 MB |
| `dcgan-saturated.zip` | 饱和度实验数据与结果 | 26.4 MB |
| `dcgan-mix.zip` | 混合预处理实验数据与结果 | 19.5 MB |
| `dcgan-contrast-sharpen.zip` | 对比度与锐化组合实验 | 16.5 MB |
| `dcgan-gaussian-blur.zip` | 高斯模糊实验 | 15.7 MB |
| `dcgan-sharpen-128.zip` | 128 像素锐化数据 | 7.3 MB |
| `dcgan-cropped-1500.zip` | 约 1,500 张裁剪图片 | 3.0 MB |
| `image-1500.zip` | 桌面 `image_1500` 数据及对应实验结果 | 254.0 MB |
| `new-10-image.zip` | 桌面 `new_10_image` 数据及对应实验结果 | 39.5 MB |
| `original-reference.zip` | 早期/原始项目备份 | 13.4 MB |

## 完整性检查

下载完成后，可在仓库根目录执行：

```powershell
Get-FileHash datasets\*.zip -Algorithm SHA256
```

将输出与 `SHA256SUMS.txt` 对照即可确认文件是否完整。

## 注意事项

- 归档中可能同时存在训练图片、阶段样图、日志、Word 实验记录和历史权重。
- 本地数据不保证全部具有统一授权；仓库应保持私有，除非完成单独的来源与授权审计。
- 原项目 README 中的第三方网盘提取密码没有复制到本仓库。
