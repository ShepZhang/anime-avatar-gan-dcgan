# DCGAN Anime 64×64 — 训练笔记 & 演进 README

> 一套从“最初版”到“稳定版”的改动脉络 + 实用配方（含监控指标、常见故障排查、推荐超参范围）。随仓库携带，便于复现与继续迭代。

---

## TL;DR（当前稳定配方）

- **数据几何**：`Resize(72, BICUBIC, antialias=True) → CenterCrop(64) → Flip(0.5)`  
- **模型**：
  - **G**：`Upsample(×2) + 3×3 Conv` 的 **resize-conv** 结构（消灭棋盘格），BN+ReLU，最后 `Tanh`
  - **D**：**SpectralNorm** 包裹 Conv/Linear，LeakyReLU(0.2)，**输出 raw logits**
- **损失**：**Hinge Loss**（`D: relu(1−D(x)) + relu(1+D(G(z)))`，`G: −E[D(G(z))]`）+ **Lazy R1**（γ≈10, 每 16 步一次）
- **优化**：**TTUR**（G: 2e-4 / D: 1e-4），当 `margin = sigmoid(D_real) − sigmoid(D_fake)` 连续偏高 → **只降 D LR ×0.5**
- **稳态件**：
  - **Instance Noise**：σ 从 `0.02~0.05` 线性衰减到 0（前 10~30% 步）
  - **轻 DiffAug**：同一批 **real/fake 同步** 的亮度/对比/饱和（±6~8%）+ 小平移（±6~8%）
- **EMA**：`decay=0.999~0.9997`，**只用 EMA 出图 & 挑最佳点**
- **采样与日志**：每 **50 epoch** 保存样张；CSV 日志：`Epoch,Step,d_loss,g_loss,D_real,D_fake`

---

## 1. 为什么这么改（问题 → 原因 → 对策）

### 1.1 拉伸变形 & 锯齿 → 早期学到伪特征
- **原因**：直接 `Resize(64,64)` 会改变人物几何；最近邻/双线性缩放锯齿明显。
- **对策**：`Resize(72, BICUBIC, antialias=True) → CenterCrop(64)`；统一 **RGB**、修正 EXIF 旋转、透明图以白底合成。

### 1.2 棋盘格 / 网纹
- **原因**：`ConvTranspose2d` 的 kernel/stride 组合易产生对齐伪影。
- **对策**：G 改为 **resize-conv**（`Upsample → Conv3×3`），末端再一层 `Conv3×3` 更干净。

### 1.3 D 过强、梯度饱和、训练“拉边”
- **原因**：BCE + Sigmoid 的 D 在极端区间梯度弱；判别器学得太快。
- **对策**：**Hinge Loss**（raw logits）+ **SpectralNorm** + **Lazy R1**；**TTUR** 只降 D LR；**Instance Noise** 平滑边界。

### 1.4 增强带来“脏感”
- **原因**：锐化/强对比会放大噪声、违背数据分布；real/fake 不同步参数会让 D“看穿”。
- **对策**：移除锐化/重对比；保留**轻** DiffAug，并**同步** real/fake 的参数。

### 1.5 Windows 多进程报错
- **对策**：在脚本底部：
  ```python
  if __name__ == "__main__":
      import torch.multiprocessing as mp
      mp.freeze_support()
      main()
  ```

---

## 2. 数据流与增广

```
读图(PIL, RGB, EXIF校正, RGBA→白底) 
→ Resize(72, BICUBIC, antialias) → CenterCrop(64) → RandomHorizontalFlip 
→ ToTensor → Normalize([-0.5..0.5])
```

- 训练时：`InstanceNoise`（热身期） + 轻 `DiffAug`（亮度/对比/饱和/平移）  
- 注意：**同一步**里 real/fake 使用**同一**组增广参数，保证对抗公平。

---

## 3. 训练循环（关键片段）

- **D-step**：`z → G(z)`（no_grad，仅 D-step 用）→ （real/fake + noise + 同步增广） → `D`  
  `d_loss = relu(1 - D(real)) + relu(1 + D(fake))`  
  每隔 `R1_INTERVAL` 步加：`0.5 * γ * interval * R1(real)`  
- **G-step**：解冻 G/冻结 D，`z → G(z)` → **带增广**的 `D(G(z))` → `g_loss = -mean(D(G(z)))`  
- **EMA**：`G_ema = decay * G_ema + (1 - decay) * G`；**用 G_ema 出图**  
- **TTUR 动作**（每 epoch）：
  - 统计：`D_real = sigmoid(D(real))`，`D_fake = sigmoid(D(fake))`，`margin = D_real - D_fake`
  - 若 `margin` 连续高于阈值（SMA 也高）：**D LR×0.5**；重复最多 N 次；仍高则**早停**。
- **采样/日志**：固定噪声，`SAVE_EVERY=50` 出图；CSV 记录指标。

---

## 4. 监控指标怎么读

| 指标              | 理想区间（经验） | 解释/动作                                                                 |
|-------------------|------------------|---------------------------------------------------------------------------|
| `D_real`(sigmoid) | 0.55 – 0.75      | 判别器对真图的置信。过高 → D 过强；过低 → D 学不好/增广过猛              |
| `D_fake`(sigmoid) | 0.30 – 0.45      | 判别器对假图的置信。过高 → G 太弱；过低 → D 太强                          |
| `margin`          | 0.10 – 0.60      | 拉得过高（>0.7）→ 触发 TTUR 降 D LR；短暂回落**正常**（平衡在恢复）       |
| `g_loss_epoch`    | 越小（更负）越好 | Hinge 下观测趋势用；挑 **最小点** 作为“最佳样张/权重”的参考               |

---

## 5. 常见症状 → 快速处方

| 症状 | 快速排查与修复 |
|---|---|
| **模式坍缩**（一堆相似脸） | 降 D LR（TTUR 再触发）/ 小幅增加 `σ0` / 减弱增广；增大 batch（若可） |
| **棋盘格/网纹** | 生成器改 **resize-conv**（已做）；末端再加 3×3 Conv；禁用强锐化 |
| **过白/过糊** | 降低增广范围/移除饱和度扰动；EMA 降一点；略增 G 通道 |
| **D_real≈D_fake≈0.5** 一直不动 | G/D 都弱或 LR 偏低：适当提高 LR 或扩大模型宽度；放宽增广 |
| **后期抖动** | 再降一次 D LR；EMA 调到 0.9997 出图 |

---

## 6. 推荐超参范围（本任务 64×64）

- `G_LR / D_LR`：`2e-4 / 1e-4`（基础），必要时 D 再降至 `5e-5 / 2.5e-5`
- `InstanceNoise σ0`：`0.02 ~ 0.05`；`warmup_ratio`：`0.1 ~ 0.3`
- `Lazy R1`：`γ = 5 ~ 10`，`interval = 8 ~ 32`
- `EMA decay`：`0.999 ~ 0.9997`
- `DiffAug`：亮度/对比/饱和 ±`0.06 ~ 0.08`；平移 ±`0.06 ~ 0.08`

---

## 7. 最佳点与导出

- 以 **EMA-G** 出图并**保存最佳点样张**（`best_fake_images_latest.png`）与对应权重（`generator_ema_best.pth`）。  
- 建议把“最佳点 epoch / g_loss_epoch / margin 均值”写入一个 `BEST.json`。

---

## 8. 复现实操（伪命令）

```bash
# 1) 数据放入 ./preprocessed_square_128_png/image
# 2) 训练
python main.py

# 3) 观察
#   - sample/fake_images_050.png, 100.png, ...（EMA-G）
#   - sample/best_fake_images_latest.png（最佳点样张）
#   - training_log.csv（折线图可自行画：D_real/D_fake/margin/g_loss）
```

---

## 9. 变更日志（从最初版到现在）

1. **数据几何修正**：`72→CenterCrop64` + BICUBIC/antialias；统一 RGB/方向/透明图合成  
2. **采样与日志**：统一每 50 epoch 出图、CSV 日志、保存最佳点  
3. **TTUR + InstanceNoise**：抑制 D 过强、平滑对抗动力学  
4. **轻 DiffAug**：同步 real/fake 参数  
5. **损失/结构升级**：BCE→Hinge，D 去 Sigmoid + SN，G 改 resize-conv，加入 Lazy R1  
6. **EMA**：只用 EMA 生成器出图与选优  
7. **Windows 兼容**：`if __name__ == '__main__': mp.freeze_support()`

---

## 10. 备注 & 小贴士

- 若样张始终偏噪：**先关掉饱和度扰动**，只留对比/平移；或把增广整体降 30%。  
- 若 64×64 跑稳，可把数据预裁到 128×128，并给 G/D 各加两级上/下采样；Lazy R1 的 `interval` 适当增大。  
- 固定 `fixed_noise` 用于跨 epoch 可比；保存时建议叠加薄灰框，方便肉眼对比。

---

**Happy training!** 这份 README 可随项目提交，后续你若升级到 128/256 或加标签条件（cGAN/ACGAN），建议继续在本文件追加“变更日志”。
