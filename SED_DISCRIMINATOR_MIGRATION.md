# SeD 判别器迁移计划

## 目标

在保留 HYPIR 当前 ImageConvNextDiscriminator 的基础上，**新增** SeD（Semantic-aware Discriminator）作为补充判别器，引入基于 CLIP 语义特征的交叉注意力判别机制，提升生成器对语义一致性的学习能力。

## 架构对比

| 项目 | 当前 HYPIR 判别器 | 新增 SeD 判别器 |
|------|-------------------|-----------------|
| 骨干网络 | 冻结 ConvNeXt-XXL (~850M) | 冻结 CLIP RN50 (~38M) |
| 可训练部分 | MultiLevelDConv (~数M) | SeD_P (~5-10M) |
| 输入 | 单输入 (图像) | 双输入 (图像, GT语义特征) |
| 判别粒度 | 多层级 (4级) | 单尺度 PatchGAN |
| 语义引导 | 无 | 有 (交叉注意力 SeFB) |
| GAN损失 | Multilevel Hinge Loss | Vanilla BCE Loss |
| 操作分辨率 | 512x512 (全分辨率) | 256x256 (resize后) |
| 角色 | 主判别器 (全分辨率细节) | 补充判别器 (语义一致性) |

## 方案：保留原 D + 新增 SeD（双判别器）

不替换原有判别器，而是同时使用两个判别器：

```
训练流程:

512x512 GT ──→ ConvNeXt D ──→ hinge loss (λ=0.5)     ← 保留不变
512x512 SR ──→ ConvNeXt D ──→ hinge loss

512x512 GT ──→ resize 256 ──→ CLIP RN50 ──→ semantic (1024ch, 16x16)
512x512 GT ──→ resize 256 ──┐
512x512 SR ──→ resize 256 ──┤
                             ↓
                      SeD_P(sr_256, semantic) → BCE loss (λ=0.05)  ← 新增
                      SeD_P(gt_256, semantic) → BCE loss
```

## 损失函数

```
L_G = 1.0 × L_mse + 5.0 × L_lpips + 0.5 × L_adv_convnext + 0.05 × L_adv_sed
```

| 损失项 | 系数 | 来源 | 说明 |
|--------|------|------|------|
| L_mse | 1.0 | HYPIR 原始 | 不变 |
| L_lpips | 5.0 | HYPIR 原始 | 不变 |
| L_adv_convnext | 0.5 | HYPIR 原始 D | 不变 |
| L_adv_sed | **0.05** | 新增 SeD_P | 参考 SeD 源码 loss_g=0.01，结合 HYPIR loss scale 放大至 0.05 |

系数选择依据：
- SeD 原始训练 config（`options/train_rrdb_P+SeD.yml`）: `loss_g: 0.01`
- HYPIR 原始 lambda_gan=0.5，整体 loss scale 远大于 SeD 原始设置
- 0.05 = ConvNeXt D 权重的 1/10，SeD 作为语义正则化器而非主导
- 总对抗权重 0.55，增幅仅 10%，训练稳定性风险低

## 文件变更清单

### 新增文件
- `HYPIR/model/sed_discriminator.py` — CLIP 语义提取器 + SeD_P 判别器
- `HYPIR/model/module_attention.py` — SeFB 交叉注意力模块

### 修改文件
- `HYPIR/trainer/base.py` — 新增 init_sed_discriminator、修改 optimize_generator / optimize_discriminator / init_optimizers / prepare_all / log_grads
- `configs/sd2_train.yaml` — 新增 SeD 判别器配置项（lambda_gan_sed, sed_type, sed_resize）

### 不修改的文件
- `HYPIR/model/D.py` — 原始 ImageConvNextDiscriminator 保留不动
- `HYPIR/model/backbone.py` — ConvNeXt 骨干保留不动
- `HYPIR/trainer/sd2.py` — 判别器逻辑全在 base.py
- `requirements.txt` — clip、einops 已安装

## 任务阶段

### Phase 1: 代码移植

- [ ] T1.1 从 basecode/SeD 复制 module_attention.py → HYPIR/model/module_attention.py（移除未使用类）
- [ ] T1.2 从 basecode/SeD 复制 sed.py → HYPIR/model/sed_discriminator.py（修改 import 路径）
- [ ] T1.3 单元测试：各模块可正常实例化，forward 输出形状正确

### Phase 2: 训练器改造

- [ ] T2.1 BaseTrainer.init_models() 新增 init_sed_discriminator() 调用
- [ ] T2.2 新增 init_sed_discriminator() — 创建 semantic_extractor + SeD_P
- [ ] T2.3 新增 _prepare_sed_inputs() — resize + 语义特征提取
- [ ] T2.4 修改 init_optimizers() — 新增 D_sed 优化器
- [ ] T2.5 修改 prepare_all() — 扩展 attrs 列表
- [ ] T2.6 修改 optimize_generator() — 新增 SeD GAN loss
- [ ] T2.7 修改 optimize_discriminator() — 新增 SeD D 训练
- [ ] T2.8 修改 log_grads() — 新增 SeD 梯度监控
- [ ] T2.9 集成测试：完整 forward → backward 可运行，无 shape 错误

### Phase 3: 配置

- [ ] T3.1 sd2_train.yaml 新增 lambda_gan_sed, sed_type, sed_resize 配置项

### Phase 4: 验证

- [ ] T4.1 模块实例化测试（CLIP extractor + SeD_P 输出形状）
- [ ] T4.2 Resize 兼容性测试（512→256 完整数据流）
- [ ] T4.3 训练冒烟测试（10 步无报错，loss_dict 包含 G_sed / D_sed）
- [ ] T4.4 显存监控（预期新增 ~250MB）

### Phase 5: 消融实验

- [ ] T5.1 原始 HYPIR D 基线指标
- [ ] T5.2 +SeD λ=0.05 指标
- [ ] T5.3 +SeD λ=0.1 指标
- [ ] T5.4 结果对比表 (PSNR / SSIM / LPIPS / FID)

## 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| resize 导致细节判别丢失 | 生成纹理不够精细 | ConvNeXt D 在全分辨率补偿 |
| CLIP RN50 额外显存 | ~200MB | 冻结 + bf16 可控 |
| SeD vanilla GAN 训练不稳定 | 模式崩塌 | 监控 D_sed logits，必要时换 hinge loss |
| 双 D 梯度冲突 | 训练不收敛 | SeD 权重小(0.05)，作为正则化器 |
| resume checkpoint 不兼容 | 无法恢复训练 | SeD 新增参数随机初始化 |

## 决策记录

| 日期 | 决策 | 原因 |
|------|------|------|
| 2026-04-24 | 采用 resize 方案适配分辨率 | SeD 空间尺寸硬编码，resize 改动最小 |
| 2026-04-24 | 保留原 D + 新增 SeD 双判别器方案 | 保留全分辨率判别能力，SeD 提供语义补充 |
| 2026-04-24 | SeD GAN loss 权重 0.05 | 参考 SeD 源码 loss_g=0.01，适配 HYPIR scale |
