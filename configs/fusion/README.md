# Fusion数据集配置说明

## 概述

这里包含了用于训练fusion数据集的各种模型配置文件。fusion数据集是多源融合降水临近预报数据集，具有500×900的高分辨率。

## 可用配置

### SimVP系列

1. **SimVP_gSTA.py** - 推荐配置
   - 使用gSTA (generalized STA) 架构
   - 适合高分辨率数据
   - batch_size = 4

2. **SimVP_IncepU.py**
   - 使用Inception-U架构
   - 计算效率较高
   - batch_size = 4

3. **SimVP_ConvNeXt.py**
   - 使用ConvNeXt架构
   - 更强的表征能力
   - batch_size = 3 (内存占用较大)

4. **SimVP_Swin.py**
   - 使用Swin Transformer架构
   - 长距离依赖建模能力强
   - batch_size = 3 (内存占用较大)

## 使用方法

### 基础训练命令

```bash
# 使用推荐的gSTA配置
python tools/train.py \
    -d fusion \
    -c configs/fusion/simvp/SimVP_gSTA.py \
    --ex_name fusion_simvp_gsta

# 使用IncepU配置
python tools/train.py \
    -d fusion \
    -c configs/fusion/simvp/SimVP_IncepU.py \
    --ex_name fusion_simvp_incepu

# 使用ConvNeXt配置
python tools/train.py \
    -d fusion \
    -c configs/fusion/simvp/SimVP_ConvNeXt.py \
    --ex_name fusion_simvp_convnext

# 使用Swin配置
python tools/train.py \
    -d fusion \
    -c configs/fusion/simvp/SimVP_Swin.py \
    --ex_name fusion_simvp_swin
```

### 自定义参数

```bash
# 修改学习率
python tools/train.py \
    -d fusion \
    -c configs/fusion/simvp/SimVP_gSTA.py \
    --lr 5e-4 \
    --ex_name fusion_custom_lr

# 修改batch size (如果GPU内存不足)
python tools/train.py \
    -d fusion \
    -c configs/fusion/simvp/SimVP_gSTA.py \
    --batch_size 2 \
    --ex_name fusion_small_batch

# 修改训练轮数
python tools/train.py \
    -d fusion \
    -c configs/fusion/simvp/SimVP_gSTA.py \
    --epoch 100 \
    --ex_name fusion_long_train
```

## 配置参数说明

### 模型参数
- `model_type`: SimVP的架构类型 ('gSTA', 'IncepU', 'ConvNeXt', 'Swin')
- `hid_S`: 空间隐藏层维度 (64)
- `hid_T`: 时间隐藏层维度 (512)
- `N_T`: 时间层数 (8)
- `N_S`: 空间层数 (4)

### 训练参数
- `lr`: 学习率 (1e-3 或 5e-4)
- `batch_size`: 批次大小 (2-4, 根据GPU内存调整)
- `drop_path`: DropPath率 (0.1-0.2)
- `sched`: 学习率调度器 ('onecycle' 或 'cosine')
- `warmup_epoch`: 预热轮数 (5-10)

### 数据参数
- `data_root`: 数据根目录 ('/scratch/jianhao/fusion')
- `csv_path`: CSV索引文件路径 ('/home/jianhao/methods/OPENSTL/train_test/csv3')
- `use_augment`: 是否使用数据增强 (False)

## 性能建议

### GPU内存优化
- **16GB GPU**: 使用batch_size=4的配置 (gSTA, IncepU)
- **12GB GPU**: 使用batch_size=3的配置 (ConvNeXt, Swin)
- **8GB GPU**: 将所有配置的batch_size改为2

### 训练速度优化
- 推荐顺序：IncepU > gSTA > ConvNeXt > Swin
- IncepU计算效率最高，适合快速实验
- gSTA平衡了性能和效率，是推荐选择
- ConvNeXt和Swin性能更强但训练较慢

### 预期性能
基于融合数据集的特点：
- **训练时间**: 每轮约30-60分钟 (取决于配置和硬件)
- **收敛轮数**: 通常50-100轮
- **内存占用**: 8-16GB GPU显存

## 故障排除

### 常见错误

1. **CUDA OOM错误**
   ```bash
   # 解决方案：减小batch_size
   --batch_size 2
   ```

2. **数据路径错误**
   ```bash
   # 检查数据路径是否正确
   ls /scratch/jianhao/fusion/2021/
   ls /home/jianhao/methods/OPENSTL/train_test/csv3/
   ```

3. **配置文件错误**
   ```bash
   # 确保使用正确的数据集名称
   -d fusion  # 不是 -d precipitation
   ```

## 实验建议

1. **首次运行**: 建议从SimVP_gSTA.py开始
2. **快速实验**: 使用SimVP_IncepU.py
3. **追求性能**: 尝试SimVP_ConvNeXt.py或SimVP_Swin.py
4. **资源有限**: 将batch_size改为2，epoch改为50