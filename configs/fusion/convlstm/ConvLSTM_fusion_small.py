method = 'ConvLSTM'

# reverse scheduled sampling - 反向计划采样
reverse_scheduled_sampling = 0
r_sampling_step_1 = 25000
r_sampling_step_2 = 50000
r_exp_alpha = 5000

# scheduled sampling - 计划采样策略
scheduled_sampling = 1
sampling_stop_iter = 50000
sampling_start_value = 1.0
sampling_changing_rate = 0.00002

# model parameters - 模型参数
# ConvLSTM层的隐藏单元数量，适配fusion数据集的高分辨率
num_hidden = '64,64,64,64'  # 4层ConvLSTM，每层64个隐藏单元
filter_size = 5             # 卷积核大小
stride = 1                  # 卷积步长
patch_size = 5              # 图像块大小，适配250x450分辨率 (250%5=0, 450%5=0)
layer_norm = 0              # 是否使用层归一化

# training parameters - 训练参数
lr = 1e-3                   # 学习率，与SimVP_gSTA保持一致
batch_size = 2              # 批次大小，考虑到ConvLSTM的内存占用
sched = 'onecycle'          # 学习率调度器
warmup_epoch = 2            # 预热轮数
drop_path = 0.1             # dropout路径比例

# data specific - 数据相关配置
data_root = '/scratch/jianhao/fusion'
csv_path = '/home/jianhao/methods/OPENSTL/train_test/csv3'
use_augment = False         # 不使用数据增强，保持与SimVP_gSTA一致

# optimization - 优化配置
epoch = 2                  # 训练轮数
weight_decay = 1e-4         # 权重衰减