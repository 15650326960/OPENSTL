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
# 针对fusion数据集的高分辨率(500x900)进行优化
num_hidden = '32,64,64,32'  # 4层ConvLSTM，渐进式隐藏单元数量
filter_size = 5             # 卷积核大小
stride = 1                  # 卷积步长
patch_size = 5              # 图像块大小，适配250x450分辨率 (250%5=0, 450%5=0)
layer_norm = 0              # 不使用层归一化

# training parameters - 训练参数
lr = 1e-4                   # 学习率
batch_size = 1              # 小批次大小，适配高分辨率数据
sched = 'onecycle'          # 学习率调度器
warmup_epoch = 2            # 预热轮数

# data specific - 数据相关配置
data_root = '/scratch/jianhao/fusion'
csv_path = '/home/jianhao/methods/OPENSTL/train_test/csv3'
use_augment = False         # 不使用数据增强