method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'ConvNeXt'
hid_S = 64
hid_T = 512
N_T = 8
N_S = 4
# training
lr = 5e-4  # ConvNeXt通常使用较小的学习率
batch_size = 3  # ConvNeXt内存占用较大
drop_path = 0.2
sched = 'cosine'
warmup_epoch = 10
# data specific
data_root = '/scratch/jianhao/fusion'
csv_path = '/home/jianhao/methods/OPENSTL/train_test/csv3'
use_augment = False