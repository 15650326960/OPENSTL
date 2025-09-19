method = 'SimVP'
# model - 使用IncepU架构，内存效率更高
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'IncepU'
hid_S = 32
hid_T = 256
N_T = 4
N_S = 2
# training
lr = 1e-3
batch_size = 1
drop_path = 0.05
sched = 'onecycle'
warmup_epoch = 2
# data specific
data_root = '/scratch/jianhao/fusion'
csv_path = '/home/jianhao/methods/OPENSTL/train_test/csv3'
use_augment = False