method = 'SimVP'
# model - 减小模型参数以节省显存
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 32  # 从64减少到32
hid_T = 256  # 从512减少到256
N_T = 4  # 从8减少到4
N_S = 2  # 从4减少到2
# training
lr = 1e-4
batch_size = 1  # 使用最小的batch size
drop_path = 0.05  # 减小drop_path
sched = 'onecycle'
warmup_epoch = 2
# data specific
data_root = '/scratch/jianhao/fusion'
csv_path = '/home/jianhao/methods/OPENSTL/train_test/csv3'
use_augment = False