method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'Swin'
hid_S = 64
hid_T = 512
N_T = 8
N_S = 4
# training
lr = 1e-3
batch_size = 3  # Swin Transformer内存占用较大
drop_path = 0.15
sched = 'cosine'
warmup_epoch = 10
# data specific
data_root = '/scratch/jianhao/fusion'
csv_path = '/home/jianhao/methods/OPENSTL/train_test/csv3'
use_augment = False