# Copyright (c) CAIRI AI Lab. All rights reserved

import argparse


def create_parser():
    parser = argparse.ArgumentParser(
        description='OpenSTL train/test a model')
    # 设置参数
    parser.add_argument('--device', default='cuda', type=str,
                        help='指定用于张量计算的设备名称(cuda/cpu)')
    parser.add_argument('--dist', action='store_true', default=False,
                        help='是否启用分布式训练(DDP)')
    parser.add_argument('--res_dir', default='work_dirs', type=str, help='指定结果保存目录')
    parser.add_argument('--ex_name', '-ex', default='Debug', type=str, help='指定实验名称')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='是否使用原生的 AMP(自动混合精度)进行混合精度训练(适用于 PyTorch 1.6.0 及以上版本)')
    parser.add_argument('--torchscript', action='store_true', default=False,
                        help='是否使用 TorchScript 模型。')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    parser.add_argument('--fps', action='store_true', default=False,
                        help='是否测量推理速度(Frames Per Second, FPS)')
    parser.add_argument('--test', action='store_true', default=False, help='是否仅进行测试')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='是否为 CUDNN 后端设置确定性选项（以提高可重复性）')

    # 数据集参数
    parser.add_argument('--batch_size', '-b', default=16, type=int, help='训练的批次大小。')
    parser.add_argument('--val_batch_size', '-vb', default=16, type=int, help='验证的批次大小。')
    parser.add_argument('--num_workers', default=4, type=int, help='数据加载器的工作线程数。')
    parser.add_argument('--data_root', default='/scratch/jianhao/fusion', type=str, help='数据集的根目录。')
    parser.add_argument('--dataname', '-d', default='mmnist', type=str,
                        choices=['bair', 'mfmnist', 'mmnist', 'mmnist_cifar', 'noisymmnist', 'taxibj', 'human',
                                'kth', 'kth20', 'kth40', 'kitticaltech', 'kinetics', 'kinetics400', 'kinetics600',
                                'weather', 'weather_t2m_5_625', 'weather_mv_4_28_s6_5_625', 'weather_mv_4_4_s6_5_625',
                                'weather_r_5_625', 'weather_uv10_5_625', 'weather_tcc_5_625', 'weather_t2m_1_40625',
                                'weather_r_1_40625', 'weather_uv10_1_40625', 'weather_tcc_1_40625',
                                'sevir_vis', 'sevir_ir069', 'sevir_ir107', 'sevir_vil','weather_tp', 'fusion', 'fusion_small'],
                        help='数据集名称 (默认: "mmnist")') #指定要使用的数据集，实验将根据此参数加载相应的数据集。
    parser.add_argument('--pre_seq_length', default=None, type=int, help='预测前的序列长度。') #用于设置输入序列的长度，在时间序列预测任务中确定输入序列的时间步数。
    parser.add_argument('--aft_seq_length', default=None, type=int, help='预测后的序列长度。') #用于设置输出序列的长度，决定模型要预测的时间步数。
    parser.add_argument('--total_length', default=None, type=int, help='预测的总序列长度。') #用于设置输入和输出序列的总长度，即输入序列和输出序列的总时间步数。
    parser.add_argument('--use_augment', action='store_true', default=False,
                        help='是否在训练中使用图像增强。') #启用后，数据加载器将在训练过程中对图像进行数据增强，提高模型的泛化能力。
    parser.add_argument('--use_prefetcher', action='store_true', default=False,
                        help='是否使用预取器以加快数据加载速度。') #启用后，数据加载过程将使用预取器技术，提高数据加载效率，减少训练过程中的数据等待时间。
    parser.add_argument('--drop_last', action='store_true', default=False,
                        help='是否在验证数据加载时丢弃最后一个不完整的批次。')

    # 方法参数
    parser.add_argument('--method', '-m', default='SimVP', type=str,
                        choices=['ConvLSTM', 'convlstm', 'E3DLSTM', 'e3dlstm', 'MAU', 'mau', 'MIM', 'mim', 
                                'PhyDNet', 'phydnet', 'PredRNN', 'predrnn', 'PredRNNpp',  'predrnnpp', 
                                'PredRNNv2', 'predrnnv2', 'SimVP', 'simvp', 'TAU', 'tau', 'MMVP', 'mmvp', 
                                'SwinLSTM', 'swinlstm', 'swinlstm_d', 'swinlstm_b'],
                        help='Name of video prediction method to train (default: "SimVP")')
    parser.add_argument('--config_file', '-c', default=None, type=str,
                        help='指定一个配置文件的路径，程序将从配置文件中加载参数。')
    parser.add_argument('--model_type', default=None, type=str,
                        help='这个参数指定 SimVP 模型的具体变体 (default: None)')
    parser.add_argument('--drop', type=float, default=0.0, help='这是训练过程中的 Dropout 概率，用于正则化，防止过拟合。')
    parser.add_argument('--drop_path', type=float, default=0.0, help='一种特殊的正则化技术，通过随机丢弃某些网络层或路径来减少过拟合')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='是否允许用args覆盖提供的配置文件')

    # Training parameters (optimizer)
    parser.add_argument('--epoch', '-e', default=200, type=int, help='end epochs (default: 200)')
    parser.add_argument('--log_step', default=1, type=int, help='Log interval by step')
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    parser.add_argument('--opt_eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer epsilon (default: None, use opt default)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer sgd momentum (default: 0.9)')
    parser.add_argument('--weight_decay', default=0., type=float, help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip_mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--no_display_method_info', action='store_true', default=False,
                        help='Do not display method info')

    # Training parameters (scheduler)
    parser.add_argument('--sched', default=None, type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "onecycle"')
    parser.add_argument('--lr', default=None, type=float, help='Learning rate (default: 1e-3)')
    parser.add_argument('--lr_k_decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=1e-5, metavar='LR',
                        help='warmup learning rate (default: 1e-5)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--final_div_factor', type=float, default=1e4,
                        help='min_lr = initial_lr/final_div_factor for onecycle scheduler')
    parser.add_argument('--warmup_epoch', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay_epoch', type=float, default=100, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--filter_bias_and_bn', type=bool, default=False,
                        help='Whether to set the weight decay of bias and bn to 0')

    # lightning
    parser.add_argument('--gpus', nargs='+', default=[1], type=int)
    parser.add_argument('--metric_for_bestckpt', default='val_loss', type=str)
    parser.add_argument('--ckpt_path', default=None, type=str)

    return parser


def default_parser():
    default_values = {
        # Set-up parameters
        'device': 'cuda',
        'dist': False,
        'res_dir': 'work_dirs',
        'ex_name': 'Debug',
        'fp16': False,
        'torchscript': False,
        'seed': 42,
        'fps': False,
        'test': False,
        'deterministic': False,
        # dataset parameters
        'batch_size': 16,
        'val_batch_size': 16,
        'num_workers': 4,
        'data_root': './data',
        'dataname': 'mmnist',
        'pre_seq_length': 10,
        'aft_seq_length': 10,
        'total_length': 20,
        'use_augment': False,
        'use_prefetcher': False,
        'drop_last': False,
        # method parameters
        'method': 'SimVP',
        'config_file': None,
        'model_type': 'gSTA',
        'drop': 0,
        'drop_path': 0,
        'overwrite': False,
        # Training parameters (optimizer)
        'epoch': 200,
        'log_step': 1,
        'opt': 'adam',
        'opt_eps': None,
        'opt_betas': None,
        'momentum': 0.9,
        'weight_decay': 0,
        'clip_grad': None,
        'clip_mode': 'norm',
        'no_display_method_info': False,
        # Training parameters (scheduler)
        'sched': 'onecycle',
        'lr': 1e-3,
        'lr_k_decay': 1.0,
        'warmup_lr': 1e-5,
        'min_lr': 1e-6,
        'final_div_factor': 1e4,
        'warmup_epoch': 0,
        'decay_epoch': 100,
        'decay_rate': 0.1,
        'filter_bias_and_bn': False,
        # Lightning parameters
        'gpus': [0],
        'metric_for_bestckpt': 'val_loss'
    }
    return default_values
