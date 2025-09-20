# Copyright (c) CAIRI AI Lab. All rights reserved

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import NearestNDInterpolator

from openstl.datasets.utils import create_loader


class FusionDataset(Dataset):
    """
    多源融合降水临近预报数据集 - 适配OpenSTL框架
    基于多源融合数据，仅使用第二个通道的降水数据
    
    Args:
        data_root (str): 数据根目录路径
        is_train (bool): 是否为训练模式
        n_frames_input (int): 输入帧数
        n_frames_output (int): 输出帧数
        csv_path (str): CSV索引文件路径
        data_split (str): 数据分割类型 ('train', 'valid', 'test')
        transform (callable, optional): 数据变换函数
        use_augment (bool): 是否使用数据增强
    """

    def __init__(self, data_root, is_train=True, n_frames_input=12, n_frames_output=12,
                 csv_path=None, data_split=None, transform=None, use_augment=False, 
                 downsample_ratio=1):
        super(FusionDataset, self).__init__()
        
        self.data_root = data_root
        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.total_frames = n_frames_input + n_frames_output
        self.transform = transform
        self.use_augment = use_augment
        self.downsample_ratio = downsample_ratio
        
        # 确定数据分割
        if data_split is None:
            self.data_split = 'train' if is_train else 'test'
        else:
            self.data_split = data_split
            
        # 确定CSV路径
        if csv_path is None:
            csv_path = os.path.join(data_root, '..', 'train_test', 'csv3')
        self.csv_path = csv_path
        
        # 数据集属性（OpenSTL标准）
        if downsample_ratio > 1:
            self.data_name = 'fusion_small'
        else:
            self.data_name = 'fusion'
        self.mean = 0.0
        self.std = 1.0
        
        # 加载数据索引
        self._load_data_index()
        
        # 预处理和构建样本
        self._preprocess_data()
        self._build_samples()
        
        print(f"降水数据集初始化完成 - {self.data_split}模式: {len(self.samples)}个样本")

    def _load_data_index(self):
        """加载CSV索引文件"""
        csv_file = os.path.join(self.csv_path, f'{self.data_split}.csv')
        
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV文件不存在: {csv_file}")
            
        self.data_info = pd.read_csv(csv_file, header=None)
        print(f"加载{self.data_split}数据索引: {len(self.data_info)}条记录")
        
        # 解析时间范围
        self.time_ranges = []
        for _, row in self.data_info.iterrows():
            start_time_str = str(row[0])
            end_time_str = str(row[1])
            
            start_time = datetime.strptime(start_time_str, "%Y%m%d%H%M")
            end_time = datetime.strptime(end_time_str, "%Y%m%d%H%M")
            
            self.time_ranges.append({
                'start_time': start_time,
                'end_time': end_time,
                'date_str': start_time.strftime("%Y%m%d")
            })

    def _preprocess_data(self):
        """预处理数据"""
        self.processed_data = {}
        self.data_masks = {}
        
        # 获取所有需要加载的日期
        unique_dates = set([tr['date_str'] for tr in self.time_ranges])
        print(f"预处理{self.data_split}数据: {len(unique_dates)}个日期")
        
        for date_str in unique_dates:
            year = date_str[:4]
            data_file = os.path.join(self.data_root, year, f"fusion_{date_str}.npy")
            
            try:
                # 加载原始数据 (144, 4, 500, 900)
                daily_data = np.load(data_file)
                
                # 只提取第二个通道的降水数据
                precipitation_data = daily_data[:, 1:2, :, :]  # (144, 1, 500, 900)
                
                # 处理缺失值：将-999、NaN和负值都设置为0
                precipitation_data[precipitation_data < -100] = 0.0  # 处理-999等缺失值标记
                precipitation_data[precipitation_data < 0] = 0.0     # 降水不能为负值
                precipitation_data[np.isnan(precipitation_data)] = 0.0  # NaN值设为0
                
                # 转换为torch张量
                filled_data = torch.from_numpy(precipitation_data).float()
                
                # 下采样处理
                if self.downsample_ratio > 1:
                    # 使用平均池化进行下采样
                    import torch.nn.functional as F
                    # 确保尺寸能被整除，先裁剪到能被downsample_ratio整除的尺寸
                    h_new = (500 // self.downsample_ratio) * self.downsample_ratio  # 500 -> 500 (4倍) 或 496 (其他)
                    w_new = (900 // self.downsample_ratio) * self.downsample_ratio  # 900 -> 900 (4倍) 或 896 (其他)
                    
                    # 裁剪数据
                    filled_data = filled_data[:, :, :h_new, :w_new]  # (144, 1, h_new, w_new)
                    
                    # 下采样
                    filled_data = F.avg_pool2d(
                        filled_data.view(-1, 1, h_new, w_new), 
                        kernel_size=self.downsample_ratio, 
                        stride=self.downsample_ratio
                    ).view(144, 1, h_new//self.downsample_ratio, w_new//self.downsample_ratio)
                
                data_mask = (filled_data == 0.0).bool()
                
                # 存储处理后的数据
                self.processed_data[date_str] = filled_data
                self.data_masks[date_str] = data_mask
                
                print(f"已处理 {date_str} 数据，形状: {filled_data.shape}")
                
            except Exception as e:
                print(f"加载文件 {data_file} 失败: {e}")
                # 创建空数据
                self.processed_data[date_str] = torch.zeros((144, 1, 500, 900), dtype=torch.float32)
                self.data_masks[date_str] = torch.ones((144, 1, 500, 900), dtype=torch.bool)

    def _interpolate_nans(self, data):
        """使用最近邻插值填充NaN值"""
        for t in range(data.shape[0]):
            for c in range(data.shape[1]):
                frame = data[t, c]
                
                if np.any(np.isnan(frame)):
                    # 获取有效点的坐标和值
                    y_indices, x_indices = np.where(~np.isnan(frame))
                    
                    if len(y_indices) > 3:
                        points = np.column_stack((y_indices, x_indices))
                        values = frame[~np.isnan(frame)]
                        
                        # 获取NaN位置
                        nan_y_indices, nan_x_indices = np.where(np.isnan(frame))
                        nan_points = np.column_stack((nan_y_indices, nan_x_indices))
                        
                        try:
                            # 最近邻插值
                            interpolator = NearestNDInterpolator(points, values)
                            frame[nan_y_indices, nan_x_indices] = interpolator(nan_points)
                        except Exception:
                            frame[np.isnan(frame)] = 0.0
                    else:
                        frame[np.isnan(frame)] = 0.0
        
        # 确保没有NaN值
        data = np.nan_to_num(data, nan=0.0)
        return data

    def _build_samples(self):
        """构建训练样本"""
        self.samples = []
        
        for time_range in self.time_ranges:
            date_str = time_range['date_str']
            start_time = time_range['start_time']
            end_time = time_range['end_time']
            
            # 计算时间索引
            day_start = datetime(start_time.year, start_time.month, start_time.day, 0, 0)
            start_idx = int((start_time - day_start).total_seconds() / 600)  # 10分钟间隔
            end_idx = int((end_time - day_start).total_seconds() / 600)
            
            # 生成滑动窗口样本
            max_start_idx = end_idx - self.total_frames + 1
            
            for sample_start in range(start_idx, min(max_start_idx + 1, 144 - self.total_frames + 1)):
                sample = {
                    'date_str': date_str,
                    'start_idx': sample_start,
                    'input_indices': list(range(sample_start, sample_start + self.n_frames_input)),
                    'target_indices': list(range(sample_start + self.n_frames_input, 
                                                sample_start + self.total_frames))
                }
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        date_str = sample['date_str']
        
        # 获取预处理的数据
        daily_data = self.processed_data[date_str]
        
        # 提取输入序列
        input_frames = daily_data[sample['input_indices']]  # (n_frames_input, 1, 500, 900)
        
        # 提取目标序列
        target_frames = daily_data[sample['target_indices']]  # (n_frames_output, 1, 500, 900)
        
        # 数据增强
        if self.use_augment and self.is_train:
            input_frames, target_frames = self._apply_augmentation(input_frames, target_frames)
        
        # 应用变换
        if self.transform:
            input_frames = self.transform(input_frames)
            target_frames = self.transform(target_frames)
        
        return input_frames, target_frames

    def _apply_augmentation(self, input_frames, target_frames):
        """应用数据增强"""
        # 随机水平翻转
        if torch.rand(1) < 0.5:
            input_frames = torch.flip(input_frames, [-1])
            target_frames = torch.flip(target_frames, [-1])
        
        # 随机垂直翻转
        if torch.rand(1) < 0.5:
            input_frames = torch.flip(input_frames, [-2])
            target_frames = torch.flip(target_frames, [-2])
        
        return input_frames, target_frames


def load_data(batch_size, val_batch_size, data_root, num_workers=4, **kwargs):
    """
    加载融合数据集 - OpenSTL标准接口
    
    Args:
        batch_size (int): 训练批次大小
        val_batch_size (int): 验证批次大小
        data_root (str): 数据根目录
        num_workers (int): 数据加载进程数
        **kwargs: 其他参数
        
    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    
    # 从kwargs中提取参数
    pre_seq_length = kwargs.get('pre_seq_length', 12)
    aft_seq_length = kwargs.get('aft_seq_length', 12)
    csv_path = kwargs.get('csv_path', None)
    use_augment = kwargs.get('use_augment', False)
    downsample_ratio = kwargs.get('downsample_ratio', 1)
    
    # 创建数据集
    train_set = FusionDataset(
        data_root=data_root,
        is_train=True,
        n_frames_input=pre_seq_length,
        n_frames_output=aft_seq_length,
        csv_path=csv_path,
        data_split='train',
        use_augment=use_augment,
        downsample_ratio=downsample_ratio
    )
    
    valid_set = FusionDataset(
        data_root=data_root,
        is_train=False,
        n_frames_input=pre_seq_length,
        n_frames_output=aft_seq_length,
        csv_path=csv_path,
        data_split='valid',
        use_augment=False,
        downsample_ratio=downsample_ratio
    )
    
    test_set = FusionDataset(
        data_root=data_root,
        is_train=False,
        n_frames_input=pre_seq_length,
        n_frames_output=aft_seq_length,
        csv_path=csv_path,
        data_split='test',
        use_augment=False,
        downsample_ratio=downsample_ratio
    )
    
    # 创建数据加载器
    dataloader_train = create_loader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        is_training=True,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers
    )
    
    dataloader_valid = create_loader(
        valid_set,
        batch_size=val_batch_size,
        shuffle=False,
        is_training=False,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers
    )
    
    dataloader_test = create_loader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        is_training=False,
        pin_memory=True,
        drop_last=True,
        num_workers=num_workers
    )
    
    return dataloader_train, dataloader_valid, dataloader_test


if __name__ == "__main__":
    # 测试代码
    data_root = "/scratch/jianhao/fusion"
    csv_path = "/home/jianhao/methods/OPENSTL/train_test/csv3"
    
    # 创建数据集
    dataset = FusionDataset(
        data_root=data_root,
        is_train=True,
        n_frames_input=12,
        n_frames_output=12,
        csv_path=csv_path,
        data_split='train'
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 获取一个样本
    if len(dataset) > 0:
        input_frames, target_frames = dataset[0]
        print(f"输入帧形状: {input_frames.shape}")
        print(f"目标帧形状: {target_frames.shape}")
        print(f"输入帧数据范围: [{input_frames.min():.3f}, {input_frames.max():.3f}]")
        print(f"目标帧数据范围: [{target_frames.min():.3f}, {target_frames.max():.3f}]")