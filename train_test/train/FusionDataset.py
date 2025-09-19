import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import glob
from datetime import datetime, timedelta

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

class FusionDataset(Dataset):
    """
    多源数据融合降水临近预报数据集
    输入：过去n帧的多源数据
    输出：未来m帧的预测数据
    """
    def __init__(self, configs, mode='train', year_filter=None):
        """
        初始化数据集
        Args:
            configs: 配置参数
            mode: 'train', 'valid', 或 'test'
            year_filter: 年份过滤器，用于分批加载训练数据，例如[2021, 2022]
        """
        self.configs = configs
        self.mode = mode
        self.year_filter = year_filter

        # 基础数据路径
        self.base_data_path = configs.get('data_path', '/scratch/jianhao/fusion')
        print(f"使用{mode}数据基础路径: {self.base_data_path}")

        self.csv_path = configs.get('csv_path', '/home/jianhao/methods/single-source/train_test/csv3')  # 获取CSV路径
        self.input_length = configs['input_length']  # 输入序列长度
        self.output_length = configs['output_length']  # 输出序列长度

        # 加载相应的CSV文件
        csv_file = os.path.join(self.csv_path, f'{mode}.csv')
        if os.path.exists(csv_file):
            self.data_info = pd.read_csv(csv_file, header=None)
            print(f"已加载{mode}数据集CSV: {len(self.data_info)}个样本，CSV路径: {csv_file}")
            
            # 打印CSV文件详细信息
            self.print_csv_info(csv_file, mode)
        else:
            raise FileNotFoundError(f"CSV文件不存在: {csv_file}")

        # 如果是训练模式且指定了年份过滤器，则过滤数据
        if mode == 'train' and year_filter is not None:
            self.filter_data_by_years(year_filter)

        # 为每一行CSV数据预处理有效索引范围（临时用于数据加载）
        self.valid_ranges = []
        for i, row in self.data_info.iterrows():
            start_time_str = str(row[0])
            end_time_str = str(row[1])

            # 解析日期和时间
            start_time = datetime.strptime(start_time_str, "%Y%m%d%H%M")
            end_time = datetime.strptime(end_time_str, "%Y%m%d%H%M")

            # 计算开始和结束索引
            day_start = datetime(start_time.year, start_time.month, start_time.day, 0, 0)
            start_minutes = int((start_time - day_start).total_seconds() / 60)
            end_minutes = int((end_time - day_start).total_seconds() / 60)

            start_idx = start_minutes // 10  # 每10分钟一个样本
            end_idx = end_minutes // 10

            # 确保有足够的序列长度
            total_required_frames = self.input_length + self.output_length
            max_start_idx = end_idx - total_required_frames + 1

            if max_start_idx < start_idx:
                print(f"警告：行{i}的时间范围不足以提取完整序列，跳过")
                self.valid_ranges.append(None)
            else:
                self.valid_ranges.append((start_idx, max_start_idx, start_time.strftime("%Y%m%d")))

        # 过滤掉无效的范围
        valid_indices = [i for i, r in enumerate(self.valid_ranges) if r is not None]
        if len(valid_indices) < len(self.data_info):
            print(f"警告：有{len(self.data_info) - len(valid_indices)}行数据因时间范围不足被过滤")
            self.data_info = self.data_info.iloc[valid_indices].reset_index(drop=True)
            self.valid_ranges = [r for r in self.valid_ranges if r is not None]

        # 预加载所有数据到内存
        self.processed_data = {}  # 存储预处理后的数据
        self.data_masks = {}      # 存储NaN掩码

        # 获取所有需要加载的日期
        unique_dates = set([date_str for _, _, date_str in self.valid_ranges])
        print(f"开始预加载{mode}数据集，共{len(unique_dates)}个日期...")
        
        for date_str in unique_dates:
            # 根据日期确定年份和对应的数据路径
            year = date_str[:4]
            data_file = os.path.join(self.base_data_path, year, f"fusion_{date_str}.npy")
            try:
                # 加载原始数据
                daily_data = np.load(data_file)  # 形状应为 (144, 4, 500, 900)

                # 创建NaN掩码（True表示是NaN）


                # 使用时空插值填充NaN
                filled_data = self.interpolate_nans(daily_data.copy())

                # 转换为torch张量
                filled_data = torch.from_numpy(filled_data).float()
                num_weight = (filled_data > -100).float()
                filled = filled_data * num_weight
                filled_mask = (filled == 0.0).bool()

                # 存储预处理后的数据和掩码
                self.processed_data[date_str] = filled
                self.data_masks[date_str] = filled_mask
                print(f"已加载并预处理 {date_str} 数据，形状: {filled_data.shape}")
            except Exception as e:
                print(f"加载文件 {data_file} 失败: {e}")
                # 创建空数据和掩码
                self.processed_data[date_str] = torch.zeros((144, 4, 500, 900), dtype=torch.float32)
                self.data_masks[date_str] = torch.ones((144, 4, 500, 900), dtype=torch.bool)  # 全部标记为NaN
        
        print(f"{mode}数据集预加载完成，共{len(self.processed_data)}个日期的数据")

        # 分析数据连续性并构建滑动窗口样本
        self.analyze_data_continuity()
        self.build_continuous_timeline()
        self.generate_sliding_samples()

    def print_csv_info(self, csv_file, mode):
        """
        打印CSV文件详细信息到终端
        Args:
            csv_file: CSV文件路径
            mode: 数据集模式 ('train', 'valid', 'test')
        """
        try:
            # 获取文件信息
            file_size = os.path.getsize(csv_file) / 1024  # KB
            file_stat = os.stat(csv_file)
            last_modified = datetime.fromtimestamp(file_stat.st_mtime)
            
            # 读取CSV内容进行详细分析
            df = pd.read_csv(csv_file, header=None)
            total_samples = len(df)
            
            # 分析时间范围
            if len(df) > 0:
                start_times = df[0].astype(str)
                end_times = df[1].astype(str)
                
                # 解析时间范围
                start_dates = [datetime.strptime(str(t), "%Y%m%d%H%M") for t in start_times]
                end_dates = [datetime.strptime(str(t), "%Y%m%d%H%M") for t in end_times]
                
                min_date = min(start_dates)
                max_date = max(end_dates)
                date_range = f"{min_date.strftime('%Y-%m-%d %H:%M')} 到 {max_date.strftime('%Y-%m-%d %H:%M')}"
                
                # 统计年份分布
                years = [d.year for d in start_dates]
                year_counts = pd.Series(years).value_counts().sort_index()
                year_distribution = ", ".join([f"{year}: {count}" for year, count in year_counts.items()])
            else:
                date_range = "无数据"
                year_distribution = "无数据"
            
            # 打印详细信息到终端
            print("=" * 80)
            print(f"📊 {mode.upper()} 数据集CSV文件详细信息")
            print("=" * 80)
            print(f"📁 文件路径: {csv_file}")
            print(f"📏 文件大小: {file_size:.2f} KB")
            print(f"📅 最后修改: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📊 样本总数: {total_samples:,}")
            print(f"📅 时间范围: {date_range}")
            print(f"📈 年份分布: {year_distribution}")
            print("=" * 80)
            
            # 打印CSV文件内容
            print(f"📄 {mode.upper()} 数据集CSV文件内容:")
            print("-" * 80)
            print("所有数据:")
            print(df.to_string(index=False))
            print("-" * 80)
            
        except Exception as e:
            print(f"⚠️  打印CSV文件信息时出错: {e}")

    def filter_data_by_years(self, year_filter):
        """
        根据年份过滤数据
        Args:
            year_filter: 年份列表，例如[2021, 2022]
        """
        print(f"正在过滤数据，只保留年份: {year_filter}")
        original_count = len(self.data_info)

        # 过滤数据
        filtered_indices = []
        for i, row in self.data_info.iterrows():
            start_time_str = str(row[0])
            year = int(start_time_str[:4])
            if year in year_filter:
                filtered_indices.append(i)

        self.data_info = self.data_info.iloc[filtered_indices].reset_index(drop=True)
        print(f"过滤完成：原始{original_count}条数据，过滤后{len(self.data_info)}条数据")

    def clear_data(self):
        """
        清空已加载的数据，释放内存
        """
        print("正在清空数据集内存...")
        # 显式删除数据引用
        for key in list(self.processed_data.keys()):
            del self.processed_data[key]
        for key in list(self.data_masks.keys()):
            del self.data_masks[key]

        self.processed_data.clear()
        self.data_masks.clear()

        # 强制垃圾回收
        import gc
        gc.collect()
        print("数据集内存已清空")

    def reload_data_with_years(self, year_filter):
        """
        重新加载指定年份的数据
        Args:
            year_filter: 年份列表，例如[2023, 2024, 2025]
        """
        print(f"重新加载年份数据: {year_filter}")

        # 清空当前数据
        self.clear_data()

        # 重新加载CSV并过滤
        csv_file = os.path.join(self.csv_path, f'{self.mode}.csv')
        self.data_info = pd.read_csv(csv_file, header=None)
        
        # 打印重新加载的CSV文件信息
        print(f"🔄 重新加载数据，年份过滤器: {year_filter}")
        self.print_csv_info(csv_file, f"{self.mode}_reloaded")
        
        self.filter_data_by_years(year_filter)
        for i, row in self.data_info.iterrows():
            start_time_str = str(row[0])
            end_time_str = str(row[1])

            # 解析日期和时间
            start_time = datetime.strptime(start_time_str, "%Y%m%d%H%M")
            end_time = datetime.strptime(end_time_str, "%Y%m%d%H%M")

            # 计算开始和结束索引
            day_start = datetime(start_time.year, start_time.month, start_time.day, 0, 0)
            start_minutes = int((start_time - day_start).total_seconds() / 60)
            end_minutes = int((end_time - day_start).total_seconds() / 60)

            start_idx = start_minutes // 10  # 每10分钟一个样本
            end_idx = end_minutes // 10

            # 确保有足够的序列长度
            total_required_frames = self.input_length + self.output_length
            max_start_idx = end_idx - total_required_frames + 1

            if max_start_idx < start_idx:
                print(f"警告：行{i}的时间范围不足以提取完整序列，跳过")
                self.valid_ranges.append(None)
            else:
                self.valid_ranges.append((start_idx, max_start_idx, start_time.strftime("%Y%m%d")))

        # 过滤掉无效的范围
        valid_indices = [i for i, r in enumerate(self.valid_ranges) if r is not None]
        if len(valid_indices) < len(self.data_info):
            print(f"警告：有{len(self.data_info) - len(valid_indices)}行数据因时间范围不足被过滤")
            self.data_info = self.data_info.iloc[valid_indices].reset_index(drop=True)
            self.valid_ranges = [r for r in self.valid_ranges if r is not None]

        # 重新加载数据
        self.processed_data = {}
        self.data_masks = {}

        # 获取所有需要加载的日期
        unique_dates = set([date_str for _, _, date_str in self.valid_ranges])
        print(f"开始重新加载数据，共{len(unique_dates)}个日期...")

        for date_str in unique_dates:
            # 根据日期确定年份和对应的数据路径
            year = date_str[:4]
            data_file = os.path.join(self.base_data_path, year, f"fusion_{date_str}.npy")
            try:
                # 检查文件是否存在
                if not os.path.exists(data_file):
                    print(f"文件不存在: {data_file}")
                    raise FileNotFoundError(f"文件不存在: {data_file}")

                # 获取文件大小信息
                file_size = os.path.getsize(data_file) / (1024**3)  # GB
                print(f"正在加载 {date_str}，文件大小: {file_size:.2f} GB")

                # 尝试安全加载数据
                try:
                    # 首先尝试内存映射模式检查文件
                    data_mmap = np.load(data_file, mmap_mode='r')
                    expected_shape = (144, 4, 500, 900)

                    if data_mmap.shape != expected_shape:
                        print(f"警告: {date_str} 数据形状异常: {data_mmap.shape}, 期望: {expected_shape}")
                        # 如果形状不对，跳过这个文件
                        raise ValueError(f"数据形状异常: {data_mmap.shape}")

                    # 强制转换为float32以节省内存
                    daily_data = np.array(data_mmap, dtype=np.float32)
                    del data_mmap  # 释放内存映射

                except Exception as load_error:
                    print(f"内存映射加载失败，尝试直接加载: {load_error}")
                    # 如果内存映射失败，尝试直接加载
                    daily_data = np.load(data_file).astype(np.float32)

                # 使用时空插值填充NaN
                filled_data = self.interpolate_nans(daily_data.copy())
                del daily_data  # 释放原始数据内存

                # 转换为torch张量
                filled_data = torch.from_numpy(filled_data).float()
                num_weight = (filled_data > -100).float()
                filled = filled_data * num_weight
                filled_mask = (filled == 0.0).bool()

                # 存储预处理后的数据和掩码
                self.processed_data[date_str] = filled
                self.data_masks[date_str] = filled_mask
                print(f"已重新加载并预处理 {date_str} 数据，形状: {filled_data.shape}")

            except Exception as e:
                print(f"加载文件 {data_file} 失败: {e}")
                print(f"错误详情: {type(e).__name__}: {str(e)}")
                # 创建空数据和掩码
                try:
                    self.processed_data[date_str] = torch.zeros((144, 4, 500, 900), dtype=torch.float32)
                    self.data_masks[date_str] = torch.ones((144, 4, 500, 900), dtype=torch.bool)  # 全部标记为NaN
                except Exception as fallback_error:
                    print(f"创建空数据也失败: {fallback_error}")
                    # 如果连创建空数据都失败，说明内存真的不足了
                    import gc
                    gc.collect()
                    continue

        print(f"重新加载完成，共{len(self.processed_data)}个日期的数据")

        # 重新分析数据连续性并生成滑动窗口样本
        self.analyze_data_continuity()
        self.build_continuous_timeline()
        self.generate_sliding_samples()

    def interpolate_nans(self, data):
        """
        使用时空插值填充NaN值
        Args:
            data: 原始数据，形状为 (T, C, H, W)
        Returns:
            filled_data: 填充后的数据
        """
        # 时间维度插值
        for c in range(data.shape[1]):
            for h in range(data.shape[2]):
                for w in range(data.shape[3]):
                    series = data[:, c, h, w]
                    if np.any(np.isnan(series)):
                        valid_indices = np.where(~np.isnan(series))[0]
                        if len(valid_indices) > 0:  # 如果有有效值
                            nan_indices = np.where(np.isnan(series))[0]
                            # 使用有效值进行插值
                            series[nan_indices] = np.interp(
                                nan_indices, valid_indices, series[valid_indices],
                                left=0, right=0  # 边界外使用0
                            )
        
        # 空间维度插值 - 对于时间插值后仍然有NaN的位置
        for t in range(data.shape[0]):
            for c in range(data.shape[1]):
                frame = data[t, c]
                if np.any(np.isnan(frame)):
                    # 获取非NaN位置的坐标和值
                    y_indices, x_indices = np.where(~np.isnan(frame))
                    if len(y_indices) > 0:  # 如果有有效值
                        points = np.column_stack((y_indices, x_indices))
                        values = frame[~np.isnan(frame)]
                        
                        # 获取NaN位置的坐标
                        nan_y_indices, nan_x_indices = np.where(np.isnan(frame))
                        nan_points = np.column_stack((nan_y_indices, nan_x_indices))
                        
                        # 使用最近邻插值
                        if len(points) > 3:  # 确保有足够的点进行插值
                            try:
                                interpolator = NearestNDInterpolator(points, values)
                                frame[nan_y_indices, nan_x_indices] = interpolator(nan_points)
                            except Exception as e:
                                print(f"空间插值失败: {e}")
                                frame[np.isnan(frame)] = 0.0  # 如果插值失败，使用0填充
                        else:
                            frame[np.isnan(frame)] = 0.0  # 如果有效点太少，使用0填充
        
        # 检查是否还有NaN，如果有则填充为0
        if np.any(np.isnan(data)):
            data = np.nan_to_num(data, nan=0.0)
            
        return data

    def analyze_data_continuity(self):
        """分析数据的时间连续性"""
        sorted_dates = sorted(self.processed_data.keys())

        for i, date_str in enumerate(sorted_dates):
            daily_data = self.processed_data[date_str]
            daily_mask = self.data_masks[date_str]

            # 分析当天的有效时间点
            valid_timepoints = []
            for t in range(daily_data.shape[0]):
                if not daily_mask[t].all():
                    valid_timepoints.append(t)

            # 只在数据异常时输出警告
            if len(valid_timepoints) < 100:
                print(f"警告: {date_str} 只有 {len(valid_timepoints)} 个有效时间点")

    def check_day_continuity(self, date1, date2):
        """检查两天之间是否时间连续"""
        # 解析日期
        d1 = datetime.strptime(date1, "%Y%m%d")
        d2 = datetime.strptime(date2, "%Y%m%d")

        # 检查是否是连续的两天
        if d2 - d1 != timedelta(days=1):
            return False

        # 检查第一天的最后时间点和第二天的第一时间点是否连续
        data1 = self.processed_data[date1]
        data2 = self.processed_data[date2]
        mask1 = self.data_masks[date1]
        mask2 = self.data_masks[date2]

        # 找到第一天最后的有效时间点
        last_valid_t1 = None
        for t in range(data1.shape[0] - 1, -1, -1):
            if not mask1[t].all():
                last_valid_t1 = t
                break

        # 找到第二天第一个有效时间点
        first_valid_t2 = None
        for t in range(data2.shape[0]):
            if not mask2[t].all():
                first_valid_t2 = t
                break

        # 检查时间连续性
        if last_valid_t1 is not None and first_valid_t2 is not None:
            return self.is_time_continuous(last_valid_t1, first_valid_t2)

        return False

    def is_time_continuous(self, t1_end, t2_start):
        """检查两个时间点是否连续（考虑跨天）"""
        # 如果第一天结束于23:50(t=143)，第二天开始于00:00(t=0)，则连续
        if t1_end == 143 and t2_start == 0:
            return True
        # 可以添加其他连续性判断逻辑
        return False

    def build_continuous_timeline(self):
        """构建全局连续的时间轴"""
        self.global_timeline = []

        sorted_dates = sorted(self.processed_data.keys())
        current_segment = []

        for i, date_str in enumerate(sorted_dates):
            daily_data = self.processed_data[date_str]
            daily_mask = self.data_masks[date_str]

            day_valid_points = []
            for t in range(daily_data.shape[0]):
                if not daily_mask[t].all():
                    day_valid_points.append({
                        'date': date_str,
                        'time_idx': t,
                        'global_time': f"{date_str}_{t:03d}",
                        'actual_time': f"{t*10//60:02d}:{t*10%60:02d}"
                    })

            if not current_segment:
                current_segment = day_valid_points
            else:
                prev_date = sorted_dates[i-1]
                if self.check_day_continuity(prev_date, date_str):
                    current_segment.extend(day_valid_points)
                else:
                    if len(current_segment) >= 24:
                        self.global_timeline.append(current_segment)
                    current_segment = day_valid_points

        if len(current_segment) >= 24:
            self.global_timeline.append(current_segment)

        # 只输出汇总信息
        total_timepoints = sum(len(segment) for segment in self.global_timeline)
        print(f"数据分析完成: {len(self.global_timeline)} 个连续时间段, 共 {total_timepoints} 个有效时间点")

    def generate_sliding_samples(self):
        """生成样本：训练模式使用滑动窗口，测试模式使用非重叠窗口"""
        self.sliding_samples = []

        if self.mode == 'test':
            # 测试模式：生成连续预测样本
            for segment_idx, segment in enumerate(self.global_timeline):
                segment_samples = []
                
                # 连续预测采样：每次跳过12个时间点（output_length），实现连续预测
                # 样本0: 00:00-01:50 → 02:00-03:50
                # 样本1: 02:00-03:50 → 04:00-05:50
                for start_pos in range(0, len(segment) - 24 + 1, 12):  # step=12
                    input_points = segment[start_pos:start_pos + 12]
                    target_points = segment[start_pos + 12:start_pos + 24]

                    sample = {
                        'segment_idx': segment_idx,
                        'start_pos': start_pos,
                        'input_points': input_points,
                        'target_points': target_points,
                    }

                    segment_samples.append(sample)

                self.sliding_samples.extend(segment_samples)
            
            print(f"连续预测样本生成完成: 共 {len(self.sliding_samples)} 个测试样本")
        else:
            # 训练和验证模式：使用滑动窗口
            for segment_idx, segment in enumerate(self.global_timeline):
                segment_samples = []

                for start_pos in range(len(segment) - 24 + 1):
                    input_points = segment[start_pos:start_pos + 12]
                    target_points = segment[start_pos + 12:start_pos + 24]

                    sample = {
                        'segment_idx': segment_idx,
                        'start_pos': start_pos,
                        'input_points': input_points,
                        'target_points': target_points,
                    }

                    segment_samples.append(sample)

                self.sliding_samples.extend(segment_samples)

            print(f"滑动窗口样本生成完成: 共 {len(self.sliding_samples)} 个{self.mode}样本")

    def __len__(self):
        """返回数据集大小"""
        return len(self.sliding_samples)
    
    def __getitem__(self, index):
        """获取指定索引的样本"""
        sample = self.sliding_samples[index]

        # 构建输入张量
        input_tensors = []
        input_masks = []

        for point in sample['input_points']:
            date_str = point['date']
            time_idx = point['time_idx']

            daily_data = self.processed_data[date_str]
            daily_mask = self.data_masks[date_str]

            input_tensors.append(daily_data[time_idx:time_idx+1, 0:3])  # [1, 1, 500, 900]
            input_masks.append(daily_mask[time_idx:time_idx+1, 1:2])

        # 构建目标张量
        target_tensors = []
        target_masks = []

        for point in sample['target_points']:
            date_str = point['date']
            time_idx = point['time_idx']

            daily_data = self.processed_data[date_str]
            daily_mask = self.data_masks[date_str]

            target_tensors.append(daily_data[time_idx:time_idx+1, 1:2])
            target_masks.append(daily_mask[time_idx:time_idx+1, 1:2])

        # 拼接张量
        input_tensor = torch.cat(input_tensors, dim=0)    # [12, 1, 500, 900]
        input_mask = torch.cat(input_masks, dim=0)
        target_tensor = torch.cat(target_tensors, dim=0)  # [12, 1, 500, 900]
        target_mask = torch.cat(target_masks, dim=0)

        # 如果是测试模式，同时返回时间信息
        if self.mode == 'test':
            # 提取时间信息
            input_times = [f"{point['date']}_{point['time_idx']:03d}" for point in sample['input_points']]
            target_times = [f"{point['date']}_{point['time_idx']:03d}" for point in sample['target_points']]
            
            return input_tensor, input_mask, target_tensor, target_mask, input_times, target_times
        else:
            return input_tensor, input_mask, target_tensor, target_mask


class FusionDatasetHindcast(Dataset):
    """用于回溯测试的数据集"""
    
    def __init__(self, configs):
        """初始化回溯测试数据集"""
        self.configs = configs
        self.data_path = configs['data_path']
        self.input_length = configs['input_length']
        self.csv_path = configs.get('csv_path', self.data_path)  # 获取CSV路径，默认与数据路径相同
        
        # 如果提供了特定日期，则使用该日期
        if 'test_date' in configs:
            self.test_date = configs['test_date']
        else:
            # 尝试从路径中提取日期
            test_date = os.path.basename(self.data_path)
            if not test_date.isdigit() or len(test_date) != 8:
                # 如果目录名不是日期格式，则尝试从路径中提取
                parts = self.data_path.split('/')
                for part in parts:
                    if part.isdigit() and len(part) == 8:
                        test_date = part
                        break
            self.test_date = test_date
        
        # 构建数据文件路径
        self.data_file = os.path.join(self.data_path, f"fusion_{self.test_date}.npy")
        
        # 加载数据
        try:
            daily_data = np.load(self.data_file)  # 形状应为 (144, 4, 500, 900)
            self.num_frames = daily_data.shape[0] - self.input_length
            print(f"已加载回溯测试数据: {self.data_file}, 可用序列数: {self.num_frames}")
            
            # 创建NaN掩码
            self.data_mask = np.isnan(daily_data)
            
            # 使用时空插值填充NaN
            self.daily_data = self.interpolate_nans(daily_data)
            
            # 转换为torch张量
            self.daily_data = torch.from_numpy(self.daily_data).float()
            self.data_mask = torch.from_numpy(self.data_mask).bool()
            
        except Exception as e:
            print(f"加载文件 {self.data_file} 失败: {e}")
            self.daily_data = torch.zeros((144, 4, 500, 900), dtype=torch.float32)
            self.data_mask = torch.ones((144, 4, 500, 900), dtype=torch.bool)  # 全部标记为NaN
            self.num_frames = 0
    
    def interpolate_nans(self, data):
        """
        使用时空插值填充NaN值
        Args:
            data: 原始数据，形状为 (T, C, H, W)
        Returns:
            filled_data: 填充后的数据
        """
        # 时间维度插值
        for c in range(data.shape[1]):
            for h in range(data.shape[2]):
                for w in range(data.shape[3]):
                    series = data[:, c, h, w]
                    if np.any(np.isnan(series)):
                        valid_indices = np.where(~np.isnan(series))[0]
                        if len(valid_indices) > 0:  # 如果有有效值
                            nan_indices = np.where(np.isnan(series))[0]
                            # 使用有效值进行插值
                            series[nan_indices] = np.interp(
                                nan_indices, valid_indices, series[valid_indices],
                                left=0, right=0  # 边界外使用0
                            )
        
        # 空间维度插值 - 对于时间插值后仍然有NaN的位置
        for t in range(data.shape[0]):
            for c in range(data.shape[1]):
                frame = data[t, c]
                if np.any(np.isnan(frame)):
                    # 获取非NaN位置的坐标和值
                    y_indices, x_indices = np.where(~np.isnan(frame))
                    if len(y_indices) > 0:  # 如果有有效值
                        points = np.column_stack((y_indices, x_indices))
                        values = frame[~np.isnan(frame)]
                        
                        # 获取NaN位置的坐标
                        nan_y_indices, nan_x_indices = np.where(np.isnan(frame))
                        nan_points = np.column_stack((nan_y_indices, nan_x_indices))
                        
                        # 使用最近邻插值
                        if len(points) > 3:  # 确保有足够的点进行插值
                            try:
                                interpolator = NearestNDInterpolator(points, values)
                                frame[nan_y_indices, nan_x_indices] = interpolator(nan_points)
                            except Exception as e:
                                print(f"空间插值失败: {e}")
                                frame[np.isnan(frame)] = 0.0  # 如果插值失败，使用0填充
                        else:
                            frame[np.isnan(frame)] = 0.0  # 如果有效点太少，使用0填充
        
        # 检查是否还有NaN，如果有则填充为0
        if np.any(np.isnan(data)):
            data = np.nan_to_num(data, nan=0.0)
            
        return data
    
    def __len__(self):
        """返回数据集大小"""
        return max(0, self.num_frames)
    
    def __getitem__(self, index):
        """获取指定索引的样本"""
        # 提取输入序列及其掩码
        input_tensor = self.daily_data[index:index+self.input_length, :3]  # 只取前3个通道
        input_mask = self.data_mask[index:index+self.input_length, :3]
        
        return input_tensor, input_mask


# 测试代码
if __name__ == "__main__":
    configs = {
        'data_path': '/scratch/jianhao/fusion',
        'input_length': 12,  # 输入序列长度
        'output_length': 12,  # 输出序列长度
    }
    
    # 测试训练数据集
    train_dataset = FusionDataset(configs, 'train')
    print(f"训练集样本数量: {len(train_dataset)}")
    
    # 获取一个样本
    input_data, input_mask, target_data, target_mask = train_dataset[0]
    print(f"输入数据形状: {input_data.shape}")
    print(f"输入掩码形状: {input_mask.shape}")
    print(f"目标数据形状: {target_data.shape}")
    print(f"目标掩码形状: {target_mask.shape}")
    
    # 测试回算模式数据集
    hindcast_dataset = FusionDatasetHindcast(configs)
    print(f"回算模式样本数量: {len(hindcast_dataset)}")
    
    # 获取一个样本
    input_data, input_mask = hindcast_dataset[0]
    print(f"输入数据形状: {input_data.shape}")
    print(f"输入掩码形状: {input_mask.shape}") 