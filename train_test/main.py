import os
import argparse
import torch
import torch.nn as nn
from datetime import datetime
from models.MS_RadarFormer.MS_RadarFormer import FusionFormer
from train.TrainerFusion import Trainer
from train.FusionDataset import FusionDataset  # 导入数据集类
from clearml import Task


class CustomClearML:
    def __init__(self, project_name, task_name):
        self.task = Task.init(project_name, task_name)
        self.logger = self.task.get_logger()
        
    def __call__(self, title, series, value, iteration):
        self.logger.report_scalar(title, series, value, iteration)


def main():
    parser = argparse.ArgumentParser(description='FusionFormer: 多源数据融合降水临近预报模型')

    parser.add_argument('--is_train', type=int, default=1, help='是否训练模式 (1: 训练, 0: 测试)')
    parser.add_argument('--load_model', type=str, help='模型加载路径（测试模式时必须提供checkpoint路径）')

    parser.add_argument('--data_path', type=str, default='/scratch/jianhao/fusion', help='默认数据路径（主要用于训练集）')
    parser.add_argument('--csv_path', type=str, default='/home/jianhao/methods/single-source/train_test/csv3', help='CSV文件路径')
    parser.add_argument('--batch_size', type=int, default=3, help='批次大小')
    parser.add_argument('--epoch', type=int, default=80, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--use_gpu_id', type=str, default='0', help='使用的GPU ID')
    parser.add_argument('--model_name', type=str, default='FusionFormer', help='模型名称')
    parser.add_argument('--save_dir', type=str, default=f'./checkpoints/{datetime.now().strftime("%Y%m%d_%H%M%S")}', help='模型保存路径（默认包含时间戳）')
    parser.add_argument('--pred_img_path', type=str, default='./pred_results', help='预测结果保存路径')
    parser.add_argument('--test_interval', type=int, default=1, help='测试间隔')
    parser.add_argument('--num_save_samples', type=int, default=5, help='保存的样本数量')
    parser.add_argument('--use_clearml', type=int, default=1, help='是否使用ClearML记录训练过程 (1: 是, 0: 否)')
    parser.add_argument('--clearml_project', type=str, default='FusionFormer', help='ClearML项目名称')
    parser.add_argument('--clearml_task', type=str, default='train_0805', help='ClearML任务名称')
    
    # 模型参数
    parser.add_argument('--input_length', type=int, default=12, help='输入序列长度')
    parser.add_argument('--output_length', type=int, default=12, help='输出序列长度')
    parser.add_argument('--img_height', type=int, default=500, help='图像高度')
    parser.add_argument('--img_width', type=int, default=900, help='图像宽度')
    parser.add_argument('--img_channel', type=int, default=3, help='图像通道数')
    parser.add_argument('--img_out_channel', type=int, default=1, help='输出像通道数')
    parser.add_argument('--patch_size', type=int, default=5, help='图像块大小')
    parser.add_argument('--model_patch_size', type=str, default='4,5,5', help='模型图像块大小')
    parser.add_argument('--embed_dim', type=int, default=256, help='嵌入维度')
    parser.add_argument('--depths', type=int, default=6, help='Transformer深度')
    parser.add_argument('--num_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--window_size', type=str, default='4,5,5', help='窗口大小')
    parser.add_argument('--drop_rate', type=float, default=0.0, help='丢弃率')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, help='注意力丢弃率')
    parser.add_argument('--drop_path_rate', type=float, default=0.2, help='路径丢弃率')
    parser.add_argument('--use_multi_resolution_branch', type=int, default=1, help='是否使用多分辨率分支')
    parser.add_argument('--use_multi_scale_patch_embedding', type=int, default=1, help='是否使用多尺度图像块嵌入')
    
    args = parser.parse_args()
    
    # 将参数转换为字典
    configs = vars(args)

    # 初始化ClearML（如果启用）
    clearml_logger = None
    if configs.get("use_clearml", 0) == 1:
        # 构建包含关键参数的task名称
        mode = "train" if configs["is_train"] else "test"
        save_dir_simple = configs["save_dir"].split('/')[-1] if '/' in configs["save_dir"] else configs["save_dir"]
        
        task_name = f"{mode}_channel_{configs['img_channel']}_{configs['model_name']}_bs{configs['batch_size']}_ep{configs['epoch']}_{save_dir_simple}"
        
        clearml_logger = CustomClearML(
            configs.get("clearml_project", "FusionFormer"),
            task_name
        )
        print(f"ClearML已启用 - 项目: {configs.get('clearml_project', 'FusionFormer')}, 任务: {task_name}")
    
    # 检查load_model参数：测试模式时必须提供checkpoint路径
    if not configs["is_train"] and not configs["load_model"]:
        raise ValueError("测试模式时必须提供 --load_model 参数来指定checkpoint路径")
    
    # 处理特殊参数
    configs["model_patch_size"] = tuple(map(int, configs["model_patch_size"].split(',')))
    configs["window_size"] = tuple(map(int, configs["window_size"].split(',')))
    
    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = configs["use_gpu_id"]
    configs["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {configs['device']}")
    print(f"运行模式: {'训练' if configs['is_train'] else '测试'}")
    
    # 预先加载数据集
    print("开始预先加载数据集...")
    print("注意：一次性训练策略")
    print(" - 训练集：一次性加载所有训练数据（2021-2025年）")
    print(" - 验证集：加载所有验证数据")
    print(" - 测试集：加载所有测试数据")

    if configs["is_train"]:
        # 一次性加载所有训练数据
        print("加载所有训练数据（2021-2025年）...")
        train_dataset = FusionDataset(configs, 'train')  # 不指定year_filter，加载所有年份
        print("加载验证数据集...")
        valid_dataset = FusionDataset(configs, 'valid')
        print("训练集和验证集预加载完成")
    else:
        # 测试模式只加载测试集
        test_dataset = FusionDataset(configs, 'test')
        print("测试集预加载完成")
    
    # 创建模型
    print("创建FusionFormer模型...")
    model = FusionFormer(configs)
    
    
    # 创建训练器，传入预加载的数据集和ClearML日志器
    if configs["is_train"]:
        trainer = Trainer(configs, model, train_dataset, valid_dataset, clearml_logger)
    else:
        trainer = Trainer(configs, model, None, test_dataset, clearml_logger)
    
    # 加载模型（如果指定）
    if configs["load_model"]:
        trainer.load(configs["load_model"])
    
    # 训练或测试
    if configs["is_train"]:
        print("开始训练...")
        trainer.train()
    else:
        print("开始测试...")
        trainer.test()


if __name__ == "__main__":
    main() 