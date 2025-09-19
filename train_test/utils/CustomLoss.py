import math
import random

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class WeightedPrecisionLoss(nn.Module):
    """基于目标值分段的加权MAE损失函数
    Args:
        weights (tuple): 每个阈值区间的权重值，长度需比thresholds多1
        thresholds (list): 定义目标值分段的阈值列表（需升序排列）
        eps (float): 数值稳定性系数，防止梯度爆炸
    """

    def __init__(self, weights=(1, 1, 2.5, 5, 10, 20),
                 thresholds=[0.016, 0.033, 0.16, 0.33, 0.66], eps=1e-8):
        super().__init__()
        assert len(thresholds) + 1 == len(weights), "weights长度必须比thresholds大1"
        assert sorted(thresholds) == thresholds, "thresholds必须为升序排列"

        # 注册缓冲区使参数支持设备切换
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        self.register_buffer('thresholds', torch.tensor(thresholds, dtype=torch.float32))
        self.eps = eps

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predict: 预测值张量，形状为(B, C, H, W)
            target: 目标值张量，形状需与predict相同
        Returns:
            加权后的MAE损失
        """
        # 输入维度校验
        assert predict.shape == target.shape, "predict和target形状不一致"

        # 创建初始权重矩阵（使用目标值进行分段）
        balance_weights = torch.full_like(target, self.weights[0], dtype=torch.float32)

        # 向量化分段处理（比循环更高效）
        for i in range(len(self.thresholds)):
            lower = self.thresholds[i] if i == 0 else self.thresholds[i - 1]
            upper = self.thresholds[i]
            mask = (target >= lower) & (target < upper)
            balance_weights[mask] = self.weights[i]

        # 处理最后一个区间
        balance_weights[target >= self.thresholds[-1]] = self.weights[-1]

        # 计算加权MAE（添加平滑项保证数值稳定性）
        absolute_error = torch.abs(predict - target)
        weighted_mae = torch.sum(balance_weights * absolute_error) / (torch.sum(balance_weights) + self.eps)

        return weighted_mae


def weighted_l1_loss(output: torch.Tensor,
                     target: torch.Tensor,
                     scale_factor: float = 1.0) -> torch.Tensor:
    """
    通道0的加权损失计算
    Args:
        output: 模型输出张量 (B, C, H, W)
        target: 目标张量 (B, C, H, W)
        scale_factor: 阈值缩放因子，允许动态调整阈值范围
    """
    # 使用可调整的阈值缩放因子
    thresholds = [0.01 * scale_factor,
                  0.05 * scale_factor,
                  0.26 * scale_factor,
                  0.57 * scale_factor,
                  0.82 * scale_factor]

    criterion = WeightedPrecisionLoss(
        weights=(1, 1, 3, 10, 10, 20),
        thresholds=thresholds
    )

    # 仅计算第一个通道的损失
    return criterion(output[:, :, 0, ...], target[:, :, 0, ...])




class CustomLoss:
    """
    Custom loss:自定义损失函数,皆为静态函数
    """

    @staticmethod
    def weighted_l1_loss(output, ground_truth):
        dBZ_ground_truth = 70.0 * ground_truth
        weight_matrix = torch.clamp(
            torch.pow(10.0, (dBZ_ground_truth - 10.0 * math.log10(58.53)) / 15.6), 1.0, 30.0
        )
        return torch.mean(weight_matrix * torch.abs(output - ground_truth))

    @staticmethod
    def weighted_l2_loss(output, ground_truth):
        dBZ_ground_truth = 70.0 * ground_truth
        weight_matrix = torch.clamp(
            torch.pow(10.0, (dBZ_ground_truth - 10.0 * math.log10(58.53)) / 15.6), 1.0, 30.0
        )
        return torch.mean(weight_matrix * torch.pow(torch.abs(output - ground_truth), 2.0))

    @staticmethod
    def perceptual_similarity_loss(
        output, ground_truth, encoder, randomly_sampling=None
    ):
        seq_len = output.size()[0]
        ground_truth = ground_truth.float()
        if randomly_sampling is not None:
            index = random.sample(range(0, seq_len), randomly_sampling)
            output_feature = encoder(output[index])
            ground_truth_feature = encoder(ground_truth[index])
        else:
            output_feature = encoder(output)
            ground_truth_feature = encoder(ground_truth)
        return torch.mean(torch.pow(torch.abs(output_feature - ground_truth_feature), 2.0))

    @staticmethod
    def seq_d_hinge_adversarial_loss(input, output, ground_truth, seq_d):
        ground_truth = ground_truth.float()
        real_seq = seq_d(torch.cat([input, ground_truth], dim=0))
        fake_seq = seq_d(torch.cat([input, output], dim=0))
        seq_d_loss_real = torch.mean(torch.relu(1.0 - real_seq))
        seq_d_loss_fake = torch.mean(torch.relu(1.0 + fake_seq))
        return seq_d_loss_real, seq_d_loss_fake

    @staticmethod
    def fra_d_hinge_adversarial_loss(output, ground_truth, fra_d):
        ground_truth = ground_truth.float()
        real_fra = fra_d(ground_truth)
        fake_fra = fra_d(output)
        fra_d_loss_real = torch.mean(torch.relu(1.0 - real_fra))
        fra_d_loss_fake = torch.mean(torch.relu(1.0 + fake_fra))
        return fra_d_loss_real, fra_d_loss_fake

    @staticmethod
    def hinge_adversarial_loss(input, output, seq_d, fra_d):
        input = input.float()
        fake_seq = seq_d(torch.cat([input, output], dim=0))
        fake_fra = fra_d(output)
        g_loss_seq = -torch.mean(fake_seq)
        g_loss_fra = -torch.mean(fake_fra)
        return g_loss_seq, g_loss_fra

    @staticmethod
    def seq_d_bce_adversarial_loss(input, output, ground_truth, seq_d):
        input = input.float()
        ground_truth = ground_truth.float()
        real_seq = seq_d(torch.cat([input, ground_truth], dim=0))
        fake_seq = seq_d(torch.cat([input, output], dim=0))
        seq_d_loss_real = F.binary_cross_entropy(real_seq, torch.ones_like(real_seq))
        seq_d_loss_fake = F.binary_cross_entropy(fake_seq, torch.zeros_like(fake_seq))
        return seq_d_loss_real, seq_d_loss_fake

    @staticmethod
    def fra_d_bce_adversarial_loss(output, ground_truth, fra_d):
        ground_truth = ground_truth.float()
        real_fra = fra_d(ground_truth)
        fake_fra = fra_d(output)
        fra_d_loss_real = F.binary_cross_entropy(real_fra, torch.ones_like(real_fra))
        fra_d_loss_fake = F.binary_cross_entropy(fake_fra, torch.zeros_like(fake_fra))
        return fra_d_loss_real, fra_d_loss_fake

    @staticmethod
    def bce_adversarial_loss(input, output, seq_d, fra_d):
        input = input.float()
        fake_seq = seq_d(torch.cat([input, output], dim=0))
        fake_fra = fra_d(output)
        g_loss_seq = F.binary_cross_entropy(fake_seq, torch.ones_like(fake_seq))
        g_loss_fra = F.binary_cross_entropy(fake_fra, torch.ones_like(fake_fra))
        return g_loss_seq, g_loss_fra

    @staticmethod
    def mixed_adversarial_loss(input, output, seq_d, fra_d):
        input = input.float()
        fake_seq = seq_d(torch.cat([input, output], dim=0))
        fake_fra = fra_d(output)
        g_loss_seq = F.binary_cross_entropy(fake_seq, torch.ones_like(fake_seq))
        g_loss_fra = -torch.mean(fake_fra)
        return g_loss_seq, g_loss_fra


def weighted_precipitation_loss(pred, target, mask = None):
    """
    改进的加权降水预测损失函数，使用区间形式的权重设置
    
    Args:
        pred: 预测值 [B, T, C, H, W]
        target: 真实值 [B, T, C, H, W]
        
    Returns:
        loss: 加权MSE损失
    """
    # 确保输入维度正确
    # a = 3.5
    # b = 1.0

    if mask is None:
        mask = torch.zeros_like(target, dtype=torch.bool)
    
    # 创建有效位置的掩码（非NaN位置）
    valid_mask = ~mask

    assert pred.shape == target.shape, f"预测形状 {pred.shape} 与目标形状 {target.shape} 不匹配"

    # 计算MSE损失
    mse_loss = (pred - target) ** 2
    
    # 创建动态权重矩阵
    # 基础权重为1（无降水区域）
    weights = torch.ones_like(target)
    
    # 使用方案1的区间形式设置权重
    # 完全无降水区域（精确等于0mm/h）：基础权重为0.1
    weights = torch.where(target == 0, torch.ones_like(target) * 0.1, weights)
    
    # 微量降水区域 (0-0.1mm/h): 权重为0.5
    weights = torch.where((target > 0.0) & (target <= 0.1), torch.ones_like(target) * 0.5, weights)
    
    # 轻度降水区域 (0.1-0.5mm/h): 权重为1.0
    weights = torch.where((target > 0.1) & (target <= 0.5), torch.ones_like(target) * 1.0, weights)
    
    # 小雨区域 (0.5-1.5mm/h): 权重为2.0
    weights = torch.where((target > 0.5) & (target <= 1.5), torch.ones_like(target) * 2.0, weights)
    
    # 中雨区域(轻) (1.5-4.0mm/h): 权重为4.0
    weights = torch.where((target > 1.5) & (target <= 4.0), torch.ones_like(target) * 4.0, weights)
    
    # 中雨区域(重) (4.0-8.0mm/h): 权重为8.0
    weights = torch.where((target > 4.0) & (target <= 8.0), torch.ones_like(target) * 8.0, weights)
    
    # 大雨区域 (8.0-10.0mm/h): 权重为10.0
    weights = torch.where((target > 8.0) & (target <= 10.0), torch.ones_like(target) * 10.0, weights)
    
    # 强降水区域 (10.0-25.0mm/h): 权重为15.0
    weights = torch.where((target > 10.0) & (target <= 25.0), torch.ones_like(target) * 15.0, weights)
    
    # 极端降水区域 (>25.0mm/h): 权重为20.0
    weights = torch.where(target > 25.0, torch.ones_like(target) * 20.0, weights)
    
    # 添加注意力机制（已注释掉）：
    # over = torch.relu(pred - target)  # 预测过度
    # under = torch.relu(target - pred)  # 预测不足
    # weights = weights * (1.0 + a * over + b * under)
    
    # 计算加权损失
    num_valid = valid_mask.float().sum()
    if num_valid > 0:
        weighted_loss = (mse_loss * weights) / num_valid
    else:
        weighted_loss = torch.tensor(0.0, device=pred.device)
    
    # 返回平均损失
    return torch.mean(weighted_loss)


