import datetime
import os
import gc
import numpy as np
import torch
import torch.utils.data as Data
from train.FusionDataset import FusionDataset
from torch.optim import AdamW
from tqdm import tqdm
from utils.CustomLoss import weighted_precipitation_loss
from utils.PreProcess import PreProcess
import torch.nn.functional as F


class Trainer(object):
    def __init__(self, configs, model, train_dataset=None, val_dataset=None, clearml_logger=None):
        self.configs = configs
        self.model = model
        self.model = self.model.to(self.configs["device"])  # 模型加载到指定设备（如GPU）
        self.loss_fn = weighted_precipitation_loss  # 使用支持掩码的加权降水损失函数

        # 初始化训练参数
        self.cur_epoch = 1
        self.best_epoch = 1
        self.best_mse = float('inf')  # 记录最佳MSE（越小越好）

        # 设置评估阈值（对应0.1mm/h, 1mm/h, 5mm/h）
        self.thresholds = [0.001, 0.01, 0.05]

        # 使用传入的ClearML日志器
        self.clearml_logger = clearml_logger
        self.use_clearml = clearml_logger is not None

        # 存储数据集引用
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if self.configs["is_train"]:
            # 如果是训练模式，初始化优化器
            self.optimizer = AdamW(self.model.parameters(), lr=configs["learning_rate"], weight_decay=1e-5)

            # 使用预先加载的数据集创建数据加载器
            if train_dataset is not None and val_dataset is not None:
                self.train_data = Data.DataLoader(
                    train_dataset,
                    batch_size=self.configs["batch_size"],
                    shuffle=True,
                    drop_last=True,
                    num_workers=4,  # 增加数据加载线程
                    pin_memory=True,  # 使用固定内存加速数据传输
                )
                self.val_data = Data.DataLoader(
                    val_dataset,
                    batch_size=self.configs["batch_size"],
                    shuffle=False,
                    drop_last=True,
                    num_workers=4,  # 增加数据加载线程
                    pin_memory=True,  # 使用固定内存加速数据传输
                )
            else:
                # 如果未提供预加载的数据集，则创建新的数据集实例
                print("警告：未提供预加载的数据集，将创建新的数据集实例")
                self.train_data = Data.DataLoader(
                    FusionDataset(configs, 'train'),
                    batch_size=self.configs["batch_size"],
                    shuffle=True,
                    drop_last=True,
                    num_workers=4,
                    pin_memory=True,
                )
                self.val_data = Data.DataLoader(
                    FusionDataset(configs, 'valid'),
                    batch_size=self.configs["batch_size"],
                    shuffle=False,
                    drop_last=True,
                    num_workers=4,
                    pin_memory=True,
                )
                
            # 学习率调度器设置为OneCycleLR
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=configs["learning_rate"],
                epochs=configs["epoch"],
                steps_per_epoch=len(self.train_data),
                pct_start=0.3
            )
        else:
            # 如果是测试模式，初始化测试数据加载器
            if val_dataset is not None:
                # 为测试模式定义自定义collate函数处理时间信息
                def custom_collate_fn(batch):
                    if len(batch[0]) == 6:  # 测试模式有时间信息
                        input_tensors = torch.stack([item[0] for item in batch])
                        input_masks = torch.stack([item[1] for item in batch])
                        target_tensors = torch.stack([item[2] for item in batch])
                        target_masks = torch.stack([item[3] for item in batch])
                        input_times = [item[4] for item in batch]  # 保持为列表
                        target_times = [item[5] for item in batch]  # 保持为列表
                        return input_tensors, input_masks, target_tensors, target_masks, input_times, target_times
                    else:  # 训练/验证模式
                        return torch.utils.data.dataloader.default_collate(batch)
                
                self.val_data = Data.DataLoader(
                    val_dataset,
                    batch_size=self.configs["batch_size"],
                    shuffle=False,
                    drop_last=True,
                    num_workers=0,  # 减少数据加载线程，避免CUDA内存错误
                    pin_memory=False,  # 禁用固定内存，避免CUDA内存错误
                    collate_fn=custom_collate_fn,  # 使用自定义collate函数
                )
            else:
                # 如果未提供预加载的测试集，则创建新的数据集实例
                print("警告：未提供预加载的测试集，将创建新的数据集实例")
                
                # 同样需要自定义collate函数
                def custom_collate_fn(batch):
                    if len(batch[0]) == 6:  # 测试模式有时间信息
                        input_tensors = torch.stack([item[0] for item in batch])
                        input_masks = torch.stack([item[1] for item in batch])
                        target_tensors = torch.stack([item[2] for item in batch])
                        target_masks = torch.stack([item[3] for item in batch])
                        input_times = [item[4] for item in batch]  # 保持为列表
                        target_times = [item[5] for item in batch]  # 保持为列表
                        return input_tensors, input_masks, target_tensors, target_masks, input_times, target_times
                    else:  # 训练/验证模式
                        return torch.utils.data.dataloader.default_collate(batch)
                
                self.val_data = Data.DataLoader(
                    FusionDataset(configs, 'test'),
                    batch_size=self.configs["batch_size"],
                    shuffle=False,
                    drop_last=True,
                    num_workers=0,  # 减少数据加载线程，避免CUDA内存错误
                    pin_memory=False,  # 禁用固定内存，避免CUDA内存错误
                    collate_fn=custom_collate_fn,  # 使用自定义collate函数
                )
            
        # 创建日志目录
        self.log_dir = '/home/jianhao/methods/single-source/logs'
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f'train_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        # 如果保存目录不存在，则创建目录
        if not os.path.isdir(self.configs["save_dir"]):
            os.makedirs(self.configs["save_dir"])
        # 如果预测图片存放路径不存在，则创建目录
        if not os.path.exists(self.configs["pred_img_path"]):
            os.makedirs(self.configs["pred_img_path"])

    def log(self, message):
        """记录日志到文件"""
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        print(message)

    def log_to_clearml(self, title, series, value, iteration):
        """记录数据到ClearML（如果启用）"""
        if self.use_clearml and self.clearml_logger:
            self.clearml_logger(title, series, value, iteration)
            
    def calculate_metrics(self, output, target, mask=None, threshold=0.001):
        """
        计算给定阈值下的POD, FAR, CSI指标，支持掩码
        Args:
            output: 预测值
            target: 目标值
            mask: 掩码，True表示是NaN位置（需要忽略）
            threshold: 阈值
        Returns:
            pod, far, csi: 评估指标
        """
        # 如果没有提供掩码，创建一个全False的掩码（不忽略任何位置）
        if mask is None:
            mask = torch.zeros_like(target, dtype=torch.bool)
        
        # 创建有效位置的掩码（非NaN位置）
        valid_mask = ~mask
        
        # 将预测值和真实值二值化，只考虑有效位置
        pred_binary = (output > threshold).float() * valid_mask.float()
        true_binary = (target > threshold).float() * valid_mask.float()
        
        # 计算TP, FP, FN
        tp = torch.sum((pred_binary == 1) & (true_binary == 1)).item()
        fp = torch.sum((pred_binary == 1) & (true_binary == 0)).item()
        fn = torch.sum((pred_binary == 0) & (true_binary == 1)).item()
        
        # 计算指标
        pod = tp / (tp + fn) if (tp + fn) > 0 else 0
        far = fp / (tp + fp) if (tp + fp) > 0 else 0
        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        return pod, far, csi

    def train(self):
        """训练模型"""
        self.log(f"开始训练，总共 {self.configs['epoch']} 个epoch")

        # 直接训练所有epoch
        self.train_all_epochs()

        self.log("训练完成！")



    def train_all_epochs(self):
        """训练所有epoch"""
        # 只在开始时输出一次样本统计
        if not hasattr(self, '_logged_sample_info'):
            print(f"使用一次性训练策略:")
            print(f"   训练集样本数: {len(self.train_dataset)}")
            print(f"   训练集批次数: {len(self.train_data)}")
            print(f"🚀 开始训练...")
            self._logged_sample_info = True

        train_loss = []  # 训练损失记录
        global_step = getattr(self, 'global_step', 0)  # 全局步数，用于ClearML记录
        for epoch in range(self.cur_epoch, self.configs["epoch"] + 1):
            t = tqdm(
                self.train_data, desc="训练中", leave=True, total=len(self.train_data)
            )
            epoch_loss = []
            epoch_mse = []
            
            # 用于计算每个epoch的评估指标
            metrics_data = {th: {'tp': 0, 'fp': 0, 'fn': 0} for th in self.thresholds}
            
            for batch in t:
                # 解包批次数据（现在包含掩码）
                input_ims, input_masks, label_ims, label_masks = batch
                
                self.model.train()  # 设置为训练模式
                # 数据预处理：reshape
                input_ims = PreProcess.reshape_patch(
                    input_ims, self.configs["patch_size"]
                )
                input_ims = input_ims.to(self.configs["device"])   # 移动到指定设备
                label_ims = label_ims.to(self.configs["device"])   # 移动到指定设备
                
                # 将掩码也移动到设备并reshape
                input_masks = PreProcess.reshape_patch(
                    input_masks.float(), self.configs["patch_size"]
                ).bool()
                input_masks = input_masks.to(self.configs["device"])
                label_masks = label_masks.to(self.configs["device"])
                
                # 模型前向传播
                output_ims = self.model(input_ims)
                output_ims = PreProcess.reshape_patch_back(
                    output_ims, self.configs["patch_size"]
                )  # 反向reshape

                self.optimizer.zero_grad()  # 清零梯度
                # 计算损失（考虑掩码）
                loss = F.mse_loss(output_ims, label_ims, reduction='mean')  # 默认就是 mean
                
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新参数
                self.scheduler.step()  # 更新学习率

                # 计算MSE（不带权重，但考虑掩码）
                valid_mask = ~label_masks
                if valid_mask.sum() > 0:
                    mse = torch.sum(((output_ims - label_ims) ** 2) * valid_mask.float()) / valid_mask.sum()
                else:
                    mse = torch.tensor(0.0, device=output_ims.device)
                

                # 记录到ClearML - 每个指标单独记录
                # self.log_to_clearml("Loss", "Train", loss.item(), global_step)
                # self.log_to_clearml("MSE", "Train", mse.item(), global_step)
                # self.log_to_clearml("Learning Rate", "Train", self.optimizer.param_groups[0]['lr'], global_step)
                
                # 累计每个batch的评估指标数据

                
                global_step += 1
                epoch_loss.append(loss.detach().cpu().numpy().item())  # 记录损失
                epoch_mse.append(mse.detach().cpu().numpy().item())    # 记录MSE
                t.set_postfix(
                    {
                        "loss": "{:.6f}".format(loss.detach().cpu().numpy().item()),
                        "mse": "{:.6f}".format(mse.detach().cpu().numpy().item()),
                        "epoch": "{}".format(epoch),
                    }
                )
            
            # 计算并记录平均训练损失和MSE
            avg_train_loss = np.mean(epoch_loss)
            avg_train_mse = np.mean(epoch_mse)
            train_loss.append(avg_train_loss)
            self.log(f"Epoch {epoch}/{self.configs['epoch']} - 平均训练损失: {avg_train_loss:.6f}, MSE: {avg_train_mse:.6f}")
            
            # 每个epoch进行验证
            avg_mse, val_metrics = self.validate(epoch)
            
            # 记录到ClearML - Epoch级别指标
            self.log_to_clearml("Epoch Loss", "Train", avg_train_loss, epoch)
            self.log_to_clearml("Epoch MSE", "Train", avg_train_mse, epoch)
            self.log_to_clearml("Epoch MSE", "Validation", avg_mse, epoch)
            self.log_to_clearml("Learning Rate", "Train", self.optimizer.param_groups[0]['lr'], epoch)
            
            # 保存最优模型
            if avg_mse < self.best_mse:
                self.best_mse = avg_mse
                self.best_epoch = epoch
                self.save(epoch, is_best=True)
                self.log(f"新的最佳模型! Epoch {epoch} - MSE: {avg_mse:.6f}")
            
            
            self.log(f"当前最佳模型: Epoch {self.best_epoch} - MSE: {self.best_mse:.6f}")

        # 更新当前epoch
        self.cur_epoch = self.configs["epoch"] + 1
        self.global_step = global_step  # 保存全局步数

        # 保存最终模型
        self.save(self.configs["epoch"], is_best=False, is_final=True)
        self.log(f"训练完成! 最佳模型: Epoch {self.best_epoch} - MSE: {self.best_mse:.6f}")

    def validate(self, epoch):
        """验证模型性能"""
        self.model.eval()  # 设置为评估模式
        valid_loss = []
        mse_values = []
        
        # 用于计算评估指标
        metrics_data = {th: {'pod': [], 'far': [], 'csi': []} for th in self.thresholds}
        
        with torch.no_grad():
            for batch in tqdm(self.val_data, desc="验证中", leave=False):
                # 解包批次数据（现在包含掩码）
                input_ims, input_masks, label_ims, label_masks = batch
                
                # 数据预处理：reshape
                input_ims = PreProcess.reshape_patch(
                    input_ims, self.configs["patch_size"]
                )
                input_ims = input_ims.to(self.configs["device"])
                label_ims = label_ims.to(self.configs["device"])
                
                # 将掩码也移动到设备并reshape
                input_masks = PreProcess.reshape_patch(
                    input_masks.float(), self.configs["patch_size"]
                ).bool()
                input_masks = input_masks.to(self.configs["device"])
                label_masks = label_masks.to(self.configs["device"])
                
                # 模型前向传播
                output_ims = self.model(input_ims)
                output_ims = PreProcess.reshape_patch_back(
                    output_ims, self.configs["patch_size"]
                )
                
                # 计算损失（考虑掩码）
                loss = self.loss_fn(output_ims, label_ims, label_masks)
                valid_loss.append(loss.item())
                
                # 计算MSE（不带权重，但考虑掩码）
                valid_mask = ~label_masks
                if valid_mask.sum() > 0:
                    mse = torch.sum(((output_ims - label_ims) ** 2) * valid_mask.float()) / valid_mask.sum()
                else:
                    mse = torch.tensor(0.0, device=output_ims.device)
                mse_values.append(mse.item())
                
                # 计算评估指标
                for th in self.thresholds:
                    pod, far, csi = self.calculate_metrics(output_ims, label_ims, label_masks, th)
                    metrics_data[th]['pod'].append(pod)
                    metrics_data[th]['far'].append(far)
                    metrics_data[th]['csi'].append(csi)
        
        # 计算平均损失和MSE
        avg_valid_loss = np.mean(valid_loss)
        avg_mse = np.mean(mse_values)
        
        # 计算并记录平均评估指标
        val_metrics = {}
        for th in self.thresholds:
            avg_pod = np.mean(metrics_data[th]['pod'])
            avg_far = np.mean(metrics_data[th]['far'])
            avg_csi = np.mean(metrics_data[th]['csi'])
            
            val_metrics[th] = {
                'pod': avg_pod,
                'far': avg_far,
                'csi': avg_csi
            }
            
            threshold_str = f"{th*100:.1f}mm"
            self.log(f"验证指标 (阈值 {threshold_str}/h) - POD: {avg_pod:.4f}, FAR: {avg_far:.4f}, CSI: {avg_csi:.4f}")
            
            # 记录到ClearML
            self.log_to_clearml(f"POD", f"Validation {threshold_str}", avg_pod, epoch)
            self.log_to_clearml(f"FAR", f"Validation {threshold_str}", avg_far, epoch)
            self.log_to_clearml(f"CSI", f"Validation {threshold_str}", avg_csi, epoch)
        
        self.log(f"验证 - 平均损失: {avg_valid_loss:.6f}, MSE: {avg_mse:.6f}")
        return avg_mse, val_metrics

    def test(self):
        """测试模型性能"""
        self.model.eval()  # 设置为评估模式
        test_loss = []
        mse_values = []
        
        # 用于计算评估指标
        metrics_data = {th: {'pod': [], 'far': [], 'csi': []} for th in self.thresholds}
        
        # 为保存预测结果创建目录
        pred_dir = self.configs["pred_img_path"]
        os.makedirs(pred_dir, exist_ok=True)
        
        # 样本计数器
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_data, desc="测试中", leave=False)):
                # 解包批次数据（测试模式包含时间信息）
                if len(batch) == 6:  # 测试模式返回6个元素
                    input_ims, input_masks, label_ims, label_masks, input_times, target_times = batch
                else:  # 训练/验证模式返回4个元素
                    input_ims, input_masks, label_ims, label_masks = batch
                    input_times, target_times = None, None
                
                # 数据预处理：reshape
                input_ims = PreProcess.reshape_patch(
                    input_ims, self.configs["patch_size"]
                )
                input_ims = input_ims.to(self.configs["device"])
                label_ims = label_ims.to(self.configs["device"])
                
                # 将掩码也移动到设备并reshape
                input_masks = PreProcess.reshape_patch(
                    input_masks.float(), self.configs["patch_size"]
                ).bool()
                input_masks = input_masks.to(self.configs["device"])
                label_masks = label_masks.to(self.configs["device"])
                
                # 模型前向传播
                output_ims = self.model(input_ims)
                output_ims = PreProcess.reshape_patch_back(
                    output_ims, self.configs["patch_size"]
                )
                
                # 计算损失（考虑掩码）
                loss = self.loss_fn(output_ims, label_ims, label_masks)
                test_loss.append(loss.item())
                
                # 计算MSE（不带权重，但考虑掩码）
                valid_mask = ~label_masks
                if valid_mask.sum() > 0:
                    mse = torch.sum(((output_ims - label_ims) ** 2) * valid_mask.float()) / valid_mask.sum()
                else:
                    mse = torch.tensor(0.0, device=output_ims.device)
                mse_values.append(mse.item())
                
                # 计算评估指标
                for th in self.thresholds:
                    pod, far, csi = self.calculate_metrics(output_ims, label_ims, label_masks, th)
                    metrics_data[th]['pod'].append(pod)
                    metrics_data[th]['far'].append(far)
                    metrics_data[th]['csi'].append(csi)
                
                # 保存所有预测结果
                    # 为当前批次创建目录
                    batch_dir = os.path.join(pred_dir, f"batch_{batch_idx}")
                    os.makedirs(batch_dir, exist_ok=True)
                    
                    # 每个样本在批次中保存
                    for i in range(input_ims.shape[0]):
                        # 创建样本目录
                        sample_dir = os.path.join(batch_dir, f"sample_{sample_count}")
                        os.makedirs(sample_dir, exist_ok=True)
                        
                        # 将预测、真实值和输入移到CPU并转为NumPy数组
                        output_sample = output_ims[i].detach().cpu().numpy()
                        label_sample = label_ims[i].detach().cpu().numpy()
                        
                    # 如果有时间信息，保存时间信息
                    if target_times is not None:
                        # 保存时间信息到文件
                        time_info = {
                            'input_times': input_times[i] if input_times else [],
                            'target_times': target_times[i] if target_times else []
                        }
                        import json
                        with open(os.path.join(sample_dir, "time_info.json"), 'w') as f:
                            json.dump(time_info, f, indent=2)
                    
                        # 保存每个时间步的预测和真实值
                        for t in range(output_sample.shape[0]):
                            pred_file = os.path.join(sample_dir, f"pred_{t}.npy")
                            gt_file = os.path.join(sample_dir, f"gt_{t}.npy")
                            
                            # 保存为NumPy文件
                            np.save(pred_file, output_sample[t, 0])  # 只保存第一个通道
                            np.save(gt_file, label_sample[t, 0])  # 只保存第一个通道
                        
                        sample_count += 1
                    
                # 每处理完一个批次输出一次进度
                if batch_idx % 10 == 0:
                    self.log(f"已处理 {batch_idx + 1} 个批次，保存了 {sample_count} 个样本")
        
        # 计算平均损失和MSE
        avg_test_loss = np.mean(test_loss)
        avg_mse = np.mean(mse_values)
        
        # 计算并记录平均评估指标
        self.log(f"测试结果 - 平均损失: {avg_test_loss:.6f}, MSE: {avg_mse:.6f}")
        for th in self.thresholds:
            avg_pod = np.mean(metrics_data[th]['pod'])
            avg_far = np.mean(metrics_data[th]['far'])
            avg_csi = np.mean(metrics_data[th]['csi'])
            
            threshold_str = f"{th*100:.1f}mm"
            self.log(f"测试指标 (阈值 {threshold_str}/h) - POD: {avg_pod:.4f}, FAR: {avg_far:.4f}, CSI: {avg_csi:.4f}")
        
        self.log(f"预测结果保存在: {pred_dir}")
        self.log(f"总共保存了 {sample_count} 个测试样本")
        return avg_mse

    def save(self, epoch, is_best=False, is_final=False):
        """保存模型"""
        if is_best:
            save_path = os.path.join(self.configs["save_dir"], "best_model.ckpt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    # "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_mse": self.best_mse,
                },
                save_path,
            )
            self.log(f"保存最佳模型到 {save_path}")
        elif is_final:
            save_path = os.path.join(self.configs["save_dir"], "final_model.ckpt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    # "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_mse": self.best_mse,
                },
                save_path,
            )
            self.log(f"保存最终模型到 {save_path}")
        else:
            save_path = os.path.join(self.configs["save_dir"], f"model_epoch_{epoch}.ckpt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    # "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_mse": self.best_mse,
                },
                save_path,
            )
            self.log(f"保存检查点到 {save_path}")

    def load(self, checkpoint_path):
        """加载模型"""
        if not os.path.exists(checkpoint_path):
            self.log(f"警告：检查点 {checkpoint_path} 不存在，将从头开始训练")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.configs["device"])
        
        # 检查使用哪个键来加载模型参数
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "net_param" in checkpoint:
            self.model.load_state_dict(checkpoint["net_param"])
        else:
            self.log(f"错误：检查点文件中找不到模型参数，可用的键: {checkpoint.keys()}")
            return
        
        if self.configs["is_train"]:
            # if "optimizer_state_dict" in checkpoint:
            #     self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.cur_epoch = checkpoint.get("epoch", 0) + 1
            self.best_mse = checkpoint.get("best_mse", float('inf'))
            self.log(f"加载检查点 {checkpoint_path}, 从epoch {self.cur_epoch} 开始训练, 最佳MSE: {self.best_mse:.6f}")
        else:
            self.log(f"加载检查点 {checkpoint_path} 用于测试, 模型epoch: {checkpoint.get('epoch', 0)}") 