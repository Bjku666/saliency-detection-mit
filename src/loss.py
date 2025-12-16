# src/loss.py

import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure

class SaliencyLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 默认权重采用 CC + KLD 的复合分布损失
        default_weights = {"cc": 1.0, "kld": 1.0, "bce": 0.0, "ssim": 0.0}
        self.weights = {**default_weights, **getattr(cfg, "loss_weights", {})}
        self.eps = 1e-8
        # BCEWithLogitsLoss 自带 Sigmoid，数值更稳定
        self.bce = nn.BCEWithLogitsLoss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(cfg.device)

    def cc_metric(self, pred, target):
        """计算 CC，输入必须是 [0,1] 的概率图"""
        # 展平
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        vx = pred_flat - pred_flat.mean(dim=1, keepdim=True)
        vy = target_flat - target_flat.mean(dim=1, keepdim=True)

        numerator = torch.sum(vx * vy, dim=1)
        denominator = torch.sqrt(torch.sum(vx ** 2, dim=1)) * torch.sqrt(torch.sum(vy ** 2, dim=1))
        
        return numerator / (denominator + 1e-8)

    def kld_loss(self, pred_probs, target):
        """KLD 计算前进行 Sum 归一化，避免负值与数值爆炸"""
        pred_norm = pred_probs / (pred_probs.sum(dim=(1, 2, 3), keepdim=True) + self.eps)
        target_norm = target / (target.sum(dim=(1, 2, 3), keepdim=True) + self.eps)

        pred_norm = pred_norm.clamp(min=self.eps)
        target_norm = target_norm.clamp(min=self.eps)

        kld = target_norm * (torch.log(target_norm) - torch.log(pred_norm))
        return kld.sum(dim=(1, 2, 3))

    def forward(self, pred_logits, target):
        """
        pred_logits: 模型直接输出的 logits (未经过 sigmoid)
        target: 真值 [0, 1]
        """
        total_loss = torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype)
        
        # 1. 先把 logits 转成 [0,1] 用于计算 CC 和 SSIM
        pred_probs = torch.sigmoid(pred_logits)

        # 2. 计算 CC Loss (权重最大)
        if self.weights.get("cc", 0) > 0:
            cc = self.cc_metric(pred_probs, target)
            loss_cc = (1.0 - cc).mean()
            total_loss += self.weights["cc"] * loss_cc

        # 2+. 计算 KLD (需要概率图)，使用 Sum 归一化
        if self.weights.get("kld", 0) > 0:
            kld_batch = self.kld_loss(pred_probs, target)
            loss_kld = kld_batch.mean()
            total_loss += self.weights["kld"] * loss_kld

        # 3. 计算 BCE Loss (注意：这里用 logits)
        if self.weights.get("bce", 0) > 0:
            loss_bce = self.bce(pred_logits, target)
            total_loss += self.weights["bce"] * loss_bce

        # 4. 计算 SSIM (用 probs)
        if self.weights.get("ssim", 0) > 0:
            ssim_val = self.ssim(pred_probs, target)
            loss_ssim = 1.0 - ssim_val
            total_loss += self.weights["ssim"] * loss_ssim
        
        return total_loss