# src/loss.py

import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure

class SaliencyLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weights = cfg.loss_weights
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