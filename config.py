# config.py

import os
import argparse
from datetime import datetime

class Config:
    def __init__(self):
        # --- 基础配置 ---
        self.project_name = "Saliency_2025"
        self.seed = 42
        self.device = 'cuda'
        self.k_folds = 5
        self.fold = 0
        
        # --- 路径配置 ---
        self.data_root = "./data/train"
        self.test_root = "./data/test"
        self.checkpoints_root = "./checkpoints"
        self.logs_root = "./logs"
        self.submission_dir = "./submissions"
        self.submission_prefix = "submission"
        # 默认提交命名为 submission_MMDD_HHMM.csv，与示例 submission_1216_2243 保持一致
        self.submission_timefmt = "%m%d_%H%M"
        
        # --- 核心训练参数 ---
        self.img_size = 512
        self.epochs = 60  # 数据增强后，可以适当多训练几轮
        self.num_workers = 16
        self.val_split = 0.1
        
        # --- 优化策略开关 ---
        self.use_augmentation = True
        self.use_scheduler = True
        self.early_stop_patience = 30 # 10个epoch没提升就停止

        # --- 损失函数权重 (总和为2.0) ---
        self.loss_weights = {
            "cc": 1.0,    # 主要目标
            "bce": 0.0,   # 像素对齐
            "ssim": 0.5   # 结构对齐
        }

        # --- 模型特定预设 ---
        self.model_presets = {
            "mit_b5": {
                "batch_size":16,     # 4090上跑512x512的稳定值
                "lr": 5e-5,          # Transformer 适合较小的学习率
                "weight_decay": 0.01,
                "warmup_epochs": 3   # 启动时预热3个epoch
            }
        }
        
        # --- 默认模型 ---
        self.backbone = "mit_b5"
        
        # --- 占位符 ---
        self.base_exp_name = ""
        self.run_tag = ""  # 兼容脚本：固定一次 5-fold 的实验名（不含 fold）
        self.batch_size = None
        self.lr = None
        self.weight_decay = None
        self.warmup_epochs = None
        self.exp_name = ""
        self.log_dir = ""
        self.ckpt_dir = ""
        # --- 评估/提交配置（可在 config.py 直接修改） ---
        # 优先使用 eval_ckpt_path；若为空则使用 eval_exp_name 对应的 ./checkpoints/<exp_name>/best_model.pth
        self.eval_ckpt_path = "checkpoints/mit_b5_1205_1544/best_model.pth"    # 示例: "./checkpoints/mit_b5_1205_1215/best_model.pth"
        self.eval_exp_name = ""     # 示例: "mit_b5_1205_1215"

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Saliency Prediction Training")
        parser.add_argument('--backbone', type=str, default=self.backbone, choices=['mit_b5'])
        parser.add_argument('--batch_size', type=int, default=None)
        parser.add_argument('--gpu_id', type=str, default="0")
        parser.add_argument('--note', type=str, default="", help='Add a suffix tag to experiment name')
        parser.add_argument('--fold', type=int, default=0, help='Fold index for K-Fold training')
        parser.add_argument('--exp_name', type=str, default=None, help='Fixed experiment name (shared across folds), do NOT include fold suffix')
        args = parser.parse_args()
        
        self.backbone = args.backbone
        self.fold = args.fold
        if not (0 <= self.fold < self.k_folds):
            raise ValueError(f"Fold index {self.fold} is out of range [0, {self.k_folds-1}]")
        preset = self.model_presets[self.backbone]
        
        self.batch_size = args.batch_size if args.batch_size is not None else preset['batch_size']
        self.lr = preset['lr']
        self.weight_decay = preset['weight_decay']
        self.warmup_epochs = preset['warmup_epochs']
        
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        
        # 生成/固定本次 5-fold 共享的实验名（不包含 fold 后缀）
        if args.exp_name:
            base_exp_name = args.exp_name
        else:
            timestamp = datetime.now().strftime("%m%d_%H%M")
            note = f"_{args.note}" if args.note else ""
            base_exp_name = f"{self.backbone}_{timestamp}{note}"

        self.run_tag = base_exp_name
        self.base_exp_name = base_exp_name
        self.exp_name = base_exp_name

        # 目录结构：checkpoints/<base_exp_name>/fold0..fold4
        self.log_dir = os.path.join(self.logs_root, base_exp_name, f"fold{args.fold}")
        self.ckpt_dir = os.path.join(self.checkpoints_root, base_exp_name, f"fold{args.fold}")
        return self

    def create_dirs(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def build_submission_name(self, save_name=None):
        """Return a submission filename; default is submission_<timestamp>.csv"""
        if save_name:
            return save_name
        timestamp = datetime.now().strftime(self.submission_timefmt)
        return f"{self.submission_prefix}_{timestamp}.csv"

cfg = Config()