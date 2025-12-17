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
        self.submission_timefmt = "%m%d_%H%M"
        
        # --- [关键修改] 伪标签配置 ---
        # Stage 1 (此时没伪标签): 设为 False
        # Stage 2 (生成伪标签后): 设为 True
        self.use_pseudo = False  
        
        # 伪标签存放根目录 (对应 gen_pseudo.py 的输出)
        self.pseudo_root = "./data/pseudo" 

        # --- 核心训练参数 ---
        self.img_size = 512
        self.epochs = 60
        self.num_workers = 16
        self.val_split = 0.1
        
        # --- 优化策略开关 ---
        self.use_augmentation = True
        self.use_scheduler = True
        self.early_stop_patience = 30 

        # --- 损失函数权重 ---
        self.loss_weights = {
            "cc": 1.0,
            "bce": 0.0,
            "ssim": 0.5
        }

        # --- 模型预设 ---
        self.model_presets = {
            "mit_b5": {
                "batch_size": 8,     
                "lr": 5e-5,          
                "weight_decay": 0.01,
                "warmup_epochs": 3   
            },
            "transalnet": {
                "batch_size": 4,      # 显存允许的话改 8
                "lr": 1e-5,           # 微调 LR 要小
                "weight_decay": 1e-4, 
                "warmup_epochs": 0    
            }
        }
        
        # 默认使用核武器
        self.backbone = "transalnet"
        
        # --- 内部变量 ---
        self.base_exp_name = ""
        self.run_tag = "" 
        self.batch_size = None
        self.lr = None
        self.weight_decay = None
        self.warmup_epochs = None
        self.exp_name = ""
        self.log_dir = ""
        self.ckpt_dir = ""
        self.eval_ckpt_path = "" 
        self.eval_exp_name = ""     

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Saliency Prediction Training")
        parser.add_argument('--backbone', type=str, default=self.backbone, choices=['mit_b5', 'transalnet'])
        parser.add_argument('--batch_size', type=int, default=None)
        parser.add_argument('--gpu_id', type=str, default=None)
        parser.add_argument('--note', type=str, default="", help='Suffix for exp name')
        parser.add_argument('--fold', type=int, default=0)
        parser.add_argument('--exp_name', type=str, default=None)
        
        # 允许从命令行覆盖伪标签开关： --use_pseudo
        parser.add_argument('--use_pseudo', action='store_true', help='Enable pseudo label training')
        
        args = parser.parse_args()
        
        self.backbone = args.backbone
        self.fold = args.fold
        
        # 如果命令行指定了 --use_pseudo，则强制开启
        if args.use_pseudo:
            self.use_pseudo = True

        preset = self.model_presets[self.backbone]
        self.batch_size = args.batch_size if args.batch_size is not None else preset['batch_size']
        self.lr = preset['lr']
        self.weight_decay = preset['weight_decay']
        self.warmup_epochs = preset['warmup_epochs']
        
        if args.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        
        if args.exp_name:
            base_exp_name = args.exp_name
        else:
            timestamp = datetime.now().strftime("%m%d_%H%M")
            note = f"_{args.note}" if args.note else ""
            pseudo_tag = "_PL" if self.use_pseudo else ""
            base_exp_name = f"{self.backbone}{pseudo_tag}_{timestamp}{note}"

        self.run_tag = base_exp_name
        self.base_exp_name = base_exp_name
        self.exp_name = base_exp_name

        self.log_dir = os.path.join(self.logs_root, base_exp_name, f"fold{args.fold}")
        self.ckpt_dir = os.path.join(self.checkpoints_root, base_exp_name, f"fold{args.fold}")
        return self

    def create_dirs(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def build_submission_name(self, save_name=None):
        if save_name:
            return save_name
        timestamp = datetime.now().strftime(self.submission_timefmt)
        return f"{self.submission_prefix}_{timestamp}.csv"

cfg = Config()