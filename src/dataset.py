# src/dataset.py

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from sklearn.model_selection import KFold
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_transforms(cfg, mode="train"):
    if mode == "train" and cfg.use_augmentation:
        return A.Compose([
            A.Resize(cfg.img_size, cfg.img_size),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(cfg.img_size, cfg.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

class SaliencyDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise FileNotFoundError(f"Image not found: {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {self.mask_paths[idx]}")
        
        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented['image']
        mask_tensor = augmented['mask']
        
        # 归一化 mask [0, 255] -> [0, 1]
        mask_tensor = mask_tensor.float() / 255.0
        return image_tensor, mask_tensor.unsqueeze(0)

def get_dataloaders(cfg):
    # --- 1. 读取真实训练集 ---
    stimuli_dir = os.path.join(cfg.data_root, "Stimuli")
    all_image_paths = sorted(glob(os.path.join(stimuli_dir, "*", "*.jpg")))

    real_images, real_masks = [], []
    for img_path in all_image_paths:
        mask_path = img_path.replace("Stimuli", "FIXATIONMAPS")
        if os.path.exists(mask_path):
            real_images.append(img_path)
            real_masks.append(mask_path)
            
    if len(real_images) == 0:
        raise RuntimeError("No valid real data found!")

    # --- 2. 读取伪标签 (Stage 2) ---
    pseudo_images, pseudo_masks = [], []
    if cfg.use_pseudo:
        # 伪标签 Mask 目录
        pseudo_mask_dir = os.path.join(cfg.pseudo_root, "FIXATIONMAPS")
        # 测试集原图目录 (伪标签是基于测试集生成的)
        test_stimuli_dir = os.path.join(cfg.test_root, "Stimuli")
        
        if os.path.exists(pseudo_mask_dir):
            # 查找所有生成的 png 伪标签
            found_masks = sorted(glob(os.path.join(pseudo_mask_dir, "*.png")))
            
            for pm in found_masks:
                # 伪标签文件名应与原图同名 (除后缀可能不同)
                # gen_pseudo.py 生成的是 .png，原图可能是 .jpg
                base_name = os.path.splitext(os.path.basename(pm))[0]
                
                # 在测试集目录里找对应的原图
                # 简单尝试 .jpg, .png, .jpeg
                img_candidate = None
                for ext in ['.jpg', '.png', '.jpeg']:
                    candidate = os.path.join(test_stimuli_dir, base_name + ext)
                    if os.path.exists(candidate):
                        img_candidate = candidate
                        break
                
                if img_candidate:
                    pseudo_images.append(img_candidate)
                    pseudo_masks.append(pm)
            
            print(f" [Dataset] Loaded {len(pseudo_images)} pseudo-label pairs from {pseudo_mask_dir}")
        else:
            print(f" [Dataset] Warning: use_pseudo=True but {pseudo_mask_dir} does not exist.")

    # --- 3. K-Fold 切分 (只切分真实数据) ---
    indices = np.arange(len(real_images))
    kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=42)

    selected = None
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        if fold_idx == cfg.fold:
            selected = (train_idx, val_idx)
            break

    if selected is None:
        raise ValueError(f"Fold index {cfg.fold} is invalid")

    train_idx, val_idx = selected
    
    # --- 4. 组装数据 ---
    # 训练集 = 真实训练集部分 + 所有伪标签数据
    train_img_paths = [real_images[i] for i in train_idx] + pseudo_images
    train_mask_paths = [real_masks[i] for i in train_idx] + pseudo_masks
    
    # 验证集 = 仅真实验证集部分 (严禁污染)
    val_img_paths = [real_images[i] for i in val_idx]
    val_mask_paths = [real_masks[i] for i in val_idx]

    print(f" Fold {cfg.fold} Summary:")
    print(f"  - Train: {len(train_img_paths)} samples (Real: {len(train_idx)} + Pseudo: {len(pseudo_images)})")
    print(f"  - Val:   {len(val_img_paths)} samples (Real only)")

    # 构建 Dataset
    train_ds = SaliencyDataset(train_img_paths, train_mask_paths, transform=build_transforms(cfg, 'train'))
    val_ds = SaliencyDataset(val_img_paths, val_mask_paths, transform=build_transforms(cfg, 'val'))

    # 构建 DataLoader
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers, 
        pin_memory=True, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        num_workers=cfg.num_workers, 
        pin_memory=True
    )

    return train_loader, val_loader