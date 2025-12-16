# src/dataset.py

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_transforms(cfg, mode="train"):
    if mode == "train" and cfg.use_augmentation:
        return A.Compose([
            # 1. 强制调整到 512x512 (不缩放，不剪裁，保证内容完整)
            A.Resize(cfg.img_size, cfg.img_size),
            
            # 2. 唯一保留的安全增强：水平翻转
            A.HorizontalFlip(p=0.5),
            
            # 3. 标准化
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        # 验证集保持不变
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        # 防御性检查：确保 mask 是单通道 (H, W)，避免变成 (H, W, 3) 触发 Albumentations 形状错误
        if mask is None:
            raise FileNotFoundError(f"Mask not found or unreadable: {self.mask_paths[idx]}")
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented['image']
        mask_tensor = augmented['mask']
        
        # 归一化 mask 并增加通道维度
        mask_tensor = mask_tensor.float() / 255.0
        return image_tensor, mask_tensor.unsqueeze(0)

def get_dataloaders(cfg):
    stimuli_dir = os.path.join(cfg.data_root, "Stimuli")
    all_image_paths = sorted(glob(os.path.join(stimuli_dir, "*", "*.jpg")))
    
    valid_images, valid_masks, categories = [], [], []
    for img_path in all_image_paths:
        mask_path = img_path.replace("Stimuli", "FIXATIONMAPS")
        if os.path.exists(mask_path):
            valid_images.append(img_path)
            valid_masks.append(mask_path)
            categories.append(os.path.basename(os.path.dirname(img_path)))

    train_img, val_img, train_mask, val_mask = train_test_split(
        valid_images, valid_masks, test_size=cfg.val_split, random_state=cfg.seed, stratify=categories
    )
    
    train_ds = SaliencyDataset(train_img, train_mask, transform=build_transforms(cfg, 'train'))
    val_ds = SaliencyDataset(val_img, val_mask, transform=build_transforms(cfg, 'val'))
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    
    return train_loader, val_loader