# gen_pseudo.py

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 导入项目模块
from config import cfg
from src.models import build_model

def generate_pseudo_labels():
    # --- 配置区 ---
    # 1. 这里填你 Stage 1 跑出来的最好的模型路径 (通常用 Fold 0 的 best_model 或者 SWA)
    #    建议：先跑完 Stage 1，然后去 checkpoints/ 找一个 val_cc 最高的模型填在这里
    MODEL_PATH = "checkpoints/mit_b5_xxxx_kfold/fold0/best_model.pth" 
    
    # 2. 伪标签保存路径 (会自动创建)
    OUTPUT_DIR = "./data/pseudo"
    # ----------------
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载模型
    print(f"Loading model from {MODEL_PATH} ...")
    model = build_model(cfg.backbone).to(device)
    
    # 智能加载权重 (处理 module. 前缀)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint.get('model_state', checkpoint) # 兼容不同保存格式
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 2. 准备测试集数据
    test_stimuli_dir = os.path.join(cfg.test_root, "Stimuli")
    if not os.path.exists(test_stimuli_dir):
        raise FileNotFoundError(f"Test data not found at {test_stimuli_dir}")
        
    image_names = [f for f in os.listdir(test_stimuli_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(image_names)} test images. Starting inference...")

    # 3. 定义预处理 (必须和训练时一致)
    transform = A.Compose([
        A.Resize(cfg.img_size, cfg.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # 4. 伪标签保存目录结构： data/pseudo/FIXATIONMAPS/xxx.png
    #    这样设计是为了配合 dataset.py 的读取逻辑
    save_mask_dir = os.path.join(OUTPUT_DIR, "FIXATIONMAPS")
    os.makedirs(save_mask_dir, exist_ok=True)

    # 5. 开始生成
    with torch.no_grad():
        for img_name in tqdm(image_names, desc="Generating Pseudo Labels"):
            img_path = os.path.join(test_stimuli_dir, img_name)
            
            # 读取原图
            image = cv2.imread(img_path)
            h, w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # --- TTA (Test Time Augmentation) ---
            # 策略：原图预测 + 水平翻转预测 -> 求平均
            
            # 1. 正向
            aug1 = transform(image=image_rgb)['image'].unsqueeze(0).to(device)
            pred1 = torch.sigmoid(model(aug1))
            
            # 2. 翻转
            img_flip = cv2.flip(image_rgb, 1)
            aug2 = transform(image=img_flip)['image'].unsqueeze(0).to(device)
            pred2 = torch.sigmoid(model(aug2))
            pred2_flip = torch.flip(pred2, [3]) # 翻转回来
            
            # 3. 平均
            pred_ensemble = (pred1 + pred2_flip) / 2.0
            
            # --- 后处理 ---
            # 还原到原图尺寸
            pred_np = pred_ensemble.squeeze().cpu().numpy()
            pred_resized = cv2.resize(pred_np, (w, h))
            
            # 保存为 0-255 的灰度图 (Soft Label)
            # 这里保存为 png 无损格式
            pred_uint8 = (pred_resized * 255).astype(np.uint8)
            
            # 保存文件名保持一致，方便 Dataset 匹配
            save_name = os.path.splitext(img_name)[0] + ".png"
            cv2.imwrite(os.path.join(save_mask_dir, save_name), pred_uint8)

    print(f"\n Pseudo labels generated at: {save_mask_dir}")
    print("Next step: Set 'use_pseudo = True' in config.py and re-run training.")

if __name__ == "__main__":
    generate_pseudo_labels()