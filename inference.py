# inference.py

import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import cfg
from src.models import build_model
from src.metric import calc_cc_score

def _load_model_weights(model, state):
    """智能加载权重：去除 module. 前缀，支持 strict=False"""
    weights = state
    raw_key = "direct"
    if isinstance(state, dict):
        for key in ['model_state', 'state_dict', 'model', 'model_state_dict']:
            if key in state:
                weights = state[key]
                raw_key = key
                break
    
    # 去除 'module.' 前缀
    new_weights = {}
    for k, v in weights.items():
        name = k[7:] if k.startswith('module.') else k
        new_weights[name] = v
    
    try:
        model.load_state_dict(new_weights, strict=True)
        return f"{raw_key} (strict=True)"
    except RuntimeError as e:
        print(f"Warning: Strict loading failed. Retrying with strict=False...")
        model.load_state_dict(new_weights, strict=False)
        return f"{raw_key} (strict=False)"

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = build_model(args.backbone).to(device)
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
        
    state = torch.load(args.ckpt_path, map_location=device)
    used_key = _load_model_weights(model, state)
    model.eval()
    print(f"Model loaded successfully using format: {used_key}")
    
    transform = A.Compose([
        A.Resize(cfg.img_size, cfg.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    results = []
    test_stimuli_dir = os.path.join(cfg.test_root, "Stimuli")
    
    print("Starting Inference with TTA...")
    # 递归查找所有图片，确保覆盖所有子文件夹
    image_paths = sorted([
        os.path.join(dp, f) 
        for dp, dn, fn in os.walk(os.path.expanduser(test_stimuli_dir)) 
        for f in fn if f.endswith(('.jpg', '.png', '.jpeg'))
    ])

    for img_path in tqdm(image_paths):
        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # TTA
        images_tta = [image_rgb, cv2.flip(image_rgb, 1)]
        preds_tta = []
        
        with torch.no_grad():
            for img_aug in images_tta:
                img_tensor = transform(image=img_aug)['image'].unsqueeze(0).to(device)
                pred = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
                preds_tta.append(pred)
        
        pred_hflip = cv2.flip(preds_tta[1], 1)
        final_pred = np.mean([preds_tta[0], pred_hflip], axis=0)
        
        pred_resized = cv2.resize(final_pred, (w, h))
        
        # 计算分数（如果有真值）
        gt_path = img_path.replace("Stimuli", "FIXATIONMAPS")
        score = 0.0
        if os.path.exists(gt_path):
            gt_map = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype('float32') / 255.0
            score = calc_cc_score(gt_map, pred_resized)
        
        # 生成 ID: 类别/文件名 (假设结构是 test/Stimuli/类别/文件名)
        # 兼容不同层级的目录结构
        rel_path = os.path.relpath(img_path, test_stimuli_dir)
        # 统一转为 Linux 风格路径分隔符
        file_id = rel_path.replace(os.sep, '/')
        
        results.append({"id": file_id, "score": score})

    df = pd.DataFrame(results)
    save_path = f"./submissions/submission_{args.backbone}_tta.csv"
    os.makedirs("./submissions", exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"\nInference Done! Submission saved to: {save_path}")
    if df['score'].sum() > 0:
        print(f"Mean CC Score with TTA: {df['score'].mean():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, required=False, choices=['mit_b5'])
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to your best_model.pth')
    args = parser.parse_args()
    
    if not args.backbone:
        args.backbone = cfg.backbone
        
    main(args)