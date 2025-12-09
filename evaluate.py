# evaluate.py

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 导入你项目中的模块
from config import cfg
from src.models import build_model
from src.metric import calc_cc_score

def evaluate_model(args):
    """
    加载指定模型权重，对测试集进行推理，并计算平均 CC 分数。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载模型 (与 inference.py 相同)
    model = build_model(args.backbone).to(device)
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {args.ckpt_path}")
    state = torch.load(args.ckpt_path, map_location=device)
    # 支持两种 checkpoint 格式：
    # 1) 直接保存的 model state_dict
    # 2) 包含元信息的 checkpoint dict（含 'model_state' 键）
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)
    model.eval()
    print(f"Model '{args.backbone}' loaded successfully from '{args.ckpt_path}'")

    # 2. 准备数据预处理
    transform = A.Compose([
        A.Resize(cfg.img_size, cfg.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # 3. 遍历测试集并评估
    all_scores = []
    test_stimuli_dir = os.path.join(cfg.test_root, "Stimuli")
    
    print("\nStarting evaluation on the test set...")
    
    # 自动查找所有测试图片
    image_paths = sorted([
        os.path.join(dp, f) 
        for dp, dn, fn in os.walk(os.path.expanduser(test_stimuli_dir)) 
        for f in fn if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {test_stimuli_dir}")

    for img_path in tqdm(image_paths, desc="Evaluating"):
        # --- 读取图片和真值 ---
        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        gt_path = img_path.replace("Stimuli", "FIXATIONMAPS")
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth not found for {img_path}, skipping.")
            continue
        gt_map = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype('float32') / 255.0

        # --- TTA 推理 (原图 + 水平翻转) ---
        images_tta = [image_rgb, cv2.flip(image_rgb, 1)]
        preds_tta = []
        
        with torch.no_grad():
            for img_aug in images_tta:
                img_tensor = transform(image=img_aug)['image'].unsqueeze(0).to(device)
                pred = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
                preds_tta.append(pred)
        
        # --- 融合 TTA 结果 ---
        pred_hflip_restored = cv2.flip(preds_tta[1], 1)
        final_pred_small = np.mean([preds_tta[0], pred_hflip_restored], axis=0)
        
        # --- 后处理并计算分数 ---
        pred_resized = cv2.resize(final_pred_small, (w, h))
        score = calc_cc_score(gt_map, pred_resized)
        all_scores.append(score)

    # 4. 打印最终结果
    mean_cc = np.mean(all_scores)
    std_cc = np.std(all_scores)
    
    print("\n" + "="*30)
    print("      EVALUATION COMPLETE")
    print("="*30)
    print(f"  - Model: {args.backbone}")
    print(f"  - Checkpoint: {args.ckpt_path}")
    print(f"  - Images Evaluated: {len(all_scores)}")
    print(f"  - Mean CC Score: {mean_cc:.6f}")
    print(f"  - Std Dev of Scores: {std_cc:.6f}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Saliency Prediction Model")
    parser.add_argument('--backbone', type=str, required=False, choices=['mit_b5'], 
                        help="The model architecture backbone. If omitted, will use value from config.py")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--ckpt_path', type=str, 
                        help="Path to the trained model checkpoint (.pth file). If provided, used directly.")
    group.add_argument('--exp_name', type=str,
                        help="Experiment name under ./checkpoints to use its best model (uses ./checkpoints/<exp_name>/best_model.pth)")
    parser.add_argument('--list', action='store_true', help='List available experiments under ./checkpoints and exit')
    
    args = parser.parse_args()

    # If backbone not provided via CLI, fallback to cfg
    if not args.backbone:
        args.backbone = cfg.backbone

    # If requested, list available experiments and exit
    if args.list:
        if os.path.exists('checkpoints'):
            for d in sorted(os.listdir('checkpoints')):
                print(d)
        else:
            print('No checkpoints directory found.')
        exit(0)

    # Resolve ckpt_path: CLI > cfg.eval_ckpt_path > cfg.eval_exp_name
    if not args.ckpt_path and not args.exp_name:
        if cfg.eval_ckpt_path:
            args.ckpt_path = cfg.eval_ckpt_path
        elif cfg.eval_exp_name:
            candidate = os.path.join('checkpoints', cfg.eval_exp_name, 'best_model.pth')
            if os.path.exists(candidate):
                args.ckpt_path = candidate
            else:
                raise FileNotFoundError(f"Configured eval_exp_name '{cfg.eval_exp_name}' has no best_model at: {candidate}")
        else:
            raise ValueError('No checkpoint specified. Provide --ckpt_path/--exp_name or set cfg.eval_ckpt_path / cfg.eval_exp_name in config.py')

    # If exp_name provided via CLI, override
    if args.exp_name:
        candidate = os.path.join('checkpoints', args.exp_name, 'best_model.pth')
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Best model not found for experiment '{args.exp_name}' at: {candidate}")
        args.ckpt_path = candidate

    evaluate_model(args)