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

def _load_model_weights(model, state):
    """
    智能加载权重：
    1. 自动识别嵌套字典 (model_state, state_dict 等)
    2. 自动去除 DataParallel 产生的 'module.' 前缀
    3. 支持 strict=False 加载 (针对 SWA 模型多余 buffer 的情况)
    """
    # 1. 尝试提取真正的权重字典
    weights = state
    raw_key = "direct"
    if isinstance(state, dict):
        for key in ['model_state', 'state_dict', 'model', 'model_state_dict']:
            if key in state:
                weights = state[key]
                raw_key = key
                break
    
    # 2. 去除 'module.' 前缀 (DataParallel 遗留问题)
    new_weights = {}
    for k, v in weights.items():
        name = k[7:] if k.startswith('module.') else k
        new_weights[name] = v
    
    # 3. 尝试加载
    try:
        model.load_state_dict(new_weights, strict=True)
        return f"{raw_key} (strict=True)"
    except RuntimeError as e:
        # 如果严格加载失败（常见于 SWA 模型有一些额外的统计 buffer），尝试非严格加载
        print(f"Warning: Strict loading failed ({e}). Retrying with strict=False...")
        keys = model.load_state_dict(new_weights, strict=False)
        return f"{raw_key} (strict=False, missing={len(keys.missing_keys)}, unexpected={len(keys.unexpected_keys)})"

def evaluate_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载模型
    model = build_model(args.backbone).to(device)
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {args.ckpt_path}")
    
    print(f"Loading checkpoint from: {args.ckpt_path}")
    state = torch.load(args.ckpt_path, map_location=device)
    
    load_info = _load_model_weights(model, state)
    model.eval()
    print(f"Model loaded successfully! Mode: {load_info}")

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
    
    image_paths = sorted([
        os.path.join(dp, f) 
        for dp, dn, fn in os.walk(os.path.expanduser(test_stimuli_dir)) 
        for f in fn if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {test_stimuli_dir}")

    for img_path in tqdm(image_paths, desc="Evaluating"):
        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        gt_path = img_path.replace("Stimuli", "FIXATIONMAPS")
        if not os.path.exists(gt_path):
            # print(f"Warning: Ground truth not found for {img_path}, skipping.")
            continue
        gt_map = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype('float32') / 255.0

        # TTA 推理
        images_tta = [image_rgb, cv2.flip(image_rgb, 1)]
        preds_tta = []
        
        with torch.no_grad():
            for img_aug in images_tta:
                img_tensor = transform(image=img_aug)['image'].unsqueeze(0).to(device)
                # Sigmoid 处理
                pred = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
                preds_tta.append(pred)
        
        pred_hflip_restored = cv2.flip(preds_tta[1], 1)
        final_pred_small = np.mean([preds_tta[0], pred_hflip_restored], axis=0)
        
        pred_resized = cv2.resize(final_pred_small, (w, h))
        score = calc_cc_score(gt_map, pred_resized)
        all_scores.append(score)

    # 4. 打印结果
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
                        help="Path to the trained model checkpoint (.pth file).")
    group.add_argument('--exp_name', type=str,
                        help="Experiment name under ./checkpoints")
    parser.add_argument('--list', action='store_true', help='List available experiments')
    
    args = parser.parse_args()

    if not args.backbone:
        args.backbone = cfg.backbone

    if args.list:
        if os.path.exists('checkpoints'):
            for d in sorted(os.listdir('checkpoints')):
                print(d)
        else:
            print('No checkpoints directory found.')
        exit(0)

    if not args.ckpt_path and not args.exp_name:
        if cfg.eval_ckpt_path:
            args.ckpt_path = cfg.eval_ckpt_path
        elif cfg.eval_exp_name:
            candidate = os.path.join('checkpoints', cfg.eval_exp_name, 'best_model.pth')
            if os.path.exists(candidate):
                args.ckpt_path = candidate
            else:
                raise FileNotFoundError(f"No best_model at: {candidate}")
        else:
            raise ValueError('No checkpoint specified. Provide --ckpt_path or --exp_name')

    if args.exp_name:
        # 优先尝试 SWA，其次尝试 best_model
        candidate_swa = os.path.join('checkpoints', args.exp_name, 'best_model_swa.pth')
        candidate_best = os.path.join('checkpoints', args.exp_name, 'best_model.pth')
        
        if os.path.exists(candidate_swa):
            print(f"Found SWA model, using: {candidate_swa}")
            args.ckpt_path = candidate_swa
        elif os.path.exists(candidate_best):
            print(f"Found Best model, using: {candidate_best}")
            args.ckpt_path = candidate_best
        else:
            raise FileNotFoundError(f"No model found for experiment '{args.exp_name}'")

    evaluate_model(args)