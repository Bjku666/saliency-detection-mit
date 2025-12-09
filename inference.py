# inference.py (放在项目根目录)

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

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = build_model(args.backbone).to(device)
    state = torch.load(args.ckpt_path, map_location=device)
    # 支持两种 checkpoint 格式
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)
    model.eval()
    print("Model loaded successfully!")
    
    transform = A.Compose([
        A.Resize(cfg.img_size, cfg.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    results = []
    test_stimuli_dir = os.path.join(cfg.test_root, "Stimuli")
    
    print("Starting Inference with TTA...")
    for cat in tqdm(sorted(os.listdir(test_stimuli_dir))):
        cat_dir = os.path.join(test_stimuli_dir, cat)
        if not os.path.isdir(cat_dir): continue
        
        for img_name in sorted(os.listdir(cat_dir)):
            img_path = os.path.join(cat_dir, img_name)
            image = cv2.imread(img_path)
            h, w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # TTA: original and horizontal flip
            images_tta = [image_rgb, cv2.flip(image_rgb, 1)]
            preds_tta = []
            
            with torch.no_grad():
                for img_aug in images_tta:
                    img_tensor = transform(image=img_aug)['image'].unsqueeze(0).to(device)
                    # --- NEW: 确保应用 sigmoid ---
                    pred = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
                    preds_tta.append(pred)
            
            # Un-flip the flipped prediction and average
            pred_hflip = cv2.flip(preds_tta[1], 1)
            final_pred = np.mean([preds_tta[0], pred_hflip], axis=0)
            
            pred_resized = cv2.resize(final_pred, (w, h))
            
            gt_path = img_path.replace("Stimuli", "FIXATIONMAPS")
            gt_map = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype('float32') / 255.0
            
            score = calc_cc_score(gt_map, pred_resized)
            results.append({"id": f"{cat}/{img_name}", "score": score})

    df = pd.DataFrame(results)
    save_path = f"./submissions/submission_{args.backbone}_tta_team.csv"
    os.makedirs("./submissions", exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"\nInference Done! Submission saved to: {save_path}")
    print(f"Mean CC Score with TTA: {df['score'].mean():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, required=False, choices=['mit_b5'],
                        help='Backbone name. If omitted, use value from config.py')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--ckpt_path', type=str, help='Path to your best_model.pth')
    group.add_argument('--exp_name', type=str, help='Experiment name under ./checkpoints to use its best model')
    parser.add_argument('--list', action='store_true', help='List available experiments under ./checkpoints and exit')

    args = parser.parse_args()

    # fallback backbone
    if not args.backbone:
        args.backbone = cfg.backbone

    # list experiments if requested
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

    if args.exp_name:
        candidate = os.path.join('checkpoints', args.exp_name, 'best_model.pth')
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Best model not found for experiment '{args.exp_name}' at: {candidate}")
        args.ckpt_path = candidate

    print('Using checkpoint:', args.ckpt_path)
    main(args)