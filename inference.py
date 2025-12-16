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

    new_weights = {}
    for k, v in weights.items():
        name = k[7:] if k.startswith('module.') else k
        new_weights[name] = v

    try:
        model.load_state_dict(new_weights, strict=True)
        return f"{raw_key} (strict=True)"
    except RuntimeError:
        print("Warning: Strict loading failed. Retrying with strict=False...")
        model.load_state_dict(new_weights, strict=False)
        return f"{raw_key} (strict=False)"


def parse_folds(folds_str, k_folds):
    fold_set = set()
    for item in folds_str.split(','):
        item = item.strip()
        if not item:
            continue
        fold_idx = int(item)
        if fold_idx < 0 or fold_idx >= k_folds:
            raise ValueError(f"Fold index {fold_idx} is out of range [0, {k_folds-1}]")
        fold_set.add(fold_idx)
    if not fold_set:
        raise ValueError("No valid folds provided")
    return sorted(fold_set)


def build_transform():
    return A.Compose([
        A.Resize(cfg.img_size, cfg.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    folds = parse_folds(args.folds, cfg.k_folds)
    exp_name = args.exp_name or cfg.eval_exp_name or cfg.base_exp_name or cfg.exp_name
    if not exp_name:
        raise ValueError("exp_name is required; set --exp_name or config.eval_exp_name")

    transform = build_transform()
    models = []
    for fold_idx in folds:
        # 新结构：checkpoints/<exp_name>/foldX/best_model.pth
        ckpt_path = os.path.join(cfg.checkpoints_root, exp_name, f"fold{fold_idx}", "best_model.pth")
        # 兼容旧结构：checkpoints/<exp_name>_foldX/best_model.pth
        if not os.path.exists(ckpt_path):
            legacy = os.path.join(cfg.checkpoints_root, f"{exp_name}_fold{fold_idx}", "best_model.pth")
            if os.path.exists(legacy):
                ckpt_path = legacy
            else:
                raise FileNotFoundError(f"Checkpoint not found for fold {fold_idx}: {ckpt_path} (and legacy: {legacy})")
        model = build_model(args.backbone).to(device)
        state = torch.load(ckpt_path, map_location=device)
        used_key = _load_model_weights(model, state)
        model.eval()
        print(f"Loaded fold {fold_idx} from {ckpt_path} using {used_key}")
        models.append(model)

    test_stimuli_dir = os.path.join(cfg.test_root, "Stimuli")
    image_paths = sorted([
        os.path.join(dp, f)
        for dp, _, fn in os.walk(os.path.expanduser(test_stimuli_dir))
        for f in fn if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found under {test_stimuli_dir}")

    results = []
    print(f"Starting inference on {len(image_paths)} images with folds {folds}...")

    for img_path in tqdm(image_paths):
        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        images_tta = [image_rgb, cv2.flip(image_rgb, 1)]
        ensemble_preds = []

        with torch.no_grad():
            for model in models:
                preds_tta = []
                for img_aug in images_tta:
                    img_tensor = transform(image=img_aug)['image'].unsqueeze(0).to(device)
                    pred = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
                    preds_tta.append(pred)

                pred_hflip = cv2.flip(preds_tta[1], 1)
                model_pred = np.mean([preds_tta[0], pred_hflip], axis=0)
                ensemble_preds.append(model_pred)

        final_pred = np.mean(ensemble_preds, axis=0)
        pred_resized = cv2.resize(final_pred, (w, h))

        gt_path = img_path.replace("Stimuli", "FIXATIONMAPS")
        score = 0.0
        if os.path.exists(gt_path):
            gt_map = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype('float32') / 255.0
            score = calc_cc_score(gt_map, pred_resized)

        rel_path = os.path.relpath(img_path, test_stimuli_dir)
        file_id = rel_path.replace(os.sep, '/')
        results.append({"id": file_id, "score": score})

    df = pd.DataFrame(results)
    os.makedirs(cfg.submission_dir, exist_ok=True)
    save_name = cfg.build_submission_name(args.save_name)
    save_path = os.path.join(cfg.submission_dir, save_name)
    df.to_csv(save_path, index=False)

    print(f"\nInference Done! Submission saved to: {save_path}")
    if df['score'].sum() > 0:
        print(f"Mean CC Score with TTA: {df['score'].mean():.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Saliency inference with multi-fold ensemble")
    parser.add_argument('--backbone', type=str, default=cfg.backbone, choices=['mit_b5'])
    parser.add_argument('--exp_name', type=str, default=None, help='Base experiment name without fold suffix')
    parser.add_argument('--folds', type=str, default=None, help='Comma-separated fold indices, e.g., "0" or "0,1,2,3,4"')
    parser.add_argument('--save_name', type=str, default=None, help='Optional output csv name')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.folds is None:
        args.folds = ",".join(str(i) for i in range(cfg.k_folds))
    run_inference(args)