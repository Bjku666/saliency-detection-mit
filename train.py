import torch
from tqdm import tqdm
import os
from torch.optim import swa_utils
import torchvision.utils as vutils

from config import cfg
from src.dataset import get_dataloaders
from src.models import build_model
from src.loss import SaliencyLoss
from src.logger import Logger

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_score = -1.0
        self.counter = 0

    def step(self, score):
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            return False # Continue training
        else:
            self.counter += 1
            return self.counter >= self.patience # Stop training


def compute_loss_components(criterion, pred_logits, target):
    """返回总损失、分项损失字典，以及 Sigmoid 后的概率图"""
    pred_probs = torch.sigmoid(pred_logits)
    weights = getattr(criterion, "weights", {})

    losses = {}
    total = pred_logits.new_tensor(0.0)

    if weights.get("cc", 0) > 0:
        cc_val = (1.0 - criterion.cc_metric(pred_probs, target)).mean()
        losses["cc"] = cc_val
        total = total + weights["cc"] * cc_val

    if weights.get("kld", 0) > 0:
        kld_val = criterion.kld_loss(pred_probs, target).mean()
        losses["kld"] = kld_val
        total = total + weights["kld"] * kld_val

    if weights.get("bce", 0) > 0:
        bce_val = criterion.bce(pred_logits, target)
        losses["bce"] = bce_val
        total = total + weights["bce"] * bce_val

    if weights.get("ssim", 0) > 0:
        ssim_val = 1.0 - criterion.ssim(pred_probs, target)
        losses["ssim"] = ssim_val
        total = total + weights["ssim"] * ssim_val

    losses["total"] = total
    return total, losses, pred_probs


def denormalize(imgs, mean, std):
    """反归一化 Tensor 图像 (N, C, H, W)"""
    device = imgs.device
    mean_t = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, -1, 1, 1)
    return imgs * std_t + mean_t

def main():
    cfg.parse_args()
    cfg.create_dirs()
    
    # 选择实际可用的设备（如果没有 GPU 回退到 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 把解析后的实际 device 存回 cfg，供 loss 等模块使用
    cfg.device = device
    logger = Logger(cfg.log_dir)
    logger.info(f"Experiment Started: {cfg.exp_name}")
    logger.info(f"Config: Backbone={cfg.backbone}, BS={cfg.batch_size}, LR={cfg.lr}, Fold={cfg.fold}/{cfg.k_folds-1}")
    
    train_loader, val_loader = get_dataloaders(cfg)
    model = build_model(cfg.backbone).to(device)

    # 多卡检测并启用 DataParallel（若可用）
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        logger.info(f"Detected {num_gpus} GPUs — using DataParallel")
    else:
        logger.info(f"Using device: {device}")

    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = SaliencyLoss(cfg)

    # SWA: start averaging in the final 25% epochs
    swa_start_epoch = int(cfg.epochs * 0.75)
    swa_model = swa_utils.AveragedModel(base_model).to(device)
    swa_scheduler = None
    swa_updates = 0
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (epoch + 1) / cfg.warmup_epochs if epoch < cfg.warmup_epochs else 1)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=1e-6)
    early_stopper = EarlyStopping(patience=cfg.early_stop_patience)

    # 保存一份配置到 ckpt_dir（便于追溯）
    try:
        import json
        cfg_path = os.path.join(cfg.ckpt_dir, 'config.json')
        with open(cfg_path, 'w') as f:
            # 只保存基础可序列化字段
            json.dump({
                'backbone': cfg.backbone,
                'batch_size': cfg.batch_size,
                'lr': cfg.lr,
                'weight_decay': cfg.weight_decay,
                'warmup_epochs': cfg.warmup_epochs,
                'epochs': cfg.epochs,
                'img_size': cfg.img_size,
            }, f, indent=2)
        logger.info(f"Saved config to: {cfg_path}")
    except Exception:
        logger.info("Warning: failed to write config.json")

    best_cc_score = -1.0

    try:
        for epoch in range(cfg.epochs):
            model.train()
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
            train_loss_accum = {}
            for imgs, masks in loop:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                preds = model(imgs)
                loss, loss_dict, _ = compute_loss_components(criterion, preds, masks)
                loss.backward()
                optimizer.step()
                # 累计分项损失
                for k, v in loss_dict.items():
                    train_loss_accum[k] = train_loss_accum.get(k, 0.0) + v.item()
                loop.set_postfix(loss=loss.item())

            # 记录训练阶段标量（按 epoch 平均）
            num_train_batches = len(train_loader)
            train_log = {f"loss/{k}": v / num_train_batches for k, v in train_loss_accum.items()}
            train_log["lr"] = optimizer.param_groups[0]['lr']
            logger.log_metrics(train_log, epoch, 'train')

            # SWA 开启与权重累计（最后 25% 的 epoch）
            if epoch >= swa_start_epoch:
                if swa_scheduler is None:
                    swa_scheduler = swa_utils.SWALR(optimizer, swa_lr=optimizer.param_groups[0]['lr'])
                    logger.info(f"SWA activated at epoch {epoch+1}")
                swa_model.update_parameters(base_model)
                swa_updates += 1

            # step schedulers: warmup first then cosine
            try:
                if swa_scheduler is not None:
                    swa_scheduler.step()
                elif epoch < cfg.warmup_epochs:
                    warmup_scheduler.step()
                else:
                    cosine_scheduler.step()
            except Exception:
                # ignore scheduler issues but log
                logger.info('Scheduler step failed or not configured')

            # Validation
            model.eval()
            val_cc_sum = 0
            val_loss_sum = 0
            val_loss_accum = {}
            viz_batch = None
            
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    preds_logits = model(imgs)
                    loss, loss_dict, preds_probs = compute_loss_components(criterion, preds_logits, masks)
                    val_loss_sum += loss.item()
                    for k, v in loss_dict.items():
                        val_loss_accum[k] = val_loss_accum.get(k, 0.0) + v.item()

                    batch_cc_list = criterion.cc_metric(preds_probs, masks)
                    val_cc_sum += batch_cc_list.sum().item()

                    # 保存第一批用于可视化
                    if viz_batch is None:
                        viz_batch = {
                            "imgs": imgs.detach().clone(),
                            "masks": masks.detach().clone(),
                            "preds": preds_probs.detach().clone(),
                            "logits": preds_logits.detach().clone(),
                        }

            avg_val_loss = val_loss_sum / len(val_loader)
            avg_val_cc = val_cc_sum / len(val_loader.dataset)
            
            logger.info(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Val CC: {avg_val_cc:.4f}")
            num_val_batches = len(val_loader)
            val_log = {f"loss/{k}": v / num_val_batches for k, v in val_loss_accum.items()}
            val_log.update({"val_loss": avg_val_loss, "val_cc": avg_val_cc, "lr": optimizer.param_groups[0]['lr']})
            logger.log_metrics(val_log, epoch, 'val')

            # 记录预测直方图
            if viz_batch is not None:
                logger.log_histogram("preds/val_logits", viz_batch["logits"].cpu(), epoch)

                # 可视化：原图 | GT | Pred
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                imgs_viz = denormalize(viz_batch["imgs"], mean, std)
                masks_viz = viz_batch["masks"]
                preds_viz = viz_batch["preds"]

                num_show = min(4, imgs_viz.size(0))
                triplets = []
                for i in range(num_show):
                    img = imgs_viz[i].clamp(0, 1)
                    gt_rgb = masks_viz[i].repeat(3, 1, 1).clamp(0, 1)
                    pred_rgb = preds_viz[i].repeat(3, 1, 1).clamp(0, 1)
                    triplets.append(torch.cat([img, gt_rgb, pred_rgb], dim=2))

                grid = vutils.make_grid(torch.stack(triplets), nrow=1)
                logger.log_image("val/visualization", grid.cpu(), epoch)

            # --- Checkpoint 保存逻辑 ---
            # 每个 epoch 都保存一个包含优化器状态的 checkpoint（覆盖或按 epoch 命名）
            ckpt_path = os.path.join(cfg.ckpt_dir, f"checkpoint_epoch{epoch+1}.pth")
            state = {
                'epoch': epoch + 1,
                'model_state': base_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'lr': optimizer.param_groups[0]['lr'],
                'val_cc': avg_val_cc,
                'val_loss': avg_val_loss,
            }
            torch.save(state, ckpt_path)
            logger.info(f"Saved epoch checkpoint: {ckpt_path}")

            # 如果当前 val CC 是最好，保存一个单独的 best 权重文件
            if avg_val_cc > best_cc_score:
                best_cc_score = avg_val_cc
                best_path = os.path.join(cfg.ckpt_dir, "best_model.pth")
                torch.save(state, best_path)
                logger.info(f"New best model (CC={best_cc_score:.4f}) saved to: {best_path}")

            # Early stopping 检查（基于 avg_val_cc）
            if early_stopper.step(avg_val_cc):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        # 训练结束后处理 SWA（若已启用）
        if swa_updates > 0:
            logger.info("Updating BatchNorm statistics for SWA model")
            swa_utils.update_bn(train_loader, swa_model, device=device)

            # 评估 SWA 模型表现
            swa_model.eval()
            swa_val_cc_sum, swa_val_loss_sum = 0.0, 0.0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    preds_logits = swa_model(imgs)
                    loss = criterion(preds_logits, masks)
                    swa_val_loss_sum += loss.item()
                    preds_probs = torch.sigmoid(preds_logits)
                    batch_cc_list = criterion.cc_metric(preds_probs, masks)
                    swa_val_cc_sum += batch_cc_list.sum().item()

            swa_avg_val_loss = swa_val_loss_sum / len(val_loader)
            swa_avg_val_cc = swa_val_cc_sum / len(val_loader.dataset)
            logger.info(f"SWA Eval | Val Loss: {swa_avg_val_loss:.4f} | Val CC: {swa_avg_val_cc:.4f}")

            swa_state = {
                'epoch': epoch + 1,
                'model_state': swa_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_cc': swa_avg_val_cc,
                'val_loss': swa_avg_val_loss,
            }
            swa_path = os.path.join(cfg.ckpt_dir, "best_model_swa.pth")
            torch.save(swa_state, swa_path)
            logger.info(f"SWA model saved to: {swa_path}")
        else:
            logger.info("SWA not activated; skipped SWA checkpoint")

    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt received — saving interrupt checkpoint')
        # 保存中断时的 checkpoint
        try:
            interrupt_path = os.path.join(cfg.ckpt_dir, f'interrupt_epoch{epoch+1}.pth')
            state = {
                'epoch': epoch + 1,
                'model_state': base_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_cc': best_cc_score,
            }
            torch.save(state, interrupt_path)
            logger.info(f"Saved interrupt checkpoint to: {interrupt_path}")
        except Exception:
            logger.info('Failed to save interrupt checkpoint')
        raise
    finally:
        try:
            logger.info('Training finished — closing logger')
            logger.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()