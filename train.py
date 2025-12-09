import torch
from tqdm import tqdm
import os

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

def main():
    cfg.parse_args()
    cfg.create_dirs()
    
    # 选择实际可用的设备（如果没有 GPU 回退到 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 把解析后的实际 device 存回 cfg，供 loss 等模块使用
    cfg.device = device
    logger = Logger(cfg.log_dir)
    logger.info(f"Experiment Started: {cfg.exp_name}")
    logger.info(f"Config: Backbone={cfg.backbone}, BS={cfg.batch_size}, LR={cfg.lr}")
    
    train_loader, val_loader = get_dataloaders(cfg)
    model = build_model(cfg.backbone).to(device)

    # 多卡检测并启用 DataParallel（若可用）
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        logger.info(f"Detected {num_gpus} GPUs — using DataParallel")
    else:
        logger.info(f"Using device: {device}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = SaliencyLoss(cfg)
    
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
            for imgs, masks in loop:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                preds = model(imgs)
                loss = criterion(preds, masks)
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())

            # step schedulers: warmup first then cosine
            try:
                if epoch < cfg.warmup_epochs:
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
            
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    
                    # 1. 模型输出 Logits
                    preds_logits = model(imgs)
                    
                    # 2. 计算 Loss (Criterion 内部会自动处理 Logits -> Sigmoid)
                    loss = criterion(preds_logits, masks)
                    val_loss_sum += loss.item()

                    # 3. ⚠️ 关键修改：手动 Sigmoid 转成 [0,1] 概率图
                    preds_probs = torch.sigmoid(preds_logits)

                    # 4. 计算 CC 指标 (使用 0-1 的图)
                    # 复用 criterion.cc_metric, 它返回的是一个 batch 的 CC 列表
                    # 我们只需要把它们加起来
                    batch_cc_list = criterion.cc_metric(preds_probs, masks)
                    val_cc_sum += batch_cc_list.sum().item()

            avg_val_loss = val_loss_sum / len(val_loader)
            avg_val_cc = val_cc_sum / len(val_loader.dataset)
            
            logger.info(f"Epoch {epoch+1} | Val Loss: {avg_val_loss:.4f} | Val CC: {avg_val_cc:.4f}")
            logger.log_metrics({'val_loss': avg_val_loss, 'val_cc': avg_val_cc}, epoch, 'epoch')

            # --- Checkpoint 保存逻辑 ---
            # 每个 epoch 都保存一个包含优化器状态的 checkpoint（覆盖或按 epoch 命名）
            ckpt_path = os.path.join(cfg.ckpt_dir, f"checkpoint_epoch{epoch+1}.pth")
            state = {
                'epoch': epoch + 1,
                'model_state': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
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
    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt received — saving interrupt checkpoint')
        # 保存中断时的 checkpoint
        try:
            interrupt_path = os.path.join(cfg.ckpt_dir, f'interrupt_epoch{epoch+1}.pth')
            state = {
                'epoch': epoch + 1,
                'model_state': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
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