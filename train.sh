#!/bin/bash

# --- 配置区 ---
BACKBONE="mit_b5"
BATCH_SIZE=8        # 单卡显存允许的最大BS (4090上mit_b5通常跑8没问题，不够就改4)
EXP_NOTE="kfold"   # 可选备注；留空则不追加

# 关键：一次 5-fold 训练必须共享同一个 BASE_EXP_NAME（不包含 fold 后缀）
# 例如：mit_b5_1216_2243 或 mit_b5_1216_2243_kfold
BASE_EXP_NAME="${BACKBONE}_$(date +%m%d_%H%M)"
if [ -n "$EXP_NOTE" ]; then
    BASE_EXP_NAME="${BASE_EXP_NAME}_${EXP_NOTE}"
fi
# ----------------

echo "开始 5-Fold 交叉验证训练..."
echo "配置: Model=$BACKBONE, BS=$BATCH_SIZE"
echo "本次实验名(BASE_EXP_NAME)=$BASE_EXP_NAME"

# --- GPU 0 任务队列 (跑 Fold 0, 1, 2) ---
(
    echo "[GPU 0] 开始训练 Fold 0..."
    python train.py --backbone $BACKBONE --fold 0 --batch_size $BATCH_SIZE --note $EXP_NOTE --exp_name $BASE_EXP_NAME --gpu_id 0
    
    echo "[GPU 0] Fold 0 完成! 开始训练 Fold 1..."
    python train.py --backbone $BACKBONE --fold 1 --batch_size $BATCH_SIZE --note $EXP_NOTE --exp_name $BASE_EXP_NAME --gpu_id 0
    
    echo "[GPU 0] Fold 1 完成! 开始训练 Fold 2..."
    python train.py --backbone $BACKBONE --fold 2 --batch_size $BATCH_SIZE --note $EXP_NOTE --exp_name $BASE_EXP_NAME --gpu_id 0
    
    echo "[GPU 0] 所有任务完成！"
) > logs_gpu0.out 2>&1 &  # 后台运行，日志写入 logs_gpu0.out

# --- GPU 1 任务队列 (跑 Fold 3, 4) ---
(
    echo "[GPU 1] 开始训练 Fold 3..."
    python train.py --backbone $BACKBONE --fold 3 --batch_size $BATCH_SIZE --note $EXP_NOTE --exp_name $BASE_EXP_NAME --gpu_id 1
    
    echo "[GPU 1] Fold 3 完成! 开始训练 Fold 4..."
    python train.py --backbone $BACKBONE --fold 4 --batch_size $BATCH_SIZE --note $EXP_NOTE --exp_name $BASE_EXP_NAME --gpu_id 1
    
    echo "[GPU 1] 所有任务完成！"
) > logs_gpu1.out 2>&1 &  # 后台运行，日志写入 logs_gpu1.out

echo "双卡并行任务已启动！"
echo "请使用 'tail -f logs_gpu0.out' 查看 GPU 0 进度"
echo "请使用 'tail -f logs_gpu1.out' 查看 GPU 1 进度"