#!/bin/bash

# --- 配置区 ---
BACKBONE="mit_b5"
BATCH_SIZE=8        # 单卡显存允许的最大BS (4090上mit_b5通常跑8没问题，不够就改4)
EXP_NOTE="kfold"   # 可选备注；留空则不追加

# 默认使用 GPU 1（你的 4090 空闲卡）。
# 如需临时改用其它卡：GPU_ID=0 bash train.sh
GPU_ID="${GPU_ID:-1}"

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

echo "使用单卡训练: GPU_ID=$GPU_ID"

# --- 单 GPU 顺序任务队列 (跑 Fold 0-4) ---
(
    for FOLD in 0 1 2 3 4; do
        echo "[GPU ${GPU_ID}] 开始训练 Fold ${FOLD}..."
        python train.py --backbone $BACKBONE --fold ${FOLD} --batch_size $BATCH_SIZE --note $EXP_NOTE --exp_name $BASE_EXP_NAME --gpu_id ${GPU_ID}
        echo "[GPU ${GPU_ID}] Fold ${FOLD} 完成!"
    done
    echo "[GPU ${GPU_ID}] 所有任务完成！"
) > "logs_gpu${GPU_ID}.out" 2>&1 &  # 后台运行，日志写入 logs_gpuX.out

echo "单卡任务已启动！"
echo "请使用 'tail -f logs_gpu${GPU_ID}.out' 查看训练进度"