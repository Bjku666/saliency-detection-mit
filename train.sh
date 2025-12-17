#!/bin/bash

# ================= 配置区 =================
# 1. 选择阶段: 1 = 初始训练 (Baseline), 2 = 伪标签微调 (Rank 1冲刺)
STAGE=1

# 2. 基础参数
BACKBONE="transalnet"  # 你的核武器
BATCH_SIZE=4           # TranSalNet 比较大，4090 建议设 4，如果显存够可试 8
GPU_ID="${GPU_ID:-0}"  # 默认使用 GPU 0

# ================= 自动逻辑 =================

# 根据阶段设置参数
if [ "$STAGE" == "1" ]; then
    echo ">>> [Stage 1] 启动 Baseline 训练 (无伪标签)..."
    EXP_NOTE="kfold"
    USE_PSEUDO=""  # 为空表示不开启
elif [ "$STAGE" == "2" ]; then
    echo ">>> [Stage 2] 启动 Pseudo Labeling 微调 (混合训练)..."
    EXP_NOTE="kfold_PL"  # 加个后缀，区分实验结果
    USE_PSEUDO="--use_pseudo" # 开启 config.py 里的开关
else
    echo "错误: STAGE 必须是 1 或 2"
    exit 1
fi

# 生成本次实验名 (不含 fold 后缀)
# 例如: transalnet_1217_1509_kfold (Stage 1)
# 例如: transalnet_1217_1820_kfold_PL (Stage 2)
BASE_EXP_NAME="${BACKBONE}_$(date +%m%d_%H%M)_${EXP_NOTE}"

echo "----------------------------------------"
echo "配置: Model=$BACKBONE, BS=$BATCH_SIZE"
echo "阶段: Stage $STAGE"
echo "实验名(BASE_EXP_NAME): $BASE_EXP_NAME"
echo "使用显卡: GPU $GPU_ID"
echo "----------------------------------------"

# --- 5-Fold 顺序训练循环 ---
(
    for FOLD in 0 1 2 3 4; do
        echo "[GPU ${GPU_ID}] 开始训练 Fold ${FOLD}..."
        
        # 核心命令：增加了 $USE_PSEUDO 参数
        python train.py \
            --backbone $BACKBONE \
            --fold ${FOLD} \
            --batch_size $BATCH_SIZE \
            --note $EXP_NOTE \
            --exp_name $BASE_EXP_NAME \
            --gpu_id ${GPU_ID} \
            $USE_PSEUDO
            
        echo "[GPU ${GPU_ID}] Fold ${FOLD} 完成!"
    done
    echo "[GPU ${GPU_ID}] 所有任务完成！实验名: $BASE_EXP_NAME"
) > "logs_gpu${GPU_ID}.out" 2>&1 &

echo "任务已后台启动！"
echo "查看日志: tail -f logs_gpu${GPU_ID}.out"
echo " 重要: 训练完成后，请复制上面的 BASE_EXP_NAME 到 submit.sh 用于推理。"