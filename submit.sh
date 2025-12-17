#!/bin/bash

# ================= 配置区 =================
# 每次训练开始时，train.sh 会打印出 "实验名(BASE_EXP_NAME)"
# 请把它复制到下面这里：

# 示例 (Stage 1): BASE_EXP_NAME="transalnet_1217_1200_kfold"
# 示例 (Stage 2): BASE_EXP_NAME="transalnet_1217_1800_kfold_PL" <-- 最终提交用这个
BASE_EXP_NAME="在这里填入你最新跑完的实验名"

# =========================================

echo "========================================"
echo "准备对实验 [ $BASE_EXP_NAME ] 进行推理"
echo "========================================"
echo "请选择模式:"
echo "  1) 单折测试 (快速验证)"
echo "  2) 五折融合 (推荐)"
echo "========================================"
read -p "请输入数字 [1 或 2]: " MODE

# 默认使用 SWA 模型，效果更稳
MODEL_FILENAME="best_model_swa.pth"

if [ "$MODE" == "1" ]; then
    read -p "请输入要测试的 Fold [0-4]: " ONE_FOLD
    python inference.py \
        --exp_name $BASE_EXP_NAME \
        --folds "$ONE_FOLD" \
        --model_file "$MODEL_FILENAME"

elif [ "$MODE" == "2" ]; then
    echo "正在进行 5-Fold 融合推理..."
    python inference.py \
        --exp_name $BASE_EXP_NAME \
        --folds "0,1,2,3,4" \
        --model_file "$MODEL_FILENAME"
else
    echo "输入无效"
    exit 1
fi

echo "推理完成！CSV 文件已保存到 submissions/ 文件夹。"