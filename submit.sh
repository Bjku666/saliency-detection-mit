#!/bin/bash

# ================= 配置区 =================
# 这里填你 train.sh 里生成/打印出来的 BASE_EXP_NAME（不包含 fold 后缀）
# 新的 checkpoint 目录结构为：
#   checkpoints/<BASE_EXP_NAME>/fold0/best_model.pth
#   checkpoints/<BASE_EXP_NAME>/fold1/best_model.pth
#   ...
# 例如：BASE_EXP_NAME="mit_b5_1216_2243" 或 "mit_b5_1216_2243_kfold"
BASE_EXP_NAME="mit_b5_1217_1030_kfold"
# =========================================

echo "========================================"
echo "准备对实验 [ $BASE_EXP_NAME ] 进行推理"
echo "========================================"
echo "请选择模式:"
echo "  1) 单折测试 (可选择 Fold 0-4)"
echo "  2) 五折融合 (融合 Fold 0-4，冲击排行榜)"
echo "========================================"
read -p "请输入数字 [1 或 2]: " MODE

echo "========================================"
echo "请选择要使用的模型权重文件:"
echo "  1) 普通最佳模型 (best_model.pth) - [默认]"
echo "  2) SWA 权重平均模型 (best_model_swa.pth) - [推荐]"
echo "========================================"
read -p "请输入数字 [1 或 2]: " WEIGHT_OPT

if [ "$WEIGHT_OPT" == "2" ]; then
    MODEL_FILENAME="best_model_swa.pth"
    echo "已选择: SWA 模型 ($MODEL_FILENAME)"
else
    MODEL_FILENAME="best_model.pth"
    echo "已选择: 普通最佳模型 ($MODEL_FILENAME)"
fi

if [ "$MODE" == "1" ]; then
    read -p "请输入要测试的 Fold [0-4，默认0]: " ONE_FOLD
    if [ -z "$ONE_FOLD" ]; then
        ONE_FOLD=0
    fi

    if ! [[ "$ONE_FOLD" =~ ^[0-4]$ ]]; then
        echo "Fold 输入无效: $ONE_FOLD （必须是 0-4）"
        exit 1
    fi

    echo "正在进行 Fold $ONE_FOLD 单折推理..."
    # 这里的 --exp_name 和 --folds 是传给 Python 脚本的参数
    # 假设你已经按上一条建议修改了 inference.py
    python inference.py \
        --exp_name $BASE_EXP_NAME \
        --folds "$ONE_FOLD" \
        --model_file "$MODEL_FILENAME"

elif [ "$MODE" == "2" ]; then
    echo "正在融合 (Fold 0-4 融合)..."
    python inference.py \
        --exp_name $BASE_EXP_NAME \
        --folds "0,1,2,3,4" \
        --model_file "$MODEL_FILENAME"

else
    echo "输入无效，退出。"
    exit 1
fi

echo "推理完成！请查看 submissions/ 文件夹下的 CSV 文件。"