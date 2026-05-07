#!/bin/bash

# --- 配置区 ---
DATASET="WN18RR_v1"
EXP_NAME="causal_unified_run"
GPU_ID=0

echo "🚀 开始 Causal-GraIL 统一自动化训练 (CIGNN 风格)..."

# 只需要运行一次 train.py
# 内部逻辑：前 50 Epoch 自动预热 VAE，后 50 Epoch 自动开启全量联合训练
python train.py \
    -d $DATASET \
    -e $EXP_NAME \
    --gpu $GPU_ID \
    --num_epochs 100 \
    --batch_size 16 \
    --lr 0.001 \
    --eval_every 5 \
    --save_every 10

# 检查是否成功
if [ $? -ne 0 ]; then
    echo "❌ 训练过程中止，请检查 log_train.txt。"
    exit 1
fi

echo "------------------------------------------------"
echo "✅ 自动化训练已完成！"
echo "📂 结果保存在: experiments/${EXP_NAME}"
echo "📝 提示：前 50 Epoch 为 VAE 磨刀期，第 51 Epoch 开始正式砍柴。"