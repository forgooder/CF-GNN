#!/bin/bash

# --- 配置区 ---
DATASET="WN18RR_v1"
EXP_NAME="causal_unified_run"
GPU_ID=0

echo "🚀 开始 Causal-GraIL 统一自动化训练 (CIGNN 风格)..."

# 只需要运行一次 train.py
# 内部逻辑：前 50 Epoch 预热 VAE/alpha-beta 解耦，随后线性打开 mask 注入并联合训练 GraIL。
python train.py \
    -d $DATASET \
    -e $EXP_NAME \
    --causal_mode full \
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
echo "📝 提示：第 51 Epoch 后开始 causal mask + causal effect 联合优化。"
